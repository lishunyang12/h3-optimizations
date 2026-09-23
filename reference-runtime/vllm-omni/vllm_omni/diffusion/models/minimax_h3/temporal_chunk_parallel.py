# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact temporal-window parallel decode for the MiniMax H3 video VAE.

The released H3 decoder evaluates independent seven-token temporal windows
and joins their decoded frames with a deterministic five-frame blend.  This
module assigns complete windows to ranks, gathers only the 22 live frames of
each window to rank zero, and replays the serial join there.  Spatial tiling
must be rank-local while this driver runs; the adapter owns that state change.

This is an eager, inference-only path.  It does not opt the VAE into
``torch.compile``: process-local autotuning can choose different kernels on
different ranks and therefore falls outside the bit-exact candidate contract.

The released checkpoint also has ``stack_tiling=False``.  Rank-local spatial
tiles therefore remain sequential: this schedule redistributes the same tile
work and reduces collective frequency, but does not reduce decoder compute.
Its latency benefit is deliberately treated as benchmark-dependent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

DecodedChunkCallback = Callable[[torch.Tensor, int, int], None]


@dataclass(frozen=True)
class MiniMaxH3TemporalChunkParallelStats:
    """Deterministic collective accounting for one parallel decode."""

    rounds: int
    windows: int
    real_segments: int
    placeholder_segments: int
    gather_calls: int
    segment_bytes: int
    input_bytes: int
    output_bytes: int
    output_frames: int


def _num_rounds(num_windows: int, world_size: int) -> int:
    return -(-num_windows // world_size)


def _gather_stack_to_rank_zero(
    segment: torch.Tensor,
    group: Any,
) -> torch.Tensor | None:
    """Destination-only gather without a second full-size ``torch.cat``.

    ``dist.gather`` writes directly into contiguous views of one leader-owned
    ``[world_size, *segment.shape]`` allocation.  Non-leaders allocate no
    receive tensor.
    """

    segment = segment.contiguous()
    output = segment.new_empty((group.world_size, *segment.shape)) if group.rank_in_group == 0 else None
    gather_list = list(output.unbind(0)) if output is not None else None
    dist.gather(
        segment,
        gather_list=gather_list,
        dst=group.ranks[0],
        group=group.device_group,
    )
    return output


def _decode_live_segment(
    model: Any,
    latent: torch.Tensor,
    window_index: int,
    *,
    tokens_per_window: int,
    token_overlap: int,
    decoded_frames_per_window: int,
    keep_ranges: tuple[tuple[int, int], ...],
    temporal_cat_dtype: torch.dtype | None,
) -> torch.Tensor:
    token_start = window_index * tokens_per_window
    token_end = token_start + tokens_per_window + token_overlap
    clip_latent = latent[:, :, token_start:token_end, :, :]
    expected_tokens = tokens_per_window + token_overlap
    if int(clip_latent.shape[2]) != expected_tokens:
        raise RuntimeError(
            "MiniMax H3 temporal chunk parallel found a short latent window: "
            f"window={window_index}, shape={tuple(clip_latent.shape)}, "
            f"expected_tokens={expected_tokens}"
        )
    decoded = model._adaptive_decode(clip_latent)
    if temporal_cat_dtype is not None and decoded.dtype != temporal_cat_dtype:
        decoded = decoded.to(temporal_cat_dtype)
    if decoded.device != latent.device:
        decoded = decoded.to(latent.device)
    if decoded.ndim != 5 or int(decoded.shape[2]) != decoded_frames_per_window:
        raise RuntimeError(
            "MiniMax H3 temporal chunk decoder violated the canonical frame contract: "
            f"window={window_index}, shape={tuple(decoded.shape)}, "
            f"expected_T={decoded_frames_per_window}"
        )
    pieces = [decoded[:, :, start:end, :, :] for start, end in keep_ranges]
    return torch.cat(pieces, dim=2).contiguous()


class _LeaderAssembler:
    """Serially join gathered segments while later windows decode."""

    def __init__(
        self,
        model: Any,
        callback: DecodedChunkCallback,
        *,
        main_frames: int,
        overlap_frames: int,
        output_frames: int,
        device: torch.device,
    ) -> None:
        self._model = model
        self._callback = callback
        self._main_frames = main_frames
        self._overlap_frames = overlap_frames
        self._output_frames = output_frames
        self._previous_overlap: torch.Tensor | None = None
        self._write_pos = 0
        self._error: BaseException | None = None
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._round_done: torch.cuda.Event | None = None

    def _remember_error(self, error: BaseException) -> None:
        if self._error is None:
            error.__traceback__ = None
            self._error = error

    def push(self, segment: torch.Tensor) -> None:
        if self._error is not None:
            return
        try:
            if self._stream is None:
                self._push(segment)
                return
            self._stream.wait_stream(torch.cuda.current_stream(segment.device))
            segment.record_stream(self._stream)
            with torch.cuda.stream(self._stream):
                self._push(segment)
        except BaseException as error:  # Keep every rank in later gathers.
            self._remember_error(error)

    def _push(self, segment: torch.Tensor) -> None:
        main = segment[:, :, : self._main_frames]
        overlap = segment[:, :, self._main_frames :]
        if self._previous_overlap is not None:
            main = self._model.blend(
                self._previous_overlap,
                main,
                self._overlap_frames,
                dim=-3,
            )
        frame_count = min(int(main.shape[2]), self._output_frames - self._write_pos)
        if frame_count > 0:
            self._callback(main[:, :, :frame_count], self._write_pos, self._output_frames)
            self._write_pos += frame_count
        # A narrow view can otherwise retain the complete leader receive
        # allocation.  Clone the five-frame tail before the next round so the
        # ~GiB gather backing storage has a bounded lifetime.
        self._previous_overlap = overlap.clone()

    def finish_round(self) -> None:
        if self._stream is None or self._error is not None:
            return
        try:
            self._round_done = torch.cuda.Event()
            self._round_done.record(self._stream)
        except BaseException as error:
            self._remember_error(error)

    def drain_previous_round(self) -> None:
        if self._round_done is not None:
            try:
                self._round_done.synchronize()
            except BaseException as error:
                self._remember_error(error)
            finally:
                self._round_done = None

    def finalize(self) -> None:
        if self._error is not None:
            return
        try:
            if self._previous_overlap is None:
                raise RuntimeError("MiniMax H3 temporal chunk parallel produced no final overlap")
            remaining = self._output_frames - self._write_pos
            tail = self._previous_overlap[:, :, :remaining]
            if self._stream is None:
                self._emit_tail(tail)
            else:
                with torch.cuda.stream(self._stream):
                    self._emit_tail(tail)
        except BaseException as error:
            self._remember_error(error)

    def _emit_tail(self, tail: torch.Tensor) -> None:
        if int(tail.shape[2]) > 0:
            self._callback(tail, self._write_pos, self._output_frames)
            self._write_pos += int(tail.shape[2])
        if self._write_pos != self._output_frames:
            raise RuntimeError(
                "MiniMax H3 temporal chunk parallel output length mismatch: "
                f"frames={self._write_pos}/{self._output_frames}"
            )

    def synchronize(self) -> None:
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except BaseException as error:
                self._remember_error(error)

    @property
    def error(self) -> BaseException | None:
        return self._error


def _agree_on_failure(group: Any, device: torch.device, failed: bool) -> bool:
    flag = torch.tensor([int(failed)], dtype=torch.int32, device=device)
    result = group.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(result.item())


def decode_minimax_h3_temporal_chunks_parallel(
    model: Any,
    latent: torch.Tensor,
    group: Any,
    output_callback: DecodedChunkCallback | None,
    *,
    num_windows: int,
    tokens_per_window: int,
    token_overlap: int,
    decoded_frames_per_window: int,
    main_frames: int,
    overlap_frames: int,
    output_frames: int,
    keep_ranges: tuple[tuple[int, int], ...],
    temporal_cat_dtype: torch.dtype | None = None,
) -> MiniMaxH3TemporalChunkParallelStats:
    """Decode canonical H3 temporal windows across a distributed group.

    All ranks call this function.  Only group rank zero supplies a callback;
    callback failures are retained until all data gathers and the final error
    agreement have completed.
    """

    world_size = int(group.world_size)
    rank = int(group.rank_in_group)
    if world_size <= 1:
        raise ValueError("MiniMax H3 temporal chunk parallel requires world_size > 1")
    if (output_callback is not None) != (rank == 0):
        raise ValueError("only MiniMax H3 temporal chunk group rank zero may provide the output callback")
    if num_windows < world_size:
        raise ValueError(
            "MiniMax H3 temporal chunk parallel requires at least one first-round window per rank: "
            f"windows={num_windows}, ranks={world_size}"
        )

    rounds = _num_rounds(num_windows, world_size)
    local_error: BaseException | None = None
    try:
        first_segment = _decode_live_segment(
            model,
            latent,
            rank,
            tokens_per_window=tokens_per_window,
            token_overlap=token_overlap,
            decoded_frames_per_window=decoded_frames_per_window,
            keep_ranges=keep_ranges,
            temporal_cat_dtype=temporal_cat_dtype,
        )
    except BaseException as error:
        error.__traceback__ = None
        local_error = error
        first_segment = None

    if rank == 0:
        metadata: tuple[torch.dtype, tuple[int, ...]] | dict[str, str]
        if first_segment is None:
            metadata = {
                "error": "rank zero failed to decode the first temporal window",
            }
        else:
            metadata = (first_segment.dtype, tuple(first_segment.shape))
    else:
        metadata = None  # type: ignore[assignment]
    metadata = group.broadcast_object(metadata, src=0)
    if isinstance(metadata, dict):
        raise RuntimeError(metadata.get("error", "invalid MiniMax H3 temporal chunk metadata")) from local_error
    if (
        not isinstance(metadata, tuple)
        or len(metadata) != 2
        or not isinstance(metadata[0], torch.dtype)
        or not isinstance(metadata[1], tuple)
    ):
        raise RuntimeError(f"invalid MiniMax H3 temporal chunk metadata: {metadata!r}")
    segment_dtype, segment_shape = metadata
    if first_segment is not None and (
        first_segment.dtype != segment_dtype or tuple(first_segment.shape) != segment_shape
    ):
        local_error = RuntimeError(
            "MiniMax H3 temporal chunk ranks produced inconsistent first-segment metadata: "
            f"local={(first_segment.dtype, tuple(first_segment.shape))}, leader={metadata}"
        )
        first_segment = None

    segment_bytes = int(torch.empty((), dtype=segment_dtype).element_size())
    for extent in segment_shape:
        segment_bytes *= int(extent)
    real_segments = num_windows
    placeholder_segments = rounds * world_size - real_segments
    stats = MiniMaxH3TemporalChunkParallelStats(
        rounds=rounds,
        windows=num_windows,
        real_segments=real_segments,
        placeholder_segments=placeholder_segments,
        gather_calls=rounds,
        segment_bytes=segment_bytes,
        input_bytes=(real_segments + placeholder_segments) * segment_bytes,
        output_bytes=rounds * world_size * segment_bytes,
        output_frames=output_frames,
    )

    assembler = (
        _LeaderAssembler(
            model,
            output_callback,
            main_frames=main_frames,
            overlap_frames=overlap_frames,
            output_frames=output_frames,
            device=latent.device,
        )
        if output_callback is not None
        else None
    )

    try:
        for round_index in range(rounds):
            window_index = round_index * world_size + rank
            if window_index >= num_windows or local_error is not None:
                segment = torch.zeros(segment_shape, dtype=segment_dtype, device=latent.device)
            elif round_index == 0:
                assert first_segment is not None
                segment = first_segment
            else:
                try:
                    segment = _decode_live_segment(
                        model,
                        latent,
                        window_index,
                        tokens_per_window=tokens_per_window,
                        token_overlap=token_overlap,
                        decoded_frames_per_window=decoded_frames_per_window,
                        keep_ranges=keep_ranges,
                        temporal_cat_dtype=temporal_cat_dtype,
                    )
                    if segment.dtype != segment_dtype or tuple(segment.shape) != segment_shape:
                        raise RuntimeError(
                            "MiniMax H3 temporal chunk segment metadata changed between rounds: "
                            f"window={window_index}, local={(segment.dtype, tuple(segment.shape))}, "
                            f"leader={(segment_dtype, segment_shape)}"
                        )
                except BaseException as error:
                    error.__traceback__ = None
                    local_error = error
                    segment = torch.zeros(segment_shape, dtype=segment_dtype, device=latent.device)

            if assembler is not None and round_index > 0:
                assembler.drain_previous_round()
            gathered = _gather_stack_to_rank_zero(segment, group)
            if round_index == 0:
                first_segment = None
            if assembler is None or gathered is None:
                del gathered, segment
                continue
            for slot in range(world_size):
                if round_index * world_size + slot >= num_windows:
                    break
                assembler.push(gathered[slot])
            assembler.finish_round()
            del gathered, segment

        if assembler is not None:
            assembler.finalize()
    finally:
        if assembler is not None:
            assembler.synchronize()

    callback_error = assembler.error if assembler is not None else None
    any_error = _agree_on_failure(
        group,
        latent.device,
        local_error is not None or callback_error is not None,
    )
    if any_error:
        cause = local_error if local_error is not None else callback_error
        raise RuntimeError("MiniMax H3 temporal chunk parallel failed after draining all data gathers") from cause
    return stats


__all__ = [
    "MiniMaxH3TemporalChunkParallelStats",
    "decode_minimax_h3_temporal_chunks_parallel",
]
