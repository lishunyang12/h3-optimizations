# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU-resident chunked NVENC output for MiniMax H3.

The H3 video VAE already decodes long videos as finalized temporal chunks.
This module consumes those chunks before a full RGB tensor is assembled,
quantizes them on CUDA, and feeds their CUDA pointers directly to NVENC.  Only
the compressed H.264 packets and the much smaller audio waveform reach host
memory.
"""

from __future__ import annotations

import io
import os
import queue
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import torch


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class EncodedH264Packet:
    """One NVENC Annex-B packet and its presentation timestamp."""

    data: bytes
    timestamp: int
    picture_type: int
    sequence: int


class _CudaArray:
    """Small owner-independent CUDA Array Interface descriptor."""

    def __init__(self, tensor: torch.Tensor) -> None:
        element_size = tensor.element_size()
        self.__cuda_array_interface__ = {
            "shape": tuple(int(dim) for dim in tensor.shape),
            "strides": tuple(int(stride) * element_size for stride in tensor.stride()),
            "data": (int(tensor.data_ptr()), False),
            "typestr": "|u1",
            "version": 3,
        }


class _CudaPackedFrame:
    """Keep a torch allocation alive while PyNvVideoCodec consumes its CAI."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self._cai = _CudaArray(tensor)

    def cuda(self) -> _CudaArray:
        return self._cai


@dataclass
class _QueuedChunk:
    rgba: torch.Tensor
    ready: torch.cuda.Event
    first_frame: int


def _audio_to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        samples = audio.detach().float().cpu().numpy()
    else:
        samples = np.asarray(audio, dtype=np.float32)
    samples = np.squeeze(samples).astype(np.float32, copy=False)
    if samples.ndim == 1:
        return np.ascontiguousarray(samples.reshape(1, -1))
    if samples.ndim != 2:
        raise ValueError(f"unexpected MiniMax H3 audio shape {samples.shape}")
    if samples.shape[0] > samples.shape[1]:
        samples = samples.T
    if samples.shape[0] > 2:
        samples = samples[:2]
    return np.ascontiguousarray(samples)


def mux_h264_aac_mp4(
    packets: list[EncodedH264Packet],
    *,
    width: int,
    height: int,
    fps: int,
    audio: Any | None,
    audio_sample_rate: int,
) -> bytes:
    """Remux NVENC Annex-B H.264 packets and optional PCM audio into MP4."""

    if not packets:
        raise RuntimeError("NVENC produced no H.264 packets")

    import av

    buffer = io.BytesIO()
    video_time_base = Fraction(1, fps)
    with av.open(buffer, mode="w", format="mp4") as container:
        video_stream = container.add_stream("h264", rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.time_base = video_time_base
        video_stream.codec_context.time_base = video_time_base

        samples: np.ndarray | None = None
        layout: str | None = None
        audio_stream = None
        if audio is not None:
            samples = _audio_to_numpy(audio)
            layout = "stereo" if samples.shape[0] == 2 else "mono"
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = layout

        for encoded in sorted(packets, key=lambda item: (item.timestamp, item.sequence)):
            packet = av.Packet(encoded.data)
            packet.stream = video_stream
            packet.pts = encoded.timestamp
            packet.dts = encoded.timestamp
            packet.duration = 1
            packet.time_base = video_time_base
            container.mux(packet)

        if audio_stream is not None and samples is not None and layout is not None:
            audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
            audio_frame.sample_rate = audio_sample_rate
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, audio_sample_rate)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)

    return buffer.getvalue()


class MiniMaxH3NvencOutput:
    """Bounded VAE-chunk -> CUDA RGBA -> NVENC -> MP4 pipeline.

    ``submit_decoded`` intentionally never raises.  All VAE ranks must finish
    their tile collectives even if rank zero's encoder fails; ``finish`` raises
    the deferred error after the distributed decode is complete.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(f"MiniMax H3 NVENC output requires a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.device = device
        self._queue: queue.Queue[_QueuedChunk | None] = queue.Queue(
            maxsize=_env_int("MINIMAX_H3_NVENC_QUEUE_CHUNKS", 2)
        )
        self._packets: list[EncodedH264Packet] = []
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._submitted_frames = 0
        self._expected_frames: int | None = None
        self._nvenc_seconds = 0.0
        self._nvenc_wait_seconds = 0.0
        self._pipeline_start: float | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._encode_worker,
            name="minimax-h3-nvenc",
            daemon=True,
        )
        self._thread.start()

    def _set_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def _get_error(self) -> BaseException | None:
        with self._error_lock:
            return self._error

    def submit_decoded(
        self,
        decoded: torch.Tensor,
        first_frame: int,
        total_frames: int,
        processor: Any,
    ) -> None:
        """Quantize one finalized BCTHW decoder chunk and enqueue it on CUDA."""

        if self._get_error() is not None:
            return
        try:
            if self._pipeline_start is None:
                self._pipeline_start = time.perf_counter()
            frames = processor.revert_tensor(decoded)
            if frames.ndim == 4:
                frames = frames.unsqueeze(0).transpose(1, 2)
            if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[1] != 3:
                raise ValueError(f"unexpected decoded video chunk shape {tuple(frames.shape)}")
            if frames.device.type != "cuda":
                raise ValueError(f"decoded video chunk must stay on CUDA, got {frames.device}")

            rgb = frames[0, :, :, : self.height, : self.width].permute(1, 2, 3, 0)
            rgb_u8 = torch.round(rgb.float().clamp_(0, 1).mul_(255)).to(torch.uint8)
            rgba = torch.empty(
                (*rgb_u8.shape[:-1], 4),
                dtype=torch.uint8,
                device=rgb_u8.device,
            )
            rgba[..., :3].copy_(rgb_u8)
            rgba[..., 3].fill_(255)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device=rgba.device))

            chunk_frames = int(rgba.shape[0])
            self._expected_frames = int(total_frames)
            self._submitted_frames += chunk_frames
            self._queue.put(_QueuedChunk(rgba=rgba, ready=ready, first_frame=int(first_frame)))
        except BaseException as exc:
            self._set_error(exc)

    def _append_packets(self, encoded_packets: list[dict[str, Any]]) -> None:
        for packet in encoded_packets:
            data = bytes(packet["data"])
            if not data:
                continue
            self._packets.append(
                EncodedH264Packet(
                    data=data,
                    timestamp=int(packet.get("timestamp", len(self._packets))),
                    picture_type=int(packet.get("picture_type", 0)),
                    sequence=len(self._packets),
                )
            )

    def _encode_worker(self) -> None:
        try:
            import PyNvVideoCodec as nvc

            torch.cuda.set_device(self.device)
            encoder = nvc.CreateEncoder(
                self.width,
                self.height,
                "ABGR",
                False,
                gpu_id=int(self.device.index),
                codec="h264",
                preset=os.environ.get("MINIMAX_H3_NVENC_PRESET", "P1"),
                tuning_info=os.environ.get("MINIMAX_H3_NVENC_TUNING", "low_latency"),
                rc=os.environ.get("MINIMAX_H3_NVENC_RC", "constqp"),
                constqp=_env_int("MINIMAX_H3_NVENC_CONSTQP", 18, minimum=0),
                fps=self.fps,
                # The PyNvVideoCodec option is the number of B frames, not a
                # boolean switch.  Keep it at zero because its packet dicts
                # expose PTS but no independent DTS for reordered output.
                bf=0,
                gop=_env_int("MINIMAX_H3_NVENC_GOP", self.fps * 10),
                repeatspspps=1,
                colorspace="bt709",
            )
            num_b_frames = int(encoder.GetNumBFrames())
            if num_b_frames != 0:
                raise RuntimeError(
                    "MiniMax H3 NVENC output requires decode-order PTS/DTS; "
                    f"encoder unexpectedly configured {num_b_frames} B frame(s)"
                )
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        break
                    wait_start = time.perf_counter()
                    item.ready.synchronize()
                    self._nvenc_wait_seconds += time.perf_counter() - wait_start
                    for offset, frame in enumerate(item.rgba.unbind(0)):
                        pic_params = nvc.NV_ENC_PIC_PARAMS()
                        pic_params.inputTimeStamp = item.first_frame + offset
                        pic_params.inputDuration = 1
                        encode_start = time.perf_counter()
                        encoded = encoder.Encode(_CudaPackedFrame(frame), pic_params)
                        self._nvenc_seconds += time.perf_counter() - encode_start
                        self._append_packets(encoded)
                finally:
                    self._queue.task_done()
            encode_start = time.perf_counter()
            self._append_packets(encoder.EndEncode())
            self._nvenc_seconds += time.perf_counter() - encode_start
        except BaseException as exc:
            self._set_error(exc)
            # Never strand the producer in queue.put() after an encoder error.
            while True:
                item = self._queue.get()
                self._queue.task_done()
                if item is None:
                    break

    def finish(self, audio: Any | None, audio_sample_rate: int) -> tuple[bytes, dict[str, float]]:
        """Flush NVENC, mux audio, and return MP4 bytes plus timing counters."""

        if self._closed:
            raise RuntimeError("MiniMax H3 NVENC output pipeline is already closed")
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        error = self._get_error()
        if error is not None:
            raise RuntimeError("MiniMax H3 GPU NVENC output pipeline failed") from error
        if self._expected_frames is None or self._submitted_frames != self._expected_frames:
            raise RuntimeError(
                "MiniMax H3 NVENC frame count mismatch: "
                f"submitted={self._submitted_frames}, expected={self._expected_frames}"
            )

        mux_start = time.perf_counter()
        mp4 = mux_h264_aac_mp4(
            self._packets,
            width=self.width,
            height=self.height,
            fps=self.fps,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
        )
        mux_seconds = time.perf_counter() - mux_start
        pipeline_seconds = 0.0
        if self._pipeline_start is not None:
            pipeline_seconds = time.perf_counter() - self._pipeline_start
        stats = {
            "nvenc_frames": float(self._submitted_frames),
            "nvenc_h264_bytes": float(sum(len(packet.data) for packet in self._packets)),
            "nvenc_mp4_bytes": float(len(mp4)),
            "nvenc_encode_s": self._nvenc_seconds,
            "nvenc_wait_for_vae_s": self._nvenc_wait_seconds,
            "nvenc_mux_s": mux_seconds,
            "nvenc_pipeline_s": pipeline_seconds,
        }
        return mp4, stats

    def abort(self) -> None:
        """Stop the encoder thread without masking the caller's exception."""

        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()


__all__ = ["EncodedH264Packet", "MiniMaxH3NvencOutput", "mux_h264_aac_mp4"]
