# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned, opt-in TAEH3 preview decoder for MiniMax H3.

The architecture and temporal mapping are adapted from madebyollin/taehv
(MIT, Copyright (c) 2025 Ollin Boer Bohan), revision
``62f7591f59dfbb4c3c02b7a621d180a9eeaba26c``.  Only safetensors weights are
loaded; this module never downloads or executes checkpoint-provided code.

TAEH3 is an approximate decoder.  It consumes the normalized diffusion
latent directly, unlike the full H3 VAE, and therefore deliberately does not
apply the full VAE's latent mean/std or pixel postprocessor.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

TAEH3_BACKEND_ENV = "VLLM_OMNI_MINIMAX_H3_VIDEO_DECODE_BACKEND"
TAEH3_CHECKPOINT_ENV = "VLLM_OMNI_MINIMAX_H3_TAEH3_CHECKPOINT"
TAEH3_DTYPE_ENV = "VLLM_OMNI_MINIMAX_H3_TAEH3_DTYPE"
TAEH3_CHUNK_SIZE_ENV = "VLLM_OMNI_MINIMAX_H3_TAEH3_CHUNK_SIZE"

TAEH3_REVISION = "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"
TAEH3_CHECKPOINT_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
TAEH3_CHECKPOINT_SIZE_BYTES = 22_709_752
TAEH3_SOURCE_URL = f"https://github.com/madebyollin/taehv/tree/{TAEH3_REVISION}"

logger = init_logger(__name__)


@dataclass(frozen=True)
class MiniMaxH3VideoDecoderConfig:
    """Startup-only video decoder selection.

    The normal full H3 VAE is the default.  TAEH3 requires an explicit local
    checkpoint so production startup never depends on network access.
    """

    backend: str = "h3-vae"
    checkpoint_path: Path | None = None
    dtype: str = "fp32"
    chunk_size: int = 5

    @property
    def taeh3_enabled(self) -> bool:
        return self.backend == "taeh3"

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "fp32":
            return torch.float32
        if self.dtype == "fp16":
            return torch.float16
        raise AssertionError(f"unvalidated TAEH3 dtype {self.dtype!r}")


def resolve_minimax_h3_video_decoder_config() -> MiniMaxH3VideoDecoderConfig:
    """Parse the opt-in decoder env contract without silent fallbacks."""

    backend = os.environ.get(TAEH3_BACKEND_ENV, "h3-vae").strip().lower()
    if backend not in {"h3-vae", "taeh3"}:
        raise ValueError(f"{TAEH3_BACKEND_ENV} must be h3-vae or taeh3, got {backend!r}")

    companion_names = (TAEH3_CHECKPOINT_ENV, TAEH3_DTYPE_ENV, TAEH3_CHUNK_SIZE_ENV)
    if backend == "h3-vae":
        configured = [name for name in companion_names if name in os.environ]
        if configured:
            raise ValueError(f"{', '.join(configured)} require {TAEH3_BACKEND_ENV}=taeh3")
        return MiniMaxH3VideoDecoderConfig()

    raw_checkpoint = os.environ.get(TAEH3_CHECKPOINT_ENV, "").strip()
    if not raw_checkpoint:
        raise ValueError(
            f"{TAEH3_BACKEND_ENV}=taeh3 requires an explicit local {TAEH3_CHECKPOINT_ENV}"
        )
    checkpoint_path = Path(raw_checkpoint).expanduser()

    dtype = os.environ.get(TAEH3_DTYPE_ENV, "fp32").strip().lower()
    if dtype not in {"fp32", "fp16"}:
        raise ValueError(f"{TAEH3_DTYPE_ENV} must be fp32 or fp16, got {dtype!r}")

    raw_chunk_size = os.environ.get(TAEH3_CHUNK_SIZE_ENV, "5").strip()
    try:
        chunk_size = int(raw_chunk_size, 10)
    except ValueError as exc:
        raise ValueError(f"{TAEH3_CHUNK_SIZE_ENV} must be a positive integer, got {raw_chunk_size!r}") from exc
    if chunk_size < 1:
        raise ValueError(f"{TAEH3_CHUNK_SIZE_ENV} must be a positive integer, got {chunk_size}")

    return MiniMaxH3VideoDecoderConfig(
        backend=backend,
        checkpoint_path=checkpoint_path,
        dtype=dtype,
        chunk_size=chunk_size,
    )


def validate_minimax_h3_taeh3_output_route(
    config: MiniMaxH3VideoDecoderConfig,
    *,
    nvenc_enabled: bool,
    chunked_cpu_mp4_enabled: bool,
) -> None:
    """Reject TAEH3 startup unless its bounded output route is selected."""

    if not config.taeh3_enabled:
        return
    if nvenc_enabled:
        raise ValueError(
            f"{TAEH3_BACKEND_ENV}=taeh3 does not support the full-VAE NVENC "
            "callback route; use the chunked CPU MP4 route"
        )
    if not chunked_cpu_mp4_enabled:
        from .chunked_cpu_output import MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT_ENV

        raise ValueError(
            f"{TAEH3_BACKEND_ENV}=taeh3 requires "
            f"{MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT_ENV}=1; the ordinary "
            "full-RGB worker path would materialize a multi-GiB FP32 clip"
        )


def _read_verified_taeh3_checkpoint(path: str | Path) -> tuple[Path, Mapping[str, torch.Tensor]]:
    """Read exactly the pinned artifact and return its safetensors mapping."""

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"TAEH3 checkpoint not found: {candidate}")
    if candidate.suffix != ".safetensors":
        raise ValueError(f"TAEH3 requires a .safetensors checkpoint, got {candidate}")
    payload = candidate.read_bytes()
    if len(payload) != TAEH3_CHECKPOINT_SIZE_BYTES:
        raise RuntimeError(
            "TAEH3 checkpoint size mismatch: "
            f"expected={TAEH3_CHECKPOINT_SIZE_BYTES}, actual={len(payload)}, path={candidate}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != TAEH3_CHECKPOINT_SHA256:
        raise RuntimeError(
            "TAEH3 checkpoint SHA-256 mismatch: "
            f"expected={TAEH3_CHECKPOINT_SHA256}, actual={digest}, path={candidate}"
        )

    from safetensors.torch import load

    return candidate.resolve(), load(payload)


class _Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3.0) * 3.0


class _MemBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return F.relu(value + self.conv(torch.cat((value, past), dim=1)))


class _TemporalGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = int(stride)
        self.conv = nn.Conv2d(channels, channels * self.stride, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        nt, channels, height, width = value.shape
        value = self.conv(value)
        return value.reshape(nt * self.stride, channels, height, width)


def _build_taeh3_decoder() -> nn.Sequential:
    """Build the pinned TAEH3 decoder with checkpoint-compatible names."""

    return nn.Sequential(
        _Clamp(),
        nn.Conv2d(24, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        _MemBlock(256),
        _MemBlock(256),
        _MemBlock(256),
        nn.Upsample(scale_factor=2, mode="nearest"),
        _TemporalGrow(256, 1),
        nn.Conv2d(256, 128, 3, padding=1, bias=False),
        _MemBlock(128),
        _MemBlock(128),
        _MemBlock(128),
        nn.Upsample(scale_factor=2, mode="nearest"),
        _TemporalGrow(128, 2),
        nn.Conv2d(128, 64, 3, padding=1, bias=False),
        _MemBlock(64),
        _MemBlock(64),
        _MemBlock(64),
        nn.Upsample(scale_factor=2, mode="nearest"),
        _TemporalGrow(64, 2),
        nn.Conv2d(64, 64, 3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 12, 3, padding=1),
    )


def taeh3_decoded_pixel_shape(latent_shape: tuple[int, ...] | torch.Size) -> tuple[int, int, int, int, int]:
    """Return the BCTHW output shape for an H3 normalized latent."""

    if len(latent_shape) != 5:
        raise ValueError(f"TAEH3 requires a rank-5 BCTHW latent, got {tuple(latent_shape)}")
    batch, channels, latent_frames, latent_height, latent_width = map(int, latent_shape)
    if batch < 1 or channels != 24 or min(latent_frames, latent_height, latent_width) < 1:
        raise ValueError(f"TAEH3 requires a nonempty 24-channel BCTHW latent, got {tuple(latent_shape)}")
    if latent_frames % 5 != 2:
        raise ValueError(f"H3 latent time must be 5*k+2, got {latent_frames}")
    raw_frames = latent_frames * 4
    output_frames = sum(index % 20 >= 3 for index in range(raw_frames))
    return batch, 3, output_frames, latent_height * 16, latent_width * 16


class _IdentityVideoProcessor:
    """Adapter for the existing normalized-frame chunked MP4 sink."""

    @staticmethod
    def revert_tensor(value: torch.Tensor) -> torch.Tensor:
        return value


class MiniMaxH3TAEH3Decoder(nn.Module):
    """Output-rank-only TAEH3 decoder with bounded temporal feature memory."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        chunk_size: int = 5,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float32, torch.float16}:
            raise ValueError(f"TAEH3 compute dtype must be float32 or float16, got {dtype}")
        if int(chunk_size) < 1:
            raise ValueError(f"TAEH3 chunk_size must be positive, got {chunk_size}")

        self.decoder = _build_taeh3_decoder()
        resolved_path, raw = _read_verified_taeh3_checkpoint(checkpoint_path)
        expected = {f"decoder.{name}": value for name, value in self.decoder.state_dict().items()}
        actual_keys = {name for name in raw if name.startswith("decoder.")}
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise ValueError(
                "TAEH3 decoder keys mismatch: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        state: dict[str, torch.Tensor] = {}
        for full_name, expected_value in expected.items():
            value = raw[full_name]
            if tuple(value.shape) != tuple(expected_value.shape):
                raise ValueError(
                    f"TAEH3 weight {full_name} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected_value.shape)}"
                )
            if not value.dtype.is_floating_point:
                raise TypeError(f"TAEH3 weight {full_name} must be floating point, got {value.dtype}")
            state[full_name.removeprefix("decoder.")] = value
        self.decoder.load_state_dict(state, strict=True)
        del raw, state

        self.compute_dtype = dtype
        self.chunk_size = int(chunk_size)
        self.checkpoint_path = resolved_path
        self._device_target = torch.device(device)
        self._omni_component_cache: Any | None = None
        self.eval().requires_grad_(False).to(device=self._device_target, dtype=dtype)
        logger.warning(
            "MINIMAX_H3_TAEH3 marker=v1 status=loaded revision=%s sha256=%s "
            "dtype=%s chunk_size=%d device=%s quality=lossy",
            TAEH3_REVISION,
            TAEH3_CHECKPOINT_SHA256,
            str(dtype).removeprefix("torch."),
            self.chunk_size,
            self._device_target,
        )

    def set_omni_component_cache(self, cache: Any | None) -> None:
        self._omni_component_cache = cache

    def load_to_device(self) -> None:
        self.to(device=self._device_target, dtype=self.compute_dtype)

    def offload_to_cpu(self) -> None:
        self.to(device="cpu")

    @staticmethod
    def supports_chunked_nvenc() -> bool:
        return False

    @staticmethod
    def validate_chunked_output() -> None:
        from .chunked_cpu_output import validate_chunked_cpu_mp4_runtime

        validate_chunked_cpu_mp4_runtime()

    def _decode_chunk(
        self,
        value: torch.Tensor,
        memory: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        batch, time, channels, height, width = value.shape
        flat = value.reshape(batch * time, channels, height, width)
        for index, layer in enumerate(self.decoder):
            if isinstance(layer, _MemBlock):
                _, current_channels, current_height, current_width = flat.shape
                sequence = flat.reshape(batch, -1, current_channels, current_height, current_width)
                previous = memory.get(index)
                if previous is None:
                    previous = torch.zeros_like(sequence[:, :1])
                past = torch.cat((previous, sequence[:, :-1]), dim=1).reshape_as(flat)
                memory[index] = sequence[:, -1:].contiguous()
                flat = layer(flat, past)
            else:
                flat = layer(flat)
        flat = F.pixel_shuffle(flat, 2).clamp_(0.0, 1.0)
        _, output_channels, output_height, output_width = flat.shape
        return flat.reshape(batch, -1, output_channels, output_height, output_width)

    @torch.inference_mode()
    def decode_ntchw(
        self,
        latents: torch.Tensor,
        *,
        output_callback: Callable[[torch.Tensor, int, int], None] | None = None,
    ) -> torch.Tensor | None:
        """Decode NTCHW latents, optionally streaming normalized BCTHW chunks."""

        expected = taeh3_decoded_pixel_shape(latents.permute(0, 2, 1, 3, 4).shape)
        parameter = next(self.decoder.parameters())
        value = latents.to(device=parameter.device, dtype=self.compute_dtype)
        memory: dict[int, torch.Tensor] = {}
        output_chunks: list[torch.Tensor] = []
        write_pos = 0
        chunk_count = 0
        total_frames = expected[2]
        for start in range(0, int(value.shape[1]), self.chunk_size):
            decoded = self._decode_chunk(value[:, start : start + self.chunk_size], memory)
            keep = [index for index in range(int(decoded.shape[1])) if (start * 4 + index) % 20 >= 3]
            decoded = decoded[:, keep]
            if int(decoded.shape[1]) < 1:
                raise RuntimeError(f"TAEH3 temporal mapping produced an empty output chunk at latent {start}")
            if output_callback is None:
                output_chunks.append(decoded)
            else:
                # Keep this as a view. The sink performs the one required
                # BTHWC materialization while producing uint8; making BCTHW
                # contiguous here would add another full-resolution buffer.
                bcthw = decoded.permute(0, 2, 1, 3, 4)
                output_callback(bcthw, write_pos, total_frames)
            write_pos += int(decoded.shape[1])
            chunk_count += 1

        if write_pos != total_frames:
            raise RuntimeError(f"TAEH3 wrote {write_pos} frames, expected {total_frames}")
        logger.info(
            "MINIMAX_H3_TAEH3 marker=v1 status=decoded chunks=%d frames=%d "
            "height=%d width=%d dtype=%s",
            chunk_count,
            total_frames,
            expected[3],
            expected[4],
            str(self.compute_dtype).removeprefix("torch."),
        )
        if output_callback is not None:
            return None
        output = torch.cat(output_chunks, dim=1)
        if tuple(output.permute(0, 2, 1, 3, 4).shape) != expected:
            raise RuntimeError(
                "TAEH3 output shape mismatch: "
                f"actual={tuple(output.permute(0, 2, 1, 3, 4).shape)}, expected={expected}"
            )
        return output

    @torch.inference_mode()
    def decode_latent(
        self,
        latent: torch.Tensor,
        *,
        return_output: bool = True,
    ) -> torch.Tensor | None:
        if not return_output:
            return None
        if latent.ndim != 5:
            raise ValueError(f"TAEH3 requires a rank-5 BCTHW latent, got {tuple(latent.shape)}")
        output = self.decode_ntchw(latent.permute(0, 2, 1, 3, 4).contiguous())
        assert output is not None
        return output.permute(0, 2, 1, 3, 4).contiguous().float()

    @torch.inference_mode()
    def decode_latent_to_chunked_cpu_mp4(
        self,
        latent: torch.Tensor,
        *,
        height: int,
        width: int,
        fps: int,
        audio_sample_rate: int,
        video_codec_options: dict[str, str] | None,
        return_output: bool,
    ) -> Any:
        """Pipeline TAEH3 chunks into the existing pinned D2H/CPU MP4 sink."""

        if not return_output:
            raise RuntimeError("TAEH3 chunked output is valid on the output rank only")
        if latent.ndim != 5:
            raise ValueError(f"TAEH3 requires a rank-5 BCTHW latent, got {tuple(latent.shape)}")
        from .chunked_cpu_output import MiniMaxH3ChunkedCpuMp4Output

        sink = MiniMaxH3ChunkedCpuMp4Output(
            width=width,
            height=height,
            fps=fps,
            audio_sample_rate=audio_sample_rate,
            max_chunk_frames=min(int(latent.shape[2]), self.chunk_size) * 4,
            device=latent.device,
            video_codec_options=video_codec_options,
        )
        processor = _IdentityVideoProcessor()
        try:
            self.decode_ntchw(
                latent.permute(0, 2, 1, 3, 4).contiguous(),
                output_callback=lambda frames, first, total: sink.submit_decoded(
                    frames,
                    first,
                    total,
                    processor,
                ),
            )
        except BaseException:
            sink.abort()
            raise
        return sink


__all__ = [
    "MiniMaxH3TAEH3Decoder",
    "MiniMaxH3VideoDecoderConfig",
    "TAEH3_BACKEND_ENV",
    "TAEH3_CHECKPOINT_ENV",
    "TAEH3_CHECKPOINT_SHA256",
    "TAEH3_CHECKPOINT_SIZE_BYTES",
    "TAEH3_CHUNK_SIZE_ENV",
    "TAEH3_DTYPE_ENV",
    "TAEH3_REVISION",
    "resolve_minimax_h3_video_decoder_config",
    "taeh3_decoded_pixel_shape",
    "validate_minimax_h3_taeh3_output_route",
]
