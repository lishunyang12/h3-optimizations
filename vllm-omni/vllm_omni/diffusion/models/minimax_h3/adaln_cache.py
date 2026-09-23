# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Default exact AdaLN memoization and optional weight-bound startup sidecars.

This is not a cross-step approximation. The main blocks' and final layer's
AdaLN outputs are reused only for identical inputs and unchanged weights. All
weights remain available for new schedules and adapters. Legacy experimental
sidecars without weight identities are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from torch import nn

from .time_request import minimax_h3_time_shift_sigmas

if TYPE_CHECKING:
    from .minimax_h3_transformer import MiniMaxH3DiTArchConfig

FORMAT_VERSION = "4"
MODES = {
    "t2va": ("video", "audio"),
    "fl2va": ("video", "audio", "image"),
    "ref2va-image": ("video", "audio", "image"),
    "ref2va-audio": ("video", "audio", "audio_ref"),
    "ref2va-mixed": ("video", "audio", "image", "audio_ref"),
}
PAYLOAD_NAMES = frozenset({"plan_timesteps", "plan_lengths", "time_embeddings", "block_params", "final_params"})


class MiniMaxH3RuntimeAdalnCache:
    """Bounded memoization of the actual runtime projection results.

    No device, task, schedule or precision is selected here. The existing
    projection remains the builder, including quantization, LoRA and TP. The
    input embedding is hashed once per DiT forward; each layer also checks its
    parameter/buffer versions. TP ranks vote before skipping a collective.
    """

    def __init__(self, *, max_bytes: int = 256 * 1024**2) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("AdaLN cache budget must be a nonnegative byte count")
        self.max_bytes = max_bytes
        self._entries: dict[tuple[str, str], tuple[Any, torch.Tensor, int]] = {}
        self._bytes = 0
        self._input: weakref.ReferenceType[torch.Tensor] | None = None
        self._key: str | None = None
        self.hits = self.misses = 0
        self.sidecar: MiniMaxH3AdalnCache | None = None
        self._sidecar_signatures: dict[str, Any] = {}
        self._sidecar_plan: int | None = None

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
        self._input = None
        self._key = None
        self.sidecar = None
        self._sidecar_signatures.clear()
        self._sidecar_plan = None

    def seed(self, sidecar: MiniMaxH3AdalnCache, modules: Mapping[str, nn.Module]) -> None:
        self.sidecar = sidecar
        self._sidecar_signatures = {name: self._signature(module) for name, module in modules.items()}

    def prepare(self, embedding: torch.Tensor) -> None:
        self._input = None
        self._key = None
        self._sidecar_plan = None
        if not self.max_bytes or torch.is_grad_enabled() or torch.compiler.is_compiling():
            return
        self._key = tensor_digest(embedding) + canonical_json(
            [
                torch.get_float32_matmul_precision(),
                torch.is_autocast_enabled(embedding.device.type),
                str(torch.get_autocast_dtype(embedding.device.type)),
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            ]
        )
        self._input = weakref.ref(embedding)
        if self.sidecar is not None:
            if math_identity(embedding.device) != self.sidecar.manifest["math"]:
                self.sidecar = None
                return
            for index, count in enumerate(self.sidecar.plan_lengths.tolist()):
                expected = self.sidecar.time_embeddings[index, :count]
                if embedding.shape == expected.shape and torch.equal(embedding, expected):
                    self._sidecar_plan = index
                    break

    @staticmethod
    def _signature(module: nn.Module) -> tuple[Any, ...]:
        # A replaced or restored storage conservatively causes a miss. The
        # containing block's offload hooks still run before this method.
        tensors = [*module.named_parameters(), *module.named_buffers()]
        # vLLM LoRA stores these tensors outside registered buffers. Its
        # suspend/resume mask can change without changing the base weights.
        for field in ("lora_a_stacked", "lora_b_stacked"):
            tensors.extend((f"{field}.{i}", value) for i, value in enumerate(getattr(module, field, ())))
        return (
            id(module),
            id(getattr(module, "quant_method", None)),
            getattr(module, "_diffusion_lora_active_slices", None),
            *(
                (name, id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)
                for name, value in tensors
            ),
        )

    def project(
        self,
        name: str,
        module: nn.Module,
        embedding: torch.Tensor,
        compute: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        if (
            torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or self._key is None
            or self._input is None
            or self._input() is not embedding
        ):
            return compute()
        key = self._key, name
        try:
            signature = self._signature(module)
            if module._forward_pre_hooks or module._forward_hooks:
                # A linear hook may mutate weights or transform the output;
                # executing it only on misses would change its semantics.
                signature = None
        except RuntimeError:
            # Inference tensors without version counters cannot establish that
            # weights stayed unchanged. Still participate in the TP vote.
            signature = None
        cached = self._entries.get(key)
        hit = signature is not None and cached is not None and cached[0] == signature
        from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group

        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            vote = torch.tensor([int(hit)], device=embedding.device, dtype=torch.int32)
            hit = int(get_tp_group().all_reduce(vote).item()) == world_size
        if hit:
            assert cached is not None
            self.hits += 1
            return cached[1]
        self.misses += 1
        if (
            world_size == 1
            and self.sidecar is not None
            and self._sidecar_plan is not None
            and signature is not None
            and self._sidecar_signatures.get(name) == signature
            and not torch.is_autocast_enabled(embedding.device.type)
        ):
            index, count = self._sidecar_plan, embedding.shape[0]
            if name.startswith("blocks."):
                value = self.sidecar.block_params[index, :count, int(name.split(".")[1])].contiguous()
            else:
                value = self.sidecar.final_params[index, :count].contiguous()
        else:
            value = compute()
        if cached is not None:
            self._bytes -= self._entries.pop(key)[2]
        size = value.numel() * value.element_size()
        if signature is not None and self._bytes + size <= self.max_bytes:
            # Retain a reusable subset when a long base-H3 schedule exceeds
            # the budget. LRU would evict every entry before the next request
            # reaches it, producing zero hits for a repeated cyclic schedule.
            # Own the cached storage even if a provider returns workspace views.
            self._entries[key] = (signature, value.detach().clone(), size)
            self._bytes += size
        return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash dtype, shape and actual bytes, including BF16, without widening."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(canonical_json([str(value.dtype), list(value.shape)]).encode())
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def projection_names(num_layers: int) -> frozenset[str]:
    prefixes = [f"blocks.{i}.adaln_proj.linear" for i in range(num_layers)]
    prefixes.append("final_layer.adaln_proj.linear")
    return frozenset(f"{prefix}.{kind}" for prefix in prefixes for kind in ("weight", "bias"))


def input_names(num_layers: int) -> frozenset[str]:
    return projection_names(num_layers) | {
        f"time_embedder.{module}.{kind}" for module in ("proj_in", "proj_out") for kind in ("weight", "bias")
    }


def architecture(arch: MiniMaxH3DiTArchConfig) -> dict[str, int]:
    return {
        key: getattr(arch, key)
        for key in ("num_layers", "hidden_size", "timestep_input_dim", "time_embed_hidden_size", "time_embed_dim")
    }


def math_identity(device: torch.device) -> dict[str, Any]:
    """Conservatively bind the builder's numerical environment (TP1 only)."""
    result = {
        "torch": str(torch.__version__),
        "device_type": device.type,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        from vllm_omni.platforms import current_omni_platform

        capability = current_omni_platform.get_device_capability(device.index or 0)
        result.update(
            cuda=torch.version.cuda,
            capability=list(capability) if capability is not None else None,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )
    return result


def schedule_contract(
    *, mode: str, num_steps: int, base_schedule: Sequence[float] | None, flow_shift: float, audio_flow_shift: float
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported AdaLN cache mode: {mode}")
    if type(num_steps) is not int or num_steps < (2 if base_schedule is None else 1):
        raise ValueError("AdaLN cache requires a positive complete denoising schedule")
    for value in (flow_shift, audio_flow_shift):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("AdaLN cache shifts must be finite and positive")
    if base_schedule is not None:
        from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

        schedule = DMD2SigmaSchedule.from_positions(base_schedule)
        if schedule.num_inference_steps != num_steps:
            raise ValueError("AdaLN cache step count does not match the explicit schedule")
        base_schedule = schedule.base_schedule
    return dict(
        mode=mode,
        num_steps=num_steps,
        base_schedule=list(base_schedule) if base_schedule is not None else None,
        flow_shift=float(flow_shift),
        audio_flow_shift=float(audio_flow_shift),
    )


def timestep_plans(contract: Mapping[str, Any]) -> list[torch.Tensor]:
    """Use the serving scheduler and its conditioning timesteps, without interpolation."""
    from .denoise_loop import MINIMAX_H3_AUDIO_REF_COND_TIMESTEP, MINIMAX_H3_IMGVID_COND_TIMESTEP

    video = minimax_h3_time_shift_sigmas(
        num_steps=contract["num_steps"], shift_scale=contract["flow_shift"], base_schedule=contract["base_schedule"]
    )
    audio = minimax_h3_time_shift_sigmas(
        num_steps=contract["num_steps"],
        shift_scale=contract["audio_flow_shift"],
        base_schedule=contract["base_schedule"],
    )
    plans: dict[tuple[float, ...], torch.Tensor] = {}
    for video_sigma, audio_sigma in zip(video[:-1], audio[:-1], strict=True):
        values = {
            "video": 1.0 - video_sigma,
            "audio": 1.0 - audio_sigma,
            "image": max(1.0 - video_sigma, MINIMAX_H3_IMGVID_COND_TIMESTEP),
            "audio_ref": max(1.0 - audio_sigma, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
        }
        plan = torch.tensor([values[key] for key in MODES[contract["mode"]]], dtype=torch.float32).unique(sorted=True)
        plans[tuple(plan.tolist())] = plan
    return list(plans.values())


class MiniMaxH3AdalnCache(nn.Module):
    """Validate an optional sidecar, then verify its weights at load.

    Hashes are checked against the *post-fusion* stream, before TP conversion.
    They cover all inputs of the cached computation, not unrelated QKV/MLP
    weights. A fixed adapter has an additional complete-file identity.
    """

    block_params: torch.Tensor
    final_params: torch.Tensor
    plan_timesteps: torch.Tensor
    plan_lengths: torch.Tensor
    time_embeddings: torch.Tensor

    def __init__(self, arch: MiniMaxH3DiTArchConfig, *, path: str, model_variant: str) -> None:
        super().__init__()
        self.path = str(Path(path).expanduser().resolve())
        self.arch = arch
        self._verified: set[str] = set()
        self._adapter_bound = False
        self._ready = False
        with safe_open(self.path, framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            if metadata.get("format_version") != FORMAT_VERSION:
                raise ValueError("Rebuild the AdaLN cache: a weight-bound format-v4 sidecar is required")
            try:
                manifest = json.loads(metadata["manifest"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid AdaLN cache manifest") from exc
            required = {"architecture", "model_variant", "schedule", "adapter_sha256", "weights", "payload", "math"}
            if not isinstance(manifest, dict) or set(manifest) != required:
                raise ValueError("Invalid AdaLN cache manifest fields")
            if any(
                not isinstance(manifest[key], dict)
                for key in ("architecture", "schedule", "weights", "payload", "math")
            ):
                raise ValueError("Invalid AdaLN cache manifest objects")
            if manifest["architecture"] != architecture(arch) or manifest["model_variant"] != model_variant:
                raise ValueError("AdaLN cache model variant or architecture mismatch")
            contract = schedule_contract(**manifest["schedule"])
            variant = "ref2va" if contract["mode"].startswith("ref2va") else "fl2va"
            if variant != model_variant:
                raise ValueError("AdaLN cache task does not match model variant")
            if set(manifest["weights"]) != input_names(arch.num_layers):
                raise ValueError("AdaLN cache must cover exactly the time embedding and AdaLN parameters")
            if set(source.keys()) != PAYLOAD_NAMES or set(manifest["payload"]) != PAYLOAD_NAMES:
                raise ValueError("Invalid AdaLN cache payload keys")
            for digest in [*manifest["weights"].values(), *manifest["payload"].values()]:
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Invalid AdaLN cache SHA256 digest")
            adapter = manifest["adapter_sha256"]
            if adapter is not None and (
                not isinstance(adapter, str) or len(adapter) != 64 or any(c not in "0123456789abcdef" for c in adapter)
            ):
                raise ValueError("Invalid AdaLN cache adapter identity")
            payload = {name: source.get_tensor(name) for name in PAYLOAD_NAMES}
        plans = timestep_plans(contract)
        width = max(len(plan) for plan in plans)
        expected_lengths = torch.tensor([len(plan) for plan in plans], dtype=torch.int64)
        expected_plans = torch.zeros((len(plans), width), dtype=torch.float32)
        for index, plan in enumerate(plans):
            expected_plans[index, : len(plan)] = plan
        shapes = {
            "block_params": ((len(plans), width, arch.num_layers, 18 * arch.hidden_size), torch.bfloat16),
            "final_params": ((len(plans), width, 2 * arch.hidden_size), torch.bfloat16),
            "time_embeddings": ((len(plans), width, arch.time_embed_dim), torch.float32),
            "plan_timesteps": (tuple(expected_plans.shape), torch.float32),
            "plan_lengths": (tuple(expected_lengths.shape), torch.int64),
        }
        for name, value in payload.items():
            shape, dtype = shapes[name]
            if tuple(value.shape) != shape or value.dtype != dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid AdaLN cache tensor: {name}")
            if tensor_digest(value) != manifest["payload"][name]:
                raise ValueError(f"AdaLN cache payload checksum mismatch: {name}")
        if not torch.equal(payload["plan_timesteps"], expected_plans) or not torch.equal(
            payload["plan_lengths"], expected_lengths
        ):
            raise ValueError("AdaLN cache timestep plans do not match the declared schedule")
        self.manifest = manifest
        self.projection_names = projection_names(arch.num_layers)
        for name, value in payload.items():
            self.register_buffer(name, value, persistent=False)

    def bind_adapter(self, adapter_sha256: str | None) -> None:
        if adapter_sha256 != self.manifest["adapter_sha256"]:
            raise ValueError("AdaLN cache adapter mismatch; rebuild for the active fixed adapter")
        self._adapter_bound = True

    def verify_weight(self, name: str, weight: torch.Tensor) -> None:
        if name not in self.manifest["weights"]:
            return
        if self._ready or name in self._verified:
            raise ValueError(f"AdaLN input was loaded more than once: {name}")
        if tensor_digest(weight) != self.manifest["weights"][name]:
            raise ValueError(f"AdaLN cache effective weight mismatch: {name}")
        self._verified.add(name)

    def finish_loading(self, device: torch.device) -> None:
        if not self._adapter_bound:
            raise ValueError("AdaLN cache requires the active adapter identity before loading")
        missing = input_names(self.arch.num_layers) - self._verified
        if missing:
            raise ValueError(f"Missing AdaLN cache source weights: {sorted(missing)}")
        if math_identity(device) != self.manifest["math"]:
            raise ValueError("AdaLN cache numerical environment mismatch; rebuild on the serving configuration")
        self.to(device=device)
        self._ready = True

    def check_request(
        self,
        *,
        mode: str,
        num_steps: int,
        base_schedule: Sequence[float] | None,
        flow_shift: float,
        audio_flow_shift: float,
    ) -> None:
        contract = schedule_contract(
            mode=mode,
            num_steps=num_steps,
            base_schedule=base_schedule,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
        )
        if contract != self.manifest["schedule"]:
            raise ValueError("AdaLN cache request schedule/task mismatch; rebuild or disable the cache")

    def lookup(self, timesteps: torch.Tensor) -> tuple[tuple[tuple[torch.Tensor, ...], ...], tuple[torch.Tensor, ...]]:
        if not self._ready:
            raise RuntimeError("AdaLN cache has not verified its source weights")
        if timesteps.dtype != torch.float32:
            raise ValueError("AdaLN cache requires exact FP32 timesteps")
        values = timesteps.reshape(-1).to(self.plan_timesteps.device)
        count = values.numel()
        if not 0 < count <= self.plan_timesteps.shape[1]:
            raise ValueError("AdaLN cache does not cover this timestep plan")
        matches = self.plan_lengths.eq(count) & self.plan_timesteps[:, :count].eq(values).all(dim=1)
        if not bool(matches.any()):
            raise ValueError("AdaLN cache does not cover this timestep plan")
        index = matches.to(torch.int64).argmax()
        blocks = self.block_params.permute(2, 0, 1, 3)[:, index, :count]
        blocks = blocks.reshape(self.arch.num_layers, -1, 6, self.arch.hidden_size)
        final = self.final_params[index, :count].reshape(-1, 2, self.arch.hidden_size)
        return tuple(tuple(layer.unbind(1)) for layer in blocks), tuple(final.unbind(1))


@torch.inference_mode()
def build_cache(
    arch: MiniMaxH3DiTArchConfig,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    contract: dict[str, Any],
    model_variant: str,
    adapter_sha256: str | None,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Stream one projection at a time; do not instantiate the full DiT.

    The four time-embedder tensors must precede the AdaLN projections. Callers
    fuse any fixed adapter before supplying this stream, just like serving.
    """
    from .minimax_h3_transformer import MiniMaxH3TimeEmbedder

    contract = schedule_contract(**contract)
    plans = timestep_plans(contract)
    width = max(len(plan) for plan in plans)
    payload = {
        "plan_timesteps": torch.zeros((len(plans), width), dtype=torch.float32),
        "plan_lengths": torch.tensor([len(plan) for plan in plans], dtype=torch.int64),
        "block_params": torch.zeros((len(plans), width, arch.num_layers, 18 * arch.hidden_size), dtype=torch.bfloat16),
        "final_params": torch.zeros((len(plans), width, 2 * arch.hidden_size), dtype=torch.bfloat16),
        "time_embeddings": torch.zeros((len(plans), width, arch.time_embed_dim), dtype=torch.float32),
    }
    for i, plan in enumerate(plans):
        payload["plan_timesteps"][i, : len(plan)] = plan

    # Use the runtime embedding implementation, including FP32 boundaries. A
    # lightweight module supplies ordinary Linear wrappers with TP1 semantics.
    class Linear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    embedder = MiniMaxH3TimeEmbedder.__new__(MiniMaxH3TimeEmbedder)
    nn.Module.__init__(embedder)
    embedder.frequency_embedding_size = arch.timestep_input_dim
    embedder.proj_in = Linear(arch.timestep_input_dim, arch.time_embed_hidden_size, device=device, dtype=torch.float32)
    embedder.proj_out = Linear(arch.time_embed_hidden_size, arch.time_embed_dim, device=device, dtype=torch.float32)
    fingerprints: dict[str, str] = {}
    pending: dict[str, torch.Tensor] = {}
    inputs: list[torch.Tensor] | None = None
    for name, weight in weights:
        if name not in input_names(arch.num_layers) or name in fingerprints:
            raise ValueError(f"Unexpected or duplicate AdaLN builder input: {name}")
        dtype = torch.float32 if name.startswith("time_embedder.") else torch.bfloat16
        if weight.dtype != dtype or not bool(torch.isfinite(weight).all()):
            raise ValueError(f"Invalid AdaLN builder input dtype/values: {name}")
        fingerprints[name] = tensor_digest(weight)
        if name.startswith("time_embedder."):
            target = dict(embedder.named_parameters())[name.removeprefix("time_embedder.")]
            if weight.shape != target.shape:
                raise ValueError(f"Invalid AdaLN time embedding shape: {name}")
            target.copy_(weight)
            continue
        if inputs is None:
            if len([key for key in fingerprints if key.startswith("time_embedder.")]) != 4:
                raise ValueError("Supply all four time embedding weights before the AdaLN projections")
            embeddings = [embedder(plan.to(device)) for plan in plans]
            for index, embedding in enumerate(embeddings):
                payload["time_embeddings"][index, : len(embedding)].copy_(embedding.cpu())
            inputs = [nn.functional.silu(embedding).to(torch.bfloat16) for embedding in embeddings]
        pending[name] = weight.to(device)
        prefix = name.rsplit(".", 1)[0]
        if f"{prefix}.weight" not in pending or f"{prefix}.bias" not in pending:
            continue
        w, bias = pending.pop(f"{prefix}.weight"), pending.pop(f"{prefix}.bias")
        is_final = prefix.startswith("final_layer.")
        out_dim = (2 if is_final else 18) * arch.hidden_size
        if w.shape != (out_dim, arch.time_embed_dim) or bias.shape != (out_dim,):
            raise ValueError(f"Invalid AdaLN projection shape: {prefix}")
        for i, x in enumerate(inputs):
            result = nn.functional.linear(x, w, bias).cpu()
            if is_final:
                payload["final_params"][i, : len(x)].copy_(result)
            else:
                payload["block_params"][i, : len(x), int(prefix.split(".")[1])].copy_(result)
        del w, bias
    if set(fingerprints) != input_names(arch.num_layers) or pending:
        raise ValueError("Incomplete AdaLN builder input weights")
    if any(not bool(torch.isfinite(value).all()) for value in payload.values()):
        raise ValueError("Non-finite AdaLN builder output")
    manifest = dict(
        architecture=architecture(arch),
        model_variant=model_variant,
        schedule=contract,
        adapter_sha256=adapter_sha256,
        weights=fingerprints,
        payload={name: tensor_digest(value) for name, value in payload.items()},
        math=math_identity(device),
    )
    return payload, manifest
