# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    FORMAT_VERSION,
    MODES,
    MiniMaxH3AdalnCache,
    MiniMaxH3RuntimeAdalnCache,
    build_cache,
    canonical_json,
    input_names,
    schedule_contract,
    tensor_digest,
    timestep_plans,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def tp1(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1)


def _contract(mode="t2va", **updates):
    values = dict(
        mode=mode, num_steps=4, base_schedule=(0.999, 0.749, 0.5, 0.25, 0.0), flow_shift=12.0, audio_flow_shift=3.0
    )
    return schedule_contract(**(values | updates))


def _fixture(tmp_path, mode="t2va", adapter=None, *, base_model=False):
    arch = h3.MiniMaxH3DiTArchConfig(
        num_layers=2, hidden_size=8, timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4
    )
    generator = torch.Generator().manual_seed(4)
    weights = {}
    for name, shape in {
        "time_embedder.proj_in.weight": (8, 4),
        "time_embedder.proj_in.bias": (8,),
        "time_embedder.proj_out.weight": (4, 8),
        "time_embedder.proj_out.bias": (4,),
    }.items():
        weights[name] = torch.randn(shape, generator=generator)
    for name in sorted(input_names(arch.num_layers)):
        if name.startswith("time_embedder."):
            continue
        width = (2 if name.startswith("final_layer.") else 18) * arch.hidden_size
        shape = (width, arch.time_embed_dim) if name.endswith("weight") else (width,)
        weights[name] = torch.randn(shape, generator=generator).to(torch.bfloat16)
    variant = "ref2va" if mode.startswith("ref2va") else "fl2va"
    payload, manifest = build_cache(
        arch,
        weights.items(),
        contract=_contract(mode, **({"num_steps": 50, "base_schedule": None} if base_model else {})),
        model_variant=variant,
        adapter_sha256=adapter,
        device=torch.device("cpu"),
    )
    path = tmp_path / "adaln.safetensors"
    _save(path, payload, manifest)
    return arch, weights, payload, manifest, path


def _save(path, payload, manifest):
    save_file(payload, str(path), metadata={"format_version": FORMAT_VERSION, "manifest": canonical_json(manifest)})


def _ready(arch, weights, path, variant="fl2va", adapter=None):
    cache = MiniMaxH3AdalnCache(arch, path=str(path), model_variant=variant)
    cache.bind_adapter(adapter)
    for name, weight in weights.items():
        cache.verify_weight(name, weight)
    cache.finish_loading(torch.device("cpu"))
    return cache


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("base_model", [False, True], ids=["explicit-ladder", "base-h3-50"])
def test_builder_projection_parity_all_tasks(tmp_path, mode, base_model):
    arch, weights, payload, manifest, path = _fixture(tmp_path, mode, base_model=base_model)
    cache = _ready(arch, weights, path, manifest["model_variant"])
    for i, plan in enumerate(timestep_plans(manifest["schedule"])):
        blocks, final = cache.lookup(plan)
        embedding = payload["time_embeddings"][i, : len(plan)]
        activated = torch.nn.functional.silu(embedding).to(torch.bfloat16)
        for block_index, pieces in enumerate(blocks):
            prefix = f"blocks.{block_index}.adaln_proj.linear"
            expected = torch.nn.functional.linear(activated, weights[prefix + ".weight"], weights[prefix + ".bias"])
            actual = torch.stack(pieces, dim=1).reshape(len(plan), -1)
            assert torch.equal(actual, expected)
        prefix = "final_layer.adaln_proj.linear"
        expected = torch.nn.functional.linear(activated, weights[prefix + ".weight"], weights[prefix + ".bias"])
        assert torch.equal(torch.stack(final, dim=1).reshape(len(plan), -1), expected)
    assert not cache.state_dict()


def test_runtime_reuses_exact_outputs_and_tracks_changed_weights(mocker):
    cache = MiniMaxH3RuntimeAdalnCache()
    linear = torch.nn.Linear(4, 8)
    x = torch.randn(2, 4)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        cache.prepare(x)
        first = cache.project("block", linear, x, compute)
        first_copy = first.clone()
        first.zero_()  # The caller/provider cannot corrupt the owned cache.
        assert torch.equal(cache.project("block", linear, x, compute), first_copy)
        assert compute.call_count == 1
        linear.weight.add_(0.25)
        assert torch.equal(cache.project("block", linear, x, compute), linear(x))
        assert compute.call_count == 2
        other = x.clone()
        other[0, 0] += 1
        cache.prepare(other)
        cache.project("block", linear, other, lambda: linear(other))
        cache.prepare(x)
        cache.project("block", linear, x, compute)
        assert compute.call_count == 2
    assert cache.hits == 2


def test_runtime_budget_retains_reusable_entries_without_schedule_thrashing(mocker):
    cache = MiniMaxH3RuntimeAdalnCache(max_bytes=32)
    linear = torch.nn.Linear(2, 4)
    x = torch.ones(2, 2)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        cache.prepare(x)
        for name in ("first", "second", "first"):
            cache.project(name, linear, x, compute)
            assert cache._bytes <= 32
        assert compute.call_count == 2
        assert cache.hits == 1
        cache.clear()
        assert cache._bytes == 0 and not cache._entries


def test_dynamic_lora_buffers_and_suspend_mask_invalidate_entries(mocker):
    cache = MiniMaxH3RuntimeAdalnCache()
    linear = torch.nn.Linear(2, 2)
    linear.lora_a_stacked = (torch.ones(1, 2),)
    linear.lora_b_stacked = (torch.ones(2, 1),)
    linear._diffusion_lora_active_slices = (True,)
    x = torch.ones(1, 2)

    def compute():
        result = linear(x)
        if linear._diffusion_lora_active_slices[0]:
            result = result + x @ linear.lora_a_stacked[0].T @ linear.lora_b_stacked[0].T
        return result

    counted = mocker.Mock(side_effect=compute)
    with torch.no_grad():
        cache.prepare(x)
        cache.project("a", linear, x, counted)
        cache.project("a", linear, x, counted)
        assert counted.call_count == 1
        linear.lora_b_stacked[0].add_(1)
        assert torch.equal(cache.project("a", linear, x, counted), compute())
        linear._diffusion_lora_active_slices = (False,)
        assert torch.equal(cache.project("a", linear, x, counted), compute())
        assert counted.call_count == 3


def test_linear_hooks_run_even_after_cache_warmup():
    cache = MiniMaxH3RuntimeAdalnCache()
    linear = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    with torch.no_grad():
        cache.prepare(x)
        cache.project("a", linear, x, lambda: linear(x))

        # A hook is allowed to change the projection on each invocation.
        def change_weight(module, args):
            module.weight.add_(1)

        hook = linear.register_forward_pre_hook(change_weight)
        try:
            previous = linear.weight.clone()
            cache.project("a", linear, x, lambda: linear(x))
            assert torch.equal(linear.weight, previous + 1)
            cache.project("a", linear, x, lambda: linear(x))
            assert torch.equal(linear.weight, previous + 2)
        finally:
            hook.remove()


def test_disabled_and_grad_paths_keep_original_computation(monkeypatch, mocker):
    import vllm_omni.diffusion.models.minimax_h3.adaln_cache as module

    monkeypatch.setattr(module, "tensor_digest", mocker.Mock(side_effect=AssertionError("must not hash")))
    linear = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2, requires_grad=True)
    for cache in (MiniMaxH3RuntimeAdalnCache(max_bytes=0), MiniMaxH3RuntimeAdalnCache()):
        cache.prepare(x)
        cache.project("a", linear, x, lambda: linear(x)).sum().backward()
        assert x.grad is not None
    with torch.no_grad():
        cache = MiniMaxH3RuntimeAdalnCache(max_bytes=0)
        cache.prepare(x)
        assert torch.equal(cache.project("a", linear, x, lambda: linear(x)), linear(x))


def test_tp_rank_miss_forces_every_rank_to_recompute(monkeypatch, mocker):
    import vllm.distributed

    cache = MiniMaxH3RuntimeAdalnCache()
    linear = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        cache.prepare(x)
        cache.project("a", linear, x, compute)
        votes = mocker.Mock(side_effect=[torch.tensor([1]), torch.tensor([2])])
        monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: SimpleNamespace(all_reduce=votes))
        cache.project("a", linear, x, compute)
        assert compute.call_count == 2
        cache.project("a", linear, x, compute)
        assert compute.call_count == 2
        assert votes.call_count == 2


@pytest.mark.parametrize("change", ["variant", "architecture", "weights", "payload", "plans", "nan", "version"])
def test_reject_invalid_sidecar(tmp_path, change):
    arch, weights, payload, manifest, path = _fixture(tmp_path)
    payload = {key: value.clone() for key, value in payload.items()}
    manifest = copy.deepcopy(manifest)
    if change == "variant":
        manifest["model_variant"] = "ref2va"
    elif change == "architecture":
        manifest["architecture"]["hidden_size"] += 1
    elif change == "weights":
        manifest["weights"].pop(next(iter(weights)))
    elif change == "payload":
        payload["block_params"][0, 0, 0, 0] += 1
    elif change == "plans":
        payload["plan_timesteps"][0, 0] += 0.01
        manifest["payload"]["plan_timesteps"] = tensor_digest(payload["plan_timesteps"])
    elif change == "nan":
        payload["final_params"][0, 0, 0] = float("nan")
        manifest["payload"]["final_params"] = tensor_digest(payload["final_params"])
    _save(path, payload, manifest)
    if change == "version":
        save_file(payload, str(path), metadata={"format_version": "3"})
    with pytest.raises(ValueError):
        MiniMaxH3AdalnCache(arch, path=str(path), model_variant="fl2va")


def test_weight_adapter_schedule_and_uncovered_timestep_checks(tmp_path):
    arch, weights, _, manifest, path = _fixture(tmp_path, adapter="a" * 64)
    cache = MiniMaxH3AdalnCache(arch, path=str(path), model_variant="fl2va")
    with pytest.raises(ValueError, match="adapter mismatch"):
        cache.bind_adapter(None)
    cache.bind_adapter("a" * 64)
    with pytest.raises(ValueError, match="Missing"):
        cache.finish_loading(torch.device("cpu"))
    name = next(iter(weights))
    with pytest.raises(ValueError, match="effective weight mismatch"):
        cache.verify_weight(name, weights[name] + 1)
    for name, weight in weights.items():
        cache.verify_weight(name, weight)
    cache.finish_loading(torch.device("cpu"))
    for update in (
        {"flow_shift": 10.0},
        {"audio_flow_shift": 4.0},
        {"mode": "fl2va"},
        {"base_schedule": [0.99, 0.74, 0.5, 0.25, 0]},
    ):
        with pytest.raises(ValueError):
            cache.check_request(**(manifest["schedule"] | update))
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(torch.tensor([0.123]))


def test_adapter_switch_invalidates_runtime_cache(mocker):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    cache = mocker.Mock()
    pipeline.transformer = SimpleNamespace(adaln_cache=cache)
    from vllm.lora.request import LoRARequest

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(lora_request=LoRARequest("one", 1, "one"), lora_scale=1.0)
    pipeline._prepare_adaln_adapter(sampling)
    pipeline._prepare_adaln_adapter(sampling)
    assert cache.clear.call_count == 1
    sampling.lora_scale = 0.5
    pipeline._prepare_adaln_adapter(sampling)
    sampling.lora_request = None
    pipeline._prepare_adaln_adapter(sampling)
    assert cache.clear.call_count == 3


def test_bad_sidecar_never_drops_runtime_weights(tmp_path):
    arch, weights, _, _, path = _fixture(tmp_path)
    candidate = MiniMaxH3AdalnCache(arch, path=str(path), model_variant="fl2va")
    component = SimpleNamespace(_adaln_sidecar_candidate=candidate)
    changed = list(weights.items())
    changed[0] = changed[0][0], changed[0][1] + 1
    result = list(MiniMaxH3Pipeline._verify_adaln_weights(component, iter(changed)))
    assert len(result) == len(changed)
    assert all(tensor is source for (_, tensor), (_, source) in zip(result, changed))
    assert component._adaln_sidecar_candidate is None


@pytest.mark.parametrize("tp_size", [1, 2])
def test_native_constructor_enables_cache_by_default(monkeypatch, tp_size):
    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import (
        _FakeAttention,
        _FakeLinear,
        _small_od_config,
    )

    for name in ("ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear"):
        monkeypatch.setattr(h3, name, _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: tp_size)
    config = _small_od_config()
    model = h3.MiniMaxH3DiTModel(config, diffusers_weights=False)
    assert model.adaln_cache.max_bytes > 0
    assert model.blocks[0].adaln_proj._adaln_cache is model.adaln_cache
    assert model.final_layer.adaln_proj._adaln_cache is model.adaln_cache
    assert "blocks.0.adaln_proj.linear.weight" in dict(model.named_parameters())
    config.cache_config = {"minimax_h3_adaln_cache": False}
    disabled = h3.MiniMaxH3DiTModel(config, diffusers_weights=False)
    assert disabled.adaln_cache.max_bytes == 0


def test_optional_sidecar_seeds_runtime_projection(tmp_path, mocker):
    arch, weights, payload, _, path = _fixture(tmp_path)
    sidecar = _ready(arch, weights, path)
    runtime = MiniMaxH3RuntimeAdalnCache()
    name = "blocks.0.adaln_proj.linear"
    linear = torch.nn.Linear(arch.time_embed_dim, 18 * arch.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(weights[name + ".weight"])
        linear.bias.copy_(weights[name + ".bias"])
        runtime.seed(sidecar, {name: linear})
        x = payload["time_embeddings"][1, :2].clone()
        runtime.prepare(x)
        compute = mocker.Mock(side_effect=lambda: linear(torch.nn.functional.silu(x).to(torch.bfloat16)))
        result = runtime.project(name, linear, x, compute)
        assert compute.call_count == 0
        assert torch.equal(result, compute())
        linear.weight.add_(1)
        runtime.project(name, linear, x, compute)
        assert compute.call_count == 2


def test_builder_rejects_missing_or_duplicate_weights(tmp_path):
    arch, weights, _, manifest, _ = _fixture(tmp_path)
    for source in (list(weights.items())[:-1], [*weights.items(), next(iter(weights.items()))]):
        with pytest.raises(ValueError):
            build_cache(
                arch,
                source,
                contract=manifest["schedule"],
                model_variant="fl2va",
                adapter_sha256=None,
                device=torch.device("cpu"),
            )


def test_native_checkpoint_builder_cli(tmp_path, monkeypatch):
    from tools.minimax_h3.build_adaln_cache import main

    arch, weights, _, _, _ = _fixture(tmp_path)
    root = tmp_path / "transformer"
    root.mkdir()
    (root / "config.json").write_text(canonical_json(asdict(arch)))
    save_file(weights, str(root / "model.safetensors"))
    output = tmp_path / "cli.safetensors"
    argv = [
        "build_adaln_cache",
        "--transformer-path",
        str(root),
        "--output",
        str(output),
        "--model-variant",
        "fl2va",
        "--num-inference-steps",
        "4",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    _ready(arch, weights, output)
    with pytest.raises(ValueError, match="overwrite"):
        main()


@pytest.mark.parametrize("base_model", [False, True], ids=["fasth3-four", "base-h3-50"])
def test_runtime_projection_matches_sidecar_using_actual_h3_forwards(tmp_path, monkeypatch, mocker, base_model):
    class Linear(torch.nn.Linear):
        def __init__(self, in_features, out_features, *, params_dtype, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)

        def forward(self, x):
            return super().forward(x), None

    monkeypatch.setattr(h3, "ColumnParallelLinear", Linear)
    monkeypatch.setattr(h3, "RowParallelLinear", Linear)
    arch, weights, payload, manifest, _ = _fixture(tmp_path, base_model=base_model)
    embedder = h3.MiniMaxH3TimeEmbedder(arch, prefix="time_embedder")
    runtime = MiniMaxH3RuntimeAdalnCache()
    proj = h3.MiniMaxH3AdalnProj(
        arch,
        18 * arch.hidden_size,
        None,
        expand_ratio=6,
        modality_num=3,
        prefix="blocks.0.adaln_proj",
        adaln_cache=runtime,
    )
    embedder.load_state_dict(
        {
            name.removeprefix("time_embedder."): value
            for name, value in weights.items()
            if name.startswith("time_embedder.")
        }
    )
    proj.load_state_dict(
        {
            name.removeprefix("blocks.0.adaln_proj."): value
            for name, value in weights.items()
            if name.startswith("blocks.0.adaln_proj.")
        }
    )
    calls = mocker.spy(proj.linear, "forward")
    with torch.no_grad():
        for _ in range(2):
            for i, plan in enumerate(timestep_plans(manifest["schedule"])):
                embedding = embedder(plan)
                assert torch.equal(embedding, payload["time_embeddings"][i, : len(plan)])
                runtime.prepare(embedding)
                actual = torch.stack(proj(embedding), dim=1).reshape(len(plan), -1)
                assert torch.equal(actual, payload["block_params"][i, : len(plan), 0])
    plan_count = len(timestep_plans(manifest["schedule"]))
    assert calls.call_count == plan_count and runtime.hits == plan_count


def test_inference_weights_without_versions_use_safe_runtime_path(mocker):
    with torch.inference_mode():
        linear = torch.nn.Linear(2, 2)
        x = torch.ones(1, 2)
        cache = MiniMaxH3RuntimeAdalnCache()
        compute = mocker.Mock(side_effect=lambda: linear(x))
        cache.prepare(x)
        cache.project("a", linear, x, compute)
        linear.weight.add_(1)
        assert torch.equal(cache.project("a", linear, x, compute), linear(x))
        assert compute.call_count == 2 and not cache._entries


def test_corrupt_optional_sidecar_falls_back_without_rejecting_model(tmp_path):
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(b"truncated")
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(cache_config={"cache": str(path)})
    transformer = SimpleNamespace(arch=h3.MiniMaxH3DiTArchConfig(), adaln_cache=MiniMaxH3RuntimeAdalnCache())
    pipeline._configure_adaln_sidecar(transformer, "cache", "fl2va", None, eligible=True)
    assert not hasattr(transformer, "_adaln_sidecar_candidate")
