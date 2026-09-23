"""Experimental Ref2VA Turbo fusion into native H3 checkpoint tensors.

Used by the isolated numerical preflight and isolated serving backend.
Native QKV is grouped per head; native FC1 stores gate before value.
"""
from pathlib import Path

import torch
from safetensors import safe_open


class Ref2VATurboFusion:
    def __init__(self, path, *, device="cpu"):
        self.device = torch.device(device)
        self.factors = {}
        expected = set()
        for prefix, count in [("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2)]:
            for index in range(count):
                for target in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                               "ff.net.0.proj", "ff.net.2"]:
                    expected.add(f"{prefix}.{index}.{target}")
        with safe_open(str(Path(path)), framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if float(metadata.get("alpha", "nan")) != 8:
                raise ValueError("This Ref2VA experiment requires alpha=8")
            for name in checkpoint.keys():
                for side in ("A", "B"):
                    suffix = f".lora_{side}.default.weight"
                    if name.endswith(suffix):
                        target = name[:-len(suffix)]
                        if target not in expected:
                            raise ValueError(f"Unexpected Ref2VA target: {target}")
                        tensor = checkpoint.get_tensor(name)
                        if tensor.ndim != 2 or tensor.dtype != torch.bfloat16:
                            raise ValueError(f"Invalid factor: {name}")
                        if tensor.shape[0 if side == "A" else 1] != 128:
                            raise ValueError(f"Expected rank 128: {name}")
                        self.factors.setdefault(target, {})[side] = tensor
                        break
                else:
                    raise ValueError(f"Unconsumed Ref2VA tensor: {name}")
        if set(self.factors) != expected or any(set(v) != {"A", "B"} for v in self.factors.values()):
            raise ValueError("Incomplete Ref2VA adapter")
        self.scale = 8 / 128
        self.applied = set()

    def _product(self, raw_target):
        pair = self.factors[raw_target]
        a = pair["A"].to(self.device).float()
        b = pair["B"].to(self.device).float()
        return (b @ a).mul_(self.scale)

    def delta(self, native_name):
        if native_name.startswith("blocks."):
            prefix, index, rest = native_name.split(".", 2)
            raw_prefix = f"transformer_blocks.{index}"
        elif native_name.startswith("token_refiner.blocks."):
            _, _, index, rest = native_name.split(".", 3)
            raw_prefix = f"token_refiner.refiner_blocks.{index}"
        else:
            return None
        if rest == "attn.qkv_proj.weight":
            parts = [self._product(raw_prefix + ".attn.to_" + letter)
                     for letter in ("q", "k", "v")]
            # [head, q/k/v, head_channel, input_channel], matching native files.
            result = torch.stack([p.reshape(56, 128, 5376) for p in parts], dim=1)
            return result.reshape(21504, 5376)
        target = {"attn.out_proj.weight": "attn.to_out.0",
                  "mlp.fc1.weight": "ff.net.0.proj",
                  "mlp.fc2.weight": "ff.net.2"}.get(rest)
        if target is None:
            return None
        result = self._product(raw_prefix + "." + target)
        if rest == "mlp.fc1.weight":
            value, gate = result.chunk(2, dim=0)
            result = torch.cat((gate, value), dim=0)
        return result

    def fuse(self, native_name, weight):
        delta = self.delta(native_name)
        if delta is None:
            return weight
        if tuple(delta.shape) != tuple(weight.shape):
            raise ValueError(f"Ref2VA shape mismatch for {native_name}")
        if native_name in self.applied:
            raise ValueError(f"Ref2VA target applied twice: {native_name}")
        self.applied.add(native_name)
        return delta.add_(weight.to(self.device)).to(weight.dtype)

    def apply(self, weights):
        for name, weight in weights:
            yield name, self.fuse(name, weight)

    def validate_complete(self):
        if len(self.applied) != 208:
            raise ValueError(f"Expected 208 fused native parameters, got {len(self.applied)}")
