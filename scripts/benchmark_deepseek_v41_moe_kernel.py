#!/usr/bin/env python3
"""Microbenchmark one real V4.1 affine-2bit MoE layer on stock vs native QMM."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deepseek_v41_affine_route_qmv import affine2_route_down_qmv  # noqa: E402
from deepseek_v41_native.config import ModelArgs  # noqa: E402
from deepseek_v41_native.moe import MoE  # noqa: E402


def _load_prefix_items(model_path: Path, index: dict, prefix: str):
    shards = sorted({shard for key, shard in index.items() if key.startswith(prefix)})
    if not shards:
        raise ValueError(f"no checkpoint tensors found for {prefix}")
    items = {}
    for shard in shards:
        if not isinstance(shard, str) or not shard:
            raise ValueError(f"invalid checkpoint shard name: {shard!r}")
        shard_name = Path(shard)
        if shard_name.is_absolute() or shard_name.name != shard:
            raise ValueError(f"checkpoint shard must be a basename: {shard!r}")
        shard_path = model_path / shard
        if shard_path.is_symlink():
            raise ValueError(f"checkpoint shard must not be a symlink: {shard_path}")
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        if shard_path.resolve().parent != model_path.resolve():
            raise ValueError(f"checkpoint shard escapes model directory: {shard_path}")
        weights = mx.load(str(shard_path))
        for key, value in weights.items():
            if not key.startswith(prefix):
                continue
            name = key.removeprefix(prefix)
            if name in items:
                raise ValueError(f"duplicate tensor across checkpoint shards: {key}")
            items[name] = value
    return sorted(items.items())


class HybridPairSwitch(nn.Module):
    """Native affine gate+up pair, stock gather_qmm down projection."""

    def __init__(self, native, switch, block_rows, variant):
        super().__init__()
        self.gate_proj = native.gate_proj
        self.up_proj = native.up_proj
        self.down_proj = native.down_proj
        self.activation = native.activation
        self.switch = switch
        self.block_rows = block_rows
        self.variant = variant

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        x, idx, inv_order = self.switch._gather_sort(x, indices)
        block_meta, block_count = self.switch._build_mxfp4_blocks(
            idx, self.up_proj.num_experts, self.block_rows
        )
        pair = self.switch.glm_fast.deepseek_affine_gather_qmm_pair_concat_blocks(
            x,
            self.up_proj.weight,
            self.up_proj.scales,
            self.up_proj.biases,
            self.gate_proj.weight,
            self.gate_proj.scales,
            self.gate_proj.biases,
            block_meta,
            block_count,
            self.up_proj.group_size,
            self.up_proj.bits,
            self.variant,
        )
        hidden = self.up_proj.output_dims
        x = self.activation(pair[..., :hidden], pair[..., hidden:])
        x = mx.gather_qmm(
            x,
            self.down_proj.weight,
            self.down_proj.scales,
            self.down_proj.biases,
            rhs_indices=idx,
            transpose=True,
            group_size=self.down_proj.group_size,
            bits=self.down_proj.bits,
            mode=self.down_proj.mode,
            sorted_indices=True,
        )
        x = self.switch._scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class StockPairDirectDown(nn.Module):
    """Preserve stock gate/up numerics and specialize the exact down QMV."""

    def __init__(self, native):
        super().__init__()
        self.gate_proj = native.gate_proj
        self.up_proj = native.up_proj
        self.down_proj = native.down_proj
        self.activation = native.activation

    def __call__(self, x, indices):
        if int(x.shape[0]) * int(indices.shape[-1]) > 36:
            expanded = mx.expand_dims(x, (-2, -3))
            up = self.up_proj(expanded, indices)
            gate = self.gate_proj(expanded, indices)
            return self.down_proj(self.activation(up, gate), indices).squeeze(-2)
        expanded = mx.expand_dims(x, (-2, -3))
        up = self.up_proj(expanded, indices).squeeze(-2)
        gate = self.gate_proj(expanded, indices).squeeze(-2)
        activated = self.activation(up, gate)
        tokens, topk, width = map(int, activated.shape)
        down = affine2_route_down_qmv(
            self.down_proj,
            activated.reshape(tokens * topk, width),
            indices.reshape(tokens * topk, 1),
        )
        return down.reshape(tokens, topk, -1)


def _load_layer(model_path: Path, layer_id: int):
    config = json.loads((model_path / "config.json").read_text())
    args = ModelArgs.from_dict(config)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"layers.{layer_id}.ffn."
    items = _load_prefix_items(model_path, index, prefix)

    moe = MoE(args)
    quantized = {
        "experts.gate_proj",
        "experts.up_proj",
        "experts.down_proj",
        "shared_experts.w1",
        "shared_experts.w2",
        "shared_experts.w3",
    }
    nn.quantize(
        moe,
        group_size=64,
        bits=2,
        class_predicate=lambda path, _module: path in quantized,
    )
    moe.load_weights(items, strict=True)
    mx.eval(moe.parameters())
    return moe, args


def _native_experts(stock, args, source_root: Path):
    os.environ["OMLX_DEEPSEEK_SORT_MIN_ROUTES"] = "1"
    os.environ["OMLX_DEEPSEEK_AFFINE_BLOCK_MIN_ROUTES"] = "1"
    sys.path.insert(0, str(source_root))
    switch = importlib.import_module("omlx.patches.deepseek_v4.switch_layers")
    native = switch.SwitchGLU(
        args.dim,
        args.moe_inter_dim,
        args.n_routed_experts,
        activation=stock.activation,
        bias=False,
    )
    nn.quantize(native, group_size=64, bits=2)
    for name in ("gate_proj", "up_proj", "down_proj"):
        old = getattr(stock, name)
        new = getattr(native, name)
        new.weight = old.weight
        new.scales = old.scales
        new.biases = old.biases
    return native, switch


def _measure(moe, values, iterations):
    for _ in range(3):
        output = moe(values)
        mx.eval(output)
    started = time.perf_counter()
    for _ in range(iterations):
        output = moe(values)
        mx.eval(output)
    return output, (time.perf_counter() - started) / iterations


def _stock_experts(native, config):
    stock = importlib.import_module("mlx_lm.models.switch_layers").SwitchGLU(
        config.dim,
        config.moe_inter_dim,
        config.n_routed_experts,
        activation=native.activation,
        bias=False,
    )
    nn.quantize(stock, group_size=64, bits=2)
    for name in ("gate_proj", "up_proj", "down_proj"):
        source = getattr(native, name)
        target = getattr(stock, name)
        target.weight = source.weight
        target.scales = source.scales
        target.biases = source.biases
    return stock


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--omlx-source", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    moe, config = _load_layer(args.model.resolve(), args.layer)
    native, switch = _native_experts(moe.experts, config, args.omlx_source.resolve())
    print(
        json.dumps(
            {
                "event": "loaded",
                "layer": args.layer,
                "active_gb": mx.get_active_memory() / 1e9,
                "native_affine_symbol": switch.glm_fast.has_symbol(
                    "deepseek_affine_gather_qmm_blocks"
                ),
                "native_pair_symbol": switch.glm_fast.has_symbol(
                    "deepseek_affine_gather_qmm_pair_concat_blocks"
                ),
            }
        ),
        flush=True,
    )
    rng = mx.random.key(42)
    for tokens in (1, 4, 5, 6):
        values = mx.random.normal((1, tokens, config.dim), key=rng).astype(mx.bfloat16)
        moe.experts = _stock_experts(native, config)
        stock_output, stock_seconds = _measure(moe, values, args.iterations)
        for variant, experts in (
            ("native_all", native),
            ("hybrid_pair_bm16", HybridPairSwitch(native, switch, 16, 1)),
            ("hybrid_pair_bm32", HybridPairSwitch(native, switch, 32, 2)),
            ("stock_pair_direct_down", StockPairDirectDown(native)),
        ):
            moe.experts = experts
            output, seconds = _measure(moe, values, args.iterations)
            print(
                json.dumps(
                    {
                        "event": "moe",
                        "variant": variant,
                        "tokens": tokens,
                        "routes": tokens * config.n_activated_experts,
                        "stock_ms": stock_seconds * 1000,
                        "candidate_ms": seconds * 1000,
                        "speedup": stock_seconds / seconds,
                        "allclose": mx.allclose(
                            stock_output, output, rtol=2e-3, atol=2e-3
                        ).item(),
                        "max_abs_diff": mx.max(
                            mx.abs(
                                stock_output.astype(mx.float32)
                                - output.astype(mx.float32)
                            )
                        ).item(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
