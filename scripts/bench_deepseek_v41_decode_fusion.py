#!/usr/bin/env python3
"""Compare native V4.1 decode graphs and synchronized single-stream latency."""

from __future__ import annotations

import argparse
import contextlib
import io
import platform
import re
import sys
import time
from collections import Counter
from unittest.mock import patch

import mlx.core as mx

from rapid_mlx.models.deepseek_v41_native import fusion
from rapid_mlx.models.deepseek_v41_native.attention import Attention
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
from rapid_mlx.models.deepseek_v41_native.model import Block, Model
from rapid_mlx.models.deepseek_v41_native.moe import MoE

try:
    from scripts.bench_metadata import format_bench_json
except ModuleNotFoundError:  # Direct script execution.
    from bench_metadata import format_bench_json


def make_tiny_model() -> Model:
    """Exercise window, ratio-2/1 compression, index ownership and sharing."""
    mx.random.seed(41)
    model = Model(
        ModelArgs(
            vocab_size=64,
            dim=64,
            n_layers=4,
            moe_inter_dim=32,
            n_heads=2,
            head_dim=32,
            rope_head_dim=8,
            q_lora_rank=32,
            o_lora_rank=16,
            o_groups=2,
            window_size=8,
            n_routed_experts=8,
            n_activated_experts=2,
            compress_ratios=(0, 2, 2, 1),
            kv_source_layers=(1, 3),
            index_source_layers=(1, 3),
            index_n_heads=2,
            index_head_dim=32,
            index_topk=4,
            original_seq_len=0,
            max_seq_len=128,
        )
    )
    # Nonzero routing / mixing parameters make the numerical check meaningful.
    for layer in model.layers:
        layer.ffn.gate.weight = mx.random.normal(layer.ffn.gate.weight.shape) * 0.1
        layer.ffn.gate.bias = mx.random.normal(layer.ffn.gate.bias.shape) * 0.01
        layer.hc_attn_fn = mx.random.normal(layer.hc_attn_fn.shape) * 0.01
        layer.hc_ffn_fn = mx.random.normal(layer.hc_ffn_fn.shape) * 0.01
        layer.hc_attn_scale = mx.ones((3,)) * 0.1
        layer.hc_ffn_scale = mx.ones((3,)) * 0.1
    mx.eval(model.parameters())
    return model


def graph_operations(*outputs) -> dict[str, str]:
    """Count operation rectangles, excluding array vertices and evaluated leaves."""
    stream = io.StringIO()
    mx.export_to_dot(stream, *outputs)
    return dict(
        re.findall(r'(\d+) \[label ="([^"]*)", shape=rectangle\]', stream.getvalue())
    )


def graph_step(model: Model, prompt_length: int = 8) -> dict:
    cache = model.make_cache(max_seq_len=prompt_length + 4)
    mx.eval(
        model(
            mx.array([[i % model.args.vocab_size for i in range(prompt_length)]]), cache
        )
    )
    # Cache writes and frequency tables must be leaves rather than prefill graphs.
    for layer in cache.layers:
        mx.eval(vars(layer))
        if layer.comp_state is not None:
            mx.eval(vars(layer.comp_state))
    for layer in model.layers:
        mx.eval(layer.attn._cos, layer.attn._sin)
    layers = []
    previous: set[str] = set()
    original = Block.__call__

    def capture(block, *args, **kwargs):
        result = original(block, *args, **kwargs)
        operations = graph_operations(*result)
        fresh = {key: value for key, value in operations.items() if key not in previous}
        layers.append(
            {
                "layer": block.layer_id,
                "nodes": len(fresh),
                "types": dict(Counter(fresh.values())),
            }
        )
        previous.update(operations)
        return result

    interval = model.eval_interval
    model.eval_interval = 0
    try:
        with patch.object(Block, "__call__", capture):
            output = model(mx.array([[1]]), cache)
        operations = graph_operations(output)
        mx.eval(output)
    finally:
        model.eval_interval = interval
    return {
        "nodes": len(operations),
        "types": dict(Counter(operations.values())),
        "layers": layers,
    }


def timed_steps(model: Model, steps: int, prompt_length: int) -> dict:
    """Uninstrumented step time, then a separate serialized attribution pass."""

    def run(profile):
        cache = model.make_cache(max_seq_len=prompt_length + steps + 4)
        token = mx.array([[1]])
        mx.eval(model(mx.ones((1, prompt_length), mx.int32), cache))
        totals = {"attention": 0.0, "moe": 0.0}
        per_layer = [{"attention": 0.0, "moe": 0.0, "total": 0.0} for _ in model.layers]
        layer_ids = {id(layer.attn): i for i, layer in enumerate(model.layers)}
        layer_ids.update({id(layer.ffn): i for i, layer in enumerate(model.layers)})

        original_block = Block.__call__

        def measured_block(block, *args, **kwargs):
            start = time.perf_counter()
            result = original_block(block, *args, **kwargs)
            mx.eval(result)
            per_layer[block.layer_id]["total"] += (
                (time.perf_counter() - start) * 1000 / steps
            )
            return result

        def measured(original, category):
            def call(module, *args, **kwargs):
                start = time.perf_counter()
                result = original(module, *args, **kwargs)
                mx.eval(result)
                elapsed = (time.perf_counter() - start) * 1000 / steps
                totals[category] += elapsed
                per_layer[layer_ids[id(module)]][category] += elapsed
                return result

            return call

        with contextlib.ExitStack() as stack:
            if profile:
                stack.enter_context(patch.object(Block, "__call__", measured_block))
                stack.enter_context(
                    patch.object(
                        Attention, "__call__", measured(Attention.__call__, "attention")
                    )
                )
                stack.enter_context(
                    patch.object(MoE, "__call__", measured(MoE.__call__, "moe"))
                )
            start = time.perf_counter()
            for _ in range(steps):
                logits = model(token, cache, last_logit_only=True)
                token = mx.argmax(logits[:, -1], axis=-1)[:, None]
                mx.eval(token)
            elapsed = (time.perf_counter() - start) * 1000 / steps
        for row in per_layer:
            row["other"] = row["total"] - row["attention"] - row["moe"]
        return elapsed, totals, per_layer

    run(False)  # warm traces before measuring
    elapsed, _, _ = run(False)
    serialized, totals, per_layer = run(True)
    return {
        "step_ms": elapsed,
        "serialized_step_ms": serialized,
        "serialized_split_ms": {**totals, "other": serialized - sum(totals.values())},
        "per_layer_ms": per_layer,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local native checkpoint directory")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--prompt-length", type=int, default=8)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.prompt_length < 1:
        parser.error("steps and prompt-length must be positive")
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    if args.model:
        from rapid_mlx.models.deepseek_v41_native.load import load

        with contextlib.redirect_stdout(sys.stderr):
            model, _ = load(args.model)
    else:
        model = make_tiny_model()
    results = {}
    enabled = fusion.ENABLED
    try:
        for mode in (False, True):
            fusion.ENABLED = mode
            result = graph_step(model, args.prompt_length)
            if args.timing:
                result["timing"] = timed_steps(model, args.steps, args.prompt_length)
            results["on" if mode else "off"] = result
    finally:
        fusion.ENABLED = enabled
    print(
        format_bench_json(
            {
                "device": args.device,
                "mlx_version": mx.__version__,
                "python_version": platform.python_version(),
                "layers": model.args.n_layers,
                "dim": model.args.dim,
                "steps": args.steps if args.timing else None,
                "synthetic": not bool(args.model),
                "prompt_length": args.prompt_length,
                "results": results,
            },
            __file__,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
