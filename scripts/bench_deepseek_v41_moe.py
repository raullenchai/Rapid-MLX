#!/usr/bin/env python3
"""Synthetic decode micro-benchmark for the DeepSeek V4.1 Flash MoE path.

This intentionally measures only the expert MLP portion of one decode layer.
It uses the published dimensions and quantization shape, but random weights;
the result is a kernel/scheduling diagnostic, not an end-to-end model claim.
"""

from __future__ import annotations

import argparse
import statistics
import time

try:
    from scripts.bench_metadata import format_bench_json, write_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json, write_bench_json


PLAN = {
    "scope": "one decode-token MoE MLP layer with synthetic resident weights",
    "model_shape": {
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
        "active_routed_experts": 6,
        "shared_experts": 1,
        "layers": 40,
    },
    "comparison": [
        "serial expert loop with a host synchronization after every expert",
        "serial expert loop with one synchronization at the layer boundary",
        "three batched quantized matmuls across all active experts",
        "the same batched path after stacking separately stored expert tensors",
    ],
    "excluded": [
        "router and expert-weight gathering",
        "attention and KV cache",
        "Engram lookup",
        "mHC residuals",
        "MTP speculative decoding",
        "model quality",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-metal", action="store_true")
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 5, 6, 8), default=2)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output")
    return parser.parse_args()


def _quantize_experts(mx, count, shape, *, bits, group_size, seed):
    experts = []
    for index in range(count):
        weights = mx.random.normal(shape, key=mx.random.key(seed + index)).astype(
            mx.bfloat16
        )
        quantized = mx.quantize(weights, bits=bits, group_size=group_size)
        mx.eval(*quantized)
        experts.append(quantized)
    return experts


def _stack_experts(mx, experts):
    stacked = tuple(
        mx.stack([expert[part] for expert in experts]) for part in range(3)
    )
    mx.eval(*stacked)
    return stacked


def _qmm(mx, x, packed, *, bits, group_size):
    weight, scales, biases = packed
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        bits=bits,
        group_size=group_size,
    )


def _silu(mx, x):
    return x * mx.sigmoid(x)


def run(args):
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    hidden_size = PLAN["model_shape"]["hidden_size"]
    intermediate_size = PLAN["model_shape"]["moe_intermediate_size"]
    expert_count = (
        PLAN["model_shape"]["active_routed_experts"]
        + PLAN["model_shape"]["shared_experts"]
    )

    # Only active experts are materialized. This isolates dispatch granularity
    # without allocating the full 384-expert checkpoint.
    split_gate = _quantize_experts(
        mx,
        expert_count,
        (intermediate_size, hidden_size),
        bits=args.bits,
        group_size=args.group_size,
        seed=4101,
    )
    split_up = _quantize_experts(
        mx,
        expert_count,
        (intermediate_size, hidden_size),
        bits=args.bits,
        group_size=args.group_size,
        seed=4201,
    )
    split_down = _quantize_experts(
        mx,
        expert_count,
        (hidden_size, intermediate_size),
        bits=args.bits,
        group_size=args.group_size,
        seed=4301,
    )
    gate = _stack_experts(mx, split_gate)
    up = _stack_experts(mx, split_up)
    down = _stack_experts(mx, split_down)
    x = mx.random.normal((hidden_size,), key=mx.random.key(4104)).astype(mx.bfloat16)
    routing = mx.softmax(
        mx.random.normal((expert_count,), key=mx.random.key(4105)).astype(mx.float32)
    ).astype(mx.bfloat16)
    mx.eval(x, routing)

    def one_expert(token_x, index):
        gated = _silu(
            mx,
            _qmm(
                mx,
                token_x,
                split_gate[index],
                bits=args.bits,
                group_size=args.group_size,
            ),
        )
        up_value = _qmm(
            mx,
            token_x,
            split_up[index],
            bits=args.bits,
            group_size=args.group_size,
        )
        return _qmm(
            mx,
            gated * up_value,
            split_down[index],
            bits=args.bits,
            group_size=args.group_size,
        )

    def serial_graph(token_x):
        output = mx.zeros((hidden_size,), dtype=mx.bfloat16)
        for index in range(expert_count):
            expert = one_expert(token_x, index)
            output = output + routing[index] * expert
        return output

    def serial_sync_each():
        output = mx.zeros((hidden_size,), dtype=mx.bfloat16)
        for index in range(expert_count):
            expert = one_expert(x, index)
            mx.eval(expert)
            output = output + routing[index] * expert
        mx.eval(output)
        return output

    def serial_layer_sync():
        output = serial_graph(x)
        mx.eval(output)
        return output

    def batched_graph(token_x):
        batch_x = mx.broadcast_to(token_x, (expert_count, 1, hidden_size))
        gated = _silu(
            mx,
            _qmm(
                mx,
                batch_x,
                gate,
                bits=args.bits,
                group_size=args.group_size,
            ),
        )
        up_value = _qmm(
            mx,
            batch_x,
            up,
            bits=args.bits,
            group_size=args.group_size,
        )
        expert_outputs = _qmm(
            mx,
            gated * up_value,
            down,
            bits=args.bits,
            group_size=args.group_size,
        )[:, 0, :]
        return mx.sum(expert_outputs * routing[:, None], axis=0)

    def batched_layer_sync():
        output = batched_graph(x)
        mx.eval(output)
        return output

    def batched_from_split_sync():
        # The community checkpoint stores every expert under a distinct tensor
        # key. A production batched path either pays this stack/copy cost or
        # converts the artifact to an expert-major packed layout once.
        stacked_gate = tuple(
            mx.stack([expert[part] for expert in split_gate]) for part in range(3)
        )
        stacked_up = tuple(
            mx.stack([expert[part] for expert in split_up]) for part in range(3)
        )
        stacked_down = tuple(
            mx.stack([expert[part] for expert in split_down]) for part in range(3)
        )
        batch_x = mx.broadcast_to(x, (expert_count, 1, hidden_size))
        gated = _silu(
            mx,
            _qmm(
                mx,
                batch_x,
                stacked_gate,
                bits=args.bits,
                group_size=args.group_size,
            ),
        )
        up_value = _qmm(
            mx,
            batch_x,
            stacked_up,
            bits=args.bits,
            group_size=args.group_size,
        )
        expert_outputs = _qmm(
            mx,
            gated * up_value,
            stacked_down,
            bits=args.bits,
            group_size=args.group_size,
        )[:, 0, :]
        output = mx.sum(expert_outputs * routing[:, None], axis=0)
        mx.eval(output)
        return output

    modes = {
        "serial_sync_each": serial_sync_each,
        "serial_layer_sync": serial_layer_sync,
        "batched_layer_sync": batched_layer_sync,
        "batched_from_split_sync": batched_from_split_sync,
    }
    reference = serial_layer_sync()
    correctness = {}
    for name, function in modes.items():
        candidate = function()
        delta = mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
        correctness[name] = {
            "max_abs": float(mx.max(delta).item()),
            "mean_abs": float(mx.mean(delta).item()),
        }

    for _ in range(args.warmup):
        for function in modes.values():
            function()

    timings = {name: [] for name in modes}
    names = tuple(modes)
    for repeat in range(args.repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            started = time.perf_counter()
            modes[name]()
            timings[name].append(time.perf_counter() - started)

    medians = {name: statistics.median(values) for name, values in timings.items()}
    baseline = medians["serial_sync_each"]

    token_modes = {}
    for graph_name, graph in (("serial", serial_graph), ("batched", batched_graph)):
        def layer_sync(graph=graph):
            state = x
            for _ in range(PLAN["model_shape"]["layers"]):
                state = graph(state)
                mx.eval(state)
            return state

        def token_sync(graph=graph):
            state = x
            for _ in range(PLAN["model_shape"]["layers"]):
                state = graph(state)
            mx.eval(state)
            return state

        token_modes[f"{graph_name}_sync_each_layer"] = layer_sync
        token_modes[f"{graph_name}_single_sync"] = token_sync

    # Multi-layer graphs are much larger, so cap this diagnostic while keeping
    # alternating order to limit thermal and ordering bias.
    token_timings = {name: [] for name in token_modes}
    token_names = tuple(token_modes)
    token_repeats = min(args.repeats, 12)
    for repeat in range(token_repeats):
        order = token_names if repeat % 2 == 0 else tuple(reversed(token_names))
        for name in order:
            started = time.perf_counter()
            token_modes[name]()
            token_timings[name].append(time.perf_counter() - started)
    token_medians = {
        name: statistics.median(values) for name, values in token_timings.items()
    }
    return {
        "plan": PLAN,
        "parameters": {
            "bits": args.bits,
            "group_size": args.group_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "device": mx.device_info(),
        "correctness": correctness,
        "timing": {
            "raw_seconds": timings,
            "median_layer_milliseconds": {
                name: seconds * 1000 for name, seconds in medians.items()
            },
            "speedup_vs_serial_sync_each": {
                name: baseline / seconds for name, seconds in medians.items()
            },
            "moe_only_40_layer_upper_bound_tps": {
                name: 1.0 / (seconds * PLAN["model_shape"]["layers"])
                for name, seconds in medians.items()
            },
            "token_graph_raw_seconds": token_timings,
            "token_graph_median_seconds": token_medians,
            "token_graph_moe_only_tps": {
                name: 1.0 / seconds for name, seconds in token_medians.items()
            },
        },
        "interpretation_guardrail": (
            "The TPS projection excludes non-MoE model work and is only an upper bound."
        ),
    }


def main() -> int:
    args = parse_args()
    if not args.execute_metal:
        print(format_bench_json({"plan_only": True, "plan": PLAN}, __file__))
        return 0
    result = run(args)
    payload = format_bench_json(result, __file__, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        write_bench_json(args.output, result, __file__, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
