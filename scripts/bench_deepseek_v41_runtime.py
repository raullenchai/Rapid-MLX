#!/usr/bin/env python3
"""Profile the standalone DeepSeek V4.1 Flash MLX runtime with real weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

try:
    from scripts.bench_metadata import format_bench_json, write_bench_json
except ImportError:  # direct `python scripts/profile_*.py` execution
    from bench_metadata import format_bench_json, write_bench_json


PLAN = {
    "scope": "real-weight standalone DeepSeek V4.1 Flash text-runtime profile",
    "baseline": "unchanged checkpoint-bundled runtime",
    "measurements": [
        "load and sequential context-build wall time",
        "warm decode token latency and throughput",
        "existing mx.eval barrier wait time by Python call site",
        "runtime disk-read and Engram-cache counters",
        "MLX active, cache, and peak memory",
        "optional fused mHC mixing with V4.1 pipelined-pre semantics",
        "optional expert-local fused gate/up quantized matmul execution",
    ],
    "guardrails": [
        "never downloads or relocates model files",
        "requires explicit opt-in before executing checkpoint-bundled Python",
        "barrier wait is not kernel attribution",
        "text-only; vision and MTP are excluded",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--execute-real-weights", action="store_true")
    parser.add_argument("--trust-checkpoint-runtime", action="store_true")
    parser.add_argument("--resident-backbone", action="store_true")
    parser.add_argument("--engram-cache-rows", type=int, default=16384)
    parser.add_argument("--execution-mode", choices=("reference", "deferred", "compiled"), default="compiled")
    parser.add_argument(
        "--optimized-hc",
        action="store_true",
        help="benchmark-only fused mHC path; not quality validated",
    )
    parser.add_argument(
        "--optimized-moe",
        action="store_true",
        help="fuse resident routed-expert gate/up projections before profiling",
    )
    parser.add_argument("--prompt", default="Explain why local inference latency matters.")
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--measure-tokens", type=int, default=128)
    parser.add_argument("--diagnostic-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _checkpoint_total_size(model_path):
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    size = (index.get("metadata") or {}).get("total_size")
    return int(size) if size is not None else None


def _resolve_runtime(model_path):
    candidates = (model_path / "runtime" / "runtime.py", model_path / "runtime.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "checkpoint has no runtime/runtime.py or runtime.py; no fallback download is allowed"
    )


def _load_runtime(runtime_path):
    spec = importlib.util.spec_from_file_location(
        "rapid_deepseek_v41_checkpoint_runtime", runtime_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import checkpoint runtime: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvalBarrierProfiler:
    """Measure waits at existing eval barriers without adding new barriers."""

    def __init__(self, original):
        self.original = original
        self.reset()

    def reset(self):
        self.calls = 0
        self.seconds = 0.0
        self.by_caller = defaultdict(lambda: {"calls": 0, "seconds": 0.0})

    def __call__(self, *args, **kwargs):
        frame = sys._getframe(1)
        caller = f"{frame.f_code.co_name}:{frame.f_lineno}"
        started = time.perf_counter()
        result = self.original(*args, **kwargs)
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.seconds += elapsed
        bucket = self.by_caller[caller]
        bucket["calls"] += 1
        bucket["seconds"] += elapsed
        return result

    def snapshot(self):
        return {
            "calls": self.calls,
            "seconds": self.seconds,
            "by_caller": dict(
                sorted(
                    self.by_caller.items(),
                    key=lambda item: item[1]["seconds"],
                    reverse=True,
                )
            ),
            "interpretation": (
                "Time is charged to the existing barrier that waited for queued work; "
                "it is not exclusive time for the caller."
            ),
        }


def _weight_counters(runtime):
    weights = runtime.w
    return {
        "disk_bytes_read": int(getattr(weights, "disk_bytes_read", 0)),
        "disk_read_calls": int(getattr(weights, "disk_read_calls", 0)),
        "engram_cache_hits": int(getattr(weights, "engram_cache_hits", 0)),
        "engram_cache_misses": int(getattr(weights, "engram_cache_misses", 0)),
        "resident_bytes": int(getattr(weights, "resident_bytes", 0)),
    }


def _counter_delta(after, before):
    return {key: after[key] - before[key] for key in after}


def _memory_snapshot(mx):
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _make_v41_hc_mix_kernel(mx):
    """Build a V4.1 mixing kernel without collapsing the current residual.

    V4.1 pipelines the ``pre`` weights across the intervening attention or FFN
    update.  The existing V4 kernel cannot be reused directly because it also
    collapses the residual supplied to the mixer.  This bounded benchmark
    kernel fuses only sigmoid + Sinkhorn and returns ``pre/post/comb`` so the
    checkpoint runtime can apply ``pre`` at its original, later point.
    """

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("--optimized-hc requires the MLX Metal GPU backend")

    source = r"""
        uint row  = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr float EPS = EPS_INT * 1e-9;

        const device float* mix = (const device float*)mixes + row * MIX;
        device float* pre_out = (device float*)pre + row * HC;
        device float* post_out = (device float*)post + row * HC;
        device float* comb_out = (device float*)comb + row * HC * HC;

        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint llane = metal::min(lane, (uint)(HC - 1));
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        float pre_z = mix[llane] * pre_scale + base[llane];
        float post_z = mix[HC + llane] * post_scale + base[HC + llane];
        float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
        if (lane < (uint)HC) {
            pre_out[lane] = pre_v;
            post_out[lane] = post_v;
        }

        float4 v = (*(const device float4*)(mix + BASE_OFF + llane * HC)
                        * comb_scale
                    + *(const device float4*)(base + BASE_OFF + llane * HC))
                    * active;
        float row_max = metal::max(metal::max(v.x, v.y),
                                   metal::max(v.z, v.w));
        float4 e = metal::fast::exp(v - row_max) * active;
        float4 result = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                      + EPS * active;
        result *= 1.0f / (float4(
            simd_sum(result.x), simd_sum(result.y),
            simd_sum(result.z), simd_sum(result.w)) + EPS);

        for (int iter = 1; iter < ITERS; ++iter) {
            result *= (1.0f / (result.x + result.y + result.z
                               + result.w + EPS)) * active;
            result *= 1.0f / (float4(
                simd_sum(result.x), simd_sum(result.y),
                simd_sum(result.z), simd_sum(result.w)) + EPS);
        }
        if (lane < (uint)HC) {
            *(device float4*)(comb_out + lane * HC) = result;
        }
    """
    return mx.fast.metal_kernel(
        name="deepseek_v41_hc_mix",
        input_names=["mixes", "scale", "base"],
        output_names=["pre", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )


def _install_optimized_hc(runtime, mx):
    """Replace only checkpoint mHC mixing while preserving V4.1 timing."""

    if runtime.c.get("hc_mult") != 4:
        raise RuntimeError("--optimized-hc currently requires hc_mult=4")
    kernel = _make_v41_hc_mix_kernel(mx)

    @mx.compile
    def project(x, weight, norm_eps):
        flat = x.reshape(*x.shape[:-2], -1).astype(mx.float32)
        projected = flat @ weight.T
        return projected * mx.rsqrt(
            mx.mean(flat * flat, axis=-1, keepdims=True) + norm_eps
        )

    def mixes(self, base, kind, x):
        config = self.c
        weight = self.w.read(base + ".hc_" + kind + "_fn")
        scale = self.w.read(base + ".hc_" + kind + "_scale").astype(mx.float32)
        bias = self.w.read(base + ".hc_" + kind + "_base").astype(mx.float32)
        projected = project(x, weight, config["rms_norm_eps"])
        rows = math.prod(projected.shape[:-1])
        return kernel(
            inputs=[projected, scale, bias],
            template=[
                ("HC", config["hc_mult"]),
                ("ITERS", config["hc_sinkhorn_iters"]),
                ("EPS_INT", round(config["hc_eps"] / 1e-9)),
            ],
            grid=(rows * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (*projected.shape[:-1], config["hc_mult"]),
                (*projected.shape[:-1], config["hc_mult"]),
                (
                    *projected.shape[:-1],
                    config["hc_mult"],
                    config["hc_mult"],
                ),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )

    runtime.mixes = types.MethodType(mixes, runtime)
    return {
        "name": "v41_fused_hc_mix",
        "preserves_pipelined_pre": True,
        "quality_validated": False,
        "warning": "40-layer numerical amplification failed teacher-forced parity",
    }


def _install_optimized_moe(runtime, mx):
    """Fuse each expert's gate/up projections without giant expert buffers."""

    if not runtime.w.resident:
        raise RuntimeError("--optimized-moe requires --resident-backbone")
    config = runtime.c
    num_experts = config["n_routed_experts"]
    quantization = runtime.w.q
    if (
        quantization.get("bits"),
        quantization.get("group_size"),
        quantization.get("mode"),
    ) != (2, 64, "affine"):
        raise RuntimeError(
            "--optimized-moe currently accepts only affine 2-bit/group-64 weights"
        )

    started = time.perf_counter()
    packed = {}
    components = ("weight", "scales", "biases")
    required = [
        f"layers.{layer}.ffn.experts.{expert}.{projection}.{component}"
        for layer in range(config["num_hidden_layers"])
        for expert in range(num_experts)
        for projection in ("w1", "w3")
        for component in components
    ]
    missing = [key for key in required if key not in runtime.w.resident]
    if missing:
        raise RuntimeError(
            "--optimized-moe requires a fully resident backbone; missing "
            + missing[0]
        )
    for layer in range(config["num_hidden_layers"]):
        base = f"layers.{layer}.ffn"
        layer_pack = []
        original_keys = []
        fused_arrays = []
        for expert in range(num_experts):
            expert_pack = {}
            for component in components:
                w1_key = f"{base}.experts.{expert}.w1.{component}"
                w3_key = f"{base}.experts.{expert}.w3.{component}"
                fused = mx.concatenate(
                    [runtime.w.resident[w1_key], runtime.w.resident[w3_key]],
                    axis=0,
                )
                expert_pack[component] = fused
                fused_arrays.append(fused)
                original_keys.extend((w1_key, w3_key))
            layer_pack.append(expert_pack)
        mx.eval(fused_arrays)
        for key in original_keys:
            del runtime.w.resident[key]
        packed[base] = layer_pack
        mx.clear_cache()
        if (layer + 1) % 5 == 0 or layer + 1 == config["num_hidden_layers"]:
            print(
                json.dumps(
                    {
                        "phase": "expert_pack",
                        "completed_layers": layer + 1,
                        "total_layers": config["num_hidden_layers"],
                        "active_bytes": int(mx.get_active_memory()),
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    def moe(self, base, x):
        layer_pack = packed[base]
        logits = (
            x.astype(mx.float32)
            @ self.w.read(base + ".gate.weight").astype(mx.float32).T
        )
        scores = mx.sqrt(mx.logaddexp(logits, mx.zeros_like(logits)))
        top_k = self.c["num_experts_per_tok"]
        picks = mx.argsort(
            scores + self.w.read(base + ".gate.bias"), axis=-1
        )[..., -top_k:]
        selected = mx.take_along_axis(scores, picks, axis=-1)
        if self.c["norm_topk_prob"] and top_k > 1:
            selected = selected / (
                mx.sum(selected, axis=-1, keepdims=True) + 1e-20
            )
        selected = selected * self.c["routed_scaling_factor"]
        mx.eval(picks, selected)

        routed = mx.zeros_like(x).astype(mx.float32)
        for expert, routing in zip(picks.tolist()[0], selected.tolist()[0]):
            expert_pack = layer_pack[expert]
            gate_up = mx.quantized_matmul(
                x,
                expert_pack["weight"],
                expert_pack["scales"],
                expert_pack["biases"],
                transpose=True,
                group_size=64,
                bits=2,
                mode="affine",
            )
            gate, up = mx.split(gate_up, 2, axis=-1)
            limit = self.c["swiglu_limit"]
            if limit > 0:
                gate = mx.minimum(gate.astype(mx.float32), limit)
                up = mx.clip(up.astype(mx.float32), -limit, limit)
            else:
                gate = gate.astype(mx.float32)
                up = up.astype(mx.float32)
            hidden = (gate * mx.sigmoid(gate) * up * routing).astype(x.dtype)
            down = self.w.linear(
                base + f".experts.{expert}.w2", hidden
            ).astype(mx.float32)
            routed = routed + down
            if self.execution_mode == "reference":
                mx.eval(routed)
        shared = self.expert(base + ".shared_experts", x).astype(mx.float32)
        return (routed + shared).astype(x.dtype)

    runtime._rapid_packed_experts = packed
    runtime.moe = types.MethodType(moe, runtime)
    packed_bytes = sum(
        value.nbytes
        for layer_pack in packed.values()
        for expert_pack in layer_pack
        for value in expert_pack.values()
    )
    runtime.w.resident_bytes = sum(
        value.nbytes for value in runtime.w.resident.values()
    ) + packed_bytes
    return {
        "name": "expert_local_fused_gate_up_quantized_matmul",
        "layers": len(packed),
        "packed_bytes": packed_bytes,
        "seconds": time.perf_counter() - started,
    }


def _run_tokens(
    runtime,
    count,
    barrier,
    mx,
    *,
    initial_token,
    teacher_tokens=None,
):
    latencies = []
    counters_before = _weight_counters(runtime)
    if barrier is not None:
        barrier.reset()
    next_token = int(initial_token)
    started = time.perf_counter()
    for index in range(count):
        token = teacher_tokens[index] if teacher_tokens is not None else next_token
        token_started = time.perf_counter()
        logits, _ = runtime.step(int(token))
        next_token = int(mx.argmax(logits).item())
        latencies.append(time.perf_counter() - token_started)
    elapsed = time.perf_counter() - started
    return {
        "tokens": count,
        "seconds": elapsed,
        "tokens_per_second": count / elapsed if elapsed else None,
        "latency_seconds": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
            "raw": latencies,
        },
        "eval_barriers": barrier.snapshot() if barrier is not None else None,
        "weight_counters": _counter_delta(_weight_counters(runtime), counters_before),
        "memory": _memory_snapshot(mx),
        "last_token_id": next_token,
    }


def _validate_args(args):
    if args.model is None or not args.model.is_dir():
        raise SystemExit("--model must name an existing local checkpoint directory")
    if not args.trust_checkpoint_runtime:
        raise SystemExit(
            "refusing to execute checkpoint-bundled Python without "
            "--trust-checkpoint-runtime"
        )
    if min(
        args.context_tokens,
        args.warmup_tokens,
        args.measure_tokens,
        args.diagnostic_tokens,
    ) < 0:
        raise SystemExit("token counts may not be negative")
    if args.measure_tokens < 1:
        raise SystemExit("--measure-tokens must be at least 1")
    if args.diagnostic_tokens < 1:
        raise SystemExit("--diagnostic-tokens must be at least 1")
    if not 0 <= args.engram_cache_rows <= 16384:
        raise SystemExit("--engram-cache-rows must be between 0 and 16384")


def run(args):
    _validate_args(args)
    model_path = args.model.expanduser().resolve()
    runtime_path = _resolve_runtime(model_path)
    total_size = _checkpoint_total_size(model_path)

    started = time.perf_counter()
    module = _load_runtime(runtime_path)
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    mx.reset_peak_memory()
    runtime = module.TextRuntime(
        model_path,
        max_tokens=(
            args.context_tokens
            + args.warmup_tokens
            + args.measure_tokens
            + args.diagnostic_tokens
            + 8
        ),
        resident_backbone=args.resident_backbone,
        execution_mode=args.execution_mode,
    )
    runtime.w.engram_cache_rows = args.engram_cache_rows
    optimized_hc = None
    if getattr(args, "optimized_hc", False):
        optimized_hc = _install_optimized_hc(runtime, mx)
    optimized_moe = None
    if getattr(args, "optimized_moe", False):
        optimized_moe = _install_optimized_moe(runtime, mx)
    load_seconds = time.perf_counter() - started
    encoded = runtime.tokenizer.encode(args.prompt).ids
    if not encoded:
        raise RuntimeError("prompt encoded to zero tokens")
    original_eval = module.mx.eval
    barrier = EvalBarrierProfiler(original_eval)
    load_memory = _memory_snapshot(mx)
    mx.reset_peak_memory()

    context = [encoded[index % len(encoded)] for index in range(args.context_tokens)]
    try:
        context_result = _run_tokens(
            runtime,
            len(context),
            None,
            mx,
            initial_token=encoded[-1],
            teacher_tokens=context,
        )
        warmup_result = _run_tokens(
            runtime,
            args.warmup_tokens,
            None,
            mx,
            initial_token=context_result["last_token_id"],
        )
        measured = _run_tokens(
            runtime,
            args.measure_tokens,
            None,
            mx,
            initial_token=warmup_result["last_token_id"],
        )
        module.mx.eval = barrier
        diagnostic = _run_tokens(
            runtime,
            args.diagnostic_tokens,
            barrier,
            mx,
            initial_token=measured["last_token_id"],
        )
        module.mx.eval = original_eval
    finally:
        module.mx.eval = original_eval

    return {
        "plan": PLAN,
        "checkpoint": {
            "path": str(model_path),
            "runtime_path": str(runtime_path),
            "indexed_total_size_bytes": total_size,
            "runtime_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        },
        "parameters": {
            "execution_mode": args.execution_mode,
            "resident_backbone": args.resident_backbone,
            "engram_cache_rows": args.engram_cache_rows,
            "context_tokens": args.context_tokens,
            "warmup_tokens": args.warmup_tokens,
            "measure_tokens": args.measure_tokens,
            "diagnostic_tokens": args.diagnostic_tokens,
            "optimized_hc": bool(getattr(args, "optimized_hc", False)),
            "optimized_moe": bool(getattr(args, "optimized_moe", False)),
        },
        "runtime_overrides": {
            "hyper_connection": optimized_hc,
            "routed_moe": optimized_moe,
        },
        "device": mx.device_info(),
        "load": {"seconds": load_seconds, "memory": load_memory},
        "context_build": context_result,
        "warmup": warmup_result,
        "measurement": measured,
        "barrier_diagnostic": diagnostic,
    }


def main() -> int:
    args = parse_args()
    if not args.execute_real_weights:
        print(format_bench_json({"plan_only": True, "plan": PLAN}, __file__))
        return 0
    result = run(args)
    print(format_bench_json(result, __file__, indent=2, sort_keys=True))
    if args.output is not None:
        write_bench_json(args.output, result, __file__, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
