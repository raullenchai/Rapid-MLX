#!/usr/bin/env python3
"""Single-stream greedy draft benchmark; tiny mode needs no checkpoint."""

from __future__ import annotations

import argparse
import importlib
import platform
import time
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx

from scripts.bench_metadata import format_bench_json


def tiny_setup(seed=7):
    """Deterministic real target and three-stage draft with synthetic weights."""
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.dspark import DSpark, DSparkWeights
    from rapid_mlx.models.deepseek_v41_native.model import Model

    mx.random.seed(seed)
    args = ModelArgs(
        vocab_size=32,
        dim=32,
        n_layers=3,
        n_heads=2,
        head_dim=32,
        rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_activated_experts=2,
        window_size=8,
        hc_sinkhorn_iters=3,
        swiglu_limit=10,
        dspark_target_layer_ids=(0, 1, 2),
    )
    model = Model(args)
    config = {
        "dspark_block_size": 5,
        "dspark_n_routed_experts": 4,
        "dspark_num_experts_per_tok": 2,
        "dspark_noise_token_id": 3,
        "dspark_target_layer_ids": [0, 1, 2],
        "hc_mult": 4,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-20,
        "qk_rope_head_dim": 8,
        "rope_theta": 10000,
        "num_attention_heads": 2,
        "head_dim": 32,
        "o_groups": 2,
        "sliding_window": 8,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10,
    }
    weights = DSparkWeights.__new__(DSparkWeights)
    weights._arrays = {}
    weights.q = {"bits": 4, "group_size": 32}
    weights.resident = {}
    weights.resident_bytes = 0

    def add(name, shape, scale=0.1):
        weights._arrays[name] = mx.random.normal(shape) * scale

    add("mtp.0.main_proj.weight", (32, 96))
    weights._arrays["mtp.0.main_norm.weight"] = mx.ones((32,))
    for stage in range(3):
        base = f"mtp.{stage}"
        for kind in ("attn", "ffn"):
            add(f"{base}.hc_{kind}_fn", (24, 128))
            add(f"{base}.hc_{kind}_scale", (3,))
            add(f"{base}.hc_{kind}_base", (24,))
            weights._arrays[f"{base}.{kind}_norm.weight"] = mx.ones((32,))
        for name, shape in {
            "wq_a": (32, 32),
            "wq_b": (64, 32),
            "wkv": (32, 32),
            "wo_a": (32, 32),
            "wo_b": (32, 32),
        }.items():
            add(f"{base}.attn.{name}.weight", shape)
        for name in ("q_norm", "kv_norm"):
            weights._arrays[f"{base}.attn.{name}.weight"] = mx.ones((32,))
        add(f"{base}.attn.attn_sink", (2,))
        add(f"{base}.ffn.gate.weight", (4, 32))
        add(f"{base}.ffn.gate.bias", (4,))
        for expert in ["shared_experts", *[f"experts.{i}" for i in range(4)]]:
            for projection in ("w1", "w2", "w3"):
                add(f"{base}.ffn.{expert}.{projection}.weight", (32, 32))
    weights._arrays["mtp.2.norm.weight"] = mx.ones((32,))
    add("mtp.2.markov_head.embed.weight", (32, 32))
    add("mtp.2.markov_head.head.weight", (32, 32))
    add("mtp.2.confidence_head.proj.weight", (1, 64))
    # Exercise the installed affine attention and packed expert replacements.
    for name in list(weights._arrays):
        if name.endswith(".weight") and (
            ".attn.wo_a." in name or ".ffn.experts." in name
        ):
            weight, scales, biases = mx.quantize(
                weights._arrays[name], group_size=32, bits=4
            )
            weights._arrays[name] = weight
            weights._arrays[name.removesuffix("weight") + "scales"] = scales
            weights._arrays[name.removesuffix("weight") + "biases"] = biases
    weights.entries = {
        name: {"shape": value.shape} for name, value in weights._arrays.items()
    }
    weights.attach_target(model)
    draft = DSpark(SimpleNamespace(w=weights, c=config), pin_weights=False)
    return model, draft


class TinyTokenizer:
    clean_up_tokenization_spaces = False
    eos_token_id = 31
    all_special_ids = [31]

    def encode(self, _text, **_kwargs):
        return [1, 2, 3, 4]

    def decode(self, ids, **_kwargs):
        return "".join(chr(65 + int(i)) for i in ids)


def plain_decode(model, prompt, count):
    cache = model.make_cache(max_seq_len=len(prompt) + count + 8)
    logits = model(mx.array([prompt]), cache)
    mx.eval(logits)
    start = time.perf_counter()
    output = []
    for _ in range(count):
        token = int(mx.argmax(logits[:, -1], axis=-1).item())
        output.append(token)
        logits = model(mx.array([[token]]), cache)
        mx.eval(logits)
    elapsed = time.perf_counter() - start
    return output, {"tokens_per_second": count / elapsed, "decode_seconds": elapsed}


def speculative_decode(serving, model, tokenizer, runtime, prompt, count):
    """Time production decode, excluding prefill and including streaming host work."""
    rounds = []
    original_propose = runtime.drafter.propose
    original_match = serving._match
    start = None

    def close_round():
        if rounds:
            rounds[-1]["round_ms"] = (time.perf_counter() - rounds[-1]["start"]) * 1000

    def propose(seed):
        nonlocal start
        close_round()
        now = time.perf_counter()
        if start is None:
            start = now
        row = {"start": now, "verify_ms": 0.0}
        rounds.append(row)
        result = original_propose(seed)
        mx.eval(result[1])
        row["draft_ms"] = (time.perf_counter() - now) * 1000
        return result

    def match(*args):
        result = original_match(*args)
        rounds[-1]["accepted"] = result[3]
        return result

    class TimedModel:
        def make_cache(self, **kwargs):
            return model.make_cache(**kwargs)

        def __call__(self, *args, **kwargs):
            now = time.perf_counter()
            result = model(*args, **kwargs)
            mx.eval(result)
            if kwargs.get("enable_rollback"):
                rounds[-1]["verify_ms"] = (time.perf_counter() - now) * 1000
            return result

    runtime.drafter.propose = propose
    serving._match = match
    try:
        output = [
            chunk.token
            for chunk in serving.stream_generate(
                TimedModel(),
                tokenizer,
                prompt,
                runtime=runtime,
                max_tokens=count,
                ignore_eos=True,
            )
        ]
        close_round()
        elapsed = time.perf_counter() - start
    finally:
        runtime.drafter.propose = original_propose
        serving._match = original_match
    histogram = Counter(row["accepted"] for row in rounds)
    means = {
        key: sum(row[key] for row in rounds) / len(rounds)
        for key in ("draft_ms", "verify_ms", "round_ms")
    }
    means["host_ms"] = means["round_ms"] - means["draft_ms"] - means["verify_ms"]
    return output, {
        "tokens_per_second": len(output) / elapsed,
        "decode_seconds": elapsed,
        "rounds": len(rounds),
        "mean_accepted_tokens": sum(k * v for k, v in histogram.items()) / len(rounds),
        "acceptance_histogram": dict(sorted(histogram.items())),
        "mean_round_ms": means,
    }


def main(argv=None):
    """Run the benchmark; tiny mode's CPU default device is restored on exit."""
    previous = mx.default_device()
    try:
        return _main(argv)
    finally:
        mx.set_default_device(previous)


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Local target checkpoint (omit for tiny CPU smoke)"
    )
    parser.add_argument(
        "--draft", help="Local draft sidecar; defaults to model directory"
    )
    parser.add_argument(
        "--runtime-module",
        default="rapid_mlx.models.deepseek_v41_native.serving",
        help="Serving provider module; select a baseline provider for old/new comparisons",
    )
    parser.add_argument("--label", default="new")
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.tokens < 1 or args.repeats < 1:
        parser.error("tokens and repeats must be positive")
    serving = importlib.import_module(args.runtime_module)
    if args.model:
        model, tokenizer, runtime = serving.load_product_runtime(
            args.model,
            args.draft or args.model,
            target_revision="local",
            mtp_revision="local",
        )
    else:
        mx.set_default_device(mx.cpu)
        model, draft = tiny_setup(args.seed)
        draft = serving.DSpark(SimpleNamespace(w=draft.w, c=draft.c), pin_weights=False)
        serving._install_vectorized_mtp_attention(draft)
        serving._install_packed_mtp_moe(draft)
        tokenizer = TinyTokenizer()
        runtime = serving.DSparkRuntime(draft, "synthetic", "synthetic", "synthetic")
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    # Warm both paths; each measured run starts with fresh target/draft caches.
    plain_decode(model, prompt, min(8, args.tokens))
    speculative_decode(serving, model, tokenizer, runtime, prompt, min(8, args.tokens))
    runs = []
    for _ in range(args.repeats):
        plain, off = plain_decode(model, prompt, args.tokens)
        speculative, on = speculative_decode(
            serving, model, tokenizer, runtime, prompt, args.tokens
        )
        runs.append(
            {
                "draft_off": off,
                "draft_on": on,
                "greedy_equal": plain == speculative,
                "speedup": on["tokens_per_second"] / off["tokens_per_second"],
            }
        )
    print(
        format_bench_json(
            {
                "label": args.label,
                "mode": "checkpoint" if args.model else "tiny",
                "runtime_module": args.runtime_module,
                "python": platform.python_version(),
                "mlx": mx.__version__,
                "device": str(mx.default_device()),
                "seed": args.seed,
                "prompt_tokens": len(prompt),
                "generated_tokens": args.tokens,
                "runs": runs,
            },
            __file__,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
