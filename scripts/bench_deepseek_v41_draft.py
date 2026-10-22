#!/usr/bin/env python3
"""Single-stream greedy draft benchmark; tiny mode needs no checkpoint.

Prompts are rendered through the serving chat template by default, so the
numbers match what a chat client sees.  Acceptance depends heavily on the
prompt: repetitive raw text accepts nearly every draft token while ordinary
chat answers accept far fewer, so each prompt class is reported separately
and ``--raw`` output is labelled as such.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import time
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx

from scripts.bench_metadata import format_bench_json

# Version 2 moves flat runs into classes[i].runs and adds prompt_format.
OUTPUT_FORMAT_VERSION = 2


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


# Built-in chat prompt classes.  Speculative acceptance differs by an order of
# magnitude between them, so a single prompt cannot stand for chat traffic.
PROMPT_CLASSES = {
    "qa": "Explain why the sky is blue, and why sunsets are red.",
    "code": (
        "Write a Python function that parses ISO-8601 durations like "
        "'P3DT4H5M' into seconds, with type hints, a docstring and pytest tests."
    ),
    "structured": (
        "Return a JSON array of the ten largest countries by area. Each item: "
        "name, capital, area_km2, continent. JSON only."
    ),
    "summarize": (
        "Summarize the following passage in three bullet points.\n\n"
        "Before the marine chronometer, sailors could find their latitude from "
        "the height of the sun at noon, but longitude required knowing the time "
        "at a reference meridian while far out at sea. Pendulum clocks failed "
        "on a rolling deck, and temperature swings changed the length of their "
        "springs. After several naval disasters, a prize was offered for a "
        "practical method. Astronomers favoured the lunar-distance method, "
        "which compared the moon's position against bright stars and then "
        "consulted printed tables, a calculation that took hours. A carpenter "
        "turned clockmaker instead spent decades building a series of sea "
        "clocks, each smaller than the last, using bimetallic strips to "
        "compensate for heat and caged roller bearings to avoid lubrication. "
        "His fourth design, roughly the size of a large pocket watch, lost only "
        "a few seconds on a voyage to the Caribbean. The prize committee was "
        "slow to accept the result, and he was paid in instalments only after "
        "copies proved that the design could be reproduced. Within a century, "
        "chronometers were standard equipment, and the lunar tables slowly "
        "fell out of use."
    ),
}


def encode_prompt(serving, model, tokenizer, text, *, raw):
    """Token ids exactly as the server would build them, unless ``raw``."""
    if not raw:
        request = SimpleNamespace(
            tools=None, messages=[SimpleNamespace(role="user", content=text)]
        )
        text = serving.render_prompt(tokenizer, model, request)
    return tokenizer.encode(text, add_special_tokens=False)


def first_difference(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b),
        None if len(left) == len(right) else min(len(left), len(right)),
    )


class TinyTokenizer:
    clean_up_tokenization_spaces = False
    bos_token = "<bos>"
    eos_token = "<eos>"
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


@contextmanager
def draft_component_timing(draft, enabled=False):
    """Diagnostic boundaries force completion; totals include synchronization cost."""
    totals = {name: 0.0 for name in ("attention", "moe", "head", "markov")}
    originals = []

    def wrap(original, name):
        def timed(*args, **kwargs):
            # Materialize inputs before starting: lazy HC/norm work belongs
            # outside these component totals, not to the next projection.
            mx.eval(
                [
                    value
                    for value in (*args, *kwargs.values())
                    if isinstance(value, mx.array)
                ]
            )
            mx.synchronize()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
            totals[name] += (time.perf_counter() - start) * 1000
            return result

        return timed

    try:
        if enabled:
            for owner, attribute, name in (
                (draft, "attention", "attention"),
                (draft.adapter, "moe", "moe"),
                (draft, "head", "head"),
                (draft, "markov", "markov"),
            ):
                original = getattr(owner, attribute)
                originals.append((owner, attribute, original, attribute in vars(owner)))
                setattr(owner, attribute, wrap(original, name))
        yield totals
    finally:
        for owner, attribute, original, was_instance in reversed(originals):
            if was_instance:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)


def speculative_decode(
    serving, model, tokenizer, runtime, prompt, count, *, component_timers=False
):
    """Time production decode, excluding prefill and including streaming host work."""
    rounds = []
    original_propose = runtime.drafter.propose
    original_match = serving._match
    start = None
    pending = False

    def close_round():
        if rounds:
            mx.synchronize()
            rounds[-1]["round_ms"] = (time.perf_counter() - rounds[-1]["start"]) * 1000

    def begin_round():
        nonlocal start, pending
        close_round()
        mx.synchronize()
        now = time.perf_counter()
        if start is None:
            start = now
        row = {"start": now, "verify_ms": 0.0, "draft_ms": 0.0, "emitted": 0}
        rounds.append(row)
        pending = True
        return row

    def propose(seed, **kwargs):
        row = begin_round()
        now = time.perf_counter()
        result = original_propose(seed, **kwargs)
        mx.eval(result)
        mx.synchronize()
        row["draft_ms"] = (time.perf_counter() - now) * 1000
        return result

    def match(*args):
        nonlocal pending
        pending = False
        result = original_match(*args)
        rounds[-1]["accepted"] = result[3]
        return result

    class TimedModel:
        def make_cache(self, **kwargs):
            return model.make_cache(**kwargs)

        def __call__(self, *args, **kwargs):
            if kwargs.get("enable_rollback") and not pending:
                begin_round()
            mx.synchronize()
            now = time.perf_counter()
            result = model(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
            if kwargs.get("enable_rollback"):
                rounds[-1]["verify_ms"] = (time.perf_counter() - now) * 1000
            return result

    runtime.drafter.propose = propose
    serving._match = match
    try:
        with draft_component_timing(runtime.drafter, component_timers) as component_ms:
            output = []
            for chunk in serving.stream_generate(
                TimedModel(),
                tokenizer,
                prompt,
                runtime=runtime,
                max_tokens=count,
                ignore_eos=True,
            ):
                output.append(chunk.token)
                # Count what each round actually emits; a round cut short by EOS
                # or the token limit emits fewer than its accepted drafts + 1.
                if rounds:
                    rounds[-1]["emitted"] += 1
            close_round()
            elapsed = time.perf_counter() - start
    finally:
        runtime.drafter.propose = original_propose
        serving._match = original_match
    histogram = Counter(row["accepted"] for row in rounds)
    accepted = sum(k * v for k, v in histogram.items()) / len(rounds)
    means = {
        key: sum(row[key] for row in rounds) / len(rounds)
        for key in ("draft_ms", "verify_ms", "round_ms")
    }
    means["host_ms"] = means["round_ms"] - means["draft_ms"] - means["verify_ms"]
    if component_timers:
        means.update(
            {
                f"draft_{name}_ms": total / len(rounds)
                for name, total in component_ms.items()
            }
        )
    return output, {
        "tokens_per_second": len(output) / elapsed,
        "decode_seconds": elapsed,
        "rounds": len(rounds),
        "mean_accepted_tokens": accepted,
        # Tokens one round emits (normally accepted drafts + the verify's own
        # token).  Speculation only pays when this exceeds round_ms / step_ms.
        "tokens_per_round": sum(row["emitted"] for row in rounds) / len(rounds),
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
    parser.add_argument(
        "--component-timers",
        action="store_true",
        help="Synchronize draft attention/MoE/head/Markov boundaries (perturbs latency)",
    )
    parser.add_argument("--label", default="new")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--prompt-class",
        action="append",
        choices=sorted(PROMPT_CLASSES),
        help="Built-in chat prompt class; repeatable (default: all classes)",
    )
    prompts.add_argument(
        "--prompt", help="Custom user message, measured as prompt class 'custom'"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Encode prompt text as-is instead of rendering the chat template",
    )
    parser.add_argument(
        "--mixed-target-precision",
        action="store_true",
        help="Load a mixed-precision target, as the server does for such checkpoints",
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.tokens < 1 or args.repeats < 1:
        parser.error("tokens and repeats must be positive")
    if args.raw and args.prompt == "":
        parser.error("raw prompt must not be empty")
    serving = importlib.import_module(args.runtime_module)
    if args.model:
        model, tokenizer, runtime = serving.load_product_runtime(
            args.model,
            args.draft or args.model,
            target_revision="local",
            mtp_revision="local",
            **({"mixed_target_precision": True} if args.mixed_target_precision else {}),
        )
    else:
        mx.set_default_device(mx.cpu)
        model, draft = tiny_setup(args.seed)
        draft = serving.DSpark(SimpleNamespace(w=draft.w, c=draft.c), pin_weights=False)
        serving._install_vectorized_mtp_attention(draft)
        serving._install_packed_mtp_moe(draft)
        tokenizer = TinyTokenizer()
        runtime = serving.DSparkRuntime(draft, "synthetic", "synthetic", "synthetic")
    if args.prompt is not None:
        texts = {"custom": args.prompt}
    else:
        texts = {
            name: PROMPT_CLASSES[name] for name in args.prompt_class or PROMPT_CLASSES
        }
    prompts = {
        name: encode_prompt(serving, model, tokenizer, text, raw=args.raw)
        for name, text in texts.items()
    }
    classes = []
    for name, prompt in prompts.items():
        # Warm both paths on this prompt's shapes before measuring it; each
        # measured run starts with fresh target/draft caches.
        plain_decode(model, prompt, min(8, args.tokens))
        speculative_decode(
            serving, model, tokenizer, runtime, prompt, min(8, args.tokens)
        )
        runs = []
        for _ in range(args.repeats):
            plain, off = plain_decode(model, prompt, args.tokens)
            speculative, on = speculative_decode(
                serving,
                model,
                tokenizer,
                runtime,
                prompt,
                args.tokens,
                component_timers=args.component_timers,
            )
            runs.append(
                {
                    "draft_off": off,
                    "draft_on": on,
                    # Batched verify and single-token decode reduce in different
                    # orders, so on real checkpoints draft-on output is exact with
                    # respect to the verify forward but may fork from plain greedy.
                    "greedy_equal": plain == speculative,
                    "first_difference": first_difference(plain, speculative),
                    "speedup": on["tokens_per_second"] / off["tokens_per_second"],
                }
            )
        classes.append(
            {
                "prompt_class": name,
                "prompt_tokens": len(prompt),
                "runs": runs,
            }
        )
    print(
        format_bench_json(
            {
                "label": args.label,
                "mode": "checkpoint" if args.model else "tiny",
                "runtime_module": args.runtime_module,
                "component_timers": args.component_timers,
                "python": platform.python_version(),
                "mlx": mx.__version__,
                "device": str(mx.default_device()),
                "seed": args.seed,
                # "raw" numbers are not chat numbers; never quote them as such.
                "prompt_format": "raw" if args.raw else "chat",
                "output_format_version": OUTPUT_FORMAT_VERSION,
                "generated_tokens": args.tokens,
                "classes": classes,
            },
            __file__,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
