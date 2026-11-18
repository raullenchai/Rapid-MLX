# SPDX-License-Identifier: Apache-2.0
"""Local real-weight Raon checks; writes measurements, never catalog claims.

Run from the repository root:
    RAPID_MLX_TELEMETRY=0 DO_NOT_TRACK=1 .venv/bin/python -m scripts.qualify_raon_1_nano

Downloads only the pinned catalog payload into the existing default HF cache.
The reference is the vendored release's direct weight loader; no checkpoint
Python or code from a reference/download directory is imported.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    os.environ["RAPID_MLX_TELEMETRY"] = "0"
    os.environ["DO_NOT_TRACK"] = "1"

    import mlx.core as mx

    from rapid_mlx.models import raon1_nano
    from rapid_mlx.reasoning import get_parser
    from rapid_mlx.utils.tokenizer import (
        RAPID_EXTRA_EOS_ATTR,
        _resolve_subfolder_checkpoint,
        load_model_with_fallback,
    )

    checkpoint = _resolve_subfolder_checkpoint("raon-1-nano")
    started = time.perf_counter()
    model, tokenizer = load_model_with_fallback(
        checkpoint, tokenizer_config={"trust_remote_code": False}
    )
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - started
    assert model.model_type == "raon1_nano"
    assert model.__class__.__module__ == "rapid_mlx.models.raon1_nano"
    caches = model.make_cache()
    cache_types = [type(c).__name__ for c in caches]
    assert cache_types.count("ArraysCache") == 18 and cache_types.count("KVCache") == 6

    fixed = mx.array(
        tokenizer.encode("The capital of France is", add_special_tokens=False)
    )[None]
    actual = model(fixed).astype(mx.float32)
    mx.eval(actual)
    # Exercise the vendor's direct loader separately from Rapid's registration,
    # trust overlay, sanitization and generic quantization/loading path.
    reference = raon1_nano.load_model(checkpoint)
    mx.eval(reference.parameters())
    expected = reference(fixed).astype(mx.float32)
    log_a = actual - mx.logsumexp(actual, axis=-1, keepdims=True)
    log_b = expected - mx.logsumexp(expected, axis=-1, keepdims=True)
    difference = mx.max(mx.abs(actual - expected)).item()
    kl = mx.mean(mx.sum(mx.exp(log_a) * (log_a - log_b), axis=-1)).item()
    print(json.dumps({"max_abs_logit_diff": difference, "mean_kl": kl}), flush=True)
    assert difference <= 1e-5 and abs(kl) <= 1e-6
    del reference, actual, expected, caches
    gc.collect()
    mx.clear_cache()

    stops = set(getattr(tokenizer, RAPID_EXTRA_EOS_ATTR, ()))
    stops.add(tokenizer.eos_token_id)
    assert {259202, 259200} <= stops
    results = []
    for prompt in (
        "What is 2 + 2?",
        "한국의 수도는 어디인가요?",
        "Write one sentence about the Moon.",
    ):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert text.endswith("<think>\n")
        ids = tokenizer.encode(text, add_special_tokens=False)
        cache = model.make_cache()
        start = time.perf_counter()
        logits = model(mx.array([ids]), cache=cache)[:, -1, :]
        token = int(mx.argmax(logits, axis=-1).item())
        first = time.perf_counter()
        tokens = []
        for _ in range(args.max_tokens):
            if token in stops:
                break
            tokens.append(token)
            logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
            token = int(mx.argmax(logits, axis=-1).item())
        end = time.perf_counter()
        output = tokenizer.decode(tokens, skip_special_tokens=False)
        reasoning, content = get_parser("raon_1")().extract_reasoning(output)
        result = dict(
            prompt=prompt,
            prompt_tokens=len(ids),
            ttft_seconds=first - start,
            generated_tokens=len(tokens),
            decode_tok_s=len(tokens) / (end - first),
            reasoning=reasoning,
            content=content,
            stopped_on_eos=token in stops,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    # Exactly 2048 real token IDs, cold recurrent/KV state. Prefill chunks
    # evaluate cache state only, matching serving; the final token includes
    # the output projection needed for the first decode token.
    seed = tokenizer.encode("Explain how rain forms. ", add_special_tokens=False)
    ids = (seed * (2048 // len(seed) + 1))[:2048]
    cache = model.make_cache()
    mx.synchronize()
    start = time.perf_counter()
    for offset in range(0, 2047, 512):
        model(mx.array([ids[offset : min(offset + 512, 2047)]]), cache=cache)
        mx.eval([c.state for c in cache])
    mx.eval(model(mx.array([[ids[-1]]]), cache=cache))
    elapsed = time.perf_counter() - start
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.machine()
    print(
        json.dumps(
            dict(
                checkpoint=str(checkpoint),
                chip=chip,
                macos=platform.mac_ver()[0],
                python=platform.python_version(),
                mlx=mx.__version__,
                mlx_lm=importlib.metadata.version("mlx-lm"),
                load_seconds=load_seconds,
                prefill_tokens=2048,
                prefill_seconds=elapsed,
                prefill_tok_s=2048 / elapsed,
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
