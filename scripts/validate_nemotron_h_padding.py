# SPDX-License-Identifier: Apache-2.0
"""Opt-in cached-model GPU check. Run under the shared large-model lock.

Use this same script from base and candidate checkouts, with each checkout on
PYTHONPATH. The JSON report records numerical errors as well as greedy tokens.
No weights are downloaded: --model must name an existing local snapshot.
Use --float32 with MLX_ENABLE_TF32=0 to control batch GEMM rounding. The
checkpoint-precision mode also reports greedy parity but can fail logit bounds
from accumulated rounding differences, including on unpadded rows.
"""

import argparse
import importlib
import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from rapid_mlx.utils.tokenizer import load_model_with_fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--float32", action="store_true", help="Promote floating parameters/activations"
    )
    args = parser.parse_args()
    assert args.model.is_dir(), args.model
    assert mx.default_device() == mx.gpu
    print(f"GPU: {mx.metal.device_info()}", flush=True)
    model, tokenizer = load_model_with_fallback(str(args.model))
    assert model.model_type == "nemotron_h", model.model_type
    if args.float32:
        # Keep the checkpoint's packed 4-bit weights; promote floating-point
        # parameters/activations to control batch GEMM rounding differences.
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
    # The unchanged, unpadded 308-token control has 0.001964 peak error on
    # both revisions with float32 quantized batch GEMMs. Keep a bounded
    # numerical allowance and always report raw errors alongside token parity.
    atol, rtol = (3e-3, 2e-4) if args.float32 else (0.125, 0.01)
    prompts = [
        "The capital of France is",
        "Explain why the sky is blue in simple terms. " * 30
        + "\nGive a concise answer:",
    ]
    tokens = [tokenizer.encode(prompt) for prompt in prompts]
    generate = importlib.import_module("mlx_lm.generate")
    batch = generate.PromptProcessingBatch(
        model=model,
        uids=[0, 1],
        caches=[model.make_cache(), model.make_cache()],
        prefill_step_size=256,
    )
    # Leave the final token for decode, as the scheduler does. This exposes
    # padded recurrent state through the very next predicted-token logits.
    batch.prompt([row[:-1] for row in tokens])
    results = []
    for row, prompt in enumerate(tokens):
        independent = model.make_cache()
        for start in range(0, len(prompt) - 1, 256):
            mx.eval(
                model(
                    mx.array([prompt[start : min(start + 256, len(prompt) - 1)]]),
                    cache=independent,
                )
            )
        batched = batch.extract_cache(row)
        inputs = [prompt[-1], prompt[-1]]
        continuations = [[], []]
        errors = []
        close = []
        for _ in range(8):
            logits = [
                model(mx.array([[token]]), cache=cache)[:, -1].astype(mx.float32)
                for token, cache in zip(inputs, [batched, independent])
            ]
            mx.eval(logits)
            delta = logits[0] - logits[1]
            errors.append(float(mx.max(mx.abs(delta))))
            close.append(bool(mx.allclose(logits[0], logits[1], atol=atol, rtol=rtol)))
            inputs = [int(mx.argmax(values)) for values in logits]
            for continuation, token in zip(continuations, inputs):
                continuation.append(token)
        results.append(
            {
                "prompt_tokens": len(prompt),
                "max_abs_logit_errors": errors,
                "logits_close": close,
                "batched_tokens": continuations[0],
                "independent_tokens": continuations[1],
                "batched_text": tokenizer.decode(continuations[0]),
                "independent_text": tokenizer.decode(continuations[1]),
                "greedy_match": continuations[0] == continuations[1],
            }
        )
        print(json.dumps(results[-1]), flush=True)
    report = {
        "device": mx.metal.device_info(),
        "snapshot": args.model.name,
        "model_type": model.model_type,
        "versions": {name: version(name) for name in ("mlx", "mlx-lm")},
        "float32": args.float32,
        "atol": atol,
        "rtol": rtol,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert all(all(row["logits_close"]) and row["greedy_match"] for row in results)


if __name__ == "__main__":
    main()
