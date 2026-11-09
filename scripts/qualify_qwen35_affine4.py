#!/usr/bin/env python3
"""Compare stock/fused B1 teacher forcing on a fixed local UTF-8 corpus.

Example: python scripts/qualify_qwen35_affine4.py --model <snapshot>
    --corpus <public-text.txt> --output <results-directory>
Runs both paths with reset caches and scores the same 50,000 next tokens.
No dataset/model downloads or cache redirection are performed by this script.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_tokenizer
from mlx_vlm import load

from rapid_mlx.compiled_precision import install_qwen35_attention_gate_precision
from rapid_mlx.gdn_in_proj_fusion import fuse_gdn_in_proj
from rapid_mlx.moe_fusion import fuse_gate_up
from rapid_mlx.patches.qwen35_affine4 import install_qwen35_affine4
from rapid_mlx.qwen35_fused_gdn_decode import install_qwen35_fused_gdn_decode
from rapid_mlx.qwen35_moe_router import install_qwen35_moe_router


def logits_of(out):
    return out.logits if hasattr(out, "logits") else out


def functional(model, tokenizer):
    prompts = {
        "math": "What is 19 multiplied by 23? Answer with the integer only.",
        "json": 'Return exactly this JSON object, with no other text: {"ok":true,"count":3}',
        "code": "Return only Python code defining square(n) which returns n multiplied by n. No markdown.",
    }
    results = {}
    for case, text in prompts.items():
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        cache = make_prompt_cache(model)
        token = mx.argmax(
            logits_of(model(mx.array([prompt]), cache=cache))[:, -1], axis=-1
        )
        generated = []
        for _ in range(128):
            tid = token.item()
            if tid in tokenizer.eos_token_ids:
                break
            generated.append(tid)
            token = mx.argmax(
                logits_of(model(token[:, None], cache=cache))[:, -1], axis=-1
            )
        response = tokenizer.decode(generated).strip()
        ok = False
        try:
            if case == "math":
                ok = response == "437"
            elif case == "json":
                ok = json.loads(response) == {"ok": True, "count": 3}
            else:
                tree = ast.parse(response)
                fn = tree.body[0]
                ret = fn.body[0].value
                ok = (
                    len(tree.body) == 1
                    and isinstance(fn, ast.FunctionDef)
                    and fn.name == "square"
                    and [a.arg for a in fn.args.args] == ["n"]
                    and not fn.args.posonlyargs
                    and not fn.args.kwonlyargs
                    and fn.args.vararg is None
                    and fn.args.kwarg is None
                    and not fn.args.defaults
                    and len(fn.body) == 1
                    and isinstance(fn.body[0], ast.Return)
                    and isinstance(ret, ast.BinOp)
                    and isinstance(ret.op, ast.Mult)
                    and isinstance(ret.left, ast.Name)
                    and ret.left.id == "n"
                    and isinstance(ret.right, ast.Name)
                    and ret.right.id == "n"
                )
        except (ValueError, SyntaxError, AttributeError, IndexError):
            pass
        results[case] = {"text": response, "correct": ok}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=50000)
    parser.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()
    if args.positions <= 0:
        parser.error("--positions must be positive")
    if args.context <= 0:
        parser.error("--context must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.corpus.read_bytes()
    model, _ = load(args.model)
    model = model.language_model
    tokenizer = load_tokenizer(Path(args.model))
    text = raw.decode("utf-8")
    # Strip the public-domain book's boilerplate and front matter if present.
    start = text.find("CHAPTER 1. Loomings.", text.find("*** START"))
    if start >= 0:
        text = text[start:]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    needed = args.positions + math.ceil(args.positions / args.context)
    if len(tokens) < needed:
        raise ValueError("corpus too short")
    tokens = tokens[:needed]
    fusion = {
        "gate_up": fuse_gate_up(model, extended_call_contract=True),
        "router": install_qwen35_moe_router(model),
        "gdn_input": fuse_gdn_in_proj(model),
        "gdn_decode": install_qwen35_fused_gdn_decode(model),
        "attention_gate": install_qwen35_attention_gate_precision(model),
    }
    os.environ["RAPID_MLX_QWEN35_AFFINE4"] = "1"
    fusion["affine4"] = install_qwen35_affine4(model)
    if fusion["affine4"] != 40:
        raise RuntimeError(f"expected 40 enrolled blocks, got {fusion}")
    result = {
        "model": args.model,
        "mlx": mx.__version__,
        "fusion": fusion,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "context": args.context,
        "positions": args.positions,
    }
    print(json.dumps(result), flush=True)
    predictions = {}
    for arm in ("stock", "fused"):
        os.environ["RAPID_MLX_QWEN35_AFFINE4"] = "1" if arm == "fused" else "0"
        nll, top1 = [], []
        start = time.perf_counter()
        for offset in range(0, len(tokens) - 1, args.context + 1):
            cache = make_prompt_cache(model)
            end = min(offset + args.context, len(tokens) - 1)
            for pos in range(offset, end):
                if len(top1) == args.positions:
                    break
                logits = logits_of(model(mx.array([[tokens[pos]]]), cache=cache))[
                    0, -1
                ].astype(mx.float32)
                loss = mx.logsumexp(logits) - logits[tokens[pos + 1]]
                prediction = mx.argmax(logits)
                mx.eval(loss, prediction)
                nll.append(loss.item())
                top1.append(prediction.item())
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "positions": len(top1),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
            if len(top1) == args.positions:
                break
        if len(top1) != args.positions:
            raise RuntimeError("incorrect position count")
        row = {"nll": nll, "top1": top1}
        (args.output / f"{arm}.json").write_text(json.dumps(row))
        predictions[arm] = top1
        result[arm] = {
            "ppl": math.exp(sum(nll) / len(nll)),
            "seconds": time.perf_counter() - start,
            "functional": functional(model, tokenizer),
        }
        (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    result["relative_ppl_delta"] = result["fused"]["ppl"] / result["stock"]["ppl"] - 1
    result["top1_agreement_pct"] = (
        100
        * sum(a == b for a, b in zip(predictions["stock"], predictions["fused"]))
        / args.positions
    )
    result["qualified"] = abs(result["relative_ppl_delta"]) <= 0.005 and all(
        v["correct"]
        for arm in ("stock", "fused")
        for v in result[arm]["functional"].values()
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
