# SPDX-License-Identifier: Apache-2.0
"""Marvin's Garden — label-readout evaluation (reference /v1/classify).

Scores a decision model WITHOUT generating: one forward pass per prompt,
softmax restricted to the candidate letter tokens, argmax = decision. This
is the exact serving behavior the future ``/v1/classify`` endpoint must
reproduce, so this script is the executable spec.

Features beyond plain accuracy (the "Marvin edge" over Nimble):

* ``--styles base,concise,spec`` — paraphrase ensemble: average per-letter
  log-probabilities across renderings. Costs 2 extra forward passes
  (slightly slower, measurably smarter and better calibrated).
* Temperature calibration fit on the TRAIN pairs (never on the eval set)
  by NLL grid search; heldout metrics are reported raw and calibrated.
* Flip robustness: fraction of contrast groups where BOTH members are
  correct — the metric contrastive curation is supposed to move.
* Abstention: decisions below ``--abstain`` confidence route to a fallback
  model; coverage/accuracy trade-off is reported.

Usage::

    python bench/marvins_garden/eval_label_readout.py \
        --model prism-ml/Ternary-Bonsai-27B-mlx-2bit \
        --adapter bench/marvins_garden/adapters/marvins-garden \
        --styles base,concise,spec --output results/eval_marvin.json

Requires mlx + mlx-lm (imported lazily; everything else runs bare).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import render

HERE = Path(__file__).resolve().parent

import generate_contrastive as gen  # same-dir import; reuses family specs


def _tokenize_letter(tokenizer, letter: str) -> list[int]:
    """Token ids for a letter under both spacing variants (union)."""
    ids: list[int] = []
    for variant in (letter, " " + letter):
        try:
            enc = tokenizer.encode(variant)
        except TypeError:
            enc = tokenizer._tokenizer.encode(variant, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    if not ids:
        raise ValueError(f"letter {letter!r} does not tokenize to a single token")
    return ids


def _apply_chat(tokenizer, user_content: str) -> list[int]:
    messages = [{"role": "user", "content": user_content}]
    try:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(text)
    except AttributeError:
        hf = tokenizer._tokenizer
        text = hf.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return hf.encode(text, add_special_tokens=False)


def read_letter_probs(model, tokenizer, prompt: str, candidates: list[str]):
    """One forward pass; return {letter: probability} plus token count."""
    import mlx.core as mx
    import numpy as np

    tokens = _apply_chat(tokenizer, prompt)
    logits = model(mx.array(tokens)[None])
    last = np.array(logits[0, -1], copy=False).astype(np.float64)
    letter_to_ids = {render.letter_for(i): _tokenize_letter(tokenizer, render.letter_for(i))
                     for i in range(len(candidates))}
    union = sorted({tid for ids in letter_to_ids.values() for tid in ids})
    sub = last[union]
    sub = sub - sub.max()
    probs = np.exp(sub)
    probs /= probs.sum()
    index = {tid: j for j, tid in enumerate(union)}
    out = {}
    for letter, ids in letter_to_ids.items():
        out[letter] = float(sum(probs[index[tid]] for tid in ids))
    return out, len(tokens)


def _softmax_np(x):
    import numpy as np

    e = np.exp(x - x.max())
    return e / e.sum()


def fit_temperature(model, tokenizer, rows, styles, limit: int, correct_letter_of) -> float:
    """NLL grid search over T on TRAIN pairs (deterministic, no leakage)."""
    import numpy as np

    best_t, best_nll = 1.0, float("inf")
    subset = rows[:limit]
    cached = []
    for row in subset:
        letter = correct_letter_of(row)
        avg = None
        for style in styles:
            prompt = _render_row(row, style)
            probs, _ = read_letter_probs(model, tokenizer, prompt, row["candidates"])
            letters = sorted(probs)
            logp = np.log(np.array([probs[l] for l in letters]) + 1e-12)
            avg = logp if avg is None else avg + logp
        avg /= len(styles)
        cached.append((avg, letters.index(letter)))
    for t in [round(0.5 + 0.05 * k, 2) for k in range(91)]:
        nll = 0.0
        for logp, target in cached:
            p = _softmax_np(logp / t)
            nll -= np.log(max(p[target], 1e-12))
        nll /= len(cached)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def _render_row(row: dict, style: str) -> str:
    if style == "base":
        return row["input"]
    spec = gen._FAMILIES[row["family"]]
    scenario = row["meta"]["scenario"]
    return render.render_prompt(
        row["family"],
        spec["display_fields"](scenario),
        row["candidates"],
        spec["option_lines"],
        style=style,
    )


def ece(confidences, correct, bins: int = 15) -> float:
    import numpy as np

    conf = np.array(confidences)
    corr = np.array(correct, dtype=float)
    total = len(conf)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        e += mask.sum() / total * abs(corr[mask].mean() - conf[mask].mean())
    return float(e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--pairs", type=Path, default=HERE / "data" / "pairs_heldout.jsonl")
    parser.add_argument("--calibrate-pairs", type=Path, default=HERE / "data" / "pairs_train.jsonl")
    parser.add_argument("--styles", default="base", help="comma list from: " + ",".join(render.STYLES))
    parser.add_argument("--temperature", type=float, default=None, help="skip fitting, use this T")
    parser.add_argument("--calibrate-limit", type=int, default=256)
    parser.add_argument("--abstain", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dump", type=Path, default=None, help="per-sample predictions JSONL")
    args = parser.parse_args(argv)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    for style in styles:
        if style not in render.STYLES:
            parser.error(f"unknown style {style!r}")

    rows = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    from mlx_lm import load

    model, tokenizer = load(args.model, adapter_path=args.adapter)

    def correct_letter_of(row: dict) -> str:
        return render.label_letter(row["candidates"], row["label"])

    if args.temperature is None:
        cal_rows = [json.loads(l) for l in args.calibrate_pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
        temperature = fit_temperature(model, tokenizer, cal_rows, styles, args.calibrate_limit, correct_letter_of)
        print(f"fitted temperature T={temperature:.2f} on {min(args.calibrate_limit, len(cal_rows))} train samples")
    else:
        temperature = args.temperature

    import numpy as np

    records = []
    started = time.perf_counter()
    prefill_tokens = 0
    for row in rows:
        truth = correct_letter_of(row)
        letters = sorted(render.letter_for(i) for i in range(len(row["candidates"])))
        avg = None
        n_tokens = 0
        for style in styles:
            probs, n_tokens = read_letter_probs(model, tokenizer, _render_row(row, style), row["candidates"])
            logp = np.log(np.array([probs[l] for l in letters]) + 1e-12)
            avg = logp if avg is None else avg + logp
        avg /= len(styles)
        scaled = avg / temperature
        p = _softmax_np(scaled)
        pred = letters[int(p.argmax())]
        conf = float(p.max())
        prefill_tokens += n_tokens * len(styles)
        records.append({
            "pair_id": row["pair_id"],
            "family": row["family"],
            "contrast_group": row["contrast_group"],
            "flip_key": row["flip_key"],
            "truth": truth,
            "pred": pred,
            "confidence": conf,
            "correct": pred == truth,
        })
    elapsed = time.perf_counter() - started

    n = len(records)
    correct = [r["correct"] for r in records]
    accuracy = sum(correct) / n
    per_family = {}
    for family in sorted({r["family"] for r in records}):
        sub = [r for r in records if r["family"] == family]
        per_family[family] = {
            "n": len(sub),
            "accuracy": sum(r["correct"] for r in sub) / len(sub),
        }
    groups: dict[str, list[bool]] = {}
    for r in records:
        groups.setdefault(r["contrast_group"], []).append(r["correct"])
    both_correct = sum(all(v) for v in groups.values()) / len(groups)

    conf = [r["confidence"] for r in records]
    mean_conf_correct = (
        sum(r["confidence"] for r in records if r["correct"]) / max(sum(correct), 1)
    )
    mean_conf_wrong = (
        sum(r["confidence"] for r in records if not r["correct"]) / max(n - sum(correct), 1)
    )
    covered = [r for r in records if r["confidence"] >= args.abstain]
    abstain = {
        "threshold": args.abstain,
        "coverage": len(covered) / n,
        "accuracy_when_covered": (sum(r["correct"] for r in covered) / len(covered)) if covered else None,
    }

    report = {
        "schema_version": 1,
        "model": args.model,
        "adapter": args.adapter,
        "pairs": str(args.pairs),
        "styles": styles,
        "temperature": temperature,
        "n_samples": n,
        "n_groups": len(groups),
        "accuracy": accuracy,
        "per_family": per_family,
        "flip_both_members_correct": both_correct,
        "ece_15bin": ece(conf, correct),
        "mean_confidence_correct": mean_conf_correct,
        "mean_confidence_wrong": mean_conf_wrong,
        "abstain": abstain,
        "timing": {
            "total_seconds": elapsed,
            "ms_per_decision": elapsed * 1000 / n,
            "forward_passes_per_decision": len(styles),
            "mean_prefill_tokens": prefill_tokens / n,
            "generated_tokens": 0,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
