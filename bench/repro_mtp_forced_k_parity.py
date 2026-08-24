#!/usr/bin/env python3
"""Reproduce greedy token parity at fixed MTP draft depths on real weights.

This is a correctness diagnostic, not a throughput benchmark. It compares
stock ``mlx_lm`` AR, the Rapid-MLX generator with speculation disabled (K=0),
and fixed K>0 runs. It requires observable speculative activity for every K>0
arm and reports the first token-stream divergence. Model downloads and
execution are opt-in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.bench_spec_decode_mtp import (
    _BENCH_PROMPTS,
    _resolve_mtp_sidecar,
    _run_once,
)

_DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-4bit"
_DEFAULT_PROMPT = _BENCH_PROMPTS[7]


def _parse_k_values(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "K values must be comma-separated integers"
        ) from exc
    if not values or values[0] != 0:
        raise argparse.ArgumentTypeError("K values must start with the K=0 control")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("K values must be non-negative")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("K values must not contain duplicates")
    return values


def _first_divergence(
    control: tuple[int, ...], candidate: tuple[int, ...]
) -> dict[str, int | None] | None:
    shared = min(len(control), len(candidate))
    for index in range(shared):
        if control[index] != candidate[index]:
            return {
                "index": index,
                "control_token": control[index],
                "candidate_token": candidate[index],
            }
    if len(control) == len(candidate):
        return None
    return {
        "index": shared,
        "control_token": control[shared] if shared < len(control) else None,
        "candidate_token": candidate[shared] if shared < len(candidate) else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--mtp-sidecar")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=_parse_k_values("0,1,2,3"),
        help="Comma-separated fixed depths beginning with 0 (default: 0,1,2,3)",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--require-parity",
        action="store_true",
        help="Exit 1 when an active K>0 arm differs from K=0",
    )
    return parser.parse_args()


def _render_markdown(report: dict[str, Any]) -> None:
    print("# Fixed-K real-weight MTP parity reproducer\n")
    print(f"- model: `{report['model']}`")
    print(f"- sidecar: `{report['mtp_sidecar']}`")
    print(f"- max tokens: {report['max_tokens']}")
    print(f"- seed: {report['seed']}")
    print("- sampling: greedy (`temp=0`)")
    print("- controller: disabled; K is fixed per arm")
    stock_divergence = report["stock_vs_k0_first_divergence"]
    print(
        "- stock `mlx_lm` vs same-generator K=0 parity: "
        f"{'yes' if stock_divergence is None else 'no'}"
    )
    if stock_divergence is not None:
        print(
            "- first stock/K=0 divergence: "
            f"index {stock_divergence['index']}: "
            f"{stock_divergence['control_token']} -> "
            f"{stock_divergence['candidate_token']}"
        )
    print()
    print(
        "| K | attempts | accepts | verify calls | tokens | K=0 parity | first divergence |"
    )
    print("|---:|---:|---:|---:|---:|---|---|")
    for row in report["rows"]:
        divergence = row["first_divergence"]
        if divergence is None:
            divergence_text = "—"
        else:
            source = row.get("candidate_source_at_divergence") or "n/a"
            divergence_text = (
                f"index {divergence['index']}: "
                f"{divergence['control_token']} -> {divergence['candidate_token']} "
                f"({source})"
            )
        print(
            f"| {row['k']} | {row['accept_attempts']} | {row['accept_count']} | "
            f"{row['verify_calls']} | {row['n_tokens']} | "
            f"{'yes' if row['matches_control'] else 'no'} | {divergence_text} |"
        )
    print(
        "\nThis diagnostic establishes whether a forced speculative path engages "
        "and preserves the K=0 greedy token stream. It does not by itself "
        "attribute a divergence to target-forward numerics, cache rollback, or "
        "another implementation detail."
    )


def main() -> int:
    args = _parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be greater than zero")

    sidecar = _resolve_mtp_sidecar(args.model, args.mtp_sidecar)
    if sidecar is None:
        raise SystemExit(
            "No MTP sidecar is known for this model; pass --mtp-sidecar explicitly"
        )

    print("[fixed-k-parity] running stock mlx_lm AR", file=sys.stderr)
    stock = _run_once(
        model_alias=args.model,
        condition="none",
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temp=0.0,
        mtp_sidecar=sidecar,
        mtp_max_k=0,
        mtp_disable_auto_k=True,
        seed=args.seed,
        capture_token_ids=True,
    )

    results = {}
    for k in args.k_values:
        print(f"[fixed-k-parity] running K={k}", file=sys.stderr)
        result = _run_once(
            model_alias=args.model,
            condition="ar" if k == 0 else "mtp",
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temp=0.0,
            mtp_sidecar=sidecar,
            mtp_max_k=k,
            mtp_disable_auto_k=True,
            seed=args.seed,
            capture_token_ids=True,
        )
        results[k] = result

    control = results[0]
    stock_vs_k0_divergence = _first_divergence(stock.token_ids, control.token_ids)
    rows = []
    activity_valid = True
    parity_held = True
    for k in args.k_values:
        result = results[k]
        divergence = _first_divergence(control.token_ids, result.token_ids)
        matches_control = divergence is None
        if k > 0:
            arm_active = result.accept_attempts > 0 and result.verify_calls > 0
            activity_valid = activity_valid and arm_active
            parity_held = parity_held and matches_control
        else:
            arm_active = result.accept_attempts == 0 and result.verify_calls == 0

        divergence_index = divergence["index"] if divergence is not None else None
        source = None
        if isinstance(divergence_index, int) and divergence_index < len(
            result.from_draft_flags
        ):
            source = (
                "draft"
                if result.from_draft_flags[divergence_index]
                else "target/non-draft"
            )
        rows.append(
            {
                "k": k,
                "active": arm_active,
                "accept_attempts": result.accept_attempts,
                "accept_count": result.accept_count,
                "verify_calls": result.verify_calls,
                "k_histogram": result.k_histogram,
                "n_tokens": result.n_tokens,
                "token_sha256": result.token_sha256,
                "matches_control": matches_control,
                "first_divergence": divergence,
                "candidate_source_at_divergence": source,
            }
        )

    report = {
        "model": args.model,
        "mtp_sidecar": sidecar,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "sampling": {"temperature": 0.0},
        "controller": "disabled_fixed_k",
        "stock_token_sha256": stock.token_sha256,
        "stock_vs_k0_first_divergence": stock_vs_k0_divergence,
        "activity_valid": activity_valid,
        "parity_held": parity_held,
        "rows": rows,
    }
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        _render_markdown(report)

    if not activity_valid:
        print(
            "[fixed-k-parity] INVALID: at least one K>0 arm recorded no "
            "speculative attempts or verify calls",
            file=sys.stderr,
        )
        return 2
    if args.require_parity and not parity_held:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
