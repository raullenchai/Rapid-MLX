#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare greedy MTP at fixed draft depths with same-generator K=0.

This is a correctness diagnostic, not a throughput benchmark or a byte-parity
gate. It records stock ``mlx_lm`` output as context, uses the Rapid MTP
generator parked at K=0 as the contract reference, and reports the first token
divergence for fixed K>0 arms. A divergence may be caused by the known
``q_len=1`` versus ``q_len>=2`` numerical fork under quantized weights; it does
not by itself prove an acceptance or rollback defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.bench_spec_decode_mtp import (
    _BENCH_PROMPTS,
    _resolve_mtp_sidecar,
    _tokenizer_stop_tokens,
)

_DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-4bit"


def _parse_k_values(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "K values must be comma-separated integers"
        ) from exc
    if not values or values[0] != 0:
        raise argparse.ArgumentTypeError("K values must start with the K=0 control")
    if len(values) < 2:
        raise argparse.ArgumentTypeError(
            "K values must include at least one speculative depth"
        )
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("K values must be non-negative")
    if any(value > 3 for value in values):
        raise argparse.ArgumentTypeError("K values must be in the supported range 0..3")
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


def _token_sha256(tokens: tuple[int, ...]) -> str:
    return hashlib.sha256(
        ",".join(str(token) for token in tokens).encode("ascii")
    ).hexdigest()


def _loaded_model_type(model: Any) -> str | None:
    """Resolve the architecture label from common loaded-model shapes."""

    candidates = (
        getattr(model, "model_type", None),
        getattr(getattr(model, "args", None), "model_type", None),
        getattr(getattr(model, "config", None), "model_type", None),
        getattr(getattr(model, "language_model", None), "model_type", None),
        getattr(
            getattr(getattr(model, "language_model", None), "args", None),
            "model_type",
            None,
        ),
        getattr(
            getattr(getattr(model, "language_model", None), "config", None),
            "model_type",
            None,
        ),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _format_prompt(tokenizer: Any, prompt: str, *, chat_template: bool) -> str:
    """Optionally mirror the server's one-user-message chat-template input."""

    if not chat_template:
        return prompt
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        raise RuntimeError("tokenizer does not expose apply_chat_template")
    formatted = apply(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(formatted, str) or not formatted:
        raise RuntimeError("tokenizer returned an empty/non-string chat template")
    return formatted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--mtp-sidecar")
    parser.add_argument(
        "--prompts",
        type=int,
        default=len(_BENCH_PROMPTS),
        help=f"Number of built-in prompts to run (default: {len(_BENCH_PROMPTS)})",
    )
    parser.add_argument("--prompt-text", help="Run one explicit prompt instead")
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help=(
            "Wrap each prompt as one user message with add_generation_prompt "
            "and enable_thinking=false, matching the default server chat path"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=_parse_k_values("0,1,2,3"),
        help="Comma-separated fixed depths beginning with 0 (default: 0,1,2,3)",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args()


def _render_markdown(report: dict[str, Any]) -> None:
    print("# Fixed-K real-weight MTP consistency diagnostic\n")
    print(f"- model: `{report['model']}`")
    print(f"- sidecar: `{report['mtp_sidecar']}`")
    print(f"- max tokens: {report['max_tokens']}")
    print(f"- seed: {report['seed']}")
    print("- sampling: greedy (`temp=0`)")
    print("- reference: same Rapid generator with speculation parked at K=0\n")
    print(
        "| Prompt | stock/K=0 | K | decode tok/s | attempts | accepts | verify calls | "
        "complete | K=0 parity | first divergence | source |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---|---|---|---|")
    for prompt in report["prompts"]:
        stock_matches = prompt["stock_vs_k0_first_divergence"] is None
        for row in prompt["rows"]:
            divergence = row["first_divergence"]
            divergence_text = (
                "—"
                if divergence is None
                else (
                    f"{divergence['index']}: {divergence['control_token']} → "
                    f"{divergence['candidate_token']}"
                )
            )
            print(
                f"| {prompt['index']} | {'yes' if stock_matches else 'no'} | "
                f"{row['k']} | {row['decode_tok_per_sec']:.2f} | "
                f"{row['attempts']} | {row['accepts']} | "
                f"{row['verify_calls']} | "
                f"{'yes' if row['complete'] else 'no'} | "
                f"{'yes' if row['matches_k0'] else 'no'} | {divergence_text} | "
                f"{row['candidate_source_at_divergence'] or '—'} |"
            )
    print(
        "\nA K=0 mismatch with stock AR localizes a generator/harness difference. "
        "A fixed-K mismatch whose first differing token is target/non-draft can "
        "be the expected batched-forward numerical fork and is not, by itself, "
        "an acceptance or rollback failure."
    )


def main() -> int:
    args = _parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be greater than zero")
    if args.prompt_text is None and (
        args.prompts <= 0 or args.prompts > len(_BENCH_PROMPTS)
    ):
        raise SystemExit(f"--prompts must be between 1 and {len(_BENCH_PROMPTS)}")

    sidecar = _resolve_mtp_sidecar(args.model, args.mtp_sidecar)
    if sidecar is None:
        raise SystemExit(
            "No MTP sidecar is known for this model; pass --mtp-sidecar explicitly"
        )
    prompts = (
        (args.prompt_text,)
        if args.prompt_text is not None
        else _BENCH_PROMPTS[: args.prompts]
    )

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from rapid_mlx.spec_decode.mtp import (
        MTPAcceptCounter,
        dispatch_mtp_inject,
        dispatch_mtp_validate,
    )
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    model, tokenizer = load(args.model)
    stop_tokens = _tokenizer_stop_tokens(tokenizer)
    stock_by_prompt: list[tuple[int, ...]] = []
    formatted_prompts = tuple(
        _format_prompt(tokenizer, prompt, chat_template=args.chat_template)
        for prompt in prompts
    )
    for index, prompt in enumerate(formatted_prompts):
        print(f"[fixed-k-consistency] stock AR prompt {index + 1}", file=sys.stderr)
        tokens: list[int] = []
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            tokens.append(int(response.token))
            if len(tokens) >= args.max_tokens:
                break
        stock_by_prompt.append(tuple(tokens))

    model_type = _loaded_model_type(model)
    if model_type is None:
        raise RuntimeError(f"Could not resolve model_type for {args.model!r}")
    if not dispatch_mtp_inject(model, model_type, mtp_sidecar=sidecar):
        raise RuntimeError(
            f"MTP injection failed for {args.model!r} "
            f"(model_type={model_type!r}) with {sidecar!r}"
        )
    if not dispatch_mtp_validate(model, model_type):
        raise RuntimeError("MTP validation failed after injection")
    # Multimodal wrappers may preserve their public __call__ and expose the
    # MTP-only target contract through mtp_target_forward. Older text wrappers
    # are still driven through their patched inner model.
    generator_model = (
        model
        if callable(getattr(model, "mtp_target_forward", None))
        else getattr(model, "language_model", model)
    )

    prompt_reports: list[dict[str, Any]] = []
    activity_valid = True
    for index, prompt in enumerate(formatted_prompts):
        prompt_ids = mx.array(tokenizer.encode(prompt), mx.uint32)
        runs: dict[
            int,
            tuple[
                tuple[int, ...],
                tuple[bool, ...],
                Any,
                int,
                str,
                float,
                float,
            ],
        ] = {}
        for k in args.k_values:
            print(f"[fixed-k-consistency] prompt {index + 1} K={k}", file=sys.stderr)
            mx.random.seed(args.seed)
            counter = MTPAcceptCounter()
            timing: dict[str, float] = {}
            tokens: list[int] = []
            from_draft: list[bool] = []
            started = time.perf_counter()
            for token, _logprobs, drafted in mtp_generate_step(
                prompt_ids,
                generator_model,
                max_tokens=args.max_tokens,
                temp=0.0,
                accept_counter=counter,
                disable_auto_k=True,
                max_k=k,
                stop_tokens=stop_tokens,
                timing_stats=timing,
            ):
                token_id = int(token)
                tokens.append(token_id)
                from_draft.append(bool(drafted))
                if token_id in stop_tokens or len(tokens) >= args.max_tokens:
                    break
            elapsed = time.perf_counter() - started
            prompt_eval = float(timing.get("prompt_eval_seconds", 0.0))
            decode_elapsed = max(0.0, elapsed - prompt_eval)
            runs[k] = (
                tuple(tokens),
                tuple(from_draft),
                counter.snapshot(),
                int(timing.get("verify_calls", 0.0)),
                "stop_token"
                if tokens and tokens[-1] in stop_tokens
                else "max_tokens"
                if len(tokens) == args.max_tokens
                else "early_termination",
                elapsed,
                decode_elapsed,
            )

        control = runs[0][0]
        rows = []
        for k in args.k_values:
            (
                tokens,
                sources,
                counter,
                verify_calls,
                termination,
                elapsed,
                decode_elapsed,
            ) = runs[k]
            divergence = _first_divergence(control, tokens)
            divergence_index = divergence["index"] if divergence else None
            source = None
            if isinstance(divergence_index, int) and divergence_index < len(sources):
                source = "draft" if sources[divergence_index] else "target/non-draft"
            arm_active = (
                counter.attempts == 0 and verify_calls == 0
                if k == 0
                else counter.attempts > 0 and verify_calls > 0
            )
            complete = termination != "early_termination"
            activity_valid = activity_valid and arm_active and complete
            rows.append(
                {
                    "k": k,
                    "active": arm_active,
                    "complete": complete,
                    "termination": termination,
                    "attempts": counter.attempts,
                    "accepts": counter.accepts,
                    "verify_calls": verify_calls,
                    "n_tokens": len(tokens),
                    "elapsed_seconds": elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "decode_tok_per_sec": (
                        max(0, len(tokens) - 1) / decode_elapsed
                        if decode_elapsed > 0
                        else 0.0
                    ),
                    "token_sha256": _token_sha256(tokens),
                    "matches_k0": divergence is None,
                    "first_divergence": divergence,
                    "candidate_source_at_divergence": source,
                }
            )
        prompt_reports.append(
            {
                "index": index + 1,
                "prompt": prompts[index],
                "chat_template": args.chat_template,
                "stock_token_sha256": _token_sha256(stock_by_prompt[index]),
                "stock_vs_k0_first_divergence": _first_divergence(
                    stock_by_prompt[index], control
                ),
                "rows": rows,
            }
        )

    report = {
        "model": args.model,
        "mtp_sidecar": sidecar,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "sampling": {"temperature": 0.0},
        "controller": "disabled_fixed_k",
        "reference": "same_generator_k0",
        "activity_valid": activity_valid,
        "prompts": prompt_reports,
    }
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        _render_markdown(report)

    if not activity_valid:
        print(
            "[fixed-k-consistency] INVALID: an arm did not engage or complete",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
