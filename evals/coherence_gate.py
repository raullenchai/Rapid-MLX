#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Output-coherence release gate — feeds golden prompts through a REAL running
``rapid-mlx serve`` and asserts the generated text is coherent (#1247).

This is the serve-path half of the coherence gate; the pure predicates and the
garbage detector live in :mod:`vllm_mlx.coherence` and are unit-tested in
ordinary CI. This script requires a server to already be listening (it does
**not** boot one) — mirroring ``evals/run_eval.py`` and
``tests/integrations/test_anthropic_sdk.py``. The release gauntlet
(``scripts/release_check_m3.sh``) boots ``rapid-mlx serve <model> --no-thinking``
and exports ``RAPID_MLX_BASE_URL``; this gate reads that env by default.

Usage
-----
    # against the server the release gauntlet already booted:
    python evals/coherence_gate.py

    # or point it explicitly:
    python evals/coherence_gate.py --base-url http://127.0.0.1:8000/v1

Two tiers:
    * BLOCKING — the deterministic golden-answer cases decide the exit code.
    * ADVISORY — the heuristic garbage detector (:func:`looks_like_garbage`) is
      printed as a diagnostic warning but NEVER changes the exit code, because a
      frequency heuristic cannot reliably tell diverse token soup from prose.

Exit codes:
    0 — every BLOCKING golden case passed (advisory warnings may still print)
    1 — one or more BLOCKING golden cases failed (wrong answer or think-leak)
    2 — no server reachable at the base URL, or it became unreachable mid-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

# Make ``vllm_mlx`` importable when run as ``python evals/coherence_gate.py``
# from a bare checkout (sys.path[0] is evals/, not the repo root). Harmless when
# rapid-mlx is already installed — an editable/site-packages copy still resolves
# first only if this insert is skipped, but preferring the checkout is correct
# for a gate that must test THIS tree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_mlx.coherence import (  # noqa: E402
    GOLDEN,
    GoldenCase,
    evaluate_case,
    evaluate_concluded,
    looks_like_garbage,
)

_DEFAULT_BASE_URL = os.environ.get("RAPID_MLX_BASE_URL", "http://127.0.0.1:8000/v1")

# DeepSeek-R1-Distill can spend roughly 450 tokens reasoning about even a
# one-word fact before emitting its conclusion.  The ordinary golden budgets
# intentionally stay tiny, but reasoning mode needs enough room to reach the
# answer instead of testing truncation behavior.
_REASONING_BUDGET_MULTIPLIER = 16


class InvalidServerResponseError(RuntimeError):
    """The server replied, but not with a valid chat-completion payload."""


def _generate(
    base_url: str, case: GoldenCase, *, timeout: float, thinking: bool = False
) -> str:
    """Non-streaming completion for ``case`` at temperature 0. Returns the
    visible assistant text (empty string if the model returned no content)."""
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": (
            case.max_tokens * _REASONING_BUDGET_MULTIPLIER
            if thinking
            else case.max_tokens
        ),
        "temperature": 0.0,
        "stream": False,
        # Match the gauntlet's --no-thinking boot for ordinary families: the
        # gate measures answer coherence, not thinking-mode behavior. For a
        # reasoning-distill model we serve WITH thinking enabled (it does not
        # honor --no-thinking anyway) so the parser can route the chain-of-
        # thought to the reasoning channel and leave the conclusion in content.
        "enable_thinking": thinking,
    }
    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout
    )
    resp.raise_for_status()
    try:
        data = resp.json()
        content = data["choices"][0]["message"].get("content")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise InvalidServerResponseError("malformed chat-completion response") from exc
    if content is None:
        return ""
    if not isinstance(content, str):
        raise InvalidServerResponseError(
            f"assistant content must be a string or null, got {type(content).__name__}"
        )
    return content


def _server_reachable(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL (default: $RAPID_MLX_BASE_URL or "
        "http://127.0.0.1:8000/v1)",
    )
    ap.add_argument(
        "--timeout", type=float, default=120.0, help="per-request timeout (s)"
    )
    ap.add_argument(
        "--reasoning-distill",
        action="store_true",
        help="serve the model with thinking enabled and score the concluded "
        "(post-reasoning) answer against the golden token instead of the raw "
        "visible text (for reasoning-distill families such as DeepSeek-R1).",
    )
    args = ap.parse_args()

    base_url = args.base_url
    thinking = args.reasoning_distill
    print("=" * 60)
    print("  output-coherence gate (#1247)")
    print(f"  base_url: {base_url}")
    print(f"  golden cases (blocking): {len(GOLDEN)}")
    print("=" * 60)

    if not _server_reachable(base_url):
        print(
            f"ERROR: no rapid-mlx server reachable at {base_url}. "
            "Start one with: rapid-mlx serve <model> --port 8000",
            file=sys.stderr,
        )
        return 2

    failures: list[tuple[str, str, str]] = []  # BLOCKING: (id, reason, snippet)
    advisories: list[tuple[str, str, str]] = []  # ADVISORY: (id, why, snippet)
    passed_n = 0
    infrastructure_failed = False
    for case in GOLDEN:
        try:
            text = _generate(base_url, case, timeout=args.timeout, thinking=thinking)
            if thinking:
                passed, reason = evaluate_concluded(case, text)
            else:
                passed, reason = evaluate_case(case, text)
        except (httpx.HTTPError, InvalidServerResponseError) as exc:
            passed, reason = False, f"server/protocol error: {exc}"
            infrastructure_failed = True
            text = ""
        except Exception as exc:  # server/protocol error mid-run -> a gate failure
            passed, reason = False, f"request error: {exc}"
            text = ""

        status = "PASS" if passed else "FAIL"
        snippet = (
            " ".join(text.split())[:80] if isinstance(text, str) else repr(text)[:80]
        )
        print(f"  [{status}] {case.id:<16} {reason}")
        if passed:
            passed_n += 1
        else:
            print(f"           output: {snippet!r}")
            failures.append((case.id, reason, snippet))

        if infrastructure_failed:
            break

        # Advisory-only: surface obvious degeneracy as a diagnostic. Never
        # affects the exit code — the heuristic can miss diverse token soup, so
        # it must not gate (or falsely gate) a release.
        if isinstance(text, str):
            is_garbage, why = looks_like_garbage(text)
            if is_garbage:
                advisories.append((case.id, why, snippet))

    print("=" * 60)
    print(f"  BLOCKING: {passed_n}/{len(GOLDEN)} golden cases passed")
    if advisories:
        print(f"  ADVISORY: garbage detector flagged {len(advisories)} output(s):")
        for cid, why, snippet in advisories:
            print(f"    - {cid}: {why}  |  {snippet!r}")
    print("=" * 60)

    if infrastructure_failed:
        print(
            "ERROR: the rapid-mlx server became unreachable or returned an "
            "invalid response while the coherence gate was running.",
            file=sys.stderr,
        )
        return 2

    if failures:
        print(
            "COHERENCE GATE FAILED — the served model gave a wrong or incoherent "
            "answer to a golden prompt. This is the class that shipped as garbage "
            "in #1234; do NOT release.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
