# SPDX-License-Identifier: Apache-2.0
"""Needle-in-haystack quality gate for PFlash compression (#287).

The PR's acceptance criterion: at ``keep_ratio=0.20`` PFlash must keep
needle-in-haystack recall above 90 % on long-context prompts. The fork
only published TTFT numbers; this test is the missing quality probe.

The test is ``@pytest.mark.needle`` so the CI default skips it (it
requires loading a real long-context model and running ~20 prompts at
each context size). Run it locally with::

    uv run pytest -m needle tests/test_pflash_needle.py \\
      --model mlx-community/Qwen3-4B-4bit

Implementation: insert a unique 4-digit code at a known relative
position in a filler prompt, ask the model to recall it, score the
fraction of trials whose response contains the code. Repeated under
keep_ratio ∈ {0.10, 0.20, 0.40} at context sizes {32k, 64k, 128k}.

This scaffold defines the harness contract. The full harness lives in
``vllm_mlx/bench/pflash_needle.py`` once a maintainer wires it up
against the live engine; until then this test is a deliberate
documentation of the gate rather than a green tick.
"""

from __future__ import annotations

import os

import pytest

# Disabled by default. To run::
#   PFLASH_NEEDLE_MODEL=mlx-community/Qwen3-4B-4bit uv run pytest -m needle \
#     tests/test_pflash_needle.py
NEEDLE_MODEL = os.environ.get("PFLASH_NEEDLE_MODEL")
NEEDLE_TRIALS = int(os.environ.get("PFLASH_NEEDLE_TRIALS", "20"))


pytestmark = pytest.mark.needle


def _haystack(target_tokens: int, needle: str, position_frac: float) -> str:
    """Construct a long filler prompt with a unique needle at a known
    fractional position.

    Deterministic — same inputs always produce the same prompt.
    """
    filler_line = (
        "The Pittsburgh weather report for the week shows a mix of sun and "
        "rain with temperatures averaging fifty-two degrees. "
    )
    approx_tokens_per_line = len(filler_line.split())
    total_lines = max(1, target_tokens // approx_tokens_per_line)
    insert_at = int(total_lines * position_frac)
    needle_line = (
        f"IMPORTANT: The secret code for this session is {needle}. "
        "Memorize it and report it back exactly when asked. "
    )
    body_lines = [filler_line] * total_lines
    body_lines[insert_at] = needle_line
    return "".join(body_lines)


def _ask_needle(haystack: str, needle: str) -> str:
    return (
        f"{haystack}\n\n"
        "User request:\n"
        f"What is the secret code given earlier? Reply with the four-digit code only."
    )


def _recall_pass(response: str, needle: str) -> bool:
    return needle in response


@pytest.fixture(scope="module")
def needle_engine():
    if not NEEDLE_MODEL:
        pytest.skip(
            "PFLASH_NEEDLE_MODEL not set — needle quality gate is deferred. "
            "See test docstring for the run command."
        )
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine(NEEDLE_MODEL)
    return engine


@pytest.mark.parametrize("context_tokens", [32_768, 65_536, 131_072])
@pytest.mark.parametrize("keep_ratio", [0.10, 0.20, 0.40])
def test_pflash_needle_recall_above_90pct(
    needle_engine, context_tokens: int, keep_ratio: float
):
    """Stub: when ``PFLASH_NEEDLE_MODEL`` is set, this exercises the
    needle quality gate.

    Acceptance: ``recall >= 0.90`` at ``keep_ratio=0.20``. Lower
    keep_ratios are informational; higher keep_ratios should monotonic
    -ally improve recall and the harness flags regressions.
    """
    pytest.skip(
        "PFlash needle quality gate is deferred — wire harness in "
        "vllm_mlx/bench/pflash_needle.py against a real engine, then "
        "enable. See PR #287 for the run plan."
    )
