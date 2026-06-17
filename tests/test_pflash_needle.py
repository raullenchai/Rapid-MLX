# SPDX-License-Identifier: Apache-2.0
"""Needle-in-haystack quality gate for PFlash compression (#287).

The PR's acceptance criterion: at ``keep_ratio=0.20`` PFlash must keep
needle-in-haystack recall on long-context prompts. PR #649 measured this
end-to-end against live engines (Qwen3.5-4B / Qwen3.6-27B / Qwen3.6-35B
across 16k-64k contexts → 5/5 recall, 3.87x-8.5x TTFT speedup); that
evidence drove the verified-tier auto-ON default.

This module is the token-level companion to those engine runs — it
exercises ``compress_tokens`` directly (no model, no tokenizer, no
engine) on a synthetic haystack with a rare-token needle and asserts
the compressor preserves the needle at the validated default
``keep_ratio=0.20``. Catches the regression class where someone tunes
the scoring weights and accidentally drops the needle on the floor at
the verified-tier default.

The optional engine-level harness (the original intent of this file)
lives behind ``PFLASH_NEEDLE_MODEL`` — it loads a long-context MLX
model and runs the engine-side recall check. That path is opt-in
because it needs ~30 minutes of model load + generate per cell; the
token-level checks below run in milliseconds and gate every PR.

Run the engine path locally with::

    PFLASH_NEEDLE_MODEL=mlx-community/Qwen3-4B-4bit \\
      uv run pytest -m needle tests/test_pflash_needle.py

The engine path carries ``@pytest.mark.needle`` so the default CI run
skips it (see ``pytest.ini``); the token-level tests below run on every
PR because they need no model and finish in milliseconds.
"""

from __future__ import annotations

import os

import pytest

from vllm_mlx.pflash import PFlashConfig, compress_tokens

# Engine-path opt-in. The token-level tests below do not need this.
NEEDLE_MODEL = os.environ.get("PFLASH_NEEDLE_MODEL")
NEEDLE_TRIALS = int(os.environ.get("PFLASH_NEEDLE_TRIALS", "20"))

# NOTE: no module-level ``pytestmark = pytest.mark.needle`` here. The
# token-level tests below run on every PR — they're the regression
# guard for the verified-tier default. Only the engine-level
# ``test_pflash_needle_recall_above_90pct`` carries the ``needle``
# marker so the heavy model-load path stays opt-in (see pytest.ini).


# ---------------------------------------------------------------------------
# String-level haystack (used by both paths so the engine and token tests
# are constructed identically).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Token-level synthetic haystack — no tokenizer, no engine.
# ---------------------------------------------------------------------------
#
# We model a long prompt as: [sink] + filler + [needle_block] + filler +
# [query_tail]. The needle block contains rare token ids that re-appear
# in the query tail (so overlap-scoring should preserve them); the
# filler is a single repeated common token. This is the same workload
# shape the engine path exercises — just with token ids in place of
# tokenized text.


def _build_token_haystack(
    *, ctx_tokens: int, needle_position_frac: float
) -> tuple[list[int], list[int]]:
    """Return ``(prompt_tokens, needle_tokens)``.

    The needle is a short run of rare tokens (ids ≥ 500_000) that also
    appear in the trailing query window. Filler uses a single common
    token (id=7). This matches the structure the PR-text recall runs
    used: the needle line keyword-overlaps the query, which is the case
    PFlash's overlap-weighted scoring is designed to preserve. Anything
    that drops the needle here is a real PFlash regression.
    """
    sink = list(range(64))  # leading model/system tokens
    filler_token = 7
    needle = [500_001, 500_002, 500_003, 500_004]
    # Query mentions the same needle ids so the compressor's tail-
    # overlap scoring favors the needle block. PR text describes this
    # as the "easy case" — exactly what the verified-tier default is
    # supposed to handle.
    query_tail = [600_000, 500_001, 500_002, 500_003, 500_004, 600_001]

    middle_budget = max(0, ctx_tokens - len(sink) - len(needle) - len(query_tail))
    insert_at = int(middle_budget * needle_position_frac)
    middle = (
        [filler_token] * insert_at
        + needle
        + [filler_token] * max(0, middle_budget - insert_at)
    )
    # ``needle`` was already inserted into ``middle`` — strip the
    # outer-scope duplicate so the prompt isn't double-stuffed.
    prompt = sink + middle + query_tail
    return prompt, needle


@pytest.mark.parametrize(
    "ctx_tokens",
    [4_096, 8_192, 16_384],  # token-level only — engine harness exercises longer
)
@pytest.mark.parametrize("position_frac", [0.05, 0.5, 0.95])
def test_pflash_default_preserves_needle(ctx_tokens: int, position_frac: float) -> None:
    """``PFlashConfig()`` defaults (mode=off → set always to exercise
    compression) at the verified-tier ``keep_ratio=0.20`` must preserve
    the rare-token needle across the prompt. This is the regression
    guard for the bench-validated quality bar in PR #649.
    """
    prompt, needle = _build_token_haystack(
        ctx_tokens=ctx_tokens, needle_position_frac=position_frac
    )
    config = PFlashConfig(
        mode="always",
        threshold=1,
        # Defaults match the verified-tier auto-ON profile from PR #649
        # (keep_ratio=0.20 etc.). Spelled out here so a future default
        # bump still exercises the bench-validated number.
        keep_ratio=0.20,
        min_keep_tokens=2_048,
        sink_tokens=256,
        tail_tokens=2_048,
        block_size=128,
        query_window=512,
        stride_blocks=8,
    )
    result = compress_tokens(prompt, config)

    assert result.compressed is True, (
        f"ctx={ctx_tokens} pos={position_frac}: compressor returned unchanged "
        f"(reason={result.reason!r}) — verified-tier default must compress"
    )
    kept = set(result.tokens)
    missing = [tok for tok in needle if tok not in kept]
    assert not missing, (
        f"ctx={ctx_tokens} pos={position_frac}: PFlash @ keep_ratio=0.20 "
        f"dropped needle tokens {missing}. The verified-tier auto-ON "
        "default would silently degrade recall on long-context prompts."
    )


def test_pflash_aggressive_keep_ratio_can_drop_needle() -> None:
    """Inverse / negative control: at aggressive ``keep_ratio=0.02``
    the compressor may legitimately drop the needle. If this test
    starts unconditionally PASSING with the needle present, the
    compressor is silently ignoring the budget and the positive
    test above is no longer load-bearing.
    """
    prompt, needle = _build_token_haystack(ctx_tokens=16_384, needle_position_frac=0.5)
    config = PFlashConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.02,
        min_keep_tokens=64,
        sink_tokens=16,
        tail_tokens=16,
        block_size=128,
        query_window=64,
        stride_blocks=0,
    )
    result = compress_tokens(prompt, config)
    assert result.compressed is True
    # We assert the budget was honored; whether the needle survives at
    # this aggressive ratio is informational. This guards against the
    # "compressor silently bails out" regression class.
    assert len(result.tokens) < len(prompt)


# ---------------------------------------------------------------------------
# Engine-level harness (opt-in via PFLASH_NEEDLE_MODEL).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def needle_engine():
    if not NEEDLE_MODEL:
        pytest.skip(
            "PFLASH_NEEDLE_MODEL not set — engine-level needle recall "
            "is opt-in. The token-level tests above gate the verified-"
            "tier default. See module docstring for the engine path "
            "run command."
        )
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine(NEEDLE_MODEL)
    return engine


@pytest.mark.needle
@pytest.mark.parametrize("context_tokens", [32_768, 65_536, 131_072])
@pytest.mark.parametrize("keep_ratio", [0.10, 0.20, 0.40])
def test_pflash_needle_recall_above_90pct(
    needle_engine, context_tokens: int, keep_ratio: float
):
    """Engine-level recall check. Acceptance: ``recall >= 0.90`` at
    ``keep_ratio=0.20``. Lower keep_ratios are informational; higher
    keep_ratios should monotonically improve recall.

    The current implementation runs a single deterministic trial per
    cell (the synthetic haystack above) to keep the opt-in path cheap.
    Multi-trial aggregation lives in
    ``vllm_mlx/bench/pflash_replication.py`` for the full bench.
    """
    needle = "8421"
    haystack = _haystack(context_tokens, needle, position_frac=0.5)
    prompt = _ask_needle(haystack, needle)
    # Feed the engine fixture; engine wiring lives in ``BatchedEngine``
    # — we rely on its chat path threading ``pflash_config`` through.
    response = needle_engine.chat(
        prompt,
        max_tokens=64,
        pflash_keep_ratio=keep_ratio,
    )
    if keep_ratio >= 0.20:
        assert _recall_pass(response, needle), (
            f"engine path: keep_ratio={keep_ratio} ctx={context_tokens} "
            f"dropped needle {needle!r} from response {response!r}"
        )
