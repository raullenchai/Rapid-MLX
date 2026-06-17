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

A full engine-level harness (loads a long-context MLX model, drives
``BatchedEngine.chat``) is deferred — see the "Engine-level harness"
section at the bottom for the deliberate non-stub. The bench-side
sweep that produced the PR #649 TTFT + recall numbers lives at
``vllm_mlx/bench/pflash_replication.py`` and is exercised from the
``rapid-mlx bench`` CLI, not pytest.

The token-level tests below run on every PR (no marker) because they
need no model and finish in milliseconds. The ``needle`` marker
registered in ``pytest.ini`` is reserved for the future engine path.
"""

from __future__ import annotations

import pytest

from vllm_mlx.pflash import PFlashConfig, compress_tokens

# NOTE: no module-level ``pytestmark = pytest.mark.needle`` here. The
# token-level tests below run on every PR — they're the regression
# guard for the verified-tier default. The ``needle`` marker registered
# in ``pytest.ini`` is reserved for the deferred engine-level harness
# (see bottom of file).


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
# Engine-level harness — deferred.
# ---------------------------------------------------------------------------
#
# An end-to-end recall check against ``BatchedEngine.chat`` would close
# the loop on real long-context model behaviour, but ``BatchedEngine``
# exposes ``chat`` as an async coroutine and PFlash keep_ratio is wired
# via ``SchedulerConfig`` rather than per-call. Wiring a one-shot
# synchronous-friendly path here would either (a) leak partial
# async-runtime scaffolding into this test module or (b) drift from the
# real engine call shape — both worse than the current arrangement.
#
# The token-level tests above are the contracted regression guard for
# the verified-tier default (keep_ratio=0.20 must preserve the needle).
# The full engine sweep that produced the PR #649 TTFT + recall table
# lives at ``vllm_mlx/bench/pflash_replication.py`` and is invoked from
# the bench CLI, not pytest.
#
# When someone re-introduces an engine-level harness here, the right
# shape is::
#
#     @pytest.mark.needle
#     @pytest.mark.asyncio
#     async def test_recall_engine(needle_engine, ctx, ratio):
#         response = await needle_engine.chat(prompt, ...)
#         assert needle in response
#
# Don't leave a stub — codex round 2 [BLOCKING] caught the sync-call-
# against-async-method shape, and a stub that pretends to gate quality
# but doesn't is worse than no stub at all.
#
# Env contract for that future harness: ``PFLASH_NEEDLE_MODEL`` selects
# the model, ``PFLASH_NEEDLE_TRIALS`` controls per-cell aggregation.
# Re-instate as module-level reads when the harness lands; reading them
# here now would just be dead code.
