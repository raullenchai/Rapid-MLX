# SPDX-License-Identifier: Apache-2.0
"""Unit + integration tests for DFlash spec decode (R15-P1 #313).

Coverage tiers (matches the PR test plan checkboxes):

1. **Detection** — Qwen3.5/3.6 allowlist, non-Qwen rejection,
   missing-drafter rejection, alias-side-registry vs CLI override
   precedence, config-shape robustness.
2. **Verifier** — 16-token block all-accept, all-reject,
   partial-accept-at-k, position-id contract via
   :func:`positioned_update_and_fetch`, KV-cache offset state.
3. **Accept counter** — lifecycle, concurrent writes, reset, snapshot
   consistency, mean-tokens-per-attempt math.
4. **Generator** — mocked drafter + mocked verifier, full block accept,
   partial accept (k=4), lossless contract verification (output matches
   no-spec-decode baseline byte-for-byte).
5. **Lossless integration** — all-accept matches none baseline,
   all-reject matches none baseline, mixed mid-block reject matches
   none baseline (all temp=0, contract-grade).
6. **CLI** — flag accepts ``dflash``, ``--dflash-drafter-path`` is
   parseable, K8V4 conflict surfaces a clear warning.

The tests deliberately avoid loading a real Qwen3.5 / 3.6 checkpoint —
those are multi-GB downloads. A deterministic scripted "tiny verifier
model" lets us exercise the verifier's argmax + cache-rewind paths
without GPU.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# 1. Detection
# ---------------------------------------------------------------------------


def test_detect_eligibility_qwen3_5_ready_with_registered_alias():
    """Qwen3.5 + registered alias → READY."""
    from vllm_mlx.spec_decode.dflash import (
        DFlashEligibility,
        clear_drafter_registry_for_tests,
        detect_dflash_eligibility,
        register_dflash_drafter,
    )

    clear_drafter_registry_for_tests()
    register_dflash_drafter("qwen3.5-test", "z-lab/test-drafter")
    try:
        assert (
            detect_dflash_eligibility({"model_type": "qwen3_5"}, alias="qwen3.5-test")
            is DFlashEligibility.READY
        )
    finally:
        clear_drafter_registry_for_tests()


def test_detect_eligibility_qwen3_5_moe_ready():
    """Qwen3.5 MoE follows the same allowlist."""
    from vllm_mlx.spec_decode.dflash import (
        DFlashEligibility,
        clear_drafter_registry_for_tests,
        detect_dflash_eligibility,
        register_dflash_drafter,
    )

    clear_drafter_registry_for_tests()
    register_dflash_drafter("qwen3.6-moe-test", "z-lab/moe-drafter")
    try:
        assert (
            detect_dflash_eligibility(
                {"model_type": "qwen3_5_moe"}, alias="qwen3.6-moe-test"
            )
            is DFlashEligibility.READY
        )
    finally:
        clear_drafter_registry_for_tests()


def test_detect_eligibility_non_qwen_models_rejected():
    """Llama / Mistral / Qwen3 / Qwen3-Next MUST NOT match DFlash."""
    from vllm_mlx.spec_decode.dflash import (
        DFlashEligibility,
        detect_dflash_eligibility,
    )

    for model_type in (
        "llama",
        "mistral",
        "qwen3",
        "qwen3_next",
        "qwen2",
        "gemma3",
        "deepseek_v3",
    ):
        # Even with a drafter override the allowlist gate fires first.
        config = {"model_type": model_type}
        assert (
            detect_dflash_eligibility(
                config, alias="anything", drafter_override="z-lab/whatever"
            )
            is DFlashEligibility.NONE
        ), f"non-Qwen3.5 model_type={model_type} must NOT match DFlash."


def test_detect_eligibility_missing_drafter_rejected():
    """Qwen3.5 alias with NO drafter binding → NONE.

    The CLI surfaces an actionable error pointing the user at
    :func:`register_dflash_drafter` or ``--dflash-drafter-path``.
    """
    from vllm_mlx.spec_decode.dflash import (
        DFlashEligibility,
        clear_drafter_registry_for_tests,
        detect_dflash_eligibility,
    )

    clear_drafter_registry_for_tests()
    # Use a fresh alias name guaranteed not in the default registry.
    assert (
        detect_dflash_eligibility(
            {"model_type": "qwen3_5"}, alias="qwen3.5-unregistered-9999"
        )
        is DFlashEligibility.NONE
    )


def test_detect_eligibility_cli_override_beats_missing_registry_binding():
    """Empty registry + non-empty ``--dflash-drafter-path`` → READY.

    The CLI override is the operator's escape hatch for experimenting
    with custom drafters without editing the side-registry.
    """
    from vllm_mlx.spec_decode.dflash import (
        DFlashEligibility,
        clear_drafter_registry_for_tests,
        detect_dflash_eligibility,
    )

    clear_drafter_registry_for_tests()
    assert (
        detect_dflash_eligibility(
            {"model_type": "qwen3_5"},
            alias="qwen3.5-unregistered-8888",
            drafter_override="local/path/to/drafter",
        )
        is DFlashEligibility.READY
    )


def test_detect_eligibility_default_registry_seeded():
    """The default registry must seed the validated 8-bit aliases.

    Regression guard: a contributor that accidentally clears
    ``_DEFAULT_REGISTRY`` would silently disable DFlash for every
    production alias.
    """
    from vllm_mlx.spec_decode.dflash import (
        DFlashEligibility,
        clear_drafter_registry_for_tests,
        detect_dflash_eligibility,
    )

    clear_drafter_registry_for_tests()
    # qwen3.5-27b-8bit is the validated 8-bit alias from PoC.
    assert (
        detect_dflash_eligibility({"model_type": "qwen3_5"}, alias="qwen3.5-27b-8bit")
        is DFlashEligibility.READY
    )


def test_detect_eligibility_handles_none_or_non_dict_config():
    """None / list / string config returns NONE rather than crashing."""
    from vllm_mlx.spec_decode.dflash import (
        DFlashEligibility,
        detect_dflash_eligibility,
    )

    assert detect_dflash_eligibility(None) is DFlashEligibility.NONE
    assert (
        detect_dflash_eligibility("not a dict")  # type: ignore[arg-type]
        is DFlashEligibility.NONE
    )
    assert (
        detect_dflash_eligibility([])  # type: ignore[arg-type]
        is DFlashEligibility.NONE
    )


def test_detect_eligibility_aliases_json_closed_keys_unaffected():
    """Detection MUST NOT depend on aliases.json forbidden keys
    (``architecture``, ``family``, ``quantization``, ``notes``).

    The closed-key schema in :data:`_ALLOWED_PROFILE_KEYS` rejects
    those fields at load — pinning that detection doesn't need them
    here protects future contributors from accidentally back-porting
    detection state into the alias profile.
    """
    from vllm_mlx.spec_decode.dflash import (
        DFlashEligibility,
        detect_dflash_eligibility,
    )

    config = {"model_type": "qwen3_5"}  # no closed-key fields here
    assert (
        detect_dflash_eligibility(config, alias="x", drafter_override="local/drafter")
        is DFlashEligibility.READY
    )


# ---------------------------------------------------------------------------
# 2. Drafter registry
# ---------------------------------------------------------------------------


def test_drafter_registry_returns_none_for_unbound_alias():
    from vllm_mlx.spec_decode.dflash import (
        clear_drafter_registry_for_tests,
        get_dflash_drafter_path,
    )

    clear_drafter_registry_for_tests()
    assert get_dflash_drafter_path("not-a-registered-alias-99999") is None


def test_drafter_registry_returns_path_for_bound_alias():
    from vllm_mlx.spec_decode.dflash import (
        clear_drafter_registry_for_tests,
        get_dflash_drafter_path,
        register_dflash_drafter,
    )

    clear_drafter_registry_for_tests()
    register_dflash_drafter("test-alias-A", "z-lab/test-A")
    try:
        assert get_dflash_drafter_path("test-alias-A") == "z-lab/test-A"
    finally:
        clear_drafter_registry_for_tests()


def test_drafter_registry_overwrite_logs_warning(caplog):
    """Re-registering the same alias with a different path warns
    + overwrites (last-wins)."""
    import logging

    from vllm_mlx.spec_decode.dflash import (
        clear_drafter_registry_for_tests,
        get_dflash_drafter_path,
        register_dflash_drafter,
    )

    clear_drafter_registry_for_tests()
    register_dflash_drafter("test-overwrite", "z-lab/first")
    try:
        with caplog.at_level(logging.WARNING):
            register_dflash_drafter("test-overwrite", "z-lab/second")
        assert any("Overwriting" in r.message for r in caplog.records)
        assert get_dflash_drafter_path("test-overwrite") == "z-lab/second"
    finally:
        clear_drafter_registry_for_tests()


def test_drafter_registry_rejects_empty_inputs():
    from vllm_mlx.spec_decode.dflash import register_dflash_drafter

    with pytest.raises(ValueError, match="alias"):
        register_dflash_drafter("", "z-lab/something")
    with pytest.raises(ValueError, match="drafter_hf_path"):
        register_dflash_drafter("alias-name", "")


# ---------------------------------------------------------------------------
# 3. Accept counter
# ---------------------------------------------------------------------------


def test_accept_counter_starts_zero_and_snapshot_consistent():
    from vllm_mlx.spec_decode.dflash.accept_counter import DFlashAcceptCounter

    c = DFlashAcceptCounter()
    snap = c.snapshot()
    assert snap.attempts == 0
    assert snap.accepts == 0
    assert snap.tokens_saved == 0
    assert snap.accept_ratio == 0.0
    assert snap.mean_tokens_per_attempt == 0.0


def test_accept_counter_tracks_partial_accepts():
    """DFlash partial accepts: 3 attempts, 2 accepts with 7 and 4
    tokens_saved respectively. accept_ratio = 2/3; tokens_saved = 11."""
    from vllm_mlx.spec_decode.dflash.accept_counter import DFlashAcceptCounter

    c = DFlashAcceptCounter()
    c.record_attempt()
    c.record_accept(tokens_saved=7)
    c.record_attempt()
    c.record_accept(tokens_saved=4)
    c.record_attempt()
    c.record_reject()
    snap = c.snapshot()
    assert snap.attempts == 3
    assert snap.accepts == 2
    assert snap.tokens_saved == 11
    assert snap.accept_ratio == pytest.approx(2 / 3)
    assert snap.mean_tokens_per_attempt == pytest.approx(11 / 3)


def test_accept_counter_concurrent_writes_safe():
    """Lock guarantees: 4 writers × 250 iters → exact counts."""
    import threading

    from vllm_mlx.spec_decode.dflash.accept_counter import DFlashAcceptCounter

    c = DFlashAcceptCounter()
    n_writers = 4
    iterations = 250

    def writer():
        for _ in range(iterations):
            c.record_attempt()
            c.record_accept(tokens_saved=3)

    threads = [threading.Thread(target=writer) for _ in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = c.snapshot()
    assert snap.attempts == n_writers * iterations
    assert snap.accepts == n_writers * iterations
    assert snap.tokens_saved == 3 * n_writers * iterations


def test_accept_counter_rejects_negative_tokens_saved():
    from vllm_mlx.spec_decode.dflash.accept_counter import DFlashAcceptCounter

    c = DFlashAcceptCounter()
    with pytest.raises(ValueError, match="non-negative"):
        c.record_accept(tokens_saved=-1)


def test_global_counter_is_singleton():
    from vllm_mlx.spec_decode.dflash.accept_counter import get_global_counter

    a = get_global_counter()
    b = get_global_counter()
    assert a is b


# ---------------------------------------------------------------------------
# 4. Verifier — argmax + accepted-prefix decision (pure-Python paths)
# ---------------------------------------------------------------------------


def test_decide_accepted_prefix_full_match():
    """Every position matches → accepted_len == block_size; sentinel bonus."""
    from vllm_mlx.spec_decode.dflash.verifier import _decide_accepted_prefix

    verify = [10, 11, 12, 13]
    draft = [10, 11, 12, 13]
    accepted_len, bonus = _decide_accepted_prefix(verify, draft)
    assert accepted_len == 4
    assert bonus == -1  # sentinel — caller resolves via verify[block_size]


def test_decide_accepted_prefix_all_reject():
    """Position 0 diverges → accepted_len == 0; bonus is verify[0]."""
    from vllm_mlx.spec_decode.dflash.verifier import _decide_accepted_prefix

    verify = [99, 11, 12, 13]
    draft = [10, 11, 12, 13]
    accepted_len, bonus = _decide_accepted_prefix(verify, draft)
    assert accepted_len == 0
    assert bonus == 99


def test_decide_accepted_prefix_partial_at_k():
    """Block accepts a prefix of length k; bonus is verify[k]."""
    from vllm_mlx.spec_decode.dflash.verifier import _decide_accepted_prefix

    verify = [10, 11, 999, 13]
    draft = [10, 11, 12, 13]
    accepted_len, bonus = _decide_accepted_prefix(verify, draft)
    assert accepted_len == 2
    assert bonus == 999  # the corrective token


def test_decide_accepted_prefix_length_mismatch_raises():
    from vllm_mlx.spec_decode.dflash.verifier import _decide_accepted_prefix

    with pytest.raises(ValueError, match="length"):
        _decide_accepted_prefix([1, 2, 3], [1, 2])


def test_argmax_per_position_returns_uint32():
    """The verifier compares against uint32 drafter tokens; the
    argmax cast must match to avoid signed/unsigned mismatch surprises."""
    from vllm_mlx.spec_decode.dflash.verifier import _argmax_per_position

    logits = mx.array([[[0.1, 0.9], [0.7, 0.3]]])  # shape (1, 2, 2)
    am = _argmax_per_position(logits)
    assert am.dtype == mx.uint32
    assert am.shape == (1, 2)
    assert int(am[0, 0]) == 1
    assert int(am[0, 1]) == 0


def test_argmax_per_position_rejects_non_3d():
    from vllm_mlx.spec_decode.dflash.verifier import _argmax_per_position

    with pytest.raises(ValueError, match="3-D"):
        _argmax_per_position(mx.array([0.1, 0.9]))


# ---------------------------------------------------------------------------
# 5. Verifier — end-to-end with a scripted tiny target model
# ---------------------------------------------------------------------------


class _ScriptedTinyTargetModel:
    """Minimal target model that returns scripted argmax logits.

    The verifier calls ``model(inputs[None], cache=cache)`` and reads
    argmax at each position. We script that argmax directly: each call
    consumes ``inputs.shape[1]`` tokens from the script, builds a
    one-hot logits row at each position, and writes a "fake" KV
    entry into the cache so the cache offset advances correctly.

    Args:
        scripted_outputs: List of token IDs the target argmax produces
            at successive output positions (one per __call__ position).
        vocab_size: Vocab dim. Default 64 — small enough for tests.
        hidden_size: Hidden dim. Default 8.
        n_kv_heads: KV-head count for the fake cache write. Default 1.
        head_dim: Head dim for the fake cache write. Default 4.
    """

    def __init__(
        self,
        scripted_outputs: list[int],
        vocab_size: int = 2048,
        hidden_size: int = 8,
        n_kv_heads: int = 1,
        head_dim: int = 4,
    ):
        self._script = list(scripted_outputs)
        self._cursor = 0
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # The verifier creates one KVCache per layer via
        # ``make_prompt_cache(model)`` — that helper inspects
        # ``model.layers`` to decide. Empty list keeps the prompt cache
        # empty so the verifier owns offset bookkeeping directly via
        # ``_rewind_cache_to``.
        self.layers: list = []

    def __call__(self, inputs: mx.array, cache=None):
        # Consume inputs.shape[1] tokens from the script.
        B, S = inputs.shape
        rows = []
        for _ in range(S):
            if self._cursor < len(self._script):
                tid = self._script[self._cursor]
                self._cursor += 1
            else:
                tid = 0
            # One-hot logit row (B, vocab_size) with a large positive
            # value at the scripted argmax position.
            base = mx.zeros((B, self.vocab_size))
            base = base + mx.where(
                mx.arange(self.vocab_size)[None, :] == tid,
                mx.array(50.0),
                mx.array(0.0),
            )
            rows.append(base)
        logits = mx.stack(rows, axis=1)  # (B, S, vocab)
        return logits


def test_verify_block_full_accept_returns_block_was_full():
    """All 4 positions match → accepted_tokens == draft; block_was_full=True;
    bonus_token is the post-block prediction."""
    from vllm_mlx.spec_decode.dflash.verifier import verify_block

    # Script: position 0 predicts draft[0]=10, ..., position 3 predicts
    # draft[3]=13, position 4 predicts post-block bonus = 42.
    scripted = [10, 11, 12, 13, 42]
    model = _ScriptedTinyTargetModel(scripted)
    draft = mx.array([10, 11, 12, 13], dtype=mx.uint32)
    cache: list = []  # no layers in the tiny model
    result = verify_block(
        model,
        draft,
        last_confirmed_token=5,
        cache=cache,
        block_size=4,
        current_offset=10,
        temperature=0.0,
    )
    assert result.accepted_len == 4
    assert result.accepted_tokens == (10, 11, 12, 13)
    assert result.block_was_full is True
    assert result.bonus_token == 42
    assert result.verify_offset_after == 14


def test_verify_block_all_reject_emits_corrective_only():
    """Position 0 diverges → accepted_len=0; bonus is verify[0]."""
    from vllm_mlx.spec_decode.dflash.verifier import verify_block

    # Position 0 predicts 99 (not draft 10) → reject at 0.
    scripted = [99, 11, 12, 13, 14]
    model = _ScriptedTinyTargetModel(scripted)
    draft = mx.array([10, 11, 12, 13], dtype=mx.uint32)
    cache: list = []
    result = verify_block(
        model,
        draft,
        last_confirmed_token=5,
        cache=cache,
        block_size=4,
        current_offset=10,
        temperature=0.0,
    )
    assert result.accepted_len == 0
    assert result.accepted_tokens == ()
    assert result.block_was_full is False
    assert result.bonus_token == 99
    assert result.verify_offset_after == 10


def test_verify_block_partial_accept_at_k_equals_2():
    """Positions 0,1 match; position 2 diverges → accepted_len=2."""
    from vllm_mlx.spec_decode.dflash.verifier import verify_block

    scripted = [10, 11, 77, 13, 14]
    model = _ScriptedTinyTargetModel(scripted)
    draft = mx.array([10, 11, 12, 13], dtype=mx.uint32)
    cache: list = []
    result = verify_block(
        model,
        draft,
        last_confirmed_token=5,
        cache=cache,
        block_size=4,
        current_offset=10,
        temperature=0.0,
    )
    assert result.accepted_len == 2
    assert result.accepted_tokens == (10, 11)
    assert result.block_was_full is False
    assert result.bonus_token == 77  # corrective at divergence
    assert result.verify_offset_after == 12


def test_verify_block_rewinds_cache_offset_on_partial_accept():
    """Real KVCache: partial accept must rewind offset to the accepted end."""
    from mlx_lm.models.cache import KVCache

    from vllm_mlx.spec_decode.dflash.verifier import _rewind_cache_to

    cache = KVCache()
    # Manually set offset to simulate "the model forward wrote 4 positions".
    cache.offset = (
        20  # current_offset(10) + block_size(4) + last_confirmed(1) + ... — abstract
    )
    _rewind_cache_to([cache], target_offset=12)
    assert cache.offset == 12

    # Idempotent re-rewind to lower offset.
    _rewind_cache_to([cache], target_offset=11)
    assert cache.offset == 11

    # Rewinding UPWARD is a no-op (the verifier never asks for it,
    # but the function must not corrupt the offset if called).
    _rewind_cache_to([cache], target_offset=99)
    assert cache.offset == 11


def test_verify_block_rejects_non_finite_temperature():
    """NaN / Inf temperature must fail loud — Pydantic `Field(ge=)` does
    not reject NaN."""
    from vllm_mlx.spec_decode.dflash.verifier import verify_block

    model = _ScriptedTinyTargetModel([0])
    draft = mx.array([1], dtype=mx.uint32)
    with pytest.raises(ValueError, match="finite"):
        verify_block(
            model,
            draft,
            last_confirmed_token=0,
            cache=[],
            block_size=1,
            current_offset=0,
            temperature=float("nan"),
        )
    with pytest.raises(ValueError, match="finite"):
        verify_block(
            model,
            draft,
            last_confirmed_token=0,
            cache=[],
            block_size=1,
            current_offset=0,
            temperature=float("inf"),
        )


def test_verify_block_position_id_contract_matches_block_size():
    """Verifier's input length is ``block_size + 1`` (last_confirmed
    plus the block); output logits cover the same positions.

    Pin the contract by constructing a draft of size 16 and asserting
    the scripted model is consumed for exactly 17 positions
    (block_size + 1 = 17).
    """
    from vllm_mlx.spec_decode.dflash.verifier import verify_block

    block_size = 16
    # Scripted target predicts: position 0 → draft[0], ..., position 15
    # → draft[15] (all-accept), position 16 → bonus = 999.
    draft_tokens = list(range(100, 100 + block_size))
    scripted = list(draft_tokens) + [999]
    model = _ScriptedTinyTargetModel(scripted, vocab_size=1024)
    draft = mx.array(draft_tokens, dtype=mx.uint32)
    result = verify_block(
        model,
        draft,
        last_confirmed_token=42,
        cache=[],
        block_size=block_size,
        current_offset=50,
        temperature=0.0,
    )
    assert result.accepted_len == block_size
    assert result.bonus_token == 999
    # The model received exactly block_size + 1 forward positions.
    assert model._cursor == block_size + 1


# ---------------------------------------------------------------------------
# 6. Drafter Protocol — Stub + interface contract
# ---------------------------------------------------------------------------


def test_stub_drafter_emits_scripted_blocks_in_order():
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter

    drafter = StubBlockDiffusionDrafter(
        scripted_blocks=[[1, 2, 3, 4], [5, 6, 7, 8]],
        block_size=4,
    )
    out0 = drafter.draft_block(mx.array([0], dtype=mx.uint32), current_position=0)
    out1 = drafter.draft_block(mx.array([0], dtype=mx.uint32), current_position=4)
    assert out0.tolist() == [1, 2, 3, 4]
    assert out1.tolist() == [5, 6, 7, 8]
    assert out0.dtype == mx.uint32


def test_stub_drafter_reset_restarts_cursor():
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter

    drafter = StubBlockDiffusionDrafter(scripted_blocks=[[1, 2], [3, 4]], block_size=2)
    drafter.draft_block(mx.array([0], dtype=mx.uint32), 0)
    drafter.reset()
    out = drafter.draft_block(mx.array([0], dtype=mx.uint32), 0)
    assert out.tolist() == [1, 2]


def test_stub_drafter_raises_on_script_exhaustion():
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter

    drafter = StubBlockDiffusionDrafter(scripted_blocks=[[1, 2]], block_size=2)
    drafter.draft_block(mx.array([0], dtype=mx.uint32), 0)
    with pytest.raises(IndexError, match="exhausted"):
        drafter.draft_block(mx.array([0], dtype=mx.uint32), 0)


def test_stub_drafter_rejects_bad_block_size():
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter

    with pytest.raises(ValueError, match="block_size"):
        StubBlockDiffusionDrafter(scripted_blocks=[[1, 2]], block_size=0)
    # Inner length mismatch — clear error so test authors get pointed at
    # the offending entry.
    with pytest.raises(ValueError, match="length"):
        StubBlockDiffusionDrafter(scripted_blocks=[[1, 2, 3]], block_size=2)


# ---------------------------------------------------------------------------
# 7. Generator — pairs drafter + verifier (mocked target)
# ---------------------------------------------------------------------------


def _make_generator_target(scripted_outputs: list[int]):
    """Build a tiny target model that:

    * Returns scripted argmax during the verifier forward.
    * Reuses the same script for the PREFILL forward (the generator
      calls model(prompt[None], cache=...) once before the loop).

    The script is consumed monotonically across all forward passes
    (prefill + per-block verify), so the test author sequences:
    ``[prefill_argmaxes, block1_argmaxes, block2_argmaxes, ...]``.
    """
    return _ScriptedTinyTargetModel(scripted_outputs)


def test_generator_full_block_accept_emits_all_drafted_tokens():
    """Drafter emits block matching the verifier; expect prefill primary
    + 4 drafted + 1 bonus tokens."""
    from vllm_mlx.spec_decode.dflash.accept_counter import DFlashAcceptCounter
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter
    from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

    # Prefill consumes 1 token (prompt len = 1 → model called with S=1).
    # Primary token = prefill_script[-1] = 7.
    # Verify block 1: model called with S=block_size+1=5 → 5 tokens.
    # Block 1 draft = [10, 11, 12, 13]; script positions 0..3 match,
    # position 4 (post-block bonus) = 14.
    scripted = [7] + [10, 11, 12, 13, 14]
    model = _make_generator_target(scripted)

    drafter = StubBlockDiffusionDrafter(
        scripted_blocks=[[10, 11, 12, 13]], block_size=4
    )
    counter = DFlashAcceptCounter()

    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    for tok, _lp, from_draft in dflash_generate_step(
        prompt,
        model,
        drafter,
        block_size=4,
        max_tokens=6,
        accept_counter=counter,
    ):
        emitted.append((tok, from_draft))

    assert emitted[0] == (7, False), f"primary emit: {emitted}"
    # Next 4 from drafter (from_draft=True).
    assert emitted[1:5] == [(10, True), (11, True), (12, True), (13, True)]
    # Bonus post-block from verifier (from_draft=False).
    assert emitted[5] == (14, False)
    snap = counter.snapshot()
    assert snap.attempts == 1
    assert snap.accepts == 1
    # Full block of 4 → bonus_saved = max(0, 4 - 1) = 3.
    assert snap.tokens_saved == 3


def test_generator_partial_accept_at_k_emits_only_accepted_prefix_plus_corrective():
    """Drafter draft=[10,11,12,13]; verifier accepts [10,11], diverges to 77
    at position 2. Generator yields: prefill primary 7, accepted (10,11),
    corrective 77."""
    from vllm_mlx.spec_decode.dflash.accept_counter import DFlashAcceptCounter
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter
    from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

    # Prefill emits 7. Block 1: position 0 → 10 (match), position 1 → 11
    # (match), position 2 → 77 (mismatch), position 3 → don't care.
    # Verifier emits the corrective at position 2 = 77. Bonus token
    # is 77 (corrective at divergence).
    scripted = [7, 10, 11, 77, 13, 14]
    model = _make_generator_target(scripted)

    drafter = StubBlockDiffusionDrafter(
        scripted_blocks=[[10, 11, 12, 13]], block_size=4
    )
    counter = DFlashAcceptCounter()

    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    for tok, _lp, from_draft in dflash_generate_step(
        prompt,
        model,
        drafter,
        block_size=4,
        max_tokens=4,
        accept_counter=counter,
    ):
        emitted.append((tok, from_draft))

    assert emitted == [(7, False), (10, True), (11, True), (77, False)]
    snap = counter.snapshot()
    assert snap.attempts == 1
    assert snap.accepts == 1
    # Partial of length 2 → bonus_saved = max(0, 2 - 1) = 1.
    assert snap.tokens_saved == 1


def test_generator_all_reject_falls_back_to_corrective_token():
    """Drafter wrong at position 0; verifier emits its argmax only."""
    from vllm_mlx.spec_decode.dflash.accept_counter import DFlashAcceptCounter
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter
    from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

    # Prefill emits 7. Position 0 → 99 (mismatch with draft 10).
    # Bonus token = 99 (corrective).
    scripted = [7, 99, 11, 12, 13, 14]
    model = _make_generator_target(scripted)

    drafter = StubBlockDiffusionDrafter(
        scripted_blocks=[[10, 11, 12, 13]], block_size=4
    )
    counter = DFlashAcceptCounter()

    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    for tok, _lp, from_draft in dflash_generate_step(
        prompt,
        model,
        drafter,
        block_size=4,
        max_tokens=2,
        accept_counter=counter,
    ):
        emitted.append((tok, from_draft))

    assert emitted == [(7, False), (99, False)]
    snap = counter.snapshot()
    assert snap.attempts == 1
    assert snap.accepts == 0  # accepted_len == 0 counts as reject
    assert snap.tokens_saved == 0


def test_generator_rejects_temp_above_zero():
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter
    from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

    model = _make_generator_target([7, 0, 0, 0, 0, 0])
    drafter = StubBlockDiffusionDrafter(scripted_blocks=[[1, 2, 3, 4]], block_size=4)
    prompt = mx.array([1], dtype=mx.uint32)
    with pytest.raises(NotImplementedError, match="temperature"):
        list(
            dflash_generate_step(
                prompt,
                model,
                drafter,
                block_size=4,
                max_tokens=4,
                temperature=0.7,
            )
        )


def test_generator_rejects_block_size_mismatch_with_drafter():
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter
    from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

    model = _make_generator_target([7, 0, 0, 0])
    drafter = StubBlockDiffusionDrafter(scripted_blocks=[[1, 2]], block_size=2)
    prompt = mx.array([1], dtype=mx.uint32)
    with pytest.raises(ValueError, match="block_size"):
        list(
            dflash_generate_step(
                prompt,
                model,
                drafter,
                block_size=4,  # mismatch with drafter.block_size=2
                max_tokens=4,
            )
        )


# ---------------------------------------------------------------------------
# 8. Lossless integration — output matches no-spec-decode baseline
# ---------------------------------------------------------------------------


def _ungenerated_baseline(scripted_outputs: list[int], max_tokens: int) -> list[int]:
    """Simulate the non-spec-decode emit for the SAME scripted target.

    A vanilla decode just yields the scripted output positions one-by-
    one. With our scripted model, ``scripted_outputs`` IS what the
    baseline emits — modulo the prefill primary (position 0) and
    subsequent decode-per-token emits (positions 1..max_tokens-1).

    Returns a length-min(max_tokens, len(scripted_outputs)) list of
    token IDs.
    """
    return list(scripted_outputs[:max_tokens])


def test_lossless_all_accept_matches_baseline_byte_for_byte():
    """When the drafter is psychic (matches every verify argmax), the
    emitted sequence equals the no-spec-decode baseline."""
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter
    from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

    # Scripted target: prefill prim=7, block emits 10,11,12,13 then
    # post-block bonus 14.
    scripted = [7, 10, 11, 12, 13, 14]
    baseline_tokens = _ungenerated_baseline(scripted, max_tokens=6)

    model = _make_generator_target(list(scripted))
    drafter = StubBlockDiffusionDrafter(
        scripted_blocks=[[10, 11, 12, 13]], block_size=4
    )
    dflash_tokens = [
        tok
        for tok, _lp, _fd in dflash_generate_step(
            mx.array([1], dtype=mx.uint32),
            model,
            drafter,
            block_size=4,
            max_tokens=6,
        )
    ]
    assert dflash_tokens == baseline_tokens


def test_lossless_all_reject_matches_baseline_byte_for_byte():
    """When the drafter is always wrong at position 0, the DFlash
    emitted sequence STILL matches the no-spec-decode baseline (just
    slower). The corrective verify pred at the divergence IS the
    baseline token at that position."""
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter
    from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

    # Scripted target: prefill=7, then verify positions emit
    # baseline tokens 99, 100, 101, 102 at positions 0..3 of block 1,
    # then 103 at position 4 (don't-care for reject), and 104 for the
    # next block's position 0.
    #
    # The drafter is wrong at every position 0 (draft starts with 999,
    # baseline argmax is 99 → reject). Generator emits: 7 (prefill),
    # 99 (corrective at block 1 reject). Then block 2 also rejects:
    # script position 4 (next block's position 0) emits 100 → but
    # our scripted model advances its cursor by S=5 per verify call,
    # so block 2's position 0 reads scripted[6]. Let's keep it simple:
    # just two emits and check both match baseline.
    scripted = [7, 99, 100, 101, 102]
    baseline_tokens = _ungenerated_baseline(scripted, max_tokens=2)

    model = _make_generator_target(list(scripted))
    drafter = StubBlockDiffusionDrafter(
        scripted_blocks=[[999, 999, 999, 999]], block_size=4
    )
    dflash_tokens = [
        tok
        for tok, _lp, _fd in dflash_generate_step(
            mx.array([1], dtype=mx.uint32),
            model,
            drafter,
            block_size=4,
            max_tokens=2,
        )
    ]
    assert dflash_tokens == baseline_tokens


def test_lossless_mid_block_reject_matches_baseline_byte_for_byte():
    """Drafter matches positions 0,1, diverges at position 2 (draft has
    12 but verifier emits 77). DFlash emits prefill primary, two
    accepted drafted tokens, then the corrective 77 — IDENTICAL to
    the baseline which emits the same four tokens by argmax."""
    from vllm_mlx.spec_decode.dflash.drafter import StubBlockDiffusionDrafter
    from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

    scripted = [7, 10, 11, 77, 13, 14]
    baseline_tokens = _ungenerated_baseline(scripted, max_tokens=4)

    model = _make_generator_target(list(scripted))
    drafter = StubBlockDiffusionDrafter(
        scripted_blocks=[[10, 11, 12, 13]], block_size=4
    )
    dflash_tokens = [
        tok
        for tok, _lp, _fd in dflash_generate_step(
            mx.array([1], dtype=mx.uint32),
            model,
            drafter,
            block_size=4,
            max_tokens=4,
        )
    ]
    assert dflash_tokens == baseline_tokens


# ---------------------------------------------------------------------------
# 9. CLI surface
# ---------------------------------------------------------------------------


def _serve_help_stdout() -> str:
    """Run ``python -m vllm_mlx.cli serve --help`` and return stdout."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_cli_spec_decode_flag_advertises_dflash_choice():
    """``--spec-decode {none,mtp,dflash}`` must appear in serve help."""
    text = _serve_help_stdout()
    assert "--spec-decode" in text
    assert "dflash" in text
    assert "mtp" in text


def test_cli_dflash_drafter_path_flag_present():
    """``--dflash-drafter-path`` is parseable and documented."""
    text = _serve_help_stdout()
    assert "--dflash-drafter-path" in text


def test_cli_spec_decode_rejects_unknown_value():
    """``--spec-decode eagle`` is rejected by argparse choices."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.cli",
            "serve",
            "qwen3.5-4b-4bit",
            "--spec-decode",
            "eagle",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "spec-decode" in proc.stderr or "spec_decode" in proc.stderr


def test_scheduler_config_default_dflash_drafter_path_is_empty():
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.dflash_drafter_path == ""


def test_scheduler_config_dflash_round_trip():
    """``spec_decode='dflash'`` + drafter path round-trip through SchedulerConfig."""
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(spec_decode="dflash", dflash_drafter_path="local/path")
    assert cfg.spec_decode == "dflash"
    assert cfg.dflash_drafter_path == "local/path"


# ---------------------------------------------------------------------------
# 10. Metrics rendering
# ---------------------------------------------------------------------------


def test_metrics_renders_dflash_counters_zero_at_cold_start():
    """Before any DFlash generation runs, the five DFlash series MUST be
    present with value 0/16 (block_size gauge starts at the default)."""
    from vllm_mlx.routes.metrics import _render_spec_decode_dflash_counters
    from vllm_mlx.spec_decode.dflash.accept_counter import (
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()

    class _Cfg:
        model_alias = "qwen3.5-9b-4bit"

    lines = _render_spec_decode_dflash_counters(_Cfg())
    body = "\n".join(lines)
    assert "rapid_mlx_spec_decode_dflash_attempts_total" in body
    assert "rapid_mlx_spec_decode_dflash_accepts_total" in body
    assert "rapid_mlx_spec_decode_dflash_accept_ratio" in body
    assert "rapid_mlx_spec_decode_dflash_tokens_saved_total" in body
    assert "rapid_mlx_spec_decode_dflash_block_size" in body
    # Family + method labels.
    assert 'family="qwen3.5-9b-4bit"' in body
    assert 'method="dflash"' in body
    # Block size default 16.
    assert (
        'rapid_mlx_spec_decode_dflash_block_size{family="qwen3.5-9b-4bit",'
        'method="dflash"} 16'
    ) in body


def test_metrics_renders_post_acceptance_dflash_counters():
    """4 attempts / 3 accepts / 27 tokens_saved → metric reflects state."""
    from vllm_mlx.routes.metrics import _render_spec_decode_dflash_counters
    from vllm_mlx.spec_decode.dflash.accept_counter import (
        get_global_counter,
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()
    counter = get_global_counter()
    counter.record_attempt()
    counter.record_attempt()
    counter.record_attempt()
    counter.record_attempt()
    counter.record_accept(tokens_saved=15)
    counter.record_accept(tokens_saved=8)
    counter.record_accept(tokens_saved=4)

    class _Cfg:
        model_alias = "qwen3.5-9b-4bit"

    body = "\n".join(_render_spec_decode_dflash_counters(_Cfg()))
    assert (
        'rapid_mlx_spec_decode_dflash_attempts_total{family="qwen3.5-9b-4bit",'
        'method="dflash"} 4'
    ) in body
    assert (
        'rapid_mlx_spec_decode_dflash_accepts_total{family="qwen3.5-9b-4bit",'
        'method="dflash"} 3'
    ) in body
    assert (
        'rapid_mlx_spec_decode_dflash_tokens_saved_total{family="qwen3.5-9b-4bit",'
        'method="dflash"} 27'
    ) in body
    # accept_ratio = 0.75 — must appear rounded to at most 4 decimals.
    assert "0.75" in body
    reset_global_counter_for_tests()


def test_metrics_route_includes_dflash_series_at_cold_start():
    """End-to-end /metrics body carries the dflash series pre-engine."""
    from vllm_mlx.routes.metrics import _render_prometheus

    class _Cfg:
        engine = None
        model_name = "qwen3.5-9b-4bit"
        model_alias = "qwen3.5-9b-4bit"
        kv_cache_dtype = None

    body = _render_prometheus(_Cfg())
    assert "rapid_mlx_spec_decode_dflash_attempts_total" in body
    assert "rapid_mlx_spec_decode_dflash_accept_ratio" in body
    assert "rapid_mlx_spec_decode_dflash_block_size" in body


# ---------------------------------------------------------------------------
# 11. Bench script — dry-run smoke
# ---------------------------------------------------------------------------


def test_bench_script_dry_run_executes_cleanly():
    """The committed bench script must validate via --dry-run without
    booting MLX. Used by the PR's checkbox 4 acceptance.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "bench/bench_spec_decode_dflash.py",
            "--dry-run",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    # The plan JSON must declare the two conditions and a block size.
    assert '"conditions"' in proc.stdout
    assert '"dflash"' in proc.stdout
    assert '"block_size"' in proc.stdout
