# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the vendored MTP speculative decode bundle (R15-P1 #302).

Coverage:

* Architecture detection (Qwen3.5 / 3.6 only; closed alias schema bypass)
* Accept-rate counter (record_attempt / record_accept / record_reject,
  snapshot consistency, ratio computation, reset semantics)
* ``ArraysCache.rollback_state`` slot patch (idempotent install, future-
  proof guard against upstream merging the same change)
* CLI flag parsing (``--spec-decode mtp|none``) + SchedulerConfig
  plumbing
* Metrics rendering (``rapid_mlx_spec_decode_*``)
* MTP head builder (constructs without weight load)
* Qwen3.5/3.6 model-side injection helper (uses a synthetic model
  shell so we don't have to load real Qwen3.5 weights)
* Generator loop verify/accept logic via a mocked model (chain MTP
  end-to-end without booting MLX-GPU)

The tests intentionally avoid loading a real Qwen3.5 / Qwen3.6
checkpoint — those are 4-50 GB downloads and the lossless integration
test in ``tests/test_mtp_lossless.py`` exercises the loop with a
deterministic mocked model that lets us assert byte-identical output
without GPU contention (R15-P1 #302 explicitly defers the GPU bench
because Stage B Viterbi is currently holding the device).
"""

from __future__ import annotations

import copy
import dataclasses
import textwrap
import types

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _reset_mtp_module_state():
    """Reset the MTP module-level singletons AND ``mlx_lm.generate``'s
    captured ``generation_stream`` between tests.

    Three pieces of cross-test state leak in the full pytest sweep and
    surface as the 7-failure transient cluster (PASS in isolation):

    * ``rapid_mlx.spec_decode.mtp.cache_patch._patched`` — sticky install
      gate; ``_unpatch_for_tests()`` clears it.
    * ``rapid_mlx.spec_decode.mtp.accept_counter._global_counter`` —
      monotonic counter singleton (monotonicity is a public contract);
      ``reset_global_counter_for_tests()`` is the explicit hatch.
    * **``mlx_lm.generate.generation_stream``** — the module-level
      ``generation_stream`` is created at import time via
      ``mx.new_thread_local_stream(...)`` (bound to the importer
      thread) and is then re-assigned by every call to
      ``engine_core._init_mlx_step_thread`` to ``mx.default_stream(
      mx.default_device())``. Crucially — and contrary to the name —
      ``mx.default_stream(device)`` returns the **current thread's**
      default stream, NOT a process-wide stream. So when a preceding
      sweep test (``test_batching_deterministic``, ``test_batching``,
      ``test_mllm_*``) spins up a ``mlx-step`` worker executor with
      ``initializer=_init_mlx_step_thread``, the worker's default
      stream gets stamped onto ``mlx_lm.generate.generation_stream``.
      When the worker shuts down and the pytest main thread later
      runs ``mtp_generate_step``, its ``with mx.stream(
      generation_stream): mx.eval(toks)`` block at
      ``generator.py:420`` crashes with ``RuntimeError: There is no
      Stream(gpu, N) in current thread.``

      The canonical fix is to re-bind ``generation_stream`` to **this
      thread's** default stream at fixture setup. This mirrors what
      ``_init_mlx_step_thread`` does for the executor worker, just
      pinned to the pytest main thread.

    (Prior fix attempted ``mx.set_default_stream(mx.new_stream(
    mx.default_device()))`` — that only resets the active default for
    the current thread, NOT ``mlx_lm.generate.generation_stream`` which
    is what ``mtp_generate_step`` actually uses. It also reintroduced
    ``mx.new_stream`` — a thread-bound allocator the production code
    deliberately avoids per
    ``tests/test_mllm_cross_thread_stream_contract.py``.)
    """
    import sys

    import mlx.core as mx

    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from rapid_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests

    _unpatch_for_tests()
    reset_global_counter_for_tests()
    # Re-bind ``mlx_lm.generate.generation_stream`` to the pytest main
    # thread's default stream. Some preceding sweep test may have left
    # it pointing at a worker thread's stream (see fixture docstring
    # for the full chain). Importing ``mlx_lm.generate`` here is a
    # no-op if a prior test already imported it; we look it up via
    # ``sys.modules`` so we never import-trigger inside the fixture
    # for tests that don't end up calling ``mtp_generate_step``.
    import mlx_lm.generate  # noqa: F401 — ensure module exists in sys.modules

    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )
    yield
    _unpatch_for_tests()
    reset_global_counter_for_tests()
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )


# ---------------------------------------------------------------------------
# 1. Architecture detection
# ---------------------------------------------------------------------------


def test_detect_eligibility_qwen3_5_chain():
    """Qwen3.5 dense with mtp_num_hidden_layers=1 → CHAIN."""
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_eligibility_qwen3_5_moe_chain():
    """Qwen3.5 MoE with mtp_num_hidden_layers=1 → CHAIN (same path)."""
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_eligibility_qwen3_5_accepts_text_config_mtp_layers():
    """MLX community Qwen3.5/3.6 configs store MTP metadata in text_config."""
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
        },
    }
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_eligibility_qwen4_exp_accepts_nested_mtp_layer():
    """Flash-Next advertises its native MTP head in nested text_config."""
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "mtp_num_hidden_layers": 1,
        },
    }
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_eligibility_qwen3_5_tree_reserved():
    """mtp_num_hidden_layers >= 2 → TREE (reserved, not implemented)."""
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 4}
    assert detect_mtp_eligibility(config) is MTPEligibility.TREE


def test_detect_eligibility_non_qwen35_models_rejected():
    """Llama / Mistral / Qwen3 / Qwen3-Next must NOT match the MTP path."""
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
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
        config = {"model_type": model_type, "mtp_num_hidden_layers": 1}
        assert detect_mtp_eligibility(config) is MTPEligibility.NONE, (
            f"non-Qwen3.5 model_type={model_type} must NOT match MTP path "
            "(would risk wrong model architecture being patched)."
        )


def test_detect_eligibility_qwen3_5_stripped_checkpoint():
    """Qwen3.5 model with mtp_num_hidden_layers=0 (MTP weights stripped)
    must reject — operator gets a clear ``re-convert from HF`` hint.
    """
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 0}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


# ---------------------------------------------------------------------------
# 1b. Gemma 4 detection (assistant sidecar path currently disabled)
# ---------------------------------------------------------------------------
# Gemma 4 ships in two ``model_type`` flavours (verified against the
# cached mlx-community configs on 2026-07-01):
#
#   * ``gemma4_unified`` — text-only variant. ``Gemma4UnifiedForConditional
#     Generation``. Used by the 12B dense checkpoints
#     (``gemma-4-12B-it-4bit`` / ``gemma-4-12B-it-8bit``).
#   * ``gemma4`` — multimodal variant. ``Gemma4ForConditionalGeneration``.
#     Covers the effective-MoE ``gemma-4-26b-a4b-it-4bit`` and the small
#     vision-tower e2b / e4b checkpoints. INTENTIONALLY OFF the
#     allowlist today — a verified sidecar or assistant drafter for
#     this lineage has not landed; a hand-edited config that stamps
#     ``mtp_num_hidden_layers: 1`` on top must still be rejected so it
#     doesn't slip into an un-exercised inject path.
#
# Base checkpoints do NOT carry ``mtp_num_hidden_layers`` in their
# ``config.json`` (verified for all four cache probes). July 2026 A/B
# validation found greedy output divergence for the Google 12B assistant
# sidecar, so all Gemma 4 model_types must stay NONE regardless of
# ``mtp_num_hidden_layers`` until a future implementation proves lossless.


def test_detect_eligibility_gemma4_dense_unified_stays_none_even_with_mtp_layers():
    """Gemma 4 12B dense (``gemma4_unified``) stays NONE.

    A hand-edited config or sidecar-derived config may stamp
    ``mtp_num_hidden_layers=1``, but Gemma 4 MTP is not considered
    supported until the assistant-sidecar path passes greedy-lossless
    server A/B validation.
    """
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4_unified", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


def test_detect_eligibility_gemma4_dense_unified_stripped_none():
    """Gemma 4 12B dense with sidecar NOT applied (mtp=0) → NONE.

    Base ``mlx-community/gemma-4-12b-it-4bit`` ships without an MTP
    head; ``mtp_num_hidden_layers`` is either absent or 0. Detection
    must collapse to NONE so ``--spec-decode mtp`` is rejected at boot.
    """
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    # Explicit 0 (stripped/no-sidecar): reject.
    config_zero = {"model_type": "gemma4_unified", "mtp_num_hidden_layers": 0}
    assert detect_mtp_eligibility(config_zero) is MTPEligibility.NONE
    # Missing key (stock HF Gemma 4 shape): reject.
    config_missing = {"model_type": "gemma4_unified"}
    assert detect_mtp_eligibility(config_missing) is MTPEligibility.NONE


def test_detect_eligibility_gemma4_multimodal_not_on_allowlist_none():
    """Gemma 4 multimodal (``gemma4``) — even with mtp=1 → NONE.

    ``mlx-community/gemma-4-26b-a4b-it-4bit/config.json`` and the e2b /
    e4b checkpoints all report top-level ``model_type: gemma4`` (the
    ``Gemma4ForConditionalGeneration`` class). Neither the Mia-AiLab
    fp16-mtp sidecar nor Google's ``google/gemma-4-*-it-assistant``
    drafter has been verified against this lineage yet, so the detect
    allowlist INTENTIONALLY excludes ``gemma4`` — a hand-edited config
    that stamps ``mtp_num_hidden_layers: 1`` on top of a multimodal
    Gemma 4 must still collapse to NONE, so ``--spec-decode mtp`` is
    rejected pre-boot rather than routed into an inject/generator/cache
    path that hasn't been exercised for that architecture. Flip this
    once a verified sidecar or assistant drafter lands for the
    multimodal lineage.
    """
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


def test_detect_eligibility_gemma4_vision_tower_still_none():
    """Gemma 4 e2b / e4b (``gemma4`` with a vision tower) → still NONE.

    ``gemma-4-e2b-it-4bit`` and ``gemma-4-e4b-it-4bit`` ship as
    ``model_type: gemma4`` with a ``vision_config`` block, an
    ``audio_config`` block, and a ``text_config`` sub-config. Detection
    reads ONLY the top-level ``model_type`` string — presence of vision
    / audio fields must not alter the verdict either way. Since
    multimodal ``gemma4`` is not on the allowlist (see the sibling
    ``_multimodal_not_on_allowlist_none`` test), even a hand-edited
    ``mtp_num_hidden_layers: 1`` on top of a multimodal shape must land
    at NONE. This test stuffs the config with the real fields observed
    on those checkpoints (``vision_config``, ``audio_config``,
    ``image_token_id``, ``architectures``) to lock the "ignore
    sub-configs, gate on top-level model_type" contract.
    """
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {
        "model_type": "gemma4",
        "mtp_num_hidden_layers": 1,
        # Fields observed on the actual e2b / e4b / 26B-A4B configs.
        # Detection must ignore all of these.
        "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {"model_type": "siglip_vision_model"},
        "audio_config": {"model_type": "gemma4_audio"},
        "text_config": {"model_type": "gemma4_text"},
        "image_token_id": 262144,
    }
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


def test_detect_eligibility_gemma_lookalikes_still_rejected():
    """Gemma 2 / Gemma 3 (and other lookalikes) MUST remain NONE.

    Regression guard against a future refactor that switches the
    allowlist to a startswith check ("gemma") or a `.split('_')[0]`
    check. Gemma 3 in particular is close enough to Gemma 4 that
    getting confused would put the wrong model class through the MTP
    inject path.
    """
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    for model_type in (
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma3_moe",
        # A plausible-looking string an operator might scribble in
        # after seeing a Gemma-3 27B checkpoint dropped by a fine-tuner.
        # Detection is an exact allowlist match — must NOT allow.
        "gemma-3-27b-it",
    ):
        config = {"model_type": model_type, "mtp_num_hidden_layers": 1}
        assert detect_mtp_eligibility(config) is MTPEligibility.NONE, (
            f"Gemma lookalike model_type={model_type!r} must NOT match MTP "
            "path (would risk wrong model architecture being patched)."
        )


def test_detect_eligibility_handles_string_and_float_config():
    """Hand-edited / HF re-uploaded configs may carry strings / floats —
    detection coerces rather than crashing.
    """
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    assert (
        detect_mtp_eligibility({"model_type": "qwen3_5", "mtp_num_hidden_layers": "1"})
        is MTPEligibility.CHAIN
    )
    assert (
        detect_mtp_eligibility({"model_type": "qwen3_5", "mtp_num_hidden_layers": 1.0})
        is MTPEligibility.CHAIN
    )
    # Garbage falls back to NONE rather than crashing.
    assert (
        detect_mtp_eligibility(
            {"model_type": "qwen3_5", "mtp_num_hidden_layers": "garbage"}
        )
        is MTPEligibility.NONE
    )


def test_detect_eligibility_none_or_non_dict_returns_none():
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    assert detect_mtp_eligibility(None) is MTPEligibility.NONE
    assert detect_mtp_eligibility("not a dict") is MTPEligibility.NONE  # type: ignore[arg-type]
    assert detect_mtp_eligibility([]) is MTPEligibility.NONE  # type: ignore[arg-type]


def test_detect_eligibility_aliases_json_schema_untouched():
    """Detection MUST NOT depend on aliases.json fields like
    ``architecture``, ``family``, ``quantization``, ``notes`` — those
    are not in the closed-key schema and would silently break loading.
    The detector reads ``model_type`` from config.json (always present)
    and ``mtp_num_hidden_layers`` (also a real config.json field).

    This test pins the contract by passing an aliases.json-shaped
    dict that lacks those keys and asserting detection still works.
    """
    from rapid_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {
        "model_type": "qwen3_5",
        "mtp_num_hidden_layers": 1,
        # Note: NO architecture / family / quantization / notes here —
        # those would fail aliases.json schema validation if anyone
        # tried to back-port the detection into the alias profile.
    }
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


# ---------------------------------------------------------------------------
# 2. Accept-rate counter
# ---------------------------------------------------------------------------


def test_accept_counter_starts_zero_and_snapshot_is_consistent():
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    snap = counter.snapshot()
    assert snap.attempts == 0
    assert snap.accepts == 0
    assert snap.tokens_saved == 0
    assert snap.accept_ratio == 0.0  # zero attempts → 0 (not NaN)


def test_accept_counter_record_attempt_and_accept():
    """5 attempts, 3 accepts → ratio 0.6, tokens_saved = 3."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    for _ in range(5):
        counter.record_attempt()
    for _ in range(3):
        counter.record_accept(tokens_saved=1)
    snap = counter.snapshot()
    assert snap.attempts == 5
    assert snap.accepts == 3
    assert snap.tokens_saved == 3
    assert snap.accept_ratio == pytest.approx(0.6)


def test_accept_counter_reject_is_noop_for_counter_state():
    """``record_reject`` is symmetry-only — rejections are derived from
    ``attempts - accepts``. Calling reject must NOT bump any counter.
    """
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    counter.record_attempt()
    counter.record_reject()
    snap = counter.snapshot()
    assert snap.attempts == 1
    assert snap.accepts == 0
    assert snap.tokens_saved == 0


def test_accept_counter_rejects_negative_tokens_saved():
    """``record_accept(tokens_saved=-1)`` is a programmer error — fail loud."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    with pytest.raises(ValueError, match="non-negative"):
        counter.record_accept(tokens_saved=-1)


def test_accept_counter_reset_for_tests_resets_all_three():
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    counter.record_attempt()
    counter.record_accept(tokens_saved=2)
    counter.reset()
    snap = counter.snapshot()
    assert (snap.attempts, snap.accepts, snap.tokens_saved) == (0, 0, 0)


def test_global_counter_singleton_identity():
    """``get_global_counter`` returns the same instance across calls."""
    from rapid_mlx.spec_decode.mtp.accept_counter import get_global_counter

    a = get_global_counter()
    b = get_global_counter()
    assert a is b


def test_accept_counter_snapshot_under_concurrent_writes_is_safe():
    """Concurrent record_* calls must not corrupt the snapshot — the
    ``threading.Lock`` keeps the three fields in lockstep.
    """
    import threading

    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    n_writers = 4
    iterations = 250

    def writer():
        for _ in range(iterations):
            counter.record_attempt()
            counter.record_accept(tokens_saved=1)

    threads = [threading.Thread(target=writer) for _ in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = counter.snapshot()
    expected = n_writers * iterations
    assert snap.attempts == expected
    assert snap.accepts == expected
    assert snap.tokens_saved == expected
    assert snap.accept_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. ArraysCache.rollback_state slot patch
# ---------------------------------------------------------------------------


def test_cache_patch_installs_rollback_state_slot():
    """The patch lifts ``rollback_state`` from missing to a class
    attribute defaulting to ``None``.
    """
    from mlx_lm.models.cache import ArraysCache

    from rapid_mlx.spec_decode.mtp.cache_patch import (
        _is_patched_for_tests,
        _unpatch_for_tests,
        patch_arrays_cache_rollback_state,
    )

    _unpatch_for_tests()
    assert "rollback_state" not in ArraysCache.__dict__
    assert _is_patched_for_tests() is False

    applied = patch_arrays_cache_rollback_state()
    try:
        assert applied is True
        assert "rollback_state" in ArraysCache.__dict__
        assert ArraysCache.rollback_state is None  # type: ignore[attr-defined]
        assert _is_patched_for_tests() is True
    finally:
        # Re-install so other tests that depend on the patch (the
        # generator import already forced it) keep working.
        if not _is_patched_for_tests():
            patch_arrays_cache_rollback_state()


def test_snapshot_rollback_predicate_tracks_the_gated_delta_install():
    """The public predicate must report the live patch, not a static answer.

    ``gated_delta_snapshot_rollback_installed()`` is what the generator's
    admission check consults before it will trust an ``ArraysCache`` to be
    recoverable, so a predicate that answered ``True`` while the chunk-split
    verify was NOT installed would admit copy-drafts that cannot be rolled
    back. Toggle the install and require the answer to follow it.
    """
    from rapid_mlx.spec_decode.mtp.cache_patch import (
        _unpatch_for_tests,
        gated_delta_snapshot_rollback_installed,
        patch_arrays_cache_rollback_state,
        patch_gated_delta_net_for_mtp,
    )

    _unpatch_for_tests()
    try:
        assert gated_delta_snapshot_rollback_installed() is False
        if not patch_gated_delta_net_for_mtp():
            pytest.skip("GatedDeltaNet unavailable in this mlx-lm")
        assert gated_delta_snapshot_rollback_installed() is True
    finally:
        # Other tests (and the generator import) assume the patch is live.
        patch_arrays_cache_rollback_state()
        patch_gated_delta_net_for_mtp()


def test_cache_patch_is_idempotent():
    """Second call returns False — already-installed is not an error."""
    from rapid_mlx.spec_decode.mtp.cache_patch import (
        patch_arrays_cache_rollback_state,
    )

    # Force at least one install
    patch_arrays_cache_rollback_state()
    second = patch_arrays_cache_rollback_state()
    assert second is False


def test_qwen35_k3_verify_fuses_recurrence_without_changing_boundaries(
    monkeypatch,
):
    """The Metal K=3 path must match the position-wise rollback reference."""
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")

    from mlx_lm.models.cache import ArraysCache
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

    from rapid_mlx.kernels import qwen4_gdn_verify
    from rapid_mlx.spec_decode.mtp.cache_patch import (
        patch_gated_delta_net_for_mtp,
    )

    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        args = TextModelArgs(
            hidden_size=8,
            num_attention_heads=2,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=32,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=3,
        )
        layer = GatedDeltaNet(args)
        layer.set_dtype(mx.bfloat16)
        inputs = mx.random.normal(
            (1, 4, args.hidden_size), key=mx.random.key(3156)
        ).astype(mx.bfloat16)
        patch_gated_delta_net_for_mtp()

        reference_cache = ArraysCache(2)
        reference_cache.n_confirmed_for_mtp = 1
        layer.train(True)
        reference = layer(inputs, cache=reference_cache)
        mx.eval(
            reference,
            *reference_cache.cache,
            *(item for pair in reference_cache.rollback_state for item in pair),
        )

        fused_cache = ArraysCache(2)
        fused_cache.n_confirmed_for_mtp = 1
        fused_calls = 0
        original_fused = qwen4_gdn_verify.gated_delta_verify_with_states

        def counted_fused(*args, **kwargs):
            nonlocal fused_calls
            fused_calls += 1
            return original_fused(*args, **kwargs)

        monkeypatch.setattr(
            qwen4_gdn_verify,
            "gated_delta_verify_with_states",
            counted_fused,
        )
        layer.eval()
        fused = layer(inputs, cache=fused_cache)
        mx.eval(
            fused,
            *fused_cache.cache,
            *(item for pair in fused_cache.rollback_state for item in pair),
        )

        assert mx.array_equal(fused, reference).item()
        assert fused_calls == 1
        assert len(fused_cache.rollback_state) == 3
        for fused_value, reference_value in zip(
            fused_cache.cache, reference_cache.cache, strict=True
        ):
            assert mx.allclose(
                fused_value, reference_value, rtol=1e-5, atol=1e-6
            ).item()
        for fused_pair, reference_pair in zip(
            fused_cache.rollback_state,
            reference_cache.rollback_state,
            strict=True,
        ):
            for fused_value, reference_value in zip(
                fused_pair, reference_pair, strict=True
            ):
                assert mx.allclose(
                    fused_value, reference_value, rtol=1e-5, atol=1e-6
                ).item()
    finally:
        mx.set_default_device(previous_device)


# ---------------------------------------------------------------------------
# 4. CLI flag parsing + SchedulerConfig plumbing
# ---------------------------------------------------------------------------


def _serve_help_stdout() -> str:
    """Run ``python -m rapid_mlx.cli serve --help`` and return stdout.

    Mirrors :mod:`tests.test_kv_cache_dtype_cli` — the serve parser is
    inlined into ``main()``, so subprocess inspection is the canonical
    way to assert that the flag landed.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_cli_speculative_config_advertised_in_help():
    """MTP is exposed through ``--speculative-config`` only."""
    text = _serve_help_stdout()
    assert "--speculative-config" in text
    assert "--spec-decode" not in text
    assert '"method":"mtp"' in text


def test_cli_spec_decode_flag_is_hidden_but_recognized():
    """The old ``--spec-decode`` alias is hidden, but parser-compatible."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rapid_mlx.cli",
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
    assert "invalid choice" in proc.stderr
    assert "--spec-decode" in proc.stderr


def test_cli_spec_decode_mtp_legacy_choice_absent_from_help():
    """Deprecated ``--spec-decode mtp`` is absent, not merely hidden."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rapid_mlx.cli",
            "serve",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--spec-decode" not in proc.stdout


def test_scheduler_config_default_spec_decode_is_none():
    from rapid_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.spec_decode == "none"


def test_scheduler_config_spec_decode_round_trip():
    """Field round-trips ``mtp`` from kwargs."""
    from rapid_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(spec_decode="mtp")
    assert cfg.spec_decode == "mtp"


def test_scheduler_config_spec_decode_suffix_translates_to_suffix_flag():
    from rapid_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(spec_decode="suffix")

    # PR #1050 codex R3: keep ``spec_decode`` as the canonical selector so
    # callers reading the value observe what they passed in; also flip the
    # legacy ``enable_suffix_decoding`` flag for downstream code that still
    # reads it.
    assert cfg.spec_decode == "suffix"
    assert cfg.enable_suffix_decoding is True


def test_scheduler_config_rejects_unknown_spec_decode():
    from rapid_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match="spec_decode='typo'.*not supported"):
        SchedulerConfig(spec_decode="typo")


def test_scheduler_config_translates_deprecated_mtp_kwargs():
    from rapid_mlx.scheduler import SchedulerConfig

    with pytest.warns(DeprecationWarning, match="enable_mtp=True"):
        cfg = SchedulerConfig(
            enable_mtp=True,
            mtp_num_draft_tokens=2,
        )

    assert cfg.spec_decode == "mtp"
    assert cfg.enable_mtp is True
    assert cfg.mtp_num_draft_tokens == 2
    assert cfg.mtp_max_k == 2


def test_scheduler_config_rejects_unsupported_migrated_mtp_optimistic():
    """PR #1050 hard-reject: mtp_optimistic under unified interface."""
    from rapid_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match="mtp_optimistic=True.*not supported"):
        SchedulerConfig(spec_decode="mtp", mtp_optimistic=True)


def test_scheduler_config_rejects_legacy_enable_mtp_with_optimistic():
    """PR #1050 hard-reject: legacy ``enable_mtp=True`` path also rejects
    ``mtp_optimistic=True`` because __post_init__ normalizes it to
    ``spec_decode='mtp'`` and the vendored installer ignores optimistic.
    """
    from rapid_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match="mtp_optimistic=True.*not supported"):
        SchedulerConfig(enable_mtp=True, mtp_optimistic=True)


def test_scheduler_config_rejects_deprecated_mtp_with_other_spec_decode():
    from rapid_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match="enable_mtp=True.*spec_decode='suffix'"):
        SchedulerConfig(enable_mtp=True, spec_decode="suffix")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"enable_mtp": True, "enable_suffix_decoding": True},
            "multiple speculative decoding methods.*mtp, suffix",
        ),
        (
            {"enable_mtp": True, "dflash_drafter_path": "local/draft"},
            "dflash_drafter_path=.*conflicts with spec_decode='mtp'",
        ),
    ],
)
def test_scheduler_config_rejects_deprecated_mtp_with_other_backends(kwargs, match):
    from rapid_mlx.scheduler import SchedulerConfig

    with (
        pytest.warns(DeprecationWarning, match="enable_mtp=True"),
        pytest.raises(ValueError, match=match),
    ):
        SchedulerConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"spec_decode": "dflash", "enable_suffix_decoding": True},
            "multiple speculative decoding methods.*dflash, suffix",
        ),
        (
            {"enable_suffix_decoding": True, "dflash_drafter_path": "local/draft"},
            "dflash_drafter_path=.*conflicts",
        ),
    ],
)
def test_scheduler_config_rejects_multiple_spec_decode_backends(kwargs, match):
    from rapid_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match=match):
        SchedulerConfig(**kwargs)


# ---------------------------------------------------------------------------
# 5. Metrics rendering
# ---------------------------------------------------------------------------


def test_metrics_renders_spec_decode_counters_zero_at_cold_start():
    """Before any MTP generation runs, the four MTP series MUST be
    present with value 0 (engine-independence rationale — same as
    response_format and mxfp4 guardrail counters).
    """
    from rapid_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()

    class _Cfg:
        model_alias = "qwen3.5-9b-4bit"

    lines = _render_spec_decode_mtp_counters(_Cfg())
    body = "\n".join(lines)
    assert "rapid_mlx_spec_decode_attempts_total" in body
    assert "rapid_mlx_spec_decode_accepts_total" in body
    assert "rapid_mlx_spec_decode_accept_ratio" in body
    assert "rapid_mlx_spec_decode_tokens_saved_total" in body
    # The family + method labels must be present.
    assert 'family="qwen3.5-9b-4bit"' in body
    assert 'method="mtp"' in body


def test_metrics_renders_post_acceptance_counters():
    """After 4 attempts / 3 accepts, the metric values must reflect it."""
    from rapid_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        get_global_counter,
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()
    counter = get_global_counter()
    for _ in range(4):
        counter.record_attempt()
    for _ in range(3):
        counter.record_accept(tokens_saved=1)

    class _Cfg:
        model_alias = "qwen3.5-9b-4bit"

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    assert (
        'rapid_mlx_spec_decode_attempts_total{family="qwen3.5-9b-4bit",method="mtp"} 4'
        in body
    )
    assert (
        'rapid_mlx_spec_decode_accepts_total{family="qwen3.5-9b-4bit",method="mtp"} 3'
        in body
    )
    assert (
        'rapid_mlx_spec_decode_tokens_saved_total{family="qwen3.5-9b-4bit",method="mtp"} 3'
        in body
    )
    # accept_ratio = 0.75 → must appear rounded to 4 decimals.
    assert "0.75" in body
    reset_global_counter_for_tests()


def test_metrics_renders_zero_ratio_when_no_attempts():
    """Zero attempts → ratio gauge MUST be 0 (not NaN, not missing).

    0.9.13 fix: family label used to hard-code ``"qwen3.5"`` when the
    alias was absent — that misreported Gemma 4 sidecar runs. Now the
    fallback is a family sniff on model_name / model_path with a
    stable ``"unknown"`` residual so the label set never changes.
    """
    from rapid_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()

    class _Cfg:
        model_alias = None

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    # No model_alias / model_name / model_path → "unknown" (stable
    # residual — never a transient empty string).
    assert 'family="unknown"' in body
    assert 'rapid_mlx_spec_decode_accept_ratio{family="unknown",method="mtp"} 0' in body


def test_metrics_family_falls_back_to_gemma4_on_model_name():
    """0.9.13 fix: when the operator loads by direct HF path (no
    alias, e.g. ``mlx-community/gemma-4-12b-it-4bit``), the family
    label must reflect Gemma 4 rather than the misleading Qwen
    fallback that broke per-family dashboards in 0.9.12.
    """
    from rapid_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()

    class _Cfg:
        model_alias = None
        model_name = "mlx-community/gemma-4-12b-it-4bit"

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    assert 'family="gemma4"' in body
    assert 'family="qwen3.5"' not in body


def test_metrics_family_falls_back_to_flash_next_on_model_path():
    from rapid_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()

    class _Cfg:
        model_alias = None
        model_path = "rapid-mlx/Qwen3.8-Flash-Next-4bit"

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    assert 'family="qwen3.8-flash-next"' in body


def test_metrics_includes_park_and_k_chosen_counters():
    """PR-B counter additions: ``rapid_mlx_spec_decode_park_total`` and
    the per-K ``rapid_mlx_spec_decode_k_chosen_total`` series must be
    present at cold-start (zero-valued) so dashboards discover the
    series before the first controller round lands.
    """
    from rapid_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        reset_controllers,
    )

    reset_global_counter_for_tests()
    reset_controllers()

    class _Cfg:
        model_alias = "gemma-4-12b-4bit"

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    assert (
        'rapid_mlx_spec_decode_park_total{family="gemma-4-12b-4bit",method="mtp"} 0'
        in body
    )
    # K-chosen histogram emits a zero-valued K=0 line even before any
    # rounds have run.
    assert "rapid_mlx_spec_decode_k_chosen_total" in body
    assert 'k="0"' in body
    assert (
        'rapid_mlx_spec_decode_k_chosen_rounds_total{family="gemma-4-12b-4bit",method="mtp"} 0'
        in body
    )


def test_metrics_k_cost_curve_absent_cold_then_present_after_rounds():
    """``rapid_mlx_spec_decode_k_cost_ms`` exports the controller's
    per-depth cost EWMA — the premise ``park_total`` rests on.

    Unlike the park/k_chosen series this one is NOT emitted at cold
    start: a zero-valued cost would read as "a round is free" rather
    than "no round has been measured", and a dashboard dividing by it
    would show an infinite speedup. Absence is the honest cold state.
    """
    from rapid_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        get_or_create_controller,
        reset_controllers,
    )

    reset_global_counter_for_tests()
    reset_controllers()

    class _Cfg:
        model_alias = "gemma-4-12b-4bit"

    assert "rapid_mlx_spec_decode_k_cost_ms" not in "\n".join(
        _render_spec_decode_mtp_counters(_Cfg())
    )

    ctrl = get_or_create_controller("test-model", max_k=1)
    ctrl.cost.observe(0, 20.0)
    ctrl.cost.observe(1, 30.0)

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    assert "# TYPE rapid_mlx_spec_decode_k_cost_ms gauge" in body
    assert (
        'rapid_mlx_spec_decode_k_cost_ms{method="mtp",'
        'model_id="test-model",k="0"} 20.000' in body
    )
    assert (
        'rapid_mlx_spec_decode_k_cost_ms{method="mtp",'
        'model_id="test-model",k="1"} 30.000' in body
    )
    reset_controllers()


def test_cost_curves_are_not_blended_across_controllers():
    """Two target+drafter combinations must stay separate series.

    Averaging them yields a cost ratio belonging to neither, published
    under a ``family`` label naming only one — an operator reading it
    would attribute the busier model's curve to whichever model they
    happened to be looking at.
    """
    from rapid_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        cost_curves_by_controller,
        get_or_create_controller,
        reset_controllers,
    )

    reset_global_counter_for_tests()
    reset_controllers()
    a = get_or_create_controller("model-a", max_k=1)
    b = get_or_create_controller("model-b", max_k=1)
    # Lopsided sample counts: under any visit-weighted blend "model-a"
    # would swallow "model-b" entirely.
    for _ in range(50):
        a.cost.observe(0, 10.0)
    b.cost.observe(0, 40.0)

    curves = cost_curves_by_controller()
    assert set(curves) == {"model-a", "model-b"}
    assert curves["model-a"][0] == pytest.approx(10.0)
    assert curves["model-b"][0] == pytest.approx(40.0)

    class _Cfg:
        model_alias = "gemma-4-12b-4bit"

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    assert 'model_id="model-a",k="0"} 10.000' in body
    assert 'model_id="model-b",k="0"} 40.000' in body
    # The per-controller series must NOT carry the route config's family:
    # neither of these controllers is the "gemma-4-12b-4bit" the renderer
    # was handed, and labelling them so would attribute one model's costs
    # to another on any family-filtered dashboard.
    for line in body.splitlines():
        if line.startswith("rapid_mlx_spec_decode_k_cost_ms{"):
            assert "family=" not in line, line
    reset_controllers()


def test_metrics_route_includes_spec_decode_series_at_cold_start():
    """End-to-end: the /metrics body emitted by the full renderer must
    carry the spec_decode series before any engine is up — matches the
    response_format + mxfp4 pre-engine surface convention.
    """
    from rapid_mlx.routes.metrics import _render_prometheus

    class _Cfg:
        engine = None
        model_name = "qwen3.5-9b-4bit"
        model_alias = "qwen3.5-9b-4bit"
        kv_cache_dtype = None

    body = _render_prometheus(_Cfg())
    assert "rapid_mlx_spec_decode_attempts_total" in body
    assert "rapid_mlx_spec_decode_accept_ratio" in body


# ---------------------------------------------------------------------------
# 5b. DepthController starvation-probe schedule (0.9.13 fix for K-lock at cap)
# ---------------------------------------------------------------------------


def _seed_controller_at_frontier(ctrl, k: int, high_accept: bool = True) -> None:
    """Push enough (record) rounds into ``ctrl`` to advance its frontier
    to ``k`` — the acceptance model needs ``ACCEPTANCE_MIN_SAMPLES=10``
    reaches at each position up to ``k``.

    Uses fixed synthetic wall_ms per K so the cost model becomes ready.
    ``high_accept=True`` means all drafts accept (frontier grows freely);
    False means acceptance oscillates so ``expected_committed`` is
    non-trivial.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        ACCEPTANCE_MIN_SAMPLES,
    )

    # Feed enough K=k rounds that positions 1..k each get >= 10 samples.
    for _ in range(ACCEPTANCE_MIN_SAMPLES + 2):
        # Cost per K rises with depth; K=3 is the most expensive.
        wall_ms = 15.0 + 3.0 * k
        accepts = [True] * k if high_accept else [True, True, False][:k]
        ctrl.record(k, wall_ms, accepts)


def test_starvation_probe_forces_undersampled_k_at_max_k_cap():
    """When the outward probe is clamped to ``sel`` (sel == max_k and
    frontier >= max_k), the new starvation probe must periodically
    override the EV pick with the least-recently-visited K in
    ``[0, min(frontier+1, max_k)]``.

    This is the direct regression test for the pathology reported by
    parent on Gemma 4 12B 4bit: 92.7% of rounds locked at K=3 after
    bootstrap. Without the starvation probe, ``pick_k`` would return
    K=3 forever once the EV comparator settled on it.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=3)

    # Bootstrap the controller into the K-lock steady state: seed all
    # K's, drive frontier to 3, then feed a long tail of K=3 rounds so
    # the EV picker locks onto K=3.
    for k in (0, 1, 2, 3):
        _seed_controller_at_frontier(ctrl, k, high_accept=True)

    # Long tail of K=3 so the outward probe clamps at max_k.
    for _ in range(80):
        k = ctrl.pick_k()
        # The starvation probe MUST fire periodically; without it,
        # every pick would be K=3. Assert we see at least one K < 3.
        wall_ms = 15.0 + 3.0 * k
        accepts = [True] * k
        ctrl.record(k, wall_ms, accepts)

    # After 80 rounds of steady-state (plus bootstrap), the K histogram
    # must show non-trivial samples at K∈{0,1,2}. The exact frequency
    # depends on the doubling cadence, but we must see AT LEAST ONE
    # starvation-probe override to know the mechanism fires.
    assert ctrl.starvation_probe_count >= 1, (
        f"Starvation probe never fired: histogram={ctrl.k_histogram} "
        f"starve_interval={ctrl._round_probe_interval}"
    )
    # And K=3 must NOT be 100% of post-bootstrap picks.
    non_three_count = sum(c for k, c in ctrl.k_histogram.items() if k != 3)
    assert non_three_count > 0, (
        f"K=3 dominates all rounds: histogram={ctrl.k_histogram}"
    )


def test_starvation_probe_argmin_over_rolling_window():
    """The starvation probe must pick the K with fewest samples in the
    recent ``_round_probe_interval`` window — not all-time. This
    prevents a briefly-explored K from being immune to future probing
    once its all-time count catches up.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        STARVATION_PROBE_INTERVAL,
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=3)
    # Manually seed so frontier is 3 and cost is ready.
    for k in (0, 1, 2, 3):
        _seed_controller_at_frontier(ctrl, k, high_accept=True)

    # Now feed a burst that biases the window heavily toward K=3.
    for _ in range(STARVATION_PROBE_INTERVAL * 4):
        ctrl.record(3, 24.0, [True, True, True])

    # Probe cadence has certainly elapsed; call pick_k enough times to
    # force at least one probe fire.
    starves_before = ctrl.starvation_probe_count
    picks = []
    for _ in range(STARVATION_PROBE_INTERVAL + 2):
        k = ctrl.pick_k()
        picks.append(k)
        ctrl.record(k, 15.0 + 3.0 * k, [True] * k)

    starves_after = ctrl.starvation_probe_count
    assert starves_after > starves_before, f"Expected probe to fire; picks={picks}"
    # The first probe must pick a K < 3 (any of 0/1/2 — window is all
    # K=3, so argmin over {0,1,2,3} is 0 with shallow tie-break).
    assert any(p < 3 for p in picks), f"Probe never picked a shallow K: picks={picks}"


def test_starvation_probe_interval_doubles_and_caps():
    """Interval doubles from the starvation-probe base up to 512 and
    does not overflow. Reset on EV pick change (``sel``) so a new
    selection gets an undisturbed interval.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        DEPTH_PROBE_INTERVAL_MAX,
        STARVATION_PROBE_INTERVAL,
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=3)
    for k in (0, 1, 2, 3):
        _seed_controller_at_frontier(ctrl, k, high_accept=True)

    # Fire many probes; interval must saturate at MAX.
    for _ in range(2000):
        k = ctrl.pick_k()
        ctrl.record(k, 15.0 + 3.0 * k, [True] * k)

    assert ctrl._round_probe_interval <= DEPTH_PROBE_INTERVAL_MAX
    # And the base is at least the min cadence.
    assert ctrl._round_probe_interval >= STARVATION_PROBE_INTERVAL


def test_starvation_probe_no_double_pick_when_probe_matches_current_depth():
    """If the argmin K equals the depth already chosen by the EV pick
    (e.g. bootstrap seed picks the shallowest under-visited K anyway),
    the probe counter still consumes its slot (interval doubles) — this
    keeps the cadence deterministic. Verify the probe counter resets.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=3)
    # Only bootstrap K=0 and K=1 so frontier stays 0.
    for _ in range(6):
        ctrl.record(0, 15.0, [])
    for _ in range(6):
        ctrl.record(1, 18.0, [True])

    # Advance rounds and observe counter behavior.
    prev_interval = ctrl._round_probe_interval
    for _ in range(20):
        k = ctrl.pick_k()
        ctrl.record(k, 15.0 + 3.0 * k, [True] * k)
    # Interval must have moved (either doubled from probing, or reset
    # from EV pick change) — either way, no NaN/negative.
    assert ctrl._round_probe_interval >= 4
    assert ctrl._round_probe_interval <= 512
    # Prev interval starts at the starvation probe base.
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        STARVATION_PROBE_INTERVAL,
    )

    assert prev_interval == STARVATION_PROBE_INTERVAL


def test_starvation_probe_resets_when_ev_pick_changes():
    """When EV pick shifts (``sel`` changes), the starvation-probe
    interval must reset to the base — the new selection deserves a
    full interval of undisturbed operation.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        STARVATION_PROBE_INTERVAL,
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=3)
    for k in (0, 1, 2, 3):
        _seed_controller_at_frontier(ctrl, k, high_accept=True)

    # Fire enough probes to grow the interval.
    for _ in range(200):
        k = ctrl.pick_k()
        ctrl.record(k, 15.0 + 3.0 * k, [True] * k)
    grown_interval = ctrl._round_probe_interval
    assert grown_interval >= STARVATION_PROBE_INTERVAL

    # Now cause EV pick to shift: feed many K=1 rounds with poor accept
    # so cost[1] gets updated and EV eventually swings.
    # This is a soft test — we just verify the reset mechanism exists.
    ctrl._round_probe_last_sel = 999  # simulate a sel change
    _ = ctrl.pick_k()
    assert ctrl._round_probe_interval == STARVATION_PROBE_INTERVAL


# ---------------------------------------------------------------------------
# 5c. DepthController diagnostics (the learned curves, for DEBUG logs)
# ---------------------------------------------------------------------------


def _seeded_controller(max_k=3, cost_ms=(65.0, 73.0, 88.0, 110.0), accept=0.8):
    """A controller fed a realistic cost curve and acceptance stream.

    ``cost_ms`` is indexed by depth and ``accept`` is the per-position
    acceptance rate; both are applied often enough to pass
    ``ACCEPTANCE_MIN_SAMPLES`` so the frontier actually moves.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=max_k)
    hits = round(accept * 10)
    for depth in range(0, max_k + 1):
        credit = 0
        for _ in range(200):
            # Bresenham: ``hits`` accepts per 10 rounds, spread evenly rather
            # than blocked, so the EWMA settles on the requested rate instead
            # of on wherever a run of rejections happened to leave it.
            credit += hits
            accepted = credit >= 10
            if accepted:
                credit -= 10
            ctrl.record(
                depth,
                cost_ms[depth],
                [True] * depth if accepted else [False] * depth,
            )
    if max_k >= 1:
        # An EWMA of a repeating pattern ripples rather than converging;
        # ``ACCEPTANCE_EWMA_ALPHA`` bounds that ripple well inside 0.1.
        assert ctrl.acc.acceptance(1) == pytest.approx(accept, abs=0.1)
    return ctrl


def test_acceptance_sample_string_reports_only_trusted_positions():
    """Positions past the frontier read back an inherited rate, not a
    measured one, so publishing them would pass off the search policy as
    data. The string stops at the frontier for that reason."""
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        ACCEPTANCE_MIN_SAMPLES,
        AcceptanceModel,
    )

    acc = AcceptanceModel()
    assert acc.sample_string() == ""

    for _ in range(ACCEPTANCE_MIN_SAMPLES):
        acc.observe(2, 1)  # position 1 accepted, position 2 rejected

    assert acc.frontier() == 2
    assert acc.sample_string() == "1:1.00 2:0.00"
    # Position 3 answers (inheriting 0.00) but is not reported.
    assert acc.acceptance(3) == pytest.approx(0.0)


def test_expected_tps_string_is_empty_until_two_depths_are_sampled():
    """One cost sample gives a point, not a slope; a tok/s figure derived
    from it would be an extrapolation reported as a measurement."""
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=3)
    for _ in range(6):
        ctrl.record(0, 65.0, [])

    assert ctrl.expected_tps_string() == ""
    assert "expected_tps=[]" in ctrl.diagnostics()


def test_expected_tps_string_is_the_quantity_the_pick_maximizes():
    """The printed curve has to be the comparator's own, or the log
    explains a decision the controller did not make. Check the argmax of
    the string equals ``pick_k``'s selection."""
    ctrl = _seeded_controller()

    parts = dict(
        (int(k), float(v))
        for k, v in (p.split(":") for p in ctrl.expected_tps_string().split())
    )
    assert set(parts) == set(range(0, min(ctrl.frontier() + 1, ctrl.max_k) + 1))
    # tok/s, not tok/ms: a 65ms park round is ~15 tok/s.
    assert parts[0] == pytest.approx(1000.0 / 65.0, rel=0.05)

    best = max(parts, key=lambda n: parts[n])
    assert ctrl._selected() == best


def test_diagnostics_carries_every_curve_a_depth_choice_depends_on():
    """A depth pick is a function of cost, acceptance and the resulting
    EV; a log missing any one of them cannot be used to second-guess it."""
    ctrl = _seeded_controller()

    line = ctrl.diagnostics()
    for field in ("cost=[", "acceptance=[", "expected_tps=[", "probe_interval="):
        assert field in line
    assert "cost=[]" not in line
    assert "acceptance=[]" not in line
    assert "expected_tps=[]" not in line


def test_record_logs_diagnostics_on_the_configured_cadence(caplog):
    """``diagnostics()`` was unreachable from a running server before
    this -- nothing called it -- so the curves could not be read off a
    DEBUG run at all."""
    import logging

    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        DIAGNOSTICS_LOG_INTERVAL,
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=3)
    with caplog.at_level(
        logging.DEBUG, logger="rapid_mlx.spec_decode.mtp.draft_k_controller_v2"
    ):
        for _ in range(DIAGNOSTICS_LOG_INTERVAL - 1):
            ctrl.record(1, 73.0, [True])
        assert not [r for r in caplog.records if "[MTP-controller]" in r.message]

        ctrl.record(1, 73.0, [True])
        hits = [r for r in caplog.records if "[MTP-controller]" in r.message]

    assert len(hits) == 1
    assert f"rounds={DIAGNOSTICS_LOG_INTERVAL}" in hits[0].getMessage()


def test_record_does_not_build_the_diagnostics_line_below_debug(monkeypatch):
    """The line walks both models' curves, so paying for it on a default
    INFO server would be a per-round cost for output nobody reads."""
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        DIAGNOSTICS_LOG_INTERVAL,
        DepthController,
        reset_controllers,
    )

    reset_controllers()
    ctrl = DepthController(max_k=3)
    calls = []
    monkeypatch.setattr(
        type(ctrl),
        "diagnostics",
        lambda self: calls.append(1) or "",
    )

    import logging

    logger = logging.getLogger("rapid_mlx.spec_decode.mtp.draft_k_controller_v2")
    old_level, logger.level = logger.level, logging.INFO
    old_propagate, logger.propagate = logger.propagate, False
    try:
        for _ in range(DIAGNOSTICS_LOG_INTERVAL):
            ctrl.record(1, 73.0, [True])
    finally:
        logger.level = old_level
        logger.propagate = old_propagate

    assert calls == []


# ---------------------------------------------------------------------------
# 6. MTP head builder
# ---------------------------------------------------------------------------


def test_build_mtp_module_rejects_zero_layers():
    """``num_layers < 1`` is a programmer error — fail loud."""
    from rapid_mlx.spec_decode.mtp.head import build_mtp_module

    class _FakeArgs:
        hidden_size = 32
        rms_norm_eps = 1e-6
        num_experts = 0
        intermediate_size = 64

    with pytest.raises(ValueError, match="num_layers >= 1"):
        build_mtp_module(_FakeArgs(), 0)


def _tiny_text_model_args():
    """Minimal ``TextModelArgs`` for shape tests on the MTP head.

    Note: our installed mlx-lm 0.31.3 doesn't define
    ``mtp_num_hidden_layers`` on ``TextModelArgs`` yet (it's added by
    PR #990). The injection helper does NOT depend on the field
    being on the dataclass schema (it reads via ``getattr`` with
    ``default=0``), so we can attach it as a post-construction
    attribute and the head builder still works.
    """
    from mlx_lm.models.qwen3_5 import TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=128,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=16,
        full_attention_interval=1,
        num_experts=0,
        num_experts_per_tok=0,
        decoder_sparse_step=0,
        shared_expert_intermediate_size=0,
        moe_intermediate_size=0,
        norm_topk_prob=True,
    )
    # Field added by PR #990 — not yet on the floor mlx-lm dataclass.
    object.__setattr__(args, "mtp_num_hidden_layers", 1)
    return args


def test_build_mtp_module_constructs_with_real_qwen3_5_args():
    """The head constructor must work against the real
    ``TextModelArgs`` schema (not just our synthetic dict). We use a
    minimal Qwen3.5 args instance (small dims so the test stays fast).
    """
    from rapid_mlx.spec_decode.mtp.head import build_mtp_module

    args = _tiny_text_model_args()
    head = build_mtp_module(args, 1)
    assert hasattr(head, "pre_fc_norm_hidden")
    assert hasattr(head, "pre_fc_norm_embedding")
    assert hasattr(head, "fc")
    assert hasattr(head, "layers")
    assert hasattr(head, "norm")
    assert len(head.layers) == 1


# ---------------------------------------------------------------------------
# 7. Qwen3.5 model-side injection
# ---------------------------------------------------------------------------


def _build_tiny_qwen3_5_text_model():
    """Construct a minimal Qwen3.5 ``TextModel`` instance for shape tests.

    No weight load. Returns the inner ``TextModel`` rather than the
    wrapping VLM-style ``Model`` because:

    * ``Model.__init__`` requires a full ``text_config`` dict that
      ``TextModelArgs.from_dict`` can parse — the field set is brittle
      across mlx-lm patch versions.
    * ``inject_mtp_support`` accepts either the ``Model`` wrapper OR
      the inner ``TextModel`` (the ``_resolve_inner_text_model``
      helper detects which is which by walking ``model.args``).
      Passing the inner model directly skips one indirection.
    """
    from mlx_lm.models.qwen3_5 import TextModel

    args = _tiny_text_model_args()
    object.__setattr__(args, "mtp_num_hidden_layers", 1)
    return TextModel(args)


def test_inject_mtp_support_attaches_four_surfaces():
    """Inject must add ``mtp_forward``, ``make_mtp_cache``, and accept
    ``return_hidden`` / ``n_confirmed`` in ``__call__``.
    """
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch in this mlx-lm: {exc}")

    # allow_random_init=True: this is the test-only wiring probe
    # (no sidecar download); production callers pass mtp_sidecar.
    injected = inject_mtp_support(model, allow_random_init=True)
    assert injected is True
    assert validate_mtp_support(model) is True
    assert model.mtp_prompt_lookup_supported is True
    # ``mtp_prompt_lookup_supported`` alone was never enough: the admission
    # guard also has to be told this family's recurrent caches carry a
    # per-boundary restore point, or it rejects every proposal and the
    # feature is dead code on the whole hybrid family.
    assert model.mtp_wide_verify_rollback_supported is True
    policy = model.mtp_prompt_lookup_policy
    assert policy.enabled_by_default is True
    assert (policy.min_ngram, policy.max_ngram) == (8, 64)
    # The depth knob is the tile edge itself, not a number that merely
    # happens to sit under it -- see that constant's own contract test. This
    # policy is where the ceiling lives: the generator honours whatever it (or
    # an operator override) says, so a default of anything else would be the
    # whole feature mis-sized.
    from rapid_mlx.spec_decode.mtp.prompt_lookup import MAX_COPY_DRAFT_TOKENS

    assert policy.max_tokens == MAX_COPY_DRAFT_TOKENS


def test_inject_mtp_support_mirrors_batch_seam_to_outer_wrapper():
    """The scheduler may retain the outer model returned by mlx-lm."""
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        inner = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch in this mlx-lm: {exc}")

    class _Outer:
        def __init__(self, language_model):
            self.language_model = language_model

    outer = _Outer(inner)
    assert inject_mtp_support(outer, allow_random_init=True) is True
    assert outer.mtp_batch_forward.__self__ is inner
    assert outer.batched_mtp_capability is inner.batched_mtp_capability
    assert outer.mtp_recursive_draft_depth == 2
    # The generator reads its capabilities off whichever object the scheduler
    # kept, so a seam that mirrors some of them and not others reintroduces
    # the bug on exactly the VLM-wrapped path.
    assert outer.mtp_wide_verify_rollback_supported is True
    assert outer.mtp_prompt_lookup_policy is inner.mtp_prompt_lookup_policy


def test_inject_mtp_support_rejects_non_qwen35_model():
    """A non-Qwen3.5 model (no ``args.mtp_num_hidden_layers``) must
    return False and not patch anything.
    """
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    class _FakeArgs:
        hidden_size = 32
        rms_norm_eps = 1e-6
        # NOTE: no mtp_num_hidden_layers attribute.

    class _FakeModel:
        args = _FakeArgs()
        model = object()

    assert inject_mtp_support(_FakeModel()) is False


def test_inject_mtp_support_rejects_stripped_checkpoint():
    """Qwen3.5 with mtp_num_hidden_layers=0 (operator passed
    pre-PR-#990 checkpoint) → inject returns False.
    """
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    class _FakeArgs:
        hidden_size = 32
        rms_norm_eps = 1e-6
        mtp_num_hidden_layers = 0

    class _FakeInner:
        args = _FakeArgs()
        model = object()

    # Pass FakeInner as both ``model`` and the inner model — the
    # resolver picks up ``model.args`` and decides on
    # ``mtp_num_hidden_layers``.
    assert inject_mtp_support(_FakeInner()) is False


def test_inject_mtp_support_refuses_no_sidecar_by_default():
    """Default ``allow_random_init=False`` must refuse a sidecar-less inject.

    Codex round-5 BLOCKING fix: silently shipping a random-init MTP
    head (~0% accept rate) under the production-default code path
    looked like spec-decode was enabled but yielded zero speedup.
    With this fix, ``inject_mtp_support(model)`` (no sidecar, no
    opt-in) must return False and leave the model unmodified.
    """
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    # No sidecar, no allow_random_init → must fail closed.
    assert inject_mtp_support(model) is False, (
        "Default inject_mtp_support without sidecar should return False"
    )
    # And the model must NOT have been patched — validate_mtp_support
    # checks the four surfaces, none should land on a failed inject.
    assert validate_mtp_support(model) is False


def test_inject_mtp_support_loads_synthetic_sidecar():
    """Lightweight quantize → load → coverage-check probe (no 5 GB download).

    Codex round-5 NIT: the heavy real-weights test is gated on
    RAPID_MLX_RUN_HEAVY_TESTS=1 and doesn't run in normal CI, so the
    quantize/load/key-coverage path it covers has no default
    safety net. This test fills the gap with a synthetic sidecar:

    1. Build a tiny Qwen3.5 TextModel (existing helper).
    2. Build the MTP head module via build_mtp_module.
    3. Persist its (random-init) parameters to a temp safetensors
       file — this becomes the "sidecar" the inject will load.
    4. Re-build a fresh model + inject with mtp_sidecar=<temp file>.
       Inject must succeed AND the loaded MTP weights must match
       what we persisted.

    Failure modes this guards against:

    * mtp.load_weights silently no-ops because key names drift
      between build and load.
    * The coverage check (expected_keys vs loaded_keys) misses
      missing tensors.
    * The custom-file-path branch of _resolve_sidecar_file
      regresses.

    Runs in <2 s on the CI machine — no network, no GPU required
    beyond what every other unit test uses.
    """
    import tempfile
    from pathlib import Path

    import mlx.core as _mx
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model_a = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    # Build the MTP head separately so we can capture its random-init
    # weights, write them to disk, and verify the inject loads them
    # byte-equally. (Note: this tiny model is FP, so the sidecar ships as
    # a metadata-less full-precision safetensors file, with no config.json
    # and no fc.scales — the inject detects a full-precision sidecar from
    # its tensors and keeps the MTP module FP, matching the sidecar layout.)
    args = model_a.args
    mtp_template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _mx.eval(mtp_template.parameters())
    flat = dict(tree_flatten(mtp_template.parameters()))
    assert flat, "build_mtp_module produced an empty parameter tree"

    with tempfile.TemporaryDirectory() as tmp:
        sidecar_path = Path(tmp) / "synthetic-mtp-head.safetensors"
        _mx.save_safetensors(str(sidecar_path), flat)
        # Full-precision sidecar (no fc.scales, no config.json) — recognised
        # as FP from its tensors, so the MTP module is kept full-precision.

        # Build a fresh model (so MTP head random init differs from
        # the persisted template), then inject with the synthetic
        # sidecar file path. Tests the custom-filename branch of
        # _resolve_sidecar_file.
        model_b = _build_tiny_qwen3_5_text_model()
        result = inject_mtp_support(model_b, mtp_sidecar=str(sidecar_path))
        assert result is True, (
            "inject_mtp_support failed on a synthetic sidecar that exactly "
            "matches the MTP module's parameter tree — likely a coverage-check "
            "false positive (expected_keys drift) or a _resolve_sidecar_file regression."
        )
        assert validate_mtp_support(model_b) is True

        # The inject MUST have loaded the persisted weights byte-equally.
        loaded = dict(tree_flatten(model_b.mtp.parameters()))
        assert set(loaded.keys()) == set(flat.keys()), (
            f"Parameter trees diverged. "
            f"In template only: {set(flat) - set(loaded)}. "
            f"In loaded only: {set(loaded) - set(flat)}."
        )
        for k in flat:
            diff = _mx.sum(loaded[k] != flat[k]).item()
            assert diff == 0, (
                f"{k}: loaded MTP weight differs from sidecar by {diff} entries. "
                f"This is the random-init defect class PR #918 shipped."
            )


def _write_synthetic_sidecar(model, tmp_dir, contract=None):
    """Persist ``model``'s freshly built MTP head as a loadable sidecar.

    Returns the sidecar path. ``contract``, when given, is written beside
    it as the ``mtp_contract`` of an ``mtplx_runtime.json`` manifest so the
    test can pin a non-default runtime contract.
    """
    import json
    from pathlib import Path

    import mlx.core as _mx
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module

    args = model.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _mx.eval(template.parameters())
    sidecar = Path(tmp_dir) / "mtp.safetensors"
    _mx.save_safetensors(str(sidecar), dict(tree_flatten(template.parameters())))
    if contract is not None:
        (Path(tmp_dir) / "mtplx_runtime.json").write_text(
            json.dumps({"mtp_contract": contract}), encoding="utf-8"
        )
    return sidecar


def test_return_hidden_hands_the_drafter_the_tensor_the_lm_head_scored():
    """Default contract: the drafter sees the backbone's FINAL hidden state.

    A Qwen3-Next-style MTP head is trained on whatever the target's own
    output head consumes -- post-norm for this family (``TextModel.__call__``
    ends in ``self.norm(...)``; vLLM's ``qwen3_next_mtp.py`` feeds
    ``pre_fc_norm_hidden`` the same post-norm tensor its ``compute_logits``
    scores). Handing the head the pre-norm hidden instead is silent: it
    still drafts fluent tokens, they are just accepted less often, so no
    functional test catches it -- only this identity does.
    """
    import mlx.core as _mx

    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        model = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    assert model.args.tie_word_embeddings is False, (
        "this assertion re-scores through ``lm_head``; a tied checkpoint "
        "would score through ``embed_tokens.as_linear`` instead"
    )
    assert inject_mtp_support(model, allow_random_init=True) is True

    logits, hidden = model(_mx.array([[3, 7, 11]]), return_hidden=True)
    _mx.eval(logits, hidden)

    assert _mx.allclose(model.lm_head(hidden), logits, atol=1e-4, rtol=1e-4), (
        "return_hidden must yield the tensor lm_head just scored; "
        "re-scoring it through the same head has to reproduce the logits"
    )


def test_mtplx_manifest_can_still_pin_the_pre_norm_base_hidden():
    """The escape hatch survives the default flip -- and proves it bites.

    An artifact exported against the pre-norm hidden pins
    ``base_hidden_variant`` in its manifest. Asserting BOTH that the
    pinned hidden reproduces the logits only after the backbone norm AND
    that it does not reproduce them without one is what makes the default
    test above discriminating rather than vacuous.
    """
    import tempfile

    import mlx.core as _mx

    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        model = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        sidecar = _write_synthetic_sidecar(
            model, tmp, contract={"base_hidden_variant": "pre_norm"}
        )
        assert inject_mtp_support(model, mtp_sidecar=str(sidecar)) is True

        logits, hidden = model(_mx.array([[3, 7, 11]]), return_hidden=True)
        _mx.eval(logits, hidden)

        assert _mx.allclose(
            model.lm_head(model.model.norm(hidden)), logits, atol=1e-4, rtol=1e-4
        ), "a pinned pre_norm hidden must be exactly the input to the backbone norm"
        assert not _mx.allclose(model.lm_head(hidden), logits, atol=1e-4, rtol=1e-4), (
            "pre-norm and post-norm hidden must be distinguishable at the head"
        )


def test_inject_mtp_support_refuses_synthetic_sidecar_missing_tensor():
    """Coverage check: dropping one required tensor must fail the inject.

    Codex round-3 BLOCKING fix added a pre-load coverage check that
    walks mtp.parameters() and refuses inject when any required key
    is missing from the sidecar. This test exercises that path with
    a tiny synthetic sidecar — no network, no GPU contention.
    """
    import tempfile
    from pathlib import Path

    import mlx.core as _mx
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        model = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    args = model.args
    mtp_template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _mx.eval(mtp_template.parameters())
    flat = dict(tree_flatten(mtp_template.parameters()))
    # Drop the FC weight — the inject's coverage check must catch this.
    fc_keys = [k for k in flat if k.startswith("fc.")]
    assert fc_keys, "tiny MTP template missing fc.* keys — test premise broken"
    drop_key = fc_keys[0]
    crippled = {k: v for k, v in flat.items() if k != drop_key}

    with tempfile.TemporaryDirectory() as tmp:
        sidecar_path = Path(tmp) / "crippled-sidecar.safetensors"
        _mx.save_safetensors(str(sidecar_path), crippled)
        # FP sidecar (no fc.scales): the quantization step keeps the MTP
        # module FP, so the inject reaches the coverage check under test —
        # which must catch the dropped tensor.

        fresh_model = _build_tiny_qwen3_5_text_model()
        result = inject_mtp_support(fresh_model, mtp_sidecar=str(sidecar_path))
        assert result is False, (
            f"inject_mtp_support should have refused a sidecar missing {drop_key!r}, "
            f"but returned True — the coverage check has regressed."
        )


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_infer_sidecar_fc_quantization_recovers_bits_and_group_size(bits, group_size):
    """The sidecar's own tensors are the source of truth: inverting the
    packed ``fc.weight`` / ``fc.scales`` shapes recovers the exact
    ``(bits, group_size)`` it was quantized with — no config.json needed.

    Regression anchor for the 8-bit-base + 4-bit-sidecar empty-response
    bug: the module is quantized to match the sidecar it loads, read off
    the sidecar tensors — not the base model, not a config proxy.
    """
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import (
        _infer_sidecar_fc_quantization,
    )

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    args = base.args
    fp = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    fc_out_dims, fc_in_dims = (int(d) for d in fp.fc.weight.shape)

    # Quantize a standalone Linear matching the fc — NOT the whole module,
    # whose other layers are only ``hidden_size``-wide and can't be
    # quantized at group_size 128 in this tiny fixture.
    fc = _nn.Linear(fc_in_dims, fc_out_dims, bias=False)
    qfc = _nn.QuantizedLinear.from_linear(fc, group_size, bits)
    _mx.eval(qfc.parameters())
    flat = {f"fc.{k}": v for k, v in dict(tree_flatten(qfc.parameters())).items()}
    assert "fc.scales" in flat, "quantized fc should carry a scales tensor"

    assert _infer_sidecar_fc_quantization(flat, fc_out_dims, fc_in_dims) == {
        "bits": bits,
        "group_size": group_size,
    }


def test_infer_sidecar_fc_quantization_full_precision_returns_none():
    """A full-precision fc has no ``fc.scales`` tensor, so the inference
    returns ``None`` and the caller keeps the MTP module FP — no config
    metadata required (fixes the metadata-less-FP-sidecar regression)."""
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _infer_sidecar_fc_quantization

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    args = base.args
    fp = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    fc_out_dims, fc_in_dims = (int(d) for d in fp.fc.weight.shape)
    flat = dict(tree_flatten(fp.parameters()))
    assert "fc.scales" not in flat
    assert _infer_sidecar_fc_quantization(flat, fc_out_dims, fc_in_dims) is None


def test_mtp_quantization_pairing_warning_for_mismatch_only(caplog):
    """Known mixed precision warns once; a matched pairing stays silent."""
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import (
        _warn_if_mtp_quantization_mismatch,
    )

    logger_name = "rapid_mlx.spec_decode.mtp.qwen3_5_inject"
    with caplog.at_level("WARNING", logger=logger_name):
        _warn_if_mtp_quantization_mismatch(
            {"bits": 8, "group_size": 64},
            {"bits": 4, "group_size": 64},
        )

    assert [record.getMessage() for record in caplog.records] == [
        "[mtp.inject] MTP sidecar quantization (4-bit, group_size=64) "
        "differs from base model (8-bit, group_size=64): pairing effects "
        "are model-dependent: slower than no speculation on Qwen3.6 "
        "(#1258), faster on Qwen3.8-27B. Benchmark your pairing."
    ]

    caplog.clear()
    with caplog.at_level("WARNING", logger=logger_name):
        _warn_if_mtp_quantization_mismatch(
            {"bits": 4, "group_size": 64},
            {"bits": 4, "group_size": 64},
        )
    assert caplog.records == []


def test_infer_sidecar_fc_quantization_raises_on_malformed_packing():
    """``fc.scales`` present but a packing we cannot interpret — an
    unsupported derived width, or a missing companion ``fc.weight`` —
    raises ``ValueError`` so the caller refuses rather than mis-pack."""
    import mlx.core as _mx

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _infer_sidecar_fc_quantization

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    args = base.args
    fp = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    fc_out_dims, fc_in_dims = (int(d) for d in fp.fc.weight.shape)

    # (a) shapes that invert to an unsupported width. For in=fc_in_dims,
    # weight cols = in*7//32 and scales cols = in//32 imply bits=7 (not an
    # MLX affine width) → ValueError.
    packed_cols = fc_in_dims * 7 // 32
    scale_cols = fc_in_dims // 32
    bad = {
        "fc.weight": _mx.zeros((fc_out_dims, packed_cols), dtype=_mx.uint32),
        "fc.scales": _mx.zeros((fc_out_dims, scale_cols), dtype=_mx.float32),
    }
    with pytest.raises(ValueError):
        _infer_sidecar_fc_quantization(bad, fc_out_dims, fc_in_dims)

    # (b) scales present but the companion weight is missing entirely.
    with pytest.raises(ValueError):
        _infer_sidecar_fc_quantization(
            {"fc.scales": _mx.zeros((fc_out_dims, fc_in_dims // 32))},
            fc_out_dims,
            fc_in_dims,
        )

    # (c) truncated quantized sidecar: a PACKED fc.weight (4-bit width)
    # but no fc.scales. Must NOT be mistaken for full-precision (which
    # would load a packed weight into an nn.Linear and crash later) —
    # the shape mismatch against the FP dims raises.
    with pytest.raises(ValueError):
        _infer_sidecar_fc_quantization(
            {
                "fc.weight": _mx.zeros(
                    (fc_out_dims, fc_in_dims * 4 // 32), dtype=_mx.uint32
                )
            },
            fc_out_dims,
            fc_in_dims,
        )

    # A correctly-shaped FP fc.weight with no scales is fine (returns None).
    assert (
        _infer_sidecar_fc_quantization(
            {"fc.weight": _mx.zeros((fc_out_dims, fc_in_dims), dtype=_mx.float32)},
            fc_out_dims,
            fc_in_dims,
        )
        is None
    )


def test_inject_quantizes_mtp_to_sidecar_bits_not_base_bits(tmp_path):
    """Regression: an 8-bit base paired with a 4-bit MTP sidecar must
    quantize the MTP module to the SIDECAR's 4 bits, not the base's 8.

    Before the fix, the MTP module was quantized to the *base* model's
    bit-width; loading the differently-packed sidecar tensors left the
    ``fc`` layer's packed ``weight`` inconsistent with its ``bits``
    attribute, so ``mx.quantized_matmul`` raised at the first MTP draft
    step — surfacing to the client as an intermittent EMPTY response
    (``prompt_tokens=0``) once the depth controller began speculating.
    Mirrors the shipped pairing ``Qwen3.6-27B-MLX-8bit`` +
    ``Qwen3.6-27B-MTP-4bit``.
    """
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import (
        _detect_base_quantization,
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    _GROUP = 32
    _BASE_BITS = 8
    _SIDECAR_BITS = 4

    # Quantize the BASE at 8-bit so ``_detect_base_quantization`` (the
    # pre-fix source) returns 8-bit — the WRONG width for a 4-bit sidecar.
    _nn.quantize(base.model, group_size=_GROUP, bits=_BASE_BITS)
    assert _detect_base_quantization(base) == {
        "bits": _BASE_BITS,
        "group_size": _GROUP,
    }

    # Build a 4-bit MTP sidecar — weights only, NO config.json: the
    # quantization is inferred from the sidecar tensors themselves.
    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _nn.quantize(template, group_size=_GROUP, bits=_SIDECAR_BITS)
    _mx.eval(template.parameters())
    flat = dict(tree_flatten(template.parameters()))
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    injected = inject_mtp_support(base, mtp_sidecar=str(tmp_path))
    assert injected is True, (
        "inject_mtp_support returned False on an 8-bit base + 4-bit sidecar; "
        "the module was likely quantized to the base's 8-bit and load_weights "
        "or coverage-check rejected the 4-bit tensors."
    )
    assert validate_mtp_support(base) is True

    # The MTP fc layer must carry the SIDECAR's 4 bits, not the base's 8.
    assert isinstance(base.mtp.fc, _nn.QuantizedLinear)
    assert int(base.mtp.fc.bits) == _SIDECAR_BITS, (
        f"MTP fc quantized to {base.mtp.fc.bits}-bit; expected the sidecar's "
        f"{_SIDECAR_BITS}-bit. Quantizing to the base model's bits reintroduces "
        "the quantized_matmul weight/scales mismatch (empty-response bug)."
    )

    # And the draft forward must run without the quantized_matmul crash —
    # this is the exact call that raised in the field repro.
    hidden = _mx.zeros((1, 1, int(args.hidden_size)), dtype=_mx.float32)
    next_ids = _mx.array([[0]], dtype=_mx.uint32)
    logits = base.mtp_forward(hidden, next_ids, base.make_mtp_cache())
    _mx.eval(logits)
    assert logits.shape[-1] == int(args.vocab_size)


def test_inject_keeps_mtp_full_precision_for_fp_sidecar(tmp_path):
    """A quantized base paired with a full-precision sidecar must leave
    the MTP module FP — NOT quantize it to the base's bits (which would
    mispack the FP sidecar tensors on load). Full-precision is recognised
    from the ABSENCE of ``fc.scales`` in the sidecar tensors, with NO
    config.json — the metadata-less-FP-sidecar path that previously
    worked and must keep working."""
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    # Quantize the base — pre-fix, this drove the MTP module's quantization.
    _nn.quantize(base.model, group_size=32, bits=8)

    # Build a full-precision MTP sidecar (no quantize, no config.json) —
    # the fc has no scales tensor, so it is recognised as FP.
    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _mx.eval(template.parameters())
    flat = dict(tree_flatten(template.parameters()))
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    injected = inject_mtp_support(base, mtp_sidecar=str(tmp_path))
    assert injected is True
    assert validate_mtp_support(base) is True
    # The fc layer must stay a plain (full-precision) Linear.
    assert isinstance(base.mtp.fc, _nn.Linear)
    assert not isinstance(base.mtp.fc, _nn.QuantizedLinear)

    hidden = _mx.zeros((1, 1, int(args.hidden_size)), dtype=_mx.float32)
    next_ids = _mx.array([[0]], dtype=_mx.uint32)
    logits = base.mtp_forward(hidden, next_ids, base.make_mtp_cache())
    _mx.eval(logits)
    assert logits.shape[-1] == int(args.vocab_size)


def test_inject_refuses_explicit_sidecar_with_malformed_packing(tmp_path):
    """An EXPLICIT sidecar whose quantized fc tensors imply a packing we
    cannot reproduce (here: shapes that invert to an unsupported 7-bit
    width) must make ``inject_mtp_support`` REFUSE (return False) — never
    fall back to the base model's bit-width. Guessing the base's width for
    a differently-packed sidecar is exactly what aborted requests at the
    first draft step (the empty-response bug); a safe non-install (plain
    base path) is the correct degradation.

    (A *well-formed* quantized sidecar with no config.json is NOT refused
    — its bits/group_size are recovered from the tensors; see
    ``test_inject_quantizes_mtp_to_sidecar_bits_not_base_bits``.)
    """
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    # Base quantized at 8-bit — the WRONG width a naive fallback would use.
    _nn.quantize(base.model, group_size=32, bits=8)

    # Start from a real 4-bit sidecar, then corrupt ONLY the fc packing so
    # the derived width is an unsupported 7 bits. fc is Linear(2H, H) →
    # in = 2H; weight cols = in*7//32, scales cols = in//32 imply bits=7.
    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _nn.quantize(template, group_size=32, bits=4)
    _mx.eval(template.parameters())
    flat = dict(tree_flatten(template.parameters()))
    fp = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _fc_out, _fc_in = (int(d) for d in fp.fc.weight.shape)
    flat["fc.weight"] = _mx.zeros((_fc_out, _fc_in * 7 // 32), dtype=_mx.uint32)
    flat["fc.scales"] = _mx.zeros((_fc_out, _fc_in // 32), dtype=_mx.float32)
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    assert inject_mtp_support(base, mtp_sidecar=str(tmp_path)) is False
    assert not hasattr(base, "mtp"), (
        "inject refused but still attached an MTP module — the refusal must "
        "leave the base model untouched (plain path)."
    )


def test_inject_refuses_corrupt_sidecar_file_without_raising(tmp_path):
    """A truncated/unreadable safetensors sidecar makes ``mx.load`` raise —
    codex flagged (PR #1201) that this exception was UNGUARDED and would
    escape ``inject_mtp_support`` instead of hitting the same return-False
    fail-safe as every other bad-sidecar path (malformed packing, shape
    mismatch, dtype mismatch, ...). A corrupt file on disk (partial
    download, disk error) must degrade the same way: refuse injection,
    never abort the request mid-generation.
    """
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    sidecar_path = tmp_path / "corrupt-sidecar.safetensors"
    sidecar_path.write_bytes(b"not a real safetensors file" * 4)

    result = inject_mtp_support(base, mtp_sidecar=str(sidecar_path))
    assert result is False, (
        "inject_mtp_support should refuse (return False) a corrupt sidecar "
        "file, not raise mx.load's exception mid-request."
    )
    assert not hasattr(base, "mtp"), (
        "inject refused but still attached an MTP module — the refusal must "
        "leave the base model untouched (plain path)."
    )


def test_inject_refuses_when_materialization_raises_without_propagating(
    tmp_path, monkeypatch
):
    """``mx.load`` (Step 3) is LAZY — it only reads the safetensors header,
    not tensor DATA. A truncated/lazily-unreadable sidecar with a VALID
    header sails through every earlier shape/dtype check and only raises at
    Step 4's ``mtp.load_weights(...)`` + ``mx.eval(mtp.parameters())``
    materialization — a SECOND escape point from the same fail-safe that
    ``test_inject_refuses_corrupt_sidecar_file_without_raising`` covers for
    the eager ``mx.load`` header-read path (codex round-2 review on #1201).
    Pin the materialization guard independent of a real truncated file:
    monkeypatch ``mx.eval`` (as ``inject_mtp_support`` resolves it — it does
    a local ``import mlx.core as mx``, the same shared module object) to
    raise a sentinel, using an OTHERWISE-VALID sidecar so, absent the
    try/except, this call would raise straight out of ``inject_mtp_support``.
    """
    import mlx.core as _mx
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    # A valid full-precision sidecar that exactly matches the MTP module's
    # parameter tree — built + saved BEFORE the patch, so this setup's own
    # eval uses the real ``mx.eval``.
    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _mx.eval(template.parameters())
    flat = dict(tree_flatten(template.parameters()))
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    class _MaterializeBoomError(RuntimeError):
        pass

    def _boom(*_a, **_k):
        raise _MaterializeBoomError("sentinel: truncated tensor data")

    # Set AFTER the setup eval above so only the in-inject materialization
    # call hits the boom.
    monkeypatch.setattr(_mx, "eval", _boom)

    # Handler must convert the raise into a clean False, not propagate.
    assert inject_mtp_support(base, mtp_sidecar=str(tmp_path)) is False
    assert not hasattr(base, "mtp"), (
        "inject refused but still attached an MTP module — the refusal must "
        "leave the base model untouched (plain path)."
    )


def test_inject_refuses_sidecar_with_shape_mismatched_non_fc_tensor(tmp_path):
    """The fc-derived quantization is applied UNIFORMLY across the module.
    A sidecar whose fc is well-formed but another tensor's shape disagrees
    with that uniform packing (a mixed-bit / differently-grouped /
    corrupted non-fc layer) must be caught by the post-quantize shape
    check and refused — loading it would recreate the quantized_matmul
    mismatch mid-generation, not just at fc.
    """
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    _nn.quantize(base.model, group_size=32, bits=8)

    # A valid 4-bit sidecar; fc is left intact so inference derives 4-bit.
    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _nn.quantize(template, group_size=32, bits=4)
    _mx.eval(template.parameters())
    flat = dict(tree_flatten(template.parameters()))

    # Corrupt ONE non-fc tensor's shape (key stays present, so the coverage
    # check passes and the SHAPE check is what must reject it).
    victim = next(
        k for k in sorted(flat) if not k.startswith("fc.") and k.endswith(".weight")
    )
    orig = flat[victim]
    flat[victim] = _mx.zeros(
        (int(orig.shape[0]) + 1, *(int(d) for d in orig.shape[1:])),
        dtype=orig.dtype,
    )
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    assert inject_mtp_support(base, mtp_sidecar=str(tmp_path)) is False
    assert not hasattr(base, "mtp")


def test_inject_refuses_sidecar_with_dtype_mismatched_packed_weight(tmp_path):
    """A shape-correct sidecar can still smuggle a wrong-*dtype* packed
    weight. MLX packs quantized ``weight`` as unsigned 32-bit; a same-shape
    ``float32``/``int32`` packed weight passes the coverage + shape checks
    but ``load_weights(strict=False)`` installs it without casting and the
    first ``mx.quantized_matmul`` rejects it (the same empty-response class).
    The by-role dtype check must catch and refuse it.
    """
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    _nn.quantize(base.model, group_size=32, bits=8)

    # A valid 4-bit sidecar; fc is intact so inference derives 4-bit.
    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _nn.quantize(template, group_size=32, bits=4)
    _mx.eval(template.parameters())
    flat = dict(tree_flatten(template.parameters()))

    # Corrupt ONE packed weight's DTYPE (uint32 -> float32) while keeping its
    # shape identical, so coverage + shape checks pass and only the dtype
    # check can reject it.
    victim = next(
        k
        for k in sorted(flat)
        if k.endswith(".weight") and _mx.issubdtype(flat[k].dtype, _mx.integer)
    )
    flat[victim] = flat[victim].astype(_mx.float32)
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    assert inject_mtp_support(base, mtp_sidecar=str(tmp_path)) is False
    assert not hasattr(base, "mtp")


def test_inject_refuses_mixed_bit_sidecar_fail_safe(tmp_path):
    """The fc-derived quantization is applied UNIFORMLY (intentional scope).
    A hypothetical mixed-bit sidecar — FP ``fc`` but a quantized decoder —
    must NOT be silently mis-packed: ``_infer_sidecar_fc_quantization``
    reads FP fc and leaves the module FP, then the Step 4 shape check sees
    the packed decoder tensors disagree with the FP module and REFUSES
    (clean non-MTP fallback), rather than crashing at the first draft step.
    This documents the fail-safe boundary for non-uniform sidecars.
    """
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    _nn.quantize(base.model, group_size=32, bits=8)

    # Mixed-bit sidecar: quantize EVERYTHING EXCEPT fc, so fc ships FP while
    # the decoder layers ship 4-bit packed tensors.
    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _nn.quantize(
        template,
        group_size=32,
        bits=4,
        class_predicate=lambda path, m: (
            hasattr(m, "to_quantized") and not path.startswith("fc")
        ),
    )
    _mx.eval(template.parameters())
    flat = dict(tree_flatten(template.parameters()))
    # Sanity: fc stays FP (no fc.scales), decoder is packed (has scales).
    assert not any(k.startswith("fc.scales") for k in flat)
    assert any(k.endswith(".scales") and not k.startswith("fc.") for k in flat)
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    assert inject_mtp_support(base, mtp_sidecar=str(tmp_path)) is False
    assert not hasattr(base, "mtp")


def test_inject_refuses_when_module_quantize_raises_fail_safe(tmp_path):
    """A sidecar whose ``fc`` is validly packed at group_size=128 infers
    ``group_size=128``, but the MTP module's narrower sibling leaves (only
    ``hidden_size``-wide in this fixture) are NOT divisible by 128, so
    ``nn.quantize(mtp, group_size=128)`` RAISES during Step 3 — before the
    Step 4 shape/dtype refusal can run. That exception must be caught and
    turned into a clean refusal (return False → non-MTP fallback), never
    propagated (an uncaught raise aborts the request at the first draft
    step: the empty-response class this fix closes).
    """
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    _nn.quantize(base.model, group_size=32, bits=8)

    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    fc_out_dims, fc_in_dims = (int(d) for d in template.fc.weight.shape)

    # Standalone group_size=128 fc (the fc IS wide enough for 128). The
    # sidecar only needs the fc tensors — the crash fires in Step 3's
    # module-wide quantize, before any other tensor is consulted.
    fc = _nn.Linear(fc_in_dims, fc_out_dims, bias=False)
    qfc = _nn.QuantizedLinear.from_linear(fc, 128, 4)
    _mx.eval(qfc.parameters())
    flat = {f"fc.{k}": v for k, v in dict(tree_flatten(qfc.parameters())).items()}
    assert "fc.scales" in flat
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    # Must not raise; must refuse cleanly.
    assert inject_mtp_support(base, mtp_sidecar=str(tmp_path)) is False
    assert not hasattr(base, "mtp")


def test_inject_catches_module_quantize_exception_deterministically(
    tmp_path, monkeypatch
):
    """Pin the Step 3 quantize exception handler independent of MLX's own
    group-size validation: monkeypatch ``nn.quantize`` so the MTP-module
    quantize RAISES a sentinel, then assert ``inject_mtp_support`` swallows
    it and returns ``False`` (never propagates). Uses an OTHERWISE-VALID
    4-bit sidecar so, absent the handler, the call would raise straight out
    of ``inject_mtp_support`` — this test fails (errors) if the try/except
    is removed, whereas the real-crash fixture could in principle be masked
    by a later refusal or a future MLX that tolerates the packing.
    """
    import mlx.core as _mx
    import mlx.nn as _nn
    from mlx.utils import tree_flatten

    from rapid_mlx.spec_decode.mtp.head import build_mtp_module
    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    try:
        base = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    # Real quantize for setup (base) + sidecar construction, BEFORE the patch.
    _nn.quantize(base.model, group_size=32, bits=8)
    args = base.args
    template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _nn.quantize(template, group_size=32, bits=4)
    _mx.eval(template.parameters())
    flat = dict(tree_flatten(template.parameters()))
    _mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

    class _QuantizeBoomError(RuntimeError):
        pass

    def _boom(*_a, **_k):
        raise _QuantizeBoomError("sentinel: module quantize failed")

    # ``inject_mtp_support`` does ``import mlx.nn as nn`` internally, so its
    # ``nn`` IS this same ``mlx.nn`` module object — patching the attribute
    # here is what its ``nn.quantize(mtp, ...)`` call resolves to. Set AFTER
    # the setup quantizes above so only the in-inject MTP call hits the boom.
    monkeypatch.setattr(_nn, "quantize", _boom)

    # Handler must convert the raise into a clean False, not propagate.
    assert inject_mtp_support(base, mtp_sidecar=str(tmp_path)) is False
    assert not hasattr(base, "mtp")


# ---------------------------------------------------------------------------
# 8. Generator loop — chain MTP verify/accept logic with mocked model
# ---------------------------------------------------------------------------


class _MockedQwen35Model:
    """Minimal model shell that satisfies the ``mtp_generate_step`` contract.

    The contract surface required:

    * ``__call__(inputs, cache, return_hidden, n_confirmed,
      input_embeddings)`` → returns ``(logits, hidden)`` when
      ``return_hidden=True``.
    * ``mtp_forward(hidden, next_token_ids, mtp_cache)`` → returns
      logits.
    * ``make_mtp_cache()`` → returns an empty list (no MTP cache state).
    * ``layers`` property → returns a list of length 0 so the
      generator builds a fresh ``[]`` model cache (the mock doesn't
      need cache state to script its logits).

    Scripting model
    ---------------

    ``backbone_outputs`` is a list of per-position token IDs the
    backbone returns. The mock consumes one token per (call,
    position):

    * Cold-start: backbone called with ``S=1, n_predict=1`` → consume
      1 token (the primary).
    * Verify: backbone called with ``S=2, n_predict=2`` → consume 2
      tokens (verify_pred at pos 0, bonus_tok at pos 1).

    ``mtp_outputs`` is a list of draft tokens the MTP head returns.
    Consumed one per ``mtp_forward`` call.

    Both lists are padded with ``-1`` if the script runs short; the
    test asserts that the early-return matched the expected token
    BEFORE the script runs out.
    """

    def __init__(
        self,
        backbone_outputs: list[int],
        mtp_outputs: list[int],
        vocab: int = 32,
        hidden_size: int = 8,
    ):
        self._backbone = list(backbone_outputs)
        self._mtp = list(mtp_outputs)
        self._backbone_cursor = 0
        self._mtp_cursor = 0
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.layers = []

    def _logits_for_positions(self, target_ids: list[int], batch: int) -> mx.array:
        """Build logits where each position's argmax is the matching target."""
        out_rows = []
        for tid in target_ids:
            row = mx.zeros((batch, self.vocab))
            row = row + mx.where(
                mx.arange(self.vocab)[None, :] == tid,
                mx.array(50.0),
                mx.array(0.0),
            )
            out_rows.append(row)
        return mx.stack(out_rows, axis=1)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        B, S = inputs.shape
        # Consume S positions from the backbone script.
        targets = []
        for _ in range(S):
            if self._backbone_cursor < len(self._backbone):
                targets.append(self._backbone[self._backbone_cursor])
                self._backbone_cursor += 1
            else:
                targets.append(0)
        logits = self._logits_for_positions(targets, B)
        hidden = mx.zeros((B, S, self.hidden_size))
        if return_hidden:
            return logits, hidden
        return logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        B = next_token_ids.shape[0]
        S = next_token_ids.shape[1]
        # Consume S draft tokens. For cache_commit calls (S==2) only
        # the LAST position's logits are read by the generator
        # (``mtp_logits = mtp_logits[:, -1, :]``), so the first
        # position can be any sentinel.
        targets = []
        for _ in range(S):
            if self._mtp_cursor < len(self._mtp):
                targets.append(self._mtp[self._mtp_cursor])
                self._mtp_cursor += 1
            else:
                targets.append(0)
        return self._logits_for_positions(targets, B)

    def make_mtp_cache(self):
        return []


class _CountingKVCache:
    """Tiny trimmable cache double for MTP rollback accounting tests."""

    def __init__(self):
        self.offset = 0
        self.trim_calls: list[int] = []

    def is_trimmable(self):
        return True

    def can_trim(self, n):
        return 0 <= n <= self.offset

    def size(self):
        return self.offset

    def trim(self, n):
        if n < 0:
            raise AssertionError(f"negative trim: {n}")
        self.trim_calls.append(n)
        trimmed = min(self.offset, n)
        self.offset -= trimmed
        return trimmed


class _CacheAdvancingQwen35Model(_MockedQwen35Model):
    """Mock that advances supplied cache doubles on each forward."""

    mtp_prompt_lookup_supported = True

    def __init__(self, backbone_outputs: list[int], mtp_outputs: list[int]):
        super().__init__(backbone_outputs, mtp_outputs)
        self.layers = [object()]

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        if cache is not None:
            for c in cache:
                c.offset += int(inputs.shape[1])
        return super().__call__(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
            n_confirmed=n_confirmed,
        )

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        for c in mtp_cache:
            c.offset += int(next_token_ids.shape[1])
        return super().mtp_forward(hidden, next_token_ids, mtp_cache)


def test_generator_emits_first_token_from_backbone_then_draft():
    """First yield comes from the backbone (``from_draft=False``); on
    accept the second yield is the MTP draft (``from_draft=True``).

    Sequence (length-1 prompt: prefill is a no-op, decode starts on
    the single prompt token):

      cold-start backbone (S=1, n_predict=1)
        → consumes backbone[0]=7 (primary emit)
      MTP head (N=1)
        → consumes mtp[0]=11 (draft proposal)
      verify backbone (S=2, n_predict=2, n_confirmed=1)
        → consumes backbone[1]=11 (verify_pred — matches draft → accept)
        → consumes backbone[2]=13 (bonus_tok)

    Yields: (7, False), (11, True — accepted draft), (13, False — bonus).

    A length-1 prompt is used because the prefill loop processes
    ``y[:n]`` for ``n = min(prefill_step_size, prompt_len - 1)`` and
    consumes both backbone and MTP slots during prefill — that
    complicates the script. With prompt length 1, ``prefill_step``
    skips and the decode loop sees the single prompt token directly.
    """
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    backbone = [7, 11, 13]
    mtp = [11]
    model = _MockedQwen35Model(backbone, mtp)

    counter = MTPAcceptCounter()
    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    # 0.9.13 PR-B: default auto-K controller bootstraps with K=0
    # rounds, which would emit backbone[1]/backbone[2] as plain-decode
    # tokens instead of the draft-verify sequence this test asserts.
    # ``disable_auto_k=True`` pins K=1 chain-of-1 — the pre-PR-B
    # behavior the test was authored against.
    for tok, _logprobs, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=3,
        accept_counter=counter,
        disable_auto_k=True,
    ):
        emitted.append((tok, from_draft))

    assert emitted[0] == (7, False), f"primary emit: {emitted}"
    assert emitted[1] == (11, True), (
        "draft == verify_pred at temp=0 → accept; second yield must be the "
        f"accepted draft with from_draft=True. Got {emitted}"
    )
    assert emitted[2] == (13, False), f"bonus emit: {emitted}"
    snap = counter.snapshot()
    assert snap.attempts == 1
    assert snap.accepts == 1
    assert snap.tokens_saved == 1
    # #3155: one verify call at depth 1, fully accepted -> bonus token
    assert snap.verify_calls == 1
    assert snap.bonus_tokens == 1
    assert snap.correction_tokens == 0
    assert snap.drafted_by_depth == ((1, 1),)
    assert snap.accepted_by_depth == ((1, 1),)


def test_generator_sampled_verify_accepts_matching_draft():
    """The probabilistic verifier is active when temperature is non-zero."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    counter = MTPAcceptCounter()
    emitted = list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            _MockedQwen35Model([7, 11, 13], [11]),
            max_tokens=3,
            temp=0.7,
            top_p=0.95,
            disable_auto_k=True,
            accept_counter=counter,
        )
    )

    # The scripted logits put effectively all mass on these tokens, while the
    # non-zero temperature forces the probabilistic accept/residual branch.
    assert [(token, drafted) for token, _lp, drafted in emitted] == [
        (7, False),
        (11, True),
        (13, False),
    ]
    assert counter.snapshot().accepts == 1


def test_generator_sampled_verify_uses_request_local_rng():
    """Sampled MTP consumes only the request-owned carried key."""
    from rapid_mlx._seeded_sampler import RequestSeededRNG
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    lane_rng = RequestSeededRNG(42)
    emitted = list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            _MockedQwen35Model([7, 11, 13], [11]),
            max_tokens=3,
            temp=0.7,
            top_p=0.95,
            disable_auto_k=True,
            accept_counter=MTPAcceptCounter(),
            lane_rng=lane_rng,
        )
    )

    assert [(token, drafted) for token, _lp, drafted in emitted] == [
        (7, False),
        (11, True),
        (13, False),
    ]
    assert lane_rng.draws > 0


def test_mtp_request_rng_and_greedy_sampler_publish_state_contract():
    """Apple MTP lane covers the request-local state exposed to the scheduler."""
    from rapid_mlx._seeded_sampler import RequestSeededRNG, make_seeded_sampler

    carried = RequestSeededRNG(42)
    initial = carried.key
    subkey = carried.next_key()
    assert carried.draws == 1
    assert carried.key is not initial
    assert subkey is not None

    greedy = make_seeded_sampler(seed=42, temperature=0.0)
    assert greedy.lane_rng is None
    assert greedy.request_seeded is True


def test_generator_accepted_draft_reports_target_logprobs(monkeypatch):
    """An accepted proposal exposes p_target, never the drafter's q row."""
    import mlx_lm.sample_utils as sample_utils

    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    class _DistinctDistributionsModel(_MockedQwen35Model):
        def _rows(self, target_ids, batch, boost):
            rows = []
            for token_id in target_ids:
                row = mx.zeros((batch, self.vocab))
                row = row + mx.where(
                    mx.arange(self.vocab)[None, :] == token_id,
                    mx.array(boost),
                    mx.array(0.0),
                )
                rows.append(row)
            return mx.stack(rows, axis=1)

        def _logits_for_positions(self, target_ids, batch):
            # Target p: deliberately soft so it differs materially from q.
            return self._rows(target_ids, batch, 1.0)

        def mtp_forward(self, hidden, next_token_ids, mtp_cache):
            batch, positions = next_token_ids.shape
            targets = []
            for _ in range(positions):
                if self._mtp_cursor < len(self._mtp):
                    targets.append(self._mtp[self._mtp_cursor])
                    self._mtp_cursor += 1
                else:
                    targets.append(0)
            # Drafter q: a much sharper distribution around the same proposal.
            return self._rows(targets, batch, 8.0)

    monkeypatch.setattr(
        sample_utils,
        "categorical_sampling",
        lambda logits, temp: mx.argmax(logits, axis=-1),
    )
    monkeypatch.setattr(
        generator_mod.mx.random,
        "uniform",
        lambda *args, **kwargs: mx.zeros(kwargs.get("shape", ())),
    )

    emitted = list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            _DistinctDistributionsModel([7, 11, 13], [11]),
            max_tokens=3,
            temp=0.7,
            disable_auto_k=True,
            accept_counter=MTPAcceptCounter(),
        )
    )

    accepted_id, served_logprobs, from_draft = emitted[1]
    assert (accepted_id, from_draft) == (11, True)
    target_logits = mx.where(mx.arange(32) == 11, mx.array(1.0), mx.array(0.0))
    target_logprobs = target_logits - mx.logsumexp(target_logits)
    draft_logits = mx.where(mx.arange(32) == 11, mx.array(8.0), mx.array(0.0))
    draft_logprobs = draft_logits - mx.logsumexp(draft_logits)
    assert mx.allclose(served_logprobs, target_logprobs)
    assert not mx.allclose(served_logprobs, draft_logprobs)


def test_generator_sampled_k3_draws_acceptance_independently_per_position(
    monkeypatch,
):
    """K>1 rejection sampling must not correlate acceptance decisions."""
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    real_uniform = generator_mod.mx.random.uniform
    shapes: list[tuple[int, ...] | None] = []

    def recording_uniform(*args, **kwargs):
        shapes.append(kwargs.get("shape"))
        return real_uniform(*args, **kwargs)

    monkeypatch.setattr(generator_mod.mx.random, "uniform", recording_uniform)
    list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            _MockedQwen35Model([7, 11, 12, 13, 14], [11, 12, 13]),
            max_tokens=4,
            max_k=3,
            temp=0.7,
            top_p=0.95,
            disable_auto_k=True,
            accept_counter=MTPAcceptCounter(),
        )
    )

    assert (3,) in shapes


def test_generator_penalty_processor_continues_existing_token_context():
    """The first MTP target sample sees the same history as mlx-lm."""
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    observed: list[list[int]] = []

    def recorder(tokens, logits):
        observed.append(tokens.tolist())
        return logits

    list(
        mtp_generate_step(
            mx.array([7], dtype=mx.uint32),
            _MockedQwen35Model([11], []),
            max_tokens=1,
            logits_processors=[recorder],
            initial_tokens=[3, 5],
            disable_auto_k=True,
        )
    )

    assert observed == [[3, 5, 7]]


def test_generator_penalty_processor_carries_full_k3_draft_history():
    """Each chained draft sees main_tok plus every preceding proposal."""
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    observed: list[list[int]] = []

    def recorder(tokens, logits):
        observed.append(tokens.tolist())
        return logits

    list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            _MockedQwen35Model([7, 11, 12, 13, 14], [11, 12, 13]),
            max_tokens=4,
            max_k=3,
            logits_processors=[recorder],
            initial_tokens=[3, 5],
            disable_auto_k=True,
        )
    )

    # Cold-start target sample, then the three sequential MTP draft samples.
    # Later entries are the batched target verification pass and are not part
    # of the chain-local-history assertion.
    assert observed[:4] == [
        [3, 5, 1],
        [3, 5, 1, 7],
        [3, 5, 1, 7, 11],
        [3, 5, 1, 7, 11, 12],
    ]


def test_prompt_lookup_point_mass_residual_removes_proposed_token():
    from rapid_mlx.spec_decode.mtp.generator import (
        _point_mass_residual_distribution,
    )

    target = mx.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    actual = _point_mass_residual_distribution(
        mx.log(target), mx.array([0, 1], dtype=mx.int32)
    )
    expected = mx.array([[0.0, 0.75, 0.25], [0.4, 0.0, 0.6]])
    mx.eval(actual)
    assert bool(mx.allclose(actual, expected, atol=1e-6).item())


def test_generator_fixed_k3_accepts_three_drafts_in_one_verify():
    """Fixed-depth mode must honor max_k=3 instead of silently using K=1."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    counter = MTPAcceptCounter()
    emitted = list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            _MockedQwen35Model([7, 11, 12, 13, 14], [11, 12, 13]),
            max_tokens=4,
            max_k=3,
            disable_auto_k=True,
            accept_counter=counter,
        )
    )

    assert [(tok, draft) for tok, _lp, draft in emitted] == [
        (7, False),
        (11, True),
        (12, True),
        (13, True),
    ]
    snap = counter.snapshot()
    assert snap.attempts == 3
    assert snap.accepts == 3


@pytest.mark.parametrize(
    ("max_k", "committed", "drafts", "targets", "intervention_position"),
    [
        (2, list(range(24)) + list(range(23)), [23, 0], [23, 0, 7], 1),
        (3, list(range(24)) + list(range(22)), [22, 23, 0], [22, 23, 0, 7], 2),
    ],
)
def test_generator_commits_tool_guard_state_only_as_tokens_are_delivered(
    max_k, committed, drafts, targets, intervention_position
):
    """K=2/K=3 verification cannot commit a not-yet-delivered guard hit."""
    from rapid_mlx.repetition_guard import AgentRepetitionLogitsProcessor
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    near_loop = list(range(24)) + list(range(25 - max_k))

    def start():
        history = list(committed)
        processor = AgentRepetitionLogitsProcessor(history)
        gen = mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            _MockedQwen35Model([7, *targets], drafts),
            max_tokens=max_k + 2,
            max_k=max_k,
            logits_processors=[processor],
            initial_tokens=[1],
            disable_auto_k=True,
            accept_counter=MTPAcceptCounter(),
        )
        assert next(gen)[0] == 7
        # The first sample above is committed before the generator is resumed.
        history[:] = near_loop
        return gen, processor

    # Cancelling before the later intervention position must leave the guard
    # exactly at the last delivered prefix.
    cancelled, cancelled_processor = start()
    for position in range(intervention_position):
        token, _lp, from_draft = next(cancelled)
        assert token == drafts[position]
        assert from_draft is True
        assert cancelled_processor.interventions == 0
    cancelled.close()
    assert cancelled_processor.interventions == 0

    # Delivering that same position commits the intervention exactly once.
    gen, processor = start()
    for _ in range(intervention_position):
        next(gen)
    token, _lp, from_draft = next(gen)
    assert from_draft is True
    assert token != 0  # the loop-extending target/draft token was masked
    assert processor.interventions == 1


def test_generator_rejection_discards_later_tool_guard_state():
    """A guard hit on a rejected K=3 suffix never consumes its safety cap."""
    from rapid_mlx.repetition_guard import AgentRepetitionLogitsProcessor
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    pattern = list(range(24))
    committed = pattern + pattern[:-2]
    processor = AgentRepetitionLogitsProcessor(committed)
    # Target rejects d1 immediately, but its batched rows still evaluate the
    # later d1+d2 prefix that would otherwise arm the repetition guard.
    gen = mtp_generate_step(
        mx.array([1], dtype=mx.uint32),
        _MockedQwen35Model([7, 9, 23, 0, 7], [22, 23, 0]),
        max_tokens=2,
        max_k=3,
        logits_processors=[processor],
        initial_tokens=[1],
        disable_auto_k=True,
        accept_counter=MTPAcceptCounter(),
    )

    assert next(gen)[0] == 7
    token, _lp, from_draft = next(gen)
    assert (token, from_draft) == (9, False)
    assert processor.interventions == 0


def test_tool_guard_ordinary_decode_still_uses_committed_output():
    """The refactored ordinary entry point retains its scheduler-owned history."""
    from rapid_mlx.repetition_guard import AgentRepetitionLogitsProcessor

    pattern = list(range(12))
    processor = AgentRepetitionLogitsProcessor(pattern * 5)
    logits = processor(mx.array([999]), mx.zeros((1, 32)))
    mx.eval(logits)
    assert float(logits[0, pattern[0]].item()) == float("-inf")


def test_generator_restores_tool_guard_state_when_target_processor_raises():
    """A verify-time processor exception restores the pre-proposal boundary."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    class FusedDraftModel(_MockedQwen35Model):
        def mtp_greedy(self, hidden, next_token_ids, mtp_cache):
            del mtp_cache
            batch, positions = next_token_ids.shape
            tokens = []
            for _ in range(positions):
                tokens.append(self._mtp[self._mtp_cursor])
                self._mtp_cursor += 1
            return (
                mx.array([tokens] * batch, dtype=mx.uint32),
                mx.zeros((batch, positions, hidden.shape[-1])),
            )

    class RaisingTransactionalProcessor:
        def __init__(self):
            self.state = 0
            self.armed = False

        def __call__(self, _tokens, logits):
            return logits

        def mtp_apply(self, _tokens, _tentative, logits):
            self.state += 1
            if self.armed:
                raise RuntimeError("sentinel target processor failure")
            return logits

        def mtp_snapshot_state(self):
            return self.state

        def mtp_restore_state(self, state):
            self.state = state

    processor = RaisingTransactionalProcessor()
    gen = mtp_generate_step(
        mx.array([1], dtype=mx.uint32),
        FusedDraftModel([7, 11, 13], [11]),
        max_tokens=3,
        max_k=1,
        logits_processors=[processor],
        initial_tokens=[1],
        disable_auto_k=True,
        accept_counter=MTPAcceptCounter(),
    )

    assert next(gen)[0] == 7
    assert processor.state == 1
    processor.armed = True
    with pytest.raises(RuntimeError, match="sentinel target processor failure"):
        next(gen)
    assert processor.state == 1


def test_generator_budget_forces_think_end_on_the_verify_row_and_rolls_back_drafts():
    """#3044: the thinking budget is a transactional MTP processor.

    Budget of one think token, prompt seeded ``<think>``. The cold-start step
    emits token 7 (1/1 spent). On the verify step the budget forces
    ``</think>`` on row 0, so the draft (11) is rejected and ``</think>`` is
    resampled; row 1 had tentatively counted the draft, and the generator's
    restore to the accepted row-0 boundary must drop that count again.
    """
    from rapid_mlx.api.reasoning_budget import ReasoningBudgetLogitsProcessor
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    think_end = 5

    class FusedDraftModel(_MockedQwen35Model):
        def mtp_greedy(self, hidden, next_token_ids, mtp_cache):
            del mtp_cache
            batch, positions = next_token_ids.shape
            tokens = []
            for _ in range(positions):
                tokens.append(self._mtp[self._mtp_cursor])
                self._mtp_cursor += 1
            return (
                mx.array([tokens] * batch, dtype=mx.uint32),
                mx.zeros((batch, positions, hidden.shape[-1])),
            )

    budget = ReasoningBudgetLogitsProcessor(think_end, 1, seeded_thinking=True)
    gen = mtp_generate_step(
        mx.array([1], dtype=mx.uint32),
        FusedDraftModel([7, 11, 13, 20, 21], [11, 12, 14]),
        max_tokens=3,
        max_k=1,
        logits_processors=[budget],
        initial_tokens=[1],
        disable_auto_k=True,
        accept_counter=MTPAcceptCounter(),
    )

    emitted = [next(gen)[0], next(gen)[0]]
    assert emitted == [7, think_end]
    # Row 0's boundary (prompt + token 7 committed, 1/1 spent) is what
    # survives; the rejected draft's tentative count (2/1) is gone.
    assert budget._committed == budget._prompt_len + 1
    assert budget._think_count == 1
    assert budget._ended is False
    # The delivered ``</think>`` is committed on the next step: generation
    # phase, no further counting, the script's next target (20) passes.
    assert next(gen)[0] == 20
    assert budget._ended is True
    assert budget._think_count == 1


def test_quantized_argmax_matches_materialized_qlinear_logits():
    """The fused greedy kernel must select the exact qlinear argmax."""
    import mlx.nn as nn

    from rapid_mlx.spec_decode.mtp.quantized_argmax import quantized_argmax

    dense = nn.Linear(1024, 4096, bias=False)
    dense.set_dtype(mx.bfloat16)
    head = nn.QuantizedLinear.from_linear(dense, group_size=64, bits=4)
    hidden = mx.random.normal((1, 3, 1024)).astype(mx.bfloat16)

    fused = quantized_argmax(head, hidden)
    reference = mx.argmax(head(hidden), axis=-1)
    assert fused is not None
    mx.eval(fused, reference)
    assert fused.tolist() == reference.tolist()


def test_generator_k3_restores_ssm_state_at_partial_accept_boundary():
    """Rejecting draft 2 restores GDN state after y + accepted draft 1."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    class SnapshotSSMCache:
        rollback_state = None

        def __init__(self):
            self.cache = [mx.array([0]), mx.array([0])]

        def __getitem__(self, idx):
            return self.cache[idx]

        def __setitem__(self, idx, value):
            self.cache[idx] = value

        def is_trimmable(self):
            return False

    class SnapshotModel(_MockedQwen35Model):
        def __init__(self):
            super().__init__([7, 11, 12, 13, 14], [11, 99, 13])
            self.layers = [object()]

        def __call__(self, inputs, cache=None, **kwargs):
            result = super().__call__(inputs, cache=cache, **kwargs)
            if cache and inputs.shape[1] > 2:
                c = cache[0]
                c.rollback_state = [
                    (mx.array([101 + i]), mx.array([201 + i]))
                    for i in range(inputs.shape[1] - 1)
                ]
                c[0], c[1] = mx.array([999]), mx.array([999])
            return result

    ssm = SnapshotSSMCache()
    emitted = list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            SnapshotModel(),
            prompt_cache=[ssm],
            max_tokens=3,
            max_k=3,
            disable_auto_k=True,
            accept_counter=MTPAcceptCounter(),
        )
    )

    assert [tok for tok, _lp, _draft in emitted] == [7, 11, 12]
    # keep=2 positions (y + one accepted draft) selects snapshot index 1.
    assert ssm[0].item() == 102
    assert ssm[1].item() == 202
    assert ssm.rollback_state is None


def test_generator_legacy_ssm_snapshot_is_limited_to_one_token():
    """The legacy tuple restores K=1 and fails closed for a K>1 rollback."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    class LegacySSMCache:
        rollback_state = None

        def __init__(self):
            self.cache = [mx.array([0]), mx.array([0])]

        def __getitem__(self, idx):
            return self.cache[idx]

        def __setitem__(self, idx, value):
            self.cache[idx] = value

        def is_trimmable(self):
            return False

    class LegacySnapshotModel(_MockedQwen35Model):
        def __init__(self, *, max_k):
            super().__init__([7, 99, 0, 0, 0], list(range(11, 11 + max_k)))
            self.layers = [object()]

        def __call__(self, inputs, cache=None, **kwargs):
            result = super().__call__(inputs, cache=cache, **kwargs)
            if cache and inputs.shape[1] > 1:
                cache[0].rollback_state = (mx.array([101]), mx.array([201]))
                cache[0][0], cache[0][1] = mx.array([999]), mx.array([999])
            return result

    k1_cache = LegacySSMCache()
    list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            LegacySnapshotModel(max_k=1),
            prompt_cache=[k1_cache],
            max_tokens=2,
            max_k=1,
            disable_auto_k=True,
            accept_counter=MTPAcceptCounter(),
        )
    )
    assert k1_cache[0].item() == 101
    assert k1_cache[1].item() == 201

    with pytest.raises(AssertionError, match="only roll back one token"):
        list(
            mtp_generate_step(
                mx.array([1], dtype=mx.uint32),
                LegacySnapshotModel(max_k=3),
                prompt_cache=[LegacySSMCache()],
                max_tokens=4,
                max_k=3,
                disable_auto_k=True,
                accept_counter=MTPAcceptCounter(),
            )
        )


def test_generator_rolls_back_verify_round_on_early_materialization_abort(
    monkeypatch,
):
    """Abort after target verify advances caches, before accept state is fresh.

    The first generator step commits the primary token and builds one MTP draft.
    The second step runs the target verify forward over ``[primary, draft]``.
    We then force the host sync/materialization boundary to raise. The guard
    must keep the committed primary in the target cache while dropping the
    uncommitted draft from both target and MTP caches.
    """

    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    model_cache = _CountingKVCache()
    mtp_cache = _CountingKVCache()
    model = _CacheAdvancingQwen35Model([7, 11, 13], [11])
    prompt = mx.array([1], dtype=mx.uint32)

    gen = mtp_generate_step(
        prompt,
        model,
        max_tokens=3,
        prompt_cache=[model_cache, mtp_cache],
        accept_counter=MTPAcceptCounter(),
        disable_auto_k=True,
    )

    assert next(gen)[0] == 7
    assert model_cache.offset == 1
    # The draft for the next round is built BEFORE this round's token is
    # handed over, so the MTP cache already holds its uncommitted position
    # when the caller gets 7. That position is exactly what the abort below
    # has to roll back.
    assert mtp_cache.offset == 1

    eval_calls = 0

    def _boom_on_verify_sync(*_args, **_kwargs):
        nonlocal eval_calls
        eval_calls += 1
        # Draft chaining stays lazy now, so the first materialization after
        # resuming the generator is the batched target-verify boundary.
        if eval_calls >= 1:
            raise RuntimeError("sentinel materialization abort")

    monkeypatch.setattr(generator_mod.mx, "eval", _boom_on_verify_sync)

    with pytest.raises(RuntimeError, match="sentinel materialization abort"):
        next(gen)

    assert model_cache.offset == 2
    assert model_cache.trim_calls[-1] == 1
    assert mtp_cache.offset == 0
    assert mtp_cache.trim_calls[-1] == 1


def test_generator_rejection_path_does_not_count_as_accept():
    """When draft != verify_pred at temp=0 the generator takes the
    reject branch — counter shows attempt without accept.

    Sequence:

      cold-start backbone → 7 (primary)
      MTP head → draft=11
      verify backbone → verify_pred=12 (≠ draft → reject), bonus=99 (unused)

    Yields: (7, False), (12, False).
    """
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    backbone = [7, 12, 99]  # 99 is for the bonus slot — unused on reject
    mtp = [11, 22]  # 22 is for the next draft after reject (cold-start MTP)
    model = _MockedQwen35Model(backbone, mtp)

    counter = MTPAcceptCounter()
    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    # 0.9.13 PR-B: disable auto-K to pin the K=1 chain-of-1 sequence
    # this test scripts against.
    for tok, _logprobs, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=2,
        accept_counter=counter,
        disable_auto_k=True,
    ):
        emitted.append((tok, from_draft))

    assert emitted[0] == (7, False), f"primary emit: {emitted}"
    assert emitted[1] == (12, False), (
        "On reject the generator yields the verify pred (not the rejected "
        f"draft) with from_draft=False. Got {emitted}"
    )
    snap = counter.snapshot()
    assert snap.attempts == 1
    assert snap.accepts == 0
    assert snap.tokens_saved == 0


def test_copy_draft_width_stays_inside_one_quantized_matmul_tile():
    """The widest copy-draft must verify in 32 rows, not 33.

    A proposal of N tokens is verified in one N+1 row forward, and MLX's
    ``quantized_matmul`` tiles rows in blocks of 32, so N = 32 buys a whole
    second tile for a single row. Measured on Qwen3.8-27B-4bit: 342.26 ms
    at 32 rows against 647.52 at 33. This is a cost-model contract, not a
    style preference -- a future rung of 32 would silently cost 89%.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_FREE_WIDTH_FLOOR,
        COPY_DRAFT_MIN_WIDTH,
        COPY_DRAFT_PROBE_WIDTH,
        MAX_COPY_DRAFT_TOKENS,
    )

    assert MAX_COPY_DRAFT_TOKENS == 31
    # Every width the sizer can hand back lands inside the same tile.
    assert COPY_DRAFT_MIN_WIDTH <= COPY_DRAFT_PROBE_WIDTH
    assert COPY_DRAFT_PROBE_WIDTH < COPY_DRAFT_FREE_WIDTH_FLOOR
    assert COPY_DRAFT_FREE_WIDTH_FLOOR <= MAX_COPY_DRAFT_TOKENS


def test_generator_honours_an_operator_ceiling_past_the_tile_edge(monkeypatch):
    """The proposal ceiling is the policy's, and an override is obeyed.

    ``RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS`` is how an operator says how
    far their workload's copies run, and that is knowledge this module does
    not have: rows inside a 32-row ``quantized_matmul`` tile are nearly free
    and the row that crosses an edge costs a whole tile, so 63 is as
    reasonable a ceiling as 31 for a workload whose matches run that long.
    The default is the first tile edge (pinned on the policy itself in
    ``test_inject_mtp_support_declares_prompt_lookup_policy``) and a silent
    clamp on top of the override would make the knob a lie -- an operator
    asking for 64 and getting 31 has no way to see it.

    What keeps the knob from being a footgun is the sizer, not a clamp:
    ``CopyDraftGate.width_cap`` proposes two rows until copies have been
    verified and widens only into acceptance the turn has actually earned,
    so a high ceiling costs nothing until the matches reach it.
    """
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_PROBE_WIDTH,
        CopyDraftGate,
    )

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "64")

    asked: list[int] = []
    real_propose = generator_mod.PromptLookupIndex.propose

    def recording_propose(self, generated, *, max_tokens=24):
        asked.append(max_tokens)
        return real_propose(self, generated, max_tokens=max_tokens)

    monkeypatch.setattr(generator_mod.PromptLookupIndex, "propose", recording_propose)

    # What the ceiling becomes: the sizer sees the operator's number and
    # returns the width the evidence supports, which is what turns into rows.
    sized: list[tuple[int, int]] = []
    real_width_cap = CopyDraftGate.width_cap

    def recording_width_cap(self, proposed):
        width = real_width_cap(self, proposed)
        sized.append((proposed, width))
        return width

    monkeypatch.setattr(CopyDraftGate, "width_cap", recording_width_cap)

    list(
        mtp_generate_step(
            mx.array([7, 8, 20, 21], dtype=mx.uint32),
            _CacheAdvancingQwen35Model(
                backbone_outputs=[0, 0, 0, 7, 8, 20, 21, 22],
                mtp_outputs=[0, 0, 0, 99, 20, 21],
            ),
            max_tokens=99,
            max_k=1,
            disable_auto_k=True,
            prompt_cache=[_CountingKVCache(), _CountingKVCache()],
            accept_counter=MTPAcceptCounter(),
        )
    )

    assert asked, "prompt lookup never consulted the index"
    # Exactly what the operator asked for: 64, with 99 tokens of generation
    # budget behind it. Anything smaller is a ceiling being applied twice.
    assert max(asked) == 64, (
        f"asked the index for up to {max(asked)} tokens against an operator "
        "ceiling of 64; the ceiling belongs to the policy and the override, "
        "not to the loop that consumes them"
    )

    # And a ceiling is still only a ceiling: every width that reached a
    # verify came from the sizer, which has no acceptance evidence on this
    # two-token match and hands back the probe width rather than the 64 rows
    # the operator allowed.
    assert sized, "the sizer was never consulted for a matched proposal"
    for proposed, width in sized:
        assert width <= min(proposed, COPY_DRAFT_PROBE_WIDTH), (proposed, width)


def test_generator_prompt_lookup_verifies_prompt_continuation(monkeypatch):
    """A prompt suffix match bypasses MTP drafting but still uses target verify."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "2")

    # The length-4 prompt prefills three positions first. Decode then emits
    # 7, rejects MTP's 99 in favour of target token 8, and finds prompt suffix
    # [7, 8] -> [20, 21]. The copied block is accepted only because the target
    # independently predicts 20 and 21; 22 is its ordinary bonus token.
    class CompositeCache:
        def __init__(self):
            self.caches = [_CountingKVCache()]

        @property
        def offset(self):
            return self.caches[0].offset

        @offset.setter
        def offset(self, value):
            self.caches[0].offset = value

    class SnapshottingModel(_CacheAdvancingQwen35Model):
        def __call__(self, inputs, cache=None, **kwargs):
            result = super().__call__(inputs, cache=cache, **kwargs)
            if cache and inputs.shape[1] > 2:
                cache[0].caches[0].rollback_state = object()
            return result

    model = SnapshottingModel(
        backbone_outputs=[0, 0, 0, 7, 8, 0, 20, 21, 22],
        mtp_outputs=[0, 0, 0, 99, 0, 0],
    )
    model_cache = CompositeCache()
    mtp_cache = _CountingKVCache()
    timing: dict[str, float] = {}

    emitted = list(
        mtp_generate_step(
            mx.array([7, 8, 20, 21], dtype=mx.uint32),
            model,
            max_tokens=5,
            max_k=1,
            disable_auto_k=True,
            prompt_cache=[model_cache, mtp_cache],
            accept_counter=MTPAcceptCounter(),
            timing_stats=timing,
            prompt_lookup_history=mx.array([7, 8, 20, 21], dtype=mx.uint32),
        )
    )

    assert [(token, drafted) for token, _lp, drafted in emitted] == [
        (7, False),
        (8, False),
        (20, True),
        (21, True),
        (22, False),
    ]
    assert timing["prompt_lookup_proposals"] == 1
    assert timing["prompt_lookup_drafted_tokens"] == 2
    assert timing["prompt_lookup_mtp_sync_seconds"] >= 0
    # Three prefill positions + one ordinary draft + two lookup-history sync
    # positions. Prompt lookup itself consumed no MTP proposal.
    assert model._mtp_cursor == 6
    assert model_cache.caches[0].trim_calls == [1]
    assert model_cache.caches[0].rollback_state is None
    assert mtp_cache.trim_calls == [1]
    assert mtp_cache.offset == 5


def test_generator_prompt_lookup_falls_through_when_cache_cannot_recover(monkeypatch):
    """A matching prompt never bypasses a failed full-width cache preflight."""
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import (
        _safe_prompt_lookup_draft_count,
        mtp_generate_step,
    )

    assert _safe_prompt_lookup_draft_count([object()], 2) == 0
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "2")
    monkeypatch.setattr(
        generator_mod,
        "_safe_prompt_lookup_draft_count",
        lambda *_a, **_kw: 0,
    )

    timing: dict[str, float] = {}
    list(
        mtp_generate_step(
            mx.array([7, 8, 20, 21], dtype=mx.uint32),
            _CacheAdvancingQwen35Model(
                backbone_outputs=[0, 0, 0, 7, 8, 20, 21, 22],
                mtp_outputs=[0, 0, 0, 99, 20, 21],
            ),
            max_tokens=3,
            max_k=1,
            disable_auto_k=True,
            prompt_cache=[_CountingKVCache(), _CountingKVCache()],
            accept_counter=MTPAcceptCounter(),
            timing_stats=timing,
        )
    )

    assert timing["prompt_lookup_cache_fallthroughs"] == 1


def test_safe_draft_count_admits_snapshot_only_caches_only_when_declared():
    """A snapshot-rollback cache is recoverable, but only on a model that says so.

    ``GatedDeltaNet``'s ``ArraysCache`` owns neither ``restore_rollback()``
    nor ``trim()``; its only restore point is the ``rollback_state`` the
    chunk-split verify patch writes during the forward. Both legacy checks
    therefore answer "cannot recover" and the guard rejects every proposal
    -- the bug that silently disabled prompt lookup on the whole Qwen3.5 /
    Qwen3.8 hybrid family. The capability is model-declared rather than
    sniffed off the cache because ``rollback_state`` lives on the CLASS: its
    presence says the slot exists, not that these layers fill it.
    """
    from rapid_mlx.spec_decode.mtp.generator import _safe_prompt_lookup_draft_count

    class _ArraysCacheLike:
        """No ``trim``, no ``restore_rollback`` -- only the snapshot slot."""

        rollback_state = None

        def __init__(self):
            self.offset = 16

    cache = [_ArraysCacheLike(), _ArraysCacheLike()]

    assert _safe_prompt_lookup_draft_count(cache, 4) == 0
    assert _safe_prompt_lookup_draft_count(cache, 4, snapshot_rollback=True) == 4
    # The flag must not conjure recoverability out of a cache that has no
    # restore point at all -- otherwise a rejected block would be unwindable
    # on paper and corrupt in practice.
    assert _safe_prompt_lookup_draft_count([object()], 4, snapshot_rollback=True) == 0


def test_generator_admits_prompt_lookup_on_declared_snapshot_rollback(monkeypatch):
    """A hybrid-SSM target that declares wide rollback stops falling through.

    The cache double is shaped like ``ArraysCache``: it owns no ``trim()``
    and no ``restore_rollback()``, and its only restore point is the
    ``rollback_state`` the chunk-split verify patch writes during the
    forward -- a bare ``(conv, ssm)`` tuple at S == 2, one tuple per
    interior boundary at S > 2. The script then takes a PARTIAL reject
    through the wide lookup block, so the copied tail unwinds through the
    snapshot rather than through a trim that does not exist.
    """
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "2")

    class _ArraysCacheLike:
        """Recurrent-layer double: two array slots and a snapshot, nothing else."""

        def __init__(self):
            self.offset = 0
            self.state = ["conv@0", "ssm@0"]
            self.rollback_state = None
            self.restores: list[tuple] = []

        def __getitem__(self, i):
            return self.state[i]

        def __setitem__(self, i, value):
            self.restores.append((i, value))
            self.state[i] = value

    class _SnapshotWritingModel(_CacheAdvancingQwen35Model):
        """Writes the restore points the chunk-split verify patch would."""

        def __call__(self, inputs, cache=None, n_confirmed: int = 0, **kwargs):
            result = super().__call__(
                inputs, cache=cache, n_confirmed=n_confirmed, **kwargs
            )
            span = int(inputs.shape[1])
            if cache and n_confirmed > 0:
                head = cache[0]
                if span == 2:
                    head.rollback_state = (f"conv@{span}", f"ssm@{span}")
                elif span > 2:
                    head.rollback_state = [
                        (f"conv@{span}.{b}", f"ssm@{span}.{b}") for b in range(span - 1)
                    ]
            return result

    def _run(*, declared: bool):
        model = _SnapshotWritingModel(
            # Lookup proposes [20, 21]. Target accepts 20, rejects 21 as 19.
            backbone_outputs=[0, 0, 0, 7, 8, 0, 20, 19, 22],
            mtp_outputs=[0, 0, 0, 77, 0],
        )
        model.mtp_wide_verify_rollback_supported = declared
        model_cache = _ArraysCacheLike()
        timing: dict[str, float] = {}
        emitted = list(
            mtp_generate_step(
                mx.array([7, 8, 20, 21], dtype=mx.uint32),
                model,
                max_tokens=4,
                max_k=1,
                disable_auto_k=True,
                prompt_cache=[model_cache, _CountingKVCache()],
                accept_counter=MTPAcceptCounter(),
                timing_stats=timing,
            )
        )
        return timing, model_cache, emitted

    # The real predicate needs the mlx-level patch installed against a real
    # GatedDeltaNet, so stand in for "the patch is live": the test pins the
    # admission logic, not the patcher.
    monkeypatch.setattr(
        generator_mod, "gated_delta_snapshot_rollback_installed", lambda: True
    )

    timing, cache, emitted = _run(declared=True)
    assert timing.get("prompt_lookup_cache_fallthroughs", 0.0) == 0.0
    assert timing["prompt_lookup_proposals"] == 1
    assert timing["prompt_lookup_accepted_tokens"] == 1
    assert timing["prompt_lookup_rejections"] == 1
    assert [(token, drafted) for token, _lp, drafted in emitted] == [
        (7, False),
        (8, False),
        (20, True),
        (19, False),
    ]
    # Two unwinds, both through the snapshot: the ordinary S=2 draft reject
    # takes the bare tuple, and the S=3 lookup block takes the boundary that
    # keeps exactly the one accepted copied token (``snapshots[keep - 1]``).
    assert cache.restores == [
        (0, "conv@2"),
        (1, "ssm@2"),
        (0, "conv@3.1"),
        (1, "ssm@3.1"),
    ]
    assert cache.rollback_state is None

    # Same model, same caches: without either half of the conjunction the
    # guard refuses the proposal and the generator books a fall-through.
    # (The refused runs book a fall-through on every round whose suffix
    # matches, and there are more such rounds precisely because no copied
    # token is ever emitted -- so pin "at least one, and nothing drafted".)
    rejected, _cache, _emitted = _run(declared=False)
    assert rejected["prompt_lookup_cache_fallthroughs"] >= 1
    assert rejected.get("prompt_lookup_proposals", 0.0) == 0.0

    monkeypatch.setattr(
        generator_mod, "gated_delta_snapshot_rollback_installed", lambda: False
    )
    unpatched, _cache, _emitted = _run(declared=True)
    assert unpatched["prompt_lookup_cache_fallthroughs"] >= 1
    assert unpatched.get("prompt_lookup_proposals", 0.0) == 0.0


def test_generator_fails_closed_when_trim_breaks_its_contract():
    """A short target-cache trim aborts rather than emitting from stale state."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    class ShortTrimCache(_CountingKVCache):
        def trim(self, n):
            self.trim_calls.append(n)
            return 0

    with pytest.raises(RuntimeError, match="violated speculative rollback"):
        list(
            mtp_generate_step(
                mx.array([1], dtype=mx.uint32),
                _CacheAdvancingQwen35Model([7, 12, 99], [11]),
                prompt_cache=[ShortTrimCache(), _CountingKVCache()],
                max_tokens=2,
                max_k=1,
                disable_auto_k=True,
                accept_counter=MTPAcceptCounter(),
            )
        )


def test_prompt_lookup_requires_an_audited_model_capability(monkeypatch):
    """An env opt-in cannot force unaudited MTP backends into prompt lookup."""
    from types import SimpleNamespace

    from rapid_mlx.spec_decode.mtp.generator import _prompt_lookup_is_enabled
    from rapid_mlx.spec_decode.mtp.prompt_lookup import PromptLookupPolicy

    monkeypatch.delenv("RAPID_MLX_MTP_PROMPT_LOOKUP", raising=False)
    assert not _prompt_lookup_is_enabled(
        SimpleNamespace(mtp_prompt_lookup_supported=True)
    )
    qualified = SimpleNamespace(
        mtp_prompt_lookup_supported=True,
        mtp_prompt_lookup_policy=PromptLookupPolicy(enabled_by_default=True),
    )
    assert _prompt_lookup_is_enabled(qualified)
    assert not _prompt_lookup_is_enabled(qualified, requested=False)

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    assert not _prompt_lookup_is_enabled(SimpleNamespace())
    assert not _prompt_lookup_is_enabled(
        SimpleNamespace(mtp_prompt_lookup_supported=False)
    )
    assert _prompt_lookup_is_enabled(SimpleNamespace(mtp_prompt_lookup_supported=True))
    assert _prompt_lookup_is_enabled(qualified)

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "off")
    assert not _prompt_lookup_is_enabled(
        SimpleNamespace(mtp_prompt_lookup_supported=True)
    )
    assert not _prompt_lookup_is_enabled(qualified)


def test_generator_prompt_lookup_partial_reject_keeps_mtp_cache_aligned(
    monkeypatch,
):
    """Only the accepted lookup prefix is appended to MTP history."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "2")

    model = _CacheAdvancingQwen35Model(
        # Lookup proposes [20, 21]. Target accepts 20, rejects 21 as 19.
        backbone_outputs=[0, 0, 0, 7, 8, 0, 20, 19, 22],
        mtp_outputs=[0, 0, 0, 77, 0],
    )
    model_cache = _CountingKVCache()
    mtp_cache = _CountingKVCache()
    timing: dict[str, float] = {}

    emitted = list(
        mtp_generate_step(
            mx.array([7, 8, 20, 21], dtype=mx.uint32),
            model,
            max_tokens=4,
            max_k=1,
            disable_auto_k=True,
            prompt_cache=[model_cache, mtp_cache],
            accept_counter=MTPAcceptCounter(),
            timing_stats=timing,
        )
    )

    assert [(token, drafted) for token, _lp, drafted in emitted] == [
        (7, False),
        (8, False),
        (20, True),
        (19, False),
    ]
    assert timing["prompt_lookup_accepted_tokens"] == 1
    assert timing["prompt_lookup_rejections"] == 1
    # Target drops the ordinary rejected draft and then lookup's unaccepted
    # tail. MTP drops only its ordinary draft: lookup never speculatively
    # advanced that cache, and appends exactly one accepted history position.
    assert model_cache.trim_calls == [1, 1]
    assert mtp_cache.trim_calls == [1]
    assert mtp_cache.offset == 4
    assert model._mtp_cursor == 5


def test_generator_prompt_lookup_rolls_back_on_verify_materialization_abort(
    monkeypatch,
):
    """A cancelled PLD verify drops every uncommitted target position."""
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "2")

    model = _CacheAdvancingQwen35Model(
        backbone_outputs=[0, 0, 0, 7, 8, 0, 20, 21, 22],
        mtp_outputs=[0, 0, 0, 99],
    )
    model_cache = _CountingKVCache()
    mtp_cache = _CountingKVCache()
    gen = mtp_generate_step(
        mx.array([7, 8, 20, 21], dtype=mx.uint32),
        model,
        max_tokens=5,
        max_k=1,
        disable_auto_k=True,
        prompt_cache=[model_cache, mtp_cache],
        accept_counter=MTPAcceptCounter(),
    )

    assert next(gen)[0] == 7
    assert next(gen)[0] == 8
    assert model_cache.trim_calls == [1]
    assert mtp_cache.trim_calls == [1]

    monkeypatch.setattr(
        generator_mod.mx,
        "eval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("sentinel PLD verify abort")
        ),
    )
    with pytest.raises(RuntimeError, match="sentinel PLD verify abort"):
        next(gen)

    assert model_cache.trim_calls == [1, 2]
    # PLD bypasses the MTP drafter, so its uncommitted cache never advanced.
    assert mtp_cache.trim_calls == [1]


def test_generator_runs_with_int4_quantized_kv_cache_kwargs():
    """Smoke: the generator accepts ``kv_bits=4`` / ``kv_group_size=32``
    and runs without crashing.

    The R15 #300 default is ``--kv-cache-dtype int4``, so MTP must
    work on the quantized path. We don't try to verify byte-level
    equivalence between bf16 and int4 outputs here — quantization
    introduces representational noise that may shift argmax in a
    tied vote, and the mocked logits don't produce ties anyway. The
    purpose is: ``mtp_generate_step(prompt, model, kv_bits=4, ...)``
    must complete a generation without raising.
    """
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    backbone = [7, 11, 13]
    mtp = [11]
    model = _MockedQwen35Model(backbone, mtp)
    counter = MTPAcceptCounter()
    prompt = mx.array([1], dtype=mx.uint32)

    emitted = list(
        mtp_generate_step(
            prompt,
            model,
            max_tokens=3,
            accept_counter=counter,
            kv_bits=4,
            kv_group_size=32,
            quantized_kv_start=0,
        )
    )
    assert len(emitted) == 3


def test_generator_runs_with_bf16_default_kv_cache():
    """Smoke: ``kv_bits=None`` (bf16 / unquantized) path also works."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    backbone = [7, 11, 13]
    mtp = [11]
    model = _MockedQwen35Model(backbone, mtp)
    counter = MTPAcceptCounter()
    prompt = mx.array([1], dtype=mx.uint32)

    emitted = list(
        mtp_generate_step(
            prompt,
            model,
            max_tokens=3,
            accept_counter=counter,
            kv_bits=None,  # bf16 path
        )
    )
    assert len(emitted) == 3


def test_generator_records_counter_on_accept_and_reject():
    """Multi-step run: 2 accepts + 1 reject → 3 attempts, 2 accepts.

    Sequence:

      cold-start backbone → 7 (primary)
      MTP → draft=11; verify backbone → 11 (accept), bonus=13
      MTP cache_commit (consumes 2 mtp slots: discard, draft=17)
      verify backbone → 17 (accept), bonus=19
      MTP cache_commit (consumes 2: discard, draft=21)
      verify backbone → 23 (reject), bonus=99 (unused)
    """
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    backbone = [
        7,  # cold-start primary
        11,
        13,  # verify1: pred=11 matches draft1, bonus=13
        17,
        19,  # verify2: pred=17 matches draft2, bonus=19
        23,
        99,  # verify3: pred=23 ≠ draft3=21 (reject), bonus=99 (unused)
    ]
    mtp = [
        11,  # cold-start draft1
        # cache_commit after accept1 consumes 2 mtp positions:
        #   first sentinel (unused) + draft2
        0,
        17,
        # cache_commit after accept2 consumes 2 mtp positions:
        0,
        21,  # draft3 (will be rejected)
        # After reject the next _step_mtp is cold (no cache_commit, S=1)
        99,
    ]
    model = _MockedQwen35Model(backbone, mtp)

    counter = MTPAcceptCounter()
    prompt = mx.array([1], dtype=mx.uint32)
    # 0.9.13 PR-B: disable auto-K so the 3-attempt script this test
    # asserts holds (the controller would otherwise park early and
    # rearrange the round counts).
    list(
        mtp_generate_step(
            prompt,
            model,
            max_tokens=6,
            accept_counter=counter,
            disable_auto_k=True,
        )
    )

    snap = counter.snapshot()
    assert snap.attempts == 3, f"expected 3 attempts; got {snap}"
    assert snap.accepts == 2, f"expected 2 accepts; got {snap}"
    assert snap.tokens_saved == 2
    assert snap.accept_ratio == pytest.approx(2 / 3)


def test_drafter_wall_time_is_charged_to_the_round_that_consumes_it():
    """``cost(K>=1)`` must include the drafting that produced the K drafts.

    Drafting happens at the tail of round N, after round N's timer has
    already been read, so a naive implementation charges it to nobody:
    ``cost(1)`` measures only the target's verify forward. The EV
    comparator then divides ``committed(K) / cost(K)`` by a cost missing
    the one term that most distinguishes K>=1 from K=0, and systematically
    under-prices drafting.

    Driven by a virtual clock that advances only when the model is
    called, so the recorded costs are exact rather than thresholded. A
    real ``time.sleep`` would make this a wall-clock race: a loaded host
    or MLX first-touch initialization can inflate the K=0 forward past
    any absolute cutoff, and the cost EWMA's innovation clamp would keep
    it there for the rest of the test.
    """
    import time as _time

    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        get_or_create_controller,
        reset_controllers,
    )
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    FORWARD_S = 0.010  # every backbone forward
    DRAFT_S = 0.050  # every drafter call, deliberately the larger term

    clock = {"t": 0.0}

    class _ClockedModel(_MockedQwen35Model):
        def __call__(self, *args, **kwargs):
            clock["t"] += FORWARD_S
            return super().__call__(*args, **kwargs)

        def mtp_forward(self, *args, **kwargs):
            clock["t"] += DRAFT_S
            return super().mtp_forward(*args, **kwargs)

    reset_controllers()
    # Long enough for the controller's bootstrap to seed both K=0 and
    # K=1; every draft matches its verify position so drafting is always
    # accepted and K=1 rounds keep being chosen.
    backbone = [7] + [11, 13] * 40
    model = _ClockedModel(backbone, [11] * 80)
    prompt = mx.array([1], dtype=mx.uint32)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_time, "perf_counter", lambda: clock["t"])
        list(
            mtp_generate_step(
                prompt,
                model,
                max_tokens=40,
                accept_counter=MTPAcceptCounter(),
                model_id="slow-drafter",
                max_k=1,
            )
        )

    curve = get_or_create_controller("slow-drafter", max_k=1).cost.samples()
    assert 0 in curve and 1 in curve, f"controller never sampled both depths: {curve}"
    park_ms, _ = curve[0]
    draft_ms, _ = curve[1]

    # A park round is one backbone forward and consumes no drafts.
    assert park_ms == pytest.approx(FORWARD_S * 1000.0), (
        f"K=0 cost {park_ms:.1f}ms != the one forward it ran "
        f"({FORWARD_S * 1000.0:.0f}ms) — drafting is being charged to a "
        "round that consumed no drafts"
    )
    # A K=1 round is one verify forward PLUS the drafting that produced
    # the draft it consumed. Without the carry this is FORWARD_S alone.
    assert draft_ms == pytest.approx((FORWARD_S + DRAFT_S) * 1000.0), (
        f"K=1 cost {draft_ms:.1f}ms != forward + drafter "
        f"({(FORWARD_S + DRAFT_S) * 1000.0:.0f}ms) — the drafter is not "
        "being charged to the round that consumed its output"
    )
    reset_controllers()


def test_fixed_k_mode_still_records_the_cost_curve():
    """``disable_auto_k=True`` must keep OBSERVING even though it stops
    the controller from CHOOSING.

    Fixed-K is the mode an operator measures in — under auto-K the
    acceptance ratio is pooled across every depth the controller sampled,
    so it describes no particular K. If fixed-K dropped the controller
    entirely, ``k_cost_ms`` would be empty in exactly the configuration
    the docs tell operators to measure with.

    Behaviour must not change: K stays pinned at 1, so no round parks.
    """
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        get_or_create_controller,
        reset_controllers,
    )
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    reset_controllers()
    backbone = [7] + [11, 13] * 20
    model = _MockedQwen35Model(backbone, [11] * 40)

    list(
        mtp_generate_step(
            mx.array([1], dtype=mx.uint32),
            model,
            max_tokens=20,
            accept_counter=MTPAcceptCounter(),
            model_id="fixed-k-model",
            max_k=1,
            disable_auto_k=True,
        )
    )

    ctrl = get_or_create_controller("fixed-k-model", max_k=1)
    curve = ctrl.cost.samples()
    assert 1 in curve, f"fixed-K recorded no cost for K=1: {curve}"
    assert curve[1][1] > 0, "K=1 sampled zero times"
    # The controller never CHOSE a depth, so the only K=0 round is the
    # cold-start one every request begins with (no drafts exist yet).
    # That single sample is not an accident to be suppressed — it is the
    # denominator the fixed-K decision rule needs, and there is exactly
    # one per request.
    assert ctrl.park_count == 1, (
        f"expected exactly the one cold-start K=0 round, got "
        f"{ctrl.park_count} — the controller is selecting depth when it "
        "must only observe"
    )
    assert set(ctrl.k_histogram) == {0, 1}, (
        f"fixed-K ran depths {sorted(ctrl.k_histogram)}, expected the "
        "cold-start K=0 plus pinned K=1"
    )
    assert 0 in curve, (
        "no K=0 reference recorded — the fixed-K decision rule has no "
        "denominator to divide by"
    )
    reset_controllers()


def test_derive_controller_key_is_stable_and_discriminating():
    """The fallback registry key must be stable across restarts AND
    distinct between different models.

    Both halves are load-bearing. It reaches ``/metrics`` as the
    ``model_id`` label, so an ``id()``-derived key would mint a fresh
    series every boot; and equal keys share one ``DepthController``, so a
    collision between different models would let one model's learned
    costs drive the other's depth selection.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        derive_controller_key,
    )

    class _Args:
        model_type = "qwen3_5_moe"
        num_hidden_layers = 40
        hidden_size = 4096
        vocab_size = 151936

    class _Model:
        args = _Args()

    a, b = _Model(), _Model()
    key = derive_controller_key(a)

    # Stable: two distinct instances of the same model shape agree, so a
    # restart reproduces the key.
    assert derive_controller_key(b) == key
    # Carries no address.
    assert hex(id(a))[2:] not in key and str(id(a)) not in key
    assert "qwen3_5_moe" in key and "num_hidden_layers=40" in key

    # Discriminating: a different shape must not collide.
    class _OtherArgs(_Args):
        num_hidden_layers = 64

    class _OtherModel:
        args = _OtherArgs()

    assert derive_controller_key(_OtherModel()) != key

    # The drafter's shape is folded in, so the same target with a
    # different-sized head is a different key.
    class _Head:
        layers = [object(), object()]

    class _WithHead(_Model):
        mtp = _Head()

    assert derive_controller_key(_WithHead()) != key

    # Nested dict config: Qwen3.5/3.6's outer args carries only
    # ``model_type`` plus a plain-dict ``text_config``. Reading just the
    # top level as an object yields "model_type=qwen3_5_moe" and nothing
    # else — every model of the family would then share one controller.
    class _NestedArgs:
        model_type = "qwen3_5_moe"
        text_config = {"num_hidden_layers": 40, "hidden_size": 2048}

    class _NestedModel:
        args = _NestedArgs()

    nested_key = derive_controller_key(_NestedModel())
    assert "num_hidden_layers=40" in nested_key, (
        f"nested dict config did not resolve: {nested_key}"
    )
    assert "hidden_size=2048" in nested_key

    class _NestedArgsWider(_NestedArgs):
        text_config = {"num_hidden_layers": 40, "hidden_size": 5120}

    class _WiderModel:
        args = _NestedArgsWider()

    assert derive_controller_key(_WiderModel()) != nested_key

    # Quantization is not shape, but two quantizations of one
    # architecture differ substantially in what a forward costs — sharing
    # a cost curve between them would be a real mis-attribution.
    class _Q4:
        model_type = "qwen3_5"
        num_hidden_layers = 64
        quantization = {"bits": 4, "group_size": 64}

    class _Q8(_Q4):
        quantization = {"bits": 8, "group_size": 64}

    # One class, swapped args — the key opens with the model's class name,
    # so two differently-named test classes would differ for that reason
    # rather than for the one under test.
    class _Quantized:
        def __init__(self, args):
            self.args = args

    assert derive_controller_key(_Quantized(_Q4())) != derive_controller_key(
        _Quantized(_Q8())
    )

    # Rendered in a fixed order so dict iteration cannot perturb the key.
    class _Q4Reordered(_Q4):
        quantization = {"group_size": 64, "bits": 4}

    assert derive_controller_key(_Quantized(_Q4Reordered())) == derive_controller_key(
        _Quantized(_Q4())
    )

    # Degrades without raising when the model exposes no config at all.
    class _Bare:
        pass

    assert derive_controller_key(_Bare())


def test_derive_controller_key_reads_quantization_off_the_modules():
    """Quantization must come from the instantiated layers, not config.

    mlx-lm's ``TextModelArgs.from_dict()`` drops the checkpoint's
    ``quantization`` block, so a config-only lookup finds nothing on
    exactly the quantized models this needs to tell apart — leaving a
    4-bit and an 8-bit build of one architecture sharing a cost curve.
    The quantized modules themselves always carry it.
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        derive_controller_key,
    )

    class _Layer:
        def __init__(self, bits, group_size):
            self.bits = bits
            self.group_size = group_size

    class _Model:
        """No config surface at all — mirrors the post-``from_dict`` state."""

        def __init__(self, layers):
            self._layers = layers

        def named_modules(self):
            return [(str(i), m) for i, m in enumerate(self._layers)]

    k4 = derive_controller_key(_Model([_Layer(4, 64), _Layer(4, 64)]))
    k8 = derive_controller_key(_Model([_Layer(8, 64), _Layer(8, 64)]))
    assert "quant=4b/g64" in k4
    assert "quant=8b/g64" in k8
    assert k4 != k8

    # A mixed-precision build is described, not reduced to whichever
    # layer happened to be visited first — and the order it is visited
    # in must not change the key.
    mixed_a = derive_controller_key(_Model([_Layer(4, 64), _Layer(8, 64)]))
    mixed_b = derive_controller_key(_Model([_Layer(8, 64), _Layer(4, 64)]))
    assert mixed_a == mixed_b
    assert "quant=4b/g64+8b/g64" in mixed_a
    assert mixed_a != k4

    # Unquantized: no marker rather than a misleading one.
    class _Plain:
        def named_modules(self):
            return [("0", object())]

    assert "quant=" not in derive_controller_key(_Plain())


def test_resolve_model_identity_prefers_engine_checkpoint_over_stale_config():
    """Unit test of the ``_resolve_model_identity`` HELPER only — the engine's
    own loaded checkpoint wins over a (possibly stale, shared) configured name,
    and the resolution is idempotent so a reused config object stays safe. The
    ACTUAL ``EngineCore.__init__`` wiring (copy + assign) is covered separately
    by ``test_engine_core_init_wires_resolved_model_name_onto_a_scheduler_copy``,
    which constructs the engine; this test deliberately does not, so keep the
    two distinct (codex #1441).
    """
    from rapid_mlx.scheduler import SchedulerConfig

    assert "model_name" in {f.name for f in dataclasses.fields(SchedulerConfig)}, (
        "SchedulerConfig lost the model_name field the controller key "
        "resolution chain depends on"
    )
    assert SchedulerConfig().model_name is None

    from rapid_mlx.engine_core import _resolve_model_identity

    engine_a = types.SimpleNamespace(model_name="engine/a", model_path=None)
    engine_b = types.SimpleNamespace(model_name="engine/b", model_path=None)

    # A blank config takes the engine's checkpoint.
    cfg = SchedulerConfig()
    cfg.model_name = _resolve_model_identity(engine_a, cfg.model_name)
    assert cfg.model_name == "engine/a"

    # THE RELOAD CASE: the same config object reused by a second engine
    # must pick up the SECOND checkpoint. A fill-a-blank guard would
    # leave "engine/a" here and hand the new model the old model's
    # process-global controller.
    cfg.model_name = _resolve_model_identity(engine_b, cfg.model_name)
    assert cfg.model_name == "engine/b"

    # And the engine must not write inference back into a config it does
    # not own: a shared SchedulerConfig stays pristine, so a second,
    # UNNAMED engine cannot mistake the first engine's inferred name for
    # its own explicit configuration.
    shared = SchedulerConfig()
    for engine in (engine_a, engine_b):
        local = copy.copy(shared)
        local.model_name = _resolve_model_identity(engine, local.model_name)
        assert local.model_name == engine.model_name
    assert shared.model_name is None
    anonymous = types.SimpleNamespace(model_name=None, model_path=None)
    local = copy.copy(shared)
    local.model_name = _resolve_model_identity(anonymous, local.model_name)
    assert local.model_name is None, (
        "an unnamed engine inherited a previous engine's identity and "
        "would be keyed as that model"
    )

    # Idempotent: re-resolving with the same engine does not drift.
    assert _resolve_model_identity(engine_b, cfg.model_name) == "engine/b"

    # An engine with no identity of its own falls back to the config.
    anon = types.SimpleNamespace(model_name=None, model_path=None)
    assert _resolve_model_identity(anon, "configured/name") == "configured/name"
    assert _resolve_model_identity(anon, None) is None

    # ``model_path`` serves when ``model_name`` is absent.
    pathy = types.SimpleNamespace(model_name=None, model_path="/models/foo")
    assert _resolve_model_identity(pathy, None) == "/models/foo"


def test_engine_core_init_wires_resolved_model_name_onto_a_scheduler_copy(
    monkeypatch,
):
    """Guards the ACTUAL ``EngineCore.__init__`` wiring, not just the helper:
    __init__ must ``copy.copy`` the SchedulerConfig and stamp the resolved
    model identity onto the COPY before handing it to the Scheduler. Deleting
    the copy+resolve lines would leave the captured config's ``model_name``
    None (or mutate the caller's object); both are caught here and neither is
    caught by the ``_resolve_model_identity``-only test above (codex #1441).
    """
    from rapid_mlx import engine_core as ec

    captured: dict = {}

    class _StopInitError(Exception):
        pass

    def _capture_scheduler(**kwargs):
        captured["config"] = kwargs.get("config")
        raise _StopInitError

    class _FakeRegistry:
        def acquire(self, **kw):
            return None

        def release(self, *a, **kw):
            return None

    monkeypatch.setattr(ec, "Scheduler", _capture_scheduler)
    monkeypatch.setattr(ec, "get_registry", lambda: _FakeRegistry())

    # Caller's pristine, shared SchedulerConfig (model_name unset).
    shared = ec.SchedulerConfig()
    cfg = ec.EngineConfig(model_name="engine/x", scheduler_config=shared)

    try:
        ec.EngineCore(model=object(), tokenizer=object(), config=cfg)
    except _StopInitError:
        pass

    sc = captured.get("config")
    assert sc is not None, "EngineCore.__init__ never constructed the Scheduler"
    # The engine's checkpoint identity reached the scheduler config...
    assert sc.model_name == "engine/x"
    # ...on a COPY: the caller's shared config stays pristine, so a second
    # unnamed engine cannot inherit this name as its own explicit config.
    assert sc is not shared
    assert shared.model_name is None


def test_mtp_controller_key_separates_sidecars():
    """The controller learns an ACCEPTANCE profile, and acceptance is a
    property of the target/drafter pair, not the target alone.

    The registry is process-global and never reset in production, so a
    key ignoring the sidecar would let the first head's profile drive
    depth selection for a different head after a reload.
    """
    from rapid_mlx.scheduler import _mtp_controller_key

    base = _mtp_controller_key("qwen3.6-35b", None)
    a = _mtp_controller_key("qwen3.6-35b", "mlx-community/Head-A")
    b = _mtp_controller_key("qwen3.6-35b", "mlx-community/Head-B")

    assert "qwen3.6-35b" in base
    assert a != b
    assert a != base and b != base
    assert "qwen3.6-35b" in a and "Head-A" in a

    # A different target with the same head is still distinct.
    assert _mtp_controller_key("qwen3.6-27b", "mlx-community/Head-A") != a

    # Injectivity regression (codex #1441): a target that literally contains
    # the old ``+mtp:`` join must NOT collide with a different (target, sidecar)
    # pair. Pre-fix, both of these rendered ``"a+mtp:b"``.
    assert _mtp_controller_key("a+mtp:b", None) != _mtp_controller_key("a", "b")

    # No target name -> None, so the caller falls through to the
    # shape-derived key rather than keying on a bare sidecar path.
    assert _mtp_controller_key(None, "mlx-community/Head-A") is None
    assert _mtp_controller_key("", "mlx-community/Head-A") is None


def test_scheduler_config_preserves_the_historical_positional_prefix():
    """New fields append after, rather than shifting, the historical tail."""
    from rapid_mlx.scheduler import SchedulerConfig

    names = [f.name for f in dataclasses.fields(SchedulerConfig)]
    historical_prefix = [
        "max_num_seqs",
        "prefill_batch_size",
        "completion_batch_size",
        "prefill_step_size",
        "enable_prefix_cache",
        "prefix_cache_size",
        "prefix_cache_index",
        "use_memory_aware_cache",
        "cache_memory_mb",
        "cache_memory_percent",
        "kv_cache_dtype",
        "kv_cache_quantization",
        "kv_cache_quantization_bits",
        "kv_cache_quantization_group_size",
        "kv_cache_min_quantize_tokens",
        "kv_cache_turboquant",
        "kv_cache_turboquant_bits",
        "kv_cache_turboquant_group_size",
        "kv_cache_turboquant_mode",
        "kv_disk_checkpoint_interval",
        "use_paged_cache",
        "paged_cache_block_size",
        "max_cache_blocks",
        "hybrid_cache_entries",
        "spec_decode",
        "enable_mtp",
        "mtp_num_draft_tokens",
        "mtp_optimistic",
        "dflash_drafter_path",
        "enable_suffix_decoding",
        "suffix_max_draft",
        "suffix_max_suffix_len",
        "suffix_min_confidence",
        "suffix_min_draft_len",
        "max_concurrent_requests",
        "gpu_memory_utilization",
        "metal_pressure_evict_fraction",
        "metal_cap_kv_bytes_per_token",
        "pflash_config",
        "mtp_sidecar",
        "mtp_model_type",
        "mtp_max_k",
        "mtp_disable_auto_k",
        "response_cache_entries",
        "non_trimmable_exact_prefix_reuse",
        "dspark_num_speculative_tokens",
        "adaptive_prefill",
        "adaptive_prefill_min_tokens",
        "adaptive_prefill_min_chunk_size",
        "model_name",
    ]
    assert names[:50] == historical_prefix
    # ``model_name`` (index 49) and the two vision fields hold their historical
    # positions; anything the future appends lands at index 52+ rather than
    # shifting them. A slice equality here (not the whole tail) is what lets a
    # genuinely-appended field — e.g. ``idle_cache_clear_seconds`` (#2038) —
    # pass while a field INSERTED into the prefix still fails.
    assert names[49:52] == ["model_name", "vision_min_pixels", "vision_max_pixels"]


def test_fixed_k_observer_leaves_the_ceiling_to_whoever_selects_depth():
    """An observer-only fixed-K run must not fix ``max_k`` for later runs.

    The registry normally keeps whatever ceiling the FIRST caller set.
    Fixed mode now exercises the configured K (including SSM-safe K=3),
    but its ceiling remains provisional: the first selecting auto-K caller
    may still replace it with its own configured ceiling.
    """
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        get_or_create_controller,
        reset_controllers,
    )
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    def _observe_only(max_k):
        """Run a fixed-K generation, which records but never selects."""
        reset_controllers()
        list(
            mtp_generate_step(
                mx.array([1], dtype=mx.uint32),
                _MockedQwen35Model([7] + [11, 13] * 20, [11] * 40),
                max_tokens=12,
                accept_counter=MTPAcceptCounter(),
                model_id="ceiling-model",
                max_k=max_k,
                disable_auto_k=True,
            )
        )
        # Peek without adopting — an authoritative read would itself set
        # the ceiling this test is trying to observe.
        return get_or_create_controller("ceiling-model", max_k=99, authoritative=False)

    # Fixed mode really ran the configured K=3, but says so provisionally.
    ctrl = _observe_only(max_k=3)
    assert ctrl.max_k == 3
    assert ctrl.ceiling_is_authoritative is False
    assert ctrl.pick_k() <= 3

    # A later auto-K run clamped to 1 (the SSM case) keeps 1.
    ctrl = _observe_only(max_k=3)
    assert get_or_create_controller("ceiling-model", max_k=1).max_k == 1

    # A later auto-K run configured for 3 gets 3 — the observer must not
    # have capped it.
    ctrl = _observe_only(max_k=3)
    adopted = get_or_create_controller("ceiling-model", max_k=3)
    assert adopted.max_k == 3
    assert adopted.ceiling_is_authoritative is True

    # Once authoritative, the pre-existing "first config wins" rule is
    # unchanged: a conflicting later ceiling is ignored.
    assert get_or_create_controller("ceiling-model", max_k=2).max_k == 3
    reset_controllers()


def test_promoted_ceiling_records_and_selects_deeper_depths_lazily():
    """Promotion lifts only ``max_k``; the depth-indexed cost/acceptance
    state grows lazily, so a controller promoted 1->3 can record and select
    depths 2/3 afterwards without out-of-range access, and keeps the
    observations made while provisional (codex #1441 r1: the promotion must
    not leave K=1-sized state that a later K=3 run overruns).
    """
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        ACCEPTANCE_MIN_SAMPLES,
        get_or_create_controller,
        reset_controllers,
    )

    reset_controllers()
    # Provisional observer ceiling at 1; teach it depths 0 and 1 only.
    ctrl = get_or_create_controller("m", max_k=1, authoritative=False)
    for _ in range(ACCEPTANCE_MIN_SAMPLES + 1):
        ctrl.cost.observe(0, 10.0)
        ctrl.cost.observe(1, 16.0)
        ctrl.acc.observe(1, accepted=True)
    assert ctrl.max_k == 1
    depth1_cost_before = ctrl.cost.cost(1)

    # First authoritative caller promotes the ceiling to 3.
    ctrl = get_or_create_controller("m", max_k=3, authoritative=True)
    assert ctrl.max_k == 3
    # Observations made while provisional survive (depth-keyed, not
    # ceiling-keyed): the depth-1 cost EWMA is still there.
    assert ctrl.cost.sampled(1)
    assert ctrl.cost.cost(1) == depth1_cost_before

    # Now record the depths a K=3 run would visit — must not raise and must
    # actually fold in (the lists grow via ``while len <= i: append``).
    for _ in range(ACCEPTANCE_MIN_SAMPLES + 1):
        ctrl.cost.observe(2, 22.0)
        ctrl.cost.observe(3, 28.0)
        ctrl.acc.observe(2, accepted=True)
        ctrl.acc.observe(3, accepted=True)
    assert ctrl.cost.sampled(3)
    assert ctrl.cost.visits(3) == ACCEPTANCE_MIN_SAMPLES + 1
    # Selection stays within the promoted ceiling, never past it.
    assert 0 <= ctrl.pick_k() <= 3
    reset_controllers()


# ---------------------------------------------------------------------------
# CopyDraftGate -- is this turn's copy-draft worth its verify block?
# ---------------------------------------------------------------------------


def _feed(gate, *, copy_rounds=(), base_rounds=()):
    """Fold ``(committed, round_ms)`` rounds into the gate's two series.

    A copy round that committed ``N`` tokens accepted ``N - 1`` copied rows
    and then the bonus, so the sizing counts these fixtures report are
    ``accepted = proposed = N - 1``: a block the target consumed end to end,
    which is the shape a long match produces. A degenerate ``N = 0`` round --
    one of the cases the gate has to ignore -- becomes a one-row block that
    missed, which is the narrowest well-formed pair. The throughput tests
    below read only the windowed rates, but the counts still have to be the
    ones a real round of that size would carry.
    """
    for committed, ms in base_rounds:
        gate.observe(is_copy_draft=False, committed=committed, round_ms=ms)
    for committed, ms in copy_rounds:
        accepted = max(0, committed - 1)
        gate.observe(
            is_copy_draft=True,
            committed=committed,
            round_ms=ms,
            accepted=accepted,
            proposed=max(1, accepted),
        )


def test_copy_draft_gate_stays_open_until_both_series_are_sampled():
    """An unproven gate never refuses: the only way to price a copy-draft is
    to verify one, so a turn that has not yet run three of each kind is given
    the benefit of the doubt."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_GATE_MIN_SAMPLES,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    assert gate.allow() is True

    # Terrible copy-drafts, but only two of them, and no baseline at all.
    _feed(gate, copy_rounds=[(1, 400.0)] * (COPY_DRAFT_GATE_MIN_SAMPLES - 1))
    assert gate.allow() is True

    # Baseline sampled, copy series still one short.
    _feed(gate, base_rounds=[(2, 50.0)] * COPY_DRAFT_GATE_MIN_SAMPLES)
    assert gate.allow() is True
    assert gate.declines == 0


def test_copy_draft_gate_refuses_copy_drafts_that_lose_on_throughput():
    """The Qwen3.6-27B shape: a copy-draft accepts ~4 rows and still pays for
    the whole 11-row verify, so it commits fewer tokens per millisecond than
    the K=1 rounds it replaced."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    gate = CopyDraftGate()
    # 5 committed per 117 ms copy-draft round against 1.95 per 37 ms MTP round.
    _feed(gate, copy_rounds=[(5, 117.0)] * 3, base_rounds=[(2, 37.0)] * 3)
    copy_rate, base_rate = gate.rates()
    assert copy_rate < base_rate

    assert gate.allow() is False
    assert gate.declines == 1


def test_copy_draft_gate_admits_copy_drafts_that_win_on_throughput():
    """The Qwen3.8-27B shape: ~10 committed rows on the ``quantized_matmul``
    plateau beat 1.9 committed rows at K=1, so the gate stays out of the way."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    gate = CopyDraftGate()
    _feed(gate, copy_rounds=[(10, 342.0)] * 3, base_rounds=[(2, 92.0)] * 3)
    copy_rate, base_rate = gate.rates()
    assert copy_rate > base_rate

    assert gate.allow() is True
    assert gate.declines == 0
    assert gate.probes == 0


def test_copy_draft_gate_is_not_flipped_by_one_missed_copy_draft():
    """A copy-draft whose first row is wrong commits one token for a full wide
    verify. That is a real outcome and the comparison should feel it -- but it
    is one round among many, and smoothing the throughput series with the
    acceptance model's alpha handed that single round the power to refuse the
    next several. Measured on a Qwen3.8-27B-4bit rename turn that copy-drafts
    win by more than 20%, an alpha of 0.3 refused 52 proposals.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_ACCEPTANCE_EWMA_ALPHA,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    # A turn copy-drafts are winning by the narrow margin production runs at:
    # 0.025 committed tokens per ms against 0.020.
    _feed(gate, copy_rounds=[(25, 1000.0)] * 5, base_rounds=[(2, 100.0)] * 5)
    assert gate.allow() is True

    # A wide block whose first row missed: one committed token for a verify
    # the size of the ones above it.
    gate.observe(
        is_copy_draft=True, committed=1, round_ms=1000.0, accepted=0, proposed=24
    )

    copy_rate, base_rate = gate.rates()
    assert copy_rate == pytest.approx(126 / 6000)
    assert copy_rate > base_rate
    assert gate.allow() is True
    assert gate.declines == 0

    # What the alpha this replaces would have made of the same sample. The
    # counterfactual is computed rather than asserted as a constant, so
    # re-smoothing ``rates`` in a later refactor cannot quietly pass here.
    smoothed = 25 / 1000.0
    smoothed += COPY_DRAFT_ACCEPTANCE_EWMA_ALPHA * (1 / 1000.0 - smoothed)
    assert smoothed < base_rate


def test_copy_draft_gate_probes_a_standing_refusal_with_exponential_backoff():
    """A refusal is a measurement, and measurements go stale: a turn that
    stops quoting the prompt and starts again has to be able to win the gate
    back. So a standing refusal is re-tested on a doubling cadence rather
    than being permanent."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_GATE_PROBE_INTERVAL,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    _feed(gate, copy_rounds=[(2, 300.0)] * 3, base_rounds=[(2, 40.0)] * 3)

    base = COPY_DRAFT_GATE_PROBE_INTERVAL
    verdicts = [gate.allow() for _ in range(base)]
    assert verdicts == [False] * (base - 1) + [True]
    assert gate.probes == 1

    # The cadence doubled, so the next probe is twice as far away.
    verdicts = [gate.allow() for _ in range(2 * base)]
    assert verdicts == [False] * (2 * base - 1) + [True]
    assert gate.probes == 2
    assert gate.declines == (base - 1) + (2 * base - 1)


def test_copy_draft_gate_resets_its_cadence_when_copy_drafts_win_again():
    """Winning on merit clears the backoff, so a later downturn is re-tested
    at the base interval instead of inheriting a long-backed-off cadence."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_GATE_PROBE_INTERVAL,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    _feed(gate, copy_rounds=[(2, 300.0)] * 3, base_rounds=[(2, 40.0)] * 3)
    for _ in range(2 * COPY_DRAFT_GATE_PROBE_INTERVAL):
        gate.allow()
    assert gate.probes >= 1

    # A run of good copy-draft rounds flips the comparison back.
    _feed(gate, copy_rounds=[(20, 300.0)] * 8)
    assert gate.allow() is True

    _feed(gate, copy_rounds=[(1, 300.0)] * 12)
    verdicts = [gate.allow() for _ in range(COPY_DRAFT_GATE_PROBE_INTERVAL)]
    assert verdicts[-1] is True
    assert verdicts[:-1] == [False] * (COPY_DRAFT_GATE_PROBE_INTERVAL - 1)


def test_copy_draft_gate_ignores_rounds_that_committed_nothing():
    """A round that delivered no token, or whose clock read non-positive, is
    not a throughput sample -- folding it in would divide by a number the
    round never measured."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    gate = CopyDraftGate()
    _feed(
        gate,
        copy_rounds=[(0, 300.0), (5, 0.0), (5, -1.0)],
        base_rounds=[(0, 40.0), (2, 0.0)],
    )
    assert gate.rates() == (0.0, 0.0)
    # Still undersampled, therefore still open.
    assert gate.allow() is True


def test_generator_prices_both_round_kinds_into_the_copy_draft_gate(monkeypatch):
    """The gate is fed from both round paths, in delivered tokens.

    A comparison is only as good as its two inputs: a copy-draft round is
    charged its verify wall time, and the rounds it displaces are charged
    their forward plus the drafter cost carried into them. Pin that both
    arrive, and that ``committed`` is the count the caller actually received.
    """
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "2")
    monkeypatch.setattr(
        generator_mod, "_safe_prompt_lookup_draft_count", lambda _c, n, **_kw: n
    )

    seen: list[tuple[bool, int]] = []
    original = CopyDraftGate.observe

    def _spy(self, *, is_copy_draft, committed, round_ms, **kwargs):
        seen.append((is_copy_draft, committed))
        assert round_ms > 0.0
        return original(
            self,
            is_copy_draft=is_copy_draft,
            committed=committed,
            round_ms=round_ms,
            **kwargs,
        )

    monkeypatch.setattr(CopyDraftGate, "observe", _spy)

    emitted = list(
        mtp_generate_step(
            mx.array([7, 8, 20, 21], dtype=mx.uint32),
            _CacheAdvancingQwen35Model(
                backbone_outputs=[0, 0, 0, 7, 8, 20, 21, 22],
                mtp_outputs=[0, 0, 0, 99, 20, 21],
            ),
            max_tokens=3,
            max_k=1,
            disable_auto_k=True,
            prompt_cache=[_CountingKVCache(), _CountingKVCache()],
            accept_counter=MTPAcceptCounter(),
            timing_stats={},
        )
    )

    assert seen, "the gate was never priced"
    assert all(committed >= 1 for _kind, committed in seen)
    # Every delivered token is accounted for by exactly one priced round.
    assert sum(committed for _kind, committed in seen) == len(emitted)


def test_generator_drops_copy_drafts_the_gate_refuses(monkeypatch):
    """A refused proposal is dropped, not narrowed, and is booked separately
    from a cache fall-through so an operator can tell "this turn's copies did
    not pay" from "this target cannot unwind a copy"."""
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "2")
    monkeypatch.setattr(
        generator_mod, "_safe_prompt_lookup_draft_count", lambda _c, n, **_kw: n
    )
    monkeypatch.setattr(CopyDraftGate, "allow", lambda _self: False)

    timing: dict[str, float] = {}
    list(
        mtp_generate_step(
            mx.array([7, 8, 20, 21], dtype=mx.uint32),
            _CacheAdvancingQwen35Model(
                backbone_outputs=[0, 0, 0, 7, 8, 20, 21, 22],
                mtp_outputs=[0, 0, 0, 99, 20, 21],
            ),
            max_tokens=3,
            max_k=1,
            disable_auto_k=True,
            prompt_cache=[_CountingKVCache(), _CountingKVCache()],
            accept_counter=MTPAcceptCounter(),
            timing_stats=timing,
        )
    )

    assert timing["prompt_lookup_ev_declines"] >= 1
    assert timing.get("prompt_lookup_proposals", 0.0) == 0.0
    assert timing.get("prompt_lookup_cache_fallthroughs", 0.0) == 0.0


def test_copy_draft_observe_rejects_counts_that_cannot_describe_a_round():
    """The accepted rows are a prefix of the proposed block.

    Both numbers steer the next block's width now, so a pair that cannot have
    happened is not a harmless log line: ``accepted > proposed`` reads as a
    block the target consumed end to end and doubles the width off a
    measurement that never occurred, and a negative acceptance drags the EWMA
    below the floor the sizer assumes. The generator's own values are ordered
    by construction -- the accept loop runs over the proposed rows and the EOS
    cut only shortens it -- so a violation is a bug upstream, and the state it
    would corrupt is worth more than the round it would cost.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_PROBE_WIDTH,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    for accepted, proposed in ((3, 2), (-1, 4), (0, -2)):
        with pytest.raises(ValueError, match=r"prefix of the proposed block"):
            gate.observe(
                is_copy_draft=True,
                committed=3,
                round_ms=90.0,
                accepted=accepted,
                proposed=proposed,
            )

    # Nothing was folded: the sizer is still at the probe width.
    assert gate.width_cap(31) == COPY_DRAFT_PROBE_WIDTH
    # The boundary itself is a real round -- a block the match filled exactly.
    gate.observe(is_copy_draft=True, committed=3, round_ms=90.0, accepted=2, proposed=2)


def test_copy_draft_gate_forgets_a_verdict_older_than_its_window():
    """A copy-rich opening must not pay for a copy-poor tail.

    One answer can change register halfway through -- quote the prompt's code
    while it edits, then write prose about it -- and the second half must be
    able to refuse on its own evidence. A total over the whole turn cannot do
    that: the counterfactual is computed in the test, and at these rates a
    pooled accumulator would still be admitting proposals here.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_GATE_WINDOW,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    _feed(gate, base_rounds=[(2, 100.0)] * 5)
    # 200 rounds of the shape a rename turn runs at: a wide block the match
    # fills, 15 tokens for 400 ms against the baseline's 2 for 100.
    for _ in range(200):
        gate.observe(
            is_copy_draft=True, committed=15, round_ms=400.0, accepted=14, proposed=14
        )
    assert gate.allow() is True

    # The text stops quoting: the 8-gram still hits, but the continuation
    # diverges on the first row, so the block commits one token.
    losses = 0
    while gate.allow():
        gate.observe(
            is_copy_draft=True, committed=1, round_ms=342.0, accepted=0, proposed=2
        )
        losses += 1
        # A literal ceiling, independent of the constant under test, so an
        # unbounded aggregate fails the assertion below with a number
        # instead of looping until the pool happens to turn over.
        if losses > 256:
            break
    assert losses <= COPY_DRAFT_GATE_WINDOW, (
        f"still admitting proposals after {losses} losing rounds, which is "
        f"longer than the {COPY_DRAFT_GATE_WINDOW}-round window that is "
        "supposed to bound how long a stale verdict stands"
    )
    assert gate.declines >= 1

    # What a total over the turn would have made of the same rounds: still
    # comfortably above the baseline, so it would still be admitting. It
    # takes ~240 losing rounds to drag this pool down to 0.02 -- over a
    # minute of degraded generation.
    _copy_rate, base_rate = gate.rates()
    pooled = (15 * 200 + losses) / (400.0 * 200 + 342.0 * losses)
    assert pooled > base_rate, (
        f"a total over the whole turn would be at {pooled:.5f} tokens/ms "
        f"against a baseline of {base_rate:.5f} after {losses} losing rounds, "
        "so the refusal above has to come from the window rather than from "
        "the aggregate having turned over on its own"
    )


def test_copy_draft_gate_window_holds_exactly_its_last_rounds():
    """The comparison's state is bounded, and bounded at the declared length.

    A gate lives per request, so unbounded per-round history is a leak in
    proportion to concurrency as well as a verdict that cannot be revised.
    Asserted through the rate rather than through the container: once a full
    window of losing rounds has gone in, nothing older can still be in the
    average, so the rate is exactly one losing round's rate.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_GATE_WINDOW,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    for _ in range(500):
        gate.observe(
            is_copy_draft=True, committed=15, round_ms=400.0, accepted=14, proposed=14
        )
    assert gate.rates()[0] == pytest.approx(15 / 400.0)

    for _ in range(COPY_DRAFT_GATE_WINDOW):
        gate.observe(
            is_copy_draft=True, committed=1, round_ms=342.0, accepted=0, proposed=2
        )
    assert gate.rates()[0] == pytest.approx(1 / 342.0)


def test_copy_draft_observe_rejects_a_copy_round_without_its_sizing_counts():
    """Zero is a measurement, so it cannot double as a missing one.

    ``accepted=0`` is a block whose first row missed -- the sizer folds it in
    and narrows. If the argument defaulted to zero instead, a caller that
    reported a copy-draft round the old three-argument way would feed the
    sizer a total rejection every round and pin every later block at
    ``COPY_DRAFT_MIN_WIDTH`` after ``COPY_DRAFT_GATE_MIN_SAMPLES`` of them,
    losing throughput while every test stayed green. The counts are
    therefore required for a copy-draft round and absent by default.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_PROBE_WIDTH,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    with pytest.raises(TypeError, match=r"accepted="):
        gate.observe(is_copy_draft=True, committed=3, round_ms=90.0)
    with pytest.raises(TypeError, match=r"proposed="):
        gate.observe(is_copy_draft=True, committed=3, round_ms=90.0, accepted=2)
    with pytest.raises(TypeError, match=r"accepted="):
        gate.observe(is_copy_draft=True, committed=3, round_ms=90.0, proposed=2)

    # A baseline round has no copied rows to report, so it stays the
    # three-argument call the generator makes from two places.
    gate.observe(is_copy_draft=False, committed=1, round_ms=90.0)

    # And a rejected call folded nothing: the sizer is still unmeasured.
    assert gate.width_cap(31) == COPY_DRAFT_PROBE_WIDTH


def test_copy_draft_width_cap_probes_narrow_while_unmeasured():
    """Nothing is known about how far the copies run until some have run, and
    below the tile edge the rows are not cheap -- two rows cost a rounding
    error over the round they replace where eight already spend three
    quarters of a full-width block -- so the first blocks probe at two."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_GATE_MIN_SAMPLES,
        COPY_DRAFT_PROBE_WIDTH,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    assert gate.width_cap(31) == COPY_DRAFT_PROBE_WIDTH

    for _ in range(COPY_DRAFT_GATE_MIN_SAMPLES - 1):
        gate.observe(
            is_copy_draft=True, committed=3, round_ms=300.0, accepted=2, proposed=2
        )
    assert gate.width_cap(31) == COPY_DRAFT_PROBE_WIDTH
    # A ceiling below the probe width still wins: the sizer only narrows.
    assert gate.width_cap(1) == 1


def test_copy_draft_width_cap_narrows_to_measured_acceptance():
    """The Qwen3.6-27B shape: copies that accept ~4 rows get a block sized
    for ~4 rows, not the 31-row block the ladder would hand a long match."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    gate = CopyDraftGate()
    for _ in range(6):
        gate.observe(
            is_copy_draft=True, committed=5, round_ms=117.0, accepted=4, proposed=7
        )

    # 4 accepted rows earn a 7-row block: 1.5x headroom plus the row that
    # discovers the match ran further than the last ones did. The estimate is
    # asserted through the width it produces rather than through an accessor
    # of its own -- the width is what the generator consumes.
    assert gate.width_cap(31) == 7
    # The ladder keeps the ceiling -- measurement only ever narrows.
    assert gate.width_cap(4) == 4


def test_copy_draft_width_cap_stops_narrowing_inside_the_free_band():
    """The Qwen3.8-27B shape: once the earned width reaches the band where
    ``quantized_matmul`` has switched to the GEMM path, narrowing buys nothing
    and can only truncate a match, so the whole ceiling is handed back.

    This is the step that makes the ratchet terminate at the tile edge rather
    than at 1.5x whatever the last blocks happened to accept -- which matters
    because acceptance is censored by the width proposed, so a cap that
    tracked it forever would never discover a longer match.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    # ``proposed`` above ``accepted`` on every round: these are blocks the
    # target cut short, which is the branch that reads the acceptance EWMA.
    # Passing it explicitly keeps the test off the doubling path rather than
    # relying on the default to keep it off.
    gate = CopyDraftGate()
    for _ in range(6):
        gate.observe(
            is_copy_draft=True,
            committed=15,
            round_ms=342.0,
            accepted=14,
            proposed=20,
        )

    assert gate.width_cap(31) == 31
    assert gate.width_cap(16) == 16

    # Just below the band it still narrows: 8 accepted rows earn 13, which is
    # the floor itself, so the ceiling comes back; 7 earn 11, which does not.
    edge = CopyDraftGate()
    for _ in range(6):
        edge.observe(
            is_copy_draft=True,
            committed=8,
            round_ms=300.0,
            accepted=7,
            proposed=10,
        )
    assert edge.width_cap(31) == 11


def test_copy_draft_width_doubles_while_the_block_comes_back_full():
    """A block the target consumed end to end has not measured the match, it
    has only proved the match runs at least that far -- so the estimate is a
    floor and the only way to lift it is to widen. Doubling reaches the tile
    edge in four steps off a two-row probe where 1.5x headroom would take
    fifteen, and every step is paid for by the rows the step before delivered.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    gate = CopyDraftGate()
    for _ in range(5):
        gate.observe(is_copy_draft=False, committed=2, round_ms=90.0)

    widths = []
    for _ in range(6):
        width = gate.width_cap(31)
        widths.append(width)
        # The match never runs out: every proposed row is accepted.
        gate.observe(
            is_copy_draft=True,
            committed=width + 1,
            round_ms=70.0 + 9.0 * width,
            accepted=width,
            proposed=width,
        )
    assert widths == [2, 2, 2, 4, 8, 31]


def test_copy_draft_width_climbs_one_tile_at_a_time():
    """An operator ceiling above the first tile edge is reached in steps.

    ``quantized_matmul`` charges a whole 32-row tile once one row lands in
    it, so a block that ends just past an edge pays for rows nothing reaches:
    at a ceiling of 64 the widths worth proposing are 31 and 63, and the jump
    between them has to be earned the same way the first one was -- by a
    block the target consumed end to end. Widening straight to the ceiling on
    the strength of an eight-row match would buy the second tile for a single
    row.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_TILE_ROWS,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    widths = []
    for _ in range(7):
        width = gate.width_cap(64)
        widths.append(width)
        # The match never runs out: every proposed row comes back accepted.
        gate.observe(
            is_copy_draft=True,
            committed=width + 1,
            round_ms=70.0 + 9.0 * width,
            accepted=width,
            proposed=width,
        )

    assert widths == [2, 2, 2, 4, 8, 31, 63], widths
    # Nothing in between: every width past the free-width floor ends exactly
    # at a tile boundary, so no block ever pays for a tile it does not fill.
    for width in widths:
        if width > COPY_DRAFT_TILE_ROWS:
            assert (width + 1) % COPY_DRAFT_TILE_ROWS == 0, width


def test_copy_draft_width_stops_doubling_on_the_first_short_block():
    """The doubling is not a schedule, it is a response to a censored
    measurement: the first block the target cuts short is a real measurement
    of the match, and from there the width comes from that number."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    gate = CopyDraftGate()
    for _ in range(5):
        gate.observe(is_copy_draft=False, committed=2, round_ms=90.0)

    # Three saturated probes, then a fourth block that only half fills.
    for _ in range(3):
        gate.observe(
            is_copy_draft=True, committed=3, round_ms=88.0, accepted=2, proposed=2
        )
    assert gate.width_cap(31) == 4

    gate.observe(
        is_copy_draft=True, committed=3, round_ms=106.0, accepted=2, proposed=4
    )
    # Not saturated any more, so the width is 1.5x the measured acceptance
    # plus the row that discovers a longer match -- not eight.
    assert gate.width_cap(31) == 4


def test_copy_draft_width_keeps_headroom_above_a_measured_short_block():
    """A measured acceptance is a floor to build on, not a ceiling to sit at.

    A block sized to exactly what the last ones accepted can never discover
    that the match now runs further -- the measurement is censored by the
    width that produced it -- so the sizer adds
    ``COPY_DRAFT_WIDTH_HEADROOM`` and one row over the acceptance EWMA. This
    is the branch the doubling in
    ``test_copy_draft_width_doubles_while_the_block_comes_back_full`` hands
    over to on the first short block, and it has to be able to climb back:
    pinned here on a turn whose matches lengthen mid-answer, through to the
    saturated block that hands the whole ceiling back.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    gate = CopyDraftGate()
    # Three eight-row blocks the match cut short at four rows: a real
    # measurement, unlike a block that filled up.
    for _ in range(3):
        gate.observe(
            is_copy_draft=True, committed=5, round_ms=110.0, accepted=4, proposed=8
        )
    # 1.5x four rows plus the row that looks past them. Sized to the
    # measurement alone this would be 4 and the fifth row would be
    # unreachable for the rest of the turn.
    assert gate.width_cap(31) == 7
    assert gate.width_cap(31) > 4

    # The match lengthens -- a later paragraph quotes more of the prompt --
    # and the blocks still come back short, so the width has to follow the
    # acceptance up rather than wait for a saturated block.
    widths = []
    for _ in range(3):
        gate.observe(
            is_copy_draft=True, committed=7, round_ms=124.0, accepted=6, proposed=7
        )
        widths.append(gate.width_cap(31))
    assert widths == sorted(widths), widths
    assert widths[-1] > 7, widths

    # And once a block does come back full, the estimate is a floor again:
    # twice the last width lands inside the flat band, where narrowing buys
    # nothing, so the ceiling comes back whole.
    gate.observe(
        is_copy_draft=True, committed=9, round_ms=132.0, accepted=8, proposed=8
    )
    assert gate.width_cap(31) == 31


def test_copy_draft_width_narrows_on_the_first_short_block_after_a_full_ladder():
    """A ladder that climbed to the tile edge must still fall off it at once.

    The estimate the sizer narrows to is an average of *measurements*, and a
    block the target consumed end to end is not one -- it says the match runs
    at least this far, not how far. Averaging saturated widths in is what
    makes the fall-back late: a ladder that reached 31 on full blocks leaves
    the average at 31, so the first block that comes back three rows long
    reads as 22 and still asks for a whole tile, for as many rounds as an
    alpha-0.3 average needs to decay past the free-width floor. Each of those
    rounds is a 31-row verify -- about four baseline rounds -- returning four
    tokens.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_FREE_WIDTH_FLOOR,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    for _ in range(5):
        gate.observe(is_copy_draft=False, committed=1, round_ms=67.0)

    # Climb the ladder on blocks the target consumes end to end.
    while gate.width_cap(31) < 31:
        width = gate.width_cap(31)
        gate.observe(
            is_copy_draft=True,
            committed=width + 1,
            round_ms=70.0 + 9.0 * width,
            accepted=width,
            proposed=width,
        )
    assert gate.width_cap(31) == 31

    # The match runs out: 3 of 31 rows accepted. That is the first actual
    # measurement of this turn's match length, so it is the whole estimate.
    gate.observe(
        is_copy_draft=True, committed=4, round_ms=342.0, accepted=3, proposed=31
    )
    narrowed = gate.width_cap(31)
    assert narrowed == 5, narrowed
    # Below the free band, so the next verify is a cheap one rather than
    # another full-tile block bought on a decaying average.
    assert narrowed < COPY_DRAFT_FREE_WIDTH_FLOOR


def test_copy_draft_width_needs_a_filled_tile_to_buy_the_next_one():
    """Headroom above a short block stays inside that block's own tile.

    Rows inside a tile the verify already pays for are free, which is why the
    sizer hands back the whole tile once the evidence lands in the free band.
    The first row of the *next* tile is not free -- it doubles the verify --
    and 1.5x of a measured acceptance can land there on its own: a block that
    accepted 21 of 31 measured a match that ended inside the first tile, and
    1.5x of 21 is 32. Crossing an edge is earned by filling the tile below
    it, never by extrapolating past it.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_TILE_ROWS,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    for _ in range(5):
        gate.observe(is_copy_draft=False, committed=1, round_ms=67.0)
    while gate.width_cap(64) < 31:
        width = gate.width_cap(64)
        gate.observe(
            is_copy_draft=True,
            committed=width + 1,
            round_ms=70.0 + 9.0 * width,
            accepted=width,
            proposed=width,
        )

    # A long match that still ends inside the first tile, three times over so
    # the average settles on it rather than on the ladder it came from.
    for _ in range(3):
        gate.observe(
            is_copy_draft=True, committed=22, round_ms=342.0, accepted=21, proposed=31
        )
    held = gate.width_cap(64)
    assert held == COPY_DRAFT_TILE_ROWS - 1, held

    # Now a block that does fill the tile. That is the evidence the second
    # tile was missing, and the width crosses on the next proposal.
    gate.observe(
        is_copy_draft=True, committed=32, round_ms=342.0, accepted=31, proposed=31
    )
    assert gate.width_cap(64) == 2 * COPY_DRAFT_TILE_ROWS - 1


def test_copy_draft_width_never_pays_for_a_tile_it_cannot_fill():
    """No ceiling, edge-aligned or not, can make the sizer buy a part tile.

    The widths the sizer chooses are tile edges, but the ceiling it is
    offered is an operator's number and need not be one. A ceiling of 32
    bounds the request at a 33 row verify -- two tiles for one row past the
    edge, very nearly double the wall time of the 31 row block it beats by a
    single token. So the invariant is stated over every ceiling a caller can
    pass, not over the family default: a returned width either ends exactly
    at a tile edge or fits inside the first tile, where the rows are already
    paid for.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_TILE_ROWS,
        CopyDraftGate,
    )

    for ceiling in range(2, 4 * COPY_DRAFT_TILE_ROWS + 2):
        gate = CopyDraftGate()
        for _ in range(5):
            gate.observe(is_copy_draft=False, committed=1, round_ms=67.0)
        # Climb until the ladder stops moving: whatever width the ceiling
        # settles the sizer on, it has to be one the verify fills.
        seen = []
        for _ in range(12):
            width = gate.width_cap(ceiling)
            seen.append(width)
            assert width <= ceiling, (ceiling, width)
            assert (
                width < COPY_DRAFT_TILE_ROWS or (width + 1) % COPY_DRAFT_TILE_ROWS == 0
            ), (ceiling, width, seen)
            gate.observe(
                is_copy_draft=True,
                committed=width + 1,
                round_ms=70.0 + 9.0 * width,
                accepted=width,
                proposed=width,
            )

    # And the step-down is a step down, not a collapse: a ceiling one row
    # past the first edge settles on that edge rather than on a probe.
    gate = CopyDraftGate()
    for _ in range(5):
        gate.observe(is_copy_draft=False, committed=1, round_ms=67.0)
    for _ in range(12):
        width = gate.width_cap(COPY_DRAFT_TILE_ROWS)
        gate.observe(
            is_copy_draft=True,
            committed=width + 1,
            round_ms=70.0 + 9.0 * width,
            accepted=width,
            proposed=width,
        )
    assert gate.width_cap(COPY_DRAFT_TILE_ROWS) == COPY_DRAFT_TILE_ROWS - 1


def test_copy_draft_width_cap_never_collapses_below_a_usable_block():
    """A run of fully rejected copies must not size the next block to zero
    rows: whether copying is worth anything at all is the gate's call, and
    the gate can still refuse."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_MIN_WIDTH,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    for _ in range(10):
        gate.observe(
            is_copy_draft=True, committed=1, round_ms=342.0, accepted=0, proposed=2
        )

    assert gate.width_cap(31) == COPY_DRAFT_MIN_WIDTH
    assert COPY_DRAFT_MIN_WIDTH >= 2


def test_copy_draft_width_cap_never_exceeds_what_the_match_offers():
    """The ceiling is the match, and the floor never reaches past it.

    ``COPY_DRAFT_MIN_WIDTH`` exists so a run of rejections cannot size the
    next block to zero rows. It must not turn a one-row match into a
    two-row request: the caller slices the match by whatever comes back, so
    a width past the end of the match is a width that claims rows nobody
    found."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_MIN_WIDTH,
        CopyDraftGate,
    )

    gate = CopyDraftGate()
    for _ in range(10):
        gate.observe(
            is_copy_draft=True, committed=1, round_ms=342.0, accepted=0, proposed=2
        )

    # Past the probe stage, where the estimate alone would floor at 2.
    assert gate.width_cap(31) == COPY_DRAFT_MIN_WIDTH
    for ceiling in range(1, 6):
        assert gate.width_cap(ceiling) <= ceiling


def test_generator_sizes_copy_drafts_through_the_gate(monkeypatch):
    """The sizer is on the proposal path, not just available to it."""
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step
    from rapid_mlx.spec_decode.mtp.prompt_lookup import CopyDraftGate

    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "2")
    monkeypatch.setattr(
        generator_mod, "_safe_prompt_lookup_draft_count", lambda _c, n, **_kw: n
    )

    asked: list[int] = []
    monkeypatch.setattr(
        CopyDraftGate,
        "width_cap",
        lambda _self, proposed: (asked.append(proposed), 1)[1],
    )

    # The sizer can only adapt on what the generator tells it afterwards, and
    # a round that forgot to report its width would leave every unit test
    # above green while adaptive sizing quietly stopped working. So record
    # the feedback too, not just the request.
    fed: list[tuple[int, int]] = []
    real_observe = CopyDraftGate.observe

    def recording_observe(self, *, is_copy_draft, committed, round_ms, **kw):
        if is_copy_draft:
            fed.append((kw.get("accepted", -1), kw.get("proposed", -1)))
        return real_observe(
            self,
            is_copy_draft=is_copy_draft,
            committed=committed,
            round_ms=round_ms,
            **kw,
        )

    monkeypatch.setattr(CopyDraftGate, "observe", recording_observe)

    timing: dict[str, float] = {}
    list(
        mtp_generate_step(
            mx.array([7, 8, 20, 21], dtype=mx.uint32),
            _CacheAdvancingQwen35Model(
                backbone_outputs=[0, 0, 0, 7, 8, 20, 21, 22],
                mtp_outputs=[0, 0, 0, 99, 20, 21],
            ),
            max_tokens=3,
            max_k=1,
            disable_auto_k=True,
            prompt_cache=[_CountingKVCache(), _CountingKVCache()],
            accept_counter=MTPAcceptCounter(),
            timing_stats=timing,
        )
    )

    assert asked, "the sizer was never consulted"
    assert timing["prompt_lookup_drafted_tokens"] == float(
        timing["prompt_lookup_proposals"]
    ), "a sizer answer of one row must draft exactly one row per proposal"

    assert fed, "no copy-draft round reported back to the gate"
    for accepted, proposed in fed:
        # ``proposed`` is the width the round actually verified, which the
        # mocked sizer pinned to one row -- not the ceiling the match offered.
        assert proposed == 1, f"round verified 1 row but reported {proposed}"
        # ``accepted`` is bounded by it: a row cannot be accepted twice, and
        # a rejected block still has to say so rather than report nothing.
        assert 0 <= accepted <= proposed, (
            f"reported {accepted} accepted rows out of {proposed} verified"
        )


def test_family_injector_still_imports_without_mlx():
    """``qwen3_5_inject`` must stay importable on a machine with no MLX.

    Two test modules -- ``test_mtp_batched_family_capability`` and
    ``test_mtp_self_contained_repo`` -- import it for pure AST and path
    checks, and CI runs them on a hosted Linux lane where ``mlx`` is not
    installed at all. A module-level import that reaches
    ``spec_decode.mtp.generator`` (which imports ``mlx.core``) turns those
    two files into collection errors on three Python versions. That is why
    ``MAX_COPY_DRAFT_TOKENS`` lives in ``prompt_lookup``: the injector needs
    the number, and this is the module that can hand it over without
    dragging the array library in.

    Run in a subprocess so blocking ``mlx`` cannot disturb this process's
    already-imported modules.
    """
    import subprocess
    import sys

    program = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "mlx" or name.startswith("mlx."):
                    raise ImportError("blocked: " + name)
                return None

        for name in [m for m in sys.modules if m.split(".")[0] in {"mlx", "rapid_mlx"}]:
            del sys.modules[name]
        sys.meta_path.insert(0, Blocker())

        import rapid_mlx.spec_decode.mtp.qwen3_5_inject as inject
        from rapid_mlx.spec_decode.mtp.prompt_lookup import MAX_COPY_DRAFT_TOKENS

        assert MAX_COPY_DRAFT_TOKENS == 31, MAX_COPY_DRAFT_TOKENS
        assert inject.MAX_COPY_DRAFT_TOKENS == MAX_COPY_DRAFT_TOKENS
        print("ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, (
        f"qwen3_5_inject no longer imports without MLX:\n{done.stdout}\n{done.stderr}"
    )
    assert done.stdout.strip().endswith("ok")


def test_prompt_lookup_exports_everything_it_asks_callers_to_use():
    """``__all__`` must name every public symbol the module actually offers.

    The gate arrived as a module-level class with eight configuration
    constants beside it, and the first version of that change left
    ``__all__`` naming only the three original types. A wildcard import then
    hands out a different surface than a direct one, and any generated API
    inventory disagrees with what the generator and these tests import by
    name. Pinning the two together means the next constant added here cannot
    quietly fall out of the public list.
    """
    from rapid_mlx.spec_decode.mtp import prompt_lookup

    # Defined here, not merely visible here: a name imported for internal use
    # (``dataclass`` today) is not part of this module's surface. Modules are
    # excluded explicitly because a module object carries no ``__module__``,
    # so a future ``import bisect`` would otherwise read as public.
    public = {
        name
        for name, value in vars(prompt_lookup).items()
        if not name.startswith("_")
        and not isinstance(value, types.ModuleType)
        and getattr(value, "__module__", prompt_lookup.__name__)
        == prompt_lookup.__name__
    }
    declared = set(prompt_lookup.__all__)
    assert public - declared == set(), (
        f"public but not exported: {sorted(public - declared)}"
    )
    assert declared - public == set(), (
        f"exported but not defined here: {sorted(declared - public)}"
    )
