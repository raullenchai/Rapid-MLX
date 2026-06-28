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

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _reset_mtp_module_state():
    """Reset the MTP module-level singletons AND ``mlx_lm.generate``'s
    captured ``generation_stream`` between tests.

    Three pieces of cross-test state leak in the full pytest sweep and
    surface as the 7-failure transient cluster (PASS in isolation):

    * ``vllm_mlx.spec_decode.mtp.cache_patch._patched`` — sticky install
      gate; ``_unpatch_for_tests()`` clears it.
    * ``vllm_mlx.spec_decode.mtp.accept_counter._global_counter`` —
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

    from vllm_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from vllm_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests

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
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_eligibility_qwen3_5_moe_chain():
    """Qwen3.5 MoE with mtp_num_hidden_layers=1 → CHAIN (same path)."""
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 1}
    assert detect_mtp_eligibility(config) is MTPEligibility.CHAIN


def test_detect_eligibility_qwen3_5_tree_reserved():
    """mtp_num_hidden_layers >= 2 → TREE (reserved, not implemented)."""
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 4}
    assert detect_mtp_eligibility(config) is MTPEligibility.TREE


def test_detect_eligibility_non_qwen35_models_rejected():
    """Llama / Mistral / Qwen3 / Qwen3-Next must NOT match the MTP path."""
    from vllm_mlx.spec_decode.mtp import (
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
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 0}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE


def test_detect_eligibility_handles_string_and_float_config():
    """Hand-edited / HF re-uploaded configs may carry strings / floats —
    detection coerces rather than crashing.
    """
    from vllm_mlx.spec_decode.mtp import (
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
    from vllm_mlx.spec_decode.mtp import (
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
    from vllm_mlx.spec_decode.mtp import (
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
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    snap = counter.snapshot()
    assert snap.attempts == 0
    assert snap.accepts == 0
    assert snap.tokens_saved == 0
    assert snap.accept_ratio == 0.0  # zero attempts → 0 (not NaN)


def test_accept_counter_record_attempt_and_accept():
    """5 attempts, 3 accepts → ratio 0.6, tokens_saved = 3."""
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

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
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    counter.record_attempt()
    counter.record_reject()
    snap = counter.snapshot()
    assert snap.attempts == 1
    assert snap.accepts == 0
    assert snap.tokens_saved == 0


def test_accept_counter_rejects_negative_tokens_saved():
    """``record_accept(tokens_saved=-1)`` is a programmer error — fail loud."""
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    with pytest.raises(ValueError, match="non-negative"):
        counter.record_accept(tokens_saved=-1)


def test_accept_counter_reset_for_tests_resets_all_three():
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    counter = MTPAcceptCounter()
    counter.record_attempt()
    counter.record_accept(tokens_saved=2)
    counter.reset()
    snap = counter.snapshot()
    assert (snap.attempts, snap.accepts, snap.tokens_saved) == (0, 0, 0)


def test_global_counter_singleton_identity():
    """``get_global_counter`` returns the same instance across calls."""
    from vllm_mlx.spec_decode.mtp.accept_counter import get_global_counter

    a = get_global_counter()
    b = get_global_counter()
    assert a is b


def test_accept_counter_snapshot_under_concurrent_writes_is_safe():
    """Concurrent record_* calls must not corrupt the snapshot — the
    ``threading.Lock`` keeps the three fields in lockstep.
    """
    import threading

    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

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

    from vllm_mlx.spec_decode.mtp.cache_patch import (
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


def test_cache_patch_is_idempotent():
    """Second call returns False — already-installed is not an error."""
    from vllm_mlx.spec_decode.mtp.cache_patch import (
        patch_arrays_cache_rollback_state,
    )

    # Force at least one install
    patch_arrays_cache_rollback_state()
    second = patch_arrays_cache_rollback_state()
    assert second is False


# ---------------------------------------------------------------------------
# 4. CLI flag parsing + SchedulerConfig plumbing
# ---------------------------------------------------------------------------


def _serve_help_stdout() -> str:
    """Run ``python -m vllm_mlx.cli serve --help`` and return stdout.

    Mirrors :mod:`tests.test_kv_cache_dtype_cli` — the serve parser is
    inlined into ``main()``, so subprocess inspection is the canonical
    way to assert that the flag landed.
    """
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


def test_cli_spec_decode_flag_advertised_in_help():
    """``--spec-decode {none,mtp}`` must appear in ``rapid-mlx serve --help``."""
    text = _serve_help_stdout()
    assert "--spec-decode" in text
    # argparse renders ``--spec-decode {none,mtp}`` for the choices.
    assert "none,mtp" in text or "mtp,none" in text


def test_cli_spec_decode_flag_rejects_unknown_value():
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


def test_scheduler_config_default_spec_decode_is_none():
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.spec_decode == "none"


def test_scheduler_config_spec_decode_round_trip():
    """Field round-trips ``mtp`` from kwargs."""
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(spec_decode="mtp")
    assert cfg.spec_decode == "mtp"


# ---------------------------------------------------------------------------
# 5. Metrics rendering
# ---------------------------------------------------------------------------


def test_metrics_renders_spec_decode_counters_zero_at_cold_start():
    """Before any MTP generation runs, the four MTP series MUST be
    present with value 0 (engine-independence rationale — same as
    response_format and mxfp4 guardrail counters).
    """
    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from vllm_mlx.spec_decode.mtp.accept_counter import (
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
    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from vllm_mlx.spec_decode.mtp.accept_counter import (
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
    """Zero attempts → ratio gauge MUST be 0 (not NaN, not missing)."""
    from vllm_mlx.routes.metrics import _render_spec_decode_mtp_counters
    from vllm_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )

    reset_global_counter_for_tests()

    class _Cfg:
        model_alias = None

    body = "\n".join(_render_spec_decode_mtp_counters(_Cfg()))
    # family label falls back to "qwen3.5" when alias is None.
    assert 'family="qwen3.5"' in body
    assert 'rapid_mlx_spec_decode_accept_ratio{family="qwen3.5",method="mtp"} 0' in body


def test_metrics_route_includes_spec_decode_series_at_cold_start():
    """End-to-end: the /metrics body emitted by the full renderer must
    carry the spec_decode series before any engine is up — matches the
    response_format + mxfp4 pre-engine surface convention.
    """
    from vllm_mlx.routes.metrics import _render_prometheus

    class _Cfg:
        engine = None
        model_name = "qwen3.5-9b-4bit"
        model_alias = "qwen3.5-9b-4bit"
        kv_cache_dtype = None

    body = _render_prometheus(_Cfg())
    assert "rapid_mlx_spec_decode_attempts_total" in body
    assert "rapid_mlx_spec_decode_accept_ratio" in body


# ---------------------------------------------------------------------------
# 6. MTP head builder
# ---------------------------------------------------------------------------


def test_build_mtp_module_rejects_zero_layers():
    """``num_layers < 1`` is a programmer error — fail loud."""
    from vllm_mlx.spec_decode.mtp.head import build_mtp_module

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
    from vllm_mlx.spec_decode.mtp.head import build_mtp_module

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
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
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


def test_inject_mtp_support_rejects_non_qwen35_model():
    """A non-Qwen3.5 model (no ``args.mtp_num_hidden_layers``) must
    return False and not patch anything.
    """
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

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
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

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
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
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

    from vllm_mlx.spec_decode.mtp.head import build_mtp_module
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model_a = _build_tiny_qwen3_5_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Qwen3.5 TextModelArgs schema mismatch: {exc}")

    # Build the MTP head separately so we can capture its random-init
    # weights, write them to disk, and verify the inject loads them
    # byte-equally. (Note: this tiny model is FP, so no quantize step
    # — the inject's _detect_base_quantization returns None and the
    # MTP module stays FP, matching the sidecar layout.)
    args = model_a.args
    mtp_template = build_mtp_module(args, int(args.mtp_num_hidden_layers))
    _mx.eval(mtp_template.parameters())
    flat = dict(tree_flatten(mtp_template.parameters()))
    assert flat, "build_mtp_module produced an empty parameter tree"

    with tempfile.TemporaryDirectory() as tmp:
        sidecar_path = Path(tmp) / "synthetic-mtp-head.safetensors"
        _mx.save_safetensors(str(sidecar_path), flat)

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

    from vllm_mlx.spec_decode.mtp.head import build_mtp_module
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

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

        fresh_model = _build_tiny_qwen3_5_text_model()
        result = inject_mtp_support(fresh_model, mtp_sidecar=str(sidecar_path))
        assert result is False, (
            f"inject_mtp_support should have refused a sidecar missing {drop_key!r}, "
            f"but returned True — the coverage check has regressed."
        )


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
        seq = len(target_ids)
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
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    backbone = [7, 11, 13]
    mtp = [11]
    model = _MockedQwen35Model(backbone, mtp)

    counter = MTPAcceptCounter()
    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    for tok, _logprobs, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=3,
        accept_counter=counter,
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


def test_generator_rejection_path_does_not_count_as_accept():
    """When draft != verify_pred at temp=0 the generator takes the
    reject branch — counter shows attempt without accept.

    Sequence:

      cold-start backbone → 7 (primary)
      MTP head → draft=11
      verify backbone → verify_pred=12 (≠ draft → reject), bonus=99 (unused)

    Yields: (7, False), (12, False).
    """
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    backbone = [7, 12, 99]  # 99 is for the bonus slot — unused on reject
    mtp = [11, 22]  # 22 is for the next draft after reject (cold-start MTP)
    model = _MockedQwen35Model(backbone, mtp)

    counter = MTPAcceptCounter()
    prompt = mx.array([1], dtype=mx.uint32)
    emitted = []
    for tok, _logprobs, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=2,
        accept_counter=counter,
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
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

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
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

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
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

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
    list(
        mtp_generate_step(
            prompt,
            model,
            max_tokens=6,
            accept_counter=counter,
        )
    )

    snap = counter.snapshot()
    assert snap.attempts == 3, f"expected 3 attempts; got {snap}"
    assert snap.accepts == 2, f"expected 2 accepts; got {snap}"
    assert snap.tokens_saved == 2
    assert snap.accept_ratio == pytest.approx(2 / 3)
