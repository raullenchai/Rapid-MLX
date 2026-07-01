# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Gemma 4 MTP inject module (PR-3 of the 0.9.11 stack).

The tests cover the **wiring contract only** — they do NOT drive the
generator end-to-end against a real Gemma 4 checkpoint. The Mia-AiLab
GGUF-derived sidecar is an ``gemma4-assistant`` architecture that the
current inject deliberately refuses (see
:mod:`vllm_mlx.spec_decode.mtp.gemma4_inject` docstring). Wiring +
routing + refusal-path coverage is what this file locks down.

Coverage
--------

1. **Wiring probe** — with ``allow_random_init=True`` the four MTP
   contract surfaces (``.mtp``, ``.mtp_forward``, ``.make_mtp_cache``,
   ``.__call__`` accepts ``return_hidden`` + ``n_confirmed``) attach
   to a synthetic Gemma 4 text model.
2. **Sidecar refusal (default)** — no sidecar and no ``allow_random_init``
   → inject returns False and the model is unmodified (matches the
   codex round-5 fail-closed default from ``qwen3_5_inject``).
3. **Weight loading (synthetic sidecar, scaffold-only)** — a hand-built
   Qwen3.5-style sidecar (matching the scaffold module's parameter
   tree) round-trips through save/load correctly. This proves the
   sidecar resolver + coverage check are wired.
4. **``gemma4-assistant`` refusal** — a sidecar carrying the Mia-AiLab
   tensor fingerprints (``mtp.pre_projection.weight`` /
   ``mtp.post_projection.weight``) is REFUSED with the architecture
   guard rather than partially loaded.
5. **Dispatcher routing** — ``gemma4`` / ``gemma4_unified`` route to
   this module; ``gemma3`` (Gemma 3, deliberately not on the MTP
   allowlist) returns False without importing this module.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_mtp_state():
    """Reset the shared MTP module state — same recipe as
    tests/test_mtp_spec_decode.py::_reset_mtp_module_state.

    The cache_patch install gate is process-global; between tests we
    unpatch so each test starts from the pre-install baseline. We also
    reset the accept counter for cleanliness (though these tests don't
    write to it).
    """
    import sys

    from vllm_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from vllm_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests

    _unpatch_for_tests()
    reset_global_counter_for_tests()

    # Re-bind ``mlx_lm.generate.generation_stream`` to this thread's
    # default stream — same reasoning as the Qwen3.5 test file (see
    # test_mtp_spec_decode._reset_mtp_module_state). Prevents the
    # cross-thread stream crash when tests run in a sweep after
    # something spun up an mlx-step executor thread.
    import mlx_lm.generate  # noqa: F401

    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )
    yield
    _unpatch_for_tests()
    reset_global_counter_for_tests()
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )


def _tiny_gemma4_text_args():
    """Minimal ``gemma4_text.ModelArgs`` sized for shape tests.

    Small dims so tests stay under a second. We drop the multi-modal
    ``gemma4`` wrapper — the inject helper accepts the inner text
    model directly (test path), which matches how ``qwen3_5_inject``'s
    tests work.
    """
    from mlx_lm.models.gemma4_text import ModelArgs

    args = ModelArgs(
        model_type="gemma4_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        num_kv_shared_layers=0,  # simplifies the wiring probe path
        hidden_size_per_layer_input=0,  # skip per-layer input gating
        sliding_window=64,
        sliding_window_pattern=1,
        max_position_embeddings=128,
        final_logit_softcapping=None,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        tie_word_embeddings=True,
    )
    # Layer in mtp_num_hidden_layers — same pattern as the Qwen3.5 test
    # helper (the mlx-lm 0.31.3 dataclass doesn't have the field yet).
    object.__setattr__(args, "mtp_num_hidden_layers", 1)
    return args


def _build_tiny_gemma4_text_model():
    """Construct the inner ``gemma4_text.Model`` for wiring tests.

    Returns the outer ``gemma4_text.Model`` (which itself contains a
    ``Gemma4TextModel`` under ``.model``). Both surfaces are what the
    inject's ``_resolve_inner_text_model`` accepts.
    """
    from mlx_lm.models.gemma4_text import Model

    args = _tiny_gemma4_text_args()
    return Model(args)


# ---------------------------------------------------------------------------
# 1. Wiring probe — four surfaces attach under allow_random_init=True
# ---------------------------------------------------------------------------


def test_inject_mtp_support_attaches_four_surfaces():
    """Wiring probe — ``.mtp`` + ``.mtp_forward`` + ``.make_mtp_cache``
    attach, and ``__call__`` accepts both ``return_hidden`` and
    ``n_confirmed`` kwargs.

    Mirrors the Qwen3.5 wiring probe test shape. The scaffold module
    that lands here (see ``gemma4_inject._build_scaffold_mtp_module``)
    is NOT the eventual gemma4-assistant architecture — the wiring
    probe only proves that the inject can attach surfaces to a Gemma
    4-shaped model.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch in this mlx-lm: {exc}")

    result = inject_mtp_support(model, allow_random_init=True)
    assert result is True, (
        "inject_mtp_support should attach the scaffold under allow_random_init=True."
    )
    assert validate_mtp_support(model) is True, (
        "validate_mtp_support should see all four surfaces after the "
        "wiring probe fires."
    )


# ---------------------------------------------------------------------------
# 2. Sidecar refusal — no sidecar + default => False
# ---------------------------------------------------------------------------


def test_inject_mtp_support_refuses_no_sidecar_by_default():
    """Default ``allow_random_init=False`` must fail closed on no sidecar.

    Codex round-5 BLOCKING fix on ``qwen3_5_inject`` established that
    silently shipping a random-init MTP head looks like spec-decode is
    live but yields zero speedup. Gemma 4 gets the same default — the
    refusal path is CRITICAL for gemma4 because the scaffold MTP head
    here isn't even the correct architecture, so a random-init inject
    would be doubly-wrong.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch: {exc}")

    assert inject_mtp_support(model) is False, (
        "Default inject_mtp_support without sidecar should return False."
    )
    assert validate_mtp_support(model) is False, (
        "The model must NOT have been patched — none of the four "
        "surfaces should exist on a failed inject."
    )


def test_inject_mtp_support_rejects_model_without_mtp_config():
    """A Gemma 4 model whose config lacks ``mtp_num_hidden_layers``
    (stock ``mlx-community/gemma-4-*`` release) → inject returns False.

    Matches the Qwen3.5 ``test_inject_mtp_support_rejects_stripped_checkpoint``
    contract.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    class _FakeArgs:
        hidden_size = 64
        rms_norm_eps = 1e-6
        # NOTE: no mtp_num_hidden_layers attribute — matches the stock
        # mlx-community release.

    class _FakeInner:
        args = _FakeArgs()
        model = object()

    assert inject_mtp_support(_FakeInner(), allow_random_init=True) is False


# ---------------------------------------------------------------------------
# 3. Weight loading — synthetic sidecar matching the scaffold's tree
# ---------------------------------------------------------------------------


def test_inject_mtp_support_loads_synthetic_sidecar_under_test_opt_in():
    """A hand-built sidecar matching the SCAFFOLD's parameter tree
    round-trips through save → load → attach when the test-only opt-in
    (``_accept_scaffold_sidecar_for_tests=True``) is passed.

    This exercises the sidecar resolver + coverage check + safetensors
    round-trip. It does NOT claim the scaffold can drive production
    drafts — codex round-2 established that the scaffold decoder layer
    has no KVCache update path, so multi-token spec decode against it
    would be broken. The private opt-in kwarg exists precisely to
    exercise the loader machinery in unit tests without opening a
    production regression window.
    """
    import tempfile
    from pathlib import Path

    import mlx.core as _mx
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        _build_scaffold_mtp_module,
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model_a = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch: {exc}")

    args = model_a.args
    mtp_template = _build_scaffold_mtp_module(args, 1)
    _mx.eval(mtp_template.parameters())
    flat = dict(tree_flatten(mtp_template.parameters()))
    assert flat, "scaffold MTP module produced an empty parameter tree"

    with tempfile.TemporaryDirectory() as tmp:
        sidecar_path = Path(tmp) / "synthetic-gemma4-mtp.safetensors"
        _mx.save_safetensors(str(sidecar_path), flat)

        model_b = _build_tiny_gemma4_text_model()

        # Default path: any sidecar is refused in production. Codex
        # round-2 blocking fix — no KVCache in the scaffold layer.
        assert inject_mtp_support(model_b, mtp_sidecar=str(sidecar_path)) is False
        assert validate_mtp_support(model_b) is False, (
            "Production-default refusal must leave the model unmodified."
        )

        # Test opt-in path — proves the loader machinery works.
        result = inject_mtp_support(
            model_b,
            mtp_sidecar=str(sidecar_path),
            _accept_scaffold_sidecar_for_tests=True,
        )
        assert result is True, (
            "With the test opt-in kwarg, a scaffold-shaped sidecar should "
            "load. If this failed, the coverage check has drifted from "
            "the scaffold's parameter tree."
        )
        assert validate_mtp_support(model_b) is True

        # Verify byte-equal load.
        loaded = dict(tree_flatten(model_b.mtp.parameters()))
        assert set(loaded.keys()) == set(flat.keys()), (
            f"Parameter trees diverged. "
            f"In template only: {set(flat) - set(loaded)}. "
            f"In loaded only: {set(loaded) - set(flat)}."
        )
        for k in flat:
            diff = _mx.sum(loaded[k] != flat[k]).item()
            assert diff == 0, f"{k}: loaded weight differs by {diff} entries"


def test_scaffold_mtp_forward_raises_loudly_when_invoked():
    """Codex round-3 fix: the scaffold ``mtp_forward`` must RAISE when
    called, not silently return zeros. Prevents a caller that reaches
    the scaffold at runtime from producing garbage drafts and thinking
    everything is fine.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    try:
        model = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch: {exc}")

    assert inject_mtp_support(model, allow_random_init=True) is True

    # mtp_forward must raise NotImplementedError with the
    # AssistantModel-follow-up message.
    with pytest.raises(NotImplementedError, match="AssistantModel"):
        model.mtp_forward(mx.zeros((1, 1, 64)), mx.zeros((1, 1), dtype=mx.uint32), [])


def test_scaffold_call_raises_on_return_hidden_or_n_confirmed():
    """Codex round-3 fix: ``__call__`` on the scaffold must raise when
    ``return_hidden=True`` or ``n_confirmed>0`` — those flags cannot
    be honestly implemented in the scaffold, and returning fake
    zeros (previous behavior) silently corrupted downstream drafts.
    """
    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    try:
        model = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch: {exc}")

    assert inject_mtp_support(model, allow_random_init=True) is True

    inputs = mx.array([[1, 2, 3]], dtype=mx.uint32)
    with pytest.raises(NotImplementedError, match="wiring-only"):
        model(inputs, return_hidden=True)
    with pytest.raises(NotImplementedError, match="wiring-only"):
        model(inputs, n_confirmed=1)

    # Bare forward (no MTP flags) must still work — the scaffold does
    # not corrupt the base model's non-MTP forward path.
    out = model(inputs)
    assert out.shape[:2] == inputs.shape, (
        f"bare forward output shape {out.shape} doesn't match input {inputs.shape}"
    )


def test_inject_mtp_support_refuses_corrupt_sidecar():
    """A sidecar file that isn't valid safetensors must be REFUSED,
    not crash the caller. Codex round-2 blocking fix: mx.load raises
    on garbage input, and the ``never raises`` inject contract must
    catch that.
    """
    import tempfile
    from pathlib import Path

    from vllm_mlx.spec_decode.mtp.gemma4_inject import inject_mtp_support

    try:
        model = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        bad_sidecar = Path(tmp) / "corrupt.safetensors"
        bad_sidecar.write_bytes(b"this is not a safetensors file, just garbage")

        # Must NOT raise; must return False.
        result = inject_mtp_support(
            model,
            mtp_sidecar=str(bad_sidecar),
            _accept_scaffold_sidecar_for_tests=True,
        )
        assert result is False


# ---------------------------------------------------------------------------
# 4. gemma4-assistant refusal — Mia-AiLab-fingerprint sidecar rejected
# ---------------------------------------------------------------------------


def test_inject_mtp_support_refuses_assistant_architecture_sidecar():
    """A sidecar carrying the ``gemma4-assistant`` pre/post-projection
    tensor fingerprints must be REFUSED — not partially loaded.

    Failure mode this guards against: an operator runs
    ``scripts/convert_gemma4_mtp_gguf.py``, points ``rapid-mlx serve
    --mtp-sidecar`` at the output, and gets a random-init draft that
    silently yields zero speedup. The architecture-guard log message
    must fire and inject must return False.
    """
    import tempfile
    from pathlib import Path

    import mlx.core as _mx

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    try:
        model = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch: {exc}")

    # Build a tiny "assistant-shaped" sidecar with pre/post projection
    # keys — enough for _looks_like_assistant_sidecar to trigger.
    # Weights are minimal zeros; shape doesn't matter because the
    # architecture guard fires before the coverage check runs.
    assistant_weights = {
        "mtp.pre_projection.weight": _mx.zeros((16, 32)),
        "mtp.post_projection.weight": _mx.zeros((32, 16)),
        "mtp.embed_tokens.weight": _mx.zeros((32, 16)),
        "mtp.norm.weight": _mx.zeros((16,)),
    }

    with tempfile.TemporaryDirectory() as tmp:
        sidecar_path = Path(tmp) / "assistant-shaped-sidecar.safetensors"
        _mx.save_safetensors(str(sidecar_path), assistant_weights)

        result = inject_mtp_support(model, mtp_sidecar=str(sidecar_path))
        assert result is False, (
            "gemma4_inject must REFUSE a sidecar carrying the assistant "
            "architecture markers. The architecture guard has regressed."
        )
        assert validate_mtp_support(model) is False, (
            "Refused inject must leave the model unmodified — no surfaces attached."
        )


# ---------------------------------------------------------------------------
# 5. Dispatcher routing — gemma4 / gemma4_unified → this module
# ---------------------------------------------------------------------------


def test_dispatcher_routes_gemma4_to_gemma4_inject():
    """``model_type='gemma4'`` dispatches to ``gemma4_inject.inject_mtp_support``."""
    from vllm_mlx.spec_decode.mtp.dispatch import dispatch_mtp_inject

    try:
        model = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch: {exc}")

    result = dispatch_mtp_inject(
        model,
        model_type="gemma4",
        allow_random_init=True,
    )
    assert result is True, (
        "Dispatcher must forward to gemma4_inject for model_type='gemma4'"
    )
    from vllm_mlx.spec_decode.mtp.gemma4_inject import validate_mtp_support

    assert validate_mtp_support(model) is True


def test_dispatcher_routes_gemma4_unified_to_gemma4_inject():
    """``model_type='gemma4_unified'`` dispatches to the same module.

    The 12B unified variants (``gemma4_unified`` model_type) share the
    inject with the multimodal ``gemma4`` — the inner text-model
    architecture is identical for MTP purposes.
    """
    from vllm_mlx.spec_decode.mtp.dispatch import dispatch_mtp_inject

    try:
        model = _build_tiny_gemma4_text_model()
    except (TypeError, AttributeError) as exc:
        pytest.skip(f"Gemma 4 text ModelArgs schema mismatch: {exc}")

    result = dispatch_mtp_inject(
        model,
        model_type="gemma4_unified",
        allow_random_init=True,
    )
    assert result is True


def test_dispatcher_rejects_gemma3():
    """Gemma 3 (not Gemma 4) is not on the MTP allowlist — dispatch is a no-op.

    Guards against a copy-paste bug in
    ``dispatch._MTP_INJECT_DISPATCH`` accidentally routing Gemma 3 to
    the Gemma 4 inject (would attempt to patch a differently-shaped
    architecture).
    """
    from vllm_mlx.spec_decode.mtp.dispatch import dispatch_mtp_inject

    class _Shell:
        args = None
        model = None
        language_model = None

    result = dispatch_mtp_inject(_Shell(), model_type="gemma3")
    assert result is False, (
        "Dispatcher must not route model_type='gemma3' to any MTP inject."
    )


def test_dispatcher_still_routes_qwen3_5():
    """Qwen3.5 routing must NOT regress — the dispatcher must
    forward ``model_type='qwen3_5'`` to ``qwen3_5_inject.inject_mtp_support``.

    Locks the dispatch table entries in place (a copy-paste swap of
    ``qwen3_5_inject`` for ``gemma4_inject`` on the ``qwen3_5``
    branch would silently misroute production Qwen3.5 requests).
    """
    from vllm_mlx.spec_decode.mtp import dispatch as _dispatch

    assert "qwen3_5" in _dispatch._MTP_INJECT_DISPATCH
    module_path, _ = _dispatch._MTP_INJECT_DISPATCH["qwen3_5"]
    assert module_path == "vllm_mlx.spec_decode.mtp.qwen3_5_inject"

    # gemma4 must ALSO be in the dispatch (the whole point of PR-3).
    assert "gemma4" in _dispatch._MTP_INJECT_DISPATCH
    assert "gemma4_unified" in _dispatch._MTP_INJECT_DISPATCH
    for mt in ("gemma4", "gemma4_unified"):
        mp, _ = _dispatch._MTP_INJECT_DISPATCH[mt]
        assert mp == "vllm_mlx.spec_decode.mtp.gemma4_inject"


def test_dispatcher_forwards_model_and_kwargs_verbatim(monkeypatch):
    """Dispatcher must forward ``model``, ``mtp_sidecar``, and
    ``allow_random_init`` verbatim to the family-specific inject.

    Guards against:
    * Silent argument drops (e.g. dispatcher forgets to pass through
      ``mtp_sidecar``, and inject silently random-inits).
    * Return-value mismangling (dispatcher returns ``None`` / truthy
      wrong object instead of the ``bool`` the family fn returned).

    We monkey-patch the target module's ``inject_mtp_support`` so the
    test doesn't have to build a real Gemma 4 model to prove the wire.
    """
    from vllm_mlx.spec_decode.mtp import dispatch as _dispatch
    from vllm_mlx.spec_decode.mtp import gemma4_inject

    calls: list[dict] = []

    def _fake_inject(model, mtp_sidecar=None, *, allow_random_init=False):
        calls.append(
            {
                "model": model,
                "mtp_sidecar": mtp_sidecar,
                "allow_random_init": allow_random_init,
            }
        )
        return True

    monkeypatch.setattr(gemma4_inject, "inject_mtp_support", _fake_inject)

    sentinel_model = object()
    sentinel_sidecar = "/tmp/nonexistent-sentinel-sidecar.safetensors"

    result = _dispatch.dispatch_mtp_inject(
        sentinel_model,
        model_type="gemma4_unified",
        mtp_sidecar=sentinel_sidecar,
        allow_random_init=False,
    )
    assert result is True, "dispatcher must return the family fn's bool verbatim"
    assert len(calls) == 1
    assert calls[0]["model"] is sentinel_model
    assert calls[0]["mtp_sidecar"] == sentinel_sidecar
    assert calls[0]["allow_random_init"] is False

    # A second call with different kwargs must forward those too.
    result = _dispatch.dispatch_mtp_inject(
        sentinel_model,
        model_type="gemma4",
        mtp_sidecar=None,
        allow_random_init=True,
    )
    assert result is True
    assert len(calls) == 2
    assert calls[1]["allow_random_init"] is True
    assert calls[1]["mtp_sidecar"] is None


def test_dispatcher_returns_false_for_unknown_model_type():
    """Unknown ``model_type`` → False, no exception. Fail-closed default.

    Guards against a KeyError leak — the dispatcher is designed to
    treat unknown types as "no MTP for this arch" and log rather than
    raise (matches the ``qwen3_5_inject`` fail-closed default codex
    round-5 installed).
    """
    from vllm_mlx.spec_decode.mtp.dispatch import dispatch_mtp_inject

    result = dispatch_mtp_inject(
        object(),
        model_type="not_a_real_model_type_xyzzy",
        allow_random_init=True,
    )
    assert result is False
