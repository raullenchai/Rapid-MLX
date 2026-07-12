# SPDX-License-Identifier: Apache-2.0
"""Regression tests for explicit ``gemma4`` vs ``gemma4_unified`` routing.

Issue #509: ``is_gemma4_model`` used a ``"gemma4" in model_type`` substring
test that matched the non-unified ``gemma4`` arch (26B/31B/e2b/e4b), the
unified ``gemma4_unified`` arch (the 12B aliases), the ``gemma4_assistant``
aliases, AND would catch any hypothetical future sibling like
``gemma4_videogen`` or the inner sub-config's own ``gemma4_text`` label.
Everything loaded through the non-unified mlx-vlm subpackage. It worked
empirically (dataclass-identical ``TextConfig`` + shared ``LanguageModel``)
but was misleading and fragile.

These tests pin the corrected behavior — an exact-match allow-list, not a
substring test:
- ``is_gemma4_nonunified_model`` matches the NON-unified arches
  (``gemma4`` + ``gemma4_assistant``), the ones served by
  ``load_gemma4_text``. NOT ``gemma4_unified``.
- ``is_gemma4_unified_model`` matches ONLY exact ``"gemma4_unified"``.
- ``is_gemma4_family_model`` is the OR of all three accepted outer types;
  ``gemma4_assistant`` is ACCEPTED (kept on the non-unified path for
  backward compat), while genuinely-unknown siblings (``gemma4_videogen``,
  the inner ``gemma4_text``, etc.) are REJECTED — the substring-match trap.
- ``is_gemma4_model`` is the family-wide back-compat alias — it delegates
  to ``is_gemma4_family_model``, preserving the broad meaning the name
  carried pre-#509 (True for all three) while no longer swallowing a
  non-family arch that merely contains the text ``"gemma4"``.
- ``gemma4_family_kind`` classifies with a single config read.
- Each loader resolves to the matching mlx-vlm subpackage when installed,
  and falls back to the vendored copy when mlx-vlm is absent (the
  0.10.0 fresh-install regression that must never come back).

The old substring implementation FAILS the ``gemma4_unified`` /
unknown-sibling discrimination assertions; the fixed implementation PASSES.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vllm_mlx.models import gemma4_text


def _write_config(tmp_path: Path, model_type: str) -> Path:
    """Write a minimal local model dir carrying only ``model_type``.

    The detectors read the TOP-LEVEL ``model_type`` from config.json, so a
    one-key config is enough to exercise the routing decision without any
    weight download or forward pass.
    """
    d = tmp_path / model_type
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return d


# --------------------------------------------------------------------------
# Detection: exact-match, no substring bleed
# --------------------------------------------------------------------------


def test_gemma4_unified_detected_only_by_unified_detector(tmp_path):
    """A ``gemma4_unified`` model (the gemma-4-12b aliases) is detected by
    ``is_gemma4_unified_model`` and NOT by the NARROWED
    ``is_gemma4_nonunified_model``. The family-wide back-compat
    ``is_gemma4_model`` still claims it."""
    d = _write_config(tmp_path, "gemma4_unified")
    assert gemma4_text.is_gemma4_unified_model(d) is True
    assert gemma4_text.is_gemma4_nonunified_model(d) is False
    assert gemma4_text.is_gemma4_family_model(d) is True
    assert gemma4_text.is_gemma4_model(d) is True  # family-wide alias
    assert gemma4_text.gemma4_family_kind(d) == "unified"


def test_gemma4_nonunified_detected_only_by_base_detector(tmp_path):
    """A non-unified ``gemma4`` model (gemma-4-31b etc.) is detected by
    ``is_gemma4_nonunified_model`` and NOT by ``is_gemma4_unified_model``."""
    d = _write_config(tmp_path, "gemma4")
    assert gemma4_text.is_gemma4_nonunified_model(d) is True
    assert gemma4_text.is_gemma4_unified_model(d) is False
    assert gemma4_text.is_gemma4_family_model(d) is True
    assert gemma4_text.is_gemma4_model(d) is True
    assert gemma4_text.gemma4_family_kind(d) == "nonunified"


def test_is_gemma4_model_is_family_wide_alias(tmp_path):
    """``is_gemma4_model`` is the back-compat family-wide predicate: it
    tracks ``is_gemma4_family_model`` for every model_type (True for
    gemma4 / gemma4_unified / gemma4_assistant, False otherwise) and,
    unlike the old substring test, does NOT claim a non-family arch that
    merely contains the text ``"gemma4"``."""
    for mt in ("gemma4", "gemma4_assistant", "gemma4_unified", "qwen3_moe"):
        d = _write_config(tmp_path, mt)
        assert gemma4_text.is_gemma4_model(d) is gemma4_text.is_gemma4_family_model(d)
    # The #509 fix: a hypothetical sibling is no longer swallowed.
    assert gemma4_text.is_gemma4_model(_write_config(tmp_path, "gemma4_videogen")) is (
        False
    )


def test_gemma4_assistant_routes_to_nonunified(tmp_path):
    """``gemma4_assistant`` (the ``gemma-4-*-assistant`` aliases) is a
    first-class NON-unified member: the old substring match sent it down
    the ``gemma4`` text fallback and its nested ``text_config`` is a
    ``gemma4_text`` shape, so we keep it on that path to avoid regressing
    those aliases. It must be claimed by ``is_gemma4_model`` +
    ``is_gemma4_family_model`` but NOT by the unified detector."""
    d = _write_config(tmp_path, "gemma4_assistant")
    assert gemma4_text.is_gemma4_model(d) is True
    assert gemma4_text.is_gemma4_unified_model(d) is False
    assert gemma4_text.is_gemma4_family_model(d) is True
    assert gemma4_text.gemma4_family_kind(d) == "nonunified"


@pytest.mark.parametrize("mt", ["gemma4_videogen", "gemma4_text", "gemma4_foo"])
def test_gemma4_unknown_siblings_not_misrouted(tmp_path, mt):
    """Unknown sibling model_types that the old ``"gemma4" in model_type``
    substring match would have swallowed must NOT be claimed by any Gemma
    4 text detector. A hypothetical future ``gemma4_videogen`` (or the
    inner sub-config's own ``gemma4_text`` label) routed through the text
    loader would be a silent misroute. These assertions fail against the
    old substring impl."""
    d = _write_config(tmp_path, mt)
    assert gemma4_text.is_gemma4_model(d) is False
    assert gemma4_text.is_gemma4_unified_model(d) is False
    assert gemma4_text.is_gemma4_family_model(d) is False
    assert gemma4_text.gemma4_family_kind(d) is None


def test_non_gemma_model_rejected(tmp_path):
    """A completely unrelated arch is rejected by all detectors."""
    d = _write_config(tmp_path, "qwen3_moe")
    assert gemma4_text.is_gemma4_model(d) is False
    assert gemma4_text.is_gemma4_unified_model(d) is False
    assert gemma4_text.is_gemma4_family_model(d) is False


def test_unreadable_config_is_not_gemma(tmp_path):
    """A directory with no config.json (and not a resolvable repo id)
    yields ``None`` model_type → not Gemma 4."""
    d = tmp_path / "empty"
    d.mkdir()
    assert gemma4_text.is_gemma4_model(d) is False
    assert gemma4_text.is_gemma4_unified_model(d) is False
    assert gemma4_text.is_gemma4_family_model(d) is False


# --------------------------------------------------------------------------
# Loader class resolution: pin to the matching subpackage
# --------------------------------------------------------------------------


def test_nonunified_resolves_to_gemma4_subpackage():
    """``load_gemma4_text`` resolves TextConfig + LanguageModel from the
    non-unified ``mlx_vlm.models.gemma4`` subpackage when mlx-vlm is
    installed."""
    pytest.importorskip("mlx_vlm.models.gemma4")
    tc, lm = gemma4_text._resolve_gemma4_text_classes()
    assert tc.__module__ == "mlx_vlm.models.gemma4.config"
    assert lm.__module__ == "mlx_vlm.models.gemma4.language"


def test_unified_resolves_to_gemma4_unified_subpackage():
    """``load_gemma4_unified_text`` resolves TextConfig from the
    ``mlx_vlm.models.gemma4_unified`` subpackage (the matching one) when
    mlx-vlm is installed.

    Note: upstream deliberately re-exports the SAME ``LanguageModel`` from
    ``gemma4.language`` inside ``gemma4_unified`` (the unified arch only
    wraps vision/audio embedders around the identical text stack), so we
    only assert the CONFIG module pin here — that's the drift-surfacing
    signal. The LanguageModel object identity is asserted separately."""
    pytest.importorskip("mlx_vlm.models.gemma4_unified")
    tc, lm = gemma4_text._resolve_gemma4_unified_text_classes()
    assert tc.__module__ == "mlx_vlm.models.gemma4_unified.config"


def test_unified_and_base_share_language_model_class():
    """Sanity: upstream's ``gemma4_unified`` LanguageModel IS the
    ``gemma4`` one (re-export). This is WHY serving gemma-4-12b through
    the non-unified classes worked empirically before this fix — and why
    the vendored fallback (which has no unified variant) is correct."""
    pytest.importorskip("mlx_vlm.models.gemma4_unified")
    _, lm_base = gemma4_text._resolve_gemma4_text_classes()
    _, lm_unified = gemma4_text._resolve_gemma4_unified_text_classes()
    assert lm_base is lm_unified


# --------------------------------------------------------------------------
# Fresh-install (no mlx-vlm) anti-regression: vendored fallback
# --------------------------------------------------------------------------


def _block_mlx_vlm(monkeypatch):
    """Make every ``import mlx_vlm*`` raise ImportError, simulating a fresh
    ``pip install rapid-mlx`` without the ``[vision]`` extra."""
    for mod_name in list(sys.modules):
        if mod_name == "mlx_vlm" or mod_name.startswith("mlx_vlm."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocking_import)


def test_unified_resolver_falls_back_to_vendored_without_mlx_vlm(monkeypatch):
    """The #1 anti-regression: the unified loader must resolve to the
    VENDORED text classes when mlx-vlm is absent, NOT raise ImportError.
    Breaking this reintroduces the 0.10.0 fresh-install regression where
    ``rapid-mlx serve gemma-4-12b`` died on a missing mlx-vlm."""
    _block_mlx_vlm(monkeypatch)
    tc, lm = gemma4_text._resolve_gemma4_unified_text_classes()
    assert tc.__module__ == "vllm_mlx.models.gemma4_vendored.config"
    assert lm.__module__ == "vllm_mlx.models.gemma4_vendored.language"


def test_base_resolver_falls_back_to_vendored_without_mlx_vlm(monkeypatch):
    """Non-unified loader also falls back to vendored classes without
    mlx-vlm (existing 0.10.1 contract, re-pinned)."""
    _block_mlx_vlm(monkeypatch)
    tc, lm = gemma4_text._resolve_gemma4_text_classes()
    assert tc.__module__ == "vllm_mlx.models.gemma4_vendored.config"
    assert lm.__module__ == "vllm_mlx.models.gemma4_vendored.language"


def test_unified_loader_reaches_weight_check_without_mlx_vlm(tmp_path, monkeypatch):
    """End-to-end-ish: with mlx-vlm blocked, ``load_gemma4_unified_text``
    gets past class construction (via the vendored fallback) and reaches
    the ``No .safetensors files`` check — proving the vendored path is
    actually exercised, mirroring ``test_gemma4_text_import_guard`` for
    the unified loader."""
    cfg = {
        "model_type": "gemma4_unified",
        "text_config": {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "sliding_window_pattern": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_double_wide_mlp": False,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    _block_mlx_vlm(monkeypatch)

    with pytest.raises(FileNotFoundError, match="No .safetensors files"):
        gemma4_text.load_gemma4_unified_text(tmp_path, None)


# --------------------------------------------------------------------------
# Dispatch wiring: tokenizer.py routes each arch to the matching loader
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,expected_loader,other_loader",
    [
        ("gemma4_unified", "load_gemma4_unified_text", "load_gemma4_text"),
        ("gemma4", "load_gemma4_text", "load_gemma4_unified_text"),
        ("gemma4_assistant", "load_gemma4_text", "load_gemma4_unified_text"),
    ],
)
def test_dispatch_routes_to_matching_loader(
    tmp_path, monkeypatch, model_type, expected_loader, other_loader
):
    """``_load_model_with_fallback_impl`` (the tokenizer dispatch) must send
    ``gemma4_unified`` to ``load_gemma4_unified_text`` and non-unified
    ``gemma4`` / ``gemma4_assistant`` to ``load_gemma4_text``.

    We force the "native mlx-lm load failed" branch (so the explicit
    loaders run) by making the ``load`` symbol the dispatch calls raise,
    and stub both loaders to record which one fired. This exercises the
    real routing decision in ``vllm_mlx/utils/tokenizer.py`` without any
    weight download.

    ``_load_model_with_fallback_impl`` rebinds ``load`` locally via
    ``from mlx_lm import load`` on every call, so patching ``mlx_lm.load``
    is what the dispatch actually resolves — but we assert the forced
    exception fired (``native_load_raised``) so the test can never pass by
    coincidence via a native load that happened to succeed or a different
    rejection path.
    """
    from vllm_mlx.utils import tokenizer as tok

    d = _write_config(tmp_path, model_type)

    native_load_calls: list[str] = []

    def _boom(model_name, *a, **k):
        native_load_calls.append(model_name)
        raise RuntimeError("native load unavailable (forced for test)")

    # The dispatch does ``from mlx_lm import load`` at call time, so the
    # authoritative binding to patch is ``mlx_lm.load``.
    monkeypatch.setattr("mlx_lm.load", _boom)
    # Neutralize side-effects the dispatch may attempt before the gemma gate.
    monkeypatch.setattr(tok, "_needs_tokenizer_fallback", lambda *_: False)
    monkeypatch.setattr(tok, "_is_vendored_arch_model", lambda *_: False)
    monkeypatch.setattr(tok, "_register_vendored_archs", lambda *_: None)

    called: dict[str, object] = {}

    def make_stub(name):
        def _stub(model_name, tokenizer_config=None):
            called["loader"] = name
            called["model_name"] = model_name
            return ("MODEL", "TOKENIZER")

        return _stub

    monkeypatch.setattr(gemma4_text, expected_loader, make_stub(expected_loader))
    monkeypatch.setattr(gemma4_text, other_loader, make_stub(other_loader))

    result = tok._load_model_with_fallback_impl(str(d), {})

    # The forced native-load failure must actually have fired — otherwise
    # the routing assertion below could pass via a native path we didn't
    # intend to test.
    assert native_load_calls, (
        "native mlx_lm.load was never called — the dispatch didn't reach "
        "the gemma4 fallback branch, so this test isn't exercising routing"
    )
    assert result == ("MODEL", "TOKENIZER")
    assert called.get("loader") == expected_loader, (
        f"{model_type} should route to {expected_loader}, got {called.get('loader')}"
    )


# --------------------------------------------------------------------------
# Wrapper reports the routed arch, not the generic inner "gemma4_text"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "routed,inner,expected",
    [
        # Shared text stack reports the generic inner label → normalize to
        # the routed arch (the real bug codex flagged: default never used).
        ("gemma4_unified", "gemma4_text", "gemma4_unified"),
        ("gemma4", "gemma4_text", "gemma4"),
        ("gemma4_assistant", "gemma4_text", "gemma4_assistant"),
        # A more-specific inner label (if a future model reports one) wins.
        ("gemma4", "gemma4_special", "gemma4_special"),
        # Missing inner attribute → routed arch.
        ("gemma4_unified", None, "gemma4_unified"),
    ],
)
def test_wrapper_reports_routed_model_type(routed, inner, expected):
    """``Gemma4TextWrapper.model_type`` must reflect the arch the caller
    routed for. Before the fix, ``getattr(lm, "model_type", default)``
    always returned the wrapped stack's generic ``"gemma4_text"`` and the
    routed default was dead code, so a unified wrapper reported the wrong
    arch."""

    class _FakeLM:
        def __init__(self):
            self.config = object()
            self.model = object()
            if inner is not None:
                self.model_type = inner

    wrapper = gemma4_text.Gemma4TextWrapper(_FakeLM(), routed_model_type=routed)
    assert wrapper.model_type == expected
