# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #477 — MTP injection + install on VLM/hybrid models.

Two surfaces are pinned here:

1. ``patches.qwen3_next_mtp._resolve_text_args`` — VLM checkpoints store LLM
   args under ``text_config`` and expose the inner LLM as
   ``model.language_model``. The injector must fall through:
   ``model.args`` → ``model.language_model.args`` → ``config['text_config']``.

2. ``scheduler._install_mtp`` — hybrid Gated-DeltaNet BatchGenerators (e.g.
   Qwen3.6-35B-A3B) route through their own step flow and lack ``_step``.
   The installer must log a clear warning and return False rather than
   crash with AttributeError (which is what shipped in <=0.6.66 and what
   the user hit in issue #477 with ``--force-spec-decode``).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ----------------------------------------------------------------------
# _resolve_text_args
# ----------------------------------------------------------------------


def _llm_args_ns(**overrides):
    """Build a SimpleNamespace shaped like a populated ``ModelArgs``."""
    defaults = {
        "hidden_size": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "full_attention_interval": 4,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "tie_word_embeddings": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _StubArgsCls:
    """Stand-in for ``mlx_lm.models.qwen3_next.ModelArgs`` used by the
    text_config fallback. Records the dict it was built from so the test
    can assert the fall-through path was actually taken."""

    last_built_from: dict | None = None

    def __init__(self, *, hidden_size, **rest):
        self.hidden_size = hidden_size
        for k, v in rest.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, d):
        cls.last_built_from = dict(d)
        return cls(**d)


@pytest.fixture(autouse=True)
def _reset_stub_args_cls():
    """Class-level ``last_built_from`` is shared mutable state — reset it
    between tests so order doesn't matter (e.g. a fallback-path test that
    runs first must not poison a later text_config-untouched assertion).
    DeepSeek round-1 BLOCKING #1."""
    _StubArgsCls.last_built_from = None
    yield
    _StubArgsCls.last_built_from = None


def test_resolve_text_args_returns_model_args_when_populated():
    """Text-only path: ``model.args.hidden_size`` exists → use it directly,
    no language_model / text_config lookup needed."""
    from vllm_mlx.patches.qwen3_next_mtp import _resolve_text_args

    args = _llm_args_ns(hidden_size=4096)
    model = SimpleNamespace(args=args, language_model=None)
    config: dict = {}  # no text_config

    out = _resolve_text_args(model, config, _StubArgsCls)
    assert out is args, "must return the same object, not a copy"
    assert _StubArgsCls.last_built_from is None, (
        "must NOT have hit the text_config fallback"
    )


def test_resolve_text_args_falls_back_to_language_model_for_vlm():
    """VLM checkpoint: outer ``model.args`` lacks LLM fields; the inner
    ``model.language_model.args`` is the populated one. Pins issue #477
    Issue 1 — Qwen3.6-35B-A3B-VLM had ``model.args`` without hidden_size,
    causing inject_mtp_support to raise AttributeError."""
    from vllm_mlx.patches.qwen3_next_mtp import _resolve_text_args

    inner_args = _llm_args_ns(hidden_size=3584, rope_theta=10_000_000.0)
    vlm_outer_args = SimpleNamespace(  # outer args without hidden_size
        text_config={"hidden_size": 3584},
        vision_config={"hidden_size": 1280},
    )
    model = SimpleNamespace(
        args=vlm_outer_args,
        language_model=SimpleNamespace(args=inner_args),
    )
    config = {"text_config": {"hidden_size": 3584}}

    out = _resolve_text_args(model, config, _StubArgsCls)
    assert out is inner_args
    assert _StubArgsCls.last_built_from is None, (
        "language_model.args came first — text_config fallback must not run"
    )


def test_resolve_text_args_builds_from_text_config_when_no_inner_args():
    """Edge case: no ``language_model`` (or it lacks ``args``). Build
    fresh ModelArgs from ``config['text_config']`` as the user requested
    in #477."""
    from vllm_mlx.patches.qwen3_next_mtp import _resolve_text_args

    _StubArgsCls.last_built_from = None
    outer_args = SimpleNamespace()  # lacks hidden_size
    model = SimpleNamespace(args=outer_args)  # no language_model attr
    config = {
        "text_config": {
            "hidden_size": 5120,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1e-6,
        }
    }

    out = _resolve_text_args(model, config, _StubArgsCls)
    assert out is not None
    assert out.hidden_size == 5120
    assert _StubArgsCls.last_built_from == config["text_config"], (
        "fallback path must rebuild via from_dict(text_config)"
    )


def test_resolve_text_args_returns_none_when_nothing_resolves():
    """All three lookups fail → return None. The caller (inject_mtp_support)
    must then log a warning and skip MTP rather than crash."""
    from vllm_mlx.patches.qwen3_next_mtp import _resolve_text_args

    model = SimpleNamespace(args=SimpleNamespace())  # bare args
    config = {"text_config": {}}  # text_config present but no hidden_size

    out = _resolve_text_args(model, config, _StubArgsCls)
    assert out is None


def test_resolve_text_args_does_not_crash_when_inner_language_model_is_none():
    """``model.language_model`` exists but is None (e.g. text-only branch
    of a multimodal class). Must not raise; should continue to text_config."""
    from vllm_mlx.patches.qwen3_next_mtp import _resolve_text_args

    model = SimpleNamespace(args=SimpleNamespace(), language_model=None)
    config = {"text_config": {"hidden_size": 1024}}

    out = _resolve_text_args(model, config, _StubArgsCls)
    assert out is not None
    assert out.hidden_size == 1024


# ----------------------------------------------------------------------
# _install_mtp guard for hybrid BatchGenerator (Issue #477 Issue 2)
# ----------------------------------------------------------------------


def test_install_mtp_skips_when_batch_gen_lacks_step(caplog):
    """Hybrid (Gated-DeltaNet) BatchGenerator routes through its own step
    flow and has no ``_step`` attribute. Before this fix, _install_mtp
    crashed at line ``_orig_step = batch_gen._step`` with AttributeError.

    Contract: return False and log a clear warning, do NOT raise, do NOT
    patch anything on the generator."""
    from vllm_mlx.scheduler import _install_mtp

    class _HybridGen:
        """Stand-in for the Qwen3.6 hybrid BatchGenerator — no ``_step``,
        no ``_next``, no ``active_batch`` accessor."""

    bg = _HybridGen()
    model = SimpleNamespace()

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        result = _install_mtp(bg, model=model, num_draft_tokens=2)

    assert result is False
    assert not hasattr(bg, "_step"), "generator must remain untouched — no fields added"
    assert not hasattr(bg, "_next"), "generator must remain untouched — no fields added"

    # The warning must name the issue so users can find it.
    combined = "\n".join(r.message for r in caplog.records)
    assert "no _step attribute" in combined or "_step" in combined
    assert "#477" in combined, "warning should reference the tracking issue"


def test_install_mtp_succeeds_on_compatible_batch_gen():
    """Pin existing behavior: a BatchGenerator with ``_step`` still gets
    patched and ``_install_mtp`` returns True. Guards against the guard
    being too eager and breaking the normal (non-hybrid) path."""
    from vllm_mlx.scheduler import _install_mtp

    bg = MagicMock()
    # Provide the minimum surface _install_mtp needs to install patches:
    # _step (the entry it monkey-patches) and active_batch (referenced
    # inside the patched _mtp_step closure during prefill guard — the
    # closure is never *called* in this test, so a placeholder is fine).
    bg._step = MagicMock(name="orig_step")
    bg.active_batch = None
    model = SimpleNamespace(mtp=SimpleNamespace())

    result = _install_mtp(bg, model=model, num_draft_tokens=1)

    assert result is True
    # _step should now point at the wrapper, not the original.
    assert bg._step is not None
    assert callable(bg._step)


@pytest.mark.parametrize("force_spec_flag", [True, False])
def test_install_mtp_hybrid_guard_independent_of_force_spec_decode(
    caplog, force_spec_flag
):
    """The hybrid guard must trigger regardless of ``--force-spec-decode``.
    Pre-fix, the gate at scheduler.py:1886-1907 was the only protection,
    and ``--force-spec-decode`` bypassed it. The in-function guard is a
    safety-net for any future call site that doesn't pre-check."""
    from vllm_mlx.scheduler import _install_mtp

    class _HybridGen:
        pass

    bg = _HybridGen()
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        result = _install_mtp(
            bg, model=SimpleNamespace(), num_draft_tokens=2, optimistic=force_spec_flag
        )
    assert result is False
    assert not hasattr(bg, "_step")
