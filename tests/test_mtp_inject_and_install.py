# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #477 — MTP injection + install on VLM/hybrid models.

Two surfaces are pinned here:

1. ``patches.qwen3_next_mtp._looks_like_vlm_wrapper`` — VLM checkpoints
   nest the LLM config under ``text_config`` and expose the inner LLM as
   ``model.language_model``. The outer ``model.args`` lacks LLM fields.
   ``inject_mtp_support`` must bail out cleanly in this shape rather than
   patch the outer class and crash on the next forward (codex round-1
   P1 on #477 — wrapper methods reference ``self.model.embed_tokens`` /
   ``self.lm_head`` that don't exist on the outer VLM).

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
# _looks_like_vlm_wrapper — gate for VLM detection in inject_mtp_support
# ----------------------------------------------------------------------


def _llm_args_ns(**overrides):
    """SimpleNamespace shaped like a populated ``ModelArgs``."""
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


def test_looks_like_vlm_wrapper_false_for_text_only_model():
    """Text-only path: ``model.args.hidden_size`` exists → not a VLM,
    inject_mtp_support proceeds normally."""
    from vllm_mlx.patches.qwen3_next_mtp import _looks_like_vlm_wrapper

    model = SimpleNamespace(args=_llm_args_ns(hidden_size=4096))
    assert _looks_like_vlm_wrapper(model) is False


def test_looks_like_vlm_wrapper_true_for_vlm_with_language_model():
    """VLM checkpoint: outer args lacks hidden_size AND
    model.language_model is present → bail out so we don't deferred-crash
    on the next forward. Pins codex round-1 P1 on issue #477."""
    from vllm_mlx.patches.qwen3_next_mtp import _looks_like_vlm_wrapper

    vlm_outer = SimpleNamespace(
        text_config={"hidden_size": 3584},
        vision_config={"hidden_size": 1280},
    )
    model = SimpleNamespace(
        args=vlm_outer,
        language_model=SimpleNamespace(args=_llm_args_ns(hidden_size=3584)),
    )
    assert _looks_like_vlm_wrapper(model) is True


def test_looks_like_vlm_wrapper_false_when_language_model_is_none():
    """Defensive — ``language_model`` attr exists but is None (e.g.
    text-only branch of a multimodal class). Not a usable VLM; let the
    "no fallback available" warning path fire instead of pretending it's
    a VLM wrapper."""
    from vllm_mlx.patches.qwen3_next_mtp import _looks_like_vlm_wrapper

    model = SimpleNamespace(args=SimpleNamespace(), language_model=None)
    assert _looks_like_vlm_wrapper(model) is False


def test_looks_like_vlm_wrapper_false_when_args_already_has_hidden_size():
    """Even if ``language_model`` is somehow attached to a text-only
    model, the populated ``model.args.hidden_size`` short-circuits the
    check (text-only path always wins)."""
    from vllm_mlx.patches.qwen3_next_mtp import _looks_like_vlm_wrapper

    model = SimpleNamespace(
        args=_llm_args_ns(hidden_size=2048),
        language_model=SimpleNamespace(args=_llm_args_ns(hidden_size=1024)),
    )
    assert _looks_like_vlm_wrapper(model) is False


# ----------------------------------------------------------------------
# _install_mtp guard for hybrid BatchGenerator (issue #477 Issue 2)
# ----------------------------------------------------------------------


def test_install_mtp_skips_when_batch_gen_lacks_step(caplog):
    """Hybrid (Gated-DeltaNet) BatchGenerator routes through its own step
    flow and has no ``_step`` attribute. Before this fix, _install_mtp
    crashed at ``_orig_step = batch_gen._step`` with AttributeError.

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
    assert "_step" in combined
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
    orig_step = MagicMock(name="orig_step")
    bg._step = orig_step
    bg.active_batch = None
    model = SimpleNamespace(mtp=SimpleNamespace())

    result = _install_mtp(bg, model=model, num_draft_tokens=1)

    assert result is True
    # _step is now the wrapper closure, NOT the original — a misconfigured
    # patch that left _step unchanged would still pass a callable check.
    assert bg._step is not orig_step, "_step was not actually re-bound by _install_mtp"
    assert callable(bg._step)
    # _next is also re-bound (the other half of the MTP install).
    assert callable(bg._next)


@pytest.mark.parametrize("optimistic_flag", [True, False])
def test_install_mtp_hybrid_guard_independent_of_optimistic(caplog, optimistic_flag):
    """The hybrid guard must trigger regardless of the ``optimistic``
    flag (and by extension regardless of how ``--force-spec-decode``
    routes through the scheduler). The in-function guard is a safety-net
    for any call site that doesn't pre-check ``supports_spec_decode``."""
    from vllm_mlx.scheduler import _install_mtp

    class _HybridGen:
        pass

    bg = _HybridGen()
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        result = _install_mtp(
            bg,
            model=SimpleNamespace(),
            num_draft_tokens=2,
            optimistic=optimistic_flag,
        )
    assert result is False
    assert not hasattr(bg, "_step")
