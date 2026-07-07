# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #477 — MTP injection on VLM/hybrid models.

Two surfaces are pinned here:

1. ``patches.qwen3_next_mtp._looks_like_vlm_wrapper`` — VLM checkpoints
   nest the LLM config under ``text_config`` and expose the inner LLM as
   ``model.language_model``. The outer ``model.args`` lacks LLM fields.
   ``inject_mtp_support`` must bail out cleanly in this shape rather than
   patch the outer class and crash on the next forward (codex round-1
   P1 on #477 — wrapper methods reference ``self.model.embed_tokens`` /
   ``self.lm_head`` that don't exist on the outer VLM).

The old ``scheduler._install_mtp`` BatchGenerator monkey-patch has been
removed. Runtime MTP now goes through the common speculative-config path
and ``_install_mtp_vendored``.
"""

from __future__ import annotations

from types import SimpleNamespace

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
