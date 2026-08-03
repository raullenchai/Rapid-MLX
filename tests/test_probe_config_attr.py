# SPDX-License-Identifier: Apache-2.0
"""``_text_attention_args`` must read ``.config``, not only ``.args``.

mlx-lm models expose their config as ``.args``; the mlx-vlm-derived
architectures expose it as ``.config``. Gemma 4's ``Gemma4TextWrapper``
has no ``.args`` at all and its ``.language_model`` carries a
``TextConfig`` on ``.config``, so probing only ``.args`` returned
``(None, None)`` and the engine logged

    [kv-cache] live KV quantization disabled: head_dim (K=None, V=None)
    unknown or not divisible by any supported group_size (32/64/128)

for every ``gemma-4-12b-*`` alias — despite ``head_dim=256`` /
``global_head_dim=512`` being divisible by all three. The live cache
silently stayed bf16, which is expensive on an architecture whose KV runs
~128 KB/token.

These tests pin both halves of the contract: ``.config`` is now consulted,
and ``.args`` still wins wherever it exists so no other model moves.
"""

from types import SimpleNamespace

from vllm_mlx.quantized_batch_cache import (
    _text_attention_args,
    probe_kv_head_dims,
    resolve_kv_quantization,
)


def _cfg(head_dim, **extra):
    return SimpleNamespace(head_dim=head_dim, **extra)


# ── the regression: language model carries .config, not .args ──────────


def test_language_model_config_attr_is_probeable():
    """Shape of Gemma4TextWrapper: no top-level .args, language_model.config
    holds the text config. Pre-fix this probed to (None, None)."""
    model = SimpleNamespace(
        language_model=SimpleNamespace(config=_cfg(256)),
    )
    assert probe_kv_head_dims(model) == (256, 256)


def test_gemma4_dims_enable_the_live_quantized_cache():
    """MUTATION-KILL: the point of the probe fix. head_dim 256 divides every
    supported group size, so the live cache must NOT be disabled."""
    k, v = probe_kv_head_dims(
        SimpleNamespace(language_model=SimpleNamespace(config=_cfg(256)))
    )
    for requested in (32, 64, 128):
        group_size, live_disabled = resolve_kv_quantization(k, v, requested)
        assert live_disabled is False
        assert group_size == requested


def test_top_level_config_text_config_is_probeable():
    """A wrapper whose nested text config hangs off ``.config.text_config``
    rather than ``.args.text_config``."""
    model = SimpleNamespace(config=SimpleNamespace(text_config=_cfg(128)))
    assert probe_kv_head_dims(model) == (128, 128)


# ── the guard: .args keeps priority, fail-safe preserved ───────────────


def test_args_wins_over_config_when_both_probeable():
    """Every mlx-lm model has ``.args``; it must keep winning so this change
    moves no existing model. A divergent ``.config`` must be ignored."""
    lm = SimpleNamespace(args=_cfg(64), config=_cfg(256))
    assert _text_attention_args(SimpleNamespace(language_model=lm)) is lm.args
    assert probe_kv_head_dims(SimpleNamespace(language_model=lm)) == (64, 64)


def test_text_only_model_with_args_unchanged():
    """Neither multimodal signal present: the top-level args ARE the language
    config, exactly as before."""
    model = SimpleNamespace(args=_cfg(128, v_head_dim=64))
    assert probe_kv_head_dims(model) == (128, 64)


def test_unprobeable_language_model_still_fails_safe():
    """A multimodal wrapper whose language config exposes no readable head dim
    must still yield None so the live cache falls back to bf16 rather than
    picking up vision/composite dims from the top level."""
    model = SimpleNamespace(
        args=SimpleNamespace(head_dim=999),  # vision/composite — must NOT leak
        language_model=SimpleNamespace(config=SimpleNamespace()),
    )
    assert probe_kv_head_dims(model) == (None, None)


def test_hidden_size_fallback_still_works_via_config():
    """``.config`` goes through the same head_dim resolution as ``.args``,
    including the hidden_size // num_attention_heads fallback."""
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            config=SimpleNamespace(hidden_size=3840, num_attention_heads=30)
        )
    )
    assert probe_kv_head_dims(model) == (128, 128)
