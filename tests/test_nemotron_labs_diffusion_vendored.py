# SPDX-License-Identifier: Apache-2.0
"""Focused contracts for the vendored Nemotron-Labs-Diffusion AR decoder.

``model_type: nemotron_labs_diffusion`` (NVIDIA 3B/8B/14B) is a Ministral3-style
decoder + a separate, untied ``diffusion_head`` LM projection. mlx-lm (0.31.3,
2026-08-21) ships no native support for the arch, so rapid-mlx vendors an AR-mode
port under ``vllm_mlx.models.nemotron_labs_diffusion`` and registers it as
``sys.modules["mlx_lm.models.nemotron_labs_diffusion"]`` so mlx-lm's loader
resolves the checkpoint transparently.

These tests pin that contract (mirrors ``test_gpt_oss_puzzle_vendored.py``):
  - registration makes the arch visible to mlx-lm's importlib lookup
  - the tokenizer-fallback loader routes the arch to the vendored module
  - the AR forward pass is a correct causal LM (logits shape, untied head)
  - ``sanitize`` normalizes the checkpoint's ``language_model.*`` prefix and
    drops rotary/base shims, so the standard mlx-lm weight-match + quantize
    path reproduces the affine 4-bit layout.
"""

import importlib
import json
import sys

import pytest


@pytest.fixture(autouse=True)
def _clear_nld_vendor_registration():
    """Keep process-global mlx-lm registration isolated across tests."""
    from vllm_mlx.utils.tokenizer import _VENDORED_MODEL_TYPES

    sys.modules.pop("mlx_lm.models.nemotron_labs_diffusion", None)
    _VENDORED_MODEL_TYPES.discard("nemotron_labs_diffusion")
    yield
    sys.modules.pop("mlx_lm.models.nemotron_labs_diffusion", None)
    _VENDORED_MODEL_TYPES.discard("nemotron_labs_diffusion")


def test_register_vendored_arch_makes_diffusion_visible_to_mlx_lm():
    from vllm_mlx.utils.tokenizer import (
        _VENDORED_MODEL_TYPES,
        _register_vendored_archs,
    )

    _register_vendored_archs()

    module = importlib.import_module("mlx_lm.models.nemotron_labs_diffusion")
    assert module.__name__ == "vllm_mlx.models.nemotron_labs_diffusion"
    assert "nemotron_labs_diffusion" in _VENDORED_MODEL_TYPES
    assert hasattr(module, "Model")
    assert hasattr(module, "ModelArgs")


def test_vendored_arch_classifier_selects_low_level_loader(tmp_path):
    from vllm_mlx.utils.tokenizer import (
        _is_vendored_arch_model,
        _register_vendored_archs,
    )

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "nemotron_labs_diffusion"})
    )
    _register_vendored_archs()

    assert _is_vendored_arch_model(str(tmp_path))


def test_loader_routes_diffusion_config_to_vendored_path(tmp_path, monkeypatch):
    from vllm_mlx.utils import tokenizer

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "nemotron_labs_diffusion"})
    )
    expected = (object(), object())
    seen = []

    def _fake_vendor_loader(model_name, *, enable_dspark=False):
        # ``enable_dspark`` mirrors the real ``_load_with_tokenizer_fallback``
        # signature (#1379); the vendored-arch route must forward it.
        seen.append(model_name)
        return expected

    monkeypatch.setattr(tokenizer, "_load_with_tokenizer_fallback", _fake_vendor_loader)

    assert tokenizer._load_model_with_fallback_impl(str(tmp_path)) is expected
    assert seen == [str(tmp_path)]


def _tiny_model_args(**overrides):
    from vllm_mlx.models.nemotron_labs_diffusion import ModelArgs

    kwargs = dict(
        model_type="nemotron_labs_diffusion",
        rms_norm_eps=1e-5,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1e6,
            "factor": 16.0,
            "original_max_position_embeddings": 16,
            "llama_4_scaling_beta": 0.1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


def test_ar_forward_is_causal_lm_with_untied_head():
    import mlx.core as mx
    import mlx.nn as nn

    import vllm_mlx.models.nemotron_labs_diffusion as nld

    model = nld.Model(_tiny_model_args())
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(logits)

    # Standard causal LLM output shape: (batch, seq, vocab) with an UNTIED
    # diffusion_head projection, not a tied embedding.
    assert logits.shape == (1, 3, 32)
    assert isinstance(model.diffusion_head, nn.Linear)
    # Head is separate from the embedding table (tie_word_embeddings=False).
    assert model.diffusion_head.weight.shape == (32, 16)
    assert model.model.embed_tokens.weight.shape == (32, 16)
    # Backbone is the Ministral3-style stack with YaRN RoPE.
    assert len(model.layers) == 2
    assert all(not layer.use_sliding for layer in model.layers)
    # Making the causal mask: second token sees the first (logits are finite, not NaN).
    assert bool(mx.isfinite(logits).all())


def test_ar_cache_is_plain_kvcache_not_rotating():
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    import vllm_mlx.models.nemotron_labs_diffusion as nld

    model = nld.Model(_tiny_model_args())
    assert isinstance(model.make_cache()[0], KVCache)
    # No sliding layers in this arch by default, so nothing may RotatingKVCache.
    assert not isinstance(model.make_cache()[0], RotatingKVCache)

    # Prefill then decode extends the cache: the AR lane is a plain autoreg.
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(model.layers[0].self_attn.v_proj.weight)
    model(mx.array([[4]], dtype=mx.int32), cache=cache)
    mx.eval(cache[0].keys)


def test_sanitize_strips_language_model_prefix_and_drops_shims():
    import mlx.core as mx

    import vllm_mlx.models.nemotron_labs_diffusion as nld

    model = nld.Model(_tiny_model_args())
    kept = mx.array([1.0])
    sanitized = model.sanitize(
        {
            # Real checkpoint spelling: everything under language_model.*
            "language_model.model.embed_tokens.weight": kept,
            "language_model.model.layers.0.self_attn.q_proj.weight": kept,
            "language_model.diffusion_head.weight": kept,
            # Safety-net shims that a repack might carry — must be dropped or
            # remapped, never left as stray keys that break weight matching.
            "language_model.model.embed_tokens.rotary_emb.inv_freq": mx.array([2.0]),
            "encoder.model.norm.weight": mx.array([3.0]),
            "language_model.encoder.layers.0.input_layernorm.weight": mx.array([4.0]),
        }
    )

    # ``language_model.`` prefix stripped onto this module's structure.
    assert "model.embed_tokens.weight" in sanitized
    assert "model.layers.0.self_attn.q_proj.weight" in sanitized
    assert "diffusion_head.weight" in sanitized
    # rotary-inv_freq dropped; encoder.-style base shims remapped. The Swift
    # reference prefixes base weights ``encoder.``, so a repack normalized
    # onto it must still collapse to ``model.*``.
    assert "embed_tokens.rotary_emb.inv_freq" not in sanitized
    assert "model.norm.weight" in sanitized
    assert "encoder.model.norm.weight" not in sanitized
    assert "model.layers.0.input_layernorm.weight" in sanitized
    assert "encoder.layers.0.input_layernorm.weight" not in sanitized
    # Values survive intact.
    assert mx.array_equal(sanitized["model.embed_tokens.weight"], kept)
    assert mx.array_equal(sanitized["diffusion_head.weight"], kept)


def test_model_args_default_kv_heads_to_full_attention():
    args = _tiny_model_args(num_key_value_heads=None)
    # num_key_value_heads is None → mirrors num_attention_heads (GQA off),
    # matching the checkpoint's 32→8 layout when populated.
    args.__post_init__()
    assert args.num_key_value_heads == 2
    assert args.layer_types == ["full_attention", "full_attention"]


def test_alias_resolves_to_ar_text_lane():
    """The 3B-4bit alias must route to the standard text AR engine.

    Nemotron-Labs-Diffusion shares one decoder across AR / block-diffusion /
    linear-self-spec modes; rapid-mlx's P0 vendors AR only, so the alias must
    NOT be misclassified as hybrid/MoE (which would flip the scheduler to a
    diffusion lane we don't serve) and must carry ``modality == "text"``.
    """
    from vllm_mlx.model_aliases import resolve_profile
    from vllm_mlx.model_auto_config import detect_model_config

    prof = resolve_profile("nemotron-labs-diffusion-3b-4bit")
    assert prof.hf_path == "mlx-community/Nemotron-Labs-Diffusion-3B-4bit"
    assert prof.modality == "text"  # AR text lane, not a vision/diffusion lane.
    assert prof.is_hybrid is False
    assert prof.is_moe is False
    # Pure attention stack → self-spec decode is allowed (opt-in only, never
    # auto-enabled), consistent with other dense attention aliases.
    assert prof.supports_spec_decode is True

    # detect_model_config resolves the alias BEFORE regex heuristics, so the
    # user-facing ``--model nemotron-labs-diffusion-3b-4bit`` hits the text
    # engine without needing the model_type regex to recognize the arch.
    cfg = detect_model_config("nemotron-labs-diffusion-3b-4bit")
    assert cfg.modality == "text"
