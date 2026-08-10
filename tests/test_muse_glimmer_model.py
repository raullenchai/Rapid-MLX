# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored Muse Glimmer text backbone.

Pins the contract that:

1. The vendored module is importable and registers into mlx-lm's
   importlib lookup (``_register_vendored_archs``).
2. Config parsing derives the released layer pattern ([sliding ×3, full],
   NoPE on full-attention layers) and validates explicit lists.
3. A tiny synthetic config constructs + runs the model: logits shape,
   final-logit softcap bound, and incremental (cache) decode matching the
   full-prefill forward.
4. ``sanitize`` strips the ``language_model.`` wrapper and drops the
   vision stack (text-only serving).
5. ``resolve_serving_lane`` auto-downgrades a muse_glimmer checkpoint to
   the text lane while the installed mlx-vlm lacks the arch.
6. The curated aliases pin text-only serving and the muse parsers.

Numeric fidelity against the transformers reference implementation was
verified out-of-band on identical random weights (8 layers, seq 24 >
sliding_window 8, full-prefill and token-by-token): max abs hidden-state
diff 1.4e-5. The reference dump requires transformers 5.15+ (muse_glimmer)
which CI does not install, so that check is not repeated here.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

from vllm_mlx.models import muse_glimmer as mg  # noqa: E402

TINY_TEXT = dict(
    hidden_size=64,
    num_hidden_layers=8,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=128,
    sliding_window=8,
    max_position_embeddings=256,
)


def tiny_model():
    args = mg.ModelArgs.from_dict(
        {"model_type": "muse_glimmer", "text_config": dict(TINY_TEXT)}
    )
    return mg.Model(args)


@pytest.fixture(autouse=True)
def _clear_vendored_register():
    """Registration is sys.modules-level state — reset around each test."""
    sys.modules.pop("mlx_lm.models.muse_glimmer", None)
    yield
    sys.modules.pop("mlx_lm.models.muse_glimmer", None)


def test_module_contract():
    assert hasattr(mg, "Model")
    assert hasattr(mg, "ModelArgs")
    assert mg.ModelArgs.__dataclass_fields__["model_type"].default == "muse_glimmer"


def test_register_vendored_archs_makes_mlx_lm_loader_find_it():
    from vllm_mlx.utils.tokenizer import (
        _VENDORED_MODEL_TYPES,
        _register_vendored_archs,
    )

    assert "mlx_lm.models.muse_glimmer" not in sys.modules
    _register_vendored_archs()
    assert "mlx_lm.models.muse_glimmer" in sys.modules
    assert "muse_glimmer" in _VENDORED_MODEL_TYPES

    # mlx-lm's _get_classes() does exactly this lookup.
    mod = importlib.import_module("mlx_lm.models.muse_glimmer")
    assert mod is sys.modules["mlx_lm.models.muse_glimmer"]
    assert hasattr(mod, "Model") and hasattr(mod, "ModelArgs")

    # Idempotent.
    _register_vendored_archs()
    assert importlib.import_module("mlx_lm.models.muse_glimmer") is mod


def test_layer_pattern_defaults():
    args = mg.ModelArgs.from_dict(
        {"model_type": "muse_glimmer", "text_config": dict(TINY_TEXT)}
    )
    t = args.text
    assert (
        t.layer_types
        == [
            mg._SLIDING,
            mg._SLIDING,
            mg._SLIDING,
            mg._FULL,
        ]
        * 2
    )
    # Full-attention layers are NoPE; sliding layers carry the theta.
    assert [th == 0.0 for th in t.layer_rope_theta] == [
        lt == mg._FULL for lt in t.layer_types
    ]
    assert all(th in (0.0, 500000.0) for th in t.layer_rope_theta)


def test_layer_pattern_validation():
    bad = dict(TINY_TEXT, layer_types=["sliding_attention"] * 3)
    with pytest.raises(ValueError, match="layer_types"):
        mg.ModelArgs.from_dict({"model_type": "muse_glimmer", "text_config": bad})
    bad = dict(TINY_TEXT, layer_rope_theta=[0.0] * 3)
    with pytest.raises(ValueError, match="layer_rope_theta"):
        mg.ModelArgs.from_dict({"model_type": "muse_glimmer", "text_config": bad})


def test_flattened_config_fallback():
    """A flattened text-only export (LM shape at the TOP level, no nested
    text_config) must not silently collapse to the 30B defaults
    (codex r1 #1)."""
    flat = dict(TINY_TEXT, model_type="muse_glimmer")
    args = mg.ModelArgs.from_dict(flat)
    assert args.text.hidden_size == TINY_TEXT["hidden_size"]
    assert args.text.num_hidden_layers == TINY_TEXT["num_hidden_layers"]
    # The flat model builds and runs.
    model = mg.Model(args)
    out = model(mx.array([[1, 2, 3]]))
    assert out.shape == (1, 3, TINY_TEXT["vocab_size"])

    # Missing shape info entirely -> released 30B defaults.
    args = mg.ModelArgs.from_dict({"model_type": "muse_glimmer", "text_config": {}})
    assert args.text.hidden_size == 6656
    assert args.text.num_hidden_layers == 52


def test_configs_without_both_layer_kinds():
    """Explicit all-sliding configs and short default-pattern configs
    (< 4 layers -> no full-attention layer) must construct and run
    (codex r1 #2)."""
    for text_cfg in (
        dict(TINY_TEXT, num_hidden_layers=2, layer_types=[mg._SLIDING] * 2),
        dict(TINY_TEXT, num_hidden_layers=3),  # default pattern -> [S,S,S]
        dict(TINY_TEXT, num_hidden_layers=2, layer_types=[mg._FULL] * 2),
    ):
        args = mg.ModelArgs.from_dict(
            {"model_type": "muse_glimmer", "text_config": text_cfg}
        )
        model = mg.Model(args)
        ids = mx.array([[1, 2, 3, 4]])
        no_cache = model(ids)
        assert no_cache.shape == (1, 4, TINY_TEXT["vocab_size"])
        cache = model.make_cache()
        steps = [model(ids[:, i : i + 1], cache=cache) for i in range(4)]
        inc = mx.concatenate(steps, axis=1)
        assert float(mx.abs(inc - no_cache).max()) < 2e-3


def test_forward_softcap_and_cache_parity():
    model = tiny_model()
    ids = mx.random.randint(0, TINY_TEXT["vocab_size"], (1, 24))

    logits = model(ids)
    assert logits.shape == (1, 24, TINY_TEXT["vocab_size"])
    # tanh softcap bounds |logits| by final_logit_softcapping.
    assert float(mx.abs(logits).max()) <= model.text_args.final_logit_softcapping + 1e-4

    cache = model.make_cache()
    steps = [model(ids[:, i : i + 1], cache=cache) for i in range(ids.shape[1])]
    inc = mx.concatenate(steps, axis=1)
    diff = float(mx.abs(inc - logits).max())
    assert diff < 2e-3, diff


def test_input_embeddings_used_raw():
    """Provided embeddings must NOT be re-normed (transformers parity:
    the post-lookup norm lives inside the embedding lookup only)."""
    model = tiny_model()
    ids = mx.array([[1, 2, 3]])
    via_ids = model(ids)
    embeds = model.model.embed_norm(model.model.embed_tokens(ids))
    via_embeds = model(ids, input_embeddings=embeds)
    assert float(mx.abs(via_ids - via_embeds).max()) < 1e-6


def test_make_cache_kinds():
    model = tiny_model()
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    kinds = model.make_cache()
    for t, c in zip(model.text_args.layer_types, kinds):
        if t == mg._SLIDING:
            assert isinstance(c, RotatingKVCache)
            assert c.max_size == TINY_TEXT["sliding_window"]
        else:
            assert isinstance(c, KVCache)


def test_sanitize_strips_wrapper_and_vision():
    model = tiny_model()
    weights = {
        "language_model.model.embed_tokens.weight": 1,
        "language_model.model.layers.0.self_attn.q_proj.weight": 2,
        "language_model.model.norm.weight": 3,
        "language_model.lm_head.weight": 4,
        "vision_tower.layers.0.attn.q_proj.weight": 5,
        "vision_adapter.fc1.weight": 6,
        "vision_projection.weight": 7,
        "multi_modal_projector.linear.weight": 8,
    }
    out = model.sanitize(weights)
    assert sorted(out) == [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    ]
    assert model.tie_word_embeddings is False


def test_layer_types_value_validation():
    """A typo'd layer kind must be rejected, not silently treated as
    full attention with RoPE (codex r2 #2)."""
    bad = dict(
        TINY_TEXT,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "silding_attention"],
    )
    with pytest.raises(ValueError, match="unknown layer_types"):
        mg.ModelArgs.from_dict({"model_type": "muse_glimmer", "text_config": bad})


def test_config_declared_tying_wins_over_shipped_head():
    """tie_word_embeddings=true in the config must tie even when the
    export also ships lm_head weights (codex r2 #1)."""
    tied_cfg = dict(TINY_TEXT, tie_word_embeddings=True)
    args = mg.ModelArgs.from_dict(
        {"model_type": "muse_glimmer", "text_config": tied_cfg}
    )
    model = mg.Model(args)
    assert model.tie_word_embeddings is True
    assert "lm_head" not in dict(model.children())
    # Stray head weights are dropped so strict loading can't trip.
    out = model.sanitize(
        {
            "language_model.lm_head.weight": 1,
            "language_model.model.embed_tokens.weight": 2,
        }
    )
    assert sorted(out) == ["model.embed_tokens.weight"]
    # Tied model runs and produces vocab-sized logits via the table.
    logits = model(mx.array([[1, 2, 3]]))
    assert logits.shape == (1, 3, TINY_TEXT["vocab_size"])


def test_sanitize_passthrough_keeps_untied_head():
    """Bare (already-stripped) keys pass through, and an untied config
    missing head weights must NOT silently tie — strict loading should
    report the incomplete export instead (codex r5 #1)."""
    model = tiny_model()
    bare = {"model.embed_tokens.weight": 1}
    out = model.sanitize(dict(bare))
    assert "model.embed_tokens.weight" in out
    assert model.tie_word_embeddings is False
    assert "lm_head" in dict(model.children())


def test_resolve_serving_lane_muse_text_fallback(monkeypatch, tmp_path):
    """A muse_glimmer checkpoint auto-downgrades to the text lane while
    the installed mlx-vlm has no muse_glimmer model package."""
    import vllm_mlx.api.utils as api_utils

    muse_config = {
        "model_type": "muse_glimmer",
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "vision_config": {"model_type": "muse_glimmer_vision"},
        "image_token_id": 200092,
        "text_config": {"model_type": "muse_glimmer_text"},
    }

    assert api_utils.mllm_arch_unsupported_but_text_vendored is not None

    # The probe reads cached metadata; fake it.
    class _Meta:
        config = muse_config
        snapshot_dir = None
        is_local = True

    monkeypatch.setattr(api_utils, "read_model_metadata", lambda name: _Meta())
    # Engine venvs may or may not ship mlx-vlm; either way it must not
    # have a muse_glimmer package yet for this fallback to make sense.
    import importlib.util

    try:
        spec = importlib.util.find_spec("mlx_vlm.models.muse_glimmer")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        pytest.skip("installed mlx-vlm has native muse_glimmer support")

    assert api_utils.mllm_arch_unsupported_but_text_vendored("fake/muse")
    is_mllm, auto_fallback = api_utils.resolve_serving_lane("fake/muse")
    assert is_mllm is False
    assert auto_fallback is True

    # Explicit flags still win.
    assert api_utils.resolve_serving_lane("fake/muse", force_mllm=True) == (
        True,
        False,
    )

    # A non-vendored VLM arch is untouched by the probe.
    class _MetaQwen:
        config = {"model_type": "qwen3_vl"}
        snapshot_dir = None
        is_local = True

    monkeypatch.setattr(api_utils, "read_model_metadata", lambda name: _MetaQwen())
    assert not api_utils.mllm_arch_unsupported_but_text_vendored("fake/qwen")


def test_aliases_pin_text_only_and_muse_parsers():
    aliases = json.loads(
        (Path(__file__).parent.parent / "vllm_mlx" / "aliases.json").read_text()
    )
    for name in ("muse-glimmer-30b-4bit", "muse-glimmer-30b-bf16"):
        entry = aliases[name]
        assert entry["hf_path"].startswith("mlx-community/Muse-Glimmer-30B")
        # The #393 state-pin: vision weights exist but text-only serving
        # is a deliberate curated decision until mlx-vlm learns the arch.
        assert entry["is_text_only"] is True
        assert entry["tool_call_parser"] == "muse"
        assert entry["reasoning_parser"] == "muse"
        assert entry["is_hybrid"] is False
        assert entry["is_hybrid_explicit"] is True
