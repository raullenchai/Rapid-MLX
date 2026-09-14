# SPDX-License-Identifier: Apache-2.0
"""Contracts for Rapid's mlx-vlm-independent K2 Horizon text adapter."""

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from vllm_mlx.models import k2_horizon  # noqa: E402

TINY = {
    "model_type": "k2_horizon",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 64,
    "max_position_embeddings": 128,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0},
    "tie_word_embeddings": False,
    "layernorm_num_groups": 4,
}


@pytest.fixture(autouse=True)
def _clear_registration():
    sys.modules.pop("mlx_lm.models.k2_horizon", None)
    yield
    sys.modules.pop("mlx_lm.models.k2_horizon", None)


def test_released_config_contract_and_rope_translation():
    args = k2_horizon.ModelArgs.from_dict({"model_type": "k2_horizon"})
    assert (args.hidden_size, args.num_hidden_layers, args.intermediate_size) == (
        4096,
        36,
        12288,
    )
    assert (args.num_attention_heads, args.num_key_value_heads, args.head_dim) == (
        32,
        8,
        128,
    )
    assert args.vocab_size == 250624
    assert args.max_position_embeddings == 524288
    assert args.rope_theta == 10_000_000.0
    assert args.rope_scaling is None
    assert args.tie_word_embeddings is False
    assert args.layernorm_num_groups == 4

    tiny = k2_horizon.ModelArgs.from_dict(TINY)
    assert tiny.rope_theta == 1_000_000.0
    assert tiny.rope_scaling is None


def test_yarn_rope_metadata_is_preserved_without_nested_theta():
    args = k2_horizon.ModelArgs.from_dict(
        {
            **TINY,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 2_000_000.0,
                "factor": 8.0,
                "original_max_position_embeddings": 8192,
            },
        }
    )
    assert args.rope_theta == 2_000_000.0
    assert args.rope_scaling == {
        "rope_type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
    }


@pytest.mark.parametrize(
    "override,match",
    [
        ({"num_hidden_layers": 0}, "num_hidden_layers"),
        ({"num_key_value_heads": 3}, "multiple"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"layernorm_num_groups": 3}, "divisible"),
        ({"num_experts": 8}, "MoE/MoVA"),
        ({"num_shared_experts": 1}, "MoE/MoVA"),
        ({"moe_intermediate_size": 128}, "MoE/MoVA"),
        ({"query_key_norm": True}, "query/key"),
        ({"attention_gate_func": "silu"}, "gated attention"),
        ({"rope_head_dim": 4}, "partial rotary"),
        ({"use_sliding_window": True}, "sliding"),
        ({"rope_parameters": []}, "rope_parameters"),
    ],
)
def test_unsupported_checkpoint_geometry_fails_closed(override, match):
    with pytest.raises(ValueError, match=match):
        k2_horizon.ModelArgs.from_dict({**TINY, **override})


def test_tiny_forward_and_incremental_cache_match_full_prefill():
    mx.random.seed(7)
    args = k2_horizon.ModelArgs.from_dict(TINY)
    model = k2_horizon.Model(args)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4]])
    full = model(tokens)
    caches = model.make_cache()
    pieces = []
    for token in (1, 2, 3, 4):
        pieces.append(model(mx.array([[token]]), cache=caches))
    mx.eval(full, *pieces)

    assert full.shape == (1, 4, 64)
    assert all(piece.shape == (1, 1, 64) for piece in pieces)
    assert mx.allclose(
        full[:, -1, :], pieces[-1][:, -1, :], rtol=1e-4, atol=1e-4
    ).item()


def test_group_rms_norm_matches_grouped_reference_not_global_rms():
    norm = k2_horizon.GroupRMSNorm(dims=4, groups=2, eps=1e-6)
    value = mx.array([[1.0, 1.0, 10.0, 10.0]])
    actual = norm(value)
    grouped = value.reshape(1, 2, 2)
    expected = grouped * mx.rsqrt(
        mx.mean(grouped * grouped, axis=-1, keepdims=True) + 1e-6
    )
    global_expected = value * mx.rsqrt(
        mx.mean(value * value, axis=-1, keepdims=True) + 1e-6
    )
    mx.eval(actual, expected, global_expected)
    assert mx.allclose(
        actual, expected.reshape(value.shape), rtol=1e-5, atol=1e-5
    ).item()
    assert not mx.allclose(actual, global_expected, rtol=1e-3, atol=1e-3).item()


def _reset_registration(monkeypatch):
    from vllm_mlx.utils import tokenizer

    monkeypatch.delitem(sys.modules, "mlx_lm.models.k2_horizon", raising=False)
    monkeypatch.setattr(
        tokenizer, "_VENDORED_MODEL_TYPES", set(tokenizer._VENDORED_MODEL_TYPES)
    )
    tokenizer._VENDORED_MODEL_TYPES.discard("k2_horizon")
    return tokenizer


def test_registration_uses_rapid_adapter_and_is_idempotent(monkeypatch):
    tokenizer = _reset_registration(monkeypatch)
    tokenizer._register_vendored_archs()
    registered = importlib.import_module("mlx_lm.models.k2_horizon")
    assert registered is k2_horizon
    assert "k2_horizon" in tokenizer._VENDORED_MODEL_TYPES
    tokenizer._register_vendored_archs()
    assert importlib.import_module("mlx_lm.models.k2_horizon") is registered


def test_registration_defers_to_future_native_module(monkeypatch):
    tokenizer = _reset_registration(monkeypatch)
    native = types.ModuleType("mlx_lm.models.k2_horizon")
    native.__spec__ = importlib.util.spec_from_loader(
        "mlx_lm.models.k2_horizon", loader=None
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.models.k2_horizon", native)
    tokenizer._register_vendored_archs()
    assert sys.modules["mlx_lm.models.k2_horizon"] is native
    assert "k2_horizon" in tokenizer._VENDORED_MODEL_TYPES


def test_adapter_has_no_mlx_vlm_import():
    source = open(k2_horizon.__file__, encoding="utf-8").read()
    assert "import mlx_vlm" not in source
    assert "from mlx_vlm" not in source


def test_repo_code_trust_boundary_is_scoped_to_rapid_owned_k2(tmp_path):
    from vllm_mlx.utils import tokenizer

    k2_dir = tmp_path / "k2"
    k2_dir.mkdir()
    (k2_dir / "config.json").write_text(json.dumps(TINY))
    other_dir = tmp_path / "existing-vendored-family"
    other_dir.mkdir()
    (other_dir / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))

    assert tokenizer._uses_rapid_owned_runtime(str(k2_dir)) is True
    assert tokenizer._uses_rapid_owned_runtime(str(other_dir)) is False
    assert tokenizer._uses_rapid_owned_runtime("org/not-cached") is False


def test_vendored_load_ignores_checkpoint_owned_model_code(tmp_path, monkeypatch):
    """K2 weights must execute Rapid's reviewed runtime, not repo Python."""
    from vllm_mlx.utils import tokenizer

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **TINY,
                "model_file": "model.py",
                "auto_map": {"AutoModel": "model.CustomModel"},
            }
        )
    )
    (tmp_path / "model.py").write_text("raise AssertionError('must not execute')\n")
    (tmp_path / "tokenizer.json").write_text("{}")

    captured = {}
    fake_model = object()

    def fake_load_model(path: Path, *, model_config=None, **_kwargs):
        captured.update(model_config or {})
        return fake_model, {}

    fake_tokenizer = MagicMock()
    fake_tokenizer.chat_template = "template"
    monkeypatch.setattr("mlx_lm.utils.load_model", fake_load_model)
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda _path: MagicMock())
    monkeypatch.setattr(
        "transformers.PreTrainedTokenizerFast", lambda **_kwargs: fake_tokenizer
    )
    monkeypatch.setattr(
        tokenizer, "augment_eos_token_ids_from_generation_config", lambda *_: None
    )
    monkeypatch.setattr(tokenizer, "repair_byte_level_decoder", lambda *_: None)

    tokenizer._register_vendored_archs()
    model, returned_tokenizer = tokenizer._load_with_tokenizer_fallback(str(tmp_path))
    assert model is fake_model
    assert returned_tokenizer is fake_tokenizer
    assert captured["model_file"] is None
    assert captured["auto_map"] is None
