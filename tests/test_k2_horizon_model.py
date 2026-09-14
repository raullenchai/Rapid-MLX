# SPDX-License-Identifier: Apache-2.0
"""Contracts for Rapid's mlx-vlm-independent K2 Horizon text adapter."""

import importlib
import json
import sys
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
        (
            {
                "hidden_size": 36,
                "num_attention_heads": 5,
                "num_key_value_heads": 5,
                "head_dim": None,
            },
            "divisible by num_attention_heads",
        ),
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
        (
            {"rope_parameters": {"rope_type": "default", "factor": 2.0}},
            "scaling metadata",
        ),
        (
            {"rope_parameters": {"rope_type": "default", "type": "yarn"}},
            "conflicting rope types",
        ),
    ],
)
def test_unsupported_checkpoint_geometry_fails_closed(override, match):
    with pytest.raises(ValueError, match=match):
        k2_horizon.ModelArgs.from_dict({**TINY, **override})


def test_derived_attention_defaults_are_explicit_and_validated():
    args = k2_horizon.ModelArgs.from_dict(
        {**TINY, "num_key_value_heads": None, "head_dim": None}
    )
    assert args.num_key_value_heads == args.num_attention_heads
    assert args.head_dim == TINY["hidden_size"] // TINY["num_attention_heads"]

    with pytest.raises(ValueError, match="num_key_value_heads must be positive"):
        k2_horizon.ModelArgs.from_dict({**TINY, "num_key_value_heads": 0})
    with pytest.raises(ValueError, match="head_dim must be positive"):
        k2_horizon.ModelArgs.from_dict({**TINY, "head_dim": 0})


def test_group_rms_norm_rejects_invalid_group_geometry():
    with pytest.raises(ValueError, match="must be divisible"):
        k2_horizon.GroupRMSNorm(dims=4, groups=3, eps=1e-6)


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


def test_tied_embeddings_supply_logits_and_drop_duplicate_lm_head():
    args = k2_horizon.ModelArgs.from_dict({**TINY, "tie_word_embeddings": True})
    model = k2_horizon.Model(args)
    logits = model(mx.array([[1, 2]]))
    mx.eval(logits)
    assert logits.shape == (1, 2, TINY["vocab_size"])
    assert "lm_head.weight" not in model.sanitize(
        {"lm_head.weight": mx.ones((1,)), "model.weight": mx.ones((1,))}
    )


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
    native_spec = importlib.util.spec_from_loader(
        "mlx_lm.models.k2_horizon", loader=None
    )
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "mlx_lm.models.k2_horizon":
            return native_spec
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    tokenizer._register_vendored_archs()
    assert "mlx_lm.models.k2_horizon" not in sys.modules
    assert "k2_horizon" in tokenizer._VENDORED_MODEL_TYPES


def test_registration_recovers_when_native_spec_probe_is_invalid(monkeypatch):
    tokenizer = _reset_registration(monkeypatch)
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "mlx_lm.models.k2_horizon":
            raise ValueError("module has no spec")
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    tokenizer._register_vendored_archs()
    assert importlib.import_module("mlx_lm.models.k2_horizon") is k2_horizon


def test_registration_failure_does_not_advertise_k2(monkeypatch, caplog):
    tokenizer = _reset_registration(monkeypatch)
    real_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "vllm_mlx.models.k2_horizon":
            raise ImportError("broken adapter")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    tokenizer._register_vendored_archs()
    assert "k2_horizon" not in tokenizer._VENDORED_MODEL_TYPES
    assert "failed to register" in caplog.text


def test_runtime_probe_failure_fails_closed(tmp_path, monkeypatch):
    from vllm_mlx.utils import tokenizer

    (tmp_path / "config.json").write_text(json.dumps(TINY))
    tokenizer._register_vendored_archs()
    monkeypatch.delitem(sys.modules, "mlx_lm.models.k2_horizon", raising=False)

    def invalid_spec(_name):
        raise ValueError("module has no spec")

    monkeypatch.setattr(importlib.util, "find_spec", invalid_spec)
    with pytest.raises(RuntimeError, match="refusing to execute checkpoint-owned"):
        tokenizer._uses_rapid_owned_runtime(str(tmp_path))


def test_adapter_has_no_mlx_vlm_import():
    source = Path(k2_horizon.__file__).read_text(encoding="utf-8")
    assert "import mlx_vlm" not in source
    assert "from mlx_vlm" not in source


def test_cache_shape_must_match_layer_count():
    model = k2_horizon.Model(k2_horizon.ModelArgs.from_dict(TINY))
    with pytest.raises(ValueError, match="expected 2 K2 cache entries, got 1"):
        model(mx.array([[1]]), cache=[None])


def test_repo_code_trust_boundary_is_scoped_to_rapid_owned_k2(tmp_path, monkeypatch):
    from vllm_mlx.utils import tokenizer

    k2_dir = tmp_path / "k2"
    k2_dir.mkdir()
    (k2_dir / "config.json").write_text(json.dumps(TINY))
    other_dir = tmp_path / "existing-vendored-family"
    other_dir.mkdir()
    (other_dir / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    scalar_dir = tmp_path / "invalid-scalar-config"
    scalar_dir.mkdir()
    (scalar_dir / "config.json").write_text(json.dumps(["k2_horizon"]))

    tokenizer._register_vendored_archs()
    assert tokenizer._uses_rapid_owned_runtime(str(k2_dir)) is True
    assert tokenizer._uses_rapid_owned_runtime(str(other_dir)) is False
    assert tokenizer._uses_rapid_owned_runtime(str(scalar_dir)) is False

    monkeypatch.setattr(
        tokenizer,
        "_read_model_config_json",
        lambda model_name: TINY if model_name == "org/not-cached" else None,
    )
    assert tokenizer._uses_rapid_owned_runtime("org/not-cached") is True


def test_public_loader_fails_closed_when_k2_registration_is_unavailable(
    tmp_path, monkeypatch
):
    """An untrusted model_type declaration cannot grant a validation bypass."""
    from vllm_mlx.utils import tokenizer

    model_root = tmp_path / "untrusted-k2"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps(
            {
                **TINY,
                "model_file": "../outside.py",
                "auto_map": {"AutoModel": "model.CustomModel"},
            }
        )
    )
    (tmp_path / "outside.py").write_text("raise AssertionError('must not run')\n")

    monkeypatch.setattr(
        tokenizer, "_VENDORED_MODEL_TYPES", set(tokenizer._VENDORED_MODEL_TYPES)
    )
    tokenizer._VENDORED_MODEL_TYPES.discard("k2_horizon")
    monkeypatch.delitem(sys.modules, "mlx_lm.models.k2_horizon", raising=False)
    monkeypatch.setattr(tokenizer, "_register_vendored_archs", lambda: None)
    loader = MagicMock(side_effect=AssertionError("loader must not run"))
    monkeypatch.setattr(tokenizer, "_load_model_with_fallback_impl", loader)

    with pytest.raises(RuntimeError, match="refusing to execute checkpoint-owned"):
        tokenizer.load_model_with_fallback(str(model_root))
    loader.assert_not_called()


def test_public_eager_loader_ignores_checkpoint_owned_model_code(tmp_path, monkeypatch):
    """The real eager loader must execute Rapid's runtime, not repo Python."""
    from mlx.utils import tree_flatten

    from vllm_mlx.utils import tokenizer

    checkpoint_config = {
        **TINY,
        "model_file": "model.py",
        "auto_map": {"AutoModel": "model.CustomModel"},
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))
    (tmp_path / "model.py").write_text("raise AssertionError('must not execute')\n")
    (tmp_path / "tokenizer.json").write_text("{}")

    source_model = k2_horizon.Model(k2_horizon.ModelArgs.from_dict(checkpoint_config))
    mx.eval(source_model.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        dict(tree_flatten(source_model.parameters())),
    )

    fake_tokenizer = MagicMock()
    fake_tokenizer.chat_template = "template"
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda _path: MagicMock())
    monkeypatch.setattr(
        "transformers.PreTrainedTokenizerFast", lambda **_kwargs: fake_tokenizer
    )
    monkeypatch.setattr(
        tokenizer, "augment_eos_token_ids_from_generation_config", lambda *_: None
    )
    monkeypatch.setattr(tokenizer, "repair_byte_level_decoder", lambda *_: None)

    model, returned_tokenizer = tokenizer.load_model_with_fallback(str(tmp_path))
    assert isinstance(model, k2_horizon.Model)
    assert returned_tokenizer is fake_tokenizer


@pytest.mark.parametrize("remote", [False, True], ids=["local", "first-remote-load"])
def test_public_lazy_loader_ignores_checkpoint_owned_model_code(
    tmp_path, monkeypatch, remote
):
    """The primary mlx-lm entry point receives the same safe config overlay."""
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
    (tmp_path / "model.py").write_text("raise AssertionError('must not run')\n")

    requested = "publisher/k2-horizon" if remote else str(tmp_path)
    resolution_events = []
    if remote:

        def cold_cache(name):
            resolution_events.append(("cache-miss", name))
            return name

        def download_snapshot(name):
            resolution_events.append(("download", name))
            return tmp_path if name == requested else Path(name)

        monkeypatch.setattr(tokenizer, "_local_snapshot_if_cached", cold_cache)
        monkeypatch.setattr(
            tokenizer,
            "_resolve_model_path",
            download_snapshot,
        )

    captured = {}
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_tokenizer.chat_template = "template"

    def fake_load(_path, *, model_config=None, **_kwargs):
        captured.update(model_config or {})
        if (
            captured.get("model_file") is not None
            or captured.get("auto_map") is not None
        ):
            raise AssertionError("checkpoint-owned code was not suppressed")
        return fake_model, fake_tokenizer

    monkeypatch.setattr("mlx_lm.load", fake_load)
    monkeypatch.setattr(tokenizer, "_try_inject_mtp_post_load", lambda *_: None)
    monkeypatch.setattr(
        tokenizer, "augment_eos_token_ids_from_generation_config", lambda *_: None
    )
    monkeypatch.setattr(tokenizer, "repair_byte_level_decoder", lambda *_: None)
    monkeypatch.setattr(tokenizer, "_post_load_ubc_evict", lambda *_: None)

    model, returned_tokenizer = tokenizer.load_model_with_fallback(requested, lazy=True)
    assert model is fake_model
    assert returned_tokenizer is fake_tokenizer
    assert captured == {"model_file": None, "auto_map": None}
    if remote:
        assert resolution_events[:2] == [
            ("cache-miss", requested),
            ("download", requested),
        ]
