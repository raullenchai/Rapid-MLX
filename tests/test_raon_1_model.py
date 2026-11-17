# SPDX-License-Identifier: Apache-2.0
"""Raon's catalog, trust boundary, root payload and hybrid cache contract."""

import json
import sys
import types
from fnmatch import fnmatchcase
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rapid_mlx.model_aliases import (
    resolve_model,
    resolve_profile,
    subfolder_allow_patterns,
)
from rapid_mlx.model_auto_config import detect_model_config

FIXTURES = Path(__file__).parent / "fixtures/raon_1"
REPO = "KRAFTON/Raon-1-nano-mlx"


def test_catalog_and_direct_repo_defaults():
    assert resolve_model("raon-1-nano") == REPO
    for name in ("raon-1-nano", REPO):
        p = resolve_profile(name)
        assert p.model_type == "raon1_nano" and p.is_hybrid and p.is_hybrid_explicit
        assert p.is_text_only and not p.is_moe
        assert not p.supports_spec_decode and not p.mtp_default_enabled
        assert p.tool_call_parser == p.reasoning_parser == "raon_1"
        assert dict(p.recommended_sampling) == {
            "temperature": 0.1,
            "presence_penalty": 1.5,
        }
        assert detect_model_config(name) == p


def test_renamed_local_checkpoint_is_hybrid(tmp_path):
    (tmp_path / "config.json").write_bytes((FIXTURES / "config.json").read_bytes())
    p = detect_model_config(str(tmp_path))
    assert p.is_hybrid and p.is_hybrid_explicit and not p.supports_spec_decode
    assert p.tool_call_parser == p.reasoning_parser == "raon_1"


@pytest.mark.parametrize("name", ["raon-1-nano", REPO])
def test_payload_is_root_data_only(name):
    selection = subfolder_allow_patterns(name)
    for file in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
    ]:
        assert any(fnmatchcase(file, pat) for pat in selection), file
    for file in [
        "mtp/model.safetensors",
        "mtp/config.json",
        "raon_mlx_model.py",
        "raon_mtp_mlx.py",
        "extra_tool_parser.py",
    ]:
        assert not any(fnmatchcase(file, pat) for pat in selection), file


def test_preflight_recognizes_vendored_architecture():
    from rapid_mlx.byom.preflight import RAPID_MODEL_TYPES

    assert "raon1_nano" in RAPID_MODEL_TYPES


@pytest.fixture
def registered_stub(monkeypatch):
    from rapid_mlx import models
    from rapid_mlx.utils import tokenizer

    module = types.ModuleType("rapid_mlx.models.raon1_nano")
    module.Model = object
    monkeypatch.setattr(models, "raon1_nano", module, raising=False)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.raon1_nano", None)
    monkeypatch.setattr(
        tokenizer, "_VENDORED_MODEL_TYPES", set(tokenizer._VENDORED_MODEL_TYPES)
    )
    tokenizer._register_vendored_archs()
    return tokenizer, module


def test_registration_and_fail_closed_config_overlay(
    tmp_path, registered_stub, monkeypatch
):
    loader, module = registered_stub
    assert sys.modules["mlx_lm.models.raon1_nano"] is module
    assert "raon1_nano" in loader._VENDORED_MODEL_TYPES
    config = {
        "model_type": "raon1_nano",
        "model_file": "../../untrusted.py",
        "auto_map": {"AutoModel": "untrusted.Model"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert loader._rapid_owned_model_config(str(tmp_path)) == {
        "model_file": None,
        "auto_map": None,
    }
    loader._VENDORED_MODEL_TYPES.discard("raon1_nano")
    with pytest.raises(RuntimeError, match="refusing to execute"):
        loader._rapid_owned_model_config(str(tmp_path))


@pytest.mark.parametrize("pinned", [False, True])
def test_root_selection_reaches_resolver(monkeypatch, tmp_path, pinned):
    from rapid_mlx import model_downloads
    from rapid_mlx.utils.tokenizer import _resolve_subfolder_checkpoint

    seen = []

    def download(repo_id, *args, **kwargs):
        seen.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(model_downloads, "snapshot_download", download)
    if pinned:
        from rapid_mlx import _download_gate, _mirror, model_metadata

        monkeypatch.setattr(_mirror, "pinned_snapshot_download", download)
        monkeypatch.setattr(
            model_metadata, "resolve_pinned_cached_snapshot", lambda *a, **k: None
        )
        monkeypatch.setattr(_download_gate, "_snapshot_is_complete", lambda *_: True)
    else:
        monkeypatch.setattr("rapid_mlx.model_aliases.resolve_revision", lambda *_: None)
    assert Path(_resolve_subfolder_checkpoint("raon-1-nano")) == tmp_path
    assert seen == [(REPO, {"allow_patterns": subfolder_allow_patterns(REPO)})]


def test_public_lazy_loader_never_enables_remote_code(
    tmp_path, monkeypatch, registered_stub
):
    loader, _ = registered_stub
    config = json.loads((FIXTURES / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    # This file is deliberately present and must never execute.
    (tmp_path / "raon_mlx_model.py").write_text(
        "raise AssertionError('repo code executed')"
    )
    seen = []
    model, tokenizer = MagicMock(), MagicMock()

    def load(path, **kwargs):
        seen.append(kwargs)
        return model, tokenizer

    fake = types.ModuleType("mlx_lm")
    fake.load = load
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setattr(loader, "_try_inject_mtp_post_load", lambda *_: None)
    monkeypatch.setattr(
        loader, "augment_eos_token_ids_from_generation_config", lambda *_: None
    )
    monkeypatch.setattr(loader, "repair_byte_level_decoder", lambda *_: None)
    monkeypatch.setattr(loader, "_post_load_ubc_evict", lambda *_: None)
    assert loader.load_model_with_fallback(str(tmp_path), lazy=True) == (
        model,
        tokenizer,
    )
    assert seen[0]["model_config"] == {"model_file": None, "auto_map": None}
    assert seen[0]["tokenizer_config"]["trust_remote_code"] is False


def _tiny_model():
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.models import raon1_nano

    config = json.loads((FIXTURES / "config.json").read_text())
    config.update(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=16,
        vocab_size=64,
        intermediate_size=64,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        rda_decay_rank=16,
        per_layer_embedding_dim=8,
        raon_variants="off",
        rda_config={
            "num_heads": 2,
            "head_dim": 16,
            "conv_kernel_size": 4,
            "full_rank_gate": True,
            "gate_lower_bound": -5.0,
            "rda_layers": [1],
            "mla_layers": [2],
        },
    )
    model = raon1_nano.Model(raon1_nano.ModelArgs.from_dict(config))
    model.eval()
    return mx, model


def test_real_registration_selects_vendored_classes():
    pytest.importorskip("mlx.core")
    from mlx_lm.utils import _get_classes

    from rapid_mlx.models import raon1_nano
    from rapid_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    cls, args = _get_classes({"model_type": "raon1_nano"})
    assert cls is raon1_nano.Model and args is raon1_nano.ModelArgs


def test_tiny_forward_chunking_and_cache_reuse():
    mx, model = _tiny_model()
    from mlx_lm.models.cache import ArraysCache, KVCache, can_trim_prompt_cache

    from rapid_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    ids = [1, 2, 3, 4, 5, 6]
    tokens = mx.array([ids])
    full = model(tokens)
    cache = model.make_cache()
    assert isinstance(cache[0], ArraysCache) and isinstance(cache[1], KVCache)
    assert not can_trim_prompt_cache(cache)
    parts = [model(tokens[:, i : i + 1], cache=cache) for i in range(len(ids))]
    joined = mx.concatenate(parts, axis=1)
    mx.eval(full, joined)
    assert mx.allclose(full, joined, atol=2e-3, rtol=2e-3).item()
    prefix = model.make_cache()
    mx.eval(model(tokens[:, :3], cache=prefix))
    default = MemoryAwarePrefixCache(model, MemoryCacheConfig(max_memory_mb=64))
    assert not default.store(ids[:3], prefix)
    store = MemoryAwarePrefixCache(
        model, MemoryCacheConfig(max_memory_mb=64, hybrid_reuse_max_entries=2)
    )
    assert store.store(ids[:3], prefix)
    warm_cache, rest = store.fetch(ids)
    assert rest == ids[3:]
    warm = model(mx.array([rest]), cache=warm_cache)
    mx.eval(warm)
    assert mx.allclose(warm, full[:, 3:], atol=2e-3, rtol=2e-3).item()
    # A stored recurrent state cannot be trimmed backward to a shorter prefix.
    missing, remaining = store.fetch(ids[:2])
    assert missing is None and remaining == ids[:2]


def test_folded_rda_does_not_run_for_batch_two(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.models import raon1_nano as raon

    # Exercise the optimized branch with representative dimensions and a
    # nonempty state, without needing real model weights.
    cfg = json.loads((FIXTURES / "config.json").read_text())
    cfg.update(
        num_hidden_layers=2,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        per_layer_embedding_dim=8,
        rda_config={**cfg["rda_config"], "rda_layers": [1], "mla_layers": [2]},
    )
    model = raon.Model(raon.ModelArgs.from_dict(cfg))
    model.eval()
    raon.install(model, "rda_fold")
    rda = model.layers[0].self_attn
    monkeypatch.setattr(rda, "_fused_decode", lambda *_: True)

    def forbidden(*_):
        raise AssertionError("batch entered singleton folded kernel")

    monkeypatch.setattr(raon, "_rda_fold_prep", forbidden)
    monkeypatch.setattr(raon, "_RDA_ON", {"rda_fold"})
    # Quantization enables the merged projection whose shape exposed the bug.
    raon.nn.quantize(rda, bits=4, group_size=32)
    cache = model.make_cache()[0]
    # Seed conv state so the singleton kernel would otherwise be eligible.
    cache.cache[0] = mx.zeros((2, 3, rda.proj_size * 3))
    x = mx.zeros((2, 1, 32), dtype=mx.bfloat16)
    out = rda(x, cache=cache)
    mx.eval(out)
    assert out.shape == (2, 1, 32)


def test_public_eager_loader_uses_registered_model(tmp_path):
    mx, source = _tiny_model()
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from rapid_mlx.models import raon1_nano
    from rapid_mlx.utils.tokenizer import load_model_with_fallback

    config = dict(vars(source.args))
    config.update(
        model_file="raon_mlx_model.py", auto_map={"AutoModel": "untrusted.Model"}
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "raon_mlx_model.py").write_text(
        "raise AssertionError('checkpoint Python executed')"
    )
    mx.eval(source.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(source.parameters()))
    )
    Tokenizer(WordLevel({"<unk>": 0, "hi": 1, "<eos>": 2})).save(
        str(tmp_path / "tokenizer.json")
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<eos>", "unk_token": "<unk>"})
    )
    (tmp_path / "chat_template.jinja").write_bytes(
        (FIXTURES / "chat_template.jinja").read_bytes()
    )
    loaded, tokenizer = load_model_with_fallback(str(tmp_path))
    assert isinstance(loaded, raon1_nano.FusedModel)
    tokens = mx.array([[1, 2]])
    assert mx.allclose(source(tokens), loaded(tokens), atol=1e-5).item()
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    ).endswith("<think>\n")


def test_generation_config_registers_both_stop_ids(tmp_path):
    from rapid_mlx.utils.tokenizer import (
        RAPID_EXTRA_EOS_ATTR,
        augment_eos_token_ids_from_generation_config,
    )

    config = {"eos_token_id": [259202, 259200]}
    (tmp_path / "generation_config.json").write_text(json.dumps(config))
    tokenizer = types.SimpleNamespace(eos_token_id=259202)
    augment_eos_token_ids_from_generation_config(tokenizer, str(tmp_path))
    assert set(getattr(tokenizer, RAPID_EXTRA_EOS_ATTR)) == {259202, 259200}
