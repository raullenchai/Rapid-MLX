"""Hermetic Engram8 derivation: real checkpoint I/O with NumPy quantization seam."""

import errno
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import hy4_requant_engram as requant
from scripts.hy4_gptq import NativeStore
from scripts.hy4_gptq_adapter import Hy4Adapter, Recipe
from scripts.streamed_gptq_io import TensorReader, atomic_json
from tests.test_hy4_gptq import write_source
from tests.test_hy_v4_flash import cpu_model as cpu_model

MODULE = "model.layers.1.engram_layer.multi_head_embedding.embedding"
WEIGHT = MODULE + ".weight"
OTHER = "model.layers.1.engram_layer.key_projs.0.weight"


@pytest.fixture
def checkpoint(tmp_path, cpu_model, monkeypatch):
    mx = sys.modules["mlx.core"]
    mx.bfloat16 = np.float16
    mx.uint16 = np.uint16
    mx.eval = lambda *args: None
    mx.clear_cache = lambda: None
    calls = []

    def quantize(value, *, bits, group_size):
        assert bits == 8 and group_size == 32
        calls.append(value.shape)
        # Deterministic packed words and BF16 stand-ins; no Metal import needed.
        rows, width = value.shape
        packed = np.repeat(value[:, :1].astype(np.uint32), width // 4, axis=1)
        scales = np.ones((rows, width // 32), np.float16)
        return packed, scales, -scales

    monkeypatch.setattr(mx, "quantize", quantize, raising=False)
    source = write_source(
        tmp_path / "source",
        {
            WEIGHT: ("F8_E4M3", np.full((9, 64), 0x38, np.uint8)),
            WEIGHT + "_scale_inv": ("F32", np.arange(1, 10, dtype=np.float32)),
        },
    )
    base = write_source(
        tmp_path / "base",
        {
            WEIGHT: ("U32", np.zeros((9, 8), np.uint32)),
            MODULE + ".scales": ("BF16", np.zeros((9, 2), np.uint16)),
            MODULE + ".biases": ("BF16", np.zeros((9, 2), np.uint16)),
            OTHER: ("F32", np.arange(64, dtype=np.float32).reshape(1, 64)),
            "model.scalar": ("I32", np.array(19, dtype=np.int32)),
        },
    )
    config = json.loads((base / "config.json").read_text())
    config["quantization"] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
        MODULE: dict(bits=4, group_size=32, mode="affine"),
        "unrelated": dict(bits=4, group_size=64, mode="affine"),
    }
    atomic_json(base / "config.json", config)
    atomic_json(
        base / "size-report.json",
        dict(recipe=dict(name="hy4-gptq-v1"), identity="base-epoch"),
    )
    index = json.loads((base / "model.safetensors.index.json").read_text())
    # One separate untouched shard, plus a mixed Engram + non-Engram shard.
    untouched = write_source(
        tmp_path / "extra", {"model.norm.weight": ("F32", np.ones(64, np.float32))}
    )
    (base / "untouched.safetensors").write_bytes(
        (untouched / "weights.safetensors").read_bytes()
    )
    index["weight_map"]["model.norm.weight"] = "untouched.safetensors"
    index["metadata"] = dict(total_size=1, preserved="yes")
    atomic_json(base / "model.safetensors.index.json", index)
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "chat_template.jinja",
        "generation_config.json",
    ):
        (base / name).write_text("copied")
    return base, source, tmp_path / "out", calls


def test_recipe_changes_only_engram():
    names = json.loads(
        (
            Path(__file__).parent / "fixtures/hy_v4_flash/text_tensor_names.json"
        ).read_text()
    )
    recipe = Recipe(requant.RECIPE)
    for name in names:
        expected = (
            (8, 32, "rtn")
            if name.endswith(requant.ENGRAM + ".weight")
            else Recipe().spec(name)
        )
        assert recipe.spec(name) == expected
    with pytest.raises(ValueError, match="only to lowbit"):
        Recipe(requant.RECIPE, gate_up_3bit_layers=(1,))


def test_plan_and_dry_run_are_read_only(checkpoint, capsys):
    base, source, out, calls = checkpoint
    plan = requant.derive(base, source, out, dry_run=True)
    printed = json.loads(capsys.readouterr().out)
    assert printed["base_identity"] == "base-epoch"
    assert [s["action"] for s in plan["shards"]] == ["hardlink", "rewrite"]
    assert plan["shards"][1]["engram_modules"] == [MODULE]
    assert plan["estimated_bytes"] == plan["hardlink_bytes"] + plan["rewrite_bytes"]
    assert not calls and not out.exists() and not out.with_name("out.tmp").exists()


def test_derive_index_config_links_and_payloads(checkpoint):
    base, source, out, calls = checkpoint
    report = requant.derive(base, source, out)
    assert calls == [(9, 64)]
    assert os.path.samefile(
        base / "untouched.safetensors", out / "untouched.safetensors"
    )
    assert not os.path.samefile(
        base / "weights.safetensors", out / "weights.safetensors"
    )
    old, new = TensorReader(base), TensorReader(out)
    for name in (OTHER, "model.scalar", "model.norm.weight"):
        np.testing.assert_array_equal(new.raw(name), old.raw(name))
    np.testing.assert_array_equal(new.raw(WEIGHT)[:, 0], np.arange(1, 10))
    assert new.index.tensors[WEIGHT].shape == (9, 16)
    assert new.index.tensors[MODULE + ".scales"].shape == (9, 2)
    config = json.loads((out / "config.json").read_text())
    assert config["quantization"][MODULE] == dict(bits=8, group_size=32, mode="affine")
    original = json.loads((base / "config.json").read_text())
    original["quantization"][MODULE] = config["quantization"][MODULE]
    assert config == original
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert (
        index["weight_map"]
        == json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    )
    assert index["metadata"]["preserved"] == "yes"
    assert index["metadata"]["total_size"] == sum(
        t.nbytes for t in new.index.tensors.values()
    )
    assert (
        report["base_identity"] == "base-epoch"
        and report["recipe"]["name"] == requant.RECIPE
    )
    assert report["safetensors_bytes"] == sum(
        p.stat().st_size for p in out.glob("*.safetensors")
    )
    assert (out / "tokenizer.model").read_text() == "copied"
    assert (out / "chat_template.jinja").read_text() == "copied"
    assert not out.with_name("out.tmp").exists()
    with pytest.raises(ValueError, match="non-empty"):
        requant.derive(base, source, out)


def test_bounded_chunks_and_full_build_rtn(checkpoint):
    base, source, out, calls = checkpoint
    adapter = Hy4Adapter(source, Recipe(requant.RECIPE))
    plan, reader, *_ = requant.shard_plan(base, adapter)
    requant.rewrite_shard(out, plan["shards"][1], reader, adapter, chunk_rows=2)
    assert calls == [(2, 64)] * 4 + [(1, 64)]
    # The ordinary full-build RTN path also accepts this recipe.
    store = NativeStore(out.parent / "full-build", adapter, [0, 1])
    store.rtn(WEIGHT, lambda: None)
    assert store.specs[MODULE] == dict(bits=8, group_size=32, mode="affine")
    assert store.reader().index.tensors[WEIGHT].shape == (9, 16)


def test_resume_after_interruption(checkpoint, monkeypatch):
    base, source, out, calls = checkpoint
    original = requant.rewrite_shard

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(requant, "rewrite_shard", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        requant.derive(base, source, out)
    assert not out.exists()
    stage = out.with_name("out.tmp")
    inode = (stage / "untouched.safetensors").stat().st_ino
    monkeypatch.setattr(requant, "rewrite_shard", original)
    requant.derive(base, source, out)
    assert (out / "untouched.safetensors").stat().st_ino == inode
    assert calls == [(9, 64)]


def test_cross_device_link_fails_without_copy_fallback(checkpoint, monkeypatch):
    base, source, out, calls = checkpoint

    def exdev(*args):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(requant.os, "link", exdev)
    with pytest.raises(OSError) as error:
        requant.derive(base, source, out)
    assert error.value.errno == errno.EXDEV
    assert not out.exists() and not calls


def test_split_affine_components(checkpoint):
    base, source, out, calls = checkpoint
    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    split = base.parent / "split"
    write_source(
        split,
        {MODULE + ".biases": ("BF16", np.zeros((9, 2), np.uint16))},
    )
    (base / "biases.safetensors").write_bytes(
        (split / "weights.safetensors").read_bytes()
    )
    index["weight_map"][MODULE + ".biases"] = "biases.safetensors"
    atomic_json(index_path, index)
    requant.derive(base, source, out)
    reader = TensorReader(out)
    assert reader.index.tensors[MODULE + ".biases"].shape == (9, 2)
    np.testing.assert_array_equal(
        reader.raw(MODULE + ".biases"),
        np.full((9, 2), -1, np.float16).view(np.uint16),
    )
    assert len(calls) == 2


def test_resume_skips_completed_rewrite(checkpoint, monkeypatch):
    base, source, out, calls = checkpoint
    original = requant.atomic_json

    def interrupted(path, value):
        if path.name == "config.json":
            raise RuntimeError("metadata interruption")
        return original(path, value)

    monkeypatch.setattr(requant, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="metadata interruption"):
        requant.derive(base, source, out)
    assert calls == [(9, 64)]
    monkeypatch.setattr(requant, "atomic_json", original)
    requant.derive(base, source, out)
    assert calls == [(9, 64)]


def test_changed_input_refuses_resume(checkpoint, monkeypatch):
    base, source, out, calls = checkpoint
    original = requant.rewrite_shard

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(requant, "rewrite_shard", interrupted)
    with pytest.raises(RuntimeError):
        requant.derive(base, source, out)
    monkeypatch.setattr(requant, "rewrite_shard", original)
    config_path = base / "config.json"
    config = json.loads(config_path.read_text())
    config["new_setting"] = True
    atomic_json(config_path, config)
    with pytest.raises(ValueError, match="identity mismatch"):
        requant.derive(base, source, out)
    assert not out.exists() and not calls


def test_real_width_96_uses_group_32(checkpoint):
    base, source, out, calls = checkpoint
    recipe = Recipe(requant.RECIPE)
    assert recipe.spec(WEIGHT) == (8, 32, "rtn")
    assert 96 % requant.GROUP == 0
