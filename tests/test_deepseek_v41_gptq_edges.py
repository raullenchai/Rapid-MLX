"""Invalid-input and provenance regressions for offline GPTQ tools."""

import copy
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from scripts import deepseek_v41_gptq as b
from scripts import deepseek_v41_gptq_amend as a
from scripts import deepseek_v41_gptq_io as io
from scripts import deepseek_v41_gptq_quant as q
from tests.test_deepseek_v41_gptq_amend import metadata
from tests.test_deepseek_v41_streaming_teacher import write_source


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize(
    "change",
    [
        {"group_size": 16},
        {"block_size": 33},
        {"damping": float("nan")},
        {"min_tokens": -1},
        {"low_token_damping": 0},
    ],
)
def test_invalid_quant_config(change):
    with pytest.raises(ValueError):
        q.QuantConfig(**change)


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": 1},
        {"pilot": 1},
        {"dense_8bit_layers": [-1]},
        {"prefill_chunk": 0},
        {"memory_limit_gb": float("nan")},
        {"gate_up_bits": 8},
        {"dense_bits": 3},
        {"engram_group_size": 1},
    ],
)
def test_invalid_recipe(change):
    with pytest.raises(ValueError):
        b.recipe(change)


def test_invalid_packing_and_factors():
    for codes, bits in [(np.zeros((1, 7)), 3), (np.full((1, 32), 8), 3)]:
        with pytest.raises(ValueError):
            q.pack_codes(codes, bits)
    with pytest.raises(ValueError, match="square"):
        q.inverse_factor(mx.zeros((2, 3)), q.QuantConfig(), 1)
    weight, h = mx.ones((2, 32)), mx.eye(32)
    c = q.QuantConfig(group_size=32)
    with pytest.raises(ValueError, match="supports"):
        q.gptq(weight, h, 6, c)
    with pytest.raises(ValueError, match="shape"):
        q.gptq(weight, mx.eye(16), 2, c)
    with pytest.raises(ValueError, match="supports"):
        q.gptq_batched(weight[None], h[None], 6, c)
    with pytest.raises(ValueError, match="weights"):
        q.gptq_batched(weight, h, 2, c)
    with pytest.raises(ValueError, match="shape"):
        q.gptq_batched(weight[None], h, 2, c)
    with pytest.raises(ValueError, match="dead mask"):
        q.gptq_batched(weight[None], h[None], 2, c, dead=mx.zeros((32,)))
    actual = q.gptq_batched(weight[None], h[None], 2, c)
    expected = q.gptq_batched(
        weight[None], h[None], 2, c, dead=mx.zeros((1, 32), mx.bool_)
    )
    assert all(
        np.array_equal(
            np.asarray(x.astype(mx.float32)), np.asarray(y.astype(mx.float32))
        )
        for x, y in zip(actual, expected)
    )


@pytest.mark.parametrize("kind", ["keys", "cap", "layers", "same"])
def test_invalid_amendment_recipe(kind):
    old = b.recipe()
    new = dict(old)
    if kind == "keys":
        new.pop("pilot")
    elif kind == "cap":
        new["memory_limit_gb"] = float("nan")
    elif kind == "layers":
        new["gate_up_3bit_layers"] = [-1]
    with pytest.raises(ValueError):
        a.allowed_change(old, new, 0)


@pytest.mark.parametrize("kind", ["type", "boundary", "audit", "digest", "layer"])
def test_invalid_amendment_chain(tmp_path, kind):
    parts, scratch, original = metadata(tmp_path)
    new = {**original["recipe"], "resident_budget_gb": 216.0}
    run = a.amend_recipe(parts, scratch, new, 0, "increase cap")
    if kind == "type":
        run["recipe_amendments"] = {}
    elif kind == "boundary":
        run["recipe_amendments"][0]["from_layer"] = -1
    elif kind == "audit":
        run["recipe_amendments"][0]["reason"] = " "
    elif kind == "digest":
        run["identity"] = "bad"
    with pytest.raises(ValueError):
        a.layer_epoch(run, -1) if kind == "layer" else a.epochs(run)


def test_source_payload_mutation_and_conversion_binding(tmp_path):
    root = tmp_path / "source"
    write_source(root)
    manifest = io.inventory(root)
    source = io.PartialSource(root, manifest)
    record = io.conversion_record(source, root)
    assert record["source_checkpoint_sha256"] == record["checkpoint_sha256"]
    from scripts.evaluate_deepseek_v41_rd import source_identity_for_checkpoint

    path = tmp_path / "conversion.json"
    path.write_text(json.dumps(record))
    assert (
        source_identity_for_checkpoint(
            record["checkpoint_sha256"], path, record["checkpoint_index_json"]
        )[0]
        == manifest["checkpoint_sha256"]
    )
    shard = root / "source.safetensors"
    with shard.open("r+b") as f:
        f.seek(-1, 2)
        value = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="payload mismatch"):
        source.verify(0)
    legacy = copy.deepcopy(manifest)
    legacy.pop("checkpoint_sha256")
    with pytest.raises(ValueError, match="immutable"):
        io.PartialSource(root, legacy)


def test_inventory_rejects_input_mutation(tmp_path, monkeypatch):
    root = tmp_path / "source"
    write_source(root)
    real = io.checkpoint_hashes
    calls = 0

    def changed(path):
        nonlocal calls
        calls += 1
        result = real(path)
        if calls == 2:
            result["config.json"] = "0" * 64
        return result

    monkeypatch.setattr(io, "checkpoint_hashes", changed)
    with pytest.raises(ValueError, match="changed during inventory"):
        io.inventory(root)
