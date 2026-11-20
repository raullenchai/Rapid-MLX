"""Portable loader, prefetch, and malformed-checkpoint contracts."""

import importlib.util
import json
import struct
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tests.test_hy_v4_flash import cpu_model as hy4_cpu_model
from tests.test_hy_v4_flash import tiny

cpu_model = hy4_cpu_model

PREFIX = "model.layers.1.engram_layer.multi_head_embedding.embedding"


def write_shard(path, tensors):
    header, data = {}, bytearray()
    for key, (dtype, value) in tensors.items():
        raw = value.tobytes()
        header[key] = dict(
            dtype=dtype,
            shape=list(value.shape),
            data_offsets=[len(data), len(data) + len(raw)],
        )
        data.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


@pytest.fixture
def loader(cpu_model, monkeypatch):
    """Load production code against inert array/module boundaries."""
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.hy_v4_flash", cpu_model)
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda value: list(value.items())
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    shared_spec = importlib.util.spec_from_file_location(
        "rapid_mlx.models._engram_offload_cpu",
        Path("rapid_mlx/models/engram_offload.py"),
    )
    shared = importlib.util.module_from_spec(shared_spec)
    shared_spec.loader.exec_module(shared)
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.engram_offload", shared)
    spec = importlib.util.spec_from_file_location(
        "rapid_mlx.models._hy4_load_cpu", Path("rapid_mlx/models/hy_v4_flash_load.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.typed_mlx_load_boundaries = nullcontext
    module.ubc_evict = lambda _: None
    module.mx.cpu = "cpu"
    module.mx.eval = lambda _: None
    return module


@pytest.fixture
def checkpoint(tmp_path):
    config = asdict(tiny(engram_n_embed_per_ngram=64))
    config["quantization"] = dict(bits=4, group_size=32, mode="affine")
    tensors = {
        PREFIX + ".weight": ("U32", np.zeros((36, 4), "<u4")),
        PREFIX + ".scales": ("BF16", np.ones((36, 1), "<u2") * 0x3F80),
        PREFIX + ".biases": ("BF16", np.zeros((36, 1), "<u2")),
    }
    write_shard(tmp_path / "table.safetensors", tensors)
    write_shard(
        tmp_path / "model.safetensors",
        {"dense.weight": ("F32", np.ones((2, 2), "<f4"))},
    )
    mapping = {key: "table.safetensors" for key in tensors}
    mapping["dense.weight"] = "model.safetensors"
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    return tmp_path, config, mapping, tensors


def fake_model(loader, monkeypatch):
    class Model:
        def __init__(self, args):
            self.layers = [
                None,
                SimpleNamespace(
                    engram_layer=SimpleNamespace(
                        multi_head_embedding=SimpleNamespace(
                            embedding=SimpleNamespace(weight=np.zeros((36, 32)))
                        )
                    )
                ),
            ]
            self.values = {}

        def sanitize(self, weights):
            return weights

        def load_weights(self, weights, strict):
            assert strict
            self.values = dict(weights)
            expected = {"dense.weight"}
            table = self.layers[1].engram_layer.multi_head_embedding.embedding
            if not isinstance(table, loader.DiskQuantizedEngramEmbedding):
                expected |= {
                    PREFIX + "." + leaf for leaf in ("weight", "scales", "biases")
                }
            if set(self.values) != expected:
                raise ValueError("strict parameter mismatch")

        def parameters(self):
            return self.values

        def eval(self):
            self.evaluated = True

    monkeypatch.setattr(loader, "Model", Model)
    seen = []

    def quantize(model, **kwargs):
        pred = kwargs["class_predicate"]
        assert not pred(PREFIX, SimpleNamespace())
        seen.append(pred("dense", SimpleNamespace(to_quantized=True)))

    monkeypatch.setattr(loader.nn, "quantize", quantize, raising=False)
    return seen


def test_supported_load_skips_table_and_bounds_eval(loader, checkpoint, monkeypatch):
    path, _, mapping, _ = checkpoint
    assert loader.supports_engram_ssd_offload(path)
    seen = fake_model(loader, monkeypatch)
    opened, evaluated = [], []

    def load_shard(name, **kwargs):
        opened.append(Path(name).name)
        return {"dense.weight": np.ones((2, 2))}

    monkeypatch.setattr(loader.mx, "load", load_shard, raising=False)
    monkeypatch.setattr(loader.mx, "eval", lambda batch: evaluated.append(list(batch)))
    model, _ = loader.load(path)
    assert model.evaluated and opened == ["model.safetensors"]
    assert len(evaluated) == 1 and seen == [False]
    table = model.layers[1].engram_layer.multi_head_embedding.embedding
    assert isinstance(table, loader.DiskQuantizedEngramEmbedding)
    table.close()
    # Explicit opt-out is stronger than environment/default selection.
    monkeypatch.setenv("HY4_ENGRAM_SSD_OFFLOAD", "0")
    monkeypatch.setattr(
        loader.mx,
        "load",
        lambda name, **kwargs: {
            k: np.ones((2, 2)) for k, v in mapping.items() if v == Path(name).name
        },
    )
    model, _ = loader.load(path)
    assert not isinstance(
        model.layers[1].engram_layer.multi_head_embedding.embedding,
        loader.DiskQuantizedEngramEmbedding,
    )
    model, _ = loader.load(path, engram_ssd_offload=True)
    model.layers[1].engram_layer.multi_head_embedding.embedding.close()


def test_bounded_groups_count_bytes_empty_and_oversized(loader, monkeypatch):
    batches = []
    monkeypatch.setattr(loader.mx, "eval", lambda batch: batches.append(list(batch)))
    values = [np.zeros(size, np.uint8) for size in (4, 4, 20, 2, 2, 2)]
    loader.evaluate_weights(enumerate(values), max_bytes=10, max_tensors=2)
    assert [[v.nbytes for v in batch] for batch in batches] == [
        [4, 4],
        [20],
        [2, 2],
        [2],
    ]
    loader.evaluate_weights([])
    assert len(batches) == 4


@pytest.mark.parametrize("leaf", ["weight", "scales", "biases"])
def test_table_byte_lengths_must_match_header(loader, checkpoint, leaf):
    path, _, _, tensors = checkpoint
    key = PREFIX + "." + leaf
    dtype, value = tensors[key]
    # Keep expected header shapes/dtypes but provide the wrong number of bytes.
    tensors[key] = dtype, value.astype("u1")
    write_shard(path / "table.safetensors", tensors)
    assert not loader.supports_engram_ssd_offload(path)


@pytest.mark.parametrize("leaf", ["weight", "scales", "biases"])
@pytest.mark.parametrize("axis", [0, 1])
def test_malformed_table_shapes_fall_back_to_resident(
    loader, checkpoint, monkeypatch, leaf, axis
):
    path, _, mapping, tensors = checkpoint
    key = PREFIX + "." + leaf
    dtype, value = tensors[key]
    shape = list(value.shape)
    shape[axis] += 1
    tensors[key] = dtype, np.zeros(shape, value.dtype)
    write_shard(path / "table.safetensors", tensors)
    assert not loader.supports_engram_ssd_offload(path)
    fake_model(loader, monkeypatch)
    opened = []

    def load_shard(name, **kwargs):
        opened.append(Path(name).name)
        return {
            k: tensors[k][1] if k in tensors else np.ones((2, 2))
            for k, v in mapping.items()
            if v == Path(name).name
        }

    monkeypatch.setattr(loader.mx, "load", load_shard, raising=False)
    model, _ = loader.load(path)
    assert set(opened) == {"table.safetensors", "model.safetensors"}
    assert model.evaluated


@pytest.mark.parametrize("dtype", ["F16", "F32"])
@pytest.mark.parametrize("leaves", [("scales",), ("biases",), ("scales", "biases")])
def test_non_bf16_affine_tables_load_resident(
    loader, checkpoint, monkeypatch, dtype, leaves
):
    path, _, mapping, tensors = checkpoint
    for leaf in leaves:
        tensors[PREFIX + "." + leaf] = (
            dtype,
            np.ones((36, 1), "<f2" if dtype == "F16" else "<f4"),
        )
    write_shard(path / "table.safetensors", tensors)
    assert not loader.supports_engram_ssd_offload(path)
    fake_model(loader, monkeypatch)
    opened = []

    def load_shard(name, **kwargs):
        opened.append(Path(name).name)
        return {
            key: tensors[key][1] if key in tensors else np.ones((2, 2))
            for key, filename in mapping.items()
            if filename == Path(name).name
        }

    monkeypatch.setattr(loader.mx, "load", load_shard, raising=False)
    model, _ = loader.load(path)
    assert model.evaluated
    assert set(opened) == {"table.safetensors", "model.safetensors"}
    assert not isinstance(
        model.layers[1].engram_layer.multi_head_embedding.embedding,
        loader.DiskQuantizedEngramEmbedding,
    )
    for leaf in leaves:
        np.testing.assert_array_equal(model.values[PREFIX + "." + leaf], 1)
        assert (
            model.values[PREFIX + "." + leaf].dtype
            == tensors[PREFIX + "." + leaf][1].dtype
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "no_index",
        "bad_map",
        "missing",
        "escape",
        "missing_file",
        "mode",
        "bits",
        "group",
        "unquantized",
        "no_layers",
        "false_entry",
        "bad_filename",
    ],
)
def test_capability_and_explicit_load_reject_invalid_layout(
    loader, checkpoint, mutation
):
    path, config, mapping, _ = checkpoint
    if mutation == "no_index":
        (path / "model.safetensors.index.json").unlink()
    elif mutation == "bad_map":
        mapping = []
    elif mutation == "missing":
        del mapping[PREFIX + ".biases"]
    elif mutation == "escape":
        mapping[PREFIX + ".weight"] = "../outside.safetensors"
    elif mutation == "missing_file":
        mapping[PREFIX + ".weight"] = "absent.safetensors"
    elif mutation in ("mode", "bits", "group"):
        config["quantization"][
            {"mode": "mode", "bits": "bits", "group": "group_size"}[mutation]
        ] = {"mode": "mxfp4", "bits": 1, "group": 7}[mutation]
    elif mutation == "unquantized":
        config.pop("quantization")
    elif mutation == "no_layers":
        config["engram_layer_ids"] = []
    elif mutation == "false_entry":
        config["quantization"][PREFIX] = False
    elif mutation == "bad_filename":
        mapping[PREFIX + ".weight"] = 42
    (path / "config.json").write_text(json.dumps(config))
    if mutation != "no_index":
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
    assert not loader.supports_engram_ssd_offload(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        loader.load(path, engram_ssd_offload=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "dtype",
        "shape",
        "offset",
        "missing_tensor",
        "wrong_index",
        "extra",
        "missing_dense",
        "duplicate",
        "no_shards",
    ],
)
def test_strict_tensor_validation_and_failure_cleanup(
    loader, checkpoint, monkeypatch, mutation
):
    path, _, mapping, tensors = checkpoint
    fake_model(loader, monkeypatch)
    if mutation == "dtype":
        tensors[PREFIX + ".scales"] = ("F16", tensors[PREFIX + ".scales"][1])
    elif mutation == "shape":
        tensors[PREFIX + ".weight"] = ("U32", np.zeros((35, 4), "<u4"))
    elif mutation == "missing_tensor":
        del tensors[PREFIX + ".biases"]
    write_shard(path / "table.safetensors", tensors)
    if mutation == "offset":
        with (path / "table.safetensors").open("ab") as file:
            file.write(b"junk")
    if mutation in ("dtype", "missing_tensor", "offset"):
        assert not loader.supports_engram_ssd_offload(path)
    if mutation == "wrong_index":
        mapping["other.weight"] = "model.safetensors"
    if mutation == "missing_dense":
        mapping["missing.weight"] = "model.safetensors"
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )

    def load_shard(name, **kwargs):
        result = {"dense.weight": np.ones((2, 2))}
        if mutation == "extra":
            result["extra.weight"] = np.ones(1)
        return result

    monkeypatch.setattr(loader.mx, "load", load_shard, raising=False)
    closed = []
    original = loader.DiskQuantizedEngramEmbedding.close

    def close(reader):
        closed.append(True)
        original(reader)

    monkeypatch.setattr(loader.DiskQuantizedEngramEmbedding, "close", close)
    if mutation in ("duplicate", "no_shards"):
        (path / "model.safetensors.index.json").unlink()
        if mutation == "duplicate":
            (path / "extra.safetensors").touch()
        else:
            for shard in path.glob("*.safetensors"):
                shard.unlink()
    with pytest.raises(ValueError):
        loader.load(path, engram_ssd_offload=mutation not in ("duplicate", "no_shards"))
    if mutation in ("wrong_index", "extra", "missing_dense"):
        assert closed


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_prefetch_uses_current_history_before_any_decoder(
    cpu_model, monkeypatch, dtype
):
    model = cpu_model.Model(
        tiny(
            engram_layer_ids=[1, 2],
            engram_layer_head_vocab_sizes={
                "1": [[5, 7], [11, 13]],
                "2": [[5, 7], [11, 13]],
            },
        )
    )
    events = []
    for index, layer in enumerate(model.layers):
        table = layer.engram_layer.multi_head_embedding.embedding

        class Table:
            def __init__(self, original, index):
                self.original, self.index = original, index

            def prefetch(self, ids):
                self.prefetched = ids.copy()
                events.append(("prefetch", self.index))

            def __call__(self, ids):
                np.testing.assert_array_equal(self.prefetched, ids)
                events.append(("read", self.index))
                return self.original(ids)

        layer.engram_layer.multi_head_embedding.embedding = Table(table, index)
    cache = model.make_cache()
    cache_type = type(cache[0][2])
    monkeypatch.setattr(
        cache_type, "state", property(lambda c: (c.keys, c.values)), raising=False
    )
    for tokens in ([[1, 2, 3]], [[4]], [[5, 6]]):
        events.clear()
        model(np.array(tokens, dtype=dtype), cache)
        assert events == [("prefetch", 0), ("prefetch", 1), ("read", 0), ("read", 1)]
    for c in cache:
        c.trim(2)
    model(np.array([[7, 8]], dtype=dtype), cache)


def test_prefetch_matching_consumed_once_and_mismatch_reread(
    loader, checkpoint, monkeypatch
):
    path, _, _, tensors = checkpoint
    tensors[PREFIX + ".weight"] = ("U32", np.arange(36 * 4, dtype="<u4").reshape(36, 4))
    write_shard(path / "table.safetensors", tensors)
    table = loader.DiskQuantizedEngramEmbedding(
        path / "table.safetensors",
        weight_key=PREFIX + ".weight",
        scales_key=PREFIX + ".scales",
        biases_key=PREFIX + ".biases",
        num_embeddings=36,
        dim=32,
        group_size=32,
        bits=4,
        cache_rows=0,
    )
    # Inspect the packed rows independently of device dequantization.
    monkeypatch.setattr(table, "_dequantize", lambda rows: rows[0])
    calls = []
    original = table._gather_rows

    def gather(flat):
        calls.append(flat.copy())
        return original(flat)

    monkeypatch.setattr(table, "_gather_rows", gather)
    requested = np.array([[3, 2, 3]])
    expected = tensors[PREFIX + ".weight"][1][requested]
    np.testing.assert_array_equal(table(requested), expected)
    calls.clear()
    table.prefetch(requested)
    np.testing.assert_array_equal(table(requested), expected)
    assert len(calls) == 1
    calls.clear()
    table.prefetch(np.array([[5, 6, 7]]))
    table._pending[1].result()
    np.testing.assert_array_equal(table(requested), expected)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[-1], requested.reshape(-1))
    table.close()


def test_mixed_shard_and_indexed_read_omission(loader, checkpoint, monkeypatch):
    path, _, mapping, tensors = checkpoint
    fake_model(loader, monkeypatch)
    all_tensors = {**tensors, "dense.weight": ("F32", np.ones((2, 2), "<f4"))}
    write_shard(path / "model.safetensors", all_tensors)
    mapping = {key: "model.safetensors" for key in all_tensors}
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    monkeypatch.setattr(
        loader.mx,
        "load",
        lambda *a, **k: {key: value for key, (_, value) in all_tensors.items()},
        raising=False,
    )
    model, _ = loader.load(path)
    assert set(model.values) == {"dense.weight"}
    model.layers[1].engram_layer.multi_head_embedding.embedding.close()
    monkeypatch.setattr(loader.mx, "load", lambda *a, **k: {}, raising=False)
    with pytest.raises(ValueError, match="do not match"):
        loader.load(path)


def test_mixed_shard_load_does_not_read_offloaded_tensors(tmp_path):
    """Unused arrays from mx.load are never read, so offloaded tables stay on disk."""
    import subprocess
    import sys

    mx = pytest.importorskip("mlx.core")
    path = tmp_path / "mixed.safetensors"
    mx.save_safetensors(
        str(path),
        {"engram": mx.zeros((256, 1024, 1024), dtype=mx.float32), "w": mx.ones((4,))},
    )
    code = (
        "import resource, sys, mlx.core as mx\n"
        "mx.set_default_device(mx.cpu)\n"
        "d = mx.load(sys.argv[1], stream=mx.cpu)\n"
        "w = d.pop('w'); d.clear(); mx.eval(w)\n"
        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    maxrss = int(out.stdout.strip())
    # macOS reports bytes, Linux kilobytes; either way far below the 1 GiB table.
    assert maxrss < 512 * 2**20


@pytest.mark.parametrize("offload", [False, None])
def test_resident_load_rejects_external_shard_symlink(
    loader, checkpoint, monkeypatch, offload
):
    path, _, _, _ = checkpoint
    model_dir = path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text((path / "config.json").read_text())
    (model_dir / "model.safetensors").symlink_to(path / "model.safetensors")
    fake_model(loader, monkeypatch)
    monkeypatch.setattr(
        loader.mx,
        "load",
        lambda *a, **k: pytest.fail("External shard opened"),
        raising=False,
    )
    assert not loader.supports_engram_ssd_offload(model_dir)
    with pytest.raises(ValueError, match="escapes"):
        loader.load(model_dir, engram_ssd_offload=offload)
