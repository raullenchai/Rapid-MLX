"""CPU contracts for recipes, bounded FP8 reads and restart artifacts."""

import json
import struct
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rapid_mlx.models.hy_v4_flash_config import checkpoint_key
from scripts import hy4_gptq as builder
from scripts.hy4_gptq_adapter import Hy4Adapter, Recipe
from scripts.streamed_gptq_io import (
    RangeSink,
    TensorReader,
    atomic_json,
    digest,
    verify_receipt,
)
from tests.test_hy_v4_flash import cpu_model as cpu_model
from tests.test_hy_v4_flash import tiny


def write_source(root, tensors, config=None):
    """Write raw synthetic FP8 too; no framework is allowed to interpret it."""
    root.mkdir(exist_ok=True)
    config = config or asdict(tiny(hidden_size=64, q_lora_rank=64, kv_lora_rank=64))
    atomic_json(root / "config.json", config)
    header, payload, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = dict(
            dtype=dtype,
            shape=list(array.shape),
            data_offsets=[offset, offset + len(raw)],
        )
        payload.append(raw)
        offset += len(raw)
    raw_header = json.dumps(header).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    (root / "weights.safetensors").write_bytes(
        struct.pack("<Q", len(raw_header)) + raw_header + b"".join(payload)
    )
    atomic_json(
        root / "model.safetensors.index.json",
        {"weight_map": dict.fromkeys(tensors, "weights.safetensors")},
    )
    return root


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("mlp.experts.gate_up_proj", (4, 64, "gptq")),
        ("mlp.experts.down_proj", (4, 64, "gptq")),
        ("mlp.shared_experts.gate_proj.weight", (8, 64, "rtn")),
        ("mlp.routed_expert_down_proj.weight", (8, 64, "rtn")),
        ("mlp.routed_expert_up_proj.weight", (8, 64, "rtn")),
        ("mlp.gate.weight", None),
        ("mlp.gate.e_score_correction_bias", None),
        ("self_attn.q_a_proj.weight", (8, 64, "rtn")),
        ("self_attn.kv_b_proj.weight", (8, 64, "rtn")),
        ("self_attn.compressed_indexer.key_gate_projection.weight", None),
        ("self_attn.compressed_indexer.key_value_projection.weight", (8, 64, "rtn")),
        ("self_attn.compressed_indexer.head_weight_projection.weight", (8, 64, "rtn")),
        ("self_attn.learnable_sink_param", None),
        ("input_layernorm.weight", None),
        ("hc_attn_layer.hc_pre.hc_fn", None),
        ("engram_layer.multi_head_embedding.embedding.weight", (4, 32, "rtn")),
        ("engram_layer.key_projs.0.weight", None),
    ],
)
def test_real_tensor_recipe_table(suffix, expected):
    name = "model.layers.1." + suffix
    names = json.loads(
        (
            Path(__file__).parent / "fixtures/hy_v4_flash/text_tensor_names.json"
        ).read_text()
    )
    assert name in names
    assert Recipe().spec(name) == expected


def test_entire_frozen_source_index_classifies_and_fusion_domains_agree():
    names = json.loads(
        (
            Path(__file__).parent / "fixtures/hy_v4_flash/text_tensor_names.json"
        ).read_text()
    )
    recipe = Recipe()
    for name in names:
        spec = recipe.spec(name)
        if spec:
            assert spec[0] in (4, 8) and spec[1] in (32, 64)
        if name.endswith("_scale_inv") or checkpoint_key(name) is None:
            assert spec is None
    for suffix in ("gate_proj", "up_proj", "gate_up_proj"):
        assert recipe.spec(f"model.layers.1.mlp.switch_mlp.{suffix}.weight") == (
            4,
            64,
            "gptq",
        )
    assert recipe.spec("model.layers.0.mlp.down_proj.weight") == (8, 64, "rtn")
    assert (
        recipe.spec("model.embed_tokens.weight")
        == recipe.spec("lm_head.weight")
        == (8, 64, "rtn")
    )
    assert Recipe(keep_embeddings_bf16=True).spec("lm_head.weight") is None
    low = Recipe("hy4-gptq-lowbit", gate_up_3bit_layers=(1,))
    assert low.spec("model.layers.1.mlp.experts.gate_up_proj") == (3, 64, "gptq")
    assert low.spec("model.layers.2.mlp.experts.gate_up_proj") == (2, 64, "gptq")
    assert low.spec("model.layers.2.mlp.experts.down_proj") == (3, 64, "gptq")
    with pytest.raises(ValueError, match="unclassified"):
        recipe.spec("model.unknown.weight")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="bad"),
        dict(keep_embeddings_bf16=1),
        dict(gate_up_3bit_layers=(1,)),
        dict(name="hy4-gptq-lowbit", gate_up_3bit_layers=(1, 1)),
        dict(name="hy4-gptq-lowbit", gate_up_3bit_layers=(-1,)),
    ],
)
def test_recipe_rejects_bad_overrides(kwargs):
    with pytest.raises(ValueError):
        Recipe(**kwargs)


def test_prefix_must_be_valid_and_contiguous():
    assert builder.layers_prefix(None, 3) == [0, 1, 2]
    assert builder.layers_prefix("0,1", 3) == [0, 1]
    for value in ("", "1", "0,2", "0,0", "0,1,2,3", "-1"):
        with pytest.raises(ValueError):
            builder.layers_prefix(value, 3)


@pytest.mark.parametrize("width", [8, 32, 63, 65])
def test_adapter_rejects_nonintegral_affine_groups(tmp_path, width):
    name = "model.layers.0.self_attn.o_proj.weight"
    source = write_source(tmp_path, {name: ("F32", np.zeros((64, width), np.float32))})
    with pytest.raises(ValueError, match=f"nonintegral affine group: {name}"):
        Hy4Adapter(source)


def test_bounded_fp8_split_and_every_row_boundary(tmp_path, cpu_model, monkeypatch):
    name = "model.layers.1.mlp.experts.gate_up_proj"
    raw = np.full((3, 128, 64), 0x38, np.uint8)
    raw[:, 64:] = 0x40
    scales = np.arange(1, 3 * 8 * 2 + 1, dtype=np.float32).reshape(3, 8, 2)
    config = asdict(
        tiny(hidden_size=64, routed_expert_hidden_size=64, moe_intermediate_size=64)
    )
    config["quantization_config"] = dict(weight_block_size=[16, 32])
    source = write_source(
        tmp_path, {name: ("F8_E4M3", raw), name + "_scale_inv": ("F32", scales)}, config
    )
    adapter = Hy4Adapter(source)
    gate = "model.layers.1.mlp.switch_mlp.gate_proj.weight"
    up = gate.replace("gate_proj", "up_proj")
    reads = []
    original = adapter.reader.raw

    def record(name, **kwargs):
        value = original(name, **kwargs)
        reads.append((name, value.shape))
        return value

    monkeypatch.setattr(adapter.reader, "raw", record)
    for expert in range(3):
        expected = np.repeat(np.repeat(scales[expert], 16, axis=0), 32, axis=1)
        for name, start, factor in ((gate, 0, 1), (up, 64, 2)):
            np.testing.assert_array_equal(
                adapter.tensor(name, expert=expert),
                expected[start : start + 64] * factor,
            )
            for split in range(1, 64):
                np.testing.assert_array_equal(
                    adapter.tensor(name, expert=expert, rows=(split, split + 1)),
                    expected[start + split : start + split + 1] * factor,
                )
    assert all(shape != raw.shape for _, shape in reads)
    with pytest.raises(ValueError, match="one expert"):
        adapter.tensor(gate)
    with pytest.raises(ValueError, match="invalid expert"):
        adapter.tensor(gate, expert=3)
    for rows in ((-1, 1), (64, 65), (1, 1)):
        with pytest.raises(ValueError, match="canonical row"):
            adapter.tensor(gate, expert=0, rows=rows)


@pytest.mark.parametrize(
    "kind", ["missing", "zero", "nan", "shape", "nonfinite_weight"]
)
def test_fp8_rejects_malformed_source(tmp_path, cpu_model, kind):
    name = "model.layers.0.mlp.up_proj.weight"
    raw = np.full((64, 64), 0x38, np.uint8)
    scales = np.ones((1, 1), np.float32)
    if kind == "zero":
        scales[:] = 0
    elif kind == "nan":
        scales[:] = np.nan
    elif kind == "shape":
        scales = np.ones((3, 3), np.float32)
    elif kind == "nonfinite_weight":
        raw[0, 0] = 0x7F
    tensors = {name: ("F8_E4M3", raw)}
    if kind != "missing":
        tensors[name + "_scale_inv"] = ("F32", scales)
    adapter = Hy4Adapter(write_source(tmp_path, tensors))
    with pytest.raises(ValueError):
        adapter.tensor(name)


def test_row_scaled_engram_partial_read(tmp_path, cpu_model):
    name = "model.layers.1.engram_layer.multi_head_embedding.embedding.weight"
    scales = np.arange(1, 10, dtype=np.float32)
    adapter = Hy4Adapter(
        write_source(
            tmp_path,
            {
                name: ("F8_E4M3", np.full((9, 32), 0x38, np.uint8)),
                name + "_scale_inv": ("F32", scales),
            },
        )
    )
    for start in range(9):
        np.testing.assert_array_equal(
            adapter.tensor(name, rows=(start, start + 1)),
            np.full((1, 32), scales[start]),
        )


def test_sink_ranges_and_resume_tamper(tmp_path):
    path = tmp_path / "output.safetensors"
    sink = RangeSink(path, {"bank": ("F32", (3, 2)), "lookup": ("I64", (3,))})
    with pytest.raises(ValueError, match="dtype or shape"):
        sink.put("bank", 0, np.zeros((1, 2), np.float64))
    sink.put("bank", 0, np.ones((2, 2), np.float32))
    with pytest.raises(ValueError, match="duplicate"):
        sink.put("bank", 1, np.ones((1, 2), np.float32))
    with pytest.raises(ValueError, match="incomplete"):
        sink.finish()
    sink.put("bank", 2, np.full((1, 2), 3, np.float32))
    sink.put("lookup", 0, np.arange(3, dtype=np.int64))
    sink.finish()
    atomic_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": dict.fromkeys(("bank", "lookup"), path.name)},
    )
    reader = TensorReader(tmp_path)
    np.testing.assert_array_equal(reader.raw("bank"), [[1, 1], [1, 1], [3, 3]])
    np.testing.assert_array_equal(reader.raw("lookup"), [0, 1, 2])
    with pytest.raises(ValueError):
        reader.raw("bank", rows=(-1, 1))
    from scripts.deepseek_v41_rd_common import file_hash

    receipt = dict(identity=digest("a"), files={path.name: file_hash(path)})
    verify_receipt(tmp_path, receipt, digest("a"))
    with pytest.raises(ValueError, match="identity"):
        verify_receipt(tmp_path, receipt, digest("b"))
    with path.open("r+b") as f:
        f.seek(-1, 2)
        f.write(b"\xff")
    with pytest.raises(ValueError, match="artifact"):
        verify_receipt(tmp_path, receipt, digest("a"))
    with pytest.raises(ValueError, match="unsafe"):
        verify_receipt(tmp_path, dict(identity="a", files={"../escape": ""}), "a")


def test_reader_rejects_truncated_and_traversing_index(tmp_path):
    source = write_source(tmp_path, {"x": ("F32", np.ones((2, 2), np.float32))})
    path = source / "weights.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        TensorReader(source)
    atomic_json(
        source / "model.safetensors.index.json",
        {"weight_map": {"x": "../weights.safetensors"}},
    )
    with pytest.raises(ValueError, match="unsafe"):
        TensorReader(source)


def test_calibration_local_chat_modes_and_slice_contract(tmp_path, monkeypatch):
    calls = []
    rendered, encoded = [], []

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == dict(local_files_only=True, trust_remote_code=False)
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            assert kwargs["tokenize"] is False
            text = json.dumps(messages) + kwargs["reasoning_effort"]
            rendered.append(text)
            return text

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            assert text == rendered[-1]
            encoded.append(text)
            return list(range(6))

    module = ModuleType("transformers")
    module.AutoTokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    slices = {}
    for name in ("english", "chinese", "code", "tools"):
        filename = name + ".json"
        atomic_json(
            tmp_path / filename,
            {"messages": [{"role": "user", "content": name}], "tools": []},
        )
        slices[name] = [filename]
    manifest = tmp_path / "slices.json"
    atomic_json(manifest, dict(slices=slices))
    windows, evidence = builder.calibration(
        SimpleNamespace(source=tmp_path), manifest, 4
    )
    assert len(windows) == len(calls) == 8
    assert encoded == rendered
    assert all(len(w) == 4 for w in windows)
    assert {c[1]["reasoning_effort"] for c in calls} == {"high", "no_think"}
    assert set(evidence) == set(slices)
    with pytest.raises(ValueError, match="window size"):
        builder.calibration(SimpleNamespace(source=tmp_path), manifest, 0)
    del slices["tools"]
    atomic_json(manifest, dict(slices=slices))
    with pytest.raises(ValueError, match="tools"):
        builder.calibration(SimpleNamespace(source=tmp_path), manifest, 4)


def test_adapter_forward_uses_production_hc_attention_and_clamp(cpu_model, monkeypatch):
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.hy_v4_flash", cpu_model)
    mx = sys.modules["mlx.core"]
    mx.eval = lambda *args: None
    model = cpu_model.Model(tiny())
    adapter = object.__new__(Hy4Adapter)
    adapter.args = model.args
    tokens = np.array([1, 2, 3], np.int32)
    hidden = model.model.embed_tokens(tokens[None])
    hidden = np.broadcast_to(
        hidden[..., None, :], (*hidden.shape[:-1], model.args.hc_mult, hidden.shape[-1])
    )
    for block in model.layers:
        hidden = adapter.forward(block, hidden, tokens)
    expected = model(tokens[None])
    np.testing.assert_allclose(
        model.lm_head(model.model.norm(model.model.hc_head(hidden))),
        expected,
        atol=1e-6,
    )
    x = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
    actual = adapter.intermediate(np.eye(4) * 20, -np.eye(4) * 20, x)
    np.testing.assert_allclose(actual, -10 * (10 / (1 + np.exp(-10))), rtol=1e-6)
    latent, ids, weights = adapter.routed_inputs(
        model.layers[1], np.ones((1, 3, 8), np.float32)
    )
    assert latent.shape == (3, 4) and ids.shape == weights.shape == (3, 2)


def test_shared_gptq_numerics_on_cpu_array_seam(cpu_model, monkeypatch):
    """Exercise error compensation with a NumPy affine codec, without Metal."""
    import importlib.util

    mx = sys.modules["mlx.core"]
    mx.eval = lambda *args: None
    path = Path(__file__).parents[1] / "scripts/streamed_gptq_quant.py"
    spec = importlib.util.spec_from_file_location("_hy4_quant_cpu_contract", path)
    quant = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, quant)
    spec.loader.exec_module(quant)

    def quantize(weight, *, bits, group_size):
        grouped = weight.reshape(*weight.shape[:-1], -1, group_size)
        bias = grouped.min(-1)
        scale = np.maximum((grouped.max(-1) - bias) / (2**bits - 1), 1e-8)
        codes = (
            np.clip(
                np.round((grouped - bias[..., None]) / scale[..., None]), 0, 2**bits - 1
            )
            .astype(np.uint32)
            .reshape(weight.shape)
        )
        return quant.pack_codes(codes, bits), scale, bias

    def dequantize(packed, scale, bias, bits=4, group_size=64):
        codes = (
            (packed[..., None] >> (np.arange(32 // bits, dtype=np.uint32) * bits))
            & (2**bits - 1)
        ).reshape(*packed.shape[:-1], -1)
        return (
            codes.reshape(*scale.shape, group_size) * scale[..., None] + bias[..., None]
        ).reshape(*codes.shape)

    mx.quantize = quantize
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1024, 64)).astype(np.float32)
    x[:, 1:] += 0.8 * x[:, :1]
    router = rng.uniform(0.1, 1, len(x)).astype(np.float32)
    h, count = quant.gram([(x, router)], 64)
    weight = rng.normal(size=(32, 64)).astype(np.float32)
    packed, stats = quant.gptq(
        weight, h, 4, quant.QuantConfig(damping=0.01, min_tokens=0), count=count
    )
    fitted = dequantize(*packed)
    rtn = dequantize(*quantize(weight, bits=4, group_size=64))

    def loss(value):
        delta = weight - value
        return ((delta @ h) * delta).sum()

    assert loss(fitted) < loss(rtn)
    assert stats["tokens"] == 1024
    np.testing.assert_allclose(h, x.T @ (x * router[:, None]), rtol=2e-5, atol=2e-3)


@pytest.mark.parametrize("layers", [[0], [0, 1]], ids=["truncated", "full"])
def test_driver_transaction_resume_without_native_device(
    tmp_path, cpu_model, monkeypatch, layers
):
    """Exercise actual receipts and residual I/O with a deterministic layer seam."""
    from safetensors.numpy import load_file, save_file

    class FloatingRecipe(Recipe):
        def spec(self, name):
            return None

    source = write_source(
        tmp_path / "source",
        {
            "model.embed_tokens.weight": ("F32", np.ones((16, 64), np.float32)),
            "model.norm.weight": ("F32", np.ones(64, np.float32)),
            "model.layers.0.input_layernorm.weight": ("F32", np.ones(64, np.float32)),
            "model.layers.1.input_layernorm.weight": ("F32", np.ones(64, np.float32)),
        },
        asdict(tiny(hidden_size=64, mlp_layer_types=["dense", "dense"])),
    )
    adapter = Hy4Adapter(source, FloatingRecipe())
    mx = sys.modules["mlx.core"]
    mx.bfloat16 = "synthetic-bf16"
    mx.eval = lambda *args: None
    mx.clear_cache = lambda: None
    mx.save_safetensors = lambda path, values: save_file(values, path)
    mx.load = load_file
    monkeypatch.setattr(builder, "load_layer", lambda *args: object())
    monkeypatch.setattr(
        builder,
        "initial_hidden",
        lambda adapter, tokens, reader: reader.array("model.embed_tokens.weight")[
            tokens
        ][None],
    )
    windows = [np.array([1, 2, 3], np.int32)]
    evidence = {"english": ["hash"]}
    destination = tmp_path / "build"

    def interrupted(*args):
        raise RuntimeError("interrupted layer")

    monkeypatch.setattr(adapter, "forward", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        builder.build(adapter, destination, windows, evidence, layers)
    global_marker = destination / "build-state/global.json"
    assert global_marker.exists()
    assert not (destination / "build-state/layer-0.json").exists()
    committed_time = global_marker.stat().st_mtime_ns
    monkeypatch.setattr(adapter, "forward", lambda block, hidden, tokens: hidden + 1)
    report = builder.build(adapter, destination, windows, evidence, layers)
    assert global_marker.stat().st_mtime_ns == committed_time
    layer_marker = destination / "build-state/layer-0.json"
    receipt = json.loads(layer_marker.read_text())
    hidden = load_file(str(destination / receipt["boundaries"][0]))["hidden"]
    np.testing.assert_array_equal(hidden, np.full((1, 3, 64), 2, np.float32))
    index_path = destination / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    weight_map = json.loads(index_bytes)["weight_map"]
    shards = {p.name: load_file(str(p)) for p in destination.glob("model*.safetensors")}
    assert shards and set(shards) == set(weight_map.values())
    discovered = {
        name: filename for filename, values in shards.items() for name in values
    }
    assert discovered == weight_map
    assert set(weight_map) == {
        name for layer in [None, *layers] for name in adapter.names_for(layer)
    }
    assert json.loads((destination / "config.json").read_text())[
        "num_hidden_layers"
    ] == len(layers)
    shard_bytes = {name: (destination / name).read_bytes() for name in shards}
    monkeypatch.setattr(adapter, "forward", interrupted)
    assert builder.build(adapter, destination, windows, evidence, layers) == report
    assert index_path.read_bytes() == index_bytes
    assert {name: (destination / name).read_bytes() for name in shards} == shard_bytes
    with pytest.raises(ValueError, match="identity"):
        builder.build(
            adapter, destination, [np.array([4, 5], np.int32)], evidence, layers
        )
    with pytest.raises(ValueError, match="inside source"):
        builder.build(adapter, source / "bad", windows, evidence, layers)
    heldout = [np.array([4, 5], np.int32)]
    with pytest.raises(ValueError, match="corpus overlaps"):
        builder.quality(adapter, destination, heldout, evidence, tmp_path / "overlap")
    with pytest.raises(ValueError, match="token windows overlap"):
        builder.quality(
            adapter,
            destination,
            windows,
            {"code": ["heldout"]},
            tmp_path / "tokens-overlap",
        )

    def logit_seam(adapter, windows, work, layers, reader=None, check=lambda: None):
        work.mkdir(parents=True, exist_ok=True)
        for tokens in windows:
            result = np.zeros((len(tokens), 16), np.float32)
            result[:, 0] = 1 if reader is None else 0.9
            yield result

    monkeypatch.setattr(builder, "logits", logit_seam)
    ruler = builder.quality(
        adapter, destination, heldout, {"code": ["heldout"]}, tmp_path / "quality"
    )
    assert ruler["tokens"] == 2
    assert ruler["truncated"] == (len(layers) < adapter.args.num_hidden_layers)
    assert ruler["top1_agreement"] == 1 and ruler["kl_nats"] > 0
    native = TensorReader(destination)
    first = next(iter(native.index.tensors.values())).path
    with first.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="artifact"):
        builder.build(adapter, destination, windows, evidence, layers)


def test_artifact_locks_exclude_concurrent_writer(tmp_path):
    from scripts.streamed_gptq_io import artifact_lock

    with (
        artifact_lock(tmp_path),
        pytest.raises(RuntimeError, match="busy"),
        artifact_lock(tmp_path),
    ):
        pass
    with artifact_lock(tmp_path):
        pass


def test_disk_experts_match_resident_production_forward(cpu_model, monkeypatch):
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.hy_v4_flash", cpu_model)
    sys.modules["mlx.core"].eval = lambda *args: None
    model = cpu_model.Model(tiny())
    adapter = object.__new__(Hy4Adapter)
    adapter.args = model.args
    bank = model.layers[1].mlp.switch_mlp
    tensors = {
        p: getattr(bank, p).weight for p in ("gate_proj", "up_proj", "down_proj")
    }
    adapter.tensor = lambda name, expert: tensors[name.split(".")[-2]][expert]
    tokens = np.array([[1, 2, 3, 4, 5]], np.int32)
    expected = model(tokens)
    model.layers[1].mlp.switch_mlp = builder.disk_switch(adapter, 1, None)
    np.testing.assert_allclose(model(tokens), expected, atol=1e-5, rtol=1e-5)
