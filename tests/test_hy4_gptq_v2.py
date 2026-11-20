"""Tiny NumPy contracts for opt-in v2; never initialize an MLX device."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import hy4_gptq as builder
from scripts.hy4_gptq_adapter import Hy4Adapter, Recipe
from scripts.hy4_gptq_v2 import (
    V2Recipe,
    budget_windows,
    coverage,
    estimate_sizes,
    low_token_policy,
    parse_bits_map,
    quantize_experts,
    routing_weights,
)
from tests.test_hy4_engram_offload import loader as loader
from tests.test_hy4_gptq import write_source
from tests.test_hy_v4_flash import cpu_model as cpu_model


def name(layer, projection):
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}.weight"


@pytest.mark.parametrize("layer", [1, 19, 20, 26, 27, 41])
@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_bits_map_defaults_and_legacy_identity(layer, projection):
    key = name(layer, projection)
    expected = 6 if projection == "down_proj" or 20 <= layer <= 26 else 4
    assert (
        V2Recipe().spec(key)
        == Recipe("hy4-gptq-v2").spec(key)
        == (expected, 64, "gptq")
    )
    for recipe_name in ("hy4-gptq-v1", "hy4-gptq-engram8"):
        recipe = Recipe(recipe_name)
        assert asdict(recipe) == dict(
            name=recipe_name, keep_embeddings_bf16=False, gate_up_3bit_layers=()
        )
        assert recipe.spec(key) == (4, 64, "gptq")


def test_map_cli_json_and_config_overrides(tmp_path):
    raw = '{"1": {"down_proj": 8, "up_proj": 3}}'
    path = tmp_path / "bits.json"
    path.write_text(raw)
    overrides = parse_bits_map(raw)
    assert overrides == parse_bits_map(str(path))
    recipe = V2Recipe(bits_map=overrides)
    tensors = {
        name(1, p): ("F32", np.ones((2, 64, 64), np.float32))
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    source = write_source(tmp_path / "source", tensors)
    adapter = Hy4Adapter(source, recipe)
    store = builder.NativeStore(tmp_path / "out", adapter, [0, 1])
    for key in tensors:
        store.entries(key)
    q = store.config()["quantization"]
    assert q["bits"] == 8
    for projection, bits in (("gate_proj", 4), ("up_proj", 3), ("down_proj", 8)):
        # Exact prefix and dictionary consumed by the Hy4 load predicate.
        assert q[name(1, projection).removesuffix(".weight")] == dict(
            bits=bits, group_size=64, mode="affine"
        )
    sizes = estimate_sizes(adapter, [0, 1])
    assert sizes["disk_delta_bytes"] == 2 * 64 * 64 * (8 + 3 + 4 - 12) // 8
    assert sizes["resident_delta_bytes"] == sizes["disk_delta_bytes"]
    assert sizes["resident_weight_target_met"]


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"-1":{"up_proj":6}}',
        '{"1":{"oops":6}}',
        '{"1":{"up_proj":5}}',
        '{"1":{"up_proj":true}}',
    ],
)
def test_invalid_maps(raw):
    with pytest.raises(ValueError):
        parse_bits_map(raw)


def test_larger_budget_caps_and_exhaustion():
    records = [np.arange(40), np.arange(40) + 100]
    windows, report = budget_windows(records, 4)
    assert report["v1_tokens"] == 8
    assert sum(map(len, windows)) == 32
    np.testing.assert_array_equal(np.concatenate(windows[::2]), np.arange(16))
    windows, report = budget_windows(records, 4, 11)
    assert list(map(len, windows)) == [4, 4, 3]
    windows, report = budget_windows(records, 4, 100)
    assert report["actual_tokens"] == 80 and report["source_exhausted"]
    assert not report["router_top_up"]


@pytest.mark.parametrize(
    ("count", "method", "coefficient"),
    [
        (0, "rtn", None),
        (31, "rtn", None),
        (32, "gptq", 0.2),
        (48, "gptq", 0.15),
        (64, "gptq", 0.1),
        (128, "gptq", 0.1),
    ],
)
def test_damping_ramp(count, method, coefficient):
    actual, damping = low_token_policy(count, 64)
    assert actual == method
    assert (
        damping == pytest.approx(coefficient)
        if coefficient is not None
        else damping is None
    )


@pytest.fixture
def quant(cpu_model, monkeypatch):
    mx = sys.modules["mlx.core"]
    mx.eval = lambda *args: None
    path = Path(__file__).parents[1] / "scripts/streamed_gptq_quant.py"
    spec = importlib.util.spec_from_file_location("_hy4_v2_quant_cpu", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_routing_squared_hessian_and_count(quant):
    x = np.array([[1, 2], [3, 4], [5, 6]], np.float32)
    weights = np.array([0.1, 0.5, 0.9], np.float32)
    for squared in (True, False):
        w = routing_weights(weights, squared)
        h, count = quant.gram([(x[:1], w[:1]), (x[1:], w[1:])], 2)
        np.testing.assert_allclose(
            h, x.T @ (x * (weights ** (2 if squared else 1))[:, None]), rtol=1e-6
        )
        assert count == 3  # raw routed count, independent of routing mass
    with pytest.raises(ValueError):
        routing_weights([-1], True)


def test_expert_build_rtn_dispatch_and_damping(quant, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "scripts.streamed_gptq_quant", quant)
    mx = sys.modules["mlx.core"]
    calls, written = [], []

    # Counts straddle the half-width cutoff; all three projections have width 64.
    def chunks(paths, expert, adapter, fitted):
        yield (
            np.ones((31 if expert == 0 else 48, 64), np.float32),
            np.full(31 if expert == 0 else 48, 0.5),
        )

    def gptq(weight, h, bits, config, *, count, check):
        assert config.damping == pytest.approx(0.15)
        np.testing.assert_allclose(h, np.full((64, 64), count * 0.25))
        calls.append((bits, count))
        return (weight, weight, weight), dict(tokens=count)

    monkeypatch.setattr(builder, "expert_chunks", chunks)
    monkeypatch.setattr(quant, "gptq", gptq)
    mx.quantize = lambda weight, **kw: (weight, weight, weight)
    mx.dequantize = lambda weight, *args, **kw: weight
    sink = SimpleNamespace(finish=lambda: None, close=lambda: None)
    adapter = SimpleNamespace(
        recipe=V2Recipe(),
        args=SimpleNamespace(n_routed_experts=2),
        tensor=lambda *a, **kw: np.ones((64, 64), np.float32),
    )
    store = SimpleNamespace(
        adapter=adapter, sink=lambda n: sink, put=lambda *a, **kw: written.append(a)
    )
    stats = quantize_experts(store, 20, [], lambda: None)
    assert len(written) == 6
    assert calls == [(6, 48)] * 3
    assert all(s["fallback"] == "low_token_rtn" for s in stats[0].values())
    assert coverage(stats, 40)["fraction"] == 0.5


def test_six_bit_gptq_native_packing(quant):
    mx = sys.modules["mlx.core"]

    def codec(weight, *, bits, group_size):
        grouped = weight.reshape(*weight.shape[:-1], -1, group_size)
        bias = grouped.min(-1)
        scale = np.maximum((grouped.max(-1) - bias) / (2**bits - 1), 1e-8)
        return None, scale, bias

    mx.quantize = codec
    weight = np.random.default_rng(1).normal(size=(2, 64)).astype(np.float32)
    triple, stats = quant.gptq(weight, np.eye(64, dtype=np.float32), 6, count=64)
    assert triple[0].shape == (2, 12)
    assert triple[1].shape == triple[2].shape == (2, 1)
    # Independently decode the little-endian bitstream, including word crossings.
    raw = triple[0].astype("<u4").view(np.uint8)
    codes = np.unpackbits(raw, axis=-1, bitorder="little").reshape(2, 64, 6)
    codes = (codes * (1 << np.arange(6))).sum(-1)
    reconstructed = codes * triple[1] + triple[2]
    assert np.max(abs(reconstructed - weight)) < 0.06


@pytest.mark.parametrize(
    ("recipe_name", "expected"),
    [
        (
            "hy4-gptq-v1",
            "2739f7be91472d4abbd38b3d19de86a48ad8ff90d55af462c57944879a842ab3",
        ),
        (
            "hy4-gptq-engram8",
            "fd4445b290c430ed931cd49fe970b4d8032b4555ffe9e2129ba617098bed3ff7",
        ),
    ],
)
def test_frozen_legacy_recipe_bytes(recipe_name, expected):
    """Baseline digest captured before v2 from the entire source naming fixture."""
    import json

    from scripts.streamed_gptq_io import digest

    names = json.loads(
        (
            Path(__file__).parent / "fixtures/hy_v4_flash/text_tensor_names.json"
        ).read_text()
    )
    recipe = Recipe(recipe_name)
    assert (
        digest(dict(recipe=asdict(recipe), specs={n: recipe.spec(n) for n in names}))
        == expected
    )


def test_calibration_v2_uses_offline_record_tails(tmp_path, monkeypatch, capsys):
    import json
    from types import ModuleType

    from scripts.streamed_gptq_io import atomic_json

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            assert kwargs == dict(local_files_only=True, trust_remote_code=False)
            return cls()

        def apply_chat_template(self, *args, **kwargs):
            return kwargs["reasoning_effort"]

        def encode(self, *args, **kwargs):
            return list(range(24))

    module = ModuleType("transformers")
    module.AutoTokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    slices = {}
    for key in ("english", "chinese", "code", "tools"):
        atomic_json(
            tmp_path / f"{key}.json", dict(messages=[dict(role="user", content=key)])
        )
        slices[key] = [f"{key}.json"]
    manifest = tmp_path / "slices.json"
    atomic_json(manifest, dict(slices=slices))
    legacy, evidence = builder.calibration(
        SimpleNamespace(source=tmp_path, recipe=Recipe()), manifest, 4
    )
    windows, v2_evidence = builder.calibration(
        SimpleNamespace(source=tmp_path, recipe=V2Recipe()), manifest, 4
    )
    assert evidence == v2_evidence
    assert sum(map(len, windows)) == 4 * sum(map(len, legacy)) == 128
    assert len(windows) == 32
    assert json.loads(capsys.readouterr().out)["calibration"]["actual_tokens"] == 128
    for before, after in zip(legacy, windows, strict=False):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("recipe_name", ["hy4-gptq-v1", "hy4-gptq-engram8"])
def test_legacy_expert_fit_keeps_weighting_and_low_token_policy(
    quant, monkeypatch, recipe_name
):
    monkeypatch.setitem(sys.modules, "scripts.streamed_gptq_quant", quant)
    mx = sys.modules["mlx.core"]
    calls = []

    def chunks(*args):
        yield np.ones((16, 64), np.float32), np.full(16, 0.5)

    def gptq(weight, h, bits, config, *, count, check):
        # Legacy still fits GPTQ below half-width, weights H by w, and uses 1.0.
        assert bits == 4 and count == 16
        np.testing.assert_allclose(h, np.full((64, 64), 8))
        assert config.low_token_damping == 1.0 and config.min_tokens == 2048
        calls.append(bits)
        return (weight, weight, weight), dict(tokens=count)

    monkeypatch.setattr(builder, "expert_chunks", chunks)
    monkeypatch.setattr(quant, "gptq", gptq)
    mx.dequantize = lambda weight, *args, **kwargs: weight
    sink = SimpleNamespace(finish=lambda: None, close=lambda: None)
    adapter = SimpleNamespace(
        recipe=Recipe(recipe_name),
        args=SimpleNamespace(n_routed_experts=1),
        tensor=lambda *a, **kw: np.ones((64, 64), np.float32),
    )
    store = SimpleNamespace(
        adapter=adapter, sink=lambda n: sink, put=lambda *a, **kw: None
    )
    builder.quantize_experts(store, 20, [], lambda: None)
    assert calls == [4, 4, 4]


def test_config_overrides_reach_hy4_loader(loader, monkeypatch, tmp_path):
    from scripts.streamed_gptq_io import atomic_json

    keys = [name(1, p) for p in ("gate_proj", "up_proj", "down_proj")]
    source = write_source(
        tmp_path / "source",
        {k: ("F32", np.ones((2, 64, 64), np.float32)) for k in keys},
    )
    adapter = Hy4Adapter(source, V2Recipe())
    store = builder.NativeStore(tmp_path / "candidate", adapter, [0, 1])
    tensors = {}
    for key in keys:
        tensors.update(
            {
                k: np.zeros(shape, np.uint32 if dtype == "U32" else np.uint16)
                for k, (dtype, shape) in store.entries(key).items()
            }
        )
    atomic_json(store.root / "config.json", store.config())
    (store.root / "model.safetensors").touch()

    class Model:
        def __init__(self, args):
            pass

        def sanitize(self, weights):
            return weights

        def load_weights(self, weights, strict):
            assert strict and set(dict(weights)) == set(tensors)

        def parameters(self):
            return tensors

        def eval(self):
            pass

    seen = {}

    def quantize(model, **kwargs):
        pred = kwargs["class_predicate"]
        for key in keys:
            prefix = key.removesuffix(".weight")
            seen[prefix] = pred(prefix, SimpleNamespace(to_quantized=True))
        assert not pred("model.norm", SimpleNamespace())

    monkeypatch.setattr(loader, "Model", Model)
    monkeypatch.setattr(loader.nn, "quantize", quantize, raising=False)
    monkeypatch.setattr(
        loader.mx, "load", lambda *args, **kwargs: dict(tensors), raising=False
    )
    loader.load(store.root, engram_ssd_offload=False)
    assert [seen[k.removesuffix(".weight")]["bits"] for k in keys] == [4, 4, 6]
    assert all(
        v == dict(bits=v["bits"], group_size=64, mode="affine") for v in seen.values()
    )


def test_v2_options_are_bound_to_identity():
    from scripts.streamed_gptq_io import digest

    default = digest(asdict(V2Recipe()))
    for kwargs in (
        dict(routing_weight_squared=False),
        dict(min_tokens=2048),
        dict(calibration_token_budget=1000),
        dict(bits_map=((20, "up_proj", 4),)),
    ):
        assert digest(asdict(V2Recipe(**kwargs))) != default
