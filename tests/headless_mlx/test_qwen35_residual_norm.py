"""Singleton residual carry enrollment, fallback, and cache contracts."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def fused(monkeypatch):
    """Load the real host logic with inert numerical dependencies on every OS."""
    dense = ModuleType("rapid_mlx.dense_decode_fusion")
    for name in ("_bits_equal", "_decode", "_norm_probe", "_probe_cache"):
        setattr(
            dense, name, Mock(side_effect=AssertionError("unexpected numerical call"))
        )
    monkeypatch.setitem(sys.modules, dense.__name__, dense)
    spec = importlib.util.spec_from_file_location(
        "rapid_mlx.qwen35_residual_norm",
        Path(__file__).resolve().parents[2] / "rapid_mlx/qwen35_residual_norm.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.mx = SimpleNamespace(bfloat16="bf16", int32="i32", device_info=lambda: {})

    # Installation only needs these cache class identities, never their tensors.
    api = SimpleNamespace(ArraysCache=type("ArraysCache", (), {}), KVCache=Cache)
    qwen = ModuleType("mlx_vlm.models.qwen3_5")
    qwen.language = api
    monkeypatch.setitem(sys.modules, qwen.__name__, qwen)
    from rapid_mlx.engine import batched

    monkeypatch.setattr(batched, "_qwen36_text_arrays_cache_type", lambda: Cache)
    return module


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_QWEN35_RESIDUAL_NORM", "1")


class Cache:
    lengths = left_padding = None


def test_call_keeps_cache_identity_and_position_ids(monkeypatch, fused):
    cache = [Cache() for _ in range(40)]
    inputs = SimpleNamespace(shape=(1, 1))
    positions = object()
    hidden = object()
    owner = SimpleNamespace(training=False, embed_tokens=lambda value: hidden)
    calls = []
    api = object()
    monkeypatch.setattr(fused, "_decode", lambda *args: calls.append(args) or hidden)
    call = fused._make_call(
        lambda *a, **k: pytest.fail("unexpected fallback"), api, {Cache}
    )
    assert call(owner, inputs, cache=cache, position_ids=positions) is hidden
    assert calls == [(owner, hidden, cache, "qwen-vlm", api, positions)]


@pytest.mark.parametrize(
    "case",
    [
        "wide",
        "embeddings",
        "embeddings-only",
        "missing-inputs",
        "mask",
        "capture",
        "sink",
        "extra",
        "training",
        "missing",
        "length",
        "unknown",
        "ragged",
        "padding",
        "disabled",
    ],
)
def test_unsupported_calls_keep_original_contract(monkeypatch, fused, case):
    inputs = SimpleNamespace(shape=(1, 1))
    cache = [Cache() for _ in range(40)]
    owner = SimpleNamespace(training=False)
    kwargs = dict(cache=cache)
    if case == "wide":
        inputs = SimpleNamespace(shape=(1, 2))
    elif case == "embeddings":
        kwargs["inputs_embeds"] = object()
    elif case in {"embeddings-only", "missing-inputs"}:
        inputs = None
        if case == "embeddings-only":
            kwargs["inputs_embeds"] = object()
    elif case == "mask":
        kwargs["mask"] = object()
    elif case == "capture":
        kwargs["capture_layer_ids"] = [0]
    elif case == "sink":
        kwargs["hidden_sink"] = []
    elif case == "extra":
        kwargs["future_argument"] = object()
    elif case == "training":
        owner.training = True
    elif case == "missing":
        kwargs["cache"] = None
    elif case == "length":
        kwargs["cache"] = cache[:1]
    elif case == "unknown":
        cache[1] = object()
    elif case == "ragged":
        cache[0].lengths = object()
    elif case == "padding":
        cache[0].left_padding = object()
    else:
        monkeypatch.setenv("RAPID_MLX_QWEN35_RESIDUAL_NORM", "0")
    calls = []
    sentinel = object()

    def original(self, value, **kw):
        calls.append((self, value, kw))
        return sentinel

    monkeypatch.setattr(fused, "_decode", lambda *a: pytest.fail("unexpected fusion"))
    call = fused._make_call(original, object(), {Cache})
    assert call(owner, inputs, **kwargs) is sentinel
    assert calls[0][0] is owner and calls[0][1] is inputs
    assert all(calls[0][2][k] is v for k, v in kwargs.items())


@pytest.mark.parametrize("mode", ["mtp", "none-disabled", "no-modules"])
def test_install_rejects_unqualified_serving(monkeypatch, fused, mode):
    monkeypatch.setattr(
        fused.mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"}
    )
    spec = "mtp" if mode == "mtp" else "none"
    if mode == "none-disabled":
        monkeypatch.setenv("RAPID_MLX_QWEN35_RESIDUAL_NORM", "0")
    assert fused.install_qwen35_residual_norm(object(), spec_decode=spec) == 0


@pytest.mark.parametrize(
    "chip,expected", [("Apple M3 Ultra", 1), ("Apple M4 Pro", 0), ("unknown", 0)]
)
def test_install_respects_device_policy_without_override(
    monkeypatch, fused, chip, expected
):
    from rapid_mlx.lattice import profiles

    class Backbone:
        def __call__(self, *args, **kwargs):
            return None

    backbone = Backbone()
    model = SimpleNamespace(named_modules=lambda: [("backbone", backbone)])
    probes = []
    monkeypatch.delenv("RAPID_MLX_QWEN35_RESIDUAL_NORM")
    monkeypatch.setattr(fused.mx, "device_info", lambda: {"device_name": chip})
    monkeypatch.setattr(profiles, "mlx_version", lambda: "0.32.4")
    monkeypatch.setattr(fused, "_qualified", lambda value: value is backbone)
    monkeypatch.setattr(
        fused, "_probe", lambda value, api: probes.append(value) or True
    )
    assert fused.install_qwen35_residual_norm(model) == expected
    assert probes == ([backbone] if expected else [])
    assert (type(backbone) is not Backbone) == bool(expected)


@pytest.mark.parametrize(
    "chip", ["Apple M3 Ultra", "Apple M4 Pro", "Apple M5 Max", "unknown"]
)
def test_install_is_model_local_and_idempotent(monkeypatch, fused, chip):
    class Backbone:
        def __call__(self, *a, **kw):
            return None

    enrolled, other = Backbone(), Backbone()
    original = Backbone.__call__
    model = SimpleNamespace(named_modules=lambda: [("backbone", enrolled)])
    monkeypatch.setattr(fused.mx, "device_info", lambda: {"device_name": chip})
    monkeypatch.setattr(fused, "_qualified", lambda value: True)
    monkeypatch.setattr(fused, "_probe", lambda *a: True)
    assert fused.install_qwen35_residual_norm(model) == 1
    assert type(enrolled) is not Backbone
    assert type(other) is Backbone and Backbone.__call__ is original
    assert fused.install_qwen35_residual_norm(model) == 0


@pytest.mark.parametrize("outcome", [False, "raises"])
def test_probe_failure_leaves_model_original(monkeypatch, fused, outcome):
    class Backbone:
        pass

    backbone = Backbone()
    model = SimpleNamespace(named_modules=lambda: [("backbone", backbone)])
    monkeypatch.setattr(
        fused.mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"}
    )
    monkeypatch.setattr(fused, "_qualified", lambda value: True)

    def probe(*args):
        if outcome == "raises":
            raise ValueError("declined")
        return outcome

    monkeypatch.setattr(fused, "_probe", probe)
    assert fused.install_qwen35_residual_norm(model) == 0
    assert type(backbone) is Backbone


@pytest.mark.parametrize(
    "change",
    [
        None,
        "module",
        "name",
        "hidden_size",
        "num_experts",
        "num_experts_per_tok",
        "layers",
        "dtype",
    ],
)
def test_qualification_requires_exact_backbone(fused, change):
    cls = type("Qwen3_5MoeModel", (), {})
    cls.__module__ = "mlx_vlm.models.qwen3_5_moe.language"
    backbone = cls()
    backbone.args = SimpleNamespace(
        hidden_size=2048, num_experts=256, num_experts_per_tok=8
    )
    backbone.layers = [
        SimpleNamespace(
            input_layernorm=SimpleNamespace(weight=SimpleNamespace(dtype="bf16"))
        )
        for _ in range(40)
    ]
    if change in {"module", "name"}:
        setattr(cls, "__" + change + "__", "other")
    elif change == "layers":
        backbone.layers.pop()
    elif change == "dtype":
        backbone.layers[-1].input_layernorm.weight.dtype = "fp16"
    elif change:
        setattr(backbone.args, change, 0)
    assert fused._qualified(backbone) is (change is None)


@pytest.mark.parametrize(
    "mismatch",
    [None, "norm", "output", "state-count", "state-path", "state-value", "offset"],
)
def test_probe_requires_output_and_cache_parity(monkeypatch, fused, mismatch):
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda states: [(str(i), v) for i, v in enumerate(states)]
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    monkeypatch.setattr(fused.mx, "array", lambda values, dtype: values, raising=False)
    evaluations = []
    monkeypatch.setattr(
        fused.mx, "eval", lambda *args: evaluations.append(args), raising=False
    )
    norms = [object(), object(), object()]
    checked = []
    monkeypatch.setattr(
        fused,
        "_norm_probe",
        lambda norm, dtype: checked.append((norm, dtype)) or mismatch != "norm",
    )
    monkeypatch.setattr(fused, "_bits_equal", lambda a, b: a == b)
    cache = [SimpleNamespace(state="prefill", offset=3)]
    monkeypatch.setattr(fused, "_probe_cache", lambda *a: cache)

    class Backbone:
        norm = norms[0]
        layers = [
            SimpleNamespace(input_layernorm=norms[1], post_attention_layernorm=norms[2])
        ]
        embed_tokens = staticmethod(lambda inputs: inputs)

        def __call__(self, inputs, *, cache):
            cache[0].state = "expected"
            cache[0].offset += len(inputs[0])
            return "output"

    def decode(owner, inputs, cache, family, api):
        assert inputs == [[4]] and family == "qwen-vlm"
        cache[0].state = "wrong" if mismatch == "state-value" else "expected"
        cache[0].offset += 2 if mismatch == "offset" else 1
        if mismatch == "state-count":
            cache.append(SimpleNamespace(state="extra", offset=0))
        return "wrong" if mismatch == "output" else "output"

    if mismatch == "state-path":
        paths = iter([[("left", "expected")], [("right", "expected")]])
        monkeypatch.setattr(utils, "tree_flatten", lambda states: next(paths))
    monkeypatch.setattr(fused, "_decode", decode)
    assert fused._probe(Backbone(), object()) is (mismatch is None)
    assert checked == [
        (norm, "bf16") for norm in (norms[:1] if mismatch == "norm" else norms)
    ]
    assert len(evaluations) == (0 if mismatch == "norm" else 2)
    # Both continuations must use copies, leaving the prefilled cache intact.
    assert cache[0].state == ("prefill" if mismatch == "norm" else "expected")
    assert cache[0].offset == (3 if mismatch == "norm" else 6)
