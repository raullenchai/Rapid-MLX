"""Singleton residual carry enrollment, fallback, and cache contracts."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
from rapid_mlx import qwen35_residual_norm as fused


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_QWEN35_RESIDUAL_NORM", "1")


class Cache:
    lengths = left_padding = None


def test_call_keeps_cache_identity_and_position_ids(monkeypatch):
    cache = [Cache() for _ in range(40)]
    inputs = mx.array([[1]])
    positions = mx.array([[[564]]])
    hidden = mx.zeros((1, 1, 2048), mx.bfloat16)
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
def test_unsupported_calls_keep_original_contract(monkeypatch, case):
    inputs = mx.array([[1]])
    cache = [Cache() for _ in range(40)]
    owner = SimpleNamespace(training=False)
    kwargs = dict(cache=cache)
    if case == "wide":
        inputs = mx.array([[1, 2]])
    elif case == "embeddings":
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


@pytest.mark.parametrize("mode", ["mtp", "none-disabled", "other-device", "no-modules"])
def test_install_rejects_unqualified_serving(monkeypatch, mode):
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"})
    spec = "mtp" if mode == "mtp" else "none"
    if mode == "none-disabled":
        monkeypatch.setenv("RAPID_MLX_QWEN35_RESIDUAL_NORM", "0")
    if mode == "other-device":
        monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M4 Pro"})
    assert fused.install_qwen35_residual_norm(object(), spec_decode=spec) == 0


def test_install_is_model_local_and_idempotent(monkeypatch):
    import mlx.nn as nn

    class Backbone(nn.Module):
        def __call__(self, *a, **kw):
            return None

    enrolled, other = Backbone(), Backbone()
    original = Backbone.__call__
    model = SimpleNamespace(named_modules=lambda: [("backbone", enrolled)])
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"})
    monkeypatch.setattr(fused, "_qualified", lambda value: True)
    monkeypatch.setattr(fused, "_probe", lambda *a: True)
    assert fused.install_qwen35_residual_norm(model) == 1
    assert type(enrolled) is not Backbone
    assert type(other) is Backbone and Backbone.__call__ is original
    assert fused.install_qwen35_residual_norm(model) == 0


@pytest.mark.parametrize("outcome", [False, "raises"])
def test_probe_failure_leaves_model_original(monkeypatch, outcome):
    class Backbone:
        pass

    backbone = Backbone()
    model = SimpleNamespace(named_modules=lambda: [("backbone", backbone)])
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"})
    monkeypatch.setattr(fused, "_qualified", lambda value: True)

    def probe(*args):
        if outcome == "raises":
            raise ValueError("declined")
        return outcome

    monkeypatch.setattr(fused, "_probe", probe)
    assert fused.install_qwen35_residual_norm(model) == 0
    assert type(backbone) is Backbone
