# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for short-forward Qwen4 eager dispatch."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.models import qwen4_exp

ROOT = Path(__file__).resolve().parents[1]


def test_eager_dispatch_default_and_small_row_gate(monkeypatch):
    previous = qwen4_exp.set_qwen4_eager_dispatch(True)
    try:
        monkeypatch.setattr(qwen4_exp, "_EAGER_DISPATCH_MAX_ROWS", 64)
        assert qwen4_exp.qwen4_eager_dispatch_admitted(1, 1)
        assert qwen4_exp.qwen4_eager_dispatch_admitted(4, 16)
        assert not qwen4_exp.qwen4_eager_dispatch_admitted(1, 65)
    finally:
        qwen4_exp.set_qwen4_eager_dispatch(previous)


def test_eager_dispatch_runtime_disable_and_validation():
    previous = qwen4_exp.set_qwen4_eager_dispatch(False)
    try:
        assert not qwen4_exp.qwen4_eager_dispatch_admitted(1, 1)
        with pytest.raises(TypeError, match="state must be a boolean"):
            qwen4_exp.set_qwen4_eager_dispatch(1)
        with pytest.raises(TypeError, match="dimensions must be integers"):
            qwen4_exp.qwen4_eager_dispatch_admitted(True, 1)
        with pytest.raises(ValueError, match="dimensions must be positive"):
            qwen4_exp.qwen4_eager_dispatch_admitted(1, 0)
    finally:
        qwen4_exp.set_qwen4_eager_dispatch(previous)


def test_decoder_submits_every_admitted_layer_without_host_sync():
    tree = ast.parse(
        (ROOT / "vllm_mlx" / "models" / "qwen4_exp.py").read_text(encoding="utf-8")
    )
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpTextModel"
    )
    forward = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)]
    async_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mx"
        and node.func.attr == "async_eval"
    ]
    forbidden = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr in {"eval", "item", "tolist"}
    ]

    assert len(async_calls) == 1
    assert forbidden == []


def test_eager_decoder_forward_is_value_exact_and_submits_each_layer(monkeypatch):
    class _Layer:
        is_linear = False

        def __call__(self, hidden, **_kwargs):
            return hidden + 1

    model = qwen4_exp.Qwen4ExpTextModel.__new__(qwen4_exp.Qwen4ExpTextModel)
    object.__setattr__(model, "args", SimpleNamespace(hc_count=1))
    object.__setattr__(model, "embed_tokens", lambda inputs: inputs[..., None])
    object.__setattr__(model, "layers", [_Layer(), _Layer(), _Layer()])
    object.__setattr__(model, "hyper_connection_mixer", lambda hidden: hidden)

    original_async_eval = mx.async_eval
    submitted = []

    def _record_submission(value):
        submitted.append(value)
        return original_async_eval(value)

    monkeypatch.setattr(mx, "async_eval", _record_submission)
    previous_device = mx.default_device()
    previous_dispatch = qwen4_exp.set_qwen4_eager_dispatch(False)
    mx.set_default_device(mx.cpu)
    try:
        inputs = mx.array([[1]])
        embeddings = mx.array([[[2.0]]])
        reference = model(inputs, cache=[None] * 3, input_embeddings=embeddings)
        assert submitted == []

        qwen4_exp.set_qwen4_eager_dispatch(True)
        candidate = model(inputs, cache=[None] * 3, input_embeddings=embeddings)
        mx.eval(reference, candidate)

        assert len(submitted) == 3
        assert mx.array_equal(reference, candidate).item()
    finally:
        qwen4_exp.set_qwen4_eager_dispatch(previous_dispatch)
        mx.set_default_device(previous_device)
