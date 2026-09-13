# SPDX-License-Identifier: Apache-2.0
"""Contracts for Qwen3.5/3.6 eager decoder-layer submission."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from vllm_mlx.patches import qwen3_5_eager_dispatch as eager


@pytest.fixture
def isolated_patch(monkeypatch):
    from mlx_lm.models import qwen3_5 as q

    was_installed = eager.is_installed()
    eager.uninstall_qwen3_5_eager_dispatch()
    real_decoder_layer = q.DecoderLayer
    saved_attributes = {
        name: getattr(q, name)
        for name in (
            "_RAPID_MLX_ORIG_DECODER_LAYER_CALL",
            "_RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED",
        )
        if hasattr(q, name)
    }
    for name in (
        "_RAPID_MLX_ORIG_DECODER_LAYER_CALL",
        "_RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED",
    ):
        if hasattr(q, name):
            delattr(q, name)

    class StubDecoderLayer:
        def __init__(self):
            self.input_layernorm = types.SimpleNamespace(
                weight=types.SimpleNamespace(shape=(2048,))
            )
            self.mlp = types.SimpleNamespace(num_experts=256, top_k=8)

        def __call__(self, output):
            return output

    q.DecoderLayer = StubDecoderLayer
    monkeypatch.setattr(eager, "_ENABLED", True)
    eager.install_qwen3_5_eager_dispatch()
    try:
        yield q, StubDecoderLayer
    finally:
        eager.uninstall_qwen3_5_eager_dispatch()
        q.DecoderLayer = real_decoder_layer
        for name in (
            "_RAPID_MLX_ORIG_DECODER_LAYER_CALL",
            "_RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED",
        ):
            if hasattr(q, name):
                delattr(q, name)
        for name, value in saved_attributes.items():
            setattr(q, name, value)
        eager._INSTALLED = False
        if was_installed:
            eager.install_qwen3_5_eager_dispatch()


def test_narrow_decode_and_verify_slabs_are_submitted(isolated_patch, monkeypatch):
    q, layer_type = isolated_patch
    submitted = []
    monkeypatch.setattr(mx, "async_eval", lambda value: submitted.append(value))

    layer = layer_type()
    decode = mx.zeros((1, 1, 8))
    verify = mx.zeros((4, 8, 8))
    assert q.DecoderLayer.__call__(layer, decode) is decode
    assert q.DecoderLayer.__call__(layer, verify) is verify
    assert submitted == [decode, verify]


def test_wide_prefill_is_not_submitted(isolated_patch, monkeypatch):
    q, layer_type = isolated_patch
    submitted = []
    monkeypatch.setattr(mx, "async_eval", lambda value: submitted.append(value))

    wide = mx.zeros((1, 65, 8))
    assert q.DecoderLayer.__call__(layer_type(), wide) is wide
    assert submitted == []


def test_unqualified_family_shape_is_not_submitted(isolated_patch, monkeypatch):
    q, layer_type = isolated_patch
    submitted = []
    monkeypatch.setattr(mx, "async_eval", lambda value: submitted.append(value))

    layer = layer_type()
    layer.mlp.num_experts = 128
    output = mx.zeros((1, 1, 8))
    assert q.DecoderLayer.__call__(layer, output) is output
    assert submitted == []


def test_explicit_opt_out_preserves_stock_submission(isolated_patch, monkeypatch):
    q, layer_type = isolated_patch
    submitted = []
    monkeypatch.setattr(mx, "async_eval", lambda value: submitted.append(value))
    monkeypatch.setattr(eager, "_ENABLED", False)

    output = mx.zeros((1, 1, 8))
    assert q.DecoderLayer.__call__(layer_type(), output) is output
    assert submitted == []


def test_non_array_like_result_fails_open(isolated_patch, monkeypatch):
    q, layer_type = isolated_patch
    submitted = []
    monkeypatch.setattr(mx, "async_eval", lambda value: submitted.append(value))

    output = types.SimpleNamespace()
    assert q.DecoderLayer.__call__(layer_type(), output) is output
    assert submitted == []


def test_install_is_idempotent_and_uninstall_restores_original(isolated_patch):
    q, _layer_type = isolated_patch
    original = q._RAPID_MLX_ORIG_DECODER_LAYER_CALL
    installed = q.DecoderLayer.__call__
    eager.install_qwen3_5_eager_dispatch()
    assert q.DecoderLayer.__call__ is installed

    eager.uninstall_qwen3_5_eager_dispatch()
    assert q.DecoderLayer.__call__ is original


def test_install_adopts_existing_process_hook(isolated_patch):
    q, _layer_type = isolated_patch
    installed = q.DecoderLayer.__call__
    eager.uninstall_qwen3_5_eager_dispatch()
    q.DecoderLayer.__call__ = installed
    q._RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED = True

    eager.install_qwen3_5_eager_dispatch()

    assert eager.is_installed()
    assert q.DecoderLayer.__call__ is installed


def test_install_fails_open_when_decoder_layer_is_unavailable(
    isolated_patch, monkeypatch
):
    q, _layer_type = isolated_patch
    eager.uninstall_qwen3_5_eager_dispatch()
    monkeypatch.delattr(q, "DecoderLayer")

    eager.install_qwen3_5_eager_dispatch()

    assert not eager.is_installed()
