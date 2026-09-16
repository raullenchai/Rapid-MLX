from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = [
    pytest.mark.requires_mlx,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]

from rapid_mlx.kernels import qsa_stage1


def _oracle(q, pooled, q_positions, *, topk, ratio):
    blocks = int(pooled.shape[1])
    scores = mx.matmul(q.transpose(0, 2, 1, 3), pooled.swapaxes(-1, -2)[:, None])
    scores = mx.sum(mx.maximum(scores.astype(mx.float32), 0), axis=1) / math.sqrt(
        q.shape[-1]
    )
    starts = mx.arange(blocks) * ratio
    valid = (starts + ratio - 1)[None, None, :] <= q_positions[..., None]
    scores = mx.where(valid, scores, -mx.inf)
    ids = mx.argpartition(scores, kth=blocks - topk, axis=-1)[..., -topk:]
    chosen = mx.put_along_axis(
        mx.zeros(valid.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1
    )
    return chosen & valid, valid


def _assert_selection_equal(q, pooled, q_positions, *, topk=4, ratio=4):
    native_ids = qsa_stage1.qsa_stage1_select(
        q,
        pooled,
        q_positions,
        block_topk=topk,
        compress_ratio=ratio,
    )
    eager, valid = _oracle(q, pooled, q_positions, topk=topk, ratio=ratio)
    native = (
        mx.put_along_axis(
            mx.zeros(valid.shape, dtype=mx.bool_),
            native_ids,
            mx.array(True),
            axis=-1,
        )
        & valid
    )
    mx.eval(native_ids, native, eager)
    np.testing.assert_array_equal(np.asarray(native), np.asarray(eager))
    return np.asarray(native_ids)


def test_stage1_gate_is_opt_in_and_machine_qualified(monkeypatch):
    monkeypatch.delenv(qsa_stage1.ENABLE_ENV, raising=False)
    assert (
        qsa_stage1.qsa_stage1_decline_reason(
            512,
            65_024,
            batch_size=1,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g15d",
        )
        == "disabled"
    )
    monkeypatch.setenv(qsa_stage1.ENABLE_ENV, "1")
    assert (
        qsa_stage1.qsa_stage1_decline_reason(
            512,
            65_024,
            batch_size=1,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g15d",
        )
        is None
    )
    assert (
        qsa_stage1.qsa_stage1_decline_reason(
            512,
            65_024,
            batch_size=1,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g14s",
        )
        is None
    )
    assert "batch size" in qsa_stage1.qsa_stage1_decline_reason(
        512,
        65_024,
        batch_size=2,
        mlx_version="0.32.2",
        metal_architecture="applegpu_g15d",
    )
    assert "query below" in qsa_stage1.qsa_stage1_decline_reason(
        63,
        65_024,
        batch_size=1,
        mlx_version="0.32.2",
        metal_architecture="applegpu_g15d",
    )
    assert "physical KV" in qsa_stage1.qsa_stage1_decline_reason(
        512,
        65_023,
        batch_size=1,
        mlx_version="0.32.2",
        metal_architecture="applegpu_g15d",
    )
    assert "unqualified MLX" in qsa_stage1.qsa_stage1_decline_reason(
        512,
        65_024,
        batch_size=1,
        mlx_version="0.32.1",
        metal_architecture="applegpu_g15d",
    )
    assert "unqualified Metal" in qsa_stage1.qsa_stage1_decline_reason(
        512,
        65_024,
        batch_size=1,
        mlx_version="0.32.2",
        metal_architecture="applegpu_g17s",
    )
    assert (
        qsa_stage1.qsa_stage1_decline_reason(
            512,
            65_024,
            batch_size=1,
            training=True,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g15d",
        )
        == "training"
    )
    monkeypatch.setattr(qsa_stage1.mx.metal, "is_available", lambda: False)
    assert (
        qsa_stage1.qsa_stage1_decline_reason(
            512,
            65_024,
            batch_size=1,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g15d",
        )
        == "Metal runtime unavailable"
    )


def test_stage1_runtime_receipts_and_metadata_fallbacks(monkeypatch):
    qsa_stage1._mlx_version.cache_clear()
    monkeypatch.setattr(qsa_stage1, "version", lambda _: "test-mlx")
    assert qsa_stage1._mlx_version() == "test-mlx"

    qsa_stage1._mlx_version.cache_clear()

    def missing(_: str):
        raise PackageNotFoundError

    monkeypatch.setattr(qsa_stage1, "version", missing)
    assert qsa_stage1._mlx_version() == "unknown"
    qsa_stage1._mlx_version.cache_clear()

    qsa_stage1._metal_architecture.cache_clear()
    monkeypatch.setattr(
        qsa_stage1.mx, "device_info", lambda: {"architecture": "test-metal"}
    )
    assert qsa_stage1._metal_architecture() == "test-metal"
    qsa_stage1._metal_architecture.cache_clear()

    receipt = qsa_stage1.qsa_stage1_kernel_cache_info()
    assert receipt.currsize >= 0
    assert receipt.maxsize == 32


def test_stage1_matches_eager_with_padding_tails_and_ties():
    rng = np.random.default_rng(7)
    q = mx.array(rng.normal(size=(2, 5, 3, 8)).astype(np.float16))
    pooled = mx.array(rng.normal(size=(2, 17, 8)).astype(np.float16))
    positions = mx.array([[-1, 3, 5, 15, 67], [3, 7, 11, 31, 63]], dtype=mx.int32)
    _assert_selection_equal(q, pooled, positions)

    zeros = mx.zeros((2, 3, 3, 8), dtype=mx.float16)
    tie_positions = mx.array([[3, 19, 67], [7, 23, 63]], dtype=mx.int32)
    ids = _assert_selection_equal(zeros, pooled, tie_positions)
    np.testing.assert_array_equal(ids[0, -1], np.array([13, 14, 15, 16]))
    np.testing.assert_array_equal(ids[1, -1], np.array([12, 13, 14, 15]))


def test_stage1_production_geometry_matches_eager_selection():
    rng = np.random.default_rng(17)
    q = mx.array(rng.normal(size=(1, 16, 4, 128)).astype(np.float16))
    pooled = mx.array(rng.normal(size=(1, 1024, 128)).astype(np.float16))
    positions = mx.arange(4096 - 16, 4096, dtype=mx.int32)[None, :]
    _assert_selection_equal(q, pooled, positions, topk=512, ratio=4)


def test_stage1_preserves_activation_dtype_matmul_before_fp32_reduction(monkeypatch):
    q = mx.zeros((1, 2, 2, 8), dtype=mx.bfloat16)
    pooled = mx.zeros((1, 9, 8), dtype=mx.bfloat16)
    positions = mx.full((1, 2), 31, dtype=mx.int32)
    original = qsa_stage1.mx.matmul
    observed = []

    def record(left, right):
        observed.append((left.dtype, right.dtype))
        return original(left, right)

    monkeypatch.setattr(qsa_stage1.mx, "matmul", record)
    selected = qsa_stage1.qsa_stage1_select(
        q, pooled, positions, block_topk=4, compress_ratio=4
    )
    mx.eval(selected)
    assert observed == [(mx.bfloat16, mx.bfloat16)]


def test_stage1_rejects_unsupported_shapes():
    q = mx.zeros((1, 2, 4, 8), dtype=mx.float16)
    pooled = mx.zeros((1, 5, 8), dtype=mx.float16)
    positions = mx.zeros((1, 2), dtype=mx.int32)
    assert not qsa_stage1.qsa_stage1_supported(
        q, pooled, positions, block_topk=8, compress_ratio=4
    )
    with pytest.raises(ValueError, match="unsupported"):
        qsa_stage1.qsa_stage1_select(
            q, pooled, positions, block_topk=8, compress_ratio=4
        )


def test_stage1_support_gate_rejects_each_static_invariant(monkeypatch):
    q = mx.zeros((1, 2, 4, 8), dtype=mx.float16)
    pooled = mx.zeros((1, 9, 8), dtype=mx.float16)
    positions = mx.zeros((1, 2), dtype=mx.int32)

    monkeypatch.setattr(qsa_stage1, "qsa_stage1_kernel_available", lambda: False)
    assert not qsa_stage1.qsa_stage1_supported(
        q, pooled, positions, block_topk=4, compress_ratio=4
    )
    monkeypatch.setattr(qsa_stage1, "qsa_stage1_kernel_available", lambda: True)

    invalid = [
        (q.reshape(2, 4, 8), pooled, positions),
        (q, pooled, mx.zeros((1, 1), dtype=mx.int32)),
        (q, mx.zeros((2, 9, 8), dtype=mx.float16), positions),
        (q, mx.zeros((1, 9, 7), dtype=mx.float16), positions),
        (q.astype(mx.int32), pooled, positions),
        (q, pooled, positions.astype(mx.uint32)),
    ]
    for invalid_q, invalid_pooled, invalid_positions in invalid:
        assert not qsa_stage1.qsa_stage1_supported(
            invalid_q,
            invalid_pooled,
            invalid_positions,
            block_topk=4,
            compress_ratio=4,
        )

    with pytest.raises(ValueError, match="threadgroup width"):
        qsa_stage1._stage1_kernel(1, 1, 2048, 4, mx.float32, mx.float32)
