from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.requires_mlx

import vllm_mlx.models.qwen4_exp as qwen4_exp
from vllm_mlx.kernels import qsa_block_sparse
from vllm_mlx.models.qwen4_exp import QSAAttention, TextModelArgs


def _args(**overrides) -> TextModelArgs:
    values = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 3,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "hc_count": 4,
        "hc_lowrank": 3,
        "layer_types": ["full_attention"],
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 64,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "ple_layer_ids": [],
        "eos_token_id": 31,
    }
    values.update(overrides)
    return TextModelArgs(**values)


def test_qsa_gate_reports_each_decline_reason(monkeypatch):
    monkeypatch.delenv(qsa_block_sparse.ENABLE_ENV, raising=False)
    assert qsa_block_sparse.block_sparse_decline_reason(64, 16_384) == "disabled"

    monkeypatch.setenv(qsa_block_sparse.ENABLE_ENV, "1")
    assert (
        qsa_block_sparse.block_sparse_decline_reason(64, 16_384, training=True)
        == "training"
    )
    assert (
        qsa_block_sparse.block_sparse_decline_reason(63, 16_384)
        == "query below crossover"
    )
    assert (
        qsa_block_sparse.block_sparse_decline_reason(64, 16_383)
        == "physical KV below crossover"
    )
    monkeypatch.setattr(qsa_block_sparse.mx.metal, "is_available", lambda: False)
    assert (
        qsa_block_sparse.block_sparse_decline_reason(64, 16_384)
        == "Metal runtime unavailable"
    )


def test_qsa_layout_gate_covers_threadgroup_and_gqa_limits():
    assert qsa_block_sparse.block_sparse_layout_supported(
        query_heads=24,
        kv_heads=2,
        head_dim=256,
        block_size=4,
        dtype=mx.bfloat16,
    )
    assert not qsa_block_sparse.block_sparse_layout_supported(
        query_heads=65,
        kv_heads=1,
        head_dim=256,
        block_size=4,
        dtype=mx.bfloat16,
    )
    assert not qsa_block_sparse.block_sparse_layout_supported(
        query_heads=24,
        kv_heads=2,
        head_dim=256,
        block_size=128,
        dtype=mx.bfloat16,
    )
    assert not qsa_block_sparse.block_sparse_layout_supported(
        query_heads=24,
        kv_heads=0,
        head_dim=256,
        block_size=4,
        dtype=mx.bfloat16,
    )
    assert not qsa_block_sparse.block_sparse_layout_supported(
        query_heads=24,
        kv_heads=2,
        head_dim=255,
        block_size=4,
        dtype=mx.bfloat16,
    )
    assert not qsa_block_sparse.block_sparse_layout_supported(
        query_heads=24,
        kv_heads=2,
        head_dim=256,
        block_size=4,
        dtype=mx.int32,
    )


def _valid_kernel_inputs(*, query_heads=2, kv_heads=1, head_dim=32, dtype=mx.float16):
    queries = mx.zeros((1, query_heads, 1, head_dim), dtype=dtype)
    keys = mx.zeros((1, kv_heads, 2, head_dim), dtype=dtype)
    return {
        "queries": queries,
        "keys": keys,
        "values": mx.zeros_like(keys),
        "block_starts": mx.zeros((1, 1, 1), dtype=mx.int32),
        "block_counts": mx.ones((1, 1), dtype=mx.int32),
        "tail_indices": mx.zeros((1, 1, 2), dtype=mx.int32),
        "tail_counts": mx.zeros((1, 1), dtype=mx.int32),
        "block_size": 2,
    }


def test_qsa_selection_rejects_invalid_structural_validity():
    indices = mx.zeros((1, 2, 10), dtype=mx.int32)
    with pytest.raises(ValueError, match="rank three"):
        qwen4_exp._QSASelection(
            token_indices=mx.zeros((2, 10), dtype=mx.int32),
            block_valid=mx.ones((1, 2, 4), dtype=mx.bool_),
            tail_valid=mx.ones((1, 2, 2), dtype=mx.bool_),
            physical_kv_length=10,
        )
    with pytest.raises(ValueError, match="block validity"):
        qwen4_exp._QSASelection(
            token_indices=indices,
            block_valid=mx.ones((1, 1, 4), dtype=mx.bool_),
            tail_valid=mx.ones((1, 2, 2), dtype=mx.bool_),
            physical_kv_length=10,
        )
    with pytest.raises(ValueError, match="tail validity"):
        qwen4_exp._QSASelection(
            token_indices=indices,
            block_valid=mx.ones((1, 2, 4), dtype=mx.bool_),
            tail_valid=mx.ones((1, 1, 2), dtype=mx.bool_),
            physical_kv_length=10,
        )
    with pytest.raises(ValueError, match="block geometry"):
        qwen4_exp._QSASelection(
            token_indices=indices,
            block_valid=mx.ones((1, 2, 3), dtype=mx.bool_),
            tail_valid=mx.ones((1, 2, 2), dtype=mx.bool_),
            physical_kv_length=10,
        )
    with pytest.raises(ValueError, match="must be boolean"):
        qwen4_exp._QSASelection(
            token_indices=indices,
            block_valid=mx.ones((1, 2, 4), dtype=mx.int32),
            tail_valid=mx.ones((1, 2, 2), dtype=mx.bool_),
            physical_kv_length=10,
        )


def test_qsa_kernel_rejects_every_unsafe_shape_and_layout():
    def rejected(message, **overrides):
        inputs = _valid_kernel_inputs()
        inputs.update(overrides)
        with pytest.raises(ValueError, match=message):
            qsa_block_sparse.block_sparse_attention(**inputs)

    rejected("rank four", queries=mx.zeros((1, 1, 32)))
    rejected("positive", block_size=0)
    rejected("block starts", block_starts=mx.zeros((1, 2, 1), dtype=mx.int32))
    rejected("block counts", block_counts=mx.zeros((1, 2), dtype=mx.int32))
    rejected("tail indices", tail_indices=mx.zeros((1, 1, 1), dtype=mx.int32))
    rejected("tail counts", tail_counts=mx.zeros((1, 2), dtype=mx.int32))
    rejected("shapes are inconsistent", values=mx.zeros((1, 1, 3, 32)))
    rejected("same dtype", keys=mx.zeros((1, 1, 2, 32), dtype=mx.float32))
    for name in ("block_starts", "block_counts", "tail_indices", "tail_counts"):
        inputs = _valid_kernel_inputs()
        inputs[name] = inputs[name].astype(mx.int64)
        with pytest.raises(ValueError, match="must use int32"):
            qsa_block_sparse.block_sparse_attention(**inputs)

    inputs = _valid_kernel_inputs(head_dim=33)
    with pytest.raises(ValueError, match="divisible by 32"):
        qsa_block_sparse.block_sparse_attention(**inputs)

    inputs = _valid_kernel_inputs(kv_heads=0)
    with pytest.raises(ValueError, match="at least one KV head"):
        qsa_block_sparse.block_sparse_attention(**inputs)

    inputs = _valid_kernel_inputs(query_heads=3, kv_heads=2)
    with pytest.raises(ValueError, match="divisible by KV heads"):
        qsa_block_sparse.block_sparse_attention(**inputs)

    inputs = _valid_kernel_inputs(query_heads=33)
    with pytest.raises(ValueError, match="at most 32"):
        qsa_block_sparse.block_sparse_attention(**inputs)

    inputs = _valid_kernel_inputs(dtype=mx.int32)
    with pytest.raises(ValueError, match="layout is unsupported"):
        qsa_block_sparse.block_sparse_attention(**inputs)


class _FakeIndexer:
    token_budget = 8
    compress_ratio = 2
    rope_theta = 10_000_000

    def __call__(
        self,
        hidden_states,
        cache,
        *,
        physical_kv_length,
        record_rollback=False,
    ):
        del cache, record_rollback
        length = int(hidden_states.shape[1])
        # Deliberately unsorted to prove the kernel receives physical order.
        block_tokens = mx.broadcast_to(
            mx.array([8, 9, 0, 1, 4, 5, 12, 13], dtype=mx.int32),
            (1, length, 8),
        )
        tail = mx.broadcast_to(mx.array([16, 17], dtype=mx.int32), (1, length, 2))
        return qwen4_exp._QSASelection(
            token_indices=mx.concatenate([block_tokens, tail], axis=-1),
            block_valid=mx.ones((1, length, 4), dtype=mx.bool_),
            tail_valid=mx.ones((1, length, 2), dtype=mx.bool_),
            physical_kv_length=physical_kv_length,
        )


class _GappedFakeIndexer(_FakeIndexer):
    def __call__(
        self,
        hidden_states,
        cache,
        *,
        physical_kv_length,
        record_rollback=False,
    ):
        selection = super().__call__(
            hidden_states,
            cache,
            physical_kv_length=physical_kv_length,
            record_rollback=record_rollback,
        )
        length = int(hidden_states.shape[1])
        return qwen4_exp._QSASelection(
            token_indices=selection.token_indices,
            block_valid=mx.broadcast_to(
                mx.array([True, False, True, False]), (1, length, 4)
            ),
            tail_valid=mx.broadcast_to(mx.array([False, True]), (1, length, 2)),
            physical_kv_length=physical_kv_length,
        )


class _FakeKVCache:
    offset = 16_384
    _idx = 16_384

    def size(self):
        return self._idx

    def update_and_fetch(self, keys, values):
        return keys, values


def test_qsa_attention_routes_compact_selection_and_records_construction(monkeypatch):
    args = _args()
    attention = QSAAttention(args)
    attention.eval()
    attention.indexer = _FakeIndexer()
    monkeypatch.setenv(qsa_block_sparse.ENABLE_ENV, "1")
    observed = []

    def fake_sparse(
        queries,
        keys,
        values,
        block_starts,
        block_counts,
        tail_indices,
        tail_counts,
        *,
        block_size,
    ):
        del keys, values
        observed.append(
            (
                np.array(block_starts[0, 0]),
                np.array(block_counts),
                tail_indices.shape,
                np.array(tail_counts),
                block_size,
            )
        )
        return mx.zeros_like(queries)

    monkeypatch.setattr(qwen4_exp, "block_sparse_attention", fake_sparse)
    output = attention(
        mx.zeros((1, 64, args.hidden_size)),
        [_FakeKVCache(), object()],
        mask="causal",
    )
    mx.eval(output)

    assert output.shape == (1, 64, args.hidden_size)
    assert len(observed) == 1
    starts, counts, tail_shape, tail_counts, block_size = observed[0]
    np.testing.assert_array_equal(starts, [0, 4, 8, 12])
    np.testing.assert_array_equal(counts, 4)
    assert tail_shape == (1, 64, 2)
    np.testing.assert_array_equal(tail_counts, 2)
    assert block_size == 2
    assert qwen4_exp.qwen4_qsa_block_sparse_stats(attention) == {
        "route_constructions": 1,
        "declines": 0,
        "decline_reasons": {},
    }


def test_qsa_attention_compacts_gapped_validity_before_counted_dispatch(monkeypatch):
    args = _args()
    attention = QSAAttention(args)
    attention.eval()
    attention.indexer = _GappedFakeIndexer()
    monkeypatch.setenv(qsa_block_sparse.ENABLE_ENV, "1")
    observed = []

    def fake_sparse(
        queries,
        keys,
        values,
        block_starts,
        block_counts,
        tail_indices,
        tail_counts,
        *,
        block_size,
    ):
        del keys, values, block_size
        mx.eval(block_starts, block_counts, tail_indices, tail_counts)
        observed.append(
            (
                np.array(block_starts[0, 0]),
                np.array(block_counts),
                np.array(tail_indices[0, 0]),
                np.array(tail_counts),
            )
        )
        return mx.zeros_like(queries)

    monkeypatch.setattr(qwen4_exp, "block_sparse_attention", fake_sparse)
    output = attention(
        mx.zeros((1, 64, args.hidden_size)),
        [_FakeKVCache(), object()],
        mask="causal",
    )
    mx.eval(output)

    starts, block_counts, tails, tail_counts = observed[0]
    np.testing.assert_array_equal(starts, [4, 8, 16_448, 16_448])
    np.testing.assert_array_equal(block_counts, 2)
    np.testing.assert_array_equal(tails, [17, 16_448])
    np.testing.assert_array_equal(tail_counts, 1)


def test_qsa_attention_disabled_control_stays_dense(monkeypatch):
    args = _args()
    attention = QSAAttention(args)
    attention.eval()
    attention.indexer = _FakeIndexer()
    monkeypatch.delenv(qsa_block_sparse.ENABLE_ENV, raising=False)
    sparse_called = False
    dense_masks = []

    def fail_if_sparse(*_args, **_kwargs):
        nonlocal sparse_called
        sparse_called = True
        raise AssertionError("disabled route must not dispatch the sparse kernel")

    def fake_dense(queries, keys, values, *, cache, scale, mask):
        del keys, values, cache, scale
        dense_masks.append(mask)
        return mx.zeros_like(queries)

    monkeypatch.setattr(qwen4_exp, "block_sparse_attention", fail_if_sparse)
    monkeypatch.setattr(qwen4_exp, "scaled_dot_product_attention", fake_dense)
    output = attention(
        mx.zeros((1, 64, args.hidden_size)),
        [_FakeKVCache(), object()],
        mask="causal",
    )
    mx.eval(output, *dense_masks)

    assert not sparse_called
    assert len(dense_masks) == 1
    assert dense_masks[0].shape == (1, 1, 64, 16_448)
    assert qwen4_exp.qwen4_qsa_block_sparse_stats(attention) == {
        "route_constructions": 0,
        "declines": 1,
        "decline_reasons": {"disabled": 1},
    }


def test_qsa_attention_unsupported_layout_stays_dense(monkeypatch):
    args = _args()
    attention = QSAAttention(args)
    attention.eval()
    attention.indexer = _FakeIndexer()
    monkeypatch.setenv(qsa_block_sparse.ENABLE_ENV, "1")
    monkeypatch.setattr(qwen4_exp, "block_sparse_layout_supported", lambda **_: False)

    def fail_if_sparse(*_args, **_kwargs):
        raise AssertionError("unsupported layout must not dispatch the sparse kernel")

    def fake_dense(queries, keys, values, *, cache, scale, mask):
        del keys, values, cache, scale, mask
        return mx.zeros_like(queries)

    monkeypatch.setattr(qwen4_exp, "block_sparse_attention", fail_if_sparse)
    monkeypatch.setattr(qwen4_exp, "scaled_dot_product_attention", fake_dense)
    output = attention(
        mx.zeros((1, 64, args.hidden_size)),
        [_FakeKVCache(), object()],
        mask="causal",
    )
    mx.eval(output)

    assert qwen4_exp.qwen4_qsa_block_sparse_stats(attention) == {
        "route_constructions": 0,
        "declines": 1,
        "decline_reasons": {"unsupported layout": 1},
    }


def test_qsa_attention_construction_failure_surfaces_without_false_receipt(monkeypatch):
    args = _args()
    attention = QSAAttention(args)
    attention.eval()
    attention.indexer = _FakeIndexer()
    monkeypatch.setenv(qsa_block_sparse.ENABLE_ENV, "1")

    def fail_sparse(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(qwen4_exp, "block_sparse_attention", fail_sparse)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        attention(
            mx.zeros((1, 64, args.hidden_size)),
            [_FakeKVCache(), object()],
            mask="causal",
        )

    assert qwen4_exp.qwen4_qsa_block_sparse_stats(attention) == {
        "route_constructions": 0,
        "declines": 0,
        "decline_reasons": {},
    }


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="QSA block-sparse kernel requires Metal"
)
@pytest.mark.parametrize("bad_block_start", [-8, 99, 2**31 - 1])
@pytest.mark.parametrize("bad_count", [-1, 99])
def test_qsa_kernel_bounds_malformed_counts_and_indices_on_device(
    bad_block_start, bad_count
):
    inputs = _valid_kernel_inputs()
    inputs.update(
        block_starts=mx.array([[[bad_block_start]]], dtype=mx.int32),
        block_counts=mx.array([[bad_count]], dtype=mx.int32),
        tail_indices=mx.array([[[99, -1]]], dtype=mx.int32),
        tail_counts=mx.array([[bad_count]], dtype=mx.int32),
    )
    output = qsa_block_sparse.block_sparse_attention(**inputs)
    mx.eval(output)
    np.testing.assert_array_equal(np.array(output), np.zeros(output.shape))


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="QSA block-sparse kernel requires Metal"
)
def test_qsa_block_sparse_matches_fp64_reference():
    rng = np.random.default_rng(11)
    batch = 2
    query_heads = 4
    kv_heads = 2
    query_length = 2
    key_length = 10
    head_dim = 32
    block_size = 2
    queries_np = rng.normal(size=(batch, query_heads, query_length, head_dim)).astype(
        np.float16
    )
    keys_np = rng.normal(size=(batch, kv_heads, key_length, head_dim)).astype(
        np.float16
    )
    values_np = rng.normal(size=(batch, kv_heads, key_length, head_dim)).astype(
        np.float16
    )
    block_starts_np = np.array([[[0, 4], [2, 6]], [[2, 4], [0, 6]]], dtype=np.int32)
    block_counts_np = np.full((batch, query_length), 2, dtype=np.int32)
    tail_indices_np = np.array([[[8, 0], [8, 9]], [[7, 0], [8, 9]]], dtype=np.int32)
    tail_counts_np = np.array([[1, 2], [1, 2]], dtype=np.int32)

    output = qsa_block_sparse.block_sparse_attention(
        mx.array(queries_np),
        mx.array(keys_np),
        mx.array(values_np),
        mx.array(block_starts_np),
        mx.array(block_counts_np),
        mx.array(tail_indices_np),
        mx.array(tail_counts_np),
        block_size=block_size,
    )
    mx.eval(output)

    reference = np.zeros(output.shape, dtype=np.float64)
    heads_per_kv = query_heads // kv_heads
    for batch_index in range(batch):
        for query_index in range(query_length):
            selected = []
            for start in block_starts_np[batch_index, query_index]:
                selected.extend(range(int(start), int(start) + block_size))
            selected.extend(
                map(
                    int,
                    tail_indices_np[
                        batch_index,
                        query_index,
                        : tail_counts_np[batch_index, query_index],
                    ],
                )
            )
            selected_array = np.array(selected)
            for head in range(query_heads):
                kv_head = head // heads_per_kv
                scores = (
                    keys_np[batch_index, kv_head, selected_array].astype(np.float64)
                    @ queries_np[batch_index, head, query_index].astype(np.float64)
                ) / np.sqrt(head_dim)
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                reference[batch_index, head, query_index] = (
                    weights[:, None]
                    * values_np[batch_index, kv_head, selected_array].astype(np.float64)
                ).sum(axis=0)

    np.testing.assert_allclose(np.array(output), reference, rtol=3e-3, atol=2e-3)
