from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.requires_mlx

import vllm_mlx.models.qwen4_exp as qwen4_exp
from vllm_mlx.kernels import qsa_indexed_splitk
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


def test_indexed_splitk_gate_is_opt_in_version_and_shape_qualified(monkeypatch):
    monkeypatch.delenv(qsa_indexed_splitk.ENABLE_ENV, raising=False)
    assert (
        qsa_indexed_splitk.indexed_splitk_decline_reason(
            3, 16_384, batch_size=1, mlx_version="0.32.2"
        )
        == "disabled"
    )
    monkeypatch.setenv(qsa_indexed_splitk.ENABLE_ENV, "1")
    assert (
        qsa_indexed_splitk.indexed_splitk_decline_reason(
            3, 16_384, batch_size=2, mlx_version="0.32.2"
        )
        == "batch size is not qualified"
    )
    assert (
        qsa_indexed_splitk.indexed_splitk_decline_reason(
            4, 65_536, batch_size=1, mlx_version="0.32.2"
        )
        == "query length is not qualified"
    )
    assert (
        qsa_indexed_splitk.indexed_splitk_decline_reason(
            2, 65_536, batch_size=1, mlx_version="0.32.2"
        )
        == "query length is not qualified"
    )
    assert (
        qsa_indexed_splitk.indexed_splitk_decline_reason(
            1, 65_535, batch_size=1, mlx_version="0.32.2"
        )
        == "physical KV below crossover"
    )
    assert (
        qsa_indexed_splitk.indexed_splitk_decline_reason(
            3, 16_384, batch_size=1, mlx_version="0.32.1"
        )
        == "unqualified MLX version 0.32.1"
    )
    monkeypatch.setattr(qsa_indexed_splitk.mx.metal, "is_available", lambda: True)
    assert (
        qsa_indexed_splitk.indexed_splitk_decline_reason(
            3,
            16_384,
            batch_size=1,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g15d",
        )
        is None
    )
    assert (
        qsa_indexed_splitk.indexed_splitk_decline_reason(
            3,
            16_384,
            batch_size=1,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g16s",
        )
        == "unqualified Metal architecture applegpu_g16s"
    )


def test_indexed_splitk_layout_gate_bounds_threads_and_dtype():
    assert qsa_indexed_splitk.indexed_splitk_layout_supported(
        query_heads=24, kv_heads=2, head_dim=256, dtype=mx.bfloat16
    )
    assert not qsa_indexed_splitk.indexed_splitk_layout_supported(
        query_heads=33, kv_heads=1, head_dim=256, dtype=mx.bfloat16
    )
    assert not qsa_indexed_splitk.indexed_splitk_layout_supported(
        query_heads=24, kv_heads=2, head_dim=255, dtype=mx.bfloat16
    )
    assert not qsa_indexed_splitk.indexed_splitk_layout_supported(
        query_heads=24, kv_heads=2, head_dim=256, dtype=mx.int32
    )


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="QSA indexed split-K kernel requires Metal"
)
def test_indexed_splitk_production_layout_matches_fp64_with_distinct_rows():
    rng = np.random.default_rng(3058)
    batch, query_heads, kv_heads = 1, 24, 2
    query_length, key_length, head_dim = 3, 128, 256
    block_size, block_topk = 4, 12
    queries_np = rng.normal(0, 0.25, (batch, query_heads, query_length, head_dim))
    keys_np = rng.normal(0, 0.25, (batch, kv_heads, key_length, head_dim))
    values_np = rng.normal(0, 0.25, (batch, kv_heads, key_length, head_dim))
    starts_np = np.empty((batch, query_length, block_topk), dtype=np.int32)
    tails_np = np.empty((batch, query_length, block_size), dtype=np.int32)
    tail_counts_np = np.array([[1, 3, 4]], dtype=np.int32)
    for query_index in range(query_length):
        blocks = np.sort(
            rng.choice(key_length // block_size, size=block_topk, replace=False)
        )
        starts_np[0, query_index] = blocks * block_size
        tails_np[0, query_index] = rng.choice(
            key_length, size=block_size, replace=False
        )

    queries = mx.array(queries_np).astype(mx.bfloat16)
    keys = mx.array(keys_np).astype(mx.bfloat16)
    values = mx.array(values_np).astype(mx.bfloat16)
    output = qsa_indexed_splitk.indexed_splitk_attention(
        queries,
        keys,
        values,
        mx.array(starts_np),
        mx.full((batch, query_length), block_topk, dtype=mx.int32),
        mx.array(tails_np),
        mx.array(tail_counts_np),
        block_size=block_size,
        scale=head_dim**-0.5,
        splits=32,
    )
    mx.eval(output)

    queries_fp = np.array(queries.astype(mx.float32))
    keys_fp = np.array(keys.astype(mx.float32))
    values_fp = np.array(values.astype(mx.float32))
    reference = np.zeros(output.shape, dtype=np.float64)
    heads_per_kv = query_heads // kv_heads
    for query_index in range(query_length):
        selected = np.concatenate(
            [
                *(
                    np.arange(start, start + block_size)
                    for start in starts_np[0, query_index]
                ),
                tails_np[0, query_index, : tail_counts_np[0, query_index]],
            ]
        )
        for head in range(query_heads):
            kv_head = head // heads_per_kv
            scores = (
                keys_fp[0, kv_head, selected].astype(np.float64)
                @ queries_fp[0, head, query_index].astype(np.float64)
            ) * (head_dim**-0.5)
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            reference[0, head, query_index] = (
                weights[:, None] * values_fp[0, kv_head, selected].astype(np.float64)
            ).sum(axis=0)

    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)), reference, rtol=8e-2, atol=7e-4
    )


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="QSA indexed split-K kernel requires Metal"
)
def test_indexed_splitk_is_bit_exact_with_native_gather_path():
    mx.random.seed(3058)
    rng = np.random.default_rng(8)
    batch, query_heads, kv_heads = 1, 24, 2
    query_length, key_length, head_dim = 3, 4096, 256
    block_size, block_topk = 4, 512
    scale = head_dim**-0.5
    # Deliberately slice padded capacity so pass 1 must honor input strides
    # instead of copying the physical KV cache on every decode layer.
    queries = mx.random.normal((batch, query_heads, query_length + 1, head_dim)).astype(
        mx.bfloat16
    )[:, :, :query_length, :]
    keys = mx.random.normal((batch, kv_heads, key_length + 7, head_dim)).astype(
        mx.bfloat16
    )[:, :, :key_length, :]
    values = mx.random.normal((batch, kv_heads, key_length + 7, head_dim)).astype(
        mx.bfloat16
    )[:, :, :key_length, :]
    starts = []
    gathered_outputs = []
    for query_index in range(query_length):
        row_starts = (
            np.sort(
                rng.choice(key_length // block_size, size=block_topk, replace=False)
            ).astype(np.int32)
            * block_size
        )
        starts.append(row_starts)
        indices = np.concatenate(
            [np.arange(start, start + block_size) for start in row_starts]
        )
        gathered_outputs.append(
            mx.fast.scaled_dot_product_attention(
                queries[:, :, query_index : query_index + 1],
                mx.take(keys, mx.array(indices), axis=2),
                mx.take(values, mx.array(indices), axis=2),
                scale=scale,
            )
        )
    gathered = mx.concatenate(gathered_outputs, axis=2)
    padded_starts = mx.full((batch, query_length, block_topk + 1), -1, dtype=mx.int32)
    padded_starts[:, :, :block_topk] = mx.array(np.array(starts, dtype=np.int32)[None])
    padded_counts = mx.full((batch, query_length + 1), block_topk, dtype=mx.int32)
    padded_tails = mx.zeros((batch, query_length, block_size + 1), dtype=mx.int32)
    padded_tail_counts = mx.zeros((batch, query_length + 1), dtype=mx.int32)
    indexed = qsa_indexed_splitk.indexed_splitk_attention(
        queries,
        keys,
        values,
        padded_starts[:, :, :block_topk],
        padded_counts[:, :query_length],
        padded_tails[:, :, :block_size],
        padded_tail_counts[:, :query_length],
        block_size=block_size,
        scale=scale,
    )
    mx.eval(indexed, gathered)
    np.testing.assert_array_equal(
        np.array(indexed.astype(mx.float32)),
        np.array(gathered.astype(mx.float32)),
    )


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="QSA indexed split-K kernel requires Metal"
)
def test_indexed_splitk_empty_or_malformed_selection_returns_zero():
    output = qsa_indexed_splitk.indexed_splitk_attention(
        mx.zeros((1, 2, 1, 32), dtype=mx.float16),
        mx.zeros((1, 1, 8, 32), dtype=mx.float16),
        mx.ones((1, 1, 8, 32), dtype=mx.float16),
        mx.array([[[2**31 - 1]]], dtype=mx.int32),
        mx.ones((1, 1), dtype=mx.int32),
        mx.array([[[-1, 99]]], dtype=mx.int32),
        mx.full((1, 1), 2, dtype=mx.int32),
        block_size=2,
        scale=32**-0.5,
        splits=32,
    )
    mx.eval(output)
    np.testing.assert_array_equal(
        np.array(output.astype(mx.float32)), np.zeros(output.shape)
    )


class _FakeIndexer(qwen4_exp.QSAIndexer):
    def __call__(
        self,
        hidden_states,
        cache=None,
        *,
        physical_kv_length=None,
        record_rollback=False,
    ):
        del cache, record_rollback
        length = int(hidden_states.shape[1])
        block_size = self.compress_ratio
        block_count = self.token_budget // block_size
        indices = mx.arange(self.token_budget + block_size, dtype=mx.int32)
        return qwen4_exp._QSASelection(
            token_indices=mx.broadcast_to(indices, (1, length, indices.size)),
            block_valid=mx.ones((1, length, block_count), dtype=mx.bool_),
            tail_valid=mx.ones((1, length, block_size), dtype=mx.bool_),
            physical_kv_length=int(physical_kv_length),
        )


class _FakeKVCache:
    offset = 16_381
    _idx = 16_381

    def size(self):
        return self._idx

    def update_and_fetch(self, keys, values):
        return keys, values


def test_qsa_attention_routes_narrow_selection_and_records_receipt(monkeypatch):
    args = _args()
    attention = QSAAttention(args)
    attention.eval()
    attention.indexer = _FakeIndexer(args)
    monkeypatch.setattr(
        qwen4_exp, "indexed_splitk_decline_reason", lambda *a, **k: None
    )
    observed = []

    def fake_indexed(
        queries,
        keys,
        values,
        block_starts,
        block_counts,
        tail_indices,
        tail_counts,
        *,
        block_size,
        scale,
    ):
        del keys, values
        observed.append(
            (
                block_starts.shape,
                block_counts.shape,
                tail_indices.shape,
                tail_counts.shape,
                block_size,
                scale,
            )
        )
        return mx.zeros_like(queries)

    monkeypatch.setattr(qwen4_exp, "indexed_splitk_attention", fake_indexed)
    monkeypatch.setattr(
        qwen4_exp,
        "block_sparse_attention",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("indexed split-K must win the narrow-route priority")
        ),
    )
    output = attention(
        mx.zeros((1, 3, args.hidden_size)),
        [_FakeKVCache(), object()],
        mask="causal",
    )
    mx.eval(output)
    assert output.shape == (1, 3, args.hidden_size)
    assert len(observed) == 1
    assert observed[0][:4] == ((1, 3, 4), (1, 3), (1, 3, 2), (1, 3))
    assert observed[0][4:] == (2, 256**-0.5)
    assert qwen4_exp.qwen4_qsa_indexed_splitk_stats(attention) == {
        "route_constructions": 1,
        "declines": 0,
        "decline_reasons": {},
    }


def test_disabled_sparse_routes_do_not_add_compaction_to_dense_graph(monkeypatch):
    args = _args()
    attention = QSAAttention(args)
    attention.eval()
    attention.indexer = _FakeIndexer(args)
    monkeypatch.delenv(qsa_indexed_splitk.ENABLE_ENV, raising=False)
    monkeypatch.delenv("RAPID_MLX_QSA_BLOCK_SPARSE", raising=False)
    monkeypatch.setattr(
        qwen4_exp.mx,
        "sort",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("disabled routes must not sort compact indices")
        ),
    )
    monkeypatch.setattr(
        qwen4_exp,
        "scaled_dot_product_attention",
        lambda queries, *a, **k: mx.zeros_like(queries),
    )
    output = attention(
        mx.zeros((1, 3, args.hidden_size)),
        [_FakeKVCache(), object()],
        mask="causal",
    )
    mx.eval(output)
    assert output.shape == (1, 3, args.hidden_size)
