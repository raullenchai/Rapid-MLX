"""Synthetic independent-position decode contracts; no checkpoint access."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.models.deepseek_v41_native.cache import BatchModelCache
from scripts.bench_deepseek_v41_batched_decode import make_tiny_model


@pytest.fixture
def model():
    return make_tiny_model()


def prefill(model, lengths, capacities=None):
    caches, tokens = [], []
    capacities = capacities or [128 + i * 2 for i in range(len(lengths))]
    for row, (length, capacity) in enumerate(zip(lengths, capacities)):
        cache = model.make_cache(max_seq_len=capacity)
        prompt = ((mx.arange(length) * 3 + row * 7) % model.args.vocab_size)[
            None
        ].astype(mx.int32)
        logits = model(prompt, cache, last_logit_only=True)
        tokens.append(mx.argmax(logits[:, -1], axis=-1).reshape(1, 1))
        caches.append(cache)
    return caches, mx.concatenate(tokens)


def assert_cache_equal(actual, expected):
    assert actual.offset == expected.offset
    if actual.engram_ids is not None:
        np.testing.assert_array_equal(
            actual.engram_ids[:, : actual.offset],
            expected.engram_ids[:, : expected.offset],
        )
    for left, right in zip(actual.layers, expected.layers):
        for field in ("win_kv", "comp_kv", "index_k"):
            a, b = getattr(left, field), getattr(right, field)
            if a is None:
                assert b is None
                continue
            if field != "win_kv":
                count = actual.offset // left.ratio
                a, b = a[:, :count], b[:, :count]
            np.testing.assert_allclose(
                np.array(a), np.array(b), rtol=3e-5, atol=3e-6, err_msg=field
            )
        if left.comp_state is not None:
            for field in ("kv_state", "score_state"):
                np.testing.assert_allclose(
                    np.array(getattr(left.comp_state, field)),
                    np.array(getattr(right.comp_state, field)),
                    rtol=3e-5,
                    atol=3e-6,
                    err_msg=field,
                )


def step(model, batch, singles, tokens, *, greedy=True):
    actual = model(tokens, batch, last_logit_only=True)
    expected = mx.concatenate(
        [
            model(tokens[row : row + 1], cache, last_logit_only=True)[:, 0]
            for row, cache in enumerate(singles)
        ]
    )
    mx.eval(actual, expected)
    # FP32 GEMM reductions depend on batch size on Metal. Allow rounding
    # noise in logits while keeping greedy token equality exact below.
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), rtol=3e-5, atol=5e-5
    )
    selected = mx.argmax(actual, axis=-1)[:, None]
    if greedy:
        np.testing.assert_array_equal(
            np.array(selected), np.array(mx.argmax(expected, axis=-1)[:, None])
        )
    return selected


@pytest.mark.parametrize("eval_interval", [0, 1, 3])
def test_mixed_positions_greedy_64_steps(model, eval_interval):
    model.eval_interval = eval_interval
    # Empty compression, short windows, both sides of the window and group
    # boundaries; different sequence capacities must survive the merge.
    singles, tokens = prefill(model, [1, 2, 3, 4, 5, 8])
    batch = BatchModelCache.from_caches(singles)
    assert batch.max_seq_len == 138
    np.testing.assert_array_equal(batch.offsets, [1, 2, 3, 4, 5, 8])
    for i in range(64):
        tokens = step(model, batch, singles, tokens)
        if i in (0, 1, 3, 31, 63):
            for row, single in enumerate(singles):
                assert_cache_equal(batch.extract(row), single)
                for layer in batch.layers:
                    if layer.comp_counts is not None:
                        assert (
                            int(layer.comp_counts[row].item())
                            == single.offset // layer.ratio
                        )


def test_join_leave_extract_and_resume(model):
    singles, tokens = prefill(model, [1, 5])
    batch = BatchModelCache.from_caches(singles)
    for _ in range(4):
        tokens = step(model, batch, singles, tokens)
    arriving, arriving_tokens = prefill(model, [2, 9], capacities=[160, 180])
    batch.extend(BatchModelCache.from_caches(arriving))
    singles += arriving
    tokens = mx.concatenate([tokens, arriving_tokens])
    assert batch.max_seq_len == 180
    for _ in range(5):
        tokens = step(model, batch, singles, tokens)
    # Reorder while removing two finished/cancelled requests.
    keep = [3, 0]
    batch.filter(keep)
    singles = [singles[i] for i in keep]
    tokens = tokens[mx.array(keep)]
    for _ in range(8):
        tokens = step(model, batch, singles, tokens)
    for row, single in enumerate(singles):
        extracted = batch.extract(row)
        assert_cache_equal(extracted, single)
        a = model(tokens[row : row + 1], extracted, last_logit_only=True)
        b = model(tokens[row : row + 1], single, last_logit_only=True)
        np.testing.assert_allclose(np.array(a), np.array(b), rtol=3e-5, atol=5e-5)
    batch.filter([])
    assert len(batch) == 0
    batch.extend(BatchModelCache.from_caches(singles))
    assert len(batch) == 2
    assert_cache_equal(batch.extract(1), singles[1])


def test_mask_uncommitted_entries_and_partial_groups(model):
    singles, tokens = prefill(model, [1, 2, 7])
    batch = BatchModelCache.from_caches(singles)
    # Invalid slots deliberately contain attractive keys/values. Neither
    # top-k nor sparse attention may consume them for a shorter row.
    for layer in batch.layers:
        if layer.comp_kv is not None:
            for row, position in enumerate(batch.offsets):
                count = position // layer.ratio
                layer.comp_kv[row, int(count) :] = 1000
                layer.index_k[row, int(count) :] = 1000
    for _ in range(8):
        tokens = step(model, batch, singles, tokens)
    for row, single in enumerate(singles):
        assert_cache_equal(batch.extract(row), single)


def test_single_row_batch_and_capacity_guards(model):
    singles, tokens = prefill(model, [1], capacities=[2])
    batch = BatchModelCache.from_caches(singles)
    with pytest.raises(ValueError, match="rows, 1"):
        model(mx.ones((1, 2), dtype=mx.int32), batch)
    with pytest.raises(NotImplementedError, match="rollback"):
        model(tokens, batch, enable_rollback=True)
    with pytest.raises(NotImplementedError, match="rollback"):
        batch.rollback(0)
    assert batch.offsets.tolist() == [1]
    step(model, batch, singles, tokens)
    with pytest.raises(ValueError, match="capacity"):
        model(tokens, batch)
    with pytest.raises(ValueError, match="capacity"):
        BatchModelCache.from_caches(singles, max_seq_len=1)
    with pytest.raises(IndexError):
        batch.filter([1])
    with pytest.raises(ValueError, match="integer"):
        batch.filter([True])
    with pytest.raises(ValueError, match="single-row"):
        BatchModelCache.from_caches([model.make_cache(bsz=2)])
    with pytest.raises(ValueError, match="at least one"):
        BatchModelCache.from_caches([])


def test_merge_and_extract_do_not_mutate_source(model):
    singles, tokens = prefill(model, [2, 5])
    batch = BatchModelCache.from_caches(singles)
    before = batch.extract(0)
    model(tokens, batch, last_logit_only=True)
    assert_cache_equal(before, singles[0])
    extracted = batch.extract(0)
    snapshot = batch.extract(0)
    model(tokens[:1], extracted)
    assert_cache_equal(batch.extract(0), snapshot)


def test_one_weight_projection_per_layer_for_the_whole_batch(model, monkeypatch):
    singles, tokens = prefill(model, [1, 3, 7, 8])
    batch = BatchModelCache.from_caches(singles)
    import mlx.nn as nn

    from rapid_mlx.models.deepseek_v41_native.moe import MoE

    linear_calls, moe_calls = [], []
    linear_call, moe_call = nn.Linear.__call__, MoE.__call__

    def linear(self, x):
        linear_calls.append((id(self), x.shape))
        return linear_call(self, x)

    def moe(self, x):
        moe_calls.append(x.shape)
        return moe_call(self, x)

    monkeypatch.setattr(nn.Linear, "__call__", linear)
    monkeypatch.setattr(MoE, "__call__", moe)
    model(tokens, batch)
    assert moe_calls == [(4, 1, model.args.dim)] * model.args.n_layers
    for layer in model.layers:
        for projection in (layer.attn.wq_a, layer.attn.wq_b, layer.attn.wkv):
            shapes = [
                shape for identity, shape in linear_calls if identity == id(projection)
            ]
            assert len(shapes) == 1
            assert shapes[0][:2] == (4, 1)


@pytest.mark.parametrize("rows", [1, 2, 4, 6, 7, 8])
def test_qualified_down_qmv_keeps_route_threshold(monkeypatch, rows):
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from rapid_mlx.models.deepseek_v41_native import serving

    experts = SwitchGLU(64, 64, 8, bias=False)
    nn.quantize(experts, group_size=64, bits=2)
    experts.down_proj.scales = experts.down_proj.scales.astype(mx.bfloat16)
    experts.down_proj.biases = experts.down_proj.biases.astype(mx.bfloat16)
    assert serving.supports_target_qmv(experts)
    experts.__class__ = serving.ExactDirectDownSwitchGLU
    fast, fallback = [], []

    def qmv(projection, x, indices):
        fast.append((x.shape, indices.shape))
        return mx.zeros((x.shape[0], 64))

    def stock(self, x, indices):
        fallback.append(x.shape)
        return mx.zeros((rows, 6, 64))

    monkeypatch.setattr(serving, "affine2_route_down_qmv", qmv)
    monkeypatch.setattr(SwitchGLU, "__call__", stock)
    result = experts(
        mx.ones((rows, 64)), mx.broadcast_to(mx.arange(6)[None], (rows, 6))
    )
    assert result.shape == (rows, 6, 64)
    assert bool(fast) == (rows * 6 <= 36)
    assert bool(fallback) == (rows * 6 > 36)


def test_empty_history_row_and_explicit_capacity(model):
    singles, other_tokens = prefill(model, [7])
    singles.insert(0, model.make_cache(max_seq_len=1))
    tokens = mx.concatenate([mx.array([[3]]), other_tokens])
    batch = BatchModelCache.from_caches(singles, max_seq_len=16)
    # The first decode row has no complete ratio-2 group while the other row
    # scores several; top-k padding must stay masked, even with poisoned keys.
    for layer in batch.layers:
        if layer.index_k is not None:
            layer.index_k[0] = 1000
    singles[0] = model.make_cache(max_seq_len=16)
    for _ in range(5):
        tokens = step(model, batch, singles, tokens)
    for row, single in enumerate(singles):
        assert_cache_equal(batch.extract(row), single)


def test_batch_without_engram_and_without_compression():
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.model import Model

    args = ModelArgs(
        vocab_size=32,
        dim=64,
        n_layers=1,
        n_heads=2,
        head_dim=32,
        rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        moe_inter_dim=64,
        n_routed_experts=8,
        n_activated_experts=6,
        compress_ratios=(0,),
        hc_mult=2,
        hc_sinkhorn_iters=3,
        window_size=4,
        norm_eps=1e-6,
    )
    model = Model(args)
    singles, tokens = prefill(model, [1, 7])
    batch = BatchModelCache.from_caches(singles)
    for _ in range(4):
        tokens = step(model, batch, singles, tokens)
    assert batch.engram_ids is None
    for row, single in enumerate(singles):
        assert_cache_equal(batch.extract(row), single)


def test_join_guards_reject_incompatible_rows(model):
    from dataclasses import replace

    singles, _ = prefill(model, [2, 3])
    batch = BatchModelCache.from_caches(singles[:1])
    other = model.make_cache(max_seq_len=128)
    other.args = replace(other.args, window_size=other.args.window_size + 1)
    with pytest.raises(ValueError, match="model configurations"):
        BatchModelCache.from_caches([singles[0], other])
    histories = model.make_cache(max_seq_len=128)
    histories.engram_ids = np.zeros((2, 128), dtype=np.int64)
    with pytest.raises(ValueError, match="single-row histories"):
        BatchModelCache.from_caches([histories])
    with pytest.raises(ValueError, match="dtypes"):
        BatchModelCache.from_caches(
            [singles[0], model.make_cache(max_seq_len=128, dtype=mx.float16)]
        )
    foreign = BatchModelCache.from_caches(singles[1:])
    foreign.args = other.args
    with pytest.raises(ValueError, match="model configurations"):
        batch.extend(foreign)
    with pytest.raises(ValueError, match="does not match model"):
        model(mx.zeros((1, 1), dtype=mx.int32), foreign)
    assert batch.extend(BatchModelCache.from_caches(singles[1:]).filter([])) is batch
    assert len(batch) == 1
    with pytest.raises(IndexError, match="outside batch"):
        batch.extract(1)


def test_plain_forward_closes_a_stale_rollback_window(model):
    cache = model.make_cache(max_seq_len=64)
    model(mx.array([[1, 2, 3]]), cache, enable_rollback=True)
    assert cache.rollback_start == 0
    model(mx.array([[4]]), cache)
    assert cache.rollback_start is None
