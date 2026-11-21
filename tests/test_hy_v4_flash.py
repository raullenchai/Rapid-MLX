"""CPU reference and checkpoint-contract tests for the Hy4 text runtime."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rapid_mlx.models import hy_v4_flash_ops as ops
from rapid_mlx.models.hy_v4_flash_config import (
    ModelArgs,
    checkpoint_key,
    dequantize_fp8,
    engram_hash,
)


def tiny(**overrides):
    config = dict(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        swa_num_attention_heads=2,
        swa_qk_nope_head_dim=2,
        swa_qk_rope_head_dim=2,
        swa_v_head_dim=4,
        sliding_window=3,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        routed_expert_hidden_size=4,
        hc_mult=2,
        layer_types=["deepseek_sparse_attention", "sliding_attention"],
        compressed_indexer_compress_ratio=2,
        compressed_indexer_group_topk=2,
        compressed_indexer_head_dim=4,
        compressed_indexer_num_attention_heads=2,
        engram_layer_ids=[2],
        engram_layer_head_vocab_sizes={"2": [[5, 7], [11, 13]]},
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=4,
        engram_pad_id=15,
    )
    config.update(overrides)
    return ModelArgs.from_dict(config)


@pytest.fixture
def cpu_model(monkeypatch):
    """Execute the production graph on NumPy; does not validate MLX dispatch."""
    import rapid_mlx._mlx_compat as compat

    monkeypatch.setattr(compat, "install", lambda: None)
    rng = np.random.default_rng(17)

    class Module:
        def __init__(self):
            pass

    class Linear(Module):
        def __init__(self, input_dims, output_dims, bias=True):
            self.weight = rng.normal(0, 0.1, (output_dims, input_dims)).astype(
                np.float32
            )
            self.bias = np.zeros(output_dims, np.float32) if bias else None

        def __call__(self, x):
            y = x @ self.weight.T
            return y if self.bias is None else y + self.bias

    class Embedding(Module):
        def __init__(self, rows, width):
            self.weight = rng.normal(0, 0.1, (rows, width)).astype(np.float32)

        def __call__(self, ids):
            return self.weight[ids]

        def as_linear(self, x):
            return x @ self.weight.T

    class RMSNorm(Module):
        def __init__(self, width, eps=1e-5):
            self.weight, self.eps = np.ones(width, np.float32), eps

        def __call__(self, x):
            return ops.rms(x, self.weight, self.eps, np).astype(x.dtype)

    class SwitchGLU(Module):
        def __init__(self, hidden, intermediate, experts, activation):
            self.gate_proj = SimpleNamespace(
                weight=rng.normal(0, 0.1, (experts, intermediate, hidden)).astype(
                    np.float32
                )
            )
            self.up_proj = SimpleNamespace(
                weight=rng.normal(0, 0.1, (experts, intermediate, hidden)).astype(
                    np.float32
                )
            )
            self.down_proj = SimpleNamespace(
                weight=rng.normal(0, 0.1, (experts, hidden, intermediate)).astype(
                    np.float32
                )
            )
            self.activation = activation

        def __call__(self, x, indices):
            gate = np.einsum("...kiu,...u->...ki", self.gate_proj.weight[indices], x)
            up = np.einsum("...kiu,...u->...ki", self.up_proj.weight[indices], x)
            return np.einsum(
                "...kui,...ki->...ku",
                self.down_proj.weight[indices],
                self.activation(up, gate),
            )

    class KVCache:
        def __init__(self):
            self.offset, self.keys, self.values = 0, None, None

        def update_and_fetch(self, k, v):
            self.keys = (
                k if self.keys is None else np.concatenate((self.keys, k), axis=2)
            )
            self.values = (
                v if self.values is None else np.concatenate((self.values, v), axis=2)
            )
            self.offset += k.shape[2]
            return self.keys, self.values

        def trim(self, n):
            self.keys, self.values = self.keys[:, :, :-n], self.values[:, :, :-n]
            self.offset -= n

        @property
        def state(self):
            return self.keys, self.values

    class CacheList(list):
        def __init__(self, *caches):
            super().__init__(caches)

        def trim(self, n):
            for c in self:
                c.trim(n)

    mx = ModuleType("mlx.core")
    mx.__dict__.update(
        {name: getattr(np, name) for name in dir(np) if not name.startswith("__")}
    )
    mx.bfloat16 = np.float32
    mx.sigmoid = lambda x: ops.sigmoid(x, np)
    mx.async_eval = lambda *args: None
    nn = ModuleType("mlx.nn")
    nn.Module, nn.Linear, nn.Embedding, nn.RMSNorm = Module, Linear, Embedding, RMSNorm
    modules = {
        "mlx": ModuleType("mlx"),
        "mlx.core": mx,
        "mlx.nn": nn,
        "mlx_lm": ModuleType("mlx_lm"),
        "mlx_lm.models": ModuleType("mlx_lm.models"),
        "mlx_lm.models.cache": ModuleType("mlx_lm.models.cache"),
        "mlx_lm.models.switch_layers": ModuleType("mlx_lm.models.switch_layers"),
    }
    modules["mlx_lm.models.cache"].CacheList = CacheList
    modules["mlx_lm.models.cache"].KVCache = KVCache
    modules["mlx_lm.models.switch_layers"].SwitchGLU = SwitchGLU
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / "rapid_mlx/models/hy_v4_flash.py"
    spec = importlib.util.spec_from_file_location(
        "rapid_mlx.models._hy4_cpu_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engram_uint32_inputs_keep_signed_padding_and_history(cpu_model):
    inputs, prior = cpu_model.DecoderLayer.engram_inputs(
        np.array([[0, 3], [4, 5]], np.uint32),
        np.array([[0, 2], [1, 2]], np.uint32),
        np.array([-1, 2], np.int32),
    )
    assert inputs.dtype == prior.dtype == np.int32
    np.testing.assert_array_equal(inputs, [[-1, 3], [4, 5]])
    np.testing.assert_array_equal(prior, [[-1, -1], [1, 2]])


def test_auxiliary_payloads_only_store_keys_that_are_read(cpu_model, monkeypatch):
    model = cpu_model.Model(tiny())
    cache = model.make_cache()
    writes = []
    update = cpu_model.KVCache.update_and_fetch

    def record(self, keys, values):
        writes.append((self, keys, values))
        return update(self, keys, values)

    monkeypatch.setattr(cpu_model.KVCache, "update_and_fetch", record)
    for tokens in ([[1, 2, 3], [0, 3, 4]], [[5], [6]], [[7, 8], [9, 10]]):
        model(np.array(tokens, np.int32), cache)
    for layer, slots in zip(model.layers, cache):
        for slot in (1, 2):
            expected_width = (
                (0 if layer.self_attn.is_window_layer else 4)
                if slot == 1
                else (0 if layer.engram_layer is None else 1)
            )
            selected = [(k, v) for c, k, v in writes if c is slots[slot]]
            assert len(selected) == 3
            for keys, values in selected:
                assert keys.shape[-1] == expected_width
                assert values.shape == (*keys.shape[:-1], 0)
                assert values.nbytes == 0
                if expected_width == 0:
                    assert keys.nbytes == 0
        assert all(c.offset == 6 for c in slots)
        assert slots[0].values.shape[-1] == 2  # Rotary keys are still required.
    for slots in cache:
        slots.trim(2)
    model(np.array([[7, 8], [9, 10]], np.int32), cache)
    assert all(c.offset == 6 for slots in cache for c in slots)


@pytest.mark.parametrize("tied", [False, True])
def test_each_call_schedules_logits_and_all_auxiliary_roots(
    cpu_model, monkeypatch, tied
):
    class BatchCache(cpu_model.KVCache):
        @property
        def state(self):
            # BatchKVCache includes counters as well as key/value storage.
            return self.keys, self.values, self.offset, self.left_padding

    model = cpu_model.Model(tiny(tie_word_embeddings=tied))
    cache = [
        cpu_model.CacheList(*(BatchCache() for _ in range(3))) for _ in model.layers
    ]
    for slots in cache:
        for c in slots:
            c.offset = np.array([0, -1])
            c.left_padding = np.array([0, 1])
    scheduled = []
    monkeypatch.setattr(
        cpu_model.mx, "async_eval", lambda *args: scheduled.append(args)
    )
    for tokens in ([[1, 2], [0, 3]], [[4], [5]], [[6], [7]]):
        logits = model(np.array(tokens, np.int32), cache)
        assert len(scheduled) == 1
        output, roots = scheduled.pop()
        assert output is logits
        expected = [c[i].state for c in cache for i in (1, 2)]
        assert len(roots) == len(expected)
        for actual, state in zip(roots, expected):
            assert all(a is b for a, b in zip(actual, state, strict=True))


def test_greedy_logits_match_legacy_auxiliary_writes(cpu_model, monkeypatch):
    """Replay the removed writes on the same weights, including padded batches."""
    model = cpu_model.Model(tiny())

    def generate():
        cache = model.make_cache()
        for slots in cache:
            for c in slots:
                c.offset = np.array([0, -2])
        tokens = np.array([[1, 2, 3, 4], [0, 0, 3, 4]], np.int32)
        outputs = []
        for _ in range(32):
            logits = model(tokens, cache)
            tokens = logits[:, -1].argmax(-1)[:, None].astype(np.int32)
            outputs.append((logits.copy(), tokens.copy()))
        return outputs

    actual = generate()
    legacy_keys = {}
    attention = cpu_model.Attention.__call__
    decoder = cpu_model.DecoderLayer.__call__

    def legacy_attention(self, x, cache):
        if self.is_window_layer:
            legacy_keys[id(cache[1])] = x[:, None, :, :1]
        return attention(self, x, cache)

    def legacy_decoder(self, hidden, inputs, cache):
        legacy_keys[id(cache[2])] = inputs[:, None, :, None]
        return decoder(self, hidden, inputs, cache)

    def legacy_update(cache, keys):
        keys = legacy_keys.get(id(cache), keys)
        return cache.update_and_fetch(keys, keys)[0]

    monkeypatch.setattr(cpu_model.Attention, "__call__", legacy_attention)
    monkeypatch.setattr(cpu_model.DecoderLayer, "__call__", legacy_decoder)
    monkeypatch.setattr(cpu_model, "_update_aux_cache", legacy_update)
    for (logits, tokens), (old_logits, old_tokens) in zip(
        actual, generate(), strict=True
    ):
        np.testing.assert_array_equal(logits, old_logits)
        np.testing.assert_array_equal(tokens, old_tokens)


def test_uint32_logits_match_int32_prefill_and_decode(cpu_model):
    model = cpu_model.Model(tiny())
    signed_cache, unsigned_cache = model.make_cache(), model.make_cache()
    for tokens in ([[1, 2, 3]], [[4]], [[5, 6]]):
        signed = np.array(tokens, np.int32)
        unsigned = signed.astype(np.uint32)
        np.testing.assert_array_equal(model(unsigned), model(signed))
        np.testing.assert_array_equal(
            model(unsigned, unsigned_cache), model(signed, signed_cache)
        )


def test_forward_dispatch_and_every_prefill_boundary(cpu_model):
    args = tiny()
    model = cpu_model.Model(args)
    assert not model.layers[0].self_attn.is_window_layer
    assert model.layers[1].self_attn.is_window_layer
    assert isinstance(model.layers[0].mlp, cpu_model.MLP)
    assert isinstance(model.layers[1].mlp, cpu_model.MoE)
    tokens = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
    full = model(tokens)
    assert full.shape == (1, 6, 16)
    assert np.isfinite(full).all()
    for boundary in range(1, tokens.shape[1]):
        cache = model.make_cache()
        first = model(tokens[:, :boundary], cache)
        second = model(tokens[:, boundary:], cache)
        np.testing.assert_allclose(
            np.concatenate((first, second), axis=1), full, atol=2e-5
        )
    cache = model.make_cache()
    pieces = [model(tokens[:, i : i + 1], cache) for i in range(tokens.shape[1])]
    np.testing.assert_allclose(np.concatenate(pieces, axis=1), full, atol=2e-5)
    for layer in cache:
        layer.trim(2)
    np.testing.assert_allclose(model(tokens[:, -2:], cache), full[:, -2:], atol=2e-5)
    with pytest.raises(ValueError, match="one cache"):
        model(tokens, [])
    assert model.supports_prefix_cache is False


def test_engram_matches_scalar_reference():
    a = tiny()
    ids = np.array([[1, 2, 3, 15, 7, -1, 4, 40]], np.int64)
    lookup = np.arange(a.vocab_size, dtype=np.int64) // 2
    primes = a.engram_layer_head_vocab_sizes["2"]
    result = engram_hash(ids, ids[:, :0], lookup, primes, 2, a)
    bound = (np.iinfo(np.int64).max // (int(max(lookup)) + 1)) // 2
    multipliers = (
        np.random.default_rng(20014).integers(0, bound, size=3, dtype=np.int64) * 2 + 1
    )
    for pos, token in enumerate(ids[0]):
        if token in (15, -1, 40):
            assert (result[0, pos] == -1).all()
            continue
        rolling, blocked, offset, expected = 0, False, 0, []
        for shift in range(3):
            src = int(ids[0, pos - shift]) if pos >= shift else -1
            blocked |= not 0 <= src < a.vocab_size
            mapped = int(lookup[15 if blocked else src])
            product = (mapped * int(multipliers[shift])) & ((1 << 64) - 1)
            rolling ^= product
            signed = rolling if rolling < 1 << 63 else rolling - (1 << 64)
            if shift:
                for p in primes[shift - 1]:
                    expected.append(signed % p + offset)
                    offset += p
        np.testing.assert_array_equal(result[0, pos], expected)
    for split in range(1, ids.shape[1]):
        np.testing.assert_array_equal(
            engram_hash(ids[:, split:], ids[:, :split], lookup, primes, 2, a),
            result[:, split:],
        )
    for lookup in ([], [[1]], [-1], [2**32], np.arange(17), [1.5]):
        with pytest.raises(ValueError, match="token map"):
            engram_hash(ids, ids[:, :0], lookup, primes, 2, a)


def test_fp8_blocks_rows_experts_and_malformed():
    raw = np.array([[0, 1, 7, 8, 56, 126, 128, 184]], np.uint8)
    expected = np.array([[0, 2**-9, 7 * 2**-9, 2**-6, 1, 448, -0.0, -1]], np.float32)
    np.testing.assert_array_equal(dequantize_fp8(raw, np.array([2.0])), expected * 2)
    raw = np.tile(raw, (3, 1))
    scales = np.array([[1, 2, 3], [4, 5, 6]])
    result = dequantize_fp8(raw, scales, (2, 3))
    np.testing.assert_array_equal(
        result,
        np.tile(expected, (3, 1)) * np.repeat(np.repeat(scales, 2, 0), 3, 1)[:3, :8],
    )
    np.testing.assert_array_equal(
        dequantize_fp8(raw[None], scales[None], (2, 3)), result[None]
    )
    with pytest.raises(ValueError, match="scale shape"):
        dequantize_fp8(raw, np.ones((4, 4)))
    for byte in (127, 255):
        with pytest.raises(ValueError, match="Non-finite"):
            dequantize_fp8(np.array([[byte]], np.uint8), np.ones(1))


def test_sink_window_rope_routing_and_hyper_connections():
    q = np.zeros((1, 1, 5, 4), np.float32)
    v = np.arange(20, dtype=np.float32).reshape(1, 1, 5, 4)
    pos = np.arange(5)[None]
    allowed = (pos[:, None, :] <= pos[..., None]) & (
        pos[:, None, :] > pos[..., None] - 3
    )
    out = ops.attention(q, q, v, allowed, np.zeros(1), np)
    for i in range(5):
        np.testing.assert_allclose(
            out[0, 0, i], v[0, 0, max(0, i - 2) : i + 1].sum(0) / (min(i + 1, 3) + 1)
        )
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    np.testing.assert_array_equal(ops.rope(x, np.zeros((1, 1)), 4, 10000, np), x)
    np.testing.assert_array_equal(
        ops.rope(x, np.zeros((1, 1)), 4, 10000, np, indexer=True), x[..., [0, 2, 1, 3]]
    )
    indices, scores = ops.route(
        np.array([[10.0, 0.0, -10.0]]),
        np.array([-100.0, 0.0, 100.0]),
        2,
        True,
        3.998,
        np,
    )
    assert set(indices[0]) == {1, 2}
    assert scores[0, np.argmax(indices[0])] < 0.001
    assert scores.sum() == pytest.approx(3.998)
    _, scores = ops.route(np.zeros((1, 3)), np.zeros(3), 1, False, 1.0, np)
    assert scores.item() == 0.5
    residual = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
    y, post = ops.hc_pre(
        residual, np.zeros((4, 8)), np.ones(2), np.zeros(4), 1e-6, 1e-5, 2.0, np
    )
    np.testing.assert_allclose(y, residual.sum(-2) * (0.5 + 1e-6))
    np.testing.assert_allclose(post, np.full((1, 1, 2), 1 + 1e-6))
    np.testing.assert_allclose(
        ops.hc_post(y, residual, post, np), residual + post[..., None] * y[..., None, :]
    )
    np.testing.assert_allclose(
        ops.hc_head(
            residual, np.zeros((2, 8)), np.ones(1), np.zeros(2), 1e-6, 1e-5, np
        ),
        y,
    )


def test_indexpool_tail_and_budget():
    q = np.arange(10)[None]
    k = np.arange(10)[None]
    scores = np.broadcast_to(np.array([4.0, 3.0, 2.0, 1.0, 0.0]), (1, 10, 5))
    mask = ops.sparse_mask(q, k, scores, 2, 2, np)
    assert np.flatnonzero(mask[0, 9]).tolist() == [0, 1, 8, 9]
    tail = ops.sparse_mask(q, k, scores, 2, 1, np)
    assert np.flatnonzero(tail[0, 9]).tolist() == [8, 9]
    value = np.ones((1, 4), np.float32)
    np.testing.assert_array_equal(ops.fwht_fp8(value, np), [[2.0, 0, 0, 0]])
    np.testing.assert_array_equal(ops.fwht_fp8(np.zeros((1, 4)), np), np.zeros((1, 4)))


@pytest.mark.parametrize(
    "batch,query_len,history_len,ratio,group_count",
    [(1, 1, 9, 2, 4), (2, 7, 17, 4, 5), (3, 4, 8, 1, 8), (1, 3, 3, 4, 0)],
)
@pytest.mark.parametrize("budget", [1, 2, 5, 20])
def test_sparse_mask_matches_broadcast_reference(
    batch, query_len, history_len, ratio, group_count, budget
):
    rng = np.random.default_rng(42)
    query = rng.integers(-1, history_len + ratio, (batch, query_len))
    if query_len > 1:
        query[:, 0] = -1
    query[:, -1] = history_len - 1
    key = np.broadcast_to(np.arange(history_len), (batch, history_len)).copy()
    key[:, 0] = -1
    scores = rng.normal(size=(batch, query_len, group_count)).astype(np.float32)
    upper = query // ratio
    live = np.arange(group_count)[None, None, :] < upper[..., None]
    indices = (
        np.argsort(np.where(live, scores, -np.inf), axis=-1)[
            ..., -min(budget - 1, group_count) :
        ]
        if budget > 1
        else np.zeros((*query.shape, 0), dtype=np.int32)
    )
    selected = np.any(key[:, None, :, None] // ratio == indices[:, :, None, :], axis=-1)
    expected = (
        (
            (selected & (key[:, None, :] // ratio < upper[..., None]))
            | (key[:, None, :] // ratio == upper[..., None])
        )
        & (key[:, None, :] >= 0)
        & (key[:, None, :] <= query[..., None])
    )
    actual = ops.sparse_mask(query, key, scores, ratio, budget, np)
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, expected)


def test_sparse_mask_never_builds_four_dimensional_intermediate():
    class AtMostThreeDimensions(np.ndarray):
        def __array_finalize__(self, obj):
            assert self.ndim <= 3, self.shape

    query = np.arange(5)[None].view(AtMostThreeDimensions)
    key = np.arange(13)[None].view(AtMostThreeDimensions)
    scores = np.ones((1, 5, 7)).view(AtMostThreeDimensions)
    assert ops.sparse_mask(query, key, scores, 2, 5, np).shape == (1, 5, 13)


def test_config_defaults_nested_and_rejection():
    assert tiny().mlp_layer_types == ["dense", "sparse"]
    assert (
        ModelArgs.from_dict({"text_config": {"num_hidden_layers": 2}}).layer_types
        == ["deepseek_sparse_attention"] * 2
    )
    cases = [
        dict(enable_ihc=False),
        dict(gate_lora_rank=4),
        dict(rope_parameters={"sliding_attention": {"rope_type": "unsupported"}}),
        dict(layer_types=["unknown"]),
        dict(mlp_layer_types=["unknown"]),
        dict(num_experts_per_tok=0),
        dict(sliding_window=0),
        dict(hc_mult=1),
        dict(engram_layer_ids=[0]),
        dict(engram_layer_ids=[1], engram_pad_id=-1),
        dict(engram_layer_ids=[1], vocab_size=16, engram_pad_id=16),
    ]
    for config in cases:
        with pytest.raises(ValueError):
            ModelArgs.from_dict(config)


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj"])
@pytest.mark.parametrize("leaf", ["weight", "scales", "biases"])
def test_sanitize_rejects_generated_split_collisions(cpu_model, projection, leaf):
    prefix = "model.layers.1.mlp.switch_mlp"
    target = f"{prefix}.{projection}.{leaf}"
    weights = {
        f"{prefix}.gate_up_proj.{leaf}": np.ones((3, 8, 4)),
        target: np.ones((3, 4, 4)),
    }
    with pytest.raises(ValueError, match=f"Duplicate Hy4 Flash weight: {target}"):
        cpu_model.Model(tiny()).sanitize(weights)


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize("leaf", ["weight", "scales", "biases"])
def test_sanitize_rejects_stacked_bank_collisions(cpu_model, projection, leaf):
    prefix = "model.layers.1.mlp.switch_mlp"
    target = f"{prefix}.{projection}.{leaf}"
    weights = {target: np.ones((3, 4, 4))}
    weights.update(
        {f"{prefix}.{e}.{projection}.{leaf}": np.ones((4, 4)) for e in range(3)}
    )
    with pytest.raises(ValueError, match=f"Duplicate Hy4 Flash weight: {target}"):
        cpu_model.Model(tiny()).sanitize(weights)


def test_sanitize_experts_quantization_and_optional_paths(cpu_model, caplog):
    a = tiny(
        tie_word_embeddings=True,
        sandwich_norm=False,
        n_shared_experts=0,
        swiglu_limit=0,
        engram_layer_ids=[],
        enable_lm_head_fp32=False,
    )
    model = cpu_model.Model(a)
    assert np.isfinite(model(np.array([[1, 2, 3]]))).all()
    weights = {
        "vit.weight": np.ones(1),
        "model.dfly_layers.0.weight": np.ones(1),
        "model.layers.1.mlp.experts.gate_up_proj": np.arange(96).reshape(3, 8, 4),
        "lm_head.weight": np.ones((16, 8)),
    }
    with caplog.at_level("INFO"):
        mapped = model.sanitize(weights)
    assert "text-only" in caplog.text
    assert set(mapped) == {
        "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        "model.layers.1.mlp.switch_mlp.up_proj.weight",
    }
    for e in range(3):
        weights[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = (
            np.ones((4, 4)) * e
        )
    mapped = model.sanitize(weights)
    assert mapped["model.layers.1.mlp.switch_mlp.down_proj.weight"].shape == (3, 4, 4)
    with pytest.raises(ValueError, match="Duplicate"):
        model.sanitize(
            {
                "model.norm.weight": np.ones(8),
                "language_model.model.norm.weight": np.ones(8),
            }
        )
    pred = model.quant_predicate
    assert pred("lm_head", None) is False
    assert pred("model.layers.1.mlp.gate", None) is False
    assert pred("model.layers.1.engram_layer.multi_head_embedding.embedding", None) == {
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
    }
    assert pred("model.layers.1.engram_layer.key_projs.0", None) is False
    assert pred("model.layers.1.self_attn.q_b_proj", None) == {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    }
    assert not model.cast_predicate("lm_head.weight")
    assert model.cast_predicate("model.layers.0.self_attn.q_b_proj.weight")
    assert checkpoint_key("model.model.norm.weight") == "model.norm.weight"
    assert (
        checkpoint_key("model.layers.2.engram.lookup_table")
        == "model.layers.1.engram_layer.lookup_table"
    )


def test_every_published_text_tensor_is_mapped(cpu_model):
    fixture = Path(__file__).parent / "fixtures/hy_v4_flash"
    config = json.loads((fixture / "config.json").read_text())
    real_args = ModelArgs.from_dict(config)
    assert real_args.num_hidden_layers == 42
    a = tiny(
        hc_mult=4,
        engram_n_head_per_ngram=8,
        engram_n_embed_per_ngram=16,
        num_hidden_layers=42,
        layer_types=real_args.layer_types,
        engram_layer_ids=real_args.engram_layer_ids,
        engram_layer_head_vocab_sizes={
            str(i): [list(range(5, 13)), list(range(13, 21))]
            for i in real_args.engram_layer_ids
        },
    )
    model = cpu_model.Model(a)

    def leaves(value, prefix=""):
        if isinstance(value, np.ndarray):
            return {prefix}
        if isinstance(value, (list, tuple)):
            return set().union(
                *(leaves(v, f"{prefix}.{i}") for i, v in enumerate(value))
            )
        if hasattr(value, "__dict__"):
            return set().union(
                *(
                    leaves(v, f"{prefix}.{k}" if prefix else k)
                    for k, v in vars(value).items()
                    if not k.startswith("_") and k not in ("args",)
                )
            )
        return set()

    parameters = leaves(model)
    names = json.loads((fixture / "text_tensor_names.json").read_text())
    for name in names:
        mapped = checkpoint_key(name)
        if mapped is None:
            assert name.startswith(("vit.", "model.dfly_layers."))
            continue
        mapped = mapped.removesuffix("_scale_inv")
        if ".gate_up_proj." in mapped:
            for projection in ("gate_proj", "up_proj"):
                assert (
                    mapped.replace(".gate_up_proj.", f".{projection}.") in parameters
                ), name
        else:
            assert mapped in parameters, name


def cpu_parameters(value, prefix=""):
    """Walk the independent CPU model graph, including list parameters."""
    if isinstance(value, np.ndarray):
        return {prefix: value}
    if isinstance(value, (list, tuple)):
        children = enumerate(value)
    elif hasattr(value, "__dict__"):
        children = (
            (k, v)
            for k, v in vars(value).items()
            if not k.startswith("_") and k != "args"
        )
    else:
        return {}
    result = {}
    for key, child in children:
        result.update(cpu_parameters(child, f"{prefix}.{key}" if prefix else str(key)))
    return result


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(sandwich_norm=False, n_shared_experts=0),
        dict(
            tie_word_embeddings=True, dsa_skip_attn_post_norm=True, n_shared_experts=2
        ),
    ],
)
def test_conversion_shapes_match_model_without_allocating_weights(cpu_model, overrides):
    from rapid_mlx.models.hy_v4_flash_config import parameter_shapes

    args = tiny(**overrides)
    actual = cpu_parameters(cpu_model.Model(args))
    assert parameter_shapes(args) == {k: v.shape for k, v in actual.items()}


@pytest.mark.parametrize("tied", [False, True])
def test_sanitized_checkpoint_matches_strict_model_parameters(cpu_model, tied):
    from rapid_mlx.models.hy_v4_flash_config import parameter_shapes

    args = tiny(tie_word_embeddings=tied)
    model = cpu_model.Model(args)
    weights = {
        k: np.zeros(shape, np.float32) for k, shape in parameter_shapes(args).items()
    }
    for key in weights:
        if key.endswith(".lookup_table"):
            weights[key] = np.arange(args.vocab_size, dtype=np.int64)
    sanitized = model.sanitize(weights)
    actual = cpu_parameters(model)
    assert set(sanitized) == set(actual)
    assert {k: v.shape for k, v in sanitized.items()} == {
        k: v.shape for k, v in actual.items()
    }
    assert hasattr(model, "lm_head") is not tied


@pytest.fixture
def fp8_conversion(cpu_model, monkeypatch, tmp_path):
    from dataclasses import asdict

    from safetensors.numpy import save_file

    from rapid_mlx.models import hy_v4_flash_convert as converter

    args = tiny(
        hidden_size=64,
        intermediate_size=64,
        routed_expert_hidden_size=64,
        moe_intermediate_size=64,
        engram_n_embed_per_ngram=64,
    )
    config = asdict(args)
    (tmp_path / "config.json").write_text(json.dumps(config))
    values = cpu_parameters(cpu_model.Model(args))
    arrays = {k: (v.astype(np.float32), "F32") for k, v in values.items()}
    arrays["model.layers.1.engram_layer.lookup_table"] = (
        np.arange(16, dtype=np.int64),
        "I64",
    )
    for projection in ("gate_proj", "up_proj"):
        arrays.pop(f"model.layers.1.mlp.switch_mlp.{projection}.weight")
    arrays.update(
        {
            "model.layers.1.mlp.experts.gate_up_proj": (
                np.full((3, 128, 64), 56, np.uint8),
                "F8_E4M3",
            ),
            "model.layers.1.mlp.experts.gate_up_proj_scale_inv": (
                np.full((3, 1, 1), 2.0, np.float32),
                "F32",
            ),
            "vit.weight": (np.ones(2, np.float32), "F32"),
        }
    )

    class Shard:
        def __init__(self, path):
            assert path == (tmp_path / "model.safetensors").resolve()

        @property
        def header(self):
            return {k: {"dtype": t} for k, (v, t) in arrays.items()}

        def read(self, name):
            v, t = arrays[name]
            return np.frombuffer(v.tobytes(), dtype=np.uint8), t, list(v.shape)

    fp8 = ModuleType("rapid_mlx.fp8_repack")
    (tmp_path / "model.safetensors").touch()
    fp8._ShardReader = Shard
    fp8._raw_to_mx = lambda raw, dtype, shape: raw.view(np.float32).reshape(shape)
    monkeypatch.setitem(sys.modules, "rapid_mlx.fp8_repack", fp8)
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.hy_v4_flash", cpu_model)
    lm_utils = ModuleType("mlx_lm.utils")
    saved = []
    lm_utils.load_tokenizer = lambda _: SimpleNamespace(
        save_pretrained=lambda path: saved.append(("tokenizer", path))
    )
    lm_utils.save_config = lambda config, config_path: Path(config_path).write_text(
        json.dumps(config)
    )
    lm_utils.create_model_card = lambda path, repo: saved.append(("card", path))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", lm_utils)

    # Construction and whole-model loading/saving must never be used.
    def forbidden(*args, **kwargs):
        pytest.fail("Conversion must not allocate or load a model")

    monkeypatch.setattr(cpu_model.Model, "__init__", forbidden)
    mx = sys.modules["mlx.core"]
    quantized = []

    def quantize(value, **params):
        quantized.append((value.copy(), params))
        return (
            np.zeros(
                (*value.shape[:-1], value.shape[-1] * params["bits"] // 32), np.uint32
            ),
            np.ones(
                (*value.shape[:-1], value.shape[-1] // params["group_size"]), np.float32
            ),
            np.zeros(
                (*value.shape[:-1], value.shape[-1] // params["group_size"]), np.float32
            ),
        )

    mx.quantize = quantize
    mx.eval = lambda *args: None
    mx.save_safetensors = lambda path, tensors, metadata: save_file(
        tensors, path, metadata=metadata
    )
    return converter, arrays, quantized, saved


def test_fp8_conversion_uses_standard_quantization_and_saver(
    fp8_conversion, tmp_path, caplog
):
    from safetensors.numpy import load_file

    converter, arrays, quantized, saved = fp8_conversion
    (tmp_path / "generation_config.json").write_text('{"test": true}')
    (tmp_path / "custom.py").write_text("# tokenizer code")
    with caplog.at_level("INFO"):
        assert converter.convert_fp8(tmp_path, tmp_path / "out", 4, 64)
    assert "text-only" in caplog.text
    assert sum(v.shape == (3, 64, 64) and np.all(v == 2.0) for v, _ in quantized) == 2
    for value, params in quantized:
        assert params == {"group_size": 32 if value.shape[-1] == 32 else 64, "bits": 4}
    output = tmp_path / "out"
    assert [kind for kind, _ in saved] == ["card"]
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert not any("vit" in key or "scale_inv" in key for key in index["weight_map"])
    for key, filename in index["weight_map"].items():
        assert key in load_file(output / filename)
    config = json.loads((output / "config.json").read_text())
    assert config["quantization"]["model.layers.1.mlp.switch_mlp.gate_proj"] == dict(
        group_size=64, bits=4, mode="affine"
    )
    assert (output / "generation_config.json").read_bytes() == (
        tmp_path / "generation_config.json"
    ).read_bytes()
    assert not (output / "custom.py").exists()
    arrays["model.layers.1.engram_layer.lookup_table"] = (
        np.arange(16, dtype=np.int32),
        "I32",
    )
    assert converter.convert_fp8(tmp_path, tmp_path / "out2", 4, 32)
    with pytest.raises(ValueError, match="already exists"):
        converter.convert_fp8(tmp_path, tmp_path, 4, 32)
    arrays.pop("model.layers.1.mlp.switch_mlp.down_proj.weight")
    for e in range(3):
        arrays[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = (
            np.full((64, 64), e, np.float32),
            "F32",
        )
    quantized.clear()
    assert converter.convert_fp8(tmp_path, tmp_path / "out3", 4, 32)
    stacked = [value for value, _ in quantized if value.shape == (3, 64, 64)]
    assert any(np.array_equal(v[:, 0, 0], np.arange(3)) for v in stacked)
    arrays.pop("model.layers.1.mlp.experts.2.down_proj.weight")
    with pytest.raises(ValueError, match="incomplete routed-expert banks"):
        converter.convert_fp8(tmp_path, tmp_path / "out4", 4, 32)
    arrays.pop("model.layers.1.mlp.experts.gate_up_proj")
    assert converter.convert_fp8(tmp_path, tmp_path / "out5", 4, 32) is False
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "other"}))
    assert converter.convert_fp8(tmp_path, tmp_path / "out6", 4, 32) is False


@pytest.mark.parametrize("nested", [False, True])
def test_fp8_conversion_copies_only_safe_artifacts(fp8_conversion, tmp_path, nested):
    converter, _, _, _ = fp8_conversion
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    remote = {
        "auto_map": {"AutoModel": "custom.Model"},
        "trust_remote_code": True,
        "_trust_remote_code": True,
        "code_revision": "custom-code",
    }
    if nested:
        config = {"model_type": "hy_v4_flash", "text_config": config}
        config["text_config"].update(remote)
    config.update(remote)
    config_path.write_text(json.dumps(config))
    artifacts = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "generation_config.json",
        "LICENSE",
        "NOTICE",
    )
    for name in (*artifacts, "custom.py", "tokenizer.py", "unrelated.json"):
        (tmp_path / name).write_text(name)
    output = tmp_path / "out"
    assert converter.convert_fp8(tmp_path, output, 4, 32)
    for name in artifacts:
        assert (output / name).read_bytes() == (tmp_path / name).read_bytes()
    assert not list(output.glob("*.py"))
    assert not (output / "unrelated.json").exists()
    emitted = json.loads((output / "config.json").read_text())
    for section in (emitted, emitted.get("text_config", {})):
        assert not remote.keys() & section.keys()
    assert "auto_map" in json.loads(config_path.read_text())


@pytest.mark.parametrize("failure", ["weights", "config", "card"])
def test_fp8_conversion_cleans_up_and_can_retry(
    fp8_conversion, monkeypatch, tmp_path, failure
):
    converter, _, _, _ = fp8_conversion
    output = tmp_path / "out"
    before = set(tmp_path.iterdir())
    module, attribute = {
        "weights": (sys.modules["mlx.core"], "save_safetensors"),
        "config": (sys.modules["mlx_lm.utils"], "save_config"),
        "card": (sys.modules["mlx_lm.utils"], "create_model_card"),
    }[failure]
    original = getattr(module, attribute)

    def fail(*args, **kwargs):
        assert not output.exists()
        staging = set(tmp_path.iterdir()) - before
        assert len(staging) == 1
        assert staging.pop().is_dir()
        original(*args, **kwargs)
        raise RuntimeError("injected conversion failure")

    with monkeypatch.context() as patch:
        patch.setattr(module, attribute, fail)
        with pytest.raises(RuntimeError, match="injected conversion failure"):
            converter.convert_fp8(tmp_path, output, 4, 32)
    assert not output.exists()
    assert set(tmp_path.iterdir()) == before
    assert converter.convert_fp8(tmp_path, output, 4, 32)
    assert (output / "config.json").is_file()
    assert (output / "model.safetensors.index.json").is_file()
    assert set(tmp_path.iterdir()) == before | {output}


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj"])
@pytest.mark.parametrize("packed_first", [True, False])
def test_fp8_conversion_rejects_split_collisions(
    fp8_conversion, tmp_path, projection, packed_first
):
    converter, arrays, _, _ = fp8_conversion
    target = f"model.layers.1.mlp.switch_mlp.{projection}.weight"
    arrays[target] = np.ones((3, 64, 64), np.float32), "F32"
    if not packed_first:
        packed = "model.layers.1.mlp.experts.gate_up_proj"
        arrays[packed] = arrays.pop(packed)
    with pytest.raises(ValueError, match=f"Duplicate Hy4 Flash weight: {target}"):
        converter.convert_fp8(tmp_path, tmp_path / "out", 4, 32)


@pytest.mark.parametrize("mutation", ["missing", "shape", "unexpected", "duplicate"])
def test_fp8_conversion_validates_parameters(fp8_conversion, tmp_path, mutation):
    converter, arrays, _, _ = fp8_conversion
    if mutation == "missing":
        arrays.pop("model.norm.weight")
    elif mutation == "shape":
        arrays["model.norm.weight"] = np.ones(1, np.float32), "F32"
    elif mutation == "unexpected":
        arrays["unexpected.weight"] = np.ones(1, np.float32), "F32"
    else:
        arrays["language_model.model.norm.weight"] = arrays["model.norm.weight"]
    with pytest.raises(ValueError, match="Missing|Unexpected|Duplicate"):
        converter.convert_fp8(tmp_path, tmp_path / "out", 4, 32)
    assert not (tmp_path / "out/model.safetensors.index.json").exists()


def test_fp8_output_shards_are_bounded_and_release_tensors(
    fp8_conversion, monkeypatch, tmp_path
):
    import weakref

    from safetensors import safe_open

    converter, _, _, _ = fp8_conversion
    writer_class = converter._ShardWriter
    limit = 40_000
    references = []
    writers = []
    release_checks = []

    class Writer(writer_class):
        def __init__(self, destination):
            super().__init__(destination, max_bytes=limit)
            writers.append(self)

        def flush(self):
            references.extend(weakref.ref(v) for v in self.pending.values())
            super().flush()

    monkeypatch.setattr(converter, "_ShardWriter", Writer)
    # This recorder deliberately retains no converted arrays.
    mx = sys.modules["mlx.core"]
    original_quantize = mx.quantize

    def quantize(*args, **kwargs):
        result = original_quantize(*args, **kwargs)
        fp8_conversion[2].clear()
        # Previously flushed output must die during conversion, not just at exit.
        if len(references) > 16:
            assert all(ref() is None for ref in references[:-16])
            release_checks.append(True)
        return result

    mx.quantize = quantize
    assert converter.convert_fp8(tmp_path, tmp_path / "out", 4, 32)
    writer = writers[0]
    assert not writer.pending
    assert release_checks
    assert all(ref() is None for ref in references)
    assert writer.shards > 1
    for shard in (tmp_path / "out").glob("*.safetensors"):
        with safe_open(shard, framework="np") as f:
            assert sum(f.get_tensor(k).nbytes for k in list(f.keys())) <= limit
    assert writer.total > limit


def test_fp8_oversized_tensor_gets_dedicated_shard(fp8_conversion, tmp_path):
    from safetensors.numpy import load_file

    converter, _, _, _ = fp8_conversion
    writer = converter._ShardWriter(tmp_path / "out", max_bytes=16)
    values = {
        "before": np.arange(2, dtype=np.float32),
        "oversized": np.arange(5, dtype=np.float32),
        "after": np.arange(4, dtype=np.float32),
    }
    for name, value in values.items():
        writer.put(name, value)
        if name == "oversized":
            assert not writer.pending
            assert writer.size == 0
        with pytest.raises(ValueError, match="Duplicate"):
            writer.put(name, value)
    writer.finish(total_parameters=11)
    index = json.loads(
        (writer.destination / "model.safetensors.index.json").read_text()
    )
    assert index["metadata"] == {"total_size": 44, "total_parameters": 11}
    assert index["weight_map"] == {
        name: f"model-{i:05d}.safetensors" for i, name in enumerate(values, 1)
    }
    assert len(list(writer.destination.glob("*.safetensors"))) == 3
    for name, filename in index["weight_map"].items():
        tensors = load_file(writer.destination / filename)
        assert list(tensors) == [name]
        np.testing.assert_array_equal(tensors[name], values[name])


def test_moe_latent_clamp_bias_and_shared_reference(cpu_model):
    a = tiny(swiglu_limit=0.1)
    moe = cpu_model.MoE(a)
    x = np.arange(16, dtype=np.float32).reshape(1, 2, 8) / 4
    moe.expert_bias = np.array([50.0, -50.0, 0.0])
    actual = moe(x)
    logits = x @ moe.gate.weight.T
    prob = 1 / (1 + np.exp(-logits))
    indices = np.argsort(prob + moe.expert_bias, axis=-1)[..., -2:]
    expected = np.empty_like(x)
    for i in range(2):
        weights = prob[0, i, indices[0, i]]
        weights = weights / weights.sum() * a.routed_scaling_factor
        latent = moe.routed_expert_down_proj.weight @ x[0, i]
        combined = np.zeros(a.routed_expert_hidden_size)
        for e, weight in zip(indices[0, i], weights):
            gate = np.minimum(moe.switch_mlp.gate_proj.weight[e] @ latent, 0.1)
            up = np.clip(moe.switch_mlp.up_proj.weight[e] @ latent, -0.1, 0.1)
            combined += weight * (
                moe.switch_mlp.down_proj.weight[e] @ (gate / (1 + np.exp(-gate)) * up)
            )
        expected[0, i] = (
            moe.routed_expert_up_proj.weight @ combined + moe.shared_experts(x[0, i])
        )
    expected = expected / np.sqrt(
        np.mean(expected**2, axis=-1, keepdims=True) + a.rms_norm_eps
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_owned_registration_is_on_production_loader_path(cpu_model, monkeypatch):
    import rapid_mlx.models as models
    from rapid_mlx.utils import tokenizer

    monkeypatch.setattr(models, "hy_v4_flash", cpu_model, raising=False)
    monkeypatch.delitem(sys.modules, "mlx_lm.models.hy_v4_flash", raising=False)
    monkeypatch.setattr(
        tokenizer, "_VENDORED_MODEL_TYPES", set(tokenizer._VENDORED_MODEL_TYPES)
    )
    # Other architectures are irrelevant here; seed their existing registrations.
    import inspect

    source = inspect.getsource(tokenizer._register_vendored_archs)
    for name in __import__("re").findall(r'"(mlx_lm.models.[a-z0-9_]+)"', source):
        if name != "mlx_lm.models.hy_v4_flash":
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
    tokenizer._register_vendored_archs()
    assert sys.modules["mlx_lm.models.hy_v4_flash"] is cpu_model
    assert "hy_v4_flash" in tokenizer._VENDORED_MODEL_TYPES
    assert cpu_model.ModelArgs is ModelArgs
    # Registration must also repair membership when the module is already
    # imported (e.g. after test isolation resets the runtime registry).
    tokenizer._VENDORED_MODEL_TYPES.discard("hy_v4_flash")
    tokenizer._register_vendored_archs()
    assert sys.modules["mlx_lm.models.hy_v4_flash"] is cpu_model
    assert "hy_v4_flash" in tokenizer._VENDORED_MODEL_TYPES


def test_left_padded_rows_keep_independent_engram_and_indexpool_state(cpu_model):
    model = cpu_model.Model(tiny())
    tokens = np.array([[1, 2, 3, 4], [0, 0, 3, 4]])
    cache = model.make_cache()
    for layer in cache:
        for c in layer:
            c.offset = np.array([0, -2])
    batched = model(tokens, cache)
    short_cache = model.make_cache()
    short = model(np.array([[3, 4]]), short_cache)
    np.testing.assert_allclose(batched[1:, -2:], short, atol=2e-5)
    decoded = model(np.array([[5], [5]]), cache)
    expected = model(np.array([[5]]), short_cache)
    np.testing.assert_allclose(decoded[1:], expected, atol=2e-5)


@pytest.mark.parametrize("fp8", [False, True])
def test_import_worker_dispatches_hy4_conversion(monkeypatch, tmp_path, fp8):
    import rapid_mlx.utils.tokenizer as tokenizer
    from rapid_mlx.byom import _import_worker as worker
    from rapid_mlx.models import hy_v4_flash_convert
    from rapid_mlx.utils import model_file_guard

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "hy_v4_flash"}))
    fake = ModuleType("mlx_lm")
    utils = ModuleType("mlx_lm.utils")
    converted = []
    fake.convert = lambda **kwargs: converted.append(kwargs)
    utils._download = lambda _: tmp_path
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)
    monkeypatch.setattr(tokenizer, "_register_vendored_archs", lambda: None)
    monkeypatch.setattr(
        model_file_guard, "validate_model_code", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        model_file_guard, "validate_local_model_file", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(hy_v4_flash_convert, "convert_fp8", lambda *args: fp8)
    worker.convert(str(tmp_path), str(tmp_path / "out"), 4, 64)
    assert len(converted) == (0 if fp8 else 1)
    if converted:
        assert converted[0]["q_group_size"] == 32
        predicate = converted[0]["quant_predicate"]
        assert predicate("model.layers.1.self_attn.q_b_proj", None)["group_size"] == 64
        assert (
            predicate(
                "model.layers.1.engram_layer.multi_head_embedding.embedding", None
            )["group_size"]
            == 32
        )


def test_phase_one_text_lane_wins_over_vision_metadata(tmp_path):
    from rapid_mlx.api.utils import is_mllm_model

    fixture = Path(__file__).parent / "fixtures/hy_v4_flash"
    (tmp_path / "config.json").write_text((fixture / "config.json").read_text())
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "vit.weight": "absent",
                    "model.embed_tokens.weight": "absent",
                }
            }
        )
    )
    assert is_mllm_model(str(tmp_path)) is False


@pytest.mark.parametrize(
    "quant", [None, {"quant_method": "fp8", "weight_block_size": [128, 128]}]
)
@pytest.mark.parametrize("nested", [False, True])
def test_import_preflight_accepts_hy4_sources_and_sizes_hash_tables(quant, nested):
    from rapid_mlx.byom import imports as im
    from rapid_mlx.byom import preflight as pf

    config = {"model_type": "hy_v4_flash"}
    metadata = config.setdefault("text_config", {}) if nested else config
    if quant:
        metadata["quantization_config"] = quant
    inspection = pf.Inspection(
        ref="org/Hy4-flash",
        is_local=False,
        files=("model.safetensors", "config.json"),
        weight_bytes=1024,
        config=config,
        dtype="bf16",
        revision="fixed-revision",
    )
    plan = im.plan_import(
        "org/Hy4-flash",
        bits=4,
        name="hy4-test-local",
        inspection=inspection,
        supported=frozenset({"hy_v4_flash"}),
        hub_cached=True,
    )
    assert plan.output_bytes == 640
    assert plan.download_bytes == 0
    metadata["quantization_config"] = {
        "quant_method": "fp8",
        "weight_block_size": [64, 64],
    }
    with pytest.raises(im.ImportRefusedError, match="already quantized"):
        im.plan_import(
            "org/Hy4-flash",
            bits=4,
            name=None,
            inspection=inspection,
            supported=frozenset({"hy_v4_flash"}),
            hub_cached=True,
        )


def test_short_frozen_maps_are_preserved_and_must_agree(cpu_model):
    model = cpu_model.Model(tiny())
    key = "model.layers.1.engram_layer.lookup_table"
    lookup = np.arange(10, dtype=np.int32)
    assert model.sanitize({key: lookup})[key] is lookup
    assert model.layers[1].engram_layer.lookup_table.shape == (10,)
    model.sanitize({key: lookup.copy()})
    with pytest.raises(ValueError, match="share the same"):
        model.sanitize({key: lookup // 2})
    with pytest.raises(ValueError, match="token map"):
        model.sanitize({key: lookup.astype(np.float32)})
    ids = np.array([[1, 2, 12]], np.int32)
    assert np.isfinite(model(ids)).all()


def test_hadamard_fp8_matches_dense_matrix_and_scalar_codebook():
    rng = np.random.default_rng(24)
    x = rng.normal(size=(3, 128)).astype(np.float32)
    hadamard = np.array(
        [[(-1) ** ((i & j).bit_count()) for j in range(128)] for i in range(128)],
        np.float32,
    ) / np.sqrt(128)
    rotated = x @ hadamard
    scale = 2.0 ** np.ceil(
        np.log2(np.maximum(np.max(abs(rotated), axis=-1, keepdims=True), 1e-4) / 448)
    )
    codes = np.arange(127, dtype=np.uint8)[None]
    positive = dequantize_fp8(codes, np.ones(1))
    codebook = np.concatenate((-positive[0, 1:][::-1], positive[0]))
    normalized = rotated / scale
    nearest = np.argmin(abs(normalized[..., None] - codebook), axis=-1)
    expected = codebook[nearest] * scale
    np.testing.assert_allclose(ops.fwht_fp8(x, np), expected, atol=1e-7)
    # Fully padded query rows attend only to the zero-valued sink.
    q = np.zeros((1, 2, 1, 4), np.float32)
    np.testing.assert_array_equal(
        ops.attention(
            q, q, np.ones_like(q), np.zeros((1, 1, 1), bool), np.zeros(2), np
        ),
        np.zeros_like(q),
    )


def test_router_precision_does_not_change_dense_or_routed_gate_precision(cpu_model):
    predicate = cpu_model.Model(tiny()).cast_predicate
    assert not predicate("model.layers.1.mlp.gate.weight")
    assert predicate("model.layers.0.mlp.gate_proj.weight")
    assert predicate("model.layers.1.mlp.switch_mlp.gate_proj.weight")


def test_checkpoint_metadata_outranks_parent_family_names(tmp_path):
    from rapid_mlx.model_auto_config import detect_model_config

    fixture = Path(__file__).parent / "fixtures/hy_v4_flash"
    path = tmp_path / "qwen3-parent" / "renamed"
    path.mkdir(parents=True)
    (path / "config.json").write_text((fixture / "config.json").read_text())
    profile = detect_model_config(str(path))
    assert profile.tool_call_parser == "hy_v4"
    assert profile.reasoning_parser == "hy_v4"
    assert profile.is_moe
    assert profile.is_text_only
    assert profile.supports_spec_decode is False
    # The same parent still routes an unrelated checkpoint through its fallback.
    (path / "config.json").write_text(json.dumps({"model_type": "unknown-control"}))
    profile = detect_model_config(str(path))
    assert profile is None or profile.tool_call_parser != "hy_v4"


@pytest.mark.parametrize("kind", ["traversal", "symlink", "single_symlink"])
def test_fp8_conversion_rejects_external_shards_before_open(
    fp8_conversion, tmp_path, monkeypatch, kind
):
    converter, _, _, _ = fp8_conversion
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text((tmp_path / "config.json").read_text())
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"must not be read")
    filename = "../outside.safetensors"
    if kind != "traversal":
        filename = "model.safetensors"
        (source / filename).symlink_to(outside)
    if kind != "single_symlink":
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": filename}})
        )
    fp8 = sys.modules["rapid_mlx.fp8_repack"]

    def forbidden(*args, **kwargs):
        pytest.fail("External shard opened")

    monkeypatch.setattr(fp8, "_ShardReader", forbidden, raising=False)
    monkeypatch.setattr(fp8, "_open_shards", forbidden, raising=False)
    with pytest.raises(ValueError, match="inside|escapes"):
        converter.convert_fp8(source, tmp_path / "out", 4, 32)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_tokenizer_hy4_fp8_requires_conversion(
    cpu_model, monkeypatch, tmp_path, nested, fp8
):
    from rapid_mlx.utils import tokenizer

    config = {"model_type": "hy_v4_flash"}
    metadata = config.setdefault("text_config", {}) if nested else config
    metadata["quantization_config" if fp8 else "quantization"] = (
        {"quant_method": "fp8", "weight_block_size": [128, 128]}
        if fp8
        else {"bits": 4, "group_size": 32}
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(tokenizer, "_register_vendored_archs", lambda: None)
    monkeypatch.setattr(
        tokenizer, "_deepseek_v4_quantization_override", lambda *a, **k: None
    )
    monkeypatch.setattr(tokenizer, "_uses_rapid_owned_runtime", lambda *a: True)

    def forbidden(*args, **kwargs):
        pytest.fail("Hy4 must not enter generic model/FP8 loading")

    repack = ModuleType("rapid_mlx.fp8_repack")
    repack.is_fp8_block_checkpoint = forbidden
    repack.load_fp8_model_online = forbidden
    monkeypatch.setitem(sys.modules, "rapid_mlx.fp8_repack", repack)
    utils = ModuleType("mlx_lm.utils")
    utils.load_model = forbidden
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)
    loader = ModuleType("rapid_mlx.models.hy_v4_flash_load")

    class LoadedHy4Error(Exception):
        pass

    def load(path):
        assert path == tmp_path
        if fp8:
            pytest.fail("Raw FP8 reached the Hy4 weight loader")
        raise LoadedHy4Error

    loader.load = load
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.hy_v4_flash_load", loader)
    if fp8:
        with pytest.raises(tokenizer.IncompatibleWeights, match="rapid-mlx import"):
            tokenizer._load_with_tokenizer_fallback(str(tmp_path))
    else:
        with pytest.raises(LoadedHy4Error):
            tokenizer._load_with_tokenizer_fallback(str(tmp_path))


@pytest.mark.parametrize("key", ["quantization", "quantization_config"])
def test_checkpoint_quantization_prefers_top_level(key):
    from rapid_mlx.models.hy_v4_flash_config import checkpoint_quantization

    outer = {"bits": 4, "group_size": 32}
    nested = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    config = {key: outer, "text_config": {"quantization_config": nested}}
    assert checkpoint_quantization(config) is outer
    del config[key]
    assert checkpoint_quantization(config) is nested
