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
    ]
    for config in cases:
        with pytest.raises(ValueError):
            ModelArgs.from_dict(config)


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


def test_fp8_conversion_uses_standard_quantization_and_saver(
    cpu_model, monkeypatch, tmp_path, caplog
):
    from dataclasses import asdict

    from rapid_mlx.models.hy_v4_flash_convert import convert_fp8

    config = asdict(tiny())
    (tmp_path / "config.json").write_text(json.dumps(config))
    arrays = {
        "model.layers.1.mlp.experts.gate_up_proj": (
            np.full((3, 8, 4), 56, np.uint8),
            "F8_E4M3",
        ),
        "model.layers.1.mlp.experts.gate_up_proj_scale_inv": (
            np.full((3, 1, 1), 2.0, np.float32),
            "F32",
        ),
        "model.norm.weight": (np.ones(8, np.float32), "F32"),
        "model.layers.1.engram_layer.lookup_table": (
            np.arange(16, dtype=np.int64),
            "I64",
        ),
        "vit.weight": (np.ones(2, np.float32), "F32"),
    }

    class Shard:
        @property
        def header(self):
            return {k: {"dtype": t} for k, (v, t) in arrays.items()}

        def read(self, name):
            v, t = arrays[name]
            return np.frombuffer(v.tobytes(), dtype=np.uint8), t, list(v.shape)

    shard = Shard()
    fp8 = ModuleType("rapid_mlx.fp8_repack")
    fp8._open_shards = lambda _: ({k: "shard" for k in arrays}, {"shard": shard})
    fp8._raw_to_mx = lambda raw, dtype, shape: raw.view(np.float32).reshape(shape)
    monkeypatch.setitem(sys.modules, "rapid_mlx.fp8_repack", fp8)
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.hy_v4_flash", cpu_model)
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda x: list(x.items())
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    lm_utils = ModuleType("mlx_lm.utils")
    expected, loaded, quantized, saved = {}, [], [], []
    for projection in ("gate_proj", "up_proj"):
        base = f"model.layers.1.mlp.switch_mlp.{projection}"
        expected.update(
            {base + "." + leaf: np.zeros(1) for leaf in ("weight", "scales", "biases")}
        )
    cpu_model.Model.parameters = lambda self: expected
    cpu_model.Model.load_weights = lambda self, weights, strict: loaded.extend(weights)
    cpu_model.Model.eval = lambda self: None
    lm_utils.quantize_model = lambda model, config, group, bits, **kwargs: (
        model,
        {**config, "quantization": {"group_size": 64, "bits": bits}},
    )
    lm_utils.load_tokenizer = lambda _: "tokenizer"
    lm_utils.save = lambda *args: saved.append(args)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", lm_utils)
    mx = sys.modules["mlx.core"]

    def quantize(value, **params):
        quantized.append((value.copy(), params))
        return value, np.ones(1), np.zeros(1)

    mx.quantize = quantize
    mx.eval = lambda *args: None
    with caplog.at_level("INFO"):
        assert convert_fp8(tmp_path, tmp_path / "out", 4, 64)
    assert "text-only" in caplog.text
    assert len(quantized) == 2
    for value, params in quantized:
        np.testing.assert_array_equal(value, np.full((3, 4, 4), 2.0))
        assert params == {"group_size": 64, "bits": 4}
    assert len(saved) == 1 and saved[0][3] == "tokenizer"
    assert not any("vit" in key or "scale_inv" in key for key, value in loaded)
    arrays["model.layers.1.engram_layer.lookup_table"] = (
        np.arange(16, dtype=np.int32),
        "I32",
    )
    assert convert_fp8(tmp_path, tmp_path / "out2", 4, 64)
    with pytest.raises(ValueError, match="already exists"):
        convert_fp8(tmp_path, tmp_path, 4, 64)
    arrays.pop("model.layers.1.mlp.experts.gate_up_proj")
    assert convert_fp8(tmp_path, tmp_path / "out3", 4, 64) is False
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "other"}))
    assert convert_fp8(tmp_path, tmp_path / "out4", 4, 64) is False


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
    tokenizer._register_vendored_archs()
    assert sys.modules["mlx_lm.models.hy_v4_flash"] is cpu_model


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
def test_import_preflight_accepts_hy4_sources_and_sizes_hash_tables(quant):
    from rapid_mlx.byom import imports as im
    from rapid_mlx.byom import preflight as pf

    config = {"model_type": "hy_v4_flash"}
    if quant:
        config["quantization_config"] = quant
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
    config["quantization_config"] = {
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
