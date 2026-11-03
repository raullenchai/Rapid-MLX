# SPDX-License-Identifier: Apache-2.0
"""Tiny Flint contracts; never open released checkpoint weights."""

import importlib
import json
import subprocess
import sys
import types

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from rapid_mlx.models import flint  # noqa: E402

TINY = dict(
    model_type="flint",
    hidden_size=64,
    num_hidden_layers=4,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    mlp_only_layers=[0],
    layer_types=[
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    sliding_window=7,
    rms_norm_eps=1e-6,
    vocab_size=128,
)


def tiny_model(**changes):
    mx.random.seed(7)
    model = flint.Model(flint.ModelArgs.from_dict(dict(TINY, **changes)))
    for layer in model.layers:
        layer.self_attn.softmax_offset = mx.array([-1.0, 0.5, 1.0, 2.0])
    return model


def assert_close(a, b):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("length", [4, 23])
@pytest.mark.parametrize("step", [1, 3, 11])
def test_incremental_and_chunked_prefill_match_full(length, step):
    model = tiny_model()
    ids = mx.array([[i % 128 for i in range(length)]])
    expected = model(ids)
    cache = model.make_cache()
    got = mx.concatenate(
        [model(ids[:, i : i + step], cache=cache) for i in range(0, length, step)],
        axis=1,
    )
    assert_close(got, expected)
    assert [c.offset for c in cache] == [length] * 4
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    assert [type(c) for c in cache] == [
        KVCache,
        RotatingKVCache,
        RotatingKVCache,
        KVCache,
    ]
    assert cache[1].max_size == 7


@pytest.mark.parametrize("kind", ["full_attention", "sliding_attention"])
def test_single_kind_masks_and_input_embeddings(kind):
    model = tiny_model(layer_types=[kind] * 4)
    ids = mx.array([[1, 2, 3]])
    assert_close(model(ids), model(ids, input_embeddings=model.model.embed_tokens(ids)))
    assert_close(model(ids), model(ids, cache=model.make_cache()))
    with pytest.raises(ValueError, match="cache length"):
        model(ids, cache=[None])


@pytest.mark.parametrize("step", [1, 3, 11])
def test_batch_cache_left_padding_and_rotation(step):
    # This is the same cache factory used by the continuous batch generator.
    generate = importlib.import_module("mlx_lm.generate")
    model = tiny_model()
    sequences = [list(range(1, 19)), list(range(3, 16))]
    padding = [0, 5]
    padded = mx.array([sequences[0], [0] * 5 + sequences[1]])
    cache = generate._make_cache(model, padding, None)
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

    assert isinstance(cache[0], BatchKVCache)
    assert isinstance(cache[1], BatchRotatingKVCache)
    outputs = mx.concatenate(
        [model(padded[:, i : i + step], cache=cache) for i in range(0, 18, step)],
        axis=1,
    )
    for b, seq in enumerate(sequences):
        assert_close(outputs[b, padding[b] :], model(mx.array([seq]))[0])
    for token in [20, 21, 22]:
        logits = model(mx.array([[token], [token]]), cache=cache)
        for b, seq in enumerate(sequences):
            seq.append(token)
            assert_close(logits[b, -1], model(mx.array([seq]))[0, -1])


@pytest.mark.parametrize("length", [4, 23])
def test_prefix_store_fetch_continue_matches_cold(length):
    from rapid_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    model = tiny_model()
    ids = list(range(1, length + 1))
    cache = model.make_cache()
    # Token-by-token forces actual ring rotation rather than just a long buffer.
    for token in ids:
        mx.eval(model(mx.array([[token]]), cache=cache))
    prefix = MemoryAwarePrefixCache(None, MemoryCacheConfig(hybrid_reuse_max_entries=4))
    assert prefix.store(ids, cache, evict_prefixes=False)
    warm, rest = prefix.fetch(ids + [30, 31])
    assert warm is not None and rest == [30, 31]
    assert_close(
        model(mx.array([rest]), cache=warm), model(mx.array([ids + rest]))[:, -2:]
    )
    if length > TINY["sliding_window"]:
        assert not cache[1].is_trimmable()
        exact, rest = prefix.fetch(ids)
        assert exact is None and rest == ids
        rewind, rest = prefix.fetch(ids[:-2])
        assert rewind is None and rest == ids[:-2]
    else:
        rewind, rest = prefix.fetch(ids[:-1] + [40])
        assert rewind is not None
        assert_close(
            model(mx.array([rest]), cache=rewind)[:, -1],
            model(mx.array([ids[:-1] + [40]]))[:, -1],
        )


def test_sinks_match_explicit_softmax():
    model = tiny_model()
    attn = model.layers[0].self_attn
    x = mx.random.normal((1, 5, 64))
    q = attn.q_norm(attn.q_proj(x).reshape(1, 5, 4, 16)).transpose(0, 2, 1, 3)
    k = attn.k_norm(attn.k_proj(x).reshape(1, 5, 2, 16)).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(1, 5, 2, 16).transpose(0, 2, 1, 3)
    k, v = mx.repeat(k, 2, axis=1), mx.repeat(v, 2, axis=1)
    scores = (q @ k.swapaxes(-1, -2)) * attn.scale
    mask = mx.arange(5)[:, None] >= mx.arange(5)[None, :]
    scores = mx.where(mask, scores, -mx.inf)
    sinks = mx.broadcast_to(attn.softmax_offset[None, :, None, None], (1, 4, 5, 1))
    probs = mx.softmax(mx.concatenate([scores, sinks], axis=-1), axis=-1)[..., :-1]
    expected = attn.o_proj((probs @ v).transpose(0, 2, 1, 3).reshape(1, 5, 64))
    actual = attn(x, mask=mask)
    assert_close(actual, expected)
    attn.softmax_offset = mx.zeros((4,))
    assert float(mx.max(mx.abs(actual - attn(x, mask=mask)))) > 1e-4


def test_router_fp32_sigmoid_selection_and_guard():
    model = tiny_model()
    block = model.layers[1].mlp
    x = mx.random.normal((1, 3, 64))
    block.gate.weight = block.gate.weight.astype(mx.bfloat16)
    scores = mx.sigmoid(x.astype(mx.float32) @ block.gate.weight.astype(mx.float32).T)
    inds = mx.argsort(-scores, axis=-1)[..., :2]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    weights /= weights.sum(axis=-1, keepdims=True)
    y = block.switch_mlp(x, inds)
    assert_close(
        block(x), (y * weights[..., None]).sum(axis=-2) + block.shared_expert(x)
    )
    block.gate.weight = mx.full((8, 64), -1e6)
    assert bool(mx.all(mx.isfinite(block(mx.ones((1, 1, 64))))))


def test_sanitize_bf16_experts_tied_head_and_conversion_router():
    model = tiny_model()
    original = dict(tree_flatten(model.parameters()))
    exploded = {}
    for key, value in original.items():
        if ".switch_mlp." in key:
            prefix, suffix = key.split(".switch_mlp.")
            for e in range(8):
                exploded[f"{prefix}.experts.{e}.{suffix}"] = value[e]
        else:
            exploded[key] = value
    sanitized = model.sanitize(exploded)
    assert sanitized.keys() == original.keys()
    for key in original:
        assert_close(sanitized[key], original[key])
    model.load_weights(list(sanitized.items()), strict=True)
    tied = tiny_model(tie_word_embeddings=True)
    assert tied.sanitize({"lm_head.weight": mx.zeros((1,))}) == {}
    assert tied(mx.array([[1, 2]])).shape == (1, 2, 128)
    from mlx_lm.utils import quantize_model

    converted, _ = quantize_model(model, dict(TINY), group_size=64, bits=4)
    assert not isinstance(converted.layers[1].mlp.gate, nn.QuantizedLinear)


def save_tiny_checkpoint(path, quantized_router=False):
    model = tiny_model()
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        class_predicate=lambda p, m: (
            {"bits": 8, "group_size": 64}
            if p == "model.embed_tokens"
            else hasattr(m, "to_quantized")
            and (quantized_router or model.quant_predicate(p, m))
        ),
    )
    weights = dict(tree_flatten(model.parameters()))
    for key, value in list(weights.items()):
        if key.endswith(".weight") and key[:-7] + ".scales" in weights:
            weights[key] = value.view(mx.uint8)
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    config = dict(TINY, quantization={"bits": 4, "group_size": 64})
    (path / "config.json").write_text(json.dumps(config))
    # A real local tokenizer lets the complete Rapid text loader run offline.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>"
    ).save_pretrained(path)
    return model


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("quantized_router", [False, True])
def test_mixed_bit_u8_checkpoint_through_rapid_text_loader(
    tmp_path, lazy, quantized_router, monkeypatch
):
    from rapid_mlx.utils.tokenizer import load_model_with_fallback

    expected = save_tiny_checkpoint(tmp_path, quantized_router)
    observed = []
    original = flint.ModelArgs.from_dict.__func__

    def capture(cls, config):
        args = original(cls, config)
        assert args._load_config is config
        observed.append(config)
        return args

    monkeypatch.setattr(flint.ModelArgs, "from_dict", classmethod(capture))
    loaded, _ = load_model_with_fallback(str(tmp_path), lazy=lazy)
    assert observed
    assert loaded.args._load_config is observed[-1]
    assert observed[-1]["quantization"]["model.embed_tokens"] == {
        "bits": 8,
        "group_size": 64,
    }
    assert loaded.model.embed_tokens.bits == 8
    assert loaded.lm_head.bits == 4
    assert isinstance(loaded.layers[1].mlp.gate, nn.QuantizedLinear) == quantized_router
    assert loaded.layers[1].self_attn.softmax_offset.shape == (4,)
    assert_close(loaded(mx.array([[1, 2, 3]])), expected(mx.array([[1, 2, 3]])))


@pytest.mark.parametrize("storage", [mx.uint8, mx.uint32])
def test_sanitize_packed_width_validation(storage):
    config = dict(TINY, quantization={"bits": 4, "group_size": 64})
    model = flint.Model(flint.ModelArgs.from_dict(config))
    key = "model.embed_tokens"
    packed, scales, biases = mx.quantize(
        model.model.embed_tokens.weight, bits=8, group_size=64
    )
    weights = {
        key + ".weight": packed.view(storage),
        key + ".scales": scales,
        key + ".biases": biases,
    }
    assert model.sanitize(dict(weights))[key + ".weight"].dtype == mx.uint32
    for bad, match in [
        ({key + ".weight": packed.astype(mx.float32)}, "uint8 or uint32"),
        ({key + ".weight": packed[..., :1]}, "infer supported"),
        ({key + ".scales": scales[..., :0]}, "input width"),
        ({key + ".scales": mx.zeros((1, 1))}, "shape mismatch"),
    ]:
        with pytest.raises(ValueError, match=match):
            model.sanitize(dict(weights, **bad))
    model.args._load_config = {}
    with pytest.raises(ValueError, match="requires quantization"):
        model.sanitize(weights)


@pytest.mark.parametrize("dtype", ["int4", "int8"])
@pytest.mark.parametrize("explicit", [True, False])
def test_sink_quantized_kv_admission_before_ready(dtype, explicit):
    from rapid_mlx.kv_cache_dtype import KVCacheQuantizationUnsupportedError
    from rapid_mlx.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = types.SimpleNamespace(
        kv_cache_quantization=True,
        kv_cache_quantization_bits=int(dtype[3:]),
        kv_cache_dtype=dtype,
        kv_cache_dtype_explicit=explicit,
        kv_cache_turboquant=None,
        kv_cache_quantization_group_size=64,
        model_name="tiny-flint",
    )
    model = tiny_model()
    assert "attention sinks" in Scheduler._quantized_attention_incompatibility(model)
    if explicit:
        with pytest.raises(
            KVCacheQuantizationUnsupportedError, match="attention sinks"
        ):
            scheduler._init_kv_quantization(model)
    else:
        scheduler._init_kv_quantization(model)
        assert scheduler._kv_quant_live_disabled


def test_registration_always_overrides_native(monkeypatch):
    from rapid_mlx.utils import tokenizer as tok

    monkeypatch.delitem(sys.modules, "mlx_lm.models.flint", raising=False)
    tok._register_vendored_archs()
    assert sys.modules["mlx_lm.models.flint"] is flint
    assert "flint" in tok._VENDORED_MODEL_TYPES
    native = types.ModuleType("mlx_lm.models.flint")
    monkeypatch.setitem(sys.modules, "mlx_lm.models.flint", native)
    # Other registered architecture keys must remain untouched.
    other = sys.modules["mlx_lm.models.g9v3"]
    tok._register_vendored_archs()
    assert sys.modules["mlx_lm.models.flint"] is flint
    assert sys.modules["mlx_lm.models.g9v3"] is other
    assert "flint" in tok._VENDORED_MODEL_TYPES
    # Even a discoverable native module must not be preferred.
    monkeypatch.delitem(sys.modules, "mlx_lm.models.flint")
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            importlib.util.spec_from_loader(name, loader=None)
            if name == "mlx_lm.models.flint"
            else real(name)
        ),
    )
    tok._register_vendored_archs()
    assert sys.modules["mlx_lm.models.flint"] is flint
    assert "flint" in tok._VENDORED_MODEL_TYPES


@pytest.mark.parametrize(
    "change, message",
    [
        ({"layer_types": []}, "every layer"),
        ({"layer_types": ["unknown"] * 4}, "unsupported"),
        ({"sliding_window": 0}, "positive"),
        ({"num_experts_per_tok": 9}, "expert count"),
    ],
)
def test_config_rejects_unsupported_layout(change, message):
    with pytest.raises(ValueError, match=message):
        flint.ModelArgs.from_dict(dict(TINY, **change))


def test_rope_parameters_override():
    args = flint.ModelArgs.from_dict(
        dict(TINY, rope_parameters={"rope_theta": 20000.0})
    )
    assert args.rope_theta == 20000.0


def test_turboquant_skips_flint_window():
    from rapid_mlx.turboquant import is_incompatible_with_turboquant

    skip, reason = is_incompatible_with_turboquant(hf_config=dict(TINY))
    assert skip and "sliding" in reason


def test_sanitize_rejects_non_affine_and_unknown_packed_modules():
    model = tiny_model()
    model.args._load_config = {
        "quantization": {"bits": 4, "group_size": 64, "mode": "mxfp4"}
    }
    packed, scales, biases = mx.quantize(model.lm_head.weight, bits=4, group_size=64)
    weights = {
        "lm_head.weight": packed,
        "lm_head.scales": scales,
        "lm_head.biases": biases,
    }
    with pytest.raises(ValueError, match="supported affine"):
        model.sanitize(weights)
    model.args._load_config["quantization"].pop("mode")
    with pytest.raises(ValueError, match="cannot be quantized"):
        model.sanitize({"unknown.weight": packed, "unknown.scales": scales})
    raw = mx.zeros((8,), dtype=mx.uint8)
    assert model.sanitize({"unpacked.weight": raw})["unpacked.weight"].dtype == mx.uint8


def test_registration_handles_import_failure(monkeypatch, caplog):
    import rapid_mlx.models
    from rapid_mlx.utils import tokenizer as tok

    monkeypatch.delitem(sys.modules, "mlx_lm.models.flint", raising=False)
    monkeypatch.delattr(rapid_mlx.models, "flint", raising=False)
    monkeypatch.setitem(sys.modules, "rapid_mlx.models.flint", None)
    tok._register_vendored_archs()
    assert "mlx_lm.models.flint" not in sys.modules
    assert "flint vendored module failed to register" in caplog.text


def test_quantized_router_matches_dequantized_fp32_reference():
    block = tiny_model().layers[1].mlp
    block.gate = block.gate.to_quantized(group_size=64, bits=4)
    assert block.gate.weight.dtype == mx.uint32
    x = mx.random.normal((1, 3, 64))
    # Compute an independent dense reference with unpacked weights.
    weight = mx.dequantize(
        block.gate.weight,
        block.gate.scales,
        block.gate.biases,
        group_size=64,
        bits=4,
    )
    logits = (x.astype(mx.float32) @ weight.astype(mx.float32).T).astype(x.dtype)
    scores = mx.sigmoid(logits.astype(mx.float32))
    indices = mx.argsort(-scores, axis=-1)[..., :2]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights /= weights.sum(axis=-1, keepdims=True)
    routed = block.switch_mlp(x, indices)
    expected = (routed * weights[..., None]).sum(axis=-2).astype(routed.dtype)
    expected += block.shared_expert(x)
    assert_close(block(x), expected)


def test_bf16_per_expert_checkpoint_through_import_worker(tmp_path):
    from mlx_lm import load
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from rapid_mlx.byom import _import_worker as worker
    from rapid_mlx.byom.imports import WORKER_BOOTSTRAP

    source, output = tmp_path / "bf16", tmp_path / "converted"
    source.mkdir()
    config = dict(
        TINY,
        num_hidden_layers=2,
        num_experts=2,
        layer_types=["full_attention", "sliding_attention"],
        torch_dtype="bfloat16",
    )
    model = tiny_model(**config)
    weights = {
        key: value.astype(mx.bfloat16)
        for key, value in tree_flatten(model.parameters())
    }
    router = weights["model.layers.1.mlp.gate.weight"]
    # Write the HF BF16 layout, with separate tensors for each expert.
    for key in list(weights):
        if ".switch_mlp." in key:
            stacked = weights.pop(key)
            for expert in range(config["num_experts"]):
                weights[key.replace(".switch_mlp.", f".experts.{expert}.")] = stacked[
                    expert
                ]
    assert not any(".switch_mlp." in key for key in weights)
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    (source / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "Hello": 1}, unk_token="<unk>"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>"
    ).save_pretrained(source)

    assert worker.main(["convert", str(source), str(output), "4", "64"]) == 0
    converted, _ = load(str(output))
    gate = converted.layers[1].mlp.gate
    assert isinstance(gate, nn.Linear) and not isinstance(gate, nn.QuantizedLinear)
    assert gate.weight.dtype == mx.bfloat16
    assert mx.array_equal(gate.weight, router).item()
    assert converted.model.embed_tokens.bits == 4
    assert converted.layers[1].self_attn.q_proj.bits == 4
    assert converted.layers[1].mlp.switch_mlp.up_proj.bits == 4
    assert converted.layers[1].mlp.switch_mlp.up_proj.weight.shape[:2] == (2, 64)
    assert converted.lm_head.bits == 4
    assert mx.all(mx.isfinite(converted(mx.array([[1, 2]])))).item()
    # Match production's fresh interpreter: other tests can leave mlx-lm's
    # process-global generation_stream pointing at another thread's GPU stream.
    result = subprocess.run(
        [sys.executable, "-c", WORKER_BOOTSTRAP, "smoke", str(output)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
