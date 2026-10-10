# SPDX-License-Identifier: Apache-2.0
"""Selective live attention KV with bounded recurrent state (#4367)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from mlx_lm.models.cache import ArraysCache, KVCache

from rapid_mlx.kv_cache_dtype import KVCacheQuantizationUnsupportedError
from rapid_mlx.models.mlx_vlm_vendored.cache import ArraysCache as VendoredArraysCache
from rapid_mlx.quantized_batch_cache import (
    _QuantizableKVCache,
    install_quantized_batch_cache,
    normalize_caches_for_quantization,
    supported_recurrent_cache_types,
)
from tests.test_kv_cache_gemma4_gate import _scheduler_stub


def test_recurrent_types_without_optional_vlm(monkeypatch):
    import sys

    # Installed and absent optional dependencies must expose the same core types.
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", None)
    assert supported_recurrent_cache_types() == (ArraysCache, VendoredArraysCache)


@pytest.mark.parametrize("state_type", [ArraysCache, VendoredArraysCache])
@pytest.mark.parametrize("bits", [4, 8])
def test_recurrent_hybrid_enables_only_attention_quantization(state_type, bits, caplog):
    class Hybrid:
        args = SimpleNamespace(head_dim=64)

        def make_cache(self):
            return [state_type(size=2), KVCache(), state_type(size=2)]

    scheduler = _scheduler_stub(explicit=True)
    scheduler.config.kv_cache_quantization_bits = bits
    with caplog.at_level("INFO"):
        scheduler._init_kv_quantization(Hybrid())
    assert not scheduler._kv_quant_live_disabled
    layout = scheduler._kv_quant_layout
    assert (
        layout.quantizable_layers,
        layout.recurrent_layers,
        layout.total_layers,
    ) == (
        1,
        2,
        3,
    )
    assert f"1/3 full-attention layers use int{bits}" in caplog.text
    assert "2 bounded recurrent layers remain unchanged" in caplog.text


@pytest.mark.parametrize("state_type", [ArraysCache, VendoredArraysCache])
@pytest.mark.parametrize("bits", [4, 8])
def test_fresh_and_restored_recurrent_state_is_untouched(state_type, bits):
    state = state_type(size=2)
    state[0] = mx.ones((1, 3, 64), dtype=mx.bfloat16)
    state[1] = mx.ones((1, 2, 64, 64), dtype=mx.float32)
    tensors = (state[0], state[1])
    plain = KVCache()
    keys = mx.ones((1, 2, 7, 64), dtype=mx.bfloat16)
    plain.update_and_fetch(keys, keys)
    generator = SimpleNamespace(_make_new_cache=lambda: [state, KVCache()])
    assert install_quantized_batch_cache(generator, group_size=64, bits=bits)
    fresh = generator._make_new_cache()
    restored = normalize_caches_for_quantization([state, plain], 64, bits)
    for caches in (fresh, restored):
        assert caches[0] is state
        assert state[0] is tensors[0] and state[1] is tensors[1]
        assert isinstance(caches[1], _QuantizableKVCache)
        assert caches[1].q_bits == bits
    assert restored[1].offset == 7
    assert restored[1].keys is plain.keys
    assert restored[1].values is plain.values


@pytest.mark.parametrize("case", ["only_state", "impostor", "subclass", "unknown"])
def test_unsupported_layouts_still_fail_closed(case):
    class DerivedArraysCache(ArraysCache):
        pass

    state = {
        "only_state": ArraysCache(size=2),
        "impostor": type("ArraysCache", (), {})(),
        "subclass": DerivedArraysCache(size=2),
        "unknown": object(),
    }[case]

    class Hybrid:
        args = SimpleNamespace(head_dim=64)

        def make_cache(self):
            return (
                [state]
                if case == "only_state"
                else [ArraysCache(size=2), KVCache(), state]
            )

    with pytest.raises(KVCacheQuantizationUnsupportedError):
        _scheduler_stub(explicit=True)._init_kv_quantization(Hybrid())
    scheduler = _scheduler_stub(explicit=False)
    scheduler._init_kv_quantization(Hybrid())
    assert scheduler._kv_quant_live_disabled


def _tiny_qwen_next():
    from mlx_lm.models.qwen3_next import Model, ModelArgs

    mx.random.seed(4367)
    return Model(
        ModelArgs(
            model_type="qwen3_next",
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            num_experts=2,
            num_experts_per_tok=1,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=64,
            mlp_only_layers=[],
            moe_intermediate_size=64,
            rms_norm_eps=1e-6,
            vocab_size=32,
            rope_theta=10000,
            partial_rotary_factor=0.5,
            max_position_embeddings=128,
            full_attention_interval=2,
        )
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_qwen_next_chunked_prefill_and_decode_with_quantized_live_kv(bits):
    from mlx_lm.generate import BatchGenerator

    from rapid_mlx.quantized_batch_cache import QuantizedBatchKVCache

    model = _tiny_qwen_next()
    scheduler = _scheduler_stub(explicit=True)
    scheduler.config.kv_cache_quantization_bits = bits
    scheduler._init_kv_quantization(model)
    assert not scheduler._kv_quant_live_disabled
    generator = BatchGenerator(
        model,
        max_tokens=3,
        prefill_step_size=4,
        prefill_batch_size=1,
        completion_batch_size=1,
        stream=mx.default_stream(mx.default_device()),
    )
    seen = []
    original_call = model.__class__.__call__

    # Observe actual caches consumed by the model, including prefill + decode.
    def observed_call(self, inputs, cache=None):
        if cache:
            seen.append((inputs.shape[1], tuple(type(c) for c in cache)))
        return original_call(self, inputs, cache=cache)

    from unittest.mock import patch

    assert install_quantized_batch_cache(generator, 64, bits)
    with patch.object(type(model), "__call__", observed_call):
        generator.insert([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        responses = []
        for _ in range(20):
            _, generated = generator.next()
            responses.extend(generated)
            if any(r.finish_reason for r in responses):
                break
    assert len(responses) == 3
    assert responses[-1].finish_reason == "length"
    assert all(0 <= r.token < 32 for r in responses)
    assert any(width > 1 for width, _ in seen)
    assert any(width == 1 for width, _ in seen)
    assert all(types == (ArraysCache, QuantizedBatchKVCache) for _, types in seen)
