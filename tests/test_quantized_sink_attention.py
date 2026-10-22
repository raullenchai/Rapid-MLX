"""Tiny packed-cache attention and hybrid serving regressions."""

import importlib
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.quantized_batch_cache import (  # noqa: E402
    QuantizedBatchKVCache,
    install_quantized_batch_cache,
    normalize_caches_for_quantization,
)
from rapid_mlx.quantized_sink_attention import quantized_sink_attention  # noqa: E402
from tests.test_flint_model import tiny_model  # noqa: E402


def close(a, b, tol=3e-5):
    np.testing.assert_allclose(
        np.array(a.astype(mx.float32)),
        np.array(b.astype(mx.float32)),
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("length", [1, 5])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "bool", "additive", "heads"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_packed_sinks_match_dense(dtype, kv_heads, length, mask_kind, head_dim):
    mx.random.seed(5083)
    q = mx.random.normal((2, 4, length, head_dim)).astype(dtype)
    k, v = [
        mx.random.normal((2, kv_heads, 9, head_dim)).astype(dtype) for _ in range(2)
    ]
    qk, qv = [mx.quantize(x, group_size=64, bits=8) for x in (k, v)]
    k, v = [mx.dequantize(*x, group_size=64, bits=8) for x in (qk, qv)]
    causal = mx.arange(9 - length, 9)[:, None] >= mx.arange(9)[None]
    mask = None
    if mask_kind == "causal":
        mask = "causal"
    elif mask_kind in ("bool", "additive", "heads"):
        # Different left padding per row, and an explicit finite window.
        mask = causal & (mx.arange(9)[None] > mx.arange(9 - length, 9)[:, None] - 4)
        mask = mask[None, None] & (
            mx.arange(9)[None, None, None] >= mx.array([0, 2])[:, None, None, None]
        )
        if mask_kind == "additive":
            mask = mx.where(mask, mx.array(-0.25, dtype), mx.array(-mx.inf, dtype))
        elif mask_kind == "heads":
            mask = mx.broadcast_to(mask, (2, 4, length, 9))
            mask = mask & (
                mx.arange(9)[None, None, None] != mx.arange(4)[None, :, None, None]
            )
    sinks = mx.array([-3.0, 0.5, 2.0, 4.0], dtype=dtype)
    actual = quantized_sink_attention(
        q, qk, qv, scale=head_dim**-0.5, mask=mask, sinks=sinks
    )
    expected = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=head_dim**-0.5, mask=mask, sinks=sinks
    )
    close(
        actual,
        expected,
        {mx.float32: 2e-6, mx.float16: 0.002, mx.bfloat16: 0.02}[dtype],
    )


def quantized_caches(model):
    # Use the serving factory hook and merge, including rotating cache dispatch.
    factory = types.SimpleNamespace(_make_new_cache=model.make_cache)
    assert install_quantized_batch_cache(
        factory, group_size=64, bits=8, cache_type=model.quantized_kv_cache_type
    )
    generate = importlib.import_module("mlx_lm.generate")
    return generate._merge_caches([factory._make_new_cache()])


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("step", [1, 3, 11])
def test_flint_int8_incremental_and_chunked_rotation(dtype, step):
    from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache

    model = tiny_model(head_dim=64, mlp_only_layers=list(range(4)))
    model.set_dtype(dtype)
    ids = mx.array([list(range(1, 24))])
    full = model(ids, cache=model.make_cache())
    caches = [
        c.to_quantized(group_size=64, bits=8) if type(c).__name__ == "KVCache" else c
        for c in model.make_cache()
    ]
    assert isinstance(caches[0], QuantizedKVCache)
    assert isinstance(caches[1], RotatingKVCache)
    one_shot = model(
        ids,
        cache=[
            c.to_quantized(group_size=64, bits=8)
            if type(c).__name__ == "KVCache"
            else c
            for c in model.make_cache()
        ],
    )
    output = mx.concatenate(
        [model(ids[:, i : i + step], cache=caches) for i in range(0, 23, step)], axis=1
    )
    close(output, one_shot, 0.04 if dtype == mx.bfloat16 else 0.001)
    close(output, full, 0.05 if dtype == mx.bfloat16 else 0.025)
    assert caches[1].keys.dtype == dtype
    assert caches[1].offset == 23
    # Also exercise the actual single-request live batch layout.
    close(
        model(ids, cache=quantized_caches(model)),
        one_shot,
        0.04 if dtype == mx.bfloat16 else 0.001,
    )


@pytest.mark.parametrize("step", [1, 3, 11])
def test_live_batch_left_padding(step):
    generate = importlib.import_module("mlx_lm.generate")
    model = tiny_model(head_dim=64, mlp_only_layers=list(range(4)))
    seqs = [list(range(1, 19)), list(range(3, 16))]
    padded = mx.array([seqs[0], [0] * 5 + seqs[1]])
    caches = generate._make_cache(model, [0, 5], None)
    caches[0] = QuantizedBatchKVCache([0, 5])
    caches[3] = QuantizedBatchKVCache([0, 5])
    output = mx.concatenate(
        [model(padded[:, i : i + step], cache=caches) for i in range(0, 18, step)],
        axis=1,
    )
    for b, seq in enumerate(seqs):
        close(
            output[b, [0, 5][b] :],
            model(mx.array([seq]), cache=quantized_caches(model))[0],
            0.001,
        )
    for token in [20, 21]:
        output = model(mx.array([[token], [token]]), cache=caches)
        for b, seq in enumerate(seqs):
            seq.append(token)
            close(
                output[b, -1],
                model(mx.array([seq]), cache=quantized_caches(model))[0, -1],
                0.001,
            )


@pytest.mark.parametrize("length", [4, 23])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("disk", [False, True])
def test_int8_prefix_reuse_identity(length, dtype, disk, tmp_path):
    from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache

    from rapid_mlx.memory_cache import (
        MemoryAwarePrefixCache,
        MemoryCacheConfig,
        _quantize_cache,
    )

    model = tiny_model(head_dim=64, mlp_only_layers=list(range(4)))
    model.set_dtype(dtype)
    ids = list(range(1, length + 1))
    live = quantized_caches(model)
    for token in ids:
        mx.eval(model(mx.array([[token]]), cache=live))
    extracted = [c.extract(0) for c in live]
    packed = _quantize_cache(extracted, bits=8, group_size=64)
    assert isinstance(packed[0], QuantizedKVCache)
    assert isinstance(packed[1], RotatingKVCache)
    prefix = MemoryAwarePrefixCache(
        None,
        MemoryCacheConfig(
            kv_quantize=True,
            kv_bits=8,
            kv_group_size=64,
            kv_min_quantize_tokens=0,
            hybrid_reuse_max_entries=4,
        ),
    )
    assert prefix.store(ids, extracted, evict_prefixes=False)
    if disk:
        assert prefix.save_to_disk(str(tmp_path))
        restored = MemoryAwarePrefixCache(None, prefix._config)
        assert restored.load_from_disk(str(tmp_path)) == 1
        prefix = restored
    warm, rest = prefix.fetch(ids + [30, 31])
    assert warm is not None and rest == [30, 31]
    generate = importlib.import_module("mlx_lm.generate")
    warm = generate._merge_caches([normalize_caches_for_quantization(warm, 64, 8)])
    cold = quantized_caches(model)
    for token in ids:
        mx.eval(model(mx.array([[token]]), cache=cold))
    close(model(mx.array([rest]), cache=warm), model(mx.array([rest]), cache=cold), 0)


def test_official_layer_layout_and_undeclared_sink_refusal():
    from rapid_mlx.kv_cache_dtype import KVCacheQuantizationUnsupportedError
    from rapid_mlx.scheduler import Scheduler

    kinds = ["full_attention"] * 14 + ["sliding_attention"] * 35
    model = tiny_model(
        head_dim=64,
        num_hidden_layers=49,
        layer_types=kinds,
        mlp_only_layers=list(range(49)),
        sliding_window=513,
    )
    reason, layout = Scheduler._quantized_live_cache_probe(model)
    assert reason is None
    assert (layout.quantizable_layers, layout.rotating_layers, layout.total_layers) == (
        14,
        35,
        49,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = types.SimpleNamespace(
        kv_cache_quantization=True,
        kv_cache_quantization_bits=8,
        kv_cache_dtype="int8",
        kv_cache_dtype_explicit=True,
        kv_cache_turboquant=None,
        kv_cache_quantization_group_size=64,
    )
    # Identical sink attributes alone must never confer admission.
    for attr in ("sinks", "attn_sink", "softmax_offset"):
        attention = types.SimpleNamespace(**{attr: mx.zeros((4,))})
        unsupported = types.SimpleNamespace(
            layers=[types.SimpleNamespace(self_attn=attention)],
            make_cache=lambda: model.make_cache(),
        )
        with pytest.raises(
            KVCacheQuantizationUnsupportedError, match="attention sinks"
        ):
            scheduler._init_kv_quantization(unsupported)


def test_packed_prefix_merge_padding_empty_and_metadata():
    from mlx_lm.models.cache import KVCache

    from rapid_mlx.quantized_sink_cache import (
        SinkQuantizableKVCache,
        SinkQuantizedBatchKVCache,
        SinkQuantizedKVCache,
    )

    empty = SinkQuantizedBatchKVCache([0])
    assert empty.extract(0).size() == 0
    assert SinkQuantizedKVCache.merge([empty.extract(0)]).empty()
    dense = KVCache()
    k = mx.random.normal((1, 2, 4, 64))
    dense.update_and_fetch(k, k)
    populated = SinkQuantizedBatchKVCache.merge([dense])
    shorter = populated.extract(0)
    shorter.trim(2)
    merged = SinkQuantizedKVCache.merge(
        [shorter, populated.extract(0), empty.extract(0)]
    )
    assert merged.left_padding.tolist() == [2, 0, 4]
    for i, expected in enumerate([shorter, populated.extract(0)]):
        actual = merged.extract(i)
        for a, b in zip(actual.state[0], expected.state[0]):
            assert mx.array_equal(a, b).item()
    # A prefix hit may start a batch whose first member is packed or plain.
    plain = SinkQuantizableKVCache()
    plain.update_and_fetch(k, k)
    mixed = SinkQuantizableKVCache.merge([plain, shorter])
    assert isinstance(mixed, SinkQuantizedBatchKVCache)
    assert mixed._idx == 4
    wrong = SinkQuantizedKVCache(group_size=32)
    wrong.update_and_fetch(k, k)
    with pytest.raises(ValueError, match="different quantization"):
        SinkQuantizedBatchKVCache.merge([wrong], group_size=64)


@pytest.mark.parametrize("empty", [False, True])
def test_scheduler_midprefill_preserves_packed_cache(empty):
    from rapid_mlx.quantized_sink_cache import (
        SinkQuantizedBatchKVCache,
        SinkQuantizedKVCache,
    )
    from rapid_mlx.scheduler import Scheduler

    batch = SinkQuantizedBatchKVCache([0])
    if not empty:
        x = mx.random.normal((1, 2, 3, 64))
        batch.update_and_fetch(x, x)
    scheduler = Scheduler.__new__(Scheduler)
    restored = scheduler._reconstruct_cache_from_states(
        [
            {
                "class_ref": type(batch),
                "state": batch.state,
                "meta_state": batch.meta_state,
            }
        ]
    )
    assert isinstance(restored[0], SinkQuantizedKVCache)
    assert restored[0].size() == (0 if empty else 3)
    if not empty:
        for a, b in zip(restored[0].state[0], batch.state[0]):
            assert mx.array_equal(a, b).item()


def test_packed_cache_trim_and_disk_mode_gate():
    from rapid_mlx.memory_cache import (
        MemoryCacheConfig,
        _cache_classes_compatible,
        _trim_cache_offset,
    )
    from rapid_mlx.quantized_sink_cache import (
        SinkQuantizedBatchKVCache,
        SinkQuantizedKVCache,
    )

    batch = SinkQuantizedBatchKVCache([0])
    x = mx.random.normal((1, 2, 4, 64))
    batch.update_and_fetch(x, x)
    trimmed = _trim_cache_offset([batch.extract(0)], 2)[0]
    assert isinstance(trimmed, SinkQuantizedKVCache)
    assert trimmed.size() == 2
    ok, reason = _cache_classes_compatible(
        ["SinkQuantizedKVCache"], MemoryCacheConfig()
    )
    assert not ok and "requires" in reason


def test_all_masked_rows_have_zero_sink_attention():
    q = mx.ones((1, 4, 2, 64), dtype=mx.float16)
    k = mx.ones((1, 2, 3, 64), dtype=mx.float16)
    packed = mx.quantize(k, bits=8, group_size=64)
    output = quantized_sink_attention(
        q,
        packed,
        packed,
        scale=0.125,
        mask=mx.zeros((2, 3), dtype=mx.bool_),
        sinks=mx.array([-100, 0, 100, 1], dtype=mx.float16),
    )
    assert mx.array_equal(output, mx.zeros_like(output)).item()


@pytest.mark.parametrize("resident", [False, True])
def test_scheduler_installs_model_qualified_live_cache(monkeypatch, resident):
    from rapid_mlx import scheduler as module
    from rapid_mlx.quantized_sink_cache import (
        SinkQuantizableKVCache,
        SinkQuantizedBatchKVCache,
    )
    from rapid_mlx.request import SamplingParams

    scheduler = module.Scheduler.__new__(module.Scheduler)
    scheduler.model = tiny_model(head_dim=64)
    scheduler.tokenizer = object()
    scheduler._get_stop_tokens = lambda: set()
    scheduler.memory_aware_cache = None
    scheduler.model_config = None
    scheduler.config = types.SimpleNamespace(
        prefill_batch_size=1,
        completion_batch_size=1,
        prefill_step_size=3,
        spec_decode="none",
        enable_suffix_decoding=False,
        kv_cache_quantization=True,
        kv_cache_turboquant=None,
        kv_cache_quantization_bits=8,
        kv_cache_quantization_group_size=64,
        kv_cache_dtype="int8",
        kv_cache_dtype_explicit=True,
    )
    if resident:
        from rapid_mlx.runtime.resident_models import (
            ResidentPerformanceConfig,
            resident_scheduler_kwargs,
        )

        vars(scheduler.config).pop("kv_cache_dtype_explicit")
        vars(scheduler.config).update(
            resident_scheduler_kwargs(ResidentPerformanceConfig(kv_cache_dtype="int8"))
        )
    scheduler._init_kv_quantization(scheduler.model)
    generator = types.SimpleNamespace(_make_new_cache=scheduler.model.make_cache)
    monkeypatch.setattr(module, "BatchGenerator", lambda **_: generator)
    monkeypatch.setattr(module, "make_sampler", lambda **_: object())
    actual = scheduler._create_batch_generator(SamplingParams(max_tokens=4))
    fresh = actual._make_new_cache()
    assert isinstance(fresh[0], SinkQuantizableKVCache)
    live = importlib.import_module("mlx_lm.generate")._merge_caches([fresh])
    assert isinstance(live[0], SinkQuantizedBatchKVCache)
    assert scheduler._live_kv_quant == (64, 8)


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("saved_bits,live_group_size", [(8, 32), (8, 64), (4, 32)])
def test_packed_prefix_geometry_gate(disk, saved_bits, live_group_size, tmp_path):
    from dataclasses import replace

    from rapid_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
    from rapid_mlx.quantized_sink_cache import SinkQuantizedBatchKVCache

    model = tiny_model(
        head_dim=64, layer_types=["full_attention"] * 4, mlp_only_layers=list(range(4))
    )

    def fresh(group_size, bits):
        return [SinkQuantizedBatchKVCache([0], group_size, bits) for _ in range(4)]

    ids = [1, 2, 3, 4]
    stored = fresh(32, saved_bits)
    for token in ids:
        mx.eval(model(mx.array([[token]]), cache=stored))
    prefix = MemoryAwarePrefixCache(
        None,
        MemoryCacheConfig(kv_quantize=True, kv_bits=saved_bits, kv_group_size=32),
    )
    assert prefix.store(ids, [c.extract(0) for c in stored])
    live_config = replace(prefix._config, kv_bits=8, kv_group_size=live_group_size)
    compatible = saved_bits == 8 and live_group_size == 32
    if disk:
        assert prefix.save_to_disk(str(tmp_path))
        prefix = MemoryAwarePrefixCache(None, live_config)
        assert prefix.load_from_disk(str(tmp_path)) == int(compatible)
    else:
        prefix._config = live_config
    request = ids + [5, 6]
    cached, remaining = prefix.fetch(request)
    cold = fresh(live_group_size, 8)
    if compatible:
        assert cached is not None and remaining == [5, 6]
        for original, restored in zip(stored, cached):
            for a, b in zip(original.extract(0).state[0], restored.state[0]):
                assert mx.array_equal(a, b).item()
        live = importlib.import_module("mlx_lm.generate")._merge_caches([cached])
        for token in ids:
            mx.eval(model(mx.array([[token]]), cache=cold))
    else:
        assert cached is None and remaining == request
        assert prefix._last_match_type == "miss"
        assert prefix._stats.hits == 0 and prefix._stats.tokens_saved == 0
        live = fresh(live_group_size, 8)
    close(
        model(mx.array([remaining]), cache=live),
        model(mx.array([remaining]), cache=cold),
        0,
    )


def test_packed_prefix_effective_group_size():
    from rapid_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
    from rapid_mlx.quantized_sink_cache import SinkQuantizedBatchKVCache

    # Requested gs64 resolves to gs32 for head_dim=96 on both live and stored KV.
    batch = SinkQuantizedBatchKVCache([0], group_size=32)
    k = mx.ones((1, 2, 4, 96))
    batch.update_and_fetch(k, k)
    prefix = MemoryAwarePrefixCache(None, MemoryCacheConfig(kv_quantize=True))
    assert prefix.store([1, 2, 3, 4], [batch.extract(0)])
    cached, remaining = prefix.fetch([1, 2, 3, 4, 5])
    assert cached is not None and remaining == [5]
    assert cached[0].group_size == 32


@pytest.mark.parametrize("explicit", [False, True])
def test_residency_flint_int8_requires_and_carries_provenance(explicit, monkeypatch):
    from rapid_mlx.kv_cache_dtype import KVCacheQuantizationUnsupportedError
    from rapid_mlx.runtime.resident_models import (
        ResidentPerformanceConfig,
        resident_scheduler_kwargs,
        resolve_resident_performance,
    )
    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    monkeypatch.setattr(
        "rapid_mlx.cli._gather_kv_cache_dtype_inputs",
        lambda _: ({"model_type": "flint"}, None),
    )
    performance = resolve_resident_performance(
        ResidentPerformanceConfig(kv_cache_dtype="int8"),
        model_name="flint",
        model_path=None,
    )
    kwargs = resident_scheduler_kwargs(performance)
    assert kwargs["kv_cache_dtype_explicit"] is True
    if not explicit:
        kwargs.pop("kv_cache_dtype_explicit")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(**kwargs)
    if explicit:
        scheduler._init_kv_quantization(tiny_model(head_dim=64))
        assert not scheduler._kv_quant_live_disabled
        assert scheduler._kv_quant_layout.quantizable_layers == 2
    else:
        with pytest.raises(
            KVCacheQuantizationUnsupportedError, match="require explicit"
        ):
            scheduler._init_kv_quantization(tiny_model(head_dim=64))


def test_reasoning_unreadable_preflight_flint_int4_exits_two(
    tmp_path, monkeypatch, capsys
):
    from rapid_mlx import cli, server
    from rapid_mlx.middleware import request_logging
    from rapid_mlx.scheduler import Scheduler

    # An opaque local path with no readable config must defer to the loaded model.
    model_dir = tmp_path / "opaque"
    model_dir.mkdir()
    args = cli.build_parser().parse_args(
        [
            "serve",
            str(model_dir),
            "--no-mllm",
            "--reasoning",
            "--kv-cache-dtype",
            "int4",
        ]
    )
    monkeypatch.setattr(cli, "_gather_kv_cache_dtype_inputs", lambda _: (None, None))
    for name in (
        "_port_preflight_or_die",
        "_check_alias_min_memory",
        "_check_disk_space",
        "_check_memory_capacity",
    ):
        monkeypatch.setattr(cli, name, lambda *a, **kw: None)
    monkeypatch.setattr(server, "configure_model_residency", lambda *a, **kw: None)
    monkeypatch.setattr(server, "configure_cors_from_env", lambda *_: [])
    monkeypatch.setattr(server, "configure_trusted_hosts", lambda *_: None)
    monkeypatch.setattr(
        request_logging, "install_request_logging_middleware", lambda *_: None
    )
    loaded = []

    def load_model(*_, scheduler_config, **kwargs):
        loaded.append(scheduler_config)
        assert scheduler_config.kv_cache_dtype == "int8"
        assert scheduler_config.kv_cache_dtype_requested == "int4"
        assert scheduler_config.kv_cache_dtype_explicit
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.config = scheduler_config
        scheduler._init_kv_quantization(tiny_model(head_dim=64))

    monkeypatch.setattr(server, "load_model", load_model)
    with pytest.raises(SystemExit) as exc:
        cli.serve_command(args)
    assert exc.value.code == 2
    assert len(loaded) == 1
    output = capsys.readouterr().out
    assert "--kv-cache-dtype int4 cannot be honored" in output
    assert "attention sinks support only int8" in output


def test_reasoning_original_int4_still_allows_other_models():
    from mlx_lm.models.cache import KVCache

    from rapid_mlx.kv_cache_dtype import resolve_kv_cache_dtype
    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    decision = resolve_kv_cache_dtype(
        "int4", reasoning=True, explicit=True, hf_config=None
    )
    assert decision.dtype == "int8"
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(
        kv_cache_dtype=decision.dtype,
        kv_cache_dtype_requested=decision.requested,
        kv_cache_dtype_explicit=True,
        kv_cache_quantization=True,
        kv_cache_quantization_bits=8,
    )
    model = types.SimpleNamespace(
        args=types.SimpleNamespace(head_dim=64),
        layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace())],
        make_cache=lambda: [KVCache()],
    )
    scheduler._init_kv_quantization(model)
    assert not scheduler._kv_quant_live_disabled
    assert scheduler.config.kv_cache_quantization_bits == 8
