# SPDX-License-Identifier: Apache-2.0
"""Contracts for request-private whole-step compiled decode replay."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import ArraysCache, KVCache

import vllm_mlx.compiled_decode as compiled_decode
import vllm_mlx.compiled_precision as compiled_precision
from vllm_mlx.compiled_decode import (
    CompiledDecodeStep,
    ShapeStableKVCache,
    convert_cache,
)
from vllm_mlx.compiled_precision import (
    compiled_decode_precision,
    gated_product,
)
from vllm_mlx.singleton_cache_fastpath import _promote_layer


class _ToyModel:
    def __call__(self, tokens, *, cache):
        batch, length = tokens.shape
        value = tokens.astype(mx.float32).reshape(batch, 1, length, 1)
        cache[0].update_and_fetch(value, value)
        values = tokens.astype(mx.float32)
        return mx.stack([values, -values], axis=-1)

    def make_cache(self):
        return [KVCache()]


def _filled_cache(token: int = 3) -> KVCache:
    cache = KVCache()
    value = mx.array([[[[token]]]], dtype=mx.float32)
    cache.update_and_fetch(value, value)
    mx.eval(cache.state)
    return cache


def test_compiled_step_replays_once_and_threads_cache_state() -> None:
    model = _ToyModel()
    stock = _filled_cache()
    stable = convert_cache([_filled_cache()])
    step = CompiledDecodeStep(model, stable)

    for token in (5, 7, 11):
        inputs = mx.array([[token]], dtype=mx.int32)
        expected = model(inputs, cache=[stock])
        actual = step(inputs)
        step.confirm_oldest(actual)
        assert bool(mx.array_equal(expected, actual).item())

    step.drain_pending()
    assert stable[0].size() == stock.size() == 4
    assert step.receipt() == {
        "traces": 1,
        "variants": 1,
        "submissions": 3,
        "completions": 3,
        "pending": 0,
        "poisoned": False,
        "poison_reason": None,
    }


def test_shape_stable_promotion_drains_and_returns_batched_cache() -> None:
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())

    class _Owner:
        calls = []

        def drain_pending(self, *, phase):
            self.calls.append(phase)

    owner = _Owner()
    stable._rapid_compiled_owner = owner
    promoted = _promote_layer(stable)

    assert owner.calls == ["cache conversion"]
    assert type(promoted).__name__ == "BatchKVCache"
    assert promoted.keys.shape[0] == 1
    assert promoted.offset == 1


def test_convert_cache_is_transactional_on_unsupported_layer() -> None:
    original = [_filled_cache(), object()]
    with pytest.raises(TypeError, match=r"cache\[1\]"):
        convert_cache(original)
    assert type(original[0]) is KVCache
    assert type(original[1]) is object


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_compiled_gate_product_preserves_eager_bytes(dtype) -> None:
    gate = mx.array([-6.84375, -1.0, 0.0, 1.0], dtype=dtype)
    value = mx.array([0.25, -2.0, 3.0, 4.0], dtype=dtype)
    expected = value * mx.sigmoid(gate)
    with compiled_decode_precision():
        actual = gated_product(gate, value)
    assert bool(mx.array_equal(expected, actual).item())


def test_batch_generator_installer_replays_and_cleans_up(monkeypatch) -> None:
    monkeypatch.setattr(compiled_decode, "model_qualification_reason", lambda *_: None)
    monkeypatch.setattr(
        compiled_decode, "install_qwen35_attention_gate_precision", lambda _: 1
    )
    from vllm_mlx.singleton_cache_fastpath import install_singleton_cache_fastpath

    install_singleton_cache_fastpath()
    generator = BatchGenerator(
        _ToyModel(), max_tokens=4, stream=mx.new_stream(mx.default_device())
    )
    assert compiled_decode.install_compiled_decode(
        generator, generator.model, model_name="qualified"
    )
    uid = generator.insert([[3, 2, 1]], max_tokens=[4])[0]
    responses = []
    for _ in range(8):
        _, generated = generator.next()
        responses.extend(generated)
        if any(
            response.uid == uid and response.finish_reason for response in generated
        ):
            break

    stats = generator._rapid_compiled_decode_stats
    assert [response.token for response in responses if response.uid == uid] == [0] * 4
    assert stats["attachments"] == 1
    assert stats["traces"] == 1
    assert stats["submissions"] == stats["completions"] == 4
    assert stats["poisoned"] is False


def test_batch_join_promotes_compiled_cache_then_new_singleton_reattaches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(compiled_decode, "model_qualification_reason", lambda *_: None)
    monkeypatch.setattr(
        compiled_decode, "install_qwen35_attention_gate_precision", lambda _: 1
    )
    from vllm_mlx.singleton_cache_fastpath import install_singleton_cache_fastpath

    install_singleton_cache_fastpath()
    generator = BatchGenerator(
        _ToyModel(),
        max_tokens=8,
        prefill_batch_size=2,
        completion_batch_size=2,
        stream=mx.new_stream(mx.default_device()),
    )
    assert compiled_decode.install_compiled_decode(
        generator, generator.model, model_name="qualified"
    )
    first = generator.insert([[3, 2, 1]], max_tokens=[8])[0]
    generator.next()
    generator.next()
    second = generator.insert([[4, 3, 2]], max_tokens=[4])[0]

    finished = set()
    for _ in range(16):
        _, generated = generator.next()
        finished.update(
            response.uid for response in generated if response.finish_reason is not None
        )
        if finished == {first, second}:
            break
    assert finished == {first, second}

    third = generator.insert([[5, 4, 3]], max_tokens=[3])[0]
    for _ in range(8):
        _, generated = generator.next()
        if any(
            response.uid == third and response.finish_reason for response in generated
        ):
            break
    stats = generator._rapid_compiled_decode_stats
    assert stats["attachments"] == 2, stats
    assert stats["poisoned"] is False


def test_declined_request_does_not_retry_cache_conversion_each_token(
    monkeypatch,
) -> None:
    monkeypatch.setattr(compiled_decode, "model_qualification_reason", lambda *_: None)
    monkeypatch.setattr(
        compiled_decode, "install_qwen35_attention_gate_precision", lambda _: 1
    )
    attempts = 0

    def decline(_cache):
        nonlocal attempts
        attempts += 1
        raise ValueError("deliberate boundary decline")

    monkeypatch.setattr(compiled_decode, "convert_cache", decline)
    generator = BatchGenerator(
        _ToyModel(), max_tokens=5, stream=mx.new_stream(mx.default_device())
    )
    assert compiled_decode.install_compiled_decode(
        generator, generator.model, model_name="qualified"
    )
    uid = generator.insert([[3, 2, 1]], max_tokens=[5])[0]
    for _ in range(10):
        _, generated = generator.next()
        if any(
            response.uid == uid and response.finish_reason for response in generated
        ):
            break

    assert attempts == 1
    assert generator._rapid_compiled_decode_stats["last_decline_reason"] == (
        "deliberate boundary decline"
    )


def test_shape_stable_cache_full_surface_and_bucket_transition() -> None:
    for buckets in ((), (0,), (2, 1), (1, 1)):
        with pytest.raises(ValueError, match="sorted positive"):
            ShapeStableKVCache(buckets=buckets)

    empty = ShapeStableKVCache(buckets=(2, 4))
    assert empty.empty() is True
    assert empty.nbytes == 0
    assert empty.state == ()
    assert empty.reserve(0) is False
    with pytest.raises(ValueError, match="cannot reserve"):
        empty.reserve(1)
    with pytest.raises(ValueError, match="non-negative"):
        empty.reserve(True)
    with pytest.raises(ValueError, match="before cache fill"):
        empty.make_mask(1)

    first = mx.array([[[[1.0], [2.0]]]])
    keys, values = empty.update_and_fetch(first, first)
    mx.eval(keys, values)
    assert empty.capacity == 2
    assert empty.size() == 2
    assert empty.empty() is False
    assert empty.is_trimmable() is True
    assert empty.nbytes == keys.nbytes + values.nbytes
    assert empty.reserve(1) is True
    assert empty.capacity == 4
    assert empty.reserve(1) is False
    mask = empty.make_mask(1, return_array=True, window_size=2)
    mx.eval(mask)
    assert mask.shape == (1, 1, 1, 4)
    assert mask[0, 0, 0].tolist() == [False, True, True, False]
    assert empty.trim(1) == 1
    assert empty.size() == 1

    saved = empty.state
    empty.state = ()
    assert empty.empty()
    empty.state = saved
    assert empty.capacity == 4
    assert empty.size() == 1
    with pytest.raises(RuntimeError, match="exceeds"):
        empty.reserve(4)


def test_shape_stable_cache_conversion_and_singleton_guards() -> None:
    empty = ShapeStableKVCache.from_kv_cache(KVCache())
    assert empty.to_kv_cache().keys is None

    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    phases = []
    stable._rapid_compiled_owner = SimpleNamespace(
        drain_pending=lambda *, phase: phases.append(phase)
    )
    stable.filter([0])
    extracted = stable.extract(0)
    assert phases == ["cache conversion"]
    assert extracted.offset == 1
    with pytest.raises(IndexError, match="row 0"):
        stable.extract(1)
    with pytest.raises(IndexError, match="row 0"):
        stable.filter([1])
    with pytest.raises(NotImplementedError, match="before batching"):
        stable.extend(object())
    with pytest.raises(NotImplementedError, match="KV quantization"):
        stable.to_quantized(group_size=64)
    stable.filter([])
    assert phases[-1] == "request removal"
    assert stable.empty()


def _arrays_cache() -> ArraysCache:
    cache = ArraysCache(size=2)
    cache.cache = [mx.array([[1.0]]), mx.array([[2.0]])]
    return cache


def test_hybrid_slots_and_compiled_step_fail_closed_guards() -> None:
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    hybrid = _arrays_cache()
    step = CompiledDecodeStep(_ToyModel(), [stable, hybrid])
    assert len(step.plan) == 2
    assert step.plan[1].collect()[0].item() == 1.0
    step.plan[1].install([mx.array([[3.0]]), mx.array([[4.0]])])
    step.plan[1].commit(None, 1)
    step.plan[1].rollback([mx.array([[5.0]]), mx.array([[6.0]])], None)
    assert step.plan[1].snapshot() is None
    assert hybrid.cache[0].item() == 5.0

    for value in (0, 9):
        with pytest.raises(ValueError, match="max_variants"):
            CompiledDecodeStep(_ToyModel(), [stable], max_variants=value)
    with pytest.raises(TypeError, match="not shape-stable"):
        CompiledDecodeStep(_ToyModel(), [object()])
    with pytest.raises(TypeError, match="requires a full-attention"):
        CompiledDecodeStep(_ToyModel(), [_arrays_cache()])

    malformed = _arrays_cache()
    malformed.lengths = mx.array([1])
    with pytest.raises(TypeError, match="not ready"):
        CompiledDecodeStep(_ToyModel(), [stable, malformed])

    lagging = ShapeStableKVCache.from_kv_cache(_filled_cache(2))
    lagging.trim(1)
    with pytest.raises(TypeError, match="not synchronized"):
        CompiledDecodeStep(_ToyModel(), [stable, lagging])

    with pytest.raises(ValueError, match="batch 1"):
        step(mx.array([[1, 2]]))
    stable.speculating = True
    with pytest.raises(RuntimeError, match="speculative"):
        step(mx.array([[1]]))
    del stable.speculating


def test_compiled_step_poison_is_terminal_and_detach_clears_owner(monkeypatch) -> None:
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    step = CompiledDecodeStep(_ToyModel(), [stable])

    def broken_build(_key):
        return (lambda *_args: (_ for _ in ()).throw(RuntimeError("graph failed"))), [
            (0, 3)
        ]

    monkeypatch.setattr(step, "_build", broken_build)
    with pytest.raises(
        compiled_decode.CompiledDecodePoisonedError, match="without retrying"
    ):
        step(mx.array([[1]]))
    assert step.poisoned
    assert step.receipt()["poison_reason"].startswith("trace:")
    with pytest.raises(compiled_decode.CompiledDecodePoisonedError, match="discarded"):
        step(mx.array([[1]]))
    first_reason = step.receipt()["poison_reason"]
    step.poison(ValueError("second"), phase="later")
    assert step.receipt()["poison_reason"] == first_reason
    step.detach()
    assert stable._rapid_compiled_owner is None


def test_compiled_step_receipt_helpers_and_variant_bound(monkeypatch) -> None:
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    step = CompiledDecodeStep(_ToyModel(), [stable], max_variants=1)
    step.confirm_oldest()
    first = step(mx.array([[2]], dtype=mx.int32))
    step.confirm_oldest(first)
    assert step.completion_count == 1
    # A different token dtype creates a second signature and evicts the bounded map.
    second = step(mx.array([[2]], dtype=mx.int64))
    step.drain_pending(phase="test drain")
    assert second.shape == (1, 1, 2)
    assert len(step._variants) == 1

    step._pending.append(mx.array([1]))
    real_eval = mx.eval

    def fail_eval(*_args):
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(mx, "eval", fail_eval)
    with pytest.raises(
        compiled_decode.CompiledDecodePoisonedError, match="without retrying"
    ):
        step.confirm_oldest()
    monkeypatch.setattr(mx, "eval", real_eval)


def test_convert_cache_and_feature_gate_contracts(monkeypatch) -> None:
    monkeypatch.setenv("RAPID_MLX_COMPILED_DECODE", " OFF ")
    assert compiled_decode.enabled() is False
    monkeypatch.setenv("RAPID_MLX_COMPILED_DECODE", "yes")
    assert compiled_decode.enabled() is True

    with pytest.raises(TypeError, match="filled batch-one"):
        convert_cache([KVCache()])
    batch_two = KVCache()
    value = mx.ones((2, 1, 1, 1))
    batch_two.update_and_fetch(value, value)
    with pytest.raises(TypeError, match="filled batch-one"):
        convert_cache([batch_two])
    with pytest.raises(TypeError, match="object"):
        convert_cache([object()])

    hybrid = _arrays_cache()
    converted = convert_cache([_filled_cache(), hybrid])
    assert type(converted[0]) is ShapeStableKVCache
    assert converted[1] is hybrid
    hybrid.cache[0] = None
    with pytest.raises(TypeError, match="hybrid cache"):
        convert_cache([hybrid])


def _qualified_model(*, attention: int = 10, gdn: int = 30, routers: int = 40):
    args = SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=40,
        num_experts=256,
        num_experts_per_tok=8,
        full_attention_interval=4,
    )
    modules = [
        SimpleNamespace(
            num_experts=256,
            top_k=8,
            _rapid_qwen35_fused_router=True,
        )
        for _ in range(routers)
    ]
    modules += [
        SimpleNamespace(_rapid_qwen35_fused_gdn_decode=True) for _ in range(gdn)
    ]
    modules += [
        SimpleNamespace(_rapid_compiled_gate_precision=True) for _ in range(attention)
    ]
    return SimpleNamespace(args=args, named_modules=lambda: enumerate(modules))


def test_model_qualification_reports_every_fail_closed_gate(monkeypatch) -> None:
    model = _qualified_model()
    monkeypatch.setattr(compiled_decode, "version", lambda _name: "0.31.3")
    monkeypatch.setattr(compiled_decode, "enabled", lambda: False)
    assert "disabled" in compiled_decode.model_qualification_reason(
        model, "qwen3.6-35b"
    )
    monkeypatch.setattr(compiled_decode, "enabled", lambda: True)
    assert "checkpoint" in compiled_decode.model_qualification_reason(model, "other")
    monkeypatch.setattr(compiled_decode, "version", lambda _name: "9.9")
    assert "has not been qualified" in compiled_decode.model_qualification_reason(
        model, "qwen3.6-35b"
    )
    monkeypatch.setattr(compiled_decode, "version", lambda _name: "0.31.3")
    assert "geometry" in compiled_decode.model_qualification_reason(
        object(), "qwen3.6-35b"
    )

    broken = _qualified_model()
    broken.named_modules = lambda: (_ for _ in ()).throw(RuntimeError("broken"))
    assert "named modules" in compiled_decode.model_qualification_reason(
        broken, "qwen3.6-35b"
    )
    assert "MoE routing" in compiled_decode.model_qualification_reason(
        _qualified_model(routers=39), "qwen3.6-35b"
    )
    assert "GDN" in compiled_decode.model_qualification_reason(
        _qualified_model(gdn=29), "qwen3.6-35b"
    )
    assert "attention" in compiled_decode.model_qualification_reason(
        _qualified_model(attention=9), "qwen3.6-35b"
    )
    assert compiled_decode.model_qualification_reason(model, "qwen3.6-35b") is None

    def missing(_name):
        raise compiled_decode.PackageNotFoundError

    monkeypatch.setattr(compiled_decode, "version", missing)
    assert "metadata" in compiled_decode.model_qualification_reason(
        model, "qwen3.6-35b"
    )


def test_installer_declines_without_mutating_incompatible_generators(
    monkeypatch,
) -> None:
    batch = SimpleNamespace()
    monkeypatch.setattr(
        compiled_decode, "model_qualification_reason", lambda *_: "wrong model"
    )
    assert not compiled_decode.install_compiled_decode(
        batch, object(), model_name="wrong"
    )

    monkeypatch.setattr(
        compiled_decode,
        "model_qualification_reason",
        lambda *_: "precision-preserving attention gates are not active on all layers",
    )
    monkeypatch.setattr(
        compiled_decode,
        "install_qwen35_attention_gate_precision",
        lambda _model: (_ for _ in ()).throw(RuntimeError("patch failed")),
    )
    assert not compiled_decode.install_compiled_decode(
        batch, object(), model_name="qwen3.6-35b"
    )

    monkeypatch.setattr(
        compiled_decode, "install_qwen35_attention_gate_precision", lambda _: 0
    )
    reasons = iter([None, "still unqualified"])
    monkeypatch.setattr(
        compiled_decode, "model_qualification_reason", lambda *_: next(reasons)
    )
    assert not compiled_decode.install_compiled_decode(
        batch, object(), model_name="qwen3.6-35b"
    )

    monkeypatch.setattr(compiled_decode, "model_qualification_reason", lambda *_: None)
    assert not compiled_decode.install_compiled_decode(
        batch, object(), model_name="qwen3.6-35b"
    )


def test_compiled_forward_rejects_foreign_cache() -> None:
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    step = CompiledDecodeStep(_ToyModel(), [stable])
    forward = compiled_decode._CompiledForward(step)
    with pytest.raises(RuntimeError, match="foreign"):
        forward(mx.array([[1]]), cache=[])


def test_precision_primitives_probe_fallback_and_context(monkeypatch) -> None:
    value = mx.array([1.0, -2.0], dtype=mx.float16)
    assert bool(
        mx.array_equal(
            compiled_precision.precise_sigmoid(value), mx.sigmoid(value)
        ).item()
    )
    assert bool(
        mx.array_equal(gated_product(value, value), value * mx.sigmoid(value)).item()
    )
    assert compiled_precision.in_compiled_decode() is False
    with compiled_decode_precision():
        assert compiled_precision.in_compiled_decode() is True
        assert bool(
            mx.array_equal(
                compiled_precision.gate_sigmoid(value), mx.sigmoid(value)
            ).item()
        )
    assert compiled_precision.in_compiled_decode() is False

    monkeypatch.setattr(compiled_precision, "_PROBE_RESULT", None)
    assert compiled_precision.probe_compiled_precision() is True
    assert compiled_precision.probe_compiled_precision() is True
    monkeypatch.setattr(compiled_precision.mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(compiled_precision, "_PROBE_RESULT", None)
    assert compiled_precision.probe_compiled_precision() is False


def test_attention_precision_patch_preserves_both_cache_paths(monkeypatch) -> None:
    from mlx_lm.models import qwen3_next as q

    class Attention:
        num_attention_heads = 1
        num_key_value_heads = 1
        scale = 1.0

        def q_proj(self, x):
            return mx.concatenate([x, x], axis=-1)

        def k_proj(self, x):
            return x

        def v_proj(self, x):
            return x

        def q_norm(self, x):
            return x

        def k_norm(self, x):
            return x

        def rope(self, x, offset=None):
            return x

        def o_proj(self, x):
            return x

    attention = Attention()
    model = SimpleNamespace(named_modules=lambda: [("attention", attention)])
    monkeypatch.setattr(q, "Qwen3NextAttention", Attention)
    monkeypatch.setattr(
        q, "scaled_dot_product_attention", lambda queries, *_a, **_k: queries
    )
    monkeypatch.setattr(compiled_precision, "probe_compiled_precision", lambda: True)

    assert compiled_precision.install_qwen35_attention_gate_precision(model) == 1
    x = mx.array([[[1.0]]])
    assert attention(x).shape == (1, 1, 1)

    class Cache:
        offset = 0

        def update_and_fetch(self, keys, values):
            return keys, values

    assert attention(x, cache=Cache()).shape == (1, 1, 1)
    # Re-installation tags new instances while preserving the one class patch.
    other = Attention()
    assert (
        compiled_precision.install_qwen35_attention_gate_precision(
            SimpleNamespace(named_modules=lambda: [("attention", other)])
        )
        == 1
    )
    assert other._rapid_compiled_gate_precision is True


def test_attention_precision_installer_fails_closed(monkeypatch) -> None:
    from mlx_lm.models import qwen3_next as q

    monkeypatch.setattr(compiled_precision, "probe_compiled_precision", lambda: False)
    assert compiled_precision.install_qwen35_attention_gate_precision(object()) == 0
    monkeypatch.setattr(compiled_precision, "probe_compiled_precision", lambda: True)
    monkeypatch.setattr(q, "Qwen3NextAttention", None)
    assert compiled_precision.install_qwen35_attention_gate_precision(object()) == 0

    class Attention:
        pass

    monkeypatch.setattr(q, "Qwen3NextAttention", Attention)
    model = SimpleNamespace(named_modules=lambda: [("other", object())])
    assert compiled_precision.install_qwen35_attention_gate_precision(model) == 0


def test_scheduler_wires_compiled_decode_only_for_plain_decode(monkeypatch) -> None:
    import vllm_mlx.scheduler as scheduler_module
    from vllm_mlx.request import SamplingParams

    installed = []
    generator = SimpleNamespace()
    monkeypatch.setattr(scheduler_module, "BatchGenerator", lambda **_kwargs: generator)
    monkeypatch.setattr(scheduler_module, "make_sampler", lambda **_kwargs: object())
    monkeypatch.setattr(
        scheduler_module, "_install_dense_sampler_fastpath", lambda _bg: None
    )
    monkeypatch.setattr(
        "vllm_mlx.singleton_cache_fastpath.install_singleton_cache_fastpath",
        lambda: None,
    )
    monkeypatch.setattr(
        compiled_decode,
        "install_compiled_decode",
        lambda bg, model, *, model_name: installed.append((bg, model, model_name)),
    )
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.model = object()
    scheduler.tokenizer = object()
    scheduler._model_name = "qualified-model"
    scheduler._get_stop_tokens = lambda: set()
    scheduler.memory_aware_cache = None
    scheduler.model_config = None
    scheduler.config = SimpleNamespace(
        prefill_batch_size=1,
        completion_batch_size=1,
        prefill_step_size=1,
        spec_decode="none",
        enable_suffix_decoding=False,
        kv_cache_quantization=False,
    )
    result = scheduler._create_batch_generator(SamplingParams(max_tokens=4))
    assert result is generator
    assert installed == [(generator, scheduler.model, "qualified-model")]


def test_counterfeit_shape_stable_marker_is_not_promoted() -> None:
    counterfeit = SimpleNamespace(_rapid_shape_stable_kv=True)
    assert _promote_layer(counterfeit) is counterfeit


def test_remaining_precision_fallbacks_and_probe_mismatch(monkeypatch) -> None:
    gate = mx.array([1.0], dtype=mx.float32)
    value = mx.array([1.0, 2.0], dtype=mx.float32)
    assert compiled_precision.precise_gated_product(gate, value).shape == (2,)

    monkeypatch.setattr(compiled_precision, "_PROBE_RESULT", None)
    monkeypatch.setattr(
        compiled_precision,
        "precise_sigmoid",
        lambda x: mx.zeros_like(x),
    )
    assert compiled_precision.probe_compiled_precision() is False


def _installed_fake_generation(monkeypatch, *, original_step=None):
    if original_step is None:
        original_step = lambda self: "eager"

    generation = SimpleNamespace(
        uids=[7],
        prompt_cache=[_filled_cache()],
        _next_tokens=mx.array([1]),
        _current_tokens=mx.array([1]),
        _current_logprobs=mx.array([0.0]),
        model=_ToyModel(),
    )
    generation._step = original_step.__get__(generation)

    def original_filter(self, keep):
        self.uids = [self.uids[int(index)] for index in keep]
        return None

    generation.filter = original_filter.__get__(generation)
    batch = SimpleNamespace(_generation_batch=generation)
    monkeypatch.setattr(compiled_decode, "model_qualification_reason", lambda *_: None)
    monkeypatch.setattr(
        compiled_decode, "install_qwen35_attention_gate_precision", lambda _: 1
    )
    assert compiled_decode.install_compiled_decode(
        batch, generation.model, model_name="qualified"
    )
    return batch, generation


def test_installed_generation_decline_and_detach_edges(monkeypatch) -> None:
    _batch, generation = _installed_fake_generation(monkeypatch)
    closure = inspect.getclosurevars(generation._step.__func__).nonlocals
    detach = closure["detach"]
    detach(convert=False, phase="already detached")

    generation._next_tokens = None
    assert generation._step() == "eager"
    generation._next_tokens = mx.array([1])

    near_boundary = KVCache()
    value = mx.ones((1, 1, 1000, 1))
    near_boundary.update_and_fetch(value, value)
    generation.prompt_cache = [near_boundary]
    assert generation._step() == "eager"
    assert "bucket boundary" in closure["stats"]["last_decline_reason"]


def test_installed_generation_context_fallback_restores_stock_cache(
    monkeypatch,
) -> None:
    _batch, generation = _installed_fake_generation(monkeypatch)
    closure = inspect.getclosurevars(generation._step.__func__).nonlocals
    state = closure["state"]
    full = KVCache()
    value = mx.ones((1, 1, compiled_decode._MAX_CONTEXT, 1))
    full.update_and_fetch(value, value)
    stable = ShapeStableKVCache.from_kv_cache(full)
    hybrid = _arrays_cache()
    step = CompiledDecodeStep(generation.model, [stable, hybrid])
    generation.prompt_cache = [stable, hybrid]
    state["step"] = step

    assert generation._step() == "eager"
    assert closure["stats"]["fallbacks"] == 1
    assert type(generation.prompt_cache[0]).__name__ == "KVCache"
    assert generation.prompt_cache[1] is hybrid
    assert state["step"] is None


def test_installed_generation_wraps_unknown_and_known_step_failures(
    monkeypatch,
) -> None:
    def explode(_self):
        raise RuntimeError("generation failed")

    _batch, generation = _installed_fake_generation(monkeypatch, original_step=explode)
    closure = inspect.getclosurevars(generation._step.__func__).nonlocals
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    step = CompiledDecodeStep(generation.model, [stable])
    generation.prompt_cache = [stable]
    closure["state"]["step"] = step
    with pytest.raises(
        compiled_decode.CompiledDecodePoisonedError, match="without retrying"
    ):
        generation._step()
    assert closure["stats"]["poisoned"] is True

    def poisoned(_self):
        raise compiled_decode.CompiledDecodePoisonedError("known poison")

    # Reinstall a fresh wrapper so the captured original raises the typed error.
    _batch, generation = _installed_fake_generation(monkeypatch, original_step=poisoned)
    closure = inspect.getclosurevars(generation._step.__func__).nonlocals
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    closure["state"]["step"] = CompiledDecodeStep(generation.model, [stable])
    generation.prompt_cache = [stable]
    with pytest.raises(
        compiled_decode.CompiledDecodePoisonedError, match="known poison"
    ):
        generation._step()
    assert closure["stats"]["poisoned"] is True


def test_context_limit_and_drain_failure_poison(monkeypatch) -> None:
    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    stable._host_offset = compiled_decode._MAX_CONTEXT
    step = CompiledDecodeStep(_ToyModel(), [stable])
    with pytest.raises(RuntimeError, match="context limit"):
        step(mx.array([[1]]))

    stable = ShapeStableKVCache.from_kv_cache(_filled_cache())
    step = CompiledDecodeStep(_ToyModel(), [stable])
    step._pending.append(mx.array([1]))
    monkeypatch.setattr(
        mx,
        "eval",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("drain failed")),
    )
    with pytest.raises(
        compiled_decode.CompiledDecodePoisonedError, match="without retrying"
    ):
        step.drain_pending(phase="shutdown")


def test_installed_generation_declines_context_already_at_limit(monkeypatch) -> None:
    _batch, generation = _installed_fake_generation(monkeypatch)
    full = KVCache()
    value = mx.ones((1, 1, compiled_decode._MAX_CONTEXT, 1))
    full.update_and_fetch(value, value)
    generation.prompt_cache = [full]
    assert generation._step() == "eager"
    stats = inspect.getclosurevars(generation._step.__func__).nonlocals["stats"]
    assert "outside the compiled replay limit" in stats["last_decline_reason"]


def test_qualification_supports_nested_text_config_and_rejects_wrapper(
    monkeypatch,
) -> None:
    model = _qualified_model()
    model.args = SimpleNamespace(
        text_config={
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "full_attention_interval": 4,
        }
    )
    monkeypatch.setattr(compiled_decode, "version", lambda _name: "0.31.3")
    assert compiled_decode.model_qualification_reason(model, "qwen3.6-35b") is None

    Wrapper = type("Wrapper", (), {})
    Wrapper.__module__ = "vllm_mlx.engine.batched"
    assert "not trace-safe" in compiled_decode.model_qualification_reason(
        Wrapper(), "qwen3.6-35b"
    )


def test_eager_dispatch_default_compilation_probe_is_false() -> None:
    from vllm_mlx.patches import qwen3_5_eager_dispatch

    assert qwen3_5_eager_dispatch._not_compiling() is False
