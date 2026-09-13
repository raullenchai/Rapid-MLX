# SPDX-License-Identifier: Apache-2.0
"""Contracts for Qwen3.6 shared-weight native-cache text serving."""

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from mlx_lm.models.cache import ArraysCache, KVCache

from scripts.benchmark_qwen36_native_text_cache import _behavioral_pass
from vllm_mlx.engine.batched import (
    BatchedEngine,
    Qwen36NativeCacheTextWrapper,
    _should_start_qwen36_native_text_cache,
    _supports_qwen36_native_text_cache,
)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        (
            "coding",
            "```python\ndef two_sum(nums, target): return []\n"
            "assert two_sum([], 0) == []\n"
            "assert two_sum([1], 1) == []\n"
            "assert two_sum([1, 2], 3) == [0, 1]\n```",
        ),
        ("creative", "A lighthouse song rose through the frozen ice."),
        ("reasoning", "The final price is unchanged, so there is 0% net change."),
        ("json", '{"prime": false, "explanation": "13 times 17"}'),
        (
            "tool",
            '```json\n{"name":"weather","arguments":{"location":"Kyoto"}}\n```',
        ),
    ],
)
def test_benchmark_behavioral_contract_accepts_valid_outputs(name, text):
    assert _behavioral_pass(name, text) is True


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("coding", "def two_sum(nums, target): return []"),
        ("creative", "A lighthouse stood in rain."),
        ("reasoning", "The price changes."),
        ("json", '{"prime": true, "explanation": "wrong"}'),
        ("tool", '{"name":"weather","arguments":{"location":"Osaka"}}'),
    ],
)
def test_benchmark_behavioral_contract_rejects_missing_requirements(name, text):
    assert _behavioral_pass(name, text) is False


class _LanguageModel:
    def __init__(self):
        self.layers = [
            SimpleNamespace(is_linear=True),
            SimpleNamespace(is_linear=False),
        ]

    def __call__(self, value):
        return SimpleNamespace(logits=value)


def test_wrapper_changes_only_cache_construction():
    model = _LanguageModel()
    wrapper = Qwen36NativeCacheTextWrapper(model)

    caches = wrapper.make_cache()

    assert isinstance(caches[0], ArraysCache)
    assert isinstance(caches[1], KVCache)
    assert wrapper.layers is model.layers
    assert wrapper("same-array") == "same-array"
    assert not hasattr(model, "_position_ids")
    assert not hasattr(model, "_rope_deltas")


def test_wrapper_isolates_lane_local_mrope_state():
    class _StatefulLanguageModel(_LanguageModel):
        _position_ids = "mllm-position"
        _rope_deltas = "mllm-delta"

        def __call__(self, value):
            assert self._position_ids is None
            assert self._rope_deltas is None
            self._position_ids = "native-position"
            self._rope_deltas = "native-delta"
            return SimpleNamespace(logits=value)

    model = _StatefulLanguageModel()
    wrapper = Qwen36NativeCacheTextWrapper(model)

    assert wrapper("same-array") == "same-array"
    assert wrapper._native_position_ids == "native-position"
    assert wrapper._native_rope_deltas == "native-delta"
    assert model._position_ids == "mllm-position"
    assert model._rope_deltas == "mllm-delta"


def test_wrapper_restores_mllm_mrope_state_when_forward_fails():
    class _FailingLanguageModel(_LanguageModel):
        _position_ids = "mllm-position"
        _rope_deltas = "mllm-delta"

        def __call__(self, _value):
            self._position_ids = "native-position-before-error"
            self._rope_deltas = "native-delta-before-error"
            raise RuntimeError("forward failed")

    model = _FailingLanguageModel()
    wrapper = Qwen36NativeCacheTextWrapper(model)

    with pytest.raises(RuntimeError, match="forward failed"):
        wrapper("unused")

    assert wrapper._native_position_ids == "native-position-before-error"
    assert wrapper._native_rope_deltas == "native-delta-before-error"
    assert model._position_ids == "mllm-position"
    assert model._rope_deltas == "mllm-delta"


def test_eligibility_is_pinned_to_qualified_qwen36_geometry():
    args = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_hidden_layers=40,
        num_experts=256,
        num_experts_per_tok=8,
        full_attention_interval=4,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layers = [SimpleNamespace(is_linear=(index + 1) % 4 != 0) for index in range(40)]
    model = SimpleNamespace(args=args, layers=layers)

    assert _supports_qwen36_native_text_cache(model) is True

    args.num_experts = 128
    assert _supports_qwen36_native_text_cache(model) is False
    args.num_experts = 256
    layers[-1].is_linear = True
    assert _supports_qwen36_native_text_cache(model) is False


def test_eligibility_fails_closed_on_malformed_layer_container():
    class _MalformedModel:
        args = SimpleNamespace(
            model_type="qwen3_5_moe_text",
            hidden_size=2048,
            num_hidden_layers=40,
            num_experts=256,
            num_experts_per_tok=8,
            full_attention_interval=4,
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )

        @property
        def layers(self):
            raise TypeError("malformed layers")

    assert _supports_qwen36_native_text_cache(_MalformedModel()) is False


def test_start_gate_rejects_spec_decode_and_no_hybrid_override():
    args = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_hidden_layers=40,
        num_experts=256,
        num_experts_per_tok=8,
        full_attention_interval=4,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    model = SimpleNamespace(
        args=args,
        layers=[SimpleNamespace(is_linear=(index + 1) % 4 != 0) for index in range(40)],
    )
    kwargs = {
        "config_model_type": "qwen3_5_moe",
        "arrays_cache_compat": True,
        "spec_decode": "none",
    }

    assert _should_start_qwen36_native_text_cache(model, **kwargs) is True
    assert (
        _should_start_qwen36_native_text_cache(
            model, **{**kwargs, "spec_decode": "mtp"}
        )
        is False
    )
    assert (
        _should_start_qwen36_native_text_cache(model, **kwargs, no_hybrid=True) is False
    )


def test_request_routing_keeps_media_on_mllm_and_text_on_native_engine():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._mllm_scheduler = object()
    engine._engine = object()

    assert engine._uses_mllm_request_path(None, None) is False
    assert engine._uses_mllm_request_path(["image.png"], None) is True
    assert engine._uses_mllm_request_path(None, ["video.mp4"]) is True

    engine._engine = None
    assert engine._uses_mllm_request_path(None, None) is True


@pytest.mark.asyncio
async def test_native_text_engine_reuses_model_and_executor(monkeypatch):
    import vllm_mlx.engine_core as engine_core

    starts = []

    class _Scheduler:
        def preflight_metal_admission(self):
            return None

    class _InnerEngine:
        def __init__(self):
            self.scheduler = _Scheduler()

        async def start(self, *, executor):
            starts.append(executor)

        def close(self):
            return None

    class _AsyncEngine:
        def __init__(self, model, tokenizer, config, *, executor):
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
            self.executor = executor
            self.engine = _InnerEngine()

        async def stop(self):
            return None

    monkeypatch.setattr(engine_core, "AsyncEngineCore", _AsyncEngine)

    engine = BatchedEngine.__new__(BatchedEngine)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    language_model = _LanguageModel()
    tokenizer = object()
    engine._model_load_executor = executor
    engine._model_name = "qwen-test"
    engine._profile_name = None
    engine._scheduler_config = None
    engine._stream_interval = 1
    engine._gpu_memory_utilization = None
    engine._tool_logits_processor_factory = None
    engine._force_hybrid = False
    engine._no_hybrid = False
    engine._force_spec_decode = False
    engine._no_spec_decode = False
    engine._is_mllm = True
    engine._processor = SimpleNamespace(tokenizer=tokenizer)
    engine._engine = None
    engine._mllm_native_text_engine = False

    try:
        await engine._start_qwen36_native_text_engine(language_model)
    finally:
        executor.shutdown(wait=True)

    assert engine._mllm_native_text_engine is True
    assert engine._engine.executor is executor
    assert engine._engine.model._model is language_model
    assert engine._engine.config.gpu_memory_utilization == 0.90
    assert starts == [executor]


@pytest.mark.asyncio
async def test_native_text_engine_failure_keeps_mllm_authoritative(monkeypatch):
    import vllm_mlx.engine_core as engine_core

    cleanup = []

    class _Scheduler:
        def preflight_metal_admission(self):
            raise RuntimeError("candidate rejected")

    class _InnerEngine:
        def __init__(self):
            self.scheduler = _Scheduler()

        async def start(self, *, executor):
            raise AssertionError("preflight must fail before start")

        def close(self):
            cleanup.append("close")
            raise RuntimeError("best-effort close failed")

    class _AsyncEngine:
        def __init__(self, model, tokenizer, config, *, executor):
            self.engine = _InnerEngine()

        async def stop(self):
            cleanup.append("stop")
            raise RuntimeError("best-effort stop failed")

    monkeypatch.setattr(engine_core, "AsyncEngineCore", _AsyncEngine)

    engine = BatchedEngine.__new__(BatchedEngine)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    engine._model_load_executor = executor
    engine._model_name = "qwen-test"
    engine._profile_name = None
    engine._scheduler_config = None
    engine._stream_interval = 1
    engine._gpu_memory_utilization = None
    engine._tool_logits_processor_factory = None
    engine._force_hybrid = False
    engine._no_hybrid = False
    engine._force_spec_decode = False
    engine._no_spec_decode = False
    engine._is_mllm = True
    engine._processor = SimpleNamespace(tokenizer=object())
    engine._engine = None
    engine._mllm_native_text_engine = False

    try:
        await engine._start_qwen36_native_text_engine(_LanguageModel())
    finally:
        executor.shutdown(wait=True)

    assert cleanup == ["stop", "close"]
    assert engine._engine is None
    assert engine._mllm_native_text_engine is False


@pytest.mark.asyncio
async def test_mllm_start_activates_native_text_lane_only_after_qualification(
    monkeypatch,
):
    from vllm_mlx import mllm_scheduler as mllm_scheduler_module
    from vllm_mlx.engine import batched as batched_module
    from vllm_mlx.models import mllm as mllm_module
    from vllm_mlx.utils import chat_template_registry

    args = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_hidden_layers=40,
        num_experts=256,
        num_experts_per_tok=8,
        full_attention_interval=4,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    language_model = SimpleNamespace(
        args=args,
        layers=[SimpleNamespace(is_linear=(index + 1) % 4 != 0) for index in range(40)],
    )

    class _FakeMultimodalLM:
        def __init__(self, *_args, **_kwargs):
            self.model = SimpleNamespace(language_model=language_model)
            self.processor = SimpleNamespace(tokenizer=SimpleNamespace())
            self.config = {"model_type": "qwen3_5_moe"}

        def load(self):
            return None

    class _FakeMLLMScheduler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self):
            return None

    monkeypatch.setattr(mllm_module, "MLXMultimodalLM", _FakeMultimodalLM)
    monkeypatch.setattr(mllm_scheduler_module, "MLLMScheduler", _FakeMLLMScheduler)
    monkeypatch.setattr(
        batched_module, "_probe_mllm_cache_type", lambda _model: "ArraysCache"
    )
    monkeypatch.setattr(
        chat_template_registry, "resolve_chat_template", lambda *_args: None
    )

    engine = BatchedEngine("fake/qwen36", force_mllm=True)
    activated = []

    async def _activate(model):
        activated.append(model)

    monkeypatch.setattr(engine, "_start_qwen36_native_text_engine", _activate)
    try:
        await engine._start_mllm()
    finally:
        assert engine._model_load_executor is not None
        engine._model_load_executor.shutdown(wait=True)
        engine._model_load_executor = None

    assert activated == [language_model]
    assert isinstance(engine._mllm_scheduler, _FakeMLLMScheduler)


@pytest.mark.asyncio
async def test_dual_lane_abort_checks_both_schedulers():
    class _MLLM:
        def __init__(self):
            self.calls = []

        def abort_request(self, request_id, *, error_kind=None):
            self.calls.append((request_id, error_kind))
            return False

    class _Text:
        def __init__(self):
            self.calls = []

        async def abort_request(self, request_id, *, error_kind=None):
            self.calls.append((request_id, error_kind))
            return True

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = _MLLM()
    engine._engine = _Text()
    engine._mllm_native_text_engine = True

    assert await engine.abort_request("text-1", error_kind="lifecycle") is True
    assert engine._mllm_scheduler.calls == [("text-1", "lifecycle")]
    assert engine._engine.calls == [("text-1", "lifecycle")]


def test_dual_lane_stats_sum_common_request_counters():
    class _MLLM:
        _step_count = 3

        def get_stats(self):
            return {"num_running": 1, "total_completion_tokens": 7}

    class _Text:
        def get_stats(self):
            return {
                "num_running": 2,
                "total_completion_tokens": 11,
                "steps_executed": 5,
            }

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model_name = "qwen-test"
    engine._is_mllm = True
    engine._loaded = True
    engine._stream_interval = 1
    engine._start_time = None
    engine._mllm_scheduler = _MLLM()
    engine._engine = _Text()

    stats = engine.get_stats()

    assert stats["num_running"] == 3
    assert stats["total_completion_tokens"] == 18
    assert stats["steps_executed"] == 8
    assert stats["text_scheduler"]["num_running"] == 2


def test_dual_lane_lifecycle_request_ids_are_unioned():
    class _Scheduler:
        def __init__(self, request_ids):
            self._request_ids = request_ids

        def request_ids_snapshot(self):
            return self._request_ids

    mllm = _Scheduler({"media-1", "shared-id"})
    text = _Scheduler({"text-1", "shared-id"})
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = mllm
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=text))

    assert engine._lifecycle_schedulers() == (mllm, text)
    assert engine._lifecycle_scheduler() is mllm
    assert engine._lifecycle_request_ids() == {"media-1", "text-1", "shared-id"}


@pytest.mark.asyncio
async def test_dual_lane_stop_drains_both_before_shared_executor_shutdown():
    events = []

    class _TextInner:
        def close(self):
            events.append("text-close")

    class _Text:
        engine = _TextInner()

        async def stop(self):
            events.append("text-stop")

    class _MLLM:
        async def stop(self):
            events.append("mllm-stop")

    class _Executor:
        def shutdown(self, *, wait):
            events.append(("executor-stop", wait))

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._abort_all_guided_requests = lambda: events.append("guided-stop")
    engine._engine = _Text()
    engine._mllm_scheduler = _MLLM()
    engine._is_mllm = True
    engine._model_load_executor = _Executor()
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = object()
    engine._mllm_instance = object()
    engine._loaded = True
    engine._engine_started = True
    engine._mllm_native_text_engine = True
    engine._start_time = 1.0

    await engine.stop()

    assert events == [
        "guided-stop",
        "text-stop",
        "text-close",
        "mllm-stop",
        ("executor-stop", False),
    ]
    assert engine._model_load_executor is None
    assert engine._mllm_native_text_engine is False


def test_dual_lane_does_not_enable_mllm_cache_persistence():
    class _Text:
        def save_cache_to_disk(self, *_args, **_kwargs):
            raise AssertionError("MLLM persistence contract must stay disabled")

        def load_cache_from_disk(self, *_args, **_kwargs):
            raise AssertionError("MLLM persistence contract must stay disabled")

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._engine = _Text()

    assert engine.save_cache_to_disk("unused") is False
    assert engine.load_cache_from_disk("unused") == 0

    from vllm_mlx.cache.protocol import EngineNotReadyError

    with pytest.raises(EngineNotReadyError, match="cannot export cache"):
        engine.save_cache_with_outcome("unused")
    with pytest.raises(EngineNotReadyError, match="cannot import cache"):
        engine.load_cache_with_result("unused")


def test_dual_lane_cache_stats_preserve_mllm_shape_and_expose_text_lane():
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = SimpleNamespace(
        get_cache_stats=lambda: {"hits": 2, "entries": 1}
    )
    engine._engine = SimpleNamespace(get_cache_stats=lambda: {"hits": 5, "entries": 3})

    assert engine.get_cache_stats() == {
        "hits": 2,
        "entries": 1,
        "text_scheduler": {"hits": 5, "entries": 3},
    }
