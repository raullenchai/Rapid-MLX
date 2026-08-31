# SPDX-License-Identifier: Apache-2.0
"""Cross-lane request-semantics contract for text and multimodal serving.

The two schedulers intentionally have different capabilities, but shared
request semantics must not disappear merely because a vision-capable model was
selected. These tests exercise the engine boundary used by every HTTP route and
keep intentional differences explicit instead of hiding them behind skips.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api import utils as api_utils
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import (
    _LANE_PARITY_PROCESSOR_KEYS,
    _LANE_PARITY_SAMPLING_KEYS,
    _TEXT_ONLY_SAMPLING_KEYS,
    BatchedEngine,
)
from vllm_mlx.middleware.exception_handlers import (
    install_exception_handlers,
)

# Explicit capability data: shared fields are asserted on both schedulers;
# lane-specific fields remain visible here so adding parity is an intentional
# contract change rather than a test skip.
LANE_CAPABILITIES = {
    "media": {"text": False, "multimodal": True},
    "speculative_decode": {"text": True, "multimodal": False},
    "top_k": {"text": True, "multimodal": False},
    "min_p": {"text": True, "multimodal": False},
    "seed": {"text": True, "multimodal": False},
    "prefix_cache_usage": {"text": True, "multimodal": True},
    "cached_tokens_usage": {"text": True, "multimodal": True},
}


class _RouteRecordingEngine:
    preserve_native_tool_format = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, *, is_mllm: bool) -> None:
        self.is_mllm = is_mllm
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def build_prompt(self, messages, **_kwargs):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.calls.append(("generate", kwargs))
        text = "ok"
        if kwargs.get("tools"):
            text = (
                '<tool_call>\n{"name":"emit_value",'
                '"arguments":{"value":"ok"}}</tool_call>'
            )
        return GenerationOutput(
            text=text,
            new_text=text,
            tokens=[1],
            prompt_tokens=2,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            matched_stop="END",
        )

    async def stream_chat(self, messages, **kwargs):
        self.calls.append(("stream", kwargs))
        yield GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=2,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            matched_stop="END",
        )


def _route_client(surface: str, engine: _RouteRecordingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "lane-parity-test"
    cfg.model_path = "lane-parity-test"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.tool_parser = None
    cfg.tool_call_parser = None

    if surface == "chat":
        from vllm_mlx.routes.chat import router
    elif surface == "responses":
        from vllm_mlx.routes.responses import router
    else:
        from vllm_mlx.routes.anthropic import router

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    return TestClient(app)


def _route_request(surface: str, *, stream: bool) -> tuple[str, dict[str, Any]]:
    common = {"model": "lane-parity-test", "stream": stream}
    if surface == "chat":
        return "/v1/chat/completions", {
            **common,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 23,
            "temperature": 0.2,
            "top_p": 0.8,
            "stop": ["END", "HALT"],
            "repetition_penalty": 1.4,
            "presence_penalty": 0.3,
            "frequency_penalty": -0.2,
        }
    if surface == "responses":
        return "/v1/responses", {
            **common,
            "input": "hello",
            "max_output_tokens": 23,
            "temperature": 0.2,
            "top_p": 0.8,
        }
    return "/v1/messages", {
        **common,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 23,
        "temperature": 0.2,
        "top_p": 0.8,
        "stop_sequences": ["END", "HALT"],
    }


@pytest.mark.parametrize("surface", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("is_mllm", [False, True], ids=["text", "multimodal"])
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
def test_routes_normalize_sampling_before_lane_dispatch(
    surface: str, is_mllm: bool, stream: bool
) -> None:
    """Every public chat surface sends the same supported semantics per lane."""

    engine = _RouteRecordingEngine(is_mllm=is_mllm)
    path, payload = _route_request(surface, stream=stream)
    client = _route_client(surface, engine)

    if stream:
        with client.stream("POST", path, json=payload) as response:
            assert response.status_code == 200, response.text
            list(response.iter_lines())
    else:
        response = client.post(path, json=payload)
        assert response.status_code == 200, response.text

    method, kwargs = engine.calls[-1]
    assert method == ("stream" if stream else "generate")
    assert kwargs["max_tokens"] == 23
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.8
    if surface in {"chat", "anthropic"}:
        assert kwargs["stop"] == ["END", "HALT"]
    if surface == "chat":
        assert kwargs["repetition_penalty"] == 1.4
        assert kwargs["presence_penalty"] == 0.3
        assert kwargs["frequency_penalty"] == -0.2

    reset_config()


@pytest.mark.parametrize(
    (
        "is_checkpoint_mllm",
        "cache_mode",
        "runtime_version",
        "force_mllm",
        "force_text",
        "memory_gb",
        "spec_decode",
        "expected",
    ),
    [
        (
            False,
            None,
            "0.6.17",
            False,
            False,
            64.0,
            "none",
            (False, "text_checkpoint", False),
        ),
        (
            True,
            None,
            "0.6.17",
            False,
            False,
            64.0,
            "none",
            (True, "vision_supported", False),
        ),
        (
            True,
            "arrays",
            "0.6.3",
            False,
            False,
            64.0,
            "none",
            (False, "vision_hybrid_runtime_unsupported", True),
        ),
        (
            True,
            "arrays",
            "0.6.16",
            False,
            False,
            64.0,
            "none",
            (True, "vision_hybrid_runtime_supported", False),
        ),
        (
            True,
            "arrays",
            "0.6.17",
            False,
            False,
            64.0,
            "none",
            (True, "vision_hybrid_runtime_supported", False),
        ),
        (
            True,
            "arrays",
            "0.6.17",
            True,
            False,
            8.0,
            "none",
            (False, "vision_memory_insufficient", True),
        ),
        (
            True,
            "arrays",
            "0.6.3",
            True,
            False,
            64.0,
            "none",
            (True, "vision_lane_forced", False),
        ),
        (
            True,
            None,
            "0.6.17",
            True,
            True,
            64.0,
            "none",
            (False, "text_lane_forced", False),
        ),
        (
            True,
            None,
            "0.6.17",
            True,
            False,
            64.0,
            "mtp",
            (False, "text_lane_speculative_decode", True),
        ),
    ],
    ids=[
        "text-checkpoint",
        "plain-vision",
        "hybrid-old-runtime",
        "hybrid-min-runtime",
        "hybrid-new-runtime",
        "memory-floor-before-force",
        "force-vision-before-runtime-fallback",
        "force-text-precedence",
        "spec-decode-precedence",
    ],
)
def test_lane_selection_precedence_and_runtime_matrix(
    monkeypatch,
    is_checkpoint_mllm: bool,
    cache_mode: str | None,
    runtime_version: str,
    force_mllm: bool,
    force_text: bool,
    memory_gb: float,
    spec_decode: str,
    expected: tuple[bool, str, bool],
) -> None:
    """Named #2472 guard: one decision owns lane, reason and fallback."""

    monkeypatch.setattr(api_utils, "is_mllm_model", lambda _name: is_checkpoint_mllm)
    monkeypatch.setattr(api_utils, "mllm_backbone_cache_mode", lambda _name: cache_mode)
    monkeypatch.setattr(api_utils, "physical_ram_gb", lambda: memory_gb)
    monkeypatch.setattr(api_utils, "version", lambda _name: runtime_version)
    monkeypatch.setattr(
        api_utils,
        "mllm_arch_unsupported_but_text_vendored",
        lambda _name: False,
    )

    decision = api_utils.resolve_serving_lane_decision(
        "checkpoint",
        force_mllm=force_mllm,
        force_text=force_text,
        vision_min_memory_gb=16.0,
        requested_spec_decode=spec_decode,
    )
    assert (decision.is_mllm, decision.reason, decision.auto_text_fallback) == expected


@pytest.mark.parametrize("is_mllm", [False, True], ids=["text", "multimodal"])
def test_forced_tool_thinking_schema_bundle_is_lane_independent(
    monkeypatch, is_mllm: bool
) -> None:
    """#2447 coupled request keeps grammar + reasoning on both lanes."""

    from vllm_mlx.routes import chat as chat_route

    grammar = SimpleNamespace(reasoning_gate_id=None)
    budget = object()

    async def _grammar(*_args, **_kwargs):
        return grammar

    monkeypatch.setattr(chat_route, "_offload_tool_grammar_build", _grammar)
    monkeypatch.setattr(
        chat_route,
        "_build_reasoning_budget_processor",
        lambda *_args, **_kwargs: budget,
    )
    monkeypatch.setattr(
        chat_route, "enforce_context_length_for_messages", lambda *_args, **_kwargs: 4
    )

    engine = _RouteRecordingEngine(is_mllm=is_mllm)
    client = _route_client("chat", engine)
    from vllm_mlx.config import get_config

    get_config().tool_call_parser = "hermes"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lane-parity-test",
            "messages": [{"role": "user", "content": "return a value"}],
            "max_tokens": 32,
            "enable_thinking": True,
            "reasoning_max_tokens": 8,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "emit_value",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "emit_value"},
            },
            "response_format": {"type": "json_object"},
        },
    )
    assert response.status_code == 200, response.text
    _, kwargs = engine.calls[-1]
    assert kwargs["grammar_logits_processor"] is grammar
    assert kwargs["reasoning_budget_logits_processor"] is budget
    assert "forced_assistant_prefix" not in kwargs
    assert kwargs["requires_prompt_integrity"] is True
    reset_config()


class _TextScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.config = SimpleNamespace(max_concurrent_requests=256)
        self.engine = SimpleNamespace(scheduler=self)

    async def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        kwargs["on_request_committed"]()
        return SimpleNamespace(
            output_text="text-result",
            output_token_ids=[],
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
            cached_tokens=1,
            matched_stop="END",
        )

    async def add_request(self, **kwargs):
        self.calls.append(("stream", kwargs))
        kwargs["on_request_committed"]()
        return kwargs["request_id"] or "text-request"

    async def stream_outputs(self, _request_id):
        yield SimpleNamespace(
            output_text="text-result",
            new_text="text-result",
            new_token_ids=[7],
            prompt_tokens=3,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            logprobs=[{"token": "text-result"}],
            cached_tokens=1,
            matched_stop="END",
        )

    def abort_request(self, _request_id) -> None:
        return None


class _MultimodalScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.config = SimpleNamespace(max_concurrent_requests=256)

    async def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        kwargs["on_request_committed"]()
        return SimpleNamespace(
            output_text="vision-result",
            output_token_ids=[8],
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
            cached_tokens=1,
            matched_stop="END",
        )

    async def add_request_async(self, **kwargs):
        self.calls.append(("stream", kwargs))
        kwargs["on_request_committed"]()
        return kwargs["request_id"] or "vision-request"

    async def stream_outputs(self, _request_id):
        yield SimpleNamespace(
            output_text="vision-result",
            new_text="vision-result",
            new_token_ids=[8],
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            logprobs=[{"token": "vision-result"}],
            cached_tokens=1,
            matched_stop="END",
        )


def _engine(*, is_mllm: bool) -> tuple[BatchedEngine, Any]:
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = is_mllm
    engine._model_name = "lane-parity-test"
    engine._is_muse_wire = False
    engine._scheduler_config = SimpleNamespace(max_concurrent_requests=256)
    engine._admission_lock = threading.Lock()
    engine._admission_reservations = 0
    engine._admission_tokens = set()
    engine._admission_tasks = {}
    engine._admission_owner_done_at = {}
    engine._lifecycle_aborted_tasks = set()
    if is_mllm:
        scheduler = _MultimodalScheduler()
        engine._mllm_scheduler = scheduler
        engine._engine = None
    else:
        scheduler = _TextScheduler()
        engine._engine = scheduler
        engine._mllm_scheduler = None
    return engine, scheduler


def _request_semantics() -> tuple[dict[str, Any], tuple[object, object, object]]:
    processors = (object(), object(), object())
    kwargs = {
        "repetition_penalty": 1.4,
        "presence_penalty": 0.3,
        "frequency_penalty": -0.2,
        "grammar_logits_processor": processors[0],
        "reasoning_budget_logits_processor": processors[1],
        "suppressed_tokens_logits_processor": processors[2],
        "top_k": 17,
        "min_p": 0.08,
        "seed": 42,
    }
    return kwargs, processors


def _assert_shared_semantics(
    *, lane: str, captured: dict[str, Any], processors: tuple[object, ...]
) -> None:
    if lane == "multimodal":
        assert captured["logits_processors"] == list(processors)
        assert {key: captured[key] for key in _LANE_PARITY_SAMPLING_KEYS} == {
            "repetition_penalty": 1.4,
            "presence_penalty": 0.3,
            "frequency_penalty": -0.2,
        }
        for key in _TEXT_ONLY_SAMPLING_KEYS:
            assert key not in captured
        return

    params = captured["sampling_params"]
    assert {key: getattr(params, key) for key in _LANE_PARITY_SAMPLING_KEYS} == {
        "repetition_penalty": 1.4,
        "presence_penalty": 0.3,
        "frequency_penalty": -0.2,
    }
    assert {key: getattr(params, key) for key in _TEXT_ONLY_SAMPLING_KEYS} == {
        "top_k": 17,
        "min_p": 0.08,
        "seed": 42,
    }
    for key, processor in zip(_LANE_PARITY_PROCESSOR_KEYS, processors, strict=True):
        assert captured[key] is processor


@pytest.mark.asyncio
@pytest.mark.parametrize("is_mllm", [False, True], ids=["text", "multimodal"])
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
async def test_engine_dispatch_preserves_shared_request_semantics(
    monkeypatch, is_mllm: bool, stream: bool
) -> None:
    """Named #2447 guard: processors, stops, lifecycle and outputs survive.

    This is deliberately one coupled request rather than a Cartesian product:
    dropping any processor or shared sampling field on either generate path
    fails one diagnostic row.
    """

    # ``check_admission`` imports only the scheduler's error type, but the
    # production scheduler module imports MLX at module load. Keep this
    # dispatch contract runnable on Linux CI, where MLX is intentionally not
    # installed, without bypassing the real admission/reservation lifecycle.
    monkeypatch.setitem(
        sys.modules,
        "vllm_mlx.scheduler",
        SimpleNamespace(BackpressureError=RuntimeError),
    )

    engine, scheduler = _engine(is_mllm=is_mllm)
    kwargs, processors = _request_semantics()
    common = {
        "prompt": "prompt",
        "max_tokens": 23,
        "temperature": 0.2,
        "top_p": 0.8,
        "stop": ["END", "HALT"],
        "prefix_boundary": 7,
        **kwargs,
    }
    if not stream:
        common["_assistant_text_prefix"] = "<tool-prefix>"

    if stream:
        outputs = [
            output
            async for output in engine.stream_generate(
                request_id="public-request", **common
            )
        ]
    else:
        outputs = [await engine.generate(**common)]

    method, captured = scheduler.calls[-1]
    assert method == ("stream" if stream else "generate")
    if is_mllm:
        assert captured["max_tokens"] == 23
        assert captured["temperature"] == 0.2
        assert captured["top_p"] == 0.8
        assert captured["stop"] == ["END", "HALT"]
    else:
        sampling_params = captured["sampling_params"]
        assert sampling_params.max_tokens == 23
        assert sampling_params.temperature == 0.2
        assert sampling_params.top_p == 0.8
        assert sampling_params.stop == ["END", "HALT"]
    assert captured["lifecycle_admission_token"]
    _assert_shared_semantics(
        lane="multimodal" if is_mllm else "text",
        captured=captured,
        processors=processors,
    )
    assert captured["prefix_boundary"] == 7

    result = outputs[-1]
    assert result.finish_reason == "stop"
    assert result.matched_stop == "END"
    if not stream:
        assert result.raw_text.startswith("<tool-prefix>")
    if stream:
        assert result.logprobs == [
            {"token": "vision-result" if is_mllm else "text-result"}
        ]
    assert engine._admission_reservations == 0


def test_intentional_lane_differences_are_explicit_and_complete() -> None:
    assert set(LANE_CAPABILITIES) == {
        "media",
        "speculative_decode",
        "top_k",
        "min_p",
        "seed",
        "prefix_cache_usage",
        "cached_tokens_usage",
    }
    assert all(set(row) == {"text", "multimodal"} for row in LANE_CAPABILITIES.values())
    for name in {"media", "speculative_decode", "top_k", "min_p", "seed"}:
        assert LANE_CAPABILITIES[name]["text"] != LANE_CAPABILITIES[name]["multimodal"]
    for name in {"prefix_cache_usage", "cached_tokens_usage"}:
        assert LANE_CAPABILITIES[name] == {"text": True, "multimodal": True}
