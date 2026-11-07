"""Same-generation hardware, never client/current replay hardware."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from rapid_mlx.api.models import ExecutionHostMetrics
from rapid_mlx.execution_host import (
    _process_host,
    capture_local_execution_host,
    prepare_execution_host,
)


@pytest.fixture(autouse=True)
def host_probe(monkeypatch):
    _process_host.cache_clear()
    monkeypatch.setattr("rapid_mlx.execution_host._prepared_host", None)
    monkeypatch.setattr("rapid_mlx.execution_host.platform.system", lambda: "Darwin")
    monkeypatch.setattr("rapid_mlx.execution_host.platform.machine", lambda: "arm64")
    for key in ("MLX_HOSTFILE", "MLX_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        monkeypatch.delenv(key, raising=False)
    yield
    _process_host.cache_clear()


async def test_real_local_terminal_capture_is_once_per_process(monkeypatch):
    calls = []

    def probe(command, **kwargs):
        calls.append(command[-1])
        assert kwargs["timeout"] == 2 and kwargs["check"] is True
        return SimpleNamespace(
            stdout="Apple M4 Pro" if "brand" in command[-1] else str(32 << 30)
        )

    monkeypatch.setattr("rapid_mlx.execution_host.subprocess.run", probe)
    reference = {"status": "known", "backend": "ordinary"}
    output = SimpleNamespace(finished=True, execution_reference=reference)
    assert capture_local_execution_host(output) is None
    await asyncio.gather(prepare_execution_host(), prepare_execution_host())
    assert capture_local_execution_host(output) == {
        "chip": "M4 Pro",
        "memory_bytes": 32 << 30,
    }
    assert capture_local_execution_host(output)["chip"] == "M4 Pro"
    assert calls == ["machdep.cpu.brand_string", "hw.memsize"]
    output.finished = False
    assert capture_local_execution_host(output) is None
    output.finished = True
    for invalid in (
        None,
        {},
        {"status": "unknown"},
        {"status": "mixed"},
        {"status": "known", "backend": "unknown"},
    ):
        output.execution_reference = invalid
        assert capture_local_execution_host(output) is None


@pytest.mark.parametrize(
    "signal", ["MLX_HOSTFILE", "MLX_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"]
)
def test_any_distributed_signal_suppresses_host_without_file_access(
    monkeypatch, signal
):
    monkeypatch.setenv(
        signal, "/unreadable/launcher.json" if signal == "MLX_HOSTFILE" else "1"
    )
    monkeypatch.setattr(
        "rapid_mlx.execution_host.subprocess.run",
        lambda *a, **k: pytest.fail("must not probe"),
    )
    assert _process_host() is None


@pytest.mark.parametrize("system,machine", [("Linux", "arm64"), ("Darwin", "x86_64")])
def test_unsupported_platform_has_no_host(monkeypatch, system, machine):
    monkeypatch.setattr("rapid_mlx.execution_host.platform.system", lambda: system)
    monkeypatch.setattr("rapid_mlx.execution_host.platform.machine", lambda: machine)
    assert _process_host() is None


def test_probe_failure_is_cached(monkeypatch):
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        raise OSError("unavailable")

    monkeypatch.setattr("rapid_mlx.execution_host.subprocess.run", failed)
    assert _process_host() is None
    assert _process_host() is None
    assert len(calls) == 1


@pytest.mark.parametrize(
    "value",
    [
        {"chip": "Apple M4", "memory_bytes": 1},
        {"chip": "M4 pro", "memory_bytes": 1},
        {"chip": "M0", "memory_bytes": 1},
        {"chip": "M4", "memory_bytes": True},
        {"chip": "M4", "memory_bytes": 0},
        {"chip": "M4", "memory_bytes": 1.5},
        {"chip": "M4", "memory_bytes": 1 << 45},
        {"chip": "M4", "memory_bytes": 1, "hostname": "private"},
    ],
)
def test_host_allowlist_rejects_invalid_or_private_fields(value):
    with pytest.raises(ValueError):
        ExecutionHostMetrics.model_validate(value)


def test_serializer_never_probes_and_drops_aggregate_host(monkeypatch):
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.service.helpers import (
        _aggregate_generation_attempts,
        _build_response_metrics,
    )

    monkeypatch.setattr(
        "rapid_mlx.execution_host.subprocess.run",
        lambda *a, **k: pytest.fail("serializer must not probe"),
    )
    reference = {
        "status": "unknown",
        "method": "none",
        "backend": "ordinary",
        "config_key": "none",
        "completion_tokens": 3,
    }
    output = GenerationOutput(
        "answer",
        execution_reference=reference,
        execution_host={"chip": "M4", "memory_bytes": 32 << 30},
        timing_metrics={"mean_itl_ms": 25.0},
    )
    metrics = _build_response_metrics(output)
    assert metrics.execution_host.chip == "M4"
    output.execution_host = None
    assert _build_response_metrics(output).execution_host is None
    output.execution_host = {"chip": "M4", "memory_bytes": 32 << 30}
    aggregate = _aggregate_generation_attempts(output, output)
    assert aggregate.execution_host is None and aggregate.execution_reference is None
    output.execution_host = {"chip": "invalid", "memory_bytes": 1}
    assert _build_response_metrics(output).execution_host is None


def test_new_output_field_preserves_every_old_positional_slot():
    from dataclasses import fields

    from rapid_mlx.engine.base import GenerationOutput

    names = [field.name for field in fields(GenerationOutput)]
    assert names[-3:] == [
        "prompt_parse_context",
        "execution_reference",
        "execution_host",
    ]
    values = [
        "answer",
        [],
        1,
        2,
        "stop",
        "",
        True,
        None,
        None,
        "",
        "",
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        {"proof": "old"},
    ]
    output = GenerationOutput(*values)
    assert output.execution_reference == {"proof": "old"}
    assert output.execution_host is None


@pytest.mark.parametrize("host", [None, {"chip": "M2 Pro", "memory_bytes": 32 << 30}])
async def test_cached_chat_replay_preserves_original_host_or_absence(monkeypatch, host):
    import json

    from rapid_mlx.api.models import (
        AssistantMessage,
        ChatCompletionChoice,
        ChatCompletionRequest,
        ChatCompletionResponse,
        PerRequestMetrics,
        Usage,
    )
    from rapid_mlx.routes.chat import _stream_buffered_chat_response

    monkeypatch.setattr(
        "rapid_mlx.execution_host._process_host",
        lambda: pytest.fail("replay must not capture this host"),
    )
    request = ChatCompletionRequest(
        model="alias",
        messages=[{"role": "user", "content": "q"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    reference = {
        "status": "unknown",
        "method": "none",
        "backend": "ordinary",
        "config_key": "none",
        "completion_tokens": 3,
    }
    response = ChatCompletionResponse(
        model="alias",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(content="answer"), finish_reason="stop"
            )
        ],
        usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        metrics=PerRequestMetrics(
            execution_reference=reference, execution_host=host, mean_itl_ms=25
        ),
    )
    events = [
        json.loads(event.removeprefix("data: ").strip())
        async for event in _stream_buffered_chat_response(response, request)
        if event.startswith("data: {")
    ]
    finish = next(
        event
        for event in events
        if event.get("choices") and event["choices"][0].get("finish_reason")
    )
    assert finish["metrics"].get("execution_host") == host
    assert events[-1]["choices"] == [] and events[-1]["usage"]["completion_tokens"] == 3


@pytest.mark.parametrize("host", [None, {"chip": "M2 Pro", "memory_bytes": 32 << 30}])
async def test_buffered_responses_preserve_same_host_without_recapture(
    monkeypatch, host
):
    import json

    from rapid_mlx.routes.responses import _stream_buffered_responses_response

    monkeypatch.setattr(
        "rapid_mlx.execution_host._process_host",
        lambda: pytest.fail("replay must not capture this host"),
    )
    reference = {
        "version": 1,
        "status": "unknown",
        "method": "none",
        "backend": "ordinary",
        "config_key": "none",
        "completion_tokens": 3,
    }
    metrics = {"execution_reference": reference, "mean_itl_ms": 25}
    if host is not None:
        metrics["execution_host"] = host
    response = {
        "id": "resp_host",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "alias",
        "output": [
            {
                "type": "message",
                "id": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "answer", "annotations": []}
                ],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3},
        "metrics": metrics,
    }
    events = [
        json.loads(event.splitlines()[1][6:])
        async for event in _stream_buffered_responses_response(
            json.dumps(response).encode()
        )
    ]
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["metrics"].get("execution_host") == host


@pytest.mark.parametrize(
    "brand,memory",
    [("Intel CPU", "34359738368"), ("Apple M4", "invalid"), ("Apple M4", "-1")],
)
def test_incomplete_hardware_probe_never_publishes_partial_host(
    monkeypatch, brand, memory
):
    values = iter((brand, memory))
    monkeypatch.setattr(
        "rapid_mlx.execution_host.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout=next(values)),
    )
    assert _process_host() is None


def test_constructed_invalid_host_dto_is_revalidated_and_omitted():
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.service.helpers import _build_response_metrics

    invalid = ExecutionHostMetrics.model_construct(
        chip="/private/host-marker", memory_bytes=True
    )
    output = GenerationOutput(
        "answer",
        execution_reference={
            "status": "unknown",
            "method": "none",
            "backend": "ordinary",
            "config_key": "none",
            "completion_tokens": 3,
        },
        execution_host=invalid,
        timing_metrics={"mean_itl_ms": 25.0},
    )
    metrics = _build_response_metrics(output)
    assert metrics.execution_host is None
    assert metrics.execution_reference.completion_tokens == 3
    assert metrics.mean_itl_ms == 25.0
    assert "host-marker" not in metrics.model_dump_json(exclude_none=True)


async def test_blocked_probe_keeps_loop_responsive_and_singleflight(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def probe(command, **kwargs):
        calls.append(command[-1])
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(
            stdout="Apple M4" if "brand" in command[-1] else str(32 << 30)
        )

    monkeypatch.setattr("rapid_mlx.execution_host.subprocess.run", probe)
    tasks = [asyncio.create_task(prepare_execution_host()) for _ in range(3)]
    output = SimpleNamespace(
        finished=True, execution_reference={"status": "known", "backend": "ordinary"}
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.sleep(0)
        assert capture_local_execution_host(output) is None
        assert not any(task.done() for task in tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert calls == ["machdep.cpu.brand_string", "hw.memsize"]
    assert capture_local_execution_host(output)["chip"] == "M4"


async def test_failed_warmup_is_cached_without_terminal_retry(monkeypatch):
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        raise OSError("unavailable")

    monkeypatch.setattr("rapid_mlx.execution_host.subprocess.run", failed)
    await asyncio.gather(*(prepare_execution_host() for _ in range(3)))
    output = SimpleNamespace(
        finished=True, execution_reference={"status": "known", "backend": "ordinary"}
    )
    assert capture_local_execution_host(output) is None
    await prepare_execution_host()
    assert len(calls) == 1


@pytest.mark.requires_mlx
@pytest.mark.parametrize("mllm", [False, True])
async def test_engine_start_prepares_host_only_for_supported_text_lane(monkeypatch, mllm):
    from rapid_mlx.engine import batched

    events = []

    async def prepare():
        events.append("host")
        await asyncio.sleep(0)

    async def load():
        events.append("load")
        assert not engine._loaded

    monkeypatch.setattr(batched, "prepare_execution_host", prepare)
    engine = object.__new__(batched.BatchedEngine)
    engine._loaded = False
    engine._is_mllm = mllm
    engine._model_name = "fixture"
    engine._guided_requests_lock = threading.Lock()
    engine._validate_lane_capabilities = lambda: events.append("validate")
    engine._start_llm = load
    engine._start_mllm = load
    await engine.start()
    assert events == (["validate", "load"] if mllm else ["validate", "host", "load"])
    assert engine._loaded and engine._guided_stopping is False
    await engine.start()
    assert events.count("load") == 1
