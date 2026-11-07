"""Generation-local metrics survive SDK parsing and owned transactional replay."""

import json

import httpx
import pytest
from openai.types.chat import ChatCompletion
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from rapid_mlx.chat_runtime.execution_metrics import (
    ExecutionChatModel,
    bounded_metrics,
    final_generation_metrics,
)

METRICS = {
    "execution_reference": {
        "status": "known",
        "repo_id": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "method": "none",
        "backend": "ordinary",
        "config_key": "none",
        "completion_tokens": 4,
    },
    "execution_host": {"chip": "M4", "memory_bytes": 32 << 30},
    "time_to_first_token_ms": 100.0,
    "mean_itl_ms": 25.0,
}


def test_nonstream_actual_sdk_allowlist_and_response_binding():
    dto = ChatCompletion.model_validate(
        {
            "id": "one",
            "created": 1,
            "model": "alias",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
            "metrics": {**METRICS, "private_path": "/private/model"},
        }
    )
    from pydantic_ai.providers.openai import OpenAIProvider

    model = ExecutionChatModel("alias", provider=OpenAIProvider(api_key="test-only"))
    response = model._process_response(dto)
    saved = final_generation_metrics([response])
    assert saved["execution_host"] == METRICS["execution_host"]
    assert saved["execution_reference"]["completion_tokens"] == 4
    assert "private_path" not in repr(saved)
    # Aggregate turn counters must never repair a mismatch in one response.
    response.usage.output_tokens = 20
    assert final_generation_metrics([response]) is None
    assert (
        final_generation_metrics(
            [ModelResponse(parts=[TextPart("synthetic fallback")])]
        )
        is None
    )
    assert final_generation_metrics([]) is None


async def test_real_streamed_adapter_keeps_finish_metrics_after_usage():
    from rapid_mlx.chat_runtime.model_client import ModelClients

    base = {
        "id": "one",
        "created": 1,
        "model": "alias",
        "object": "chat.completion.chunk",
    }
    chunks = [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "answer"},
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "metrics": METRICS,
        },
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
        },
    ]
    payload = (
        "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )
    clients = ModelClients(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, text=payload, headers={"content-type": "text/event-stream"}
                )
            )
        )
    )
    try:
        model = clients.model("http://localhost:18000", "alias")
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart("question")])],
            None,
            ModelRequestParameters(),
        ) as response:
            async for _ in response:
                pass
            result = response.get()
        assert result.usage.output_tokens == 4
        assert final_generation_metrics([result])["mean_itl_ms"] == 25
    finally:
        await clients.close()


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        {},
        {**METRICS, "mean_itl_ms": float("nan")},
        {**METRICS, "execution_host": {"chip": "M4", "memory_bytes": True}},
        {**METRICS, "execution_reference": {"method": "none"}},
    ],
)
def test_invalid_or_missing_metrics_remain_absent(metrics):
    assert bounded_metrics(metrics) is None


async def test_owned_atomic_finish_replay_and_access_revocation(tmp_path):
    from rapid_mlx.chat_runtime.api import create_app
    from rapid_mlx.chat_runtime.runtime import TurnRequest
    from rapid_mlx.chat_runtime.store import Store

    store = await Store.open(tmp_path)
    conversation = await store.create_conversation()
    request = TurnRequest(
        turn_id="turn-metrics",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="question",
        inference_url="http://localhost:18000",
        model="alias",
    )
    await store.begin_turn(request.model_dump(mode="json"))
    assert await store.execution_metrics(conversation["id"], request.turn_id) is None
    with pytest.raises(ValueError):
        await store.finish_turn(
            request.turn_id,
            "completed",
            parts=[{"type": "text", "text": "answer"}],
            execution_metrics={"private": "not allowed"},
        )
    assert await store.execution_metrics(conversation["id"], request.turn_id) is None
    await store.finish_turn(
        request.turn_id,
        "completed",
        parts=[{"type": "text", "text": "answer"}],
        usage={"output_tokens": 40, "requests": 2},
        execution_metrics=METRICS,
    )
    saved = await store.execution_metrics(conversation["id"], request.turn_id)
    assert saved["metrics"]["execution_reference"]["completion_tokens"] == 4
    # Idempotent finish cannot rewrite the immutable original facts.
    await store.finish_turn(
        request.turn_id, "completed", execution_metrics={**METRICS, "mean_itl_ms": 99}
    )
    assert await store.execution_metrics(conversation["id"], request.turn_id) == saved
    app = create_app(store, None, "secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://owner"
    ) as client:
        route = f"/v1/rapid/chat/conversations/{conversation['id']}/turns/{request.turn_id}/execution-metrics"
        assert (await client.get(route)).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        result = await client.get(route, headers=headers)
        assert result.status_code == 200 and result.json()["generation"] == saved
        other = await store.create_conversation()
        wrong = route.replace(conversation["id"], other["id"])
        assert (await client.get(wrong, headers=headers)).status_code == 400
        await store.delete_conversation(conversation["id"])
        assert (await client.get(route, headers=headers)).status_code == 404
    await store.close()


@pytest.mark.parametrize("folder", [False, True])
async def test_actual_runtime_adapter_persists_only_final_generation(tmp_path, folder):
    from rapid_mlx.chat_runtime.model_client import ModelClients
    from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
    from rapid_mlx.chat_runtime.store import Store
    from tests.test_chat_folder_grounding import FolderServer

    search = FolderServer()
    calls = []
    base = {
        "id": "one",
        "created": 1,
        "model": "alias",
        "object": "chat.completion.chunk",
    }
    chunks = [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Answer [1]"},
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "metrics": METRICS,
        },
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
        },
    ]
    payload = (
        "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )

    def transport(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200, text=payload, headers={"content-type": "text/event-stream"}
            )
        return search(request)

    store = await Store.open(tmp_path)
    conversation = await store.create_conversation()
    runtime = Runtime(store)
    await runtime.clients.close()
    runtime.clients = ModelClients(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport))
    )
    request = TurnRequest(
        turn_id="actual-sdk",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="能退款吗？",
        inference_url="http://localhost:18800",
        model="alias",
        inference_api_key="local-secret",
        local_search={"folder_ids": ["selected"]} if folder else None,
    )
    try:
        await runtime.submit(request)
        events = [event async for event in runtime.events(request.turn_id) if event]
        terminal = events[-1]["event"]
        assert terminal["outcome"] == "completed"
        assert "metrics" not in terminal and "execution_reference" not in terminal
        saved = await store.execution_metrics(conversation["id"], request.turn_id)
        assert saved["message_id"] == terminal["final_message_id"]
        assert saved["metrics"]["mean_itl_ms"] == 25
        assert saved["metrics"]["execution_host"] == METRICS["execution_host"]
        assert calls.count("/v1/chat/completions") == 1
    finally:
        await runtime.close()
        await store.close()


def test_shared_native_vectors_validate_same_bounded_generation():
    from pathlib import Path

    from rapid_mlx.decode_expectations import snapshot

    rows = json.loads(Path("tests/fixtures/decode_execution_golden.json").read_text())
    for row in rows:
        metrics = bounded_metrics(row["metrics"])
        assert (metrics is not None) is row["valid"], row["name"]
        if metrics is None:
            continue
        reference, host = metrics["execution_reference"], metrics.get("execution_host")
        rate = (
            1000 / metrics["mean_itl_ms"]
            if reference["completion_tokens"] >= 2 and metrics.get("mean_itl_ms")
            else None
        )
        assert rate == row["rate"]
        assert "private_path" not in metrics and "hostname" not in metrics
        matching = []
        if (
            host
            and reference["status"] == "known"
            and rate is not None
            and metrics.get("time_to_first_token_ms") is not None
        ):
            for model in snapshot()["models"].values():
                if model["repo_id"] != reference.get("repo_id"):
                    continue
                matching.extend(
                    cohort
                    for cohort in model["cohorts"]
                    if cohort["chip"] == host["chip"]
                    and cohort["config_key"] == reference.get("config_key")
                    and cohort["method"] == reference["method"]
                    and cohort["n"] >= 3
                    and cohort["ram_min_gib"]
                    <= host["memory_bytes"] / (1 << 30)
                    <= cohort["ram_max_gib"]
                )
        assert bool(matching) is row["cohort"], row["name"]


async def test_successful_legacy_turn_has_no_inferred_generation(tmp_path):
    from rapid_mlx.chat_runtime.store import Store

    store = await Store.open(tmp_path)
    try:
        conversation = await store.create_conversation()
        await store.begin_turn(
            {
                "turn_id": "legacy",
                "conversation_id": conversation["id"],
                "branch_id": conversation["branch_id"],
                "expected_revision": 0,
                "text": "question",
                "inference_url": "http://localhost:18000",
                "model": "alias",
            }
        )
        await store.finish_turn(
            "legacy", "completed", parts=[{"type": "text", "text": "legacy answer"}]
        )
        assert await store.execution_metrics(conversation["id"], "legacy") is None
    finally:
        await store.close()
