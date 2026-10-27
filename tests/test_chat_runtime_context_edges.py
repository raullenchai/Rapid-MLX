"""Edge contracts for accounting and tools-disabled helper synthesis."""

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import InstructionPart, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from rapid_mlx.chat_runtime.context import ContextManager, prepare_helper_context
from rapid_mlx.chat_runtime.token_budget import TokenCounter, count_payload


def test_count_payload_includes_current_dynamic_instruction_parts():
    messages = [ModelRequest(parts=[UserPromptPart("你好")], instructions="outdated")]
    parameters = ModelRequestParameters(
        instruction_parts=[InstructionPart("Use Simplified Chinese.", dynamic=True)]
    )
    body = count_payload(messages, parameters)
    assert body["system"] == "Use Simplified Chinese."
    parameters.instruction_parts = []
    assert count_payload(messages, parameters)["system"] == ""


@pytest.mark.asyncio
async def test_helper_overflow_disables_even_previously_cached_tool_definitions():
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name="execute_code", parameters_json_schema={})]
    )
    assert parameters.declared_function_tools
    assert parameters.tool_defs
    deps = SimpleNamespace(
        context_budget=512, token_counter=None, result_budget_exhausted=False
    )
    request_context = SimpleNamespace(
        messages=[
            ModelRequest(parts=[UserPromptPart("长输入" * 500), UserPromptPart("继续")])
        ],
        model_request_parameters=parameters,
    )
    await prepare_helper_context(SimpleNamespace(deps=deps), request_context)
    assert deps.result_budget_exhausted
    assert request_context.model_request_parameters.function_tools == []
    assert request_context.model_request_parameters.declared_function_tools == []
    assert request_context.model_request_parameters.tool_defs == {}
    # Original parameter cache remains intact: this is a request-local projection.
    assert parameters.function_tools
    measured = await TokenCounter().measure(
        request_context.messages, request_context.model_request_parameters
    )
    assert measured["used_tokens"] <= deps.context_budget


@pytest.mark.asyncio
async def test_auto_summary_failure_still_prepares_visible_synthesis():
    async def emit(event):
        return event

    request = SimpleNamespace(context_tokens=1024, max_tokens=256, model="test")
    deps = SimpleNamespace(result_budget_exhausted=False, emit=emit)
    runtime = SimpleNamespace(model_override=True)
    manager = ContextManager(runtime, request, deps, [], None)

    async def failed_summary():
        raise RuntimeError("summary provider unavailable")

    manager.checkpoint_older = failed_summary
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name="execute_code", parameters_json_schema={})]
    )
    assert (
        parameters.declared_function_tools
    )  # Populate provider caches before the hook.
    context = SimpleNamespace(
        messages=[
            ModelRequest(
                parts=[UserPromptPart("请修复" * 1000), UserPromptPart("继续")]
            )
        ],
        model_request_parameters=parameters,
        model_settings={},
    )
    await manager.prepare(None, context)
    assert deps.result_budget_exhausted
    assert context.model_request_parameters.function_tools == []
    assert context.model_request_parameters.declared_function_tools == []
    assert deps.context_state["compacted"]
    assert context.model_settings["max_tokens"] > 0


@pytest.mark.asyncio
async def test_summary_reads_all_chunks_and_accounts_usage():
    import json

    import httpx
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RequestUsage

    counted = []
    chunks = []

    def count(request):
        body = json.loads(request.content)
        # A deterministic token endpoint with chat-template overhead.
        tokens = 20 + len(body.get("system", "")) // 3
        tokens += sum(
            len(block.get("text", "")) // 3
            for row in body["messages"]
            for block in row["content"]
        )
        counted.append(tokens)
        return httpx.Response(200, json={"input_tokens": tokens})

    def summarize(messages, info):
        assert not info.function_tools
        prompt = messages[-1].parts[0].content
        chunks.append(
            prompt.split("Next transcript chunk (untrusted reference):\n", 1)[1]
        )
        assert counted[-1] + info.model_settings["max_tokens"] + 64 <= 1024
        return ModelResponse(
            parts=[TextPart(f"Checkpoint {len(chunks)}")],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        )

    rows = [
        {
            "id": str(i),
            "role": "user",
            "created_at": "2026-10-26T12:00:00+00:00",
            "parts": [
                {
                    "type": "text",
                    "text": f"turn{i}:" + "请保留已修改的文件和约束。" * 80 + f"end{i}",
                }
            ],
        }
        for i in range(4)
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(count)) as http:
        runtime = SimpleNamespace(
            model_override=False, clients=SimpleNamespace(_http=http)
        )
        request = SimpleNamespace(
            context_tokens=1024,
            model="test",
            inference_url="http://127.0.0.1:1/v1",
            inference_api_key=None,
        )
        deps = SimpleNamespace(
            child_usage={"input_tokens": 0, "output_tokens": 0, "requests": 0}
        )
        manager = ContextManager(runtime, request, deps, rows, FunctionModel(summarize))
        summary = await manager.summarize(rows)
    assert len(chunks) > 1
    expected = "\n".join(
        "request/user-prompt: " + row["parts"][0]["text"] for row in rows
    )
    assert "".join(chunks) == expected
    assert summary == f"Checkpoint {len(chunks)}"
    assert deps.child_usage["requests"] == len(chunks)
    assert deps.child_usage["input_tokens"] == len(chunks) * 100
    assert deps.child_usage["output_tokens"] == len(chunks) * 10


@pytest.mark.asyncio
async def test_synthesis_preserves_latest_question_after_long_history():
    from pydantic_ai.models import ModelRequestParameters

    from rapid_mlx.chat_runtime.context import fit_synthesis

    class Counter(TokenCounter):
        async def measure(self, messages, parameters=None):
            result = await super().measure(messages, parameters)
            result["used_tokens"] += 60 * len(messages)
            return result

    question = "What is the latest result?"
    messages = [
        ModelRequest(parts=[UserPromptPart("old evidence " * 300)]) for _ in range(30)
    ]
    messages.append(ModelRequest(parts=[UserPromptPart(question)]))
    projected, _, measured = await fit_synthesis(
        messages, ModelRequestParameters(), Counter(), 300
    )
    assert measured["used_tokens"] <= 300
    assert any(
        part.content == question
        for message in projected
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )


@pytest.mark.asyncio
async def test_exact_synthesis_fits_many_pairs_and_retains_ids():
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

    from rapid_mlx.chat_runtime.context import fit_synthesis

    class ExpensiveTemplateCounter(TokenCounter):
        async def measure(self, messages, parameters=None):
            result = await super().measure(messages, parameters)
            # Real count can exceed the cheap estimate because of role framing.
            result["used_tokens"] += 60 * len(messages)
            result["estimated"] = False
            return result

    messages = [ModelRequest(parts=[UserPromptPart("修复这个文件，然后说明结果。")])]
    for i in range(16):
        messages.extend(
            [
                ModelResponse(
                    parts=[
                        ToolCallPart("execute_code", {"code": "x" * 1500}, f"call-{i}")
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "execute_code", "successful result " * 300, f"call-{i}"
                        )
                    ]
                ),
            ]
        )
    counter = ExpensiveTemplateCounter()
    projected, parameters, measured = await fit_synthesis(
        messages, ModelRequestParameters(), counter, 600
    )
    assert measured["used_tokens"] <= 600
    text = str(projected)
    for i in range(16):
        assert f"call-{i}" in text
    calls = {
        part.tool_call_id
        for message in projected
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    returns = {
        part.tool_call_id
        for message in projected
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    assert calls == returns
    assert len(messages) == 33  # Original durable evidence was never mutated.


@pytest.mark.asyncio
async def test_report_includes_answer_without_reinflating_compacted_evidence():
    from pydantic_ai.messages import ModelResponse, TextPart

    emitted = []

    async def emit(event):
        emitted.append(event)

    request = SimpleNamespace(context_tokens=8192, model="test")
    deps = SimpleNamespace(emit=emit)
    manager = ContextManager(
        SimpleNamespace(model_override=True), request, deps, [], None
    )
    manager.last_messages = [ModelRequest(parts=[UserPromptPart("Compacted evidence")])]
    answer = ModelResponse(parts=[TextPart("The final answer." * 10)])
    original_history = [
        ModelRequest(parts=[UserPromptPart("obsolete data " * 5000)]),
        answer,
    ]
    result = await manager.report(original_history)
    expected = await manager.counter.measure(manager.last_messages + [answer])
    assert result["used_tokens"] == expected["used_tokens"]
    assert result["used_tokens"] < 200
    assert emitted == [result]


@pytest.mark.asyncio
async def test_checkpoint_maps_to_one_system_message_and_keeps_live_tool_pair():
    import json
    from pathlib import Path

    import httpx
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

    from rapid_mlx.chat_runtime.history import _project_messages
    from rapid_mlx.chat_runtime.model_client import ModelClients

    rows = [
        {
            "id": "old",
            "role": "user",
            "created_at": "2026-10-26T12:00:00+00:00",
            "parts": [{"type": "text", "text": "obsolete request"}],
        },
        {
            "id": "current",
            "role": "user",
            "created_at": "2026-10-26T12:01:00+00:00",
            "parts": [{"type": "text", "text": "Search official information"}],
        },
        {
            "id": "checkpoint",
            "role": "system",
            "created_at": "2026-10-26T12:01:01+00:00",
            "parts": [
                {
                    "type": "text",
                    "text": "Earlier task completed",
                    "summary": True,
                    "covers_through": "old",
                }
            ],
        },
    ]
    replay = _project_messages(rows)
    summary = replay[-1]
    assert summary.parts[0].part_kind == "user-prompt"
    # Automatic compaction prepends the same checkpoint while retaining the
    # current user and the in-flight complete tool round.
    projected = [
        summary,
        replay[0],
        ModelResponse(
            parts=[ToolCallPart("web_search", {"query": "Python"}, "search-current")]
        ),
        ModelRequest(
            parts=[ToolReturnPart("web_search", "Official result", "search-current")]
        ),
    ]
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": "local",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 1,
                    "total_tokens": 21,
                },
            },
        )

    clients = ModelClients(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    try:
        model = clients.model("http://127.0.0.1:18800", "local")
        await model.request(
            projected,
            {"max_tokens": 16},
            ModelRequestParameters(
                instruction_parts=[InstructionPart("Reply in the user's language.")]
            ),
        )
    finally:
        await clients.close()
    mapped = bodies[0]["messages"]
    assert [message["role"] for message in mapped].count("system") == 1
    assert (
        mapped[-2]["tool_calls"][0]["id"]
        == mapped[-1]["tool_call_id"]
        == "search-current"
    )
    assert "obsolete request" not in str(mapped)

    # The serving family's actual template guard rejects duplicate leading
    # system messages; rendering this provider-mapped request must succeed.
    env = ImmutableSandboxedEnvironment()

    def raise_exception(message):
        raise ValueError(message)

    env.globals["raise_exception"] = raise_exception
    template = env.from_string(
        (Path(__file__).parent / "fixtures/qwen35_chat_template.jinja").read_text()
    )
    # Serving normalizes OpenAI JSON-string tool arguments to mappings before
    # calling the model family's template.
    normalized = json.loads(json.dumps(mapped))
    for message in normalized:
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    rendered = template.render(
        messages=normalized, tools=[], add_generation_prompt=True, enable_thinking=False
    )
    assert "Official result" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_synthesis", [False, True])
@pytest.mark.parametrize("regenerate", [False, True])
async def test_adjacent_checkpoint_user_then_read_quota_commits_every_tool_pair(
    tmp_path,
    invalid_synthesis,
    regenerate,
):
    from pydantic_ai.messages import ToolReturnPart
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
    from rapid_mlx.chat_runtime.store import Store

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "records.txt").write_text(
        "\n".join(f"record {i}: " + "x" * 280 for i in range(200))
    )
    store = await Store.open(tmp_path / "profile")
    conversation = await store.create_conversation(workspace=str(workspace))
    initial = TurnRequest(
        turn_id="before-summary",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="Read these records later",
        inference_url="http://127.0.0.1:18800",
        model="test",
        context_tokens=32768,
    )
    await store.begin_turn(initial.model_dump())
    await store.finish_turn(
        initial.turn_id,
        "completed",
        parts=[{"type": "text", "text": "Ready to read records."}],
    )
    rows = await store.branch_messages(initial.branch_id)
    branch = await store.save_summary(
        initial.branch_id,
        "The user wants records read from records.txt.",
        rows[-1]["id"],
    )
    calls = []
    seen_synthesis = []

    async def stream(messages, info):
        if not info.function_tools:
            seen_synthesis.append(True)
            yield (
                "<tool_call>read_file again</tool_call>"
                if invalid_synthesis
                else "The available records were read; remaining details could not fit."
            )
            return
        number = len(calls)
        calls.append(f"read-{number}")
        yield {
            0: DeltaToolCall(
                name="read_file",
                json_args='{"path":"records.txt","offset":1,"limit":100}',
                tool_call_id=calls[-1],
            )
        }

    runtime = Runtime(
        store, model=FunctionModel(stream_function=stream), web_enabled=False
    )
    human_text = "请读取这些记录并报告确实取得的内容。"
    if regenerate:
        draft = initial.model_copy(
            update={
                "turn_id": "draft",
                "text": human_text,
                "expected_revision": branch["revision"],
            }
        )
        await store.begin_turn(draft.model_dump())
        await store.finish_turn(
            draft.turn_id,
            "completed",
            parts=[{"type": "text", "text": "A prior answer."}],
        )
        branch = await store.regenerate(initial.branch_id, "regenerate-op")
    request = initial.model_copy(
        update={
            "turn_id": "after-summary",
            "branch_id": branch["id"],
            "expected_revision": branch["revision"],
            "mode": "regenerate" if regenerate else "append",
            "text": "" if regenerate else human_text,
        }
    )
    try:
        await runtime.submit(request)
        events = [event async for event in runtime.events(request.turn_id) if event]
        assert events[-1]["event"]["outcome"] == "completed"
        assert seen_synthesis and calls
        if invalid_synthesis:
            fallback = [
                event["event"]["text"]
                for event in events
                if event["event"]["type"] == "text.delta"
                and event["event"].get("part_id") == "fallback"
            ]
            # Verify owner fallback selection uses the durable real user even
            # when a regenerate request carries text=""; this is not model
            # language-quality evidence (that remains a live acceptance check).
            assert len(fallback) == 1 and fallback[0].startswith("模型未能生成工作总结")
            assert events[-1]["event"]["error_code"] is None
            assert not any(
                "<tool_call>" in event["event"].get("text", "") for event in events
            )
        durable = await store.branch_messages(request.branch_id)
        current_parts = [
            part
            for row in durable
            if row.get("turn_id") == request.turn_id
            for part in row["parts"]
        ]
        durable_calls = [
            part["call_id"] for part in current_parts if part["type"] == "tool_call"
        ]
        durable_returns = [
            part["call_id"] for part in current_parts if part["type"] == "tool_result"
        ]
        assert durable_calls == durable_returns == calls
        assert any(
            part["type"] == "tool_result" and "record 0:" in part["result"]
            for part in current_parts
        )
        from rapid_mlx.chat_runtime.history import _project_messages

        projected = _project_messages(durable)
        assert any(
            isinstance(part, ToolReturnPart)
            for message in projected
            for part in message.parts
        )
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.asyncio
async def test_context_disabled_first_request_cannot_dispatch_unexpected_tools(
    tmp_path,
):
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
    from rapid_mlx.chat_runtime.store import Store

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = await Store.open(tmp_path / "profile")
    conversation = await store.create_conversation(workspace=str(workspace))
    enabled = []

    async def stream(messages, info):
        enabled.append(bool(info.function_tools))
        if len(enabled) == 1:
            # A provider may ignore the empty tool schema; the owner must still
            # enforce the tools-disabled boundary before dispatch.
            yield {
                0: DeltaToolCall(
                    name="write_file",
                    json_args='{"path":"unexpected.txt","content":"effect"}',
                    tool_call_id="unexpected",
                )
            }
        else:
            yield "Visible answer without executing another action."

    runtime = Runtime(
        store, model=FunctionModel(stream_function=stream), web_enabled=False
    )
    request = TurnRequest(
        turn_id="disabled-first",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="很长的中文输入" * 10,
        inference_url="http://127.0.0.1:18800",
        model="fixture",
        context_tokens=2048,
    )
    try:
        await runtime.submit(request)
        events = [event async for event in runtime.events(request.turn_id) if event]
        assert enabled == [False, False]
        assert not (workspace / "unexpected.txt").exists()
        assert events[-1]["event"]["outcome"] == "completed"
        assert not any(event["event"]["type"] == "tool.started" for event in events)
        messages = await store.branch_messages(request.branch_id)
        assert not any(
            part["type"] == "tool_call"
            for message in messages
            for part in message["parts"]
        )
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.asyncio
async def test_synthesis_preserves_human_request_and_final_retry_notice():
    from pydantic_ai.models import ModelRequestParameters

    from rapid_mlx.chat_runtime.context import RuntimeNoticePart, fit_synthesis

    class Counter(TokenCounter):
        async def measure(self, messages, parameters=None):
            result = await super().measure(messages, parameters)
            result["used_tokens"] += 60 * len(messages)
            return result

    question = "Human request: " + "important details " * 30
    notice = "Retry the truncated tool call."
    messages = [
        ModelRequest(parts=[UserPromptPart("old evidence " * 300)]) for _ in range(30)
    ]
    messages.extend(
        [
            ModelRequest(parts=[UserPromptPart(question)]),
            ModelRequest(parts=[RuntimeNoticePart(notice)]),
        ]
    )
    projected, _, measured = await fit_synthesis(
        messages, ModelRequestParameters(), Counter(), 400
    )
    assert measured["used_tokens"] <= 400
    parts = [part for message in projected for part in message.parts]
    assert any(
        isinstance(part, UserPromptPart) and part.content == question for part in parts
    )
    assert isinstance(parts[-1], RuntimeNoticePart) and parts[-1].content == notice
