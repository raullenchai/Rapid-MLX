"""Token accounting and checkpoint regressions for long tool conversations."""

import json
from dataclasses import replace

import httpx
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from rapid_mlx.chat_runtime.history import _project_messages, compact_tool_history
from rapid_mlx.chat_runtime.model_client import ModelClients
from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Conflict, Store
from rapid_mlx.chat_runtime.token_budget import (
    TokenCounter,
    context_tokens,
    estimate_text,
)


def request(c, **kwargs):
    return TurnRequest(
        turn_id="turn",
        conversation_id=c["id"],
        branch_id=c["branch_id"],
        expected_revision=0,
        inference_url="http://localhost:18800",
        model="local",
        context_tokens=8192,
        **kwargs,
    )


async def test_server_tokens_include_tools_cached_and_calibrated():
    bodies = []

    def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"input_tokens": 1234})

    from pydantic_ai.messages import InstructionPart
    from pydantic_ai.tools import ToolDefinition

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    clients = ModelClients(http_client=http)
    try:
        counter = TokenCounter(clients, request({"id": "c", "branch_id": "b"}))
        params = ModelRequestParameters(
            instruction_parts=[InstructionPart("中文系统提示")],
            function_tools=[
                ToolDefinition(name="read", parameters_json_schema={"type": "object"})
            ],
        )
        messages = [ModelRequest(parts=[UserPromptPart("这是简体中文。" * 300)])]
        first = await counter.measure(messages, params)
        again = await counter.measure(messages, params)
        assert first["used_tokens"] == again["used_tokens"] == 1234
        assert not first["estimated"]
        assert len(bodies) == 1
        assert "中文系统提示" in bodies[0]["system"]
        assert bodies[0]["tools"][0]["name"] == "read"
        counter.calibrate(1260)
        assert (await counter.measure(messages, params))["used_tokens"] == 1260
        assert (
            sum(
                first[k]
                for k in (
                    "system_tools_tokens",
                    "conversation_tokens",
                    "tool_output_tokens",
                )
            )
            == 1234
        )
    finally:
        await clients.close()


async def test_count_unavailable_uses_unicode_estimate_not_bytes():
    calls = []

    def handle(req):
        calls.append(req)
        return httpx.Response(404)

    clients = ModelClients(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    try:
        counter = TokenCounter(clients, request({"id": "c", "branch_id": "b"}))
        message = ModelRequest(parts=[UserPromptPart("中文" * 100)])
        result = await counter.measure([message])
        await counter.measure(
            [replace(message, parts=[UserPromptPart("新中文" * 100)])]
        )
        assert result["estimated"]
        assert len(calls) == 1
        assert estimate_text("中文" * 100) == 400
        assert estimate_text("a" * 1500) == 500
    finally:
        await clients.close()


def test_compaction_keeps_call_result_pairs_and_originals():
    messages = [
        ModelRequest(parts=[UserPromptPart("读文件并修复")]),
        ModelResponse(
            parts=[ToolCallPart("read_file", {"path": "large.txt"}, "read-1")]
        ),
        ModelRequest(parts=[ToolReturnPart("read_file", "数据" * 3000, "read-1")]),
    ]
    compacted = compact_tool_history(messages, 2048)
    assert context_tokens(compacted) < 2048
    assert (
        compacted[1].parts[0].tool_call_id
        == compacted[2].parts[0].tool_call_id
        == "read-1"
    )
    assert messages[-1].parts[0].content == "数据" * 3000


async def test_durable_checkpoint_replay_and_stale_revision(tmp_path):
    store = await Store.open(tmp_path)
    c = await store.create_conversation()
    req = request(c, text="我的项目使用简体中文，文件是 calc.py。")
    turn, _ = await store.begin_turn(req.model_dump())
    await store.finish_turn(
        req.turn_id, "completed", parts=[{"type": "text", "text": "记住了。"}]
    )
    rows = await store.branch_messages(req.branch_id)
    branch = (await store.get_conversation(c["id"], req.branch_id))["branches"][0]
    await store.save_summary(
        req.branch_id,
        "用户用简体中文，项目文件 calc.py。",
        rows[-1]["id"],
        expected_revision=branch["revision"],
    )
    with __import__("pytest").raises(Conflict):
        await store.save_summary(
            req.branch_id, "stale", rows[-1]["id"], expected_revision=branch["revision"]
        )
    await store.close()
    store = await Store.open(tmp_path)
    try:
        replay = await store.branch_messages(req.branch_id)
        assert len(replay) == 3  # Originals remain available to history/export.
        projected = _project_messages(replay)
        assert len(projected) == 1
        assert "calc.py" in projected[0].parts[0].content
        assert replay[-1]["parts"][0]["summary"] is True
    finally:
        await store.close()


async def test_long_chinese_tool_args_and_results_complete(tmp_path):
    calls = 0
    code = (
        "# 中文代码说明\n"
        + "values = "
        + repr(list(range(350)))
        + "\nprint(sum(values))"
    )

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            yield "已经完成读取，保留了执行结果。"
        else:
            yield {
                0: DeltaToolCall(
                    name="read_attachment",
                    json_args=json.dumps({"attachment_id": attachment["id"]}),
                    tool_call_id="large-read",
                )
            }

    store = await Store.open(tmp_path)
    c = await store.create_conversation()
    attachment = await store.put_attachment(
        c["id"], "large.txt", ("大文件内容\n" * 1800).encode()
    )
    runtime = Runtime(
        store, model=FunctionModel(stream_function=stream), web_enabled=False
    )
    req = request(
        c,
        text=("请用简体中文读取文件并解释这段代码。\n" * 30) + code,
        attachment_ids=[attachment["id"]],
    )
    try:
        await runtime.submit(req)
        events = [e["event"] async for e in runtime.events(req.turn_id) if e]
        assert events[-1]["outcome"] == "completed"
        assert any(e["type"] == "tool.result" for e in events)
        assert any(e["type"] == "text.delta" and "完成" in e["text"] for e in events)
        assert any(e["type"] == "context" for e in events)
        assert not any(e.get("error_code") == "context_budget_exceeded" for e in events)
        _project_messages(await store.branch_messages(req.branch_id))
    finally:
        await runtime.close()
        await store.close()


async def test_auto_summary_is_durable_and_next_turn_replays_it(tmp_path):
    summaries = []

    def summarize(messages, info):
        summaries.append(messages)
        return ModelResponse(
            parts=[
                TextPart(
                    "摘要：用户使用简体中文，目标是修复 calc.py，保留已完成的测试。"
                )
            ]
        )

    seen = []

    async def stream(messages, info):
        seen.append(messages)
        yield "已保留摘要，继续处理。"

    store = await Store.open(tmp_path)
    c = await store.create_conversation()
    for i in range(5):
        branch = await store.get_branch(c["branch_id"])
        req = request(c, text="这是较早的对话内容，项目是 calc.py。" * 90)
        req.turn_id = f"old-{i}"
        req.expected_revision = branch["revision"]
        await store.begin_turn(req.model_dump())
        await store.finish_turn(
            req.turn_id,
            "completed",
            parts=[{"type": "text", "text": "已记录项目约束和测试结果。" * 70}],
        )
    runtime = Runtime(
        store,
        model=FunctionModel(function=summarize, stream_function=stream),
        web_enabled=False,
    )
    try:
        req = request(c, text="请继续处理。")
        req.expected_revision = (await store.get_branch(req.branch_id))["revision"]
        await runtime.submit(req)
        events = [e["event"] async for e in runtime.events(req.turn_id) if e]
        assert events[-1]["outcome"] == "completed"
        assert summaries
        assert any(e["type"] == "context" and e["compacted"] for e in events)
        rows = await store.branch_messages(req.branch_id)
        checkpoints = [r for r in rows if any(p.get("summary") for p in r["parts"])]
        assert len(checkpoints) == 1
        assert len(rows) == 13
        projection = _project_messages(rows)
        assert any(
            "摘要" in getattr(p, "content", "") for m in projection for p in m.parts
        )
        assert len(projection) < len(rows)
        req = req.model_copy(
            update={
                "turn_id": "next",
                "text": "摘要是否保留？",
                "expected_revision": (await store.get_branch(req.branch_id))[
                    "revision"
                ],
            }
        )
        await runtime.submit(req)
        events = [e["event"] async for e in runtime.events(req.turn_id) if e]
        assert events[-1]["outcome"] == "completed"
        assert any(
            "摘要" in getattr(p, "content", "") for m in seen[-1] for p in m.parts
        )
    finally:
        await runtime.close()
        await store.close()


async def test_executed_long_code_and_large_output_never_budget_interrupt(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code = (
        "values = "
        + repr(list(range(450)))
        + '\nprint("大结果" * 1800)\nprint(sum(values))'
    )
    assert len(code.encode()) > 1500
    calls = []

    async def stream(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            assert info.function_tools
            yield {
                0: DeltaToolCall(
                    name="execute_code",
                    json_args=json.dumps({"language": "python", "code": code}),
                    tool_call_id="long-code",
                )
            }
        else:
            yield "代码已执行，已保存结果。"

    store = await Store.open(tmp_path / "store")
    c = await store.create_conversation(workspace=str(workspace))
    runtime = Runtime(
        store, model=FunctionModel(stream_function=stream), web_enabled=False
    )
    req = request(c, text="请运行长代码并保留输出结果。" * 100)
    # Leave room for current system/tool instructions before exercising output growth.
    req.context_tokens = 16384
    try:
        await runtime.submit(req)
        events = [e["event"] async for e in runtime.events(req.turn_id) if e]
        assert any(
            e["type"] == "tool.result" and e["outcome"] == "success" for e in events
        )
        assert events[-1]["outcome"] == "completed"
        assert any(e["type"] == "text.delta" and "已执行" in e["text"] for e in events)
        rows = await store.branch_messages(req.branch_id)
        projected = _project_messages(rows)
        native_calls = [
            p for m in projected for p in m.parts if isinstance(p, ToolCallPart)
        ]
        native_results = [
            p for m in projected for p in m.parts if isinstance(p, ToolReturnPart)
        ]
        assert len(native_calls) == len(native_results) == 1
        assert (
            native_calls[0].tool_call_id
            == native_results[0].tool_call_id
            == "long-code"
        )
    finally:
        await runtime.close()
        await store.close()


async def test_summary_during_turn_preserves_action_dispatch_fence(tmp_path):
    store = await Store.open(tmp_path)
    try:
        c = await store.create_conversation()
        req = request(c, text="Run a safe action after compaction")
        turn, _ = await store.begin_turn(req.model_dump())
        await store.save_summary(
            req.branch_id, "Prior context", turn["input_id"], turn_id=req.turn_id
        )
        current = await store.get_turn(req.turn_id)
        branch = await store.get_branch(req.branch_id)
        assert current["revision"] == branch["revision"]
        assert current["input_id"] == turn["input_id"]
        await store.finish_turn(
            req.turn_id, "completed", parts=[{"type": "text", "text": "Done"}]
        )
        rows = await store.branch_messages(req.branch_id)
        assert rows[-2]["parts"][0]["summary"]
        assert rows[-1]["parent_id"] == rows[-2]["id"]
    finally:
        await store.close()
