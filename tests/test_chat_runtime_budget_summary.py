"""Tool exhaustion ends with a grounded summary, never another action."""

import asyncio
import json

import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.usage import UsageLimits

from tests.test_chat_runtime_actions import setup
from tests.test_chat_runtime_truncation import LengthModel


def tool(name, args, call_id):
    return {
        0: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id=call_id)
    }


@pytest.mark.parametrize("chinese", [True, False])
@pytest.mark.parametrize("empty_first", [True, False])
@pytest.mark.parametrize("limit_kind", ["requests", "tools"])
async def test_exhaustion_summarizes_completed_work_with_tools_disabled(
    tmp_path, chinese, empty_first, limit_kind
):
    calls, summaries = [], []
    answer = (
        "已写入 quicksort.asm；尚未验证编译。"
        if chinese
        else "Saved quicksort.asm; compilation remains unverified."
    )

    async def stream(messages, info):
        calls.append(info)
        if info.function_tools:
            yield tool(
                "write_file",
                {"path": "quicksort.asm", "content": "ret\n"},
                f"saved-{len(calls)}",
            )
        else:
            summaries.append((messages, info))
            if empty_first and len(summaries) == 1:
                yield ""
            else:
                yield answer

    store, runtime, request, workspace = await setup(
        tmp_path, FunctionModel(stream_function=stream)
    )
    runtime.usage_limits = UsageLimits(
        request_limit=1 if limit_kind == "requests" else None,
        tool_calls_limit=1 if limit_kind == "tools" else None,
    )
    request.text = "创建并编译快排程序" if chinese else "Create and compile quicksort"
    try:
        await runtime.submit(request)
        async with asyncio.timeout(10):
            events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        assert len(summaries) == 1 + empty_first
        assert all(
            "summarizing what you've found and accomplished so far" in info.instructions
            for _, info in summaries
        )
        assert all(
            any(p.part_kind == "tool-return" for m in messages for p in m.parts)
            for messages, _ in summaries
        )
        text = "".join(e["text"] for e in events if e["type"] == "text.delta")
        assert answer in text
        assert (
            ("预算" in text and "继续" in text)
            if chinese
            else ("budget" in text and "continue" in text)
        )
        assert events[-1]["outcome"] == "completed"
        assert events[-1]["usage"]["requests"] == 2 + empty_first + (
            limit_kind == "tools"
        )
        assert (workspace / "quicksort.asm").read_text() == "ret\n"
        assert len([e for e in events if e["type"] == "tool.started"]) == 1
        rows = await store.branch_messages(request.branch_id)
        persisted = json.dumps(rows, ensure_ascii=False)
        assert answer in persisted
        assert ("继续" if chinese else "continue") in persisted
        assert await store.replay_events(request.turn_id) == [
            e async for e in runtime.events(request.turn_id) if e
        ]
    finally:
        await runtime.close()
        await store.close()


async def test_default_iteration_budget_allows_more_than_old_request_and_tool_caps(
    tmp_path,
):
    calls = []

    async def stream(messages, info):
        calls.append(info)
        if len(calls) <= 18:
            yield tool(
                "write_file",
                {"path": "count.txt", "content": str(len(calls))},
                f"write-{len(calls)}",
            )
        else:
            yield "Eighteen writes completed."

    store, runtime, request, workspace = await setup(
        tmp_path, FunctionModel(stream_function=stream)
    )
    try:
        await runtime.submit(request)
        async with asyncio.timeout(10):
            events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        assert (workspace / "count.txt").read_text() == "18"
        assert len(calls) == 19
        assert events[-1]["outcome"] == "completed"
        assert not any("budget" in e.get("text", "") for e in events)
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.parametrize("failure", ["error", "empty", "tool"])
async def test_failed_summary_preserves_work_and_continuation_notice(tmp_path, failure):
    summaries = 0

    async def stream(messages, info):
        nonlocal summaries
        if info.function_tools:
            yield tool("write_file", {"path": "done.asm", "content": "ret\n"}, "saved")
        else:
            summaries += 1
            if failure == "error":
                raise RuntimeError("PRIVATE_SUMMARY_INPUT")
            if failure == "tool":
                yield tool(
                    "write_file",
                    {"path": "must-not-exist", "content": "bad"},
                    "disabled",
                )
            else:
                yield ""

    store, runtime, request, workspace = await setup(
        tmp_path, FunctionModel(stream_function=stream)
    )
    runtime.usage_limits = UsageLimits(request_limit=1)
    request.text = "创建并编译程序"
    try:
        await runtime.submit(request)
        async with asyncio.timeout(10):
            events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        text = "".join(e["text"] for e in events if e["type"] == "text.delta")
        assert "预算" in text and "继续" in text and "总结" in text
        assert "我无法根据现有信息" not in text
        assert "PRIVATE_SUMMARY_INPUT" not in text
        assert summaries == (1 if failure == "error" else 2)
        assert events[-1]["outcome"] == "completed"
        assert (workspace / "done.asm").read_text() == "ret\n"
        assert not (workspace / "must-not-exist").exists()
        rows = await store.branch_messages(request.branch_id)
        assert (
            sum(p.get("call_id") == "saved" for row in rows for p in row["parts"]) == 2
        )
        assert "disabled" not in json.dumps(rows)
    finally:
        await runtime.close()
        await store.close()


async def test_truncated_call_at_request_limit_still_summarizes_completed_work(
    tmp_path,
):
    requests, summaries = [], []

    async def stream(messages, info):
        requests.append(info)
        if not info.function_tools:
            summaries.append(messages)
            yield "已保存 done.asm；后续写入被截断，尚未编译。"
        elif len(requests) == 1:
            yield tool("write_file", {"path": "done.asm", "content": "ret\n"}, "saved")
        else:
            yield {
                0: DeltaToolCall(
                    name="write_file",
                    json_args='{"path":"partial.asm","content":"',
                    tool_call_id="truncated",
                )
            }

    store, runtime, request, workspace = await setup(tmp_path, LengthModel(stream, [2]))
    runtime.usage_limits = UsageLimits(request_limit=2)
    request.text = "创建并编译快排程序"
    try:
        await runtime.submit(request)
        async with asyncio.timeout(10):
            events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        assert len(summaries) == 1
        assert any(p.part_kind == "tool-return" for m in summaries[0] for p in m.parts)
        assert events[-1]["usage"]["requests"] == 3
        text = "".join(e["text"] for e in events if e["type"] == "text.delta")
        assert "已保存 done.asm" in text and "预算" in text and "继续" in text
        assert (workspace / "done.asm").read_text() == "ret\n"
        assert not (workspace / "partial.asm").exists()
        rows = await store.branch_messages(request.branch_id)
        assert (
            sum(p.get("call_id") == "saved" for row in rows for p in row["parts"]) == 2
        )
        assert not any(
            p.get("call_id") == "truncated" for row in rows for p in row["parts"]
        )
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.parametrize("overflow_after_write", [False, True])
async def test_context_overflow_transitions_before_generation_to_work_summary(
    tmp_path, monkeypatch, overflow_after_write
):
    from rapid_mlx.chat_runtime.token_budget import TokenCounter

    original_measure = TokenCounter.measure
    requests = []

    async def measure(self, messages, parameters=None):
        measured = await original_measure(self, messages, parameters)
        has_result = any(
            p.part_kind == "tool-return" for m in messages for p in m.parts
        )
        if parameters.function_tools and (has_result or not overflow_after_write):
            measured["used_tokens"] = 40000
        return measured

    monkeypatch.setattr(TokenCounter, "measure", measure)

    async def stream(messages, info):
        requests.append(info)
        if info.function_tools:
            yield tool("write_file", {"path": "done.asm", "content": "ret\n"}, "saved")
        elif (
            "summarizing what you've found and accomplished so far" in info.instructions
        ):
            yield (
                "已保存文件；编译尚未验证。"
                if overflow_after_write
                else "尚未开始写入，当前上下文已满。"
            )
        else:
            yield '<tool_call>{"name":"write_file","arguments":{}}</tool_call>'

    store, runtime, request, workspace = await setup(
        tmp_path, FunctionModel(stream_function=stream)
    )
    request.text = "创建并编译程序"
    request.approvals = "always"
    try:
        await runtime.submit(request)
        events = []
        async with asyncio.timeout(10):
            async for envelope in runtime.events(request.turn_id):
                if not envelope:
                    continue
                event = envelope["event"]
                events.append(event)
                if event["type"] == "permission.request":
                    await runtime.actions.respond(
                        event["op_id"], event["revision"], "allow"
                    )
        assert len(requests) == 1 + overflow_after_write
        assert not requests[-1].function_tools
        assert (
            "summarizing what you've found and accomplished so far"
            in requests[-1].instructions
        )
        text = "".join(e["text"] for e in events if e["type"] == "text.delta")
        assert "<tool_call>" not in text
        assert "预算" in text and "继续" in text
        assert ("已保存文件" if overflow_after_write else "尚未开始写入") in text
        assert events[-1]["outcome"] == "completed"
        assert (workspace / "done.asm").exists() == overflow_after_write
        rows = await store.branch_messages(request.branch_id)
        assert (
            sum(p.get("call_id") == "saved" for row in rows for p in row["parts"])
            == 2 * overflow_after_write
        )
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.parametrize("length_summary", [False, True])
async def test_tool_shaped_summary_text_is_retried_without_display_or_persistence(
    tmp_path, length_summary
):
    requests = []

    async def stream(messages, info):
        requests.append(info)
        if len(requests) == 1:
            yield tool("write_file", {"path": "done.asm", "content": "ret\n"}, "saved")
        elif len(requests) == 2:
            yield "<tool_call><function=terminal><parameter=command>nasm done.asm"
        else:
            yield "已写入 done.asm，尚未验证编译。"

    model = LengthModel(stream, [2] if length_summary else [])
    store, runtime, request, workspace = await setup(tmp_path, model)
    runtime.usage_limits = UsageLimits(request_limit=1)
    request.text = "创建并编译程序"
    try:
        await runtime.submit(request)
        async with asyncio.timeout(10):
            events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        assert len(requests) == 3
        assert all(not info.function_tools for info in requests[1:])
        text = "".join(e["text"] for e in events if e["type"] == "text.delta")
        assert "<tool_call>" not in text and "<function=" not in text
        assert "已写入 done.asm" in text
        rows = await store.branch_messages(request.branch_id)
        assert "<tool_call>" not in json.dumps(rows)
        assert (workspace / "done.asm").read_text() == "ret\n"
        assert events[-1]["outcome"] == "completed"
    finally:
        await runtime.close()
        await store.close()
