"""Output truncation must never dispatch partial actions or lose completed work."""

import json
from contextlib import asynccontextmanager

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import IncompleteToolCall
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from tests.test_chat_runtime_actions import setup


class LengthModel(FunctionModel):
    """Real agent streaming/validation, with a provider length finish reason."""

    def __init__(self, stream, lengths):
        super().__init__(stream_function=stream)
        self.lengths = set(lengths)
        self.requests = 0

    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        self.requests += 1
        async with super().request_stream(*args, **kwargs) as response:
            if self.requests in self.lengths:
                response.finish_reason = "length"
            yield response


def call(name, args, call_id="broken"):
    return {0: DeltaToolCall(name=name, json_args=args, tool_call_id=call_id)}


@pytest.mark.asyncio
async def test_dependency_raises_incomplete_tool_call():
    async def stream(messages, info):
        yield call("write_file", '{"path":"a","content":"unfinished')

    agent = Agent(LengthModel(stream, [1]), retries=0)

    @agent.tool_plain
    def write_file(path: str, content: str) -> str:
        pytest.fail("partial call executed")

    with pytest.raises(IncompleteToolCall):
        async with agent.iter("write a file") as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as events:
                        async for _ in events:
                            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["write_file", "patch", "terminal", "execute_code", "delegate_task"]
)
@pytest.mark.parametrize(
    "args", ['{"content":"unfinished', '{"content":"valid but cut"}']
)
async def test_length_tool_response_retries_without_dispatch(tmp_path, name, args):
    if args.endswith("}"):
        args = json.dumps(
            {
                "write_file": {"path": "must-not-exist", "content": "partial"},
                "patch": {"path": "must-not-exist", "old_text": "a", "new_text": "b"},
                "terminal": {"command": "echo should-not-run"},
                "execute_code": {
                    "language": "python",
                    "code": "print('should-not-run')",
                },
                "delegate_task": {"task": "Do not run this helper"},
            }[name]
        )
    requests = []

    async def stream(messages, info):
        requests.append(messages)
        if len(requests) == 1:
            yield "准备好了。"
            yield call(name, args)
        elif len(requests) == 2:
            assert not any(
                p.part_kind == "tool-call" for m in messages for p in m.parts
            )
            assert "truncated" in str(messages).lower() or "截断" in str(messages)
            yield call(
                "write_file",
                json.dumps({"path": "ok.asm", "content": "ret\n"}),
                "complete",
            )
        else:
            yield "已完成。"

    store, runtime, request, workspace = await setup(tmp_path, LengthModel(stream, [1]))
    request.text = "用汇编帮我写一个快排算法"
    try:
        await runtime.submit(request)
        task = runtime._tasks[request.turn_id]
        await task
        events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        assert events[-1]["outcome"] == "completed"
        assert (workspace / "ok.asm").read_text() == "ret\n"
        assert [e["call_id"] for e in events if e["type"] == "tool.started"] == [
            "complete"
        ]
        rows = await store.branch_messages(request.branch_id)
        assert '"call_id": "broken"' not in json.dumps(rows)
        assert (
            sum(p.get("call_id") == "complete" for r in rows for p in r["parts"]) == 2
        )
        assert len(requests) == 3
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("chinese", [True, False])
async def test_error_preserves_completed_write_and_visible_text(
    tmp_path, chinese, caplog
):
    requests = 0

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield call(
                "write_file",
                json.dumps({"path": "done.asm", "content": "ret\n"}),
                "saved",
            )
        else:
            yield "Visible partial answer"
            raise RuntimeError("PRIVATE_PROMPT_SECRET")

    store, runtime, request, workspace = await setup(
        tmp_path, FunctionModel(stream_function=stream)
    )
    request.text = "请写文件" if chinese else "Write a file"
    try:
        await runtime.submit(request)
        task = runtime._tasks[request.turn_id]
        await task
        events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        assert events[-1]["outcome"] == "error"
        assert (workspace / "done.asm").read_text() == "ret\n"
        rows = await store.branch_messages(request.branch_id)
        parts = [p for r in rows for p in r["parts"]]
        assert sum(p.get("call_id") == "saved" for p in parts) == 2
        assert any(p.get("text") == "Visible partial answer" for p in parts)
        fallback = next(e["text"] for e in events if e.get("part_id") == "fallback")
        assert ("未能" in fallback) if chinese else ("couldn't" in fallback)
        assert parts[-1]["text"] == fallback
        assert "RuntimeError" in caplog.text
        assert "PRIVATE_PROMPT_SECRET" not in caplog.text
        assert events[-1]["usage"]["requests"] >= 1
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.asyncio
async def test_text_length_has_durable_continue_notice(tmp_path):
    async def stream(messages, info):
        yield "这是部分答案"

    store, runtime, request, _ = await setup(tmp_path, LengthModel(stream, [1]))
    request.text = "写长文"
    try:
        await runtime.submit(request)
        task = runtime._tasks[request.turn_id]
        await task
        events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        text = "".join(e["text"] for e in events if e["type"] == "text.delta")
        assert "这是部分答案" in text and "继续" in text and "截断" in text
        rows = await store.branch_messages(request.branch_id)
        assert "继续" in json.dumps(rows, ensure_ascii=False)
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.asyncio
async def test_repeated_truncation_is_bounded_and_keeps_first_write(tmp_path):
    requests = []

    async def stream(messages, info):
        requests.append(messages)
        if len(requests) == 1:
            yield call(
                "write_file",
                json.dumps({"path": "first.asm", "content": "ret\n"}),
                "first",
            )
        else:
            yield call(
                "write_file",
                '{"path":"second.asm","content":"unfinished',
                f"broken-{len(requests)}",
            )

    model = LengthModel(stream, range(2, 20))
    store, runtime, request, workspace = await setup(tmp_path, model)
    try:
        await runtime.submit(request)
        events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        assert events[-1]["outcome"] == "completed"
        assert len(requests) == 6  # original attempt plus four retries, then stop
        assert not (workspace / "second.asm").exists()
        assert (workspace / "first.asm").read_text() == "ret\n"
        assert [e["call_id"] for e in events if e["type"] == "tool.started"] == [
            "first"
        ]
        rows = await store.branch_messages(request.branch_id)
        assert sum(p.get("call_id") == "first" for r in rows for p in r["parts"]) == 2
        assert "smaller operation" in rows[-1]["parts"][-1]["text"]
        assert events[-1]["usage"]["requests"] == 6
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.asyncio
async def test_next_turn_sees_failed_turn_receipts(tmp_path):
    requests = 0

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield call(
                "write_file",
                json.dumps({"path": "first.asm", "content": "ret\n"}),
                "first",
            )
        elif requests == 2:
            raise RuntimeError("failed after write")
        else:
            assert (
                sum(
                    p.part_kind == "tool-return" and p.tool_call_id == "first"
                    for m in messages
                    for p in m.parts
                )
                == 1
            )
            yield "The first file was already written."

    store, runtime, request, _ = await setup(
        tmp_path, FunctionModel(stream_function=stream)
    )
    try:
        for index in range(2):
            if index:
                conversation = await store.get_conversation(request.conversation_id)
                branch = next(
                    b for b in conversation["branches"] if b["id"] == request.branch_id
                )
                request = request.model_copy(
                    update={
                        "turn_id": "next",
                        "expected_revision": branch["revision"],
                        "text": "continue",
                    }
                )
            await runtime.submit(request)
            events = [e["event"] async for e in runtime.events(request.turn_id) if e]
            assert events[-1]["outcome"] == ("error" if not index else "completed")
        assert requests == 3
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [None, 512])
async def test_output_default_uses_remaining_context_and_honors_override(tmp_path, cap):
    observed = []

    async def stream(messages, info):
        observed.append(info.model_settings["max_tokens"])
        yield "Done"

    store, runtime, request, _ = await setup(
        tmp_path, FunctionModel(stream_function=stream)
    )
    if cap:
        request.max_tokens = cap
    try:
        await runtime.submit(request)
        events = [e["event"] async for e in runtime.events(request.turn_id) if e]
        context = next(e for e in events if e["type"] == "context")
        assert observed[0] == cap if cap else observed[0] > 8192
        assert observed[0] + context["used_tokens"] + 128 <= request.context_tokens
    finally:
        await runtime.close()
        await store.close()
