"""One model-facing result allowance across reads, web and local execution."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic_ai import ApprovalRequired
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Store
from rapid_mlx.chat_runtime.tools import TurnDeps, bounded_result


async def test_parallel_results_share_exact_utf8_allowance_and_preserve_events():
    events = []

    async def emit(event):
        events.append(event)

    deps = TurnDeps(None, "c", frozenset(), emit, read_budget_bytes=50000)

    @bounded_result
    async def result(ctx):
        await asyncio.sleep(0)
        content = "é" * 5000
        await ctx.deps.emit({"type": "tool.result", "result": content})
        return content

    results = await asyncio.gather(
        *(result(SimpleNamespace(deps=deps)) for _ in range(3))
    )
    assert deps.result_bytes == sum(len(r.encode()) for r in results) <= 16000
    assert deps.result_budget_exhausted
    assert len(results[0].encode()) == 10000
    assert "budget reached" in results[1]
    assert results[2] == ""
    assert all(len(event["result"].encode()) == 10000 for event in events)
    # History compaction may replenish context room but not the turn allowance.
    deps.read_budget_bytes = 50000
    assert await result(SimpleNamespace(deps=deps)) == ""


async def test_approval_deferral_does_not_consume_result_budget():
    deps = TurnDeps(None, "c", frozenset(), None)

    @bounded_result
    async def deferred(ctx):
        raise ApprovalRequired()

    with pytest.raises(ApprovalRequired):
        await deferred(SimpleNamespace(deps=deps))
    assert deps.result_bytes == 0
    assert not deps.result_budget_exhausted


@pytest.mark.parametrize("tool", ["terminal", "read_file"])
async def test_runtime_output_cap_keeps_full_event_and_synthesizes(tmp_path, tool):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "large.txt").write_text(("x" * 99 + "\n") * 180)
    store = await Store.open(tmp_path / "profile")
    conversation = await store.create_conversation(workspace=str(workspace))
    requests = []

    async def stream(messages, info):
        requests.append(info)
        if info.function_tools:
            yield {
                index: DeltaToolCall(
                    name=tool,
                    json_args=(
                        '{"command":"cat large.txt"}'
                        if tool == "terminal"
                        else '{"path":"large.txt"}'
                    ),
                    tool_call_id=f"large-{index}",
                )
                for index in range(3)
            }
        else:
            results = [
                part.content
                for message in messages
                for part in message.parts
                if part.part_kind == "tool-return"
            ]
            assert results
            assert sum(len(r.encode()) for r in results) <= 16000
            assert "No tools remain" in info.instructions
            yield "The result was large; here is the available evidence."

    runtime = Runtime(
        store, model=FunctionModel(stream_function=stream), web_enabled=False
    )
    request = TurnRequest(
        turn_id="bounded",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="Read the large file",
        inference_url="http://localhost:18800",
        model="fake",
        context_tokens=32768,
        max_tokens=1024,
    )
    try:
        await runtime.submit(request)
        events = [e async for e in runtime.events(request.turn_id) if e]
        assert events[-1]["event"]["outcome"] == "completed", events[-1]
        assert len(requests) == 2
        results = [e["event"] for e in events if e["event"]["type"] == "tool.result"]
        if tool == "terminal":
            assert sum(len(result["result"].encode()) for result in results) > 16000
        else:
            assert any("1|" in result["result"] for result in results)
        assert await store.replay_events(request.turn_id) == events
        assert any(
            "available evidence" in e["event"].get("text", "")
            for e in events
            if e["event"]["type"] == "text.delta"
        )
    finally:
        await runtime.close()
        await store.close()
