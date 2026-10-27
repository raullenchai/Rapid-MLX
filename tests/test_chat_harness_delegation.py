# SPDX-License-Identifier: Apache-2.0
"""Child progress stays separate from answers, permissions, and branch history."""

import io
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from rich.console import Console

from rapid_mlx.chat_client import PREFIX, ChatClient
from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState
from tests.test_chat_harness_terminal import terminal_capture


def envelope(seq, kind, **data):
    return {
        "wire_version": 1,
        "turn_id": "turn",
        "seq": seq,
        "event": {"type": kind, **data},
    }


def child(kind, id="child", status="running", count=0, **kwargs):
    return dict(
        type=kind,
        id=id,
        call_id="delegate",
        title="Researching docs",
        status=status,
        tool_count=count,
        summary="Searching documents",
        **kwargs,
    )


def apply(state, data):
    result = envelope(state.cursor + 1, **data)
    assert state.apply(result)
    return result


def view(renderer):
    output = io.StringIO()
    Console(file=output, width=120, color_system=None).print(renderer.view())
    return output.getvalue()


def test_child_replay_separates_activity_from_answer_and_collapses_completion():
    state = TurnState("turn")
    renderer = HarnessRenderer(state, stream=io.StringIO(), expanded=True)
    apply(state, dict(kind="tool.started", call_id="delegate", name="delegate_task"))
    item = child("subagent.started")
    item["kind"] = item.pop("type")
    first = apply(state, item)
    assert not state.apply(first)
    assert state.text == state.final_text == ""
    progress = child("subagent.progress", count=2)
    progress["kind"] = progress.pop("type")
    apply(state, progress)
    active = view(renderer)
    assert "↳ Researching docs… (2 tools)" in active
    assert "Searching documents" in active
    assert state.tools["delegate"]["status"] == "running"
    done = child("subagent.finished", status="completed", count=2)
    done["kind"] = done.pop("type")
    apply(state, done)
    collapsed = view(renderer)
    assert "↳ Researching docs · done (2 tools)" in collapsed
    assert "Searching documents" not in collapsed
    with pytest.raises(ValueError, match="no visible answer"):
        state.apply(envelope(state.cursor + 1, "turn.terminal", outcome="completed"))


@pytest.mark.parametrize(
    "outcome,expected", [("cancelled", "cancelled"), ("error", "error")]
)
def test_parent_terminal_closes_unfinished_child_rows(outcome, expected):
    state = TurnState("turn")
    data = child("subagent.started", status="queued")
    data["kind"] = data.pop("type")
    apply(state, data)
    apply(state, dict(kind="turn.terminal", outcome=outcome))
    assert state.subagents["child"]["status"] == expected
    assert state.text == ""


def test_plain_child_progress_is_bounded_and_control_safe():
    state = TurnState("turn")
    output = io.StringIO()
    renderer = HarnessRenderer(state, stream=output)
    for kind, count, status in [
        ("subagent.started", 0, "running"),
        ("subagent.progress", 1, "running"),
        ("subagent.progress", 2, "running"),
        ("subagent.finished", 2, "budget_exhausted"),
    ]:
        data = child(kind, status=status, count=count)
        data["title"] = "Researching docs\x1b\x07\nINJECT"
        data["kind"] = data.pop("type")
        renderer.update(apply(state, data))
    rendered = output.getvalue()
    assert len(rendered.splitlines()) == 2
    assert "budget reached (2 tools)" in rendered
    assert "\x1b" not in rendered and "\x07" not in rendered
    assert state.text == ""


def test_nested_children_pty_progress_finishes_once_and_never_shows_transcripts():
    events = [
        {"type": "tool.started", "call_id": "delegate", "name": "delegate_task"},
        child("subagent.started", "docs"),
        child("subagent.started", "todos"),
        child("subagent.progress", "docs", count=2),
        child("subagent.progress", "todos", count=1),
        child("subagent.finished", "docs", status="completed", count=2),
        child("subagent.finished", "todos", status="cancelled", count=1),
        {
            "type": "tool.result",
            "call_id": "delegate",
            "outcome": "success",
            "result": "HIDDEN_CHILD_TRANSCRIPT",
        },
        {"type": "text.delta", "part_id": "answer", "text": "COMBINED_REPORT"},
        {"type": "turn.terminal", "outcome": "completed"},
    ]
    for data in events:
        if data.get("id") == "todos":
            data["title"] = "Finding TODOs"
    script = f"""
from rapid_mlx.chat_harness_render import HarnessRenderer
from rapid_mlx.chat_state import TurnState
state = TurnState("turn")
renderer = HarnessRenderer(state, expanded=True)
for seq, data in enumerate({events!r}, 1):
    item = {{"wire_version": 1, "turn_id": "turn", "seq": seq, "event": data}}
    state.apply(item)
    renderer.update(item)
renderer.finish()
renderer.finish()
"""
    display, raw = terminal_capture(script, columns=110)
    assert "↳ Researching docs… (2 tools)".encode() in raw
    assert display.count("↳ Researching docs · done (2 tools)") == 1
    assert display.count("↳ Finding TODOs · cancelled (1 tool)") == 1
    assert display.count("COMBINED_REPORT") == 1
    assert "Searching documents" not in display
    assert "HIDDEN_CHILD_TRANSCRIPT" not in display


def test_child_approval_is_attributed_and_uses_existing_permission_route(
    monkeypatch, capsys
):
    client = ChatClient(SimpleNamespace(), "http://127.0.0.1:18800", "fixture")
    state = TurnState("turn")
    state.subagents["child"] = {"title": "Edit docs\x1b\x07"}
    renderer = HarnessRenderer(state, stream=io.StringIO())
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"state": "intent", "revision": 3}

    monkeypatch.setattr(client, "request", request)
    approvals = []
    monkeypatch.setattr(
        client.input, "approval", lambda **kwargs: approvals.append(kwargs) or "n"
    )
    client.input.approval_reason = "Do not edit"
    try:
        client.permission(
            {
                "op_id": "operation",
                "revision": 3,
                "tool": "write_file",
                "preview": "diff",
                "reason": "Requires approval",
                "options": ["once", "deny"],
                "expires_at": time.time() + 60,
                "subagent_id": "child",
            },
            renderer,
        )
        text = capsys.readouterr().out
        assert "(helper: Edit docs)" in approvals[0]["data"]["reason"]
        assert "\x1b" not in text and "\x07" not in text
        assert calls[-1][1] == "/permission.response"
        assert calls[-1][2]["json"] == {
            "op_id": "operation",
            "revision": 3,
            "decision": "deny",
            "scope": "once",
            "reason": "Do not edit",
        }
    finally:
        client.client.close()


def test_export_fetches_authenticated_hidden_runs_once_without_visible_history_changes(
    monkeypatch, tmp_path, capsys
):
    client = ChatClient(SimpleNamespace(), "http://127.0.0.1:18800", "fixture")
    client.client.close()
    client.endpoint = "http://127.0.0.1:18801"
    client.headers = {"Authorization": "Bearer fixture-owner-token"}
    client.snapshot = {
        "messages": [
            {
                "role": "user",
                "turn_id": "turn",
                "parts": [{"type": "text", "text": "Research"}],
            },
            {
                "role": "assistant",
                "turn_id": "turn",
                "parts": [{"type": "text", "text": "Parent summary"}],
            },
        ]
    }
    original = json.dumps(client.snapshot)
    calls = []
    detail = {
        "id": "child",
        "turn_id": "turn",
        "call_id": "delegate",
        "title": "Docs </summary><script>",
        "status": "completed",
        "tool_count": 2,
        "task": "Read docs",
        "context": "<private>",
        "summary": "Child findings",
        "tools": ["read_file"],
        "usage": {},
        "messages": [
            {
                "role": "assistant",
                "parts": [{"type": "text", "text": "CHILD_ONLY </pre>"}],
            }
        ],
        "events": [{"seq": 1, "event": {"type": "tool.started", "name": "read_file"}}],
        "sources": [
            {"source_id": "1", "title": "Source", "url": "https://example.com/docs"}
        ],
    }

    def serve(request):
        assert request.headers["Authorization"] == "Bearer fixture-owner-token"
        calls.append(request.url.path)
        if request.url.path == PREFIX + "/turns/turn/subruns":
            return httpx.Response(200, json=[{"id": "child"}])
        assert request.url.path == PREFIX + "/subruns/child"
        return httpx.Response(200, json=detail)

    client.client = httpx.Client(transport=httpx.MockTransport(serve))
    monkeypatch.setattr(client, "refresh", lambda: client.snapshot)
    try:
        client.show_history()
        assert "CHILD_ONLY" not in capsys.readouterr().out
        path = tmp_path / "export.md"
        client.command(f"/export {path}")
        exported = path.read_text()
        assert len(calls) == 2
        assert exported.count("<details>") == 1
        assert "CHILD_ONLY &lt;/pre&gt;" in exported
        assert "&lt;/summary&gt;&lt;script&gt;" in exported
        assert "https://example.com/docs" in exported
        assert json.dumps(client.snapshot) == original
    finally:
        client.client.close()
