# SPDX-License-Identifier: Apache-2.0
"""In-process terminal lifecycle, reconnect and presentation regressions."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from rapid_mlx import chat_client, chat_harness_render
from rapid_mlx.chat_client import COMMANDS, ChatClient, select_item
from rapid_mlx.chat_harness_render import HarnessRenderer, approval_card, source_lines
from rapid_mlx.chat_input import ChatInput, command_completer
from rapid_mlx.chat_state import TurnState


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def editor(monkeypatch):
    from prompt_toolkit.output.vt100 import Vt100_Output

    output = Terminal()
    monkeypatch.setattr(
        "rapid_mlx.chat_input.sys", SimpleNamespace(stdin=Terminal(), stdout=output)
    )
    monkeypatch.setattr(Vt100_Output, "from_pty", lambda *a, **kw: DummyOutput())
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        reader = ChatInput(COMMANDS)
        yield reader, pipe, output
        reader.close()


def test_editor_reuses_session_context_and_timeout(editor, monkeypatch):
    reader, pipe, _ = editor
    reader.context_percent = 91
    monkeypatch.setenv("NO_COLOR", "1")
    pipe.send_text("hello\r")
    assert reader.prompt("› ", timeout=0.5) == "hello"
    session = reader._session
    pipe.send_text("second\r")
    assert reader.prompt("Thinking ›", complete_commands=False, timeout=0.5) == "second"
    assert reader._session is session
    assert reader.prompt("› ", timeout=0, timeout_result="expired") == "expired"
    reader.close()
    assert reader._session is None


@pytest.mark.parametrize(
    "keys,allow,expected",
    [
        ("y", False, "y"),
        ("a", True, "a"),
        ("n", False, "n"),
        ("\x1b[B\r", False, "n"),
        ("\x1b[A\r", True, "n"),
        ("3", True, "n"),
        ("4y", False, "y"),
    ],
)
def test_editor_approval_keys_select_only_offered_choices(
    editor, keys, allow, expected
):
    reader, pipe, output = editor
    pipe.send_text(keys)
    result = reader.approval(
        allow_conversation=allow, timeout=0.5, data={"title": "Run action?"}
    )
    assert result == expected
    assert "Run action?" in output.getvalue()
    assert reader._approval_options == ()
    assert reader._approval_data is None


def test_editor_denial_reason_is_bounded_and_approval_resets(editor):
    reader, pipe, _ = editor
    pipe.send_text("N" + "reason" * 110 + "\r")
    assert reader.approval(timeout=1) == "n"
    assert reader.approval_reason == ("reason" * 110)[:512]
    assert reader._approval_options == ()
    assert reader.approval(timeout=0) == "n"
    assert reader.approval_reason == ""


def test_editor_completion_binding_preserves_existing_argument(editor):
    reader, pipe, _ = editor
    # Complete a command at the cursor without duplicating its argument separator.
    pipe.send_text("/att file.txt\x01\x1b[C\x1b[C\x1b[C\x1b[C\t\r")
    assert reader.prompt("› ", timeout=1) == "/attach file.txt"


@pytest.mark.parametrize("text", ["/attach '\"", "/attach '\\"])
def test_attachment_completion_rejects_unclosed_mixed_quotes(text):
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    assert (
        list(
            command_completer(COMMANDS).get_completions(Document(text), CompleteEvent())
        )
        == []
    )


def render_text(renderable):
    out = io.StringIO()
    Console(file=out, width=100, color_system=None).print(renderable)
    return out.getvalue()


def test_approval_expanded_card_keeps_all_lines_and_sanitizes_heading(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    card = approval_card(
        {
            "title": "Edit\x1b[31m file",
            "preview": "\n".join(f"line {i}" for i in range(9)),
        },
        expanded=True,
    )
    assert "line 8" in card
    assert "v to collapse" in card
    assert "\x1b" not in card
    assert "Allow this action?" in approval_card({})


@pytest.mark.parametrize(
    "url",
    [
        None,
        "https://bad\n.example",
        "file:///secret",
        "https:///missing",
        "https://user:pass@example.org",
        "http://[invalid",
    ],
)
def test_source_renderer_omits_invalid_or_credentialed_urls(url):
    assert source_lines([{"url": url, "source_id": 1, "title": "source"}]) == []


def test_renderer_plain_reasoning_helpers_and_notices(monkeypatch):
    monkeypatch.setattr(
        chat_harness_render, "supports_rich_output", lambda stream: False
    )
    stream = Terminal()
    state = TurnState("turn")
    renderer = HarnessRenderer(state, stream=stream, expanded=True)
    state.reasoning_since = 1.0
    renderer.update({"event": {"type": "reasoning.delta", "text": "thought\x1b[31m"}})
    state.reasoning_since = None
    renderer.update({"event": {"type": "context", "notice": "Context compacted"}})
    renderer.update({"event": {"type": "usage", "notice": "Usage notice"}})
    child = {
        "type": "subagent.finished",
        "title": "Helper",
        "tool_count": 1,
        "status": "budget_exhausted",
    }
    renderer.update({"event": child})
    renderer.finish()
    text = stream.getvalue()
    assert "Thinking…" in text and "Thought for" in text
    assert "Context compacted" in text and "Usage notice" in text
    assert "budget reached (1 tool)" in text
    assert "\x1b[31m" not in text


def test_renderer_result_details_and_child_summary(monkeypatch):
    monkeypatch.setattr(
        chat_harness_render, "supports_rich_output", lambda stream: False
    )
    renderer = HarnessRenderer(TurnState("turn"), stream=io.StringIO(), expanded=True)
    tool = {
        "name": "patch",
        "status": "error",
        "summary": "Rejected",
        "details": ["+added", "-removed", "same"],
        "details_kind": "diff",
        "omitted_lines": 4,
    }
    rows = renderer.result_view(tool, identify=True)
    assert [str(row.style) for row in rows[1:4]] == ["green", "red", "dim"]
    assert "4 more lines" in rows[-1].plain
    child = {
        "id": "child",
        "title": "Helper",
        "status": "queued",
        "tool_count": 0,
        "summary": "Searching",
    }
    rows = renderer.child_view(child)
    assert "queued" in rows[0].plain and rows[1].plain.strip() == "Searching"
    child["status"] = "completed"
    assert len(renderer.child_view(child)) == 1


def test_renderer_live_pause_resume_preserves_committed_rows(monkeypatch):
    monkeypatch.setattr(
        chat_harness_render, "supports_rich_output", lambda stream: True
    )
    live = Mock()
    live.console = Console(file=io.StringIO(), width=100)
    monkeypatch.setattr(chat_harness_render, "Live", Mock(return_value=live))
    state = TurnState("turn")
    state.reasoning = "private thought"
    state.reasoning_since = 1.0
    state.parts = {"part": "answer"}
    state.tools = {
        "call": {
            "name": "read_file",
            "args": {},
            "status": "success",
            "summary": "Read file",
        }
    }
    state.timeline = [("tool", "call"), ("text", "part"), ("result", "call")]
    state.subagents = {
        "child": {
            "id": "child",
            "call_id": "call",
            "title": "Helper",
            "tool_count": 1,
            "status": "completed",
            "summary": "Done",
        }
    }
    renderer = HarnessRenderer(state, stream=Terminal(), expanded=True)
    shown = render_text(renderer.view())
    assert "private thought" in shown and "answer" in shown and "Helper" in shown
    renderer.pause()
    assert renderer._committed_timeline == 3 and renderer._committed_children == {
        "child"
    }
    renderer.resume()
    renderer.update()
    renderer.finish()
    renderer.finish()
    assert live.start.call_count == 2
    assert live.stop.call_count == 2
    assert "answer" not in render_text(renderer.view())


@pytest.fixture
def client(monkeypatch):
    instance = ChatClient(SimpleNamespace(), "http://127.0.0.1:8000", "fixture")
    instance.endpoint = "http://127.0.0.1:9000"
    instance.conversation_id = "conversation"
    instance.branch_id = "branch"
    yield instance
    instance.input.close()
    instance.client.close()


@pytest.mark.parametrize("choice", ["missing", "a", "0"])
def test_selection_rejects_missing_and_ambiguous_ids(choice):
    with pytest.raises(ValueError, match="choose"):
        select_item([{"id": "abc"}, {"id": "abd"}], choice, "choose")


@pytest.mark.parametrize("first", [401, "transport"])
def test_request_reconnects_once_and_retries_original_payload(
    client, monkeypatch, first
):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            if first == "transport":
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(401)
        return httpx.Response(204)

    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(respond))
    reconnect = Mock()
    monkeypatch.setattr(client, "connect", reconnect)
    assert (
        client.request("POST", "/clients", json={"client_id": "one"}, idempotent=True)
        == {}
    )
    assert reconnect.call_count == 1 and len(calls) == 2
    assert json.loads(calls[-1].content) == {"client_id": "one"}


def test_request_transport_failure_after_retry_propagates(client, monkeypatch):
    client.client.close()

    def fail(request):
        raise httpx.ConnectError("offline", request=request)

    client.client = httpx.Client(transport=httpx.MockTransport(fail))
    reconnect = Mock()
    monkeypatch.setattr(client, "connect", reconnect)
    with pytest.raises(httpx.ConnectError):
        client.request("GET", "/capabilities")
    assert reconnect.call_count == 1
    with pytest.raises(httpx.ConnectError):
        client.request("GET", "/capabilities", retry=False)
    assert reconnect.call_count == 1


def test_connect_requires_supported_wire_version(client, monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.chat_runtime.owner.discover_or_start",
        lambda profile: {"endpoint": "http://127.0.0.1:1/", "token": "test-token"},
    )
    request = Mock(return_value={"wire_version": 2})
    monkeypatch.setattr(client, "request", request)
    with pytest.raises(ValueError, match="upgrade"):
        client.connect()
    assert client.endpoint == "http://127.0.0.1:1"
    assert request.call_count == 1


def test_heartbeat_tolerates_transient_errors_and_close_releases_lease(
    client, monkeypatch
):
    stopping = Mock()
    stopping.wait.side_effect = [False, True]
    client.stopping = stopping
    monkeypatch.setattr(
        client, "request", Mock(side_effect=httpx.ConnectError("offline"))
    )
    client.heartbeat()
    thread = Mock()
    client.lease_thread = thread
    client.close()
    stopping.set.assert_called_once()
    thread.join.assert_called_once_with(timeout=6)


def test_refresh_falls_back_to_first_branch_and_no_saved_history(
    client, monkeypatch, capsys
):
    client.branch_id = None
    monkeypatch.setattr(
        client, "request", Mock(return_value={"branches": [{"id": "first"}]})
    )
    client.refresh()
    assert client.branch_id == "first"
    monkeypatch.setattr(client, "conversations", lambda: [])
    assert client.select_conversation("last") is False
    assert "No saved conversations" in capsys.readouterr().out


def test_approval_stale_revision_does_not_prompt(client, monkeypatch):
    monkeypatch.setattr(
        client, "request", lambda *a, **kw: {"state": "intent", "revision": 3}
    )
    renderer = Mock()
    client.permission({"op_id": "op", "revision": 2}, renderer)
    renderer.pause.assert_not_called()


@pytest.mark.parametrize("status", [409, 500])
def test_approval_conflict_is_visible_and_renderer_always_resumes(
    client, monkeypatch, capsys, status
):
    response = httpx.Response(
        status, request=httpx.Request("POST", "http://owner/permission.response")
    )
    error = httpx.HTTPStatusError(
        "rejected", request=response.request, response=response
    )
    monkeypatch.setattr(
        client, "request", Mock(side_effect=[{"state": "intent", "revision": 1}, error])
    )
    monkeypatch.setattr(client.input, "approval", Mock(return_value="a"))
    renderer = Mock()
    renderer.state.subagents = {"helper": {"title": "Research"}}
    data = {
        "op_id": "op",
        "revision": 1,
        "subagent_id": "helper",
        "reason": "Run?",
        "options": ["conversation"],
        "expires_at": 0,
    }
    if status == 500:
        with pytest.raises(httpx.HTTPStatusError):
            client.permission(data, renderer)
    else:
        client.permission(data, renderer)
        assert "already resolved" in capsys.readouterr().out
    renderer.resume.assert_called_once()
    assert (
        "helper: Research" in client.input.approval.call_args.kwargs["data"]["reason"]
    )


def test_command_invalid_arguments_and_revealed_reasoning(
    client, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(client, "refresh", lambda: client.snapshot)
    for command in ["/think maybe", "/attach", "/attach a b", "/regen"]:
        with pytest.raises(ValueError):
            client.command(command)
    client.snapshot = {"messages": [{"role": "user", "id": "user"}]}
    monkeypatch.setattr(client.input, "prompt", lambda *a, **kw: "  ")
    with pytest.raises(ValueError, match="empty"):
        client.command("/edit")
    path = tmp_path / "large.txt"
    path.write_bytes(b"abc")
    monkeypatch.setattr(chat_client, "MAX_ATTACHMENT_BYTES", 2)
    with pytest.raises(ValueError, match="20 MiB"):
        client.command(f"/attach {path}")
    client.last_state = SimpleNamespace(reasoning="visible thought")
    client.command("/show-thinking")
    client.command("/show-thinking")
    client.command("/unknown")
    text = capsys.readouterr().out
    assert (
        "visible thought" in text
        and "Thinking collapsed" in text
        and "Unknown command" in text
    )


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("bad profile"),
        RuntimeError("bad state"),
        OSError("offline"),
        httpx.ConnectError("offline"),
    ],
)
def test_entrypoint_closes_client_on_connection_failure(monkeypatch, failure, capsys):
    instance = Mock()
    instance.run.side_effect = failure
    monkeypatch.setattr(chat_client, "ChatClient", Mock(return_value=instance))
    with pytest.raises(SystemExit) as exc:
        chat_client.run_chat(SimpleNamespace(), "http://127.0.0.1", "fixture")
    assert exc.value.code == 1
    instance.close.assert_called_once()
    assert capsys.readouterr().err


@pytest.mark.parametrize("option", ["mcp_config", "system"])
def test_entrypoint_refuses_unsupported_features_before_constructing_client(
    monkeypatch, option
):
    constructor = Mock()
    monkeypatch.setattr(chat_client, "ChatClient", constructor)
    with pytest.raises(SystemExit) as exc:
        chat_client.run_chat(
            SimpleNamespace(**{option: "configured"}), "http://127.0.0.1", "fixture"
        )
    assert exc.value.code == 2
    constructor.assert_not_called()
