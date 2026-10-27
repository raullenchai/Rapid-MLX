# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for chat client and presentation boundaries."""

import io
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from rich.console import Console

from rapid_mlx import chat_client
from rapid_mlx.chat_client import ChatClient
from rapid_mlx.chat_harness_render import HarnessRenderer, _Preview, approval_card
from rapid_mlx.chat_state import TurnState, sse_events


@pytest.fixture
def client(monkeypatch):
    c = ChatClient(SimpleNamespace(), "http://localhost", "model")
    c.conversation_id = "conv"
    c.branch_id = "branch"
    c.refresh = Mock()
    yield c
    c.client.close()


def test_state_validation_and_context(monkeypatch):
    with pytest.raises(ValueError, match="size limit"):
        list(sse_events(["x" * (2 * 1024 * 1024 + 1)]))
    state = TurnState("t", reasoning_since=10)
    monkeypatch.setattr("rapid_mlx.chat_state.time.monotonic", lambda: 12)
    assert state.reasoning_seconds == 2
    with pytest.raises(ValueError, match="Incompatible"):
        state.apply({"wire_version": 2})
    with pytest.raises(ValueError, match="sequence"):
        state.apply({"wire_version": 1, "turn_id": "t", "seq": 0})
    event = {
        "wire_version": 1,
        "turn_id": "t",
        "seq": 1,
        "event": {"type": "context", "notice": "near limit"},
    }
    assert state.apply(event)
    assert state.context["notice"] == "near limit"
    assert state.timeline == [("notice", "1")]
    event.update(seq=2, event={"type": "turn.terminal", "outcome": "mystery"})
    with pytest.raises(ValueError, match="Unknown"):
        state.apply(event)


def test_renderer_preview_and_notice():
    assert "more lines" in approval_card({"preview": "\n".join(map(str, range(8)))})
    output = io.StringIO()
    console = Console(file=output, width=40, height=4)
    console.print(_Preview("one\ntwo\nthree"))
    assert "two" in output.getvalue() and "one" not in output.getvalue()
    state = TurnState("t", notices={"1": "notice"}, timeline=[("notice", "1")])
    state.subagents["c"] = {
        "id": "c",
        "title": "helper",
        "tool_count": 0,
        "status": "queued",
        "call_id": "missing",
    }
    renderer = HarnessRenderer(state, stream=output)
    console.print(renderer.view())
    assert "notice" in output.getvalue() and "helper" in output.getvalue()
    renderer._plain_thinking = True
    renderer.finish()
    assert "Thought for" in output.getvalue()


def test_client_status_and_context(client, monkeypatch, capsys):
    console = Mock()
    monkeypatch.setattr(chat_client, "supports_rich_output", lambda _: True)
    monkeypatch.setattr(chat_client, "_console", lambda _: console)
    chat_client.print_status("hello")
    console.print.assert_called_once()
    client.request = Mock(return_value={"conversations": [{"id": "a"}]})
    assert client.conversations() == [{"id": "a"}]
    client.snapshot = {"uncertain_operations": [{"tool": "write", "id": "x"}]}
    client.show_history = Mock()
    assert client.select_conversation("last")
    assert "Outcome unknown" in capsys.readouterr().out
    client.branch_id = "branch"
    client.snapshot = {"branches": [{"id": "branch", "revision": 1}]}
    client.context_tokens = Mock(return_value=100)
    monkeypatch.setenv("RAPID_MLX_API_KEY", "secret")
    assert client.context_request()["inference_api_key"] == "secret"


def test_client_commands(client, tmp_path, capsys):
    client.snapshot = {
        "branches": [{"id": "branch"}],
        "messages": [
            {
                "id": "m",
                "role": "user",
                "parts": [{"type": "text", "text": "hello"}],
                "sources": [
                    {"source_id": "1", "title": "site", "url": "https://example.com"}
                ],
            }
        ],
    }
    client.request = Mock(return_value={"id": "next"})
    client.turn = Mock()
    client.input.prompt = Mock(return_value=" replacement ")
    assert client.command("/edit")
    assert client.request.call_args.kwargs["json"]["text"] == "replacement"
    client.last_state = TurnState("t", reasoning="thought")
    client.command("/show-thinking")
    assert "thought" in capsys.readouterr().out
    path = tmp_path / "export.md"
    client.export_subruns = Mock(return_value=[])
    client.command(f"/export {path}")
    assert "hello" in path.read_text() and "example.com" in path.read_text()
    with pytest.raises(ValueError, match="Use /export"):
        client.command("/export a b")
    client.snapshot["messages"] = []
    with pytest.raises(ValueError, match="Send a message"):
        client.command("/regen")


def test_attach_errors(client, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("hello")
    response = httpx.Response(
        400,
        json={"detail": {"message": "bad file"}},
        request=httpx.Request("POST", "http://localhost"),
    )
    client.request = Mock(
        side_effect=httpx.HTTPStatusError(
            "bad", request=response.request, response=response
        )
    )
    with pytest.raises(ValueError, match="bad file"):
        client.command(f"/attach {path}")
    response = httpx.Response(
        400, content=b"bad", request=httpx.Request("POST", "http://localhost")
    )
    client.request.side_effect = httpx.HTTPStatusError(
        "bad", request=response.request, response=response
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.command(f"/attach {path}")


def test_history_and_branches(client, capsys):
    client.snapshot = {"branches": [{"id": "branch"}, {"id": "other"}], "messages": []}
    client.show_history = Mock()
    client.select_conversation = Mock()
    client.command("/history last")
    client.select_conversation.assert_called_once_with("last")
    client.command("/branch")
    assert "* 1." in capsys.readouterr().out
    client.attachments = ["a"]
    client.command("/branch 2")
    assert client.branch_id == "other" and client.attachments == []
    client.show_history.assert_called_once()


@pytest.mark.parametrize("rich", [False, True])
def test_turn_reconnect_context_permission_and_error(client, monkeypatch, capsys, rich):
    client.context_request = Mock(return_value={"turn_id": "t"})
    client.connect = Mock()
    client.permission = Mock()
    monkeypatch.setattr(chat_client.time, "sleep", lambda _: None)
    monkeypatch.setattr(chat_client, "supports_rich_output", lambda _: rich)
    console = Mock()
    monkeypatch.setattr(chat_client, "_console", lambda _: console)
    events = [
        {
            "wire_version": 1,
            "turn_id": "t",
            "seq": 1,
            "event": {"type": "context", "limit_tokens": 100, "percent": 75},
        },
        {
            "wire_version": 1,
            "turn_id": "t",
            "seq": 2,
            "event": {"type": "permission.request"},
        },
        {
            "wire_version": 1,
            "turn_id": "t",
            "seq": 3,
            "event": {"type": "turn.terminal", "outcome": "error", "message": "failed"},
        },
    ]
    response = Mock(status_code=200)
    response.iter_lines.return_value = []
    unauthorized = Mock(status_code=401)
    from contextlib import nullcontext

    client.client.stream = Mock(
        side_effect=[nullcontext(unauthorized), nullcontext(response)]
    )
    monkeypatch.setattr(chat_client, "sse_events", lambda _: iter(events))
    client.turn("hello")
    assert client.input.context_percent == 75
    client.permission.assert_called_once()
    client.connect.assert_called_once()
    if rich:
        assert "failed" in str(console.print.call_args)
    else:
        assert "failed" in capsys.readouterr().out


def test_run_handles_errors_and_interrupts(client, monkeypatch, capsys):
    monkeypatch.setattr(chat_client.threading, "Thread", lambda **kwargs: Mock())
    client.connect = Mock()
    client.new = Mock()
    client.turn = Mock()
    response = httpx.Response(400, request=httpx.Request("GET", "http://localhost"))
    client.input.prompt = Mock(
        side_effect=[
            "",
            "hello",
            ValueError("bad"),
            httpx.ConnectError("offline"),
            httpx.HTTPStatusError("bad", request=response.request, response=response),
            KeyboardInterrupt(),
            KeyboardInterrupt(),
        ]
    )
    client.run()
    client.turn.assert_called_once_with("hello")
    out = capsys.readouterr().out
    assert (
        "bad" in out
        and "connection unavailable" in out
        and "rejected (400)" in out
        and "Ctrl-C" in out
    )


@pytest.mark.parametrize("last", ["quit", EOFError()])
def test_run_exit(client, monkeypatch, last):
    monkeypatch.setattr(chat_client.threading, "Thread", lambda **kwargs: Mock())
    client.connect = Mock()
    client.new = Mock()
    client.input.prompt = Mock(side_effect=[last])
    client.run()
    client.new.assert_called_once()


def test_editor_binding_navigation_and_acceptance(monkeypatch):
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.output.vt100 import Vt100_Output

    from rapid_mlx.chat_client import COMMANDS
    from rapid_mlx.chat_input import ChatInput

    monkeypatch.setattr(
        "rapid_mlx.chat_input.sys",
        SimpleNamespace(
            stdin=Mock(isatty=Mock(return_value=True)),
            stdout=Mock(isatty=Mock(return_value=True)),
        ),
    )
    monkeypatch.setattr(
        Vt100_Output, "from_pty", lambda *_: SimpleNamespace(enable_cpr=True)
    )
    session = Mock()
    monkeypatch.setattr("prompt_toolkit.PromptSession", Mock(return_value=session))
    reader = ChatInput(COMMANDS)
    reader.prompt("> ")
    registry = __import__("prompt_toolkit").PromptSession.call_args.kwargs[
        "key_bindings"
    ]
    handlers = {
        binding.handler.__name__: binding.handler for binding in registry.bindings
    }
    state = Mock(
        complete_index=None, completions=[Completion("/help"), Completion("/history")]
    )
    buffer = Mock(complete_state=state, text="/help")
    event = SimpleNamespace(
        current_buffer=buffer, key_sequence=[SimpleNamespace(key="up")], app=Mock()
    )
    handlers["navigate"](event)
    state.go_to_index.assert_called_with(1)
    state.complete_index = 1
    event.key_sequence[0].key = "down"
    handlers["navigate"](event)
    state.go_to_index.assert_called_with(0)
    event.key_sequence[0].key = "c-m"
    handlers["accept_completion"](event)
    buffer.validate_and_handle.assert_called_once()
    from prompt_toolkit.document import Document

    buffer.complete_state = state
    buffer.text = "/he"
    buffer.document = Document("/he")
    state.current_completion = None
    event.key_sequence[0].key = "tab"
    handlers["accept_completion"](event)
    buffer.apply_completion.assert_called_once_with(state.completions[0])
    buffer.complete_state = None
    buffer.document = Document("/unknown")
    handlers["accept_completion"](event)
    assert buffer.apply_completion.call_count == 1
    reader._approval_options = ("y", "n")
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", lambda fn: fn())
    reader._print_approval_card = Mock()
    handlers["expand_approval"](event)
    assert reader._approval_expanded
    reader._print_approval_card.assert_called_once()


def test_attach_completion_rejects_unrecoverable_quote():
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from rapid_mlx.chat_client import COMMANDS
    from rapid_mlx.chat_input import command_completer

    assert (
        list(
            command_completer(COMMANDS).get_completions(
                Document('/attach "\\'), CompleteEvent()
            )
        )
        == []
    )
