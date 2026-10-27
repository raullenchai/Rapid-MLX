# SPDX-License-Identifier: Apache-2.0
"""Harness reducer, terminal rendering, and opt-in CLI contracts."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import httpx
import pytest

from rapid_mlx.chat_client import COMMAND_BY_NAME, COMMANDS, HELP, ChatClient
from rapid_mlx.chat_harness_render import HarnessRenderer, safe_markdown, tool_progress
from rapid_mlx.chat_state import ReplayRequiredError, TurnState, sse_events
from rapid_mlx.cli_parser import build_parser


@pytest.fixture
def client_factory():
    clients = []

    def create(args=None, url="http://127.0.0.1:18800", model="fixture"):
        client = ChatClient(args or SimpleNamespace(), url, model)
        clients.append(client)
        return client

    yield create
    for client in clients:
        client.client.close()


@pytest.mark.parametrize(
    "method,idempotent",
    [("POST", False), ("POST", True), ("GET", False), ("DELETE", False)],
)
def test_transport_retry_respects_idempotency(
    client_factory, monkeypatch, method, idempotent
):
    client = client_factory()
    calls = []
    reconnects = []

    def send(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise httpx.ConnectError("lost")
        return httpx.Response(
            200, json={"ok": True}, request=httpx.Request(method, "http://localhost")
        )

    monkeypatch.setattr(client.client, "request", send)
    monkeypatch.setattr(client, "connect", lambda: reconnects.append(True))
    if method == "POST" and not idempotent:
        with pytest.raises(httpx.TransportError):
            client.request(method, "/mutation", idempotent=idempotent)
        assert len(calls) == 1
    else:
        assert client.request(method, "/mutation", idempotent=idempotent) == {
            "ok": True
        }
        assert len(calls) == 2
    assert reconnects == ([] if method == "POST" and not idempotent else [True])


def test_unauthorized_mutation_is_retried(client_factory, monkeypatch):
    client = client_factory()
    calls = []

    def send(*args, **kwargs):
        calls.append(args)
        return httpx.Response(
            401 if len(calls) == 1 else 200,
            json={"ok": True},
            request=httpx.Request("POST", "http://localhost"),
        )

    monkeypatch.setattr(client.client, "request", send)
    monkeypatch.setattr(client, "connect", lambda: None)
    assert client.request("POST", "/mutation") == {"ok": True}
    assert len(calls) == 2


def test_cancel_is_marked_idempotent(client_factory, monkeypatch):
    client = client_factory()
    monkeypatch.setattr(
        client, "context_request", lambda *args, **kwargs: {"turn_id": "turn"}
    )
    monkeypatch.setattr(client, "refresh", lambda: None)
    monkeypatch.setattr(
        client.client,
        "stream",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    calls = []

    def cancel(*args, **kwargs):
        calls.append((args, kwargs))
        client.last_state.terminal = {"outcome": "cancelled"}

    monkeypatch.setattr(client, "request", cancel)
    client.turn("hello")
    assert calls == [(("POST", "/turns/turn/cancel"), {"json": {}, "idempotent": True})]


@pytest.mark.parametrize(
    ("entry", "requested", "expected"),
    [
        ({"context_window": 262144, "max_model_len": 65536}, None, 32768),
        ({"context_window": 262144, "max_model_len": 8192}, None, 8192),
        ({"context_window": 4096, "max_model_len": 8192}, None, 4096),
        ({"context_window": 262144, "max_model_len": 16384}, 32768, 16384),
        ({"context_window": 262144, "max_model_len": 65536}, 32768, 32768),
        ({"max_model_len": 4096}, None, 4096),
        ({"context_window": 32768}, None, 32768),
        ({"context_window": True, "max_model_len": "16384"}, None, 32768),
        ({"context_window": -1, "max_model_len": None}, None, 32768),
        ({}, 4096, 4096),
    ],
)
def test_context_budget_respects_native_and_served_limits(
    client_factory, entry, requested, expected
):
    client = client_factory(
        SimpleNamespace(context_length=requested),
        "http://localhost:18800/v1",
        "fixture",
    )
    client.client.close()
    client.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"data": [{"id": "fixture", **entry}]}
            )
        )
    )
    assert client.context_tokens() == expected


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ([{"id": "other", "max_model_len": 4096}], 4096),
        ([{"id": "other", "max_model_len": 4096}, {"id": "another"}], 32768),
        (
            [
                {"id": "other", "max_model_len": 4096},
                {"id": "fixture", "max_model_len": 12288},
            ],
            12288,
        ),
        (None, 32768),
    ],
)
def test_context_budget_model_selection_and_auth(
    client_factory, entries, expected, monkeypatch
):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "local-test-key")
    client = client_factory(SimpleNamespace(), "http://localhost:18800", "fixture")
    client.client.close()

    def serve(request):
        assert str(request.url) == "http://localhost:18800/v1/models"
        assert request.headers["Authorization"] == "Bearer local-test-key"
        return httpx.Response(200, json={"data": entries})

    client.client = httpx.Client(transport=httpx.MockTransport(serve))
    assert client.context_tokens() == expected


def test_context_budget_refreshes_server_limit_and_falls_back_on_failure(
    client_factory,
):
    responses = iter(
        [
            httpx.Response(
                200, json={"data": [{"id": "fixture", "max_model_len": 16384}]}
            ),
            httpx.Response(
                200, json={"data": [{"id": "fixture", "max_model_len": 4096}]}
            ),
            httpx.Response(503),
        ]
    )
    client = client_factory()
    client.client.close()
    client.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: next(responses))
    )
    assert [client.context_tokens() for _ in range(3)] == [16384, 4096, 4096]


def test_help_uses_the_command_registry(client_factory, capsys):
    client = client_factory()
    assert client.command("/help")
    assert capsys.readouterr().out == HELP + "\n"
    rows = [line.split() for line in HELP.splitlines() if line.startswith("  /")]
    assert [row[0] for row in rows] == [command.name for command in COMMANDS]
    for row, command in zip(rows, COMMANDS):
        assert " ".join(row[1:]) == " ".join(
            (command.argument_hint + " " + command.description).split()
        )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("/reset", "/new"),
        ("/clear", "/new"),
        ("/?", "/help"),
        ("/save", "/export"),
        ("/exit", "/quit"),
        ("/bye", "/quit"),
    ],
)
def test_command_aliases_resolve_and_execute_identically(
    client_factory, alias, canonical, monkeypatch, capsys
):
    assert COMMAND_BY_NAME[alias] is COMMAND_BY_NAME[canonical]
    client = client_factory()
    calls = []
    monkeypatch.setattr(client, "new", lambda: calls.append("new"))
    monkeypatch.setattr(client, "refresh", lambda: calls.append("refresh"))
    client.snapshot = {"messages": []}
    expected = client.command(canonical)
    expected_output = capsys.readouterr().out
    expected_calls = calls[:]
    calls.clear()
    assert client.command(alias) == expected
    assert capsys.readouterr().out == expected_output
    assert calls == expected_calls


def event(seq, kind, **data):
    return {
        "wire_version": 1,
        "turn_id": "turn",
        "seq": seq,
        "event": {"type": kind, **data},
    }


def encode(events):
    return "".join(
        f"id: turn:{item['seq']}\nevent: {item['event']['type']}\ndata: {json.dumps(item)}\n\n"
        for item in events
    )


def test_harness_is_opt_in_and_resume_optional():
    parser = build_parser()
    defaults = parser.parse_args(["chat"])
    assert defaults.harness is False
    assert defaults.resume is None
    selected = parser.parse_args(["chat", "--harness", "--resume", "--profile", "work"])
    assert selected.harness and selected.resume == "last" and selected.profile == "work"


def test_replay_deduplicates_and_gap_does_not_advance():
    state = TurnState("turn")
    assert state.apply(event(1, "turn.started"))
    assert not state.apply(event(1, "turn.started"))
    with pytest.raises(ReplayRequiredError):
        state.apply(event(3, "text.delta", part_id="answer", text="later"))
    assert state.cursor == 1
    state.apply(event(2, "text.delta", part_id="answer", text="Hello"))
    state.apply(event(3, "turn.terminal", outcome="completed"))
    assert state.text == "Hello"
    assert not state.apply(event(3, "turn.terminal", outcome="completed"))
    with pytest.raises(ValueError, match="after terminal"):
        state.apply(event(4, "usage"))


def test_reasoning_is_not_visible_answer_and_completion_requires_text():
    state = TurnState("turn")
    state.apply(event(1, "reasoning.delta", part_id="thought", text="secret thought"))
    assert not state.text
    with pytest.raises(ValueError, match="no visible answer"):
        state.apply(event(2, "turn.terminal", outcome="completed"))
    assert state.cursor == 1
    state.apply(event(2, "turn.terminal", outcome="cancelled"))


def test_redacted_reasoning_preserves_cursor_and_unknown_optional_event():
    state = TurnState("turn")
    state.apply(event(1, "reasoning.delta", part_id="thought", omitted=True))
    state.apply(event(2, "future.optional"))
    assert state.cursor == 2 and not state.reasoning


def test_sse_identity_comments_multiline_and_incomplete_frame():
    item = event(1, "turn.started")
    lines = [
        ": heartbeat",
        "id: turn:1",
        "event: turn.started",
        "data: " + json.dumps(item),
        "",
        "data: {incomplete",
    ]
    assert list(sse_events(lines)) == [item]
    with pytest.raises(ValueError, match="identity"):
        list(
            sse_events(
                ["id: other:1", "event: turn.started", "data: " + json.dumps(item), ""]
            )
        )


def test_pipe_output_keeps_markdown_and_hides_reasoning():
    state = TurnState("turn")
    output = io.StringIO()
    renderer = HarnessRenderer(state, stream=output)
    for item in [
        event(1, "reasoning.delta", part_id="r", text="private"),
        event(2, "text.delta", part_id="a", text="**Hello**\n\nWorld\x1b\x07"),
    ]:
        state.apply(item)
        renderer.update(item)
    renderer.finish()
    assert output.getvalue() == "**Hello**\n\nWorld\n"
    assert safe_markdown("a\tb\nc") == "a    b\nc"


def test_tools_and_multiple_model_parts_are_separate():
    state = TurnState("turn")
    for item in [
        event(1, "text.delta", part_id="a", text="Checking."),
        event(2, "tool.started", call_id="call", name="read_attachment"),
        event(3, "tool.result", call_id="call", outcome="completed"),
        event(4, "text.delta", part_id="b", text="Answer."),
    ]:
        state.apply(item)
    assert state.text == "Checking.\n\nAnswer."
    assert state.tools["call"]["status"] == "completed"


@pytest.mark.parametrize("mode", ["append", "regenerate"])
def test_turn_replays_after_connection_loss_without_duplicate_text(
    client_factory, monkeypatch, capsys, mode
):
    client = client_factory(
        SimpleNamespace(profile="test", think=False, response_timeout=5),
        "http://127.0.0.1:18800",
        "test",
    )
    client.endpoint = "http://owner"
    client.conversation_id = "conversation"
    client.branch_id = "branch"
    client.snapshot = {"branches": [{"id": "branch", "revision": 0}]}
    client.attachments = ["pending-attachment"]
    monkeypatch.setattr(client, "refresh", lambda: client.snapshot)
    monkeypatch.setattr(client, "connect", lambda: None)
    calls = []
    turn_id = None

    def handle(request):
        nonlocal turn_id
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": "test", "max_model_len": 12288}]}
            )
        calls.append(request)
        if request.method == "POST":
            turn_id = json.loads(request.content)["turn_id"]
            events = [event(1, "text.delta", part_id="answer", text="Hello")]
        else:
            assert request.headers["Last-Event-ID"] == f"{turn_id}:1"
            events = [
                event(1, "text.delta", part_id="answer", text="Hello"),
                event(2, "text.delta", part_id="answer", text=" world"),
                event(3, "turn.terminal", outcome="completed"),
            ]
        for item in events:
            item["turn_id"] = turn_id
        encoded = encode(events).replace("id: turn:", f"id: {turn_id}:")
        return httpx.Response(200, text=encoded)

    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handle))
    client.turn("Hi", mode=mode)
    assert client.last_state.text == "Hello world"
    assert len(calls) == 2
    assert capsys.readouterr().out.count("Hello") == 1
    sent = json.loads(calls[0].content)
    assert sent["temperature"] == 0.7
    assert sent["context_tokens"] == 12288
    assert sent["attachment_ids"] == (
        ["pending-attachment"] if mode == "append" else []
    )
    assert client.attachments == ([] if mode == "append" else ["pending-attachment"])


def test_pty_stop_continue_regenerate_and_resume(tmp_path):
    """Exercise the real CLI and owner against a deterministic SSE model."""
    import contextlib
    import os
    import signal
    import sys
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    pexpect = pytest.importorskip("pexpect")
    received_prompts = []

    class ModelHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            last = next(
                (m for m in reversed(body["messages"]) if m["role"] == "user"), {}
            )
            content = str(last.get("content", ""))
            # The runtime adds wire-only clock lines. Keep the user's
            # remaining text exact so these checks still catch lost keystrokes.
            if content.startswith("[Local time: "):
                clock, separator, remainder = content.partition("\n")
                if separator and clock.endswith("]"):
                    content = remainder
            remainder, separator, clock = content.rpartition(
                "\nAnswering now: [Local time: "
            )
            if separator and "\n" not in clock and clock.endswith("]"):
                content = remainder
            received_prompts.append(content)
            slow = "slow" in content
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for word in (
                    ["Working "] * 100 if slow else ["Hello ", "from ", "the fixture."]
                ):
                    chunk = {
                        "id": "completion",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "fixture",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": word},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.04 if not slow else 0.1)
                chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                self.wfile.write(
                    f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    children = []
    owner_pids = set()
    env = {
        **os.environ,
        "RAPID_MLX_CHAT_HOME": str(tmp_path / "profiles"),
        "RAPID_MLX_CHAT_IDLE_SECONDS": "0.3",
        "NO_COLOR": "1",
        "RAPID_MLX_SKIP_UPDATE_CHECK": "1",
    }
    env.pop("RAPID_MLX_API_KEY", None)
    script = "from types import SimpleNamespace; from rapid_mlx.chat_client import run_chat; import sys; run_chat(SimpleNamespace(profile='pty',think=False,response_timeout=10,resume=sys.argv[2] or None),sys.argv[1],'fixture')"

    def start(resume=""):
        child = pexpect.spawn(
            sys.executable,
            ["-c", script, f"http://127.0.0.1:{server.server_port}", resume],
            env=env,
            encoding="utf-8",
            timeout=90,
        )
        children.append(child)
        child.expect("› ")
        discovery = tmp_path / "profiles" / "pty" / "run" / "discovery.json"
        owner_pids.add(json.loads(discovery.read_text())["pid"])
        return child

    try:
        child = start()
        child.sendline("hello")
        child.expect("Hello from the fixture.")
        child.expect("› ")
        child.sendline("slow please")
        child.expect("Working")
        child.sendcontrol("c")
        child.expect("Stopped.")
        child.expect("› ")
        child.sendline("continue")
        child.expect("Hello from the fixture.")
        child.expect("› ")
        child.sendline("/regen")
        child.expect("Hello from the fixture.")
        child.expect("› ")
        child.sendline("/edit edited prompt")
        child.expect("Hello from the fixture.")
        child.expect("› ")
        attachment = tmp_path / "note.txt"
        attachment.write_text("The test document says hello.")
        child.sendline(f"/attach {attachment}")
        child.expect("Attached note.txt")
        child.expect("› ")
        child.sendline("Read my attachment")
        child.expect("Hello from the fixture.")
        child.expect("› ")
        # Every command hands stdin back to the same persistent reader. Check
        # admission out of band while the PTY master deliberately does not read.
        for index, command in enumerate(
            [
                "/help",
                "/?",
                "/history",
                "/history 1",
                "/branch",
                "/branch 1",
                "/think on",
                "/think off",
                "/show-thinking",
                "/show-thinking",
                "/export",
                "/save",
                "/unknown",
                "/new",
                "/reset",
                "/clear",
            ]
        ):
            child.sendline(command)
            child.expect(r"(?:Thinking )?› ")
            prompt = f"命令之后的输入{index}"
            child.sendline(prompt)
            deadline = time.monotonic() + 30
            while prompt not in received_prompts and time.monotonic() < deadline:
                time.sleep(0.02)
            assert prompt in received_prompts, command
            child.expect("Hello from the fixture.")
            child.expect(r"(?:Thinking )?› ")
        child.sendline("/edit")
        child.expect("Edited message> ")
        child.sendline("编辑子提示也不能丢失")
        deadline = time.monotonic() + 30
        while (
            "编辑子提示也不能丢失" not in received_prompts
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert "编辑子提示也不能丢失" in received_prompts
        child.expect("Hello from the fixture.")
        child.expect("› ")
        child.sendline("/quit")
        child.expect(pexpect.EOF)
        child.close()
        assert child.exitstatus == 0
        time.sleep(0.8)
        resumed = start("last")
        resumed.sendline("hello again")
        resumed.expect("Hello from the fixture.")
        resumed.expect("› ")
        resumed.sendline("slow please")
        resumed.expect("Working")
        discovery = tmp_path / "profiles" / "pty" / "run" / "discovery.json"
        os.kill(json.loads(discovery.read_text())["pid"], signal.SIGKILL)
        resumed.expect("The response could not finish")
        assert "generation_failed" not in resumed.before
        resumed.expect("› ")
        owner_pids.add(json.loads(discovery.read_text())["pid"])
        resumed.sendline("continue after restart")
        resumed.expect("Hello from the fixture.")
        resumed.expect("› ")
        resumed.sendline("/quit")
        resumed.expect(pexpect.EOF)
        resumed.close()
        assert resumed.exitstatus == 0
    finally:
        for child in children:
            if child.isalive():
                child.terminate(force=True)
        for pid in owner_pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_no_color_terminal_still_shows_progress(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    output = Terminal()
    state = TurnState("turn")
    renderer = HarnessRenderer(state, stream=output)
    delta = event(1, "text.delta", part_id="answer", text="Answer")
    state.apply(delta)
    renderer.update(delta)
    renderer.finish()
    assert output.getvalue() == "…\nAnswer\n"


def test_tool_preamble_cannot_satisfy_final_answer():
    state = TurnState("turn")
    state.apply(event(1, "text.delta", part_id="preamble", text="Checking your file."))
    state.apply(event(2, "tool.started", call_id="call", name="read_attachment"))
    state.apply(event(3, "tool.result", call_id="call", outcome="success"))
    with pytest.raises(ValueError, match="no visible answer"):
        state.apply(event(4, "turn.terminal", outcome="completed"))
    state.apply(event(4, "text.delta", part_id="answer", text="The file says hello."))
    state.apply(event(5, "turn.terminal", outcome="completed"))
    assert state.final_text == "The file says hello."


def test_history_displays_title_local_time_and_message_count(
    client_factory, monkeypatch, capsys
):
    import os
    import time

    from rapid_mlx.chat_client import history_time

    old_timezone = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/Los_Angeles"
        time.tzset()
        assert history_time("2026-10-26T04:42:55+00:00") == "10-25 21:42"
        client = client_factory()
        monkeypatch.setattr(
            client,
            "conversations",
            lambda: [
                {
                    "id": "conversation-id",
                    "title": "New chat",
                    "display_title": "介绍一下你自己",
                    "updated_at": "2026-10-26T04:42:55+00:00",
                    "message_count": 6,
                }
            ],
        )
        assert client.command("/history")
        output = capsys.readouterr().out
        assert "介绍一下你自己  10-25 21:42  6 messages" in output
        assert "New chat" not in output and "+00:00" not in output
    finally:
        if old_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_timezone
        time.tzset()


def test_startup_and_new_have_one_status_line_each(client_factory, monkeypatch, capsys):
    client = client_factory()
    monkeypatch.setattr(client, "connect", lambda: None)
    monkeypatch.setattr(
        client,
        "request",
        lambda *args, **kwargs: {"id": "conversation-123", "branch_id": "branch"},
    )
    monkeypatch.setattr(client, "refresh", lambda: {})
    commands = iter(["/new", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))
    try:
        client.run()
        assert capsys.readouterr().out == "New chat · Type / for commands\n" * 2
    finally:
        client.close()


@pytest.mark.parametrize(
    "choice", ["last", "1", "12345678", "12345678-abcd-1234-abcd-123456789abc"]
)
def test_resume_status_uses_safe_derived_title_and_preserves_id_selection(
    client_factory, choice, monkeypatch, capsys
):
    client = client_factory()
    conversation_id = "12345678-abcd-1234-abcd-123456789abc"
    monkeypatch.setattr(
        client,
        "conversations",
        lambda: [
            {
                "id": conversation_id,
                "title": "New chat",
                "display_title": "Ocean study\n[bold]\x07",
            }
        ],
    )
    monkeypatch.setattr(client, "refresh", lambda: {})
    assert client.select_conversation(choice)
    assert client.conversation_id == conversation_id
    assert (
        capsys.readouterr().out == "Resumed: Ocean study[bold] · Type / for commands\n"
    )


@pytest.mark.parametrize(
    "encoding, marker",
    [("utf-8", "›"), ("UTF8", "›"), ("ascii", ">"), ("latin-1", ">"), (None, ">")],
)
def test_thinking_labels_and_prompt_encoding(
    client_factory, encoding, marker, monkeypatch
):
    from rapid_mlx.chat_client import HELP

    class Output(io.StringIO):
        pass

    Output.encoding = encoding
    output = Output()
    monkeypatch.setattr("sys.stdout", output)
    client = client_factory()
    assert client.prompt_text() == f"{marker} "
    client.command("/think on")
    assert client.thinking and client.prompt_text() == f"Thinking {marker} "
    client.command("/show-thinking")
    client.command("/show-thinking")
    client.command("/think off")
    assert not client.thinking and client.prompt_text() == f"{marker} "
    assert output.getvalue() == (
        "Thinking on.\nThinking expanded.\nThinking collapsed.\nThinking off.\n"
    )
    assert "Turn thinking on or off" in HELP
    assert "Reasoning" not in HELP and "Fast" not in HELP


def test_live_tools_progress_sources_and_replay_are_quiet_until_needed():
    state = TurnState("turn")
    output = io.StringIO()
    renderer = HarnessRenderer(state, stream=output)
    items = [
        event(
            1,
            "tool.started",
            call_id="search",
            name="web_search",
            args={"query": "Apple news\x1b\x07"},
        ),
        event(
            2,
            "source",
            source_id="1",
            call_id="search",
            title="Apple Newsroom",
            url="https://www.apple.com/newsroom/",
        ),
        event(3, "tool.result", call_id="search", outcome="success"),
        event(4, "text.delta", part_id="answer", text="The latest news [1]."),
        event(5, "turn.terminal", outcome="completed"),
    ]
    for item in items:
        assert state.apply(item)
        renderer.update(item)
        assert not state.apply(item)
    assert "https://" not in output.getvalue()
    renderer.finish()
    renderer.finish()
    text = output.getvalue()
    assert text.count("Searching the web: Apple news…") == 1
    assert text.count("[1] Apple Newsroom — https://www.apple.com/newsroom/") == 1
    assert text.index("The latest news") < text.index("https://")
    assert "\x1b" not in text and "\x07" not in text
    assert state.tools["search"]["args"] == {"query": "Apple news\x1b\x07"}


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/article?private=query", "Reading example.com…"),
        ("https://user:password@example.com/", "Reading example.com…"),
        ("https://[invalid/", "Reading web page…"),
        ("https://example.com/\x1b\x07", "Reading example.com…"),
    ],
)
def test_fetch_progress_shows_only_a_safe_hostname(url, expected):
    assert tool_progress({"name": "web_fetch", "args": {"url": url}}) == expected


def test_source_links_export_and_resumed_history(
    client_factory, monkeypatch, capsys, tmp_path
):
    client = client_factory()
    client.snapshot = {
        "messages": [
            {
                "role": "assistant",
                "parts": [{"type": "text", "text": "News [1]."}],
                "sources": [
                    {
                        "source_id": "1",
                        "title": "Apple [news]",
                        "url": "https://example.com/latest?q=a>b",
                    },
                    {"source_id": "2", "title": "Unsafe", "url": "javascript:alert(1)"},
                    {
                        "source_id": "3",
                        "title": "Control",
                        "url": "https://example.com/\x1b",
                    },
                ],
            }
        ]
    }
    monkeypatch.setattr(client, "refresh", lambda: client.snapshot)
    client.show_history()
    history = capsys.readouterr().out
    assert "[1] Apple [news] — https://example.com/latest?q=a>b" in history
    path = tmp_path / "export.md"
    assert client.command(f"/export {path}")
    exported = path.read_text()
    assert "\\[1\\] Apple \\[news\\] — <https://example.com/latest?q=a%3Eb>" in exported
    assert "Unsafe" not in exported and "Control" not in exported
    assert exported.index("News [1].") < exported.index("https://")


@pytest.mark.parametrize("explicit,expected", [(None, 32768), (512, 512)])
@pytest.mark.parametrize("thinking", [False, True])
def test_harness_output_budget_default_and_override(
    client_factory, monkeypatch, explicit, expected, thinking
):
    client = client_factory(
        SimpleNamespace(max_tokens=explicit, think=thinking),
        "http://localhost:18800",
        "fixture",
    )
    monkeypatch.setattr(client, "refresh", lambda: {})
    monkeypatch.setattr(client, "branch", lambda: {"revision": 0})
    monkeypatch.setattr(client, "context_tokens", lambda: 32768)
    assert client.context_request()["max_tokens"] == expected


@pytest.mark.parametrize("succeed", [True, False])
def test_cancel_transport_failure_retries_or_reports(
    client_factory, monkeypatch, capsys, succeed
):
    client = client_factory()
    monkeypatch.setattr(client, "context_request", lambda *a, **k: {"turn_id": "turn"})
    monkeypatch.setattr(client, "refresh", lambda: None)
    monkeypatch.setattr(
        client.client,
        "stream",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    reconnects = []
    monkeypatch.setattr(client, "connect", lambda: reconnects.append(True))
    calls = []

    def cancel(*args, **kwargs):
        calls.append(kwargs)
        if not succeed or len(calls) < 3:
            raise httpx.ConnectError("lost")
        client.last_state.terminal = {"outcome": "cancelled"}

    monkeypatch.setattr(client, "request", cancel)
    client.turn("hello")
    assert len(calls) == 3 and len(reconnects) == 2
    assert all(call["idempotent"] for call in calls)
    assert ("Couldn't confirm the stop" in capsys.readouterr().out) is not succeed


def test_attachment_displays_safe_error(client_factory, monkeypatch, tmp_path):
    client = client_factory()
    path = tmp_path / "doc.txt"
    path.write_text("hello")
    response = httpx.Response(
        504,
        json={"detail": {"code": "timeout", "message": "File operation timed out."}},
        request=httpx.Request("POST", "http://localhost"),
    )
    monkeypatch.setattr(client, "request", lambda *a, **k: response.raise_for_status())
    with pytest.raises(ValueError, match="File operation timed out"):
        client.command(f"/attach {path}")


def test_cancel_reconnect_can_fail_without_exiting_client(
    client_factory, monkeypatch, capsys
):
    client = client_factory()
    monkeypatch.setattr(client, "context_request", lambda *a, **k: {"turn_id": "turn"})
    monkeypatch.setattr(
        client.client,
        "stream",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        client,
        "request",
        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("lost")),
    )
    monkeypatch.setattr(
        client, "connect", lambda: (_ for _ in ()).throw(httpx.ConnectError("lost"))
    )
    client.turn("hello")
    assert "Couldn't confirm the stop" in capsys.readouterr().out


@pytest.mark.parametrize("body", [b"invalid", b'{"detail": {}}'])
def test_attachment_error_without_safe_message_preserves_http_error(
    client_factory, monkeypatch, tmp_path, body
):
    client = client_factory()
    path = tmp_path / "doc.txt"
    path.write_text("hello")
    response = httpx.Response(
        500, content=body, request=httpx.Request("POST", "http://localhost")
    )
    monkeypatch.setattr(client, "request", lambda *a, **k: response.raise_for_status())
    with pytest.raises(httpx.HTTPStatusError):
        client.command(f"/attach {path}")


def test_turn_keeps_replaying_after_keepalives_and_timeouts(
    client_factory, monkeypatch
):
    client = client_factory()
    client.endpoint = "http://owner"
    monkeypatch.setattr(client, "context_request", lambda *a, **k: {"turn_id": "turn"})
    monkeypatch.setattr(client, "connect", lambda: None)
    monkeypatch.setattr(client, "refresh", lambda: None)
    monkeypatch.setattr("rapid_mlx.chat_client.time.sleep", lambda _: None)
    calls = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b": keepalive\n\n"
            raise httpx.ReadTimeout("slow model")

    def handle(request):
        calls.append(request)
        if len(calls) <= 3:
            return httpx.Response(200, stream=Stream())
        return httpx.Response(
            200,
            text=encode(
                [
                    event(1, "text.delta", part_id="a", text="done"),
                    event(2, "turn.terminal", outcome="completed"),
                ]
            ),
        )

    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handle))
    client.turn("hello")
    assert client.last_state.terminal["outcome"] == "completed"
    assert [r.method for r in calls] == ["POST"] + ["GET"] * 3
    assert all(r.headers["Last-Event-ID"] == "turn:0" for r in calls[1:])


def test_turn_gives_up_after_keepalive_only_streams(
    client_factory, monkeypatch, capsys
):
    client = client_factory()
    client.endpoint = "http://owner"
    monkeypatch.setattr(client, "context_request", lambda *a, **k: {"turn_id": "turn"})
    monkeypatch.setattr(client, "connect", lambda: None)
    monkeypatch.setattr(client, "refresh", lambda: None)
    calls = []

    def handle(request):
        calls.append(request)
        assert len(calls) <= 4, "keepalive-only streams must not reconnect indefinitely"
        return httpx.Response(200, text=": keepalive\n\n")

    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handle))
    client.turn("hello")
    assert [r.method for r in calls] == ["POST"] + ["GET"] * 3
    assert client.last_state.terminal is None
    assert "Connection interrupted" in capsys.readouterr().out


def test_cancel_before_first_event_replays_without_resubmission(
    client_factory, monkeypatch
):
    client = client_factory()
    client.endpoint = "http://owner"
    monkeypatch.setattr(client, "context_request", lambda *a, **k: {"turn_id": "turn"})
    monkeypatch.setattr(client, "refresh", lambda: None)
    calls = []

    class Interrupt(httpx.SyncByteStream):
        def __iter__(self):
            raise KeyboardInterrupt
            yield b""  # pragma: no cover

    def handle(request):
        calls.append(request)
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={})
        if request.method == "POST":
            assert len(calls) == 1, "turn was submitted twice"
            return httpx.Response(200, stream=Interrupt())
        return httpx.Response(
            200, text=encode([event(1, "turn.terminal", outcome="cancelled")])
        )

    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handle))
    client.turn("hello")
    assert [r.method for r in calls] == ["POST", "POST", "GET"]
    assert calls[-1].headers["Last-Event-ID"] == "turn:0"
    assert client.last_state.terminal["outcome"] == "cancelled"


@pytest.mark.parametrize("accepted", [False, True])
def test_permission_transport_error_replays_request(
    client_factory, monkeypatch, accepted
):
    client = client_factory()
    client.endpoint = "http://owner"
    monkeypatch.setattr(client, "context_request", lambda *a, **k: {"turn_id": "turn"})
    monkeypatch.setattr(client, "connect", lambda: None)
    monkeypatch.setattr(client, "refresh", lambda: None)
    monkeypatch.setattr("rapid_mlx.chat_client.time.sleep", lambda _: None)
    prompts = []
    monkeypatch.setattr(
        client.input, "approval", lambda **kwargs: prompts.append(kwargs) or "y"
    )
    permission = event(
        1,
        "permission.request",
        op_id="op",
        revision=1,
        options=["once"],
        expires_at=9999999999,
        reason="test",
    )
    decisions = []

    def handle(request):
        if request.url.path.endswith("/operations/op"):
            return httpx.Response(
                200,
                json={
                    "state": "resolved" if accepted and decisions else "intent",
                    "revision": 1,
                },
            )
        if request.url.path.endswith("/permission.response"):
            decisions.append(True)
            if len(decisions) == 1:
                raise httpx.ReadError("decision response lost")
            return httpx.Response(200, json={})
        if request.method == "POST":
            return httpx.Response(200, text=encode([permission]))
        assert request.headers["Last-Event-ID"] == "turn:0"
        return httpx.Response(
            200,
            text=encode(
                [
                    permission,
                    event(2, "text.delta", part_id="a", text="done"),
                    event(3, "turn.terminal", outcome="completed"),
                ]
            ),
        )

    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handle))
    client.turn("hello")
    assert len(prompts) == (1 if accepted else 2)
    assert client.last_state.terminal["outcome"] == "completed"
