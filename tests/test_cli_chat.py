# SPDX-License-Identifier: Apache-2.0
"""Tests for `rapid-mlx chat` (interactive REPL command)."""

from __future__ import annotations

import io
import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from vllm_mlx import cli


def _sse(events: list[dict]) -> bytes:
    """Build an SSE byte stream from a list of OpenAI-format chunks."""
    out = []
    for ev in events:
        out.append(f"data: {json.dumps(ev)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def _delta(content: str | None) -> dict:
    return {"choices": [{"delta": {"content": content} if content else {}}]}


class _FakeChatHandler(BaseHTTPRequestHandler):
    """Minimal HTTP server that pretends to be /v1/chat/completions."""

    canned_response: list[dict] = []
    received_payloads: list[dict] = []

    def log_message(self, *_args, **_kwargs):  # silence stderr
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        self.received_payloads.append(json.loads(body))
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(_sse(self.canned_response))

    def do_GET(self):  # noqa: N802
        if self.path == "/health/ready":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()


@contextmanager
def _fake_server(canned: list[dict]):
    _FakeChatHandler.canned_response = canned
    _FakeChatHandler.received_payloads = []
    server = HTTPServer(("127.0.0.1", 0), _FakeChatHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield port, _FakeChatHandler.received_payloads
    finally:
        server.shutdown()
        server.server_close()


def test_chat_subcommand_registered_in_cli():
    """`rapid-mlx chat --help` exits 0 (subparser is wired)."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


def test_chat_no_model_defaults_to_qwen35_4b():
    """`rapid-mlx chat` (no model) routes chat_command with qwen3.5-4b.

    Goes through the real ``cli.main()`` so a parser-wiring regression
    (e.g. dropping ``nargs='?'`` or changing the default alias) fails the
    test. ``chat_command`` is patched to capture args before the REPL
    runs.
    """
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat"]),
        patch.object(cli, "chat_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    args = captured[0]
    # Either the alias name itself or the resolved HF repo path — either
    # signals the default plumbed through. The canonical alias is the one
    # we documented as the default; confirm via the round-trip name.
    assert (
        args.model == "qwen3.5-4b"
        or getattr(args, "_original_alias", None) == "qwen3.5-4b"
    )


def test_chat_with_alias_overrides_default():
    """`rapid-mlx chat <alias>` uses the user-supplied alias, not the default."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat", "smollm3-3b"]),
        patch.object(cli, "chat_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    args = captured[0]
    assert (
        args.model == "smollm3-3b"
        or getattr(args, "_original_alias", None) == "smollm3-3b"
    )


def test_stream_chat_response_concatenates_deltas():
    """`_stream_chat_response` streams chunks and returns concatenated content."""
    canned = [_delta("Hello"), _delta(", "), _delta("world!")]
    with _fake_server(canned) as (port, _payloads):
        buf = io.StringIO()
        with patch.object(sys, "stdout", buf):
            full = cli._stream_chat_response(
                f"http://127.0.0.1:{port}",
                {"model": "x", "messages": [], "stream": True},
                timeout_s=10,
            )
    assert full == "Hello, world!"
    assert buf.getvalue() == "Hello, world!"


def test_stream_chat_response_skips_empty_deltas():
    """Tool-only / role-only deltas (no content) are ignored."""
    canned = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        _delta("hi"),
        {"choices": [{"delta": {}}]},
    ]
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", io.StringIO()),
    ):
        full = cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
        )
    assert full == "hi"


def test_stream_chat_response_raises_on_http_error():
    """A non-200 response raises RuntimeError carrying the body."""

    class _ErrHandler(BaseHTTPRequestHandler):
        def log_message(self, *_a, **_kw):
            pass

        def do_POST(self):  # noqa: N802
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"boom")

    server = HTTPServer(("127.0.0.1", 0), _ErrHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(RuntimeError, match=r"HTTP 500"):
            cli._stream_chat_response(
                f"http://127.0.0.1:{port}",
                {"model": "x", "messages": [], "stream": True},
                timeout_s=5,
            )
    finally:
        server.shutdown()
        server.server_close()


def test_chat_command_repl_multi_turn(monkeypatch, capsys):
    """End-to-end: `chat --base-url ...` accumulates multi-turn history."""
    canned = [_delta("Hi there!")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["hello", "/reset", "again", "exit"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

        ns = type("Args", (), {})()
        ns.base_url = f"http://127.0.0.1:{port}"
        ns.port = None
        ns.system = None
        ns.think = (
            True  # request the server-default behavior — no enable_thinking field sent
        )
        ns.max_tokens = 50
        ns.temperature = 0.0
        ns.ready_timeout = 5
        ns.response_timeout = 5
        ns.model = "qwen3.5-4b"

        cli.chat_command(ns)

    # Two POSTs — one before /reset, one after — both should ask for the
    # latest user turn only on second request because /reset clears history.
    assert len(payloads) == 2
    # First request: history = [{"role":"user","content":"hello"}]
    assert payloads[0]["messages"] == [{"role": "user", "content": "hello"}]
    # After /reset and "again", history should NOT contain "hello".
    assert payloads[1]["messages"] == [{"role": "user", "content": "again"}]


def test_chat_command_system_prompt_prepended(monkeypatch):
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["q1", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = type("Args", (), {})()
        ns.base_url = f"http://127.0.0.1:{port}"
        ns.port = None
        ns.system = "be terse"
        ns.think = True  # server-default thinking behavior
        ns.max_tokens = 50
        ns.temperature = 0.0
        ns.ready_timeout = 5
        ns.response_timeout = 5
        ns.model = "qwen3.5-4b"
        cli.chat_command(ns)
    assert payloads[0]["messages"][0] == {"role": "system", "content": "be terse"}
    assert payloads[0]["messages"][1] == {"role": "user", "content": "q1"}


def test_chat_command_default_thinking_off_sends_enable_thinking_false(monkeypatch):
    """Chat REPL defaults to thinking OFF.

    Reasoning models like Qwen3.5 otherwise leak raw chain-of-thought into
    the user-visible REPL output, and on the default qwen3.5-4b model
    degenerate into infinite repetition until max-tokens — producing zero
    usable output for a brand-new user. Pinning the default here so a
    refactor doesn't silently restore the broken behavior shipped in 0.6.26.
    """
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = type("Args", (), {})()
        ns.base_url = f"http://127.0.0.1:{port}"
        ns.port = None
        ns.system = None
        ns.think = False  # default
        ns.max_tokens = 50
        ns.temperature = 0.0
        ns.ready_timeout = 5
        ns.response_timeout = 5
        ns.model = "qwen3.5-4b"
        cli.chat_command(ns)
    assert payloads[0].get("enable_thinking") is False
    # The unsupported nested form must NOT be present.
    assert "chat_template_kwargs" not in payloads[0]


def test_chat_command_explicit_think_omits_enable_thinking_field(monkeypatch):
    """``--think`` opts back into reasoning mode. We omit the
    ``enable_thinking`` field entirely so the server falls back to its
    own default (which is True on Qwen3-family templates)."""
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = type("Args", (), {})()
        ns.base_url = f"http://127.0.0.1:{port}"
        ns.port = None
        ns.system = None
        ns.think = True
        ns.max_tokens = 50
        ns.temperature = 0.0
        ns.ready_timeout = 5
        ns.response_timeout = 5
        ns.model = "qwen3.5-4b"
        cli.chat_command(ns)
    assert "enable_thinking" not in payloads[0]


def test_chat_subcommand_accepts_legacy_no_think_flag():
    """``--no-think`` is preserved via argparse BooleanOptionalAction so
    users with prior shell history don't break on upgrade. Behavior matches
    the new default (thinking off)."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat", "--no-think"]),
        patch.object(cli, "chat_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    assert captured[0].think is False


def test_chat_command_survives_connection_failure(monkeypatch, capsys):
    """If the server is unreachable, the REPL must keep running (not crash)
    and roll back the failed user turn so the next request is clean."""
    # Bind a port and immediately release it so connect() will fail.
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()

    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))

    ns = type("Args", (), {})()
    ns.base_url = f"http://127.0.0.1:{dead_port}"
    ns.port = None
    ns.system = None
    ns.think = False
    ns.max_tokens = 50
    ns.temperature = 0.0
    ns.ready_timeout = 1
    ns.response_timeout = 2
    ns.model = "qwen3.5-4b"
    # Should not raise — REPL prints "Request failed" and continues to "exit".
    cli.chat_command(ns)
    captured = capsys.readouterr()
    assert "Request failed" in captured.out


def test_chat_command_history_unchanged_on_http_error(monkeypatch):
    """A failed turn must not leave a user message in history (would corrupt
    the next turn). The user-side rollback is a contract we test explicitly."""

    class _ErrHandler(BaseHTTPRequestHandler):
        def log_message(self, *_a, **_kw):
            pass

        def do_POST(self):  # noqa: N802
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), _ErrHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # We can't easily inspect `messages` post-hoc since chat_command
        # holds it locally — but we can confirm the second turn is sent
        # WITHOUT the failed first turn in the history.
        # Wire two failing POSTs but check the second request body.
        recorded = []
        orig = _ErrHandler.do_POST

        def _capture(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            recorded.append(json.loads(body))
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")

        _ErrHandler.do_POST = _capture  # type: ignore[assignment]

        inputs = iter(["bad1", "bad2", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = type("Args", (), {})()
        ns.base_url = f"http://127.0.0.1:{port}"
        ns.port = None
        ns.system = None
        ns.think = False
        ns.max_tokens = 50
        ns.temperature = 0.0
        ns.ready_timeout = 5
        ns.response_timeout = 5
        ns.model = "qwen3.5-4b"
        cli.chat_command(ns)

        _ErrHandler.do_POST = orig  # type: ignore[assignment]
    finally:
        server.shutdown()
        server.server_close()

    # Both turns were sent — and the second turn must NOT carry the failed
    # first turn (rollback contract).
    assert len(recorded) == 2
    assert recorded[1]["messages"] == [{"role": "user", "content": "bad2"}]


def _ns_for_chat(port: int, **overrides) -> object:
    """Build a chat_command argparse namespace pointing at a fake server."""
    ns = type("Args", (), {})()
    ns.base_url = f"http://127.0.0.1:{port}"
    ns.port = None
    ns.system = None
    ns.think = False
    ns.max_tokens = 50
    ns.temperature = 0.0
    ns.ready_timeout = 5
    ns.response_timeout = 5
    ns.model = "qwen3.5-4b"
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_stream_chat_response_captures_usage_into_metrics():
    """When the server emits a usage chunk (stream_options.include_usage),
    `_stream_chat_response` populates the metrics dict so the chat REPL
    can print the token-speed line."""
    canned = [
        _delta("Hello"),
        _delta(", world!"),
        # Final usage-only chunk: empty choices, populated usage block.
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4}},
    ]
    metrics: dict = {}
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", io.StringIO()),
    ):
        full = cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
            metrics=metrics,
        )
    assert full == "Hello, world!"
    assert metrics["completion_tokens"] == 4
    assert metrics["prompt_tokens"] == 7


def test_chat_command_help_command_prints_help(monkeypatch, capsys):
    """`/help` lists the slash commands and exits to the prompt without
    sending anything to the server."""
    canned = [_delta("ok")]  # never sent — REPL exits before any POST
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["/help", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    for needle in ("/help", "/reset", "/model", "/save", "/exit"):
        assert needle in out, f"help output missing {needle!r}"
    assert payloads == [], "help must not send any chat completion request"


def test_chat_command_unknown_slash_command_warns(monkeypatch, capsys):
    """`/foo` produces a friendly hint and does NOT POST to the server."""
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["/madeup", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    assert "Unknown command" in out
    assert "/help" in out
    assert payloads == []


def test_chat_command_save_writes_markdown_file(monkeypatch, tmp_path, capsys):
    """`/save <path>` serialises history (sans system prompt) to markdown."""
    canned = [_delta("Hi there!")]
    out_path = tmp_path / "convo.md"
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["hello", f"/save {out_path}", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    body = out_path.read_text(encoding="utf-8")
    assert "# rapid-mlx chat" in body
    assert "## User" in body and "hello" in body
    assert "## Assistant" in body and "Hi there!" in body
    assert "Saved" in capsys.readouterr().out


def test_chat_command_save_without_arg_prints_usage(monkeypatch, capsys):
    """Bare `/save` should not crash — prints a Usage hint."""
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["/save", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert "Usage: /save" in capsys.readouterr().out


def test_chat_command_multiline_heredoc_collected_into_one_message(monkeypatch):
    """Triple-quote heredoc collects multiple input lines into a single
    user message. Critical for pasting code blocks."""
    canned = [_delta("noted")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(['"""', "line one", "line two", '"""', "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert len(payloads) == 1
    msg = payloads[0]["messages"][0]
    assert msg["role"] == "user"
    assert msg["content"] == "line one\nline two"


def test_chat_command_sends_stream_options_include_usage(monkeypatch):
    """Chat payload must request usage in the stream so the speed line
    can show real (not estimated) token counts."""
    canned = [_delta("hi"), {"choices": [], "usage": {"completion_tokens": 1}}]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert payloads[0].get("stream_options") == {"include_usage": True}


def test_chat_command_speed_line_uses_server_token_count(monkeypatch, capsys):
    """When the server reports usage, the speed line shows the real count
    (not an estimate prefixed with `~`)."""
    canned = [
        _delta("hello world"),
        {"choices": [], "usage": {"completion_tokens": 17}},
    ]
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    assert "17 tok" in out
    assert "tok/s" in out
    assert "~17" not in out


def test_stream_chat_response_renders_atx_headings(monkeypatch):
    """ATX headings (`# h1`..`###### h6`) at line start get a colored
    style applied across the heading line. We simulate a TTY so the
    state machine path runs."""

    class _Tty(io.StringIO):
        def isatty(self):
            return True

    canned = [
        _delta("# Big\n"),
        _delta("## Sub\n"),
        _delta("Body line with `code`.\n"),
        _delta("### Smaller\n"),
        _delta("Tail.\n"),
    ]
    out_buf = _Tty()
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", out_buf),
        patch.dict("os.environ", {}, clear=False),
    ):
        # Make sure NO_COLOR isn't set in the test env.
        os = __import__("os")
        os.environ.pop("NO_COLOR", None)
        full = cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
        )
    rendered = out_buf.getvalue()
    # Plain text content survived intact.
    assert full == "# Big\n## Sub\nBody line with `code`.\n### Smaller\nTail.\n"
    # Heading lines are wrapped in ANSI escapes.
    assert "\x1b[" in rendered, "expected ANSI escapes on a TTY render"
    assert "# Big" in rendered and "## Sub" in rendered
    # Plain body line did not pick up a heading style — only inline `code`
    # got the cyan single-backtick wrap.
    assert "Body line with " in rendered


def test_chat_command_heredoc_does_not_trigger_slash_dispatch(monkeypatch):
    """A heredoc body whose first line starts with `/save` (or any
    slash) must reach the model as a regular user message, not get
    silently swallowed by the slash-command dispatcher. Pasted markdown
    docs whose first line is a path (`/path/to/file.py`) was a real
    regression in round-1 review."""
    canned = [_delta("ack")]
    with _fake_server(canned) as (port, payloads):
        # Heredoc body opens with `/save`-looking text — must NOT be
        # dispatched as the /save slash command.
        inputs = iter(['"""', "/save broken.txt", "second line", '"""', "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert len(payloads) == 1, "heredoc body must reach the server as a chat turn"
    msg = payloads[0]["messages"][0]
    assert msg["role"] == "user"
    assert msg["content"] == "/save broken.txt\nsecond line"


def test_chat_command_save_refuses_to_overwrite(monkeypatch, tmp_path, capsys):
    """`/save` must NOT silently clobber an existing file — destructive
    and easily triggered by typing the same path twice."""
    canned = [_delta("ok")]
    target = tmp_path / "convo.md"
    target.write_text("PRE-EXISTING CONTENT")
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["hi", f"/save {target}", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert target.read_text() == "PRE-EXISTING CONTENT", "must not overwrite"
    assert "already exists" in capsys.readouterr().out


def test_chat_command_save_creates_parent_directories(monkeypatch, tmp_path):
    """`/save logs/2026/convo.md` should auto-create the parent dirs
    instead of failing with a confusing ENOENT."""
    canned = [_delta("ok")]
    nested = tmp_path / "logs" / "subdir" / "convo.md"
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["hi", f"/save {nested}", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert nested.exists()
    assert "## User" in nested.read_text(encoding="utf-8")


def test_stream_chat_response_no_false_positive_on_repeated_lists(monkeypatch):
    """Legitimate repetitive content (a list of zeros, a markdown table
    separator) used to trip the round-1 guard's `≤2 unique tokens in 30`
    rule. The new guard requires the SAME single token to repeat
    consecutively, so these must stream through cleanly."""
    canned = [_delta("[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]")]
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", io.StringIO()) as buf,
    ):
        full = cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
        )
    assert full == "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
    assert "repeating" not in buf.getvalue()


def test_stream_chat_response_aborts_on_repetition(monkeypatch):
    """The repetition guard must cut the stream when the model degenerates
    into the same token repeated 30+ times — otherwise the screen fills
    with garbage and the REPL feels broken. We model the real-world
    scenario by feeding the same word as 80 separate chunks (matching how
    a token-streaming server actually delivers degenerate output).
    """
    canned = [_delta("Barley ") for _ in range(80)]
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", io.StringIO()) as buf,
    ):
        full = cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
        )
    # Guard kicked in well before the 80th chunk.
    assert full.count("Barley") < 80
    assert "repeating" in buf.getvalue() or "repetition" in buf.getvalue()


def test_ensure_model_downloaded_calls_disk_check(monkeypatch):
    """`_ensure_model_downloaded` must gate on `_check_disk_space` so a
    user without room for a 20 GB model fails fast with a clear error
    instead of a 90 % partial download."""
    # Force the cache-miss path so we reach the download branch.
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache", lambda *_a, **_kw: None
    )
    # Force a non-existent model_name path.
    monkeypatch.setattr("os.path.exists", lambda _p: False)

    called: list = []

    def _fake_check(name, force=False):
        called.append(name)

    monkeypatch.setattr(cli, "_check_disk_space", _fake_check)
    # Make snapshot_download a no-op so we don't hit the network.
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *_a, **_kw: "/tmp/fake"
    )
    # Stub model_info too so we don't query the API.
    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda *_a, **_kw: type("I", (), {"siblings": []})(),
    )

    cli._ensure_model_downloaded("mlx-community/Fake-Model-1B")
    assert called == ["mlx-community/Fake-Model-1B"]
