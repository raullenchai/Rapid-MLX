# SPDX-License-Identifier: Apache-2.0
"""Tests for `rapid-mlx chat` (interactive REPL command)."""

from __future__ import annotations

import io
import json
import os
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


def test_has_short_pattern_dominating_suffix_unit():
    """Direct unit tests for the char-level repetition detector.

    The token-level counter (whitespace-split) misses degenerate output
    that has NO whitespace separator. This helper is the safety net.
    """
    # 1. The smoking-gun case: long suffix of one word, no separator.
    assert cli._has_short_pattern_dominating_suffix("Barley" * 200)
    # 2. Same word with a single-char separator also degenerate.
    assert cli._has_short_pattern_dominating_suffix("Barley " * 200)
    # 3. Single-char repetition.
    assert cli._has_short_pattern_dominating_suffix("x" * 800)
    # 4. Pattern partially through the window — still triggers once the
    #    suffix is long enough to dominate.
    prefix = "Here is a poem about hops, malt, and barley:\n\n"
    assert cli._has_short_pattern_dominating_suffix(prefix + "BarleyBarley" * 100)
    # 5. Long-cycle phrase loop (the "Roman Empire" bug found by the
    #    chat-bug-hunt agent). A ~190-char clause repeating 4+ times
    #    must trigger — the window is 600, so we need >2 reps to fill
    #    it. Earlier ``pattern_max_len=30`` was too tight; KMP now
    #    detects periods up to ``max_period=300``.
    long_phrase = (
        "saw the suicide of the last Republic in 44 BCE, "
        "witnessed the rise of the First Triumvirate and the Gallic "
        "Wars of the 1st century BCE, experienced the suicide of the "
        "last Republic in 44 BCE, "
    )
    # The agent observed ~6500 chars of looping; 5 reps (~960 chars)
    # comfortably exceeds the 600-char window.
    assert cli._has_short_pattern_dominating_suffix(long_phrase * 5)
    # 6. NOT triggered: short content (below window threshold).
    assert not cli._has_short_pattern_dominating_suffix("Barley" * 10)
    # 7. NOT triggered: diverse list content.
    diverse = ", ".join(str(i) for i in range(300))  # well > 600 chars
    assert not cli._has_short_pattern_dominating_suffix(diverse)
    # 8. NOT triggered: legit prose at length. (Note: simple repetitions
    #    of the same sentence DO count as periodic and would trigger —
    #    that's correct, since "same sentence 20 times" is itself
    #    degenerate output. We use truly varied prose here.)
    diverse_prose = (
        "Computers were built to free us from drudgery, yet we have "
        "made them slaves of distraction. The pixel does not care "
        "what wakes it; only the human does. To name a thing is to "
        "begin to own it; to share that name is to lose half. Great "
        "engineers prefer boring problems with consequential answers. "
        "Sometimes the simplest tool is the one that survives, not "
        "because it was clever, but because nobody could break it. "
        "Latency is rude; jitter is hostile; both deserve apology. "
        "If a feature ships without an off switch, the off switch "
        "ships next quarter as a P0. Most production incidents begin "
        "with a person who could not say no. The only fearless review "
        "is the one before the merge."
    )
    assert not cli._has_short_pattern_dominating_suffix(diverse_prose)
    # 9. NOT triggered: short identical-element list (15 zeros). This is
    #    the round-1 false-positive case — keep it safe.
    assert not cli._has_short_pattern_dominating_suffix(
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
    )
    # 10. NOT triggered: aperiodic content longer than the window. The
    #     KMP detector must not fire on text whose smallest period
    #     exceeds ``max_period`` even when the window is full. We build
    #     a 700-char string with deterministic but non-repeating content
    #     (LCG-style hex digits — irregular enough that no short cycle
    #     dominates).
    import hashlib

    aperiodic = "".join(hashlib.md5(str(i).encode()).hexdigest()[0] for i in range(700))
    assert len(aperiodic) >= 700
    assert not cli._has_short_pattern_dominating_suffix(aperiodic)
    # 11. NOT triggered: a deterministic 400-char block repeated only
    #     1.5x — period (~400) exceeds the 300-char ceiling, so the
    #     detector must let it pass. (Block built from sha256 hex
    #     truncated to a length that has no shorter internal period.)
    long_random = hashlib.sha256(b"seed").hexdigest()  # 64 hex
    long_random = (long_random * 7)[:400]  # 400-char deterministic block
    assert not cli._has_short_pattern_dominating_suffix(long_random + long_random[:200])
    # 12. Custom max_period override works: a 400-char repeating pattern
    #     should trigger when max_period is bumped above its length.
    pattern_400 = (hashlib.sha256(b"x").hexdigest() * 7)[:400]
    assert cli._has_short_pattern_dominating_suffix(
        pattern_400 * 3, window=600, max_period=450
    )
    # 13. ``period == n`` boundary: an aperiodic full-window string
    #     must NOT trigger even when the caller passes
    #     ``max_period >= window``. Without an explicit ``period < n``
    #     guard, KMP returns ``period == n`` and the comparison
    #     ``n <= max_period`` would fire a false positive. Defaults
    #     never hit this (max_period=300 < window=600), but the helper
    #     exposes both knobs so the contract must hold.
    assert not cli._has_short_pattern_dominating_suffix(
        aperiodic, window=300, max_period=300
    )


def test_chat_command_save_refuses_on_empty_conversation(monkeypatch, tmp_path, capsys):
    """``/save`` against a fresh chat (no turns yet) must not create a
    0-message file. Previously the empty file blocked subsequent saves
    to the same path, since exclusive-mode open refuses overwrite."""
    canned = [_delta("ack")]
    target = tmp_path / "early-save.md"
    with _fake_server(canned) as (port, _payloads):
        # Save BEFORE sending any chat turn.
        inputs = iter([f"/save {target}", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    assert not target.exists(), "/save on empty conversation must not create a file"
    assert "Nothing to save" in out, f"expected friendly empty-save hint; got {out!r}"


def test_stream_chat_response_aborts_on_no_whitespace_repetition(monkeypatch):
    """The new char-level guard must fire on the qwen3.5-4b regression
    where the model emits ``BarleyBarleyBarley...`` with NO whitespace
    separator. The whitespace-token guard cannot catch this — a single
    chunk of 6000 chars splits to one token whose count is 1.

    We feed a single delta carrying 1000 ``Barley`` repetitions. With
    only the old guard the whole thing would dump and the abort would
    never fire."""
    canned = [_delta("Barley" * 1000)]
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", io.StringIO()) as buf,
    ):
        full = cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
        )
    rendered = buf.getvalue()
    # Because the entire run is in one chunk and the char-guard runs
    # AFTER ``full +=``, the chunk has been fully appended before the
    # guard fires — so the abort message is what the user sees right
    # after the dump. ``full`` therefore equals the input but the abort
    # message MUST be present.
    assert "Barley" in full
    assert "repeating" in rendered or "repetition" in rendered, (
        "char-level guard must fire on no-whitespace repetition"
    )


def test_stream_chat_response_token_guard_wins_when_both_eligible(monkeypatch):
    """When a single chunk satisfies BOTH the token-level guard
    (whitespace-separated repetition) AND the char-level guard
    (KMP periodicity), the token-level guard must fire first — the
    char-level guard is a fallback, not a duplicate firing path.

    Two independent assertions pin the contract so a future refactor
    that flips precedence (or makes the char guard slice mid-chunk)
    cannot accidentally pass:

    1. Behavioral: ``"Barley " * 200`` produces a far-shorter ``full``
       than the full 1400-char chunk that the post-emit char guard
       would yield.
    2. Spy: assert ``_has_short_pattern_dominating_suffix`` is NOT
       invoked once the token guard has already set
       ``repetition_aborted`` (the call site short-circuits on the
       ``not repetition_aborted`` precondition).
    """
    canned = [_delta("Barley " * 200)]
    char_guard_calls: list = []
    real_helper = cli._has_short_pattern_dominating_suffix

    def _spy(*args, **kwargs):
        char_guard_calls.append(args)
        return real_helper(*args, **kwargs)

    monkeypatch.setattr(cli, "_has_short_pattern_dominating_suffix", _spy)
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", io.StringIO()) as buf,
    ):
        full = cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
        )
    # (1) Behavioral: token-level guard wins → mid-chunk slice →
    # ``full`` should be well under the full 200-rep dump (1400 chars).
    # Allow some slack since REPEAT_LIMIT controls the exact cutoff.
    assert "Barley" in full
    assert len(full) < 700, (
        f"token-level guard must slice mid-chunk; got {len(full)} chars "
        "— suspect char-level guard fired on the full chunk"
    )
    assert "repeating" in buf.getvalue() or "repetition" in buf.getvalue()
    # (2) Spy: char-level guard must not run once token guard has
    # already aborted. Single-chunk streams: token guard fires inside
    # the chunk and then ``repetition_aborted`` short-circuits the
    # ``if not repetition_aborted and _has_short_pattern...`` check.
    assert char_guard_calls == [], (
        "char-level guard must not be invoked once token guard has aborted; "
        f"got {len(char_guard_calls)} calls"
    )


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


def test_chat_command_heredoc_preserves_indentation_and_blank_lines(monkeypatch):
    """Heredoc must preserve leading whitespace (Python indentation) and
    trailing blank lines verbatim — calling ``.strip()`` corrupts exactly
    the code-paste workflow the heredoc exists for."""
    canned = [_delta("ack")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(
            [
                '"""',
                "    def f():",
                "        return 1",
                "",  # blank line in the middle
                "    g()",
                "",  # trailing blank
                '"""',
                "exit",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert len(payloads) == 1
    msg = payloads[0]["messages"][0]
    expected = "    def f():\n        return 1\n\n    g()\n"
    assert msg["content"] == expected, (
        f"heredoc must preserve leading spaces + trailing blank, "
        f"got: {msg['content']!r}"
    )


def test_chat_command_save_uses_exclusive_mode_no_toctou(monkeypatch, tmp_path):
    """``/save`` must call ``open(path, 'x')`` so the existence check is
    atomic. An ``exists()``-then-``open('w')`` pair is TOCTOU-racy and a
    symlink to an arbitrary path can defeat ``os.path.exists`` on the
    first probe but still get clobbered on the open."""
    canned = [_delta("ok")]
    target = tmp_path / "x.md"
    seen_modes: list[str] = []
    real_open = open

    def _spy_open(path, mode="r", *args, **kwargs):
        if str(path) == str(target):
            seen_modes.append(mode)
        return real_open(path, mode, *args, **kwargs)

    with (
        _fake_server(canned) as (port, _payloads),
        patch("builtins.open", side_effect=_spy_open),
    ):
        inputs = iter(["hi", f"/save {target}", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert "x" in seen_modes, (
        f"expected /save to open with exclusive mode 'x' (got modes={seen_modes})"
    )


def test_chat_command_sigterm_handler_installed_before_spawn(monkeypatch):
    """SIGTERM handler MUST be installed before any ``_spawn_chat_server``
    call. A SIGTERM landing in the window between Popen() and
    signal.signal() uses Python's default handler (skips atexit) and
    orphans the spawned server."""
    import signal as _signal

    call_order: list[str] = []

    real_signal = _signal.signal

    def _spy_signal(signum, handler):
        if signum == _signal.SIGTERM:
            call_order.append("signal.SIGTERM")
        return real_signal(signum, handler)

    def _fake_spawn(*_a, **_kw):
        call_order.append("spawn")

        # Return a no-op proc plus a base_url that points at the fake
        # server so the rest of the REPL flow is unaffected.
        class _NoopProc:
            _rapid_mlx_log = None
            _rapid_mlx_log_path = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                pass

            def kill(self):
                pass

        return _NoopProc(), f"http://127.0.0.1:{port}"

    monkeypatch.setattr("signal.signal", _spy_signal)
    monkeypatch.setattr(cli, "_spawn_chat_server", _fake_spawn)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_wait_for_chat_server", lambda *_a, **_kw: None)

    canned = [_delta("ok")]
    with _fake_server(canned) as (fake_port, _payloads):
        port = fake_port
        inputs = iter(["hi", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        # Take the spawn path: clear base_url and port so chat_command
        # falls into the "spawn our own" branch.
        ns = _ns_for_chat(fake_port)
        ns.base_url = None
        ns.port = None
        cli.chat_command(ns)
    assert "signal.SIGTERM" in call_order, "SIGTERM handler was never installed"
    assert "spawn" in call_order, "spawn never happened"
    assert call_order.index("signal.SIGTERM") < call_order.index("spawn"), (
        f"SIGTERM handler installed AFTER spawn — orphan window. "
        f"call_order={call_order}"
    )


def test_chat_command_switch_model_rollback_on_wait_failure(monkeypatch, capsys):
    """When the candidate server fails the readiness wait, ``_switch_model``
    must (1) tear down the candidate proc, (2) keep the old proc as the
    active one, and (3) NOT clear chat history. Round-1 P0 regression test."""

    spawned: list[object] = []
    teardowns: list[object] = []

    class _FakeProc:
        def __init__(self, name):
            self.name = name
            self._rapid_mlx_log = None
            self._rapid_mlx_log_path = None
            self._terminated = False

        def poll(self):
            return None if not self._terminated else 0

        def terminate(self):
            self._terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._terminated = True

    def _fake_spawn(model, log_path, served_name=None, *, register_in=None):
        proc = _FakeProc(model)
        spawned.append(proc)
        if register_in is not None:
            register_in.append(proc)
        return proc, f"http://127.0.0.1:{port}"

    wait_calls = {"n": 0}

    def _fake_wait(base_url, proc, timeout_s):
        wait_calls["n"] += 1
        # First call (initial spawn) succeeds; second call (the /model
        # candidate) fails.
        if wait_calls["n"] >= 2:
            raise RuntimeError("simulated load failure")

    monkeypatch.setattr(cli, "_spawn_chat_server", _fake_spawn)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_wait_for_chat_server", _fake_wait)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_model",
        lambda alias: f"mlx-community/{alias}-resolved",
    )

    canned = [_delta("ack")]
    with _fake_server(canned) as (fake_port, payloads):
        port = fake_port
        inputs = iter(["first turn", "/model bogus", "second turn", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = _ns_for_chat(fake_port, model="qwen3.5-4b")
        ns.base_url = None
        ns.port = None
        cli.chat_command(ns)

    out = capsys.readouterr().out
    assert "Failed to start new server" in out, (
        "expected explicit rollback message on candidate failure"
    )
    assert "previous server still running" in out
    # Two user turns: history must NOT be cleared by a failed switch.
    assert len(payloads) == 2, (
        f"expected both turns to land on the original server "
        f"(history preserved); saw {len(payloads)}"
    )
    # Both turns sent the SAME conversation list; the second turn carries
    # the first as history.
    assert any(m["content"] == "first turn" for m in payloads[1]["messages"]), (
        "second-turn payload lost the first turn after the failed /model swap"
    )


def test_chat_command_slash_command_dispatch_uses_exact_match(
    monkeypatch, tmp_path, capsys
):
    """``/savefoo bar`` (typo) must NOT match ``/save``. ``startswith``
    matched the prefix and silently wrote a file from a typo. Exact-token
    parsing now treats the unknown command like any other slash typo and
    surfaces the help."""
    canned = [_delta("ack")]
    target = tmp_path / "should_not_appear.md"
    with _fake_server(canned) as (port, payloads):
        # Three inputs: a typo of /save, a typo of /model, then exit. None
        # of the typos must trigger the corresponding command.
        inputs = iter(
            [
                f"/savefoo {target}",
                "/modelfoo qwen3.5-4b",
                "exit",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    assert not target.exists(), "/savefoo must NOT trigger /save"
    assert "Unknown command: /savefoo" in out
    assert "Unknown command: /modelfoo" in out
    # Neither typo should have reached the server as a chat turn either —
    # they're slash-prefixed, so the /-handler swallows them with a hint.
    assert payloads == [], (
        f"slash typos must be swallowed by the dispatcher, "
        f"not forwarded to the server. payloads={payloads}"
    )


def test_chat_command_slash_command_accepts_tab_separator(
    monkeypatch, tmp_path, capsys
):
    """``/save\\tpath.md`` (tab as separator) must work like
    ``/save path.md``. Splitting on a literal space character would treat
    the whole tab-separated form as an unknown command — split() with no
    separator handles all whitespace."""
    canned = [_delta("ok")]
    target = tmp_path / "tabbed.md"
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["hi", f"/save\t{target}", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    assert target.exists(), (
        f"/save with tab separator should write the file, "
        f"got: {capsys.readouterr().out!r}"
    )
    assert "## User" in target.read_text(encoding="utf-8")


def test_stream_chat_response_repetition_truncates_at_cutoff_in_one_chunk(
    monkeypatch,
):
    """If a single SSE delta coalesces many repeated tokens (servers do
    batch under load), the abort message must land before the user sees
    the full 50-token wall — emit the prefix up to the cutoff, abort,
    then explain. Round-3 codex finding: previously the entire chunk
    was emitted before the rolling counter detected the run."""
    # 60 copies of the same single token in ONE delta — well past the
    # REPEAT_LIMIT=25 threshold.
    big_chunk = (" Barley" * 60).strip() + " "
    canned = [_delta(big_chunk)]
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", io.StringIO()) as buf,
    ):
        full = cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
        )
    rendered = buf.getvalue()
    # Only the prefix up to the cutoff should land; not all 60 copies.
    barley_count = full.count("Barley")
    assert barley_count < 60, (
        f"emitted full degenerate chunk before abort detected ({barley_count}/60)"
    )
    # And the abort hint did print.
    assert "repeating" in rendered or "repetition" in rendered, (
        "expected the repetition-abort hint to be visible"
    )


# ----------------------------------------------------------------------
# B1 — --think raises max_tokens default + length-cut warning
# ----------------------------------------------------------------------


def test_chat_think_bumps_max_tokens_default_to_4096():
    """``--think`` with no explicit ``--max-tokens`` raises the default
    from 2048 to 4096 so reasoning + final answer fit a small-model
    budget. Round-1 finding: ``chat qwen3.5-4b --think`` consumed the
    full 2048 budget with reasoning alone and emitted an empty answer
    with ``finish_reason='length'``."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat", "qwen3.5-4b", "--think"]),
        patch.object(cli, "chat_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    # argparse layer: default sentinel is None; chat_command resolves.
    # We test the resolved value by invoking chat_command's logic up to
    # the resolution point via _ns_for_chat-style namespace.
    assert captured[0].max_tokens is None, (
        "argparse must leave --max-tokens unresolved so chat_command can "
        "distinguish user-supplied 2048 from the unset sentinel"
    )
    assert captured[0].think is True


def test_chat_think_default_resolution_runtime(monkeypatch, capsys):
    """End-to-end: with ``--think`` and an unset ``--max-tokens``,
    chat_command resolves to 4096 AND prints a one-line note."""
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = _ns_for_chat(port, think=True, max_tokens=None)
        cli.chat_command(ns)
    # Resolved value lands in the payload.
    assert payloads[0]["max_tokens"] == 4096
    out = capsys.readouterr().out
    assert "raised --max-tokens to 4096" in out, (
        f"expected the --think bump notice in output; got: {out!r}"
    )


def test_chat_explicit_max_tokens_with_think_is_not_overridden(monkeypatch, capsys):
    """User-supplied ``--max-tokens=512`` with ``--think`` is honored —
    no silent bump, no banner. (Distinguishes "user passed 2048" from
    "default sentinel".)"""
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = _ns_for_chat(port, think=True, max_tokens=512)
        cli.chat_command(ns)
    assert payloads[0]["max_tokens"] == 512
    assert "raised --max-tokens" not in capsys.readouterr().out


def test_chat_warns_on_length_cut_empty_content(monkeypatch, capsys):
    """When the server emits ``finish_reason='length'`` AND no visible
    content streamed (reasoning consumed the budget), the REPL must
    warn so the user knows to bump --max-tokens."""
    # Reasoning-only stream, length cut on the last chunk.
    canned = [
        {"choices": [{"delta": {"reasoning_content": "thinking ..."}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"completion_tokens": 42}},
    ]
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    assert "reasoning consumed" in out, f"expected length+empty warning; got: {out!r}"


def test_chat_no_length_warning_when_content_present(monkeypatch, capsys):
    """Length cut WITH visible content (model finished mid-sentence) is
    a legitimate state and must NOT trigger the empty-answer warning."""
    canned = [
        _delta("here is half an answer"),
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"completion_tokens": 8}},
    ]
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    assert "reasoning consumed" not in out, (
        "length cut WITH content must not trigger the empty-answer warning"
    )


def test_stream_chat_response_captures_finish_reason_into_metrics():
    """``_stream_chat_response`` must record ``finish_reason`` in the
    metrics dict so the REPL can decide whether to warn on length-cut."""
    canned = [
        _delta("hi"),
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    ]
    metrics: dict = {}
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", io.StringIO()),
    ):
        cli._stream_chat_response(
            f"http://127.0.0.1:{port}",
            {"model": "x", "messages": [], "stream": True},
            timeout_s=10,
            metrics=metrics,
        )
    assert metrics.get("finish_reason") == "length"


# ----------------------------------------------------------------------
# B5 — port validator + pre-flight probe
# ----------------------------------------------------------------------


def test_chat_port_out_of_range_rejected_by_argparse(capsys):
    """``--port 99999`` must exit via argparse with a clear message,
    not drop into the REPL."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat", "--port", "99999"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    # argparse exits 2 on a type error.
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "port must be between 1 and 65535" in err


def test_chat_port_zero_rejected_by_argparse(capsys):
    """Port 0 (would mean "let kernel pick") is not a valid connect target."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat", "--port", "0"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    assert "port must be between 1 and 65535" in capsys.readouterr().err


def test_chat_port_nonnumeric_rejected_by_argparse(capsys):
    """Non-numeric ``--port`` value is rejected with a friendly error."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat", "--port", "abc"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    assert "port must be an integer" in capsys.readouterr().err


def test_chat_port_unbound_exits_with_friendly_error(capsys, monkeypatch):
    """Valid-range port with nothing listening must surface a one-line
    'no server reachable' error and exit, not drop into the REPL."""
    import socket as _socket

    s = _socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()

    # Make sure we never reach input().
    monkeypatch.setattr("builtins.input", lambda _p="": "exit")

    ns = type("Args", (), {})()
    ns.base_url = None
    ns.port = dead_port
    ns.system = None
    ns.think = False
    ns.max_tokens = 50
    ns.temperature = 0.0
    ns.ready_timeout = 1
    ns.response_timeout = 1
    ns.model = "qwen3.5-4b"
    with pytest.raises(SystemExit) as exc:
        cli.chat_command(ns)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "no rapid-mlx server reachable" in out
    assert f"127.0.0.1:{dead_port}" in out


# ----------------------------------------------------------------------
# D1 — `run` alias for chat
# ----------------------------------------------------------------------


def test_run_is_alias_for_chat(monkeypatch):
    """``rapid-mlx run <model>`` must route to ``chat_command`` with the
    same args as ``rapid-mlx chat <model>``."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "run", "qwen3.5-4b"]),
        patch.object(cli, "chat_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    ns = captured[0]
    # All chat-only flags should be present with their defaults.
    assert hasattr(ns, "think")
    assert hasattr(ns, "max_tokens")
    assert hasattr(ns, "port")
    assert hasattr(ns, "base_url")
    assert hasattr(ns, "system")


def test_run_alias_accepts_chat_flags():
    """Flags accepted by ``chat`` must be accepted by the ``run`` alias too."""
    captured: list = []
    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "run", "qwen3.5-4b", "--think", "--max-tokens", "1024"],
        ),
        patch.object(cli, "chat_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    assert captured[0].think is True
    assert captured[0].max_tokens == 1024


# ----------------------------------------------------------------------
# D3 — /bye and /? slash aliases
# ----------------------------------------------------------------------


def test_chat_command_bye_exits_like_exit(monkeypatch, capsys):
    """``/bye`` must terminate the REPL like ``/exit``."""
    canned = [_delta("hi")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["hello", "/bye", "this would crash if /bye didnt exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    # /bye stopped iteration BEFORE the 3rd input was consumed → the
    # iterator still has it. If /bye were ignored we'd hit StopIteration.
    # The first turn ran, the second turn was the /bye exit.
    assert len(payloads) == 1


def test_chat_command_question_mark_lists_help(monkeypatch, capsys):
    """``/?`` must print the help text (alias for ``/help``)."""
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["/?", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    # Same content as /help — pin a couple of canonical strings.
    assert "/help" in out
    assert "/exit" in out
    assert "/bye" in out
    assert payloads == [], "/? must not POST"


def test_chat_help_text_lists_bye_and_question_mark(monkeypatch, capsys):
    """The ``/help`` body must advertise both alias sets so a user
    skimming for "how do I quit" sees ``/bye`` and the equivalent
    ``/?`` lookup."""
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["/help", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    assert "/bye" in out
    assert "/?" in out


# ----------------------------------------------------------------------
# D4 — cross-alias --no-thinking (chat) and --no-think (serve)
# ----------------------------------------------------------------------


def test_chat_accepts_no_thinking_as_alias_for_no_think():
    """``chat --no-thinking`` (serve-style spelling) must land on the
    same ``think=False`` destination as ``chat --no-think``."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat", "--no-thinking"]),
        patch.object(cli, "chat_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    assert captured[0].think is False


def test_serve_accepts_no_think_as_alias_for_no_thinking():
    """``serve --no-think`` (chat-style spelling) must land on the same
    ``no_thinking=True`` destination as ``serve --no-thinking``."""
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", "qwen3.5-4b", "--no-think"]),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    assert len(captured) == 1
    assert captured[0].no_thinking is True


def test_chat_no_thinking_hidden_from_help():
    """The cross-alias is back-compat-only; it must NOT appear in
    ``chat --help`` (otherwise we double-document the same flag)."""
    import argparse as _argparse

    parser = _argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")
    # Re-create the chat parser via main() inspection — easier to run
    # the real CLI and capture --help output.
    import subprocess as _sp

    out = _sp.run(
        [sys.executable, "-m", "vllm_mlx.cli", "chat", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--no-thinking" not in out.stdout, (
        "hidden cross-alias must not appear in chat --help"
    )


# ----------------------------------------------------------------------
# D8 — first-launch-only banner for "agents codex" tip
# ----------------------------------------------------------------------


def test_seen_tips_marker_round_trip(tmp_path, monkeypatch):
    """``_has_seen_tip`` returns False before, True after ``_mark_tip_seen``."""
    monkeypatch.setenv("RAPID_MLX_CONFIG_HOME", str(tmp_path))
    assert cli._has_seen_tip("chat_intro_codex") is False
    cli._mark_tip_seen("chat_intro_codex")
    assert cli._has_seen_tip("chat_intro_codex") is True
    # Marker file landed in the override dir.
    assert (tmp_path / "seen-tips.json").exists()


def test_seen_tips_marker_survives_corrupt_file(tmp_path, monkeypatch):
    """A corrupt marker (parse error) must be treated as 'not seen' so
    the tip re-fires once, rather than being silently hidden forever."""
    monkeypatch.setenv("RAPID_MLX_CONFIG_HOME", str(tmp_path))
    (tmp_path / "seen-tips.json").write_text("not json {{")
    assert cli._has_seen_tip("chat_intro_codex") is False


def test_chat_banner_shown_on_first_launch_only(monkeypatch, capsys, tmp_path):
    """First chat launch shows the agents-codex tip; subsequent launches
    do NOT. The marker file under ``RAPID_MLX_CONFIG_HOME`` records the
    first-seen state."""
    monkeypatch.setenv("RAPID_MLX_CONFIG_HOME", str(tmp_path))

    # Force the TTY/NO_COLOR gate to think we're interactive (otherwise
    # the marker logic short-circuits to "skip everything").
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)

    canned = [_delta("ok")]
    # First launch.
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", _Tty()) as buf1,
    ):
        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
        first = buf1.getvalue()
    assert "agents codex" in first, (
        f"first launch should show the agents-codex tip; got {first!r}"
    )

    # Second launch — marker file now exists, banner should be suppressed.
    canned2 = [_delta("ok")]
    with (
        _fake_server(canned2) as (port2, _payloads),
        patch.object(sys, "stdout", _Tty()) as buf2,
    ):
        inputs2 = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs2))
        cli.chat_command(_ns_for_chat(port2))
        second = buf2.getvalue()
    assert "agents codex" not in second, (
        f"second launch must NOT re-show the banner; got {second!r}"
    )


def test_chat_banner_skipped_when_no_color_set(monkeypatch, capsys, tmp_path):
    """``NO_COLOR`` or non-TTY stdout: skip the marker logic AND the
    banner entirely so pipe/CI runs don't pollute the user's config."""
    monkeypatch.setenv("RAPID_MLX_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("NO_COLOR", "1")

    canned = [_delta("ok")]
    with _fake_server(canned) as (port, _payloads):
        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        cli.chat_command(_ns_for_chat(port))
    out = capsys.readouterr().out
    assert "agents codex" not in out, "NO_COLOR run should suppress the banner entirely"
    # And no marker file was written.
    assert not (tmp_path / "seen-tips.json").exists(), (
        "NO_COLOR run must not pollute the user's config dir"
    )


def test_chat_banner_write_failure_does_not_abort(monkeypatch, tmp_path, capsys):
    """If the marker dir is unwritable (read-only FS / permission
    denied), the tip should still print and chat must continue — best
    effort only."""
    monkeypatch.setenv("RAPID_MLX_CONFIG_HOME", str(tmp_path / "no_perm"))

    class _Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)

    # Mock os.makedirs to raise so the write path fails.
    real_makedirs = os.makedirs

    def _failing(*a, **kw):
        if str(tmp_path / "no_perm") in str(a[0]):
            raise PermissionError("read only fs")
        return real_makedirs(*a, **kw)

    monkeypatch.setattr("os.makedirs", _failing)

    canned = [_delta("ok")]
    with (
        _fake_server(canned) as (port, _payloads),
        patch.object(sys, "stdout", _Tty()) as buf,
    ):
        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        # Must NOT raise.
        cli.chat_command(_ns_for_chat(port))
    # Banner still printed.
    assert "agents codex" in buf.getvalue()


# ----------------------------------------------------------------------
# B3 — log file unlink policy
# ----------------------------------------------------------------------


def test_teardown_unlinks_only_empty_log_files(tmp_path, monkeypatch):
    """``_teardown_proc`` (closed-over inside ``chat_command``) must
    unlink a zero-byte log (no useful info) but PRESERVE a non-empty
    log (post-mortem breadcrumbs).

    Drive teardown via ``/model`` swap: the chat REPL spawns the
    initial server, then ``/model <other>`` replaces it — which calls
    ``_teardown_proc(old_proc)`` synchronously. That fires the unlink
    policy without needing process-level atexit to fire.
    """
    empty_log = tmp_path / "empty.log"
    full_log = tmp_path / "full.log"
    empty_log.write_text("")
    full_log.write_text("some real error trace\n")

    class _FakeProc:
        def __init__(self, log_path):
            self._rapid_mlx_log = open(log_path, "a")
            self._rapid_mlx_log_path = str(log_path)

        def poll(self):
            return 0  # already "exited"

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_wait_for_chat_server", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_model",
        lambda alias: f"mlx-community/{alias}-resolved",
    )

    # --- Case 1: empty log → unlinked on swap ---
    call_n = {"n": 0}

    def _spawn_empty(model, log_path, served_name=None, *, register_in=None):
        # First spawn: backed by empty_log (the one that should be
        # unlinked when swapped out). Second spawn: a noop replacement.
        call_n["n"] += 1
        if call_n["n"] == 1:
            p = _FakeProc(empty_log)
        else:
            tmp = tmp_path / f"new-{call_n['n']}.log"
            tmp.write_text("")
            p = _FakeProc(tmp)
        if register_in is not None:
            register_in.append(p)
        return p, f"http://127.0.0.1:{port_e}"

    monkeypatch.setattr(cli, "_spawn_chat_server", _spawn_empty)

    canned = [_delta("ok")]
    with _fake_server(canned) as (fake_port, _payloads):
        port_e = fake_port
        inputs = iter(["/model other-alias", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = _ns_for_chat(fake_port)
        ns.base_url = None
        ns.port = None
        cli.chat_command(ns)

    assert not empty_log.exists(), (
        "empty log should be unlinked when swapped out (no debugging value)"
    )

    # --- Case 2: full log → preserved on swap ---
    call_n2 = {"n": 0}

    def _spawn_full(model, log_path, served_name=None, *, register_in=None):
        call_n2["n"] += 1
        if call_n2["n"] == 1:
            p = _FakeProc(full_log)
        else:
            tmp = tmp_path / f"new2-{call_n2['n']}.log"
            tmp.write_text("")
            p = _FakeProc(tmp)
        if register_in is not None:
            register_in.append(p)
        return p, f"http://127.0.0.1:{port_f}"

    monkeypatch.setattr(cli, "_spawn_chat_server", _spawn_full)

    canned2 = [_delta("ok")]
    with _fake_server(canned2) as (fake_port2, _payloads):
        port_f = fake_port2
        inputs2 = iter(["/model other-alias", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs2))
        ns = _ns_for_chat(fake_port2)
        ns.base_url = None
        ns.port = None
        cli.chat_command(ns)

    assert full_log.exists(), "full log was unexpectedly removed"
    assert full_log.read_text() == "some real error trace\n"
