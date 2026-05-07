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


def test_chat_subcommand_requires_model():
    """`rapid-mlx chat` (no model) exits non-zero."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "chat"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code != 0


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
        ns.no_think = False
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
        ns.no_think = False
        ns.max_tokens = 50
        ns.temperature = 0.0
        ns.ready_timeout = 5
        ns.response_timeout = 5
        ns.model = "qwen3.5-4b"
        cli.chat_command(ns)
    assert payloads[0]["messages"][0] == {"role": "system", "content": "be terse"}
    assert payloads[0]["messages"][1] == {"role": "user", "content": "q1"}


def test_chat_command_no_think_passes_chat_template_kwargs(monkeypatch):
    canned = [_delta("ok")]
    with _fake_server(canned) as (port, payloads):
        inputs = iter(["q", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        ns = type("Args", (), {})()
        ns.base_url = f"http://127.0.0.1:{port}"
        ns.port = None
        ns.system = None
        ns.no_think = True
        ns.max_tokens = 50
        ns.temperature = 0.0
        ns.ready_timeout = 5
        ns.response_timeout = 5
        ns.model = "qwen3.5-4b"
        cli.chat_command(ns)
    assert payloads[0].get("chat_template_kwargs") == {"enable_thinking": False}


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
        ns.no_think = False
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
