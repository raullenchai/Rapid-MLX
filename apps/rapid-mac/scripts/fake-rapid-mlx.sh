#!/usr/bin/env bash
# fake-rapid-mlx.sh — a minimal stand-in for ``rapid-mlx`` so the
# Rapid.app chat smoke can run in seconds instead of paying the
# real model's ~60 s cold start.
#
# Used by:
#   - TestDriver chat smoke (RAPID_BIN=/path/to/fake-rapid-mlx.sh)
#   - Local UI iteration when the actual rapid-mlx isn't installed
#     or the user wants stable canned output to eyeball layout
#     against
#
# What it implements (just enough to pass ChatStreamClient + the
# health-check polling in ServerManager):
#
#   GET  /healthz             -> 200 {"ok": true}
#   POST /v1/chat/completions -> SSE stream of fake reasoning +
#                                content deltas, terminated by
#                                "data: [DONE]"
#
# CLI shape mirrors the real binary so the spawn-arg pipeline in
# ServerManager doesn't need to branch:
#
#   fake-rapid-mlx serve <alias> --host <h> --port <p>
#
# Anything else is silently accepted (so future flag additions on
# the real side don't break this).
set -euo pipefail

# We re-exec into Python because doing SSE + HTTP from bash is a
# losing proposition; Python ships on every macOS we target (14+).
# The Python program is embedded as a heredoc to keep the fake to
# a single file — no path resolution issues, no second artifact to
# install.
exec /usr/bin/env python3 - "$@" <<'PYEOF'
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


try:
    with open(os.path.join(os.environ.get("HOME", ""), ".rapid-golden-fake.json"), encoding="utf-8") as stream:
        FILE_CONFIG = json.load(stream)
except (OSError, ValueError):
    FILE_CONFIG = {}


def _setting(name, default=None):
    return os.environ.get(name, FILE_CONFIG.get(name, default))


def _parse_args(argv):
    """Match the real rapid-mlx CLI shape closely enough that
    ServerManager's spawn arguments work unmodified.

    We don't model every flag — only the ones the SwiftUI app
    actually sends today (``serve <alias> --host --port``).
    Unknown flags are silently ignored so a future addition on
    the real side (e.g. ``--reasoning-effort``) doesn't break
    the mock.
    """
    if len(argv) >= 2 and argv[1] == "--version":
        # ServerManager doesn't call this today, but Homebrew's
        # post-install hook on the real binary does — handy to
        # have for parity.
        print("fake-rapid-mlx 0.0.0")
        sys.exit(0)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("subcommand", nargs="?", default="serve")
    parser.add_argument("alias", nargs="?", default="fake-alias")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args, _unknown = parser.parse_known_args(argv[1:])
    return args


CONTENT_CHUNKS = [
    "Hello", " from", " the", " fake", " rapid-mlx", " mock.",
    " I", " return", " deterministic", " content", " so", " the",
    " smoke", " test", " has", " something", " to", " assert", " on.",
]

# Answer SHAPES, chosen by a marker in the user's message.
#
# The fake has no model, so it cannot vary answer QUALITY — a golden flow
# asserting that a poem is good would be theatre. What it can vary, and what
# the app genuinely does different work for, is the shape of what has to be
# rendered: a fenced code block gets highlighting and its own copy button, a
# table has to become a table rather than pipes, LaTeX has to typeset, CJK and
# emoji have to measure correctly, and a long answer has to scroll without
# losing the turns above it. Judging what a model actually SAYS to a literary
# or coding prompt belongs to the eval suites, against a real model.
#
# Chunked deliberately mid-token in places: a renderer that only works when a
# fence or a table row arrives whole is a renderer that breaks on a real
# stream.
RESPONSE_SHAPES = {
    "shape:code": [
        "Here is the function you asked for:\n\n",
        "```", "python", "\n",
        "def fib(n):\n", "    a, b = 0, 1\n",
        "    for _ in range(n):\n", "        a, b = b, a + b\n",
        "    return a\n",
        "```", "\n\n",
        "It runs in O(n) time and constant space.",
    ],
    "shape:table": [
        "| model | size | speed |\n",
        "| --- | --- | ---", " |\n",
        "| qwen3.5-9b | 5.2 GB | 74 tok/s |\n",
        "| llama-3.1-8b | 4.5 GB | 68 tok/s |\n",
        "\nBoth fit comfortably in 16 GB.",
    ],
    "shape:math": [
        "The Gaussian integral is\n\n",
        "$$\\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx = \\sqrt{\\pi}$$",
        "\n\nand inline it reads $e^{i\\pi} + 1 = 0$.",
    ],
    "shape:list": [
        "Three things, in order:\n\n",
        "1. First, ", "read the prompt.\n",
        "2. Second, ", "plan the answer.\n",
        "   - a nested point\n", "   - another one\n",
        "3. Third, ", "write it down.\n",
    ],
    "shape:unicode": [
        "中文排版测试:", "这是一段中文回答,", "用来检查换行和字宽。",
        " Emoji: ", "🎯", "🚀", "。",
        " Right-to-left: ", "مرحبا", ".",
    ],
    "shape:prose": [
        "The lighthouse keeper ", "kept two logbooks. ",
        "One recorded the weather, ", "the ships, ", "the hours of the lamp. ",
        "The other recorded ", "what he thought about ", "while he watched. ",
        "Only the first was ever read ", "by anyone else.",
    ],
}

# Long output is the same default text repeated, so the scroll/perf case does
# not need its own vocabulary to assert on.
RESPONSE_SHAPES["shape:long"] = CONTENT_CHUNKS * 30


def _shape_for(body):
    """The chunk list this request should stream.

    Reads the LAST user message, so a multi-turn conversation gets a different
    shape per turn rather than whatever the first turn asked for.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return CONTENT_CHUNKS * CONTENT_REPEAT
    text = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # OpenAI content-parts form.
                text = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
            break
    for marker, chunks in RESPONSE_SHAPES.items():
        if marker in text:
            return chunks
    return CONTENT_CHUNKS * CONTENT_REPEAT
REASONING_CHUNKS = [
    "Let", " me", " think", " about", " the", " prompt", "."
]
try:
    INTER_TOKEN_SLEEP_S = float(_setting("FAKE_INTER_TOKEN_SLEEP_S", "0.01"))
except ValueError:
    INTER_TOKEN_SLEEP_S = 0.01
try:
    CONTENT_REPEAT = max(1, int(_setting("FAKE_CONTENT_REPEAT", "1")))
except ValueError:
    CONTENT_REPEAT = 1


def _event(name, **fields):
    """Append machine-readable lifecycle evidence for GUI golden flows."""
    path = _setting("FAKE_EVENT_LOG")
    if not path:
        return
    record = {"event": name, "time": time.time(), **fields}
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _sse(data):
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def _delta(content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    choice = {"delta": delta, "finish_reason": finish}
    return {"choices": [choice]}


def _tool_call_delta(call_id):
    return {"choices": [{
        "delta": {"tool_calls": [{
            "index": 0,
            "id": call_id,
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"query": "golden tool loop evidence"}),
            },
        }]},
        "finish_reason": "tool_calls",
    }]}


class Handler(BaseHTTPRequestHandler):
    """Minimal OpenAI-shaped surface that's enough to satisfy
    ChatStreamClient + ServerManager's /healthz poll.

    Logs are silenced because BaseHTTPRequestHandler.log_message
    spams stderr on every request, which the parent's log tail
    surfaces as noise during normal smoke runs.
    """

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_GET(self):
        if self.path == "/healthz":
            self._json(200, {"ok": True})
            return
        if self.path == "/v1/models":
            # ModelPickerBar reads this on real rapid-mlx — return
            # one canned entry so the picker isn't empty.
            self._json(200, {
                "data": [
                    {"id": "fake-alias", "object": "model"}
                ]
            })
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not_found"})
            return
        # Decode only enough of the request to support both normal SSE chat
        # and the in-app loaded-model speed test (`stream: false`).
        length = int(self.headers.get("content-length", "0") or "0")
        body = {}
        if length:
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        definitions = body.get("tools") if isinstance(body.get("tools"), list) else []
        _event(
            "chat_request",
            roles=[m.get("role") for m in messages if isinstance(m, dict)],
            tools=[
                d.get("function", {}).get("name")
                for d in definitions
                if isinstance(d, dict) and isinstance(d.get("function"), dict)
            ],
            user_texts=[
                m.get("content") for m in messages
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
            ],
        )

        if body.get("stream") is False:
            max_tokens = int(body.get("max_tokens", 8) or 8)
            completion_tokens = min(max_tokens, 128)
            _event("benchmark_request", max_tokens=max_tokens)
            self._json(200, {
                "id": "fake-benchmark",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "measured output"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 12 + completion_tokens,
                },
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Deterministic runaway-model fixture. The app must execute only its
        # bounded budget and then issue one final request with no tools; that
        # request gets a useful synthesis rather than another tool call.
        last_user = next((
            m.get("content", "") for m in reversed(messages)
            if isinstance(m, dict) and m.get("role") == "user"
        ), "")
        if "shape:tool-loop" in last_user:
            tool_results = sum(
                1 for m in messages
                if isinstance(m, dict) and m.get("role") == "tool"
            )
            if definitions:
                call_id = f"golden_loop_{tool_results + 1}"
                self.wfile.write(_sse(_tool_call_delta(call_id)))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                _event("tool_loop_call", call_id=call_id)
                return
            synthesis = "Golden tool-loop synthesis from existing evidence."
            self.wfile.write(_sse(_delta(content=synthesis, finish="stop")))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            _event("tool_loop_synthesis", tool_results=tool_results)
            return

        # #896 crash-recovery harness: when FAKE_DIE_AFTER_CHUNKS
        # is set to a positive integer N, we abruptly os._exit(1)
        # after streaming N content deltas — simulating the real
        # rapid-mlx being SIGKILL'd mid-response. The TCP socket
        # closes with whatever's buffered already in flight, so
        # ChatStreamClient sees partial deltas followed by EOF
        # WITHOUT a [DONE] sentinel or a finish_reason chunk — the
        # exact failure mode v0.4.5 maps to ``.streamTruncated``.
        die_after_chunks_raw = _setting("FAKE_DIE_AFTER_CHUNKS", "")
        try:
            die_after_chunks = int(die_after_chunks_raw)
        except ValueError:
            die_after_chunks = 0
        die_once_state = _setting("FAKE_DIE_ONCE_STATE")
        if die_after_chunks > 0 and die_once_state:
            try:
                marker_fd = os.open(die_once_state, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(marker_fd)
            except FileExistsError:
                die_after_chunks = 0
        content_emitted = 0
        try:
            for r in REASONING_CHUNKS:
                self.wfile.write(_sse(_delta(reasoning=r)))
                self.wfile.flush()
                time.sleep(INTER_TOKEN_SLEEP_S)
            for c in _shape_for(body):
                self.wfile.write(_sse(_delta(content=c)))
                self.wfile.flush()
                content_emitted += 1
                if die_after_chunks > 0 and content_emitted >= die_after_chunks:
                    # Hard-kill the process — no atexit, no
                    # buffered flush, no SIGTERM grace window.
                    # Mirrors how ServerManager.terminationHandler
                    # is supposed to fire on a real crash. We use
                    # os._exit (not sys.exit) so the
                    # ThreadingHTTPServer doesn't get a chance to
                    # clean up gracefully.
                    sys.stderr.write(
                        f"fake-rapid-mlx: hard-exit after {content_emitted}"
                        f" chunk(s) (FAKE_DIE_AFTER_CHUNKS={die_after_chunks})\n"
                    )
                    sys.stderr.flush()
                    os._exit(137)  # 128 + SIGKILL — matches a real OOM kill
                time.sleep(INTER_TOKEN_SLEEP_S)
            # Server says "stop"; ChatStreamClient watches for the
            # finish_reason field and then for the [DONE] sentinel.
            self.wfile.write(_sse(_delta(finish="stop")))
            self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            _event("chat_finished", chunks=content_emitted)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-stream — fine, the smoke
            # cancelled the stream on its own.
            _event("chat_cancelled", chunks=content_emitted)
            return

    def _json(self, status, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


FAKE_REPO = "fake-org/fake-repo"


def _emit_catalog(subcommand, alias):
    """Print the canned output for a NON-``serve`` subcommand.

    ``ModelCatalog`` shells out to ``models`` / ``ls`` / ``info`` to
    populate the picker, and every one of those is a print-and-exit
    command on the real binary. The formats below mirror what
    ``ModelCatalog.parseAvailable`` / ``parseCached`` / ``parseInfoRepo``
    actually parse (column-aligned rows behind a header + divider; an
    ``Alias: <alias> -> <repo>`` line for ``info``).

    Returns True when ``subcommand`` was handled, so ``main`` knows not
    to fall through to the server.
    """
    if subcommand == "models":
        print("Available models")
        print("Alias                  Parser           Reasoning")
        print("---------------------  ---------------  ---------")
        print("fake-alias             hermes           qwen3")
        # A video-generation row, in the tagged section the real engine
        # emits (#1607). It has no tokenizer and cannot answer a chat
        # request, so the desktop must filter it out of every catalog
        # surface. Emitting it here lets `flow_catalog_integrity` prove the
        # FILTER works, rather than asserting against whatever the real
        # registry happens to contain today.
        print()
        print("Video models (1 aliases)")
        print("Alias                  Size       Kind        HF id")
        print("---------------------  ---------  ----------  ------")
        print("fake-video-alias       13.3 GiB   [video:gen] fake/video-mlx")
        return True
    if subcommand == "ls":
        print("Cached models")
        print("Alias                  Repo                   Size")
        print("---------------------  ---------------------  ------")
        print(f"fake-alias             {FAKE_REPO}        1.2 GB")
        return True
    if subcommand == "info":
        print(f"Alias: {alias} -> {FAKE_REPO}")
        return True
    return False


def main():
    args = _parse_args(sys.argv)

    # Only ``serve`` runs a server. Every other subcommand prints and
    # exits, exactly like the real binary.
    #
    # This branch is the whole point of parsing ``subcommand``: without
    # it the fake ignored the verb and started the HTTP server for ANY
    # invocation. ``ModelCatalog.runRapidMlx(args: ["models"])`` (the
    # catalog refresh on app launch) therefore spawned a child that
    # never exited, so its pipe write ends never closed, so both
    # ``readPipeData`` drainers blocked on an EOF that could not arrive
    # and ``terminationHandler``'s ``drainGroup.wait()`` deadlocked the
    # continuation. Net effect: ``scripts/smoke.sh`` hung forever
    # instead of running its chat-lifecycle assertions, and left a
    # stray listener squatting :8000 (which the next Rapid launch's
    # PortSweep would then reap — along with any real rapid-mlx the
    # developer had running on that port).
    #
    # Default-deny is deliberate: an unknown verb exits 0 with no
    # output rather than falling through to the server, so a future
    # ``ModelCatalog`` subcommand can't resurrect the hang.
    if args.subcommand != "serve":
        _emit_catalog(args.subcommand, args.alias)
        sys.exit(0)

    _event("server_started", alias=args.alias, pid=os.getpid())

    # Match the real server's startup banner shape closely enough
    # that DownloadProgress's "Loading model with" detection fires
    # — this lets the SwiftUI overlay flip out of the spinner state
    # immediately instead of waiting for the (nonexistent) GPU warmup.
    print(f"Loading model with BatchedEngine: {args.alias}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"INFO:     Uvicorn running on http://{args.host}:{args.port} "
        f"(Press CTRL+C to quit)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
PYEOF
