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
REASONING_CHUNKS = [
    "Let", " me", " think", " about", " the", " prompt", "."
]
INTER_TOKEN_SLEEP_S = 0.01  # ~20 deltas in 200 ms — fast enough for smoke


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
        # Drain (and ignore) the request body — we don't echo the
        # prompt; the fake's output is deterministic so test
        # assertions can pin specific strings.
        length = int(self.headers.get("content-length", "0") or "0")
        if length:
            self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # #896 crash-recovery harness: when FAKE_DIE_AFTER_CHUNKS
        # is set to a positive integer N, we abruptly os._exit(1)
        # after streaming N content deltas — simulating the real
        # rapid-mlx being SIGKILL'd mid-response. The TCP socket
        # closes with whatever's buffered already in flight, so
        # ChatStreamClient sees partial deltas followed by EOF
        # WITHOUT a [DONE] sentinel or a finish_reason chunk — the
        # exact failure mode v0.4.5 maps to ``.streamTruncated``.
        die_after_chunks_raw = os.environ.get("FAKE_DIE_AFTER_CHUNKS", "")
        try:
            die_after_chunks = int(die_after_chunks_raw)
        except ValueError:
            die_after_chunks = 0
        try:
            for r in REASONING_CHUNKS:
                self.wfile.write(_sse(_delta(reasoning=r)))
                self.wfile.flush()
                time.sleep(INTER_TOKEN_SLEEP_S)
            content_emitted = 0
            for c in CONTENT_CHUNKS:
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
        except BrokenPipeError:
            # Client disconnected mid-stream — fine, the smoke
            # cancelled the stream on its own.
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
