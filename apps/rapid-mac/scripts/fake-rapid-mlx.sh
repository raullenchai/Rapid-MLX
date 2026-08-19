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
import base64
import hashlib
import io
import json
import os
import struct
import sys
import threading
import time
import wave
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


if sys.argv[1:] == ["launch", "list", "--json"]:
    print(json.dumps([
        {"id": "cline", "name": "Cline", "kind": "config_writer", "config_path": "~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"},
        {"id": "claude-code", "name": "Claude Code", "kind": "config_writer", "config_path": "~/.claude/settings.json"},
        {"id": "continue-dev", "name": "Continue.dev", "kind": "config_writer", "config_path": "~/.continue/config.json"},
        {"id": "cursor", "name": "Cursor", "kind": "config_writer", "config_path": "~/Library/Application Support/Cursor/User/settings.json"},
        {"id": "aider", "name": "Aider", "kind": "adapter_profile", "config_path": None},
        {"id": "codex", "name": "Codex CLI", "kind": "adapter_profile", "config_path": None},
        {"id": "hermes", "name": "Hermes Agent", "kind": "adapter_profile", "config_path": None},
        {"id": "kilo-code", "name": "Kilo Code", "kind": "adapter_profile", "config_path": None},
        {"id": "langchain", "name": "LangChain", "kind": "adapter_profile", "config_path": None},
        {"id": "opencode", "name": "OpenCode", "kind": "adapter_profile", "config_path": None},
        {"id": "openhands", "name": "OpenHands", "kind": "adapter_profile", "config_path": None},
        {"id": "pydanticai", "name": "PydanticAI", "kind": "adapter_profile", "config_path": None},
        {"id": "qwen-code", "name": "Qwen Code", "kind": "adapter_profile", "config_path": None},
        {"id": "smolagents", "name": "smolagents", "kind": "adapter_profile", "config_path": None},
    ]))
    sys.exit(0)


try:
    with open(os.path.join(os.environ.get("HOME", ""), ".rapid-golden-fake.json"), encoding="utf-8") as stream:
        FILE_CONFIG = json.load(stream)
except (OSError, ValueError):
    FILE_CONFIG = {}


def _setting(name, default=None):
    return os.environ.get(name, FILE_CONFIG.get(name, default))


def _pulled_audio_aliases():
    state_path = _setting("FAKE_AUDIO_PULL_STATE")
    if not state_path:
        return set()
    try:
        with open(state_path) as stream:
            return {line.strip() for line in stream if line.strip()}
    except OSError:
        return set()


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
        "The same renderer also handles punctuation-bearing configured tokens:\n\n",
        "```", "css", "\n",
        ".card { background-", "color: red; }\n",
        "@font-", "face { font-family: Demo; }\n",
        "```", "\n\n",
        "```", "makefile", "\n",
        ".PH", "ONY: all\n",
        "FILES := $(filter-", "out %.tmp,$(ALL_FILES))\n",
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
        "\n\nA bridged congruence is $$a^{p-1} \\equiv 1 \\mod p$$.",
        "\n\nA bridged alignment is $$\\begin{align}x &= 1 \\\\ y &= \\boxed{2}\\end{align}$$.",
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


# --------------------------------------------------------------------------
# Image generation.
#
# The Images tab talks to ``/v1/images/*`` and nothing else, so the fake can
# answer it without any notion of diffusion. Two properties matter and neither
# needs weights:
#
#   * a render TAKES TIME, so the in-flight progress card (and its
#     ``Images.Cancel``) is observable rather than a frame the flow can never
#     catch;
#   * each render returns DIFFERENT bytes, and reports their SHA-256, so the
#     flow can tell "the sidecar produced two images" from "it produced one
#     twice". That is a claim about the WIRE; whether the app then draws both
#     is past what an accessibility dump can see.
#
# The PNG is built here rather than pasted as a base64 literal: a blob nobody
# can read is a blob nobody can verify, and a corrupt one would fail as
# "the gallery stayed empty" — pointing at the app instead of at this file.
FAKE_IMAGE_ALIAS = "fake-image-alias"
FAKE_IMAGE_REPO = "fake-org/fake-image-repo"

# The alias currently being served by THIS process. Set in ``main`` for the
# ``serve`` branch so the ``/v1/models/residency`` snapshot can report a
# resident model — which is what keeps ``ServerManager`` on the in-process
# ``/v1/models/load`` path instead of the legacy stop/start fallback when the
# GUI asks for a second model while the sidecar is already running (#1838).
SERVED_ALIAS = ""

# The engine's own, actionable rejection reason. Kept out of the snapshot so a
# stock persona never changes residency shape; a flow opts in with
# ``FAKE_REJECT_IMAGE_LOAD=1`` to exercise the rejected non-404/405 load path.
FAKE_REJECTION_DETAIL = (
    "image generation requires the 'rapid-mlx[image]' Python extra "
    "(pip install 'rapid-mlx[image]')"
)


def _one_pixel_png(rgb):
    """A real, decodable 1x1 truecolour PNG."""

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" + bytes(rgb))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


class _ImageRenders:
    """Server-side render state, shared by the generate and progress routes.

    ``ThreadingHTTPServer`` serves the progress polls on other threads while a
    generate call is still sleeping through its steps, which is exactly the
    concurrency the real engine has — and the reason the counters are taken
    under a lock rather than read raw.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.running = False
        self.step = 0
        self.total = 0
        self.started_at = 0.0
        self.cancelled = False
        self.count = 0

    def begin(self, total):
        with self._lock:
            # One render at a time, decided atomically under the lock. The real
            # server is a single model in a single process; a second concurrent
            # generation would not race its counters, it would be refused. The
            # shared-singleton state below is only safe because of this gate —
            # without it a second `begin` would reset `step`/`total`/`cancelled`
            # out from under a render still sleeping through its loop.
            if self.running:
                return None
            self.running = True
            self.step = 0
            self.total = total
            self.started_at = time.time()
            self.cancelled = False
            self.count += 1
            return self.count

    def advance(self):
        with self._lock:
            self.step += 1
            return self.cancelled

    def end(self):
        with self._lock:
            self.running = False
            self.step = self.total

    def cancel(self):
        with self._lock:
            self.cancelled = True

    def snapshot(self):
        with self._lock:
            elapsed = int((time.time() - self.started_at) * 1000) if self.started_at else 0
            return {
                "running": self.running,
                "step": self.step,
                "total": self.total,
                "elapsed_ms": elapsed,
            }

RENDERS = _ImageRenders()


def _extract_image_part(raw_body, content_type):
    """The raw image bytes carried by the ``name="image"`` part of a multipart
    edit request.

    ``raw_body`` is the full request body and ``content_type`` is the request's
    ``Content-Type`` header. The image part's own headers end at the first
    blank line after ``name="image"``; the bytes run from there until the
    CRLF + the next ``--<boundary>`` separator, so a stray ``\r\n--`` that
    happens to occur inside PNG pixel data cannot truncate it. Returns the
    image bytes, or ``None`` when no image part is present (parsing is
    best-effort and is not the authority on image validity — hermetic unit
    tests are).
    """
    boundary = None
    for token in (content_type or "").split(";"):
        token = token.strip()
        if token.startswith("boundary="):
            boundary = token.split("=", 1)[1].strip('"')
            break
    marker = b'name="image"'
    i = raw_body.find(marker)
    if i < 0:
        return None
    hdr = raw_body.find(b"\r\n\r\n", i)
    if hdr < 0:
        return None
    start = hdr + 4
    if boundary:
        sep = ("\r\n--" + boundary).encode()
        j = raw_body.find(sep, start)
        end = j if j >= 0 else len(raw_body)
    else:
        j = raw_body.find(b"\r\n--", start)
        end = j if j >= 0 else len(raw_body)
    return raw_body[start:end]


def _png_rgba_sha256(png_bytes):
    """SHA-256 of the DECODED RGBA pixels of a PNG (including its
    width and height, so geometry is pinned as well as pixels).

    The app's ``EditImageImporter.pngData`` decodes and re-encodes an import
    through ``NSBitmapImageRep``, so the uploaded PNG is not byte-identical
    to the file the user picked — ancillary chunks (iCCP, eXIf ...) and the
    IDAT zlib stream can legitimately differ. Comparing raw bytes against the
    fixture would be fragile across macOS encoder versions. Comparing the
    decoded pixel payload instead pins the user contract that matters: the
    same pixels reached the wire. Returns the hex digest, or ``None`` when the
    PNG cannot be decoded.

    Supports 8-bit, non-interlaced PNGs of color types 0 (gray), 2 (RGB),
    3 (palette), 4 (gray+alpha) and 6 (RGBA); everything else is normalized
    to 8-bit RGBA before hashing. This is the same code the ``png-rgba-sha``
    subcommand runs, so the golden flow and the request fake agree exactly.
    """
    try:
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        pos = 8
        width = height = bit_depth = color_type = interlace = None
        palette = None
        idat = bytearray()
        while pos < len(png_bytes):
            length = int.from_bytes(png_bytes[pos:pos + 4], "big")
            ctype = png_bytes[pos + 4:pos + 8]
            data = png_bytes[pos + 8:pos + 8 + length]
            if ctype == b"IHDR":
                width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                    ">IIBBBBB", data
                )
            elif ctype == b"PLTE":
                palette = data
            elif ctype == b"IDAT":
                idat += data
            pos += 12 + length
        if not (width and height and bit_depth == 8 and interlace == 0):
            return None
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
        if channels is None:
            return None
        raw = bytearray(zlib.decompress(bytes(idat)))
        stride = width * channels
        prev = bytearray(stride)
        rows = bytearray()
        row_len = stride + 1
        for y in range(height):
            filter_type = raw[y * row_len]
            line = bytearray(raw[y * row_len + 1:(y + 1) * row_len])
            for x in range(stride):
                a = line[x - channels] if x >= channels else 0
                b = prev[x]
                c = prev[x - channels] if x >= channels else 0
                if filter_type == 0:
                    value = line[x]
                elif filter_type == 1:
                    value = line[x] + a
                elif filter_type == 2:
                    value = line[x] + b
                elif filter_type == 3:
                    value = line[x] + (a + b) // 2
                elif filter_type == 4:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    value = line[x] + (a if pa <= pb and pa <= pc else (b if pb <= pc else c))
                else:
                    return None
                line[x] = value & 0xFF
            prev = line
            rows += line
        rgba = bytearray()
        if color_type == 6:
            rgba = rows
        elif color_type == 2:
            for i in range(0, len(rows), 3):
                rgba += rows[i:i + 3] + b"\xff"
        elif color_type == 4:
            for i in range(0, len(rows), 2):
                rgba += rows[i:i + 1] * 3 + rows[i + 1:i + 2]
        elif color_type == 0:
            for value in rows:
                rgba += bytes((value, value, value, 255))
        elif color_type == 3:
            if palette is None:
                return None
            for entry in rows:
                r, g, b = palette[entry * 3:entry * 3 + 3]
                rgba += bytes((r, g, b, 255))
        else:
            return None
        # Include the decoded width/height in the digest. Hashing only the
        # flattened RGBA stream would let two images with identical pixels
        # but different dimensions (e.g. 1×4 vs 2×2) collide, so a golden
        # flow could pass even if import/re-encoding corrupted the geometry.
        h = hashlib.sha256()
        h.update(width.to_bytes(4, "big"))
        h.update(height.to_bytes(4, "big"))
        h.update(bytes(rgba))
        return h.hexdigest()
    except (zlib.error, struct.error, IndexError):
        return None


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
        if self.path == "/v1/models/residency":
            self._json(200, self._residency_snapshot())
            return
        if self.path.partition("?")[0] == "/v1/images/progress":
            # Polled every few hundred ms while a render is in flight. Answer
            # it even when nothing is running: the client treats a transport
            # failure and "idle" identically, so a 404 here would be
            # indistinguishable from the daemon being down.
            self._json(200, RENDERS.snapshot())
            return
        if self.path.partition("?")[0] == "/v1/audio/voices":
            _event("audio_voices")
            self._json(200, {"voices": ["Golden", "Harbor"]})
            return
        self._json(404, {"error": "not_found"})

    def _residency_snapshot(self):
        """``GET /v1/models/residency`` — the served alias as the sole
        resident model. This is what makes ``ServerManager`` take the
        in-process ``/v1/models/load`` path when the user asks for a second
        model while the sidecar is already up (#1838)."""
        if not SERVED_ALIAS:
            return {
                "memory_limit_bytes": 0,
                "memory_used_bytes": 0,
                "memory_available_bytes": 0,
                "idle_ttl_seconds": 1800,
                "loads_total": 0,
                "evictions_total": 0,
                "models": [],
            }
        return {
            "memory_limit_bytes": 34359738368,
            "memory_used_bytes": 10737418240,
            "memory_available_bytes": 23622320128,
            "idle_ttl_seconds": 1800,
            "loads_total": 1,
            "evictions_total": 0,
            "models": [{
                "id": SERVED_ALIAS,
                "model_path": "fake-org/fake-repo",
                "aliases": [SERVED_ALIAS],
                "modality": "text",
                "state": "resident",
                "pinned": True,
                "primary": True,
                "active_requests": 0,
                "estimated_bytes": 1,
                "measured_bytes": None,
                "idle_seconds": 0.0,
            }],
        }

    def _models_load(self):
        """``POST /v1/models/load`` — the in-process residency load the GUI
        uses to admit a second engine while the sidecar is already running.

        This is where #1838's silent-failure scenario lives: when the engine
        cannot serve the requested model (e.g. a stock bundle with no
        ``[image]`` extra), ``load`` must be answered with a non-2xx,
        non-404/405 response whose ``detail`` carries the actionable reason.
        ``ServerManager`` maps that to ``.rejected(detail)`` and, after this
        fix, publishes it so the initiating surface shows it instead of
        dropping it into the log.

        The flow opts into the rejection with ``FAKE_REJECT_IMAGE_LOAD=1``;
        every other persona keeps the legacy 404 (unsupported) path so their
        stop/start fallback is unchanged.
        """
        length = int(self.headers.get("content-length", "0") or "0")
        body = {}
        if length:
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
        target = body.get("model") if isinstance(body.get("model"), str) else ""
        _event("model_load", alias=target)
        delay_ms = int(_setting("FAKE_RESIDENT_LOAD_DELAY_MS", "0") or "0")
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)
        if _setting("FAKE_REJECT_IMAGE_LOAD") == "1" and target == FAKE_IMAGE_ALIAS:
            # The engine refuses to admit this model (missing Python extra).
            # Mirror the real server's rejection envelope so
            # ``ServerResidencyClient.load`` lands on ``.rejected(detail)``.
            _event("model_load_rejected", alias=target)
            self._json(422, {"detail": FAKE_REJECTION_DETAIL})
            return
        if target == SERVED_ALIAS:
            # Already resident — idempotent success.
            self._json(200, self._residency_snapshot()["models"][0])
            return
        # Unknown target → the legacy 404 the app treats as "no residency
        # support here", falling back to its stop/start path. Preserves the
        # behaviour every other persona depends on.
        self._json(404, {"error": "not_found"})

    def _images_generate(self, *, editing=False):
        """Timed generation/edit render of a real PNG.

        Edit requests are multipart, so parse only the named text fields the
        journey needs to pin. The source bytes themselves are deliberately not
        decoded here; app/server image validation has hermetic unit coverage.
        """
        length = int(self.headers.get("content-length", "0") or "0")
        body = {}
        raw_body = b""
        if length:
            raw_body = self.rfile.read(length)
            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
        def multipart_field(name):
            marker = ('name="' + name + '"\r\n\r\n').encode()
            start = raw_body.find(marker)
            if start < 0:
                return None
            start += len(marker)
            end = raw_body.find(b"\r\n", start)
            return raw_body[start:end].decode("utf-8", errors="replace")

        prompt = multipart_field("prompt") if editing else body.get("prompt")
        prompt = prompt if isinstance(prompt, str) else ""
        raw_count = body.get("n")
        if editing:
            raw_count = multipart_field("n")
            raw_count = int(raw_count) if raw_count and raw_count.isdigit() else 1
        count = raw_count if isinstance(raw_count, int) and raw_count > 0 else 1
        total = max(1, int(_setting("FAKE_IMAGE_STEPS", 8)))
        step_ms = max(0, int(_setting("FAKE_IMAGE_STEP_MS", 300)))
        index = RENDERS.begin(total)
        if index is None:
            # A render is already in flight. The real server runs one model in
            # one process and refuses an overlapping generation rather than
            # interleaving it; mirror that with a 409 instead of clobbering the
            # in-flight render's shared counters.
            _event("image_request_rejected", prompt=prompt)
            self._json(409, {"error": {
                "message": "a render is already in progress; this server "
                           "generates one image at a time",
                "code": "image_render_in_progress",
            }})
            return
        _event(
            "image_request",
            prompt=prompt,
            model=multipart_field("model") if editing else body.get("model"),
            size=multipart_field("size") if editing else body.get("size"),
            n=count,
            operation="edit" if editing else "generation",
            has_image=(b'name="image"; filename="input.png"' in raw_body) if editing else False,
            image_rgba_sha256=(_png_rgba_sha256(_extract_image_part(raw_body, self.headers.get("content-type")))
                               if editing else None),
        )
        cancelled = False
        for _ in range(total):
            time.sleep(step_ms / 1000)
            if RENDERS.advance():
                cancelled = True
                break
        # Real image engines still perform VAE decode / PNG encoding after the
        # last denoise step. Keep that tail observable for GUI phase coverage.
        finish_ms = max(0, int(_setting("FAKE_IMAGE_FINISH_MS", 0)))
        time.sleep(finish_ms / 1000)
        RENDERS.end()
        png = _one_pixel_png(((index * 70) % 256, (index * 130) % 256, (index * 190) % 256))
        encoded = base64.b64encode(png).decode("ascii")
        # The digest is of the BYTES that go on the wire, so a fixture (or an
        # engine) that returns one image twice is visible even while the
        # index keeps incrementing. An index is a counter; only a hash is a
        # statement about content.
        _event(
            "image_response",
            index=index,
            cancelled=cancelled,
            bytes=len(png),
            sha256=hashlib.sha256(png).hexdigest(),
        )
        self._json(
            200,
            {"data": [{"b64_json": encoded} for _ in range(count)], "cancelled": cancelled},
        )

    def do_POST(self):
        if self.path == "/v1/models/load":
            self._models_load()
            return
        if self.path == "/v1/images/generations":
            self._images_generate()
            return
        if self.path == "/v1/images/cancel":
            RENDERS.cancel()
            _event("image_cancel")
            self._json(200, {"cancelled": True})
            return
        if self.path == "/v1/images/edits":
            self._images_generate(editing=True)
            return
        if self.path == "/v1/audio/speech":
            length = int(self.headers.get("content-length", "0") or "0")
            body = json.loads(self.rfile.read(length) or b"{}")
            _event(
                "audio_speech",
                model=body.get("model"),
                voice=body.get("voice"),
                speed=body.get("speed"),
                text=body.get("input"),
            )
            audio = io.BytesIO()
            with wave.open(audio, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                # Long enough for the AX flow to observe Play -> Stop, while
                # still tiny and silent for unattended GUI tests.
                wav.writeframes(b"\x00\x00" * 32000)
            payload = audio.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/v1/audio/transcriptions":
            length = int(self.headers.get("content-length", "0") or "0")
            if length:
                self.rfile.read(length)
            _event("audio_transcription")
            self._json(200, {
                "text": "Golden transcription result.",
                "language": "en",
                "duration": 0.1,
            })
            return
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not_found"})
            return
        # Decode only enough of the request to drive normal SSE chat.
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
        print("Alias                  Parser           Reasoning        Preset")
        print("---------------------  ---------------  ---------------  --------")
        if _setting("FAKE_INCLUDE_STARTER") == "1":
            # A production catalog always contains the onboarding starter.
            # Most flows deliberately keep the compact single-chat-row
            # fixture, but fresh-install must exercise the real default
            # selection contract rather than falling back to fake-alias.
            print("lfm2.5-1b-4bit        hermes           none")
        if _setting("FAKE_CACHED_CURATED_TRADEUP") == "1":
            for index in range(6):
                print(f"a-cached-{index}             hermes           none")
            print("qwen3.5-4b-4bit       hermes           qwen3")
        print("fake-alias             hermes           qwen3")
        print("fake-external-alias    hermes           qwen3")
        if _setting("FAKE_SETTINGS_MTP") == "1":
            print("qwen3.8-27b-4bit       hermes           qwen3           MTP@rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX@3")
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
        # An image-generation row, in its own tagged section (mirroring video).
        # It feeds TWO surfaces from one line, which is the point: the Images
        # tab must OFFER it (``ModelCatalog.parseImageRows``), and the chat
        # picker must REFUSE it (``hasNonChatKindTag``). A single fixture keeps
        # those two assertions about the same model.
        print()
        print("Image models (1 aliases)")
        print("Alias                  Size       Kind        HF id")
        print("---------------------  ---------  ----------  ------")
        print(f"{FAKE_IMAGE_ALIAS}       4.6 GiB    [image:both] {FAKE_IMAGE_REPO}")
        print()
        print("Audio models (2 aliases)")
        print("Alias                  Size       Kind        Family      HF id")
        print("---------------------  ---------  ----------  ----------  ------")
        print("fake-qwen3-tts         1.1 GiB    [audio:tts] qwen3_tts   fake/qwen3-tts")
        print("fake-whisper-small     461 MiB    [audio:stt] whisper     fake/whisper-small")
        return True
    if subcommand == "ls":
        if _setting("FAKE_EMPTY_CACHE_NOTICE") == "1":
            print("No models cached yet. Run 'rapid-mlx pull <alias>' or 'rapid-mlx chat <alias>' to download one.")
        print("Cached models")
        print("Alias                  Repo                   Size")
        print("---------------------  ---------------------  ------")
        if _setting("FAKE_SETTINGS_MTP") != "1":
            print(f"fake-alias             {FAKE_REPO}        1.2 GB")
            print("(external)             fake-external-alias     2.4 GB")
        if _setting("FAKE_CACHED_CURATED_TRADEUP") == "1":
            for index in range(6):
                print(f"a-cached-{index}             fake-org/a-cached-{index}        100 MB")
            print("qwen3.5-4b-4bit       mlx-community/Qwen3.5-4B-MLX-4bit  2.9 GB")
        if _setting("FAKE_SETTINGS_MTP") == "1":
            print("qwen3.8-27b-4bit       rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX  15.2 GB")
        # Cached, so the Images tab resolves to it without a download path —
        # ``ImageGenViewModel.resolveAlias`` prefers a cached entry.
        print(f"{FAKE_IMAGE_ALIAS}       {FAKE_IMAGE_REPO}  4.6 GB")
        pulled_audio = _pulled_audio_aliases()
        if "fake-qwen3-tts" in pulled_audio:
            print("fake-qwen3-tts         fake/qwen3-tts        1.1 GiB")
        elif _setting("FAKE_PARTIAL_AUDIO_CACHE") == "1":
            # A real interrupted multi-shard pull remains visible in `ls` for
            # disk cleanup, but its status alias must never make Audio render
            # Start. The audio-readiness flow clicks through this row and
            # requires a resumptive `pull fake-qwen3-tts`.
            print("(incomplete)           fake/qwen3-tts        633 MB")
        if "fake-whisper-small" in pulled_audio:
            print("fake-whisper-small     fake/whisper-small    461 MiB")
        return True
    if subcommand == "info":
        # Per-alias, not a constant: `ls`/`models` map fake-image-alias to its
        # own repo, and `info` returning the chat repo instead would make
        # ModelCatalog.parseInfoRepo disagree with the row it just parsed —
        # readiness/resolution for the image model would target the chat
        # repository. Default to the chat repo for the chat alias and unknowns.
        repo = {
            FAKE_IMAGE_ALIAS: FAKE_IMAGE_REPO,
            "fake-video-alias": "fake/video-mlx",
            "fake-qwen3-tts": "fake/qwen3-tts",
            "fake-whisper-small": "fake/whisper-small",
            "qwen3.8-27b-4bit": "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
        }.get(alias, FAKE_REPO)
        print(f"Alias: {alias} -> {repo}")
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
        # Utility subcommand: print the SHA-256 of a PNG file's DECODED RGBA
        # pixels and exit. The golden flow shells out to this to get the
        # fixture's expected hash, so it shares the EXACT decoder the request
        # fake uses (``_png_rgba_sha256``) — one source of truth, no drift
        # between the upload assertion and the fixture expectation.
        if args.subcommand == "png-rgba-sha":
            try:
                with open(args.alias, "rb") as stream:
                    digest = _png_rgba_sha256(stream.read())
            except OSError:
                digest = None
            if digest is None:
                print("error: cannot decode PNG", file=sys.stderr)
                sys.exit(1)
            print(digest)
            sys.exit(0)
        _event(
            "command",
            subcommand=args.subcommand,
            alias=args.alias,
            pid=os.getpid(),
            do_not_track=os.environ.get("DO_NOT_TRACK"),
        )
        if args.subcommand == "pull" and _setting("FAKE_DOWNLOAD_OVERRUN") == "1":
            # #1550 fixture: the alias-derived estimate said 563 MiB after
            # 633 MiB had already arrived. Keep the process alive long enough
            # for the AX flow to inspect the in-flight progress card.
            print("[bytes] 663748608/590348288", flush=True)
            time.sleep(5)
            state_path = _setting("FAKE_AUDIO_PULL_STATE")
            if state_path and args.alias in {"fake-qwen3-tts", "fake-whisper-small"}:
                with open(state_path, "a") as stream:
                    stream.write(f"{args.alias}\n")
            sys.exit(0)
        _emit_catalog(args.subcommand, args.alias)
        sys.exit(0)

    # The residency snapshot reports the served alias as resident, which is
    # what keeps the GUI on the in-process /v1/models/load path (#1838).
    # The assignment must use ``global``: the handler methods read the
    # module-level name, and without ``global`` this would only create a local
    # that shadows it, leaving the snapshot permanently empty.
    global SERVED_ALIAS
    SERVED_ALIAS = args.alias
    _event("server_started", alias=args.alias, pid=os.getpid(), port=args.port)

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
