#!/usr/bin/env python3
"""Text-only loopback gate between the Work helper and Rapid inference.

The helper is sandboxed away from Rapid's real port. This gate rejects media
parts before forwarding a chat request, so Rapid cannot fetch a remote URL on
the helper's behalf through its multimodal input handlers.
"""

import hmac
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT = int(sys.argv[1])
RAPID_PORT = int(sys.argv[2])
MODEL = sys.argv[3]
FULL_ACCESS = len(sys.argv) > 4 and sys.argv[4] == "--full-access"
GOOSE_CHAT = "--goose-chat" in sys.argv[4:]
BEARER = os.environ["WORK_PROXY_BEARER"]
UPSTREAM_BEARER = os.environ.get("WORK_PROXY_UPSTREAM_BEARER", BEARER)
BEARER_FILE = os.environ.get("WORK_PROXY_BEARER_FILE", "")
METRICS_KEY = os.environ.get("WORK_PROXY_METRICS_KEY", "")
CONTRACT_FILE = os.environ.get("WORK_PROXY_CONTRACT_FILE", "")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
METRICS_LOCK = threading.Lock()
LATEST_METRIC = {"sequence": 0, "tokensPerSecond": None}


def upstream_bearer():
    """Allow a restarted local model server to rotate its key mid-turn."""
    if BEARER_FILE:
        try:
            with open(BEARER_FILE, encoding="utf-8") as handle:
                return handle.read().strip() or BEARER
        except OSError:
            pass
    return UPSTREAM_BEARER


def compact_goose_request(body):
    """Keep the one-system-message contract without changing Goose's guidance."""
    messages = body.get("messages") or []
    desktop_system = body.pop("_rapid_desktop_system", False)
    if desktop_system and messages and messages[0].get("role") == "system":
        # The local chat template accepts one leading system row. Append the
        # complete Goose prompt after explicit Desktop instructions.
        for index in range(1, len(messages)):
            if messages[index].get("role") == "system":
                messages[0]["content"] += "\n\n" + messages[index]["content"]
                del messages[index]
                break
    return body


def apply_goose_contract(body, contract):
    """Apply the launcher's explicit options without replacing Goose guidance."""
    messages = body.get("messages", [])
    system = contract.get("system_prompt", "")
    if system:
        messages.insert(0, {"role": "system", "content": system})
        body["_rapid_desktop_system"] = True
    body["messages"] = messages
    if not contract.get("use_goose_sampling_defaults", False):
        body["temperature"] = contract["temperature"]
        body["top_p"] = contract["top_p"]
        body["max_tokens"] = contract["max_tokens"]
        body["repetition_penalty"] = contract["repetition_penalty"]
        body["frequency_penalty"] = 0.0
        body["presence_penalty"] = 0.0
        kwargs = body.get("chat_template_kwargs") or {}
        if contract["enable_thinking"]:
            kwargs.pop("enable_thinking", None)
        else:
            kwargs["enable_thinking"] = False
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        else:
            body.pop("chat_template_kwargs", None)
    return body


def publish_metric(sequence, rate):
    with METRICS_LOCK:
        if LATEST_METRIC["sequence"] == sequence:
            LATEST_METRIC["tokensPerSecond"] = rate


class StreamRate:
    """Observe the unchanged SSE bytes and pair engine usage with decode time."""

    def __init__(self):
        self.pending = b""
        self.first_token_at = None
        self.last_token_at = None
        self.completion_tokens = None
        self.streamed_chars = 0

    def feed(self, chunk):
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            if not line.startswith(b"data:"):
                continue
            try:
                event = json.loads(line[5:])
            except (ValueError, UnicodeDecodeError):
                continue
            usage = event.get("usage")
            if isinstance(usage, dict):
                self.completion_tokens = usage.get("completion_tokens")
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if any(
                    delta.get(key)
                    for key in ("content", "reasoning_content", "tool_calls")
                ):
                    now = time.monotonic()
                    if self.first_token_at is None:
                        self.first_token_at = now
                    self.last_token_at = now
                    self.streamed_chars += sum(
                        len(delta.get(key) or "")
                        for key in ("content", "reasoning_content")
                    )
                    for tool in delta.get("tool_calls") or []:
                        function = tool.get("function") or {}
                        self.streamed_chars += len(function.get("name") or "")
                        self.streamed_chars += len(function.get("arguments") or "")

    def live_tokens_per_second(self):
        """Approximate decode rate until the terminal engine usage arrives."""
        if self.first_token_at is None or self.last_token_at is None:
            return None
        duration = self.last_token_at - self.first_token_at
        estimated_tokens = self.streamed_chars / 4
        return (
            (estimated_tokens - 1) / duration
            if estimated_tokens > 1 and duration > 0.05
            else None
        )

    def tokens_per_second(self):
        count = self.completion_tokens
        if not isinstance(count, int) or count <= 1 or self.first_token_at is None:
            return None
        duration = self.last_token_at - self.first_token_at
        return (count - 1) / duration if duration > 0.05 else None


def text_only(body):
    if not isinstance(body, dict) or body.get("model") != MODEL:
        return False
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    if body.get("modalities") not in (None, ["text"]):
        return False
    if any(key in body for key in ("input", "image", "audio", "video", "files")):
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if message.get("role") not in (
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        ):
            return False
        if any(
            key in message
            for key in ("image_url", "audio_url", "video_url", "file_url")
        ):
            return False
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            return False
        if any(
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
            or set(part) != {"type", "text"}
            for part in content
        ):
            return False
    return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/metrics" and GOOSE_CHAT:
            if not hmac.compare_digest(
                self.headers.get("X-Metrics-Key", ""), METRICS_KEY
            ):
                self.send_error(401)
                return
            with METRICS_LOCK:
                data = json.dumps(LATEST_METRIC).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + BEARER
        ):
            self.send_error(401)
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_error(400)
            return
        if not 0 < length <= (100_000_000 if FULL_ACCESS else 8_000_000):
            self.send_error(413)
            return
        try:
            raw = self.rfile.read(length)
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self.send_error(400)
            return
        if not FULL_ACCESS and not text_only(body):
            self.send_error(403, "Work allows text-only inference")
            return
        if GOOSE_CHAT and CONTRACT_FILE:
            try:
                with open(CONTRACT_FILE, encoding="utf-8") as contract_stream:
                    body = apply_goose_contract(body, json.load(contract_stream))
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(500, "Goose contract unavailable")
                return
        if GOOSE_CHAT and body.get("stream") is True:
            body = compact_goose_request(body)
            options = body.get("stream_options") or {}
            options["include_usage"] = True
            body["stream_options"] = options
        if GOOSE_CHAT:
            raw = json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{RAPID_PORT}/v1/chat/completions",
            raw,
            {
                "Authorization": "Bearer " + upstream_bearer(),
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            response = OPENER.open(request, timeout=900)
        except urllib.error.HTTPError as error:
            response = error
        except Exception:
            self.send_error(502, "Local inference unavailable")
            return
        with response:
            rate = StreamRate() if GOOSE_CHAT and response.status == 200 else None
            if rate is not None:
                with METRICS_LOCK:
                    LATEST_METRIC["sequence"] += 1
                    LATEST_METRIC["tokensPerSecond"] = None
                    metric_sequence = LATEST_METRIC["sequence"]
            self.send_response(response.status)
            self.send_header(
                "Content-Type", response.headers.get("Content-Type", "application/json")
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while chunk := response.read1(16384):
                    if rate is not None:
                        rate.feed(chunk)
                        if rate.completion_tokens is not None:
                            publish_metric(metric_sequence, rate.tokens_per_second())
                        elif live_rate := rate.live_tokens_per_second():
                            publish_metric(metric_sequence, live_rate)
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if rate is not None:
                    publish_metric(metric_sequence, rate.tokens_per_second())


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
