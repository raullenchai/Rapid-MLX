"""Focused regression test for the Goose Chat inference rate while streaming."""

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

PROXY = pathlib.Path(__file__).with_name("work-inference-proxy.py")
GOOSE_PREFIX = pathlib.Path(__file__).with_name("goose-prefix.json")


class GoosePrefixTests(unittest.TestCase):
    def test_bundled_prefix_is_the_captured_stock_system_and_tools(self):
        # Captured from the pinned Goose helper in the packaged app path.
        # The warm request must not invent a visible user turn.
        prefix = json.loads(GOOSE_PREFIX.read_text())
        self.assertEqual([m["role"] for m in prefix["messages"]], ["system"])
        self.assertEqual(len(prefix["tools"]), 34)
        self.assertEqual(
            hashlib.sha256(prefix["messages"][0]["content"].encode()).hexdigest(),
            "b3bc3ac3600356bc18515a0f2c91a4f67f9ebdc56397238081eac8adbbc42631",
        )
        self.assertEqual(
            hashlib.sha256(
                json.dumps(
                    prefix["tools"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "6aabccf39198f9c4710179168d80e13b6b8f2776db26509b064bf03bfbd042ff",
        )


class StreamRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("work_inference_proxy", PROXY)
        cls.module = importlib.util.module_from_spec(spec)
        with (
            patch.object(sys, "argv", [str(PROXY), "0", "0", "test-model"]),
            patch.dict(os.environ, {"WORK_PROXY_BEARER": "test"}),
        ):
            spec.loader.exec_module(cls.module)

    def test_rate_is_available_before_terminal_usage(self):
        module = self.module
        rate = module.StreamRate()
        with patch.object(module.time, "monotonic", side_effect=[10.0, 10.5]):
            rate.feed(b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n')
            self.assertIsNone(rate.live_tokens_per_second())
            rate.feed(b'data: {"choices":[{"delta":{"content":"world!"}}]}\n\n')

        self.assertEqual(rate.live_tokens_per_second(), 4.0)
        self.assertIsNone(rate.tokens_per_second())
        rate.feed(b'data: {"choices":[],"usage":{"completion_tokens":11}}\n\n')
        self.assertEqual(rate.tokens_per_second(), 20.0)

    def test_goose_request_preserves_prompt_and_tool_schemas(self):
        original = {
            "model": "test-model",
            "stream": True,
            "messages": [
                {"role": "system", "content": "global skills " * 2000},
                {"role": "user", "content": "今天谷歌股价多少？"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "large schema " * 50,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "argument detail " * 50,
                                    "enum": ["text", "file"],
                                },
                                "description": {
                                    "type": "string",
                                    "description": "User-facing description",
                                },
                            },
                            "required": ["input"],
                        },
                    },
                }
                for name in (
                    "shell",
                    "tree",
                    "write",
                    "edit",
                    "memory__remember",
                    "autovisualiser__show_chart",
                    "computercontroller__open",
                    "tutorial__help",
                )
            ],
        }
        compact = self.module.compact_goose_request(json.loads(json.dumps(original)))
        self.assertEqual(compact, original)
        self.assertEqual(
            [t["function"]["name"] for t in compact["tools"]],
            [
                "shell",
                "tree",
                "write",
                "edit",
                "memory__remember",
                "autovisualiser__show_chart",
                "computercontroller__open",
                "tutorial__help",
            ],
        )
        self.assertEqual(
            compact["tools"][0]["function"]["parameters"]["properties"]["input"][
                "type"
            ],
            "string",
        )
        parameters = compact["tools"][0]["function"]["parameters"]
        self.assertEqual(parameters["required"], ["input"])
        self.assertEqual(parameters["properties"]["input"]["enum"], ["text", "file"])
        self.assertIn("description", parameters["properties"]["input"])
        self.assertIn("description", parameters["properties"])
        self.assertEqual(
            parameters["properties"]["description"]["description"],
            "User-facing description",
        )

    def test_focused_inner_tool_keeps_parameter_guidance(self):
        body = {
            "tools": [
                {
                    "function": {
                        "name": "create_app_content",
                        "description": "Create content",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "Describe the app",
                                },
                            },
                            "required": ["description"],
                        },
                    }
                }
            ]
        }
        compact = self.module.compact_goose_request(body)
        self.assertEqual(
            compact["tools"][0]["function"]["parameters"]["properties"]["description"][
                "description"
            ],
            "Describe the app",
        )

    def test_desktop_system_merges_with_complete_goose_system(self):
        request = {
            "_rapid_desktop_system": True,
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 1234,
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": "User memory and instructions"},
                {"role": "system", "content": "Goose long prompt"},
                {"role": "user", "content": "Hello"},
            ],
        }
        compact = self.module.compact_goose_request(request)
        self.assertEqual(len(compact["messages"]), 2)
        self.assertEqual(
            compact["messages"][0]["content"],
            "User memory and instructions\n\nGoose long prompt",
        )
        self.assertEqual(compact["messages"][1]["content"], "Hello")
        self.assertEqual(compact["temperature"], 0.2)
        self.assertEqual(compact["top_p"], 0.8)
        self.assertEqual(compact["max_tokens"], 1234)
        self.assertEqual(compact["repetition_penalty"], 1.05)
        self.assertEqual(compact["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("_rapid_desktop_system", compact)

    def test_bridge_applies_per_turn_contract(self):
        received = {}

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                received.update(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

        upstream = HTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            bridge_port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            (pathlib.Path(directory) / "chat-contract.json").write_text(
                json.dumps(
                    {
                        "system_prompt": "Use Swift. Remember the project.",
                        "temperature": 0.19,
                        "top_p": 0.77,
                        "max_tokens": 768,
                        "repetition_penalty": 1.08,
                        "enable_thinking": False,
                    }
                )
            )
            script = pathlib.Path(__file__).with_name("work-helper-watchdog.py")
            bridge = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    "--credential-bridge",
                    str(bridge_port),
                    str(upstream.server_port),
                ],
                cwd=directory,
                env={
                    **os.environ,
                    "WORK_BRIDGE_SCOPED_KEY": "scoped",
                    "WORK_BRIDGE_UPSTREAM_KEY": "upstream",
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                health = f"http://127.0.0.1:{bridge_port}/health"
                for _ in range(100):
                    try:
                        urllib.request.urlopen(health, timeout=0.1).close()
                        break
                    except OSError:
                        threading.Event().wait(0.01)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{bridge_port}/v1/chat/completions",
                    data=json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": "Goose"},
                                {"role": "user", "content": "Hi"},
                            ],
                            "temperature": 0.9,
                            "max_tokens": 4000,
                        }
                    ).encode(),
                    headers={
                        "Authorization": "Bearer scoped",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    self.assertEqual(response.read(), b"ok")
                self.assertEqual(
                    received["messages"][0]["content"],
                    "Use Swift. Remember the project.",
                )
                self.assertEqual(received["messages"][1]["content"], "Goose")
                self.assertEqual(received["temperature"], 0.19)
                self.assertEqual(received["top_p"], 0.77)
                self.assertEqual(received["max_tokens"], 768)
                self.assertEqual(received["repetition_penalty"], 1.08)
                self.assertEqual(
                    received["chat_template_kwargs"], {"enable_thinking": False}
                )
                (pathlib.Path(directory) / "chat-contract.json").write_text(
                    json.dumps(
                        {"system_prompt": "", "use_goose_sampling_defaults": True}
                    )
                )
                received.clear()
                with urllib.request.urlopen(request, timeout=2) as response:
                    self.assertEqual(response.read(), b"ok")
                self.assertEqual(
                    received["messages"],
                    [
                        {"role": "system", "content": "Goose"},
                        {"role": "user", "content": "Hi"},
                    ],
                )
                self.assertEqual(received["temperature"], 0.9)
                self.assertEqual(received["max_tokens"], 4000)
                self.assertNotIn("repetition_penalty", received)
                self.assertNotIn("chat_template_kwargs", received)
            finally:
                bridge.terminate()
                bridge.wait(timeout=3)
                upstream.shutdown()
                upstream.server_close()
                thread.join(timeout=3)

    def test_rotated_server_key_is_read_for_each_upstream_request(self):
        module = self.module
        with tempfile.TemporaryDirectory() as directory:
            key_file = pathlib.Path(directory) / "key"
            with patch.object(module, "BEARER_FILE", str(key_file)):
                key_file.write_text("first")
                self.assertEqual(module.upstream_bearer(), "first")
                key_file.write_text("rotated")
                self.assertEqual(module.upstream_bearer(), "rotated")

    def test_greetings_and_other_languages_retain_tool_schemas(self):
        body = {
            "messages": [
                {"role": "system", "content": "large prompt " * 2000},
                {
                    "role": "user",
                    "content": "Reply in the user's language.\nhi\n<turn-context>time</turn-context>",
                },
            ],
            "tools": [{"type": "function", "function": {"name": "shell"}}],
            "tool_choice": "auto",
        }
        for prompt in (
            "hi",
            "你好",
            "帮我看看 Hacker News 首页前 5 条",
            "write snake.html",
        ):
            candidate = json.loads(json.dumps(body))
            candidate["messages"][1]["content"] = prompt
            compact = self.module.compact_goose_request(candidate)
            self.assertEqual(compact["tools"], body["tools"])
            self.assertEqual(compact["tool_choice"], "auto")
            self.assertEqual(compact["messages"][1]["content"], prompt)

    def test_tool_arguments_contribute_to_live_rate(self):
        module = self.module
        rate = module.StreamRate()
        with patch.object(module.time, "monotonic", side_effect=[10.0, 10.5]):
            rate.feed(
                b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"write_file","arguments":"{\\"path\\":\\"a"}}]}}]}\n\n'
            )
            rate.feed(
                b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":".txt\\"}"}}]}}]}\n\n'
            )
        self.assertGreater(rate.live_tokens_per_second(), 0)

    def test_older_request_cannot_overwrite_newer_rate(self):
        module = self.module
        previous = module.LATEST_METRIC.copy()
        try:
            module.LATEST_METRIC.update(sequence=2, tokensPerSecond=5.0)
            module.publish_metric(1, 99.0)
            self.assertEqual(module.LATEST_METRIC["tokensPerSecond"], 5.0)
            module.publish_metric(2, 20.0)
            self.assertEqual(module.LATEST_METRIC["tokensPerSecond"], 20.0)
        finally:
            module.LATEST_METRIC.update(previous)

    def test_terminal_usage_is_published_before_upstream_eof(self):
        module = self.module
        previous = module.LATEST_METRIC.copy()

        class Response:
            status = 200
            headers = {"Content-Type": "text/event-stream"}
            index = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read1(self, _size):
                self.index += 1
                if self.index == 1:
                    return b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
                if self.index == 2:
                    return b'data: {"choices":[{"delta":{"content":"world!"}}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":11}}\n\ndata: [DONE]\n\n'
                self_test.assertEqual(module.LATEST_METRIC["tokensPerSecond"], 20.0)
                return b""

        self_test = self
        body = json.dumps(
            {
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode()
        handler = object.__new__(module.Handler)
        handler.path = "/v1/chat/completions"
        handler.headers = {
            "Authorization": "Bearer test",
            "Content-Length": str(len(body)),
        }
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda _status: None
        handler.send_header = lambda _name, _value: None
        handler.end_headers = lambda: None
        try:
            module.LATEST_METRIC.update(sequence=0, tokensPerSecond=None)
            with (
                patch.object(module, "GOOSE_CHAT", True),
                patch.object(module.OPENER, "open", return_value=Response()),
                patch.object(module.time, "monotonic", side_effect=[10.0, 10.5]),
            ):
                handler.do_POST()
            self.assertEqual(module.LATEST_METRIC["tokensPerSecond"], 20.0)
        finally:
            module.LATEST_METRIC.update(previous)


if __name__ == "__main__":
    unittest.main()
