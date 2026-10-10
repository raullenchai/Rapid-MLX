"""Exercise the standalone harness with real SDKs and deterministic HTTP replies."""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "integrations/test_pydantic_ai_full.py"


@pytest.fixture
def harness():
    # These are optional integration SDKs, not runtime/test dependencies.
    pytest.importorskip("pydantic_ai")
    pytest.importorskip("openai")
    spec = importlib.util.spec_from_file_location("pydantic_harness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bad_stream", [False, True])
def test_real_sdk_sync_stream_sync(harness, bad_stream):
    requests = []
    connections = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def reply(self, body, content_type="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def do_GET(self):
            assert self.path == "/v1/models"
            self.reply(json.dumps({"data": [{"id": "test-model"}]}).encode())

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            # The peer port stays stable for a persistent TCP connection.
            # A fresh socket per request would evade the original pool bug.
            connections.append(self.client_address)
            base = {"id": "chat-test", "created": 1, "model": "test-model"}
            if body.get("stream"):
                events = []
                for text in ["x"] if bad_stream else ["1, ", "2, ", "3, ", "4, ", "5"]:
                    chunk = {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    events.append("data: " + json.dumps(chunk) + "\n\n")
                events.append(
                    "data: "
                    + json.dumps(
                        {
                            **base,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        }
                    )
                    + "\n\n"
                )
                events.append("data: [DONE]\n\n")
                self.reply("".join(events).encode(), "text/event-stream")
                return

            tools = {t["function"]["name"] for t in body.get("tools", [])}
            completed = {
                m.get("tool_call_id") for m in body["messages"] if m["role"] == "tool"
            }
            call = None
            if "final_result" in tools:
                call = ("final_result", {"name": "Alice", "age": 30})
            elif "get_weather" in tools and "get_weather" not in completed:
                call = ("get_weather", {"city": "Paris"})
            elif "add" in tools and "add" not in completed:
                call = ("add", {"a": 3, "b": 4})
            elif "multiply" in tools and "multiply" not in completed:
                call = ("multiply", {"a": 7, "b": 5})
            message = {"role": "assistant", "content": "4; Bob; Paris 22; 35"}
            if call:
                name, args = call
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": name,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            self.reply(
                json.dumps(
                    {
                        **base,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": "tool_calls" if call else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 10,
                            "total_tokens": 20,
                        },
                    }
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            env={
                **os.environ,
                "RAPID_MLX_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "PYDANTIC_AI_NO_BANNER": "1",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    output = result.stdout + result.stderr
    assert result.returncode == (1 if bad_stream else 0), output
    assert f"PydanticAI: {5 if bad_stream else 6}/6 passed" in output, output
    assert requests[0].get("stream", False) is False
    assert requests[1]["stream"] is True
    assert all(not r.get("stream", False) for r in requests[2:])
    assert requests[1]["stream_options"]["include_usage"] is True
    # The failure is reuse of the plain request's pooled connection for
    # streaming on another loop. The SDK may discard that socket after the
    # stream, so later synchronous calls need not keep the same peer port.
    assert connections[0] == connections[1], connections
    assert "different event loop" not in output
    if bad_stream:
        assert "2_stream: FAIL: Too short" in output


@pytest.mark.parametrize("fail", [False, True])
def test_client_closed_on_own_loop_before_loop_exit(harness, monkeypatch, fail):
    clients = []
    loops = []

    class Client(harness.AsyncOpenAI):
        async def close(self):
            assert asyncio.get_running_loop() is loops[-1]
            await super().close()

    def client_factory(**kwargs):
        client = Client(**kwargs)
        clients.append(client)
        return client

    def run_tests(model, loop):
        loops.append(loop)
        assert asyncio.get_event_loop() is loop
        assert model.provider.client is clients[-1]
        if fail:
            raise RuntimeError("scenario crashed")
        return 0

    monkeypatch.setattr(
        harness._httpx, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    monkeypatch.setattr(harness, "AsyncOpenAI", client_factory)
    monkeypatch.setattr(harness, "run_tests", run_tests)
    # Calling main twice must never recover a cached pool from a closed loop.
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="scenario crashed"):
                harness.main()
        else:
            assert harness.main() == 0
        assert clients[-1].is_closed()
        assert loops[-1].is_closed()
    assert clients[0] is not clients[1]
    assert loops[0] is not loops[1]
