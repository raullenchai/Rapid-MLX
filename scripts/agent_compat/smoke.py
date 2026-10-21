"""Exercise current agent CLIs against an authenticated, model-free loopback API.

Run from the repository root with ``python scripts/agent_compat/smoke.py``.
The process HOME and all agent state live in a disposable directory. A request
with the wrong model or key fails even if the CLI exits successfully.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rapid_mlx.agents import isolated

MODEL = "compat-local-model"
REPLY = "COMPAT_REPLY_OK"


class Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    key = ""

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self.send_json(
                {"object": "list", "data": [{"id": MODEL, "object": "model"}]}
            )
        self.send_error(404)

    def send_json(self, body, status=200, content_type="application/json"):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_events(self, events):
        data = "".join(
            f"event: {name}\ndata: {json.dumps(payload)}\n\n"
            for name, payload in events
        )
        data += "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data.encode())))
        self.end_headers()
        self.wfile.write(data.encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        auth = self.headers.get("Authorization", "")
        key_ok = (
            auth == f"Bearer {self.key}" or self.headers.get("x-api-key") == self.key
        )
        self.requests.append(
            {"path": self.path, "model": body.get("model"), "key_ok": key_ok}
        )
        if not key_ok:
            return self.send_json({"error": {"message": "invalid local key"}}, 401)
        if body.get("model") != MODEL:
            return self.send_json({"error": {"message": "wrong local model"}}, 400)
        if self.path.split("?", 1)[0].endswith("/messages"):
            message = {
                "id": "msg_compat",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": REPLY}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            if not body.get("stream"):
                return self.send_json(message)
            return self.send_events(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {**message, "content": [], "stop_reason": None},
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": REPLY},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": 1},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
        if self.path.endswith("/responses"):
            response = {
                "id": "resp_compat",
                "object": "response",
                "model": MODEL,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_compat",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": REPLY, "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
            if not body.get("stream"):
                return self.send_json(response)
            return self.send_events(
                [
                    (
                        "response.created",
                        {
                            "type": "response.created",
                            "response": {
                                **response,
                                "status": "in_progress",
                                "output": [],
                            },
                        },
                    ),
                    (
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": "msg_compat",
                                "type": "message",
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        },
                    ),
                    (
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "item_id": "msg_compat",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        },
                    ),
                    (
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": "msg_compat",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": REPLY,
                            "logprobs": [],
                        },
                    ),
                    (
                        "response.output_text.done",
                        {
                            "type": "response.output_text.done",
                            "item_id": "msg_compat",
                            "output_index": 0,
                            "content_index": 0,
                            "text": REPLY,
                            "logprobs": [],
                        },
                    ),
                    (
                        "response.content_part.done",
                        {
                            "type": "response.content_part.done",
                            "item_id": "msg_compat",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": REPLY,
                                "annotations": [],
                            },
                        },
                    ),
                    (
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": response["output"][0],
                        },
                    ),
                    (
                        "response.completed",
                        {"type": "response.completed", "response": response},
                    ),
                ]
            )
        if self.path.endswith("/chat/completions"):
            completion = {
                "id": "chatcmpl_compat",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": REPLY},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            if not body.get("stream"):
                return self.send_json(completion)
            chunks = [
                {
                    "id": "chatcmpl_compat",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": REPLY},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_compat",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            data = (
                "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data.encode())))
            self.end_headers()
            return self.wfile.write(data.encode())
        self.send_error(404)


def main() -> None:
    known_bad = "--known-bad-opencode" in sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="rapid-agent-compat-") as tmp:
        home = Path(tmp)
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(home)
        for name in (
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_CACHE_HOME",
            "OPENCODE_CONFIG",
            "OPENCODE_CONFIG_DIR",
            "OPENCODE_CONFIG_CONTENT",
            "CODEX_HOME",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "OPENAI_API_KEY",
        ):
            os.environ.pop(name, None)
        Handler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            for agent in (
                ("opencode",) if known_bad else ("codex", "claude-code", "opencode")
            ):
                isolated.configure(agent, MODEL, port=port)
                root = isolated.connection_dir() / "agents" / agent
                Handler.key = isolated.stable_key()
                # CLIs see only the fixture key, never credentials inherited
                # from a developer shell or a self-hosted runner.
                env = {
                    name: os.environ[name]
                    for name in (
                        "PATH",
                        "LANG",
                        "LC_ALL",
                        "TERM",
                        "TMPDIR",
                        "USER",
                        "LOGNAME",
                    )
                    if name in os.environ
                }
                env["HOME"] = str(home)
                env["NO_PROXY"] = "127.0.0.1,localhost"
                env["XDG_CONFIG_HOME"] = str(home / "config")
                env["XDG_DATA_HOME"] = str(home / "data")
                env["XDG_CACHE_HOME"] = str(home / "cache")
                env["RAPID_MLX_API_KEY"] = Handler.key
                if agent == "codex":
                    env["CODEX_HOME"] = str(root)
                    cmd = [
                        "codex",
                        "exec",
                        "--profile",
                        "rapid",
                        "--skip-git-repo-check",
                        "Say " + REPLY,
                    ]
                elif agent == "claude-code":
                    cmd = [
                        "claude",
                        "--settings",
                        str(root / "settings.json"),
                        "-p",
                        "Say " + REPLY,
                    ]
                else:
                    user_config = home / "config" / "opencode" / "opencode.json"
                    user_config.parent.mkdir(parents=True, exist_ok=True)
                    user_config.write_text('{"model":"remote/user-model"}\n')
                    project = home / "project"
                    project.mkdir()
                    (project / "opencode.json").write_text(
                        '{"model":"remote/project-model"}\n'
                    )
                    from rapid_mlx.agents.opencode_version import installed_version

                    if known_bad:
                        # Recreate the removed top-level --model invocation.
                        command = isolated._command(
                            agent, root, MODEL, port, version=installed_version()
                        )
                        command = command.removesuffix(" --standalone")
                        command += f" --model rapid-mlx/{MODEL} run"
                    else:
                        command = isolated._command(
                            agent,
                            root,
                            MODEL,
                            port,
                            version=installed_version(),
                            opencode_run=True,
                        )
                    command += " " + shlex.quote("Say " + REPLY)
                    cmd = ["sh", "-c", command]
                before = len(Handler.requests)
                result = subprocess.run(
                    cmd,
                    cwd=project if agent == "opencode" else home,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=90,
                    check=False,
                )
                arrived = Handler.requests[before:]
                print(
                    f"{agent}: exit={result.returncode}, requests={arrived}", flush=True
                )
                assert arrived, (
                    f"{agent}: no API request; {result.stdout[-2000:]} {result.stderr[-2000:]}"
                )
                assert all(r["key_ok"] and r["model"] == MODEL for r in arrived), (
                    arrived
                )
                assert result.returncode == 0, (
                    f"{agent}: {result.stdout[-2000:]} {result.stderr[-2000:]}"
                )
                assert REPLY in result.stdout, (
                    f"{agent}: reply absent: {result.stdout[-2000:]} {result.stderr[-2000:]}"
                )
                if agent == "opencode":
                    assert (
                        user_config.read_text() == '{"model":"remote/user-model"}\n'
                    ), user_config.read_text()
        finally:
            server.shutdown()
            server.server_close()
            if old_home is not None:
                os.environ["HOME"] = old_home


if __name__ == "__main__":
    main()
