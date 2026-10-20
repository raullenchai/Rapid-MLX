"""Real CLI regression for a subscription-shaped profile and local launch."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rapid_mlx.agents import isolated


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("claude") is None, reason="Claude Code CLI absent")
def test_subscription_profile_sends_local_bearer(tmp_path, monkeypatch):
    monkeypatch.setattr(isolated.Path, "home", lambda: tmp_path)
    account_file = tmp_path / ".claude.json"
    account_bytes = json.dumps(
        {"oauthAccount": {"accountUuid": "simulated-login"}}
    ).encode()
    account_file.write_bytes(account_bytes)
    received = []
    expected = isolated.stable_key()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            received.append((self.path, self.headers.get("Authorization")))
            if self.headers.get("Authorization") != f"Bearer {expected}":
                self.send_response(401)
                self.end_headers()
                return
            if self.path.split("?", 1)[0].endswith("/count_tokens"):
                payload = {"input_tokens": 10}
            else:
                payload = {
                    "id": "msg_local_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "OK"}],
                    "model": "local-model",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        port = server.server_address[1]
        isolated.configure("claude-code", "local-model", port=port)
        settings = isolated.connection_dir() / "agents/claude-code/settings.json"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = dict(os.environ)
        env["HOME"] = str(tmp_path)
        env["CLAUDE_CONFIG_DIR"] = str(settings.parent / "config")
        env["CLAUDE_CODE_OAUTH_TOKEN"] = "simulated-invalid-oauth-token"
        env["DISABLE_TELEMETRY"] = "1"
        env["DISABLE_ERROR_REPORTING"] = "1"
        try:
            result = subprocess.run(
                ["claude", "--settings", str(settings), "-p", "Reply OK"],
                env=env,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=90,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert result.returncode == 0, result.stderr[-500:]
    assert "OK" in result.stdout
    assert any(
        path.startswith("/v1/messages") and authorization == f"Bearer {expected}"
        for path, authorization in received
    )
    assert account_file.read_bytes() == account_bytes
    assert not (tmp_path / ".claude").exists()
