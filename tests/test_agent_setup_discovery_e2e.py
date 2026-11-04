# SPDX-License-Identifier: Apache-2.0
"""Exercise real process discovery, without models or installed agent binaries."""

import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest


def test_setup_discovers_server_before_it_binds(tmp_path):
    import psutil

    try:
        list(psutil.process_iter(["pid", "cmdline"]))
    except (psutil.AccessDenied, PermissionError):
        pytest.skip("host sandbox denies process discovery")
    # Default-port priority is intentional. An unrelated listener there makes
    # this test's non-default discovery precondition false, so skip it.
    try:
        with socket.create_connection(("localhost", 8000), timeout=0.2):
            pytest.skip("discovery gives an existing default-port listener priority")
    except OSError:
        pass
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    script = tmp_path / "rapid-mlx"
    script.write_text(
        textwrap.dedent("""\
        import json, sys, time
        from http.server import BaseHTTPRequestHandler, HTTPServer
        port = int(sys.argv[-1])
        time.sleep(2)
        bound = time.monotonic()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                ready = time.monotonic() - bound >= 2
                payload = {"ready": ready, "phase": "loading_weights",
                           "progress": {"done": 42, "total": 100}}
                if self.path == "/v1/models":
                    payload = {"data": [{"id": "qwen3-0.6b-4bit"}]}
                body = json.dumps(payload).encode()
                self.send_response(200 if ready else 503)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()
        """)
    )
    env = dict(
        os.environ,
        HOME=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        RAPID_MLX_TELEMETRY="0",
        DO_NOT_TRACK="1",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script), "serve", "qwen3-0.6b-4bit", "--port", str(port)]
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rapid_mlx.cli",
                "agents",
                "opencode",
                "--setup",
                "--yes",
                "--model",
                "qwen3-0.6b-4bit",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        config = json.loads((tmp_path / ".config/opencode/opencode.json").read_text())
        assert f"localhost:{port}" in json.dumps(config)
        assert "qwen3-0.6b-4bit" in json.dumps(config)
    finally:
        proc.terminate()
        proc.wait(timeout=5)
