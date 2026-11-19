"""Read-only terminal monitor contract, using an actual local HTTP server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from rapid_mlx.cli_parser import build_parser


def test_top_parser():
    args = build_parser().parse_args(["top", "--port", "8123", "--once"])
    assert (args.port, args.once) == (8123, True)


@pytest.fixture
def server():
    payloads = {
        "/v1/status": {
            "status": "generating",
            "uptime_s": 10,
            "num_waiting": 2,
            "total_requests_processed": 3,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 20,
            "metal": {"active_memory_gb": 2, "peak_memory_gb": 3},
            "cache": {"hits": 3, "misses": 1},
            "requests": [
                {
                    "phase": "generation",
                    "prompt_tokens": 100,
                    "tokens_per_second": 25,
                    "elapsed_s": 4,
                    "prompt": "SECRET",
                    "response": "SECRET",
                    "request_id": "SECRET",
                    "progress": 0.5,
                }
            ],
        },
        "/v1/models/residency": {
            "models": [
                {"id": "org/model", "state": "resident", "measured_bytes": 2**20},
                {
                    "id": "/private/SECRET",
                    "state": "resident",
                    "estimated_bytes": 2**21,
                },
                {"id": "SECRET", "state": "registered"},
            ]
        },
        "/metrics": "rapid_mlx_prefix_cache_current_bytes 1048576\n",
    }
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.command, self.path, self.headers.get("Authorization")))
            data = payloads[self.path]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                (data if isinstance(data, str) else json.dumps(data)).encode()
            )

        def log_message(self, *_):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield http.server_port, payloads, calls
    http.shutdown()
    http.server_close()
    thread.join()


def test_active_idle_and_stopped(server, capsys, monkeypatch):
    from rapid_mlx.top import top_command

    port, payloads, calls = server
    monkeypatch.setenv("RAPID_MLX_API_KEY", "test-key")
    args = SimpleNamespace(port=port, once=True)
    top_command(args)
    text = capsys.readouterr().out
    assert "generating" in text
    assert "generation | 100.0 | 25.0 | 4.0 | - | -" in text
    assert "org/model | 1.0 (measured)" in text
    assert "[local/unknown model] | 2.0 (estimated)" in text
    assert "MiB 1.0 | hit % 75.0" in text
    assert "active 2.0 | peak 3.0" in text
    assert "queue 2.0" in text
    assert "SECRET" not in text and "\033" not in text
    assert calls == [("GET", path, "Bearer test-key") for path in payloads]
    payloads["/v1/status"] = {"status": "idle", "requests": []}
    top_command(args)
    assert "idle" in capsys.readouterr().out
    # The same client works for serve and service: only the HTTP contract matters.
    import socket

    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        stopped_port = closed.getsockname()[1]
    top_command(SimpleNamespace(port=stopped_port, once=True))
    assert "stopped/unreachable" in capsys.readouterr().out


def test_render_rates_and_missing_fields():
    from rapid_mlx.top import render

    def sample(uptime, count):
        return {
            "status": {
                "uptime_s": uptime,
                "total_requests_processed": count,
                "total_prompt_tokens": count * 10,
                "total_completion_tokens": count * 2,
            },
            "metrics": {},
            "residency": {},
            "error": "",
            "metrics_error": "",
            "residency_error": "",
        }

    before = sample(10, 2)
    after = sample(12, 4)
    assert "requests/min 60.0 | in tok/s 10.0 | out tok/s 2.0" in render(
        after, before, 8000
    )
    for current in (sample(1, 1), sample(11, 1), sample(None, 1)):
        assert "requests/min -" in render(current, before, 8000)
    after["error"] = "HTTP 401"
    assert "HTTP 401" in render(after, before, 8000)
    after["status"]["requests"] = [{"phase": "SECRET"}]
    assert "unknown" in render(after, before, 8000)


def test_partial_failures_and_invalid_payloads():
    import httpx

    from rapid_mlx.top import _metrics, _model, _number, render, snapshot

    def handler(request):
        if request.url.path == "/v1/status":
            return httpx.Response(200, text="bad JSON SECRET")
        if request.url.path == "/metrics":
            return httpx.Response(403, text="SECRET")
        return httpx.Response(200, json=[])

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://localhost"
    ) as client:
        result = snapshot(client)
    assert "SECRET" not in render(result, None, 8000)
    assert "metrics: HTTP 403" in render(result, None, 8000)
    assert result["residency"] == {}
    assert _metrics("# comment\nrapid_mlx_x bad\nrapid_mlx_y NaN\nrapid_mlx_z 3\n") == {
        "rapid_mlx_z": 3
    }
    assert _model("org/m\033odel") == "org/model"
    assert _number(float("inf")) is None


def test_tty_refresh_and_pipe_fallback(server, monkeypatch, capsys):
    import rapid_mlx.top as top

    port, _, calls = server
    monkeypatch.setattr(top.sys.stdout, "isatty", lambda: True)
    ticks = []

    def sleep(_):
        ticks.append(1)
        if len(ticks) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(top.time, "sleep", sleep)
    top.top_command(SimpleNamespace(port=port, once=False))
    assert len(calls) == 6
    assert capsys.readouterr().out.count("\033[2J\033[H") == 2
    monkeypatch.setattr(top.sys.stdout, "isatty", lambda: False)
    top.top_command(SimpleNamespace(port=port, once=False))
    assert len(calls) == 9
    assert "\033" not in capsys.readouterr().out


def test_cli_dispatch_is_read_only(server, monkeypatch, capsys):
    import rapid_mlx.cli as cli

    port, _, calls = server
    monkeypatch.setattr("sys.argv", ["rapid-mlx", "top", "--once", "--port", str(port)])

    def forbidden(*_, **__):
        pytest.fail("monitor must bypass lifecycle/telemetry startup")

    monkeypatch.setattr(cli, "_start_v2_lifecycle", forbidden)
    monkeypatch.setattr("rapid_mlx.telemetry.consent_runtime.startup", forbidden)
    cli.main()
    assert len(calls) == 3
    assert "generating" in capsys.readouterr().out


@pytest.mark.parametrize("port", ["0", "-1", "65536", "bad"])
def test_top_invalid_port(port):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["top", "--port", port, "--once"])
    assert exc.value.code == 2
