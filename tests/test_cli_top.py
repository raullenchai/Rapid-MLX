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
            self.send_response(data if isinstance(data, int) else 200)
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


@pytest.mark.parametrize("once", [True, False])
def test_active_idle_and_stopped(server, capsys, monkeypatch, once):
    from rapid_mlx.top import top_command

    port, payloads, calls = server
    monkeypatch.setenv("RAPID_MLX_API_KEY", "test-key")
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    args = SimpleNamespace(port=port, once=once)
    top_command(args)
    text = capsys.readouterr().out
    assert "generating" in text
    assert "generation | 100 | 25.0 | 4.0\n" in text
    assert "prefill %" not in text and "client route" not in text
    assert "org/model | 1.0 (measured)" in text
    assert "[local/unknown model] | 2.0 (estimated)" in text
    assert "MiB 1.0 | hit % 75.0" in text
    assert "active 2.0 | peak 3.0" in text
    assert "queue 2\n" in text
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
    with pytest.raises(SystemExit) as exc:
        top_command(SimpleNamespace(port=stopped_port, once=once))
    assert exc.value.code == 1
    assert capsys.readouterr().out == (
        f"No Rapid-MLX server at 127.0.0.1:{stopped_port}. "
        "Start one with: rapid-mlx serve <model>\n"
    )


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
    after["error"] = ""
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
    assert render(result, None, 8000).count("\n") == 0
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


@pytest.mark.parametrize("once", [True, False])
@pytest.mark.parametrize("key", [None, "wrong-key"])
@pytest.mark.parametrize("path", ["/v1/status", "/v1/models/residency", "/metrics"])
def test_auth_failure_is_one_line(server, monkeypatch, capsys, once, key, path):
    from rapid_mlx.top import top_command

    port, payloads, calls = server
    payloads[path] = 401
    if key is None:
        monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_API_KEY", key)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    with pytest.raises(SystemExit) as exc:
        top_command(SimpleNamespace(port=port, once=once))
    assert exc.value.code == 1
    assert capsys.readouterr().out == (
        f"Rapid-MLX server at 127.0.0.1:{port}: HTTP 401 Unauthorized. "
        "Set RAPID_MLX_API_KEY to the server's API key.\n"
    )
    assert all(call[2] == (f"Bearer {key}" if key else None) for call in calls)


def test_tty_recovers_from_unreachable_and_unauthorized(server, monkeypatch, capsys):
    import httpx

    import rapid_mlx.top as top

    port, payloads, _ = server
    original_fetch = top._fetch
    ticks = []

    def fetch(client, path):
        if not ticks:
            # Exercise the real network-error handling with a failing transport.
            def disconnected(request):
                raise httpx.ConnectError("SECRET")

            with httpx.Client(
                transport=httpx.MockTransport(disconnected), base_url="http://localhost"
            ) as offline:
                return original_fetch(offline, path)
        return original_fetch(client, path)

    def sleep(_):
        ticks.append(1)
        payloads["/v1/status"] = 401 if len(ticks) == 1 else {"status": "idle"}
        if len(ticks) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(top, "_fetch", fetch)
    monkeypatch.setattr(top.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(top.time, "sleep", sleep)
    top.top_command(SimpleNamespace(port=port, once=False))
    frames = capsys.readouterr().out.split("\033[2J\033[H")[1:]
    assert len(frames) == 3
    assert frames[0].startswith("No Rapid-MLX server") and frames[0].count("\n") == 1
    assert "HTTP 401" in frames[1] and frames[1].count("\n") == 1
    assert "| idle" in frames[2] and "Requests:" in frames[2]
    assert all("SECRET" not in frame for frame in frames)


def test_partial_endpoint_failure_still_shows_available_data(server, capsys):
    from rapid_mlx.top import top_command

    port, payloads, _ = server
    payloads["/metrics"] = 403
    top_command(SimpleNamespace(port=port, once=True))
    text = capsys.readouterr().out
    assert "metrics: HTTP 403" in text and "generation | 100 |" in text


def test_status_http_failure_is_one_line(server, capsys):
    from rapid_mlx.top import top_command

    port, payloads, _ = server
    payloads["/v1/status"] = 503
    with pytest.raises(SystemExit) as exc:
        top_command(SimpleNamespace(port=port, once=True))
    assert exc.value.code == 1
    assert (
        capsys.readouterr().out == f"Rapid-MLX server at 127.0.0.1:{port}: HTTP 503.\n"
    )
