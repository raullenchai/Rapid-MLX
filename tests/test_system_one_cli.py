# SPDX-License-Identifier: Apache-2.0
import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from rapid_mlx.cli import _resolve_system_one_backend, build_parser, system_one_command


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(self.rfile.read(length))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_system_one_cli_defaults_to_laya_service():
    args = build_parser().parse_args(["system-one"])
    assert args.model == "convaiinnovations/laya"
    assert args.backend == "auto"
    assert args.host == "127.0.0.1"
    assert args.port == 8700
    assert args._port_explicit is False
    assert args.max_concurrent_requests == 8


def test_system_one_busy_explicit_port_reports_context_to_loopback_sink(
    tmp_path,
):
    home = tmp_path / "home"
    home.mkdir()
    sink = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    sink.bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()

    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        program = f"""
import sys
import types

import rapid_mlx
from rapid_mlx import cli
from rapid_mlx.telemetry import build_gate, common_props, consent_runtime, posthog_sender, server_start, state
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts

rapid_mlx.__version__ = "0.15.1"
stamp = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
build_gate.official_build = lambda: stamp
consent_runtime.upload_allowed = lambda: True
common_props.read_platform_facts = lambda: PlatformFacts(
    os="darwin", os_version="25.3", arch="arm64", chip="m1-pro",
    memory_gb=32, python_version="3.11"
)
state.get_or_create_client_id = lambda: "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
state.session_id = lambda: "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"

class Backend:
    default_model = "fake-system-one"

    def __init__(self, *_args, **_kwargs):
        pass

    def models(self):
        return []

backends = types.ModuleType("rapid_mlx.system_one.backends")
backends.CLMBackend = Backend
backends.DecisionBackend = Backend
backends.LayaBackend = Backend
sys.modules[backends.__name__] = backends

async def app(scope, receive, send):
    if scope["type"] != "lifespan":
        return
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({{"type": "lifespan.startup.complete"}})
        elif message["type"] == "lifespan.shutdown":
            await send({{"type": "lifespan.shutdown.complete"}})
            return

server = types.ModuleType("rapid_mlx.system_one.server")
server.create_app = lambda *_args, **_kwargs: app
sys.modules[server.__name__] = server

cli._port_preflight_or_die = lambda *_args, **_kwargs: None
args = cli.build_parser().parse_args(["system-one", "--port", "{port}"])
server_start.attempted("system-one", load_policy="eager")
try:
    cli.system_one_command(args)
finally:
    posthog_sender.get_sender().flush(5.0)
"""
        env = dict(
            os.environ,
            HOME=str(home),
            RAPID_MLX_POSTHOG_URL=f"http://127.0.0.1:{sink.server_port}/batch/",
        )
        for name in ("CI", "GITHUB_ACTIONS", "RAPID_MLX_TELEMETRY", "DO_NOT_TRACK"):
            env.pop(name, None)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", program],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        finally:
            sink.shutdown()
            thread.join(timeout=2.0)
            sink.server_close()

    assert proc.returncode != 0, proc.stderr
    start_events = [
        item
        for body in sink.bodies  # type: ignore[attr-defined]
        for item in json.loads(body)["batch"]
        if item["event"] == "server_start_state"
    ]
    assert [item["properties"]["state"] for item in start_events] == [
        "attempted",
        "failed",
    ]
    failure = start_events[-1]["properties"]
    assert failure["failure_stage"] == "bind"
    assert failure["port_explicit"] is True


def test_system_one_cli_accepts_clm_runtime_inputs():
    args = build_parser().parse_args(
        [
            "system-one",
            "clm-latest",
            "--backend",
            "clm",
            "--head",
            "/tmp/head",
            "--encoder",
            "Qwen/Qwen3-8B",
        ]
    )
    assert args.head == "/tmp/head"
    assert args.encoder == "Qwen/Qwen3-8B"


def test_server_parsers_stamp_port_context_at_parse_time():
    from rapid_mlx import server

    parser = build_parser()
    implicit = parser.parse_args(["serve", "qwen3.5-4b-4bit"])
    explicit = parser.parse_args(["serve", "qwen3.5-4b-4bit", "--port", "8123"])
    inherited = parser.parse_args(["serve", "qwen3.5-4b-4bit", "--listen-fd", "7"])
    standalone = server._build_parser()

    assert implicit._port_explicit is False
    assert explicit._port_explicit is True
    assert inherited._port_explicit is None
    assert standalone.parse_args([])._port_explicit is False
    assert standalone.parse_args(["--port=8123"])._port_explicit is True


def test_system_one_cli_rejects_non_positive_laya_batch_size():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["system-one", "--batch-size", "0"])


def test_system_one_cli_rejects_backend_specific_argument_mismatches():
    with pytest.raises(SystemExit, match="CLM requires --head"):
        system_one_command(
            build_parser().parse_args(["system-one", "clm", "--backend", "clm"])
        )
    with pytest.raises(SystemExit, match="only valid with --backend clm"):
        system_one_command(
            build_parser().parse_args(
                ["system-one", "--backend", "laya", "--head", "x"]
            )
        )


def test_system_one_auto_backend_does_not_substring_match_clm():
    assert _resolve_system_one_backend("org/aclmish-laya", "auto") == "laya"
    assert _resolve_system_one_backend("Contrastive-LM/CLM-v0.1-8B", "auto") == "clm"


def test_system_one_extra_is_optional_and_included_in_all():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert all(not item.startswith("laya-mlx") for item in project["dependencies"])
    extras = project["optional-dependencies"]
    assert any(item.startswith("laya-mlx") for item in extras["system-one"])
    assert any(item.startswith("laya-mlx") for item in extras["all"])


def test_system_one_checks_port_before_backend_initialization(monkeypatch):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.system_one.backends as backend_module

    events = []

    class Backend:
        default_model = "laya"

        def __init__(self, *args, **kwargs):
            events.append("backend")

        def models(self):
            return []

    # Patch the exact globals mapping used by the imported command. Some of
    # the full-suite CLI tests reload ``rapid_mlx.cli`` during collection,
    # so patching the current sys.modules entry can target a newer module
    # object than this function references.
    monkeypatch.setitem(
        system_one_command.__globals__,
        "_port_preflight_or_die",
        lambda *args, **kwargs: events.append("port"),
    )
    monkeypatch.setattr(backend_module, "LayaBackend", Backend)
    monkeypatch.setattr(
        uvicorn_module, "run_uvicorn", lambda *args, **kwargs: events.append("serve")
    )
    system_one_command(build_parser().parse_args(["system-one"]))
    assert events == ["port", "backend", "serve"]


def test_system_one_passes_public_clm_model_name(monkeypatch):
    import rapid_mlx._uvicorn as uvicorn_module
    import rapid_mlx.system_one.backends as backend_module

    captured = {}

    class Backend:
        default_model = "public-clm"

        def __init__(self, encoder, head, **kwargs):
            captured.update(encoder=encoder, head=head, **kwargs)

        def models(self):
            return []

    monkeypatch.setitem(
        system_one_command.__globals__,
        "_port_preflight_or_die",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(backend_module, "CLMBackend", Backend)
    monkeypatch.setattr(uvicorn_module, "run_uvicorn", lambda *args, **kwargs: None)
    args = build_parser().parse_args(
        [
            "system-one",
            "public-clm",
            "--backend",
            "clm",
            "--head",
            "/tmp/head",
            "--device",
            "cpu",
        ]
    )
    system_one_command(args)
    assert captured["model_name"] == "public-clm"
    assert captured["device"] == "cpu"


def test_main_dispatches_system_one(monkeypatch):
    import rapid_mlx.cli as cli_module

    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "system-one",
            "clm-latest",
            "--backend",
            "clm",
            "--head",
            "/tmp/head",
        ],
    )
    monkeypatch.setattr(
        cli_module, "system_one_command", lambda args: captured.append(args.model)
    )
    cli_module.main()
    assert captured == ["clm-latest"]


def test_main_still_dispatches_serve_after_system_one_branch(monkeypatch):
    import rapid_mlx.cli as cli_module

    captured = []
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "qwen3.5-4b-4bit"])
    monkeypatch.setattr(
        cli_module, "serve_command", lambda args: captured.append(args.command)
    )
    cli_module.main()
    assert captured == ["serve"]
