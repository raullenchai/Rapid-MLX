# SPDX-License-Identifier: Apache-2.0
"""No-Metal control-plane contracts, using real subprocesses and Unix sockets."""

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.cli_parser import build_parser
from rapid_mlx.control import RUNNER_ENV, SECRET_ENV, SECRET_HEADER, SOCKET_ENV
from rapid_mlx.control.modes import supervised_mode
from rapid_mlx.control.protocol import discover, request, runtime_root, status_command
from rapid_mlx.control.supervisor import Supervisor


@pytest.mark.parametrize(
    "argv,mode",
    [
        (["qwen3.5-4b-4bit"], "text-vlm"),
        (["qwen3.5-4b-4bit", "--lazy-load"], "lazy"),
        (
            ["qwen3.5-4b-4bit", "--embedding-model", "e", "--rerank-model", "r"],
            "text-vlm",
        ),
        (["--embedding-model", "e"], "retrieval"),
        (["--rerank-model", "r"], "retrieval"),
        (["--embedding-model", "e", "--rerank-model", "r"], "retrieval"),
        (["--cua-only"], None),
        (["cogvideox-fun-5b-q4"], None),
        (["flux3-klein"], None),
        (["diffusion-gemma-26b-4bit"], None),
        (["kokoro"], None),
        (["unknown/unqualified"], None),
        (["qwen3.5-4b-4bit", "--enable-dflash"], None),
        (["qwen3.5-4b-4bit", "--speculative-config", '{"method":"mtp"}'], None),
        (["qwen3.5-4b-4bit", "--enable-ddtree"], None),
        (["qwen3.5-4b-4bit", "--draft-model", "draft"], None),
        (["qwen3.5-4b-4bit", "--tensorfold-lanes", "2"], None),
        (["qwen3.5-4b-4bit", "--enable-audio"], None),
        (["qwen3.5-4b-4bit", "--watchdog-ppid", "123"], None),
        ([], None),
    ],
)
def test_allowlist(argv, mode):
    assert supervised_mode(build_parser().parse_args(["serve", *argv]), {}) == mode


@pytest.mark.parametrize(
    "env",
    [
        {"RAPID_MLX_WATCHDOG_PPID": "123"},
        {"RAPID_MLX_WATCHDOG_PPID": ""},
        {"RAPID_MLX_SINGLE_PROCESS": "1"},
        {RUNNER_ENV: "1"},
        {RUNNER_ENV: ""},
    ],
)
def test_fallback_environment(env):
    assert (
        supervised_mode(build_parser().parse_args(["serve", "qwen3.5-4b-4bit"]), env)
        is None
    )


def eventually(fn, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("timed out waiting for control state")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    # macOS's sockaddr_un has a 104-byte limit; use a short TMPDIR.
    import shutil
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="cp-", dir="/private/tmp"))
    monkeypatch.setenv("TMPDIR", str(base))
    runner = base / "fake.py"
    runner.write_text("""
import argparse, json, os, signal, socket, sys, time
from pathlib import Path
from rapid_mlx.control.protocol import report
p = argparse.ArgumentParser()
p.add_argument('--port', type=int)
p.add_argument('--listen-fd', type=int)
a = p.parse_args()
base = Path(os.environ['TMPDIR'])
countfile = base / 'count'
n = int(countfile.read_text()) + 1 if countfile.exists() else 1
countfile.write_text(str(n))
(base / ('argv' + str(n))).write_text(json.dumps({'argv': sys.argv, 'pgid': os.getpgrp(), 'pid': os.getpid()}))
mode = os.environ.get('FAKE_MODE', 'wait')
if mode == 'bind':
    report('bind_failed')
    sys.exit(3)
if mode == 'startup':
    report('startup_failed', reason='test')
    sys.exit(7)
if mode == 'logs':
    os.write(1, b'x' * 262144 + b'\\x00\\xff')
    os.write(2, b'y' * 262144 + b'\\x00\\xff')
    sys.exit(0)
if a.listen_fd is not None:
    listener = socket.socket(fileno=os.dup(a.listen_fd))
else:
    listener = socket.socket()
    listener.bind(('127.0.0.1', a.port))
    listener.listen()
report('bound', port=listener.getsockname()[1], ready=(mode != 'loading'))
if mode == 'crash' and n == 1:
    sys.exit(9)
if mode in ('ignore', 'loghang'):
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
else:
    def stopped(sig, frame):
        (base / 'signal').write_text(str(sig))
        sys.exit(0)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stopped)
os.write(1, b'runner-output\\n')
os.write(2, b'runner-error\\n')
(base / 'armed').touch()
if mode == 'loghang':
    os.write(1, b'z' * 1048576)
time.sleep(60)
""")
    processes = []

    def launch(mode="wait", fd=None, fast=False, service=False):
        env = os.environ.copy()
        env["FAKE_MODE"] = mode
        env["PYTHONPATH"] = str(Path.cwd())
        code = """
import sys
from rapid_mlx.control.supervisor import supervise
raise SystemExit(supervise([sys.executable, sys.argv[1]], port=0, mode='text-vlm', listen_fd=None))
"""
        if fast:
            code = code.replace(
                "raise SystemExit(supervise(",
                "from rapid_mlx.control.supervisor import Supervisor\nSupervisor.restart_delay = lambda self: 0.02\nraise SystemExit(supervise(",
            )
        if service:
            code = f"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
from rapid_mlx.headless_service import runtime
runtime.log_dir_for = lambda _: Path(os.environ['TMPDIR']) / 'logs'
config = SimpleNamespace(service_user='test', log_max_mb=1, log_backup_count=1, log_retention_days=1)
raise SystemExit(runtime._supervise([sys.executable, '-c', {code!r}, sys.argv[1]], os.environ.copy(), config))
"""
        argv = [sys.executable, "-c", code, str(runner)]
        if fd is not None:
            code = code.replace(
                "[sys.executable, sys.argv[1]]",
                "[sys.executable, sys.argv[1], '--listen-fd', sys.argv[2]]",
            ).replace("listen_fd=None", "listen_fd=int(sys.argv[2])")
            argv = [sys.executable, "-c", code, str(runner), str(fd)]
        proc = subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(() if fd is None else (fd,)),
        )
        processes.append(proc)
        return proc

    yield base, launch
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.communicate(timeout=6)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
    shutil.rmtree(base)


def live_status():
    statuses = discover()
    return statuses[0] if statuses else None


def test_socket_permissions_secret_and_status_cli(fake, capsys):
    base, launch = fake
    proc = launch("loading")
    status = eventually(live_status)
    assert status["runner"]["state"] == "starting"
    directory = next(runtime_root().glob("i-*"))
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "control.sock").stat().st_mode) == 0o600
    assert stat.S_IMODE((directory / "instance.json").stat().st_mode) == 0o600
    secret = json.loads((directory / "instance.json").read_text())["secret"]
    with pytest.raises(OSError):
        request(directory / "control.sock", "wrong")
    child = json.loads(
        eventually(
            lambda: (base / "argv1").read_text() if (base / "argv1").exists() else None
        )
    )
    assert secret not in str(child["argv"])
    assert child["pgid"] == proc.pid
    assert status_command(SimpleNamespace(port=None)) == 0
    assert json.loads(capsys.readouterr().out)["control_version"] == 1
    result = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "status"],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runner"]["state"] == "starting"
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=6)
    assert secret.encode() not in stdout + stderr
    assert not directory.exists()


def test_crash_restart_pins_ephemeral_port_and_status_stays_live(fake):
    base, launch = fake
    proc = launch("crash")

    def restarting():
        status = live_status()
        return status if status and status["runner"]["state"] == "restarting" else None

    first = eventually(restarting)
    backoff_started = time.monotonic()
    assert first["port"] > 0

    def ready():
        status = live_status()
        return (
            status
            if status
            and status["runner"]["restarts"] == 1
            and status["runner"]["state"] == "ready"
            else None
        )

    second = eventually(ready)
    assert time.monotonic() - backoff_started >= 0.8
    assert second["port"] == first["port"]
    assert json.loads((base / "argv2").read_text())["argv"][-1] == str(first["port"])
    proc.terminate()
    proc.communicate(timeout=6)


def test_bind_failed_never_restarts(fake):
    base, launch = fake
    proc = launch("bind")
    _, err = proc.communicate(timeout=5)
    assert proc.returncode == 3, err
    assert (base / "count").read_text() == "1"


def test_binary_logs_are_drained_verbatim(fake):
    _, launch = fake
    proc = launch("logs")
    stdout, stderr = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert stdout == b"x" * 262144 + b"\x00\xff"
    assert stderr == b"y" * 262144 + b"\x00\xff"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_signal_forward_and_reap(fake, sig):
    base, launch = fake
    proc = launch()
    eventually(lambda: (base / "armed").exists())
    child_pid = json.loads((base / "argv1").read_text())["pid"]
    proc.send_signal(sig)
    proc.communicate(timeout=6)
    assert proc.returncode == 128 + sig
    assert int((base / "signal").read_text()) == sig
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_four_second_kill_deadline_double_signal_and_process_group(fake):
    base, launch = fake
    proc = launch("ignore")
    eventually(lambda: (base / "armed").exists())
    before = time.monotonic()
    os.killpg(proc.pid, signal.SIGTERM)
    time.sleep(0.2)
    proc.send_signal(signal.SIGINT)
    proc.communicate(timeout=6)
    elapsed = time.monotonic() - before
    assert 3.8 <= elapsed < 5.5
    assert proc.returncode == 128 + signal.SIGTERM
    assert (base / "count").read_text() == "1"


def test_listen_fd_retained_across_restarts(fake):
    _, launch = fake
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        proc = launch("crash", listener.fileno())

        def ready():
            status = live_status()
            return (
                status
                if status
                and status["runner"]["restarts"] == 1
                and status["runner"]["state"] == "ready"
                else None
            )

        assert eventually(ready)["port"] == listener.getsockname()[1]
        proc.terminate()
        proc.communicate(timeout=6)


def test_control_imports_do_not_import_mlx():
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.startswith(('mlx', 'rapid_mlx.engine', 'rapid_mlx.server')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import rapid_mlx.control.modes, rapid_mlx.control.protocol, rapid_mlx.control.supervisor
import rapid_mlx.cli
"""
    assert (
        subprocess.run([sys.executable, "-c", code], capture_output=True).returncode
        == 0
    )


@pytest.mark.parametrize(
    "peer,headers,expected",
    [
        (
            "192.168.1.2",
            {"Authorization": "Bearer key", "X-Forwarded-For": "127.0.0.1"},
            403,
        ),
        ("127.0.0.1", {}, 401),
        ("127.0.0.1", {"Authorization": "Bearer wrong"}, 401),
        ("127.0.0.1", {"Authorization": "Bearer key", "x-api-key": "wrong"}, 401),
        (
            "127.0.0.1",
            {"Authorization": "Bearer key", SECRET_HEADER: "client-secret"},
            200,
        ),
        ("::1", {"x-api-key": "key"}, 200),
        ("::ffff:127.0.0.1", {"x-api-key": "key"}, 200),
    ],
)
def test_forwarder_security(monkeypatch, peer, headers, expected):
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import control_status

    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", "key")
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "/fake/socket")
    monkeypatch.setenv(SECRET_ENV, "startup-secret")
    seen = []

    def query(path, secret):
        seen.append((path, secret))
        return {"control_version": 1}

    monkeypatch.setattr(control_status, "control_request", query)
    app = FastAPI()
    control_status.mount_status(app)
    with TestClient(app, client=(peer, 1234)) as client:
        response = client.get("/rapid/v1/status", headers=headers)
        assert response.status_code == expected
        assert client.get("/rapid/v1/anything").status_code == 404
        assert client.post("/rapid/v1/status").status_code == 405
    assert seen == ([("/fake/socket", "startup-secret")] if expected == 200 else [])


def test_forwarder_body_bound_and_fallback_absent(monkeypatch):
    from rapid_mlx.config import get_config
    from rapid_mlx.routes.control_status import mount_status

    monkeypatch.setattr(get_config(), "api_key", None)
    monkeypatch.delenv(RUNNER_ENV, raising=False)
    app = FastAPI()
    mount_status(app)
    with TestClient(app, client=("127.0.0.1", 1)) as client:
        assert client.get("/rapid/v1/status").status_code == 404
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "/fake")
    monkeypatch.setenv(SECRET_ENV, "secret")
    mount_status(app)
    with TestClient(app, client=("127.0.0.1", 1)) as client:
        assert (
            client.request("GET", "/rapid/v1/status", content=b"x" * 16385).status_code
            == 413
        )


def test_status_no_instance(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert status_command(SimpleNamespace(port=None)) == 1
    assert "No supervised" in capsys.readouterr().err


def test_retry_backoff_cap():
    supervisor = Supervisor([], port=8000, mode="text-vlm")
    delays = []
    for index in range(10):
        supervisor.restarts = index
        delays.append(supervisor.restart_delay())
    assert delays == [1, 2, 4, 8, 16, 32, 60, 60, 60, 60]


@pytest.mark.parametrize(
    "kind", ["desktop-env", "desktop-arg", "single", "unknown", "runner"]
)
def test_dispatch_fallback_preserves_namespace_environment_and_fds(
    monkeypatch, capsys, kind
):
    from rapid_mlx import cli
    from rapid_mlx.control import protocol, supervisor

    for key in (RUNNER_ENV, "RAPID_MLX_WATCHDOG_PPID", "RAPID_MLX_SINGLE_PROCESS"):
        monkeypatch.delenv(key, raising=False)
    argv = ["serve", "qwen3.5-4b-4bit", "--listen-fd", "3"]
    if kind == "desktop-env":
        monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", "123")
    elif kind == "desktop-arg":
        argv += ["--watchdog-ppid", "123"]
    elif kind == "single":
        monkeypatch.setenv("RAPID_MLX_SINGLE_PROCESS", "1")
    elif kind == "runner":
        monkeypatch.setenv(RUNNER_ENV, "1")
        monkeypatch.setattr(protocol, "install_startup_report", lambda: None)
    else:
        argv[1] = "unknown/target"
    args = build_parser().parse_args(argv)
    before_args, before_env = vars(args).copy(), os.environ.copy()

    def forbidden(*a, **kw):
        pytest.fail("fallback must not resolve ports, spawn or allocate sockets")

    monkeypatch.setattr(cli, "_resolve_serve_port", forbidden)
    monkeypatch.setattr(supervisor, "supervise", forbidden)
    monkeypatch.setattr(Supervisor, "open", forbidden)
    cli._dispatch_serve_control(args)
    assert vars(args) == before_args
    assert dict(os.environ) == before_env
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        "Control functionality is unavailable in this mode.\n"
        if kind == "unknown"
        else ""
    )


def test_dispatch_resolves_auto_port_once_before_spawn(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.control import supervisor

    for key in (RUNNER_ENV, "RAPID_MLX_WATCHDOG_PPID", "RAPID_MLX_SINGLE_PROCESS"):
        monkeypatch.delenv(key, raising=False)
    args = build_parser().parse_args(["serve", "qwen3.5-4b-4bit"])
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "qwen3.5-4b-4bit"])
    calls = []

    def resolve(host, port, **kw):
        calls.append(("resolve", port))
        return 8007

    def spawn(argv, **kw):
        calls.append(("spawn", kw["port"]))
        assert argv == [
            sys.executable,
            "-m",
            "rapid_mlx.cli",
            "serve",
            "qwen3.5-4b-4bit",
        ]
        return 13

    monkeypatch.setattr(cli, "_resolve_serve_port", resolve)
    monkeypatch.setattr(supervisor, "supervise", spawn)
    with pytest.raises(SystemExit) as exited:
        cli._dispatch_serve_control(args)
    assert exited.value.code == 13
    assert calls == [("resolve", None), ("spawn", 8007)]


def test_forwarder_preserves_transport_peer_with_proxy_headers(monkeypatch):
    from fastapi import Request

    from rapid_mlx import _uvicorn
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import control_status

    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "/fake")
    monkeypatch.setenv(SECRET_ENV, "secret")
    monkeypatch.setattr(get_config(), "api_key", None)
    monkeypatch.setattr(control_status, "control_request", lambda *a: {"ok": True})
    app = FastAPI()
    control_status.mount_status(app)

    @app.get("/v1/peer")
    async def peer(request: Request):
        return {"peer": request.client.host}

    captured = []

    class Server:
        started = True

        def __init__(self, config, **kwargs):
            captured.append(config.app)

        def run(self):
            pass

    monkeypatch.setattr(_uvicorn, "AcceptingConnectionsServer", Server)
    _uvicorn._run_server(app, forwarded_allow_ips="*", proxy_headers=True)
    with TestClient(captured[0], client=("192.168.1.2", 1234)) as client:
        headers = {"X-Forwarded-For": "127.0.0.1"}
        assert client.get("/rapid/v1/status", headers=headers).status_code == 403
        assert client.get("/v1/peer", headers=headers).json() == {"peer": "127.0.0.1"}


@pytest.mark.parametrize(
    "alias",
    [
        "deepseek-v4-flash-tensorfold",
        "deepseek-v41-flash-reap-2bit",
        "deepseek-v41-flash-1017",
    ],
)
def test_alternate_catalog_servers_fall_back(alias):
    assert supervised_mode(build_parser().parse_args(["serve", alias]), {}) is None


def test_five_startup_failures_give_up_nonzero(fake):
    base, launch = fake
    proc = launch("startup", fast=True)
    proc.communicate(timeout=8)
    assert proc.returncode == 7
    assert (base / "count").read_text() == "5"
    assert not discover()


def test_no_restart_during_backoff_shutdown(fake):
    base, launch = fake
    proc = launch("crash")
    eventually(
        lambda: (status := live_status()) and status["runner"]["state"] == "restarting"
    )
    proc.terminate()
    proc.communicate(timeout=6)
    assert (base / "count").read_text() == "1"


def test_service_supervisor_runner_signal_and_log_relay(fake):
    base, launch = fake
    proc = launch(service=True)
    eventually(lambda: (base / "armed").exists())
    status = eventually(live_status)
    assert status["runner"]["pid"] != proc.pid
    assert json.loads((base / "argv1").read_text())["pgid"] == proc.pid
    proc.terminate()
    proc.communicate(timeout=6)
    assert proc.returncode == 143
    assert (base / "count").read_text() == "1"
    assert (base / "signal").read_text() == str(signal.SIGTERM)
    assert (base / "logs" / "server.stdout.log").read_bytes() == b"runner-output\n"
    assert (base / "logs" / "server.stderr.log").read_bytes() == b"runner-error\n"


def test_failure_window_expires_after_ten_minutes():
    supervisor = Supervisor([], port=8000, mode="text-vlm")
    for now in (0, 1, 2, 3):
        assert not supervisor.record_failure(now)
    assert not supervisor.record_failure(604)
    assert list(supervisor.failures) == [604]
    for now in (605, 606, 607):
        assert not supervisor.record_failure(now)
    assert supervisor.record_failure(608)


def test_stalled_log_consumer_cannot_hold_shutdown_open(fake):
    base, launch = fake
    proc = launch("loghang")
    eventually(lambda: (base / "armed").exists())
    proc.terminate()
    # Deliberately do not consume the parent's stdout until it has exited.
    assert proc.wait(timeout=6) == 143
    proc.communicate(timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "bind", "startup"])
async def test_production_uvicorn_startup_reports(monkeypatch, failure):
    import uvicorn

    from rapid_mlx import _signal_observability, _uvicorn
    from rapid_mlx.control import protocol
    from rapid_mlx.telemetry import server_start

    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setattr(protocol, "_startup_terminal", False)
    events = []
    monkeypatch.setattr(
        protocol, "report", lambda event, **kw: events.append((event, kw))
    )
    monkeypatch.setattr(server_start, "failed", lambda *a, **kw: None)
    monkeypatch.setattr(server_start, "ready", lambda: None)
    monkeypatch.setattr(_signal_observability, "ensure_crash_sink", lambda: None)
    server = _uvicorn.AcceptingConnectionsServer(
        uvicorn.Config(FastAPI(), host="127.0.0.1", port=0)
    )
    server.lifespan = SimpleNamespace(should_exit=failure == "startup")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()

        async def startup(sockets):
            if failure:
                raise SystemExit(3)
            server.started = True
            server.servers = [SimpleNamespace(sockets=[listener])]

        monkeypatch.setattr(server, "_startup_with_progress", startup)
        if failure:
            with pytest.raises(SystemExit):
                await server.startup()
            assert events[0][0] == (
                "bind_failed" if failure == "bind" else "startup_failed"
            )
        else:
            await server.startup()
            assert events == [("bound", {"port": listener.getsockname()[1]})]
