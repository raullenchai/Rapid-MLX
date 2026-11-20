# SPDX-License-Identifier: Apache-2.0
"""No-Metal control-plane contracts, using real subprocesses and Unix sockets."""

import fcntl
import json
import os
import pty
import select
import signal
import socket
import stat
import subprocess
import sys
import termios
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
    monkeypatch.setenv("HOME", str(base))
    runner = base / "fake.py"
    runner.write_text("""
import argparse, json, os, signal, socket, sys, time
from pathlib import Path
from rapid_mlx.control.protocol import report
from rapid_mlx.cli import _dispatch_serve_control
from types import SimpleNamespace
_dispatch_serve_control(SimpleNamespace())
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
if mode == 'prompt':
    from rapid_mlx._download_gate import confirm_or_abort
    assert confirm_or_abort('test/no-download', 11 * 1024**3, auto_yes_env='TEST_UNUSED_AUTO_YES')
    (base / 'accepted').touch()
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
if mode in ('grandchild', 'grandchild-term'):
    import subprocess
    child = subprocess.Popen([sys.executable, '-c',
        'import signal,time,sys; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(sys.argv[1]).touch(); time.sleep(60)', str(base / 'grandchild-ready')])
    while not (base / 'grandchild-ready').exists():
        time.sleep(0.01)
    (base / 'grandchild').write_text(str(child.pid))
if mode in ('ignore', 'loghang', 'grandchild'):
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

    def launch(
        mode="wait", fd=None, fast=False, service=False, injection="", terminal=False
    ):
        env = os.environ.copy()
        env["FAKE_MODE"] = mode
        env["PYTHONPATH"] = str(Path.cwd())
        code = """
import sys
from rapid_mlx.control.supervisor import supervise
raise SystemExit(supervise([sys.executable, sys.argv[1]], port=0, mode='text-vlm', listen_fd=None))
"""
        code = code.replace("raise SystemExit", injection + "\nraise SystemExit", 1)
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
        if terminal:
            master, slave = pty.openpty()

            def foreground():
                os.setsid()
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        proc = subprocess.Popen(
            argv,
            env=env,
            stdin=slave if terminal else None,
            stdout=slave if terminal else subprocess.PIPE,
            stderr=slave if terminal else subprocess.PIPE,
            start_new_session=not terminal,
            preexec_fn=foreground if terminal else None,
            pass_fds=(() if fd is None else (fd,)),
        )
        proc.terminal = master if terminal else None
        if terminal:
            os.close(slave)
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
        if proc.terminal is not None:
            os.close(proc.terminal)
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
    assert child["pgid"] == os.getpgid(proc.pid)
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
    _, stderr = proc.communicate(timeout=6)
    elapsed = time.monotonic() - before
    assert 3.8 <= elapsed < 4.5
    assert proc.returncode == 128 + signal.SIGTERM, stderr.decode(errors="replace")
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
    monkeypatch.setenv("HOME", str(tmp_path))
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
    assert json.loads((base / "argv1").read_text())["pgid"] == os.getpgid(proc.pid)
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


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--", "qwen3.5-4b-4bit"],
        ["serve", "--port", "8000", "--", "qwen3.5-4b-4bit"],
        ["serve", "qwen3.5-4b-4bit", "--port=8000"],
    ],
)
def test_pinned_port_before_end_of_options(argv):
    supervisor = Supervisor(argv, port=8017, mode="text-vlm")
    args = build_parser().parse_args(supervisor.spawn_argv())
    assert args.port == 8017
    assert args.model == "qwen3.5-4b-4bit"


def test_sigkill_supervisor_runner_watchdog(fake):
    import psutil

    base, launch = fake
    proc = launch()
    eventually(lambda: (base / "armed").exists())
    runner_pid = json.loads((base / "argv1").read_text())["pid"]
    before = time.monotonic()
    proc.kill()
    proc.communicate(timeout=3)
    eventually(lambda: (base / "signal").exists(), timeout=2.8)
    eventually(lambda: not psutil.pid_exists(runner_pid), timeout=0.5)
    assert (base / "signal").read_text() == str(signal.SIGTERM)
    assert time.monotonic() - before < 2.8


@pytest.mark.parametrize("mode", ["grandchild", "grandchild-term"])
def test_grandchild_inherited_pipes_shutdown_bounded(fake, mode):
    import psutil

    base, launch = fake
    proc = launch(mode)
    eventually(lambda: (base / "armed").exists())
    grandchild = int((base / "grandchild").read_text())
    before = time.monotonic()
    proc.terminate()
    proc.communicate(timeout=4.5)
    assert time.monotonic() - before < 4.5
    assert proc.returncode == 143
    assert (
        not psutil.pid_exists(grandchild)
        or psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE
    )


def test_status_cross_tmpdir_and_stale_cleanup(fake, monkeypatch, tmp_path):
    from rapid_mlx.control.protocol import discovery_root

    _, launch = fake
    proc = launch()
    first = eventually(
        lambda: (s := live_status()) and s["runner"]["state"] == "ready" and s
    )
    index = discovery_root()
    assert stat.S_IMODE(index.stat().st_mode) == 0o700
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert discover() == [first]
    result = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "status"],
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["port"] == first["port"]
    stale = index / "stale.json"
    stale.write_text(json.dumps({"socket": str(index / "gone.sock"), "secret": "gone"}))
    stale.chmod(0o600)
    discover()
    assert not stale.exists()
    proc.terminate()
    proc.communicate(timeout=5)
    assert not list(index.glob("*.json"))


def test_process_discovery_deduplicates_and_uses_bound_port(monkeypatch):
    import psutil

    from rapid_mlx import cli
    from rapid_mlx.control import protocol

    instance = {"supervisor_pid": 100, "runner": {"pid": 101}, "port": 8017}
    monkeypatch.setattr(protocol, "discover", lambda: [instance])

    def process(pid):
        return SimpleNamespace(
            info={
                "pid": pid,
                "cmdline": ["rapid-mlx", "serve", "tiny"],
                "create_time": time.time(),
            },
            environ=lambda: {},
            net_connections=lambda **kw: [],
        )

    monkeypatch.setattr(
        psutil, "process_iter", lambda *a: [process(100), process(101), process(102)]
    )
    monkeypatch.setattr(cli, "_ps_ready", lambda url: None)
    states = []
    rows = cli._scan_running_servers(states=states)
    assert [(row[0], row[1]) for row in rows] == [(101, "8017"), (102, "8000")]
    assert [state["port"] for state in states] == [8017, 8000]
    assert states[0]["base_url"] == "http://127.0.0.1:8017"


def test_discovery_timeout_keeps_live_index(fake, monkeypatch):
    from rapid_mlx.control import protocol

    _, launch = fake
    proc = launch()
    eventually(live_status)
    entries = list(protocol.discovery_root().glob("*.json"))
    assert len(entries) == 1

    def busy(*args):
        raise TimeoutError("busy")

    monkeypatch.setattr(protocol, "request", busy)
    assert protocol.discover() == []
    assert entries[0].exists()
    proc.terminate()
    proc.communicate(timeout=5)


def test_private_discovery_directory_tightens_permissions(tmp_path, monkeypatch):
    from rapid_mlx.control.protocol import discovery_root, private_directory

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".rapid-mlx" / "control"
    root.mkdir(parents=True, mode=0o755)
    root.chmod(0o755)
    assert discovery_root() == root
    assert private_directory(root) == root
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_legacy_run_untouched_and_control_tightened(fake, capsys, monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.control.protocol import discovery_root

    base, launch = fake
    legacy = base / ".rapid-mlx" / "run"
    legacy.mkdir(parents=True, mode=0o755)
    legacy.chmod(0o755)
    old_entry = legacy / "old.json"
    old_entry.write_text("legacy entry must not be read or removed")
    old_entry.chmod(0o600)
    root = discovery_root()
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    proc = launch()
    status = eventually(
        lambda: (s := live_status()) and s["runner"]["state"] == "ready" and s
    )
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert status_command(SimpleNamespace(port=status["port"])) == 0
    assert (
        json.loads(capsys.readouterr().out)["runner"]["pid"] == status["runner"]["pid"]
    )
    import psutil

    process = SimpleNamespace(
        info={
            "pid": status["runner"]["pid"],
            "cmdline": ["rapid-mlx", "serve", "qwen3.5-4b-4bit"],
            "create_time": time.time(),
        }
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *a: [process])
    monkeypatch.setattr(cli, "_ps_ready", lambda url: None)
    rows = cli._scan_running_servers()
    assert any(row[0] == status["runner"]["pid"] for row in rows)
    assert stat.S_IMODE(legacy.stat().st_mode) == 0o755
    assert list(legacy.iterdir()) == [old_entry]
    assert old_entry.read_text() == "legacy entry must not be read or removed"
    proc.terminate()
    proc.communicate(timeout=5)


@pytest.mark.parametrize("kind", ["foreign", "symlink", "file"])
def test_unsafe_discovery_falls_back_and_serves(tmp_path, kind):
    root = tmp_path / ".rapid-mlx" / "control"
    root.parent.mkdir()
    target = tmp_path / "target"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    if kind == "symlink":
        root.symlink_to(target, target_is_directory=True)
    elif kind == "file":
        root.write_text("untouched")
    else:
        root.mkdir(mode=0o755)
        root.chmod(0o755)
    code = """
import os, socket, sys
from rapid_mlx import cli
from rapid_mlx.control import protocol
if sys.argv[1] == 'foreign':
    real = os.fstat
    def foreign(fd):
        values = list(real(fd))
        values[4] = os.getuid() + 1
        return os.stat_result(values)
    protocol.os.fstat = foreign
args = cli.build_parser().parse_args(['serve', 'qwen3.5-4b-4bit', '--port', '0'])
cli._dispatch_serve_control(args)
# The legacy serve continuation must still be reached in this process.
with socket.socket() as listener:
    listener.bind(('127.0.0.1', args.port))
    listener.listen()
    print(listener.getsockname()[1], flush=True)
    listener.settimeout(5)
    conn, _ = listener.accept()
    with conn:
        conn.sendall(b'served')
"""
    env = dict(os.environ, HOME=str(tmp_path), PYTHONPATH=str(Path.cwd()))
    for key in (RUNNER_ENV, "RAPID_MLX_SINGLE_PROCESS", "RAPID_MLX_WATCHDOG_PPID"):
        env.pop(key, None)
    proc = subprocess.Popen(
        [sys.executable, "-c", code, kind],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        port = int(proc.stdout.readline())
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            assert client.recv(32) == b"served"
        _, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 0, stderr
        assert len(stderr.splitlines()) == 1
        assert b"serving in single process" in stderr
        assert b"Traceback" not in stderr
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    if kind == "foreign":
        assert stat.S_IMODE(root.stat().st_mode) == 0o755
    elif kind == "symlink":
        assert root.is_symlink()
    else:
        assert root.read_text() == "untouched"


@pytest.mark.parametrize(
    "step",
    [
        "mode",
        "port",
        "construct",
        "event_loop",
        "signal",
        "discovery",
        "discovery_mkdir",
        "discovery_open",
        "discovery_stat",
        "discovery_chmod",
        "runtime_root",
        "runtime_mkdir",
        "runtime_stat",
        "runtime_permissions",
        "instance_directory",
        "socket_bind",
        "socket_chmod",
        "metadata_open",
        "metadata_create",
        "metadata_dump",
        "environment",
        "spawn",
    ],
)
def test_setup_failure_returns_to_exact_legacy_path(fake, monkeypatch, caplog, step):
    import asyncio
    import builtins

    from rapid_mlx import cli
    from rapid_mlx.control import modes, protocol, supervisor

    base, _ = fake
    for key in (RUNNER_ENV, "RAPID_MLX_SINGLE_PROCESS", "RAPID_MLX_WATCHDOG_PPID"):
        monkeypatch.delenv(key, raising=False)
    root = protocol.discovery_root()
    root.mkdir(parents=True, mode=0o755)
    root.chmod(0o755)
    handlers = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }

    def fail(*args, **kwargs):
        raise RuntimeError("injected setup failure\nwithout traceback")

    direct = {
        "mode": (modes, "supervised_mode"),
        "port": (cli, "_resolve_serve_port"),
        "construct": (supervisor.secrets, "token_urlsafe"),
        "signal": (asyncio.unix_events._UnixSelectorEventLoop, "add_signal_handler"),
        "discovery": (supervisor, "private_directory"),
        "discovery_open": (protocol.os, "open"),
        "discovery_stat": (protocol.os, "fstat"),
        "discovery_chmod": (protocol.os, "fchmod"),
        "runtime_root": (supervisor, "runtime_root"),
        "instance_directory": (supervisor.tempfile, "mkdtemp"),
        "socket_bind": (asyncio, "start_unix_server"),
        "socket_chmod": (Path, "chmod"),
        "metadata_open": (builtins, "open"),
        "metadata_dump": (supervisor.json, "dump"),
        "spawn": (asyncio, "create_subprocess_exec"),
    }
    if step == "discovery_mkdir":
        real_mkdir = Path.mkdir

        def mkdir(path, *args, **kwargs):
            if path == root:
                fail()
            return real_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", mkdir)
    elif step == "metadata_create":
        real_open = os.open

        def opened(path, *args, **kwargs):
            if str(path).endswith("instance.json"):
                fail()
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", opened)
    elif step in ("discovery_open", "discovery_stat"):
        real = getattr(*direct[step])
        inode = root.stat().st_ino

        def directory_only(value, *args, **kwargs):
            if step == "discovery_open":
                if value == root:
                    fail()
                return real(value, *args, **kwargs)
            info = real(value)
            if info.st_ino == inode:
                fail()
            return info

        monkeypatch.setattr(*direct[step], directory_only)
    elif step in direct:
        monkeypatch.setattr(*direct[step], fail)
    elif step == "event_loop":

        def run(coro):
            coro.close()
            fail()

        monkeypatch.setattr(asyncio, "run", run)
    elif step.startswith("runtime_"):
        name = "mkdir" if step == "runtime_mkdir" else "lstat"
        real = getattr(Path, name)

        def runtime_op(path, *args, **kwargs):
            if path == runtime_root():
                if step == "runtime_permissions":
                    return SimpleNamespace(
                        st_mode=stat.S_IFDIR | 0o755, st_uid=os.getuid()
                    )
                fail()
            return real(path, *args, **kwargs)

        monkeypatch.setattr(Path, name, runtime_op)
    elif step == "environment":
        real = Supervisor.__init__

        class BrokenEnv(dict):
            update = fail

        def init(self, *args, **kwargs):
            real(self, *args, **kwargs)
            self.env = BrokenEnv(self.env)

        monkeypatch.setattr(Supervisor, "__init__", init)

    def forbidden_spawn(*args, **kwargs):
        pytest.fail("setup failure must never spawn a runner")

    if step != "spawn":
        monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        args = build_parser().parse_args(
            ["serve", "qwen3.5-4b-4bit", "--listen-fd", str(listener.fileno())]
        )
        before_args, before_env = vars(args).copy(), dict(os.environ)
        cli._dispatch_serve_control(args)
        if step in ("discovery_open", "discovery_stat"):
            monkeypatch.setattr(*direct[step], real)
        assert vars(args) == before_args
        assert dict(os.environ) == before_env
        # The inherited listener remains usable by the single-process path.
        with socket.create_connection(listener.getsockname(), timeout=2) as client:
            conn, _ = listener.accept()
            with conn:
                conn.sendall(b"served")
            assert client.recv(32) == b"served"
    assert len(caplog.records) == 1
    assert "serving in single process" in caplog.records[0].message
    assert caplog.records[0].exc_info is None
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
    assert not list(base.glob("rmlx-cp-*/i-*"))


@pytest.mark.parametrize(
    "step",
    [
        "index_open",
        "index_dump",
        "index_replace",
        "index_cleanup",
        "runtime_cleanup",
        "socket_cleanup",
    ],
)
def test_post_spawn_control_failures_are_nonfatal(fake, step):
    from rapid_mlx.control.protocol import discovery_root

    base, launch = fake
    injection = """
import asyncio, builtins, json, shutil
from pathlib import Path
def fail(*a, **kw):
    raise OSError('injected control failure')
"""
    if step == "index_open":
        injection += """
real = builtins.open
def opened(path, *a, **kw):
    if str(path).endswith('.tmp'):
        fail()
    return real(path, *a, **kw)
builtins.open = opened
"""
    elif step == "index_dump":
        injection += """
real = json.dump
def dump(data, output, *a, **kw):
    if str(output.name).endswith('.tmp'):
        fail()
    return real(data, output, *a, **kw)
json.dump = dump
"""
    elif step == "index_replace":
        injection += "Path.replace = fail\n"
    elif step == "index_cleanup":
        injection += "Path.unlink = fail\n"
    elif step == "runtime_cleanup":
        injection += "shutil.rmtree = fail\n"
    elif step == "socket_cleanup":
        injection += "asyncio.Server.wait_closed = fail\n"
    proc = launch(injection=injection)
    eventually(lambda: (base / "armed").exists())
    assert proc.poll() is None
    assert (base / "count").read_text() == "1"
    proc.terminate()
    _, stderr = proc.communicate(timeout=5)
    assert proc.returncode == 143, stderr
    assert b"Traceback" not in stderr
    assert b"Control plane" in stderr
    assert b"failed (OSError)" in stderr
    assert b"serving in single process" not in stderr
    if step.startswith("index_") and step != "index_cleanup":
        assert list(discovery_root().iterdir()) == []


def test_pre_spawn_cleanup_failure_cannot_prevent_fallback(fake, monkeypatch, caplog):
    from rapid_mlx import cli
    from rapid_mlx.control import supervisor

    base, _ = fake
    for key in (RUNNER_ENV, "RAPID_MLX_SINGLE_PROCESS", "RAPID_MLX_WATCHDOG_PPID"):
        monkeypatch.delenv(key, raising=False)
    args = build_parser().parse_args(["serve", "qwen3.5-4b-4bit", "--port", "0"])

    def fail(*args, **kwargs):
        raise OSError("injected setup and cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(supervisor.json, "dump", fail)
        patch.setattr(supervisor.shutil, "rmtree", fail)
        cli._dispatch_serve_control(args)
    assert len(caplog.records) == 1
    assert "serving in single process" in caplog.records[0].message
    assert caplog.records[0].exc_info is None
    assert list(base.glob("rmlx-cp-*/i-*"))


def test_foreground_prompt_and_ctrl_c(fake):
    base, launch = fake
    proc = launch("prompt", terminal=True)
    output = bytearray()

    def prompted():
        readable, _, _ = select.select([proc.terminal], [], [], 0.05)
        if readable:
            output.extend(os.read(proc.terminal, 65536))
        return b"Continue?" in output

    eventually(prompted)
    os.write(proc.terminal, b"y\n")
    eventually(lambda: (base / "accepted").exists(), timeout=2)
    eventually(lambda: (base / "armed").exists())
    runner = json.loads((base / "argv1").read_text())["pid"]
    assert os.getpgid(runner) == os.tcgetpgrp(proc.terminal) == proc.pid
    before = time.monotonic()
    os.write(proc.terminal, b"\x03")
    proc.wait(timeout=4.5)
    assert time.monotonic() - before < 4.5
    assert proc.returncode == 130
    with pytest.raises(ProcessLookupError):
        os.kill(runner, 0)


def test_stopped_runner_is_alive_and_status_recovers(fake):
    base, launch = fake
    proc = launch()
    eventually(lambda: (base / "armed").exists())
    runner = json.loads((base / "argv1").read_text())["pid"]
    os.kill(runner, signal.SIGSTOP)
    eventually(lambda: (s := live_status()) and s["runner"]["state"] == "stopped")
    time.sleep(0.3)
    assert (base / "count").read_text() == "1"
    os.kill(runner, signal.SIGCONT)
    eventually(lambda: (s := live_status()) and s["runner"]["state"] == "ready")
    proc.terminate()
    proc.communicate(timeout=4.5)


@pytest.mark.parametrize("mode", ["grandchild", "grandchild-term"])
def test_sigkill_supervisor_cleans_grandchild(fake, mode):
    import psutil

    base, launch = fake
    proc = launch(mode)
    eventually(lambda: (base / "armed").exists())
    runner = json.loads((base / "argv1").read_text())["pid"]
    grandchild = int((base / "grandchild").read_text())
    try:
        deadline = time.monotonic() + 4.0
        proc.kill()
        proc.wait(timeout=1)
        eventually(
            lambda: all(
                not psutil.pid_exists(pid)
                or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
                for pid in (runner, grandchild)
            ),
            timeout=max(0, deadline - time.monotonic()),
        )
        proc.communicate(timeout=0.5)
    finally:
        for pid in (runner, grandchild):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_shared_service_parent_survives_runner_cleanup(fake):
    base, launch = fake
    proc = launch("grandchild", service=True)
    status = eventually(lambda: (base / "armed").exists() and live_status())
    supervisor = status["supervisor_pid"]
    runner = status["runner"]["pid"]
    assert os.getpgid(proc.pid) == os.getpgid(supervisor) == os.getpgid(runner)
    before = time.monotonic()
    os.kill(supervisor, signal.SIGTERM)
    _, stderr = proc.communicate(timeout=4.5)
    assert proc.returncode == 143, stderr
    assert time.monotonic() - before < 4.5


def test_process_tree_retains_orphans_and_rejects_recycled_pids(monkeypatch):
    from rapid_mlx.control import process_tree

    # The ancestor/service and a sibling must never receive a cleanup signal.
    table = {
        10: (1, "S", "service"),
        20: (10, "S", "runner"),
        21: (10, "S", "sibling"),
        30: (20, "S", "installer"),
        40: (30, "S", "worker"),
    }
    monkeypatch.setattr(process_tree, "process_table", lambda: table)
    signals = []
    monkeypatch.setattr(process_tree.os, "kill", lambda *args: signals.append(args))
    tree = process_tree.ProcessTree(20)
    tree.send(signal.SIGTERM)
    assert set(signals) == {
        (20, signal.SIGTERM),
        (30, signal.SIGTERM),
        (40, signal.SIGTERM),
    }
    assert signals[-1] == (20, signal.SIGTERM)
    # Runner exits, installer is reparented, and the worker PID is reused.
    del table[20]
    table[30] = (1, "S", "installer")
    table[40] = (10, "S", "unrelated replacement")
    table[50] = (30, "S", "late worker")
    signals.clear()
    tree.send(signal.SIGKILL)
    assert set(signals) == {(30, signal.SIGKILL), (50, signal.SIGKILL)}


@pytest.mark.parametrize("state", ["T", "T+", "t"])
def test_process_tree_stopped_counts_as_alive(monkeypatch, state):
    from rapid_mlx.control import process_tree

    monkeypatch.setattr(
        process_tree, "process_table", lambda: {20: (10, state, "runner")}
    )
    tree = process_tree.ProcessTree(20)
    assert tree.refresh() == {20}
    assert tree.stopped
    supervisor = Supervisor([], port=8000, mode="text-vlm")
    supervisor.child = SimpleNamespace(pid=20, returncode=None)
    supervisor.tree = tree
    assert supervisor.status()["runner"]["state"] == "stopped"
    supervisor.child.returncode = -signal.SIGKILL
    supervisor.state = "restarting"
    assert supervisor.status()["runner"]["state"] == "restarting"


def test_process_tree_snapshot_failure_never_signals_stale_pids(monkeypatch):
    from rapid_mlx.control import process_tree

    monkeypatch.setattr(
        process_tree, "process_table", lambda: {20: (10, "S", "runner")}
    )
    tree = process_tree.ProcessTree(20)
    tree.refresh()

    def denied():
        raise PermissionError("process table unavailable")

    monkeypatch.setattr(process_tree, "process_table", denied)
    signals = []
    monkeypatch.setattr(process_tree.os, "kill", lambda *args: signals.append(args))
    assert tree.send(signal.SIGKILL) == set()
    assert signals == []


def test_process_table_portable_columns_and_locale(monkeypatch):
    from rapid_mlx.control import process_tree

    class Ps:
        pid = 99
        returncode = 0

        def __init__(self, argv, **kwargs):
            assert argv == ["/bin/ps", "-Ao", "pid=,ppid=,stat=,lstart="]
            assert kwargs["env"]["LC_ALL"] == "C"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, timeout):
            assert timeout <= 0.2
            return (
                "   20  10 T+   Thu Nov 19 12:30:00 2026\n"
                " 30 20 S Mon Nov  9 01:02:03 2026\n"
                " 99 20 R Thu Nov 19 12:31:00 2026\n",
                "",
            )

    monkeypatch.setattr(process_tree.subprocess, "Popen", Ps)
    assert process_tree.process_table() == {
        20: (10, "T+", "Thu Nov 19 12:30:00 2026"),
        30: (20, "S", "Mon Nov  9 01:02:03 2026"),
    }


def test_supervised_runner_watchdog_poll_interval(monkeypatch):
    from rapid_mlx import _parent_watchdog, cli
    from rapid_mlx.control import PARENT_ENV, protocol
    from rapid_mlx.control.process_tree import supervised_on_orphan

    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(PARENT_ENV, "123")
    installed = []
    monkeypatch.setattr(
        _parent_watchdog,
        "install_parent_watchdog",
        lambda *args, **kwargs: installed.append((args, kwargs)),
    )
    monkeypatch.setattr(protocol, "install_startup_report", lambda: None)
    cli._dispatch_serve_control(SimpleNamespace())
    assert installed == [((123,), {"interval": 0.5, "on_orphan": supervised_on_orphan})]


@pytest.mark.parametrize("recycled", [False, True])
def test_supervised_orphan_cleans_children_before_default_callback(
    monkeypatch, recycled
):
    from rapid_mlx import _parent_watchdog
    from rapid_mlx.control import process_tree

    runner = os.getpid()
    table = {runner: (1, "S", "runner"), 30: (runner, "S", "installer")}
    monkeypatch.setattr(process_tree, "process_table", lambda: table)
    clock = iter([0, 0, 2])
    monkeypatch.setattr(process_tree.time, "monotonic", lambda: next(clock))

    def reparent(_):
        table[30] = (1, "S", "replacement" if recycled else "installer")

    monkeypatch.setattr(process_tree.time, "sleep", reparent)
    signals = []
    monkeypatch.setattr(process_tree.os, "kill", lambda *args: signals.append(args))

    def default(expected, observed, *, grace):
        assert (expected, observed) == (10, 1)
        assert grace == 1.0
        expected_signals = [(30, signal.SIGTERM)]
        if not recycled:
            expected_signals.append((30, signal.SIGKILL))
        assert signals == expected_signals

    monkeypatch.setattr(_parent_watchdog, "_default_on_orphan", default)
    process_tree.supervised_on_orphan(10, 1)
