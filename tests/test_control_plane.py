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
from fastapi import FastAPI, Request
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

    base = Path(tempfile.mkdtemp(prefix="cp-", dir="/tmp")).resolve()
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
if mode == 'usage':
    sys.exit(2)
if mode == 'crash-before-report':
    os._exit(7)
if mode == 'unflushed':
    print('small unflushed startup line')
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
if mode in ('uvicorn', 'uvicorn-slow'):
    import asyncio
    import uvicorn
    from rapid_mlx._uvicorn import AcceptingConnectionsServer
    async def app(scope, receive, send):
        if scope['type'] == 'lifespan':
            while True:
                event = await receive()
                if event['type'] == 'lifespan.startup':
                    await send({'type': 'lifespan.startup.complete'})
                elif event['type'] == 'lifespan.shutdown':
                    with (base / 'lifespan-shutdown').open('a') as output:
                        output.write('shutdown\\n')
                    await asyncio.sleep(0.25)
                    (base / 'lifespan-complete').touch()
                    await send({'type': 'lifespan.shutdown.complete'})
                    return
        elif scope['type'] == 'http':
            (base / 'request-active').touch()
            await asyncio.sleep(30)
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'ok'})
    class Server(AcceptingConnectionsServer):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            report('bound', port=listener.getsockname()[1])
            (base / 'armed').touch()
        async def shutdown(self, sockets=None):
            try:
                await super().shutdown(sockets=sockets)
            finally:
                # Record before Uvicorn replays captured signals to the original
                # handlers (SIGTERM can terminate without running Python finally).
                (base / 'force-exit').write_text(str(self.force_exit))
        def handle_exit(self, sig, frame):
            super().handle_exit(sig, frame)
            with (base / 'received-signals').open('a') as output:
                output.write(str(sig) + '\\n')
    server = Server(uvicorn.Config(app, loop='asyncio', lifespan='on', log_level='error'))
    server.run(sockets=[listener])
    sys.exit(0)
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


def test_runner_inherits_log_file_redirection(fake):
    base, launch = fake
    proc = launch(
        "logs",
        injection="""
import os
from pathlib import Path
log = open(Path(os.environ['TMPDIR']) / 'redirected.log', 'wb', buffering=0)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
""",
    )
    stdout, stderr = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert stdout == stderr == b""
    assert (base / "redirected.log").read_bytes() == (
        b"x" * 262144 + b"\x00\xff" + b"y" * 262144 + b"\x00\xff"
    )


async def test_runner_inherits_output_descriptors_without_processes(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from rapid_mlx.control import supervisor as module

    instance = Supervisor([], port=8000, mode="text-vlm")
    child = SimpleNamespace(pid=123, returncode=0, stdout=None, stderr=None)
    create = AsyncMock(return_value=child)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", Mock())
    monkeypatch.setattr(loop, "remove_signal_handler", Mock())
    monkeypatch.setattr(module.signal, "signal", Mock())
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(module, "ProcessTree", Mock())
    monkeypatch.setattr(instance, "open", AsyncMock())
    monkeypatch.setattr(instance, "publish_index", Mock())
    monkeypatch.setattr(instance, "signal_descendants", Mock())

    assert await instance.run() == 0
    assert create.await_args.kwargs["stdout"] is None
    assert create.await_args.kwargs["stderr"] is None


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_signal_forward_and_reap(fake, sig):
    base, launch = fake
    proc = launch()
    eventually(lambda: (base / "armed").exists())
    child_pid = json.loads((base / "argv1").read_text())["pid"]
    proc.send_signal(sig)
    proc.communicate(timeout=6)
    assert proc.returncode == 128 + sig
    # PID-only SIGINT cannot be distinguished from terminal SIGINT using
    # asyncio signal callbacks; it receives TERM at the bounded escalation.
    expected = signal.SIGTERM if sig == signal.SIGINT else sig
    assert int((base / "signal").read_text()) == expected
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


@pytest.mark.parametrize("code,reported", [(2, False), (7, True), (1, True)])
async def test_startup_errors_are_terminal_without_processes(
    monkeypatch, code, reported
):
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from rapid_mlx.control import supervisor as module

    instance = Supervisor([], port=8000, mode="text-vlm")

    async def spawn(*a, **kw):
        instance.startup_failed = reported
        return SimpleNamespace(pid=123, returncode=code, stdout=None, stderr=None)

    create = AsyncMock(side_effect=spawn)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", Mock())
    monkeypatch.setattr(loop, "remove_signal_handler", Mock())
    monkeypatch.setattr(module.signal, "signal", Mock())
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(module, "ProcessTree", Mock())
    monkeypatch.setattr(instance, "open", AsyncMock())
    monkeypatch.setattr(instance, "publish_index", Mock())
    monkeypatch.setattr(instance, "signal_descendants", Mock())
    monkeypatch.setattr(instance, "restart_delay", lambda: 0)

    assert await instance.run() == code
    assert create.await_count == 1
    assert instance.restarts == 0


@pytest.mark.parametrize("port", [None, 0, 8123])
def test_dispatch_resolves_auto_port_once_before_spawn(monkeypatch, port):
    from rapid_mlx import cli
    from rapid_mlx.control import supervisor

    for key in (RUNNER_ENV, "RAPID_MLX_WATCHDOG_PPID", "RAPID_MLX_SINGLE_PROCESS"):
        monkeypatch.delenv(key, raising=False)
    args = build_parser().parse_args(["serve", "qwen3.5-4b-4bit"])
    args.port = port
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "qwen3.5-4b-4bit"])
    calls = []

    def resolve(host, port, **kw):
        calls.append(("resolve", port))
        return 8007

    def spawn(argv, **kw):
        calls.append(("spawn", kw["port"]))
        assert argv == [
            sys.executable,
            "-u",
            "-X",
            "utf8",
            "-m",
            "rapid_mlx",
            "serve",
            "qwen3.5-4b-4bit",
        ]
        return 13

    import subprocess

    # Interpreter flags (e.g. ``python -u``) must reach the runner.
    monkeypatch.setattr(
        subprocess, "_args_from_interpreter_flags", lambda: ["-u", "-X", "utf8"]
    )
    monkeypatch.setattr(cli, "_resolve_serve_port", resolve)
    monkeypatch.setattr(supervisor, "supervise", spawn)
    with pytest.raises(SystemExit) as exited:
        cli._dispatch_serve_control(args)
    assert exited.value.code == 13
    assert calls == (
        [("resolve", None), ("spawn", 8007)] if port is None else [("spawn", port)]
    )


@pytest.mark.parametrize("console", [False, True])
def test_only_console_entrypoint_supervises(monkeypatch, console):
    from rapid_mlx import cli

    class DispatchedError(Exception):
        pass

    def control(args):
        assert console
        assert args.model == "qwen3.5-4b-4bit"
        raise DispatchedError

    def in_process(args):
        assert not console
        raise DispatchedError

    monkeypatch.setattr(cli, "_dispatch_serve_control", control)
    monkeypatch.setattr(cli, "serve_command", in_process)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *a, **kw: None)
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["rapid-mlx", "--disable-version-check", "serve", "qwen3.5-4b-4bit"],
    )
    with pytest.raises(DispatchedError):
        (cli.cli_entrypoint if console else cli.main)()


def test_console_exposure_gate_precedes_control_dispatch(monkeypatch, capsys):
    import builtins

    from rapid_mlx import cli

    def forbidden(*a, **kw):
        pytest.fail("invalid exposure must fail before supervision or resolution")

    original_import = builtins.__import__

    def lightweight_import(name, *a, **kw):
        if name in ("server", "rapid_mlx.server", "mlx") or name.startswith("mlx."):
            pytest.fail("exposure validation must not import the inference runtime")
        return original_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", lightweight_import)
    monkeypatch.setattr(cli, "_dispatch_serve_control", forbidden)
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_model", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            "qwen3.5-4b-4bit",
            "--host",
            "0.0.0.0",
            "--enable-local-search",
        ],
    )
    with pytest.raises(SystemExit) as exited:
        cli.cli_entrypoint()
    assert exited.value.code == 2
    assert "requires API-key authentication" in capsys.readouterr().err


def test_control_fallback_notice_waits_for_log_redirection(monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.local_models import flush_serve_startup_notes

    for key in (RUNNER_ENV, "RAPID_MLX_WATCHDOG_PPID", "RAPID_MLX_SINGLE_PROCESS"):
        monkeypatch.delenv(key, raising=False)
    args = build_parser().parse_args(
        ["serve", "unknown/target", "--log-file", "serve.log"]
    )
    cli._dispatch_serve_control(args)
    assert capsys.readouterr().err == ""
    flush_serve_startup_notes(args)
    assert (
        capsys.readouterr().err
        == "Control functionality is unavailable in this mode.\n"
    )


@pytest.mark.parametrize("root_path", ["", "/prefix"])
@pytest.mark.parametrize("app_path", ["object", "string", "factory", "object-factory"])
@pytest.mark.parametrize(
    "transport_peer,expected", [("192.168.1.2", 403), ("127.0.0.1", 200)]
)
def test_forwarder_preserves_transport_peer_with_proxy_headers(
    monkeypatch, root_path, app_path, transport_peer, expected
):
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

    # In-process config checks must not start a delivery thread using shared env.
    from rapid_mlx.control import observer

    monkeypatch.setattr(observer, "_delivery", observer.Delivery())
    captured = []
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return app

    monkeypatch.setitem(
        sys.modules, "control_test_app", SimpleNamespace(app=app, factory=factory)
    )
    target = {
        "object": app,
        "string": "control_test_app:app",
        "factory": "control_test_app:factory",
        "object-factory": factory,
    }[app_path]

    class Server:
        started = True

        def __init__(self, config, **kwargs):
            self.config = config
            assert not config.loaded

        def run(self):
            self.config.load()
            captured.append(self.config.loaded_app)

    monkeypatch.setattr(_uvicorn, "AcceptingConnectionsServer", Server)
    _uvicorn._run_server(
        target,
        factory="factory" in app_path,
        forwarded_allow_ips="*",
        proxy_headers=True,
        root_path=root_path,
    )
    assert len(factory_calls) == int("factory" in app_path)
    with TestClient(
        captured[0], client=(transport_peer, 1234), root_path=root_path
    ) as client:
        headers = {"X-Forwarded-For": "127.0.0.1"}
        assert (
            client.get(f"{root_path}/rapid/v1/status", headers=headers).status_code
            == expected
        )
        assert client.get(f"{root_path}/v1/peer", headers=headers).json() == {
            "peer": "127.0.0.1"
        }


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


@pytest.mark.parametrize("mode,code", [("startup", 7), ("usage", 2)])
def test_invalid_startup_never_restarts(fake, mode, code):
    base, launch = fake
    proc = launch(mode, fast=True)
    proc.communicate(timeout=5)
    assert proc.returncode == code
    assert (base / "count").read_text() == "1"
    assert not discover()


def test_five_unreported_crashes_give_up_nonzero(fake):
    base, launch = fake
    proc = launch("crash-before-report", fast=True)
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
def test_setup_failure_returns_to_exact_legacy_path(
    fake, monkeypatch, caplog, capsys, step
):
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
    # Every pre-spawn failure uses the buffered note, including socket setup.
    assert not caplog.records
    assert "serving in single process" in capsys.readouterr().err
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


def test_pre_spawn_cleanup_failure_cannot_prevent_fallback(
    fake, monkeypatch, caplog, capsys
):
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
    assert not caplog.records
    assert capsys.readouterr().err.count("serving in single process") == 1
    assert list(base.glob("rmlx-cp-*/i-*"))


def test_unflushed_runner_line_reaches_terminal_while_alive(fake, monkeypatch):
    # No -u or explicit flush: inheriting the TTY must retain line buffering.
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    base, launch = fake
    proc = launch("unflushed", terminal=True)
    eventually(lambda: (base / "armed").exists())
    output = bytearray()

    def printed():
        readable, _, _ = select.select([proc.terminal], [], [], 0.05)
        if readable:
            output.extend(os.read(proc.terminal, 65536))
        return b"small unflushed startup line" in output

    eventually(printed, timeout=2)
    assert proc.poll() is None
    proc.terminate()
    assert proc.wait(timeout=6) == 143


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


@pytest.fixture
def mock_process_info(monkeypatch):
    from rapid_mlx.control import process_tree

    monkeypatch.setattr(
        process_tree, "process_info", lambda pid: process_tree.process_table().get(pid)
    )


def test_process_tree_retains_orphans_and_rejects_recycled_pids(
    monkeypatch, mock_process_info
):
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
def test_process_tree_stopped_counts_as_alive(monkeypatch, mock_process_info, state):
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


def test_process_tree_snapshot_failure_never_signals_stale_pids(
    monkeypatch, mock_process_info
):
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


def test_linux_native_starttime_and_comm_parsing(monkeypatch):
    from rapid_mlx.control import process_tree

    monkeypatch.setattr(process_tree.sys, "platform", "linux")
    stat = "20 (a tricky ) name) T 10 " + "0 " * 17 + "123456789 0"
    monkeypatch.setattr(process_tree.Path, "read_text", lambda _: stat)
    assert process_tree.process_info(20) == (10, "T", 123456789)


def test_macos_native_starttime_and_enumeration(monkeypatch):
    import ctypes

    from rapid_mlx.control import process_tree

    class Libproc:
        def proc_listallpids(self, buffer, size):
            if buffer is None:
                return 2
            assert size >= 8
            buffer[0], buffer[1] = 20, 30
            return 2

        def proc_pidinfo(self, pid, flavor, arg, buffer, size):
            assert (flavor, arg, size) == (3, 0, 136)
            info = buffer._obj
            info.ppid = 10 if pid == 20 else 20
            info.status = 4 if pid == 20 else 5
            info.start_sec, info.start_usec = 1700000000, pid
            return size

    assert ctypes.sizeof(process_tree._ProcBsdInfo) == 136
    monkeypatch.setattr(process_tree.sys, "platform", "darwin")
    monkeypatch.setattr(process_tree, "_libproc", lambda: Libproc())
    assert process_tree.process_table() == {
        20: (10, "T", (1700000000, 20)),
        30: (20, "Z", (1700000000, 30)),
    }


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_process_tree_revalidates_each_signal_same_second_reuse(monkeypatch, sig):
    from rapid_mlx.control import process_tree

    original = (1700000000, 100)
    replacement = (1700000000, 101)
    table = {20: (10, "S", original), 30: (20, "S", original)}
    monkeypatch.setattr(process_tree, "process_table", lambda: table.copy())
    monkeypatch.setattr(process_tree, "process_info", lambda pid: table.get(pid))
    tree = process_tree.ProcessTree(20)
    tree.refresh()
    # Snapshot still has the old identity, but the per-signal lookup sees reuse.
    monkeypatch.setattr(
        process_tree, "process_info", lambda pid: (10, "S", replacement)
    )
    signals = []
    monkeypatch.setattr(process_tree.os, "kill", lambda *args: signals.append(args))
    assert tree.send(sig) == set()
    tree.send_root(sig)
    supervisor = Supervisor([], port=0, mode="text-vlm")
    supervisor.tree = tree
    supervisor.child = SimpleNamespace(pid=20, returncode=None)
    supervisor.signal_descendants(sig)
    assert signals == []


def test_supervisor_sigint_waits_without_forward_or_deadline_reset(monkeypatch):
    supervisor = Supervisor([], port=0, mode="text-vlm")
    signals = []
    monkeypatch.setattr(supervisor, "signal_descendants", signals.append)
    supervisor.stop(signal.SIGINT)
    started = supervisor.stop_started
    supervisor.stop(signal.SIGINT)
    supervisor.stop(signal.SIGTERM)
    assert signals == []
    assert supervisor.stop_started == started
    assert supervisor.stop_signal == signal.SIGINT


def test_tree_sampling_waits_five_seconds_off_caller_thread(monkeypatch):
    import threading

    from rapid_mlx.control import process_tree

    monkeypatch.setattr(process_tree, "process_info", lambda _: (1, "S", 123))
    tree = process_tree.ProcessTree(20)
    calls = []
    finished = threading.Event()

    class Stop:
        def is_set(self):
            return False

        def wait(self, interval):
            calls.append(interval)
            finished.set()
            return True

    tree.tracking_stop = Stop()
    monkeypatch.setattr(tree, "refresh", lambda: calls.append(threading.get_ident()))
    tree.start_tracking()
    assert finished.wait(2)
    assert calls[0] != threading.get_ident()
    assert calls[1:] == [5.0]


def test_supervised_runner_watchdog_poll_interval(monkeypatch):
    from rapid_mlx import _parent_watchdog, cli
    from rapid_mlx.control import PARENT_ENV, process_tree, protocol
    from rapid_mlx.control.process_tree import supervised_on_orphan

    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(PARENT_ENV, "123")
    installed = []
    monkeypatch.setattr(
        _parent_watchdog,
        "install_parent_watchdog",
        lambda *args, **kwargs: installed.append((args, kwargs)),
    )
    tracked = []
    monkeypatch.setattr(
        process_tree, "start_orphan_tracking", lambda: tracked.append(True)
    )
    monkeypatch.setattr(protocol, "install_startup_report", lambda: None)
    cli._dispatch_serve_control(SimpleNamespace())
    assert tracked == [True]
    assert installed == [((123,), {"interval": 0.5, "on_orphan": supervised_on_orphan})]


@pytest.mark.parametrize("recycled", [False, True])
@pytest.mark.parametrize("pretracked", [False, True])
def test_supervised_orphan_cleans_children_before_default_callback(
    monkeypatch, mock_process_info, recycled, pretracked
):
    from rapid_mlx import _parent_watchdog
    from rapid_mlx.control import process_tree

    monkeypatch.setattr(process_tree, "_orphan_tree", None)
    runner = os.getpid()
    table = {runner: (1, "S", "runner"), 30: (runner, "S", "installer")}
    monkeypatch.setattr(process_tree, "process_table", lambda: table)
    if pretracked:
        tree = process_tree.ProcessTree(runner)
        tree.refresh()
        monkeypatch.setattr(process_tree, "_orphan_tree", tree)
        # The installer was observed by the idle sampler, then detached
        # BEFORE parent loss. The orphan callback must reuse that identity.
        table[30] = (1, "S", "installer")
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


@pytest.mark.parametrize(
    "supervised,force", [(True, True), (True, False), (False, True)]
)
async def test_uvicorn_connection_close_wait_respects_force_exit(
    monkeypatch, supervised, force
):
    import asyncio

    import uvicorn

    from rapid_mlx._uvicorn import AcceptingConnectionsServer

    monkeypatch.delenv(RUNNER_ENV, raising=False)
    if supervised:
        monkeypatch.setenv(RUNNER_ENV, "1")
    server = AcceptingConnectionsServer(uvicorn.Config(FastAPI()))
    waiting, closed, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def wait_closed():
        waiting.set()
        try:
            await closed.wait()
        finally:
            finished.set()

    server.servers = [SimpleNamespace(wait_closed=wait_closed)]
    drain = asyncio.create_task(server._wait_tasks_to_complete())
    try:
        await asyncio.wait_for(waiting.wait(), 0.5)
        server.handle_exit(signal.SIGINT, None)
        await asyncio.sleep(0.1)
        assert not drain.done(), "first Ctrl-C must preserve graceful drain"
        if force:
            server.handle_exit(signal.SIGINT, None)
            assert server.force_exit
        if not (supervised and force):
            await asyncio.sleep(0.1)
            assert not drain.done()
            closed.set()
        await asyncio.wait_for(drain, 0.5)
        assert finished.is_set(), "connection wait task leaked"
    finally:
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.parametrize("delivery", ["group-int", "pid-term", "pid-int"])
def test_real_uvicorn_graceful_lifespan_shutdown(fake, delivery):
    base, launch = fake
    proc = launch("uvicorn")
    eventually(lambda: (base / "armed").exists())
    sig = signal.SIGTERM if delivery == "pid-term" else signal.SIGINT
    if delivery == "group-int":
        os.killpg(proc.pid, sig)
    else:
        proc.send_signal(sig)
    _, stderr = proc.communicate(timeout=4.5)
    assert proc.returncode == 128 + sig, stderr.decode(errors="replace")
    received = signal.SIGTERM if delivery == "pid-int" else sig
    assert (base / "received-signals").read_text().splitlines() == [str(received)]
    assert (base / "lifespan-shutdown").read_text().splitlines() == ["shutdown"]
    assert (base / "lifespan-complete").exists()
    assert (base / "force-exit").read_text() == "False"
    assert (base / "count").read_text() == "1"


def test_real_uvicorn_second_ctrl_c_forces_during_request_drain(fake):
    base, launch = fake
    proc = launch("uvicorn-slow")
    eventually(lambda: (base / "armed").exists())
    status = eventually(live_status)
    with socket.create_connection(("127.0.0.1", status["port"]), timeout=2) as client:
        client.sendall(b"GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n")
        eventually(lambda: (base / "request-active").exists())
        os.killpg(proc.pid, signal.SIGINT)
        eventually(
            lambda: (
                (base / "received-signals").exists()
                and len((base / "received-signals").read_text().splitlines()) == 1
            )
        )
        before = time.monotonic()
        os.killpg(proc.pid, signal.SIGINT)
        try:
            _, stderr = proc.communicate(timeout=1.5)
        except subprocess.TimeoutExpired as exc:
            details = {
                name: (base / name).read_text() if (base / name).exists() else None
                for name in ("received-signals", "force-exit", "count")
            }
            pytest.fail(
                f"second Ctrl-C did not exit within 1.5s; "
                f"supervisor returncode={proc.poll()}; {details}; "
                f"stderr={(exc.stderr or b'').decode(errors='replace')}",
                pytrace=False,
            )
    diagnostic = stderr.decode(errors="replace")
    assert time.monotonic() - before < 1.5, diagnostic
    assert proc.returncode == 130, diagnostic
    assert (base / "received-signals").read_text().splitlines() == ["2", "2"], (
        diagnostic
    )
    assert (base / "force-exit").read_text() == "True", diagnostic
    assert not (base / "lifespan-shutdown").exists(), diagnostic
    assert (base / "count").read_text() == "1", diagnostic


def test_module_entrypoint_and_status_dispatch(monkeypatch):
    import runpy
    from unittest.mock import Mock

    from rapid_mlx import cli
    from rapid_mlx.control import protocol

    entry = Mock()
    with monkeypatch.context() as patch:
        patch.setattr(cli, "cli_entrypoint", entry)
        runpy.run_module("rapid_mlx", run_name="__main__")
    entry.assert_called_once_with()
    status = Mock(return_value=7)
    monkeypatch.setattr(protocol, "status_command", status)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "status", "--port", "8123"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 7
    assert status.call_args.args[0].port == 8123


def test_native_mtp_profile_stays_single_process(monkeypatch):
    from rapid_mlx.control import modes

    monkeypatch.setattr(
        modes.json, "loads", lambda _: {"model": {"mtp_draft_model": "draft"}}
    )
    assert modes.supervised_mode(SimpleNamespace(model="model"), {}) is None
    assert (
        modes.supervised_mode(SimpleNamespace(model="model", no_spec_decode=True), {})
        == "text-vlm"
    )


@pytest.mark.parametrize(
    "method,target,data,headers,code,state",
    [
        ("GET", "/rapid/v1/status", None, {}, 200, "starting"),
        ("GET", "/other", None, {}, 404, "starting"),
        ("GET", "/rapid/v1/status", None, {SECRET_HEADER: "wrong"}, 403, "starting"),
        ("POST", "/startup", {"pid": 99}, {}, 409, "starting"),
        (
            "POST",
            "/startup",
            {"pid": 123, "event": "bound", "port": 8123},
            {},
            200,
            "ready",
        ),
        (
            "POST",
            "/startup",
            {"pid": 123, "event": "bound", "port": 8123, "ready": False},
            {},
            200,
            "starting",
        ),
        ("POST", "/startup", {"pid": 123, "event": "bind_failed"}, {}, 200, "starting"),
        (
            "POST",
            "/startup",
            {"pid": 123, "event": "startup_failed"},
            {},
            200,
            "failed",
        ),
        ("POST", "/startup", {"pid": 123, "event": "unknown"}, {}, None, "starting"),
        (
            "POST",
            "/startup",
            {"pid": 123, "event": "bound", "port": 0},
            {},
            None,
            "starting",
        ),
        ("GET", "/rapid/v1/status", None, {"Content-Length": "-1"}, None, "starting"),
        (
            "GET",
            "/rapid/v1/status",
            None,
            {"Transfer-Encoding": "chunked"},
            None,
            "starting",
        ),
    ],
)
async def test_control_handler_in_process(
    method, target, data, headers, code, state, tmp_path
):
    import asyncio
    from unittest.mock import AsyncMock, Mock

    instance = Supervisor([], port=8000, mode="text-vlm")
    from rapid_mlx.control.history import History

    instance.history = History(tmp_path / "history")
    instance.child = SimpleNamespace(pid=123, returncode=None)
    instance.spawn_ready.set()
    body = json.dumps(data).encode() if data else b""
    fields = {
        SECRET_HEADER: instance.secret,
        "Content-Length": str(len(body)),
        **headers,
    }
    head = (
        f"{method} {target} HTTP/1.1\r\n"
        + "".join(f"{k}: {v}\r\n" for k, v in fields.items())
        + "\r\n"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(head.encode() + body)
    reader.feed_eof()
    writer = Mock(drain=AsyncMock(), wait_closed=AsyncMock(side_effect=OSError))
    await instance.handle(reader, writer)
    assert instance.state == state
    assert instance.bind_failed == (
        data is not None and data.get("event") == "bind_failed"
    )
    assert instance.startup_failed == (state == "failed")
    if code is None:
        writer.write.assert_not_called()
    else:
        response = writer.write.call_args.args[0]
        assert response.startswith(f"HTTP/1.1 {code} ".encode())
        result = json.loads(response.split(b"\r\n\r\n")[1])
        if code == 200 and method == "GET":
            assert result["runner"]["pid"] == 123
            assert result["port"] == 8000
        else:
            assert result == {}
    if state == "ready":
        assert instance.port == 8123
    writer.close.assert_called_once_with()
    writer.wait_closed.assert_awaited_once_with()


async def test_control_handler_duplicate_headers_and_transport_error():
    import asyncio
    from unittest.mock import AsyncMock, Mock

    instance = Supervisor([], port=8000, mode="text-vlm")
    for head in (b"GET / HTTP/1.1\r\nX: 1\r\nx: 2\r\n\r\n", b"partial"):
        reader = asyncio.StreamReader()
        reader.feed_data(head)
        reader.feed_eof()
        writer = Mock(drain=AsyncMock(), wait_closed=AsyncMock())
        await instance.handle(reader, writer)
        writer.write.assert_not_called()
        writer.close.assert_called_once_with()


async def test_supervisor_open_publish_and_cleanup_in_process(
    monkeypatch, tmp_path, short_control_root
):
    from rapid_mlx.control import supervisor as module

    monkeypatch.setattr(module, "runtime_root", lambda: short_control_root / "runtime")
    monkeypatch.setattr(module, "discovery_root", lambda: tmp_path / "index")
    instance = Supervisor([], port=0, mode="lazy")
    await instance.open()
    try:
        instance.publish_index()
        metadata = json.loads(instance.index_entry.read_text())
        assert metadata == {
            "socket": instance.env[SOCKET_ENV],
            "secret": instance.secret,
        }
        assert stat.S_IMODE(instance.index_entry.stat().st_mode) == 0o600
        assert not instance.index_entry.with_suffix(".tmp").exists()
        assert stat.S_ISSOCK(Path(metadata["socket"]).stat().st_mode)
        assert json.loads((instance.directory / "instance.json").read_text()) == {
            "secret": instance.secret
        }
    finally:
        instance.server.close()
        await instance.server.wait_closed()
    (short_control_root / "runtime").chmod(0o755)
    with pytest.raises(OSError, match="owned 0700"):
        await Supervisor([], port=0, mode="lazy").open()


@pytest.mark.parametrize(
    "scenario",
    [
        "crashes",
        "backoff_stop",
        "term_during_spawn",
        "int_escalate",
        "graceful",
        "pre_stopped",
        "spawn_error",
        "wait_error",
    ],
)
async def test_supervisor_lifecycle_in_process(monkeypatch, tmp_path, caplog, scenario):
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from rapid_mlx.control import supervisor as module

    instance = Supervisor(["runner"], port=8123, mode="text-vlm")
    from rapid_mlx.control.history import History

    instance.history = History(tmp_path / "history")
    child = SimpleNamespace(pid=123, returncode=None)
    tree = Mock()
    tree.send.return_value = set()
    tree.refresh.return_value = {123} if scenario == "int_escalate" else set()
    loop = asyncio.get_running_loop()
    handlers = {}
    monkeypatch.setattr(
        loop,
        "add_signal_handler",
        lambda sig, fn, arg: handlers.update({sig: (fn, arg)}),
    )
    remove = Mock()
    restore = Mock()
    monkeypatch.setattr(loop, "remove_signal_handler", remove)
    monkeypatch.setattr(module.signal, "signal", restore)
    monkeypatch.setattr(module, "ProcessTree", Mock(return_value=tree))
    server = Mock(wait_closed=AsyncMock())
    listener = os.open(tmp_path / "listener", os.O_CREAT | os.O_RDWR, 0o600)
    instance.listen_fd = listener
    directory = tmp_path / "runtime"
    directory.mkdir()
    instance.directory = directory
    instance.index_entry = tmp_path / "entry.json"
    instance.index_entry.write_text("{}")
    instance.server = server

    async def setup():
        if scenario == "pre_stopped":
            handlers[signal.SIGTERM][0](signal.SIGTERM)

    monkeypatch.setattr(instance, "open", setup)
    publish = Mock(side_effect=OSError("index unavailable"))
    monkeypatch.setattr(instance, "publish_index", publish)

    async def spawn(*argv, **kwargs):
        assert kwargs["pass_fds"] == (listener,)
        if scenario == "spawn_error":
            raise OSError("spawn failed")
        child.returncode = None
        if scenario == "term_during_spawn":
            handlers[signal.SIGTERM][0](signal.SIGTERM)
        return child

    create = AsyncMock(side_effect=spawn)
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", create)
    real_sleep = asyncio.sleep
    tick = [0]

    async def sleep(delay):
        tick[0] += 1
        if scenario in ("int_escalate", "graceful"):
            if not instance.stopping.is_set():
                handlers[signal.SIGINT][0](signal.SIGINT)
                # Age only the supervisor's deadline, not the event loop clock.
                instance.stop_started -= 3.1
            if scenario == "graceful":
                child.returncode = 0
                tree.refresh.return_value = set()
            elif delay != 0.1:
                instance.stop_started -= 2
                child.returncode = -signal.SIGKILL
        else:
            child.returncode = 7 if scenario in ("crashes", "backoff_stop") else 0
        await real_sleep(0)

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    monkeypatch.setattr(instance, "restart_delay", lambda: 0)
    if scenario == "backoff_stop":

        def backoff():
            instance.stop(signal.SIGTERM)
            return 1

        monkeypatch.setattr(instance, "restart_delay", backoff)
    if scenario == "wait_error":
        calls = [0]

        async def broken_wait():
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("watcher failed")
            child.returncode = 0
            return 0

        monkeypatch.setattr(instance, "wait_exit", broken_wait)
    emit = Mock(wraps=instance.emit)
    monkeypatch.setattr(instance, "emit", emit)
    try:
        if scenario in ("spawn_error", "wait_error"):
            with pytest.raises((OSError, RuntimeError)):
                await instance.run()
        else:
            result = await instance.run()
            if scenario == "crashes":
                assert result == 7
                assert create.await_count == 5
                assert instance.restarts == 4
                assert instance.state == "failed"
            else:
                expected_signal = (
                    signal.SIGINT
                    if scenario in ("int_escalate", "graceful")
                    else signal.SIGTERM
                )
                assert result == 128 + expected_signal
                if instance.child is not None and scenario != "backoff_stop":
                    # backoff_stop follows a genuine crash (code 7).
                    emitted = [c.args[0] for c in emit.call_args_list]
                    assert "runner_crashed" not in emitted
                    assert "runner_exited" in emitted
        assert not directory.exists()
        assert not instance.index_entry.exists()
        server.close.assert_called_once_with()
        server.wait_closed.assert_awaited_once_with()
        assert remove.call_count == restore.call_count == 3
        if instance.child is not None:
            tree.stop_tracking.assert_called()
            assert signal.SIGKILL in [call.args[0] for call in tree.send.call_args_list]
            with pytest.raises(OSError):
                os.fstat(listener)
            assert "discovery index publication failed (OSError)" in caplog.text
        if scenario == "int_escalate":
            assert signal.SIGTERM in [call.args[0] for call in tree.send.call_args_list]
            assert signal.SIGINT not in [
                call.args[0] for call in tree.send.call_args_list
            ]
    finally:
        if instance.child is None:
            os.close(listener)


def test_supervise_fallback_and_post_spawn_propagation(monkeypatch, caplog):
    from unittest.mock import AsyncMock

    from rapid_mlx.control import supervisor as module

    instance = Supervisor([], port=0, mode="lazy")
    monkeypatch.setattr(module, "Supervisor", lambda *a, **kw: instance)
    monkeypatch.setattr(instance, "run", AsyncMock(side_effect=OSError("setup")))
    assert module.supervise([]) == (
        "Control plane setup failed (OSError); serving in single process."
    )
    assert not caplog.text
    instance.child = SimpleNamespace(pid=123, returncode=0)
    with pytest.raises(OSError, match="setup"):
        module.supervise([])


def test_native_process_lookup_failures_and_libproc_signatures(monkeypatch):
    import ctypes
    from unittest.mock import Mock

    from rapid_mlx.control import process_tree as module

    lib = SimpleNamespace(
        proc_pidinfo=Mock(return_value=0), proc_listallpids=Mock(return_value=0)
    )
    loader = Mock(return_value=lib)
    module._libproc.cache_clear()
    monkeypatch.setattr(module.ctypes, "CDLL", loader)
    try:
        assert module._libproc() is lib
        assert module._libproc() is lib
        loader.assert_called_once_with("/usr/lib/libproc.dylib", use_errno=True)
        assert lib.proc_pidinfo.restype is ctypes.c_int
        assert len(lib.proc_pidinfo.argtypes) == 5
        assert lib.proc_listallpids.argtypes == [ctypes.c_void_p, ctypes.c_int]
        monkeypatch.setattr(module.sys, "platform", "darwin")
        assert module.process_info(123) is None
        for counts in ([0], [2, 0]):
            lib.proc_listallpids.side_effect = counts
            with pytest.raises(OSError, match="Cannot enumerate"):
                module.process_table()
        monkeypatch.setattr(module.sys, "platform", "linux")
        monkeypatch.setattr(module.Path, "read_text", Mock(side_effect=OSError))
        assert module.process_info(123) is None
        monkeypatch.setattr(module.Path, "read_text", lambda _: "broken stat")
        assert module.process_info(123) is None
        monkeypatch.setattr(
            module.Path,
            "iterdir",
            lambda _: iter([Path("/proc/123"), Path("/proc/sys")]),
        )
        monkeypatch.setattr(module, "process_info", lambda pid: (1, "S", 456))
        assert module.process_table() == {123: (1, "S", 456)}
        monkeypatch.setattr(module.sys, "platform", "unsupported")
        with pytest.raises(OSError, match="Unsupported"):
            module.process_table()
    finally:
        module._libproc.cache_clear()


def test_unknown_platform_process_info(monkeypatch):
    from rapid_mlx.control import process_tree

    monkeypatch.setattr(process_tree.sys, "platform", "unsupported")
    assert process_tree.process_info(123) is None


def test_tree_root_state_signals_and_orphan_tracking(monkeypatch):
    from unittest.mock import Mock

    from rapid_mlx.control import process_tree as module

    monkeypatch.setattr(module, "process_info", lambda pid: (1, "T", 456))
    tree = module.ProcessTree(123)
    tree.update_root_state()
    assert tree.stopped
    kill = Mock()
    monkeypatch.setattr(module.os, "kill", kill)
    tree.send_root(signal.SIGTERM)
    kill.assert_called_once_with(123, signal.SIGTERM)
    monkeypatch.setattr(module, "process_info", lambda pid: None)
    tree.update_root_state()
    assert not tree.stopped
    tree.send_root(signal.SIGKILL)
    assert kill.call_count == 1
    tracked = Mock()
    monkeypatch.setattr(module, "_orphan_tree", None)
    constructor = Mock(return_value=tracked)
    monkeypatch.setattr(module, "ProcessTree", constructor)
    module.start_orphan_tracking()
    module.start_orphan_tracking()
    constructor.assert_called_once_with(os.getpid())
    tracked.start_tracking.assert_called_once_with()


def test_orphan_cleanup_stops_when_only_root_remains(monkeypatch):
    from unittest.mock import Mock

    from rapid_mlx import _parent_watchdog
    from rapid_mlx.control import process_tree as module

    tree = Mock()
    tree.refresh.return_value = {os.getpid()}
    monkeypatch.setattr(module, "_orphan_tree", tree)
    callback = Mock()
    monkeypatch.setattr(_parent_watchdog, "_default_on_orphan", callback)
    module.supervised_on_orphan(123, 1)
    assert tree.send.call_args_list[0].args == (signal.SIGTERM,)
    assert tree.send.call_args_list[1].args == (signal.SIGKILL,)
    callback.assert_called_once_with(123, 1, grace=1.0)


def test_protocol_transport_and_bounds(monkeypatch):
    from unittest.mock import Mock

    from rapid_mlx.control import protocol

    sock = Mock()
    monkeypatch.setattr(protocol.socket, "socket", Mock(return_value=sock))
    conn = protocol.UnixConnection("/private/control.sock")
    conn.connect()
    sock.settimeout.assert_called_once_with(protocol.TIMEOUT)
    sock.connect.assert_called_once_with("/private/control.sock")
    sock.connect.side_effect = OSError("connect failed")
    with pytest.raises(OSError, match="connect failed"):
        conn.connect()
    sock.close.assert_called_once_with()
    connection = Mock()
    monkeypatch.setattr(protocol, "UnixConnection", Mock(return_value=connection))
    with pytest.raises(ValueError, match="request too large"):
        protocol.request("socket", "secret", "POST", "x" * protocol.MAX_BODY)
    for code, data in (
        (403, b"{}"),
        (200, b"x" * (protocol.MAX_RESPONSE + 1)),
        (200, b'{"port":8123}'),
    ):
        response = Mock(status=code)
        response.read.return_value = data
        connection.getresponse.return_value = response
        if code == 200 and len(data) <= protocol.MAX_RESPONSE:
            assert protocol.request("socket", "secret") == {"port": 8123}
        else:
            with pytest.raises(OSError, match="invalid control response"):
                protocol.request("socket", "secret")
        response.read.assert_called_once_with(protocol.MAX_RESPONSE + 1)
    assert connection.close.call_count == 3


def test_startup_reports_and_exit_hook(monkeypatch):
    import atexit
    from unittest.mock import Mock

    from rapid_mlx.control import protocol

    monkeypatch.setattr(protocol, "_startup_terminal", False)
    reporter = Mock()
    monkeypatch.setattr(protocol, "request", reporter)
    monkeypatch.delenv(RUNNER_ENV, raising=False)
    protocol.report("bound", port=8123)
    reporter.assert_not_called()
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "control.sock")
    monkeypatch.setenv(SECRET_ENV, "secret")
    registered = []
    monkeypatch.setattr(atexit, "register", registered.append)
    protocol.install_startup_report()
    assert len(registered) == 1
    registered[0]()
    reporter.assert_called_once_with(
        "control.sock",
        "secret",
        "POST",
        {
            "event": "startup_failed",
            "pid": os.getpid(),
            "reason": "runner exited before readiness",
        },
    )
    protocol.report_startup_failure("duplicate")
    assert reporter.call_count == 1
    monkeypatch.setattr(protocol, "_startup_terminal", False)
    protocol.report("bound", port=8123, ready=False)
    assert not protocol._startup_terminal
    reporter.side_effect = OSError("unavailable")
    protocol.report("bound", port=8123)
    assert protocol._startup_terminal


def test_private_directory_rejects_unowned_fd(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from rapid_mlx.control import protocol

    original_fstat = os.fstat

    def unowned(fd):
        fields = list(original_fstat(fd))
        fields[4] = os.getuid() + 1
        return os.stat_result(fields)

    close = Mock(wraps=os.close)
    with monkeypatch.context() as patch:
        patch.setattr(protocol.os, "fstat", unowned)
        patch.setattr(protocol.os, "close", close)
        with pytest.raises(OSError, match="owned directory"):
            protocol.private_directory(tmp_path / "private")
    close.assert_called_once()


def test_discovery_skips_unsafe_files_and_runtime_directories(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from rapid_mlx.control import protocol

    root = tmp_path / "index"
    root.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o755)
    entry = root / "a.json"
    entry.write_text(
        json.dumps({"socket": str(runtime / "control.sock"), "secret": "secret"})
    )
    entry.chmod(0o644)
    monkeypatch.setattr(protocol, "discovery_root", lambda: root)
    transport = Mock(return_value={"port": 8123})
    monkeypatch.setattr(protocol, "request", transport)
    assert protocol.discover() == []
    entry.chmod(0o600)
    assert protocol.discover() == []
    transport.assert_not_called()
    runtime.chmod(0o700)
    assert protocol.discover(9999) == []
    assert protocol.discover(8123) == [{"port": 8123}]
    assert entry.exists()


async def test_status_forwarder_invalid_peer_and_transport_failure(monkeypatch):
    from fastapi import HTTPException
    from starlette.requests import Request

    from rapid_mlx.routes import control_status

    with pytest.raises(HTTPException) as exc:
        await control_status.require_loopback(
            Request({"type": "http", "client": ("invalid-ip", 123)})
        )
    assert exc.value.status_code == 403
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "socket")
    monkeypatch.setenv(SECRET_ENV, "secret")

    def unavailable(*args):
        raise OSError("unavailable")

    monkeypatch.setattr(control_status, "control_request", unavailable)
    app = FastAPI()
    control_status.mount_status(app)
    with TestClient(app, client=("127.0.0.1", 123)) as client:
        response = client.get("/rapid/v1/status")
    assert response.status_code == 503
    assert response.json()["detail"] == "Control plane unavailable"


@pytest.fixture
def short_control_root():
    import tempfile

    scratch = Path("/tmp/rapid-mlx-serve-cov")
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="socket-", dir=scratch) as directory:
        yield Path(directory)


@pytest.mark.parametrize("failure", ["mode", "resolve", "supervise"])
def test_control_dispatch_setup_failure_in_process(monkeypatch, capsys, failure):
    from unittest.mock import Mock

    from rapid_mlx import cli
    from rapid_mlx.control import modes, supervisor

    for key in (RUNNER_ENV, "RAPID_MLX_SINGLE_PROCESS", "RAPID_MLX_WATCHDOG_PPID"):
        monkeypatch.delenv(key, raising=False)
    args = build_parser().parse_args(["serve", "qwen3.5-4b-4bit"])
    mode = Mock(return_value="text-vlm")
    resolve = Mock(return_value=8123)
    supervise = Mock(return_value=None)
    if failure == "mode":
        mode.side_effect = ValueError("bad mode")
    elif failure == "resolve":
        resolve.side_effect = OSError("bad port")
    monkeypatch.setattr(modes, "supervised_mode", mode)
    monkeypatch.setattr(cli, "_resolve_serve_port", resolve)
    monkeypatch.setattr(supervisor, "supervise", supervise)
    assert cli._dispatch_serve_control(args) is None
    if failure == "supervise":
        supervise.assert_called_once()
        assert capsys.readouterr().err == ""
    else:
        supervise.assert_not_called()
        assert "serving in single process" in capsys.readouterr().err


def test_discovery_missing_stale_busy_and_status_output(monkeypatch, tmp_path, capsys):
    import http.client
    from unittest.mock import Mock

    from rapid_mlx.control import protocol

    root = tmp_path / "index"
    monkeypatch.setattr(protocol, "discovery_root", lambda: root)
    assert protocol.discover() == []
    root.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    entry = root / "a.json"
    entry.write_text(
        json.dumps({"socket": str(runtime / "control.sock"), "secret": "secret"})
    )
    entry.chmod(0o600)
    transport = Mock(side_effect=http.client.HTTPException("busy"))
    monkeypatch.setattr(protocol, "request", transport)
    assert protocol.discover() == []
    assert entry.exists()
    transport.side_effect = ConnectionRefusedError
    assert protocol.discover() == []
    assert not entry.exists()
    results = [{"port": 8123}, {"port": 8124}]
    monkeypatch.setattr(protocol, "discover", lambda port: results)
    assert protocol.status_command(SimpleNamespace(port=None)) == 0
    assert json.loads(capsys.readouterr().out) == results
    results.pop()
    assert protocol.status_command(SimpleNamespace()) == 0
    assert json.loads(capsys.readouterr().out) == results[0]


def test_runtime_root_and_unsafe_discovery_root(monkeypatch, tmp_path):
    from rapid_mlx.control import protocol

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert protocol.runtime_root() == tmp_path / f"rmlx-cp-{os.getuid()}"
    monkeypatch.delenv("TMPDIR")
    assert protocol.runtime_root() == Path("/tmp") / f"rmlx-cp-{os.getuid()}"
    root = tmp_path / "index"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    monkeypatch.setattr(protocol, "discovery_root", lambda: root)
    assert protocol.discover() == []


@pytest.mark.parametrize("collision", [False, True])
def test_bind_preflight_failures_report_terminal_event(monkeypatch, capsys, collision):
    from unittest.mock import Mock

    from rapid_mlx import cli
    from rapid_mlx.control import protocol
    from rapid_mlx.telemetry import server_start

    report = Mock()
    monkeypatch.setattr(protocol, "report", report)
    monkeypatch.setattr(server_start, "failed", Mock())
    with pytest.raises(SystemExit) as exc:
        if collision:
            cli._exit_for_port_collision(
                8123, "127.0.0.1", model="model", port_explicit=True
            )
        else:
            cli._exit_for_host_bind_error("invalid-host", OSError("bad bind"))
    assert exc.value.code == (1 if collision else 2)
    report.assert_called_once_with("bind_failed")
    output = capsys.readouterr()
    assert (
        "already in use" in output.out if collision else "Invalid --host" in output.err
    )


async def test_uvicorn_lifespan_failure_without_system_exit(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    import uvicorn

    from rapid_mlx import _uvicorn
    from rapid_mlx.control import protocol
    from rapid_mlx.telemetry import server_start

    server = _uvicorn.AcceptingConnectionsServer(uvicorn.Config(FastAPI()))
    server.lifespan = SimpleNamespace(should_exit=True)
    monkeypatch.setattr(server, "_startup_with_progress", AsyncMock())
    report = Mock()
    failed = Mock()
    monkeypatch.setattr(protocol, "report_startup_failure", report)
    monkeypatch.setattr(server_start, "failed", failed)
    with pytest.raises(SystemExit) as exc:
        await server.startup()
    assert exc.value.code == _uvicorn.STARTUP_FAILURE
    report.assert_called_once_with("lifespan startup failed")
    failed.assert_called_once_with("engine_start")
    app = FastAPI()

    @app.get("/peer")
    async def peer(request: Request):
        return {"captured": "_rapid_mlx_transport_peer" in request.scope}

    from rapid_mlx.control import observer

    monkeypatch.setattr(observer, "_delivery", observer.Delivery())
    config = _uvicorn._TransportPeerConfig(app, proxy_headers=False)
    monkeypatch.setenv(RUNNER_ENV, "1")
    config.load()
    assert config.loaded
    with TestClient(config.loaded_app) as client:
        assert client.get("/peer").json() == {"captured": False}
