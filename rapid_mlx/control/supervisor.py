# SPDX-License-Identifier: Apache-2.0
"""A small stdlib-only parent; the runner owns the public listener."""

import asyncio
import collections
import contextlib
import json
import logging
import os
import queue
import secrets
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time

from . import (
    CONTROL_VERSION,
    PARENT_ENV,
    RUNNER_ENV,
    SECRET_ENV,
    SECRET_HEADER,
    SOCKET_ENV,
)
from .process_tree import ProcessTree
from .protocol import MAX_BODY, TIMEOUT, discovery_root, private_directory, runtime_root

logger = logging.getLogger(__name__)


def exit_code(code):
    return code if code >= 0 else 128 - code


class Supervisor:
    def __init__(self, argv, *, port, mode, listen_fd=None, env=None):
        self.argv = list(argv)
        self.env = dict(os.environ if env is None else env)
        self.port = port
        self.mode = mode
        self.listen_fd = listen_fd
        self.child = None
        self.tree = None
        self.started_at = None
        self.restarts = 0
        self.state = "starting"
        self.bind_failed = False
        self.startup_failed = False
        self.stopping = asyncio.Event()
        self.stop_signal = None
        self.failures = collections.deque()
        self.secret = secrets.token_urlsafe(32)
        self.directory = None
        self.server = None
        self.spawn_ready = asyncio.Event()
        self.index_entry = None
        self.stop_started = None

    def status(self):
        return {
            "control_version": CONTROL_VERSION,
            "supervisor_pid": os.getpid(),
            "runner": {
                "pid": self.child.pid if self.child else None,
                "started_at": self.started_at,
                "restarts": self.restarts,
                "state": (
                    "stopped"
                    if (
                        self.tree
                        and self.child.returncode is None
                        and self.tree.stopped
                        and not self.stopping.is_set()
                    )
                    else self.state
                ),
            },
            "port": self.port,
            "mode": self.mode,
        }

    def stop(self, signum):
        # SIGINT reaches both foreground processes. Forwarding it would make
        # Uvicorn force_exit and skip lifespan shutdown. Wait for graceful exit;
        # a PID-only SIGINT is handled by the bounded TERM -> KILL escalation.
        # Further terminal Ctrl-C reaches Uvicorn directly and can still force
        # exit. Never reset the deadline or duplicate a subsequent signal.
        if self.stopping.is_set():
            return
        self.stop_started = time.monotonic()
        if signum != signal.SIGINT:
            self.signal_descendants(signum)
        self.stop_signal = signum
        self.state = "stopping"
        self.stopping.set()

    async def handle(self, reader, writer):
        try:
            async with asyncio.timeout(TIMEOUT):
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode("ascii").split("\r\n")
                method, target, _ = lines[0].split()
                headers = {}
                for line in lines[1:]:
                    if line:
                        key, value = line.split(":", 1)
                        key = key.lower()
                        if key in headers:
                            raise ValueError("duplicate header")
                        headers[key] = value.strip()
                authorized = secrets.compare_digest(
                    headers.get(SECRET_HEADER.lower(), ""), self.secret
                )
                length = int(headers.get("content-length", "0"))
                if not 0 <= length <= MAX_BODY or "transfer-encoding" in headers:
                    raise ValueError("invalid body size")
                if not authorized:
                    code, result = 403, {}
                elif method == "GET" and target == "/rapid/v1/status" and length == 0:
                    code, result = 200, self.status()
                elif method == "POST" and target == "/startup":
                    data = json.loads(await reader.readexactly(length))
                    await self.spawn_ready.wait()
                    if self.child is None or data.get("pid") != self.child.pid:
                        code, result = 409, {}
                    else:
                        event = data.get("event")
                        if event == "bound":
                            port = int(data["port"])
                            if not 1 <= port <= 65535:
                                raise ValueError("invalid port")
                            self.port = port
                            if not self.stopping.is_set() and data.get("ready", True):
                                self.state = "ready"
                        elif event == "bind_failed":
                            self.bind_failed = True
                        elif event == "startup_failed" and not self.stopping.is_set():
                            self.startup_failed = True
                            self.state = "failed"
                        else:
                            raise ValueError("unknown startup report")
                        code, result = 200, {}
                else:
                    code, result = 404, {}
                body = json.dumps(result).encode()
                writer.write(
                    f"HTTP/1.1 {code} Response\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            pass
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    async def open(self):
        index = private_directory(discovery_root())
        root = runtime_root()
        root.mkdir(mode=0o700, exist_ok=True)
        info = root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise OSError("control runtime root must be an owned 0700 directory")
        from pathlib import Path

        self.directory = Path(tempfile.mkdtemp(prefix="i-", dir=root))
        path = self.directory / "control.sock"
        self.server = await asyncio.start_unix_server(
            self.handle, path=str(path), limit=MAX_BODY
        )
        path.chmod(0o600)
        metadata = self.directory / "instance.json"
        with open(
            metadata, "x", opener=lambda p, flags: os.open(p, flags, 0o600)
        ) as output:
            json.dump({"secret": self.secret}, output)
        self.index_entry = index / (self.directory.name + ".json")
        self.env.update(
            {
                RUNNER_ENV: "1",
                SOCKET_ENV: str(path),
                SECRET_ENV: self.secret,
                PARENT_ENV: str(os.getpid()),
            }
        )

    def publish_index(self):
        temporary = self.index_entry.with_suffix(".tmp")
        try:
            with open(
                temporary, "x", opener=lambda p, f: os.open(p, f, 0o600)
            ) as output:
                json.dump(
                    {"socket": self.env[SOCKET_ENV], "secret": self.secret}, output
                )
            temporary.replace(self.index_entry)
        finally:
            temporary.unlink(missing_ok=True)

    @contextlib.contextmanager
    def best_effort(self, operation):
        try:
            yield
        except Exception as exc:
            # Pre-spawn cleanup must not mask the original setup failure or
            # add warnings to its single fallback diagnostic.
            if self.child is not None:
                logger.warning(
                    "Control plane %s failed (%s).", operation, type(exc).__name__
                )

    async def pump(self, source, sink):
        # Mirror the service's daemon pipe pumps. A stalled output consumer
        # must not make asyncio's executor join exceed the shutdown deadline.
        loop = asyncio.get_running_loop()
        inbox = queue.Queue()

        def copy():
            while (item := inbox.get()) is not None:
                chunk, completed = item
                error = None
                try:
                    self.write(sink, chunk)
                except (OSError, ValueError) as exc:
                    error = exc

                def finish(future=completed, result=error):
                    if not future.done():
                        future.set_result(result)

                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(finish)

        threading.Thread(target=copy, daemon=True).start()
        broken = False
        stopping = asyncio.create_task(self.stopping.wait())
        try:
            while chunk := await source.read(65536):
                if not broken:
                    completed = loop.create_future()
                    # Only one chunk can be outstanding: await its completion
                    # before reading again, keeping memory and queue bounded.
                    inbox.put_nowait((chunk, completed))
                    await asyncio.wait(
                        (completed, stopping), return_when=asyncio.FIRST_COMPLETED
                    )
                    if not completed.done():
                        try:
                            await asyncio.wait_for(asyncio.shield(completed), 0.1)
                        except TimeoutError:
                            # Continue draining the child so Process.wait() can
                            # finish even when its pipe transport was paused.
                            broken = True
                            continue
                    if completed.result() is not None:
                        broken = True
                        self.stop(signal.SIGTERM)
        finally:
            inbox.put_nowait(None)
            stopping.cancel()
            await asyncio.gather(stopping, return_exceptions=True)

    @staticmethod
    def write(sink, chunk):
        data = memoryview(chunk)
        while data:
            written = os.write(sink.fileno(), data)
            if written <= 0:
                raise OSError("control log pipe stopped accepting bytes")
            data = data[written:]

    def spawn_argv(self):
        if self.listen_fd is not None:
            return self.argv
        # Last option wins, but options must precede the positional separator.
        index = self.argv.index("--") if "--" in self.argv else len(self.argv)
        return [*self.argv[:index], "--port", str(self.port), *self.argv[index:]]

    def signal_descendants(self, signum):
        # The runner shares the terminal/service group. Never signal that group:
        # it can also contain the service runtime parent or the invoking shell.
        signalled = self.tree.send(signum) if self.tree is not None else set()
        if (
            self.child is not None
            and self.child.returncode is None
            and self.child.pid not in signalled
        ):
            if self.tree is not None:
                self.tree.send_root(signum)

    async def wait_exit(self):
        # asyncio Process.wait can wait for inherited pipe EOF after exit.
        # The child watcher sets returncode independently of those pipes.
        while self.child.returncode is None:
            if not self.stopping.is_set():
                self.tree.update_root_state()
            await asyncio.sleep(0.1)
        return self.child.returncode

    def close_pipes(self):
        for fd in (1, 2):
            transport = self.child._transport.get_pipe_transport(fd)
            if transport is not None:
                transport.close()

    def restart_delay(self):
        return min(60, 2 ** min(self.restarts, 6))

    def record_failure(self, now):
        self.failures.append(now)
        while self.failures and now - self.failures[0] > 600:
            self.failures.popleft()
        return len(self.failures) >= 5

    async def run(self):
        previous = {}
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous[sig] = signal.getsignal(sig)
                loop.add_signal_handler(sig, self.stop, sig)
            await self.open()
            while not self.stopping.is_set():
                self.bind_failed = False
                self.startup_failed = False
                self.started_at = time.time()
                self.spawn_ready.clear()
                self.child = await asyncio.create_subprocess_exec(
                    *self.spawn_argv(),
                    env=self.env,
                    pass_fds=(() if self.listen_fd is None else (self.listen_fd,)),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self.tree = ProcessTree(self.child.pid)
                self.tree.start_tracking()
                self.spawn_ready.set()
                with self.best_effort("discovery index publication"):
                    self.publish_index()
                if self.stopping.is_set():
                    if self.stop_signal != signal.SIGINT:
                        self.signal_descendants(self.stop_signal)
                pumps = [
                    asyncio.create_task(
                        self.pump(self.child.stdout, sys.stdout.buffer)
                    ),
                    asyncio.create_task(
                        self.pump(self.child.stderr, sys.stderr.buffer)
                    ),
                ]
                wait = asyncio.create_task(self.wait_exit())
                stopping = asyncio.create_task(self.stopping.wait())
                await asyncio.wait(
                    (wait, stopping), return_when=asyncio.FIRST_COMPLETED
                )
                if self.stopping.is_set():
                    # Keep tracked descendants through runner exit/reparenting.
                    # A stopped runner is alive too; only exit or the deadline
                    # ends this grace period, never a job-control stop.
                    deadline = self.stop_started + 4.0
                    term_sent = self.stop_signal != signal.SIGINT
                    while time.monotonic() < deadline:
                        if not term_sent and time.monotonic() >= deadline - 1.0:
                            self.signal_descendants(signal.SIGTERM)
                            term_sent = True
                        live = self.tree.refresh()
                        if wait.done() and not live:
                            break
                        await asyncio.sleep(
                            min(0.05, max(0, deadline - time.monotonic()))
                        )
                self.tree.stop_tracking()
                self.signal_descendants(signal.SIGKILL)
                code = await asyncio.wait_for(wait, 0.2)
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)
                # A runner's descendants must not hold shutdown open forever.
                drain = (
                    max(0, 4.3 - (time.monotonic() - self.stop_started))
                    if self.stop_started is not None
                    else 0.3
                )
                done, pending = await asyncio.wait(pumps, timeout=drain)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pumps, return_exceptions=True)
                self.close_pipes()
                if self.stopping.is_set():
                    return 128 + self.stop_signal
                # Retrying the same invalid invocation cannot repair startup.
                # Exit 2 also covers usage errors before startup reporting is wired.
                if self.bind_failed or self.startup_failed or code in (0, 2):
                    return exit_code(code)
                now = time.monotonic()
                if self.record_failure(now):
                    self.state = "failed"
                    return exit_code(code) or 1
                self.state = "restarting"
                delay = self.restart_delay()
                try:
                    await asyncio.wait_for(self.stopping.wait(), delay)
                except TimeoutError:
                    self.restarts += 1
                    self.state = "starting"
            return 128 + self.stop_signal
        finally:
            if self.tree is not None:
                self.tree.stop_tracking()
            if self.child is not None:
                self.signal_descendants(signal.SIGKILL)
                self.close_pipes()
                if self.child.returncode is None:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self.wait_exit(), 0.2)
            if self.server:
                with self.best_effort("socket cleanup"):
                    self.server.close()
                    await self.server.wait_closed()
            if self.index_entry:
                with self.best_effort("discovery index cleanup"):
                    self.index_entry.unlink(missing_ok=True)
            if self.directory:
                with self.best_effort("runtime directory cleanup"):
                    shutil.rmtree(self.directory)
            if self.listen_fd is not None and self.child is not None:
                with self.best_effort("listener cleanup"):
                    os.close(self.listen_fd)
            for sig, handler in previous.items():
                with self.best_effort("signal handler cleanup"):
                    loop.remove_signal_handler(sig)
                    signal.signal(sig, handler)


def supervise(argv, **kwargs):
    """None means setup failed before spawning; continue the legacy serve path."""
    supervisor = None
    try:
        supervisor = Supervisor(argv, **kwargs)
        return asyncio.run(supervisor.run())
    except Exception as exc:
        if supervisor is not None and supervisor.child is not None:
            raise
        logger.warning(
            "Control plane setup failed (%s); serving in single process.",
            type(exc).__name__,
        )
        return None
