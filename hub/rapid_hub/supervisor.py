"""Separate origins and process groups for trusted plugins."""

import asyncio
import os
import secrets
import signal
import socket
import sys
from pathlib import Path


class Supervisor:
    def __init__(self, hub, manifests, retries=5, delay=1, ready_timeout=20):
        self.hub, self.manifests = hub, manifests
        self.retries, self.delay, self.ready_timeout = retries, delay, ready_timeout
        self.items, self.tasks = {}, {}
        self.locks = {ident: asyncio.Lock() for ident in manifests}
        self.stopping = False
        for ident, manifest in manifests.items():
            token = secrets.token_urlsafe(32)
            hub.tokens[token] = ident
            self.items[ident] = dict(
                id=ident,
                name=manifest["name"],
                state="stopped",
                token=token,
                process=None,
            )

    async def start(self):
        catalog = await self.hub.catalog()
        capabilities = {
            "models." + c for m in catalog["data"] for c in m["capabilities"]
        }
        for ident, manifest in self.manifests.items():
            missing = [
                r
                for r in manifest.get("requires", [])
                if not r.endswith("?") and r not in capabilities
            ]
            if missing:
                self.items[ident].update(
                    state="blocked", error="Missing capabilities: " + ", ".join(missing)
                )
            else:
                self.tasks[ident] = asyncio.create_task(self.run(ident))

    async def terminate(self, item):
        process = item.get("process")
        if process is None:
            return
        # Kill the group even if its leader has already exited.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            item["process"] = None
            return
        except PermissionError:
            # Some macOS sandboxes return EPERM rather than ESRCH for a group
            # whose reaped guardian and descendants have already disappeared.
            if process.returncode is None:
                raise
            item["process"] = None
            return
        try:
            await asyncio.wait_for(process.wait(), 3)
        except asyncio.TimeoutError:
            pass
        # Descendants may outlive a leader that exited promptly on SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.returncode is None:
                raise
        await process.wait()
        item["process"] = None

    async def run(self, ident):
        item, manifest = self.items[ident], self.manifests[ident]
        root = self.hub.data_dir / "plugins" / ident
        for folder in ("data", "logs"):
            (root / folder).mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            for attempt in range(self.retries + 1):
                self.hub.tokens.pop(item["token"], None)
                item["token"] = secrets.token_urlsafe(32)
                self.hub.tokens[item["token"]] = ident
                item.update(state="starting", restarts=attempt)
                self.hub.emit("plugin_state", {"id": ident, "state": "starting"})
                with socket.socket() as sock:
                    sock.bind((self.hub.plugin_host, 0))
                    port = sock.getsockname()[1]
                item["port"] = port
                env = {
                    k: v
                    for k, v in os.environ.items()
                    if not any(
                        s in k.upper()
                        for s in (
                            "TOKEN",
                            "SECRET",
                            "PASSWORD",
                            "API_KEY",
                            "CREDENTIAL",
                        )
                    )
                }
                env.update(
                    RAPID_HUB_URL=self.hub.url,
                    RAPID_HUB_TOKEN=item["token"],
                    RAPID_PLUGIN_ID=ident,
                    RAPID_PLUGIN_HOST=self.hub.plugin_host,
                    RAPID_PLUGIN_PORT=str(port),
                    RAPID_PLUGIN_DATA_DIR=str(root / "data"),
                )
                command = [
                    s.format(python=sys.executable, plugin_dir=manifest["directory"])
                    for s in manifest["run"]["command"]
                ]
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("runner.py")),
                    *command,
                ]
                try:
                    with (
                        (root / "logs" / "stdout.log").open("ab") as out,
                        (root / "logs" / "stderr.log").open("ab") as err,
                    ):
                        item["process"] = await asyncio.create_subprocess_exec(
                            *command,
                            env=env,
                            cwd=manifest["directory"],
                            stdout=out,
                            stderr=err,
                            start_new_session=True,
                        )
                    deadline = asyncio.get_running_loop().time() + self.ready_timeout
                    while (
                        item["process"].returncode is None
                        and asyncio.get_running_loop().time() < deadline
                    ):
                        try:
                            response = await self.hub.client.get(
                                f"http://{self.hub.plugin_host}:{port}"
                                + manifest["ui"]["health"],
                                timeout=1,
                            )
                            if response.status_code == 200:
                                item["state"] = "ready"
                                self.hub.emit(
                                    "plugin_state", {"id": ident, "state": "ready"}
                                )
                                break
                        except Exception:
                            pass
                        await asyncio.sleep(0.1)
                    if item["state"] == "ready":
                        await item["process"].wait()
                except OSError as exc:
                    item["error"] = str(exc)
                finally:
                    self.hub.tokens.pop(item["token"], None)
                    await self.terminate(item)
                if attempt < self.retries:
                    item["state"] = "backoff"
                    self.hub.emit("plugin_state", {"id": ident, "state": "backoff"})
                    await asyncio.sleep(min(self.delay * 2**attempt, 16))
            item["state"] = "failed"
            self.hub.emit("plugin_state", {"id": ident, "state": "failed"})
        finally:
            self.hub.tokens.pop(item["token"], None)
            await self.terminate(item)

    async def restart(self, ident):
        async with self.locks[ident]:
            if self.stopping:
                return
            old = self.items[ident]
            self.hub.tokens.pop(old["token"], None)
            task = self.tasks.get(ident)
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            # A runner owns its record for its entire lifetime.
            self.items[ident] = dict(old, state="starting", process=None)
            self.tasks[ident] = asyncio.create_task(self.run(ident))

    async def stop(self):
        self.stopping = True

        async def stop_one(ident):
            async with self.locks[ident]:
                task = self.tasks.get(ident)
                if task:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        await asyncio.gather(*(stop_one(ident) for ident in self.locks))

    def public(self):
        result = []
        for ident, item in self.items.items():
            row = {k: v for k, v in item.items() if k not in ("token", "process")}
            task = self.tasks.get(ident)
            if task and task.done() and not task.cancelled() and task.exception():
                row.update(state="failed", error=str(task.exception()))
            row["url"] = (
                f"http://{ident}.localhost:{item['port']}" if "port" in item else None
            )
            row["supervisor_pid"] = item["process"].pid if item.get("process") else None
            pid_path = self.hub.data_dir / "plugins" / ident / "data" / "plugin.pid"
            row["pid"] = None
            if item.get("process") and pid_path.exists():
                try:
                    row["pid"] = int(pid_path.read_text())
                except (OSError, ValueError):
                    pass
            row["logs"] = {}
            for name in ("stdout", "stderr"):
                path = self.hub.data_dir / "plugins" / ident / "logs" / (name + ".log")
                if path.exists():
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 8192))
                        row["logs"][name] = (
                            stream.read().decode(errors="replace").splitlines()[-20:]
                        )
            result.append(row)
        return result
