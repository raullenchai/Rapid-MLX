# SPDX-License-Identifier: Apache-2.0
"""Lifecycle of the supervised ``rapid-mlx serve`` child.

Owning the child, rather than pointing at one the user started, is what
keeps the external port fixed: switching models has no hot-swap path — a
different model is a different process — so a page pointed straight at
the engine would break on every switch. The child also gets an ephemeral
port picked here, so this can run alongside an existing ``rapid-mlx
serve``.

The child is driven as a subprocess of the CLI, never by importing
``vllm_mlx``: the contract is the documented command line, which is what
keeps this package installable and testable without the engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import httpx

# A cold start compiles Metal shaders and may pull weights, so the ceiling
# is minutes. Too low shows up as a spurious "failed to start" on exactly
# the large models people most want to run.
DEFAULT_READY_TIMEOUT_S = 900.0

# Evict an idle unpinned secondary model after half an hour.
_RESIDENT_IDLE_TTL_S = 1800


def resident_memory_ceiling_gb() -> int:
    """The engine's resident-model ceiling, in GiB.

    80% of physical RAM, floored, with a 4 GiB minimum.

    Passed on every spawn because the engine's default is 0, which disables
    the ceiling entirely: ``/v1/models/residency`` then reports a limit of
    zero and the page has a numerator with no denominator.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 4
    return max(4, int(total / 1024**3 * 0.80))


# The child is doing GPU work; polling tightly buys nothing.
_READY_POLL_INTERVAL_S = 1.0

# SIGTERM→SIGKILL grace. mlx releases GPU buffers on the way out; killing
# immediately leaves wired memory attributed to a dead process.
_TERM_GRACE_S = 10.0


class ChildState(str, Enum):
    """Coarse state of the supervised engine, as reported to the page."""

    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


class ResidencyOutcome(str, Enum):
    """What a hot ``POST /v1/models/load`` did.

    Only ``LOADED`` avoids a respawn. The rest are all "fall back to
    restarting the child", but they are distinguished because the caller
    surfaces different copy for a capacity refusal than for a transport
    error, and because ``UNSUPPORTED`` must not be reported to the user at
    all — it is the ordinary path on an older engine.
    """

    LOADED = "loaded"
    #: The engine refused: 507 over the ceiling, 409 busy, 500 modality.
    REJECTED = "rejected"
    #: No such route (old engine), or the engine was unreachable.
    UNSUPPORTED = "unsupported"


class SupervisorError(RuntimeError):
    """The child could not be started, or died during startup."""


@dataclass
class ChildStatus:
    """Snapshot handed to ``/api/status``.

    A value object rather than a live view: the HTTP handler serialises it
    after the lock is released, so it must not change underneath.
    """

    state: ChildState
    model: str | None = None
    port: int | None = None
    detail: str | None = None
    recent_output: list[str] = field(default_factory=list)
    #: Aliases loaded into this child, INCLUDING ``model``. More than one
    #: once a hot load succeeds — a chat model and an image model can be
    #: resident together, and the page needs both to consider each usable.
    resident: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "model": self.model,
            "port": self.port,
            "detail": self.detail,
            "resident": list(self.resident),
        }


def find_rapid_mlx_binary(explicit: str | None = None) -> str:
    """Locate the ``rapid-mlx`` command.

    Precedence order, so a user with several installs (venv, Homebrew,
    source checkout) can be explicit without editing PATH.
    """
    if explicit:
        if os.path.isabs(explicit) and not os.access(explicit, os.X_OK):
            raise SupervisorError(f"not executable: {explicit}")
        return explicit

    env_override = os.environ.get("RAPID_MLX_BIN")
    if env_override:
        return env_override

    found = shutil.which("rapid-mlx") or shutil.which("rmlx")
    if not found:
        raise SupervisorError(
            "could not find the `rapid-mlx` command on PATH. Install it with "
            "`pip install rapid-mlx`, or pass --rapid-mlx-bin /path/to/rapid-mlx."
        )
    return found


def _replacement_group(modality: str) -> str:
    """The engine's replacement group for a modality.

    Mirrors ``resident_models._replacement_group``: text and vision share
    the single-slot ``assistant`` group, every media modality owns its own.
    That is precisely why a chat model and an image model can be resident
    together while two chat models cannot.
    """
    return "assistant" if modality in ("text", "vision") else modality


def _residency_refusal(response: httpx.Response) -> str | None:
    """The engine's explanation for refusing a resident load.

    Its 507 body carries a ``replacement_projection`` naming exactly which
    models it would have had to evict — worth surfacing verbatim, since no
    message composed here knows that. FastAPI nests a dict ``detail``, so
    the engine's ``{"error": {...}}`` envelope arrives one level down.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None

    detail = body.get("detail", body)
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        error = detail.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return None


def pick_free_port() -> int:
    """Ask the OS for an unused localhost port.

    Bind-then-close leaves a race window. Accepted: the alternative
    (handing the child an inherited socket) couples to engine internals,
    and a collision surfaces immediately as a startup failure.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EngineSupervisor:
    """Owns at most one ``rapid-mlx serve`` child process."""

    # The attached variant below sets this False so the HTTP layer can
    # refuse up front instead of raising mid-request.
    can_switch = True

    def __init__(
        self,
        *,
        binary: str,
        api_key: str,
        serve_args: list[str] | None = None,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        mcp_config_path: Callable[[], str | None] | None = None,
    ) -> None:
        self._binary = binary
        self._api_key = api_key
        self._serve_args = list(serve_args or [])
        self._ready_timeout_s = ready_timeout_s
        # A callable, not a path: ``--mcp-config`` is read once at spawn, so
        # the value that matters is the one true at the MOMENT of the spawn.
        # A snapshot taken at construction would arm connectors the user
        # switched off half an hour ago, and miss ones they just added.
        self._mcp_config_path = mcp_config_path or (lambda: None)

        self._process: asyncio.subprocess.Process | None = None
        self._model: str | None = None
        self._port: int | None = None
        self._state = ChildState.STOPPED
        self._detail: str | None = None
        # Aliases this child holds. Reset by a respawn (a new process holds
        # nothing), appended to by a successful hot load. `_model` stays the
        # PRIMARY — the one the child was spawned for and the one the engine
        # falls back to for an unrouted request.
        self._resident: list[str] = []
        # Modality per resident alias, so the group arithmetic above does not
        # have to re-consult the catalog.
        self._modalities: dict[str, str] = {}
        # Startup failures (bad alias, OOM, missing checkpoint) are
        # explained in the child's stderr and nowhere else. Bounded
        # because a long-running server logs every request.
        self._output_tail: list[str] = []
        self._drain_task: asyncio.Task | None = None
        # Two concurrent switch requests would otherwise both spawn a
        # child and leak one.
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str | None:
        """Where the child is listening, or ``None`` if it is not."""
        if self._port is None or self._state is not ChildState.READY:
            return None
        return f"http://127.0.0.1:{self._port}"

    @property
    def api_key(self) -> str:
        return self._api_key

    def status(self) -> ChildStatus:
        return ChildStatus(
            state=self._state,
            model=self._model,
            port=self._port,
            detail=self._detail,
            recent_output=list(self._output_tail),
            resident=list(self._resident),
        )

    async def start(self, model: str, *, modality: str = "text") -> None:
        """Spawn the child for ``model`` and wait until it is ready.

        ``modality`` only groups the spawned model for later hot loads; the
        engine detects the real one itself from the checkpoint.
        """
        async with self._lock:
            await self._stop_locked()
            await self._start_locked(model, modality)

    async def residency_load(
        self,
        model: str,
        *,
        modality: str,
        size_bytes: int | None = None,
        image_mode: str | None = None,
    ) -> tuple[ResidencyOutcome, str | None]:
        """Load ``model`` into the RUNNING child, without respawning.

        This is what lets a chat model and an image model be resident at
        once: the engine groups ``text``/``vision`` into one single-slot
        ``assistant`` group and gives every media modality its own, so a
        second text model evicts the first while an image model coexists.

        Returns the outcome and, on a refusal, the engine's own explanation
        — it names the models it would have to evict, which no message
        composed here could.

        The whole call is best-effort by design: every failure mode maps to
        "restart the child instead", which is what this package did
        unconditionally before.
        """
        base_url = self.base_url
        if base_url is None:
            return ResidencyOutcome.UNSUPPORTED, None

        payload: dict[str, object] = {"model": model}
        # The catalog's byte count, never a name-parsed estimate. The
        # engine's own fallback regexes a parameter count out of the alias
        # and sizes `embeddinggemma-300m-6bit` to zero — passing a measured
        # size is what stops a correct load being refused over the ceiling.
        if size_bytes is not None and size_bytes > 0:
            payload["estimated_size_gb"] = round(size_bytes / 1024**3, 3)
        # Only `assistant` is accepted on the wire; media groups are derived
        # by the engine and are implicitly single-slot already.
        if modality in ("text", "vision"):
            payload["replace_group"] = "assistant"
        if image_mode is not None:
            payload["image_mode"] = image_mode

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            # No read timeout: a resident load is a real model load, minutes
            # on a cold cache. Bounded connect so a dead child fails fast.
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
            ) as client:
                response = await client.post(
                    f"{base_url}/v1/models/load", json=payload, headers=headers
                )
        except httpx.HTTPError:
            return ResidencyOutcome.UNSUPPORTED, None

        if response.status_code == 200:
            self._record_resident(model, modality)
            return ResidencyOutcome.LOADED, None
        # 404/405 is an engine predating the route — the ordinary path on an
        # older install, and not something to report as a failure.
        if response.status_code in (404, 405):
            return ResidencyOutcome.UNSUPPORTED, None
        return ResidencyOutcome.REJECTED, _residency_refusal(response)

    def _record_resident(self, model: str, modality: str) -> None:
        """Mirror the engine's group bookkeeping after a successful load.

        Mirrored rather than re-read from ``/v1/models/residency`` because
        that reports the engine's canonical ids (``mlx-community/…``) while
        every other surface here speaks catalog aliases, and mapping between
        them needs an alias list the snapshot does not always carry.

        The engine's grouping is what is reproduced: ``text``/``vision``
        share one single-slot ``assistant`` group, and each media modality
        gets its own. So a second text model REPLACES the first, while an
        image model joins it.
        """
        group = _replacement_group(modality)
        self._resident = [
            alias
            for alias in self._resident
            if alias != model
            and _replacement_group(self._modalities.get(alias, "text")) != group
        ]
        self._resident.append(model)
        self._modalities[model] = modality
        # The primary is whichever assistant-group model is loaded: the
        # engine promotes a replacement to primary, and an unrouted request
        # falls back to it.
        if group == "assistant":
            self._model = model

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _start_locked(self, model: str, modality: str = "text") -> None:
        port = pick_free_port()
        argv = [
            self._binary,
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            # Mount ``/v1/audio/*`` on EVERY child, so speech runs beside the
            # primary model in the same process. The gate short-circuits on
            # this flag before it looks at the model, so a text server gets
            # the lane too — and the STT/TTS engines stay lazy, loading only
            # on the first audio request, so the flag costs nothing until used.
            "--enable-audio",
            # Without a ceiling the engine's residency snapshot reports a
            # limit of 0, and the memory panel has nothing to measure
            # against.
            "--resident-memory-limit-gb",
            str(resident_memory_ceiling_gb()),
            "--resident-model-idle-ttl",
            str(_RESIDENT_IDLE_TTL_S),
        ]

        # Connectors are armed at spawn or not at all: the engine reads this
        # file once and builds its MCP subsystem from it. That is why turning
        # the master switch on cannot take effect on a running child, and why
        # the page offers a Restart rather than pretending it can.
        mcp_config = self._mcp_config_path()
        if mcp_config:
            argv += ["--mcp-config", mcp_config]

        argv += self._serve_args

        env = dict(os.environ)
        # The bearer travels by environment, not argv: on macOS `ps -axww`
        # shows argv to any user, while `ps eww` gates environment behind
        # same-UID-or-root.
        env["RAPID_MLX_API_KEY"] = self._api_key

        self._state = ChildState.STARTING
        self._model = model
        self._port = port
        self._detail = None
        self._output_tail = []
        # Nothing is resident until the child is READY. Recording the alias
        # here instead would claim a model is loaded in a process that may
        # still fail to spawn — and the page would enable Send against it.
        self._resident = []
        self._modalities = {}

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                # Own process group: without this a Ctrl-C in the terminal
                # reaches only us and leaves a multi-GB model resident
                # with no owner.
                start_new_session=True,
            )
        except OSError as exc:
            self._state = ChildState.FAILED
            self._detail = str(exc)
            raise SupervisorError(f"failed to launch {self._binary}: {exc}") from exc

        self._process = process
        self._drain_task = asyncio.create_task(self._drain_output(process))

        try:
            await self._await_ready(process, port)
        except SupervisorError as exc:
            self._state = ChildState.FAILED
            # Recorded, not just raised: `start` is called as a detached task
            # (app.py `_boot`/`_switch` swallow the exception), so without
            # this `/api/status` reports `failed` with a null detail and the
            # page has nothing to show but "unknown error".
            self._detail = self._failure_reason() or str(exc)
            # A half-started child may still hold GPU memory, which the
            # next start would then contend with.
            await self._stop_locked(preserve_failure=True)
            raise

        self._state = ChildState.READY
        # The child bound its port, so the model it was spawned for is now
        # genuinely loaded and can be reported as resident.
        self._resident = [model]
        self._modalities = {model: modality}

    def _failure_reason(self) -> str | None:
        """The engine's own explanation, pulled out of its output.

        The CLI prints a single ``  Error: …`` line for the failures a user
        can act on — a missing extra, a partial download, an unusable alias —
        and then a page of unrelated banner text. Reporting the last few
        lines instead buries the one line that says what to do.
        """
        for line in reversed(self._output_tail):
            stripped = line.strip()
            if stripped.startswith("Error:"):
                return stripped.removeprefix("Error:").strip()
        return None

    async def _await_ready(
        self, process: asyncio.subprocess.Process, port: int
    ) -> None:
        """Poll ``/health/ready`` until the engine finishes startup.

        Not ``/v1/models``: that returns 200 as soon as FastAPI binds,
        before warmup and prefix-cache load, so a request sent in that
        window competes with warmup and looks like a hang. ``/health/ready``
        answers 503 until lifespan startup is genuinely complete.
        """
        deadline = asyncio.get_running_loop().time() + self._ready_timeout_s
        url = f"http://127.0.0.1:{port}/health/ready"

        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if process.returncode is not None:
                    # The drain task may still be reading the last of the
                    # child's output, and the `Error:` line explaining WHY it
                    # exited is typically the final thing written. Give the
                    # pipe a moment to close, or the detail is empty exactly
                    # when it is most needed.
                    if self._drain_task is not None:
                        with contextlib.suppress(
                            asyncio.TimeoutError, asyncio.CancelledError
                        ):
                            await asyncio.wait_for(
                                asyncio.shield(self._drain_task), timeout=2.0
                            )
                    raise SupervisorError(
                        "the engine exited during startup "
                        f"(code {process.returncode}). "
                        f"Last output: {self._tail_text()}"
                    )
                if asyncio.get_running_loop().time() > deadline:
                    raise SupervisorError(
                        "the engine did not become ready within "
                        f"{self._ready_timeout_s:.0f}s. "
                        f"Last output: {self._tail_text()}"
                    )
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    # Connection refused is the normal case for most of
                    # this loop — the child has not bound yet.
                    pass
                await asyncio.sleep(_READY_POLL_INTERVAL_S)

    async def _drain_output(self, process: asyncio.subprocess.Process) -> None:
        """Continuously read the child's output into a bounded tail.

        Not optional bookkeeping: stdout is a pipe with a fixed kernel
        buffer, and once it fills the child blocks on write and the engine
        stops mid-generation.
        """
        assert process.stdout is not None
        while True:
            try:
                line = await process.stdout.readline()
            except (ValueError, OSError):
                # ValueError on an over-long line with no newline. Treat as
                # end of usable output rather than killing the drain task
                # and re-introducing the stall.
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                self._output_tail.append(text)
                if len(self._output_tail) > 200:
                    del self._output_tail[:-200]

    def _tail_text(self, lines: int = 8) -> str:
        return " | ".join(self._output_tail[-lines:]) or "(no output)"

    async def _stop_locked(self, *, preserve_failure: bool = False) -> None:
        process = self._process
        if process is None:
            if not preserve_failure:
                self._state = ChildState.STOPPED
            return

        if process.returncode is None:
            try:
                # Signal the whole group: the engine spawns helpers, and
                # signalling only the leader would orphan them.
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()

            try:
                await asyncio.wait_for(process.wait(), timeout=_TERM_GRACE_S)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=_TERM_GRACE_S)

        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None

        self._process = None
        self._port = None
        # The child is gone, so nothing is resident regardless of why.
        self._resident = []
        self._modalities = {}
        if not preserve_failure:
            self._state = ChildState.STOPPED
            self._model = None


class AttachedEngine:
    """Stand-in for :class:`EngineSupervisor` in ``--attach`` mode.

    Same surface so the HTTP layer does not branch, but owns nothing.
    Switching is impossible, so callers check :attr:`can_switch` rather
    than discovering it from a failure.
    """

    can_switch = False

    def __init__(self, base_url: str, *, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or ""

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    def status(self) -> ChildStatus:
        # READY without probing: the caller asserted this endpoint exists,
        # and a probe would only move the failure to startup while needing
        # to be repeated anyway.
        return ChildStatus(state=ChildState.READY, model=None, port=None)

    async def start(self, model: str, *, modality: str = "text") -> None:
        raise SupervisorError("cannot switch models in --attach mode")

    async def stop(self) -> None:
        return None
