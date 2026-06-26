# SPDX-License-Identifier: Apache-2.0
"""Parent-PID watchdog for embedded ``rapid-mlx serve`` children.

Problem (rapid-desktop issue #449)
----------------------------------

Desktop apps that spawn ``rapid-mlx serve`` as a subprocess hold a 20-30 GB
model weight in unified memory. When the parent (Rapid-MLX Desktop, ``rapid``
CLI wrapper, ...) is killed with SIGKILL — macOS app hang force-quit, kernel
OOM kill, panic, ``kill -9`` from a watchdog — the kernel re-parents the
sidecar to launchd (PID 1) and the sidecar keeps running forever:

  * listening on the same port the next launch will try to bind, so the next
    launch sees "port in use" until the orphan is reaped by hand,
  * holding the full model in RAM, so two crashes can stack 60+ GB of phantom
    resident memory the operator did not notice they had.

The desktop side has a PortSweep (rapid-desktop PR #170) that reaps detected
orphans on the *next* launch, but if the operator does not relaunch (closes
the lid, walks away), the orphan persists indefinitely.

Mitigation
----------

The supervisor passes its own PID as ``--watchdog-ppid`` (or
``RAPID_MLX_WATCHDOG_PPID`` env). A background thread polls
``os.getppid()`` every ``interval`` seconds. When the live PPID stops
matching the expected one — POSIX guarantees the child is re-parented
the instant its parent dies, so on macOS/Linux the live PPID flips to 1
(launchd / init) — the watchdog sends SIGTERM to itself for a clean
shutdown, then SIGKILL after ``grace`` seconds if the lifespan drain
hangs.

Design notes
------------

* The thread is a daemon — never blocks interpreter exit and never holds
  a reference to user-supplied state.
* The check is intentionally PPID-comparison (not ``kill(ppid, 0)``):
  ``kill(pid, 0)`` can race against PID-recycle (the kernel could
  re-assign the dead parent's PID to an unrelated process before our
  next poll, giving a false negative). ``os.getppid()`` is authoritative
  for "who owns me right now".
* The ``expected_ppid <= 1`` early-out keeps the helper safe for the
  pathological cases (caller passed 0, or the supervisor was launchd
  itself). It also keeps unit tests / direct invocations from foot-gunning
  themselves on a never-fire-condition.
* No-op when the platform lacks ``os.getppid`` (Windows on older Python).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


# Public env var name the rapid-desktop spawn helper (or any future
# supervisor) can stamp instead of passing the CLI flag. Kept in sync
# with ``rapid-desktop/Sources/Rapid/Server/ServerManager.swift``'s
# ``serveEnvironmentAdditions`` writer. The CLI ``--watchdog-ppid``
# flag wins when both are set.
ENV_VAR = "RAPID_MLX_WATCHDOG_PPID"


def resolve_expected_ppid(cli_value: int | None) -> int | None:
    """Resolve the expected parent PID from CLI flag + env var.

    Precedence: ``--watchdog-ppid`` (CLI) > ``RAPID_MLX_WATCHDOG_PPID`` (env).

    Returns ``None`` if neither source supplied a positive integer
    (the watchdog is a no-op in that case — there is no expected parent
    to compare against). A malformed env value is ignored with a
    DEBUG log; the caller is the supervisor, not the operator, so
    malformed input here is a programming bug worth surfacing in DEBUG
    but not loud enough to spam the operator's terminal.
    """
    if cli_value is not None:
        return cli_value if cli_value > 1 else None
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        logger.debug(
            "ignoring malformed %s=%r (not an integer)",
            ENV_VAR,
            raw,
        )
        return None
    return parsed if parsed > 1 else None


def _default_on_orphan(expected_ppid: int, observed_ppid: int) -> None:
    """Default orphan reaction: log + SIGTERM self + SIGKILL after grace.

    The lifespan ``shutdown`` hook in ``server.py`` runs on SIGTERM and
    is responsible for flushing the prefix cache, finalising telemetry,
    and emitting the ``Application shutdown complete.`` banner. We give
    it the same ~5 s wall-clock window the rapid-desktop side hands its
    ``terminateChild`` SIGTERM grace before escalating to SIGKILL —
    keeps the two halves of the kill protocol symmetric (see
    ``ServerManager.swift`` ``terminationGrace``).
    """
    # Single-line marker so log scrapers / dogfood postmortems can grep
    # one consistent string. Print AND log: the logger may not be set up
    # yet for a sidecar that died before ``configure_logging`` ran (or
    # was deliberately set to ERROR), and the user-visible "why did my
    # server vanish" answer must reach stderr unconditionally.
    msg = (
        f"[rapid-mlx] parent watchdog: expected PPID {expected_ppid} but "
        f"observed PPID {observed_ppid} (parent died, re-parented to "
        f"launchd/init); shutting down to release model weights and port"
    )
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:  # pragma: no cover — defensive against broken stderr
        pass
    try:
        logger.warning(msg)
    except Exception:  # pragma: no cover — defensive
        pass

    pid = os.getpid()
    # First, polite SIGTERM — uvicorn / FastAPI lifespan drain.
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:  # pragma: no cover — defensive
        os._exit(0)

    # Give lifespan shutdown ~5 s, then escalate. The desktop side
    # waits the same window before SIGKILL on ``terminateChild``, so
    # this mirrors the contract from both directions.
    time.sleep(5.0)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:  # pragma: no cover
        pass
    os._exit(0)


def _watchdog_loop(
    expected_ppid: int,
    interval: float,
    on_orphan: Callable[[int, int], None],
    stop_event: threading.Event,
) -> None:
    """Body of the watchdog daemon thread.

    Loops until ``stop_event`` is set OR an orphan condition is detected
    (which fires ``on_orphan`` and is normally terminal — but tests can
    inject a non-terminating ``on_orphan`` for assertions).
    """
    # Initial poll runs immediately (not after a sleep) so a supervisor
    # that died between spawn and the first interval is still caught.
    while not stop_event.is_set():
        try:
            observed = os.getppid()
        except OSError:  # pragma: no cover — extremely unusual
            return
        if observed != expected_ppid:
            try:
                on_orphan(expected_ppid, observed)
            except SystemExit:
                raise
            except Exception:  # pragma: no cover — defensive
                logger.exception("parent watchdog on_orphan callback failed")
            # Default ``_default_on_orphan`` does not return; a custom
            # callback might. In either case, stop polling: the orphan
            # decision was made, repeated firings would just spam the log.
            return
        stop_event.wait(interval)


def install_parent_watchdog(
    expected_ppid: int | None,
    *,
    interval: float = 2.0,
    on_orphan: Callable[[int, int], None] | None = None,
) -> threading.Thread | None:
    """Start a daemon thread that terminates this process when its parent dies.

    Parameters
    ----------
    expected_ppid:
        The PID the supervisor expects to be our parent. Pass the result
        of :func:`resolve_expected_ppid` to honour both ``--watchdog-ppid``
        and ``RAPID_MLX_WATCHDOG_PPID``. ``None`` (or any value ``<= 1``)
        is a no-op so the helper is safe to call unconditionally from
        ``serve_command``.
    interval:
        Seconds between polls. Default 2 s — that matches the rapid-desktop
        ``runtimeHealthInterval`` cadence so the two halves of the lifecycle
        watchdog tick at the same frequency. Floor of 0.1 s in case a test
        wants tighter polling.
    on_orphan:
        Override the default "SIGTERM self then SIGKILL" reaction. Tests
        substitute a no-op + flag-flip so they can drive the loop without
        terminating the test runner. Production callers always want the
        default.

    Returns
    -------
    The watchdog thread, or ``None`` if the configuration was a no-op
    (no ``expected_ppid``, or the current PPID already matches PID 1 —
    the daemon-launched case has no parent to watch). Tests can ``join``
    the thread after setting ``stop_event``; production callers can
    ignore the return value.
    """
    if expected_ppid is None or expected_ppid <= 1:
        return None

    # If the current PPID is ALREADY != expected_ppid at install time,
    # we missed the death (the supervisor died between spawn and now).
    # Fire the callback synchronously so the operator sees the same
    # marker as the in-flight case, then return None (no thread).
    try:
        live_ppid = os.getppid()
    except OSError:  # pragma: no cover
        return None
    callback = on_orphan or _default_on_orphan
    if live_ppid != expected_ppid:
        callback(expected_ppid, live_ppid)
        return None

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_watchdog_loop,
        name="rapid-mlx-parent-watchdog",
        args=(expected_ppid, max(0.1, float(interval)), callback, stop_event),
        daemon=True,
    )
    # Expose the stop event on the thread so tests (and any future
    # graceful-shutdown caller that wants to disarm the watchdog before
    # an intentional execve) can stop the loop without waiting for the
    # next poll interval.
    thread._rapid_mlx_stop_event = stop_event  # type: ignore[attr-defined]
    thread.start()
    return thread
