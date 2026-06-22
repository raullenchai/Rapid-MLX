# SPDX-License-Identifier: Apache-2.0
"""Process-death observability for rapid-mlx servers.

Installs a signal handler chain + ``faulthandler`` so the operator can tell
the difference between

  * a SIGKILL (no handler can run; nothing in the log; the *absence* of
    these stack dumps is itself a signal that the death was un-catchable),
  * a SIGTERM/SIGHUP/SIGABRT (handler logs the signal name + every alive
    thread's stack BEFORE the existing shutdown machinery runs), and
  * a C-level segfault inside MLX / Metal (``faulthandler.enable()`` writes
    a Python traceback to stderr from the signal handler before the
    interpreter dies).

This was lifted out of ``server.py`` because:

  1. It's a small, stdlib-only piece of code that's easier to unit-test in
     isolation than from inside the FastAPI lifespan.
  2. The C-04 dogfood recon (``/tmp/dogfood-085/c04-recon.md`` §1 + §3.R1)
     showed the canonical "process disappears between two consecutive
     stdout writes" shape — operators currently have zero observability of
     their own server's death. R1 ("Install a top-level signal handler that
     logs receipt and survives stdout buffering") is the cited fix.

The handlers are deliberately ADDITIVE — they call ``faulthandler`` first,
then chain into whatever uvicorn (or any prior caller) registered. They
must NOT change graceful-shutdown semantics: a SIGTERM still has to land
on uvicorn's normal handler so the FastAPI lifespan shutdown drains
in-flight requests, persists the prefix cache, and emits the
``Application shutdown complete.`` banner the dogfood logs were missing.

NOTE on threading: ``signal.signal`` MUST be called from the main thread
(POSIX restriction enforced by CPython). The install helper raises a
clear ``RuntimeError`` if invoked off the main thread instead of failing
silently with a confusing ``ValueError: signal only works in main thread``.
"""

from __future__ import annotations

import faulthandler
import logging
import signal
import sys
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


# Signals we want to observe. Each is mapped to its symbolic name so the
# log line is self-explanatory even on Linux/macOS variants where the
# integer values differ.
#
# Deliberately NOT included:
#   * SIGINT — Ctrl-C is operator-initiated, no need to spew per-thread
#     stacks. uvicorn's existing SIGINT handler is fine.
#   * SIGKILL / SIGSTOP — cannot be caught (kernel restriction).
#   * SIGSEGV / SIGBUS / SIGILL / SIGFPE — handled via
#     ``faulthandler.enable()`` (which writes a Python traceback BEFORE
#     the interpreter dies; a plain ``signal.signal`` for SIGSEGV cannot
#     safely run Python because the C-level state may already be
#     corrupt).
#   * SIGABRT — codex r1 BLOCKING #2: ``faulthandler.enable()`` already
#     installs an async-signal-safe C-level handler for SIGABRT, which
#     writes the Python traceback from a signal-safe context. Installing
#     our Python-level ``_on_signal`` on top of it would (a) overwrite
#     the faulthandler hook with a non-async-signal-safe Python handler
#     that calls ``logging`` (re-entrant on stdio locks) and (b)
#     downgrade the crash-path observability we just added. Let
#     faulthandler keep SIGABRT.
_OBSERVED_SIGNALS: tuple[int, ...] = tuple(
    sig
    for sig in (
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGHUP", None),
    )
    if sig is not None
)


# Idempotency latch. Lifespan startup can fire more than once in
# in-process test harnesses (the FastAPI lifespan is driven from
# ``TestClient`` setup as well as ``uvicorn.run``), and re-registering
# the handler chain would stack our handler on top of ITSELF — every
# subsequent SIGTERM would dump traceback N times. Latch in module-scope
# so the install is exactly-once per process.
_installed = False
_install_lock = threading.Lock()

# Saved prior handlers so we can chain to them. Keyed by signal number.
# Visible to tests via ``_get_installed_handlers``.
_prior_handlers: dict[int, signal.Handlers | Callable[..., object] | int | None] = {}


def _signal_name(signum: int) -> str:
    """Return a stable, human-readable name for a signal number.

    Prefer ``signal.Signals(signum).name`` (yields ``"SIGTERM"`` etc.) and
    fall back to the raw integer if the value isn't in the enum (which
    can happen for platform-specific custom signals).
    """
    try:
        return signal.Signals(signum).name
    except (ValueError, AttributeError):
        return f"signal {signum}"


def _on_signal(signum: int, frame) -> None:  # noqa: ARG001 — frame unused
    """Chained signal handler: log receipt + dump per-thread stacks, then
    delegate to whatever was registered before us.

    Runs on the main thread (Python signal-handler invariant). Must be
    *quick* and *async-signal-safe-ish* — we deliberately do only:

      * one ``logger.warning`` call (single ``write``),
      * ``faulthandler.dump_traceback(all_threads=True)`` to stderr
        (which the faulthandler module guarantees is async-signal-safe),
      * chain into the prior handler.

    We do NOT flush logging handlers explicitly (the warning goes through
    the standard handler path; explicit ``handler.flush()`` from a signal
    handler is unsafe because it can re-enter the C stdio lock).
    """
    name = _signal_name(signum)
    # Single-line preamble so log scrapers can grep one consistent
    # marker. The stack dump itself goes to stderr via faulthandler
    # (not through the logging tree), so this WARNING line is the
    # "table of contents" entry that points readers at the stderr dump.
    try:
        logger.warning(
            "rapid-mlx received signal %s; thread stacks follow (faulthandler)",
            name,
        )
    except Exception:  # pragma: no cover — defensive
        # A logging failure must not block the chain to uvicorn's handler.
        pass

    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:  # pragma: no cover — defensive
        pass

    # Chain to whatever was registered before us so the graceful
    # shutdown path still runs (uvicorn installs a SIGTERM handler that
    # initiates ``Server.shutdown`` — losing that would convert every
    # SIGTERM into a SIGKILL-equivalent for the lifespan hook).
    #
    # Codex r1 BLOCKING #1: we install ONLY when ``prior`` is callable
    # (see ``install_signal_observability``), so this branch is the only
    # one that can fire in practice. The ``SIG_DFL`` / ``SIG_IGN``
    # branches are kept as defensive fall-throughs but DO NOT re-raise
    # the signal — re-installing ``SIG_DFL`` and ``raise_signal``-ing
    # would converted an observable signal into an immediate process
    # kill that races any FastAPI/uvicorn shutdown machinery the
    # operator might bring up out-of-band (e.g. embedded ASGI servers
    # that wire their own shutdown later in startup). Keeping the
    # original disposition's terminate behaviour requires no action
    # from us: ``signal.signal`` returns ``SIG_DFL``/``SIG_IGN`` as
    # the prior only when the kernel-side disposition was unchanged,
    # and the *previous* time the signal arrives the OS will redeliver
    # under the original (now-restored-by-uvicorn-or-FastAPI) handler.
    prior = _prior_handlers.get(signum)
    if callable(prior):
        try:
            prior(signum, frame)
        except Exception:  # pragma: no cover — defensive
            logger.debug(
                "prior signal handler for %s raised during chain", name, exc_info=True
            )
    # SIG_DFL / SIG_IGN: do nothing further; we've already logged + dumped
    # stacks. The operator now has the receipt evidence the C-04 recon
    # was missing, and downstream shutdown drivers (uvicorn,
    # FastAPI lifespan) own the actual termination semantics.


def install_signal_observability(
    *,
    observed_signals: tuple[int, ...] | None = None,
) -> bool:
    """Install ``faulthandler`` + a chained signal handler for SIGTERM /
    SIGHUP / SIGABRT.

    Returns ``True`` if installed (or was already installed), ``False``
    if installation was skipped because we're off the main thread.

    Parameters
    ----------
    observed_signals
        Override the default ``(SIGTERM, SIGHUP, SIGABRT)`` set. Tests
        pass a narrower tuple to avoid clobbering pytest's own handlers.

    The function is **idempotent** — repeated calls after the first
    succeed and become no-ops. This matters because the FastAPI lifespan
    can fire multiple times in test harnesses and we don't want to
    stack our handler on top of itself (re-entry would emit N copies
    of the stack dump per signal).

    On non-main threads (e.g. when ``uvicorn.run`` is driven from a
    worker thread in some embedded contexts), ``signal.signal`` raises
    ``ValueError``. We catch that and return ``False`` rather than
    crashing the server boot — the operator simply doesn't get the
    enhanced observability, but the server still starts.
    """
    global _installed

    with _install_lock:
        if _installed:
            return True

        # CPython enforces "signal only works in main thread of the main
        # interpreter". Check explicitly so the failure mode is a clear
        # log line rather than a buried ValueError partway through.
        if threading.current_thread() is not threading.main_thread():
            logger.debug(
                "signal observability skipped: not on main thread"
                " (current=%s); faulthandler/signal install requires"
                " the main thread on POSIX",
                threading.current_thread().name,
            )
            return False

        # ``faulthandler.enable()`` installs SIGSEGV/SIGFPE/SIGABRT/SIGBUS/
        # SIGILL handlers at the C level. Calling it twice is safe — the
        # second call is a no-op once enabled. We send the dump to stderr
        # so it lands in the same stream operators tail with the server
        # log; redirecting stderr to the log file (the typical
        # ``rapid-mlx serve ... 2>&1 | tee server.log`` shape) captures
        # it for post-mortem.
        try:
            faulthandler.enable(file=sys.stderr, all_threads=True)
        except (ValueError, RuntimeError) as exc:  # pragma: no cover
            # ValueError raised if stderr was redirected to a closed fd.
            # Not fatal — proceed with signal install.
            logger.debug("faulthandler.enable failed: %r", exc)

        signals_to_install = (
            observed_signals if observed_signals is not None else _OBSERVED_SIGNALS
        )

        for sig in signals_to_install:
            # Codex r1 BLOCKING #1 follow-on: peek at the current
            # handler BEFORE installing ours, so we can skip the
            # signals where no graceful-shutdown driver is already in
            # place. ``signal.getsignal`` returns the same Python-level
            # callable / ``SIG_DFL`` / ``SIG_IGN`` value that
            # ``signal.signal`` would return as the prior. If the
            # current handler isn't a Python callable, replacing it
            # with our observability hook would silently swallow the
            # signal — operator gets a log line but the process never
            # terminates (because our chain explicitly does NOT
            # re-raise on the ``SIG_DFL`` branch — see ``_on_signal``).
            # Preserve the original disposition by NOT installing in
            # that case; the operator loses the receipt evidence but
            # keeps the kernel-level terminate-or-ignore semantics
            # they expected.
            try:
                current = signal.getsignal(sig)
            except (OSError, ValueError) as exc:
                logger.debug(
                    "could not query current handler for %s: %r",
                    _signal_name(sig),
                    exc,
                )
                continue
            if not callable(current):
                logger.debug(
                    "skipping rapid-mlx handler for %s: prior=%r is not"
                    " a Python callable; preserving kernel-level"
                    " disposition to avoid changing shutdown semantics",
                    _signal_name(sig),
                    current,
                )
                continue
            try:
                prior = signal.signal(sig, _on_signal)
            except (OSError, ValueError) as exc:
                # ValueError for invalid signals on platform; OSError
                # for permission issues. Skip and continue with the rest.
                logger.debug(
                    "could not install rapid-mlx handler for %s: %r",
                    _signal_name(sig),
                    exc,
                )
                continue
            _prior_handlers[sig] = prior
            logger.debug(
                "rapid-mlx signal handler installed for %s (prior=%r)",
                _signal_name(sig),
                prior,
            )

        _installed = True
        return True


def _reset_for_tests() -> None:
    """Internal test helper: restore prior handlers and clear the latch.

    Production code MUST NOT call this. The signal-observability test
    module uses it to install/uninstall the handler set within a single
    pytest process without leaking handlers to the next test.
    """
    global _installed
    with _install_lock:
        for sig, prior in list(_prior_handlers.items()):
            try:
                signal.signal(sig, prior if prior is not None else signal.SIG_DFL)
            except (OSError, ValueError):
                pass
        _prior_handlers.clear()
        _installed = False


def _get_installed_handlers() -> dict[int, object]:
    """Internal test helper: snapshot of the saved prior-handler map."""
    return dict(_prior_handlers)
