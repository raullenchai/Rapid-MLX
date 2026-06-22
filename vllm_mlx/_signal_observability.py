# SPDX-License-Identifier: Apache-2.0
"""Process-death observability for rapid-mlx servers.

Installs a signal handler chain + ``faulthandler`` so the operator can tell
the difference between

  * a SIGKILL (no handler can run; nothing in the log; the *absence* of
    these stack dumps is itself a signal that the death was un-catchable),
  * a SIGTERM/SIGHUP (Python-level ``signal.signal`` chain logs the
    signal name + every alive thread's stack BEFORE the existing
    shutdown machinery runs), and
  * a C-level segfault or abort inside MLX / Metal
    (``faulthandler.enable()`` writes a Python traceback to stderr for
    SIGSEGV / SIGBUS / SIGILL / SIGFPE / SIGABRT directly from the
    C-level signal handler before the interpreter dies — see the note
    in ``_OBSERVED_SIGNALS`` for why we don't double-install on
    SIGABRT).

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

    # Chain to whatever was registered before us so the original
    # disposition is preserved end-to-end:
    #   * callable prior (uvicorn's ``handle_exit`` etc.) → call it so
    #     graceful shutdown still runs;
    #   * SIG_DFL → restore default + ``signal.raise_signal`` so the
    #     kernel-level terminate-by-default fires (this is correct
    #     because if the prior was SIG_DFL, no shutdown driver was
    #     listening — termination IS the original behaviour, and we've
    #     already added the observability via the WARNING + stack dump
    #     above);
    #   * SIG_IGN → return; ignore-by-default IS the original behaviour.
    #
    # Codex r2 BLOCKING #1: an earlier round of this PR skipped the
    # install entirely when the prior was non-callable, which meant the
    # operator's SIGHUP (default disposition is ``SIG_DFL`` because
    # uvicorn does not capture SIGHUP) was never observed in production
    # — exactly the silent-death shape C-04 was trying to fix. Always
    # install, and make the SIG_DFL branch preserve termination via
    # ``raise_signal`` after restoring the default handler.
    prior = _prior_handlers.get(signum)
    if callable(prior):
        try:
            prior(signum, frame)
        except Exception:  # pragma: no cover — defensive
            logger.debug(
                "prior signal handler for %s raised during chain", name, exc_info=True
            )
    elif prior == signal.SIG_DFL:
        # Restore the default disposition and re-deliver the signal so
        # the kernel-level terminate behaviour fires. ``signal.signal``
        # is async-signal-safe in CPython's signal module; the
        # ``raise_signal`` call lands on the now-restored SIG_DFL
        # handler and terminates the process the same way it would
        # have without our hook — just AFTER we've logged + dumped.
        try:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        except Exception:  # pragma: no cover — defensive
            pass
    # SIG_IGN means "ignore" — do nothing (the original disposition was
    # ignore, and we've already logged the receipt).


def install_signal_observability(
    *,
    observed_signals: tuple[int, ...] | None = None,
) -> bool:
    """Install ``faulthandler`` + a chained signal handler for SIGTERM
    and SIGHUP. SIGABRT is intentionally NOT chained — see
    ``_OBSERVED_SIGNALS``; faulthandler's C-level handler owns that
    path because it's async-signal-safe and our Python-level
    ``_on_signal`` (which calls ``logging``) is not.

    Returns ``True`` if at least one handler was installed (or all
    handlers were already installed), ``False`` if installation was
    skipped because we're off the main thread or because every
    ``signal.signal`` call raised on this platform. Subsequent calls
    after a returning-``False`` attempt are NOT latched off — the
    install retries fresh.

    Parameters
    ----------
    observed_signals
        Override the default ``(SIGTERM, SIGHUP)`` set (see
        ``_OBSERVED_SIGNALS`` for why SIGABRT is intentionally not in
        this list). Tests pass a narrower tuple (e.g.
        ``(SIGUSR1,)``) to avoid clobbering pytest's own handlers.

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

        # Codex r2 BLOCKING #2: latch ONLY after at least one handler
        # was successfully installed. If every install raised (the
        # all-ValueError path on a platform that rejects every observed
        # signal), an early call would otherwise permanently disable
        # later installation attempts in the same process — e.g. a
        # uvicorn-managed lifespan that fires the install after a
        # later config-reload would also be skipped because
        # ``_installed`` was True from the failed first attempt.
        installed_any = False
        for sig in signals_to_install:
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
            installed_any = True
            logger.debug(
                "rapid-mlx signal handler installed for %s (prior=%r)",
                _signal_name(sig),
                prior,
            )

        # Only latch when something actually got installed. A no-op
        # install (every signal rejected on this platform) returns
        # False so a future call from a different entry point gets to
        # try again. faulthandler.enable() above is idempotent and
        # doesn't need re-running.
        if not installed_any:
            return False

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
