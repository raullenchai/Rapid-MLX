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
(POSIX restriction enforced by CPython). The install helper detects
the off-main-thread case and returns ``False`` (with a DEBUG log line)
rather than letting the underlying ``ValueError: signal only works in
main thread`` propagate. The server boot proceeds — the operator
simply doesn't get the enhanced observability for that lifespan. Codex
r7 NIT #4: the prior docstring incorrectly said this branch raised
``RuntimeError``; the actual implementation has always been
non-raising.
"""

from __future__ import annotations

import atexit
import faulthandler
import json
import logging
import os
import re
import signal
import stat
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from rapid_mlx._process_identity import (
    is_same_process,
    marker_identity,
    process_identity,
)

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


# Module-level lock around the install path. Lifespan startup can fire
# more than once in in-process test harnesses (the FastAPI lifespan is
# driven from ``TestClient`` setup as well as ``uvicorn.run``); without
# the lock two parallel installs on the same signal could race and
# leave ``_prior_handlers`` storing the wrong prior. Per-signal
# idempotency is enforced inline by the ``sig in _prior_handlers``
# check in ``install_signal_observability`` (codex r7 NIT #3).
_install_lock = threading.Lock()

# Saved prior handlers so we can chain to them. Keyed by signal number.
# Visible to tests via ``_get_installed_handlers``.
_prior_handlers: dict[int, signal.Handlers | Callable[..., object] | int | None] = {}
_crash_fd: int | None = None
_crash_fd_identity: tuple[int, int] | None = None
_crash_file_identity: tuple[int, int] | None = None
_crash_path: Path | None = None
_crash_pipe = None
_crash_tee: subprocess.Popen[bytes] | None = None
# If an external close lets the pipe fd number be reused, closing the stale
# Python pipe object would close the unrelated replacement fd. Retain such
# objects until interpreter teardown instead; the process is already exiting
# then, so their eventual finalizers cannot corrupt a live application fd.
_abandoned_crash_pipes: list[object] = []
_crash_cleanup_registered = False
_faulthandler_was_enabled = False
_tee_fallback_warned = False
_MAX_MARKER_BYTES = 4096
_ACKNOWLEDGED_CRASH_FILE = re.compile(r"\.reported(?:-\d+)?\.txt\Z")
_CRASH_TEE_SCRIPT = """\
import os
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_IGN)
crash_fd = int(sys.argv[1])
durable_sink_ok = True
while chunk := os.read(0, 65536):
    if durable_sink_ok:
        try:
            pending = chunk
            while pending:
                pending = pending[os.write(crash_fd, pending):]
            os.fsync(crash_fd)
        except OSError:
            durable_sink_ok = False
    try:
        pending = chunk
        while pending:
            pending = pending[os.write(2, pending):]
    except (AttributeError, OSError):
        pass
"""


def _move_fd_above_stdio(fd: int) -> int:
    if fd > 2:
        return fd
    import fcntl

    replacement_fd = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)
    os.close(fd)
    return replacement_fd


def _crash_logs_dir() -> Path:
    from rapid_mlx.telemetry.state import _default_telemetry_dir

    return _default_telemetry_dir() / "logs"


def _crash_files(log_dir: Path) -> list[Path]:
    candidates: list[tuple[int, str, Path]] = []
    try:
        paths = log_dir.glob("crash-*.txt")
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            if path.is_file():
                candidates.append((stat.st_mtime_ns, path.name, path))
    except OSError:
        return []
    candidates.sort(reverse=True)
    return [path for _, _, path in candidates]


def _is_acknowledged(name: str) -> bool:
    return _ACKNOWLEDGED_CRASH_FILE.search(name) is not None


def _has_acknowledged_inode(path: Path) -> bool:
    """Return whether an acknowledged hard link already names ``path``."""

    try:
        source = path.lstat()
        candidates = path.parent.glob(f"{path.stem}.reported*{path.suffix}")
        for candidate in candidates:
            try:
                acknowledged = candidate.lstat()
            except OSError:
                continue
            if (acknowledged.st_dev, acknowledged.st_ino) == (
                source.st_dev,
                source.st_ino,
            ):
                return True
    except OSError:
        return False
    return False


def _crash_file_pid(path: Path) -> int | None:
    try:
        stem = path.stem
        if stem.endswith(".reported"):
            stem = stem.removesuffix(".reported")
        pid = int(stem.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None
    return pid if pid > 0 else None


def _marker_for_pid(log_dir: Path, pid: int) -> dict[str, object] | None:
    marker_path = log_dir.parent / "state" / f"serve-inflight-{pid}.json"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(marker_path, flags)
        try:
            marker_stat = os.fstat(fd)
            if (
                not stat.S_ISREG(marker_stat.st_mode)
                or marker_stat.st_size > _MAX_MARKER_BYTES
            ):
                return None
            chunks: list[bytes] = []
            remaining = _MAX_MARKER_BYTES + 1
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > _MAX_MARKER_BYTES:
                return None
        finally:
            os.close(fd)
        marker = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return marker if isinstance(marker, dict) and marker_identity(marker) else None


def _crash_file_is_live(path: Path, log_dir: Path) -> bool:
    pid = _crash_file_pid(path)
    if pid is None:
        return False
    marker = _marker_for_pid(log_dir, pid)
    if marker is not None:
        return marker.get("pid") == pid and is_same_process(marker)
    # Crash-sink installation precedes the startup marker. Conservatively
    # retain an unmarked file while its owner PID is alive so concurrent
    # startups cannot rotate away one another's open diagnostics.
    if pid == os.getpid():
        return True
    return process_identity(pid) is not None


def _prepare_crash_logs_dir(path: Path) -> bool:
    """Create/open the private crash directory without following a symlink."""
    try:
        try:
            log_stat = os.lstat(path)
        except FileNotFoundError:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            log_stat = os.lstat(path)
        if stat.S_ISLNK(log_stat.st_mode) or not stat.S_ISDIR(log_stat.st_mode):
            logger.warning("rapid-mlx crash log directory is unavailable: %s", path)
            return False
        if os.name == "nt":
            os.chmod(path, 0o700)
            opened_stat = os.lstat(path)
            if stat.S_ISLNK(opened_stat.st_mode) or (
                opened_stat.st_dev,
                opened_stat.st_ino,
            ) != (log_stat.st_dev, log_stat.st_ino):
                return False
        else:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            try:
                opened_stat = os.fstat(fd)
                if (opened_stat.st_dev, opened_stat.st_ino) != (
                    log_stat.st_dev,
                    log_stat.st_ino,
                ):
                    return False
                os.fchmod(fd, 0o700)
            finally:
                os.close(fd)
    except OSError as exc:
        logger.warning("rapid-mlx crash log directory is unavailable: %s", exc)
        return False
    return True


def _report_previous_crash(log_dir: Path) -> None:
    for path in _crash_files(log_dir):
        if _is_acknowledged(path.name):
            continue
        if _has_acknowledged_inode(path):
            continue
        if _crash_file_is_live(path, log_dir):
            continue
        try:
            if path.stat().st_size <= 0:
                continue
            sys.stderr.write(
                f"Previous run crashed; details in {path} "
                "(and macOS DiagnosticReports under "
                "~/Library/Logs/DiagnosticReports).\n"
            )
            sys.stderr.flush()
            suffix = 0
            while True:
                reported = path.with_name(
                    f"{path.stem}.reported"
                    f"{'-' + str(suffix) if suffix else ''}{path.suffix}"
                )
                try:
                    os.link(path, reported)
                except FileExistsError:
                    suffix += 1
                    continue
                path.unlink()
                break
        except (AttributeError, OSError, ValueError):
            pass
        return


def _rotate_crash_files(log_dir: Path) -> None:
    retained_inactive = 0
    for path in _crash_files(log_dir):
        if not _is_acknowledged(path.name) and _crash_file_is_live(path, log_dir):
            continue
        retained_inactive += 1
        if retained_inactive <= 5:
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.debug("could not rotate old crash file %s: %r", path, exc)


def _enable_faulthandler(target_fd) -> None:
    """The sole process-wide faulthandler registration point."""
    faulthandler.enable(file=target_fd, all_threads=True)


def _enable_stderr_faulthandler() -> None:
    """Retain fatal traceback coverage when durable storage is unavailable."""
    if sys.stderr is None or getattr(sys.stderr, "closed", False):
        return
    try:
        _enable_faulthandler(sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("could not enable fallback faulthandler on stderr: %s", exc)


def _warn_tee_fallback(exc: BaseException) -> None:
    global _tee_fallback_warned
    if _tee_fallback_warned:
        return
    _tee_fallback_warned = True
    logger.warning(
        "could not mirror fatal tracebacks to stderr; using crash file only: %s",
        exc,
    )


def _stop_crash_tee(
    process: subprocess.Popen[bytes] | None,
    pipe,
    *,
    close_pipe: bool = True,
) -> None:
    if pipe is not None:
        if close_pipe:
            try:
                pipe.close()
            except OSError:
                pass
        else:
            _abandoned_crash_pipes.append(pipe)
    if process is None:
        return
    try:
        process.wait(timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                # A killed child can still take time to become waitable. Do
                # not block shutdown indefinitely, but keep a live reference
                # and reap it asynchronously instead of abandoning a zombie.
                threading.Thread(
                    target=_reap_crash_tee,
                    args=(process,),
                    daemon=True,
                    name="rapid-mlx-crash-tee-reaper",
                ).start()


def _reap_crash_tee(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait()
    except OSError:
        pass


def _cleanup_crash_file() -> None:
    """Remove an empty clean-run file and release faulthandler's descriptor."""
    global _crash_fd, _crash_fd_identity, _crash_file_identity
    global _crash_path, _crash_pipe, _crash_tee
    fd, path = _crash_fd, _crash_path
    file_identity = _crash_file_identity
    pipe, process = _crash_pipe, _crash_tee
    if fd is None or path is None:
        return
    _crash_fd = None
    _crash_fd_identity = None
    _crash_file_identity = None
    _crash_path = None
    _crash_pipe = None
    _crash_tee = None
    try:
        faulthandler.disable()
    except (OSError, RuntimeError, ValueError):
        pass
    if pipe is not None:
        _stop_crash_tee(process, pipe)
    else:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        path_stat = path.lstat()
        if (
            stat.S_ISREG(path_stat.st_mode)
            and (path_stat.st_dev, path_stat.st_ino) == file_identity
            and path_stat.st_size == 0
        ):
            path.unlink(missing_ok=True)
    except (AttributeError, OSError):
        pass
    if (
        _faulthandler_was_enabled
        and sys.stderr is not None
        and not getattr(sys.stderr, "closed", False)
    ):
        try:
            _enable_faulthandler(sys.stderr)
        except (OSError, RuntimeError, ValueError):
            pass


def _install_crash_file() -> None:
    """Mirror fatal tracebacks to stderr and a private rotating file."""
    global _crash_cleanup_registered, _crash_fd, _crash_fd_identity
    global _crash_file_identity, _crash_path
    global _crash_pipe
    global _crash_tee, _faulthandler_was_enabled
    if _crash_fd is not None:
        _ensure_crash_sink_locked()
        return
    path: Path | None = None
    fd: int | None = None
    pipe = None
    process: subprocess.Popen[bytes] | None = None
    try:
        log_dir = _crash_logs_dir()
        if not _prepare_crash_logs_dir(log_dir):
            _enable_stderr_faulthandler()
            return
        _report_previous_crash(log_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = log_dir / f"crash-{timestamp}-{os.getpid()}.txt"
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o600,
        )
        os.chmod(path, 0o600)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError("crash sink is not a regular file")
        if os.name == "nt":
            _warn_tee_fallback(OSError("crash mirror helper is unavailable on Windows"))
        else:
            try:
                fd = _move_fd_above_stdio(fd)
                process = subprocess.Popen(
                    [sys.executable, "-c", _CRASH_TEE_SCRIPT, str(fd)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=None,
                    close_fds=True,
                    pass_fds=(fd,),
                    start_new_session=True,
                    shell=False,
                )
                if process.stdin is None:
                    raise OSError("tee helper has no stdin pipe")
                pipe = process.stdin
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                _warn_tee_fallback(exc)
                _stop_crash_tee(process, pipe)
                process = None
                pipe = None
        target_fd = pipe.fileno() if pipe is not None else fd
        target_stat = os.fstat(target_fd)
        _faulthandler_was_enabled = faulthandler.is_enabled()
        _enable_faulthandler(target_fd)
        _crash_fd = target_fd
        _crash_fd_identity = (target_stat.st_dev, target_stat.st_ino)
        _crash_file_identity = (file_stat.st_dev, file_stat.st_ino)
        _crash_path = path
        _crash_pipe = pipe
        _crash_tee = process
        if pipe is not None:
            os.close(fd)
        fd = None
        pipe = None
        process = None
        _rotate_crash_files(log_dir)
        if not _crash_cleanup_registered:
            atexit.register(_cleanup_crash_file)
            _crash_cleanup_registered = True
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("could not create durable rapid-mlx crash file: %s", exc)
        _stop_crash_tee(process, pipe)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        _enable_stderr_faulthandler()


def _ensure_crash_sink_locked() -> bool:
    """Re-arm the installed sink while the caller holds ``_install_lock``."""
    global _crash_fd, _crash_fd_identity, _crash_file_identity
    global _crash_pipe, _crash_tee
    if _crash_fd is None:
        return False
    failure = OSError("crash descriptor no longer identifies the installed sink")
    try:
        current_stat = os.fstat(_crash_fd)
        descriptor_intact = (
            current_stat.st_dev,
            current_stat.st_ino,
        ) == _crash_fd_identity
    except OSError as exc:
        descriptor_intact = False
        failure = exc
    tee_alive = _crash_tee is None or _crash_tee.poll() is None
    sink_intact = descriptor_intact and tee_alive
    if descriptor_intact and not tee_alive:
        failure = OSError("crash mirror helper exited unexpectedly")
    if not sink_intact:
        # An external close can leave the Python pipe object holding the
        # same integer that os.open would immediately reuse. Stop the tee
        # before reopening so its close cannot close the replacement fd.
        pipe, process = _crash_pipe, _crash_tee
        _crash_fd = None
        _crash_fd_identity = None
        _crash_pipe = None
        _crash_tee = None
        if pipe is not None:
            # The stored fd identity did not match. Its integer may now belong
            # to another resource, so never call close() through the stale pipe
            # object. Retaining it prevents its finalizer doing the same later.
            _stop_crash_tee(process, pipe, close_pipe=descriptor_intact)
        if _crash_path is None:
            logger.warning("durable rapid-mlx crash sink is closed: %s", failure)
            return False
        new_fd: int | None = None
        try:
            flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            new_fd = os.open(_crash_path, flags)
            new_stat = os.fstat(new_fd)
            if (
                not stat.S_ISREG(new_stat.st_mode)
                or (
                    new_stat.st_dev,
                    new_stat.st_ino,
                )
                != _crash_file_identity
            ):
                raise OSError("crash file path no longer identifies the installed sink")
            _enable_faulthandler(new_fd)
            _crash_fd = new_fd
            _crash_fd_identity = (new_stat.st_dev, new_stat.st_ino)
            new_fd = None
        except (OSError, RuntimeError, ValueError) as reopen_exc:
            logger.warning(
                "could not recover closed durable rapid-mlx crash file: %s",
                reopen_exc,
            )
        finally:
            if new_fd is not None:
                try:
                    os.close(new_fd)
                except OSError:
                    pass
        return _crash_fd is not None
    try:
        _enable_faulthandler(_crash_fd)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("could not re-arm durable rapid-mlx crash file: %s", exc)
        return False
    return True


def ensure_crash_sink() -> bool:
    """Re-arm faulthandler on Rapid-MLX's installed crash sink."""
    with _install_lock:
        return _ensure_crash_sink_locked()


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
    #   * SIG_DFL → restore default + ``raise_signal``
    #     so the kernel-level terminate-by-default fires (we've already
    #     added the observability via the WARNING + stack dump above);
    #   * SIG_IGN → return; ignore-by-default IS the original behaviour.
    #
    # Codex r2 BLOCKING #1: an earlier round of this PR skipped the
    # install entirely when the prior was non-callable, which meant the
    # operator's SIGHUP (default disposition is ``SIG_DFL`` because
    # uvicorn does not capture SIGHUP) was never observed in production
    # — exactly the silent-death shape C-04 was trying to fix. Always
    # install, and make the SIG_DFL branch preserve termination via
    # ``raise_signal`` after restoring the default handler. Preserving
    # SIGHUP's default termination is important: terminal/supervisor
    # hangup must not orphan a model server that keeps its port and RAM.
    prior = _prior_handlers.get(signum)
    if callable(prior):
        # Codex r8 BLOCKING #2: do NOT swallow exceptions from the prior
        # handler — uvicorn raises ``KeyboardInterrupt`` from its own
        # SIGTERM/SIGINT handlers to drive shutdown, and other prior
        # handlers may raise ``SystemExit`` for the same reason. Catching
        # ``Exception`` blanket-style here would prevent process
        # termination, re-introducing C-04 silent-death shape. Log + re-
        # raise so observability and shutdown semantics both win.
        try:
            prior(signum, frame)
        except Exception:  # pragma: no cover — defensive logging only
            logger.debug(
                "prior signal handler for %s raised during chain", name, exc_info=True
            )
            raise
    elif prior == signal.SIG_DFL:
        # Restore the default disposition and re-deliver the signal so
        # the kernel-level terminate behaviour fires. ``signal.signal``
        # is async-signal-safe in CPython's signal module; the
        # ``raise_signal`` call lands on the now-restored SIG_DFL
        # handler and terminates the process the same way it would
        # have without our hook — just AFTER we've logged + dumped.
        #
        # Codex r8 BLOCKING #1: if EITHER ``signal.signal`` or
        # ``signal.raise_signal`` raises (extremely rare — would
        # require a kernel-level disagreement about the signum, or
        # the signal module being torn down mid-shutdown), the
        # silent-swallow path used in earlier revisions would let
        # the process keep running after a SIGTERM whose default
        # disposition is "terminate". That re-introduces the exact
        # silent-death shape C-04 was trying to make observable —
        # except now with the OPPOSITE problem: the operator sees
        # the WARNING + stack dump and assumes the process died,
        # but it didn't. Fall back to ``os._exit(128 + signum)``
        # (POSIX convention: exit code = 128 + signal number for
        # signal-terminated processes) so the termination semantic
        # is preserved end-to-end even on the failure path. We use
        # ``os._exit`` rather than ``sys.exit`` because the latter
        # raises ``SystemExit`` which can be caught by surrounding
        # code (and we're already in a signal handler — no atexit
        # / finally should fire).
        terminate_failed = False
        try:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        except Exception:  # pragma: no cover — defensive
            terminate_failed = True
            logger.error(
                "could not chain SIGTERM-class signal %s to SIG_DFL"
                " for termination; forcing os._exit(128+%d)",
                name,
                signum,
                exc_info=True,
            )
        if terminate_failed:
            import os

            os._exit(128 + signum)
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

    Return value semantics (codex r6 BLOCKING #1 clarification):
    the return value tracks **the Python-level signal-chain
    install only**. ``faulthandler.enable()`` is the crash-path
    observability layer and is idempotent + side-effect-only, so it
    runs unconditionally regardless of the per-signal install
    outcome — there is no observable difference between "fault-
    handler was enabled by us vs by an earlier call". The bool is
    True if at least one of the requested signals got a handler
    (or all already had one from a prior call), False if none of
    the requested signals could be installed (off main thread,
    every ``signal.signal`` call raised, OR ``observed_signals=()``
    explicitly requested a no-op chain install). Subsequent calls
    after a returning-``False`` attempt are NOT latched off — the
    install retries fresh on the next call.

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
    with _install_lock:
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

        # The fatal-signal handler writes to a real 0600 descriptor that
        # survives Desktop's bounded in-memory stderr buffer. The explicit
        # SIGTERM/SIGHUP chain above continues to dump to stderr as before.
        _install_crash_file()

        signals_to_install = (
            observed_signals if observed_signals is not None else _OBSERVED_SIGNALS
        )

        # Codex r7 NIT #3: track installed signals per-signum instead of
        # a single global latch. Otherwise an early test/custom install
        # for a narrow tuple (e.g. ``(SIGUSR1,)``) latches the function
        # off, and a later production install for the default
        # ``(SIGTERM, SIGHUP)`` returns True without actually
        # registering those handlers. Now: install any requested
        # signal that isn't already in ``_prior_handlers``, and return
        # True iff at least one signal in the requested set is
        # installed at the end (either freshly here, or because a
        # prior call already had it).
        installed_any = False
        for sig in signals_to_install:
            if sig in _prior_handlers:
                # Already installed by a previous call. Counts toward
                # the "True if at least one" bool but we don't
                # re-register (would stack the handler).
                installed_any = True
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
            installed_any = True
            logger.debug(
                "rapid-mlx signal handler installed for %s (prior=%r)",
                _signal_name(sig),
                prior,
            )

        return installed_any


def _reset_for_tests() -> None:
    """Internal test helper: restore prior handlers and clear the
    per-signal map.

    Production code MUST NOT call this. The signal-observability test
    module uses it to install/uninstall the handler set within a single
    pytest process without leaking handlers to the next test.
    """
    with _install_lock:
        _cleanup_crash_file()
        for sig, prior in list(_prior_handlers.items()):
            try:
                signal.signal(sig, prior if prior is not None else signal.SIG_DFL)
            except (OSError, ValueError):
                pass
        _prior_handlers.clear()


def _get_installed_handlers() -> dict[int, object]:
    """Internal test helper: snapshot of the saved prior-handler map."""
    return dict(_prior_handlers)
