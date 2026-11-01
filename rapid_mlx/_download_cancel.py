# SPDX-License-Identifier: Apache-2.0
"""Interruptible foreground waits for model transfers."""

from __future__ import annotations

import atexit
import logging
import os
import re
import signal
import stat
import sys
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def _hard_exit_cancelled(code: int) -> None:
    if "pytest" in sys.modules and os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        atexit._run_exitfuncs()  # noqa: SLF001
    except BaseException:  # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except BaseException:  # noqa: BLE001
            pass
    os._exit(code)


def _cancel_backstop(code: int) -> None:
    if "pytest" in sys.modules and os.environ.get("PYTEST_CURRENT_TEST"):
        return
    os._exit(code)


# Upper bound for cleanup and exit hooks after a cancelled transfer.
_CANCEL_DEADLINE_SECONDS = 2.0


def _remove_owned_incomplete_files() -> None:
    """Unlink only open, non-resumable Hub scratch files."""
    try:
        fd_root = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"
        descriptors = os.listdir(fd_root)
    except Exception:
        return
    for descriptor in descriptors:
        try:
            if sys.platform == "darwin":
                import fcntl

                path = os.fsdecode(
                    fcntl.fcntl(int(descriptor), fcntl.F_GETPATH, bytes(1024)).split(
                        b"\0", 1
                    )[0]
                )
            else:
                path = os.readlink(f"{fd_root}/{descriptor}")
            if (
                os.path.basename(os.path.dirname(path)) == "blobs"
                and re.fullmatch(
                    r"[^.]+\.[0-9a-f]{8}\.incomplete", os.path.basename(path)
                )
                and stat.S_ISREG(os.lstat(path).st_mode)
            ):
                os.unlink(path)
        except Exception:
            pass


def run_cancellable_download(operation: Callable[[], T], *, retry_hint: str) -> T:
    """Run a transfer with prompt signal cancellation in the main thread."""
    if threading.current_thread() is not threading.main_thread():
        return operation()
    # A caller with its own SIGINT handler owns cancellation. An ignored
    # signal (e.g. SIGINT in a background job) stays ignored.
    sigint = signal.getsignal(signal.SIGINT)
    if callable(sigint) and sigint is not signal.default_int_handler:
        return operation()
    owned = [
        sig
        for sig in (signal.SIGINT, signal.SIGTERM)
        if signal.getsignal(sig) != signal.SIG_IGN
    ]
    if not owned:
        return operation()

    done = threading.Event()
    result: list[T] = []
    errors: list[BaseException] = []
    first_signum: int | None = None

    def worker() -> None:
        try:
            result.append(operation())
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal first_signum
        if first_signum is not None:
            os._exit(128 + first_signum)
        first_signum = signum
        raise KeyboardInterrupt

    previous = {sig: signal.getsignal(sig) for sig in owned}
    try:
        try:
            for sig in previous:
                signal.signal(sig, handle_signal)
            threading.Thread(
                target=worker, daemon=True, name="rapid-mlx-download"
            ).start()
            while not done.wait(0.1):
                pass
        except KeyboardInterrupt:
            code = 128 + (first_signum or signal.SIGINT)
            # Exit hooks or Xet teardown may wait on an abandoned worker.
            backstop = threading.Timer(
                _CANCEL_DEADLINE_SECONDS, _cancel_backstop, [code]
            )
            backstop.daemon = True
            backstop.start()
            print("\n  Cancelling download…", file=sys.stderr, flush=True)
            # Exit hooks close the Hub session under the abandoned workers;
            # keep their "resume download" warnings off the terminal.
            logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL + 1)
            try:
                from huggingface_hub.utils._xet import abort_xet_session

                abort_xet_session()
            except Exception:
                pass
            _remove_owned_incomplete_files()
            print("  Download cancelled. Completed files stay cached.", file=sys.stderr)
            print(retry_hint, file=sys.stderr, flush=True)
            _hard_exit_cancelled(code)
            raise SystemExit(code)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if errors:
        raise errors[0]
    return result[0]
