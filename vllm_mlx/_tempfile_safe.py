"""Guaranteed-cleanup tempfile helper for rapid-mlx.

Fix for GH #719: ``rapid-mlx chat`` was leaking one zero-byte
``rapid-mlx-chat-*.log`` per invocation (3496+ stragglers reported
on a single dev machine). The root cause was the documented Python
anti-pattern::

    log_path = tempfile.NamedTemporaryFile(
        prefix="rapid-mlx-chat-", suffix=".log", delete=False,
    ).name

This creates the file on disk and returns just the path. Because
``delete=False`` was passed and no surrounding ``try/finally`` or
``atexit`` hook owns the path, ANY exception between creation and
the eventual cleanup-by-proc-teardown (KeyboardInterrupt, BrokenPipe
on ``print``, ImportError of a downstream module, etc.) orphans the
file. The downstream ``_teardown_proc`` cleanup walks ``_active_procs``
— if the proc never made it onto that list, the file path is lost.

This module provides ``managed_tempfile_path`` — a context manager
that:

1. Creates the tempfile via ``mkstemp`` (closes the FD immediately,
   so there is no dangling descriptor) and returns the path string.
2. Registers an ``atexit`` hook to unlink the path *before* the
   context body runs. This guarantees cleanup even if the body
   raises BEFORE the context's ``__exit__`` runs (e.g. the body
   triggers a sub-helper that calls ``sys.exit()`` from within an
   ``atexit``-bypassing path, or a hard interpreter shutdown).
3. On normal context exit, unlinks the path and unregisters the
   ``atexit`` hook so we don't accumulate dead callables in long-
   running parents that create + tear down many tempfiles.
4. Exposes ``release()`` for callers that need to hand ownership
   over to another lifecycle (e.g. the chat REPL hands the log
   path to ``_teardown_proc`` which has its own per-proc cleanup
   policy — the helper steps aside, but only AFTER the proc has
   been registered onto the active-procs list, closing the race
   window that #719 reported).

Why not just use ``NamedTemporaryFile()`` (no ``delete=False``)?
- The chat-server use case opens the path in a *separate* file
  handle in the parent and inherits that handle into a subprocess.
  ``NamedTemporaryFile``'s own FD is irrelevant; what matters is
  the path persisting until the subprocess terminates. The unlink
  policy is therefore decoupled from the FD lifecycle and has to
  be expressed explicitly — that is what this helper provides.

Why both ``atexit`` AND ``__exit__``?
- ``__exit__`` covers normal control flow (return, exception
  caught by the caller).
- ``atexit`` covers paths where ``__exit__`` is bypassed:
  ``sys.exit()`` from inside the body (the chat command does this
  on readiness failure), a hard exception that propagates past
  the context, or any helper that does ``os._exit`` in an unusual
  way. ``atexit`` does NOT cover SIGKILL or ``os._exit`` itself,
  which is acceptable — those are not the failure modes #719
  observed.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import tempfile
import threading
from collections.abc import Iterator

# Module-level set of every path the helper is currently watching.
# A single shared atexit hook walks this set on interpreter exit so
# we don't register N hooks for N tempfiles (atexit walks them all
# anyway, but a single hook keeps the registry small + introspectable
# in tests).
_pending_paths: set[str] = set()
_pending_lock = threading.Lock()
_atexit_registered = False


def _atexit_reap_all() -> None:
    """Unlink every still-pending tempfile path. Idempotent."""
    # Snapshot under the lock so a parallel ``release`` can't mutate
    # the set mid-walk. We don't hold the lock during ``os.unlink``
    # because that could deadlock if the unlink path triggers a
    # finalizer that touches the same lock (unlikely but cheap to
    # avoid).
    with _pending_lock:
        snapshot = list(_pending_paths)
        _pending_paths.clear()
    for path in snapshot:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            # Best-effort: we're tearing down the interpreter, no
            # one will see the error anyway. Silencing here matches
            # the rest of the codebase's atexit conventions.
            pass


def _ensure_atexit_registered() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    # The lock guards the flag flip so two threads racing on first
    # ``managed_tempfile_path`` call only register once.
    with _pending_lock:
        if _atexit_registered:
            return
        atexit.register(_atexit_reap_all)
        _atexit_registered = True


class _TempfileHandle:
    """Returned object that exposes ``release`` while behaving like ``str``.

    The body of ``managed_tempfile_path`` yields one of these so that
    callers can either:

    - Use it as a path string (because ``__fspath__`` and ``__str__``
      both return the underlying path — drop-in for code that does
      ``open(log_path, "w")`` or ``subprocess.Popen(..., stdout=open(log_path))``).
    - Call ``.release()`` to hand ownership over to another lifecycle,
      after which the helper will NOT unlink on context exit.
    """

    __slots__ = ("_path", "_released")

    def __init__(self, path: str) -> None:
        self._path = path
        self._released = False

    @property
    def path(self) -> str:
        return self._path

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> str:
        """Hand ownership over. Caller is now responsible for unlink.

        Returns the path so common usage can be a one-liner::

            with managed_tempfile_path(...) as h:
                proc = spawn(h)  # spawn registers h.path on the proc
                final_path = h.release()
        """
        self._released = True
        with _pending_lock:
            _pending_paths.discard(self._path)
        return self._path

    # Make the handle interchangeable with the path string for code
    # that takes ``str | PathLike``. This keeps call-sites readable
    # (``with managed_tempfile_path(...) as p: open(p)``).
    def __fspath__(self) -> str:
        return self._path

    def __str__(self) -> str:
        return self._path

    def __repr__(self) -> str:
        state = "released" if self._released else "pending"
        return f"_TempfileHandle({self._path!r}, {state})"


@contextlib.contextmanager
def managed_tempfile_path(
    *,
    prefix: str = "tmp",
    suffix: str = "",
    dir: str | None = None,  # noqa: A002 — mirrors tempfile.mkstemp signature
) -> Iterator[_TempfileHandle]:
    """Create a tempfile and guarantee cleanup on context exit OR atexit.

    Drop-in replacement for the ``NamedTemporaryFile(...).name`` anti-
    pattern. The yielded object is path-like (``os.PathLike[str]``)
    so existing call-sites can pass it directly to ``open()``,
    ``subprocess.Popen(stdout=...)``, etc.

    On context exit (normal OR exception) the path is unlinked.

    If the caller transfers ownership to another lifecycle, calling
    ``.release()`` on the handle suppresses both the context-exit
    unlink and the atexit fallback. The caller becomes responsible
    for unlinking; the helper is now hands-off.

    Args:
        prefix: Filename prefix (default ``"tmp"``).
        suffix: Filename suffix (default ``""``).
        dir: Parent directory (default: system temp dir).

    Yields:
        ``_TempfileHandle`` exposing ``.path``, ``.release()``, and
        ``__fspath__``.
    """
    # ``mkstemp`` returns an open FD + path. We close the FD here so
    # there is no descriptor leak — the caller will open the path
    # themselves in whichever mode they need. This is the recommended
    # pattern in the stdlib docs for "I just need the path".
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
    try:
        os.close(fd)
    except OSError:
        # mkstemp succeeded but close failed — extremely rare. We
        # still want the path tracked so cleanup happens.
        pass

    _ensure_atexit_registered()
    with _pending_lock:
        _pending_paths.add(path)

    handle = _TempfileHandle(path)
    try:
        yield handle
    finally:
        if not handle.released:
            with _pending_lock:
                _pending_paths.discard(path)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                # Same rationale as ``_atexit_reap_all``: best-
                # effort cleanup; do not let a cleanup error mask
                # the real exception (if any) propagating through
                # the ``finally``.
                pass


def _pending_snapshot() -> set[str]:
    """Return a snapshot of currently-tracked paths.

    Exposed only for tests — production code should not depend on
    the internal registry.
    """
    with _pending_lock:
        return set(_pending_paths)


__all__ = ["managed_tempfile_path"]
