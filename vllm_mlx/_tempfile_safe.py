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
2. Adds the path to a module-level ``_pending_paths`` set. A single
   shared ``atexit`` hook (installed lazily the first time anyone
   uses the helper) walks this set at interpreter shutdown so any
   path still pending at exit gets unlinked. The hook is installed
   once per process and stays registered for the lifetime of the
   interpreter — only the registry shrinks as paths are released
   or cleaned up by ``__exit__``.
3. On normal context exit (including exceptions and ``SystemExit``,
   both of which trigger ``__exit__``), unlinks the path and drops
   it from the registry.
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

Coverage matrix:

- Normal exit / exception inside body / ``SystemExit`` (``sys.exit``):
  Python invokes the context manager's ``__exit__`` (or, for the
  generator-based helper here, the ``finally`` inside the ``@contextmanager``
  wrapper), which unlinks the path and removes it from the registry.
- Interpreter exit while a path is still pending (caller forgot to
  exit the context, or a hard exception aborted the unwind):
  The shared ``atexit`` hook reaps everything in ``_pending_paths``.
- ``os._exit()`` / SIGKILL: NEITHER ``__exit__`` nor ``atexit`` runs.
  These paths are explicitly NOT covered. They are not the failure
  modes #719 reported, and covering them would require a separate
  janitor process — out of scope for this helper.
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

        Codex round-4 finding: discard from the registry BEFORE
        setting ``self._released = True``, under the lock. The
        reverse order had a window where an interruption left
        ``released=True`` (so the context manager's ``finally``
        would skip the unlink) while the path was still in
        ``_pending_paths`` — atexit would then unlink a file the
        caller has just taken ownership of. With this ordering,
        an interruption mid-call either:

        - happens before the discard → ``released`` is still
          ``False``, ``finally`` reaps the path (and the atexit
          fallback would also reap it).
        - happens after the discard but before ``_released=True``
          → ``released`` is still ``False``, ``finally`` reaps the
          path (the caller's "I took ownership" is incomplete; the
          helper conservatively still owns).
        - happens after ``_released=True`` → call has completed
          successfully; both context manager and atexit step away.
        """
        with _pending_lock:
            _pending_paths.discard(self._path)
            self._released = True
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
    # Codex round-2 BLOCKING: ``mkstemp`` has already created the file
    # at this point. If anything between here and the ``yield`` raises
    # — including ``BaseException`` such as ``KeyboardInterrupt``
    # (SIGINT) or ``SystemExit`` (e.g. ``_sigterm_handler`` running on
    # the main thread) — the file would leak: not yet in
    # ``_pending_paths`` (so atexit can't see it), not yet inside the
    # yielded-context ``finally`` (so ``__exit__`` won't either). Wrap
    # the whole setup phase in a ``try/except BaseException`` so a
    # signal or import-time error during ``_ensure_atexit_registered``
    # cleans up the just-created file before propagating.
    try:
        try:
            os.close(fd)
        except OSError:
            # mkstemp succeeded but close failed — extremely rare. We
            # still want the path tracked so cleanup happens.
            pass

        _ensure_atexit_registered()
        with _pending_lock:
            _pending_paths.add(path)
    except BaseException:
        # Setup failed. Codex pr_validate round-1 BLOCKING: the cleanup
        # MUST leave the file owned by SOMETHING (this finally, or the
        # shared atexit registry) regardless of how it unwinds. The
        # simplest invariant: ensure the path is in ``_pending_paths``
        # BEFORE we attempt the best-effort unlink, and only discard
        # AFTER a successful unlink. That way:
        #   - If the helper made it to ``_pending_paths.add`` already,
        #     the discard-on-success ordering hands ownership cleanly.
        #   - If the exception came from BEFORE the ``add`` (e.g.
        #     ``_ensure_atexit_registered`` raised), we still arm the
        #     registry here so a second interrupt during ``os.unlink``
        #     below does NOT strand the file outside the registry.
        # Atexit also has to be registered at this point, otherwise
        # adding to the set is useless.
        try:
            _ensure_atexit_registered()
        except BaseException:  # noqa: BLE001 — best-effort
            pass
        with _pending_lock:
            _pending_paths.add(path)
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            # File still exists OR we don't have perms. Either way the
            # registry now owns it for the atexit pass; do NOT discard.
            pass
        else:
            # Unlink succeeded — drop from the registry.
            with _pending_lock:
                _pending_paths.discard(path)
        raise

    handle = _TempfileHandle(path)
    try:
        yield handle
    finally:
        # Codex round-5 finding #2: read ``handle.released`` under the
        # registry lock so a concurrent ``release()`` cannot interleave
        # between the read and the unlink. With the lock-held check,
        # either:
        #   - ``release()`` finished first → ``released=True`` → we
        #     skip the unlink (caller now owns).
        #   - ``release()`` is concurrent → it blocks on the lock,
        #     reads ``released=False``, and we unlink. The unlink is
        #     idempotent (``FileNotFoundError`` is tolerated by
        #     ``release()``'s atexit pass), so even if release()
        #     observes ``released=False`` and the caller later
        #     unlinks again, nothing breaks.
        # This serializes the ownership transition for the multi-
        # threaded edge case codex flagged. The chat REPL is
        # single-threaded today; the lock is cheap and removes the
        # foot-gun for future callers.
        with _pending_lock:
            should_unlink = not handle.released
        if should_unlink:
            # Codex round-3 BLOCKING: unlink BEFORE discarding from
            # the registry. The original order (discard → unlink) had
            # a window where a ``BaseException`` (Ctrl-C, SIGTERM-
            # induced ``SystemExit``) landing between the two
            # statements would leave the file on disk with no entry
            # in ``_pending_paths`` — the atexit fallback could never
            # see it.
            #
            # Pr_validate round-4 BLOCKING refinement: only discard
            # from the registry after a SUCCESSFUL unlink (or
            # ``FileNotFoundError``, which means the file was already
            # gone). If ``os.unlink`` raises another ``OSError`` (EBUSY,
            # EPERM, EIO), the file may still be on disk; the atexit
            # hook is then the only fallback that could retry, so we
            # keep the registry entry. ``BaseException`` (KI/SE) from
            # ``os.unlink`` propagates out before we reach the
            # ``else`` clause — same effect: entry stays.
            unlinked = False
            try:
                os.unlink(path)
                unlinked = True
            except FileNotFoundError:
                unlinked = True
            except OSError:
                # Same rationale as ``_atexit_reap_all``: best-
                # effort cleanup; do not let a cleanup error mask
                # the real exception (if any) propagating through
                # the ``finally``.
                pass
            if unlinked:
                with _pending_lock:
                    _pending_paths.discard(path)


def _pending_snapshot() -> set[str]:
    """Return a snapshot of currently-tracked paths.

    Exposed only for tests — production code should not depend on
    the internal registry.
    """
    with _pending_lock:
        return set(_pending_paths)


__all__ = ["managed_tempfile_path"]
