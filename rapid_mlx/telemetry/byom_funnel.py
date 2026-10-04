# SPDX-License-Identifier: Apache-2.0
"""Bring-your-own-model funnel context for the model lifecycle events.

``rapid-mlx serve`` / ``pull`` of a model that is not in the catalog runs the
metadata preflight (``rapid_mlx/byom/preflight.py``). This module remembers,
for the ONE model reference this process was invoked with, what that funnel
step decided, and the lifecycle builders in ``model_events`` merge it into
``model_pulled`` / ``model_pull_failed`` / ``model_served`` /
``model_serve_failed`` for that same reference only (a secondary audio or
embedding lane, or a later model swap, never inherits it).

Every value is a closed registry enum or ``true``:

* ``preflight`` — ``passed`` / ``refused`` / ``no_verdict`` (metadata could
  not prove anything, or the Hub is offline) / ``skipped``
  (``--no-preflight``) / ``cached`` (already in the local cache, not checked).
  Absent for catalog models.
* ``suggestion`` — refusals only: which kind of alternative the refusal named
  (``mlx_build`` / ``catalog`` / ``none``).
* ``support_request`` — refusals only: the closed outcome of the opt-in
  "Ask us to support it?" offer.
* ``via_suggestion`` — ``true`` when this invocation's model is one a refusal
  on this Mac suggested in the last seven days.

The last one needs a memory across processes. When (and only when) uploads
are allowed, a refusal records a one-way SHA-256 digest of each suggested
reference with its timestamp in ``~/.rapid-mlx/state/byom-suggested-recent.json``
(mode 0600, at most 32 entries, seven-day window); a later serve/pull of a
matching reference consumes the entry. The file holds no repo name in clear
and is never sent; only the ``via_suggestion`` boolean reaches the wire.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

PREFLIGHT_OUTCOMES = frozenset({"passed", "refused", "no_verdict", "skipped", "cached"})
SUGGESTION_KINDS = frozenset({"mlx_build", "catalog", "none"})
SUPPORT_REQUEST_OUTCOMES = frozenset(
    {
        "not_eligible",
        "sent",
        "declined",
        "no_answer",
        "non_interactive",
        "busy",
        "unreachable",
    }
)

SUGGESTION_WINDOW_SECONDS = 7 * 24 * 3600
_SUGGESTION_MAX_KEYS = 32

_lock = threading.Lock()
_refs: frozenset[str] = frozenset()
_context: dict[str, object] = {}
_consumed = [False]
_clock = time.time


def _norm(ref: object) -> str | None:
    if not isinstance(ref, str):
        return None
    value = ref.strip()
    if not value:
        return None
    try:
        if os.path.exists(value):
            return os.path.realpath(value)
    except Exception:
        pass
    return value.lower()


def _digest(ref: str) -> str:
    return hashlib.sha256(ref.encode("utf-8")).hexdigest()[:32]


def _upload_allowed() -> bool:
    try:
        from rapid_mlx.telemetry import track

        return track._upload_allowed() is True
    except Exception:
        return False


def _ledger_path() -> Path:
    from rapid_mlx.telemetry.state import _default_telemetry_dir

    return _default_telemetry_dir() / "state" / "byom-suggested-recent.json"


def _mutate_ledger(change: Callable[[dict[str, float], float], bool]) -> bool:
    """Apply ``change`` to the fresh ledger under its directory lock.

    ``change`` edits the mapping in place and returns its answer; the file is
    rewritten only when the mapping changed. Never raises: an unavailable
    state root, lock contention or a corrupt file all answer ``False``.
    """
    try:
        from rapid_mlx.telemetry.model_events import (
            _acquire_serve_failed_lock,
            _read_serve_failed_recent,
        )
        from rapid_mlx.telemetry.server_start import (
            _atomic_write_marker,
            _prepare_state_dir,
        )

        path = _ledger_path()
        if not _prepare_state_dir(path.parent):
            return False
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        dir_fd = os.open(path.parent, flags)
    except Exception:
        return False
    try:
        if not _acquire_serve_failed_lock(dir_fd):
            return False
        now = _clock()
        if not math.isfinite(now):
            return False
        before = _read_serve_failed_recent(path)
        recent = {
            key: stamp
            for key, stamp in before.items()
            if stamp <= now and now - stamp < SUGGESTION_WINDOW_SECONDS
        }
        answer = change(recent, now)
        if len(recent) > _SUGGESTION_MAX_KEYS:
            recent = dict(
                sorted(recent.items(), key=lambda item: item[1], reverse=True)[
                    :_SUGGESTION_MAX_KEYS
                ]
            )
        if recent != before:
            _atomic_write_marker(path, value=recent)
        return answer
    except Exception:
        return False
    finally:
        os.close(dir_fd)


def _suggested(refs: Iterable[str], *, consume: bool) -> bool:
    """Whether a fresh ledger entry matches ``refs``; ``consume`` removes it."""
    digests = {_digest(ref) for ref in refs}
    if not digests or not _ledger_path().exists():
        return False

    def change(recent: dict[str, float], now: float) -> bool:
        hit = [key for key in digests if key in recent]
        if consume:
            for key in hit:
                del recent[key]
        return bool(hit)

    return _mutate_ledger(change)


def _consume_suggestion(refs: Iterable[str]) -> bool:
    return _suggested(refs, consume=True)


def _record_suggestions(refs: Iterable[object]) -> None:
    digests = {_digest(norm) for norm in map(_norm, refs) if norm is not None}
    if not digests:
        return

    def change(recent: dict[str, float], now: float) -> bool:
        for key in digests:
            recent[key] = now
        return True

    _mutate_ledger(change)


def begin(refs: Iterable[object]) -> None:
    """Start the funnel context for this invocation's model reference(s).

    Called once by the serve/pull preflight hook for every invocation,
    catalog or not, so a suggested catalog alias can be recognised too.
    Never raises.
    """
    global _refs
    try:
        normalized = frozenset(n for n in map(_norm, refs) if n is not None)
        # Peek only: the entry is consumed by the first lifecycle event that
        # reports it, so a run that emits nothing (a cached pull, an aborted
        # confirmation) leaves the suggestion for the next attempt.
        via = (
            bool(normalized)
            and _upload_allowed()
            and _suggested(normalized, consume=False)
        )
        with _lock:
            _refs = normalized
            _context.clear()
            _consumed[0] = False
            if via:
                _context["via_suggestion"] = True
    except Exception:
        return


def set_preflight(outcome: str) -> None:
    if outcome not in PREFLIGHT_OUTCOMES:
        return
    with _lock:
        _context["preflight"] = outcome


def note_refusal(
    *,
    suggestion: object,
    support_request: object,
    suggested_refs: Iterable[object] = (),
) -> None:
    """Record a preflight refusal's suggestion kind and support outcome.

    ``suggested_refs`` are the model references the refusal told the user to
    try; their digests are remembered locally (uploads allowed only) so a
    later serve/pull of one can report ``via_suggestion``. Never raises.
    """
    try:
        with _lock:
            _context["preflight"] = "refused"
            if suggestion in SUGGESTION_KINDS:
                _context["suggestion"] = suggestion
            if support_request in SUPPORT_REQUEST_OUTCOMES:
                _context["support_request"] = support_request
        refs = list(suggested_refs)
        if refs and _upload_allowed():
            _record_suggestions(refs)
    except Exception:
        return


_REFUSAL_ONLY = ("suggestion", "support_request")


def props_for(model_ref: object, *, failed: bool) -> dict[str, object]:
    """Funnel props for a lifecycle event about ``model_ref``; never raises."""
    try:
        norm = _norm(model_ref)
        with _lock:
            if norm is None or norm not in _refs:
                return {}
            context = dict(_context)
            consume = context.get("via_suggestion") is True and not _consumed[0]
            if consume:
                _consumed[0] = True
            refs = _refs
        if consume:
            _consume_suggestion(refs)
        props: dict[str, object] = {}
        preflight = context.get("preflight")
        if preflight in PREFLIGHT_OUTCOMES:
            props["preflight"] = preflight
        if context.get("via_suggestion") is True:
            props["via_suggestion"] = True
        if failed and preflight == "refused":
            for key in _REFUSAL_ONLY:
                if key in context:
                    props[key] = context[key]
        return props
    except Exception:
        return {}


def _reset_for_tests() -> None:
    global _refs
    with _lock:
        _refs = frozenset()
        _context.clear()
        _consumed[0] = False
