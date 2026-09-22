# SPDX-License-Identifier: Apache-2.0
"""Runtime wiring for the default-on consent decision (telemetry v2).

:mod:`rapid_mlx.telemetry.consent_decision` holds the complete rule set as
one pure function (:func:`~consent_decision.decide`); this module is the
ONLY thing that calls it. It turns the decision into process behaviour:

- :func:`read_stored_consent` — raw, schema-tolerant read of
  ``~/.rapid-mlx/telemetry-consent.yaml``.
- :func:`detect_role` — classify this process (interactive CLI, headless
  CLI, desktop sidecar).
- :func:`kill_switch_active` — env kill switches OR ``--no-telemetry``.
- :func:`resolve` — the memoized per-process :class:`Decision`.
- :func:`deliver_notice_if_needed` — write the role-appropriate disclosure.
- :func:`apply_write_back` — atomic, merging, locked write-back.
- :func:`startup` — resolve + deliver notice + write back, in that order.
- :func:`upload_allowed` — the single v2 gate the sender consults, with a
  live re-check of the kill switches and the stored ``consent`` so a
  mid-session ``telemetry off`` goes dark on the next capture.

Two v2 compatibility rules are load-bearing:

1. The schema-validating v1 consent reader is NEVER called on this path. It
   collapses every record whose ``schema_version`` is not the current one
   into "absent / never prompted" — and the desktop app writes
   ``schema_version: 1`` — so a stored ``consent: false`` from the desktop
   would read as a fresh install and turn row 4
   (``legacy_refusal_migrated``, never uploads) into row 1
   (``fresh_install_notice``, uploads after the notice). Feeding that
   into :func:`resolve` would start uploading on installs that said no.
   :func:`read_stored_consent` therefore ignores ``schema_version``
   completely and reads the fields it needs straight from the mapping.
2. Every Python consent write now uses the locked merge writer in ``state.py``.
   Unknown keys such as ``desktop_consent`` survive, while the v2 write-back
   still never bumps ``schema_version``. Atomic replacement deliberately
   replaces a symlinked consent path with a regular file, matching v1.

The Swift writer does not yet take the sibling lock and still replaces the
whole file. T12 must add flock + merge there and set
``RAPID_MLX_PROCESS_ROLE=desktop-sidecar`` for spawned sidecars.

``RAPID_MLX_WATCHDOG_PPID`` implying SIDECAR deliberately includes child
servers spawned by rapid-mlx on CLI machines: those children ride the parent
process's marker and must not independently disclose or mutate consent.

The notice goes straight to file descriptor 2 via :func:`os.write` — never
through Python's buffered ``sys.stderr`` (a reader-less pipe would otherwise
turn a successful command into exit status 120 during interpreter shutdown),
never stdout (whose byte-cleanliness many ``--json`` modes depend on), and
never logging (whose configuration must not be able to silence disclosure).

``ProcessRole.DESKTOP`` is a decision-table role for the Swift application,
not a role Python detects. A Python caller that nevertheless claims DESKTOP is
treated as inert: it prints nothing, writes nothing, and never uploads.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import yaml

import rapid_mlx
from rapid_mlx._parent_watchdog import ENV_VAR as _WATCHDOG_PPID_ENV
from rapid_mlx.telemetry import state
from rapid_mlx.telemetry.consent_decision import (
    DISCLOSURE_REVISION,
    REASON_INVALID_INPUT,
    REASON_LEGACY_REFUSAL_MIGRATED,
    REASON_SIDECAR_WAITS_FOR_DESKTOP,
    Decision,
    ProcessRole,
    StoredConsent,
    WriteBack,
    decide,
)

logger = logging.getLogger(__name__)

#: Env var a supervisor (the desktop app, in T12) can set to declare this
#: process a sidecar. Only the single value below is honoured; anything
#: else falls through to the tty / watchdog-ppid detection.
_PROCESS_ROLE_ENV: Final = "RAPID_MLX_PROCESS_ROLE"
_DESKTOP_SIDECAR_ROLE_VALUE: Final = "desktop-sidecar"

#: How long :func:`upload_allowed` may reuse a stored-consent read keyed on
#: the consent file's ``(st_mtime_ns, st_size)``. Bounds how long a
#: mid-session ``telemetry off`` (or an externally withdrawn record) stays
#: invisible to the capture path.
_LIVE_CACHE_TTL_SECONDS: Final = 5.0

#: The default-on disclosure. ASCII-only for the same reason as
#: ``consent._DISCLOSURE``: a terminal with an ASCII-only stderr encoding
#: must never turn a disclosure into ``UnicodeEncodeError``.
NOTICE_TEXT: Final[str] = """\
NOTICE: This version of Rapid-MLX turns anonymous usage reporting on by
default -- including for installs that previously turned it off.

What is reported: anonymous, metadata-only usage events (chip family,
OS family, subcommand, coarse timing and performance buckets, anonymous
crash fingerprints -- never prompts, responses, file paths, or API key
values). Events are received by PostHog Cloud, a hosted analytics
service located in the United States. IP and location are not recorded;
no per-person profile is built. Nothing is reported before this notice
has been shown.

To turn reporting back off, use any of:

  rapid-mlx telemetry disable
  export RAPID_MLX_TELEMETRY=0
  export DO_NOT_TRACK=1

Opting out takes effect immediately. This notice is shown once per
disclosure revision; see `rapid-mlx telemetry preview` for the exact
event shape and the project README's Telemetry section for details.
"""

NOTICE_LINE: Final[str] = (
    "rapid-mlx: anonymous usage reporting is ON (PostHog Cloud, US; no "
    "IP/location, no prompts or outputs). Turn off: rapid-mlx telemetry "
    "disable | RAPID_MLX_TELEMETRY=0 | DO_NOT_TRACK=1. Details: "
    "https://rapidmlx.com/docs/telemetry"
)

_NOTICE_MIGRATION_LINE: Final[str] = (
    "rapid-mlx: anonymous usage reporting was turned on by default in this "
    "version, including for installs that had turned it off (PostHog Cloud, "
    "US; no IP/location, no prompts or outputs). Turn off: rapid-mlx telemetry "
    "disable | RAPID_MLX_TELEMETRY=0 | DO_NOT_TRACK=1. Details: "
    "https://rapidmlx.com/docs/telemetry"
)

REASON_READ_ERROR: Final[str] = "read_error"
_NO_WRITE_BACK: Final = WriteBack(False, False, False)

_ABSENT_CONSENT: Final = StoredConsent(
    consent=None, recorded_version=None, notice_revision_seen=None
)

#: Injectable monotonic clock (Manager Decision 5) so tests can expire the
#: live-recheck TTL without sleeping.
_clock: Callable[[], float] = time.monotonic

# Per-process latches. ``_decision`` / ``_resolved_role`` memoize the
# resolution; ``_resolve_lock`` makes the once-per-process call exact even
# when several capture threads race the first resolution.
_decision: Decision | None = None
_resolved_role: ProcessRole | None = None
_notice_delivered = False
_startup_write_back_done = False
_live_cache: _LiveCache | None = None
_resolve_lock = threading.Lock()


@dataclass(frozen=True)
class _LiveCache:
    """Cached :class:`StoredConsent` keyed on the file's stat fingerprint."""

    fingerprint: tuple[int, int] | None
    read_at: float
    stored: StoredConsent | None


def _reset_runtime_state_for_tests() -> None:
    """Clear every per-process latch. Test infrastructure only."""
    global _decision, _resolved_role, _notice_delivered, _startup_write_back_done
    global _live_cache
    _decision = None
    _resolved_role = None
    _notice_delivered = False
    _startup_write_back_done = False
    _live_cache = None


def _running_version() -> str:
    """The running rapid-mlx version, passed through unmodified.

    Deliberately NOT normalized: the ``0.0.0`` editable-install sentinel
    from ``rapid_mlx/__init__.py`` must reach :func:`decide` verbatim so a
    version-unknown runtime takes the unparseable branch (and a refusal
    recorded beside it is never migrated).
    """
    return cast(str, rapid_mlx.__version__)


def _read_consent_mapping(path: Path) -> dict[str, Any] | None:
    """Parse the consent file. ``{}`` when absent, ``None`` when unreadable.

    Distinguishing absent (a fresh install may need its record CREATED by
    the row-1/7 write-back) from unreadable (a corrupt record must be
    preserved, not overwritten — merging is impossible without risking the
    loss of unknown keys such as ``desktop_consent``).
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        # OSError: permissions, a directory where the file should be, ...
        # ValueError: UnicodeDecodeError on binary junk.
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        # An empty file, a scalar, a list — anything that is not a
        # mapping carries no readable decision.
        return None
    return data


def read_stored_consent() -> StoredConsent | None:
    """Raw, schema-tolerant read of the consent record.

    Never delegates to the schema-validating v1 reader (see module
    docstring): ``schema_version`` is ignored completely, because the desktop writes
    records stamped ``1`` and v1 collapses them to "absent" — which would
    misread a stored refusal as a fresh install.

    ``consent`` is taken from the ``consent`` key only if it is a real
    bool (YAML ``1`` / ``"true"`` do not count); ``recorded_version`` from
    ``prompted_version`` only if it is a str; ``notice_revision_seen``
    from the flat top-level marker only if it is an int that is not a
    bool. A missing file yields the all-``None`` record. An unreadable file
    or non-mapping document returns ``None`` so :func:`resolve` can fail
    closed instead of treating corruption as a fresh install. Never raises.
    """
    data = _read_consent_mapping(state.consent_path())
    if data is None:
        return None
    consent = data.get("consent")
    if not isinstance(consent, bool):
        consent = None
    recorded_version = data.get("prompted_version")
    if not isinstance(recorded_version, str):
        recorded_version = None
    notice_revision_seen = data.get("notice_revision_seen")
    if not isinstance(notice_revision_seen, int) or isinstance(
        notice_revision_seen, bool
    ):
        notice_revision_seen = None
    return StoredConsent(
        consent=consent,
        recorded_version=recorded_version,
        notice_revision_seen=notice_revision_seen,
    )


def detect_role() -> ProcessRole:
    """Classify this process for the decision table.

    Order matters:

    1. ``RAPID_MLX_PROCESS_ROLE=desktop-sidecar`` — explicit supervisor
       declaration (honoured now; T12 makes the desktop set it).
    2. ``RAPID_MLX_WATCHDOG_PPID`` set to an integer greater than one — the
       desktop spawn helper and every Python spawner (``cli.py``, ``run``,
       ``share``, ``bench``) stamp it for the parent-watchdog. Empty,
       malformed, zero, negative, and PID 1 values fall through.
    3. Both stdin and stderr are ttys — interactive CLI.
    4. Otherwise — headless CLI. ``DESKTOP`` is never produced by Python;
       it is the Swift app itself.
    """
    role_raw = os.environ.get(_PROCESS_ROLE_ENV, "").strip()
    if role_raw == _DESKTOP_SIDECAR_ROLE_VALUE:
        return ProcessRole.SIDECAR
    ppid_raw = os.environ.get(_WATCHDOG_PPID_ENV, "").strip()
    if ppid_raw:
        try:
            ppid = int(ppid_raw)
        except ValueError:
            pass
        else:
            if ppid > 1:
                return ProcessRole.SIDECAR
    try:
        stdin_isatty = bool(sys.stdin is not None and sys.stdin.isatty())
        stderr_isatty = bool(sys.stderr is not None and sys.stderr.isatty())
    except (AttributeError, OSError, ValueError):
        stdin_isatty = stderr_isatty = False
    if stdin_isatty and stderr_isatty:
        return ProcessRole.INTERACTIVE_CLI
    return ProcessRole.HEADLESS_CLI


def kill_switch_active() -> bool:
    """Absolute off-switch for the v2 path: env kill switches OR the CLI flag.

    ``state._env_kill_switch_active()`` deliberately does not know about
    ``--no-telemetry``; the v2 caller must OR it in. The CLI flag is read
    live through the module (never a from-import copy) so a mid-process
    ``set_cli_kill_switch(...)`` is honoured by the next check.
    """
    return state._env_kill_switch_active() or bool(state._cli_kill_switch_active)


def resolve(role: ProcessRole | None = None) -> Decision:
    """Resolve the consent decision once per process; memoized.

    Calls :func:`~consent_decision.decide` with the raw stored record,
    the detected (or caller-provided) role, the live kill-switch state,
    and ``rapid_mlx.__version__`` passed through untouched. The ``role``
    override only applies to the FIRST resolution — later calls return
    the memoized decision whatever they pass. The decision's ``reason``
    is logged at DEBUG; ``invalid_input`` is additionally logged at
    WARNING because it means a corrupt record or a bug and must be
    visible.
    """
    global _decision, _resolved_role
    if _decision is not None:
        return _decision
    with _resolve_lock:
        if _decision is not None:
            return _decision
        detected = detect_role() if role is None else role
        stored = read_stored_consent()
        if detected is ProcessRole.DESKTOP:
            # DESKTOP belongs to the Swift owner of the consent record. Python
            # never detects it; a caller claiming it must be completely inert.
            decision = Decision(
                False,
                False,
                _NO_WRITE_BACK,
                REASON_SIDECAR_WAITS_FOR_DESKTOP,
            )
        elif stored is None:
            decision = Decision(False, False, _NO_WRITE_BACK, REASON_READ_ERROR)
        else:
            decision = decide(
                stored,
                detected,
                kill_switch_active=kill_switch_active(),
                running_version=_running_version(),
            )
        if decision.reason == REASON_READ_ERROR:
            logger.warning(
                "Telemetry consent record is unreadable (reason=%s); "
                "telemetry stays off for this run.",
                decision.reason,
            )
        if decision.reason == REASON_INVALID_INPUT:
            logger.warning(
                "Telemetry consent record is corrupt or unreadable "
                "(reason=%s); telemetry stays off for this run.",
                decision.reason,
            )
        logger.debug(
            "Telemetry consent decision: reason=%s upload_now=%s "
            "deliver_notice=%s write_back=(set_consent_true=%s, "
            "stamp_version=%s, mark_notice_seen=%s)",
            decision.reason,
            decision.upload_now,
            decision.deliver_notice,
            decision.write_back.set_consent_true,
            decision.write_back.stamp_version,
            decision.write_back.mark_notice_seen,
        )
        _decision = decision
        _resolved_role = detected
        return _decision


def refresh_decision() -> Decision:
    """Re-resolve the upload decision from disk after an explicit consent write.

    This narrow production seam replaces only the memoized decision and live
    consent cache. Notice delivery and startup write-back latches are retained.
    The same lock as :func:`resolve` makes the replacement atomic for capture
    threads. The process role already resolved at startup is preserved.
    """
    global _decision, _resolved_role, _live_cache
    with _resolve_lock:
        detected = _resolved_role if _resolved_role is not None else detect_role()
        stored = read_stored_consent()
        if detected is ProcessRole.DESKTOP:
            decision = Decision(
                False,
                False,
                _NO_WRITE_BACK,
                REASON_SIDECAR_WAITS_FOR_DESKTOP,
            )
        elif stored is None:
            decision = Decision(False, False, _NO_WRITE_BACK, REASON_READ_ERROR)
        else:
            decision = decide(
                stored,
                detected,
                kill_switch_active=kill_switch_active(),
                running_version=_running_version(),
            )
        _decision = decision
        _resolved_role = detected
        _live_cache = None
        return decision


def _write_stderr_unbuffered(data: bytes) -> bool:
    """Write all ``data`` to fd 2 without touching ``sys.stderr``.

    Retries interrupted writes, handles partial progress, and returns False on
    EPIPE, EBADF, zero progress, or any other OS-level failure. In particular,
    no bytes remain buffered for CPython to flush during finalization.
    """
    if sys.__stderr__ is None:
        return False
    offset = 0
    while offset < len(data):
        try:
            written = os.write(2, data[offset:])
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            return False
        if written <= 0:
            return False
        offset += written
    return True


def deliver_notice_if_needed(
    decision: Decision | None = None, *, long_lived: bool = False
) -> bool:
    """Write the role-appropriate disclosure to fd 2, once per process.

    Returns True IFF the notice bytes were fully written by THIS call.
    The latch flips only on success: a failed write keeps
    :func:`upload_allowed` False for rows 1/7, because the privacy
    invariant is "delivered", not "attempted". For a headless short command,
    the one-line notice is emitted only when ``decision.deliver_notice`` is
    true (rows 1/4/7), then the marker silences later invocations. Long-lived
    server starts additionally emit the one-line reminder on every uploading
    invocation. Interactive CLI behavior is unchanged. A broken fd 2 never
    crashes the user's command.
    """
    global _notice_delivered
    if _notice_delivered:
        return False
    if decision is None:
        decision = resolve()
    headless_notice = _resolved_role is ProcessRole.HEADLESS_CLI and (
        decision.deliver_notice or (long_lived and decision.upload_now)
    )
    interactive_notice = (
        _resolved_role is ProcessRole.INTERACTIVE_CLI and decision.deliver_notice
    )
    if not interactive_notice and not headless_notice:
        return False
    if headless_notice:
        notice = (
            _NOTICE_MIGRATION_LINE
            if decision.reason == REASON_LEGACY_REFUSAL_MIGRATED
            else NOTICE_LINE
        ) + "\n"
    else:
        notice = NOTICE_TEXT
    if not _write_stderr_unbuffered(notice.encode("ascii")):
        return False
    _notice_delivered = True
    return True


def notice_was_delivered() -> bool:
    """Whether this process fully delivered the current v2 disclosure."""
    return _notice_delivered


def _merge_write_back(data: dict[str, Any], write_back: WriteBack) -> dict[str, Any]:
    """Merge the requested fields into ``data``, keeping every other key.

    ``mark_notice_seen`` only ever RAISES ``notice_revision_seen`` to
    :data:`~consent_decision.DISCLOSURE_REVISION` — never lowers it (a
    newer-revision marker written by a future release must survive a
    downgrade's write-back).
    """
    merged = dict(data)
    if write_back.set_consent_true:
        merged["consent"] = True
    if write_back.stamp_version:
        merged["prompted_version"] = _running_version()
    if write_back.mark_notice_seen:
        seen = merged.get("notice_revision_seen")
        if not isinstance(seen, int) or isinstance(seen, bool):
            seen = 0
        merged["notice_revision_seen"] = max(seen, DISCLOSURE_REVISION)
    return merged


def _locked_write_back(path: Path, write_back: WriteBack) -> bool:
    """Read-merge-write the consent file under an exclusive sibling lock.

    Writers never unlink the lock file: unlinking would split waiters onto
    different inodes and defeat serialization (same reasoning as
    ``rapid_mlx/_mirror.py``). ``telemetry reset`` removes it best-effort.
    """
    return state._locked_merge_consent(lambda data: _merge_write_back(data, write_back))


def apply_write_back(write_back: WriteBack | None = None) -> bool:
    """Persist the decision's write-back; atomic, merging, locked.

    Resolves (memoized) first so the SIDECAR gate sees this process's
    role even when called with an explicit ``write_back``. Returns False
    — touching nothing — when the role is :attr:`ProcessRole.SIDECAR`
    (the desktop owns the record and the disclosure, decision-table
    invariant b), when the requested write-back is empty, when the stored
    file is present but unreadable, or when any filesystem step fails.
    Any failure preserves the old record. ``schema_version`` is never
    bumped (Manager Decision 2).
    """
    decision = resolve()
    if write_back is None:
        write_back = decision.write_back
    if _resolved_role in (ProcessRole.SIDECAR, ProcessRole.DESKTOP):
        return False
    if not (
        write_back.set_consent_true
        or write_back.stamp_version
        or write_back.mark_notice_seen
    ):
        return False
    try:
        return _locked_write_back(state.consent_path(), write_back)
    except (OSError, ValueError, yaml.YAMLError):
        logger.debug(
            "consent write-back failed; stored record unchanged", exc_info=True
        )
        return False


def _live_stored_consent() -> StoredConsent | None:
    """Stored consent with a stat-keyed TTL cache for the capture hot path.

    Cache key is ``(st_mtime_ns, st_size)``; entries older than
    :data:`_LIVE_CACHE_TTL_SECONDS` (on the injectable ``_clock``) are
    re-read even when the fingerprint matches. A missing file caches a
    ``None`` fingerprint, so repeated captures on a fresh install do not
    re-stat-and-read every time either.
    """
    path = state.consent_path()
    try:
        st = path.stat()
        fingerprint: tuple[int, int] | None = (st.st_mtime_ns, st.st_size)
    except OSError:
        fingerprint = None
    now = _clock()
    cached = _live_cache
    if (
        cached is not None
        and cached.fingerprint == fingerprint
        and (now - cached.read_at) < _LIVE_CACHE_TTL_SECONDS
    ):
        return cached.stored
    stored = read_stored_consent()
    _set_live_cache(_LiveCache(fingerprint=fingerprint, read_at=now, stored=stored))
    return stored


def _set_live_cache(entry: _LiveCache) -> None:
    """Store the live-read cache entry. Racy writes are benign (same data)."""
    global _live_cache
    _live_cache = entry


def upload_allowed() -> bool:
    """The single v2 answer: may this process upload right now?

    False when the resolved decision refuses to upload (kill switch,
    refusal, sidecar without the marker, the migrating run itself), when
    the decision hands out a notice that has not actually been delivered
    in this process (rows 1/7), and — even on an uploading decision —
    when the kill switches have flipped on or the stored ``consent`` has
    become ``False`` since startup (live re-check, TTL-cached).
    """
    decision = resolve()
    if not decision.upload_now:
        return False
    if decision.deliver_notice and not _notice_delivered:
        return False
    if kill_switch_active():
        return False
    stored = _live_stored_consent()
    if stored is None:
        return False
    if stored.consent is False:
        return False
    marker_present = (
        stored.notice_revision_seen is not None
        and stored.notice_revision_seen >= DISCLOSURE_REVISION
    )
    if stored.consent is not True and not marker_present:
        return False
    return True


def startup(*, role: ProcessRole | None = None, long_lived: bool = False) -> Decision:
    """One-shot startup: resolve, deliver the notice, apply the write-back.

    ``long_lived=True`` is reserved for server entrypoints and enables their
    every-start headless reminder; short headless commands disclose only when
    the decision requires it for the current revision.

    The order is load-bearing: rows 1 and 7 permit uploading in this run,
    but only AFTER the notice has actually been delivered, so the notice
    must precede everything else and the write-back must not be read as
    permission. The write-back runs once per process (``startup`` is by
    definition called once per entrypoint; repeated calls skip it so a
    memoized decision is never replayed onto a changed disk state).
    """
    global _decision, _resolved_role, _startup_write_back_done
    try:
        decision = resolve(role=role)
        delivered = deliver_notice_if_needed(decision, long_lived=long_lived)
        if not _startup_write_back_done:
            _startup_write_back_done = True
            if not decision.deliver_notice or delivered:
                apply_write_back(decision.write_back)
        return decision
    except Exception:
        # Consent wiring is subordinate to the host command. Unexpected bugs,
        # exotic stdio, and filesystem surprises all fail closed for this
        # process and must never alter its success/failure behavior.
        try:
            logger.debug(
                "unexpected telemetry consent startup failure; telemetry "
                "disabled for this process",
                exc_info=True,
            )
        except Exception:
            pass
        blocked = Decision(False, False, _NO_WRITE_BACK, REASON_INVALID_INPUT)
        _decision = blocked
        _resolved_role = role if isinstance(role, ProcessRole) else ProcessRole.SIDECAR
        _startup_write_back_done = True
        return blocked
