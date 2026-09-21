# SPDX-License-Identifier: Apache-2.0
"""The default-on consent decision table as one pure function.

Release 0.15.0 moves telemetry from opt-in to default-on (owner decision,
Orca-style), including installs that previously recorded ``consent: false``.
"Flip the default" is not enough on its own: the stored decision outranks the
default, so every existing ``consent: false`` would stay off forever, and a
naive flip would also start uploading before anyone saw the new disclosure.
This module holds the complete rule set as ONE pure, exhaustively tested
function (:func:`decide`) that the CLI, the headless server, and the desktop
app's sidecar all obey (the desktop later mirrors it in Swift). It performs
no I/O and nothing calls it yet; wiring lands in a follow-up task.

The table below is the contract. ``C`` = :data:`DEFAULT_ON_CUTOFF`,
``R`` = :data:`DISCLOSURE_REVISION`. The "marker" is present iff
``notice_revision_seen >= R`` (an older revision counts as absent — the
install has not seen the *current* disclosure).

=== ========= ========= =============================================== ==== ====== ====== ============================================== ================================
 #  consent   marker    role                                            kill upload notice write-back                                    reason
=== ========= ========= =============================================== ==== ====== ====== ============================================== ================================
 0  any       any       any                                             on   no     no     none                                           ``kill_switch``
 -  any       any       any                                             off  no     no     none                                           ``pre_cutoff_runtime`` (only if ``running_version`` parses strictly below C)
 1  absent    absent    INTERACTIVE_CLI / HEADLESS_CLI / DESKTOP        off  yes*   yes    ``mark_notice_seen`` only                      ``fresh_install_notice``
 2  absent    absent    SIDECAR                                         off  no     no     none                                           ``sidecar_waits_for_desktop``
 3  absent    present   any (incl. SIDECAR)                             off  yes    no     none                                           ``marker_authorises``
 4  False     absent    INTERACTIVE_CLI / HEADLESS_CLI / DESKTOP        off  no     yes    ``set_consent_true`` + ``stamp_version`` + ``mark_notice_seen``  ``legacy_refusal_migrated``
 5  False     absent    SIDECAR                                         off  no     no     none                                           ``sidecar_waits_for_desktop``
 6  False     any       any (not legacy — a current refusal)            off  no     no     none                                           ``current_refusal``
 7  True      absent    INTERACTIVE_CLI / HEADLESS_CLI / DESKTOP        off  yes*   yes    ``mark_notice_seen`` only                      ``consented_needs_notice``
 8  True      absent    SIDECAR                                         off  no     no     none                                           ``sidecar_waits_for_desktop``
 9  True      present   any (incl. SIDECAR)                             off  yes    no     none                                           ``consented``
=== ========= ========= =============================================== ==== ====== ====== ============================================== ================================

(*) Rows 1 and 7: uploading is permitted in this run, but only AFTER the
notice has been delivered — the caller MUST deliver the notice before the
first upload of the run. Row 4 (the migrating run itself) never uploads at
all; migration takes effect from the next run.

A row-4 refusal is LEGACY iff ``consent is False`` AND the marker is absent
AND ``recorded_version`` parses as a release version strictly lower than C.
Pre-releases of C (e.g. ``0.15.0rc1``) are strictly lower than C, so they are
legacy. A missing or unparseable ``recorded_version`` is NOT legacy — the
refusal is treated as current and never auto-migrated (fail toward respecting
a refusal). A False with the marker present is always a current refusal.

Three invariants, enforced by tests over the full input cross product:

(a) Nothing is uploaded before the current disclosure revision has been
    delivered on this install: every ``upload_now=True`` row either has the
    marker already present (rows 3, 9) or hands the notice to the caller to
    deliver first (rows 1, 7); the migrating run (row 4) never uploads.
(b) A sidecar never writes (no ``WriteBack``) and never discloses (no
    notice) — the desktop process owns the record and the disclosure.
(c) A refusal recorded at or after the cutoff is never reversed
    automatically (row 6), whatever the role.

``decide`` never raises: any non-conforming input (wrong types, ``None``
role, hostile objects) yields the kill-switch-shaped decision (no upload, no
notice, no write-back) with reason ``invalid_input``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final, TypeAlias

#: The first release whose consent-writing code implements the new
#: default-on disclosure. Refusals recorded by versions strictly below this
#: cutoff are legacy and may be migrated (row 4); refusals recorded at or
#: after it are never reversed automatically.
DEFAULT_ON_CUTOFF: Final = "0.15.0"

#: Which revision of the disclosure copy this release ships. Bump whenever
#: the disclosure materially changes: bumping makes every install see the
#: notice again (their stored marker counts as absent against the new
#: revision).
DISCLOSURE_REVISION: Final = 1

REASON_KILL_SWITCH: Final = "kill_switch"
REASON_PRE_CUTOFF_RUNTIME: Final = "pre_cutoff_runtime"
REASON_INVALID_INPUT: Final = "invalid_input"
REASON_FRESH_INSTALL_NOTICE: Final = "fresh_install_notice"
REASON_SIDECAR_WAITS_FOR_DESKTOP: Final = "sidecar_waits_for_desktop"
REASON_MARKER_AUTHORISES: Final = "marker_authorises"
REASON_LEGACY_REFUSAL_MIGRATED: Final = "legacy_refusal_migrated"
REASON_CURRENT_REFUSAL: Final = "current_refusal"
REASON_CONSENTED_NEEDS_NOTICE: Final = "consented_needs_notice"
REASON_CONSENTED: Final = "consented"

#: The closed set of machine-readable reasons ``Decision.reason`` may hold.
#: Callers may switch on these; anything outside this tuple is a bug.
KNOWN_REASONS: Final[tuple[str, ...]] = (
    REASON_KILL_SWITCH,
    REASON_PRE_CUTOFF_RUNTIME,
    REASON_INVALID_INPUT,
    REASON_FRESH_INSTALL_NOTICE,
    REASON_SIDECAR_WAITS_FOR_DESKTOP,
    REASON_MARKER_AUTHORISES,
    REASON_LEGACY_REFUSAL_MIGRATED,
    REASON_CURRENT_REFUSAL,
    REASON_CONSENTED_NEEDS_NOTICE,
    REASON_CONSENTED,
)


class ProcessRole(Enum):
    """Which kind of process is asking for a consent decision."""

    INTERACTIVE_CLI = "interactive_cli"
    HEADLESS_CLI = "headless_cli"
    DESKTOP = "desktop"
    SIDECAR = "sidecar"


@dataclass(frozen=True)
class StoredConsent:
    """What the consent file says, stripped to the fields this table reads.

    ``consent=None`` means the field is absent (never prompted / no record).
    ``recorded_version`` is the version string stored next to the decision;
    it may be missing or garbage, in which case the record is not legacy.
    ``notice_revision_seen`` is the highest disclosure revision this install
    has been shown; ``None`` means never. Negative values are accepted and
    simply count as "absent" against :data:`DISCLOSURE_REVISION`.
    """

    consent: bool | None
    recorded_version: str | None
    notice_revision_seen: int | None


@dataclass(frozen=True)
class WriteBack:
    """What the caller must persist after acting on a :class:`Decision`.

    ``set_consent_true``: overwrite the stored ``consent`` with ``True``
    (legacy-refusal migration, row 4 — takes effect from the next run).
    ``stamp_version``: write the running rapid-mlx version next to the
    decision. ``mark_notice_seen``: record :data:`DISCLOSURE_REVISION` as
    the notice revision this install has seen.
    """

    set_consent_true: bool
    stamp_version: bool
    mark_notice_seen: bool


@dataclass(frozen=True)
class Decision:
    """The outcome for one run of one process.

    ``upload_now``: telemetry may be uploaded in this run. ``deliver_notice``
    is True. For rows 1 and 7 the caller MUST deliver the notice BEFORE the
    first upload; for row 4 ``upload_now`` is False and migration only takes
    effect from the next run.
    """

    upload_now: bool
    deliver_notice: bool
    write_back: WriteBack
    reason: str


_NO_WRITE_BACK: Final = WriteBack(
    set_consent_true=False, stamp_version=False, mark_notice_seen=False
)
_MARK_NOTICE_ONLY: Final = WriteBack(
    set_consent_true=False, stamp_version=False, mark_notice_seen=True
)
_MIGRATE_REFUSAL: Final = WriteBack(
    set_consent_true=True, stamp_version=True, mark_notice_seen=True
)

# Strict MAJOR.MINOR.PATCH with an optional pre-release suffix: ``rcN`` /
# ``aN`` / ``bN`` (attached) or ``.devN`` (dotted). Combined suffixes, other
# separators, whitespace, ``v`` prefixes and anything else fail to match.
_VERSION_PATTERN: Final = re.compile(
    r"(\d+)\.(\d+)\.(\d+)(?:(rc|a|b)(\d+)|\.dev(\d+))?"
)

# Pre-release ordering within one X.Y.Z: dev < a < b < rc < final.
_PHASE_DEV: Final = 0
_PHASE_ALPHA: Final = 1
_PHASE_BETA: Final = 2
_PHASE_RC: Final = 3
_PHASE_FINAL: Final = 4
_PHASE_RANK: Final[dict[str, int]] = {
    "a": _PHASE_ALPHA,
    "b": _PHASE_BETA,
    "rc": _PHASE_RC,
}

#: Sort key for a parsed release version. The last two elements order
#: pre-releases below the final release of the same X.Y.Z.
VersionKey: TypeAlias = tuple[int, int, int, int, int]


def _parse_release_version(version: str) -> VersionKey | None:
    """Parse ``MAJOR.MINOR.PATCH`` with an optional pre-release suffix.

    Returns ``None`` for anything unparseable. ``0.15.0rc1`` sorts strictly
    below ``0.15.0`` (pre-releases of the cutoff are legacy).
    """
    match = _VERSION_PATTERN.fullmatch(version)
    if match is None:
        return None
    major = int(match[1])
    minor = int(match[2])
    patch = int(match[3])
    phase = match[4]
    if phase is None:
        dev = match[6]
        if dev is None:
            return (major, minor, patch, _PHASE_FINAL, 0)
        return (major, minor, patch, _PHASE_DEV, int(dev))
    return (major, minor, patch, _PHASE_RANK[phase], int(match[5]))


def _parse_release_version_or_raise(version: str) -> VersionKey:
    """Like :func:`_parse_release_version` but raises on garbage.

    Used only for module constants that must parse; a corrupt
    :data:`DEFAULT_ON_CUTOFF` fails loudly at import instead of silently
    disabling the legacy check.
    """
    parsed = _parse_release_version(version)
    if parsed is None:
        raise ValueError(f"unparseable version constant: {version!r}")
    return parsed


_CUTOFF_KEY: Final[VersionKey] = _parse_release_version_or_raise(DEFAULT_ON_CUTOFF)


def _is_legacy_refusal(recorded_version: str | None, *, marker_present: bool) -> bool:
    """True for a refusal recorded by a version strictly below the cutoff.

    Missing or unparseable ``recorded_version`` → not legacy: fail toward
    respecting the refusal rather than guessing it predates the disclosure.
    A present marker means the refusal was recorded under the current
    disclosure, so it is never legacy regardless of the recorded version.
    """
    if marker_present or recorded_version is None:
        return False
    recorded_key = _parse_release_version(recorded_version)
    if recorded_key is None:
        return False
    return recorded_key < _CUTOFF_KEY


def _blocked(reason: str) -> Decision:
    """The kill-switch-shaped decision: no upload, no notice, no write-back."""
    return Decision(
        upload_now=False,
        deliver_notice=False,
        write_back=_NO_WRITE_BACK,
        reason=reason,
    )


def _well_formed(stored: StoredConsent) -> bool:
    """Runtime type check for :class:`StoredConsent` fields.

    ``decide`` promises never to raise, so the field types annotated on the
    dataclass are re-validated here against hostile input. ``bool`` is
    rejected for ``notice_revision_seen`` — a bool is not a disclosure
    revision.
    """
    consent_ok = stored.consent is None or isinstance(stored.consent, bool)
    version_ok = stored.recorded_version is None or isinstance(
        stored.recorded_version, str
    )
    revision = stored.notice_revision_seen
    revision_ok = revision is None or (
        isinstance(revision, int) and not isinstance(revision, bool)
    )
    return consent_ok and version_ok and revision_ok


def decide(
    stored: StoredConsent,
    role: ProcessRole,
    *,
    kill_switch_active: bool,
    running_version: str,
) -> Decision:
    """Resolve one consent decision. Pure; never raises.

    Implements exactly the table in the module docstring. ``running_version``
    is accepted for forward compatibility (and for callers that log it), but
    the decision does not depend on it except for one guard: a
    ``running_version`` that parses strictly below :data:`DEFAULT_ON_CUTOFF`
    means this module was backported into an older release that never showed
    the new disclosure — behave as if the kill switch were active and return
    the kill-switch decision with reason ``pre_cutoff_runtime``. An
    unparseable ``running_version`` does not block.

    Any non-conforming input (wrong types, ``None`` role, hostile objects)
    yields the kill-switch-shaped decision with reason ``invalid_input``.
    """
    if not (
        isinstance(stored, StoredConsent)
        and isinstance(role, ProcessRole)
        and isinstance(kill_switch_active, bool)
        and isinstance(running_version, str)
        and _well_formed(stored)
    ):
        return _blocked(REASON_INVALID_INPUT)
    if kill_switch_active:
        return _blocked(REASON_KILL_SWITCH)
    running_key = _parse_release_version(running_version)
    if running_key is not None and running_key < _CUTOFF_KEY:
        return _blocked(REASON_PRE_CUTOFF_RUNTIME)
    marker_present = (
        stored.notice_revision_seen is not None
        and stored.notice_revision_seen >= DISCLOSURE_REVISION
    )
    if stored.consent is None:
        if marker_present:
            # Row 3: the marker is the authorisation, sidecar included.
            return Decision(
                upload_now=True,
                deliver_notice=False,
                write_back=_NO_WRITE_BACK,
                reason=REASON_MARKER_AUTHORISES,
            )
        if role is ProcessRole.SIDECAR:
            # Row 2: the desktop owns disclosure and record creation.
            return _blocked(REASON_SIDECAR_WAITS_FOR_DESKTOP)
        # Row 1: upload is permitted, but the caller must deliver the
        # notice BEFORE the first upload (see docstring).
        return Decision(
            upload_now=True,
            deliver_notice=True,
            write_back=_MARK_NOTICE_ONLY,
            reason=REASON_FRESH_INSTALL_NOTICE,
        )
    if stored.consent is False:
        if not _is_legacy_refusal(
            stored.recorded_version, marker_present=marker_present
        ):
            # Row 6: a current refusal is never reversed automatically.
            return Decision(
                upload_now=False,
                deliver_notice=False,
                write_back=_NO_WRITE_BACK,
                reason=REASON_CURRENT_REFUSAL,
            )
        if role is ProcessRole.SIDECAR:
            # Row 5: the desktop owns the migration conversation.
            return _blocked(REASON_SIDECAR_WAITS_FOR_DESKTOP)
        # Row 4: the migrating run itself never uploads.
        return Decision(
            upload_now=False,
            deliver_notice=True,
            write_back=_MIGRATE_REFUSAL,
            reason=REASON_LEGACY_REFUSAL_MIGRATED,
        )
    # stored.consent is True
    if marker_present:
        # Row 9: explicit yes plus current marker — plain sailing.
        return Decision(
            upload_now=True,
            deliver_notice=False,
            write_back=_NO_WRITE_BACK,
            reason=REASON_CONSENTED,
        )
    if role is ProcessRole.SIDECAR:
        # Row 8: explicit true without the current marker is not enough for
        # a sidecar — same rule as every other process (notice before
        # collection); the sidecar just cannot deliver it.
        return _blocked(REASON_SIDECAR_WAITS_FOR_DESKTOP)
    # Row 7: upload permitted, but the caller must deliver the notice
    # BEFORE the first upload (see docstring).
    return Decision(
        upload_now=True,
        deliver_notice=True,
        write_back=_MARK_NOTICE_ONLY,
        reason=REASON_CONSENTED_NEEDS_NOTICE,
    )
