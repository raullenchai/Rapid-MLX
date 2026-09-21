# SPDX-License-Identifier: Apache-2.0
"""The telemetry v2 common-properties block — pure assembly, fail closed.

Every telemetry v2 event is stamped with ONE identical "common
properties" dict, validated against ``events.json`` -> ``common_props``
by :func:`rapid_mlx.telemetry.registry.validate_common`. The registry
is strict — one unknown key, one wrong type, one out-of-enum value and
the WHOLE event is dropped — so this module assembles the block only
from closed values and hands the finished dict through
``validate_common``: :func:`build_common_props` returns ``None`` when
anything is invalid, never a partially valid block.

The block is identical for every event a process emits, so callers may
build it once per session and reuse the result. For that reason every
install- and process-specific value arrives as an ARGUMENT:
``build_common_props`` is pure — it opens no file and reads no store,
and in particular does NOT call ``store.note_model_served()`` (that
function MUTATES the install's cohort state, and stamping event N must
not advance the counter event N+1 reads). Callers source
``nth_model_served`` and ``days_since_first_run_bucket`` from the
telemetry store beforehand and pass the values in.

The two cohort fields are OPTIONAL in the registry: when the local state
store cannot answer (a read-only HOME, a locked or corrupt database —
``store.days_since_first_run_bucket()`` returns ``None`` by design
there), the argument is ``None``, the key is omitted, and the block
still ships. Absence on the wire means "the local store could not
answer"; analysts must read it as unknown, never as 0 or first day.
Callers must therefore pass ``None`` — not 0 — when the store could not
answer: ``store.note_model_served()`` returns 0 only on FAILURE (a
successful note always returns >= 1), so a failure-0 passed through
would ship a broken store as a real first-time user; a wire 0 is valid
only when a store genuinely answered zero.

Privacy rule (design sec 1.5): the block carries closed enums, buckets,
pattern-capped version strings and UUIDs only. No hostname, no
username, no path, no locale, no timezone, no country. The
machine-derived half (:class:`PlatformFacts`) is held to the same bar:
the raw chip brand string never passes through — it is mapped onto the
registry's closed ``chip`` vocabulary by
:func:`rapid_mlx.telemetry.chip.chip_token`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapid_mlx.telemetry import chip, redact, registry

__all__ = ["PlatformFacts", "build_common_props", "read_platform_facts"]


@dataclass(frozen=True)
class PlatformFacts:
    """Machine-derived common-prop values, each already registry-closed.

    ``os`` / ``arch`` / ``chip`` are always members of their
    ``events.json`` enums — anything unreadable or off-list collapses to
    the enum's ``other`` fallback. ``memory_gb`` is the whole-GB bucket
    (:func:`rapid_mlx.telemetry.redact.bucket_memory_gb`), whose ``0``
    is itself the documented "unknown". ``os_version`` and
    ``python_version`` carry a major.minor string only when it satisfies
    the registry's own pattern for the slot; ``None`` means "could not
    be read", which the slot's required/optional rule then resolves: an
    unreadable ``os_version`` (required) drops the whole block, an
    unreadable ``python_version`` (optional) merely omits the key.
    """

    os: str
    os_version: str | None
    arch: str
    chip: str
    memory_gb: int
    python_version: str | None


def _enum_member(raw: object, enum: str, fallback: str) -> str:
    """Narrow ``raw`` to one member of the closed ``enum``, else ``fallback``."""
    if not isinstance(raw, str):
        return fallback
    values: list[object] = registry.load_registry()["enums"][enum]["values"]
    return raw if raw in frozenset(values) else fallback


def _pattern_conforming(raw: object, slot: str) -> str | None:
    """Narrow ``raw`` to the registry's own pattern for common-prop ``slot``.

    ``slot`` names a ``version``-kind key of ``events.json`` ->
    ``common_props`` (``os_version`` / ``python_version``). Reading the
    pattern from the registry instead of restating it here is what makes
    drift between the two impossible.
    """
    if not isinstance(raw, str):
        return None
    pattern: str = registry.load_registry()["common_props"][slot]["pattern"]
    if re.fullmatch(pattern, raw) is None:
        return None
    return raw


def _memory_gb(raw: object) -> int:
    """Narrow ``raw`` to an int for the ``memory_gb`` slot, else ``0``.

    ``0`` is not an invented sentinel: it is what
    :func:`rapid_mlx.telemetry.redact.bucket_memory_gb` itself reports
    for unknown memory, and the registry's ``min`` is ``0``. A readable
    but out-of-range int passes through UNTOUCHED so
    ``validate_common`` — not this module — decides its fate: clamping
    an 8192-GB host to 0 would understate a fact we actually read.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    return raw


_UNKNOWN_FACTS = PlatformFacts(
    os="other",
    os_version=None,
    arch="other",
    chip="other",
    memory_gb=0,
    python_version=None,
)


def read_platform_facts() -> PlatformFacts:
    """Read the machine-derived half of the block. Never raises.

    One call to :func:`rapid_mlx.telemetry.redact.platform_info` — the
    same source v1 reported from, sysctl chip brand included — then
    every field is narrowed to what the registry accepts: the chip
    through the closed :func:`rapid_mlx.telemetry.chip.chip_token`
    mapping, the enums to their fallback member, the version strings to
    the registry's own patterns. ANY failure along the way — a raising
    ``platform`` function, an exploding sysctl read, a registry that
    cannot load (a wheel missing ``events.json``) — degrades the whole
    snapshot to the unknown values instead of propagating; a build from
    such a snapshot is then dropped by the registry, which is the
    fail-closed outcome, not a crash.
    """
    try:
        info: dict[object, object] = redact.platform_info()
        raw_chip: object = info.get("chip")
        return PlatformFacts(
            os=_enum_member(info.get("os"), "os", "other"),
            os_version=_pattern_conforming(info.get("os_version"), "os_version"),
            arch=_enum_member(info.get("arch"), "arch", "other"),
            chip=chip.chip_token(raw_chip if isinstance(raw_chip, str) else None),
            memory_gb=_memory_gb(info.get("memory_gb")),
            python_version=_pattern_conforming(
                info.get("python_version"), "python_version"
            ),
        )
    except Exception:
        # ``registry.validate*`` swallow the same failure; the narrowing
        # helpers call ``load_registry`` outside them, so this guard is
        # what keeps the never-raises contract. KeyboardInterrupt and
        # SystemExit are BaseException and still propagate.
        return _UNKNOWN_FACTS


def build_common_props(
    *,
    surface: str,
    install_id: str,
    session_id: str,
    app_version: str,
    channel: str,
    nth_model_served: int | None,
    days_since_first_run_bucket: str | None,
    platform: PlatformFacts | None = None,
) -> dict[str, object] | None:
    """Assemble the sec 1.2 common-properties block, or ``None``.

    Pure: every install-/process-specific value is an argument; no file
    or store is touched here. ``platform`` defaults to a fresh
    :func:`read_platform_facts` snapshot; a caller that emits many
    events reads the facts (or builds the block) once and passes them
    in.

    Every value is either registry-accepted or omitted, and the
    assembled block then goes through
    :func:`rapid_mlx.telemetry.registry.validate_common`, so ANY invalid
    argument — an off-enum surface, a malformed UUID, a version with
    garbage in it — makes the WHOLE build ``None`` instead of a
    partially valid block.

    The cohort fields are OPTIONAL and are omitted when ``None``: the
    registry sanctions absence exactly for the case where the local
    state store cannot answer (read-only HOME, locked or corrupt
    database — ``store.days_since_first_run_bucket()`` returns ``None``
    by design there). Callers MUST pass ``None`` — not 0 — when the
    store could not answer: ``store.note_model_served()`` returns the
    count INCLUDING the model just noted (success is always >= 1) and 0
    only on FAILURE, so a failure-0 passed through would ship a broken
    store as a real first-time user; a wire 0 is valid only when a
    store genuinely answered zero. Read real values from the store
    (``store.note_model_served`` / ``store.days_since_first_run_bucket``)
    before building; ``note_model_served`` mutates state, so it belongs
    to the emit path, never inside this builder.
    """
    facts = read_platform_facts() if platform is None else platform
    if not isinstance(facts, PlatformFacts):
        # The parameter is typed, but a caller ignoring types (a str, a
        # bare object) is an invalid argument like any other: fail
        # closed here rather than raise AttributeError on the attribute
        # access below.
        return None
    block: dict[str, object] = {
        "app_version": app_version,
        "surface": surface,
        "os": facts.os,
        "arch": facts.arch,
        "chip": facts.chip,
        "memory_gb": facts.memory_gb,
        "install_id": install_id,
        "session_id": session_id,
        "channel": channel,
    }
    # Version strings and cohort stamps are omitted when unreadable or
    # unknown — never replaced with a sentinel the registry does not
    # define. python_version is optional ("engine surfaces only"); the
    # cohort stamps are optional so a local store that cannot answer
    # leaves the block valid, their absence meaning "unknown" on the
    # wire, never 0 / first day.
    if facts.os_version is not None:
        block["os_version"] = facts.os_version
    if facts.python_version is not None:
        block["python_version"] = facts.python_version
    if nth_model_served is not None:
        block["nth_model_served"] = nth_model_served
    if days_since_first_run_bucket is not None:
        block["days_since_first_run_bucket"] = days_since_first_run_bucket
    return registry.validate_common(block)
