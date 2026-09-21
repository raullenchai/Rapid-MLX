# SPDX-License-Identifier: Apache-2.0
"""Contract pins for ``rapid_mlx.telemetry.common_props``.

The common-properties builder is the stamp on EVERY v2 event, so each
test here guards one rule and is written to turn red when the rule is
removed — verified by fault injection before merge, not assumed:

1. The happy path validates, and the emitted key set equals the key set
   ``events.json`` declares for the given arguments — the registry file
   is loaded HERE, so drift in either direction (a key the module
   invents, a key the registry adds) fails loudly.
2. Optional fields are omitted, never sentinels. The two cohort stamps
   are optional so a local state store that cannot answer (read-only
   HOME, locked/corrupt db) leaves the block valid: absence reads as
   "unknown" on the wire, never as 0 / first day — while a REAL 0
   ("has never served a model") must ship as 0.
3. Any invalid argument drops the build: the registry's strict
   "drop the WHOLE event" semantics must survive the assembly layer.
4. Raw chip brand strings never reach the block — only the closed
   ``chip_token`` vocabulary does.
5. ``read_platform_facts`` never raises, on this machine or under
   injected platform failures.
6. The finished block feeds ``envelope.build_batch_item`` unchanged.
"""

from __future__ import annotations

import json
import platform
import re
from dataclasses import replace
from itertools import product
from pathlib import Path
from uuid import uuid4

import pytest

from rapid_mlx.telemetry import chip, envelope, redact, registry
from rapid_mlx.telemetry.common_props import (
    PlatformFacts,
    build_common_props,
    read_platform_facts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_REGISTRY = REPO_ROOT / "rapid_mlx" / "telemetry" / "events.json"

INSTALL_ID = "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
SESSION_ID = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"

# Hermetic machine facts: every value is one the registry accepts, so a
# failure below is the builder's fault, not the machine's.
FACTS = PlatformFacts(
    os="darwin",
    os_version="25.3",
    arch="arm64",
    chip="m3-ultra",
    memory_gb=64,
    python_version="3.11",
)


def _registry() -> dict[str, object]:
    """events.json, read fresh — the spec the builder must satisfy."""
    loaded: dict[str, object] = json.loads(ENGINE_REGISTRY.read_text(encoding="utf-8"))
    return loaded


def _common_spec() -> dict[str, object]:
    """The declared common-prop specs, ``_``-prefixed doc keys stripped."""
    raw = _registry()["common_props"]
    assert isinstance(raw, dict)
    return {name: spec for name, spec in raw.items() if not name.startswith("_")}


def _build(
    *,
    surface: str = "cli",
    install_id: str = INSTALL_ID,
    session_id: str = SESSION_ID,
    app_version: str = "0.14.3",
    channel: str = "stable",
    nth_model_served: int | None = 3,
    days_since_first_run_bucket: str | None = "7-29",
    platform: PlatformFacts | None = FACTS,
) -> dict[str, object] | None:
    """``build_common_props`` with valid defaults, per-test overrides."""
    return build_common_props(
        surface=surface,
        install_id=install_id,
        session_id=session_id,
        app_version=app_version,
        channel=channel,
        nth_model_served=nth_model_served,
        days_since_first_run_bucket=days_since_first_run_bucket,
        platform=platform,
    )


def test_happy_path_builds_the_full_valid_block():
    """Every argument lands under its registry name, value unchanged."""
    block = _build()
    assert block == {
        "app_version": "0.14.3",
        "surface": "cli",
        "os": "darwin",
        "os_version": "25.3",
        "arch": "arm64",
        "chip": "m3-ultra",
        "memory_gb": 64,
        "python_version": "3.11",
        "install_id": INSTALL_ID,
        "session_id": SESSION_ID,
        "channel": "stable",
        "nth_model_served": 3,
        "days_since_first_run_bucket": "7-29",
    }
    # The registry itself accepts exactly this dict — a pin, not a
    # tautology: validate_common re-checks every key against events.json.
    assert block is not None
    assert registry.validate_common(block) == block


def test_key_set_equals_the_registry_declaration_when_every_field_is_present():
    """Drift (a): a key the module invents cannot ship. The expected set
    comes from events.json, not from the builder's own output."""
    block = _build()
    assert block is not None
    assert set(block) == set(_common_spec())


def test_key_set_equals_the_declared_required_set_when_python_version_is_none():
    """Drift (b): with the one optional field unreadable, the block
    carries EXACTLY the required keys — none missing, nothing added."""
    block = _build(platform=replace(FACTS, python_version=None))
    assert block is not None
    assert set(block) == set(_common_spec()) - {"python_version"}


@pytest.mark.parametrize(
    ("with_python", "with_nth", "with_days"),
    list(product((True, False), repeat=3)),
)
def test_every_optional_field_combination(with_python, with_nth, with_days):
    """All three OPTIONAL props — ``python_version`` (engine surfaces
    only) and the two cohort stamps (omitted when the local store cannot
    answer) — may be present or absent in any of the 8 combinations, and
    the key set is exactly the registry's required keys plus the
    provided optionals."""
    facts = replace(FACTS, python_version="3.11" if with_python else None)
    block = _build(
        platform=facts,
        nth_model_served=3 if with_nth else None,
        days_since_first_run_bucket="7-29" if with_days else None,
    )
    omitted = {
        name
        for name, present in (
            ("python_version", with_python),
            ("nth_model_served", with_nth),
            ("days_since_first_run_bucket", with_days),
        )
        if not present
    }
    assert block is not None
    assert set(block) == set(_common_spec()) - omitted


def test_zero_is_sent_as_zero_while_none_omits_the_key():
    """The two cohort meanings must not blur: ``nth_model_served=0`` is
    a legitimate value ("has never served a model") and
    ``days_since_first_run_bucket="0"`` a legitimate bucket (first day),
    so both must SHIP as given — while ``None`` (the local store could
    not answer) must OMIT the key, because absence is what analysts read
    as "unknown" on the wire."""
    served_none_yet = _build(nth_model_served=0, days_since_first_run_bucket="0")
    assert served_none_yet is not None
    assert served_none_yet["nth_model_served"] == 0
    assert served_none_yet["days_since_first_run_bucket"] == "0"

    store_unavailable = _build(
        nth_model_served=None,
        days_since_first_run_bucket=None,
    )
    assert store_unavailable is not None
    assert "nth_model_served" not in store_unavailable
    assert "days_since_first_run_bucket" not in store_unavailable


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("surface", "web"),  # not in the surface enum
        ("surface", ""),  # empty is not a member either
        ("channel", "beta"),  # channel is stable|rc only
        ("install_id", "not-a-uuid"),
        ("install_id", ""),
        ("session_id", ""),  # the empty session_id
        ("session_id", "6f1b1d3e-4a2b-4c9d-8e7f"),  # truncated UUID
        ("app_version", "garbage"),
        ("app_version", "0.14"),  # missing the third component
        ("app_version", "1.2.3.4"),  # four components
        ("app_version", "0.14.3-rc1"),  # rc spelled with a dash
        ("nth_model_served", -1),  # below the registry's min
        ("nth_model_served", 10001),  # above the registry's max
        ("days_since_first_run_bucket", "42"),  # not a bucket label
        ("days_since_first_run_bucket", "unknown"),  # no such member
    ],
)
def test_an_invalid_argument_drops_the_whole_block(field, bad_value):
    overrides: dict[str, object] = {field: bad_value}
    assert _build(**overrides) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "facts",
    [
        replace(FACTS, os="freebsd"),  # not an os enum member
        replace(FACTS, arch="i386"),  # not an arch enum member
        replace(FACTS, chip="Apple M3 Ultra"),  # raw brand, not the token
        replace(FACTS, memory_gb=-1),  # below min
        replace(FACTS, memory_gb=4097),  # above max
        replace(FACTS, os_version="25"),  # major only — pattern needs major.minor
        replace(FACTS, os_version="25.3.1"),  # patch must not ride along
        replace(FACTS, python_version="3.11.9"),  # ditto
    ],
)
def test_an_off_registry_platform_value_drops_the_whole_block(facts):
    assert _build(platform=facts) is None


def test_a_raw_chip_brand_never_reaches_the_block():
    """The registry-adjacent task's core privacy rule: the free-form
    sysctl brand goes in, only the closed token comes out."""
    facts = replace(FACTS, chip=chip.chip_token("Apple M3 Ultra"))
    assert facts.chip == "m3-ultra"  # the real converter, not a passthrough
    block = _build(platform=facts)
    assert block is not None
    assert block["chip"] == "m3-ultra"
    assert "Apple M3 Ultra" not in repr(block)


def test_read_platform_facts_maps_the_raw_brand_through_chip_token(monkeypatch):
    """The machine-derived half converts the chip itself: inject the raw
    sysctl brand and the snapshot must carry the enum token."""
    monkeypatch.setattr(redact, "_read_chip_brand", lambda: "Apple M3 Ultra")
    facts = read_platform_facts()
    assert facts.chip == "m3-ultra"
    block = _build(platform=facts)
    assert block is not None
    assert block["chip"] == "m3-ultra"
    assert "Apple M3 Ultra" not in repr(block)


def test_read_platform_facts_on_this_machine_builds_a_valid_block():
    """Real call, no monkeypatch: THIS machine's facts are registry-clean
    end to end — enums are members, versions match the registry patterns,
    and the resulting block validates."""
    facts = read_platform_facts()
    loaded = _registry()
    spec = _common_spec()
    assert facts.os in loaded["enums"]["os"]["values"]
    assert facts.arch in loaded["enums"]["arch"]["values"]
    assert facts.chip in loaded["enums"]["chip"]["values"]
    assert isinstance(facts.memory_gb, int) and not isinstance(facts.memory_gb, bool)
    assert facts.memory_gb >= 0
    assert facts.os_version is not None
    assert re.fullmatch(spec["os_version"]["pattern"], facts.os_version) is not None
    assert facts.python_version is None or (
        re.fullmatch(spec["python_version"]["pattern"], facts.python_version)
        is not None
    )
    block = build_common_props(
        surface="cli",
        install_id=str(uuid4()),
        session_id=str(uuid4()),
        app_version="0.14.3",
        channel="stable",
        nth_model_served=1,
        days_since_first_run_bucket="0",
    )
    assert block is not None
    assert block["os"] == facts.os
    assert block["arch"] == facts.arch
    assert block["chip"] == facts.chip
    assert block["memory_gb"] == facts.memory_gb


def test_read_platform_facts_survives_platform_functions_raising(monkeypatch):
    """A broken ``platform`` module degrades to the unknown snapshot —
    the call never raises and the composed block fails closed."""

    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("platform is broken")

    for name in ("system", "release", "machine", "python_version_tuple"):
        monkeypatch.setattr(platform, name, boom)
    facts = read_platform_facts()
    assert facts == PlatformFacts(
        os="other",
        os_version=None,
        arch="other",
        chip="other",
        memory_gb=0,
        python_version=None,
    )
    assert _build(platform=facts) is None


def test_read_platform_facts_survives_sysctl_raising(monkeypatch):
    """An exploding chip read inside platform_info degrades the whole
    snapshot to unknown values; nothing propagates."""

    def boom() -> str:
        raise RuntimeError("sysctl exploded")

    monkeypatch.setattr(redact, "_read_chip_brand", boom)
    facts = read_platform_facts()
    assert facts == PlatformFacts(
        os="other",
        os_version=None,
        arch="other",
        chip="other",
        memory_gb=0,
        python_version=None,
    )
    assert _build(platform=facts) is None


def test_read_platform_facts_survives_a_registry_that_cannot_load(monkeypatch):
    """A wheel missing events.json must degrade to the same unknown
    snapshot as any other unreadable value — the contract is 'never
    raises', exactly like registry.validate* which swallows the same
    failure."""

    def boom() -> dict[str, object]:
        raise FileNotFoundError("events.json missing from the wheel")

    monkeypatch.setattr(registry, "load_registry", boom)
    facts = read_platform_facts()
    assert facts == PlatformFacts(
        os="other",
        os_version=None,
        arch="other",
        chip="other",
        memory_gb=0,
        python_version=None,
    )
    assert _build(platform=facts) is None


def test_build_common_props_survives_a_registry_that_cannot_load(monkeypatch):
    """The builder itself never raises either. ``platform=None`` makes
    it read the facts itself — without the guard, the FileNotFoundError
    would propagate straight out of the builder."""

    def boom() -> dict[str, object]:
        raise FileNotFoundError("events.json missing from the wheel")

    monkeypatch.setattr(registry, "load_registry", boom)
    assert (
        build_common_props(
            surface="cli",
            install_id=INSTALL_ID,
            session_id=SESSION_ID,
            app_version="0.14.3",
            channel="stable",
            nth_model_served=3,
            days_since_first_run_bucket="7-29",
        )
        is None
    )


@pytest.mark.parametrize("hostile", ["darwin", object()])
def test_a_non_platform_facts_argument_drops_the_build(hostile):
    """The parameter is typed, but a caller ignoring types must get the
    fail-closed answer, not an AttributeError: ANY invalid argument
    makes the whole build None."""
    assert _build(platform=hostile) is None  # type: ignore[arg-type]


def test_read_platform_facts_narrows_malformed_info_to_unknown(monkeypatch):
    """Every narrowing helper holds: off-enum strings, wrong types, a
    bool masquerading as the memory int — all collapse to the
    registry's own unknown values, and the build drops."""

    def malformed() -> dict[str, object]:
        return {
            "os": "tritan",  # off-enum
            "os_version": 25.5,  # wrong type
            "arch": None,  # missing
            "chip": 7,  # wrong type
            "memory_gb": True,  # bool is not an int here
            "python_version": ["3"],  # wrong type
        }

    monkeypatch.setattr(redact, "platform_info", malformed)
    facts = read_platform_facts()
    assert facts == PlatformFacts(
        os="other",
        os_version=None,
        arch="other",
        chip="other",
        memory_gb=0,
        python_version=None,
    )
    assert _build(platform=facts) is None


def test_read_platform_facts_rejects_pattern_violating_version_strings(monkeypatch):
    """A version-shaped string that violates the registry's own pattern
    narrows to None: a Windows-style bare-major ``platform.release()``
    ("11", no minor) and a bare-major python version must not ride the
    wire — the whole block then drops for lack of a required value."""

    def windows_like() -> dict[str, object]:
        return {
            "os": "windows",
            "os_version": "11",
            "arch": "x86_64",
            "chip": "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz",
            "memory_gb": 32,
            "python_version": "3",
        }

    monkeypatch.setattr(redact, "platform_info", windows_like)
    facts = read_platform_facts()
    assert facts == PlatformFacts(
        os="windows",
        os_version=None,
        arch="x86_64",
        chip="intel",
        memory_gb=32,
        python_version=None,
    )
    assert _build(platform=facts) is None


def test_the_block_feeds_envelope_build_batch_item():
    """Integration: the assembled block is exactly what the envelope
    builder consumes for a real event."""
    block = _build()
    assert block is not None
    item = envelope.build_batch_item("app_opened", {}, block)
    assert item is not None
    assert item["event"] == "app_opened"
    assert item["distinct_id"] == INSTALL_ID
    properties = item["properties"]
    assert isinstance(properties, dict)
    assert properties["surface"] == "cli"
    assert properties["chip"] == "m3-ultra"
    assert properties["nth_model_served"] == 3
    assert properties["$geoip_disable"] is True
    assert properties["$process_person_profile"] is False
