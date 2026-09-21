# SPDX-License-Identifier: Apache-2.0
"""Proof for the default-on consent decision table (``consent_decision``).

The cross-product test enumerates EVERY combination of

    consent {None, False, True}
  x marker {None, older revision, current, newer}
  x role (4)
  x kill switch {True, False}
  x recorded_version {None, "garbage", "0.0.0", "0.14.3", "0.14.9rc1",
                      "0.14.10", "0.15.0rc1", "0.15.0.dev3", "0.15.0a1",
                      "0.15.0b2", "0.15.0", "0.16.2"}

(3 x 4 x 4 x 2 x 12 = 1152 cases) and checks each against an oracle written
here as explicit row matching. The oracle shares NO logic with the
implementation: reason strings, the legacy rule and the marker threshold are
restated literally below, so any drift in ``consent_decision.py`` turns red.

The venv running these tests has rapid-mlx installed editable from a
DIFFERENT worktree, so first assert that the module under test really is the
file in this worktree.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

import rapid_mlx.telemetry.consent_decision as consent_decision_module
from rapid_mlx.telemetry.consent_decision import (
    DEFAULT_ON_CUTOFF,
    DISCLOSURE_REVISION,
    Decision,
    ProcessRole,
    StoredConsent,
    WriteBack,
    decide,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Oracle — independent restatement of the contract. No imports of
# implementation helpers; every reason and rule is a literal here.
# ---------------------------------------------------------------------------

_ALL_OFF = WriteBack(False, False, False)
_MARKER_ONLY = WriteBack(False, False, True)
_MIGRATE = WriteBack(True, True, True)

#: The closed reason set, restated literally (invariant f).
_ORACLE_REASONS = frozenset(
    {
        "kill_switch",
        "pre_cutoff_runtime",
        "invalid_input",
        "fresh_install_notice",
        "sidecar_waits_for_desktop",
        "marker_authorises",
        "legacy_refusal_migrated",
        "current_refusal",
        "consented_needs_notice",
        "consented",
    }
)

#: Independent legacy determination for the fixed recorded_version column
#: of the cross product. LEGACY = consent is False AND marker absent AND
#: the recorded version parses AND its (major, minor, patch) TRIPLE is
#: strictly below (0, 15, 0); pre-release suffixes are ignored for the
#: comparison. Missing/unparseable is NEVER legacy; the triple-equal
#: pre-releases of the cutoff (rc1/dev/a/b — they already carry the new
#: disclosure code) are NEVER legacy; equal (0.15.0) is NOT strictly lower;
#: "0.0.0" is rapid_mlx/__init__.py's version-unknown sentinel and is
#: NEVER legacy (a source checkout may run any real version).
_ORACLE_LEGACY: dict[str | None, bool] = {
    None: False,
    "garbage": False,
    "0.0.0": False,
    "0.14.3": True,
    "0.14.9rc1": True,
    "0.14.10": True,
    "0.15.0rc1": False,
    "0.15.0.dev3": False,
    "0.15.0a1": False,
    "0.15.0b2": False,
    "0.15.0": False,
    "0.16.2": False,
}

#: Marker threshold restated literally: the pinned DISCLOSURE_REVISION (1).
_ORACLE_MARKER_THRESHOLD = 1

_ROLES = list(ProcessRole)
_RECORDED_VERSIONS = [
    None,
    "garbage",
    "0.0.0",
    "0.14.3",
    "0.14.9rc1",
    "0.14.10",
    "0.15.0rc1",
    "0.15.0.dev3",
    "0.15.0a1",
    "0.15.0b2",
    "0.15.0",
    "0.16.2",
]
_REVISIONS = [None, 0, 1, 2]  # absent / older / current / newer
_CONSENTS = [None, False, True]
_KILL_VALUES = [True, False]


def _oracle(
    consent: bool | None,
    notice_revision_seen: int | None,
    role: ProcessRole,
    kill_switch_active: bool,
    recorded_version: str | None,
) -> Decision:
    """Explicit row matching of the decision table. Nothing shared."""
    if kill_switch_active:
        # Row 0, checked first for every row.
        return Decision(False, False, _ALL_OFF, "kill_switch")
    marker_present = (
        notice_revision_seen is not None
        and notice_revision_seen >= _ORACLE_MARKER_THRESHOLD
    )
    if consent is None:
        if marker_present:
            # Row 3: the marker is the authorisation, sidecar included.
            return Decision(True, False, _ALL_OFF, "marker_authorises")
        if role is ProcessRole.SIDECAR:
            # Row 2.
            return Decision(False, False, _ALL_OFF, "sidecar_waits_for_desktop")
        # Row 1.
        return Decision(True, True, _MARKER_ONLY, "fresh_install_notice")
    if consent is False:
        legacy = (not marker_present) and _ORACLE_LEGACY[recorded_version]
        if not legacy:
            # Row 6: a current refusal, any role.
            return Decision(False, False, _ALL_OFF, "current_refusal")
        if role is ProcessRole.SIDECAR:
            # Row 5.
            return Decision(False, False, _ALL_OFF, "sidecar_waits_for_desktop")
        # Row 4: the migrating run itself never uploads.
        return Decision(False, True, _MIGRATE, "legacy_refusal_migrated")
    # consent is True
    if marker_present:
        # Row 9.
        return Decision(True, False, _ALL_OFF, "consented")
    if role is ProcessRole.SIDECAR:
        # Row 8: explicit true without the marker is not enough for a sidecar.
        return Decision(False, False, _ALL_OFF, "sidecar_waits_for_desktop")
    # Row 7.
    return Decision(True, True, _MARKER_ONLY, "consented_needs_notice")


def _all_cases():
    """Every case of the cross product as a tuple of inputs."""
    for consent in _CONSENTS:
        for revision in _REVISIONS:
            for role in _ROLES:
                for recorded in _RECORDED_VERSIONS:
                    for kill in _KILL_VALUES:
                        yield consent, revision, role, recorded, kill


def _decide_case(consent, revision, role, recorded, kill, running="0.15.0"):
    stored = StoredConsent(
        consent=consent,
        recorded_version=recorded,
        notice_revision_seen=revision,
    )
    return decide(stored, role, kill_switch_active=kill, running_version=running)


# ---------------------------------------------------------------------------
# Environment and contract pins
# ---------------------------------------------------------------------------


def test_module_under_test_is_this_worktrees_file():
    module_file = Path(consent_decision_module.__file__).resolve()
    assert module_file == _REPO_ROOT / "rapid_mlx" / "telemetry" / (
        "consent_decision.py"
    )


def test_contract_constants_are_pinned():
    # The oracle hard-codes the cutoff ("0.15.0") and marker threshold (1);
    # if either constant moves, this pin forces an oracle review too.
    assert DEFAULT_ON_CUTOFF == "0.15.0"
    assert DISCLOSURE_REVISION == 1


# ---------------------------------------------------------------------------
# Full cross product vs the independent oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kill_switch_active", _KILL_VALUES, ids=["kill", "no-kill"])
@pytest.mark.parametrize("recorded_version", _RECORDED_VERSIONS)
@pytest.mark.parametrize("role", _ROLES, ids=lambda r: r.name)
@pytest.mark.parametrize(
    "notice_revision_seen", _REVISIONS, ids=["absent", "older", "current", "newer"]
)
@pytest.mark.parametrize("consent", _CONSENTS, ids=["absent", "refused", "granted"])
def test_full_cross_product_matches_oracle(
    consent, notice_revision_seen, role, recorded_version, kill_switch_active
):
    got = _decide_case(
        consent, notice_revision_seen, role, recorded_version, kill_switch_active
    )
    expected = _oracle(
        consent, notice_revision_seen, role, kill_switch_active, recorded_version
    )
    assert got == expected, (
        consent,
        notice_revision_seen,
        role.name,
        recorded_version,
        kill_switch_active,
    )


# ---------------------------------------------------------------------------
# Invariants over the whole cross product
# ---------------------------------------------------------------------------


def test_invariant_a_upload_implies_marker_present_or_notice_delivered():
    # (a) Nothing is uploaded before the current disclosure revision has
    # been delivered on this install.
    for case in _all_cases():
        consent, revision, role, recorded, kill = case
        decision = _decide_case(consent, revision, role, recorded, kill)
        if decision.upload_now:
            marker_present = revision is not None and revision >= 1
            assert marker_present or decision.deliver_notice, case


def test_invariant_a_sidecar_upload_requires_marker():
    for case in _all_cases():
        consent, revision, role, recorded, kill = case
        decision = _decide_case(consent, revision, role, recorded, kill)
        if decision.upload_now and role is ProcessRole.SIDECAR:
            assert revision is not None and revision >= 1, case


def test_invariant_b_sidecar_never_writes_and_never_discloses():
    for case in _all_cases():
        consent, revision, role, recorded, kill = case
        decision = _decide_case(consent, revision, role, recorded, kill)
        if role is ProcessRole.SIDECAR:
            assert not decision.deliver_notice, case
            assert decision.write_back == _ALL_OFF, case


def test_invariant_c_unparseable_or_at_cutoff_refusal_never_migrated():
    # (c) A refusal recorded at or after the cutoff — including its
    # pre-releases — or with an unknown/unparseable recorded version is
    # never reversed automatically.
    never_migrate = {
        None,
        "garbage",
        "0.0.0",
        "0.15.0rc1",
        "0.15.0.dev3",
        "0.15.0a1",
        "0.15.0b2",
        "0.15.0",
        "0.16.2",
    }
    for case in _all_cases():
        consent, revision, role, recorded, kill = case
        decision = _decide_case(consent, revision, role, recorded, kill)
        if consent is False and recorded in never_migrate:
            assert decision.write_back == _ALL_OFF, case


def test_invariant_d_kill_switch_forces_all_false():
    for case in _all_cases():
        consent, revision, role, recorded, _ = case
        decision = _decide_case(consent, revision, role, recorded, kill=True)
        assert decision.upload_now is False, case
        assert decision.deliver_notice is False, case
        assert decision.write_back == _ALL_OFF, case


def test_invariant_e_set_consent_true_implies_not_uploading():
    # (e) The migrating run itself never uploads.
    for case in _all_cases():
        consent, revision, role, recorded, kill = case
        decision = _decide_case(consent, revision, role, recorded, kill)
        if decision.write_back.set_consent_true:
            assert decision.upload_now is False, case


def test_invariant_f_reason_is_always_in_the_closed_set():
    for case in _all_cases():
        consent, revision, role, recorded, kill = case
        decision = _decide_case(consent, revision, role, recorded, kill)
        assert decision.reason in _ORACLE_REASONS, case


# ---------------------------------------------------------------------------
# Named row tests (documentation value; all also covered by the product)
# ---------------------------------------------------------------------------


def test_row1_fresh_install_notice_writes_marker_only():
    decision = _decide_case(None, None, ProcessRole.INTERACTIVE_CLI, None, kill=False)
    assert decision == Decision(True, True, _MARKER_ONLY, "fresh_install_notice")


def test_row2_absent_consent_sidecar_waits_for_desktop():
    decision = _decide_case(None, None, ProcessRole.SIDECAR, None, kill=False)
    assert decision == Decision(False, False, _ALL_OFF, "sidecar_waits_for_desktop")


def test_row3_marker_authorises_sidecar_included():
    decision = _decide_case(None, 1, ProcessRole.SIDECAR, None, kill=False)
    assert decision == Decision(True, False, _ALL_OFF, "marker_authorises")


def test_row4_legacy_refusal_migrated_but_never_uploads():
    decision = _decide_case(False, None, ProcessRole.HEADLESS_CLI, "0.14.3", kill=False)
    assert decision == Decision(False, True, _MIGRATE, "legacy_refusal_migrated")


def test_row5_legacy_refusal_sidecar_waits_for_desktop():
    decision = _decide_case(False, None, ProcessRole.SIDECAR, "0.14.3", kill=False)
    assert decision == Decision(False, False, _ALL_OFF, "sidecar_waits_for_desktop")


@pytest.mark.parametrize(
    "recorded", [None, "garbage", "0.0.0", "0.15.0rc1", "0.15.0", "0.16.2"]
)
def test_row6_current_refusal_is_never_auto_migrated(recorded):
    for role in _ROLES:
        decision = _decide_case(False, None, role, recorded, kill=False)
        assert decision == Decision(False, False, _ALL_OFF, "current_refusal"), (
            role,
            recorded,
        )


def test_row7_consented_without_marker_needs_notice():
    decision = _decide_case(True, None, ProcessRole.DESKTOP, "0.16.2", kill=False)
    assert decision == Decision(True, True, _MARKER_ONLY, "consented_needs_notice")


def test_row8_sidecar_explicit_true_without_marker_waits():
    decision = _decide_case(True, None, ProcessRole.SIDECAR, None, kill=False)
    assert decision == Decision(False, False, _ALL_OFF, "sidecar_waits_for_desktop")


def test_row9_consented_with_marker_uploads_everywhere():
    for role in _ROLES:
        decision = _decide_case(True, 1, role, "0.14.3", kill=False)
        assert decision == Decision(True, False, _ALL_OFF, "consented"), role


def test_marker_older_revision_counts_as_absent():
    decision = _decide_case(None, 0, ProcessRole.INTERACTIVE_CLI, None, kill=False)
    assert decision.reason == "fresh_install_notice"


def test_marker_newer_revision_counts_as_present():
    decision = _decide_case(None, 2, ProcessRole.SIDECAR, None, kill=False)
    assert decision.reason == "marker_authorises"


def test_kill_switch_beats_every_row_including_marker_present():
    decision = _decide_case(None, 2, ProcessRole.DESKTOP, None, kill=True)
    assert decision == Decision(False, False, _ALL_OFF, "kill_switch")


# ---------------------------------------------------------------------------
# Legacy rule: release-triple comparison (P1-1 / P1-2 regressions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recorded", ["0.15.0rc1", "0.15.0.dev3", "0.15.0a1", "0.15.0b2"]
)
def test_prerelease_of_cutoff_refusal_is_a_current_refusal(recorded):
    # Pre-releases of the cutoff already carry the new disclosure code and
    # are published to real users, so a refusal recorded there (e.g. by
    # running `rapid-mlx telemetry off` first thing in an rc) is NEVER
    # auto-migrated — for any role.
    for role in _ROLES:
        decision = _decide_case(False, None, role, recorded, kill=False)
        assert decision == Decision(False, False, _ALL_OFF, "current_refusal"), (
            role,
            recorded,
        )


@pytest.mark.parametrize("recorded", ["0.14.3", "0.14.9rc1", "0.14.10"])
def test_refusal_below_the_cutoff_triple_is_still_legacy(recorded):
    # Guard against over-correction: strictly-below-triple refusals (with
    # or without a pre-release suffix) are still legacy and migrate.
    for role in _ROLES:
        if role is ProcessRole.SIDECAR:
            continue
        decision = _decide_case(False, None, role, recorded, kill=False)
        assert decision == Decision(False, True, _MIGRATE, "legacy_refusal_migrated"), (
            role,
            recorded,
        )


def test_version_unknown_sentinel_refusal_is_respected():
    # "0.0.0" is rapid_mlx/__init__.py's __version__ fallback when package
    # metadata is missing (editable / source installs). It is a sentinel
    # for "version unknown", not a real release below the cutoff, so a
    # refusal recorded beside it is respected for EVERY role — never
    # migrated, not even by the official build sharing the consent file.
    for role in _ROLES:
        decision = _decide_case(False, None, role, "0.0.0", kill=False)
        assert decision == Decision(False, False, _ALL_OFF, "current_refusal"), role


# ---------------------------------------------------------------------------
# running_version: pre-cutoff runtime guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("running", ["0.14.9", "0.14.9rc1", "0.14.3", "0.0.1"])
def test_pre_cutoff_runtime_behaves_like_kill_switch(running):
    decision = _decide_case(
        None, None, ProcessRole.INTERACTIVE_CLI, None, kill=False, running=running
    )
    assert decision == Decision(False, False, _ALL_OFF, "pre_cutoff_runtime"), running


@pytest.mark.parametrize(
    "running", ["0.15.0", "0.15.0rc1", "0.15.0.dev3", "0.16.2", "nonsense", "0.0.0"]
)
def test_cutoff_or_unparseable_runtime_does_not_block(running):
    # Triple-equal pre-releases of the cutoff DO carry the new disclosure
    # code and run the table normally; the 0.0.0 sentinel (unknown
    # version) does not block either.
    decision = _decide_case(
        None, None, ProcessRole.INTERACTIVE_CLI, None, kill=False, running=running
    )
    assert decision.upload_now is True, running
    assert decision.deliver_notice is True, running


def test_kill_switch_precedes_pre_cutoff_runtime():
    decision = _decide_case(
        None, None, ProcessRole.INTERACTIVE_CLI, None, kill=True, running="0.14.9"
    )
    assert decision.reason == "kill_switch"


# ---------------------------------------------------------------------------
# Version parser edge cases
# ---------------------------------------------------------------------------


def test_version_parser_orders_pre_releases_below_the_release():
    parse = consent_decision_module._parse_release_version
    dev = parse("0.15.0.dev1")
    alpha = parse("0.15.0a1")
    beta = parse("0.15.0b2")
    rc1 = parse("0.15.0rc1")
    rc2 = parse("0.15.0rc10")
    final = parse("0.15.0")
    older_release = parse("0.14.3")
    newer_release = parse("0.16.2")
    assert dev is not None
    assert alpha is not None
    assert beta is not None
    assert rc1 is not None
    assert rc2 is not None
    assert final is not None
    assert older_release is not None
    assert newer_release is not None
    assert dev < alpha < beta < rc1 < rc2 < final
    assert older_release < dev  # 0.14.3 predates any 0.15.0 pre-release
    assert final < newer_release


def test_version_parser_accepts_canonical_shapes():
    parse = consent_decision_module._parse_release_version
    # Same X.Y.Z parses to the same key regardless of suffix kind present.
    assert parse("0.15.0") == parse("0.15.0")
    assert parse("10.20.30") == parse("10.20.30")
    assert parse("0.15.0a1") == parse("0.15.0a1")


# Every spelling of the version-unknown release triple (0, 0, 0). The
# exact string rapid_mlx/__init__.py stamps is "0.0.0", but the sentinel
# domain is the triple, not the string — nothing stops a hand-edited or
# hostile consent record from holding an equivalent spelling, with or
# without a pre-release suffix.
_ALL_ZERO_TRIPLE_SPELLINGS = [
    "0.0.0",
    "00.00.00",
    "0.0.00",
    "000.000.000",
    "0.00.0",
    "0.0.0rc1",
    "0.0.0.dev1",
    "0.0.0a1",
    "0.0.0b1",
]


def test_version_parser_rejects_the_version_unknown_sentinel():
    # The release triple (0, 0, 0) never occurs as a real release (earliest
    # tag v0.1.0); it is the version-unknown sentinel domain that
    # rapid_mlx/__init__.py stamps when package metadata is missing.
    # EVERY spelling — leading zeros, suffixed or not — must take the
    # unparseable branch so a refusal recorded beside an unknown version is
    # never migrated.
    parse = consent_decision_module._parse_release_version
    for spelling in _ALL_ZERO_TRIPLE_SPELLINGS:
        assert parse(spelling) is None, spelling


def test_zero_triple_refusal_spellings_are_respected():
    # Decision level: a refusal whose recorded version is any (0, 0, 0)
    # spelling is unparseable → NOT legacy → current refusal, for every
    # role.
    for spelling in _ALL_ZERO_TRIPLE_SPELLINGS:
        for role in _ROLES:
            decision = _decide_case(False, None, role, spelling, kill=False)
            assert decision == Decision(False, False, _ALL_OFF, "current_refusal"), (
                role,
                spelling,
            )


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "0",
        "0.15",
        "v0.15.0",
        "0.15.0-rc1",
        "0.15.0RC1",
        "0.15.0.dev",
        "0.15.0rc",
        "0.15.0a",
        "0.15.0.a1",
        "0.15.0.dev1rc1",
        "0.15.0rc1.dev1",
        "1.2.3.4",
        " 0.15.0",
        "0.15.0 ",
        "garbage",
        "0.15.0+build1",
        # Non-ASCII digits: \\d would match these and int() would convert
        # them, silently turning an unknown-version refusal into a legacy
        # one. Fullwidth, Arabic-Indic and Devanagari must all fail to
        # parse.
        "０.１４.３",
        "٠.١٤.٣",
        "०.१४.३",
        "0.１４.3",
        "０.14.3",
    ],
)
def test_version_parser_rejects_garbage(garbage):
    parse = consent_decision_module._parse_release_version
    assert parse(garbage) is None, garbage


def test_strict_parse_raises_on_garbage_for_module_constants():
    strict = consent_decision_module._parse_release_version_or_raise
    assert strict("0.15.0") == consent_decision_module._CUTOFF_KEY
    with pytest.raises(ValueError, match="unparseable version constant"):
        strict("garbage")


def test_non_ascii_refusals_are_respected_not_migrated():
    # End to end: a refusal whose recorded version uses non-ASCII digits is
    # unparseable → NOT legacy → current refusal, refusal respected.
    for recorded in ("０.１４.３", "٠.١٤.٣", "०.१४.३"):
        decision = _decide_case(
            False, None, ProcessRole.INTERACTIVE_CLI, recorded, kill=False
        )
        assert decision == Decision(False, False, _ALL_OFF, "current_refusal"), recorded


# ---------------------------------------------------------------------------
# decide never raises: hostile StoredConsent subclasses (P2-2)
# ---------------------------------------------------------------------------


class _ExplodingConsent(StoredConsent):
    """Constructible StoredConsent whose ``consent`` field raises on read."""

    def __getattribute__(self, name: str) -> Any:
        if name == "consent":
            raise RuntimeError("hostile consent field")
        return object.__getattribute__(self, name)


class _ExplodingRevision(StoredConsent):
    """Same, for the ``notice_revision_seen`` field."""

    def __getattribute__(self, name: str) -> Any:
        if name == "notice_revision_seen":
            raise RuntimeError("hostile revision field")
        return object.__getattribute__(self, name)


@pytest.mark.parametrize("kill_switch_active", [True, False], ids=["kill", "no-kill"])
def test_hostile_raising_stored_consent_yields_invalid_input(kill_switch_active):
    # isinstance admits subclasses, so the field-level guard cannot catch
    # this statically: decide must swallow the raise and return the
    # invalid-input decision — with and without the kill switch.
    result = decide(
        _ExplodingConsent(None, None, None),
        ProcessRole.DESKTOP,
        kill_switch_active=kill_switch_active,
        running_version="0.15.0",
    )
    assert result == _INVALID_DECISION


def test_hostile_raising_revision_field_yields_invalid_input():
    result = decide(
        _ExplodingRevision(None, None, None),
        ProcessRole.INTERACTIVE_CLI,
        kill_switch_active=False,
        running_version="0.15.0",
    )
    assert result == _INVALID_DECISION


def test_keyboard_interrupt_from_hostile_stored_propagates():
    class _Interrupting(StoredConsent):
        def __getattribute__(self, name: str) -> Any:
            if name == "consent":
                raise KeyboardInterrupt
            return object.__getattribute__(self, name)

    with pytest.raises(KeyboardInterrupt):
        decide(
            _Interrupting(None, None, None),
            ProcessRole.DESKTOP,
            kill_switch_active=False,
            running_version="0.15.0",
        )


# ---------------------------------------------------------------------------
# decide never raises: fuzz
# ---------------------------------------------------------------------------

_INVALID_DECISION = Decision(False, False, _ALL_OFF, "invalid_input")

_VALID_STORED = [
    StoredConsent(None, None, None),
    StoredConsent(False, "0.14.3", None),
    StoredConsent(True, "0.16.2", 1),
    StoredConsent(None, "0.15.0rc1", 2),
    StoredConsent(False, None, 0),
]

# Hostile StoredConsent substitutes: wrong types outright, or a real
# StoredConsent whose fields violate their annotations. The per-line
# ``type: ignore[arg-type]`` is the point: these fixtures MUST be
# type-hostile for the never-raises contract.
_HOSTILE_STORED = [
    None,
    0,
    "consent",
    object(),
    [],
    {},
    StoredConsent("yes", None, None),  # type: ignore[arg-type]
    StoredConsent(1, None, None),  # type: ignore[arg-type]
    StoredConsent(None, 7, None),  # type: ignore[arg-type]
    StoredConsent(None, ["0.14.3"], None),  # type: ignore[arg-type]
    StoredConsent(None, None, True),  # type: ignore[arg-type]
    StoredConsent(None, None, "1"),  # type: ignore[arg-type]
    StoredConsent(None, None, 1.5),  # type: ignore[arg-type]
    _ExplodingConsent(None, None, None),
    _ExplodingRevision(None, None, None),
]

_HOSTILE_ROLES = [None, "DESKTOP", 0, object(), [ProcessRole.DESKTOP]]
_HOSTILE_KILL = [1, 0, None, "true", object()]
_HOSTILE_RUNNING = [None, 42, object(), ["0.15.0"]]


def test_fuzz_exactly_one_hostile_slot_yields_invalid_input():
    rng = random.Random(1501)
    for _ in range(1500):
        slot = rng.randrange(4)
        stored = rng.choice(_HOSTILE_STORED) if slot == 0 else rng.choice(_VALID_STORED)
        role = rng.choice(_HOSTILE_ROLES) if slot == 1 else rng.choice(_ROLES)
        kill = rng.choice(_HOSTILE_KILL) if slot == 2 else rng.choice(_KILL_VALUES)
        running = (
            rng.choice(_HOSTILE_RUNNING)
            if slot == 3
            else rng.choice(["0.15.0", "0.16.2", "0.14.9"])
        )
        result = decide(
            stored,
            role,
            kill_switch_active=kill,
            running_version=running,  # type: ignore[arg-type]
        )
        # NOTE: ``stored`` is deliberately absent from the failure message —
        # a hostile instance's repr can itself raise.
        assert result == _INVALID_DECISION, (slot, role, kill, running)


def test_fuzz_valid_random_inputs_never_raise_and_keep_closed_reasons():
    rng = random.Random(1502)
    revisions = [None, -3, 0, 1, 2, 99]
    recorded_pool = [
        None,
        "0.14.3",
        "0.15.0rc1",
        "0.15.0",
        "0.16.2",
        "garbage",
        "10.20.30rc7",
    ]
    for _ in range(800):
        stored = StoredConsent(
            consent=rng.choice(_CONSENTS),
            recorded_version=rng.choice(recorded_pool),
            notice_revision_seen=rng.choice(revisions),
        )
        result = decide(
            stored,
            rng.choice(_ROLES),
            kill_switch_active=rng.choice(_KILL_VALUES),
            running_version=rng.choice(["0.15.0", "0.16.2", "0.14.9", "nonsense"]),
        )
        assert isinstance(result, Decision)
        assert result.reason in _ORACLE_REASONS
        assert result.upload_now in (True, False)
        assert result.deliver_notice in (True, False)
