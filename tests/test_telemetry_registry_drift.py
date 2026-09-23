# SPDX-License-Identifier: Apache-2.0
"""Drift check for the telemetry v2 event registry.

Two jobs:

(a) **Self-consistency.** Every enum a property references exists, every
    ``_failed`` event has its success twin with the same identifying
    properties, no property uses a free-form string kind, and the
    ``count_bucket`` scale is Orca's exactly (design sec 1.3).

(b) **One physical file.** The desktop app must read the SAME
    ``rapid_mlx/telemetry/events.json`` — not a copy that can rot. The
    Swift validator's source fallback points at it, ``scripts/build.sh``
    copies that file into the shipped ``.app``, and no second copy exists
    anywhere under ``apps/rapid-mac``. If one ever appears, this fails.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path

import pytest

from rapid_mlx.telemetry import registry as reg

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_REGISTRY = REPO_ROOT / "rapid_mlx" / "telemetry" / "events.json"
MAC_APP = REPO_ROOT / "apps" / "rapid-mac"
SWIFT_VALIDATOR = (
    MAC_APP / "Sources" / "Rapid" / "Telemetry" / "TelemetryRegistry.swift"
)
MAC_BUILD_SCRIPT = MAC_APP / "scripts" / "build.sh"

# The property kinds the registry is allowed to declare. There is
# deliberately no "string": a free-form string is how caller-controlled
# text reaches the wire (design sec 1.5).
EVENT_KINDS = {"enum", "bool", "int", "model_id"}
COMMON_ONLY_KINDS = {"version", "uuid"}

# Orca's bucket scale, reproduced from design sec 1.3. Order matters:
# the chart is "installs that reached bucket >= N".
COUNT_BUCKET_SCALE = [
    "1",
    "2",
    "3_4",
    "5_9",
    "10_19",
    "20_49",
    "50_99",
    "100_199",
    "200_499",
    "500_999",
    "1000_plus",
]


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(ENGINE_REGISTRY.read_text(encoding="utf-8"))


def _specs(mapping: dict) -> dict:
    """Drop the ``_``-prefixed documentation keys the JSON carries."""

    return {k: v for k, v in mapping.items() if not k.startswith("_")}


def _load_mutated_registry(
    monkeypatch, tmp_path, registry, only_when, *, controller_kind=None
):
    mutated = deepcopy(registry)
    mutated["events"]["_test_documentation"] = {}
    if controller_kind is not None:
        mutated["events"]["server_start_state"]["props"]["state"]["kind"] = (
            controller_kind
        )
    mutated["events"]["server_start_state"]["props"]["failure_stage"]["only_when"] = (
        only_when
    )
    path = tmp_path / "events.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    reg.load_registry.cache_clear()
    monkeypatch.setattr(reg, "registry_path", lambda: path)
    try:
        return reg.load_registry()
    finally:
        reg.load_registry.cache_clear()


# ------------------------------------------------- (a) self-consistency


def test_every_referenced_enum_exists(registry):
    known = set(_specs(registry["enums"]))
    for owner, specs in [
        ("common_props", _specs(registry["common_props"])),
        *[
            (name, _specs(event["props"]))
            for name, event in _specs(registry["events"]).items()
        ],
    ]:
        for prop, spec in specs.items():
            if spec["kind"] != "enum":
                continue
            assert spec["enum"] in known, (
                f"{owner}.{prop} -> unknown enum {spec['enum']}"
            )


def test_every_declared_enum_is_referenced(registry):
    """An orphan enum is dead weight that quietly stops matching reality."""

    used = {
        spec["enum"]
        for specs in [
            _specs(registry["common_props"]),
            *[_specs(e["props"]) for e in _specs(registry["events"]).values()],
        ]
        for spec in specs.values()
        if spec["kind"] == "enum"
    }
    assert set(_specs(registry["enums"])) == used


def test_enum_values_are_unique_and_non_empty(registry):
    for name, body in _specs(registry["enums"]).items():
        values = body["values"]
        assert values, f"enum {name} is empty"
        assert len(values) == len(set(values)), f"enum {name} has duplicates"
        for value in values:
            assert isinstance(value, str) and value, f"enum {name} has a bad value"


def test_no_property_uses_a_free_form_string_kind(registry):
    for name, event in _specs(registry["events"]).items():
        for prop, spec in _specs(event["props"]).items():
            assert spec["kind"] in EVENT_KINDS, f"{name}.{prop} kind={spec['kind']}"
    for prop, spec in _specs(registry["common_props"]).items():
        assert spec["kind"] in EVENT_KINDS | COMMON_ONLY_KINDS, (
            f"common_props.{prop} kind={spec['kind']}"
        )


def test_pattern_capped_kinds_carry_a_pattern_and_a_cap(registry):
    for prop, spec in _specs(registry["common_props"]).items():
        if spec["kind"] == "version":
            assert spec["pattern"] and spec["max_length"], prop
            re.compile(spec["pattern"])
        if spec["kind"] == "uuid":
            assert spec.get("required") is True, prop


def test_int_properties_declare_both_bounds(registry):
    sources = [_specs(registry["common_props"])] + [
        _specs(e["props"]) for e in _specs(registry["events"]).values()
    ]
    for specs in sources:
        for prop, spec in specs.items():
            if spec["kind"] != "int":
                continue
            assert isinstance(spec.get("min"), int), prop
            assert isinstance(spec.get("max"), int), prop
            assert spec["min"] < spec["max"], prop


def test_count_bucket_scale_is_exact(registry):
    assert registry["enums"]["count_bucket"]["values"] == COUNT_BUCKET_SCALE


def test_server_start_state_contract_is_exact(registry):
    assert registry["enums"]["server_start_state"]["values"] == [
        "attempted",
        "ready",
        "failed",
    ]
    assert registry["enums"]["load_policy"]["values"] == ["eager", "lazy", "none"]
    assert registry["enums"]["failure_stage"]["values"] == [
        "resolve",
        "download",
        "preflight",
        "prepare",
        "engine_start",
        "bind",
    ]
    props = _specs(registry["events"]["server_start_state"]["props"])
    assert set(props) == {"state", "model_type", "load_policy", "failure_stage"}
    assert props["state"] == {
        "kind": "enum",
        "enum": "server_start_state",
        "required": True,
    }
    assert props["failure_stage"] == {
        "kind": "enum",
        "enum": "failure_stage",
        "required": False,
        "only_when": {"state": ["failed"]},
    }


@pytest.mark.parametrize(
    "only_when",
    [
        {},
        {"undeclared_controller": ["failed"]},
        {"state": ["not-a-server-start-state"]},
    ],
    ids=["empty-shape-while-property-absent", "unknown-controller", "unknown-value"],
)
def test_registry_load_rejects_invalid_only_when(
    monkeypatch, tmp_path, registry, only_when
):
    """Conditional metadata is registry schema, not caller-dependent data."""

    with pytest.raises(ValueError, match="only_when"):
        _load_mutated_registry(monkeypatch, tmp_path, registry, only_when)
    if only_when == {}:
        with pytest.raises(ValueError, match="only_when"):
            _load_mutated_registry(
                monkeypatch,
                tmp_path,
                registry,
                {"state": []},
            )
        with pytest.raises(ValueError, match="only_when"):
            _load_mutated_registry(
                monkeypatch,
                tmp_path,
                registry,
                {"state": ["failed"]},
                controller_kind="bool",
            )


def test_model_id_pattern_accepts_only_the_four_declared_shapes(registry):
    spec = registry["model_id"]
    pattern = re.compile(spec["pattern"])
    for reserved in spec["reserved"]:
        assert pattern.fullmatch(reserved), reserved
    for good in ("qwen3.5-4b-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit"):
        assert pattern.fullmatch(good), good
    for bad in ("/abs/path", "a/b/c", "has space", "~/x"):
        assert not pattern.fullmatch(bad), bad
    assert spec["max_length"] <= 256


def test_failed_twins_exist_and_mirror_their_success_event(registry):
    events = _specs(registry["events"])
    failed = [name for name in events if name.endswith("_failed")]
    assert failed, "the release-1 set has _failed twins"
    for name in failed:
        success = name[: -len("_failed")]
        # ``model_pull_failed`` twins ``model_pulled``; the declared
        # ``twin_of`` is authoritative, the suffix is only a hint.
        declared = events[name].get("twin_of")
        assert declared in events, f"{name} declares no live twin_of"
        assert declared.startswith(success), f"{name} twins {declared}?"

        success_props = _specs(events[declared]["props"])
        twin_props = _specs(events[name]["props"])

        assert "error_class" in twin_props, f"{name} has no error_class"
        assert twin_props["error_class"]["required"] is True, name
        assert twin_props["error_class"]["kind"] == "enum", name

        identifying = {k: v for k, v in twin_props.items() if k != "error_class"}
        assert set(identifying) == set(success_props), (
            f"{name} identifying props differ from {declared}"
        )
        for prop, spec in identifying.items():
            assert spec["required"] is False, f"{name}.{prop} must be optional"
            same = dict(success_props[prop])
            same["required"] = False
            assert {k: v for k, v in spec.items() if not k.startswith("_")} == {
                k: v for k, v in same.items() if not k.startswith("_")
            }, f"{name}.{prop} does not mirror {declared}.{prop}"


def test_every_success_event_with_a_failure_mode_has_a_twin(registry):
    events = _specs(registry["events"])
    twinned = {e["twin_of"] for e in events.values() if "twin_of" in e}
    for expected in ("model_pulled", "model_served", "agent_configured"):
        assert expected in twinned, f"{expected} lost its _failed twin"


def test_event_names_and_property_names_are_snake_case(registry):
    name_re = re.compile(r"^[a-z][a-z0-9_]*$")
    for name, event in _specs(registry["events"]).items():
        assert name_re.fullmatch(name), name
        for prop in _specs(event["props"]):
            assert name_re.fullmatch(prop), f"{name}.{prop}"


def test_loader_agrees_with_the_raw_file(registry):
    assert reg.registry_version() == registry["registry_version"]
    assert reg.event_names() == frozenset(_specs(registry["events"]))


# ------------------------------------------------- (b) engine <-> Swift


def test_desktop_app_holds_no_second_copy_of_the_registry():
    """One physical file, or the two validators drift apart in a release."""

    # ``os.walk`` rather than ``rglob`` so the 1 GB+ local SwiftPM build
    # directories are pruned instead of walked.
    copies: list[str] = []
    for dirpath, dirnames, filenames in os.walk(MAC_APP):
        dirnames[:] = [d for d in dirnames if d not in (".build", "build", "Vendor")]
        if "events.json" in filenames:
            copies.append(str(Path(dirpath) / "events.json"))
    assert copies == [], f"duplicate registry under apps/rapid-mac: {copies}"


def test_swift_validator_resolves_the_engine_file():
    source = SWIFT_VALIDATOR.read_text(encoding="utf-8")
    assert "rapid_mlx/telemetry/events.json" in source


def test_mac_build_script_ships_the_engine_file():
    script = MAC_BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "$ROOT/../../rapid_mlx/telemetry/events.json" in script
    # And it must be fatal, not a warning: an app that silently ships
    # without the registry would fail closed on every event, invisibly.
    assert "refusing to ship a telemetry validator with no registry" in script


def test_swift_validator_does_not_redeclare_the_enums():
    """The enums live in JSON. A Swift copy is the drift we are preventing."""

    source = SWIFT_VALIDATOR.read_text(encoding="utf-8")
    registry = json.loads(ENGINE_REGISTRY.read_text(encoding="utf-8"))
    # Compare against STRING LITERALS only: `continue` is a Swift keyword
    # and `other` reads as English, so a substring scan would false-alarm.
    literals = set(re.findall(r'"([^"\\\n]*)"', source))
    for enum_name, values in (
        (name, body["values"]) for name, body in _specs(registry["enums"]).items()
    ):
        leaked = sorted(literals.intersection(values))
        assert not leaked, (
            f"{enum_name} values {leaked} are hard-coded in TelemetryRegistry.swift"
        )


def test_wheel_ships_the_registry():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"telemetry/events.json",' in pyproject


# ------------------------------- (c) enums that must track live engine code
#
# Three enums in events.json are not free inventions: they are copies of
# tables that already exist in the engine. If someone adds an agent profile,
# a User-Agent marker or an instrumented route without touching the registry,
# the new value would silently ship as "other" — these tests make that a
# build failure instead of a slow leak of information.


def test_agent_enum_matches_the_shipped_agent_profiles(registry):
    import rapid_mlx.agents as agents

    declared = set(registry["enums"]["agent"]["values"]) - {"other"}
    assert declared == {profile.name for profile in agents.list_profiles()}


def test_caller_enum_matches_the_user_agent_marker_table(registry):
    from rapid_mlx.client_header import RAPID_CLIENT_LABELS
    from rapid_mlx.telemetry.redact import _CALLER_AGENT_MARKERS

    declared = set(registry["enums"]["caller"]["values"]) - {"other", "unknown"}
    assert declared == {
        *(label for _, label in _CALLER_AGENT_MARKERS),
        *RAPID_CLIENT_LABELS,
    }


def test_model_type_enum_covers_every_alias_modality(registry):
    from rapid_mlx.model_aliases import _RESERVED_MODALITIES, _VALID_MODALITIES

    declared = set(registry["enums"]["model_type"]["values"])
    # ``text`` splits into llm / vlm on ``supports_image_input``; every other
    # modality maps to itself. Reserved modalities have no dispatch path yet.
    assert (_VALID_MODALITIES - {"text"}) <= declared
    assert {"llm", "vlm"} <= declared
    assert not (_RESERVED_MODALITIES & declared)


# ``quant`` is the fourth such enum, and the one the catalog can move under:
# every new alias ships a quantization spelling in its name. The registry
# carries ONE canonical token per concept (``q4`` normalizes to ``4bit``), so
# these three tests pin the normalization in both directions.

CATALOG = REPO_ROOT / "rapid_mlx" / "aliases.json"

# Deliberately NOT the module's own regex — a copy of it would make the test
# tautological. This is an independently written "looks like a quantization
# marker" shape; if it drifts wider than the module's, the test gets stricter,
# which is the safe direction.
QUANT_SHAPED = re.compile(
    r"^(?:q\d+|dq\d+|\d+bit|\d+bpw|int\d+|[a-z]{0,2}fp\d+|bf\d+"
    r"|nf\d+|dwq|awq|gptq)$"
)

# Spellings we see in the catalog, have decided NOT to give a canonical token,
# and therefore accept as ``other``. Every entry is a deliberate decision:
#   5bpw -> ``qwen3.8-27b-mixed-3.5bpw``, a mixed 3.5-bits-per-weight
#           checkpoint that no single bit-width token describes honestly.
# A spelling that is in neither this set nor the module's table fails below,
# which is the point: the registry cannot fall behind the catalog silently.
ACCEPTED_AS_OTHER = {"5bpw"}


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


#: Alias fields that hold a Hugging Face repo id. ``hf_path`` is the served
#: checkpoint; the draft entries are separate repos with their own quant
#: spelling in the name, and review round 1 caught that skipping them left a
#: hole in the drift net for 22 catalog entries.
REPO_ID_FIELDS = (
    "hf_path",
    "mtp_draft_model",
    "native_mtp_draft_model",
    "ddtree_draft_model",
    "dflash_draft_model",
)


def _catalog_names(catalog: dict) -> list[str]:
    """Every alias spelling AND every repo id the catalog names."""

    names: list[str] = []
    for alias, entry in catalog.items():
        names.append(alias)
        for field in REPO_ID_FIELDS:
            value = entry.get(field)
            if value:
                names.append(value)
    return names


def test_the_draft_repo_fields_are_really_in_the_catalog():
    """If a field is renamed, ``_catalog_names`` must not go quietly blind."""

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    present = {field for entry in catalog.values() for field in entry}
    assert set(REPO_ID_FIELDS) <= present, (
        f"REPO_ID_FIELDS no longer match aliases.json: {sorted(present)}"
    )


def test_every_catalog_name_maps_inside_the_quant_enum(registry, catalog):
    from rapid_mlx.telemetry.quant import quant_token

    allowed = set(registry["enums"]["quant"]["values"])
    for name in _catalog_names(catalog):
        token = quant_token(name)
        assert token in allowed, f"{name} -> {token!r}, outside the quant enum"


def test_no_catalog_quant_spelling_is_unrecognized(catalog):
    """A new alias spelling must be mapped on purpose, not absorbed."""

    from rapid_mlx.telemetry.quant import known_quant_tokens

    known = known_quant_tokens() | ACCEPTED_AS_OTHER
    seen: set[str] = set()
    for name in _catalog_names(catalog):
        for raw in re.findall(r"[a-z0-9]+", name.lower()):
            if QUANT_SHAPED.match(raw):
                seen.add(raw)
    unmapped = sorted(seen - known)
    assert not unmapped, (
        f"catalog quant spellings with no canonical token: {unmapped}; map "
        "them in rapid_mlx/telemetry/quant.py or accept them as 'other' here"
    )
    # And the mapping is not dead weight: the catalog really does use it.
    assert seen, "no quant spellings found in the catalog at all?"


def test_quant_enum_and_the_normalizer_agree_exactly(registry):
    """Neither side may grow a value the other does not know about."""

    from rapid_mlx.telemetry.quant import canonical_quant_values

    assert canonical_quant_values() == set(registry["enums"]["quant"]["values"])
