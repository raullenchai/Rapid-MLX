# SPDX-License-Identifier: Apache-2.0
"""Strict-validator contract for the telemetry v2 event registry.

Each test here guards ONE strictness rule and is written so that removing
the rule turns it red — that was verified by fault injection before merge,
not assumed. The registry itself is inspected separately in
``test_telemetry_registry_drift.py``.
"""

from __future__ import annotations

import json

import pytest

from rapid_mlx.telemetry import registry as reg


@pytest.fixture(autouse=True)
def _clear_log_ledger():
    """The debug log fires once per key per PROCESS; tests share one."""

    reg._reset_log_state_for_tests()
    yield
    reg._reset_log_state_for_tests()


def _served(**overrides):
    props = {
        "model": "qwen3.5-4b-4bit",
        "model_type": "llm",
        "auto_selected": True,
        "quant": "4bit",
    }
    props.update(overrides)
    return props


def _common(**overrides):
    props = {
        "app_version": "0.15.0",
        "surface": "cli",
        "os": "darwin",
        "os_version": "25.3",
        "arch": "arm64",
        "chip": "m3-max",
        "memory_gb": 64,
        "python_version": "3.12",
        "install_id": "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f",
        "session_id": "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9",
        "channel": "stable",
        "nth_model_served": 3,
        "days_since_first_run_bucket": "2-6",
    }
    props.update(overrides)
    return props


# ------------------------------------------------------------- packaging


def test_registry_loads_via_importlib_resources():
    """The wheel must carry events.json, not just the source tree."""

    raw = reg.registry_path().read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["registry_version"] == reg.registry_version() == 1
    assert parsed["events"]


def test_release_one_event_set_is_present():
    assert reg.event_names() == frozenset(
        {
            "app_opened",
            "active_day",
            "model_pulled",
            "model_pull_failed",
            "model_served",
            "model_serve_failed",
            "capability_rejected",
            "inference_bucket_reached",
            "agent_configured",
            "agent_configure_failed",
            "telemetry_opted_out",
            "telemetry_opted_in",
        }
    )


# ------------------------------------------------------------ strictness


def test_valid_event_passes_through_unchanged():
    assert reg.validate("model_served", _served()) == _served()


def test_unknown_event_is_dropped():
    assert reg.validate("model_teleported", {}) is None


def test_unknown_property_drops_the_whole_event():
    """NOT "strip the key and send the rest" — the whole event goes."""

    assert reg.validate("model_served", _served(prompt="hello")) is None


def test_missing_required_property_is_dropped():
    props = _served()
    del props["model_type"]
    assert reg.validate("model_served", props) is None


def test_optional_property_may_be_absent():
    assert reg.validate("model_serve_failed", {"error_class": "corrupt_weights"}) == {
        "error_class": "corrupt_weights"
    }


def test_out_of_enum_value_is_dropped():
    assert reg.validate("model_served", _served(model_type="telepathy")) is None


def test_wrong_type_is_dropped():
    assert reg.validate("model_served", _served(auto_selected="true")) is None
    assert reg.validate("model_served", _served(quant=4)) is None


def test_bool_is_not_accepted_where_an_int_is_declared():
    """``isinstance(True, int)`` is True in Python; the validator must not be."""

    assert reg.validate_common(_common(memory_gb=True)) is None


def test_int_range_is_enforced():
    assert reg.validate_common(_common(memory_gb=-1)) is None
    assert reg.validate_common(_common(memory_gb=999_999)) is None
    assert reg.validate_common(_common(memory_gb=0)) is not None


@pytest.mark.parametrize(
    "model",
    [
        "/Users/someone/models/secret",
        "~/models/secret",
        "org/name/extra",
        "my model",
        "a" * 200,
        "",
        "<unknown>",
    ],
)
def test_model_id_pattern_and_cap_are_enforced(model):
    assert reg.validate("model_served", _served(model=model)) is None


@pytest.mark.parametrize(
    "model",
    ["qwen3.5-4b-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit", "<custom>", "<local>"],
)
def test_model_id_accepts_the_four_declared_shapes(model):
    assert reg.validate("model_served", _served(model=model)) is not None


def test_unknown_kind_fails_closed():
    """A malformed registry must reject, not wave the value through.

    ``_check_value`` is exercised directly because the shipped
    ``events.json`` (correctly) declares no such kind — the branch exists
    so that a future bad edit to the registry cannot open a hole.
    """

    loaded = reg.load_registry()
    assert reg._check_value({"kind": "freeform"}, "anything", loaded) is False
    assert reg._check_value({}, "anything", loaded) is False


def test_props_must_be_a_mapping():
    assert reg.validate("app_opened", ["model"]) is None  # type: ignore[arg-type]


def test_validate_never_raises_on_hostile_input():
    for props in ({"model": object()}, {object(): 1}, None):
        assert reg.validate("model_served", props) is None  # type: ignore[arg-type]


def test_count_bucket_values_round_trip():
    for bucket in ("1", "3_4", "1000_plus"):
        assert (
            reg.validate(
                "inference_bucket_reached",
                {
                    "model": "<custom>",
                    "endpoint": "/v1/chat/completions",
                    "caller": "claude-code",
                    "result": "failed",
                    "count_bucket": bucket,
                    "bucket_source": "crossed_now",
                },
            )
            is not None
        )
    assert (
        reg.validate(
            "inference_bucket_reached",
            {
                "model": "<custom>",
                "endpoint": "/v1/chat/completions",
                "caller": "claude-code",
                "result": "failed",
                "count_bucket": "3-4",
                "bucket_source": "crossed_now",
            },
        )
        is None
    )


# ---------------------------------------------------------- common props


def test_common_props_validate():
    assert reg.validate_common(_common()) == _common()


def test_python_version_is_optional_for_the_desktop_surface():
    props = _common(surface="desktop")
    del props["python_version"]
    assert reg.validate_common(props) is not None


def test_cohort_props_are_optional_when_the_store_cannot_answer():
    """The two cohort stamps are optional on purpose: a read-only HOME,
    a locked or corrupt database means the store cannot answer, and
    dropping EVERY event over it would blind telemetry on exactly the
    machines having trouble. Absence must validate — analysts read it
    as "unknown", never as 0 / first day."""

    props = _common()
    del props["nth_model_served"]
    del props["days_since_first_run_bucket"]
    accepted = reg.validate_common(props)
    assert accepted is not None
    assert "nth_model_served" not in accepted
    assert "days_since_first_run_bucket" not in accepted


def test_non_uuid_install_id_is_dropped():
    assert reg.validate_common(_common(install_id="not-a-uuid")) is None


def test_unknown_common_property_drops_the_envelope():
    """``country`` is exactly what design D6 removed; it must not sneak back."""

    assert reg.validate_common(_common(country="US")) is None


def test_bad_version_strings_are_dropped():
    for bad in ("0.15", "0.15.0-dirty", "v0.15.0", "x" * 40):
        assert reg.validate_common(_common(app_version=bad)) is None


# ------------------------------------------------------------- log guard


def test_rejection_logs_at_most_once_per_event_name(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        "rapid_mlx.telemetry.debug_log._log", lambda msg: seen.append(msg)
    )
    monkeypatch.setenv("RAPID_MLX_TELEMETRY_DEBUG", "1")
    for _ in range(50):
        assert reg.validate("model_served", _served(model_type="nope")) is None
    assert len(seen) == 1


@pytest.mark.parametrize("value", ("1", "true", "YES"))
def test_v2_debug_log_writes_only_when_enabled(monkeypatch, capsys, value):
    from rapid_mlx.telemetry import debug_log

    monkeypatch.setenv(debug_log.DEBUG_ENV, value)
    assert debug_log.debug_enabled() is True
    debug_log._log("hello")
    assert capsys.readouterr().err == "[telemetry] hello\n"


@pytest.mark.parametrize("value", ("", "0", "false", "no", "off"))
def test_v2_debug_log_is_silent_when_disabled(monkeypatch, capsys, value):
    from rapid_mlx.telemetry import debug_log

    monkeypatch.setenv(debug_log.DEBUG_ENV, value)
    assert debug_log.debug_enabled() is False
    debug_log._log("hello")
    assert capsys.readouterr().err == ""
