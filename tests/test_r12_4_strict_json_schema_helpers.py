# SPDX-License-Identifier: Apache-2.0
"""R12-4 — unit tests for the strict-json-schema helper module."""
from __future__ import annotations

import json

import pytest

from vllm_mlx.api.strict_json_schema import (
    build_repair_messages,
    build_violation_envelope,
    extract_json_payload,
    repair_retry_enabled,
    strict_enforcement_enabled,
    validate_and_envelope,
)


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


def test_strict_enforcement_enabled_default_true(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_STRICT_JSON_SCHEMA", raising=False)
    assert strict_enforcement_enabled() is True


@pytest.mark.parametrize("value", ["off", "0", "false", "no", "disable", "disabled", "OFF", "False"])
def test_strict_enforcement_enabled_off_values_disable(monkeypatch, value):
    monkeypatch.setenv("RAPID_MLX_STRICT_JSON_SCHEMA", value)
    assert strict_enforcement_enabled() is False


@pytest.mark.parametrize("value", ["on", "1", "true", "yes", "", "anything-else"])
def test_strict_enforcement_enabled_truthy_or_empty_values_enable(monkeypatch, value):
    monkeypatch.setenv("RAPID_MLX_STRICT_JSON_SCHEMA", value)
    assert strict_enforcement_enabled() is True


def test_repair_retry_enabled_default_true(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_STRICT_JSON_SCHEMA_REPAIR", raising=False)
    assert repair_retry_enabled() is True


def test_repair_retry_enabled_off_disables(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_STRICT_JSON_SCHEMA_REPAIR", "off")
    assert repair_retry_enabled() is False


# ---------------------------------------------------------------------------
# extract_json_payload
# ---------------------------------------------------------------------------


def test_extract_json_payload_strips_outer_fence():
    raw = '```json\n{"x": 1}\n```'
    assert extract_json_payload(raw) == '{"x": 1}'


def test_extract_json_payload_strips_fence_without_language_tag():
    raw = "```\n{\"x\": 1}\n```"
    assert extract_json_payload(raw) == '{"x": 1}'


def test_extract_json_payload_preserves_unfenced_input():
    assert extract_json_payload('{"x": 1}') == '{"x": 1}'


def test_extract_json_payload_empty_input():
    assert extract_json_payload("") == ""
    assert extract_json_payload("   ") == ""


def test_extract_json_payload_does_not_strip_inner_fence():
    """Inner fences (mid-stream) are preserved — json.loads will
    handle them by rejecting (correct failure mode)."""
    raw = '{"code": "```json"}'
    assert extract_json_payload(raw) == '{"code": "```json"}'


# ---------------------------------------------------------------------------
# validate_and_envelope
# ---------------------------------------------------------------------------


_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {"type": "integer", "minimum": 18},
        "name": {"type": "string"},
    },
    "required": ["age", "name"],
    "additionalProperties": False,
}


def test_validate_and_envelope_accepts_valid_input():
    ok, details = validate_and_envelope(
        json.dumps({"age": 30, "name": "Ada"}), _SCHEMA
    )
    assert ok is True
    assert details is None


def test_validate_and_envelope_rejects_empty_input():
    ok, details = validate_and_envelope("", _SCHEMA)
    assert ok is False
    assert details["reason"] == "empty"


def test_validate_and_envelope_rejects_invalid_json():
    ok, details = validate_and_envelope("not json", _SCHEMA)
    assert ok is False
    assert details["reason"] == "invalid_json"


def test_validate_and_envelope_schema_violation_minimum():
    ok, details = validate_and_envelope(
        json.dumps({"age": 5, "name": "Ada"}), _SCHEMA
    )
    assert ok is False
    assert details["reason"] == "schema_violation"
    assert details["failing_path"] == "/age"
    assert "minimum" in details["expected"]
    assert details["got"] == 5


def test_validate_and_envelope_schema_violation_required():
    ok, details = validate_and_envelope(json.dumps({"age": 25}), _SCHEMA)
    assert ok is False
    assert details["reason"] == "schema_violation"


def test_validate_and_envelope_strips_fence_before_validating():
    ok, details = validate_and_envelope(
        '```json\n{"age": 30, "name": "Ada"}\n```', _SCHEMA
    )
    assert ok is True, details


# ---------------------------------------------------------------------------
# build_repair_messages
# ---------------------------------------------------------------------------


def test_build_repair_messages_includes_schema_and_path():
    original = [{"role": "user", "content": "make a JSON object"}]
    failure = {
        "reason": "schema_violation",
        "failing_path": "/age",
        "expected": "minimum: 18",
        "got": 5,
        "message": "5 is less than the minimum of 18",
    }
    repair = build_repair_messages(original, '{"age": 5}', _SCHEMA, failure)
    # Original + assistant turn (failed output) + system hint + user re-ask
    assert len(repair) == 4
    assert repair[0] == original[0]
    assert repair[1]["role"] == "assistant"
    assert repair[2]["role"] == "system"
    assert "REPAIR" in repair[2]["content"].upper()
    assert "/age" in repair[2]["content"]
    assert repair[3]["role"] == "user"


def test_build_repair_messages_skips_assistant_turn_on_empty_output():
    original = [{"role": "user", "content": "make a JSON object"}]
    failure = {"reason": "empty", "message": "model emitted no content"}
    repair = build_repair_messages(original, "", _SCHEMA, failure)
    # No assistant turn when failed_output is empty.
    assert len(repair) == 3
    assert repair[0] == original[0]
    assert repair[1]["role"] == "system"
    assert "empty" in repair[1]["content"].lower()


def test_build_repair_messages_handles_invalid_json_reason():
    original = [{"role": "user", "content": "json"}]
    failure = {"reason": "invalid_json", "message": "Expecting value: line 1 column 1"}
    repair = build_repair_messages(original, "not json", _SCHEMA, failure)
    sys_content = repair[2]["content"]
    assert "not valid JSON" in sys_content


# ---------------------------------------------------------------------------
# build_violation_envelope
# ---------------------------------------------------------------------------


def test_build_violation_envelope_default_param():
    env = build_violation_envelope(
        {
            "reason": "schema_violation",
            "failing_path": "/x",
            "expected": "type: 'integer'",
            "got": "foo",
            "message": "'foo' is not of type 'integer'",
        },
        attempts=2,
    )
    err = env["error"]
    assert err["type"] == "validation_error"
    assert err["code"] == "json_schema_violation"
    assert err["param"] == "response_format.json_schema"
    assert err["details"]["attempts"] == 2
    assert err["details"]["failing_path"] == "/x"
    assert err["details"]["got"] == "foo"


def test_build_violation_envelope_custom_param():
    env = build_violation_envelope(
        {"reason": "schema_violation", "failing_path": "/x"},
        param="text.format",
        attempts=1,
    )
    assert env["error"]["param"] == "text.format"


def test_build_violation_envelope_invalid_json_reason_prefix():
    env = build_violation_envelope(
        {"reason": "invalid_json", "message": "expected value"},
        attempts=2,
    )
    assert "not valid JSON" in env["error"]["message"]


# ---------------------------------------------------------------------------
# Codex r1 #1: FormatChecker integration — `format` keywords MUST be
# enforced rather than treated as annotation-only. Pre-fix the
# validator was instantiated without ``format_checker=FormatChecker()``
# so ``{"format":"email"}`` validated any string. This test pins the
# fix so a refactor that drops the FormatChecker is caught.
# ---------------------------------------------------------------------------


def test_validate_and_envelope_enforces_format_email():
    schema = {
        "type": "object",
        "properties": {"e": {"type": "string", "format": "email"}},
        "required": ["e"],
    }
    ok, details = validate_and_envelope(json.dumps({"e": "not-an-email"}), schema)
    assert ok is False
    # The FAILING keyword must be ``format`` — not ``type`` (which
    # would mean the validator regressed to type-checking only).
    assert details["expected"].startswith("format:"), details
