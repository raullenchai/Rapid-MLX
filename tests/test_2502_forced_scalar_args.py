# SPDX-License-Identifier: Apache-2.0
"""#2502 — a concrete forced named tool call must yield schema-valid arguments
when the small model emits the bare required value, while preserving the #1256
fail-closed 422 for genuinely invalid generations.

Repro (from the issue): ``tool_choice={"type":"function","function":{"name":
"weather"}}`` against a JSON-schema tool whose ``parameters`` require ``city``.
Qwen3.5-4B-4bit (Hermes parser) forced onto that tool emits ``"arguments":
"San Francisco"`` — a bare scalar, not ``{"city": "San Francisco"}``. The
forced-choice repair collapsed that to ``"{}"`` and the #1256 schema gate 422'd
on the missing required ``city``; a more imperative re-prompt happened to work.

Fix (schema-aware scalar salvage): when the target tool is an OBJECT-schema tool
with EXACTLY ONE required property and the recovered ``arguments`` is a bare
scalar whose type matches that property, synthesise ``{<prop>: <value>}``.
Gated tightly — multi/zero-required schemas, type mismatches, JSON ``null``,
arrays, and broken-object fragments all still fail closed exactly as before.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from vllm_mlx.routes.chat import (
    _forced_synth_schema_error,
    _recover_bare_scalar_from_raw,
    _repair_forced_call_arguments,
    _salvage_forced_scalar_arguments,
    _synthesize_forced_tool_call,
)

# DeepSeek-style tool-call wire delimiters (begin / end), matching the route's
# ``_WIRE_OPENERS`` / ``_WIRE_CLOSERS`` byte-for-byte.
_BEGIN_MARKER = (
    "<"
    + chr(0xFF5C)
    + "tool"
    + chr(0x2581)
    + "calls"
    + chr(0x2581)
    + "begin"
    + chr(0xFF5C)
    + ">"
)
_END_MARKER = (
    "<"
    + chr(0xFF5C)
    + "tool"
    + chr(0x2581)
    + "calls"
    + chr(0x2581)
    + "end"
    + chr(0xFF5C)
    + ">"
)


def _tool(
    name: str,
    required: list[str] | None,
    *,
    ptype: str = "string",
    props: dict | None = None,
):
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props or {required[0]: {"type": ptype}},
    }
    if required is not None:
        schema["required"] = required
    return SimpleNamespace(
        type="function", function={"name": name, "parameters": schema}
    )


def _no_required_schema():
    return SimpleNamespace(
        type="function",
        function={
            "name": "ping",
            "parameters": {"type": "object", "properties": {}},
        },
    )


_WEATHER = _tool("weather", ["city"])  # single required STRING prop
_ADD = _tool("add", ["a", "b"], ptype="integer")  # multi-required
_NUM = _tool("temperature", ["degrees"], ptype="number")  # single required NUMBER
_OBJ_ARR = SimpleNamespace(
    type="function",
    function={
        "name": "obj",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
        },
    },
)


# =====================================================================
# Unit: _salvage_forced_scalar_arguments (the gated decision core)
# =====================================================================


class TestSalvageForcedScalarArguments:
    def test_bare_string_maps_single_required_string_prop(self):
        out = _salvage_forced_scalar_arguments("weather", "San Francisco", [_WEATHER])
        assert out == '{"city": "San Francisco"}'

    def test_json_string_value_maps_single_required_string_prop(self):
        out = _salvage_forced_scalar_arguments("weather", '"SF"', [_WEATHER])
        assert out == '{"city": "SF"}'

    def test_number_maps_single_required_number_prop(self):
        out = _salvage_forced_scalar_arguments("temperature", "72.5", [_NUM])
        assert out == '{"degrees": 72.5}'

    def test_integer_maps_number_prop(self):
        out = _salvage_forced_scalar_arguments("temperature", "72", [_NUM])
        assert json.loads(out) == {"degrees": 72}

    def test_multi_required_never_guesses(self):
        assert _salvage_forced_scalar_arguments("add", "7", [_ADD]) is None

    def test_no_required_never_guesses(self):
        assert (
            _salvage_forced_scalar_arguments("ping", "x", [_no_required_schema()])
            is None
        )

    def test_type_mismatch_never_guesses(self):
        # Number scalar cannot satisfy a required STRING property.
        assert _salvage_forced_scalar_arguments("weather", "42", [_WEATHER]) is None

    def test_null_never_guesses(self):
        # Explicit JSON ``null`` is a real value, not a salvageable scalar.
        assert _salvage_forced_scalar_arguments("weather", "null", [_WEATHER]) is None

    def test_array_never_guesses(self):
        assert _salvage_forced_scalar_arguments("weather", "[1,2]", [_WEATHER]) is None

    def test_object_value_never_guesses(self):
        assert _salvage_forced_scalar_arguments("weather", "{}", [_WEATHER]) is None
        assert (
            _salvage_forced_scalar_arguments("weather", '{"city": "SF"}', [_WEATHER])
            is None
        )

    def test_empty_arguments_never_guesses(self):
        assert _salvage_forced_scalar_arguments("weather", "", [_WEATHER]) is None
        assert _salvage_forced_scalar_arguments("weather", None, [_WEATHER]) is None

    def test_broken_object_fragment_never_guesses_as_string(self):
        # A fragment that was clearly aiming at a JSON object must not be mapped
        # onto a string property — it fails closed instead.
        assert (
            _salvage_forced_scalar_arguments("weather", '{"unbalanced": ', [_WEATHER])
            is None
        )
        assert _salvage_forced_scalar_arguments("weather", "[bad", [_WEATHER]) is None

    def test_unknown_tool_never_guesses(self):
        assert _salvage_forced_scalar_arguments("other", "x", [_WEATHER]) is None

    def test_boolean_prop(self):
        flag = _tool("toggle", ["on"], ptype="boolean")
        assert (
            _salvage_forced_scalar_arguments("toggle", "true", [flag]) == '{"on": true}'
        )
        # A string cannot satisfy a boolean prop.
        assert _salvage_forced_scalar_arguments("toggle", "yes", [flag]) is None
        # A bool must not be coerced to a NUMBER prop.
        assert _salvage_forced_scalar_arguments("temperature", "true", [_NUM]) is None


# =====================================================================
# _forced_synth_schema_error: salvage succeeds (no 422) where appropriate,
# and genuinely invalid cases still fail closed with the existing message.
# =====================================================================


class TestForcedSynthSchemaErrorWithSalvage:
    def test_bare_string_now_valid_for_single_required_string_prop(self):
        # NEW: no longer a false 422.
        assert (
            _forced_synth_schema_error("weather", "San Francisco", [_WEATHER]) is None
        )

    def test_json_string_now_valid_for_single_required_string_prop(self):
        assert _forced_synth_schema_error("weather", '"SF"', [_WEATHER]) is None

    def test_number_now_valid_for_single_required_number_prop(self):
        assert _forced_synth_schema_error("temperature", "72", [_NUM]) is None

    def test_missing_required_multi_still_422(self):
        err = _forced_synth_schema_error("add", "{}", [_ADD])
        assert err is not None
        assert "required" in err.lower() or "1256" in err

    def test_scalar_violating_type_still_422(self):
        # Number scalar against a required STRING prop → no salvage → 422.
        assert _forced_synth_schema_error("weather", "42", [_WEATHER]) is not None

    def test_null_still_422(self):
        # Explicit null offers nothing to salvage (codex r3 preserved).
        assert _forced_synth_schema_error("weather", "null", [_WEATHER]) is not None

    def test_array_against_object_schema_still_422(self):
        assert _forced_synth_schema_error("obj", "[1,2]", [_OBJ_ARR]) is not None

    def test_broken_object_fragment_still_422(self):
        # A fragment that looks like a broken object is not a salvageable
        # scalar; it keeps failing closed (regression guard).
        assert (
            _forced_synth_schema_error("weather", '{"unbalanced": ', [_WEATHER])
            is not None
        )

    def test_valid_object_unchanged(self):
        assert (
            _forced_synth_schema_error("weather", '{"city": "SF"}', [_WEATHER]) is None
        )


# =====================================================================
# _synthesize_forced_tool_call: produces a schema-valid object for a bare scalar
# =====================================================================


class TestSynthesizeForcedToolCallScalar:
    def test_direct_bare_scalar_with_tools_synthesizes_valid_object(self):
        call = _synthesize_forced_tool_call(
            "weather", "San Francisco", tools=[_WEATHER]
        )
        assert call.function.name == "weather"
        assert json.loads(call.function.arguments) == {"city": "San Francisco"}

    def test_direct_bare_scalar_no_tools_keeps_default(self):
        # Without the schema we must NOT guess — keep the caller's argument.
        call = _synthesize_forced_tool_call("weather", "San Francisco")
        assert call.function.arguments == "San Francisco"

    def test_recovery_from_raw_scalar_emission(self):
        raw = '<tool_call>{"name": "weather", "arguments": "New York"}'
        call = _synthesize_forced_tool_call("weather", raw_text=raw, tools=[_WEATHER])
        assert json.loads(call.function.arguments) == {"city": "New York"}

    def test_multi_required_never_guesses_in_synthesis(self):
        # tools has multi-required → no salvage → unchanged ("{}" default).
        call = _synthesize_forced_tool_call("add", raw_text="", tools=[_ADD])
        assert call.function.arguments == "{}"

    def test_recovered_object_still_preferred(self):
        # A recoverable object beats scalar salvage.
        raw = '<tool_call>{"name": "weather", "arguments": {"city": "LA"}}'
        call = _synthesize_forced_tool_call("weather", raw_text=raw, tools=[_WEATHER])
        assert json.loads(call.function.arguments) == {"city": "LA"}


# =====================================================================
# _repair_forced_call_arguments (the live repro path): writes the valid object
# back onto the wire; genuinely invalid cases still 422.
# =====================================================================


def _call(arguments):
    return SimpleNamespace(
        id="call_x",
        type="function",
        function=SimpleNamespace(name="weather", arguments=arguments),
    )


class TestRepairForcedCallArgumentsScalar:
    def test_bare_scalar_repairs_to_schema_valid_object(self):
        tc = _call("San Francisco")
        err = _repair_forced_call_arguments([tc], "", "weather", [_WEATHER])
        assert err is None
        assert json.loads(tc.function.arguments) == {"city": "San Francisco"}

    def test_multi_required_invalid_still_returns_error(self):
        tc = _call("1")  # actually ends up "1" against add
        tc.function.name = "add"
        err = _repair_forced_call_arguments([tc], "", "add", [_ADD])
        assert err is not None
        assert tc.function.arguments == "{}"

    def test_no_required_still_repairs_to_empty_object(self):
        tc = _call("1")
        err = _repair_forced_call_arguments([tc], "", "ping", [_no_required_schema()])
        assert err is None
        assert tc.function.arguments == "{}"


# =====================================================================
# _recover_bare_scalar_from_raw: bounded scalar recovery from response text
# =====================================================================


class TestRecoverBareScalarFromRaw:
    def test_extracts_quoted_scalar_in_envelope(self):
        raw = '<tool_call>{"name": "weather", "arguments": "San Francisco"}'
        assert (
            _recover_bare_scalar_from_raw(raw, expected_name="weather")
            == '"San Francisco"'
        )

    def test_extracts_unquoted_number_in_envelope(self):
        raw = '<tool_call>{"name": "temperature", "arguments": 72}'
        assert _recover_bare_scalar_from_raw(raw, expected_name="temperature") == "72"

    def test_ignores_object_argument(self):
        raw = '<tool_call>{"name": "weather", "arguments": {"city": "SF"}}'
        assert _recover_bare_scalar_from_raw(raw, expected_name="weather") is None

    def test_ignores_mismatched_name(self):
        raw = '<tool_call>{"name": "other", "arguments": "San Francisco"}'
        assert _recover_bare_scalar_from_raw(raw, expected_name="weather") is None

    def test_ignores_null(self):
        raw = '<tool_call>{"name": "weather", "arguments": null}'
        assert _recover_bare_scalar_from_raw(raw, expected_name="weather") is None

    def test_empty_or_none_returns_none(self):
        assert _recover_bare_scalar_from_raw(None) is None
        assert _recover_bare_scalar_from_raw("") is None

    def test_begin_end_marker_wire_format_recovers_scalar(self):
        # codex BLOCKING (round 3): the begin marker must not also be treated as
        # a closer, or no scalar inside this wire format would recover. The begin
        # (more) and end (尽) markers are the DeepSeek-style tool-call delimiters.
        begin = _BEGIN_MARKER
        end = _END_MARKER
        raw = begin + '{"name": "weather", "arguments": "SF"}' + end
        assert _recover_bare_scalar_from_raw(raw, expected_name="weather") == '"SF"'

    def test_rejects_colon_with_intervening_garbage(self):
        # codex BLOCKING #2: the key must be immediately followed by a colon.
        raw = '<tool_call>{"name": "x", "arguments" garbage, "foo": 7}'
        assert _recover_bare_scalar_from_raw(raw, expected_name="x") is None

    def test_rejects_malformed_value_prefixes(self):
        # codex BLOCKING: ``72oops`` / ``trueish`` must NOT be truncated to a
        # valid-looking scalar prefix.
        assert (
            _recover_bare_scalar_from_raw(
                '<tool_call>{"name": "x", "arguments": 72oops}', expected_name="x"
            )
            is None
        )
        assert (
            _recover_bare_scalar_from_raw(
                '<tool_call>{"name": "x", "arguments": trueish}', expected_name="x"
            )
            is None
        )
        # Trailing garbage after whitespace is also rejected.
        assert (
            _recover_bare_scalar_from_raw(
                '<tool_call>{"name": "x", "arguments": 72 oops}', expected_name="x"
            )
            is None
        )
        assert (
            _recover_bare_scalar_from_raw(
                '<tool_call>{"name": "x", "arguments": "SF" garbage}',
                expected_name="x",
            )
            is None
        )
        # codex round-4: a colon or stray angle-bracket is NOT a valid terminator.
        assert (
            _recover_bare_scalar_from_raw(
                '<tool_call>{"name": "x", "arguments": "SF": garbage}',
                expected_name="x",
            )
            is None
        )
        assert (
            _recover_bare_scalar_from_raw(
                '<tool_call>{"name": "x", "arguments": 72<junk}', expected_name="x"
            )
            is None
        )

    def test_multiple_matching_candidates_is_ambiguous_none(self):
        # codex round-4: two scalar candidates pairing with the SAME target name
        # is ambiguous — fail closed (None) rather than pick the last.
        raw = (
            '<tool_call>{"name": "weather", "arguments": "SF"}'
            '{"name": "weather", "arguments": "NY"}'
        )
        assert _recover_bare_scalar_from_raw(raw, expected_name="weather") is None
        # A single candidate still resolves.
        raw2 = '<tool_call>{"name": "weather", "arguments": "SF"}'
        assert _recover_bare_scalar_from_raw(raw2, expected_name="weather") == '"SF"'

    def test_name_pairing_decodes_json_escapes(self):
        # codex NIT: any valid JSON string escape decodes for the pairing check.
        raw = '<tool_call>{"name": "we\\u0061ther", "arguments": "SF"}'
        assert _recover_bare_scalar_from_raw(raw, expected_name="weather") == '"SF"'

    def test_pairing_uses_nearest_preceding_name_in_multi_call_span(self):
        # codex BLOCKING #1: two calls in one span — the second scalar must pair
        # with the SECOND tool ("other"), not the span's first ("weather").
        raw = (
            '<tool_call>{"name": "weather", "arguments": 1}'
            '{"name": "other", "arguments": "San Francisco"}'
        )
        # Searching for the FIRST tool's scalar yields nothing (only "other" has
        # a string scalar, and 1 is a number but pairs with weather... both here).
        recovered = _recover_bare_scalar_from_raw(raw, expected_name="other")
        assert recovered == '"San Francisco"'
        # Searching for the first tool yields ITS OWN numeric scalar "1" — the
        # names in a multi-call span must not cross-attach.
        assert _recover_bare_scalar_from_raw(raw, expected_name="weather") == "1"
        # A name that is not present pairs with nothing.
        assert _recover_bare_scalar_from_raw(raw, expected_name="nope") is None


class TestSalvageRejectsNonFinite:
    def test_non_finite_float_never_salvages(self):
        # codex BLOCKING #3: NaN/Infinity are not strict JSON — never emit them.
        assert _salvage_forced_scalar_arguments("temperature", "NaN", [_NUM]) is None
        assert (
            _salvage_forced_scalar_arguments("temperature", "Infinity", [_NUM]) is None
        )
        assert (
            _salvage_forced_scalar_arguments("temperature", "72.5", [_NUM])
            == '{"degrees": 72.5}'
        )
