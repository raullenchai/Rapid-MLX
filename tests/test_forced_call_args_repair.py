# SPDX-License-Identifier: Apache-2.0
"""Forced ``tool_choice`` calls must carry OBJECT arguments on the wire.

Dogfood 2026-08-12 (post #1866/#1874/#1878 merge, pre-existing on main):
the hermes forced-choice prefix hands the model ``..."arguments": ``
mid-envelope and Qwen3.5-35B-A3B-8bit at temp 0 continues with a bare
``1`` — the parser surfaces ``arguments="1"``, and every OpenAI client
``json.loads``\\ es a non-dict and breaks. The repair mirrors the #571
synthesis fallback (recover from raw text, else ``"{}"``) and holds the
result to the #1256 schema gate.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rapid_mlx.routes.chat import _repair_forced_call_arguments


def _call(arguments):
    return SimpleNamespace(
        id="call_x",
        type="function",
        function=SimpleNamespace(name="release_probe", arguments=arguments),
    )


def _tool(name: str = "release_probe", required: list[str] | None = None):
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    if required is not None:
        schema["required"] = required
    return SimpleNamespace(
        type="function", function={"name": name, "parameters": schema}
    )


def test_int_arguments_repaired_to_empty_object():
    tc = _call("1")
    err = _repair_forced_call_arguments([tc], "", "release_probe", [_tool()])
    assert err is None
    assert tc.function.arguments == "{}"


def test_valid_object_arguments_untouched():
    tc = _call('{"note": "ok"}')
    err = _repair_forced_call_arguments([tc], "", "release_probe", [_tool()])
    assert err is None
    assert tc.function.arguments == '{"note": "ok"}'


def test_unparseable_arguments_repaired():
    tc = _call("not json at all {")
    err = _repair_forced_call_arguments([tc], "", "release_probe", [_tool()])
    assert err is None
    assert tc.function.arguments == "{}"


def test_recovery_from_raw_text_preferred_over_empty():
    import json

    tc = _call("1")
    raw = '<tool_call>{"name": "release_probe", "arguments": {"note": "hi"}}'
    err = _repair_forced_call_arguments([tc], raw, "release_probe", [_tool()])
    assert err is None
    assert json.loads(tc.function.arguments) == {"note": "hi"}


def test_dict_arguments_normalized_to_json_string():
    """codex r3: a dict-typed arguments value is valid CONTENT but the
    wrong wire SHAPE — normalize to a JSON-encoded string, preserving
    the content, never discarding it."""
    import json

    tc = _call({"note": "keep me"})
    err = _repair_forced_call_arguments([tc], "", "release_probe", [_tool()])
    assert err is None
    assert isinstance(tc.function.arguments, str)
    assert json.loads(tc.function.arguments) == {"note": "keep me"}


def test_recovery_skipped_when_valid_sibling_call_present():
    """codex r3: recovery must not lift a VALID sibling call's
    arguments into the broken one — with any other call present, the
    broken call repairs to {}."""
    import json

    good = _call('{"note": "mine"}')
    bad = _call("1")
    raw = '<tool_call>{"name": "release_probe", "arguments": {"note": "mine"}}'
    err = _repair_forced_call_arguments([good, bad], raw, "release_probe", [_tool()])
    assert err is None
    assert json.loads(good.function.arguments) == {"note": "mine"}
    assert json.loads(bad.function.arguments) == {}


def test_multiple_broken_calls_never_share_recovered_args():
    """codex r2: raw-text recovery is unambiguous only for a single
    broken call — several must all repair to {} rather than every one
    receiving the same first recovered object."""
    import json

    a, b = _call("1"), _call("2")
    raw = '<tool_call>{"name": "release_probe", "arguments": {"note": "hi"}}'
    err = _repair_forced_call_arguments([a, b], raw, "release_probe", [_tool()])
    assert err is None
    assert json.loads(a.function.arguments) == {}
    assert json.loads(b.function.arguments) == {}


def test_wire_envelope_names_decodes_escapes_and_ignores_prose():
    """codex r2 on #1880: envelope decoding (not a literal regex) so
    escaped names cannot evade the mismatch gate and prose outside
    envelopes contributes nothing."""
    from rapid_mlx.routes.chat import _wire_envelope_names

    # Escaped different-tool name decodes and mismatches.
    raw = '<tool_call>{"name": "other\\u005ftool", "arguments": 1}</tool_call>'
    assert _wire_envelope_names(raw) == ["other_tool"]
    # Prose mentioning "name": "release_probe" outside an envelope is inert.
    raw2 = 'the "name": "release_probe" field... <tool_call>{"name": "x", "arguments": 1}</tool_call>'
    assert _wire_envelope_names(raw2) == ["x"]
    # The live dogfood shape: valid JSON, scalar arguments, missing closer.
    raw3 = '<tool_call>\n{"name": "release_probe", "arguments": 1}'
    assert _wire_envelope_names(raw3) == ["release_probe"]
    # Undecodable envelope refuses (None), never permits.
    assert _wire_envelope_names("<tool_call>garbage no brace") is None
    # Multiple same-target envelopes surface as multiple names — the
    # streaming gate requires exactly one, since synthesis emits a
    # single call and would otherwise collapse several (codex r4).
    raw4 = (
        '<tool_call>{"name": "release_probe", "arguments": 1}</tool_call>'
        '<tool_call>{"name": "release_probe", "arguments": 2}</tool_call>'
    )
    assert _wire_envelope_names(raw4) == ["release_probe", "release_probe"]


def test_schema_required_violation_surfaces_error():
    """The repair must not fabricate a passing call when the tool's
    schema requires properties the repaired arguments cannot provide —
    same #1256 contract as the synthesis fallback."""
    tc = _call("1")
    err = _repair_forced_call_arguments(
        [tc], "", "release_probe", [_tool(required=["level"])]
    )
    assert err is not None
    assert "level" in err or "required" in err.lower()
