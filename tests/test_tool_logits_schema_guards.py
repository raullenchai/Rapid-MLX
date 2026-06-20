# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for F-140 (retires F-031).

Three unguarded ``.get()``/``.items()`` calls in
``vllm_mlx/api/tool_logits.py`` at lines 42, 43, and 419 turned a wide
family of malformed tool-schema shapes into an unmapped
``AttributeError`` — surfacing as HTTP 500 (instead of a clean 400/422)
and, on the route layer's older envelope, leaking the raw Python error
text. F-031 was the narrowest subcase (``parameters: null``); F-140
covers the full ``≥7`` known crash shapes including non-dict
``parameters`` (list / scalar / string) and non-dict ``properties``.

The fix shape: ``isinstance(_, dict)`` guards at the three call sites
that previously assumed dict-shaped input. When the shape is wrong, the
function skips that tool / returns an empty schema map cleanly — no
exception leaves the helper.

These tests don't boot the server (CI-cheap); they pin the helper-level
contract so a regression can be caught without an integration harness.
"""

from __future__ import annotations

import pytest

from vllm_mlx.api.tool_logits import _extract_param_schemas, validate_param_value

# ---------------------------------------------------------------------------
# F-140 known crash shapes — must NOT raise AttributeError
# ---------------------------------------------------------------------------


def _tool(parameters):
    """Build a tool dict with the given ``parameters`` payload."""
    return {"type": "function", "function": {"name": "f", "parameters": parameters}}


# The seven shapes from the F-140 repro, plus three structural variants
# (top-level tool not-a-dict, ``function`` not-a-dict, missing
# ``function``).
F140_MALFORMED_SHAPES = [
    pytest.param([_tool(None)], id="parameters_null"),
    pytest.param([_tool("foo")], id="parameters_string"),
    pytest.param([_tool([])], id="parameters_empty_list"),
    pytest.param([_tool([1, 2, 3])], id="parameters_nonempty_list"),
    pytest.param([_tool(42)], id="parameters_int"),
    pytest.param([_tool(3.14)], id="parameters_float"),
    pytest.param([_tool(True)], id="parameters_bool"),
    pytest.param(
        [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"properties": []}},
            }
        ],
        id="properties_list",
    ),
    pytest.param(
        [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"properties": "foo"}},
            }
        ],
        id="properties_string",
    ),
    pytest.param(
        [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"properties": 42}},
            }
        ],
        id="properties_int",
    ),
    pytest.param(
        [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"properties": None}},
            }
        ],
        id="properties_null",
    ),
    # Structural variants — tool itself / function field is malformed.
    pytest.param(["not-a-dict"], id="tool_not_dict_str"),
    pytest.param([42], id="tool_not_dict_int"),
    pytest.param([None], id="tool_none"),
    pytest.param([{"type": "function", "function": None}], id="function_none"),
    pytest.param([{"type": "function", "function": "bogus"}], id="function_string"),
    pytest.param([{"type": "function", "function": []}], id="function_list"),
]


@pytest.mark.parametrize("tools", F140_MALFORMED_SHAPES)
def test_extract_param_schemas_tolerates_malformed_shapes(tools):
    """Helper must return an empty schema map cleanly, never raise."""
    result = _extract_param_schemas(tools)
    assert result == {}, f"expected empty schemas for {tools!r}, got {result!r}"


@pytest.mark.parametrize(
    "bad_schema",
    [
        pytest.param(None, id="schema_none"),
        pytest.param("foo", id="schema_string"),
        pytest.param([], id="schema_empty_list"),
        pytest.param([1, 2, 3], id="schema_nonempty_list"),
        pytest.param(42, id="schema_int"),
        pytest.param(3.14, id="schema_float"),
        pytest.param(True, id="schema_bool"),
    ],
)
def test_validate_param_value_tolerates_non_dict_schema(bad_schema):
    """
    ``validate_param_value`` ran ``schema.get("type", "")`` blindly at
    line 419, raising ``AttributeError`` for the same family of shapes.
    With the guard in place, treat non-dict schema as "no constraint"
    and pass the value through as valid.
    """
    is_valid, err = validate_param_value('"hello"', bad_schema)
    assert is_valid is True
    assert err is None


# ---------------------------------------------------------------------------
# Mixed-shape lists — partial malformation must not poison sibling entries
# ---------------------------------------------------------------------------


def test_extract_skips_bad_tool_keeps_good_one():
    """
    A malformed tool in position 0 must not prevent extraction of a
    well-formed tool in position 1 — guards ``continue`` per-tool, they
    don't ``return {}`` early.
    """
    tools = [
        _tool(None),  # malformed — should be skipped
        {
            "type": "function",
            "function": {
                "name": "ok",
                "parameters": {"properties": {"x": {"type": "string"}}},
            },
        },
    ]
    result = _extract_param_schemas(tools)
    assert result == {"ok.x": {"type": "string"}}


def test_extract_skips_bad_properties_keeps_good_tool():
    """``properties`` non-dict on one tool, good ``properties`` on another."""
    tools = [
        {
            "type": "function",
            "function": {"name": "bad", "parameters": {"properties": []}},
        },
        {
            "type": "function",
            "function": {
                "name": "good",
                "parameters": {"properties": {"y": {"type": "integer"}}},
            },
        },
    ]
    result = _extract_param_schemas(tools)
    assert result == {"good.y": {"type": "integer"}}


# ---------------------------------------------------------------------------
# Sanity — the well-formed path still works after adding guards
# ---------------------------------------------------------------------------


def test_extract_well_formed_unchanged():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "properties": {
                        "location": {"type": "string"},
                        "units": {"type": "string", "enum": ["c", "f"]},
                    }
                },
            },
        }
    ]
    result = _extract_param_schemas(tools)
    assert result == {
        "get_weather.location": {"type": "string"},
        "get_weather.units": {"type": "string", "enum": ["c", "f"]},
    }


def test_validate_well_formed_unchanged():
    is_valid, err = validate_param_value('"celsius"', {"type": "string"})
    assert is_valid is True
    assert err is None

    is_valid, err = validate_param_value("not-json", {"type": "integer"})
    assert is_valid is False
    assert err is not None
