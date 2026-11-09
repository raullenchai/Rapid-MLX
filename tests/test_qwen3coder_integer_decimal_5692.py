# SPDX-License-Identifier: Apache-2.0
"""Integral float literals under integer schemas must survive tool validation."""

import json

import pytest

from rapid_mlx.api.tool_calling import validate_json_schema
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import _convert_param_value
from tests.qwen3coder_stream_harness import non_stream, stream


@pytest.mark.parametrize(
    "schema",
    [
        {"type": name}
        for name in (
            "integer",
            "int",
            "uint32",
            "long",
            "short",
            "unsigned",
            " INTEGER ",
        )
    ]
    + [
        {"type": ["integer", "null"]},
        {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        {"oneOf": [{"type": "null"}, {"type": "integer"}]},
    ],
)
@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("680", 680),
        ("680.0", 680),
        ("30.0", 30),
        (" 680.0 ", 680),
        ("-680.0", -680),
        ("+680.0", 680),
        ("-0.0", 0),
        ("6.8e2", 680),
        ("9007199254740993", 9007199254740993),
        ("680.5", "680.5"),
        (" 680.5 ", " 680.5 "),
        ("-680.5", "-680.5"),
        ("1e-3", "1e-3"),
        ("NaN", "NaN"),
        ("inf", "inf"),
        ("-Infinity", "-Infinity"),
        ("1e999", "1e999"),
        ("abc", "abc"),
        ("", ""),
    ],
)
def test_integer_conversion(schema, literal, expected):
    value = _convert_param_value(literal, "offset", {"offset": schema}, "Read")
    assert type(value) is type(expected)
    assert value == expected


@pytest.mark.parametrize("size", [0, 1, 2, 7, 10_000])
@pytest.mark.parametrize(
    ("literal", "expected"),
    [("680.0", 680), ("30.0", 30), ("6.8e2", 680), ("680.5", "680.5")],
)
def test_read_tool_streaming_and_non_streaming_validation(literal, expected, size):
    schema = {
        "type": "object",
        "properties": {"offset": {"type": "integer"}},
        "required": ["offset"],
    }
    request = {
        "tools": [
            {"type": "function", "function": {"name": "Read", "parameters": schema}}
        ]
    }
    wire = (
        "<tool_call>\n<function=Read>\n<parameter=offset>"
        + literal
        + "</parameter>\n</function>\n</tool_call>"
    )
    calls = [("Read", json.dumps({"offset": expected}))]
    assert non_stream(wire, request) == (calls, "")
    assert stream(wire, request, size) == (calls, "")
    valid, error = validate_json_schema(json.loads(calls[0][1]), schema)
    assert valid is (type(expected) is int)
    if not valid:
        assert "integer" in error
