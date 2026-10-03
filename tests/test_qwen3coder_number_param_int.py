# SPDX-License-Identifier: Apache-2.0
"""Integer literals for ``type: number`` parameters stay integers.

A schema typing a parameter as ``number`` admits integers. Converting
``10000`` to ``10000.0`` makes strictly typed callers (Rust ``usize``,
Go ``int``) reject the tool call. Integer literals must stay ``int``;
fractional or exponent forms remain ``float``.
"""

import pytest

from rapid_mlx.tool_parsers.qwen3coder_tool_parser import _convert_param_value

SCHEMA = {
    "max_output_tokens": {"type": "number"},
    "ratio": {"type": "number"},
    "count": {"type": "integer"},
}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("10000", 10000),
        ("-7", -7),
        ("0", 0),
        ("  42  ", 42),
    ],
)
def test_number_param_integer_literal_returns_int(raw, expected):
    value = _convert_param_value(raw, "max_output_tokens", SCHEMA, "exec_command")
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.5", 0.5),
        ("-2.25", -2.25),
        ("1e3", 1000.0),
    ],
)
def test_number_param_fractional_literal_stays_float(raw, expected):
    value = _convert_param_value(raw, "ratio", SCHEMA, "f")
    assert value == expected
    assert type(value) is float


def test_integer_param_still_returns_int():
    value = _convert_param_value("9", "count", SCHEMA, "f")
    assert value == 9 and type(value) is int


def test_number_param_non_numeric_falls_back_to_raw_text():
    assert _convert_param_value("abc", "ratio", SCHEMA, "f") == "abc"
