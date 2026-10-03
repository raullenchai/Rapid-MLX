# SPDX-License-Identifier: Apache-2.0
"""Issue #4037: qwen3_coder_xml keeps integer literals under ``type: number``.

Codex rejected ``{"max_output_tokens": 10000.0}`` ("invalid type: floating
point `10000.0`, expected usize") because the parser forced every ``number``
value through ``float()``. A ``number`` literal with no fraction and no
exponent is now emitted as a JSON integer. Every other (schema, literal) pair
is pinned to origin/main's output by a golden file captured from 9221ff773,
and streaming must produce the same arguments as non-streaming at any chunk
granularity.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from rapid_mlx.api.tool_calling import _schema_type
from tests.qwen3coder_stream_harness import non_stream, stream

GOLDEN = json.loads(
    (
        pathlib.Path(__file__).parent
        / "fixtures"
        / "qwen3coder_scalar_args_golden_main.json"
    ).read_text()
)["cases"]

_INTEGER = re.compile(r"[+-]?[0-9]+")


def _changed_by_4037(case: dict) -> bool:
    schema_type = _schema_type(case["schema"]) or ""
    return schema_type.startswith("num") and bool(
        _INTEGER.fullmatch(case["literal"].strip())
    )


def _request(schema: dict) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {"v": schema}},
                },
            }
        ]
    }


def _non_streaming(wire: str, request: dict) -> str | None:
    calls, _ = non_stream(wire, request)
    return calls[0][1] if calls else None


@pytest.mark.parametrize(
    "case",
    GOLDEN,
    ids=[f"{c['schema_key']}-{c['literal']!r}" for c in GOLDEN],
)
def test_non_streaming_matches_main_except_integer_numbers(case):
    got = _non_streaming(case["wire"], _request(case["schema"]))
    if not _changed_by_4037(case):
        assert got == case["arguments"]
        return
    main_value = json.loads(case["arguments"])["v"]
    value = json.loads(got)["v"]
    assert isinstance(main_value, float)
    assert type(value) is int
    assert value == int(case["literal"].strip())
    assert got == json.dumps({"v": value})


def test_the_golden_actually_covers_the_change():
    changed = [c for c in GOLDEN if _changed_by_4037(c)]
    # 9 integer literals (incl. padded, signed, 007, 20 digits) x 5 schemas.
    assert len(changed) == 45
    assert {c["schema_key"] for c in changed} == {
        "number",
        "number_upper",
        "numeric",
        "number_nullable",
        "number_anyof",
    }


@pytest.mark.parametrize("size", [0, 1, 2, 3, 5, 8, 13, 10_000])
@pytest.mark.parametrize(
    "case",
    [c for c in GOLDEN if c["schema_key"] in ("number", "float", "integer", "string")],
    ids=lambda c: f"{c['schema_key']}-{c['literal']!r}",
)
def test_streaming_agrees_with_non_streaming(case, size):
    request = _request(case["schema"])
    assert stream(case["wire"], request, size) == non_stream(case["wire"], request)


CODEX_EXEC = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "max_output_tokens": {"type": "number"},
                        "yield_time_ms": {"type": "number"},
                        "ratio": {"type": "number"},
                    },
                    "required": ["cmd"],
                },
            },
        }
    ]
}
CODEX_WIRE = (
    "I'll run the tests.\n\n<tool_call>\n<function=exec_command>\n"
    "<parameter=cmd>\n/work/.venv/bin/pytest 2>&1\n</parameter>\n"
    "<parameter=max_output_tokens>\n10000\n</parameter>\n"
    "<parameter=yield_time_ms>\n1e3\n</parameter>\n"
    "<parameter=ratio>\n0.5\n</parameter>\n"
    "</function>\n</tool_call>"
)
CODEX_ARGS = (
    '{"cmd": "/work/.venv/bin/pytest 2>&1", "max_output_tokens": 10000, '
    '"yield_time_ms": 1000.0, "ratio": 0.5}'
)


def test_codex_exec_command_shape_non_streaming():
    assert non_stream(CODEX_WIRE, CODEX_EXEC) == (
        [("exec_command", CODEX_ARGS)],
        "I'll run the tests.\n\n",
    )


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 7, 16, 10_000])
def test_codex_exec_command_shape_streaming(size):
    assert stream(CODEX_WIRE, CODEX_EXEC, size) == (
        [("exec_command", CODEX_ARGS)],
        "I'll run the tests.\n\n",
    )


@pytest.mark.parametrize("size", [0, 7, 10_000])
def test_integer_past_the_int_string_limit_keeps_main_float(size):
    """Python 3.11+ refuses int() on > 4300 digits; there such a literal
    keeps origin/main's float() conversion instead of failing the call."""
    literal = "9" * 5000
    request = _request({"type": "number"})
    wire = (
        "<tool_call>\n<function=f>\n<parameter=v>\n"
        + literal
        + "\n</parameter>\n</function>\n</tool_call>"
    )
    try:  # Python 3.10 has no int-string digit limit
        value: int | float = int(literal)
    except ValueError:
        value = float(literal)
    expected = json.dumps({"v": value})
    assert non_stream(wire, request) == ([("f", expected)], "")
    assert stream(wire, request, size) == ([("f", expected)], "")
