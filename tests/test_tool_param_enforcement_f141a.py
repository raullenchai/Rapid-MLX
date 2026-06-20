# SPDX-License-Identifier: Apache-2.0
"""F-141a: extend tool-arg validator with ``pattern`` / ``format`` /
``multipleOf`` / ``uniqueItems``.

PR #736 (F-141 scoped) enforced ``enum`` / ``type`` / ``minimum`` /
``maximum`` / ``minLength`` / ``maxLength`` by raising
``HTTPException(400)`` on violation. ``pattern``, ``format``,
``multipleOf`` and ``uniqueItems`` were deferred. F-141a closes that
gap — these constraints now also escalate to a 400 with the same
canonical "violates declared schema" envelope.

Format scope (operator note): ``email`` / ``uri`` / ``uuid`` /
``date`` / ``date-time``. Unknown ``format`` values pass through
(advisory) to stay loose for real-world schemas.

Pattern semantics: ``re.fullmatch``, not ``re.match`` — partial
matches do not count.

multipleOf: integer arithmetic for int values; ``math.isclose`` for
float values so a 0.1-vs-0.3 float-drift case doesn't 400.

uniqueItems: structural equality via JSON-canonical serialisation.
"""

from __future__ import annotations

import pytest

# Linux CI runners don't ship MLX; ``vllm_mlx.service.helpers``
# transitively imports it through the engine wiring. Skip cleanly so the
# diff-aware targeted_tests step doesn't flag the whole file as
# regressions. Same pattern as ``tests/test_audio_upload_size_limit.py``.
pytest.importorskip(
    "mlx.core",
    reason="tool-arg validator helper imports transitively pull in mlx",
)
pytest.importorskip(
    "mlx_lm",
    reason="tool-arg validator helper imports transitively pull in mlx_lm",
)

from fastapi import HTTPException  # noqa: E402

from vllm_mlx.api.models import FunctionCall, ToolCall  # noqa: E402


def _tool(name: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": properties},
        },
    }


def _call(name: str, arguments: str) -> ToolCall:
    return ToolCall(
        id="call_abc",
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


# ---------------------------------------------------------------------------
# Pattern enforcement
# ---------------------------------------------------------------------------


class TestPatternEnforcement:
    """``pattern`` (regex) — ``re.fullmatch`` per operator note."""

    def test_pattern_violation_raises_400(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "book",
                {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
            )
        ]
        with pytest.raises(HTTPException) as exc:
            _validate_tool_call_params([_call("book", '{"date": "tomorrow"}')], tools)
        assert exc.value.status_code == 400
        assert "does not match pattern" in exc.value.detail

    def test_pattern_partial_match_is_rejected(self):
        """``re.fullmatch``, not ``re.match`` — ``"2024-01-01x"`` must
        still 400 even though the prefix matches the regex."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "book",
                {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
            )
        ]
        with pytest.raises(HTTPException):
            _validate_tool_call_params(
                [_call("book", '{"date": "2024-01-01x"}')], tools
            )

    def test_pattern_valid_passes(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "book",
                {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
            )
        ]
        _validate_tool_call_params([_call("book", '{"date": "2024-12-31"}')], tools)

    def test_bogus_regex_in_schema_does_not_raise(self):
        """If the *schema* ships an invalid regex (``[unclosed``), the
        bug is the schema author's, not the model's. Fall back to
        advisory pass-through rather than 400-ing on every call."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"type": "string", "pattern": "[unclosed"}})]
        _validate_tool_call_params([_call("f", '{"x": "anything"}')], tools)


# ---------------------------------------------------------------------------
# Format enforcement
# ---------------------------------------------------------------------------


class TestFormatEnforcement:
    """``format`` — narrow allow-list per operator scope."""

    @pytest.mark.parametrize(
        "fmt, bad, good",
        [
            ("email", "notanemail", "user@example.com"),
            ("uri", "no scheme here", "https://example.com/path"),
            (
                "uuid",
                "not-a-uuid",
                "550e8400-e29b-41d4-a716-446655440000",
            ),
            ("date", "tomorrow", "2024-01-15"),
            ("date-time", "2024-01-15", "2024-01-15T10:30:00Z"),
        ],
    )
    def test_format_enforced(self, fmt: str, bad: str, good: str):
        """Each scoped format rejects the obvious bad value and
        accepts the canonical good value."""
        import json as _json

        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"type": "string", "format": fmt}})]

        with pytest.raises(HTTPException) as exc:
            _validate_tool_call_params([_call("f", _json.dumps({"x": bad}))], tools)
        assert exc.value.status_code == 400
        assert f"not a valid {fmt}" in exc.value.detail

        # Good value must NOT raise.
        _validate_tool_call_params([_call("f", _json.dumps({"x": good}))], tools)

    def test_unknown_format_passes_through(self):
        """Operator scope: unknown ``format`` values stay loose (200)
        rather than 400-ing on every bespoke format hint in the wild."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"type": "string", "format": "my-custom-format"}})]
        _validate_tool_call_params([_call("f", '{"x": "anything"}')], tools)

    def test_date_time_requires_timezone_offset(self):
        """codex r2 BLOCKING: RFC 3339 ``date-time`` requires a
        timezone offset (``Z`` or ``±HH:MM``). A naive datetime
        like ``2024-01-15T10:30:00`` (no tz) used to slip past
        ``datetime.fromisoformat`` and pass as 200."""
        import json as _json

        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"type": "string", "format": "date-time"}})]
        with pytest.raises(HTTPException):
            _validate_tool_call_params(
                [_call("f", _json.dumps({"x": "2024-01-15T10:30:00"}))],
                tools,
            )
        # +00:00 explicit offset must pass.
        _validate_tool_call_params(
            [_call("f", _json.dumps({"x": "2024-01-15T10:30:00+00:00"}))],
            tools,
        )


# ---------------------------------------------------------------------------
# multipleOf enforcement
# ---------------------------------------------------------------------------


class TestMultipleOfEnforcement:
    """``multipleOf`` — integer / float."""

    def test_int_multiple_of_violation_raises_400(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("step", {"n": {"type": "integer", "multipleOf": 3}})]
        with pytest.raises(HTTPException) as exc:
            _validate_tool_call_params([_call("step", '{"n": 7}')], tools)
        assert exc.value.status_code == 400
        assert "not a multiple of 3" in exc.value.detail

    def test_int_multiple_of_pass(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("step", {"n": {"type": "integer", "multipleOf": 3}})]
        _validate_tool_call_params([_call("step", '{"n": 9}')], tools)

    def test_float_multiple_of_with_isclose_drift_pass(self):
        """0.3 = 3 * 0.1 in math, but ``0.3 % 0.1 == 0.09999...`` in
        float. Operator note: use ``math.isclose`` so this case passes."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"type": "number", "multipleOf": 0.1}})]
        _validate_tool_call_params([_call("f", '{"x": 0.3}')], tools)

    def test_float_multiple_of_violation_raises(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"type": "number", "multipleOf": 0.5}})]
        with pytest.raises(HTTPException):
            _validate_tool_call_params([_call("f", '{"x": 0.7}')], tools)

    def test_zero_multiple_of_treated_as_advisory(self):
        """``multipleOf: 0`` is invalid per JSON-schema. Bogus schema,
        fall back to advisory rather than 400-ing every call."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"type": "integer", "multipleOf": 0}})]
        _validate_tool_call_params([_call("f", '{"x": 5}')], tools)


# ---------------------------------------------------------------------------
# uniqueItems enforcement
# ---------------------------------------------------------------------------


class TestUniqueItemsEnforcement:
    """``uniqueItems`` — array dedup."""

    def test_duplicate_strings_raises_400(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "tags",
                {"items": {"type": "array", "uniqueItems": True}},
            )
        ]
        with pytest.raises(HTTPException) as exc:
            _validate_tool_call_params(
                [_call("tags", '{"items": ["a", "b", "a"]}')], tools
            )
        assert exc.value.status_code == 400
        assert "uniqueItems" in exc.value.detail

    def test_unique_strings_pass(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "tags",
                {"items": {"type": "array", "uniqueItems": True}},
            )
        ]
        _validate_tool_call_params([_call("tags", '{"items": ["a", "b", "c"]}')], tools)

    def test_structurally_equal_dicts_count_as_duplicate(self):
        """JSON-schema uniqueItems uses structural equality, not Python
        identity. ``{"a": 1}`` and ``{"a": 1}`` are duplicates."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "things",
                {"items": {"type": "array", "uniqueItems": True}},
            )
        ]
        with pytest.raises(HTTPException):
            _validate_tool_call_params(
                [_call("things", '{"items": [{"a": 1}, {"a": 1}]}')], tools
            )

    def test_unique_items_false_skips_check(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "tags",
                {"items": {"type": "array", "uniqueItems": False}},
            )
        ]
        _validate_tool_call_params([_call("tags", '{"items": ["a", "a"]}')], tools)

    def test_numerically_equal_int_and_float_count_as_duplicate(self):
        """codex r2 BLOCKING: JSON-schema uniqueItems compares numeric
        *value*, not representation. ``[1, 1.0]`` must be rejected as
        a duplicate. Previously ``json.dumps`` keying made them
        distinct and silently let it pass."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "nums",
                {"items": {"type": "array", "uniqueItems": True}},
            )
        ]
        with pytest.raises(HTTPException):
            _validate_tool_call_params([_call("nums", '{"items": [1, 1.0]}')], tools)

    def test_unique_numbers_pass(self):
        """Negative control for the above."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "nums",
                {"items": {"type": "array", "uniqueItems": True}},
            )
        ]
        _validate_tool_call_params([_call("nums", '{"items": [1, 2, 3.5]}')], tools)

    def test_large_distinct_integers_are_not_collapsed(self):
        """codex r3 BLOCKING: ``float(9007199254740993) ==
        float(9007199254740992)`` because IEEE-754 double can't
        represent both. The canonical-key path must use ``Decimal``
        so genuinely distinct large ints stay distinct."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "nums",
                {"items": {"type": "array", "uniqueItems": True}},
            )
        ]
        # 2**53 and 2**53 + 1 — both exact ints, distinct values.
        _validate_tool_call_params(
            [_call("nums", '{"items": [9007199254740992, 9007199254740993]}')],
            tools,
        )


class TestMultipleOfPrecision:
    """codex r3 BLOCKING: ``math.isclose(..., rel_tol=0.0,
    abs_tol=1e-9)`` so large non-multiples don't slip through on
    relative-tolerance grounds."""

    def test_large_non_multiple_is_rejected(self):
        """``rel_tol`` default would have accepted this — pinning it
        to 0.0 means the absolute tolerance is the only allowance."""
        import json as _json

        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"type": "number", "multipleOf": 0.5}})]
        # Far above the absolute tolerance, but the default relative
        # tolerance (1e-9) was nearly enough to make this slip.
        with pytest.raises(HTTPException):
            _validate_tool_call_params(
                [_call("f", _json.dumps({"x": 10000000000.3}))], tools
            )
