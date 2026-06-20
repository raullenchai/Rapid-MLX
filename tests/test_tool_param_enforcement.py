# SPDX-License-Identifier: Apache-2.0
"""Regression tests for F-141 — scoped tool-param schema enforcement.

Before this fix, ``_validate_tool_call_params`` (in
``vllm_mlx/service/helpers.py``) was log-only: when the model emitted
arguments that violated the declared JSON schema for a tool, the engine
wrote a ``logger.warning(...)`` and returned the bad payload to the
client as a normal 200. F-141 ports the validator to ENFORCEMENT: a
schema violation now raises ``HTTPException(400)`` so the caller can
react (retry / fallback / surface to user) instead of silently shipping
a broken contract.

Scope this PR (see TODO.md F-141 partial-fixed entry):

  * ``enum``        — string value must be in the declared list
  * ``type``        — strict pydantic-style int/float/str/bool/array/object
  * ``minimum``     — numeric lower bound (integer / number)
  * ``maximum``     — numeric upper bound (integer / number)
  * ``minLength``   — string lower length bound
  * ``maxLength``   — string upper length bound

Explicitly DEFERRED (covered by the pass-through tests at the bottom
of this file to lock the boundary — over-enforcing these would
regress real-world tool schemas that ship loose ``pattern`` /
``format`` hints):

  * ``pattern``      — regex (TODO(F-141-followup))
  * ``format``       — email/date-time/uuid (TODO(F-141-followup))
  * ``multipleOf``   — TODO(F-141-followup)
  * ``uniqueItems``  — TODO(F-141-followup)
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from vllm_mlx.api.models import FunctionCall, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Enforcement — 4 violation classes that MUST 400
# ---------------------------------------------------------------------------


class TestEnforcement:
    """F-141 scoped fix — these four constraint classes are enforced."""

    def test_enum_violation_raises_400(self):
        """enum: model emits ``"purple"`` against ``enum:["red","green","blue"]``."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "set_color",
                {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
            )
        ]
        calls = [_call("set_color", '{"color": "purple"}')]

        with pytest.raises(HTTPException) as exc_info:
            _validate_tool_call_params(calls, tools)
        assert exc_info.value.status_code == 400
        # The detail must name the offending field so the client can
        # surface a useful error to the end user.
        assert "color" in exc_info.value.detail
        assert "purple" in exc_info.value.detail or "enum" in exc_info.value.detail

    def test_type_violation_raises_400(self):
        """type: schema says ``integer``, model emits string ``"twentyfive"``."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("set_age", {"age": {"type": "integer"}})]
        calls = [_call("set_age", '{"age": "twentyfive"}')]

        with pytest.raises(HTTPException) as exc_info:
            _validate_tool_call_params(calls, tools)
        assert exc_info.value.status_code == 400
        assert "age" in exc_info.value.detail
        assert "integer" in exc_info.value.detail.lower()

    def test_range_violation_raises_400(self):
        """minimum/maximum: ``score=200`` against ``maximum:100``."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "set_score",
                {"score": {"type": "integer", "minimum": 0, "maximum": 100}},
            )
        ]
        calls = [_call("set_score", '{"score": 200}')]

        with pytest.raises(HTTPException) as exc_info:
            _validate_tool_call_params(calls, tools)
        assert exc_info.value.status_code == 400
        assert "score" in exc_info.value.detail
        assert "maximum" in exc_info.value.detail or "100" in exc_info.value.detail

    def test_min_below_floor_raises_400(self):
        """Symmetric direction: ``score=-5`` against ``minimum:0``."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "set_score",
                {"score": {"type": "integer", "minimum": 0, "maximum": 100}},
            )
        ]
        calls = [_call("set_score", '{"score": -5}')]

        with pytest.raises(HTTPException) as exc_info:
            _validate_tool_call_params(calls, tools)
        assert exc_info.value.status_code == 400
        assert "minimum" in exc_info.value.detail or "score" in exc_info.value.detail

    def test_length_violation_raises_400(self):
        """minLength: ``username="bob"`` (len 3) against ``minLength:5``."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "set_username",
                {"username": {"type": "string", "minLength": 5, "maxLength": 20}},
            )
        ]
        calls = [_call("set_username", '{"username": "bob"}')]

        with pytest.raises(HTTPException) as exc_info:
            _validate_tool_call_params(calls, tools)
        assert exc_info.value.status_code == 400
        assert "username" in exc_info.value.detail
        assert (
            "minLength" in exc_info.value.detail
            or "length" in exc_info.value.detail.lower()
        )

    def test_max_length_violation_raises_400(self):
        """maxLength: 30-char username against ``maxLength:20``."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        long_name = "x" * 30
        tools = [
            _tool(
                "set_username",
                {"username": {"type": "string", "minLength": 1, "maxLength": 20}},
            )
        ]
        calls = [_call("set_username", f'{{"username": "{long_name}"}}')]

        with pytest.raises(HTTPException) as exc_info:
            _validate_tool_call_params(calls, tools)
        assert exc_info.value.status_code == 400
        assert "maxLength" in exc_info.value.detail or "20" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Union types — `{"type": ["string", "null"]}` must be honoured
# ---------------------------------------------------------------------------


class TestUnionTypes:
    """Codex round 2 BLOCKING on PR #736 — JSON-Schema lets ``type`` be
    a list, e.g. ``{"type": ["string", "null"]}``. The original branch
    tree skipped every check because ``param_type`` was a list, so an
    integer slipped past for a string-or-null schema. These tests lock
    the union-type path."""

    def test_union_string_or_null_accepts_string(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("nick", {"name": {"type": ["string", "null"]}})]
        _validate_tool_call_params([_call("nick", '{"name": "alice"}')], tools)

    def test_union_string_or_null_accepts_null(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("nick", {"name": {"type": ["string", "null"]}})]
        _validate_tool_call_params([_call("nick", '{"name": null}')], tools)

    def test_union_string_or_null_rejects_integer(self):
        """Integer should NOT match a ``["string", "null"]`` union."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("nick", {"name": {"type": ["string", "null"]}})]
        with pytest.raises(HTTPException) as exc_info:
            _validate_tool_call_params([_call("nick", '{"name": 123}')], tools)
        assert exc_info.value.status_code == 400
        assert "name" in exc_info.value.detail
        # Should mention the union members or the offending int type.
        detail = exc_info.value.detail
        assert "string" in detail or "null" in detail or "int" in detail


# ---------------------------------------------------------------------------
# Valid baseline — must NOT raise (proves enforcement isn't over-zealous)
# ---------------------------------------------------------------------------


class TestValidPasses:
    """A perfectly schema-conformant call must continue to succeed —
    locks in the upper bound of enforcement so we don't silently start
    rejecting good payloads."""

    def test_valid_enum_passes(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "set_color",
                {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
            )
        ]
        calls = [_call("set_color", '{"color": "red"}')]
        _validate_tool_call_params(calls, tools)  # no raise

    def test_valid_all_constraints_pass(self):
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "register",
                {
                    "username": {
                        "type": "string",
                        "minLength": 3,
                        "maxLength": 20,
                    },
                    "age": {"type": "integer", "minimum": 0, "maximum": 150},
                    "role": {"type": "string", "enum": ["admin", "user"]},
                },
            )
        ]
        calls = [
            _call(
                "register",
                '{"username": "alice", "age": 30, "role": "admin"}',
            )
        ]
        _validate_tool_call_params(calls, tools)  # no raise

    def test_boundary_inclusive_passes(self):
        """JSON Schema ``minimum``/``maximum`` are inclusive."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "set_score",
                {"score": {"type": "integer", "minimum": 0, "maximum": 100}},
            )
        ]
        _validate_tool_call_params([_call("set_score", '{"score": 0}')], tools)
        _validate_tool_call_params([_call("set_score", '{"score": 100}')], tools)


# ---------------------------------------------------------------------------
# Deferred — `pattern` and `format` MUST still pass-through (NOT enforced)
# ---------------------------------------------------------------------------


class TestDeferredPassThrough:
    """Constraint classes still intentionally left advisory after the
    F-141a follow-up. Locking the pass-through behaviour as regression
    tests ensures a future tightening is a deliberate, reviewed change
    rather than an accidental side-effect of touching
    ``validate_param_value``.

    F-141a (PR following #736) flipped ``pattern`` / ``format`` /
    ``multipleOf`` / ``uniqueItems`` to enforcement — those have moved
    to ``tests/test_tool_param_enforcement_f141a.py``. The remaining
    deferred set (``required``, ``oneOf``, ``anyOf``, ``allOf``,
    ``additionalProperties``) still pass-through.
    """

    def test_required_missing_key_does_not_raise(self):
        """``required`` violation: schema requires ``"x"`` but the model
        emitted ``{}``. Compound semantics — deferred for now."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [
            _tool(
                "f",
                {"x": {"type": "string"}},
            )
        ]
        # Inject ``required`` into the parameters block.
        tools[0]["function"]["parameters"]["required"] = ["x"]
        _validate_tool_call_params([_call("f", "{}")], tools)

    def test_one_of_violation_does_not_raise(self):
        """``oneOf`` (and ``anyOf`` / ``allOf``) violations remain
        advisory — multi-branch schema traversal is a separate lift."""
        from vllm_mlx.service.helpers import _validate_tool_call_params

        tools = [_tool("f", {"x": {"oneOf": [{"const": "a"}, {"const": "b"}]}})]
        _validate_tool_call_params([_call("f", '{"x": "c"}')], tools)
