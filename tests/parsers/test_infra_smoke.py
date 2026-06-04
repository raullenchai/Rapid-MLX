# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the parser-test infrastructure itself.

Locks the per-delta invariants, the dispatcher signatures, and the
round-trip property of the token-delta splitters. The actual
regression suites (``tests/parsers/regressions/test_issue_*``) build
on these helpers, so a regression here invalidates every downstream
test — keep this file passing.
"""

from __future__ import annotations

import pytest

from vllm_mlx.reasoning.base import DeltaMessage

from ._harmony_markers import (
    HARMONY_CONTROL_TOKENS,
    HARMONY_LEAK_MARKERS,
    assert_no_harmony_marker_leak,
)
from .streaming_reconstructor import (
    ReconstructedToolCall,
    StreamingReasoningReconstructor,
    StreamingToolReconstructor,
)
from .token_delta_splitter import batch_deltas_with_stream_interval

# ----- StreamingReasoningReconstructor -----------------------------------


def test_reasoning_reconstructor_accumulates_separately():
    rec = StreamingReasoningReconstructor()
    rec.append_delta(DeltaMessage(reasoning="step 1, "))
    rec.append_delta(DeltaMessage(reasoning="step 2."))
    rec.append_delta(DeltaMessage(content="answer "))
    rec.append_delta(DeltaMessage(content="here"))
    assert rec.reasoning == "step 1, step 2."
    assert rec.other_content == "answer here"


def test_reasoning_reconstructor_rejects_both_populated():
    rec = StreamingReasoningReconstructor()
    with pytest.raises(AssertionError, match="leaked"):
        rec.append_delta(DeltaMessage(reasoning="r", content="c"))


def test_reasoning_reconstructor_ignores_empty_role_only_delta():
    rec = StreamingReasoningReconstructor()
    rec.append_delta(DeltaMessage(role="assistant"))
    assert rec.reasoning is None
    assert rec.other_content is None


# ----- StreamingToolReconstructor -----------------------------------------


def _tool_delta(
    index: int = 0,
    *,
    name: str | None = None,
    args: str = "",
    id: str | None = None,
    content: str | None = None,
):
    """Build a Rapid-MLX-shaped streaming tool delta."""
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    call: dict = {"index": index, "function": {}}
    if id is not None:
        call["id"] = id
    if name is not None:
        call["function"]["name"] = name
    if args:
        call["function"]["arguments"] = args
    # Only attach tool_calls if at least one tool-call field is set.
    if id is not None or name is not None or args:
        delta["tool_calls"] = [call]
    return delta


def test_tool_reconstructor_accumulates_arguments():
    rec = StreamingToolReconstructor()
    rec.append_delta(_tool_delta(id="call_1", name="get_weather"))
    rec.append_delta(_tool_delta(args='{"city"'))
    rec.append_delta(_tool_delta(args=': "NYC"}'))
    assert rec.tool_calls == [
        ReconstructedToolCall(
            id="call_1", name="get_weather", arguments='{"city": "NYC"}'
        )
    ]
    assert rec.other_content == ""


def test_tool_reconstructor_rejects_duplicate_id():
    rec = StreamingToolReconstructor()
    rec.append_delta(_tool_delta(id="call_1", name="f"))
    with pytest.raises(AssertionError, match="id only once"):
        rec.append_delta(_tool_delta(id="call_1", args="x"))


def test_tool_reconstructor_rejects_duplicate_name():
    rec = StreamingToolReconstructor()
    rec.append_delta(_tool_delta(id="call_1", name="f"))
    with pytest.raises(AssertionError, match="function.name only once"):
        rec.append_delta(_tool_delta(name="f", args="x"))


def test_tool_reconstructor_rejects_first_delta_without_id():
    rec = StreamingToolReconstructor()
    with pytest.raises(AssertionError, match="id on first appearance"):
        rec.append_delta(_tool_delta(name="f"))


def test_tool_reconstructor_rejects_first_delta_without_name():
    rec = StreamingToolReconstructor()
    with pytest.raises(AssertionError, match="function.name on first"):
        rec.append_delta(_tool_delta(id="call_1"))


def test_tool_reconstructor_rejects_bad_type():
    rec = StreamingToolReconstructor()
    delta = _tool_delta(id="call_1", name="f")
    delta["tool_calls"][0]["type"] = "something_else"
    with pytest.raises(AssertionError, match="type='function'"):
        rec.append_delta(delta)


def test_tool_reconstructor_rejects_empty_delta():
    rec = StreamingToolReconstructor()
    with pytest.raises(AssertionError, match="content or tool_calls"):
        rec.append_delta({})


def test_tool_reconstructor_accepts_pure_content():
    rec = StreamingToolReconstructor()
    rec.append_delta({"content": "hello"})
    rec.append_delta({"content": " world"})
    assert rec.other_content == "hello world"
    assert rec.tool_calls == []


def test_tool_reconstructor_rejects_two_tools_per_delta_by_default():
    rec = StreamingToolReconstructor()
    delta = {
        "tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "f1"}},
            {"index": 1, "id": "b", "function": {"name": "f2"}},
        ]
    }
    with pytest.raises(AssertionError, match="only one tool call per delta"):
        rec.append_delta(delta)


def test_tool_reconstructor_allows_two_tools_when_opted_in():
    rec = StreamingToolReconstructor(assert_one_tool_per_delta=False)
    delta = {
        "tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "f1"}},
            {"index": 1, "id": "b", "function": {"name": "f2"}},
        ]
    }
    rec.append_delta(delta)
    assert [c.name for c in rec.tool_calls] == ["f1", "f2"]


# ----- batch_deltas_with_stream_interval ---------------------------------


@pytest.mark.parametrize("interval", [1, 2, 3, 5, 8])
def test_batch_deltas_preserves_round_trip(interval):
    deltas = ["a", "b", "c", "d", "e", "f", "g"]
    batched = batch_deltas_with_stream_interval(deltas, interval)
    assert "".join(batched) == "".join(deltas)


def test_batch_deltas_interval_1_is_identity():
    deltas = ["x", "y", "z"]
    assert batch_deltas_with_stream_interval(deltas, 1) == deltas


def test_batch_deltas_handles_empty():
    assert batch_deltas_with_stream_interval([], 4) == []


def test_batch_deltas_rejects_zero_interval():
    with pytest.raises(AssertionError, match=">= 1"):
        batch_deltas_with_stream_interval(["x"], 0)


# ----- Harmony marker allow-list -----------------------------------------


def test_harmony_markers_match_source():
    """Pin the marker list against ``_strip_control_tokens_inner``.

    If the parser ever extends its control-token allowlist without
    updating ``_harmony_markers.HARMONY_CONTROL_TOKENS``, every
    harmony regression file would start missing the new marker as a
    leak-check. This test makes that drift loud. Checks the inner
    (whitespace-preserving) helper that contains the actual marker
    list; the public ``_strip_control_tokens`` is a thin trim wrapper.
    """
    import inspect

    from vllm_mlx.tool_parsers import harmony_tool_parser as _htp

    src = inspect.getsource(_htp._strip_control_tokens_inner)
    for tok in HARMONY_CONTROL_TOKENS:
        assert tok in src, (
            f"Marker {tok!r} is in HARMONY_CONTROL_TOKENS but missing "
            f"from harmony_tool_parser._strip_control_tokens_inner source. "
            f"Either the parser drifted or the marker list is stale."
        )


def test_assert_no_marker_leak_accepts_clean_content():
    assert_no_harmony_marker_leak(None)
    assert_no_harmony_marker_leak("")
    assert_no_harmony_marker_leak("The weather is sunny.")


@pytest.mark.parametrize("marker", HARMONY_LEAK_MARKERS)
def test_assert_no_marker_leak_catches_each_marker(marker):
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_harmony_marker_leak(f"prefix {marker} suffix")


# ----- Reconstructor None-handling (codex K2) ---------------------------


def test_tool_reconstructor_rejects_malformed_tool_calls():
    """``tool_calls`` must be a list when present — string / dict / int /
    bytes all rejected. Renamed from ``rejects_null_tool_calls`` (codex
    re-review NIT): the prior name claimed null coverage, but the
    reconstructor deliberately treats ``tool_calls=None`` as equivalent
    to "key absent" (so JSON deltas that explicitly serialize the field
    as ``None`` round-trip cleanly). Only non-list types are rejected.
    """
    for bad in ("not-a-list", {"index": 0}, 42, b"bytes"):
        rec = StreamingToolReconstructor()
        with pytest.raises(AssertionError, match="malformed 'tool_calls'"):
            rec.append_delta({"content": "ok", "tool_calls": bad})


def test_tool_reconstructor_accepts_explicit_null_tool_calls():
    """``tool_calls=None`` is semantically equivalent to ``tool_calls``
    being absent — both yield an empty ``tool_calls`` list. Pins this
    contract so a future "reject None" tightening doesn't silently
    break clients that serialize the field as ``None`` on
    content-only deltas.
    """
    rec = StreamingToolReconstructor()
    rec.append_delta({"content": "hello", "tool_calls": None})
    assert rec.other_content == "hello"
    assert rec.tool_calls == []


def test_tool_reconstructor_rejects_null_function():
    rec = StreamingToolReconstructor()
    with pytest.raises(AssertionError, match="malformed 'function'"):
        rec.append_delta({"tool_calls": [{"index": 0, "id": "x", "function": None}]})


def test_tool_reconstructor_rejects_non_string_arguments():
    """Codex re-review BLOCKING: ``arguments`` was previously coerced via
    ``function.get("arguments") or ""``, silently swallowing ``None`` /
    ``0`` / ``False``. Strict typing now requires the field to be absent
    or a string; any other type fails the reconstructor's assertion.
    """
    rec = StreamingToolReconstructor()
    rec.append_delta(_tool_delta(id="call_1", name="get_weather"))
    for bad in (0, False, ["arg"], {"a": 1}, b"bytes"):
        with pytest.raises(AssertionError, match="malformed 'arguments'"):
            rec.append_delta(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": bad}},
                    ]
                }
            )
