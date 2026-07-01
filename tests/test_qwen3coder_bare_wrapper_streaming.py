# SPDX-License-Identifier: Apache-2.0
"""Streaming parity for ``Qwen3CoderToolParser`` when the model output does
NOT carry the ``<tool_call>`` wrapper token.

Closes issue #978. Repro: ``Shiftedx/qwopus3.6-35b-a3b-coder-mxfp4-mlx``
tokenizer lacks ``<|tool_call|>`` / ``<tool_call>`` as a special token, so
the model reliably emits well-formed ``<function=NAME>...</function>``
XML *without* the wrapper. The non-streaming ``extract_tool_calls`` path
already handles both shapes (it triggers on the ``<function=`` prefix);
the streaming path used to anchor on ``<tool_call>`` and therefore never
fired for wrapper-less output — Claude Code (which always streams) then
saw the entire tool call as raw assistant content.

The four tests below pin the invariant:

* ``test_bare_function_block_streams_as_tool_call`` — the failing case
  from the issue: no ``<tool_call>`` wrapper, chunked incrementally,
  must emit a ``tool_calls`` delta with the correct name + arguments
  BEFORE the stream ends.
* ``test_bare_multi_function_blocks_stream_both_calls`` — two bare
  ``<function=...>...</function>`` blocks back-to-back must emit both
  as separate ``tool_calls`` deltas with distinct indices.
* ``test_wrapped_streaming_still_works`` — regression guard: the native
  ``<tool_call><function=...>...</function></tool_call>`` shape MUST
  continue to stream correctly (this is what would break if the fix
  went ad-hoc and swapped one anchor for the other).
* ``test_content_before_bare_function_is_emitted_as_content`` — prose
  before a bare ``<function=`` opener must surface as a ``content``
  delta, not be swallowed or misclassified as tool_call payload.
"""

from __future__ import annotations

import json

from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


def _request_with_tool(name: str, properties: dict) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object", "properties": properties},
                },
            }
        ]
    }


def _feed(parser: Qwen3CoderToolParser, chunks: list[str], request: dict | None):
    parser.reset()
    deltas: list[dict] = []
    previous = ""
    for chunk in chunks:
        if not chunk:
            continue
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        if delta is not None:
            deltas.append(delta)
        previous = current
    return deltas


def _argument_fragments_for_index(deltas: list[dict], index: int) -> list[str]:
    """Flatten ``function.arguments`` strings for a specific tool call index."""
    out: list[str] = []
    for d in deltas:
        for tc in d.get("tool_calls") or []:
            if tc.get("index") != index:
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if args:
                out.append(args)
    return out


def _names_by_index(deltas: list[dict]) -> dict[int, str]:
    """Collect tool call name assigned per index (header emit)."""
    seen: dict[int, str] = {}
    for d in deltas:
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name and tc.get("index") not in seen:
                seen[tc.get("index")] = name
    return seen


def _content_events(deltas: list[dict]) -> list[str]:
    return [d["content"] for d in deltas if "content" in d]


def test_bare_function_block_streams_as_tool_call():
    """The exact failing case from #978: bare ``<function=...>...</function>``
    with NO ``<tool_call>`` wrapper streamed incrementally must emit a
    ``tool_calls`` delta with the correct name and JSON-valid arguments.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("read_file", {"path": {"type": "string"}})

    chunks = [
        "<function=read_file>\n",
        "<parameter=path>\n",
        "/src/main.py",
        "\n</parameter>\n",
        "</function>",
    ]

    deltas = _feed(parser, chunks, request)

    names = _names_by_index(deltas)
    assert names.get(0) == "read_file", (
        f"bare <function=…> stream never emitted a tool_calls header; "
        f"names={names!r}, deltas={deltas!r}"
    )

    fragments = _argument_fragments_for_index(deltas, 0)
    # Exclude the header emit (empty arguments) so we count only body deltas.
    body_fragments = [f for f in fragments if f != ""]
    combined = "".join(body_fragments)
    decoded = json.loads(combined)
    assert decoded == {"path": "/src/main.py"}, (
        f"bare-wrapper arguments did not stream correctly. "
        f"combined={combined!r}, decoded={decoded!r}"
    )

    # No content event should have leaked the raw tool-call markup.
    for text in _content_events(deltas):
        assert "<function=" not in text, (
            f"bare tool-call markup leaked as content: {text!r}"
        )
        assert "</function>" not in text, (
            f"bare tool-call markup leaked as content: {text!r}"
        )
        assert "<parameter=" not in text, (
            f"bare tool-call markup leaked as content: {text!r}"
        )


def test_bare_multi_function_blocks_stream_both_calls():
    """Two bare ``<function=A>...</function><function=B>...</function>`` blocks
    back-to-back must emit BOTH as tool_calls with distinct indices."""
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            },
        ]
    }

    chunks = [
        "<function=read_file>",
        "<parameter=path>/a.py</parameter>",
        "</function>",
        "\n",
        "<function=write_file>",
        "<parameter=path>/b.py</parameter>",
        "<parameter=content>hello</parameter>",
        "</function>",
    ]

    deltas = _feed(parser, chunks, request)

    names = _names_by_index(deltas)
    assert names.get(0) == "read_file", f"tool index 0 must be read_file, got {names!r}"
    assert names.get(1) == "write_file", (
        f"tool index 1 must be write_file, got {names!r}"
    )

    args_0 = json.loads("".join(_argument_fragments_for_index(deltas, 0)))
    args_1 = json.loads("".join(_argument_fragments_for_index(deltas, 1)))
    assert args_0 == {"path": "/a.py"}, args_0
    assert args_1 == {"path": "/b.py", "content": "hello"}, args_1


def test_wrapped_streaming_still_works():
    """Regression guard: the native ``<tool_call><function=…>…</function>
    </tool_call>`` shape MUST continue to stream correctly. This is what
    breaks if the bare-wrapper fix goes ad-hoc and merely swaps anchors
    instead of unifying detection.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("read_file", {"path": {"type": "string"}})

    chunks = [
        "<tool_call>\n",
        "<function=read_file>\n",
        "<parameter=path>\n",
        "/src/main.py",
        "\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    deltas = _feed(parser, chunks, request)

    names = _names_by_index(deltas)
    assert names.get(0) == "read_file", names

    fragments = _argument_fragments_for_index(deltas, 0)
    combined = "".join(f for f in fragments if f != "")
    decoded = json.loads(combined)
    assert decoded == {"path": "/src/main.py"}, (
        f"wrapped-mode regression: expected {{'path': '/src/main.py'}}, got {decoded!r}"
    )

    for text in _content_events(deltas):
        assert "<tool_call>" not in text, (
            f"wrapped tool-call markup leaked as content: {text!r}"
        )
        assert "<function=" not in text, (
            f"wrapped tool-call markup leaked as content: {text!r}"
        )


def test_function_prefix_in_parameter_value_not_counted_as_tool_boundary():
    """Unit test on ``_function_start_positions``: a ``<function=``
    substring inside a ``<parameter=…>…</parameter>`` value MUST NOT
    count as a top-level tool boundary.

    Codex adversarial review on this PR flagged the naive ``str.find``
    variant as a corruption vector: with it, streaming would advance
    ``current_tool_index`` mid-argument and either miss trailing
    content or emit a phantom second tool call. The fix scans
    top-level occurrences by skipping past parameter-value spans.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)

    text = (
        "<function=echo>"
        "<parameter=text>call <function=inner> in a code snippet</parameter>"
        "</function>"
    )
    starts = parser._function_start_positions(text)
    assert starts == [0], (
        f"top-level ``<function=`` scanner miscounted: expected [0], got {starts!r}"
    )


def test_function_end_in_parameter_value_not_counted_as_tool_close():
    """Unit test on ``_top_level_function_close_count``: a
    ``</function>`` substring inside a parameter value MUST NOT count
    as a structural tool close.

    Same codex concern as above, applied to the ``</function>`` counter
    that drives ``current_tool_index`` advancement.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)

    text = "<function=echo><parameter=code>foo</function>bar</parameter></function>"
    starts = parser._function_start_positions(text)
    close_count = parser._top_level_function_close_count(text, starts)
    assert close_count == 1, (
        f"top-level ``</function>`` scanner miscounted: expected 1, "
        f"got {close_count}. starts={starts!r}"
    )


def test_streaming_state_not_corrupted_by_function_prefix_in_value():
    """End-to-end: even when a parameter value carries a full inner
    ``<function=inner></function>`` block, the state machine MUST emit
    exactly ONE tool call — not a phantom second call anchored on the
    inner literal, and not a stalled advance that swallows trailing
    content.

    This is the exact regression codex round-1 caught: with the naive
    ``count(<function=)`` and ``count(</function>)`` the advance block
    would try to promote the inner literal to a real tool_call.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("echo", {"note": {"type": "string"}})

    chunks = [
        "<function=echo>",
        "<parameter=note>",
        # Full inner XML — the naive scanner would see 2 openers and 2
        # closers here and try to advance mid-argument.
        "see also <function=inner></function> below",
        "</parameter>",
        "</function>",
        # Trailing content after the real tool closes — the naive
        # scanner would still think ``is_tool_call_started`` is True
        # and try to slice a phantom second block instead of emitting
        # this as a content event.
        " done.",
    ]

    deltas = _feed(parser, chunks, request)
    names = _names_by_index(deltas)
    assert set(names.keys()) == {0}, (
        f"phantom tool index emitted; names={names!r}, deltas={deltas!r}"
    )
    assert names[0] == "echo"
    # Trailing content after the real tool close must reach the
    # content stream, not vanish into a phantom advance.
    contents = _content_events(deltas)
    assert any("done" in c for c in contents), (
        f"trailing content swallowed by phantom advance; contents={contents!r}"
    )


def test_content_before_bare_function_is_emitted_as_content():
    """Prose before a bare ``<function=`` opener must surface as a ``content``
    delta — not be swallowed and not be miscategorized as tool_call payload.
    Mirrors the wrapped-mode behavior that already extracts prose before
    ``<tool_call>``.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("read_file", {"path": {"type": "string"}})

    # Fine-grained chunking so the streaming state machine has enough
    # deltas to reach the params-loop after (a) emitting content-before,
    # (b) parsing the header, and (c) emitting the JSON opener ``{``.
    # This mirrors the per-token deltas real streams deliver.
    chunks = [
        "Let me read that file. ",
        "<function=read_file>",
        "<parameter=path>",
        "/src/main.py",
        "</parameter>",
        "</function>",
    ]

    deltas = _feed(parser, chunks, request)

    contents = _content_events(deltas)
    assert any("Let me read that file. " in c for c in contents), (
        f"prose before <function=…> was dropped. contents={contents!r}"
    )
    # And the tool call still fired.
    names = _names_by_index(deltas)
    assert names.get(0) == "read_file", names
    combined = "".join(f for f in _argument_fragments_for_index(deltas, 0) if f != "")
    assert json.loads(combined) == {"path": "/src/main.py"}
