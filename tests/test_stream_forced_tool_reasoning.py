# SPDX-License-Identifier: Apache-2.0
"""F-200 regression — streaming forced ``tool_choice`` on reasoning models
must not leak schema-violating scratch tool_calls.

When the chat template pre-injects ``<think>\\n`` AND the route forces a
named-function ``tool_choice`` (via the assistant-prefix-injection
lever ``_forced_tool_call_prefix``), reasoning models like
qwen3-4b-thinking-2507 / phi-4-mini-reasoning routinely emit a SCRATCH
``<tool_call>{"name": "X", "arguments": <bare int/string>}</tool_call>``
block INSIDE the implicit ``<think>`` before producing the real
post-think tool call. The MiniMax tool-markup redirect in
``_process_with_reasoning`` (load-bearing for the forced-prefix-in-think
flow) promotes those scratch blocks to content + ``_detect_tool_calls``,
which then ships a SECOND tool_call delta with arguments that aren't
JSON objects — a hard violation of the OpenAI spec which requires every
``tool_calls[i].function.arguments`` to be a JSON-encoded string.

The fix layers a ``_apply_forced_tool_choice_filter`` in front of
``_apply_parallel_cap`` (and the finalize cross-format fallback) that
drops any anchor whose name doesn't match the forced choice OR whose
``arguments`` parsed as a JSON non-object — and tells the parallel-cap
layer to drop downstream argument-fragment continuations of the dropped
anchor via ``_no_index_last_dropped``.

These tests pin the four-quadrant behaviour exercised by the live repro
(reasoning ✕ stream/non-stream ✕ forced/auto) plus the non-reasoning
control.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vllm_mlx.service.postprocessor import StreamingPostProcessor


def _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None):
    """Build a mock ServerConfig with a real hermes tool parser."""
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = reasoning_parser_name
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = tool_call_parser
    cfg.tool_parser_instance = None
    return cfg


def _make_output(text="", finished=False, finish_reason=None):
    """Build a mock GenerationOutput for the text-parser path."""
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = None
    out.finish_reason = finish_reason or ("stop" if finished else None)
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


def _forced_request(name="get_weather"):
    """Build a request dict carrying ``tool_choice={"type":"function","function":{"name":...}}``."""
    return {
        "tool_choice": {"type": "function", "function": {"name": name}},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }


def _auto_request(name="get_weather"):
    """Build a request dict with NO forced tool_choice (auto)."""
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }


# ── _forced_tool_choice_name helper pin ──────────────────────────────


class TestForcedNameHelper:
    def test_returns_name_for_function_form(self):
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("X"))
        assert pp._forced_tool_choice_name() == "X"

    def test_returns_none_for_auto(self):
        pp = StreamingPostProcessor(_make_cfg(), request=_auto_request())
        assert pp._forced_tool_choice_name() is None

    def test_returns_none_for_required_no_name(self):
        # ``"required"`` (string form) is NOT a named function — the
        # model can pick any tool. Filter must NOT engage.
        pp = StreamingPostProcessor(
            _make_cfg(),
            request={"tool_choice": "required", "tools": [{"function": {"name": "x"}}]},
        )
        assert pp._forced_tool_choice_name() is None

    def test_returns_none_for_missing_request(self):
        pp = StreamingPostProcessor(_make_cfg(), request=None)
        assert pp._forced_tool_choice_name() is None

    def test_returns_none_for_pydantic_unset(self):
        req = MagicMock()
        req.tool_choice = None
        pp = StreamingPostProcessor(_make_cfg(), request=req)
        assert pp._forced_tool_choice_name() is None

    def test_returns_name_for_pydantic_shaped_tool_choice(self):
        """Codex r4 BLOCKING — production routes pass
        ``request.model_dump(exclude_none=True)`` (a dict), but a
        typed-request callpath (test fixtures, future refactors) may
        leave ``tool_choice`` as a Pydantic model with ``.type`` /
        ``.function.name`` attributes. The dict-only gate would have
        silently disabled the filter on that path. Helper must read
        both shapes."""
        function_obj = MagicMock()
        function_obj.name = "get_weather"
        tc = MagicMock()
        tc.type = "function"
        tc.function = function_obj
        req = MagicMock()
        req.tool_choice = tc
        pp = StreamingPostProcessor(_make_cfg(), request=req)
        assert pp._forced_tool_choice_name() == "get_weather"


# ── _apply_forced_tool_choice_filter pin ─────────────────────────────


class TestForcedToolChoiceFilter:
    def test_drops_wrong_function_anchor(self):
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [
                {
                    "index": 0,
                    "function": {"name": "other_tool", "arguments": '{"x":1}'},
                },
                {
                    "index": 1,
                    "function": {"name": "get_weather", "arguments": '{"city":"P"}'},
                },
            ]
        )
        assert len(out) == 1
        assert out[0]["function"]["name"] == "get_weather"
        assert pp._no_index_last_dropped is True

    def test_drops_bare_integer_arguments(self):
        """The F-200 root case — model emits scratch
        ``<tool_call>{"name":"get_weather","arguments":1234567890}</tool_call>``
        inside ``<think>`` and the MiniMax redirect promotes it to a
        tool_call delta. ``arguments="1234567890"`` parses as a JSON
        integer (not object) — drop per OpenAI schema requirement."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [
                {
                    "index": 0,
                    "function": {"name": "get_weather", "arguments": "1234567890"},
                },
                {
                    "index": 1,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                },
            ]
        )
        assert len(out) == 1
        assert out[0]["function"]["arguments"] == '{"city":"Paris"}'

    def test_drops_json_quoted_string_arguments(self):
        """Same root pattern, JSON-quoted-string variant — the parser
        round-trips ``{"arguments": "..."}`` into a stringified JSON
        value (``'"..."'``) when the model emits a string literal in
        place of the object. Valid JSON but non-object — drop."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [
                {
                    "index": 0,
                    "function": {
                        "name": "get_weather",
                        "arguments": '"☉ Paris output output"',
                    },
                },
                {
                    "index": 1,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                },
            ]
        )
        assert len(out) == 1
        assert json.loads(out[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_drops_bare_unquoted_text_arguments(self):
        """Codex r2 BLOCKING #1 — phi-4-mini-reasoning has been
        observed panicking inside ``<think>`` and emitting BARE
        UNQUOTED prose where a JSON body should be:
        ``arguments: ☉ Paris output output``. The hermes parser
        forwards the bytes verbatim as the ``arguments`` string —
        this is NOT valid JSON at all (no surrounding quotes,
        non-ASCII). The OpenAI spec mandates a JSON-object string,
        so a non-JSON value can never satisfy the contract — drop."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [
                {
                    "index": 0,
                    "function": {
                        "name": "get_weather",
                        "arguments": "☉ Paris output output",
                    },
                },
                {
                    "index": 1,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                },
            ]
        )
        assert len(out) == 1
        assert json.loads(out[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_passes_anchor_with_partial_unclosed_json_arguments(self):
        """Codex r3 BLOCKING #1 — a hypothetical future parser could
        emit a single delta carrying ``name`` PLUS the first PARTIAL
        JSON fragment ``{"city":"Pa``. ``json.loads`` raises but
        ``{`` count > ``}`` count signals the body is mid-stream;
        pass through so subsequent fragments can complete it. Only
        when the JSON is well-formed-but-non-object OR
        balanced-but-broken does the helper drop."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [
                {
                    "index": 0,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Pa',
                    },
                }
            ]
        )
        assert len(out) == 1, "partial-JSON anchor was wrongly dropped"

    def test_drops_anchor_with_balanced_but_broken_json(self):
        """Mirror of the partial-pass-through test — when braces are
        balanced (``{`` count == ``}`` count) and parsing still fails,
        the body is finalized garbage (e.g. mis-escaped quotes
        ``{"city": Paris}``). Drop."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [
                {
                    "index": 0,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": Paris}',
                    },
                },
                {
                    "index": 1,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                },
            ]
        )
        assert len(out) == 1
        assert json.loads(out[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_drops_wrapped_function_shape_anchor(self):
        """Codex r3 BLOCKING #2 — channel-routed callers (harmony /
        gemma4 / future routers) emit calls in the wrapped
        ``{"function": {"name": ..., "arguments": ...}}`` shape.
        The forced-name filter must handle BOTH flat and wrapped
        shapes consistently — the earlier inline channel-routed
        filter accepted only flat and silently dropped wrapped
        valid calls."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [
                # Wrong wrapped name — drop.
                {"function": {"name": "other_tool", "arguments": '{"x":1}'}},
                # Right wrapped name + valid JSON object — keep.
                {"function": {"name": "get_weather", "arguments": '{"city":"P"}'}},
                # Right FLAT name + valid JSON object — also keep.
                {"name": "get_weather", "arguments": '{"city":"Q"}'},
            ]
        )
        assert len(out) == 2

    def test_passes_argument_fragment_continuation(self):
        """A continuation delta (no name, only argument fragment) must
        pass through — the parallel-cap layer routes it to whichever
        anchor was last admitted."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [{"index": 0, "function": {"arguments": '{"city": "Pa'}}]
        )
        assert len(out) == 1

    def test_drops_continuation_with_finalized_non_object_arguments(self):
        """r10-J round-3 — codex r3 HIGH #1.

        Pre-fix the ``anchor_name is None`` branch passed every
        continuation fragment through unconditionally. Some streaming
        parsers admit a name-only anchor first and then send the
        arguments in a follow-up continuation as a *finalized
        non-object* root (e.g. ``"20230805"``, a bare date string).
        The cap layer routed those into the admitted anchor,
        delivering schema-violating ``arguments`` to clients despite
        ``tool_choice="required"`` (or a forced named choice).

        Post-fix: the same object-root gate the finalized-anchor branch
        uses now fires on continuation fragments too. Partial-fragment
        JSON (unbalanced braces) is still passed through — covered by
        ``test_passes_argument_fragment_continuation`` above.
        """
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        # Continuation delta with FULLY-FORMED non-object JSON. Pre-fix
        # this passed through; post-fix it must be dropped.
        out = pp._apply_forced_tool_choice_filter(
            [{"index": 0, "function": {"arguments": '"20230805"'}}]
        )
        assert out == [], (
            "Finalized non-object continuation arguments must be dropped "
            "(codex r10-J r3 HIGH #1)."
        )
        assert pp._no_index_last_dropped is True

    def test_drops_continuation_with_bare_unquoted_text(self):
        """Same gate, bare unquoted text variant (the codex r2
        BLOCKING #1 / phi-4 ``<think>`` panic shape applied at the
        continuation level)."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [{"function": {"arguments": "Paris output output"}}]
        )
        assert out == []
        assert pp._no_index_last_dropped is True

    def test_drops_continuation_with_array_root_arguments(self):
        """Array-root variant — ``[1,2,3]`` parses as JSON but the
        spec requires an object. Continuation fragments with array
        roots must be dropped just like finalized anchors with array
        roots are (covered earlier in this class)."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [{"function": {"arguments": "[1,2,3]"}}]
        )
        assert out == []

    def test_no_op_when_no_forced_choice(self):
        """``auto`` mode — the filter MUST be a no-op so multi-call
        flows and other tools still work."""
        pp = StreamingPostProcessor(_make_cfg(), request=_auto_request())
        calls = [
            {"function": {"name": "get_weather", "arguments": "1234567890"}},
            {"function": {"name": "other_tool", "arguments": '{"y":2}'}},
        ]
        out = pp._apply_forced_tool_choice_filter(calls)
        assert out == calls

    def test_passes_anchor_with_no_arguments_yet(self):
        """First chunk often carries just ``name`` + ``id`` with empty
        / missing arguments — body streams in later fragments. Must
        not drop these as ``parsed != object``."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [{"index": 0, "id": "call_x", "function": {"name": "get_weather"}}]
        )
        assert len(out) == 1
        out2 = pp._apply_forced_tool_choice_filter(
            [
                {
                    "index": 0,
                    "id": "call_x",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ]
        )
        assert len(out2) == 1


# ── Four-quadrant behaviour pin (reasoning ✕ stream/non-stream ✕ forced/auto) ──


class TestStreamForcedReasoningEndToEnd:
    """End-to-end through ``process_chunk``: feed bytes from the F-200
    repro (scratch ``<tool_call>`` body inside the implicit ``<think>``
    block followed by the real call after ``</think>``) and assert
    only one structured tool_call event surfaces."""

    def _drain_stream(self, chunks, processor):
        """Replay a list of ``(text, finished)`` chunks through the
        processor and return the assembled event list."""
        out = []
        for i, (text, finished) in enumerate(chunks):
            evs = processor.process_chunk(
                _make_output(
                    text=text,
                    finished=finished,
                    finish_reason=("stop" if finished else None),
                )
            )
            out.extend(evs)
        out.extend(processor.finalize())
        return out

    def test_qwen3_thinking_stream_forced_drops_scratch_call(self):
        """qwen3-thinking + hermes + forced ``tool_choice``: scratch
        in-think ``<tool_call>...1234567890...</tool_call>`` followed
        by real ``<tool_call>...city:Paris...</tool_call>`` must emit
        EXACTLY ONE tool_call to ``get_weather`` with ``{"city":"Paris"}``."""
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name="qwen3")
        pp = StreamingPostProcessor(cfg, request=_forced_request("get_weather"))
        pp.reset()

        # Replay the recorded model output from the live repro. Chat
        # template pre-injected ``<think>``; model emits scratch call,
        # ``</think>``, real call.
        full = (
            '<tool_call>\n{"name": "get_weather", "arguments": 1234567890}\n</tool_call>\n'
            "</think>\n"
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        events = pp.process_chunk(_make_output(text=full, finished=True))
        events.extend(pp.finalize())

        tool_events = [e for e in events if e.type == "tool_call"]
        # Flatten all emitted tool_calls (parser may split into
        # multiple events). Every surviving entry MUST have valid-
        # object arguments matching the forced name.
        all_calls: list[dict] = []
        for ev in tool_events:
            for tc in ev.tool_calls or []:
                all_calls.append(tc)

        # F-200 contract: EXACTLY ONE surviving call AND that call
        # matches the forced name AND its ``arguments`` parse as a
        # JSON object. Codex r1 BLOCKING — the earlier draft asserted
        # only "every emitted call is valid" which would accept
        # duplicate valid calls; the spec requires a single tool_call
        # per forced ``tool_choice``.
        assert len(all_calls) == 1, (
            f"expected exactly 1 surviving tool_call, got {len(all_calls)}: {all_calls!r}"
        )
        tc = all_calls[0]
        fn = tc.get("function") or {}
        assert fn.get("name") == "get_weather"
        parsed = json.loads(fn["arguments"])
        assert isinstance(parsed, dict)
        assert parsed.get("city", "").lower() == "paris"

    def test_qwen3_thinking_stream_auto_keeps_call(self):
        """Same model, no forced ``tool_choice`` (``auto``): the filter
        is a no-op so the real call still surfaces."""
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name="qwen3")
        pp = StreamingPostProcessor(cfg, request=_auto_request("get_weather"))
        pp.reset()

        full = (
            "</think>\n"
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        events = pp.process_chunk(_make_output(text=full, finished=True))
        events.extend(pp.finalize())

        tool_events = [e for e in events if e.type == "tool_call"]
        all_calls = [tc for ev in tool_events for tc in (ev.tool_calls or [])]
        assert all_calls, "auto mode lost the call"
        for tc in all_calls:
            fn = tc.get("function") or {}
            assert fn.get("name") == "get_weather"

    def test_finalize_extract_drops_same_name_primitive_args(self):
        """Codex r1 BLOCKING #1 — when the streaming parser missed
        the call and ``finalize()`` runs ``extract_tool_calls`` over
        the accumulated buffer, the parser may return both a scratch
        call (same forced name, primitive ``arguments``) AND the real
        call. Filtering by name alone leaks the scratch; the finalize
        path must apply the same arguments-root-object validation."""
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_forced_request("get_weather"))
        pp.reset()
        # Seed the tool buffer so finalize's extract_tool_calls runs
        # the hermes parser over a buffer containing both shapes —
        # bypasses the streaming detection entirely (mimics the path
        # where the per-chunk parser holds bytes and only the buffer
        # carries them).
        pp.tool_accumulated_text = (
            '<tool_call>\n{"name": "get_weather", "arguments": 1234567890}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        events = pp.finalize()
        all_calls = [
            tc
            for ev in events
            if ev.type == "tool_call"
            for tc in (ev.tool_calls or [])
        ]
        assert len(all_calls) == 1, f"finalize leaked a scratch call: {all_calls!r}"
        fn = all_calls[0].get("function") or {}
        parsed = json.loads(fn["arguments"])
        assert isinstance(parsed, dict)
        assert parsed.get("city") == "Paris"

    def test_non_reasoning_stream_forced_keeps_call(self):
        """Hermes parser without a reasoning parser (e.g. qwen3-instruct):
        forced ``tool_choice`` + stream still surfaces the call with
        valid args. The filter is engaged but has nothing to drop."""
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_forced_request("get_weather"))
        pp.reset()

        full = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        events = pp.process_chunk(_make_output(text=full, finished=True))
        events.extend(pp.finalize())

        all_calls = [
            tc for e in events if e.type == "tool_call" for tc in (e.tool_calls or [])
        ]
        assert all_calls, "non-reasoning forced call was dropped"
        for tc in all_calls:
            fn = tc.get("function") or {}
            assert fn.get("name") == "get_weather"
            parsed = json.loads(fn["arguments"])
            assert isinstance(parsed, dict)
