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

    def test_passes_bare_text_continuation_for_merge_layer(self):
        """r10-J round-5 trade-off (codex r5 HIGH #1).

        Round-3 wrongly dropped bare unquoted text continuations at
        the streaming-gate layer. Round-5 narrowed the continuation
        predicate to ONLY drop on confirmed JSON non-object roots —
        bare prose ``Paris output output`` fails ``json.loads`` and
        is therefore indistinguishable at this layer from a middle
        fragment of a legitimate split JSON string. We must pass it
        through so split-arg streams aren't truncated.

        Safety net: the cap+merge layer assembles fragments back
        into the full ``arguments`` string of the admitted anchor,
        and the finalized-anchor gate runs at finalize time over
        that assembled string. If the merged result is still bare
        prose (no closing brace ever arrived), the finalize-side
        ``_forced_tool_choice_arguments_violate_object_root`` (which
        retains the strict balanced-but-broken classifier) drops
        the full call before it ships to the client. Coverage of
        that finalize-side drop lives in the cross-format fallback
        tests under TestStreamForcedReasoningEndToEnd below.
        """
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [{"function": {"arguments": "Paris output output"}}]
        )
        assert len(out) == 1
        assert pp._no_index_last_dropped is False

    def test_passes_closing_fragment_of_split_args(self):
        """r10-J round-5 — codex r5 HIGH #1 regression guard.

        Streaming parsers commonly split JSON arguments across
        deltas: first ``{"city":"Pa``, then ``ris"}``. The opening
        fragment has more ``{`` than ``}`` and the round-3 helper
        passes it through (unbalanced = partial). The CLOSING
        fragment has more ``}`` than ``{`` — the pre-round-5 helper
        wrongly classified it as "balanced-but-broken garbage" and
        dropped it, truncating an otherwise-valid forced or
        required tool call.

        Post-round-5: continuations use the narrower
        ``_continuation_arguments_definitively_non_object`` predicate
        which only drops on confirmed JSON-non-object roots. Both
        halves of a split must pass through.
        """
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        # Opening fragment — { > } — already-known-good.
        out_open = pp._apply_forced_tool_choice_filter(
            [{"function": {"arguments": '{"city":"Pa'}}]
        )
        assert len(out_open) == 1, "opening fragment wrongly dropped"
        # Closing fragment — } > { — REGRESSION from round-3 if dropped.
        out_close = pp._apply_forced_tool_choice_filter(
            [{"function": {"arguments": 'ris"}'}}]
        )
        assert len(out_close) == 1, (
            "closing fragment of split JSON args wrongly dropped — the "
            "continuation predicate must not over-rotate balanced-but-"
            "broken garbage onto legitimate split-JSON continuations "
            "(codex r10-J r5 HIGH #1)."
        )

    def test_passes_middle_fragment_of_three_way_split(self):
        """Defense-in-depth: a middle fragment like ``"PARI`` (no
        braces at all, bare bytes mid-string) must also pass — the
        cap+merge layer is what reassembles the full string."""
        pp = StreamingPostProcessor(_make_cfg(), request=_forced_request("get_weather"))
        out = pp._apply_forced_tool_choice_filter(
            [{"function": {"arguments": '"PARI'}}]
        )
        assert len(out) == 1

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


class TestRequiredModeFilter:
    """r10-J round-4 — codex r4 HIGH #1.

    Required-mode (``tool_choice="required"``) admits any tool but
    every recovered call must still produce a JSON-object
    ``arguments`` string. Pre-fix the streaming filter caught this
    via ``_apply_forced_tool_choice_filter`` (object-root gate
    bypasses ``forced_name is None``), but the finalize fallback
    paths in ``postprocessor.py:3061+`` only applied the gate when
    ``_forced_tool_choice_name()`` returned a named function — so
    fallback-recovered ``arguments="20230805"`` reached the client
    despite required semantics.

    Round-4 added a ``_is_tool_choice_required()`` arm to BOTH the
    ``extract_tool_calls`` recovery branch and the cross-format
    fallback branch. These tests pin both with direct calls to
    ``_apply_forced_tool_choice_filter`` (the streaming twin of the
    finalize gate, sharing the same object-root helper) — full
    finalize-path coverage is exercised by the broader integration
    suite.
    """

    @staticmethod
    def _required_request():
        return {
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"type": "object"},
                    },
                },
            ],
        }

    def test_required_mode_helper_detects_required(self):
        pp = StreamingPostProcessor(_make_cfg(), request=self._required_request())
        assert pp._is_tool_choice_required() is True

    def test_required_mode_helper_skips_auto(self):
        pp = StreamingPostProcessor(_make_cfg(), request=_auto_request())
        assert pp._is_tool_choice_required() is False

    def test_required_mode_drops_primitive_arguments_via_streaming_gate(self):
        """The streaming gate (``_apply_forced_tool_choice_filter``)
        already drops primitive-args calls under required mode — this
        is the contract the finalize fallback now mirrors. Pins that
        the gate engaged via the object-root check is intact regardless
        of whether the call carries a name (required allows ANY name).
        """
        pp = StreamingPostProcessor(_make_cfg(), request=self._required_request())
        out = pp._apply_forced_tool_choice_filter(
            [
                # Required mode, bare-string args — drop.
                {"function": {"name": "search", "arguments": '"20230805"'}},
                # Required mode, object args — keep.
                {"function": {"name": "search", "arguments": '{"q":"x"}'}},
            ]
        )
        assert len(out) == 1
        assert out[0]["function"]["arguments"] == '{"q":"x"}'

    def test_required_mode_continuation_finalized_non_object_dropped(self):
        """Continuation fragment with FINALIZED non-object root under
        required mode — round-3 already extended the
        anchor_name=None branch to drop these. Pin under required-mode
        request shape (codex r4 specifically called out required as the
        forced_name-None case)."""
        pp = StreamingPostProcessor(_make_cfg(), request=self._required_request())
        out = pp._apply_forced_tool_choice_filter(
            [{"function": {"arguments": '"20230805"'}}]
        )
        assert out == []
        assert pp._no_index_last_dropped is True

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

    def test_finalize_extract_drops_primitive_args_under_required_mode(self):
        """r10-J round-4 finalize-path coverage (codex r6 LOW #6).

        Required-mode twin of
        ``test_finalize_extract_drops_same_name_primitive_args`` above.
        Pre-round-4 the finalize ``extract_tool_calls`` branch only
        applied the object-root gate when ``_forced_tool_choice_name``
        was set. For ``tool_choice="required"`` the forced name is
        None and the gate was disabled, so fallback-recovered calls
        with primitive args reached the client.

        Pin that after round-4 the required-mode arm of the
        finalize ``extract_tool_calls`` branch drops the scratch
        call and keeps only the real one.
        """
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        required_request = {
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
        pp = StreamingPostProcessor(cfg, request=required_request)
        pp.reset()
        # Buffer carries BOTH the scratch primitive-args call AND the
        # real object-args call — mirrors the forced-name case.
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
        assert len(all_calls) == 1, (
            f"required-mode finalize leaked a scratch call: {all_calls!r}"
        )
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


# ── R11-A invariant: finish_reason="tool_calls" ⇒ ≥1 tool_call delta ─


def _required_request(name="format_date"):
    """Build a request dict carrying ``tool_choice="required"``."""
    return {
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {"raw": {"type": "integer"}},
                        "required": ["raw"],
                    },
                },
            }
        ],
    }


def _collect_terminal_state(events):
    """Aggregate stream events into the invariant-relevant state.

    Returns a dict with:
      - ``tool_call_delta_count``: count of ``tool_calls`` array entries
        across every ``tool_call`` StreamEvent (the wire-equivalent of
        ``delta.tool_calls`` chunks).
      - ``finish_reasons``: list of every non-None ``finish_reason`` on
        any event in order (R11-V2 invariant: exactly one non-None
        finish_reason should appear before stream close).
    """
    tool_call_delta_count = 0
    finish_reasons: list = []
    for ev in events:
        if ev.type == "tool_call" and ev.tool_calls:
            tool_call_delta_count += len(ev.tool_calls)
        if getattr(ev, "finish_reason", None):
            finish_reasons.append(ev.finish_reason)
    return {
        "tool_call_delta_count": tool_call_delta_count,
        "finish_reasons": finish_reasons,
    }


class TestR11AInvariantFilterEmptyDoesNotPromiseToolCalls:
    """R11-V1: when ``_apply_forced_tool_choice_filter`` drops every
    anchor (scratch with primitive-args root), the terminal event MUST
    NOT carry ``finish_reason="tool_calls"``. The pre-fix path lied to
    the client: it stamped ``tool_calls_detected=True`` AND emitted
    ``finish_reason="tool_calls"`` even though zero ``delta.tool_calls``
    chunks reached the wire. Clients depending on the promised tool
    call hung their agent loop.

    Repro shape mirrors the live Qwen3-0.6B + tool_choice=required
    failure: model emits a SINGLE complete ``<tool_call>`` with
    ``arguments=<bare-int>``.
    """

    def test_required_mode_bare_int_args_in_one_chunk_keeps_stop(self):
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
        pp.reset()

        # Single-chunk emission of a complete scratch tool_call —
        # arguments parse to bare int 20230805 (non-object), filter
        # drops it. ``output.finished=True`` on the same chunk so the
        # terminal finish event fires here.
        scratch = (
            '<tool_call>\n{"name": "format_date", "arguments": 20230805}\n'
            "</tool_call>"
        )
        events = pp.process_chunk(_make_output(text=scratch, finished=True, finish_reason="stop"))
        events.extend(pp.finalize())

        state = _collect_terminal_state(events)
        # Invariant: zero tool_call deltas reached the wire AND no
        # finish_reason==tool_calls was promised.
        assert state["tool_call_delta_count"] == 0, (
            f"filter should have dropped the scratch call but tool_call deltas leaked: {state!r}"
        )
        assert "tool_calls" not in state["finish_reasons"], (
            f"R11-V1 regression: finish_reason=tool_calls emitted with zero deltas: {state!r}"
        )
        # Downgrade target: engine's natural finish_reason ("stop").
        assert "stop" in state["finish_reasons"], (
            f"terminal finish_reason missing or wrong: {state!r}"
        )
        # Wire-truth counter MUST be zero.
        assert pp._tool_calls_emitted_to_wire == 0

    def test_required_mode_filter_drop_then_followup_chunk_stays_stop(self):
        """Mid-stream filter drop then a later finished chunk still
        downgrades the finish to ``stop``.

        Pre-fix the early-exit branch ``if self.tool_calls_detected:``
        on the next chunk stamped ``finish_reason="tool_calls"``
        unconditionally — even though the prior chunk's tool_calls had
        been filter-dropped. This pins that path through
        ``_compute_finish_reason``.
        """
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
        pp.reset()

        scratch = (
            '<tool_call>\n{"name": "format_date", "arguments": 20230805}\n'
            "</tool_call>"
        )
        # Chunk 1: parser surfaces the call → filter drops → no event
        events_a = pp.process_chunk(_make_output(text=scratch, finished=False))
        # Chunk 2 (zero-text finished chunk): hits the
        # ``if self.tool_calls_detected: ... finished:`` branch.
        events_b = pp.process_chunk(_make_output(text="", finished=True, finish_reason="stop"))
        events = list(events_a) + list(events_b) + list(pp.finalize())

        state = _collect_terminal_state(events)
        assert state["tool_call_delta_count"] == 0
        assert "tool_calls" not in state["finish_reasons"], (
            f"R11-V1 regression on next-chunk early-exit: {state!r}"
        )
        assert "stop" in state["finish_reasons"]

    def test_required_mode_real_call_keeps_tool_calls(self):
        """Symmetric pin: when the parser emits a VALID call (args is
        a JSON object), the filter passes it through, the wire emits a
        tool_call delta, AND ``finish_reason="tool_calls"`` is honored.
        Ensures the fix doesn't over-rotate and downgrade legitimate
        calls.
        """
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
        pp.reset()

        full = (
            '<tool_call>\n{"name": "format_date", "arguments": {"raw": 20230805}}\n'
            "</tool_call>"
        )
        events = pp.process_chunk(_make_output(text=full, finished=True, finish_reason="stop"))
        events.extend(pp.finalize())

        state = _collect_terminal_state(events)
        assert state["tool_call_delta_count"] >= 1, (
            f"valid required-mode call must surface a tool_call delta: {state!r}"
        )
        assert "tool_calls" in state["finish_reasons"], (
            f"valid required-mode call must keep finish=tool_calls: {state!r}"
        )
        assert pp._tool_calls_emitted_to_wire >= 1

    def test_finalize_recovery_increments_wire_counter(self):
        """Cross-format fallback in finalize() also satisfies the
        invariant: when streaming missed the call and ``finalize()``
        recovered it via ``extract_tool_calls`` / the cross-format
        scan, ``_tool_calls_emitted_to_wire`` MUST be incremented so
        downstream consumers see the consistent state.
        """
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
        pp.reset()
        # Seed the buffer so finalize() runs the fallback parser.
        pp.tool_accumulated_text = (
            '<tool_call>\n{"name": "format_date", "arguments": {"raw": 20230805}}\n'
            "</tool_call>"
        )
        events = pp.finalize()
        emitted_calls = [
            tc for ev in events if ev.type == "tool_call" for tc in (ev.tool_calls or [])
        ]
        assert emitted_calls, "finalize recovery did not surface the call"
        assert pp._tool_calls_emitted_to_wire == len(emitted_calls)

    def test_reset_clears_wire_counter(self):
        """``reset()`` MUST zero ``_tool_calls_emitted_to_wire`` so a
        re-used processor doesn't carry the prior turn's emitted count
        forward and lie to ``_compute_finish_reason`` on the next
        request (BatchedEngine singleton-parser path)."""
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
        pp._tool_calls_emitted_to_wire = 7
        pp.reset()
        assert pp._tool_calls_emitted_to_wire == 0


class TestR11AInvariantHoldsAcrossSuite:
    """Cross-suite invariant: ``finish_reason=="tool_calls" ⇒
    tool_call_delta_count >= 1`` MUST hold across every postprocessor-
    only stream test in this module. The check is structural — it
    drives the postprocessor with the matrix of repro shapes seen
    in dogfood-0812 (R11-V1 evidence) and confirms the invariant.
    """

    SCRATCH_CASES = [
        # (label, arguments_value_in_scratch_block)
        ("bare_int", "20230805"),
        ("bare_quoted_string", '"20230805"'),
        ("bare_unquoted_prose", "Paris output"),
        ("bare_array_root", "[1,2,3]"),
    ]

    def test_invariant_holds_for_every_scratch_shape(self):
        for label, args in self.SCRATCH_CASES:
            cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
            pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
            pp.reset()
            scratch = (
                '<tool_call>\n{"name": "format_date", "arguments": ' + args + "}\n"
                "</tool_call>"
            )
            events = pp.process_chunk(
                _make_output(text=scratch, finished=True, finish_reason="stop")
            )
            events.extend(pp.finalize())
            state = _collect_terminal_state(events)
            # Core invariant — same assertion the route-level prompt
            # spells out: ``finish_reason=="tool_calls" ⇒ tool_call_delta_count >= 1``.
            if "tool_calls" in state["finish_reasons"]:
                assert state["tool_call_delta_count"] >= 1, (
                    f"R11-A invariant violated on {label!r}: {state!r}"
                )
            # And the wire-truth counter must always match the emitted
            # count (no orphan increments / decrements).
            assert pp._tool_calls_emitted_to_wire == state["tool_call_delta_count"], (
                f"wire-counter drift on {label!r}: counter="
                f"{pp._tool_calls_emitted_to_wire} emitted={state['tool_call_delta_count']}"
            )


class TestR11V2ExactlyOneFinishReason:
    """R11-V2 invariant: every closed stream MUST emit exactly one
    ``finish_reason`` chunk before ``[DONE]``. Pre-fix some Qwen3 +
    tool_choice=required streams in dogfood-0812 emitted zero
    finish_reason chunks — clients hung waiting for the spec-mandated
    terminal marker. Pinned by exercising the postprocessor-only
    invariant: any path that ends a stream (process_chunk on the
    finished chunk + finalize()) MUST surface at least one
    finish_reason somewhere in the combined event list.
    """

    def test_filter_drop_finished_chunk_still_emits_finish(self):
        """Filter drops the only scratch call AND the chunk is
        ``finished=True`` — the early branch must emit a ``finish``
        event (with downgraded reason). Pre-fix this path always
        emitted a finish event so this just confirms we didn't break
        it; the regression risk lives in the assertion that NO chunk
        is silently lost."""
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
        pp.reset()
        scratch = (
            '<tool_call>\n{"name": "format_date", "arguments": 20230805}\n'
            "</tool_call>"
        )
        events = pp.process_chunk(
            _make_output(text=scratch, finished=True, finish_reason="stop")
        )
        finish_events = [e for e in events if e.type == "finish"]
        assert finish_events, "filter-drop on finished chunk must still emit a finish event"
        # And the lone finish carries a non-None reason.
        assert all(e.finish_reason for e in finish_events)

    def test_real_call_finished_chunk_emits_exactly_one_finish_reason(self):
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
        pp.reset()
        full = (
            '<tool_call>\n{"name": "format_date", "arguments": {"raw": 20230805}}\n'
            "</tool_call>"
        )
        events = pp.process_chunk(
            _make_output(text=full, finished=True, finish_reason="stop")
        )
        events.extend(pp.finalize())
        state = _collect_terminal_state(events)
        # Exactly one finish_reason across the stream (R11-V2 spec).
        assert len(state["finish_reasons"]) == 1, (
            f"expected exactly one finish_reason but got {state['finish_reasons']!r}"
        )
        assert state["finish_reasons"][0] == "tool_calls"


class TestR11ARegressionFromDogfoodSSE:
    """Direct regression scrape: drive the postprocessor with the
    scratch-buffer state observed in ``/tmp/dogfood-0812/sse-required-*.txt``
    (the live Qwen3 + tool_choice=required repro). The hermes parser
    surfaces ``arguments`` as a bare-int string; the postprocessor
    must drop the call AND downgrade the terminal to ``stop``.
    """

    def test_dogfood_qwen3_required_scratch_buffer(self):
        cfg = _make_cfg(tool_call_parser="hermes", reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, request=_required_request("format_date"))
        pp.reset()
        # The bytes observed in the live SSE captures (canonical
        # hermes wire shape that the parser drives through the
        # streaming path).
        wire_text = (
            "<tool_call>\n"
            '{"name": "format_date", "arguments": 20230805}\n'
            "</tool_call>"
        )
        # Stream it byte-by-byte to exercise the cumulative
        # streaming-parser path AND its interaction with the filter
        # on the final ``</tool_call>`` closing delta.
        events = []
        for i, ch in enumerate(wire_text):
            is_last = i == len(wire_text) - 1
            events.extend(
                pp.process_chunk(
                    _make_output(
                        text=ch,
                        finished=is_last,
                        finish_reason="stop" if is_last else None,
                    )
                )
            )
        events.extend(pp.finalize())

        state = _collect_terminal_state(events)
        assert state["tool_call_delta_count"] == 0, (
            f"byte-streamed repro leaked a tool_call delta: {state!r}"
        )
        assert "tool_calls" not in state["finish_reasons"], (
            f"R11-V1 byte-stream regression: {state!r}"
        )
        assert "stop" in state["finish_reasons"]
