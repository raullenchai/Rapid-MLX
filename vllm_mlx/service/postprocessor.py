# SPDX-License-Identifier: Apache-2.0
"""Streaming post-processor — unified reasoning + tool call + sanitization pipeline.

Replaces 500+ lines of duplicated logic across stream_chat_completion,
_stream_anthropic_messages, and stream_completion. NOT a filter chain —
one cohesive orchestrator, because reasoning/tool/sanitize are tightly coupled.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from ..api.tool_calling import parse_tool_calls
from ..api.utils import sanitize_output, strip_special_tokens
from ..domain.events import StreamEvent

if TYPE_CHECKING:
    from ..config.server_config import ServerConfig
    from ..engine.base import GenerationOutput

logger = logging.getLogger(__name__)


def _find_json_start(text: str) -> int:
    """Find the first `{` or `[` that is NOT inside `<think>...</think>` tags.

    Returns the index in ``text``, or -1 if no JSON delimiter found outside
    think blocks.  Handles unclosed `<think>` (still accumulating) by
    treating everything after it as inside the block.
    """
    in_think = False
    i = 0
    while i < len(text):
        # Check for <think> open tag
        if text[i : i + 7] == "<think>":
            in_think = True
            i += 7
            continue
        # Check for </think> close tag
        if text[i : i + 8] == "</think>":
            in_think = False
            i += 8
            continue
        # Outside think block — check for JSON delimiter
        if not in_think and text[i] in ("{", "["):
            return i
        i += 1
    return -1


class StreamingPostProcessor:
    """Processes streaming engine output into StreamEvents.

    Handles:
    1. Channel routing (OutputRouter models like Gemma 4)
    2. Reasoning extraction (text-based parsers for Qwen3, DeepSeek, MiniMax)
    3. Tool call streaming detection (incremental parser)
    4. Output sanitization (strip special tokens, markup)

    Usage::

        processor = StreamingPostProcessor(cfg, request)
        processor.reset()
        async for output in engine.stream_chat(...):
            for event in processor.process_chunk(output):
                yield format_for_my_api_spec(event)
        for event in processor.finalize():
            yield format_for_my_api_spec(event)
    """

    def __init__(
        self,
        cfg: ServerConfig,
        tools_requested: bool = False,
        enable_thinking: bool | None = None,
        json_mode: bool = False,
        request: dict | None = None,
    ):
        self.cfg = cfg
        self.tools_requested = tools_requested
        self.json_mode = json_mode
        # Forwarded to streaming tool parsers — qwen3_coder needs request.tools
        # for schema-driven type conversion (#171). Without it, raw XML leaks
        # into delta.content instead of structured tool_calls deltas.
        self.request = request
        # When the client explicitly sets enable_thinking=False, the chat
        # template suppresses the <think> generation prompt and the model
        # answers directly. The streaming reasoning parser's implicit-think
        # heuristic (treat ambiguous tokens as reasoning until </think> is
        # seen) misclassifies that direct answer as reasoning_content,
        # leaving content empty. Track the explicit signal so process_chunk
        # can skip the reasoning path in that case.
        self.enable_thinking = enable_thinking

        # Per-request parser instances — each streaming request gets its
        # own parser to avoid state corruption under concurrent
        # BatchedEngine requests.
        #
        # Production path: reasoning_parser_name / tool_call_parser are set
        # at startup → _create_*() builds a fresh instance per request.
        #
        # Legacy/test path: cfg.reasoning_parser / cfg.tool_parser_instance
        # may be pre-built (mocks in tests, or singleton from server.py).
        # When reasoning_parser_name is set, always create fresh.
        if cfg.reasoning_parser_name:
            self.reasoning_parser = self._create_reasoning_parser(cfg)
        else:
            self.reasoning_parser = cfg.reasoning_parser  # None or injected mock

        if cfg.tool_call_parser:
            self.tool_parser = self._create_tool_parser(cfg, tools_requested)
        elif cfg.tool_parser_instance:
            self.tool_parser = cfg.tool_parser_instance  # injected mock
        else:
            self.tool_parser = self._create_tool_parser(cfg, tools_requested)

        # State
        self.accumulated_text = ""
        self.tool_accumulated_text = ""
        # Accumulated reasoning content (split out by the reasoning parser
        # from the raw model output). Surfaced on the streaming Usage
        # chunk so clients see ``completion_tokens_details.reasoning_tokens``
        # in parity with the non-streaming response shape. v0.6.63
        # onboarding sweep finding #5.
        self.accumulated_reasoning = ""
        self.tool_calls_detected = False
        self.tool_markup_possible = False
        # Monotonic counter for structured tool-call indices across the
        # whole response. Each TOOL_CALL channel ``GenerationOutput`` may
        # carry a single structured call; if multiple chunks fire
        # separately (router emits one per ``<|call|>``) the index field
        # must keep counting up so clients can disambiguate them
        # (OpenAI spec: tool_calls deltas merge on ``index``). Codex
        # round-15 BLOCKING #1.
        self._structured_tool_call_count = 0
        # Set of tool_call indices we've already admitted under the
        # ``parallel_tool_calls`` cap. Text-parser streaming paths
        # (hermes, qwen3_coder, etc.) emit MANY deltas per logical call:
        # name first, then argument fragments, all with the same
        # ``index``. The cap consumes a slot only on the FIRST sighting
        # of a new index; subsequent deltas for an already-admitted
        # index are continuations and must pass through so the client
        # can reassemble the JSON. PR #518 codex round-1 BLOCKING.
        self._admitted_tool_call_indices: set[int] = set()
        # Parallel to the indexed-set above, but for parsers that emit
        # continuation deltas without an ``index`` field. Treated as a
        # single in-flight call: first no-index delta admits, every
        # subsequent no-index delta is forwarded as a continuation.
        # PR #518 round-2 codex BLOCKING: without this, no-index
        # continuations were re-classified as new calls and dropped
        # once the cap was full, silently truncating arguments.
        self._no_index_call_admitted: bool = False
        # Tracks whether the MOST RECENT no-index "anchor" delta (one
        # carrying a fresh ``id`` or function ``name``) was DROPPED
        # because the cap was full. Subsequent argument-only no-index
        # fragments belong to whichever anchor came last — so if the
        # last anchor was dropped, the fragments must be dropped too,
        # not silently appended to the admitted call's arguments.
        # PR #518 round-3 codex BLOCKING fix surfaced this leak; this
        # closes it. Assumes sequential parser emission (one call
        # completes before the next starts) — interleaved no-index
        # calls without ``id`` are indistinguishable from delta shape
        # alone; well-behaved parsers either disambiguate via
        # ``index``/``id`` or emit sequentially.
        self._no_index_last_dropped: bool = False

        # Nemotron thinking prefix
        self._is_thinking_model = False
        self._think_prefix_sent = False

        # JSON mode: suppress thinking preamble before JSON content (#46).
        # When json_mode=True and no reasoning parser, buffer content until
        # the first JSON delimiter ({ or [) is seen, then emit from there.
        self._json_preamble_stripped = False
        self._json_preamble_buffer = ""

    @staticmethod
    def _create_reasoning_parser(cfg: ServerConfig):
        """Create a per-request reasoning parser instance."""
        if not cfg.reasoning_parser_name:
            return None
        try:
            from ..reasoning import get_parser

            parser_cls = get_parser(cfg.reasoning_parser_name)
            return parser_cls()
        except Exception as e:
            logger.warning(f"Failed to create reasoning parser: {e}")
            return None

    @staticmethod
    def _create_tool_parser(cfg: ServerConfig, tools_requested: bool):
        """Create a per-request tool parser instance."""
        from ..tool_parsers import ToolParserManager

        tokenizer = None
        if cfg.engine is not None and hasattr(cfg.engine, "_tokenizer"):
            tokenizer = cfg.engine._tokenizer

        # Primary: explicit tool parser configured
        if cfg.enable_auto_tool_choice and cfg.tool_call_parser:
            try:
                parser_cls = ToolParserManager.get_tool_parser(cfg.tool_call_parser)
                return parser_cls(tokenizer)
            except Exception as e:
                logger.warning(f"Failed to create tool parser for streaming: {e}")

        # Fallback: auto-infer from reasoning parser
        if tools_requested and cfg.reasoning_parser_name:
            _PARSER_MAP = {"minimax": "minimax"}
            inferred = _PARSER_MAP.get(cfg.reasoning_parser_name)
            if inferred:
                try:
                    parser_cls = ToolParserManager.get_tool_parser(inferred)
                    return parser_cls(tokenizer)
                except Exception as e:
                    logger.debug(f"Auto-infer tool parser for streaming failed: {e}")

        return None

    def set_thinking_model(self, model_name: str):
        """Enable Nemotron-style thinking prefix injection."""
        self._is_thinking_model = (
            "nemotron" in model_name.lower() and not self.reasoning_parser
        )

    def _parallel_tool_calls_allowed(self) -> bool:
        """Return False iff the request explicitly opted out of
        parallel tool calls via ``parallel_tool_calls=false``.

        OpenAI spec: ``True`` and unset both mean "no cap". Only the
        explicit ``false`` triggers single-call enforcement (matches
        the non-streaming trim in ``routes/chat.py`` post-parse). The
        request may arrive as a pydantic model (production) or a dict
        (test fixtures, lifted bench scaffolds); accept both.
        """
        req = self.request
        if req is None:
            return True
        if isinstance(req, dict):
            val = req.get("parallel_tool_calls")
        else:
            val = getattr(req, "parallel_tool_calls", None)
        return val is not False

    def _apply_parallel_cap(self, tool_calls: list[dict]) -> list[dict]:
        """Filter a streaming tool_calls delta list under the
        ``parallel_tool_calls=false`` cap, distinguishing NEW tool
        calls (unseen ``index``) from CONTINUATION deltas (seen
        ``index`` — name + incremental argument fragments for an
        already-admitted call).

        Text-parser streaming paths (hermes, qwen3_coder, etc.) emit
        many deltas per logical call: a header carrying ``{index, id,
        function: {name}}``, then a sequence of deltas carrying only
        ``{index, function: {arguments: "<fragment>"}}``. PR #518 round-1
        codex BLOCKING: the prior implementation consumed a cap slot
        per delta, so the first argument fragment for index 0 took the
        only slot and every subsequent fragment of THE SAME CALL was
        dropped — silently truncating the JSON arguments mid-string.

        New rule:
          - Uncapped (parallel=true / unset): pass everything; ONLY
            track ``index`` admits (for the channel-routed branch's
            monotonic-counter math). Do NOT touch
            ``_no_index_call_admitted`` here — that field is cap-only
            state, and mutating it in the uncapped path could
            pollute cap accounting if request flags change mid-stream
            (PR #518 round-3 codex NIT).
          - Capped (parallel=false): for each delta, if its ``index``
            is already admitted, pass through (continuation). If
            ``index`` is absent AND the delta carries ONLY argument
            fragments (no new ``id`` / ``name``), treat it as a
            continuation of the in-flight no-index call. A no-index
            delta carrying a fresh ``id`` or function ``name`` is a
            NEW call — admit only if the cap allows. PR #518 round-3
            codex BLOCKING: previously, every subsequent no-index
            delta was treated as a continuation, leaking a second
            full call past the cap.
          - Cap-full new calls are dropped, AND their later
            continuations are dropped too (no admit ever fired, so
            the index/no-index slot was never taken).

        Returns the filtered list (possibly empty if every delta in
        the batch is a new call past the cap).
        """
        if self._parallel_tool_calls_allowed():
            # Still track admitted indices so the channel-routed branch
            # can use the same set when assigning its own monotonic
            # ``index`` values from the count.
            for tc in tool_calls:
                idx = tc.get("index") if isinstance(tc, dict) else None
                if isinstance(idx, int):
                    self._admitted_tool_call_indices.add(idx)
            self._structured_tool_call_count = max(
                self._structured_tool_call_count,
                len(self._admitted_tool_call_indices),
            )
            return list(tool_calls)

        allowed: list[dict] = []
        for tc in tool_calls:
            idx = tc.get("index") if isinstance(tc, dict) else None
            if isinstance(idx, int) and idx in self._admitted_tool_call_indices:
                # Continuation of an already-admitted indexed call —
                # always forward so the client's arguments JSON is
                # complete.
                allowed.append(tc)
                continue
            if idx is None and self._no_index_call_admitted:
                # Only argument-fragment deltas (no fresh ``id`` /
                # ``name``) count as continuations of the in-flight
                # no-index call. A delta carrying a new ``id`` or
                # function ``name`` is a distinct new call and must
                # run the cap check (round-3 codex BLOCKING #2).
                fn = tc.get("function") if isinstance(tc, dict) else None
                has_name = (
                    isinstance(fn, dict)
                    and isinstance(fn.get("name"), str)
                    and fn.get("name")
                )
                has_id = (
                    isinstance(tc, dict)
                    and isinstance(tc.get("id"), str)
                    and tc.get("id")
                )
                if not (has_name or has_id):
                    # Continuation fragment — route to whichever
                    # anchor was last seen. If the last anchor was a
                    # dropped new call (cap-full new-call branch
                    # below set ``_no_index_last_dropped``), suppress
                    # so the dropped call's arguments don't leak
                    # into the admitted call's payload (round-4 codex
                    # BLOCKING #1).
                    if self._no_index_last_dropped:
                        continue
                    allowed.append(tc)
                    continue
            # New call (unseen index, fresh no-index call with id/name,
            # or first no-index delta).
            already_admitted = len(self._admitted_tool_call_indices) + (
                1 if self._no_index_call_admitted else 0
            )
            if already_admitted >= 1:
                # Cap full — drop this new call AND any further
                # continuations of its index, since we never admit it.
                if idx is None:
                    # Mark so subsequent no-index argument fragments
                    # are routed to "dropped" rather than silently
                    # appended to the admitted call.
                    self._no_index_last_dropped = True
                continue
            if isinstance(idx, int):
                self._admitted_tool_call_indices.add(idx)
            else:
                # Mark the no-index slot as taken; subsequent no-index
                # deltas hit the continuation branch above. Reset the
                # dropped-anchor flag — this delta is the most recent
                # anchor and it was admitted, so its fragments belong
                # here.
                self._no_index_call_admitted = True
                self._no_index_last_dropped = False
            self._structured_tool_call_count = max(
                self._structured_tool_call_count,
                len(self._admitted_tool_call_indices)
                + (1 if self._no_index_call_admitted else 0),
            )
            allowed.append(tc)
        return allowed

    def reset(self):
        """Reset all parser states for a new stream.

        Safe for concurrent BatchedEngine requests — each PostProcessor
        instance holds its own parser instances (created in __init__).
        """
        self.accumulated_text = ""
        self.tool_accumulated_text = ""
        self.accumulated_reasoning = ""
        self.tool_calls_detected = False
        self.tool_markup_possible = False
        self._think_prefix_sent = False
        self._json_preamble_stripped = False
        self._json_preamble_buffer = ""
        self._structured_tool_call_count = 0
        self._admitted_tool_call_indices = set()
        self._no_index_call_admitted = False
        self._no_index_last_dropped = False

        if self.reasoning_parser:
            self.reasoning_parser.reset_state()
        if self.tool_parser:
            self.tool_parser.reset()

    def process_chunk(self, output: GenerationOutput) -> list[StreamEvent]:
        """Process a single engine output chunk.

        Returns a list of StreamEvents (may be empty if content is suppressed).
        """
        delta_text = output.new_text
        if not delta_text:
            # Handle finish-only chunks
            if output.finished:
                return [self._make_finish_event(output)]
            return []

        # Step 1: Separate content from reasoning
        if output.channel is not None:
            return self._process_channel_routed(delta_text, output)
        if self.reasoning_parser and self.enable_thinking is not False:
            # When enable_thinking is explicitly False, the model is told to
            # skip thinking and answer directly. Bypass the reasoning parser
            # so its implicit-think heuristic doesn't reroute the answer to
            # reasoning_content.
            return self._process_with_reasoning(delta_text, output)
        return self._process_standard(delta_text, output)

    def _process_channel_routed(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle OutputRouter models (Gemma 4 etc.) with token-level routing."""
        # Engine-surfaced structured tool calls (HarmonyStreamingRouter
        # via openai-harmony's StreamableParser). Emit a structured
        # StreamEvent directly — the router has already done the
        # parse and re-running text-based extraction over the wire
        # representation would re-introduce the round-trip lossy path
        # this refactor exists to eliminate (PR #515 codex round-12 /
        # round-14 BLOCKING — tool calls whose JSON args contain
        # literal harmony sentinels were corrupted by sentinel-
        # anchored regex parsing).
        engine_tool_calls = getattr(output, "tool_calls", None) or []
        if output.channel == "tool_call" and engine_tool_calls:
            # ``parallel_tool_calls=false`` is a hard external contract:
            # the non-streaming path caps the parsed list at one
            # (routes/chat.py); the streaming path must do the same or
            # clients with the flag set get extra calls they explicitly
            # opted out of. Drop everything past the cap on this chunk
            # AND mark ``tool_calls_detected`` so subsequent chunks
            # short-circuit before emission. Codex round-15 BLOCKING #2.
            #
            # Engine surfaces ONE complete structured call per
            # ``<|call|>`` boundary (openai-harmony StreamableParser),
            # so each entry here is a distinct logical call — no
            # continuation-delta concern (that's the text-parser path,
            # see ``_apply_parallel_cap``). PR #518 round-1: keep this
            # branch's per-entry counting but share the admitted-set
            # with the text-parser path so the response-wide counter
            # stays consistent.
            parallel_allowed = self._parallel_tool_calls_allowed()
            allowed_calls: list[dict] = []
            for tc in engine_tool_calls:
                if not parallel_allowed and len(self._admitted_tool_call_indices) >= 1:
                    break
                new_idx = self._structured_tool_call_count
                self._admitted_tool_call_indices.add(new_idx)
                self._structured_tool_call_count = new_idx + 1
                allowed_calls.append(tc)
            if not allowed_calls:
                # Cap exhausted — preserve finish semantics but skip
                # emission. The buffered_finish gate fires through the
                # existing tool_calls_detected branch below.
                self.tool_calls_detected = True
                if output.finished:
                    return [
                        StreamEvent(
                            type="finish",
                            finish_reason="tool_calls",
                            tool_calls_detected=True,
                        )
                    ]
                return []
            # Monotonic indices across the whole response so clients
            # can disambiguate calls that arrive in separate router
            # chunks. ``OpenAI`` clients merge ``tool_calls`` deltas
            # on ``index`` — colliding indices cause one call to
            # overwrite another. Codex round-15 BLOCKING #1.
            structured = []
            for offset, tc in enumerate(allowed_calls):
                idx = self._structured_tool_call_count - len(allowed_calls) + offset
                structured.append(
                    {
                        "index": idx,
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                )
            self.tool_calls_detected = True
            return [
                StreamEvent(
                    type="tool_call",
                    tool_calls=structured,
                    finish_reason="tool_calls" if output.finished else None,
                    tool_calls_detected=True,
                )
            ]

        if output.channel == "reasoning":
            content, reasoning = None, delta_text
        elif output.channel == "tool_call":
            content, reasoning = delta_text, None
        else:
            content, reasoning = delta_text, None

        # Tool call detection on content
        if self.tool_parser and content:
            result = self._detect_tool_calls(content)
            if result is None:
                # Suppressed (inside tool markup OR prefix-held partial
                # sentinel). If this was ALSO the finished chunk, we
                # still must emit a finish event so the chat route's
                # buffered_finish gate fires — otherwise the
                # defensive-elif synthetic chunk path would re-emit
                # ``accumulated_text + finalize_content``, double-counting
                # already-streamed deltas (codex round-6 BLOCKING).
                if output.finished:
                    return [
                        StreamEvent(
                            type="finish",
                            finish_reason=self._compute_finish_reason(output),
                            tool_calls_detected=self.tool_calls_detected,
                        )
                    ]
                return []
            if result.get("tool_calls"):
                # Issue #517 — apply ``parallel_tool_calls=false`` cap
                # uniformly across all streaming paths. Round-1 codex
                # BLOCKING: admit by ``index`` so continuation deltas
                # (incremental argument fragments for the same call)
                # don't each consume a slot.
                allowed_tcs = self._apply_parallel_cap(result["tool_calls"])
                if not allowed_tcs:
                    self.tool_calls_detected = True
                    if output.finished:
                        return [
                            StreamEvent(
                                type="finish",
                                finish_reason="tool_calls",
                                tool_calls_detected=True,
                            )
                        ]
                    return []
                self.tool_calls_detected = True
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                ]
            content = result.get("content", "")

        if self.tool_calls_detected:
            if output.finished:
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Sanitize
        if content:
            content = strip_special_tokens(content)
        if reasoning:
            reasoning = strip_special_tokens(reasoning)

        finish_reason = self._compute_finish_reason(output)
        if not content and not reasoning and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        # Accumulate post-sanitize so the final usage chunk can compute
        # ``completion_tokens_details.reasoning_tokens`` via _build_usage's
        # proportional split (PR #453 logic). Without this, OutputRouter
        # models (Gemma 4, harmony/gpt-oss) emit reasoning_content deltas
        # to the client but leave both accumulators empty — _build_usage
        # then sees ``reasoning_text=None`` and omits the field entirely,
        # creating stream/non-stream usage shape drift. Verified on
        # gemma-4-26b + gpt-oss-20b during the v0.6.66 onboarding sweep.
        if content:
            self.accumulated_text += content
        if reasoning:
            self.accumulated_reasoning += reasoning

        # When finish_reason is set, emit ONE finish event with content/reasoning
        # merged in to avoid double-emission.
        if finish_reason:
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    reasoning=reasoning,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        events = []
        if content:
            events.append(StreamEvent(type="content", content=content))
        if reasoning:
            events.append(StreamEvent(type="reasoning", reasoning=reasoning))
        return events

    def _process_with_reasoning(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle models with text-based reasoning parsers."""
        previous_text = self.accumulated_text
        self.accumulated_text += delta_text
        delta_msg = self.reasoning_parser.extract_reasoning_streaming(
            previous_text, self.accumulated_text, delta_text
        )

        if delta_msg is None:
            # Skip (e.g., <think> token itself)
            if output.finished:
                return [self._make_finish_event(output)]
            return []

        content = delta_msg.content
        reasoning = delta_msg.reasoning

        if reasoning:
            self.accumulated_reasoning += reasoning

        # MiniMax redirect: tool calls wrapped in <think> blocks
        if self.tool_parser and reasoning:
            _check = self.tool_accumulated_text + reasoning
            if (
                "<minimax:tool_call>" in _check
                or "<tool_call>" in _check
                or '<invoke name="' in _check
            ):
                content = (content or "") + reasoning
                reasoning = None

        # Tool call detection
        if self.tool_parser and content:
            result = self._detect_tool_calls(content)
            if result is None:
                # Suppressed (inside tool markup OR prefix-held). When
                # also the finished chunk, emit finish so the chat
                # route's buffered_finish gate fires (codex round-6
                # BLOCKING — defensive-elif duplication path).
                if output.finished:
                    return [
                        StreamEvent(
                            type="finish",
                            finish_reason=self._compute_finish_reason(output),
                            tool_calls_detected=self.tool_calls_detected,
                        )
                    ]
                return []
            if result.get("tool_calls"):
                # Issue #517 — apply ``parallel_tool_calls=false`` cap
                # uniformly across all streaming paths. Round-1 codex
                # BLOCKING: admit by ``index`` so continuation deltas
                # (incremental argument fragments for the same call)
                # don't each consume a slot.
                allowed_tcs = self._apply_parallel_cap(result["tool_calls"])
                if not allowed_tcs:
                    self.tool_calls_detected = True
                    if output.finished:
                        return [
                            StreamEvent(
                                type="finish",
                                finish_reason="tool_calls",
                                tool_calls_detected=True,
                            )
                        ]
                    return []
                self.tool_calls_detected = True
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                ]
            content = result.get("content", "")

        if self.tool_calls_detected:
            if output.finished:
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Sanitize
        if content:
            content = strip_special_tokens(content)
        if reasoning:
            reasoning = strip_special_tokens(reasoning)

        finish_reason = self._compute_finish_reason(output)
        if not content and not reasoning and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        if finish_reason:
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    reasoning=reasoning,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        events = []
        if content:
            events.append(StreamEvent(type="content", content=content))
        if reasoning:
            events.append(StreamEvent(type="reasoning", reasoning=reasoning))
        return events

    def _process_standard(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle standard models (no reasoning parser, no channel router)."""
        content = strip_special_tokens(delta_text)

        # JSON mode preamble stripping (#46): when response_format is set and
        # no reasoning parser is active, the model may emit a thinking preamble
        # (e.g. "Let me think...\n{json}") before the actual JSON. Suppress
        # everything before the first JSON delimiter.
        if (
            self.json_mode
            and not self.reasoning_parser
            and not self._json_preamble_stripped
        ):
            if content:
                self._json_preamble_buffer += content
                json_start = _find_json_start(self._json_preamble_buffer)
                if json_start >= 0:
                    self._json_preamble_stripped = True
                    content = self._json_preamble_buffer[json_start:]
                else:
                    return []

        # Nemotron thinking prefix
        if self._is_thinking_model and not self._think_prefix_sent and content:
            content = "<think>" + content
            self._think_prefix_sent = True

        # Tool call detection
        if self.tool_parser and delta_text:
            result = self._detect_tool_calls(delta_text)
            if result is None:
                # Suppressed. When also finished, emit finish so the
                # chat route's buffered_finish gate fires (codex
                # round-6 BLOCKING).
                if output.finished:
                    return [
                        StreamEvent(
                            type="finish",
                            finish_reason=self._compute_finish_reason(output),
                            tool_calls_detected=self.tool_calls_detected,
                        )
                    ]
                return []
            if result.get("tool_calls"):
                # Apply ``parallel_tool_calls=false`` cap (issue #517).
                # Round-1 codex BLOCKING: admit by ``index`` so
                # incremental argument fragments don't each consume a
                # cap slot (qwen3_coder pattern — header delta + N
                # argument-fragment deltas all share the same index).
                allowed_tcs = self._apply_parallel_cap(result["tool_calls"])
                if not allowed_tcs:
                    self.tool_calls_detected = True
                    if output.finished:
                        return [
                            StreamEvent(
                                type="finish",
                                finish_reason="tool_calls",
                                tool_calls_detected=True,
                            )
                        ]
                    return []
                self.tool_calls_detected = True
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                ]
            content = strip_special_tokens(result.get("content", ""))

        if self.tool_calls_detected:
            if output.finished:
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Filter empty
        if content is not None and content == "":
            content = None

        finish_reason = self._compute_finish_reason(output)

        if not content and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        # When finish_reason is set, emit ONE finish event with content merged in.
        # Never emit separate content + finish events — that would cause
        # double-emission of the same content and duplicate logprobs.
        if finish_reason:
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        if content:
            return [StreamEvent(type="content", content=content)]
        return []

    def finalize(self) -> list[StreamEvent]:
        """Finalize stream — flush remaining tool calls, emit corrections.

        Call after the engine stream ends.
        """
        events = []

        # Fallback tool call detection: streaming parser missed a tool call
        # that the non-stream parser can recover. The streaming code path of
        # each parser is necessarily simpler than ``extract_tool_calls`` —
        # it can't backtrack and typically only handles the canonical
        # wrapper format. ``extract_tool_calls`` has the full set of fallback
        # patterns (bare JSON, alternate XML forms, text-format degradation).
        # Running it here gives streaming the same tolerance as non-stream.
        #
        # Previously gated on ``has_pending_tool_call`` — but that gate
        # uses the SAME canonical-wrapper check as the streaming parser, so
        # by construction it can never catch what the streaming parser
        # missed. The 2026-05-20 ≥20B onboarding sweep caught gemma-4-26b
        # producing structured tool_calls in non-stream mode that the
        # streaming parser dropped on the floor; the only difference between
        # the two modes was this gate. See knowledge/guided_generation_gaps_2026-05-20.md
        # "Bug A — Streaming tool-parser coverage gap is family-wide".
        #
        # Cheap pre-check: every known tool-call format carries at least
        # one structural marker — ``<`` (XML wrappers: ``<tool_call>``,
        # ``<function=>``, ``<|tool_call>``), ``{`` (bare JSON, parameter
        # blocks), or ``[Calling`` (text-format degradation). Skipping the
        # full regex scan when none of these markers is present keeps
        # end-of-stream cost flat on plain-text responses that happened to
        # have ``tools=...`` in the request (DeepSeek pr_validate finding
        # on PR #424 — high-throughput servers with tool-enabled
        # endpoints would otherwise pay the parser cost on every reply
        # that didn't actually call a tool).
        _fallback_text = self.tool_accumulated_text or self.accumulated_text
        _has_plausible_markup = bool(_fallback_text) and (
            "<" in _fallback_text
            or "{" in _fallback_text
            or "[Calling" in _fallback_text
        )
        if (
            self.tool_parser
            and _fallback_text
            and not self.tool_calls_detected
            and _has_plausible_markup
        ):
            result = self.tool_parser.extract_tool_calls(
                _fallback_text, request=self.request
            )
            if result.tools_called:
                events.append(
                    self._build_tool_call_event(
                        {
                            "id": tc["id"],
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        }
                        for tc in result.tool_calls
                    )
                )
                self.tool_calls_detected = True
            else:
                # Cross-format fallback. The configured streaming parser is bound to
                # ONE wire format; ``parse_tool_calls`` in ``api/tool_calling.py``
                # scans every known format and recovers calls the per-parser path
                # misses (e.g. ``qwen3_xml`` is registered to ``QwenToolParser``
                # which expects JSON inside ``<tool_call>``, but Qwen3.6-35B-A3B
                # emits the ``<function=name><parameter=...>`` XML body). The
                # non-stream path at ``service/helpers.py:604`` already falls back;
                # this mirrors it on streaming. Wrapped defensively to match the
                # non-stream try/except — a parser bug must not abort the stream.
                # See #425.
                try:
                    _, fb_tcs = parse_tool_calls(_fallback_text, self.request)
                except Exception as e:
                    logger.warning(
                        "finalize cross-format fallback parser raised: %s", e
                    )
                    fb_tcs = None
                if fb_tcs:
                    logger.info(
                        "[finalize] cross-format fallback recovered %d tool_call(s); "
                        "configured parser=%r returned tools_called=False — "
                        "consider whether --tool-call-parser matches the model's wire format",
                        len(fb_tcs),
                        getattr(self.cfg, "tool_call_parser", None),
                    )
                    events.append(
                        self._build_tool_call_event(
                            {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in fb_tcs
                        )
                    )
                    self.tool_calls_detected = True

        # Release any prefix-held content trailing the stream. Hermes
        # and harmony streaming parsers hold back partial sentinel
        # suffixes (``<``, ``<|``, ``<func``...) so per-char streaming
        # doesn't leak them before the full sentinel arrives. If the
        # stream ends with bytes still held AND no tool call ever
        # fired, those bytes are ordinary content and would otherwise
        # be silently dropped (codex round-3 CRITICAL on the streaming-
        # parser cluster PR). When a tool call DID fire, the held
        # bytes are part of the tool-call body and stay suppressed.
        if (
            self.tool_parser
            and self.tool_accumulated_text
            and not self.tool_calls_detected
        ):
            held = self.tool_parser.flush_held_content(self.tool_accumulated_text)
            # Strict-string check: ``flush_held_content`` is part of the
            # parser interface and must return a real ``str``. Defending
            # against accidental ``None`` / non-string returns avoids a
            # buggy override surfacing as a malformed StreamEvent
            # downstream.
            if isinstance(held, str) and held:
                events.append(StreamEvent(type="content", content=held))

        return events

    def _build_tool_call_event(self, items) -> StreamEvent:
        """Build a tool_call StreamEvent from an iterable of {id, name, arguments} dicts.

        Used by both finalize() branches (configured parser succeeded, and the
        cross-format ``parse_tool_calls`` fallback) so the two paths can't drift
        in wire shape.
        """
        return StreamEvent(
            type="tool_call",
            tool_calls=[
                {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for i, tc in enumerate(items)
            ],
            finish_reason="tool_calls",
            tool_calls_detected=True,
        )

    def _detect_tool_calls(self, content: str) -> dict | None:
        """Run incremental tool call detection.

        Returns None if content is suppressed (inside tool markup).
        Returns {"tool_calls": [...]} if tool calls detected.
        Returns {"content": "..."} for normal content pass-through.
        """
        if not self.tool_markup_possible and "<" not in content and "[" not in content:
            self.tool_accumulated_text += content
            return {"content": content}

        if not self.tool_markup_possible:
            self.tool_markup_possible = True

        tool_previous = self.tool_accumulated_text
        self.tool_accumulated_text += content
        tool_result = self.tool_parser.extract_tool_calls_streaming(
            tool_previous,
            self.tool_accumulated_text,
            content,
            request=self.request,
        )

        if tool_result is None:
            return None  # inside tool markup

        if "tool_calls" in tool_result:
            self.tool_calls_detected = True
            return tool_result

        return {"content": tool_result.get("content", "")}

    def _compute_finish_reason(self, output: GenerationOutput) -> str | None:
        if not output.finished:
            return None
        if self.tool_calls_detected:
            return "tool_calls"
        return output.finish_reason

    def _make_finish_event(self, output: GenerationOutput) -> StreamEvent:
        return StreamEvent(
            type="finish",
            finish_reason=self._compute_finish_reason(output),
            tool_calls_detected=self.tool_calls_detected,
        )
