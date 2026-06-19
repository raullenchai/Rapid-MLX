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
        reasoning_max_tokens: int | None = None,
    ):
        self.cfg = cfg
        self.tools_requested = tools_requested
        self.json_mode = json_mode
        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # When set and the model is still emitting on the reasoning
        # channel after this many tokens, the processor force-closes
        # the channel: text-parser engines see an injected ``</think>``
        # marker on the next chunk so subsequent text routes to content;
        # channel-routed engines (gemma4 / harmony) reclassify further
        # reasoning deltas as content. ``None`` means "no cap" and is
        # the documented default.
        self._reasoning_max_tokens = reasoning_max_tokens
        # Approximate count of reasoning tokens we've emitted so far.
        # Engine deltas don't always carry per-channel token counts, so
        # we approximate from the text length divided by 4 (the OpenAI
        # spec's documented chars→tokens heuristic — same constant
        # ``_build_usage`` uses in helpers.py for the reasoning_tokens
        # split). This intentionally tracks EMITTED reasoning, not the
        # raw model output, so the cap counts what the client sees.
        self._reasoning_tokens_emitted = 0
        # Flag: cap was hit. Set once the running count crosses the
        # threshold; once True, the channel-routed branch reclassifies
        # subsequent reasoning chunks as content and the text-parser
        # branch injects ``</think>`` into the parser stream so it
        # flips to content. Single-bit latch — never reset within a
        # request (the cap is monotonic).
        self._reasoning_cap_hit = False
        # Whether the text-parser injection has already fired. Idempotent
        # guard so we don't keep stuffing ``</think>`` into every
        # subsequent chunk after the cap fired.
        self._reasoning_close_injected = False
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
        # Identity of the admitted no-index call. Some parsers re-emit
        # the same ``id`` / function ``name`` on every cumulative
        # argument-update delta (rather than emitting an anchor once
        # and bare-argument continuations after). Round-10 codex
        # BLOCKING: without remembering the admitted identity, the
        # repeated anchor was misclassified as a NEW call and dropped
        # under ``parallel_tool_calls=false``, truncating the JSON.
        # Set together with ``_no_index_call_admitted`` on admit;
        # cleared on ``reset()``.
        self._no_index_admitted_id: str | None = None
        self._no_index_admitted_name: str | None = None
        # Tracks whether the MOST RECENT anchor delta (one carrying a
        # fresh ``id`` / function ``name`` / new ``index``) was DROPPED
        # because the cap was full. Subsequent argument-only no-index
        # fragments belong to whichever anchor came last — so if the
        # last anchor was dropped, the fragments must be dropped too,
        # not silently appended to the admitted call's arguments.
        # Reset on every admit (indexed or no-index). Set on every
        # cap-full drop (indexed or no-index). PR #518 round-3 first
        # surfaced the leak; round-6 codex widened the set to also
        # cover indexed dropped anchors (name kept ``no_index`` for
        # backwards refs, but semantically tracks "last anchor was
        # dropped"). Assumes sequential parser emission — interleaved
        # no-index continuations of distinct admitted indexed calls
        # are indistinguishable from delta shape alone; well-behaved
        # parsers either disambiguate via ``index``/``id`` or emit
        # sequentially.
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

    def _consume_reasoning_budget(self, reasoning_text: str) -> tuple[str, str]:
        """Account for ``reasoning_text`` against the per-request cap.

        Returns ``(reasoning_kept, content_overflow)``:

        * ``reasoning_kept`` — the portion that fits under the cap; this
          is emitted as ``reasoning_content`` to the client.
        * ``content_overflow`` — the portion past the cap; the caller
          re-routes it to the CONTENT channel so no model output is
          silently dropped.

        Codex round-12 BLOCKING #1: cumulative-CHARACTER accounting
        (not per-chunk ceiling). The earlier draft converted each
        chunk to ``max(1, ceil(len/4))`` tokens, which made 4
        one-character reasoning deltas consume 4 "tokens" while the
        SAME 4 characters consume 1 token when chunked together. The
        cap then depended on engine chunking — a transient SSE flush
        could fire the cap pages earlier than expected. Fix: track
        cumulative reasoning chars and compare against ``cap * 4``
        (same character ceiling the non-stream
        ``_apply_reasoning_cap`` uses). All chunking patterns yield
        identical cap-firing positions, matching the non-stream path.

        Sets ``_reasoning_cap_hit`` to True the moment the running
        char count meets-or-exceeds ``cap * 4``.
        ``_reasoning_max_tokens=None`` short-circuits to "no cap".
        """
        if self._reasoning_max_tokens is None or not reasoning_text:
            return reasoning_text, ""
        if self._reasoning_cap_hit:
            # Cap already fired — anything still arriving on the
            # reasoning channel is overflow content.
            return "", reasoning_text
        max_chars = self._reasoning_max_tokens * 4
        # ``_reasoning_tokens_emitted`` actually stores CHARACTERS
        # post-round-12 (the field name is kept for backward-compat
        # with downstream usage-block consumers that grep for the
        # symbol — the value is still divided by 4 for the
        # ``completion_tokens_details.reasoning_tokens`` derivation).
        new_total_chars = self._reasoning_tokens_emitted + len(reasoning_text)
        if new_total_chars < max_chars:
            self._reasoning_tokens_emitted = new_total_chars
            return reasoning_text, ""
        if new_total_chars == max_chars:
            # Exact-boundary fit: the current chunk uses up the budget
            # but doesn't overflow. Keep it as reasoning AND latch the
            # cap so the NEXT incoming chunk is rerouted / triggers the
            # ``</think>`` injection. Codex round-2 BLOCKING #1.
            self._reasoning_tokens_emitted = new_total_chars
            self._reasoning_cap_hit = True
            return reasoning_text, ""
        # Cap crosses inside this chunk. Split at the remaining char
        # budget so the kept prefix stays under the ceiling and the
        # rest spills to content.
        remaining_chars = max_chars - self._reasoning_tokens_emitted
        keep_chars = max(0, remaining_chars)
        kept = reasoning_text[:keep_chars]
        overflow = reasoning_text[keep_chars:]
        self._reasoning_tokens_emitted = max_chars
        self._reasoning_cap_hit = True
        return kept, overflow

    def _maybe_inject_reasoning_close(self, delta_text: str) -> str:
        """Inject ``</think>`` once into the next model-text chunk when
        the cap fires on a text-parser engine.

        Text-parser engines (hermes / qwen3 / glm47) emit
        ``<think>...</think>`` themselves and rely on the streaming
        reasoning parser to split content from reasoning. Once the cap
        fires, we forge the close marker so the parser flips to content
        on the very next call to ``extract_reasoning_streaming`` —
        mirrors the channel-routed engines' force-close behavior so the
        client-visible semantic is identical across parser families.

        Codex round-10 BLOCKING #1: the latch
        (``_reasoning_close_injected = True``) used to flip HERE,
        BEFORE the parser call. If the parser then raised on the
        injected chunk, the next chunk would still see the latch set
        and skip injection — leaving the parser permanently mid-think.
        The latch is now flipped in the CALLER
        (``_process_with_reasoning``) AFTER the parser call succeeds.
        This function still gates on the latch (idempotency) and
        prepends the marker, but no longer mutates state.
        """
        if not self._reasoning_cap_hit or self._reasoning_close_injected:
            return delta_text
        if self.reasoning_parser is None:
            # Standard / channel-routed path doesn't need the injection.
            return delta_text
        # Prepend the marker so the parser sees ``</think>`` BEFORE the
        # next body bytes. The caller flips ``_reasoning_close_injected``
        # only after the parser call succeeds.
        return "</think>" + delta_text

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
            fn = tc.get("function") if isinstance(tc, dict) else None
            has_wrapped_name = (
                isinstance(fn, dict)
                and isinstance(fn.get("name"), str)
                and fn.get("name")
            )
            # Round-8 codex BLOCKING #2: parsers can emit FLAT-shape
            # tool calls (``{"name": "X", "arguments": ...}`` — no
            # ``function`` wrapper, mirrored from raw engine output
            # via ``_tool_call_name`` shape #3 in chat.py). Without
            # the top-level ``name`` check, a flat-shape second call
            # was misclassified as a continuation and leaked past
            # the ``parallel_tool_calls=false`` cap.
            has_flat_name = (
                isinstance(tc, dict)
                and isinstance(tc.get("name"), str)
                and tc.get("name")
            )
            has_id = (
                isinstance(tc, dict) and isinstance(tc.get("id"), str) and tc.get("id")
            )
            is_anchor = bool(has_wrapped_name or has_flat_name or has_id)

            if isinstance(idx, int) and idx in self._admitted_tool_call_indices:
                # Continuation of an already-admitted indexed call —
                # always forward so the client's arguments JSON is
                # complete. Round-9 codex BLOCKING #2: seeing a fresh
                # continuation of an admitted indexed call signals
                # that the in-flight call is still alive, so reset
                # the dropped-anchor flag — otherwise a NO-INDEX
                # argument fragment immediately following this
                # indexed continuation would be wrongly dropped as
                # "belongs to a dropped call" when it really belongs
                # to THIS admitted call.
                self._no_index_last_dropped = False
                allowed.append(tc)
                continue

            # No-index anchor matching the admitted no-index call's
            # identity: cumulative argument-update parsers re-emit
            # ``{"id": "<same>", "function": {"name": "<same>",
            # "arguments": "<grew>"}}`` on every delta rather than
            # emitting a single anchor and bare-argument continuations.
            # Without this branch, every such re-emission would be
            # mis-classified as a new call and dropped under
            # ``parallel_tool_calls=false`` (round-10 codex BLOCKING #2).
            # Match if BOTH the delta and the admitted call carry id
            # AND ids match, OR if id is absent on the delta and the
            # function names match — never silently accept a different
            # call identity as continuation.
            if idx is None and is_anchor and self._no_index_call_admitted:
                delta_id = tc.get("id") if has_id else None
                delta_name = (
                    fn.get("name")
                    if has_wrapped_name
                    else (tc.get("name") if has_flat_name else None)
                )
                id_matches = (
                    delta_id is not None
                    and self._no_index_admitted_id is not None
                    and delta_id == self._no_index_admitted_id
                )
                name_matches_no_id_conflict = (
                    delta_id is None
                    and delta_name is not None
                    and self._no_index_admitted_name is not None
                    and delta_name == self._no_index_admitted_name
                )
                if id_matches or name_matches_no_id_conflict:
                    self._no_index_last_dropped = False
                    allowed.append(tc)
                    continue

            # Argument-only no-index fragment: routes to whichever
            # anchor was most recently seen. Any admitted call (indexed
            # OR no-index slot) keeps the fragment unless the most
            # recent anchor was dropped.
            #
            # Round-5 codex BLOCKING #2: previously this branch only
            # fired when ``_no_index_call_admitted`` was True. An
            # indexed FIRST delta (e.g. ``{"index": 0, "id": "a",
            # "function": {"name": "a", "arguments": "{"}}``) followed
            # by argument-only no-index deltas (``{"function":
            # {"arguments": "}"}}``) routed the fragments to the
            # new-call cap-check and dropped them as cap-full —
            # truncating the JSON. Now any admitted call (indexed
            # or no-index) absorbs no-index argument fragments.
            if idx is None and not is_anchor:
                has_admitted_call = bool(self._admitted_tool_call_indices) or (
                    self._no_index_call_admitted
                )
                if has_admitted_call:
                    if self._no_index_last_dropped:
                        # Most recent anchor was dropped; suppress so
                        # the dropped call's args don't leak into the
                        # admitted call's payload.
                        continue
                    allowed.append(tc)
                    continue
                # Falls through to new-call branch (first delta of the
                # stream has no index AND no anchor — treat as new).

            # New call: unseen index, fresh no-index call with id/name,
            # or first no-index delta with no admitted call yet.
            already_admitted = len(self._admitted_tool_call_indices) + (
                1 if self._no_index_call_admitted else 0
            )
            if already_admitted >= 1:
                # Cap full — drop this new call AND any further
                # continuations of its index, since we never admit it.
                # Mark so subsequent no-index argument-only fragments
                # are routed to "dropped" rather than silently
                # appended to the admitted call. Round-6 codex
                # BLOCKING: previously this flag was only set when
                # the dropped anchor was no-index, so an INDEXED
                # dropped anchor would leave the flag clear and the
                # next no-index argument fragment would leak into
                # the admitted call's payload.
                self._no_index_last_dropped = True
                continue
            if isinstance(idx, int):
                self._admitted_tool_call_indices.add(idx)
                # Indexed admit: subsequent no-index argument fragments
                # belong to the in-flight admitted call. Reset the
                # dropped-anchor flag (the cap-full branch above is
                # the only writer).
                self._no_index_last_dropped = False
            else:
                # Mark the no-index slot as taken; subsequent no-index
                # deltas hit the continuation branch above. Reset the
                # dropped-anchor flag — this delta is the most recent
                # anchor and it was admitted, so its fragments belong
                # here. Capture the admitted identity (id + name) so a
                # later anchor delta carrying the SAME id/name (parsers
                # that re-emit the anchor with cumulative arguments) is
                # matched as a continuation rather than misclassified
                # as a new call. PR #518 round-10 codex BLOCKING #2.
                self._no_index_call_admitted = True
                self._no_index_last_dropped = False
                if has_id:
                    self._no_index_admitted_id = tc.get("id")
                if has_wrapped_name:
                    self._no_index_admitted_name = fn.get("name")
                elif has_flat_name:
                    self._no_index_admitted_name = tc.get("name")
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
        self._no_index_admitted_id = None
        self._no_index_admitted_name = None
        self._no_index_last_dropped = False
        # Per-request reasoning-cap counters reset to baseline. The
        # configured cap itself (``self._reasoning_max_tokens``) is
        # immutable — it was set at __init__ from the request.
        self._reasoning_tokens_emitted = 0
        self._reasoning_cap_hit = False
        self._reasoning_close_injected = False

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
                # Defense in depth: include the no-index slot in the
                # cap total even though a single stream rarely hits
                # both the channel-routed AND text-parser paths
                # (channel-routed is gated on ``output.channel`` being
                # set, which only happens for OutputRouter models).
                # Round-5 codex BLOCKING #1: if any future flow lets
                # cross-pollination happen, the cap would leak.
                already_admitted = len(self._admitted_tool_call_indices) + (
                    1 if self._no_index_call_admitted else 0
                )
                if not parallel_allowed and already_admitted >= 1:
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

        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # When the reasoning budget is exhausted, route the overflow
        # portion of the current chunk — and every subsequent reasoning
        # chunk — to the content channel instead of dropping it.
        # Channel-routed engines (gemma4 / harmony) DON'T need a
        # ``</think>`` injection since channels are tracked at the
        # token level upstream; reclassifying the chunk is enough.
        if reasoning is not None:
            kept_reasoning, overflow_content = self._consume_reasoning_budget(reasoning)
            reasoning = kept_reasoning or None
            if overflow_content:
                content = (content or "") + overflow_content

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
                # When the streaming parser carries BOTH a content
                # delta AND a tool-call delta in one return (one
                # delta carried ``preface + tool_close`` — codex r4
                # BLOCKING on llama parser), emit the content event
                # before the tool_call event so wire order matches
                # what the client would see if the two had arrived
                # in separate deltas. Other parsers that return only
                # one channel are unaffected.
                mixed_content = result.get("content")
                events: list[StreamEvent] = []
                if isinstance(mixed_content, str) and mixed_content:
                    events.append(StreamEvent(type="content", content=mixed_content))
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                )
                return events
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
        # gemma-4-26b-4bit + gpt-oss-20b-mxfp4-q8 during the v0.6.66 onboarding sweep.
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
        # If the reasoning cap fired on a prior chunk, splice ``</think>``
        # into the parser's view of the stream so it flips to content on
        # this call. Idempotent — only fires once per request.
        #
        # Codex round-8 BLOCKING #1: keep the synthetic ``</think>``
        # marker OUT of the shared ``self.accumulated_text``. The
        # earlier draft mutated ``delta_text`` to ``"</think>" +
        # delta_text`` and then appended that mutated value to
        # ``self.accumulated_text`` — poisoning the buffer with forged
        # model bytes that downstream (usage chars-÷4 in chat.py, the
        # ``finalize()`` tool-call fallback) would see and account
        # against. Build the parser's ``current`` argument LOCALLY
        # from the (true) ``previous_text`` + the injected marker +
        # the ORIGINAL ``delta_text``. The shared buffer only ever
        # holds real model output. Symmetric with the routes-side
        # local-buffer pattern (round-6 fix).
        original_delta_text = delta_text
        previous_text = self.accumulated_text
        parser_delta_text = self._maybe_inject_reasoning_close(original_delta_text)
        injected_this_chunk = parser_delta_text is not original_delta_text
        if not injected_this_chunk:
            # No injection — common path. Keep the shared buffer
            # update minimal.
            self.accumulated_text += original_delta_text
            parser_current = self.accumulated_text
        else:
            # Injection fired this chunk: parser sees ``</think>`` +
            # ``original_delta``; shared buffer only gets the original.
            self.accumulated_text += original_delta_text
            parser_current = previous_text + parser_delta_text
        try:
            delta_msg = self.reasoning_parser.extract_reasoning_streaming(
                previous_text, parser_current, parser_delta_text
            )
        except Exception:
            # Codex round-10 BLOCKING #1: if the parser raises on a
            # chunk that carried the injected ``</think>``, do NOT
            # flip the ``_reasoning_close_injected`` latch — let the
            # NEXT chunk retry the forced transition. Re-raise so the
            # caller can decide (a transient parser bug is still a
            # bug; just don't lose retry on the cap-flush path).
            raise
        if injected_this_chunk:
            # Parser flip succeeded this chunk — latch so subsequent
            # chunks don't re-inject. Latch flip lives HERE (not in
            # ``_maybe_inject_reasoning_close``) so a parser exception
            # on the injection-carrying chunk leaves the latch clear
            # and the next chunk retries.
            self._reasoning_close_injected = True

        if delta_msg is None:
            # Skip (e.g., <think> token itself)
            if output.finished:
                return [self._make_finish_event(output)]
            return []

        content = delta_msg.content
        reasoning = delta_msg.reasoning

        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # Account for any reasoning bytes this chunk produced. Overflow
        # is rerouted to content so no model output is silently dropped
        # and the SSE stream gets a clean transition from reasoning to
        # content even when the parser hasn't actually seen ``</think>``.
        #
        # Codex round-9 BLOCKING #1: when overflow is produced on the
        # cap-crossing chunk and the parser hasn't yet seen
        # ``</think>``, the parser is still LOGICALLY mid-think.
        # Emitting overflow as content from that state leaks
        # still-in-thinking bytes onto the wire as
        # ``delta.content`` on the chat stream. Symmetric with the
        # routes-side fix (round-7 + round-8): force the parser flip
        # in THIS same chunk with a synthetic ``</think>`` against a
        # LOCAL ``current`` (don't pollute ``self.accumulated_text`` —
        # round-8 invariant). Only promote overflow to content when
        # the flip succeeds; suppress on flip failure rather than
        # mixing channels under a broken state machine.
        if reasoning:
            # Capture the FULL original reasoning text the parser
            # returned BEFORE the cap truncates it. We need this to
            # position the synthetic ``</think>`` marker at the
            # CAP BOUNDARY (between kept and overflow) on the flip
            # call below — not after the full over-budget chunk.
            full_reasoning = reasoning
            kept_reasoning, overflow_content = self._consume_reasoning_budget(reasoning)
            reasoning = kept_reasoning or None
            if overflow_content:
                flip_succeeded = self._reasoning_close_injected
                if not self._reasoning_close_injected:
                    # Codex round-10 BLOCKING #1: only mark the close-
                    # injected latch AFTER a SUCCESSFUL parser flip.
                    # If the parser raises, we want the NEXT chunk to
                    # retry the forced transition rather than skipping
                    # it forever — otherwise a transient parser bug
                    # leaves the parser permanently mid-think for the
                    # rest of the request.
                    #
                    # Codex round-13 BLOCKING #1: position the
                    # synthetic ``</think>`` AT THE CAP BOUNDARY (not
                    # after the full over-budget chunk). The earlier
                    # draft built ``flip_previous = self.accumulated_text``
                    # which included the OVERFLOW bytes — the parser
                    # was asked to close AFTER the over-budget bytes
                    # rather than at the kept-reasoning boundary,
                    # which would let stateful parsers mis-classify
                    # the overflow as still-in-thinking. Build the
                    # flip from ``previous_text + kept_reasoning`` —
                    # this represents the model output "up to the cap
                    # firing point" from the parser's POV.
                    flip_previous = previous_text + kept_reasoning
                    flip_delta = "</think>"
                    flip_current = flip_previous + flip_delta
                    try:
                        flip_msg = self.reasoning_parser.extract_reasoning_streaming(
                            flip_previous, flip_current, flip_delta
                        )
                        self._reasoning_close_injected = True
                        flip_succeeded = True
                    except Exception as e:
                        logger.warning(
                            "postprocessor in-chunk close-marker flip raised "
                            "on %r: %s — parser state may stay mid-think; "
                            "suppressing %d-byte overflow on this chunk; "
                            "next chunk will retry the forced transition",
                            type(self.reasoning_parser).__name__,
                            e,
                            len(overflow_content),
                        )
                        flip_msg = None
                    flip_content = (
                        getattr(flip_msg, "content", None)
                        if flip_msg is not None
                        else None
                    )
                    if isinstance(flip_content, str) and flip_content:
                        content = (content or "") + flip_content
                if flip_succeeded:
                    content = (content or "") + overflow_content
            # ``full_reasoning`` only needed within this block; release
            # the reference to drop the temporary view.
            del full_reasoning

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
                # Mixed content+tool delta — emit content event before
                # tool_call event (codex r4 BLOCKING on llama parser).
                mixed_content = result.get("content")
                events: list[StreamEvent] = []
                if isinstance(mixed_content, str) and mixed_content:
                    events.append(StreamEvent(type="content", content=mixed_content))
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                )
                return events
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
                # Mixed content+tool delta — emit content before
                # tool_call (codex r4 BLOCKING on llama parser).
                mixed_content = result.get("content")
                events: list[StreamEvent] = []
                if isinstance(mixed_content, str) and mixed_content:
                    mixed_content = strip_special_tokens(mixed_content)
                    if mixed_content:
                        events.append(
                            StreamEvent(type="content", content=mixed_content)
                        )
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_calls=allowed_tcs,
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                )
                return events
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

        # Codex round-3 BLOCKING #1: when the per-request reasoning cap
        # latches on the LAST reasoning chunk of the stream (model stops
        # immediately at the budget, or stops within the exact-boundary
        # chunk), no subsequent ``process_chunk`` call ever runs the
        # ``</think>`` injection — the parser is left mid-think, any
        # held content past the cap stays buffered, and the client sees
        # a reasoning-only response with no visible answer. Force the
        # injection here so a terminal cap-hit still flips the parser
        # to content and any held bytes are flushed as a final content
        # delta. Idempotent via ``_reasoning_close_injected`` so a
        # mid-stream injection on a normal chunk doesn't double-fire.
        if (
            self.reasoning_parser is not None
            and self._reasoning_cap_hit
            and not self._reasoning_close_injected
        ):
            self._reasoning_close_injected = True
            previous_text = self.accumulated_text
            injected_delta = "</think>"
            # Codex round-5 BLOCKING #1: build the parser's view of
            # ``current`` LOCALLY rather than mutating
            # ``self.accumulated_text``. Downstream (routes/chat.py
            # post-finalize usage assembly) reads ``accumulated_text``
            # to compute the chars-÷4 reasoning split for the usage
            # block. Appending the forged ``</think>`` to the shared
            # buffer would (a) make the usage tokens differ by 2 from
            # what was actually streamed, and (b) — more importantly —
            # if any future code path runs the parser's non-stream
            # ``finalize_streaming`` over ``accumulated_text``, it
            # would re-emit the same buffered bytes the in-finalize
            # injection just released. Keep the mutation scoped.
            local_current = previous_text + injected_delta
            delta_msg = None
            try:
                delta_msg = self.reasoning_parser.extract_reasoning_streaming(
                    previous_text, local_current, injected_delta
                )
            except Exception as e:
                # Codex round-5 BLOCKING #2 / #3: a parser exception on
                # the forced close path is an INTERNAL server failure,
                # not a model answer. The earlier draft emitted a
                # diagnostic string ``"[reasoning cap hit — parser
                # flush failed]"`` directly into ``content``, which
                # leaks server implementation details into the
                # assistant message. Drop the fabrication and log only;
                # the client sees an empty completion if the cap path
                # fails (the route's existing error semantics handle
                # truly catastrophic failures via 5xx).
                logger.warning(
                    "finalize close-marker injection raised on %r: %s — "
                    "trailing reasoning content (if any) will not be "
                    "promoted to content for this request",
                    type(self.reasoning_parser).__name__,
                    e,
                )
            if delta_msg is not None:
                trailing_content = getattr(delta_msg, "content", None)
                if isinstance(trailing_content, str) and trailing_content:
                    trailing_content = sanitize_output(
                        strip_special_tokens(trailing_content)
                    )
                    if trailing_content:
                        events.append(
                            StreamEvent(type="content", content=trailing_content)
                        )

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
        # missed. The 2026-05-20 ≥20B onboarding sweep caught gemma-4-26b-4bit
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
            # The hardcoded ``<``/``[`` heuristic catches every parser
            # whose wire markers open with one of those chars. The
            # gemma4 stripped wire form is the exception: on
            # DiffusionGemma, HF's ``tokenizer.decode(skip_special_
            # tokens=True)`` removes the ``<|tool_call>``/``<tool_call|>``
            # outer wrappers, so what reaches the postprocessor is the
            # bare body ``call:NAME{...}`` — no ``<``, no ``[``. Without
            # the parser-level fallback below, those deltas would slip
            # straight through this fast-path as plain ``content`` and
            # leak ``call:calculator{expression:432+1}``-style raw wire
            # text to the SSE client (regression reported via vnsh.dev
            # share probe 2026-06-11, PR #558).
            candidate = self.tool_accumulated_text + content
            pending = False
            if self.tool_parser is not None:
                _check = getattr(self.tool_parser, "has_pending_tool_call", None)
                if callable(_check):
                    try:
                        pending = bool(_check(candidate))
                    except Exception:
                        pending = False
            if not pending:
                self.tool_accumulated_text += content
                return {"content": content}
            # Parser sees in-flight markup with non-``<``/``[`` opener
            # (the gemma4 stripped form). Fall through to the full
            # streaming path so it can suppress / emit structured
            # tool_calls instead of leaking the body as content.
            self.tool_markup_possible = True

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
