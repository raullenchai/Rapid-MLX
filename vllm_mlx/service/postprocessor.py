# SPDX-License-Identifier: Apache-2.0
"""Streaming post-processor — unified reasoning + tool call + sanitization pipeline.

Replaces 500+ lines of duplicated logic across stream_chat_completion,
_stream_anthropic_messages, and stream_completion. NOT a filter chain —
one cohesive orchestrator, because reasoning/tool/sanitize are tightly coupled.
"""

from __future__ import annotations

import logging
import os
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

        # Nemotron thinking prefix
        self._is_thinking_model = False
        self._think_prefix_sent = False

        # JSON mode: suppress thinking preamble before JSON content (#46).
        # When json_mode=True and no reasoning parser, buffer content until
        # the first JSON delimiter ({ or [) is seen, then emit from there.
        self._json_preamble_stripped = False
        self._json_preamble_buffer = ""

        # Optional unified Parser path (Phase 2 of the vllm-style parser
        # orchestrator migration — see ``vllm_mlx/parser/``). Activated by
        # ``RAPID_MLX_UNIFIED_PARSER=1`` so the old per-call sequence
        # (``extract_reasoning_streaming`` → ``_detect_tool_calls``) and
        # the new single-call orchestrator (``parse_delta``) coexist while
        # we bake parity. Channel-routed models (Gemma 4, harmony) still
        # take the OutputRouter path; this flag only swaps the text-based
        # reasoning + tool flow that was the source of #436 / #443 /
        # #454 / #456 / #460 / #462.
        self.unified_parser = None
        if (
            os.environ.get("RAPID_MLX_UNIFIED_PARSER", "0") == "1"
            and self.reasoning_parser is not None
        ):
            self.unified_parser = self._build_unified_parser()

        # When the unified path is active it OWNS the post-reasoning
        # ``tool_accumulated_text`` value — finalize() must treat an
        # empty string as "scoped tool text is intentionally empty" and
        # NOT fall back to the full ``accumulated_text`` (which would
        # re-run the cross-format parser over the reasoning body and
        # synthesize a bogus tool call from any bare JSON shape
        # mentioned there). Codex R9 #1.
        self._unified_tool_scope_active = self.unified_parser is not None
        # MiniMax redirect (tool calls embedded in ``<think>``) runs the
        # legacy ``_detect_tool_calls`` to mutate
        # ``tool_accumulated_text``. While that path is engaged for the
        # current stream, the per-chunk mirror from the orchestrator's
        # ``tool_phase_text`` MUST stop — it never saw the redirected
        # text and would otherwise clobber the buffer between chunks,
        # losing the opening ``<minimax:tool_call>`` marker. Codex R9 #2.
        self._minimax_redirect_active = False

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

    def _build_unified_parser(self):
        """Construct a unified ``Parser`` from the already-built reasoning
        and tool parser instances.

        Direct instantiation (rather than ``ParserManager.get_parser``)
        because the manager would re-build the parsers from class names
        and lose the request-scoped wiring this constructor already did
        (mock injection, MiniMax inference, Nemotron heuristic, etc.).
        We only borrow the orchestration shape — ``DelegatingParser.parse_delta``.
        """
        from ..parser import DelegatingParser

        parser = DelegatingParser(
            tokenizer=getattr(self.cfg, "engine", None)
            and getattr(self.cfg.engine, "_tokenizer", None)
        )
        parser._reasoning_parser = self.reasoning_parser
        parser._tool_parser = self.tool_parser
        return parser

    def set_thinking_model(self, model_name: str):
        """Enable Nemotron-style thinking prefix injection."""
        self._is_thinking_model = (
            "nemotron" in model_name.lower() and not self.reasoning_parser
        )

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
        # ``_unified_tool_scope_active`` is derived from ``unified_parser``
        # at construction time; it stays constant across resets in the
        # current stream. ``_minimax_redirect_active`` is per-stream
        # state — clear it on every reset.
        self._minimax_redirect_active = False

        if self.reasoning_parser:
            self.reasoning_parser.reset_state()
        if self.tool_parser:
            self.tool_parser.reset()
        if self.unified_parser is not None:
            # The reasoning/tool sub-parsers are reset above; only the
            # orchestrator's own StreamState needs to be cleared here.
            from ..parser import StreamState

            self.unified_parser._stream_state = StreamState()

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
            if self.unified_parser is not None:
                # Phase 2 opt-in: route the reasoning + tool sequence
                # through the unified ``Parser.parse_delta`` orchestrator
                # instead of the legacy two-call sequence. The surrounding
                # post-processing (MiniMax redirect, sanitize, accumulate,
                # finish_reason) is the same — only the boundary handling
                # changes, structurally eliminating the bug class behind
                # PRs #436 / #443 / #454 / #456 / #460 / #462.
                return self._process_with_unified_parser(delta_text, output)
            return self._process_with_reasoning(delta_text, output)
        return self._process_standard(delta_text, output)

    def _process_channel_routed(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle OutputRouter models (Gemma 4 etc.) with token-level routing."""
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
                return []  # suppressed (inside tool markup)
            if result.get("tool_calls"):
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=result["tool_calls"],
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
                return []
            if result.get("tool_calls"):
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=result["tool_calls"],
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

    def _process_with_unified_parser(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Unified-Parser variant of ``_process_with_reasoning``.

        Replaces the two-call ``extract_reasoning_streaming`` +
        ``_detect_tool_calls`` sequence with a single
        ``parse_delta(delta_text)`` call. The surrounding pipeline
        (accumulators, MiniMax tool-in-think redirect, sanitizers,
        ``finish_reason`` computation, ``StreamEvent`` shaping) is
        intentionally kept byte-identical to ``_process_with_reasoning``
        so that the only behavior difference is the boundary handling
        the orchestrator now owns.
        """
        previous_text = self.accumulated_text
        self.accumulated_text += delta_text

        delta_msg = self.unified_parser.parse_delta(
            delta_text=delta_text, request=self.request
        )

        # Mirror the orchestrator's post-reasoning ``tool_phase_text``
        # into ``self.tool_accumulated_text`` so ``finalize()``'s
        # fallback (which scans tool_accumulated_text or
        # accumulated_text) doesn't reparse the full raw stream
        # — bare JSON shapes mentioned inside ``<think>`` would
        # otherwise produce bogus end-of-stream tool_call events under
        # the unified path while the legacy ``_detect_tool_calls``
        # path is correctly scoped (codex R7).
        #
        # GATE: skip the mirror once the MiniMax redirect has engaged
        # for this stream. The redirect runs the legacy
        # ``_detect_tool_calls`` which appends the reclassified
        # reasoning text into ``tool_accumulated_text``; the unified
        # state never saw that content so mirroring would clobber the
        # opening ``<minimax:tool_call>`` marker between chunks and the
        # tool parser would lose the split tool call (codex R9 #2).
        if not self._minimax_redirect_active:
            self.tool_accumulated_text = (
                self.unified_parser._stream_state.tool_phase_text
            )

        if delta_msg is None:
            if output.finished:
                return [self._make_finish_event(output)]
            return []

        content = delta_msg.content
        reasoning = delta_msg.reasoning
        # ``parse_delta`` may emit a fully formed tool_calls payload when
        # the underlying tool parser flushes — translate that into the
        # same StreamEvent shape the legacy ``_detect_tool_calls`` path
        # produces, so downstream SSE encoding is unchanged.
        if delta_msg.tool_calls:
            # Pin tool_calls_detected so finalize() knows not to re-parse
            # accumulated_text (which would duplicate this call) and so
            # the terminal finish event is stamped "tool_calls" rather
            # than "stop". Mirrors the legacy ``_detect_tool_calls``
            # branch that sets the same flag.
            self.tool_calls_detected = True

            # Coalesced boundary chunk — the orchestrator may attach
            # reasoning / content from the pre-boundary side of the
            # same delta. Surface those first (mirroring the legacy
            # path which would have emitted them on the previous
            # process_chunk call) so clients don't lose the final
            # reasoning text or partial content that preceded the
            # tool call.
            events: list[StreamEvent] = []
            if reasoning:
                cleaned_reasoning = strip_special_tokens(reasoning)
                if cleaned_reasoning:
                    self.accumulated_reasoning += cleaned_reasoning
                    events.append(
                        StreamEvent(type="reasoning", reasoning=cleaned_reasoning)
                    )
            if content:
                cleaned_content = strip_special_tokens(content)
                if cleaned_content:
                    cleaned_content = sanitize_output(cleaned_content)
                if cleaned_content:
                    events.append(StreamEvent(type="content", content=cleaned_content))

            events.append(
                StreamEvent(
                    type="tool_call",
                    tool_calls=delta_msg.tool_calls,
                    finish_reason="tool_calls" if output.finished else None,
                    tool_calls_detected=True,
                )
            )
            return events

        if reasoning:
            self.accumulated_reasoning += reasoning

        # MiniMax redirect — same heuristic as the legacy path. parse_delta
        # routes ``<think>...<tool_call>`` reasoning into ``reasoning`` and
        # leaves the tool detection for the tool parser's text inspection,
        # but MiniMax wraps its tool calls inside the think block so the
        # tool parser would never see them. Reclassify reasoning → content
        # on the trigger tags so the next delta's tool parser run can
        # consume them.
        if self.tool_parser and reasoning:
            _check = self.tool_accumulated_text + reasoning
            if (
                "<minimax:tool_call>" in _check
                or "<tool_call>" in _check
                or '<invoke name="' in _check
            ):
                # Latch the redirect for the rest of this stream so the
                # per-chunk mirror to ``tool_accumulated_text`` stays
                # out of the way (codex R9 #2). ``_detect_tool_calls``
                # is now the source of truth for the buffer.
                self._minimax_redirect_active = True
                content = (content or "") + reasoning
                reasoning = None
                # Re-run tool detection on the now-content portion so the
                # tool call surfaces in this delta — the legacy path
                # called ``_detect_tool_calls`` unconditionally below; we
                # keep that contract.
                result = self._detect_tool_calls(content)
                if result is None:
                    return []
                if result.get("tool_calls"):
                    return [
                        StreamEvent(
                            type="tool_call",
                            tool_calls=result["tool_calls"],
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
                return []
            if result.get("tool_calls"):
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=result["tool_calls"],
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
        # When the unified Parser path owns the post-reasoning tool
        # scope, ``tool_accumulated_text`` is authoritative — including
        # the "" case which means "nothing post-reasoning to parse".
        # The legacy ``or self.accumulated_text`` fallback would
        # otherwise re-run ``extract_tool_calls`` over the reasoning
        # body and synthesize a bogus tool call from any bare JSON
        # shape mentioned there (codex R9 #1). The MiniMax redirect
        # exception (``_minimax_redirect_active``) lets the legacy
        # ``_detect_tool_calls`` path keep ownership when it engages —
        # in that mode ``tool_accumulated_text`` is non-empty and the
        # fallback would behave correctly anyway, so we still take the
        # scoped value.
        if self._unified_tool_scope_active:
            _fallback_text = self.tool_accumulated_text
        else:
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
