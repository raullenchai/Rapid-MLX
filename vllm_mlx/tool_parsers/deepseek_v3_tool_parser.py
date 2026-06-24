# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3 tool call parser for rapid-mlx (D-DSV31, R12-5).

Targets the DeepSeek-V3 "function-typed, JSON-fenced" tool-call body
shape, emitted by the upstream V3 chat template and by every checkpoint
that inherited it (most notably ``DeepSeek-R1-0528-Qwen3-8B`` — whose
``chat_template.jinja`` is V3, NOT V3.1):

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {ARGS}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

All the envelope characters are the fullwidth pipe ``｜`` (U+FF5C), not
ASCII ``|``.

Why this is a dedicated module (R12-5 follow-up to D-DSV31)
----------------------------------------------------------
The original D-DSV31 hotfix extended the V3.1 parser
(``DeepSeekV31ToolParser``) with auto-detection for the V3 wire shape.
That fix shipped on 0.8.2 and unblocked R1-0528-Qwen3-8B, but it left
the V3.1 parser carrying two unrelated wire shapes plus an end-of-stream
streaming gate that suppresses incremental ``arguments`` deltas any time
a V3 marker is seen — a footgun for anyone reading the V3.1 parser
expecting V3.1 semantics. R12-5 splits the two shapes into separate
parsers so each is single-purpose:

  * ``DeepSeekV3ToolParser``  (this module)   — V3 fenced-JSON
  * ``DeepSeekV31ToolParser`` (deepseekv31_tool_parser.py) — V3.1 plain

The shared outer envelope (``<｜tool▁calls▁begin｜>`` /
``<｜tool▁calls▁end｜>``) and block envelope (``<｜tool▁call▁begin｜>`` /
``<｜tool▁call▁end｜>``) are duplicated as constants on both classes
rather than hoisted to a common base — the two parsers are otherwise
fully independent and the constant duplication is preferable to a
shared-base coupling that would re-introduce the cross-shape blast
radius this refactor is removing.

Hardening lessons preserved from the 10 codex rounds on D-DSV31
---------------------------------------------------------------
  * Block-wise scanning bounded to the outer envelope (codex r8
    BLOCKING-1) — literal ``<｜tool▁call▁begin｜>`` text in plain content
    cannot be misparsed as a tool call.
  * Forward-scanning state machine for ``<call_begin>`` /
    ``<call_end>`` pairs rather than a single greedy regex — parallel
    calls parse as N entries, truncated trailing blocks are dropped.
  * Malformed V3 blocks return ``None`` rather than emitting tool calls
    with partial-fence arguments (codex r10 BLOCKING) — the raw text is
    surfaced via ``content`` so the caller can decide whether to
    display or retry.
  * Streaming defers to ``extract_tool_calls`` at end-of-stream via the
    postprocessor's ``tool_calls_detected`` fallback path. Per-token
    ``arguments`` deltas are NOT emitted — the V3 fenced-JSON body
    cannot be split mid-fence without producing wire-invalid partial
    JSON. This matches GLM-4.7 / Seed-OSS behaviour for their
    wrapped formats.
"""

import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


# Literal type tag emitted by the V3 chat template inside each
# ``<call_begin>...<call_end>`` body. The body shape is:
#
#     function<｜tool▁sep｜>NAME\n```json\n{ARGS}\n```
#
# A body that does NOT start with ``function<sep>`` is not V3 and the
# parser drops the block (the V3.1 parser would still parse it as a
# V3.1 body — that's the whole point of having two parsers).
_V3_TYPE_TAG = "function"


@ToolParserManager.register_module(["deepseek_v3", "deepseek_r1_0528"])
class DeepSeekV3ToolParser(ToolParser):
    """
    Tool call parser for DeepSeek-V3 family models — the
    ``function``-typed, fenced-JSON wire shape.

    Used when ``--enable-auto-tool-choice --tool-call-parser deepseek_v3``
    is set, or by ``aliases.json`` entries that point at the
    ``deepseek_v3`` / ``deepseek_r1_0528`` registry names (e.g.
    ``DeepSeek-R1-0528-Qwen3-8B``).
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("deepseek_native",)

    TOOL_CALLS_START = "<｜tool▁calls▁begin｜>"
    TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
    TOOL_CALL_START = "<｜tool▁call▁begin｜>"
    TOOL_CALL_END = "<｜tool▁call▁end｜>"
    TOOL_SEP = "<｜tool▁sep｜>"

    # Strict body: ``function<sep>NAME\n```json\n{...}\n```\s*``.
    _V3_BODY_REGEX = re.compile(
        r"^function<｜tool▁sep｜>(?P<name>.*?)\n```json\n(?P<args>.*?)\n```\s*$",
        re.DOTALL,
    )
    # Tolerant: some checkpoints omit the newline before the closing
    # fence (``...}```<call_end>``).
    _V3_BODY_REGEX_TOLERANT = re.compile(
        r"^function<｜tool▁sep｜>(?P<name>.*?)\n```json\n(?P<args>.*?)```\s*$",
        re.DOTALL,
    )

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Streaming token IDs (graceful fallback if absent from vocab).
        self.tool_calls_start_token_id = self.vocab.get(self.TOOL_CALLS_START)
        self.tool_calls_end_token_id = self.vocab.get(self.TOOL_CALLS_END)
        self.tool_call_start_token_id = self.vocab.get(self.TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(self.TOOL_CALL_END)

    # -----------------------------------------------------------------
    # Block-wise scanner.
    # -----------------------------------------------------------------
    @classmethod
    def _envelope_bounds(cls, model_output: str) -> tuple[int, int] | None:
        """Locate the outer ``<tool_calls_begin>...<tool_calls_end>``
        envelope. Returns ``(inner_start, inner_end)`` pointing at the
        substring strictly between the two markers, or ``None`` if the
        outer ``<tool_calls_begin>`` is absent.

        If the closing marker hasn't arrived yet (partial stream), the
        inner end is the length of the string — callers can still scan
        the partial inner region for already-closed blocks.

        Scanning MUST be bounded to the outer envelope; otherwise a
        response that quotes ``<｜tool▁call▁begin｜>`` as literal content
        could have that content treated as a tool call.
        """
        outer_start = model_output.find(cls.TOOL_CALLS_START)
        if outer_start == -1:
            return None
        inner_start = outer_start + len(cls.TOOL_CALLS_START)
        outer_end = model_output.find(cls.TOOL_CALLS_END, inner_start)
        inner_end = outer_end if outer_end != -1 else len(model_output)
        return (inner_start, inner_end)

    @classmethod
    def _iter_block_bodies(cls, model_output: str) -> list[str]:
        """Yield the body text between each ``<call_begin>`` /
        ``<call_end>`` pair found INSIDE the outer
        ``<tool_calls_begin>...<tool_calls_end>`` envelope, in order.

        Uses a forward-scanning state machine (find next ``<call_begin>``,
        find the matching ``<call_end>``) rather than a single greedy
        regex. This is what makes parallel calls parse as N entries
        instead of one over-greedy match, and what makes a truncated
        trailing block ignored rather than swallowing the rest of the
        payload.
        """
        bounds = cls._envelope_bounds(model_output)
        if bounds is None:
            return []
        inner_start, inner_end = bounds
        bodies: list[str] = []
        pos = inner_start
        start_len = len(cls.TOOL_CALL_START)
        end_len = len(cls.TOOL_CALL_END)
        while pos < inner_end:
            start = model_output.find(cls.TOOL_CALL_START, pos, inner_end)
            if start == -1:
                break
            body_start = start + start_len
            end = model_output.find(cls.TOOL_CALL_END, body_start, inner_end)
            if end == -1:
                # Truncated trailing block — drop and stop. The raw text
                # is still surfaced via ``content`` (see
                # ``_has_open_or_unparsed_block``).
                break
            bodies.append(model_output[body_start:end])
            pos = end + end_len
        return bodies

    @classmethod
    def _has_open_or_unparsed_block(cls, model_output: str) -> bool:
        """True if there are ``<call_begin>`` markers inside the envelope
        that ``_iter_block_bodies`` did NOT pair with a ``<call_end>``
        (truncated trailing block). Used by ``extract_tool_calls`` to
        decide whether to preserve the raw text in ``content``.
        """
        bounds = cls._envelope_bounds(model_output)
        if bounds is None:
            return False
        inner_start, inner_end = bounds
        inner = model_output[inner_start:inner_end]
        return inner.count(cls.TOOL_CALL_START) > inner.count(cls.TOOL_CALL_END)

    @classmethod
    def _parse_block_body(cls, body: str) -> tuple[str, str] | None:
        """Parse one ``<call_begin>...<call_end>`` body into
        ``(name, args)``.

        Strict V3-only contract: the body MUST start with the literal
        ``function<sep>`` type tag, OR the block is malformed and we
        return ``None``. We do NOT silently fall back to V3.1 splitting
        — that fallback is what made the original V3.1 parser emit
        ``name="function"`` with the real name embedded in
        ``arguments`` (recreating the D-DSV31 failure mode). The V3.1
        parser handles V3.1-shaped bodies; this one only handles V3.

        Malformed (V3-anchored body that matches neither fenced-JSON
        regex): return ``None`` and let the caller drop the block. A
        partial recovery here would emit a tool call with
        truncated / non-JSON arguments (codex r10 BLOCKING on the
        original D-DSV31 PR — preserved here).
        """
        body = body.strip("\n")
        if not body.startswith(f"{_V3_TYPE_TAG}{cls.TOOL_SEP}"):
            return None
        m = cls._V3_BODY_REGEX.match(body) or cls._V3_BODY_REGEX_TOLERANT.match(body)
        if m is None:
            return None
        return m.group("name").strip(), m.group("args").strip()

    # -----------------------------------------------------------------
    # Non-streaming extraction.
    # -----------------------------------------------------------------
    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.TOOL_CALLS_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            bodies = self._iter_block_bodies(model_output)
            tool_calls: list[dict[str, Any]] = []
            had_malformed = False
            for body in bodies:
                parsed = self._parse_block_body(body)
                if parsed is None:
                    had_malformed = True
                    continue
                name, args = parsed
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": name,
                        # Args body passes through verbatim. Downstream
                        # JSON canonicalisation is the caller's job
                        # (and changing the bytes here breaks
                        # ``tests/test_upstream_regression``).
                        "arguments": args,
                    }
                )

            has_truncated = self._has_open_or_unparsed_block(model_output)

            if tool_calls:
                prefix_content = model_output[
                    : model_output.find(self.TOOL_CALLS_START)
                ]
                if had_malformed or has_truncated:
                    # Surface the entire raw model output so the caller
                    # can see the malformed text rather than silently
                    # dropping it.
                    content = model_output
                else:
                    content = prefix_content
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            # Envelope present but nothing parsed — pass through raw.
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        except Exception:
            logger.exception("Error extracting V3 tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def has_pending_tool_call(self, text: str) -> bool:
        return (
            self.TOOL_CALLS_START in text
            or self.TOOL_CALL_START in text
            or self.has_text_format_tool_call(text)
        )

    # -----------------------------------------------------------------
    # Streaming.
    # -----------------------------------------------------------------
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Streaming strategy: defer to end-of-stream finalize.

        The V3 fenced-JSON body cannot be split mid-fence without
        producing wire-invalid partial JSON (clients that parse
        ``arguments`` per-delta would see ``{"city":`` and choke). We
        therefore emit nothing during streaming once the outer envelope
        is seen — the postprocessor's
        ``tool_calls_detected``-guarded finalize path
        (``vllm_mlx/service/postprocessor.py``) runs
        ``extract_tool_calls`` on the cumulative buffer at end-of-stream
        and emits a well-formed one-shot ``tool_call`` event per block.

        Tradeoff: clients receive one ``tool_call`` event per call at
        end-of-stream rather than incremental ``arguments`` deltas.
        This matches GLM-4.7 / Seed-OSS behaviour for their
        wrapped-body formats and is the conservative contract until a
        client emerges that demonstrably needs per-token V3 streaming.

        Plain content before the envelope is streamed normally so the
        client sees the model's pre-tool-call narration unchanged.
        """
        current_token_ids = current_token_ids or []

        has_tool_start = (
            self.tool_calls_start_token_id is not None
            and self.tool_calls_start_token_id in current_token_ids
        ) or self.TOOL_CALLS_START in current_text

        if not has_tool_start:
            return {"content": delta_text}

        # Envelope reached. Stop emitting; let finalize handle it.
        return None
