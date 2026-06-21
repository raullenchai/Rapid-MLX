# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3 / V3.1 / R1-0528 tool call parser for rapid-mlx.

Originally ported from vLLM upstream
(vllm/tool_parsers/deepseekv31_tool_parser.py) targeting the V3.1
"thinking-channel" body format:

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>ARGS<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

Real-world checkpoints (incl. DeepSeek-R1-0528-Qwen3-8B, whose
``chat_template.jinja`` was inherited from DeepSeek-V3) emit the V3
"function-typed, JSON-fenced" body shape instead:

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {ARGS}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

Both shapes share the *outer* envelope (``<｜tool▁calls▁begin｜>`` /
``<｜tool▁calls▁end｜>``) and the *block* envelope (``<｜tool▁call▁begin｜>``
/ ``<｜tool▁call▁end｜>``) plus the ``<｜tool▁sep｜>`` separator — only the
body inside each ``<call_begin>...<call_end>`` block differs.

D-DSV31 (0.8.2 P0): the original V3.1-only regex matched
``function<sep>NAME\\n```json\\n{...}\\n```<end>`` as
``name="function"``, ``arguments="NAME\\n```json\\n{...}\\n```"`` — i.e.
the type tag became the function name and the real name + fenced JSON
became the arguments string. Downstream OpenAI clients then attempted to
invoke a tool literally named ``function`` with garbage arguments.

Fix: parse each ``<call_begin>...<call_end>`` block with explicit
format auto-detection on the block body. The block-wise scanner walks
the outer envelope as a state machine instead of relying on a single
greedy regex over the full payload — that's what makes parallel /
malformed payloads parse correctly rather than collapsing into one
mis-shaped match.
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


# Body shape inside a single <call_begin>...<call_end> block.
#
# V3 (DeepSeek-V3, DeepSeek-R1-0528-Qwen3-8B, and any checkpoint whose
# chat template inherits V3's "function-typed, JSON-fenced" tool-call
# emission):
#
#     function<｜tool▁sep｜>NAME\n```json\n{...}\n```
#
# V3.1 (DeepSeek V3.1 thinking-channel):
#
#     NAME<｜tool▁sep｜>{...}
#
# We detect V3 by checking whether the body starts with the literal
# ``function`` type tag *followed by* the separator (anchored, no
# regex needed) — that's what the V3 chat template always emits and
# what V3.1 never emits. Without that anchor a V3.1 tool literally
# named ``function_lookup`` would be misclassified.
_V3_TYPE_TAG = "function"


@ToolParserManager.register_module(["deepseek_v31", "deepseek_r1_0528"])
class DeepSeekV31ToolParser(ToolParser):
    """
    Tool call parser for DeepSeek V3 / V3.1 / R1-0528 models.

    Auto-detects both wire shapes (see module docstring) — a single
    parser covers every DeepSeek alias whose chat template emits the
    shared ``<｜tool▁calls▁begin｜>`` envelope, regardless of whether
    the body inside each block uses the V3 ``function<sep>NAME\\n\\`\\`\\`json
    {...}\\`\\`\\`\\n`` shape or the V3.1 ``NAME<sep>{...}`` shape.

    Used when --enable-auto-tool-choice --tool-call-parser deepseek_v31
    (or the V3 aliases) are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("deepseek_native", "deepseek_v31_native")

    TOOL_CALLS_START = "<｜tool▁calls▁begin｜>"
    TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
    TOOL_CALL_START = "<｜tool▁call▁begin｜>"
    TOOL_CALL_END = "<｜tool▁call▁end｜>"
    TOOL_SEP = "<｜tool▁sep｜>"

    # V3 body: "function<sep>NAME\n```json\n{...}\n```"
    # The fenced JSON block is greedy-but-bounded by ``\n\`\`\``` at the end.
    _V3_BODY_REGEX = re.compile(
        r"^function<｜tool▁sep｜>(?P<name>.*?)\n```json\n(?P<args>.*?)\n```\s*$",
        re.DOTALL,
    )
    # V3 body with a tolerant trailing fence (some checkpoints omit the
    # newline before the closing fence, e.g. ``...}```<call_end>``).
    _V3_BODY_REGEX_TOLERANT = re.compile(
        r"^function<｜tool▁sep｜>(?P<name>.*?)\n```json\n(?P<args>.*?)```\s*$",
        re.DOTALL,
    )

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        # IDs we have already emitted on the streaming V3 short-circuit
        # path, keyed by ABSOLUTE block index in the cumulative parse.
        # We re-parse the cumulative text every time a block closes (so
        # any newly-closed block is included), and we MUST NOT
        # regenerate IDs for blocks we have already announced —
        # downstream clients track tool calls by ID and seeing the same
        # block under a new ID on every subsequent delta would corrupt
        # their state (codex r1 BLOCKING).
        #
        # Must be keyed by absolute index, NOT a positional list:
        # ``_streamed_v3_ids`` records ONLY V3 emissions; a V3.1 block
        # in front of a V3 block makes the absolute index of the V3
        # block 1 while ``len(_streamed_v3_ids)`` is 0 — a positional
        # ``idx < len(...)`` would then re-emit the V3 block on every
        # subsequent delta (codex r4 BLOCKING).
        self._streamed_v3_ids: dict[int, str] = {}

        # V3.1 streaming regex (unchanged). V3-shaped streaming
        # arrives through the same regex because both shapes share the
        # ``NAME<sep>...`` skeleton; the V3 "function" type tag plus
        # JSON fence become part of the staged ``arguments`` buffer
        # mid-stream and are only resolved into the correct ``name`` /
        # ``arguments`` split once the closing ``<call_end>`` lands and
        # the non-streaming ``extract_tool_calls`` path runs.
        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<function_name>.*)<｜tool▁sep｜>(?P<function_arguments>.*)",
            re.DOTALL,
        )
        self.stream_tool_call_name_regex = re.compile(
            r"(?P<function_name>.*)<｜tool▁sep｜>"
        )

        # Token IDs for streaming (graceful fallback if absent)
        self.tool_calls_start_token_id = self.vocab.get(self.TOOL_CALLS_START)
        self.tool_calls_end_token_id = self.vocab.get(self.TOOL_CALLS_END)
        self.tool_call_start_token_id = self.vocab.get(self.TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(self.TOOL_CALL_END)

    # -----------------------------------------------------------------
    # Block-wise scanner — the structural fix for D-DSV31
    # -----------------------------------------------------------------
    @classmethod
    def _iter_block_bodies(cls, model_output: str) -> list[str]:
        """Yield the body text between each ``<call_begin>`` /
        ``<call_end>`` pair, in order.

        Uses a forward-scanning state machine (find next ``<call_begin>``,
        find the *matching* ``<call_end>``) rather than a single greedy
        regex. This is what makes parallel calls parse as N entries
        instead of one over-greedy match, and what makes a truncated
        trailing block ignored rather than swallowing the rest of the
        payload.
        """
        bodies: list[str] = []
        pos = 0
        start_len = len(cls.TOOL_CALL_START)
        end_len = len(cls.TOOL_CALL_END)
        while True:
            start = model_output.find(cls.TOOL_CALL_START, pos)
            if start == -1:
                break
            body_start = start + start_len
            end = model_output.find(cls.TOOL_CALL_END, body_start)
            if end == -1:
                # Truncated block — drop and stop (caller emits the
                # leading text as content). Matches the "malformed →
                # graceful no-op" contract.
                break
            bodies.append(model_output[body_start:end])
            pos = end + end_len
        return bodies

    @classmethod
    def _parse_block_body(cls, body: str) -> tuple[str, str] | None:
        """Parse one ``<call_begin>...<call_end>`` body into ``(name, args)``.

        Tries V3 (fenced JSON) first when the body starts with the
        ``function<sep>`` type-tag prefix; otherwise the V3.1 (plain
        ``NAME<sep>ARGS``) shape. Returns ``None`` if the body is not
        parseable in the claimed shape.

        Critically (codex r2 BLOCKING): a body that anchors as V3 must
        NOT silently fall through to the V3.1 split — that path would
        emit ``name="function"`` with the real tool name embedded in
        ``arguments``, recreating the exact production failure mode
        we're patching. If the V3 anchor matches but neither V3 regex
        does, the body is malformed; we attempt one bounded recovery
        (look for the inner JSON between ``\\`\\`\\`json`` and any
        terminator) and return ``None`` if that fails so the caller
        drops the block.
        """
        body = body.strip("\n")
        if body.startswith(f"{_V3_TYPE_TAG}{cls.TOOL_SEP}"):
            m = cls._V3_BODY_REGEX.match(body) or cls._V3_BODY_REGEX_TOLERANT.match(
                body
            )
            if m is not None:
                return m.group("name").strip(), m.group("args").strip()
            # V3 anchor but malformed fence. Try one bounded recovery:
            # the body shape is ``function<sep>NAME\n``\`\`\`json\n…``
            # — split off the type-tag-plus-sep and look for the name
            # terminator (``\n``\`\`\`json``). If both halves
            # materialise we recover; otherwise the block is unparseable
            # and we return ``None`` to drop it (NOT fall through to
            # the V3.1 ``name=function`` mis-split).
            v3_prefix = f"{_V3_TYPE_TAG}{cls.TOOL_SEP}"
            tail = body[len(v3_prefix) :]
            fence_idx = tail.find("\n```json")
            if fence_idx <= 0:
                return None
            name = tail[:fence_idx].strip()
            args_part = tail[fence_idx + len("\n```json") :].lstrip("\n")
            # Strip any trailing ``\`\`\`` fence the model produced.
            if args_part.endswith("```"):
                args_part = args_part[: -len("```")].rstrip("\n")
            args = args_part.strip()
            if not name:
                return None
            return name, args

        sep_idx = body.find(cls.TOOL_SEP)
        if sep_idx == -1:
            return None
        name = body[:sep_idx].strip()
        args = body[sep_idx + len(cls.TOOL_SEP) :].strip()
        if not name:
            return None
        return name, args

    # -----------------------------------------------------------------
    # Streaming helper — closed-V3 block recovery (codex r3 BLOCKING-2)
    # -----------------------------------------------------------------
    def _maybe_emit_closed_v3_blocks(
        self, current_text: str, request: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Scan the cumulative streamed text for closed V3-shaped blocks
        that have not yet been announced via ``_streamed_v3_ids``. If
        any exist, emit them now as a one-shot ``tool_calls`` delta and
        return that delta. Otherwise return ``None`` and let the legacy
        streaming path run.

        Why this lives at the top of streaming dispatch: a single
        delta may both close a V3 block AND open a non-V3-yet block,
        in which case the open-block sniffer would miss the closed V3
        (it only inspects the latest body). The cumulative scan catches
        it regardless of what the next block is doing.
        """
        # ``extract_tool_calls`` runs the same block-wise scanner used
        # for non-streaming parses, so it sees all closed blocks. We
        # use index correlation: ``_streamed_v3_ids`` records the IDs
        # of every block we've already emitted (in order), so any
        # parsed call past ``len(_streamed_v3_ids)`` is unannounced.
        result = self.extract_tool_calls(current_text, request)
        if not result.tools_called:
            return None

        # Find the V3-shaped block indices in the cumulative text so we
        # only short-circuit for V3 blocks; V3.1 blocks continue to use
        # the legacy delta machine for token-by-token emission.
        is_v3_index = self._classify_block_shapes(current_text)

        # Walk the parsed calls; for any index whose block is V3-shaped
        # AND hasn't been emitted yet, queue it for emission. Emission
        # state is tracked by ABSOLUTE block index keyed in
        # ``_streamed_v3_ids`` — NOT by positional length — because
        # ``_streamed_v3_ids`` records only V3 emissions and the
        # cumulative parse interleaves V3 and V3.1 blocks (codex r4).
        to_emit: list[tuple[int, dict[str, Any]]] = []
        for idx, tc in enumerate(result.tool_calls):
            if idx >= len(is_v3_index) or not is_v3_index[idx]:
                continue
            if idx in self._streamed_v3_ids:
                continue
            to_emit.append((idx, tc))

        if not to_emit:
            return None

        emitted: list[dict[str, Any]] = []
        for abs_idx, tc in to_emit:
            tc_id = tc["id"]
            self._streamed_v3_ids[abs_idx] = tc_id
            # Keep ``current_tool_id`` advanced past this index so the
            # legacy machine doesn't try to re-emit the same block as
            # V3.1 on a subsequent delta. We're a bit defensive: if the
            # legacy path already advanced past this index (e.g. a
            # V3.1 block sat in front of a V3 block), we don't rewind.
            if abs_idx > self.current_tool_id:
                self.current_tool_id = abs_idx
            emitted.append(
                {
                    "index": abs_idx,
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
            )
        return {"tool_calls": emitted}

    @classmethod
    def _classify_block_shapes(cls, model_output: str) -> list[bool]:
        """Return ``[is_v3_per_block]`` for every CLOSED block, in
        order. Open trailing blocks are not included — they have no
        determined shape until ``<call_end>`` arrives.
        """
        out: list[bool] = []
        v3_marker = f"{_V3_TYPE_TAG}{cls.TOOL_SEP}"
        for body in cls._iter_block_bodies(model_output):
            out.append(body.lstrip("\n").startswith(v3_marker))
        return out

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.TOOL_CALLS_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls: list[dict[str, Any]] = []
            for body in self._iter_block_bodies(model_output):
                parsed = self._parse_block_body(body)
                if parsed is None:
                    # Malformed block — skip, don't poison the rest of
                    # the envelope. Matches the "malformed → graceful"
                    # contract exercised by the test suite.
                    continue
                name, args = parsed
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": name,
                        # Args body passes through verbatim. The
                        # upstream vLLM contract is bytes-equal
                        # passthrough — downstream JSON canonicalisation
                        # is the caller's job (and changing the bytes
                        # here breaks ``tests/test_upstream_regression``).
                        "arguments": args,
                    }
                )

            if tool_calls:
                # Tool calls fired — content is anything strictly before
                # the outer begin marker (V3.1 semantics).
                content = model_output[: model_output.find(self.TOOL_CALLS_START)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            # Outer envelope present but nothing parsed (truncated /
            # malformed). Pass the full model output through as content
            # so callers see the raw text rather than silently dropping
            # everything after ``<call_begin>``. Mirrors the V3 parser's
            # "no match" behaviour and the test suite's malformed
            # graceful-passthrough contract.
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def has_pending_tool_call(self, text: str) -> bool:
        return (
            self.TOOL_CALLS_START in text
            or self.TOOL_CALL_START in text
            or self.has_text_format_tool_call(text)
        )

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
        if not previous_text:
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []
            self.current_tool_id = -1
            self.prev_tool_call_arr = []
            self._streamed_v3_ids = {}

        current_token_ids = current_token_ids or []
        previous_token_ids = previous_token_ids or []
        delta_token_ids = delta_token_ids or []

        # Use token IDs if available, fall back to string matching
        has_tool_start = (
            self.tool_calls_start_token_id is not None
            and self.tool_calls_start_token_id in current_token_ids
        ) or self.TOOL_CALLS_START in current_text

        if not has_tool_start:
            return {"content": delta_text}

        # ----------------------------------------------------------------
        # V3-shape streaming gate (D-DSV31 follow-on).
        #
        # The legacy delta machine captures ``name<sep>args`` from the
        # currently-open block via a single regex — that would emit
        # ``name="function"`` mid-stream for V3 bodies because the
        # literal ``function`` type tag *is* what arrives first.
        #
        # The gate has two responsibilities, applied in order:
        #
        #   A. Detect V3 in any CLOSED block via the cumulative
        #      non-streaming parser. If a V3 block has closed but
        #      hasn't yet been announced (tracked by
        #      ``self._streamed_v3_ids``), emit it now. This handles
        #      the "delta closes a V3 block and also opens an empty
        #      next block" case (codex r3 BLOCKING-2): the V3 detection
        #      lives in the closed-block scan, not in inspection of
        #      the latest possibly-open body.
        #
        #   B. For the currently-OPEN block: buffer silently only when
        #      V3 is *confirmed* by the body starting with
        #      ``function<｜tool▁sep｜>`` (the V3 type tag + separator).
        #      We do NOT buffer on a mere prefix-could-become-V3 because
        #      a V3.1 tool whose name happens to start with ``f``,
        #      ``fun``, ``function_lookup`` etc. would otherwise be
        #      swallowed (codex r3 BLOCKING-1) and the legacy state
        #      machine never gets to advance ``current_tool_id``.
        #
        # The closed-V3-emit path runs first because it must fire even
        # when the *latest* (possibly empty) block isn't V3-confirmed
        # but a previously-closed block was V3.
        # ----------------------------------------------------------------
        cur_start_count = current_text.count(self.TOOL_CALL_START)
        cur_end_count = current_text.count(self.TOOL_CALL_END)
        v3_marker = f"{_V3_TYPE_TAG}{self.TOOL_SEP}"

        # --- A. Closed V3 block recovery ---
        # Scan the closed blocks for V3-shaped bodies; emit any whose
        # absolute index hasn't been streamed yet.
        closed_v3_emit = self._maybe_emit_closed_v3_blocks(current_text, request)
        if closed_v3_emit is not None:
            return closed_v3_emit

        # --- B. Open-block confirmed-V3 buffer ---
        # Latest (possibly open) block body.
        is_block_open = cur_end_count < cur_start_count
        if is_block_open:
            tail = current_text.split(self.TOOL_CALL_START)[-1]
            open_body = tail.split(self.TOOL_CALL_END)[0].lstrip("\n")
            if open_body.startswith(v3_marker):
                # V3 confirmed in open block — silently buffer. The
                # closed-V3 path will emit it once <call_end> lands in
                # a later delta.
                return None

        delta_text = delta_text.replace(self.TOOL_CALLS_START, "").replace(
            self.TOOL_CALLS_END, ""
        )

        try:
            # Count tool call tokens (string-based fallback)
            prev_tool_start_count = previous_text.count(self.TOOL_CALL_START)
            prev_tool_end_count = previous_text.count(self.TOOL_CALL_END)
            cur_tool_start_count = current_text.count(self.TOOL_CALL_START)
            cur_tool_end_count = current_text.count(self.TOOL_CALL_END)

            tool_call_portion = None

            # Generating text (no open tool calls)
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.TOOL_CALL_END not in delta_text
            ):
                return {"content": delta_text}

            if self.TOOL_CALL_END in delta_text:
                full_text = current_text
                tool_call_portion = (
                    full_text.split(self.TOOL_CALL_START)[-1]
                    .split(self.TOOL_CALL_END)[0]
                    .rstrip()
                )
                delta_text = delta_text.split(self.TOOL_CALL_END)[0].rstrip()

            # Starting new tool call
            if (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                if len(delta_text) > 1:
                    tool_call_portion = current_text.split(self.TOOL_CALL_START)[-1]
                else:
                    tool_call_portion = None

                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")

            # Updating existing tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                tool_call_portion = current_text.split(self.TOOL_CALL_START)[-1]

            # Closing tool call
            elif (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count >= prev_tool_end_count
            ):
                if not self.prev_tool_call_arr or self.current_tool_id >= len(
                    self.prev_tool_call_arr
                ):
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff and '"}' in delta_text:
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_id,
                                "function": {"arguments": diff},
                            }
                        ]
                    }
                return None
            else:
                text = delta_text.replace(self.TOOL_CALL_START, "").replace(
                    self.TOOL_CALL_END, ""
                )
                return {"content": text} if text else None

            # Parse tool call portion
            current_tool_call: dict = {}
            if tool_call_portion:
                m = self.stream_tool_call_portion_regex.match(tool_call_portion)
                if m:
                    current_tool_call["name"] = m.group("function_name")
                    current_tool_call["arguments"] = m.group("function_arguments")
                else:
                    m2 = self.stream_tool_call_name_regex.match(tool_call_portion)
                    if m2:
                        current_tool_call["name"] = m2.group("function_name")
                        current_tool_call["arguments"] = ""
                    else:
                        return None

            # Send tool name
            if not self.current_tool_name_sent:
                if not current_tool_call:
                    return None
                func_name = current_tool_call.get("name")
                if func_name:
                    self.current_tool_name_sent = True
                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_id,
                                "id": _generate_tool_id(),
                                "type": "function",
                                "function": {"name": func_name, "arguments": ""},
                            }
                        ]
                    }
                return None

            if tool_call_portion is None:
                return None

            # Ensure prev_tool_call_arr has entry
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            cur_arguments = current_tool_call.get("arguments")

            delta = None
            if not cur_arguments and not prev_arguments:
                delta = None
            elif cur_arguments and not prev_arguments:
                delta = {
                    "tool_calls": [
                        {
                            "index": self.current_tool_id,
                            "function": {"arguments": cur_arguments},
                        }
                    ]
                }
                self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
            elif cur_arguments and prev_arguments:
                if len(cur_arguments) > len(
                    prev_arguments
                ) and cur_arguments.startswith(prev_arguments):
                    diff = cur_arguments[len(prev_arguments) :]
                    delta = {
                        "tool_calls": [
                            {
                                "index": self.current_tool_id,
                                "function": {"arguments": diff},
                            }
                        ]
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = cur_arguments

            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
