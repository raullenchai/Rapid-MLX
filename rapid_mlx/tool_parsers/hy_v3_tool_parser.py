# SPDX-License-Identifier: Apache-2.0
"""
Hy3 (Tencent Hunyuan 3) tool call parser for rapid-mlx.

Ported from vLLM's ``HYV3ToolParser`` (vllm/tool_parsers/hy_v3_tool_parser.py)
and SGLang's ``hunyuan_detector`` (python/sglang/srt/function_call/
hunyuan_detector.py::resolve_hunyuan_tokens). The wire format is:

    <tool_call:opensource>NAME<tool_sep:opensource>{"k1": "v1", ...}<end_of_tool_call:opensource>

with an ``<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V
</arg_value:opensource>`` XML-pair variant instead of / in addition to the
JSON body. The label suffix (``:opensource``) marks the checkpoint's
reasoning-mode variant; a future revision may drop it or swap it.

Design (why this file was rewritten from the bespoke full-text-regex design)
===========================================================================

The earlier rapid-mlx implementation re-parsed the entire accumulated text
on every streaming delta, used a suffix-alternation regex ``(?::[\\w-]+)?``
everywhere, and carried ~85 LOC of partial-opener straddle-guard code
(``_tool_call_open_straddle_suffix_len``, ``_is_strict_prefix_of_tool_call_
opener``, a ``_streamed_bytes`` content watermark). Seven codex rounds all
chased symptoms of that architecture. This rewrite ports vLLM/SGLang's
proven approach instead:

1. **Resolve the suffix ONCE at ``__init__``** by scanning
   ``tokenizer.get_vocab()`` for ``<tool_call(:LABEL)?>`` and pinning the
   real tag strings as FIXED strings (``self.tool_call_start_token``,
   ``self.tool_sep_token``, …). No regex alternation on the hot path.
   (SGLang ``resolve_hunyuan_tokens`` pattern.)

2. **Token-ID gate the streaming entry.** ``self.tool_call_start_token_id``
   is resolved from vocab; special tokens are ATOMIC on the tokenizer
   boundary so they cannot straddle SSE chunks. Once the start token id is
   absent from ``current_token_ids`` (and the fixed start string is absent
   from the accumulated text), the delta is pure content — pass it through.
   This single gate deletes the entire straddle-guard family.

3. **Buffer accumulates only INSIDE the tool-call span**, keyed on
   ``str.find`` of the pinned fixed strings — never a full-history regex
   re-scan.

4. **Two-phase state machine**: ``SEEKING_NAME`` (find ``<tool_sep>`` in the
   buffer → emit the function-name header) → ``STREAMING_ARGS`` (emit the
   COMPLETE args document as a single valid-JSON delta once
   ``<end_of_tool_call>`` arrives). Arguments are never emitted as a partial
   fragment, so every ``function.arguments`` delta is a valid-JSON piece
   whose concatenation is the final document (OpenAI streaming contract). A
   whole call arriving in one delta emits BOTH the header and the args in
   that delta. Multiple calls advance index-by-index: on each close the FSM
   resets to ``SEEKING_NAME`` so the next opener is a fresh indexed call.

5. **``<think>`` handling lives entirely in the separate reasoning parser**
   (``rapid_mlx/reasoning/hy3_parser.py::Hy3ReasoningParser``, registered as
   ``--reasoning-parser hy_v3``). This tool parser has ZERO ``<think>`` code
   — the two parsers see disjoint token streams, exactly as vLLM's do.

6. **Watermark on args**: ``self.streamed_args_for_tool`` records the args
   already emitted per tool call (vLLM base pattern) so a call's args are
   emitted exactly once.

7. **Malformed-close salvage** (``<tool_call>NAME</arg_value>`` — 4-bit
   numerical noise empirically observed on ``pipenetwork/Hy3-REAP50-MLX-4bit``
   and ``Hy3-REAP75-MLX-4bit``, 10/10 BFCL simple_python prompts) is
   rapid-mlx's unique value-add. It runs ONLY on the non-streaming
   ``extract_tool_calls`` path — 4-bit noise is rare and streaming clients
   re-parse on completion, so streaming never runs salvage.

**Wire format label.** ``hy3_native`` — declared in ``EXPECTED_WIRE_FORMATS``.
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from transformers import PreTrainedTokenizerBase

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

# Default label baked into ``pipenetwork/Hy3-*-MLX-4bit`` and the
# ``chat_template.jinja`` sentinel scheme. Used only when no tokenizer is
# available to resolve the real suffix from vocab.
_DEFAULT_LABEL = "opensource"

# Bare Hy3 wire token names whose real (possibly suffixed) vocab string is
# resolved once at ``__init__``. Mirrors SGLang ``_HUNYUAN_TOKEN_NAMES``.
_TOKEN_NAMES = frozenset(
    {"tool_call", "tool_sep", "arg_key", "arg_value", "end_of_tool_call"}
)

# Matches a Hy3 wire token in the vocab, capturing the bare name. ``resolve``
# uses this to pin the real (possibly suffixed) strings.
_VOCAB_TOKEN_RE = re.compile(r"^<(?P<name>[a-z_]+)(?::[\w-]+)?>$")


def generate_tool_id() -> str:
    """Generate a unique tool call ID (OpenAI-compatible short form)."""
    return f"call_{uuid.uuid4().hex[:8]}"


def _resolve_suffix(tokenizer: PreTrainedTokenizerBase | None) -> str:
    """Resolve the wire label suffix (e.g. ``:opensource``) ONCE from vocab.

    Collects EVERY suffix under which the Hy3 wire tokens exist, then selects
    ONLY among suffixes carrying a COMPLETE token set (``tool_call``,
    ``tool_sep``, ``end_of_tool_call`` all present) — a suffix that has only
    ``<tool_call>`` but not its matching ``<tool_sep>`` / ``<end_of_tool_call>``
    would make every downstream fixed-string ``find`` look for a non-existent
    separator/close and silently drop valid calls (codex R4 BLOCKING). Among
    complete candidates the choice is deterministic (codex R2 NIT — dict
    iteration order must not decide the wire shape): prefer the labelled
    ``:opensource`` default, then any other label sorted, then the bare form.

    Falls back to ``:opensource`` when no tokenizer is available OR no complete
    candidate exists (the shape every current ``pipenetwork/Hy3-*-MLX-4bit``
    checkpoint emits). SGLang ``resolve_hunyuan_tokens`` pattern.
    """
    default = f":{_DEFAULT_LABEL}"
    if tokenizer is None:
        return default
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = None
    if not isinstance(vocab, dict):
        return default

    # suffix -> set of bare token names present under it.
    by_suffix: dict[str, set[str]] = {}
    for tok in vocab:
        if not isinstance(tok, str):
            continue
        m = _VOCAB_TOKEN_RE.match(tok)
        if m is None:
            continue
        name = m.group("name")
        if name not in _TOKEN_NAMES:
            continue
        # ``tok`` is ``<name>`` or ``<name:LABEL>`` — the suffix is the span
        # between the name and the closing ``>``.
        suffix = tok[1 + len(name) : -1]  # ``""`` or ``:LABEL``
        by_suffix.setdefault(suffix, set()).add(name)

    # ONLY consider suffixes whose parsing-critical trio is fully present —
    # an incomplete suffix would break every downstream ``find``.
    #
    # The trio is ``tool_call`` / ``tool_sep`` / ``end_of_tool_call`` — the
    # tokens EVERY call carries. ``arg_key`` / ``arg_value`` are DELIBERATELY
    # excluded (codex R9 NIT): the XML-pair argument form is an OPTIONAL variant
    # (a checkpoint may emit only JSON bodies and never carry the arg tokens),
    # so requiring them would wrongly reject a valid JSON-only suffix. When a
    # checkpoint does emit XML-pair args it carries the arg tokens under the
    # SAME suffix as the trio (they are minted together), so the JSON-only
    # completeness check still resolves the right suffix for XML-pair parsing.
    required = {"tool_call", "tool_sep", "end_of_tool_call"}
    candidates = [s for s in by_suffix if required.issubset(by_suffix[s])]
    if not candidates:
        return default

    def _rank(suffix: str) -> tuple:
        # Lower tuple sorts first: :opensource default, then other labels
        # alphabetically, then the bare form.
        if suffix == default:
            tier = 1
        elif suffix == "":
            tier = 3
        else:
            tier = 2
        return (tier, suffix)

    return min(candidates, key=_rank)


def _deserialize_arg_value(value: str) -> Any:
    """Coerce a raw ``<arg_value>`` payload to a JSON-native Python type.

    Tries ``json.loads`` first so ``true`` / ``42`` / ``[1,2]`` / ``null``
    round-trip cleanly; falls back to the trimmed string for free-form text
    so we do not silently drop the argument.
    """
    stripped = value.strip()
    if not stripped:
        return stripped
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped


@ToolParserManager.register_module(["hy_v3", "hy3"])
class HyV3ToolParser(ToolParser):
    """
    Tool call parser for Tencent Hunyuan 3 (``pipenetwork/Hy3-*-MLX-4bit``).

    Format:
        <tool_call:opensource>NAME<tool_sep:opensource>{"k":"v",...}
        <end_of_tool_call:opensource>

    or the XML-pair argument variant:
        <tool_call:opensource>NAME<tool_sep:opensource>
        <arg_key:opensource>K</arg_key:opensource>
        <arg_value:opensource>V</arg_value:opensource>
        <end_of_tool_call:opensource>

    The label suffix (``:opensource``) is resolved once from the tokenizer
    vocab at ``__init__``; all matching downstream uses the pinned fixed
    strings. The 4-bit malformed close (``<tool_call>NAME</arg_value>``) is
    salvaged on the non-streaming path.

    Used when ``--enable-auto-tool-choice --tool-call-parser hy_v3`` are set,
    or auto-wired for the ``hy3-*`` aliases via ``aliases.json`` /
    ``model_auto_config``.
    """

    # The Hy3 chat template renders assistant ``tool_calls`` back into the
    # same ``<tool_call:opensource>…<end_of_tool_call:opensource>`` markup the
    # parser reads. Feed previous-turn tool calls in native format rather
    # than converting them to synthetic text.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("hy3_native",)

    def __init__(self, tokenizer: PreTrainedTokenizerBase | None = None):
        super().__init__(tokenizer)

        # --- Resolve the suffix ONCE and pin FIXED tag strings (step 1) ---
        suffix = _resolve_suffix(tokenizer)  # ``":opensource"`` or ``""``
        self.suffix = suffix
        self.tool_call_start_token = f"<tool_call{suffix}>"
        self.tool_sep_token = f"<tool_sep{suffix}>"
        self.tool_call_end_token = f"<end_of_tool_call{suffix}>"
        self.arg_key_start_token = f"<arg_key{suffix}>"
        self.arg_key_end_token = f"</arg_key{suffix}>"
        self.arg_value_start_token = f"<arg_value{suffix}>"
        self.arg_value_end_token = f"</arg_value{suffix}>"

        # Non-streaming regex built from the FIXED strings (no alternation).
        esc = re.escape
        # Malformed-close salvage, matched on the post-opener BODY segment:
        # ``NAME</arg_value>`` with no ``<end_of_tool_call>`` — body captured
        # up to the FIRST ``</arg_value>``, explicitly forbidding an interior
        # ``<end_of_tool_call>`` so a well-formed block never falls here. Used
        # by ``_next_block`` only after the JSON-aware canonical close missed.
        self._tool_call_malformed_re_body = re.compile(
            r"(?P<body>(?:(?!"
            + esc(self.tool_call_end_token)
            + r").)*?)"
            + esc(self.arg_value_end_token),
            re.DOTALL,
        )
        self._arg_pair_re = re.compile(
            esc(self.arg_key_start_token)
            + r"\s*(?P<key>.*?)\s*"
            + esc(self.arg_key_end_token)
            + r"\s*"
            + esc(self.arg_value_start_token)
            + r"(?P<val>.*?)"
            + esc(self.arg_value_end_token),
            re.DOTALL,
        )
        self._json_decoder = json.JSONDecoder()

        # --- Token-ID gate (step 2). Special tokens are atomic on the ---
        # tokenizer boundary, so the start id (when present) cannot straddle
        # an SSE chunk. ``None`` when the tokenizer does not expose the token
        # as a single id; the streaming path then falls back to the fixed
        # string containment check, which is still atomic per-delta because
        # the whole opener arrives in one delta once the model emits it.
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        self._reset_streaming_state()

    # ------------------------------------------------------------------
    # Streaming state
    # ------------------------------------------------------------------
    def _reset_streaming_state(self) -> None:
        # Phase: name not sent yet = SEEKING_NAME, else STREAMING_ARGS.
        self.current_tool_id: int = -1
        self._name_sent: bool = False
        self._current_tool_ref: str | None = None
        # JSON already emitted for the current tool's arguments (watermark on
        # args, NOT content). vLLM base ``streamed_args_for_tool`` pattern.
        self.streamed_args_for_tool: list[str] = []
        self.prev_tool_call_arr: list[dict] = []
        # Indices whose name was off the request allowlist: their header was
        # never emitted, so their args must not be emitted either (but the FSM
        # still advances on close so the next opener is a fresh call).
        self._suppressed_tools: set[int] = set()
        # Client-visible tool-call index decoupled from the PHYSICAL opener
        # index (``current_tool_id`` / index into ``_opener_positions``). The
        # physical index advances for EVERY opener span the FSM consumes —
        # including garbled/sep-less residue openers that are skipped for
        # EMISSION and suppressed off-allowlist calls that emit nothing. The
        # client (OpenAI SDK) accumulates tool calls into an array keyed by
        # the emitted ``index``, so a first REAL call emitted at physical index
        # 1 (after a skipped residue at 0) would leave a null hole at 0 and
        # corrupt the reconstructed array. ``_client_index_of`` maps a physical
        # idx → the client-visible index it was ASSIGNED when its header was
        # actually emitted; ``_next_client_index`` is the monotonic counter
        # that only advances for calls the client actually sees. Physical
        # bookkeeping (``streamed_args_for_tool`` / ``prev_tool_call_arr`` /
        # positions) stays keyed by the physical idx; only the EMITTED
        # ``{"index": ...}`` uses the client-visible value.
        self._client_index_of: dict[int, int] = {}
        self._next_client_index: int = 0
        # High-water mark of content chars already emitted (against the raw
        # ``current_text``). Lets the tool-call entry flush any content that
        # preceded the FIRST opener when the opener and its leading content
        # arrive in the SAME delta (codex R3 BLOCKING: pre-opener content was
        # silently dropped once any opener was present).
        self._content_emitted: int = 0

    def reset(self) -> None:
        super().reset()
        self._reset_streaming_state()

    def _get_tool_names(self, request: dict[str, Any] | None) -> set[str]:
        """Extract valid tool names from the request payload."""
        if not request or "tools" not in request:
            return set()
        return {
            t.get("function", {}).get("name", "")
            for t in request.get("tools", [])
            if isinstance(t, dict)
        }

    def _opener_positions(self, text: str) -> list[int]:
        """Byte offsets of every GENUINE ``<tool_call>`` call-start in ``text``.

        NOT a plain substring scan (codex R6 BLOCKING): a JSON string argument
        value may legitimately contain the literal ``<tool_call…>`` opener text,
        and a raw scan would split that interior substring into a phantom call,
        corrupting the argument stream. Instead we walk call spans forward:

          * from the cursor, find the next opener → that is a genuine call start;
          * find its close JSON-aware (``_find_call_close_in_body`` runs
            ``raw_decode`` over a ``{``-body, so any opener/end-token literal
            inside a string value is consumed, not treated as a boundary);
          * advance the cursor PAST that close and repeat.

        The last call may be still streaming (no close yet). Its opener is
        included (the FSM is driving it), and — crucially — we STOP there: any
        opener-substring inside its in-progress body is opaque until the body's
        JSON completes and a real close is seen, so it can never become a
        phantom boundary.
        """
        positions: list[int] = []
        tok = self.tool_call_start_token
        cursor = 0
        n = len(text)
        while cursor <= n:
            opener = text.find(tok, cursor)
            if opener == -1:
                break
            positions.append(opener)
            body_start = opener + len(tok)
            rest = text[body_start:]

            # A ``<tool_call>`` that appears BEFORE this call's own
            # ``<tool_sep>`` is a genuine separate opener — this opener is a
            # garbled/sep-less residue and must NOT reach across the later
            # opener to steal its separator/close (codex R11 BLOCKING #2). A
            # ``<tool_call>`` that appears AFTER the separator lives inside the
            # args region: it may be a literal inside a JSON string value and
            # is resolved JSON-aware by ``_find_call_close_in_body`` (codex R6),
            # so it is NOT a boundary here. Bounding only at a pre-separator
            # opener keeps R6 (interior literal opaque) and R11 (garbled residue
            # split) both correct.
            own_sep = rest.find(self.tool_sep_token)
            next_opener_rel = rest.find(tok)
            garbled_boundary = next_opener_rel != -1 and (
                own_sep == -1 or next_opener_rel < own_sep
            )
            body_end_rel = next_opener_rel if garbled_boundary else len(rest)
            close_rel = self._find_call_close_in_body(rest[:body_end_rel])
            if close_rel == -1:
                if garbled_boundary:
                    # No close within THIS opener's own body and a genuine later
                    # opener precedes any separator → garbled residue. Keep it
                    # recorded (the streaming FSM / non-stream salvage treats a
                    # name-only span as residual / bare-name) and resume
                    # scanning at the later opener.
                    cursor = body_start + next_opener_rel
                    continue
                # Genuine in-progress (or truncated) final call; its interior is
                # opaque until it closes.
                break
            cursor = body_start + close_rel + len(self.tool_call_end_token)
        return positions

    # ------------------------------------------------------------------
    # Body parsing (shared by non-streaming extraction)
    # ------------------------------------------------------------------
    def _parse_body(self, body: str) -> tuple[str, dict[str, Any]]:
        """Parse a ``NAME<tool_sep>ARGS`` body into ``(name, arguments)``.

        Two on-wire arg shapes coexist:

        1. **JSON body** — ``NAME<tool_sep>{"k": "v", …}`` — the chat
           template's default emission.
        2. **XML-pair body** — ``NAME<tool_sep><arg_key>k</arg_key>
           <arg_value>v</arg_value>…`` — each pair transcribed separately.

        A sep-less body (``NAME`` alone, or ``NAME`` followed straight by an
        ``<arg_key>``/``<arg_value>`` opener) is handled too: the residue is
        the name and the args are recovered from the pairs if present.
        """
        sep_idx = body.find(self.tool_sep_token)
        if sep_idx == -1:
            # No separator. Either sep-less XML pairs, or just a bare name.
            ak = body.find(self.arg_key_start_token)
            av = body.find(self.arg_value_start_token)
            candidates = [c for c in (ak, av) if c != -1]
            if candidates:
                arg_open = min(candidates)
                name = body[:arg_open].strip()
                tail = body[arg_open:]
            else:
                raw = body.strip()
                # Strip any close-tag residue defensively (malformed close).
                raw = raw.replace(self.arg_value_end_token, "")
                raw = raw.replace(self.arg_key_end_token, "")
                return raw.strip(), {}
        else:
            name = body[:sep_idx].strip()
            tail = body[sep_idx + len(self.tool_sep_token) :]

        return name, self._parse_args_tail(tail)

    def _parse_args_tail(self, tail: str) -> dict[str, Any]:
        """Parse the post-``<tool_sep>`` tail into an arguments dict.

        Probes the JSON-body shape first (``raw_decode`` consumes only a
        well-formed JSON prefix so a value string containing the literal
        ``</arg_value>`` round-trips unchanged), then falls back to walking
        ``<arg_key>K</arg_key><arg_value>V</arg_value>`` pairs.
        """
        stripped = tail.strip()
        if stripped.startswith("{"):
            try:
                parsed, _end = self._json_decoder.raw_decode(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        args: dict[str, Any] = {}
        for m in self._arg_pair_re.finditer(tail):
            key = m.group("key").strip()
            if not key:
                continue
            args[key] = _deserialize_arg_value(m.group("val"))
        return args

    # ------------------------------------------------------------------
    # Non-streaming extraction (with malformed-close salvage)
    # ------------------------------------------------------------------
    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract Hy3 tool calls from a complete model response.

        Malformed-close salvage (``<tool_call>NAME</arg_value>``) runs here —
        4-bit noise is rare and this path sees the whole body at once, so it
        can distinguish the malformed close from a legitimate interior
        ``</arg_value>`` in an XML-pair body. Streaming never runs salvage.
        """
        if self.tool_call_start_token not in model_output:
            # No native opener. A low-quant checkpoint may still degrade into
            # the shared ``[Calling tool="X" k="v"]`` text form, so consult
            # that fallback before returning plain content (codex BLOCKING:
            # the early return otherwise made this branch unreachable).
            return self._text_format_or_content(model_output, request)

        valid_names = self._get_tool_names(request)
        tool_calls: list[dict[str, Any]] = []
        residual_parts: list[str] = []
        cursor = 0
        length = len(model_output)
        while cursor < length:
            block = self._next_block(model_output, cursor)
            if block is None:
                break
            block_start, block_end, body = block
            residual_parts.append(model_output[cursor:block_start])
            name, args = self._parse_body(body)
            if not name:
                cursor = block_end
                continue
            if valid_names and name not in valid_names:
                # Preserve the rejected span in residual text so a request
                # with a ``tools`` allowlist + a hallucinated off-list name
                # surfaces the attempted call rather than a silent empty
                # ``tool_calls`` array that looks like a refusal.
                residual_parts.append(model_output[block_start:block_end])
                cursor = block_end
                continue
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
            cursor = block_end
        residual_parts.append(model_output[cursor:])
        residual_text = "".join(residual_parts).strip()

        if tool_calls:
            # Suppress content that precedes tool calls (exclusive-turn
            # policy — content=None when tools_called is True).
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )

        # No native call parsed — try the shared text-format degradation
        # fallback on the residual before giving up.
        return self._text_format_or_content(residual_text or model_output, request)

    def _text_format_or_content(
        self, text: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Consult the shared ``[Calling tool="X" k="v"]`` text-format
        fallback, else return ``text`` as plain content.

        Shared by the no-native-opener early return AND the end of
        ``extract_tool_calls`` so the low-quant text-degradation path is
        reachable in both (codex BLOCKING #2).

        Applies the request ``tools`` allowlist to the text-format calls too
        (codex R7 BLOCKING #1): otherwise a degraded ``[Calling tool="bogus"]``
        would bypass the ``request["tools"]`` filtering that native Hy3 calls
        enforce. Off-list names are dropped and their raw span preserved as
        content, mirroring the native path."""
        if self.has_text_format_tool_call(text):
            text_calls = self.extract_text_format_tool_calls(text)
            if text_calls:
                valid_names = self._get_tool_names(request)
                normalised = [
                    {
                        "id": tc.get("id", generate_tool_id()),
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                    for tc in text_calls
                    if not valid_names or tc["name"] in valid_names
                ]
                if normalised:
                    return ExtractedToolCallInformation(
                        tools_called=True, tool_calls=normalised, content=None
                    )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=text
        )

    def _next_block(self, text: str, cursor: int) -> tuple[int, int, str] | None:
        """Locate the next COMPLETE tool-call block at/after ``cursor``.

        Returns ``(block_start, block_end, body)`` where ``body`` is the span
        between the opener and its close. Prefers the canonical
        ``<end_of_tool_call>`` close found JSON-aware (so a literal end-token
        inside a JSON string value is not mistaken for the close); falls back
        to the malformed-close regex (``<tool_call>NAME</arg_value>``) when no
        canonical close exists.

        Returns ``None`` when no opener remains OR when an opener is present but
        has NEITHER a canonical NOR a malformed close (codex R4 BLOCKING: a
        truncated ``<tool_call>get_weather`` must NOT become a parsed call with
        ``{}`` args — the caller then keeps the raw tail as residual content,
        i.e. pending/plain text, not a fabricated empty call).
        """
        opener = text.find(self.tool_call_start_token, cursor)
        if opener == -1:
            return None
        body_start = opener + len(self.tool_call_start_token)

        # Find the canonical close JSON-aware. A later ``<tool_call>`` that
        # appears AFTER this call's own ``<tool_sep>`` may be a literal inside a
        # JSON string value (codex R6) — it must NOT bound the body, so
        # ``_find_call_close_in_body`` runs ``raw_decode`` over the JSON and
        # consumes it. But a later ``<tool_call>`` that appears BEFORE this
        # call's separator is a genuine separate opener: this opener is
        # garbled/sep-less residue and must NOT reach across it to steal the
        # later call's separator/close (codex R12 BLOCKING #2, mirroring
        # ``_opener_positions``). Bound the body at such a pre-separator opener.
        remainder = text[body_start:]
        own_sep = remainder.find(self.tool_sep_token)
        next_opener_rel = remainder.find(self.tool_call_start_token)
        garbled_boundary = next_opener_rel != -1 and (
            own_sep == -1 or next_opener_rel < own_sep
        )
        search_span = remainder[:next_opener_rel] if garbled_boundary else remainder
        close_rel = self._find_call_close_in_body(search_span)
        if close_rel != -1:
            body = remainder[:close_rel]
            block_end = body_start + close_rel + len(self.tool_call_end_token)
            return opener, block_end, body

        # No canonical close. For the malformed-close salvage (``</arg_value>``,
        # no JSON body to protect) bound the segment at the next opener so one
        # call's regex can't run into the following call's body.
        next_opener = text.find(self.tool_call_start_token, body_start)
        scan_end = next_opener if next_opener != -1 else len(text)
        segment = text[body_start:scan_end]
        m = self._tool_call_malformed_re_body.search(segment)
        if m is not None:
            body = m.group("body")
            # Restrict salvage to the DOCUMENTED ``NAME</arg_value>`` shape only
            # (codex R7 BLOCKING: a truncated XML-pair call that is simply
            # missing its ``<end_of_tool_call>`` — but carries ``<tool_sep>`` /
            # ``<arg_key>`` / ``<arg_value>`` — must NOT be promoted to a
            # completed executable call; it is incomplete output). The real
            # 4-bit noise is the bare ``NAME</arg_value>`` with none of those
            # structural tokens before the malformed close.
            if not any(
                t in body
                for t in (
                    self.tool_sep_token,
                    self.arg_key_start_token,
                    self.arg_value_start_token,
                )
            ):
                block_end = body_start + m.end()
                return opener, block_end, body

        # This opener has NO close of any kind. If it is garbled residue before
        # a genuine later opener (a pre-separator ``<tool_call>``), skip it and
        # resume the search at that later opener so the REAL call is still found
        # (codex R12 BLOCKING #2) — its raw span is preserved as residual by the
        # caller. Only when there is no genuine later opener is this a truncated /
        # streaming-incomplete final opener: signal end-of-parse so the caller
        # keeps the raw tail as content rather than fabricating a bogus call.
        if garbled_boundary:
            return self._next_block(text, body_start + next_opener_rel)
        return None

    def _find_call_close_in_body(self, segment: str) -> int:
        """Offset of the canonical close within a post-opener ``segment``.

        JSON-aware: locates ``<tool_sep>``, and when the args after it are a
        JSON body, searches for ``<end_of_tool_call>`` only AFTER the
        well-formed JSON prefix so an interior literal is ignored.

        Multi-call safety (codex R10 BLOCKING): ``segment`` may run past THIS
        call's close into the NEXT call (two calls in one delta). A ``<tool_sep>``
        that appears only AFTER this call's own ``<end_of_tool_call>`` belongs to
        the next call and must NOT be treated as this call's separator — else the
        JSON-aware search would jump to the next call's body and swallow its
        opener. So a sep found beyond the first end-token is ignored, and the
        first end-token is taken as the close. This matters for the SEP-LESS
        first call (XML-pair or bare-name body): its close is the first
        ``<end_of_tool_call>``; the next call's ``<tool_sep>`` lives past it.
        """
        first_end = segment.find(self.tool_call_end_token)
        sep = segment.find(self.tool_sep_token)
        # A sep belonging to a LATER call (after this call's own close) is not
        # this call's separator. Treat this call as sep-less so the first
        # end-token bounds it.
        if sep != -1 and first_end != -1 and sep > first_end:
            sep = -1
        if sep == -1:
            # No sep for THIS call. A sep-less XML-pair body can still carry an
            # ``<arg_value>`` whose free-form text contains the literal
            # ``<end_of_tool_call>`` string, so a plain find would truncate the
            # call and drop the argument (codex R15 BLOCKING). Use the same
            # tag-aware scan as the streaming path: accept only an end-token that
            # lands OUTSIDE any ``<arg_value>…</arg_value>`` span (there is no
            # JSON body of this call's own to protect).
            return self._end_token_outside_arg_value(segment)
        args_at = sep + len(self.tool_sep_token)
        rel = self._find_call_close(segment[args_at:])
        return -1 if rel == -1 else args_at + rel

    # ------------------------------------------------------------------
    # Streaming extraction — token-ID gate + 2-phase FSM
    # ------------------------------------------------------------------
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
        """Extract Hy3 tool calls from streaming model output.

        Token-ID gate (step 2): before any ``<tool_call>`` opener has entered
        the stream, the delta is pure content — pass it through. A special
        token is atomic on the tokenizer boundary so the opener cannot
        straddle an SSE chunk; no partial-opener buffering is needed.

        2-phase FSM (step 4):
          * SEEKING_NAME — buffer from the opener until ``<tool_sep>`` lands,
            then emit the function-name header (arguments empty).
          * STREAMING_ARGS — buffer the args body until ``<end_of_tool_call>``
            arrives, then emit the COMPLETE args document as a single
            valid-JSON delta. Args are never emitted as a partial fragment, so
            every ``function.arguments`` delta is a valid-JSON piece whose
            concatenation is the final document (OpenAI streaming contract). A
            call arriving whole in one delta emits BOTH header and args.

        Text-format degradation (``[Calling tool="X" k="v"]``) is handled ONLY
        on the NON-STREAMING path and at stream end (codex R8 BLOCKING: parity
        note). Deliberately NOT streamed incrementally: the degraded form is a
        rare low-quant artifact, has no native token boundaries to drive an
        incremental FSM, and the postprocessor's ``finalize()`` re-runs
        ``extract_tool_calls`` (allowlist-aware) over the full accumulated text
        whenever it contains the ``[Calling`` marker — recovering the structured
        call. So during streaming the ``[Calling …]`` bytes flow as ordinary
        content and are promoted to ``tool_calls`` exactly once, at finalize.
        """
        if not previous_text:
            self._reset_streaming_state()

        # ---- Token-ID gate: no tool call has begun → pure content. ----
        if not self._opener_seen(current_text, current_token_ids):
            # Emit content, but hold back a trailing partial-opener prefix so
            # a char-split ``<tool_call:opensou`` (the tokenizer/driver did
            # not deliver the opener as one atomic special token) does not
            # leak raw markup to the client. This is a single-string prefix
            # hold (the established ``_safe_content_prefix`` idiom), NOT the
            # deleted 85-LOC suffix-alternation straddle machinery — the held
            # bytes are released the moment they either complete the opener
            # (handled below) or falsify into ordinary content.
            return self._emit_safe_content(previous_text, current_text)

        # A tool call has opened somewhere in ``current_text``. Before any
        # tool-call delta, flush content that preceded the FIRST opener but was
        # never emitted — this happens when the opener and its leading content
        # arrive in the SAME delta, so ``_emit_safe_content`` above was skipped
        # (codex R3 BLOCKING). Emit that pending content AND the tool-call
        # deltas in the SAME return via the postprocessor's mixed-content
        # contract (``_detect_tool_calls`` preserves a ``content`` key alongside
        # ``tool_calls`` and the caller splits it into a leading content event
        # then the tool events). This drains everything this tick — no deferral
        # to a "later invocation that may never happen" on the FINAL delta
        # (codex R5 BLOCKING). After the pre-opener gap is drained, plain
        # content is suppressed (content and tool_calls are exclusive for the
        # rest of the turn).
        pending_content = self._flush_pre_opener_content(current_text)
        tool_result = self._stream_tool_call(current_text, request)
        if pending_content is not None:
            content_str = pending_content.get("content", "")
            if tool_result is not None:
                # Fold the leading content into the tool-call result so both
                # halves reach the wire in one tick.
                return {"content": content_str, **tool_result}
            return pending_content
        return tool_result

    def _safe_content_prefix(self, text: str) -> str:
        """Return the portion of ``text`` safe to emit as content now.

        Holds back the longest suffix of ``text`` that is a non-empty proper
        prefix of the tool-call opener (``<tool_call:opensource>``), so a
        char-split opener never leaks. Returns ``text`` unchanged when its
        tail cannot begin the opener.
        """
        opener = self.tool_call_start_token
        max_hold = 0
        for length in range(min(len(text), len(opener) - 1), 0, -1):
            if text.endswith(opener[:length]):
                max_hold = length
                break
        return text if max_hold == 0 else text[: len(text) - max_hold]

    def _emit_safe_content(
        self, previous_text: str, current_text: str
    ) -> dict[str, Any] | None:
        """Emit the new content diff with a partial-opener tail held back.

        When everything new is a held opener prefix, returns ``None`` so no
        content event fires this round; the bytes surface once the tail
        resolves (opener completes → tool-call turn; or falsifies → content).
        Advances ``_content_emitted`` so the tool-call entry knows exactly how
        much of the pre-opener content already went out.
        """
        safe_current = self._safe_content_prefix(current_text)
        safe_previous = self._safe_content_prefix(previous_text)
        if len(safe_current) <= len(safe_previous):
            return None
        self._content_emitted = len(safe_current)
        return {"content": safe_current[len(safe_previous) :]}

    def _flush_pre_opener_content(self, current_text: str) -> dict[str, Any] | None:
        """Emit content that preceded the FIRST opener but was never sent.

        When the opener and its leading content land in the SAME delta, the
        streaming entry skipped ``_emit_safe_content`` (an opener is present),
        so the leading content was never emitted. Flush the gap between the
        content high-water mark and the first opener exactly once, then advance
        the mark past the opener so it never re-fires (codex R3 BLOCKING).
        """
        first_opener = current_text.find(self.tool_call_start_token)
        if first_opener <= self._content_emitted:
            return None
        pending = current_text[self._content_emitted : first_opener]
        self._content_emitted = first_opener
        return {"content": pending} if pending else None

    def flush_held_content(self, full_text: str) -> str:
        """Release any prefix-held opener tail at stream end.

        A stream ending in ``abc<tool_ca`` (a partial opener that never
        completed) has held those bytes back; they are ordinary content and
        must be released so the last chars are not dropped.
        """
        # Only meaningful when no tool call actually opened — a real opener
        # commits the turn to tool_calls and the held tail is markup, not
        # content.
        if self.tool_call_start_token in full_text:
            return ""
        return full_text[len(self._safe_content_prefix(full_text)) :]

    def _opener_seen(
        self, current_text: str, current_token_ids: Sequence[int] | None
    ) -> bool:
        """True once the tool-call opener has entered the stream.

        Prefers the atomic token-ID signal (the opener is a single special
        token that cannot split across SSE chunks); falls back to the pinned
        fixed-string containment check when the tokenizer does not expose the
        opener as a single id.
        """
        if (
            self.tool_call_start_token_id is not None
            and current_token_ids is not None
            and self.tool_call_start_token_id in current_token_ids
        ):
            return True
        return self.tool_call_start_token in current_text

    def _stream_tool_call(
        self, current_text: str, request: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Drive the 2-phase FSM against the accumulated ``current_text``.

        Supports MULTIPLE tool calls in one turn: the parser advances index
        by index. The block for the call currently being streamed is the
        span from ITS opener up to the next opener (or end of text) — found
        by ``str.find`` on the pinned fixed strings, never a full re-parse.
        When the current call's ``<end_of_tool_call>`` lands, the args are
        finalized and the FSM transitions back to SEEKING_NAME so the next
        opener starts a fresh indexed call (codex BLOCKING: ``_name_sent``
        must reset per call).

        DRAINS all calls processable THIS tick: after a call closes, if the
        next opener is already present in ``current_text`` it is processed in
        the same invocation, so two complete calls arriving in one streaming
        delta both emit (codex R3 BLOCKING: one-opener-per-call dropped the
        second same-delta call). The loop stops at the first call that does
        not close (still streaming) or when no further opener has arrived.
        """
        deltas: list[dict[str, Any]] = []
        while True:
            step = self._process_one_call(current_text, request)
            if step is None:
                break
            step_deltas, closed = step
            deltas.extend(step_deltas)
            if not closed:
                # Current call still open — stop; the rest arrives next tick.
                break
            # Call closed and the FSM already advanced to the next index; if
            # its opener is present we loop and drain it too, else stop.

        return {"tool_calls": deltas} if deltas else None

    def _assign_client_index(self, physical_idx: int) -> int:
        """Return the client-visible index for a call being EMITTED now.

        Assigns the next monotonic client index to ``physical_idx`` the first
        time its header is emitted and caches it so a later args delta (a
        different tick) reuses the SAME client index. Only called on the
        emission path, so skipped residue openers and suppressed off-allowlist
        calls never consume a client index — the client sees a dense 0,1,2…
        sequence regardless of how many physical openers were skipped.
        """
        if physical_idx not in self._client_index_of:
            self._client_index_of[physical_idx] = self._next_client_index
            self._next_client_index += 1
        return self._client_index_of[physical_idx]

    def _process_one_call(
        self, current_text: str, request: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], bool] | None:
        """Process the single call at ``current_tool_id``.

        Returns ``(deltas, closed)`` — the deltas to emit for this call this
        tick and whether ``<end_of_tool_call>`` was seen (FSM advanced). Returns
        ``None`` when there is nothing to do (no opener for this index yet, or
        the name is not yet delimited).
        """
        opener_positions = self._opener_positions(current_text)
        if not opener_positions:
            return None

        # ``current_tool_id`` starts at -1; the first SEEKING_NAME emit bumps
        # it to 0. While streaming a call it stays put; on close it advances.
        idx = self.current_tool_id if self.current_tool_id >= 0 else 0
        if idx >= len(opener_positions):
            # Finished the last opener we know about and no new opener has
            # arrived yet — nothing to do.
            return None

        opener_pos = opener_positions[idx]
        block_end = (
            opener_positions[idx + 1]
            if idx + 1 < len(opener_positions)
            else len(current_text)
        )
        buffer = current_text[opener_pos + len(self.tool_call_start_token) : block_end]
        sep_idx = buffer.find(self.tool_sep_token)

        # ---------- Phase 1: SEEKING_NAME ----------
        # Emit the tool-call HEADER (name, empty arguments) the moment the
        # ``<tool_sep>`` delimits the name. Arguments are then emitted as a
        # SINGLE complete-JSON delta when the call closes (phase 2) — never a
        # partial/unterminated fragment. This keeps every ``function.arguments``
        # delta a valid-JSON piece whose concatenation is the final document
        # (codex R2: partial number / unterminated string prefixes violate the
        # OpenAI streaming contract). If the whole call arrived in one delta we
        # fall through so the same call emits BOTH the header and the args
        # (codex R2: single-delta case must not drop arguments).
        header: dict[str, Any] | None = None
        if not self._name_sent:
            if sep_idx == -1:
                # No ``<tool_sep>`` in this block. Two sub-cases:
                #   * the block is already CLOSED (``<end_of_tool_call>`` present)
                #     → a SEP-LESS call: XML-pair or bare-name body with no
                #       separator (codex R10 BLOCKING: the second call in a
                #       ``<xmlpairs><end><tool_call>bar<sep>{}<end>`` delta was
                #       swallowed because the sep-less first call stalled the
                #       FSM). Parse it whole via the shared non-streaming body
                #       parser and emit header + args in one tick, then advance.
                #   * the block is NOT closed → the name simply is not delimited
                #     yet; keep buffering, emit nothing.
                # Gate on a close-token OUTSIDE any ``<arg_value>…</arg_value>``
                # span, not a plain ``in`` check: a sep-less XML-pair value may
                # carry the literal ``<end_of_tool_call>`` as free-form text while
                # its ``<arg_value>`` is still streaming (no ``</arg_value>`` yet).
                # A plain ``in`` would fire ``_emit_sepless_closed_call`` on that
                # interior literal, parse the still-open body to ``{}`` and drop
                # the argument (codex R16, same hazard as R15's ``_find_call_close``
                # / ``_find_call_close_in_body`` fixes). Keep buffering until a
                # real close lands outside every value span.
                if self._end_token_outside_arg_value(buffer) != -1:
                    return self._emit_sepless_closed_call(buffer, idx, request)
                if idx + 1 < len(opener_positions):
                    # No sep and no close in this block, but a LATER opener
                    # exists (codex R11 BLOCKING): this opener is garbled
                    # residue (a truncated/aborted opener before a real call).
                    # Skip it — emit nothing, advance the FSM to the next index
                    # and let the drain loop process the real call. The raw
                    # residue span is preserved for the non-streaming salvage.
                    self.current_tool_id = idx + 1
                    self._name_sent = False
                    self._current_tool_ref = None
                    return [], True
                return None
            name = buffer[:sep_idx].strip()
            if not name:
                return None
            valid_names = self._get_tool_names(request)
            suppressed = bool(valid_names) and name not in valid_names
            self.current_tool_id = idx
            self._name_sent = True
            if idx >= len(self.streamed_args_for_tool):
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
            if suppressed:
                # Hallucinated off-list name — emit no header, but the FSM
                # still advances on close (below); the non-streaming path
                # preserves the raw span for diagnostics.
                self._current_tool_ref = None
                self._suppressed_tools.add(idx)
            else:
                self._current_tool_ref = generate_tool_id()
                header = {
                    "index": self._assign_client_index(idx),
                    "id": self._current_tool_ref,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }

        # ---------- Phase 2: STREAMING_ARGS (emit complete args on close) ---
        return self._emit_args_and_advance(buffer, sep_idx, header)

    def _emit_args_and_advance(
        self, buffer: str, sep_idx: int, header: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], bool]:
        """Emit the args document once the call closes and advance the FSM.

        Returns ``(deltas, closed)``. ``header`` is the phase-1 header dict for
        a same-delta call (name + empty args); it is folded into the same
        emission so a whole call in one delta yields BOTH name and args. When
        the call is a suppressed off-list call, ``header`` is ``None`` and
        nothing is emitted — but the FSM still advances on close so the next
        opener is a fresh call.

        Arguments are ONLY emitted when ``<end_of_tool_call>`` has arrived, as
        a single complete-JSON delta — never a partial fragment.
        """
        idx = self.current_tool_id
        deltas: list[dict[str, Any]] = []
        if header is not None:
            deltas.append(header)

        closed = False
        if sep_idx != -1:
            args_tail = buffer[sep_idx + len(self.tool_sep_token) :]
            end_idx = self._find_call_close(args_tail)
            closed = end_idx != -1
            if closed:
                args_tail = args_tail[:end_idx]
                # Emit the FULL args document exactly once, on close — whether
                # the header shipped in a prior delta (multi-delta stream) or in
                # this same delta (whole call in one chunk). Only skip when the
                # call was suppressed (off-list name, no header) or its args were
                # already emitted (idempotent on repeated ticks post-close).
                already = (
                    self.streamed_args_for_tool[idx]
                    if idx < len(self.streamed_args_for_tool)
                    else ""
                )
                if idx not in self._suppressed_tools and not already:
                    final_args = self._final_args_json(args_tail)
                    if final_args:
                        if idx < len(self.streamed_args_for_tool):
                            self.streamed_args_for_tool[idx] = final_args
                        if idx < len(self.prev_tool_call_arr):
                            self.prev_tool_call_arr[idx]["arguments"] = final_args
                        deltas.append(
                            {
                                "index": self._assign_client_index(idx),
                                "function": {"arguments": final_args},
                            }
                        )

        if closed:
            # Transition back to SEEKING_NAME so the next opener (if any)
            # starts a fresh indexed call on the next drain iteration / tick.
            self.current_tool_id = idx + 1
            self._name_sent = False
            self._current_tool_ref = None

        return deltas, closed

    def _emit_sepless_closed_call(
        self, buffer: str, idx: int, request: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], bool]:
        """Emit a SEP-LESS closed call (no ``<tool_sep>`` in the body).

        The Hy3 wire default is ``NAME<tool_sep>{JSON}<end>``; the streaming FSM
        is built around that separator. A degraded XML-pair or bare-name body
        (``NAME<arg_key>k</arg_key><arg_value>v</arg_value><end>`` or ``NAME<end>``)
        carries no separator, so the incremental phase-1/phase-2 split cannot
        apply. Since such a call only becomes recognizable ONCE it is closed
        (``<end_of_tool_call>`` present — the caller gates on that), we parse the
        whole body via the shared non-streaming ``_parse_body`` and emit the
        header + complete args in a single tick, mirroring the whole-call-in-one-
        delta JSON path. Then the FSM advances to the next index so a following
        opener in the same delta drains too (codex R10 BLOCKING).
        """
        # A sep-less XML-pair value may carry the literal ``<end_of_tool_call>``
        # string as free-form text; skip complete ``<arg_value>…</arg_value>``
        # spans so it is not mistaken for the close (codex R16, same hazard as
        # R15's ``_find_call_close`` / ``_find_call_close_in_body`` fixes).
        end_at = self._end_token_outside_arg_value(buffer)
        body = buffer[:end_at] if end_at != -1 else buffer
        name, args = self._parse_body(body)

        # Advance the FSM regardless of what we emit (matches the JSON close path).
        self.current_tool_id = idx
        self._name_sent = True
        if idx >= len(self.streamed_args_for_tool):
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})

        deltas: list[dict[str, Any]] = []
        if name:
            valid_names = self._get_tool_names(request)
            suppressed = bool(valid_names) and name not in valid_names
            if not suppressed:
                final_args = json.dumps(args, ensure_ascii=False)
                if idx < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[idx] = final_args
                if idx < len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr[idx]["arguments"] = final_args
                client_idx = self._assign_client_index(idx)
                deltas.append(
                    {
                        "index": client_idx,
                        "id": generate_tool_id(),
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                )
                deltas.append(
                    {"index": client_idx, "function": {"arguments": final_args}}
                )
            else:
                self._suppressed_tools.add(idx)

        # Close: advance to the next index; the drain loop picks up any
        # subsequent opener in this same delta.
        self.current_tool_id = idx + 1
        self._name_sent = False
        self._current_tool_ref = None
        return deltas, True

    def _resync_args_body(self, args_tail: str) -> str:
        """Strip a poisoned/noise prefix from an args body up to its last
        resync boundary (codex R14).

        ``_find_call_close`` accepts a close after RESYNCHRONIZING past a
        mid-stream noise ``<end_of_tool_call>`` that appears outside a string
        while an object is still open (``{"a": <end>{"a": 42}`` — a never-closed
        leading object then the REAL one). The close offset it returns points
        at the REAL close, so the sliced ``args_tail`` still carries the broken
        noise prefix (``{"a": <end>`` before the real ``{"a": 42}``). Serializing
        from the START would ``raw_decode``-fail on that prefix and yield
        ``{}``. Apply the SAME resync rule the close-finder uses: walk the body,
        and each time an ``<end_of_tool_call>`` lands outside a string while an
        object is still open (depth > 0), discard everything up to and including
        it and restart at depth 0. What remains after the last such boundary is
        the real object body. A clean body (no noise ``<end>``) is returned
        unchanged.
        """
        end_tok = self.tool_call_end_token
        in_string = False
        escaped = False
        depth = 0
        i = 0
        n = len(args_tail)
        start = 0
        while i < n:
            if not in_string and args_tail.startswith(end_tok, i):
                if depth > 0:
                    # Noise close inside a still-open object — resync: everything
                    # up to and including this token is broken prefix.
                    i += len(end_tok)
                    start = i
                    depth = 0
                    continue
                # A close at depth <= 0 should already have bounded args_tail
                # upstream; nothing more to strip.
                break
            ch = args_tail[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        return args_tail[start:]

    def _final_args_json(self, args_tail: str) -> str:
        """Serialize the complete args body (JSON or XML pairs) to a JSON
        string. Returns ``"{}"`` on an empty / unparseable body so the
        emitted ``arguments`` is always valid JSON."""
        # Drop any poisoned noise prefix left in place by the close-finder's
        # resync (codex R14), so the real trailing object decodes cleanly.
        args_tail = self._resync_args_body(args_tail)
        stripped = args_tail.strip()
        if stripped.startswith("{"):
            try:
                parsed, _end = self._json_decoder.raw_decode(stripped)
                if isinstance(parsed, dict):
                    return json.dumps(parsed, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                return "{}"
            return "{}"
        args: dict[str, Any] = {}
        for m in self._arg_pair_re.finditer(args_tail):
            key = m.group("key").strip()
            if not key:
                continue
            args[key] = _deserialize_arg_value(m.group("val"))
        return json.dumps(args, ensure_ascii=False)

    def _find_call_close(self, args_tail: str) -> int:
        """Offset of the ``<end_of_tool_call>`` that closes this call, or -1.

        JSON-aware: when the args are a WELL-FORMED JSON body, a literal
        ``<end_of_tool_call>`` inside a string value is NOT a close — search for
        the real close only AFTER the JSON prefix (``raw_decode`` consumes only
        the valid JSON prefix so an interior literal is ignored). This mirrors
        the ``</arg_value>`` literal handling.

        When the ``{``-body does NOT ``raw_decode`` (malformed OR still
        truncated mid-stream) we must NOT blindly accept the first
        ``<end_of_tool_call>`` substring (codex R11 BLOCKING): a still-streaming
        JSON string value can legitimately contain the literal close token
        (``{"m": "contains <end_of_tool_call> inside``) with the JSON not yet
        finished, and treating that literal as the real close prematurely emits
        ``{}``. So for a ``{``-body we only accept a close token that lies
        OUTSIDE any JSON string (``_end_token_outside_string``). A completed but
        malformed body (``{bad}<end>`` — codex R9) still closes because its
        ``<end>`` sits outside a string; a truncated body whose only close-token
        occurrence is inside an unterminated string returns -1 and keeps
        streaming.
        """
        stripped = args_tail.lstrip()
        lead_ws = len(args_tail) - len(stripped)
        if stripped.startswith("{"):
            try:
                parsed, end = self._json_decoder.raw_decode(stripped)
                if isinstance(parsed, dict):
                    after = self.tool_call_end_token
                    pos = args_tail.find(after, lead_ws + end)
                    return pos
            except (json.JSONDecodeError, ValueError):
                # Malformed OR still-truncated body. Accept only a close token
                # that is (a) not inside an (open) JSON string AND (b) reached
                # after the object's braces have balanced back to depth 0 — so
                # neither an interior literal in an unterminated string
                # (``{"m": "…<end>``) NOR a close after a still-open object
                # (``{"a": <end>`` — value not yet arrived) is mistaken for a
                # real close. A completed-but-malformed body (``{bad}<end>``,
                # codex R9) closes because its braces balance before the token.
                return self._end_token_at_object_close(args_tail)
        # Non-JSON (XML-pair / empty / bare). A ``<arg_value>`` payload CAN
        # legitimately contain the literal ``<end_of_tool_call>`` string
        # (free-form text), so a plain ``find`` would truncate the call early
        # and drop the argument (codex R15 BLOCKING). Skip over each complete
        # ``<arg_value>…</arg_value>`` span so an end-token INSIDE a value is
        # never mistaken for the real close; accept the first end-token that
        # lands OUTSIDE any arg-value span.
        return self._end_token_outside_arg_value(args_tail)

    def _end_token_outside_arg_value(self, body: str) -> int:
        """Offset of the first ``<end_of_tool_call>`` in ``body`` that is NOT
        inside a complete ``<arg_value>…</arg_value>`` span, or -1.

        The XML-pair argument form (``<arg_key>K</arg_key><arg_value>V
        </arg_value>``) may carry the literal wire end-token inside a value ``V``
        as ordinary text. Walk the body tracking whether we are inside an
        ``<arg_value>`` span (opened by ``<arg_value>``, closed by
        ``</arg_value>``); an end-token seen while inside a span is argument
        text, not the close. An UNTERMINATED trailing ``<arg_value>`` (value
        still streaming, no ``</arg_value>`` yet) keeps us in-span to the end so
        a literal end-token in the partial value does not close prematurely —
        the caller keeps buffering until the value and the real close arrive.
        """
        end_tok = self.tool_call_end_token
        av_open = self.arg_value_start_token
        av_close = self.arg_value_end_token
        i = 0
        n = len(body)
        in_value = False
        while i < n:
            if not in_value:
                if body.startswith(end_tok, i):
                    return i
                if body.startswith(av_open, i):
                    in_value = True
                    i += len(av_open)
                    continue
            elif body.startswith(av_close, i):
                in_value = False
                i += len(av_close)
                continue
            i += 1
        return -1

    def _end_token_at_object_close(self, body: str) -> int:
        """First ``<end_of_tool_call>`` in a ``{``-body that closes a
        brace-balanced object outside a JSON string, or -1 (codex R12 / R14).

        Walks the body as a lightweight lexer tracking two things:
          * JSON string state — a ``"`` toggles in-string unless
            backslash-escaped; a close-token inside an open string is argument
            text, not a real close.
          * brace depth — ``{`` / ``}`` outside strings raise / lower depth.

        A close token is a real close only when it is reached while NOT in a
        string and with brace depth ``<= 0`` (the object immediately preceding
        it has balanced closed). This rejects both an interior literal in an
        unterminated string (``{"m": "…<end>``, in-string) and a close after a
        still-open object (``{"a": <end>``, depth 1 — the value has not arrived
        yet), while still accepting a completed-but-malformed body
        (``{bad}<end>`` — its ``}`` drops depth to 0 before the token).

        RESYNC on noise (codex R14 BLOCKING): when a close token is reached
        OUTSIDE a string but while an object is still OPEN (depth > 0), that
        token is mid-stream NOISE — an ``<end_of_tool_call>`` cannot legitimately
        appear outside a string inside a well-formed JSON object, so the object
        so far is broken. Rather than let the never-closed leading ``{`` poison
        depth forever (so the REAL later ``{…}<end>`` never balances back to 0
        and the call hangs pending), we ABANDON the broken prefix: skip PAST the
        noise close token and RESET depth to 0, treating the bytes after it as a
        fresh object candidate. So ``{"a": <end>{"a": 42}<end>`` (noise open +
        real object) resynchronizes and accepts the real close, while a
        genuinely still-open object with no later balanced object
        (``{"a": <end>``) still returns -1 (keep streaming). Used only on the
        ``raw_decode``-failed path.
        """
        end_tok = self.tool_call_end_token
        in_string = False
        escaped = False
        depth = 0
        i = 0
        n = len(body)
        while i < n:
            if not in_string and body.startswith(end_tok, i):
                if depth <= 0:
                    # Brace-balanced object precedes this close → real close.
                    return i
                # Depth > 0: an ``<end>`` outside a string but inside a still-open
                # object is noise. Abandon the broken prefix — skip past this
                # token and resync depth to 0 so a later balanced ``{…}<end>``
                # is still recognized (codex R14).
                i += len(end_tok)
                depth = 0
                continue
            ch = body[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        return -1

    def has_pending_tool_call(self, text: str) -> bool:
        """Override — Hy3 opener/closer are the pinned fixed strings.

        "Pending" means the stream may end mid-markup and the parser is still
        waiting for a closing delimiter. That is true ONLY for a NATIVE call:
        the LAST ``<tool_call>`` opener has no ``<end_of_tool_call>`` after it.
        A completed call earlier in ``text`` does not leave the parser pending
        forever.

        The text-format degradation (``[Calling tool="X" k="v"]``) is NOT
        pending (codex R10 BLOCKING): it is a COMPLETE, self-delimited call with
        no trailing close delimiter to wait for, and it is finalized via the
        non-streaming recovery path in ``finalize()`` (gated on the ``[Calling``
        marker, not on this predicate). Reporting it as pending made streaming
        shutdown treat a finished ``[Calling …]`` message as perpetually
        in-flight. So a bare ``[Calling …]`` with no unmatched native opener
        returns ``False``.
        """
        opener = text.rfind(self.tool_call_start_token)
        return opener != -1 and self.tool_call_end_token not in text[opener:]
