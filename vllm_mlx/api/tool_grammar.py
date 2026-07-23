# SPDX-License-Identifier: Apache-2.0
"""Grammar-constrained tool calling (#558) — grammar builder + runtime processor.

This module turns a chat request's ``tools`` + ``tool_choice`` into an
llguidance grammar that STRUCTURALLY GUARANTEES every emitted tool call
(a) names a tool that actually exists in the list, (b) whose ``arguments``
satisfy that tool's JSON Schema, and (c) uses the model family's tool-call
wire format. It constrains only the FORM of a call, never the decision of
whether/which to call.

Layering: the pure, side-effect-free grammar BUILDER (``build_tool_grammar`` /
``build_tool_lark``), the ``StructureInfo`` wire-triple dataclass and the
compiled-grammar LRU cache landed in PR-1; the per-family
``ToolParser.structure_info()`` overrides landed in PR-2. PR-3 (this change)
adds the RUNTIME half: ``GrammarLogitsProcessor`` (the per-token mask that
applies a compiled grammar to logits each decode step) and ``build_lltokenizer``
(the ``LLTokenizer`` factory). The chat route / scheduler wiring that carries a
per-request processor into the decode loop lands alongside these in PR-3.

Design + prior-art: ``design-558-constrained-tool-calling.md``. The
mechanism is the "structural tags with triggers" pattern that vLLM
(``TriggeredTagsFormat``), SGLang (``LegacyStructuralTagResponseFormat`` +
``BaseFormatDetector.structure_info``) and llguidance (``StructTag``) all
converge on. We PORT llguidance's ``StructTag.to_grammar`` Lark-assembly
algorithm (``llguidance/_struct_tag.py``) rather than inventing our own,
per the charter guardrail. ``StructureInfo`` mirrors SGLang's per-detector
``structure_info() -> (begin, end, trigger)`` contract.

Ground-truth correction to the design doc (verified on the real Qwen3.5
tokenizer, 2026-07): the ``<tool_call>`` / ``</tool_call>`` sentinels are
SINGLE SPECIAL TOKENS (ids 248058 / 248059), not multi-byte text. So we
must reference them as Lark special-token literals on BOTH the
trigger/begin AND the ``end`` — ``StructTag.to_grammar`` only special-cases
the trigger, leaving ``</tool_call>`` in ``end`` as a byte string the
model's single ``</tool_call>`` token can never satisfy. We therefore build
the Lark directly (still following StructTag's algorithm) with a per-family
list of "sentinel" substrings that must render as special-token refs.
"""

import json
import logging
import re
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Reuse the exact llguidance surface the guided-decoding path already imports
# (proves availability + shares the native MLX mask kernel). Detection is split
# into two INDEPENDENT layers so a failure in the runtime-only bridge cannot
# disable the grammar BUILDER (codex #558-PR3):
#
#   * BUILDER + mask layer (``HAS_LLGUIDANCE``): the compile-side factory
#     (``LLMatcher.grammar_from_lark``) drives ``build_tool_grammar``, and the
#     per-token mask kernel (``allocate_token_bitmask`` /
#     ``fill_next_token_bitmask`` / ``apply_token_bitmask``) drives the mask in
#     ``GrammarLogitsProcessor``. If this layer is present the builder works.
#   * RUNTIME BRIDGE layer (``HAS_LL_TOKENIZER``): ``llguidance.hf`` +
#     ``LLTokenizer`` build the tokenizer that the runtime processor needs. If
#     ONLY this layer is missing, grammars still compile — the runtime falls
#     back (``get_lltokenizer`` returns ``None`` -> free-form) without falsely
#     reporting the whole feature unavailable.
try:
    from llguidance.mlx import (
        LLMatcher,
        allocate_token_bitmask,
        apply_token_bitmask,
        fill_next_token_bitmask,
    )

    HAS_LLGUIDANCE = True
except ImportError:  # pragma: no cover - mirrors guided.py degrade path
    HAS_LLGUIDANCE = False
    LLMatcher = None
    allocate_token_bitmask = None
    apply_token_bitmask = None
    fill_next_token_bitmask = None

try:
    import llguidance.hf as _llguidance_hf
    from llguidance import LLTokenizer

    HAS_LL_TOKENIZER = True
except ImportError:  # pragma: no cover - runtime bridge only
    HAS_LL_TOKENIZER = False
    _llguidance_hf = None
    LLTokenizer = None


@dataclass(frozen=True)
class StructureInfo:
    """Per-tool wire triple, mirroring SGLang's ``StructureInfo``.

    ``begin`` MUST start with ``trigger`` (StructTag invariant). ``{name}``
    in ``begin`` is already substituted with the concrete tool name by
    ``ToolParser.structure_info()``. ``sentinels`` lists literal substrings
    inside ``begin``/``end`` that are single special tokens for this family
    and must be emitted as Lark special-token refs, not byte strings.

    ``arg_style`` selects how the tool's ARGUMENTS are constrained between
    ``begin`` and ``end``:

      * ``"json"`` (default) — a single JSON object constrained by the tool's
        JSON Schema via ``%json`` (hermes / qwen / harmony wire).
      * ``"xml"`` — a Qwen3-Coder XML arg body: one
        ``<parameter=KEY>\nVALUE\n</parameter>`` block per property, each VALUE
        constrained per its sub-schema (raw text for strings, ``%json`` for
        scalars/objects/arrays, an alternation for enums). See
        ``_emit_xml_arg_body``. Default stays ``"json"`` so hermes/qwen/harmony
        (and any out-of-tree JSON-body family) are byte-identical to before.
      * ``"gemma4"`` (#558 E4) — the Gemma-4 native arg body:
        ``call:NAME{k1:v1,k2:v2,...}`` with DICTSORT-ordered, COMMA-separated
        ``key:VALUE`` pairs, string values wrapped in the ``<|"|>`` special-token
        marker, bare scalars/booleans, ``%json`` objects/arrays, and an
        alternation for enums. See ``_emit_gemma4_arg_body``.
    """

    begin: str
    end: str
    trigger: str
    sentinels: tuple[str, ...] = field(default=())
    arg_style: str = "json"


def _is_registered_added_token(tokenizer: Any, tok_id: int) -> bool:
    """True iff ``tok_id`` is an EXPLICITLY-REGISTERED added/special token.

    ``len(encode(s)) == 1`` is necessary but NOT sufficient to render ``s`` as a
    Lark special-token ref: an ordinary single BPE-merge vocabulary token (e.g.
    ``"hello"``) is not a referenceable atomic token, and an unknown string may
    collapse to a single ``[UNK]`` id. The grammar builder emits ``<s>`` as a
    special-token reference that only resolves against a token the tokenizer
    holds as an EXPLICITLY-ADDED atomic entry (the tokenizer will never split
    such a token further, so the model can emit it as one piece). So we require
    the id to appear in the tokenizer's added-token registry.

    GROUND TRUTH (verified on the pinned Qwen3.5 tokenizer, 2026-07): its
    ``<tool_call>``/``</tool_call>`` added tokens carry ``AddedToken.special ==
    False`` and are NOT in ``all_special_ids`` — yet they ARE distinct atomic
    added tokens the grammar's special-token ref resolves against (the #558
    enforcement tests pass on exactly this tokenizer). We therefore key on
    ADDED-TOKEN registration (``added_tokens_decoder`` membership), NOT the HF
    ``special`` flag: gating on ``special==True`` / ``all_special_ids`` would
    wrongly REJECT the real target's ``<tool_call>`` and disable the feature for
    the very tokenizer it ships for. The ``special`` flag distinguishes control
    tokens (``<|endoftext|>``) from added content tokens; both are atomic and
    both are valid special-token-ref targets, so it is not the right gate here.

    What this correctly REJECTS: an ordinary BPE token not in the added-token
    registry (``"hello"`` -> not in ``added_tokens_decoder`` -> False). Probes
    the added-token surface, then falls back to ``all_special_ids`` for
    tokenizers that expose no ``added_tokens_decoder`` (older/slow). Degrades to
    ``False`` (caller opts out) if neither is available or any raises.
    """
    # transformers fast tokenizers expose ``added_tokens_decoder``: {id: AddedToken}.
    # Membership means the id is an explicitly-added atomic token (special flag
    # irrelevant — see docstring GROUND TRUTH).
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(decoder, dict):
        return tok_id in decoder
    # Fallback for tokenizers without added_tokens_decoder: all_special_ids
    # (control tokens). This path only sees special==True ids, which is fine —
    # it is a strictly-narrower best-effort fallback, not the primary gate.
    try:
        special_ids = getattr(tokenizer, "all_special_ids", None)
        if special_ids is not None:
            return tok_id in set(special_ids)
    except Exception:
        pass
    return False


def are_single_special_tokens(tokenizer: Any, candidates: tuple[str, ...]) -> bool:
    """True iff EVERY candidate is a DISTINCT single registered special token.

    A per-family ``structure_info()`` declares its ``<...>`` sentinels as
    special-token refs (see ``StructureInfo.sentinels``) ONLY when the model's
    tokenizer actually encodes each one as a single special token — this is the
    ground-truth-correction-#1 assumption (``<tool_call>``/``</tool_call>`` are
    single special tokens in Qwen3/Hermes tokenizers). It is NOT universal: the
    same ``hermes`` wire on a Llama-based Hermes tokenizer encodes
    ``<tool_call>`` as ordinary multi-token text, where declaring it a
    special-token sentinel would build an UNENFORCEABLE grammar (the model has
    no single ``<tool_call>`` token to satisfy the ref). A family that cannot
    prove single-token sentinels must OPT OUT (``structure_info() -> None``) and
    fall back to today's free-form-then-parse behavior rather than emit a
    grammar its tokenizer can never satisfy.

    ``len(encode(s)) == 1`` alone is NOT enough (codex review): it also accepts
    a string that collapses to a single ``[UNK]`` id or resolves to an ordinary
    (non-special) vocabulary token, either of which yields an unenforceable
    special-token grammar. So each candidate must ALSO:

      * round-trip: ``decode([id]) == s`` (rejects ``[UNK]`` collapse / lossy
        normalization — an unknown sentinel decoding to ``"<unk>"`` fails here);
      * be an EXPLICITLY-REGISTERED added token (rejects an ordinary single
        BPE-merge vocab token — e.g. ``"hello"`` — that is not a referenceable
        atomic token; keys on added-token registration, NOT the HF ``special``
        flag, because the real Qwen ``<tool_call>`` is ``special=False``);
      * resolve to an id DISTINCT from every other candidate (rejects two
        sentinels collapsing to the same id — e.g. both to ``[UNK]`` — which
        would make the open/close tags grammar-indistinguishable).

    Returns ``False`` (conservative: caller opts out) when the tokenizer is
    absent or lacks ``encode``/``decode`` / raises — grammar constraint is a
    best-effort opt-in, never a hard requirement, so an unknown or partially
    featured tokenizer degrades safely to free-form.
    """
    if tokenizer is None:
        return False
    encode = getattr(tokenizer, "encode", None)
    decode = getattr(tokenizer, "decode", None)
    if encode is None or decode is None:
        return False
    seen_ids: set[int] = set()
    for tok_str in candidates:
        try:
            ids = encode(tok_str, add_special_tokens=False)
        except Exception:
            # A tokenizer that rejects the probe cannot prove single-token
            # status -> conservatively report False so the family opts out.
            return False
        if len(ids) != 1:
            return False
        tok_id = ids[0]
        if tok_id in seen_ids:
            return False  # two sentinels collapsed to the same id (e.g. [UNK])
        seen_ids.add(tok_id)
        # Round-trip: the single id must decode back to the exact candidate.
        # Rejects [UNK] collapse and any lossy normalization.
        try:
            if decode([tok_id]) != tok_str:
                return False
        except Exception:
            return False
        # Must be an explicitly-registered added token, not an ordinary BPE
        # vocab token (see _is_registered_added_token — keyed on added-token
        # registration, not the HF ``special`` flag).
        if not _is_registered_added_token(tokenizer, tok_id):
            return False
    return True


def resolve_reasoning_sentinels(
    reasoning_parser_name: str | None, tokenizer: Any
) -> tuple[str, ...]:
    """Reasoning-boundary special tokens (``<think>``/``</think>``) for path A.

    Looks up the configured reasoning parser (by the server config's
    ``reasoning_parser_name``) and reads its ``start_token`` / ``end_token``
    markers (e.g. ``<think>`` / ``</think>``). Returns ONLY the markers that are
    PROVEN single special tokens on ``tokenizer`` (via
    ``are_single_special_tokens``), so the grammar's reasoning-tolerant prefix
    references real atomic tokens the model can emit — never a byte string the
    lazy prefix cannot match (#558 PR-4 GROUND TRUTH; see ``build_tool_lark``).

    Returns ``()`` (the non-reasoning path — grammar byte-identical to PR-3)
    when: no reasoning parser is configured, the parser class exposes no
    ``start_token``/``end_token``, the markers are not single special tokens on
    this tokenizer, or anything raises. A missing/degenerate reasoning parser
    must NOT disable tool-call enforcement — it only means the free prefix is
    the bare ``TAG_TEXT`` (no reasoning tolerance), which is correct for a
    non-reasoning model.

    We reuse ``are_single_special_tokens`` — the exact gate the tool sentinels
    use — so ``<think>`` on a tokenizer that splits it into ordinary text is
    correctly excluded (declaring it a special-token ref there would build an
    unenforceable prefix, the same failure mode the sentinel gate guards).
    """
    if not reasoning_parser_name or tokenizer is None:
        return ()
    try:
        from ..reasoning import get_parser
    except Exception:
        # The reasoning package failed to import — an unexpected defect, not an
        # unconfigured parser. Log it (with the parser name) so it is not
        # silently indistinguishable from the benign case (codex #558-PR4 nit).
        logger.exception(
            "tool-grammar: reasoning package import failed for parser %r; "
            "no reasoning tolerance",
            reasoning_parser_name,
        )
        return ()
    try:
        parser_cls = get_parser(reasoning_parser_name)
    except KeyError:
        # EXPECTED: an unknown/unregistered reasoning parser name. No reasoning
        # tolerance (free prefix stays bare TAG_TEXT); never disables tool
        # enforcement. This is a benign config case, not logged as an error.
        return ()
    except Exception:
        # UNEXPECTED lookup failure (registry defect) — surface it with the
        # parser name rather than silently degrading, so a real bug is not
        # masked as "parser not configured" (codex #558-PR4 nit).
        logger.exception(
            "tool-grammar: unexpected error resolving reasoning parser %r; "
            "no reasoning tolerance",
            reasoning_parser_name,
        )
        return ()
    markers: list[str] = []
    for attr in ("start_token", "end_token"):
        try:
            # ``start_token``/``end_token`` are read-only properties on the
            # concrete parsers (deepseek_r1/qwen3/glm4); reading them off the
            # CLASS returns the property object, so instantiate to get the str.
            # The reasoning parsers take no required ctor args.
            value = getattr(parser_cls(), attr, None)
        except Exception:
            value = None
        if isinstance(value, str) and value:
            markers.append(value)
    if not markers:
        return ()
    # Dedup preserving order (start_token/end_token are distinct, but a parser
    # could in principle repeat one).
    ordered = tuple(dict.fromkeys(markers))
    # Only keep markers that are single special tokens on THIS tokenizer.
    kept = tuple(m for m in ordered if are_single_special_tokens(tokenizer, (m,)))
    return kept


def _is_lark_special_token_ref(s: str) -> bool:
    """True iff ``s`` is safe to emit as a BARE llguidance special-token ref.

    llguidance Lark treats a bare ``<name>`` as a special-token reference, so a
    reasoning sentinel is interpolated verbatim into the grammar source. Being a
    single special token on the tokenizer does NOT guarantee the string is valid
    Lark rule syntax when interpolated raw — e.g. a hypothetical ``[THINK]``
    marker would be parsed as a Lark char-class, and an inner ``>``/whitespace
    would break the reference (codex #558-PR4 nit). We therefore require the
    ``<...>`` special-token-ref shape: a leading ``<``, a trailing ``>``, a
    non-empty body, and no interior ``<``/``>`` or whitespace that would
    desync the reference. Reasoning markers that fail this are DROPPED (the free
    prefix silently loses that marker's tolerance) rather than emitting Lark
    that fails to compile — a best-effort degrade consistent with the rest of
    this module. In practice the shipped reasoning parsers only ever produce
    ``<think>``/``</think>``, which pass.
    """
    if len(s) < 3 or s[0] != "<" or s[-1] != ">":
        return False
    body = s[1:-1]
    if not body:
        return False
    # Reject anything that would break out of the ``<...>`` reference: interior
    # angle brackets or any whitespace (Lark token refs are a single lexeme).
    return not any(c in body for c in "<>") and not any(c.isspace() for c in body)


def _lark_escape(s: str) -> str:
    """Render ``s`` as a Lark double-quoted string literal (JSON-escaped)."""
    if not s:
        return ""
    return json.dumps(s)


def _emit_literal_with_sentinels(text: str, sentinels: tuple[str, ...]) -> str:
    """Split ``text`` on sentinel substrings, emitting sentinels as special
    tokens and the rest as quoted byte-string literals.

    E.g. ``"}\\n</tool_call>"`` with sentinel ``"</tool_call>"`` renders to
    ``"}\\n" </tool_call>`` — the trailing sentinel becomes a special-token
    ref so the model's single ``</tool_call>`` token satisfies the grammar.
    """
    if not text:
        return ""
    ordered = sorted((s for s in sentinels if s), key=len, reverse=True)
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for sent in ordered:
            if text.startswith(sent, i):
                matched = sent
                break
        if matched is not None:
            # Sentinels are already in ``<...>`` special-token form (e.g.
            # ``<tool_call>``). Emit verbatim — llguidance Lark treats a
            # bare ``<name>`` as a special-token reference. Wrapping it
            # again would yield ``<<tool_call>>`` (a lexer error).
            parts.append(matched)
            i += len(matched)
        else:
            j = i + 1
            while j < n and not any(text.startswith(s, j) for s in ordered):
                j += 1
            parts.append(_lark_escape(text[i:j]))
            i = j
    return " ".join(p for p in parts if p)


# The raw-string value construct used by the XML arg body (see
# ``_emit_xml_param_value``). A Qwen3-Coder string parameter value is "any text
# up to the FIRST literal ``</parameter>`` close tag" — so a value MAY contain a
# literal ``<`` (a ``code`` arg such as ``a < b``, ``<html>...``, or C++
# ``vector<int>``), it just may not contain the whole ``</parameter>`` delimiter.
# We express that with llguidance's first-class LAZY lexeme (``rule[lazy]:``): a
# lazy rule matches its leading text terminal MINIMALLY, stopping at the first
# occurrence of the trailing literal. This is the exact idiom llguidance's own
# ``StructTag.to_grammar`` uses for "text until a tag" (``_struct_tag.py``:
# ``..._trig[lazy]: TAG_TEXT <trigger>``), and it mirrors XGrammar's ``qwen_xml``
# style (value = any text up to the first ``</parameter>``).
#
# ``XML_PARAM_TEXT: /(.|\n)*/`` admits ANY byte (crucially including ``<``); the
# lazy rule then binds it to the ``</parameter>`` delimiter so the value
# terminates at the first close. The lazy rule consumes the value, the single
# ``\n`` the wire puts BEFORE ``</parameter>``, AND the ``</parameter>`` tag
# itself — so ``_emit_xml_param_value`` appends only the trailing ``\n`` (the
# separator AFTER the close). The ``qwen3_coder`` parser strips the leading /
# trailing ``\n`` from the captured value, so this reproduces the exact surface
# form it round-trips. FIRST-``</parameter>`` semantics: a value that literally
# contains ``</parameter>`` closes there (same as XGrammar — acceptable).
# NOTE (previous limitation, now FIXED): the pilot used ``XMLSTR: /[^<]*/`` which
# stopped at ANY ``<`` and thus SILENTLY TRUNCATED a ``<``-containing code arg.
_XML_STRING_VALUE_TERMINAL = "XML_PARAM_TEXT"
_XML_STRING_VALUE_RULE = "xml_param_value"
# ``\\n`` here is a LITERAL backslash-n (the Lark regex ``/(.|\n)*/``), matching
# how ``TAG_TEXT`` is declared; the trailing ``\n`` are real line breaks.
_XML_STRING_TERMINAL_DECL = (
    f"{_XML_STRING_VALUE_TERMINAL}: /(.|\\n)*/\n"
    f'{_XML_STRING_VALUE_RULE}[lazy]: {_XML_STRING_VALUE_TERMINAL} "</parameter>"\n'
)


# The gemma4 string-value construct (#558 E4). A string value is wrapped in the
# ``<|"|>`` marker, a SINGLE SPECIAL TOKEN (id 52 on the target tokenizer). The
# value is a RULE ``<|"|> GEMMA_STR_TEXT <|"|>`` with BOTH ``<|"|>`` as RULE-LEVEL
# special-token refs — the atomic close token 52 the model actually emits. The
# close MUST stay a rule-level special-token ref (not a ``[lazy]``/byte construct):
# llguidance REJECTS a special token inside a lexer terminal, so a lazy form
# (``rule[lazy]: TEXT <|"|>``) cannot use the special token as its terminator, and a
# byte-literal ``"<|\"|>"`` close COMPILES but is runtime-DEAD (the atomic token 52
# never satisfies a byte literal). Both verified on the real tokenizer — the lazy /
# byte-literal-close forms accept ZERO valid calls.
#
# THE CONTENT MUST EXCLUDE THE BYTE SPELLING OF ``<|"|>`` (#558 E4, codex r4 — a
# REAL under-constraint fix). A bare ``GEMMA_STR_TEXT: /(.|\n)*/`` admits ANY byte
# sequence, so the model can spell the 5 bytes ``<|"|>`` with ORDINARY tokens (ids
# 236820/236909/236775/236909/236813 — none is the atomic token 52) INSIDE a value.
# The gemma4 parser text-SCANS the DECODED output for the byte substring ``<|"|>``:
# ``_scan_gemma4_tool_calls`` (``tool_parsers/gemma4_tool_parser.py`` line ~97)
# toggles ``in_gemma_string`` on EVERY ``<|"|>`` occurrence, NOT on token ids — so
# an ordinary-byte-spelled ``<|"|>`` mid-value is INDISTINGUISHABLE from the atomic
# close and terminates the string EARLY, desyncing the ``key:value`` framing. We
# therefore EXCLUDE the byte sequence ``<|"|>`` from the content via llguidance's
# documented regex And/Not algebra (``docs/syntax.md``):
# ``GEMMA_STR_TEXT: /(.|\n)*/ & ~/(?s:.*)<\|"\|>(?s:.*)/`` — "any UTF-8 bytes AND
# NOT (contains ``<|"|>``)". ``~`` (complement) / ``&`` (intersection) are the
# engine's real operators (no lookahead — which llguidance / Rust ``regex`` lack);
# intersecting with ``/(.|\n)*/`` keeps the negation UTF-8-safe (``~`` alone may
# match invalid UTF-8 — this is the syntax doc's own
# ``/(?s:.*)/ & ~/(?s:.*)AAA(?s:.*)/`` recipe). VERIFIED on the real tokenizer: the
# terminal accepts values containing ``<``/``>``/``|``/``{``/``}``/``,``/quotes/
# newlines and even a 4-byte marker PREFIX (``<|"|``) adjacent to the close, yet
# REJECTS at the exact ``>`` token that would complete an ordinary-byte ``<|"|>``
# mid-value — keeping the constrained wire and the parser boundary in EXACT
# agreement (the grammar forbids precisely the byte substring the parser reads as a
# close). Only ``<|"|>`` needs excluding: while ``in_gemma_string`` the parser skips
# ALL other bytes (frame tokens ``<|tool_call>``/``<tool_call|>``/``<|channel>``/
# ``<channel|>``, braces, commas) until the next ``<|"|>``, so a byte-spelling of
# those inside a value is harmless. Enum values get the same guarantee structurally
# (``_gemma4_enum_wire_unsafe`` / ``_enum_wire_embeds_special_token`` opt out any
# value whose wire embeds a structural marker).
_GEMMA4_STRING_VALUE_TERMINAL = "GEMMA_STR_TEXT"
_GEMMA4_STRING_VALUE_RULE = "gemma_str_value"
# The gemma4 string-delimiter marker (``<|"|>``). Declared as a single special
# token by ``Gemma4ToolParser`` and referenced as a bare Lark special-token ref;
# it passes ``_is_lark_special_token_ref`` (leading ``<``, trailing ``>``, no
# interior ``<``/``>`` or whitespace — the inner ``|"|`` is fine).
_GEMMA4_STRING_MARKER = '<|"|>'
# The SAME ``<|"|>`` marker as a REGEX-ESCAPED byte literal (the two ``|`` escaped
# for the regex engine; ``<``/``"``/``>`` are literal), used ONLY in the negated
# substring that keeps the byte spelling out of string content. Co-located with the
# special-token-ref spelling so the marker has one obvious source of truth.
_GEMMA4_STRING_MARKER_RE = r'<\|"\|>'
# ``GEMMA_STR_TEXT`` is a fresh terminal (a DISTINCT name from
# ``XML_PARAM_TEXT``/``TAG_TEXT`` so a mixed tool-set never cross-binds them). Its
# base is the same ``/(.|\n)*/`` any-byte pattern, intersected (``&``) with the
# complement (``~``) of "contains ``<|"|>``" so the content can never spell the
# string delimiter (see the block above). The lowercase ``gemma_str_value`` is a
# RULE (not a terminal) because the special-token markers must live at rule level.
_GEMMA4_STRING_RULE_DECL = (
    f"{_GEMMA4_STRING_VALUE_TERMINAL}: /(.|\\n)*/ & "
    f"~/(?s:.*){_GEMMA4_STRING_MARKER_RE}(?s:.*)/\n"
    f"{_GEMMA4_STRING_VALUE_RULE}: {_GEMMA4_STRING_MARKER} "
    f"{_GEMMA4_STRING_VALUE_TERMINAL} {_GEMMA4_STRING_MARKER}\n"
)


# ------------------------------------------------------------------------
# STRICT-ALLOWLIST representability guard for the Qwen3-Coder XML arg wire
# (#558 E3). The XML emitter (``_emit_xml_arg_body`` / ``_emit_xml_param_value``)
# is a BEST-EFFORT OPT-IN: it can faithfully constrain the common tool schema
# (typed top-level ``properties``, enums, nested objects/arrays via ``%json``,
# required + optional ``( )?``), but MANY JSON-Schema shapes CANNOT be
# represented on the delimiter-based XML wire without silently permitting
# schema-invalid or mis-typed output.
#
# WHY AN ALLOWLIST (closed positive set), NOT A BLACKLIST. JSON Schema has
# dozens of keywords; an opt-out-on-known-bad-keywords BLACKLIST is inherently
# incomplete — a reviewer can always name "one more" unsupported shape
# (``minProperties`` / ``propertyNames`` / ``contains`` / ``unevaluatedProperties``
# / a ``false`` schema / a ``$ref`` with siblings / …). We instead REQUIRE the
# schema to use EXCLUSIVELY an explicitly-enumerated, known-safe set of
# keywords/shapes: ``_xml_schema_representable`` returns ``True`` ONLY when every
# part of the schema falls inside that closed set; ANYTHING outside → opt out.
# This is complete BY CONSTRUCTION (no unknown keyword can be faithfully
# constrained, so any unknown keyword safely opts out) and ends the whack-a-mole.
# When a request contains ANY non-allowlisted shape on an XML-arg tool, the whole
# request OPTS OUT of grammar (``build_tool_grammar`` returns ``None`` ->
# free-form-then-parse, the safe existing path) rather than emit a lax grammar.
#
# THE ALLOWLIST (see ``_xml_schema_representable`` / ``_xml_property_representable``):
#   * TOP-LEVEL: a dict whose keys ⊆ ``_XML_ALLOWED_TOPLEVEL_KEYS``; ``type`` (if
#     present) == ``"object"``; ``additionalProperties`` (if present) is literally
#     ``False`` (a schema value OR ``True`` opts out — extra props can't be
#     constrained); ``set(required) ⊆ set(properties)``; a genuine no-arg tool
#     (no ``properties`` + no ``required``) stays representable (empty body).
#   * EACH PROPERTY (after resolving a single-level local ``$ref``): a non-empty
#     ``enum`` (alternation) whose keys are annotation-only, whose values are
#     type-consistent, and whose wire form is delimiter-safe
#     (``_xml_enum_representable`` — codex r3 #2/#3), OR ``type=="string"`` with keys ⊆
#     ``_XML_ALLOWED_STRING_KEYS`` (no ``const``/``pattern``/``minLength``/
#     ``maxLength``/``format``), OR ``type in {integer,number,boolean}`` (bare
#     terminal), OR ``type in {object,array}`` (``%json`` — llguidance's JSON
#     grammar handles the inner keywords), OR else opt out (a ``false``/``true``
#     bool schema, a ``null``/absent/union ``type``, an unresolved ``$ref``).
# F3 remains a REAL fix, not an opt-out: ``_emit_xml_param_value`` resolves a
# local ``$ref`` BEFORE the string-vs-``%json`` decision so a ``$ref``->string
# round-trips without quotes; only an UNRESOLVABLE ``$ref`` (or one with siblings)
# opts out.
#
# The closed positive sets. Membership is REQUIRED (not merely tolerated): any
# key outside these opts the request out.
_XML_ALLOWED_TOPLEVEL_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "description",
        "title",
        "$defs",
        "definitions",
        "additionalProperties",
    }
)
# Keys a ``type=="string"`` property may carry. Anything else (``const`` /
# ``pattern`` / ``minLength`` / ``maxLength`` / ``format`` / any unknown facet)
# cannot be enforced by the raw lazy value path -> opt out. This annotation-only
# set is ALSO the allowlist for an ENUM property's keys (``_xml_enum_representable``):
# a validation sibling (``minLength`` / ``pattern`` / …) next to an ``enum`` is
# not enforced by a bare literal alternation, so it opts out too.
_XML_ALLOWED_STRING_KEYS = frozenset(
    {
        "type",
        "enum",
        "description",
        "title",
        "default",
    }
)
# Property ``type`` values that render as a bare JSON terminal (the parser
# type-converts VALUE per int/float/bool paths).
_XML_SCALAR_TERMINAL_TYPES = frozenset({"integer", "number", "boolean"})
# Property ``type`` values that ride ``%json`` (llguidance's JSON grammar
# enforces the whole sub-schema, inner keywords included — do NOT recurse).
_XML_JSON_SUBSCHEMA_TYPES = frozenset({"object", "array"})
# Characters that would break out of the ``<parameter=KEY>`` delimiter wire (or
# collide with the JSON-ish surfaces the parser round-trips). OpenAI tool
# parameter names are normally ``[\w-]+``; be conservative.
_XML_UNSAFE_KEY_CHARS = frozenset("<>{},:")


def _collect_xml_defs(params: dict[str, Any]) -> dict[str, Any]:
    """Return the ``$defs`` / ``definitions`` containers present in ``params``."""
    defs: dict[str, Any] = {}
    for def_key in ("$defs", "definitions"):
        d = params.get(def_key)
        if isinstance(d, dict):
            defs[def_key] = d
    return defs


def _xml_key_is_delimiter_safe(key: Any) -> bool:
    """True iff ``key`` is safe to insert RAW into ``<parameter=KEY>`` (F5).

    Rejects a non-string / empty key, any ``< > { } : ,`` (which would desync the
    delimiter wire so the parser reads back a DIFFERENT key), and any whitespace
    (a newline would split the ``<parameter=KEY>`` header across the wire's own
    ``\\n`` framing).
    """
    if not isinstance(key, str) or not key:
        return False
    if any(c in _XML_UNSAFE_KEY_CHARS for c in key):
        return False
    return not any(c.isspace() for c in key)


# gemma4 emits a property key BARE in the ``KEY:`` wire position and the parser
# reads it back with ``GEMMA4_KEY_PATTERN`` (``\w+``), so only a ``\w+`` key
# round-trips faithfully. This is STRICTER than the XML key check (which allows
# ``-``/``.``): a ``my-key`` would be read back as just ``my`` (the parser stops
# at ``-``) and desync the pair. A leading digit is fine (``\w`` admits it).
_GEMMA4_KEY_RE = re.compile(r"\w+")


def _gemma4_key_is_safe(key: Any) -> bool:
    """True iff ``key`` round-trips through gemma4's BARE ``KEY:`` wire position."""
    return isinstance(key, str) and _GEMMA4_KEY_RE.fullmatch(key) is not None


def _xml_enum_wire_unsafe(wire: str) -> bool:
    """XML enum-value wire is unsafe iff it carries a delimiter-breaking byte.

    An XML string value truncates at the FIRST ``</parameter>`` and a ``<``/``>``
    desyncs the tag; a CR/LF splits the wire's own ``\\n`` framing.
    """
    return any(c in "<>\r\n" for c in wire)


# The FULL set of gemma4 structural special-token markers. Each is a SINGLE
# SPECIAL TOKEN on the gemma4 tokenizer (verified ids on
# ``mlx-community/gemma-4-e2b-it-4bit``): the tool-call frame ``<|tool_call>`` (48)
# / ``<tool_call|>`` (49), the string delimiter ``<|"|>`` (52), and the
# reasoning-channel frame ``<|channel>`` (100) / ``<channel|>`` (101). The first
# three mirror ``Gemma4ToolParser._GRAMMAR_SENTINELS`` — the parser is the single
# source of truth for the sentinel triple; the channel pair is documented there in
# prose (``TOOL_GRAMMAR_AUTO_SAFE`` comment) but has no constant, so it is
# enumerated here. Kept in THIS module (not imported from the parser) because the
# dependency direction is parsers -> tool_grammar; ``_GEMMA4_STRING_MARKER`` is
# reused so the ``<|"|>`` spelling has one definition.
_GEMMA4_STRUCTURAL_MARKERS = (
    "<|tool_call>",
    "<tool_call|>",
    _GEMMA4_STRING_MARKER,  # ``<|"|>``
    "<|channel>",
    "<channel|>",
)


def _gemma4_enum_wire_unsafe(wire: str) -> bool:
    """gemma4 enum-value wire is unsafe iff it embeds a STRUCTURAL special token.

    This is the STRUCTURAL, TOKENIZER-FREE fallback (the ``tokenizer is None``
    degraded / warmup path). The COMPLETE, complete-by-construction check lives in
    ``_gemma4_schema_representable`` (codex r3 E4): when the model tokenizer is
    available it ALSO rejects an enum value that tokenizes through ANY registered
    special token (``_enum_wire_embeds_special_token``), not just the five hard-coded
    markers below. This function stays a cheap fast-path there (belt & suspenders)
    and the sole check when no tokenizer is present.

    An enum value's WIRE form is rendered as BYTE LITERALS (a ``<|"|>``-wrapped
    string alt or a bare ``json.dumps`` scalar). Every gemma4 structural marker in
    ``_GEMMA4_STRUCTURAL_MARKERS`` — the ``<|"|>`` string delimiter AND the
    ``<|tool_call>``/``<tool_call|>``/``<|channel>``/``<channel|>`` frame tokens —
    is a SINGLE SPECIAL TOKEN, which llguidance emits atomically and NEVER as its
    spelled-out bytes. So a value whose wire form contains one compiles to a DEAD
    byte-literal alternation branch the model can never reach (the same
    runtime-unsatisfiable reason a byte-literal ``<|"|>`` delimiter cannot match
    the atomic token); were such a value ever to reach the wire it would also
    corrupt the parser's ``<|"|>`` boundary or the call frame. Faithful-or-opt-out:
    reject it so the whole request opts out (return ``None`` -> free-form),
    mirroring E3's ``_xml_enum_wire_unsafe``. Ordinary ``<``/``>``/CR/LF/commas/
    braces are SAFE inside the ``<|"|>`` pair (verified on the real tokenizer), so —
    unlike XML — a bare ``<``/``>`` must NOT opt out; only a FULL marker substring
    does.
    """
    return any(marker in wire for marker in _GEMMA4_STRUCTURAL_MARKERS)


def _enum_wire_embeds_special_token(tokenizer: Any, wire: str) -> bool:
    """True iff ``wire``'s BYTE content tokenizes THROUGH a registered special token.

    Completes the gemma4 enum guard BY CONSTRUCTION (codex r3 E4). The 5-marker
    ``_gemma4_enum_wire_unsafe`` substring BLACKLIST is inherently incomplete — a
    reviewer can always name one more special token (on the pinned gemma-4
    tokenizer ``<bos>`` / ``<eos>`` / ``<pad>`` / ``<unk>`` are single registered
    tokens outside the five markers). This check is TOKENIZER-DRIVEN, so it opts out
    a value ONLY when it ACTUALLY collapses to a registered token on THIS tokenizer
    (a surface like ``<start_of_turn>`` that tokenizes to ordinary multi-token text
    is a REACHABLE byte-literal and correctly kept). A gemma4 enum
    value is emitted as a BYTE-LITERAL alternation branch (the raw ``wire`` bytes
    between ``<|"|>`` markers, or a bare ``json.dumps`` scalar); if ANY of those
    bytes resolves to a REGISTERED added/special token on ``tokenizer``, llguidance
    emits that token ATOMICALLY, never as its spelled-out bytes, so the branch is
    unreachable/unsatisfiable and the model can never produce it. Report it unsafe
    so the whole request opts out (return ``None`` -> free-form) — the same
    faithful-or-opt-out policy as the structural-marker check, now closed over the
    tokenizer's FULL special-token set rather than five hard-coded strings.

    Reuses ``_is_registered_added_token`` — the EXACT predicate
    ``are_single_special_tokens`` uses — so "registered special token" means one
    consistent thing across the module (do NOT reinvent). Encodes with
    ``add_special_tokens=False`` so no BOS/EOS is auto-prepended; a value collapses
    to a special token ONLY when it literally spells one. Degrades to ``False`` (the
    cheap 5-marker structural fallback still applies) when the tokenizer lacks
    ``encode`` / raises — grammar constraint is best-effort opt-in, never a hard
    requirement.
    """
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return False
    try:
        ids = encode(wire, add_special_tokens=False)
    except Exception:
        return False
    return any(_is_registered_added_token(tokenizer, tok_id) for tok_id in ids)


@dataclass(frozen=True)
class _ArgWirePolicy:
    """Per-wire representability policy for the delimiter-based arg guards (E4).

    The strict-allowlist representability guard (``_arg_schema_representable`` /
    ``_arg_property_representable`` / ``_arg_enum_representable``) is IDENTICAL
    across the delimiter-based arg wires (E3 Qwen3-Coder XML, E4 gemma4) EXCEPT for
    two leaf checks that depend on the wire's concrete delimiters:

      * ``key_safe`` — whether a property key can be inserted RAW into the wire's
        key position (XML ``<parameter=KEY>`` rejects ``<>{},:``+whitespace; gemma4
        ``KEY:`` round-trips only a ``\\w+`` key).
      * ``enum_wire_unsafe`` — whether an enum value's WIRE form carries a byte
        sequence that breaks the value framing. XML: any ``<>\\r\\n`` byte. gemma4:
        ANY of the five structural special-token markers
        (``<|tool_call>`` / ``<tool_call|>`` / ``<|"|>`` / ``<|channel>`` /
        ``<channel|>``) AND — when ``build_tool_grammar`` threads the model
        tokenizer into ``_gemma4_schema_representable`` — ANY tokenizer-registered
        special token the value tokenizes through (``<bos>`` / ``<eos>`` /
        ``<pad>`` / …), so the gemma4 enum guard is COMPLETE BY CONSTRUCTION, not a
        five-string blacklist (codex r3 E4). Without a tokenizer it falls back to
        the five structural markers alone.

    Everything else (top-level key allowlist, ``additionalProperties`` /
    ``required`` totality, ``$ref`` resolution, scalar/object/array/string
    dispatch, string-facet allowlist) is wire-independent and SHARED — so the two
    families cannot drift into inconsistent (weaker) guards. ``_xml_*`` /
    ``_gemma4_*`` are thin wrappers that bind their policy.
    """

    key_safe: Callable[[Any], bool]
    enum_wire_unsafe: Callable[[str], bool]


_XML_WIRE_POLICY = _ArgWirePolicy(
    key_safe=_xml_key_is_delimiter_safe, enum_wire_unsafe=_xml_enum_wire_unsafe
)
_GEMMA4_WIRE_POLICY = _ArgWirePolicy(
    key_safe=_gemma4_key_is_safe, enum_wire_unsafe=_gemma4_enum_wire_unsafe
)


def _resolve_local_ref(subschema: Any, defs: dict[str, Any]) -> Any:
    """Resolve a single-level local ``$ref`` against ``defs`` (F3 / finding 4).

    ``defs`` is the ``{"$defs": {...}, "definitions": {...}}`` mapping
    ``_collect_xml_defs`` returns. Handles ONLY a single-level local pointer
    (``#/$defs/NAME`` or ``#/definitions/NAME``):

      * a schema WITHOUT a ``$ref`` is returned UNCHANGED (the caller decides its
        type as before);
      * a resolvable single-level local ``$ref`` (with NO sibling keys) returns
        the target dict;
      * a ``$ref`` object carrying SIBLING keys beyond ``$ref`` (e.g.
        ``{"$ref": "#/$defs/x", "enum": [...]}``) returns ``None`` — the sibling
        keywords would be SILENTLY DROPPED if we resolved to the bare target
        (finding 4), so we opt out rather than under-constrain. (Cleanly merging
        the siblings into the resolved target is possible but the safe choice is
        to opt out.)
      * an unresolvable / unsupported ``$ref`` (remote, non-local, multi-hop, or
        a missing/ non-dict target) returns ``None`` — the caller treats that as
        UNREPRESENTABLE and opts the request out.
    """
    if not isinstance(subschema, dict):
        return subschema
    ref = subschema.get("$ref")
    if ref is None:
        return subschema
    # Finding 4: a ``$ref`` alongside OTHER keys would drop those siblings on
    # resolution. Opt out (return ``None``) rather than silently ignore them.
    if len(subschema) != 1:
        return None
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    if len(parts) != 2 or parts[0] not in ("$defs", "definitions"):
        return None
    container = defs.get(parts[0])
    if not isinstance(container, dict):
        return None
    # JSON-pointer unescape (``~1`` -> ``/``, ``~0`` -> ``~``).
    name = parts[1].replace("~1", "/").replace("~0", "~")
    resolved = container.get(name)
    if not isinstance(resolved, dict):
        return None
    return resolved


def _enum_value_matches_declared_type(value: Any, declared: str) -> bool:
    """True iff ``value``'s JSON type is consistent with a declared scalar ``type``.

    Maps the four JSON-Schema scalar types to their Python instance test
    (string→``str``, integer→``int``, number→``int``|``float``, boolean→``bool``),
    excluding ``bool`` from the numeric types (a Python ``bool`` is an ``int``
    subclass). ANY other declared type — ``object`` / ``array`` / ``null`` / an
    unknown string — carrying an enum returns ``False`` so the caller opts out:
    the delimiter-based XML value wire cannot faithfully constrain a non-scalar
    enum.
    """
    t = declared.strip().lower()
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    return False


def _xml_enum_representable(
    resolved: dict[str, Any], enum: list[Any], policy: _ArgWirePolicy = _XML_WIRE_POLICY
) -> bool:
    """True iff an ENUM property is inside the emitter's closed allowlist.

    Shared across the delimiter-based arg wires; ``policy`` selects the concrete
    wire's enum-value delimiter check (``policy.enum_wire_unsafe`` — XML by
    default, gemma4 via ``_GEMMA4_WIRE_POLICY``). The XML-specific prose below
    describes the default policy.

    Routes the enum-first branch THROUGH the allowlist (codex r3 #2/#3) so the
    enum axis is complete-by-construction like the rest of the guard, instead of
    blanket-accepting ANY non-empty ``enum``. Representable ONLY when ALL hold:

      * keys ⊆ the annotation-only set (``_XML_ALLOWED_STRING_KEYS`` =
        ``{type, enum, description, title, default}``): any VALIDATION sibling
        (``minLength`` / ``pattern`` / ``minItems`` / ``const`` / a size facet /
        any unknown key) is NOT enforced by a bare literal alternation, so it
        opts out rather than being silently ignored (codex r3 #2 —
        ``{"enum": ["a", "bb"], "minLength": 2}``);
      * if ``type`` is present it must be a STRING and EVERY enum value's JSON
        type must be consistent with it (codex r3 #2 —
        ``{"type": "integer", "enum": ["x"]}`` opts out on the value/type
        mismatch; an ``object``/``array``/``null`` declared type opts out);
      * NO enum value's WIRE form carries a delimiter-breaking byte (``<`` / ``>``
        / CR / LF). The wire form mirrors ``_emit_xml_param_value`` EXACTLY (the
        raw text for a string value, ``json.dumps`` otherwise), so a value such as
        ``"a</parameter>b"`` — grammar-accepted but truncated at the parser's
        FIRST ``</parameter>`` — is rejected here rather than emitted (codex r3
        #3).
    """
    # (a) annotation-only keys: a validation sibling cannot be enforced next to a
    # bare literal alternation, so its presence must opt out (never be dropped).
    if not set(resolved.keys()) <= _XML_ALLOWED_STRING_KEYS:
        return False
    declared = resolved.get("type")
    if declared is not None and not isinstance(declared, str):
        # A non-string ``type`` (union list / bool / …) alongside an enum is not a
        # shape we can verify -> opt out.
        return False
    for value in enum:
        # (b) value/type consistency when a scalar ``type`` is declared.
        if isinstance(declared, str) and not _enum_value_matches_declared_type(
            value, declared
        ):
            return False
        # (c) delimiter safety on the EXACT wire form the emitter would produce
        # (raw for a string value, ``json.dumps`` otherwise — enum values come
        # from parsed client JSON, so ``json.dumps`` is total and never raises).
        # The concrete unsafe-byte set is the wire's (``policy.enum_wire_unsafe``).
        wire = value if isinstance(value, str) else json.dumps(value)
        if policy.enum_wire_unsafe(wire):
            return False
    return True


def _xml_property_representable(
    subschema: Any, defs: dict[str, Any], policy: _ArgWirePolicy = _XML_WIRE_POLICY
) -> bool:
    """True iff ONE property schema is inside the emitter's closed allowlist.

    Shared across the delimiter-based arg wires; ``policy`` threads to the enum
    check (XML default, gemma4 via ``_GEMMA4_WIRE_POLICY``). The prose below
    describes the default XML policy.

    After resolving a single-level local ``$ref`` (``_resolve_local_ref`` — an
    unresolvable ``$ref``, a ``$ref`` with siblings, or a non-dict such as a
    ``false``/``true`` bool schema returns ``None`` -> opt out), the resolved
    schema is representable ONLY as one of the enumerated shapes:

      * a non-empty ``enum`` list — rendered as a literal ALTERNATION, admitted
        ONLY through ``_xml_enum_representable`` (annotation-only keys, value/
        declared-type consistency, delimiter-safe wire values — codex r3 #2/#3), OR
      * ``type == "string"`` whose keys ⊆ ``_XML_ALLOWED_STRING_KEYS`` — the raw
        lazy value path (a ``const``/``pattern``/``minLength``/``maxLength``/
        ``format`` or ANY unknown facet it cannot enforce opts out), OR
      * ``type in {integer,number,boolean}`` — a bare JSON terminal, OR
      * ``type in {object,array}`` — ``%json`` over the sub-schema (llguidance's
        JSON grammar enforces the inner keywords; we do NOT recurse/restrict), OR
      * else opt out (a ``null``/absent/union/non-string ``type``).
    """
    resolved = _resolve_local_ref(subschema, defs)
    if not isinstance(resolved, dict):
        # Unresolvable/sibling ``$ref`` (None) or a ``false``/``true`` bool schema
        # / non-dict -> not representable.
        return False
    # Enum-first: a non-empty enum renders as an alternation of literals. Route it
    # THROUGH the allowlist (codex r3 #2/#3) rather than blanket-accepting any
    # non-empty enum — validate annotation-only keys, value/declared-type
    # consistency, and delimiter-safe wire values, so the enum axis is
    # complete-by-construction too.
    enum = resolved.get("enum")
    if isinstance(enum, list):
        if not enum:
            # An EMPTY enum is UNSATISFIABLE under JSON Schema (no value validates).
            # Falling through to the type-based path would compile an unrestricted
            # value and ADMIT schema-invalid output (codex r6 #3). Opt out instead —
            # faithful-or-opt-out. SHARED guard, so this also closes the latent same
            # gap on the E3 XML wire; opting out only ever WIDENS the free-form
            # fallback, so it cannot break a previously-valid grammar.
            return False
        return _xml_enum_representable(resolved, enum, policy)
    prop_type = resolved.get("type")
    if not isinstance(prop_type, str):
        # Absent / union (list) / non-string ``type`` -> opt out.
        return False
    t = prop_type.strip().lower()
    if t == "string":
        # Only annotation keys may accompany a string; any facet the raw lazy
        # value path cannot enforce (or an unknown key) opts out.
        return set(resolved.keys()) <= _XML_ALLOWED_STRING_KEYS
    if t in _XML_SCALAR_TERMINAL_TYPES:
        return True
    if t in _XML_JSON_SUBSCHEMA_TYPES:
        return True
    # ``null`` or any unknown type -> opt out.
    return False


def _xml_schema_representable(
    params: Any, policy: _ArgWirePolicy = _XML_WIRE_POLICY
) -> bool:
    """True iff a delimiter-arg emitter can FAITHFULLY constrain ``params``.

    Shared strict-allowlist guard for the delimiter-based arg wires (#558 E3 XML,
    E4 gemma4). ``policy`` (default ``_XML_WIRE_POLICY``) selects the two
    wire-specific leaf checks — key safety and enum-value delimiter safety — while
    every structural rule (top-level keys, ``additionalProperties`` / ``required``
    totality, ``$ref``, per-property dispatch) is shared, so no family can drift to
    a weaker guard. ``_gemma4_schema_representable`` binds the gemma4 policy; the
    prose below describes the default XML policy (#558 E3).

    STRICT ALLOWLIST (closed positive set — see the module block above): returns
    ``True`` ONLY when every part of ``params`` uses exclusively a known-safe,
    explicitly-enumerated keyword/shape; ANYTHING outside opts the request out
    (``False`` -> free-form fallback). Complete by construction, so no unknown
    JSON-Schema keyword can slip a lax grammar through.

    Representable ``True`` iff ALL hold:
      * ``params`` is a dict whose keys ⊆ ``_XML_ALLOWED_TOPLEVEL_KEYS``;
      * ``type`` (if present) is ``"object"``;
      * ``additionalProperties`` (if present) is literally ``False`` (a schema
        value or ``True`` opts out — extra props can't be constrained);
      * ``properties`` (if present) is a dict;
      * ``set(required) ⊆ set(properties)`` (a required name with no declared
        property opts out — finding 3);
      * every property key is delimiter-safe (``_xml_key_is_delimiter_safe``, F5)
        AND every property schema is in the per-property allowlist
        (``_xml_property_representable``).
    A genuine no-argument tool (no ``properties`` + no ``required``) is
    representable — its empty body IS the faithful representation.
    """
    # A ``false``/``true`` bool schema — or any non-dict — cannot be rendered as
    # an XML parameter body; opt out.
    if not isinstance(params, dict):
        return False
    # CLOSED top-level key set: any key outside the allowlist (``minProperties`` /
    # ``maxProperties`` / ``propertyNames`` / ``patternProperties`` /
    # ``dependentRequired`` / ``dependencies`` / ``if`` / ``then`` / ``else`` /
    # ``not`` / ``allOf`` / ``anyOf`` / ``oneOf`` / ``contains`` /
    # ``unevaluatedProperties`` / a top-level ``$ref`` / … ) opts out. This single
    # check subsumes the former object-keyword + composition + top-level-``$ref``
    # blacklists — completely, since membership is required not merely tolerated.
    if not set(params.keys()) <= _XML_ALLOWED_TOPLEVEL_KEYS:
        return False
    # ``type``, if present, must be exactly ``object`` (a non-object top-level
    # type has no XML parameter-body representation).
    top_type = params.get("type")
    if top_type is not None:
        if not isinstance(top_type, str) or top_type.strip().lower() != "object":
            return False
    # ``additionalProperties``: absent OK; literal ``False`` OK; a schema value or
    # ``True`` opts out (the XML wire cannot constrain undeclared extra props).
    if "additionalProperties" in params and params["additionalProperties"] is not False:
        return False
    props = params.get("properties")
    if props is not None and not isinstance(props, dict):
        return False
    prop_map: dict[str, Any] = props if isinstance(props, dict) else {}
    # ``required`` must be a list naming ONLY declared properties (finding 3) — an
    # undeclared required name would never be emitted, silently unenforced. This
    # also opts out the property-less-but-``required`` case (empty body wrong).
    required = params.get("required")
    if required is not None:
        # TOTAL guard (codex r3 #4): ``required`` must be a list of STRINGS. A
        # non-list, or ANY non-str member (a client-supplied list/dict is
        # UNHASHABLE), would otherwise make ``set(required)`` below raise
        # ``TypeError`` — and this guard runs OUTSIDE the builder's exception
        # handling, so an arbitrary-client-JSON ``required`` would 500. Validate
        # the shape first and opt out gracefully; only then is ``set(required)``
        # provably safe. (The guard must NEVER raise on arbitrary client JSON.)
        if not isinstance(required, list) or not all(
            isinstance(r, str) for r in required
        ):
            return False
        if not set(required) <= set(prop_map.keys()):
            return False
    # Validate each TOP-LEVEL property against the closed per-property allowlist
    # (nested object/array schemas ride ``%json`` and are enforced there, so only
    # the top level needs the key-safety + shape guard). An empty ``prop_map`` (no
    # arguments) skips the loop and stays representable.
    defs = _collect_xml_defs(params)
    for key, subschema in prop_map.items():
        # The key is emitted RAW into the wire's key position -> must be safe for
        # THIS wire (XML ``<parameter=KEY>`` vs gemma4 bare ``KEY:``).
        if not policy.key_safe(key):
            return False
        if not _xml_property_representable(subschema, defs, policy):
            return False
    return True


def _gemma4_schema_representable(params: Any, *, tokenizer: Any = None) -> bool:
    """True iff the gemma4 arg emitter can FAITHFULLY constrain ``params`` (E4).

    Thin binding of the shared strict-allowlist guard to the gemma4 wire policy
    (``\\w+`` bare keys; an enum value is unsafe iff it embeds a structural marker
    OR — when ``tokenizer`` is given — ANY registered special token). Reuses the
    SAME structural allowlist as XML — a request opts out of grammar
    (``build_tool_grammar`` -> ``None`` -> free-form) on any unrepresentable shape.

    ``tokenizer`` (the model tokenizer, read off the parser by
    ``build_tool_grammar``) makes the enum guard COMPLETE BY CONSTRUCTION (codex r3
    E4): the five-marker ``_gemma4_enum_wire_unsafe`` substring blacklist cannot
    enumerate every special token, so when a tokenizer is available the per-request
    policy ALSO rejects any enum value that tokenizes through a registered special
    token (``_enum_wire_embeds_special_token``) — an enum such as ``["<bos>"]`` /
    ``["<eos>"]`` compiles to an unreachable byte-literal branch and opts the
    request out. When ``tokenizer is None`` (degraded / warmup, which uses a
    fixed NO-enum tool) the guard falls back to the five-marker structural subset —
    a safe under-approximation that never bites the warmup path. The cheap 5-marker
    check runs FIRST inside the composed check (belt & suspenders) even when a
    tokenizer is present.
    """
    if tokenizer is None:
        policy = _GEMMA4_WIRE_POLICY
    else:

        def _enum_wire_unsafe(wire: str) -> bool:
            # Complete-by-construction: the cheap structural 5-marker check FIRST,
            # then the tokenizer-complete registered-special-token check.
            return _gemma4_enum_wire_unsafe(wire) or _enum_wire_embeds_special_token(
                tokenizer, wire
            )

        policy = replace(_GEMMA4_WIRE_POLICY, enum_wire_unsafe=_enum_wire_unsafe)
    return _xml_schema_representable(params, policy)


def _emit_xml_param_value(subschema: Any, defs: dict[str, Any]) -> str:
    """Return the Lark for one XML parameter's VALUE plus its closing literal.

    The Qwen3-Coder wire is ``<parameter=KEY>\\nVALUE\\n</parameter>\\n`` and the
    ``qwen3_coder`` parser type-converts VALUE per the tool schema, so the
    grammar must emit each value in the SAME surface form the parser round-trips:

      * ENUM -> an alternation of the literal enum values (raw string form for
        string enums, JSON scalar for numeric/bool enums), then the
        ``\\n</parameter>\\n`` close.
      * STRING (non-enum) -> the LAZY ``xml_param_value`` rule (NOT ``%json``,
        whose surrounding quotes the parser keeps verbatim, yielding
        ``"\\"Paris\\""``). The lazy rule matches ANY text (``<`` included) up to
        AND INCLUDING the first ``</parameter>`` — so it absorbs the value, the
        ``\\n`` before ``</parameter>``, AND the ``</parameter>`` tag itself. The
        close appended here is therefore just the trailing ``\\n`` separator (NO
        ``</parameter>`` — the rule already consumed it). This is what lets a
        ``<``-containing code arg (``a < b``, ``vector<int>``) round-trip instead
        of truncating at the first ``<`` (the pilot's ``/[^<]*/`` bug).
      * EVERYTHING ELSE (number / integer / boolean / object / array / null) ->
        JSON-constrained via ``%json`` (these surface forms match the parser's
        int / float / bool / ``json.loads`` paths), then the ``\\n</parameter>\\n``
        close (leading newline preserved — ``%json`` does not consume it).

    ``defs`` (the parent schema's ``$defs`` / ``definitions``) is merged into each
    ``%json`` value sub-schema so an internal ``$ref`` still resolves.
    """
    close_json = _lark_escape("\n</parameter>\n")
    # The lazy rule already consumed ``\n</parameter>``; only the trailing
    # separator newline (AFTER the close tag) remains for the string case.
    close_lazy = _lark_escape("\n")
    if not isinstance(subschema, dict):
        return f"{_XML_STRING_VALUE_RULE} {close_lazy}"
    # F3 (#558 E3 codex converge): resolve a single-level local ``$ref`` BEFORE
    # the enum/string-vs-``%json`` decision, so a ``$ref``->string takes the
    # RAW-string (lazy) path and round-trips WITHOUT surrounding quotes — the
    # pilot attached ``$defs`` only AFTER this decision, so a ``$ref``->string
    # fell through to ``%json`` and emitted QUOTED JSON the parser returned with
    # quotes preserved (``"\\"Paris\\""``). A ``$ref``->object still takes the
    # ``%json`` path below (over the ORIGINAL ``$ref`` + merged ``$defs``, which
    # llguidance resolves internally). ``_resolve_local_ref`` returns the schema
    # unchanged when there is no ``$ref``; a ``None`` (unresolvable) is defensive
    # — the representability guard already opted such a request out — and falls
    # back to the ``%json`` path over the original.
    resolved = _resolve_local_ref(subschema, defs)
    if resolved is None:
        resolved = subschema
    enum = resolved.get("enum")
    if isinstance(enum, list) and enum:
        # Every value's wire form here is already delimiter-safe and type-consistent:
        # ``_xml_enum_representable`` (in the representability guard) opted the whole
        # request out otherwise (codex r3 #2/#3), so this raw emission cannot break
        # the ``</parameter>`` framing. Keep the wire form byte-identical to that
        # guard's check (raw for a string value, ``json.dumps`` otherwise).
        alts = []
        for value in enum:
            literal = value if isinstance(value, str) else json.dumps(value)
            alts.append(_lark_escape(literal))
        return f"({' | '.join(alts)}) {close_json}"
    # Lazy import: keep this module importable independently of api.tool_calling
    # (no cycle today, but the lazy import future-proofs it).
    from .tool_calling import _schema_type

    if _schema_type(resolved) == "string":
        return f"{_XML_STRING_VALUE_RULE} {close_lazy}"
    value_schema = dict(subschema)
    for def_key, def_val in defs.items():
        value_schema.setdefault(def_key, def_val)
    return f"%json {json.dumps(value_schema)} {close_json}"


def _emit_xml_arg_body(params: Any) -> str:
    """Emit the Lark for a Qwen3-Coder XML argument body from a JSON Schema.

    One property renders as ``<parameter=KEY>\\n<value>\\n</parameter>\\n``.
    REQUIRED properties are mandatory; the rest are OPTIONAL (``( ... )?``).
    Properties are emitted in schema order — any-order permutation is DEFERRED
    (a forced grammar makes the model follow this order, which real Qwen3-Coder
    already does for its own schemas). Returns ``""`` for a no-argument tool
    (empty / absent ``properties``), which the caller renders as the bare
    ``<function=NAME>\\n</function>`` frame.
    """
    if not isinstance(params, dict):
        return ""
    props = params.get("properties")
    if not isinstance(props, dict) or not props:
        # INTENTIONAL tool-calling-domain choice (codex r3 #1), NOT an oversight:
        # a tool whose ``parameters`` declares no properties takes NO arguments, so
        # the EMPTY body IS the correct, desired constraint — it forces a no-arg
        # call. JSON-Schema's default-OPEN-object semantics (where ``{}`` /
        # ``{"type": "object"}`` permit arbitrary properties) are DELIBERATELY not
        # applied on the tool-call wire: opting out here would leave a no-arg tool
        # LESS constrained (the model could emit arbitrary ``<parameter=…>``
        # blocks). The genuinely-OPEN case — an EXPLICIT ``additionalProperties:
        # true`` (or a schema value) — is already opted out UPSTREAM by
        # ``_xml_schema_representable`` (only a literal ``additionalProperties:
        # false`` / absent reaches here), so a truly-open object never collapses to
        # an empty body while the common no-arg shape stays fully constrained.
        return ""
    required = params.get("required")
    # Only STRING members can name a property (mirrors the total guard in
    # ``_xml_schema_representable`` — codex r3 #4); a non-str member is unhashable
    # and would raise in a bare ``set(required)``. The representability guard has
    # already opted out any malformed ``required`` before we reach here, but stay
    # total regardless of caller.
    required_set = (
        {r for r in required if isinstance(r, str)}
        if isinstance(required, list)
        else set()
    )
    defs = _collect_xml_defs(params)
    frags: list[str] = []
    for key, subschema in props.items():
        open_lit = _lark_escape(f"<parameter={key}>\n")
        value_lark = _emit_xml_param_value(subschema, defs)
        block = f"{open_lit} {value_lark}"
        frags.append(block if key in required_set else f"( {block} )?")
    return " ".join(frags)


def _emit_gemma4_param_value(subschema: Any, defs: dict[str, Any]) -> str:
    """Return the Lark for ONE gemma4 property VALUE (no key, no comma).

    The gemma4 wire (``chat_template.jinja``'s ``format_argument``) serialises a
    value by JSON type:

      * ENUM -> a literal ALTERNATION, each value rendered in its OWN wire form:
        a STRING value is wrapped in the ``<|"|>`` marker
        (``<|"|> "python" <|"|>``), a numeric/boolean value is BARE
        (``json.dumps`` -> ``5`` / ``true``). Per-value wrapping (not an
        all-or-nothing split) faithfully renders a mixed-type enum too.
      * STRING (non-enum) -> the ``gemma_str_value`` rule
        (``<|"|> GEMMA_STR_TEXT <|"|>`` — any-byte content EXCEPT the byte spelling
        of the ``<|"|>`` marker, closed by the atomic special-token marker; see
        ``_GEMMA4_STRING_RULE_DECL``).
      * EVERYTHING ELSE (integer / number / boolean / object / array) -> ``%json``
        over the sub-schema. ``format_argument`` emits a BARE JSON scalar for
        numbers/bools (``3`` / ``true``) and a JSON container for object/array,
        all of which ``%json`` produces and the lenient gemma4 parser
        (``_Gemma4ArgumentParser``) round-trips (its ``_parse_value`` accepts a
        bare JSON object/array and a JSON string, so the standard-JSON surface
        ``%json`` emits parses back correctly).

    ``defs`` is merged into the ``%json`` sub-schema so an internal ``$ref``
    resolves (mirrors ``_emit_xml_param_value``). ``_resolve_local_ref`` is applied
    FIRST so a ``$ref`` -> string takes the raw ``<|"|>`` path (not quoted
    ``%json``); a ``None`` (unresolvable) is defensive — the representability guard
    already opted such a request out — and falls back to ``%json``.
    """
    if not isinstance(subschema, dict):
        return _GEMMA4_STRING_VALUE_RULE
    resolved = _resolve_local_ref(subschema, defs)
    if resolved is None:
        resolved = subschema
    enum = resolved.get("enum")
    if isinstance(enum, list) and enum:
        # Every value's wire form is already delimiter-safe (the representability
        # guard opted the request out otherwise). Wrap each STRING value in the
        # ``<|"|>`` marker, emit every other value BARE via ``json.dumps``.
        alts = []
        for value in enum:
            if isinstance(value, str):
                alts.append(
                    f"{_GEMMA4_STRING_MARKER} {_lark_escape(value)} "
                    f"{_GEMMA4_STRING_MARKER}"
                )
            else:
                alts.append(_lark_escape(json.dumps(value)))
        return f"({' | '.join(alts)})"
    # Lazy import: keep this module importable independently of api.tool_calling.
    from .tool_calling import _schema_type

    if _schema_type(resolved) == "string":
        return _GEMMA4_STRING_VALUE_RULE
    value_schema = dict(subschema)
    for def_key, def_val in defs.items():
        value_schema.setdefault(def_key, def_val)
    return f"%json {json.dumps(value_schema)}"


def _emit_gemma4_arg_body(params: Any, rule_prefix: str) -> tuple[str, str]:
    """Emit the Lark for a gemma4 arg body (the part BETWEEN ``{`` and ``}``).

    Returns ``(inline_body, extra_rules)``: ``inline_body`` is spliced into the
    ``tag_i`` rule (``""`` for a no-argument tool), and ``extra_rules`` is the block
    of generated ``<rule_prefix>_rest<i>`` nonterminal definitions that ``inline_body``
    references (the caller appends it to the grammar; ``""`` when none are needed).

    The gemma4 wire is ``call:NAME{k1:v1,k2:v2,...}`` where keys are emitted in
    DICTSORT order and separated by COMMAS (``chat_template.jinja``'s ``found_first``
    logic — a comma precedes every pair except the FIRST PRESENT one). Required
    properties are mandatory; optional ones may be omitted.

    Because commas are SEPARATORS (not per-field terminators), a naive per-field
    ``( )?`` misplaces the comma when an early optional field is absent (it would
    emit a leading ``,`` or a doubled ``,,``). We build a FIRST-PRESENT construction
    whose comma-prefixed SUFFIXES are SHARED via named nonterminals, so the grammar
    is O(n)-size. (An earlier inline version regenerated the entire remaining suffix
    for EVERY first-present alternative, making a request-controlled all-optional
    schema O(n^2) in grammar size AND construction time — a mild DoS amplifier.)

      * ``<rule_prefix>_rest<i>`` — a NAMED nonterminal for "fields ``i..n-1``, each
        as a LEADING-comma continuation" (required: ``"," k:v``; optional:
        ``( "," k:v )?``). ``rest_i`` ENDS in a reference to ``rest_{i+1}`` (right
        recursion), so the whole suffix chain is ``n-1`` rules of O(1) size — NOT
        O(n) inline text copied into each alternative. Used AFTER the first present
        field, so a comma always separates two present pairs. The final field has an
        empty tail, so there is no ``rest_n`` rule.
      * the body is an ALTERNATION over WHICH field is first-present: for each
        candidate first field ``j`` (reachable only if every earlier field is
        optional, i.e. ``j`` is at or before the first REQUIRED field) emit
        ``kj:vj`` with NO leading comma, then a REFERENCE to ``rest_{j+1}``. This
        alternation is emitted ONCE (O(n) total), not per-suffix.
      * if EVERY field is optional the whole body is additionally ``( ... )?`` so
        an empty ``{}`` body (no args emitted) is admitted.

    The whole body is wrapped in a single ``( ... )`` group so an alternation binds
    correctly when embedded as ``... "{" <body> "}" ...``. Returns ``("", "")`` for a
    no-argument tool (empty/absent ``properties``) -> the caller renders the bare
    ``call:NAME{}`` frame.

    The accepted LANGUAGE is IDENTICAL to the previous inline construction — any
    subset of the optionals in dictsort order, required fields mandatory, no leading/
    trailing/doubled comma, empty ``{}`` iff all-optional — only the grammar SIZE
    drops from O(n^2) to O(n).
    """
    if not isinstance(params, dict):
        return "", ""
    props = params.get("properties")
    if not isinstance(props, dict) or not props:
        # A no-argument tool: the EMPTY body IS the faithful constraint (forces a
        # ``{}`` call). Mirrors ``_emit_xml_arg_body`` — a truly-open object is
        # already opted out upstream by ``_gemma4_schema_representable``. This is a
        # deliberate TOOL-CALLING-DOMAIN choice, consistent with the E3 XML path's
        # identical decision (#1170): a tool whose ``parameters`` declares no
        # properties takes NO arguments, so forcing ``call:NAME{}`` is the desired
        # constraint. JSON-Schema's default-open-object semantics (an empty/omitted
        # ``properties`` "permits arbitrary properties") are DELIBERATELY not applied
        # on the wire — opting out here would make a no-arg tool LESS constrained
        # (free-form body), the opposite of what tool-calling wants.
        return "", ""
    required = params.get("required")
    # Only STRING members can name a property (mirrors the total guard in the
    # representability check); a non-str member is unhashable in a bare ``set``.
    required_set = (
        {r for r in required if isinstance(r, str)}
        if isinstance(required, list)
        else set()
    )
    defs = _collect_xml_defs(params)
    # DICTSORT order — matches how the chat template renders arguments (``properties
    # | dictsort``), so a forced grammar constrains the model to the ordering it was
    # trained to emit. Jinja's ``dictsort`` defaults to ``case_sensitive=False``, so
    # the sort key is the LOWERCASED key; Python's ``sorted`` is STABLE, so a
    # case-insensitive collision (``a`` vs ``A``) keeps the schema's insertion order
    # — EXACTLY the tiebreak ``dictsort`` uses. (A secondary case-sensitive sort would
    # force ``A`` before ``a``, diverging from the model's actual emission order.)
    keys = sorted(props.keys(), key=lambda k: k.lower())
    fields = [
        (key, _emit_gemma4_param_value(props[key], defs), key in required_set)
        for key in keys
    ]
    n = len(fields)

    def _kv(idx: int, *, leading_comma: bool) -> str:
        key, value_lark, _ = fields[idx]
        kv = f"{_lark_escape(key + ':')} {value_lark}"
        return f'"," {kv}' if leading_comma else kv

    def _rest_ref(i: int) -> str:
        # Reference to the suffix nonterminal for fields ``i..n-1``; empty past the
        # last field (no ``rest_n`` rule exists).
        return f"{rule_prefix}_rest{i}" if i < n else ""

    # Emit each comma-prefixed SUFFIX exactly ONCE as a named nonterminal, chained by
    # right recursion (``rest_i -> elem_i rest_{i+1}``) for an O(n)-size grammar.
    rest_rules: list[str] = []
    for i in range(1, n):
        frag = _kv(i, leading_comma=True)
        elem = frag if fields[i][2] else f"( {frag} )?"
        rest_rules.append(f"{rule_prefix}_rest{i}: {elem} {_rest_ref(i + 1)}".rstrip())

    # Index of the first REQUIRED field (or ``n`` if all optional): the first-present
    # field can be any field at or before it (earlier fields, being optional, may be
    # absent — a required field can never be skipped).
    first_required = n
    for j, (_key, _val, req) in enumerate(fields):
        if req:
            first_required = j
            break

    alts: list[str] = []
    for j in range(min(first_required + 1, n)):
        head = _kv(j, leading_comma=False)
        tail = _rest_ref(j + 1)
        alts.append(f"{head} {tail}" if tail else head)
    body = " | ".join(alts)
    inline = f"( {body} )?" if first_required == n else f"( {body} )"
    extra_rules = ("\n".join(rest_rules) + "\n") if rest_rules else ""
    return inline, extra_rules


def build_tool_lark(
    tools: list[dict[str, Any]],
    tool_choice: str,
    structure_infos: list["StructureInfo"],
    *,
    single_call: bool = False,
    reasoning_sentinels: tuple[str, ...] = (),
) -> str:
    """Assemble the Lark grammar for a set of per-tool structure triples.

    Ports the ``StructTag.to_grammar`` layout (``start: (tag_0|...)*
    tag_end``) but with special-token-aware begin/end rendering (see module
    docstring). ``tool_choice`` × ``single_call`` select the repetition
    quantifier:

      * ``auto``                        -> ``(...)*``  (may emit zero calls)
      * ``auto`` + ``single_call``      -> ``(...)?``  (ZERO-OR-ONE call)
      * ``required`` / a function name  -> ``(...)+``  (design R1: ≥1 forced)
      * ``required`` + ``single_call``  -> ``(...)``   (EXACTLY ONE call)

    ``single_call=True`` (OpenAI ``parallel_tool_calls=False``) means "if the
    model calls a tool, at most ONE call". For ``required``/named that is
    EXACTLY ONE (no repetition quantifier). For ``auto`` — which may always emit
    ZERO calls — it is ZERO-OR-ONE (``?``): auto's "may call zero" combined with
    no-parallel's "at most one" (codex #558-PR5). Auto with ``single_call``
    therefore drops the ``*`` down to ``?`` rather than staying zero-or-more —
    honouring the client's explicit parallel cap on the auto path too. Without
    this cap the ``required`` grammar could emit multiple calls, and the ``auto``
    grammar could emit ≥2 calls, even when the client disabled parallel calls
    (codex #558-PR3 / PR-5).

    ``"none"`` must NOT reach this function — ``none`` produces no grammar at
    all (the model sees no tools, design §4); passing it here is a caller
    bug and raises ``ValueError`` rather than silently forcing a call.

    The grammar is built over exactly the ``tools`` passed in — one ``tag_i``
    alternative per tool. For a NAMED ``tool_choice`` the caller narrows
    ``tools`` to the single requested function BEFORE calling the builder
    (design §4 / chat.py named routing), so the alternation naturally
    collapses to a single forced tag. The builder does not itself resolve a
    function name out of a multi-tool list — that routing lands in PR-3.

    Every tag is: ``<free-prefix> <trigger-and-begin> %json <schema> <end>``.
    The free prefix is the lazy region that swallows reasoning/prose until the
    trigger — this is the reasoning-aware delay (design §5 path A).
    REQUIREMENT (applies to ALL modes, not just ``auto``): the trigger MUST be
    a single special token, declared in ``sentinels``. This is enforced
    UNCONDITIONALLY because the lazy free prefix can reassemble a
    multi-byte *text* trigger from ordinary token pieces in any mode — a
    text trigger is unenforceable regardless of the ``*``/``+`` quantifier, so
    the builder rejects it (raises ``ValueError``) rather than silently
    producing an unenforceable grammar. Full text-trigger support (excluding
    the trigger byte sequence from the free prefix across token boundaries) is
    design §7 open-Q1, deferred to the PR-5 auto path.

    REASONING-TOLERANT PREFIX (#558 PR-4, path A). ``reasoning_sentinels`` is an
    ORDERED ``(open, close)`` pair of the model family's reasoning-boundary
    strings (``<think>`` / ``</think>``) that the caller has PROVEN are single
    special tokens on this tokenizer (via ``are_single_special_tokens``). GROUND
    TRUTH (verified on the real Qwen3.6 tokenizer, 2026-07): a bare
    ``TAG_TEXT: /(.|\\n)*/`` free prefix is a BYTE regex and therefore CANNOT
    match a ``<think>`` *special token* (id 248068) — the design-doc §5.3 claim
    that a bare ``TAG_TEXT`` "swallows the whole ``<think>...</think>`` block" is
    FALSE whenever those markers are special tokens (exactly the reasoning case
    this PR targets). The matcher rejects the very first ``<think>`` token and
    path A collapses. To make path A actually correct we build the prefix from
    two rule flavours (refs MUST be rule-level, not lexer terminals — llguidance
    rejects special tokens inside a terminal):

      * ``lead: opened? bal_prefix`` — consumed ONCE at ``start: lead ...``.
        ``opened: TAG_TEXT <close>`` permits a SINGLE optional leading close.
      * ``bal_prefix: TAG_TEXT (reasoning_block TAG_TEXT)*`` with
        ``reasoning_block: <open> TAG_TEXT <close>`` — BALANCED-only, reused by
        every ``tag_i`` and the trailing ``tag_end``.

    The block is BALANCED (codex #558-PR4 round-3): a mid-stream ``<think>``
    opener MUST be closed by ``</think>`` before the trigger. An
    unbalanced-tolerant ``rtok: <think> | </think>`` would accept an UNCLOSED
    ``<think>...<tool_call>...</tool_call>`` that a lenient reasoning parser
    could classify entirely as reasoning, letting the tool call be swallowed and
    ``tool_choice="required"`` slip.

    The ``opened?`` leading-close handles the PREFILLED-``<think>`` case (codex
    #558-PR4 round-4): many reasoning chat templates prefill ``<think>`` at the
    END of the prompt (verified on Qwen3.6 and DeepSeek-R1 — the assistant turn
    ends ``...<think>``), so GENERATION begins already inside reasoning. The
    grammar processor baselines past the prompt and only sees the GENERATED
    tokens, which then begin with reasoning text and a ``</think>`` whose opener
    lives in the (unseen) prompt. Without it the balanced block would reject that
    leading ``</think>`` and BLOCK the required call on every prefilled-think
    model. Because the leading close lives ONLY in the one-time ``lead`` rule
    (every ``tag_i``/``tag_end`` uses the balanced-only ``bal_prefix``), the
    tolerance is GLOBALLY at-most-one: a stray ``</think>`` before a LATER call
    or after the final call is rejected (codex #558-PR4 round-5 — a prefix that
    reused ``opened?`` everywhere would allow a stray close at each position).
    The no-reasoning path stays direct (the tag can begin immediately, no
    ``<think>`` required).

    When ``reasoning_sentinels`` is empty / has fewer than two distinct
    well-formed ``<...>`` refs (no reasoning parser, or its markers are NOT
    single special tokens on this tokenizer) the prefix is the bare ``TAG_TEXT``
    — the non-reasoning grammar is byte-identical to PR-3, so this change is a
    strict superset with ZERO regression for non-reasoning constrained calls. The
    reasoning tolerance lives in the compiled grammar, never in a runtime on/off
    gate (path B, a footgun that desyncs the matcher on multi-token boundaries —
    see ``GrammarLogitsProcessor``).
    """
    if not tools:
        raise ValueError("build_tool_lark: tools must not be empty")
    if len(tools) != len(structure_infos):
        raise ValueError(
            "build_tool_lark: tools and structure_infos length mismatch "
            f"({len(tools)} != {len(structure_infos)})"
        )
    if tool_choice == "none":
        raise ValueError(
            "build_tool_lark: tool_choice='none' must not build a grammar "
            "(none produces no constraint at all — caller bug)"
        )
    for si in structure_infos:
        # A trigger is mandatory (StructureInfo contract) — an empty trigger
        # would produce a triggerless grammar the lazy TAG_TEXT prefix could
        # never gate.
        if not si.trigger:
            raise ValueError("build_tool_lark: StructureInfo.trigger must be non-empty")
        # StructTag invariant: begin must start with trigger. Enforce it so a
        # malformed per-family structure_info() is caught at build time rather
        # than silently producing a grammar whose trigger prefix is unused.
        if not si.begin.startswith(si.trigger):
            raise ValueError(
                "build_tool_lark: StructureInfo.begin must start with its "
                f"trigger (trigger={si.trigger!r}, begin={si.begin!r})"
            )
        # The trigger must be a special-token sentinel (see REQUIREMENT above)
        # in EVERY mode — the lazy TAG_TEXT prefix could otherwise reassemble a
        # text trigger from ordinary token pieces and bypass enforcement.
        if si.trigger not in si.sentinels:
            raise ValueError(
                "build_tool_lark: trigger must be declared as a special-token "
                f"sentinel (trigger={si.trigger!r}, sentinels={si.sentinels!r})"
            )

    # Quantifier = (auto/required) × (parallel_tool_calls):
    #   * auto,     parallel OK   -> ``*``  zero-or-more (may emit no call / many)
    #   * auto,     single_call   -> ``?``  ZERO-OR-ONE (auto's "may call zero" ∧
    #                               no-parallel's "at most one" — codex #558-PR5;
    #                               NOT ``*``, which would let auto emit ≥2 calls
    #                               despite the client's parallel_tool_calls=False)
    #   * required, single_call   -> ``""`` EXACTLY ONE (parallel_tool_calls=False)
    #   * required, parallel OK   -> ``+``  one-or-more (design R1: ≥1 forced)
    if tool_choice == "auto":
        quant = "?" if single_call else "*"
    elif single_call:
        quant = ""  # exactly one tag: parallel_tool_calls=False
    else:
        quant = "+"
    tag_names = " | ".join(f"tag_{i}" for i in range(len(tools)))

    # Free-prefix nonterminals (design §5 path A). With NO reasoning pair every
    # tag and the trailing ``tag_end`` consume a bare lazy ``TAG_TEXT`` byte
    # regex, so the emitted grammar is byte-identical to PR-3 (no regression).
    # With a reasoning (open, close) pair we split into a one-time ``lead``
    # (optional prefilled-close) and a balanced-only ``bal_prefix`` reused per
    # tag / tag_end — see the build_tool_lark docstring for the full rationale
    # (round-3 balance, round-4 prefill, round-5 globally-at-most-one).
    #
    # ``reasoning_sentinels`` is an ORDERED ``(open, close)`` pair. We take the
    # first two DISTINCT, well-formed ``<...>`` special-token refs as
    # ``(open, close)``; anything else (fewer than two, a marker that is a single
    # tokenizer token but not a valid bare Lark ref like ``[THINK]``, or open ==
    # close) drops reasoning tolerance and falls back to the bare ``TAG_TEXT``
    # prefix — a safe degrade, never a compile failure.
    reasoning_refs = tuple(
        dict.fromkeys(
            s for s in reasoning_sentinels if s and _is_lark_special_token_ref(s)
        )
    )
    reasoning_pair = reasoning_refs[:2] if len(reasoning_refs) >= 2 else ()

    # FORCED (required / named) + NON-REASONING: AUTO's leading free-text
    # ``TAG_TEXT`` prefix — which exists so an AUTO model may DECLINE to call —
    # would let a *forced* call be deferred indefinitely. A weak-tool-prior
    # family (e.g. Hermes-on-Llama, which natively emits ``get_weather --city
    # ...`` CLI / prose) fills the unbounded ``TAG_TEXT`` prefix and NEVER reaches
    # the trigger before ``max_tokens``; the parser then extracts nothing and the
    # route synthesises an EMPTY-arg call (observed ~40-60% ``{}`` on hermes/qwen;
    # harmony escaped ONLY because its prior emits the trigger immediately). So the
    # FIRST forced tag starts DIRECTLY AT the trigger — NO free/whitespace prefix
    # at all — which is what forces the mandatory call and stops the deferral.
    # Repeated (parallel) calls are separated by a whitespace-only ``SEP``: real
    # models emit them as ``</tool_call>\n<tool_call>`` (newline-separated), so
    # ``SEP`` admits that inter-call gap. ``SEP`` is deliberately UNBOUNDED
    # whitespace: the mandatory first call is ALREADY forced at token 0, so
    # whitespace BETWEEN already-valid calls cannot defer anything (and capping it
    # would wrongly reject a real ``\n\n…`` gap while ``tag_end: TAG_TEXT`` allows
    # unbounded trailing text anyway — codex #558-PR4 round-5 nit). Matches vLLM
    # xgrammar / SGLang forced-tool constraint (forced ⇒ constrain from the
    # trigger, no free prefix). REASONING models never reach here — forced + a
    # normalized reasoning pair is gated to ``None`` in ``build_tool_grammar``
    # (opts out of the #558 grammar; the route forces the call via the pre-#558
    # ``forced_assistant_prefix`` lever). Forcing the trigger at token 0 would be
    # WRONG for a reasoning model — it leaves the prompt-level ``<think>``
    # unclosed, and with ``enable_thinking=True`` the qwen3 reasoning parser then
    # buries the whole output (tool call included) in ``reasoning_content``
    # (``content=None``), losing the call (verified against ``qwen3_parser.py``'s
    # no-``</think>`` branch). A prefill-aware BOUNDED forced-reasoning grammar
    # (single leading close when the prompt prefills ``<think>``; ``<open>…
    # </close>`` otherwise) needs the prompt's reasoning-open state, unavailable at
    # grammar-build time — tracked as a follow-up. (The ``and not reasoning_pair``
    # guard below is defensive: production never reaches ``build_tool_lark`` with a
    # forced reasoning pair, but a direct caller / test might.)
    forced = tool_choice != "auto"
    if forced and not reasoning_pair:
        # ``quant`` here is ``+`` (required / named, ≥1) or ``""`` (single_call,
        # EXACTLY one). Build the call sequence explicitly so the FIRST call sits
        # at the trigger with no separator and only SUBSEQUENT calls carry the
        # ``SEP`` — an interleaved ``(tag)(SEP (tag))*`` rather than a
        # ``(SEP tag)+`` that would admit a leading separator before call one.
        if quant == "+":
            start_calls = f"({tag_names}) (SEP ({tag_names}))*"
        else:  # quant == "" -> EXACTLY ONE forced call (parallel_tool_calls=False)
            start_calls = f"({tag_names})"
        lark = (
            "%llguidance {}\n"
            f"start: {start_calls} tag_end\n"
            "tag_end: TAG_TEXT\n"
            r"SEP: /[ \t\r\n]*/"
            "\n"
            r"TAG_TEXT: /(.|\n)*/"
            "\n"
        )
        prefix_ref = ""  # first call AT the trigger; ``SEP`` separates repeats
    elif not reasoning_pair:
        # No reasoning tolerance: the free prefix is the bare lazy ``TAG_TEXT``
        # byte regex EVERYWHERE, so the emitted grammar is byte-identical to
        # PR-3 (no reasoning regression).
        lark = (
            "%llguidance {}\n"
            f"start: ({tag_names}){quant} tag_end\n"
            "tag_end: TAG_TEXT\n"
            r"TAG_TEXT: /(.|\n)*/"
            "\n"
        )
        prefix_ref = "TAG_TEXT"
    else:
        # BALANCED + PREFILL-TOLERANT (codex #558-PR4 rounds 3-5), AUTO + a
        # reasoning pair. AUTO may DECLINE to call, so an unbounded free prefix
        # (and the one-time prefilled-close tolerance) is correct here — there is
        # no forced call to defer. FORCED + reasoning does NOT reach this branch in
        # production: it is gated to ``None`` upstream in ``build_tool_grammar``
        # (opts out of the #558 grammar → pre-#558 ``forced_assistant_prefix``
        # forcing), because neither this unbounded prefix (weak-tool-prior defer /
        # ``{}`` leak) nor the bounded trigger-first shape (loses a prefilled
        # reasoning model's call) is correct without the prompt's prefill state. A
        # direct caller / test that passes a forced choice + reasoning pair still
        # lands here (harmless — dead in the request path). Two prefix flavours:
        #   * ``lead`` — consumed ONCE at the very start (``start: lead ...``).
        #     It permits a SINGLE optional leading close (``opened?``) modelling
        #     a prompt-prefilled ``<think>``: many reasoning chat templates
        #     prefill ``<think>`` at the END of the prompt (verified on Qwen3.6
        #     and DeepSeek-R1 — the assistant turn ends ``...<think>``), so
        #     GENERATION begins already inside reasoning. The grammar processor
        #     baselines PAST the prompt and only sees the GENERATED tokens, which
        #     then begin with reasoning text + a ``</think>`` whose opener lives
        #     in the unseen prompt. Without this the balanced block would reject
        #     that leading ``</think>`` and BLOCK the required call.
        #   * ``bal_prefix`` — BALANCED-ONLY, reused by every ``tag_i`` and the
        #     trailing ``tag_end``. It admits zero+ EXPLICIT balanced
        #     ``<open> ... <close>`` blocks but NO stray close. Using it (not the
        #     lead prefix) between/after calls means the one-time prefill
        #     tolerance is GLOBALLY at-most-one — a stray ``</think>`` before a
        #     later call, or after the final call, is rejected (codex round-5:
        #     reusing an ``opened?`` prefix everywhere would allow a stray close
        #     at each of those positions).
        # A mid-stream ``<open>`` that never closes is rejected everywhere (it is
        # not the single leading close and has no matching ``<close>``).
        open_ref, close_ref = reasoning_pair
        lark = (
            "%llguidance {}\n"
            f"start: lead ({tag_names}){quant} tag_end\n"
            "lead: opened? bal_prefix\n"
            f"opened: TAG_TEXT {close_ref}\n"
            "bal_prefix: TAG_TEXT (reasoning_block TAG_TEXT)*\n"
            f"reasoning_block: {open_ref} TAG_TEXT {close_ref}\n"
            "tag_end: bal_prefix\n"
            r"TAG_TEXT: /(.|\n)*/"
            "\n"
        )
        prefix_ref = "bal_prefix"

    # Declare the lazy string-value construct (``XML_PARAM_TEXT`` terminal +
    # ``xml_param_value[lazy]`` rule) ONCE, iff any tag uses the XML arg body
    # (``arg_style == "xml"``). A JSON-only tool-set (hermes/qwen/harmony) never
    # emits it, so its grammar is byte-identical to before this change. The rule
    # is declared whenever ANY xml tag is present (even one with no string
    # params); llguidance tolerates the reference-free rule, and gating it on the
    # per-param string check would need a second schema walk for no benefit.
    if any(getattr(si, "arg_style", "json") == "xml" for si in structure_infos):
        lark += _XML_STRING_TERMINAL_DECL
    # Same for the gemma4 string-value rule (``<|"|> GEMMA_STR_TEXT <|"|>``),
    # declared once iff any tag uses the gemma4 arg body. A JSON/XML-only tool-set
    # never emits it, so those grammars stay byte-identical.
    if any(getattr(si, "arg_style", "json") == "gemma4" for si in structure_infos):
        lark += _GEMMA4_STRING_RULE_DECL

    for i, (tool, si) in enumerate(zip(tools, structure_infos)):
        # Only substitute the default when ``parameters`` is ABSENT. A
        # falsy-but-present schema ({} = allow-any, false = allow-none) is a
        # deliberate, meaningful JSON Schema and must be preserved verbatim —
        # ``tool.get("parameters") or default`` would silently clobber it. The
        # default itself is CLOSED (``additionalProperties: false``): a tool
        # that documents no parameters must accept NO arguments, otherwise the
        # grammar would let the model hallucinate arbitrary keys into a
        # no-arg call.
        if "parameters" in tool and tool["parameters"] is not None:
            params = tool["parameters"]
        else:
            params = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        begin_body = _emit_literal_with_sentinels(si.begin, si.sentinels)
        end_body = _emit_literal_with_sentinels(si.end, si.sentinels)
        # ARG BODY, per the family's ``arg_style``: a JSON object (``%json`` over
        # the whole schema) for the default JSON wire, a Qwen3-Coder XML
        # per-parameter body for ``"xml"`` (E3), or the Gemma-4 comma-separated
        # ``key:VALUE`` body for ``"gemma4"`` (E4). The XML/gemma4 body may be
        # EMPTY (a no-argument tool), in which case the tag is just the
        # ``begin``/``end`` frame with no argument region.
        arg_style = getattr(si, "arg_style", "json")
        # ``gemma4_extra_rules`` collects the O(n) suffix nonterminals the gemma4 body
        # references (empty for JSON/XML tags and for a no-arg gemma4 tool); appended
        # to the grammar after the tag rule below.
        gemma4_extra_rules = ""
        if arg_style == "xml":
            arg_body = _emit_xml_arg_body(params)
        elif arg_style == "gemma4":
            arg_body, gemma4_extra_rules = _emit_gemma4_arg_body(params, f"g{i}")
        else:
            arg_body = f"%json {json.dumps(params)}"
        # ``prefix_ref`` is empty on the forced-non-reasoning path (the first call
        # sits AT the trigger; ``SEP`` in ``start`` separates repeats), non-empty
        # otherwise (``TAG_TEXT`` / ``bal_prefix``). Omit the leading space when
        # empty so the tag rule starts cleanly at the begin literal.
        pfx = f"{prefix_ref} " if prefix_ref else ""
        lark += f"\ntag_{i}: {pfx}{begin_body}"
        if arg_body:
            lark += f" {arg_body}"
        if end_body:
            lark += f" {end_body}"
        lark += "\n"
        # Append the gemma4 body's generated suffix nonterminals (rule order is
        # irrelevant in Lark; ``""`` when the tag emits none).
        lark += gemma4_extra_rules
    return lark


# --------------------------------------------------------------------------
# Compiled-grammar cache (design R5).
# --------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _compile_lark_cached(lark: str) -> str | None:
    if not HAS_LLGUIDANCE:
        return None
    try:
        return LLMatcher.grammar_from_lark(lark)
    except Exception:
        logger.exception("tool-grammar: failed to compile Lark")
        return None


# --------------------------------------------------------------------------
# Compiled-matcher template cache (removes the per-request LLMatcher automaton
# build from the decode-setup path).
#
# ``_compile_lark_cached`` above only memoizes the CHEAP Lark -> grammar-JSON
# string translation (``grammar_from_lark`` is a pure, sub-millisecond string
# transform). The EXPENSIVE step is constructing ``LLMatcher(lltokenizer,
# grammar)``: that builds the token-level automaton/lexer for THIS tokenizer's
# vocab (llguidance's own ``LLMatcher.__new__`` docstring: "drops the GIL for the
# duration of the grammar construction, which can be 100-1000ms for extremely
# complex grammars"). ``GrammarLogitsProcessor.__init__`` rebuilt that automaton
# fresh on EVERY request, even for an identical tool-set (measured 1-35ms for
# typical schemas, more for large multi-tool schemas), so we cache it.
#
# The compiled matcher is STATEFUL (``consume_token`` advances a parse cursor),
# so it cannot be shared across concurrent requests. We instead cache an
# IMMUTABLE TEMPLATE — an ``LLMatcher`` in its initial, never-consumed state —
# and hand each request its OWN matcher via ``deep_copy()`` (a ~0.01ms clone,
# verified to produce byte-identical masks to a fresh construct). The template is
# never fed a token, so it stays read-only in the initial state.
#
# KEY = (id(lltokenizer), grammar-JSON string). The grammar string already
# encodes the full tool-set + each tool's JSON Schema + the model family's
# wire-format sentinels (``build_tool_lark`` bakes all of that in), so it is the
# canonicalized schema/parser-family half of the key. The tokenizer identity is
# the other half — the compiled automaton is vocab-specific, so two models with
# different tokenizers but the same grammar MUST NOT share a template. We pin the
# ``lltokenizer`` object in the cache VALUE so its ``id()`` cannot be recycled to
# a different tokenizer while an entry is live. In practice rapid-mlx holds ONE
# tokenizer per model for its lifetime (``get_lltokenizer`` memoizes a singleton
# per tokenizer), so the id is stable; the pin is defense-in-depth for the
# multi-model routing path.
#
# BOUNDED LRU (``OrderedDict`` + a lock) so a client that streams unbounded
# distinct schemas cannot grow the cache without limit. The bound is BOTH a
# count cap AND a byte budget: a count cap alone does not bound memory, because a
# client-controlled grammar string (and the native automaton built from it) can
# be large — the upstream route already rejects a schema whose serialized tools
# list exceeds 64 KiB (``_TOOL_GRAMMAR_MAX_SCHEMA_BYTES``), but ``%json``
# expansion can still make the emitted grammar several hundred KiB. So we evict
# by whichever limit binds first and REFUSE to cache a single grammar larger than
# the whole byte budget (it would evict everything and still overflow); such an
# outlier rebuilds per request (rare — the schema is already ≤64 KiB). ``len(
# grammar)`` is the byte proxy (the automaton size scales with grammar size).
#
# THREAD-SAFE with PER-KEY SINGLE-FLIGHT. The chat route runs this on a bounded
# build-executor pool, so N threads may hit the same key concurrently. The global
# lock guards the dict + the ~0.01ms ``deep_copy`` only; the EXPENSIVE
# ``LLMatcher`` construction runs OUTSIDE the global lock so one slow schema miss
# never head-of-line-blocks cache hits or OTHER keys' builds. A PER-KEY
# ``Event`` makes construction at-most-once per key: the first thread to miss a
# key builds it; concurrent threads for the SAME key wait on the Event and then
# re-read the cache (codex #1155 — a barrier test asserts a single construction
# under a cold burst). Different keys still build fully in parallel.
#
# MEMORY BOUND — a DELIBERATELY CONSERVATIVE entry cap AND a byte budget, evicting
# on whichever binds first. A count cap alone does not bound memory: a
# client-controlled grammar (and the native automaton built from it) can be
# large, and llguidance exposes NO native byte count, so exact automaton/tokenizer
# accounting is not possible — ``len(grammar)`` is the in-cache size proxy (the
# automaton scales with grammar size). We therefore compensate with a SMALL entry
# cap (32): real deployments expose a handful of distinct tool schemas, so 32
# comfortably covers reuse while keeping the WORST-case cached automaton count
# small even though each automaton's exact bytes are unmeasurable (codex #1155 —
# "substantially safer entry bound"). Input is additionally pre-bounded upstream:
# the route rejects a serialized tools list over 64 KiB
# (``_TOOL_GRAMMAR_MAX_SCHEMA_BYTES``) BEFORE a grammar is built. A single grammar
# larger than the whole byte budget is NOT cached (it would evict everything and
# still overflow) and rebuilds per request. Retention is thus LRU-BOUNDED, not a
# leak: a burst of distinct one-off schemas evicts oldest-first back under the
# caps. The cached template pins its ``lltokenizer`` (so ``id()`` can't be
# recycled to a different tokenizer while live); rapid-mlx's engine already holds
# that tokenizer for the model's lifetime, so this adds only bounded, LRU-evicted
# post-unload retention — no explicit model-lifecycle hook is warranted for a
# cache this small.
_COMPILED_MATCHER_CACHE_MAX = 32
_COMPILED_MATCHER_CACHE_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB grammar-string budget
_compiled_matcher_cache: "OrderedDict[tuple[int, str], tuple[Any, Any, int]]" = (
    OrderedDict()
)
_compiled_matcher_cache_bytes = 0
_compiled_matcher_lock = threading.Lock()


class _BuildSlot:
    """Per-key single-flight slot: an Event + the built template for waiters.

    The builder publishes a VALID template on ``template`` before waking waiters,
    so a concurrent burst on a grammar that is NOT retained in the LRU (a valid
    grammar too large for the byte budget) still CLONES the one build instead of
    each waiter re-running the expensive construction (codex #1155). A broken
    grammar is left unpublished (``template`` stays ``None``) — those are cheap to
    rebuild and rare, and this keeps ``deep_copy`` off an errored matcher.
    """

    __slots__ = ("event", "template")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.template: Any = None


# Per-key in-flight build registry for single-flight (key -> _BuildSlot).
_compiled_matcher_building: "dict[tuple[int, str], _BuildSlot]" = {}


def _evict_compiled_matchers_locked() -> None:
    """Evict LRU entries until BOTH the count cap and byte budget are satisfied.

    Caller must hold ``_compiled_matcher_lock``.
    """
    global _compiled_matcher_cache_bytes
    while _compiled_matcher_cache and (
        len(_compiled_matcher_cache) > _COMPILED_MATCHER_CACHE_MAX
        or _compiled_matcher_cache_bytes > _COMPILED_MATCHER_CACHE_MAX_BYTES
    ):
        _old_key, _old_val = _compiled_matcher_cache.popitem(last=False)
        _compiled_matcher_cache_bytes -= _old_val[2]


def _cache_hit_copy_locked(key: "tuple[int, str]") -> Any:
    """Return a per-request ``deep_copy`` if ``key`` is cached, else ``None``.

    Caller must hold ``_compiled_matcher_lock``. Refreshes LRU recency on a hit.
    """
    entry = _compiled_matcher_cache.get(key)
    if entry is None:
        return None
    _compiled_matcher_cache.move_to_end(key)
    return entry[1].deep_copy()


def get_request_matcher(lltokenizer: Any, grammar: str) -> Any:
    """Return a FRESH per-request ``LLMatcher`` for ``(lltokenizer, grammar)``.

    Builds the compiled automaton AT MOST ONCE per distinct ``(tokenizer,
    grammar)`` — even under a concurrent cold burst (per-key single-flight) — and
    clones it per request via ``deep_copy`` (see the cache-block comment above).
    The returned matcher is a private, initial-state instance the caller owns and
    may ``consume_token`` freely — the cached template is never mutated. A grammar
    that fails to compile (non-empty ``get_error()``) is NOT cached and is
    returned as-is, so the broken-matcher fallback in ``GrammarLogitsProcessor``
    behaves exactly as before.
    """
    global _compiled_matcher_cache_bytes
    key = (id(lltokenizer), grammar)
    while True:
        with _compiled_matcher_lock:
            hit = _cache_hit_copy_locked(key)
            if hit is not None:
                return hit
            slot = _compiled_matcher_building.get(key)
            if slot is None:
                # We own the build for this key; publish a slot others wait on.
                slot = _BuildSlot()
                _compiled_matcher_building[key] = slot
                is_builder = True
            else:
                is_builder = False
        if not is_builder:
            # Another thread is building this exact key: wait, then prefer the
            # cache; else reuse the builder's PUBLISHED template so a burst never
            # re-runs the build — even for a grammar that is not retained in the
            # LRU (valid-but-oversized) or is broken (codex #1155). A broken
            # matcher is INERT (``is_broken()`` short-circuits — it never masks or
            # consumes and the route discards it), so waiters may SHARE it
            # directly; a valid template is cloned per request. Only a build that
            # crashed before publishing leaves the slot empty -> loop and rebuild.
            slot.event.wait()
            with _compiled_matcher_lock:
                hit = _cache_hit_copy_locked(key)
            if hit is not None:
                return hit
            published = slot.template
            if published is not None:
                return published if published.get_error() else published.deep_copy()
            continue

        # We own the build. Construct OUTSIDE the global lock so this slow step
        # blocks neither cache hits nor OTHER keys' builds.
        try:
            template = LLMatcher(lltokenizer, grammar)
            # Publish immediately (valid OR broken) so concurrent waiters on the
            # SAME key reuse this single build instead of each re-compiling it —
            # covers the valid-but-oversized AND the broken-grammar bursts.
            slot.template = template
            if template.get_error():
                # Broken -> uncached (is_broken() fallback unchanged); returned
                # as-is, and shared with waiters (inert, so sharing is safe).
                return template
            # Byte count (NOT len()): a non-ASCII grammar's UTF-8 size can be up
            # to ~4x its code-point count, so ``len(grammar)`` would under-count
            # the byte budget (codex #1155).
            nbytes = len(grammar.encode("utf-8"))
            with _compiled_matcher_lock:
                # Refuse to cache a single grammar larger than the whole byte
                # budget — it would evict everything and still overflow; it
                # rebuilds per request (rare: schema pre-capped at 64 KiB).
                if nbytes <= _COMPILED_MATCHER_CACHE_MAX_BYTES:
                    _compiled_matcher_cache[key] = (lltokenizer, template, nbytes)
                    _compiled_matcher_cache_bytes += nbytes
                    _evict_compiled_matchers_locked()
            return template.deep_copy()
        finally:
            # Release single-flight ATOMICALLY: signal completion AND drop the
            # registry entry under the SAME lock, so no arrival can observe "key
            # not building" before the Event is set and start a duplicate build
            # of an uncached (oversized) grammar (codex #1155).
            with _compiled_matcher_lock:
                slot.event.set()
                _compiled_matcher_building.pop(key, None)


def build_tool_grammar(
    tools: list[dict[str, Any]],
    tool_choice: str,
    parser: Any,
    *,
    single_call: bool = False,
    reasoning_sentinels: tuple[str, ...] = (),
) -> str | None:
    """Public entry: (tools, tool_choice, parser) -> compiled llguidance grammar.

    INPUT CONTRACT (NORMALIZED — not raw OpenAI request shapes): ``tools`` is
    a list of ``{"name": str, "parameters": <json-schema>}`` dicts and
    ``tool_choice`` is a resolved string — ``"auto"`` / ``"required"`` /
    ``"none"`` / a concrete function name. Un-wrapping the OpenAI request
    shapes (``{"type":"function","function":{"name":...,"parameters":...}}``
    for tools; ``{"type":"function","function":{"name":...}}`` for a named
    choice) is the CALLER's job — it happens in the PR-3 chat.py routing that
    wires this builder in. This function deliberately does NOT normalize
    request objects; that responsibility lives with the routing PR so the
    builder stays a small, pure, request-shape-agnostic unit. An unexpected
    shape degrades safely to ``None`` (free-form) rather than crashing.

    ``reasoning_sentinels`` (#558 PR-4, path A): an ORDERED ``(open, close)``
    pair of reasoning-boundary special tokens (``<think>`` / ``</think>``) the
    CALLER has proven single special tokens on this tokenizer. Passed through to
    ``build_tool_lark`` so the free prefix admits a BALANCED ``<think>...
    </think>`` block before the trigger. Empty (the default) reproduces the PR-3
    non-reasoning grammar exactly.

    Returns ``None`` when ``tool_choice`` is ``"none"`` (no constraint at
    all), when the parser declares no ``structure_info`` (family not yet
    supported -> caller falls back to today's free-form behavior), or when
    llguidance is unavailable / a per-family factory raises / compilation
    fails. Any of these paths degrades safely to today's free-form behavior.
    (PR-3 note: this collapses "constraint unavailable" and "requested
    constraint malformed" into the same ``None``; the routing PR that adds a
    real caller decides whether a malformed *required/named* constraint
    should hard-fail instead of falling open — PR-1 has no caller to make
    that policy call, so it uniformly fails open.)

    NOTE (PR-1): this function has NO call sites in the request path yet.
    ``chat.py`` / ``scheduler.py`` routing is deliberately not wired here —
    it lands with the runtime processor in PR-3. Until then this is inert
    scaffolding and cannot change tool-call behavior.
    """
    if not HAS_LLGUIDANCE or not tools:
        return None
    if tool_choice == "none":
        # ``none`` means the model sees no tools -> no grammar (design §4).
        return None
    # AUTO-path soundness gate (#558). The builder's "free byte prefix + single-
    # special-token TRIGGER" model expresses auto's zero-call invariant ONLY when
    # the trigger token appears EXCLUSIVELY at a tool-call boundary. A family
    # whose only single-special-token trigger is SHARED with non-tool responses
    # (harmony: ``<|channel|>`` precedes commentary/final/analysis alike) cannot
    # — committing to the tool tag on that shared trigger FORCES a call, turning
    # ``auto`` into ``required`` (the #1 regression #558 guards against; verified
    # offline: a ``<|channel|>final...`` reply is rejected at the first token).
    # Such a family declares ``TOOL_GRAMMAR_AUTO_SAFE = False`` and stays
    # free-form on auto (non-regressive), while ``required``/named still build a
    # grammar below (a forced tool-call structure is exactly what those modes
    # ask for). Every single-special-token-trigger family (hermes/qwen) defaults
    # ``True`` and is unaffected.
    if tool_choice == "auto" and not getattr(parser, "TOOL_GRAMMAR_AUTO_SAFE", True):
        return None
    # SECTION-WRAPPER soundness gate (#558 E1). A family whose structure_info folds
    # the whole native tool-calls SECTION envelope into each call's begin/end
    # (DeepSeek/Kimi) is sound only when the grammar emits AT MOST ONE call:
    # repeating the tag (not single_call -> `+`/`*`) yields back-to-back sections,
    # which these parsers' single-envelope scanner drops after the first section.
    # Opt OUT to free-form when >1 call is possible (non-regressive: the model then
    # emits its canonical one-section-N-calls wire, which the parser handles). True
    # multi-call section-wrapper grammar support is a tracked follow-up.
    if getattr(parser, "TOOL_GRAMMAR_SECTION_WRAPPER", False) and not single_call:
        return None
    # FORCED (required / named) + a REASONING model: opt OUT of the #558 grammar
    # and return ``None`` (the route then FORCES the call via the pre-#558
    # ``forced_assistant_prefix`` injection — a proven, shipped lever). Neither
    # static grammar shape works here:
    #   * the shared reasoning grammar (``lead: opened? bal_prefix``) keeps an
    #     UNBOUNDED free prefix so a weak-tool-prior reasoning model can exhaust
    #     ``max_tokens`` before the trigger and reproduce the empty-``{}`` leak
    #     (codex #558-PR4 round-5);
    #   * the bounded trigger-first forced grammar forces the trigger at token 0,
    #     which leaves the prompt-level ``<think>`` unclosed — with
    #     ``enable_thinking=True`` the qwen3 reasoning parser then buries the whole
    #     output (tool call included) in ``reasoning_content`` and loses the call
    #     (verified against ``qwen3_parser.py``'s no-``</think>`` branch).
    # A CORRECT bounded forced-reasoning grammar must know whether the PROMPT
    # prefills ``<think>`` (prefilled ⇒ a single leading close before the trigger;
    # non-prefilled ⇒ ``<open>…</close>`` required first) — that prefill signal is
    # not available at grammar-build time (grammar built before the prompt is
    # rendered; a runtime ``</think>``-gate was already rejected in-tree as a
    # footgun a no-``</think>`` model defeats). So forced+reasoning stays on the
    # pre-#558 lever until that prefill-state plumbing lands (tracked follow-up).
    # Gate on the NORMALIZED reasoning pair (dedup + a valid, distinct ``<...>``
    # (open, close)), MIRRORING ``build_tool_lark``: a single / empty /
    # duplicate-only / malformed sentinel sequence is NOT a reasoning model and
    # must take the NON-reasoning constrained path, not be disabled here.
    _reasoning_refs = tuple(
        dict.fromkeys(
            s for s in reasoning_sentinels if s and _is_lark_special_token_ref(s)
        )
    )
    _reasoning_pair = _reasoning_refs[:2] if len(_reasoning_refs) >= 2 else ()
    if tool_choice != "auto" and _reasoning_pair:
        return None
    info_fn = getattr(parser, "structure_info", None)
    if info_fn is None:
        return None
    try:
        get_info = info_fn()
    except Exception:
        logger.exception("tool-grammar: parser.structure_info() raised")
        return None
    if get_info is None:
        return None  # family opted out (default ABC behavior)

    # NAMED tool_choice: any value other than the reserved ``auto``/``required``
    # (``none`` already returned above) names a specific function. Narrow
    # ``tools`` to just that function INSIDE the builder so a named choice can
    # only ever emit the requested tool — never one of the other supplied
    # tools — regardless of how the caller passed ``tools`` (design §4). An
    # unknown name degrades to free-form rather than silently constraining.
    if tool_choice not in ("auto", "required"):
        named = [t for t in tools if t.get("name") == tool_choice]
        if not named:
            logger.warning(
                "tool-grammar: named tool_choice %r not in tools; free-form",
                tool_choice,
            )
            return None
        tools = named

    structure_infos: list[StructureInfo] = []
    for tool in tools:
        name = tool.get("name")
        if not name:
            return None
        # Mirror the structure_info() guard: a per-family factory that raises
        # on a specific tool name must degrade to free-form, not crash the
        # request (codex round-2 nit — consistent fallback policy).
        try:
            si = get_info(name)
        except Exception:
            logger.exception("tool-grammar: structure_info factory raised")
            return None
        if si is None:
            return None
        structure_infos.append(si)

    # FAITHFUL-OR-OPT-OUT gate for the Qwen3-Coder XML arg wire (#558 E3, codex
    # converge). The XML emitter cannot faithfully constrain a handful of schema
    # shapes (property-less-but-non-trivial / unconstrained string facets /
    # object-level relational keywords / delimiter-unsafe keys / an unresolvable
    # ``$ref`` — see ``_xml_schema_representable``). When ANY xml-arg tool carries
    # such a shape, opt the WHOLE request OUT of grammar (return ``None`` -> the
    # route falls back to free-form-then-parse) rather than emit a grammar that
    # silently allows schema-invalid or mis-typed output. This mirrors the
    # existing opt-outs (``structure_info() -> None`` / ``TOOL_GRAMMAR_AUTO_SAFE``)
    # and applies ONLY to ``arg_style != "json"`` tools — the JSON-body families
    # (hermes / qwen / harmony) are ``%json <schema>`` and stay byte-identical.
    #
    # The model tokenizer (stored on every ``ToolParser`` as ``model_tokenizer``,
    # constructed ``parser_cls(tokenizer=...)`` on the chat route / warmup) makes
    # the gemma4 ENUM guard COMPLETE BY CONSTRUCTION (codex r3 E4): an enum value
    # that tokenizes through ANY registered special token — not just the five
    # hard-coded structural markers — compiles to an unreachable byte-literal branch
    # and opts the request out. ``None`` (no tokenizer / degraded) falls back to the
    # structural 5-marker subset; the XML wire is unaffected (it already rejects
    # ``<``/``>``).
    tok = getattr(parser, "model_tokenizer", None)
    for tool, si in zip(tools, structure_infos):
        arg_style = getattr(si, "arg_style", "json")
        if arg_style == "json":
            continue
        # Resolve ``parameters`` EXACTLY as ``build_tool_lark`` does (a genuine
        # no-arg tool gets the closed default, which IS representable).
        if "parameters" in tool and tool["parameters"] is not None:
            params = tool["parameters"]
        else:
            params = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        # Same strict-allowlist guard, per-wire policy (E3 XML / E4 gemma4). An
        # unknown non-json arg_style is treated conservatively (opt out).
        if arg_style == "gemma4":
            representable = _gemma4_schema_representable(params, tokenizer=tok)
        elif arg_style == "xml":
            representable = _xml_schema_representable(params)
        else:
            representable = False
        if not representable:
            logger.debug(
                "tool-grammar: %s arg schema for tool %r not faithfully "
                "representable; opting request out of grammar (free-form)",
                arg_style,
                tool.get("name"),
            )
            return None

    try:
        lark = build_tool_lark(
            tools,
            tool_choice,
            structure_infos,
            single_call=single_call,
            reasoning_sentinels=reasoning_sentinels,
        )
    except ValueError:
        # Malformed structure_info / unsupported tool_choice -> free-form.
        logger.exception("tool-grammar: build_tool_lark rejected inputs")
        return None
    return _compile_lark_cached(lark)


def model_stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Collect the model's stop/eos token ids from a tokenizer (0.10.16 P1-①).

    Unions the same surfaces the scheduler's ``_get_stop_tokens`` reads so the
    ``GrammarLogitsProcessor`` re-admits EXACTLY the ids that will actually halt
    generation:

      * ``_eos_token_ids`` — mlx-lm ``TokenizerWrapper``'s curated set, grown at
        load by ``augment_eos_token_ids_from_generation_config`` to include the
        chat-template terminators declared in ``generation_config.json``'s
        ``eos_token_id`` list (Gemma-4 ``<turn|>`` id 106 / ``<|tool_response>``
        id 50, Qwen3 ``<|endoftext|>``, Llama-3 ``<|eot_id|>``, …).
      * ``eos_token_id`` — the singular HF surface (int or list).
      * ``eos_token_ids`` — the plural processor surface.
      * the Rapid-MLX extras stash (``_rapid_extra_eos_token_ids``) for raw HF
        tokenizers whose ``eos_token_ids`` property rejects assignment.

    Returns a sorted tuple (possibly empty — the processor then behaves exactly
    as before this change)."""
    ids: set[int] = set()
    if tokenizer is None:
        return ()
    wrapper_ids = getattr(tokenizer, "_eos_token_ids", None)
    if wrapper_ids:
        ids.update(int(t) for t in wrapper_ids)
    single = getattr(tokenizer, "eos_token_id", None)
    if single is not None:
        if isinstance(single, (list, set, tuple)):
            ids.update(int(t) for t in single)
        else:
            ids.add(int(single))
    plural = getattr(tokenizer, "eos_token_ids", None)
    if plural is not None and isinstance(plural, (list, set, tuple)):
        ids.update(int(t) for t in plural)
    extras = getattr(tokenizer, "_rapid_extra_eos_token_ids", None)
    if extras:
        ids.update(int(t) for t in extras)
    return tuple(sorted(ids))


# --------------------------------------------------------------------------
# Runtime logits processor (design §3.3 / §5). Mirrors the
# ``MiniMaxToolLogitsProcessor.__call__(token_ids, logits)`` signature the
# scheduler's per-request ``request_processors`` slot expects, so a
# ``GrammarLogitsProcessor`` composes with the penalty processors already in
# that slot.
# --------------------------------------------------------------------------
class GrammarLogitsProcessor:
    """Applies a compiled llguidance grammar mask to logits each decode step.

    Contract with mlx-lm's decode loop: the processor is called every step
    with ``(token_ids, logits)`` where ``token_ids`` is the FULL CUMULATIVE
    sequence so far (prompt + everything generated), NOT just the newly
    sampled token. ``TokenBuffer.update_and_fetch`` returns ``buffer[:end]``,
    so the prompt tokens are present on the very first call. Feeding those
    prompt tokens into the grammar matcher would immediately reject them
    (the grammar describes the tool-call OUTPUT, not the prompt) and break
    the constraint. This processor therefore BASELINES past the prompt on the
    first call and only ever advances the matcher over generated tokens.

    Reasoning-aware (design §5): if a ``reasoning_end_token`` is supplied, the
    mask is held OFF (all tokens allowed) until that token string appears in
    the decoded output, so a ``<think>...</think>`` block is not constrained.
    The chat route deliberately passes ``reasoning_end_token=None`` and relies
    on PATH A (the grammar's own lazy ``TAG_TEXT`` free prefix swallows
    reasoning before the trigger) — a runtime gate keyed on ``</think>`` is a
    footgun for models that emit no reasoning (the mask would stay off
    forever). The flag remains for defense-in-depth callers that want path B.
    """

    def __init__(
        self,
        lltokenizer: Any,
        grammar: str,
        *,
        reasoning_end_token: str | None = None,
        tokenizer: Any = None,
        stop_token_ids: Any = None,
    ):
        self._lltok = lltokenizer
        # Reuse a cached compiled-grammar TEMPLATE (deep-copied per request) so
        # the automaton is built at most once per distinct (tokenizer, grammar)
        # instead of on every request; see ``get_request_matcher``. Falls back to
        # a direct build on the (broken-grammar) uncached path, so the
        # ``get_error()`` / ``is_broken()`` handling below is unchanged.
        self._matcher = get_request_matcher(lltokenizer, grammar)
        err = self._matcher.get_error()
        # A non-empty ``get_error()`` means the grammar failed to compile. A
        # broken matcher masks nothing (``__call__`` returns logits unchanged),
        # so callers MUST NOT treat a broken processor as an active constraint —
        # they should fall back to free-form / forced-prefix instead. The route
        # builder (``_maybe_build_tool_grammar_processor``) checks ``is_broken``
        # and returns ``None`` in that case (codex #558-PR3).
        self._compile_error = err or None
        self._broken = bool(err)
        if err:
            logger.error("tool-grammar: matcher compile error: %s", err)
        self._vocab = lltokenizer.vocab_size
        self._bitmask = allocate_token_bitmask(1, self._vocab)
        self._reasoning_end_token = reasoning_end_token
        self._reasoning_ended = reasoning_end_token is None
        self._tokenizer = tokenizer
        # MODEL STOP/EOS tokens re-admitted at accepting states (0.10.16 dogfood
        # P1-①). llguidance's compiled grammar terminates on the tokenizer's
        # SINGLE ``eos_token``. That is correct for families whose learned turn-
        # terminator IS that token (Qwen ``<|im_end|>`` == tokenizer eos) or is
        # the tool-call ``end`` literal the grammar already consumes (gpt-oss
        # harmony ``<|call|>`` is one of ``generation_config.eos_token_id``). It
        # BREAKS Gemma-4: the model ends a tool-call turn with ``<turn|>`` (id
        # 106) or ``<|tool_response>`` (id 50) — special tokens the grammar's
        # ``TAG_TEXT: /(.|\n)*/`` byte-regex tail CANNOT match, and which are NOT
        # the tokenizer eos ``<eos>`` (id 1) llguidance treats as the grammar
        # terminator. So after ONE complete call every one of the model's real
        # turn-terminators is masked; under the ``required``/named ``(tag)+``
        # grammar the only unmasked structural continuation is another
        # ``<|tool_call>`` and the model emits the identical call forever (never
        # reaching ``finish_reason="tool_calls"`` without a ``max_tokens`` cap).
        # We therefore re-admit the model's stop tokens WHENEVER the matcher is
        # in an ACCEPTING state (grammar structurally complete → a call already
        # emitted for ``required``, or zero-or-more satisfied for ``auto``), so
        # the model can stop naturally — exactly how Qwen/gpt-oss already do. The
        # ``is_accepting`` gate preserves ``required``'s force-≥1 guarantee: the
        # start state is NON-accepting, so stop tokens stay masked until the
        # first call completes.
        stop_ids = sorted(
            {int(t) for t in (stop_token_ids or ()) if 0 <= int(t) < self._vocab}
        )
        self._stop_ids: tuple[int, ...] = tuple(stop_ids)
        self._stop_ids_arr = None
        if self._stop_ids:
            import mlx.core as mx

            self._stop_ids_arr = mx.array(self._stop_ids)
        # mlx-lm passes the FULL cumulative token sequence each step (see class
        # docstring). ``_prompt_len`` is the baseline captured on the first
        # call; ``_committed`` tracks how many of the cumulative ids have been
        # consumed into the matcher so far.
        self._prompt_len: int | None = None
        self._committed = 0
        # Desync abort latch (codex #558-PR3). If the matcher ever REJECTS an
        # already-sampled+committed token, its internal state is desynced from
        # the real output stream — every subsequent mask it computes is garbage.
        # HONESTY (codex #558-PR3 nit): this is a controlled FAIL-OPEN fallback,
        # NOT fail-closed — on desync we DROP the constraint and let the request
        # finish under the downstream free-form parser, rather than terminating
        # it with an error. We choose free-form-fallback over hard-error because
        # (a) the grammar constraint is best-effort throughout this module (a
        # missing ``[guided]`` extra, an uncompilable grammar, and an unsupported
        # tokenizer all already degrade to free-form, never a 500), and (b)
        # continuing to mask from a desynced matcher — the previous behavior —
        # would emit garbage constraints, strictly worse than dropping it. A
        # desync should be unreachable in practice (the matcher is fed exactly
        # the tokens it produced masks for), so the latch is defense-in-depth.
        self._aborted = False

    def is_broken(self) -> bool:
        """Whether the grammar failed to compile (masks nothing; use fallback)."""
        return self._broken

    def _maybe_open_after_reasoning(self, token_ids: Any) -> None:
        if self._reasoning_ended or self._tokenizer is None:
            return
        try:
            gen_ids = list(token_ids)
            if self._prompt_len is not None:
                gen_ids = gen_ids[self._prompt_len :]
            text = self._tokenizer.decode(gen_ids)
        except Exception:
            return
        if self._reasoning_end_token in text:
            self._reasoning_ended = True

    def __call__(self, token_ids: Any, logits: Any) -> Any:
        if self._broken or self._aborted:
            # Broken (never compiled) or aborted (matcher rejected a committed
            # token and is desynced): mask nothing so downstream free-form
            # parsing owns the request rather than a garbage mask (codex
            # #558-PR3).
            return logits
        # mlx-lm hands us the FULL cumulative sequence (prompt + generated) each
        # step. NEVER copy the whole thing (``list(token_ids)`` would be O(n) per
        # step => O(n^2) over a long constrained generation — codex #558-PR3).
        # ``len(...)`` is O(1); we only ever materialize the UNCOMMITTED TAIL.
        n = len(token_ids)
        if self._prompt_len is None:
            # First step: everything present so far is the prompt — no token
            # has been sampled and appended yet at mask-compute time. Baseline
            # here so the matcher is NEVER fed prompt tokens (MUST-FIX #1).
            self._prompt_len = n
            self._committed = n
        # Commit any newly-generated tokens since the last call so the matcher
        # state tracks the real output stream (design §6: commit every token).
        # Slice ONLY the new tail — never the already-committed prefix.
        if self._committed < n:
            tail = token_ids[self._committed : n]
            for t in tail:
                self._committed += 1
                if not self._reasoning_ended:
                    continue
                tok = int(t)
                if not self._matcher.consume_token(tok):
                    # DESYNC FALLBACK (codex #558-PR3): the matcher rejected a
                    # token that was ALREADY sampled and committed to the output
                    # stream, so its state no longer tracks the real stream. Latch
                    # the abort and STOP consuming further tail tokens into the
                    # now-invalid matcher; the mask branch below short-circuits on
                    # ``_aborted`` and returns logits unchanged. This is a
                    # controlled FAIL-OPEN — we DROP the constraint and finish
                    # under the free-form parser rather than mask from a desynced
                    # matcher (which the old behavior did — strictly worse).
                    logger.warning(
                        "tool-grammar: matcher rejected committed token %d; "
                        "dropping grammar constraint (free-form fallback)",
                        tok,
                    )
                    self._aborted = True
                    break

        # Once aborted (this step or a prior one), never mask again — the matcher
        # is desynced, so any bitmask it produces is garbage. Return logits
        # unchanged so downstream free-form parsing owns the request.
        if self._aborted:
            return logits

        # Reasoning gate is checked AFTER committing this step's tokens. NOTE
        # (codex #558-PR3 nit, path B only): if a single multi-token update
        # carried both ``</think>`` and the first post-reasoning tool-call
        # tokens, those tokens are advanced past without being consumed here.
        # Precise ``</think>``-boundary token alignment (excluding the reasoning
        # bytes but consuming the tail) is design §7 open-Q1, deferred with the
        # PR-5 auto path. This is dormant in production: the chat route uses
        # PATH A (``reasoning_end_token=None`` => ``_reasoning_ended`` is True
        # from construction, so the ``continue`` above never fires and every
        # token is consumed). Only a defense-in-depth path-B caller that sets
        # ``reasoning_end_token`` hits the deferral.
        if not self._reasoning_ended:
            self._maybe_open_after_reasoning(token_ids)
        if not self._reasoning_ended:
            return logits  # free generation during reasoning

        if self._matcher.is_stopped():
            return logits

        fill_next_token_bitmask(self._matcher, self._bitmask, 0)
        model_vocab = logits.shape[-1]
        if model_vocab > self._vocab:
            # The model head can be wider than the tokenizer vocab (padded
            # embedding). The bitmask is real-vocab-width, so mask only the
            # real-vocab prefix — but PRESERVE the original ``[..., model_vocab]``
            # shape (the logits-processor contract: mlx-lm concatenates every
            # row's processed logits, so a narrower return breaks batched
            # decode — codex #558-PR3). Force the padded tail to ``-inf`` so
            # those never-valid ids can't be sampled, then re-join.
            import mlx.core as mx

            head = apply_token_bitmask(logits[..., : self._vocab], self._bitmask)
            tail_shape = (*logits.shape[:-1], model_vocab - self._vocab)
            tail = mx.full(tail_shape, -float("inf"), dtype=logits.dtype)
            out = mx.concatenate([head, tail], axis=-1)
        elif model_vocab < self._vocab:
            # Inverse case (codex #558-PR3): the TOKENIZER carries added tokens
            # beyond the model's narrower output head. The bitmask is packed to
            # the full tokenizer vocab (``ceil(vocab/32)`` int32 words), so
            # applying it whole to the narrower logits would over-cover. Slice
            # the packed bitmask to the model-width WORD count (each int32 word
            # covers 32 ids) and apply that — any tokenizer id >= model_vocab
            # has no logit column to sample from anyway, so dropping those tail
            # words is safe. ``apply_token_bitmask`` reads only ``batch`` rows
            # (no vocab-width assert), so the model-width prefix mask is enough.
            words = (model_vocab + 31) // 32
            out = apply_token_bitmask(logits, self._bitmask[..., :words])
        else:
            out = apply_token_bitmask(logits, self._bitmask)
        return self._readmit_stop_tokens_if_accepting(out, logits)

    def _readmit_stop_tokens_if_accepting(self, out: Any, logits: Any) -> Any:
        """Un-mask the model's stop/eos tokens when the grammar could terminate.

        See ``__init__`` for the Gemma-4 root cause (0.10.16 dogfood P1-①): a
        family whose learned turn-terminator is neither the tokenizer's single
        grammar-EOS nor the tool-call ``end`` literal can never emit a stop token
        under the mask, so ``tool_choice="required"`` loops the same call forever.
        Whenever the matcher ``is_accepting()`` — the grammar is structurally
        complete and MAY end here — we restore the ORIGINAL (unmasked) logits for
        exactly the model's stop token columns, letting the sampler pick one and
        terminate. The gate is essential: in a NON-accepting state (e.g. before
        the mandatory first ``required`` call, or mid-call) the stop tokens stay
        masked, preserving the force-≥1 and shape guarantees. A no-op for
        families that already terminate cleanly (their stop token is already
        grammar-allowed at the accepting state, so restoring its finite logit
        changes nothing)."""
        if self._stop_ids_arr is None:
            return out
        try:
            if not self._matcher.is_accepting():
                return out
        except Exception:  # pragma: no cover - defensive; matcher API is stable
            return out
        idx = self._stop_ids_arr
        width = out.shape[-1]
        # Defensive width clamp: ``_stop_ids`` is already bounded by the tokenizer
        # vocab at construction, but the model head can be NARROWER than the
        # tokenizer (``model_vocab < self._vocab`` branch above) — never index a
        # stop id past the actual logits width.
        if self._stop_ids[-1] >= width:
            import mlx.core as mx

            clamped = [t for t in self._stop_ids if t < width]
            if not clamped:
                return out
            idx = mx.array(clamped)
        out[..., idx] = logits[..., idx]
        return out

    def reset(self) -> None:
        self._matcher.reset()
        self._prompt_len = None
        self._committed = 0
        self._reasoning_ended = self._reasoning_end_token is None
        self._aborted = False


def build_lltokenizer(tokenizer: Any) -> Any:
    """Build an llguidance ``LLTokenizer`` from an mlx-lm ``TokenizerWrapper``.

    llguidance needs the underlying *fast* (Rust-backed) transformers
    tokenizer. mlx-lm's ``TokenizerWrapper`` exposes it at ``._tokenizer``
    (what ``guided.py`` uses). We try candidates in priority order:

      1. the wrapper's inner ``._tokenizer`` (a transformers 5.x
         ``TokenizersBackend`` — this is what ``llguidance.hf.from_tokenizer``
         accepts and is the common case);
      2. the object as-is (for a raw fast tokenizer passed directly);
      3. a DIRECT ``LLTokenizer(<tokenizer.json>)`` build from the fast
         tokenizer's ``backend_tokenizer.to_str()``.

    Candidate 3 is the spike-proven fallback for the transformers gotcha: on
    some transformers revisions ``from_tokenizer``'s ``isinstance`` gate
    rejects a ``Qwen2Tokenizer``-family object even though its serialized
    ``tokenizer.json`` builds a valid ``LLTokenizer`` directly. ``guided.py``
    has no such fallback, so tool-calling models on those revisions would
    silently degrade to free-form; candidate 3 closes that gap.

    Returns ``None`` if none of the candidates yields an ``LLTokenizer`` (the
    caller degrades to today's free-form-then-parse behavior).
    """
    # Needs the runtime bridge (``llguidance.hf`` / ``LLTokenizer``), which is
    # detected independently of the builder layer (codex #558-PR3).
    if not HAS_LL_TOKENIZER:
        return None

    candidates = []
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None:
        candidates.append(inner)
    candidates.append(tokenizer)

    # Candidates 1 & 2: llguidance.hf.from_tokenizer (the fast path).
    for cand in candidates:
        if getattr(cand, "is_fast", True) is False:
            continue
        try:
            return _llguidance_hf.from_tokenizer(cand)
        except Exception:
            continue

    # Candidate 3: direct build from the serialized fast tokenizer. Probe both
    # the wrapper's inner tokenizer and the object itself for a
    # ``backend_tokenizer`` exposing ``to_str()``.
    for cand in candidates:
        backend = getattr(cand, "backend_tokenizer", None)
        to_str = getattr(backend, "to_str", None)
        if to_str is None:
            continue
        try:
            # No explicit ``n_vocab``: llguidance infers the true vocab from
            # the serialized tokenizer (a too-small override is rejected).
            return LLTokenizer(to_str())
        except Exception:
            continue

    logger.warning("tool-grammar: could not build an LLTokenizer for this model")
    return None


# One ``LLTokenizer`` per model tokenizer. Building it is a ~1s, vocab-scale
# operation (llguidance's own docstring flags it "expensive … should be
# cached"), so rebuilding it on every explicit tool request would add real
# latency + allocation churn to the async request path. The engine holds ONE
# tokenizer for its lifetime, so we memoize per tokenizer.
#
# The ``LLTokenizer`` is a Rust object that CANNOT be weak-referenced, and a
# ``TokenizerWrapper`` isn't reliably weak-referenceable either — so we cannot
# use a WeakKey/WeakValue dictionary. Instead we (a) stash the built tokenizer
# as a private attribute ON the tokenizer (primary cache; dies with the
# tokenizer, no leak) and (b) fall back to a ``WeakKeyDictionary`` on the
# tokenizer when it's weak-referenceable but rejects attribute assignment.
# Both are keyed on the live tokenizer object, so a swapped/reloaded model
# drops its stale entry automatically. A tokenizer that can't back an
# ``LLTokenizer`` is remembered via a per-tokenizer sentinel so we don't
# rebuild-and-fail on every request — but only AFTER a bounded retry budget, so
# a TRANSIENT conversion/resource failure doesn't permanently disable grammar
# enforcement for that tokenizer (codex #558-PR3 nit). Successful builds are
# cached immediately; failures accrue toward the budget and are only sealed as
# ``_LLTOKENIZER_UNAVAILABLE`` once exhausted.
_LLTOKENIZER_ATTR = "_rapid_mlx_lltokenizer"
_LLTOKENIZER_FAIL_ATTR = "_rapid_mlx_lltokenizer_fails"
_LLTOKENIZER_MAX_BUILD_ATTEMPTS = 3  # transient failures retried before sealing
_LLTOKENIZER_UNAVAILABLE = object()  # sentinel stored when build permanently fails
_LLTOKENIZER_WEAK_CACHE: "weakref.WeakKeyDictionary[Any, Any]" = (
    weakref.WeakKeyDictionary()
)
_LLTOKENIZER_FAIL_COUNTS: "weakref.WeakKeyDictionary[Any, int]" = (
    weakref.WeakKeyDictionary()
)
# Single-flight lock around the cache-MISS build. ``get_lltokenizer`` now runs
# inside ``asyncio.to_thread`` (chat route), so a cold traffic burst can land N
# concurrent misses for the SAME tokenizer and, without this, each thread would
# run the documented ~1s build (codex #558-PR3). We serialize the miss path and
# re-check the caches inside the lock so the build happens at most once. The
# lock is process-global (coarse) but only contended on the cold path — steady
# state hits the lock-free cache reads above.
_LLTOKENIZER_BUILD_LOCK = threading.Lock()


def _lltokenizer_cache_get(tokenizer: Any) -> tuple[bool, Any]:
    """Return ``(hit, value)`` from the attribute + weak caches (no build)."""
    cached = getattr(tokenizer, _LLTOKENIZER_ATTR, None)
    if cached is _LLTOKENIZER_UNAVAILABLE:
        return True, None
    if cached is not None:
        return True, cached
    try:
        wcached = _LLTOKENIZER_WEAK_CACHE.get(tokenizer)
    except TypeError:  # tokenizer not weak-referenceable
        wcached = None
    if wcached is _LLTOKENIZER_UNAVAILABLE:
        return True, None
    if wcached is not None:
        return True, wcached
    return False, None


def get_lltokenizer(tokenizer: Any) -> Any:
    """Return a cached ``LLTokenizer`` for ``tokenizer``, building once.

    Thin memoizing wrapper over :func:`build_lltokenizer` (see it for the
    candidate/fallback details). The ~1s build runs at most once per
    tokenizer even under concurrent cold misses. Returns ``None`` (and
    remembers it) when no ``LLTokenizer`` can be built.
    """
    if not HAS_LL_TOKENIZER or tokenizer is None:
        return None
    # Fast path: lock-free cache read.
    hit, value = _lltokenizer_cache_get(tokenizer)
    if hit:
        return value

    # Miss: serialize so concurrent to_thread callers don't all build (~1s each).
    with _LLTOKENIZER_BUILD_LOCK:
        # Re-check under the lock — another thread may have built it while we
        # waited.
        hit, value = _lltokenizer_cache_get(tokenizer)
        if hit:
            return value

        built = build_lltokenizer(tokenizer)
        if built is not None:
            # Success: cache immediately and clear any accrued failure count.
            _store_lltokenizer(tokenizer, built)
            try:
                _LLTOKENIZER_FAIL_COUNTS.pop(tokenizer, None)
            except TypeError:
                pass
            return built

        # Failure: only SEAL as permanently-unavailable once the retry budget is
        # exhausted, so a transient conversion/resource error is retried on the
        # next request instead of disabling grammar enforcement for the whole
        # process lifetime (codex #558-PR3 nit).
        fails = _bump_lltokenizer_fail_count(tokenizer)
        if fails >= _LLTOKENIZER_MAX_BUILD_ATTEMPTS:
            _store_lltokenizer(tokenizer, _LLTOKENIZER_UNAVAILABLE)
        return None


def _store_lltokenizer(tokenizer: Any, result: Any) -> None:
    """Persist ``result`` (an ``LLTokenizer`` or the UNAVAILABLE sentinel)."""
    try:
        setattr(tokenizer, _LLTOKENIZER_ATTR, result)
        return
    except Exception:
        pass
    try:
        _LLTOKENIZER_WEAK_CACHE[tokenizer] = result
    except TypeError:
        pass  # un-cacheable tokenizer: rebuild each call (correctness OK)


def _bump_lltokenizer_fail_count(tokenizer: Any) -> int:
    """Increment and return this tokenizer's accumulated build-failure count.

    Prefers a per-tokenizer attribute (dies with the tokenizer, no leak); falls
    back to a ``WeakKeyDictionary``; if neither is writable, treats every
    failure as the FIRST attempt (never seals) so an un-cacheable tokenizer
    keeps retrying rather than being wrongly disabled.
    """
    cur = getattr(tokenizer, _LLTOKENIZER_FAIL_ATTR, None)
    if isinstance(cur, int):
        nxt = cur + 1
        try:
            setattr(tokenizer, _LLTOKENIZER_FAIL_ATTR, nxt)
            return nxt
        except Exception:
            pass
    else:
        try:
            setattr(tokenizer, _LLTOKENIZER_FAIL_ATTR, 1)
            return 1
        except Exception:
            pass
    try:
        nxt = _LLTOKENIZER_FAIL_COUNTS.get(tokenizer, 0) + 1
        _LLTOKENIZER_FAIL_COUNTS[tokenizer] = nxt
        return nxt
    except TypeError:
        return 1  # un-cacheable: always look like the first attempt, keep retrying
