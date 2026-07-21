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
import threading
import weakref
from dataclasses import dataclass, field
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
    """

    begin: str
    end: str
    trigger: str
    sentinels: tuple[str, ...] = field(default=())


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

    if not reasoning_pair:
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
    else:
        # BALANCED + PREFILL-TOLERANT (codex #558-PR4 rounds 3-5). Two prefix
        # flavours:
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
    prefix_ref = "bal_prefix" if reasoning_pair else "TAG_TEXT"

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
        schema = json.dumps(params)
        begin_body = _emit_literal_with_sentinels(si.begin, si.sentinels)
        end_body = _emit_literal_with_sentinels(si.end, si.sentinels)
        lark += f"\ntag_{i}: {prefix_ref} {begin_body} %json {schema}"
        if end_body:
            lark += f" {end_body}"
        lark += "\n"
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
    ):
        self._lltok = lltokenizer
        self._matcher = LLMatcher(lltokenizer, grammar)
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
            return mx.concatenate([head, tail], axis=-1)
        if model_vocab < self._vocab:
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
            return apply_token_bitmask(logits, self._bitmask[..., :words])
        return apply_token_bitmask(logits, self._bitmask)

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
