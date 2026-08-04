# SPDX-License-Identifier: Apache-2.0
"""Generation-time thinking-token budget (force-close ``</think>``).

A reasoning model emits ``<think>…</think>`` before its answer. Without a
budget it can think for an unbounded number of tokens (burning compute and
latency) before committing. This module enforces a per-request *thinking*
budget AT DECODE TIME: once the model has spent ``max_think_tokens`` inside the
thinking span, the next-token logits are OVERRIDDEN so the ONLY sampleable token
is the single ``</think>`` id — the model is forced to close the block and move
on to its answer. This is a real generation-time control, not the post-hoc
reasoning-text trim in :mod:`vllm_mlx.service.postprocessor` (which lets the
model keep generating and only relabels/injects the close marker in the OUTPUT
the client sees).

Three mature engines converge on exactly this mechanism, and we copy their
shape rather than invent one:

* **SGLang** ``ReasonerGrammarObject`` (``constrained/reasoner_grammar_backend.py``):
  a THINKING→GENERATION state machine; the instant ``tokens_in_think`` reaches
  the budget it rewrites the vocab mask to allow ONLY ``think_end_id``.
* **vLLM** ``ThinkingBudgetStateHolder`` (``v1/sample/thinking_budget_state.py``):
  fills the forced end-token logit to a DOMINATING value once the budget is
  exhausted (an override, not a nudge).
* **mlx-vlm** ``ThinkingBudgetCriteria`` (``mlx_vlm/utils.py``): resolves the
  end token by id (``tokenizer.encode(...)[-1]``), counts only inside the span,
  forces the close sequence.

All three: identify ``</think>`` by TOKEN ID (not string), count only tokens
inside the thinking span, treat ``budget < 0`` as unlimited, and force by making
the end token the sole choice. We express that force as an mlx-lm logits
processor so it drops into the existing per-step decode loop (the same slot the
``GrammarLogitsProcessor`` uses) with no decode-loop changes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SuppressTokensLogitsProcessor:
    """Statelessly prevent a small set of structural tokens from decoding."""

    def __init__(self, token_ids: list[int] | tuple[int, ...]) -> None:
        self._token_ids = tuple(dict.fromkeys(int(token_id) for token_id in token_ids))
        self._masks: dict[int, Any | None] = {}

    def __call__(self, _token_ids: Any, logits: Any) -> Any:
        import mlx.core as mx

        width = int(logits.shape[-1])
        if width not in self._masks:
            valid = tuple(
                token_id for token_id in self._token_ids if 0 <= token_id < width
            )
            if valid:
                suppressed = mx.array(valid)
                self._masks[width] = mx.any(
                    mx.arange(width)[:, None] == suppressed[None, :], axis=1
                )
            else:
                self._masks[width] = None
        mask = self._masks[width]
        if mask is None:
            return logits
        return mx.where(mask[None, :], -mx.inf, logits)


class ReasoningBudgetLogitsProcessor:
    """Force ``</think>`` once a per-request thinking-token budget is spent.

    mlx-lm contract: called every decode step as ``processor(token_ids,
    logits) -> logits`` where ``token_ids`` is the FULL cumulative sequence
    (prompt + everything generated so far). We baseline past the prompt on the
    first call and commit only the newly generated tail each step (O(new
    tokens), never O(n) — mirrors :class:`GrammarLogitsProcessor`), so the hot
    loop stays linear.

    Phase machine (mirrors SGLang ``ReasonerGrammarObject``):

    * THINKING — from the start of generation (when ``seeded_thinking``, i.e.
      the prompt prefilled ``<think>``) or from the first ``think_start_id`` in
      the generated span, until ``think_end_id`` is emitted. Every generated
      token in this span counts toward the budget, with two exemptions: the ONE
      opener that BEGINS the span is free (a seeded ``<think>`` never reaches the
      generated tail; an emitted opener transitions the machine in without being
      counted) and ``think_end_id`` is free. A ``think_start_id`` REPEATED while
      already thinking is NOT exempt — it counts, else a model looping ``<think>``
      could stall the budget forever (see ``_commit_tail``). At budget, force
      ``think_end_id``.
    * GENERATION — once ``think_end_id`` is seen, this processor is inert
      (returns logits unchanged) so a chained grammar/penalty processor owns
      the rest of the request. A latch makes this O(1) after the boundary.

    ``seeded_thinking`` selects between the TWO shapes a reasoning template can
    take: one that PREFILLS ``<think>`` into the prompt (seeded — the opener
    never appears in the generated tail, so counting starts at the first
    generated token) AND one whose model EMITS ``<think>`` as a generated token
    (not seeded — counting begins only AFTER that opener, and ``think_start_id``
    identifies it). The caller derives which shape applies from the TEMPLATE'S
    rendered generation prefix (``rendered_prompt_opens_think``), never from raw
    conversation content, so a ``<think>`` a user typed cannot mis-seed the
    budget. Either way the opener token is free, so a ``budget == 0`` request
    closes to a well-formed ``<think></think>`` rather than an unmatched
    ``</think>``.

    A request whose thinking is not enabled (``seeded_thinking`` false and no
    ``think_start_id`` ever appears) is never force-closed — the budget cannot
    fire on a non-thinking request (guards the vLLM #39130 footgun: never wait
    on / force a reasoning end that cannot occur).
    """

    def __init__(
        self,
        think_end_id: int,
        max_think_tokens: int,
        *,
        think_start_id: int | None = None,
        seeded_thinking: bool = True,
    ) -> None:
        self._think_end_id = int(think_end_id)
        self._budget = int(max_think_tokens)
        self._think_start_id = None if think_start_id is None else int(think_start_id)
        # Initial THINKING state, authoritative: True when the caller determined
        # (from the template's rendered generation prefix — see
        # ``rendered_prompt_opens_think``) that the prompt PREFILLS ``<think>``,
        # so counting starts at the first generated token. False for an emitting
        # template — counting waits until ``think_start_id`` appears in the tail.
        self._started = bool(seeded_thinking)
        # Cumulative-token bookkeeping (mirrors GrammarLogitsProcessor).
        self._prompt_len: int | None = None
        self._committed = 0
        self._think_count = 0
        self._ended = False
        # Cached OVERRIDE distribution (an mlx array), built lazily once the
        # vocab width is known; import mlx lazily so this module is importable
        # without the runtime for pure-logic unit tests of the phase machine.
        self._force_logits: Any = None
        self._force_width: int | None = None
        # One-shot observability latch: log exactly once, at the decode step the
        # budget first forces </think>, so operators can confirm the cap fired
        # without a per-step hot-loop log.
        self._force_logged = False
        # One-shot latch for the out-of-range </think> guard (see
        # ``_force_distribution``): a tokenizer/model vocab mismatch that puts
        # ``think_end_id`` outside the logits width disables the budget instead
        # of masking to an all -inf row.
        self._oob_logged = False

    # ---- phase machine (pure, unit-testable without mlx) -------------------

    def _commit_tail(self, token_ids: Any, n: int) -> None:
        """Advance the phase counters over newly generated tokens only."""
        tail = token_ids[self._committed : n]
        for t in tail:
            self._committed += 1
            tid = int(t)
            if self._ended:
                continue
            if not self._started:
                # Only the FIRST opener transitions us into the thinking span,
                # and it is free (never counted). Everything before the opener
                # (a model-emitted-``<think>`` template's preamble) is also not
                # counted — the span has not begun. See the class docstring.
                if self._think_start_id is not None and tid == self._think_start_id:
                    self._started = True
                continue
            # Inside the thinking span:
            if tid == self._think_end_id:
                self._ended = True
                # Release the cached OVERRIDE mask (codex R14): once </think> is
                # observed the span is closed and ``__call__`` short-circuits on
                # ``_ended`` for the whole remaining answer, so retaining the
                # full-vocab row would only pin hundreds of KB per request for no
                # further use. Whether the mask was ever built (budget forced the
                # close) or not (the model closed on its own), dropping the
                # reference here is safe and reclaims it promptly.
                self._force_logits = None
                self._force_width = None
                continue
            # Every other generated token counts — INCLUDING a repeated
            # ``<think>`` emitted while already thinking. Skipping it (the old
            # bug) would let a model loop the opener forever without advancing
            # the budget, violating the hard token cap (codex).
            self._think_count += 1

    def _phase(self, token_ids: Any) -> str:
        """Return ``"generation"`` | ``"force"`` | ``"free"``.

        ``force`` — budget exhausted, override all but ``</think>``.
        ``free``  — thinking, under budget, nothing to do.
        ``generation`` — inert (not thinking yet, or thinking has ended).
        """
        n = len(token_ids)
        if self._prompt_len is None:
            self._prompt_len = n
            self._committed = n
        if self._committed < n:
            self._commit_tail(token_ids, n)
        if self._ended or not self._started:
            return "generation"
        if self._budget >= 0 and self._think_count >= self._budget:
            return "force"
        return "free"

    # ---- mlx-lm logits-processor entry point -------------------------------

    def __call__(self, token_ids: Any, logits: Any) -> Any:
        # Unlimited budget is a pure no-op; skip even the tail walk so an unset
        # cap costs nothing. The ended-latch short-circuits the O(1) tail.
        if self._budget < 0 or self._ended:
            return logits
        if self._phase(token_ids) != "force":
            return logits
        return self._force_distribution(logits)

    # ---- forced-distribution construction (cached per vocab width) ---------

    def _force_distribution(self, logits: Any) -> Any:
        import mlx.core as mx

        width = logits.shape[-1]
        # CRASH-SAFETY net (codex R15): the budget is REQUIRED to pass a build-time
        # bounds check against the model's ACTUAL output-head WEIGHT width (see
        # build_reasoning_budget_processor's mandatory vocab_size check, fed by
        # routes/chat.py::_engine_output_vocab_size — now weight-derived ONLY, no
        # config/tokenizer fallback). Since ``logits = head(hidden)`` means
        # ``logits.shape[-1] == head.weight.shape[0]``, the width checked here is
        # the SAME number the builder validated ``</think>`` against, so for any
        # processor we install this guard is PROVABLY unreachable — there is no
        # config over-declaration or tokenizer over-count left that could make a
        # later width disagree with the head. It remains only as a defensive net:
        # were it ever reached, masking to an id no column matches would yield an
        # all -inf row (NaN after softmax → invalid sample / crash), so degrade
        # gracefully — latch inert and return the incoming logits unchanged. The
        # budget staying enforced does not depend on this branch: it can't fire.
        if not (0 <= self._think_end_id < width):
            if not self._oob_logged:
                self._oob_logged = True
                logger.warning(
                    "reasoning budget disabled: </think> id=%d is outside the "
                    "model's logits width %d (tokenizer/model vocab mismatch)",
                    self._think_end_id,
                    width,
                )
            self._ended = True  # latch inert: __call__ short-circuits hereafter
            return logits
        # Log the successful force EXACTLY once — AFTER the range check passes, so
        # an out-of-range disable never emits a false "forcing </think>" success
        # line immediately followed by a disable warning (codex R13 NIT).
        if not self._force_logged:
            self._force_logged = True
            logger.debug(
                "reasoning budget spent (%d think tokens) — forcing "
                "</think> (id=%d) at decode time",
                self._think_count,
                self._think_end_id,
            )
        dtype = logits.dtype
        if self._force_logits is None or self._force_width != width:
            vocab = mx.arange(width)
            # OVERRIDE (not additive): a fresh row that is -inf on every token
            # except </think>, which gets a finite value. Returned IN PLACE OF the
            # incoming logits so it DOMINATES any earlier processor in the chain —
            # critically, a grammar that drove </think> to -inf during the
            # thinking span would, under an additive mask, leave an all -inf row
            # (NaN after softmax). Overriding guarantees </think> is the sole
            # sampleable token (vLLM fills the end-token logit to a dominating
            # value for the same reason).
            self._force_logits = mx.where(
                vocab == self._think_end_id,
                mx.array(0.0),
                mx.array(-float("inf")),
            ).astype(dtype)
            self._force_width = width
        elif self._force_logits.dtype != dtype:
            self._force_logits = self._force_logits.astype(dtype)
        # PRESERVE the incoming shape ([..., vocab]). The mlx-lm logits-processor
        # contract requires the return to match the incoming logits shape (the
        # decode loop concatenates every row's processed logits — GrammarLogits-
        # Processor makes the same guarantee). A bare (vocab,) return collapses
        # the batch dim of a (1, vocab) input; the sampler then yields a 0-dim
        # scalar token and the NEXT decode step crashes on ``inputs[:, None]``
        # ("Too many indices for array with 0 dimensions"). Broadcasting the
        # cached row keeps the override cheap and shape-correct for (vocab,),
        # (1, vocab), or any (batch, vocab).
        return mx.broadcast_to(self._force_logits, logits.shape)


def reasoning_seed_state(
    generation_prefix: str, reasoning_parser_name: str | None
) -> str:
    """Classify the reasoning state at generation start from the ISOLATED
    template generation-prefix delta: ``"open"`` | ``"emit"`` | ``"ambiguous"``.

    ``generation_prefix`` MUST be the delta the two-render probe isolates
    (``routes/chat.py::_template_generation_prefix`` — the exact bytes the
    template appends for the new assistant turn), never the full prompt. Because
    the delta already excludes all conversation content, a ``<think>`` a USER
    typed cannot appear here, so we can PARSE the delta directly instead of only
    peeking at its suffix (codex R12: an ``endswith`` check mis-reads a template
    that prefills ``<think>`` followed by a non-whitespace preamble as NOT
    seeded, installing an unseeded processor that never starts yet suppresses the
    post-hoc cap):

      * ``"open"``  — the delta contains an UNCLOSED ``<think>`` (a start marker
        with no end marker after it): generation begins inside the thinking span
        → seed the budget (count from token 0).
      * ``"emit"``  — the delta has NEITHER marker (plain assistant header):
        the model itself will emit ``<think>`` → the budget waits for the opener.
      * ``"ambiguous"`` — a CLOSED pair (``<think>…</think>`` — thinking already
        finished in the prefix) or a stray end marker: the seed state cannot be
        proven, so the caller DECLINES the processor and keeps the post-hoc cap.
    """
    if not reasoning_parser_name:
        return "ambiguous"
    try:
        from ..reasoning import get_parser

        parser = get_parser(reasoning_parser_name)()
        start_marker = parser.start_token
        end_marker = parser.end_token
    except Exception:
        return "ambiguous"
    if not isinstance(start_marker, str) or not start_marker:
        return "ambiguous"
    text = generation_prefix or ""
    last_open = text.rfind(start_marker)
    last_close = (
        text.rfind(end_marker) if isinstance(end_marker, str) and end_marker else -1
    )
    if last_open == -1 and last_close == -1:
        return "emit"
    # An unclosed opener (last ``<think>`` sits AFTER any ``</think>``) → open.
    if last_open != -1 and last_open > last_close:
        return "open"
    # Closed pair or stray end marker → cannot prove the seed state → decline.
    return "ambiguous"


def rendered_prompt_opens_think(
    rendered_prompt: str, reasoning_parser_name: str | None
) -> bool:
    """True iff the ISOLATED generation-prefix delta opens an unclosed
    ``<think>`` span (thin bool wrapper over :func:`reasoning_seed_state`)."""
    return reasoning_seed_state(rendered_prompt, reasoning_parser_name) == "open"


def reasoning_stop_conflicts(stop: Any, reasoning_parser_name: str | None) -> bool:
    """True iff a client stop sequence would fire on the FORCED ``</think>``.

    The generation-time budget closes thinking by forcing ``</think>`` as a real
    decoded token. If the client ALSO listed ``</think>`` (or a substring /
    superstring the forced close would match) in ``stop``, that forced token
    trips the stop matcher and generation halts AT the reasoning boundary — the
    whole budget is spent on thinking and the ANSWER is never produced (codex).
    A model's reasoning end is a template-internal boundary, not a user stop; to
    keep that invariant on this degenerate config we DECLINE the generation-time
    budget and keep the post-hoc cap, which appends ``</think>`` to the FINAL
    text WITHOUT passing it through the decode-time stop matcher, so the answer
    still generates. Mirrors the tools / thinking-off opt-outs.

    We flag ANY stop the forced ``</think>`` can PARTICIPATE in matching — the
    forced token sits between the last reasoning tokens and the first answer
    tokens (``…R</think>A…``), and R/A are arbitrary model output, so the stop
    can be completed across EITHER boundary (codex R13: an earlier revision only
    checked the marker-prefix boundary and wrongly let ``x</think>y`` /
    ``</think>A`` through — the trailing text is supplied by the ANSWER tokens
    that follow the forced close, so those DO fire and truncate the answer).
    Over-declining is safe (falls back to the post-hoc cap, which still enforces
    the budget); only genuinely independent stops (``<|im_end|>``, ``\\n\\n``)
    stay on the decode-time path.
    """
    if not stop or not reasoning_parser_name:
        return False
    try:
        from ..reasoning import get_parser

        end_marker = get_parser(reasoning_parser_name)().end_token
    except Exception:
        return False
    if not isinstance(end_marker, str) or not end_marker:
        return False
    stops = [stop] if isinstance(stop, str) else stop
    return any(
        isinstance(s, str) and s and _overlaps_forced_marker(s, end_marker)
        for s in stops
    )


def _overlaps_forced_marker(s: str, marker: str) -> bool:
    """True iff the forced ``marker`` (``</think>``) can participate in matching
    stop ``s`` in the stream ``…R{marker}A…`` (R/A = arbitrary surrounding
    output). Any of four overlap alignments makes it a conflict:

      (a) ``s`` wholly INSIDE the marker (``think>``, ``>``) — the forced token
          alone matches it;
      (b) the marker wholly INSIDE ``s`` (``x</think>y``, ``</think>A``) — R and
          A supply the surrounding chars;
      (c) a SUFFIX of ``s`` is a PREFIX of the marker (``foo<``, ``bar</thi``) —
          preceding reasoning R supplies ``s``'s head, the forced token its tail;
      (d) a PREFIX of ``s`` is a SUFFIX of the marker (``>A``, ``think>foo``) —
          the forced token supplies ``s``'s head, the answer A its tail.
    """
    if s in marker or marker in s:
        return True
    span = range(1, min(len(s), len(marker)) + 1)
    # (c) suffix-of-s == prefix-of-marker  |  (d) prefix-of-s == suffix-of-marker
    return any(s[-k:] == marker[:k] for k in span) or any(
        s[:k] == marker[-k:] for k in span
    )


def build_budget_from_render(
    tokenizer: Any,
    reasoning_parser_name: str | None,
    max_think_tokens: int | None,
    rendered_prompt: str | None,
    *,
    vocab_size: int | None = None,
) -> ReasoningBudgetLogitsProcessor | None:
    """Build a generation-time budget processor from a RENDERED prompt, or None.

    ``rendered_prompt is None`` signals the caller could NOT render the prompt
    (e.g. an MLLM engine that rejects ``build_prompt``). In that case we install
    NO processor and return ``None`` so the caller retains the post-hoc reasoning
    cap — installing a non-seeded processor instead would be a silent footgun:
    it would never fire for a prefilling model (it never saw the ``<think>``
    open) YET its presence suppresses the post-hoc cap, leaving the request's
    reasoning budget COMPLETELY unenforced (codex). When the prompt rendered,
    seeding is derived from its trusted generation prefix.

    ``vocab_size`` is the model's output width (or a sound floor such as
    ``len(tokenizer)``); the builder requires it and returns ``None`` (post-hoc
    cap retained) when it is unknown OR ``</think>`` resolves outside it, so a
    processor never exists with an unverified force id (see below).
    """
    if rendered_prompt is None:
        return None
    state = reasoning_seed_state(rendered_prompt, reasoning_parser_name)
    if state == "ambiguous":
        # Seed state cannot be proven from the isolated delta (a closed
        # ``<think>…</think>`` pair, a stray end marker, or an unresolvable
        # parser). Decline so the post-hoc cap stays active rather than install a
        # processor that might never start yet suppress the cap (codex R12).
        return None
    return build_reasoning_budget_processor(
        tokenizer,
        reasoning_parser_name,
        max_think_tokens,
        seeded_thinking=(state == "open"),
        vocab_size=vocab_size,
    )


def _encode_single_special(tokenizer: Any, marker: str) -> int | None:
    """Return the single token id ``marker`` encodes to, or ``None``.

    Mirrors SGLang ``ReasonerGrammarBackend.__init__``: a reasoning-boundary
    marker is usable only when the tokenizer maps it to EXACTLY ONE token
    (otherwise the id comparison in the hot loop is meaningless). Any tokenizer
    that splits ``</think>`` into ordinary bytes (some channel-routed families)
    yields ``None`` and the caller opts out to the post-hoc reasoning cap.
    """
    if not marker:
        return None
    try:
        ids = tokenizer.encode(marker, add_special_tokens=False)
    except Exception:
        return None
    if not ids or len(ids) != 1:
        return None
    return int(ids[0])


def resolve_think_token_ids(
    tokenizer: Any, reasoning_parser_name: str | None
) -> tuple[int | None, int | None]:
    """Resolve ``(think_start_id, think_end_id)`` for the configured parser.

    Reads the reasoning parser's ``start_token`` / ``end_token`` (e.g.
    ``<think>`` / ``</think>``) and requires each to be a single special token
    on ``tokenizer`` (via ``resolve_reasoning_sentinels``, the exact gate the
    tool sentinels use) AND to encode to exactly one id. Returns ``None`` for
    either id that is unavailable — a missing ``think_end_id`` disables the
    generation-time budget for that model (caller falls back to the post-hoc
    cap; a tracked follow-up covers channel-routed families).
    """
    if tokenizer is None or not reasoning_parser_name:
        return (None, None)
    # Only trust markers that resolve_reasoning_sentinels already proved are
    # single special tokens on this tokenizer (reuse, don't reinvent the gate).
    from .tool_grammar import resolve_reasoning_sentinels

    sentinels = set(resolve_reasoning_sentinels(reasoning_parser_name, tokenizer))
    if not sentinels:
        return (None, None)
    try:
        from ..reasoning import get_parser

        parser = get_parser(reasoning_parser_name)()
        start_marker = getattr(parser, "start_token", None)
        end_marker = getattr(parser, "end_token", None)
    except Exception:
        return (None, None)
    start_id = (
        _encode_single_special(tokenizer, start_marker)
        if isinstance(start_marker, str) and start_marker in sentinels
        else None
    )
    end_id = (
        _encode_single_special(tokenizer, end_marker)
        if isinstance(end_marker, str) and end_marker in sentinels
        else None
    )
    return (start_id, end_id)


def build_reasoning_budget_processor(
    tokenizer: Any,
    reasoning_parser_name: str | None,
    max_think_tokens: int | None,
    *,
    seeded_thinking: bool,
    vocab_size: int | None = None,
) -> ReasoningBudgetLogitsProcessor | None:
    """Build a generation-time thinking-budget processor, or ``None``.

    Returns ``None`` (caller keeps the post-hoc reasoning cap) when there is no
    budget, or the model's ``</think>`` is not a single resolvable token. When
    ``seeded_thinking`` is False the ``<think>`` open id must also resolve, else
    the budget cannot know when thinking begins and opts out (guards the vLLM
    #39130 footgun: never force a reasoning-end that cannot occur).

    ``vocab_size`` (the model's output width, or a sound floor such as
    ``len(tokenizer)``) is REQUIRED and validated HERE, at BUILD time. Building a
    processor is what suppresses the post-hoc cap at the call site, so a
    processor may exist ONLY when its ``</think>`` id is build-time verified to be
    in range. Therefore we return ``None`` (post-hoc cap retained) when the width
    is UNKNOWN (``vocab_size is None`` — cannot verify) OR ``</think>`` lands
    outside it. Either way the budget is still enforced by the post-hoc cap; what
    we refuse is the unsound combination of "suppress the cap AND install a
    processor a decode-time out-of-range guard might silently disable, leaving the
    budget completely unenforced" (codex R11). The caller passes the model's
    ACTUAL output-head width (``_engine_output_vocab_size`` reads the real lm-head
    / tied-embedding weight rows, not a possibly-divergent declared config value —
    codex R12), so a built processor's ``</think>`` id is verified emittable by the
    head. For a real loaded engine the width is always determinable, so this never
    declines in practice; the decode-time guard in ``_force_distribution`` remains
    purely as a crash-safety net for the impossible case of a runtime logits width
    disagreeing with the inspected head weight.
    """
    if max_think_tokens is None or max_think_tokens < 0:
        return None
    start_id, end_id = resolve_think_token_ids(tokenizer, reasoning_parser_name)
    if end_id is None:
        return None
    if vocab_size is None or not (0 <= end_id < vocab_size):
        return None
    if not seeded_thinking:
        # An EMITTING template waits for the model to generate ``<think>`` — that
        # id must be resolvable AND emittable by the head, else the processor can
        # never start yet its presence suppresses the post-hoc cap (codex R13:
        # bounds-check ``start_id`` symmetrically with ``end_id``, not just for
        # None).
        if start_id is None or not (0 <= start_id < vocab_size):
            return None
    return ReasoningBudgetLogitsProcessor(
        end_id,
        max_think_tokens,
        think_start_id=start_id,
        seeded_thinking=seeded_thinking,
    )
