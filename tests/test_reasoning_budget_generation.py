# SPDX-License-Identifier: Apache-2.0
"""Generation-time thinking-token budget (force-close ``</think>``).

Covers the decode-time budget introduced for #558: the
``ReasoningBudgetLogitsProcessor`` phase machine, the OVERRIDE force mask, the
``build_reasoning_budget_processor`` gate, and the chat-route
``_effective_posthoc_reasoning_cap`` single-mechanism selector. The processor is
the MLX-native expression of the same lever vLLM
(``ThinkingBudgetStateHolder``), SGLang (``ReasonerGrammarObject``) and mlx-vlm
(``ThinkingBudgetCriteria``) implement, so the assertions here pin the copied
contract, not an invented one.

The tokenizer→token-id resolution (``resolve_think_token_ids`` /
``resolve_reasoning_sentinels``) is reused upstream code, exercised elsewhere and
proven on a live model in the pilot; these tests monkeypatch it so the budget
logic is isolated from tokenizer specifics.
"""

from __future__ import annotations

import math

import mlx.core as mx

from vllm_mlx.api import reasoning_budget as rb
from vllm_mlx.api.reasoning_budget import (
    ReasoningBudgetLogitsProcessor,
    SuppressTokensLogitsProcessor,
    build_reasoning_budget_processor,
)

THINK_END = 99
THINK_START = 50


def test_suppress_tokens_processor_masks_only_valid_requested_ids():
    processor = SuppressTokensLogitsProcessor([2, 2, -1, 99])
    logits = processor(mx.array([1]), mx.zeros((1, 5)))
    mx.eval(logits)

    values = logits.tolist()[0]
    assert values[2] == -math.inf
    assert all(value == 0.0 for index, value in enumerate(values) if index != 2)


def test_suppress_tokens_processor_reuses_width_mask(monkeypatch):
    calls = 0
    original_arange = mx.arange

    def counted_arange(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(mx, "arange", counted_arange)
    processor = SuppressTokensLogitsProcessor([2])
    processor(mx.array([1]), mx.zeros((1, 5)))
    processor(mx.array([1, 2]), mx.zeros((1, 5)))
    assert calls == 1


def _feed(proc: ReasoningBudgetLogitsProcessor, seq: list[int]) -> str:
    """Feed one more token (``seq`` is the FULL cumulative sequence) and return
    the phase the processor is in for the NEXT step — mirrors the mlx-lm decode
    loop which calls the processor with the growing token list each step."""
    return proc._phase(seq)


# ─────────────────────────── phase machine ────────────────────────────


def test_seeded_counts_from_generation_start_and_forces_at_budget():
    # Prompt already opened <think> (seeded), so every generated token counts.
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 3, seeded_thinking=True)
    prompt = [1, 2, 3]  # baselined as prompt on first call
    assert _feed(proc, prompt) == "free"  # thinking, 0/3 spent
    assert _feed(proc, prompt + [10]) == "free"  # 1/3
    assert _feed(proc, prompt + [10, 11]) == "free"  # 2/3
    # 3/3 spent → budget exhausted → force </think>.
    assert _feed(proc, prompt + [10, 11, 12]) == "force"
    # The force latch persists every subsequent step until </think> is sampled.
    assert _feed(proc, prompt + [10, 11, 12, 13]) == "force"


def test_think_end_token_ends_span_and_makes_processor_inert():
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 3, seeded_thinking=True)
    prompt = [1, 2]
    _feed(proc, prompt)
    _feed(proc, prompt + [10])
    # Model emits </think> BEFORE the budget is hit → generation phase forever.
    assert _feed(proc, prompt + [10, THINK_END]) == "generation"
    assert _feed(proc, prompt + [10, THINK_END, 20, 21]) == "generation"
    assert proc._ended is True


def test_think_end_not_counted_toward_budget():
    # Budget of 2: two real think tokens then </think> should NOT force (the
    # </think> ends the span, it is not itself a budgeted token).
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 2, seeded_thinking=True)
    prompt = [1]
    _feed(proc, prompt)
    _feed(proc, prompt + [10])  # 1/2
    _feed(proc, prompt + [10, 11])  # 2/2 -> next step would force...
    # ...but the model closes here, so we land in generation, never forcing.
    assert _feed(proc, prompt + [10, 11, THINK_END]) == "generation"
    # The two real tokens counted; </think> itself did NOT (asserting the phase
    # alone would pass even if </think> were wrongly counted, since the span has
    # ended either way).
    assert proc._think_count == 2
    assert proc._ended is True


def test_model_emitted_opener_is_not_counted():
    # An EMITTING template (route derives seeded_thinking=False): the model
    # generates <think>. The opener, when it appears in the tail, transitions us
    # into the span and is itself free — not counted toward the budget.
    proc = ReasoningBudgetLogitsProcessor(
        THINK_END, 2, think_start_id=THINK_START, seeded_thinking=False
    )
    prompt = [1]
    _feed(proc, prompt)
    _feed(proc, prompt + [THINK_START])  # opener — NOT counted
    assert proc._think_count == 0
    _feed(proc, prompt + [THINK_START, 30])  # 1/2
    assert _feed(proc, prompt + [THINK_START, 30, 31]) == "force"  # 2/2


def test_budget_zero_seeded_closes_to_matched_pair():
    # budget=0 with a prefilled <think>: force </think> on the very first
    # generated token → a well-formed (prompt) <think></think>, never an
    # unmatched close.
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 0, seeded_thinking=True)
    assert _feed(proc, [1, 2, 3]) == "force"


def test_non_seeded_waits_for_think_start():
    proc = ReasoningBudgetLogitsProcessor(
        THINK_END, 2, think_start_id=THINK_START, seeded_thinking=False
    )
    prompt = [1, 2]
    # Not started (no <think> yet) → inert.
    assert _feed(proc, prompt) == "generation"
    assert _feed(proc, prompt + [7]) == "generation"
    # <think> appears in the generated stream → counting begins AFTER it.
    assert _feed(proc, prompt + [7, THINK_START]) == "free"
    assert _feed(proc, prompt + [7, THINK_START, 30]) == "free"  # 1/2
    assert _feed(proc, prompt + [7, THINK_START, 30, 31]) == "force"  # 2/2


# ───────────── emit-template + repeated-opener budget integrity ─────────


def test_emit_template_never_forces_before_opener():
    # An emitting template (route → seeded_thinking=False) with the SMALLEST
    # budget. The processor must NOT force </think> before the model opens
    # <think> — otherwise it injects an unmatched close (codex). Once the opener
    # is emitted, the very next real think token hits budget=1 → matched pair.
    proc = ReasoningBudgetLogitsProcessor(
        THINK_END, 1, think_start_id=THINK_START, seeded_thinking=False
    )
    prompt = [1, 2]  # no prefilled <think>
    assert _feed(proc, prompt) == "generation"  # not seeded → inert
    assert _feed(proc, prompt + [5]) == "generation"  # preamble, still no opener
    assert _feed(proc, prompt + [5, THINK_START]) == "free"  # model opens now
    assert _feed(proc, prompt + [5, THINK_START, 30]) == "force"  # 1/1 → matched


def test_repeated_opener_while_thinking_counts_toward_budget():
    # A model that emits <think> AGAIN while already inside the span must have
    # those repeats COUNTED — skipping every opener (the old bug) let the model
    # loop <think> forever without advancing the budget, breaking the hard cap.
    proc = ReasoningBudgetLogitsProcessor(
        THINK_END, 2, think_start_id=THINK_START, seeded_thinking=True
    )
    prompt = [1]  # seeded (prefilled): span already open
    _feed(proc, prompt)
    assert _feed(proc, prompt + [THINK_START]) == "free"  # repeat opener → 1/2
    assert proc._think_count == 1
    # A second repeated opener spends the last unit → force on the next step.
    assert _feed(proc, prompt + [THINK_START, THINK_START]) == "force"  # 2/2
    assert proc._think_count == 2


# ─────────────────────────── logit masks ──────────────────────────────


def test_force_mask_makes_think_end_the_argmax():
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 1, seeded_thinking=True)
    # Step 1: [1] is baselined as prompt; 0/1 spent → "free", logits untouched.
    _ = proc([1], mx.zeros((128,)))
    # Step 2: one generated think token → 1/1 spent → force on THIS call.
    logits = mx.random.normal((128,))
    out = proc([1, 10], logits)
    assert int(mx.argmax(out).item()) == THINK_END
    # </think> keeps a finite value; every other column is -inf.
    assert math.isfinite(out[THINK_END].item())
    masked = mx.sum((out == -math.inf).astype(mx.int32)).item()
    assert masked == 127  # all but the single kept column


def test_force_mask_released_when_think_end_observed():
    # codex R14 nit: the cached OVERRIDE row must be dropped the moment </think>
    # is observed, not pinned through the whole answer. Force once (mask built),
    # then feed </think> (span ends) and assert the reference is cleared.
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 1, seeded_thinking=True)
    _ = proc([1], mx.zeros((128,)))  # baseline, 0/1
    _ = proc([1, 10], mx.random.normal((128,)))  # 1/1 → force → mask allocated
    assert proc._force_logits is not None  # mask cached during forcing
    # Next step the sampler picked </think>; the processor sees it in the tail.
    out = proc([1, 10, THINK_END], mx.random.normal((128,)))
    assert proc._ended is True
    assert proc._force_logits is None  # released promptly, not retained
    assert proc._force_width is None
    # And it stays inert / a no-op thereafter (returns logits unchanged).
    logits = mx.random.normal((128,))
    assert bool(mx.all(proc([1, 10, THINK_END, 20], logits) == logits))


def test_bounded_reasoning_suppresses_reopen_after_public_content():
    proc = ReasoningBudgetLogitsProcessor(
        THINK_END, 1, think_start_id=THINK_START, seeded_thinking=True
    )
    _ = proc([1], mx.zeros((128,)))
    _ = proc([1, 10], mx.zeros((128,)))
    _ = proc([1, 10, THINK_END], mx.zeros((128,)))

    logits = mx.zeros((128,))
    out = proc([1, 10, THINK_END, 20], logits)
    assert out[0, THINK_START].item() == -math.inf
    assert out[0, THINK_END].item() == 0.0
    assert out[0, 20].item() == 0.0


def test_invalid_reopen_id_is_ignored_against_runtime_logits_width():
    proc = ReasoningBudgetLogitsProcessor(
        THINK_END, 1, think_start_id=-1, seeded_thinking=True
    )
    _ = proc([1], mx.zeros((128,)))
    _ = proc([1, 10], mx.zeros((128,)))
    _ = proc([1, 10, THINK_END], mx.zeros((128,)))

    logits = mx.zeros((128,))
    out = proc([1, 10, THINK_END, 20], logits)
    assert bool(mx.all(out == logits))


def test_force_override_dominates_a_prior_neg_inf_at_think_end():
    # The load-bearing #1 fix: the force is an OVERRIDE, not an additive mask.
    # Simulate a chained grammar that already drove </think> (and much else) to
    # -inf during the thinking span. An additive mask would leave an all -inf row
    # (NaN after softmax); the override must still make </think> the sole finite,
    # sampleable token.
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 1, seeded_thinking=True)
    _ = proc([1], mx.zeros((128,)))
    neg = float("-inf")
    grammar_masked = mx.full((128,), neg)  # a hostile prior processor: all -inf
    out = proc([1, 10], grammar_masked)
    assert math.isfinite(out[THINK_END].item()), "</think> must be forced finite"
    assert int(mx.argmax(out).item()) == THINK_END
    finite = mx.sum((out != -math.inf).astype(mx.int32)).item()
    assert finite == 1, "exactly one sampleable token"


def test_force_preserves_batched_2d_shape():
    # Regression for the pilot crash: mlx-lm's decode loop hands the processor a
    # (1, vocab) row and concatenates each uid's processed logits, so the return
    # MUST keep the leading batch dim. A bare (vocab,) return collapses it; the
    # sampler then yields a 0-dim scalar token and the next step crashes on
    # inputs[:, None] ("Too many indices for array with 0 dimensions").
    # token_ids is the 1-D cumulative sequence (as mlx-lm hands it, same as the
    # GrammarLogitsProcessor contract); logits is the (1, vocab) row.
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 1, seeded_thinking=True)
    _ = proc([1], mx.zeros((1, 128)))  # baseline, 0/1
    out = proc([1, 10], mx.random.normal((1, 128)))
    assert out.shape == (1, 128), "must preserve the incoming (1, vocab) shape"
    assert int(mx.argmax(out[0]).item()) == THINK_END
    assert math.isfinite(out[0, THINK_END].item())


def test_unlimited_budget_is_identity_noop():
    # budget < 0 → the exact same array back (no work).
    proc = ReasoningBudgetLogitsProcessor(THINK_END, -1, seeded_thinking=True)
    logits = mx.random.normal((64,))
    out = proc([1, 2, 3], logits)
    assert out is logits


def test_ended_latch_short_circuits_call():
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 1, seeded_thinking=True)
    _ = proc([1], mx.zeros((32,)))
    _ = proc([1, THINK_END], mx.zeros((32,)))  # closes the span
    logits = mx.random.normal((32,))
    out = proc([1, THINK_END, 5], logits)
    assert out is logits  # inert after </think>


# ─────────────────────────── builder gate ─────────────────────────────


class _StubTokenizer:
    """Placeholder — the real token-id resolution is monkeypatched out below.
    Since R14 the build-time vocab width comes from the model head/config, not
    ``len(tokenizer)`` (added specials make that vacuous), so callers pass an
    explicit ``vocab_size`` and ``_FakeEngine`` exposes ``_model.vocab_size``;
    this ``__len__`` is retained only so the stub quacks like a tokenizer."""

    def __len__(self):
        return 1000


def _patch_ids(monkeypatch, start_id, end_id):
    monkeypatch.setattr(
        rb, "resolve_think_token_ids", lambda tok, name: (start_id, end_id)
    )


def test_build_none_when_no_budget(monkeypatch):
    _patch_ids(monkeypatch, THINK_START, THINK_END)
    assert (
        build_reasoning_budget_processor(
            _StubTokenizer(), "qwen3", None, seeded_thinking=True
        )
        is None
    )


def test_build_none_when_budget_negative(monkeypatch):
    _patch_ids(monkeypatch, THINK_START, THINK_END)
    assert (
        build_reasoning_budget_processor(
            _StubTokenizer(), "qwen3", -1, seeded_thinking=True
        )
        is None
    )


def test_build_none_when_end_token_unresolved(monkeypatch):
    # Channel-routed family: </think> is not a single token → opt out to the
    # post-hoc cap. This is the gate that keeps gpt-oss on the old path.
    _patch_ids(monkeypatch, None, None)
    assert (
        build_reasoning_budget_processor(
            _StubTokenizer(), "harmony", 64, seeded_thinking=True
        )
        is None
    )


def test_build_none_when_non_seeded_and_no_start(monkeypatch):
    # Cannot know when thinking begins → never risk force-closing a
    # non-thinking request (vLLM #39130 footgun).
    _patch_ids(monkeypatch, None, THINK_END)
    assert (
        build_reasoning_budget_processor(
            _StubTokenizer(), "qwen3", 64, seeded_thinking=False
        )
        is None
    )


def test_build_ok_wires_ids_and_budget(monkeypatch):
    _patch_ids(monkeypatch, THINK_START, THINK_END)
    proc = build_reasoning_budget_processor(
        _StubTokenizer(), "qwen3", 128, seeded_thinking=True, vocab_size=1000
    )
    assert isinstance(proc, ReasoningBudgetLogitsProcessor)
    assert proc._think_end_id == THINK_END
    assert proc._budget == 128
    assert proc._started is True


# ─────────────── seeding derived from the template prefix ──────────────


def test_reasoning_seed_state_open_for_prefill():
    from vllm_mlx.api.reasoning_budget import reasoning_seed_state

    # Template prefilled <think> as the generation prefix → open span, seed.
    assert reasoning_seed_state("<|im_start|>assistant\n<think>\n", "qwen3") == "open"


def test_reasoning_seed_state_open_for_prefill_with_preamble():
    # codex R12: a template that prefills <think> then a NON-whitespace preamble
    # is STILL an open span — the old endswith("<think>") check wrongly read this
    # as not-seeded (unseeded processor never starts, yet suppresses post-hoc).
    # Parsing the isolated delta detects the unclosed opener regardless of tail.
    from vllm_mlx.api.reasoning_budget import reasoning_seed_state

    delta = "<|im_start|>assistant\n<think>\nLet me reason:"
    assert reasoning_seed_state(delta, "qwen3") == "open"


def test_reasoning_seed_state_emit_when_no_markers():
    from vllm_mlx.api.reasoning_budget import reasoning_seed_state

    # Plain assistant header, no prefilled opener → the model emits <think>.
    assert reasoning_seed_state("<|im_start|>assistant\n", "qwen3") == "emit"


def test_reasoning_seed_state_ambiguous_for_closed_pair():
    from vllm_mlx.api.reasoning_budget import reasoning_seed_state

    # A CLOSED <think></think> in the prefix (thinking already finished) → the
    # seed state cannot be proven → ambiguous → caller declines (post-hoc cap).
    delta = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert reasoning_seed_state(delta, "qwen3") == "ambiguous"


def test_reasoning_seed_state_ambiguous_without_parser():
    from vllm_mlx.api.reasoning_budget import reasoning_seed_state

    assert reasoning_seed_state("<think>", None) == "ambiguous"  # no parser
    assert reasoning_seed_state("", "qwen3") == "emit"  # empty → no markers → emit


def test_rendered_prompt_opens_think_bool_wrapper():
    # The bool wrapper is True only for the "open" state.
    from vllm_mlx.api.reasoning_budget import rendered_prompt_opens_think

    assert rendered_prompt_opens_think("<|im_start|>assistant\n<think>\n", "qwen3")
    assert not rendered_prompt_opens_think("<|im_start|>assistant\n", "qwen3")
    assert not rendered_prompt_opens_think("<think>", None)


def test_build_budget_from_render_none_render_installs_nothing(monkeypatch):
    # codex: when the prompt could NOT be rendered (None), install NO processor
    # so the caller retains the post-hoc cap — never a non-seeded processor that
    # silently disables the cap AND never fires.
    from vllm_mlx.api import reasoning_budget as rb2

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    assert rb2.build_budget_from_render(_StubTokenizer(), "qwen3", 64, None) is None


def test_build_budget_from_render_seeds_from_prefill_suffix(monkeypatch):
    from vllm_mlx.api import reasoning_budget as rb2

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    proc = rb2.build_budget_from_render(
        _StubTokenizer(),
        "qwen3",
        64,
        "<|im_start|>assistant\n<think>\n",
        vocab_size=1000,
    )
    assert isinstance(proc, ReasoningBudgetLogitsProcessor)
    assert proc._started is True  # prefill suffix → seeded


def test_build_budget_from_render_not_seeded_for_emit_suffix(monkeypatch):
    from vllm_mlx.api import reasoning_budget as rb2

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    proc = rb2.build_budget_from_render(
        _StubTokenizer(), "qwen3", 64, "<|im_start|>assistant\n", vocab_size=1000
    )
    assert isinstance(proc, ReasoningBudgetLogitsProcessor)
    assert proc._started is False  # no prefilled opener → wait for emitted one


def test_build_budget_from_render_declines_on_ambiguous_closed_pair(monkeypatch):
    # codex R12: a CLOSED <think></think> prefix → ambiguous seed state → install
    # NO processor (retain the post-hoc cap) rather than one that never fires yet
    # suppresses the cap.
    from vllm_mlx.api import reasoning_budget as rb2

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    assert (
        rb2.build_budget_from_render(
            _StubTokenizer(),
            "qwen3",
            64,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            vocab_size=1000,
        )
        is None
    )


# ──────────────────── stop-sequence conflict opt-out ──────────────────


def test_reasoning_stop_conflicts_exact_end_marker():
    from vllm_mlx.api.reasoning_budget import reasoning_stop_conflicts

    # Client listed </think> as a stop → forcing it would halt at the boundary.
    assert reasoning_stop_conflicts(["</think>"], "qwen3") is True


def test_reasoning_stop_conflicts_substring_is_conflict():
    from vllm_mlx.api.reasoning_budget import reasoning_stop_conflicts

    # Stop is a substring of </think> ("think>") — the forced </think> contains
    # it, so it fires at the boundary → conflict.
    assert reasoning_stop_conflicts(["think>"], "qwen3") is True
    # Accepts a bare string too, not only a list.
    assert reasoning_stop_conflicts("</think>", "qwen3") is True


def test_reasoning_stop_conflicts_marker_inside_stop_is_conflict():
    from vllm_mlx.api.reasoning_budget import reasoning_stop_conflicts

    # codex R13: a stop that CONTAINS the marker mid-string ("x</think>y") or with
    # a trailing suffix ("</think>A") IS a conflict — the forced </think> sits
    # between reasoning and answer tokens (…R</think>A…), so the ANSWER tokens
    # supply the trailing chars and the stop fires, truncating the answer. An
    # earlier revision wrongly left "x</think>y" enabled (it assumed the forced
    # </think> was the last token; answer tokens follow it).
    assert reasoning_stop_conflicts(["x</think>y"], "qwen3") is True
    assert reasoning_stop_conflicts(["</think>A"], "qwen3") is True


def test_reasoning_stop_conflicts_cross_boundary_both_sides():
    from vllm_mlx.api.reasoning_budget import reasoning_stop_conflicts

    # (c) stop ENDS in a marker prefix — preceding reasoning + forced token:
    assert reasoning_stop_conflicts(["x</think>"], "qwen3") is True
    assert reasoning_stop_conflicts(["foo<"], "qwen3") is True
    assert reasoning_stop_conflicts(["bar</thi"], "qwen3") is True
    # (d) stop STARTS with a marker suffix — forced token + following answer:
    assert reasoning_stop_conflicts([">A"], "qwen3") is True
    assert reasoning_stop_conflicts(["think>foo"], "qwen3") is True


def test_reasoning_stop_conflicts_false_for_unrelated_stop():
    from vllm_mlx.api.reasoning_budget import reasoning_stop_conflicts

    assert reasoning_stop_conflicts(["<|im_end|>", "\n\n"], "qwen3") is False
    assert reasoning_stop_conflicts([], "qwen3") is False
    assert reasoning_stop_conflicts(None, "qwen3") is False
    # No parser → cannot resolve the end marker → no conflict.
    assert reasoning_stop_conflicts(["</think>"], None) is False


# ──────────────────── single-mechanism route selector ─────────────────


def test_effective_posthoc_cap_selector():
    from vllm_mlx.routes.chat import _effective_posthoc_reasoning_cap

    class _Req:
        reasoning_max_tokens = 256

    req = _Req()
    # No generation-time budget in the kwargs → post-hoc cap owns the request.
    assert _effective_posthoc_reasoning_cap({}, req) == 256
    # A budget processor present → post-hoc cap MUST be suppressed (return None)
    # so the two mechanisms never both run.
    active = {"reasoning_budget_logits_processor": object()}
    assert _effective_posthoc_reasoning_cap(active, req) is None
    # A None-valued key is treated as "no processor" (defensive).
    assert (
        _effective_posthoc_reasoning_cap(
            {"reasoning_budget_logits_processor": None}, req
        )
        == 256
    )


# ─────────── template generation-prefix seeding (codex #2 immunity) ─────────


class _FakeEngine:
    """Minimal text engine: renders the conversation body, then appends a fixed
    template generation prefix ONLY when ``add_generation_prompt`` is True. The
    seed decision diffs the two renders (``full`` − ``base``) to isolate that
    prefix — proving seeding derives from the TEMPLATE's delta, never raw
    conversation content."""

    class _HeadWeight:
        # (vocab, hidden) — shape[0] is the true logits width the R15 weight-only
        # _engine_output_vocab_size reads. 1000 > THINK_END=99 so the build-time
        # </think> bounds check admits.
        shape = (1000, 4)

    class _Embed:
        weight = None  # set in __init__ to a _HeadWeight instance

    class _Model:
        embed_tokens = None  # set in __init__ to an _Embed instance

    def __init__(self, gen_prefix):
        self._gen = gen_prefix
        self.tokenizer = _StubTokenizer()
        embed = _FakeEngine._Embed()
        embed.weight = _FakeEngine._HeadWeight()
        model = _FakeEngine._Model()
        model.embed_tokens = embed
        self._model = model

    def build_prompt(
        self, messages, tools=None, enable_thinking=None, add_generation_prompt=True
    ):
        body = "".join(str(m.get("content", "")) for m in messages)
        base = f"<|im_start|>user\n{body}<|im_end|>\n"
        return base + self._gen if add_generation_prompt else base


def test_template_generation_prefix_isolates_prefill():
    from vllm_mlx.routes.chat import _template_generation_prefix

    eng = _FakeEngine("<|im_start|>assistant\n<think>\n")  # template opens <think>
    delta = _template_generation_prefix(
        eng, [{"role": "user", "content": "hi"}], None, True
    )
    assert delta.rstrip().endswith("<think>")


def test_template_generation_prefix_isolates_emit():
    from vllm_mlx.routes.chat import _template_generation_prefix

    eng = _FakeEngine("<|im_start|>assistant\n")  # emit template, no prefill
    delta = _template_generation_prefix(
        eng, [{"role": "user", "content": "hi"}], None, True
    )
    assert not delta.rstrip().endswith("<think>")


def test_template_generation_prefix_immune_to_user_typed_think():
    # codex #2: a user whose message ENDS in <think> must NOT falsely seed the
    # budget when the template appends no opener. The user's content lives in
    # BOTH renders (full and base), so it cancels out of the delta.
    from vllm_mlx.routes.chat import _template_generation_prefix

    eng = _FakeEngine("<|im_start|>assistant\n")  # NO prefilled <think>
    delta = _template_generation_prefix(
        eng, [{"role": "user", "content": "please <think>"}], None, True
    )
    assert "<think>" not in delta  # user marker isolated out of the delta
    assert not delta.rstrip().endswith("<think>")


def test_template_generation_prefix_none_for_empty_messages():
    # codex R10 #3: an empty message list renders nothing — a prefill template
    # must NOT be mistaken for an emit one. Decline (retain the post-hoc cap).
    from vllm_mlx.routes.chat import _template_generation_prefix

    eng = _FakeEngine("<|im_start|>assistant\n<think>\n")
    assert _template_generation_prefix(eng, [], None, True) is None


def test_template_generation_prefix_none_for_non_string_content():
    from vllm_mlx.routes.chat import _template_generation_prefix

    eng = _FakeEngine("<|im_start|>assistant\n")
    delta = _template_generation_prefix(
        eng,
        [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
        None,
        True,
    )
    assert delta is None  # non-str content → retain post-hoc cap


def test_template_generation_prefix_none_when_full_not_startswith_base():
    # If toggling add_generation_prompt RESTRUCTURES the render (the no-gen-prompt
    # render is not a prefix of the gen-prompt one), the boundary can't be
    # isolated cleanly → decline rather than risk a wrong seed (codex R10 #1).
    from vllm_mlx.routes.chat import _template_generation_prefix

    class _Restructure:
        tokenizer = _StubTokenizer()

        def build_prompt(
            self,
            messages,
            tools=None,
            enable_thinking=None,
            add_generation_prompt=True,
        ):
            # base is NOT a prefix of full (leading char differs).
            return "A<think>" if add_generation_prompt else "Bxxxx"

    delta = _template_generation_prefix(
        _Restructure(), [{"role": "user", "content": "hi"}], None, True
    )
    assert delta is None


# ─────────── route builder gates (thinking / tools / stop / seeding) ────────


class _FakeCfg:
    def __init__(self, parser="qwen3", model="mlx-community/Qwen3-0.6B-4bit"):
        self.reasoning_parser_name = parser
        self.model_path = model
        self.model_name = model


class _FakeReq:
    def __init__(self, budget=64, tools=None, stop=None):
        self.reasoning_max_tokens = budget
        self.tools = tools
        self.stop = stop


_MSGS = [{"role": "user", "content": "hi"}]


def test_build_reasoning_budget_processor_stop_conflict_returns_none(monkeypatch):
    from vllm_mlx.routes import chat as chatmod

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    proc = chatmod._build_reasoning_budget_processor(
        _FakeEngine("<|im_start|>assistant\n<think>\n"),
        _FakeReq(budget=64, stop=["</think>"]),
        _FakeCfg(),
        _MSGS,
        True,
    )
    assert proc is None


def test_build_reasoning_budget_processor_tools_returns_none(monkeypatch):
    from vllm_mlx.routes import chat as chatmod

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    proc = chatmod._build_reasoning_budget_processor(
        _FakeEngine("<|im_start|>assistant\n<think>\n"),
        _FakeReq(budget=64, tools=[{"type": "function"}]),
        _FakeCfg(),
        _MSGS,
        True,
    )
    assert proc is None


def test_build_reasoning_budget_processor_thinking_off_returns_none(monkeypatch):
    from vllm_mlx.routes import chat as chatmod

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    # resolved_thinking=False → _effective_enable_thinking is False → opt out.
    proc = chatmod._build_reasoning_budget_processor(
        _FakeEngine("<|im_start|>assistant\n<think>\n"),
        _FakeReq(budget=64),
        _FakeCfg(),
        _MSGS,
        False,
    )
    assert proc is None


def test_build_reasoning_budget_processor_seeds_from_template_prefill(monkeypatch):
    from vllm_mlx.routes import chat as chatmod

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    proc = chatmod._build_reasoning_budget_processor(
        _FakeEngine("<|im_start|>assistant\n<think>\n"),  # prefill
        _FakeReq(budget=64),
        _FakeCfg(),
        _MSGS,
        True,
    )
    assert isinstance(proc, ReasoningBudgetLogitsProcessor)
    assert proc._started is True  # seeded from the TEMPLATE's prefilled <think>


def test_build_reasoning_budget_processor_emit_template_not_seeded(monkeypatch):
    from vllm_mlx.routes import chat as chatmod

    _patch_ids(monkeypatch, THINK_START, THINK_END)
    proc = chatmod._build_reasoning_budget_processor(
        _FakeEngine("<|im_start|>assistant\n"),  # emit template, no prefill
        _FakeReq(budget=64),
        _FakeCfg(),
        _MSGS,
        True,
    )
    assert isinstance(proc, ReasoningBudgetLogitsProcessor)
    assert proc._started is False  # waits for the model-emitted opener


# ─────────── out-of-range </think> guard (codex #3 defensive NIT) ───────────


def test_force_distribution_disables_on_out_of_range_think_end():
    # A tokenizer/model vocab mismatch puts </think> beyond the logits width.
    # The force step must DISABLE the budget (return logits unchanged, latch
    # inert) rather than mask to an all -inf row that yields an invalid sample.
    proc = ReasoningBudgetLogitsProcessor(200, 0, seeded_thinking=True)  # end_id=200
    logits = mx.zeros((1, 100))  # width 100 < 200 → out of range
    out = proc([1, 2, 3], logits)  # budget=0 + seeded → force phase → guard trips
    assert out.shape == (1, 100)
    assert bool(mx.all(mx.isfinite(out)))  # NOT all -inf — unchanged
    assert proc._ended is True  # latched inert
    # The "forcing </think>" success line must NOT have been emitted: the guard
    # trips FIRST, so no false success-then-disable log pair (codex R13 #4).
    assert proc._force_logged is False
    # subsequent steps are no-ops (never crash on the bad id again)
    out2 = proc([1, 2, 3, 4], logits)
    assert bool(mx.all(mx.isfinite(out2)))
    assert proc._force_logged is False


# ─────────── build-time </think> vocab bounds check (codex #1) ──────────────


def test_build_none_when_think_end_exceeds_vocab(monkeypatch):
    # </think> beyond the model's vocab → decline at BUILD time so the post-hoc
    # cap stays active (no decode-time-only disable that would leave the budget
    # unenforced after the cap was already suppressed).
    _patch_ids(monkeypatch, THINK_START, 500)
    assert (
        build_reasoning_budget_processor(
            _StubTokenizer(), "qwen3", 64, seeded_thinking=True, vocab_size=100
        )
        is None
    )


def test_build_ok_when_think_end_within_vocab(monkeypatch):
    _patch_ids(monkeypatch, THINK_START, 99)
    proc = build_reasoning_budget_processor(
        _StubTokenizer(), "qwen3", 64, seeded_thinking=True, vocab_size=100
    )
    assert isinstance(proc, ReasoningBudgetLogitsProcessor)


def test_build_none_when_non_seeded_start_exceeds_vocab(monkeypatch):
    # NOT seeded → counting only begins once <think> (start_id) appears in the
    # generated tail. An out-of-range start_id (500 ≥ width 100) can never match
    # any sampled token (all < width), so the budget would silently NEVER fire —
    # yet the processor's mere existence suppresses the post-hoc cap, leaving the
    # request wholly unbudgeted. Decline at BUILD time, symmetric with the
    # </think> end-id bounds check (codex R13 #2). </think> itself IS in range
    # here (99 < 100), isolating start_id as the sole cause.
    _patch_ids(monkeypatch, 500, 99)
    assert (
        build_reasoning_budget_processor(
            _StubTokenizer(), "qwen3", 64, seeded_thinking=False, vocab_size=100
        )
        is None
    )


def test_build_none_when_vocab_size_unknown(monkeypatch):
    # codex R11: an UNKNOWN width cannot be verified → decline so the post-hoc cap
    # stays (a processor must never exist with an unverified force id, since its
    # mere existence suppresses the cap). For real engines the width is always
    # determinable (len(tokenizer)), so this residual never triggers in practice.
    _patch_ids(monkeypatch, THINK_START, THINK_END)
    assert (
        build_reasoning_budget_processor(
            _StubTokenizer(), "qwen3", 64, seeded_thinking=True, vocab_size=None
        )
        is None
    )


# ─────────── content-dependent template seeding (codex R10 #1) ──────────────


class _ContentDepEngine:
    """A template whose generation prefix DEPENDS on the last message: it opens
    <think> only when the content contains "math". The two-render diff always
    uses the REAL content in both renders, so the delta reflects the true prefix
    for THIS request — no content substitution, no mis-seed."""

    tokenizer = _StubTokenizer()

    def build_prompt(
        self, messages, tools=None, enable_thinking=None, add_generation_prompt=True
    ):
        body = "".join(str(m.get("content", "")) for m in messages)
        gen = (
            "<|im_start|>assistant\n<think>\n"
            if "math" in body
            else "<|im_start|>assistant\n"
        )
        base = f"<|im_start|>user\n{body}<|im_end|>\n"
        return base + gen if add_generation_prompt else base


def test_template_generation_prefix_content_dependent_seeds_correctly():
    from vllm_mlx.routes.chat import _template_generation_prefix

    # Content has "math" → the template DOES prefill <think> for this request.
    # The old sentinel probe (content replaced) lost "math" and mis-declined; the
    # two-render keeps the real content in both renders → seeds correctly.
    delta = _template_generation_prefix(
        _ContentDepEngine(), [{"role": "user", "content": "do math"}], None, True
    )
    assert delta is not None
    assert delta.rstrip().endswith("<think>")


def test_template_generation_prefix_content_dependent_emit_not_seeded():
    from vllm_mlx.routes.chat import _template_generation_prefix

    # Content lacks "math" → the SAME template emits (no prefill) for this
    # request; the delta must not end in <think>.
    delta = _template_generation_prefix(
        _ContentDepEngine(), [{"role": "user", "content": "say hi"}], None, True
    )
    assert delta is not None
    assert not delta.rstrip().endswith("<think>")


def test_engine_output_vocab_size_declines_on_config_only():
    # codex R15: a declared config vocab_size is NOT trusted — a padded/over-
    # declared vocab can exceed the true head width, which is exactly the mismatch
    # that would trip the decode-time guard. With no inspectable head WEIGHT,
    # decline (None) so the post-hoc cap stays. Only the weight-derived width (==
    # the decode logits width) may admit the force id.
    from vllm_mlx.routes.chat import _engine_output_vocab_size

    class _M:
        vocab_size = 12345  # declared only — no head weight → must be ignored

    class _E:
        _model = _M()
        tokenizer = None

    assert _engine_output_vocab_size(_E()) is None


def test_engine_output_vocab_size_declines_when_only_len_tokenizer():
    # codex R14: len(tokenizer) is NOT a sound width for the </think> bounds
    # check — it counts added specials, so </think> (an added special) is always
    # within it, making the check vacuous. With no inspectable head/config width,
    # decline (None) so the post-hoc cap stays rather than admit an id the output
    # head may not be able to emit.
    from vllm_mlx.routes.chat import _engine_output_vocab_size

    class _Tok:
        def __len__(self):
            return 777

    class _E:
        _model = None
        tokenizer = _Tok()

    assert _engine_output_vocab_size(_E()) is None


def test_engine_output_vocab_size_prefers_actual_weight_shape():
    # codex R12: the ACTUAL output-head width (weight rows) is authoritative and
    # must win over a (possibly divergent) declared config vocab_size. Mirrors a
    # tied-embedding model (qwen3): logits come from embed_tokens.as_linear, so
    # embed_tokens.weight.shape[0] is the true logits width.
    from vllm_mlx.routes.chat import _engine_output_vocab_size

    class _W:
        shape = (151936, 128)  # (vocab, packed_hidden) — shape[0] is the vocab

    class _Embed:
        weight = _W()

    class _Inner:
        embed_tokens = _Embed()

    class _M:
        model = _Inner()
        vocab_size = 999  # declared value DIVERGES — must be ignored

    class _E:
        _model = _M()
        tokenizer = None

    assert _engine_output_vocab_size(_E()) == 151936


def test_engine_output_vocab_size_resolves_language_model_nested_head():
    # #1185 REGRESSION: a multimodal-capable wrapper (qwen3_5, gemma3n, …) nests
    # the LM one level deeper — the head lives at
    # language_model.model.embed_tokens.weight, matching NONE of the original
    # three fixed paths, so _actual_output_head_width returned None and the budget
    # silently declined to the post-hoc cap (no decode-time force). The added
    # fixed path must resolve it to the real width.
    from vllm_mlx.routes.chat import _engine_output_vocab_size

    class _W:
        shape = (248320, 320)  # qwen3.5-4b: (vocab, packed_hidden)

    class _Embed:
        weight = _W()

    class _Inner:
        embed_tokens = _Embed()

    class _LM:
        model = _Inner()

    class _M:
        language_model = _LM()

    class _E:
        _model = _M()
        tokenizer = None

    assert _engine_output_vocab_size(_E()) == 248320


def test_actual_output_head_width_fixed_path_prefers_lm_head_over_embed():
    # codex: the fixed-path scan must probe ALL lm_head paths before any
    # embed_tokens path, so an untied nested output head is never shadowed by an
    # input embedding of a differing width. Both live under `model.*` with
    # DISTINCT widths; the lm_head (output projection) must win.
    from vllm_mlx.routes.chat import _actual_output_head_width

    class _WH:
        shape = (151936, 64)  # lm_head — authoritative output width

    class _WE:
        shape = (151000, 64)  # embed_tokens — input embedding, different width

    class _Head:
        weight = _WH()

    class _Embed:
        weight = _WE()

    class _Inner:
        lm_head = _Head()
        embed_tokens = _Embed()

    class _Model:
        model = _Inner()

    assert _actual_output_head_width(_Model()) == 151936


def test_actual_output_head_width_resolves_tied_nested_embed():
    # The #1185 regression at the head-width level: a multimodal-capable wrapper
    # nests its TIED text head at language_model.model.embed_tokens.weight. That
    # fixed path must resolve it (else the budget silently declines to post-hoc).
    from vllm_mlx.routes.chat import _actual_output_head_width

    class _W:
        shape = (248320, 64)

    class _Embed:
        weight = _W()

    class _Inner:
        embed_tokens = _Embed()

    class _LMModel:
        model = _Inner()

    class _Model:
        language_model = _LMModel()

    assert _actual_output_head_width(_Model()) == 248320


def test_actual_output_head_width_resolves_shallow_lm_wrapped_tied_embed():
    # codex: the tied group must mirror the lm_head group and cover BOTH wrapper
    # layouts. A tied head at the SHALLOW language_model.embed_tokens.weight (no
    # inner .model) must resolve, not just the deeper language_model.model.* one.
    from vllm_mlx.routes.chat import _actual_output_head_width

    class _W:
        shape = (262144, 64)

    class _Embed:
        weight = _W()

    class _LMModel:
        embed_tokens = _Embed()  # language_model.embed_tokens — shallow layout

    class _Model:
        language_model = _LMModel()

    assert _actual_output_head_width(_Model()) == 262144


def test_actual_output_head_width_declines_on_unknown_nesting():
    # A head reachable only at an UNRECOGNIZED path (no fixed path matches) yields
    # None — we deliberately do NOT tree-walk, so an unknown nesting declines to
    # the safe post-hoc cap rather than risk validating against the wrong head.
    from vllm_mlx.routes.chat import _actual_output_head_width

    class _W:
        shape = (151936, 64)

    class _Head:
        weight = _W()

    class _Weird:
        lm_head = _Head()  # buried under an attribute no fixed path probes

    class _Model:
        transformer = _Weird()  # not `model` / `language_model`

    assert _actual_output_head_width(_Model()) is None


def test_actual_output_head_width_ignores_stray_nonfixed_lm_head():
    # codex soundness guard: a stray lm_head at a NON-fixed path (a vision/draft
    # head) must NEVER hijack the width of the TIED text head on a fixed path.
    # Fixed-paths-only resolution uses model.embed_tokens (fixed) and never even
    # inspects the off-path stray head.
    from vllm_mlx.routes.chat import _actual_output_head_width

    class _WE:
        shape = (200000, 64)  # real tied text head, on a fixed path

    class _WV:
        shape = (99999, 64)  # stray vision/draft head, off the fixed paths

    class _Embed:
        weight = _WE()

    class _Head:
        weight = _WV()

    class _Vision:
        lm_head = _Head()

    class _Inner:
        embed_tokens = _Embed()

    class _Model:
        model = _Inner()  # model.embed_tokens.weight — fixed tied path
        visual = _Vision()  # visual.lm_head — off every fixed path

    assert _actual_output_head_width(_Model()) == 200000


# ─────────── add_generation_prompt plumbing (codex R11 #1) ──────────────────
# The two-render seed probe needs build_prompt to HONOR add_generation_prompt so
# the True/False renders differ. base.py's build_prompt is @abstractmethod (no
# body to forward); these pin that the shared sink and the concrete BatchedEngine
# path both forward the flag — the real-hardware pilot already proved the
# end-to-end effect (the budget could not have force-closed at N tokens if the
# two renders were identical → empty delta → no processor).


def test_apply_chat_template_forwards_add_generation_prompt():
    from vllm_mlx.utils.chat_template import apply_chat_template

    captured = []

    class _Rec:
        def apply_chat_template(self, messages, **kw):
            captured.append(kw)
            return "PROMPT"

    apply_chat_template(
        _Rec(), [{"role": "user", "content": "hi"}], add_generation_prompt=False
    )
    assert captured[-1]["add_generation_prompt"] is False
    # Default stays True (every serving path) when not overridden.
    apply_chat_template(_Rec(), [{"role": "user", "content": "hi"}])
    assert captured[-1]["add_generation_prompt"] is True


def test_batched_engine_apply_chat_template_forwards_add_generation_prompt(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx.engine import batched as batched_mod
    from vllm_mlx.engine.batched import BatchedEngine

    captured = {}

    def _rec(applicator, messages, **kw):
        captured.update(kw)
        return "PROMPT"

    monkeypatch.setattr(batched_mod, "shared_apply_chat_template", _rec)

    class _Tok:
        def apply_chat_template(self, *a, **k):
            return "x"

    # Bind the production method onto a bare stub (text engine, no MLLM) so the
    # exact admission-path forwarding is exercised without loading a model.
    stub = SimpleNamespace(
        _is_mllm=False, _processor=None, tokenizer=_Tok(), _model_name="qwen3"
    )
    BatchedEngine._apply_chat_template(
        stub, [{"role": "user", "content": "hi"}], add_generation_prompt=False
    )
    assert captured.get("add_generation_prompt") is False
