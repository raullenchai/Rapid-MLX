# SPDX-License-Identifier: Apache-2.0
"""#1049 — user-supplied ``stop=[...]`` on harmony (gpt-oss) requests
must scope to the ``final`` channel body only, never the ``analysis``
channel CoT.

Discovered by PR #1048's real Docker OpenHands E2E harness: the
CodeActAgent sends ``stop=['</execute_ipython>', ...]`` on every
request, and the harmony model's analysis-channel CoT routinely
mentions those markers while reasoning about which action to take.
Applying the stop match to the raw decoded surface terminates the
request in mid-CoT — ``content=""``, ``reasoning_content`` ends with
the stop marker — and CodeActAgent falls back to an empty
``MessageAction``.

Correct behavior (from the issue body):

    stop=[...] applies ONLY to emitted final channel content — never
    to the analysis channel. Analysis-channel token production stays
    stop-agnostic; only harmony's own channel delimiter (``<|channel|>``,
    ``<|end|>``, ``<|return|>``) terminates it.

These tests pin three layers:

1. The pure ``find_stop_in_final_channel`` helper — no scheduler
   plumbing, just the string-matching contract.
2. The ``Scheduler._process_batch_responses`` integration — harmony
   family flag set, decoder surface contains analysis + final, user
   stop appears in BOTH regions, ONLY the final-channel match fires.
3. Non-harmony parity — a plain Qwen-style tokenizer keeps raw-stream
   stop matching identical to the pre-fix code path (regression test
   for the branch that non-harmony models take).

Reference: https://github.com/raullenchai/Rapid-MLX/issues/1049
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from unittest.mock import MagicMock

from vllm_mlx.reasoning.harmony_stop import (
    HARMONY_FINAL_MARKER,
    find_harmony_final_span,
    find_stop_in_final_channel,
    is_harmony_family_tokenizer,
)
from vllm_mlx.request import Request, RequestStatus, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig

# ---------------------------------------------------------------------------
# Layer 1: pure string-matching contract
# ---------------------------------------------------------------------------


def test_final_span_none_before_final_marker():
    """Analysis-only surface has no final channel yet — no span."""
    text = "<|channel|>analysis<|message|>reasoning about </execute_ipython> markers"
    assert find_harmony_final_span(text) is None


def test_final_span_open_when_final_marker_present():
    """Once ``<|channel|>final<|message|>`` appears, the body starts."""
    text = (
        "<|channel|>analysis<|message|>cot text<|end|>"
        "<|start|>assistant<|channel|>final<|message|>the answer"
    )
    span = find_harmony_final_span(text)
    assert span is not None
    body_start, body_end = span
    assert text[body_start:body_end] == "the answer"


def test_final_span_closed_at_return_marker():
    """``<|return|>`` terminates the final-channel body."""
    text = "<|channel|>final<|message|>answer body<|return|>"
    span = find_harmony_final_span(text)
    assert span is not None
    body_start, body_end = span
    assert text[body_start:body_end] == "answer body"


def test_stop_ignores_analysis_channel_mention():
    """The classic #1049 reproducer surface: analysis mentions the
    stop marker; final channel is empty. No user stop should fire."""
    stop_params = ["</execute_ipython>", "</execute_bash>"]
    text = (
        "<|channel|>analysis<|message|>To print hello world I will "
        "use </execute_ipython> at the end<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )
    assert find_stop_in_final_channel(text, stop_params) is None


def test_stop_fires_inside_final_channel():
    """When the model emits the stop marker INSIDE the final channel,
    the match fires and the global offset points at the final-channel
    occurrence, NOT the analysis-channel occurrence."""
    stop_params = ["</execute_ipython>"]
    text = (
        "<|channel|>analysis<|message|>I'll use </execute_ipython>"
        "<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        '<execute_ipython>\nprint("hello world")\n</execute_ipython>'
    )
    match = find_stop_in_final_channel(text, stop_params)
    assert match is not None
    stop_str, global_idx = match
    assert stop_str == "</execute_ipython>"
    # The match position must be inside the final-channel body, not
    # at the analysis-channel occurrence.
    analysis_pos = text.index("</execute_ipython>")
    assert global_idx > analysis_pos
    # And the trimmed text should retain everything up to (but not
    # including) the final ``</execute_ipython>``.
    trimmed = text[:global_idx]
    assert trimmed.endswith('print("hello world")\n')
    # Trimmed prefix must NOT itself end with the marker; the marker
    # must sit AT ``global_idx`` in the raw surface (codex round-2
    # BLOCKING — the previous ``text[global_idx:global_idx]`` slice
    # was always empty so this line asserted nothing).
    assert not trimmed.endswith("</execute_ipython>")
    assert text.startswith("</execute_ipython>", global_idx)


def test_stop_earliest_position_wins_inside_final():
    """With multiple user stops that all appear inside the final
    channel, the earliest-in-the-body wins."""
    stop_params = ["</execute_browse>", "</execute_ipython>"]
    text = (
        "<|channel|>final<|message|>"
        "action</execute_ipython> then browse</execute_browse>"
    )
    match = find_stop_in_final_channel(text, stop_params)
    assert match is not None
    stop_str, _global_idx = match
    assert stop_str == "</execute_ipython>"


def test_stop_ignores_empty_and_none_stop_strings():
    """Empty / None stop entries must not spuriously match at offset 0."""
    text = "<|channel|>final<|message|>real content"
    assert find_stop_in_final_channel(text, ["", None]) is None  # type: ignore[list-item]


def test_final_content_containing_literal_channel_string_still_matches():
    """Codex round-2 NIT: a model answering "what does <|channel|> mean?"
    might emit the literal string ``<|channel|>`` inside the final-
    channel body. That must NOT prematurely close the final span —
    only true harmony control markers (``<|end|>``, ``<|return|>``,
    ``<|call|>``) terminate the body. The stop-string still fires
    correctly at a later position in the same body.
    """
    stop_params = ["STOP"]
    text = (
        "<|channel|>final<|message|>"
        "The <|channel|> token opens a channel block. STOP here."
    )
    match = find_stop_in_final_channel(text, stop_params)
    assert match is not None
    stop_str, global_idx = match
    assert stop_str == "STOP"
    # Trim retains the literal ``<|channel|>`` mention inside the
    # final body — the fix removed ``<|channel|>`` from the terminator
    # set precisely so this case doesn't false-close the span.
    trimmed = text[:global_idx]
    assert "<|channel|>" in trimmed[len("<|channel|>final<|message|>") :]


# ---------------------------------------------------------------------------
# Layer 2: scheduler integration
# ---------------------------------------------------------------------------


def _make_harmony_scheduler() -> Scheduler:
    """Build a ``Scheduler`` with a mock tokenizer whose vocab identifies
    it as harmony family. ``is_harmony_family_tokenizer`` reads
    ``get_vocab`` first, so a mock vocab with ``<|channel|>`` +
    ``<|message|>`` is enough — no HF tokenizer required."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s.split())))
    # Vocab-based detection — matches OutputRouter.from_tokenizer.
    tokenizer.get_vocab = lambda: {
        "<|channel|>": 200005,
        "<|message|>": 200006,
        "<|start|>": 200007,
        "<|end|>": 200008,
    }
    tokenizer.name_or_path = "mlx-community/gpt-oss-20b-MXFP4-Q8"
    scheduler = Scheduler(model, tokenizer, SchedulerConfig(max_num_seqs=4))
    assert scheduler._is_harmony_family is True, (
        "Scheduler should have detected the mock harmony vocab."
    )
    return scheduler


def _make_non_harmony_scheduler() -> Scheduler:
    """Build a ``Scheduler`` with a mock tokenizer that is NOT harmony
    family — the vocab has no harmony markers and the name is a
    non-harmony model. Regression parity gate: this scheduler should
    match the pre-fix raw-stream stop behavior byte-for-byte."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s.split())))
    tokenizer.get_vocab = lambda: {"hello": 1, "world": 2}
    tokenizer.name_or_path = "mlx-community/Qwen3-4B-4bit"
    scheduler = Scheduler(model, tokenizer, SchedulerConfig(max_num_seqs=4))
    assert scheduler._is_harmony_family is False
    return scheduler


def _make_request_with_decoder(
    rid: str,
    *,
    stop_strings: list[str],
    accumulated_full_text: str,
    prefilled_tokens: list[int],
) -> Request:
    sp = SamplingParams(max_tokens=100, stop=stop_strings)
    req = Request(request_id=rid, prompt="ignored", sampling_params=sp)
    req.num_prompt_tokens = 4
    req.status = RequestStatus.RUNNING
    for t in prefilled_tokens:
        req.append_output_token(t)
    decoder = MagicMock()
    decoder.get_full_text = lambda: accumulated_full_text
    decoder.add_token = lambda _t: ""
    decoder.prev_text = accumulated_full_text
    req._decoder = decoder
    return req


def _run_step(scheduler: Scheduler, request: Request):
    scheduler.running[request.request_id] = request
    scheduler.uid_to_request_id[0] = request.request_id
    scheduler._decode_tokens = lambda tokens: ""  # type: ignore[method-assign]

    response = MagicMock()
    response.uid = 0
    response.token = 42
    response.finish_reason = None
    response.logprobs = None
    del response.prompt_cache
    outputs, finished = scheduler._process_batch_responses([response])
    return outputs[0], finished


def test_scheduler_harmony_ignores_analysis_channel_stop_mention():
    """#1049 core regression: the OpenHands CodeActAgent stop set
    appears verbatim inside analysis-channel CoT. Pre-fix the
    scheduler stopped mid-CoT and content was empty; post-fix
    generation continues (no stop fires) so the final channel gets
    a chance to emit."""
    scheduler = _make_harmony_scheduler()
    stop_strings = [
        "</execute_ipython>",
        "</execute_bash>",
        "</execute_browse>",
    ]
    analysis_only_surface = (
        "<|channel|>analysis<|message|>I need to run an ipython "
        "block. The action will end with </execute_ipython>."
    )
    req = _make_request_with_decoder(
        "rA",
        stop_strings=stop_strings,
        accumulated_full_text=analysis_only_surface,
        prefilled_tokens=[10, 11],
    )
    output, finished = _run_step(scheduler, req)
    assert output.finish_reason is None, (
        "Analysis-channel mention of the stop marker triggered a "
        "premature stop — this is issue #1049 regressing."
    )
    assert output.finished is False
    assert finished == set()


def test_scheduler_harmony_stops_on_final_channel_marker():
    """When the model emits the stop marker inside the final channel,
    the scheduler DOES stop — and truncates at the final-channel
    occurrence, NOT the analysis-channel one that appears earlier
    in the raw surface."""
    scheduler = _make_harmony_scheduler()
    stop_strings = ["</execute_ipython>"]
    accumulated = (
        "<|channel|>analysis<|message|>I'll use </execute_ipython>"
        " to run code<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        '<execute_ipython>\nprint("hello world")\n</execute_ipython>'
    )
    req = _make_request_with_decoder(
        "rB",
        stop_strings=stop_strings,
        accumulated_full_text=accumulated,
        prefilled_tokens=[10, 11],
    )
    output, finished = _run_step(scheduler, req)
    assert output.finish_reason == "stop"
    assert output.finished is True
    assert "rB" in finished
    # The trimmed text must retain the analysis-channel usage of
    # ``</execute_ipython>`` (that's part of the CoT surface, not the
    # user-visible final content) and cut off at the FINAL channel's
    # ``</execute_ipython>`` — the emitted action must be intact
    # except for the trailing stop marker.
    assert output.output_text.endswith('print("hello world")\n')
    # The analysis-channel occurrence must still be inside the text
    # (proves the trim happened at the final-channel occurrence, not
    # the analysis-channel one).
    assert "analysis<|message|>I'll use </execute_ipython>" in output.output_text


def test_scheduler_non_harmony_stops_on_raw_stream_unchanged():
    """Non-harmony model with stop=['STOP']: pre-fix behavior — the
    stop fires against the raw decoded surface regardless of channels
    (there are no channels). Regression parity test."""
    scheduler = _make_non_harmony_scheduler()
    stop_strings = ["STOP"]
    accumulated = "hello world STOP tail"
    req = _make_request_with_decoder(
        "rC",
        stop_strings=stop_strings,
        accumulated_full_text=accumulated,
        prefilled_tokens=[10, 11],
    )
    output, _ = _run_step(scheduler, req)
    assert output.finish_reason == "stop"
    assert output.output_text == "hello world "


def test_scheduler_harmony_without_stop_param_unaffected():
    """Same accumulated surface (analysis-channel mentions the marker
    that WOULD be a stop) but no ``stop`` argument set — generation
    must NOT terminate. Sanity guard against a naive fix that would
    have applied the harmony scoping unconditionally."""
    scheduler = _make_harmony_scheduler()
    analysis_only_surface = "<|channel|>analysis<|message|>I'll emit </execute_ipython>"
    req = _make_request_with_decoder(
        "rD",
        stop_strings=[],  # empty stop list
        accumulated_full_text=analysis_only_surface,
        prefilled_tokens=[10],
    )
    output, _ = _run_step(scheduler, req)
    assert output.finish_reason is None
    assert output.finished is False


# ---------------------------------------------------------------------------
# Layer 3: family gate
# ---------------------------------------------------------------------------


def test_is_harmony_family_via_vocab():
    """Vocab-based detection: any tokenizer with the harmony markers
    in its vocab is treated as harmony family, regardless of name.
    Matches OutputRouter.from_tokenizer's detection."""
    tok = MagicMock()
    tok.get_vocab = lambda: {"<|channel|>": 1, "<|message|>": 2}
    tok.name_or_path = "some-user/unnamed"
    assert is_harmony_family_tokenizer(tok) is True


def test_is_harmony_family_via_name():
    """Name-based fallback for tokenizers whose vocab is opaque
    (mock tokenizers, custom wrappers). Matches the
    ``_is_known_harmony_identity`` allowlist."""
    tok = MagicMock(spec=[])  # no get_vocab attribute
    tok.name_or_path = "mlx-community/gpt-oss-20b-MXFP4-Q8"
    assert is_harmony_family_tokenizer(tok) is True


def test_is_harmony_family_rejects_qwen():
    """Non-harmony family: no vocab markers, non-allowlisted name."""
    tok = MagicMock()
    tok.get_vocab = lambda: {"hello": 1}
    tok.name_or_path = "Qwen/Qwen3-4B"
    assert is_harmony_family_tokenizer(tok) is False


def test_is_harmony_family_handles_none_tokenizer():
    """Defensive: None tokenizer should not crash."""
    assert is_harmony_family_tokenizer(None) is False


# ---------------------------------------------------------------------------
# Layer 4: exact LiteLLM reproducer surface (issue body)
# ---------------------------------------------------------------------------


def test_literal_issue_1049_reproducer_surface():
    """Wire-level regression: the exact stop set + raw surface from
    issue #1049's LiteLLM reproducer produces the correct outcome.

    Pre-fix: ``content=""``, ``reasoning_content`` ends with
    ``</execute_ipython>`` — the analysis-channel mention fired.
    Post-fix: no stop fires inside the analysis-only surface (below);
    when the model eventually emits the final channel with the SAME
    marker inside, THAT one fires and content is populated.
    """
    stops = ["</execute_ipython>", "</execute_bash>", "</execute_browse>"]

    # Analysis-only surface — pre-fix stopped here.
    analysis_only = (
        "<|channel|>analysis<|message|>The user wants me to print "
        "hello world in ipython. I will use "
        "<execute_ipython>print('hello world')</execute_ipython> "
        "for that."
    )
    assert find_stop_in_final_channel(analysis_only, stops) is None

    # After the final marker emits AND completes — the stop fires,
    # correctly this time.
    full = (
        analysis_only + "<|end|><|start|>assistant<|channel|>final<|message|>"
        "<execute_ipython>\nprint('hello world')\n</execute_ipython>"
    )
    match = find_stop_in_final_channel(full, stops)
    assert match is not None
    stop_str, global_idx = match
    assert stop_str == "</execute_ipython>"
    # The truncated content is the FINAL-channel action body without
    # the trailing stop — this is what OpenAI clients see as
    # ``choice.message.content``.
    trimmed_final_body = full[
        full.rfind(HARMONY_FINAL_MARKER) + len(HARMONY_FINAL_MARKER) : global_idx
    ]
    assert trimmed_final_body == "<execute_ipython>\nprint('hello world')\n"


# ---------------------------------------------------------------------------
# Layer 5: MLLMScheduler rolling matcher parity (codex round-1 finding)
# ---------------------------------------------------------------------------
#
# Codex claimed the MLLMScheduler's rolling-window matcher would miss a
# final-channel stop once ``<|channel|>final<|message|>`` scrolled out of
# the ``max_stop_len - 1`` lookback window. That reading is incorrect —
# ``request.stop_text`` accumulates the FULL decoded surface (assigned
# monotonically via ``request.stop_text = streamed_so_far`` at
# ``vllm_mlx/mllm_scheduler.py:924/929``), and ``find_harmony_final_span``
# scans that full text via ``rfind`` before the rolling matcher touches
# the body substring. These tests pin that behavior explicitly so a
# future refactor that DID turn ``stop_text`` into a rolling tail would
# break here loudly instead of silently regressing #1049.


def test_mllm_wrapper_finds_final_marker_far_before_window():
    """Simulate a long-running MLLM request: analysis body is 5000+
    characters, then final marker opens, then final body emits the
    stop marker. The rolling matcher's ``keep = max_stop_len - 1``
    window (~18 chars for ``</execute_ipython>``) is dwarfed by the
    5000-char analysis prefix, but the wrapper still resolves the
    final-channel body correctly and fires the stop."""
    from vllm_mlx.reasoning.harmony_stop import find_stop_in_final_channel

    analysis_body = "reasoning step. " * 350  # ~5600 chars
    text = (
        f"<|channel|>analysis<|message|>{analysis_body}<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "the action</execute_ipython>trailing"
    )
    match = find_stop_in_final_channel(text, ["</execute_ipython>"])
    assert match is not None
    stop_str, global_idx = match
    assert stop_str == "</execute_ipython>"
    # The global offset must land inside the final-channel body — not
    # in the analysis-channel region (which is 5000+ chars earlier).
    final_marker_idx = text.rfind(HARMONY_FINAL_MARKER)
    assert global_idx > final_marker_idx
    assert text[:global_idx].endswith("the action")


def test_mllm_match_user_stop_uses_full_text_span():
    """Direct pin on ``MLLMScheduler._match_user_stop``: build an
    MLLMScheduler with a mock harmony processor, feed a text whose
    final marker is far before ``new_text_start_len`` (simulating
    many decode steps since the marker opened), and confirm the
    wrapper returns a match in the final-channel body.

    Pre-fix (raw ``_find_stop_match_in_new_window``) would have
    scanned only the ``max_stop_len - 1`` tail and missed markers
    that spanned earlier text. The wrapper computes the final-channel
    span on the FULL provided ``text`` first, then delegates to the
    rolling matcher on the bounded body substring.
    """
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.get_vocab = lambda: {
        "<|channel|>": 200005,
        "<|message|>": 200008,
    }
    processor.tokenizer.name_or_path = "mlx-community/gpt-oss-20b-MXFP4-Q8"
    processor.tokenizer.eos_token_id = 200002
    # Bypass __init__'s multimodal / model-config plumbing —
    # ``_match_user_stop`` only reads ``self._is_harmony_family``.
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._is_harmony_family = True

    analysis_body = "cot text. " * 400  # ~4000 chars
    text = (
        f"<|channel|>analysis<|message|>{analysis_body}<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "answer body</execute_ipython>"
    )
    # Simulate: we just decoded the last chunk containing ``</execute_ipython>``.
    # ``new_text_start_len`` is the length before this chunk arrived.
    new_text_start_len = len(text) - len("</execute_ipython>")
    match = scheduler._match_user_stop(text, new_text_start_len, ["</execute_ipython>"])
    assert match is not None
    idx, stop_str = match
    assert stop_str == "</execute_ipython>"
    # The match position is in the final-channel body, not the analysis.
    final_marker_idx = text.rfind(HARMONY_FINAL_MARKER)
    assert idx > final_marker_idx


def test_mllm_match_user_stop_ignores_analysis_only():
    """Pre-final MLLMScheduler surface: analysis body mentions the
    stop marker verbatim, no final marker yet. Wrapper returns None
    so generation continues into the (yet-unseen) final channel."""
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._is_harmony_family = True
    text = (
        "<|channel|>analysis<|message|>I will use </execute_ipython>"
        " at the end of my action to close the block."
    )
    match = scheduler._match_user_stop(text, 0, ["</execute_ipython>"])
    assert match is None


def test_mllm_match_user_stop_non_harmony_unchanged():
    """Non-harmony family: wrapper must delegate to the pre-#1049
    ``_find_stop_match_in_new_window`` byte-for-byte. Regression
    parity gate — codex round-1 finding conflated non-harmony and
    harmony paths."""
    from vllm_mlx.mllm_scheduler import (
        MLLMScheduler,
        _find_stop_match_in_new_window,
    )

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._is_harmony_family = False
    text = "regular content STOP tail"
    stop_params = ["STOP"]
    wrapper_match = scheduler._match_user_stop(text, 0, stop_params)
    direct_match = _find_stop_match_in_new_window(text, 0, stop_params)
    assert wrapper_match == direct_match
    assert wrapper_match is not None
    idx, stop_str = wrapper_match
    assert stop_str == "STOP"
    assert text[:idx] == "regular content "
