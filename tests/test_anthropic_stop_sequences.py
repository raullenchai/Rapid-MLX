# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for Anthropic /v1/messages stop_sequences enforcement (#469).

Covers three layers:

1. ``_convert_stop_reason`` / ``openai_to_anthropic`` adapter helpers — when
   ``matched_stop`` is supplied, the Anthropic spec requires
   ``stop_reason="stop_sequence"`` and ``stop_sequence=<matched>``.
2. ``server._convert_anthropic_stop_reason`` — same contract on the live
   endpoint helper used by ``create_anthropic_message``.
3. ``Scheduler._process_batch_responses`` — engine-level enforcement: when a
   client-supplied stop string appears in the cumulative output, the request
   terminates with the truncated text, ``finish_reason="stop"``, and the
   matched string surfaced on ``RequestOutput.matched_stop``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_mlx.api.anthropic_adapter import (
    _convert_stop_reason,
    openai_to_anthropic,
)
from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    Usage,
)
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


# ---------------------------------------------------------------------------
# Adapter layer
# ---------------------------------------------------------------------------


class TestConvertStopReasonMatchedStop:
    def test_matched_stop_forces_stop_sequence_label(self):
        # Regardless of the upstream OpenAI label, a matched stop string
        # MUST surface as the Anthropic spec ``stop_sequence``.
        assert (
            _convert_stop_reason("stop", matched_stop="END")
            == "stop_sequence"
        )
        assert (
            _convert_stop_reason("length", matched_stop="END")
            == "stop_sequence"
        )
        assert (
            _convert_stop_reason(None, matched_stop="END") == "stop_sequence"
        )

    def test_matched_stop_none_falls_back_to_existing_mapping(self):
        assert _convert_stop_reason("stop", matched_stop=None) == "end_turn"
        assert _convert_stop_reason("length") == "max_tokens"


class TestOpenaiToAnthropicMatchedStop:
    def _make_response(self, content="hi"):
        msg = AssistantMessage(content=content)
        choice = ChatCompletionChoice(message=msg, finish_reason="stop")
        return ChatCompletionResponse(
            model="m",
            choices=[choice],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def test_matched_stop_populates_both_fields(self):
        resp = self._make_response()
        out = openai_to_anthropic(resp, "m", matched_stop="END")
        assert out.stop_reason == "stop_sequence"
        assert out.stop_sequence == "END"

    def test_no_matched_stop_leaves_stop_sequence_null(self):
        resp = self._make_response()
        out = openai_to_anthropic(resp, "m")
        assert out.stop_reason == "end_turn"
        assert out.stop_sequence is None

    def test_matched_stop_with_no_choices(self):
        empty = ChatCompletionResponse(
            model="m",
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        out = openai_to_anthropic(empty, "m", matched_stop="END")
        assert out.stop_reason == "stop_sequence"
        assert out.stop_sequence == "END"


class TestServerConvertAnthropicStopReason:
    """The server module's own ``_convert_anthropic_stop_reason`` helper.

    Imported lazily so test collection does not pull in heavy MLX deps when
    only running the adapter portion above.
    """

    def test_matched_stop_overrides(self):
        from vllm_mlx.server import _convert_anthropic_stop_reason

        assert (
            _convert_anthropic_stop_reason("stop", matched_stop="END")
            == "stop_sequence"
        )
        assert (
            _convert_anthropic_stop_reason("tool_calls", matched_stop="END")
            == "stop_sequence"
        )
        assert _convert_anthropic_stop_reason("stop") == "end_turn"
        assert _convert_anthropic_stop_reason("length") == "max_tokens"


# ---------------------------------------------------------------------------
# Engine layer — scheduler stop-string enforcement
# ---------------------------------------------------------------------------


def _make_scheduler():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(max(1, len(x.split()))))
    tokenizer.decode = lambda ids: " ".join(str(i) for i in ids)
    tokenizer.eos_token_id = 0
    sched = Scheduler(model, tokenizer, SchedulerConfig(max_num_seqs=4))
    return sched


def _make_running_request(sched, request_id, uid, stop):
    req = Request(
        request_id=request_id,
        prompt="prompt",
        sampling_params=SamplingParams(max_tokens=128, stop=stop or []),
    )
    req.prompt_token_ids = [1, 2, 3]
    req.num_prompt_tokens = 3
    sched.requests[request_id] = req
    sched.running[request_id] = req
    sched.uid_to_request_id[uid] = request_id
    sched.request_id_to_uid[request_id] = uid
    return req


def _fake_response(uid, token, finish_reason=None):
    # Mimic the mlx_lm GenerationBatch.Response shape used by the scheduler.
    # Only the fields _process_batch_responses reads need to be present.
    resp = SimpleNamespace(
        uid=uid,
        token=token,
        logprobs=0.0,
        finish_reason=finish_reason,
        prompt_cache=None,
    )
    return resp


def _stream_text_into_request(sched, request_id, uid, chunks):
    """Drive _process_batch_responses with a sequence of text chunks.

    Each chunk in ``chunks`` becomes one fake response whose detokenizer
    surfaces the chunk as ``new_text``.  We monkey-patch the scheduler's
    detokenizer pool so we don't need a real tokenizer.
    """

    class FakeDetok:
        def __init__(self):
            self._segments: list[str] = []
            self.last_segment = ""
            self.text = ""

        def add_token(self, token):
            seg = chunks[token]
            self.last_segment = seg
            self._segments.append(seg)
            self.text = "".join(self._segments)

        def finalize(self):
            self.text = "".join(self._segments)

    sched._detokenizer_pool[request_id] = FakeDetok()
    # Stub batch_generator.remove so early-stop path doesn't crash.
    sched.batch_generator = MagicMock()

    outputs_total = []
    finished_total = set()
    for token_id in range(len(chunks)):
        outs, finished = sched._process_batch_responses(
            [_fake_response(uid, token_id)]
        )
        outputs_total.extend(outs)
        finished_total |= finished
        if finished:
            break
    return outputs_total, finished_total


class TestSchedulerStopSequenceEnforcement:
    def test_truncates_at_first_match_and_surfaces_matched_stop(self):
        sched = _make_scheduler()
        _make_running_request(sched, "r1", uid=10, stop=["STOP"])
        outputs, finished = _stream_text_into_request(
            sched,
            "r1",
            10,
            ["hello ", "world ", "STOP", " ignored"],
        )
        assert "r1" in finished
        last = outputs[-1]
        assert last.finished is True
        assert last.finish_reason == "stop"
        assert last.matched_stop == "STOP"
        # Truncated at the match — STOP and everything after MUST be cut.
        assert last.output_text == "hello world "
        assert "STOP" not in last.output_text

    def test_earliest_match_wins_when_multiple_candidates(self):
        sched = _make_scheduler()
        _make_running_request(sched, "r2", uid=20, stop=["END", "STOP"])
        outputs, finished = _stream_text_into_request(
            sched,
            "r2",
            20,
            ["A ", "B ", "STOP ", "C ", "END"],
        )
        assert "r2" in finished
        last = outputs[-1]
        assert last.matched_stop == "STOP"
        assert last.output_text == "A B "

    def test_no_stop_strings_supplied_is_unchanged(self):
        sched = _make_scheduler()
        _make_running_request(sched, "r3", uid=30, stop=None)

        class FakeDetok:
            def __init__(self):
                self.last_segment = ""
                self.text = ""

            def add_token(self, t):
                self.last_segment = f"t{t}"
                self.text += self.last_segment

            def finalize(self):
                pass

        sched._detokenizer_pool["r3"] = FakeDetok()
        sched.batch_generator = MagicMock()
        outs1, fin1 = sched._process_batch_responses(
            [_fake_response(30, token=1, finish_reason=None)]
        )
        outs2, fin2 = sched._process_batch_responses(
            [_fake_response(30, token=2, finish_reason="stop")]
        )
        # Without client-supplied stop strings, matched_stop must stay None
        # and the request finishes only when the engine reports finish.
        assert all(o.matched_stop is None for o in outs1 + outs2)
        assert "r3" not in fin1
        assert "r3" in fin2

    def test_match_at_token_boundary_does_not_leak_suffix(self):
        # Common harness behaviour: model emits the literal stop string in
        # one chunk after some prior content.  We MUST NOT include the
        # match itself nor anything after it in output_text.
        sched = _make_scheduler()
        _make_running_request(sched, "r4", uid=40, stop=["END"])
        outputs, finished = _stream_text_into_request(
            sched,
            "r4",
            40,
            ["hi", "END", "leak"],
        )
        assert "r4" in finished
        last = outputs[-1]
        assert last.matched_stop == "END"
        assert last.output_text == "hi"

    def test_streamed_new_text_never_leaks_matched_suffix(self):
        # Regression: per-token ``new_text`` is what SSE clients see as
        # ``content_block_delta``.  The terminal chunk MUST NOT contain the
        # matched stop string, and the sum of all emitted deltas MUST equal
        # the final ``output_text`` exactly.
        sched = _make_scheduler()
        _make_running_request(sched, "r5", uid=50, stop=["END"])
        outputs, finished = _stream_text_into_request(
            sched,
            "r5",
            50,
            ["hello ", "wor", "ld", "END"],
        )
        assert "r5" in finished
        # Concatenated stream of new_text deltas must equal final output.
        streamed = "".join(o.new_text for o in outputs)
        assert "END" not in streamed
        assert streamed == outputs[-1].output_text == "hello world"

    def test_partial_stop_prefix_is_held_back_until_disambiguated(self):
        # When the model emits a chunk we don't yet know whether the tail
        # is the start of a stop string — the scheduler MUST hold back
        # the trailing ``max_stop_len - 1`` chars until the next token
        # disambiguates.  For ``stop=["END"]`` (len 3) hold-back is 2.
        sched = _make_scheduler()
        _make_running_request(sched, "r6", uid=60, stop=["END"])
        outputs, finished = _stream_text_into_request(
            sched,
            "r6",
            60,
            ["hello ", "E", "ND"],
        )
        assert "r6" in finished
        # Buffer "hello " (6 chars) → safe to emit "hell" (len-2), hold "o ".
        assert outputs[0].new_text == "hell"
        # "E" arrives → buf="hello E" (7 chars) → safe_end=5, published=4
        # → delta is buf[4:5]="o".  The space and "E" stay held back.
        assert outputs[1].new_text == "o"
        # Cumulative "hello END" — match fires, truncate to "hello ".
        assert outputs[-1].matched_stop == "END"
        assert outputs[-1].output_text == "hello "
        # No partial-of-stop ever reached the client stream.
        streamed = "".join(o.new_text for o in outputs)
        assert streamed == "hello "
        assert "E" not in streamed
        assert "N" not in streamed
        assert "D" not in streamed

    def test_length_finish_with_stop_match_reports_stop_sequence(self):
        # Codex r2 regression: the token that hits max_tokens MAY also be
        # the one that completes the stop string.  The Anthropic-spec-
        # correct answer is ``stop_sequence``, not ``max_tokens``.
        sched = _make_scheduler()
        _make_running_request(sched, "r8", uid=80, stop=["END"])

        class FakeDetok:
            def __init__(self):
                self._segs = []
                self.last_segment = ""
                self.text = ""

            def add_token(self, t):
                seg = {1: "hi ", 2: "END"}.get(t, "")
                self.last_segment = seg
                self._segs.append(seg)
                self.text = "".join(self._segs)

            def finalize(self):
                self.text = "".join(self._segs)

        sched._detokenizer_pool["r8"] = FakeDetok()
        sched.batch_generator = MagicMock()
        outs1, _ = sched._process_batch_responses(
            [_fake_response(80, token=1, finish_reason=None)]
        )
        # Final token: max_tokens hit AND completes the stop string.
        outs2, fin2 = sched._process_batch_responses(
            [_fake_response(80, token=2, finish_reason="length")]
        )
        assert "r8" in fin2
        last = outs2[-1]
        # spec-correct: report stop_sequence, NOT max_tokens
        assert last.matched_stop == "END"
        assert last.finish_reason == "stop"
        assert last.output_text == "hi "
        streamed = "".join(o.new_text for o in outs1 + outs2)
        assert "END" not in streamed
        assert streamed == "hi "

    def test_held_back_tail_flushed_on_length_termination(self):
        # When stops are supplied but the engine finishes via "length"
        # (max_tokens hit) without any match, the held-back tail MUST be
        # flushed into the terminal new_text so the client sees the full
        # output.  ("stop" termination skips content decoding by design —
        # that's an EOS token, not content — so no flush is needed there.)
        sched = _make_scheduler()
        _make_running_request(sched, "r7", uid=70, stop=["XYZ"])

        class FakeDetok:
            def __init__(self):
                self._segs = []
                self.last_segment = ""
                self.text = ""

            def add_token(self, t):
                seg = {1: "hello", 2: " ", 3: "wo"}.get(t, "")
                self.last_segment = seg
                self._segs.append(seg)
                self.text = "".join(self._segs)

            def finalize(self):
                self.text = "".join(self._segs)

        sched._detokenizer_pool["r7"] = FakeDetok()
        sched.batch_generator = MagicMock()
        outs1, _ = sched._process_batch_responses(
            [_fake_response(70, token=1, finish_reason=None)]
        )
        outs2, _ = sched._process_batch_responses(
            [_fake_response(70, token=2, finish_reason=None)]
        )
        outs3, fin3 = sched._process_batch_responses(
            [_fake_response(70, token=3, finish_reason="length")]
        )
        assert "r7" in fin3
        streamed = "".join(o.new_text for o in outs1 + outs2 + outs3)
        # Held-back portion must reappear in the terminal flush.
        assert streamed == "hello wo"
        assert outs3[-1].output_text == "hello wo"


# ---------------------------------------------------------------------------
# OpenAI /v1/chat/completions: no regression — stop param still flows through
# ---------------------------------------------------------------------------


class TestSimpleEngineStopEnforcement:
    """Codex r3 regression: the default (non-continuous-batching) engine
    is ``SimpleEngine``, which wraps ``MLXLanguageModel`` directly.  The
    ``stream_generate`` path now hold-back-buffers and surfaces
    ``matched_stop``; the non-streaming ``generate`` aggregates the same
    stream so it inherits the enforcement and stop-string reporting.
    """

    def test_stream_generate_emits_matched_stop_and_truncates(self):
        from vllm_mlx.models.llm import MLXLanguageModel, StreamingOutput

        # Build a fake mlx_lm.stream_generate that emits a known token
        # sequence so we don't need a real model.
        class FakeResp:
            def __init__(self, text, token=1):
                self.text = text
                self.token = token

        seq = [
            FakeResp("hello "),
            FakeResp("wor"),
            FakeResp("ld"),
            FakeResp("END"),
        ]

        # Monkey-patch the lazy mlx_lm imports inside stream_generate.
        original_stream_generate = None
        try:
            import mlx_lm as _mlx_lm
            original_stream_generate = _mlx_lm.stream_generate
            _mlx_lm.stream_generate = lambda *a, **kw: iter(seq)
            model = MLXLanguageModel.__new__(MLXLanguageModel)
            model._loaded = True
            model._mtp = False
            model._mtp_num_draft_tokens = 1
            model.model = MagicMock()
            model.tokenizer = MagicMock()
            model.tokenizer.encode = lambda s: list(range(len(s.split())))
            model._create_sampler = lambda *a, **kw: None
            model._create_logits_processors = lambda *a, **kw: None

            chunks = list(
                model.stream_generate(
                    prompt="ignored",
                    max_tokens=10,
                    stop=["END"],
                )
            )
            assert chunks, "stream_generate yielded nothing"
            terminal = chunks[-1]
            assert isinstance(terminal, StreamingOutput)
            assert terminal.finished is True
            assert terminal.finish_reason == "stop"
            assert terminal.matched_stop == "END"
            streamed = "".join(c.text for c in chunks)
            assert "END" not in streamed
        finally:
            if original_stream_generate is not None:
                import mlx_lm as _mlx_lm  # noqa: F811
                _mlx_lm.stream_generate = original_stream_generate

    def test_generate_truncates_and_surfaces_matched_stop(self):
        # Non-streaming path: mlx_lm.generate has no native stop support,
        # so our wrapper scans + truncates after the fact and surfaces
        # ``matched_stop`` for the Anthropic adapter.
        from vllm_mlx.models.llm import GenerationOutput, MLXLanguageModel

        import mlx_lm as _mlx_lm
        orig = _mlx_lm.generate
        try:
            _mlx_lm.generate = lambda *a, **kw: "hi ENDworld"
            model = MLXLanguageModel.__new__(MLXLanguageModel)
            model._loaded = True
            model._mtp = False
            model._mtp_num_draft_tokens = 1
            model.model = MagicMock()
            model.tokenizer = MagicMock()
            model.tokenizer.encode = lambda s: list(range(len(s)))
            model._create_sampler = lambda *a, **kw: None
            model._create_logits_processors = lambda *a, **kw: None

            out = model.generate(
                prompt="ignored",
                max_tokens=10,
                stop=["END"],
            )
            assert isinstance(out, GenerationOutput)
            assert out.finish_reason == "stop"
            assert out.matched_stop == "END"
            assert "END" not in out.text
            assert out.text == "hi "
        finally:
            _mlx_lm.generate = orig


class TestSimpleEngineTextModePathSanitizes:
    """Codex r4 regression: the SimpleEngine text-mode MLLM streaming
    path (``_stream_generate_text``) used to scan ``accumulated_text``
    for stop strings but yielded ``new_text`` / ``text`` unchanged after
    the match.  The match itself MUST be truncated from both the SSE
    delta and the final aggregated content.
    """

    def test_match_in_single_chunk_truncates_delta_and_total(self):
        # Pure-logic check mirroring the truncation block in
        # ``_stream_generate_text``: when the matched stop arrives inside
        # one chunk, both the per-token ``new_text`` and the cumulative
        # ``accumulated_text`` must be sanitized.
        accumulated_text = ""
        stop = ["END"]
        out_deltas: list[str] = []
        out_finals: list[str] = []
        for raw in ["hello", " ", "END world"]:
            accumulated_text += raw
            matched_here = None
            best_idx = None
            for s in stop:
                idx = accumulated_text.find(s)
                if idx == -1:
                    continue
                if best_idx is None or idx < best_idx:
                    best_idx = idx
                    matched_here = s
            if matched_here is not None and best_idx is not None:
                prev_published = len(accumulated_text) - len(raw)
                accumulated_text = accumulated_text[:best_idx]
                new_text = accumulated_text[prev_published:]
            else:
                new_text = raw
            out_deltas.append(new_text)
            if matched_here is not None:
                out_finals.append(accumulated_text)
                break
        assert "".join(out_deltas) == "hello "
        assert out_finals == ["hello "]
        assert "END" not in "".join(out_deltas)


class TestOpenAIStopUnchanged:
    def test_openai_chat_request_stop_field_unchanged(self):
        # Sanity check: the OpenAI ChatCompletionRequest still accepts and
        # carries a ``stop`` field exactly as before.  The Anthropic adapter
        # maps onto this field, so any drift would silently break agents.
        from vllm_mlx.api.models import ChatCompletionRequest, Message

        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            stop=["END", "STOP"],
        )
        assert req.stop == ["END", "STOP"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
