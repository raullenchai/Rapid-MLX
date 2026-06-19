# SPDX-License-Identifier: Apache-2.0
"""Regression tests: the text scheduler's stop check uses the same
``IncrementalDecoder``-backed surface as the streaming detokenizer.

Pre-fix, the stop check called ``self._decode_tokens(output_token_ids)``
which goes through ``tokenizer.decode()`` with the wrapper's default
``skip_special_tokens=True``. On tokenizer families where that default
strips text the streaming detokenizer preserves
(``skip_special_tokens=False``), the two surfaces diverged — and the
stop string the user sent NEVER appeared in the surface the stop check
scanned, so the check silently failed and the model kept generating
past the marker.

These tests use a mock tokenizer that DELIBERATELY produces different
output between ``decode()`` and the ``IncrementalDecoder``-backed
surface — to pin that the stop check is reading from the
authoritative streaming surface.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_mlx.request import Request, RequestStatus, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler() -> Scheduler:
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s.split())))
    return Scheduler(model, tokenizer, SchedulerConfig(max_num_seqs=4))


def _make_request_with_decoder(
    scheduler: Scheduler,
    rid: str,
    *,
    stop_strings: list[str],
    accumulated_full_text: str,
    decoder_new_text: str,
    prefilled_tokens: list[int],
):
    """Build a Request with a fake IncrementalDecoder whose
    ``get_full_text`` returns ``accumulated_full_text`` and whose
    ``add_token`` returns ``decoder_new_text`` on the next call.

    Critically, the tokenizer's plain ``decode()`` returns a DIFFERENT
    surface — a "broken" version that doesn't contain the stop string —
    so the test fails if the stop check falls back to ``_decode_tokens``.
    """
    sp = SamplingParams(max_tokens=100, stop=stop_strings)
    req = Request(request_id=rid, prompt="ignored", sampling_params=sp)
    req.num_prompt_tokens = 4
    req.status = RequestStatus.RUNNING
    for t in prefilled_tokens:
        req.append_output_token(t)
    # Fake incremental decoder
    decoder = MagicMock()
    decoder.get_full_text = lambda: accumulated_full_text
    decoder.add_token = lambda _t: decoder_new_text
    req._decoder = decoder
    return req


def test_stop_check_uses_incremental_decoder_surface_not_tokenizer_decode():
    """The stop check must read from the decoder's accumulated text,
    not from ``tokenizer.decode()``. Pinned by setting tokenizer.decode
    to return text that does NOT contain the stop marker while the
    decoder's surface DOES contain it."""
    scheduler = _make_scheduler()

    req = _make_request_with_decoder(
        scheduler,
        "rD",
        stop_strings=["FINIS"],
        accumulated_full_text="prefix FINIS continue",
        decoder_new_text=" continue",
        prefilled_tokens=[10, 11],
    )
    scheduler.running["rD"] = req
    scheduler.uid_to_request_id[0] = "rD"

    # tokenizer.decode REMOVES "FINIS" — simulates the
    # skip-special-tokens default skew that bit Phi-3.5 / Gemma-3n.
    scheduler._decode_tokens = lambda tokens: "prefix  continue"  # type: ignore[method-assign]

    response = MagicMock()
    response.uid = 0
    response.token = 12
    response.finish_reason = None
    response.logprobs = None
    del response.prompt_cache

    outputs, finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    out = outputs[0]
    assert out.finish_reason == "stop", (
        "Stop check did not fire — the decoder surface contains the "
        "stop marker but the check apparently fell back to "
        "tokenizer.decode() which strips it. This regression would let "
        "Phi-3.5 / Gemma-3n stream past the user's stop sequence."
    )
    assert "FINIS" not in out.output_text
    assert out.output_text == "prefix "


def test_stop_check_decoder_surface_truncates_streaming_new_text():
    """Once the stop check fires using the decoder surface, the
    emitted ``new_text`` must not leak the stop marker."""
    scheduler = _make_scheduler()
    req = _make_request_with_decoder(
        scheduler,
        "rE",
        stop_strings=["STOP"],
        accumulated_full_text="hello STOP world",
        decoder_new_text="STOP world",
        prefilled_tokens=[10],
    )
    scheduler.running["rE"] = req
    scheduler.uid_to_request_id[0] = "rE"
    scheduler._decode_tokens = lambda tokens: "hello  world"  # type: ignore[method-assign]

    response = MagicMock()
    response.uid = 0
    response.token = 11
    response.finish_reason = None
    response.logprobs = None
    del response.prompt_cache

    outputs, _finished = scheduler._process_batch_responses([response])
    assert outputs[0].finish_reason == "stop"
    assert "STOP" not in outputs[0].output_text
    assert "STOP" not in outputs[0].new_text


def test_stop_check_uses_decoder_prev_text_not_length_subtraction():
    """When the incremental decoder held back the previous token
    (``new_text == ""`` because a U+FFFD-incomplete sequence is
    pending), the stop-trim path must reconstruct the streaming
    surface from the decoder's ``prev_text`` accessor — NOT from
    ``decoded_so_far[:-len(new_text)]`` (which degenerates to the
    full surface, dropping the stop-marker truncation, codex r8
    BLOCKING)."""
    scheduler = _make_scheduler()
    req = _make_request_with_decoder(
        scheduler,
        "rF",
        stop_strings=["END"],
        accumulated_full_text="visible ENDtail",
        decoder_new_text="",  # held-back step
        prefilled_tokens=[10, 11],
    )
    # Decoder records the streaming surface as of the LAST emitted
    # delta — must be everything BEFORE the in-flight (held-back)
    # token slice. Streaming clients have only seen "visible ".
    req._decoder.prev_text = "visible "
    scheduler.running["rF"] = req
    scheduler.uid_to_request_id[0] = "rF"
    scheduler._decode_tokens = lambda tokens: "visible "  # type: ignore[method-assign]

    response = MagicMock()
    response.uid = 0
    response.token = 12
    response.finish_reason = None
    response.logprobs = None
    del response.prompt_cache

    outputs, _finished = scheduler._process_batch_responses([response])
    out = outputs[0]
    assert out.finish_reason == "stop"
    # Final output is the surface before the stop marker — must NOT
    # contain ``END`` or ``tail``.
    assert out.output_text == "visible "
    # ``new_text`` is the delta between prev_text and trimmed_total;
    # both equal "visible " so the delta is empty.
    assert out.new_text == ""
