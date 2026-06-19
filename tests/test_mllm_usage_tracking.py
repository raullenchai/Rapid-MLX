# SPDX-License-Identifier: Apache-2.0
"""Tests for ``prompt_tokens`` plumbing through the MLLM scheduler.

Pre-fix, the MLLM path (qwen3-vl, gemma-3n-e2b/e4b) reported
``usage.prompt_tokens=0`` on every chat completion because
``MLLMRequest.num_prompt_tokens`` had no producer — the field defaulted
to 0 and the scheduler never assigned it. The text path uses
``len(tokenizer.encode(prompt))`` to compute the count, but MLLM can't:
vision-token expansion happens inside the processor, so the real prompt
length is only known AFTER ``MLLMBatchGenerator._process_prompts`` has
run the multimodal preprocessing.

Fix shape:
- ``MLLMBatchRequest`` gained a ``num_prompt_tokens`` field stamped in
  ``_process_prompts`` from ``input_ids.size`` (snapshot BEFORE the
  per-request buffers are released to free memory).
- ``MLLMBatchResponse`` gained a ``prompt_tokens`` field that ``_next()``
  reads off the stashed ``req.num_prompt_tokens``.
- ``MLLMScheduler._process_batch_responses`` memoises the value onto
  ``MLLMRequest.num_prompt_tokens`` so every streaming chunk + the
  final response carries the real count.

These unit tests pin the plumbing without needing a real model load.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx

from vllm_mlx.mllm_batch_generator import (
    MLLMBatch,
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchResponse,
    MLLMBatchStats,
)
from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
from vllm_mlx.request import RequestStatus


def _make_scheduler() -> MLLMScheduler:
    model = MagicMock()
    processor = MagicMock()
    # Some attributes touched by ``_get_stop_tokens`` — keep harmless.
    processor.tokenizer = MagicMock()
    processor.tokenizer.eos_token_id = 0
    processor.tokenizer.eos_token_ids = None
    processor.tokenizer._eos_token_ids = None
    processor.tokenizer.decode = lambda toks: "hello world"
    config = MLLMSchedulerConfig(max_num_seqs=2)
    return MLLMScheduler(model=model, processor=processor, config=config)


def _make_mllm_request(scheduler: MLLMScheduler, rid: str):
    """Build a minimal ``MLLMRequest`` in the running state with
    ``num_prompt_tokens`` UNSET (the pre-fix default)."""
    from vllm_mlx.mllm_scheduler import MLLMRequest

    req = MLLMRequest(
        request_id=rid,
        prompt="hi",
        # Critical: default 0, mirroring the bug. The fix must populate
        # this from the MLLMBatchResponse on the first response that
        # carries a non-zero prompt_tokens value.
        num_prompt_tokens=0,
        stop=[],
    )
    req.status = RequestStatus.RUNNING
    scheduler.running[rid] = req
    return req


# ---------------------------------------------------------------------------
# Scheduler-side memoisation: first response with prompt_tokens>0 stamps
# MLLMRequest.num_prompt_tokens; later responses inherit the value.
# ---------------------------------------------------------------------------


def test_scheduler_memoises_prompt_tokens_from_first_response():
    """``MLLMScheduler._process_batch_responses`` must lift
    ``MLLMBatchResponse.prompt_tokens`` onto
    ``MLLMRequest.num_prompt_tokens`` so the RequestOutput carries the
    real count instead of the default 0.
    """
    scheduler = _make_scheduler()
    req = _make_mllm_request(scheduler, "r1")
    scheduler.uid_to_request_id[0] = "r1"

    assert req.num_prompt_tokens == 0

    response = MagicMock(spec=MLLMBatchResponse)
    response.uid = 0
    response.token = 42
    response.finish_reason = None
    response.logprobs = None
    response.prompt_tokens = 137  # text + image-patch tokens

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    # The RequestOutput must carry the real count.
    assert outputs[0].prompt_tokens == 137, (
        f"RequestOutput.prompt_tokens must reflect the value the batch "
        f"generator stamped on MLLMBatchResponse; got {outputs[0].prompt_tokens}."
    )
    # And the count must be memoised onto MLLMRequest so subsequent
    # streaming chunks inherit it without re-reading the response.
    assert req.num_prompt_tokens == 137


def test_scheduler_does_not_overwrite_existing_count():
    """Second and later responses (per request) must NOT overwrite the
    memoised ``num_prompt_tokens``. The first response per request is
    the only one that carries the snapshot from ``_process_prompts``;
    subsequent responses default to 0 and we must not clobber the real
    count with 0.
    """
    scheduler = _make_scheduler()
    req = _make_mllm_request(scheduler, "r1")
    req.num_prompt_tokens = 137  # already memoised by an earlier response
    scheduler.uid_to_request_id[0] = "r1"

    response = MagicMock(spec=MLLMBatchResponse)
    response.uid = 0
    response.token = 7
    response.finish_reason = None
    response.logprobs = None
    response.prompt_tokens = 0  # subsequent response — default zero

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].prompt_tokens == 137, "must NOT overwrite memoised count with 0"
    assert req.num_prompt_tokens == 137


def test_scheduler_handles_response_without_prompt_tokens_attr():
    """Defensive — if a future ``MLLMBatchResponse`` variant drops the
    field the scheduler must not crash. The output falls back to 0
    (matching the pre-fix shape) and the request stays at its default.
    """
    scheduler = _make_scheduler()
    req = _make_mllm_request(scheduler, "r2")
    scheduler.uid_to_request_id[0] = "r2"

    response = MagicMock(spec=["uid", "token", "finish_reason", "logprobs"])
    response.uid = 0
    response.token = 5
    response.finish_reason = None
    response.logprobs = None

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].prompt_tokens == 0
    assert req.num_prompt_tokens == 0


# ---------------------------------------------------------------------------
# Generator-side stamping: _next() reads MLLMBatchRequest.num_prompt_tokens
# (set in _process_prompts) and writes it onto every MLLMBatchResponse.
# ---------------------------------------------------------------------------


def test_next_stamps_prompt_tokens_from_request(monkeypatch):
    """``MLLMBatchGenerator._next()`` must populate
    ``MLLMBatchResponse.prompt_tokens`` from the per-request snapshot
    stashed on ``MLLMBatchRequest.num_prompt_tokens`` (set in
    ``_process_prompts`` BEFORE input_ids is released).

    This is the same harness shape as
    ``test_mllm_logprobs_plumbing.test_mllm_next_evals_outgoing_logprobs_
    before_response`` — bypass Metal + vision wiring and drive the real
    ``_next()`` method.
    """
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen.stop_tokens = set()
    gen.unprocessed_requests = []
    gen._shared_batch_sampler = None
    gen.completion_batch_size = 16
    gen.prefill_batch_size = 4
    gen.prefill_step_size = 1024
    gen.sampler = lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    def _fake_step(input_tokens, cache, requests):
        return (
            mx.zeros((input_tokens.shape[0],), dtype=mx.uint32),
            [mx.zeros((4,)) for _ in range(input_tokens.shape[0])],
        )

    gen._step = _fake_step

    # Two requests with DIFFERENT prompt-token counts so we catch any
    # off-by-one or index swap. Vision-heavy requests have larger counts
    # (image-patch expansion), text-only requests are smaller.
    request_a = MLLMBatchRequest(
        uid=0, request_id="ra", prompt="x", max_tokens=8, num_prompt_tokens=259
    )
    request_b = MLLMBatchRequest(
        uid=1, request_id="rb", prompt="y", max_tokens=8, num_prompt_tokens=12
    )

    gen.active_batch = MLLMBatch(
        uids=[0, 1],
        request_ids=["ra", "rb"],
        y=mx.zeros((2,), dtype=mx.uint32),
        logprobs=[mx.zeros((4,)), mx.zeros((4,))],
        max_tokens=[8, 8],
        num_tokens=[0, 0],
        cache=[],
        requests=[request_a, request_b],
    )

    responses = gen._next()
    assert len(responses) == 2
    # Per-request stamping — each response carries the count of the
    # request that produced it. The bug was either "always 0" (no
    # stamping) or "always the same" (constant shared count).
    assert responses[0].prompt_tokens == 259, (
        f"First response must carry request_a's prompt_tokens=259; "
        f"got {responses[0].prompt_tokens}."
    )
    assert responses[1].prompt_tokens == 12, (
        f"Second response must carry request_b's prompt_tokens=12; "
        f"got {responses[1].prompt_tokens}."
    )


# ---------------------------------------------------------------------------
# End-to-end shape: MLLMBatchResponse + MLLMRequest fields exist.
# Cheap dataclass-existence assertion so a future refactor that drops
# either field fails noisily here instead of silently re-zeroing usage.
# ---------------------------------------------------------------------------


def test_dataclass_fields_present():
    """Pin the wire-shape additions. ``MLLMBatchResponse.prompt_tokens``
    and ``MLLMBatchRequest.num_prompt_tokens`` are the two storage slots
    the fix relies on; dropping either re-introduces the bug.
    """
    resp = MLLMBatchResponse(uid=1, request_id="x", token=5, logprobs=None)
    assert hasattr(resp, "prompt_tokens")
    assert resp.prompt_tokens == 0  # default

    req = MLLMBatchRequest(uid=1, request_id="x", prompt="hi")
    assert hasattr(req, "num_prompt_tokens")
    assert req.num_prompt_tokens == 0  # default
