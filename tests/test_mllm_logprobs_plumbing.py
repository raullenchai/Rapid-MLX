# SPDX-License-Identifier: Apache-2.0
"""Tests for ``logprobs`` plumbing through the MLLM scheduler.

The MLLM batch generator already produces per-token logprobs via
``MLLMBatchResponse.logprobs`` (the field has existed since launch). The
gap was at the ``_process_batch_responses`` → ``RequestOutput`` boundary:
the constructor silently dropped the field, so every MLLM chat completion
served ``logprobs=null`` even when the client asked for
``logprobs=true, top_logprobs=K``.

This unit test pins the plumbing without needing a real model load — it
drives ``MLLMScheduler._process_batch_responses`` with a mocked response
and asserts the ``RequestOutput.logprobs`` field is populated.

Affected models (qwen3-vl, gemma-3n e2b/e4b) all route through this code
path; the fix is at the scheduler layer (one constructor call) so no
per-model handling is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
from vllm_mlx.request import RequestStatus


def _make_scheduler() -> MLLMScheduler:
    model = MagicMock()
    processor = MagicMock()
    # Some attributes touched by ``_get_stop_tokens`` — keep them harmless.
    processor.tokenizer = MagicMock()
    processor.tokenizer.eos_token_id = 0
    processor.tokenizer.eos_token_ids = None
    processor.tokenizer._eos_token_ids = None
    processor.tokenizer.decode = lambda toks: "hello world"
    config = MLLMSchedulerConfig(max_num_seqs=2)
    scheduler = MLLMScheduler(model=model, processor=processor, config=config)
    return scheduler


def _make_mllm_request(scheduler: MLLMScheduler, rid: str):
    """Build a minimal ``MLLMRequest`` in the running state."""
    from vllm_mlx.mllm_scheduler import MLLMRequest

    req = MLLMRequest(
        request_id=rid,
        prompt="hi",
        num_prompt_tokens=4,
        stop=[],
    )
    req.status = RequestStatus.RUNNING
    scheduler.running[rid] = req
    return req


def test_mllm_response_logprobs_reach_request_output():
    """``MLLMBatchResponse.logprobs`` must land in
    ``RequestOutput.logprobs`` so the chat route's per-token extractor
    sees real data (not None)."""
    scheduler = _make_scheduler()
    _make_mllm_request(scheduler, "r1")
    scheduler.uid_to_request_id[0] = "r1"

    fake_logprobs = "<sentinel-logprobs>"
    response = MagicMock()
    response.uid = 0
    response.token = 42
    response.finish_reason = None
    response.logprobs = fake_logprobs

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].logprobs is fake_logprobs


def test_mllm_logprobs_field_present_even_when_response_lacks_attr():
    """Defensive — if a future MLLMBatchResponse variant drops the
    ``logprobs`` attribute the scheduler must not crash. The output
    field falls back to None (matching the pre-fix shape)."""
    scheduler = _make_scheduler()
    _make_mllm_request(scheduler, "r2")
    scheduler.uid_to_request_id[0] = "r2"

    response = MagicMock(spec=["uid", "token", "finish_reason"])
    response.uid = 0
    response.token = 7
    response.finish_reason = None

    outputs, _finished = scheduler._process_batch_responses([response])
    assert len(outputs) == 1
    assert outputs[0].logprobs is None
