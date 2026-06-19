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


# ---------------------------------------------------------------------------
# 0.7.41 hotfix regression — MLLM logprobs cross-thread stream crash
# ---------------------------------------------------------------------------
#
# PR #716 plumbed ``MLLMBatchResponse.logprobs`` through to the route, but
# the MLLM batch generator had been pinning its work to a stream created
# with ``mx.new_stream(mx.default_device())`` — bound to the constructing
# (worker) thread under mlx-lm 0.31.3+. The per-step logprob slice was
# tagged with that thread-local stream and any consumer thread that later
# called ``np.array(logprobs_array)`` (the route handler thread invoking
# ``service.helpers._extract_token_logprob``) aborted the process with
#
#     libc++abi: terminating due to uncaught exception of type
#     std::runtime_error: There is no Stream(gpu, 4) in current thread.
#
# Repro signature (0.7.41): qwen3-vl-2b-4bit, gemma-3n-e2b-4bit,
# gemma-3n-e4b-4bit all aborted on the FIRST chat completion that
# requested ``logprobs=true, top_logprobs=K``. Other MLLM requests on
# the same models WITHOUT logprobs worked fine — the crash was specific
# to the new logprobs cross-thread materialisation path.
#
# Fix: adopt the worker's process-wide default stream
# (``mx.default_stream(mx.default_device())``) — parity with the text
# scheduler's ``_init_mlx_step_thread``. Plus belt-and-suspenders
# ``mx.eval(logprobs)`` in ``_next()`` to make the outgoing per-step
# logprob slice non-lazy before it crosses thread boundaries.


class _RecordingVLMModel:
    """Minimal VLM stub that exposes a ``language_model`` attribute and
    accepts ``__call__`` so ``MLLMBatchGenerator.__init__`` succeeds.
    """

    def __init__(self):
        # Real VLMs set ``language_model`` on the wrapper — the generator
        # picks it up to route text-only steps through the LM directly.
        self.language_model = object()


def test_mllm_batch_generator_init_does_not_call_new_stream(monkeypatch):
    """The real ``MLLMBatchGenerator.__init__`` path must NOT call
    ``mx.new_stream(...)`` and must leave ``_stream`` pointing at the
    worker's process-wide default stream.

    Codex r1 caught the original version of this test bypassing
    ``__init__``: this revision exercises the real constructor and
    monkey-patches ``mx.new_stream`` to immediately fail, so any
    reintroduction of the buggy allocation pattern blows up the test.
    """
    import mlx.core as mx

    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

    # Reset the class-level singleton so this test exercises the
    # construction-time stream assignment regardless of test ordering.
    monkeypatch.setattr(MLLMBatchGenerator, "_stream", None)

    new_stream_calls: list[object] = []

    def _trap_new_stream(device):
        # Record + fail loudly. The fix replaced ``mx.new_stream`` with
        # ``mx.default_stream`` — if anyone reintroduces the legacy
        # allocation, ``__init__`` raises here and the test fails with
        # a clear marker rather than silently regressing.
        new_stream_calls.append(device)
        raise AssertionError(
            "MLLMBatchGenerator.__init__ called mx.new_stream — under "
            "mlx-lm 0.31.3+ the resulting stream is bound to the "
            "constructing thread and the logprobs mx.array crashes the "
            "route-handler thread on cross-thread np.array(...). "
            "Use mx.default_stream(mx.default_device()) instead."
        )

    monkeypatch.setattr(mx, "new_stream", _trap_new_stream)

    # Build the generator through the real constructor.
    gen = MLLMBatchGenerator(
        model=_RecordingVLMModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=False,
    )

    try:
        assert new_stream_calls == [], (
            "Construction must not allocate a new mx.stream. "
            "Saw mx.new_stream calls: " + repr(new_stream_calls)
        )
        # And the stream must equal the worker's process-wide default.
        expected = mx.default_stream(mx.default_device())
        assert MLLMBatchGenerator._stream is not None
        assert MLLMBatchGenerator._stream == expected, (
            "MLLMBatchGenerator._stream must be the worker default "
            "stream (process-wide, materialisable from any thread). "
            f"Got: {MLLMBatchGenerator._stream!r}, expected: {expected!r}"
        )
    finally:
        # Cleanup the singleton so other tests start clean.
        MLLMBatchGenerator._stream = None


def test_mllm_next_evals_outgoing_logprobs_before_response(monkeypatch):
    """``_next()`` must call ``mx.eval`` on the EXACT
    ``outgoing_logprobs`` array whose slice rides into
    ``MLLMBatchResponse``.

    Codex r1 caught the original version of this test only checking the
    static source for ``mx.eval(logprobs)`` — that would pass even if
    ``_next`` evaluated an unrelated variable. This revision drives the
    real ``_next()`` method with a stubbed batch, intercepts
    ``mlx.core.eval`` to record every object passed, and asserts the
    intercept saw the same array object that was attached to the
    response's ``logprobs`` field.
    """
    import mlx.core as mx

    from vllm_mlx.mllm_batch_generator import (
        MLLMBatch,
        MLLMBatchGenerator,
        MLLMBatchRequest,
        MLLMBatchStats,
    )

    # ---- Build a minimal generator that bypasses Metal + vision wiring.
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen.stop_tokens = set()
    gen.unprocessed_requests = []
    gen._shared_batch_sampler = None
    gen.completion_batch_size = 16
    gen.prefill_batch_size = 4
    gen.prefill_step_size = 1024
    gen.sampler = lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    # ``_step`` returns (sampled_tokens, list_of_logprobs_rows). Stub it
    # so we don't need the real language model — the regression target
    # is the eval / response-assembly logic, not the forward pass.
    next_step_logprobs = [mx.zeros((4,)) for _ in range(2)]

    def _fake_step(input_tokens, cache, requests):
        return (
            mx.zeros((input_tokens.shape[0],), dtype=mx.uint32),
            next_step_logprobs,
        )

    gen._step = _fake_step

    # ---- Wire an active batch carrying a SENTINEL previous-step
    # ``logprobs`` array. ``_next()`` should pop this array into
    # ``outgoing_logprobs`` and pass it to ``mx.eval`` before slicing
    # it into the per-request responses.
    sentinel_prev_logprobs = [mx.zeros((4,)), mx.zeros((4,))]
    request_a = MLLMBatchRequest(uid=0, request_id="ra", prompt="x", max_tokens=8)
    request_b = MLLMBatchRequest(uid=1, request_id="rb", prompt="y", max_tokens=8)
    gen.active_batch = MLLMBatch(
        uids=[0, 1],
        request_ids=["ra", "rb"],
        y=mx.zeros((2,), dtype=mx.uint32),
        logprobs=sentinel_prev_logprobs,
        max_tokens=[8, 8],
        num_tokens=[0, 0],
        cache=[],
        requests=[request_a, request_b],
    )

    # ---- Intercept ``mx.eval`` so the test can verify which object was
    # actually evaluated. ``mx.async_eval`` happens earlier on
    # ``batch.y`` / ``batch.logprobs`` (the NEXT step's outputs); the
    # regression target is the explicit ``mx.eval(outgoing_logprobs)``
    # call this fix added.
    eval_args: list[tuple] = []
    orig_eval = mx.eval

    def _record_eval(*args, **kwargs):
        eval_args.append(args)
        return orig_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", _record_eval)

    # ---- Drive a single step.
    responses = gen._next()

    # ---- mx.eval was called and one of the calls was on the same
    # ``logprobs`` list that the responses sliced from.
    assert eval_args, (
        "_next() did not invoke mx.eval at all — the fix that forces "
        "the outgoing per-step logprobs array to materialise on the "
        "worker thread is missing."
    )
    # Find the call whose single positional arg matches the sentinel
    # we wired into the batch. Identity check — slicing
    # ``sentinel_prev_logprobs[i]`` would still share the same outer
    # list object that mx.eval was handed.
    matched = any(
        len(args) == 1 and args[0] is sentinel_prev_logprobs for args in eval_args
    )
    assert matched, (
        "_next() called mx.eval but NOT on the outgoing logprobs array "
        "that gets sliced into MLLMBatchResponse. The regression "
        "target is the cross-thread crash on the exact per-step "
        "logprob slice — evaluating a different variable does not "
        "close the bug. Recorded mx.eval call args: "
        + repr([[type(a).__name__ for a in args] for args in eval_args])
    )

    # ---- And the responses' logprobs slice from that exact array.
    assert len(responses) == 2
    assert responses[0].logprobs is sentinel_prev_logprobs[0]
    assert responses[1].logprobs is sentinel_prev_logprobs[1]
