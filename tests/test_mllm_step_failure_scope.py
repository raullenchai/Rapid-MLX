# SPDX-License-Identifier: Apache-2.0
from collections import deque
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler  # noqa: E402
from tests.test_mllm_singleton_fastpath import (  # noqa: E402
    _make_request,
    _stub_generator,
)

OOM = "[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"


class Batch:
    def __init__(self, request):
        self.uids = [request.uid]
        self.y = mx.array([1])
        self.logprobs = mx.zeros((1, 2))
        self.cache = []
        self.requests = [request]

    def __len__(self):
        return len(self.uids)


def setup(active=True):
    gen = _stub_generator([])
    a, b, c = (_make_request(i) for i in (1, 2, 3))
    gen.active_batch = Batch(a) if active else None
    gen.unprocessed_requests = [b] if active else [b, c]
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    requests = [a, b] if active else [b, c]
    scheduler.requests = {
        r.request_id: MLLMRequest(request_id=r.request_id, prompt="hi")
        for r in requests
    }
    scheduler.running = dict(scheduler.requests)
    scheduler.waiting = deque()
    scheduler.request_id_to_uid = {r.request_id: r.uid for r in requests}
    scheduler.uid_to_request_id = {r.uid: r.request_id for r in requests}
    scheduler.finished_req_ids = set()
    scheduler._detokenizer_pool = {}
    scheduler._pending_abort_ids = set()
    scheduler._aborted_queue_ids = set()
    scheduler.batch_generator = gen
    scheduler._process_pending_aborts = MagicMock()
    scheduler._schedule_waiting = MagicMock(return_value=[])
    scheduler._record_terminal_performance = MagicMock()
    return scheduler, gen, a, b, c


def test_decode_failure_preserves_queued_request():
    scheduler, gen, a, b, _ = setup()
    gen._step = MagicMock(side_effect=RuntimeError(OOM))
    result = scheduler._step_no_queue()
    assert [(o.request_id, o.error_kind) for o in result.outputs] == [
        (a.request_id, "insufficient_memory")
    ]
    assert b.request_id in scheduler.running
    assert gen.unprocessed_requests == [b]
    assert b.request_id not in result.finished_request_ids
    processed = []
    gen._process_prompts = lambda requests: processed.extend(requests)
    gen.next()
    assert processed == [b]


def test_prefill_failure_preserves_next_queued_request():
    scheduler, gen, _, b, c = setup(active=False)
    gen._process_prompts = MagicMock(side_effect=RuntimeError(OOM))
    result = scheduler._step_no_queue()
    assert [(o.request_id, o.error_kind) for o in result.outputs] == [
        (b.request_id, "insufficient_memory")
    ]
    assert c.request_id in scheduler.running
    assert gen.unprocessed_requests == [c]
    assert c.request_id not in result.finished_request_ids


def test_fatal_failure_classifies_only_step_requests_as_oom():
    scheduler, gen, a, b, _ = setup()
    gen._step = MagicMock(side_effect=MemoryError())
    with pytest.raises(MemoryError):
        gen.next()
    gen.close = MagicMock()
    output = scheduler._fail_all_inflight(MemoryError())
    assert {o.request_id: o.error_kind for o in output.outputs} == {
        a.request_id: "insufficient_memory",
        b.request_id: "engine_aborted",
    }
