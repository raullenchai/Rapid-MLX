# SPDX-License-Identifier: Apache-2.0
"""Regression for #1367 — a generation-step exception whose type is not
``ValueError``/``RuntimeError`` must fail the running batch, not hang.

Background
----------
``MLLMScheduler._step_no_queue`` runs the model forward pass via
``self.batch_generator.next()``. That call used to be wrapped in a narrow
``except (ValueError, RuntimeError)`` that fails all running requests
(emits an error output, evicts them from the batch generator, cleans up)
so the loop advances instead of retrying a poisoned batch forever.

A checkpoint whose mlx-vlm model class drifts from the batch generator's
call convention raises ``TypeError`` from the forward pass — the reported
repro was ``Model.__call__() missing 1 required positional argument:
'mask'`` on ``ministral-3b``. ``TypeError`` is neither ``ValueError`` nor
``RuntimeError``, so it escaped the handler, propagated to
``_process_loop``'s outer ``except Exception``, and the loop logged +
retried the SAME batch every step (~565 identical lines in one 90 s
client request). The waiting request's ``output_queue`` never received an
output or the ``None`` sentinel, so from the caller it was an unbounded
silent hang (``/v1/status`` stuck at ``steps_executed: 0``).

Same class as the F-061 image-decode ``OSError``
(``tests/test_mllm_corrupt_image.py``). The fix broadens the handler to
catch every exception from ``batch_generator.next()`` so future
call-convention drift fails closed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_mlx.mllm_scheduler import (
    MLLMRequest,
    MLLMScheduler,
    MLLMSchedulerConfig,
)
from vllm_mlx.request import RequestStatus, SamplingParams


class _SignatureDriftBatchGenerator:
    """Fake batch generator whose forward pass raises ``TypeError`` — the
    #1367 model call-convention drift — and records ``remove()`` calls so
    the test can assert the poisoned request is evicted.
    """

    MESSAGE = "Model.__call__() missing 1 required positional argument: 'mask'"

    def __init__(self) -> None:
        self.next_calls = 0
        self.removed_uids: list[int] = []

    def next(self):
        self.next_calls += 1
        raise TypeError(self.MESSAGE)

    def remove(self, uids):
        self.removed_uids.extend(uids)


def _make_scheduler_with_running_request(request_id: str = "req-drift", uid: int = 0):
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()

    scheduler = MLLMScheduler(mock_model, mock_processor, MLLMSchedulerConfig())

    request = MLLMRequest(
        request_id=request_id,
        prompt="describe this",
        sampling_params=SamplingParams(max_tokens=16),
    )
    request.status = RequestStatus.RUNNING
    scheduler.requests[request_id] = request
    scheduler.running[request_id] = request
    scheduler.request_id_to_uid[request_id] = uid
    scheduler.uid_to_request_id[uid] = request_id

    bg = _SignatureDriftBatchGenerator()
    scheduler.batch_generator = bg
    return scheduler, request, bg


def test_typeerror_from_forward_pass_fails_request_instead_of_raising():
    """A ``TypeError`` from ``batch_generator.next()`` must be caught and
    turned into a finished error output — not propagate out of the step.
    """
    scheduler, _request, bg = _make_scheduler_with_running_request()

    # Before the fix this raised TypeError straight out of the step.
    output = scheduler._step_no_queue()

    assert bg.next_calls == 1
    # The poisoned request is reported finished so the waiting client
    # gets a terminal signal instead of hanging.
    finished = [o for o in output.outputs if o.request_id == "req-drift"]
    assert len(finished) == 1
    assert finished[0].finished is True
    assert "req-drift" in output.finished_request_ids


def test_poisoned_request_is_evicted_so_the_loop_cannot_respin():
    """After the failing step the request must be gone from the batch
    generator and scheduler bookkeeping, so a subsequent step does NOT
    re-run the same crashing forward pass (the #1367 hang).
    """
    scheduler, _request, bg = _make_scheduler_with_running_request()

    scheduler._step_no_queue()

    # Evicted from the batch generator (remove() called with its uid)...
    assert bg.removed_uids == [0]
    # ...and from every scheduler map, so it can never be scheduled again.
    assert "req-drift" not in scheduler.running
    assert "req-drift" not in scheduler.requests
    assert "req-drift" not in scheduler.request_id_to_uid
    assert 0 not in scheduler.uid_to_request_id

    # A second step has no running request, so the forward pass is not
    # invoked again — the pre-fix behaviour retried it every step forever.
    bg.next_calls = 0
    scheduler._step_no_queue()
    assert bg.next_calls == 0


def test_valueerror_client_error_path_still_surfaces_message():
    """Broadening the ``except`` must not regress the existing
    client-error classification: an oversized-image ``ValueError`` still
    surfaces its message (route layer maps it to HTTP 400).
    """
    scheduler, _request, bg = _make_scheduler_with_running_request()

    def _raise_client_error():
        bg.next_calls += 1
        raise ValueError("Failed to process image: cannot identify image file")

    bg.next = _raise_client_error  # type: ignore[method-assign]

    output = scheduler._step_no_queue()

    finished = [o for o in output.outputs if o.request_id == "req-drift"]
    assert len(finished) == 1
    assert finished[0].finished is True
    assert finished[0].error is not None
    assert "Failed to process image" in finished[0].error
    assert finished[0].finish_reason == "error"
