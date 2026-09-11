"""Regression contracts for MLLM tokenizer execution-domain isolation."""

from __future__ import annotations

import copy
import threading

from vllm_mlx.engine.batched import _clone_mllm_worker_processor


class _BorrowCheckedTokenizer:
    """Model the exclusive mutable borrow used by a fast-tokenizer backend."""

    def __init__(self) -> None:
        self._borrowed = False
        self._state_lock = threading.Lock()

    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        return clone

    def enter_mutation(self) -> None:
        with self._state_lock:
            if self._borrowed:
                raise RuntimeError("Already borrowed")
            self._borrowed = True

    def leave_mutation(self) -> None:
        with self._state_lock:
            self._borrowed = False


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _BorrowCheckedTokenizer()
        self.image_processor = object()


def test_mllm_worker_processor_owns_an_independent_tokenizer_backend():
    request_processor = _Processor()

    worker_processor = _clone_mllm_worker_processor(request_processor)

    assert worker_processor is not request_processor
    assert worker_processor.tokenizer is not request_processor.tokenizer
    assert worker_processor.image_processor is request_processor.image_processor

    # Hold the request-side mutable borrow while the worker enters its own.
    # Reusing the original tokenizer would deterministically raise the exact
    # ``Already borrowed`` failure observed in #3303.
    request_processor.tokenizer.enter_mutation()
    try:
        worker_processor.tokenizer.enter_mutation()
        worker_processor.tokenizer.leave_mutation()
    finally:
        request_processor.tokenizer.leave_mutation()


def test_tokenizer_like_processor_is_deep_copied():
    request_tokenizer = _BorrowCheckedTokenizer()

    worker_tokenizer = _clone_mllm_worker_processor(request_tokenizer)

    assert worker_tokenizer is not request_tokenizer
    request_tokenizer.enter_mutation()
    try:
        worker_tokenizer.enter_mutation()
        worker_tokenizer.leave_mutation()
    finally:
        request_tokenizer.leave_mutation()


def test_processor_copy_does_not_mutate_request_side_tokenizer():
    request_processor = _Processor()
    original_tokenizer = request_processor.tokenizer

    worker_processor = _clone_mllm_worker_processor(request_processor)

    assert request_processor.tokenizer is original_tokenizer
    assert worker_processor.tokenizer is not original_tokenizer
    assert copy.deepcopy(worker_processor.tokenizer) is not worker_processor.tokenizer
