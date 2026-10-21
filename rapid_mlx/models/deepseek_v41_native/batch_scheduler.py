"""Draft-off continuous batching with a single owner of all model work."""

from __future__ import annotations

import copy
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from typing import Any

import mlx.core as mx
from fastapi import HTTPException

from .cache import BatchModelCache
from .lanes import request_lanes  # noqa: F401  (re-export)
from .serving import (
    MAX_INPUT_TOKENS,
    GenerationChunk,
    GenerationResult,
    NaiveStreamingDetokenizer,
    generation_kwargs,
)

logger = logging.getLogger(__name__)


def _own_copy(error: Exception) -> Exception:
    """One instance per consumer, so concurrent re-raises cannot share tracebacks."""
    if isinstance(error, HTTPException):
        own: Exception = HTTPException(
            status_code=error.status_code, detail=error.detail, headers=error.headers
        )
    else:
        try:
            own = copy.copy(error)
        except Exception:  # exceptions whose constructor cannot be replayed
            own = RuntimeError(f"{type(error).__name__}: {error}")
    # no chaining: the shared original (and its traceback) stays with the log
    own.__cause__ = None
    own.__suppress_context__ = True
    return own


_END = object()


@dataclass(eq=False)
class _Request:
    prompt: str | list[int]
    max_tokens: int
    ignore_eos: bool
    cancel_event: Any
    closed: threading.Event = field(default_factory=threading.Event)
    output: Queue = field(default_factory=Queue)
    ids: list[int] = field(default_factory=list)
    cache: Any = None
    detokenizer: Any = None
    emitted_text: str = ""
    generated: int = 0
    token: int = 0
    done: bool = False

    @property
    def cancelled(self):
        return self.closed.is_set() or (
            self.cancel_event is not None and self.cancel_event.is_set()
        )

    def finish(self, error=None):
        if not self.done:
            self.done = True
            self.cache = None
            self.output.put(error.with_traceback(None) if error is not None else _END)


class TokenStream:
    """Closeable iterator; closing even before the first read cancels admission."""

    def __init__(self, request, wake):
        self._request = request
        self._wake = wake
        self._ended = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._ended:
            raise StopIteration
        item = self._request.output.get()
        if item is _END or isinstance(item, Exception):
            self._ended = True
            if isinstance(item, Exception):
                raise item
            raise StopIteration
        return item

    def close(self):
        self._ended = True
        self._request.closed.set()
        self._wake()


class BatchBackend:
    """FIFO admission, chunked single-row prefill and per-row greedy decode.

    ``executor`` only runs queue consumers for the HTTP adapter. The separate
    single-worker model executor loads (when supplied a loader), evaluates and
    releases caches on the same thread throughout the backend's lifetime.
    """

    def __init__(self, model=None, tokenizer=None, *, lanes=2, load_fn=None):
        if lanes not in (1, 2, 4, 8) or isinstance(lanes, bool):
            raise ValueError("lanes must be one of 1, 2, 4, 8")
        self.lanes = lanes
        self._condition = threading.Condition()
        self._pending = deque()
        self._closed = False
        self._model_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dsv41-model"
        )
        self.executor = self._executor = None
        try:
            if load_fn is not None:
                model, tokenizer, self.runtime = self._model_executor.submit(
                    load_fn
                ).result()
            self.model, self.tokenizer = model, tokenizer
            self.executor = self._executor = ThreadPoolExecutor(
                max_workers=lanes + 1, thread_name_prefix="dsv41-stream"
            )
            self._loop = self._model_executor.submit(self._run)
        except BaseException:
            self._model_executor.shutdown(wait=True, cancel_futures=True)
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            raise

    def _wake(self):
        with self._condition:
            self._condition.notify_all()

    def submit(
        self,
        prompt,
        *,
        max_tokens,
        ignore_eos=False,
        cancel_event=None,
        temperature=0.0,
        top_p=1.0,
        **_kwargs,
    ):
        generation_kwargs(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        request = _Request(
            list(prompt) if isinstance(prompt, list) else prompt,
            max_tokens,
            ignore_eos,
            cancel_event,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("batch backend is closed")
            self._pending.append(request)
            self._condition.notify_all()
        return TokenStream(request, self._wake)

    def stream_generate(self, _model, _processor, prompt, **kwargs):
        return self.submit(prompt, **kwargs)

    def generate(self, _model, _processor, prompt, **kwargs):
        stream = self.submit(prompt, **kwargs)
        try:
            chunks = list(stream)
            return GenerationResult(
                "".join(chunk.text for chunk in chunks),
                chunks[-1].prompt_tokens if chunks else 0,
                chunks[-1].generation_tokens if chunks else 0,
            )
        finally:
            stream.close()

    def _prepare(self, request):
        request.ids = (
            request.prompt
            if isinstance(request.prompt, list)
            else self.tokenizer.encode(request.prompt, add_special_tokens=False)
        )
        if len(request.ids) < 2:
            raise ValueError("prompt must contain at least two tokens")
        if len(request.ids) > MAX_INPUT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"DeepSeek V4.1 Flash currently supports at most {MAX_INPUT_TOKENS} input tokens in the qualified product lane",
            )
        request.cache = self.model.make_cache(
            max_seq_len=len(request.ids) + request.max_tokens + 8
        )
        request.detokenizer = NaiveStreamingDetokenizer(self.tokenizer)

    def _emit(self, request, token):
        request.token = token
        request.generated += 1
        eos = token == self.tokenizer.eos_token_id
        ending = (
            eos and not request.ignore_eos
        ) or request.generated >= request.max_tokens
        if not eos:
            request.detokenizer.add_token(token)
        if ending:
            request.detokenizer.finalize()
        current = request.detokenizer.text
        text = (
            current[len(request.emitted_text) :]
            if current.startswith(request.emitted_text)
            else ""
        )
        if text:
            request.emitted_text = current
        request.output.put(
            GenerationChunk(text, token, len(request.ids), request.generated)
        )
        if ending:
            request.finish()

    def _filter(self, rows, batch):
        keep = []
        for index, request in enumerate(rows):
            if request.cancelled:
                request.finish()
            if not request.done:
                keep.append(index)
        if len(keep) != len(rows):
            batch = batch.filter(keep) if keep else None
            rows = [rows[index] for index in keep]
        return rows, batch

    def _run(self):
        rows, batch, prefill = [], None, None
        try:
            while True:
                with self._condition:
                    if self._closed:
                        break
                    waiting = deque()
                    for request in self._pending:
                        if request.cancelled:
                            request.finish()
                        else:
                            waiting.append(request)
                    self._pending = waiting
                    if not rows and prefill is None and not waiting:
                        self._condition.wait()
                        continue
                try:
                    rows, batch = self._filter(rows, batch)
                    if prefill is not None and prefill.cancelled:
                        prefill.finish()
                        prefill = None
                    with self._condition:
                        if prefill is None and len(rows) < self.lanes and self._pending:
                            prefill = self._pending.popleft()
                    if prefill is not None:
                        try:
                            if prefill.cache is None:
                                self._prepare(prefill)
                            start = prefill.cache.offset
                            logits = self.model(
                                mx.array([prefill.ids[start : start + 512]]),
                                prefill.cache,
                                last_logit_only=True,
                            )
                            mx.eval(logits)
                            if prefill.cache.offset == len(prefill.ids):
                                token = int(mx.argmax(logits[:, -1]).item())
                                # Build the cache before emitting so a merge failure
                                # cannot leave an apparently successful request.
                                eos = token == self.tokenizer.eos_token_id
                                if prefill.max_tokens > 1 and (
                                    prefill.ignore_eos or not eos
                                ):
                                    joined = BatchModelCache.from_caches(
                                        [prefill.cache]
                                    )
                                    batch = (
                                        joined
                                        if batch is None
                                        else batch.extend(joined)
                                    )
                                    del joined
                                    rows.append(prefill)
                                # cancelled while its prefill ran: no token
                                if prefill.cancelled:
                                    prefill.finish()
                                else:
                                    self._emit(prefill, token)
                                prefill.cache = None
                                prefill = None
                        except Exception as error:
                            logger.exception("DeepSeek V4.1 prefill failed")
                            prefill.finish(error)
                            prefill = None
                    # Newly finished/cancelled rows must not consume a decode step.
                    rows, batch = self._filter(rows, batch)
                    if rows:
                        # Model.__call__ hashes all row histories and prefetches
                        # their combined Engram indices before layer execution.
                        logits = self.model(
                            mx.array([[request.token] for request in rows]),
                            batch,
                            last_logit_only=True,
                        )
                        tokens = mx.argmax(logits, axis=-1)
                        mx.eval(tokens)
                        for request, token in zip(rows, tokens.tolist()):
                            # cancelled during the step: finish without a token
                            if request.cancelled:
                                request.finish()
                                continue
                            # Detokenizer failures end only their own request;
                            # the next _filter compacts it out of the batch.
                            try:
                                self._emit(request, int(token))
                            except Exception as error:
                                logger.exception("DeepSeek V4.1 token emission failed")
                                request.finish(error)
                except Exception as error:
                    # A shared forward cannot attribute its failure to one row,
                    # so every row of the failed step ends, each with its own
                    # exception instance. Per-request failures are isolated at
                    # admission, prefill and token emission; a partial prefill
                    # owns its cache and took no part in the step, so it keeps
                    # going (its own chunk failure ends only it).
                    logger.exception("DeepSeek V4.1 batch step failed")
                    for request in rows:
                        request.finish(_own_copy(error))
                    rows, batch = [], None
        finally:
            with self._condition:
                self._closed = True
                pending, self._pending = self._pending, deque()
            for request in [*rows, *pending, *([prefill] if prefill else [])]:
                request.finish(RuntimeError("batch backend is closed"))
            # Cache references, including partial prefill, die on the owner thread.
            rows, batch, prefill = [], None, None
            # Drain asynchronous kernels before the owner thread and its
            # thread-local stream/encoder resources are destroyed.
            mx.synchronize()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._model_executor.shutdown(wait=True, cancel_futures=False)
        self.executor.shutdown(wait=True, cancel_futures=True)
