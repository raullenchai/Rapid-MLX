# SPDX-License-Identifier: Apache-2.0
"""
Output collector for streaming with low-latency optimizations.

This module implements the RequestOutputCollector pattern from vLLM,
providing non-blocking output collection with intelligent aggregation.
"""

import asyncio
import threading
from dataclasses import dataclass

from .request import RequestOutput


class RequestOutputCollector:
    """
    Per-request output collector with smart buffering.

    This class implements the vLLM pattern for efficient streaming:
    - Non-blocking get_nowait() to avoid unnecessary task switches
    - Output aggregation when producer is faster than consumer
    - Event-based signaling for efficient waiting
    - Tracking of active consumers for yield optimization

    Usage:
        collector = RequestOutputCollector()

        # Producer side (engine loop)
        collector.put(output)

        # Consumer side (streaming generator)
        output = collector.get_nowait() or await collector.get()
    """

    # Global counter of collectors with waiting consumers
    # Used to optimize: only yield when someone is waiting
    _waiting_consumers: int = 0
    _waiting_lock: threading.Lock = threading.Lock()

    def __init__(self, aggregate: bool = True):
        """
        Initialize the collector.

        Args:
            aggregate: If True, merge outputs when producer gets ahead.
                       This prevents buffer explosion under load.
        """
        self.output: RequestOutput | None = None
        self.ready = asyncio.Event()
        self.aggregate = aggregate
        self._is_waiting = False

    def put(self, output: RequestOutput) -> None:
        """
        Put an output into the collector (non-blocking).

        If aggregation is enabled and an output already exists,
        the new output is merged with the existing one.

        Args:
            output: The RequestOutput to store
        """
        if self.output is None:
            self.output = output
        elif self.aggregate:
            # Merge: combine tokens when producer is ahead
            self.output = self._merge_outputs(self.output, output)
        else:
            # Replace: just use the new output
            self.output = output
        self.ready.set()

    def get_nowait(self) -> RequestOutput | None:
        """
        Get output without blocking.

        This avoids task switching when output is available,
        reducing latency under load.

        Returns:
            The output if available, None otherwise
        """
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        return output

    async def get(self) -> RequestOutput:
        """
        Get output, blocking only if none available.

        This method blocks until an output is available.
        For low-latency streaming, prefer:
            output = collector.get_nowait() or await collector.get()

        Returns:
            The RequestOutput
        """
        # Track that we're waiting (for yield optimization)
        if not self._is_waiting:
            self._is_waiting = True
            with RequestOutputCollector._waiting_lock:
                RequestOutputCollector._waiting_consumers += 1
        try:
            while self.output is None:
                await self.ready.wait()
            output = self.get_nowait()
            # This should never be None after wait, but satisfy type checker
            assert output is not None
            return output
        finally:
            if self._is_waiting:
                self._is_waiting = False
                with RequestOutputCollector._waiting_lock:
                    RequestOutputCollector._waiting_consumers -= 1

    def _merge_outputs(
        self,
        existing: RequestOutput,
        new: RequestOutput,
    ) -> RequestOutput:
        """
        Merge two outputs when producer gets ahead of consumer.

        This combines the token lists and text, keeping the latest
        status information.

        Args:
            existing: The existing output in the buffer
            new: The new output to merge

        Returns:
            Merged RequestOutput
        """
        # Combine new tokens
        merged_new_token_ids = existing.new_token_ids + new.new_token_ids
        merged_new_text = existing.new_text + new.new_text

        # ``error`` and ``error_kind`` are a MATCHED PAIR — a discriminator is
        # only meaningful for its own error string. Select them ATOMICALLY:
        # if the newer chunk carries an error, take BOTH its ``error`` and its
        # ``error_kind`` (even when the kind is None); otherwise inherit BOTH
        # from the existing buffer. Selecting them independently could pair a
        # newer genuine runtime abort (error set, error_kind None) with an
        # older "repetition" kind, making the engine clear the runtime error
        # and return 200 instead of 503. (Both branches preserve the prior
        # ``new.error or existing.error`` behaviour for the ``error`` field.)
        if new.error:
            merged_error, merged_error_kind = new.error, new.error_kind
        else:
            merged_error, merged_error_kind = existing.error, existing.error_kind

        return RequestOutput(
            request_id=new.request_id,
            new_token_ids=merged_new_token_ids,
            new_text=merged_new_text,
            output_token_ids=new.output_token_ids,  # Use latest cumulative
            output_text=new.output_text,  # Use latest cumulative
            finished=new.finished,
            finish_reason=new.finish_reason,
            prompt_tokens=new.prompt_tokens,
            completion_tokens=new.completion_tokens,
            cached_tokens=new.cached_tokens,
            logprobs=new.logprobs,  # Use latest token's logprobs
            # Terminal scheduler failures (exact repetition, explicit abort,
            # Metal recovery) must survive aggregation. Dropping ``error`` turns
            # an aborted generation into a successful blank response; dropping
            # ``error_kind`` (the companion discriminator) turns an aggregated
            # repetition abort into a 503 that discards the partial. Both are
            # selected atomically above so a runtime error never inherits a
            # stale "repetition" kind.
            error=merged_error,
            error_kind=merged_error_kind,
            # H-03: prefer the newer chunk's matched_stop (the scheduler
            # pins it exactly once on the chunk where the stop fires);
            # fall back to the existing buf so we never demote a
            # previously-set value to None just because a later flush
            # arrived after the stop already fired. Anthropic
            # ``/v1/messages`` reads this on the FINAL output and would
            # otherwise lose the stop_sequence signal under aggregation.
            matched_stop=new.matched_stop or existing.matched_stop,
            # Request-scoped speculative metrics are terminal-only. Prefer the
            # newest terminal payload, while retaining one already buffered if
            # a later synthetic flush carries no metrics.
            spec_decode_metrics=(
                new.spec_decode_metrics or existing.spec_decode_metrics
            ),
        )

    def clear(self) -> None:
        """Clear any pending output."""
        self.output = None
        self.ready.clear()
        if self._is_waiting:
            self._is_waiting = False
            with RequestOutputCollector._waiting_lock:
                RequestOutputCollector._waiting_consumers -= 1

    @classmethod
    def has_waiting_consumers(cls) -> bool:
        """Check if any collector has waiting consumers.

        Used by engine to optimize: only yield when someone is waiting.
        """
        with cls._waiting_lock:
            return cls._waiting_consumers > 0


@dataclass
class RequestStreamState:
    """
    Tracks streaming state for a request.

    This is used to implement stream_interval batching,
    allowing tokens to be accumulated before sending.
    """

    stream_interval: int = 1
    sent_tokens: int = 0

    def should_send(self, total_tokens: int, finished: bool) -> bool:
        """
        Determine if output should be sent based on stream_interval.

        Args:
            total_tokens: Total tokens generated so far
            finished: Whether generation is complete

        Returns:
            True if output should be sent
        """
        # Always send on finish
        if finished:
            return True
        # Always send first token (for low TTFT)
        if self.sent_tokens == 0:
            return True
        # Send if we've accumulated enough tokens
        return (total_tokens - self.sent_tokens) >= self.stream_interval

    def mark_sent(self, total_tokens: int) -> None:
        """
        Update state after sending output.

        Args:
            total_tokens: Total tokens at time of send
        """
        self.sent_tokens = total_tokens
