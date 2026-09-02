# SPDX-License-Identifier: Apache-2.0
"""Per-model request performance counters for the text scheduler.

The scheduler already records process-lifetime token and cancellation counters,
but those aggregates are not enough for a per-model inspector: they cannot say
which requests failed, how long the first token took, or how fast a request
decoded after that first token.  This ledger keeps the small amount of additional
state needed by :mod:`vllm_mlx.routes.metrics` without changing request
semantics.

Schedulers obtain their ledger from a process-owned, per-model registry. Model
reloads therefore retain terminal events even when they occur between Prometheus
scrapes, while a sidecar process restart starts fresh counters in the standard
Prometheus fashion. The series carry the model label so operators can separate
models without coupling metric ownership to a replaceable scheduler instance.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm_mlx.request import Request

# Prometheus histograms need fixed buckets.  These cover the local-model
# operating range without making every bucket width a UI policy: TTFT spans
# warm-cache requests through long cold prefills, and decode speed spans tiny
# dense models through large MoE deployments.
TTFT_SECONDS_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    math.inf,
)

DECODE_TOKENS_PER_SECOND_BUCKETS = (
    1.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    math.inf,
)

# Keep duplicate-delivery dedupe bounded for 24/7 servers while retaining
# enough recent IDs to absorb duplicate terminal/cancellation events.
SEEN_REQUEST_ID_LIMIT = 65_536
MODEL_LEDGER_REGISTRY_LIMIT = 128
RETIRED_MODEL_SNAPSHOT_LIMIT = 128


def _empty_bucket_counts(buckets: tuple[float, ...]) -> dict[str, int]:
    """Return zeroed cumulative histogram buckets."""
    counts = dict.fromkeys(
        (
            "+Inf" if math.isinf(bucket) else _format_bucket(bucket)
            for bucket in buckets
        ),
        0,
    )

    return counts


def _bucket_count(
    counts: dict[str, int],
    value: float,
    buckets: tuple[float, ...],
) -> None:
    """Add one finite, non-negative value to cumulative histogram buckets."""
    for bucket in buckets:
        label = "+Inf" if math.isinf(bucket) else _format_bucket(bucket)
        if value <= bucket:
            counts[label] += 1


def _format_bucket(value: float) -> str:
    return f"{value:g}"


@dataclass(frozen=True)
class ModelPerformanceSnapshot:
    """An immutable Prometheus-ready view of one model's request outcomes."""

    model_name: str
    requests_succeeded: int
    requests_cancelled: int
    requests_failed: int
    prompt_tokens: int
    completion_tokens: int
    ttft_bucket_counts: dict[str, int]
    ttft_seconds_count: int
    ttft_seconds_sum: float
    ttft_seconds_max: float | None
    decode_bucket_counts: dict[str, int]
    decode_observations: int
    decode_tokens_per_second_sum: float
    decode_tokens_per_second_max: float | None
    last_decode_tokens_per_second: float | None

    @property
    def total_requests(self) -> int:
        return self.requests_succeeded + self.requests_cancelled + self.requests_failed


class ModelPerformanceLedger:
    """Thread-safe, process-lifetime performance observations for one model."""

    def __init__(
        self,
        model_name: str | None = None,
        baseline: ModelPerformanceSnapshot | None = None,
    ):
        self._model_name = model_name or ""
        self._lock = threading.Lock()
        self._seen_request_ids: OrderedDict[str | tuple[str, float], None] = (
            OrderedDict()
        )
        self._requests_succeeded = baseline.requests_succeeded if baseline else 0
        self._requests_cancelled = baseline.requests_cancelled if baseline else 0
        self._requests_failed = baseline.requests_failed if baseline else 0
        self._prompt_tokens = baseline.prompt_tokens if baseline else 0
        self._completion_tokens = baseline.completion_tokens if baseline else 0
        self._ttft_bucket_counts = (
            dict(baseline.ttft_bucket_counts)
            if baseline
            else _empty_bucket_counts(TTFT_SECONDS_BUCKETS)
        )
        self._ttft_observations = baseline.ttft_seconds_count if baseline else 0
        self._ttft_seconds_sum = baseline.ttft_seconds_sum if baseline else 0.0
        self._ttft_seconds_max = baseline.ttft_seconds_max if baseline else None
        self._decode_bucket_counts = (
            dict(baseline.decode_bucket_counts)
            if baseline
            else _empty_bucket_counts(DECODE_TOKENS_PER_SECOND_BUCKETS)
        )
        self._decode_observations = baseline.decode_observations if baseline else 0
        self._decode_tokens_per_second_sum = (
            baseline.decode_tokens_per_second_sum if baseline else 0.0
        )
        self._decode_tokens_per_second_max = (
            baseline.decode_tokens_per_second_max if baseline else None
        )
        self._last_decode_tokens_per_second = (
            baseline.last_decode_tokens_per_second if baseline else None
        )

    def decode_rate_for_request(self, request: Request) -> float | None:
        """Return inverse-TPOT decode speed, or None when not measurable."""
        if request.first_token_time is None or request.num_output_tokens < 2:
            return None
        decode_seconds = time.time() - request.first_token_time
        if decode_seconds <= 0:
            return None
        return (request.num_output_tokens - 1) / decode_seconds

    def ttft_for_request(self, request: Request) -> float | None:
        if request.first_token_time is None or request.num_output_tokens == 0:
            return None
        return max(0.0, request.first_token_time - request.arrival_time)

    @staticmethod
    def prompt_tokens_for_request(request: Any) -> int:
        """Return prompt work attributable to the model for this lifetime."""
        status = getattr(request, "status", None)
        if getattr(status, "name", None) == "WAITING":
            return 0
        model_prompt_tokens = int(getattr(request, "model_prompt_tokens", 0) or 0)
        if model_prompt_tokens > 0:
            return model_prompt_tokens
        return int(request.num_prompt_tokens)

    def record_finished_performance(self, request: Request) -> None:
        """Best-effort performance accounting for a terminal response."""
        try:
            self.record_request_performance(
                request,
                "succeeded",
                ttft_seconds=self.ttft_for_request(request),
                decode_tokens_per_second=self.decode_rate_for_request(request),
            )
        except Exception:
            logger.debug("Failed to record performance for %s", request.request_id)

    def record_cancelled_performance(self, request: Request) -> None:
        """Best-effort performance accounting for an aborted request."""
        try:
            self.record_request_performance(
                request,
                "cancelled",
                ttft_seconds=self.ttft_for_request(request),
                decode_tokens_per_second=self.decode_rate_for_request(request),
            )
        except Exception:
            logger.debug("Failed to record cancellation for %s", request.request_id)

    @property
    def model_name(self) -> str:
        return self._model_name

    def record_request_performance(
        self,
        request: Any,
        outcome: str,
        *,
        ttft_seconds: float | None = None,
        decode_tokens_per_second: float | None = None,
    ) -> bool:
        """Atomically account one request object exactly once for its lifetime."""
        with self._lock:
            if getattr(request, "_performance_recorded", False):
                return False
            request._performance_recorded = True
            prompt_tokens = max(0, self.prompt_tokens_for_request(request))
            completion_tokens = max(0, int(request.num_output_tokens))
            if outcome == "succeeded":
                self._requests_succeeded += 1
            elif outcome == "cancelled":
                self._requests_cancelled += 1
            elif outcome == "failed":
                self._requests_failed += 1
            else:
                request._performance_recorded = False
                raise ValueError(f"unsupported terminal outcome: {outcome}")
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._observe_timings(
                ttft_seconds=ttft_seconds,
                decode_tokens_per_second=decode_tokens_per_second,
            )
        _touch_model_performance_ledger(self)
        return True

    def record_success(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_seconds: float | None,
        decode_tokens_per_second: float | None,
        request_lifetime: float | None = None,
    ) -> bool:
        """Record a completed request; return False when already accounted."""
        with self._lock:
            request_key = self._request_key(request_id, request_lifetime)
            if request_key in self._seen_request_ids:
                self._seen_request_ids.move_to_end(request_key)
                return False
            prompt_tokens = max(0, int(prompt_tokens))
            completion_tokens = max(0, int(completion_tokens))
            self._remember_request_id(request_key)
            self._requests_succeeded += 1
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._observe_timings(
                ttft_seconds=ttft_seconds,
                decode_tokens_per_second=decode_tokens_per_second,
            )
            return True

    def record_cancelled(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_seconds: float | None,
        decode_tokens_per_second: float | None,
        request_lifetime: float | None = None,
    ) -> bool:
        """Record an explicitly cancelled request exactly once."""
        with self._lock:
            request_key = self._request_key(request_id, request_lifetime)
            if request_key in self._seen_request_ids:
                self._seen_request_ids.move_to_end(request_key)
                return False
            prompt_tokens = max(0, int(prompt_tokens))
            completion_tokens = max(0, int(completion_tokens))
            self._remember_request_id(request_key)
            self._requests_cancelled += 1
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._observe_timings(
                ttft_seconds=ttft_seconds,
                decode_tokens_per_second=decode_tokens_per_second,
            )
            return True

    def record_failure(
        self,
        request_id: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttft_seconds: float | None = None,
        decode_tokens_per_second: float | None = None,
        request_lifetime: float | None = None,
    ) -> bool:
        """Record an engine/runtime failure exactly once."""
        with self._lock:
            request_key = self._request_key(request_id, request_lifetime)
            if request_key in self._seen_request_ids:
                self._seen_request_ids.move_to_end(request_key)
                return False
            prompt_tokens = max(0, int(prompt_tokens))
            completion_tokens = max(0, int(completion_tokens))
            self._remember_request_id(request_key)
            self._requests_failed += 1
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._observe_timings(
                ttft_seconds=ttft_seconds,
                decode_tokens_per_second=decode_tokens_per_second,
            )
            return True

    def record_failed_performance(self, request: Request) -> bool:
        """Record one failed request lifetime, even when its ID is later reused."""
        return self.record_request_performance(
            request,
            "failed",
            ttft_seconds=self.ttft_for_request(request),
            decode_tokens_per_second=self.decode_rate_for_request(request),
        )

    def snapshot(self) -> ModelPerformanceSnapshot:
        """Return a coherent copy of the counters."""
        with self._lock:
            return ModelPerformanceSnapshot(
                model_name=self._model_name,
                requests_succeeded=self._requests_succeeded,
                requests_cancelled=self._requests_cancelled,
                requests_failed=self._requests_failed,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                ttft_bucket_counts=dict(self._ttft_bucket_counts),
                ttft_seconds_count=self._ttft_observations,
                ttft_seconds_sum=self._ttft_seconds_sum,
                ttft_seconds_max=self._ttft_seconds_max,
                decode_bucket_counts=dict(self._decode_bucket_counts),
                decode_observations=self._decode_observations,
                decode_tokens_per_second_sum=self._decode_tokens_per_second_sum,
                decode_tokens_per_second_max=self._decode_tokens_per_second_max,
                last_decode_tokens_per_second=self._last_decode_tokens_per_second,
            )

    @staticmethod
    def _request_key(
        request_id: str, request_lifetime: float | None
    ) -> str | tuple[str, float]:
        return (
            request_id if request_lifetime is None else (request_id, request_lifetime)
        )

    def _remember_request_id(self, request_id: str | tuple[str, float]) -> None:
        """Store a terminal request ID under a bounded memory limit."""
        self._seen_request_ids[request_id] = None
        if len(self._seen_request_ids) > SEEN_REQUEST_ID_LIMIT:
            self._seen_request_ids.popitem(last=False)

    def _observe_timings(
        self,
        *,
        ttft_seconds: float | None,
        decode_tokens_per_second: float | None,
    ) -> None:
        if (
            ttft_seconds is not None
            and math.isfinite(ttft_seconds)
            and ttft_seconds >= 0
        ):
            _bucket_count(
                self._ttft_bucket_counts,
                ttft_seconds,
                TTFT_SECONDS_BUCKETS,
            )
            self._ttft_observations += 1
            self._ttft_seconds_sum += ttft_seconds
            if self._ttft_seconds_max is None or ttft_seconds > self._ttft_seconds_max:
                self._ttft_seconds_max = ttft_seconds

        if (
            decode_tokens_per_second is not None
            and math.isfinite(decode_tokens_per_second)
            and decode_tokens_per_second >= 0
        ):
            _bucket_count(
                self._decode_bucket_counts,
                decode_tokens_per_second,
                DECODE_TOKENS_PER_SECOND_BUCKETS,
            )
            self._decode_observations += 1
            self._decode_tokens_per_second_sum += decode_tokens_per_second
            self._last_decode_tokens_per_second = decode_tokens_per_second
            if (
                self._decode_tokens_per_second_max is None
                or decode_tokens_per_second > self._decode_tokens_per_second_max
            ):
                self._decode_tokens_per_second_max = decode_tokens_per_second


# Scheduler instances are replaceable; Prometheus counters are not. Keep one
# process-owned ledger per model so terminal events completed between scrapes
# survive an unload/reload of that model.
_MODEL_LEDGER_REGISTRY: OrderedDict[str, ModelPerformanceLedger] = OrderedDict()
_RETIRED_MODEL_SNAPSHOTS: OrderedDict[
    str,
    tuple[
        ModelPerformanceSnapshot,
        weakref.ReferenceType[ModelPerformanceLedger],
    ],
] = OrderedDict()
_MODEL_LEDGER_REGISTRY_LOCK = threading.Lock()


def _retire_oldest_model_ledger_locked() -> None:
    if len(_MODEL_LEDGER_REGISTRY) <= MODEL_LEDGER_REGISTRY_LIMIT:
        return
    key, ledger = _MODEL_LEDGER_REGISTRY.popitem(last=False)
    _RETIRED_MODEL_SNAPSHOTS[key] = (ledger.snapshot(), weakref.ref(ledger))
    _RETIRED_MODEL_SNAPSHOTS.move_to_end(key)
    if len(_RETIRED_MODEL_SNAPSHOTS) > RETIRED_MODEL_SNAPSHOT_LIMIT:
        _RETIRED_MODEL_SNAPSHOTS.popitem(last=False)


def _touch_model_performance_ledger(ledger: ModelPerformanceLedger) -> None:
    """Make a writing ledger visible, including after an LRU retirement."""
    key = ledger.model_name
    with _MODEL_LEDGER_REGISTRY_LOCK:
        current = _MODEL_LEDGER_REGISTRY.get(key)
        if current is ledger:
            _MODEL_LEDGER_REGISTRY.move_to_end(key)
            return
        _RETIRED_MODEL_SNAPSHOTS.pop(key, None)
        _MODEL_LEDGER_REGISTRY[key] = ledger
        _MODEL_LEDGER_REGISTRY.move_to_end(key)
        _retire_oldest_model_ledger_locked()


def get_model_performance_ledger(
    model_name: str | None = None,
) -> ModelPerformanceLedger:
    key = model_name or ""
    with _MODEL_LEDGER_REGISTRY_LOCK:
        ledger = _MODEL_LEDGER_REGISTRY.get(key)
        if ledger is None:
            retired = _RETIRED_MODEL_SNAPSHOTS.pop(key, None)
            retained_ledger = retired[1]() if retired is not None else None
            ledger = retained_ledger or ModelPerformanceLedger(
                key, baseline=retired[0] if retired is not None else None
            )
            _MODEL_LEDGER_REGISTRY[key] = ledger
            _retire_oldest_model_ledger_locked()
        else:
            _MODEL_LEDGER_REGISTRY.move_to_end(key)
        return ledger


def get_model_performance_snapshots() -> list[ModelPerformanceSnapshot]:
    """Return deterministic snapshots for every model observed by this process."""
    with _MODEL_LEDGER_REGISTRY_LOCK:
        retired = [entry[0] for entry in _RETIRED_MODEL_SNAPSHOTS.values()]
        ledgers = [ledger for _, ledger in sorted(_MODEL_LEDGER_REGISTRY.items())]
    active = [ledger.snapshot() for ledger in ledgers]
    return [*sorted(retired, key=lambda item: item.model_name), *active]


def _reset_model_performance_registry_for_tests() -> None:
    with _MODEL_LEDGER_REGISTRY_LOCK:
        _MODEL_LEDGER_REGISTRY.clear()
        _RETIRED_MODEL_SNAPSHOTS.clear()
