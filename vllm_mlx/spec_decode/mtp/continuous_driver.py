# SPDX-License-Identifier: Apache-2.0
"""Scheduler-facing delivery driver for continuous self-MTP.

``ContinuousMTPGenerationBatch`` owns the transactional model and cache state,
but deliberately returns scheduler-neutral multi-lane bursts.  The scheduler
consumes one response object per delivered token.  This module is the small
adapter between those contracts:

* each :meth:`ContinuousMTPDriver.next` call emits at most one response per
  lane, draining variable-length burst queues before another model step;
* joins are staged and applied together immediately before the next burst, so
  membership never changes while a burst is being delivered; and
* terminal detach packages populate the last response for their lane with the
  canonical target/draft cache state used by scheduler finalization.

The default response is a pure-Python dataclass.  Production passes mlx-lm's
``GenerationBatch.Response`` (or an equivalent factory) without importing MLX
here.  Consequently this module remains unit-testable without a model, Metal,
or mlx-lm.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .continuous_batch import (
    ContinuousMTPDetachPackage,
    ContinuousMTPGenerationBatch,
    ContinuousMTPGenerationBatchError,
    ContinuousMTPGenerationBurst,
)
from .continuous_engine import (
    ContinuousSelfMTPRuntime,
    ContinuousSelfMTPUnsupportedError,
    SelfMTPLaneSpec,
)


class ContinuousMTPDriverError(ContinuousMTPGenerationBatchError):
    """A scheduler-delivery invariant failed in the continuous driver."""


@dataclass
class ContinuousMTPDriverResponse:
    """Pure-Python twin of mlx-lm ``GenerationBatch.Response``.

    The scheduler mutates ``finish_reason`` when a text stop or an abort wins,
    so this response is intentionally not frozen.
    """

    uid: int
    token: int
    logprobs: Any
    finish_reason: str | None
    prompt_cache: Any = None
    all_tokens: list[int] | None = None
    from_draft: bool = False
    mtp_state: tuple[Any, Any] | None = None


@dataclass(frozen=True)
class _PendingJoin:
    specs: tuple[SelfMTPLaneSpec, ...]
    stop_tokens: Mapping[int, tuple[int, ...]]


@dataclass(frozen=True)
class _QueuedToken:
    uid: int
    token: int
    logprobs: Any
    from_draft: bool
    finish_reason: str | None


class ContinuousMTPDriver:
    """Drive one continuous generation batch through scheduler responses.

    ``response_factory`` must accept the keyword fields of
    :class:`ContinuousMTPDriverResponse`.  mlx-lm's response class has the same
    core fields.  The factory is invoked only after the wrapper has committed a
    burst, so production factories should be simple, non-raising constructors.
    """

    def __init__(
        self,
        batch: ContinuousMTPGenerationBatch,
        *,
        response_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(batch, ContinuousMTPGenerationBatch):
            raise TypeError("batch must be a ContinuousMTPGenerationBatch")
        if response_factory is not None and not callable(response_factory):
            raise TypeError("response_factory must be callable")
        self._batch = batch
        self._response_factory = response_factory or ContinuousMTPDriverResponse
        self._pending_joins: list[_PendingJoin] = []
        self._delivery_order: tuple[int, ...] = ()
        self._delivery_queues: dict[int, deque[_QueuedToken]] = {}
        self._held_detaches: dict[int, ContinuousMTPDetachPackage] = {}
        self._terminal_detaches: list[ContinuousMTPDetachPackage] = []
        self._resumable_detaches: list[ContinuousMTPDetachPackage] = []
        self._recorded_package_ids: set[int] = set()
        self._last_burst: ContinuousMTPGenerationBurst | None = None
        self._last_attached_uids: tuple[int, ...] = ()

    @classmethod
    def create(
        cls,
        specs: Sequence[SelfMTPLaneSpec],
        runtime: ContinuousSelfMTPRuntime,
        *,
        stop_tokens: Mapping[int, Iterable[int]] | None = None,
        response_factory: Callable[..., Any] | None = None,
    ) -> ContinuousMTPDriver:
        """Prepare an initial cohort and return its delivery driver."""
        return cls(
            ContinuousMTPGenerationBatch.create(
                specs,
                runtime,
                stop_tokens=stop_tokens,
            ),
            response_factory=response_factory,
        )

    @classmethod
    def resume(
        cls,
        packages: Sequence[ContinuousMTPDetachPackage],
        *,
        response_factory: Callable[..., Any] | None = None,
    ) -> ContinuousMTPDriver:
        """Resume nonterminal detach packages without prefill or replay."""
        return cls(
            ContinuousMTPGenerationBatch.resume(packages),
            response_factory=response_factory,
        )

    @property
    def batch(self) -> ContinuousMTPGenerationBatch:
        """The owned wrapper, exposed for scheduler lifecycle inspection."""
        return self._batch

    @property
    def lane_uids(self) -> tuple[int, ...]:
        if self.closed:
            return ()
        return self._batch.lane_uids

    @property
    def pending_join_uids(self) -> tuple[int, ...]:
        return tuple(spec.uid for join in self._pending_joins for spec in join.specs)

    @property
    def dynamic_membership(self) -> bool:
        return self._batch.dynamic_membership

    @property
    def closed(self) -> bool:
        return self._batch.closed

    @property
    def has_pending_responses(self) -> bool:
        return any(self._delivery_queues.values())

    @property
    def has_work(self) -> bool:
        """Whether another :meth:`next` call can produce or advance work."""
        return bool(
            self.has_pending_responses or self._pending_joins or not self.closed
        )

    @property
    def last_burst(self) -> ContinuousMTPGenerationBurst | None:
        return self._last_burst

    @property
    def last_attached_uids(self) -> tuple[int, ...]:
        """UIDs attached at the start of the most recent :meth:`next` call."""
        return self._last_attached_uids

    def queue_lanes(
        self,
        specs: Sequence[SelfMTPLaneSpec],
        *,
        stop_tokens: Mapping[int, Iterable[int]] | None = None,
    ) -> tuple[int, ...]:
        """Stage lanes for atomic attachment at the next cycle boundary.

        This does not prepare caches or mutate the live cohort.  All joins
        queued between two calls to :meth:`next` are combined into one wrapper
        ``attach_lanes`` call, avoiding a partially applied multi-queue merge.
        """
        if self.closed:
            raise ContinuousMTPDriverError(
                "cannot queue lanes on a detached continuous MTP driver"
            )
        if not self.dynamic_membership:
            raise ContinuousSelfMTPUnsupportedError(
                "continuous MTP driver cannot queue incremental lanes without "
                "dynamic-membership attestation"
            )
        specs = tuple(specs)
        if not specs:
            return ()
        uids = tuple(spec.uid for spec in specs)
        if len(uids) != len(set(uids)):
            raise ValueError("queued continuous MTP lane uid values must be unique")
        occupied = (
            set(self.lane_uids)
            .union(self.pending_join_uids)
            .union(self._delivery_queues)
            .union(self._held_detaches)
        )
        overlap = occupied.intersection(uids)
        if overlap:
            raise ValueError(
                f"queued lanes reuse occupied uid values: {sorted(overlap)}"
            )

        raw_stops = {} if stop_tokens is None else dict(stop_tokens)
        unknown = set(raw_stops).difference(uids)
        if unknown:
            raise ValueError(
                "stop_tokens contains unknown queued lane uid values: "
                f"{sorted(unknown)}"
            )
        frozen_stops = {uid: tuple(raw_stops.get(uid, ())) for uid in uids}
        self._pending_joins.append(_PendingJoin(specs, frozen_stops))
        return uids

    def next(self) -> list[Any]:
        """Drain one response per uid, advancing only when delivery is empty.

        At most one response per uid is returned.  If the prior burst still
        has accepted tokens queued, this call drains those queues without
        advancing the model.  Only an empty delivery queue permits staged
        joins and the next transactional burst.  A terminal reason is attached
        only to the final delivered token for that lane.
        """
        if not self.has_pending_responses:
            if self.closed:
                self._last_attached_uids = ()
                return []
            self._last_attached_uids = self._apply_pending_joins()
            burst = self._batch.next_burst()
            self._last_burst = burst
            self._queue_burst(burst)
        else:
            self._last_attached_uids = ()
        return self._drain_one_per_uid()

    def _queue_burst(self, burst: ContinuousMTPGenerationBurst) -> None:
        if self.has_pending_responses:
            raise ContinuousMTPDriverError(
                "cannot queue a continuous MTP burst before delivery drains"
            )
        self._delivery_order = tuple(emission.uid for emission in burst.emissions)
        detached_by_uid = {package.uid: package for package in burst.detached}
        for emission in burst.emissions:
            if not emission.tokens:
                raise ContinuousMTPDriverError(
                    f"continuous MTP lane {emission.uid} emitted an empty burst"
                )
            queue = self._delivery_queues.setdefault(emission.uid, deque())
            for index, token in enumerate(emission.tokens):
                last = index == len(emission.tokens) - 1
                queue.append(
                    _QueuedToken(
                        uid=emission.uid,
                        token=token.token,
                        logprobs=token.logprobs,
                        from_draft=token.from_draft,
                        finish_reason=emission.finish_reason
                        if last and emission.terminal
                        else None,
                    )
                )
            package = detached_by_uid.get(emission.uid)
            if package is not None:
                self._held_detaches[emission.uid] = package

    def _drain_one_per_uid(self) -> list[Any]:
        responses: list[Any] = []
        for uid in self._delivery_order:
            queue = self._delivery_queues.get(uid)
            if not queue:
                continue
            item = queue.popleft()
            final_for_lane = not queue
            package = self._held_detaches.get(uid) if final_for_lane else None
            finishing_package = (
                package if package is not None and package.terminal else None
            )
            if item.finish_reason is not None and finishing_package is None:
                raise ContinuousMTPDriverError(
                    f"terminal lane {uid} has no retained detach package"
                )
            responses.append(
                self._response_factory(
                    uid=uid,
                    token=item.token,
                    logprobs=item.logprobs,
                    finish_reason=(
                        item.finish_reason if finishing_package is not None else None
                    ),
                    prompt_cache=(
                        finishing_package.target_cache
                        if finishing_package is not None
                        else None
                    ),
                    all_tokens=(
                        list(finishing_package.token_ids)
                        if finishing_package is not None
                        else None
                    ),
                    from_draft=item.from_draft,
                    mtp_state=(
                        (
                            finishing_package.draft_cache,
                            finishing_package.lane.seed_hidden,
                        )
                        if finishing_package is not None
                        else None
                    ),
                )
            )
            if final_for_lane:
                self._delivery_queues.pop(uid, None)
                if package is not None:
                    self._held_detaches.pop(uid, None)
                    self._record_detaches((package,))
        if not self.has_pending_responses:
            self._delivery_order = ()
        return responses

    def detach_all(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Detach the live cohort for cancellation, shutdown, or turnover.

        This cache-preserving operation requires prior delivery to drain.  A
        shutdown that intentionally discards pending responses uses
        :meth:`discard_all` instead.  Queued-but-unattached lanes allocate no
        state and are discarded.  Returned packages remain available through
        the corresponding ``take_*_detaches`` method.
        """
        if self.has_pending_responses:
            raise ContinuousMTPDriverError(
                "cannot detach resumable MTP state before delivery drains"
            )
        self._pending_joins.clear()
        self._last_attached_uids = ()
        packages = self._batch.detach_all()
        self._record_detaches(packages)
        return packages

    def discard_all(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Detach for shutdown while explicitly discarding queued delivery.

        The returned packages reflect cache state after the fully committed
        burst, which may be ahead of scheduler-visible delivery.  They are
        therefore cleanup handles only and are deliberately not published by
        ``take_resumable_detaches`` or accepted for automatic turnover.
        """
        self._pending_joins.clear()
        self._last_attached_uids = ()
        held = tuple(self._held_detaches.values())
        self._held_detaches.clear()
        self._delivery_queues.clear()
        self._delivery_order = ()
        detached = self._batch.detach_all()
        unique = {id(package): package for package in (*held, *detached)}
        return tuple(unique.values())

    def remove_uids(
        self, uids: Sequence[int]
    ) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Cancel uids without leaving them in live or queued driver state.

        Dynamic membership extracts only requested live rows.  A fixed cohort
        turns over as a unit; companion packages stay held until their already
        committed response queues drain, after which the scheduler can resume
        them with :meth:`resume_turnover` or :meth:`resume`.
        """
        uids = tuple(dict.fromkeys(uids))
        if not uids:
            return ()
        requested = set(uids)

        # A cancellation can race a lane staged for the next boundary.  Such a
        # lane owns no cache yet, so removing its pending specification is the
        # complete operation.
        retained_joins: list[_PendingJoin] = []
        for pending in self._pending_joins:
            specs = tuple(spec for spec in pending.specs if spec.uid not in requested)
            if specs:
                stops = {spec.uid: pending.stop_tokens[spec.uid] for spec in specs}
                retained_joins.append(_PendingJoin(specs, stops))
        self._pending_joins = retained_joins

        removed: list[ContinuousMTPDetachPackage] = []
        # Terminal lanes may already be detached transactionally while their
        # final response is still queued.  Cancellation discards that response
        # and returns the held cache package to the caller.
        for uid in requested:
            held = self._held_detaches.pop(uid, None)
            if held is not None:
                removed.append(held)
            self._delivery_queues.pop(uid, None)
        self._delivery_order = tuple(
            uid for uid in self._delivery_order if uid not in requested
        )

        live_requested = [uid for uid in self.lane_uids if uid in requested]
        if live_requested:
            detached = self._batch.detach_lanes(live_requested)
            for package in detached:
                if package.uid in requested:
                    removed.append(package)
                    self._delivery_queues.pop(package.uid, None)
                    self._held_detaches.pop(package.uid, None)
                elif self._delivery_queues.get(package.uid):
                    # Fixed-cohort companion: publish its resumable package only
                    # after every committed token for that lane is delivered.
                    self._held_detaches[package.uid] = package
                else:
                    self._record_detaches((package,))
            self._delivery_order = tuple(
                uid for uid in self._delivery_order if self._delivery_queues.get(uid)
            )

        if not self.has_pending_responses:
            self._delivery_order = ()
        return tuple(removed)

    def resume_turnover(self) -> tuple[int, ...]:
        """Resume held fixed-cohort companions after delivery fully drains."""
        if self.has_pending_responses:
            raise ContinuousMTPDriverError(
                "cannot resume continuous MTP turnover before delivery drains"
            )
        if not self.closed:
            raise ContinuousMTPDriverError(
                "continuous MTP turnover requires a detached cohort"
            )
        packages = tuple(self._resumable_detaches)
        if not packages:
            return ()
        self._resumable_detaches.clear()
        self._forget_recorded(packages)
        self._batch = ContinuousMTPGenerationBatch.resume(packages)
        return self.lane_uids

    def _forget_recorded(self, packages: Sequence[ContinuousMTPDetachPackage]) -> None:
        # Drop drained packages from the dedup set so it stays bounded and a
        # future package cannot be skipped because CPython reused a freed id.
        for package in packages:
            self._recorded_package_ids.discard(id(package))

    def take_terminal_detaches(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Drain terminal packages accumulated by completed bursts."""
        packages = tuple(self._terminal_detaches)
        self._terminal_detaches.clear()
        self._forget_recorded(packages)
        return packages

    def take_resumable_detaches(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Drain nonterminal packages retained for scheduler turnover."""
        packages = tuple(self._resumable_detaches)
        self._resumable_detaches.clear()
        self._forget_recorded(packages)
        return packages

    def _apply_pending_joins(self) -> tuple[int, ...]:
        if not self._pending_joins:
            return ()
        specs = tuple(spec for pending in self._pending_joins for spec in pending.specs)
        stops = {
            uid: values
            for pending in self._pending_joins
            for uid, values in pending.stop_tokens.items()
        }
        attached = self._batch.attach_lanes(specs, stop_tokens=stops)
        self._pending_joins.clear()
        return attached

    def _record_detaches(self, packages: Sequence[ContinuousMTPDetachPackage]) -> None:
        for package in packages:
            identity = id(package)
            if identity in self._recorded_package_ids:
                continue
            self._recorded_package_ids.add(identity)
            if package.terminal:
                self._terminal_detaches.append(package)
            else:
                self._resumable_detaches.append(package)


__all__ = [
    "ContinuousMTPDriver",
    "ContinuousMTPDriverError",
    "ContinuousMTPDriverResponse",
]
