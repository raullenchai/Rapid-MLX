# SPDX-License-Identifier: Apache-2.0
"""Scheduler-neutral generation wrapper for continuous self-MTP.

The transaction engine in :mod:`continuous_engine` deliberately knows
nothing about request delivery.  This module adds that missing, pure-Python
boundary: it delivers the prepared first token, limits one proposal to each
lane's remaining token budget and stop-token set, and commits exactly the
prefix that the caller receives.

Membership policy follows the runtime.  When dynamic membership is **not**
attested the wrapper is fixed-cohort: a lane reaching a terminal condition
tears down the *whole* cohort after the open proposal is committed, terminal
lanes are marked for finalization, and companion lanes are returned as
resumable detach packages.  ``attach_lanes`` refuses incremental joins.

When the runtime attests dynamic membership (``allow_dynamic_membership`` plus
the ``dynamic_membership`` capability, and the Flash-specific attestation for a
Flash architecture) the wrapper is a living batch: ``attach_lanes`` merges new
lanes at a closed-transaction boundary and a terminal lane detaches by itself
while its companions keep decoding.  A freshly attached lane delivers its
prepared first token on its next serviced burst before it joins a proposal,
mirroring the proven source contract.  Every membership change still happens
only between transactions, and the underlying engine fails closed if the
attestation is missing, so an over-eager driver cannot force an unsafe merge.

No MLX type is imported here.  Model execution, cache merge/rollback/extract,
and the target/MTP forward calls remain injected through ``continuous_engine``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .continuous_engine import (
    BatchedSelfMTPState,
    ContinuousSelfMTPError,
    ContinuousSelfMTPRuntime,
    ContinuousSelfMTPUnsupportedError,
    DetachedSelfMTPLane,
    MTPToken,
    SelfMTPLaneSpec,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
    supports_dynamic_membership,
)


class ContinuousMTPGenerationBatchError(ContinuousSelfMTPError):
    """A delivery or lifecycle invariant failed in the generation wrapper."""


@dataclass(frozen=True)
class ContinuousMTPLaneState:
    """Immutable scheduler-facing snapshot of one lane's delivery state."""

    uid: int
    emitted_tokens: int
    max_tokens: int
    remaining_tokens: int
    stop_tokens: frozenset[int]
    terminal: bool
    finish_reason: str | None


@dataclass(frozen=True)
class ContinuousMTPLaneEmission:
    """The exact token prefix delivered for one lane in one burst."""

    uid: int
    tokens: tuple[MTPToken, ...]
    terminal: bool = False
    finish_reason: str | None = None

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token for token in self.tokens)

    @property
    def logprobs(self) -> tuple[Any, ...]:
        return tuple(token.logprobs for token in self.tokens)


@dataclass(frozen=True)
class ContinuousMTPDetachPackage:
    """A detached lane plus its exact delivered-token ledger.

    ``terminal`` distinguishes a request the scheduler should finalize from a
    companion that was detached only because fixed-cohort turnover was
    required.  Companion packages retain their canonical lane and cache pair
    and are therefore resumable by a later integration.  Under dynamic
    membership a terminal lane detaches on its own, so every package a per-lane
    detach yields is terminal and the surviving companions are never detached.
    """

    detached: DetachedSelfMTPLane
    tokens: tuple[MTPToken, ...]
    stop_tokens: frozenset[int]
    terminal: bool
    finish_reason: str | None

    @property
    def uid(self) -> int:
        return self.detached.lane.uid

    @property
    def lane(self):
        return self.detached.lane

    @property
    def caches(self):
        return self.detached.caches

    @property
    def target_cache(self):
        return self.detached.caches.target

    @property
    def draft_cache(self):
        return self.detached.caches.draft

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token for token in self.tokens)

    @property
    def logprobs(self) -> tuple[Any, ...]:
        return tuple(token.logprobs for token in self.tokens)


@dataclass(frozen=True)
class ContinuousMTPGenerationBurst:
    """One delivery event: initial tokens or one proposal transaction."""

    emissions: tuple[ContinuousMTPLaneEmission, ...]
    emitted_counts: tuple[int, ...]
    initial: bool
    detached: tuple[ContinuousMTPDetachPackage, ...] = ()

    @property
    def cohort_detached(self) -> bool:
        return bool(self.detached)

    @property
    def terminal_detaches(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        return tuple(package for package in self.detached if package.terminal)

    @property
    def resumable_detaches(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        return tuple(package for package in self.detached if not package.terminal)


@dataclass
class _LaneDeliveryState:
    uid: int
    max_tokens: int
    stop_tokens: frozenset[int]
    tokens: list[MTPToken] = field(default_factory=list)
    finish_reason: str | None = None
    # A prepared first token that has not been delivered yet.  Set on cohort
    # creation and on every dynamic join; cleared once the initial burst emits
    # it.  A lane with a pending initial must never enter a proposal.
    pending_initial: MTPToken | None = None

    @property
    def terminal(self) -> bool:
        return self.finish_reason is not None


class ContinuousMTPGenerationBatch:
    """Own one cohort from preparation through extraction.

    Fixed-cohort by default; a living batch when the runtime attests dynamic
    membership.
    """

    def __init__(
        self,
        *,
        batch: BatchedSelfMTPState,
        first_tokens: Sequence[MTPToken],
        stop_tokens: Mapping[int, frozenset[int]],
    ) -> None:
        if len(first_tokens) != len(batch.lanes):
            raise ValueError("first_tokens must have one entry per lane")
        self._batch = batch
        self._closed = False
        self._detached: tuple[ContinuousMTPDetachPackage, ...] = ()
        self._states = {
            lane.uid: _LaneDeliveryState(
                uid=lane.uid,
                max_tokens=lane.max_tokens,
                stop_tokens=stop_tokens[lane.uid],
                pending_initial=first,
            )
            for lane, first in zip(batch.lanes, first_tokens)
        }

    @classmethod
    def create(
        cls,
        specs: Sequence[SelfMTPLaneSpec],
        runtime: ContinuousSelfMTPRuntime,
        *,
        stop_tokens: Mapping[int, Iterable[int]] | None = None,
    ) -> ContinuousMTPGenerationBatch:
        """Prepare and attach one initial cohort.

        ``stop_tokens`` is keyed by lane uid.  Unknown keys are rejected so a
        scheduler typo cannot silently disable a request's stop condition.
        """
        specs = tuple(specs)
        if not specs:
            raise ValueError("cannot create an empty continuous MTP cohort")
        uids = tuple(spec.uid for spec in specs)
        if len(uids) != len(set(uids)):
            raise ValueError("continuous MTP lane uid values must be unique")
        normalized_stops = _normalize_stop_tokens(uids, stop_tokens)

        prepared: list[DetachedSelfMTPLane] = []
        first_tokens: list[MTPToken] = []
        for spec in specs:
            detached, first = prepare_self_mtp_lane(spec, runtime)
            prepared.append(detached)
            first_tokens.append(first)
        batch = attach_self_mtp_lanes(None, prepared, runtime=runtime)
        return cls(
            batch=batch,
            first_tokens=first_tokens,
            stop_tokens=normalized_stops,
        )

    @classmethod
    def resume(
        cls,
        packages: Sequence[ContinuousMTPDetachPackage],
    ) -> ContinuousMTPGenerationBatch:
        """Resume a nonterminal cohort from canonical detach packages.

        The packages already own prepared target/draft caches and their exact
        delivered-token ledgers, so resumption attaches them directly without
        re-prefill or replay.  Terminal packages are rejected: their cache
        ownership belongs to scheduler finalization, not a new live cohort.
        """
        packages = tuple(packages)
        if not packages:
            raise ValueError("cannot resume an empty continuous MTP cohort")
        if any(package.terminal for package in packages):
            raise ValueError("cannot resume terminal continuous MTP packages")
        uids = tuple(package.uid for package in packages)
        if len(uids) != len(set(uids)):
            raise ValueError("resumed continuous MTP lane uid values must be unique")
        runtime = packages[0].detached._runtime
        if any(package.detached._runtime is not runtime for package in packages[1:]):
            raise ValueError("resumed continuous MTP packages use different runtimes")

        batch = attach_self_mtp_lanes(
            None,
            [package.detached for package in packages],
            runtime=runtime,
        )
        resumed = cls.__new__(cls)
        resumed._batch = batch
        resumed._closed = False
        resumed._detached = ()
        resumed._states = {
            package.uid: _LaneDeliveryState(
                uid=package.uid,
                max_tokens=package.lane.max_tokens,
                stop_tokens=package.stop_tokens,
                tokens=list(package.tokens),
            )
            for package in packages
        }
        return resumed

    @property
    def dynamic_membership(self) -> bool:
        """Whether this cohort's runtime attests incremental membership."""
        return supports_dynamic_membership(self._batch._runtime)

    @property
    def lane_uids(self) -> tuple[int, ...]:
        return tuple(self._states)

    @property
    def initial_pending(self) -> bool:
        return any(state.pending_initial is not None for state in self._states.values())

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def lane_states(self) -> tuple[ContinuousMTPLaneState, ...]:
        snapshots = []
        for state in self._states.values():
            emitted = len(state.tokens)
            snapshots.append(
                ContinuousMTPLaneState(
                    uid=state.uid,
                    emitted_tokens=emitted,
                    max_tokens=state.max_tokens,
                    remaining_tokens=max(state.max_tokens - emitted, 0),
                    stop_tokens=state.stop_tokens,
                    terminal=state.terminal,
                    finish_reason=state.finish_reason,
                )
            )
        return tuple(snapshots)

    def attach_lanes(
        self,
        specs: Sequence[SelfMTPLaneSpec],
        *,
        stop_tokens: Mapping[int, Iterable[int]] | None = None,
    ) -> tuple[int, ...]:
        """Merge new lanes into the living batch at a transaction boundary.

        Refused unless the runtime attests dynamic membership (the engine's
        ``attach_self_mtp_lanes`` fails closed on the same rule, so this can
        never merge an unsafe cohort).  Each joined lane is prepared here so it
        arrives with its own canonical caches and first token; that first token
        is delivered on the next serviced burst before the lane joins any
        proposal.  Returns the uids attached, in row order.
        """
        specs = tuple(specs)
        self._require_open()
        if not self.dynamic_membership:
            raise ContinuousSelfMTPUnsupportedError(
                "ContinuousMTPGenerationBatch is fixed-cohort; incremental join "
                "is unsupported (including Flash) without dynamic-membership "
                "attestation"
            )
        if not specs:
            return ()
        joining_uids = tuple(spec.uid for spec in specs)
        if len(joining_uids) != len(set(joining_uids)):
            raise ValueError("joining lane uid values must be unique")
        overlap = set(joining_uids).intersection(self._states)
        if overlap:
            raise ValueError(f"joining lanes reuse live uid values: {sorted(overlap)}")
        normalized_stops = _normalize_stop_tokens(joining_uids, stop_tokens)

        runtime = self._batch._runtime
        prepared: list[DetachedSelfMTPLane] = []
        first_tokens: list[MTPToken] = []
        for spec in specs:
            detached, first = prepare_self_mtp_lane(spec, runtime)
            prepared.append(detached)
            first_tokens.append(first)
        self._batch = attach_self_mtp_lanes(self._batch, prepared, runtime=runtime)
        for spec, first in zip(specs, first_tokens):
            self._states[spec.uid] = _LaneDeliveryState(
                uid=spec.uid,
                max_tokens=spec.max_tokens,
                stop_tokens=normalized_stops[spec.uid],
                pending_initial=first,
            )
        return joining_uids

    def next_burst(self) -> ContinuousMTPGenerationBurst:
        """Deliver pending initial tokens, or run and commit one proposal."""
        self._require_open()
        pending = [
            uid
            for uid in self.lane_uids
            if self._states[uid].pending_initial is not None
        ]
        if pending:
            return self._deliver_initial(pending)

        proposal = propose_batched_self_mtp(self._batch)
        planned: list[tuple[int, tuple[MTPToken, ...], str | None]] = []
        emitted_counts: list[int] = []
        terminal: list[bool] = []
        for uid, outputs in zip(proposal.lane_uids, proposal.outputs):
            state = self._states[uid]
            delivered, finish_reason = _bounded_prefix(state, outputs)
            planned.append((uid, delivered, finish_reason))
            emitted_counts.append(len(delivered))
            terminal.append(finish_reason is not None)

        commit_batched_self_mtp(
            self._batch,
            proposal,
            emitted_counts=emitted_counts,
            terminal=terminal,
        )
        emissions: list[ContinuousMTPLaneEmission] = []
        for uid, delivered, finish_reason in planned:
            state = self._states[uid]
            state.tokens.extend(delivered)
            state.finish_reason = finish_reason
            emissions.append(
                ContinuousMTPLaneEmission(
                    uid=uid,
                    tokens=delivered,
                    terminal=state.terminal,
                    finish_reason=state.finish_reason,
                )
            )
        detached = self._resolve_terminals(terminal)
        return ContinuousMTPGenerationBurst(
            emissions=tuple(emissions),
            emitted_counts=tuple(emitted_counts),
            initial=False,
            detached=detached,
        )

    def detach_all(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Extract every lane/cache pair without inventing finish reasons.

        This is idempotent after a successful teardown.  It is suitable for
        cancellation, shutdown, or scheduler turnover.  If a lane's prepared
        first token has not yet been delivered, its token ledger is
        intentionally empty even though the detached canonical lane retains its
        ``cur``.
        """
        if self._closed:
            return self._detached
        return self._detach_cohort()

    def detach_lanes(
        self, uids: Sequence[int]
    ) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Detach live nonterminal lanes at a closed transaction boundary.

        Dynamic cohorts detach only the requested rows.  Fixed cohorts must
        turn over as a unit, so requesting any live uid detaches every row and
        returns the companions as resumable packages too.
        """
        self._require_open()
        uids = tuple(uids)
        if not uids:
            return ()
        if len(uids) != len(set(uids)):
            raise ValueError("detached continuous MTP lane uid values must be unique")
        unknown = set(uids).difference(self._states)
        if unknown:
            raise KeyError(f"unknown continuous MTP lane uid values: {sorted(unknown)}")
        requested = set(uids)
        indices = [
            index
            for index, lane in enumerate(self._batch.lanes)
            if lane.uid in requested
        ]
        if self.dynamic_membership and len(indices) < len(self._batch.lanes):
            return self._detach_indices(indices, terminal=False)
        return self._detach_cohort()

    def _deliver_initial(self, pending: Sequence[int]) -> ContinuousMTPGenerationBurst:
        emissions: list[ContinuousMTPLaneEmission] = []
        terminal_by_row = [False] * len(self._batch.lanes)
        row_of = {lane.uid: index for index, lane in enumerate(self._batch.lanes)}
        for uid in pending:
            state = self._states[uid]
            token = state.pending_initial
            if token is None:
                raise ContinuousMTPGenerationBatchError(
                    f"lane {uid} has no pending initial token"
                )
            state.pending_initial = None
            state.tokens.append(token)
            if token.token in state.stop_tokens:
                state.finish_reason = "stop"
            elif len(state.tokens) >= state.max_tokens:
                state.finish_reason = "length"
            emissions.append(
                ContinuousMTPLaneEmission(
                    uid=uid,
                    tokens=(token,),
                    terminal=state.terminal,
                    finish_reason=state.finish_reason,
                )
            )
            if state.terminal:
                terminal_by_row[row_of[uid]] = True
        detached = self._resolve_terminals(terminal_by_row)
        return ContinuousMTPGenerationBurst(
            emissions=tuple(emissions),
            emitted_counts=tuple(1 for _ in emissions),
            initial=True,
            detached=detached,
        )

    def _resolve_terminals(
        self, terminal: Sequence[bool]
    ) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Detach terminal lanes after a committed burst.

        Dynamic membership detaches only the terminal rows and keeps the batch
        alive for its companions; the fixed-cohort milestone (and the case
        where every lane is terminal at once) tears the whole cohort down so
        the survivors return as resumable packages.
        """
        indices = [row for row, is_terminal in enumerate(terminal) if is_terminal]
        if not indices:
            return ()
        if self.dynamic_membership and len(indices) < len(self._batch.lanes):
            return self._detach_indices(indices, terminal=True)
        return self._detach_cohort()

    def _detach_indices(
        self, indices: Sequence[int], *, terminal: bool
    ) -> tuple[ContinuousMTPDetachPackage, ...]:
        """Detach the given rows; the batch stays open for the rest."""
        self._batch, detached = detach_self_mtp_lanes(self._batch, indices)
        packages = []
        for item in detached:
            state = self._states.pop(item.lane.uid)
            packages.append(
                ContinuousMTPDetachPackage(
                    detached=item,
                    tokens=tuple(state.tokens),
                    stop_tokens=state.stop_tokens,
                    terminal=terminal,
                    finish_reason=state.finish_reason,
                )
            )
        return tuple(packages)

    def _detach_cohort(self) -> tuple[ContinuousMTPDetachPackage, ...]:
        if self._closed:
            return self._detached
        indices = tuple(range(len(self._batch.lanes)))
        self._batch, detached = detach_self_mtp_lanes(self._batch, indices)
        packages = []
        for item in detached:
            state = self._states[item.lane.uid]
            packages.append(
                ContinuousMTPDetachPackage(
                    detached=item,
                    tokens=tuple(state.tokens),
                    stop_tokens=state.stop_tokens,
                    terminal=state.terminal,
                    finish_reason=state.finish_reason,
                )
            )
        self._detached = tuple(packages)
        self._closed = True
        return self._detached

    def _require_open(self) -> None:
        if self._closed:
            raise ContinuousMTPGenerationBatchError(
                "continuous MTP generation cohort is already detached"
            )


def _normalize_stop_tokens(
    uids: Sequence[int],
    stop_tokens: Mapping[int, Iterable[int]] | None,
) -> dict[int, frozenset[int]]:
    raw = {} if stop_tokens is None else dict(stop_tokens)
    unknown = set(raw).difference(uids)
    if unknown:
        raise ValueError(
            f"stop_tokens contains unknown lane uid values: {sorted(unknown)}"
        )
    normalized: dict[int, frozenset[int]] = {}
    for uid in uids:
        values = frozenset(raw.get(uid, ()))
        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in values
        ):
            raise ValueError("stop token ids must be integers")
        normalized[uid] = values
    return normalized


def _bounded_prefix(
    state: _LaneDeliveryState,
    outputs: Sequence[MTPToken],
) -> tuple[tuple[MTPToken, ...], str | None]:
    remaining = state.max_tokens - len(state.tokens)
    if remaining <= 0:
        raise ContinuousMTPGenerationBatchError(
            f"lane {state.uid} was proposed after exhausting max_tokens"
        )
    bounded = tuple(outputs[:remaining])
    for index, token in enumerate(bounded):
        if token.token in state.stop_tokens:
            return bounded[: index + 1], "stop"
    if len(bounded) < len(outputs) or len(bounded) == remaining:
        return bounded, "length"
    return bounded, None


__all__ = [
    "ContinuousMTPDetachPackage",
    "ContinuousMTPGenerationBatch",
    "ContinuousMTPGenerationBatchError",
    "ContinuousMTPGenerationBurst",
    "ContinuousMTPLaneEmission",
    "ContinuousMTPLaneState",
]
