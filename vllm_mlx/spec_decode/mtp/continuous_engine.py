# SPDX-License-Identifier: Apache-2.0
"""Fixed-membership continuous self-MTP transaction core.

This is a scheduler-neutral port of the persistent batched self-MTP lifecycle
introduced by immutable source commit ``92576ced``.  It deliberately imports no
MLX modules and makes no model or cache-layout assumptions.  Model execution and
ragged cache operations cross explicit injected protocols, allowing Rapid's
``return_hidden=True`` / ``n_confirmed=...`` ABI and its version-gated ragged
adapter to evolve independently.

The first milestone is fixed membership: an initial group may attach, transact,
and detach as a whole.  Incremental attach or partial detach is refused unless
both policy and runtime capabilities attest dynamic membership.  Flash-family
models require an additional architecture-specific attestation.  The feature is
default-off and XTC is unconditionally refused until an exact verifier exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class ContinuousSelfMTPError(RuntimeError):
    """Base error for a fail-closed engine invariant."""


class ContinuousSelfMTPUnsupportedError(ContinuousSelfMTPError):
    """The requested policy lacks an explicitly attested runtime capability."""


@dataclass(frozen=True)
class ContinuousSelfMTPConfig:
    enabled: bool = False
    allow_dynamic_membership: bool = False
    architecture: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(
            self.allow_dynamic_membership, bool
        ):
            raise ValueError("continuous self-MTP policy flags must be booleans")
        if not isinstance(self.architecture, str) or not self.architecture:
            raise ValueError("architecture must be a non-empty string")


@dataclass(frozen=True)
class ContinuousSelfMTPCapabilities:
    """Runtime facts.  Every field defaults to refusal."""

    target_return_hidden: bool = False
    mtp_return_hidden: bool = False
    confirmed_target_forward: bool = False
    ragged_rollback: bool = False
    atomic_cache_commit: bool = False
    per_lane_rng: bool = False
    transformed_sampling: bool = False
    logits_processors_exact: bool = False
    dynamic_membership: bool = False
    flash_dynamic_membership_attested: bool = False

    def missing_fixed_core(self) -> tuple[str, ...]:
        required = (
            "target_return_hidden",
            "mtp_return_hidden",
            "confirmed_target_forward",
            "ragged_rollback",
            "atomic_cache_commit",
        )
        return tuple(name for name in required if getattr(self, name) is not True)


@dataclass(frozen=True)
class RapidForwardSeams:
    """Adapters for Rapid's current target and MTP call signatures.

    ``target`` always requests hidden states and explicitly supplies
    ``n_confirmed``. ``draft`` requests the post-MTP hidden state.  Backends use
    these helpers instead of assuming mlx-lm-unified's ``mtp_step`` API.
    """

    model_forward: Callable[..., Any]
    mtp_forward: Callable[..., Any]

    def __post_init__(self) -> None:
        if not callable(self.model_forward) or not callable(self.mtp_forward):
            raise TypeError("Rapid forward seams must be callable")

    def target(
        self,
        inputs: Any,
        cache: Any,
        *,
        n_confirmed: int = 0,
    ) -> Any:
        if isinstance(n_confirmed, bool) or not isinstance(n_confirmed, int):
            raise ValueError("n_confirmed must be an integer")
        if n_confirmed < 0:
            raise ValueError("n_confirmed cannot be negative")
        return self.model_forward(
            inputs,
            cache=cache,
            return_hidden=True,
            n_confirmed=n_confirmed,
        )

    def draft(self, hidden: Any, token_ids: Any, cache: Any) -> Any:
        return self.mtp_forward(
            hidden,
            token_ids,
            cache,
            return_hidden=True,
        )


@dataclass(frozen=True)
class SelfMTPSampling:
    temperature: float = 0.0
    has_logits_processors: bool = False
    uses_xtc: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise ValueError("temperature must be numeric")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not isinstance(self.has_logits_processors, bool) or not isinstance(
            self.uses_xtc, bool
        ):
            raise ValueError("sampling feature flags must be booleans")


@dataclass(frozen=True)
class SelfMTPLaneSpec:
    uid: int
    prompt: Any
    max_tokens: int
    num_draft: int = 1
    sampling: SelfMTPSampling = SelfMTPSampling()
    prompt_cache: Any = None
    mtp_cache: Any = None

    def __post_init__(self) -> None:
        if isinstance(self.uid, bool) or not isinstance(self.uid, int):
            raise ValueError("uid must be an integer")
        for name in ("max_tokens", "num_draft"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MTPToken:
    token: int
    logprobs: Any
    from_draft: bool

    def __post_init__(self) -> None:
        if isinstance(self.token, bool) or not isinstance(self.token, int):
            raise ValueError("token must be an integer")
        if not isinstance(self.from_draft, bool):
            raise ValueError("from_draft must be a boolean")


@dataclass
class SelfMTPCachePair:
    target: Any
    draft: Any


@dataclass
class SelfMTPLane:
    uid: int
    cur: int
    seed_hidden: Any
    token_prefix: Any
    ntoks: int
    max_tokens: int
    num_draft: int
    sampling: SelfMTPSampling
    pending_hidden: Any = None
    pending_tokens: list[int] = field(default_factory=list)
    backend_state: Any = None


@dataclass
class DetachedSelfMTPLane:
    lane: SelfMTPLane
    caches: SelfMTPCachePair
    _runtime: ContinuousSelfMTPRuntime = field(repr=False, compare=False)


@dataclass(frozen=True)
class PreparedLaneData:
    """Data-plane result returned by ``SelfMTPComputeBackend.prepare``."""

    cur: int
    seed_hidden: Any
    token_prefix: Any
    caches: SelfMTPCachePair
    first_token: MTPToken
    backend_state: Any = None


@dataclass(frozen=True)
class CycleComputation:
    """Uncommitted result returned by ``SelfMTPComputeBackend.propose``."""

    lane_uids: tuple[int, ...]
    draft_depths: tuple[int, ...]
    accepted_lengths: tuple[int, ...]
    target_drops: tuple[int, ...]
    draft_drops: tuple[int, ...]
    outputs: tuple[tuple[MTPToken, ...], ...]
    payload: Any = None


@dataclass(frozen=True)
class SelfMTPCycleResult:
    membership_epoch: int
    lane_uids: tuple[int, ...]
    draft_depths: tuple[int, ...]
    accepted_lengths: tuple[int, ...]
    target_drops: tuple[int, ...]
    draft_drops: tuple[int, ...]
    outputs: tuple[tuple[MTPToken, ...], ...]
    _computation: CycleComputation = field(repr=False, compare=False)


class SelfMTPComputeBackend(Protocol):
    """Model-specific compute seam; implementations may import MLX."""

    def prepare(
        self, spec: SelfMTPLaneSpec, forwards: RapidForwardSeams
    ) -> PreparedLaneData: ...

    def propose(
        self,
        lanes: Sequence[SelfMTPLane],
        caches: SelfMTPCachePair,
        forwards: RapidForwardSeams,
    ) -> CycleComputation: ...

    def commit(
        self,
        lanes: Sequence[SelfMTPLane],
        computation: CycleComputation,
        *,
        emitted_counts: tuple[int, ...],
        terminal: tuple[bool, ...],
    ) -> None: ...

    def abort(
        self,
        lanes: Sequence[SelfMTPLane],
        caches: SelfMTPCachePair,
        computation: CycleComputation | None,
        cause: BaseException | None,
    ) -> None:
        """Restore lane and cache state to the pre-proposal boundary.

        This method is the backend half of the transaction contract.  It must
        be safe after a partially executed ``propose`` and after a failed
        ``commit``.  Returning certifies that both cache groups and every lane
        again match the boundary before the proposal began; raising poisons the
        batch so no retry, detach, or fallback can reuse inconsistent state.
        """
        ...

    def detach_lane(self, lane: SelfMTPLane, caches: SelfMTPCachePair) -> None: ...


class SelfMTPCacheAdapter(Protocol):
    """Injected merge/rollback/extract seam for Rapid's ragged adapter."""

    def attach(
        self,
        current: SelfMTPCachePair | None,
        joining: Sequence[SelfMTPCachePair],
    ) -> SelfMTPCachePair: ...

    def rollback(
        self,
        caches: SelfMTPCachePair,
        *,
        target_drops: Sequence[int],
        draft_drops: Sequence[int],
        verify_width: int,
    ) -> None: ...

    def detach(
        self,
        caches: SelfMTPCachePair,
        indices: Sequence[int],
        keep_indices: Sequence[int],
    ) -> tuple[SelfMTPCachePair, list[SelfMTPCachePair]]: ...


@dataclass(frozen=True)
class ContinuousSelfMTPRuntime:
    config: ContinuousSelfMTPConfig
    capabilities: ContinuousSelfMTPCapabilities
    forwards: RapidForwardSeams
    compute: SelfMTPComputeBackend
    caches: SelfMTPCacheAdapter


@dataclass
class BatchedSelfMTPState:
    lanes: list[SelfMTPLane]
    caches: SelfMTPCachePair
    membership_epoch: int
    _runtime: ContinuousSelfMTPRuntime = field(repr=False, compare=False)
    proposal_open: bool = False
    poisoned: bool = False
    poison_reason: str | None = None
    _open_proposal: SelfMTPCycleResult | None = field(
        default=None, repr=False, compare=False
    )


def _require_fixed_core(runtime: ContinuousSelfMTPRuntime) -> None:
    if runtime.config.enabled is not True:
        raise ContinuousSelfMTPUnsupportedError("continuous self-MTP is disabled")
    missing = runtime.capabilities.missing_fixed_core()
    if missing:
        raise ContinuousSelfMTPUnsupportedError(
            "missing fixed-membership capability: " + ", ".join(missing)
        )


def _require_healthy(batch: BatchedSelfMTPState) -> None:
    if batch.poisoned:
        reason = batch.poison_reason or "unknown rollback failure"
        raise ContinuousSelfMTPError(f"self-MTP batch is poisoned: {reason}")


def _abort_backend_transaction(
    batch: BatchedSelfMTPState,
    computation: CycleComputation | None,
    cause: BaseException | None,
) -> None:
    """Ask the backend to restore the exact pre-proposal boundary."""

    abort = getattr(batch._runtime.compute, "abort", None)
    if not callable(abort):
        batch.poisoned = True
        batch.poison_reason = "compute backend has no abort surface"
        raise ContinuousSelfMTPError(batch.poison_reason) from cause
    try:
        abort(batch.lanes, batch.caches, computation, cause)
    except BaseException as abort_error:  # noqa: BLE001 - poison on any failure
        batch.poisoned = True
        batch.poison_reason = f"backend abort failed: {abort_error}"
        raise ContinuousSelfMTPError(batch.poison_reason) from abort_error


def supports_dynamic_membership(runtime: ContinuousSelfMTPRuntime) -> bool:
    """Return whether incremental attach / partial detach is attested.

    This is the non-raising twin of ``_require_dynamic``: callers that must
    branch on membership policy (the generation wrapper choosing per-lane
    versus whole-cohort detach) read this, while call sites that perform a
    membership change keep calling ``_require_dynamic`` to fail closed with a
    specific reason.  Both share one gating rule so they can never disagree.
    """
    if runtime.config.allow_dynamic_membership is not True:
        return False
    if runtime.capabilities.dynamic_membership is not True:
        return False
    if (
        "flash" in runtime.config.architecture.lower()
        and runtime.capabilities.flash_dynamic_membership_attested is not True
    ):
        return False
    return True


def _require_dynamic(runtime: ContinuousSelfMTPRuntime) -> None:
    if runtime.config.allow_dynamic_membership is not True:
        raise ContinuousSelfMTPUnsupportedError(
            "dynamic membership is disabled; fixed-membership milestone only"
        )
    if runtime.capabilities.dynamic_membership is not True:
        raise ContinuousSelfMTPUnsupportedError(
            "dynamic membership lacks runtime capability attestation"
        )
    if (
        "flash" in runtime.config.architecture.lower()
        and runtime.capabilities.flash_dynamic_membership_attested is not True
    ):
        raise ContinuousSelfMTPUnsupportedError(
            "Flash dynamic membership lacks capability attestation"
        )


def _require_sampling(
    sampling: SelfMTPSampling, capabilities: ContinuousSelfMTPCapabilities
) -> None:
    if sampling.uses_xtc:
        raise ContinuousSelfMTPUnsupportedError(
            "XTC is not supported by the exact continuous self-MTP verifier"
        )
    if sampling.temperature > 0:
        if capabilities.per_lane_rng is not True:
            raise ContinuousSelfMTPUnsupportedError(
                "transformed sampling lacks per-lane RNG attestation"
            )
        if capabilities.transformed_sampling is not True:
            raise ContinuousSelfMTPUnsupportedError(
                "transformed sampling lacks exact-verification attestation"
            )
    if (
        sampling.has_logits_processors
        and capabilities.logits_processors_exact is not True
    ):
        raise ContinuousSelfMTPUnsupportedError(
            "logits processors lack exact-verification attestation"
        )


def prepare_self_mtp_lane(
    spec: SelfMTPLaneSpec,
    runtime: ContinuousSelfMTPRuntime,
) -> tuple[DetachedSelfMTPLane, MTPToken]:
    """Prepare one canonical lane without changing batch membership."""
    _require_fixed_core(runtime)
    _require_sampling(spec.sampling, runtime.capabilities)
    prepared = runtime.compute.prepare(spec, runtime.forwards)
    if not isinstance(prepared, PreparedLaneData):
        raise TypeError("compute.prepare must return PreparedLaneData")
    if prepared.first_token.from_draft:
        raise ContinuousSelfMTPError("the prepared first token cannot be a draft")
    if prepared.first_token.token != prepared.cur:
        raise ContinuousSelfMTPError("prepared first token and lane cur disagree")

    lane = SelfMTPLane(
        uid=spec.uid,
        cur=prepared.cur,
        seed_hidden=prepared.seed_hidden,
        token_prefix=prepared.token_prefix,
        ntoks=1,
        max_tokens=spec.max_tokens,
        num_draft=spec.num_draft,
        sampling=spec.sampling,
        backend_state=prepared.backend_state,
    )
    return (
        DetachedSelfMTPLane(lane, prepared.caches, runtime),
        prepared.first_token,
    )


def attach_self_mtp_lanes(
    batch: BatchedSelfMTPState | None,
    joining: Sequence[DetachedSelfMTPLane],
    *,
    runtime: ContinuousSelfMTPRuntime | None = None,
) -> BatchedSelfMTPState:
    """Attach initial lanes; later membership changes require attestation."""
    joining = list(joining)
    if batch is not None and batch.proposal_open:
        raise ContinuousSelfMTPError("cannot attach while a proposal is open")
    if batch is not None:
        _require_healthy(batch)
    if batch is None:
        if not joining:
            raise ValueError("cannot create an empty self-MTP batch")
        active_runtime = runtime or joining[0]._runtime
    else:
        active_runtime = batch._runtime
        if not joining:
            return batch
        _require_dynamic(active_runtime)
    _require_fixed_core(active_runtime)
    if any(item._runtime is not active_runtime for item in joining):
        raise ContinuousSelfMTPError("all lanes must share one runtime")

    existing = [] if batch is None else batch.lanes
    uids = [lane.uid for lane in existing] + [item.lane.uid for item in joining]
    if len(uids) != len(set(uids)):
        raise ValueError("self-MTP lane uid values must be unique")
    depths = {lane.num_draft for lane in existing}
    depths.update(item.lane.num_draft for item in joining)
    if len(depths) != 1:
        raise ValueError("adaptive per-lane draft depth is excluded")

    merged = active_runtime.caches.attach(
        None if batch is None else batch.caches,
        [item.caches for item in joining],
    )
    if batch is None:
        return BatchedSelfMTPState(
            lanes=[item.lane for item in joining],
            caches=merged,
            membership_epoch=1,
            _runtime=active_runtime,
        )
    batch.caches = merged
    batch.lanes.extend(item.lane for item in joining)
    batch.membership_epoch += 1
    return batch


def _validate_computation(
    batch: BatchedSelfMTPState, computation: CycleComputation
) -> None:
    n = len(batch.lanes)
    expected_uids = tuple(lane.uid for lane in batch.lanes)
    if computation.lane_uids != expected_uids:
        raise ContinuousSelfMTPError("compute backend changed lane order")
    vectors = (
        computation.draft_depths,
        computation.accepted_lengths,
        computation.target_drops,
        computation.draft_drops,
        computation.outputs,
    )
    if any(len(vector) != n for vector in vectors):
        raise ContinuousSelfMTPError("proposal vectors must match lane count")
    for row, (lane, depth, accepted, target_drop, draft_drop, outputs) in enumerate(
        zip(batch.lanes, *vectors)
    ):
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (depth, accepted, target_drop, draft_drop)
        ):
            raise ContinuousSelfMTPError(f"proposal row {row} has non-integer counts")
        if depth < 0 or accepted < 0 or accepted > depth:
            raise ContinuousSelfMTPError(f"proposal row {row} has invalid acceptance")
        maximum_depth = min(
            lane.num_draft,
            max(lane.max_tokens - lane.ntoks - 1, 0),
        )
        if depth > maximum_depth:
            raise ContinuousSelfMTPError(
                f"proposal row {row} exceeds its remaining draft budget"
            )
        if target_drop != depth - accepted or draft_drop != depth:
            raise ContinuousSelfMTPError(f"proposal row {row} rollback is inconsistent")
        if len(outputs) != accepted + 1:
            raise ContinuousSelfMTPError(
                f"proposal row {row} must contain accepted drafts plus one target token"
            )
        if any(not isinstance(token, MTPToken) for token in outputs):
            raise ContinuousSelfMTPError(f"proposal row {row} has an invalid token")
        if any(not token.from_draft for token in outputs[:-1]):
            raise ContinuousSelfMTPError(
                f"proposal row {row} accepted output is not marked as a draft"
            )
        if outputs[-1].from_draft:
            raise ContinuousSelfMTPError(
                f"proposal row {row} final target token is marked as a draft"
            )


def propose_batched_self_mtp(batch: BatchedSelfMTPState) -> SelfMTPCycleResult:
    """Open one fixed-membership batched draft/verify transaction."""
    _require_fixed_core(batch._runtime)
    _require_healthy(batch)
    if batch.proposal_open:
        raise ContinuousSelfMTPError("a self-MTP proposal is already open")
    if not batch.lanes:
        raise ValueError("cannot propose on an empty self-MTP batch")
    computation: CycleComputation | None = None
    try:
        candidate = batch._runtime.compute.propose(
            batch.lanes, batch.caches, batch._runtime.forwards
        )
        if not isinstance(candidate, CycleComputation):
            raise TypeError("compute.propose must return CycleComputation")
        computation = candidate
        _validate_computation(batch, computation)
    except BaseException as error:  # noqa: BLE001 - transaction includes cancellation
        _abort_backend_transaction(batch, computation, error)
        raise
    assert computation is not None
    proposal = SelfMTPCycleResult(
        membership_epoch=batch.membership_epoch,
        lane_uids=computation.lane_uids,
        draft_depths=computation.draft_depths,
        accepted_lengths=computation.accepted_lengths,
        target_drops=computation.target_drops,
        draft_drops=computation.draft_drops,
        outputs=computation.outputs,
        _computation=computation,
    )
    batch.proposal_open = True
    batch._open_proposal = proposal
    return proposal


def commit_batched_self_mtp(
    batch: BatchedSelfMTPState,
    proposal: SelfMTPCycleResult,
    *,
    emitted_counts: Sequence[int],
    terminal: Sequence[bool],
) -> None:
    """Commit exactly the delivered prefix of the open proposal."""
    _require_healthy(batch)
    if not batch.proposal_open or batch._open_proposal is not proposal:
        raise ContinuousSelfMTPError("commit requires the currently open proposal")
    if proposal.membership_epoch != batch.membership_epoch:
        raise ContinuousSelfMTPError("membership changed during an open proposal")
    if proposal.lane_uids != tuple(lane.uid for lane in batch.lanes):
        raise ContinuousSelfMTPError("lane order changed during an open proposal")
    n = len(batch.lanes)
    if len(emitted_counts) != n or len(terminal) != n:
        raise ValueError("commit vectors must have one entry per lane")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in emitted_counts
    ):
        raise ValueError("emitted_counts values must be integers")
    if any(not isinstance(value, bool) for value in terminal):
        raise ValueError("terminal values must be booleans")
    emitted = tuple(emitted_counts)
    terminal_flags = tuple(terminal)
    delivery_drops: list[int] = []
    for row, (count, is_terminal, outputs, accepted) in enumerate(
        zip(emitted, terminal_flags, proposal.outputs, proposal.accepted_lengths)
    ):
        if count < 0 or count > len(outputs):
            raise ValueError(f"lane {row} emitted_count {count} is out of range")
        if not is_terminal and count != len(outputs):
            raise ValueError("a nonterminal lane must consume its entire proposal")
        delivery_drops.append(
            accepted - count + 1 if is_terminal and count <= accepted else 0
        )

    try:
        # Keep proposal cache state provisional until the exact delivered
        # prefix is known. Rejected drafts and tokens hidden by a terminal
        # condition must be rewound in one atomic operation: recurrent cache
        # snapshots describe the original verify block and cannot safely be
        # consumed by two sequential trims.
        target_drops = tuple(
            rejected + undelivered
            for rejected, undelivered in zip(proposal.target_drops, delivery_drops)
        )
        batch._runtime.caches.rollback(
            batch.caches,
            target_drops=target_drops,
            draft_drops=proposal.draft_drops,
            verify_width=max(depth + 1 for depth in proposal.draft_depths),
        )
        batch._runtime.compute.commit(
            batch.lanes,
            proposal._computation,
            emitted_counts=emitted,
            terminal=terminal_flags,
        )
    except BaseException as error:  # noqa: BLE001 - transaction includes cancellation
        _abort_backend_transaction(batch, proposal._computation, error)
        batch.proposal_open = False
        batch._open_proposal = None
        raise
    for lane, count in zip(batch.lanes, emitted):
        lane.ntoks += count
    batch.proposal_open = False
    batch._open_proposal = None


def abort_batched_self_mtp(
    batch: BatchedSelfMTPState,
    proposal: SelfMTPCycleResult,
) -> None:
    """Abort the open proposal and restore its pre-proposal boundary."""

    _require_healthy(batch)
    if not batch.proposal_open or batch._open_proposal is not proposal:
        raise ContinuousSelfMTPError("abort requires the currently open proposal")
    _abort_backend_transaction(batch, proposal._computation, None)
    batch.proposal_open = False
    batch._open_proposal = None


def detach_self_mtp_lanes(
    batch: BatchedSelfMTPState,
    indices: Sequence[int],
) -> tuple[BatchedSelfMTPState, list[DetachedSelfMTPLane]]:
    """Detach all lanes at teardown; partial detach requires dynamic support."""
    _require_healthy(batch)
    if batch.proposal_open:
        raise ContinuousSelfMTPError("cannot detach while a proposal is open")
    requested = [int(index) for index in indices]
    if len(requested) != len(set(requested)):
        raise ValueError("detach indices must be unique")
    if any(index < 0 or index >= len(batch.lanes) for index in requested):
        raise IndexError("detach index is outside the self-MTP batch")
    if not requested:
        return batch, []
    if len(requested) != len(batch.lanes):
        _require_dynamic(batch._runtime)

    leaving = set(requested)
    keep = [index for index in range(len(batch.lanes)) if index not in leaving]
    remaining_caches, detached_caches = batch._runtime.caches.detach(
        batch.caches, requested, keep
    )
    if len(detached_caches) != len(requested):
        raise ContinuousSelfMTPError("cache adapter returned wrong detached row count")
    detached: list[DetachedSelfMTPLane] = []
    for index, caches in zip(requested, detached_caches):
        lane = batch.lanes[index]
        batch._runtime.compute.detach_lane(lane, caches)
        lane.pending_hidden = None
        lane.pending_tokens = []
        detached.append(DetachedSelfMTPLane(lane, caches, batch._runtime))
    batch.lanes = [batch.lanes[index] for index in keep]
    batch.caches = remaining_caches
    batch.membership_epoch += 1
    return batch, detached


__all__ = [
    "BatchedSelfMTPState",
    "ContinuousSelfMTPConfig",
    "ContinuousSelfMTPCapabilities",
    "ContinuousSelfMTPError",
    "ContinuousSelfMTPRuntime",
    "ContinuousSelfMTPUnsupportedError",
    "CycleComputation",
    "DetachedSelfMTPLane",
    "MTPToken",
    "PreparedLaneData",
    "RapidForwardSeams",
    "SelfMTPCacheAdapter",
    "SelfMTPCachePair",
    "SelfMTPComputeBackend",
    "SelfMTPCycleResult",
    "SelfMTPLane",
    "SelfMTPLaneSpec",
    "SelfMTPSampling",
    "attach_self_mtp_lanes",
    "abort_batched_self_mtp",
    "commit_batched_self_mtp",
    "detach_self_mtp_lanes",
    "prepare_self_mtp_lane",
    "propose_batched_self_mtp",
    "supports_dynamic_membership",
]
