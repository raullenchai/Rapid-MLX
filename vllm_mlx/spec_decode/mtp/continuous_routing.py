# SPDX-License-Identifier: Apache-2.0
"""Pure integration planner for default-off continuous self-MTP.

This module is the scheduler boundary, not a live token coordinator.  It turns
immutable request metadata into a fixed-cohort plan, validates joint APC/MTP
prepared-state sidecars, and chooses continuous, legacy-MTP, queue, or plain
decode without touching scheduler queues, caches, or generation batches.

The scheduler may attach an admitted router for later use, but PR 9 deliberately
leaves its existing vendored MTP ``_step`` implementation as the live data
plane.  ``live_token_delivery`` therefore remains unconditionally false.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .batched import (
    AdmissionDecision,
    BatchedMTPCapabilities,
    BatchedMTPConfig,
    BatchedMTPRoute,
    LaneAdmission,
    SamplingContract,
    plan_admission,
)
from .continuous_engine import SelfMTPLaneSpec, SelfMTPSampling
from .prepared_state import (
    PreparedMTPState,
    PreparedStateIdentity,
    RestoreEligibility,
    evaluate_restore,
)


class ContinuousMTPIntegrationRoute(str, enum.Enum):
    CONTINUOUS_PLANNED = "continuous_planned"
    LEGACY_MTP = "legacy_mtp"
    PLAIN_DECODE = "plain_decode"
    QUEUE = "queue"


@dataclass(frozen=True)
class ContinuousMTPAPCHit:
    """Opaque joint target/MTP/hidden sidecar plus its live cache counts."""

    state: PreparedMTPState
    expected_identity: PreparedStateIdentity
    target_cache_tokens: int
    mtp_cache_pairs: int
    now: float | None = None
    max_age_seconds: float | None = None
    min_useful_prefix_tokens: int = 64


@dataclass(frozen=True)
class ContinuousMTPRequestMetadata:
    """Scheduler-owned facts copied into the non-mutating planner."""

    lane_id: str
    uid: int
    prompt_tokens: tuple[int, ...]
    max_tokens: int
    stop_tokens: frozenset[int] = frozenset()
    sampling: SamplingContract = SamplingContract()
    temperature: float = 0.0
    base_bytes: int = 0
    bytes_per_draft_token: int = 0
    cache_ready: bool = True
    cache_quantized: bool = False
    cache_windowed: bool = False
    terminal: bool = False
    apc_hit: ContinuousMTPAPCHit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.uid, int) or isinstance(self.uid, bool):
            raise ValueError("uid must be an integer")
        if not self.prompt_tokens or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.prompt_tokens
        ):
            raise ValueError("prompt_tokens must contain non-negative integers")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if any(
            not isinstance(token, int) or isinstance(token, bool)
            for token in self.stop_tokens
        ):
            raise ValueError("stop_tokens must contain integers")
        if self.sampling.greedy and self.temperature != 0:
            raise ValueError("greedy sampling requires temperature=0")
        if not self.sampling.greedy and self.temperature <= 0:
            raise ValueError("non-greedy sampling requires temperature>0")


@dataclass(frozen=True)
class ContinuousMTPRuntimeFacts:
    """Static install facts; no architecture or cache support is inferred."""

    model_family: str
    capability_descriptor: Mapping[str, Any]
    capabilities: BatchedMTPCapabilities
    legacy_mtp_available: bool
    cache_quantized: bool = False
    cache_windowed: bool = False

    def __post_init__(self) -> None:
        descriptor = MappingProxyType(dict(self.capability_descriptor))
        object.__setattr__(self, "capability_descriptor", descriptor)


@dataclass(frozen=True)
class PlannedContinuousMTPLane:
    lane_id: str
    spec: SelfMTPLaneSpec
    stop_tokens: frozenset[int]
    prepared_state: PreparedMTPState | None = None
    resume_at: int | None = None


@dataclass(frozen=True)
class ContinuousMTPRoutingDecision:
    route: ContinuousMTPIntegrationRoute
    cohort: tuple[PlannedContinuousMTPLane, ...] = ()
    legacy_lane_ids: tuple[str, ...] = ()
    plain_lane_ids: tuple[str, ...] = ()
    queued_lane_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    admission: AdmissionDecision | None = None
    live_token_delivery: bool = False


@dataclass(frozen=True)
class ContinuousMTPRouterInstallDecision:
    admitted: bool
    router: ContinuousMTPIntegrationRouter | None
    fallback: ContinuousMTPIntegrationRoute
    reasons: tuple[str, ...] = ()
    live_token_delivery: bool = False


class ContinuousMTPIntegrationRouter:
    """Immutable, repeatable planner for one loaded model/runtime."""

    def __init__(
        self,
        *,
        config: BatchedMTPConfig,
        runtime: ContinuousMTPRuntimeFacts,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self._static_refusals = _runtime_refusals(config, runtime)

    @property
    def static_refusals(self) -> tuple[str, ...]:
        return self._static_refusals

    def plan(
        self,
        requests: Sequence[ContinuousMTPRequestMetadata],
        *,
        free_bytes: int,
    ) -> ContinuousMTPRoutingDecision:
        requests = tuple(requests)
        lane_ids = tuple(request.lane_id for request in requests)
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("lane_id values must be unique")
        uids = tuple(request.uid for request in requests)
        if len(uids) != len(set(uids)):
            raise ValueError("uid values must be unique")
        if self._static_refusals:
            return _fallback_decision(
                lane_ids,
                legacy=self.runtime.legacy_mtp_available,
                reasons=self._static_refusals,
            )

        by_id = {request.lane_id: request for request in requests}
        restore: dict[str, RestoreEligibility] = {}
        apc_plain: list[str] = []
        apc_reasons: list[str] = []
        candidates: list[LaneAdmission] = []
        for request in requests:
            if request.cache_quantized or request.cache_windowed:
                apc_plain.append(request.lane_id)
                apc_reasons.append(
                    f"{request.lane_id}: quantized/windowed cache is unsupported"
                )
                continue
            if request.apc_hit is not None:
                eligibility = _evaluate_apc(request)
                restore[request.lane_id] = eligibility
                if not eligibility.eligible and not eligibility.bypass_hit:
                    apc_plain.append(request.lane_id)
                    apc_reasons.append(
                        f"{request.lane_id}: APC prepared state refused "
                        f"({eligibility.reason.value})"
                    )
                    continue
            candidates.append(
                LaneAdmission(
                    lane_id=request.lane_id,
                    base_bytes=request.base_bytes,
                    bytes_per_draft_token=request.bytes_per_draft_token,
                    sampling=request.sampling,
                    cache_ready=request.cache_ready,
                    terminal=request.terminal,
                )
            )

        admission = plan_admission(
            candidates,
            config=self.config,
            capabilities=self.runtime.capabilities,
            free_bytes=free_bytes,
        )
        reasons = tuple(apc_reasons) + admission.reasons
        if admission.route is not BatchedMTPRoute.BATCHED_MTP:
            fallback_ids = tuple(
                lane_id for lane_id in lane_ids if lane_id not in apc_plain
            )
            if admission.route is BatchedMTPRoute.QUEUE and not fallback_ids:
                route = ContinuousMTPIntegrationRoute.QUEUE
            elif self.runtime.legacy_mtp_available and fallback_ids:
                route = ContinuousMTPIntegrationRoute.LEGACY_MTP
            else:
                route = ContinuousMTPIntegrationRoute.PLAIN_DECODE
            return ContinuousMTPRoutingDecision(
                route=route,
                legacy_lane_ids=(
                    fallback_ids
                    if route is ContinuousMTPIntegrationRoute.LEGACY_MTP
                    else ()
                ),
                plain_lane_ids=tuple(apc_plain)
                + (
                    fallback_ids
                    if route is ContinuousMTPIntegrationRoute.PLAIN_DECODE
                    else ()
                ),
                queued_lane_ids=admission.queued_lane_ids,
                reasons=reasons,
                admission=admission,
            )

        cohort = tuple(
            _planned_lane(by_id[lane_id], admission.draft_tokens, restore.get(lane_id))
            for lane_id in admission.batched_lane_ids
        )
        fallthrough = tuple(
            lane_id
            for lane_id in lane_ids
            if lane_id not in admission.batched_lane_ids and lane_id not in apc_plain
        )
        return ContinuousMTPRoutingDecision(
            route=ContinuousMTPIntegrationRoute.CONTINUOUS_PLANNED,
            cohort=cohort,
            legacy_lane_ids=(fallthrough if self.runtime.legacy_mtp_available else ()),
            plain_lane_ids=tuple(apc_plain)
            + (() if self.runtime.legacy_mtp_available else fallthrough),
            queued_lane_ids=admission.queued_lane_ids,
            reasons=reasons,
            admission=admission,
        )


def plan_router_install(
    model: Any,
    *,
    enabled: bool,
    cache_quantized: bool = False,
    cache_windowed: bool = False,
    max_lanes: int = 4,
    min_batch_lanes: int = 2,
    hard_reserve_bytes: int = 8 * 1024**3,
    allow_dynamic_membership: bool = False,
) -> ContinuousMTPRouterInstallDecision:
    """Feature-detect a model and return an install plan without mutation."""
    candidate = getattr(model, "language_model", model)
    descriptor = getattr(candidate, "batched_mtp_capability", None)
    descriptor_map = descriptor if isinstance(descriptor, Mapping) else {}
    family = descriptor_map.get("model_family", "unknown")
    batch_forward_name = descriptor_map.get("batch_forward")
    batch_forward = (
        getattr(candidate, batch_forward_name, None)
        if isinstance(batch_forward_name, str)
        else None
    )
    legacy = (
        all(
            callable(getattr(candidate, name, None))
            for name in ("mtp_forward", "make_mtp_cache")
        )
        and getattr(candidate, "mtp", None) is not None
    )
    descriptor_core = (
        descriptor_map.get("protocol_version") == 1
        and descriptor_map.get("fixed_membership") is True
        and descriptor_map.get("recursive_draft_depth") == 2
        and callable(batch_forward)
    )
    capabilities = BatchedMTPCapabilities(
        target_batch_forward=descriptor_core and callable(candidate),
        mtp_batch_forward=descriptor_core,
        ragged_target_rollback=descriptor_core
        and not cache_quantized
        and not cache_windowed,
        ragged_mtp_rollback=descriptor_core
        and not cache_quantized
        and not cache_windowed,
        atomic_cache_commit=descriptor_core
        and not cache_quantized
        and not cache_windowed,
        dynamic_membership=descriptor_core
        and descriptor_map.get("dynamic_join") is True,
    )
    runtime = ContinuousMTPRuntimeFacts(
        model_family=str(family),
        capability_descriptor=descriptor_map,
        capabilities=capabilities,
        legacy_mtp_available=legacy,
        cache_quantized=cache_quantized,
        cache_windowed=cache_windowed,
    )
    router = ContinuousMTPIntegrationRouter(
        config=BatchedMTPConfig(
            enabled=enabled,
            max_lanes=max_lanes,
            min_batch_lanes=min_batch_lanes,
            hard_reserve_bytes=hard_reserve_bytes,
            allow_dynamic_membership=allow_dynamic_membership,
        ),
        runtime=runtime,
    )
    if router.static_refusals:
        return ContinuousMTPRouterInstallDecision(
            admitted=False,
            router=None,
            fallback=(
                ContinuousMTPIntegrationRoute.LEGACY_MTP
                if legacy
                else ContinuousMTPIntegrationRoute.PLAIN_DECODE
            ),
            reasons=router.static_refusals,
        )
    return ContinuousMTPRouterInstallDecision(
        admitted=True,
        router=router,
        fallback=ContinuousMTPIntegrationRoute.LEGACY_MTP,
    )


def _runtime_refusals(
    config: BatchedMTPConfig,
    runtime: ContinuousMTPRuntimeFacts,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if config.enabled is not True:
        reasons.append("continuous self-MTP is disabled")
    if runtime.model_family != "qwen3_5":
        reasons.append(f"unsupported model family: {runtime.model_family}")
    descriptor = runtime.capability_descriptor
    required_descriptor = {
        "protocol_version": 1,
        "recursive_draft_depth": 2,
        "fixed_membership": True,
        "dynamic_join": True,
        "quantized_cache": False,
        "windowed_cache": False,
        "xtc": False,
    }
    for name, expected in required_descriptor.items():
        if descriptor.get(name) != expected:
            reasons.append(f"capability descriptor mismatch: {name}")
    if runtime.cache_quantized:
        reasons.append("quantized cache is unsupported")
    if runtime.cache_windowed:
        reasons.append("windowed cache is unsupported")
    if config.allow_dynamic_membership and not runtime.capabilities.dynamic_membership:
        reasons.append("dynamic membership is not attested")
    reasons.extend(
        f"missing capability: {name}" for name in runtime.capabilities.missing_core()
    )
    return tuple(reasons)


def _evaluate_apc(request: ContinuousMTPRequestMetadata) -> RestoreEligibility:
    hit = request.apc_hit
    assert hit is not None
    return evaluate_restore(
        hit.state,
        expected_identity=hit.expected_identity,
        request_tokens=request.prompt_tokens,
        target_cache_tokens=hit.target_cache_tokens,
        mtp_cache_pairs=hit.mtp_cache_pairs,
        now=hit.now,
        max_age_seconds=hit.max_age_seconds,
        min_useful_prefix_tokens=hit.min_useful_prefix_tokens,
    )


def _planned_lane(
    request: ContinuousMTPRequestMetadata,
    draft_tokens: int,
    restore: RestoreEligibility | None,
) -> PlannedContinuousMTPLane:
    prepared = None
    resume_at = None
    prompt = request.prompt_tokens
    target_cache = None
    mtp_cache = None
    if restore is not None and restore.eligible:
        assert request.apc_hit is not None and restore.resume_at is not None
        prepared = request.apc_hit.state
        resume_at = restore.resume_at
        prompt = request.prompt_tokens[resume_at:]
        target_cache = prepared.target_cache
        mtp_cache = prepared.mtp_cache
    return PlannedContinuousMTPLane(
        lane_id=request.lane_id,
        spec=SelfMTPLaneSpec(
            uid=request.uid,
            prompt=prompt,
            max_tokens=request.max_tokens,
            num_draft=draft_tokens,
            sampling=SelfMTPSampling(
                temperature=request.temperature,
                has_logits_processors=request.sampling.has_logits_processors,
                uses_xtc=request.sampling.uses_xtc,
            ),
            prompt_cache=target_cache,
            mtp_cache=mtp_cache,
        ),
        stop_tokens=request.stop_tokens,
        prepared_state=prepared,
        resume_at=resume_at,
    )


def _fallback_decision(
    lane_ids: tuple[str, ...],
    *,
    legacy: bool,
    reasons: tuple[str, ...],
) -> ContinuousMTPRoutingDecision:
    route = (
        ContinuousMTPIntegrationRoute.LEGACY_MTP
        if legacy
        else ContinuousMTPIntegrationRoute.PLAIN_DECODE
    )
    return ContinuousMTPRoutingDecision(
        route=route,
        legacy_lane_ids=lane_ids if legacy else (),
        plain_lane_ids=() if legacy else lane_ids,
        reasons=reasons,
    )


__all__ = [
    "ContinuousMTPAPCHit",
    "ContinuousMTPIntegrationRoute",
    "ContinuousMTPIntegrationRouter",
    "ContinuousMTPRequestMetadata",
    "ContinuousMTPRouterInstallDecision",
    "ContinuousMTPRoutingDecision",
    "ContinuousMTPRuntimeFacts",
    "PlannedContinuousMTPLane",
    "plan_router_install",
]
