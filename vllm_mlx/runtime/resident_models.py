"""Budgeted lifecycle management for models resident in one server process."""

from __future__ import annotations

import asyncio
import gc
import logging
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol

from .model_registry import ModelEntry, ModelRegistry
from .process_memory import get_phys_footprint

logger = logging.getLogger(__name__)

_GIB = 1024**3
_PARAM_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b(?![a-z])", re.IGNORECASE)
_QUANT_RE = re.compile(r"(?<!\d)(2|3|4|6|8|16)[-]?bit", re.IGNORECASE)


class ResidentModelError(RuntimeError):
    """Base class for resident-model control-plane failures."""


class ResidentModelCapacityError(ResidentModelError):
    """The configured ceiling cannot admit a model after eligible eviction."""

    def __init__(
        self,
        message: str,
        *,
        replacement_projection: ReplacementProjection | None = None,
    ) -> None:
        super().__init__(message)
        self.replacement_projection = replacement_projection


class ResidentModelBusyError(ResidentModelError):
    """A model cannot be removed while it owns active work."""


class _CommittedReplacementCancelled(asyncio.CancelledError):
    """Cancellation observed after replacement routing became authoritative."""


@dataclass(frozen=True)
class RoleRef:
    """Identity and charge of one runtime role in a capacity decision."""

    role: str
    model: str
    bytes: int

    def payload(self) -> dict[str, object]:
        return {"role": self.role, "model": self.model, "bytes": self.bytes}


@dataclass(frozen=True)
class RoleConflict:
    """A budget holder blocking admission of another role."""

    role: str
    model: str
    bytes: int
    evictable: bool
    reason: str

    def payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "model": self.model,
            "bytes": self.bytes,
            "evictable": self.evictable,
            "reason": self.reason,
        }


class ResidentRoleConflictError(ResidentModelCapacityError):
    """Typed auxiliary-role conflict rendered as an HTTP 507 by callers."""

    def __init__(
        self,
        *,
        requested: RoleRef,
        conflicts: list[RoleConflict],
        limit_bytes: int,
        usage_bytes: int,
        capacity_unknown: bool = False,
        measured_overshoot: bool = False,
        request_buffer: bool = False,
        request_bytes: int = 0,
        param: str | None = None,
    ) -> None:
        self.requested = requested
        self.conflicts = conflicts
        self.limit_bytes = limit_bytes
        self.usage_bytes = usage_bytes
        self.capacity_unknown = capacity_unknown
        self.measured_overshoot = measured_overshoot
        self.request_buffer = request_buffer
        self.request_bytes = max(0, int(request_bytes))
        if request_buffer and not self.request_bytes:
            self.request_bytes = max(0, requested.bytes)
        if param is None and request_buffer:
            param = {
                ROLE_SPEECH_INPUT: "file",
                ROLE_ALIGNMENT: "file",
                ROLE_SPEECH_OUTPUT: "input",
            }.get(requested.role, "model")
        self.param = param or "model"
        super().__init__(self.message)

    @property
    def code(self) -> str:
        if self.capacity_unknown:
            return "role_capacity_unknown"
        if self.request_buffer:
            return "role_request_too_large"
        return "role_capacity_conflict"

    @property
    def message(self) -> str:
        if self.capacity_unknown:
            return (
                f"cannot load {self.requested.model!r} as {self.requested.role}: "
                "its memory footprint could not be determined from the model "
                "catalog or the local cache, so it cannot be admitted against "
                f"the {self.limit_bytes / _GIB:.2f} GiB ceiling. Download the "
                "model first, then retry."
            )
        held = ", ".join(
            f"{conflict.role} ({conflict.model}, {conflict.bytes / _GIB:.2f} GiB)"
            for conflict in self.conflicts
        )
        if self.request_buffer:
            return (
                f"this {self.requested.role} request needs "
                f"{self.request_bytes / _GIB:.2f} GiB of working memory, which "
                f"does not fit under the {self.limit_bytes / _GIB:.2f} GiB "
                f"ceiling already holding {held or 'the running process'}. Send "
                "a shorter request."
            )
        if self.measured_overshoot:
            return (
                f"unloaded {self.requested.model!r} ({self.requested.role}): it "
                f"measured {self.requested.bytes / _GIB:.2f} GiB once loaded, "
                f"which exceeds the {self.limit_bytes / _GIB:.2f} GiB ceiling "
                f"already holding {held or 'the running process'}"
            )
        return (
            f"cannot load {self.requested.model!r} as {self.requested.role}: "
            f"it needs {self.requested.bytes / _GIB:.2f} GiB and "
            f"{self.usage_bytes / _GIB:.2f} GiB of the "
            f"{self.limit_bytes / _GIB:.2f} GiB ceiling is already held by "
            f"{held or 'the running process'}"
        )

    def envelope(self) -> dict[str, object]:
        """Render the OpenAI-shaped error body served with HTTP 507."""

        return {
            "error": {
                "message": self.message,
                "type": "insufficient_capacity_error",
                "code": self.code,
                "param": self.param,
                "requested": self.requested.payload(),
                "request_bytes": self.request_bytes,
                "limit_bytes": self.limit_bytes,
                "usage_bytes": self.usage_bytes,
                "conflicts": [conflict.payload() for conflict in self.conflicts],
            }
        }


@dataclass(frozen=True)
class ResidentPerformanceConfig:
    """Audited scheduler overrides attached to one resident text model.

    ``None`` fields mean no operator opinion. Keeping this import-light value
    in the lifecycle layer lets the FastAPI request model and desktop client
    share a typed contract without making residency depend on CLI argv.
    """

    kv_cache_dtype: str | None = None
    kv_cache_turboquant: str | None = None
    prefix_cache_enabled: bool | None = None
    cache_memory_mb: int | None = None

    @property
    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (
                self.kv_cache_dtype,
                self.kv_cache_turboquant,
                self.prefix_cache_enabled,
                self.cache_memory_mb,
            )
        )

    def payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "kv_cache_dtype": self.kv_cache_dtype,
                "kv_cache_turboquant": self.kv_cache_turboquant,
                "prefix_cache_enabled": self.prefix_cache_enabled,
                "cache_memory_mb": self.cache_memory_mb,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class ReplacementProjection:
    """Admission truth for one assistant replacement request."""

    strategy: str
    reason: str
    models_to_free: tuple[tuple[str, int], ...]
    current_bytes: int
    requested_bytes: int
    projected_bytes: int
    limit_bytes: int

    def payload(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "models_to_free": [
                {"id": model_id, "estimated_bytes": estimated_bytes}
                for model_id, estimated_bytes in self.models_to_free
            ],
            "current_bytes": self.current_bytes,
            "requested_bytes": self.requested_bytes,
            "projected_bytes": self.projected_bytes,
            "limit_bytes": self.limit_bytes,
        }


def resident_scheduler_kwargs(
    performance: ResidentPerformanceConfig | None,
) -> dict[str, object]:
    """Translate the control-plane value into ``SchedulerConfig`` fields."""

    if performance is None:
        return {}
    result: dict[str, object] = {}
    if performance.kv_cache_dtype is not None:
        from ..kv_cache_dtype import dtype_to_quantization_bits

        quantized, bits = dtype_to_quantization_bits(performance.kv_cache_dtype)
        result.update(
            kv_cache_dtype=performance.kv_cache_dtype,
            kv_cache_quantization=quantized,
            kv_cache_quantization_bits=bits,
        )
    if performance.kv_cache_turboquant is not None:
        result.update(
            kv_cache_turboquant=True,
            kv_cache_turboquant_mode=performance.kv_cache_turboquant,
        )
    if performance.prefix_cache_enabled is not None:
        result["enable_prefix_cache"] = performance.prefix_cache_enabled
    if performance.cache_memory_mb is not None:
        result["cache_memory_mb"] = performance.cache_memory_mb
    return result


def resolve_resident_performance(
    performance: ResidentPerformanceConfig | None,
    *,
    model_name: str,
    model_path: str | None,
) -> ResidentPerformanceConfig | None:
    """Apply the same audited KV-cache eligibility gate as CLI startup."""

    if performance is None or performance.kv_cache_dtype is None:
        return performance

    # Keep startup and runtime residency on one gate. Importing lazily avoids
    # pulling the CLI dependency graph into the lifecycle module at import time.
    from ..cli import _gather_kv_cache_dtype_inputs
    from ..kv_cache_dtype import log_kv_cache_decision, resolve_kv_cache_dtype

    lookup_name = model_path or model_name
    hf_config, alias_metadata = _gather_kv_cache_dtype_inputs(lookup_name)
    decision = resolve_kv_cache_dtype(
        performance.kv_cache_dtype,
        model_name=model_name,
        hf_path=model_path or (alias_metadata or {}).get("hf_path"),
        hf_config=hf_config,
        alias_metadata=alias_metadata,
    )
    log_kv_cache_decision(decision, model_name=model_name)
    if decision.dtype == performance.kv_cache_dtype:
        return performance
    return ResidentPerformanceConfig(
        kv_cache_dtype=decision.dtype,
        kv_cache_turboquant=performance.kv_cache_turboquant,
        prefix_cache_enabled=performance.prefix_cache_enabled,
        cache_memory_mb=performance.cache_memory_mb,
    )


def _carry_served_identity(entry: ModelEntry, prior: ModelEntry) -> ModelEntry:
    """Attach the exact pre-reload routing identity to a rebuilt engine."""
    entry.model_name = prior.model_name
    entry.model_path = prior.model_path
    entry.aliases = set(prior.aliases)
    return entry


@dataclass
class ResidencyRecord:
    """Mutable lifecycle metadata kept outside the route-facing registry entry."""

    entry: ModelEntry
    estimated_bytes: int
    loaded_at: float
    last_used_at: float
    pinned: bool = False
    primary: bool = False
    active_requests: int = 0
    state: str = "resident"
    measured_bytes: int = 0
    performance: ResidentPerformanceConfig | None = None
    replacement_projection: ReplacementProjection | None = None
    lease_idle: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def __post_init__(self) -> None:
        if self.active_requests == 0:
            self.lease_idle.set()

    @property
    def model_id(self) -> str:
        return self.entry.model_name


Loader = Callable[..., Awaitable[ModelEntry]]
PrimaryChanged = Callable[[ModelEntry | None], None]


# Lifecycle roles used by admission and telemetry.
ROLE_CONVERSATION = "conversation"
ROLE_VISION = "vision"
ROLE_IMAGE_GEN = "image-gen"
ROLE_VIDEO_GEN = "video-gen"
ROLE_SPEECH_INPUT = "speech-input"
ROLE_SPEECH_OUTPUT = "speech-output"
ROLE_ALIGNMENT = "alignment"

_MODALITY_ROLES = {
    "text": ROLE_CONVERSATION,
    "mllm": ROLE_VISION,
    "image-gen": ROLE_IMAGE_GEN,
    "video-gen": ROLE_VIDEO_GEN,
}


@dataclass
class AuxiliaryRoleRecord:
    """Budget and lifecycle state for an audio role outside ModelRegistry."""

    role: str
    lane: str
    model_id: str
    reserved_bytes: int
    capacity_source: str
    weight_bytes: int | None = None
    measured_bytes: int = 0
    state: str = "admitting"
    active_requests: int = 0
    loaded_at: float = 0.0
    last_used_at: float = 0.0
    unload: Callable[[], object] | None = None
    request_bytes: int = 0
    pending_request_bytes: int = 0
    can_release: Callable[[], bool] | None = None

    @property
    def releasable(self) -> bool:
        if self.active_requests:
            return False
        if self.pending_request_bytes:
            return False
        if self.can_release is None:
            return True
        try:
            return bool(self.can_release())
        except Exception:
            return False

    @property
    def charged_bytes(self) -> int:
        return (
            max(self.reserved_bytes, self.measured_bytes)
            + self.request_bytes
            + self.pending_request_bytes
        )


class PrimaryHandoffLease(Protocol):
    """Serving-layer transaction coupled to a primary residency change."""

    def commit(self, entry: ModelEntry | None) -> None: ...

    def rollback(self) -> None: ...


PrimaryHandoff = Callable[[ModelEntry], PrimaryHandoffLease]


def _modality(entry: ModelEntry) -> str:
    engine = entry.engine
    if getattr(engine, "is_image_gen", False):
        return "image-gen"
    if getattr(engine, "is_video_gen", False):
        return "video-gen"
    if getattr(engine, "is_mllm", False):
        return "mllm"
    return "text"


def _replacement_group(entry: ModelEntry) -> str:
    """Map request-facing modalities to lifecycle replacement groups."""

    modality = _modality(entry)
    return "assistant" if modality in {"text", "mllm"} else modality


def _entry_role(entry: ModelEntry) -> str:
    """Lifecycle role of a registry-backed model."""

    return _MODALITY_ROLES.get(_modality(entry), ROLE_CONVERSATION)


# Generative-media lanes hold multi-GB checkpoints and are driven one model at
# a time, so they are inherently single-slot: loading another image/video model
# should evict the previous one even when the client sends no ``replace_group``.
# Text/VLM stay client-controlled through the explicit ``assistant`` group so
# the chat picker's replacement semantics are unchanged. Without this, image
# engines only ever accumulated (two resident image models measured at 9.1 GB).
_SINGLE_SLOT_MEDIA_GROUPS = frozenset({"image-gen", "video-gen"})


def _effective_replace_group(
    entry: ModelEntry, replace_group: str | None
) -> str | None:
    """Resolve the replacement group to enforce for a just-touched model.

    An explicit group must match the entry's actual modality group. Otherwise
    a generative-media entry derives its own single-slot group; everything else
    stays unmanaged (``None``) so a bare text load never evicts a sibling.
    """

    derived = _replacement_group(entry)
    if replace_group is not None:
        if replace_group != derived:
            raise ResidentModelError(
                f"model {entry.model_name!r} belongs to replacement group "
                f"{derived!r}, not {replace_group!r}"
            )
        return replace_group
    return derived if derived in _SINGLE_SLOT_MEDIA_GROUPS else None


def estimate_model_bytes(model_name: str) -> int:
    """Conservative fallback charge when the caller has no catalog estimate.

    This mirrors the desktop's weight + runtime + KV shape closely enough for
    admission, without pretending it is an allocator measurement. Callers that
    know the downloaded size should pass ``estimated_bytes`` to ``load``.
    """

    folded = model_name.casefold()
    known_image_gib = {
        "flux2-klein-4b": 5.9,
        "z-image-turbo": 5.9,
        # 6-bit-transformer Qwen-Image (20B) — measured peak RSS during a
        # real generation at 1024x1024 (mflux-community/qwen-image-mflux-q6,
        # the API/GUI default resolution; `/usr/bin/time -l`): ~55.7 GiB.
        # (512x512 measured lower, ~40.2 GiB — the API/GUI default is what
        # this charge must cover.) The text encoder in this repo is full
        # precision (quantizing it "causes significant semantic degradation"
        # per mflux's own weight definition), so it dominates the footprint
        # over the quantized transformer. Without this entry the digit-free
        # alias falls through to the 4 GB default and mis-admits.
        "qwen-image": 55.7,
    }
    for token, gib in known_image_gib.items():
        if token not in folded:
            continue
        if token == "qwen-image" and "qwen-image-edit" in folded:
            # "qwen-image" is a substring of "qwen-image-edit" — this charge
            # was measured against the txt2img family only (see the comment
            # above), and the edit variant's extra image-conditioning input
            # makes its real footprint unverified, not merely "the same
            # number". Falls through to the generic param-count estimate
            # below rather than asserting an unmeasured number.
            continue
        return int(gib * _GIB)

    params = [float(value) for value in _PARAM_RE.findall(folded)]
    if not params:
        return 4 * _GIB
    bits_match = _QUANT_RE.search(folded)
    bits = int(bits_match.group(1)) if bits_match else 4
    bytes_per_param = {
        2: 0.28,
        3: 0.42,
        4: 0.55,
        6: 0.80,
        8: 1.05,
        16: 2.0,
    }.get(bits, 0.55)
    largest = max(params)
    kv_gib = (
        1.5 if largest < 4 else 2.5 if largest < 10 else 4.0 if largest < 25 else 6.0
    )
    return int((largest * bytes_per_param + 1.2 + kv_gib) * _GIB)


def _engine_active_requests(engine: object) -> int | None:
    """Return the engine's running plus queued request count.

    ``ResidencyRecord.active_requests`` covers manager leases, but the primary
    startup engine is reached directly through the model registry. Its live
    text/VLM requests therefore exist only in the engine scheduler. Return
    ``None`` when an exposed activity probe fails so destructive lifecycle
    guards keep their existing fail-closed behavior.
    """

    active = 0

    progress = getattr(engine, "progress_snapshot", None)
    if callable(progress):
        try:
            if bool(progress().get("running", False)):
                active = 1
        except Exception:
            return None

    get_stats = getattr(engine, "get_stats", None)
    if callable(get_stats):
        try:
            stats = get_stats() or {}
            running = max(0, int(stats.get("num_running", 0) or 0))
            waiting = max(0, int(stats.get("num_waiting", 0) or 0))
            active = max(active, running + waiting)
        except Exception:
            return None
    return active


def _engine_is_idle(engine: object) -> bool:
    """Best-effort idle check shared by explicit, LRU, and TTL eviction."""

    return _engine_active_requests(engine) == 0


def _release_allocator_cache() -> None:
    """Return dead model buffers to MLX/Metal after dropping Python refs."""

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        # Non-MLX unit-test hosts and older MLX builds are valid here.
        pass


class ResidentModelManager:
    """Own dynamic engines and enforce a process-wide residency ceiling.

    Loads and evictions are serialized under one asyncio lock. The primary
    startup model is registered as pinned because legacy health/cache routes
    still expose it through ``ServerConfig.engine``; dynamic engines are fully
    owned by this manager and may be evicted.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        loader: Loader,
        *,
        memory_limit_bytes: int = 0,
        idle_ttl_seconds: float = 0,
        audio_role_idle_ttl_seconds: float = 0,
        clock: Callable[[], float] = time.monotonic,
        memory_reader: Callable[[], int] = get_phys_footprint,
        on_primary_handoff: PrimaryHandoff | None = None,
        on_primary_changed: PrimaryChanged | None = None,
    ) -> None:
        self.registry = registry
        self.loader = loader
        self.memory_limit_bytes = max(0, int(memory_limit_bytes))
        self.idle_ttl_seconds = max(0.0, float(idle_ttl_seconds))
        self.audio_role_idle_ttl_seconds = max(0.0, float(audio_role_idle_ttl_seconds))
        self._clock = clock
        self._memory_reader = memory_reader
        # Residency is configured before startup loading. Preserve the process
        # baseline so the protected startup model receives an attributable
        # footprint instead of treating Python/server memory as releasable.
        self._baseline_memory_bytes = self._read_memory()
        self._on_primary_handoff = on_primary_handoff
        self._on_primary_changed = on_primary_changed
        self._records: dict[str, ResidencyRecord] = {}
        self._index: dict[str, str] = {}
        self._roles: dict[str, AuxiliaryRoleRecord] = {}
        self._lock = asyncio.Lock()
        self._ttl_task: asyncio.Task | None = None
        self.evictions_total = 0
        self.loads_total = 0
        self.registry.on_engine_access = self.touch

    def _canonical(self, name: str | None) -> str | None:
        if not name or name == "default":
            return self.registry.default_name
        return self._index.get(name, name if name in self._records else None)

    def _index_record(self, record: ResidencyRecord) -> None:
        canonical = record.model_id
        self._records[canonical] = record
        self._index[canonical] = canonical
        self._index[record.entry.model_path] = canonical
        for alias in record.entry.aliases:
            self._index[alias] = canonical

    def _drop_record(self, canonical: str) -> ResidencyRecord | None:
        record = self._records.pop(canonical, None)
        if record is None:
            return None
        for key in [key for key, value in self._index.items() if value == canonical]:
            self._index.pop(key, None)
        return record

    def register_primary(
        self, entry: ModelEntry, *, estimated_bytes: int | None = None
    ) -> ResidencyRecord:
        """Register the already-started legacy engine as protected primary."""

        now = self._clock()
        record = ResidencyRecord(
            entry=entry,
            estimated_bytes=max(
                1, estimated_bytes or estimate_model_bytes(entry.model_name)
            ),
            measured_bytes=max(
                0,
                self._read_memory() - self._baseline_memory_bytes,
            ),
            loaded_at=now,
            last_used_at=now,
            pinned=True,
            primary=True,
        )
        self._index_record(record)
        return record

    def _read_memory(self) -> int:
        try:
            return max(0, int(self._memory_reader()))
        except Exception:
            return 0

    def _accounted_usage(self) -> int:
        measured = self._read_memory()
        resident_reservations = sum(
            max(record.estimated_bytes, record.measured_bytes)
            for record in self._records.values()
            if record.state == "resident"
        )
        resident_reservations += sum(
            max(role.reserved_bytes, role.measured_bytes)
            for role in self._roles.values()
            if role.state in {"resident", "evicting"}
        )

        # The footprint and resident reservations are two views of the same
        # aggregate allocation. Taking their maximum preserves lazy model
        # estimates without adding a secondary estimate again after the
        # process footprint has already grown to cover the complete resident
        # set. Audio weights still loading are different: their promised bytes
        # are not part of the settled aggregate yet, so concurrent admissions
        # must continue to treat them as pending.
        pending_weight_reservations = sum(
            max(role.reserved_bytes, role.measured_bytes)
            for role in self._roles.values()
            if role.state in {"admitting", "loading"}
        )
        request_reservations = sum(
            role.request_bytes + role.pending_request_bytes
            for role in self._roles.values()
            if role.state in {"admitting", "loading", "resident", "evicting"}
        )
        return (
            max(measured, resident_reservations)
            + pending_weight_reservations
            + request_reservations
        )

    def contains(self, model_name: str) -> bool:
        return self._canonical(model_name) is not None

    def touch(self, model_name: str | None) -> None:
        canonical = self._canonical(model_name)
        if canonical and canonical in self._records:
            self._records[canonical].last_used_at = self._clock()

    async def start(self) -> None:
        if self._ttl_task is not None:
            return
        # Either ledger having a TTL is reason enough to run the sweeper.
        # Gating only on ``idle_ttl_seconds`` would silently disable the
        # audio-role TTL on a server started without a model idle TTL.
        if self.idle_ttl_seconds <= 0 and self.audio_role_idle_ttl_seconds <= 0:
            return
        self._ttl_task = asyncio.create_task(self._ttl_loop())

    async def shutdown(self) -> None:
        task = self._ttl_task
        self._ttl_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for role in list(self._roles.values()):
                await self._release_role_locked(role, reason="shutdown")
            dynamic = [
                record for record in self._records.values() if not record.primary
            ]
            for record in dynamic:
                await self._evict_locked(record, reason="shutdown", count=False)

    async def _ttl_loop(self) -> None:
        ttls = [
            value
            for value in (self.idle_ttl_seconds, self.audio_role_idle_ttl_seconds)
            if value > 0
        ]
        interval = min(60.0, max(1.0, min(ttls) / 4.0))
        while True:
            await asyncio.sleep(interval)
            await self.evict_expired()

    async def evict_expired(self) -> list[str]:
        evicted: list[str] = []
        async with self._lock:
            now = self._clock()
            if self.idle_ttl_seconds > 0:
                expired = sorted(
                    (
                        record
                        for record in self._records.values()
                        if not record.pinned
                        and not record.primary
                        and record.active_requests == 0
                        and now - record.last_used_at >= self.idle_ttl_seconds
                        and _engine_is_idle(record.entry.engine)
                    ),
                    key=lambda record: record.last_used_at,
                )
                for record in expired:
                    evicted.append(record.model_id)
                    await self._evict_locked(record, reason="idle_ttl")
            if self.audio_role_idle_ttl_seconds > 0:
                # Speech engines are transient by nature: a dictation burst is
                # followed by minutes of typing. Their TTL is separate from the
                # model TTL (and much shorter by default) so a role that will
                # not be used again stops holding budget away from the
                # conversation model.
                stale = sorted(
                    (
                        role
                        for role in self._roles.values()
                        if role.state == "resident"
                        and role.releasable
                        and now - role.last_used_at >= self.audio_role_idle_ttl_seconds
                    ),
                    key=lambda role: role.last_used_at,
                )
                for role in stale:
                    if await self._release_role_locked(
                        role, reason="idle_ttl", require_idle=True
                    ):
                        evicted.append(role.role)
        return evicted

    def _idle_auxiliary_roles(self) -> list[AuxiliaryRoleRecord]:
        """Auxiliary roles that can be released to reclaim budget, LRU first."""

        return sorted(
            (
                role
                for role in self._roles.values()
                if role.state == "resident" and role.releasable
            ),
            key=lambda role: role.last_used_at,
        )

    async def _evict_for_locked(self, incoming_bytes: int, exclude: set[str]) -> None:
        if self.memory_limit_bytes <= 0:
            return
        while self._accounted_usage() + incoming_bytes > self.memory_limit_bytes:
            candidates = self._eviction_candidates_locked(exclude)
            if candidates:
                await self._evict_locked(candidates[0], reason="memory_pressure")
                continue
            # A transient speech engine must never outrank a model the user
            # explicitly asked for. Reclaim idle audio roles before declaring
            # the ceiling unreachable — they reload lazily on the next request.
            # ``require_idle`` re-checks after the await; a role that a request
            # claimed in the meantime is skipped rather than retried, so this
            # loop always makes progress toward the capacity error.
            reclaimed = False
            for role in self._idle_auxiliary_roles():
                if await self._release_role_locked(
                    role, reason="memory_pressure", require_idle=True
                ):
                    reclaimed = True
                    break
            if reclaimed:
                continue
            usage = self._accounted_usage()
            raise ResidentModelCapacityError(
                "resident model memory ceiling exceeded: "
                f"usage={usage / _GIB:.2f} GiB, "
                f"incoming={incoming_bytes / _GIB:.2f} GiB, "
                f"limit={self.memory_limit_bytes / _GIB:.2f} GiB; "
                "no idle unpinned model is eligible for eviction"
            )

    def _eviction_candidates_locked(self, exclude: set[str]) -> list[ResidencyRecord]:
        """Return the exact idle-LRU order used by admission and eviction."""

        return sorted(
            (
                record
                for record in self._records.values()
                if record.model_id not in exclude
                and not record.pinned
                and not record.primary
                and record.active_requests == 0
                and record.state == "resident"
                and _engine_is_idle(record.entry.engine)
            ),
            key=lambda record: record.last_used_at,
        )

    async def load(
        self,
        model_name: str,
        *,
        model_path: str | None = None,
        estimated_bytes: int | None = None,
        pin: bool = False,
        replace_group: str | None = None,
        image_mode: str | None = None,
        performance: ResidentPerformanceConfig | None = None,
        reload_if_changed: bool = False,
        replace_mode: str = "reject",
        memory_policy: str = "keep_then_commit",
        resolved_group: str | None = None,
    ) -> ResidencyRecord:
        model_name = model_name.strip()
        if not model_name:
            raise ResidentModelError("model must not be empty")
        estimate = max(1, estimated_bytes or estimate_model_bytes(model_name))
        if memory_policy not in {"keep_then_commit", "evict_first_if_needed"}:
            raise ResidentModelError(f"unsupported memory policy {memory_policy!r}")

        async with self._lock:
            canonical = self._canonical(model_name)
            if canonical is not None:
                existing_record = self._records[canonical]
                group = _effective_replace_group(existing_record.entry, replace_group)
                did_reload = False
                if reload_if_changed and existing_record.performance != performance:
                    reload_candidates: list[ResidencyRecord] = []
                    reload_paused_engines: list[object] = []
                    try:
                        if group is not None:
                            (
                                group_records,
                                reload_paused_engines,
                            ) = await self._quiesce_replacement_group_locked(
                                group, replace_mode
                            )
                            reload_candidates = [
                                candidate
                                for candidate in group_records
                                if candidate is not existing_record
                            ]
                        else:
                            reload_paused_engines = await self._quiesce_records_locked(
                                [existing_record], replace_mode
                            )
                        existing_record = await self._reload_locked(
                            existing_record, performance
                        )
                        did_reload = True
                        if group is not None:
                            await self._commit_group_replacement_locked(
                                existing_record, group, reload_candidates
                            )
                    except BaseException:
                        await self._resume_engines(reload_paused_engines)
                        raise
                existing_record.last_used_at = self._clock()
                if pin:
                    existing_record.pinned = True
                if group is not None and not did_reload:
                    await self._replace_group_locked(
                        existing_record, group, replace_mode
                    )
                return existing_record

            record: ResidencyRecord | None = None
            candidates: list[ResidencyRecord] = []
            paused_engines: list[object] = []
            destructive_handoff: PrimaryHandoffLease | None = None
            destructive_primary = False
            destructive_primary_publish_attempted = False
            destructive_replacement = False
            projection: ReplacementProjection | None = None
            try:
                if replace_group is not None:
                    if resolved_group is not None and resolved_group != replace_group:
                        raise ResidentModelError(
                            f"model {model_name!r} belongs to replacement group "
                            f"{resolved_group!r}, not {replace_group!r}"
                        )
                    if replace_mode == "reject":
                        # Preserve the established busy-before-capacity
                        # contract without evicting anything: the zero-timeout
                        # pause closes admission while the projection is read.
                        (
                            candidates,
                            paused_engines,
                        ) = await self._quiesce_replacement_group_locked(
                            replace_group, replace_mode
                        )
                    else:
                        candidates = self._replacement_candidates_locked(
                            replace_group,
                            replace_mode=replace_mode,
                        )
                    projection = self._replacement_projection_locked(
                        estimate,
                        candidates,
                        (
                            memory_policy
                            if resolved_group == replace_group
                            else "keep_then_commit"
                        ),
                    )
                    if projection.reason == "role_capacity_insufficient_after_eviction":
                        raise ResidentModelCapacityError(
                            "resident model memory ceiling exceeded after projected "
                            "assistant replacement",
                            replacement_projection=projection,
                        )
                    evict_first = projection.strategy == "evict_first"
                    if evict_first:
                        if replace_mode != "reject":
                            (
                                candidates,
                                paused_engines,
                            ) = await self._quiesce_replacement_group_locked(
                                replace_group, replace_mode
                            )
                        (
                            destructive_handoff,
                            destructive_primary,
                        ) = await self._evict_replacement_before_load_locked(
                            candidates,
                            replace_group,
                        )
                        destructive_replacement = True
                        paused_engines = []
                await self._evict_for_locked(
                    estimate,
                    exclude={model_name, *(item.model_id for item in candidates)},
                )
                before = self._read_memory()
                if image_mode is None:
                    entry = await self.loader(model_name, model_path, performance)
                else:
                    entry = await self.loader(
                        model_name, model_path, performance, image_mode
                    )
                now = self._clock()
                after = self._read_memory()
                delta = max(0, after - before) if before and after else 0
                record = ResidencyRecord(
                    entry=entry,
                    estimated_bytes=estimate,
                    measured_bytes=delta,
                    loaded_at=now,
                    last_used_at=now,
                    pinned=pin,
                    performance=performance,
                    replacement_projection=projection,
                )
                group = _effective_replace_group(record.entry, replace_group)
                if destructive_replacement:
                    record.primary = destructive_primary
                    if destructive_primary:
                        record.pinned = True
                elif replace_group is not None and replace_mode != "reject":
                    (
                        candidates,
                        paused_engines,
                    ) = await self._quiesce_replacement_group_locked(
                        replace_group,
                        replace_mode,
                    )
                elif group is not None and replace_group is None:
                    candidates, paused_engines = await self._quiesce_group_locked(
                        record, group, replace_mode
                    )
                # Keep the replacement private until the old inference engines
                # have reached the policy boundary. Publication only makes the
                # already-quiesced replacement visible to residency readers.
                self.registry.add(entry, is_default=record.primary)
                self._index_record(record)
                self.loads_total += 1
                if destructive_primary and self._on_primary_changed is not None:
                    destructive_primary_publish_attempted = True
                    self._on_primary_changed(entry)
                await self._evict_for_locked(
                    0,
                    exclude={
                        record.model_id,
                        *(item.model_id for item in candidates),
                    },
                )
                if destructive_handoff is not None:
                    destructive_handoff.commit(entry)
                    destructive_handoff = None
                if group is not None and not destructive_replacement:
                    await self._commit_group_replacement_locked(
                        record, group, candidates
                    )
            except _CommittedReplacementCancelled:
                # Routing already names the new target. Preserve that truth,
                # reopen any sibling engines not yet retired, and propagate
                # cancellation without treating the target as a failed load.
                await self._resume_engines(paused_engines)
                raise
            except BaseException:
                # Once the loader returns, this manager owns the engine.  A
                # later admission/replacement failure must not leave a model
                # resident even though the control-plane request was rejected.
                try:
                    if record is not None and record.model_id in self._records:
                        await self._evict_locked(
                            record, reason="load_rollback", count=False
                        )
                    elif record is not None:
                        stop = getattr(record.entry.engine, "stop", None)
                        if callable(stop):
                            result = stop()
                            if asyncio.iscoroutine(result):
                                await result
                        _release_allocator_cache()
                finally:
                    if (
                        destructive_primary_publish_attempted
                        and self._on_primary_changed is not None
                    ):
                        # Removing the rejected target may auto-promote an
                        # unrelated secondary. A failed primary publication
                        # has no valid default until a later load succeeds.
                        self.registry.clear_default()
                        try:
                            self._on_primary_changed(None)
                        except BaseException:
                            logger.exception(
                                "Failed to clear rejected replacement primary"
                            )
                    if destructive_handoff is not None:
                        destructive_handoff.commit(None)
                    if not destructive_replacement:
                        await self._resume_engines(paused_engines)
                raise
            return record

    def _replacement_projection_locked(
        self,
        incoming_bytes: int,
        candidates: list[ResidencyRecord],
        memory_policy: str,
    ) -> ReplacementProjection:
        """Choose rollback-safe or destructive admission before mutation."""

        measured = self._read_memory()
        reserved = sum(
            max(record.estimated_bytes, record.measured_bytes)
            for record in self._records.values()
            if record.state == "resident"
        )
        current = max(measured, reserved)
        limit = self.memory_limit_bytes
        replacement_ids = {record.model_id for record in candidates}
        idle_candidates = self._eviction_candidates_locked(replacement_ids)

        def payload(records: list[ResidencyRecord]) -> tuple[tuple[str, int], ...]:
            return tuple(
                (
                    record.model_id,
                    max(record.estimated_bytes, record.measured_bytes),
                )
                for record in records
            )

        def projected(records: list[ResidencyRecord]) -> int:
            released_measured = sum(record.measured_bytes for record in records)
            released_reserved = sum(
                max(record.estimated_bytes, record.measured_bytes) for record in records
            )
            remaining = max(
                max(0, measured - released_measured),
                max(0, reserved - released_reserved),
            )
            return remaining + incoming_bytes

        keep_projected = current + incoming_bytes
        evict_projected = projected(candidates)
        if limit <= 0 or keep_projected <= limit:
            return ReplacementProjection(
                strategy="keep_then_commit",
                reason="keep_both_fits",
                models_to_free=payload(candidates),
                current_bytes=current,
                requested_bytes=incoming_bytes,
                projected_bytes=evict_projected,
                limit_bytes=limit,
            )
        if memory_policy == "evict_first_if_needed":
            selected = list(candidates)
            for record in idle_candidates:
                if evict_projected <= limit:
                    break
                selected.append(record)
                evict_projected = projected(selected)
            if evict_projected <= limit:
                return ReplacementProjection(
                    strategy="evict_first",
                    reason="role_capacity_evict_first_required",
                    models_to_free=payload(selected),
                    current_bytes=current,
                    requested_bytes=incoming_bytes,
                    projected_bytes=evict_projected,
                    limit_bytes=limit,
                )
        else:
            selected_idle: list[ResidencyRecord] = []
            keep_after_lru = keep_projected
            for record in idle_candidates:
                if keep_after_lru <= limit:
                    break
                selected_idle.append(record)
                keep_after_lru = projected(selected_idle)
            if keep_after_lru <= limit:
                return ReplacementProjection(
                    strategy="keep_then_commit",
                    reason="keep_both_fits",
                    models_to_free=payload(selected_idle + candidates),
                    current_bytes=current,
                    requested_bytes=incoming_bytes,
                    projected_bytes=projected(selected_idle + candidates),
                    limit_bytes=limit,
                )
        all_releasable = candidates + idle_candidates
        return ReplacementProjection(
            strategy="reject",
            reason="role_capacity_insufficient_after_eviction",
            models_to_free=payload(all_releasable),
            current_bytes=current,
            requested_bytes=incoming_bytes,
            projected_bytes=projected(all_releasable),
            limit_bytes=limit,
        )

    async def _evict_replacement_before_load_locked(
        self,
        candidates: list[ResidencyRecord],
        group: str,
    ) -> tuple[PrimaryHandoffLease | None, bool]:
        """Destructively retire a quiesced group while holding audio ownership."""

        old_primary = next((record for record in candidates if record.primary), None)
        handoff = None
        if old_primary is not None and self._on_primary_handoff is not None:
            handoff = self._on_primary_handoff(old_primary.entry)
        destructive_started = False
        try:
            # Retire sibling assistants while the healthy primary remains
            # published. A sibling stop failure can therefore roll back the
            # handoff without stranding the server's default route.
            for record in candidates:
                if record is old_primary:
                    continue
                record.pinned = False
                await self._evict_locked(record, reason=f"replace_{group}_evict_first")
            if old_primary is not None:
                if self._on_primary_changed is not None:
                    self._on_primary_changed(None)
                self.registry.clear_default()
                destructive_started = True
                old_primary.primary = False
                old_primary.pinned = False
                await self._evict_locked(
                    old_primary,
                    reason=f"replace_{group}_evict_first",
                )
        except BaseException:
            if handoff is not None:
                if destructive_started:
                    handoff.commit(None)
                else:
                    handoff.rollback()
            raise
        return handoff, old_primary is not None

    async def _reload_locked(
        self,
        record: ResidencyRecord,
        performance: ResidentPerformanceConfig | None,
    ) -> ResidencyRecord:
        """Replace one idle engine without restarting or disturbing siblings."""

        if record.active_requests or not _engine_is_idle(record.entry.engine):
            raise ResidentModelBusyError("model is serving an active request")

        model_name = record.model_id
        model_path = record.entry.model_path
        estimate = record.estimated_bytes
        pinned = record.pinned
        primary = record.primary
        handoff = (
            self._on_primary_handoff(record.entry)
            if primary and self._on_primary_handoff is not None
            else None
        )

        if primary and self._on_primary_changed is not None:
            try:
                self._on_primary_changed(None)
            except BaseException:
                # The old engine is still intact and registered at this point.
                # Restore its serving-layer publication before releasing the
                # handoff so a callback failure cannot strand the transaction.
                try:
                    self._on_primary_changed(record.entry)
                finally:
                    if handoff is not None:
                        handoff.rollback()
                raise
        self.registry.remove(model_name)
        self._drop_record(model_name)
        if primary:
            # Removing the registry default otherwise promotes an unrelated
            # secondary while the replacement loader is still in flight.  Make
            # primary unavailability visible through both routing surfaces
            # before the first await; publication below restores them together.
            self.registry.clear_default()
        try:
            stop = getattr(record.entry.engine, "stop", None)
            if callable(stop):
                result = stop()
                if asyncio.iscoroutine(result):
                    await result
        except BaseException as stop_error:
            # A stop attempt may already have disabled part of the engine.
            # Rebuild the last known-good configuration instead of routing a
            # possibly half-stopped worker after rollback.
            await self._restore_reload_locked(record, handoff)
            raise stop_error
        _release_allocator_cache()

        before = self._read_memory()
        try:
            entry = _carry_served_identity(
                await self.loader(model_name, model_path, performance),
                record.entry,
            )
        except BaseException as reload_error:
            # The old engine has already released its Metal allocations so the
            # replacement can fit under the same budget. Best-effort restore
            # the last known-good config; never let a rejected Settings change
            # silently take every route for the primary model down.
            await self._restore_reload_locked(record, handoff)
            raise reload_error
        after = self._read_memory()
        now = self._clock()
        replacement = ResidencyRecord(
            entry=entry,
            estimated_bytes=estimate,
            measured_bytes=max(0, after - before) if before and after else 0,
            loaded_at=now,
            last_used_at=now,
            pinned=pinned,
            primary=primary,
            performance=performance,
        )
        try:
            self.registry.add(entry, is_default=primary)
            self._index_record(replacement)
            self.loads_total += 1
            if primary and self._on_primary_changed is not None:
                self._on_primary_changed(entry)
        except BaseException as publish_error:
            # The replacement is not committed until every serving-layer
            # publisher accepts it. Remove and stop it while the audio lease
            # still gates requests, then restore the last known-good config.
            self.registry.remove(model_name)
            self._drop_record(model_name)
            try:
                stop = getattr(entry.engine, "stop", None)
                if callable(stop):
                    result = stop()
                    if asyncio.iscoroutine(result):
                        await result
            except BaseException:
                logger.exception(
                    "Failed to stop rejected resident model %r",
                    model_name,
                )
            _release_allocator_cache()
            await self._restore_reload_locked(record, handoff)
            raise publish_error
        if handoff is not None:
            handoff.commit(entry)
        return replacement

    async def _restore_reload_locked(
        self,
        record: ResidencyRecord,
        handoff: PrimaryHandoffLease | None,
    ) -> None:
        """Restore the prior reload config and always finalize its handoff."""

        restored_entry = None
        try:
            restored_entry = _carry_served_identity(
                await self.loader(
                    record.model_id,
                    record.entry.model_path,
                    record.performance,
                ),
                record.entry,
            )
            restored = ResidencyRecord(
                entry=restored_entry,
                estimated_bytes=record.estimated_bytes,
                loaded_at=self._clock(),
                last_used_at=self._clock(),
                pinned=record.pinned,
                primary=record.primary,
                performance=record.performance,
            )
            self.registry.add(restored_entry, is_default=record.primary)
            self._index_record(restored)
            if record.primary and self._on_primary_changed is not None:
                self._on_primary_changed(restored_entry)
        except BaseException:
            logger.exception(
                "Failed to restore resident model %r after reload failure",
                record.model_id,
            )
            # A failed rebuild cannot remain partially published.  Remove any
            # entry that made it through the registry before a later publisher
            # failed, then clear every default/legacy owner for a lost primary.
            if restored_entry is not None:
                self.registry.remove(restored_entry.model_name)
                self._drop_record(restored_entry.model_name)
                try:
                    stop = getattr(restored_entry.engine, "stop", None)
                    if callable(stop):
                        result = stop()
                        if asyncio.iscoroutine(result):
                            await result
                except BaseException:
                    logger.exception(
                        "Failed to stop partially restored resident model %r",
                        record.model_id,
                    )
                _release_allocator_cache()
                if record.primary and self._on_primary_changed is not None:
                    try:
                        self._on_primary_changed(None)
                    except BaseException:
                        logger.exception(
                            "Failed to clear serving-layer primary after "
                            "restore publication failure"
                        )
                restored_entry = None
            if record.primary:
                self.registry.clear_default()
        finally:
            if handoff is not None:
                handoff.commit(restored_entry)

    async def _replace_group_locked(
        self,
        target: ResidencyRecord,
        group: str,
        replace_mode: str = "reject",
    ) -> None:
        """Make ``target`` the sole unpinned model in a lifecycle group.

        The desktop uses the ``assistant`` group for its chat picker: changing
        chat models replaces the previous text/VLM engine while independent
        image engines remain resident. A protected startup assistant hands its
        primary role to the replacement before the old engine is stopped, so
        legacy health/cache routes never retain a reference to unloaded weights.
        """

        candidates, paused_engines = await self._quiesce_group_locked(
            target, group, replace_mode
        )
        try:
            await self._commit_group_replacement_locked(target, group, candidates)
        except BaseException:
            await self._resume_engines(paused_engines)
            raise

    async def _quiesce_group_locked(
        self,
        target: ResidencyRecord,
        group: str,
        replace_mode: str,
    ) -> tuple[list[ResidencyRecord], list[object]]:
        """Atomically close admission and reach the requested policy boundary."""

        if group != _replacement_group(target.entry):
            raise ResidentModelError(
                f"model {target.model_id!r} does not belong to replacement group {group!r}"
            )
        return await self._quiesce_replacement_group_locked(
            group, replace_mode, exclude_model_id=target.model_id
        )

    async def _quiesce_replacement_group_locked(
        self,
        group: str,
        replace_mode: str,
        *,
        exclude_model_id: str | None = None,
    ) -> tuple[list[ResidencyRecord], list[object]]:
        """Close one group before any externally visible lifecycle mutation."""

        candidates = self._replacement_candidates_locked(
            group,
            exclude_model_id=exclude_model_id,
            replace_mode=replace_mode,
        )

        paused_engines = await self._quiesce_records_locked(candidates, replace_mode)
        return candidates, paused_engines

    async def _quiesce_records_locked(
        self,
        records: list[ResidencyRecord],
        replace_mode: str,
    ) -> list[object]:
        """Close admission and drain/abort exact records before mutation."""

        if replace_mode not in {"reject", "wait", "abort"}:
            raise ResidentModelError(f"unsupported replacement mode {replace_mode!r}")
        paused_engines: list[object] = []
        try:
            for record in records:
                engine = record.entry.engine
                pause = getattr(engine, "pause_generation", None)
                if record.active_requests and (
                    replace_mode == "reject" or not callable(pause)
                ):
                    raise ResidentModelBusyError("model is serving an active request")
                if callable(pause):
                    paused_engines.append(engine)
                    try:
                        await pause(
                            "wait" if replace_mode == "reject" else replace_mode,
                            timeout=0 if replace_mode == "reject" else None,
                        )
                    except TimeoutError as exc:
                        raise ResidentModelBusyError(
                            "model is serving an active request"
                        ) from exc
                elif not _engine_is_idle(engine):
                    raise ResidentModelBusyError("model is serving an active request")
                if record.active_requests:
                    await record.lease_idle.wait()
        except BaseException:
            await self._resume_engines(paused_engines)
            raise
        return paused_engines

    def _replacement_candidates_locked(
        self,
        group: str,
        *,
        exclude_model_id: str | None = None,
        replace_mode: str = "reject",
    ) -> list[ResidencyRecord]:
        """Validate and identify a group without changing engine admission."""

        if replace_mode not in {"reject", "wait", "abort"}:
            raise ResidentModelError(f"unsupported replacement mode {replace_mode!r}")

        candidates = [
            record
            for record in self._records.values()
            if record.model_id != exclude_model_id
            and _replacement_group(record.entry) == group
        ]
        for record in candidates:
            if record.pinned and not record.primary:
                raise ResidentModelError(
                    f"pinned model {record.model_id!r} cannot be replaced"
                )
        return candidates

    async def _resume_engines(self, engines: list[object]) -> None:
        """Best-effort reopen every engine paused by a failed transaction."""

        for engine in reversed(engines):
            resume = getattr(engine, "resume_generation", None)
            if callable(resume):
                try:
                    await resume()
                except BaseException:
                    logger.exception(
                        "Failed to resume a model engine after replacement rollback"
                    )

    async def _commit_group_replacement_locked(
        self,
        target: ResidencyRecord,
        group: str,
        candidates: list[ResidencyRecord],
    ) -> None:
        """Apply the existing primary/audio handoff to quiesced engines."""

        old_primary = next((record for record in candidates if record.primary), None)
        handoff = None
        if old_primary is not None:
            # Reserve serving-layer ownership before changing any primary
            # truth. The lease rejects active auxiliary work and prevents a
            # new request from entering until commit or rollback.
            if self._on_primary_handoff is not None:
                handoff = self._on_primary_handoff(old_primary.entry)
            old_pinned = old_primary.pinned
            target_primary = target.primary
            target_pinned = target.pinned

        try:
            if old_primary is not None:
                old_primary.primary = False
                old_primary.pinned = False
                target.primary = True
                target.pinned = True
                self.registry.set_default(target.model_id)
                if self._on_primary_changed is not None:
                    self._on_primary_changed(target.entry)
        except BaseException:
            if old_primary is not None:
                try:
                    old_primary.state = "resident"
                    old_primary.primary = True
                    old_primary.pinned = old_pinned
                    target.primary = target_primary
                    target.pinned = target_pinned
                    self.registry.add(old_primary.entry, is_default=True)
                    self._index_record(old_primary)
                    if self._on_primary_changed is not None:
                        self._on_primary_changed(old_primary.entry)
                finally:
                    if handoff is not None:
                        handoff.rollback()
            raise
        else:
            # Publishing the target is the point of no return. ``stop()`` may
            # mutate an engine incrementally before raising or being cancelled,
            # so no stop-attempted engine can truthfully be restored as primary.
            if handoff is not None:
                handoff.commit(target.entry)
            if old_primary is not None:
                try:
                    await self._evict_locked(
                        old_primary,
                        reason=f"replace_{group}",
                    )
                except asyncio.CancelledError as exc:
                    logger.warning(
                        "Primary retirement cancelled after routing commit: %r",
                        old_primary.model_id,
                    )
                    raise _CommittedReplacementCancelled from exc
                except Exception:
                    logger.exception(
                        "Failed to stop replaced primary %r after routing commit",
                        old_primary.model_id,
                    )
            # Finish retiring secondary candidates as non-failing cleanup: each
            # is already quiesced and removed from routing before stop(), so a
            # cleanup failure may leak resources but cannot resurrect a dead
            # route or undo the committed primary handoff.
            for record in candidates:
                if record is old_primary:
                    continue
                try:
                    await self._evict_locked(
                        record,
                        reason=f"replace_{group}",
                    )
                except asyncio.CancelledError as exc:
                    logger.warning(
                        "Replacement cleanup cancelled after routing commit: %r",
                        record.model_id,
                    )
                    raise _CommittedReplacementCancelled from exc
                except Exception:
                    logger.exception(
                        "Failed to stop replaced model %r after routing retirement",
                        record.model_id,
                    )

    async def set_pinned(self, model_name: str, pinned: bool) -> ResidencyRecord:
        async with self._lock:
            canonical = self._canonical(model_name)
            if canonical is None:
                raise KeyError(model_name)
            record = self._records[canonical]
            if record.primary and not pinned:
                raise ResidentModelError("the primary startup model cannot be unpinned")
            record.pinned = pinned
            record.last_used_at = self._clock()
            return record

    async def unload(self, model_name: str) -> None:
        async with self._lock:
            canonical = self._canonical(model_name)
            if canonical is None:
                raise KeyError(model_name)
            record = self._records[canonical]
            if record.pinned or record.primary:
                raise ResidentModelError("pinned models cannot be unloaded")
            if record.active_requests or not _engine_is_idle(record.entry.engine):
                raise ResidentModelBusyError("model is serving an active request")
            await self._evict_locked(record, reason="explicit")

    async def _evict_locked(
        self,
        record: ResidencyRecord,
        *,
        reason: str,
        count: bool = True,
    ) -> None:
        if record.active_requests:
            raise ResidentModelBusyError("model is serving an active request")
        record.state = "evicting"
        self.registry.remove(record.model_id)
        self._drop_record(record.model_id)
        stop = getattr(record.entry.engine, "stop", None)
        if callable(stop):
            result = stop()
            if asyncio.iscoroutine(result):
                await result
        _release_allocator_cache()
        if count:
            self.evictions_total += 1
        logger.info("Evicted resident model %r (%s)", record.model_id, reason)

    @asynccontextmanager
    async def lease(self, model_name: str):
        async with self._lock:
            canonical = self._canonical(model_name)
            if canonical is None:
                raise KeyError(model_name)
            record = self._records[canonical]
            if record.state != "resident":
                raise ResidentModelBusyError("model is being evicted")
            record.active_requests += 1
            record.lease_idle.clear()
            record.last_used_at = self._clock()
            engine = record.entry.engine
        try:
            yield engine
        finally:
            # Replacement may be holding the manager lock while it waits for
            # this lease to finish. These synchronous event-loop operations
            # are the release edge; no residency mutation can interleave.
            current = self._records.get(canonical)
            if current is record:
                current.active_requests = max(0, current.active_requests - 1)
                current.last_used_at = self._clock()
                if current.active_requests == 0:
                    current.lease_idle.set()

    # ------------------------------------------------------------------
    # Auxiliary audio roles (#2305)
    # ------------------------------------------------------------------

    def _role_conflicts(self, exclude_role: str) -> list[RoleConflict]:
        """Describe budget holders in stable conversation-first order."""

        conflicts: list[RoleConflict] = []
        for record in sorted(self._records.values(), key=lambda item: item.loaded_at):
            if record.state != "resident":
                continue
            busy = record.active_requests > 0 or not _engine_is_idle(
                record.entry.engine
            )
            if record.primary:
                reason = "active_conversation_model"
            elif busy:
                reason = "serving_active_request"
            elif record.pinned:
                reason = "pinned"
            else:
                reason = "resident"
            conflicts.append(
                RoleConflict(
                    role=_entry_role(record.entry),
                    model=record.model_id,
                    bytes=max(record.estimated_bytes, record.measured_bytes),
                    evictable=False,
                    reason=reason,
                )
            )
        for role in sorted(self._roles.values(), key=lambda item: item.last_used_at):
            if role.role == exclude_role or role.state not in {
                "admitting",
                "loading",
                "resident",
                "evicting",
            }:
                continue
            busy = role.active_requests > 0 or not role.releasable
            conflicts.append(
                RoleConflict(
                    role=role.role,
                    model=role.model_id,
                    bytes=role.charged_bytes,
                    evictable=not busy,
                    reason="serving_active_request" if busy else "idle",
                )
            )
        return conflicts

    async def _admit_role_locked(
        self,
        requested: RoleRef,
        exclude_role: str,
        *,
        request_bytes: int = 0,
    ) -> None:
        """Reclaim idle auxiliary roles or raise a typed conflict."""

        if self.memory_limit_bytes <= 0:
            return
        if requested.bytes <= 0:
            raise ResidentRoleConflictError(
                requested=requested,
                conflicts=self._role_conflicts(exclude_role),
                limit_bytes=self.memory_limit_bytes,
                usage_bytes=self._accounted_usage(),
                capacity_unknown=True,
            )
        weights_only = max(0, requested.bytes - max(0, int(request_bytes)))
        while self._accounted_usage() + requested.bytes > self.memory_limit_bytes:
            reclaimed = False
            for candidate in self._idle_auxiliary_roles():
                if candidate.role == exclude_role:
                    continue
                if await self._release_role_locked(
                    candidate, reason="role_admission", require_idle=True
                ):
                    reclaimed = True
                    break
            if not reclaimed:
                usage = self._accounted_usage()
                blamed_on_request = (
                    request_bytes > 0
                    and usage + weights_only <= self.memory_limit_bytes
                )
                reported = requested
                if blamed_on_request:
                    reported = RoleRef(
                        role=requested.role,
                        model=requested.model,
                        bytes=max(0, int(request_bytes)),
                    )
                raise ResidentRoleConflictError(
                    requested=reported,
                    conflicts=self._role_conflicts(exclude_role),
                    limit_bytes=self.memory_limit_bytes,
                    usage_bytes=usage,
                    request_buffer=blamed_on_request,
                    request_bytes=request_bytes if blamed_on_request else 0,
                )

    @asynccontextmanager
    async def admitting_role(
        self,
        *,
        role: str,
        lane: str,
        model_id: str,
        reserved_bytes: int,
        capacity_source: str,
        weight_bytes: int | None = None,
        pending_request_bytes: int = 0,
    ):
        """Reserve weights and pending request bytes around an async load."""

        async with self._lock:
            existing = self._roles.get(role)
            if existing is not None:
                if existing.state in {"admitting", "loading"}:
                    raise ResidentModelBusyError(f"{role} is already being admitted")
                await self._release_role_locked(existing, reason="role_replaced")
            requested = RoleRef(
                role=role,
                model=model_id,
                bytes=reserved_bytes + max(0, int(pending_request_bytes)),
            )
            await self._admit_role_locked(
                requested,
                exclude_role=role,
                request_bytes=max(0, int(pending_request_bytes)),
            )
            now = self._clock()
            record = AuxiliaryRoleRecord(
                role=role,
                lane=lane,
                model_id=model_id,
                reserved_bytes=max(0, int(reserved_bytes)),
                capacity_source=capacity_source,
                weight_bytes=weight_bytes,
                state="loading",
                loaded_at=now,
                last_used_at=now,
                pending_request_bytes=max(0, int(pending_request_bytes)),
            )
            self._roles[role] = record
            before = self._read_memory()

        try:
            yield record
        except BaseException:
            async with self._lock:
                if self._roles.get(role) is record:
                    await self._release_role_locked(record, reason="load_rollback")
            raise

        async with self._lock:
            if self._roles.get(role) is not record:
                return
            after = self._read_memory()
            record.measured_bytes = max(0, after - before) if before and after else 0
            record.state = "resident"
            record.loaded_at = self._clock()
            record.last_used_at = record.loaded_at
            self.loads_total += 1

            if (
                self.memory_limit_bytes > 0
                and self._accounted_usage() > self.memory_limit_bytes
            ):
                overshoot = RoleRef(
                    role=role, model=model_id, bytes=record.charged_bytes
                )
                usage = self._accounted_usage()
                await self._release_role_locked(record, reason="measured_overshoot")
                raise ResidentRoleConflictError(
                    requested=overshoot,
                    conflicts=self._role_conflicts(role),
                    limit_bytes=self.memory_limit_bytes,
                    usage_bytes=usage,
                    measured_overshoot=True,
                )

    async def _release_role_locked(
        self, record: AuxiliaryRoleRecord, *, reason: str, require_idle: bool = False
    ) -> bool:
        """Retire a role, optionally rechecking that it is still idle."""

        if require_idle and not record.releasable:
            return False
        if record.active_requests:
            raise ResidentModelBusyError(f"{record.role} is serving an active request")
        record.state = "evicting"
        unload = record.unload
        cancelled: asyncio.CancelledError | None = None
        if callable(unload):
            try:
                result = unload()
                if asyncio.iscoroutine(result):
                    await result
            except asyncio.CancelledError as exc:
                cancelled = exc
            except Exception:
                logger.exception(
                    "Failed to unload %s role %r", record.role, record.model_id
                )
        _release_allocator_cache()
        if self._roles.get(record.role) is record:
            del self._roles[record.role]
        self.evictions_total += 1
        logger.info("Released %s role %r (%s)", record.role, record.model_id, reason)
        if cancelled is not None:
            raise cancelled
        return True

    async def release_role(self, role: str) -> None:
        """Explicitly drop an auxiliary role, e.g. during lane shutdown."""

        async with self._lock:
            record = self._roles.get(role)
            if record is None:
                return
            await self._release_role_locked(record, reason="explicit")

    @asynccontextmanager
    async def claiming_request_bytes(self, role: str, request_bytes: int):
        """Release any admission charge not consumed by the matching lease."""

        charge = max(0, int(request_bytes))
        try:
            yield
        finally:
            if charge:
                async with self._lock:
                    record = self._roles.get(role)
                    if record is not None:
                        record.pending_request_bytes = max(
                            0, record.pending_request_bytes - charge
                        )

    @asynccontextmanager
    async def lease_role(self, role: str, *, request_bytes: int = 0):
        """Hold a role and its request working set against eviction."""

        async with self._lock:
            record = self._roles.get(role)
            if record is None:
                raise KeyError(role)
            if record.state != "resident":
                raise ResidentModelBusyError(f"{role} is not resident")
            charge = max(0, int(request_bytes))
            claimed = min(charge, record.pending_request_bytes)
            record.pending_request_bytes -= claimed
            outstanding = charge - claimed
            if outstanding and self.memory_limit_bytes > 0:
                if self._accounted_usage() + outstanding > self.memory_limit_bytes:
                    record.pending_request_bytes += claimed
                    raise ResidentRoleConflictError(
                        requested=RoleRef(
                            role=role, model=record.model_id, bytes=charge
                        ),
                        conflicts=self._role_conflicts(exclude_role=""),
                        limit_bytes=self.memory_limit_bytes,
                        usage_bytes=self._accounted_usage(),
                        request_buffer=True,
                        request_bytes=charge,
                    )
            record.active_requests += 1
            record.request_bytes += charge
            record.last_used_at = self._clock()
        try:
            yield record
        finally:
            async with self._lock:
                current = self._roles.get(role)
                if current is record:
                    current.active_requests = max(0, current.active_requests - 1)
                    current.request_bytes = max(0, current.request_bytes - charge)
                    current.last_used_at = self._clock()

    def role_snapshot(self) -> list[dict[str, object]]:
        """Budget-bearing view of the auxiliary roles, for the residency API."""

        now = self._clock()
        return [
            {
                "role": role.role,
                "lane": role.lane,
                "model": role.model_id,
                "state": role.state,
                "active_requests": role.active_requests,
                "reserved_bytes": role.reserved_bytes,
                "measured_bytes": role.measured_bytes or None,
                "request_bytes": role.request_bytes,
                "pending_request_bytes": role.pending_request_bytes,
                "charged_bytes": role.charged_bytes,
                "weight_bytes": role.weight_bytes,
                "capacity_source": role.capacity_source,
                "idle_seconds": (
                    max(0.0, now - role.last_used_at)
                    if role.active_requests == 0 and not role.pending_request_bytes
                    else 0.0
                ),
            }
            for role in sorted(self._roles.values(), key=lambda item: item.loaded_at)
        ]

    def snapshot(self) -> dict:
        now = self._clock()
        models = []
        for record in sorted(self._records.values(), key=lambda item: item.loaded_at):
            engine = record.entry.engine
            resident = not hasattr(engine, "is_resident") or bool(engine.is_resident)
            engine_active = _engine_active_requests(engine)
            lifecycle_status = getattr(engine, "lifecycle_status", None)
            lifecycle = lifecycle_status() if callable(lifecycle_status) else None
            active_requests = max(
                record.active_requests,
                engine_active if engine_active is not None else 0,
            )
            if lifecycle is not None:
                active_requests = max(
                    active_requests,
                    int(lifecycle.get("active_requests", 0) or 0),
                    int(lifecycle.get("admitted_requests", 0) or 0),
                    int(lifecycle.get("running_requests", 0) or 0),
                    int(lifecycle.get("queued_requests", 0) or 0),
                )
            models.append(
                {
                    "id": record.model_id,
                    "model_path": record.entry.model_path,
                    "aliases": sorted(record.entry.aliases),
                    "modality": (_modality(record.entry)),
                    "role": _entry_role(record.entry),
                    "serving_lane": getattr(engine, "serving_lane", None),
                    "serving_lane_reason": getattr(engine, "serving_lane_reason", None),
                    "state": record.state if resident else "registered",
                    "pinned": record.pinned,
                    "primary": record.primary,
                    # A manager lease and a scheduler request describe
                    # overlapping lifetimes for dynamic engines, so use the
                    # larger count rather than double-counting. Primary traffic
                    # has no manager lease and is supplied by the scheduler.
                    "active_requests": active_requests,
                    "lifecycle": lifecycle,
                    "estimated_bytes": record.estimated_bytes,
                    "measured_bytes": record.measured_bytes or None,
                    "idle_seconds": max(0.0, now - record.last_used_at),
                    "performance": (
                        record.performance.payload() if record.performance else None
                    ),
                    "replacement_projection": (
                        record.replacement_projection.payload()
                        if record.replacement_projection
                        else None
                    ),
                }
            )
        usage = self._accounted_usage()
        return {
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_used_bytes": usage,
            "memory_available_bytes": (
                max(0, self.memory_limit_bytes - usage)
                if self.memory_limit_bytes > 0
                else None
            ),
            "idle_ttl_seconds": self.idle_ttl_seconds,
            "audio_role_idle_ttl_seconds": self.audio_role_idle_ttl_seconds,
            "loads_total": self.loads_total,
            "evictions_total": self.evictions_total,
            "models": models,
            "roles": self.role_snapshot(),
        }
