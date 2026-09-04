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
from typing import Protocol, TypedDict

from .model_registry import ModelEntry, ModelRegistry
from .process_memory import get_phys_footprint

logger = logging.getLogger(__name__)

_GIB = 1024**3
_PARAM_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b(?![a-z])", re.IGNORECASE)
_QUANT_RE = re.compile(r"(?<!\d)(2|3|4|6|8|16)[-]?bit", re.IGNORECASE)


class ResidentModelError(RuntimeError):
    """Base class for resident-model control-plane failures."""


class ResidentModelCapacityError(ResidentModelError):
    """The configured ceiling cannot admit a model/role after eligible eviction.

    Two admission shapes share this error while keeping their wire contracts
    distinct:

    * **Assistant/replacement admission** keeps the legacy ``message`` +
      ``replacement_projection`` shape so ``/v1/models/load`` can return its
      established projection-based 507 envelope unchanged.
    * **Protected-role admission** (``residency.admit_role``) carries the
      typed role fields below, surfaced via :meth:`envelope` as the stable
      ``insufficient_capacity_error`` 507 for lanes like forced alignment.
    """

    def __init__(
        self,
        message: str,
        *,
        replacement_projection: ReplacementProjection | None = None,
        reason: str | None = None,
        requested_bytes: int | None = None,
        limit_bytes: int | None = None,
        used_bytes: int | None = None,
        requested_role: str | None = None,
    ) -> None:
        self.replacement_projection = replacement_projection
        self.reason = reason
        self.requested_bytes = requested_bytes
        self.limit_bytes = limit_bytes
        self.used_bytes = used_bytes
        self.requested_role = requested_role
        super().__init__(message)

    def envelope(self) -> dict[str, object]:
        """Return the stable machine-readable 507 role-capacity contract.

        Populated only when the error was raised from ``admit_role`` (role
        fields present). A legacy replacement/assistant capacity error has no
        typed role envelope — its caller keeps using ``replacement_projection``.
        """
        if self.reason is None:
            return {
                "error": {
                    "message": str(self),
                    "type": "insufficient_capacity_error",
                    "code": "insufficient_capacity_error",
                    "param": "model",
                }
            }
        return {
            "error": {
                "message": str(self),
                "type": "insufficient_capacity_error",
                "code": "insufficient_capacity_error",
                "reason": self.reason,
                "param": "model",
                "requested_bytes": self.requested_bytes,
                "limit_bytes": self.limit_bytes,
                "used_bytes": self.used_bytes,
            }
        }


class ResidentModelBusyError(ResidentModelError):
    """A model cannot be removed while it owns active work."""


class _CommittedReplacementCancelled(asyncio.CancelledError):
    """Cancellation observed after replacement routing became authoritative."""


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
    # A control-plane dtype is operator-explicit: an unsupported family
    # raises KVCacheQuantizationUnsupportedError before any weights load
    # (mapped to 422 by the residency route, #78).
    decision = resolve_kv_cache_dtype(
        performance.kv_cache_dtype,
        explicit=True,
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


@dataclass
class ResidentRoleReservation:
    """A non-registry role charged to the process residency ceiling.

    Auxiliary lanes (forced alignment today; dictation speech-input under the
    companion role work) own their MLX engine outside the model registry, so
    their footprints cannot be ledgers through ``ResidencyRecord``. This
    reservation charges them to the same shared ceiling so admission can
    reject before their weights load.
    """

    role: str
    model_id: str
    reserved_bytes: int
    capacity_source: str
    state: str
    loaded_at: float


@dataclass
class ResidentRoleAdmission:
    """Transaction handle for replacing one auxiliary role reservation."""

    record: ResidentRoleReservation
    previous: ResidentRoleReservation | None = None
    previous_retired: bool = False
    exclusive_retired: bool = False
    committed: bool = False

    def retire_previous(self) -> None:
        """Signal that the prior engine was dropped; do not restore on rollback."""
        self.previous_retired = True

    def retire_exclusive(self) -> None:
        """Signal that the mutually-exclusive sibling's engine was discarded.

        Called by the alignment lane the INSTANT the ASR engine evicted by
        ``_evict_other_lane_sync("aligner")`` is actually gone. After this
        point the retained ``speech-input`` reservation points at a phantom
        engine, so a rollback must DROP it rather than leave a charge for an
        engine that no longer exists. Before the discard the sibling is still
        its engine's true reservation and stays in the ledger.
        """
        self.exclusive_retired = True

    def commit(self) -> None:
        """Mark the new engine as resident even when the caller re-raises.

        Used by the alignment lane when its weight load completed on the model
        worker (the engine was published) but the async handler is being
        cancelled: the weights are loaded, so the reservation must be kept
        rather than rolled back into a ledger desync.
        """
        self.committed = True
        self.record.state = "resident"


Loader = Callable[..., Awaitable[ModelEntry]]
PrimaryChanged = Callable[[ModelEntry | None], None]


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


class _SnapshotModelDict(TypedDict):
    """Per-model row of :meth:`ResidentModelManager.snapshot`.

    Typing the heterogeneous snapshot rows lets mypy check the ``max`` on
    ``estimated_bytes``/``measured_bytes`` as real ``int`` comparisons instead
    of joining every value type into one broad dict union.
    """

    id: str
    model_path: str
    aliases: list[str]
    modality: str
    role: str
    serving_lane: object
    serving_lane_reason: object
    state: str
    pinned: bool
    primary: bool
    active_requests: int
    lifecycle: object
    estimated_bytes: int
    measured_bytes: int | None
    idle_seconds: float
    performance: dict[str, object] | None
    replacement_projection: dict[str, object] | None


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
        clock: Callable[[], float] = time.monotonic,
        memory_reader: Callable[[], int] = get_phys_footprint,
        on_primary_handoff: PrimaryHandoff | None = None,
        on_primary_changed: PrimaryChanged | None = None,
    ) -> None:
        self.registry = registry
        self.loader = loader
        self.memory_limit_bytes = max(0, int(memory_limit_bytes))
        self.idle_ttl_seconds = max(0.0, float(idle_ttl_seconds))
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
        self._roles: dict[str, ResidentRoleReservation] = {}
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
        reserved = sum(
            max(record.estimated_bytes, record.measured_bytes)
            for record in self._records.values()
            if record.state == "resident"
        )
        reserved += sum(
            record.reserved_bytes
            for record in self._roles.values()
            if record.state in {"loading", "resident"}
        )
        # Some engines (notably mflux) construct lazy MLX arrays without
        # faulting all weight pages into the process. The footprint delta at
        # load time can therefore be much smaller than the memory the first
        # request will materialize. Keep the catalog/heuristic reservation in
        # force until the actual process footprint grows past it.
        return max(measured, reserved)

    def contains(self, model_name: str) -> bool:
        return self._canonical(model_name) is not None

    def touch(self, model_name: str | None) -> None:
        canonical = self._canonical(model_name)
        if canonical and canonical in self._records:
            self._records[canonical].last_used_at = self._clock()

    async def start(self) -> None:
        if self.idle_ttl_seconds <= 0 or self._ttl_task is not None:
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
            dynamic = [
                record for record in self._records.values() if not record.primary
            ]
            for record in dynamic:
                await self._evict_locked(record, reason="shutdown", count=False)

    async def _ttl_loop(self) -> None:
        interval = min(60.0, max(1.0, self.idle_ttl_seconds / 4.0))
        while True:
            await asyncio.sleep(interval)
            await self.evict_expired()

    async def evict_expired(self) -> list[str]:
        if self.idle_ttl_seconds <= 0:
            return []
        async with self._lock:
            now = self._clock()
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
            evicted = []
            for record in expired:
                evicted.append(record.model_id)
                await self._evict_locked(record, reason="idle_ttl")
            return evicted

    async def _evict_for_locked(
        self,
        incoming_bytes: int,
        exclude: set[str],
        *,
        requested_role: str | None = None,
        usage_credit_bytes: int = 0,
    ) -> None:
        if self.memory_limit_bytes <= 0:
            return
        while (
            max(0, self._accounted_usage() - usage_credit_bytes) + incoming_bytes
            > self.memory_limit_bytes
        ):
            candidates = self._eviction_candidates_locked(exclude)
            if not candidates:
                usage = max(0, self._accounted_usage() - usage_credit_bytes)
                if requested_role is None:
                    raise ResidentModelCapacityError(
                        "resident model memory ceiling exceeded: "
                        f"usage={usage / _GIB:.2f} GiB, "
                        f"incoming={incoming_bytes / _GIB:.2f} GiB, "
                        f"limit={self.memory_limit_bytes / _GIB:.2f} GiB; "
                        "no idle unpinned model is eligible for eviction"
                    )
                raise ResidentModelCapacityError(
                    (
                        "insufficient capacity for role "
                        f"{requested_role!r}: requested="
                        f"{incoming_bytes / _GIB:.2f} GiB, "
                        f"used={usage / _GIB:.2f} GiB, "
                        f"limit={self.memory_limit_bytes / _GIB:.2f} GiB; "
                        "no idle unpinned model is eligible for eviction"
                    ),
                    reason=f"role_capacity_{requested_role.replace('-', '_')}",
                    requested_bytes=incoming_bytes,
                    limit_bytes=self.memory_limit_bytes,
                    used_bytes=usage,
                    requested_role=requested_role,
                )
            await self._evict_locked(candidates[0], reason="memory_pressure")

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

    @asynccontextmanager
    async def admit_role(
        self,
        *,
        role: str,
        model_id: str,
        requested_bytes: int | None,
        capacity_source: str,
        replace_existing: bool = False,
        release_exclusive_role: str | None = None,
    ):
        """Reserve a protected auxiliary role before its weights load.

        Mirrors the shared ledger's transaction discipline: admission is
        decided with the footprint known (``requested_bytes`` from catalog or
        local-cache metadata), and the reservation is committed only on
        success or rolled back on failure/cancellation so no leaked
        reservation desyncs the ledger.

        ``replace_existing=True`` charges the new footprint against the old
        role's reservation (crediting the previous bytes) and, on rollback,
        restores the previous reservation unless the caller retired it via
        :meth:`ResidentRoleAdmission.retire_previous`.

        ``release_exclusive_role`` adds a SECOND mutually-exclusive auxiliary
        role to the same transaction (e.g. alignment retiring the dictation
        ``speech-input`` role whose ASR engine the aligner evicts). The
        sibling's reservation is RETAINED in ``self._roles`` — the real
        auxiliary ledger, never the ``snapshot()`` projection — for the whole
        transaction and its bytes are CREDITED against the new role's admission,
        so the two are never double-charged (no false 507) yet no concurrent
        admission can consume the sibling's capacity (no steal window). The new
        role's admission and the sibling's retention happen under the SAME
        ``self._lock`` critical section. On success (commit) — including a load
        that finished under cancellation — the sibling is retired when the
        aligner evicts its engine. On failure/cancellation the sibling stays in
        the ledger (matching its still-resident engine) unless the load evicted
        it first, in which case the caller's
        :meth:`ResidentRoleAdmission.retire_exclusive` signal drops the now
        phantom reservation. A still-resident engine is never left unaccounted,
        and the configured ceiling is never breached by a restore.
        """

        async with self._lock:
            previous = self._roles.get(role)
            # Capture the mutually-exclusive sibling WITHOUT removing it. The
            # sibling's reservation is RETAINED in ``_roles`` for the whole
            # transaction — its bytes stay accounted, so NO concurrent admission
            # can consume them and a rollback can never be forced to choose
            # between an unaccounted engine and an over-ceiling ledger (pr_validate
            # round-19). ``self._roles`` is the authoritative auxiliary-role
            # ledger (in contrast to ``snapshot()["roles"]``, which mixes in
            # synthesized projection entries for ordinary models and would let a
            # caller release the wrong thing / a non-reservation).
            exclusive_sibling = None
            if release_exclusive_role is not None and release_exclusive_role != role:
                exclusive_sibling = self._roles.get(release_exclusive_role)
            try:
                if previous is not None and not replace_existing:
                    raise ResidentModelError(f"role {role!r} is already resident")
                # A role must never host two concurrent in-flight LOADS: an
                # overwrite here would orphan the earlier load's reservation and,
                # if that earlier load later rolls back, could resurrect a stale
                # ``"loading"`` record. Callers serialise per-lane (the STT lane
                # lock), so this is a defensive invariant at the ledger layer
                # itself — reject rather than corrupt.
                if previous is not None and previous.state == "loading":
                    raise ResidentModelError(
                        f"role {role!r} already has a loading admission in flight"
                    )
                # Credit the new role against BOTH the same-role previous and the
                # retained mutually-exclusive sibling: the aligner admission nets
                # out the ASR engine bytes it will evict, so it is not falsely
                # rejected while the ASR engine is still resident.
                usage_credit = (
                    previous.reserved_bytes if previous is not None else 0
                ) + (
                    exclusive_sibling.reserved_bytes
                    if exclusive_sibling is not None
                    else 0
                )
                used = max(0, self._accounted_usage() - usage_credit)
                if self.memory_limit_bytes > 0 and requested_bytes is None:
                    raise ResidentModelCapacityError(
                        (
                            f"insufficient capacity for role {role!r}: the requested "
                            "model has no catalog or local-cache size metadata and "
                            f"limit={self.memory_limit_bytes / _GIB:.2f} GiB; "
                            "refusing blind admission under a configured ceiling"
                        ),
                        reason="role_capacity_unknown",
                        requested_bytes=None,
                        limit_bytes=self.memory_limit_bytes,
                        used_bytes=used,
                        requested_role=role,
                    )
                reserved_bytes = max(0, int(requested_bytes or 0))
                await self._evict_for_locked(
                    reserved_bytes,
                    exclude=set(),
                    requested_role=role,
                    usage_credit_bytes=usage_credit,
                )
                record = ResidentRoleReservation(
                    role=role,
                    model_id=model_id,
                    reserved_bytes=reserved_bytes,
                    capacity_source=capacity_source,
                    state="loading",
                    loaded_at=self._clock(),
                )
                self._roles[role] = record
                admission = ResidentRoleAdmission(record=record, previous=previous)
            except BaseException:
                # Any failure BEFORE admission is decided (capacity error,
                # invariant conflict) leaves the retained sibling untouched in
                # ``_roles`` — nothing to restore, nothing leaked.
                raise
        try:
            yield admission
        except BaseException:
            if admission.committed:
                # The engine was already published on the model worker (e.g.
                # the load finished under cancellation); keep it accounted. The
                # successful load evicted the ASR engine -> drop the retained
                # (by now phantom) sibling reservation.
                async with self._lock:
                    if (
                        exclusive_sibling is not None
                        and release_exclusive_role is not None
                        and self._roles.get(release_exclusive_role) is exclusive_sibling
                    ):
                        # Only retire the sibling WE retained — preserve any
                        # NEWER reservation a concurrent admission installed for
                        # this role while the aligner loaded (its engine is
                        # authoritative and still resident).
                        self._roles.pop(release_exclusive_role, None)
                raise
            async with self._lock:
                if self._roles.get(role) is record:
                    if previous is not None and not admission.previous_retired:
                        self._roles[role] = previous
                    else:
                        self._roles.pop(role, None)
                # The retained sibling was never removed, so it needs no
                # "restore". If the load evicted its engine before failing
                # (``retire_exclusive`` -> ``exclusive_retired`` — the round-18
                # case), the reservation now guards a phantom engine and must be
                # dropped; otherwise it stays charged, exactly matching its
                # still-resident engine. Identity-check so a concurrently
                # installed NEWER reservation for the role is never erased.
                if (
                    exclusive_sibling is not None
                    and release_exclusive_role is not None
                    and admission.exclusive_retired
                    and self._roles.get(release_exclusive_role) is exclusive_sibling
                ):
                    self._roles.pop(release_exclusive_role, None)
            raise
        else:
            # Success finalization runs under a cancellation SHIELD (pr_validate
            # round-22): after the route's ``admission.commit()`` the engine is
            # already resident, so a cancellation arriving while the success-path
            # ``await self._lock`` is blocked must NOT skip the finalization —
            # that would leave the committed engine's reservation stuck in
            # ``"loading"`` and the evicted sibling (if any) charged forever.
            # The shield lets the finalize run to completion; if the surrounding
            # task is cancelled, we wait for the shielded finalize first and
            # THEN re-raise, never leaking a phantom sibling charge.
            finalize = asyncio.ensure_future(
                self._finalize_role_commit(
                    role=role,
                    record=record,
                    exclusive_sibling=exclusive_sibling,
                    release_exclusive_role=release_exclusive_role,
                )
            )
            try:
                await asyncio.shield(finalize)
            except asyncio.CancelledError:
                # Un-cancellable drain: the shielded ``finalize`` task must run
                # to completion BEFORE we propagate cancellation (pr_validate
                # round-23). A second cancellation while this handler awaits is
                # itself re-raised as ``CancelledError`` inside the loop, which
                # we swallow and re-arm the shield until ``finalize.done()`` —
                # so callers can never observe a stale ``"loading"`` record or
                # a retained phantom sibling charge after cancelling here.
                while not finalize.done():
                    try:
                        await asyncio.shield(finalize)
                    except asyncio.CancelledError:
                        continue
                raise

    async def _finalize_role_commit(
        self,
        *,
        role: str,
        record: ResidentRoleReservation,
        exclusive_sibling,
        release_exclusive_role: str | None,
    ) -> None:
        """Complete the success-path role commit under a shielded lock.

        Marks the admitted record resident and retires the retained
        mutually-exclusive sibling (the aligner evicted its engine). Runs inside
        :meth:`admit_role`'s success finalization, shielded from cancellation so
        a committed engine is never left ``"loading"`` and a phantom sibling
        charge is never leaked when the context is cancelled mid-commit.
        """
        async with self._lock:
            if self._roles.get(role) is record:
                record.state = "resident"
                record.loaded_at = self._clock()
            # Success: the aligner loaded and evicted the ASR engine ->
            # retire the retained sibling reservation, but ONLY if it is still
            # the reservation WE retained — a concurrent admission may have
            # replaced the role with a newer (authoritative) reservation while
            # the aligner loaded, and must not be erased here.
            if (
                exclusive_sibling is not None
                and release_exclusive_role is not None
                and self._roles.get(release_exclusive_role) is exclusive_sibling
            ):
                self._roles.pop(release_exclusive_role, None)

    async def release_role(self, role: str) -> None:
        """Stop charging a role after its owning lane released the engine."""

        async with self._lock:
            self._roles.pop(role, None)

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
                    elif paused_engines:
                        await self._resume_engines(paused_engines)
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
                elif replace_group is not None:
                    if replace_mode != "reject":
                        (
                            candidates,
                            paused_engines,
                        ) = await self._quiesce_replacement_group_locked(
                            replace_group,
                            replace_mode,
                        )
                    else:
                        paused_engines = await self._quiesce_records_locked(
                            candidates,
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

    def snapshot(self) -> dict:
        now = self._clock()
        models: list[_SnapshotModelDict] = []
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
                    "role": _replacement_group(record.entry),
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
        roles = [
            {
                "role": model["role"],
                "model": model["id"],
                "state": model["state"],
                "pinned": model["pinned"],
                "active_requests": model["active_requests"],
                "reserved_bytes": max(
                    model["estimated_bytes"], model["measured_bytes"] or 0
                ),
                "capacity_source": "model",
            }
            for model in models
        ]
        roles.extend(
            {
                "role": record.role,
                "model": record.model_id,
                "state": record.state,
                "pinned": True,
                "active_requests": 0,
                "reserved_bytes": record.reserved_bytes,
                "capacity_source": record.capacity_source,
            }
            for record in sorted(self._roles.values(), key=lambda item: item.role)
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
            "loads_total": self.loads_total,
            "evictions_total": self.evictions_total,
            "models": models,
            "roles": roles,
        }
