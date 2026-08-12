"""Budgeted lifecycle management for models resident in one server process."""

from __future__ import annotations

import asyncio
import gc
import logging
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

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


class ResidentModelBusyError(ResidentModelError):
    """A model cannot be removed while it owns active work."""


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

    @property
    def model_id(self) -> str:
        return self.entry.model_name


Loader = Callable[..., Awaitable[ModelEntry]]
PrimaryChanged = Callable[[ModelEntry], None]


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
    }
    for token, gib in known_image_gib.items():
        if token in folded:
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


def _engine_is_idle(engine: object) -> bool:
    """Best-effort idle check shared by explicit, LRU, and TTL eviction."""

    progress = getattr(engine, "progress_snapshot", None)
    if callable(progress):
        try:
            if bool(progress().get("running", False)):
                return False
        except Exception:
            return False

    get_stats = getattr(engine, "get_stats", None)
    if callable(get_stats):
        try:
            stats = get_stats() or {}
            if int(stats.get("num_running", 0) or 0) > 0:
                return False
            if int(stats.get("num_waiting", 0) or 0) > 0:
                return False
        except Exception:
            return False
    return True


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
        clock: Callable[[], float] = time.monotonic,
        memory_reader: Callable[[], int] = get_phys_footprint,
        on_primary_changed: PrimaryChanged | None = None,
    ) -> None:
        self.registry = registry
        self.loader = loader
        self.memory_limit_bytes = max(0, int(memory_limit_bytes))
        self.idle_ttl_seconds = max(0.0, float(idle_ttl_seconds))
        self._clock = clock
        self._memory_reader = memory_reader
        self._on_primary_changed = on_primary_changed
        self._records: dict[str, ResidencyRecord] = {}
        self._index: dict[str, str] = {}
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
            # Process footprint is exposed separately as the authoritative
            # total. Charging it to the primary row would make that one model
            # appear to own Python, uvicorn, Metal caches, and every secondary.
            measured_bytes=0,
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

    async def _evict_for_locked(self, incoming_bytes: int, exclude: set[str]) -> None:
        if self.memory_limit_bytes <= 0:
            return
        while self._accounted_usage() + incoming_bytes > self.memory_limit_bytes:
            candidates = sorted(
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
            if not candidates:
                usage = self._accounted_usage()
                raise ResidentModelCapacityError(
                    "resident model memory ceiling exceeded: "
                    f"usage={usage / _GIB:.2f} GiB, "
                    f"incoming={incoming_bytes / _GIB:.2f} GiB, "
                    f"limit={self.memory_limit_bytes / _GIB:.2f} GiB; "
                    "no idle unpinned model is eligible for eviction"
                )
            await self._evict_locked(candidates[0], reason="memory_pressure")

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
    ) -> ResidencyRecord:
        model_name = model_name.strip()
        if not model_name:
            raise ResidentModelError("model must not be empty")
        estimate = max(1, estimated_bytes or estimate_model_bytes(model_name))

        async with self._lock:
            canonical = self._canonical(model_name)
            if canonical is not None:
                record = self._records[canonical]
                if reload_if_changed and record.performance != performance:
                    record = await self._reload_locked(record, performance)
                record.last_used_at = self._clock()
                if pin:
                    record.pinned = True
                if replace_group is not None:
                    await self._replace_group_locked(record, replace_group)
                return record

            await self._evict_for_locked(estimate, exclude={model_name})
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
            )
            self.registry.add(entry)
            self._index_record(record)
            self.loads_total += 1

            try:
                await self._evict_for_locked(0, exclude={record.model_id})
                if replace_group is not None:
                    await self._replace_group_locked(record, replace_group)
            except BaseException:
                # Once the loader returns, this manager owns the engine.  A
                # later admission/replacement failure must not leave a model
                # resident even though the control-plane request was rejected.
                await self._evict_locked(record, reason="load_rollback", count=False)
                raise
            return record

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

        self.registry.remove(model_name)
        self._drop_record(model_name)
        try:
            stop = getattr(record.entry.engine, "stop", None)
            if callable(stop):
                result = stop()
                if asyncio.iscoroutine(result):
                    await result
        except BaseException:
            # A failed stop must not also make the still-existing engine
            # disappear from routing and residency accounting.
            self.registry.add(record.entry, is_default=primary)
            self._index_record(record)
            raise
        _release_allocator_cache()

        before = self._read_memory()
        try:
            entry = await self.loader(model_name, model_path, performance)
        except BaseException as reload_error:
            # The old engine has already released its Metal allocations so the
            # replacement can fit under the same budget. Best-effort restore
            # the last known-good config; never let a rejected Settings change
            # silently take every route for the primary model down.
            try:
                restored_entry = await self.loader(
                    model_name, model_path, record.performance
                )
                restored = ResidencyRecord(
                    entry=restored_entry,
                    estimated_bytes=estimate,
                    loaded_at=self._clock(),
                    last_used_at=self._clock(),
                    pinned=pinned,
                    primary=primary,
                    performance=record.performance,
                )
                self.registry.add(restored_entry, is_default=primary)
                self._index_record(restored)
                if primary and self._on_primary_changed is not None:
                    self._on_primary_changed(restored_entry)
            except BaseException:
                logger.exception(
                    "Failed to restore resident model %r after reload failure",
                    model_name,
                )
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
        self.registry.add(entry, is_default=primary)
        self._index_record(replacement)
        self.loads_total += 1
        if primary and self._on_primary_changed is not None:
            self._on_primary_changed(entry)
        return replacement

    async def _replace_group_locked(self, target: ResidencyRecord, group: str) -> None:
        """Make ``target`` the sole unpinned model in a lifecycle group.

        The desktop uses the ``assistant`` group for its chat picker: changing
        chat models replaces the previous text/VLM engine while independent
        image engines remain resident. A protected startup assistant hands its
        primary role to the replacement before the old engine is stopped, so
        legacy health/cache routes never retain a reference to unloaded weights.
        """

        if group != _replacement_group(target.entry):
            raise ResidentModelError(
                f"model {target.model_id!r} does not belong to replacement group {group!r}"
            )

        candidates = [
            record
            for record in self._records.values()
            if record.model_id != target.model_id
            and _replacement_group(record.entry) == group
        ]
        for record in candidates:
            if record.active_requests or not _engine_is_idle(record.entry.engine):
                raise ResidentModelBusyError("model is serving an active request")
            if record.pinned and not record.primary:
                raise ResidentModelError(
                    f"pinned model {record.model_id!r} cannot be replaced"
                )

        old_primary = next((record for record in candidates if record.primary), None)
        if old_primary is not None:
            old_primary.primary = False
            old_primary.pinned = False
            target.primary = True
            target.pinned = True
            self.registry.set_default(target.model_id)
            if self._on_primary_changed is not None:
                self._on_primary_changed(target.entry)

        for record in candidates:
            await self._evict_locked(record, reason=f"replace_{group}")

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
        self, record: ResidencyRecord, *, reason: str, count: bool = True
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
            record.last_used_at = self._clock()
            engine = record.entry.engine
        try:
            yield engine
        finally:
            async with self._lock:
                current = self._records.get(canonical)
                if current is not None:
                    current.active_requests = max(0, current.active_requests - 1)
                    current.last_used_at = self._clock()

    def snapshot(self) -> dict:
        now = self._clock()
        models = []
        for record in sorted(self._records.values(), key=lambda item: item.loaded_at):
            engine = record.entry.engine
            resident = not hasattr(engine, "is_resident") or bool(engine.is_resident)
            models.append(
                {
                    "id": record.model_id,
                    "model_path": record.entry.model_path,
                    "aliases": sorted(record.entry.aliases),
                    "modality": (_modality(record.entry)),
                    "state": record.state if resident else "registered",
                    "pinned": record.pinned,
                    "primary": record.primary,
                    "active_requests": record.active_requests,
                    "estimated_bytes": record.estimated_bytes,
                    "measured_bytes": record.measured_bytes or None,
                    "idle_seconds": max(0.0, now - record.last_used_at),
                    "performance": (
                        record.performance.payload() if record.performance else None
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
            "loads_total": self.loads_total,
            "evictions_total": self.evictions_total,
            "models": models,
        }
