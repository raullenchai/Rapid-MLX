# SPDX-License-Identifier: Apache-2.0
"""Shape-stable, whole-step compiled replay for qualified width-one decode.

The implementation keeps mutable cache arrays explicit inputs and outputs of
``mx.compile``.  It never captures request state in a compiled closure.  A
fixed-shape KV slab removes Python graph construction from steady-state decode;
hybrid ``ArraysCache`` state is threaded alongside it.

The cache/replay design was adapted from MIT-licensed work Copyright © 2023
Apple Inc.
"""

from __future__ import annotations

import logging
import os
import types
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import mlx.core as mx

# MUST install the MLX hardware-compat shim before importing any ``mlx_lm``
# submodule. ``mlx_lm.__init__`` imports its generation module, which captures
# a thread-local GPU stream at import time; that stream is unusable on M5
# single-stream GPUs unless the compatibility wrapper is already active.
from . import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.cache import ArraysCache, KVCache, _BaseCache  # noqa: E402

from .compiled_precision import (
    compiled_decode_precision,
    install_qwen35_attention_gate_precision,
)

logger = logging.getLogger(__name__)

_BUCKETS = (1023, 1024, 2048, 4096, 8192, 16384)
_MAX_CONTEXT = 16384
_MAX_VARIANTS = 8
_MIN_INITIAL_BUCKET_HEADROOM = 32
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class CompiledDecodePoisonedError(RuntimeError):
    """A submitted compiled graph failed and must never be retried eagerly."""


class ShapeStableKVCache(_BaseCache):
    """Full-context KV cache with fixed array shapes inside a capacity bucket."""

    _rapid_shape_stable_kv = True

    def __init__(self, *, buckets: tuple[int, ...] = _BUCKETS):
        if not buckets or tuple(sorted(set(buckets))) != buckets or buckets[0] <= 0:
            raise ValueError("shape-stable KV buckets must be sorted positive values")
        self.buckets = buckets
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.capacity = 0
        self.offset = mx.array(0, dtype=mx.int32)
        self._host_offset = 0
        self._columns: mx.array | None = None
        self._rapid_compiled_owner: Any | None = None

    def _bucket_for(self, needed: int) -> int:
        for bucket in self.buckets:
            if needed <= bucket:
                return bucket
        raise RuntimeError(
            f"compiled decode context exceeds the qualified {self.buckets[-1]} tokens"
        )

    def _allocate(self, keys: mx.array, values: mx.array, needed: int) -> None:
        batch, heads, _, key_dim = keys.shape
        value_dim = values.shape[-1]
        capacity = self._bucket_for(needed)
        next_keys = mx.zeros((batch, heads, capacity, key_dim), keys.dtype)
        next_values = mx.zeros((batch, heads, capacity, value_dim), values.dtype)
        if self.keys is not None:
            assert self.values is not None
            live = self._host_offset
            next_keys[..., :live, :] = self.keys[..., :live, :]
            next_values[..., :live, :] = self.values[..., :live, :]
        self.keys = next_keys
        self.values = next_values
        self.capacity = capacity
        self._columns = None

    def reserve(self, width: int) -> bool:
        if isinstance(width, bool) or not isinstance(width, int) or width < 0:
            raise ValueError("shape-stable KV reserve width must be non-negative")
        if self.keys is None:
            if width == 0:
                return False
            raise ValueError("cannot reserve an empty shape-stable KV cache")
        assert self.values is not None
        needed = self._host_offset + width
        if needed <= self.capacity:
            return False
        self._allocate(self.keys, self.values, needed)
        return True

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        width = int(keys.shape[2])
        needed = self._host_offset + width
        if self.keys is None or needed > self.capacity:
            self._allocate(keys, values, needed)
        assert self.keys is not None and self.values is not None
        start = self.offset[None]
        self.keys = mx.slice_update(self.keys, keys, start, axes=(2,))
        self.values = mx.slice_update(self.values, values, start, axes=(2,))
        self.offset = self.offset + width
        self._host_offset += width
        return self.keys, self.values

    def make_mask(self, width: int, return_array: bool = False, window_size=None):
        del return_array
        if self.capacity == 0:
            raise ValueError("shape-stable KV mask requested before cache fill")
        self.reserve(width)
        if self._columns is None or self._columns.size != self.capacity:
            self._columns = mx.arange(self.capacity, dtype=mx.int32)
        assert self._columns is not None
        rows = self.offset + mx.arange(width, dtype=mx.int32)
        mask = self._columns[None, :] <= rows[:, None]
        if window_size is not None:
            mask = mask & (self._columns[None, :] > rows[:, None] - window_size)
        return mask[None, None]

    def size(self) -> int:
        return self._host_offset

    def empty(self) -> bool:
        return self.keys is None

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._host_offset, int(n))
        self._host_offset -= n
        self.offset = self.offset - n
        return n

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        assert self.values is not None
        return int(self.keys.nbytes + self.values.nbytes)

    @property
    def state(self):
        if self.keys is None:
            return ()
        return self.keys, self.values, self.offset

    @state.setter
    def state(self, value) -> None:
        if not value:
            self.keys = self.values = None
            self.capacity = 0
            self.offset = mx.array(0, dtype=mx.int32)
            self._host_offset = 0
            self._columns = None
            return
        self.keys, self.values, offset = value
        assert self.keys is not None
        self.capacity = int(self.keys.shape[2])
        self.offset = offset.astype(mx.int32)
        self._host_offset = int(self.offset.item())
        self._columns = None

    @classmethod
    def from_kv_cache(cls, cache: KVCache):
        result = cls()
        if cache.keys is None:
            return result
        assert cache.values is not None
        live = int(cache.offset)
        keys = cache.keys[..., :live, :]
        values = cache.values[..., :live, :]
        result._allocate(keys, values, live)
        assert result.keys is not None and result.values is not None
        result.keys[..., :live, :] = keys
        result.values[..., :live, :] = values
        result.offset = mx.array(live, dtype=mx.int32)
        result._host_offset = live
        return result

    def _drain_owner(self, phase: str) -> None:
        owner = self._rapid_compiled_owner
        if owner is not None:
            owner.drain_pending(phase=phase)

    def to_kv_cache(self, *, drain: bool = True) -> KVCache:
        if drain:
            self._drain_owner("cache conversion")
        result = KVCache()
        if self.keys is not None:
            assert self.values is not None
            live = self._host_offset
            result.keys = mx.contiguous(self.keys[..., :live, :])
            result.values = mx.contiguous(self.values[..., :live, :])
            result.offset = live
        return result

    # Minimal singleton batch surface. A B>1 join converts through
    # ``singleton_cache_fastpath._promote_layer`` before ``extend`` is called.
    def filter(self, keep) -> None:
        indices = [int(value) for value in keep]
        if indices == [0]:
            return
        if indices:
            raise IndexError("shape-stable singleton KV cache only owns row 0")
        self._drain_owner("request removal")
        self.state = ()

    def extract(self, index: int):
        if int(index) != 0:
            raise IndexError("shape-stable singleton KV cache only owns row 0")
        return self.to_kv_cache()

    def extend(self, other) -> None:
        del other
        raise NotImplementedError("promote shape-stable KV cache before batching")

    def to_quantized(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "compiled decode is incompatible with KV quantization"
        )


class _KVSlot:
    n_arrays = 3

    def __init__(self, cache: ShapeStableKVCache):
        self.cache = cache

    def collect(self):
        return [self.cache.keys, self.cache.values, self.cache.offset]

    def install(self, arrays) -> None:
        self.cache.keys, self.cache.values, self.cache.offset = arrays

    def snapshot(self):
        return self.cache._host_offset

    def commit(self, snapshot: int, width: int) -> None:
        self.cache._host_offset = snapshot + width

    def rollback(self, arrays, snapshot: int) -> None:
        self.install(arrays)
        self.cache._host_offset = snapshot


class _ArraysSlot:
    def __init__(self, cache: ArraysCache):
        self.cache = cache
        self.n_arrays = len(cache.cache)

    def collect(self):
        return list(self.cache.cache)

    def install(self, arrays) -> None:
        self.cache.cache = list(arrays)

    def snapshot(self):
        return None

    def commit(self, snapshot, width: int) -> None:
        del snapshot, width
        assert self.cache.lengths is None and self.cache.left_padding is None

    def rollback(self, arrays, snapshot) -> None:
        del snapshot
        self.install(arrays)


class CompiledDecodeStep:
    """Compile once per KV capacity and replay with explicit request state."""

    def __init__(self, model: Any, cache: list[Any], *, max_variants: int = 8):
        if not 1 <= max_variants <= _MAX_VARIANTS:
            raise ValueError(f"max_variants must be between 1 and {_MAX_VARIANTS}")
        self.model = model
        self.cache = cache
        self.max_variants = max_variants
        self.plan: list[_KVSlot | _ArraysSlot] = []
        self._kv_slots: list[_KVSlot] = []
        for index, layer in enumerate(cache):
            slot: _KVSlot | _ArraysSlot
            if type(layer) is ShapeStableKVCache:
                slot = _KVSlot(layer)
                self._kv_slots.append(slot)
                layer._rapid_compiled_owner = self
            elif type(layer) is ArraysCache:
                if (
                    layer.lengths is not None
                    or layer.left_padding is not None
                    or any(value is None for value in layer.cache)
                ):
                    raise TypeError(f"hybrid cache[{index}] is not ready for replay")
                slot = _ArraysSlot(layer)
            else:
                raise TypeError(f"cache[{index}] is not shape-stable")
            self.plan.append(slot)
        if not self._kv_slots:
            raise TypeError("compiled decode requires a full-attention KV cache")
        if len({slot.cache.size() for slot in self._kv_slots}) != 1:
            raise TypeError("full-attention KV positions are not synchronized")
        self._variants: dict[tuple[Any, ...], tuple[Any, list[tuple[int, int]]]] = {}
        self._pending: list[mx.array] = []
        self.trace_counts: dict[tuple[Any, ...], int] = {}
        self.submission_count = 0
        self.completion_count = 0
        self._poison_reason: str | None = None

    @property
    def poisoned(self) -> bool:
        return self._poison_reason is not None

    def _guard(self) -> None:
        if self.poisoned:
            raise CompiledDecodePoisonedError(
                f"compiled decode is poisoned and must be discarded: {self._poison_reason}"
            )

    def _signature(self, tokens: mx.array):
        return (
            tokens.shape,
            tokens.dtype,
            *(slot.cache.capacity for slot in self._kv_slots),
        )

    def _build(self, key):
        splits = []
        start = 0
        for slot in self.plan:
            splits.append((start, start + slot.n_arrays))
            start += slot.n_arrays
        self.trace_counts.setdefault(key, 0)

        def traced(tokens, *state):
            self.trace_counts[key] += 1
            for slot, (low, high) in zip(self.plan, splits, strict=True):
                slot.install(state[low:high])
            with compiled_decode_precision():
                logits = self.model(tokens, cache=self.cache)
            next_state = []
            for slot in self.plan:
                next_state.extend(slot.collect())
            return (logits, *next_state)

        return mx.compile(traced), splits

    def poison(
        self, error: BaseException, *, phase: str
    ) -> CompiledDecodePoisonedError:
        if not self.poisoned:
            self._poison_reason = f"{phase}: {error!r}"
            self._variants.clear()
            self._pending.clear()
        return CompiledDecodePoisonedError(
            "compiled decode failed after submission; the request was aborted "
            f"without retrying its token ({self._poison_reason})"
        )

    def __call__(self, tokens: mx.array) -> mx.array:
        self._guard()
        if tokens.ndim != 2 or tuple(tokens.shape) != (1, 1):
            raise ValueError("compiled decode is qualified only for batch 1, width 1")
        if any(getattr(layer, "speculating", False) for layer in self.cache):
            raise RuntimeError("compiled decode cannot run during speculative rollback")
        position = self._kv_slots[0].cache.size()
        if position + 1 > _MAX_CONTEXT:
            raise RuntimeError("compiled decode reached its qualified context limit")
        for slot in self._kv_slots:
            slot.cache.reserve(1)
        key = self._signature(tokens)
        entry = self._variants.get(key)
        is_new = entry is None
        if entry is None:
            if len(self._variants) >= self.max_variants:
                self._variants.clear()
            entry = self._build(key)
            self._variants[key] = entry
        compiled, splits = entry
        state = []
        snapshots = []
        for plan_slot in self.plan:
            state.extend(plan_slot.collect())
            snapshots.append(plan_slot.snapshot())
        try:
            output = compiled(tokens, *state)
            if is_new:
                mx.eval(output)
        except Exception as error:
            for plan_slot, (low, high), snapshot in zip(
                self.plan, splits, snapshots, strict=True
            ):
                plan_slot.rollback(state[low:high], snapshot)
            raise self.poison(
                error, phase="trace" if is_new else "submission"
            ) from error
        logits, next_state = output[0], output[1:]
        for plan_slot, (low, high), snapshot in zip(
            self.plan, splits, snapshots, strict=True
        ):
            plan_slot.install(next_state[low:high])
            plan_slot.commit(snapshot, 1)
        self._pending.append(logits)
        self.submission_count += 1
        return logits

    def confirm_oldest(self, *dependent_values) -> None:
        """Materialize and receipt the oldest submitted step."""
        self._guard()
        if not self._pending:
            return
        try:
            mx.eval(*dependent_values)
        except Exception as error:
            raise self.poison(error, phase="output materialization") from error
        self._pending.pop(0)
        self.completion_count += 1

    def drain_pending(self, *, phase: str = "request finalization") -> None:
        self._guard()
        while self._pending:
            output = self._pending[0]
            try:
                mx.eval(output, [layer.state for layer in self.cache])
            except Exception as error:
                raise self.poison(error, phase=phase) from error
            self._pending.pop(0)
            self.completion_count += 1

    def detach(self) -> None:
        for slot in self._kv_slots:
            if slot.cache._rapid_compiled_owner is self:
                slot.cache._rapid_compiled_owner = None

    def receipt(self) -> dict[str, Any]:
        return {
            "traces": sum(self.trace_counts.values()),
            "variants": len(self._variants),
            "submissions": self.submission_count,
            "completions": self.completion_count,
            "pending": len(self._pending),
            "poisoned": self.poisoned,
            "poison_reason": self._poison_reason,
        }


def enabled() -> bool:
    return (
        os.environ.get("RAPID_MLX_COMPILED_DECODE", "1").strip().lower()
        not in _FALSE_VALUES
    )


def convert_cache(cache: list[Any]) -> list[Any]:
    """Build and materialize a shape-stable candidate without mutating input."""
    converted = []
    for index, layer in enumerate(cache):
        if type(layer) is KVCache:
            if (
                layer.keys is None
                or layer.values is None
                or int(layer.keys.shape[0]) != 1
            ):
                raise TypeError(f"KV cache[{index}] is not a filled batch-one cache")
            converted.append(ShapeStableKVCache.from_kv_cache(layer))
        elif type(layer) is ArraysCache:
            if (
                layer.lengths is not None
                or layer.left_padding is not None
                or any(
                    value is None or int(value.shape[0]) != 1 for value in layer.cache
                )
            ):
                raise TypeError(
                    f"hybrid cache[{index}] is not a filled batch-one cache"
                )
            converted.append(layer)
        else:
            raise TypeError(f"cache[{index}] is a {type(layer).__name__}")
    mx.eval([layer.state for layer in converted])
    return converted


def model_qualification_reason(model: Any, model_name: str | None) -> str | None:
    """Fail closed to the measured Qwen3.6-35B-A3B text architecture."""
    if not enabled():
        return "disabled by RAPID_MLX_COMPILED_DECODE"
    name = (model_name or "").lower()
    if "qwen3.6-35b" not in name:
        return "checkpoint is not a Qwen3.6-35B artifact"
    try:
        mlx_lm_version = version("mlx-lm")
    except PackageNotFoundError:
        return "mlx-lm package metadata is unavailable"
    if mlx_lm_version != "0.31.3":
        return f"mlx-lm {mlx_lm_version} has not been qualified for replay"
    # Python state swaps in the shared multimodal wrapper must remain eager.
    if type(model).__module__ == "vllm_mlx.engine.batched":
        return "shared multimodal wrappers are not trace-safe"
    args = getattr(model, "args", None)
    text_config = getattr(args, "text_config", None)

    def arg(name: str):
        if isinstance(text_config, dict) and name in text_config:
            return text_config[name]
        return getattr(args, name, None)

    expected = {
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "full_attention_interval": 4,
    }
    if args is None or any(arg(key) != value for key, value in expected.items()):
        return "model geometry is outside the qualified Qwen3.6-35B topology"
    try:
        modules = [module for _, module in model.named_modules()]
    except Exception:
        return "model does not expose named modules"
    routers = [
        module
        for module in modules
        if hasattr(module, "num_experts") and hasattr(module, "top_k")
    ]
    if (
        len(
            [
                module
                for module in routers
                if getattr(module, "_rapid_qwen35_fused_router", False)
            ]
        )
        != 40
    ):
        return "precision-preserving fused MoE routing is not active on all layers"
    gdn = [
        module
        for module in modules
        if getattr(module, "_rapid_qwen35_fused_gdn_decode", False)
    ]
    if len(gdn) != 30:
        return (
            "precision-preserving fused GDN decode is not active on all linear layers"
        )
    attention = [
        module
        for module in modules
        if getattr(module, "_rapid_compiled_gate_precision", False)
    ]
    if len(attention) != 10:
        return "precision-preserving attention gates are not active on all layers"
    return None


class _CompiledForward:
    def __init__(self, step: CompiledDecodeStep):
        self.step = step

    def __call__(self, tokens, *, cache):
        if cache is not self.step.cache:
            raise RuntimeError("compiled decode received a foreign request cache")
        return self.step(tokens)


def install_compiled_decode(
    batch_gen: Any, model: Any, *, model_name: str | None
) -> bool:
    """Install request-private B=1 replay beneath mlx-lm's batch scheduler."""
    preliminary = model_qualification_reason(model, model_name)
    # The attention tag is the only expected preliminary failure that the
    # installer itself can resolve. Everything else declines without touching
    # a process-global dependency class.
    if preliminary not in (
        None,
        "precision-preserving attention gates are not active on all layers",
    ):
        logger.debug("[compiled-decode] declined: %s", preliminary)
        return False
    try:
        install_qwen35_attention_gate_precision(model)
    except Exception:
        logger.warning(
            "[compiled-decode] attention precision patch failed", exc_info=True
        )
        return False
    reason = model_qualification_reason(model, model_name)
    if reason is not None:
        logger.debug("[compiled-decode] declined: %s", reason)
        return False
    generation = getattr(batch_gen, "_generation_batch", None)
    if generation is None or not hasattr(generation, "_step"):
        logger.warning("[compiled-decode] incompatible mlx-lm GenerationBatch")
        return False

    original_step = generation._step
    original_filter = generation.filter
    state: dict[str, CompiledDecodeStep | None] = {"step": None}
    declined_uids: set[int] = set()
    stats: dict[str, Any] = {
        "attachments": 0,
        "fallbacks": 0,
        "traces": 0,
        "submissions": 0,
        "completions": 0,
        "poisoned": False,
        "last_decline_reason": None,
    }

    def detach(*, convert: bool, phase: str) -> None:
        step = state["step"]
        if step is None:
            return
        step.drain_pending(phase=phase)
        stats["traces"] += sum(step.trace_counts.values())
        stats["submissions"] += step.submission_count
        stats["completions"] += step.completion_count
        if convert:
            from .singleton_cache_fastpath import _bind_singleton_surface

            restored = []
            for layer in generation.prompt_cache:
                if type(layer) is ShapeStableKVCache:
                    eager = layer.to_kv_cache(drain=False)
                    _bind_singleton_surface(eager)
                    restored.append(eager)
                else:
                    restored.append(layer)
            generation.prompt_cache[:] = restored
        step.detach()
        state["step"] = None

    def try_attach() -> None:
        if state["step"] is not None or len(generation.uids) != 1:
            return
        uid = generation.uids[0]
        if uid in declined_uids:
            return
        if generation._next_tokens is None or tuple(generation._next_tokens.shape) != (
            1,
        ):
            return
        try:
            converted = convert_cache(generation.prompt_cache)
            position = max(
                layer.size() for layer in converted if type(layer) is ShapeStableKVCache
            )
            if position >= _MAX_CONTEXT:
                raise ValueError("context is already outside the compiled replay limit")
            headroom = min(
                layer.capacity - layer.size()
                for layer in converted
                if type(layer) is ShapeStableKVCache
            )
            if 0 < headroom < _MIN_INITIAL_BUCKET_HEADROOM:
                raise ValueError(
                    "context is too close to a replay bucket boundary to "
                    "amortize the initial retraces"
                )
            step = CompiledDecodeStep(model, converted)
        except (TypeError, ValueError, RuntimeError) as error:
            stats["last_decline_reason"] = str(error)
            declined_uids.add(uid)
            return
        generation.prompt_cache[:] = converted
        step.cache = generation.prompt_cache
        state["step"] = step
        stats["attachments"] += 1
        stats["last_decline_reason"] = None

    def compiled_step(self):
        step = state["step"]
        if step is not None and (
            len(self.uids) != 1
            or not self.prompt_cache
            or any(
                type(layer) not in (ShapeStableKVCache, ArraysCache)
                for layer in self.prompt_cache
            )
        ):
            declined_uids.update(self.uids)
            detach(convert=False, phase="batch transition")
            step = None
        if step is None:
            try_attach()
            step = state["step"]
        if step is not None and step._kv_slots[0].cache.size() >= _MAX_CONTEXT:
            stats["fallbacks"] += 1
            declined_uids.update(self.uids)
            detach(convert=True, phase="context-limit fallback")
            step = None
        if step is None:
            return original_step()

        saved_model = self.model
        self.model = _CompiledForward(step)
        try:
            result = original_step()
            # GenerationBatch materialized the prior logprobs before returning.
            # Keep this call's output pending so mlx-lm retains depth-1 overlap.
            if len(step._pending) > 1:
                step.confirm_oldest(self._current_tokens, self._current_logprobs)
            return result
        except Exception as error:
            if isinstance(error, CompiledDecodePoisonedError):
                stats["poisoned"] = True
                raise
            stats["poisoned"] = True
            raise step.poison(error, phase="generation step") from error
        finally:
            self.model = saved_model

    def compiled_filter(self, keep):
        previous_uids = list(self.uids)
        step = state["step"]
        if step is not None and list(keep) != [0]:
            detach(convert=False, phase="request removal")
        result = original_filter(keep)
        kept = {previous_uids[int(index)] for index in keep}
        declined_uids.intersection_update(kept)
        return result

    generation._step = types.MethodType(compiled_step, generation)
    generation.filter = types.MethodType(compiled_filter, generation)
    generation._rapid_compiled_decode_state = state
    generation._rapid_compiled_decode_declined_uids = declined_uids
    batch_gen._rapid_compiled_decode_stats = stats
    logger.info("[compiled-decode] installed for qualified Qwen3.6-35B B=1 decode")
    return True
