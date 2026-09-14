"""Recurrent-state checkpoints for hybrid (GatedDeltaNet / Mamba) prompt caches.

A KV cache can be rewound to any earlier token by slicing ``offset``; a
recurrent layer's ``ArraysCache`` cannot — its state is a running summary,
so a stored hybrid entry is only reusable at the *exact* token length it
was captured at. That is why ``MemoryAwarePrefixCache`` refuses the
supersequence and longest-common-prefix paths for hybrid models: a request
that shares 6 000 of an entry's 6 500 tokens falls back to a cold prefill.

This module records the recurrent state at prefill chunk boundaries and lets
the fetch side "snap" a trim to the newest checkpoint at or below the shared
prefix, instead of refusing. The idea follows mlx-lm's ``state_checkpoint``
/ ``trim_to_position`` (mlx-lm-unified, APC v2): keep a bounded number of
checkpoints per layer, keep the newest, and thin by dropping the one with
the smallest gap to its predecessor.

Checkpoints ride on the per-layer cache object as an immutable
:class:`StateCheckpoints` holder (``_rapid_state_checkpoints``), so they
survive the ``deepcopy`` a prefix-cache hit performs and are shared, never
duplicated, across copies. MLX arrays are immutable, so holding a reference
to a superseded state array is the whole snapshot — no copy is made.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_ATTR = "_rapid_state_checkpoints"

_DEFAULT_MAX = 4
_DEFAULT_STRIDE = 2048


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def checkpoint_max() -> int:
    """Checkpoints kept per recurrent layer (0 disables recording)."""
    return _env_int("RAPID_MLX_HYBRID_CHECKPOINT_MAX", _DEFAULT_MAX)


def checkpoint_stride() -> int:
    """Minimum token gap between two recorded checkpoints."""
    return max(1, _env_int("RAPID_MLX_HYBRID_CHECKPOINT_STRIDE", _DEFAULT_STRIDE))


def _array_bytes(arr: Any) -> int:
    if arr is None:
        return 0
    shape = getattr(arr, "shape", None)
    dtype = getattr(arr, "dtype", None)
    if shape is not None and dtype is not None and hasattr(dtype, "size"):
        return int(math.prod(shape)) * int(dtype.size)
    return int(getattr(arr, "nbytes", 0) or 0)


class StateCheckpoints:
    """Immutable, position-sorted ``(position, arrays)`` snapshots of one layer.

    Every mutation returns a new holder, so a holder attached to a stored
    cache entry can be shared by every copy of that entry.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[tuple[int, tuple[Any, ...]]] = ()):
        self._items: tuple[tuple[int, tuple[Any, ...]], ...] = tuple(
            sorted(((int(p), tuple(a)) for p, a in items), key=lambda it: it[0])
        )

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: dict[int, Any]) -> StateCheckpoints:
        return self

    def __copy__(self) -> StateCheckpoints:
        return self

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(p for p, _ in self._items)

    @property
    def nbytes(self) -> int:
        return sum(_array_bytes(a) for _, arrays in self._items for a in arrays)

    def arrays_at(self, position: int) -> tuple[Any, ...] | None:
        for p, arrays in self._items:
            if p == position:
                return arrays
        return None

    def newest_at_or_below(self, position: int) -> int:
        """Largest checkpoint position ``<= position``, or 0."""
        best = 0
        for p, _ in self._items:
            if p <= position and p > best:
                best = p
        return best

    def truncated(self, position: int) -> StateCheckpoints:
        """Keep only checkpoints at or below ``position``."""
        return StateCheckpoints(it for it in self._items if it[0] <= position)

    def with_checkpoint(
        self,
        position: int,
        arrays: Sequence[Any],
        *,
        max_count: int,
        stride: int,
    ) -> StateCheckpoints:
        """Return a holder that also records ``arrays`` at ``position``.

        Recording is skipped (``self`` returned) when the position is not
        newer than the newest checkpoint by at least ``stride`` tokens.
        When the bound is exceeded, the checkpoint with the smallest gap to
        its predecessor is dropped; the newest one is never dropped, so the
        most recent prefill work is always the cheapest to resume.
        """
        if max_count <= 0:
            return self
        position = int(position)
        if self._items:
            newest = self._items[-1][0]
            if position <= newest or position - newest < stride:
                return self
        items = list(self._items) + [(position, tuple(arrays))]
        while len(items) > max_count:
            # gap of item i = items[i].pos - items[i-1].pos (first item: pos)
            victim = 0
            smallest = items[0][0]
            for i in range(1, len(items) - 1):
                gap = items[i][0] - items[i - 1][0]
                if gap < smallest:
                    smallest = gap
                    victim = i
            del items[victim]
        return StateCheckpoints(items)


def _recurrent_cache_types() -> tuple[type, ...]:
    """The cache classes whose ``cache`` list this module knows how to
    checkpoint and restore: mlx-lm's ``ArraysCache`` (and subclasses such as
    the Mamba/GatedDeltaNet state caches). Positive identification only —
    a look-alike wrapper with a ``cache`` attribute is NOT accepted, so the
    fetch path keeps refusing it instead of restoring a shallow copy of
    state it does not understand."""
    global _RECURRENT_TYPES
    if _RECURRENT_TYPES is None:
        try:
            from mlx_lm.models.cache import ArraysCache

            _RECURRENT_TYPES = (ArraysCache,)
        except Exception:
            _RECURRENT_TYPES = ()
    return _RECURRENT_TYPES


_RECURRENT_TYPES: tuple[type, ...] | None = None


def is_recurrent_layer(layer: Any) -> bool:
    """True only for a genuine mlx-lm ``ArraysCache`` whose ``cache`` is the
    expected list of state arrays."""
    if layer is None:
        return False
    types = _recurrent_cache_types()
    if not types or not isinstance(layer, types):
        return False
    return isinstance(getattr(layer, "cache", None), list)


def layer_checkpoints(layer: Any) -> StateCheckpoints | None:
    holder = getattr(layer, CHECKPOINT_ATTR, None)
    return holder if isinstance(holder, StateCheckpoints) else None


def attach_checkpoints(
    cache: Sequence[Any],
    holders: Sequence[StateCheckpoints | None],
    *,
    max_position: int | None = None,
) -> None:
    """Attach per-layer holders (aligned with ``cache``) onto recurrent layers.

    ``max_position`` is the token length the cache is stored at. Checkpoints
    past it were recorded from tokens the entry does not contain (e.g. the
    tail beyond a message boundary), so they are dropped rather than let a
    later request resume from state computed on a different prompt.
    """
    if len(holders) != len(cache):
        return
    for layer, holder in zip(cache, holders):
        if not is_recurrent_layer(layer):
            continue
        if holder is not None and max_position is not None:
            holder = holder.truncated(max_position)
        if holder is None or not holder.positions:
            # ``holders`` is the source of truth: a layer copied from a
            # checkpoint-bearing cache must not keep a stale holder.
            if hasattr(layer, CHECKPOINT_ATTR):
                delattr(layer, CHECKPOINT_ATTR)
            continue
        setattr(layer, CHECKPOINT_ATTR, holder)


def collect_checkpoints(cache: Sequence[Any]) -> list[StateCheckpoints | None]:
    """Per-layer holders (aligned with ``cache``); ``None`` for other layers."""
    return [
        layer_checkpoints(layer) if is_recurrent_layer(layer) else None
        for layer in cache
    ]


def checkpoint_bytes(cache: Sequence[Any]) -> int:
    """Bytes held by checkpoints across ``cache`` (for memory accounting)."""
    return sum(h.nbytes for h in collect_checkpoints(cache) if h is not None)


def _materialise(states: dict[int, tuple[Any, ...]]) -> bool:
    """Force the checkpointed arrays now. A failure (allocation, Metal
    error) must not leave a lazily-built graph registered as a checkpoint
    that only blows up on a later cache hit, so the caller records nothing."""
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - MLX absent in some test envs
        return True
    try:
        mx.eval(*[a for arrays in states.values() for a in arrays])
    except Exception as exc:
        logger.debug("[hybrid_checkpoint] materialisation failed: %s", exc)
        return False
    return True


def record_checkpoints(
    cache: Sequence[Any],
    holders: list[StateCheckpoints | None],
    position: int,
    *,
    max_count: int | None = None,
    stride: int | None = None,
) -> bool:
    """Record every recurrent layer of ``cache`` at ``position`` into ``holders``.

    ``cache`` must hold single-row (batch 1) layers, e.g. what
    ``BatchGenerator.extract_cache`` returns for one request. ``holders`` is
    updated in place (aligned with ``cache``). Returns True when a checkpoint
    was recorded. Nothing is recorded unless *every* recurrent layer has a
    materialised state, so all layers always share one position set.
    """
    max_count = checkpoint_max() if max_count is None else max_count
    stride = checkpoint_stride() if stride is None else stride
    if max_count <= 0 or position <= 0 or len(holders) != len(cache):
        return False
    recurrent = [i for i, layer in enumerate(cache) if is_recurrent_layer(layer)]
    if not recurrent:
        return False
    states: dict[int, tuple[Any, ...]] = {}
    for i in recurrent:
        arrays = tuple(cache[i].cache)
        if any(a is None for a in arrays):
            return False
        states[i] = arrays
    # Gate on one layer: all layers share the same position history.
    probe = holders[recurrent[0]]
    newest = probe.positions[-1] if probe is not None and probe.positions else None
    if newest is not None and (position <= newest or position - newest < stride):
        return False
    if not _materialise(states):
        return False
    for i in recurrent:
        base = holders[i] or StateCheckpoints()
        holders[i] = base.with_checkpoint(
            position, states[i], max_count=max_count, stride=stride
        )
    return True


def achievable_position(cache: Sequence[Any], target: int) -> int:
    """Largest position ``<= target`` at which every recurrent layer has a
    checkpoint. Returns ``target`` when the cache has no recurrent layer and
    0 when no common checkpoint exists."""
    common: set[int] | None = None
    for layer in cache:
        if not is_recurrent_layer(layer):
            continue
        holder = layer_checkpoints(layer)
        positions = {p for p in holder.positions if p <= target} if holder else set()
        common = positions if common is None else common & positions
        if not common:
            return 0
    if common is None:
        return int(target)
    return max(common)


def restore_recurrent_layer(layer: Any, position: int) -> Any | None:
    """A copy of ``layer`` rewound to its checkpoint at ``position``."""
    holder = layer_checkpoints(layer)
    if holder is None:
        return None
    arrays = holder.arrays_at(position)
    if arrays is None:
        return None
    restored = copy.copy(layer)
    restored.cache = list(arrays)
    for attr in ("left_padding", "lengths"):
        if hasattr(restored, attr):
            setattr(restored, attr, None)
    setattr(restored, CHECKPOINT_ATTR, holder.truncated(position))
    return restored
