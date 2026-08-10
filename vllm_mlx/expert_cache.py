# SPDX-License-Identifier: Apache-2.0
"""Byte-budgeted LRU cache of MoE expert weight bundles (`--disk-stream`).

Ported unmodified from the disk-streaming MoE spike's round-2
`expert_cache.py` (see PRD-rapid-mlx-integration.md) — no logic changes,
only the license header and this module docstring's framing were
adjusted for its new home in the package.

Deliberately deep and narrow: the only things this module knows are
`key = (layer_idx, expert_id)`, `value = opaque bundle`, LRU order, and a
byte budget. It has zero knowledge of safetensors, MLX, or the model --
`fetch_fn` (how to get a bundle on miss) and `size_fn` (how to measure a
bundle's bytes) are both injected by the caller. A caller wraps
`offset_reader.fetch_expert_weights` (bound to a checkpoint path, possibly
merged across gate/up/down sub-projections into one 9-array dict) as
`fetch_fn`; this module never imports or knows about either.

Default `size_fn` duck-types via `.nbytes`, recursing into dict/list/tuple
containers -- this covers both real bundles (dicts of `mx.array`, which all
carry `.nbytes`) and the dict-of-dicts shape a merged gate/up/down bundle
would have, with no MLX import required here.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any


def _default_size_fn(bundle: Any) -> int:
    """Duck-typed byte size: `.nbytes` if present, else sum over a
    dict/list/tuple's values/items. Raises if neither applies -- callers
    with an exotic bundle shape should pass their own `size_fn`.
    """
    if hasattr(bundle, "nbytes"):
        return int(bundle.nbytes)
    if isinstance(bundle, dict):
        return sum(_default_size_fn(v) for v in bundle.values())
    if isinstance(bundle, (list, tuple)):
        return sum(_default_size_fn(v) for v in bundle)
    raise TypeError(
        f"cannot measure size of bundle of type {type(bundle).__name__}; "
        "pass size_fn= to ExpertCache"
    )


class ExpertCache:
    """LRU cache of `(layer_idx, expert_id) -> bundle`, bounded by total
    bundle bytes rather than entry count. Single public method: `get`.

    Not thread-safe. `get()`'s check-then-fetch-then-insert sequence is
    unguarded: concurrent misses on the same `(layer_idx, expert_id)` key
    from two threads can both pass the `key in self._store` check, both
    call `fetch_fn` (double-fetch), and both insert -- `_total_bytes` then
    accumulates both entries' sizes even though only one survives in
    `_store`, permanently drifting the byte count upward relative to what
    is actually cached. This class was built for the current disk-stream
    round's single-request/single-threaded generation flow, where that
    race cannot occur. Rapid-MLX does handle concurrent requests elsewhere
    in the codebase, so callers using an `ExpertCache` instance from a
    multi-threaded/concurrent-request context must serialize access to it
    externally -- or this class needs real locking added first, which is
    out of scope for this round.
    """

    def __init__(
        self,
        fetch_fn: Callable[[int, int], Any],
        budget_bytes: int = 1_000_000_000,
        size_fn: Callable[[Any], int] = _default_size_fn,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
        self._fetch_fn = fetch_fn
        self._budget_bytes = budget_bytes
        self._size_fn = size_fn
        self._store: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._total_bytes = 0
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def get(self, layer_idx: int, expert_id: int) -> Any:
        key = (layer_idx, expert_id)
        if key in self._store:
            self._hits += 1
            self._store.move_to_end(key)  # refresh recency on hit
            return self._store[key][0]

        self._misses += 1
        bundle = self._fetch_fn(layer_idx, expert_id)
        size = self._size_fn(bundle)

        # A single expert may exceed a deliberately tiny operator budget.
        # Return it for this forward pass but do not retain it; the cache's
        # advertised byte bound must remain a hard invariant.
        if size > self._budget_bytes:
            return bundle

        # Evict LRU entries (oldest-first in OrderedDict) until it fits.
        while self._store and self._total_bytes + size > self._budget_bytes:
            _, (_, evicted_size) = self._store.popitem(last=False)
            self._total_bytes -= evicted_size

        self._store[key] = (bundle, size)
        self._total_bytes += size
        return bundle


if __name__ == "__main__":
    # Smallest runnable self-check (ponytail rule) -- full coverage lives
    # in tests/test_expert_cache.py, run that via pytest for the real suite.
    class _Bundle:
        nbytes = 10

    cache = ExpertCache(lambda layer, expert: _Bundle(), budget_bytes=25)
    cache.get(0, 0)
    cache.get(0, 1)
    cache.get(0, 0)  # hit, refreshes recency
    cache.get(0, 2)  # overflow -> evicts (0,1), the LRU one
    assert cache.hits == 1 and cache.misses == 3
    assert cache.total_bytes <= 25
    print("expert_cache self-check OK")
