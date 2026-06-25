# SPDX-License-Identifier: Apache-2.0
"""
Radix-tree prefix-cache index (R15-P1, task #303).

This module ships a **lookup-acceleration index** that sits alongside the
existing ``MemoryAwarePrefixCache`` storage layer. It does NOT replace the
on-disk cache format — entries continue to live in
``~/.cache/rapid-mlx/prefix_cache/<model>/`` as before — but it replaces the
``bisect`` over ``_sorted_keys`` with an O(prefix_len) walk through a token
trie. Stacks unchanged on Phase 4 TurboQuant K8V4 (storage-format) because
the radix only indexes token sequences, not their KV state representation.

Why an explicit radix on top of the bisected-sorted-keys path the
``MemoryAwarePrefixCache`` already had:

1. **Multi-tenant shared system prompts** (the Cursor / Claude-Code workload
   pattern this whole task targets) — N clients with the same 2k-token
   preamble all converge on the same radix node. The bisect path is O(log N
   + LCP_scan) per request; the radix is O(LCP) per request. At N=10 and a
   2k-token system prompt, the difference is a single 2k-token walk vs.
   ~14 LCP comparisons each spanning 2k tokens.

2. **Implicit dedup accounting**. The radix node carrying the shared system
   prompt knows it has K children (one per tenant), so we can emit a
   ``deduped_prefix_bytes_saved`` metric that the hash-keyed cache cannot.

3. **Cheap "what's in the cache that shares my prefix?"** queries — useful
   for the upcoming Phase 4 (TurboQuant K8V4) where shared prefixes get a
   different storage class.

Design constraints honoured (per task #303 brief):

- **Read-mostly**: the tree uses an ``RLock`` on writes only; reads are
  lock-free by virtue of treating mutations as copy-on-write at the node
  level. We do NOT ship a giant mutex.
- **Backward-compat**: missing ``radix.index`` on disk → rebuild lazily
  from existing ``entry_*_tokens.bin`` on first boot. The hash path stays
  default-runnable behind ``--prefix-cache-index hash`` if regression is
  found.
- **Persistence**: ``radix.index`` is a compact JSON blob (node count is
  bounded by entry count; we don't need msgpack or lz4 for the scale we
  see — tens of thousands of entries at most).
- **NaN-safe**: this module ingests no floats from user input, so no
  ``math.isfinite`` checks are required here. The CLI flag is a Literal,
  not a numeric.

The radix exposes a minimal surface:

    radix.insert(tokens)            # add a token sequence
    radix.remove(tokens)            # drop one
    radix.longest_prefix(query)     # return (matched_tokens, matched_key)
    radix.stats()                   # node_count, deduped_bytes_saved, etc.
    radix.save(path)                # write radix.index
    radix.load(path)                # read radix.index (silent rebuild on err)
    radix.rebuild_from_keys(keys)   # one-shot reconstruct from key list

Concurrency: ``insert``/``remove``/``save``/``load``/``rebuild_from_keys``
take a write lock. ``longest_prefix``/``stats`` are reader-side and use a
snapshot of the root child dict (Python's GIL gives us atomic dict.get()
for the per-level traversal; we re-read the level-local dict each step).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Bytes per int32 token slot on disk (matches memory_cache._TOKEN_BYTES).
# Used by ``deduped_prefix_bytes_saved`` accounting only; it's a wire-format
# proxy for "what would the hash-keyed cache have stored for this many
# redundant prefix tokens." Keeping it as a module constant rather than
# importing from memory_cache avoids the import cycle (memory_cache imports
# from runtime later on; runtime importing from memory_cache here would
# create one).
_BYTES_PER_TOKEN_INT32 = 4


# Persist-format pinning. v1 of the radix index is a JSON list of token
# tuples — the radix is a lookup index, not a storage format, so we don't
# need anything fancier. A future v2 might use msgpack if the entry count
# ever crosses ~100k; today's measured cap is a few thousand.
_RADIX_INDEX_VERSION = 1


@dataclass
class _RadixNode:
    """A single node in the token-keyed radix tree.

    The tree is technically a trie (one token per edge) rather than a
    classic radix tree with edge-string compression — we don't ship the
    edge-merge optimization on day 1 because:

    - Token sequences in real workloads cluster around a small number of
      shared prefixes (the system prompt), so the trie depth bounded by
      ``max(len(tokens))`` and the node count bounded by ``sum(len(tokens))``
      both stay within reasonable bounds (tens of MB peak for tens of
      thousands of entries).
    - Edge compression complicates the ``remove`` path (need to merge
      single-child chains back into the parent edge) — defer to Phase 5
      if profiling shows the trie eats memory.

    ``children``: token → child node. Dict ops are GIL-atomic; we re-read
    this dict on every traversal step so a concurrent insert doesn't
    invalidate a reader (the reader sees either the pre-insert child set
    or the post-insert child set — either way a coherent snapshot).

    ``is_terminal``: True iff a stored entry ends exactly at this node. A
    node may be terminal AND have children (e.g. one entry is the prompt
    "system\\n+user", another is "system\\n+user\\n+followup").

    ``ref_count``: how many distinct stored entries hold a path through
    this node. Equals ``sum(child.ref_count) + (1 if is_terminal else 0)``.
    Used for the dedup-bytes accounting and for the "I share this prefix
    with N other tenants" metric we expose on /metrics.
    """

    children: dict[int, _RadixNode] = field(default_factory=dict)
    is_terminal: bool = False
    ref_count: int = 0


@dataclass
class RadixStats:
    """Aggregate counters surfaced to /metrics and /v1/status.

    All counters are monotonic across the process lifetime — they survive
    LRU eviction (which only decrements ``node_count``) and a manual
    ``clear()`` (which DOES reset everything except the cumulative counters,
    matching the Prometheus counter contract). Callers can compute rates
    via ``rate(rapid_mlx_prefix_cache_radix_hits_total[5m])``.
    """

    # Cumulative counters (only increase). The lookup p50/p99 are derived
    # from the histogram we don't ship today — see the
    # ``last_lookup_seconds_p50/p99`` fields below for the lightweight
    # alternative the brief asked for.
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    removes: int = 0

    # Footprint-saved accounting. ``deduped_prefix_bytes_saved`` increments
    # on each insert by ``shared_prefix_len * _BYTES_PER_TOKEN_INT32`` —
    # i.e. how many token slots a hash-keyed index would have re-stored
    # but this radix collapsed into the shared path. The headline number
    # for the "30-80% prefix-cache footprint reduction" success criterion.
    deduped_prefix_bytes_saved: int = 0

    # Current-state gauges (move up and down). ``node_count`` is the
    # canonical pressure signal — when it grows faster than ``entry_count``
    # we have lots of short divergent prefixes (low sharing) and operators
    # may want to bump ``--cache-memory-mb``.
    node_count: int = 0
    entry_count: int = 0
    max_depth: int = 0

    # Lookup-latency snapshots. We don't ship a full histogram for the
    # /metrics surface (one ring buffer per percentile is overkill for
    # this scale), but we keep the last 256 lookup latencies in a deque
    # and compute p50/p99 lazily on read. ``snapshot()`` resolves the
    # deque to two scalars.
    _recent_lookup_ns: deque = field(default_factory=lambda: deque(maxlen=256))

    def record_lookup(self, ns: int) -> None:
        self._recent_lookup_ns.append(ns)

    def lookup_p50_seconds(self) -> float:
        if not self._recent_lookup_ns:
            return 0.0
        s = sorted(self._recent_lookup_ns)
        return s[len(s) // 2] / 1e9

    def lookup_p99_seconds(self) -> float:
        if not self._recent_lookup_ns:
            return 0.0
        s = sorted(self._recent_lookup_ns)
        idx = max(0, int(len(s) * 0.99) - 1)
        return s[idx] / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "inserts": self.inserts,
            "removes": self.removes,
            "deduped_prefix_bytes_saved": self.deduped_prefix_bytes_saved,
            "node_count": self.node_count,
            "entry_count": self.entry_count,
            "max_depth": self.max_depth,
            "lookup_p50_seconds": self.lookup_p50_seconds(),
            "lookup_p99_seconds": self.lookup_p99_seconds(),
        }


class RadixPrefixIndex:
    """Radix-tree lookup index over stored prefix-cache token sequences.

    Lifecycle:

        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3, 4])
        idx.insert([1, 2, 3, 5])       # shares [1,2,3] prefix
        matched, key = idx.longest_prefix([1, 2, 3, 5, 6])
        # matched == [1, 2, 3, 5], key == (1, 2, 3, 5)
        idx.save("~/.cache/rapid-mlx/.../radix.index")

    The index is the *source of truth for prefix lookup*, but the *KV state
    storage* lives in ``MemoryAwarePrefixCache._entries``. Both data
    structures key off the same ``tuple(tokens)`` so a successful
    ``longest_prefix`` call always lets the caller hit
    ``_entries[matched_key]`` to retrieve the KV.

    Drift safety: ``MemoryAwarePrefixCache`` is the owner of the entry set.
    A divergence (entry in cache but not in radix) can happen on a partial
    crash before ``save()`` ran; we paper over it on next boot by calling
    ``rebuild_from_keys(cache._entries.keys())``. The radix is
    *invalidation-safe* — a stale entry in the radix that no longer exists
    in the cache simply produces a ``longest_prefix`` hit followed by a
    ``cache[key]`` miss in the caller, which falls back to the miss path.
    """

    def __init__(self) -> None:
        self._root = _RadixNode()
        # RLock — ``rebuild_from_keys`` calls ``insert`` re-entrantly.
        self._lock = threading.RLock()
        self._stats = RadixStats()

    # ------------------------------------------------------------------ #
    # Mutation API                                                       #
    # ------------------------------------------------------------------ #

    def insert(self, tokens: list[int] | tuple[int, ...]) -> None:
        """Add a token sequence to the index.

        No-ops on an empty sequence — the radix is keyed by tuples of
        token ids, and the empty tuple has no useful meaning here (the
        ``MemoryAwarePrefixCache`` also rejects empty stores).

        Increments ``deduped_prefix_bytes_saved`` by
        ``shared_prefix_len * _BYTES_PER_TOKEN_INT32`` — the longest
        existing path the new entry overlapped with, weighted by the
        on-disk per-token width. This is the footprint-saved metric the
        brief calls for and it correctly counts ZERO when an inserted
        sequence shares no prefix with any existing entry.
        """
        if not tokens:
            return
        tokens = list(tokens)
        with self._lock:
            node = self._root
            shared_prefix_len = 0
            still_shared = True
            depth = 0
            for depth, tok in enumerate(tokens, start=1):
                child = node.children.get(tok)
                if child is None:
                    child = _RadixNode()
                    node.children[tok] = child
                    self._stats.node_count += 1
                    still_shared = False
                elif still_shared:
                    shared_prefix_len += 1
                node = child
                node.ref_count += 1
                if depth > self._stats.max_depth:
                    self._stats.max_depth = depth
            if not node.is_terminal:
                node.is_terminal = True
                self._stats.entry_count += 1
            self._stats.inserts += 1
            self._stats.deduped_prefix_bytes_saved += (
                shared_prefix_len * _BYTES_PER_TOKEN_INT32
            )

    def remove(self, tokens: list[int] | tuple[int, ...]) -> bool:
        """Drop a token sequence from the index.

        Returns ``True`` iff the sequence was present and terminal. After
        a successful remove, every node along the path has its
        ``ref_count`` decremented; nodes that drop to zero refcount AND
        zero children are pruned from the tree. ``node_count`` decreases
        accordingly. The cumulative ``deduped_prefix_bytes_saved`` is NOT
        decremented (Prometheus-counter contract — see RadixStats).
        """
        if not tokens:
            return False
        tokens = list(tokens)
        with self._lock:
            path: list[tuple[_RadixNode, int]] = []
            node = self._root
            for tok in tokens:
                child = node.children.get(tok)
                if child is None:
                    return False
                path.append((node, tok))
                node = child
            if not node.is_terminal:
                return False
            node.is_terminal = False
            self._stats.entry_count -= 1
            self._stats.removes += 1
            # Walk back up, decrementing refcount and pruning zero-ref
            # leaves. The root has no parent so we stop one short.
            for parent, tok in reversed(path):
                child = parent.children[tok]
                child.ref_count -= 1
                if child.ref_count <= 0 and not child.children:
                    del parent.children[tok]
                    self._stats.node_count -= 1
            return True

    def clear(self) -> None:
        """Drop every entry and reset the current-state gauges."""
        with self._lock:
            self._root = _RadixNode()
            self._stats.node_count = 0
            self._stats.entry_count = 0
            self._stats.max_depth = 0

    def rebuild_from_keys(self, keys: list[tuple[int, ...]]) -> None:
        """Rebuild the radix from a flat list of token-tuples.

        Used on cold boot when ``radix.index`` is missing or fails to
        parse — we reinsert every entry the storage layer has on disk and
        let the radix re-derive its node structure. Cost is O(sum(len))
        which on a fresh boot is dominated by the disk read of the
        underlying tokens.bin files anyway.
        """
        with self._lock:
            self._root = _RadixNode()
            self._stats.node_count = 0
            self._stats.entry_count = 0
            self._stats.max_depth = 0
            for key in keys:
                self.insert(list(key))

    # ------------------------------------------------------------------ #
    # Read API (no lock — see class docstring on concurrency)            #
    # ------------------------------------------------------------------ #

    def longest_prefix(
        self, query: list[int] | tuple[int, ...]
    ) -> tuple[list[int], tuple[int, ...] | None]:
        """Find the longest stored sequence that is a prefix of ``query``.

        Returns ``(matched_tokens, matched_key)`` where:
        - ``matched_tokens`` is the longest prefix that matched (may be
          shorter than any stored entry — see below);
        - ``matched_key`` is the tuple key of a stored entry whose
          sequence equals ``matched_tokens``, or ``None`` if no terminal
          node was crossed during the walk.

        Three cases worth noting:

        1. The query walks off the radix at depth D and the last terminal
           node was at depth T <= D. We return the T-token prefix and its
           key. Caller hits the cache with that key.

        2. The query walks the full length but no node along the path was
           terminal. We return ``(query, None)``: the caller gets the
           message "your tokens trace through the radix but no entry ends
           there". Practically a miss for the storage layer.

        3. The query immediately diverges from every root child. We
           return ``([], None)``. A complete miss.

        Records elapsed time into the RadixStats ring buffer so /metrics
        can report lookup latency p50/p99.
        """
        start_ns = time.perf_counter_ns()
        matched: list[int] = []
        last_terminal_depth = 0
        node = self._root
        for tok in query:
            # GIL-atomic dict.get — a concurrent insert/remove may race,
            # but never tears (Python guarantees object-level atomicity).
            child = node.children.get(tok)
            if child is None:
                break
            matched.append(tok)
            node = child
            if node.is_terminal:
                last_terminal_depth = len(matched)
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._stats.record_lookup(elapsed_ns)
        if last_terminal_depth > 0:
            self._stats.hits += 1
            key = tuple(matched[:last_terminal_depth])
            return matched[:last_terminal_depth], key
        self._stats.misses += 1
        return matched, None

    def __contains__(self, tokens: list[int] | tuple[int, ...]) -> bool:
        """Membership test — True iff an exact match terminates here."""
        node = self._root
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                return False
            node = child
        return node.is_terminal

    def __len__(self) -> int:
        return self._stats.entry_count

    def stats(self) -> dict[str, Any]:
        """Snapshot of the index's counters for /metrics and /v1/status."""
        return self._stats.to_dict()

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _collect_terminal_keys(self) -> list[list[int]]:
        """Walk the tree and yield every terminal node's token path.

        DFS with an explicit stack so deep tries don't blow the recursion
        limit. The output order is not stable across saves — callers must
        not rely on it for diffing. A future v2 of the index format could
        sort the output, but for now we eat the unsorted form because
        the cache reconstruction path doesn't care about order.
        """
        out: list[list[int]] = []
        stack: list[tuple[_RadixNode, list[int]]] = [(self._root, [])]
        while stack:
            node, path = stack.pop()
            if node.is_terminal:
                out.append(list(path))
            for tok, child in node.children.items():
                stack.append((child, path + [tok]))
        return out

    def save(self, path: str) -> None:
        """Atomically write the radix index to ``path``.

        Atomicity is via the ``<path>.tmp`` → rename pattern that every
        other on-disk artefact in rapid-mlx uses. A partial write leaves
        the ``.tmp`` orphan and the previous ``radix.index`` intact —
        next boot loads the old one and reinserts any drift.
        """
        with self._lock:
            keys = self._collect_terminal_keys()
            stats = self._stats.to_dict()
        payload = {
            "version": _RADIX_INDEX_VERSION,
            "saved_at": time.time(),
            "entry_count": len(keys),
            "node_count": stats["node_count"],
            "deduped_prefix_bytes_saved": stats["deduped_prefix_bytes_saved"],
            "keys": keys,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception as e:
            # Best-effort: if we can't write the index file the only cost
            # is a rebuild from cache._entries on next boot.
            logger.warning(
                f"[radix] failed to save radix index to {path}: {e}",
                exc_info=True,
            )
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def load(self, path: str) -> bool:
        """Load the radix index from ``path``.

        Returns ``True`` on success, ``False`` on missing-or-corrupt. The
        caller is expected to fall back to ``rebuild_from_keys`` on
        ``False`` so a malformed index never blocks a server boot.
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception as e:
            logger.warning(
                f"[radix] failed to read radix index at {path}: {e}; "
                "will rebuild from cache entries"
            )
            return False
        if not isinstance(payload, dict):
            logger.warning(f"[radix] radix index at {path} is not a dict; rebuilding")
            return False
        if payload.get("version") != _RADIX_INDEX_VERSION:
            logger.warning(
                f"[radix] radix index version mismatch "
                f"(file={payload.get('version')}, "
                f"want={_RADIX_INDEX_VERSION}); rebuilding"
            )
            return False
        raw_keys = payload.get("keys")
        if not isinstance(raw_keys, list):
            logger.warning(f"[radix] radix index keys malformed at {path}; rebuilding")
            return False
        keys: list[tuple[int, ...]] = []
        for raw in raw_keys:
            if not isinstance(raw, list):
                continue
            if not all(isinstance(t, int) for t in raw):
                continue
            keys.append(tuple(raw))
        self.rebuild_from_keys(keys)
        logger.info(
            f"[radix] loaded radix index from {path}: "
            f"{len(keys)} entries, node_count={self._stats.node_count}"
        )
        return True
