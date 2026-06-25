#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Multi-tenant prefix-cache index bench (R15-P1, task #303).

Simulates the SGLang HiRadix benchmark spirit: N concurrent tenants each
issuing a request whose prompt is

    <shared_system_prompt> + <tenant-specific user message>

against the prefix-cache index. We measure:

- Aggregate request-handling rate (requests/sec)
- Per-request lookup latency p50/p99
- Index-storage footprint (cache "stored tokens" vs deduped accounting)

We do this WITHOUT booting a real server — running mlx for this bench
would (a) require a real model load (~2-20 GB), (b) entangle the
prefix-cache numbers with attention kernel throughput, and (c) take
minutes per run. The radix index is a *lookup data structure*, not a
generation kernel, so a pure index microbench cleanly isolates the
contribution of #303.

The reported "tok/s aggregate" is a synthetic — it counts the number of
*prompt tokens that an LLM would have skipped* per second thanks to a
cache hit. A radix hit on a 2k-token shared prefix is worth 2k "saved
prompt tokens", and at N=10 tenants/sec the aggregate is 20k/s. This is
the same metric vLLM and SGLang report under "prefix-cache TPS" — what
the model would have processed if the cache weren't there.

Usage:

    python bench/bench_radix_vs_hash.py
    python bench/bench_radix_vs_hash.py --tenants 20 --preamble 4096 --turns 5
    python bench/bench_radix_vs_hash.py --json   # machine-readable output

Reads ``--index radix|hash|both`` (default both) so the same script runs
both backends back-to-back and prints a side-by-side comparison.

The bench is deterministic (seeded RNG for tenant suffixes), so re-runs
on the same hardware produce stable numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from unittest.mock import MagicMock

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vllm_mlx.runtime.radix_index import RadixPrefixIndex


class _FakeCacheLayer:
    """Stand-in cache layer that mimics the KVCache interface enough for
    storage memory accounting; no MLX arrays so the bench runs CPU-only.

    Implements ``trim`` so the LCP / supersequence-trim fallback paths
    treat the layer as reusable across divergent queries (mirrors mlx-lm
    ``KVCache.trim`` semantics — only the method's existence is checked)."""

    def __init__(self):
        self.offset = 0
        self._payload = b"\x00" * 4096

    @property
    def state(self):
        return self._payload

    @property
    def meta_state(self):
        return (str(self.offset),)

    def trim(self, n: int) -> None:
        self.offset = max(0, self.offset - n)

    def is_trimmable(self) -> bool:
        return True


def _fake_cache(n_layers: int = 4):
    return [_FakeCacheLayer() for _ in range(n_layers)]


def _build_cache(index_kind: str) -> MemoryAwarePrefixCache:
    """Construct a MemoryAwarePrefixCache with the given index choice."""
    model = MagicMock()
    # Generous budget so eviction doesn't muddy the read path.
    config = MemoryCacheConfig(max_memory_mb=1024, max_entries=100_000)
    radix = RadixPrefixIndex() if index_kind == "radix" else None
    return MemoryAwarePrefixCache(model=model, config=config, radix_index=radix)


def _synthesize_tenants(
    n_tenants: int, preamble_len: int, user_msg_len: int, seed: int = 0xA734
) -> tuple[list[int], list[list[int]]]:
    """Generate one shared preamble plus N tenant-specific user messages."""
    rng = random.Random(seed)
    # Tokens 0..1023 reserved for "common vocabulary" in the preamble.
    preamble = [rng.randint(0, 1023) for _ in range(preamble_len)]
    tenant_msgs: list[list[int]] = []
    for tid in range(n_tenants):
        # Each tenant uses a distinct token range so the suffix diverges
        # immediately after the preamble (no accidental sharing).
        base = 10_000 + tid * 1_000
        tenant_msgs.append([base + rng.randint(0, 999) for _ in range(user_msg_len)])
    return preamble, tenant_msgs


def _run_workload(
    cache: MemoryAwarePrefixCache,
    preamble: list[int],
    tenant_msgs: list[list[int]],
    turns: int,
) -> dict:
    """Run the shared-system-prompt workload against ``cache``.

    First pass: store one entry per tenant (preamble + user_msg + fake
    "assistant_reply"). This is the cold-write phase.

    Subsequent ``turns`` passes: each tenant issues a NEW user message
    (preamble + new_msg) and we measure the fetch latency. This is the
    warm-read phase that dominates production traffic.

    Returns a dict with hits, misses, prompt_tokens_saved, latencies, etc.
    """
    # Cold-write phase: prime the cache with each tenant's initial turn.
    for tid, msg in enumerate(tenant_msgs):
        # Fake an "assistant reply" — 32 tokens of distinct content per tenant.
        reply = [50_000 + tid * 100 + i for i in range(32)]
        cache.store(preamble + msg + reply, _fake_cache())

    # Warm-read phase. Each tenant gets ``turns`` more requests, each
    # carrying the same preamble + a NEW user message. We expect a prefix
    # match on the preamble for every request.
    rng = random.Random(0xBEEF)
    fetch_latencies_ns: list[int] = []
    hits = 0
    misses = 0
    prompt_tokens_saved = 0
    started = time.perf_counter()
    total_requests = 0

    for turn in range(turns):
        for tid in range(len(tenant_msgs)):
            new_msg = [
                10_000 + tid * 1_000 + rng.randint(0, 999)
                for _ in range(len(tenant_msgs[0]))
            ]
            query = preamble + new_msg
            t0 = time.perf_counter_ns()
            kv, remaining = cache.fetch(query)
            t1 = time.perf_counter_ns()
            fetch_latencies_ns.append(t1 - t0)
            total_requests += 1
            if kv is not None:
                hits += 1
                prompt_tokens_saved += len(query) - len(remaining)
            else:
                misses += 1

    elapsed = time.perf_counter() - started
    fetch_latencies_ns.sort()
    return {
        "elapsed_seconds": elapsed,
        "total_requests": total_requests,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / max(1, total_requests),
        "prompt_tokens_saved": prompt_tokens_saved,
        "saved_tps": prompt_tokens_saved / max(1e-9, elapsed),
        "requests_per_sec": total_requests / max(1e-9, elapsed),
        "p50_lookup_us": fetch_latencies_ns[len(fetch_latencies_ns) // 2] / 1_000.0,
        "p99_lookup_us": fetch_latencies_ns[
            max(0, int(len(fetch_latencies_ns) * 0.99) - 1)
        ]
        / 1_000.0,
        "mean_lookup_us": statistics.fmean(fetch_latencies_ns) / 1_000.0,
        "cache_entries": len(cache._entries),
        "cache_memory_mb": cache._current_memory / (1024 * 1024),
    }


def _radix_footprint(cache: MemoryAwarePrefixCache) -> dict:
    """Pull the radix's dedup-bytes-saved + node count (None for hash mode)."""
    if cache._radix_index is None:
        return {
            "radix_dedup_bytes_saved": 0,
            "radix_node_count": 0,
            "radix_entry_count": 0,
        }
    s = cache._radix_index.stats()
    return {
        "radix_dedup_bytes_saved": s["deduped_prefix_bytes_saved"],
        "radix_node_count": s["node_count"],
        "radix_entry_count": s["entry_count"],
    }


def _run_one(index_kind: str, args) -> dict:
    """Run the full bench for one index choice and return the result dict."""
    cache = _build_cache(index_kind)
    preamble, tenant_msgs = _synthesize_tenants(
        n_tenants=args.tenants,
        preamble_len=args.preamble,
        user_msg_len=args.user_msg,
        seed=args.seed,
    )
    result = _run_workload(cache, preamble, tenant_msgs, turns=args.turns)
    result.update(_radix_footprint(cache))
    result["index"] = index_kind
    return result


def _print_human(result: dict) -> None:
    print(f"\n=== index={result['index']} ===")
    print(f"  total requests     : {result['total_requests']}")
    print(f"  hits / misses      : {result['hits']} / {result['misses']}")
    print(f"  hit rate           : {result['hit_rate'] * 100:.1f}%")
    print(f"  elapsed            : {result['elapsed_seconds']:.3f}s")
    print(f"  requests / sec     : {result['requests_per_sec']:.0f}")
    print(f"  prompt tokens saved: {result['prompt_tokens_saved']:,}")
    print(
        f"  aggregate saved tps: {result['saved_tps']:,.0f}  "
        "(prompt tokens NOT processed thanks to cache hits)"
    )
    print(
        f"  lookup latency p50 : {result['p50_lookup_us']:.2f}µs "
        f"| p99 : {result['p99_lookup_us']:.2f}µs "
        f"| mean : {result['mean_lookup_us']:.2f}µs"
    )
    print(f"  cache entries      : {result['cache_entries']}")
    print(f"  cache memory MB    : {result['cache_memory_mb']:.2f}")
    if result["index"] == "radix":
        print(
            f"  radix dedup bytes  : {result['radix_dedup_bytes_saved']:,}"
            f"   (≈{result['radix_dedup_bytes_saved'] / 1024:.1f}KB of "
            "redundant prefix tokens collapsed)"
        )
        print(
            f"  radix node count   : {result['radix_node_count']} "
            f"(vs {result['radix_entry_count']} entries — node/entry ratio "
            f"{result['radix_node_count'] / max(1, result['radix_entry_count']):.2f})"
        )


def _print_comparison(hash_r: dict, radix_r: dict) -> None:
    print("\n=== comparison (radix / hash) ===")
    speed_ratio = radix_r["saved_tps"] / max(1e-9, hash_r["saved_tps"])
    rps_ratio = radix_r["requests_per_sec"] / max(1e-9, hash_r["requests_per_sec"])
    p50_speedup = hash_r["p50_lookup_us"] / max(1e-9, radix_r["p50_lookup_us"])
    p99_speedup = hash_r["p99_lookup_us"] / max(1e-9, radix_r["p99_lookup_us"])
    print(f"  aggregate saved-tps ratio : {speed_ratio:.2f}×")
    print(f"  requests/sec ratio        : {rps_ratio:.2f}×")
    print(f"  lookup p50 speedup        : {p50_speedup:.2f}×")
    print(f"  lookup p99 speedup        : {p99_speedup:.2f}×")
    if radix_r["radix_dedup_bytes_saved"] > 0:
        # Estimate footprint reduction. A hash-keyed index would have
        # carried len(preamble) tokens for EACH stored entry; the radix
        # collapsed dedup_bytes_saved of those into shared nodes.
        equivalent_full = (
            radix_r["radix_dedup_bytes_saved"] + radix_r["radix_node_count"] * 4
        )
        reduction_pct = (
            radix_r["radix_dedup_bytes_saved"] / max(1, equivalent_full) * 100
        )
        print(f"  estimated footprint cut   : ~{reduction_pct:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenants", type=int, default=10, help="N concurrent tenants")
    ap.add_argument(
        "--preamble",
        type=int,
        default=2048,
        help="Shared system prompt length in tokens (default 2048)",
    )
    ap.add_argument(
        "--user-msg",
        type=int,
        default=64,
        help="Per-tenant user message length in tokens (default 64)",
    )
    ap.add_argument(
        "--turns",
        type=int,
        default=20,
        help="Warm-read passes per tenant (default 20)",
    )
    ap.add_argument(
        "--index",
        choices=("radix", "hash", "both"),
        default="both",
        help="Index implementation under test (default both)",
    )
    ap.add_argument("--seed", type=int, default=0xA734, help="RNG seed")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable lines",
    )
    args = ap.parse_args()

    results: dict[str, dict] = {}
    if args.index in ("hash", "both"):
        results["hash"] = _run_one("hash", args)
    if args.index in ("radix", "both"):
        results["radix"] = _run_one("radix", args)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for r in results.values():
        _print_human(r)
    if "hash" in results and "radix" in results:
        _print_comparison(results["hash"], results["radix"])


if __name__ == "__main__":
    main()
