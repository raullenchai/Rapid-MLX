#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic repro for Talia r12 SEVERE finding: 2-cycle SIGTERM
round-trip produces inconsistent (save_uuid mismatch + length-prefix
mismatch) tokens.bin vs index.json.

Boots the cache in-process (no server), saves with deadline ≈3.5s
(matching prod), back-to-back. The first save mirrors a long-running
production save: 100 entries × varied length. The second save adds
a couple of new entries, then re-saves the same dir.

Run from repo root:

    python scripts/repro_r12_atomicity.py [cache_dir]

Exits non-zero if any tokens.bin disagrees with index.json on the
second-cycle load (save_uuid mismatch OR length-prefix drift) — the
exact failure mode Talia observed on probe-5 cycle 1 of dogfood r12.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

# Make repo root importable when run from the script dir directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pull mlx-lm primitives the way the production save_to_disk does.
import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402

from vllm_mlx.memory_cache import (  # noqa: E402
    _TOKENS_HEADER_FIXED_LEN,
    _TOKENS_MAGIC,
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)


def make_kvcache(num_tokens: int, *, n_layers: int = 2, fill: float = 1.0) -> list:
    layers = []
    for layer_idx in range(n_layers):
        c = KVCache()
        keys = mx.full((1, 4, num_tokens, 8), fill + layer_idx, dtype=mx.float16)
        values = mx.full((1, 4, num_tokens, 8), -(fill + layer_idx), dtype=mx.float16)
        c.update_and_fetch(keys, values)
        layers.append(c)
    return layers


def fresh_cache() -> MemoryAwarePrefixCache:
    return MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=4096, max_entries=200),
    )


def parse_token_bin_header(path: Path) -> tuple[int, str]:
    """Return (token_count, save_uuid_hex) from a v3 tokens.bin file."""
    raw = path.read_bytes()
    assert raw.startswith(_TOKENS_MAGIC), f"{path}: not v3 (missing magic)"
    token_count, uuid_len = struct.unpack(
        "<II", raw[len(_TOKENS_MAGIC) : _TOKENS_HEADER_FIXED_LEN]
    )
    uuid_bytes = raw[_TOKENS_HEADER_FIXED_LEN : _TOKENS_HEADER_FIXED_LEN + uuid_len]
    return token_count, uuid_bytes.decode("ascii")


def assert_consistent(cache_dir: Path, cycle: int) -> None:
    """Walk every entry_K_tokens.bin and check its (count, uuid) match index.json."""
    idx = json.loads((cache_dir / "index.json").read_text())
    idx_uuid = idx.get("save_uuid")
    print(f"  cycle {cycle}: index.json save_uuid = {idx_uuid}")
    print(f"  cycle {cycle}: index.json claims {idx['num_entries']} entries")
    bad = []
    for entry in idx["entries"]:
        i = entry["index"]
        expected_count = entry["num_tokens"]
        tb = cache_dir / f"entry_{i}_tokens.bin"
        try:
            count, uuid = parse_token_bin_header(tb)
        except Exception as exc:
            bad.append((i, f"parse failed: {exc}"))
            continue
        if uuid != idx_uuid:
            bad.append(
                (
                    i,
                    f"save_uuid mismatch: tokens.bin={uuid!r}, index={idx_uuid!r}",
                )
            )
        if count != expected_count:
            bad.append(
                (
                    i,
                    f"length-prefix mismatch: tokens.bin says {count}, "
                    f"index.json says {expected_count}",
                )
            )
    if bad:
        print(
            f"  cycle {cycle}: FAIL — {len(bad)} of {len(idx['entries'])} entries "
            "inconsistent with index.json:"
        )
        for i, reason in bad[:10]:
            print(f"      entry {i}: {reason}")
        if len(bad) > 10:
            print(f"      … and {len(bad) - 10} more")
        raise SystemExit(1)
    print(f"  cycle {cycle}: OK — every entry's (uuid, length-prefix) matches index")


def run(cache_dir: Path, n_first: int = 100, n_added: int = 20) -> None:
    print(f"Repro target: {cache_dir}")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    for suffix in (".new", ".old"):
        sib = cache_dir.with_suffix(cache_dir.suffix + suffix)
        if sib.exists():
            shutil.rmtree(sib)

    # --- cycle 1: populate from cold, save, exit ---
    print(f"\n=== cycle 1: cold start, {n_first} entries ===")
    c1 = fresh_cache()
    for i in range(n_first):
        toks = list(range(i * 1000, i * 1000 + 10 + (i % 5)))
        c1.store(toks, make_kvcache(num_tokens=len(toks), fill=float(i + 1)))
    assert c1.save_to_disk(str(cache_dir)) is True
    assert_consistent(cache_dir, 1)

    # --- cycle 2: load + add a few entries, save, exit ---
    # This is the cycle where Talia saw the corruption land on the
    # NEXT boot (cycle 3) — but the producer is cycle 2's save.
    print(f"\n=== cycle 2: load + add {n_added}, save ===")
    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    print(f"  loaded {loaded} from cycle 1")
    assert loaded == n_first, f"cycle 2 load: {loaded} != {n_first}"
    for j in range(n_added):
        toks = list(range(900_000 + j * 100, 900_000 + j * 100 + 12))
        c2.store(toks, make_kvcache(num_tokens=len(toks), fill=float(j + 200)))
    assert c2.save_to_disk(str(cache_dir)) is True
    assert_consistent(cache_dir, 2)

    # --- cycle 3: load — Talia's "LOADED 0 entries SKIPPED 100" landed here ---
    print("\n=== cycle 3: load from cycle 2 save ===")
    c3 = fresh_cache()
    loaded = c3.load_from_disk(str(cache_dir))
    print(f"  loaded {loaded} entries from cycle 2 save")
    stats = c3.get_stats()
    print(f"  load_skipped (corrupt): {stats['load_skipped']}")
    if stats["load_skipped"] > 0:
        print(f"REPRODUCED: {stats['load_skipped']} entries rejected as corrupt")
        raise SystemExit(2)
    assert loaded == n_first + n_added, f"cycle 3 load: {loaded} != {n_first + n_added}"
    print("\nALL CONSISTENT — no repro under this scenario")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cache_dir",
        nargs="?",
        default=None,
        help="cache dir (default: tmp)",
    )
    parser.add_argument("--n-first", type=int, default=100)
    parser.add_argument("--n-added", type=int, default=20)
    args = parser.parse_args()
    if args.cache_dir:
        run(Path(args.cache_dir), args.n_first, args.n_added)
    else:
        with tempfile.TemporaryDirectory() as td:
            run(Path(td) / "snap", args.n_first, args.n_added)


if __name__ == "__main__":
    main()
