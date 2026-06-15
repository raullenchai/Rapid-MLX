#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate raw submissions into ``aggregated.json`` for the website.

Run:

    python community-benchmarks/scripts/aggregate.py

Reads every JSON file under ``community-benchmarks/submissions/`` and
groups by the bucketing key (``chip``, ``ram_gb``, ``model.alias``,
``software.rapid_mlx``, ``config.sampling``), then computes
``median / min / max / p25 / p75 / count`` per metric per workload
bucket. Writes the result to ``community-benchmarks/aggregated.json``.

Design notes:

- **Stdlib only.** The website / CI should be able to regenerate
  ``aggregated.json`` from a clean checkout without ``pip install``.
- **Fault-tolerant.** A single malformed submission file logs a warning
  and is skipped; the aggregator never crashes mid-run. The validator
  workflow is what gates malformed files at PR time — this script is
  defense in depth.
- **Forward-compatible.** Submissions with ``schema_version`` newer
  than what this script understands are skipped (with a warning) rather
  than blowing up. Old submissions are kept until their ``schema_version``
  drops off the supported list.
- **Median-of-medians.** Each submission already carries a per-bucket
  median across 5 rounds; the aggregator takes the median of those
  medians across submissions. This dampens the influence of one user's
  thermally-throttled outlier without discarding it (the raw file is
  still in submissions/ and the count is reported).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = REPO_ROOT / "community-benchmarks" / "submissions"
OUTPUT_PATH = REPO_ROOT / "community-benchmarks" / "aggregated.json"

# Schema versions this script knows how to read. When we bump the
# schema, add the new version here AND keep the old one until enough
# submitters have migrated.
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})


def _load_all(paths: Iterable[Path]) -> list[dict]:
    """Read every JSON file, skipping unreadable / unsupported ones."""
    out: list[dict] = []
    for p in paths:
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARN: skipping {p.name}: {e}", file=sys.stderr)
            continue
        ver = payload.get("schema_version")
        if ver not in SUPPORTED_SCHEMA_VERSIONS:
            print(
                f"  WARN: skipping {p.name}: schema_version={ver!r} not in "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}",
                file=sys.stderr,
            )
            continue
        out.append(payload)
    return out


def _bucket_key(payload: dict) -> tuple[str, int, str, str, str]:
    """Stable grouping key for the aggregator.

    Includes ``rapid_mlx`` version because the same chip × model can
    move 10-20% between Rapid-MLX releases — collapsing across versions
    would hide regressions. Sampling is also a key because greedy and
    sampled aren't comparable.
    """
    return (
        payload["hardware"]["chip"],
        payload["hardware"]["ram_gb"],
        payload["model"]["alias"],
        payload["software"]["rapid_mlx"],
        payload["config"]["sampling"],
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (numpy's default method).

    ``sorted_values`` must already be sorted ascending. ``pct`` in
    ``[0, 100]``. Single-element lists return that element for every
    percentile (no interpolation possible).
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _agg(values: list[float]) -> dict:
    """``{median, min, max, p25, p75, count}`` over a list of medians.

    Empty list yields zeros + count=0 — the caller should filter these
    out before writing, but we don't crash on them.
    """
    if not values:
        return {
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "count": 0,
        }
    sorted_v = sorted(values)
    return {
        "median": float(statistics.median(sorted_v)),
        "min": float(sorted_v[0]),
        "max": float(sorted_v[-1]),
        "p25": _percentile(sorted_v, 25.0),
        "p75": _percentile(sorted_v, 75.0),
        "count": len(sorted_v),
    }


def _aggregate(payloads: list[dict]) -> list[dict]:
    """Group + aggregate. Returns a sorted list of rows.

    One row = one (chip, ram_gb, alias, rapid_mlx, sampling) combination
    with aggregated stats per bucket per metric. Sorted by (chip, alias)
    so a hand-readable diff of ``aggregated.json`` makes sense.
    """
    by_key: dict[tuple, list[dict]] = {}
    for p in payloads:
        by_key.setdefault(_bucket_key(p), []).append(p)

    rows: list[dict] = []
    for key, group in by_key.items():
        chip, ram_gb, alias, rapid_mlx, sampling = key
        row: dict = {
            "chip": chip,
            "ram_gb": ram_gb,
            "model_alias": alias,
            "rapid_mlx_version": rapid_mlx,
            "sampling": sampling,
            "sample_count": len(group),
            # ``hf_path`` is denormalized to make the row self-contained
            # for the website; we take it from the most recent submission
            # in the group (they should all match, but the latest one
            # wins in case of any divergence).
            "hf_path": sorted(
                group, key=lambda p: p["submitted_at"]
            )[-1]["model"]["hf_path"],
            "buckets": {},
        }
        for bucket_name in ("short", "long"):
            bucket_agg: dict = {}
            for metric in ("decode_tps", "prefill_tps", "ttft_ms"):
                medians = [
                    p["buckets"][bucket_name][metric]["median"] for p in group
                ]
                bucket_agg[metric] = _agg(medians)
            row["buckets"][bucket_name] = bucket_agg
        rows.append(row)

    rows.sort(key=lambda r: (r["chip"], r["model_alias"], r["rapid_mlx_version"]))
    return rows


def main(argv: list[str]) -> int:
    if not SUBMISSIONS_DIR.exists():
        print(f"  No submissions directory at {SUBMISSIONS_DIR}; nothing to do.")
        return 0
    paths = sorted(SUBMISSIONS_DIR.glob("*.json"))
    if not paths:
        print("  No submissions to aggregate.")
        # Still write an empty file so the website never has to handle
        # "404 aggregated.json" — empty list is the explicit "nothing yet"
        # signal.
        OUTPUT_PATH.write_text(
            json.dumps({"rows": [], "submission_count": 0}, indent=2) + "\n"
        )
        return 0

    payloads = _load_all(paths)
    rows = _aggregate(payloads)
    OUTPUT_PATH.write_text(
        json.dumps(
            {"rows": rows, "submission_count": len(payloads)},
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    print(
        f"  Aggregated {len(payloads)} submissions into {len(rows)} rows "
        f"→ {OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
