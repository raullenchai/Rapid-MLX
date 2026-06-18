#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate community-submitted benchmark rows into a sortable table.

Reads every JSON file under ``community-benchmarks/submissions/``,
groups by ``(chip, model_alias, rapid_mlx_version)`` (the axis the
schema description names), computes per-metric median + IQR
(25th / 75th percentile) within each group, and writes
``community-benchmarks/aggregated.json`` — the single static artifact
external consumers (rapidmlx.com Performance tab; this directory's own
``index.html``) fetch to render the leaderboard.

Stdlib only — runs in CI without any pip install.

Run modes::

    # Regenerate the aggregate. Exit 0 on clean write, 1 on any
    # validation error in the raw store (the validator runs separately
    # in CI; this script trusts what's in submissions/).
    python community-benchmarks/scripts/aggregate.py

    # Verify the on-disk aggregate matches what's regenerated from
    # current submissions. Used by the CI freshness check — fails if
    # the committed aggregate is stale vs. the current raw rows.
    python community-benchmarks/scripts/aggregate.py --check

Output shape (also documented in the file header that gets written)::

    {
      "schema_version": 1,
      "generated_at": "<UTC ISO-8601>",
      "source_rows": <int>,
      "groups": [
        {
          "chip": "Apple M3 Ultra",
          "model_alias": "qwen3-8b-4bit",
          "rapid_mlx_version": "0.7.26",
          "row_count": <int>,
          "submission_ids": [...],
          "short": {
            "decode_tps":  {"median": ..., "p25": ..., "p75": ...},
            "prefill_tps": {...},
            "ttft_ms":     {...}
          },
          "long": {...}
        },
        ...
      ]
    }

Aggregation is intentionally pure-function: same input → same output.
That makes the ``--check`` mode meaningful as a CI gate.

Schema versions: v1 and v2 carry the same speed-bucket shape. v2's
extra ``smoke_result`` / ``harness_result`` fields are NOT included in
the aggregate today — they're per-row narrative data, not numeric
distributions we'd want to median across rows. Add a follow-up if/when
we want a "harness pass rate" column. See discussion in #663 / the
schema.json comment block.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = REPO_ROOT / "community-benchmarks" / "submissions"
AGGREGATE_PATH = REPO_ROOT / "community-benchmarks" / "aggregated.json"

# Metrics we surface per bucket. Keep this list in lockstep with what
# the schema's ``bucketResult`` defines under ``properties`` — adding a
# field there without adding it here is silent loss of signal in the
# aggregate; the converse just means the field shows up empty in the
# UI table (better failure mode).
BUCKETED_METRICS: tuple[str, ...] = ("decode_tps", "prefill_tps", "ttft_ms")
BUCKETS: tuple[str, ...] = ("short", "long")

# Aggregate schema version. Bump on incompatible shape changes (e.g.
# adding harness pass-rate columns). Consumers should skip aggregates
# they don't understand the same way they skip raw submissions.
AGGREGATE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, no external deps.

    For a 1-element list, ``p25 == p50 == p75 == value`` — important
    because the current corpus has every group at row_count=1 and we
    don't want the aggregate to mis-shape on the singleton case.
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    # NumPy "linear" / type-7 quantile: index = (n-1) * pct
    idx = (len(s) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _summarize(values: list[float]) -> dict[str, float]:
    """Median + IQR (p25, p75) for one metric across rows of a group.

    IQR — not min/max — because outliers in a crowdsourced corpus are
    expected (thermal throttle, machine under load, etc.); the
    aggregator should de-weight them in the displayed numbers rather
    than amplify them. The raw store keeps every row so anyone who
    wants the full distribution can compute it from there.
    """
    return {
        "median": float(median(values)),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
    }


# ---------------------------------------------------------------------------
# Row loading and grouping
# ---------------------------------------------------------------------------


def _iter_rows(submissions_dir: Path) -> Iterable[tuple[Path, dict]]:
    """Yield (path, parsed_json) for every submission file, sorted by name.

    Sort by filename (which embeds the submission date prefix) so the
    aggregate's row-ordering inside ``submission_ids`` is deterministic
    — needed so ``--check`` can compare byte-for-byte against the
    committed artifact.
    """
    for path in sorted(submissions_dir.glob("*.json")):
        with path.open() as f:
            yield path, json.load(f)


def _group_key(row: dict) -> tuple[str, str, str]:
    """The (chip, model_alias, rapid_mlx_version) tuple the schema
    description specifies as the aggregation axis. NOT including RAM /
    GPU cores / MLX version: those vary within the same chip and the
    aggregate's job is to show the chip's median, not partition it
    into single-machine cells (the raw store can answer those finer
    questions).
    """
    return (
        row["hardware"]["chip"],
        row["model"]["alias"],
        row["software"]["rapid_mlx"],
    )


def _collect_metric_values(rows: list[dict], bucket: str, metric: str) -> list[float]:
    """Pull this (bucket, metric)'s per-row median out of each row.

    The raw row already medianed across its 5 internal rounds; we
    median THAT across rows of a group. Two-stage aggregation. Picking
    the row-level median (rather than raw rounds) keeps users with
    unstable rigs from skewing the group: their per-row median already
    reflects their own jitter, and we don't double-count it.
    """
    out: list[float] = []
    for r in rows:
        bucket_data = (r.get("buckets") or {}).get(bucket)
        if not bucket_data:
            continue
        metric_data = bucket_data.get(metric)
        if not metric_data:
            continue
        med = metric_data.get("median")
        if med is None:
            continue
        out.append(float(med))
    return out


def _aggregate_groups(rows: list[dict]) -> list[dict[str, Any]]:
    """Group rows and reduce to per-group summary objects.

    Output is sorted by (chip, model_alias, rapid_mlx_version) so the
    aggregate is deterministic across runs — critical for the
    ``--check`` CI gate.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        chip, alias, version = key
        entry: dict[str, Any] = {
            "chip": chip,
            "model_alias": alias,
            "rapid_mlx_version": version,
            "row_count": len(group_rows),
            "submission_ids": sorted(r["submission_id"] for r in group_rows),
        }
        for bucket in BUCKETS:
            bucket_out: dict[str, dict[str, float]] = {}
            for metric in BUCKETED_METRICS:
                values = _collect_metric_values(group_rows, bucket, metric)
                if values:
                    bucket_out[metric] = _summarize(values)
            if bucket_out:
                entry[bucket] = bucket_out
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def build_aggregate(submissions_dir: Path = SUBMISSIONS_DIR) -> dict[str, Any]:
    """Pure function: raw submissions on disk → aggregate dict.

    Doesn't write anything. The split exists so unit tests can drive
    it with a tmpdir of fixture submissions and assert on the returned
    structure without touching the real artifact path.

    ``generated_at`` is set by the caller (``write_aggregate``) — pure
    functions don't read a clock — so the test path can substitute a
    deterministic timestamp instead of letting ``datetime.now`` make
    the assertion flaky.
    """
    rows = [r for _, r in _iter_rows(submissions_dir)]
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "generated_at": None,
        "source_rows": len(rows),
        "groups": _aggregate_groups(rows),
    }


def write_aggregate(
    submissions_dir: Path = SUBMISSIONS_DIR,
    output_path: Path = AGGREGATE_PATH,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the aggregate and write it to ``output_path``. Returns the dict.

    Indent=2 + trailing newline + sort_keys=False (groups are already
    sorted; the field-order inside each group is intentional for
    readability when the file is browsed on GitHub) so diffs across
    submissions stay small and reviewable.
    """
    data = build_aggregate(submissions_dir)
    data["generated_at"] = (now or datetime.now(timezone.utc)).isoformat()
    output_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    return data


def _check_freshness(submissions_dir: Path, output_path: Path) -> int:
    """``--check`` mode: verify committed aggregate matches current submissions.

    Compares the structural parts of the JSON (everything except
    ``generated_at``, which legitimately changes every run). Returns 0
    if matched, 1 if stale.
    """
    if not output_path.exists():
        print(
            f"ERROR: {output_path} does not exist — run "
            f"{Path(__file__).name} (no args) to generate.",
            file=sys.stderr,
        )
        return 1

    expected = build_aggregate(submissions_dir)
    with output_path.open() as f:
        actual = json.load(f)

    # Compare the parts that depend on the input. ``generated_at`` is
    # explicitly excluded — it's a write-time-only field, not derived
    # from inputs.
    expected_cmp = {k: v for k, v in expected.items() if k != "generated_at"}
    actual_cmp = {k: v for k, v in actual.items() if k != "generated_at"}

    if expected_cmp == actual_cmp:
        return 0

    print(
        f"ERROR: {output_path} is stale. Regenerate with:\n"
        f"  python community-benchmarks/scripts/aggregate.py",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str]) -> int:
    if "--check" in argv[1:]:
        return _check_freshness(SUBMISSIONS_DIR, AGGREGATE_PATH)
    data = write_aggregate()
    print(
        f"Wrote {AGGREGATE_PATH.relative_to(REPO_ROOT)}: "
        f"{data['source_rows']} rows → {len(data['groups'])} groups"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
