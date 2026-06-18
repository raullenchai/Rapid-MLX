# SPDX-License-Identifier: Apache-2.0
"""Tests for ``community-benchmarks/scripts/aggregate.py``.

What's actually being tested (in priority order):

1. **Deterministic output** — same submissions → byte-identical aggregate
   (modulo ``generated_at``). This is what makes the ``--check`` CI gate
   meaningful.
2. **Grouping correctness** — (chip, alias, version) is the key; rows
   that differ in RAM or GPU cores collapse into the same group; rows
   that differ in the key go to distinct groups.
3. **Statistical reduction** — median + IQR computed correctly for
   singletons (degenerate to point), 2-row groups, and 5-row groups.
4. **Schema-version forward compat** — v1 and v2 rows feed the same
   aggregator without one shape stealing fields from the other.
5. **``--check`` mode** — succeeds when aggregate matches, fails with
   a useful message when stale.

Not tested here: the schema validator itself (lives in
``community-benchmarks/scripts/validate.py`` with its own coverage) and
the HTML page (no behavior to assert beyond "loads aggregate"; the
selectors and arithmetic are simple enough that targeted assertions
would just pin layout).
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "community-benchmarks" / "scripts" / "aggregate.py"


def _load_module():
    """Load ``aggregate.py`` from its on-disk path.

    The script lives under ``community-benchmarks/scripts/`` which
    isn't a Python package — no ``__init__.py``, no entry in
    pyproject's package list. Import by file path instead of
    polluting sys.path or asking the test runner to know about the
    location.
    """
    spec = importlib.util.spec_from_file_location("aggregate", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


AGG = _load_module()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _row(
    *,
    sub_id: str,
    chip: str = "Apple M3 Ultra",
    ram_gb: int = 256,
    cpu: int = 28,
    gpu: int = 60,
    alias: str = "qwen3.5-9b-4bit",
    hf_path: str = "mlx-community/Qwen3.5-9B-4bit",
    version: str = "0.7.28",
    schema_version: int = 1,
    short_decode: float = 100.0,
    long_decode: float = 80.0,
    short_ttft: float = 200.0,
    long_ttft: float = 400.0,
    short_prefill: float = 1000.0,
    long_prefill: float = 1200.0,
) -> dict:
    """Minimal valid-ish submission row for grouping/aggregation tests.

    We don't validate against the schema here — the validator has its
    own coverage. What matters for the aggregator is the small subset
    of fields it reads (key fields + ``buckets.*.median``).
    """
    row = {
        "schema_version": schema_version,
        "submission_id": sub_id,
        "submitted_at": "2026-06-17T00:00:00+00:00",
        "hardware": {
            "chip": chip,
            "ram_gb": ram_gb,
            "cpu_cores": cpu,
            "gpu_cores": gpu,
        },
        "software": {
            "macos": "26.3.1",
            "rapid_mlx": version,
            "mlx": "0.31.3",
            "python": "3.12.13",
        },
        "model": {"alias": alias, "hf_path": hf_path},
        "config": {
            "rounds": 5,
            "warmup_rounds": 1,
            "sampling": "greedy",
            "buckets_spec": {
                "short": {"prompt_tokens": 512, "max_tokens": 128},
                "long": {"prompt_tokens": 2048, "max_tokens": 512},
            },
            "prompt_hash": "0123456789abcdef",
        },
        "buckets": {
            "short": {
                "decode_tps": {
                    "median": short_decode,
                    "min": short_decode,
                    "max": short_decode,
                    "stddev": 0.0,
                },
                "prefill_tps": {
                    "median": short_prefill,
                    "min": short_prefill,
                    "max": short_prefill,
                    "stddev": 0.0,
                },
                "ttft_ms": {
                    "median": short_ttft,
                    "min": short_ttft,
                    "max": short_ttft,
                    "stddev": 0.0,
                },
                "rounds_raw": [],
            },
            "long": {
                "decode_tps": {
                    "median": long_decode,
                    "min": long_decode,
                    "max": long_decode,
                    "stddev": 0.0,
                },
                "prefill_tps": {
                    "median": long_prefill,
                    "min": long_prefill,
                    "max": long_prefill,
                    "stddev": 0.0,
                },
                "ttft_ms": {
                    "median": long_ttft,
                    "min": long_ttft,
                    "max": long_ttft,
                    "stddev": 0.0,
                },
                "rounds_raw": [],
            },
        },
    }
    return row


def _write_rows(tmp: Path, rows: list[dict]) -> Path:
    """Drop a list of rows into ``tmp/submissions/*.json``.

    Filenames embed an index for deterministic on-disk ordering so the
    aggregator's ``sorted(...glob)`` step stays predictable in tests.
    """
    d = tmp / "submissions"
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        (d / f"row-{i:03d}-{r['submission_id']}.json").write_text(json.dumps(r))
    return d


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------


def test_percentile_singleton_collapses_to_value():
    """1-element list: p25 == p50 == p75 == value.

    Critical: today's real corpus has every group at row_count=1. A
    bug here would shape every singleton group as ``p25=NaN`` or
    similar and the entire frontend would render "—" for IQR.
    """
    assert AGG._percentile([42.0], 0.25) == 42.0
    assert AGG._percentile([42.0], 0.5) == 42.0
    assert AGG._percentile([42.0], 0.75) == 42.0


def test_percentile_two_values_linear_interp():
    """Linear-interp percentile on [10, 20]: p25 → 12.5, p75 → 17.5."""
    assert AGG._percentile([10.0, 20.0], 0.25) == pytest.approx(12.5)
    assert AGG._percentile([10.0, 20.0], 0.5) == pytest.approx(15.0)
    assert AGG._percentile([10.0, 20.0], 0.75) == pytest.approx(17.5)


def test_percentile_five_values_matches_numpy_default():
    """[1..5] type-7 quantile: p25=2, p50=3, p75=4. Same as ``np.quantile``."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert AGG._percentile(vals, 0.25) == pytest.approx(2.0)
    assert AGG._percentile(vals, 0.5) == pytest.approx(3.0)
    assert AGG._percentile(vals, 0.75) == pytest.approx(4.0)


def test_summarize_returns_all_three_stats():
    s = AGG._summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s == pytest.approx({"median": 30.0, "p25": 20.0, "p75": 40.0})


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_group_key_uses_chip_alias_version_only(tmp_path):
    """Rows that differ only in RAM/GPU/MLX-version land in the SAME group.

    The schema description is explicit: ``aggregates by (chip, model,
    rapid_mlx_version)``. RAM and GPU cores are bucketing axes for the
    raw store, not the aggregate — partitioning them would shatter
    every (chip family, model) cell into single-machine cells and the
    leaderboard becomes useless.
    """
    rows = [
        _row(sub_id="a1" * 6, ram_gb=64, gpu=20, short_decode=100.0),
        _row(sub_id="b2" * 6, ram_gb=128, gpu=32, short_decode=110.0),
        _row(sub_id="c3" * 6, ram_gb=256, gpu=60, short_decode=120.0),
    ]
    sub_dir = _write_rows(tmp_path, rows)
    out = AGG.build_aggregate(sub_dir)
    assert out["source_rows"] == 3
    assert len(out["groups"]) == 1
    g = out["groups"][0]
    assert g["row_count"] == 3
    assert g["short"]["decode_tps"]["median"] == pytest.approx(110.0)
    assert g["short"]["decode_tps"]["p25"] == pytest.approx(105.0)
    assert g["short"]["decode_tps"]["p75"] == pytest.approx(115.0)
    assert g["submission_ids"] == sorted(r["submission_id"] for r in rows)


def test_distinct_versions_become_distinct_groups(tmp_path):
    """Same machine, same model, different rapid_mlx versions → 2 groups.

    This is the regression-tracking story: ``v0.7.28`` improved on
    ``v0.7.27`` decode by N%, and the aggregator needs to PRESERVE
    that split so the website can show "this version got faster" not
    just "this hardware is fast".
    """
    rows = [
        _row(sub_id="a1" * 6, version="0.7.27", short_decode=100.0),
        _row(sub_id="b2" * 6, version="0.7.28", short_decode=120.0),
    ]
    sub_dir = _write_rows(tmp_path, rows)
    out = AGG.build_aggregate(sub_dir)
    assert len(out["groups"]) == 2
    versions = {g["rapid_mlx_version"]: g for g in out["groups"]}
    assert versions["0.7.27"]["short"]["decode_tps"]["median"] == pytest.approx(100.0)
    assert versions["0.7.28"]["short"]["decode_tps"]["median"] == pytest.approx(120.0)


def test_groups_are_sorted_for_determinism(tmp_path):
    """Output order is sorted (chip, alias, version) regardless of file order.

    Determinism is THE contract that makes ``--check`` meaningful. If
    the aggregator's output order depended on filesystem read order
    you'd get spurious CI failures on contributor PRs.
    """
    rows = [
        _row(sub_id="z9" * 6, alias="zeta", version="0.7.28"),
        _row(sub_id="a0" * 6, alias="alpha", version="0.7.28"),
        _row(sub_id="m5" * 6, alias="mu", version="0.7.27"),
    ]
    sub_dir = _write_rows(tmp_path, rows)
    out = AGG.build_aggregate(sub_dir)
    keys = [
        (g["chip"], g["model_alias"], g["rapid_mlx_version"]) for g in out["groups"]
    ]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Schema-version mixing
# ---------------------------------------------------------------------------


def test_schema_v1_and_v2_aggregate_together(tmp_path):
    """A v1 row and a v2 row that share the same (chip, alias, version)
    must collapse into ONE group — v2's optional ``smoke_result`` /
    ``harness_result`` fields don't change the aggregation key.

    Pins forward-compatibility: a future v3 row with a new optional
    field still aggregates correctly with v1/v2 rows as long as the
    speed bucket shape is unchanged.
    """
    rows = [
        _row(sub_id="a1" * 6, schema_version=1, short_decode=100.0),
        _row(sub_id="b2" * 6, schema_version=2, short_decode=120.0),
    ]
    sub_dir = _write_rows(tmp_path, rows)
    out = AGG.build_aggregate(sub_dir)
    assert len(out["groups"]) == 1
    assert out["groups"][0]["row_count"] == 2
    assert out["groups"][0]["short"]["decode_tps"]["median"] == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# Determinism and --check
# ---------------------------------------------------------------------------


def test_aggregate_is_deterministic(tmp_path):
    """Same rows on disk → byte-identical aggregate body. The only
    field that legitimately changes per run is ``generated_at``;
    everything else must hash to the same value.
    """
    rows = [
        _row(sub_id="a1" * 6, short_decode=100.0),
        _row(sub_id="b2" * 6, short_decode=120.0),
    ]
    sub_dir = _write_rows(tmp_path, rows)
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)

    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    AGG.write_aggregate(sub_dir, out1, now=fixed)
    AGG.write_aggregate(sub_dir, out2, now=fixed)

    assert out1.read_bytes() == out2.read_bytes(), (
        "Same input + same clock must produce byte-identical output. "
        "Non-determinism breaks the --check CI gate."
    )


def test_check_mode_passes_when_aggregate_matches(tmp_path):
    """Generate, then ``_check_freshness`` against the same dir → returns 0."""
    rows = [_row(sub_id="a1" * 6)]
    sub_dir = _write_rows(tmp_path, rows)
    out = tmp_path / "aggregated.json"
    AGG.write_aggregate(sub_dir, out)

    rc = AGG._check_freshness(sub_dir, out)
    assert rc == 0


def test_check_mode_fails_when_aggregate_is_stale(tmp_path, capsys):
    """Aggregate exists, then a NEW submission is added → ``--check`` fails."""
    rows = [_row(sub_id="a1" * 6, short_decode=100.0)]
    sub_dir = _write_rows(tmp_path, rows)
    out = tmp_path / "aggregated.json"
    AGG.write_aggregate(sub_dir, out)

    # Drop a second row directly into submissions/ WITHOUT regenerating
    # — this is the bug --check is designed to catch on contributor PRs.
    new_row = _row(sub_id="b2" * 6, short_decode=120.0)
    (sub_dir / "row-002-b2b2b2b2b2b2.json").write_text(json.dumps(new_row))

    rc = AGG._check_freshness(sub_dir, out)
    assert rc == 1
    captured = capsys.readouterr()
    assert "stale" in captured.err.lower()
    assert "aggregate.py" in captured.err  # Hint at the fix command


def test_check_mode_fails_when_aggregate_is_missing(tmp_path, capsys):
    """No on-disk aggregate at all → ``--check`` returns 1 with a hint."""
    rows = [_row(sub_id="a1" * 6)]
    sub_dir = _write_rows(tmp_path, rows)
    rc = AGG._check_freshness(sub_dir, tmp_path / "does-not-exist.json")
    assert rc == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


# ---------------------------------------------------------------------------
# Real corpus smoke test
# ---------------------------------------------------------------------------


def test_main_rejects_unknown_args(capsys):
    """Typos like ``--chek`` must exit non-zero, not silently regenerate.

    Before the round-2 fix, ``"--check" in argv[1:]`` silently routed
    any typo to the regenerate branch — a CI gate that runs
    ``aggregate.py --chek`` would clobber the artifact in CI instead
    of failing as the operator intended (codex PR #666 round-2 NIT).
    """
    rc = AGG.main(["aggregate.py", "--chek"])
    assert rc == 2, f"unknown arg must exit 2, got {rc}"
    captured = capsys.readouterr()
    assert "unrecognized" in captured.err.lower()
    assert "usage" in captured.err.lower()


def test_main_accepts_no_args_and_check(tmp_path, monkeypatch):
    """The two valid forms (no args / ``--check``) keep working after the
    strict-arg tightening. Drives ``main()`` against a tmp submissions
    dir to avoid touching the real artifact."""
    rows = [_row(sub_id="a1" * 6)]
    sub_dir = _write_rows(tmp_path, rows)
    out = tmp_path / "aggregated.json"

    monkeypatch.setattr(AGG, "SUBMISSIONS_DIR", sub_dir)
    monkeypatch.setattr(AGG, "AGGREGATE_PATH", out)
    # main() prints AGGREGATE_PATH.relative_to(REPO_ROOT) on success —
    # patch REPO_ROOT too so the tmp_path-based path stays inside the
    # patched root.
    monkeypatch.setattr(AGG, "REPO_ROOT", tmp_path)

    assert AGG.main(["aggregate.py"]) == 0
    assert out.exists()
    assert AGG.main(["aggregate.py", "--check"]) == 0


def test_real_corpus_committed_aggregate_is_fresh():
    """The on-disk ``community-benchmarks/aggregated.json`` matches the
    current ``submissions/`` directory. This catches the case where a
    submission lands but the aggregator wasn't re-run before commit.

    Identical check to what the CI workflow runs — covered here too so
    the failure is local on ``pytest`` instead of requiring a CI run.
    """
    rc = AGG._check_freshness(AGG.SUBMISSIONS_DIR, AGG.AGGREGATE_PATH)
    assert rc == 0, (
        "Committed aggregated.json is stale. Regenerate with: "
        "python community-benchmarks/scripts/aggregate.py"
    )
