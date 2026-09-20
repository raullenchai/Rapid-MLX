# SPDX-License-Identifier: Apache-2.0
"""Pin the cross-process contract of ``rapid_mlx.telemetry.store``.

The whole point of this store is that the CLI, a standalone ``serve``
and the desktop sidecar can run *at the same time* against one
``~/.rapid-mlx/telemetry.db``. A single-process test cannot show that,
so the two contracts that matter are exercised with real OS processes:

* ``record`` — N processes × M increments on one database must leave the
  count at exactly ``N*M``, and each bucket threshold must be reported by
  exactly one caller in total: no duplicate milestones, none missing.
* ``claim_active_day`` — exactly one ``True`` across all processes.

Everything else (failure policy, caps, boundaries) is cheap to test
in-process.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Enough contention that a non-atomic read-then-write loses updates
# reliably (verified by fault injection). 6 x 400 runs in well under
# a second.
CONCURRENCY_PROCESSES = 6
CONCURRENCY_INCREMENTS = 400


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Reroute ``Path.home()`` so the database lands under tmp.

    Same mechanism as ``tests/test_telemetry_state.py``: the store
    resolves the state dir at call time on purpose.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from rapid_mlx.telemetry import store

    store._reset_quarantine_latch_for_tests()
    return tmp_path


def _child_env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(home)
    # The worker imports rapid_mlx from wherever the parent found it.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return env


_RECORD_WORKER = """
import json, sys
from rapid_mlx.telemetry import store
key, n = sys.argv[1], int(sys.argv[2])
out = []
for _ in range(n):
    crossing = store.record(key)
    if crossing is not None:
        out.append(crossing.bucket)
print(json.dumps(out))
"""

# Each claimer does a single, very short piece of work, so without a
# rendezvous the processes simply do not overlap (interpreter start-up
# jitter alone separates them) and the race the test exists to catch
# never happens. Spin until a shared wall-clock deadline first.
_CLAIM_WORKER = """
import json, sys, time
from rapid_mlx.telemetry import store
start = float(sys.argv[2])
while time.time() < start:
    pass
print(json.dumps(store.claim_active_day(sys.argv[1])))
"""


def _run_workers(script: str, args_per_worker: list[list[str]], home: Path):
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_child_env(home),
            text=True,
        )
        for args in args_per_worker
    ]
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, f"worker failed: {err}"
        results.append(json.loads(out.strip().splitlines()[-1]))
    return results


# --------------------------------------------------------------------------
# Real multi-process concurrency
# --------------------------------------------------------------------------


def test_concurrent_record_never_loses_or_duplicates_a_milestone(fake_home):
    from rapid_mlx.telemetry import store

    total = CONCURRENCY_PROCESSES * CONCURRENCY_INCREMENTS
    results = _run_workers(
        _RECORD_WORKER,
        [["k", str(CONCURRENCY_INCREMENTS)]] * CONCURRENCY_PROCESSES,
        fake_home,
    )
    reported = [bucket for worker in results for bucket in worker]

    with sqlite3.connect(str(store.db_path())) as conn:
        (count,) = conn.execute("SELECT count FROM counters WHERE key = 'k'").fetchone()
    assert count == total, "lost increments — the transaction is not atomic"

    expected = [b for minimum, b in store.BUCKET_SPECS if minimum <= total]
    assert sorted(reported) == sorted(expected), (
        f"milestones wrong: got {sorted(reported)} want {sorted(expected)}"
    )
    assert len(reported) == len(set(reported)), "a milestone was emitted twice"


def test_concurrent_claim_active_day_has_exactly_one_winner(fake_home):
    start = f"{time.time() + 2.0:.3f}"
    results = _run_workers(_CLAIM_WORKER, [["2026-09-20", start]] * 8, fake_home)
    assert results.count(True) == 1
    assert results.count(False) == 7


# --------------------------------------------------------------------------
# record()
# --------------------------------------------------------------------------


def test_bucket_boundaries(fake_home):
    from rapid_mlx.telemetry import store

    first = store.record("k")
    assert first is not None
    assert (first.count, first.bucket, first.bucket_source) == (
        1,
        "1",
        "crossed_now",
    )

    second = store.record("k")
    assert second is not None and second.bucket == "2"

    third = store.record("k")
    assert third is not None and third.bucket == "3_4"
    # 4 is inside 3_4 — no second event for the same bucket.
    assert store.record("k") is None

    crossings = {}
    while True:
        crossing = store.record("k")
        if crossing is not None:
            crossings[crossing.bucket] = crossing.count
        if crossing is not None and crossing.bucket == "1000_plus":
            break
    assert crossings["1000_plus"] == 1000
    assert crossings["500_999"] == 500
    # Terminal bucket: counting past 1000 never emits again.
    for _ in range(50):
        assert store.record("k") is None


def test_bucket_for_below_first_bucket(fake_home):
    from rapid_mlx.telemetry import store

    assert store.bucket_for(0) is None
    assert store.bucket_for(1) == "1"
    assert store.bucket_for(999) == "500_999"
    assert store.bucket_for(1000) == "1000_plus"
    assert store.bucket_for(10**9) == "1000_plus"


def test_observed_existing_when_counter_predates_bucket_tracking(fake_home):
    """A pre-existing count reports the milestone, honestly labelled.

    Orca's rule: if we have never emitted a bucket for this key but the
    *previous* count was already in the bucket the increment lands in,
    the milestone is real but we did not watch it happen.
    """
    from rapid_mlx.telemetry import store

    store.record("k")  # creates the row, and the db
    with sqlite3.connect(str(store.db_path())) as conn:
        conn.execute(
            "UPDATE counters SET count = 7, last_bucket = NULL WHERE key = 'k'"
        )
    crossing = store.record("k")
    assert crossing is not None
    assert crossing.count == 8
    assert crossing.bucket == "5_9"
    assert crossing.bucket_source == "observed_existing"


def test_keys_are_independent(fake_home):
    from rapid_mlx.telemetry import store

    assert store.record("a").bucket == "1"
    assert store.record("b").bucket == "1"
    assert store.record("a").bucket == "2"


def test_key_length_cap(fake_home):
    from rapid_mlx.telemetry import store

    assert store.record("x" * store.MAX_KEY_LENGTH) is not None
    assert store.record("x" * (store.MAX_KEY_LENGTH + 1)) is None
    assert store.record("") is None


def test_distinct_key_cap(fake_home, monkeypatch):
    from rapid_mlx.telemetry import store

    monkeypatch.setattr(store, "MAX_KEYS", 5)
    for index in range(5):
        assert store.record(f"k{index}") is not None
    assert store.record("k5") is None
    # Existing keys keep counting past the cap.
    assert store.record("k0") is not None


# --------------------------------------------------------------------------
# claim_active_day()
# --------------------------------------------------------------------------


def test_claim_active_day_is_once_per_day(fake_home):
    from rapid_mlx.telemetry import store

    assert store.claim_active_day("2026-09-20") is True
    assert store.claim_active_day("2026-09-20") is False
    assert store.claim_active_day("2026-09-21") is True


def test_claim_active_day_prunes_old_rows(fake_home):
    from rapid_mlx.telemetry import store

    old = (
        datetime.now(timezone.utc) - timedelta(days=store.ACTIVE_DAY_RETENTION_DAYS + 5)
    ).date()
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).date()
    assert store.claim_active_day(old) is True
    assert store.claim_active_day(recent) is True
    assert store.claim_active_day(datetime.now(timezone.utc)) is True
    with sqlite3.connect(str(store.db_path())) as conn:
        days = {row[0] for row in conn.execute("SELECT day FROM active_days")}
    assert old.strftime("%Y-%m-%d") not in days
    assert recent.strftime("%Y-%m-%d") in days


def test_claim_active_day_defaults_to_today(fake_home):
    from rapid_mlx.telemetry import store

    assert store.claim_active_day() is True
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert store.claim_active_day(today) is False


# --------------------------------------------------------------------------
# Cohort stamp
# --------------------------------------------------------------------------


def test_note_model_served_counts_distinct_models(fake_home):
    from rapid_mlx.telemetry import store

    assert store.note_model_served("qwen3-8b") == 1
    assert store.note_model_served("qwen3-8b") == 1
    assert store.note_model_served("gemma-3-4b") == 2
    assert store.note_model_served("") == 0
    assert store.note_model_served("x" * (store.MAX_KEY_LENGTH + 1)) == 0


def test_note_model_served_respects_the_cap(fake_home, monkeypatch):
    from rapid_mlx.telemetry import store

    monkeypatch.setattr(store, "MAX_KEYS", 2)
    assert store.note_model_served("a") == 1
    assert store.note_model_served("b") == 2
    assert store.note_model_served("c") == 2


def test_first_run_date_is_written_once(fake_home):
    from rapid_mlx.telemetry import store

    day_one = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert store.first_run_date(day_one) == "2026-01-01"
    later = datetime(2026, 3, 9, tzinfo=timezone.utc)
    assert store.first_run_date(later) == "2026-01-01"


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (0, "0"),
        (1, "1"),
        (2, "2-6"),
        (6, "2-6"),
        (7, "7-29"),
        (29, "7-29"),
        (30, "30+"),
        (400, "30+"),
    ],
)
def test_days_since_first_run_bucket_boundaries(fake_home, days, expected):
    from rapid_mlx.telemetry import store

    first = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert store.first_run_date(first) == "2026-01-01"
    assert store.days_since_first_run_bucket(first + timedelta(days=days)) == expected


def test_days_since_first_run_bucket_tolerates_a_backwards_clock(fake_home):
    from rapid_mlx.telemetry import store

    first = datetime(2026, 5, 1, tzinfo=timezone.utc)
    store.first_run_date(first)
    assert store.days_since_first_run_bucket(first - timedelta(days=10)) == "0"


def test_cohort_values_are_in_the_declared_enum(fake_home):
    from rapid_mlx.telemetry import store

    first = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.first_run_date(first)
    for days in (0, 1, 3, 10, 90):
        assert (
            store.days_since_first_run_bucket(first + timedelta(days=days))
            in store.DAY_BUCKETS
        )


# --------------------------------------------------------------------------
# Failure policy: telemetry must never break `serve`
# --------------------------------------------------------------------------


def test_read_only_state_dir_returns_nothing_to_emit(fake_home):
    from rapid_mlx.telemetry import store

    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o500)
    try:
        assert store.record("k") is None
        assert store.claim_active_day("2026-09-20") is False
        assert store.note_model_served("qwen3-8b") == 0
        assert store.first_run_date() is None
        assert store.days_since_first_run_bucket() is None
    finally:
        os.chmod(state_dir, 0o700)


def test_read_only_home_returns_nothing_to_emit(tmp_path, monkeypatch):
    home = tmp_path / "ro-home"
    home.mkdir()
    os.chmod(home, 0o500)
    monkeypatch.setenv("HOME", str(home))
    from rapid_mlx.telemetry import store

    store._reset_quarantine_latch_for_tests()
    try:
        assert store.record("k") is None
        assert store.claim_active_day() is False
    finally:
        os.chmod(home, 0o700)


def test_corrupt_db_is_renamed_aside_once_and_recreated(fake_home):
    from rapid_mlx.telemetry import store

    assert store.record("k") is not None  # create a real db first
    store.db_path().write_bytes(b"this is not a database, not even close")

    crossing = store.record("k")
    assert crossing is not None
    # Recreated from scratch: the old counter went with the corrupt file.
    assert crossing.count == 1
    quarantined = list(fake_home.glob(".rapid-mlx/telemetry.db.corrupt-*"))
    assert len(quarantined) == 1

    # Second corruption in the same process is not renamed again (one
    # rename per process), and still never raises.
    store.db_path().write_bytes(b"corrupt again")
    assert store.record("k") is None


def test_db_file_is_private(fake_home):
    from rapid_mlx.telemetry import store

    store.record("k")
    assert (store.db_path().stat().st_mode & 0o777) == 0o600


def test_schema_version_is_stamped(fake_home):
    from rapid_mlx.telemetry import store

    store.record("k")
    with sqlite3.connect(str(store.db_path())) as conn:
        (value,) = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        (mode,) = conn.execute("PRAGMA journal_mode").fetchone()
    assert value == str(store.SCHEMA_VERSION)
    assert mode.lower() == "wal"


def test_db_is_created_lazily(fake_home):
    from rapid_mlx.telemetry import store

    assert not store.db_path().exists()
