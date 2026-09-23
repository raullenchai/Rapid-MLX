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
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

# Enough contention that a non-atomic read-then-write loses updates
# reliably (verified by fault injection). 6 x 400 runs in well under
# a second.
CONCURRENCY_PROCESSES = 6
CONCURRENCY_INCREMENTS = 400


#: ``chmod 0500`` does not stop root, so the two tests that make a real
#: directory unwritable are skipped there. The same code paths are also
#: covered deterministically by
#: ``test_unwritable_state_dir_returns_nothing_to_emit``, which fails the
#: ``mkdir`` directly — so skipping here never opens a coverage hole.
requires_non_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions",
)


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


# Workers report the number of calls that returned ``None`` as well as
# the milestones they saw. A ``None`` is either "stayed inside the
# bucket" (expected, the common case) or "storage failed" (a lost
# increment). Only the total count can tell them apart, so the worker
# reports the raw call count too and the test does the arithmetic —
# otherwise a loaded CI box that trips the 2 s busy timeout fails with
# "lost increments — the transaction is not atomic", which is a
# diagnosis, and the wrong one.
_RECORD_WORKER = """
import json, sys
from rapid_mlx.telemetry import store
key, n = sys.argv[1], int(sys.argv[2])
out = []
for _ in range(n):
    crossing = store.record(key)
    if crossing is not None:
        out.append(crossing.bucket)
print(json.dumps({"buckets": out, "calls": n}))
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
    reported = [bucket for worker in results for bucket in worker["buckets"]]
    attempted = sum(worker["calls"] for worker in results)

    with sqlite3.connect(str(store.db_path())) as conn:
        (count,) = conn.execute("SELECT count FROM counters WHERE key = 'k'").fetchone()

    # Two different failures both show up as "count < attempted", and
    # they call for opposite responses: a dropped write under contention
    # is a broken transaction, while every call simply returning
    # "nothing to emit" would mean the store never worked here at all.
    # Name which one happened.
    assert count > 0, (
        "the store recorded nothing at all — every call failed, so this "
        "says nothing about atomicity"
    )
    assert count == total, (
        f"lost increments: {attempted} calls left the counter at {count}, "
        f"want {total}. Every call is one BEGIN IMMEDIATE transaction, so a "
        f"shortfall here is a read-then-write interleaving, not contention: "
        f"a call that loses the write lock raises and returns None without "
        f"incrementing, which cannot move the counter backwards."
    )

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


def test_claim_active_day_prunes_rows_by_write_time(fake_home):
    """The table stays bounded, and the clock that bounds it is write time.

    Retention cannot key off the day a row *names* — see
    ``test_claim_active_day_is_once_per_day_for_a_date_past_the_cutoff`` for what that
    breaks. So age a row the only way that is now meaningful: rewrite
    its ``claimed_at`` to long ago, then make a fresh claim and watch it
    go.
    """
    from rapid_mlx.telemetry import store

    assert store.claim_active_day("2026-01-01") is True
    aged = (
        datetime.now(timezone.utc) - timedelta(days=store.ACTIVE_DAY_RETENTION_DAYS + 5)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    with sqlite3.connect(str(store.db_path())) as conn:
        conn.execute(
            "UPDATE active_days SET claimed_at = ? WHERE day = '2026-01-01'", (aged,)
        )

    assert store.claim_active_day("2026-01-02") is True  # triggers the prune

    with sqlite3.connect(str(store.db_path())) as conn:
        days = {row[0] for row in conn.execute("SELECT day FROM active_days")}
    assert "2026-01-01" not in days, "an aged row was not pruned"
    assert "2026-01-02" in days, "the row that triggered the prune was pruned"


def test_claim_active_day_is_once_per_day_for_a_date_past_the_cutoff(fake_home):
    """Pruning must not delete the row the same transaction just claimed.

    A machine whose clock is wrong (or a caller holding a stale date)
    claims a day older than the retention cutoff. If the prune in that
    same transaction removes the row it just inserted, the claim leaves
    no trace and *every* later caller wins the same day — the
    once-per-day contract inverted exactly where it is hardest to spot.
    """
    from rapid_mlx.telemetry import store

    stale = (
        datetime.now(timezone.utc) - timedelta(days=store.ACTIVE_DAY_RETENTION_DAYS + 5)
    ).strftime("%Y-%m-%d")
    assert store.claim_active_day(stale) is True
    assert store.claim_active_day(stale) is False

    # The claim that actually breaks it. A row naming a past-cutoff date
    # is born prunable if retention keys off ``day``, so the next
    # successful claim of ANY other date sweeps it away and the stale
    # date becomes winnable all over again. Without an intervening
    # prune-triggering claim this test passes against that bug.
    assert store.claim_active_day() is True
    assert store.claim_active_day(stale) is False, (
        "a past-cutoff day was won twice — retention pruned the row that "
        "was proving the day had already been claimed"
    )


def test_a_naive_datetime_is_read_as_utc_not_local_time(fake_home):
    """Every docstring here says UTC; ``astimezone`` would say local.

    A caller that builds a timestamp in UTC and drops the tzinfo (the
    common shape) must land on the same day as the aware value for the
    same instant. Under ``astimezone``'s naive-means-local rule it lands
    on a different day for every hour of offset around midnight, and a
    different one per machine.
    """
    from rapid_mlx.telemetry import store

    naive = datetime(2026, 9, 20, 23, 30)
    aware = naive.replace(tzinfo=timezone.utc)
    assert store._as_day(naive) == store._as_day(aware) == "2026-09-20"

    # A plain ``date`` carries no time at all and is taken as-is.
    assert store._as_day(date(2026, 9, 20)) == "2026-09-20"
    # ...and it reaches the table under that name.
    assert store.claim_active_day(date(2026, 9, 20)) is True
    assert store.claim_active_day("2026-09-20") is False


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


def test_first_run_date_before_floor_is_clamped_and_written_once(fake_home):
    from rapid_mlx.telemetry import store

    before_release = datetime(2025, 12, 31, 12, 0, tzinfo=timezone.utc)
    assert store.first_run_date(before_release) == "2026-01-06"
    evidence = fake_home / ".rapid-mlx" / "telemetry-client-id"
    evidence.write_text("existing install")
    newer = datetime(2026, 2, 1, tzinfo=timezone.utc).timestamp()
    os.utime(evidence, (newer, newer))
    later = datetime(2026, 3, 9, tzinfo=timezone.utc)
    assert store.first_run_date(later) == "2026-01-06"


def test_first_run_date_seeds_bucket_from_old_evidence(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    evidence = fake_home / ".rapid-mlx" / "telemetry-client-id"
    evidence.parent.mkdir()
    evidence.write_text("existing install")
    old = (now - timedelta(days=40)).timestamp()
    os.utime(evidence, (old, old))

    assert store.days_since_first_run_bucket(now) == "30+"
    assert store.first_run_date(now) == "2026-08-11"


def test_first_run_date_uses_oldest_of_all_install_evidence(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    ages = {
        "telemetry-client-id": 4,
        "session_seen": 5,
        "activation_seen_server": 6,
        "activation_seen_desktop_first_chat_reply": 7,
        "bench-install-id": 8,
    }
    for name, days in ages.items():
        path = state_dir / name
        path.write_text(name)
        modified = (now - timedelta(days=days)).timestamp()
        os.utime(path, (modified, modified))
    (state_dir / "telemetry-consent.yaml").write_text(
        "consent: true\nprompted_at: '2026-08-18T12:00:00Z'\n"
    )

    assert store.first_run_date(now) == "2026-08-18"


def test_first_run_date_clamps_future_evidence_to_today(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    evidence = fake_home / ".rapid-mlx" / "session_seen"
    evidence.parent.mkdir()
    evidence.write_text("seen")
    future = (now + timedelta(days=40)).timestamp()
    os.utime(evidence, (future, future))

    assert store.first_run_date(now) == "2026-09-20"


def test_first_run_date_ignores_evidence_before_public_release(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    evidence = fake_home / ".rapid-mlx" / "session_seen"
    evidence.parent.mkdir()
    evidence.write_text("impossibly old install")
    ancient = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(evidence, (ancient, ancient))

    assert store.first_run_date(now) == "2026-09-20"


def test_first_run_date_ignores_prompt_before_public_release(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    consent = fake_home / ".rapid-mlx" / "telemetry-consent.yaml"
    consent.parent.mkdir()
    consent.write_text("prompted_at: '1970-01-01T00:00:00Z'\n")

    assert store.first_run_date(now) == "2026-09-20"


def test_first_run_date_keeps_valid_evidence_alongside_ancient_item(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    ancient = state_dir / "session_seen"
    ancient.write_text("impossibly old install")
    ancient_time = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(ancient, (ancient_time, ancient_time))
    valid = state_dir / "bench-install-id"
    valid.write_text("existing install")
    valid_time = (now - timedelta(days=12)).timestamp()
    os.utime(valid, (valid_time, valid_time))

    assert store.first_run_date(now) == "2026-09-08"


def test_first_run_date_permission_error_keeps_valid_fixed_evidence(
    fake_home, monkeypatch
):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    inaccessible = state_dir / "activation_seen_server"
    inaccessible.write_text("unreadable evidence")
    valid = state_dir / "session_seen"
    valid.write_text("existing install")
    valid_time = (now - timedelta(days=9)).timestamp()
    os.utime(valid, (valid_time, valid_time))
    original_stat = Path.stat
    original_lstat = Path.lstat

    def unreadable_stat(path, *args, **kwargs):
        if path == inaccessible:
            raise PermissionError("unreadable evidence")
        return original_stat(path, *args, **kwargs)

    def unreadable_lstat(path, *args, **kwargs):
        if path == inaccessible:
            raise PermissionError("unreadable evidence")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unreadable_stat)
    monkeypatch.setattr(Path, "lstat", unreadable_lstat)

    assert store.first_run_date(now) == "2026-09-11"


def test_first_run_date_does_not_follow_consent_symlink_to_fifo(fake_home):
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    fifo = fake_home / "consent-fifo"
    os.mkfifo(fifo)
    (state_dir / "telemetry-consent.yaml").symlink_to(fifo)
    script = """
from datetime import datetime, timezone
from rapid_mlx.telemetry import store
print(store.first_run_date(datetime(2026, 9, 20, tzinfo=timezone.utc)))
"""

    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=_child_env(fake_home),
        text=True,
        timeout=1.0,
    )

    assert time.monotonic() - started < 1.0
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2026-09-20"


@pytest.mark.parametrize(
    "evidence_name", ["telemetry-client-id", "activation_seen_server"]
)
def test_first_run_date_ignores_symlink_to_old_external_file(fake_home, evidence_name):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    external = fake_home / "external-client-id"
    external.write_text("not install evidence")
    old = (now - timedelta(days=100)).timestamp()
    os.utime(external, (old, old))
    (state_dir / evidence_name).symlink_to(external)

    assert store.first_run_date(now) == "2026-09-20"


@pytest.mark.parametrize(
    "evidence_name", ["telemetry-client-id", "activation_seen_server"]
)
@pytest.mark.parametrize("file_kind", ["directory", "socket", "fifo"])
def test_first_run_date_ignores_non_regular_evidence(
    fake_home, monkeypatch, evidence_name, file_kind
):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    marker = state_dir / evidence_name
    marker_socket = None
    if file_kind == "directory":
        marker.mkdir()
    elif file_kind == "socket":
        marker_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # macOS limits AF_UNIX names to 104 bytes; binding relative to the
        # state directory still creates the marker at the exact target path.
        monkeypatch.chdir(state_dir)
        marker_socket.bind(evidence_name)
    else:
        os.mkfifo(marker)
    old = (now - timedelta(days=15)).timestamp()
    os.utime(marker, (old, old))

    try:
        assert store.first_run_date(now) == "2026-09-20"
    finally:
        if marker_socket is not None:
            marker_socket.close()


@pytest.mark.parametrize(
    "evidence_name", ["telemetry-client-id", "activation_seen_server"]
)
def test_first_run_date_accepts_hard_link_to_regular_evidence(fake_home, evidence_name):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    original = fake_home / "regular-evidence"
    original.write_text("existing install")
    old = (now - timedelta(days=15)).timestamp()
    os.utime(original, (old, old))
    os.link(original, state_dir / evidence_name)

    assert store.first_run_date(now) == "2026-09-05"


def test_first_run_date_bounds_oversized_consent_read(fake_home, monkeypatch):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    consent = fake_home / ".rapid-mlx" / "telemetry-consent.yaml"
    consent.parent.mkdir()
    consent.write_text(
        "consent: true\nprompted_at: '2026-08-18T12:00:00Z'\n# " + "x" * 8_192
    )
    read_sizes = []
    real_read = os.read

    def recording_read(descriptor, size):
        read_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(os, "read", recording_read)

    assert consent.stat().st_size > 4_096
    assert store.first_run_date(now) == "2026-08-18"
    assert read_sizes == [store._MAX_CONSENT_BYTES]


def test_first_run_date_does_not_block_on_consent_fifo(fake_home):
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    os.mkfifo(state_dir / "telemetry-consent.yaml")
    script = """
from datetime import datetime, timezone
from rapid_mlx.telemetry import store
print(store.first_run_date(datetime(2026, 9, 20, tzinfo=timezone.utc)))
"""

    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=_child_env(fake_home),
        text=True,
        timeout=1.0,
    )

    assert time.monotonic() - started < 1.0
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2026-09-20"


def test_bounded_consent_reader_ignores_open_error(fake_home, monkeypatch):
    from rapid_mlx.telemetry import store

    consent = fake_home / "consent.yaml"
    consent.write_text("consent: true\n")
    real_open = os.open

    def unavailable_open(path, *args, **kwargs):
        if Path(path) == consent:
            raise PermissionError("consent is unreadable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", unavailable_open)

    assert store._bounded_regular_file_text(consent) is None


def test_bounded_consent_reader_skips_directory(fake_home):
    from rapid_mlx.telemetry import store

    consent = fake_home / "consent-directory"
    consent.mkdir()

    assert store._bounded_regular_file_text(consent) is None


def test_bounded_consent_reader_ignores_close_error(fake_home, monkeypatch):
    from rapid_mlx.telemetry import store

    consent = fake_home / "consent.yaml"
    consent.write_text("consent: true\n")
    real_close = os.close

    def close_then_fail(descriptor):
        real_close(descriptor)
        raise OSError("close failed")

    monkeypatch.setattr(os, "close", close_then_fail)
    assert store._bounded_regular_file_text(consent) == "consent: true\n"


def test_malformed_prompted_at_does_not_discard_other_evidence(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    evidence = state_dir / "session_seen"
    evidence.write_text("seen")
    old = (now - timedelta(days=9)).timestamp()
    os.utime(evidence, (old, old))
    (state_dir / "telemetry-consent.yaml").write_text(
        "consent: true\nprompted_at: definitely-not-a-timestamp\n"
    )

    assert store.first_run_date(now) == "2026-09-11"


def test_activation_scan_error_does_not_discard_fixed_evidence(fake_home, monkeypatch):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    evidence = fake_home / ".rapid-mlx" / "session_seen"
    evidence.parent.mkdir()
    evidence.write_text("seen")
    old = (now - timedelta(days=3)).timestamp()
    os.utime(evidence, (old, old))

    def unavailable_glob(path, pattern):
        raise PermissionError(f"cannot scan {path / pattern}")

    monkeypatch.setattr(Path, "glob", unavailable_glob)

    assert store.first_run_date(now) == "2026-09-17"


@requires_non_root
def test_seed_first_run_date_does_not_write_read_only_state_dir(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    state_dir = fake_home / ".rapid-mlx"
    state_dir.mkdir()
    marker = state_dir / "session_seen"
    marker.write_text("seen")
    old = (now - timedelta(days=2)).timestamp()
    os.utime(marker, (old, old))
    before = tuple(state_dir.iterdir())
    os.chmod(state_dir, 0o500)
    try:
        assert store._seed_first_run_date(now) == "2026-09-18"
        assert tuple(state_dir.iterdir()) == before
    finally:
        os.chmod(state_dir, 0o700)


def test_first_run_date_and_bucket_use_same_utc_day(fake_home):
    from rapid_mlx.telemetry import store

    local_evening = datetime(2026, 9, 20, 18, 30, tzinfo=timezone(timedelta(hours=-7)))

    assert store.first_run_date(local_evening) == "2026-09-21"
    assert store.days_since_first_run_bucket(local_evening) == "0"


def test_first_run_date_without_evidence_starts_today(fake_home):
    from rapid_mlx.telemetry import store

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)

    assert store.first_run_date(now) == "2026-09-20"


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

    first = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    assert store.first_run_date(first) == "2026-02-01"
    assert store.days_since_first_run_bucket(first + timedelta(days=days)) == expected


def test_days_since_first_run_bucket_tolerates_a_backwards_clock(fake_home):
    from rapid_mlx.telemetry import store

    first = datetime(2026, 5, 1, tzinfo=timezone.utc)
    store.first_run_date(first)
    assert store.days_since_first_run_bucket(first - timedelta(days=10)) == "0"


def test_cohort_values_are_in_the_declared_enum(fake_home):
    from rapid_mlx.telemetry import store

    first = datetime(2026, 2, 1, tzinfo=timezone.utc)
    store.first_run_date(first)
    for days in (0, 1, 3, 10, 90):
        assert (
            store.days_since_first_run_bucket(first + timedelta(days=days))
            in store.DAY_BUCKETS
        )


# --------------------------------------------------------------------------
# Failure policy: telemetry must never break `serve`
# --------------------------------------------------------------------------


@requires_non_root
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


@requires_non_root
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


def test_unwritable_state_dir_returns_nothing_to_emit(fake_home, monkeypatch):
    """Every public function degrades to "nothing to emit", never raises.

    Fails the ``mkdir`` itself rather than relying on directory
    permissions, so this holds for any user (including root) and on any
    filesystem. An ``OSError`` is not corruption, so nothing is renamed
    aside either.
    """
    from rapid_mlx.telemetry import store

    def boom(self, *args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "mkdir", boom)
    assert store.record("k") is None
    assert store.claim_active_day("2026-09-20") is False
    assert store.note_model_served("qwen3-8b") == 0
    assert store.first_run_date() is None
    assert store.days_since_first_run_bucket() is None
    assert not list(fake_home.glob(".rapid-mlx/telemetry.db.corrupt-*"))


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


# --------------------------------------------------------------------------
# Internal failure branches
#
# These live below the public API but are exactly the paths that decide
# whether a broken machine breaks `serve`, so they are tested directly
# rather than left to a lucky integration test.
# --------------------------------------------------------------------------


class _ExplodingConnection:
    """A stand-in connection whose statements fail on demand.

    ``fail_on`` matches the *start* of a statement (case-insensitive);
    a matching statement raises the configured error instead of running.
    """

    def __init__(self, fail_on, error, *, results=None):
        self.fail_on = tuple(prefix.lower() for prefix in fail_on)
        self.error = error
        self.results = results or {}
        self.statements = []

    def execute(self, sql, *args):
        self.statements.append(sql)
        lowered = sql.lower()
        if lowered.startswith(self.fail_on):
            raise self.error
        for prefix, row in self.results.items():
            if lowered.startswith(prefix.lower()):
                return _Row(row)
        return _Row(None)


class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


def test_transaction_rolls_back_and_re_raises(fake_home):
    from rapid_mlx.telemetry import store

    conn = _ExplodingConnection(["insert"], sqlite3.OperationalError("disk full"))
    with pytest.raises(sqlite3.OperationalError), store._transaction(conn):
        conn.execute("INSERT INTO counters VALUES (1)")
    assert conn.statements[0] == "BEGIN IMMEDIATE"
    assert conn.statements[-1] == "ROLLBACK"
    assert "COMMIT" not in conn.statements


def test_transaction_survives_a_rollback_that_also_fails(fake_home):
    """The original error is what the caller needs, not the rollback's."""
    from rapid_mlx.telemetry import store

    conn = _ExplodingConnection(
        ["insert", "rollback"], sqlite3.OperationalError("database is locked")
    )
    with (
        pytest.raises(sqlite3.OperationalError, match="locked"),
        store._transaction(conn),
    ):
        conn.execute("INSERT INTO counters VALUES (1)")
    assert conn.statements[-1] == "ROLLBACK"


def test_quarantine_returns_false_when_the_rename_fails(fake_home, monkeypatch):
    """An unwritable state dir must not turn into an exception."""
    from rapid_mlx.telemetry import store

    store.record("k")  # a real database...
    store.db_path().write_bytes(b"not a database")  # ...that is now corrupt

    def boom(self, target):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "rename", boom)
    assert store._quarantine_corrupt_db(store._db_identity()) is False


def test_quarantine_leaves_a_database_another_process_already_recreated(fake_home):
    """The latch is per process; the file it guards is shared.

    Process A finds the database corrupt, renames it aside, recreates it
    and starts counting again. Process B, which hit the same corruption
    a moment earlier, then reaches its own quarantine — and must not
    rename A's healthy database away with A's counters in it.
    """
    from rapid_mlx.telemetry import store

    assert store.record("k") is not None  # a real database exists
    doomed = store._db_identity()  # what "process B" is about to fail on
    store.db_path().write_bytes(b"this is not a database")
    store._reset_quarantine_latch_for_tests()
    assert store.record("k") is not None  # "process A": renamed + recreated
    store._reset_quarantine_latch_for_tests()  # "process B": a fresh latch

    assert store._quarantine_corrupt_db(doomed) is True  # retry, do not rename
    assert len(list(fake_home.glob(".rapid-mlx/telemetry.db.corrupt-*"))) == 1
    # A's counter is still there, and B's latch was not spent.
    crossing = store.record("k")
    assert crossing is not None and crossing.count == 2


def test_a_world_readable_database_is_made_private_again(fake_home):
    """0600 is a promise about the file, not only about its creation.

    ``sqlite3.connect`` creates the file under the ambient umask; a
    process killed before the ``chmod`` (or an older build) leaves a
    world-readable database that no later run would ever repair.
    """
    from rapid_mlx.telemetry import store

    store.record("k")
    db = store.db_path()
    wal = db.with_name(db.name + "-wal")

    # SQLite deletes the WAL when the last connection closes, so after a
    # plain ``record()`` there is no sibling to check and an
    # ``if wal.exists()`` guard would quietly assert nothing. Hold a
    # connection open (no transaction in flight, so ``record`` below is
    # not blocked) to keep the sibling on disk for real.
    with closing(sqlite3.connect(str(db))) as holder:
        holder.execute("PRAGMA journal_mode = WAL")
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT OR REPLACE INTO install_facts VALUES ('probe', 'x')")
        holder.execute("COMMIT")
        assert wal.exists(), "no WAL sibling — the mode checks below would be dead"

        os.chmod(db, 0o644)
        os.chmod(wal, 0o644)

        assert store.record("k") is not None
        assert (db.stat().st_mode & 0o777) == 0o600
        assert (wal.stat().st_mode & 0o777) == 0o600


def test_a_key_that_cannot_be_encoded_is_nothing_to_emit(fake_home):
    """A lone surrogate is a normal ``str`` the driver cannot encode.

    ``os.fsdecode`` returns exactly this for an undecodable byte in a
    filename, so a caller that builds a key from a path hands us one
    without doing anything unusual. The sqlite3 driver raises
    ``UnicodeEncodeError`` — a ``ValueError``, not a ``sqlite3.Error`` —
    while binding the parameter, and telemetry must never be the thing
    that breaks a request.
    """
    from rapid_mlx.telemetry import store

    surrogate = "model-\ud800"
    assert store.record(surrogate) is None
    assert store.note_model_served(surrogate) == 0
    assert store.claim_active_day(surrogate) is False

    # ...and the store still works for everything else afterwards.
    crossing = store.record("k")
    assert crossing is not None and crossing.count == 1


def test_a_host_with_no_home_directory_is_nothing_to_emit(monkeypatch):
    """Every public entry point, on a container with no resolvable home.

    ``Path.home()`` raises ``RuntimeError`` — not an ``OSError`` — when
    neither ``$HOME`` nor the passwd database names a home, which is the
    ordinary shape of ``docker --user 1000:1000`` with no HOME, or
    OpenShift's random uid. The call that resolves it runs before
    ``_run``'s guard, so a narrow catch there makes the whole module
    raise on exactly those hosts.

    Deliberately does NOT use the ``fake_home`` fixture: that fixture
    sets ``HOME``, which is precisely what this host does not have, and
    is why 100% line coverage did not notice.
    """
    from rapid_mlx.telemetry import store

    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    monkeypatch.delenv("HOME", raising=False)
    store._reset_quarantine_latch_for_tests()

    assert store.record("k") is None
    assert store.claim_active_day() is False
    assert store.note_model_served("m") == 0
    assert store.first_run_date() is None
    assert store.days_since_first_run_bucket() is None


def test_home_disappearing_mid_call_is_nothing_to_emit(fake_home, monkeypatch):
    """``db_path()`` is called several times; the environment can move.

    Round 3 guarded the ``_db_identity()`` that runs before ``_run``'s
    ``try``, and reasoned that the ``db_path()`` inside
    ``_quarantine_corrupt_db`` could not then fail. That holds only if
    both calls see the same environment. A process whose HOME is unset
    while it runs, with a corrupt database already on disk, takes the
    quarantine branch and hits the later call — and the quarantine
    decision is evaluated inside ``_run``'s own except clause, so
    anything it raises replaces the original error and leaves the
    module.
    """
    from rapid_mlx.telemetry import store

    store.record("k")  # a real database...
    store.db_path().write_bytes(b"not a database")  # ...now corrupt

    real_db_path = store.db_path
    calls = {"n": 0}

    def db_path_that_stops_working():
        calls["n"] += 1
        # The calls, in order: 1 the pre-``try`` identity capture,
        # 2 the connect, 3 quarantine's own identity re-check, 4 the one
        # quarantine uses to build the rename target. Only the fourth
        # may fail — failing earlier makes ``_db_identity`` return
        # ``None``, which short-circuits quarantine before it gets
        # there, so the raising line is never reached and the test would
        # pass against the bug.
        if calls["n"] == 4:
            raise RuntimeError("Could not determine home directory.")
        return real_db_path()

    monkeypatch.setattr(store, "db_path", db_path_that_stops_working)
    assert store.record("k") is None
    assert calls["n"] >= 4, "quarantine's own db_path() call was never reached"


def test_no_caller_input_escapes_as_an_exception(fake_home):
    """The backstop itself: a non-storage error inside the work returns.

    ``_valid_key`` screens the surrogate case, so this pins the guard
    that catches whatever the screen does not anticipate.
    """
    from rapid_mlx.telemetry import store

    def work(conn):
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed")

    assert store._run(work, "nothing-to-emit") == "nothing-to-emit"


def test_quarantine_does_not_delete_a_racing_process_wal(fake_home, monkeypatch):
    """The siblings must be dealt with before the path stops being ours.

    Process A renames the corrupt database aside. In the instant after
    that rename the path names nothing, so process B can create a fresh
    database and its WAL there. If A then removes ``telemetry.db-wal``
    *by path*, it destroys B's live WAL — and the resulting
    ``disk I/O error`` is not classified as corruption, so every process
    afterwards silently records nothing.
    """
    from rapid_mlx.telemetry import store

    db = store.db_path()
    store.record("k")
    doomed = store._db_identity()
    db.write_bytes(b"not a database")
    (db.with_name(db.name + "-wal")).write_bytes(b"stale wal")

    real_rename = Path.rename

    def rename_then_let_b_in(self, target):
        result = real_rename(self, target)
        if self == db:  # the database itself just moved aside
            db.write_bytes(b"process B's fresh database")
            (db.with_name(db.name + "-wal")).write_bytes(b"process B's live WAL")
        return result

    monkeypatch.setattr(Path, "rename", rename_then_let_b_in)
    assert store._quarantine_corrupt_db(doomed) is True

    assert db.read_bytes() == b"process B's fresh database"
    assert (db.with_name(db.name + "-wal")).read_bytes() == b"process B's live WAL", (
        "quarantine destroyed the WAL of a database another process had "
        "already recreated"
    )
    # The corrupt file's own WAL travelled with it rather than being deleted.
    quarantined_wal = list(fake_home.glob(".rapid-mlx/telemetry.db.corrupt-*-wal"))
    assert len(quarantined_wal) == 1
    assert quarantined_wal[0].read_bytes() == b"stale wal"


def test_quarantine_rechecks_identity_after_moving_the_siblings(fake_home, monkeypatch):
    """The sibling renames sit between the first check and the rename.

    Nothing here is atomic across processes, so the identity can change
    during those two syscalls: another process quarantines and recreates
    one step earlier than we do. The late re-check is what stops us
    renaming its healthy database aside.
    """
    from rapid_mlx.telemetry import store

    db = store.db_path()
    store.record("k")
    doomed = store._db_identity()
    db.write_bytes(b"not a database")
    (db.with_name(db.name + "-wal")).write_bytes(b"stale wal")

    real_rename = Path.rename

    def rename_and_let_b_win_the_race(self, target):
        result = real_rename(self, target)
        if self.name.endswith("-wal"):
            # "Process B" got there first, so the path holds a different
            # inode now. Build B's file beside the old one and swap it in
            # rather than unlinking first: while the corrupt inode is
            # still linked it cannot be reused, so the replacement is
            # guaranteed to differ. Unlinking first frees it, and Linux
            # hands the very same inode straight back — which would make
            # this test pass or fail by filesystem rather than by
            # behaviour, and would be simulating something quarantine
            # cannot do anyway (the renamed-aside file keeps that inode
            # alive).
            elsewhere = db.with_name("process-b-database")
            elsewhere.write_bytes(b"process B's fresh database")
            os.replace(elsewhere, db)
        return result

    monkeypatch.setattr(Path, "rename", rename_and_let_b_win_the_race)
    assert store._quarantine_corrupt_db(doomed) is True
    assert db.read_bytes() == b"process B's fresh database", (
        "the identity changed while the siblings were moving and the "
        "rename went ahead anyway"
    )


def test_chmod_failure_does_not_stop_the_store(fake_home, monkeypatch):
    """A filesystem without POSIX modes still gets a working database."""
    from rapid_mlx.telemetry import store

    def boom(path, mode):
        raise OSError("operation not supported")

    monkeypatch.setattr(store.os, "chmod", boom)
    assert store.record("k") is not None


def test_enable_wal_tolerates_a_busy_journal_mode_change(fake_home):
    """Changing journal_mode bypasses the busy handler — never fatal."""
    from rapid_mlx.telemetry import store

    conn = _ExplodingConnection(
        ["pragma journal_mode ="],
        sqlite3.OperationalError("database is locked"),
        results={"pragma journal_mode": ("delete",)},
    )
    store._enable_wal(conn)  # must not raise
    assert conn.statements[-1] == "PRAGMA journal_mode = WAL"


def test_enable_wal_is_a_no_op_when_already_wal(fake_home):
    from rapid_mlx.telemetry import store

    conn = _ExplodingConnection(
        ["pragma journal_mode ="],
        AssertionError("should not have tried to change journal_mode"),
        results={"pragma journal_mode": ("wal",)},
    )
    store._enable_wal(conn)
    assert conn.statements == ["PRAGMA journal_mode"]


def test_run_returns_the_default_when_the_work_fails(fake_home):
    from rapid_mlx.telemetry import store

    def work(conn):
        raise sqlite3.OperationalError("database is locked")

    assert store._run(work, "nothing-to-emit") == "nothing-to-emit"


def test_run_retries_once_after_quarantining_a_corrupt_db(fake_home):
    """A corruption raised by the *work* (not the connect) also retries."""
    from rapid_mlx.telemetry import store

    # A corruption error out of the *work* means real pages were read,
    # so the precondition is a database that already exists.
    store.record("seed")
    calls = []

    def work(conn):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return "second-attempt"

    assert store._run(work, None) == "second-attempt"
    assert len(calls) == 2
    assert len(list(fake_home.glob(".rapid-mlx/telemetry.db.corrupt-*"))) == 1


def test_run_gives_up_after_the_bounded_number_of_attempts(fake_home, monkeypatch):
    """The retry budget is exactly one even if quarantining keeps working."""
    from rapid_mlx.telemetry import store

    monkeypatch.setattr(store, "_quarantine_corrupt_db", lambda _identity: True)
    calls = []

    def work(conn):
        calls.append(1)
        raise sqlite3.DatabaseError("database disk image is malformed")

    assert store._run(work, "gave-up") == "gave-up"
    assert len(calls) == 2


def test_unparseable_first_run_date_reads_as_no_cohort(fake_home):
    """A hand-edited or future-format row must not raise at emit time."""
    from rapid_mlx.telemetry import store

    store.first_run_date()
    with sqlite3.connect(str(store.db_path())) as conn:
        conn.execute(
            "UPDATE install_facts SET value = 'not-a-date' WHERE key = 'first_run_date'"
        )
    assert store.days_since_first_run_bucket() is None
