# SPDX-License-Identifier: Apache-2.0
"""Cross-process local telemetry state (``~/.rapid-mlx/telemetry.db``).

One install can run the CLI, a standalone ``rapid-mlx serve`` and the
desktop app's sidecar server *at the same time*, all sharing
``~/.rapid-mlx/``. Anything that must happen "once per install" —
a bucket-crossing milestone, the once-per-UTC-day ``active_day`` event,
the cohort stamp — therefore cannot live in a per-process variable or in
a read-modify-write of a YAML file. It lives in one SQLite database in
WAL mode, and every decision is taken inside a single
``BEGIN IMMEDIATE`` transaction so exactly one process wins.

What this module stores:

``counters``
    Lifetime count per opaque ``key`` plus the highest bucket we have
    already emitted for it. ``record()`` increments, compares and
    persists in one transaction and only *then* returns the crossing, so
    the caller emits after the write. A crash in between loses one
    milestone; it can never duplicate one. Buckets are Orca's scale
    (``1, 2, 3_4, … 1000_plus``) — one event per threshold, O(log n)
    events, no timing trail, no sampling.

``active_days``
    ``INSERT OR IGNORE`` of a UTC date. ``True`` is returned to the one
    process whose insert actually created the row; every concurrent
    racer gets ``False``. Rows older than ~40 days are pruned so the
    table cannot grow without bound.

``models_served`` / ``install_facts``
    The cohort stamp of §1.2: ``nth_model_served`` (how many distinct
    models this install has ever served) and the first-run date behind
    ``days_since_first_run_bucket``.

**Failure policy.** Telemetry must never break ``serve``. Every public
function swallows ``sqlite3`` and ``OSError`` failures — read-only
``HOME``, a locked or corrupt database, a full disk — and returns the
"nothing to emit" value (``None`` / ``False`` / ``0``). A corrupt
database is renamed aside once per process and recreated. No call blocks
longer than :data:`BUSY_TIMEOUT_SECONDS`.

Nothing here transmits anything. Callers decide what (if anything) to
emit, and the consent gate in :mod:`rapid_mlx.telemetry.state` still
governs that decision.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TypeVar

from rapid_mlx.telemetry.state import _default_telemetry_dir

#: Bumped when the on-disk schema changes incompatibly. Stored in
#: ``schema_meta`` so a future version can migrate or quarantine.
SCHEMA_VERSION = 1

#: Longest key we accept. Keys are built by callers from closed enum
#: values (model id + endpoint + caller + result), so a long one means a
#: caller leaked something free-form — drop it rather than store it.
MAX_KEY_LENGTH = 200

#: Hard cap on distinct rows in ``counters`` / ``models_served``. A
#: pathological caller (or a model id that turns out not to be closed
#: after all) must not grow this file without bound. Past the cap a new
#: key is ignored; existing keys keep counting.
MAX_KEYS = 2000

#: Nothing in here may block a request path. SQLite retries a locked
#: database internally for at most this long, then raises and we fall
#: back to "nothing to emit".
BUSY_TIMEOUT_SECONDS = 2.0

#: ``active_days`` rows older than this are pruned. Comfortably longer
#: than any retention window we need locally (the row only answers "did
#: someone already claim today?"), short enough to stay tiny.
ACTIVE_DAY_RETENTION_DAYS = 40

#: Orca's ``FEATURE_INTERACTION_USAGE_BUCKET_SPECS``, minus the
#: ``count_`` prefix: ``(minimum count, bucket name)``, ascending.
BUCKET_SPECS: tuple[tuple[int, str], ...] = (
    (1, "1"),
    (2, "2"),
    (3, "3_4"),
    (5, "5_9"),
    (10, "10_19"),
    (20, "20_49"),
    (50, "50_99"),
    (100, "100_199"),
    (200, "200_499"),
    (500, "500_999"),
    (1000, "1000_plus"),
)

#: Bucket names in ascending order — the ``count_bucket`` enum.
BUCKETS: tuple[str, ...] = tuple(name for _, name in BUCKET_SPECS)

_BUCKET_RANK = {name: index for index, name in enumerate(BUCKETS)}

#: ``days_since_first_run_bucket`` values, ascending.
DAY_BUCKETS: tuple[str, ...] = ("0", "1", "2-6", "7-29", "30+")

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS schema_meta ("
    " key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS counters ("
    " key TEXT PRIMARY KEY, count INTEGER NOT NULL, last_bucket TEXT)",
    "CREATE TABLE IF NOT EXISTS active_days ("
    " day TEXT PRIMARY KEY, claimed_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS models_served ("
    " model_id TEXT PRIMARY KEY, first_seen TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS install_facts ("
    " key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)

_T = TypeVar("_T")

# A corrupt file is renamed aside at most once per process: a rename loop
# on a file that keeps failing for some *other* reason (a directory in its
# place, say) would churn the state dir on every event.
_quarantine_lock = threading.Lock()
_quarantined = False


@dataclass(frozen=True)
class BucketCrossing:
    """A milestone worth one ``inference_bucket_reached``-style event.

    ``bucket_source`` follows Orca: ``crossed_now`` when the increment
    itself moved the counter into a new bucket, ``observed_existing``
    when this install already had a counter at that bucket before we
    started tracking buckets for it (so the milestone is real but we
    cannot claim we watched it happen).
    """

    key: str
    count: int
    bucket: str
    bucket_source: str


def db_path() -> Path:
    """Resolved at call time so ``HOME`` overrides in tests take effect."""
    return _default_telemetry_dir() / "telemetry.db"


def bucket_for(count: int) -> str | None:
    """Orca's bucket for ``count``, or ``None`` below the first bucket."""
    bucket: str | None = None
    for minimum, name in BUCKET_SPECS:
        if count >= minimum:
            bucket = name
    return bucket


def _is_corruption(exc: BaseException) -> bool:
    """True for "this file is not a usable database" errors only.

    ``OperationalError`` (locked, disk full, read-only) is a transient or
    environmental failure and must not trigger a rename — deleting a
    perfectly good database because the disk filled up would lose every
    counter.
    """
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    if isinstance(exc, sqlite3.OperationalError):
        return "malformed" in str(exc).lower()
    message = str(exc).lower()
    return (
        "malformed" in message
        or "not a database" in message
        or "encrypted" in message
        or "corrupt" in message
    )


def _quarantine_corrupt_db() -> bool:
    """Rename a corrupt database aside so the next call recreates it.

    Returns ``True`` if a retry is worth attempting. Once per process:
    see ``_quarantined``.
    """
    global _quarantined
    with _quarantine_lock:
        if _quarantined:
            return False
        _quarantined = True
    path = db_path()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        path.rename(path.with_name(f"{path.name}.corrupt-{stamp}"))
        # The WAL/SHM siblings belong to the file we just moved aside; a
        # leftover WAL next to a freshly created database is itself a
        # corruption source.
        for suffix in ("-wal", "-shm"):
            sibling = path.with_name(path.name + suffix)
            try:
                sibling.unlink()
            except OSError:
                pass
    except OSError:
        return False
    return True


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Put the database in WAL mode if it is not already.

    WAL is what makes concurrent readers plus one writer cheap across
    processes, and it is a persistent property of the file — so set it
    only when it is not already set. *Changing* ``journal_mode`` takes an
    exclusive lock and, unlike ordinary statements, does NOT go through
    the busy handler, so a redundant set under contention raises
    "database is locked" and would cost us an increment. A genuine
    failure is tolerated: a rollback-journal database still works, it
    just serialises more.
    """
    (mode,) = conn.execute("PRAGMA journal_mode").fetchone()
    if str(mode).lower() == "wal":
        return
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass


def _connect() -> sqlite3.Connection:
    """Open (creating on first use) the telemetry database.

    ``isolation_level=None`` turns off the driver's implicit transaction
    handling: every transaction in this module is an explicit
    ``BEGIN IMMEDIATE`` so the write lock is taken up front and two
    processes can never interleave a read-then-write.
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    conn = sqlite3.connect(
        str(path), timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None
    )
    try:
        if not existed:
            # Before anything is written: the file holds no secrets, but
            # it is the user's machine's activity and nobody else's
            # business.
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
        _enable_wal(conn)
        conn.execute("PRAGMA synchronous = NORMAL")
        # ``user_version`` lives in the file header and is a plain read,
        # so the common case (schema already current) costs no write
        # lock at all. Only a brand-new or quarantined-and-recreated
        # file takes the CREATE TABLE path.
        (user_version,) = conn.execute("PRAGMA user_version").fetchone()
        if int(user_version) != SCHEMA_VERSION:
            _ensure_schema(conn)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """One ``BEGIN IMMEDIATE`` … ``COMMIT``, rolled back on any failure.

    ``IMMEDIATE`` takes the write lock up front, which is the whole
    point: a deferred transaction would read, then try to upgrade, and
    two processes could interleave a read-then-write. Every decision in
    this module goes through here so the rollback-on-failure handling
    exists once rather than five times.

    A ``ROLLBACK`` that itself fails is swallowed: the original error is
    what the caller needs to see, and the connection is closed
    immediately afterwards anyway, which rolls back anything open.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    conn.execute("COMMIT")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the tables lazily and stamp the schema version."""
    with _transaction(conn):
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")


def _attempt(work: Callable[[sqlite3.Connection], _T]) -> _T:
    """Open the database, run ``work``, close. Raises on any failure."""
    with closing(_connect()) as conn:
        return work(conn)


def _run(work: Callable[[sqlite3.Connection], _T], default: _T) -> _T:
    """Run ``work`` against the database, never raising.

    Exactly one retry, and only after a genuinely corrupt file has been
    renamed aside — every other failure returns ``default`` ("nothing to
    emit") immediately rather than blocking a caller twice. Failures
    from opening the database and from the work itself are handled the
    same way: either one means we have nothing to say.
    """
    try:
        return _attempt(work)
    except (sqlite3.Error, OSError) as exc:
        if not (_is_corruption(exc) and _quarantine_corrupt_db()):
            return default
    try:
        return _attempt(work)
    except (sqlite3.Error, OSError):
        return default


def _valid_key(key: str) -> bool:
    return isinstance(key, str) and 0 < len(key) <= MAX_KEY_LENGTH


def record(key: str) -> BucketCrossing | None:
    """Increment the lifetime counter for ``key``; return a new milestone.

    One ``BEGIN IMMEDIATE`` transaction does all of it: read the current
    count and the last bucket we emitted, increment, compute the new
    bucket, and — only if it is strictly higher than the last emitted one
    — persist it. The crossing is returned **after** that write is
    committed, so the caller captures an event that the database already
    knows about. Losing power in between loses one milestone; it can
    never emit the same one twice.

    Returns ``None`` when the increment stayed inside the current bucket,
    when ``key`` is unusable, when the distinct-key cap is reached, or on
    any storage failure.
    """
    if not _valid_key(key):
        return None

    def work(conn: sqlite3.Connection) -> BucketCrossing | None:
        with _transaction(conn):
            row = conn.execute(
                "SELECT count, last_bucket FROM counters WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                (known,) = conn.execute("SELECT COUNT(*) FROM counters").fetchone()
                if known >= MAX_KEYS:
                    # Nothing written; the empty transaction just commits.
                    return None
                previous, last_bucket = 0, None
            else:
                previous, last_bucket = int(row[0]), row[1]
            count = previous + 1
            bucket = bucket_for(count)
            crossed = bucket is not None and (
                last_bucket is None
                or _BUCKET_RANK.get(bucket, -1) > _BUCKET_RANK.get(last_bucket, -1)
            )
            conn.execute(
                "INSERT INTO counters (key, count, last_bucket) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET count = excluded.count,"
                " last_bucket = excluded.last_bucket",
                (key, count, bucket if crossed else last_bucket),
            )
        if not crossed or bucket is None:
            return None
        source = (
            "observed_existing"
            if last_bucket is None and bucket_for(previous) == bucket
            else "crossed_now"
        )
        return BucketCrossing(key=key, count=count, bucket=bucket, bucket_source=source)

    return _run(work, None)


def _as_day(value: date | datetime | str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def claim_active_day(utc_date: date | datetime | str | None = None) -> bool:
    """Claim "first successful inference today" for this install.

    ``True`` for exactly one caller per UTC day across every process on
    the machine — the one whose ``INSERT OR IGNORE`` created the row.
    Everyone else, and every failure, gets ``False`` (under-reporting a
    DAU is conservative; double-reporting it is a lie).
    """
    day = _as_day(utc_date)

    def work(conn: sqlite3.Connection) -> bool:
        with _transaction(conn):
            cursor = conn.execute(
                "INSERT OR IGNORE INTO active_days (day, claimed_at) VALUES (?, ?)",
                (day, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            )
            claimed = cursor.rowcount == 1
            if claimed:
                # ISO dates sort lexicographically, so a string compare
                # is a date compare. Only prune on the day we claimed —
                # a losing racer has no reason to touch the table.
                cutoff = (
                    datetime.now(timezone.utc)
                    - timedelta(days=ACTIVE_DAY_RETENTION_DAYS)
                ).strftime("%Y-%m-%d")
                conn.execute("DELETE FROM active_days WHERE day < ?", (cutoff,))
        return claimed

    return _run(work, False)


def note_model_served(model_id: str) -> int:
    """Record that ``model_id`` was served; return ``nth_model_served``.

    The returned count is the number of *distinct* models this install
    has ever served, including the one just noted — Orca's
    ``nth_repo_added`` idea, which lets any chart split newcomer from
    veteran with no server-side join. Returns ``0`` when the id is
    unusable or storage failed; the cohort stamp is optional, so ``0``
    simply means "no stamp".
    """
    if not _valid_key(model_id):
        return 0

    def work(conn: sqlite3.Connection) -> int:
        with _transaction(conn):
            (known,) = conn.execute("SELECT COUNT(*) FROM models_served").fetchone()
            exists = conn.execute(
                "SELECT 1 FROM models_served WHERE model_id = ?", (model_id,)
            ).fetchone()
            if exists is None and known < MAX_KEYS:
                conn.execute(
                    "INSERT INTO models_served (model_id, first_seen) VALUES (?, ?)",
                    (
                        model_id,
                        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ),
                )
                known += 1
        return int(known)

    return _run(work, 0)


def first_run_date(now: datetime | None = None) -> str | None:
    """Return the install's first-run date (``YYYY-MM-DD``), setting it once.

    Written by whichever process gets there first and never rewritten, so
    the cohort stamp is stable for the life of the install. ``None`` on
    any storage failure.
    """
    today = _as_day(now)

    def work(conn: sqlite3.Connection) -> str | None:
        with _transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO install_facts (key, value)"
                " VALUES ('first_run_date', ?)",
                (today,),
            )
            row = conn.execute(
                "SELECT value FROM install_facts WHERE key = 'first_run_date'"
            ).fetchone()
        return str(row[0]) if row else None

    return _run(work, None)


def days_since_first_run_bucket(now: datetime | None = None) -> str | None:
    """``0`` / ``1`` / ``2-6`` / ``7-29`` / ``30+``, or ``None``.

    ``now`` is injectable so the boundaries are testable without waiting
    a month. A clock that has gone backwards (a negative age) reads as
    ``0`` rather than as an error — the stamp is a cohort hint, not an
    audit trail.
    """
    stored = first_run_date(now)
    if stored is None:
        return None
    try:
        first = datetime.strptime(stored, "%Y-%m-%d").date()
    except ValueError:
        return None
    today = datetime.strptime(_as_day(now), "%Y-%m-%d").date()
    days = (today - first).days
    if days <= 0:
        return "0"
    if days == 1:
        return "1"
    if days <= 6:
        return "2-6"
    if days <= 29:
        return "7-29"
    return "30+"


def _reset_quarantine_latch_for_tests() -> None:
    """Allow a second corrupt-file quarantine inside one test process."""
    global _quarantined
    with _quarantine_lock:
        _quarantined = False
