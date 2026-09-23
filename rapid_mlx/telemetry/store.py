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
database is renamed aside once per process and recreated. No single
database operation waits longer than :data:`BUSY_TIMEOUT_SECONDS`; a
call that opens, transacts and (once) retries can therefore wait a small
multiple of it, which is why callers on a request path run these off the
event loop.

Nothing here transmits anything. Callers decide what (if anything) to
emit, and the consent gate in :mod:`rapid_mlx.telemetry.state` still
governs that decision.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TypeVar

import yaml

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
#: key is ignored; existing keys keep counting. Twelve thousand rows cover
#: every endpoint/caller/result combination for 35 models; this local SQLite
#: state remains tiny while avoiding exhaustion on ordinary multi-model hosts.
MAX_KEYS = 12_000

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

#: Earliest plausible install evidence: the first public Rapid-MLX release,
#: v0.2.0, was released on 2026-01-06. Older filesystem timestamps or consent
#: records cannot describe a Rapid-MLX install and must not permanently seed
#: its write-once cohort date.
FIRST_RUN_EVIDENCE_FLOOR = date(2026, 1, 6)

_INSTALL_EVIDENCE_FILES: tuple[str, ...] = (
    "telemetry-client-id",
    "session_seen",
    "bench-install-id",
)

_MAX_CONSENT_BYTES = 4_096

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

    ``observed_existing`` is **unreachable from rows this module writes**
    — every counter it creates starts at zero with its bucket tracked
    from the first increment. It is kept deliberately, for the case the
    name describes: a ``counters`` row that arrives with a count already
    past a threshold and no ``last_bucket``, which is what a future
    import, backfill or schema migration would produce. Emitting such a
    milestone as ``crossed_now`` would be a lie about when it happened.
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


def _db_identity() -> tuple[int, int] | None:
    """``(device, inode)`` of the database file, or ``None`` if absent.

    Captured *before* an attempt so a failure can be tied to the exact
    file that produced it. See :func:`_quarantine_corrupt_db`.

    The guard is ``Exception`` because ``db_path()`` is not only a
    ``stat``: it resolves ``Path.home()``, which raises ``RuntimeError``
    — not an ``OSError`` — when neither ``$HOME`` nor the passwd
    database names a home. That is the ordinary shape of a container run
    under an anonymous uid (``docker --user 1000:1000`` with no HOME,
    OpenShift's random uid), and this call is the one step into the
    module that runs before ``_run``'s own guard, so a narrow catch here
    makes every public function raise on exactly those hosts.
    """
    try:
        info = db_path().stat()
    except Exception:
        return None
    return (info.st_dev, info.st_ino)


def _quarantine_corrupt_db(identity: tuple[int, int] | None) -> bool:
    """Rename the corrupt database aside so the next call recreates it.

    Returns ``True`` if a retry is worth attempting. Once per process:
    see ``_quarantined``.

    ``identity`` is the ``(device, inode)`` observed before the failing
    attempt, and the rename happens **only** if the path still names
    that same file. The latch is per process but the file is shared by
    every process on the install: while this process was failing,
    another one may already have quarantined the same corrupt database
    and recreated a healthy one at the same path. Renaming by path alone
    would move *that* database aside — discarding the counters it has
    already started restoring — and do it again for every process that
    was mid-failure. A changed (or vanished) inode means someone else
    did the work, so there is nothing to quarantine and the retry should
    simply run against the new file.

    The check and the rename both hold ``_quarantine_lock`` so two
    threads in this process cannot both decide to rename.
    """
    global _quarantined
    with _quarantine_lock:
        if _quarantined:
            return False
        if identity is None or _db_identity() != identity:
            # Someone else already swapped a healthy database in. Retry
            # against it, and do not spend this process's one rename.
            return True
        _quarantined = True
        # ``db_path()`` normally cannot raise here: if it could, the
        # ``_db_identity()`` above would have returned ``None``, which
        # never equals a non-``None`` ``identity``, and we would already
        # have returned. "Normally" because that reasoning assumes the
        # two calls see the same environment, and nothing guarantees
        # that — the backstop is the guard around this call in ``_run``.
        path = db_path()
        # A second-resolution stamp collides when two quarantines land in
        # the same second and ``rename`` replaces silently, so the pid
        # disambiguates them.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = path.with_name(f"{path.name}.corrupt-{stamp}-{os.getpid()}")
        # The WAL/SHM siblings belong to the corrupt file and must travel
        # with it: a leftover WAL next to a freshly created database is
        # itself a corruption source.
        #
        # They move FIRST, and they move rather than being deleted. Both
        # halves matter. Once ``path`` has been renamed it names nothing,
        # so another process can create a fresh database (and a fresh
        # WAL) there immediately — anything we then do to those names by
        # path hits *its* live files, not ours. Deleting a live WAL out
        # from under an open connection does not even fail loudly: every
        # later open gets "disk I/O error", which :func:`_is_corruption`
        # deliberately does not treat as corruption, so the store would
        # go quietly dead for the life of the process. Doing the siblings
        # while ``path`` still holds the corrupt inode *narrows* that
        # window; it does not close it. Nothing here is atomic across
        # processes, so another process can still quarantine and recreate
        # between the identity check and the rename. That residual race
        # is bounded and self-healing: SQLite discards a WAL whose header
        # does not match the database beside it, so a loser that has had
        # its ``-wal`` taken recreates and loses only counters. Losing
        # ``-shm`` as well can leave real corruption behind, and that is
        # fine too — the next call finds it and quarantines it, which is
        # the path this whole function implements. ``serve`` is
        # unaffected either way, which is why the race is tolerated
        # rather than locked.
        # Renaming instead of unlinking keeps even that case
        # recoverable.
        for suffix in ("-wal", "-shm"):
            try:
                path.with_name(path.name + suffix).rename(
                    target.with_name(target.name + suffix)
                )
            except OSError:
                pass
        # Re-checked as late as possible: the sibling renames above sit
        # between the first check and this one.
        if _db_identity() != identity:
            return True
        try:
            path.rename(target)
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


def _enforce_private_mode(path: Path) -> None:
    """Keep the database and its WAL siblings owner-only.

    The file holds no secrets, but it is the user's machine's activity
    and nobody else's business, and the module docstring promises 0600.
    Checking on every connect rather than only on creation is what makes
    that promise true: ``sqlite3.connect`` creates the file under the
    ambient umask, so a process killed in the microseconds before the
    ``chmod`` — or a database written by an older build — would
    otherwise stay world-readable for the life of the install, silently.
    The mode is compared first, so the common case costs a ``stat`` and
    no syscall beyond it. A filesystem without POSIX modes raises
    ``OSError`` and is tolerated: a readable database beats none.
    """
    for candidate in (
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ):
        try:
            if (candidate.stat().st_mode & 0o777) != 0o600:
                os.chmod(candidate, 0o600)
        except OSError:
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
    conn = sqlite3.connect(
        str(path), timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None
    )
    try:
        # Before anything is written.
        _enforce_private_mode(path)
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

    The guard is ``Exception``, not a list of storage errors. "Telemetry
    must never break ``serve``" is a promise about *every* way this code
    can fail, and the interesting ones are not all ``sqlite3.Error``: a
    key carrying a lone surrogate — exactly what ``os.fsdecode`` hands
    back for an undecodable byte in a filename — raises
    ``UnicodeEncodeError`` (a ``ValueError``) from inside
    ``conn.execute`` when the driver encodes the parameter, and a
    caller-supplied ``datetime`` can raise from ``astimezone``. A
    narrower guard would let those reach a request path, which is the
    one thing this module exists to prevent. ``KeyboardInterrupt``,
    ``SystemExit`` and ``GeneratorExit`` are ``BaseException`` and still
    propagate: swallowing a shutdown would be a different bug.
    """
    identity = _db_identity()
    try:
        return _attempt(work)
    except Exception as exc:
        # The quarantine decision needs its own guard. It runs inside a
        # handler, so anything it raises replaces ``exc`` and leaves the
        # module — and it calls ``db_path()`` again, which can start
        # failing between two calls if the environment changes under a
        # running process.
        try:
            retry = _is_corruption(exc) and _quarantine_corrupt_db(identity)
        except Exception:
            retry = False
        if not retry:
            return default
    try:
        return _attempt(work)
    except Exception:
        return default


def _valid_key(key: str) -> bool:
    """A key we are willing to put in the database.

    The UTF-8 check is not decoration: a lone surrogate (``"\\ud800"``,
    the normal result of ``os.fsdecode`` on an undecodable filename
    byte) is a perfectly ordinary ``str`` of ordinary length that the
    sqlite3 driver cannot encode. Rejecting it here turns a would-be
    exception into the documented "nothing to emit", and keeps the
    ``_run`` guard as a backstop rather than the only defence.
    """
    if not isinstance(key, str) or not 0 < len(key) <= MAX_KEY_LENGTH:
        return False
    try:
        key.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


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
    """The UTC calendar day ``value`` falls on, as ``YYYY-MM-DD``.

    A *naive* ``datetime`` is read as UTC rather than as local time.
    Every docstring here says UTC, and ``astimezone`` on a naive value
    silently assumes the machine's zone — so a caller that built its
    timestamp in UTC and dropped the tzinfo (the common shape) would
    land on the wrong day for every hour of offset around midnight, and
    would do it differently on each machine.
    """
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
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

    def work(conn: sqlite3.Connection) -> bool:
        day = _as_day(utc_date)
        now = datetime.now(timezone.utc)
        with _transaction(conn):
            cursor = conn.execute(
                "INSERT OR IGNORE INTO active_days (day, claimed_at) VALUES (?, ?)",
                (day, now.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )
            claimed = cursor.rowcount == 1
            if claimed:
                # Retention is measured from when the row was WRITTEN,
                # never from the day it names. Pruning by ``day`` looks
                # equivalent and is not: a row naming a date already past
                # the cutoff is born prunable, so the next claim of any
                # other day deletes it and that same date can be won all
                # over again — the once-per-day contract broken for
                # exactly the dates a wrong or rolled-back clock
                # produces. By write time, a row this install just
                # created survives its full window whatever date it
                # carries. ISO timestamps sort lexicographically, so the
                # string compare is a time compare. Only prune on the
                # claim that won — a losing racer has no reason to touch
                # the table.
                cutoff = (now - timedelta(days=ACTIVE_DAY_RETENTION_DAYS)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                conn.execute("DELETE FROM active_days WHERE claimed_at < ?", (cutoff,))
        return claimed

    return _run(work, False)


def note_model_served(model_id: str) -> int:
    """Record that ``model_id`` was served; return ``nth_model_served``.

    The returned count is the number of *distinct* models this install
    has ever served, including the one just noted — Orca's
    ``nth_repo_added`` idea, which lets any chart split newcomer from
    veteran with no server-side join. A successful call therefore
    always returns >= 1. A ``0`` return means "could not answer" — the
    id was unusable or storage failed — never "zero models served":
    map it to ``None`` for the telemetry wire, where ``common_props``
    omits the key and analysts read absence as "unknown", never as 0.
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


def _regular_file_date(path: Path) -> date | None:
    """Return a regular file's UTC mtime date without following symlinks."""
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            return None
        return datetime.fromtimestamp(info.st_mtime, timezone.utc).date()
    except Exception:
        return None


def _bounded_regular_file_text(path: Path) -> str | None:
    """Read a small regular file without following symlinks or blocking."""
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        return os.read(descriptor, _MAX_CONSENT_BYTES).decode("utf-8")
    except Exception:
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _seed_first_run_date(now: datetime | None) -> str:
    """Infer an upgrade's first-run date from existing local state.

    Evidence is read-only and independently best-effort: an unusable item is
    ignored without discarding dates recovered from the other items.
    """
    today = _as_day(now)
    state_dir = _default_telemetry_dir()
    evidence: list[date] = []
    for name in _INSTALL_EVIDENCE_FILES:
        if found := _regular_file_date(state_dir / name):
            evidence.append(found)

    try:
        for marker in state_dir.glob("activation_seen_*"):
            if found := _regular_file_date(marker):
                evidence.append(found)
    except Exception:
        pass

    consent_text = _bounded_regular_file_text(state_dir / "telemetry-consent.yaml")
    if consent_text is not None:
        try:
            consent = yaml.safe_load(consent_text)
            if isinstance(consent, dict) and "prompted_at" in consent:
                evidence.append(
                    datetime.strptime(
                        str(consent["prompted_at"]), "%Y-%m-%dT%H:%M:%SZ"
                    ).date()
                )
        except Exception:
            pass

    current = max(datetime.strptime(today, "%Y-%m-%d").date(), FIRST_RUN_EVIDENCE_FLOOR)
    valid_evidence = [
        candidate
        for candidate in evidence
        if FIRST_RUN_EVIDENCE_FLOOR <= candidate <= current
    ]
    return min(valid_evidence, default=current).isoformat()


def first_run_date(now: datetime | None = None) -> str | None:
    """Return the install's first-run date (``YYYY-MM-DD``), setting it once.

    Written by whichever process gets there first and never rewritten, so
    the cohort stamp is stable for the life of the install. An upgrade is
    seeded from the oldest pre-existing install evidence; a fresh 0.15.0
    install has no evidence and correctly starts at day 0. ``None`` on any
    storage failure.
    """

    def work(conn: sqlite3.Connection) -> str | None:
        with _transaction(conn):
            row = conn.execute(
                "SELECT value FROM install_facts WHERE key = 'first_run_date'"
            ).fetchone()
            if row is not None:
                return str(row[0])
            seeded = _seed_first_run_date(now)
            conn.execute(
                "INSERT INTO install_facts (key, value) VALUES ('first_run_date', ?)",
                (seeded,),
            )
        return seeded

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
        # ``_as_day`` runs outside ``_run`` here, so it carries its own
        # guard: a caller-supplied ``datetime`` can raise from
        # ``astimezone``, and a cohort hint is never worth an exception
        # on a request path.
        today = datetime.strptime(_as_day(now), "%Y-%m-%d").date()
    except Exception:
        return None
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
