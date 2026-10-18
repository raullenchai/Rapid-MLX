# SPDX-License-Identifier: Apache-2.0
"""Atomic SQLite index with one budgeted vector arena and cached file moments."""

from __future__ import annotations

import errno
import json
import os
import sqlite3
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np

from .chunking import CHUNKING_VERSION, Chunk
from .eligibility import ELIGIBILITY_VERSION, semantic_chunk
from .extraction import (
    AUDIO_SUFFIXES,
    EXTRACTION_VERSION,
    MAX_EMBEDDED_CHUNKS_PER_FILE,
    SKIP_REASONS,
    excluded,
    excluded_location,
    file_kind,
    skip_reason,
)

MAX_CORPUS_EMBEDDED_CHUNKS = 262_144
KEYWORD_CANDIDATE_LIMIT = 4096
MAX_NAME_ONLY_FILES_PER_FOLDER = 200_000
MAX_KEYWORD_TERMS = 64
# Maximum uncompressed UTF-8 vocabulary bytes per file, including separators.
# Overflow files get neutral support, never partial/truncated term evidence.
MAX_FILE_SIGNATURE_BYTES = 256 * 1024
SCHEMA_VERSION = "3"
DISTINCTIVE_FRACTION = 0.10
DISTINCTIVE_MIN_FILES = 1
# Retain at most a small vocabulary (roughly hundreds of KiB), not an unbounded
# history of user queries. Eviction affects computation only, never ranking.
FREQUENCY_CACHE_TERMS = 1024


def _owner_only(fd: int, mode: int) -> None:
    info = os.fstat(fd)
    if info.st_uid != os.geteuid():
        raise PermissionError("Local-search index must be owned by the current user")
    if stat.S_IMODE(info.st_mode) != mode:
        os.fchmod(fd, mode)
        if stat.S_IMODE(os.fstat(fd).st_mode) != mode:
            raise PermissionError("Could not make local-search index owner-only")


def _acl_present(fd: int) -> bool:
    if sys.platform != "darwin":
        return False
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    libc.acl_get_fd.argtypes = [ctypes.c_int]
    libc.acl_get_fd.restype = ctypes.c_void_p
    libc.acl_free.argtypes = [ctypes.c_void_p]
    libc.acl_free.restype = ctypes.c_int
    ctypes.set_errno(0)
    acl = libc.acl_get_fd(fd)
    if not acl:
        if ctypes.get_errno() == errno.ENOENT:
            return False
        raise PermissionError("Could not inspect local-search index ACL")
    if libc.acl_free(acl) != 0:
        raise PermissionError("Could not inspect local-search index ACL")
    return True


def _ancestor_acl_dangerous(fd: int) -> bool:
    """Only deny-only macOS ACLs are safe on the stable ancestor chain."""
    if sys.platform != "darwin":
        return False
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    libc.acl_get_fd.argtypes = [ctypes.c_int]
    libc.acl_get_fd.restype = ctypes.c_void_p
    libc.acl_get_entry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    libc.acl_get_entry.restype = ctypes.c_int
    libc.acl_get_tag_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    libc.acl_get_tag_type.restype = ctypes.c_int
    libc.acl_free.argtypes = [ctypes.c_void_p]
    libc.acl_free.restype = ctypes.c_int
    message = "Could not inspect local-search ancestor ACL"
    ctypes.set_errno(0)
    acl = libc.acl_get_fd(fd)
    if not acl:
        if ctypes.get_errno() == errno.ENOENT:
            return False
        raise PermissionError(message)
    try:
        entry = ctypes.c_void_p()
        tag = ctypes.c_int()
        position = 0  # ACL_FIRST_ENTRY
        while True:
            ctypes.set_errno(0)
            if libc.acl_get_entry(acl, position, ctypes.byref(entry)) != 0:
                # Darwin reports end-of-entries as EINVAL, not POSIX's zero.
                if ctypes.get_errno() == errno.EINVAL:
                    return False
                raise PermissionError(message)
            if libc.acl_get_tag_type(entry, ctypes.byref(tag)) != 0:
                raise PermissionError(message)
            # ACL_EXTENDED_DENY is 2; ALLOW (1) and unknown tags fail closed.
            if tag.value != 2:
                return True
            position = -1  # ACL_NEXT_ENTRY
    finally:
        if libc.acl_free(acl) != 0:
            raise PermissionError(message)


def _remove_acl(fd: int) -> None:
    if sys.platform != "darwin" or not _acl_present(fd):
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.acl_init.argtypes = [ctypes.c_int]
    libc.acl_init.restype = ctypes.c_void_p
    libc.acl_set_fd.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.acl_set_fd.restype = ctypes.c_int
    libc.acl_free.argtypes = [ctypes.c_void_p]
    libc.acl_free.restype = ctypes.c_int
    acl = libc.acl_init(0)
    if not acl:
        raise PermissionError("Could not secure local-search index ACL")
    try:
        if libc.acl_set_fd(fd, acl) != 0:
            raise PermissionError("Could not secure local-search index ACL")
    finally:
        libc.acl_free(acl)
    if _acl_present(fd):
        raise PermissionError("Could not remove local-search index ACL")


def _parent_safety_reason(directory: int, owner: int) -> str | None:
    """Explain whether another account can replace an entry owned by owner."""
    parent = os.fstat(directory)
    reason = None
    if parent.st_uid not in {0, os.geteuid()}:
        reason = "foreign owner"
    elif _ancestor_acl_dangerous(directory):
        reason = "ACL that grants access to other accounts"
    writable = parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    sticky_owner = parent.st_mode & stat.S_ISVTX and owner == os.geteuid()
    if writable and not sticky_owner:
        reason = "group- or world-writable without sticky ownership"
    return reason


def _resolve_index_path(path: Path) -> Path:
    """Resolve links only after checking their ownership and real parent chain."""
    absolute = path.absolute()
    real = Path(absolute.anchor)
    pending = list(absolute.parts[1:])
    hops = 0
    while pending:
        part = pending.pop(0)
        if part == "..":
            real = real.parent
            continue
        component = real / part
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            # mkdir will create the remainder beneath the now-real parent.
            return component.joinpath(*pending)
        if not stat.S_ISLNK(info.st_mode):
            real = component
            continue
        reason = (
            "foreign symlink owner" if info.st_uid not in {0, os.geteuid()} else None
        )
        if reason is None:
            try:
                directory = _open_stable_directory(real)
                try:
                    reason = _parent_safety_reason(directory, info.st_uid)
                finally:
                    os.close(directory)
            except PermissionError as exc:
                reason = str(exc)
        if reason:
            raise PermissionError(
                "Local-search index symlink can be replaced by another account: "
                f"{component} ({reason})"
            )
        hops += 1
        if hops > 40:
            raise PermissionError(
                f"Local-search index symlink hop limit exceeded: {component}"
            )
        target = Path(os.readlink(component))
        if target.is_absolute():
            real = Path(target.anchor)
            pending = list(target.parts[1:]) + pending
        else:
            pending = list(target.parts) + pending
    return real


def _open_stable_directory(path: Path) -> int:
    """Open a path whose components cannot be replaced by another account."""
    absolute = path.absolute()
    current = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    ancestor = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            component = ancestor / part
            try:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current
                )
            except OSError as exc:
                if exc.errno not in {errno.ENOTDIR, errno.ELOOP}:
                    raise
                raise PermissionError(
                    f"Local-search index path component {component} changed or is a symlink"
                ) from exc
            try:
                reason = _parent_safety_reason(current, os.fstat(child).st_uid)
                if reason:
                    raise PermissionError(
                        "Local-search index path can be replaced by another account: "
                        f"{ancestor} ({reason})"
                    )
            except Exception:
                os.close(child)
                raise
            os.close(current)
            current = child
            ancestor = component
        return current
    except Exception:
        os.close(current)
        raise


def _validate_index_entries(directory: int, names: set[str]) -> None:
    entries = set(os.listdir(directory))
    if entries - names or any(
        not stat.S_ISREG(os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode)
        for name in entries
    ):
        raise PermissionError(
            "Local-search index directory contains unrelated files; "
            "choose a dedicated index directory"
        )


def _prepare_index(path: Path) -> None:
    """Secure the index before SQLite can read, recover, or write any text."""
    names = {path.name + suffix for suffix in ("", "-wal", "-shm", "-journal")}
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = _open_stable_directory(path.parent)
    try:
        # Never chmod parents or a pre-existing directory containing unrelated
        # entries. A private mixed directory needs no chmod and is safe to use.
        needs_repair = stat.S_IMODE(os.fstat(directory).st_mode) != 0o700 or (
            sys.platform == "darwin" and _acl_present(directory)
        )
        if needs_repair:
            _validate_index_entries(directory, names)
        _owner_only(directory, 0o700)
        _remove_acl(directory)
        if sys.platform == "darwin" and _acl_present(directory):
            raise PermissionError("Could not remove local-search index ACL")
        if needs_repair:
            _validate_index_entries(directory, names)
        for name in sorted(names):
            flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
            if name == path.name:
                flags |= os.O_CREAT
            try:
                fd = os.open(name, flags, 0o600, dir_fd=directory)
            except FileNotFoundError:
                continue
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise PermissionError(
                        "Local-search index files must be regular files without hard links"
                    )
                _owner_only(fd, 0o600)
                _remove_acl(fd)
                if sys.platform == "darwin" and _acl_present(fd):
                    raise PermissionError("Could not remove local-search index ACL")
            finally:
                os.close(fd)
    finally:
        os.close(directory)


class SchemaVersionError(ValueError):
    """The index needs a newer application; opening it would be unsafe."""


class Store:
    def __init__(self, path: Path, model_id: str):
        if path.is_symlink():
            raise PermissionError(
                f"Local-search index file {path} is a symlink; "
                "choose a dedicated index directory"
            )
        path = _resolve_index_path(path)
        _prepare_index(path)
        self.path = path
        # SQLite's Unix VFS creates WAL/SHM/rollback journals with the database
        # mode, so precreation at 0600 protects even their first write. Existing
        # companions are repaired above before SQLite can recover their content.
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        # Refuse future indexes before any SQL schema/data/WAL change. The
        # permission/ACL hardening above runs first and never alters contents.
        try:
            if self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
            ).fetchone():
                incompatible_message = (
                    "Local Search index was created by a newer or incompatible "
                    "version. Update Rapid-MLX to open this index, or choose a "
                    "new index directory."
                )
                try:
                    version = self.db.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()
                except sqlite3.OperationalError as exc:
                    if not str(exc).startswith(("no such table:", "no such column:")):
                        raise
                    raise SchemaVersionError(incompatible_message) from exc
                if version is None:
                    # Released stores commit the version before accepting data.
                    # DDL is separate, so a crashed first init may leave empty tables.
                    tables = self.db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name IN ('folders','files','chunks')"
                    ).fetchall()
                    if any(
                        self.db.execute(f"SELECT 1 FROM {table[0]} LIMIT 1").fetchone()
                        for table in tables
                    ):
                        raise SchemaVersionError(incompatible_message)
                if version and version[0] not in ("1", "2", SCHEMA_VERSION):
                    raise SchemaVersionError(
                        f"Local Search index schema version {version[0]} is unsupported "
                        f"(this application supports {SCHEMA_VERSION}). Update Rapid-MLX "
                        "to open this index, or choose a new index directory."
                    )
        except BaseException:
            self.db.close()
            raise
        self.db.create_function(
            "local_basename",
            1,
            lambda path: path.rsplit("/", 1)[-1].casefold(),
            deterministic=True,
        )
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                state TEXT NOT NULL DEFAULT 'idle', files_skipped INTEGER NOT NULL DEFAULT 0,
                last_indexed_at REAL, error TEXT);
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY, folder_id TEXT NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
                path TEXT NOT NULL, title TEXT NOT NULL, size INTEGER NOT NULL,
                mtime REAL NOT NULL, mtime_ns INTEGER NOT NULL, content_hash TEXT,
                status TEXT NOT NULL, UNIQUE(folder_id, path));
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                text TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'text',
                locator TEXT NOT NULL, vector BLOB);
            CREATE INDEX IF NOT EXISTS chunks_file_id ON chunks(file_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, filename, tokenize='trigram');
            CREATE TRIGGER IF NOT EXISTS chunks_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid=old.id;
            END;
        """)
        version = self.db.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        previous = self.db.execute(
            "SELECT value FROM meta WHERE key='model_id'"
        ).fetchone()
        chunking = self.db.execute(
            "SELECT value FROM meta WHERE key='chunking_version'"
        ).fetchone()
        extraction = self.db.execute(
            "SELECT value FROM meta WHERE key='extraction_version'"
        ).fetchone()
        with self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS file_signatures (
                file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
                vocabulary BLOB)""")
            if version is None or version[0] != SCHEMA_VERSION:
                for file in self.db.execute("SELECT id FROM files"):
                    self._write_signature(
                        file[0],
                        (
                            row[0]
                            for row in self.db.execute(
                                "SELECT text FROM chunks WHERE file_id=? AND kind='text' ORDER BY id",
                                (file[0],),
                            )
                        ),
                    )
            if extraction is None or extraction[0] != EXTRACTION_VERSION:
                # Additive upgrade: keep unrelated chunks/vectors and refresh only
                # newly supported Office files, expanded DOCX, and empty results.
                self.db.execute("""UPDATE files SET status='pending'
                    WHERE lower(path) LIKE '%.docx' OR lower(path) LIKE '%.pptx'
                    OR lower(path) LIKE '%.xlsx'
                    OR (status IN ('indexed','keyword') AND NOT EXISTS (
                        SELECT 1 FROM chunks WHERE file_id=files.id))
                    OR (lower(path) LIKE '%.pdf' AND status LIKE 'skipped:%')""")
                for row in self.db.execute(
                    "SELECT id,path,status FROM files WHERE status LIKE 'skipped:%'"
                ).fetchall():
                    reason = skip_reason(row["status"].split(":", 1)[1].strip())
                    if Path(row["path"]).suffix.lower() in AUDIO_SUFFIXES:
                        reason = SKIP_REASONS["audio"]
                    self.db.execute(
                        "UPDATE files SET status=? WHERE id=?",
                        ("skipped: " + reason, row["id"]),
                    )
                self.db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('extraction_version',?)",
                    (EXTRACTION_VERSION,),
                )
            eligibility = self.db.execute(
                "SELECT value FROM meta WHERE key='eligibility_version'"
            ).fetchone()
            if eligibility is None or eligibility[0] != ELIGIBILITY_VERSION:
                # One streaming pass, no source reads or model work. Keep IDs,
                # FTS and locators; intentional keyword-only files are complete.
                cursor = self.db.execute("""SELECT chunks.id,text,files.path
                    FROM chunks JOIN files ON files.id=file_id
                    WHERE kind='text' AND vector IS NOT NULL""")
                for row in cursor:
                    if not semantic_chunk(Path(row["path"]), row["text"]):
                        self.db.execute(
                            "UPDATE chunks SET vector=NULL WHERE id=?", (row["id"],)
                        )
                for file in self.db.execute(
                    "SELECT id,path FROM files WHERE status='keyword'"
                ).fetchall():
                    texts = self.db.execute(
                        "SELECT text,kind FROM chunks WHERE file_id=?", (file["id"],)
                    ).fetchall()
                    if texts and all(
                        r["kind"] == "text"
                        and not semantic_chunk(Path(file["path"]), r["text"])
                        for r in texts
                    ):
                        self.db.execute(
                            "UPDATE files SET status='indexed' WHERE id=?",
                            (file["id"],),
                        )
                self.db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('eligibility_version',?)",
                    (ELIGIBILITY_VERSION,),
                )
            if (version and version[0] == "1") or (
                chunking is None or chunking[0] != CHUNKING_VERSION
            ):
                self.db.execute("UPDATE chunks SET vector=NULL")
                self.db.execute(
                    "UPDATE files SET status='pending' WHERE status IN ('indexed','keyword')"
                )
                # The media schema added support for formerly unsupported files;
                # size/password/private skips do not depend on chunking.
                for row in self.db.execute(
                    "SELECT id,path,status FROM files WHERE status LIKE 'skipped:%'"
                ).fetchall():
                    if (
                        file_kind(Path(row["path"])) != "text"
                        and skip_reason(row["status"].split(":", 1)[1].strip())
                        == SKIP_REASONS["unsupported"]
                    ):
                        self.db.execute(
                            "UPDATE files SET status='pending' WHERE id=?", (row["id"],)
                        )
            elif previous and previous[0] != model_id:
                # Keyword-only files/chunks have no model-dependent state.
                # Ordinary keyword-to-vector upgrades still run when available.
                self.db.execute(
                    """UPDATE files SET status='pending' WHERE id IN (
                        SELECT file_id FROM chunks WHERE vector IS NOT NULL
                    )"""
                )
                self.db.execute(
                    "UPDATE chunks SET vector=NULL WHERE vector IS NOT NULL"
                )
            self.db.executemany(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                [
                    ("schema_version", SCHEMA_VERSION),
                    ("model_id", model_id),
                    ("chunking_version", CHUNKING_VERSION),
                ],
            )
            self.db.execute(
                "UPDATE folders SET state='queued', error=NULL WHERE state IN ('indexing','queued')"
            )
            # Apply the embedding budget to older indexes without rechunking or
            # re-embedding. Chunk IDs follow source order; FTS/locators stay intact.
            self.db.execute(
                """UPDATE chunks SET vector=NULL WHERE vector IS NOT NULL AND id IN (
                    SELECT id FROM (
                        SELECT id, row_number() OVER (PARTITION BY file_id ORDER BY id) AS position
                        FROM chunks WHERE kind='text'
                    ) WHERE position > ?
                )""",
                (MAX_EMBEDDED_CHUNKS_PER_FILE,),
            )
        with self.db:
            self.db.execute(
                "UPDATE files SET status='keyword' WHERE id IN ("
                "SELECT file_id FROM chunks WHERE vector IS NOT NULL ORDER BY id LIMIT -1 OFFSET ?)",
                (MAX_CORPUS_EMBEDDED_CHUNKS,),
            )
            self.db.execute(
                "UPDATE chunks SET vector=NULL WHERE id IN ("
                "SELECT id FROM chunks WHERE vector IS NOT NULL ORDER BY id LIMIT -1 OFFSET ?)",
                (MAX_CORPUS_EMBEDDED_CHUNKS,),
            )
        # Reserve virtual address space once; only inserted rows become resident.
        # vectors contains views into this arena, never a second corpus copy.
        self._matrix: np.ndarray | None = None
        self._slots: dict[int, int] = {}
        self._free_slots: list[int] = []
        self._used_slots = 0
        self.vectors: dict[int, np.ndarray] = {}
        self._file_stats: dict[int, Any] = {}
        self._folder_stats: dict[str, Any] = {}
        self._view_scope: tuple[Any, set[str], set[str]] | None = None
        self._search_cache: dict[Any, dict[int, dict]] = {}
        self._frequency_cache: dict[str, int] = {}
        self._background_cache: tuple[dict[int, dict], dict[str, float]] | None = None
        self._score_scope: (
            tuple[Any, np.ndarray, np.ndarray, np.ndarray, int] | None
        ) = None
        self._keyword_scope: set[int] | None = None
        self._file_count: int | None = None
        for r in self.db.execute(
            "SELECT id,vector FROM chunks WHERE vector IS NOT NULL ORDER BY id"
        ):
            self._put_vector(r["id"], np.frombuffer(r["vector"], dtype="<f2"))
        for file in self.db.execute("SELECT id,folder_id FROM files"):
            self._refresh_file_stats(file["id"], file["folder_id"])

    def _put_vector(self, chunk_id, value):
        if self._matrix is None:
            self._matrix = np.empty(
                (MAX_CORPUS_EMBEDDED_CHUNKS, len(value)), dtype=np.float32
            )
        if len(value) != self._matrix.shape[1]:
            raise ValueError("inconsistent embedding dimensions")
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            slot = self._used_slots
            self._used_slots += 1
        self._matrix[slot] = value
        view = self._matrix[slot]
        view /= max(float(np.linalg.norm(view)), 1e-12)
        self._slots[chunk_id] = slot
        self.vectors[chunk_id] = view

    def _drop_vector(self, chunk_id):
        if chunk_id in self._slots:
            self._free_slots.append(self._slots.pop(chunk_id))
            self.vectors.pop(chunk_id)

    def _file_moments(self, members):
        # File-sized scratch is used only at startup or index changes. Never
        # retain one dimension-sized centroid per file (a second corpus at
        # singleton-file scale); retain IDs and aggregate folder moments.
        states = {}
        for kind, ids in members.items():
            total = np.zeros(self._matrix.shape[1], dtype=np.float64)
            for chunk_id in ids:
                total += self.vectors[chunk_id]
            states[kind] = [total, len(ids)]
        return states

    def _refresh_file_stats(self, file_id, folder_id):
        self._remove_file_stats(file_id)
        members = {}
        for row in self.db.execute(
            "SELECT id,kind FROM chunks WHERE file_id=?", (file_id,)
        ):
            if row["id"] in self.vectors:
                members.setdefault(row["kind"], []).append(row["id"])
        if members:
            self._file_stats[file_id] = (folder_id, members)
            self._adjust_file_stats(folder_id, self._file_moments(members), 1)

    def _remove_file_stats(self, file_id):
        old = self._file_stats.pop(file_id, None)
        if old is not None:
            folder_id, members = old
            self._adjust_file_stats(folder_id, self._file_moments(members), -1)

    def _adjust_file_stats(self, folder_id, states, sign):
        total = sum(value[0] for value in states.values())
        count = sum(value[1] for value in states.values())
        centroid = total / count
        if folder_id not in self._folder_stats:
            self._folder_stats[folder_id] = [
                np.zeros(len(total), dtype=np.float64),
                0.0,
                0,
                {},
            ]
        pooled = self._folder_stats[folder_id]
        pooled[0] += sign * centroid
        pooled[1] += sign * float(centroid @ centroid)
        pooled[2] += sign
        for kind, (total, count) in states.items():
            state = pooled[3].setdefault(
                kind, [np.zeros(len(total), dtype=np.float64), 0, 0, 0.0]
            )
            state[0] += sign * total
            state[1] += sign * count
            state[2] += sign * count**2
            state[3] += sign * float(total @ total)
        if not pooled[2]:
            del self._folder_stats[folder_id]

    def _invalidate_search(self):
        # Keep query work within twice the live corpus after deletions. Moving
        # rows in slot order is safe in place: every destination is <= source.
        if self._used_slots > 2 * len(self._slots):
            for slot, (chunk_id, old_slot) in enumerate(
                sorted(self._slots.items(), key=lambda item: item[1])
            ):
                self._matrix[slot] = self._matrix[old_slot]
                self._slots[chunk_id] = slot
                self.vectors[chunk_id] = self._matrix[slot]
            self._used_slots = len(self._slots)
            self._free_slots.clear()
        self._search_cache.clear()
        self._frequency_cache.clear()
        self._background_cache = None
        self._score_scope = None
        self._keyword_scope = None
        self._file_count = None
        self._view_scope = None

    def search_rows(self, folder_ids=None, kinds=None) -> dict[int, dict]:
        """Cache metadata, never the corpus bodies; callers treat rows as read-only.

        Invalidation follows committed index mutations. Scope caches are bounded
        to one view so arbitrarily many folder filters cannot retain the index.
        """
        key = (
            None if folder_ids is None else tuple(sorted(folder_ids)),
            None if kinds is None else tuple(sorted(kinds)),
        )
        if key not in self._search_cache:
            with self.db:
                self.db.execute("BEGIN")  # Pin metadata to one WAL snapshot.
                # File strings are shared by all of a file's chunks. Fetching them
                # once avoids allocating thousands of repeated paths/metadata rows.
                file_sql = "SELECT id,path,folder_id,title,size,mtime,content_hash FROM files WHERE status NOT LIKE 'skipped:%'"
                file_params = []
                if folder_ids is not None:
                    file_sql += (
                        " AND folder_id IN (" + ",".join("?" for _ in folder_ids) + ")"
                    )
                    file_params = folder_ids
                files = {
                    r["id"]: dict(r) for r in self.db.execute(file_sql, file_params)
                }
                sql = """SELECT chunks.id, chunks.file_id, chunks.kind, chunks.locator
                    FROM chunks JOIN files ON file_id=files.id"""
                clauses, params = [], []
                for column, values in (("folder_id", folder_ids), ("kind", kinds)):
                    if values is not None:
                        clauses.append(
                            column + " IN (" + ",".join("?" for _ in values) + ")"
                        )
                        params.extend(values)
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                self._search_cache.clear()
                self._search_cache[key] = {
                    r["id"]: {
                        **files[r["file_id"]],
                        **dict(r),
                        "content_indexed": True,
                        "skip_reason": None,
                    }
                    for r in self.db.execute(sql, params)
                }
                for row in self.name_only_rows(folder_ids):
                    kind = file_kind(Path(row["path"]))
                    if (folder_ids is None or row["folder_id"] in folder_ids) and (
                        kinds is None or kind in kinds
                    ):
                        self._search_cache[key][-row["id"]] = {
                            **row,
                            "id": -row["id"],
                            "file_id": row["id"],
                            "kind": kind,
                            "locator": "{}",
                            "content_indexed": False,
                            "skip_reason": row["status"].split(":", 1)[1].strip(),
                        }
                self._view_scope = (
                    self._search_cache[key],
                    {r["folder_id"] for r in self._search_cache[key].values()},
                    {r["kind"] for r in self._search_cache[key].values()},
                )
        return self._search_cache[key]

    def name_only_rows(self, folder_ids=None):
        scope = (
            ""
            if folder_ids is None
            else " AND folder_id IN (" + ",".join("?" for _ in folder_ids) + ")"
        )
        params = [
            "skipped: " + SKIP_REASONS["private"],
            *(folder_ids or []),
            MAX_NAME_ONLY_FILES_PER_FOLDER,
        ]
        return [
            dict(row)
            for row in self.db.execute(
                """
            SELECT * FROM (
                SELECT files.*, row_number() OVER (PARTITION BY folder_id ORDER BY id) AS ordinal
                FROM files WHERE status LIKE 'skipped:%' AND status != ?
            """
                + scope
                + ") WHERE ordinal <= ?",
                params,
            )
        ]

    def _write_signature(self, file_id, texts):
        import zlib

        tokens = set()
        size = 0
        for text in texts:
            for token in text.casefold().split():
                if token not in tokens:
                    size += len(token.encode("utf-8")) + 1
                    if size > MAX_FILE_SIGNATURE_BYTES:
                        self.db.execute(
                            "INSERT OR REPLACE INTO file_signatures VALUES (?,NULL)",
                            (file_id,),
                        )
                        return
                    tokens.add(token)
        vocabulary = zlib.compress("\n".join(sorted(tokens)).encode("utf-8"))
        self.db.execute(
            "INSERT OR REPLACE INTO file_signatures VALUES (?,?)",
            (file_id, vocabulary),
        )

    def query_evidence(self, candidates, rows, terms):
        """Bounded primary-key reads, independent of corpus postings/chunk count.

        Match substrings inside whitespace tokens, preserving identifiers,
        morphology and CJK runs. Unavailable signatures get neutral support
        without contributing to the file-evidence population.
        """
        import zlib

        from .search import file_evidence

        if not terms:
            return {}
        files = sorted(
            {rows[i]["file_id"] for i in candidates if rows[i]["kind"] == "text"}
        )
        wanted = {term.casefold() for term in terms}
        coverage = {}
        for file_id in files:
            row = self.db.execute(
                "SELECT vocabulary FROM file_signatures WHERE file_id=?", (file_id,)
            ).fetchone()
            vocabulary = None
            if row and row[0]:
                try:
                    decoder = zlib.decompressobj()
                    raw = decoder.decompress(row[0], MAX_FILE_SIGNATURE_BYTES + 1)
                    if (
                        len(raw) <= MAX_FILE_SIGNATURE_BYTES
                        and decoder.eof
                        and not decoder.unconsumed_tail
                        and not decoder.unused_data
                    ):
                        vocabulary = raw.decode("utf-8")
                except (zlib.error, UnicodeDecodeError):
                    pass
            if vocabulary is not None:
                coverage[file_id] = {term for term in wanted if term in vocabulary}
        support = file_evidence(coverage, wanted)
        return {i: support.get(rows[i]["file_id"], 0.0) for i in candidates}

    def chunk_text(self, chunk_id: int) -> str:
        row = self.db.execute(
            "SELECT text FROM chunks WHERE id=?", (chunk_id,)
        ).fetchone()
        return row[0] if row else ""

    def close(self):
        self.db.close()

    def _forget(self, chunk_ids: list[int]):
        live = {r[0] for r in self.db.execute("SELECT id FROM files")}
        for file_id in list(self._file_stats):
            if file_id not in live:
                self._remove_file_stats(file_id)
        for chunk_id in chunk_ids:
            self._drop_vector(chunk_id)
        self._invalidate_search()

    def add_folder(self, path: Path) -> tuple[str, bool]:
        nested = []
        for row in self.db.execute("SELECT id,path FROM folders"):
            existing = Path(row["path"])
            if path.is_relative_to(existing) and walk_reaches(existing, path):
                return row["id"], False
            if existing.is_relative_to(path) and walk_reaches(path, existing):
                nested.append(row["id"])
        folder_id = uuid.uuid4().hex
        with self.db:
            self.db.execute(
                "INSERT INTO folders(id,path,state) VALUES (?,?,'queued')",
                (folder_id, str(path)),
            )
            # Preserve file/chunk IDs, fingerprints, FTS entries and cached vectors.
            for absorbed_id in nested:
                self.db.execute(
                    "UPDATE files SET folder_id=? WHERE folder_id=?",
                    (folder_id, absorbed_id),
                )
                self.db.execute("DELETE FROM folders WHERE id=?", (absorbed_id,))
        for file_id, (old_folder, members) in list(self._file_stats.items()):
            if old_folder in nested:
                states = self._file_moments(members)
                self._adjust_file_stats(old_folder, states, -1)
                self._file_stats[file_id] = (folder_id, members)
                self._adjust_file_stats(folder_id, states, 1)
        self._invalidate_search()
        return folder_id, True

    def folder(self, folder_id: str) -> dict | None:
        return next((f for f in self.folders() if f["id"] == folder_id), None)

    def folders(self) -> list[dict]:
        folders = [
            dict(r)
            for r in self.db.execute("""
            SELECT folders.*, (SELECT count(*) FROM files WHERE folder_id=folders.id AND status IN ('indexed','keyword','pending','budget')) AS files_indexed,
            (SELECT count(*) FROM chunks JOIN files ON file_id=files.id WHERE folder_id=folders.id) AS chunks
            FROM folders ORDER BY path
        """)
        ]
        reasons: dict[str, dict[str, int]] = {folder["id"]: {} for folder in folders}
        for row in self.db.execute("""SELECT folder_id,status,count(*) AS count
            FROM files WHERE status LIKE 'skipped:%' GROUP BY folder_id,status"""):
            reason = skip_reason(row["status"].split(":", 1)[1].strip())
            counts = reasons[row["folder_id"]]
            counts[reason] = counts.get(reason, 0) + row["count"]
        for folder in folders:
            folder["skip_reasons"] = reasons[folder["id"]]
        return folders

    def set_folder(
        self, folder_id: str, state: str, *, error=None, skipped=None, completed=False
    ):
        with self.db:
            self.db.execute(
                "UPDATE folders SET state=?,error=? WHERE id=?",
                (state, error, folder_id),
            )
            if skipped is not None:
                self.db.execute(
                    "UPDATE folders SET files_skipped=? WHERE id=?",
                    (skipped, folder_id),
                )
            if completed:
                self.db.execute(
                    "UPDATE folders SET last_indexed_at=? WHERE id=?",
                    (time.time(), folder_id),
                )

    def delete_folder(self, folder_id: str):
        chunk_ids = [
            r[0]
            for r in self.db.execute(
                "SELECT chunks.id FROM chunks JOIN files ON file_id=files.id WHERE folder_id=?",
                (folder_id,),
            )
        ]
        with self.db:
            self.db.execute("DELETE FROM folders WHERE id=?", (folder_id,))
        self._forget(chunk_ids)

    def remove_file(self, file_id: int):
        chunk_ids = [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM chunks WHERE file_id=?",
                (file_id,),
            )
        ]
        with self.db:
            self.db.execute("DELETE FROM files WHERE id=?", (file_id,))
        self._forget(chunk_ids)

    def files(self, folder_id: str) -> dict[str, dict]:
        return {
            r["path"]: dict(r)
            for r in self.db.execute(
                "SELECT files.*, (SELECT count(vector) FROM chunks WHERE file_id=files.id) AS embedded FROM files WHERE folder_id=?",
                (folder_id,),
            )
        }

    def touch(self, file_id: int, info):
        with self.db:
            self.db.execute(
                "UPDATE files SET size=?,mtime=?,mtime_ns=? WHERE id=?",
                (info.st_size, info.st_mtime, info.st_mtime_ns, file_id),
            )

        self._invalidate_search()

    def replace(
        self,
        folder_id: str,
        path: Path,
        info,
        digest: str | None,
        chunks: list[Chunk],
        vectors,
        status: str,
    ):
        room = self.embedding_room(path)
        change = self._write_replace(
            folder_id, path, info, digest, chunks, vectors, status, room
        )
        self._publish_replace(change)

    def _write_replace(
        self, folder_id, path, info, digest, chunks, vectors, status, room
    ):
        old = self.db.execute(
            "SELECT id FROM files WHERE folder_id=? AND path=?", (folder_id, str(path))
        ).fetchone()
        additions = {}
        # Evict only after a successful commit; a failed transaction preserves
        # both old keyword results and the corresponding in-memory vectors.
        old_ids = (
            [
                r[0]
                for r in self.db.execute(
                    "SELECT id FROM chunks WHERE file_id=? AND vector IS NOT NULL",
                    (old[0],),
                )
            ]
            if old
            else []
        )
        with self.db:
            if old:
                self.db.execute("DELETE FROM files WHERE id=?", (old[0],))
            file_id = self.db.execute(
                """INSERT INTO files(folder_id,path,title,size,mtime,mtime_ns,content_hash,status)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    folder_id,
                    str(path),
                    path.stem,
                    info.st_size,
                    info.st_mtime,
                    info.st_mtime_ns,
                    digest,
                    status,
                ),
            ).lastrowid
            self._write_signature(
                file_id, (chunk.text for chunk in chunks if chunk.kind == "text")
            )
            for index, chunk in enumerate(chunks):
                vector = (
                    None
                    if vectors is None or vectors[index] is None
                    else np.asarray(vectors[index], dtype="<f2")
                )
                if vector is not None:
                    if room <= 0:
                        vector = None
                    else:
                        room -= 1
                chunk_id = self.db.execute(
                    "INSERT INTO chunks(file_id,text,kind,locator,vector) VALUES (?,?,?,?,?)",
                    (
                        file_id,
                        chunk.text,
                        chunk.kind,
                        json.dumps(
                            {
                                "line": chunk.line,
                                "page": chunk.page,
                                "timestamp": chunk.timestamp,
                            }
                        ),
                        None if vector is None else vector.tobytes(),
                    ),
                ).lastrowid
                self.db.execute(
                    "INSERT INTO chunks_fts(rowid,text,filename) VALUES (?,?,?)",
                    (chunk_id, chunk.text, path.name),
                )
                if vector is not None:
                    value = vector.astype(np.float32)
                    additions[chunk_id] = (value, chunk.kind)
        return folder_id, file_id, old[0] if old else None, old_ids, additions

    def _publish_replace(self, change):
        folder_id, file_id, old_file, old_ids, additions = change
        if old_file is not None:
            self._remove_file_stats(old_file)
        for chunk_id in old_ids:
            self._drop_vector(chunk_id)
        owner = self.db.execute(
            "SELECT folder_id FROM files WHERE id=?", (file_id,)
        ).fetchone()
        if owner is None:
            self._invalidate_search()
            return
        folder_id = owner[0]
        states = {}
        members = {}
        for chunk_id, (value, kind) in additions.items():
            members.setdefault(kind, []).append(chunk_id)
            self._put_vector(chunk_id, value)
            value = self.vectors[chunk_id]
            if kind not in states:
                states[kind] = [np.zeros(len(value), dtype=np.float64), 0]
            states[kind][0] += value
            states[kind][1] += 1
        if states:
            self._file_stats[file_id] = (folder_id, members)
            self._adjust_file_stats(folder_id, states, 1)
        self._invalidate_search()

    def rows(self, folder_ids: list[str] | None = None) -> dict[int, dict]:
        if folder_ids == []:
            return {}
        sql = "SELECT chunks.id, chunks.file_id, chunks.text, chunks.kind, chunks.locator, files.path, files.folder_id, files.title, files.size, files.mtime, files.id AS source_id FROM chunks JOIN files ON file_id=files.id"
        params = []
        if folder_ids is not None:
            sql += " WHERE folder_id IN (" + ",".join("?" for _ in folder_ids) + ")"
            params = folder_ids
        return {r["id"]: dict(r) for r in self.db.execute(sql, params)}

    def sql(self, statement, parameters=()):
        """Materialized diagnostic query; cursor never leaves its owning thread."""
        with self.db:
            return list(self.db.execute(statement, parameters))

    def embedding_room(self, path):
        old_count = self.db.execute(
            "SELECT count(vector) FROM chunks JOIN files ON files.id=file_id WHERE path=?",
            (str(path),),
        ).fetchone()[0]
        return max(0, MAX_CORPUS_EMBEDDED_CHUNKS - len(self.vectors) + old_count)

    def semantic_scores(self, query, allowed) -> list[tuple[int, float]]:
        if not self.vectors or not allowed:
            return []
        if self._score_scope is None or self._score_scope[0] is not allowed:
            rows = allowed if isinstance(allowed, dict) else self.search_rows()
            ids = [i for i in self.vectors if i in allowed]
            files: dict[int, int] = {}
            groups = np.array(
                [files.setdefault(rows[i]["file_id"], len(files)) for i in ids],
                dtype=np.int32,
            )
            self._score_scope = (
                allowed,
                np.array(ids, dtype=np.int64),
                np.array([self._slots[i] for i in ids], dtype=np.int32),
                groups,
                len(files),
            )
        _, scope_ids, slots, groups, count = self._score_scope
        vector = np.asarray(query, dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        # One BLAS call, one resident matrix; gather only scalar dot products.
        scores = (np.asarray(self._matrix)[: self._used_slots] @ vector)[slots]
        best = np.full(count, -np.inf, dtype=np.float32)
        np.maximum.at(best, groups, scores)
        winners = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(
            winners,
            groups,
            np.where(scores == best[groups], scope_ids, np.iinfo(np.int64).max),
        )
        order = np.lexsort((winners, -best))
        return [(int(winners[i]), float(best[i])) for i in order]

    def semantic(self, query, allowed: set[int]) -> list[int]:
        return [i for i, _ in self.semantic_scores(query, allowed)]

    def semantic_background(self, rows: dict[int, dict]) -> dict[str, float]:
        """Combine incrementally maintained folder/file moments, never vectors.

        Equal-file pooled and per-kind chunk backgrounds retain their formulas.
        The folder pool includes hidden kinds when the display is filtered.
        """
        if self._background_cache is not None and self._background_cache[0] is rows:
            return self._background_cache[1]
        if self._view_scope is not None and self._view_scope[0] is rows:
            _, folders, selected_kinds = self._view_scope
        else:
            folders = {r["folder_id"] for r in rows.values()}
            selected_kinds = {r["kind"] for r in rows.values()}
        stats: dict[str, list[Any]] = {
            kind: [None, 0, 0, 0.0] for kind in ("text", "image", "video")
        }
        pooled_total, pooled_squares, pooled_count = None, 0.0, 0
        # O(folders * dimensions), with no SQL or per-file/chunk work here.
        for folder_id in sorted(folders):
            pooled = self._folder_stats.get(folder_id)
            if pooled is None:
                continue
            if pooled_total is None:
                pooled_total = np.zeros(len(pooled[0]), dtype=np.float64)
            pooled_total += pooled[0]
            pooled_squares += pooled[1]
            pooled_count += pooled[2]
            for kind, values in pooled[3].items():
                if kind not in selected_kinds:
                    continue
                state = stats[kind]
                if state[0] is None:
                    state[0] = np.zeros(len(values[0]), dtype=np.float64)
                state[0] += values[0]
                for n in (1, 2, 3):
                    state[n] += values[n]
        pooled_similarity = 1.0
        if pooled_count > 1:
            pooled_similarity = max(
                0.0,
                min(
                    1.0,
                    (
                        float(np.asarray(pooled_total) @ np.asarray(pooled_total))
                        - pooled_squares
                    )
                    / (pooled_count * (pooled_count - 1)),
                ),
            )
        background = {}
        for kind, (total, count, squared_counts, squared_sums) in stats.items():
            pairs = count * count - squared_counts
            background[kind] = (
                max(
                    0.0,
                    min(
                        pooled_similarity, (float(total @ total) - squared_sums) / pairs
                    ),
                )
                if pairs
                else 0.0
            )
        self._background_cache = (rows, background)
        return background

    @staticmethod
    def _match_expression(terms: list[str]) -> str:
        return " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)

    def distinctive_terms(self, terms: list[str]) -> list[str]:
        """File frequency, independent of chunk count and embedding coverage."""
        if self._file_count is None:
            self._file_count = self.db.execute(
                "SELECT count(DISTINCT file_id) FROM chunks"
            ).fetchone()[0]
        count = self._file_count
        threshold = max(DISTINCTIVE_MIN_FILES, int(count * DISTINCTIVE_FRACTION))
        distinctive = []
        for term in terms:
            folded = term.casefold()
            if folded not in self._frequency_cache:
                if len(term) >= 3:
                    frequency_rows = self.db.execute(
                        "SELECT DISTINCT file_id FROM chunks_fts "
                        "JOIN chunks ON chunks.id=chunks_fts.rowid WHERE chunks_fts MATCH ? LIMIT ?",
                        (self._match_expression([term]), threshold + 1),
                    ).fetchall()
                    frequency = len(frequency_rows)
                else:
                    # Trigrams cannot index short terms. Stop once enough files
                    # prove a term common, rather than scanning every chunk.
                    if term.isascii() or term.lower() == term.upper():
                        literal = (
                            "%"
                            + term.replace("\\", "\\\\")
                            .replace("%", "\\%")
                            .replace("_", "\\_")
                            + "%"
                        )
                        frequency = len(
                            self.db.execute(
                                "SELECT DISTINCT file_id FROM chunks_fts "
                                "JOIN chunks ON chunks.id=chunks_fts.rowid "
                                "WHERE chunks_fts.text LIKE ? ESCAPE '\\' OR filename LIKE ? ESCAPE '\\' LIMIT ?",
                                (literal, literal, threshold + 1),
                            ).fetchall()
                        )
                    else:
                        # Preserve Unicode casefold expansion for cased scripts.
                        files = set()
                        for r in self.db.execute(
                            "SELECT file_id,text,path FROM chunks JOIN files ON file_id=files.id"
                        ):
                            if (
                                folded in r["text"].casefold()
                                or folded in Path(r["path"]).name.casefold()
                            ):
                                files.add(r["file_id"])
                                if len(files) > threshold:
                                    break
                        frequency = len(files)
                # Bound retained query vocabulary independently of corpus size.
                if len(self._frequency_cache) >= FREQUENCY_CACHE_TERMS:
                    self._frequency_cache.clear()
                self._frequency_cache[folded] = frequency
            if self._frequency_cache[folded] <= threshold:
                distinctive.append(term)
        return distinctive

    def keyword(
        self,
        terms: list[str],
        allowed: set[int],
        *,
        query: str | None = None,
        distinctive: list[str] | None = None,
        evidence: dict[int, int] | None = None,
    ) -> list[int]:
        from .search import filename_match, lexical_tier

        terms = bounded_terms(terms, distinctive)
        if distinctive is not None:
            distinctive = [term for term in distinctive if term in terms]

        if not allowed:
            return []
        if self._keyword_scope is not allowed:
            self.db.execute(
                "CREATE TEMP TABLE IF NOT EXISTS local_allowed(id INTEGER PRIMARY KEY)"
            )
            self.db.execute("DELETE FROM local_allowed")
            self.db.executemany(
                "INSERT INTO local_allowed VALUES (?)", ((i,) for i in allowed)
            )
            self.db.commit()  # End the temporary-table transaction before WAL reads.
            self._keyword_scope = allowed
        precise = self.distinctive_terms(terms) if distinctive is None else distinctive
        precise = list(
            dict.fromkeys(
                precise
                + [term for term in terms if any(mark in term[1:-1] for mark in "-./_")]
            )
        )
        phrase = (query if query is not None else " ".join(terms)).strip()
        if not precise:
            if not phrase:
                return []
            precise = [phrase]
        long_terms = [term for term in precise if len(term) >= 3]
        if long_terms:
            candidates = list(
                self.db.execute(
                    "SELECT rowid,text,filename FROM chunks_fts WHERE chunks_fts MATCH ? AND EXISTS (SELECT 1 FROM local_allowed WHERE id=chunks_fts.rowid) "
                    "ORDER BY bm25(chunks_fts),rowid LIMIT "
                    + str(KEYWORD_CANDIDATE_LIMIT),
                    (self._match_expression(long_terms),),
                )
            )
        elif all(term.isascii() or term.lower() == term.upper() for term in precise):
            # LIKE has no short trigram postings, but does the unavoidable scan
            # inside SQLite and materializes only matching bodies in Python.
            patterns = [
                "%"
                + term.casefold()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
                + "%"
                for term in precise
            ]
            where = " AND ".join(
                "(text LIKE ? ESCAPE '\\' OR filename LIKE ? ESCAPE '\\')"
                for _ in patterns
            )
            candidates = list(
                self.db.execute(
                    "SELECT rowid,text,filename FROM chunks_fts WHERE "
                    + where
                    + " AND EXISTS (SELECT 1 FROM local_allowed WHERE id=chunks_fts.rowid) ORDER BY rowid LIMIT "
                    + str(KEYWORD_CANDIDATE_LIMIT),
                    [pattern for pattern in patterns for _ in range(2)],
                )
            )
        else:
            candidates = list(
                self.db.execute(
                    "SELECT rowid,text,filename FROM chunks_fts WHERE rowid IN (SELECT id FROM local_allowed) ORDER BY rowid LIMIT "
                    + str(KEYWORD_CANDIDATE_LIMIT)
                )
            )
        # Full basename evidence is independent of body-frequency admission.
        # This preserves filenames whose terms are common throughout the corpus.
        filename_terms = [term for term in terms if len(term) >= 3]
        if filename_terms:
            existing = {r[0] for r in candidates}
            candidates.extend(
                r
                for r in self.db.execute(
                    "SELECT rowid,text,filename FROM chunks_fts WHERE chunks_fts MATCH ? AND EXISTS (SELECT 1 FROM local_allowed WHERE id=chunks_fts.rowid) "
                    "ORDER BY bm25(chunks_fts),rowid LIMIT "
                    + str(KEYWORD_CANDIDATE_LIMIT),
                    ("filename : (" + self._match_expression(filename_terms) + ")",),
                )
                if r[0] not in existing
            )
        if terms:
            patterns = [
                "%"
                + term.casefold()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
                + "%"
                for term in terms
            ]
            where = " AND ".join(
                "local_basename(path) LIKE ? ESCAPE '\\'" for _ in patterns
            )
            existing = {r[0] for r in candidates}
            candidates.extend(
                (row["id"], row["text"], Path(row["path"]).name)
                for row in self.db.execute(
                    "SELECT chunks.id,chunks.text,files.path FROM local_allowed "
                    "JOIN chunks ON chunks.id=local_allowed.id "
                    "JOIN files ON files.id=chunks.file_id WHERE "
                    + where
                    + " ORDER BY chunks.id LIMIT "
                    + str(KEYWORD_CANDIDATE_LIMIT),
                    patterns,
                )
                if row["id"] not in existing
            )
            candidates.extend(
                (-row["id"], "", Path(row["path"]).name)
                for row in self.db.execute(
                    "SELECT files.id,path FROM local_allowed JOIN files ON files.id=-local_allowed.id WHERE local_allowed.id<0 AND "
                    + where
                    + " ORDER BY local_allowed.id DESC LIMIT "
                    + str(KEYWORD_CANDIDATE_LIMIT),
                    patterns,
                )
            )
        ranked, strengths, name_matches = [], {}, {}
        for chunk_id, text, filename in candidates:
            body, name = text.casefold(), filename.casefold()
            if not (
                all(t.casefold() in body or t.casefold() in name for t in precise)
                or (terms and all(t.casefold() in name for t in terms))
            ):
                continue
            strengths[chunk_id] = lexical_tier(phrase, terms, text, filename)
            name_matches[chunk_id] = filename_match(phrase, terms, filename)
            ranked.append(chunk_id)
        # Filename hits share the top tier, with exact stems then token coverage.
        # Stable BM25 ties and content-only ordering remain unchanged.
        ranked.sort(
            key=lambda i: (-strengths[i], -name_matches[i][0], -name_matches[i][1])
        )
        if evidence is not None:
            evidence.update(strengths)
        return ranked

    def totals(self) -> dict:
        from .extraction import file_kind

        counts = {"text": 0, "image": 0, "video": 0}
        chunks, embedded = self.db.execute(
            "SELECT count(*), count(vector) FROM chunks"
        ).fetchone()
        for row in self.db.execute(
            "SELECT path FROM files WHERE status IN ('indexed','keyword','pending','budget')"
        ):
            counts[file_kind(Path(row["path"]))] += 1
        return {
            "files_by_kind": counts,
            "name_only_files": self.db.execute(
                """
                SELECT coalesce(sum(min(n,?)),0) FROM (
                    SELECT count(*) AS n FROM files WHERE status LIKE 'skipped:%'
                    AND status != ? GROUP BY folder_id)
            """,
                (MAX_NAME_ONLY_FILES_PER_FOLDER, "skipped: " + SKIP_REASONS["private"]),
            ).fetchone()[0],
            "files": self.db.execute(
                "SELECT count(*) FROM files WHERE status IN ('indexed','keyword','pending','budget')"
            ).fetchone()[0],
            "chunks": chunks,
            "embedded_chunks": embedded,
            "semantic_index_full": embedded >= MAX_CORPUS_EMBEDDED_CHUNKS,
            "semantic_chunk_budget": MAX_CORPUS_EMBEDDED_CHUNKS,
            "keyword_only_chunks": chunks - embedded,
            "index_bytes": sum(
                p.stat().st_size
                for p in (
                    self.path,
                    Path(str(self.path) + "-wal"),
                    Path(str(self.path) + "-shm"),
                )
                if p.exists()
            ),
        }


class StoreWorker:
    """A single CPU worker owns SQLite, including construction and teardown.

    Synchronous access is for offline inspection only; loop callers await call().
    """

    def __init__(self, *args):
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="local-search-db"
        )
        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="local-search-writer"
        )
        self._ready = self._executor.submit(Store, *args)
        self.path = args[0]

    @property
    def _store(self):
        return self._ready.result()

    def __getattr__(self, name):
        value = self._executor.submit(getattr, self._store, name).result()
        if callable(value):
            return lambda *args, **kwargs: self._executor.submit(
                value, *args, **kwargs
            ).result()
        return value

    async def ready(self) -> None:
        """Await construction separately from input validation and store calls."""
        import asyncio

        await asyncio.wrap_future(self._ready)

    @overload
    async def call(self, name: Literal["folder"], folder_id: str) -> dict | None: ...

    @overload
    async def call(self, name: Literal["folders"]) -> list[dict]: ...

    @overload
    async def call(self, name: str, *args: Any, **kwargs: Any) -> Any: ...

    async def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        import asyncio
        from functools import partial

        loop = asyncio.get_running_loop()
        await self.ready()

        async def replace():
            room = await loop.run_in_executor(
                self._executor, self._store.embedding_room, args[1]
            )
            change = await loop.run_in_executor(
                self._writer, partial(self._write_file, args, kwargs, room)
            )
            await loop.run_in_executor(
                self._executor, self._store._publish_replace, change
            )

        if name == "replace" and name not in self.__dict__:
            task = asyncio.create_task(replace())
        else:
            task = asyncio.ensure_future(
                loop.run_in_executor(
                    self._executor,
                    partial(
                        self.__dict__.get(name, getattr(self._store, name)),
                        *args,
                        **kwargs,
                    ),
                )
            )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def _write_file(self, args, kwargs, room):
        # A WAL writer publishes atomically; searches use the separate owning
        # reader connection and never queue behind a whole-file FTS transaction.
        writer = Store.__new__(Store)
        writer.db = sqlite3.connect(self._store.path)
        writer.db.row_factory = sqlite3.Row
        writer.db.execute("PRAGMA foreign_keys=ON")
        try:
            return writer._write_replace(*args, **kwargs, room=room)
        finally:
            writer.db.close()

    def close(self):
        self._writer.shutdown()
        try:
            # A refused open has no connection, but its workers still need teardown.
            if self._ready.exception() is None:
                self._executor.submit(self._store.close).result()
        finally:
            self._executor.shutdown()


def bounded_terms(terms, distinctive=None):
    """Bound SQL expression depth; prefer rare terms, then longer literals."""
    unique = list(dict.fromkeys(terms))
    if len(unique) <= MAX_KEYWORD_TERMS:
        return unique
    rare = set(distinctive or ())
    return sorted(unique, key=lambda term: (term not in rare, -len(term), term))[
        :MAX_KEYWORD_TERMS
    ]


def walk_reaches(root, descendant):
    """A selected root overrides location pruning only within its own subtree."""
    for relative in [
        Path(),
        *descendant.relative_to(root).parents,
        descendant.relative_to(root),
    ]:
        path = root / relative
        if path == root:
            continue
        if (
            excluded(path, directory=True)
            or excluded_location(path, root)
            or (path / "pyvenv.cfg").exists()
        ):
            return False
    return True
