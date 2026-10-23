# SPDX-License-Identifier: Apache-2.0
"""Verify pulled snapshots against immutable Hugging Face object metadata.

This proves content identity, not publisher authenticity. Receipts avoid hashing
unchanged cache blobs repeatedly; they are invalidated by file identity/timestamps.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath


class ModelIntegrityError(RuntimeError):
    pass


def _field(value, key, default=None):
    return (
        value.get(key, default)
        if isinstance(value, dict)
        else getattr(value, key, default)
    )


def _identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def verify_snapshot(repo_id, snapshot_path, *, revision=None, allow_patterns=None):
    """Fail before pull completion when selected files differ from HF's commit.

    A completed receipt also permits offline verification of unchanged files.
    Cache directories are locally trusted, as with the Hugging Face cache itself.
    No files are removed: interrupted transfers retain their resume data.
    """
    path = Path(snapshot_path)
    commit = path.name
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or (revision and revision != commit):
        raise ModelIntegrityError(
            "Download did not resolve to the requested immutable snapshot"
        )
    patterns = sorted(allow_patterns or [])
    lock_path = Path(__file__).with_name("model_catalog_lock.json")
    locked = (
        json.loads(lock_path.read_text()).get("repositories", {}).get(repo_id, {})
        if lock_path.is_file()
        else {}
    )
    if locked and locked.get("revision") != commit:
        raise ModelIntegrityError(
            "Downloaded snapshot differs from the catalog integrity manifest"
        )
    lock_identity = hashlib.sha256(
        json.dumps(locked, sort_keys=True).encode()
    ).hexdigest()
    key = hashlib.sha256(
        json.dumps([repo_id, commit, patterns, lock_identity]).encode()
    ).hexdigest()
    receipt_path = path.parent.parent / ".rapid-mlx-integrity" / (key + ".json")
    previous = {}
    try:
        previous = json.loads(receipt_path.read_text())
        if (previous.get("repo"), previous.get("commit"), previous.get("patterns")) != (
            repo_id,
            commit,
            patterns,
        ):
            previous = {}
        entries = previous.get("files", {})
        if entries and all(
            _identity(path / name) == entry["stat"] for name, entry in entries.items()
        ):
            return
    except (OSError, ValueError, KeyError, TypeError):
        previous = {}
    if locked:
        # The packaged manifest is authoritative; remote omission cannot drop
        # a reviewed weight/config from verification, and warm pulls work offline.
        siblings = [
            {
                "rfilename": name,
                "size": entry.get("size"),
                "lfs": {"sha256": entry["sha256"]},
            }
            for name, entry in {
                **locked.get("configs", {}),
                **locked.get("lfs_files", {}),
            }.items()
        ]
    else:
        from huggingface_hub import HfApi

        from rapid_mlx._download_gate import call_with_deadline

        info = call_with_deadline(
            lambda: HfApi().model_info(repo_id, revision=commit, files_metadata=True),
            30,
        )
        if _field(info, "sha") != commit:
            raise ModelIntegrityError(
                "Hub integrity manifest does not match the downloaded commit"
            )
        siblings = _field(info, "siblings", [])
    verified = {}
    for sibling in siblings:
        name = _field(sibling, "rfilename")
        if not isinstance(name, str):
            raise ModelIntegrityError("Invalid integrity manifest filename")
        if patterns and not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            continue
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ModelIntegrityError("Unsafe integrity manifest filename")
        lfs = _field(sibling, "lfs")
        digest = _field(lfs, "sha256") if lfs else _field(sibling, "blob_id")
        algorithm = "sha256" if lfs else "sha1"
        reviewed = locked.get("lfs_files", {}).get(name) or locked.get(
            "configs", {}
        ).get(name)
        if reviewed:
            digest = reviewed["sha256"]
            algorithm = "sha256"
        if not isinstance(digest, str) or not re.fullmatch(
            r"[0-9a-f]{" + ("64" if algorithm == "sha256" else "40") + "}", digest
        ):
            raise ModelIntegrityError(f"Missing content hash for {name}")
        file = path / name
        try:
            before = _identity(file)
            expected_size = _field(sibling, "size")
            if expected_size is not None and before[2] != expected_size:
                raise ModelIntegrityError(f"Downloaded file size mismatch: {name}")
            old = previous.get("files", {}).get(name, {})
            if old.get("digest") != digest or old.get("stat") != before:
                hasher = hashlib.new(algorithm)
                if algorithm == "sha1":
                    hasher.update(f"blob {before[2]}\0".encode())
                with file.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        hasher.update(chunk)
                if hasher.hexdigest() != digest or before != _identity(file):
                    raise ModelIntegrityError(
                        f"Downloaded content hash mismatch: {name}; repair this cached file and retry pull"
                    )
            verified[name] = {"digest": digest, "stat": before}
        except OSError as exc:
            raise ModelIntegrityError(f"Cannot verify downloaded file: {name}") from exc
    if not verified:
        raise ModelIntegrityError("Integrity manifest contains no selected files")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    descriptor, temporary = tempfile.mkstemp(
        dir=receipt_path.parent, prefix=".receipt-"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(
                {
                    "repo": repo_id,
                    "commit": commit,
                    "patterns": patterns,
                    "files": verified,
                },
                handle,
            )
        os.replace(temporary, receipt_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
