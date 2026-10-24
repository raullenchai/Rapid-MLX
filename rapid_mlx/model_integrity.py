# SPDX-License-Identifier: Apache-2.0
"""Verify pulled snapshots against immutable Hugging Face object metadata.

This proves content identity, not publisher authenticity. Receipts avoid hashing
unchanged cache blobs repeatedly; they are invalidated by file identity/timestamps.
"""

from __future__ import annotations

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


def verify_snapshot(
    repo_id,
    snapshot_path,
    *,
    revision=None,
    allow_patterns=None,
    ignore_patterns=None,
    filenames=None,
    resolved_revision=None,
    local_files_only=False,
    token=None,
):
    """Fail before pull completion when selected files differ from HF's commit.

    A completed receipt also permits offline verification of unchanged files.
    Cache directories are locally trusted, as with the Hugging Face cache itself.
    No files are removed: interrupted transfers retain their resume data.
    """
    path = Path(snapshot_path)
    commit = resolved_revision or path.name
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or (revision and revision != commit):
        raise ModelIntegrityError(
            "Download did not resolve to the requested immutable snapshot"
        )
    allow_all = allow_patterns is None
    patterns = sorted(
        [allow_patterns] if isinstance(allow_patterns, str) else allow_patterns or []
    )
    ignored = sorted(
        [ignore_patterns] if isinstance(ignore_patterns, str) else ignore_patterns or []
    )
    exact = sorted(filenames) if filenames is not None else None
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
        json.dumps(
            [
                repo_id,
                commit,
                patterns,
                allow_all,
                ignored,
                exact,
                str(path.absolute()),
                lock_identity,
            ]
        ).encode()
    ).hexdigest()
    receipt_root = (
        path / ".cache" / "rapid-mlx-integrity"
        if resolved_revision
        else path.parent.parent / ".rapid-mlx-integrity"
    )
    receipt_path = receipt_root / (key + ".json")
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
            {"rfilename": name, "size": entry.get("size"), "blob_id": entry["blob_id"]}
            for name, entry in locked.get("git_files", {}).items()
        ] + [
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
    elif local_files_only:
        # A partial warm checkpoint may be covered by a prior broader pull.
        # Reuse only exact-snapshot, manifest-scoped digest evidence; modified
        # bytes still go through hashing below, never a presence-only shortcut.
        known = {}
        for receipt in receipt_root.glob("*.json"):
            try:
                proof = json.loads(receipt.read_text())
                if (
                    proof.get("repo"),
                    proof.get("commit"),
                    proof.get("snapshot"),
                    proof.get("lock_identity"),
                ) != (repo_id, commit, str(path.absolute()), lock_identity):
                    continue
                known.update(proof["files"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if exact is None or not exact or any(name not in known for name in exact):
            raise ModelIntegrityError(
                "Offline integrity verification requires a packaged manifest or verified receipts for selected files; run an online pull first"
            )
        siblings = []
        for name in exact:
            digest = known[name]["digest"]
            siblings.append(
                {
                    "rfilename": name,
                    "size": known[name]["stat"][2],
                    "lfs": {"sha256": digest} if len(digest) == 64 else None,
                    "blob_id": digest if len(digest) == 40 else None,
                }
            )
    else:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RevisionNotFoundError

        from rapid_mlx._download_gate import call_with_deadline

        try:
            info = call_with_deadline(
                lambda: HfApi().model_info(
                    repo_id, revision=commit, files_metadata=True, token=token
                ),
                30,
            )
        except RevisionNotFoundError:
            if revision != commit:
                raise
            # Preserve the mirror's explicit-pin recovery trust contract when
            # upstream has lost a commit. Reuse its strict complete-inventory
            # validator, then independently hash the installed selected bytes.
            from rapid_mlx._mirror import _mirror_base, _revision_manifest_info

            try:
                info = _revision_manifest_info(_mirror_base(), repo_id, commit)
            except RuntimeError as exc:
                raise ModelIntegrityError(str(exc)) from exc
        if _field(info, "sha") != commit:
            raise ModelIntegrityError(
                "Hub integrity manifest does not match the downloaded commit"
            )
        siblings = _field(info, "siblings", [])
    siblings = list({_field(item, "rfilename"): item for item in siblings}.values())
    from huggingface_hub.utils import filter_repo_objects

    names = [_field(item, "rfilename") for item in siblings]
    if not all(isinstance(name, str) for name in names):
        raise ModelIntegrityError("Invalid integrity manifest filename")
    if exact is not None and any(name not in names for name in exact):
        raise ModelIntegrityError(
            "Requested file is absent from the immutable integrity manifest"
        )
    selected = set(
        filter_repo_objects(
            names,
            allow_patterns=None if allow_all else patterns,
            ignore_patterns=ignored,
        )
    )
    verified = {}
    for sibling in siblings:
        name = _field(sibling, "rfilename")
        if not isinstance(name, str):
            raise ModelIntegrityError("Invalid integrity manifest filename")
        if exact is not None and name not in exact:
            continue
        if name not in selected:
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
        if not selected:
            return
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
                    "snapshot": str(path.absolute()),
                    "lock_identity": lock_identity,
                    "files": verified,
                },
                handle,
            )
        os.replace(temporary, receipt_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_cached_checkpoint(repo_id, checkpoint_path, *, revision=None):
    """Verify existing checkpoint files at the loader boundary, without fetching.

    The caller must first establish its runtime-specific completeness contract.
    Optional repository assets need not be present; every existing selected file
    must belong to and match the immutable manifest. Inventory scans don't call
    this function. Explicit local model directories retain local semantics.
    """
    from rapid_mlx.model_catalog import resolve_repository_identity

    canonical, catalog_revision = resolve_repository_identity(repo_id)
    revision = revision or catalog_revision
    if revision is None:
        return
    checkpoint = Path(checkpoint_path)
    snapshot = next(
        (p for p in (checkpoint, *checkpoint.parents) if p.name == revision), None
    )
    if snapshot is None:
        raise ModelIntegrityError(
            "Cached checkpoint is not inside its immutable snapshot"
        )
    filenames = sorted(
        str(path.relative_to(snapshot).as_posix())
        for path in checkpoint.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
    )
    if not filenames:
        raise ModelIntegrityError("Cached checkpoint contains no verifiable files")
    verify_snapshot(
        canonical,
        snapshot,
        revision=revision,
        filenames=filenames,
        local_files_only=True,
    )
