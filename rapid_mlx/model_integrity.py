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
from contextlib import suppress
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
    ``revision`` and ``resolved_revision`` accept only full lowercase commit
    SHAs, never symbolic revisions. Callers resolve branch/tag names beforehand.
    """
    path = Path(snapshot_path)
    commit = path.name if resolved_revision is None else resolved_revision
    if (
        not isinstance(commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", commit)
        or (revision is not None and revision != commit)
    ):
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
    exact = (
        sorted([filenames] if isinstance(filenames, str) else filenames)
        if filenames is not None
        else None
    )
    from huggingface_hub.utils import filter_repo_objects

    def inventory():
        try:
            local_names = [
                file.relative_to(path).as_posix()
                for file in path.rglob("*")
                if file.is_file()
                and not file.relative_to(path)
                .as_posix()
                .startswith((".cache/huggingface/", ".cache/rapid-mlx-integrity/"))
            ]
        except OSError as exc:
            raise ModelIntegrityError(
                f"Cannot inventory downloaded snapshot at {path}; check cache access and retry pull"
            ) from exc
        selected_files = set(
            filter_repo_objects(
                local_names,
                allow_patterns=None if allow_all else patterns,
                ignore_patterns=ignored,
            )
        )
        if exact is not None:
            selected_files.intersection_update(exact)
        return selected_files

    def ensure_unchanged(entries):
        try:
            unchanged = inventory() == set(entries) and all(
                _identity(path / name) == entry["stat"]
                for name, entry in entries.items()
            )
        except OSError as exc:
            raise ModelIntegrityError(
                "Snapshot changed during verification; retry pull"
            ) from exc
        if not unchanged:
            raise ModelIntegrityError(
                "Snapshot changed during verification; retry pull"
            )

    local_selected = inventory()
    lock_path = Path(__file__).with_name("model_catalog_lock.json")
    try:
        locked = (
            json.loads(lock_path.read_text()).get("repositories", {}).get(repo_id, {})
            if lock_path.is_file()
            else {}
        )
        if not isinstance(locked, dict):
            raise ValueError("Invalid packaged repository manifest")
        for collection, digest_key, length in (
            ("git_files", "blob_id", 40),
            ("configs", "sha256", 64),
            ("lfs_files", "sha256", 64),
        ):
            entries = locked.get(collection, {})
            if not isinstance(entries, dict):
                raise ValueError("Invalid packaged file collection")
            for name, entry in entries.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(entry, dict)
                    or not isinstance(entry.get(digest_key), str)
                    or not re.fullmatch(
                        r"[0-9a-f]{" + str(length) + "}", entry[digest_key]
                    )
                ):
                    raise ValueError("Invalid packaged file entry")
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        raise ModelIntegrityError(
            "Cannot read packaged integrity manifest; repair the installation and retry"
        ) from exc
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
        if not isinstance(entries, dict):
            raise ValueError("Invalid receipt file inventory")
        if (
            entries
            and set(entries) == local_selected
            and all(
                _identity(path / name) == entry["stat"]
                for name, entry in entries.items()
            )
        ):
            ensure_unchanged(entries)
            return
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
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
    else:
        from huggingface_hub import HfApi

        from rapid_mlx._download_gate import call_with_deadline

        if local_files_only:
            raise ModelIntegrityError(
                "Offline integrity verification requires a packaged manifest or an unchanged verified receipt; run an online pull first"
            )
        from httpx import TransportError
        from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError
        from requests.exceptions import ConnectionError as RequestsConnectionError
        from requests.exceptions import Timeout as RequestsTimeout

        try:
            info = call_with_deadline(
                lambda: HfApi().model_info(
                    repo_id, revision=commit, files_metadata=True, token=token
                ),
                30,
            )
        except (
            HfHubHTTPError,
            TimeoutError,
            TransportError,
            RequestsConnectionError,
            RequestsTimeout,
        ) as exc:
            if (
                isinstance(exc, HfHubHTTPError)
                and not isinstance(exc, RevisionNotFoundError)
                and getattr(exc.response, "status_code", None)
                not in (429, 500, 502, 503, 504)
            ):
                raise
            if revision != commit and resolved_revision != commit:
                raise
            # Preserve explicit-pin recovery when upstream has lost a commit
            # or its transport is unavailable. Reuse the strict complete-inventory
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
    try:
        siblings = list(siblings)
    except TypeError as exc:
        raise ModelIntegrityError("Invalid integrity manifest file inventory") from exc
    names = [_field(item, "rfilename") for item in siblings]
    if not all(isinstance(name, str) for name in names):
        raise ModelIntegrityError("Invalid integrity manifest filename")
    siblings = list({_field(item, "rfilename"): item for item in siblings}.values())
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
    if local_selected - set(verified):
        raise ModelIntegrityError(
            "Downloaded snapshot inventory mismatch: selected local files are absent from the immutable manifest"
        )
    if not verified:
        if not allow_all and not patterns and exact is None:
            return
        raise ModelIntegrityError("Integrity manifest contains no selected files")
    ensure_unchanged(verified)
    import tempfile

    temporary = None
    try:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=receipt_path.parent, prefix=".receipt-"
        )
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
    except OSError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Content verified, but cannot save receipt at %s: %s. "
            "Check cache permissions and free space; future pulls may need online verification.",
            receipt_path,
            exc,
        )
    finally:
        if temporary is not None:
            with suppress(OSError):
                Path(temporary).unlink(missing_ok=True)
    ensure_unchanged(verified)
