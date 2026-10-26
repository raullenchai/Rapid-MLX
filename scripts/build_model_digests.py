#!/usr/bin/env python3
"""Build the package-owned, exact-commit mirror integrity inventory.

Generation uses Hugging Face; --check validates catalog coverage entirely offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "rapid_mlx/model_digests.json"
INVENTORY_LOCK = ROOT / "scripts/model_digest_inventory.json"


def selections() -> dict[str, dict]:
    from rapid_mlx._download_gate import (
        IMAGE_MODEL_DATA_FILES,
        IMAGE_MODEL_REVISIONS,
        IMAGE_MODEL_RUNTIME_ASSETS,
    )

    # Load the pure-data artifact contract directly: importing its parent
    # package also imports MLX/numpy, which the offline CI drift gate lacks.
    mtp = runpy.run_path(
        str(ROOT / "rapid_mlx/models/deepseek_v41_native/artifacts.py")
    )

    main = json.loads((ROOT / "rapid_mlx/aliases.json").read_text())
    audio = json.loads((ROOT / "rapid_mlx/audio/aliases.json").read_text())
    excluded = {
        entry["hf_path"]
        for entry in json.loads((ROOT / "scripts/mirror_unmirrored.json").read_text())[
            "entries"
        ]
    }
    result = {}
    for prefix, aliases, repo_key in (
        ("", main, "hf_path"),
        ("audio/", audio, "hf_id"),
    ):
        for alias, value in aliases.items():
            if not isinstance(value, dict):
                continue
            repo, revision = value.get(repo_key), value.get("revision")
            if not repo or not revision or repo in excluded:
                continue
            if (
                repo in IMAGE_MODEL_DATA_FILES
                and IMAGE_MODEL_REVISIONS.get(repo) != revision
            ):
                raise ValueError(f"image runtime pin differs from alias pin: {repo}")
            patterns = (
                list(IMAGE_MODEL_DATA_FILES[repo])
                if repo in IMAGE_MODEL_DATA_FILES
                else [f"{value['subfolder']}/*"]
                if value.get("subfolder")
                else None
            )
            result[prefix + alias] = {
                "repo": repo,
                "revision": revision,
                "allow_patterns": patterns,
            }
    for assets in IMAGE_MODEL_RUNTIME_ASSETS.values():
        for repo, revision, patterns in assets:
            if repo not in excluded:
                result[f"asset/image/{repo}"] = {
                    "repo": repo,
                    "revision": revision,
                    "allow_patterns": list(patterns),
                }
    for family_assets in audio.get("_runtime_assets", {}).values():
        for asset in family_assets:
            repo = asset["repo_id"]
            if repo not in excluded and asset.get("revision"):
                result[f"asset/audio/{repo}"] = {
                    "repo": repo,
                    "revision": asset["revision"],
                    "allow_patterns": list(asset["allow_patterns"]),
                }
    if mtp["MTP_REPO"] not in excluded:
        result[f"asset/mtp/{mtp['MTP_REPO']}"] = {
            "repo": mtp["MTP_REPO"],
            "revision": mtp["MTP_REVISION"],
            "allow_patterns": list(mtp["MTP_ALLOW_PATTERNS"]),
        }
    return result


def inventory_hash(files: list[dict]) -> str:
    names = sorted(file["path"] for file in files)
    return hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()


def check(payload: dict, expected: dict, inventory: dict | None = None) -> None:
    from rapid_mlx._mirror import _validate_relative_filename

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported packaged digest schema")
    entries = payload.get("aliases")
    if not isinstance(entries, dict) or set(entries) != set(expected):
        missing = sorted(set(expected) - set(entries or {}))
        extra = sorted(set(entries or {}) - set(expected))
        raise ValueError(f"digest alias drift: missing={missing}, extra={extra}")
    if inventory is None:
        inventory = json.loads(INVENTORY_LOCK.read_text())
    if set(inventory) != set(expected):
        raise ValueError("digest inventory lock alias drift")
    for alias, selection in expected.items():
        entry = entries[alias]
        if any(entry.get(key) != value for key, value in selection.items()):
            raise ValueError(f"digest revision or selection drift: {alias}")
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"empty digest inventory: {alias}")
        names = set()
        for file in files:
            name = file.get("path") if isinstance(file, dict) else None
            digest = (
                file.get("sha256") or file.get("git_oid")
                if isinstance(file, dict)
                else None
            )
            if (
                not isinstance(name, str)
                or not _validate_relative_filename(name)
                or name in names
                or type(file.get("size")) is not int
                or file["size"] < 0
                or ("sha256" in file) == ("git_oid" in file)
                or not isinstance(digest, str)
                or len(digest) != (64 if "sha256" in file else 40)
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"invalid digest entry: {alias}/{name}")
            names.add(name)
            patterns = selection["allow_patterns"]
            if patterns and not any(fnmatchcase(name, pat) for pat in patterns):
                raise ValueError(f"unselected digest file: {alias}/{name}")
        if inventory_hash(files) != inventory[alias]:
            raise ValueError(f"incomplete digest inventory: {alias}")


def fetch(repo: str, revision: str) -> list[dict]:
    from huggingface_hub import model_info

    info = model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"revision mismatch: {repo}")
    files = []
    for sibling in info.siblings:
        # Git LFS tracking rules are not runtime model data and are often
        # omitted by the mirror. The package-owned pull skips this file too.
        if sibling.rfilename == ".gitattributes":
            continue
        size = sibling.size
        sha = getattr(sibling.lfs, "sha256", None) if sibling.lfs else None
        oid = sibling.blob_id
        if type(size) is not int or size < 0 or not (sha or oid):
            raise ValueError(f"missing size/digest: {repo}/{sibling.rfilename}")
        files.append(
            {
                "path": sibling.rfilename,
                "size": size,
                **({"sha256": sha} if sha else {"git_oid": oid}),
            }
        )
    return sorted(files, key=lambda value: value["path"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="offline catalog coverage check"
    )
    args = parser.parse_args()
    expected = selections()
    if args.check:
        check(json.loads(OUTPUT.read_text()), expected)
    else:
        keys = {(value["repo"], value["revision"]) for value in expected.values()}
        inventories = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch, *key): key for key in keys}
            for future in as_completed(futures):
                key = futures[future]
                inventories[key] = future.result()
                print(f"{len(inventories)}/{len(keys)} {key[0]}", flush=True)
        entries = {}
        for alias, selection in sorted(expected.items()):
            files = inventories[selection["repo"], selection["revision"]]
            patterns = selection["allow_patterns"]
            selected = [
                file
                for file in files
                if patterns is None
                or any(fnmatchcase(file["path"], pat) for pat in patterns)
            ]
            entries[alias] = {**selection, "files": selected}
        payload = {"schema_version": 1, "aliases": entries}
        inventory = {
            alias: inventory_hash(entry["files"]) for alias, entry in entries.items()
        }
        check(payload, expected, inventory)
        OUTPUT.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        )
        INVENTORY_LOCK.write_text(
            json.dumps(inventory, separators=(",", ":"), sort_keys=True) + "\n"
        )
    assets = sum(key.startswith("asset/") for key in expected)
    print(
        f"{len(expected) - assets} aliases, {assets} assets, "
        f"{OUTPUT.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
