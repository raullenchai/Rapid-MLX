#!/usr/bin/env python3
"""Conservative, report-first hygiene for a shared macOS build host."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Set
from pathlib import Path

from protected_models import cache_name, load_manifest


def command(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(args, cwd=cwd, check=False, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed: " + " ".join(args))
    return result.stdout


def worktrees(repo: Path) -> list[dict[str, object]]:
    fields: dict[str, object] | None = None
    rows: list[dict[str, object]] = []
    raw = command("git", "worktree", "list", "--porcelain", "-z", cwd=repo)
    for token in raw.split("\0"):
        if not token:
            if fields:
                rows.append(fields)
                fields = None
            continue
        key, _, value = token.partition(" ")
        if key == "worktree":
            if fields:
                rows.append(fields)
            fields = {"path": Path(value)}
        elif fields is not None:
            fields[key] = value if value else True
    if fields:
        rows.append(fields)
    return rows


def in_use(path: Path) -> bool:
    lsof = shutil.which("lsof")
    if not lsof:
        return True
    try:
        result = subprocess.run(
            [lsof, "+D", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return True
    return result.returncode == 0


def open_pr_heads(repo: Path) -> set[str] | None:
    gh = shutil.which("gh")
    if not gh:
        return None
    result = subprocess.run(
        [
            gh,
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "1000",
            "--json",
            "headRefName",
        ],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        return None
    try:
        rows = json.loads(result.stdout)
        if len(rows) >= 1000:
            return None
        return {row["headRefName"] for row in rows}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def classify_worktree(
    repo: Path,
    row: dict[str, object],
    cutoff: float,
    open_heads: Set[str] | None = frozenset(),
) -> str:
    path = Path(row["path"])
    if path.resolve() == repo.resolve():
        return "primary"
    if "locked" in row:
        return "locked"
    if not path.exists():
        return "missing"
    if path.stat().st_mtime >= cutoff:
        return "recent"
    if command("git", "status", "--porcelain", cwd=path).strip():
        return "dirty"
    upstream = command(
        "git",
        "rev-parse",
        "--symbolic-full-name",
        "@{upstream}",
        cwd=path,
        check=False,
    ).strip()
    if not upstream:
        return "no-upstream"
    if not upstream.startswith("refs/remotes/"):
        return "non-remote-upstream"
    ahead = command("git", "rev-list", "--count", f"{upstream}..HEAD", cwd=path)
    if int(ahead) != 0:
        return "unpushed"
    branch = str(row.get("branch", "")).removeprefix("refs/heads/")
    if open_heads is None:
        return "pr-status-unknown"
    if branch in open_heads:
        return "open-pr"
    if in_use(path):
        return "in-use"
    return "eligible"


def finished_dogfood_dirs(scratch: Path, cutoff: float) -> list[Path]:
    if not scratch.exists():
        return []
    rows: list[Path] = []
    for marker in scratch.glob("rapid-*/.dogfood-finished"):
        candidate = marker.parent.resolve()
        try:
            candidate.relative_to(scratch.resolve())
        except ValueError:
            continue
        if (
            (candidate / ".dogfood-owned").is_file()
            and candidate.stat().st_mtime < cutoff
            and not in_use(candidate)
        ):
            rows.append(candidate)
    return rows


def mounted_images() -> list[tuple[Path, Path]]:
    if sys.platform != "darwin" or not shutil.which("hdiutil"):
        return []
    raw = command("hdiutil", "info", "-plist", check=False)
    if not raw:
        return []
    import plistlib

    result: list[tuple[Path, Path]] = []
    for image in plistlib.loads(raw.encode()).get("images", []):
        image_path = image.get("image-path")
        if not image_path:
            continue
        for entity in image.get("system-entities", []):
            mount = entity.get("mount-point")
            if mount:
                result.append((Path(image_path), Path(mount)))
    return result


def stale_rapid_dmg(image: Path, mount: Path, cutoff: float) -> bool:
    """Recognize only old tool-owned temporary mounts, including legacy ones."""
    if not image.is_file() or not mount.is_dir():
        return False
    mount_text = str(mount.resolve())
    tool_mount = mount_text.startswith(("/private/tmp/rapid-", "/tmp/rapid-")) or (
        mount_text.startswith("/private/var/folders/")
        and mount.name.startswith("rapid-")
    )
    if not tool_mount:
        return False
    if not (image.name.startswith("rapid-mlx-desktop") and image.name.endswith(".dmg")):
        return False
    if max(image.stat().st_mtime, mount.stat().st_mtime) >= cutoff:
        return False
    return not in_use(mount)


def owner_remains_removable(
    repo: Path,
    owner: Path,
    worktree_owners: set[Path],
    dogfood_owners: set[Path],
    scratch: Path,
    cutoff: float,
) -> bool:
    if owner in worktree_owners:
        heads = open_pr_heads(repo)
        if heads is None or not owner.resolve().is_relative_to(scratch):
            return False
        current = next(
            (row for row in worktrees(repo) if Path(row["path"]) == owner), None
        )
        return (
            current is not None
            and classify_worktree(repo, current, cutoff, heads) == "eligible"
        )
    if owner in dogfood_owners:
        return owner in set(finished_dogfood_dirs(scratch, cutoff))
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--scratch", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--min-age-hours", type=float, default=6)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.min_age_hours <= 0:
        parser.error("min-age-hours must be positive")
    repo = args.repo.resolve()
    scratch = args.scratch.resolve()
    if scratch == Path("/") or str(scratch).startswith("/Volumes/Extreme SSD"):
        parser.error("scratch must be a narrow local directory, never / or Extreme SSD")
    cutoff = time.time() - args.min_age_hours * 3600
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"MODE {mode}")

    hf_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
        )
    )
    for model in load_manifest():
        cache = hf_root / cache_name(str(model["repository"]))
        state = "present" if cache.exists() else "absent"
        print(f"PROTECTED {model['repository']} {state} {cache}")
    print("POLICY model caches are report-only and are never removed")

    open_heads = open_pr_heads(repo)
    if open_heads is None:
        print("POLICY GitHub PR state unavailable; no worktree is cleanup-eligible")
    eligible_worktrees: list[Path] = []
    for row in worktrees(repo):
        path = Path(row["path"])
        reason = classify_worktree(repo, row, cutoff, open_heads)
        if reason == "eligible" and not path.resolve().is_relative_to(scratch):
            reason = "outside-scratch"
        print(f"WORKTREE {reason} {path}")
        if reason == "eligible":
            eligible_worktrees.append(path)

    dogfoods = finished_dogfood_dirs(scratch, cutoff)
    dogfood_set = set(dogfoods)
    worktree_set = set(eligible_worktrees)
    removable_owners = [*eligible_worktrees, *dogfoods]
    for image, mount in mounted_images():
        owner = next(
            (
                path
                for path in removable_owners
                if image.resolve().is_relative_to(path.resolve())
            ),
            None,
        )
        legacy_stale = owner is None and stale_rapid_dmg(image, mount, cutoff)
        action = "eligible" if owner or legacy_stale else "skip-unowned"
        print(f"DMG {action} {mount} image={image}")
        if args.apply and (owner or legacy_stale):
            if owner is not None and (
                in_use(mount)
                or not owner_remains_removable(
                    repo, owner, worktree_set, dogfood_set, scratch, cutoff
                )
            ):
                raise RuntimeError(
                    f"DMG owner or mount changed after planning; refusing: {mount}"
                )
            if owner is None and not stale_rapid_dmg(image, mount, cutoff):
                raise RuntimeError(
                    f"DMG mount changed after planning; refusing: {mount}"
                )
            subprocess.run(["hdiutil", "detach", str(mount)], check=True)
    for path in dogfoods:
        print(f"DOGFOOD eligible {path}")

    if args.apply:
        for path in eligible_worktrees:
            current_open_heads = open_pr_heads(repo)
            if current_open_heads is None:
                raise RuntimeError(
                    "GitHub PR state unavailable at apply time; refusing"
                )
            current = next(
                (row for row in worktrees(repo) if Path(row["path"]) == path), None
            )
            if (
                current is None
                or classify_worktree(repo, current, cutoff, current_open_heads)
                != "eligible"
            ):
                raise RuntimeError(f"worktree changed after planning; refusing: {path}")
            subprocess.run(
                ["git", "worktree", "remove", str(path)], cwd=repo, check=True
            )
            print(f"REMOVED worktree {path}")
        for path in dogfood_set:
            marker = path / ".dogfood-finished"
            if (
                not path.resolve().is_relative_to(scratch)
                or not marker.is_file()
                or not (path / ".dogfood-owned").is_file()
                or in_use(path)
            ):
                raise RuntimeError(f"dogfood changed after planning; refusing: {path}")
            shutil.rmtree(path)
            print(f"REMOVED dogfood {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
