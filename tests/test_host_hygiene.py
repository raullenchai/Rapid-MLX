from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import protected_models  # noqa: E402
import studio_hygiene  # noqa: E402


def test_protected_inventory_tracks_sidecar_pins_and_release_fleet() -> None:
    models = protected_models.load_manifest()
    by_repository = {str(model["repository"]): model for model in models}
    sidecars = json.loads(
        (ROOT / "apps/rapid-mac/scripts/sidecar-smoke-models.json").read_text()
    )["models"]
    for pin in sidecars.values():
        protected = by_repository[pin["repository"]]
        assert protected["revision"] == pin["revision"]
        assert "sidecar" in protected["sources"]

    aliases = json.loads((ROOT / "vllm_mlx/aliases.json").read_text())
    fleet = json.loads((ROOT / "scripts/release_fleet.json").read_text())
    for family in fleet["families"].values():
        repository = aliases[family["coherence_model"]]["hf_path"]
        assert "release_fleet" in by_repository[repository]["sources"]


def test_protected_inventory_has_the_telemetry_roster() -> None:
    expected = {
        "mlx-community/Qwen3.5-4B-MLX-4bit",
        "mlx-community/Qwen3.5-9B-4bit",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "mlx-community/Qwen3.6-35B-A3B-8bit",
        "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
        "mlx-community/Qwen3.6-27B-4bit",
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
        "rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX",
    }
    actual = {
        str(model["repository"])
        for model in protected_models.load_manifest()
        if "telemetry_top10" in model["sources"]
    }
    assert actual == expected


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def test_worktree_is_eligible_only_when_clean_old_and_pushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked").write_text("one\n")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    tree = tmp_path / "tree"
    _git(repo, "worktree", "add", "-b", "safe", str(tree), "main")
    _git(tree, "push", "-u", "origin", "safe")
    old = time.time() - 10_000
    os.utime(tree, (old, old))
    monkeypatch.setattr(studio_hygiene, "in_use", lambda _: False)
    row = next(row for row in studio_hygiene.worktrees(repo) if row["path"] == tree)
    assert (
        studio_hygiene.classify_worktree(repo, row, time.time() - 3600, set())
        == "eligible"
    )
    assert (
        studio_hygiene.classify_worktree(repo, row, time.time() - 3600, {"safe"})
        == "open-pr"
    )
    (tree / "tracked").write_text("dirty\n")
    assert (
        studio_hygiene.classify_worktree(repo, row, time.time() - 3600, set())
        == "dirty"
    )


def test_apply_uses_git_to_remove_only_an_eligible_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked").write_text("one\n")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    tree = tmp_path / "eligible"
    _git(repo, "worktree", "add", "-b", "eligible", str(tree), "main")
    _git(tree, "push", "-u", "origin", "eligible")
    old = time.time() - 10_000
    os.utime(tree, (old, old))
    monkeypatch.setattr(studio_hygiene, "in_use", lambda _: False)
    monkeypatch.setattr(studio_hygiene, "mounted_images", lambda: [])
    monkeypatch.setattr(studio_hygiene, "open_pr_heads", lambda _: set())
    assert (
        studio_hygiene.main(
            [
                "--repo",
                str(repo),
                "--scratch",
                str(tmp_path),
                "--min-age-hours",
                "1",
                "--apply",
            ]
        )
        == 0
    )
    assert not tree.exists()
    assert repo.exists()


def test_only_explicitly_finished_dogfood_is_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished = tmp_path / "rapid-finished"
    active = tmp_path / "rapid-active"
    unowned = tmp_path / "rapid-unowned"
    finished.mkdir()
    active.mkdir()
    unowned.mkdir()
    (finished / ".dogfood-owned").touch()
    (finished / ".dogfood-finished").touch()
    (active / ".dogfood-owned").touch()
    (unowned / ".dogfood-finished").touch()
    old = time.time() - 10_000
    os.utime(finished, (old, old))
    os.utime(active, (old, old))
    os.utime(unowned, (old, old))
    monkeypatch.setattr(studio_hygiene, "in_use", lambda _: False)
    assert studio_hygiene.finished_dogfood_dirs(tmp_path, time.time() - 3600) == [
        finished
    ]


def test_only_old_unused_rapid_temp_dmg_mount_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "rapid-mlx-desktop.udrw.dmg"
    image.write_bytes(b"stub")
    mount = Path(tempfile.mkdtemp(prefix="rapid-test-dmg-mount.", dir="/tmp"))
    old = time.time() - 10_000
    os.utime(image, (old, old))
    os.utime(mount, (old, old))
    monkeypatch.setattr(studio_hygiene, "in_use", lambda _: False)
    try:
        assert studio_hygiene.stale_rapid_dmg(image, mount, time.time() - 3600)
        other = tmp_path / "personal-backup.dmg"
        other.write_bytes(b"stub")
        os.utime(other, (old, old))
        assert not studio_hygiene.stale_rapid_dmg(other, mount, time.time() - 3600)
        assert not studio_hygiene.stale_rapid_dmg(
            image, Path("/Volumes/Rapid-MLX"), time.time() - 3600
        )
    finally:
        mount.rmdir()


def test_apply_rechecks_owner_and_mount_before_detaching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked").write_text("one\n")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    tree = tmp_path / "eligible"
    _git(repo, "worktree", "add", "-b", "eligible", str(tree), "main")
    _git(tree, "push", "-u", "origin", "eligible")
    image = tree / "rapid-mlx-desktop.dmg"
    image.write_bytes(b"stub")
    _git(tree, "add", image.name)
    _git(tree, "commit", "-m", "artifact")
    _git(tree, "push")
    mount = Path(tempfile.mkdtemp(prefix="rapid-test-owner-mount.", dir="/tmp"))
    old = time.time() - 10_000
    os.utime(tree, (old, old))
    os.utime(image, (old, old))
    os.utime(mount, (old, old))
    monkeypatch.setattr(studio_hygiene, "open_pr_heads", lambda _: set())
    monkeypatch.setattr(studio_hygiene, "mounted_images", lambda: [(image, mount)])
    monkeypatch.setattr(studio_hygiene, "in_use", lambda path: path == mount)
    try:
        with pytest.raises(RuntimeError, match="owner or mount changed"):
            studio_hygiene.main(
                [
                    "--repo",
                    str(repo),
                    "--scratch",
                    str(tmp_path),
                    "--min-age-hours",
                    "1",
                    "--apply",
                ]
            )
        assert tree.exists()
        assert mount.exists()
    finally:
        mount.rmdir()


def test_hygiene_default_is_dry_run_and_never_removes_models() -> None:
    source = (ROOT / "scripts/studio_hygiene.py").read_text()
    assert 'parser.add_argument("--apply", action="store_true")' in source
    assert "model caches are report-only and are never removed" in source
    assert "HF_HUB_CACHE" in source
    assert "shutil.rmtree(cache" not in source
