# SPDX-License-Identifier: Apache-2.0
"""Cache gates accept a relocated or Hub-wide shared HF blob store (#4096).

Newer Hugging Face caches deduplicate LFS blobs across repositories::

    snapshots/<rev>/[sub/]file -> <repo>/blobs/<sha256> -> <hub>/blobs/<xx>/<digest>

and a user may relocate a repository's ``blobs`` directory to another
volume. Both are this repository's own files. What must stay refused is a
snapshot rebound to ANOTHER repository's files — either through its blob
directory or through a crafted leaf / blob link.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from rapid_mlx import _download_gate as gate

SHA = "a" * 40


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _share_leaf(cache_root: Path, repo_root: Path, leaf: Path, *, relative=True):
    """Rewrite one snapshot leaf into the two-hop shared-store layout."""
    payload = leaf.read_bytes()
    label = f"{repo_root.name}/{leaf}"
    digest = _hex("shared:" + label)
    shared = cache_root / "blobs" / digest[:2] / digest
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_bytes(payload)
    owned_dir = repo_root / "blobs"
    owned_dir.mkdir(parents=True, exist_ok=True)
    owned = owned_dir / _hex("etag:" + label)
    owned_real_dir = Path(os.path.realpath(owned_dir))
    owned.symlink_to(os.path.relpath(shared, owned_real_dir) if relative else shared)
    leaf.unlink()
    leaf.symlink_to(os.path.relpath(owned, leaf.parent))


def _share_all(cache_root: Path, repo_root: Path) -> None:
    for leaf in sorted((repo_root / "snapshots").rglob("*")):
        if leaf.is_file():
            _share_leaf(cache_root, repo_root, leaf)


def _seed_subfolder_repo(cache_root: Path, repo_id: str, subfolder="4bit") -> Path:
    """An indexed checkpoint in a subfolder — the ``lfm2.5-2.6b-4bit`` shape."""
    repo_root = cache_root / f"models--{repo_id.replace('/', '--')}"
    checkpoint = repo_root / "snapshots" / SHA / subfolder
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"w" * 64)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model.safetensors"}})
    )
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text(SHA)
    return checkpoint


def _move_to_blobs(repo_root: Path, checkpoint: Path, blob_dir: Path) -> None:
    """Normal Hub layout: every leaf symlinks into ``blob_dir``."""
    blob_dir.mkdir(parents=True, exist_ok=True)
    for leaf in sorted(checkpoint.iterdir()):
        blob = blob_dir / _hex(str(leaf))
        blob.write_bytes(leaf.read_bytes())
        leaf.unlink()
        leaf.symlink_to(os.path.relpath(repo_root / "blobs" / blob.name, leaf.parent))


@pytest.fixture
def hub(tmp_path, monkeypatch):
    cache_root = tmp_path / "hub"
    cache_root.mkdir()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    return cache_root


# --- text / subfolder checkpoints (_is_nonempty_snapshot_manifest_file) -----


def test_subfolder_checkpoint_in_normal_layout_is_complete(hub):
    checkpoint = _seed_subfolder_repo(hub, "LiquidAI/LFM2.5-2.6B-MLX")
    repo_root = checkpoint.parents[2]
    _move_to_blobs(repo_root, checkpoint, repo_root / "blobs")

    assert gate._snapshot_is_complete(str(checkpoint)) is True


def test_subfolder_checkpoint_in_shared_blob_store_is_complete(hub):
    """The #4096 repro: every byte on disk, deduplicated through hub/blobs."""
    checkpoint = _seed_subfolder_repo(hub, "LiquidAI/LFM2.5-2.6B-MLX")
    _share_all(hub, checkpoint.parents[2])

    assert gate._snapshot_is_complete(str(checkpoint)) is True


def test_root_indexed_checkpoint_in_shared_blob_store_is_cached(hub):
    checkpoint = _seed_subfolder_repo(hub, "org/flat", subfolder="")
    _share_all(hub, checkpoint.parents[1])

    assert gate.is_repo_cached("org/flat") is True


def test_shared_layout_under_symlinked_repo_root_with_relative_links(hub, tmp_path):
    """Relative blob links resolve against the repo's real location."""
    cold = tmp_path / "cold"
    checkpoint = _seed_subfolder_repo(cold, "org/moved")
    real_repo = checkpoint.parents[2]
    _share_all(cold, real_repo)
    (hub / real_repo.name).symlink_to(real_repo, target_is_directory=True)

    linked = hub / real_repo.name / "snapshots" / SHA / "4bit"
    assert gate._snapshot_is_complete(str(linked)) is True


def test_relocated_blob_directory_is_accepted(hub, tmp_path):
    """``<repo>/blobs`` moved to another volume is still this repo's store."""
    checkpoint = _seed_subfolder_repo(hub, "org/relocated")
    repo_root = checkpoint.parents[2]
    store = tmp_path / "external-volume" / "relocated-blobs"
    _move_to_blobs(repo_root, checkpoint, store)
    (repo_root / "blobs").symlink_to(store, target_is_directory=True)

    assert gate._snapshot_is_complete(str(checkpoint)) is True


def test_relocated_blob_directory_vouches_only_for_digest_named_blobs(hub, tmp_path):
    """A relocated store must not turn any file beneath it into a blob."""
    checkpoint = _seed_subfolder_repo(hub, "org/relocated")
    repo_root = checkpoint.parents[2]
    store = tmp_path / "external-volume" / "anything"
    store.mkdir(parents=True)
    (repo_root / "blobs").symlink_to(store, target_is_directory=True)
    for leaf in sorted(checkpoint.iterdir()):
        plain = store / "nested" / leaf.name
        plain.parent.mkdir(exist_ok=True)
        plain.write_bytes(leaf.read_bytes())
        leaf.unlink()
        leaf.symlink_to(plain)

    assert gate._snapshot_is_complete(str(checkpoint)) is False


def test_blob_directory_rebound_to_another_repo_is_rejected(hub):
    checkpoint = _seed_subfolder_repo(hub, "org/victim")
    repo_root = checkpoint.parents[2]
    foreign = hub / "models--other--repo" / "blobs"
    _move_to_blobs(repo_root, checkpoint, foreign)
    (repo_root / "blobs").symlink_to(foreign, target_is_directory=True)

    assert gate._snapshot_is_complete(str(checkpoint)) is False


def _rebind_weight(hub: Path, checkpoint: Path, how: str) -> None:
    repo_root = checkpoint.parents[2]
    _share_all(hub, repo_root)
    weight = checkpoint / "model.safetensors"
    payload = weight.read_bytes()
    weight.unlink()
    digest = _hex("attack")
    shared = hub / "blobs" / digest[:2] / digest
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_bytes(payload)
    foreign_blobs = hub / "models--other--repo" / "blobs"
    foreign_blobs.mkdir(parents=True, exist_ok=True)
    if how == "leaf-into-foreign-blobs":
        target = foreign_blobs / _hex("foreign-file")
        target.write_bytes(payload)
        weight.symlink_to(target)
    elif how == "leaf-through-foreign-link":
        target = foreign_blobs / _hex("foreign-link")
        target.symlink_to(shared)
        weight.symlink_to(target)
    elif how == "owned-link-chains-through-foreign-link":
        foreign = foreign_blobs / _hex("foreign-link")
        foreign.symlink_to(shared)
        owned = repo_root / "blobs" / _hex("chained")
        owned.symlink_to(foreign)
        weight.symlink_to(owned)
    elif how == "second-hop-is-another-link":
        hop = hub / "blobs" / digest[:2] / _hex("attack-hop")
        hop.symlink_to(shared.name)
        owned = repo_root / "blobs" / _hex("hop")
        owned.symlink_to(hop)
        weight.symlink_to(owned)
    elif how == "leaf-directly-into-shared-store":
        weight.symlink_to(shared)
    elif how == "malformed-shared-layout":
        bad = hub / "blobs" / "zz" / digest
        bad.parent.mkdir(parents=True)
        bad.write_bytes(payload)
        owned = repo_root / "blobs" / _hex("malformed")
        owned.symlink_to(bad)
        weight.symlink_to(owned)
    elif how == "owned-blob-name-not-a-digest":
        owned = repo_root / "blobs" / "not-a-digest"
        owned.symlink_to(shared)
        weight.symlink_to(owned)
    elif how == "outside-any-store":
        outside = hub.parent / "outside.safetensors"
        outside.write_bytes(payload)
        owned = repo_root / "blobs" / _hex("outside")
        owned.symlink_to(outside)
        weight.symlink_to(owned)
    else:
        raise AssertionError(how)


@pytest.mark.parametrize(
    "how",
    [
        "leaf-into-foreign-blobs",
        "leaf-through-foreign-link",
        "owned-link-chains-through-foreign-link",
        "second-hop-is-another-link",
        "leaf-directly-into-shared-store",
        "malformed-shared-layout",
        "owned-blob-name-not-a-digest",
        "outside-any-store",
    ],
)
def test_rebinding_a_weight_to_foreign_files_is_rejected(hub, how):
    checkpoint = _seed_subfolder_repo(hub, "org/victim")
    _rebind_weight(hub, checkpoint, how)

    assert gate._snapshot_is_complete(str(checkpoint)) is False


def test_local_checkpoint_symlink_outside_is_rejected(tmp_path):
    local = tmp_path / "local-checkpoint"
    local.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"w")
    (local / "model.safetensors").symlink_to(outside)

    assert (
        gate._is_nonempty_snapshot_manifest_file(
            str(local / "model.safetensors"), str(local)
        )
        is False
    )


def test_shared_blob_probe_rejects_plain_owned_blob_and_racing_links(hub, monkeypatch):
    checkpoint = _seed_subfolder_repo(hub, "org/plain")
    repo_root = checkpoint.parents[2]
    _move_to_blobs(repo_root, checkpoint, repo_root / "blobs")
    weight = checkpoint / "model.safetensors"
    # A one-hop link into the repo's own blobs is not the shared layout (the
    # caller's own-store branch accepts it), and a plain file is not a link.
    assert gate._is_shared_cache_blob(str(weight), str(repo_root)) is False
    assert (
        gate._is_shared_cache_blob(str(repo_root / "refs" / "main"), str(repo_root))
        is False
    )

    _share_all(hub, repo_root)
    assert gate._is_shared_cache_blob(str(weight), str(repo_root)) is True
    monkeypatch.setattr(
        gate.os, "readlink", lambda _p: (_ for _ in ()).throw(OSError("raced"))
    )
    assert gate._is_shared_cache_blob(str(weight), str(repo_root)) is False


def test_shared_blob_probe_rejects_a_rebound_blob_directory(hub):
    checkpoint = _seed_subfolder_repo(hub, "org/victim")
    repo_root = checkpoint.parents[2]
    foreign = hub / "models--other--repo" / "blobs"
    foreign.mkdir(parents=True)
    (repo_root / "blobs").symlink_to(foreign, target_is_directory=True)
    weight = checkpoint / "model.safetensors"
    weight.unlink()
    weight.symlink_to(os.path.relpath(foreign / _hex("x"), weight.parent))

    assert gate._is_shared_cache_blob(str(weight), str(repo_root)) is False


# --- other cache gates with the same containment rule -----------------------


def _seed_audio_repo(hub: Path, repo_id: str, weight: str) -> Path:
    repo_root = hub / f"models--{repo_id.replace('/', '--')}"
    snap = repo_root / "snapshots" / SHA
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / weight).write_bytes(b"w" * 64)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text(SHA)
    return repo_root


def test_whisper_gate_accepts_shared_blob_store(hub):
    repo = "mlx-community/whisper-small-mlx"
    _share_all(hub, _seed_audio_repo(hub, repo, "weights.npz"))

    assert gate._snapshot_is_complete_whisper_model(repo) is True


def test_audio_gate_accepts_shared_blob_store(hub):
    repo = "mlx-community/Kokoro-82M-bf16"
    _share_all(hub, _seed_audio_repo(hub, repo, "kokoro-v1_0.safetensors"))

    assert gate._snapshot_is_complete_audio_model(repo, "kokoro") is True


def test_split_model_gate_accepts_shared_blob_store(hub):
    repo = "dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4"
    repo_root = hub / f"models--{repo.replace('/', '--')}"
    snap = repo_root / "snapshots" / SHA
    snap.mkdir(parents=True)
    components = ["transformer", "text_encoder", "vae"]
    (snap / "split_model.json").write_text(json.dumps({"components": components}))
    for component in components:
        (snap / f"{component}.safetensors").write_bytes(b"w" * 64)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text(SHA)
    _share_all(hub, repo_root)

    assert gate.split_model_local_snapshot(repo) == str(snap)


def test_wan_gate_accepts_shared_blob_store(hub):
    from rapid_mlx.video.wan import WAN_REVISIONS

    repo = next(r for r in WAN_REVISIONS if "5b" in r.casefold())
    repo_root = hub / f"models--{repo.replace('/', '--')}"
    snap = repo_root / "snapshots" / WAN_REVISIONS[repo]
    snap.mkdir(parents=True)
    for name in (
        "config.json",
        "t5_encoder.safetensors",
        "vae.safetensors",
        "model.safetensors",
    ):
        (snap / name).write_bytes(b"{}" if name.endswith(".json") else b"w" * 64)
    assert gate._snapshot_is_complete_wan_model(repo) is False  # not in blobs
    _share_all(hub, repo_root)

    assert gate._snapshot_is_complete_wan_model(repo) is True


def test_pinned_image_snapshot_accepts_shared_blob_store(hub):
    repo, revision = gate.SD35_REPO, gate.SD35_REVISION
    repo_root = hub / f"models--{repo.replace('/', '--')}"
    snap = repo_root / "snapshots" / revision
    for relative in gate.SD35_DATA_FILES:
        target = snap / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    _share_all(hub, repo_root)

    assert gate.pinned_image_snapshot(repo) == str(snap)


def test_unreferenced_commit_pinned_snapshot_accepts_shared_blob_store(hub):
    from rapid_mlx import model_metadata

    repo = "publisher/pinned"
    repo_root = hub / "models--publisher--pinned"
    snap = repo_root / "snapshots" / "immutable"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (snap / "model.safetensors").write_bytes(b"w" * 64)
    _share_all(hub, repo_root)

    assert model_metadata.resolve_unreferenced_cached_snapshot(repo) == snap

    # A weight borrowed from another repository's blobs is still refused.
    weight = snap / "model.safetensors"
    foreign = hub / "models--other--repo" / "blobs" / _hex("foreign")
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"w" * 64)
    weight.unlink()
    weight.symlink_to(foreign)
    assert model_metadata.resolve_unreferenced_cached_snapshot(repo) is None
