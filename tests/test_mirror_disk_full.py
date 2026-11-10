"""Disk exhaustion must survive mirror fallback without touching the real cache."""

import argparse
import errno
import urllib.error

import pytest

from rapid_mlx import _mirror, cli
from tests.test_mirror_pinned_revision import (
    BASE,
    CONFIG,
    PIN,
    REPO,
    WEIGHTS,
    _install_packaged_digests,
    _Router,
    _snap,
)


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))


def failure(wrapper):
    disk = OSError(errno.ENOSPC, "private cache path")
    if wrapper == "direct":
        return disk
    if wrapper == "reason":
        return urllib.error.URLError(disk)
    outer = RuntimeError("wrapped") if wrapper == "runtime" else OSError("wrapped")
    setattr(outer, "__cause__" if wrapper == "runtime" else wrapper, disk)
    return outer


@pytest.mark.parametrize(
    "wrapper", ["direct", "__cause__", "__context__", "reason", "runtime"]
)
@pytest.mark.parametrize("source", ["mirror", "hub", "worker"])
def test_packaged_pull_preserves_disk_full(monkeypatch, tmp_path, source, wrapper):
    _install_packaged_digests(tmp_path, monkeypatch)
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    router = _Router({"config.json": CONFIG})
    monkeypatch.setattr("urllib.request.urlopen", router)
    error = failure(wrapper)

    def fail(*args, **kwargs):
        raise error

    if source == "mirror":
        original = router

        def urlopen(req, **kwargs):
            if "model.safetensors" in req.full_url:
                raise error
            return original(req, **kwargs)

        monkeypatch.setattr("urllib.request.urlopen", urlopen)
    elif source == "hub":
        monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    else:
        monkeypatch.setattr(_mirror, "_download_one_from_r2", fail)

    # A verified completed file and an existing resumable prefix survive.
    snapshot = _snap(tmp_path)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(CONFIG)
    sidecar = snapshot.parents[1] / ".rapid-mlx-mirror"
    sidecar.mkdir()
    partial = sidecar / "retained.part"
    partial.write_bytes(WEIGHTS[:8])
    with pytest.raises(OSError) as caught:
        _mirror.download_with_mirror_fallback(REPO, cache_dir=tmp_path, revision=PIN)
    assert caught.value.errno == errno.ENOSPC
    assert (snapshot / "config.json").read_bytes() == CONFIG
    assert partial.read_bytes() == WEIGHTS[:8]


@pytest.mark.parametrize("command", ["pull", "serve"])
@pytest.mark.parametrize("disk_full", [False, True])
def test_cli_mirror_failure_exits_with_guidance(
    monkeypatch, tmp_path, capsys, command, disk_full
):
    error = failure("direct") if disk_full else _mirror.MirrorOnlyPullError("private")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "_try_mirror_prefetch", fail)
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    with pytest.raises(SystemExit) as caught:
        if command == "serve":
            cli._ensure_model_downloaded("owner/model", force_disk_check=True)
        else:
            cli._pull_repository(
                argparse.Namespace(model="owner/model", bits=None, format=None),
                emit_lifecycle_event=False,
            )
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert cli.render_hub_error(error, "owner/model") in output.err
    assert "private" not in output.err
    assert "Traceback" not in output.err
    assert "server will retry" not in output.err


def test_error_chain_cycles_and_non_exceptions():
    outer = OSError("ordinary failure")
    inner = urllib.error.URLError("ordinary URL error")
    outer.__cause__ = inner
    inner.__context__ = outer
    _mirror._raise_if_disk_full(outer)


@pytest.mark.parametrize("stage", ["mkdir", "stage", "rename", "symlink"])
def test_disk_full_during_blob_install_keeps_bytes(monkeypatch, tmp_path, stage):
    from pathlib import Path

    repo = tmp_path / "repo"
    target = repo / "snapshots" / PIN / "model.safetensors"
    target.parent.mkdir(parents=True)
    partial = tmp_path / "download.part"
    partial.write_bytes(WEIGHTS)
    blob = repo / "blobs" / "digest"
    staged = blob.with_suffix(".tmp")
    error = OSError(errno.ENOSPC, "full")
    method = {
        "mkdir": "mkdir",
        "stage": "rename",
        "rename": "rename",
        "symlink": "symlink_to",
    }[stage]
    fail_path = {
        "mkdir": blob.parent,
        "stage": partial,
        "rename": staged,
        "symlink": target,
    }[stage]
    original = getattr(Path, method)

    def operation(path, *args, **kwargs):
        if path == fail_path:
            raise error
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, operation)
    with pytest.raises(OSError) as caught:
        _mirror._install_lfs_blob_and_symlink(partial, target, repo, "digest")
    assert caught.value is error
    assert (
        partial
        if stage in ("mkdir", "stage")
        else staged
        if stage == "rename"
        else blob
    ).read_bytes() == WEIGHTS


@pytest.mark.parametrize("stage", ["mkdir", "write", "stat", "oid", "rename"])
def test_disk_full_during_mirror_transfer_keeps_partial(monkeypatch, tmp_path, stage):
    from pathlib import Path

    from tests.test_mirror_pinned_revision import _Resp

    target = tmp_path / "snapshot" / "model.safetensors"
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    partial = sidecar / "file.part"
    error = OSError(errno.ENOSPC, "full")
    response = _Resp(WEIGHTS)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response)
    if stage == "mkdir":
        original = Path.mkdir

        def mkdir(path, *args, **kwargs):
            if path == target.parent:
                raise error
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", mkdir)
    elif stage == "write":
        # Deliver one chunk before failing, leaving a real resumable prefix.
        original = response.read
        reads = []

        def read(*args):
            reads.append(1)
            if len(reads) > 1:
                raise error
            return original(8)

        monkeypatch.setattr(response, "read", read)
    elif stage == "stat":
        original = Path.stat

        def stat(path, *args, **kwargs):
            if path == partial and path.open("rb").read() == WEIGHTS:
                raise error
            return original(path, *args, **kwargs)

        # Avoid opening the partial before the transfer has created it.
        def guarded_stat(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            return stat(path, *args, **kwargs) if result.st_size else result

        monkeypatch.setattr(Path, "stat", guarded_stat)
    elif stage == "oid":

        def oid(*args):
            raise error

        monkeypatch.setattr(_mirror, "_git_blob_oid", oid)
    else:
        original = Path.rename

        def rename(path, *args, **kwargs):
            if path == partial:
                raise error
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError) as caught:
        _mirror._download_one_from_r2(
            "https://mirror.invalid/model",
            target,
            len(WEIGHTS),
            expected_git_oid="digest" if stage == "oid" else None,
            sidecar_dir=sidecar,
            sidecar_key="file",
        )
    assert caught.value is error
    if stage != "mkdir":
        # Use open rather than stat: the stat fault remains installed.
        with partial.open("rb") as stream:
            assert stream.read() == (WEIGHTS[:8] if stage == "write" else WEIGHTS)
