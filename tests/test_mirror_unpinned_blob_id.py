# SPDX-License-Identifier: Apache-2.0
"""Default-branch (unpinned) pulls prove non-LFS mirror files by git blob id.

The mirror holds one upload of each file. Upstreams edit configs, tokenizer
files and chat templates in place on ``main``, often without changing their
size, so a size check cannot tell the current file from the stale upload.
HF lists a git blob id for every file; a non-LFS mirror object is used only
when its recomputed blob id matches HF's at the resolved ``main`` commit.
Anything else comes from HF, per file. Weights keep the LFS SHA-256 check.

All HTTP is mocked; no test touches the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rapid_mlx import _mirror

REPO = "org/text-model"
MAIN = "89abcdef0123456789abcdef0123456789abcdef"
BASE = "https://models.rapidmlx.com"
WEIGHTS = b"W" * 64
CONFIG = b'{"a": 1}\n'
STALE_CONFIG = b'{"a": 2}\n'  # same size, older content
TEMPLATE = b"{{ messages }}\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _oid(data: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(data)}\0".encode() + data, usedforsecurity=False
    ).hexdigest()


def _sibling(name: str, data: bytes, *, lfs: bool, blob_id: str | None = "auto"):
    s = SimpleNamespace(rfilename=name, size=len(data))
    s.lfs = SimpleNamespace(sha256=_sha256(data)) if lfs else None
    if blob_id == "auto":
        # For an LFS file HF's blob id names the pointer, never the bytes.
        blob_id = _oid(b"pointer " + name.encode()) if lfs else _oid(data)
    s.blob_id = blob_id
    return s


class _Resp:
    def __init__(self, body: bytes):
        self.status = 200
        self._buf = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(None if n == -1 else n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Router:
    def __init__(self, files: dict[str, bytes | int]):
        self.files = files
        self.urls: list[str] = []

    def __call__(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        self.urls.append(url)
        if url == f"{BASE}/api/models":
            payload = {
                "models": [
                    {
                        "alias": "text",
                        "hf_path": REPO,
                        "status": "mirrored",
                        "download_url_base": f"/{REPO}/",
                    }
                ]
            }
            return _Resp(json.dumps(payload).encode())
        name = url.removeprefix(f"{BASE}/{REPO}/")
        body = self.files.get(name)
        if body is None or isinstance(body, int):
            raise urllib.error.HTTPError(url, body or 404, "miss", {}, None)
        return _Resp(body)

    def file_urls(self) -> list[str]:
        return [u for u in self.urls if not u.endswith("/api/models")]


def _repo_root(tmp_path: Path) -> Path:
    return tmp_path / "models--org--text-model"


def _snap(tmp_path: Path) -> Path:
    return _repo_root(tmp_path) / "snapshots" / MAIN


def _run(tmp_path, monkeypatch, siblings, mirror_files, hf_bytes=None):
    """Run a default-branch pull; return (ok, router, hf_calls, out)."""
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    router = _Router(mirror_files)
    hf_calls: list[tuple[str, str]] = []
    hf_bytes = hf_bytes or {}

    def fake_hf(repo_id, filename, revision, cache_dir=None, **_kw):
        hf_calls.append((filename, revision))
        # Stand-in for HF's own layout: the real bytes behind a blob link.
        body = hf_bytes.get(filename, b"H")
        blob = _repo_root(Path(cache_dir)) / "blobs" / _oid(body)
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():  # HF re-links an existing blob by name
            blob.write_bytes(body)
        link = _snap(Path(cache_dir)) / filename
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(blob, link.parent))
        return str(link)

    out: dict = {}
    info = SimpleNamespace(sha=MAIN, siblings=siblings)
    info_mock = MagicMock(return_value=info)
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch("huggingface_hub.model_info", info_mock),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_hf),
    ):
        ok = _mirror.download_with_mirror_fallback(REPO, cache_dir=tmp_path, out=out)
    # A default-branch pull reads metadata at moving main.
    assert "revision" not in info_mock.call_args.kwargs
    return ok, router, hf_calls, out


def _assert_blob_layout(tmp_path: Path, name: str, data: bytes) -> None:
    link = _snap(tmp_path) / name
    assert link.is_symlink()
    resolved = link.resolve()
    assert resolved.parent == (_repo_root(tmp_path) / "blobs").resolve()
    assert resolved.name == _oid(data)
    assert resolved.read_bytes() == data


def test_matching_blob_ids_serve_every_file_from_the_mirror(tmp_path, monkeypatch):
    siblings = [
        _sibling("model.safetensors", WEIGHTS, lfs=True),
        _sibling("config.json", CONFIG, lfs=False),
        _sibling("chat_template.jinja", TEMPLATE, lfs=False),
    ]
    ok, router, hf_calls, out = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {
            "model.safetensors": WEIGHTS,
            "config.json": CONFIG,
            "chat_template.jinja": TEMPLATE,
        },
    )
    assert ok is True
    assert hf_calls == []
    assert len(router.file_urls()) == 3
    assert out["source"] == "mirror"
    # Non-LFS files land in HF's own blob layout (blobs/<git blob id>).
    _assert_blob_layout(tmp_path, "config.json", CONFIG)
    _assert_blob_layout(tmp_path, "chat_template.jinja", TEMPLATE)
    # Weights keep the SHA-256 path: the blob is named by the LFS sha, not
    # the pointer's blob id.
    weights = (_snap(tmp_path) / "model.safetensors").resolve()
    assert weights.name == _sha256(WEIGHTS)
    assert (_repo_root(tmp_path) / "refs" / "main").read_text() == MAIN


def test_same_size_stale_config_comes_from_hf(tmp_path, monkeypatch):
    """The drift case: HF edited a config in place; the mirror kept the old one."""
    assert len(STALE_CONFIG) == len(CONFIG)
    siblings = [
        _sibling("model.safetensors", WEIGHTS, lfs=True),
        _sibling("config.json", CONFIG, lfs=False),
    ]
    ok, router, hf_calls, out = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"model.safetensors": WEIGHTS, "config.json": STALE_CONFIG},
        hf_bytes={"config.json": CONFIG},
    )
    assert ok is True
    # The mirror was asked (and refused), then HF served that one file.
    assert f"{BASE}/{REPO}/config.json" in router.file_urls()
    assert hf_calls == [("config.json", MAIN)]
    assert out["source"] == "hf"
    assert (_snap(tmp_path) / "config.json").read_bytes() == CONFIG
    # The rejected bytes left nothing behind under the snapshot or blobs.
    assert not (_repo_root(tmp_path) / "blobs" / _oid(STALE_CONFIG)).exists()
    # The weight still came from the mirror.
    assert (_snap(tmp_path) / "model.safetensors").read_bytes() == WEIGHTS


def test_config_without_a_blob_id_never_touches_the_mirror(tmp_path, monkeypatch):
    """No published digest means the bytes cannot be proven: HF, not size-only."""
    siblings = [
        _sibling("model.safetensors", WEIGHTS, lfs=True, blob_id=None),
        _sibling("config.json", CONFIG, lfs=False, blob_id=None),
    ]
    ok, router, hf_calls, _out = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"model.safetensors": WEIGHTS, "config.json": CONFIG},
        hf_bytes={"config.json": CONFIG},
    )
    assert ok is True
    assert router.file_urls() == [f"{BASE}/{REPO}/model.safetensors"]
    assert hf_calls == [("config.json", MAIN)]


@pytest.mark.parametrize("blob_id", ["short", 12345])
def test_malformed_blob_id_counts_as_missing(tmp_path, monkeypatch, blob_id):
    siblings = [_sibling("config.json", CONFIG, lfs=False, blob_id=blob_id)]
    ok, router, hf_calls, _out = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"config.json": CONFIG},
        hf_bytes={"config.json": CONFIG},
    )
    assert ok is True
    assert router.file_urls() == []
    assert hf_calls == [("config.json", MAIN)]


def test_mirror_404_falls_back_to_hf(tmp_path, monkeypatch):
    siblings = [
        _sibling("model.safetensors", WEIGHTS, lfs=True),
        _sibling("config.json", CONFIG, lfs=False),
    ]
    ok, _router, hf_calls, _out = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"model.safetensors": WEIGHTS, "config.json": 404},
        hf_bytes={"config.json": CONFIG},
    )
    assert ok is True
    assert hf_calls == [("config.json", MAIN)]
    assert (_snap(tmp_path) / "config.json").read_bytes() == CONFIG


def test_size_mismatch_still_rejects_before_hashing(tmp_path, monkeypatch):
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, _router, hf_calls, _out = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"config.json": CONFIG + b" "},
        hf_bytes={"config.json": CONFIG},
    )
    assert ok is True
    assert hf_calls == [("config.json", MAIN)]


# --------------------------------------------------------------------------
# Warm snapshots: an older client accepted mirror configs by size alone and
# kept them as plain files. They are re-proven by blob id on the next pull.
# --------------------------------------------------------------------------


def _plant(tmp_path: Path, name: str, data: bytes) -> Path:
    path = _snap(tmp_path) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_warm_stale_same_size_config_is_replaced(tmp_path, monkeypatch):
    _plant(tmp_path, "config.json", STALE_CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [f"{BASE}/{REPO}/config.json"]
    assert hf_calls == []
    assert out["network_fetch"] is True
    _assert_blob_layout(tmp_path, "config.json", CONFIG)


def test_warm_proven_plain_file_moves_into_the_blob_layout(tmp_path, monkeypatch):
    _plant(tmp_path, "config.json", CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": STALE_CONFIG}
    )
    assert ok is True
    assert router.file_urls() == []
    assert hf_calls == []
    assert out["network_fetch"] is False
    _assert_blob_layout(tmp_path, "config.json", CONFIG)


def test_warm_proven_blob_link_is_kept_without_refetch(tmp_path, monkeypatch):
    blob = _repo_root(tmp_path) / "blobs" / _oid(CONFIG)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(CONFIG)
    link = _snap(tmp_path) / "config.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == []
    assert hf_calls == []
    assert out["network_fetch"] is False
    assert link.is_symlink() and link.resolve() == blob.resolve()


def test_warm_blob_link_with_wrong_bytes_is_refetched(tmp_path, monkeypatch):
    blob = _repo_root(tmp_path) / "blobs" / _oid(CONFIG)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(STALE_CONFIG)  # corrupted in place, same size
    link = _snap(tmp_path) / "config.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, _hf_calls, _out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [f"{BASE}/{REPO}/config.json"]
    assert (_snap(tmp_path) / "config.json").read_bytes() == CONFIG


def test_warm_file_without_a_blob_id_keeps_historical_acceptance(tmp_path, monkeypatch):
    """Nothing can prove or disprove it; HF's snapshot_download accepts it too."""
    _plant(tmp_path, "config.json", CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False, blob_id=None)]
    ok, router, hf_calls, out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == []
    assert hf_calls == []
    assert out["network_fetch"] is False


def test_warm_file_unreadable_for_hashing_is_refetched(tmp_path, monkeypatch):
    _plant(tmp_path, "config.json", CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    real = _mirror._git_blob_oid
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("unreadable")
        return real(path)

    monkeypatch.setattr(_mirror, "_git_blob_oid", flaky)
    ok, router, hf_calls, _out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [f"{BASE}/{REPO}/config.json"]
    assert hf_calls == []
    _assert_blob_layout(tmp_path, "config.json", CONFIG)


def test_cold_pull_replaces_a_corrupt_blob_stored_under_the_expected_id(
    tmp_path, monkeypatch
):
    """A verified download is never discarded in favour of a bad blob."""
    blob = _repo_root(tmp_path) / "blobs" / _oid(CONFIG)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(STALE_CONFIG)  # no snapshot entry links to it yet
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, _out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [f"{BASE}/{REPO}/config.json"]
    assert hf_calls == []
    _assert_blob_layout(tmp_path, "config.json", CONFIG)


def test_migrating_a_proven_file_replaces_a_corrupt_blob(tmp_path, monkeypatch):
    blob = _repo_root(tmp_path) / "blobs" / _oid(CONFIG)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(STALE_CONFIG)
    _plant(tmp_path, "config.json", CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, _out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": STALE_CONFIG}
    )
    assert ok is True
    assert router.file_urls() == []
    assert hf_calls == []
    _assert_blob_layout(tmp_path, "config.json", CONFIG)


def test_matching_existing_blob_is_reused(tmp_path, monkeypatch):
    blob = _repo_root(tmp_path) / "blobs" / _oid(CONFIG)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(CONFIG)
    before = blob.stat().st_ino
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, _router, _hf_calls, _out = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert blob.stat().st_ino == before
    _assert_blob_layout(tmp_path, "config.json", CONFIG)


def test_blob_oid_predicate_treats_unreadable_as_mismatch(tmp_path):
    assert _mirror._blob_oid_is(_oid(CONFIG))(tmp_path / "missing") is False


def test_hf_fallback_never_relinks_a_corrupt_blob(tmp_path, monkeypatch):
    """Mirror miss + a same-size corrupt blobs/<blob id>: HF must refetch."""
    blob = _repo_root(tmp_path) / "blobs" / _oid(CONFIG)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(STALE_CONFIG)
    link = _snap(tmp_path) / "config.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, _router, hf_calls, _out = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"config.json": 404},
        hf_bytes={"config.json": CONFIG},
    )
    assert ok is True
    assert hf_calls == [("config.json", MAIN)]
    _assert_blob_layout(tmp_path, "config.json", CONFIG)


def test_drop_bad_blob_keeps_good_and_absent_blobs(tmp_path):
    root = tmp_path / "repo"
    good = root / "blobs" / _oid(CONFIG)
    good.parent.mkdir(parents=True)
    good.write_bytes(CONFIG)
    _mirror._drop_bad_blob(root, _oid(CONFIG))
    assert good.read_bytes() == CONFIG
    _mirror._drop_bad_blob(root, _oid(TEMPLATE))  # absent: no-op
    assert not (root / "blobs" / _oid(TEMPLATE)).exists()


def test_drop_bad_blob_ignores_an_unstatable_path(tmp_path, monkeypatch):
    def boom(_self):
        raise OSError("stat failed")

    monkeypatch.setattr(Path, "is_file", boom)
    _mirror._drop_bad_blob(tmp_path, _oid(CONFIG))  # must not raise


def test_drop_bad_blob_removes_only_the_link_to_a_corrupt_shared_file(tmp_path):
    shared = tmp_path / "shared-store" / "ab" / ("c" * 64)
    shared.parent.mkdir(parents=True)
    shared.write_bytes(STALE_CONFIG)
    root = tmp_path / "repo"
    link = root / "blobs" / _oid(CONFIG)
    link.parent.mkdir(parents=True)
    link.symlink_to(shared)
    _mirror._drop_bad_blob(root, _oid(CONFIG))
    assert not link.is_symlink() and not link.exists()
    assert shared.read_bytes() == STALE_CONFIG  # the shared file is untouched
