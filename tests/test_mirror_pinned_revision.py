# SPDX-License-Identifier: Apache-2.0
"""Pinned-revision downloads through the mirror.

Image, video and speculative-decoding checkpoints are pinned to an exact
commit. The mirror stores one build of each file, which may come from a
different commit, so for a pinned pull a mirror object may be used only when
its bytes are proven identical to the pinned file: the size from HF's metadata
AT the pinned commit, plus the LFS SHA-256 (weights) or the git blob id (every
other file), both recomputed over the downloaded bytes. Anything else falls
back to HF at the pinned commit, per file.

All HTTP is mocked; no test touches the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rapid_mlx import _mirror

REPO = "org/pinned-image"
PIN = "0123456789abcdef0123456789abcdef01234567"
BASE = "https://models.rapidmlx.com"
WEIGHTS = b"W" * 64
CONFIG = b'{"a": 1}\n'


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _oid(data: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(data)}\0".encode() + data, usedforsecurity=False
    ).hexdigest()


def _sibling(name: str, data: bytes, *, lfs: bool, blob_id: str | None = "auto"):
    s = SimpleNamespace(rfilename=name, size=len(data))
    s.lfs = SimpleNamespace(sha256=_sha256(data)) if lfs else None
    s.blob_id = _oid(data) if blob_id == "auto" else blob_id
    return s


def _info(siblings, sha: str = PIN):
    return SimpleNamespace(sha=sha, siblings=siblings)


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
                        "alias": "pinned",
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


def _run(tmp_path, monkeypatch, siblings, mirror_files, *, info=None, **kwargs):
    """Run a pinned pull; return (ok, router, hf_calls, model_info mock)."""
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    router = _Router(mirror_files)
    hf_calls: list[tuple[str, str]] = []
    real = {s.rfilename: s for s in siblings}

    def fake_hf(repo_id, filename, revision, cache_dir=None, **_kw):
        hf_calls.append((filename, revision))
        path = Path(cache_dir) / "hf" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"H" * (real[filename].size or 1))
        return str(path)

    info_mock = MagicMock(return_value=info if info is not None else _info(siblings))
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch("huggingface_hub.model_info", info_mock),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_hf),
    ):
        ok = _mirror.download_with_mirror_fallback(
            REPO, cache_dir=tmp_path, revision=PIN, **kwargs
        )
    return ok, router, hf_calls, info_mock


def _snap(tmp_path: Path) -> Path:
    return tmp_path / "models--org--pinned-image" / "snapshots" / PIN


def test_matching_mirror_bytes_serve_the_pinned_commit(tmp_path, monkeypatch):
    siblings = [
        _sibling("model.safetensors", WEIGHTS, lfs=True),
        _sibling("config.json", CONFIG, lfs=False),
    ]
    ok, router, hf_calls, info_mock = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"model.safetensors": WEIGHTS, "config.json": CONFIG},
    )
    assert ok is True
    assert hf_calls == []
    assert len(router.file_urls()) == 2
    # Metadata is read AT the pin, never from moving main.
    assert info_mock.call_args.kwargs["revision"] == PIN
    assert (_snap(tmp_path) / "model.safetensors").read_bytes() == WEIGHTS
    assert (_snap(tmp_path) / "config.json").read_bytes() == CONFIG
    # A SHA-addressed snapshot writes no ref, like snapshot_download(revision=sha).
    assert not (tmp_path / "models--org--pinned-image" / "refs" / "main").exists()


@pytest.mark.parametrize(
    ("served", "reason"),
    [
        (b"W" * 63, "size mismatch"),
        (b"X" * 64, "sha256 mismatch"),
        (404, "mirror 404"),
    ],
)
def test_unprovable_weight_falls_back_to_hf_at_the_pin(
    tmp_path, monkeypatch, served, reason
):
    siblings = [
        _sibling("model.safetensors", WEIGHTS, lfs=True),
        _sibling("config.json", CONFIG, lfs=False),
    ]
    ok, _router, hf_calls, _ = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"model.safetensors": served, "config.json": CONFIG},
    )
    assert ok is True, reason
    assert hf_calls == [("model.safetensors", PIN)], reason


def test_same_size_config_from_another_commit_falls_back(tmp_path, monkeypatch):
    """Size alone cannot prove a non-LFS file; the git blob id must match."""
    other = b'{"a": 2}\n'
    assert len(other) == len(CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, _router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": other}
    )
    assert ok is True
    assert hf_calls == [("config.json", PIN)]
    # The rejected mirror bytes never reach the snapshot.
    assert not (_snap(tmp_path) / "config.json").exists()


def test_file_without_a_digest_never_touches_the_mirror(tmp_path, monkeypatch):
    siblings = [_sibling("config.json", CONFIG, lfs=False, blob_id=None)]
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == []
    assert hf_calls == [("config.json", PIN)]


def test_file_without_a_size_never_touches_the_mirror(tmp_path, monkeypatch):
    sibling = _sibling("model.safetensors", WEIGHTS, lfs=True)
    sibling.size = None
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    router = _Router({"model.safetensors": WEIGHTS})

    def fake_hf(repo_id, filename, revision, cache_dir=None, **_kw):
        path = Path(cache_dir) / "hf" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(WEIGHTS)
        return str(path)

    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch("huggingface_hub.model_info", return_value=_info([sibling])),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_hf) as hf,
    ):
        ok = _mirror.download_with_mirror_fallback(
            REPO, cache_dir=tmp_path, revision=PIN
        )
    assert ok is True
    assert router.file_urls() == []
    assert hf.call_args.kwargs["revision"] == PIN


@pytest.mark.parametrize("error", [OSError("offline"), TimeoutError("slow")])
def test_hf_metadata_unavailable_fails_safe_to_hf(tmp_path, monkeypatch, error):
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    router = _Router({})
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch("huggingface_hub.model_info", side_effect=error),
        patch("huggingface_hub.hf_hub_download") as hf,
    ):
        ok = _mirror.download_with_mirror_fallback(
            REPO, cache_dir=tmp_path, revision=PIN
        )
    assert ok is False  # caller runs snapshot_download(revision=PIN)
    assert router.urls == []
    assert hf.call_count == 0


def test_metadata_for_a_different_commit_is_refused(tmp_path, monkeypatch):
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, _ = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"config.json": CONFIG},
        info=_info(siblings, sha="f" * 40),
    )
    assert ok is False
    assert router.urls == []
    assert hf_calls == []


def test_branch_or_tag_revision_keeps_the_snapshot_download_path(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    with (
        patch("urllib.request.urlopen") as urlopen,
        patch("huggingface_hub.model_info") as info,
    ):
        ok = _mirror.download_with_mirror_fallback(
            REPO, cache_dir=tmp_path, revision="v1.0"
        )
    assert ok is False
    assert urlopen.call_count == 0
    assert info.call_count == 0


def test_allow_patterns_scope_a_pinned_pull(tmp_path, monkeypatch):
    siblings = [
        _sibling("config.json", CONFIG, lfs=False),
        _sibling("unused.onnx", WEIGHTS, lfs=True),
    ]
    ok, router, hf_calls, _ = _run(
        tmp_path,
        monkeypatch,
        siblings,
        {"config.json": CONFIG},
        allow_patterns=["config.json"],
    )
    assert ok is True
    assert router.file_urls() == [f"{BASE}/{REPO}/config.json"]
    assert hf_calls == []


def test_git_blob_oid_matches_git(tmp_path):
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    # ``git hash-object /dev/null``
    assert _mirror._git_blob_oid(empty) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    config = tmp_path / "config"
    config.write_bytes(CONFIG)
    assert _mirror._git_blob_oid(config) == _oid(CONFIG)


# --------------------------------------------------------------------------
# pinned_snapshot_download
# --------------------------------------------------------------------------


def test_pinned_snapshot_download_returns_the_mirror_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(_mirror, "_hf_cache_root", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(
        _mirror,
        "download_with_mirror_fallback",
        lambda repo, **kw: calls.append((repo, kw)) or True,
    )
    with patch("huggingface_hub.snapshot_download") as snap:
        path = _mirror.pinned_snapshot_download(REPO, PIN, allow_patterns=["a"])
    assert path == str(_snap(tmp_path))
    assert calls == [(REPO, {"revision": PIN, "allow_patterns": ["a"]})]
    assert snap.call_count == 0


def test_pinned_snapshot_download_falls_back_at_the_pin(monkeypatch):
    monkeypatch.setattr(
        _mirror, "download_with_mirror_fallback", lambda *_a, **_k: False
    )
    with patch("huggingface_hub.snapshot_download", return_value="/snap") as snap:
        assert _mirror.pinned_snapshot_download(REPO, PIN) == "/snap"
        assert _mirror.pinned_snapshot_download(REPO, PIN, allow_patterns=["x"])
    assert snap.call_args_list[0].kwargs == {"revision": PIN}
    assert snap.call_args_list[1].kwargs == {"revision": PIN, "allow_patterns": ["x"]}


def test_pinned_snapshot_download_never_mirrors_a_moving_revision(monkeypatch):
    monkeypatch.setattr(
        _mirror,
        "download_with_mirror_fallback",
        lambda *_a, **_k: pytest.fail("a branch name must not use the mirror"),
    )
    with patch("huggingface_hub.snapshot_download", return_value="/snap") as snap:
        assert _mirror.pinned_snapshot_download(REPO, "main") == "/snap"
    assert snap.call_args.kwargs == {"revision": "main"}


# --------------------------------------------------------------------------
# Callers: every pinned download goes mirror-first
# --------------------------------------------------------------------------


def _mirror_completes(monkeypatch, tmp_path):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(_mirror, "_hf_cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        _mirror,
        "download_with_mirror_fallback",
        lambda repo, **kw: seen.append((repo, kw)) or True,
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_k: pytest.fail("the mirror already completed the download"),
    )
    return seen


def _snapshot_of(tmp_path: Path, repo: str, revision: str) -> str:
    return str(tmp_path / f"models--{repo.replace('/', '--')}" / "snapshots" / revision)


def test_image_engine_cold_fetch_uses_the_mirror_at_the_pin(monkeypatch, tmp_path):
    from rapid_mlx import _download_gate
    from rapid_mlx.image.engine import ImageGenerationEngine

    repo = _download_gate.HIDREAM_O1_REPO
    revision = _download_gate.IMAGE_MODEL_REVISIONS[repo]
    monkeypatch.setattr(_download_gate, "mflux_local_snapshot", lambda _r: None)
    seen = _mirror_completes(monkeypatch, tmp_path)
    engine = ImageGenerationEngine(repo)
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)

    assert engine._model_path_for_mflux() == _snapshot_of(tmp_path, repo, revision)
    assert seen == [
        (
            repo,
            {
                "revision": revision,
                "allow_patterns": list(_download_gate.HIDREAM_O1_DATA_FILES),
            },
        )
    ]


def test_image_runtime_assets_use_the_mirror_at_their_pins(monkeypatch, tmp_path):
    from rapid_mlx import _download_gate
    from rapid_mlx.image.engine import ImageGenerationEngine

    monkeypatch.setattr(_download_gate, "pinned_image_snapshot", lambda _r: None)
    seen = _mirror_completes(monkeypatch, tmp_path)
    engine = ImageGenerationEngine(_download_gate.SD35_REPO)
    engine._ensure_runtime_assets()

    expected = [
        (repo, {"revision": revision, "allow_patterns": list(files)})
        for repo, revision, files in _download_gate.image_runtime_assets_for(
            _download_gate.SD35_REPO
        )
    ]
    assert expected and seen == expected


def test_wan_checkpoint_uses_the_mirror_at_the_pin(monkeypatch, tmp_path):
    from rapid_mlx.video import wan

    repo = "Anes1032/Wan2.2-I2V-A14B-mlx-q8"
    monkeypatch.delenv("RAPID_MLX_WAN_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    seen = _mirror_completes(monkeypatch, tmp_path)
    resolved = wan._resolve_model_path(repo)
    assert str(resolved) == _snapshot_of(tmp_path, repo, wan.WAN_REVISIONS[repo])
    assert seen == [
        (repo, {"revision": wan.WAN_REVISIONS[repo], "allow_patterns": None})
    ]


def test_ensure_downloaded_pins_main_ref_after_a_mirror_pinned_pull(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS

    repo = "mlx-community/Qwen-Image-2.1-mflux-q4"
    revision = IMAGE_MODEL_REVISIONS[repo]
    observed: dict = {}
    monkeypatch.setattr(cli.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "rapid_mlx._mirror.download_with_mirror_fallback",
        lambda model, **kw: (
            observed.setdefault("mirror", (model, kw.get("revision"))) and True
        ),
    )
    monkeypatch.setattr(
        "rapid_mlx._download_gate.pin_main_ref",
        lambda model, pinned: observed.setdefault("ref", (model, pinned)),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_k: pytest.fail("the mirror already completed the pull"),
    )
    cli._ensure_model_downloaded(repo)
    assert observed == {"mirror": (repo, revision), "ref": (repo, revision)}


def test_unreadable_downloaded_config_falls_back_to_hf(tmp_path, monkeypatch):
    def unreadable(_path):
        raise OSError("gone")

    monkeypatch.setattr(_mirror, "_git_blob_oid", unreadable)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, _router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert hf_calls == [("config.json", PIN)]


def test_default_branch_pull_still_fails_when_refs_main_is_unwritable(
    tmp_path, monkeypatch
):
    """The ref transaction moved under ``if not pinned``; keep its failure path."""
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    (tmp_path / "models--org--pinned-image" / "refs" / "main").mkdir(parents=True)
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    with (
        patch("urllib.request.urlopen", side_effect=_Router({"config.json": CONFIG})),
        patch("huggingface_hub.model_info", return_value=_info(siblings)),
        patch("huggingface_hub.hf_hub_download"),
    ):
        ok = _mirror.download_with_mirror_fallback(REPO, cache_dir=tmp_path)
    assert ok is False


@pytest.mark.parametrize(
    ("repo", "source"),
    [
        ("mlx-community/Qwen-Image-2.1-mflux-q4", "image"),
        ("Runpod/FLUX.2-klein-4B-mflux-4bit", "image"),
        ("Anes1032/Wan2.2-I2V-A14B-mlx-q8", "wan"),
    ],
)
def test_pull_fetches_the_commit_serve_uses(monkeypatch, repo, source):
    """``rapid-mlx pull`` (also the Mac app's downloader) pins image/video repos."""
    from rapid_mlx import cli
    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS
    from rapid_mlx.video.wan import WAN_REVISIONS

    calls = []
    monkeypatch.setattr(
        cli, "_pull_repository", lambda args, **kw: calls.append((args.model, kw))
    )
    cli.pull_command(SimpleNamespace(model=repo))
    pin = (IMAGE_MODEL_REVISIONS if source == "image" else WAN_REVISIONS)[repo]
    assert calls == [(repo, {"revision_override": pin})]


def test_pull_of_an_unpinned_repo_follows_main(monkeypatch):
    from rapid_mlx import cli

    calls = []
    monkeypatch.setattr(
        cli, "_pull_repository", lambda args, **kw: calls.append((args.model, kw))
    )
    cli.pull_command(SimpleNamespace(model="mlx-community/Qwen3-0.6B-4bit"))
    assert calls == [("mlx-community/Qwen3-0.6B-4bit", {})]


def test_stale_same_size_config_in_the_pinned_snapshot_is_refetched(
    tmp_path, monkeypatch
):
    """A warm non-LFS file must still prove its git blob id at the pin."""
    stale = b'{"a": 2}\n'
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(stale)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert hf_calls == []
    assert router.file_urls() == [f"{BASE}/{REPO}/config.json"]
    assert (snap / "config.json").read_bytes() == CONFIG


def test_matching_warm_config_is_kept_without_a_transfer(tmp_path, monkeypatch):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [] and hf_calls == []


def test_part_left_by_another_revision_cannot_poison_a_pinned_file(
    tmp_path, monkeypatch
):
    """A ``.part`` prefix from a different build fails the digest and falls back."""
    siblings = [_sibling("model.safetensors", WEIGHTS, lfs=True)]
    part = (
        tmp_path
        / "models--org--pinned-image"
        / ".rapid-mlx-mirror"
        / f"{_mirror._sidecar_key_for('model.safetensors')}.part"
    )
    part.parent.mkdir(parents=True)
    part.write_bytes(b"Z" * 10)

    class _Ranged(_Router):
        def __call__(self, req, timeout=None):
            resp = super().__call__(req, timeout)
            rng = (
                dict(req.header_items()).get("Range")
                if hasattr(req, "header_items")
                else None
            )
            if rng and not str(req.full_url).endswith("/api/models"):
                start = int(rng.split("=")[1].rstrip("-"))
                body = WEIGHTS[start:]
                resp = _Resp(body)
                resp.status = 206
                resp.headers["Content-Range"] = (
                    f"bytes {start}-{len(WEIGHTS) - 1}/{len(WEIGHTS)}"
                )
            return resp

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    hf_calls = []

    def fake_hf(repo_id, filename, revision, cache_dir=None, **_kw):
        hf_calls.append((filename, revision))
        path = Path(cache_dir) / "hf" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(WEIGHTS)
        return str(path)

    with (
        patch(
            "urllib.request.urlopen",
            side_effect=_Ranged({"model.safetensors": WEIGHTS}),
        ),
        patch("huggingface_hub.model_info", return_value=_info(siblings)),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_hf),
    ):
        ok = _mirror.download_with_mirror_fallback(
            REPO, cache_dir=tmp_path, revision=PIN
        )
    assert ok is True
    assert hf_calls == [("model.safetensors", PIN)]
    assert not part.exists()


def test_uppercase_pin_is_not_served_from_a_lowercase_snapshot(tmp_path, monkeypatch):
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    router = _Router({"config.json": CONFIG})
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch("huggingface_hub.model_info", return_value=_info(siblings)),
        patch("huggingface_hub.hf_hub_download") as hf,
    ):
        ok = _mirror.download_with_mirror_fallback(
            REPO, cache_dir=tmp_path, revision=PIN.upper()
        )
    assert ok is False  # snapshot_download resolves it at the exact pin
    assert router.urls == [] and hf.call_count == 0


def test_mirror_built_wan_snapshot_passes_the_wan_readiness_probe(
    tmp_path, monkeypatch
):
    """Desktop/`models --cached` readiness needs every file inside ``blobs/``."""
    from huggingface_hub import constants

    from rapid_mlx import _download_gate
    from rapid_mlx.video.wan import WAN_REVISIONS

    repo = "rickylin20260522/Wan2.2-TI2V-5B-mlx"
    pin = WAN_REVISIONS[repo]
    payload = {
        "config.json": b'{"model_type": "ti2v"}\n',
        "model.safetensors": b"M" * 32,
        "t5_encoder.safetensors": b"T" * 32,
        "vae.safetensors": b"V" * 32,
    }
    siblings = [
        _sibling(name, data, lfs=name.endswith(".safetensors"))
        for name, data in payload.items()
    ]
    owner, name = repo.split("/")

    class _WanRouter(_Router):
        def __call__(self, req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url == f"{BASE}/api/models":
                self.urls.append(url)
                return _Resp(
                    json.dumps(
                        {
                            "models": [
                                {
                                    "alias": "wan",
                                    "hf_path": repo,
                                    "status": "mirrored",
                                    "download_url_base": f"/{repo}/",
                                }
                            ]
                        }
                    ).encode()
                )
            self.urls.append(url)
            return _Resp(payload[url.removeprefix(f"{BASE}/{repo}/")])

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", BASE)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    router = _WanRouter({})
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch("huggingface_hub.model_info", return_value=_info(siblings, sha=pin)),
        patch("huggingface_hub.hf_hub_download") as hf,
    ):
        assert _mirror.download_with_mirror_fallback(
            repo, cache_dir=tmp_path, revision=pin
        )
    assert hf.call_count == 0
    config = tmp_path / f"models--{owner}--{name}" / "snapshots" / pin / "config.json"
    assert config.is_symlink()
    assert config.resolve().parent.name == "blobs"
    assert config.resolve().name == _oid(payload["config.json"])
    assert _download_gate._snapshot_is_complete_wan_model(repo) is True


def test_warm_pinned_file_of_unknown_size_still_proves_its_blob_id(
    tmp_path, monkeypatch
):
    stale = b"stale"
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(stale)
    sibling = _sibling("config.json", CONFIG, lfs=False)
    sibling.size = None
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, [sibling], {"config.json": CONFIG}
    )
    assert ok is True
    # Unknown size keeps the file off the mirror; HF at the pin replaces it.
    assert router.file_urls() == []
    assert hf_calls == [("config.json", PIN)]
    # The unproven warm file was dropped before the HF fetch.
    assert not (snap / "config.json").exists()


def test_warm_regular_pinned_file_moves_into_the_blob_layout(tmp_path, monkeypatch):
    """An earlier default-branch pull leaves configs as regular files; a pinned
    pull of the same commit relinks them HF-style so readiness gates agree."""
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [] and hf_calls == []
    config = snap / "config.json"
    assert config.is_symlink()
    assert (
        config.resolve()
        == (tmp_path / "models--org--pinned-image" / "blobs" / _oid(CONFIG)).resolve()
    )
    assert config.read_bytes() == CONFIG


def test_blob_install_failure_falls_back_to_hf(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _mirror, "_install_lfs_blob_and_symlink", lambda *_a: (False, "blob-mkdir:X")
    )
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, _router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert hf_calls == [("config.json", PIN)]


def test_second_pinned_pull_keeps_the_proven_blob_link(tmp_path, monkeypatch):
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    _run(tmp_path, monkeypatch, siblings, {"config.json": CONFIG})
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [] and hf_calls == []
    assert (_snap(tmp_path) / "config.json").is_symlink()


def test_warm_pinned_file_of_unknown_size_is_kept_when_proven(tmp_path, monkeypatch):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(CONFIG)
    sibling = _sibling("config.json", CONFIG, lfs=False)
    sibling.size = None
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, [sibling], {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [] and hf_calls == []


def test_unreadable_warm_pinned_file_is_refetched(tmp_path, monkeypatch):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(CONFIG)
    real = _mirror._git_blob_oid
    calls = []

    def flaky(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("unreadable")
        return real(path)

    monkeypatch.setattr(_mirror, "_git_blob_oid", flaky)
    siblings = [_sibling("config.json", CONFIG, lfs=False)]
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == [f"{BASE}/{REPO}/config.json"]
    assert hf_calls == []


def test_warm_pinned_file_without_a_digest_is_refetched(tmp_path, monkeypatch):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(CONFIG)
    siblings = [_sibling("config.json", CONFIG, lfs=False, blob_id=None)]
    ok, router, hf_calls, _ = _run(
        tmp_path, monkeypatch, siblings, {"config.json": CONFIG}
    )
    assert ok is True
    assert router.file_urls() == []
    assert hf_calls == [("config.json", PIN)]
