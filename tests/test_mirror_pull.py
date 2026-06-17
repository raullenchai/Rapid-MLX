"""Tests for ``vllm_mlx._mirror.download_with_mirror_fallback``.

Covers the seven scenarios called out in the PR #649 spec:

1. Catalog parsing — given a fake catalog, build correct file URLs and
   identify mirrored vs not-mirrored entries.
2. Per-file fallback — for each file, R2 returns 200 OR 404
   (parametrized). Every file lands once, R2 hits use R2 bytes, R2
   misses use HF bytes.
3. Whole-mirror miss — catalog says ``not yet mirrored`` → zero R2
   requests, all requests via HF.
4. Catalog fetch failure — catalog endpoint returns 500 → pull still
   completes via HF.
5. ``RAPID_MLX_MODEL_MIRROR=""`` — env disable → zero R2 requests even
   when alias is fully mirrored.
6. Size mismatch — R2 returns bytes whose size disagrees with HF's
   advertised size → R2 file is deleted and HF is used.
7. Resume — a partial ``.part`` exists on disk → R2 path makes a
   ``Range`` request for the remaining bytes.

All tests mock HTTP layer — no real network. Catalog and per-file URLs
are routed through a single in-test ``urlopen`` stub keyed by URL.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import _mirror

# ---------------------------------------------------------------------------
# Test fixtures — fake catalog + fake HF model_info + URL-routed HTTP stub.
# ---------------------------------------------------------------------------


def _catalog_payload(
    aliases: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Build a catalog JSON shape matching the verified contract.

    ``aliases`` is a list of ``(alias, hf_path, status)`` triples.
    Default fixture: two aliases, one mirrored + one not.
    """
    if aliases is None:
        aliases = [
            ("qwen3-0.6b-4bit", "mlx-community/Qwen3-0.6B-4bit", "mirrored"),
            ("gemma-4-31b-4bit", "mlx-community/Gemma-4-31B-4bit", "not yet mirrored"),
        ]
    models = []
    for alias, hf_path, status in aliases:
        owner, _, repo = hf_path.partition("/")
        models.append(
            {
                "alias": alias,
                "hf_path": hf_path,
                "status": status,
                "download_url_base": f"/{owner}/{repo}/",
                "file_count": 3,
                "size_gb_est": 0.5,
                "is_moe": False,
                "is_hybrid": False,
                "install_command": f"rapid-mlx pull {alias}",
            }
        )
    return {
        "total": len(models),
        "mirrored_count": sum(1 for m in models if m["status"] == "mirrored"),
        "generated_at": "2026-06-17T18:35:10.937Z",
        "models": models,
    }


def _mk_sibling(rfilename: str, size: int):
    """Build a minimal HF sibling object — mimics ``RepoSibling``."""
    s = MagicMock()
    s.rfilename = rfilename
    s.size = size
    return s


def _mk_model_info(sha: str, files: list[tuple[str, int]]):
    info = MagicMock()
    info.sha = sha
    info.siblings = [_mk_sibling(name, size) for name, size in files]
    return info


class _FakeResponse:
    """Minimal stand-in for the ``http.client.HTTPResponse`` urlopen returns.

    Supports the context-manager protocol + ``read([n])`` + ``status`` +
    ``headers`` — enough for the production code in ``_mirror.py``.
    """

    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None):
        self.status = status
        self._buf = io.BytesIO(body)
        self.headers = headers or {}
        if "Content-Length" not in self.headers and status in (200, 206):
            self.headers["Content-Length"] = str(len(body))

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n if n != -1 else None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _UrlRouter:
    """Routes urlopen calls by URL prefix to canned responses.

    Tracks every request so tests can assert call shapes. Each route is
    a callable that takes the ``Request`` and returns a ``_FakeResponse``
    (or raises an exception to simulate network / HTTP error).
    """

    def __init__(self):
        self.routes: dict[str, Any] = {}
        self.requests: list[dict[str, Any]] = []

    def add(self, url: str, response: Any) -> None:
        self.routes[url] = response

    def __call__(self, req, timeout: float | None = None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        headers = dict(req.headers) if hasattr(req, "headers") else {}
        self.requests.append({"url": url, "headers": headers, "timeout": timeout})
        handler = self.routes.get(url)
        if handler is None:
            # Unmocked URL — fail loudly so tests catch missing routes.
            raise AssertionError(f"unmocked URL in test: {url}")
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            return handler(req)
        return handler


# ---------------------------------------------------------------------------
# 1. Catalog parsing.
# ---------------------------------------------------------------------------


def test_catalog_parsing_builds_correct_file_urls():
    catalog = _catalog_payload()
    entry = _mirror.find_catalog_entry(catalog, "mlx-community/Qwen3-0.6B-4bit")
    assert entry is not None
    assert entry["status"] == "mirrored"
    url = _mirror._build_r2_url(
        "https://models.rapidmlx.com",
        entry["download_url_base"],
        "config.json",
    )
    assert (
        url == "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json"
    )


def test_catalog_parsing_identifies_not_mirrored():
    catalog = _catalog_payload()
    entry = _mirror.find_catalog_entry(catalog, "mlx-community/Gemma-4-31B-4bit")
    assert entry is not None
    assert not _mirror._is_mirrored(entry)


def test_catalog_parsing_returns_none_for_unknown_hf_path():
    catalog = _catalog_payload()
    entry = _mirror.find_catalog_entry(catalog, "mlx-community/Unknown-Model")
    assert entry is None


def test_catalog_url_encodes_special_chars():
    url = _mirror._build_r2_url(
        "https://models.rapidmlx.com",
        "/foo/bar/",
        "file with spaces.txt",
    )
    assert url == "https://models.rapidmlx.com/foo/bar/file%20with%20spaces.txt"


# ---------------------------------------------------------------------------
# 2. Per-file fallback — fully mirrored.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "r2_status_per_file,expected_r2_hit_names,expected_hf_hit_names",
    [
        # All files mirrored — every file served from R2.
        (
            [200, 200, 200],
            ["config.json", "model.safetensors", "tokenizer.json"],
            [],
        ),
        # Mixed — config from R2, weights + tokenizer miss → HF.
        (
            [200, 404, 404],
            ["config.json"],
            ["model.safetensors", "tokenizer.json"],
        ),
        # Total R2 miss — every file falls back to HF.
        (
            [404, 404, 404],
            [],
            ["config.json", "model.safetensors", "tokenizer.json"],
        ),
    ],
)
def test_per_file_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    r2_status_per_file: list[int],
    expected_r2_hit_names: list[str],
    expected_hf_hit_names: list[str],
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "deadbeef" * 5
    files = [("config.json", 100), ("model.safetensors", 200), ("tokenizer.json", 50)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    for (fname, size), status in zip(files, r2_status_per_file, strict=True):
        url = f"https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/{fname}"
        if status == 200:
            router.add(url, _FakeResponse(200, b"x" * size))
        else:
            router.add(url, _FakeResponse(status, b""))

    # HF fallback writes a placeholder file at the snapshot path. Track
    # the calls so we can assert which files HF was asked for.
    hf_calls: list[str] = []

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        hf_calls.append(filename)
        snap = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap.mkdir(parents=True, exist_ok=True)
        target = snap / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        # Use the size HF told us about — picked from the test fixture.
        expected_size = next(s for n, s in files if n == filename)
        target.write_bytes(b"h" * expected_size)
        return str(target)

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download", side_effect=_fake_hf),
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok, "download should succeed when every file is reachable from R2 or HF"
    # Snapshot directory should contain all three files, exactly once.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    on_disk = sorted(p.name for p in snap.iterdir() if p.is_file())
    assert on_disk == sorted(f for f, _ in files)
    # refs/main pins the snapshot — required for is_repo_cached.
    refs_main = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "refs" / "main"
    assert refs_main.read_text() == revision
    # Codex round-1 NIT #4: assert the EXACT filenames that fell back
    # to HF, not just the count. A wrong-file mix would otherwise pass.
    assert sorted(hf_calls) == sorted(expected_hf_hit_names)
    # And the R2 file requests match the expected R2 hits (ignore the
    # catalog request).
    r2_file_requests = [
        r["url"].rsplit("/", 1)[-1]
        for r in router.requests
        if "/mlx-community/Qwen3-0.6B-4bit/" in r["url"]
    ]
    # Every expected R2 hit must have been requested; misses also issue
    # a request (which returns 404), so the set of requested files is
    # the union of expected hits and HF-fallbacks.
    assert sorted(set(r2_file_requests)) == sorted(
        set(expected_r2_hit_names + expected_hf_hit_names)
    )


# ---------------------------------------------------------------------------
# 3. Whole-mirror miss — catalog reports "not yet mirrored".
# ---------------------------------------------------------------------------


def test_not_yet_mirrored_skips_r2_entirely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Gemma-4-31B-4bit"
    revision = "f00ba1" * 6
    files = [("config.json", 100), ("model.safetensors", 200)]
    catalog = _catalog_payload([("gemma-4-31b-4bit", repo_id, "not yet mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # Deliberately do NOT register any R2 file URLs — if production code
    # tries to hit one, the router raises AssertionError.

    hf_calls: list[str] = []

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        hf_calls.append(filename)
        snap = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap.mkdir(parents=True, exist_ok=True)
        target = snap / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_size = next(s for n, s in files if n == filename)
        target.write_bytes(b"h" * expected_size)
        return str(target)

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download", side_effect=_fake_hf),
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    # Catalog hit, but ZERO per-file R2 calls.
    r2_file_calls = [
        r for r in router.requests if "/mlx-community/Gemma-4-31B-4bit/" in r["url"]
    ]
    assert r2_file_calls == []
    # HF served everything.
    assert sorted(hf_calls) == sorted(f for f, _ in files)


# ---------------------------------------------------------------------------
# 4. Catalog fetch failure (5xx) — pull still completes via HF.
# ---------------------------------------------------------------------------


def test_catalog_500_falls_through_to_hf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abcd" * 10
    files = [("config.json", 50)]

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(500, b"oops"),
    )

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        snap = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap.mkdir(parents=True, exist_ok=True)
        (snap / filename).write_bytes(b"h" * 50)
        return str(snap / filename)

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download", side_effect=_fake_hf) as hf_mock,
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    # HF served the only file even though catalog 500'd.
    assert hf_mock.call_count == 1
    # No per-file R2 requests (catalog miss disables R2 for this run).
    r2_file_calls = [
        r
        for r in router.requests
        if r["url"] != "https://models.rapidmlx.com/api/models"
    ]
    assert r2_file_calls == []


# ---------------------------------------------------------------------------
# 5. RAPID_MLX_MODEL_MIRROR="" — env disable — zero R2 requests.
# ---------------------------------------------------------------------------


def test_env_disable_skips_r2_entirely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "ffff" * 10
    files = [("config.json", 100)]

    # Empty env value means "force HF" — production code returns False
    # from download_with_mirror_fallback before touching the network.
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "")

    router = _UrlRouter()
    # No routes registered — any HTTP call would AssertionError.
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download") as hf_mock,
    ):
        result = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    # When the mirror is disabled, the function bails early so the caller
    # falls through to snapshot_download. No HF or R2 calls were made.
    assert result is False
    assert router.requests == []
    assert hf_mock.call_count == 0


# ---------------------------------------------------------------------------
# 6. Size mismatch — R2 returns bytes whose size disagrees with HF.
# ---------------------------------------------------------------------------


def test_size_mismatch_deletes_r2_file_and_uses_hf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "1234" * 10
    # HF advertises 100 bytes; R2 will return 90 bytes — mirror has a
    # stale build.
    files = [("config.json", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 90),
    )

    hf_calls: list[str] = []

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        hf_calls.append(filename)
        snap = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap.mkdir(parents=True, exist_ok=True)
        # HF writes the CORRECT 100 bytes.
        (snap / filename).write_bytes(b"h" * 100)
        return str(snap / filename)

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download", side_effect=_fake_hf),
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    # HF was called for the file because R2 size disagreed with HF.
    assert hf_calls == ["config.json"]
    # No stray .part file left behind on disk.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    leftover = list(snap.glob("*.part"))
    assert leftover == [], f"unexpected .part files: {leftover}"
    # The file on disk is HF's bytes, not R2's truncated bytes.
    assert (snap / "config.json").read_bytes() == b"h" * 100


# ---------------------------------------------------------------------------
# 7. Resume — partial .part exists → R2 issues a Range request.
# ---------------------------------------------------------------------------


def test_resume_sends_range_header_for_partial_part_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "cafe" * 10
    # Single 200-byte file. We'll pre-create a 50-byte .part on disk.
    files = [("model.safetensors", 200)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Pre-stage the partial file at the exact snapshot path the
    # production code will compute.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    part = snap / "model.safetensors.part"
    part.write_bytes(b"a" * 50)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # R2 honors the Range request with 206 + remaining 150 bytes.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(206, b"b" * 150),
    )

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download") as hf_mock,
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    # Range header was sent on the file request.
    file_requests = [r for r in router.requests if "/model.safetensors" in r["url"]]
    assert len(file_requests) == 1
    headers_lower = {k.lower(): v for k, v in file_requests[0]["headers"].items()}
    assert "range" in headers_lower, (
        f"missing Range header: {file_requests[0]['headers']}"
    )
    assert headers_lower["range"] == "bytes=50-"
    # Final file is the concatenation of the 50 pre-staged + 150 resumed
    # bytes — 200 total.
    final = snap / "model.safetensors"
    assert final.exists()
    assert final.stat().st_size == 200
    assert final.read_bytes() == b"a" * 50 + b"b" * 150
    # HF was not called.
    assert hf_mock.call_count == 0
    # No .part leftover after the rename.
    assert not part.exists()


# ---------------------------------------------------------------------------
# Codex round-1 BLOCKING #1 regression — a stale (truncated) cache entry
# must NOT be accepted as already-cached. The production code must
# re-fetch when the on-disk size disagrees with HF's advertised size.
# ---------------------------------------------------------------------------


def test_truncated_cached_file_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "0001" * 10
    # HF says 200 bytes; we'll pre-stage a 50-byte truncated file at
    # the snapshot path (simulating an aborted prior pull).
    files = [("model.safetensors", 200)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    truncated = snap / "model.safetensors"
    truncated.write_bytes(b"a" * 50)  # truncated relic

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # R2 will serve the full 200 bytes once the truncated file is dropped.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, b"x" * 200),
    )

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download") as hf_mock,
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    # The truncated file MUST have been replaced — final size is 200,
    # not 50.
    assert truncated.exists()
    assert truncated.stat().st_size == 200
    assert truncated.read_bytes() == b"x" * 200
    # HF was not needed because R2 had a fresh copy.
    assert hf_mock.call_count == 0


# ---------------------------------------------------------------------------
# Codex round-1 BLOCKING #2 regression — refs/main must NOT be clobbered
# when it already points at a different sha (the user pulled some other
# revision separately). The snapshot is still addressable by sha; the
# ref belongs to whoever wrote it.
# ---------------------------------------------------------------------------


def test_refs_main_not_clobbered_when_pointing_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "2222" * 10
    foreign_sha = "9999" * 10
    files = [("config.json", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Pre-stage a refs/main pointing at a different (foreign) sha.
    repo_root = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
    refs_dir = repo_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(foreign_sha)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 100),
    )

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download"),
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    # The pre-existing refs/main MUST be preserved — we don't own it.
    assert (refs_dir / "main").read_text() == foreign_sha
    # Our snapshot is still addressable by sha — that's what matters.
    assert (repo_root / "snapshots" / revision / "config.json").exists()


def test_refs_main_written_when_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Common case: refs/main absent → we write it (idempotent)."""
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "3333" * 10
    files = [("config.json", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 100),
    )

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download"),
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    refs_main = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "refs" / "main"
    assert refs_main.read_text() == revision
