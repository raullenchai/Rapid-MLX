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
import os
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import _mirror


def _sidecar_part_path(cache_root: Path, owner_repo: str, fname: str) -> Path:
    """Compute the per-file sidecar ``.part`` path that production now uses.

    Codex round-14 BLOCKING #1+#2 moved ``.part``/``.lock`` out of the
    snapshot dir into ``repo_root/.rapid-mlx-mirror/<key>.{part,lock}``
    where ``<key>`` is :func:`_mirror._sidecar_key_for`(fname).
    """
    repo_root = cache_root / f"models--{owner_repo.replace('/', '--')}"
    return repo_root / ".rapid-mlx-mirror" / f"{_mirror._sidecar_key_for(fname)}.part"


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


def _mk_sibling(rfilename: str, size: int, lfs_sha256: str | None = None):
    """Build a minimal HF sibling object — mimics ``RepoSibling``.

    ``lfs_sha256`` is the SHA-256 of the file's bytes when HF tracks it
    via LFS (only LFS-tracked files expose this). When set, the mirror
    module uses it to validate downloaded bytes (codex round-5 BLOCKING
    #1).
    """
    s = MagicMock()
    s.rfilename = rfilename
    s.size = size
    if lfs_sha256 is not None:
        lfs = MagicMock()
        lfs.sha256 = lfs_sha256
        s.lfs = lfs
    else:
        s.lfs = None
    return s


def _mk_model_info(sha: str, files: list[tuple]):
    """Build a fake ``ModelInfo``.

    ``files`` is a list of ``(rfilename, size)`` or
    ``(rfilename, size, lfs_sha256)``.
    """
    info = MagicMock()
    info.sha = sha
    siblings = []
    for f in files:
        if len(f) == 2:
            name, size = f
            siblings.append(_mk_sibling(name, size))
        else:
            name, size, lfs_sha = f
            siblings.append(_mk_sibling(name, size, lfs_sha256=lfs_sha))
    info.siblings = siblings
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

    Codex round-9 NIT #4: real ``urllib.request.urlopen`` RAISES
    ``HTTPError`` for HTTP 4xx/5xx — it does not return a response with
    ``status == 404``. Translate ``_FakeResponse`` with a 4xx/5xx code
    into an ``HTTPError`` so the production exception path in
    ``fetch_catalog_with_status`` is exercised.
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
            handler = handler(req)
        # Codex round-9 NIT #4: surface 4xx/5xx as HTTPError to match
        # real urlopen semantics. Production catalog code catches both
        # the ``raise`` path and the ``return non-200`` path, so existing
        # tests stay green either way — but raising is the correct
        # mirror of what production callers see.
        if isinstance(handler, _FakeResponse) and handler.status >= 400:
            body = handler._buf.getvalue()
            raise urllib.error.HTTPError(
                url,
                handler.status,
                f"HTTP {handler.status}",
                handler.headers,
                io.BytesIO(body),
            )
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
    # Codex round-12 BLOCKING: the cross-process flock sidecar
    # (``.<file>.rapid-mlx-mirror.lock``) is intentionally retained on
    # disk after release — filter it out of the comparison.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    on_disk = sorted(
        p.name
        for p in snap.iterdir()
        if p.is_file() and not p.name.endswith(".rapid-mlx-mirror.lock")
    )
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

    # Codex round-8 BLOCKING #1+#2: when the catalog says the alias is
    # not mirrored, ``download_with_mirror_fallback`` must return False
    # so the caller invokes the real ``snapshot_download(repo_id)`` —
    # which preserves allow/ignore patterns, retries, and the existing
    # HF logging. Per-file ``hf_hub_download`` is NOT an equivalent.
    assert ok is False
    # Catalog hit, but ZERO per-file R2 calls.
    r2_file_calls = [
        r for r in router.requests if "/mlx-community/Gemma-4-31B-4bit/" in r["url"]
    ]
    assert r2_file_calls == []
    # No per-file HF calls either — caller is expected to use
    # ``snapshot_download`` instead.
    assert hf_calls == []


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
    # Default mirror with catalog outage routes straight to HF — we
    # don't probe the direct-layout because the default mirror is known
    # to serve /api/models, so an outage there means we should fail
    # fast to HF rather than incur 60s timeouts per file (codex round-4
    # BLOCKING #3). Therefore: no R2 file URL is registered.

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

    # Codex round-8: when the DEFAULT mirror's catalog 5xx's we treat
    # the mirror as wholly skipped and return False so the caller falls
    # through to ``snapshot_download``. No per-file HF or R2 work here.
    assert ok is False
    assert hf_mock.call_count == 0
    r2_file_calls = [
        r
        for r in router.requests
        if r["url"] != "https://models.rapidmlx.com/api/models"
    ]
    assert r2_file_calls == []


# ---------------------------------------------------------------------------
# PR #647 compat — custom mirror URLs without /api/models must still
# get the direct-layout fallback (codex round-4 BLOCKING #1+#2).
# ---------------------------------------------------------------------------


def test_custom_mirror_without_catalog_uses_direct_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """User sets RAPID_MLX_MODEL_MIRROR=<custom URL> that has no
    /api/models endpoint. We must try ``<base>/<owner>/<repo>/<file>``
    (the PR #647 contract) instead of silently routing everything via
    HF.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "5555" * 10
    files = [("config.json", 100), ("model.safetensors", 200)]

    router = _UrlRouter()
    # Custom mirror's /api/models doesn't exist — 404.
    router.add(
        "https://custom.example.com/api/models",
        _FakeResponse(404, b"not found"),
    )
    # But the direct file URLs DO work.
    router.add(
        "https://custom.example.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 100),
    )
    router.add(
        "https://custom.example.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, b"y" * 200),
    )

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://custom.example.com")
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
    # Both files came from the custom mirror, not HF.
    assert hf_mock.call_count == 0
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    assert (snap / "config.json").read_bytes() == b"x" * 100
    assert (snap / "model.safetensors").read_bytes() == b"y" * 200


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

    # Pre-stage the partial file at the sidecar temp path the
    # production code computes — round-14 BLOCKING #1+#2 moved the
    # ``.part`` out of ``snapshots/<sha>/`` into
    # ``repo_root/.rapid-mlx-mirror/<key>.part`` to avoid collisions
    # with legitimate repo assets named ``.<file>.rapid-mlx-mirror.part``.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    part = _sidecar_part_path(
        tmp_path, "mlx-community/Qwen3-0.6B-4bit", "model.safetensors"
    )
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"a" * 50)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # R2 honors the Range request with 206 + remaining 150 bytes.
    # Codex round-4 BLOCKING #2: the production code now validates
    # ``Content-Range`` matches the resume offset, so the mock must
    # include it.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(206, b"b" * 150, headers={"Content-Range": "bytes 50-199/200"}),
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
# Codex round-4 BLOCKING #2 regression — a 206 with an INCORRECT
# ``Content-Range`` header must be rejected. A proxy serving the wrong
# range would otherwise let us concatenate corrupt bytes into the
# .part and silently cache them as valid.
# ---------------------------------------------------------------------------


def test_resume_rejects_bad_content_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "ed1e" * 10
    files = [("model.safetensors", 200)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    part = _sidecar_part_path(
        tmp_path, "mlx-community/Qwen3-0.6B-4bit", "model.safetensors"
    )
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"a" * 50)  # resume from offset 50

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # Mirror returns 206 but Content-Range starts at byte 0, not 50.
    # This must be rejected — concatenating these bytes onto our 50-byte
    # prefix would corrupt the file.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(206, b"x" * 200, headers={"Content-Range": "bytes 0-199/200"}),
    )

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        snap_local = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap_local.mkdir(parents=True, exist_ok=True)
        target = snap_local / filename
        target.write_bytes(b"h" * 200)
        return str(target)

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
    # HF served the file because R2's bad Content-Range was rejected.
    assert hf_mock.call_count == 1
    # The truncated .part was wiped — no leftover.
    assert not part.exists()
    # Final file is HF's bytes (all h's), not R2's corrupted concat.
    assert (snap / "model.safetensors").read_bytes() == b"h" * 200


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
# Codex round-2 BLOCKING #1+#2 regression — refs/main MUST be updated
# to the downloaded sha, because ``pull_command`` reads ``refs/main`` to
# print the cache path and downstream consumers resolve ``main`` through
# it. We always pull HEAD of ``main`` (``model_info`` with no revision),
# so overwriting the ref is correct.
# ---------------------------------------------------------------------------


def test_refs_main_updated_when_pointing_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "2222" * 10
    stale_sha = "9999" * 10
    files = [("config.json", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Pre-stage a refs/main pointing at a stale sha (e.g. left over from
    # an earlier explicit ``snapshot_download(revision="<sha>")``).
    repo_root = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
    refs_dir = repo_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(stale_sha)

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
    # refs/main MUST be updated to the new sha so that downstream
    # consumers (including pull_command's "Cached at:" print and
    # is_repo_cached) resolve to the snapshot we just populated.
    assert (refs_dir / "main").read_text() == revision
    # Snapshot is on disk under the new sha.
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


# ---------------------------------------------------------------------------
# Codex round-2 BLOCKING #3 regression — when ``target.stat()`` raises
# OSError (e.g. EACCES on a permission-locked path, or unusual mount),
# the worker would otherwise return "miss" and force the whole pull to
# fall back to ``snapshot_download``. The fix wraps the cached-stat
# block in OSError protection and falls through to the R2/HF re-fetch.
#
# Easiest way to reliably trigger OSError on stat() in a unit test is
# to monkey-patch ``Path.stat`` to raise on the specific target path.
# ---------------------------------------------------------------------------


def test_unstatable_cached_path_falls_through_to_r2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "4444" * 10
    files = [("model.safetensors", 200)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Pre-stage a file at the snapshot path that we'll claim raises
    # on stat(). The ``exists()`` returns True (file is there), but the
    # ``stat()`` call inside the worker hits our raising stub.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    pre_existing = snap / "model.safetensors"
    pre_existing.write_bytes(b"a" * 50)  # truncated relic

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, b"x" * 200),
    )

    # Monkey-patch ``Path.stat`` to raise on the FIRST stat of our
    # specific target path (the cached-check stat). Subsequent stats
    # for the same path (after the R2 rename) succeed normally — that's
    # the realistic shape of a transient EACCES / ELOOP / etc.
    original_stat = Path.stat
    raised_once = {"done": False}
    target_path_suffix = f"snapshots/{revision}/model.safetensors"

    def _raising_stat(self, *args, **kwargs):
        if str(self).endswith(target_path_suffix) and not raised_once["done"]:
            raised_once["done"] = True
            raise OSError("simulated stat failure")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _raising_stat)
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

    # The pull should have completed via R2 despite the stat OSError on
    # the existing file. (We rely on the rename-from-.part to overwrite
    # the broken target.)
    assert ok
    # No HF fallback needed — R2 served the file fresh.
    assert hf_mock.call_count == 0


# ---------------------------------------------------------------------------
# Codex round-7 BLOCKING #1 — resume Content-Range must validate START,
# END, and TOTAL. ``bytes 50-99/200`` should NOT be accepted for a
# 200-byte file resumed from offset 50 — that gives us only 50 of the
# 150 remaining bytes.
# ---------------------------------------------------------------------------


def test_resume_rejects_short_content_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "5e5e" * 10
    files = [("model.safetensors", 200)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    part = _sidecar_part_path(
        tmp_path, "mlx-community/Qwen3-0.6B-4bit", "model.safetensors"
    )
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"a" * 50)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # Server says it's giving us bytes 50-99/200 (only 50 of the
    # remaining 150 bytes). This must be rejected because the resumed
    # download would terminate short.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(206, b"x" * 50, headers={"Content-Range": "bytes 50-99/200"}),
    )

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        snap_local = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap_local.mkdir(parents=True, exist_ok=True)
        target = snap_local / filename
        target.write_bytes(b"h" * 200)
        return str(target)

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
    # HF served the file because R2's short Content-Range was rejected.
    assert hf_mock.call_count == 1
    # The truncated .part was wiped — no leftover.
    assert not part.exists()
    # Final file is HF's bytes (all h's), not R2's truncated concat.
    assert (snap / "model.safetensors").read_bytes() == b"h" * 200


def test_resume_rejects_wrong_total_in_content_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If HF says the file is 200 bytes but the server's Content-Range
    declares a different total, the mirror has the wrong build —
    reject.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "ab7e" * 10
    files = [("model.safetensors", 200)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    part = _sidecar_part_path(
        tmp_path, "mlx-community/Qwen3-0.6B-4bit", "model.safetensors"
    )
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"a" * 50)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # Server's total disagrees with HF (300 vs 200).
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(206, b"x" * 150, headers={"Content-Range": "bytes 50-199/300"}),
    )

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        snap_local = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap_local.mkdir(parents=True, exist_ok=True)
        target = snap_local / filename
        target.write_bytes(b"h" * 200)
        return str(target)

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
    assert hf_mock.call_count == 1
    # Final file is HF's bytes.
    assert (snap / "model.safetensors").read_bytes() == b"h" * 200


# ---------------------------------------------------------------------------
# Codex round-7 BLOCKING #2 — a symlinked parent component under the
# snapshot directory must NOT let writes escape to other locations.
# ---------------------------------------------------------------------------


def test_symlinked_parent_under_snapshot_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "5ba5" * 10
    # File at ``subdir/escape.bin``. We'll plant ``subdir`` as a
    # symlink pointing outside the snapshot — production code MUST
    # refuse to write through it.
    files = [("subdir/escape.bin", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    escape_target = tmp_path / "escape"
    escape_target.mkdir()
    # Plant the malicious symlink.
    (snap / "subdir").symlink_to(escape_target)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # Deliberately do NOT register the file URL — if production code
    # tries to fetch (and write through the symlink), the AssertionError
    # in _UrlRouter will surface. The expected behaviour is "skip this
    # file as untrusted, fall back through download_with_mirror_fallback
    # returning False".

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

    # Mirror module returns False so caller falls back to snapshot_download.
    assert not ok
    # The symlink target was NOT written to. (The whole snapshot pull
    # was refused; we leave it to the safer snapshot_download to deal
    # with the questionable cache state.)
    assert not (escape_target / "escape.bin").exists()
    # HF wasn't called either — the worker bailed out before any
    # download attempt.
    assert hf_mock.call_count == 0


# ---------------------------------------------------------------------------
# Codex round-4 BLOCKING #1 regression — the .part temp file name must
# not collide with a real repo asset like ``model.safetensors.part``.
# The mirror module namespaces temp files as
# ``.<target.name>.rapid-mlx-mirror.part`` so a hypothetical sibling
# ``foo.part`` repo asset is safe.
# ---------------------------------------------------------------------------


def test_part_tempfile_does_not_collide_with_dot_part_repo_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "weirdco/repo-with-part-files"
    revision = "aabb" * 10
    # The repo contains BOTH ``foo.bin`` and ``foo.bin.part`` as
    # legitimate assets — pathological but valid. If our temp file
    # were ``foo.bin`` + ``.part``, the two workers would race over
    # the same temp path.
    files = [("foo.bin", 100), ("foo.bin.part", 50)]
    catalog = _catalog_payload([("weird-repo", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/weirdco/repo-with-part-files/foo.bin",
        _FakeResponse(200, b"X" * 100),
    )
    router.add(
        "https://models.rapidmlx.com/weirdco/repo-with-part-files/foo.bin.part",
        _FakeResponse(200, b"Y" * 50),
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
    snap = tmp_path / "models--weirdco--repo-with-part-files" / "snapshots" / revision
    # Both real files land with their distinct content — no clobber.
    assert (snap / "foo.bin").read_bytes() == b"X" * 100
    assert (snap / "foo.bin.part").read_bytes() == b"Y" * 50
    # No leftover hidden temp files.
    leftovers = list(snap.glob(".*rapid-mlx-mirror.part"))
    assert leftovers == [], f"unexpected leftover temp files: {leftovers}"
    assert hf_mock.call_count == 0


# ---------------------------------------------------------------------------
# Codex round-6 NIT #3 — custom mirror with 5xx on /api/models should
# NOT trigger direct-layout (would waste up to ~60s per file on a
# misconfigured mirror). 404 is the "no catalog endpoint here" signal
# and DOES trigger direct-layout (handled by
# test_custom_mirror_without_catalog_uses_direct_layout above).
# ---------------------------------------------------------------------------


def test_custom_mirror_with_5xx_catalog_skips_direct_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "5555" * 10
    files = [("config.json", 100)]

    router = _UrlRouter()
    # Custom mirror's /api/models returns 503 (transient or
    # misconfigured) — NOT 404. We must NOT probe the direct-layout R2
    # URL, just route to HF.
    router.add(
        "https://custom.example.com/api/models",
        _FakeResponse(503, b"backend down"),
    )

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        snap = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap.mkdir(parents=True, exist_ok=True)
        target = snap / filename
        target.write_bytes(b"h" * 100)
        return str(target)

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://custom.example.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download", side_effect=_fake_hf) as hf_mock,
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    # Codex round-8: custom mirror catalog 5xx (vs 404) means "mirror
    # transient/misconfigured" — skip the mirror wholesale and return
    # False so caller invokes ``snapshot_download``. No per-file work.
    assert ok is False
    assert hf_mock.call_count == 0
    r2_file_calls = [
        r
        for r in router.requests
        if r["url"] != "https://custom.example.com/api/models"
    ]
    assert r2_file_calls == [], (
        f"5xx catalog must not trigger direct-layout, got: {r2_file_calls}"
    )


# ---------------------------------------------------------------------------
# Codex round-5 BLOCKING #1 — LFS sha256 from HF metadata is verified on
# R2 downloads. A same-size-but-corrupt mirror object is rejected.
# ---------------------------------------------------------------------------


def test_r2_lfs_sha256_mismatch_falls_back_to_hf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import hashlib

    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "11ee" * 10
    payload_correct = b"x" * 200
    payload_corrupt = b"z" * 200  # same size, different bytes
    correct_sha = hashlib.sha256(payload_correct).hexdigest()
    # files: (name, size, lfs_sha256)
    files = [("model.safetensors", 200, correct_sha)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # R2 returns the SAME 200 bytes but a different payload (same
    # size). Without SHA check, this would silently cache as valid.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, payload_corrupt),
    )

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        snap = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap.mkdir(parents=True, exist_ok=True)
        target = snap / filename
        target.write_bytes(payload_correct)
        return str(target)

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
    # HF was called because R2's SHA didn't match.
    assert hf_mock.call_count == 1
    # The final file is HF's correct bytes, not R2's corrupt ones.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    assert (snap / "model.safetensors").read_bytes() == payload_correct
    # No stray hidden temp files left behind.
    leftovers = list(snap.glob(".*rapid-mlx-mirror.part"))
    assert leftovers == []


def test_r2_lfs_sha256_match_accepts_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import hashlib

    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "22ee" * 10
    payload = b"a" * 200
    sha = hashlib.sha256(payload).hexdigest()
    files = [("model.safetensors", 200, sha)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, payload),
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
    # SHA matched → R2 bytes were accepted, no HF fallback.
    assert hf_mock.call_count == 0
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    assert (snap / "model.safetensors").read_bytes() == payload


# ---------------------------------------------------------------------------
# Codex round-5 NIT #3 + #4 — zero-byte files. ``if expected_size`` is
# falsy for 0, so a legitimate empty file would skip integrity checks
# and be reprocessed on every pull. Fixed to ``is not None``.
# ---------------------------------------------------------------------------


def test_zero_byte_file_handled_correctly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "3333" * 10
    # ``.gitkeep``-style 0-byte file alongside a normal one.
    files = [("empty.txt", 0), ("config.json", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/empty.txt",
        _FakeResponse(200, b""),
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
        patch("huggingface_hub.hf_hub_download") as hf_mock,
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    # The 0-byte file lands as a 0-byte file.
    assert (snap / "empty.txt").exists()
    assert (snap / "empty.txt").stat().st_size == 0
    # The 100-byte file lands correctly.
    assert (snap / "config.json").read_bytes() == b"x" * 100
    # No HF fallback needed.
    assert hf_mock.call_count == 0

    # Second pull → the 0-byte file is recognized as cached and skipped
    # (NIT #4: ``cached_size == expected_size`` including 0).
    router2 = _UrlRouter()
    router2.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # Deliberately do NOT register any file URLs — the test fails if
    # production code re-fetches them.
    with (
        patch("urllib.request.urlopen", side_effect=router2),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download") as hf_mock2,
    ):
        ok2 = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok2
    # No HF or R2 file calls — everything was cache hits.
    assert hf_mock2.call_count == 0
    r2_file_calls = [
        r
        for r in router2.requests
        if r["url"] != "https://models.rapidmlx.com/api/models"
    ]
    assert r2_file_calls == [], (
        f"0-byte cached file should not be re-fetched, got: {r2_file_calls}"
    )


# ---------------------------------------------------------------------------
# Issue #?? — silent empty-response misclassification.
#
# An R2 worker that returns ``200 OK`` with ``Content-Length: 0`` (instead
# of the correct 404) for a file HF didn't expose a size for would
# otherwise be accepted as a legitimate empty file: the puller writes
# an empty file at the snapshot path, the summary logger prints
# ``[N/M] file R2 (0 MB)`` (looks like success), and downstream the file
# looks "cached" forever — the next pull sees ``cached_size == 0`` and
# skips it, propagating the silent failure. Force the puller to fall
# through to HF for the file so the real bytes land.
#
# This is the empty-mirror-response counterpart to the 404 path. The
# difference: the worker did NOT raise 404, so ``urlopen`` returns a
# normal 200 response with empty body. Without a size from HF we have
# no way to assert the bytes are correct, so the safe move is to refuse
# the empty response and let HF re-fetch.
#
# When HF DID expose a size (the common case), the
# ``final-size-mismatch`` check at the bottom of ``_do_r2_download``
# catches the 0-byte response before this guard fires. This test
# exercises the size-unknown case explicitly.
# ---------------------------------------------------------------------------


def test_r2_empty_response_without_expected_size_falls_back_to_hf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """R2 returns 200 + Content-Length: 0 for a file whose size HF didn't
    tell us → puller must fall back to HF, NOT accept the empty file as
    a successful R2 hit.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "ffff" * 10
    # ``model.safetensors.index.json`` — picked deliberately to match
    # the user-reported regression filename. HF's ``model_info`` doesn't
    # expose a size for it (passed as None here).
    files = [("model.safetensors.index.json", None), ("config.json", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # R2 returns 200 OK with empty body for the index.json — this is
    # the silent-failure path the bug filed against: a worker bug
    # returns 200 instead of the correct 404, and without a size from
    # HF the puller was previously accepting the empty file.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors.index.json",
        _FakeResponse(200, b""),
    )
    # config.json works normally.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 100),
    )

    # Track HF fallback calls so we can assert the index.json was
    # re-fetched from HF (not silently accepted as R2 success).
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
        # Mirror HF returning real bytes for the index.json — small
        # plausible payload.
        if filename == "model.safetensors.index.json":
            target.write_bytes(b'{"metadata":{}}')
        else:
            target.write_bytes(b"h" * 100)
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

    assert ok, "pull should succeed via HF fallback when R2 returns empty body"
    # The index.json MUST have come from HF — that's the bug fix. If
    # production code accepted the empty R2 body as a legit file, this
    # assertion fails (hf_calls would be empty for the index.json).
    assert "model.safetensors.index.json" in hf_calls, (
        "model.safetensors.index.json must fall back to HF when R2 returns "
        "empty body without an expected_size from HF — silent acceptance is "
        "the user-reported '[6/12] R2 (0 MB)' bug."
    )
    # And the real bytes landed on disk.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    assert (snap / "model.safetensors.index.json").read_bytes() == b'{"metadata":{}}'


def test_r2_empty_response_with_expected_size_still_falls_back_via_size_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Belt-and-braces: when HF DID tell us a size for the file, the
    existing ``final-size-mismatch`` check (line 492 in ``_do_r2_download``)
    already catches the 0-byte response. Pin that path too so a future
    refactor doesn't accidentally rely on the empty-response guard
    alone — both guards must coexist.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "eeee" * 10
    # Same shape as the bug, but HF exposes a size (100 bytes).
    files = [("model.safetensors.index.json", 100), ("config.json", 50)]

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(
            200,
            json.dumps(
                _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])
            ).encode(),
        ),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors.index.json",
        _FakeResponse(200, b""),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 50),
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
        target = snap / filename
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
    # The size-mismatch path forces HF fallback for the index.json.
    assert "model.safetensors.index.json" in hf_calls


# ---------------------------------------------------------------------------
# Codex round-9 NIT #4 — when the catalog endpoint raises ``HTTPError``
# directly (which is what real ``urlopen`` does for HTTP 4xx/5xx), the
# production code's ``except urllib.error.HTTPError`` branch must still
# route correctly. Lock in the exception path explicitly so a refactor
# of ``fetch_catalog_with_status`` doesn't silently regress.
# ---------------------------------------------------------------------------


def test_custom_mirror_catalog_httperror_404_uses_direct_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Real ``urlopen`` raises ``HTTPError`` for HTTP 404 — not returns a
    response with ``.status == 404``. Make sure the exception path in
    ``fetch_catalog_with_status`` reaches the same direct-layout
    fallback that the response-style test
    (``test_custom_mirror_without_catalog_uses_direct_layout``)
    exercises.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abcd" * 10
    files = [("config.json", 100)]

    router = _UrlRouter()
    # Raise HTTPError directly — this is how real urlopen signals 404.
    router.add(
        "https://custom2.example.com/api/models",
        urllib.error.HTTPError(
            "https://custom2.example.com/api/models",
            404,
            "Not Found",
            {},
            io.BytesIO(b""),
        ),
    )
    router.add(
        "https://custom2.example.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 100),
    )

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://custom2.example.com")
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
    # File came from the custom mirror direct-layout path, not HF.
    assert hf_mock.call_count == 0
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    assert (snap / "config.json").read_bytes() == b"x" * 100


# ---------------------------------------------------------------------------
# Codex round-10 BLOCKING — many static-bucket mirrors (S3 with
# list-bucket denied, vanilla nginx, plain CDN) return 403 / 400 for
# unknown ``/api/models`` rather than 404. Custom-mirror users on PR
# #647's contract should still get the direct-layout fallback. Cover
# 403 and 400 explicitly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("catalog_status", [400, 401, 403, 404])
def test_custom_mirror_catalog_4xx_uses_direct_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    catalog_status: int,
):
    """ANY 4xx on a custom mirror's ``/api/models`` should fall back to
    the legacy ``<base>/<owner>/<repo>/<file>`` layout — narrowing this
    to exactly 404 would break PR #647 users whose mirror returns 403
    (S3 with list-bucket denied) or 400 (CDN path-style rejection)."""
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "5555" * 10
    files = [("config.json", 100)]

    router = _UrlRouter()
    router.add(
        f"https://custom-{catalog_status}.example.com/api/models",
        urllib.error.HTTPError(
            f"https://custom-{catalog_status}.example.com/api/models",
            catalog_status,
            f"HTTP {catalog_status}",
            {},
            io.BytesIO(b""),
        ),
    )
    router.add(
        f"https://custom-{catalog_status}.example.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 100),
    )

    monkeypatch.setenv(
        "RAPID_MLX_MODEL_MIRROR", f"https://custom-{catalog_status}.example.com"
    )
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download") as hf_mock,
    ):
        ok = _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok, (
        f"4xx={catalog_status} on custom mirror should fall back to direct-layout"
    )
    assert hf_mock.call_count == 0
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    assert (snap / "config.json").read_bytes() == b"x" * 100


# ---------------------------------------------------------------------------
# Codex round-9 BLOCKING #2 — non-default revision must skip the mirror
# wholesale so the caller's ``snapshot_download(..., revision="<sha>")``
# can pin the right ref. We do not currently support per-revision
# mirroring (the R2 catalog is built from default-branch snapshots).
# ---------------------------------------------------------------------------


def test_non_default_revision_skips_mirror_entirely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"

    router = _UrlRouter()
    # No routes — any HTTP call would AssertionError, proving we
    # short-circuit before touching the network.
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch("huggingface_hub.model_info") as info_mock,
        patch("huggingface_hub.hf_hub_download") as hf_mock,
    ):
        ok = _mirror.download_with_mirror_fallback(
            repo_id, cache_dir=tmp_path, revision="abcd" * 10
        )

    assert ok is False
    # We didn't even ask HF for metadata, let alone touch the network.
    assert info_mock.call_count == 0
    assert hf_mock.call_count == 0
    assert router.requests == []


def test_revision_main_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``revision="main"`` is equivalent to default — must NOT short-circuit."""
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abcd" * 10
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
        ok = _mirror.download_with_mirror_fallback(
            repo_id, cache_dir=tmp_path, revision="main"
        )

    assert ok is True


# ---------------------------------------------------------------------------
# Codex round-9 BLOCKING #1 — when we sent a ``Range`` request but the
# server returned 200 (range ignored), we must discard the stale
# ``.part`` prefix AND not feed it to the SHA hasher. Otherwise a valid
# fresh download is rejected as sha-mismatch.
# ---------------------------------------------------------------------------


def test_resume_range_ignored_200_response_discards_stale_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Some R2 / proxy configs ignore Range and return the full body with
    status 200. With LFS sha256 enabled, the hasher must not see the
    discarded prefix — otherwise the digest mismatches and the file
    falls back to HF unnecessarily.
    """
    import hashlib

    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "feed" * 10
    full_body = b"Q" * 200
    sha = hashlib.sha256(full_body).hexdigest()
    files = [("model.safetensors", 200, sha)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Plant a stale .part with different bytes — server will ignore our
    # Range header and return the full body fresh.
    snap_dir = (
        tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    )
    snap_dir.mkdir(parents=True, exist_ok=True)
    stale_part = _sidecar_part_path(
        tmp_path, "mlx-community/Qwen3-0.6B-4bit", "model.safetensors"
    )
    stale_part.parent.mkdir(parents=True, exist_ok=True)
    stale_part.write_bytes(b"Z" * 50)  # 50 bytes of garbage

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # Server returns 200 (range ignored), Content-Length is the FULL body.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, full_body),
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
    # File was written from R2 (no HF fallback needed); sha matches.
    assert hf_mock.call_count == 0
    final = snap_dir / "model.safetensors"
    assert final.read_bytes() == full_body
    # The Range header WAS sent (existing .part > 0 triggers it) — we
    # didn't pretend nothing was there.
    range_reqs = [
        r
        for r in router.requests
        if "model.safetensors" in r["url"] and r["headers"].get("Range") is not None
    ]
    assert len(range_reqs) == 1, f"expected exactly one Range request, got {range_reqs}"


# ---------------------------------------------------------------------------
# Codex round-11 BLOCKING #1 — a cached LFS file with the right SIZE but
# wrong BYTES (e.g. a previous partial corruption or a bit-flip on disk)
# must be re-hashed against ``expected_sha256`` and refetched if it
# fails. Size-only acceptance is not enough for weight shards.
# ---------------------------------------------------------------------------


def test_cached_lfs_file_with_wrong_sha_is_refetched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Plant a same-size but BYTE-WRONG file at the snapshot path. The
    mirror module must hash it, detect the mismatch, drop it, and
    refetch from R2. Returning ``"cached"`` would be a silent
    integrity regression — round-11 fix forbids that."""
    import hashlib

    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "deca" * 10
    good_bytes = b"G" * 200
    corrupt_bytes = b"X" * 200  # same size, wrong content
    sha = hashlib.sha256(good_bytes).hexdigest()
    files = [("model.safetensors", 200, sha)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Pre-plant the corrupt bytes at the snapshot path.
    snap_dir = (
        tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    )
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "model.safetensors").write_bytes(corrupt_bytes)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # R2 serves the GOOD bytes so the refetch can succeed.
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, good_bytes),
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
    # The corrupt cache was dropped and replaced with the right bytes.
    assert (snap_dir / "model.safetensors").read_bytes() == good_bytes
    # R2 served the refetch (so an R2 file request happened — i.e.
    # the cached-as-"cached" short-circuit was NOT taken).
    r2_file_calls = [r for r in router.requests if "model.safetensors" in r["url"]]
    assert len(r2_file_calls) == 1, (
        f"expected exactly one R2 refetch, got {r2_file_calls}"
    )
    # HF was not asked to do anything — R2 had it.
    assert hf_mock.call_count == 0


def test_cached_lfs_file_with_correct_sha_is_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The same path with CORRECT bytes is NOT refetched — the sha-256
    matches, the file is accepted as cached, and no R2 / HF traffic
    happens for that file."""
    import hashlib

    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "feed" * 10
    good_bytes = b"G" * 200
    sha = hashlib.sha256(good_bytes).hexdigest()
    files = [("model.safetensors", 200, sha)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    snap_dir = (
        tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    )
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "model.safetensors").write_bytes(good_bytes)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # No file URL registered — if production tries to hit R2 for this
    # file, the router raises AssertionError.

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
    assert (snap_dir / "model.safetensors").read_bytes() == good_bytes
    assert hf_mock.call_count == 0
    # Only the catalog was hit.
    file_calls = [r for r in router.requests if "model.safetensors" in r["url"]]
    assert file_calls == []


# ---------------------------------------------------------------------------
# Codex round-11 BLOCKING #2 — concurrent ``rapid-mlx`` runs on the
# same model must serialize on the ``.part`` lock. Smoke-test that
# acquiring + releasing the lock doesn't break and works on the test
# platform (macOS posix). Full concurrency simulation is impractical
# in a unit test, but verifying the helper round-trips guards against
# regressions like accidentally dropping the lock import.
# ---------------------------------------------------------------------------


def test_part_lock_roundtrip(tmp_path: Path):
    lock_path = tmp_path / "x.lock"
    fh = _mirror._acquire_part_lock(lock_path)
    # On posix we get a file handle; on Windows we get None. Either way
    # the release call must not raise.
    try:
        assert fh is not None  # posix CI
        # Reacquiring without releasing would deadlock on the same fd —
        # don't try that. Just smoke-test release.
    finally:
        _mirror._release_part_lock(fh, lock_path)
    # Codex round-12 BLOCKING: the lock file is INTENTIONALLY NOT
    # cleaned up on release — unlinking would split waiters and new
    # acquirers onto different inodes and let concurrent writers race
    # on the same ``.part``. Verify the sidecar persists so the next
    # acquirer can re-lock the same inode.
    assert lock_path.exists()


def test_part_lock_reacquire_uses_same_inode(tmp_path: Path):
    """After release, a fresh ``_acquire_part_lock`` on the same path
    must see the SAME inode as before — otherwise process A's release
    and process C's acquire would happen on a different file from B's
    in-flight ``flock``, allowing the very race round-12 patched out.
    """
    lock_path = tmp_path / "y.lock"
    fh1 = _mirror._acquire_part_lock(lock_path)
    inode1 = lock_path.stat().st_ino
    _mirror._release_part_lock(fh1, lock_path)
    # File must still exist on disk after release.
    assert lock_path.exists()
    inode_after_release = lock_path.stat().st_ino
    assert inode_after_release == inode1
    # And a re-acquire grabs the same inode.
    fh2 = _mirror._acquire_part_lock(lock_path)
    inode2 = lock_path.stat().st_ino
    try:
        assert inode2 == inode1
    finally:
        _mirror._release_part_lock(fh2, lock_path)


# ---------------------------------------------------------------------------
# Codex round-11 NIT #3 — a malformed ``RAPID_MLX_MODEL_MIRROR`` URL
# should not crash the pull. The catalog fetch must catch the
# ``Request()`` constructor's own ``ValueError``.
# ---------------------------------------------------------------------------


def test_malformed_mirror_url_returns_false_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # ``urllib.request.Request`` raises ValueError on unknown URL types.
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "not-a-url://garbage")
    # No urlopen patch — if anything bypasses the guard we'll get a
    # real network error not an assertion.
    data, status = _mirror.fetch_catalog_with_status("not-a-url://garbage")
    # Should fall through to "transient" (None, None) without raising.
    assert data is None
    # status may be None (URLError) or some int — both are acceptable
    # "no catalog here" signals.


# ---------------------------------------------------------------------------
# Codex round-13 BLOCKING #1 — a symlink target pointing OUTSIDE the
# repo's cache dir must be rejected (refused as cached) even if the
# pointed-to file happens to match expected_size / sha256.
# ---------------------------------------------------------------------------


def test_cached_symlink_escaping_repo_root_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Plant a malicious symlink at the snapshot path pointing to a
    file outside the repo's cache dir. Even though the pointed-to file
    has the right size, the mirror module must drop it and refetch
    instead of pinning ``refs/main`` to a snapshot dir containing a
    rogue symlink.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "1234" * 10
    files = [("config.json", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Outside-the-repo file with the same expected size.
    outside = tmp_path / "outside_payload.bin"
    outside.write_bytes(b"M" * 100)  # matching size

    # Plant the malicious symlink at the cached path.
    snap_dir = (
        tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    )
    snap_dir.mkdir(parents=True, exist_ok=True)
    symlink_path = snap_dir / "config.json"
    symlink_path.symlink_to(outside)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # R2 serves the real content for the refetch.
    real_bytes = b"R" * 100
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, real_bytes),
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
    # The symlink was dropped — the file at the path is now the real
    # bytes from R2, NOT the outside payload.
    assert symlink_path.is_symlink() is False
    assert symlink_path.read_bytes() == real_bytes
    # Outside file is untouched (we deleted the symlink, not the
    # target).
    assert outside.read_bytes() == b"M" * 100
    # R2 served the refetch.
    r2_file_calls = [r for r in router.requests if "config.json" in r["url"]]
    assert len(r2_file_calls) == 1
    assert hf_mock.call_count == 0


# ---------------------------------------------------------------------------
# Codex round-13 BLOCKING #2 — _safe_unlink must remove BROKEN symlinks
# (path.exists() returns False for them, but the dangling link still
# needs cleanup or the later tmp.rename() fails).
# ---------------------------------------------------------------------------


def test_safe_unlink_removes_broken_symlink(tmp_path: Path):
    broken_target = tmp_path / "does_not_exist"
    link = tmp_path / "broken_link"
    link.symlink_to(broken_target)
    # Confirm precondition: link exists as a symlink, target does not.
    assert link.is_symlink()
    assert not link.exists()  # Path.exists() follows the link → False

    _mirror._safe_unlink(link)

    # The broken symlink is now gone.
    assert not link.is_symlink()
    # And os.lstat would raise → confirm via the parent listing.
    assert "broken_link" not in [p.name for p in tmp_path.iterdir()]


# ---------------------------------------------------------------------------
# Codex round-13 NIT #3 — refs/main is written as UTF-8, not platform
# default encoding.
# ---------------------------------------------------------------------------


def test_refs_main_written_as_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abcd" * 10  # ASCII; any encoding gives the same bytes,
    # but we check the file is utf-8-decodable as a contract.
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
    # Explicitly read as utf-8 — would raise on non-utf-8 encodings.
    content = refs_main.read_text(encoding="utf-8")
    assert content == revision


# ---------------------------------------------------------------------------
# Codex round-14 BLOCKING #1+#2 — sidecar dir contract:
#   * ``.part`` and ``.lock`` live in ``repo_root/.rapid-mlx-mirror/``,
#     NEVER under ``snapshots/<sha>/``.
#   * Their names are derived from a flattened key, not from
#     ``.<file>.rapid-mlx-mirror.{part,lock}`` (which could collide
#     with a legitimate repo file).
# ---------------------------------------------------------------------------


def test_sidecar_dir_holds_part_and_lock_not_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "5a1d" * 10
    files = [("model.safetensors", 200)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, b"X" * 200),
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

    # Snapshot dir contains the real file, nothing else.
    snap_contents = sorted(p.name for p in snap.iterdir() if p.is_file())
    assert snap_contents == ["model.safetensors"]

    # Sidecar dir holds the lock file (kept on disk per round-12).
    sidecar = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / ".rapid-mlx-mirror"
    assert sidecar.is_dir()
    sidecar_contents = sorted(p.name for p in sidecar.iterdir() if p.is_file())
    # Lock stays; ``.part`` was renamed to target on success.
    assert all(n.endswith(".lock") or n.endswith(".part") for n in sidecar_contents), (
        f"unexpected sidecar contents: {sidecar_contents}"
    )


def test_sidecar_key_collision_safe_with_hidden_repo_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A repo can legitimately contain a file named like our OLD temp
    file (``.foo.rapid-mlx-mirror.part``). Verify the sidecar key derived
    from that filename doesn't collide with anything in snap/, and the
    real repo file lands at the snapshot path with the right bytes
    while the sidecar artifacts live in a SEPARATE dir."""
    repo_id = "mlx-community/Hidden-Asset"
    revision = "babe" * 10
    # 12 bytes — matches the literal R2 payload below.
    files = [(".foo.rapid-mlx-mirror.part", 12)]
    catalog = _catalog_payload([("hidden-asset", repo_id, "mirrored")])

    repo_root = tmp_path / "models--mlx-community--Hidden-Asset"
    snap = repo_root / "snapshots" / revision

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    payload = b"legit-asset"  # 11 bytes — fix expected size to match
    files = [(".foo.rapid-mlx-mirror.part", len(payload))]
    router.add(
        "https://models.rapidmlx.com/mlx-community/Hidden-Asset/.foo.rapid-mlx-mirror.part",
        _FakeResponse(200, payload),
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
    assert hf_mock.call_count == 0
    # The repo file lands at the real path — same name as the OLD
    # temp file pattern, but now safe because the temp file lives in
    # the sidecar dir.
    assert (snap / ".foo.rapid-mlx-mirror.part").read_bytes() == payload
    # And the sidecar dir is distinct from snap.
    sidecar = repo_root / ".rapid-mlx-mirror"
    assert sidecar.is_dir()
    # The lock file lives there (kept on disk).
    assert any(p.name.endswith(".lock") for p in sidecar.iterdir())
    # And ``snap_dir`` does NOT contain any sidecar artifacts that
    # would collide with the legitimate repo file's name.
    assert sorted(p.name for p in snap.iterdir() if p.is_file()) == [
        ".foo.rapid-mlx-mirror.part"
    ]


# ---------------------------------------------------------------------------
# Codex round-14 BLOCKING #3 — cached symlinks must resolve under
# ``repo_root/blobs/``, not just anywhere under repo_root. A symlink
# pointing at ``refs/main`` (40 bytes) would otherwise be accepted as a
# 40-byte cached file.
# ---------------------------------------------------------------------------


def test_cached_symlink_to_refs_main_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "ab12" * 10  # 40 ASCII chars
    files = [("config.json", 40)]  # SAME size as the ref file's bytes
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    repo_root = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
    refs_dir = repo_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(revision, encoding="utf-8")  # 40 bytes

    snap = repo_root / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    # Plant a symlink at the cached path → refs/main (inside repo_root
    # but NOT under blobs/).
    sym = snap / "config.json"
    sym.symlink_to(refs_dir / "main")

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    real_bytes = b"R" * 40
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, real_bytes),
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
    # The intra-cache symlink was rejected (not under blobs/) and the
    # real bytes were re-downloaded from R2.
    assert sym.is_symlink() is False
    assert sym.read_bytes() == real_bytes
    # ``refs/main`` itself was untouched until our final pin.
    assert hf_mock.call_count == 0


# ---------------------------------------------------------------------------
# Codex round-14 BLOCKING #4 — snap_dir itself being a symlink must be
# caught up-front, not just its descendant parents.
# ---------------------------------------------------------------------------


def test_snap_dir_as_symlink_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "cafe" * 10
    files = [("config.json", 100)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Pre-create the snapshot path AS A SYMLINK to a malicious dir.
    # NB: production calls ``snap_dir.mkdir(exist_ok=True)`` which
    # succeeds even if snap_dir is already a symlink to a real dir.
    # The new guard must reject downloads on such a setup.
    malicious_dir = tmp_path / "outside_malicious_dir"
    malicious_dir.mkdir()
    repo_root = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
    snapshots_parent = repo_root / "snapshots"
    snapshots_parent.mkdir(parents=True, exist_ok=True)
    (snapshots_parent / revision).symlink_to(malicious_dir)

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/config.json",
        _FakeResponse(200, b"x" * 100),
    )

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        # HF would write to its own cache layout — for this test it
        # doesn't matter, we just need the per-file loop to complete.
        snap = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap.mkdir(parents=True, exist_ok=True)
        # Note: snap is the SYMLINKED dir → writes go to malicious_dir.
        # HF doesn't know this; the rapid-mlx mirror's job was to refuse
        # to participate. We only care here that R2 didn't write.
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
        _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    # The mirror module refused to write into the symlinked snap_dir.
    # R2 must NOT have been hit for this file (the per-file _do_file
    # returned "skip-symlink-snapdir" before R2 attempt).
    r2_file_calls = [r for r in router.requests if "config.json" in r["url"]]
    assert r2_file_calls == [], (
        f"R2 was probed despite symlinked snap_dir: {r2_file_calls}"
    )


# ---------------------------------------------------------------------------
# Codex round-14 NIT #5 — when an HF-style symlink already points at
# ``blobs/<expected_sha256>``, skip the full rehash (would otherwise
# turn a no-op warm pull into a multi-GB disk scan).
# ---------------------------------------------------------------------------


def test_cached_hf_blob_symlink_skips_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import hashlib

    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "f0f0" * 10
    payload = b"P" * 200
    sha = hashlib.sha256(payload).hexdigest()
    files = [("model.safetensors", 200, sha)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    # Build the HF-style cache layout: blobs/<sha> + snapshots/<rev>/foo
    # symlinked to ../../blobs/<sha>.
    repo_root = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
    blobs = repo_root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / sha).write_bytes(payload)
    snap = repo_root / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    link = snap / "model.safetensors"
    link.symlink_to(blobs / sha)

    # If the production code DID rehash, it would open the file and
    # we'd get a successful "cached". Either way the test only cares
    # that the hasher was NOT used — we instrument hashlib.sha256 to
    # detect any new hasher created during the call.
    import vllm_mlx._mirror as m

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # NO file URL registered — if production tries to download, we'll
    # AssertionError, which would catch the case where the blob-name
    # shortcut accidentally falls through.

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download") as hf_mock,
    ):
        ok = m.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)

    assert ok
    # The HF blob symlink was accepted on shortcut (name matches sha).
    # No file URL was hit, no HF call.
    file_reqs = [r for r in router.requests if "model.safetensors" in r["url"]]
    assert file_reqs == []
    assert hf_mock.call_count == 0
    # File still resolves correctly.
    assert link.read_bytes() == payload


# ---------------------------------------------------------------------------
# Issue #652 — after a sha-verified R2 download of an LFS file, the
# bytes must land at ``blobs/<sha>`` and the snapshot path must be a
# RELATIVE symlink to that blob. Matches HF's own cache layout, so the
# blob-name shortcut at the warm-cache check fires uniformly for both
# R2-sourced and HF-sourced cache state — avoiding multi-GB rehashes
# on every warm pull.
# ---------------------------------------------------------------------------


def test_r2_lfs_download_writes_blob_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import hashlib

    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "a1b2" * 10
    payload = b"W" * 1024
    sha = hashlib.sha256(payload).hexdigest()
    files = [("model.safetensors", 1024, sha)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, payload),
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
    assert hf_mock.call_count == 0

    repo_root = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
    blob = repo_root / "blobs" / sha
    snap_file = repo_root / "snapshots" / revision / "model.safetensors"

    # The verified bytes live at ``blobs/<sha>`` as a regular file.
    assert blob.is_file()
    assert not blob.is_symlink()
    assert blob.read_bytes() == payload

    # The snapshot path is a symlink (NOT a regular file) pointing at
    # the blob via a RELATIVE path. Matching HF's own layout exactly
    # is what makes the warm-pull blob-name shortcut fire — an
    # absolute symlink would also work for ``resolve().name`` but
    # diverges from HF's layout, so codify "relative" here.
    assert snap_file.is_symlink()
    link_target = os.readlink(snap_file)
    assert not os.path.isabs(link_target), (
        f"snapshot symlink must be relative, got {link_target!r}"
    )
    # Resolves to the blob.
    assert snap_file.resolve() == blob.resolve()
    # And reading through the symlink still yields the payload.
    assert snap_file.read_bytes() == payload


def test_warm_r2_pull_uses_blob_name_shortcut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Second pull of the same model must NOT rehash the LFS blob.

    Issue #652 root cause: the old code wrote R2 bytes as a regular
    file at ``snapshots/<sha>/<file>``, so the cached-check site
    couldn't take its ``symlink.name == expected_sha256`` shortcut on
    the second pull — every warm pull rehashed multi-GB shards. With
    the blob+symlink layout, the shortcut fires and the second pull
    is rehash-free.
    """
    import hashlib

    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "b2c3" * 10
    payload = b"S" * 4096
    sha = hashlib.sha256(payload).hexdigest()
    files = [("model.safetensors", 4096, sha)]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    router.add(
        "https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/model.safetensors",
        _FakeResponse(200, payload),
    )

    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")

    # First (cold) pull — populates blobs/<sha> and the snapshot
    # symlink via the R2 path.
    with (
        patch("urllib.request.urlopen", side_effect=router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download") as hf_mock,
    ):
        assert _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)
        assert hf_mock.call_count == 0

    repo_root = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
    snap_file = repo_root / "snapshots" / revision / "model.safetensors"
    assert snap_file.is_symlink()  # cold pull installed the layout.

    # Second (warm) pull — instrument ``hashlib.sha256`` so any new
    # hasher creation during the second pull is detected. The blob-
    # name shortcut at the cached-check site must skip the rehash
    # entirely. A second router with NO file URL registered ensures
    # an accidental refetch would AssertionError too.
    warm_router = _UrlRouter()
    warm_router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )

    import hashlib as _hashlib_mod

    real_sha256 = _hashlib_mod.sha256
    hasher_calls: list[None] = []

    def _spy_sha256(*args, **kwargs):
        hasher_calls.append(None)
        return real_sha256(*args, **kwargs)

    with (
        patch("urllib.request.urlopen", side_effect=warm_router),
        patch(
            "huggingface_hub.model_info",
            return_value=_mk_model_info(revision, files),
        ),
        patch("huggingface_hub.hf_hub_download") as hf_mock_warm,
        patch.object(_hashlib_mod, "sha256", side_effect=_spy_sha256),
    ):
        assert _mirror.download_with_mirror_fallback(repo_id, cache_dir=tmp_path)
        assert hf_mock_warm.call_count == 0

    # No file URL probed on the warm pull (only the catalog).
    file_reqs = [r for r in warm_router.requests if "model.safetensors" in r["url"]]
    assert file_reqs == [], f"warm pull refetched LFS file: {file_reqs}"
    # And — the load-bearing assertion — no sha256 hasher was created
    # during the warm pull. The blob-name shortcut fired.
    assert hasher_calls == [], (
        f"warm pull rehashed the LFS blob ({len(hasher_calls)} hashers)"
    )


# ---------------------------------------------------------------------------
# Wiring assertions: serve must NOT bypass the mirror
# ---------------------------------------------------------------------------


def _function_calls_global(func, name: str) -> bool:
    """Return True iff ``func``'s bytecode contains a LOAD_GLOBAL/NAME/DEREF
    of ``name`` followed by a CALL op within the same function body.

    Stricter than a bare ``LOAD_GLOBAL`` check — a future refactor that
    merely references ``name`` (e.g. ``f = _ensure_model_downloaded``
    without ever calling it) would still satisfy LOAD_GLOBAL but
    silently re-introduce #651. The CALL-follows requirement closes
    that gap.

    Codex round-1 BLOCKING #1 to PR #654.
    """
    import dis

    insts = list(dis.get_instructions(func))
    for idx, ins in enumerate(insts):
        if (
            ins.opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF")
            and ins.argval == name
        ):
            # Walk forward looking for a CALL. Any number of LOAD_FAST /
            # LOAD_ATTR / LOAD_CONST / PUSH_NULL ops can stack arguments
            # between the load and the call; anything else (STORE_*, a
            # second LOAD_GLOBAL of an unrelated name, RETURN_*, etc.)
            # means the loaded reference was used for something other
            # than calling it — keep scanning for another LOAD of the
            # same name.
            allowed_between = {
                "LOAD_FAST",
                "LOAD_ATTR",
                "LOAD_CONST",
                "LOAD_DEREF",
                "PUSH_NULL",
                "COPY",
                "SWAP",
                "PRECALL",
                "KW_NAMES",
                "LOAD_METHOD",
                "CACHE",
            }
            for follow in insts[idx + 1 : idx + 12]:
                if follow.opname in ("CALL", "CALL_FUNCTION", "CALL_METHOD"):
                    return True
                if follow.opname in allowed_between:
                    continue
                # Hit a non-argument-stacking op — this LOAD was not
                # immediately followed by a CALL.
                break
    return False


def test_serve_command_calls_ensure_model_downloaded():
    """``rapid-mlx serve <alias>`` on a cold cache must route through the
    R2 mirror — not fall into ``mlx_lm.load`` → ``snapshot_download``
    directly. Issue #651: Desktop saw HF tqdm ``Fetching 9 files: 0%``
    streaming the 6.7 GB shard at ~5 MB/s while the mirror would have
    delivered it at ~50 MB/s.

    The cheapest defense is ``_ensure_model_downloaded(args.model)``
    early in serve_command — it's a no-op on local paths / fully-cached
    repos, and tries the mirror before HF on cold pulls. A future
    refactor that drops this call would re-introduce #651, so this is
    a bytecode-level wiring assertion that requires both the LOAD and
    a CALL following it.
    """
    from vllm_mlx import cli

    assert _function_calls_global(cli.serve_command, "_ensure_model_downloaded"), (
        "serve_command must CALL _ensure_model_downloaded (not just "
        "reference it) so cold-cache serves go through the R2 mirror "
        "(issue #651). Codex round-1 BLOCKING #1 to PR #654."
    )


# ---------------------------------------------------------------------------
# Issue #651 follow-up: per-file progress UX.
#
# User reported (rapid-mlx v0.7.27) that ``rapid-mlx pull <alias>`` prints
# the banner then sits silent for minutes while multi-GB shards stream
# via R2. The HF fallback path shows tqdm progress naturally; only the
# R2 path was silent. Fix: emit one line per file at the point it lands
# (under stdlib only — no tqdm / no carriage-return animations). Output
# must degrade gracefully when stdout is non-TTY (no ANSI escapes).
# ---------------------------------------------------------------------------


def _full_pull_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: list[tuple[str, int]],
    repo_id: str = "mlx-community/Qwen3-0.6B-4bit",
    revision: str | None = None,
):
    """Set up a fully-mirrored pull with R2 serving every file.

    Returns ``(router, revision)`` for the caller's assertions. The
    caller is responsible for invoking ``download_with_mirror_fallback``
    inside its own ``with patch(...)`` block — that way each progress
    test can layer on its own stdout / isatty patches.
    """
    if revision is None:
        revision = "f00d" * 10
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])
    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    for fname, size in files:
        url = f"https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/{fname}"
        router.add(url, _FakeResponse(200, b"x" * size))
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    return router, revision


def test_progress_lines_print_in_expected_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Each landed file emits a ``[N/M] <fname> R2 (X MB)`` line.

    The fix for #651: previously the R2 pull printed only a banner, then
    nothing for minutes. Now the user sees one completion line per file.
    """
    files = [
        ("config.json", 100),
        ("model.safetensors", 250),
        ("tokenizer.json", 75),
    ]
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    router, revision = _full_pull_scaffold(tmp_path, monkeypatch, files, repo_id)

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
    assert hf_mock.call_count == 0  # every file landed via R2
    captured = capsys.readouterr()

    # Strip ANSI for the format assertions (the suite runs under pytest,
    # which captures stdout — the production code therefore takes the
    # non-TTY branch and emits no ANSI here. We still strip defensively
    # in case a future fixture turns isatty back on.)
    import re

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    plain = ansi.sub("", captured.out)

    # One ``[N/M] <fname>`` line per file, all three present, none repeated.
    for n, (fname, _size) in enumerate(files, start=1):
        marker = f"[{n}/{len(files)}]"
        # Find the line that mentions this index AND this filename — order
        # is non-deterministic across workers, so we don't assert which
        # filename pairs with which index, only that every filename has
        # SOME ``[*/3]`` marker line.
        assert marker in plain, f"missing progress marker {marker} in:\n{plain}"
        assert fname in plain, f"missing filename {fname} in progress output"
    # R2 tag appears on every per-file line (3 files, all R2 hits).
    assert plain.count("R2 (") == len(files), (
        f"expected one ``R2 (`` tag per file, got:\n{plain}"
    )
    # Up-front file-count line — the user's #651 complaint was "no
    # feedback after the banner" — this is the first signal.
    assert f"Found {len(files)} files" in plain
    # Final summary still printed.
    assert "Pulled 3 files" in plain


def test_progress_no_ansi_escapes_when_stdout_not_a_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """When stdout is not a TTY (pipe, CI), progress must be plain text.

    No bold, no dim, no carriage returns — just one appended line per
    file. The production code gates ANSI on
    ``sys.stdout.isatty() and "NO_COLOR" not in os.environ``; this test
    locks that contract in place so a future refactor doesn't spam
    escape codes into CI logs.
    """
    files = [("config.json", 100), ("model.safetensors", 200)]
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    router, revision = _full_pull_scaffold(tmp_path, monkeypatch, files, repo_id)

    # pytest's capsys already captures stdout into a non-TTY StringIO —
    # ``isatty()`` returns False there. Belt-and-braces: also clear
    # NO_COLOR (which would force ANSI off regardless) to make the
    # isatty branch the actual cause of plain output.
    monkeypatch.delenv("NO_COLOR", raising=False)

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
    captured = capsys.readouterr()
    # No ANSI escapes anywhere in the output.
    assert "\x1b[" not in captured.out, (
        f"non-TTY output must not contain ANSI escapes, got:\n{captured.out!r}"
    )
    # No carriage-return animation either — only newlines.
    assert "\r" not in captured.out, (
        f"non-TTY output must not use carriage returns, got:\n{captured.out!r}"
    )


def test_progress_file_count_matches_downloaded_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """``[N/M]`` totals must match the number of files actually pulled.

    Off-by-one regression guard: every file emits exactly one progress
    line, the M-of-N denominator matches the file list length, and the
    final ``Pulled <K> files`` count agrees with both.
    """
    # Mix R2 hits (200) and HF fallbacks (404 → HF) so the test covers
    # both per-file paths flowing through the same progress counter.
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "beef" * 10
    files = [
        ("config.json", 50),
        ("model.safetensors", 150),
        ("tokenizer.json", 30),
        ("tokenizer_config.json", 20),
    ]
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo_id, "mirrored")])

    router = _UrlRouter()
    router.add(
        "https://models.rapidmlx.com/api/models",
        _FakeResponse(200, json.dumps(catalog).encode()),
    )
    # config.json + tokenizer.json land via R2; the others 404 → HF.
    r2_ok = {"config.json", "tokenizer.json"}
    for fname, size in files:
        url = f"https://models.rapidmlx.com/mlx-community/Qwen3-0.6B-4bit/{fname}"
        if fname in r2_ok:
            router.add(url, _FakeResponse(200, b"x" * size))
        else:
            router.add(url, _FakeResponse(404, b""))

    def _fake_hf(repo_id, filename, revision, cache_dir=None):
        snap = (
            Path(cache_dir)
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snap.mkdir(parents=True, exist_ok=True)
        expected_size = next(s for n, s in files if n == filename)
        (snap / filename).write_bytes(b"h" * expected_size)
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
    captured = capsys.readouterr()
    import re

    # Count the ``[N/4]`` markers — should be exactly len(files), each
    # with a distinct N. The denominator M must equal len(files) on
    # every line (no off-by-one in ``total_files_planned``).
    markers = re.findall(r"\[(\d+)/(\d+)\]", captured.out)
    # Filter to the progress markers (denominator == file count). Up-
    # front and summary lines don't use the ``[N/M]`` shape.
    progress_markers = [(int(n), int(m)) for n, m in markers if int(m) == len(files)]
    assert len(progress_markers) == len(files), (
        f"expected {len(files)} progress lines, got {len(progress_markers)} in:\n"
        f"{captured.out}"
    )
    # Every N in 1..M appears exactly once.
    assert sorted(n for n, _m in progress_markers) == list(range(1, len(files) + 1))
    # Final summary count matches.
    assert f"Pulled {len(files)} files" in captured.out


def test_safe_display_name_strips_control_chars():
    """Filenames from external HF metadata can't inject terminal escapes.

    Codex round-2 BLOCKING on PR #657: ``_validate_relative_filename``
    only blocks path traversal — it does NOT strip ANSI / control
    characters. A malicious or accidentally-malformed sibling entry
    like ``evil\\x1b[2Jfile.bin`` would clear the terminal when echoed
    by the per-file progress line. ``_safe_display_name`` is the
    display-side defense.
    """
    # ANSI clear-screen sequence stripped.
    assert "\x1b" not in _mirror._safe_display_name("evil\x1b[2Jfile.bin")
    # Tab / newline / carriage-return stripped.
    assert "\n" not in _mirror._safe_display_name("a\nb.bin")
    assert "\r" not in _mirror._safe_display_name("a\rb.bin")
    assert "\t" not in _mirror._safe_display_name("a\tb.bin")
    # NUL stripped.
    assert "\x00" not in _mirror._safe_display_name("a\x00b.bin")
    # DEL (0x7f) stripped.
    assert "\x7f" not in _mirror._safe_display_name("a\x7fb.bin")
    # Unicode C1 control (CSI, U+009B) — same terminal-injection vector
    # as ESC[. Codex round-3 BLOCKING on PR #657.
    assert "" not in _mirror._safe_display_name("a[2Jb.bin")
    # Bidi override (U+202E) — would visually swap ``...exe.txt`` into
    # ``...txt.exe`` and mislead the user about the file type.
    assert "‮" not in _mirror._safe_display_name("a‮exe.txt")
    # Zero-width joiner (U+200D) — Cf category, also stripped.
    assert "‍" not in _mirror._safe_display_name("a‍b.bin")
    # Non-control characters preserved verbatim, including Unicode.
    assert (
        _mirror._safe_display_name("model-v1.2.safetensors") == "model-v1.2.safetensors"
    )
    assert _mirror._safe_display_name("café.bin") == "café.bin"
    # Empty-after-strip falls back to a placeholder.
    assert _mirror._safe_display_name("\x00\x01\x02") == "<unprintable>"
    # Long filenames are truncated in the middle so the head + tail
    # stay visible — the user still recognizes their file.
    long = "a" * 200 + "_TAIL.bin"
    out = _mirror._safe_display_name(long, max_len=40)
    assert len(out) <= 40
    assert out.endswith("_TAIL.bin") or "TAIL" in out
