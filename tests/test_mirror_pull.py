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
import urllib.error
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

    # Pre-stage the partial file at the namespaced temp path the
    # production code computes (``.<file>.rapid-mlx-mirror.part``).
    # Codex round-4 BLOCKING #1 made the temp name distinct from a
    # potential ``<file>.part`` repo asset.
    snap = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    part = snap / ".model.safetensors.rapid-mlx-mirror.part"
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
    part = snap / ".model.safetensors.rapid-mlx-mirror.part"
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
    part = snap / ".model.safetensors.rapid-mlx-mirror.part"
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
    part = snap / ".model.safetensors.rapid-mlx-mirror.part"
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
    stale_part = snap_dir / ".model.safetensors.rapid-mlx-mirror.part"
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
