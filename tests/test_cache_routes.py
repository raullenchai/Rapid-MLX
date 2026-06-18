# SPDX-License-Identifier: Apache-2.0
"""Wire-level tests for the KV cache export/import HTTP API (#476 stub).

The engine integration is the follow-up PR's job — these tests cover
the protocol surface that the stub freezes: auth, path sandbox,
manifest validation, and the explicit 501s on the engine-touching paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.cache.protocol import (
    PROTOCOL_VERSION,
    InvalidExportPathError,
    MalformedManifestError,
    Manifest,
    read_manifest,
    resolve_cache_dir,
    write_manifest,
)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Point the export sandbox at an isolated tmp dir for the test."""
    export_root = tmp_path / "exports"
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(export_root))
    return export_root


@pytest.fixture
def cache_client(monkeypatch, sandbox):
    """FastAPI TestClient with the cache router + auth enabled."""
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.cache import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = SimpleNamespace()  # unused — stub doesn't touch the engine
    cfg.model_name = "test-model"

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), sandbox=sandbox)

    reset_config()


def _auth() -> dict:
    return {"Authorization": "Bearer test-secret"}


# ---------------------------------------------------------------------------
# protocol.resolve_cache_dir — unit tests at the helper level
# ---------------------------------------------------------------------------


def test_resolve_cache_dir_returns_sandbox_root_for_none(sandbox):
    """``None`` resolves to the sandbox root itself, which is created."""
    resolved = resolve_cache_dir(None)
    assert resolved == Path(sandbox).resolve()
    assert resolved.is_dir()


def test_resolve_cache_dir_relative_path_is_joined(sandbox):
    """Relative paths resolve under the sandbox root."""
    resolved = resolve_cache_dir("session-a")
    assert resolved == (Path(sandbox).resolve() / "session-a")


def test_resolve_cache_dir_rejects_dotdot_segment(sandbox):
    """``..`` in any segment is rejected before realpath even runs."""
    with pytest.raises(InvalidExportPathError, match="not allowed"):
        resolve_cache_dir("../etc/passwd")


def test_resolve_cache_dir_rejects_absolute_outside(sandbox):
    """An absolute path outside the sandbox is rejected by commonpath."""
    with pytest.raises(InvalidExportPathError, match="outside sandbox"):
        resolve_cache_dir("/tmp/anywhere-else")


def test_resolve_cache_dir_rejects_symlink_escape(sandbox):
    """A symlink whose target leaves the sandbox is rejected.

    realpath alone would pass us (we land outside), commonpath catches
    it — but a symlink ancestor whose *target* points outside still
    counts as escape even if the resolved name happens to live inside.
    """
    sandbox.mkdir(parents=True, exist_ok=True)
    outside = sandbox.parent / "outside_dir"
    outside.mkdir()
    link = sandbox / "escape"
    link.symlink_to(outside)

    with pytest.raises(InvalidExportPathError):
        resolve_cache_dir("escape/anything")


# ---------------------------------------------------------------------------
# protocol.Manifest — roundtrip + additive evolution
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    """``write_manifest`` then ``read_manifest`` recovers every field."""
    original = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="mlx-community/Qwen3.5-9B-4bit",
        quantization="4bit",
        paged_cache=True,
        turboquant_kv=False,
        index_format_version=2,
        entries=42,
        total_bytes=12_345_678,
        rapid_mlx_version="0.7.29",
        created_at="2026-06-18T00:00:00Z",
    )
    write_manifest(tmp_path, original)
    recovered = read_manifest(tmp_path)
    assert recovered == original


def test_read_manifest_rejects_invalid_json(tmp_path):
    """Malformed JSON at the manifest path surfaces a typed exception.

    Without this branch the JSONDecodeError would propagate as a 500 in
    the routes — a caller-controlled bug masquerading as a server fault.
    """
    (tmp_path / "manifest.json").write_text("not even close to JSON {")
    with pytest.raises(MalformedManifestError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_read_manifest_rejects_non_object_payload(tmp_path):
    """A JSON list at the manifest path is structurally malformed."""
    (tmp_path / "manifest.json").write_text('["this", "is", "a", "list"]')
    with pytest.raises(MalformedManifestError, match="JSON object"):
        read_manifest(tmp_path)


def test_manifest_from_dict_drops_unknown_fields(tmp_path):
    """An older reader handling a newer writer's extra fields just ignores them."""
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "x",
        "future_field_v2": "something the current reader doesn't know about",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    m = read_manifest(tmp_path)
    assert m.model_id == "x"
    assert m.protocol_version == PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# auth — every route requires the bearer when --api-key is set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/v1/cache/export", {}),
        ("post", "/v1/cache/import", {"source": "anywhere"}),
        ("get", "/v1/cache/info", None),
    ],
)
def test_routes_require_auth(cache_client, method, path, body):
    """No bearer → 401 on every route."""
    client = cache_client.client
    if method == "post":
        resp = client.post(path, json=body)
    else:
        resp = client.get(path)
    assert resp.status_code == 401, resp.text


def test_routes_reject_wrong_bearer(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /v1/cache/export — stub returns 501 after passing the sandbox check
# ---------------------------------------------------------------------------


def test_export_default_destination_returns_501(cache_client):
    """No destination → uses sandbox root, then 501 with #476 link."""
    resp = cache_client.client.post("/v1/cache/export", json={}, headers=_auth())
    assert resp.status_code == 501
    detail = resp.json()["detail"]
    assert "issue" in detail and "476" in detail["issue"]
    assert detail["validated"]["protocol_version"] == PROTOCOL_VERSION
    # Resolved destination must be inside the sandbox.
    sandbox_real = str(Path(cache_client.sandbox).resolve())
    assert detail["validated"]["destination"].startswith(sandbox_real)


def test_export_rejects_path_traversal(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "../../../etc"},
        headers=_auth(),
    )
    assert resp.status_code == 403
    assert "not allowed" in resp.json()["detail"]


def test_export_rejects_absolute_outside(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"destination": "/tmp/escape-target"},
        headers=_auth(),
    )
    assert resp.status_code == 403


def test_export_rejects_invalid_max_bytes(cache_client):
    """pydantic catches the ge=1 violation as 422 before the handler runs."""
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={"max_bytes": 0},
        headers=_auth(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/cache/import — manifest mismatches surface as 409 before engine work
# ---------------------------------------------------------------------------


def _write_export_root(sandbox: Path, name: str, manifest: Manifest) -> Path:
    root = sandbox / name
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root, manifest)
    return root


def test_import_malformed_manifest_returns_400(cache_client):
    """Corrupt manifest.json at the source → 400, not 500.

    Without the dedicated mapping in ``_read_manifest_or_http``, the
    underlying ``json.JSONDecodeError`` would escape and FastAPI would
    surface it as an opaque 500 — hiding a caller-supplied bad blob
    inside a server-fault status. Codex blocking-finding regression.
    """
    bad = cache_client.sandbox / "corrupt"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ not valid json")
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "corrupt"},
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert "not valid JSON" in resp.json()["detail"]


def test_info_malformed_manifest_returns_400(cache_client):
    bad = cache_client.sandbox / "corrupt-info"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text('"a bare JSON string is not an object"')
    resp = cache_client.client.get(
        "/v1/cache/info?path=corrupt-info",
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert "JSON object" in resp.json()["detail"]


def test_import_missing_manifest_returns_404(cache_client):
    """Source path exists but has no manifest.json."""
    (cache_client.sandbox / "no-manifest").mkdir(parents=True)
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "no-manifest"},
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_import_protocol_version_mismatch_returns_409(cache_client):
    _write_export_root(
        cache_client.sandbox,
        "v999",
        Manifest(protocol_version="999", model_id="any"),
    )
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "v999", "expected_protocol_version": PROTOCOL_VERSION},
        headers=_auth(),
    )
    assert resp.status_code == 409
    assert "protocol_version" in resp.json()["detail"]


def test_import_model_id_mismatch_returns_409(cache_client):
    _write_export_root(
        cache_client.sandbox,
        "qwen",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="qwen3.5-9b-4bit"),
    )
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={
            "source": "qwen",
            "expected_model_id": "gpt-oss-20b-mxfp4-q8",
        },
        headers=_auth(),
    )
    assert resp.status_code == 409
    assert "model_id" in resp.json()["detail"]


def test_import_validated_request_returns_501(cache_client):
    """All checks pass → 501 with the parsed manifest echoed back."""
    manifest = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="qwen3.5-9b-4bit",
        entries=18,
        total_bytes=4_096_000,
    )
    _write_export_root(cache_client.sandbox, "ready", manifest)
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={
            "source": "ready",
            "expected_model_id": "qwen3.5-9b-4bit",
            "merge_strategy": "replace",
        },
        headers=_auth(),
    )
    assert resp.status_code == 501
    detail = resp.json()["detail"]
    assert detail["issue"].endswith("/476")
    assert detail["validated"]["merge_strategy"] == "replace"
    assert detail["validated"]["manifest"]["entries"] == 18


def test_import_rejects_path_traversal(cache_client):
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={"source": "../etc"},
        headers=_auth(),
    )
    assert resp.status_code == 403


def test_import_missing_source_returns_422(cache_client):
    """``source`` is required — pydantic rejects the missing field."""
    resp = cache_client.client.post(
        "/v1/cache/import",
        json={},
        headers=_auth(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/cache/info — fully implemented (the only non-stub endpoint)
# ---------------------------------------------------------------------------


def test_info_returns_manifest(cache_client):
    manifest = Manifest(
        protocol_version=PROTOCOL_VERSION,
        model_id="qwen3.5-9b-4bit",
        quantization="4bit",
        entries=18,
    )
    _write_export_root(cache_client.sandbox, "ready", manifest)
    resp = cache_client.client.get(
        "/v1/cache/info?path=ready",
        headers=_auth(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["protocol_version"] == PROTOCOL_VERSION
    assert body["manifest"]["model_id"] == "qwen3.5-9b-4bit"
    assert body["manifest"]["entries"] == 18


def test_info_missing_manifest_returns_404(cache_client):
    (cache_client.sandbox / "empty").mkdir(parents=True)
    resp = cache_client.client.get(
        "/v1/cache/info?path=empty",
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_info_rejects_path_traversal(cache_client):
    resp = cache_client.client.get(
        "/v1/cache/info?path=../etc",
        headers=_auth(),
    )
    assert resp.status_code == 403
