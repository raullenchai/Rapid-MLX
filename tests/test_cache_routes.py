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
    # F-180: pin loopback as the TestClient origin so the
    # ``verify_internal_admin`` loopback escape hatch resolves true. The
    # default ``("testclient", 50000)`` host is NOT loopback and would 403
    # every test once the cache router moved onto the admin gate.
    yield SimpleNamespace(
        client=TestClient(app, client=("127.0.0.1", 50000)),
        sandbox=sandbox,
    )

    reset_config()


def _auth() -> dict:
    """Auth headers for the cache router (F-180).

    Bearer alone is no longer sufficient — the cache router now sits behind
    ``verify_internal_admin``, which requires ``X-Rapid-MLX-Internal: true``
    AND (because ``--api-key`` is configured in the fixture) a valid bearer.
    """
    return {
        "Authorization": "Bearer test-secret",
        "X-Rapid-MLX-Internal": "true",
    }


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
    """A symlink whose realpath leaves the sandbox is rejected.

    ``os.path.realpath`` follows the link to ``outside_dir``, and the
    subsequent ``commonpath`` check sees the result is no longer a
    descendant of the sandbox root. Without realpath the literal path
    ``sandbox/escape/anything`` would look safe — this is the case
    that justifies the realpath step.
    """
    sandbox.mkdir(parents=True, exist_ok=True)
    outside = sandbox.parent / "outside_dir"
    outside.mkdir()
    link = sandbox / "escape"
    link.symlink_to(outside)

    with pytest.raises(InvalidExportPathError, match="outside sandbox"):
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


def test_write_manifest_failed_rename_preserves_prior_manifest(tmp_path, monkeypatch):
    """A crash mid-rename must not corrupt the prior manifest.

    Atomic write idiom: write tmp → fsync → ``os.replace``. If ``replace``
    fails (here we monkeypatch it to ValueError), the prior manifest.json
    must be untouched and the tmp file cleaned up. Without this the
    next ``read_manifest`` would 400 against a truncated file even
    though a valid one existed before.
    """
    original = Manifest(model_id="qwen3.5-9b-4bit", entries=18)
    write_manifest(tmp_path, original)
    assert (tmp_path / "manifest.json").is_file()

    import vllm_mlx.cache.protocol as protocol_mod

    def _boom(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(protocol_mod.os, "replace", _boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        write_manifest(tmp_path, Manifest(model_id="will-not-land", entries=99))

    # Prior manifest intact: same fields, no truncation.
    recovered = read_manifest(tmp_path)
    assert recovered.model_id == "qwen3.5-9b-4bit"
    assert recovered.entries == 18

    # No temp file left behind.
    leaked = list(tmp_path.glob(".manifest-*.json"))
    assert leaked == [], f"leaked temp files: {leaked}"


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


def test_manifest_from_dict_rejects_wrong_type(tmp_path):
    """A known field with the wrong JSON type → MalformedManifestError.

    Codex round-3 BLOCKING: ``"entries": "not-an-int"`` previously
    constructed the dataclass blindly, so a peer could serve a manifest
    that violated its own advertised schema and the route would return
    200 anyway. Now each known field's value is checked against its
    expected Python type at read time.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "entries": "not-an-int"})
    )
    with pytest.raises(MalformedManifestError, match="entries"):
        read_manifest(tmp_path)


def test_manifest_from_dict_rejects_bool_for_int_field(tmp_path):
    """``isinstance(True, int)`` is True in Python — but JSON ``true`` is
    clearly not the integer 1. The strict check rejects this."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "entries": True})
    )
    with pytest.raises(MalformedManifestError, match="entries"):
        read_manifest(tmp_path)


def test_manifest_from_dict_rejects_string_for_bool_field(tmp_path):
    """``"paged_cache": "yes"`` is structurally wrong even if intuitive."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"protocol_version": "1", "paged_cache": "yes"})
    )
    with pytest.raises(MalformedManifestError, match="paged_cache"):
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
    """F-180: no ``X-Rapid-MLX-Internal`` header → 403 on every route.

    Pre-F-180 these routes were on ``verify_api_key`` which returned 401 on
    a missing bearer. Post-F-180 the header gate fires first and surfaces
    as 403 even before the bearer check (auth-stage signals are distinct
    so monitoring rules can grep them apart)."""
    client = cache_client.client
    if method == "post":
        resp = client.post(path, json=body)
    else:
        resp = client.get(path)
    assert resp.status_code == 403, resp.text
    # The 403 detail must call out the missing header so an operator can
    # diagnose the misconfiguration without grepping source.
    body_json = resp.json()
    detail = body_json.get("detail") or body_json.get("error", {}).get("message", "")
    if isinstance(detail, dict):
        detail = detail.get("message", "")
    assert "X-Rapid-MLX-Internal" in detail, body_json


def test_info_requires_auth_even_with_valid_manifest(cache_client):
    """An unauthenticated ``GET /v1/cache/info`` against a path with a
    valid manifest must still 403 (F-180), not 200.

    Codex round-3 NIT (original): with auth-fires-before-handler the empty
    default path returns the same status as a missing-handler-output. With
    a real manifest in place, a bypassed auth dependency would surface as
    a 200 — this test catches that exact regression for the F-180 gate.
    """
    _write_export_root(
        cache_client.sandbox,
        "valid",
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="x", entries=1),
    )
    resp = cache_client.client.get("/v1/cache/info?path=valid")
    assert resp.status_code == 403


def test_routes_reject_wrong_bearer(cache_client):
    """With the internal header present but a wrong bearer, the api-key gate
    fires (because ``cfg.api_key`` is set) and returns 401. Pre-F-180 this
    test sent only the bearer; post-F-180 we must also send the header so
    the request gets past the header gate to the bearer check."""
    resp = cache_client.client.post(
        "/v1/cache/export",
        json={},
        headers={
            "Authorization": "Bearer wrong",
            "X-Rapid-MLX-Internal": "true",
        },
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /v1/cache/export — stub returns 501 after passing the sandbox check
# ---------------------------------------------------------------------------


def test_export_default_destination_returns_501(cache_client):
    """No destination → uses sandbox root, then 501 with the sanitized
    F-180 envelope (no resolved-path leak, no tracking-URL leak)."""
    resp = cache_client.client.post("/v1/cache/export", json={}, headers=_auth())
    assert resp.status_code == 501
    detail = resp.json()["detail"]
    # F-180: minimal envelope, no operator-controlled fields.
    assert detail == {
        "error": {
            "message": "not implemented",
            "type": "not_implemented_error",
            "code": None,
        }
    }
    # The resolved destination must NOT appear anywhere in the body.
    sandbox_real = str(Path(cache_client.sandbox).resolve())
    assert sandbox_real not in resp.text
    assert "github.com" not in resp.text


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


def test_info_400_detail_does_not_leak_resolved_path(cache_client):
    """The 400 body must not include the server's resolved cache root.

    Codex round-3 NIT: leaking ``/Users/raullen/.cache/rapid-mlx/...`` to
    any bearer-token holder is unnecessary information disclosure.
    """
    bad = cache_client.sandbox / "leak-probe"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ syntax error here")
    resp = cache_client.client.get(
        "/v1/cache/info?path=leak-probe",
        headers=_auth(),
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert str(cache_client.sandbox) not in detail
    assert "/" not in detail or "JSON" in detail  # may mention syntax but no path


def test_info_404_detail_does_not_leak_resolved_path(cache_client):
    """The 404 body must not include the server's resolved cache root."""
    (cache_client.sandbox / "no-such").mkdir(parents=True)
    resp = cache_client.client.get(
        "/v1/cache/info?path=no-such",
        headers=_auth(),
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert str(cache_client.sandbox) not in detail
    assert detail == "no manifest.json at the requested cache path"


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
    """All checks pass → 501 with the sanitized F-180 envelope (no resolved
    source path, no parsed-manifest echo, no tracking-URL leak)."""
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
    # F-180: minimal envelope, no operator-controlled fields.
    assert detail == {
        "error": {
            "message": "not implemented",
            "type": "not_implemented_error",
            "code": None,
        }
    }
    # Belt + braces: the resolved source dir and the model id must not surface.
    sandbox_real = str(Path(cache_client.sandbox).resolve())
    assert sandbox_real not in resp.text
    assert "qwen3.5-9b-4bit" not in resp.text
    assert "github.com" not in resp.text


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
