"""Hermetic coverage for mirror integrity tooling; no real HTTP is allowed."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import runpy
import signal
import sys
import types
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


drift = _load("mirror_drift_check", ROOT / "scripts" / "mirror_drift_check.py")
mirror = _load("mirror_to_r2_integrity", ROOT / "scripts" / "mirror_to_r2.py")


class Response:
    def __init__(self, status: int, body: bytes = b"", length: str | None = None):
        self.status = status
        self.body = body
        self.headers = Message()
        if length is not None:
            self.headers["Content-Length"] = length

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, amount: int | None = None) -> bytes:
        return self.body if amount is None else self.body[:amount]

    def getcode(self) -> int:
        return self.status


def _write_aliases(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "aliases.json"
    audio = tmp_path / "audio.json"
    main.write_text(
        json.dumps(
            {
                "good": {"hf_path": "org/good"},
                "mismatch": {"hf_path": "org/right"},
                "false": {"hf_path": "org/false"},
                "ignored": "comment",
                "bad": {"hf_path": "not-a-repo"},
            }
        )
    )
    audio.write_text(
        json.dumps(
            {
                "_comment": "metadata",
                "speech": {"hf_id": "audio/speech"},
                "wrong-key": {"hf_path": "audio/not-used"},
            }
        )
    )
    return main, audio


def test_both_alias_schemas_and_allow_list(tmp_path):
    main, audio = _write_aliases(tmp_path)
    specs = drift._load_aliases(main, audio)
    assert [(item.alias, item.hf_path, item.source) for item in specs] == [
        ("good", "org/good", "main"),
        ("mismatch", "org/right", "main"),
        ("false", "org/false", "main"),
        ("speech", "audio/speech", "audio"),
    ]
    files = [
        drift.HfFile(".gitattributes", 1, None),
        drift.HfFile("4bit/model.safetensors", 2, None),
        drift.HfFile("5bit/model.safetensors", 3, None),
        drift.HfFile("README.md", 4, None),
    ]
    assert [item.path for item in drift._required_files(files, "4bit/")] == [
        "4bit/model.safetensors"
    ]
    assert [item.path for item in drift._required_files(files, None)] == [
        "4bit/model.safetensors",
        "5bit/model.safetensors",
    ]
    assert drift._valid_repo_id("org/repo")
    assert not drift._valid_repo_id("org/repo/extra")


def test_invalid_alias_file_and_hf_payload(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        drift._load_aliases(bad, bad)
    monkeypatch.setattr(
        drift, "_model_info", lambda _repo: types.SimpleNamespace(siblings="bad")
    )
    monkeypatch.setattr(drift, "_throttle_hf", lambda: None)
    with pytest.raises(RuntimeError, match="invalid listing"):
        drift._hf_files("org/repo")


def test_hf_listing_filters_unsafe_and_invalid_metadata(monkeypatch):
    sha = "a" * 64
    siblings = [
        types.SimpleNamespace(
            rfilename="model.safetensors",
            size=9,
            lfs=types.SimpleNamespace(sha256=sha),
            blob_id="blob-a",
        ),
        types.SimpleNamespace(
            rfilename="README.md",
            size="unknown",
            lfs=types.SimpleNamespace(sha256="short"),
            blob_id=None,
        ),
        types.SimpleNamespace(rfilename="/absolute"),
        types.SimpleNamespace(rfilename="../escape"),
        object(),
    ]
    monkeypatch.setattr(
        drift,
        "_model_info",
        lambda _repo: types.SimpleNamespace(siblings=siblings, sha="revision"),
    )
    monkeypatch.setattr(drift, "_throttle_hf", lambda: None)
    assert drift._hf_files("org/a b") == [
        drift.HfFile("model.safetensors", 9, sha, "blob-a"),
        drift.HfFile("README.md", None, None),
    ]


def test_model_info_wrapper_and_hf_retry_paths(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    seen = []
    info = types.SimpleNamespace(siblings=[], sha="revision")
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda repo_id, **kwargs: seen.append((repo_id, kwargs)) or info,
    )
    assert drift._model_info("org/repo") is info
    assert seen == [("org/repo", {"files_metadata": True})]

    limited = HfHubHTTPError(
        "limited",
        response=httpx.Response(
            429,
            headers={"Retry-After": "2"},
            request=httpx.Request("GET", "https://huggingface.co"),
        ),
    )
    attempts = iter([limited, info])

    def flaky(_repo):
        value = next(attempts)
        if isinstance(value, BaseException):
            raise value
        return value

    sleeps = []
    monkeypatch.setattr(drift, "_model_info", flaky)
    monkeypatch.setattr(drift, "_throttle_hf", lambda: None)
    monkeypatch.setattr(drift.time, "sleep", sleeps.append)
    assert drift._hf_repo("org/repo") == drift.HfRepo("revision", [])
    assert sleeps == [2.0]

    denied = HfHubHTTPError(
        "denied",
        response=httpx.Response(
            404, request=httpx.Request("GET", "https://huggingface.co")
        ),
    )
    monkeypatch.setattr(
        drift, "_model_info", lambda _repo: (_ for _ in ()).throw(denied)
    )
    with pytest.raises(HfHubHTTPError, match="denied"):
        drift._hf_repo("org/repo")

    monkeypatch.setattr(
        drift, "_model_info", lambda _repo: (_ for _ in ()).throw(OSError("down"))
    )
    with pytest.raises(OSError, match="down"):
        drift._hf_repo("org/repo")


def test_hf_listing_gate_paces_requests(monkeypatch):
    moments = iter([10.0, 10.2])
    sleeps = []
    monkeypatch.setattr(drift.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(drift.time, "sleep", sleeps.append)
    monkeypatch.setattr(drift, "_hf_next_request", 0.0)
    drift._throttle_hf()
    drift._throttle_hf()
    assert sleeps == [pytest.approx(0.8)]
    assert drift._hf_next_request == 12.0


def test_every_finding_class_and_optional_sha(monkeypatch, tmp_path):
    sha = "a" * 64
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(
        json.dumps(
            {
                "good": {"hf_path": "org/good"},
                "mismatch": {"hf_path": "org/right"},
                "false": {"hf_path": "org/false"},
            }
        )
    )
    audio.write_text(json.dumps({"absent": {"hf_id": "org/absent"}}))
    files = {
        "org/good": [drift.HfFile("ok", 10, sha)],
        "org/right": [drift.HfFile("wrong-size", 11, None)],
        "org/false": [drift.HfFile("gone", 12, None)],
        "org/absent": [drift.HfFile("present", None, None)],
    }
    probes = {
        ("org/good", "ok"): drift.MirrorProbe(200, 10, None, None),
        ("org/right", "wrong-size"): drift.MirrorProbe(200, 99, None, None),
        ("org/false", "gone"): drift.MirrorProbe(404, None, None, None),
        ("org/absent", "present"): drift.MirrorProbe(200, None, None, None),
    }
    monkeypatch.setattr(
        drift, "_hf_repo", lambda repo: drift.HfRepo("rev", files[repo])
    )
    monkeypatch.setattr(
        drift, "_public_probe", lambda repo, item: probes[(repo, item.path)]
    )
    monkeypatch.setattr(drift, "_r2_metadata", lambda *_args: {"hf-sha256": "b" * 64})
    entries = {
        "good": {"hf_path": "org/good", "status": "mirrored"},
        "mismatch": {"hf_path": "org/wrong", "status": "mirrored"},
        "false": {"hf_path": "org/false", "status": "mirrored"},
    }
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [dict(alias=k, **v) for k, v in entries.items()],
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", object)
    reports = drift.audit(main, audio)
    kinds = {finding.kind for report in reports for finding in report.findings}
    assert kinds == {
        "content_mismatch",
        "hf_path_mismatch",
        "size_mismatch",
        "missing_file",
        "false_mirrored",
        "not_in_catalog",
    }
    assert (
        next(report for report in reports if report.alias == "absent").state
        == "findings"
    )


def test_audit_filters_workers_catalog_only_and_unknown(monkeypatch, tmp_path):
    main, audio = _write_aliases(tmp_path)
    entries = [
        {"alias": "good", "hf_path": "org/good", "status": "mirrored"},
        {"alias": "bucket/repo", "hf_path": "bucket/repo", "status": "mirrored"},
    ]
    monkeypatch.setattr(drift, "_catalog_entries", lambda: entries)
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(drift, "_hf_repo", lambda _repo: drift.HfRepo("rev", []))
    reports = drift.audit(main, audio, aliases={"good"}, workers=99)
    assert [report.alias for report in reports] == ["good"]
    assert reports[0].state == "ok"
    assert reports[0].sha_check == "skipped_no_credentials"
    all_reports = drift.audit(main, audio, only_used=False, workers=0)
    assert any(report.source == "catalog_only" for report in all_reports)
    used = drift.audit(main, audio, only_used=True)
    assert all(report.source != "catalog_only" for report in used)
    with pytest.raises(ValueError, match="unknown alias"):
        drift.audit(main, audio, aliases={"missing"})


def test_audit_deduplicates_repo_listings_and_mirror_probes(
    monkeypatch, tmp_path, capsys
):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(
        json.dumps(
            {
                "first": {"hf_path": "org/shared"},
                "second": {"hf_path": "org/shared"},
            }
        )
    )
    audio.write_text("{}")
    item = drift.HfFile("config.json", 2, None, "x" * 40)
    hf_calls = []
    mirror_calls = []

    def hf_repo(repo):
        hf_calls.append(repo)
        return drift.HfRepo("revision", [item])

    def public_probe(repo, selected):
        mirror_calls.append((repo, selected.path))
        return drift.MirrorProbe(200, 2, '"etag"', "x" * 40)

    monkeypatch.setattr(drift, "_hf_repo", hf_repo)
    monkeypatch.setattr(drift, "_public_probe", public_probe)
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [
            {"alias": alias, "hf_path": "org/shared", "status": "mirrored"}
            for alias in ("first", "second")
        ],
    )
    monkeypatch.setattr(drift, "PROGRESS_REPO_INTERVAL", 1)
    monkeypatch.setattr(drift, "PROGRESS_PROBE_INTERVAL", 1)
    progress = drift.AuditProgress()
    reports = drift.audit(main, audio, progress=progress)
    assert len(reports) == 2
    assert hf_calls == ["org/shared"]
    assert mirror_calls == [("org/shared", "config.json")]
    assert (progress.repos_done, progress.probes_done) == (1, 1)
    assert "reason=probe-batch" in capsys.readouterr().err


def test_optional_assets_and_in_flight_sync_severity(monkeypatch, tmp_path):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(
        json.dumps(
            {
                "optional": {"hf_path": "org/optional"},
                "partial": {"hf_path": "org/partial"},
                "recent": {"hf_path": "org/recent"},
                "stable": {"hf_path": "org/stable"},
            }
        )
    )
    audio.write_text("{}")
    files = {
        "org/optional": [drift.HfFile("README.md", 10, None, "a" * 40)],
        "org/partial": [drift.HfFile("config.json", 10, None, "b" * 40)],
        "org/recent": [drift.HfFile("tokenizer.json", 10, None, "c" * 40)],
        "org/stable": [drift.HfFile("preprocessor_config.json", 10, None, "d" * 40)],
    }
    now = datetime.now(timezone.utc).isoformat()
    entries = [
        {"alias": "optional", "hf_path": "org/optional", "status": "mirrored"},
        {"alias": "partial", "hf_path": "org/partial", "status": "partial"},
        {
            "alias": "recent",
            "hf_path": "org/recent",
            "status": "mirrored",
            "latest_uploaded": now,
        },
        {"alias": "stable", "hf_path": "org/stable", "status": "mirrored"},
    ]
    monkeypatch.setattr(
        drift, "_hf_repo", lambda repo: drift.HfRepo("revision", files[repo])
    )
    monkeypatch.setattr(
        drift,
        "_public_probe",
        lambda *_args: drift.MirrorProbe(404, None, None, None),
    )
    monkeypatch.setattr(drift, "_catalog_entries", lambda: entries)
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    reports = {report.alias: report for report in drift.audit(main, audio)}
    assert reports["optional"].findings[0].severity == "info"
    for alias in ("partial", "recent"):
        finding = reports[alias].findings[0]
        assert finding.severity == "warning"
        assert "sync in progress" in finding.detail
    assert reports["stable"].findings[0].severity == "error"


def test_hf_failure_is_reported_without_aborting(monkeypatch, tmp_path):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"gone": {"hf_path": "org/gone"}}))
    audio.write_text("{}")
    monkeypatch.setattr(
        drift, "_hf_repo", lambda _repo: (_ for _ in ()).throw(RuntimeError("gone"))
    )
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "gone", "hf_path": "org/gone", "status": "mirrored"}],
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    report = drift.audit(main, audio)[0]
    assert report.checked_files == 0
    assert report.findings[-1] == drift.Finding(
        "hf_unavailable", "error", detail="gone"
    )


def test_probe_size_fallback_timestamps_and_etags(monkeypatch):
    body = b"{}"
    monkeypatch.setattr(
        drift, "_request", lambda *_args, **_kwargs: Response(200, body)
    )
    probe = drift._public_probe(
        "org/repo", drift.HfFile("config.json", len(body), None, "x" * 40)
    )
    assert probe.size == len(body)
    now = datetime.now(timezone.utc)
    assert drift._recent_timestamp(now.timestamp(), now)
    assert drift._recent_timestamp(now.replace(tzinfo=None).isoformat(), now)
    assert not drift._recent_timestamp("invalid", now)
    assert drift._etag_sha256(f'W/"{"a" * 64}"') == "a" * 64
    assert drift._etag_sha256('"not-a-sha"') is None


def test_catalog_shapes(monkeypatch):
    monkeypatch.setattr(
        drift, "_get_json", lambda _url: {"models": [{"alias": "a"}, "bad"]}
    )
    assert drift._catalog_entries() == [{"alias": "a"}]
    monkeypatch.setattr(drift, "_get_json", lambda _url: [{"alias": "b"}])
    assert drift._catalog_entries() == [{"alias": "b"}]
    monkeypatch.setattr(drift, "_get_json", lambda _url: {})
    with pytest.raises(RuntimeError, match="no models list"):
        drift._catalog_entries()


def test_unchanged_mirror_tuple_does_not_hide_changed_hf_body(monkeypatch, tmp_path):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"same": {"hf_path": "org/same"}}))
    audio.write_text("{}")
    entry = {
        "alias": "same",
        "hf_path": "org/same",
        "status": "mirrored",
        "total_bytes": 2,
        "file_count": 1,
        "latest_uploaded": "2020-01-01T00:00:00Z",
    }
    monkeypatch.setattr(drift, "_catalog_entries", lambda: [entry])
    # Simulate the deleted fast path's matching catalog tuple. The current HF
    # revision and config body have changed even though the mirror has not.
    monkeypatch.setattr(
        drift,
        "_snapshot_aliases",
        lambda: {"same": dict(entry)},
        raising=False,
    )
    fresh = b'{"model_type":"fresh"}'
    stale = b'{"model_type":"stale"}'
    assert len(fresh) == len(stale)
    fresh_oid = hashlib.sha1(
        f"blob {len(fresh)}\0".encode() + fresh, usedforsecurity=False
    ).hexdigest()
    monkeypatch.setattr(
        drift,
        "_hf_repo",
        lambda _repo: drift.HfRepo(
            "hf-revision-after-change",
            [drift.HfFile("config.json", len(fresh), None, fresh_oid)],
        ),
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    mirror_calls = []

    def stale_probe(repo, item):
        mirror_calls.append((repo, item.path))
        return drift.MirrorProbe(
            200, len(stale), '"not-a-sha256"', drift._git_blob_oid(stale)
        )

    monkeypatch.setattr(drift, "_public_probe", stale_probe)
    report = drift.audit(main, audio)[0]
    assert mirror_calls == [("org/same", "config.json")]
    assert report.checked_files == 1
    assert any(
        finding.kind == "content_mismatch" and finding.path == "config.json"
        for finding in report.findings
    )


def test_lfs_missing_r2_checksum_metadata_is_content_mismatch(monkeypatch, tmp_path):
    sha = "a" * 64
    item = drift.HfFile("model.safetensors", 1024, sha)
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"stale": {"hf_path": "org/stale"}}))
    audio.write_text("{}")
    monkeypatch.setattr(
        drift, "_hf_repo", lambda _repo: drift.HfRepo("new-revision", [item])
    )
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "stale", "hf_path": "org/stale", "status": "mirrored"}],
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", object)
    monkeypatch.setattr(
        drift,
        "_public_probe",
        lambda *_args: drift.MirrorProbe(200, 1024, '"multipart-etag-2"', None),
    )
    monkeypatch.setattr(drift, "_r2_metadata", lambda *_args: {})
    report = drift.audit(main, audio)[0]
    assert report.sha_check == "checked"
    assert any(
        finding.kind == "content_mismatch"
        and finding.path == "model.safetensors"
        and "no checksum metadata" in (finding.detail or "")
        for finding in report.findings
    )


def test_cache_buster_reaches_redirect_destination_and_cached_404_is_avoided(
    monkeypatch,
):
    handler = drift._FinalUrlRedirectHandler()
    initial = urllib.request.Request(
        drift._cache_busted("https://models.example/repo/file"),
        method="HEAD",
        headers={"Cache-Control": "no-cache"},
    )
    redirected = handler.redirect_request(
        initial,
        io.BytesIO(),
        302,
        "Found",
        Message(),
        "https://dl.example/repo/file",
    )
    assert redirected is not None
    parsed = urllib.parse.urlsplit(redirected.full_url)
    assert parsed.netloc == "dl.example"
    assert urllib.parse.parse_qs(parsed.query)["mirror_drift"]
    assert redirected.get_header("Cache-control") == "no-cache"
    assert redirected.get_method() == "HEAD"

    class RedirectingOpener:
        def open(self, request, timeout):
            assert timeout == 30.0
            final = handler.redirect_request(
                request,
                io.BytesIO(),
                302,
                "Found",
                Message(),
                "https://dl.example/repo/file",
            )
            assert final is not None
            # Without a final-url token this fixture represents the cached 404.
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(final.full_url).query)
            return Response(200 if "mirror_drift" in query else 404, length="7")

    monkeypatch.setattr(drift, "_OPENER", RedirectingOpener())
    assert drift._public_head("org/repo", drift.HfFile("file", 7, None)) == (200, 7)


def test_same_size_stale_non_lfs_body_is_content_mismatch(monkeypatch, tmp_path):
    expected = b'{"model_type":"fresh"}'
    stale = b'{"model_type":"stale"}'
    assert len(expected) == len(stale)
    expected_oid = hashlib.sha1(
        f"blob {len(expected)}\0".encode() + expected, usedforsecurity=False
    ).hexdigest()
    item = drift.HfFile("config.json", len(expected), None, expected_oid)
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"stale": {"hf_path": "org/stale"}}))
    audio.write_text("{}")
    monkeypatch.setattr(drift, "_hf_repo", lambda _repo: drift.HfRepo("rev", [item]))
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "stale", "hf_path": "org/stale", "status": "mirrored"}],
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift,
        "_request",
        lambda *_args, **_kwargs: Response(200, stale, str(len(stale))),
    )
    report = drift.audit(main, audio)[0]
    assert any(finding.kind == "content_mismatch" for finding in report.findings)


def test_request_retries_and_http_errors(monkeypatch):
    attempts = []

    class Flaky:
        def open(self, request, timeout):
            attempts.append((request, timeout))
            if len(attempts) < 5:
                raise urllib.error.URLError("temporary")
            return Response(200, b"{}")

    monkeypatch.setattr(drift, "_OPENER", Flaky())
    monkeypatch.setattr(drift.time, "sleep", lambda _seconds: None)
    assert drift._get_json("https://example/api") == {}
    assert len(attempts) == 5
    assert all(call[0].get_header("Cache-control") == "no-cache" for call in attempts)

    error = urllib.error.HTTPError("https://example", 404, "missing", Message(), None)
    monkeypatch.setattr(
        drift,
        "_OPENER",
        types.SimpleNamespace(open=lambda *_a, **_k: (_ for _ in ()).throw(error)),
    )
    with pytest.raises(RuntimeError, match="HTTP 404"):
        drift._get_json("https://example")


def test_request_exhaustion(monkeypatch):
    error = TimeoutError("late")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(drift, "_OPENER", types.SimpleNamespace(open=fail))
    monkeypatch.setattr(drift.time, "sleep", lambda _seconds: None)
    drift._REQUEST_CONTEXT.kind = "mirror"
    try:
        with pytest.raises(TimeoutError, match="late"):
            drift._request("https://example")
    finally:
        drift._REQUEST_CONTEXT.kind = None
    assert drift._profile_counts["mirror_retry_seconds"] > 0

    transient = urllib.error.HTTPError("https://example", 503, "late", Message(), None)
    monkeypatch.setattr(
        drift,
        "_OPENER",
        types.SimpleNamespace(open=lambda *_a, **_k: (_ for _ in ()).throw(transient)),
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        drift._request("https://example")
    assert raised.value.code == 503

    headers = Message()
    headers["Retry-After"] = "60"
    limited = urllib.error.HTTPError("https://example", 429, "slow down", headers, None)
    sleeps = []
    monkeypatch.setattr(
        drift,
        "_OPENER",
        types.SimpleNamespace(open=lambda *_a, **_k: (_ for _ in ()).throw(limited)),
    )
    monkeypatch.setattr(drift.time, "sleep", sleeps.append)
    with pytest.raises(urllib.error.HTTPError):
        drift._request("https://example")
    assert sleeps == [30.0, 30.0, 30.0, 30.0]


def test_r2_client_and_metadata(monkeypatch):
    class Session:
        def __init__(self, credentials):
            self.credentials = credentials

        def get_credentials(self):
            return self.credentials

        def client(self, service, endpoint_url):
            return service, endpoint_url

    monkeypatch.setitem(
        sys.modules, "boto3", types.SimpleNamespace(Session=lambda: Session(None))
    )
    assert drift._maybe_r2_client() is None
    monkeypatch.setitem(
        sys.modules, "boto3", types.SimpleNamespace(Session=lambda: Session(object()))
    )
    monkeypatch.setenv("RAPID_MLX_R2_ENDPOINT_URL", "https://r2.example")
    assert drift._maybe_r2_client() == ("s3", "https://r2.example")

    assert drift._r2_metadata(
        types.SimpleNamespace(
            head_object=lambda **_kwargs: {"Metadata": {"hf-sha256": "x"}}
        ),
        "org/repo",
        "file",
    ) == {"hf-sha256": "x"}
    assert (
        drift._r2_metadata(
            types.SimpleNamespace(
                head_object=lambda **_kwargs: {"Metadata": "invalid"}
            ),
            "org/repo",
            "file",
        )
        == {}
    )

    class MissingError(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    def missing(**_kwargs):
        raise MissingError

    assert (
        drift._r2_metadata(
            types.SimpleNamespace(head_object=missing), "org/repo", "file"
        )
        is None
    )
    with pytest.raises(RuntimeError, match="boom"):
        drift._r2_metadata(
            types.SimpleNamespace(
                head_object=lambda **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("boom")
                )
            ),
            "org/repo",
            "file",
        )


def test_no_boto3_is_clean_skip(monkeypatch):
    real_import = __import__

    def import_without_boto(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_boto)
    assert drift._maybe_r2_client() is None


def test_main_json_text_exit_codes_and_failures(monkeypatch, capsys):
    clean = drift.AliasReport("a", "main", "org/a", True, "org/a", "mirrored")
    dirty = drift.AliasReport(
        "b",
        "audio",
        "org/b",
        False,
        None,
        None,
        findings=[drift.Finding("not_in_catalog", "error")],
    )
    monkeypatch.setattr(drift, "audit", lambda *_args, **_kwargs: [clean, dirty])
    assert drift.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert payload["summary"] == {"not_in_catalog": 1, "ok": 1}
    assert payload["aliases"][0]["state"] == "ok"
    assert drift.main(["--fail-on", "never"]) == 0
    text = capsys.readouterr().out
    assert "ALIAS" in text and "SUMMARY aliases=2" in text
    monkeypatch.setattr(
        drift,
        "audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert drift.main([]) == 2
    assert "audit failed: bad" in capsys.readouterr().err


def test_periodic_progress_and_sigterm_partial_report(monkeypatch, capsys):
    monkeypatch.setattr(drift, "PROGRESS_INTERVAL_SECONDS", 0.001)
    progress = drift.AuditProgress()
    progress.start_periodic()
    assert not progress._stop.wait(0.01)
    progress.stop_periodic()
    assert "reason=timer" in capsys.readouterr().err

    partial = drift.AliasReport(
        "partial-alias",
        "main",
        "org/partial",
        True,
        "org/partial",
        "mirrored",
        checked_files=1,
        findings=[drift.Finding("missing_file", "error", "config.json")],
    )

    def interrupt(*_args, progress, **_kwargs):
        progress.reports = [partial]
        progress.repos_total = 1
        progress.repos_done = 1
        progress.probes_total = 2
        progress.probes_done = 1
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(drift, "audit", interrupt)
    with pytest.raises(SystemExit) as raised:
        drift.main([])
    assert raised.value.code == 124
    error = capsys.readouterr().err
    assert "reason=sigterm" in error
    assert "PARTIAL REPORT" in error
    assert "partial-alias" in error
    assert "missing_file config.json" in error


def test_script_entrypoint_handles_missing_alias_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mirror_drift_check.py",
            "--aliases-path",
            str(tmp_path / "missing.json"),
            "--audio-aliases-path",
            str(tmp_path / "also-missing.json"),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(
            str(ROOT / "scripts" / "mirror_drift_check.py"), run_name="__main__"
        )
    assert raised.value.code == 2


def test_mirror_uploader_public_404_is_advisory(monkeypatch, capsys):
    item = mirror.FileMeta("file.bin", 7, "org/repo/file.bin", None)
    monkeypatch.setattr(mirror, "_hf_files", lambda _repo: [item])
    monkeypatch.setattr(mirror, "_r2_client", lambda *_args: object())
    monkeypatch.setattr(mirror, "_r2_head_size", lambda *_args: 7)
    monkeypatch.setattr(mirror, "_http_range_get_status", lambda _url: 404)
    assert mirror.mirror_repo("org/repo", verify_only=True) == 0
    output = capsys.readouterr().out
    assert "ADVISORY" in output
    assert "signed R2 HEAD is authoritative and passed" in output
    assert "1 public advisories" in output


def test_mirror_uploader_zero_byte_head_and_verify_failure(monkeypatch, capsys):
    item = mirror.FileMeta("empty", 0, "org/repo/empty", None)
    monkeypatch.setattr(mirror, "_hf_files", lambda _repo: [item])
    monkeypatch.setattr(mirror, "_r2_client", lambda *_args: object())
    monkeypatch.setattr(mirror, "_r2_head_size", lambda *_args: 0)
    head_calls = []
    monkeypatch.setattr(
        mirror, "_http_head_status", lambda url: head_calls.append(url) or 200
    )
    assert mirror.mirror_repo("org/repo", verify_only=True) == 0
    assert head_calls

    monkeypatch.setattr(mirror, "_r2_head_size", lambda *_args: 1)
    assert mirror.mirror_repo("org/repo", verify_only=True) == 3
    assert "r2-size:1!=0" in capsys.readouterr().err


def test_mirror_uploader_prints_cdn_reminder_after_upload(
    monkeypatch, capsys, tmp_path
):
    item = mirror.FileMeta("file.bin", 7, "org/repo/file.bin", None)
    local = tmp_path / "download.bin"
    local.write_bytes(b"content")
    monkeypatch.setattr(mirror, "_hf_files", lambda _repo: [item])
    monkeypatch.setattr(mirror, "_r2_client", lambda *_args: object())
    monkeypatch.setattr(mirror, "_r2_head", lambda *_args: None)
    monkeypatch.setattr(mirror, "_download_one_hf", lambda *_args: local)
    monkeypatch.setattr(mirror, "_upload_one", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mirror, "_r2_head_size", lambda *_args: 7)
    monkeypatch.setattr(mirror, "_http_range_get_status", lambda _url: 200)
    assert mirror.mirror_repo("org/repo", tmp_dir=tmp_path) == 0
    assert "max-age=3600" in capsys.readouterr().out


def test_uploader_redirect_and_http_helpers(monkeypatch):
    handler = mirror._FinalUrlRedirectHandler()
    request = urllib.request.Request("https://models.example/file", method="HEAD")
    redirected = handler.redirect_request(
        request, io.BytesIO(), 302, "Found", Message(), "https://dl.example/file"
    )
    assert redirected is not None
    assert "mirror_verify=" in redirected.full_url
    assert redirected.get_header("Cache-control") == "no-cache"
    assert redirected.get_method() == "HEAD"

    class Opener:
        def __init__(self, response=None, error=None):
            self.response = response
            self.error = error
            self.requests = []

        def open(self, req, timeout):
            self.requests.append((req, timeout))
            if self.error:
                raise self.error
            return self.response

    opener = Opener(Response(206, b"x"))
    monkeypatch.setattr(mirror, "_PUBLIC_OPENER", opener)
    assert mirror._http_range_get_status("https://models.example/file") == 200
    req = opener.requests[0][0]
    assert "mirror_verify=" in req.full_url
    assert req.get_header("Range") == "bytes=0-0"
    assert req.get_header("Cache-control") == "no-cache"

    monkeypatch.setattr(mirror, "_PUBLIC_OPENER", Opener(Response(200)))
    assert mirror._http_head_status("https://models.example/empty") == 200
    http_error = urllib.error.HTTPError(
        "https://example", 404, "missing", Message(), None
    )
    monkeypatch.setattr(mirror, "_PUBLIC_OPENER", Opener(error=http_error))
    assert mirror._http_head_status("https://example") == 404
    assert mirror._http_range_get_status("https://example") == 404
    monkeypatch.setattr(mirror, "_PUBLIC_OPENER", Opener(error=TimeoutError()))
    assert mirror._http_head_status("https://example") == 0
    assert mirror._http_range_get_status("https://example") == 0
