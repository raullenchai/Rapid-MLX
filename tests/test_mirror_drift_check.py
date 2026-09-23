"""Hermetic coverage for mirror integrity tooling; no real HTTP is allowed."""

from __future__ import annotations

import importlib.util
import io
import json
import runpy
import sys
import types
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

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
        "README.md",
    ]
    assert drift._valid_repo_id("org/repo")
    assert not drift._valid_repo_id("org/repo/extra")


def test_invalid_alias_file_and_hf_payload(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        drift._load_aliases(bad, bad)
    monkeypatch.setattr(drift, "_get_json", lambda _url: [])
    with pytest.raises(RuntimeError, match="invalid listing"):
        drift._hf_files("org/repo")


def test_hf_listing_filters_unsafe_and_invalid_metadata(monkeypatch):
    sha = "a" * 64
    monkeypatch.setattr(
        drift,
        "_get_json",
        lambda _url: {
            "siblings": [
                {"rfilename": "model.safetensors", "size": 9, "lfs": {"sha256": sha}},
                {
                    "rfilename": "README.md",
                    "size": "unknown",
                    "lfs": {"sha256": "short"},
                },
                {"rfilename": "/absolute"},
                {"rfilename": "../escape"},
                "not-an-object",
            ]
        },
    )
    monkeypatch.setattr(drift, "_throttle_hf", lambda: None)
    assert drift._hf_files("org/a b") == [
        drift.HfFile("model.safetensors", 9, sha),
        drift.HfFile("README.md", None, None),
    ]


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


def test_every_finding_class_and_optional_sha(monkeypatch):
    sha = "a" * 64
    specs = {
        "good": drift.AliasSpec("good", "org/good", None, "main"),
        "mismatch": drift.AliasSpec("mismatch", "org/right", None, "main"),
        "false": drift.AliasSpec("false", "org/false", None, "main"),
        "absent": drift.AliasSpec("absent", "org/absent", None, "audio"),
    }
    files = {
        "org/good": [drift.HfFile("ok", 10, sha)],
        "org/right": [drift.HfFile("wrong-size", 11, None)],
        "org/false": [drift.HfFile("gone", 12, None)],
        "org/absent": [drift.HfFile("present", None, None)],
    }
    heads = {
        ("org/good", "ok"): (200, 10),
        ("org/right", "wrong-size"): (200, 99),
        ("org/false", "gone"): (404, None),
        ("org/absent", "present"): (200, None),
    }
    monkeypatch.setattr(drift, "_hf_files", lambda repo: files[repo])
    monkeypatch.setattr(
        drift, "_public_head", lambda repo, item: heads[(repo, item.path)]
    )
    monkeypatch.setattr(drift, "_r2_metadata", lambda *_args: {"hf-sha256": "b" * 64})
    entries = {
        "good": {"hf_path": "org/good", "status": "mirrored"},
        "mismatch": {"hf_path": "org/wrong", "status": "mirrored"},
        "false": {"hf_path": "org/false", "status": "mirrored"},
    }
    reports = {
        name: drift._audit_alias(spec, entries.get(name), object())
        for name, spec in specs.items()
    }
    kinds = {finding.kind for report in reports.values() for finding in report.findings}
    assert kinds == {
        "sha_mismatch",
        "hf_path_mismatch",
        "size_mismatch",
        "missing_file",
        "false_mirrored",
        "not_in_catalog",
    }
    assert reports["absent"].state == "findings"


def test_audit_filters_workers_catalog_only_and_unknown(monkeypatch, tmp_path):
    main, audio = _write_aliases(tmp_path)
    entries = [
        {"alias": "good", "hf_path": "org/good", "status": "mirrored"},
        {"alias": "bucket/repo", "hf_path": "bucket/repo", "status": "mirrored"},
    ]
    monkeypatch.setattr(drift, "_catalog_entries", lambda: entries)
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(drift, "_hf_files", lambda _repo: [])
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
    with pytest.raises(TimeoutError, match="late"):
        drift._request("https://example")

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
