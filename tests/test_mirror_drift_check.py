"""Hermetic coverage for mirror integrity tooling; no real HTTP is allowed."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import runpy
import selectors
import signal
import subprocess
import sys
import textwrap
import threading
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


drift = _load("scripts.mirror_drift_check", ROOT / "scripts" / "mirror_drift_check.py")
mirror = _load("scripts.mirror_to_r2_integrity", ROOT / "scripts" / "mirror_to_r2.py")


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


def _write_unmirrored(tmp_path, entries):
    path = tmp_path / "mirror_unmirrored.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": entries}))
    return path


def test_unmirrored_loader_accepts_valid_file(tmp_path):
    main, audio = _write_aliases(tmp_path)
    path = _write_unmirrored(
        tmp_path,
        [
            {
                "hf_path": "org/good",
                "reason": "unused",
                "since": "2026-09-24",
            },
            {
                "hf_path": "audio/speech",
                "reason": "served upstream",
                "since": "2026-09-23",
            },
        ],
    )

    entries = drift.load_unmirrored(path, main, audio)

    assert entries["org/good"].reason == "unused"
    assert entries["audio/speech"].since == "2026-09-23"


def test_unmirrored_loader_rejects_stale_hf_path(tmp_path):
    main, audio = _write_aliases(tmp_path)
    path = _write_unmirrored(
        tmp_path,
        [{"hf_path": "org/stale", "reason": "unused", "since": "2026-09-24"}],
    )
    with pytest.raises(ValueError, match="not present in the alias catalogs"):
        drift.load_unmirrored(path, main, audio)


def test_unmirrored_loader_rejects_duplicate(tmp_path):
    main, audio = _write_aliases(tmp_path)
    entry = {"hf_path": "org/good", "reason": "unused", "since": "2026-09-24"}
    path = _write_unmirrored(tmp_path, [entry, entry])
    with pytest.raises(ValueError, match="duplicate hf_path"):
        drift.load_unmirrored(path, main, audio)


def test_unmirrored_loader_rejects_bad_date(tmp_path):
    main, audio = _write_aliases(tmp_path)
    path = _write_unmirrored(
        tmp_path,
        [{"hf_path": "org/good", "reason": "unused", "since": "09/24/2026"}],
    )
    with pytest.raises(ValueError, match="ISO date"):
        drift.load_unmirrored(path, main, audio)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "exactly the keys"),
        ({"schema_version": 2, "entries": []}, "schema_version must be 1"),
        ({"schema_version": 1, "entries": {}}, "entries must be a JSON array"),
        (
            {
                "schema_version": 1,
                "entries": [
                    {
                        "hf_path": "org/good",
                        "reason": "unused",
                        "since": "2026-09-24",
                        "extra": True,
                    }
                ],
            },
            "exactly the keys hf_path",
        ),
        (
            {
                "schema_version": 1,
                "entries": [
                    {"hf_path": "org/good", "reason": " ", "since": "2026-09-24"}
                ],
            },
            "non-empty string",
        ),
        (
            {
                "schema_version": 1,
                "entries": [
                    {"hf_path": "org/good", "reason": "unused", "since": "20260924"}
                ],
            },
            "ISO date",
        ),
    ],
)
def test_unmirrored_loader_rejects_invalid_shapes(tmp_path, payload, message):
    main, audio = _write_aliases(tmp_path)
    path = tmp_path / "mirror_unmirrored.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        drift.load_unmirrored(path, main, audio)


def test_unmirrored_loader_rejects_non_object_alias_catalog(tmp_path):
    main = tmp_path / "aliases.json"
    audio = tmp_path / "audio.json"
    registry = _write_unmirrored(tmp_path, [])
    main.write_text("[]")
    audio.write_text("{}")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        drift.load_unmirrored(registry, main, audio)


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


def test_intentionally_unmirrored_skips_mirror_but_checks_hf(
    monkeypatch, tmp_path, capsys
):
    main = tmp_path / "aliases.json"
    audio = tmp_path / "audio.json"
    main.write_text(
        json.dumps(
            {
                "retired": {"hf_path": "org/retired"},
                "gone": {"hf_path": "org/gone"},
            }
        )
    )
    audio.write_text("{}")
    unmirrored = _write_unmirrored(
        tmp_path,
        [
            {
                "hf_path": repo,
                "reason": "unused",
                "since": "2026-09-24",
            }
            for repo in ("org/retired", "org/gone")
        ],
    )
    seen_hf = []

    def hf_repo(repo):
        seen_hf.append(repo)
        if repo == "org/gone":
            raise RuntimeError("HTTP 401")
        return drift.HfRepo("revision", [drift.HfFile("config.json", 2, None)])

    monkeypatch.setattr(drift, "_hf_repo", hf_repo)
    monkeypatch.setattr(drift, "_catalog_entries", lambda: [])
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift,
        "_public_probe",
        lambda *_args: pytest.fail("intentional repos must not probe the mirror"),
    )

    reports = {
        report.alias: report
        for report in drift.audit(main, audio, unmirrored_path=unmirrored)
    }

    assert set(seen_hf) == {"org/retired", "org/gone"}
    assert reports["retired"].state == "unmirrored (intentional)"
    assert reports["retired"].findings == []
    assert reports["gone"].state == "findings"
    assert reports["gone"].findings == [
        drift.Finding("hf_unavailable", "error", detail="HTTP 401")
    ]
    assert drift._summary_counts(list(reports.values())) == {
        "hf_unavailable": 1,
        "unmirrored_intentional": 2,
    }
    assert drift._fails(list(reports.values()), "error")
    rendered = drift._render_text(list(reports.values()))
    assert "unmirrored (intentional)" in rendered
    assert "reason=unused" in rendered

    monkeypatch.setattr(
        drift, "audit", lambda *_args, **_kwargs: list(reports.values())
    )
    assert drift.main([]) == 1
    summary = capsys.readouterr().out
    assert "gone" in summary
    assert "findings" in summary
    assert "reason=unused" in summary


def test_alternate_catalog_without_registry_does_not_skip_matching_repo(
    monkeypatch, tmp_path
):
    main = tmp_path / "aliases.json"
    audio = tmp_path / "audio.json"
    repo = "lmstudio-community/MiniMax-M2.5-MLX-4bit"
    main.write_text(json.dumps({"custom": {"hf_path": repo}}))
    audio.write_text("{}")
    item = drift.HfFile("config.json", 2, None)
    probes = []

    monkeypatch.setattr(drift, "_hf_repo", lambda _repo: drift.HfRepo("rev", [item]))
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "custom", "hf_path": repo, "status": "mirrored"}],
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift,
        "_public_probe",
        lambda repo_id, file: (
            probes.append((repo_id, file.path))
            or drift.MirrorProbe(200, file.size, None, None)
        ),
    )

    report = drift.audit(main, audio)[0]

    assert probes == [(repo, "config.json")]
    assert report.state == "ok"
    assert not report.intentionally_unmirrored


def test_default_catalogs_select_default_unmirrored_registry(monkeypatch):
    loaded = []
    monkeypatch.setattr(drift, "_load_aliases", lambda *_args: [])
    monkeypatch.setattr(
        drift,
        "load_unmirrored",
        lambda *args: loaded.append(args) or {},
    )
    monkeypatch.setattr(drift, "_catalog_entries", lambda: [])
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)

    assert drift.audit(drift.ALIASES_PATH, drift.AUDIO_ALIASES_PATH) == []
    assert loaded == [
        (drift.UNMIRRORED_PATH, drift.ALIASES_PATH, drift.AUDIO_ALIASES_PATH)
    ]


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


def test_large_non_lfs_body_without_blob_oid_is_unverified_not_mismatch():
    size = drift.SMALL_NON_LFS_MAX_BYTES + 1
    item = drift.HfFile("vocab.json", size, None, "a" * 40)
    report = drift.AliasReport(
        "large", "main", "org/large", True, "org/large", "mirrored"
    )

    drift._apply_probe_result(
        report,
        item,
        drift.MirrorProbe(200, size, None, None),
        None,
        has_r2=False,
        sync_in_progress=False,
    )

    assert not any(finding.kind == "content_mismatch" for finding in report.findings)
    assert report.findings == [
        drift.Finding(
            "content_check",
            "info",
            "vocab.json",
            "unverified: body exceeds 1048576-byte probe limit",
        )
    ]


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
    drift._reset_retry_state()
    drift._REQUEST_CONTEXT.kind = "mirror"
    try:
        with pytest.raises(RuntimeError, match="HTTP 404"):
            drift._get_json("https://example")
    finally:
        drift._REQUEST_CONTEXT.kind = None


def test_admission_abort_wait_and_immediate_deadline_paths():
    gate = drift._MirrorAdmission()
    gate.reset(2, deadline=10.0)
    with pytest.raises(RuntimeError, match="deadline reached"):
        gate.acquire(lambda: 10.0, lambda _seconds: None)
    assert gate.abort_event.is_set()

    class AbortDuringWait:
        aborted = False

        def is_set(self):
            return self.aborted

        def wait(self, delay):
            assert delay == 30.0
            self.aborted = True
            return True

    gate.reset(2, deadline=None)
    gate.abort_event = AbortDuringWait()
    with pytest.raises(drift._AuditAbortError, match="exhausted probe"):
        gate.wait_delay(30.0, lambda: 0.0, drift._REAL_SLEEP)


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
    now = [0.0]

    def advance(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    with pytest.raises(urllib.error.HTTPError):
        drift._request("https://example", clock=lambda: now[0], sleeper=advance)
    assert sleeps == [60.0, 60.0, 60.0, 60.0]


def test_429_honors_full_retry_after_and_aimd_canary(monkeypatch):
    headers = Message()
    headers["Retry-After"] = "45"
    limited = urllib.error.HTTPError("https://example", 429, "slow down", headers, None)
    attempts = iter([limited, Response(200)])
    opened_at = []
    now = [0.0]

    def open_request(*_args, **_kwargs):
        opened_at.append(now[0])
        value = next(attempts)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(drift, "_OPENER", types.SimpleNamespace(open=open_request))
    drift._reset_retry_state(workers=8)
    drift._REQUEST_CONTEXT.kind = "mirror"
    try:
        assert (
            drift._request(
                "https://example",
                clock=lambda: now[0],
                sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
            ).status
            == 200
        )
    finally:
        drift._REQUEST_CONTEXT.kind = None
    assert opened_at == [0.0, 45.0]
    assert drift._MIRROR_ADMISSION.limit == 4

    # A fresh cooldown admits exactly one canary, even when several workers wait.
    drift._MIRROR_ADMISSION.rate_limited(
        45.0,
        lambda: now[0],
        drift._MIRROR_ADMISSION.admission_epoch,
    )
    now[0] += 45.0
    first_open = threading.Event()
    release = threading.Event()
    concurrent_opens = []

    def held_open(*_args, **_kwargs):
        concurrent_opens.append(threading.get_ident())
        first_open.set()
        assert release.wait(timeout=1)
        return Response(200)

    monkeypatch.setattr(drift, "_OPENER", types.SimpleNamespace(open=held_open))

    def request_once():
        drift._REQUEST_CONTEXT.kind = "mirror"
        try:
            drift._request(
                "https://example",
                clock=lambda: now[0],
                sleeper=lambda _seconds: pytest.fail("cooldown already elapsed"),
            )
        finally:
            drift._REQUEST_CONTEXT.kind = None

    threads = [threading.Thread(target=request_once) for _ in range(3)]
    for thread in threads:
        thread.start()
    assert first_open.wait(timeout=1)
    assert len(concurrent_opens) == 1
    release.set()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    # The second 429 halved 4 -> 2; one full successful window grows it to 3.
    assert drift._MIRROR_ADMISSION.limit == 3


def test_429_halves_once_per_admission_epoch_with_canary_epoch():
    gate = drift._MirrorAdmission()
    gate.reset(32, deadline=None)
    now = [0.0]
    ready = threading.Barrier(2)
    errors = []

    def limited_worker(delay):
        try:
            canary, epoch = gate.acquire(lambda: now[0], lambda _seconds: None)
            ready.wait(timeout=1)
            gate.rate_limited(delay, lambda: now[0], epoch)
            gate.finished(canary=canary, success=False)
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=limited_worker, args=(delay,)) for delay in (10.0, 20.0)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    assert errors == []
    assert gate.limit == 16
    assert gate.cooldown_until == 20.0

    now[0] = 20.0
    canary, epoch = gate.acquire(lambda: now[0], lambda _seconds: None)
    assert canary
    gate.rate_limited(5.0, lambda: now[0], epoch)
    gate.finished(canary=canary, success=False)
    assert gate.limit == 8
    assert gate.limit >= 1

    for epochs, expected_limits in (
        ((0, 1, 0), (16, 8, 8)),
        ((0, 0, 1, 0, 1), (16, 16, 8, 8, 8)),
    ):
        gate.reset(32, deadline=None)
        clock_values = iter(float(step) for step in range(len(epochs)))
        clock = clock_values.__next__
        for epoch, expected_limit in zip(epochs, expected_limits, strict=True):
            gate.rate_limited(0.0, clock, epoch)
            assert gate.limit == expected_limit


@pytest.mark.parametrize(
    ("retry_after", "expected_delay", "expected_metric"),
    [
        ("301", 300.0, {"301"}),
        ("Wed, 21 Oct 2015 07:28:00 GMT", 1.0, set()),
        ("later", 1.0, set()),
        ("²", 1.0, set()),
        ("0", 1.0, set()),
        ("-5", 1.0, set()),
    ],
)
def test_retry_after_parsing_is_bounded_and_malformed_safe(
    monkeypatch, retry_after, expected_delay, expected_metric
):
    headers = Message()
    headers["Retry-After"] = retry_after
    limited = urllib.error.HTTPError("https://example", 429, "slow down", headers, None)
    attempts = iter([limited, Response(200)])

    def open_request(*_args, **_kwargs):
        value = next(attempts)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(drift, "_OPENER", types.SimpleNamespace(open=open_request))
    drift._reset_retry_state(workers=1)
    now = [0.0]
    sleeps = []
    drift._REQUEST_CONTEXT.kind = "mirror"
    try:
        response = drift._request(
            "https://example",
            clock=lambda: now[0],
            sleeper=lambda seconds: (
                sleeps.append(seconds),
                now.__setitem__(0, now[0] + seconds),
            ),
        )
    finally:
        drift._REQUEST_CONTEXT.kind = None

    assert response.status == 200
    assert sleeps == [expected_delay]
    assert drift._mirror_retry_after_values == expected_metric


def test_retry_after_past_deadline_emits_one_partial_report(
    monkeypatch, tmp_path, capsys
):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"partial": {"hf_path": "org/partial"}}))
    audio.write_text("{}")
    item = drift.HfFile("config.json", 2, None, "a" * 40)
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "partial", "hf_path": "org/partial", "status": "mirrored"}],
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift, "_hf_repo", lambda _repo: drift.HfRepo("revision", [item])
    )
    monkeypatch.setattr(drift.time, "monotonic", lambda: 100.0)
    headers = Message()
    headers["Retry-After"] = "45"
    limited = urllib.error.HTTPError("https://example", 429, "slow down", headers, None)
    opens = []

    def fail(*_args, **_kwargs):
        opens.append(1)
        raise limited

    monkeypatch.setattr(drift, "_OPENER", types.SimpleNamespace(open=fail))
    assert (
        drift.main(
            [
                "--aliases-path",
                str(main),
                "--audio-aliases-path",
                str(audio),
                "--workers",
                "1",
                "--deadline-seconds",
                "10",
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert opens == [1]
    assert error.count("PARTIAL REPORT") == 1
    assert "partial" in error
    assert "Retry-After 45" in error


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        (
            urllib.error.HTTPError("https://example", 429, "limited", Message(), None),
            "429",
        ),
        (
            urllib.error.HTTPError(
                "https://example", 503, "unavailable", Message(), None
            ),
            "503",
        ),
        (TimeoutError("late"), "timeout"),
        (urllib.error.URLError("offline"), "url_error"),
        (OSError("reset"), "os_error"),
    ],
)
def test_mirror_retry_counters_by_cause(monkeypatch, error, cause):
    attempts = iter([error, Response(200)])

    def open_request(*_args, **_kwargs):
        value = next(attempts)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(drift, "_OPENER", types.SimpleNamespace(open=open_request))
    drift._reset_retry_state()
    now = [0.0]
    drift._REQUEST_CONTEXT.kind = "mirror"
    try:
        drift._request(
            "https://example",
            clock=lambda: now[0],
            sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
    finally:
        drift._REQUEST_CONTEXT.kind = None
    assert drift._mirror_retry_causes == {cause: 1}


def test_429_shared_cooldown_gates_new_mirror_request(monkeypatch):
    events = []
    monkeypatch.setattr(
        drift,
        "_OPENER",
        types.SimpleNamespace(
            open=lambda *_a, **_k: events.append("open") or Response(200)
        ),
    )
    drift._reset_retry_state()
    now = [10.0]
    drift._MIRROR_ADMISSION.rate_limited(
        2.0,
        lambda: now[0],
        drift._MIRROR_ADMISSION.admission_epoch,
    )

    def pass_cooldown(seconds):
        events.append(("cooldown", seconds))
        now[0] += seconds

    drift._REQUEST_CONTEXT.kind = "mirror"
    try:
        drift._request(
            "https://example",
            clock=lambda: now[0],
            sleeper=pass_cooldown,
        )
    finally:
        drift._REQUEST_CONTEXT.kind = None
    assert events == [("cooldown", 2.0), "open"]


def test_retry_after_values_are_in_metrics(monkeypatch):
    headers = Message()
    headers["Retry-After"] = "17"
    limited = urllib.error.HTTPError("https://example", 429, "limited", headers, None)
    attempts = iter([limited, Response(200)])

    def open_request(*_args, **_kwargs):
        value = next(attempts)
        if isinstance(value, BaseException):
            raise value
        return value

    now = [0.0]
    monkeypatch.setattr(drift, "_OPENER", types.SimpleNamespace(open=open_request))
    drift._reset_retry_state()
    drift._REQUEST_CONTEXT.kind = "mirror"
    try:
        drift._request(
            "https://example",
            clock=lambda: now[0],
            sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
    finally:
        drift._REQUEST_CONTEXT.kind = None
    metrics = drift.AuditProgress().metrics_line()
    assert "mirror_retry_causes=429:1" in metrics
    assert "mirror_retry_after=17" in metrics


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


def test_periodic_progress(monkeypatch, capsys):
    monkeypatch.setattr(drift, "PROGRESS_INTERVAL_SECONDS", 0.001)
    progress = drift.AuditProgress()
    progress.start_periodic()
    assert not progress._stop.wait(0.01)
    progress.stop_periodic()
    assert "reason=timer" in capsys.readouterr().err


def test_sigterm_exits_promptly_with_one_partial_report(tmp_path):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"partial-alias": {"hf_path": "org/partial"}}))
    audio.write_text("{}")
    child = textwrap.dedent(
        f"""
        import importlib.util
        import os
        import sys
        import threading

        path = {str(ROOT / "scripts" / "mirror_drift_check.py")!r}
        spec = importlib.util.spec_from_file_location("scripts.sigterm_drift", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module._catalog_entries = lambda: [{{
            "alias": "partial-alias",
            "hf_path": "org/partial",
            "status": "mirrored",
        }}]
        module._maybe_r2_client = lambda: None
        module._hf_repo = lambda _repo: module.HfRepo(
            "rev", [module.HfFile("config.json", 2, None, "a" * 40)]
        )
        blocked = threading.Event()
        def block_probe(*_args):
            os.write(2, b"PROBE_READY\\n")
            blocked.wait()
        module._probe_with_metadata = block_probe
        raise SystemExit(module.main([
            "--aliases-path", {str(main)!r},
            "--audio-aliases-path", {str(audio)!r},
        ]))
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stderr, selectors.EVENT_READ)
    before = []
    while not any("PROBE_READY" in line for line in before):
        assert selector.select(timeout=5), "subprocess did not start its probe"
        line = proc.stderr.readline()
        assert line, f"subprocess exited before readiness: {''.join(before)}"
        before.append(line)
    proc.send_signal(signal.SIGTERM)
    try:
        returncode = proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
        pytest.fail("SIGTERM waited for the blocked executor")
    remainder = proc.stderr.read()
    error = "".join(before) + remainder
    assert returncode == 124
    assert error.count("PARTIAL REPORT") == 1
    assert "partial-alias" in error


def test_sigterm_handler_is_idempotent_and_uses_one_low_level_write(monkeypatch):
    partial = drift.AliasReport(
        "partial-alias",
        "main",
        "org/partial",
        True,
        "org/partial",
        "mirrored",
    )
    current_handler = [signal.SIG_DFL]

    def install_handler(_signum, handler):
        previous = current_handler[0]
        current_handler[0] = handler
        return previous

    writes = []
    exit_codes = []
    monkeypatch.setattr(drift.signal, "signal", install_handler)
    monkeypatch.setattr(drift.os, "write", lambda fd, data: writes.append((fd, data)))
    monkeypatch.setattr(drift.os, "_exit", exit_codes.append)

    def audit_then_signal(*_args, progress, **_kwargs):
        progress.reports = [partial]
        handler = current_handler[0]
        handler(signal.SIGTERM, None)
        handler(signal.SIGTERM, None)
        return [partial]

    monkeypatch.setattr(drift, "audit", audit_then_signal)
    assert drift.main(["--fail-on", "never"]) == 0
    assert exit_codes == [124]
    assert len(writes) == 1
    assert writes[0][0] == 2
    assert writes[0][1].count(b"PARTIAL REPORT") == 1


def test_main_restores_previous_sigterm_handler(monkeypatch):
    def previous_handler(_signum, _frame):
        return None

    original = signal.signal(signal.SIGTERM, previous_handler)
    clean = drift.AliasReport(
        "clean", "main", "org/clean", True, "org/clean", "mirrored"
    )
    monkeypatch.setattr(drift, "audit", lambda *_args, **_kwargs: [clean])
    try:
        assert drift.main(["--fail-on", "never"]) == 0
        assert signal.getsignal(signal.SIGTERM) is previous_handler
    finally:
        signal.signal(signal.SIGTERM, original)


def test_exhausted_transient_emits_partial_report(monkeypatch, capsys):

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

    def exhaust(*_args, progress, **_kwargs):
        progress.reports = [partial]
        drift._reset_retry_state()
        error = TimeoutError("late")
        monkeypatch.setattr(
            drift,
            "_OPENER",
            types.SimpleNamespace(open=lambda *_a, **_k: (_ for _ in ()).throw(error)),
        )
        drift._REQUEST_CONTEXT.kind = "mirror"
        try:
            drift._request(
                "https://example",
                clock=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )
        finally:
            drift._REQUEST_CONTEXT.kind = None

    monkeypatch.setattr(drift, "audit", exhaust)
    assert drift.main([]) == 2
    error = capsys.readouterr().err
    assert error.count("PARTIAL REPORT") == 1
    assert "partial-alias" in error
    assert "missing_file config.json" in error
    assert "audit failed: late" in error


def test_real_audit_exhaustion_aborts_bounded_probe_window(
    monkeypatch, tmp_path, capsys
):
    main = tmp_path / "main.json"
    audio = tmp_path / "audio.json"
    main.write_text(json.dumps({"partial": {"hf_path": "org/partial"}}))
    audio.write_text("{}")
    files = [
        drift.HfFile(f"file-{index}.json", 2, None, f"{index:040x}")
        for index in range(10)
    ]
    monkeypatch.setattr(
        drift,
        "_catalog_entries",
        lambda: [{"alias": "partial", "hf_path": "org/partial", "status": "mirrored"}],
    )
    monkeypatch.setattr(drift, "_maybe_r2_client", lambda: None)
    monkeypatch.setattr(
        drift, "_hf_repo", lambda _repo: drift.HfRepo("revision", files)
    )
    monkeypatch.setattr(drift.time, "monotonic", lambda: 42.0)
    started = []

    def exhaust_first(_repo, item):
        started.append(item.path)
        if item.path == "file-0.json":
            drift._MIRROR_ADMISSION.abort()
            raise TimeoutError("injected exhausted transient")
        assert drift._MIRROR_ADMISSION.abort_event.wait(timeout=1)
        return drift.MirrorProbe(200, item.size, None, item.oid)

    monkeypatch.setattr(drift, "_public_probe", exhaust_first)
    assert (
        drift.main(
            [
                "--aliases-path",
                str(main),
                "--audio-aliases-path",
                str(audio),
                "--workers",
                "2",
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert error.count("PARTIAL REPORT") == 1
    assert "partial" in error
    assert "file-0.json" in started
    assert set(started) <= {"file-0.json", "file-1.json"}


def test_script_entrypoint_handles_missing_alias_file(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
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


def test_mirror_uploader_direct_execution_import(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    namespace = runpy.run_path(str(ROOT / "scripts" / "mirror_to_r2.py"))
    assert namespace["load_unmirrored"].__module__ == "mirror_unmirrored"


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


def test_mirror_uploader_refuses_unmirrored_without_force(monkeypatch, capsys):
    entry = types.SimpleNamespace(reason="unused: no pulls", since="2026-09-24")
    monkeypatch.setattr(mirror, "load_unmirrored", lambda *_args: {"org/repo": entry})
    hf_calls = []
    monkeypatch.setattr(mirror, "_hf_files", lambda repo: hf_calls.append(repo) or [])
    monkeypatch.setattr(mirror, "_r2_client", lambda *_args: object())
    assert (
        mirror._build_parser()
        .parse_args(["org/repo", "--force-unmirrored"])
        .force_unmirrored
    )

    assert mirror.mirror_repo("org/repo") == 2
    assert hf_calls == []
    refusal = capsys.readouterr().err
    assert refusal.count("SKIP intentionally unmirrored") == 1
    assert "unused: no pulls" in refusal
    assert "2026-09-24" in refusal
    assert "--force-unmirrored" in refusal

    assert mirror.mirror_repo("org/repo", force_unmirrored=True) == 0
    assert hf_calls == ["org/repo"]


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
