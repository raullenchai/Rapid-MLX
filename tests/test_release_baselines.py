#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the committed benchmark baseline inventory."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "release_baselines.py"


@pytest.fixture(scope="module")
def baselines():
    spec = importlib.util.spec_from_file_location("release_baselines", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_registry(path: Path, families: list[dict]) -> None:
    import yaml

    path.write_text(yaml.safe_dump({"families": families}))


def _payload() -> dict:
    path = (
        REPO_ROOT
        / "harness"
        / "baselines"
        / "bench-mlx-community--Qwen3.5-35B-A3B-8bit.json"
    )
    return json.loads(path.read_text())


def _single_candidate(tmp_path: Path, baselines, payload: dict) -> tuple[Path, Path]:
    registry = tmp_path / "registry.yaml"
    _write_registry(
        registry,
        [
            {
                "family": payload["family"],
                "candidates": [{"id": payload["model"]["id"]}],
            }
        ],
    )
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    candidate = baselines.load_candidates(registry)[0]
    (baseline_dir / candidate.filename).write_text(json.dumps(payload))
    return registry, baseline_dir


def test_committed_inventory_covers_every_selectable_candidate(baselines):
    report = baselines.audit(
        release=("v0.11.0", datetime(2026, 7, 24, tzinfo=timezone.utc))
    )
    assert report["candidate_count"] == 5
    assert report["covered_count"] == 5
    assert report["errors"] == []
    assert report["stale"] == []
    assert report["warnings"] == []


def test_candidate_filename_is_consumer_compatible(baselines):
    candidate = baselines.Candidate("qwen", "org/model")
    assert candidate.filename == "bench-org--model.json"


@pytest.mark.parametrize(
    ("families", "message"),
    [
        ([], "non-empty families"),
        ([{}], "needs a name"),
        ([{"family": "qwen", "candidates": []}], "must be non-empty"),
        ([{"family": "qwen", "candidates": [{}]}], "candidate needs an id"),
        (
            [
                {"family": "one", "candidates": [{"id": "org/model"}]},
                {"family": "two", "candidates": [{"id": "org/model"}]},
            ],
            "duplicate benchmark candidate",
        ),
        (
            [
                {"family": "one", "candidates": [{"id": "org/model"}]},
                {"family": "two", "candidates": [{"id": "org--model"}]},
            ],
            "collide at",
        ),
    ],
)
def test_registry_validation_fails_loudly(tmp_path, baselines, families, message):
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, families)
    with pytest.raises(ValueError, match=message):
        baselines.load_candidates(registry)


def test_audit_reports_missing_baseline(tmp_path, baselines):
    payload = _payload()
    registry, baseline_dir = _single_candidate(tmp_path, baselines, payload)
    next(baseline_dir.iterdir()).unlink()
    report = baselines.audit(
        registry_path=registry,
        baseline_dir=baseline_dir,
        release=None,
    )
    assert report["covered_count"] == 0
    assert report["errors"] == [
        "missing bench-mlx-community--Qwen3.5-35B-A3B-8bit.json (qwen3.5)"
    ]


def test_audit_reports_invalid_json(tmp_path, baselines):
    payload = _payload()
    registry, baseline_dir = _single_candidate(tmp_path, baselines, payload)
    next(baseline_dir.iterdir()).write_text("{")
    report = baselines.audit(
        registry_path=registry, baseline_dir=baseline_dir, release=None
    )
    assert "invalid JSON" in report["errors"][0]


def test_audit_reports_schema_family_and_model_errors(tmp_path, baselines):
    payload = _payload()
    registry, baseline_dir = _single_candidate(tmp_path, baselines, payload)
    path = next(baseline_dir.iterdir())

    invalid = dict(payload)
    invalid.pop("metrics")
    path.write_text(json.dumps(invalid))
    report = baselines.audit(
        registry_path=registry, baseline_dir=baseline_dir, release=None
    )
    assert "schema invalid" in report["errors"][0]

    wrong_family = _payload()
    wrong_family["family"] = "other"
    path.write_text(json.dumps(wrong_family))
    report = baselines.audit(
        registry_path=registry, baseline_dir=baseline_dir, release=None
    )
    assert "family 'other' != 'qwen3.5'" in report["errors"][0]

    wrong_model = _payload()
    wrong_model["model"]["id"] = "org/other"
    path.write_text(json.dumps(wrong_model))
    report = baselines.audit(
        registry_path=registry, baseline_dir=baseline_dir, release=None
    )
    assert "model 'org/other'" in report["errors"][0]


def test_audit_surfaces_stale_and_orphan_files(tmp_path, baselines):
    payload = _payload()
    payload["captured_at"] = "2026-01-01T00:00:00Z"
    registry, baseline_dir = _single_candidate(tmp_path, baselines, payload)
    (baseline_dir / "bench-orphan.json").write_text("{}")
    report = baselines.audit(
        registry_path=registry,
        baseline_dir=baseline_dir,
        release=("v0.11.0", datetime(2026, 7, 24, tzinfo=timezone.utc)),
    )
    assert report["covered_count"] == 1
    assert report["stale"] == [
        "bench-mlx-community--Qwen3.5-35B-A3B-8bit.json: "
        "captured 2026-01-01T00:00:00Z before v0.11.0"
    ]
    assert report["errors"] == ["orphan baseline bench-orphan.json"]
    assert report["warnings"] == []


def test_latest_release_reads_release_tag_timestamp(baselines, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["git", "describe"]:
            return subprocess.CompletedProcess(command, 0, "v0.11.0\n", "")
        return subprocess.CompletedProcess(
            command,
            0,
            "2026-07-25T10:11:12-07:00\x002026-07-24T08:09:43-07:00\n",
            "",
        )

    monkeypatch.setattr(baselines.subprocess, "run", fake_run)
    tag, captured = baselines.latest_release()
    assert tag == "v0.11.0"
    assert captured.isoformat() == "2026-07-25T10:11:12-07:00"
    assert calls[0][3:5] == ["--match", "v[0-9]*"]


def test_latest_release_falls_back_for_lightweight_tag(baselines, monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "v0.11.0\n", ""),
            subprocess.CompletedProcess([], 0, "\x002026-07-24T08:09:43-07:00\n", ""),
        ]
    )
    monkeypatch.setattr(
        baselines.subprocess, "run", lambda *args, **kwargs: next(results)
    )
    tag, captured = baselines.latest_release()
    assert tag == "v0.11.0"
    assert captured.isoformat() == "2026-07-24T08:09:43-07:00"


def test_latest_release_rejects_empty_tag_dates(baselines, monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "v0.11.0\n", ""),
            subprocess.CompletedProcess([], 0, "\x00\n", ""),
        ]
    )
    monkeypatch.setattr(
        baselines.subprocess, "run", lambda *args, **kwargs: next(results)
    )
    assert baselines.latest_release() is None


@pytest.mark.parametrize("failure_call", [0, 1])
def test_latest_release_handles_missing_git_metadata(
    baselines, monkeypatch, failure_call
):
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        current = calls
        calls += 1
        if current == failure_call:
            return subprocess.CompletedProcess(command, 128, "", "missing")
        return subprocess.CompletedProcess(command, 0, "v0.11.0\n", "")

    monkeypatch.setattr(baselines.subprocess, "run", fake_run)
    assert baselines.latest_release() is None


def test_main_human_and_json_exit_contract(baselines, monkeypatch, capsys):
    report = {
        "candidate_count": 2,
        "covered_count": 1,
        "errors": [],
        "stale": ["old"],
        "warnings": [],
    }
    monkeypatch.setattr(baselines, "latest_release", lambda: None)
    monkeypatch.setattr(baselines, "audit", lambda **kwargs: report)

    monkeypatch.setattr(sys, "argv", ["release_baselines.py"])
    assert baselines.main() == 0
    output = capsys.readouterr().out
    assert "1/2 candidates covered" in output
    assert "latest release tag unavailable" in output

    report["warnings"] = []
    monkeypatch.setattr(sys, "argv", ["release_baselines.py", "--strict-stale"])
    assert baselines.main() == 1
    assert "WARNING: stale old" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["release_baselines.py", "--json"])
    assert baselines.main() == 0
    assert json.loads(capsys.readouterr().out)["stale"] == ["old"]

    monkeypatch.setattr(
        baselines,
        "latest_release",
        lambda: ("v0.11.0", datetime(2026, 7, 24, tzinfo=timezone.utc)),
    )
    report["warnings"] = []
    monkeypatch.setattr(sys, "argv", ["release_baselines.py"])
    assert baselines.main() == 0
    assert "latest release tag unavailable" not in capsys.readouterr().out

    report["errors"] = ["broken"]
    monkeypatch.setattr(sys, "argv", ["release_baselines.py"])
    assert baselines.main() == 1
    assert "ERROR: broken" in capsys.readouterr().out


def test_stress_consumer_reads_schema_v1_and_legacy(tmp_path):
    from scripts.pr_validate.steps import stress_e2e_bench

    current = tmp_path / "current.json"
    payload = _payload()
    current.write_text(json.dumps(payload))
    assert stress_e2e_bench._load_benchmark_baseline(
        current, payload["model"]["id"]
    ) == (
        payload["metrics"],
        {
            "cold_request_ms_median": 5.0,
            "warm_request_ms_median": 5.0,
        },
        payload["model"]["revision"],
        payload["environment"]["hardware"]["chip"],
    )

    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "cold_request_ms_median": 1.0,
                "warm_request_ms_median": 2.0,
            }
        )
    )
    assert stress_e2e_bench._load_benchmark_baseline(legacy, "any") == (
        {"cold_request_ms_median": 1.0, "warm_request_ms_median": 2.0},
        {
            "cold_request_ms_median": stress_e2e_bench.BENCH_THRESHOLD_PCT,
            "warm_request_ms_median": stress_e2e_bench.BENCH_THRESHOLD_PCT,
        },
        None,
        None,
    )


def test_stress_consumer_rejects_mismatch_and_missing_metrics(tmp_path):
    from scripts.pr_validate.steps import stress_e2e_bench

    payload = _payload()
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        stress_e2e_bench._load_benchmark_baseline(path, "org/other")

    payload["metrics"] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="metrics must be an object"):
        stress_e2e_bench._load_benchmark_baseline(path, payload["model"]["id"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema": 2}, "unsupported baseline schema"),
        ({"cold_request_ms_median": 1.0}, "unrecognized legacy baseline shape"),
    ],
)
def test_stress_consumer_rejects_unknown_baseline_shapes(tmp_path, payload, message):
    from scripts.pr_validate.steps import stress_e2e_bench

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        stress_e2e_bench._load_benchmark_baseline(path, "org/model")


def test_stress_consumer_reads_loaded_huggingface_revision(tmp_path):
    from scripts.pr_validate.steps import stress_e2e_bench

    ref = tmp_path / "models--org--model" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    revision = "a" * 40
    ref.write_text(revision)
    assert (
        stress_e2e_bench._cached_model_revision("org/model", cache_dir=tmp_path)
        == revision
    )


@pytest.mark.parametrize("contents", [None, "main", "a" * 39])
def test_stress_consumer_rejects_unverifiable_revision(tmp_path, contents):
    from scripts.pr_validate.steps import stress_e2e_bench

    if contents is not None:
        ref = tmp_path / "models--org--model" / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_text(contents)
    with pytest.raises(ValueError, match="cannot verify loaded revision"):
        stress_e2e_bench._cached_model_revision("org/model", cache_dir=tmp_path)


def test_release_gauntlet_runs_baseline_audit():
    script = (REPO_ROOT / "scripts" / "release_check_m3.sh").read_text()
    assert '"$PY" scripts/release_baselines.py' in script
