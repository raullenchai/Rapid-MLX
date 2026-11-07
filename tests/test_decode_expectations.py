"""Offline chip/configuration-specific references never borrow faster hardware."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx import decode_expectations as de
from scripts.refresh_decode_expectations import build_snapshot

ROOT = Path(__file__).resolve().parents[1]


def resolve(alias, chip, ram, **kwargs):
    repo = kwargs.pop("repo_id", de.snapshot()["models"].get(alias, {}).get("repo_id"))
    return de.expectation(alias, chip, ram, repo_id=repo, **kwargs)


@pytest.mark.parametrize(
    "brand,expected",
    [
        ("Apple M4", "M4"),
        ("Apple M4 Pro", "M4 Pro"),
        ("Apple M5 Max", "M5 Max"),
        ("Apple M3 Ultra", "M3 Ultra"),
        ("Apple M40", "M40"),
        ("Intel M4 spoof", None),
        ("Apple Silicon", None),
    ],
)
def test_chip_identity(brand, expected):
    assert de.chip_name(brand) == expected


def test_probe_success_and_unavailable(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="Apple M4 Pro\n")
    )
    assert de.detected_chip() == "M4 Pro"
    for failure in (OSError(), subprocess.TimeoutExpired("sysctl", 2)):

        def fail(*a, error=failure, **k):
            raise error

        monkeypatch.setattr(subprocess, "run", fail)
        assert de.detected_chip() is None


def test_real_m4_defaults_do_not_borrow_pro_or_mtp_off():
    nine = resolve("qwen3.5-9b-4bit", "M4", 32)
    assert nine["status"] == "unavailable" and nine["method"] == "mtp"
    assert nine["baseline"]["n"] == 6
    assert "MTP-off reference" in de.summary(nine)
    large = resolve("qwen3.8-27b-4bit", "M4", 32)
    assert large["status"] == "unavailable"
    assert "baseline" not in large  # two submissions are not three cases
    assert "baseline_ceiling" not in large
    assert "smooth" not in de.summary(large)
    four = resolve("qwen3.5-4b-4bit", "M4", 32)
    assert four["status"] == "measured" and four["n"] == 6
    assert four["method"] == "none"
    assert "smooth" in de.summary(four)
    assert four["upper"] < resolve("qwen3.5-4b-4bit", "M4 Pro", 48)["upper"]


@pytest.mark.parametrize(
    "alias,chip,ram",
    [
        ("unknown", "M4", 32),
        ("qwen3.5-4b-4bit", None, 32),
        ("qwen3.5-4b-4bit", "M4", float("nan")),
        ("qwen3.5-4b-4bit", "M4", 0),
    ],
)
def test_missing_identity_never_claims_expected(alias, chip, ram):
    assert de.summary(resolve(alias, chip, ram)) == "Expected decode range unavailable"


def test_memory_scope_and_foreign_repo():
    assert resolve("qwen3.8-27b-4bit", "M4", 8)["status"] == "insufficient_ram"
    assert "RAM requirement" in de.summary(resolve("qwen3.8-27b-4bit", "M4", 8))
    assert resolve("qwen3.5-4b-4bit", "M4", 64)["status"] == "unavailable"
    assert (
        resolve("qwen3.5-4b-4bit", "M4", 32, repo_id="custom/foreign")["status"]
        == "unavailable"
    )
    assert resolve("qwen3.6-35b-4bit", "M4", 64).get("baseline_ceiling") is None
    assert resolve("qwen3.5-4b-4bit", "M42", 32)["status"] == "unavailable"


def test_feel_only_from_real_measurements():
    for median, feel in [(35, "smooth"), (15, "usable"), (7, "tight")]:
        value = {
            "status": "measured",
            "method": "mtp",
            "lower": median - 1,
            "upper": median + 1,
            "median": median,
            "n": 3,
            "ram_min_gib": 32,
            "ram_max_gib": 32,
        }
        assert f"MTP on · {feel}" in de.summary(value)


def run(
    identifier="one",
    *,
    chip="Apple M4",
    method="none",
    ram=32,
    repo="mlx-community/Qwen3.5-4B-MLX-4bit",
    values=(20, 40),
):
    return {
        "submission_id": identifier,
        "contributor": {"private": "/private/secret"},
        "task_type": "text_generation",
        "model": {"repo_id": repo},
        "machine": {"chip": chip, "memory_gib": ram},
        "execution": {"speculative_decoding": {"method": method}},
        "protocol": {"id": "rapid-community-speed", "version": 2},
        "cases": [
            {"case_id": case, "decode_tps": value}
            for case, value in zip(("pp512-tg128", "pp2048-tg512"), values)
        ],
    }


def test_generator_distinct_submission_not_cases_config_and_privacy():
    first = run()
    rows = [
        first,
        first,
        run("two", values=(30, 50)),
        run("three", values=(40, 60)),
        run("pro", chip="Apple M4 Pro"),
        run("mtp", method="mtp"),
        run("otherquant", repo="mlx-community/Qwen3.5-4B-8bit"),
    ]
    result = build_snapshot(rows, "2026-11-07")
    cohorts = result["models"]["qwen3.5-4b-4bit"]["cohorts"]
    baseline = next(c for c in cohorts if c["chip"] == "M4" and c["method"] == "none")
    assert baseline == {
        "chip": "M4",
        "method": "none",
        "n": 3,
        "ram_min_gib": 32,
        "ram_max_gib": 32,
        "lower": 35,
        "median": 40,
        "upper": 45,
    }
    text = json.dumps(result)
    for private in ("submission_id", "contributor", "/private/secret", "otherquant"):
        assert private not in text


def test_generator_excludes_invalid_and_incompatible_runs():
    invalid = []
    for key, value in [
        ("task_type", "image_generation"),
        ("protocol", {"id": "other", "version": 2}),
        ("cases", []),
        ("submission_id", ""),
    ]:
        row = run()
        row[key] = value
        invalid.append(row)
    invalid += [
        run(chip="not a chip"),
        run(ram=float("nan")),
        run(ram=0),
        run(method="dflash"),
        run(values=(float("nan"), -1)),
    ]
    assert not build_snapshot(invalid, "2026-11-07")["models"]["qwen3.5-4b-4bit"][
        "cohorts"
    ]


def test_shared_golden_vectors_offline():
    for vector in json.loads(
        (ROOT / "tests/fixtures/decode_expectations_golden.json").read_text()
    ):
        assert (
            de.summary(
                resolve(
                    vector["alias"],
                    vector["chip"],
                    vector["ram_gib"],
                    repo_id=vector["repo_id"],
                )
            )
            == vector["summary"]
        )
        assert (
            de.summary(
                resolve(
                    vector["alias"],
                    vector["chip"],
                    vector["ram_gib"],
                    repo_id=vector["repo_id"],
                ),
                compact=True,
            )
            == vector["compact"]
        )


def test_models_and_doctor_match_same_offline_resolver(monkeypatch, capsys):
    from rapid_mlx import recommendations
    from rapid_mlx.cli import _available_models_json_payload, models_command
    from rapid_mlx.doctor import env_health as eh

    monkeypatch.setattr(de, "detected_chip", lambda: "M4")
    monkeypatch.setattr(recommendations, "physical_ram_gb", lambda: 32)
    payload = _available_models_json_payload()
    entry = next(e for e in payload["text"] if e["alias"] == "qwen3.5-4b-4bit")
    assert entry["expected_decode"] == resolve(entry["alias"], "M4", 32)
    models_command(
        SimpleNamespace(
            search="qwen3.5-4b-4bit", modality="text", json=False, cached=False
        )
    )
    expected = de.summary(entry["expected_decode"])
    assert de.summary(entry["expected_decode"], compact=True) in capsys.readouterr().out
    monkeypatch.setattr(eh, "_detect_apple_silicon", lambda: ("Apple M4", 32))
    section = eh.section_system()
    assert any(expected in check.label for check in section.checks)


def test_bundle_is_one_physical_source_and_packaged():
    assert not (
        ROOT / "apps/rapid-mac/Sources/Rapid/Resources/decode_expectations.json"
    ).exists()
    assert '"decode_expectations.json"' in (ROOT / "pyproject.toml").read_text()
    assert (
        'cp "$DECODE_EXPECTATIONS_SRC" "$CONTENTS/Resources/decode_expectations.json"'
        in (ROOT / "apps/rapid-mac/scripts/build.sh").read_text()
    )


def test_refresh_command_uses_explicit_local_input_only(tmp_path, monkeypatch):
    import runpy
    import sys

    source = tmp_path / "runs.json"
    source.write_text(json.dumps([run()]))
    output = tmp_path / "snapshot.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_decode_expectations.py",
            "--input",
            str(source),
            "--refreshed-at",
            "2026-11-07",
            "--output",
            str(output),
        ],
    )
    runpy.run_module("scripts.refresh_decode_expectations", run_name="__main__")
    assert json.loads(output.read_text())["schema_version"] == 1


def test_backend_owned_or_unqualified_presets_never_assume_ordinary_decode():
    result = build_snapshot([], "2026-11-07")
    assert (
        result["models"]["qwen3.8-flash-next-tensorfold"]["default_method"] == "unknown"
    )
    assert result["models"]["qwen3.5-4b-4bit"]["default_method"] == "none"
    assert result["models"]["qwen3.5-9b-4bit"]["default_method"] == "mtp"
    assert "configuration unavailable" in de.summary(
        resolve("qwen3.8-flash-next-tensorfold", "M4", 4096)
    )


def test_packaging_snapshot_requires_object(monkeypatch):
    de.snapshot.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", lambda *a, **k: "[]")
        with pytest.raises(ValueError, match="JSON object"):
            de.snapshot()
    de.snapshot.cache_clear()
    assert de.snapshot()["schema_version"] == 1
