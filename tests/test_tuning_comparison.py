# SPDX-License-Identifier: Apache-2.0
"""Offline comparison execution, privacy and profile distillation contracts."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rapid_mlx import __version__, tuning
from rapid_mlx.catalog import rcj_digest
from rapid_mlx.community_bench import comparison
from rapid_mlx.community_bench.benchmark_contracts import BenchmarkRunValidator
from rapid_mlx.community_bench.run_builder import execution_config_digest
from scripts import tuning_distill

NOW = datetime(2026, 11, 10, 23, tzinfo=timezone.utc)
REAL_PLANNING_CONTEXT = comparison.planning_context
SESSION = "00000000-0000-4000-8000-000000000001"
DIGEST = "sha256:" + "1" * 64


def sample(arm="default", rate=100):
    run = json.loads(
        (Path(__file__).parent / "fixtures/community_bench_cached_run.json").read_text()
    )
    run["run_id"] = str(uuid4())
    run["started_at"] = run["completed_at"] = "2026-11-10T22:00:00Z"
    runtime = run["execution"]["runtime"]
    runtime.update(rapid_mlx=__version__, mlx="0.32.4")
    run["machine"]["os"]["version"] = "26.5.2"
    run["machine"]["profile"].update(
        chip="Apple M3 Ultra", gpu_cores=80, memory_gib=256
    )
    run["machine"]["profile_digest"] = rcj_digest(run["machine"]["profile"])
    ctx = tuning.TuningContext(
        tuning.HardwareProfile("applegpu_g15d", 15, "Ultra", 80, 274877906944),
        "0.32.4",
        "26.5.2",
        tuning.ModelProfile("qwen3_5_moe_text", 2048, 512, 256, 8, 4, 64, (), "affine"),
        mode="mtp",
        has_mtp_head=True,
        rapid_mlx=__version__,
    )
    run["tuning"] = {
        "context": json.loads(json.dumps(tuning.dump(ctx))),
        "overrides": {"mtp_max_depth": 1 if arm == "default" else 3},
        "outputs": {c["case_id"]: [DIGEST] * 5 for c in run["workload"]["cases"]},
    }
    language = run["execution"]["task"]["language"]
    language["speculative_decoding"] = {
        "method": "mtp",
        "max_draft_tokens": run["tuning"]["overrides"]["mtp_max_depth"],
        "assistant_source": "embedded_head",
    }
    run["execution"]["config_digest"] = execution_config_digest(run["execution"])
    run["experiment"] = dict(
        group_id=SESSION,
        arm_id=arm,
        role="baseline" if arm == "default" else "candidate",
        sequence_index=1,
        order_strategy="alternating",
        varied_fields=["/execution/task/language/speculative_decoding"],
    )
    for m in run["measurements"]:
        m["decode_duration_ms"] = (m["output_tokens"] - 1) * 1000 / rate
        m["ttft_ms"] = 20
        m["total_duration_ms"] = m["decode_duration_ms"] + 20
    return run


def loaded_sample():
    """A run whose recorded knobs reflect the arm values the worker received."""
    run = sample()
    values = json.loads(os.environ.get("RAPID_MLX_BENCHMARK_KNOBS", "{}"))
    for name, value in values.items():
        run["tuning"]["context"]["knobs"][name]["value"] = value
        if name in run["tuning"]["overrides"]:
            run["tuning"]["overrides"][name] = value
    return run


@pytest.fixture(autouse=True)
def offline_comparison_plan(monkeypatch):
    def context(alias, request):
        ctx = sample()["tuning"]["context"]
        spec = json.loads(request.overrides.get("speculative_config", "{}"))
        ctx["knobs"]["mtp_max_depth"]["value"] = spec.get("num_speculative_tokens", 1)
        return ctx

    monkeypatch.setattr(comparison, "planning_context", context)


def paired():
    return [
        sample(arm, rate + i / 10)
        for i in range(3)
        for arm, rate in [("default", 100), ("mtp-3", 110)]
    ]


def test_distill_exact_cell_and_profile_shape():
    result = tuning_distill.distill(paired(), as_of=NOW)
    assert not result["rejected"]
    entry = result["entries"][0]
    assert entry["knobs"]["mtp_max_depth"] == 3
    assert entry["hardware"][0]["gpu_cores"] == [80, 80]
    assert entry["hardware"][0]["memory_size"] == [274877906944] * 2
    assert entry["software"]["mlx"] == "===0.32.4"
    assert entry["workload"] == dict(mode="mtp", width=1, batch=1)
    assert entry["model"]["geometry"] == [2048, 512, 256, 8, 4, 64]
    assert entry["provenance"]["repetitions"] == {"default": 3, "mtp-3": 3}


@pytest.mark.parametrize(
    "experts,top_k,accepted",
    [
        (0, 0, True),
        (8, 8, True),
        (8, 2, True),
        (0, 1, False),
        (8, 0, False),
        (8, 9, False),
    ],
)
def test_distiller_dense_and_moe_geometry(experts, top_k, accepted):
    runs = paired()
    for run in runs:
        model = run["tuning"]["context"]["model"]
        model.update(num_experts=experts, top_k=top_k)
        if experts == 0:
            model["family"] = "qwen3_5_text"
    result = tuning_distill.distill(runs, as_of=NOW)
    assert bool(result["entries"]) is accepted
    if accepted:
        assert result["entries"][0]["model"]["geometry"][2:4] == [experts, top_k]
        assert not result["rejected"]
    else:
        assert all("facts" in row["reason"] for row in result["rejected"])


@pytest.mark.parametrize("fact", ["checkpoint", "profiles", "revision"])
@pytest.mark.parametrize("conflict", [False, True])
def test_distiller_conflicts_use_emitted_selector(fact, conflict):
    first, second = paired(), paired()
    for run in second:
        run["experiment"]["group_id"] = SESSION.replace("0001", "0002")
        if fact == "checkpoint":
            run["model"]["components"][0]["source"]["resolved_revision"] = "2" * 40
        elif fact == "profiles":
            run["tuning"]["context"]["profiles"]["overlay"] = "other.1"
        else:
            run["execution"]["runtime"]["distribution"] = "source"
            run["execution"]["runtime"]["rapid_mlx_revision"] = "2" * 40
            run["execution"]["config_digest"] = execution_config_digest(
                run["execution"]
            )
        if conflict and run["experiment"]["arm_id"] != "default":
            run["tuning"]["overrides"]["mtp_max_depth"] = 4
    assert tuning_distill.cell_key(first[0]) != tuning_distill.cell_key(second[0])
    result = tuning_distill.distill(first + second, as_of=NOW)
    if conflict:
        assert not result["entries"]
        assert len(result["rejected"]) == 1
        assert "conflicting" in result["rejected"][0]["reason"]
        assert "selector" in result["rejected"][0]["reason"]
    else:
        assert not result["rejected"]
        assert len(result["entries"]) == 1
        identities = json.loads(
            result["entries"][0]["provenance"]["evidence"].removeprefix(
                "community-comparison:"
            )
        )
        assert identities == [
            {
                "session": group[0]["experiment"]["group_id"],
                "facts": json.loads(tuning_distill.cell_key(group[0])),
            }
            for group in (first, second)
        ]


def test_distiller_keeps_distinct_emitted_selectors():
    first, second = paired(), paired()
    for run in second:
        run["tuning"]["context"]["model"]["hidden_size"] = 4096
        if run["experiment"]["arm_id"] != "default":
            run["tuning"]["overrides"]["mtp_max_depth"] = 4
    result = tuning_distill.distill(first + second, as_of=NOW)
    assert not result["rejected"]
    assert [entry["knobs"]["mtp_max_depth"] for entry in result["entries"]] == [3, 4]


@pytest.mark.parametrize(
    "field", ["assistant_source", "draft_model_identity_digest", "assistant_component"]
)
def test_distiller_separates_assistant_identities(field):
    runs = paired()
    for run in runs:
        spec = run["execution"]["task"]["language"]["speculative_decoding"]
        candidate = run["experiment"]["arm_id"] != "default"
        if field != "assistant_source" or candidate:
            spec.update(
                assistant_source="external_model", draft_model_identity_digest=DIGEST
            )
        if field == "assistant_component":
            component = copy.deepcopy(run["model"]["components"][0])
            component.update(component_id="draft", role="language_model")
            component["source"]["resolved_revision"] = ("2" if candidate else "1") * 40
            run["model"]["components"].append(component)
            run["model"]["components"].sort(key=lambda c: c["component_id"])
        elif field == "draft_model_identity_digest" and candidate:
            spec[field] = "sha256:" + "2" * 64
        run["execution"]["config_digest"] = execution_config_digest(run["execution"])
    assert tuning_distill.cell_key(runs[0]) != tuning_distill.cell_key(runs[1])
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]


@pytest.mark.parametrize("arm", ["default", "mtp-3", "both"])
@pytest.mark.parametrize("count", [0, 1, 4, 6])
def test_digest_equivalence_requires_every_measured_round(arm, count):
    runs = paired()
    for run in runs:
        if arm == "both" or run["experiment"]["arm_id"] == arm:
            run["tuning"]["outputs"] = {
                case["case_id"]: [DIGEST] * count for case in run["workload"]["cases"]
            }
    assert not tuning_distill.quality_passes(runs[1::2], runs[::2])


@pytest.mark.parametrize("missing", [False, True])
def test_digest_equivalence_requires_each_case(missing):
    runs = paired()
    for run in runs:
        outputs = run["tuning"]["outputs"]
        if missing:
            outputs.pop(next(iter(outputs)))
        else:
            outputs["extra-case"] = [DIGEST] * 5
    assert not tuning_distill.quality_passes(runs[1::2], runs[::2])


@pytest.mark.parametrize(
    "change",
    [
        "few",
        "below_gain",
        "overlap",
        "diverged",
        "null",
        "unstable",
        "battery",
        "thermal",
        "pressure",
        "power",
        "stale",
        "future",
        "version",
        "mlx",
        "macos",
        "hardware",
        "batch",
        "session",
        "duplicate",
        "no-default",
        "legacy",
        "overrides",
        "knobs",
        "plain",
    ],
)
def test_distill_rejects_unqualified_evidence(change):
    runs = paired()
    candidate = runs[1]
    ctx = candidate["tuning"]["context"]
    if change == "few":
        runs = runs[:4]
    elif change == "below_gain":
        runs = [
            sample(arm, rate)
            for _ in range(3)
            for arm, rate in [("default", 100), ("mtp-3", 101)]
        ]
    elif change == "overlap":
        candidate["measurements"][0]["decode_duration_ms"] = (128 - 1) * 1000 / 99
        candidate["measurements"][0]["total_duration_ms"] = (
            candidate["measurements"][0]["decode_duration_ms"] + 20
        )
    elif change in ("diverged", "null", "unstable"):
        candidate["tuning"]["outputs"]["pp512-tg128"][0] = (
            None if change == "null" else "sha256:" + "2" * 64
        )
    elif change in ("battery", "thermal", "pressure", "power"):
        field, value = {
            "battery": ("power_source", "battery"),
            "thermal": ("thermal_state", "fair"),
            "pressure": ("memory_pressure", "warning"),
            "power": ("low_power_mode", True),
        }[change]
        candidate["machine"]["conditions_after"][field] = value
    elif change in ("stale", "future"):
        candidate["completed_at"] = (
            "2026-01-01T00:00:00Z" if change == "stale" else "2027-01-01T00:00:00Z"
        )
    elif change == "version":
        candidate["execution"]["runtime"]["rapid_mlx"] = "0.1"
    elif change == "mlx":
        ctx["mlx"] = "0.32.3"
    elif change == "macos":
        ctx["macos"] = "26.6.2"
    elif change == "hardware":
        ctx["hardware"]["gpu_cores"] = None
    elif change == "batch":
        ctx["batch"] = 2
    elif change == "session":
        candidate["experiment"]["group_id"] = str(uuid4())
    elif change == "duplicate":
        runs.append(copy.deepcopy(candidate))
    elif change == "no-default":
        runs = runs[1::2]
    elif change == "legacy":
        for run in runs:
            del run["tuning"]
    elif change == "overrides":
        candidate["tuning"]["overrides"]["mtp_max_depth"] = 4
    elif change == "knobs":
        candidate["tuning"]["context"]["knobs"]["qwen35_affine4_bank_views"][
            "value"
        ] = False
    elif change == "plain":
        for run in runs[1::2]:
            run["tuning"]["context"]["mode"] = "plain"
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]


def test_quality_note_and_commit_preserved():
    runs = paired()
    for run in runs[1::2]:
        run["tuning"]["quality"] = dict(
            n=100,
            questions_digest=DIGEST,
            plain=dict(delta_pp=2, paired_p=0.5),
            default=dict(delta_pp=1, paired_p=1),
        )
    for run in runs:
        run["execution"]["runtime"].update(
            distribution="source", rapid_mlx_revision="a" * 40
        )
    entry = tuning_distill.distill(runs, as_of=NOW)["entries"][0]
    assert entry["provenance"]["commit"] == "a" * 40
    assert entry["provenance"]["quality"][0]["n"] == 100
    runs[1]["tuning"]["quality"].pop("plain")
    assert not tuning_distill.quality_passes(runs[1::2], runs[::2])


def test_sessions_conflict_and_agreement():
    first = paired()
    second = copy.deepcopy(first)
    for r in second:
        r["run_id"] = str(uuid4())
        r["experiment"]["group_id"] = str(uuid4())
    # Keep every second-session arm in one paired group.
    for r in second:
        r["experiment"]["group_id"] = SESSION.replace("0001", "0002")
    assert len(tuning_distill.distill(first + second, as_of=NOW)["entries"]) == 1
    for r in second[1::2]:
        r["tuning"]["overrides"]["mtp_max_depth"] = 4
    result = tuning_distill.distill(first + second, as_of=NOW)
    assert not result["entries"] and "conflicting" in result["rejected"][-1]["reason"]


def test_schema_optional_and_identifier_free():
    run = sample()
    validator = BenchmarkRunValidator()
    validator.validate(run)
    for block, key in [
        ("context", "hostname"),
        ("overrides", "invalid-knob"),
        ("outputs", "prompt"),
    ]:
        bad = copy.deepcopy(run)
        bad["tuning"][block][key] = "private"
        with pytest.raises(ValueError):
            validator.validate(bad)
    del run["tuning"]
    validator.validate(run)


def test_candidates_come_from_registry():
    context = sample()["tuning"]["context"]
    candidates = list(comparison.candidate_values(context))
    assert {"mtp_max_depth": 2} in candidates
    assert {"mtp_gdn_fusion": False} in candidates
    assert not any(
        name in c
        for name in ("mtp_prefill_block_size", "mtp_forward_costs_ms")
        for c in candidates
    )
    assert {"qwen35_affine4_bank_views": False} in candidates
    assert {"qwen35_affine4_rows": (2, 4)} in candidates
    distances = [len(c) for c in candidates]
    assert set(distances) == {1}
    names = list(dict.fromkeys(next(iter(c)) for c in candidates))
    assert names == [name for name in tuning.KNOBS if name in names]
    for salt, name in enumerate(names):
        assert next(comparison.candidate_values(context, salt=salt)) == next(
            c for c in candidates if name in c
        )
    assert not any(
        "lane_simd_probe" in c or "lane_tensor_units" in c for c in candidates
    )


@pytest.mark.parametrize("repetitions", [3, 7])
@pytest.mark.parametrize("with_progress", [False, True])
def test_run_comparison_interleaves_and_archives(
    monkeypatch, tmp_path, repetitions, with_progress
):
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    observed = []

    def run_local(alias, **kwargs):
        assert kwargs["tuning_observation"]
        observed.append(kwargs["acceleration"])
        return loaded_sample()

    from rapid_mlx.community_bench import local_runner

    monkeypatch.setattr(local_runner, "run_local", run_local)
    monkeypatch.setattr(comparison, "measure_configuration", run_local)
    archive = LocalRunArchive(tmp_path)
    progress = []
    result = comparison.run_comparison(
        "model",
        configs=4,
        salt=0,
        repetitions=repetitions,
        archive=archive,
        progress=progress.append if with_progress else None,
    )
    if with_progress:
        assert f"Plan: 4 configs × {repetitions} repetitions" in progress
        assert "estimated time: shown after the first repetition" in progress
        assert any("current source:" in line for line in progress)
    else:
        assert progress == []
    runs = result["configurations"]
    count = 4 * repetitions
    assert len(runs) == count
    order = [r["experiment"]["arm_id"] for r in runs]
    assert order[:4] == ["default", "nearby-1", "nearby-2", "nearby-3"]
    assert order[4:8] == ["nearby-3", "nearby-2", "nearby-1", "default"]
    assert runs[0]["paired_runs"] == runs[1:]
    assert len(runs[0]["paired_runs"]) == count - 1
    assert len({r["experiment"]["group_id"] for r in runs}) == 1
    assert [r["experiment"]["sequence_index"] for r in runs] == list(
        range(1, count + 1)
    )
    assert len(archive.list()) == count
    assert "none" in comparison.summary_lines(runs)[-1]


@pytest.mark.parametrize(
    "repetitions,acceleration",
    [(0, None), (2, None), (11, None)],
)
def test_comparison_rejects_invalid_options(repetitions, acceleration):
    with pytest.raises(ValueError):
        comparison.run_comparison(
            "model", configs=2, repetitions=repetitions, acceleration=acceleration
        )


@pytest.mark.parametrize("observations", [False, True])
def test_default_without_neighbors_is_legacy_run(monkeypatch, tmp_path, observations):
    from rapid_mlx.community_bench import local_runner
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    run = sample()
    if observations:
        for decision in run["tuning"]["context"]["knobs"].values():
            decision["source"] = "capability"
    else:
        del run["tuning"]
    run.pop("experiment")
    monkeypatch.setattr(
        comparison, "planning_context", lambda *a: run.get("tuning", {}).get("context")
    )
    calls = []

    def legacy(alias, **kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setattr(local_runner, "run_local", legacy)
    monkeypatch.setattr(
        comparison,
        "measure_configuration",
        lambda *a, **kw: pytest.fail("no neighbor must never start a worker"),
    )
    archive = LocalRunArchive(tmp_path)
    progress = []
    assert (
        comparison.run_comparison(
            "model", configs=2, archive=archive, progress=progress.append
        )
        is run
    )
    assert "No tunable settings apply here; ran a single benchmark instead." in progress
    assert len(calls) == 1
    assert calls[0]["archive"] is archive
    assert "experiment" not in run and "paired_runs" not in run
    assert "selection" not in run.get("tuning", {})


def test_summary_winners_divergence():
    runs = paired()
    assert "mtp-3" in comparison.summary_lines(runs)[-1]
    runs[1]["tuning"]["outputs"]["pp512-tg128"][0] = "sha256:" + "2" * 64
    assert any(
        "outputs differ — not comparable" in line
        for line in comparison.summary_lines(runs)
    )


def test_distiller_cli(tmp_path, capsys):
    file = tmp_path / "comparison.json"
    file.write_text(
        json.dumps(
            {"configurations": paired(), "measured_configs": 2, "requested_configs": 2}
        )
    )
    assert tuning_distill.main([str(file), "--as-of", NOW.isoformat()]) == 0
    assert json.loads(capsys.readouterr().out)["entries"]
    file.write_text(json.dumps(sample()))
    assert tuning_distill.main([str(file)]) == 0
    assert not json.loads(capsys.readouterr().out)["entries"]


def test_cli_comparison_entry(monkeypatch, tmp_path, capsys):
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    monkeypatch.setattr(
        cli.LocalRunArchive, "default", lambda: LocalRunArchive(tmp_path)
    )
    seen = {}

    def run(*a, **kw):
        seen.update(kw)
        return {
            "configurations": paired(),
            "measured_configs": 2,
            "requested_configs": 2,
        }

    monkeypatch.setattr(comparison, "run_comparison", run)
    args = build_parser().parse_args(
        ["benchmark", "run", "model", "--configs", "4", "--repetitions", "3"]
    )
    assert cli._benchmark_command(args) == 0
    assert seen["repetitions"] == 3
    assert seen["configs"] == 4
    assert "local observation" in capsys.readouterr().out


@pytest.mark.parametrize(
    "mode,experts,quant,has_args",
    [
        ("mtp", True, (4, 64, "affine"), True),
        ("mtp", True, None, True),
        ("plain", False, None, False),
        ("plain", False, None, True),
    ],
)
def test_capture_tuning_is_json_and_identifier_free(
    monkeypatch, mode, experts, quant, has_args
):
    import sys
    import types

    core = types.ModuleType("mlx.core")
    core.device_info = lambda: {
        "device_name": "Apple M3 Ultra",
        "memory_size": 274877906944,
    }
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    model = SimpleNamespace(
        mtp=object(),
        mtp_forward=lambda: None,
        make_mtp_cache=lambda: None,
        mtp_max_speculative_tokens=3,
    )
    if has_args:
        model.args = SimpleNamespace(
            model_type="qwen3_5_moe_text",
            num_experts=256 if experts else 0,
            hidden_size=2048,
        )
    monkeypatch.setattr(comparison, "_expert_quantization", lambda _: quant)
    monkeypatch.setattr(comparison, "resolve_max_depth", lambda *a, **kw: 4)
    result = SimpleNamespace(
        short=SimpleNamespace(rounds_raw=[SimpleNamespace(output_digest=DIGEST)]),
        long=SimpleNamespace(rounds_raw=[SimpleNamespace(output_digest=DIGEST)]),
    )
    spec = {"method": mode, "max_draft_tokens": 4}
    block = comparison.capture_tuning(
        SimpleNamespace(language_model=model), {"speculative_decoding": spec}, result
    )
    assert block["overrides"] == ({"mtp_max_depth": 3} if mode == "mtp" else {})
    assert block["outputs"]["pp512-tg128"] == [DIGEST]
    assert "Apple" not in json.dumps(block)
    assert block["context"]["model"]["bits"] == (4 if experts and quant else None)
    assert isinstance(block["context"]["model"]["attention_kinds"], list)


def test_distiller_module_entry(tmp_path, monkeypatch, capsys):
    import runpy
    import sys

    path = tmp_path / "run.json"
    path.write_text(json.dumps(sample()))
    monkeypatch.setattr(sys, "argv", ["tuning_distill", str(path)])
    monkeypatch.delitem(sys.modules, "scripts.tuning_distill")
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.tuning_distill", run_name="__main__")
    assert exc.value.code == 0
    assert "entries" in capsys.readouterr().out


def test_tuning_schema_projection_is_generated():
    from scripts import sync_model_catalog_schemas

    source, name = sync_model_catalog_schemas.SOURCES[0]
    assert name == "benchmark-run.schema.json"
    assert (
        source.read_bytes()
        == (sync_model_catalog_schemas.DESTINATION / name).read_bytes()
    )


@pytest.mark.parametrize(
    "field,value", [("gpu_cores", 60), ("memory_size", 12345), ("tier", "Max")]
)
def test_distiller_rejects_inconsistent_hardware(field, value):
    runs = paired()
    runs[1]["tuning"]["context"]["hardware"][field] = value
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]


def test_distiller_separates_fixed_execution_and_missing_model_facts():
    runs = paired()
    runs[1]["execution"]["task"]["language"]["prefill_chunk_size"] = 8192
    runs[1]["execution"]["config_digest"] = execution_config_digest(
        runs[1]["execution"]
    )
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]
    runs = paired()
    runs[1]["tuning"]["context"]["model"]["hidden_size"] = None
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]


def test_preview_preserves_the_optional_tuning_block():
    from rapid_mlx.community_bench.atomic_upload import preview_run

    run = sample()
    preview = preview_run(run, install_id="a" * 12)
    assert json.loads(preview["payload_json"])["tuning"] == run["tuning"]


def test_candidates_respect_the_model_depth_contract():
    context = sample()["tuning"]["context"]
    context["model"]["family"] = "qwen4_exp"
    context["knobs"]["mtp_max_depth"]["value"] = 2
    candidates = list(comparison.candidate_values(context))
    assert {"mtp_max_depth": 1} in candidates
    assert not any(c.get("mtp_max_depth", 0) > 2 for c in candidates)


def test_distiller_requires_stable_distinct_configurations():
    runs = paired()
    runs[0]["tuning"]["overrides"]["mtp_max_depth"] = 2
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]
    runs = paired()
    for run in runs[::2]:
        run["tuning"]["overrides"]["mtp_max_depth"] = 3
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]


def test_distiller_handles_utc_dates_and_zero_decode_duration():
    assert tuning_distill.distill(paired(), as_of=NOW.replace(tzinfo=None))["entries"]
    runs = paired()
    runs[1]["measurements"][0]["decode_duration_ms"] = 0
    runs[1]["measurements"][0]["total_duration_ms"] = 20
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]


@pytest.mark.parametrize(
    "field,value", [("n", 99), ("delta_pp", 4), ("paired_p", 0.04)]
)
def test_failed_quality_studies_remain_valid_observations(field, value):
    runs = paired()
    for run in runs[1::2]:
        run["tuning"]["outputs"]["pp512-tg128"] = ["sha256:" + "2" * 64] * 5
        note = dict(
            n=100,
            questions_digest=DIGEST,
            plain=dict(delta_pp=0, paired_p=1),
            default=dict(delta_pp=0, paired_p=1),
        )
        if field == "n":
            note[field] = value
        else:
            note["plain"][field] = value
        run["tuning"]["quality"] = note
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]


def test_distilled_entry_loads_and_resolves_only_its_explicit_cell(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from rapid_mlx.tuning import defaults

    runs = paired()
    entry = tuning_distill.distill(runs, as_of=NOW)["entries"][0]
    blocked = {
        name
        for name, decision in runs[1]["tuning"]["context"]["knobs"].items()
        if decision["source"] == "capability"
    }
    assert blocked and not blocked.intersection(entry["knobs"])
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version=1,
                profile_version="comparison.1",
                generated_at=NOW.isoformat(),
                entries=[entry],
            )
        )
    )
    profile = defaults.load_profile(path, tuning.KNOBS)
    assert len(profile.entries) == 1
    monkeypatch.setattr(tuning, "overlay_profile", lambda: profile)
    ctx = tuning.TuningContext(
        tuning.HardwareProfile(**runs[1]["tuning"]["context"]["hardware"]),
        "0.32.4",
        "26.5.2",
        tuning.ModelProfile("qwen3_5_moe_text", 2048, 512, 256, 8, 4, 64, (), "affine"),
        mode="mtp",
        has_mtp_head=True,
        rapid_mlx=__version__,
    )
    decision = tuning.resolve("mtp_max_depth", ctx, environ={})
    assert decision == tuning.Decision(
        3, "overlay:" + entry["id"], profile_version="comparison.1"
    )
    for changed in (
        replace(ctx, hardware=replace(ctx.hardware, gpu_cores=79)),
        replace(ctx, hardware=replace(ctx.hardware, memory_size=None)),
        replace(ctx, mlx="0.32.5"),
        replace(ctx, rapid_mlx="0.1"),
        replace(ctx, model=replace(ctx.model, hidden_size=4096)),
        replace(ctx, width=2),
    ):
        assert not tuning.resolve(
            "mtp_max_depth", changed, environ={}
        ).source.startswith("overlay:")


@pytest.mark.parametrize("change", ["runtime", "profiles", "geometry"])
def test_distiller_v2_facts_must_match(change):
    runs = paired()
    ctx = runs[1]["tuning"]["context"]
    if change == "runtime":
        ctx["rapid_mlx"] = "0.1"
    elif change == "profiles":
        ctx["profiles"]["overlay"] = "other.1"
    else:
        ctx["model"]["num_experts"] = 0
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]


def test_schema_preserves_full_v2_profile_sources_and_versions():
    run = sample()
    ctx = run["tuning"]["context"]
    assert ctx["rapid_mlx"] == __version__
    assert set(ctx["profiles"]) == {"builtin", "overlay"}
    decision = ctx["knobs"]["mtp_max_depth"]
    decision.update(source="overlay:" + "a" * 128, profile_version="test.1")
    BenchmarkRunValidator().validate(run)
    for location in (ctx["profiles"], decision):
        bad = copy.deepcopy(run)
        target = (
            bad["tuning"]["context"]["profiles"]
            if location is ctx["profiles"]
            else bad["tuning"]["context"]["knobs"]["mtp_max_depth"]
        )
        target["builtin" if location is ctx["profiles"] else "profile_version"] = (
            "private/path"
        )
        with pytest.raises(ValueError):
            BenchmarkRunValidator().validate(bad)


def test_comparison_environment_restores_even_after_failure(monkeypatch):
    import os

    key = "RAPID_MLX_BENCHMARK_KNOBS"
    monkeypatch.setenv(key, "previous")
    with (
        pytest.raises(RuntimeError),
        comparison.configuration_environment({"mtp_max_depth": 2}),
    ):
        assert json.loads(os.environ[key]) == {"mtp_max_depth": 2}
        raise RuntimeError("cancelled")
    assert os.environ[key] == "previous"
    with comparison.configuration_environment({}):
        assert key not in os.environ
    assert os.environ[key] == "previous"


@pytest.mark.parametrize(
    "raw", ['{"mtp_max_depth":2}', "null", "[]", "{", '{"mtp_max_depth":99}']
)
def test_registered_benchmark_overrides_obey_capabilities(raw):
    context = sample()["tuning"]["context"]
    ctx = tuning.TuningContext(
        tuning.HardwareProfile(**context["hardware"]),
        context["mlx"],
        context["macos"],
        tuning.ModelProfile(**context["model"]),
        mode="mtp",
        has_mtp_head=True,
    )
    env = {"RAPID_MLX_BENCHMARK_KNOBS": raw, "RAPID_MLX_MTP_MAX_DEPTH": "1"}
    decision = tuning.resolve("mtp_max_depth", ctx, environ=env)
    assert decision.value == (2 if raw == '{"mtp_max_depth":2}' else 1)
    assert decision.source == ("benchmark" if decision.value == 2 else "env")
    from dataclasses import replace

    assert (
        tuning.resolve(
            "mtp_max_depth", replace(ctx, has_mtp_head=False), environ=env
        ).source
        == "capability"
    )


def test_single_configuration_preserves_original_runner(monkeypatch, tmp_path):
    from rapid_mlx.community_bench import local_runner
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    expected = sample()
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(local_runner, "run_local", run)
    assert (
        comparison.run_comparison("model", configs=1, archive=LocalRunArchive(tmp_path))
        is expected
    )
    assert len(calls) == 1 and "tuning_observation" not in calls[0]
    for configs in (0, 17):
        with pytest.raises(ValueError, match="configs"):
            comparison.run_comparison("model", configs=configs)


def test_neighbor_boundaries_and_rotation():
    context = sample()["tuning"]["context"]
    context["knobs"]["mtp_max_depth"]["value"] = 1
    candidates = list(comparison.candidate_values(context))
    assert {"mtp_max_depth": 2} in candidates
    context["knobs"]["mtp_max_depth"]["value"] = 4
    candidates = list(comparison.candidate_values(context))
    assert {"mtp_max_depth": 3} in candidates
    names = {next(iter(c)) for c in candidates}
    choices = {
        next(iter(next(comparison.candidate_values(context, salt=salt))))
        for salt in range(len(names))
    }
    assert choices == names
    context["knobs"]["mtp_max_depth"]["value"] = 99
    assert all("mtp_max_depth" not in c for c in comparison.candidate_values(context))
    context["mode"] = "plain"
    candidates = list(comparison.candidate_values(context))
    assert all("mtp_max_depth" not in c for c in candidates)
    context["knobs"]["qwen35_residual_norm"]["source"] = "default"
    assert any(
        "qwen35_residual_norm" in c for c in comparison.candidate_values(context)
    )


def test_distiller_accepts_boolean_neighbor_and_bundled_submission(tmp_path, capsys):
    runs = paired()
    for run in runs[1::2]:
        run["tuning"]["overrides"]["mtp_max_depth"] = 1
        run["tuning"]["context"]["knobs"]["qwen35_affine4_bank_views"].update(
            value=False, source="benchmark"
        )
    assert (
        tuning_distill.distill(runs, as_of=NOW)["entries"][0]["knobs"][
            "qwen35_affine4_bank_views"
        ]
        is False
    )
    first = runs[0]
    first["paired_runs"] = runs[1:]
    file = tmp_path / "paired.json"
    file.write_text(json.dumps(first))
    assert tuning_distill.main([str(file), "--as-of", NOW.isoformat()]) == 0
    assert json.loads(capsys.readouterr().out)["entries"]


def test_preview_projects_all_paired_runs_and_rejects_nested_bundles():
    from rapid_mlx.community_bench.atomic_upload import preview_run

    first, second = sample(), sample()
    second["model"]["components"][0]["source"]["local_path"] = "/private/local/model"
    first["paired_runs"] = [second]
    from rapid_mlx.community_bench.publication import project_run_for_publication

    projected, hidden = project_run_for_publication(first)
    preview = preview_run(projected, install_id="a" * 12)
    wire = json.loads(preview["payload_json"])
    assert len(wire["paired_runs"]) == 1
    assert (
        "local_path" not in wire["paired_runs"][0]["model"]["components"][0]["source"]
    )
    assert any(
        f["path"].startswith("paired_runs[0].")
        for f in [fact.as_dict() for fact in hidden]
    )
    assert (
        second["model"]["components"][0]["source"]["local_path"]
        == "/private/local/model"
    )
    second["paired_runs"] = [sample()]
    with pytest.raises(ValueError):
        BenchmarkRunValidator().validate(first)


def test_cli_configs_one_and_default_help(monkeypatch, tmp_path, capsys):
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    parser = build_parser()
    assert parser.parse_args(["benchmark", "run", "model"]).configs == 1
    seen = []
    monkeypatch.setattr(
        cli.LocalRunArchive, "default", lambda: LocalRunArchive(tmp_path)
    )
    monkeypatch.setattr(cli, "run_local", lambda *a, **kw: seen.append(kw) or sample())
    args = parser.parse_args(["benchmark", "run", "model", "--configs", "1"])
    assert cli._benchmark_command(args) == 0
    assert len(seen) == 1 and "tuning_observation" not in seen[0]
    assert "Comparing" not in capsys.readouterr().err


def test_requests_preserve_options_and_record_rotation(monkeypatch, tmp_path):
    import os

    from rapid_mlx.community_bench.acceleration import AccelerationRequest
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    requests = []

    def measure(*args, **kwargs):
        request = kwargs["acceleration"]
        requests.append(request)
        values = json.loads(os.environ.get("RAPID_MLX_BENCHMARK_KNOBS", "{}"))
        run = sample()
        for name, value in values.items():
            run["tuning"]["context"]["knobs"][name].update(
                value=value, source="benchmark"
            )
        run["tuning"]["overrides"]["mtp_max_depth"] = values.get("mtp_max_depth", 3)
        return run

    request = AccelerationRequest(
        overrides={
            "prefill_step_size": 512,
            "speculative_config": json.dumps(
                {"method": "mtp", "model": "draft", "num_speculative_tokens": 3}
            ),
        }
    )
    # Exercise depth override merging regardless of its registry position.
    context = comparison.planning_context("model", request)
    names = list(
        dict.fromkeys(name for c in comparison.candidate_values(context) for name in c)
    )
    salt = names.index("mtp_max_depth")
    from rapid_mlx.community_bench import local_runner

    monkeypatch.setattr(local_runner, "run_local", measure)
    monkeypatch.setattr(comparison, "measure_configuration", measure)
    result = comparison.run_comparison(
        "model",
        configs=2,
        salt=salt,
        archive=LocalRunArchive(tmp_path),
        acceleration=request,
    )
    assert result["measured_configs"] == 2
    for run in result["configurations"]:
        choice = run["tuning"]["selection"]
        assert choice["salt"] == salt
        assert choice["changed_knobs"] == (
            [] if run["experiment"]["arm_id"] == "default" else ["mtp_max_depth"]
        )
    for candidate in requests:
        assert candidate.overrides["prefill_step_size"] == 512
        assert json.loads(candidate.overrides["speculative_config"])["model"] == "draft"
    assert (
        json.loads(requests[1].overrides["speculative_config"])[
            "num_speculative_tokens"
        ]
        == 2
    )
    assert "RAPID_MLX_BENCHMARK_KNOBS" not in os.environ


@pytest.mark.parametrize("failed", [False, True])
def test_configuration_worker_forwards_and_closes(monkeypatch, failed):
    from rapid_mlx.community_bench import local_runner

    messages = []
    sender = SimpleNamespace(
        send=messages.append, close=lambda: messages.append("closed")
    )
    monkeypatch.setattr(local_runner, "_enter_worker_lifetime", lambda *a, **kw: None)

    def run(*args, **kwargs):
        kwargs["progress"]("loading")
        kwargs["events"]({"kind": "ready"})
        if failed:
            raise RuntimeError("recorded failure")
        return sample()

    monkeypatch.setattr(local_runner, "run_local", run)
    comparison._configuration_worker("model", {}, sender, object())
    assert messages[:2] == [("progress", "loading"), ("event", {"kind": "ready"})]
    assert messages[-2][0] == "ok"
    assert ("failure" in messages[-2][1]) is failed
    assert messages[-1] == "closed"


@pytest.mark.parametrize("sinks", [False, True])
def test_measure_configuration_uses_supervised_fresh_worker(monkeypatch, sinks):
    from rapid_mlx.community_bench import local_runner

    progress, events = [], []
    expected = sample()

    def run(target, args, **kwargs):
        assert target is comparison._configuration_worker
        assert args[0] == "model"
        assert args[1]["tuning_observation"]
        kwargs["on_message"]("progress", "loading")
        kwargs["on_message"]("event", {"kind": "ready"})
        return expected

    monkeypatch.setattr(local_runner, "_run_detached_worker", run)
    assert (
        comparison.measure_configuration(
            "model",
            tuning_observation=True,
            progress=progress.append if sinks else None,
            events=events.append if sinks else None,
        )
        is expected
    )
    assert progress == (["loading"] if sinks else [])
    assert events == ([{"kind": "ready"}] if sinks else [])


def _message_worker(sender, lifeline):
    from rapid_mlx.community_bench.local_runner import _enter_worker_lifetime

    _enter_worker_lifetime(lifeline, sender=sender)
    sender.send(("progress", "loading"))
    sender.send(("event", {"kind": "ready"}))
    sender.send(("ok", "complete"))
    sender.close()


def test_real_supervisor_forwards_worker_messages():
    from rapid_mlx.community_bench.local_runner import _run_detached_worker

    messages = []
    assert (
        _run_detached_worker(
            _message_worker,
            (),
            timeout_s=30,
            phase="headless configuration",
            on_message=lambda kind, value: messages.append((kind, value)),
        )
        == "complete"
    )
    assert messages == [("progress", "loading"), ("event", {"kind": "ready"})]


def test_measure_configuration_retains_failed_archive_state(monkeypatch):
    from rapid_mlx.community_bench import local_runner

    run = sample()
    monkeypatch.setattr(
        local_runner,
        "_run_detached_worker",
        lambda *a, **kw: {
            "failure": {"message": "measured failure", "run": run, "saved": True}
        },
    )
    with pytest.raises(local_runner.LocalBenchmarkError) as exc:
        comparison.measure_configuration("model")
    assert exc.value.run == run and exc.value.saved is True


def test_cli_reports_insufficient_neighbors(monkeypatch, tmp_path, capsys):
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    monkeypatch.setattr(
        cli.LocalRunArchive, "default", lambda: LocalRunArchive(tmp_path)
    )
    monkeypatch.setattr(cli, "run_local", lambda *a, **kw: sample())
    monkeypatch.setattr(
        comparison, "measure_configuration", lambda *a, **kw: loaded_sample()
    )
    monkeypatch.setattr(
        comparison, "candidate_values", lambda *a, **kw: iter([{"mtp_max_depth": 2}])
    )
    args = build_parser().parse_args(["benchmark", "run", "model", "--configs", "16"])
    assert cli._benchmark_command(args) == 0
    output = capsys.readouterr()
    assert (
        "Only 1 single-change neighbors are valid for this model/mode (requested 15)"
        in output.err
    )
    assert "Plan: 2 configs × 3 repetitions" in output.err


@pytest.mark.parametrize(
    "task", ["image_generation", "video_generation", "audio", "text_generation"]
)
def test_default_cli_without_applicable_knobs_keeps_legacy_process(
    monkeypatch, tmp_path, capsys, task
):
    import os

    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    run = sample()
    run.pop("experiment")
    if task == "text_generation":
        # Real registry decisions for an M3 text cell with no eligible knobs.
        context = tuning.TuningContext(
            tuning.HardwareProfile("applegpu_g15d", 15, "Ultra", 60, 274877906944),
            "0.32.4",
            "26.5.2",
            tuning.ModelProfile(),
        )
        run["tuning"]["context"] = json.loads(json.dumps(tuning.dump(context)))
        run["tuning"]["overrides"] = {}
        assert not list(comparison.candidate_values(run["tuning"]["context"]))
    else:
        del run["tuning"]
    monkeypatch.setattr(
        comparison, "planning_context", lambda *a: run.get("tuning", {}).get("context")
    )
    run["workload"]["task_type"] = task
    calls = []
    parent = os.getpid()

    def legacy(alias, **kwargs):
        assert os.getpid() == parent
        calls.append(kwargs)
        return run

    monkeypatch.setattr(cli, "run_local", legacy)
    monkeypatch.setattr(
        cli.LocalRunArchive, "default", lambda: LocalRunArchive(tmp_path)
    )
    monkeypatch.setattr(
        comparison,
        "measure_configuration",
        lambda *a, **kw: pytest.fail(
            "legacy default must not start comparison workers"
        ),
    )
    monkeypatch.setattr(cli, "summarize_measurements", lambda run: [])
    args = build_parser().parse_args(["benchmark", "run", "model", "--json"])
    assert args.configs == 1
    assert cli._benchmark_command(args) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == run
    assert len(calls) == 1 and "paired_runs" not in run


def test_summary_tolerates_zero_duration_measurements():
    runs = paired()
    # One candidate repetition has no decode-rate evidence at all, which leaves
    # it below the repetition floor instead of raising ZeroDivisionError.
    for m in runs[1]["measurements"]:
        m["decode_duration_ms"] = 0
    # A single unusable sample inside an otherwise valid run is ignored.
    runs[3]["measurements"][0]["decode_duration_ms"] = 0
    lines = comparison.summary_lines(runs)
    assert lines[-1].endswith("none")
    assert comparison.arm_statistics(runs)["mtp-3"]

    # Without usable default evidence there is nothing to compare against.
    for run in runs[::2]:
        for m in run["measurements"]:
            m["decode_duration_ms"] = 0
    assert "default" not in comparison.arm_statistics(runs)
    lines = comparison.summary_lines(runs)
    assert not any(line.startswith("Nearby config") for line in lines)
    assert lines[-1].endswith("none")


@pytest.mark.parametrize(
    "change,expected",
    [
        ("none", ("faster", "faster")),
        ("missing_arm", ("incomplete", "incomplete")),
        ("missing_case", ("incomplete", "incomplete")),
        ("few_reps", ("incomplete", "incomplete")),
        ("failed", ("incomplete", "incomplete")),
        ("digest", ("outputs_differ", "outputs_differ")),
        ("baseline_unstable", ("outputs_differ", "outputs_differ")),
        ("quality", ("quality_failed", "quality_failed")),
        ("below", ("no_reliable_gain", "below_threshold")),
        ("overlap", ("no_reliable_gain", "overlapping_ranges")),
        ("ttft", ("no_reliable_gain", "ttft_regressed")),
    ],
)
def test_shared_verdict_reasons(change, expected):
    runs = paired()
    if change == "missing_arm":
        runs = runs[1::2]
    elif change == "missing_case":
        runs[1]["workload"]["cases"].pop()
    elif change == "few_reps":
        runs.pop()
    elif change == "failed":
        runs[1]["outcome"]["status"] = "failed"
    elif change in {"digest", "baseline_unstable"}:
        run = runs[1] if change == "digest" else runs[0]
        run["tuning"]["outputs"]["pp512-tg128"][0] = "sha256:" + "2" * 64
    elif change == "quality":
        runs[1]["tuning"]["quality"] = {"n": 0}
    elif change == "below":
        for run in runs[1::2]:
            for m in run["measurements"]:
                m["decode_duration_ms"] *= 1.1
    elif change == "overlap":
        runs[1]["measurements"][0]["decode_duration_ms"] *= 1.2
    elif change == "ttft":
        for run in runs[1::2]:
            for m in run["measurements"]:
                m["ttft_ms"] = 21.01
    assert comparison.arm_verdict(runs, "mtp-3") == expected
    assert comparison.verdicts(runs)["mtp-3"]["reason_code"] == expected[1]
    assert bool(comparison.speed_winners(runs)) == (expected[0] == "faster")
    assert any(
        comparison.VERDICT_LABELS[expected[0]] in line
        for line in comparison.summary_lines(runs)
    )


def test_distill_uses_shared_verdict_and_quality_is_additional(monkeypatch):
    runs = paired()
    for run in runs[1::2]:
        run["tuning"]["quality"] = dict(
            n=100,
            questions_digest=DIGEST,
            plain=dict(delta_pp=0, paired_p=1),
            default=dict(delta_pp=0, paired_p=1),
        )
        run["tuning"]["outputs"]["pp512-tg128"] = ["sha256:" + "2" * 64] * 5
    assert not tuning_distill.distill(runs, as_of=NOW)["entries"]
    calls = []

    def reject(group, arm):
        calls.append(arm)
        return "no_reliable_gain", "below_threshold"

    monkeypatch.setattr(tuning_distill, "arm_verdict", reject)
    assert not tuning_distill.distill(paired(), as_of=NOW)["entries"]
    assert calls == ["mtp-3"]


@pytest.mark.parametrize(
    "failure", [None, "interrupt", "error", "saved_error", "outcome"]
)
def test_incremental_manifest_and_balanced_order(monkeypatch, tmp_path, failure):
    from rapid_mlx.community_bench.local_runner import LocalBenchmarkError
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    archive = LocalRunArchive(tmp_path)
    progress = []
    calls = []

    def measure(*a, **kw):
        manifest = json.loads(
            next((tmp_path / "experiments").glob("*.json")).read_text()
        )
        statuses = [r["status"] for r in manifest["runs"]]
        assert manifest["status"] == "running"
        assert statuses.count("completed") == len(calls)
        assert statuses.count("running") == 1
        if not calls:
            assert "Plan: 2 configs × 3 repetitions" in progress
            assert "estimated time: shown after the first repetition" in progress
            source = sample()["tuning"]["context"]["knobs"]["mtp_max_depth"][
                "source"
            ].split(":")[0]
            assert f"mtp_max_depth: 1 → 2 (current source: {source})" in progress
        calls.append(1)
        if len(calls) == 3:
            if failure == "interrupt":
                raise KeyboardInterrupt
            if failure == "error":
                raise RuntimeError("broken")
            if failure == "saved_error":
                run = sample()
                archive.save(run)
                raise LocalBenchmarkError("broken", run, saved=True)
        run = loaded_sample()
        if failure == "outcome" and len(calls) == 3:
            run["outcome"]["status"] = "failed"
            run["outcome"]["failure_code"] = "runtime_error"
        return run

    monkeypatch.setattr(comparison, "measure_configuration", measure)
    monkeypatch.setattr(
        comparison, "candidate_values", lambda *a, **kw: iter([{"mtp_max_depth": 2}])
    )

    def run():
        return comparison.run_comparison(
            "model",
            configs=2,
            archive=archive,
            current_runner=measure,
            progress=progress.append,
            salt=0,
        )

    if failure:
        with pytest.raises((KeyboardInterrupt, RuntimeError, ValueError)):
            run()
    else:
        result = run()
        assert [r["experiment"]["arm_id"] for r in result["configurations"]] == [
            "default",
            "nearby-1",
            "nearby-1",
            "default",
            "default",
            "nearby-1",
        ]
        assert result["verdicts"]["nearby-1"]["reason_code"] == "below_threshold"
    path = next((tmp_path / "experiments").glob("*.json"))
    manifest = json.loads(path.read_text())
    assert manifest["status"] == (
        "completed"
        if failure is None
        else "interrupted"
        if failure == "interrupt"
        else "failed"
    )
    assert not list(path.parent.glob("*.tmp"))
    assert any("estimated time remaining:" in line for line in progress)
    for entry in manifest["runs"]:
        for saved in entry["saved_paths"]:
            assert Path(saved).exists()
    if failure:
        assert (
            progress[-1]
            == f"Experiment {manifest['group_id']}: 2 completed runs; manifest: {path}"
        )


def test_interrupt_without_progress_reports_saved_experiment(
    monkeypatch, tmp_path, capsys
):
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    def interrupt(*a, **kw):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        comparison.run_comparison(
            "model",
            configs=2,
            archive=LocalRunArchive(tmp_path),
            current_runner=interrupt,
        )
    path = next((tmp_path / "experiments").glob("*.json"))
    assert path.stem in capsys.readouterr().err
    assert json.loads(path.read_text())["status"] == "interrupted"


@pytest.mark.parametrize(
    "code,detail,message",
    [
        (400, "schema unsupported", "unsupported/validation"),
        (422, "validation failed", "unsupported/validation"),
        (413, "too big", "too large"),
        (401, "unauthorized", "rejected"),
        (503, "unavailable", "rejected"),
        (None, "offline", "network error"),
    ],
)
def test_comparison_upload_error_classes(monkeypatch, code, detail, message):
    import io
    import urllib.error

    from rapid_mlx.community_bench import upload

    def fail(*a, **kw):
        if code is None:
            raise urllib.error.URLError(detail)
        raise urllib.error.HTTPError(
            "https://example.org", code, detail, {}, io.BytesIO(detail.encode())
        )

    monkeypatch.setattr(upload.urllib.request, "urlopen", fail)
    monkeypatch.setattr(upload, "_sleep_before_retry", lambda *a: None)
    with pytest.raises(upload.SubmitError, match=message):
        upload.post_submission({"paired_runs": [{}]}, url="https://example.org")


@pytest.mark.parametrize(
    "task,lane,cached,mode,explicit",
    [
        ("image_generation", "engine", True, "plain", False),
        ("text_generation", "dflash", True, "plain", False),
        ("text_generation", "engine", False, "plain", False),
        ("text_generation", "engine", True, "mtp", False),
        ("text_generation", "engine", True, "mtp", True),
    ],
)
def test_metadata_plan_before_loading(
    monkeypatch, tmp_path, task, lane, cached, mode, explicit
):
    import sys

    from rapid_mlx import cli, model_aliases, model_downloads
    from rapid_mlx.community_bench import acceleration, workspace

    config = dict(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
        num_experts_per_tok=8,
        quantization=dict(bits=4, group_size=64),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    downloads = []

    def download(repo, filename, **kw):
        downloads.append((repo, filename))
        return str(path)

    monkeypatch.setattr(
        workspace,
        "plan_for_alias",
        lambda a: {"model": {"task_type": task, "repo_id": "public/model"}},
    )
    monkeypatch.setattr(model_aliases, "resolve_profile", lambda a: None)
    monkeypatch.setattr(model_aliases, "resolve_subfolder", lambda a: None)
    monkeypatch.setattr(
        cli, "_resolve_checkpoint_config", lambda *a: config if cached else None
    )
    monkeypatch.setattr(model_downloads, "hf_hub_download", download)
    monkeypatch.setattr(
        acceleration,
        "resolve_benchmark_acceleration",
        lambda *a: (
            SimpleNamespace(
                lane=lane, scheduler_kwargs={"spec_decode": mode, "mtp_max_k": 2}
            ),
            SimpleNamespace(_speculative_config_explicit=explicit),
        ),
    )
    # No inference or weights are needed for planning.
    core = SimpleNamespace(
        device_info=lambda: dict(
            device_name="Apple M3 Ultra",
            architecture="applegpu_g15d",
            gpu_cores=80,
            memory_size=274877906944,
        )
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    ctx = REAL_PLANNING_CONTEXT("alias", acceleration.AccelerationRequest())
    if task != "text_generation" or lane != "engine":
        assert ctx is None and not downloads
    else:
        assert ctx["mode"] == mode
        assert downloads == ([] if cached else [("public/model", "config.json")])
        if explicit:
            assert ctx["knobs"]["mtp_max_depth"]["value"] == 2


@pytest.mark.parametrize("stage", ["post", "receipt"])
@pytest.mark.parametrize("group", [False, True])
def test_group_upload_retains_local_results_for_protocol_failures(
    monkeypatch, stage, group
):
    from rapid_mlx.community_bench import atomic_upload
    from rapid_mlx.community_bench.upload import SubmitError

    run = sample()
    if group:
        run["paired_runs"] = [sample("mtp-3")]
    preview = dict(
        target="https://example.org",
        install_id="abc",
        payload=run,
        payload_digest="digest",
        body_digest="body",
    )
    monkeypatch.setattr(atomic_upload, "preview_run", lambda *a, **kw: preview)
    monkeypatch.setattr(atomic_upload, "commit_install_id", lambda candidate: candidate)

    def fail(*a, **kw):
        raise SubmitError("invalid response")

    monkeypatch.setattr(
        atomic_upload,
        "post_submission",
        fail if stage == "post" else lambda *a, **kw: {},
    )
    if stage == "receipt":
        monkeypatch.setattr(atomic_upload, "_validated_receipt", fail)
    with pytest.raises(
        SubmitError, match="local results are kept" if group else "invalid response"
    ):
        atomic_upload.upload_run(run, assume_yes=True)


def test_gain_and_ttft_boundaries_apply_to_every_case():
    runs = [
        sample(arm, rate)
        for _ in range(3)
        for arm, rate in [("default", 100), ("candidate", 102)]
    ]
    for run in runs[1::2]:
        for m in run["measurements"]:
            m["ttft_ms"] = 21
    assert comparison.arm_verdict(runs, "candidate") == ("faster", "faster")
    for run in runs[1::2]:
        for m in run["measurements"]:
            if m["case_id"] == "pp2048-tg512":
                m["decode_duration_ms"] *= 102 / 101.99
    assert comparison.arm_verdict(runs, "candidate") == (
        "no_reliable_gain",
        "below_threshold",
    )


def test_comparison_plan_and_progress_remain_rs_tagged(monkeypatch, tmp_path, capsys):
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    monkeypatch.setattr(
        cli.LocalRunArchive, "default", lambda: LocalRunArchive(tmp_path)
    )
    monkeypatch.setattr(cli, "run_local", lambda *a, **kw: sample())
    monkeypatch.setattr(
        comparison, "measure_configuration", lambda *a, **kw: loaded_sample()
    )
    args = build_parser().parse_args(
        ["benchmark", "run", "model", "--json", "--progress", "--configs", "2"]
    )
    assert cli._benchmark_command(args) == 0
    output = capsys.readouterr()
    assert (
        json.loads(output.out)["verdicts"]["nearby-1"]["reason_code"]
        == "below_threshold"
    )
    assert "\x1ePlan: 2 configs × 3 repetitions" in output.err
    assert "\x1econfig 2/2 · repetition 2/3" in output.err
    assert all(line.startswith("\x1e") for line in output.err.split("\n") if line)


def test_repetition_floor_cannot_be_relaxed():
    assert comparison.arm_verdict(paired()[:2], "mtp-3", min_repetitions=1) == (
        "incomplete",
        "incomplete",
    )
    assert not comparison.speed_winners(paired()[:2], min_repetitions=1)


def test_loaded_current_settings_must_match_plan(monkeypatch, tmp_path, capsys):
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    def current(*a, **kw):
        run = sample()
        run["tuning"]["overrides"]["mtp_max_depth"] = 4
        return run

    monkeypatch.setattr(
        comparison, "candidate_values", lambda *a, **kw: iter([{"mtp_max_depth": 2}])
    )
    monkeypatch.setattr(
        comparison,
        "measure_configuration",
        lambda *a, **kw: pytest.fail("plan drift must stop before another load"),
    )
    with pytest.raises(ValueError, match="Loaded settings differ"):
        comparison.run_comparison(
            "model",
            configs=2,
            current_runner=current,
            archive=LocalRunArchive(tmp_path),
        )
    manifest = json.loads(next((tmp_path / "experiments").glob("*.json")).read_text())
    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["status"] == "completed"
    assert "1 completed runs" in capsys.readouterr().err


def test_every_repetition_must_load_its_planned_arm(monkeypatch, tmp_path):
    """A candidate whose override is ignored cannot be reported as faster."""
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    monkeypatch.setattr(
        comparison, "candidate_values", lambda *a, **kw: iter([{"mtp_max_depth": 2}])
    )
    monkeypatch.setattr(comparison, "measure_configuration", lambda *a, **kw: sample())
    with pytest.raises(ValueError, match="Loaded settings differ"):
        comparison.run_comparison(
            "model",
            configs=2,
            current_runner=lambda *a, **kw: sample(),
            archive=LocalRunArchive(tmp_path),
        )
    manifest = json.loads(next((tmp_path / "experiments").glob("*.json")).read_text())
    assert manifest["status"] == "failed"
    assert [r["status"] for r in manifest["runs"][:2]] == ["completed", "completed"]
