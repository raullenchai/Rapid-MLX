# SPDX-License-Identifier: Apache-2.0
"""Offline sweep execution, privacy and profile distillation contracts."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rapid_mlx import __version__, tuning
from rapid_mlx.catalog import rcj_digest
from rapid_mlx.community_bench import sweep
from rapid_mlx.community_bench.acceleration import AccelerationRequest
from rapid_mlx.community_bench.benchmark_contracts import BenchmarkRunValidator
from rapid_mlx.community_bench.run_builder import execution_config_digest
from scripts import tuning_distill

NOW = datetime(2026, 11, 10, 23, tzinfo=timezone.utc)
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
        run["tuning"]["outputs"]["pp512-tg128"] = ["sha256:" + "2" * 64] * 5
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
        ("overrides", "secret"),
        ("outputs", "prompt"),
    ]:
        bad = copy.deepcopy(run)
        bad["tuning"][block][key] = "private"
        with pytest.raises(ValueError):
            validator.validate(bad)
    del run["tuning"]
    validator.validate(run)


def test_candidates_come_from_registry(monkeypatch):
    context = sample()["tuning"]["context"]
    request = AccelerationRequest(
        overrides={
            "prefill_step_size": 512,
            "speculative_config": json.dumps({"method": "mtp", "model": "draft"}),
        }
    )
    arms = sweep.candidate_requests(context, request)
    assert [a for a, _ in arms] == ["mtp-off", "mtp-1", "mtp-2", "mtp-3", "mtp-4"]
    assert json.loads(arms[2][1].overrides["speculative_config"]) == dict(
        method="mtp", model="draft", num_speculative_tokens=2
    )
    assert all(r.overrides["prefill_step_size"] == 512 for _, r in arms)
    context["has_mtp_head"] = False
    assert not sweep.candidate_requests(context, request)


@pytest.mark.parametrize("repetitions", [3, 7])
def test_run_sweep_interleaves_and_archives(monkeypatch, tmp_path, repetitions):
    from rapid_mlx.community_bench import local_runner
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    observed = []

    def run_local(alias, **kwargs):
        assert kwargs["tuning_observation"]
        observed.append(kwargs["acceleration"])
        return sample()

    monkeypatch.setattr(local_runner, "run_local", run_local)
    archive = LocalRunArchive(tmp_path)
    result = sweep.run_sweep("model", repetitions=repetitions, archive=archive)
    runs = result["sweep"]
    count = 6 * repetitions
    assert len(runs) == count
    order = [r["experiment"]["arm_id"] for r in runs]
    assert order[:6] == ["default", "mtp-off", "mtp-1", "mtp-2", "mtp-3", "mtp-4"]
    assert order[6:12] == ["mtp-off", "mtp-1", "mtp-2", "mtp-3", "mtp-4", "default"]
    if repetitions == 7:
        assert order[-6:] == order[:6]
    assert len({r["experiment"]["group_id"] for r in runs}) == 1
    assert [r["experiment"]["sequence_index"] for r in runs] == list(
        range(1, count + 1)
    )
    assert len(archive.list()) == count
    assert "none" in sweep.summary_lines(runs)[-1]


@pytest.mark.parametrize(
    "repetitions,acceleration",
    [(0, None), (11, None), (3, AccelerationRequest(baseline=True))],
)
def test_sweep_rejects_invalid_options(repetitions, acceleration):
    with pytest.raises(ValueError):
        sweep.run_sweep("model", repetitions=repetitions, acceleration=acceleration)


def test_sweep_unsupported_and_no_candidates(monkeypatch, tmp_path):
    from rapid_mlx.community_bench import local_runner
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    run = sample()
    run["tuning"]["context"]["has_mtp_head"] = False
    monkeypatch.setattr(local_runner, "run_local", lambda *a, **kw: copy.deepcopy(run))
    assert (
        len(sweep.run_sweep("model", archive=LocalRunArchive(tmp_path))["sweep"]) == 3
    )
    del run["tuning"]
    with pytest.raises(ValueError, match="native text"):
        sweep.run_sweep("model", archive=LocalRunArchive(tmp_path))


def test_summary_winners_divergence():
    runs = paired()
    assert "mtp-3" in sweep.summary_lines(runs)[-1]
    runs[1]["tuning"]["outputs"]["pp512-tg128"][0] = "sha256:" + "2" * 64
    assert any("diverged" in line for line in sweep.summary_lines(runs))


def test_distiller_cli(tmp_path, capsys):
    file = tmp_path / "sweep.json"
    file.write_text(json.dumps({"sweep": paired()}))
    assert tuning_distill.main([str(file), "--as-of", NOW.isoformat()]) == 0
    assert json.loads(capsys.readouterr().out)["entries"]
    file.write_text(json.dumps(sample()))
    assert tuning_distill.main([str(file)]) == 0
    assert not json.loads(capsys.readouterr().out)["entries"]


def test_cli_sweep_entry(monkeypatch, tmp_path, capsys):
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    monkeypatch.setattr(
        cli.LocalRunArchive, "default", lambda: LocalRunArchive(tmp_path)
    )
    seen = {}

    def run(*a, **kw):
        seen.update(kw)
        return {"sweep": paired()}

    monkeypatch.setattr(sweep, "run_sweep", run)
    args = build_parser().parse_args(
        ["benchmark", "run", "model", "--sweep", "--sweep-reps", "2"]
    )
    assert cli._benchmark_command(args) == 0
    assert seen["repetitions"] == 2
    assert "Local observation" in capsys.readouterr().out


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
    monkeypatch.setattr(sweep, "_expert_quantization", lambda _: quant)
    monkeypatch.setattr(sweep, "resolve_max_depth", lambda *a, **kw: 4)
    result = SimpleNamespace(
        short=SimpleNamespace(rounds_raw=[SimpleNamespace(output_digest=DIGEST)]),
        long=SimpleNamespace(rounds_raw=[SimpleNamespace(output_digest=DIGEST)]),
    )
    spec = {"method": mode, "max_draft_tokens": 4}
    block = sweep.capture_tuning(
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
    arms = sweep.candidate_requests(context, AccelerationRequest())
    assert [name for name, _ in arms] == ["mtp-off", "mtp-1", "mtp-2"]


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
