# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

import pytest

from vllm_mlx.bench import _server
from vllm_mlx.catalog import rcj_digest
from vllm_mlx.community_bench import local_runner
from vllm_mlx.community_bench.benchmark_contracts import (
    BenchmarkRunValidator,
)
from vllm_mlx.community_bench.hardware import Hardware, Software
from vllm_mlx.community_bench.run_builder import build_run, utc_now
from vllm_mlx.community_bench.workspace import (
    LocalRunArchive,
    benchmark_catalog,
    plan_for_alias,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("packaged", "source"),
    [
        ("benchmark-run.schema.json", "benchmark-run.schema.json"),
        ("rapid-community-speed-v1.json", "protocols/rapid-community-speed-v1.json"),
        ("rapid-image-speed-v1.json", "protocols/rapid-image-speed-v1.json"),
        ("rapid-video-speed-v1.json", "protocols/rapid-video-speed-v1.json"),
        ("rapid-public-prompts-v1.json", "datasets/rapid-public-prompts-v1.json"),
        (
            "rapid-synthetic-token-dataset-v1.json",
            "datasets/rapid-synthetic-token-dataset-v1.json",
        ),
    ],
)
def test_packaged_benchmark_contracts_are_exact_proto_copies(
    packaged: str, source: str
) -> None:
    installed = resources.files("vllm_mlx.community_bench.contracts").joinpath(packaged)
    proto = REPO_ROOT / "proto" / "community-benchmark" / "v1" / source
    assert installed.read_bytes() == proto.read_bytes()


def test_catalog_is_model_first_and_derives_protocol_from_atomic_task() -> None:
    catalog = benchmark_catalog(memory_gib=32)
    by_alias = {model["alias"]: model for model in catalog["models"]}

    assert by_alias["flux2-klein-4b"]["protocol_id"] == "rapid-image-speed"
    assert by_alias["wan2.2-ti2v-5b-q8"]["protocol_id"] == "rapid-video-speed"
    assert by_alias["qwen3.8-27b-4bit"]["protocol_id"] == "rapid-community-speed"
    assert by_alias["qwen3.5-9b-4bit"]["focus"] is True
    assert by_alias["gemma-4-e4b-4bit"]["focus"] is True
    assert by_alias["qwen-image"]["estimated_memory_gib"] == 64
    assert by_alias["qwen-image"]["memory_fit"] == "does_not_fit"
    assert by_alias["qwen-image"]["memory_estimate_source"] == "profile_minimum"
    assert all("modality" not in model for model in catalog["models"])


def test_unresolved_alias_is_local_evidence_not_formally_comparable() -> None:
    plan = plan_for_alias("flux2-klein-4b")
    assert plan["model"]["identity_strength"] == "unresolved"
    assert plan["model"]["comparable"] is False
    assert plan["privacy"] == {"storage": "local", "uploads": False}


def _image_run() -> dict:
    return build_run(
        repo_id="mlx-community/example-image-model",
        task_type="image_generation",
        hardware=Hardware("Apple M4 Pro", 24, 12, 16),
        software=Software("15.6", "0.13.2", "0.32.1", "3.12.1"),
        started_at=utc_now(),
        measurements=[
            {
                "case_id": "t2i-1024-square",
                "round_index": 1,
                "total_duration_ms": 1234.5,
                "peak_active_memory_mib": 8192,
                "completed": True,
                "image_count": 1,
                "width": 1024,
                "height": 1024,
            }
        ],
    )


def test_archive_validates_then_writes_private_atomic_result(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path / "benchmark-home")
    run = _image_run()
    path = archive.save(run)

    assert path.stat().st_mode & 0o777 == 0o600
    assert archive.runs_dir.stat().st_mode & 0o777 == 0o700
    assert archive.get(run["run_id"]) == run
    assert archive.list() == [run]


def test_archive_skips_corrupt_rows_and_never_uploads(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    archive.runs_dir.mkdir(parents=True)
    (archive.runs_dir / "bad.json").write_text("not-json", encoding="utf-8")
    assert archive.list() == []


def test_registered_workload_cannot_be_relabeled_after_measurement() -> None:
    run = _image_run()
    run["workload"]["cases"][0]["steps"] = 21
    with pytest.raises(ValueError, match="registered workload differs"):
        BenchmarkRunValidator().validate(run)


def test_completed_run_requires_every_declared_round() -> None:
    run = _image_run()
    run["measurements"] = []
    with pytest.raises(ValueError, match="measurements"):
        BenchmarkRunValidator().validate(run)


def test_machine_profile_digest_is_recomputed() -> None:
    run = _image_run()
    run["machine"]["profile"]["memory_gib"] = 48
    with pytest.raises(ValueError, match="profile_digest"):
        BenchmarkRunValidator().validate(run)


def test_failed_attempt_is_archived_without_exception_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    monkeypatch.setattr(
        local_runner,
        "plan_for_alias",
        lambda alias: {
            "model": {
                "alias": alias,
                "repo_id": "mlx-community/example-image-model",
                "task_type": "image_generation",
            }
        },
    )
    monkeypatch.setattr(
        local_runner,
        "collect",
        lambda: (
            Hardware("Apple M4 Pro", 24, 12, 16),
            Software("15.6", "0.13.2", "0.32.1", "3.12.1"),
        ),
    )
    monkeypatch.setattr(
        local_runner,
        "_run_image",
        lambda alias: (_ for _ in ()).throw(MemoryError("secret path")),
    )

    with pytest.raises(local_runner.LocalBenchmarkError) as failure:
        local_runner.run_local("example", archive=archive)

    saved = archive.list()
    assert saved == [failure.value.run]
    assert saved[0]["outcome"] == {"status": "failed", "failure_code": "runtime_oom"}
    assert "secret path" not in json.dumps(saved[0])


def test_execution_digest_is_over_effective_task_and_resources() -> None:
    run = _image_run()
    execution = run["execution"]
    assert execution["config_digest"] == rcj_digest(
        {
            "task_type": execution["task_type"],
            "resources": execution["resources"],
            "task": execution["task"],
        }
    )


def test_bench_server_scopes_protocol_environment_to_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, str] = {}

    class Process:
        returncode = None

        def poll(self):
            return None

    def popen(*args, **kwargs):
        observed.update(kwargs["env"])
        return Process()

    monkeypatch.setattr(_server.subprocess, "Popen", popen)
    monkeypatch.setattr(_server, "_wait_for_health", lambda *args: None)
    monkeypatch.setattr(_server, "_terminate", lambda *args: None)
    monkeypatch.delenv("RAPID_MLX_WAN_STEPS", raising=False)

    with _server.serve(
        "wan2.2-ti2v-5b-q8",
        log_path=tmp_path / "server.log",
        extra_env={"RAPID_MLX_WAN_STEPS": "20"},
    ):
        pass

    assert observed["RAPID_MLX_WAN_STEPS"] == "20"
    assert "RAPID_MLX_WAN_STEPS" not in os.environ
