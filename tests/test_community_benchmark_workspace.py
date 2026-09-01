# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx.bench import _server
from vllm_mlx.catalog import rcj_digest
from vllm_mlx.community_bench import cli as community_cli
from vllm_mlx.community_bench import local_runner
from vllm_mlx.community_bench.benchmark_contracts import (
    BenchmarkRunValidator,
    registered_workload,
    registered_workload_history,
)
from vllm_mlx.community_bench.hardware import Hardware, Software
from vllm_mlx.community_bench.run_builder import build_run, execution_config, utc_now
from vllm_mlx.community_bench.runner import BenchResult, BucketResult, RoundResult
from vllm_mlx.community_bench.workspace import (
    LocalRunArchive,
    benchmark_catalog,
    plan_for_alias,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _png_base64(width: int, height: int) -> str:
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _mock_local_context(
    monkeypatch: pytest.MonkeyPatch, task_type: str, repo_id: str
) -> None:
    monkeypatch.setattr(
        local_runner,
        "plan_for_alias",
        lambda alias: {
            "model": {"alias": alias, "repo_id": repo_id, "task_type": task_type}
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


@pytest.mark.parametrize(
    ("packaged", "source"),
    [
        ("benchmark-run.schema.json", "benchmark-run.schema.json"),
        ("rapid-community-speed-v1.json", "protocols/rapid-community-speed-v1.json"),
        ("rapid-community-speed-v2.json", "protocols/rapid-community-speed-v2.json"),
        ("rapid-image-speed-v1.json", "protocols/rapid-image-speed-v1.json"),
        ("rapid-video-speed-v1.json", "protocols/rapid-video-speed-v1.json"),
        ("rapid-public-prompts-v1.json", "datasets/rapid-public-prompts-v1.json"),
        (
            "rapid-synthetic-token-dataset-v1.json",
            "datasets/rapid-synthetic-token-dataset-v1.json",
        ),
        (
            "rapid-synthetic-token-dataset-v2.json",
            "datasets/rapid-synthetic-token-dataset-v2.json",
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
    assert by_alias["flux2-klein-4b"]["protocol_version"] == 1
    assert by_alias["wan2.2-ti2v-5b-q8"]["protocol_id"] == "rapid-video-speed"
    assert by_alias["wan2.2-ti2v-5b-q8"]["protocol_version"] == 1
    assert by_alias["qwen3.8-27b-4bit"]["protocol_id"] == "rapid-community-speed"
    assert by_alias["qwen3.8-27b-4bit"]["protocol_version"] == 2
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
    assert registered_workload("text_generation")["protocol_version"] == 2


def test_v1_text_run_remains_visible_in_local_archive(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _text_run()
    run["workload"] = registered_workload_history("text_generation")[0]

    archive.save(run)

    assert archive.list() == [run]


def test_unknown_run_model_returns_structured_unsaved_cli_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        community_cli,
        "run_local",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            local_runner.LocalBenchmarkError(
                "unknown or unsupported benchmark model 'missing'",
                None,
                saved=False,
            )
        ),
    )
    args = SimpleNamespace(
        benchmark_action="run",
        benchmark_model="missing",
        inherit_process_group=False,
        json=True,
    )

    assert community_cli.benchmark_command(args) == 1
    assert json.loads(capsys.readouterr().err) == {
        "error": "unknown or unsupported benchmark model 'missing'",
        "saved": False,
    }


def test_run_local_translates_planning_error_without_fabricating_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_runner,
        "plan_for_alias",
        lambda alias: (_ for _ in ()).throw(ValueError(f"unknown model {alias}")),
    )

    with pytest.raises(
        local_runner.LocalBenchmarkError, match="unknown model missing"
    ) as error:
        local_runner.run_local("missing")

    assert error.value.run is None
    assert error.value.saved is False


@pytest.mark.parametrize("action", ["catalog", "plan", "results", "inspect"])
def test_non_run_cli_actions_return_structured_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
) -> None:
    class BrokenArchive:
        def list(self):
            raise OSError("archive unavailable")

        def get(self, run_id: str):
            raise ValueError(f"unknown run {run_id}")

    monkeypatch.setattr(
        community_cli.LocalRunArchive,
        "default",
        classmethod(lambda cls: BrokenArchive()),
    )
    monkeypatch.setattr(
        community_cli,
        "benchmark_catalog",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("catalog unavailable")),
    )
    monkeypatch.setattr(
        community_cli,
        "plan_for_alias",
        lambda alias: (_ for _ in ()).throw(ValueError(f"unknown model {alias}")),
    )
    args = SimpleNamespace(
        benchmark_action=action,
        benchmark_model="missing",
        memory_gib=None,
        run_id="missing-run",
        json=True,
    )

    assert community_cli.benchmark_command(args) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["saved"] is False
    assert "error" in payload


@pytest.mark.parametrize(
    ("decimal_gb", "expected_mib"),
    [(8, 7629), (12.5, 11921), (0, None), (None, None)],
)
def test_peak_memory_converts_decimal_gb_and_preserves_unknown(
    monkeypatch: pytest.MonkeyPatch,
    decimal_gb: float | None,
    expected_mib: int | None,
) -> None:
    monkeypatch.setattr(
        local_runner.requests,
        "get",
        lambda *args, **kwargs: _Response({"metal": {"peak_memory_gb": decimal_gb}}),
    )
    assert local_runner._peak_memory_mib("http://local/v1") == expected_mib


@pytest.mark.parametrize("task_type", ["image_generation", "video_generation"])
def test_unobserved_diffusion_execution_fields_remain_unknown(task_type: str) -> None:
    execution = execution_config(task_type)
    diffusion = execution["task"]["diffusion"]

    assert diffusion == {
        "attention_backend": "unknown",
        "compilation": "unknown",
        "vae_tiling": None,
        "vae_slicing": None,
    }
    assert "offload" not in execution["resources"]
    if task_type == "video_generation":
        assert execution["task"]["temporal_chunking"] == {"enabled": None}


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


def _text_run() -> dict:
    measurements = []
    for case_id, prompt_tokens, output_tokens in (
        ("pp512-tg128", 512, 128),
        ("pp2048-tg512", 2048, 512),
    ):
        for round_index in range(1, 6):
            measurements.append(
                {
                    "case_id": case_id,
                    "round_index": round_index,
                    "total_duration_ms": 100,
                    "peak_active_memory_mib": 4096,
                    "completed": True,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "ttft_ms": 10,
                    "decode_duration_ms": 90,
                }
            )
    return build_run(
        repo_id="mlx-community/example-text-model",
        task_type="text_generation",
        hardware=Hardware("Apple M4 Pro", 24, 12, 16),
        software=Software("15.6", "0.13.2", "0.32.1", "3.12.1"),
        started_at=utc_now(),
        measurements=measurements,
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


def test_completed_run_can_mark_peak_memory_unavailable() -> None:
    run = _image_run()
    run["measurements"][0]["peak_active_memory_mib"] = None
    BenchmarkRunValidator().validate(run)


def test_registered_text_run_rejects_actual_token_count_drift() -> None:
    run = _text_run()
    run["measurements"][0]["prompt_tokens"] = 510
    with pytest.raises(ValueError, match="target_prompt_tokens"):
        BenchmarkRunValidator().validate(run)


def test_reported_zero_token_count_is_never_replaced_by_protocol_target() -> None:
    from vllm_mlx.community_bench.runner import _reported_token_count

    assert _reported_token_count(0, 512) == 0
    assert _reported_token_count(None, 512) == 512
    with pytest.raises(RuntimeError, match="requires observed token counters"):
        _reported_token_count(None, 512, require_observed=True)


def test_run_local_archives_registered_token_drift_as_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "text_generation", "mlx-community/example-text-model"
    )
    measurements = _text_run()["measurements"]
    measurements[0]["prompt_tokens"] = 510

    async def drifted_measurements(repo_id: str):
        return measurements, 32768

    monkeypatch.setattr(local_runner, "_text_measurements", drifted_measurements)

    with pytest.raises(local_runner.LocalBenchmarkError, match="target_prompt_tokens"):
        local_runner.run_local("example-text", archive=archive)

    assert archive.list()[0]["outcome"] == {
        "status": "failed",
        "failure_code": "runtime_error",
    }


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
        lambda alias, **kwargs: (_ for _ in ()).throw(MemoryError("secret path")),
    )

    with pytest.raises(local_runner.LocalBenchmarkError) as failure:
        local_runner.run_local("example", archive=archive)

    saved = archive.list()
    assert saved == [failure.value.run]
    assert saved[0]["outcome"] == {"status": "failed", "failure_code": "runtime_oom"}
    assert "secret path" not in json.dumps(saved[0])


def test_archive_failure_reports_execution_and_persistence_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_local_context(
        monkeypatch, "image_generation", "mlx-community/example-image-model"
    )
    monkeypatch.setattr(
        local_runner,
        "_run_image",
        lambda alias, **kwargs: (_ for _ in ()).throw(RuntimeError("generation broke")),
    )

    class BrokenArchive:
        def save(self, run: dict) -> None:
            raise OSError("disk full")

    with pytest.raises(local_runner.LocalBenchmarkError) as error:
        local_runner.run_local("example-image", archive=BrokenArchive())

    assert error.value.saved is False
    assert error.value.run is not None
    assert "generation broke" in str(error.value)
    assert "failed outcome could not be saved: disk full" in str(error.value)


def test_completed_run_persistence_failure_does_not_fabricate_failed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_local_context(
        monkeypatch, "image_generation", "mlx-community/example-image-model"
    )
    monkeypatch.setattr(
        local_runner,
        "_run_image",
        lambda alias, **kwargs: _image_run()["measurements"],
    )

    class BrokenArchive:
        def __init__(self) -> None:
            self.attempts: list[dict] = []

        def save(self, run: dict) -> None:
            self.attempts.append(run)
            raise OSError("disk full")

    archive = BrokenArchive()
    with pytest.raises(local_runner.LocalBenchmarkError) as error:
        local_runner.run_local("example-image", archive=archive)

    assert error.value.saved is False
    assert error.value.run["outcome"] == {"status": "completed"}
    assert archive.attempts == [error.value.run]
    assert "completed but result could not be saved: disk full" in str(error.value)


def test_machine_probe_failure_is_archived_without_traceback_or_fake_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "image_generation", "mlx-community/example-image-model"
    )
    monkeypatch.setattr(
        local_runner,
        "collect",
        lambda: (_ for _ in ()).throw(RuntimeError("sysctl unavailable")),
    )
    monkeypatch.setattr(
        local_runner,
        "_run_image",
        lambda alias, **kwargs: pytest.fail(
            "executor must not start after probe failure"
        ),
    )

    with pytest.raises(
        local_runner.LocalBenchmarkError, match="sysctl unavailable"
    ) as error:
        local_runner.run_local("example-image", archive=archive)

    failed = error.value.run
    assert failed["outcome"] == {
        "status": "failed",
        "failure_code": "machine_probe_failed",
    }
    assert "machine" not in failed  # Never fabricate an atomic machine identity.
    assert failed["execution"]["task_type"] == "image_generation"
    assert archive.list() == [failed]


def test_completed_run_cannot_omit_atomic_machine_identity() -> None:
    run = _image_run()
    del run["machine"]
    with pytest.raises(ValueError, match="machine"):
        BenchmarkRunValidator().validate(run)


def test_run_local_executes_image_protocol_and_excludes_warmup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "image_generation", "mlx-community/example-image-model"
    )
    calls: list[dict] = []

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        assert alias == "example-image"
        assert kwargs["isolate_process_group"] is False
        yield {"base_url": "http://local/v1"}

    def post(url: str, *, json: dict, timeout: float) -> _Response:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _Response({"data": [{"b64_json": _png_base64(1024, 1024)}]})

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(local_runner.requests, "post", post)
    monkeypatch.setattr(
        local_runner.requests,
        "get",
        lambda *args, **kwargs: _Response({"metal": {"peak_memory_gb": 8}}),
    )
    timings = iter((0.0, 1.0, 2.0, 4.5))
    monkeypatch.setattr(local_runner.time, "perf_counter", lambda: next(timings))

    run = local_runner.run_local(
        "example-image", archive=archive, inherit_process_group=True
    )

    assert len(calls) == 2  # one warmup plus one measured round
    assert calls[0]["url"] == "http://local/v1/images/generations"
    assert calls[0]["json"] == {
        "model": "example-image",
        "prompt": "A handcrafted ceramic teapot beside three red pears on a linen cloth, soft window light, neutral studio background, realistic product photography.",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
        "steps": 20,
        "guidance": 3.5,
        "seed": 12648430,
    }
    assert run["measurements"] == [
        {
            "case_id": "t2i-1024-square",
            "round_index": 1,
            "total_duration_ms": 2500.0,
            "peak_active_memory_mib": 7629,
            "completed": True,
            "image_count": 1,
            "width": 1024,
            "height": 1024,
        }
    ]
    assert archive.get(run["run_id"]) == run


def test_run_local_stops_when_image_warmup_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "image_generation", "mlx-community/example-image-model"
    )
    calls = 0

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        yield {"base_url": "http://local/v1"}

    def post(*args, **kwargs) -> _Response:
        nonlocal calls
        calls += 1
        return _Response({"cancelled": True})

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(local_runner.requests, "post", post)

    with pytest.raises(local_runner.LocalBenchmarkError) as error:
        local_runner.run_local("example-image", archive=archive)

    assert calls == 1
    assert error.value.run["outcome"] == {
        "status": "cancelled",
        "failure_code": "user_cancelled",
    }
    assert archive.list() == [error.value.run]


def test_run_local_rejects_wrong_image_artifact_dimensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "image_generation", "mlx-community/example-image-model"
    )

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        yield {"base_url": "http://local/v1"}

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(
        local_runner.requests,
        "post",
        lambda *args, **kwargs: _Response(
            {"data": [{"b64_json": _png_base64(512, 512)}]}
        ),
    )

    with pytest.raises(local_runner.LocalBenchmarkError, match="512x512") as error:
        local_runner.run_local("example-image", archive=archive)

    assert error.value.run["outcome"]["status"] == "failed"


def test_run_local_executes_video_protocol_and_polls_to_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "video_generation", "mlx-community/example-video-model"
    )
    serve_options: dict = {}
    posts: list[dict] = []
    artifacts: list[tuple] = []
    job_states = iter(("running", "completed"))

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        serve_options.update(kwargs)
        yield {"base_url": "http://local/v1"}

    def post(url: str, *, data: dict, timeout: float) -> _Response:
        posts.append({"url": url, "data": data, "timeout": timeout})
        return _Response({"id": "job-1", "status": "queued"})

    def get(url: str, *, timeout: float) -> _Response:
        if url.endswith("/status"):
            return _Response({"metal": {"peak_memory_gb": 12.5}})
        status = next(job_states)
        return _Response(
            {
                "id": "job-1",
                "status": status,
                "size": "832x480",
                "frames": 81,
                "fps": 24,
            }
        )

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(local_runner.requests, "post", post)
    monkeypatch.setattr(local_runner.requests, "get", get)
    monkeypatch.setattr(
        local_runner,
        "_validated_video_artifact",
        lambda base_url, job_id, **shape: artifacts.append((base_url, job_id, shape)),
    )
    monkeypatch.setattr(local_runner.time, "sleep", lambda seconds: None)
    timings = iter((10.0, 15.0))
    monkeypatch.setattr(local_runner.time, "perf_counter", lambda: next(timings))

    run = local_runner.run_local("example-video", archive=archive)

    assert serve_options["extra_env"] == {"RAPID_MLX_WAN_STEPS": "20"}
    assert posts == [
        {
            "url": "http://local/v1/videos",
            "data": {
                "model": "example-video",
                "prompt": "A wide coastal cliff at sunrise as the camera slowly moves forward above the grass, ocean waves below, stable horizon, natural cinematic light.",
                "size": "832x480",
                "frames": "81",
                "fps": "24",
                "seed": "12648430",
                "guidance_scale": "5.0",
            },
            "timeout": 30,
        }
    ]
    assert artifacts == [
        (
            "http://local/v1",
            "job-1",
            {"width": 832, "height": 480, "frames": 81, "fps": 24.0},
        )
    ]
    assert run["measurements"][0] == {
        "case_id": "t2v-480p-81f",
        "round_index": 1,
        "total_duration_ms": 5000.0,
        "peak_active_memory_mib": 11921,
        "completed": True,
        "frames": 81,
        "width": 832,
        "height": 480,
    }


def test_video_artifact_is_downloaded_and_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContentResponse(_Response):
        def iter_content(self, *, chunk_size: int):
            assert chunk_size == 1024 * 1024
            yield b"small-mp4-fixture"

    content_response = ContentResponse({})

    def get(url: str, *, stream: bool, timeout: float) -> ContentResponse:
        assert url == "http://local/v1/videos/job-1/content"
        assert stream is True
        assert timeout == 60
        return content_response

    def probe(path: str) -> tuple[int, int, int, float]:
        assert Path(path).read_bytes() == b"small-mp4-fixture"
        return 832, 480, 81, 24.0

    monkeypatch.setattr(local_runner.requests, "get", get)
    monkeypatch.setattr(local_runner, "_probe_video_artifact", probe)

    local_runner._validated_video_artifact(
        "http://local/v1",
        "job-1",
        width=832,
        height=480,
        frames=81,
        fps=24,
    )
    assert content_response.closed is True


def test_video_artifact_probe_has_a_hard_deadline(tmp_path: Path) -> None:
    artifact = tmp_path / "untrusted.mp4"
    artifact.write_bytes(b"not-an-mp4")

    with pytest.raises(TimeoutError, match="probe exceeded its hard deadline"):
        local_runner._probe_video_artifact(str(artifact), timeout_s=0)


def test_video_artifact_probe_returns_worker_validation_error(tmp_path: Path) -> None:
    artifact = tmp_path / "corrupt.mp4"
    artifact.write_bytes(b"not-an-mp4")

    with pytest.raises(
        RuntimeError, match=r"invalid MP4 artifact|requires rapid-mlx\[video\]"
    ):
        local_runner._probe_video_artifact(str(artifact), timeout_s=10)


@pytest.mark.parametrize("mode", ["empty", "corrupt"])
def test_video_artifact_rejects_missing_or_corrupt_content(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    class ContentResponse(_Response):
        def iter_content(self, *, chunk_size: int):
            if mode == "corrupt":
                yield b"not-an-mp4"

    monkeypatch.setattr(
        local_runner.requests,
        "get",
        lambda *args, **kwargs: ContentResponse({}),
    )
    monkeypatch.setattr(
        local_runner,
        "_probe_video_artifact",
        lambda path: (_ for _ in ()).throw(RuntimeError("invalid MP4")),
    )

    with pytest.raises(RuntimeError, match="empty MP4|invalid MP4"):
        local_runner._validated_video_artifact(
            "http://local/v1",
            "job-1",
            width=832,
            height=480,
            frames=81,
            fps=24,
        )


def test_video_artifact_continuous_stream_is_size_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContentResponse(_Response):
        def iter_content(self, *, chunk_size: int):
            yield b"123"
            yield b"456"  # Continuous progress must not bypass the total cap.

    content_response = ContentResponse({})
    monkeypatch.setattr(
        local_runner.requests,
        "get",
        lambda *args, **kwargs: content_response,
    )
    monkeypatch.setattr(local_runner, "_MAX_VIDEO_ARTIFACT_BYTES", 5)
    monkeypatch.setattr(local_runner.time, "monotonic", lambda: 0.0)

    with pytest.raises(RuntimeError, match="safety limit"):
        local_runner._validated_video_artifact(
            "http://local/v1",
            "job-1",
            width=832,
            height=480,
            frames=81,
            fps=24,
        )
    assert content_response.closed is True


def test_video_artifact_continuous_stream_has_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContentResponse(_Response):
        def iter_content(self, *, chunk_size: int):
            yield b"still-streaming"

    monkeypatch.setattr(
        local_runner.requests,
        "get",
        lambda *args, **kwargs: ContentResponse({}),
    )
    monkeypatch.setattr(local_runner, "_VIDEO_ARTIFACT_DOWNLOAD_TIMEOUT_S", 1.0)
    clock = iter((0.0, 0.1, 1.1))
    monkeypatch.setattr(local_runner.time, "monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="overall deadline"):
        local_runner._validated_video_artifact(
            "http://local/v1",
            "job-1",
            width=832,
            height=480,
            frames=81,
            fps=24,
        )


@pytest.mark.parametrize(
    ("probe_result", "message"),
    [
        ((640, 480, 81, 24.0), "640x480"),
        ((832, 480, 41, 24.0), "41 frames"),
        ((832, 480, 81, 16.0), "16 fps"),
    ],
)
def test_video_artifact_rejects_actual_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
    probe_result: tuple[int, int, int, float],
    message: str,
) -> None:
    class ContentResponse(_Response):
        def iter_content(self, *, chunk_size: int):
            yield b"mp4"

    monkeypatch.setattr(
        local_runner.requests,
        "get",
        lambda *args, **kwargs: ContentResponse({}),
    )
    monkeypatch.setattr(
        local_runner, "_probe_video_artifact", lambda path: probe_result
    )

    with pytest.raises(RuntimeError, match=message):
        local_runner._validated_video_artifact(
            "http://local/v1",
            "job-1",
            width=832,
            height=480,
            frames=81,
            fps=24,
        )


@pytest.mark.parametrize("terminal_status", ["cancelled", "canceled"])
def test_run_local_video_cancellation_is_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal_status: str
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "video_generation", "mlx-community/example-video-model"
    )

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        yield {"base_url": "http://local/v1"}

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(
        local_runner.requests,
        "post",
        lambda *args, **kwargs: _Response({"id": "job-1", "status": terminal_status}),
    )

    with pytest.raises(local_runner.LocalBenchmarkError) as error:
        local_runner.run_local("example-video", archive=archive)

    assert error.value.run["outcome"] == {
        "status": "cancelled",
        "failure_code": "user_cancelled",
    }


@pytest.mark.parametrize("terminal_status", ["failed", "expired"])
def test_run_local_video_other_terminal_errors_do_not_poll_to_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal_status: str
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "video_generation", "mlx-community/example-video-model"
    )

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        yield {"base_url": "http://local/v1"}

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(
        local_runner.requests,
        "post",
        lambda *args, **kwargs: _Response({"id": "job-1", "status": terminal_status}),
    )

    with pytest.raises(local_runner.LocalBenchmarkError) as error:
        local_runner.run_local("example-video", archive=archive)

    assert error.value.run["outcome"]["failure_code"] == "runtime_error"


def test_run_local_video_rejects_mismatched_artifact_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "video_generation", "mlx-community/example-video-model"
    )

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        yield {"base_url": "http://local/v1"}

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(
        local_runner.requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "id": "job-1",
                "status": "completed",
                "size": "832x480",
                "frames": 41,
                "fps": 24,
            }
        ),
    )

    with pytest.raises(
        local_runner.LocalBenchmarkError, match="artifact metadata"
    ) as error:
        local_runner.run_local("example-video", archive=archive)

    assert error.value.run["outcome"] == {
        "status": "failed",
        "failure_code": "runtime_error",
    }


def test_run_local_video_deadline_archives_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "video_generation", "mlx-community/example-video-model"
    )

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        yield {"base_url": "http://local/v1"}

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(
        local_runner.requests,
        "post",
        lambda *args, **kwargs: _Response({"id": "job-1", "status": "queued"}),
    )
    monkeypatch.setattr(
        local_runner.requests,
        "get",
        lambda *args, **kwargs: _Response({"id": "job-1", "status": "running"}),
    )
    monkeypatch.setattr(local_runner.time, "sleep", lambda seconds: None)
    monotonic = iter((0.0, 0.2, 0.4, 0.6))
    monkeypatch.setattr(local_runner.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(local_runner, "_VIDEO_JOB_TIMEOUT_S", 0.5)

    with pytest.raises(local_runner.LocalBenchmarkError, match="timed out") as error:
        local_runner.run_local("example-video", archive=archive)

    assert error.value.run["outcome"] == {
        "status": "failed",
        "failure_code": "timeout",
    }
    assert archive.list() == [error.value.run]


def test_run_local_converts_text_engine_result_to_atomic_measurements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vllm_mlx import engine_core
    from vllm_mlx.community_bench import runner
    from vllm_mlx.utils import tokenizer as tokenizer_module

    archive = LocalRunArchive(tmp_path)
    shutdown_calls: list[tuple[bool, bool]] = []
    executor_type = local_runner.concurrent.futures.ThreadPoolExecutor

    class RecordingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.inner = executor_type(*args, **kwargs)

        def submit(self, *args, **kwargs):
            return self.inner.submit(*args, **kwargs)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))
            self.inner.shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(
        local_runner.concurrent.futures, "ThreadPoolExecutor", RecordingExecutor
    )
    _mock_local_context(
        monkeypatch, "text_generation", "mlx-community/example-text-model"
    )

    class Engine:
        def __init__(self, model, tokenizer, *args, **kwargs) -> None:
            self.engine = SimpleNamespace(_model=model, tokenizer=tokenizer)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

    async def standardized(
        engine, tokenizer, *, sampling: str, registered_token_ids: bool
    ) -> BenchResult:
        assert sampling == "greedy"
        assert registered_token_ids is True
        short = [RoundResult(100, 200, 10, prompt_tokens=512, output_tokens=128)] * 5
        long = [RoundResult(50, 150, 20, prompt_tokens=2048, output_tokens=512)] * 5
        return BenchResult(
            short=BucketResult(short),
            long=BucketResult(long),
            peak_ram_mb=4096,
            prompt_hash="unused",
            sampling="greedy",
        )

    monkeypatch.setattr(engine_core, "AsyncEngineCore", Engine)
    monkeypatch.setattr(engine_core, "_init_mlx_step_thread", lambda: None)
    monkeypatch.setattr(
        tokenizer_module,
        "load_model_with_fallback",
        lambda repo_id: (
            SimpleNamespace(args=SimpleNamespace(max_position_embeddings=32768)),
            object(),
        ),
    )
    monkeypatch.setattr(runner, "run_standardized_bench", standardized)

    run = local_runner.run_local("example-text", archive=archive)

    assert len(run["measurements"]) == 10
    assert [(row["case_id"], row["round_index"]) for row in run["measurements"]] == [
        *(("pp512-tg128", index) for index in range(1, 6)),
        *(("pp2048-tg512", index) for index in range(1, 6)),
    ]
    assert run["measurements"][0] == {
        "case_id": "pp512-tg128",
        "round_index": 1,
        "total_duration_ms": 1280.0,
        "peak_active_memory_mib": 4096,
        "completed": True,
        "prompt_tokens": 512,
        "output_tokens": 128,
        "ttft_ms": 10,
        "decode_duration_ms": 1270.0,
    }
    assert run["execution"]["task"]["language"]["context_length"] == 32768
    assert archive.get(run["run_id"]) == run
    assert shutdown_calls == [(True, True)]


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
    monkeypatch.setattr(_server, "_terminate", lambda *args, **kwargs: None)
    monkeypatch.delenv("RAPID_MLX_WAN_STEPS", raising=False)

    with _server.serve(
        "wan2.2-ti2v-5b-q8",
        log_path=tmp_path / "server.log",
        extra_env={"RAPID_MLX_WAN_STEPS": "20"},
    ):
        pass

    assert observed["RAPID_MLX_WAN_STEPS"] == "20"
    assert "RAPID_MLX_WAN_STEPS" not in os.environ


def test_bench_server_can_inherit_supervisor_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    class Process:
        returncode = None

        def poll(self):
            return None

    def popen(*args, **kwargs):
        observed["preexec_fn"] = kwargs["preexec_fn"]
        return Process()

    def terminate(*args, **kwargs):
        observed["isolated_process_group"] = kwargs["isolated_process_group"]

    monkeypatch.setattr(_server.subprocess, "Popen", popen)
    monkeypatch.setattr(_server, "_wait_for_health", lambda *args: None)
    monkeypatch.setattr(_server, "_terminate", terminate)

    with _server.serve(
        "flux2-klein-4b",
        log_path=tmp_path / "server.log",
        isolate_process_group=False,
    ):
        pass

    assert observed == {
        "preexec_fn": None,
        "isolated_process_group": False,
    }
