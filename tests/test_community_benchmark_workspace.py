# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import builtins
import contextlib
import io
import json
import multiprocessing
import os
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from vllm_mlx.bench import _server
from vllm_mlx.catalog import rcj_digest
from vllm_mlx.community_bench import (
    atomic_upload,
    benchmark_contracts,
    local_runner,
    run_builder,
)
from vllm_mlx.community_bench import cli as community_cli
from vllm_mlx.community_bench import runner as bench_runner
from vllm_mlx.community_bench import upload as benchmark_upload
from vllm_mlx.community_bench import workspace as workspace_module
from vllm_mlx.community_bench.benchmark_contracts import (
    BenchmarkRunValidator,
    registered_workload,
    registered_workload_history,
)
from vllm_mlx.community_bench.hardware import Hardware, Software
from vllm_mlx.community_bench.run_builder import build_run, execution_config, utc_now
from vllm_mlx.community_bench.runner import BenchResult, BucketResult, RoundResult
from vllm_mlx.community_bench.upload import SubmitError
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


class _HTTPErrorResponse(_Response):
    status_code = 400

    def raise_for_status(self) -> None:
        response = requests.Response()
        response.status_code = self.status_code
        raise requests.HTTPError("400 Client Error", response=response)


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
        ("submission-receipt.schema.json", "submission-receipt.schema.json"),
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
    installed = resources.files("vllm_mlx.catalog.schemas").joinpath(packaged)
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
    assert plan["privacy"] == {
        "storage": "local",
        "uploads": False,
        "upload": "explicit_consent_only",
    }
    assert registered_workload("text_generation")["protocol_version"] == 2


def test_v1_text_run_remains_visible_in_local_archive(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _text_run()
    run["workload"] = registered_workload_history("text_generation")[0]

    archive.save(run)

    assert archive.list() == [run]


def test_local_archive_can_return_only_the_latest_runs(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    for index in range(3):
        run = _text_run()
        run["run_id"] = f"00000000-0000-4000-8000-{index:012d}"
        run["started_at"] = f"2026-08-{index + 1:02d}T00:00:00Z"
        run["completed_at"] = f"2026-08-{index + 1:02d}T00:01:00Z"
        archive.save(run)

    assert [run["run_id"] for run in archive.list(limit=2)] == [
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000001",
    ]


def test_execution_records_source_checkout_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    monkeypatch.setattr(run_builder, "_source_checkout_revision", lambda: revision)

    runtime = execution_config("text_generation")["runtime"]

    assert runtime["distribution"] == "source"
    assert runtime["rapid_mlx_revision"] == revision


def test_execution_records_release_without_source_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_builder, "_source_checkout_revision", lambda: None)

    runtime = execution_config("text_generation")["runtime"]

    assert runtime["distribution"] == "release"
    assert "rapid_mlx_revision" not in runtime


def test_results_cli_forwards_latest_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Archive:
        def list(self, *, limit=None):
            assert limit == 8
            return []

    monkeypatch.setattr(
        community_cli.LocalRunArchive,
        "default",
        classmethod(lambda cls: Archive()),
    )
    args = SimpleNamespace(benchmark_action="results", limit=8, json=True)

    assert community_cli.benchmark_command(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "runs": [],
        "receipts": {},
    }


def _receipt(
    run_id: str, *, already: bool = False, run_digest: str | None = None
) -> dict:
    return {
        "schema_version": 1,
        "submission_id": run_id,
        "status": "accepted",
        "already_exists": already,
        "accepted_at": "2026-09-01T20:00:00Z",
        "run_digest": run_digest or "sha256:" + "a" * 64,
        "contributor": {"name": "rapid-silver-otter", "tag": "abc"},
    }


def test_atomic_upload_decline_has_no_disk_or_network_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _text_run()
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    called = []
    monkeypatch.setattr(
        atomic_upload,
        "post_submission",
        lambda *args, **kwargs: called.append(1),
    )
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "a" * 12)
    output = io.StringIO()

    result = atomic_upload.upload_run(
        run,
        stdin=io.StringIO("n\n"),
        stdout=output,
        url="https://rapidmlx.com/api/benchmarks/atomic",
    )

    assert result is None
    assert called == []
    assert not (tmp_path / "bench-install-id").exists()
    assert "observes the source IP" in output.getvalue()
    assert "does not put it in the benchmark record" in output.getvalue()
    exact_body = benchmark_upload.submission_body(
        {**run, "install_id": "a" * 12}
    ).decode()
    assert exact_body in output.getvalue()


def test_atomic_upload_rejects_cleartext_explicit_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    sent = []
    monkeypatch.setattr(
        atomic_upload,
        "post_submission",
        lambda *args, **kwargs: sent.append(args),
    )

    with pytest.raises(SubmitError, match="must be an https:// URL"):
        atomic_upload.upload_run(
            _text_run(),
            assume_yes=True,
            url="http://collector.example/submit",
        )
    assert sent == []
    assert not (tmp_path / "bench-install-id").exists()


def test_atomic_upload_sends_validated_run_and_requires_matching_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _text_run()
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    sent = {}

    def post(payload, **kwargs):
        assert kwargs["url"] == "https://rapidmlx.com/api/benchmarks/atomic"
        sent.update(payload)
        return {
            "ok": True,
            "receipt": _receipt(
                run["run_id"], run_digest=atomic_upload.atomic_run_digest(payload)
            ),
        }

    monkeypatch.setattr(atomic_upload, "post_submission", post)
    monkeypatch.setattr(
        atomic_upload,
        "board_url",
        lambda: "https://rapidmlx.com/api/benchmarks",
    )

    acceptance = atomic_upload.upload_run(run, assume_yes=True)

    assert acceptance is not None
    assert acceptance.receipt == _receipt(
        run["run_id"], run_digest=atomic_upload.atomic_run_digest(sent)
    )
    assert acceptance.install_id == sent["install_id"]
    assert len(sent["install_id"]) == 12
    assert "install_id" not in run
    assert sent
    BenchmarkRunValidator().validate(sent)


def test_atomic_preview_is_the_exact_later_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _text_run()
    run["measurements"][0]["total_duration_ms"] = 100.25
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    preview = atomic_upload.preview_run(run)
    assert not (tmp_path / "bench-install-id").exists()
    assert preview["payload_json"].encode() == benchmark_upload.submission_body(
        preview["payload"]
    )

    def post(payload, **kwargs):
        assert payload == preview["payload"]
        assert kwargs["url"] == preview["target"]
        return {
            "receipt": _receipt(
                run["run_id"], run_digest=atomic_upload.atomic_run_digest(payload)
            )
        }

    monkeypatch.setattr(atomic_upload, "post_submission", post)
    atomic_upload.upload_run(
        run,
        assume_yes=True,
        approved_install_id=preview["install_id"],
        approved_payload_digest=preview["payload_digest"],
        approved_body_digest=preview["body_digest"],
        approved_target=preview["target"],
    )


def test_atomic_run_digest_uses_ingestion_service_number_format() -> None:
    render = atomic_upload._ecmascript_number
    assert render(100.0) == "100"
    assert render(-0.0) == "0"
    assert render(0.00123) == "0.00123"
    assert render(1e-6) == "0.000001"
    assert render(1e-7) == "1e-7"
    assert render(1e20) == "100000000000000000000"
    assert render(1e21) == "1e+21"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"ok": True}, "without a submission receipt"),
        ({"receipt": {"schema_version": 1}}, "invalid submission receipt"),
    ],
)
def test_atomic_upload_rejects_missing_or_invalid_receipt(
    response: dict, message: str
) -> None:
    with pytest.raises(SubmitError, match=message):
        atomic_upload._validated_receipt(
            response, _text_run()["run_id"], "sha256:" + "a" * 64
        )


def test_atomic_upload_rejects_receipt_for_different_run() -> None:
    run = _text_run()
    receipt = _receipt(
        "00000000-0000-4000-8000-000000000099",
        run_digest="sha256:" + "a" * 64,
    )
    with pytest.raises(SubmitError, match="uploaded run"):
        atomic_upload._validated_receipt(
            {"receipt": receipt}, run["run_id"], receipt["run_digest"]
        )


def test_atomic_digest_rejects_non_finite_and_unsupported_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        atomic_upload.atomic_run_digest({"value": float("nan")})
    with pytest.raises(TypeError, match="unsupported benchmark payload value"):
        atomic_upload.atomic_run_digest({"value": object()})


def test_atomic_upload_rejects_receipt_for_different_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _text_run()
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    monkeypatch.setattr(
        atomic_upload,
        "post_submission",
        lambda payload, **kwargs: {"receipt": _receipt(run["run_id"])},
    )

    with pytest.raises(SubmitError, match="uploaded payload"):
        atomic_upload.upload_run(run, assume_yes=True)


def test_atomic_upload_aborts_if_approved_install_id_loses_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _text_run()
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    sent = []
    monkeypatch.setattr(atomic_upload, "commit_install_id", lambda candidate: "b" * 12)
    monkeypatch.setattr(
        atomic_upload,
        "post_submission",
        lambda *args, **kwargs: sent.append(args),
    )

    with pytest.raises(SubmitError, match="install id changed"):
        atomic_upload.upload_run(
            run,
            assume_yes=True,
            approved_install_id="a" * 12,
        )
    assert sent == []


def test_atomic_upload_aborts_if_destination_changes_after_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _text_run()
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    monkeypatch.setenv(
        "RAPID_MLX_BENCH_BOARD_URL", "https://first.example/api/benchmarks/atomic"
    )
    preview = atomic_upload.preview_run(run)
    monkeypatch.setenv(
        "RAPID_MLX_BENCH_BOARD_URL", "https://second.example/api/benchmarks/atomic"
    )
    sent = []
    monkeypatch.setattr(
        atomic_upload,
        "post_submission",
        lambda *args, **kwargs: sent.append(args),
    )

    with pytest.raises(SubmitError, match="destination changed"):
        atomic_upload.upload_run(
            run,
            assume_yes=True,
            approved_install_id=preview["install_id"],
            approved_payload_digest=preview["payload_digest"],
            approved_body_digest=preview["body_digest"],
            approved_target=preview["target"],
        )
    assert sent == []
    assert not (tmp_path / "bench-install-id").exists()


def test_atomic_upload_binds_exact_serialized_body_not_only_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _text_run()
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    preview = atomic_upload.preview_run(run)
    reordered = {key: run[key] for key in reversed(run)}
    sent = []
    monkeypatch.setattr(
        atomic_upload,
        "post_submission",
        lambda *args, **kwargs: sent.append(args),
    )

    with pytest.raises(SubmitError, match="serialized benchmark changed"):
        atomic_upload.upload_run(
            reordered,
            assume_yes=True,
            approved_install_id=preview["install_id"],
            approved_payload_digest=preview["payload_digest"],
            approved_body_digest=preview["body_digest"],
            approved_target=preview["target"],
        )
    assert sent == []
    assert not (tmp_path / "bench-install-id").exists()


def test_share_cli_rejects_archive_changed_after_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _text_run()
    archive.save(run)
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    preview = atomic_upload.preview_run(run)

    changed = json.loads(json.dumps(run))
    changed["measurements"][0]["total_duration_ms"] += 0.5
    archive.save(changed)
    sent = []
    monkeypatch.setattr(
        community_cli.LocalRunArchive,
        "default",
        classmethod(lambda cls: archive),
    )
    monkeypatch.setattr(
        atomic_upload,
        "post_submission",
        lambda *args, **kwargs: sent.append(args),
    )
    args = SimpleNamespace(
        benchmark_action="share",
        run_id=run["run_id"],
        yes=True,
        json=True,
        preview=False,
        install_id=preview["install_id"],
        payload_digest=preview["payload_digest"],
        body_digest=preview["body_digest"],
        target=preview["target"],
    )

    assert community_cli.benchmark_command(args) == 1
    assert "changed after preview" in capsys.readouterr().err
    assert sent == []
    assert not (tmp_path / "bench-install-id").exists()


def test_install_id_commit_is_stable_across_same_process_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    (tmp_path / "bench-install-id").write_text("malformed\n")
    candidates = [f"{index:012x}" for index in range(16)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        settled = list(pool.map(benchmark_upload.commit_install_id, candidates))

    assert len(set(settled)) == 1
    assert settled[0] in candidates
    assert (tmp_path / "bench-install-id").read_text().strip() == settled[0]
    assert list(tmp_path.glob(".bench-install-id.*.tmp")) == []


def test_install_id_cleanup_errors_do_not_reverse_a_committed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    real_flock = benchmark_upload.fcntl.flock
    real_close = benchmark_upload.os.close
    lock_fd = None

    def flaky_flock(descriptor, operation):
        nonlocal lock_fd
        if operation == benchmark_upload.fcntl.LOCK_EX:
            lock_fd = descriptor
            return real_flock(descriptor, operation)
        raise OSError("unlock failed")

    def flaky_close(descriptor):
        real_close(descriptor)
        if descriptor == lock_fd:
            raise OSError("close reported failure")

    monkeypatch.setattr(benchmark_upload.fcntl, "flock", flaky_flock)
    monkeypatch.setattr(benchmark_upload.os, "close", flaky_close)

    assert benchmark_upload.commit_install_id("a" * 12) == "a" * 12
    assert (tmp_path / "bench-install-id").read_text().strip() == "a" * 12


def test_install_id_commit_syncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    real_fsync = benchmark_upload.os.fsync
    synced = []

    def observed_fsync(descriptor):
        synced.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        return real_fsync(descriptor)

    monkeypatch.setattr(benchmark_upload.os, "fsync", observed_fsync)

    assert benchmark_upload.commit_install_id("a" * 12) == "a" * 12
    assert synced == [False, True]


def test_install_id_commit_completes_a_partial_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    real_write = benchmark_upload.os.write
    calls = 0

    def partial_first_write(descriptor, data):
        nonlocal calls
        calls += 1
        return real_write(descriptor, data[:3] if calls == 1 else data)

    monkeypatch.setattr(benchmark_upload.os, "write", partial_first_write)

    assert benchmark_upload.commit_install_id("a" * 12) == "a" * 12
    assert calls == 2
    assert (tmp_path / "bench-install-id").read_text() == "a" * 12 + "\n"


def test_install_id_commit_handles_zero_progress_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    monkeypatch.setattr(benchmark_upload.os, "write", lambda descriptor, data: 0)

    assert benchmark_upload.commit_install_id("a" * 12) == "a" * 12
    assert not (tmp_path / "bench-install-id").exists()
    assert list(tmp_path.glob(".bench-install-id.*.tmp")) == []


def test_install_id_directory_close_error_is_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path))
    real_close = benchmark_upload.os.close

    def close_then_report_directory_error(descriptor):
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        real_close(descriptor)
        if is_directory:
            raise OSError("directory close reported failure")

    monkeypatch.setattr(benchmark_upload.os, "close", close_then_report_directory_error)

    assert benchmark_upload.commit_install_id("a" * 12) == "a" * 12
    assert (tmp_path / "bench-install-id").read_text().strip() == "a" * 12


def test_local_archive_receipt_marks_only_an_existing_run_shared(
    tmp_path: Path,
) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _text_run()
    archive.save(run)
    install_id = "012345abcdef"
    wire = {**run, "install_id": install_id}
    receipt = _receipt(run["run_id"], run_digest=atomic_upload.atomic_run_digest(wire))

    path = archive.save_receipt(receipt, install_id=install_id)

    assert path.stat().st_mode & 0o777 == 0o600
    assert archive.receipt(run["run_id"]) == receipt
    unknown = dict(receipt, submission_id="00000000-0000-4000-8000-000000000099")
    with pytest.raises(FileNotFoundError):
        archive.save_receipt(unknown, install_id=install_id)

    malformed = dict(receipt, status="maybe")
    with pytest.raises(ValueError, match="submission_receipt"):
        archive.save_receipt(malformed, install_id=install_id)

    changed = json.loads(json.dumps(run))
    changed["measurements"][0]["total_duration_ms"] += 0.5
    archive.save(changed)
    assert archive.receipt(run["run_id"]) is None


@pytest.mark.parametrize("contents", [b'{"schema_version":', b"\xff\xfe"])
def test_corrupt_optional_receipt_does_not_hide_local_runs(
    tmp_path: Path, contents: bytes
) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _text_run()
    archive.save(run)
    archive.receipts_dir.mkdir(parents=True, exist_ok=True)
    (archive.receipts_dir / f"{run['run_id']}.json").write_bytes(contents)

    assert archive.receipt(run["run_id"]) is None
    assert archive.list() == [run]


def test_optional_receipt_rejects_invalid_envelope_shapes(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _text_run()
    archive.save(run)
    archive.receipts_dir.mkdir(parents=True, exist_ok=True)
    path = archive.receipts_dir / f"{run['run_id']}.json"

    for envelope in (
        {"schema_version": 2},
        {"schema_version": 1, "install_id": 7, "receipt": {}},
        {
            "schema_version": 1,
            "install_id": "012345abcdef",
            "receipt": _receipt(run["run_id"]) | {"status": "invalid"},
        },
    ):
        path.write_text(json.dumps(envelope))
        assert archive.receipt(run["run_id"]) is None


def test_save_receipt_defensively_rejects_missing_submission_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = LocalRunArchive(tmp_path)
    monkeypatch.setattr(
        workspace_module.SubmissionReceiptValidator,
        "validate",
        lambda self, receipt: None,
    )

    with pytest.raises(ValueError, match="missing a submission id"):
        archive.save_receipt({}, install_id="012345abcdef")


def test_share_cli_saves_server_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _text_run()
    archive.save(run)
    install_id = "012345abcdef"
    wire = {**run, "install_id": install_id}
    receipt = _receipt(run["run_id"], run_digest=atomic_upload.atomic_run_digest(wire))
    acceptance = atomic_upload.AtomicUploadAcceptance(
        receipt=receipt,
        install_id=install_id,
        payload_digest=receipt["run_digest"],
    )
    monkeypatch.setattr(
        community_cli.LocalRunArchive,
        "default",
        classmethod(lambda cls: archive),
    )
    monkeypatch.setattr(
        community_cli,
        "upload_run",
        lambda local_run, **kwargs: acceptance,
    )
    args = SimpleNamespace(
        benchmark_action="share", run_id=run["run_id"], yes=True, json=True
    )

    assert community_cli.benchmark_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["uploaded"] is True
    assert payload["receipt"] == receipt
    assert archive.receipt(run["run_id"]) == receipt


def test_share_cli_reports_acceptance_when_receipt_persistence_loses_archive_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = LocalRunArchive(tmp_path)
    run = _text_run()
    archive.save(run)
    install_id = "012345abcdef"
    receipt = _receipt(
        run["run_id"],
        run_digest=atomic_upload.atomic_run_digest({**run, "install_id": install_id}),
    )
    acceptance = atomic_upload.AtomicUploadAcceptance(
        receipt=receipt,
        install_id=install_id,
        payload_digest=receipt["run_digest"],
    )

    def accepted_then_archive_changes(local_run, **kwargs):
        changed = json.loads(json.dumps(local_run))
        changed["measurements"][0]["total_duration_ms"] += 0.5
        archive.save(changed)
        return acceptance

    monkeypatch.setattr(
        community_cli.LocalRunArchive,
        "default",
        classmethod(lambda cls: archive),
    )
    monkeypatch.setattr(community_cli, "upload_run", accepted_then_archive_changes)
    args = SimpleNamespace(
        benchmark_action="share", run_id=run["run_id"], yes=True, json=True
    )

    assert community_cli.benchmark_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["uploaded"] is True
    assert payload["receipt_saved"] is False
    assert archive.receipt(run["run_id"]) is None


def test_share_cli_requires_confirmation_for_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cli_archive(monkeypatch, SimpleNamespace())
    args = SimpleNamespace(
        benchmark_action="share",
        run_id="00000000-0000-4000-8000-000000000001",
        yes=False,
        json=True,
        preview=False,
    )

    assert community_cli.benchmark_command(args) == 1
    assert "requires --yes" in capsys.readouterr().err


@pytest.mark.parametrize("json_output", [False, True])
def test_share_cli_preview_prints_exact_preview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    run = _text_run()

    class Archive:
        def get(self, run_id: str):
            return run

    _cli_archive(monkeypatch, Archive())
    monkeypatch.setattr(
        community_cli,
        "preview_run",
        lambda local_run: {"target": "https://example.test/atomic"},
    )
    args = SimpleNamespace(
        benchmark_action="share",
        run_id=run["run_id"],
        yes=False,
        json=json_output,
        preview=True,
    )

    assert community_cli.benchmark_command(args) == 0
    assert (
        json.loads(capsys.readouterr().out)["target"] == "https://example.test/atomic"
    )


def test_share_cli_text_reports_cancel_and_unsaved_existing_acceptance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _text_run()
    assert (
        community_cli._contributor_profile({"contributor": {"name": "", "tag": "abc"}})
        is None
    )

    class Archive:
        def get(self, run_id: str):
            return run

        def save_receipt(self, receipt, *, install_id):
            raise OSError("read-only archive")

    _cli_archive(monkeypatch, Archive())
    args = SimpleNamespace(
        benchmark_action="share",
        run_id=run["run_id"],
        yes=False,
        json=False,
        preview=False,
    )
    monkeypatch.setattr(community_cli, "upload_run", lambda local_run, **kwargs: None)
    assert community_cli.benchmark_command(args) == 0
    assert "Upload cancelled" in capsys.readouterr().out

    receipt = _receipt(run["run_id"], already=True)
    acceptance = atomic_upload.AtomicUploadAcceptance(
        receipt=receipt,
        install_id="012345abcdef",
        payload_digest=receipt["run_digest"],
    )
    monkeypatch.setattr(
        community_cli, "upload_run", lambda local_run, **kwargs: acceptance
    )
    args.yes = True
    assert community_cli.benchmark_command(args) == 0
    output = capsys.readouterr().out
    assert "already uploaded" in output
    assert "You contributed as rapid-silver-otter ·abc." in output
    assert (
        "https://rapidmlx.com/leaderboard/contributors/rapid-silver-otter-abc" in output
    )
    assert "local receipt could not be saved" in output

    anonymous_receipt = {**receipt, "contributor": None}
    anonymous_acceptance = atomic_upload.AtomicUploadAcceptance(
        receipt=anonymous_receipt,
        install_id="012345abcdef",
        payload_digest=anonymous_receipt["run_digest"],
    )
    monkeypatch.setattr(
        community_cli,
        "upload_run",
        lambda local_run, **kwargs: anonymous_acceptance,
    )
    assert community_cli.benchmark_command(args) == 0
    output = capsys.readouterr().out
    assert "You contributed as" not in output
    assert "View Community Benchmark: https://rapidmlx.com/leaderboard" in output


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
        def list(self, *, limit=None):
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
    (archive.runs_dir / "0000.json").write_text("[]", encoding="utf-8")
    assert archive.list() == []
    with pytest.raises(ValueError, match="not a JSON object"):
        archive.get("0000")


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


def test_completed_run_construction_failure_is_not_retried_as_failed_outcome(
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
    execution_calls = 0

    def unresolved_revision(*args, **kwargs):
        nonlocal execution_calls
        execution_calls += 1
        raise RuntimeError("could not resolve the Rapid-MLX source revision")

    monkeypatch.setattr(local_runner, "execution_config", unresolved_revision)
    monkeypatch.setattr(
        local_runner, "build_run", lambda **kwargs: pytest.fail("must not build twice")
    )
    archive = SimpleNamespace(save=lambda run: pytest.fail("must not save"))

    with pytest.raises(
        local_runner.LocalBenchmarkError,
        match="completed but result could not be constructed: could not resolve",
    ) as error:
        local_runner.run_local("example-image", archive=archive)

    assert execution_calls == 1
    assert error.value.run is None
    assert error.value.saved is False


def test_execution_and_failure_envelope_errors_are_both_reported(
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
    monkeypatch.setattr(
        local_runner,
        "build_run",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("could not resolve the Rapid-MLX source revision")
        ),
    )
    archive = SimpleNamespace(save=lambda run: pytest.fail("must not save"))

    with pytest.raises(local_runner.LocalBenchmarkError) as error:
        local_runner.run_local("example-image", archive=archive)

    assert error.value.run is None
    assert error.value.saved is False
    assert "generation broke" in str(error.value)
    assert "failed outcome could not be constructed: could not resolve" in str(
        error.value
    )


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
    # ``inherit_process_group`` is only honored for a verified dedicated
    # group leader; the pytest process shares the runner's group, so stand in
    # for the supervisor spawn topology here.
    monkeypatch.setattr(
        local_runner, "_is_dedicated_process_group_leader", lambda: True
    )

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


def test_run_local_rejects_inherited_group_without_dedicated_leader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unverified topology must fail closed before any model/server work."""

    archive = LocalRunArchive(tmp_path)
    planned: list[str] = []
    monkeypatch.setattr(
        local_runner, "_is_dedicated_process_group_leader", lambda: False
    )
    monkeypatch.setattr(
        local_runner, "plan_for_alias", lambda alias: planned.append(alias)
    )
    monkeypatch.setattr(
        local_runner,
        "collect",
        lambda: pytest.fail("machine probe ran despite unsafe topology"),
    )

    with pytest.raises(
        local_runner.LocalBenchmarkError, match="dedicated process group"
    ) as error:
        local_runner.run_local(
            "example-image", archive=archive, inherit_process_group=True
        )

    assert error.value.run is None
    assert error.value.saved is False
    assert planned == []
    assert archive.list() == []


def test_cli_run_reports_unsafe_inherit_process_group_topology(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        community_cli.LocalRunArchive,
        "default",
        classmethod(lambda cls: SimpleNamespace()),
    )
    monkeypatch.setattr(
        local_runner, "_is_dedicated_process_group_leader", lambda: False
    )
    args = SimpleNamespace(
        benchmark_action="run",
        benchmark_model="example-image",
        inherit_process_group=True,
        json=True,
    )

    assert community_cli.benchmark_command(args) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["saved"] is False
    assert "dedicated process group" in payload["error"]


def test_run_local_without_flag_never_consults_group_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The normal CLI path keeps its isolated server group unconditionally."""

    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "image_generation", "mlx-community/example-image-model"
    )
    monkeypatch.setattr(
        local_runner,
        "_is_dedicated_process_group_leader",
        lambda: pytest.fail("topology was consulted without the flag"),
    )
    observed: dict[str, bool] = {}

    def run_image(alias: str, *, isolate_process_group: bool) -> list[dict]:
        observed["isolate_process_group"] = isolate_process_group
        return [
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

    monkeypatch.setattr(local_runner, "_run_image", run_image)

    local_runner.run_local("example-image", archive=archive)

    assert observed == {"isolate_process_group": True}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
def test_inherit_process_group_verification_matches_real_topology(
    tmp_path: Path,
) -> None:
    """Drive the real leader check from both spawn topologies.

    A child sharing this process's group models direct shell-script
    invocation and must be rejected before planning; a child spawned as a
    session (and therefore group) leader models ``ProcessGroupChild.spawn``
    and must clear the gate — its failure is the ordinary planning error for
    a made-up alias, proving the gate itself passed.
    """

    script = tmp_path / "leader_probe.py"
    script.write_text(
        textwrap.dedent(
            """
            from vllm_mlx.community_bench.local_runner import (
                LocalBenchmarkError,
                _is_dedicated_process_group_leader,
                run_local,
            )

            print(f"leader={_is_dedicated_process_group_leader()}")
            try:
                run_local("model-that-does-not-exist", inherit_process_group=True)
            except LocalBenchmarkError as exc:
                print(f"error={exc}")
            """
        )
    )

    def probe(*, start_new_session: bool) -> str:
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            start_new_session=start_new_session,
            cwd=REPO_ROOT,
            env=_subprocess_env_for_this_checkout(),
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    inherited = probe(start_new_session=False)
    assert "leader=False" in inherited
    assert "dedicated process group" in inherited

    dedicated = probe(start_new_session=True)
    assert "leader=True" in dedicated
    assert "dedicated process group" not in dedicated
    assert "model-that-does-not-exist" in dedicated


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


def test_video_request_failure_preserves_server_detail(
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
        lambda *args, **kwargs: _HTTPErrorResponse(
            {"detail": "video width/height must be divisible by 64"}
        ),
    )

    with pytest.raises(
        local_runner.LocalBenchmarkError,
        match="video benchmark request failed with HTTP 400: .*divisible by 64",
    ) as error:
        local_runner.run_local("example-video", archive=archive)

    assert error.value.saved is True
    assert error.value.run["outcome"] == {
        "status": "failed",
        "failure_code": "runtime_error",
    }


def test_local_request_failure_uses_bounded_fallback_for_non_json_error() -> None:
    class NonJSONErrorResponse(_HTTPErrorResponse):
        def json(self) -> dict:
            raise ValueError("not JSON")

    with pytest.raises(
        RuntimeError,
        match="video benchmark request failed with HTTP 400: "
        "local server rejected the request",
    ):
        local_runner._raise_for_status(
            NonJSONErrorResponse({}), phase="video benchmark request"
        )


def test_video_artifact_is_downloaded_and_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def download(base_url: str, job_id: str, path: str) -> None:
        assert base_url == "http://local/v1"
        assert job_id == "job-1"
        Path(path).write_bytes(b"small-mp4-fixture")

    def probe(path: str) -> tuple[int, int, int, float]:
        assert Path(path).read_bytes() == b"small-mp4-fixture"
        return 832, 480, 81, 24.0

    monkeypatch.setattr(local_runner, "_download_video_artifact", download)
    monkeypatch.setattr(local_runner, "_probe_video_artifact", probe)

    local_runner._validated_video_artifact(
        "http://local/v1",
        "job-1",
        width=832,
        height=480,
        frames=81,
        fps=24,
    )


def test_video_download_worker_logic_streams_and_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

    artifact = tmp_path / "artifact.mp4"
    monkeypatch.setattr(local_runner.requests, "get", get)
    local_runner._download_video_artifact_unbounded(
        "http://local/v1", "job-1", str(artifact)
    )

    assert artifact.read_bytes() == b"small-mp4-fixture"
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
    def download(base_url: str, job_id: str, path: str) -> None:
        if mode == "empty":
            raise RuntimeError("video benchmark returned an empty MP4 artifact")
        Path(path).write_bytes(b"not-an-mp4")

    monkeypatch.setattr(local_runner, "_download_video_artifact", download)
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    artifact = tmp_path / "oversized.mp4"

    with pytest.raises(RuntimeError, match="safety limit"):
        local_runner._download_video_artifact_unbounded(
            "http://local/v1", "job-1", str(artifact)
        )
    assert content_response.closed is True


def test_video_artifact_download_has_process_hard_deadline(tmp_path: Path) -> None:
    artifact = tmp_path / "deadline.mp4"

    with pytest.raises(TimeoutError, match="download exceeded its hard deadline"):
        local_runner._download_video_artifact(
            "http://local/v1", "job-1", str(artifact), timeout_s=0
        )


def _detached_child_of(parent_pid: int) -> int | None:
    """Find a child of ``parent_pid`` that moved into its own process group."""

    listing = subprocess.run(
        ["pgrep", "-P", str(parent_pid)], capture_output=True, text=True
    )
    for token in listing.stdout.split():
        child = int(token)
        try:
            if os.getpgid(child) == child:
                return child
        except (ProcessLookupError, PermissionError):
            continue
    return None


def _wait_until(predicate, *, timeout_s: float, message: str):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail(message)


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _subprocess_env_for_this_checkout() -> dict[str, str]:
    """Environment forcing child scripts to import this checkout's package.

    For script execution ``sys.path[0]`` is the script's directory, not the
    working directory, so an editable install from another checkout on the
    interpreter's path would otherwise win the ``vllm_mlx`` import.
    """

    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    return env


def _reap_leftovers(pids: list[int | None]) -> None:
    for pid in pids:
        if pid:
            with contextlib.suppress(OSError):
                os.killpg(pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
def test_external_group_sigterm_reaps_detached_download_worker(
    tmp_path: Path,
) -> None:
    """Supervisor Stop must reach a setsid download worker mid-transfer.

    The Desktop cancels a benchmark by signalling the CLI's process group.
    The download worker deliberately detaches from that group, so this drives
    the real ``_download_video_artifact`` path against a server that accepts
    and then never responds, SIGTERMs the externally supervised group like the
    Desktop does, and requires the detached worker to die and the orphaned
    temporary artifact to disappear.
    """

    destination = tmp_path / "artifact.mp4"
    connected_marker = tmp_path / "connected"
    script = tmp_path / "benchmark_parent.py"
    script.write_text(
        textwrap.dedent(
            """
            import socket
            import sys
            import threading

            from vllm_mlx.community_bench import local_runner


            def main() -> None:
                destination, connected_marker = sys.argv[1], sys.argv[2]
                listener = socket.socket()
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                port = listener.getsockname()[1]

                def hold_connection() -> None:
                    # Keep the accepted socket referenced: dropping it would
                    # reset the worker's connection instead of blocking it.
                    connection, _ = listener.accept()
                    with open(connected_marker, "w") as marker:
                        marker.write("connected")
                    threading.Event().wait()
                    connection.close()

                threading.Thread(target=hold_connection, daemon=True).start()
                # The production flow pre-creates the temporary artifact file.
                with open(destination, "wb") as file:
                    file.write(b"placeholder")
                local_runner._download_video_artifact(
                    f"http://127.0.0.1:{port}", "job-1", destination, timeout_s=60
                )


            if __name__ == "__main__":
                main()
            """
        )
    )
    process = subprocess.Popen(
        [sys.executable, str(script), str(destination), str(connected_marker)],
        start_new_session=True,
        cwd=REPO_ROOT,
        env=_subprocess_env_for_this_checkout(),
    )
    worker_pid: int | None = None
    try:
        worker_pid = _wait_until(
            lambda: _detached_child_of(process.pid),
            timeout_s=30,
            message="download worker never detached into its own group",
        )
        _wait_until(
            connected_marker.exists,
            timeout_s=30,
            message="worker never entered the blocked download phase",
        )
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        _wait_until(
            lambda: _pid_gone(worker_pid),
            timeout_s=15,
            message="detached download worker survived group cancellation",
        )
        assert not destination.exists(), "orphaned temp artifact was not removed"
    finally:
        _reap_leftovers([process.pid, worker_pid])
        process.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
def test_external_group_sigterm_reaps_worker_descendants_and_artifact(
    tmp_path: Path,
) -> None:
    """Parent death must take down the whole detached group, ffmpeg included.

    The probe worker's blocking work spawns ffmpeg descendants inside its
    detached group. This exercises the real worker supervision and lifetime
    entry with a worker that blocks after spawning a descendant (standing in
    for ffmpeg), SIGTERMs the externally supervised benchmark group, and
    requires worker, descendant, and orphaned artifact to all go away.
    """

    descendant_pid_path = tmp_path / "descendant.pid"
    artifact = tmp_path / "artifact.mp4"
    script = tmp_path / "benchmark_parent.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            import subprocess
            import sys
            import time

            from vllm_mlx.community_bench import local_runner


            def blocked_probe_worker(
                descendant_pid_path, artifact_path, sender, lifeline
            ):
                local_runner._enter_worker_lifetime(
                    lifeline, cleanup_path=artifact_path
                )
                descendant = subprocess.Popen(["sleep", "300"])
                with open(descendant_pid_path + ".tmp", "w") as file:
                    file.write(str(descendant.pid))
                os.replace(descendant_pid_path + ".tmp", descendant_pid_path)
                time.sleep(300)


            def main() -> None:
                descendant_pid_path, artifact_path = sys.argv[1], sys.argv[2]
                with open(artifact_path, "wb") as file:
                    file.write(b"artifact")
                local_runner._run_detached_worker(
                    blocked_probe_worker,
                    (descendant_pid_path, artifact_path),
                    timeout_s=60,
                    phase="probe",
                )


            if __name__ == "__main__":
                main()
            """
        )
    )
    process = subprocess.Popen(
        [sys.executable, str(script), str(descendant_pid_path), str(artifact)],
        start_new_session=True,
        cwd=REPO_ROOT,
        env=_subprocess_env_for_this_checkout(),
    )
    worker_pid: int | None = None
    descendant_pid: int | None = None
    try:
        worker_pid = _wait_until(
            lambda: _detached_child_of(process.pid),
            timeout_s=30,
            message="worker never detached into its own group",
        )
        descendant_pid = int(
            _wait_until(
                lambda: (
                    descendant_pid_path.read_text()
                    if descendant_pid_path.exists()
                    else None
                ),
                timeout_s=30,
                message="worker never spawned its descendant",
            )
        )
        assert os.getpgid(descendant_pid) == worker_pid, (
            "descendant must live inside the worker's detached group"
        )
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        _wait_until(
            lambda: _pid_gone(worker_pid),
            timeout_s=15,
            message="detached worker survived group cancellation",
        )
        _wait_until(
            lambda: _pid_gone(descendant_pid),
            timeout_s=15,
            message="worker descendant survived group cancellation",
        )
        assert not artifact.exists(), "orphaned temp artifact was not removed"
    finally:
        _reap_leftovers([process.pid, worker_pid])
        if descendant_pid:
            with contextlib.suppress(OSError):
                os.kill(descendant_pid, signal.SIGKILL)
        process.wait(timeout=10)


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
    monkeypatch.setattr(
        local_runner,
        "_download_video_artifact",
        lambda base_url, job_id, path: Path(path).write_bytes(b"mp4"),
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


def test_run_local_does_not_infer_cancellation_from_error_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "video_generation", "mlx-community/example-video-model"
    )

    monkeypatch.setattr(
        local_runner,
        "_run_video",
        lambda alias, *, isolate_process_group: (_ for _ in ()).throw(
            RuntimeError("backend cancelled an internal request")
        ),
    )

    with pytest.raises(local_runner.LocalBenchmarkError) as error:
        local_runner.run_local("example-video", archive=archive)

    assert error.value.run["outcome"] == {
        "status": "failed",
        "failure_code": "runtime_error",
    }


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


@pytest.mark.requires_mlx
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


# ---------------------------------------------------------------------------
# Packaged-contract and validator failure branches
# ---------------------------------------------------------------------------


def test_packaged_contract_must_be_a_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Root:
        def joinpath(self, name: str):
            return SimpleNamespace(read_text=lambda encoding: "[]")

    monkeypatch.setattr(benchmark_contracts.resources, "files", lambda package: Root())

    with pytest.raises(ValueError, match="not a JSON object"):
        benchmark_contracts._read_json("benchmark-run.schema.json")


def test_unregistered_task_types_and_prompts_are_rejected() -> None:
    with pytest.raises(ValueError, match="no registered community benchmark"):
        registered_workload("audio_generation")
    with pytest.raises(ValueError, match="no registered community benchmark"):
        registered_workload_history("audio_generation")
    with pytest.raises(ValueError, match="no registered prompt"):
        benchmark_contracts.public_prompt("case-that-does-not-exist")


def test_execution_digest_mismatch_is_rejected() -> None:
    run = _image_run()
    run["execution"]["config_digest"] = rcj_digest({"tampered": True})
    with pytest.raises(ValueError, match="does not match effective task"):
        BenchmarkRunValidator().validate(run)


def test_duplicate_measured_rounds_are_rejected() -> None:
    run = _text_run()
    run["measurements"][1]["round_index"] = 1  # duplicates (pp512-tg128, 1)
    with pytest.raises(ValueError, match="unique"):
        BenchmarkRunValidator().validate(run)


def test_measured_rounds_must_match_the_declared_set_exactly() -> None:
    run = _text_run()
    run["measurements"][0]["round_index"] = 9  # unique, but outside 1..5
    with pytest.raises(ValueError, match="declared measured rounds"):
        BenchmarkRunValidator().validate(run)


def test_measurement_shape_must_match_the_registered_case() -> None:
    run = _image_run()
    run["measurements"][0]["width"] = 512
    with pytest.raises(ValueError, match="does not match the registered case"):
        BenchmarkRunValidator().validate(run)


def test_measured_phases_cannot_exceed_total_duration() -> None:
    run = _text_run()
    run["measurements"][0]["ttft_ms"] = 1_000_000.0
    with pytest.raises(ValueError, match="shorter than its measured phases"):
        BenchmarkRunValidator().validate(run)


# ---------------------------------------------------------------------------
# CLI human-readable output and failure reporting
# ---------------------------------------------------------------------------


def _cli_archive(monkeypatch: pytest.MonkeyPatch, archive: object) -> None:
    monkeypatch.setattr(
        community_cli.LocalRunArchive, "default", classmethod(lambda cls: archive)
    )


def test_cli_catalog_prints_focus_marker_and_run_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cli_archive(monkeypatch, SimpleNamespace())
    monkeypatch.setattr(
        community_cli,
        "benchmark_catalog",
        lambda **kwargs: {
            "models": [
                {
                    "alias": "focus-model",
                    "task_type": "text_generation",
                    "memory_fit": "does_not_fit",
                    "focus": True,
                },
                {
                    "alias": "other-model",
                    "task_type": "image_generation",
                    "memory_fit": "fits",
                    "focus": False,
                },
            ]
        },
    )
    args = SimpleNamespace(benchmark_action="catalog", memory_gib=8, json=False)

    assert community_cli.benchmark_command(args) == 0
    out = capsys.readouterr().out
    assert "Community Benchmark models (local by default)" in out
    assert "★ focus-model" in out
    assert "does not fit" in out
    assert "Run: rapid-mlx benchmark run <model>" in out


def test_cli_plan_prints_protocol_and_local_storage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cli_archive(monkeypatch, SimpleNamespace())
    monkeypatch.setattr(
        community_cli,
        "plan_for_alias",
        lambda alias: {
            "model": {
                "alias": alias,
                "task_type": "text_generation",
                "protocol_id": "rapid-community-speed",
            },
            "workload": {"protocol_version": 2},
        },
    )
    args = SimpleNamespace(
        benchmark_action="plan", benchmark_model="example-text", json=False
    )

    assert community_cli.benchmark_command(args) == 0
    out = capsys.readouterr().out
    assert "Model:    example-text" in out
    assert "Protocol: rapid-community-speed v2" in out
    assert (
        "Storage:  local; upload requires a separate share command and consent" in out
    )


def test_cli_results_prints_empty_hint_then_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows: list[dict] = []

    class Archive:
        def list(self, *, limit=None):
            return list(rows)

        def receipt(self, run_id: str):
            return None

    _cli_archive(monkeypatch, Archive())
    args = SimpleNamespace(benchmark_action="results", limit=None, json=False)

    assert community_cli.benchmark_command(args) == 0
    assert "No local benchmark results yet." in capsys.readouterr().out

    rows.append(
        {
            "run_id": "00000000-0000-4000-8000-000000000001",
            "workload": {"task_type": "text_generation"},
            "outcome": {"status": "completed"},
            "completed_at": "2026-09-01T00:00:00Z",
        }
    )
    assert community_cli.benchmark_command(args) == 0
    out = capsys.readouterr().out
    assert "00000000-0000-4000-8000-000000000001" in out
    assert "text_generation" in out
    assert "completed" in out


def test_cli_inspect_prints_full_json_without_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Archive:
        def get(self, run_id: str):
            assert run_id == "00000000-0000-4000-8000-000000000001"
            return {"run_id": run_id}

    _cli_archive(monkeypatch, Archive())
    args = SimpleNamespace(
        benchmark_action="inspect",
        run_id="00000000-0000-4000-8000-000000000001",
        json=False,
    )

    assert community_cli.benchmark_command(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "00000000-0000-4000-8000-000000000001"
    }


def test_cli_run_prints_local_only_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cli_archive(monkeypatch, SimpleNamespace())
    monkeypatch.setattr(
        community_cli, "run_local", lambda alias, **kwargs: {"run_id": "abc-123"}
    )
    args = SimpleNamespace(
        benchmark_action="run",
        benchmark_model="example-text",
        inherit_process_group=False,
        json=False,
    )

    assert community_cli.benchmark_command(args) == 0
    out = capsys.readouterr().out
    assert "Saved local result abc-123" in out
    assert "Nothing was uploaded." in out


def test_cli_failure_json_includes_saved_run_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cli_archive(monkeypatch, SimpleNamespace())
    failed_run = {"run_id": "failed-run", "outcome": {"status": "failed"}}
    monkeypatch.setattr(
        community_cli,
        "run_local",
        lambda alias, **kwargs: (_ for _ in ()).throw(
            local_runner.LocalBenchmarkError("engine broke", failed_run, saved=True)
        ),
    )
    args = SimpleNamespace(
        benchmark_action="run",
        benchmark_model="example-text",
        inherit_process_group=False,
        json=True,
    )

    assert community_cli.benchmark_command(args) == 1
    assert json.loads(capsys.readouterr().err) == {
        "error": "engine broke",
        "saved": True,
        "run": failed_run,
    }


@pytest.mark.parametrize(
    ("saved", "run", "expected"),
    [
        (True, {"run_id": "failed-run"}, "local outcome saved as failed-run"),
        (False, {"run_id": "failed-run"}, "local outcome could not be saved"),
        (False, None, "Benchmark command failed"),
    ],
)
def test_cli_failure_text_reports_archive_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    saved: bool,
    run: dict | None,
    expected: str,
) -> None:
    _cli_archive(monkeypatch, SimpleNamespace())
    monkeypatch.setattr(
        community_cli,
        "run_local",
        lambda alias, **kwargs: (_ for _ in ()).throw(
            local_runner.LocalBenchmarkError("engine broke", run, saved=saved)
            if run is not None or saved
            else RuntimeError("engine broke")
        ),
    )
    args = SimpleNamespace(
        benchmark_action="run",
        benchmark_model="example-text",
        inherit_process_group=False,
        json=False,
    )

    assert community_cli.benchmark_command(args) == 1
    err = capsys.readouterr().err
    assert expected in err
    assert "engine broke" in err


def test_top_level_cli_dispatches_community_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rapid-mlx benchmark …` must route to the community CLI and exit with
    its return code — the SystemExit is the CLI's only success/failure
    signal for supervisors."""

    import vllm_mlx.cli as top_cli
    from vllm_mlx.community_bench import cli as community_module

    observed: dict[str, str] = {}

    def fake_benchmark_command(args) -> int:
        observed["action"] = args.benchmark_action
        return 3

    monkeypatch.setattr(community_module, "benchmark_command", fake_benchmark_command)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "benchmark", "results"])

    with pytest.raises(SystemExit) as excinfo:
        top_cli.main()

    assert excinfo.value.code == 3
    assert observed == {"action": "results"}


# ---------------------------------------------------------------------------
# Bench server teardown escalation
# ---------------------------------------------------------------------------


def test_bench_server_terminate_escalates_group_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM the isolated group first; SIGKILL it when it will not die."""

    signals: list[tuple[int, int]] = []

    class Proc:
        pid = 4242

        def poll(self):
            return None

        def wait(self, timeout):
            if signals == [(4242, signal.SIGTERM)]:
                raise subprocess.TimeoutExpired(cmd="serve", timeout=timeout)
            return 0

    monkeypatch.setattr(_server.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        _server.os, "killpg", lambda pgid, sig: signals.append((pgid, sig))
    )

    _server._terminate(Proc(), isolated_process_group=True)

    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]


# ---------------------------------------------------------------------------
# Run-builder provenance probes
# ---------------------------------------------------------------------------


def test_source_revision_is_none_outside_any_git_checkout(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("")
    assert run_builder._source_checkout_revision(module) is None


def test_source_revision_is_none_for_untracked_module_inside_a_repo(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    module = tmp_path / "module.py"
    module.write_text("")
    assert run_builder._source_checkout_revision(module) is None


def test_source_revision_probe_failure_is_a_hard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_run(*args, **kwargs):
        raise OSError("git binary unavailable")

    monkeypatch.setattr(run_builder.subprocess, "run", broken_run)

    with pytest.raises(RuntimeError, match="could not resolve"):
        run_builder._source_checkout_revision()


def test_source_revision_rejects_malformed_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="not-a-sha\n", stderr=""),
        ]
    )
    monkeypatch.setattr(
        run_builder.subprocess, "run", lambda *args, **kwargs: next(responses)
    )

    with pytest.raises(RuntimeError, match="could not resolve"):
        run_builder._source_checkout_revision()


def test_execution_config_records_installed_optional_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"mlx": "0.32.1", "mlx-lm": "0.28.4"}
    monkeypatch.setattr(
        run_builder,
        "_installed",
        lambda name, fallback=None: versions.get(name, fallback),
    )

    runtime = execution_config("text_generation")["runtime"]

    assert runtime["mlx"] == "0.32.1"
    assert runtime["mlx_lm"] == "0.28.4"
    assert "mlx_vlm" not in runtime
    assert "mflux" not in runtime


def test_execution_config_rejects_unregistered_task_type() -> None:
    with pytest.raises(ValueError, match="unsupported task type"):
        execution_config("audio_generation")


# ---------------------------------------------------------------------------
# Catalog projection and archive edge cases
# ---------------------------------------------------------------------------


def test_catalog_excludes_image_alias_without_text_to_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rapid-image-speed is a text_to_image protocol; an edit-only alias has
    no comparable registered workload and must not be offered."""

    snapshot = {
        "catalog_digest": "sha256:" + "0" * 64,
        "models": [
            {
                "registry_model_id": "model-1",
                "source": {"repo_id": "mlx-community/edit-only"},
                "estimated_download_size_bytes": 1 << 30,
            }
        ],
        "aliases": [
            {
                "alias": "img-edit-only",
                "capabilities": {
                    "task_types": ["image_generation"],
                    "operation_modes": ["image_to_image"],
                },
                "target": {
                    "registry_model_id": "model-1",
                    "resolution_status": "unresolved",
                },
            }
        ],
    }
    monkeypatch.setattr(
        workspace_module, "build_legacy_catalog_snapshot", lambda: snapshot
    )
    monkeypatch.setattr("vllm_mlx.model_aliases.list_profiles", lambda: {})

    assert benchmark_catalog()["models"] == []


def test_plan_for_unknown_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown or unsupported benchmark model"):
        plan_for_alias("alias-that-does-not-exist")


def test_archive_get_rejects_non_hex_run_ids(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    for run_id in ("", "../escape", "UPPER"):
        with pytest.raises(ValueError, match="invalid run id"):
            archive.get(run_id)


def test_archive_list_rejects_non_positive_limit(tmp_path: Path) -> None:
    archive = LocalRunArchive(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        archive.list(limit=0)


def test_archive_limit_keeps_latest_regardless_of_scan_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The heap must replace older entries even when the directory scan
    yields the oldest files first — glob order is filesystem-dependent."""

    archive = LocalRunArchive(tmp_path)
    for index in range(3):
        run = _text_run()
        run["run_id"] = f"00000000-0000-4000-8000-{index:012d}"
        run["started_at"] = f"2026-08-{index + 1:02d}T00:00:00Z"
        run["completed_at"] = f"2026-08-{index + 1:02d}T00:01:00Z"
        archive.save(run)
    real_dir = archive.runs_dir

    class OrderedDir:
        def exists(self) -> bool:
            return True

        def glob(self, pattern: str):
            return sorted(
                real_dir.glob(pattern),
                key=lambda path: json.loads(path.read_text(encoding="utf-8"))[
                    "started_at"
                ],
            )

        def __truediv__(self, name: str) -> Path:
            return real_dir / name

    monkeypatch.setattr(
        LocalRunArchive, "runs_dir", property(lambda self: OrderedDir())
    )

    assert [run["run_id"] for run in archive.list(limit=2)] == [
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000001",
    ]


# ---------------------------------------------------------------------------
# Local-runner failure taxonomy and artifact validation branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (local_runner.BenchmarkCancelledError("stop requested"), "user_cancelled"),
        (MemoryError("exhausted"), "runtime_oom"),
        (RuntimeError("Metal buffer alloc failed"), "runtime_oom"),
        (RuntimeError("unsupported benchmark task"), "unsupported_task"),
        (TimeoutError("probe timed out"), "timeout"),
        (RuntimeError("model repo not found"), "invalid_model"),
        (RuntimeError("something else"), "runtime_error"),
    ],
)
def test_failure_codes_classify_without_leaking_text(
    error: Exception, code: str
) -> None:
    assert local_runner._failure_code(error) == code


def test_image_artifact_response_shape_is_validated() -> None:
    with pytest.raises(RuntimeError, match="no artifact list"):
        local_runner._validated_image_count({"data": "oops"}, width=8, height=8)
    with pytest.raises(RuntimeError, match="no base64 artifact"):
        local_runner._validated_image_count({"data": [{}]}, width=8, height=8)
    with pytest.raises(RuntimeError, match="invalid artifact"):
        local_runner._validated_image_count(
            {"data": [{"b64_json": base64.b64encode(b"not-a-png").decode("ascii")}]},
            width=8,
            height=8,
        )


def test_run_local_rejects_incomplete_image_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = LocalRunArchive(tmp_path)
    _mock_local_context(
        monkeypatch, "image_generation", "mlx-community/example-image-model"
    )

    @contextlib.contextmanager
    def serve(alias: str, **kwargs):
        yield {"base_url": "http://local/v1"}

    artifact = {"b64_json": _png_base64(1024, 1024)}
    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(
        local_runner.requests,
        "post",
        lambda *args, **kwargs: _Response({"data": [artifact, artifact]}),
    )

    with pytest.raises(
        local_runner.LocalBenchmarkError, match="incomplete batch"
    ) as error:
        local_runner.run_local("example-image", archive=archive)

    assert error.value.run["outcome"] == {
        "status": "failed",
        "failure_code": "runtime_error",
    }


def _install_fake_imageio(monkeypatch: pytest.MonkeyPatch, reader: object) -> None:
    imageio_pkg = types.ModuleType("imageio")
    v2 = types.ModuleType("imageio.v2")
    v2.get_reader = lambda path, format: reader  # type: ignore[attr-defined]
    imageio_pkg.v2 = v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", imageio_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)


class _FakeVideoReader:
    def __init__(self, metadata: dict, frames: int) -> None:
        self._metadata = metadata
        self._frames = frames
        self.closed = False

    def get_meta_data(self) -> dict:
        return self._metadata

    def count_frames(self) -> int:
        return self._frames

    def close(self) -> None:
        self.closed = True


def test_video_probe_reads_shape_and_closes_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _FakeVideoReader({"size": (832, 480), "fps": 24.0}, frames=81)
    _install_fake_imageio(monkeypatch, reader)

    assert local_runner._probe_video_artifact_unbounded("clip.mp4") == (
        832,
        480,
        81,
        24.0,
    )
    assert reader.closed is True


def test_video_probe_requires_dimension_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _FakeVideoReader({"size": None, "fps": 24.0}, frames=81)
    _install_fake_imageio(monkeypatch, reader)

    with pytest.raises(RuntimeError, match="no dimensions"):
        local_runner._probe_video_artifact_unbounded("clip.mp4")
    assert reader.closed is True


def test_video_probe_translates_decoder_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imageio_pkg = types.ModuleType("imageio")
    v2 = types.ModuleType("imageio.v2")

    def get_reader(path: str, format: str):
        raise ValueError("moov atom not found")

    v2.get_reader = get_reader  # type: ignore[attr-defined]
    imageio_pkg.v2 = v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", imageio_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)

    with pytest.raises(RuntimeError, match="invalid MP4 artifact"):
        local_runner._probe_video_artifact_unbounded("clip.mp4")


def test_video_probe_uses_bundled_ffmpeg_without_imageio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_imageio(name: str, *args: object, **kwargs: object):
        if name == "imageio.v2":
            raise ImportError("Desktop intentionally omits imageio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_imageio)
    monkeypatch.setenv("FFMPEG_BINARY", "/app/rapid-mlx/bin/ffmpeg")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="",
        stderr=(
            "Stream #0:0: Video: h264, yuv420p, 832x480, 24 fps, 24 tbr\n"
            "frame=   81 fps=0.0 q=-1.0 Lsize=1KiB\n"
        ),
    )
    monkeypatch.setattr(
        local_runner.subprocess, "run", lambda *args, **kwargs: completed
    )

    assert local_runner._probe_video_artifact_unbounded("clip.mp4") == (
        832,
        480,
        81,
        24.0,
    )


def test_bundled_ffmpeg_probe_rejects_unparseable_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="invalid data"
        ),
    )

    with pytest.raises(RuntimeError, match="invalid MP4 artifact"):
        local_runner._probe_video_with_ffmpeg("clip.mp4", "/app/bin/ffmpeg")


def test_bundled_ffmpeg_probe_has_no_second_artifact_and_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured.update(command=command, **kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(local_runner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="invalid MP4 artifact"):
        local_runner._probe_video_with_ffmpeg(
            "clip.mp4", "/app/bin/ffmpeg", desktop_bundle=True
        )

    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["timeout"] == local_runner._VIDEO_ARTIFACT_PROBE_TIMEOUT_S
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("-c:v") + 1] == "h264_videotoolbox"
    assert command[-1] == "pipe:1"


def test_system_ffmpeg_probe_uses_portable_null_muxer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="",
        stderr=(
            "Stream #0:0: Video: h264, yuv420p, 832x480, 24 fps, 24 tbr\n"
            "frame=   81 fps=0.0 q=-0.0 Lsize=N/A\n"
        ),
    )

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        return completed

    monkeypatch.setattr(local_runner.subprocess, "run", run)
    assert local_runner._probe_video_with_ffmpeg("clip.mp4", "/usr/bin/ffmpeg") == (
        832,
        480,
        81,
        24.0,
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert command[-3:] == ["-f", "null", "-"]
    assert "h264_videotoolbox" not in command


def test_ffmpeg_environment_variable_does_not_imply_sidecar_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_runner.sys, "executable", "/usr/bin/python3")
    assert local_runner._is_sidecar_bundled_ffmpeg("/usr/local/bin/ffmpeg") is False

    monkeypatch.setattr(
        local_runner.sys,
        "executable",
        "/Applications/Rapid.app/Contents/Resources/rapid-mlx/python/bin/python3.12",
    )
    assert local_runner._is_sidecar_bundled_ffmpeg(
        "/Applications/Rapid.app/Contents/Resources/rapid-mlx/bin/ffmpeg"
    )


def test_sidecar_ffmpeg_detection_fails_closed_on_unresolvable_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_runner.os.path,
        "realpath",
        lambda path: (_ for _ in ()).throw(OSError("unresolvable path")),
    )
    assert local_runner._is_sidecar_bundled_ffmpeg("/app/bin/ffmpeg") is False


def test_video_probe_without_imageio_or_ffmpeg_explains_required_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_imageio(name: str, *args: object, **kwargs: object):
        if name == "imageio.v2":
            raise ImportError("imageio unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_imageio)
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setattr(local_runner.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match=r"validation requires rapid-mlx\[video\]"):
        local_runner._probe_video_artifact_unbounded("clip.mp4")


def test_probe_video_artifact_returns_worker_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_runner,
        "_run_detached_worker",
        lambda target, args, *, timeout_s, phase: [832, 480, 81, 24.0],
    )
    assert local_runner._probe_video_artifact("clip.mp4") == (832, 480, 81, 24.0)


class _HeaderResponse(_Response):
    def __init__(self, headers: dict, chunks: list[bytes]) -> None:
        super().__init__({})
        self.headers = headers
        self._chunks = chunks

    def iter_content(self, *, chunk_size: int):
        yield from self._chunks


@pytest.mark.parametrize(
    ("content_length", "message"),
    [
        ("not-a-number", "invalid Content-Length"),
        ("-5", "invalid Content-Length"),
        (str(2 * 1024 * 1024 * 1024), "safety limit"),
    ],
)
def test_video_download_rejects_bad_declared_sizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    content_length: str,
    message: str,
) -> None:
    response = _HeaderResponse({"content-length": content_length}, [b"data"])
    monkeypatch.setattr(local_runner.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match=message):
        local_runner._download_video_artifact_unbounded(
            "http://local/v1", "job-1", str(tmp_path / "artifact.mp4")
        )
    assert response.closed is True


def test_video_download_skips_keepalive_chunks_and_rejects_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ok = _HeaderResponse({"content-length": "4"}, [b"", b"data"])
    monkeypatch.setattr(local_runner.requests, "get", lambda *args, **kwargs: ok)
    artifact = tmp_path / "artifact.mp4"
    local_runner._download_video_artifact_unbounded(
        "http://local/v1", "job-1", str(artifact)
    )
    assert artifact.read_bytes() == b"data"

    empty = _HeaderResponse({}, [])
    monkeypatch.setattr(local_runner.requests, "get", lambda *args, **kwargs: empty)
    with pytest.raises(RuntimeError, match="empty MP4 artifact"):
        local_runner._download_video_artifact_unbounded(
            "http://local/v1", "job-1", str(artifact)
        )


# ---------------------------------------------------------------------------
# Detached worker lifetime, in-process
# ---------------------------------------------------------------------------


def test_parent_lifeline_watchdog_cleans_up_and_kills_owned_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    sender.close()  # the parent is already gone: poll() reports EOF instantly
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"partial")
    events: list[tuple] = []
    monkeypatch.setattr(local_runner.os, "getpgid", lambda pid: os.getpid())
    monkeypatch.setattr(
        local_runner.os, "killpg", lambda pid, sig: events.append(("killpg", pid, sig))
    )
    monkeypatch.setattr(
        local_runner.os, "_exit", lambda code: events.append(("exit", code))
    )

    local_runner._watch_parent_lifeline(receiver, str(artifact))

    assert not artifact.exists()
    assert events == [("killpg", os.getpid(), signal.SIGKILL), ("exit", 1)]


def test_parent_lifeline_watchdog_never_signals_a_foreign_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    sender.close()
    events: list[tuple] = []
    monkeypatch.setattr(local_runner.os, "getpgid", lambda pid: os.getpid() + 1)
    monkeypatch.setattr(
        local_runner.os, "killpg", lambda pid, sig: events.append(("killpg", pid, sig))
    )
    monkeypatch.setattr(
        local_runner.os, "_exit", lambda code: events.append(("exit", code))
    )

    # The cleanup path may already be gone; that must not stop the exit.
    local_runner._watch_parent_lifeline(receiver, str(tmp_path / "missing.mp4"))

    assert events == [("exit", 1)]


def test_enter_worker_lifetime_detaches_and_arms_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    armed = threading.Event()
    observed: dict[str, object] = {}

    def fake_watch(lifeline, cleanup_path) -> None:
        observed["lifeline"] = lifeline
        observed["cleanup_path"] = cleanup_path
        armed.set()

    monkeypatch.setattr(
        local_runner.os, "setsid", lambda: events.append("setsid"), raising=False
    )
    monkeypatch.setattr(local_runner, "_watch_parent_lifeline", fake_watch)
    lifeline = object()

    local_runner._enter_worker_lifetime(lifeline, cleanup_path="/tmp/artifact.mp4")

    assert armed.wait(timeout=10), "watchdog thread never started"
    assert events == ["setsid"]
    assert observed == {"lifeline": lifeline, "cleanup_path": "/tmp/artifact.mp4"}


def test_video_probe_worker_reports_result_and_closes_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered: dict[str, object] = {}
    monkeypatch.setattr(
        local_runner,
        "_enter_worker_lifetime",
        lambda lifeline, *, cleanup_path=None: entered.update(
            {"lifeline": lifeline, "cleanup_path": cleanup_path}
        ),
    )
    monkeypatch.setattr(
        local_runner,
        "_probe_video_artifact_unbounded",
        lambda path: (832, 480, 81, 24.0),
    )
    receiver, sender = multiprocessing.Pipe(duplex=False)
    lifeline = object()

    local_runner._video_probe_worker("clip.mp4", sender, lifeline)

    assert receiver.recv() == ("ok", (832, 480, 81, 24.0))
    assert sender.closed is True
    assert entered == {"lifeline": lifeline, "cleanup_path": "clip.mp4"}


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RuntimeError("video artifact has no dimensions"), "no dimensions"),
        (ValueError("decoder blew up"), "invalid MP4 artifact"),
    ],
)
def test_video_probe_worker_reports_errors_without_leaking_internals(
    monkeypatch: pytest.MonkeyPatch, error: Exception, message: str
) -> None:
    monkeypatch.setattr(
        local_runner,
        "_enter_worker_lifetime",
        lambda lifeline, *, cleanup_path=None: None,
    )
    monkeypatch.setattr(
        local_runner,
        "_probe_video_artifact_unbounded",
        lambda path: (_ for _ in ()).throw(error),
    )
    receiver, sender = multiprocessing.Pipe(duplex=False)

    local_runner._video_probe_worker("clip.mp4", sender, object())

    status, payload = receiver.recv()
    assert status == "error"
    assert message in payload


def test_video_probe_worker_survives_a_torn_result_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_runner,
        "_enter_worker_lifetime",
        lambda lifeline, *, cleanup_path=None: None,
    )
    monkeypatch.setattr(
        local_runner,
        "_probe_video_artifact_unbounded",
        lambda path: (_ for _ in ()).throw(RuntimeError("bad artifact")),
    )

    class TornSender:
        def __init__(self) -> None:
            self.closed = False

        def send(self, payload) -> None:
            raise BrokenPipeError("parent already reaped the pipe")

        def close(self) -> None:
            self.closed = True

    sender = TornSender()
    local_runner._video_probe_worker("clip.mp4", sender, object())
    assert sender.closed is True


@pytest.mark.parametrize("outcome", ["ok", "runtime", "unexpected"])
def test_video_download_worker_reports_each_outcome(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setattr(
        local_runner,
        "_enter_worker_lifetime",
        lambda lifeline, *, cleanup_path=None: None,
    )

    def download(base_url: str, job_id: str, destination_path: str) -> None:
        if outcome == "runtime":
            raise RuntimeError("video artifact exceeds the 1 GiB safety limit")
        if outcome == "unexpected":
            raise OSError("disk pulled")

    monkeypatch.setattr(local_runner, "_download_video_artifact_unbounded", download)
    receiver, sender = multiprocessing.Pipe(duplex=False)

    local_runner._video_download_worker(
        "http://local/v1", "job-1", "artifact.mp4", sender, object()
    )

    status, payload = receiver.recv()
    assert sender.closed is True
    if outcome == "ok":
        assert (status, payload) == ("ok", None)
    elif outcome == "runtime":
        assert (status, payload) == (
            "error",
            "video artifact exceeds the 1 GiB safety limit",
        )
    else:
        assert (status, payload) == ("error", "video artifact download failed")


def test_video_download_worker_survives_a_torn_result_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_runner,
        "_enter_worker_lifetime",
        lambda lifeline, *, cleanup_path=None: None,
    )
    monkeypatch.setattr(
        local_runner,
        "_download_video_artifact_unbounded",
        lambda base_url, job_id, destination_path: (_ for _ in ()).throw(
            RuntimeError("connection reset mid-stream")
        ),
    )

    class TornSender:
        def __init__(self) -> None:
            self.closed = False

        def send(self, payload) -> None:
            raise BrokenPipeError("parent already reaped the pipe")

        def close(self) -> None:
            self.closed = True

    sender = TornSender()
    local_runner._video_download_worker(
        "http://local/v1", "job-1", "artifact.mp4", sender, object()
    )
    assert sender.closed is True


class _FakeWorkerConnection:
    def __init__(self, recv=None, poll_result: bool = True) -> None:
        self._recv = recv
        self._poll_result = poll_result
        self.closed = False

    def poll(self, timeout) -> bool:
        return self._poll_result

    def recv(self):
        return self._recv()

    def close(self) -> None:
        self.closed = True


class _FakeWorkerProcess:
    def __init__(self, alive_after_join: bool = False) -> None:
        self.pid = 99999
        self._alive_after_join = alive_after_join
        self.started = False
        self.joins: list[float | None] = []

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:
        self.joins.append(timeout)

    def is_alive(self) -> bool:
        return self._alive_after_join


def _fake_spawn_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recv,
    alive_after_join: bool = False,
):
    receiver = _FakeWorkerConnection(recv=recv)
    result_sender = _FakeWorkerConnection()
    lifeline_receiver = _FakeWorkerConnection()
    lifeline_sender = _FakeWorkerConnection()
    process = _FakeWorkerProcess(alive_after_join=alive_after_join)
    pipes = [(receiver, result_sender), (lifeline_receiver, lifeline_sender)]

    class Context:
        def Pipe(self, duplex):  # noqa: N802 — multiprocessing context API
            return pipes.pop(0)

        def Process(self, target, args):  # noqa: N802 — multiprocessing context API
            process.target = target
            process.args = args
            return process

    monkeypatch.setattr(
        local_runner.multiprocessing, "get_context", lambda method: Context()
    )
    return receiver, result_sender, lifeline_sender, process


def test_detached_worker_returns_payload_and_keeps_lifeline_until_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver, result_sender, lifeline_sender, process = _fake_spawn_context(
        monkeypatch, recv=lambda: ("ok", (1, 2))
    )

    payload = local_runner._run_detached_worker(
        lambda *args: None, ("clip.mp4",), timeout_s=5, phase="probe"
    )

    assert payload == (1, 2)
    assert process.started is True
    assert receiver.closed is True
    assert result_sender.closed is True  # parent's copy of the child's end
    assert lifeline_sender.closed is True  # closed only after the reap


def test_detached_worker_translates_a_silent_worker_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def recv():
        raise EOFError

    _fake_spawn_context(monkeypatch, recv=recv)

    with pytest.raises(RuntimeError, match="probe exited without a result"):
        local_runner._run_detached_worker(
            lambda *args: None, ("clip.mp4",), timeout_s=5, phase="probe"
        )


def test_detached_worker_surfaces_worker_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_spawn_context(monkeypatch, recv=lambda: ("error", "bad artifact"))

    with pytest.raises(RuntimeError, match="bad artifact"):
        local_runner._run_detached_worker(
            lambda *args: None, ("clip.mp4",), timeout_s=5, phase="download"
        )


def test_detached_worker_reaps_a_survivor_after_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reaped: list[object] = []
    _, _, _, process = _fake_spawn_context(
        monkeypatch, recv=lambda: ("ok", None), alive_after_join=True
    )
    monkeypatch.setattr(
        local_runner, "_terminate_worker_process", lambda proc: reaped.append(proc)
    )

    assert (
        local_runner._run_detached_worker(
            lambda *args: None, ("clip.mp4",), timeout_s=5, phase="download"
        )
        is None
    )
    assert reaped == [process]


class _FakeSupervisedProcess:
    def __init__(self, pid, alive_sequence: list[bool]) -> None:
        self.pid = pid
        self._alive = iter(alive_sequence)
        self.calls: list[str] = []

    def is_alive(self) -> bool:
        return next(self._alive)

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")

    def join(self, timeout=None) -> None:
        self.calls.append("join")


def test_terminate_worker_skips_a_process_that_never_started() -> None:
    process = _FakeSupervisedProcess(None, alive_sequence=[])
    local_runner._terminate_worker_process(process)
    assert process.calls == []


def test_terminate_worker_escalates_group_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(local_runner.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        local_runner.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    process = _FakeSupervisedProcess(777, alive_sequence=[True, True])

    local_runner._terminate_worker_process(process)

    assert signals == [(777, signal.SIGTERM), (777, signal.SIGKILL)]
    assert process.calls == ["join", "join"]


def test_terminate_worker_tolerates_group_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_runner.os, "getpgid", lambda pid: pid)

    def killpg(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(local_runner.os, "killpg", killpg)
    process = _FakeSupervisedProcess(777, alive_sequence=[True, True])

    local_runner._terminate_worker_process(process)

    assert process.calls == ["join", "join"]


def test_terminate_worker_falls_back_to_direct_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def getpgid(pid):
        raise ProcessLookupError

    monkeypatch.setattr(local_runner.os, "getpgid", getpgid)
    process = _FakeSupervisedProcess(777, alive_sequence=[True, True])

    local_runner._terminate_worker_process(process)

    assert process.calls == ["terminate", "join", "kill", "join"]


def test_terminate_worker_stops_after_a_clean_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(local_runner.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        local_runner.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    process = _FakeSupervisedProcess(777, alive_sequence=[True, False])

    local_runner._terminate_worker_process(process)

    assert signals == [(777, signal.SIGTERM)]
    assert process.calls == ["join"]


def test_dedicated_group_leader_probe_reads_real_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = os.getpid() == os.getpgrp()
    assert local_runner._is_dedicated_process_group_leader() is expected

    monkeypatch.delattr(local_runner.os, "getpgrp")
    assert local_runner._is_dedicated_process_group_leader() is False


def test_run_local_surfaces_a_task_type_outside_every_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan can only carry the three registered task types; anything else
    is a programming error and must not silently pick an executor. The
    failure-run builder rejects the same unregistered task, so the raw
    ValueError is the surfaced contract here."""

    _mock_local_context(monkeypatch, "audio_generation", "mlx-community/example")

    with pytest.raises(ValueError, match="unsupported task type"):
        local_runner.run_local("example-audio", archive=SimpleNamespace())


# ---------------------------------------------------------------------------
# Standardized-bench runner on the no-MLX lane
# ---------------------------------------------------------------------------


class _RegisteredTokenizer:
    vocab_size = 1000
    all_special_ids = [368, 522, 834, 999]


def test_registered_token_workload_matches_golden_vector_without_mlx() -> None:
    ids = bench_runner._build_registered_token_ids(
        _RegisteredTokenizer(), 8, seed=12_648_430
    )
    assert ids == [469, 845, 945, 415, 950, 718, 771, 464]


def test_registered_token_workload_requires_vocab_and_special_evidence() -> None:
    class TinyVocab:
        vocab_size = 200
        all_special_ids: list[int] = []

    with pytest.raises(RuntimeError, match="vocab too small"):
        bench_runner._build_registered_token_ids(TinyVocab(), 8, seed=1)

    class NoSpecialEvidence:
        vocab_size = 1000

    with pytest.raises(RuntimeError, match="all_special_ids"):
        bench_runner._build_registered_token_ids(NoSpecialEvidence(), 8, seed=1)

    class EverythingSpecial:
        vocab_size = 258
        all_special_ids = [256, 257]

    with pytest.raises(RuntimeError, match="no eligible"):
        bench_runner._build_registered_token_ids(EverythingSpecial(), 8, seed=1)


class _StreamingEngine:
    def __init__(self, outputs: list[SimpleNamespace]) -> None:
        self.outputs = outputs
        self.requests: list[object] = []

    async def add_request(self, prompt, sampling_params) -> str:
        self.requests.append(prompt)
        return "request-1"

    async def stream_outputs(self, request_id: str, timeout: int):
        for output in self.outputs:
            yield output


def test_run_one_round_reports_engine_observed_counters() -> None:
    outputs = [
        SimpleNamespace(
            new_token_ids=[7],
            prompt_tokens=8,
            completion_tokens=4,
            output_token_ids=[7, 8, 9, 10],
        )
    ]
    engine = _StreamingEngine(outputs)

    result = asyncio.run(
        bench_runner._run_one_round(
            engine, [1] * 8, object(), 8, 4, require_observed_counts=True
        )
    )

    assert engine.requests == [[1] * 8]
    assert result.prompt_tokens == 8
    assert result.output_tokens == 4
    assert result.prefill_tps > 0
    assert result.decode_tps > 0
    assert result.ttft_ms >= 0


def test_run_one_round_rejects_an_empty_stream() -> None:
    engine = _StreamingEngine([])
    with pytest.raises(RuntimeError, match="no tokens"):
        asyncio.run(bench_runner._run_one_round(engine, [1] * 8, object(), 8, 4))


def test_run_one_round_rejects_early_stopped_rounds() -> None:
    outputs = [
        SimpleNamespace(
            new_token_ids=[7],
            prompt_tokens=8,
            completion_tokens=2,
            output_token_ids=[7, 8],
        )
    ]
    with pytest.raises(RuntimeError, match="requires exactly 4"):
        asyncio.run(
            bench_runner._run_one_round(
                _StreamingEngine(outputs), [1] * 8, object(), 8, 4
            )
        )


def test_run_bucket_selects_registered_or_synthetic_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[object] = []

    async def run_one(
        engine,
        prompt,
        sampling,
        target_prompt_tokens,
        max_tokens,
        *,
        require_observed_counts=False,
    ):
        observed.append((prompt, require_observed_counts))
        return bench_runner.RoundResult(
            decode_tps=1,
            prefill_tps=1,
            ttft_ms=1,
            prompt_tokens=target_prompt_tokens,
            output_tokens=max_tokens,
        )

    monkeypatch.setattr(bench_runner, "_run_one_round", run_one)
    monkeypatch.setattr(bench_runner, "_reset_peak_ram", lambda: None)

    result, registered_ids = asyncio.run(
        bench_runner._run_bucket(
            object(),
            _RegisteredTokenizer(),
            lambda max_tokens: object(),
            8,
            4,
            registered_token_ids=True,
        )
    )
    assert registered_ids == [469, 845, 945, 415, 950, 718, 771, 464]
    assert observed == [(registered_ids, True)] * 6
    assert len(result.rounds_raw) == 5

    class SyntheticTokenizer:
        vocab_size = 2000

        def decode(self, ids) -> str:
            return "synthetic prompt text"

    observed.clear()
    result, synthetic_ids = asyncio.run(
        bench_runner._run_bucket(
            object(),
            SyntheticTokenizer(),
            lambda max_tokens: object(),
            8,
            4,
            registered_token_ids=False,
        )
    )
    assert len(synthetic_ids) == 8
    assert observed == [("synthetic prompt text", False)] * 6
    assert len(result.rounds_raw) == 5
