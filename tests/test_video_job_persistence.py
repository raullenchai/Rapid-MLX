# SPDX-License-Identifier: Apache-2.0
"""Durable completed-job contract for the asynchronous Videos API."""

from __future__ import annotations

import asyncio
import inspect
import json
import stat
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx import cli, server
from vllm_mlx.routes import video


@pytest.fixture(autouse=True)
def _isolated_video_store():
    configure = video.configure_video_jobs
    configure(None)
    video.start_video_jobs()
    yield
    configure(None)
    video.start_video_jobs()


async def _wait_for_completion(video_id: str) -> dict:
    for _ in range(200):
        current = await video.retrieve_video(video_id)
        if current["status"] == "completed":
            return current
        await asyncio.sleep(0.01)
    raise AssertionError(f"video job {video_id} did not complete")


def _completed_job(job_id: str, *, created_at: int = 1) -> video._VideoJob:
    return video._VideoJob(
        id=job_id,
        model="ltx-2.3-mlx-q4",
        prompt=f"prompt {created_at}",
        seconds="1",
        size="512x512",
        status="completed",
        progress=100,
        created_at=created_at,
        completed_at=created_at + 1,
        output_path=str(video._jobs_root / job_id / "output.mp4"),
        generation_finished=True,
    )


def _write_completed_job(job: video._VideoJob) -> None:
    job_dir = video._jobs_root / job.id
    job_dir.mkdir(mode=0o700)
    (job_dir / "output.mp4").write_bytes(b"generated-mp4")
    video._persist_completed_job(job)


@pytest.mark.asyncio
async def test_completed_job_survives_store_reconfiguration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "videos"
    assert video.configure_video_jobs(store) == store.resolve()
    video.start_video_jobs()

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="Ocean waves at sunset",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=42,
        input_reference=None,
    )
    completed = await _wait_for_completion(created["id"])
    assert completed["progress"] == 100
    job_dir = store / created["id"]
    assert (job_dir / "job.json").is_file()
    assert not list(job_dir.glob(".job-*.json.tmp"))

    # Reconfiguration clears process memory and models a fresh server process
    # selecting the same operator-owned store.
    await asyncio.sleep(0)
    video.configure_video_jobs(store)
    assert created["id"] not in video._jobs
    video.start_video_jobs()

    restored = await video.retrieve_video(created["id"])
    assert restored == completed
    listing = await video.list_videos(limit=20)
    assert [item["id"] for item in listing["data"]] == [created["id"]]
    response = await video.retrieve_video_content(created["id"])
    assert (
        b"".join([chunk async for chunk in response.body_iterator]) == b"generated-mp4"
    )

    deleted = await video.delete_video(created["id"])
    assert deleted["deleted"] is True
    assert not job_dir.exists()


@pytest.mark.asyncio
async def test_default_store_remains_process_temporary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert video._jobs_are_persistent is False

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="Temporary result",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=1,
        input_reference=None,
    )
    await _wait_for_completion(created["id"])
    assert not (video._jobs_root / created["id"] / "job.json").exists()
    await video.delete_video(created["id"])


def test_restore_ignores_partial_malformed_and_noncompleted_records(
    tmp_path: Path,
) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)

    malformed_id = "video_" + "a" * 32
    malformed = store / malformed_id
    malformed.mkdir()
    (malformed / "output.mp4").write_bytes(b"mp4")
    (malformed / "job.json").write_text("{not-json", encoding="utf-8")

    partial_id = "video_" + "b" * 32
    partial = store / partial_id
    partial.mkdir()
    partial_job = _completed_job(partial_id)
    (partial / "job.json").write_text(
        json.dumps(video._completed_job_record(partial_job)), encoding="utf-8"
    )

    queued_id = "video_" + "c" * 32
    queued = store / queued_id
    queued.mkdir()
    (queued / "output.mp4").write_bytes(b"mp4")
    queued_record = video._completed_job_record(_completed_job(queued_id))
    queued_record.update(status="queued", progress=0, completed_at=None)
    (queued / "job.json").write_text(json.dumps(queued_record), encoding="utf-8")

    video.start_video_jobs()

    assert video._jobs == {}
    # Invalid records are ignored, not destructively removed. A future server
    # version or an operator can still inspect and recover them.
    assert malformed.exists()
    assert partial.exists()
    assert queued.exists()


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"schema_version": 2},
        {"schema_version": 1, "object": "not-video"},
        {
            "schema_version": 1,
            "object": "video",
            "status": "completed",
            "progress": 100,
            "error": {"code": "unexpected"},
        },
        {
            "schema_version": 1,
            "object": "video",
            "status": "completed",
            "progress": 100,
            "error": None,
            "model": "",
        },
    ],
)
def test_restore_rejects_invalid_completed_metadata(
    tmp_path: Path, record: object
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    job_id = "video_" + "e" * 32
    job_dir = video._jobs_root / job_id
    job_dir.mkdir()
    (job_dir / "output.mp4").write_bytes(b"mp4")
    if isinstance(record, dict):
        record = {"id": job_id, **record}
    (job_dir / "job.json").write_text(json.dumps(record), encoding="utf-8")

    assert video._load_completed_job(job_dir) is None


def test_restore_rejects_unowned_shapes_and_symlinks(tmp_path: Path) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    assert video._load_completed_job(video._jobs_root / "not-a-video-id") is None

    empty_id = "video_" + "f" * 32
    empty_dir = video._jobs_root / empty_id
    empty_dir.mkdir()
    (empty_dir / "output.mp4").write_bytes(b"")
    (empty_dir / "job.json").write_text("{}", encoding="utf-8")
    assert video._load_completed_job(empty_dir) is None

    linked_id = "video_" + "1" * 32
    linked_dir = video._jobs_root / linked_id
    linked_dir.mkdir()
    (linked_dir / "output.mp4").write_bytes(b"mp4")
    external_metadata = tmp_path / "external.json"
    external_metadata.write_text("{}", encoding="utf-8")
    (linked_dir / "job.json").symlink_to(external_metadata)
    assert video._load_completed_job(linked_dir) is None


def test_restore_reads_validated_metadata_descriptor_during_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    job = _completed_job("video_" + "4" * 32)
    _write_completed_job(job)
    job_dir = video._jobs_root / job.id
    metadata = job_dir / "job.json"
    metadata_inode = metadata.stat().st_ino
    replacement = tmp_path / "oversized.json"
    replacement.write_bytes(b"{" + b" " * video._MAX_VIDEO_JOB_METADATA_BYTES + b"}")
    original_fstat = video.os.fstat
    swapped = False

    def swap_path_after_validation(fd: int):
        nonlocal swapped
        opened = original_fstat(fd)
        if opened.st_ino == metadata_inode and not swapped:
            swapped = True
            metadata.unlink()
            metadata.symlink_to(replacement)
        return opened

    monkeypatch.setattr(video.os, "fstat", swap_path_after_validation)

    restored = video._load_completed_job(job_dir)

    assert swapped is True
    assert restored is not None
    assert restored.prompt == job.prompt


def test_restore_rejects_nonregular_metadata(tmp_path: Path) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    job_id = "video_" + "3" * 32
    job_dir = video._jobs_root / job_id
    job_dir.mkdir()
    (job_dir / "output.mp4").write_bytes(b"mp4")
    (job_dir / "job.json").mkdir()

    assert video._load_completed_job(job_dir) is None


def test_restore_bounds_metadata_that_grows_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    job = _completed_job("video_" + "2" * 32)
    _write_completed_job(job)
    job_dir = video._jobs_root / job.id
    metadata = job_dir / "job.json"
    metadata_inode = metadata.stat().st_ino
    original_fstat = video.os.fstat
    expanded = False

    def expand_after_validation(fd: int):
        nonlocal expanded
        opened = original_fstat(fd)
        if opened.st_ino == metadata_inode and not expanded:
            expanded = True
            with metadata.open("ab") as destination:
                destination.write(b" " * (video._MAX_VIDEO_JOB_METADATA_BYTES + 1))
        return opened

    monkeypatch.setattr(video.os, "fstat", expand_after_validation)

    assert video._load_completed_job(job_dir) is None
    assert expanded is True


@pytest.mark.asyncio
async def test_retrieve_missing_video_returns_not_found() -> None:
    with pytest.raises(video.HTTPException) as exc:
        await video.retrieve_video("video_" + "1" * 32)

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_download_rejects_noncompleted_video() -> None:
    job = video._VideoJob(
        id="video_" + "0" * 32,
        model="ltx-2.3-mlx-q4",
        prompt="Not finished",
        seconds="1",
        size="512x512",
        status="in_progress",
        created_at=1,
    )
    video._jobs[job.id] = job

    with pytest.raises(video.HTTPException) as exc:
        await video.retrieve_video_content(job.id)

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_download_closes_output_descriptor_when_wrapping_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    job = _completed_job("video_" + "d" * 32)
    _write_completed_job(job)
    video.start_video_jobs()

    def fail_fdopen(fd: int, mode: str):
        raise OSError("cannot wrap descriptor")

    monkeypatch.setattr(video.os, "fdopen", fail_fdopen)

    with pytest.raises(video.HTTPException) as exc:
        await video.retrieve_video_content(job.id)

    assert exc.value.status_code == 410


def test_restore_scan_failure_is_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "videos")

    def fail_scan(self):
        raise OSError("offline volume")

    monkeypatch.setattr(Path, "iterdir", fail_scan)
    assert video._restore_completed_jobs() == []


def test_restore_enforces_existing_hundred_job_retention(tmp_path: Path) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)
    oldest_id = "video_" + f"{0:032x}"
    for index in range(video._MAX_JOBS + 1):
        job_id = "video_" + f"{index:032x}"
        _write_completed_job(_completed_job(job_id, created_at=index))

    video.start_video_jobs()

    assert len(video._jobs) == video._MAX_JOBS
    assert oldest_id not in video._jobs
    assert not (store / oldest_id).exists()


def test_same_process_restart_rebuilds_registry_from_disk(tmp_path: Path) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)
    retained = _completed_job("video_" + "a" * 32, created_at=2)
    removed = _completed_job("video_" + "b" * 32, created_at=1)
    _write_completed_job(retained)
    _write_completed_job(removed)
    video.start_video_jobs()
    assert set(video._jobs) == {retained.id, removed.id}

    # Model a stopped server whose operator removed one artifact, while stale
    # failed state remains in this process from the previous lifespan.
    (store / removed.id / "job.json").unlink()
    failed = video._VideoJob(
        id="video_" + "c" * 32,
        model="ltx-2.3-mlx-q4",
        prompt="Stale failed job",
        seconds="1",
        size="512x512",
        status="failed",
        created_at=3,
        generation_finished=True,
    )
    video._jobs[failed.id] = failed

    video.start_video_jobs()

    assert set(video._jobs) == {retained.id}


def test_failed_metadata_replace_leaves_no_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    job = _completed_job("video_" + "d" * 32)
    job_dir = video._jobs_root / job.id
    job_dir.mkdir()
    (job_dir / "output.mp4").write_bytes(b"mp4")

    def fail_replace(source, destination, **kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(video.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        video._persist_completed_job(job)

    assert not (job_dir / "job.json").exists()
    assert not list(job_dir.glob(".job-*.json.tmp"))


def test_persistence_orders_crash_consistency_barriers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)
    job = _completed_job("video_" + "7" * 32)
    job_dir = store / job.id
    job_dir.mkdir()
    output = job_dir / "output.mp4"
    output.write_bytes(b"mp4")
    events: list[str] = []
    original_fsync = video.os.fsync
    original_replace = video.os.replace

    def track_fsync(fd: int) -> None:
        opened = video.os.fstat(fd)
        if opened.st_ino == output.stat().st_ino:
            events.append("output")
        elif stat.S_ISREG(opened.st_mode):
            events.append("manifest")
        elif opened.st_ino == job_dir.stat().st_ino:
            events.append("job-directory")
        elif opened.st_ino == store.stat().st_ino:
            events.append("store-directory")
        original_fsync(fd)

    def track_replace(source, destination, **kwargs) -> None:
        events.append("replace")
        original_replace(source, destination, **kwargs)

    monkeypatch.setattr(video.os, "fsync", track_fsync)
    monkeypatch.setattr(video.os, "replace", track_replace)

    video._persist_completed_job(job)

    assert events == [
        "output",
        "manifest",
        "replace",
        "job-directory",
        "store-directory",
    ]
    assert (job_dir / "job.json").is_file()


@pytest.mark.asyncio
async def test_metadata_failure_keeps_completed_video_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    video.start_video_jobs()

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    def fail_persist(job) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    monkeypatch.setattr(video, "_persist_completed_job", fail_persist)
    created = await video.create_video(
        prompt="Keep the completed output",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=2,
        input_reference=None,
    )

    assert (await _wait_for_completion(created["id"]))["status"] == "completed"
    assert "Unable to persist completed video job metadata" in caplog.text
    response = await video.retrieve_video_content(created["id"])
    assert (
        b"".join([chunk async for chunk in response.body_iterator]) == b"generated-mp4"
    )
    await video.delete_video(created["id"])


@pytest.mark.asyncio
async def test_persistence_thread_start_failure_keeps_completed_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)
    video.start_video_jobs()

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    original_start = video.threading.Thread.start

    def fail_persistence_start(thread: threading.Thread) -> None:
        if thread.name == "rapid-mlx-video-persistence":
            raise RuntimeError("cannot start thread")
        original_start(thread)

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    monkeypatch.setattr(video.threading.Thread, "start", fail_persistence_start)
    created = await video.create_video(
        prompt="Keep output after thread exhaustion",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=8,
        input_reference=None,
    )

    assert (await _wait_for_completion(created["id"]))["status"] == "completed"
    assert not video._persistence_threads
    assert "Unable to persist completed video job metadata" in caplog.text
    response = await video.retrieve_video_content(created["id"])
    assert (
        b"".join([chunk async for chunk in response.body_iterator]) == b"generated-mp4"
    )
    await asyncio.sleep(0)
    assert video.configure_video_jobs(store) == store.resolve()


@pytest.mark.asyncio
async def test_shutdown_does_not_wait_for_blocked_metadata_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    video.start_video_jobs()
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    persist_completed_job = video._persist_completed_job

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    def block_persist(job) -> None:
        persistence_started.set()
        release_persistence.wait(timeout=5)
        persist_completed_job(job)

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    monkeypatch.setattr(video, "_persist_completed_job", block_persist)
    created = await video.create_video(
        prompt="Finish within shutdown budget",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=3,
        input_reference=None,
    )

    try:
        assert await asyncio.to_thread(persistence_started.wait, 1)
        await asyncio.wait_for(video.shutdown_video_jobs(timeout=0), timeout=0.5)
        assert (await video.retrieve_video(created["id"]))["status"] == "completed"
        video.start_video_jobs()
        assert created["id"] not in video._jobs
        oldest_id = "video_" + f"{0:032x}"
        for index in range(video._MAX_JOBS):
            restored = _completed_job("video_" + f"{index:032x}", created_at=index)
            video._jobs[restored.id] = restored
        release_persistence.set()
        for _ in range(100):
            if not video._persistence_threads and created["id"] in video._jobs:
                break
            await asyncio.sleep(0.01)
        assert (await video.retrieve_video(created["id"]))["status"] == "completed"
        assert (video._jobs_root / created["id"] / "job.json").is_file()
        assert len(video._jobs) == video._MAX_JOBS
        assert oldest_id not in video._jobs
    finally:
        release_persistence.set()
        for _ in range(100):
            if not video._persistence_threads:
                break
            await asyncio.sleep(0.01)
        assert not video._persistence_threads


@pytest.mark.asyncio
async def test_download_rejects_output_symlink_swapped_after_restore(
    tmp_path: Path,
) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)
    job = _completed_job("video_" + "9" * 32)
    _write_completed_job(job)
    video.start_video_jobs()

    external = tmp_path / "private.txt"
    external.write_bytes(b"server-readable-secret")
    output = store / job.id / "output.mp4"
    output.unlink()
    output.symlink_to(external)

    with pytest.raises(video.HTTPException) as exc:
        await video.retrieve_video_content(job.id)

    assert exc.value.status_code == 410


@pytest.mark.asyncio
async def test_download_rejects_nonregular_output_swapped_after_restore(
    tmp_path: Path,
) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)
    job = _completed_job("video_" + "8" * 32)
    _write_completed_job(job)
    video.start_video_jobs()

    output = store / job.id / "output.mp4"
    output.unlink()
    output.mkdir()

    with pytest.raises(video.HTTPException) as exc:
        await video.retrieve_video_content(job.id)

    assert exc.value.status_code == 410


@pytest.mark.asyncio
async def test_download_rejects_job_directory_symlink_swapped_after_restore(
    tmp_path: Path,
) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)
    job = _completed_job("video_" + "6" * 32)
    _write_completed_job(job)
    video.start_video_jobs()

    original = store / job.id
    original.rename(store / f"{job.id}-original")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "output.mp4").write_bytes(b"server-readable-data")
    original.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(video.HTTPException) as exc:
        await video.retrieve_video_content(job.id)

    assert exc.value.status_code == 410


@pytest.mark.asyncio
async def test_shutdown_cancels_queued_job_with_active_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            started.set()
            release.wait(timeout=5)
            output_path.write_bytes(b"generated-mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: BlockingEngine())
    first = await video.create_video(
        prompt="Active job",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=4,
        input_reference=None,
    )
    assert await asyncio.to_thread(started.wait, 1)
    queued = await video.create_video(
        prompt="Queued job",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=5,
        input_reference=None,
    )

    try:
        await video.shutdown_video_jobs(timeout=0)
        assert video._jobs[queued["id"]].error == {
            "code": "video_server_shutdown",
            "message": "Video generation was cancelled during server shutdown.",
        }
        assert video._jobs[queued["id"]].generation_finished is True
    finally:
        release.set()
        for _ in range(200):
            if not video._generation_threads and not video._cleanup_tasks:
                break
            await asyncio.sleep(0.01)
        assert not video._generation_threads
        assert first["id"] in video._jobs


@pytest.mark.asyncio
async def test_failed_job_never_enters_generation_gate() -> None:
    job = video._VideoJob(
        id="video_" + "5" * 32,
        model="ltx-2.3-mlx-q4",
        prompt="Already failed",
        seconds="1",
        size="512x512",
        status="failed",
        created_at=1,
    )

    class UnexpectedEngine:
        video_family = "ltx-2.3"

        def generate(self, **kwargs) -> None:
            raise AssertionError("failed job reached generation")

    await video._run_job(
        job,
        engine=UnexpectedEngine(),
        width=512,
        height=512,
        num_frames=25,
        fps=24,
        seed=6,
        image_path=None,
        negative_prompt=None,
        guidance_scale=None,
        conditioning_strength=None,
    )

    assert job.status == "failed"


@pytest.mark.asyncio
async def test_new_job_evicts_oldest_finished_job_at_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oldest_id = "video_" + f"{0:032x}"
    for index in range(video._MAX_JOBS):
        job = _completed_job("video_" + f"{index:032x}", created_at=index)
        video._jobs[job.id] = job

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="Newest result",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=7,
        input_reference=None,
    )
    await _wait_for_completion(created["id"])

    assert oldest_id not in video._jobs
    assert created["id"] in video._jobs


def test_video_output_directory_rejects_a_file(tmp_path: Path) -> None:
    destination = tmp_path / "not-a-directory"
    destination.write_text("occupied", encoding="utf-8")

    with pytest.raises((FileExistsError, ValueError)):
        video.configure_video_jobs(destination)


def test_video_store_rejects_blank_path_and_live_reconfiguration(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        video.configure_video_jobs("   ")

    marker = threading.current_thread()
    video._generation_threads.add(marker)
    try:
        with pytest.raises(RuntimeError, match="while jobs run"):
            video.configure_video_jobs(tmp_path / "videos")
    finally:
        video._generation_threads.discard(marker)


def test_ephemeral_cleanup_visits_every_owned_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[Path] = []
    monkeypatch.setattr(
        video.shutil,
        "rmtree",
        lambda root, *, ignore_errors: removed.append(Path(root)),
    )

    video._cleanup_jobs()

    assert set(removed) == video._ephemeral_jobs_roots


def test_unified_serve_parser_exposes_video_output_directory(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        ["serve", "ltx-2.3-mlx-q4", "--video-output-dir", str(tmp_path)]
    )

    assert args.video_output_dir == str(tmp_path)


def test_both_server_entrypoints_configure_the_shared_video_store() -> None:
    unified_source = inspect.getsource(cli.serve_command)
    standalone_source = inspect.getsource(server.main)

    assert (
        'configure_video_jobs(getattr(args, "video_output_dir", None))'
        in unified_source
    )
    assert "_add_video_job_args_to_server_parser(parser)" in standalone_source
    assert "configure_video_jobs(args.video_output_dir)" in standalone_source
    assert unified_source.index("configure_video_jobs(") < unified_source.index(
        "_ensure_model_downloaded("
    )


def test_unified_serve_reports_video_store_configuration_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        video,
        "configure_video_jobs",
        lambda output_dir: (_ for _ in ()).throw(OSError("read-only")),
    )

    with pytest.raises(SystemExit) as exc:
        cli.serve_command(SimpleNamespace(video_output_dir="/unwritable"))

    assert exc.value.code == 2
    assert (
        "cannot configure video output directory: read-only" in capsys.readouterr().err
    )


def test_standalone_server_reports_video_store_configuration_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        video,
        "configure_video_jobs",
        lambda output_dir: (_ for _ in ()).throw(OSError("read-only")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm_mlx.server", "--video-output-dir", "/unwritable"],
    )

    with pytest.raises(SystemExit) as exc:
        server.main()

    assert exc.value.code == 2
    assert (
        "cannot configure video output directory: read-only" in capsys.readouterr().err
    )
