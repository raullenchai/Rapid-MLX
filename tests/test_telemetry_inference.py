# SPDX-License-Identifier: Apache-2.0
"""Telemetry-v2 inference and capability emitter contracts."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import threading
import time
import types
import urllib.request
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.requests import Request

REPO_ROOT = Path(__file__).resolve().parents[1]


def _request(
    user_agent: str = "openai-python/1.2", client: str = "rapid-cli-chat"
) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [
                (b"user-agent", user_agent.encode()),
                (b"x-rapid-client", client.encode()),
            ],
        }
    )


def _hold_telemetry_store_lock(home: str, ready, release) -> None:
    os.environ["HOME"] = home
    from rapid_mlx.telemetry import store

    store.record("lock-holder-seed")
    connection = sqlite3.connect(store.db_path())
    connection.execute("BEGIN IMMEDIATE")
    ready.set()
    release.wait(10)
    connection.rollback()
    connection.close()


@pytest.fixture(autouse=True)
def _isolated_home_and_kill_switches(monkeypatch, tmp_path):
    from rapid_mlx.telemetry import inference, state

    # The production worker is process-wide. Keep callbacks submitted by one
    # test from observing the next test's temporary HOME or monkeypatches.
    inference._QUEUE.join()
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    yield
    inference._QUEUE.join()


def test_completed_inference_records_one_normalized_counter_and_claims_success(
    monkeypatch,
):
    from rapid_mlx.telemetry import inference

    crossing = SimpleNamespace(bucket="3_4", bucket_source="crossed_now")
    records: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    active_days: list[None] = []
    monkeypatch.setattr(
        inference.store, "record", lambda key: records.append(key) or crossing
    )
    monkeypatch.setattr(
        inference.track_module,
        "track",
        lambda event, props: events.append((event, dict(props))),
    )
    monkeypatch.setattr(
        inference.track_module,
        "emit_active_day",
        lambda: active_days.append(None),
    )

    inference._record_completed_request(
        model="neohorse-9b-4bit",
        endpoint="https://localhost/v1/chat/completions?token=secret",
        caller_agent="private-agent/99 cursor/1.0",
        caller_client=None,
        result="ok",
    )

    assert records == ["inf|neohorse-9b-4bit|/v1/chat/completions|cursor|ok"]
    assert active_days == [None]
    assert events == [
        (
            "inference_bucket_reached",
            {
                "model": "neohorse-9b-4bit",
                "endpoint": "/v1/chat/completions",
                "caller": "cursor",
                "result": "ok",
                "count_bucket": "3_4",
                "bucket_source": "crossed_now",
            },
        )
    ]
    assert "secret" not in repr(records + events)


@pytest.mark.parametrize(
    "endpoint",
    [
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/images/generations",
        "/v1/audio/transcriptions",
    ],
)
def test_rapid_desktop_caller_survives_registry_validation_before_record(
    monkeypatch, endpoint
):
    from rapid_mlx.telemetry import inference, registry

    crossing = SimpleNamespace(bucket="1", bucket_source="crossed_now")
    keys: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        inference.store, "record", lambda key: keys.append(key) or crossing
    )
    monkeypatch.setattr(
        inference.track_module,
        "track",
        lambda event, props: (
            events.append((event, validated))
            if (validated := registry.validate(event, props)) is not None
            else None
        ),
    )
    monkeypatch.setattr(inference.track_module, "emit_active_day", lambda: None)

    inference._record_completed_request(
        model="<custom>",
        endpoint=endpoint,
        caller_agent="hostile-private-agent/1.0",
        caller_client="rapid-desktop",
        result="ok",
    )

    assert keys == [f"inf|<custom>|{endpoint}|rapid-desktop|ok"]
    assert events[0][1]["caller"] == "rapid-desktop"


def test_unknown_normalized_caller_falls_back_before_record(monkeypatch):
    from rapid_mlx.telemetry import inference

    keys: list[str] = []
    monkeypatch.setattr(
        inference.redact, "normalize_caller_agent", lambda *_args: "future-client"
    )
    monkeypatch.setattr(inference.store, "record", lambda key: keys.append(key))
    monkeypatch.setattr(inference.track_module, "emit_active_day", lambda: None)

    inference._record_completed_request(
        model="<custom>",
        endpoint="/v1/chat/completions",
        caller_agent="hostile-private-agent/1.0",
        caller_client="hostile-client",
        result="ok",
    )

    assert keys == ["inf|<custom>|/v1/chat/completions|other|ok"]


def test_completed_inference_failure_does_not_claim_active_day(monkeypatch):
    from rapid_mlx.telemetry import inference

    active_days: list[None] = []
    monkeypatch.setattr(inference.store, "record", lambda _key: None)
    monkeypatch.setattr(
        inference.track_module,
        "emit_active_day",
        lambda: active_days.append(None),
    )

    inference._record_completed_request(
        model="<custom>",
        endpoint="/not-registered",
        caller_agent=None,
        caller_client=None,
        result="failed",
    )

    assert active_days == []


class _TypeErrorEndpoint:
    def decode(self, *_args):
        raise TypeError("malformed non-string endpoint")


@pytest.mark.parametrize("endpoint", [None, _TypeErrorEndpoint()])
def test_completed_inference_with_non_string_endpoint_falls_back_to_other(
    monkeypatch, endpoint
):
    from rapid_mlx.telemetry import inference

    records: list[str] = []
    monkeypatch.setattr(inference.track_module, "_upload_allowed", lambda: True)
    monkeypatch.setattr(
        inference.store, "record", lambda key: records.append(key) or None
    )

    inference.emit_completed_request(
        model="<custom>",
        endpoint=endpoint,
        caller_agent=None,
        caller_client=None,
        result="failed",
    )
    inference._QUEUE.join()

    assert records == ["inf|<custom>|other|unknown|failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("eligible_case", ["unofficial", "opted_out"])
async def test_ineligible_completed_request_creates_no_home_state(
    monkeypatch, tmp_path, eligible_case
):
    from rapid_mlx.telemetry import consent_runtime, inference
    from rapid_mlx.telemetry import track as track_module

    if eligible_case == "unofficial":
        monkeypatch.setattr(track_module.build_gate, "official_build", lambda: None)
        monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    else:
        monkeypatch.setattr(
            track_module.build_gate,
            "official_build",
            lambda: SimpleNamespace(channel="stable"),
        )
        monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: False)

    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    inference.emit_completed_request(
        model="<custom>",
        endpoint="/v1/chat/completions",
        caller_agent=None,
        caller_client=None,
        result="ok",
    )
    await asyncio.sleep(0)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert not (tmp_path / ".rapid-mlx" / "telemetry.db").exists()


@pytest.mark.asyncio
async def test_locked_store_never_blocks_request_coroutine(monkeypatch, tmp_path):
    from rapid_mlx.telemetry import inference

    monkeypatch.setattr(inference.track_module, "_upload_allowed", lambda: True)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_telemetry_store_lock,
        args=(str(tmp_path), ready, release),
    )
    process.start()
    assert ready.wait(5), "lock-holder process did not acquire SQLite write lock"

    worker_finished = threading.Event()
    real_record = inference.store.record

    def observed_record(key):
        try:
            return real_record(key)
        finally:
            worker_finished.set()

    monkeypatch.setattr(inference.store, "record", observed_record)
    started = time.perf_counter()
    inference.emit_completed_request(
        model="<custom>",
        endpoint="/v1/chat/completions",
        caller_agent=None,
        caller_client=None,
        result="ok",
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 0.050, f"request-path telemetry took {elapsed * 1000:.2f} ms"

    default_started = time.perf_counter()
    await asyncio.to_thread(lambda: None)
    default_elapsed = time.perf_counter() - default_started
    assert default_elapsed < 0.050, (
        "telemetry occupied the loop default executor for "
        f"{default_elapsed * 1000:.2f} ms"
    )

    release.set()
    loop = asyncio.get_running_loop()
    assert await loop.run_in_executor(None, worker_finished.wait, 5)
    process.join(timeout=5)
    assert process.exitcode == 0


def test_worker_start_failure_never_escapes(monkeypatch):
    from rapid_mlx.telemetry import inference

    monkeypatch.setattr(inference.track_module, "_upload_allowed", lambda: True)
    monkeypatch.setattr(inference, "_ensure_worker", lambda: False)

    assert (
        inference.emit_completed_request(
            model="<custom>",
            endpoint="/v1/chat/completions",
            caller_agent=None,
            caller_client=None,
            result="ok",
        )
        is None
    )


@pytest.mark.parametrize("emitter", ["completed", "capability"])
def test_public_emitter_gate_failure_never_escapes(monkeypatch, emitter):
    from rapid_mlx.telemetry import inference

    monkeypatch.setattr(
        inference.track_module,
        "_upload_allowed",
        lambda: (_ for _ in ()).throw(RuntimeError("gate failed")),
    )

    if emitter == "completed":
        inference.emit_completed_request(
            model="<custom>",
            endpoint="/v1/chat/completions",
            caller_agent=None,
            caller_client=None,
            result="ok",
        )
    else:
        inference.emit_capability_rejected("multi_sample_unsupported")


def test_request_caller_headers_is_total():
    from rapid_mlx.telemetry.inference import request_caller_headers

    class HostileRequest:
        @property
        def headers(self):
            raise RuntimeError("bad request object")

    assert request_caller_headers(None) == (None, None)
    assert request_caller_headers(HostileRequest()) == (None, None)


def test_worker_overflow_drops_without_blocking(monkeypatch):
    from rapid_mlx.telemetry import inference

    release = threading.Event()
    started = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(5)

    monkeypatch.setattr(
        inference, "_QUEUE", inference.queue.Queue(maxsize=inference._MAX_PENDING)
    )
    monkeypatch.setattr(inference, "_WORKER", None)
    monkeypatch.setattr(inference, "_SHUTTING_DOWN", False)
    try:
        assert inference._submit(blocked)
        assert started.wait(1)
        for _ in range(inference._MAX_PENDING):
            assert inference._submit(blocked)
        before = time.perf_counter()
        assert inference._submit(blocked) is False
        assert time.perf_counter() - before < 0.050
    finally:
        release.set()


def test_asyncio_run_does_not_drain_saturated_telemetry_lane(monkeypatch, tmp_path):
    from rapid_mlx.telemetry import inference

    monkeypatch.setattr(inference.track_module, "_upload_allowed", lambda: True)
    monkeypatch.setattr(
        inference,
        "_QUEUE",
        inference.queue.Queue(maxsize=inference._MAX_PENDING),
    )
    monkeypatch.setattr(inference, "_WORKER", None)
    monkeypatch.setattr(inference, "_SHUTTING_DOWN", False)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_telemetry_store_lock,
        args=(str(tmp_path), ready, release),
    )
    process.start()
    assert ready.wait(5), "lock-holder process did not acquire SQLite write lock"

    async def enqueue() -> None:
        for _ in range(200):
            inference.emit_completed_request(
                model="<custom>",
                endpoint="/v1/chat/completions",
                caller_agent=None,
                caller_client=None,
                result="ok",
            )

    try:
        started = time.perf_counter()
        asyncio.run(enqueue())
        elapsed = time.perf_counter() - started
        assert elapsed < 3.0, f"asyncio.run drained telemetry for {elapsed:.2f}s"
    finally:
        release.set()
        process.join(timeout=5)
    assert process.exitcode == 0


def test_process_exit_does_not_join_blocked_telemetry_worker(tmp_path):
    home = tmp_path / "child-home"
    home.mkdir()
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.bodies = []  # type: ignore[attr-defined]
    sink_thread = threading.Thread(target=server.serve_forever, daemon=True)
    sink_thread.start()
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "USER": "rc",
            "RAPID_MLX_POSTHOG_URL": (f"http://127.0.0.1:{server.server_port}/batch/"),
        }
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from rapid_mlx.telemetry import store; store.record('seed')",
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            timeout=3,
        )
        db = home / ".rapid-mlx" / "telemetry.db"
        connection = sqlite3.connect(db)
        connection.execute("BEGIN IMMEDIATE")
        script = """
from rapid_mlx.telemetry import inference
inference.track_module._upload_allowed = lambda: True
for _ in range(16):
    inference.emit_completed_request(
        model='<custom>', endpoint='/v1/chat/completions',
        caller_agent=None, caller_client=None, result='ok')
"""
        started = time.perf_counter()
        try:
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                timeout=3,
            )
        finally:
            connection.rollback()
            connection.close()
        assert time.perf_counter() - started < 3.0
    finally:
        server.shutdown()
        sink_thread.join(timeout=2)
        server.server_close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_forked_child_starts_fresh_telemetry_worker(monkeypatch):
    from rapid_mlx.telemetry import inference

    release = threading.Event()
    started = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(5)

    monkeypatch.setattr(
        inference, "_QUEUE", inference.queue.Queue(maxsize=inference._MAX_PENDING)
    )
    monkeypatch.setattr(inference, "_WORKER", None)
    monkeypatch.setattr(inference, "_SHUTTING_DOWN", False)
    assert inference._submit(blocked)
    assert started.wait(1)

    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:  # pragma: no branch - child exits directly
        try:
            os.close(read_fd)

            def report() -> None:
                os.write(write_fd, b"ran")
                os._exit(0)

            accepted = inference._submit(report)
            if not accepted:
                os._exit(2)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                time.sleep(0.01)
            os._exit(3)
        except BaseException:
            os._exit(4)

    os.close(write_fd)
    try:
        ready, _, _ = __import__("select").select([read_fd], [], [], 3)
        assert ready and os.read(read_fd, 3) == b"ran"
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        os.close(read_fd)
        release.set()


@pytest.mark.asyncio
async def test_midstream_generation_error_records_failed_without_active_day(
    monkeypatch,
):
    from rapid_mlx.telemetry import inference

    # The production daemon intentionally outlives individual requests. Establish
    # a clean observation boundary before replacing its global worker hooks.
    inference._QUEUE.join()
    monkeypatch.setattr(inference.track_module, "_upload_allowed", lambda: True)
    events: list[tuple[str, dict[str, object]]] = []
    active_days: list[None] = []
    captured = threading.Event()

    def capture(event, props):
        events.append((event, dict(props)))
        captured.set()

    monkeypatch.setattr(inference.track_module, "track", capture)
    monkeypatch.setattr(
        inference.track_module,
        "emit_active_day",
        lambda: active_days.append(None),
    )

    async def broken_stream():
        yield "first-token"
        raise RuntimeError("generation failed")

    guarded = inference.emit_failed_on_stream_error(
        broken_stream(),
        model="private hostile model?",
        endpoint="/v1/chat/completions",
        caller_agent="cursor/1.0",
        caller_client=None,
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        async for _ in guarded:
            pass

    loop = asyncio.get_running_loop()
    assert await loop.run_in_executor(None, captured.wait, 5)
    assert active_days == []
    assert events == [
        (
            "inference_bucket_reached",
            {
                "model": "<custom>",
                "endpoint": "/v1/chat/completions",
                "caller": "cursor",
                "result": "failed",
                "count_bucket": "1",
                "bucket_source": "crossed_now",
            },
        )
    ]


@pytest.mark.asyncio
async def test_stream_cancellation_does_not_record_failed(monkeypatch):
    from rapid_mlx.telemetry import inference

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        inference, "emit_completed_request", lambda **kwargs: calls.append(kwargs)
    )

    async def cancelled_stream():
        yield "first-token"
        raise asyncio.CancelledError

    guarded = inference.emit_failed_on_stream_error(
        cancelled_stream(),
        model="<custom>",
        endpoint="/v1/chat/completions",
        caller_agent=None,
        caller_client=None,
    )
    with pytest.raises(asyncio.CancelledError):
        async for _ in guarded:
            pass

    assert calls == []


@pytest.mark.parametrize("failure", ["record", "track", "active_day"])
def test_completed_inference_never_raises(monkeypatch, failure):
    from rapid_mlx.telemetry import inference

    def explode(*_args, **_kwargs):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(inference.store, "record", lambda _key: None)
    monkeypatch.setattr(inference.track_module, "track", lambda *_a, **_k: None)
    monkeypatch.setattr(inference.track_module, "emit_active_day", lambda: None)
    target = inference.store if failure == "record" else inference.track_module
    attribute = {
        "record": "record",
        "track": "track",
        "active_day": "emit_active_day",
    }[failure]
    monkeypatch.setattr(target, attribute, explode)

    assert (
        inference._record_completed_request(
            model="<custom>",
            endpoint="/v1/completions",
            caller_agent=None,
            caller_client=None,
            result="ok",
        )
        is None
    )


def test_capability_rejected_requires_closed_model_type_and_never_raises(monkeypatch):
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        inference.track_module,
        "track",
        lambda event, props: calls.append((event, dict(props))),
    )
    inference._record_capability_rejected(
        capability="mcp_unsupported", model_type="other"
    )
    inference._record_capability_rejected(
        capability="logprobs_unsupported",
        model_type=inference.model_type_token(object()),
    )
    inference._record_capability_rejected(
        capability="mcp_unsupported", model_type="private-type"
    )
    inference._record_capability_rejected(
        capability="private_capability_name", model_type="other"
    )
    monkeypatch.setattr(inference.track_module, "track", lambda *_a, **_k: 1 / 0)
    inference._record_capability_rejected(
        capability="mcp_unsupported", model_type="other"
    )

    assert calls == [
        (
            "capability_rejected",
            {"capability": "mcp_unsupported", "model_type": "other"},
        ),
        (
            "capability_rejected",
            {"capability": "logprobs_unsupported", "model_type": "llm"},
        ),
        (
            "capability_rejected",
            {"capability": "mcp_unsupported", "model_type": "other"},
        ),
    ]


def test_daemon_worker_lifecycle_branches_are_best_effort(monkeypatch):
    from rapid_mlx.telemetry import inference

    work_queue = inference.queue.Queue()
    completed = threading.Event()

    def explode():
        raise RuntimeError("discarded telemetry failure")

    worker = threading.Thread(
        target=inference._worker_main,
        args=(work_queue,),
        daemon=True,
    )
    worker.start()
    work_queue.put(explode)
    work_queue.put(completed.set)
    work_queue.join()
    assert completed.is_set()

    monkeypatch.setattr(inference, "_SHUTTING_DOWN", True)
    assert inference._ensure_worker() is False
    monkeypatch.setattr(inference, "_SHUTTING_DOWN", False)
    monkeypatch.setattr(
        inference.threading,
        "Thread",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("thread unavailable")),
    )
    monkeypatch.setattr(inference, "_WORKER", None)
    assert inference._ensure_worker() is False


def test_exit_hook_drops_queue_and_fork_hook_resets_state(monkeypatch):
    from rapid_mlx.telemetry import inference

    pending = inference.queue.Queue(maxsize=inference._MAX_PENDING)
    pending.put(lambda: None)
    pending.put(lambda: None)
    monkeypatch.setattr(inference, "_QUEUE", pending)
    monkeypatch.setattr(inference, "_SHUTTING_DOWN", False)

    inference._drop_pending_at_exit()

    assert inference._SHUTTING_DOWN is True
    assert pending.empty()
    assert pending.unfinished_tasks == 0

    inherited_queue = inference._QUEUE
    inherited_lock = inference._WORKER_LOCK
    monkeypatch.setattr(inference, "_WORKER", object())
    inference._after_fork_child()

    assert inference._QUEUE is not inherited_queue
    assert inference._QUEUE.maxsize == inference._MAX_PENDING
    assert inference._WORKER is None
    assert inference._WORKER_LOCK is not inherited_lock
    assert inference._SHUTTING_DOWN is False


@pytest.mark.asyncio
async def test_capability_rejection_never_blocks_loop_default_executor(monkeypatch):
    from rapid_mlx.telemetry import inference

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        inference,
        "_QUEUE",
        inference.queue.Queue(maxsize=inference._MAX_PENDING),
    )
    monkeypatch.setattr(inference, "_WORKER", None)
    monkeypatch.setattr(inference, "_SHUTTING_DOWN", False)
    monkeypatch.setattr(inference.track_module, "_upload_allowed", lambda: True)

    def blocked_track(*_args, **_kwargs):
        entered.set()
        release.wait(5)

    monkeypatch.setattr(inference.track_module, "track", blocked_track)
    try:
        inference.emit_capability_rejected("multi_sample_unsupported")
        assert entered.wait(1)
        started = time.perf_counter()
        await asyncio.to_thread(lambda: None)
        assert time.perf_counter() - started < 0.050
    finally:
        release.set()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (None, "other"),
        (SimpleNamespace(modality="text", supports_image_input=False), "llm"),
        (SimpleNamespace(modality="text", supports_image_input=True), "vlm"),
        (SimpleNamespace(modality="embedding"), "embedding"),
        (SimpleNamespace(is_image_gen=True), "image-gen"),
        (SimpleNamespace(is_video_gen=True), "video-gen"),
        (SimpleNamespace(is_mllm=True), "vlm"),
        (object(), "llm"),
    ],
)
def test_model_type_token_uses_only_registry_values(source, expected):
    from rapid_mlx.telemetry.inference import model_type_token

    assert model_type_token(source) == expected


def test_model_type_token_covers_engine_flags_and_hostile_objects():
    from rapid_mlx.telemetry.inference import model_type_token

    class Hostile:
        def __getattr__(self, _name):
            raise RuntimeError("hostile engine")

    assert model_type_token(SimpleNamespace(is_embedding=True)) == "embedding"
    assert model_type_token(SimpleNamespace(is_audio=True)) == "audio"
    assert model_type_token(Hostile()) == "other"


def test_model_type_tokens_match_registry():
    from rapid_mlx.telemetry import inference, registry

    declared = frozenset(registry.load_registry()["enums"]["model_type"]["values"])
    assert declared == inference._MODEL_TYPES


@pytest.mark.asyncio
async def test_legacy_completion_multi_sample_rejection_emits_capability(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.api.models import CompletionRequest
    from rapid_mlx.routes import completions
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(completions, "_validate_model_name", lambda _model: None)
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda capability, *, model_type="other": calls.append(
            (capability, model_type)
        ),
    )
    request = CompletionRequest.model_construct(
        model="ignored-routing-name", prompt="hello", n=2, suffix=None
    )

    with pytest.raises(HTTPException, match="n > 1"):
        await completions.create_completion(request, SimpleNamespace(headers={}))

    assert calls == [("multi_sample_unsupported", "other")]


@pytest.mark.asyncio
async def test_chat_multi_sample_rejection_emits_capability(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.routes import chat
    from rapid_mlx.telemetry import inference

    engine = SimpleNamespace(modality="text", supports_image_input=False)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda capability, *, model_type="other": calls.append(
            (capability, model_type)
        ),
    )
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hello"}]
    ).model_copy(update={"n": 2})

    with pytest.raises(HTTPException, match="n > 1"):
        await chat._create_chat_completion_impl(
            request,
            _request(),
            engine,
            _commit_state=[False],
            _admission_acquired=[False],
        )

    assert calls == [("multi_sample_unsupported", "llm")]


@pytest.mark.asyncio
async def test_responses_stateless_rejection_emits_capability(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.routes import responses
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )
    body = json.dumps(
        {
            "model": "test-model",
            "input": "hello",
            "previous_response_id": "resp_prior",
        }
    ).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/responses", "headers": []},
        receive,
    )
    with pytest.raises(HTTPException) as exc_info:
        await responses.create_response(request)

    assert exc_info.value.status_code == 400
    assert calls == [("stateless_api_only", "other")]


@pytest.mark.asyncio
async def test_residency_perf_rejection_emits_capability(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.routes import residency
    from rapid_mlx.telemetry import inference

    profile = SimpleNamespace(modality="image-gen")
    monkeypatch.setattr(residency, "_manager", lambda: object())
    monkeypatch.setattr(residency, "resolve_profile", lambda _model: profile)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )

    request = residency.ModelLoadRequest(
        model="flux-schnell",
        performance=residency.ModelPerformanceRequest(prefix_cache_enabled=True),
    )
    with pytest.raises(HTTPException) as exc_info:
        await residency.load_resident_model(request)

    assert exc_info.value.status_code == 422
    assert calls == [("perf_overrides_unsupported", "image-gen")]


def test_context_length_rejection_emits_capability(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.service import helpers
    from rapid_mlx.telemetry import inference

    engine = SimpleNamespace(modality="text", supports_image_input=False)
    monkeypatch.setattr(helpers, "get_model_max_context", lambda _engine: 8)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )

    with pytest.raises(HTTPException) as exc_info:
        helpers.enforce_context_length(engine, 8, max_tokens=1)

    assert exc_info.value.status_code == 400
    assert calls == [("context_length_exceeded", "llm")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "capability"),
    [
        ({"best_of": 2}, "multi_sample_unsupported"),
        ({"echo": True, "logprobs": 1}, "logprobs_unsupported"),
    ],
)
async def test_legacy_completion_early_rejections_emit_capability(
    monkeypatch, fields, capability
):
    from fastapi import HTTPException

    from rapid_mlx.api.models import CompletionRequest
    from rapid_mlx.routes import completions
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(completions, "_validate_model_name", lambda _model: None)
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )
    values = {"model": "ignored", "prompt": "hello", "suffix": None, "n": 1}
    values.update(fields)
    request = CompletionRequest.model_construct(**values)

    with pytest.raises(HTTPException):
        await completions.create_completion(request, _request())

    assert calls == [(capability, "other")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "capability"),
    [
        ({"suffix": "tail"}, "fim_suffix_unsupported"),
        (
            {"response_format": {"type": "json_schema"}, "logprobs": None},
            "structured_output_unsupported",
        ),
    ],
)
async def test_legacy_completion_format_rejections_emit_capability(
    monkeypatch, fields, capability
):
    from fastapi import HTTPException

    from rapid_mlx.api.models import CompletionRequest
    from rapid_mlx.routes import completions
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(completions, "_validate_model_name", lambda _model: None)
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )
    values = {
        "model": "ignored",
        "prompt": "hello",
        "suffix": None,
        "n": 1,
        "best_of": 1,
        "echo": False,
    }
    values.update(fields)
    request = CompletionRequest.model_construct(**values)

    with pytest.raises(HTTPException):
        await completions.create_completion(request, _request())

    assert calls == [(capability, "other")]


@pytest.mark.asyncio
async def test_legacy_completion_engine_logprobs_rejection_emits_capability(
    monkeypatch,
):
    from fastapi import HTTPException

    from rapid_mlx.api.models import CompletionRequest
    from rapid_mlx.routes import completions
    from rapid_mlx.telemetry import inference

    engine = SimpleNamespace()
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(completions, "_validate_model_name", lambda _model: None)
    monkeypatch.setattr(completions, "get_engine", lambda _model: engine)

    async def ready(_engine):
        return None

    monkeypatch.setattr(completions, "ensure_engine_ready", ready)
    monkeypatch.setattr(completions, "_check_admission_or_503", lambda _engine: None)
    monkeypatch.setattr(
        completions, "enforce_context_length_for_prompt", lambda *_a, **_k: None
    )
    monkeypatch.setattr(completions, "_resolve_max_tokens", lambda *_a: 8)
    monkeypatch.setattr(
        completions, "_engine_supports_completion_logprobs", lambda _engine: False
    )
    monkeypatch.setattr(
        completions, "_release_admission_unless_committed", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )
    request = CompletionRequest.model_construct(
        model="ignored",
        prompt="hello",
        suffix=None,
        n=1,
        best_of=1,
        echo=False,
        logprobs=1,
        stream=False,
        max_tokens=8,
        temperature=0.0,
    )

    with pytest.raises(HTTPException, match="does not expose"):
        await completions.create_completion(request, _request())

    assert calls == [("logprobs_unsupported", "llm")]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing_model", "wrong_model"])
async def test_embedding_configuration_rejections_emit_capability(monkeypatch, case):
    from fastapi import HTTPException

    from rapid_mlx.api.models import EmbeddingRequest
    from rapid_mlx.routes import embeddings
    from rapid_mlx.telemetry import inference

    fake_embedding = types.ModuleType("rapid_mlx.embedding")
    fake_embedding.EMBEDDINGS_EXTRA_INSTALL_HINT = "install embeddings"
    fake_embedding.EmbeddingInputTooLongError = RuntimeError
    monkeypatch.setitem(sys.modules, "rapid_mlx.embedding", fake_embedding)
    cfg = SimpleNamespace(
        embedding_engine=object(),
        embedding_model_locked=None if case == "missing_model" else "resolved/model",
    )
    monkeypatch.setattr(embeddings, "get_config", lambda: cfg)
    if case == "wrong_model":
        monkeypatch.setattr(
            "rapid_mlx.service.helpers._resolve_request_alias_or_default",
            lambda *_a, **_k: None,
        )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )

    with pytest.raises(HTTPException):
        await embeddings.create_embeddings(
            EmbeddingRequest(model="wrong", input="hello"), _request()
        )

    expected_type = "other" if case == "missing_model" else "embedding"
    assert calls == [("embeddings_unavailable", expected_type)]


@pytest.mark.parametrize(
    ("helper", "args"),
    [
        ("_reject_non_whisper_for_translation", ("org/parakeet",)),
        ("_reject_word_timestamps_for_non_whisper", ("org/parakeet", ["word"])),
    ],
)
def test_audio_capability_helpers_emit_before_http_error(monkeypatch, helper, args):
    from fastapi import HTTPException

    from rapid_mlx.routes import audio
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )

    with pytest.raises(HTTPException):
        getattr(audio, helper)(*args)

    assert calls == [("speech_capability_unsupported", "audio")]


def test_audio_word_timestamp_guard_ignores_empty_model():
    """Keep the pre-resolution empty-model path free of false capability events."""
    from rapid_mlx.routes import audio

    assert audio._reject_word_timestamps_for_non_whisper("", ["word"]) is None


@pytest.mark.asyncio
async def test_audio_alignment_wrong_model_emits_before_http_error(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.routes import audio
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(audio, "_resolve_stt_model", lambda _model: "resolved/asr")
    monkeypatch.setattr(audio, "_is_aligner_model", lambda _model: False)
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )

    with pytest.raises(HTTPException):
        await audio._run_alignment_request(
            object(), "asr", "known transcript", None, "json"
        )

    assert calls == [("speech_capability_unsupported", "audio")]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["voice_seed", "clone", "qwen_base"])
async def test_audio_speech_capability_rejections_emit(monkeypatch, case):
    from fastapi import HTTPException

    from rapid_mlx.api.models import AudioSpeechRequest
    from rapid_mlx.audio import probe
    from rapid_mlx.routes import audio
    from rapid_mlx.telemetry import inference

    fake_tts = types.ModuleType("rapid_mlx.audio.tts")
    fake_tts.UnsupportedAudioFormatError = RuntimeError
    fake_tts.is_indextts_model = lambda _model: False
    fake_tts.is_kokoro_family_model = lambda _model: False
    fake_tts.is_qwen3_voicedesign_model = lambda _model: False
    monkeypatch.setitem(sys.modules, "rapid_mlx.audio.tts", fake_tts)
    monkeypatch.setattr(probe, "require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(probe, "require_kokoro_runtime", lambda: None)
    resolved = "org/Qwen3-TTS-0.6B-Base" if case == "qwen_base" else "org/kokoro"
    monkeypatch.setattr(audio, "_resolve_tts_model", lambda _model: resolved)
    monkeypatch.setattr(audio, "_is_clone_capable_model", lambda _model: False)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda value, *, model_type="other": calls.append((value, model_type)),
    )
    request = AudioSpeechRequest.model_construct(
        model="model",
        input="hello",
        voice="default",
        speed=1.0,
        response_format="wav",
        sample_rate=None,
        channels=None,
        instructions=None,
        voice_seed=1 if case == "voice_seed" else None,
        ref_audio="data:audio/wav;base64,AA==" if case == "clone" else None,
        ref_text="hello" if case == "clone" else None,
        exaggeration=None,
    )

    with pytest.raises(HTTPException):
        await audio.create_speech(request)

    assert calls == [("speech_capability_unsupported", "audio")]


@pytest.mark.asyncio
async def test_embedding_runtime_rejection_emits_capability(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx import server
    from rapid_mlx.api.models import EmbeddingRequest
    from rapid_mlx.routes import embeddings
    from rapid_mlx.telemetry import inference

    fake_embedding = types.ModuleType("rapid_mlx.embedding")
    fake_embedding.EMBEDDINGS_EXTRA_INSTALL_HINT = "install embeddings"
    fake_embedding.EmbeddingInputTooLongError = type(
        "EmbeddingInputTooLongError", (Exception,), {}
    )
    monkeypatch.setitem(sys.modules, "rapid_mlx.embedding", fake_embedding)
    cfg = SimpleNamespace(
        embedding_engine=object(), embedding_model_locked="embeddinggemma-300m-6bit"
    )
    monkeypatch.setattr(embeddings, "get_config", lambda: cfg)
    monkeypatch.setattr(
        server,
        "load_embedding_model",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError("missing extra")),
    )
    monkeypatch.setattr(
        "rapid_mlx.service.helpers._resolve_request_alias_or_default",
        lambda *_a, **_k: cfg.embedding_model_locked,
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda capability, *, model_type="other": calls.append(
            (capability, model_type)
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await embeddings.create_embeddings(
            EmbeddingRequest(model="default", input="hello"), _request()
        )

    assert exc_info.value.status_code == 503
    assert calls == [("runtime_extra_missing", "embedding")]


@pytest.mark.asyncio
async def test_embedding_success_emits_completed_request(monkeypatch):
    from rapid_mlx import server
    from rapid_mlx.api.models import EmbeddingRequest
    from rapid_mlx.routes import embeddings
    from rapid_mlx.telemetry import inference

    fake_embedding = types.ModuleType("rapid_mlx.embedding")
    fake_embedding.EMBEDDINGS_EXTRA_INSTALL_HINT = "install embeddings"
    fake_embedding.EmbeddingInputTooLongError = type(
        "EmbeddingInputTooLongError", (Exception,), {}
    )
    monkeypatch.setitem(sys.modules, "rapid_mlx.embedding", fake_embedding)
    engine = SimpleNamespace(
        model_name="private-local-embedding",
        effective_max_length=512,
        count_tokens=lambda _texts: 2,
        embed=lambda _texts: [[0.25, 0.75]],
    )
    cfg = SimpleNamespace(
        embedding_engine=engine,
        embedding_model_locked="embeddinggemma-300m-6bit",
    )
    monkeypatch.setattr(embeddings, "get_config", lambda: cfg)
    monkeypatch.setattr(server, "load_embedding_model", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "rapid_mlx.service.helpers._resolve_request_alias_or_default",
        lambda *_a, **_k: cfg.embedding_model_locked,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        inference, "emit_completed_request", lambda **kwargs: calls.append(kwargs)
    )

    response = await embeddings.create_embeddings(
        EmbeddingRequest(model="default", input="hello"), _request()
    )

    assert len(response.data) == 1
    assert calls == [
        {
            "model": "<custom>",
            "endpoint": "/v1/embeddings",
            "caller_agent": "openai-python/1.2",
            "caller_client": "rapid-cli-chat",
            "result": "ok",
        }
    ]


def test_video_engine_and_runtime_rejections_emit_capabilities(monkeypatch, tmp_path):
    import builtins

    from fastapi import HTTPException

    from rapid_mlx import config
    from rapid_mlx.routes import video
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda capability, *, model_type="other": calls.append(
            (capability, model_type)
        ),
    )
    monkeypatch.setattr(config, "get_config", lambda: SimpleNamespace(engine=None))
    with pytest.raises(HTTPException) as exc_info:
        video._video_engine()
    assert exc_info.value.status_code == 409

    real_import = builtins.__import__

    def no_pillow(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("Pillow absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pillow)
    with pytest.raises(HTTPException) as exc_info:
        video._validate_reference_image(tmp_path / "input.png")
    assert exc_info.value.status_code == 503
    assert calls == [
        ("video_generation_unavailable", "other"),
        ("runtime_extra_missing", "video-gen"),
    ]


@pytest.mark.asyncio
async def test_audio_runtime_rejections_emit_capabilities(monkeypatch):
    import builtins

    from fastapi import HTTPException

    from rapid_mlx.routes import audio
    from rapid_mlx.telemetry import inference

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda capability, *, model_type="other": calls.append(
            (capability, model_type)
        ),
    )
    monkeypatch.setattr(audio, "_resolve_stt_model", lambda _model: "whisper")

    async def drain(_file, _tmp):
        return None

    monkeypatch.setattr(audio, "_stream_upload_to_tempfile", drain)
    real_import = builtins.__import__

    def no_stt(name, *args, **kwargs):
        if name.endswith("audio.stt"):
            raise ImportError("mlx-audio absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_stt)
    with pytest.raises(HTTPException) as exc_info:
        await audio._run_stt_request(object(), "whisper", None, "json", "transcribe")
    assert exc_info.value.status_code == 503

    aligner = SimpleNamespace(model_name="aligner")
    monkeypatch.setattr(audio, "_resolve_stt_model", lambda _model: "aligner")
    monkeypatch.setattr(audio, "_is_aligner_model", lambda _model: True)
    monkeypatch.setattr(audio, "_aligner_engine", aligner)

    async def import_failure(*_args, **_kwargs):
        raise ImportError("mlx-audio absent")

    monkeypatch.setattr(audio, "run_to_completion", import_failure)
    with pytest.raises(HTTPException) as exc_info:
        await audio._run_alignment_request(
            object(), "aligner", "known transcript", None, "json"
        )
    assert exc_info.value.status_code == 503
    assert calls == [
        ("runtime_extra_missing", "audio"),
        ("runtime_extra_missing", "audio"),
    ]


@pytest.mark.asyncio
async def test_audio_transcription_success_emits_completed_request(monkeypatch):
    from rapid_mlx.audio import probe
    from rapid_mlx.routes import audio
    from rapid_mlx.telemetry import inference

    response = {"text": "transcribed"}

    async def fake_stt_request(**_kwargs):
        return response

    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    monkeypatch.setattr(
        audio, "_reject_word_timestamps_for_non_whisper", lambda *_a: None
    )
    monkeypatch.setattr(audio, "_run_stt_request", fake_stt_request)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        inference, "emit_completed_request", lambda **kwargs: calls.append(kwargs)
    )

    result = await audio.create_transcription(
        request=_request(),
        file=object(),
        model_form="whisper-large-v3",
        language_form=None,
        response_format_form="json",
        text_form=None,
        context_form=None,
        timestamp_granularities_bracket_form=None,
        timestamp_granularities_plain_form=None,
        model_query=None,
        language_query=None,
        response_format_query=None,
        text_query=None,
        timestamp_granularities_bracket_query=None,
        timestamp_granularities_plain_query=None,
    )

    assert result is response
    assert calls == [
        {
            "model": "<custom>",
            "endpoint": "/v1/audio/transcriptions",
            "caller_agent": "openai-python/1.2",
            "caller_client": "rapid-cli-chat",
            "result": "ok",
        }
    ]


@pytest.mark.asyncio
async def test_audio_alignment_success_emits_completed_request(monkeypatch):
    from rapid_mlx.audio import probe
    from rapid_mlx.routes import audio
    from rapid_mlx.telemetry import inference

    response = {"text": "aligned"}

    async def fake_alignment_request(**_kwargs):
        return response

    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    monkeypatch.setattr(
        audio, "_reject_word_timestamps_for_non_whisper", lambda *_a: None
    )
    monkeypatch.setattr(audio, "_run_alignment_request", fake_alignment_request)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        inference, "emit_completed_request", lambda **kwargs: calls.append(kwargs)
    )

    result = await audio.create_transcription(
        request=_request(),
        file=object(),
        model_form="qwen3-forced-aligner",
        language_form="en",
        response_format_form="json",
        text_form="known transcript",
        context_form=None,
        timestamp_granularities_bracket_form=None,
        timestamp_granularities_plain_form=None,
        model_query=None,
        language_query=None,
        response_format_query=None,
        text_query=None,
        timestamp_granularities_bracket_query=None,
        timestamp_granularities_plain_query=None,
    )

    assert result is response
    assert len(calls) == 1
    assert calls[0]["endpoint"] == "/v1/audio/transcriptions"
    assert calls[0]["result"] == "ok"


def test_completed_inference_key_stays_within_store_limit(monkeypatch):
    from rapid_mlx.telemetry import inference

    keys: list[str] = []
    monkeypatch.setattr(inference.store, "record", lambda key: keys.append(key))
    inference._record_completed_request(
        model="x" * 128,
        endpoint="/v1/chat/completions",
        caller_agent="openai-python/1.0",
        caller_client=None,
        result="ok",
    )
    assert len(keys) == 1
    assert len(keys[0]) <= inference.store.MAX_KEY_LENGTH


def test_twelve_real_requests_emit_only_bucket_crossings(monkeypatch):
    from rapid_mlx.telemetry import inference

    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        inference.track_module,
        "track",
        lambda event, props: events.append((event, dict(props))),
    )
    monkeypatch.setattr(inference.track_module, "emit_active_day", lambda: None)

    for _ in range(12):
        inference._record_completed_request(
            model="<custom>",
            endpoint="/v1/chat/completions",
            caller_agent="cursor/1.0",
            caller_client=None,
            result="ok",
        )

    assert [props["count_bucket"] for _, props in events] == [
        "1",
        "2",
        "3_4",
        "5_9",
        "10_19",
    ]
    assert all(name == "inference_bucket_reached" for name, _ in events)


def test_preexisting_counter_preserves_observed_existing_source(monkeypatch):
    from rapid_mlx.telemetry import inference

    key = "inf|<custom>|/v1/chat/completions|cursor|ok"
    inference.store.record(key)
    with sqlite3.connect(inference.store.db_path()) as connection:
        connection.execute(
            "UPDATE counters SET count = 3, last_bucket = NULL WHERE key = ?", (key,)
        )
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        inference.track_module,
        "track",
        lambda _event, props: events.append(dict(props)),
    )
    monkeypatch.setattr(inference.track_module, "emit_active_day", lambda: None)

    inference._record_completed_request(
        model="<custom>",
        endpoint="/v1/chat/completions",
        caller_agent="cursor/1.0",
        caller_client=None,
        result="ok",
    )

    assert events[0]["bucket_source"] == "observed_existing"


def test_hostile_model_and_result_are_clamped_at_emitter(monkeypatch):
    from rapid_mlx.telemetry import inference

    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        inference.store,
        "record",
        lambda key: SimpleNamespace(bucket="1", bucket_source="crossed_now", key=key),
    )
    monkeypatch.setattr(
        inference.track_module,
        "track",
        lambda _event, props: events.append(dict(props)),
    )

    inference._record_completed_request(
        model="private-secret-model?token=x",
        endpoint="/v1/chat/completions",
        caller_agent=None,
        caller_client=None,
        result="invented",
    )

    assert events[0]["model"] == "<custom>"
    assert events[0]["result"] == "failed"
    assert "secret-model" not in repr(events)


def test_worst_case_counter_cardinality_supports_28_complete_models():
    from rapid_mlx.telemetry import registry, store

    enums = registry.load_registry()["enums"]
    keys_per_model = (
        len(enums["endpoint"]["values"])
        * len(enums["caller"]["values"])
        * len(enums["result"]["values"])
    )
    assert keys_per_model == 8 * 26 * 2
    assert store.MAX_KEYS // keys_per_model == 28


@pytest.mark.parametrize(
    ("relative_path", "endpoint"),
    [
        ("rapid_mlx/routes/responses.py", "/v1/responses"),
        ("rapid_mlx/routes/embeddings.py", "/v1/embeddings"),
        ("rapid_mlx/routes/audio.py", "/v1/audio/transcriptions"),
        ("rapid_mlx/routes/images.py", "/v1/images/generations"),
    ],
)
def test_additional_endpoint_has_completed_request_emit(relative_path, endpoint):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "emit_completed_request(" in source
    assert f'endpoint="{endpoint}"' in source
    assert 'result="ok"' in source


@pytest.mark.parametrize(
    ("relative_path", "failed_count"),
    [
        ("rapid_mlx/routes/chat.py", 4),
        ("rapid_mlx/routes/completions.py", 1),
        ("rapid_mlx/routes/anthropic.py", 1),
    ],
)
def test_each_terminal_site_uses_only_v2_emit(relative_path, failed_count):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    v2 = "_telemetry_inference.emit_completed_request("
    assert "_telemetry_emit.request(" not in source
    assert source.count(v2) >= 2
    assert source.count('result="ok"') >= 2
    assert source.count('result="failed"') == failed_count
    assert source.count("emit_failed_on_stream_error(") == 1
    for call in source.split(v2)[1:]:
        assert "request.model" not in call[: call.index(")") + 1]


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(self.rfile.read(length))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.mark.parametrize(
    ("endpoint", "client_header", "expected_caller"),
    [
        ("/v1/chat/completions", "rapid-desktop", "rapid-desktop"),
        ("/v1/embeddings", "rapid-desktop", "rapid-desktop"),
        ("/v1/images/generations", "rapid-desktop", "rapid-desktop"),
        ("/v1/audio/transcriptions", "rapid-desktop", "rapid-desktop"),
        ("/v1/chat/completions", "hostile-private-client", "other"),
    ],
)
def test_inference_and_capability_events_reach_loopback_as_exact_json(
    monkeypatch, tmp_path, endpoint, client_header, expected_caller
):
    import rapid_mlx
    from rapid_mlx.telemetry import (
        consent_runtime,
        inference,
        posthog_sender,
        state,
    )
    from rapid_mlx.telemetry import track as track_module
    from rapid_mlx.telemetry.build_gate import ReleaseStamp
    from rapid_mlx.telemetry.common_props import PlatformFacts

    stamp = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
    install_id = "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
    session_id = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
    facts = PlatformFacts(
        os="darwin",
        os_version="25.3",
        arch="arm64",
        chip="m3-ultra",
        memory_gb=64,
        python_version="3.11",
    )
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: stamp)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(track_module.common_props, "read_platform_facts", lambda: facts)
    monkeypatch.setattr(state, "get_or_create_client_id", lambda: install_id)
    monkeypatch.setattr(state, "session_id", lambda: session_id)
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: "7-29"
    )
    crossing = SimpleNamespace(bucket="1", bucket_source="crossed_now")
    monkeypatch.setattr(inference.store, "record", lambda _key: crossing)
    track_module._reset_for_tests()
    posthog_sender._reset_for_tests()

    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.bodies = []  # type: ignore[attr-defined]
    loopback_url = f"http://127.0.0.1:{server.server_port}/batch/"
    monkeypatch.setenv(posthog_sender.POSTHOG_URL_ENV, loopback_url)

    def post(url: str, body: bytes, timeout: float) -> int:
        assert url == loopback_url
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)

    sender = posthog_sender.PostHogSender(
        post=post,
        gate=lambda: stamp,
        allowed=lambda: True,
    )
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        caller_agent, caller_client = inference.request_caller_headers(
            _request("hostile-private-agent/1.0", client_header)
        )
        inference._record_completed_request(
            model="neohorse-9b-4bit",
            endpoint=endpoint,
            caller_agent=caller_agent,
            caller_client=caller_client,
            result="ok",
        )
        inference.emit_capability_rejected("logprobs_unsupported", model_type="llm")
        sender.flush(2.0)
    finally:
        sender.close(0.5)
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()

    items = [
        item
        for body in server.bodies  # type: ignore[attr-defined]
        for item in json.loads(body)["batch"]
    ]
    assert [item["event"] for item in items] == [
        "inference_bucket_reached",
        "active_day",
        "capability_rejected",
    ]
    expected_common: Mapping[str, object] = {
        "$geoip_disable": True,
        "$process_person_profile": False,
        "surface": "cli",
        "app_version": "0.15.1",
        "os": "darwin",
        "os_version": "25.3",
        "arch": "arm64",
        "chip": "m3-ultra",
        "memory_gb": 64,
        "python_version": "3.11",
        "install_id": install_id,
        "session_id": session_id,
        "channel": "stable",
        "days_since_first_run_bucket": "7-29",
    }
    assert items[0]["distinct_id"] == install_id
    assert items[0]["properties"] == {
        **expected_common,
        "model": "neohorse-9b-4bit",
        "endpoint": endpoint,
        "caller": expected_caller,
        "result": "ok",
        "count_bucket": "1",
        "bucket_source": "crossed_now",
    }
    assert items[1]["distinct_id"] == install_id
    assert items[1]["properties"] == expected_common
    assert items[2]["distinct_id"] == install_id
    assert items[2]["properties"] == {
        **expected_common,
        "capability": "logprobs_unsupported",
        "model_type": "llm",
    }
    assert "hostname" not in repr(items)
    assert "username" not in repr(items)
    assert str(tmp_path) not in repr(items)
    assert "127.0.0.1" not in repr(items)
