# SPDX-License-Identifier: Apache-2.0
"""Telemetry-v2 inference and capability emitter contracts."""

from __future__ import annotations

import ast
import asyncio
import json
import multiprocessing
import os
import re
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
        capability="mcp_unsupported",
        model_type="other",
        model="mlx-community/private-user/model",
        caller_agent="private-agent/99 cursor/1.0",
        caller_client=None,
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
            {
                "capability": "mcp_unsupported",
                "model_type": "other",
                "model": "<local>",
                "caller": "cursor",
            },
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


_REQUEST_ROOT_NAME = re.compile(r"^(request|req|body|payload|.*_request|.*_body)$")
_REQUEST_TYPE_SUFFIXES = ("Request", "Body", "Params")
#: FastAPI markers whose parameter VALUE is client-controlled request input.
_REQUEST_PARAM_MARKERS = frozenset(
    {"Form", "Query", "Body", "Header", "Cookie", "File", "Path", "UploadFile"}
)
_ROUTE_DECORATOR_METHODS = frozenset(
    {"get", "post", "put", "patch", "delete", "api_route", "websocket"}
)
_REQUEST_MODEL_DEBUG_HANDLERS = frozenset(
    {
        "rapid_mlx/routes/anthropic.py:_stream_anthropic_messages",
        "rapid_mlx/routes/completions.py:create_completion",
        "rapid_mlx/routes/responses.py:_non_stream",
    }
)


class _FunctionScopeNodes(ast.NodeVisitor):
    """Collect one function's nodes without leaking into nested scopes."""

    def __init__(self):
        self.nodes: list[ast.AST] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.nodes.append(node)
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.nodes.append(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.nodes.append(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.nodes.append(node)


def _scope_nodes(body: list[ast.AST]) -> list[ast.AST]:
    collector = _FunctionScopeNodes()
    for statement in body:
        collector.visit(statement)
    return collector.nodes


class _NestedScopes(ast.NodeVisitor):
    """Collect directly nested callable and comprehension scopes."""

    def __init__(self):
        self.nodes: list[ast.AST] = []

    def _add(self, node: ast.AST) -> None:
        self.nodes.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._add(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._add(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._add(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._add(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._add(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # A class body is not part of its enclosing function scope, but methods
        # and other callable scopes nested in it still need their own analysis.
        super().generic_visit(node)


def _nested_scopes(body: list[ast.AST]) -> list[ast.AST]:
    collector = _NestedScopes()
    for node in body:
        collector.visit(node)
    return collector.nodes


def _target_names(target: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _module_name(repo_root: Path, path: Path) -> str:
    return ".".join(path.relative_to(repo_root).with_suffix("").parts)


def _imported_names(module_name: str, tree: ast.Module) -> dict[str, str]:
    imported: dict[str, str] = {}
    package = module_name.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parent = package[: len(package) - max(node.level - 1, 0)]
            source = ".".join([*parent, node.module]) if node.level else node.module
            for alias in node.names:
                imported[alias.asname or alias.name] = f"{source}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = alias.name
    return imported


def _pydantic_model_classes(repo_root: Path) -> set[str]:
    """Resolve BaseModel subclasses without importing MLX-bearing route modules."""
    class_bases: dict[str, set[str]] = {}
    for package in ("api", "routes", "schemas"):
        directory = repo_root / "rapid_mlx" / package
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_name = _module_name(repo_root, path)
            imported = _imported_names(module_name, tree)
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                bases: set[str] = set()
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.add(imported.get(base.id, f"{module_name}.{base.id}"))
                    elif isinstance(base, ast.Attribute):
                        root = base.value
                        while isinstance(root, ast.Attribute):
                            root = root.value
                        if isinstance(root, ast.Name):
                            bases.add(f"{imported.get(root.id, root.id)}.{base.attr}")
                class_bases[f"{module_name}.{node.name}"] = bases

    models = {"pydantic.BaseModel"}
    changed = True
    while changed:
        changed = False
        for class_name, bases in class_bases.items():
            if class_name not in models and models.intersection(bases):
                models.add(class_name)
                changed = True
    return models


def _annotation_names(annotation: ast.AST | None) -> set[str]:
    if annotation is None:
        return set()
    names: set[str] = set()
    pending = [annotation]
    parsed_strings: set[str] = set()
    while pending:
        node = pending.pop()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                names.add(child.attr)
            elif (
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value not in parsed_strings
            ):
                parsed_strings.add(child.value)
                try:
                    parsed = ast.parse(child.value, mode="eval").body
                except SyntaxError:
                    names.update(re.findall(r"[A-Za-z_]\w*", child.value))
                else:
                    if isinstance(parsed, ast.Constant) and isinstance(
                        parsed.value, str
                    ):
                        names.update(re.findall(r"[A-Za-z_]\w*", parsed.value))
                    else:
                        pending.append(parsed)
    return names


def _assigned_request_value(node: ast.AST) -> bool:
    while isinstance(node, ast.Await):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id == "Request"
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id.endswith("Request")
    return isinstance(node.func, ast.Attribute) and (
        node.func.attr in {"json", "parse_obj", "model_validate"}
        or node.func.attr.endswith("Request")
    )


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_route_handler(
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> bool:
    return not isinstance(function, ast.Lambda) and any(
        _call_name(decorator) in _ROUTE_DECORATOR_METHODS
        for decorator in function.decorator_list
    )


def _parameter_defaults(
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> dict[str, ast.AST]:
    positional = [*function.args.posonlyargs, *function.args.args]
    defaults = dict(
        zip(
            (
                argument.arg
                for argument in positional[
                    len(positional) - len(function.args.defaults) :
                ]
            ),
            function.args.defaults,
            strict=True,
        )
    )
    defaults.update(
        (argument.arg, default)
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults, strict=True
        )
        if default is not None
    )
    return defaults


def _function_request_roots(
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    nodes: list[ast.AST],
    imported: dict[str, str],
    pydantic_models: set[str],
) -> set[str]:
    """Names whose value (or ``.model``) is client-controlled in ``function``.

    Every parameter of a route handler is request input as a VALUE — a bare
    multipart ``model: str = Form(...)`` is exactly as attacker-controlled as
    ``request.model`` — except ``Depends(...)`` injections. Outside route
    handlers, a ``Form``/``Query``/``Body``/``Header``/``File`` default or
    annotation marks the same thing.
    """
    roots: set[str] = set()
    route_handler = _is_route_handler(function)
    defaults = _parameter_defaults(function)
    arguments = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
        *(
            argument
            for argument in (function.args.vararg, function.args.kwarg)
            if argument is not None
        ),
    ]
    for argument in arguments:
        annotation_names = _annotation_names(argument.annotation)
        typed_request = any(
            name.endswith(_REQUEST_TYPE_SUFFIXES) for name in annotation_names
        ) or any(imported.get(name) in pydantic_models for name in annotation_names)
        default_marker = _call_name(defaults.get(argument.arg, ast.Constant(None)))
        request_param = default_marker in _REQUEST_PARAM_MARKERS or bool(
            annotation_names & _REQUEST_PARAM_MARKERS
        )
        injected = default_marker == "Depends" or "Depends" in annotation_names
        if (
            _REQUEST_ROOT_NAME.fullmatch(argument.arg)
            or typed_request
            or request_param
            or (route_handler and not injected)
        ):
            roots.add(argument.arg)

    for node in nodes:
        assignments: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(node, ast.Assign):
            assignments.extend((target, node.value) for target in node.targets)
        elif (isinstance(node, ast.AnnAssign) and node.value is not None) or isinstance(
            node, ast.NamedExpr
        ):
            assignments.append((node.target, node.value))
        for target, value in assignments:
            names = _target_names(target)
            roots.update(name for name in names if _REQUEST_ROOT_NAME.fullmatch(name))
            if _assigned_request_value(value):
                roots.update(names)
    return roots


def _function_bound_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    nodes: list[ast.AST],
) -> set[str]:
    arguments = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
        *(
            argument
            for argument in (function.args.vararg, function.args.kwarg)
            if argument is not None
        ),
    ]
    return (
        {argument.arg for argument in arguments}
        | {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        | (
            {
                statement.name
                for statement in function.body
                if isinstance(
                    statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            }
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            else set()
        )
    )


def _scope_lookup_bindings(
    module_name: str,
    nodes: list[ast.AST],
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | None,
) -> dict[str, list[tuple[tuple[int, int], str | None]]]:
    """Collect ordered name bindings without entering nested scopes."""
    bindings: dict[str, list[tuple[tuple[int, int], str | None]]] = {}

    def bind(node: ast.AST, name: str, value: str | None) -> None:
        bindings.setdefault(name, []).append(((node.lineno, node.col_offset), value))

    if function is not None:
        arguments = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
            *(
                argument
                for argument in (function.args.vararg, function.args.kwarg)
                if argument is not None
            ),
        ]
        for argument in arguments:
            bind(argument, argument.arg, None)

    package = module_name.split(".")[:-1]
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module:
            parent = package[: len(package) - max(node.level - 1, 0)]
            source = ".".join([*parent, node.module]) if node.level else node.module
            for alias in node.names:
                bind(node, alias.asname or alias.name, f"{source}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bind(
                    node,
                    alias.asname or alias.name.split(".")[0],
                    alias.name,
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            value = f"{module_name}.{node.name}" if function is None else None
            bind(node, node.name, value)
        elif isinstance(node, ast.ClassDef) or (
            isinstance(node, ast.ExceptHandler) and node.name
        ):
            bind(node, node.name, None)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bind(node, node.id, None)

    for events in bindings.values():
        events.sort(key=lambda event: event[0])
    return bindings


_ENGINE_LOOKUP_FUNCTIONS = {
    "rapid_mlx.routes.images._image_engine",
    "rapid_mlx.service.helpers.get_engine",
}


def _request_model_violations(
    repo_root: Path, root_debug: dict[str, list[str]] | None = None
) -> list[str]:
    """Find request-derived model identities reaching route telemetry sinks.

    Request model names stop being attacker-controlled telemetry identity only
    at the exact engine lookup boundaries used by the routes. Imported
    ``rapid_mlx.service.helpers.get_engine`` and the module-local
    ``rapid_mlx.routes.images._image_engine`` return a loaded engine or raise;
    neither can echo its string argument. Calls are matched by qualified symbol
    identity, so a same-named local function is not a privacy boundary. All
    other calls, including telemetry model-id helpers, propagate taint.

    Taint also crosses module-local calls: an argument that is tainted at a
    call to a module-level function makes the matching parameter a request
    root inside that function (iterated to a fixpoint), so a route handler
    handing its ``model`` form field to a helper that emits telemetry is
    caught at the helper's sink. The two engine-lookup boundaries above are
    the exception: they are the reviewed place where a request name becomes a
    resident engine, so their parameters are not re-tainted.
    """
    violations: list[str] = []
    pydantic_models = _pydantic_model_classes(repo_root)

    for directory in (
        repo_root / "rapid_mlx/routes",
        repo_root / "rapid_mlx/api",
    ):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_name = _module_name(repo_root, path)
            imported = _imported_names(module_name, tree)
            module_functions = {
                f"{module_name}.{node.name}": node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            param_taint: dict[str, set[str]] = {}
            telemetry_names: set[str] = set()
            telemetry_modules: set[str] = set()
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and (
                        node.module.startswith("rapid_mlx.telemetry")
                        or node.module.startswith("telemetry")
                    )
                ):
                    for alias in node.names:
                        imported_name = alias.asname or alias.name
                        if alias.name == "track" or alias.name.startswith("emit_"):
                            telemetry_names.add(imported_name)
                        telemetry_modules.add(imported_name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("rapid_mlx.telemetry"):
                            telemetry_modules.add(
                                alias.asname or alias.name.split(".")[0]
                            )

            def is_telemetry_call(
                node: ast.Call,
                telemetry_names: set[str] = telemetry_names,
                telemetry_modules: set[str] = telemetry_modules,
            ) -> bool:
                if isinstance(node.func, ast.Name):
                    return node.func.id in telemetry_names
                if not isinstance(node.func, ast.Attribute):
                    return False
                root = node.func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                return (
                    isinstance(root, ast.Name)
                    and root.id in telemetry_modules
                    and (
                        node.func.attr == "track" or node.func.attr.startswith("emit_")
                    )
                )

            def sink_expressions(node: ast.Call) -> list[ast.AST]:
                expressions = [
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "telemetry_model"
                ]
                if is_telemetry_call(node):
                    function_name = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else node.func.attr
                    )
                    positional = node.args
                    if function_name == "emit_failed_on_stream_error":
                        positional = positional[1:]
                    expressions.extend(positional)
                    expressions.extend(
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg in {"model", "model_id"}
                    )
                return expressions

            def expression_is_tainted(
                node: ast.AST,
                tainted: set[str],
                request_roots: set[str],
                lookup_scopes: list[
                    dict[str, list[tuple[tuple[int, int], str | None]]]
                ],
            ) -> bool:
                def lookup_binding(name: str) -> str | None:
                    position = (node.lineno, node.col_offset)
                    resolved = None
                    for scope in lookup_scopes:
                        for binding_position, value in scope.get(name, ()):
                            if binding_position <= position:
                                resolved = value
                    return resolved

                if isinstance(node, ast.Call):
                    resolved_name = None
                    if isinstance(node.func, ast.Name):
                        resolved_name = lookup_binding(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        root = node.func.value
                        attributes = [node.func.attr]
                        while isinstance(root, ast.Attribute):
                            attributes.append(root.attr)
                            root = root.value
                        if (
                            isinstance(root, ast.Name)
                            and (root_binding := lookup_binding(root.id)) is not None
                        ):
                            resolved_name = ".".join(
                                [root_binding, *reversed(attributes)]
                            )
                    if resolved_name in _ENGINE_LOOKUP_FUNCTIONS:
                        return False
                if isinstance(node, ast.Name) and node.id in request_roots:
                    return True
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "model"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in request_roots
                ):
                    return True
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in request_roots
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "model"
                ):
                    return True
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in request_roots
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "model"
                ):
                    return True
                if isinstance(node, ast.Name) and node.id in tainted:
                    return True
                return any(
                    expression_is_tainted(child, tainted, request_roots, lookup_scopes)
                    for child in ast.iter_child_nodes(node)
                )

            def expression_contains_request_root(
                node: ast.AST,
                request_roots: set[str],
                lookup_scopes: list[
                    dict[str, list[tuple[tuple[int, int], str | None]]]
                ],
            ) -> bool:
                if isinstance(node, ast.Call):
                    resolved_name = None
                    if isinstance(node.func, ast.Name):
                        position = (node.lineno, node.col_offset)
                        for scope in lookup_scopes:
                            for binding_position, value in scope.get(node.func.id, ()):
                                if binding_position <= position:
                                    resolved_name = value
                    elif isinstance(node.func, ast.Attribute):
                        root = node.func.value
                        attributes = [node.func.attr]
                        while isinstance(root, ast.Attribute):
                            attributes.append(root.attr)
                            root = root.value
                        if isinstance(root, ast.Name):
                            position = (node.lineno, node.col_offset)
                            root_binding = None
                            for scope in lookup_scopes:
                                for binding_position, value in scope.get(root.id, ()):
                                    if binding_position <= position:
                                        root_binding = value
                            if root_binding is not None:
                                resolved_name = ".".join(
                                    [root_binding, *reversed(attributes)]
                                )
                    if resolved_name in _ENGINE_LOOKUP_FUNCTIONS:
                        return False
                if isinstance(node, ast.Name) and node.id in request_roots:
                    return True
                return any(
                    expression_contains_request_root(
                        child, request_roots, lookup_scopes
                    )
                    for child in ast.iter_child_nodes(node)
                )

            def add_assignment_taint(
                target: ast.AST,
                value: ast.AST,
                tainted: set[str],
                request_roots: set[str],
                lookup_scopes: list[
                    dict[str, list[tuple[tuple[int, int], str | None]]]
                ],
            ) -> bool:
                changed = False
                if (
                    isinstance(target, (ast.Tuple, ast.List))
                    and isinstance(value, (ast.Tuple, ast.List))
                    and len(target.elts) == len(value.elts)
                ):
                    for child_target, child_value in zip(
                        target.elts, value.elts, strict=True
                    ):
                        changed |= add_assignment_taint(
                            child_target,
                            child_value,
                            tainted,
                            request_roots,
                            lookup_scopes,
                        )
                    return changed
                if expression_is_tainted(value, tainted, request_roots, lookup_scopes):
                    before = len(tainted)
                    tainted.update(_target_names(target))
                    changed = len(tainted) != before
                if expression_contains_request_root(
                    value, request_roots, lookup_scopes
                ):
                    before = len(request_roots)
                    request_roots.update(_target_names(target))
                    changed |= len(request_roots) != before
                return changed

            def scope_taint(
                nodes: list[ast.AST],
                request_roots: set[str],
                inherited_taint: set[str],
                lookup_scopes: list[
                    dict[str, list[tuple[tuple[int, int], str | None]]]
                ],
            ) -> set[str]:
                tainted = set(inherited_taint)
                changed = True
                while changed:
                    changed = False
                    for node in nodes:
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                changed |= add_assignment_taint(
                                    target,
                                    node.value,
                                    tainted,
                                    request_roots,
                                    lookup_scopes,
                                )
                        elif (
                            isinstance(node, ast.AnnAssign) and node.value is not None
                        ) or isinstance(node, (ast.AugAssign, ast.NamedExpr)):
                            changed |= add_assignment_taint(
                                node.target,
                                node.value,
                                tainted,
                                request_roots,
                                lookup_scopes,
                            )
                        elif isinstance(
                            node, (ast.For, ast.AsyncFor, ast.comprehension)
                        ):
                            changed |= add_assignment_taint(
                                node.target,
                                node.iter,
                                tainted,
                                request_roots,
                                lookup_scopes,
                            )
                        elif isinstance(node, (ast.With, ast.AsyncWith)):
                            for item in node.items:
                                if item.optional_vars is not None:
                                    changed |= add_assignment_taint(
                                        item.optional_vars,
                                        item.context_expr,
                                        tainted,
                                        request_roots,
                                        lookup_scopes,
                                    )
                return tainted

            def check_calls(
                nodes: list[ast.AST],
                tainted: set[str],
                request_roots: set[str],
                lookup_scopes: list[
                    dict[str, list[tuple[tuple[int, int], str | None]]]
                ],
                path: Path = path,
                module_functions: dict[
                    str, ast.FunctionDef | ast.AsyncFunctionDef
                ] = module_functions,
                param_taint: dict[str, set[str]] = param_taint,
            ) -> None:
                for node in nodes:
                    if not isinstance(node, ast.Call):
                        continue
                    callee_name = None
                    if isinstance(node.func, ast.Name):
                        position = (node.lineno, node.col_offset)
                        for scope in lookup_scopes:
                            for binding_position, value in scope.get(node.func.id, ()):
                                if binding_position <= position:
                                    callee_name = value
                    # The engine-lookup boundaries are reviewed not to echo
                    # their argument; their callers already stop taint there.
                    callee = (
                        None
                        if callee_name in _ENGINE_LOOKUP_FUNCTIONS
                        else module_functions.get(callee_name or "")
                    )
                    if callee is not None:
                        positional = [*callee.args.posonlyargs, *callee.args.args]
                        passed = [
                            *zip(positional, node.args, strict=False),
                            *(
                                (argument, keyword.value)
                                for keyword in node.keywords
                                for argument in (
                                    *positional,
                                    *callee.args.kwonlyargs,
                                )
                                if argument.arg == keyword.arg
                            ),
                        ]
                        param_taint.setdefault(callee_name or "", set()).update(
                            argument.arg
                            for argument, value in passed
                            if expression_is_tainted(
                                value, tainted, request_roots, lookup_scopes
                            )
                        )
                    if any(
                        expression_is_tainted(
                            expression, tainted, request_roots, lookup_scopes
                        )
                        for expression in sink_expressions(node)
                    ):
                        violations.append(
                            f"{path.relative_to(repo_root)}:{node.lineno}"
                        )

            def analyze_scope(
                body: list[ast.AST],
                inherited_roots: set[str],
                inherited_taint: set[str],
                function: ast.FunctionDef
                | ast.AsyncFunctionDef
                | ast.Lambda
                | None = None,
                inherited_lookup_scopes: list[
                    dict[str, list[tuple[tuple[int, int], str | None]]]
                ]
                | None = None,
                imported: dict[str, str] = imported,
                module_name: str = module_name,
                path: Path = path,
                pydantic_models: set[str] = pydantic_models,
                module_functions: dict[
                    str, ast.FunctionDef | ast.AsyncFunctionDef
                ] = module_functions,
                param_taint: dict[str, set[str]] = param_taint,
            ) -> None:
                nodes = _scope_nodes(body)
                request_roots = set(inherited_roots)
                inherited_scope_taint = set(inherited_taint)
                lookup_scopes = [*(inherited_lookup_scopes or [])]
                lookup_scopes.append(
                    _scope_lookup_bindings(module_name, nodes, function)
                )
                if function is not None:
                    bound_names = _function_bound_names(function, nodes)
                    request_roots.difference_update(bound_names)
                    inherited_scope_taint.difference_update(bound_names)
                    positional = [
                        *function.args.posonlyargs,
                        *function.args.args,
                    ]
                    defaults = [
                        *zip(
                            positional[len(positional) - len(function.args.defaults) :],
                            function.args.defaults,
                            strict=True,
                        ),
                        *(
                            (argument, default)
                            for argument, default in zip(
                                function.args.kwonlyargs,
                                function.args.kw_defaults,
                                strict=True,
                            )
                            if default is not None
                        ),
                    ]
                    request_roots.update(
                        argument.arg
                        for argument, default in defaults
                        if expression_contains_request_root(
                            default, inherited_roots, lookup_scopes
                        )
                    )
                request_roots.update(
                    child.id
                    for child in nodes
                    if isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and _REQUEST_ROOT_NAME.fullmatch(child.id)
                )
                if function is not None:
                    request_roots.update(
                        _function_request_roots(
                            function, nodes, imported, pydantic_models
                        )
                    )
                    for qualified, candidate in module_functions.items():
                        if candidate is function:
                            request_roots.update(param_taint.get(qualified, set()))
                    debug_key = (
                        f"{path.relative_to(repo_root)}:{function.name}"
                        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else None
                    )
                    if (
                        debug_key is not None
                        and root_debug is not None
                        and debug_key in _REQUEST_MODEL_DEBUG_HANDLERS
                    ):
                        root_debug[debug_key] = sorted(request_roots)
                tainted = scope_taint(
                    nodes, request_roots, inherited_scope_taint, lookup_scopes
                )
                check_calls(nodes, tainted, request_roots, lookup_scopes)

                for nested in _nested_scopes(body):
                    if isinstance(nested, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        analyze_scope(
                            nested.body,
                            request_roots,
                            tainted,
                            function=nested,
                            inherited_lookup_scopes=lookup_scopes,
                        )
                    elif isinstance(nested, ast.Lambda):
                        analyze_scope(
                            [nested.body],
                            request_roots,
                            tainted,
                            function=nested,
                            inherited_lookup_scopes=lookup_scopes,
                        )
                    elif isinstance(
                        nested, (ast.ListComp, ast.SetComp, ast.GeneratorExp)
                    ):
                        analyze_scope(
                            [nested.elt, *nested.generators],
                            request_roots,
                            tainted,
                            inherited_lookup_scopes=lookup_scopes,
                        )
                    elif isinstance(nested, ast.DictComp):
                        analyze_scope(
                            [nested.key, nested.value, *nested.generators],
                            request_roots,
                            tainted,
                            inherited_lookup_scopes=lookup_scopes,
                        )

            module_start = len(violations)
            while True:
                before = {key: set(params) for key, params in param_taint.items()}
                del violations[module_start:]
                analyze_scope(tree.body, set(), set())
                if param_taint == before:
                    break

    return violations


def test_no_route_or_api_telemetry_call_receives_request_model_expression():
    """Client model fields are routing input, never telemetry identity."""
    root_debug: dict[str, list[str]] = {}
    violations = _request_model_violations(REPO_ROOT, root_debug)
    assert violations == [], f"derived request roots: {root_debug}"


@pytest.mark.parametrize(
    "function_body",
    [
        "emit_capability_rejected('unsupported', model=request.model)",
        (
            "emit_capability_rejected('unsupported', "
            "model=telemetry_model_id(responses_request.model))"
        ),
        "m = request.model\nemit_capability_rejected('unsupported', model=m)",
        (
            "m, ignored = body.model, None\n"
            "emit_completed_request(model=m, endpoint='/v1/test')"
        ),
        (
            'm = f"model={payload.model}"\n'
            "emit_capability_rejected('unsupported', model_id=m)"
        ),
        "emit_capability_rejected('unsupported', model=engine_telemetry_id(request.model))",
        "emit_capability_rejected('unsupported', model=telemetry_model_id(request.model))",
        (
            "def engine_telemetry_id(x):\n"
            "    return x\n"
            "emit_capability_rejected('unsupported', model=engine_telemetry_id(request.model))"
        ),
        "emit_capability_rejected('unsupported', model=str(request.model))",
        (
            "def get_engine(x):\n"
            "    return x\n"
            "engine = get_engine(request.model)\n"
            "emit_capability_rejected('unsupported', model=engine_telemetry_id(engine))"
        ),
        "alias = request\nemit_capability_rejected('unsupported', model=alias.model)",
        "alias = request\nemit_capability_rejected('unsupported', model=alias['model'])",
        (
            "alias = request\n"
            "emit_capability_rejected('unsupported', model=getattr(alias, 'model'))"
        ),
        "alias, _ = request, 1\nemit_capability_rejected('unsupported', model=alias.model)",
        (
            "def inner(rq=request):\n"
            "    emit_capability_rejected('unsupported', model=rq.model)"
        ),
        (
            "alias = wrap(request)\n"
            "emit_capability_rejected('unsupported', model=alias.model)"
        ),
        (
            "if (alias := request):\n"
            "    emit_capability_rejected('unsupported', model=alias.model)"
        ),
        (
            "with request as alias:\n"
            "    emit_capability_rejected('unsupported', model=alias.model)"
        ),
        (
            "for alias in (request,):\n"
            "    emit_capability_rejected('unsupported', model=alias.model)"
        ),
    ],
    ids=[
        "direct",
        "wrapped",
        "local-alias",
        "tuple-unpack",
        "f-string",
        "engine-helper",
        "telemetry-model-helper",
        "shadowed-engine-helper",
        "string-call",
        "shadowed-engine-lookup",
        "bare-root-alias",
        "bare-root-mapping-alias",
        "bare-root-getattr-alias",
        "bare-root-unpack",
        "bare-root-default-argument",
        "bare-root-wrapper-call",
        "bare-root-walrus",
        "bare-root-with-as",
        "bare-root-for-target",
    ],
)
def test_request_model_privacy_gate_rejects_scratch_variants(tmp_path, function_body):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from rapid_mlx.service.helpers import get_engine\n"
        "from rapid_mlx.telemetry.inference import (\n"
        "    emit_capability_rejected, emit_completed_request, telemetry_model_id,\n"
        ")\n\n"
        "def scratch(request, responses_request, body, payload):\n"
        + "\n".join(f"    {line}" for line in function_body.splitlines())
        + "\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    expected_line = 6 + len(function_body.splitlines())
    assert _request_model_violations(tmp_path) == [
        f"rapid_mlx/routes/scratch.py:{expected_line}"
    ]


def test_request_model_privacy_gate_accepts_imported_engine_lookup(tmp_path):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from rapid_mlx.service.helpers import get_engine as lookup_engine\n"
        "from rapid_mlx.telemetry.inference import emit_completed_request\n"
        "from rapid_mlx.telemetry.model_id import engine_telemetry_id\n\n"
        "def scratch(request):\n"
        "    engine = lookup_engine(request.model)\n"
        "    emit_completed_request(\n"
        "        model=engine_telemetry_id(engine), endpoint='/v1/test'\n"
        "    )\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == []


def test_lambda_default_request_root_fails_closed(tmp_path):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n\n"
        "def scratch(request):\n"
        "    callback = lambda rq=request: emit_capability_rejected(\n"
        "        'unsupported', model=rq.model\n"
        "    )\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == ["rapid_mlx/routes/scratch.py:4"]


def test_module_qualified_engine_lookup_remains_a_privacy_boundary(tmp_path):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "import rapid_mlx.service.helpers as helpers\n"
        "from rapid_mlx.telemetry.inference import emit_completed_request\n"
        "from rapid_mlx.telemetry.model_id import engine_telemetry_id\n\n"
        "def scratch(request):\n"
        "    engine = helpers.get_engine(request.model)\n"
        "    emit_completed_request(\n"
        "        model=engine_telemetry_id(engine), endpoint='/v1/test'\n"
        "    )\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == []


def test_request_model_privacy_gate_rejects_module_rebound_lookup(tmp_path):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from rapid_mlx.service.helpers import get_engine as lookup\n"
        "from rapid_mlx.telemetry.inference import emit_completed_request\n"
        "from rapid_mlx.telemetry.model_id import engine_telemetry_id\n"
        "lookup = lambda value: value\n\n"
        "def scratch(request):\n"
        "    engine = lookup(request.model)\n"
        "    emit_completed_request(\n"
        "        model=engine_telemetry_id(engine), endpoint='/v1/test'\n"
        "    )\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == ["rapid_mlx/routes/scratch.py:8"]


def test_request_model_privacy_gate_rejects_enclosing_rebound_lookup(tmp_path):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from rapid_mlx.service.helpers import get_engine as lookup\n"
        "from rapid_mlx.telemetry.inference import emit_completed_request\n"
        "from rapid_mlx.telemetry.model_id import engine_telemetry_id\n\n"
        "def outer(request):\n"
        "    lookup = lambda value: value\n"
        "    def scratch():\n"
        "        engine = lookup(request.model)\n"
        "        emit_completed_request(\n"
        "            model=engine_telemetry_id(engine), endpoint='/v1/test'\n"
        "        )\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == ["rapid_mlx/routes/scratch.py:9"]


@pytest.mark.parametrize(
    ("module", "type_name", "parameter"),
    [
        ("rapid_mlx.api.anthropic_models", "AnthropicRequest", "anthropic_request"),
        ("rapid_mlx.api.models", "ChatCompletionRequest", "openai_request"),
    ],
    ids=["anthropic-request", "openai-request"],
)
def test_request_model_privacy_gate_derives_real_handler_roots(
    tmp_path, module, type_name, parameter
):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        f"from {module} import {type_name}\n"
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n\n"
        f"def scratch({parameter}: {type_name}):\n"
        f"    model = {parameter}.model\n"
        "    emit_capability_rejected('unsupported', model=model)\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == ["rapid_mlx/routes/scratch.py:6"]


@pytest.mark.parametrize(
    "annotation",
    [
        "models.ChatCompletionRequest",
        "Optional[models.ChatCompletionRequest]",
        "Annotated[models.ChatCompletionRequest, 'request']",
        "models.ChatCompletionRequest | None",
        "'models.ChatCompletionRequest'",
    ],
    ids=["qualified", "optional", "annotated", "union", "string"],
)
def test_request_model_privacy_gate_reads_terminal_annotation_names(
    tmp_path, annotation
):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from typing import Annotated, Optional\n"
        "from rapid_mlx.api import models\n"
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n\n"
        f"def scratch(foo: {annotation}):\n"
        "    emit_capability_rejected('unsupported', model=foo.model)\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == ["rapid_mlx/routes/scratch.py:6"]


@pytest.mark.parametrize(
    ("parameter", "setup", "nested"),
    [
        (
            "foo: ChatCompletionRequest",
            "",
            "def nested():\n        emit_capability_rejected('unsupported', model=foo.model)",
        ),
        (
            "ignored",
            "",
            "def nested():\n        emit_capability_rejected('unsupported', model=request.model)",
        ),
        (
            "foo: ChatCompletionRequest",
            "model = foo.model",
            "def nested():\n        emit_capability_rejected('unsupported', model=model)",
        ),
        (
            "foo: ChatCompletionRequest",
            "",
            "nested = lambda: emit_capability_rejected('unsupported', model=foo.model)",
        ),
        (
            "foo: ChatCompletionRequest",
            "",
            "nested = [emit_capability_rejected('unsupported', model=foo.model) for _ in range(1)]",
        ),
    ],
    ids=[
        "nested-typed-root",
        "nested-regex-root",
        "nested-tainted-name",
        "lambda-closure",
        "comprehension-closure",
    ],
)
def test_request_model_privacy_gate_inherits_closure_roots_and_taint(
    tmp_path, parameter, setup, nested
):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    body = [line for line in (setup, nested) if line]
    source = (
        "from rapid_mlx.api.models import ChatCompletionRequest\n"
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n\n"
        f"def scratch({parameter}):\n"
        + "\n".join(f"    {line}" for line in "\n".join(body).splitlines())
        + "\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    violations = _request_model_violations(tmp_path)
    assert len(violations) == 1
    assert violations[0].startswith("rapid_mlx/routes/scratch.py:")


def test_request_model_privacy_gate_checks_class_methods(tmp_path):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n\n"
        "class Handler:\n"
        "    def serve(self, request):\n"
        "        emit_capability_rejected('unsupported', model=request.model)\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == ["rapid_mlx/routes/scratch.py:5"]


@pytest.mark.parametrize(
    "body",
    [
        (
            "def outer(foo: ChatCompletionRequest):\n"
            "    def nested(foo):\n"
            "        emit_capability_rejected('unsupported', model=foo.model)\n"
        ),
        (
            "def outer(foo: ChatCompletionRequest):\n"
            "    model = foo.model\n"
            "    def nested():\n"
            "        model = 'resident'\n"
            "        emit_capability_rejected('unsupported', model=model)\n"
        ),
    ],
    ids=["parameter-shadows-root", "assignment-shadows-taint"],
)
def test_request_model_privacy_gate_respects_nested_function_shadowing(tmp_path, body):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from rapid_mlx.api.models import ChatCompletionRequest\n"
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n\n" + body
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == []


@pytest.mark.parametrize(
    "model_expression",
    ['request["model"]', 'getattr(request, "model")'],
    ids=["mapping-access", "getattr"],
)
def test_request_model_privacy_gate_rejects_all_model_access_forms(
    tmp_path, model_expression
):
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    source = (
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n\n"
        "def scratch(request):\n"
        f"    model = {model_expression}\n"
        "    emit_capability_rejected('unsupported', model=model)\n"
    )
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")

    assert _request_model_violations(tmp_path) == ["rapid_mlx/routes/scratch.py:5"]


def test_image_unavailable_reports_resident_engine_telemetry_id(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.routes import images
    from rapid_mlx.telemetry import inference, model_id

    engine = SimpleNamespace(is_image_gen=False, modality="text")
    cfg = SimpleNamespace(
        engine=engine,
        model_alias="private-alias",
        model_name="private-name",
        model_path="qwen3.5-4b-4bit",
        model_registry=None,
    )
    monkeypatch.setattr("rapid_mlx.config.get_config", lambda: cfg)
    monkeypatch.setattr("rapid_mlx.config.server_config.get_config", lambda: cfg)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda _capability, **context: calls.append(context),
    )

    with pytest.raises(HTTPException) as exc_info:
        images._image_engine()

    assert exc_info.value.status_code == 409
    assert calls == [
        {
            "model_type": "llm",
            "model": model_id.engine_telemetry_id(engine),
            "caller_agent": None,
            "caller_client": None,
        }
    ]


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
        lambda capability, *, model_type="other", **_context: calls.append(
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
        lambda capability, *, model_type="other", **_context: calls.append(
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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(inference.track_module, "_upload_allowed", lambda: True)
    monkeypatch.setattr(inference, "_submit", lambda work: work())
    monkeypatch.setattr(
        inference.track_module,
        "track",
        lambda event, props: calls.append((event, dict(props))),
    )

    with pytest.raises(HTTPException) as exc_info:
        helpers.enforce_context_length(
            engine,
            8,
            max_tokens=1,
            telemetry_model="qwen3.5-4b-4bit",
            caller_agent="private-agent/1.0 cursor/0.50",
            caller_client="rapid-desktop",
        )

    assert exc_info.value.status_code == 400
    assert calls == [
        (
            "capability_rejected",
            {
                "capability": "context_length_exceeded",
                "model_type": "llm",
                "model": "qwen3.5-4b-4bit",
                "caller": "rapid-desktop",
            },
        )
    ]


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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
        lambda value, *, model_type="other", **_context: calls.append(
            (value, model_type)
        ),
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
        lambda capability, *, model_type="other", **_context: calls.append(
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
        lambda capability, *, model_type="other", **_context: calls.append(
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
        lambda capability, *, model_type="other", **_context: calls.append(
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
        # The runner names the RESIDENT engine that served the request; the
        # completed event reports that, never the request's form field.
        audio._note_served_stt_engine(
            SimpleNamespace(model_name=audio._resolve_stt_model("whisper-large-v3"))
        )
        return response

    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    monkeypatch.setattr(
        audio, "_reject_word_timestamps_for_non_whisper", lambda *_a, **_k: None
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
            "model": "whisper",
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
        audio, "_reject_word_timestamps_for_non_whisper", lambda *_a, **_k: None
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
    # No engine was noted by the (stubbed) runner: fail closed to <custom>.
    assert calls[0]["model"] == "<custom>"
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
        inference.emit_capability_rejected(
            "logprobs_unsupported",
            model_type="llm",
            model="neohorse-9b-4bit",
            caller_agent=caller_agent,
            caller_client=caller_client,
        )
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
        "model": "neohorse-9b-4bit",
        "caller": expected_caller,
    }
    assert "hostname" not in repr(items)
    assert "username" not in repr(items)
    assert str(tmp_path) not in repr(items)
    assert "127.0.0.1" not in repr(items)


_PRIVATE_FORM_MODEL = "evil-org/private-repo"


@pytest.mark.parametrize(
    ("path", "data", "code"),
    [
        (
            "/v1/audio/translations",
            {"model": _PRIVATE_FORM_MODEL},
            "invalid_model_for_translation",
        ),
        (
            "/v1/audio/transcriptions",
            {
                "model": _PRIVATE_FORM_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
            "invalid_model_for_word_timestamps",
        ),
    ],
    ids=["translation-non-whisper", "word-timestamps-non-whisper"],
)
def test_audio_pre_engine_rejection_never_reports_form_model_on_the_wire(
    monkeypatch, path, data, code
):
    """A multipart ``model`` field is request input, not telemetry identity.

    ``telemetry_model_id`` is stubbed to echo its input (as it would for a
    repo with public proof), so any request-derived value that reached it
    would land on the loopback wire verbatim.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import rapid_mlx
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import audio
    from rapid_mlx.telemetry import (
        consent_runtime,
        inference,
        model_id,
        posthog_sender,
    )
    from rapid_mlx.telemetry import track as track_module
    from rapid_mlx.telemetry.build_gate import ReleaseStamp

    stamp = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: stamp)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    identity_inputs: list[object] = []
    monkeypatch.setattr(
        model_id,
        "telemetry_model_id",
        lambda ref: identity_inputs.append(ref) or str(ref),
    )
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
        post=post, gate=lambda: stamp, allowed=lambda: True
    )
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", None)
    app = FastAPI()
    app.include_router(audio.router)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = TestClient(app).post(
            path,
            data=data,
            files={"file": ("clip.wav", b"RIFF", "audio/wav")},
        )
        inference._QUEUE.join()
        sender.flush(2.0)
    finally:
        sender.close(0.5)
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == code
    items = [
        item
        for body in server.bodies  # type: ignore[attr-defined]
        for item in json.loads(body)["batch"]
    ]
    rejected = [item for item in items if item["event"] == "capability_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["properties"]["capability"] == "speech_capability_unsupported"
    assert "model" not in rejected[0]["properties"]
    assert identity_inputs == []
    wire = b"".join(server.bodies).decode()  # type: ignore[attr-defined]
    assert "evil-org" not in wire
    assert "private-repo" not in wire


def _scratch_route(tmp_path, source: str) -> list[str]:
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    (route_dir / "scratch.py").write_text(source, encoding="utf-8")
    return _request_model_violations(tmp_path)


def test_privacy_gate_taints_form_field_through_module_helpers(tmp_path):
    """The pre-fix audio shape: handler Form field -> resolver -> helper sink."""
    source = (
        "from fastapi import APIRouter, Form, UploadFile\n"
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n"
        "router = APIRouter()\n\n"
        "def _resolve(model):\n"
        "    return ALIASES.get(model, model)\n\n"
        "def _reject(model, *, caller_agent=None):\n"
        "    resolved = _resolve(model)\n"
        "    emit_capability_rejected('x', model=resolved)\n\n"
        "def _later(file, choice=None):\n"
        "    emit_capability_rejected('x', model=_resolve(choice))\n\n"
        "@router.post('/v1/scratch')\n"
        "async def handler(file: UploadFile, model_form: str | None = Form(None)):\n"
        "    model = model_form or 'default'\n"
        "    _reject(model, caller_agent=None)\n"
        "    _later(file, choice=model)\n"
    )
    assert _scratch_route(tmp_path, source) == [
        "rapid_mlx/routes/scratch.py:10",
        "rapid_mlx/routes/scratch.py:13",
    ]


@pytest.mark.parametrize(
    ("signature", "decorator"),
    [
        ("model: str = ''", "@router.get('/v1/scratch')\n"),
        ("model: str = Query('')", ""),
        ("model: Annotated[str, Header()] = ''", ""),
        ("model: str = Body(...)", ""),
    ],
    ids=["route-handler-bare-param", "query-default", "header-annotation", "body"],
)
def test_privacy_gate_treats_route_and_marker_params_as_values(
    tmp_path, signature, decorator
):
    source = (
        "from typing import Annotated\n"
        "from fastapi import APIRouter, Body, Header, Query\n"
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n"
        "router = APIRouter()\n\n"
        f"{decorator}async def handler({signature}):\n"
        "    emit_capability_rejected('x', model=model.strip())\n"
    )
    violations = _scratch_route(tmp_path, source)
    assert len(violations) == 1
    assert violations[0].startswith("rapid_mlx/routes/scratch.py:")


def test_privacy_gate_leaves_depends_and_engine_lookup_boundaries_clean(tmp_path):
    source = (
        "from fastapi import APIRouter, Depends, Form\n"
        "from rapid_mlx.telemetry.inference import emit_capability_rejected\n"
        "from rapid_mlx.telemetry.model_id import engine_telemetry_id\n"
        "router = APIRouter()\n\n"
        "def _image_engine(model_name=''):\n"
        "    engine = REGISTRY.get_engine(model_name)\n"
        "    emit_capability_rejected('x', model=engine_telemetry_id(engine))\n"
        "    return engine\n\n"
        "@router.post('/v1/scratch')\n"
        "async def handler(model: str = Form(''), engine=Depends(current_engine)):\n"
        "    _image_engine(model)\n"
        "    emit_capability_rejected('x', model=engine_telemetry_id(engine))\n"
    )
    route_dir = tmp_path / "rapid_mlx/routes"
    route_dir.mkdir(parents=True)
    (route_dir / "images.py").write_text(source, encoding="utf-8")
    assert _request_model_violations(tmp_path) == []
