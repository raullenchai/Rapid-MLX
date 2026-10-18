# SPDX-License-Identifier: Apache-2.0
"""Retrieval admission and lifecycle contracts, runnable without MLX."""

import asyncio
import threading

import pytest

from rapid_mlx.errors import BackpressureError
from rapid_mlx.routes import _mlx_worker as worker


@pytest.fixture(autouse=True)
def clean_worker():
    assert worker.MAX_PENDING_RETRIEVAL_JOBS == 64
    worker.shutdown_mlx_worker()
    yield
    worker.shutdown_mlx_worker()


async def test_success_failure_and_sync_bypass(monkeypatch):
    assert await worker.run_on_mlx_worker(lambda x: x + 1, 2) == 3
    assert worker._pending_jobs == 0

    def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        await worker.run_on_mlx_worker(fail)
    assert worker._pending_jobs == 0
    monkeypatch.setattr(worker, "MAX_PENDING_RETRIEVAL_JOBS", 0)
    assert worker.run_on_mlx_worker_sync(lambda: 42) == 42
    with pytest.raises(BackpressureError):
        await worker.run_on_mlx_worker(lambda: None)
    assert worker._pending_jobs == 0


async def test_full_queue_and_cancelled_awaiters_keep_slots(monkeypatch):
    monkeypatch.setattr(worker, "MAX_PENDING_RETRIEVAL_JOBS", 2)
    started, release = threading.Event(), threading.Event()
    queued_ran = threading.Event()

    def blocking():
        started.set()
        assert release.wait(5)

    first = asyncio.create_task(worker.run_on_mlx_worker(blocking))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(worker.run_on_mlx_worker(queued_ran.set))
        await asyncio.sleep(0)
        assert worker._pending_jobs == 2
        for task in [first, second]:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        with pytest.raises(worker.RetrievalWorkerBusyError):
            await worker.run_on_mlx_worker(lambda: None)
        assert worker._pending_jobs == 2
        release.set()
        await asyncio.to_thread(worker.run_on_mlx_worker_sync, lambda: None)
        assert queued_ran.is_set()
        assert worker._pending_jobs == 0
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )


async def test_shutdown_reuse_and_detached_completion():
    release = threading.Event()
    started = threading.Event()

    def blocking():
        started.set()
        assert release.wait(5)

    old = asyncio.create_task(worker.run_on_mlx_worker(blocking))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        executor = worker._retrieval_executor
        worker.shutdown_mlx_worker(wait=False)
        assert worker._pending_jobs == 0
        assert await worker.run_on_mlx_worker(lambda: "new") == "new"
        assert worker._retrieval_executor is not executor
        release.set()
        await old
        assert worker._pending_jobs == 0
        worker.shutdown_mlx_worker()
        assert worker._retrieval_executor is None
        worker.shutdown_mlx_worker()
        assert worker.run_on_mlx_worker_sync(lambda: 7) == 7
    finally:
        release.set()
        await old


@pytest.mark.parametrize(
    "payload", [{"documents": []}, {"documents": ["d"], "instruction": ""}]
)
def test_rerank_request_validation_returns_400(payload):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.api.models import RerankRequest
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers

    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/v1/rerank")
    def validate(request: RerankRequest):
        pytest.fail("Invalid payload reached handler")

    response = TestClient(app).post("/v1/rerank", json={"query": "q", **payload})
    assert response.status_code == 400


def test_retrieval_cli_defaults_and_override():
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.config.server_config import ServerConfig

    parser = build_parser()
    args = parser.parse_args(["serve", "--rerank-model", "model"])
    assert args.rerank_max_documents == ServerConfig().rerank_max_documents == 1000
    assert args.embedding_max_inputs == ServerConfig().embedding_max_inputs == 2048
    args = parser.parse_args(
        ["serve", "--embedding-max-inputs", "0", "--rerank-max-documents", "-1"]
    )
    assert args.embedding_max_inputs == 0
    assert args.rerank_max_documents == -1
