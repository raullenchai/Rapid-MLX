# SPDX-License-Identifier: Apache-2.0
"""Image lane MLX thread affinity (#5004).

MLX binds streams to the thread that creates them, so a lazy array built while
loading an image model on one worker thread cannot be evaluated by a later
request that the route dispatches to a different worker thread. The image lane
must run model load and generation on one dedicated thread.
"""

import asyncio
import concurrent.futures
import os
import signal
import threading
import warnings

import pytest
from PIL import Image

from rapid_mlx.api.models import ImageGenerationRequest
from rapid_mlx.image import engine as image_engine_module
from rapid_mlx.image.engine import ImageGenerationEngine
from rapid_mlx.routes import images as images_route
from rapid_mlx.routes._async_utils import run_to_completion
from rapid_mlx.runtime import image_lane
from rapid_mlx.runtime.image_lane import ImageEngine

_MODEL = "black-forest-labs/FLUX.1-schnell"


class _GeneratedImage:
    def __init__(self) -> None:
        self.image = Image.new("RGB", (4, 4), (10, 20, 30))


class _ThreadRecordingModel:
    """Fake mflux model that records which thread loads and renders it."""

    def __init__(self, on_load=None, on_generate=None) -> None:  # noqa: ANN001
        self.load_thread = threading.current_thread()
        self.generate_threads: list[threading.Thread] = []
        self.state = on_load() if on_load is not None else None
        self._on_generate = on_generate

    def generate_image(self, **kwargs) -> _GeneratedImage:  # noqa: ANN003
        del kwargs
        self.generate_threads.append(threading.current_thread())
        if self._on_generate is not None:
            self._on_generate(self.state)
        return _GeneratedImage()


def _image_engine(monkeypatch: pytest.MonkeyPatch, model_factory) -> ImageEngine:  # noqa: ANN001
    engine = ImageEngine(_MODEL)
    inner: ImageGenerationEngine = engine._engine  # noqa: SLF001
    monkeypatch.setattr(inner, "_build_model", model_factory)
    monkeypatch.setattr(inner, "_ensure_runtime_assets", lambda: None)
    monkeypatch.setattr(inner, "_verify_weights_complete", lambda: None)
    monkeypatch.setattr(
        image_engine_module, "_release_allocator_cache", lambda: None, raising=True
    )
    monkeypatch.setattr("rapid_mlx.model_downloads.require_local", lambda _name: None)
    monkeypatch.setattr(engine, "_emit_model_served", lambda: None)
    return engine


def _request() -> ImageGenerationRequest:
    return ImageGenerationRequest(prompt="a lighthouse", size="256x256", steps=1)


async def _route_generate_on_fresh_thread(engine: ImageEngine, seed: int) -> bytes:
    """Dispatch exactly like the images route, forcing a new worker thread.

    Swapping the loop's default executor guarantees this request's
    ``asyncio.to_thread`` hop lands on a thread no earlier request used — the
    situation a request arriving during a busy render hits in production.
    """
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    try:
        png_bytes, _performance = await run_to_completion(
            images_route._generate_one,  # noqa: SLF001
            engine,
            _request(),
            seed,
        )
        return png_bytes
    finally:
        executor.shutdown(wait=True)


def test_load_and_generations_share_one_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: list[_ThreadRecordingModel] = []

    def build() -> _ThreadRecordingModel:
        models.append(_ThreadRecordingModel())
        return models[-1]

    engine = _image_engine(monkeypatch, build)

    async def scenario() -> None:
        await asyncio.to_thread(engine.ensure_resident, mode="generation")
        await _route_generate_on_fresh_thread(engine, 1)
        await _route_generate_on_fresh_thread(engine, 2)

    asyncio.run(scenario())

    (model,) = models
    assert len(model.generate_threads) == 2
    assert model.generate_threads[0] is model.load_thread
    assert model.generate_threads[1] is model.load_thread
    assert model.load_thread.name.startswith("mlx-image")


def test_direct_generate_runs_on_worker_and_nested_calls_run_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: list[_ThreadRecordingModel] = []

    def build() -> _ThreadRecordingModel:
        models.append(_ThreadRecordingModel())
        return models[-1]

    engine = _image_engine(monkeypatch, build)
    png = engine.generate(prompt="x", width=256, height=256, seed=3)
    assert png.startswith(b"\x89PNG")
    (model,) = models
    assert model.generate_threads == [model.load_thread]

    # Re-entrant use from the worker itself must not deadlock the one thread.
    nested = image_lane.run_on_image_worker(
        lambda: image_lane.run_on_image_worker(threading.current_thread)
    )
    assert nested is model.load_thread


def test_worker_propagates_engine_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_state) -> None:  # noqa: ANN001
        raise ValueError("boom")

    engine = _image_engine(monkeypatch, lambda: _ThreadRecordingModel(on_generate=fail))
    with pytest.raises(image_engine_module.ImageRuntimeError, match="boom"):
        engine.generate_with_performance(prompt="x", width=256, height=256, seed=1)


@pytest.mark.requires_mlx
def test_lazy_mlx_state_from_load_survives_requests_on_new_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A lazily loaded CPU array must stay usable by later requests (#5004)."""
    mx = pytest.importorskip("mlx.core")
    weights = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(weights), {"w": mx.ones((4,), stream=mx.cpu)})

    def load() -> dict:
        # ``mx.load`` is lazy: the read is scheduled on the loading thread's
        # CPU stream and only executes when first evaluated.
        return mx.load(str(weights))

    def render(state: dict) -> None:
        mx.eval(mx.add(state["w"], 1, stream=mx.cpu))

    engine = _image_engine(
        monkeypatch, lambda: _ThreadRecordingModel(on_load=load, on_generate=render)
    )

    async def scenario() -> None:
        await asyncio.to_thread(engine.ensure_resident, mode="generation")
        await _route_generate_on_fresh_thread(engine, 1)
        await _route_generate_on_fresh_thread(engine, 2)

    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["generations", "edits"])
def test_image_500_logs_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, endpoint: str
) -> None:
    import io
    import types

    from fastapi.testclient import TestClient

    from rapid_mlx.server import app

    def fail(**_kwargs) -> bytes:  # noqa: ANN003
        try:
            raise RuntimeError("There is no Stream(cpu, 2) in current thread.")
        except RuntimeError as exc:
            raise image_engine_module.ImageRuntimeError(
                f"Image generation failed: {exc}"
            ) from exc

    engine = types.SimpleNamespace(
        is_image_gen=True,
        is_edit=endpoint == "edits",
        supports_generation=True,
        supports_editing=True,
        default_steps=1,
        family="flux2-klein",
        model_name="repo/fake-image",
        generate=fail,
    )
    monkeypatch.setattr(
        "rapid_mlx.config.get_config", lambda: types.SimpleNamespace(engine=engine)
    )
    caplog.set_level("ERROR", logger=images_route.logger.name)
    client = TestClient(app)
    if endpoint == "generations":
        response = client.post("/v1/images/generations", json={"prompt": "fox"})
    else:
        upload = io.BytesIO()
        Image.new("RGB", (16, 16), (90, 90, 90)).save(upload, format="PNG")
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "fox"},
            files={"image": ("source.png", upload.getvalue(), "image/png")},
        )

    assert response.status_code == 500
    (record,) = [
        r
        for r in caplog.records
        if r.name == images_route.logger.name and r.exc_info is not None
    ]
    assert "no Stream" in str(record.exc_info[1])
    assert isinstance(record.exc_info[1].__cause__, RuntimeError)


def test_forked_child_gets_a_fresh_image_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Start the worker, then simulate the after-fork hook a child process runs.
    image_lane.run_on_image_worker(lambda: None)
    parent_lock = image_lane._image_worker_lock  # noqa: SLF001
    parent_executor = image_lane._image_executor  # noqa: SLF001
    assert parent_executor is not None
    monkeypatch.setattr(image_lane, "_image_worker_lock", parent_lock)
    monkeypatch.setattr(image_lane, "_image_executor", parent_executor)

    # Model a fork taken from the worker: the child's only thread carries the
    # inherited worker marker until the hook clears it.
    monkeypatch.setattr(image_lane, "_image_worker_state", threading.local())
    image_lane._mark_image_worker_thread()  # noqa: SLF001

    image_lane._reset_image_worker_after_fork()  # noqa: SLF001

    assert image_lane._image_executor is None  # noqa: SLF001
    assert image_lane._image_worker_lock is not parent_lock  # noqa: SLF001
    worker = image_lane.run_on_image_worker(threading.current_thread)
    assert worker is not threading.current_thread()
    assert worker.name.startswith("mlx-image")
    child_executor = image_lane._image_executor  # noqa: SLF001
    assert child_executor is not parent_executor
    child_executor.shutdown(wait=True)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_real_fork_from_worker_gives_child_a_live_worker() -> None:
    """A child forked from the image worker can still dispatch (no hang)."""

    def fork_from_worker() -> int:
        with warnings.catch_warnings():
            # CPython warns about forking a multi-threaded process; the
            # at-fork hook is exactly what makes this child usable.
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid != 0:
            return pid
        status = 1
        try:
            signal.alarm(10)  # a hang fails the child instead of the suite
            caller = threading.get_ident()
            worker_ident, worker_name = image_lane.run_on_image_worker(
                lambda: (threading.get_ident(), threading.current_thread().name)
            )
            if worker_ident != caller and worker_name.startswith("mlx-image"):
                status = 0
        finally:
            os._exit(status)

    pid = image_lane.run_on_image_worker(fork_from_worker)
    _, wait_status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(wait_status) == 0
