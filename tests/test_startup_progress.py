# SPDX-License-Identifier: Apache-2.0
"""Startup counters observe completed evaluations without changing readiness."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rapid_mlx import startup_progress as progress_module
from rapid_mlx.startup_progress import StartupProgress, load_with_progress


@pytest.fixture
def progress(monkeypatch):
    state = StartupProgress()
    monkeypatch.setattr(progress_module, "startup_progress", state)
    return state


def test_monotonic_thread_safe_snapshots(progress):
    progress.begin("organization/checkpoint")
    assert progress.claim_weights()
    progress.weights_total(1000)

    def update(done):
        progress.weights_done(done)
        snapshot = progress.snapshot()
        assert done <= snapshot["progress"]["done"] <= 1000
        snapshot["progress"]["done"] = -1  # snapshots cannot mutate shared state

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(update, range(1001)))
    progress.weights_done(-1)
    progress.weights_total(2000)  # denominator is immutable
    assert progress.snapshot()["progress"] == {
        "done": 1000,
        "total": 1000,
        "unit": "bytes",
    }
    progress.phase("initializing")
    progress.phase("loading_weights")
    assert progress.snapshot() == {"phase": "initializing", "model": "checkpoint"}
    assert not progress.claim_weights()  # draft model never resets percentage
    progress.phase("ready")
    progress.phase("resolving")
    assert progress.snapshot()["phase"] == "ready"


def test_only_one_concurrent_loader_claims_progress(progress):
    progress.begin("model")
    with ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(pool.map(lambda _: progress.claim_weights(), range(100))) == 1


@pytest.mark.requires_mlx
def test_materialization_reports_completed_bytes_on_calling_thread(
    progress, monkeypatch
):
    import threading

    import mlx.core as mx

    arrays = [SimpleNamespace(nbytes=256 * 1024**2) for _ in range(3)]
    model = SimpleNamespace(parameters=lambda: {"weights": arrays, "tied": arrays[0]})
    progress.begin("tiny")
    monkeypatch.setattr(progress_module, "_EVAL_BATCH_BYTES", 256 * 1024**2)
    observed = []
    thread = threading.get_ident()

    def evaluate(batch):
        assert threading.get_ident() == thread
        # The previous chunk alone is complete before this evaluation returns.
        observed.append(progress.snapshot()["progress"]["done"])
        assert len(batch) == 1

    monkeypatch.setattr(mx, "eval", evaluate)
    calls = []

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return model, {"model_type": "fixture"}

    assert load_with_progress(loader, "fixture")[0] is model
    assert calls == [("fixture", {"lazy": True})]
    assert observed == [0, 256 * 1024**2, 512 * 1024**2]
    assert progress.snapshot()["phase"] == "initializing"
    assert "progress" not in progress.snapshot()
    load_with_progress(loader, "draft")
    assert calls[-1] == ("draft", {})


@pytest.mark.parametrize("active,lazy", [(False, False), (True, True)])
def test_lazy_and_non_startup_loads_keep_original_contract(progress, active, lazy):
    if active:
        progress.begin("disk-stream")
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        return "unchanged"

    assert load_with_progress(loader, lazy=lazy) == "unchanged"
    assert calls == [{"lazy": lazy}]
    assert "progress" not in progress.snapshot()


@pytest.mark.requires_mlx
def test_failed_loader_falls_back_without_stale_percentage(progress):
    progress.begin("broken")

    def loader(**kwargs):
        raise ValueError("fixture failure")

    with pytest.raises(ValueError, match="fixture failure"):
        load_with_progress(loader)
    assert progress.snapshot() == {"phase": "initializing", "model": "broken"}
    assert not progress.claim_weights()


@pytest.mark.asyncio
async def test_readiness_body_is_additive_and_ready_body_unchanged(
    progress, monkeypatch
):
    from rapid_mlx.routes import health

    cfg = SimpleNamespace(
        ready=False,
        primary_model_lifecycle=None,
        model_name="organization/model",
        model_loaded=True,
    )
    monkeypatch.setattr(health, "get_config", lambda: cfg)
    monkeypatch.setattr(health, "primary_switching", lambda: False)
    progress.begin("/private/models/model")
    progress.claim_weights()
    progress.weights_total(100)
    progress.weights_done(63)
    response = await health.health_ready()
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "error": {
            "message": "model loading",
            "type": "api_error",
            "code": None,
            "param": None,
        },
        "ready": False,
        "model": "model",
        "phase": "loading_weights",
        "progress": {"done": 63, "total": 100, "unit": "bytes"},
    }
    cfg.ready = True
    assert await health.health_ready() == {
        "ready": True,
        "model": "organization/model",
        "state": "ready",
        "model_loaded": True,
    }


def test_stubbed_mlx_materializes_unique_arrays_in_batches(progress, monkeypatch):
    import sys
    from types import ModuleType

    mlx = ModuleType("mlx")
    mlx.__path__ = []
    core = ModuleType("mlx.core")
    utils = ModuleType("mlx.utils")
    first = SimpleNamespace(nbytes=4)
    second = SimpleNamespace(nbytes=3)
    third = SimpleNamespace(nbytes=1)
    utils.tree_flatten = lambda params: list(params.items())
    batches = []
    core.eval = lambda batch: batches.append((list(batch), progress.snapshot()))
    for name, module in (("mlx", mlx), ("mlx.core", core), ("mlx.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(progress_module, "_EVAL_BATCH_BYTES", 5)
    progress.begin("fixture")
    model = SimpleNamespace(
        parameters=lambda: {
            "first": first,
            "tied": first,
            "second": second,
            "third": third,
        }
    )
    calls = []

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return model, "processor"

    assert load_with_progress(loader, "fixture", option=True) == (model, "processor")
    assert calls == [("fixture", {"option": True, "lazy": True})]
    assert [batch for batch, _ in batches] == [[first, second], [third]]
    assert [state["progress"] for _, state in batches] == [
        {"done": 0, "total": 8, "unit": "bytes"},
        {"done": 7, "total": 8, "unit": "bytes"},
    ]
    assert progress.snapshot() == {"phase": "initializing", "model": "fixture"}


def test_stubbed_mlx_loader_failure_advances_phase(progress, monkeypatch):
    import sys
    from types import ModuleType

    mlx = ModuleType("mlx")
    mlx.__path__ = []
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda params: []
    for name, module in (
        ("mlx", mlx),
        ("mlx.core", ModuleType("mlx.core")),
        ("mlx.utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    progress.begin("broken")

    def loader(**kwargs):
        assert kwargs == {"lazy": True}
        raise ValueError("fixture failure")

    with pytest.raises(ValueError, match="fixture failure"):
        load_with_progress(loader)
    assert progress.snapshot() == {"phase": "initializing", "model": "broken"}


def test_loader_without_lazy_contract_keeps_original_call(progress):
    progress.begin("fixture")
    calls = []

    def strict_loader(path):
        calls.append(path)
        return "model"

    assert progress_module.load_with_progress(strict_loader, "p") == "model"
    assert calls == ["p"]
    assert progress.claim_weights()


def test_uninspectable_loader_is_not_wrapped(monkeypatch):
    def broken(_):
        raise ValueError("no signature")

    monkeypatch.setattr(progress_module.inspect, "signature", broken)
    assert progress_module._accepts_lazy(lambda **kw: None) is False
