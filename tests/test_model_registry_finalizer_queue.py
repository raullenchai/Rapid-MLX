# SPDX-License-Identifier: Apache-2.0
import pytest

from rapid_mlx.model_registry import ModelRegistry


class Stub:
    pass


@pytest.fixture
def registry():
    registry = object.__new__(ModelRegistry)
    registry._init()
    return registry


def queue_release(registry, model, engine, engine_id="e1"):
    with registry._registry_lock:
        registry.release_from_finalizer(model, engine_id, engine=engine)
        assert len(registry._pending_releases) == 1


@pytest.mark.parametrize("method", ["is_owned", "cleanup", "get_stats"])
def test_methods_drain_pending_release(registry, method):
    model, engine = Stub(), Stub()
    registry.acquire(model, engine, "e1")
    queue_release(registry, model, engine)

    if method == "is_owned":
        registry.is_owned(model)
    else:
        getattr(registry, method)()

    assert not registry._pending_releases
    assert id(model) not in registry._owners


def test_release_drains_pending_release(registry):
    model, engine, other_model = Stub(), Stub(), Stub()
    registry.acquire(model, engine, "e1")
    queue_release(registry, model, engine)

    registry.release(other_model, "x")

    assert not registry._pending_releases
    assert id(model) not in registry._owners


def test_acquire_drains_pending_release(registry):
    model, engine, new_engine = Stub(), Stub(), Stub()
    registry.acquire(model, engine, "e1")
    queue_release(registry, model, engine)

    registry.acquire(model, new_engine, "e2")

    assert registry._owners[id(model)][1] == "e2"
    assert not registry._pending_releases


def test_stale_release_preserves_live_owner(registry):
    model, engine, other_engine = Stub(), Stub(), Stub()
    registry.acquire(model, engine, "same-id")
    queue_release(registry, model, other_engine, "same-id")

    registry.get_stats()

    assert not registry._pending_releases
    assert registry._owners[id(model)][0]() is engine
    assert registry._owners[id(model)][1] == "same-id"


def test_pending_release_removes_dead_owner(registry):
    model, engine = Stub(), Stub()
    registry.acquire(model, engine, "e1")
    del engine
    queue_release(registry, model, None)

    registry.cleanup()

    assert not registry._pending_releases
    assert id(model) not in registry._owners
