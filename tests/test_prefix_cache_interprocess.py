"""Automatic prefix-cache shutdown saves sharing one HOME."""

from __future__ import annotations

import fcntl
import multiprocessing
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.runtime import cache as runtime_cache


def _save_worker(cache_dir, name, entered, release, outcome):
    from rapid_mlx.runtime import cache as runtime_cache

    saved = []

    class Engine:
        def save_cache_to_disk(self, path, should_abort=None):
            stage = Path(path + ".new")
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir()
            marker = stage / name
            marker.write_text(name)
            entered.set()
            if name == "first" and not release.wait(10):
                return False
            result = marker.exists()
            saved.append(result)
            return result

    runtime_cache.get_config = lambda: SimpleNamespace(engine=Engine())
    runtime_cache.get_cache_dir = lambda: cache_dir
    runtime_cache.save_prefix_cache_to_disk(budget_sec=0)
    outcome.put(saved[0] if saved else None)


def test_concurrent_shutdown_saves_do_not_clobber_staging(tmp_path):
    """The second server must not remove the first writer's ``.new`` dir."""
    context = multiprocessing.get_context("spawn")
    cache_dir = str(tmp_path / "model")
    first_entered = context.Event()
    second_entered = context.Event()
    release_first = context.Event()
    first_outcome = context.Queue()
    second_outcome = context.Queue()
    first = context.Process(
        target=_save_worker,
        args=(cache_dir, "first", first_entered, release_first, first_outcome),
    )
    second = context.Process(
        target=_save_worker,
        args=(cache_dir, "second", second_entered, release_first, second_outcome),
    )
    try:
        first.start()
        assert first_entered.wait(10)
        second.start()
        second.join(10)
    finally:
        release_first.set()
        first.join(10)
        if second.pid and second.is_alive():
            second.terminate()
            second.join(10)
        if first.pid and first.is_alive():
            first.terminate()
            first.join(10)

    assert first.exitcode == second.exitcode == 0
    assert first_outcome.get(timeout=2) is True
    assert second_outcome.get(timeout=2) is None
    assert not second_entered.is_set()
    assert (Path(cache_dir + ".new") / "first").read_text() == "first"


def test_startup_load_skips_active_shutdown_save(tmp_path, monkeypatch):
    """Crash recovery on startup must not clear an active writer's stage."""
    context = multiprocessing.get_context("spawn")
    cache_dir = str(tmp_path / "model")
    save_entered = context.Event()
    release_save = context.Event()
    outcome = context.Queue()
    saver = context.Process(
        target=_save_worker,
        args=(cache_dir, "first", save_entered, release_save, outcome),
    )

    class Engine:
        def load_cache_from_disk(self, path, protected_import=False):
            shutil.rmtree(path + ".new", ignore_errors=True)
            return 0

    monkeypatch.setattr(
        runtime_cache, "get_config", lambda: SimpleNamespace(engine=Engine())
    )
    monkeypatch.setattr(runtime_cache, "get_cache_dir", lambda: cache_dir)
    try:
        saver.start()
        assert save_entered.wait(10)
        runtime_cache.load_prefix_cache_from_disk()
        assert (Path(cache_dir + ".new") / "first").is_file()
    finally:
        release_save.set()
        saver.join(10)
        if saver.pid and saver.is_alive():
            saver.terminate()
            saver.join(10)

    assert saver.exitcode == 0
    assert outcome.get(timeout=2) is True


def test_busy_shutdown_save_skips_without_touching_cache(tmp_path, monkeypatch):
    """A second shutdown must leave the active writer's stage alone."""
    cache_dir = str(tmp_path / "model")
    calls = []

    class Engine:
        def save_cache_to_disk(self, path, should_abort=None):
            calls.append(path)
            return False

    monkeypatch.setattr(
        runtime_cache, "get_config", lambda: SimpleNamespace(engine=Engine())
    )
    monkeypatch.setattr(runtime_cache, "get_cache_dir", lambda: cache_dir)
    with open(cache_dir + ".txlock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        runtime_cache.save_prefix_cache_to_disk(budget_sec=0)
    assert calls == []


@pytest.mark.parametrize("saved,budget", [(True, 1.0), (False, 0)])
def test_shutdown_save_holds_lock_through_radix_commit(
    tmp_path, monkeypatch, saved, budget
):
    """The cache writer and its radix commit share one transaction lock."""
    cache_dir = str(tmp_path / "model")
    writes = []
    radix = []

    def assert_locked():
        with open(cache_dir + ".txlock") as lock_file, pytest.raises(BlockingIOError):
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    class Engine:
        def save_cache_to_disk(self, path, should_abort=None):
            assert_locked()
            writes.append((path, callable(should_abort)))
            return saved

    def save_radix(engine, path):
        assert_locked()
        radix.append(path)

    monkeypatch.setattr(
        runtime_cache, "get_config", lambda: SimpleNamespace(engine=Engine())
    )
    monkeypatch.setattr(runtime_cache, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(runtime_cache, "_save_radix_index_after_cache", save_radix)

    runtime_cache.save_prefix_cache_to_disk(budget_sec=budget)

    assert writes == [(cache_dir, budget > 0)]
    assert radix == ([cache_dir] if saved else [])


@pytest.mark.parametrize("loaded", [0, 1])
def test_startup_load_holds_lock_through_radix_restore(tmp_path, monkeypatch, loaded):
    """Recovery and radix restore must see one settled cache snapshot."""
    cache_dir = str(tmp_path / "model")
    reads = []
    radix = []

    def assert_locked():
        with open(cache_dir + ".txlock") as lock_file, pytest.raises(BlockingIOError):
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    class Engine:
        def load_cache_from_disk(self, path, protected_import=False):
            assert_locked()
            reads.append((path, protected_import))
            return loaded

    def restore_radix(engine, path):
        assert_locked()
        radix.append(path)

    monkeypatch.setattr(
        runtime_cache, "get_config", lambda: SimpleNamespace(engine=Engine())
    )
    monkeypatch.setattr(runtime_cache, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(runtime_cache, "_load_radix_index_after_cache", restore_radix)

    runtime_cache.load_prefix_cache_from_disk()

    assert reads == [(cache_dir, False)]
    assert radix == [cache_dir]
