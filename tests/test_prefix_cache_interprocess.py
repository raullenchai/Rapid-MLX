"""Automatic prefix-cache shutdown saves sharing one HOME."""

from __future__ import annotations

import multiprocessing
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

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


def _load_worker(cache_dir, ready, entered):
    from rapid_mlx.runtime import cache as runtime_cache

    class Engine:
        def load_cache_from_disk(self, path, protected_import=False):
            entered.set()
            shutil.rmtree(path + ".new", ignore_errors=True)
            return 0

    runtime_cache.get_config = lambda: SimpleNamespace(engine=Engine())
    runtime_cache.get_cache_dir = lambda: cache_dir
    ready.set()
    runtime_cache.load_prefix_cache_from_disk()


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


def test_startup_load_waits_for_shutdown_save(tmp_path):
    """Crash recovery on startup must not clear an active writer's stage."""
    context = multiprocessing.get_context("spawn")
    cache_dir = str(tmp_path / "model")
    save_entered = context.Event()
    load_ready = context.Event()
    load_entered = context.Event()
    release_save = context.Event()
    outcome = context.Queue()
    saver = context.Process(
        target=_save_worker,
        args=(cache_dir, "first", save_entered, release_save, outcome),
    )
    loader = context.Process(
        target=_load_worker, args=(cache_dir, load_ready, load_entered)
    )
    try:
        saver.start()
        assert save_entered.wait(10)
        loader.start()
        assert load_ready.wait(10)
        time.sleep(0.5)
        assert not load_entered.is_set()
    finally:
        release_save.set()
        saver.join(10)
        if loader.pid:
            loader.join(10)
        for process in (saver, loader):
            if process.pid and process.is_alive():
                process.terminate()
                process.join(10)

    assert saver.exitcode == loader.exitcode == 0
    assert outcome.get(timeout=2) is True
    assert load_entered.is_set()


def test_busy_shutdown_save_skips_without_touching_cache(tmp_path, monkeypatch):
    """A second shutdown must leave the active writer's stage alone."""
    cache_dir = str(tmp_path / "model")

    class Engine:
        def save_cache_to_disk(self, path, should_abort=None):
            raise AssertionError("contending save entered the cache writer")

    monkeypatch.setattr(
        runtime_cache, "get_config", lambda: SimpleNamespace(engine=Engine())
    )
    monkeypatch.setattr(runtime_cache, "get_cache_dir", lambda: cache_dir)
    with runtime_cache._exclusive_cache_lock(cache_dir, blocking=False) as acquired:
        assert acquired
        runtime_cache.save_prefix_cache_to_disk(budget_sec=0)
