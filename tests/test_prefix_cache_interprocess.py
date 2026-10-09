"""Automatic prefix-cache shutdown saves sharing one HOME."""

from __future__ import annotations

import multiprocessing
import shutil
from pathlib import Path
from types import SimpleNamespace


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
