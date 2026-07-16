# SPDX-License-Identifier: Apache-2.0
"""#1103 codex BLOCKING-2 / BLOCKING-3: ``--hybrid-cache-entries`` must be
honored by the benchmark path (not only ``serve``), and the test that proves
it must run fully OFFLINE.

The bench ``SchedulerConfig`` assembly reads ``args.hybrid_cache_entries`` (via
``getattr(..., 0)``) at ``cli.py`` and passes it as
``hybrid_cache_entries=...``; the scheduler then forwards it to the memory
cache as ``hybrid_reuse_max_entries``. The flag was originally registered ONLY
on ``serve_parser``, so ``rapid-mlx bench --hybrid-cache-entries N`` was
rejected by argparse and, had a caller reached the config assembly, it would
have silently fallen back to 0.

These tests exercise the real ``main()`` parser AND the real ``bench_command``
plumbing in-process, mocking only the model-loading / disk / network
boundaries, so:

* BLOCKING-2 — we assert the parsed value ACTUALLY arrives at
  ``SchedulerConfig(hybrid_cache_entries=4)``. If the ``cli.py`` bench plumbing
  line is deleted, this test fails (the argparse-only check could not).
* BLOCKING-3 — NO network access happens: the disk/memory checks, the mirror
  pre-fetch (``_ensure_model_downloaded``) and ``mlx_lm.load`` are all mocked,
  and the download gate is skipped under a non-TTY stdin, so a bogus HF-style
  model id never triggers external resolution / 60s retry timeouts.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

import vllm_mlx.cli as cli

# Pre-import ``engine_core`` so its module-level ``SchedulerConfig | None``
# annotation evaluates against the REAL class BEFORE any test patches
# ``vllm_mlx.scheduler.SchedulerConfig``. ``bench_command`` imports it lazily
# (``from .engine_core import ...``); with the module already in ``sys.modules``
# that is a pure name rebind — the class body never re-runs under the patch.
import vllm_mlx.engine_core as _engine_core  # noqa: E402,F401


class _StopBenchError(Exception):
    """Sentinel raised right after SchedulerConfig is built to short-circuit
    the benchmark before any engine boot."""


def _run_bench_capturing_scheduler_config(argv: list[str]) -> dict:
    """Drive ``cli.main()`` for a ``bench`` invocation with every model-loading
    and network boundary mocked, capturing the kwargs passed to
    ``SchedulerConfig``. Returns those kwargs.

    Raises ``AssertionError`` if ``SchedulerConfig`` was never constructed
    (i.e. the flow died before the plumbing under test).
    """
    captured: dict = {}

    def _fake_scheduler_config(*args, **kwargs):
        assert not args, "bench builds SchedulerConfig keyword-only"
        captured.update(kwargs)
        # Stop before EngineConfig / engine boot — the value has arrived, which
        # is all this test needs to prove.
        raise _StopBenchError

    with (
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        # ``bench_command`` does ``from mlx_lm import load`` — patch at source.
        mock.patch("mlx_lm.load", return_value=(object(), object())),
        # ``bench_command`` does ``from .scheduler import SchedulerConfig`` —
        # patch on the scheduler module so the local import binds the mock.
        mock.patch("vllm_mlx.scheduler.SchedulerConfig", _fake_scheduler_config),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        # Guarantee the non-interactive path so the download gate never
        # touches the HF API even if a runner attaches a TTY.
        mock.patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises((_StopBenchError, SystemExit)),
    ):
        cli.main()

    assert captured, (
        "SchedulerConfig was never constructed — the bench flow died before "
        "reaching the hybrid-cache plumbing under test"
    )
    return captured


def test_bench_hybrid_cache_entries_reaches_scheduler_config():
    """BLOCKING-2: the parsed ``--hybrid-cache-entries 4`` must actually arrive
    at ``SchedulerConfig(hybrid_cache_entries=4)`` — guarding the bench plumbing
    line, not merely that argparse accepted the flag."""
    captured = _run_bench_capturing_scheduler_config(
        [
            "bench",
            "does-not-exist/definitely-not-a-real-model",
            "--hybrid-cache-entries",
            "4",
            "--num-prompts",
            "1",
            "--max-tokens",
            "1",
        ]
    )
    assert captured.get("hybrid_cache_entries") == 4


def test_bench_hybrid_cache_entries_defaults_to_zero():
    """Without the flag the bench path must pass ``hybrid_cache_entries=0`` —
    the #1075 drop-at-store default (opt-in stays off)."""
    captured = _run_bench_capturing_scheduler_config(
        [
            "bench",
            "does-not-exist/definitely-not-a-real-model",
            "--num-prompts",
            "1",
            "--max-tokens",
            "1",
        ]
    )
    assert captured.get("hybrid_cache_entries") == 0


def test_bench_rejects_negative_hybrid_cache_entries():
    """A negative value is nonsensical (the bound is ``>= 0``). argparse PARSES
    it unchanged (``type=int`` does not clamp), so ``--hybrid-cache-entries -1``
    flows through the bench plumbing into ``SchedulerConfig(hybrid_cache_entries
    =-1)`` — and, at engine boot, the scheduler feeds that straight into
    ``MemoryCacheConfig(hybrid_reuse_max_entries=...)`` (scheduler.py:1976),
    whose ``__post_init__`` rejects ``< 0`` (memory_cache.py:844).

    A positive-value assertion cannot prove the rejection — it would stay green
    even if the ``>= 0`` guard were deleted. So here we (1) drive the real CLI /
    ``SchedulerConfig`` assembly to confirm ``-1`` arrives UNCLAMPED, then
    (2) reproduce the exact scheduler → cache mapping and assert the real
    ``MemoryCacheConfig`` construction raises. Fully offline: no engine boot,
    no network (the helper mocks every model-loading / disk / network
    boundary)."""
    from vllm_mlx.memory_cache import MemoryCacheConfig

    captured = _run_bench_capturing_scheduler_config(
        [
            "bench",
            "does-not-exist/definitely-not-a-real-model",
            "--hybrid-cache-entries",
            "-1",
            "--num-prompts",
            "1",
            "--max-tokens",
            "1",
        ]
    )
    # (1) The negative value reaches SchedulerConfig UNCLAMPED (nothing between
    # argparse and the scheduler config sanitizes it away).
    assert captured.get("hybrid_cache_entries") == -1

    # (2) Feeding that scheduler value into MemoryCacheConfig exactly as
    # scheduler.py:1976 does must raise — this is the guard the bench flow
    # would trip at engine boot.
    with pytest.raises(ValueError, match=r"hybrid_reuse_max_entries must be >= 0"):
        MemoryCacheConfig(hybrid_reuse_max_entries=captured["hybrid_cache_entries"])
