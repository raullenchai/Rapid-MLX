# SPDX-License-Identifier: Apache-2.0
"""#2356 — a normal ``rapid-mlx serve`` must not bury the Ready/Connect block.

A first ``serve <model> --port N`` interleaved the user-facing startup card
with low-level INFO trace (per-cache auto-load counts, radix rebuild) and
printed the KV-cache dtype decision twice, so the terminal never settled on a
clear final next step. These tests pin the CALM startup-shape contract:

  * The deferred post-ready prefix-cache auto-load and the radix rebuild
    (which run on a background task scheduled AFTER the Ready banner) must log
    at DEBUG, not INFO — otherwise they scroll the Ready/Connect block off the
    terminal after it prints.
  * The KV-cache dtype resolution must be surfaced exactly once
    (``log_kv_cache_decision`` emits a single line, not a log line *and* a
    duplicate ``print``).

These are unit-level: they stub the engine/config/disk surfaces and assert on
the captured log records, so no model weights need to boot.
"""

from __future__ import annotations

import io
import logging
import threading
from contextlib import redirect_stdout


def _stub_config_and_cache_dir(monkeypatch, *, entries: int = 1):
    """Wire the module's ``get_config``/``get_cache_dir`` to stub surfaces so
    ``load_prefix_cache_from_disk`` runs without touching the real cache dir.

    Mirrors the stubbing already used by
    ``tests/test_deferred_prefix_cache_load.py`` so the two files agree on the
    injection seams. ``entries`` is the value ``load_cache_from_disk`` returns
    (0 selects the "No ... found on disk" branch).
    """
    from vllm_mlx.runtime import cache as cache_mod

    class _StubEngine:
        def load_cache_from_disk(self, _path, *, protected_import):
            return entries

    class _Config:
        engine = _StubEngine()

    monkeypatch.setattr(cache_mod, "get_config", lambda: _Config())
    monkeypatch.setattr(cache_mod, "get_cache_dir", lambda: "/tmp/calm-cache")
    return cache_mod


def _info_records(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


def test_post_ready_prefix_cache_load_lines_are_debug_not_info(monkeypatch, caplog):
    """The background prefix-cache auto-load must NOT emit at INFO.

    #2356: ``server._deferred_load_prefix_cache`` schedules this load AFTER
    ``_cfg.ready = True`` and after the Ready banner prints. At INFO, the
    ``[lifespan] Loading ...`` / ``Loaded N`` / ``No ... found`` lines landed
    after the connect card and pushed it off the terminal. They are warm-start
    detail, so they move to DEBUG.
    """
    from vllm_mlx.runtime import cache as cache_mod

    cache_mod = _stub_config_and_cache_dir(monkeypatch)

    with caplog.at_level(logging.INFO):
        cache_mod.load_prefix_cache_from_disk()

    # Default INFO output must be free of the post-ready per-boot chatter.
    assert not any(
        "prefix cache" in m or "prefix cache entries" in m
        for m in _info_records(caplog)
    ), (
        "post-ready prefix-cache lines must be silent at INFO (#2356) but were "
        f"logged:\n{caplog.text}"
    )

    # The detail is reachable at DEBUG (captured separately).
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        cache_mod.load_prefix_cache_from_disk()
    assert "Loading prefix cache from /tmp/calm-cache" in caplog.text
    assert "Loaded 1 prefix cache entries" in caplog.text


def test_post_ready_prefix_cache_no_entries_branch_is_debug(monkeypatch, caplog):
    """The ``No prefix cache entries found on disk`` branch is DEBUG too."""
    from vllm_mlx.runtime import cache as cache_mod

    cache_mod = _stub_config_and_cache_dir(monkeypatch, entries=0)

    with caplog.at_level(logging.INFO):
        cache_mod.load_prefix_cache_from_disk()
    assert not any("No prefix cache entries found" in m for m in _info_records(caplog))

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        cache_mod.load_prefix_cache_from_disk()
    assert "No prefix cache entries found on disk" in caplog.text


def test_post_ready_radix_rebuild_is_debug_not_info(caplog):
    """The radix-index rebuild (post-ready background task) must log at DEBUG.

    Same class of offender: ``_load_radix_index_after_cache`` runs on the
    deferred post-ready task, so its ``[radix] rebuilt index ...`` line used to
    bury the Ready banner.
    """
    from vllm_mlx.runtime import cache as cache_mod

    class _StubRadix:
        def load(self, _path):
            return None  # falsy → rebuild path fires

        def rebuild_from_keys(self, _keys):
            pass

    class _StubCache:
        _radix_index = _StubRadix()
        _lock = threading.Lock()
        _entries = {"k1": object()}

    class _StubScheduler:
        memory_aware_cache = _StubCache()

    class _StubEngine:
        scheduler = _StubScheduler()

    with caplog.at_level(logging.INFO):
        cache_mod._load_radix_index_after_cache(_StubEngine(), "/tmp/calm-cache")
    assert "rebuilt index" not in caplog.text, (
        f"radix rebuild must be silent at INFO (#2356) but was logged:\n{caplog.text}"
    )

    with caplog.at_level(logging.DEBUG):
        cache_mod._load_radix_index_after_cache(_StubEngine(), "/tmp/calm-cache")
    assert "rebuilt index" in caplog.text


def test_kv_cache_dtype_decision_logged_exactly_once(caplog):
    """The KV-cache dtype decision surfaces exactly once.

    #2356: ``log_kv_cache_decision`` used to emit the message twice — once as
    ``logger.info`` (stderr) and again via ``print`` (stdout) — so a foreground
    serve showed two identical dtype lines. It must now produce a single
    emission (one INFO log record, nothing printed to stdout).
    """
    from vllm_mlx.kv_cache_dtype import (
        KVCacheDtypeDecision,
        log_kv_cache_decision,
    )

    decision = KVCacheDtypeDecision(
        dtype="bf16",
        reason="bf16 selected (no QuantizedKVCache wrap)",
        downgraded=False,
        requested="bf16",
    )

    buf = io.StringIO()
    with caplog.at_level(logging.INFO), redirect_stdout(buf):
        log_kv_cache_decision(decision, model_name="qwen3.5-4b")

    matches = [r for r in caplog.records if "KV cache dtype:" in r.getMessage()]
    assert len(matches) == 1, (
        f"KV-cache dtype decision must be logged exactly once (#2356); got "
        f"{len(matches)} INFO records:\n"
        + "\n".join(r.getMessage() for r in caplog.records)
    )
    assert "KV cache dtype:" in matches[0].getMessage()
    # The decision must not be double-emitted via a raw stdout print.
    assert "KV cache dtype:" not in buf.getvalue(), (
        "KV-cache dtype decision must not also be printed to stdout — that "
        "doubles it on the terminal (#2356)."
    )
