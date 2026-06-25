# SPDX-License-Identifier: Apache-2.0
"""/metrics surfaces the R15 #300 ``rapid_mlx_kv_cache_dtype`` gauge."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.routes.metrics import _render_kv_cache_dtype_gauge


def _cfg_with_engine(dtype: str) -> SimpleNamespace:
    """Build a minimal cfg-like object that exposes ``engine.scheduler_config.kv_cache_dtype``."""
    sc = SimpleNamespace(kv_cache_dtype=dtype)
    engine = SimpleNamespace(scheduler_config=sc)
    return SimpleNamespace(engine=engine, kv_cache_dtype=None)


def test_gauge_emits_three_series_one_active():
    """One series per dtype label; exactly one is 1, others 0."""
    cfg = _cfg_with_engine("int4")
    lines = _render_kv_cache_dtype_gauge(cfg)
    text = "\n".join(lines)
    assert "# TYPE rapid_mlx_kv_cache_dtype gauge" in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 0' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 0' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 1' in text


def test_gauge_reflects_int8_when_engine_is_int8():
    cfg = _cfg_with_engine("int8")
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 0' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 1' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 0' in text


def test_gauge_falls_back_to_cfg_stash_pre_engine_load():
    """During the engine-load window, ``cfg.kv_cache_dtype`` is the source."""
    cfg = SimpleNamespace(engine=None, kv_cache_dtype="int4")
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 1' in text


def test_gauge_defaults_to_bf16_when_unknown():
    """When nothing is set, emit bf16=1 — the only value that is a no-op
    everywhere, so observability never lies about quantization status."""
    cfg = SimpleNamespace()
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 1' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 0' in text


def test_gauge_engine_bf16_wins_over_stale_stash():
    """codex r1 NIT #2 (paired with BLOCKING #2): when the engine has
    fully loaded and stamped its scheduler_config to ``"bf16"`` (e.g.
    the safelist downgraded int4 → bf16 mid-load), a stale
    ``cfg.kv_cache_dtype="int4"`` from the pre-load stash MUST NOT
    override it. The live engine value is the source of truth once
    it's available — anything else lies to the operator about the
    running configuration.
    """
    sc = SimpleNamespace(kv_cache_dtype="bf16")
    engine = SimpleNamespace(scheduler_config=sc)
    cfg = SimpleNamespace(engine=engine, kv_cache_dtype="int4")  # stale stash
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 1' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 0' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 0' in text


def test_gauge_uses_underscore_scheduler_config_when_public_missing():
    """``BatchedEngine`` stores its config under ``_scheduler_config``
    (private). The gauge must reach that fallback name so a real
    deployment doesn't silently fall through to the stash.
    """
    sc = SimpleNamespace(kv_cache_dtype="int8")
    engine = SimpleNamespace(_scheduler_config=sc)
    cfg = SimpleNamespace(engine=engine, kv_cache_dtype=None)
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 1' in text
