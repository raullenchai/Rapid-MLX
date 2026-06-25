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


def test_gauge_derives_dtype_from_legacy_quantization_fields_int4():
    """codex r2 BLOCKING #2: a programmatic caller that only set the
    pre-existing legacy fields (``kv_cache_quantization=True`` +
    ``kv_cache_quantization_bits=4``) without touching the new
    ``kv_cache_dtype`` field would have the gauge mis-report ``bf16``
    because ``SchedulerConfig.kv_cache_dtype`` now defaults to
    ``"bf16"``. The gauge must derive the effective dtype from the
    legacy bits in that case.
    """
    sc = SimpleNamespace(
        kv_cache_dtype="bf16",  # unmodified default
        kv_cache_quantization=True,
        kv_cache_quantization_bits=4,
    )
    engine = SimpleNamespace(scheduler_config=sc)
    cfg = SimpleNamespace(engine=engine, kv_cache_dtype=None)
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 1' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 0' in text


def test_gauge_derives_dtype_from_legacy_quantization_fields_int8():
    """Same fix as int4 above but for bits=8."""
    sc = SimpleNamespace(
        kv_cache_dtype="bf16",
        kv_cache_quantization=True,
        kv_cache_quantization_bits=8,
    )
    engine = SimpleNamespace(scheduler_config=sc)
    cfg = SimpleNamespace(engine=engine, kv_cache_dtype=None)
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 1' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 0' in text


def test_gauge_legacy_fallback_does_not_override_explicit_int8_dtype():
    """The legacy fallback only fires when ``kv_cache_dtype`` is at
    the unmodified ``"bf16"`` default. An explicit ``"int8"`` from the
    canonical knob must win, even if legacy fields are also set (bits=4
    is illegal here per codex r2 BLOCKING #1, but the gauge defends).
    """
    sc = SimpleNamespace(
        kv_cache_dtype="int8",  # canonical knob wins
        kv_cache_quantization=True,
        kv_cache_quantization_bits=8,
    )
    engine = SimpleNamespace(scheduler_config=sc)
    cfg = SimpleNamespace(engine=engine, kv_cache_dtype=None)
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 1' in text


def test_gauge_legacy_fallback_leaves_bf16_when_bits_out_of_range():
    """If a programmatic caller landed legacy fields with an out-of-range
    bits value, the gauge must not guess a label — the CLI rejects this
    case (codex r2 BLOCKING #1) but the metrics endpoint stays honest
    and reports bf16 (the default) rather than fabricate a series."""
    sc = SimpleNamespace(
        kv_cache_dtype="bf16",
        kv_cache_quantization=True,
        kv_cache_quantization_bits=3,  # illegal
    )
    engine = SimpleNamespace(scheduler_config=sc)
    cfg = SimpleNamespace(engine=engine, kv_cache_dtype=None)
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 1' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 0' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 0' in text


def test_gauge_unknown_dtype_falls_back_to_bf16():
    """codex r3 BLOCKING: a typo / future dtype string / stale field
    value not in {"bf16","int8","int4"} would render every series at
    0 (none active), which violates the gauge's "exactly one is 1"
    contract and looks like a broken metric on dashboards. Validate
    against the known set and fall back to ``"bf16"`` for unknowns."""
    sc = SimpleNamespace(kv_cache_dtype="fp8")  # not in known set
    engine = SimpleNamespace(scheduler_config=sc)
    cfg = SimpleNamespace(engine=engine, kv_cache_dtype=None)
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 1' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 0' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 0' in text


def test_gauge_unknown_stash_dtype_falls_back_to_bf16():
    """Same fix path as above, but via the pre-load stash code branch
    (no engine yet). An unknown stash value must not silently emit
    zero across all three series."""
    cfg = SimpleNamespace(engine=None, kv_cache_dtype="int2")  # not in known set
    text = "\n".join(_render_kv_cache_dtype_gauge(cfg))
    assert 'rapid_mlx_kv_cache_dtype{dtype="bf16"} 1' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int4"} 0' in text
    assert 'rapid_mlx_kv_cache_dtype{dtype="int8"} 0' in text
