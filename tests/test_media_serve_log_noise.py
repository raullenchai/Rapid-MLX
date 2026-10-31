import asyncio
import logging
from types import SimpleNamespace

import pytest


@pytest.mark.requires_mlx
def test_mamba_noop_is_debug(monkeypatch, caplog):
    from rapid_mlx.utils import mamba_cache

    monkeypatch.setattr(mamba_cache, "_patched", False)
    with caplog.at_level(logging.DEBUG, logger=mamba_cache.__name__):
        mamba_cache.ensure_mamba_support()
    record = next(
        r for r in caplog.records if "Skipping _make_cache patch" in r.message
    )
    assert record.levelno == logging.DEBUG


@pytest.mark.requires_mlx
def test_gdn_install_is_debug(monkeypatch, caplog):
    from mlx_lm.models import gated_delta

    from rapid_mlx import gdn_prefill

    monkeypatch.setattr(gdn_prefill, "_installed", False)
    monkeypatch.delenv("RAPID_MLX_GDN_PREFILL", raising=False)
    monkeypatch.setattr(gdn_prefill.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(gated_delta, "gated_delta_kernel", lambda *args: None)
    monkeypatch.setattr(gdn_prefill.mx, "zeros", lambda *args, **kwargs: None)
    monkeypatch.setattr(gdn_prefill.mx, "eval", lambda *args: None)
    monkeypatch.setattr(
        gdn_prefill, "gated_delta_blocked_seq", lambda *args: (None, None)
    )
    monkeypatch.setattr(gdn_prefill, "_original_kernel", None)
    with caplog.at_level(logging.DEBUG, logger=gdn_prefill.__name__):
        assert gdn_prefill.install()
    record = next(
        r
        for r in caplog.records
        if "blocked-seq GDN prefill kernel installed" in r.message
    )
    assert record.levelno == logging.DEBUG


@pytest.mark.parametrize(
    "alias,media", [("flux2-klein-4b", True), ("qwen3.5-4b-4bit", False)]
)
def test_alias_resolution_log(alias, media, caplog):
    from rapid_mlx import model_auto_config

    model_auto_config._reset_resolution_log_cache()
    with caplog.at_level(logging.INFO, logger=model_auto_config.__name__):
        assert model_auto_config.detect_model_config(alias) is not None
    message = next(
        r.message for r in caplog.records if "Resolved alias profile" in r.message
    )
    assert ("tool_call_parser=" not in message) == media
    if media:
        assert "modality=image-gen" in message
    model_auto_config._reset_resolution_log_cache()


@pytest.mark.parametrize("media", [True, False])
def test_runtime_summary(media, capsys):
    from rapid_mlx.cli import _print_text_lane_runtime_summary

    args = SimpleNamespace(
        stream_interval=3,
        use_paged_cache=False,
        no_memory_aware_cache=True,
        prefix_cache_size=12,
    )
    _print_text_lane_runtime_summary(args, True, media)
    output = capsys.readouterr().out
    assert output == (
        "" if media else "Stream interval: 3 tokens\nPrefix cache: max_entries=12\n"
    )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (
            {"use_paged_cache": True},
            "Paged cache requested: block_size=32, max_blocks=128 "
            "(validated at engine startup)\n",
        ),
        ({}, "Memory-aware cache: 512MB (index=radix)\n"),
        (
            {"cache_memory_mb": None},
            "Memory-aware cache: 20% of RAM (index=radix)\n",
        ),
        (
            {"kv_cache_turboquant": "k8v4"},
            "Memory-aware cache: 512MB (index=radix)\n"
            "TurboQuant K8V4: K=8-bit Walsh-Hadamard, V=4-bit Lloyd-Max, "
            "group_size=64\n",
        ),
        (
            {"kv_cache_turboquant": "mse", "kv_cache_turboquant_bits": 3},
            "Memory-aware cache: 512MB (index=radix)\n"
            "TurboQuant V-cache (mse): 3-bit, group_size=64 (K stays FP16)\n",
        ),
        (
            {"kv_cache_turboquant": "mse"},
            "Memory-aware cache: 512MB (index=radix)\n"
            "TurboQuant V-cache (mse): auto-bit, group_size=64 (K stays FP16)\n",
        ),
        (
            {"kv_cache_quantization": True},
            "Memory-aware cache: 512MB (index=radix)\n"
            "KV cache quantization: 4-bit, group_size=128\n",
        ),
    ],
)
def test_text_lane_runtime_summary_cache_options(
    overrides, expected, monkeypatch, capsys
):
    from rapid_mlx import cli

    args = SimpleNamespace(
        stream_interval=3,
        use_paged_cache=False,
        paged_cache_block_size=32,
        max_cache_blocks=128,
        no_memory_aware_cache=False,
        cache_memory_mb=512,
        kv_cache_turboquant=None,
        kv_cache_turboquant_bits=None,
        kv_cache_turboquant_group_size=64,
        kv_cache_quantization=False,
        kv_cache_quantization_bits=4,
        kv_cache_quantization_group_size=128,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    if args.cache_memory_mb is None:
        monkeypatch.setattr(cli, "_cache_memory_percent", lambda args: 0.2)

    cli._print_text_lane_runtime_summary(args, True, False)
    assert capsys.readouterr().out == "Stream interval: 3 tokens\n" + expected

    cli._print_text_lane_runtime_summary(args, True, True)
    assert capsys.readouterr().out == ""


def test_media_engine_warmup_logs(caplog):
    from rapid_mlx.runtime.image_lane import ImageEngine
    from rapid_mlx.runtime.video_lane import VideoEngine
    from rapid_mlx.server import _is_generative_media_engine, _warmup_primary_engine

    for cls in (ImageEngine, VideoEngine):
        engine = cls.__new__(cls)
        assert _is_generative_media_engine(engine)
        with caplog.at_level(logging.INFO, logger="rapid_mlx.server"):
            asyncio.run(_warmup_primary_engine(engine))
        assert "Warming up" not in caplog.text
        assert "Warmup complete" not in caplog.text
    assert not _is_generative_media_engine(object())
