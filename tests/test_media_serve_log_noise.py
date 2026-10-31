import asyncio
import logging
from types import SimpleNamespace

import pytest


def test_mamba_noop_is_debug(monkeypatch, caplog):
    from rapid_mlx.utils import mamba_cache

    monkeypatch.setattr(mamba_cache, "_patched", False)
    with caplog.at_level(logging.DEBUG, logger=mamba_cache.__name__):
        mamba_cache.ensure_mamba_support()
    record = next(
        r for r in caplog.records if "Skipping _make_cache patch" in r.message
    )
    assert record.levelno == logging.DEBUG


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
