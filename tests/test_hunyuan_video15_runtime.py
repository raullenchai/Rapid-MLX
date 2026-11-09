# SPDX-License-Identifier: Apache-2.0
"""Opt-in MLX lifecycle/loader tests; no real weights or torch imports."""

import asyncio
import os
import threading
import weakref
from contextlib import contextmanager

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("HUNYUAN_VIDEO15_REFERENCE_PYTHON"),
    reason="Opt-in MLX runtime checks (Metal unavailable in some sandboxes)",
)


def test_sharded_loading_casts_and_rejects_missing_or_extra(tmp_path):
    import json

    import mlx.core as mx
    import mlx.nn as nn
    from safetensors.numpy import save_file

    from rapid_mlx.video.hunyuan_video15.loading import load_component

    model = nn.Linear(3, 2)
    save_file({"weight": np.ones((2, 3), np.float32)}, tmp_path / "part1.safetensors")
    save_file({"bias": np.ones((2,), np.float32) * 2}, tmp_path / "part2.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"weight": "part1.safetensors", "bias": "part2.safetensors"}}
        )
    )
    load_component(model, tmp_path, dtype=mx.bfloat16)
    assert model.weight.dtype == model.bias.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(model(mx.ones((1, 3), dtype=mx.bfloat16)).astype(mx.float32)),
        [[5, 5]],
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "part1.safetensors"}})
    )
    with pytest.raises(ValueError, match="Missing"):
        load_component(nn.Linear(3, 2), tmp_path, dtype=mx.float32)
    save_file({"surprise": np.ones((1,), np.float32)}, tmp_path / "part1.safetensors")
    with pytest.raises(ValueError, match="Unexpected"):
        load_component(nn.Linear(3, 2), tmp_path, dtype=mx.float32)


@pytest.mark.parametrize("cancel_stage", [None, "generating", "encoding"])
def test_worker_releases_models_bounds_cache_and_cleans_output(
    monkeypatch, tmp_path, cancel_stage
):
    import mlx.core as mx

    from rapid_mlx import _mlx_cache_limit
    from rapid_mlx.video import encoding
    from rapid_mlx.video.cancel import VideoCancelToken, VideoGenerationCancelled
    from rapid_mlx.video.hunyuan_video15 import (
        PipelineConfig,
        loading,
        text,
        transformer,
        vae,
    )
    from rapid_mlx.video.hunyuan_video15.backend import HunyuanVideo15Engine

    threads = []
    dit_ref = None
    cache_active = False
    events = []

    def record(stage):
        threads.append(threading.get_ident())
        events.append(stage)

    @contextmanager
    def bounded(bound):
        nonlocal cache_active
        assert bound == 1024**3
        cache_active = True
        try:
            yield
        finally:
            cache_active = False

    monkeypatch.setattr(_mlx_cache_limit, "bounded_cache_limit", bounded)

    def encode(root, prompts, config, check):
        assert cache_active
        check()
        record("prompt")
        return [
            [mx.zeros((1, 1, 2)), mx.ones((1, 1)), mx.zeros((1, 1, 2)), mx.ones((1, 1))]
            for _ in prompts
        ]

    monkeypatch.setattr(text, "encode_prompts", encode)

    class FakeTransformer:
        def __init__(self, config):
            nonlocal dit_ref
            record("dit_init")
            dit_ref = weakref.ref(self)

        def __call__(self, x, t, *embedding, cancel_check):
            assert cache_active
            record("denoise")
            cancel_check()
            return mx.zeros((1, 4, *x.shape[2:]), dtype=mx.bfloat16)

    class FakeVAE:
        def __init__(self, config):
            record("vae_init")
            assert dit_ref() is None, "DiT must be released before VAE loading"

        def tiled_decode(self, latent, check, on_progress):
            assert cache_active
            check()
            record("decode")
            return np.zeros((1, 3, 5, 256, 256), np.float32)

    monkeypatch.setattr(transformer, "Transformer", FakeTransformer)
    monkeypatch.setattr(vae, "VAE", FakeVAE)
    monkeypatch.setattr(loading, "load_component", lambda *a, **k: record("load"))
    generated = []

    def encode_video(pixels, path, fps):
        assert pixels.shape == (5, 256, 256, 3) and fps == 24
        record("mp4")
        path.write_bytes(b"mp4")
        generated.append(path)

    monkeypatch.setattr(encoding, "encode_rgb_video", encode_video)
    token = VideoCancelToken()

    def progress(stage, done, total):
        if stage == cancel_stage:
            token.cancel()

    engine = HunyuanVideo15Engine(str(tmp_path))
    engine._model_path = str(tmp_path)
    engine.config = PipelineConfig(
        {},
        dict(
            latent_channels=4,
            spatial_compression_ratio=16,
            temporal_compression_ratio=4,
            scaling_factor=1.03682,
        ),
        dict(shift=5),
        dict(guidance_scale=6),
    )
    output = tmp_path / "output.mp4"
    try:
        kwargs = dict(
            output_path=output,
            prompt="fox",
            width=256,
            height=256,
            frames=5,
            fps=24,
            steps=2,
            seed=42,
            cancel_token=token,
            on_progress=progress,
            on_loaded=lambda: record("loaded"),
        )
        if cancel_stage:
            with pytest.raises(VideoGenerationCancelled):
                engine.generate_sync(**kwargs)
            assert not output.exists()
        else:
            engine.generate_sync(**kwargs)
            assert output.read_bytes() == b"mp4"
            assert (
                events.index("prompt")
                < events.index("dit_init")
                < events.index("loaded")
                < events.index("denoise")
                < events.index("vae_init")
                < events.index("decode")
                < events.index("mp4")
            )
            assert engine.timings["load_seconds"] >= 0
            assert engine.timings["decode_seconds"] >= 0
        assert not cache_active
        assert all(t == engine._worker.ident for t in threads)
        assert all(not p.exists() for p in generated)
    finally:
        asyncio.run(engine.close())
