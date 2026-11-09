# SPDX-License-Identifier: Apache-2.0
"""Always-on tiny MLX behavior checks; no downloads, torch, or real weights."""

import asyncio
import threading
import weakref
from contextlib import contextmanager

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)


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
    engine.output_dir = tmp_path / "new" / "staging"
    assert not engine.output_dir.exists()
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
            assert engine.output_dir.is_dir()
            assert generated[0].parent == engine.output_dir
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


# One 8-wide attention block, and two upsampling stages with 8 internal channels.
# The temporal shortcut rearranges inputs in groups of 2*2*2 before repeating;
# like the reference decoder, its input channels must be divisible by eight.
# Four latent channels remain sufficient for the decoder's input projection.
# The fixed ByT5 projection is the largest allocation (~16 MB); no checkpoint
# is fetched.
TRANSFORMER = dict(
    in_channels=9,
    out_channels=4,
    num_attention_heads=1,
    attention_head_dim=8,
    num_layers=1,
    num_refiner_layers=1,
    mlp_ratio=2,
    patch_size=1,
    patch_size_t=1,
    qk_norm="rms_norm",
    text_embed_dim=8,
    text_embed_2_dim=8,
    image_embed_dim=4,
    rope_theta=256,
    rope_axes_dim=[2, 2, 4],
    task_type="t2v",
    use_meanflow=False,
)
VAE_CONFIG = dict(
    latent_channels=4,
    out_channels=3,
    block_out_channels=[8, 8, 8],
    layers_per_block=0,
    spatial_compression_ratio=4,
    temporal_compression_ratio=2,
    upsample_match_channel=True,
    scaling_factor=1.03682,
)


def randomize(model):
    """VAE convolutions initialize to zero; exercise actual signal propagation."""
    from mlx.utils import tree_flatten

    rng = np.random.default_rng(17)
    model.load_weights(
        [
            (name, mx.array(rng.normal(0, 0.1, value.shape).astype(np.float32)))
            for name, value in tree_flatten(model.parameters())
        ]
    )
    return model


def save_component(model, root, key_map=lambda k: k):
    """Serialize the upstream OITHW layout, so the production loader is used."""
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file

    root.mkdir(parents=True, exist_ok=True)
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if value.ndim == 5:
            value = value.transpose(0, 4, 1, 2, 3)
        weights[key_map(name)] = np.asarray(value.astype(mx.float32)).copy()
    save_file(weights, root / "model.safetensors")


def test_transformer_masks_padding_preserves_dtype_and_cancels():
    from rapid_mlx.video.cancel import VideoCancelToken, VideoGenerationCancelled
    from rapid_mlx.video.hunyuan_video15.transformer import Transformer

    mx.random.seed(17)
    model = Transformer(TRANSFORMER)
    x = mx.random.normal((1, 9, 2, 2, 2))
    t = mx.array([500.0])
    text, glyph = mx.random.normal((1, 3, 8)), mx.random.normal((1, 2, 8))
    mask, glyph_mask = mx.array([[1, 1, 0]]), mx.array([[1, 0]])
    calls = []
    y = model(x, t, text, mask, glyph, glyph_mask, cancel_check=lambda: calls.append(1))
    assert y.shape == (1, 4, 2, 2, 2) and y.dtype == mx.float32
    assert np.isfinite(np.asarray(y)).all() and calls == [1]
    changed_latent = model(x + 1, t, text, mask, glyph, glyph_mask)
    assert not np.allclose(np.asarray(y), np.asarray(changed_latent))
    # Padded text and glyph tokens cannot affect the latent output.
    changed_text = mx.where(mask[..., None] != 0, text, text + 100)
    changed_glyph = mx.where(glyph_mask[..., None] != 0, glyph, glyph + 100)
    padded = model(x, t, changed_text, mask, changed_glyph, glyph_mask)
    np.testing.assert_allclose(np.asarray(y), np.asarray(padded), atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(y), np.asarray(model(x, t, text, mask, glyph, glyph_mask))
    )
    # Explicit image slots and the implicit T2V zero slots are both masked.
    explicit = model(x, t, text, mask, glyph, glyph_mask, mx.zeros((1, 729, 4)))
    np.testing.assert_array_equal(np.asarray(y), np.asarray(explicit))
    token = VideoCancelToken()
    token.cancel()
    with pytest.raises(VideoGenerationCancelled):
        model(x, t, text, mask, glyph, glyph_mask, cancel_check=token.check)
    model.set_dtype(mx.bfloat16)
    bf = model(
        x.astype(mx.bfloat16),
        t,
        text.astype(mx.bfloat16),
        mask,
        glyph.astype(mx.bfloat16),
        glyph_mask,
    )
    assert bf.dtype == mx.bfloat16 and bf.shape == y.shape
    np.testing.assert_allclose(
        np.asarray(bf.astype(mx.float32)), np.asarray(y), atol=0.05, rtol=0.05
    )


@pytest.mark.parametrize(
    "override,match",
    [
        ({"use_meanflow": True}, "non-distilled"),
        ({"task_type": "i2v"}, "non-distilled"),
        ({"qk_norm": "layer_norm"}, "rms_norm"),
    ],
)
def test_transformer_rejects_unsupported_architecture(override, match):
    from rapid_mlx.video.hunyuan_video15.transformer import Transformer

    with pytest.raises(ValueError, match=match):
        Transformer(dict(TRANSFORMER, **override))


def test_rotary_preserves_norm_and_masked_attention_is_zero():
    from rapid_mlx.video.hunyuan_video15.transformer import (
        Attention,
        apply_rotary,
        rotary_embedding,
        timestep_embedding,
    )

    x = mx.random.normal((1, 8, 1, 8))
    rope = rotary_embedding((1, 9, 2, 2, 2), 1, 1, [2, 2, 4], 256)
    rotated = apply_rotary(x, rope)
    np.testing.assert_allclose(
        np.asarray(mx.sum(x * x, axis=-1)),
        np.asarray(mx.sum(rotated * rotated, axis=-1)),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(timestep_embedding(mx.array([0.0]), 8)), [[1, 1, 1, 1, 0, 0, 0, 0]]
    )
    attn = Attention(8, 1, 8)
    h = mx.ones((1, 3, 8))
    y = attn(h, mask=mx.zeros((1, 3), dtype=mx.int32))
    expected = mx.broadcast_to(attn.to_out[0].bias, y.shape)
    np.testing.assert_allclose(np.asarray(y), np.asarray(expected), atol=1e-6)


def test_vae_decode_is_causal_and_single_tile_matches():
    from rapid_mlx.video.cancel import VideoCancelToken, VideoGenerationCancelled
    from rapid_mlx.video.hunyuan_video15.vae import VAE, ResnetBlock

    model = randomize(VAE(VAE_CONFIG))
    assert [
        up.temporal for block in model.decoder.up_blocks for up in block.upsamplers
    ] == [True, False]
    z = mx.random.normal((1, 4, 3, 2, 2))
    checks = []
    decoded = model.decode(z, lambda: checks.append(True))
    assert len(checks) == 3
    assert decoded.shape == (1, 3, 5, 8, 8) and decoded.dtype == mx.float32
    assert np.isfinite(np.asarray(decoded)).all()
    np.testing.assert_allclose(
        model.tiled_decode(z, tile_sample_size=16), np.asarray(decoded), atol=1e-6
    )
    later = mx.concatenate([z[:, :, :1], z[:, :, 1:] + 10], axis=2)
    np.testing.assert_allclose(
        np.asarray(model.decode(later)[:, :, :1]),
        np.asarray(decoded[:, :, :1]),
        atol=1e-5,
    )
    assert not np.allclose(
        np.asarray(model.decode(later)[:, :, 1:]), np.asarray(decoded[:, :, 1:])
    )
    # A differing channel count needs the learned residual projection.
    assert randomize(ResnetBlock(4, 8))(z).shape == (1, 8, 3, 2, 2)
    token = VideoCancelToken()
    token.cancel()
    for decode in (model.decode, model.tiled_decode):
        with pytest.raises(VideoGenerationCancelled):
            decode(z, token.check)

    # Complete the temporal block, then cancel before the spatial-only block.
    token = VideoCancelToken()
    checks = []

    def cancel_after_first_block():
        checks.append(True)
        if len(checks) == 2:
            token.cancel()
        token.check()

    with pytest.raises(VideoGenerationCancelled):
        model.decode(z, cancel_after_first_block)
    assert len(checks) == 2


def test_vae_tiles_blend_constant_signal_and_report_progress(monkeypatch):
    from rapid_mlx.video.hunyuan_video15.vae import VAE

    model = VAE(VAE_CONFIG)

    # A constant decoder is an oracle for overlap placement: no holes or seams,
    # even at the shortened right/bottom tiles. Real convolutions run above.
    def constant(z, check):
        check()
        return mx.full(
            (1, 3, (z.shape[2] - 1) * 2 + 1, z.shape[3] * 4, z.shape[4] * 4), 0.25
        )

    monkeypatch.setattr(model, "decoder", constant)
    events = []
    y = model.tiled_decode(
        mx.zeros((1, 4, 2, 5, 5)),
        on_progress=lambda *a: events.append(a),
        tile_sample_size=16,
    )
    assert y.shape == (1, 3, 3, 20, 20) and y.dtype == np.float32
    np.testing.assert_array_equal(y, np.full(y.shape, 0.25, np.float32))
    assert events == [("decoding", i, 4) for i in range(1, 5)]


@pytest.mark.parametrize(
    "case,match",
    [
        ("empty", "No safetensors"),
        ("shape", "Shape mismatch"),
        ("duplicate", "duplicate"),
    ],
)
def test_loader_rejects_invalid_checkpoints(tmp_path, case, match):
    import mlx.nn as nn
    from safetensors.numpy import save_file

    from rapid_mlx.video.hunyuan_video15.loading import load_component

    if case == "shape":
        save_file({"weight": np.zeros((2, 4), np.float32)}, tmp_path / "a.safetensors")
    elif case == "duplicate":
        for name in ("a", "b"):
            save_file(
                {"weight": np.zeros((2, 3), np.float32)},
                tmp_path / f"{name}.safetensors",
            )
    with pytest.raises(ValueError, match=match):
        load_component(nn.Linear(3, 2), tmp_path, dtype=mx.float32)


def test_loader_filters_encoder_and_transposes_conv_weights(tmp_path):
    from safetensors.numpy import save_file

    from rapid_mlx.video.cancel import VideoCancelToken, VideoGenerationCancelled
    from rapid_mlx.video.hunyuan_video15.loading import load_component
    from rapid_mlx.video.hunyuan_video15.vae import Conv3d

    weight = np.arange(16, dtype=np.float32).reshape(2, 1, 2, 2, 2)
    save_file(
        {
            "decoder.weight": weight,
            "decoder.bias": np.zeros(2, np.float32),
            "encoder.weight": np.ones(1, np.float32),
        },
        tmp_path / "model.safetensors",
    )
    model = Conv3d(1, 2, 2)
    kwargs = dict(
        dtype=mx.float32,
        key_map=lambda k: (
            k.removeprefix("decoder.") if k.startswith("decoder.") else None
        ),
    )
    load_component(model, tmp_path, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(model.weight), weight.transpose(0, 2, 3, 4, 1)
    )
    np.testing.assert_array_equal(
        np.asarray(model(mx.ones((1, 1, 2, 2, 2)))).reshape(-1),
        weight.sum((1, 2, 3, 4)),
    )
    token = VideoCancelToken()
    token.cancel()
    with pytest.raises(VideoGenerationCancelled):
        load_component(model, tmp_path, cancel_check=token.check, **kwargs)


def t5_checkpoint_key(key):
    if key == "wte.weight":
        return "shared.weight"
    if key == "encoder.ln.weight":
        return "encoder.final_layer_norm.weight"
    if key == "encoder.relative_attention_bias.embeddings.weight":
        return "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    key = key.replace("encoder.layers.", "encoder.block.")
    key = key.replace(".ln1.", ".layer.0.layer_norm.").replace(
        ".ln2.", ".layer.1.layer_norm."
    )
    key = key.replace(".attention.", ".layer.0.SelfAttention.")
    key = key.replace(".dense.", ".layer.1.DenseReluDense.")
    for old, new in (
        ("query_proj", "q"),
        ("key_proj", "k"),
        ("value_proj", "v"),
        ("out_proj", "o"),
    ):
        key = key.replace(f".SelfAttention.{old}.", f".SelfAttention.{new}.")
    return key


@pytest.fixture
def tiny_text_checkpoint(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model
    from transformers import AutoTokenizer, ByT5Tokenizer, T5Config

    from rapid_mlx.image.sd35_runtime._vendor.t5 import SD3T5Encoder

    mx.random.seed(17)
    qconfig = dict(
        model_type="qwen2",
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_hidden_layers=3,
        vocab_size=16,
        rms_norm_eps=1e-6,
    )
    qwen = Qwen2Model(ModelArgs.from_dict(qconfig))
    save_component(qwen, tmp_path / "text_encoder", lambda k: "model." + k)
    (tmp_path / "text_encoder/config.json").write_text(json.dumps(qconfig))
    tc = T5Config(
        vocab_size=16,
        d_model=8,
        d_ff=16,
        d_kv=4,
        num_heads=1,
        num_layers=1,
        feed_forward_proj="gated-gelu",
        dropout_rate=0,
    )
    t5 = SD3T5Encoder(tc, low_memory_mode=False)
    save_component(t5, tmp_path / "text_encoder_2", t5_checkpoint_key)
    tc.save_pretrained(tmp_path / "text_encoder_2")
    chats, glyphs, loads = [], [], []

    class ChatTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            chats.append((messages, kwargs))
            # Keep the real fixed 108-token crop, but only four content slots.
            ids = np.ones((1, 112), np.int32)
            ids[0, 108:] = [2, 3, 0, 0]
            valid = np.ones_like(ids)
            valid[0, -2:] = 0
            return dict(input_ids=ids, attention_mask=valid)

    class GlyphTokenizer:
        def __call__(self, glyph, **kwargs):
            glyphs.append((glyph, kwargs))
            return dict(
                input_ids=np.array([[1, 2, 0, 0]], np.int32),
                attention_mask=np.array([[1, 1, 0, 0]], np.int32),
            )

    def local_tokenizer(path, *, local_files_only):
        assert local_files_only
        loads.append(path.name)
        return ChatTokenizer() if path.name == "tokenizer" else GlyphTokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", local_tokenizer)
    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", local_tokenizer)
    return SimpleNamespace(root=tmp_path, chats=chats, glyphs=glyphs, loads=loads)


def test_dual_text_conditioning_crops_templates_and_handles_glyphs(
    tiny_text_checkpoint,
):
    from rapid_mlx.video.hunyuan_video15.text import SYSTEM_MESSAGE, encode_prompts

    f = tiny_text_checkpoint
    prompts = ['fox "hi" “hi” “bye”', ""]
    checks = []
    results = encode_prompts(f.root, prompts, None, lambda: checks.append(True))
    assert len(results) == 2 and checks
    for h, valid, glyph, glyph_valid in results:
        assert h.shape == (1, 4, 8) and h.dtype == mx.bfloat16
        np.testing.assert_array_equal(np.asarray(valid), [[1, 1, 0, 0]])
        assert glyph.dtype == mx.bfloat16
        assert glyph_valid.dtype == mx.int32
        assert np.isfinite(np.asarray(h.astype(mx.float32))).all()
    assert results[0][2].shape == (1, 4, 8)
    np.testing.assert_array_equal(np.asarray(results[0][3]), [[1, 1, 0, 0]])
    assert results[1][2].shape == (1, 256, 8)
    assert not mx.any(results[1][2]).item() and not mx.any(results[1][3]).item()
    assert f.loads == ["tokenizer", "tokenizer_2"]
    assert f.chats[0][0] == [
        dict(role="system", content=SYSTEM_MESSAGE),
        dict(role="user", content=prompts[0]),
    ]
    assert f.chats[1][0][1]["content"] == " "
    assert f.chats[0][1] == dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding="max_length",
        max_length=1108,
        truncation=True,
        return_tensors="np",
    )
    assert f.glyphs == [
        (
            'Text "hi". Text "bye". ',
            dict(
                padding="max_length",
                max_length=256,
                truncation=True,
                add_special_tokens=True,
                return_tensors="np",
            ),
        )
    ]
    repeated = encode_prompts(f.root, prompts, None, lambda: None)
    for before, after in zip(results, repeated):
        for a, b in zip(before, after):
            np.testing.assert_array_equal(
                np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
            )


def test_text_hidden_masks_block_future_and_padding_tokens():
    import mlx.nn as nn
    from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model
    from transformers import T5Config

    from rapid_mlx.image.sd35_runtime._vendor.t5 import SD3T5Encoder
    from rapid_mlx.video.cancel import VideoCancelToken, VideoGenerationCancelled
    from rapid_mlx.video.hunyuan_video15.text import (
        encode_qwen_hidden,
        encode_t5_hidden,
    )

    qwen = Qwen2Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            rms_norm_eps=1e-6,
            vocab_size=16,
        )
    )
    t5 = SD3T5Encoder(
        T5Config(
            vocab_size=16,
            d_model=8,
            d_ff=16,
            d_kv=4,
            num_heads=1,
            num_layers=1,
            feed_forward_proj="gated-gelu",
        ),
        low_memory_mode=False,
    )
    t5.encoder.layers[0].dense.act = nn.gelu_approx
    ids, valid = mx.array([[1, 2, 3, 0]]), mx.array([[1, 1, 1, 0]])
    for model, encode in ((qwen, encode_qwen_hidden), (t5, encode_t5_hidden)):
        h = encode(model, ids, valid)
        changed = encode(model, mx.array([[1, 2, 3, 15]]), valid)
        assert h.shape == (1, 4, 8)
        np.testing.assert_allclose(
            np.asarray(h[:, :3]), np.asarray(changed[:, :3]), atol=1e-6
        )
        token = VideoCancelToken()
        token.cancel()
        with pytest.raises(VideoGenerationCancelled):
            encode(model, ids, valid, token.check)
    h = encode_qwen_hidden(qwen, ids, valid)
    future = encode_qwen_hidden(qwen, mx.array([[1, 2, 15, 0]]), valid)
    np.testing.assert_allclose(
        np.asarray(h[:, :2]), np.asarray(future[:, :2]), atol=1e-6
    )


@pytest.mark.parametrize(
    "enabled,original", [(False, False), (True, False), (True, True)]
)
def test_tiny_pipeline_generates_deterministic_video(
    monkeypatch, tiny_text_checkpoint, enabled, original
):
    from rapid_mlx.video import encoding
    from rapid_mlx.video.hunyuan_video15 import PipelineConfig, backend
    from rapid_mlx.video.hunyuan_video15.transformer import Transformer
    from rapid_mlx.video.hunyuan_video15.vae import VAE

    f = tiny_text_checkpoint
    save_component(Transformer(TRANSFORMER), f.root / "transformer")
    save_component(randomize(VAE(VAE_CONFIG)), f.root / "vae")
    # Public size limits are independently tested on Linux. A synthetic 4x/2x
    # decoder lets this orchestration test use 8x8 output instead of 256x256.
    monkeypatch.setattr(backend, "validate_request", lambda **kw: None)
    encoded = []

    def encode(pixels, path, fps):
        assert pixels.shape == (5, 8, 8, 3) and pixels.dtype == np.uint8 and fps == 24
        encoded.append(pixels.copy())
        path.write_bytes(pixels.tobytes())

    monkeypatch.setattr(encoding, "encode_rgb_video", encode)
    engine = backend.HunyuanVideo15Engine(str(f.root))
    engine._model_path = str(f.root)
    engine.output_dir = f.root / "staging"
    engine.config = PipelineConfig(
        TRANSFORMER,
        VAE_CONFIG,
        dict(shift=5),
        dict(enabled=enabled, guidance_scale=2, use_original_formulation=original),
    )
    events, loaded = [], []
    try:
        for _ in range(2):
            output = f.root / "result.mp4"
            engine.generate_sync(
                output_path=output,
                prompt='fox "hi"',
                negative_prompt="",
                width=8,
                height=8,
                frames=5,
                fps=24,
                steps=2,
                seed=17,
                guidance_scale=None,
                on_progress=lambda *a: events.append(a),
                on_loaded=lambda: loaded.append(True),
            )
            assert output.read_bytes() == encoded[-1].tobytes()
            assert not list(engine.output_dir.iterdir())
        np.testing.assert_array_equal(encoded[0], encoded[1])
        assert loaded == [True, True]
        assert ("generating", 2, 2) in events and events[-1] == ("encoding", 1, 1)
        assert set(engine.timings) == {
            "load_seconds",
            "denoise_seconds",
            "vae_load_seconds",
            "decode_seconds",
            "peak_memory_bytes",
        }
        assert all(value >= 0 for value in engine.timings.values())
    finally:
        asyncio.run(engine.close())


@pytest.mark.parametrize(
    "override,match",
    [
        ({"fps": 12}, "fixed at 24"),
        ({"guidance_scale": float("nan")}, "guidance_scale"),
        ({"guidance_scale": 0}, "guidance_scale"),
        ({"steps": 0}, "steps"),
    ],
)
def test_backend_rejects_invalid_controls_before_loading(
    monkeypatch, tmp_path, override, match
):
    from rapid_mlx.video.hunyuan_video15.backend import HunyuanVideo15Engine

    engine = HunyuanVideo15Engine(str(tmp_path))
    monkeypatch.setattr(
        engine, "_load_sync", lambda *a: pytest.fail("loaded invalid request")
    )
    try:
        with pytest.raises(ValueError, match=match):
            engine.generate_sync(
                output_path=tmp_path / "out.mp4",
                prompt="fox",
                width=256,
                height=256,
                frames=5,
                **override,
            )
        assert not (tmp_path / "out.mp4").exists()
    finally:
        asyncio.run(engine.close())
