"""Independent float64 transliteration of the official reference implementation.

Functions deliberately use checkpoint-shaped operations, not runtime helpers.
The operator's real_parity.py loads this file under _minimax_numpy_reference,
which exposes only NumPy helpers and never imports MLX or the downloaded source.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict

import numpy as np
import pytest

if __name__ != "_minimax_numpy_reference":
    if sys.platform != "darwin":
        pytest.skip("MLX parity requires Darwin", allow_module_level=True)
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm.models.minimax_h3")


def silu(x):
    return x / (1 + np.exp(-x))


def linear(x, weights, prefix):
    y = x @ weights[prefix + ".weight"].T
    return y + weights.get(prefix + ".bias", 0)


def rms(x, weights, prefix, eps):
    return (
        x
        / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        * weights.get(prefix + ".weight", 1)
    )


def layer_norm(x, weights, prefix, eps):
    centered = x - x.mean(axis=-1, keepdims=True)
    return (
        centered
        / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
        * weights[prefix + ".weight"]
        + weights[prefix + ".bias"]
    )


def image_rope(position_ids, dim, theta=100.0):
    # MiniMaxH3ImageRotaryPosEmbed.forward; num_axes=2.
    inv_freq = 1.0 / theta ** np.arange(0, 1, 4 / dim, dtype=np.float64)
    angles = 2.0 * math.pi * position_ids[:, :, :, None] * inv_freq[None, None, None, :]
    angles = np.tile(angles.reshape(*angles.shape[:2], -1), (1, 1, 2))[:, :, None]
    return np.cos(angles), np.sin(angles)


def dit_rope(position_ids, config):
    # MiniMaxH3ImageRotaryPosEmbed.forward in transformer_minimax_h3_image.py.
    freq_dim = config["rope_freq_dim"]
    inv_freq = 1.0 / config["rope_theta"] ** (
        np.arange(0, 2 * freq_dim, 2, dtype=np.float64) / (2 * freq_dim)
    )
    frequencies = position_ids[..., None] * inv_freq
    frequencies = np.concatenate(
        [frequencies[:, 0], frequencies[:, 1], frequencies[:, 2]], axis=-1
    )
    frequencies = np.concatenate([frequencies, frequencies], axis=-1)
    return np.cos(frequencies)[None, :, None], np.sin(frequencies)[None, :, None]


def apply_rope(x, rotary):
    cos, sin = rotary
    size = cos.shape[-1]
    rotating, rest = x[..., :size], x[..., size:]
    first, second = np.split(rotating, 2, axis=-1)
    return np.concatenate(
        [rotating * cos + np.concatenate([-second, first], axis=-1) * sin, rest],
        axis=-1,
    )


def attention(x, weights, prefix, heads, eps, rotary=None):
    # MiniMaxH3ImageAttnProcessor.__call__ (both DiT and ViT).
    b, n, _ = x.shape
    q, k, v = [
        linear(x, weights, prefix + name).reshape(b, n, heads, -1)
        for name in (".to_q", ".to_k", ".to_v")
    ]
    q = rms(q, weights, prefix + ".norm_q", eps)
    k = rms(k, weights, prefix + ".norm_k", eps)
    if rotary is not None:
        q, k = apply_rope(q, rotary), apply_rope(k, rotary)
    scores = np.einsum("bnhd,bmhd->bhnm", q, k) / np.sqrt(q.shape[-1])
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    scores /= scores.sum(axis=-1, keepdims=True)
    output = np.einsum("bhnm,bmhd->bnhd", scores, v).reshape(b, n, -1)
    return linear(output, weights, prefix + ".to_out.0")


def feed_forward(x, weights, prefix):
    hidden, gate = np.split(linear(x, weights, prefix + ".net.0.proj"), 2, axis=-1)
    return linear(hidden * silu(gate), weights, prefix + ".net.2")


def decoder2d(z, weights, config, prefix="decoder"):
    # MiniMaxH3ImageViTDecoder2d.forward, line-by-line token and pixel order.
    b, c, h, w = z.shape
    x = linear(
        z.transpose(0, 2, 3, 1).reshape(b, h * w, c), weights, prefix + ".proj_in"
    )
    count = x.shape[1]
    registers = np.broadcast_to(
        weights[prefix + ".register_tokens"],
        (b, config["decoder_num_register_tokens"], x.shape[-1]),
    )
    x = np.concatenate([x, registers, np.zeros_like(x[:, :1])], axis=1)
    grids = [
        2.0 * (np.arange(0.5, size, dtype=np.float64) / size) - 1.0 for size in (h, w)
    ]
    positions = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 2)
    positions = np.broadcast_to(positions[None], (b, count, 2))
    suffix = np.zeros((b, config["decoder_num_register_tokens"] + 1, 2))
    rotary = image_rope(
        np.concatenate([positions, suffix], axis=1),
        int(config["decoder_attention_head_dim"] * config["decoder_rope_dim_ratio"]),
        config["decoder_rope_theta"],
    )
    eps = config["decoder_norm_eps"]
    for i in range(config["decoder_num_layers"]):
        p = f"{prefix}.transformer_blocks.{i}"
        x = (
            x
            + attention(
                rms(x, weights, p + ".norm1", eps),
                weights,
                p + ".attn",
                config["decoder_num_attention_heads"],
                eps,
                rotary,
            )
            * weights[p + ".scale1"]
        )
        x = (
            x
            + feed_forward(rms(x, weights, p + ".norm2", eps), weights, p + ".ff")
            * weights[p + ".scale2"]
        )
    x = linear(
        layer_norm(x, weights, prefix + ".norm_out", eps), weights, prefix + ".proj_out"
    )[:, :count]
    patch = math.prod(config["spatial_downsample_factors"])
    x = x.reshape(b, h, w, config["out_channels"], patch, patch)
    return x.transpose(0, 3, 1, 4, 2, 5).reshape(
        b, config["out_channels"], h * patch, w * patch
    )


def conv_single_frame(x, weights, prefix, stride=1, padding=0, asymmetric=False):
    # MiniMaxH3ImageCausalConv3d.forward: a single frame has only the last
    # temporal kernel slice nonzero, including at temporal stride 2.
    kernel = weights[prefix + ".weight"]
    kernel = kernel[:, -1] if kernel.ndim == 5 else kernel
    # Sanitized MLX layout is O,H,W,I; convert to the reference O,I,H,W.
    kernel = kernel.transpose(0, 3, 1, 2)
    if padding:
        x = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="reflect"
        )
    if asymmetric:
        x = np.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        x, kernel.shape[-2:], axis=(-2, -1)
    )
    windows = windows[:, :, ::stride, ::stride]
    # Keep the im2col temporary bounded for the real-weight encoder check.
    result = np.empty((x.shape[0], kernel.shape[0], windows.shape[2], windows.shape[3]))
    matrix = kernel.reshape(kernel.shape[0], -1).T
    for row in range(windows.shape[2]):
        columns = (
            windows[:, :, row].transpose(0, 2, 1, 3, 4).reshape(-1, matrix.shape[0])
        )
        result[:, :, row] = (
            (columns @ matrix)
            .reshape(x.shape[0], windows.shape[3], -1)
            .transpose(0, 2, 1)
        )
    return result + weights[prefix + ".bias"][None, :, None, None]


def group_norm(x, weights, prefix, groups, eps):
    # MiniMaxH3ImageGroupNorm.forward; no time dimension mixes statistics.
    b, c, h, w = x.shape
    grouped = x.reshape(b, groups, -1)
    centered = grouped - grouped.mean(axis=-1, keepdims=True)
    norm = centered / np.sqrt(
        np.mean(centered * centered, axis=-1, keepdims=True) + eps
    )
    return (
        norm.reshape(b, c, h, w) * weights[prefix + ".weight"][None, :, None, None]
        + weights[prefix + ".bias"][None, :, None, None]
    )


def encoder_single_frame(x, weights, config):
    # AutoencoderKLMiniMaxH3Image._encode: encoder(x[:, :, None])[:, :, -1]
    # with quant_conv applied before taking the last frame.
    x = conv_single_frame(x, weights, "encoder.conv_in", padding=1)
    for i in range(len(config["block_out_channels"])):
        p = f"encoder.down_blocks.{i}"
        for j in range(config["layers_per_block"]):
            r = f"{p}.resnets.{j}"
            residual = x
            y = silu(
                group_norm(
                    x,
                    weights,
                    r + ".norm1",
                    config["norm_num_groups"],
                    config["norm_eps"],
                )
            )
            y = conv_single_frame(y, weights, r + ".conv1", padding=1)
            y = silu(
                group_norm(
                    y,
                    weights,
                    r + ".norm2",
                    config["norm_num_groups"],
                    config["norm_eps"],
                )
            )
            y = conv_single_frame(y, weights, r + ".conv2", padding=1)
            if r + ".conv_shortcut.weight" in weights:
                residual = conv_single_frame(residual, weights, r + ".conv_shortcut")
            x = residual + y
        if (
            config["spatial_downsample_factors"][i]
            * config["temporal_downsample_factors"][i]
            > 1
        ):
            stride = config["spatial_downsample_factors"][i]
            x = conv_single_frame(
                x,
                weights,
                p + ".downsamplers.0.conv",
                stride=stride,
                asymmetric=stride == 2,
            )
    x = silu(
        group_norm(
            x,
            weights,
            "encoder.norm_out",
            config["norm_num_groups"],
            config["norm_eps"],
        )
    )
    x = conv_single_frame(x, weights, "encoder.conv_out", padding=1)
    return conv_single_frame(x, weights, "quant_conv")


def split_tiles(length, tile_size, overlap, ratio):
    # AutoencoderKLMiniMaxH3Image._split_tiles
    if tile_size >= length:
        return [0], [length], []
    count = math.ceil(length / tile_size)
    while tile_size * count - overlap * (count - 1) - length < 0:
        count += 1
    overlaps = [overlap] * (count - 1)
    remaining = tile_size * count - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (count - 1)] += ratio
    starts = [0]
    for i in range(count - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * count, overlaps


def blend(a, b, extent, axis):
    # AutoencoderKLMiniMaxH3Image._blend; original neighbor, endpoint excluded.
    extent = min(a.shape[axis], b.shape[axis], extent)
    shape = [1] * b.ndim
    shape[axis] = extent
    positions = np.arange(extent).reshape(shape)
    sa, sb, rest = (
        [slice(None)] * b.ndim,
        [slice(None)] * b.ndim,
        [slice(None)] * b.ndim,
    )
    sa[axis], sb[axis], rest[axis] = (
        slice(-extent, None),
        slice(0, extent),
        slice(extent, None),
    )
    mixed = a[tuple(sa)] * (1 - positions / extent) + b[tuple(sb)] * (
        positions / extent
    )
    return np.concatenate([mixed, b[tuple(rest)]], axis=axis)


def tiled_vae(values, weights, config, decode, tile=256, overlap=64):
    # AutoencoderKLMiniMaxH3Image._encode/_decode/_stitch_tiles
    ratio = math.prod(config["spatial_downsample_factors"])
    scale = ratio if decode else 1
    ys, yl, yo = split_tiles(values.shape[-2] * scale, tile, overlap, ratio)
    xs, xl, xo = split_tiles(values.shape[-1] * scale, tile, overlap, ratio)
    tiles = []
    for y, h in zip(ys, yl):
        row = []
        for x, w in zip(xs, xl):
            z = values[
                ..., y // scale : (y + h) // scale, x // scale : (x + w) // scale
            ]
            row.append(
                decoder2d(
                    conv_single_frame(z, weights, "post_quant_conv"), weights, config
                )
                if decode
                else encoder_single_frame(z, weights, config)
            )
        tiles.append(row)
    if not decode:
        yo, xo = [v // ratio for v in yo], [v // ratio for v in xo]
    result = []
    for i, row in enumerate(tiles):
        out = []
        for j, z in enumerate(row):
            if i:
                z = blend(tiles[i - 1][j], z, yo[i - 1], -2)
            if j:
                z = blend(row[j - 1], z, xo[j - 1], -1)
            if i < len(tiles) - 1:
                z = z[..., : -yo[i], :]
            if j < len(row) - 1:
                z = z[..., : -xo[j]]
            out.append(z)
        result.append(np.concatenate(out, axis=-1))
    return np.concatenate(result, axis=-2)


def dit_forward(
    image,
    text,
    timesteps,
    timestep_indices,
    positions,
    tags,
    image_indices,
    text_indices,
    head,
    weights,
    config,
):
    # MiniMaxH3ImageTransformer2DModel.forward and its named submodules.
    rotary = dit_rope(positions, config)
    image = linear(image, weights, "proj_in")
    text = linear(text, weights, "context_embedder")
    for i in range(config["num_refiner_layers"]):
        p = f"token_refiner.refiner_blocks.{i}"
        text = text + attention(
            rms(text, weights, p + ".norm1", config["norm_eps"]),
            weights,
            p + ".attn",
            config["num_attention_heads"],
            config["qk_norm_eps"],
        )
        text = text + feed_forward(
            rms(text, weights, p + ".norm2", config["norm_eps"]), weights, p + ".ff"
        )
    text = rms(text, weights, "token_refiner.final_norm", config["final_norm_eps"])
    x = np.zeros((text.shape[0], len(positions), text.shape[-1]))
    x[:, text_indices], x[:, image_indices] = text, image
    half = config["freq_dim"] // 2
    frequencies = np.exp(-math.log(10000) * np.arange(half, dtype=np.float64) / half)
    arguments = timesteps[:, None] * frequencies[None]
    temb = np.concatenate([np.cos(arguments), np.sin(arguments)], axis=-1)
    if config["freq_dim"] % 2:
        temb = np.pad(temb, ((0, 0), (0, 1)))
    temb = linear(
        silu(linear(temb, weights, "time_embedder.linear_1")),
        weights,
        "time_embedder.linear_2",
    )
    adaln_indices = timestep_indices * 3 + tags
    for i in range(config["num_layers"]):
        p = f"transformer_blocks.{i}"
        table = linear(silu(temb), weights, p + ".adaln_proj.linear").reshape(
            -1, 6 * config["hidden_size"]
        )
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = [
            v[adaln_indices] for v in np.split(table, 6, axis=-1)
        ]
        norm = (
            rms(x, weights, p + ".norm1", config["norm_eps"]) * (1 + scale_a) + shift_a
        )
        x = x + gate_a * attention(
            norm,
            weights,
            p + ".attn",
            config["num_attention_heads"],
            config["qk_norm_eps"],
            rotary,
        )
        norm = (
            rms(x, weights, p + ".norm2", config["norm_eps"]) * (1 + scale_f) + shift_f
        )
        x = x + gate_f * feed_forward(norm, weights, p + ".ff")
    shift, scale = np.split(linear(silu(temb), weights, "norm_out.linear"), 2, axis=-1)
    x = (
        rms(x, weights, "norm_out.norm", config["final_norm_eps"])
        * (1 + scale[timestep_indices])
        + shift[timestep_indices]
    )
    return linear(x, weights, f"pdd_proj_out.{head}")[:, image_indices]


def euler_step(sample, velocity, sigma, sigma_next):
    # MiniMaxH3Scheduler.step, euler_update_form='velocity', deterministic.
    return sample + (sigma - sigma_next) * velocity


def randomize(model, seed=19):
    from mlx.utils import tree_flatten

    rng = np.random.default_rng(seed)
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        values = rng.normal(0, 0.08, value.shape).astype(np.float32)
        if key.endswith(".weight") and value.ndim == 1:
            values += 1.0
        weights[key] = values.astype(np.float64)
    model.load_weights(
        [(k, mx.array(v.astype(np.float32))) for k, v in weights.items()]
    )
    return weights


def tiny_dit():
    from mlx_vlm.models.minimax_h3.config import MiniMaxH3TransformerConfig

    from rapid_mlx.image.minimax_h3_image.transformer import MiniMaxH3ImageTransformer

    config = MiniMaxH3TransformerConfig(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=32,
        num_refiner_layers=1,
        ffn_dim=128,
        in_channels=2,
        text_dim=32,
        freq_dim=16,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
        rope_freq_dim=4,
    )
    return MiniMaxH3ImageTransformer(config), asdict(config)


def tiny_vae():
    from rapid_mlx.image.minimax_h3_image.vae import (
        MiniMaxH3ImageVAE,
        MiniMaxH3ImageVAEConfig,
    )

    config = MiniMaxH3ImageVAEConfig(
        latent_channels=2,
        block_out_channels=(4,),
        layers_per_block=1,
        spatial_downsample_factors=(2,),
        temporal_downsample_factors=(1,),
        norm_num_groups=2,
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=16,
        decoder_num_register_tokens=2,
        latents_mean=(0.2, -0.3),
        latents_std=(0.8, 1.2),
    )
    return MiniMaxH3ImageVAE(config), asdict(config)


def assert_close(actual, expected):
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, rtol=1e-4, atol=1e-4
    )


def test_two_axis_rope():
    from rapid_mlx.image.minimax_h3_image.vae import MiniMaxH3ImageRotaryPosEmbed

    positions = np.array([[[-0.75, 0.25], [0, 0], [0.5, -0.5]]])
    actual = MiniMaxH3ImageRotaryPosEmbed(32)(mx.array(positions, mx.float32))
    for a, e in zip(actual, image_rope(positions, 32)):
        assert_close(a, e)


def test_decoder2d():
    model, config = tiny_vae()
    weights = randomize(model)
    z = np.random.default_rng(2).normal(size=(1, 2, 3, 4))
    assert_close(model.decoder(mx.array(z, mx.float32)), decoder2d(z, weights, config))


def test_single_frame_encoder_and_tiled_blend():
    model, config = tiny_vae()
    weights = randomize(model)
    pixels = np.random.default_rng(3).normal(size=(1, 3, 12, 16))
    model.use_tiling = False
    assert_close(
        model.encode(mx.array(pixels, mx.float32)).parameters,
        encoder_single_frame(pixels, weights, config),
    )
    model.use_tiling = True
    model.tile_sample_min_height = model.tile_sample_min_width = 8
    model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = 2
    assert_close(
        model.encode(mx.array(pixels, mx.float32)).parameters,
        tiled_vae(pixels, weights, config, False, 8, 2),
    )
    z = np.random.default_rng(4).normal(size=(1, 2, 6, 8))
    assert_close(
        model.decode(mx.array(z, mx.float32)), tiled_vae(z, weights, config, True, 8, 2)
    )
    assert_close(model.denormalize(model.normalize(mx.array(z, mx.float32))), z)


def test_dit_pdd_heads_and_edit_timesteps():
    from rapid_mlx.image.minimax_h3_image.layout import (
        build_image_packed_sequence,
        row_timestep_plan,
    )

    model, config = tiny_dit()
    weights = randomize(model)
    pos, tags, ii, ti, nr = build_image_packed_sequence(
        np.array([1, 0, 1]), [(2, 2)], 4, 4
    )
    times, indices = row_timestep_plan(len(tags), ii, nr)[2]
    rng = np.random.default_rng(5)
    image, text = rng.normal(size=(1, len(ii), 8)), rng.normal(size=(1, len(ti), 32))
    outputs = []
    for head in (0, 5):
        actual = model(
            mx.array(image, mx.float32),
            mx.array(text, mx.float32),
            mx.array(times),
            mx.array(indices),
            mx.array(pos, mx.float32),
            mx.array(tags, mx.int32),
            mx.array(ii, mx.int32),
            mx.array(ti, mx.int32),
            head,
        )
        expected = dit_forward(
            image, text, times, indices, pos, tags, ii, ti, head, weights, config
        )
        assert_close(actual, expected)
        outputs.append(expected)
    assert not np.allclose(*outputs)


def test_velocity_euler_and_six_step_t2i():
    from rapid_mlx.image.minimax_h3_image.layout import (
        SIGMAS,
        build_image_packed_sequence,
        row_timestep_plan,
    )
    from rapid_mlx.image.minimax_h3_image.pipeline import velocity_euler_step

    model, config = tiny_dit()
    weights = randomize(model)
    vae, vc = tiny_vae()
    vw = randomize(vae, seed=21)
    vae.use_tiling = False
    pos, tags, ii, ti, nr = build_image_packed_sequence(np.ones(3), [], 4, 4)
    rng = np.random.default_rng(6)
    expected = rng.normal(size=(1, 4, 8))
    actual = mx.array(expected, mx.float32)
    text = rng.normal(size=(1, 3, 32))
    times = np.asarray([1 - s for s in SIGMAS[:-1]], np.float32)
    sigmas = np.r_[np.float32(1) - times, np.float32(0)]
    for step, (ts, indices) in enumerate(row_timestep_plan(len(tags), ii, nr)):
        ev = dit_forward(
            expected, text, ts, indices, pos, tags, ii, ti, step, weights, config
        )
        av = model(
            actual,
            mx.array(text, mx.float32),
            mx.array(ts),
            mx.array(indices),
            mx.array(pos, mx.float32),
            mx.array(tags, mx.int32),
            mx.array(ii, mx.int32),
            mx.array(ti, mx.int32),
            step,
        )
        actual = velocity_euler_step(
            actual, av, float(sigmas[step]), float(sigmas[step + 1])
        )
        expected = euler_step(
            expected, ev, float(sigmas[step]), float(sigmas[step + 1])
        )
        assert_close(actual, expected)

    def unpack(x):
        return (
            x.reshape(1, 2, 2, 2, 2, 2).transpose(0, 3, 1, 4, 2, 5).reshape(1, 2, 4, 4)
        )

    az = vae.denormalize(unpack(actual))
    ez = (
        unpack(expected) * np.array(vc["latents_std"])[None, :, None, None]
        + np.array(vc["latents_mean"])[None, :, None, None]
    )
    assert_close(
        vae.decode(az), decoder2d(conv_single_frame(ez, vw, "post_quant_conv"), vw, vc)
    )


def test_stream_load_quantizes_only_allowed_linears(tmp_path):
    from dataclasses import replace

    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from rapid_mlx.image.minimax_h3_image.layout import (
        build_image_packed_sequence,
        row_timestep_plan,
    )
    from rapid_mlx.image.minimax_h3_image.quantize import (
        quantization_predicate,
        stream_load,
    )
    from rapid_mlx.image.minimax_h3_image.transformer import MiniMaxH3ImageTransformer

    original, _ = tiny_dit()
    # All quantized Linear input dimensions must be divisible by group_size=64.
    config = replace(original.config, time_embed_dim=64)
    original = MiniMaxH3ImageTransformer(config)
    randomize(original)
    all_weights = dict(tree_flatten(original.parameters()))
    # Put biases and weights in separate shards to exercise completion tracking.
    mx.save_safetensors(
        str(tmp_path / "a.safetensors"),
        {k: v for k, v in all_weights.items() if k.endswith(".bias")},
    )
    mx.save_safetensors(
        str(tmp_path / "b.safetensors"),
        {k: v for k, v in all_weights.items() if not k.endswith(".bias")},
    )
    pos, tags, ii, ti, nr = build_image_packed_sequence(np.ones(3), [], 4, 4)
    times, indices = row_timestep_plan(len(tags), ii, nr)[2]
    rng = np.random.default_rng(20)
    image, text = rng.normal(size=(1, 4, 8)), rng.normal(size=(1, 3, 32))
    for bits in (4, 8):
        model = MiniMaxH3ImageTransformer(config)
        eligible = {p for p, m in model.named_modules() if quantization_predicate(p, m)}
        stream_load(model, tmp_path, bits=bits)
        quantized = {
            p for p, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)
        }
        assert quantized == eligible
        # Build the independent control using the dequantized weights, to check
        # the actual packed Linear/AdaLN path without conflating quantization loss.
        weights = {
            k: np.array(v.astype(mx.float32)).astype(np.float64)
            for k, v in tree_flatten(model.parameters())
        }
        for path, module in model.named_modules():
            if isinstance(module, nn.QuantizedLinear):
                value = mx.dequantize(
                    module.weight,
                    module.scales,
                    module.biases,
                    group_size=64,
                    bits=bits,
                )
                weights[path + ".weight"] = np.array(value).astype(np.float64)
        for prefix in ("proj_in", "time_embedder", "norm_out", "pdd_proj_out"):
            for k, v in tree_flatten(model.parameters()):
                if k.startswith(prefix):
                    assert_close(v, all_weights[k])
        actual = model(
            mx.array(image, mx.float32),
            mx.array(text, mx.float32),
            mx.array(times),
            mx.array(indices),
            mx.array(pos, mx.float32),
            mx.array(tags, mx.int32),
            mx.array(ii, mx.int32),
            mx.array(ti, mx.int32),
            2,
        )
        expected = dit_forward(
            image, text, times, indices, pos, tags, ii, ti, 2, weights, asdict(config)
        )
        assert_close(actual, expected)


def test_strict_weight_coverage(tmp_path):
    from mlx.utils import tree_flatten

    from rapid_mlx.image.minimax_h3_image.quantize import stream_load

    model, _ = tiny_dit()
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {"unexpected": mx.ones((1,))}
    )
    with pytest.raises(ValueError, match="unexpected"):
        stream_load(model, tmp_path)
    missing = dict(weights)
    missing.pop("proj_in.weight")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), missing)
    with pytest.raises(ValueError, match="missing checkpoint parameters"):
        stream_load(model, tmp_path)
    wrong_shape = {**weights, "proj_in.weight": mx.ones((1, 1))}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), wrong_shape)
    with pytest.raises(ValueError, match="shape mismatch"):
        stream_load(model, tmp_path)


@pytest.mark.parametrize("bits", [0, 6, 16])
def test_invalid_quantization_bits(bits):
    from rapid_mlx.image.minimax_h3_image.quantize import validate_bits

    with pytest.raises(ValueError, match="quantize must be"):
        validate_bits(bits)


@pytest.mark.parametrize(
    "case,match",
    [
        ("empty", "no safetensors"),
        ("indexes", "multiple checkpoint indexes"),
        ("duplicate_source", "duplicate checkpoint tensor"),
        ("wrong_index", "checkpoint index disagrees"),
        ("missing_index_tensor", "index contains missing tensors"),
        ("duplicate_parameter", "unexpected or duplicate parameter"),
    ],
)
def test_stream_load_checkpoint_errors(tmp_path, case, match):
    import json

    import mlx.nn as nn

    from rapid_mlx.image.minimax_h3_image.quantize import stream_load

    model = nn.Linear(2, 2, bias=False)
    weight = mx.ones((2, 2))
    kwargs = {}
    if case != "empty":
        mx.save_safetensors(str(tmp_path / "a.safetensors"), {"weight": weight})
    if case == "indexes":
        for name in ("a", "b"):
            (tmp_path / f"{name}.safetensors.index.json").write_text("{}")
    elif case == "duplicate_source":
        mx.save_safetensors(str(tmp_path / "b.safetensors"), {"weight": weight})
    elif case in {"wrong_index", "missing_index_tensor"}:
        mapping = {
            "weight": "wrong.safetensors" if case == "wrong_index" else "a.safetensors"
        }
        if case == "missing_index_tensor":
            mapping["absent"] = "b.safetensors"
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
    elif case == "duplicate_parameter":
        mx.save_safetensors(str(tmp_path / "b.safetensors"), {"other": weight})
        kwargs["sanitize"] = lambda tensors: {"weight": next(iter(tensors.values()))}
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        stream_load(model, tmp_path, **kwargs)


def test_stream_load_selection_and_index(tmp_path):
    import json

    import mlx.nn as nn

    from rapid_mlx.image.minimax_h3_image.quantize import stream_load

    model = nn.Linear(2, 2, bias=False)
    mx.save_safetensors(
        str(tmp_path / "a.safetensors"),
        {"source": mx.ones((2, 2)), "ignored": mx.zeros((1,))},
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"source": "a.safetensors", "ignored": "a.safetensors"}}
        )
    )
    stream_load(
        model,
        tmp_path,
        select=lambda key: key != "ignored",
        sanitize=lambda w: {"weight": w["source"]},
        cast=lambda k, v: v.astype(mx.bfloat16),
    )
    assert model.weight.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.array(model.weight.astype(mx.float32)), np.ones((2, 2))
    )
    assert not model.training


@pytest.mark.parametrize(
    "change,match",
    [
        ({"block_out_channels": ()}, "encoder levels"),
        ({"spatial_downsample_factors": ()}, "encoder levels"),
        ({"temporal_downsample_factors": ()}, "encoder levels"),
        ({"latents_mean": ()}, "latent statistics"),
        ({"latents_std": ()}, "latent statistics"),
        ({"latents_std": (0, 1)}, "standard deviations"),
        ({"decoder_rope_dim_ratio": 0}, "rotary dimension"),
        ({"decoder_rope_dim_ratio": 0.2}, "rotary dimension"),
        ({"decoder_rope_dim_ratio": 2}, "rotary dimension"),
    ],
)
def test_vae_config_validation(change, match):
    from dataclasses import replace

    model, _ = tiny_vae()
    with pytest.raises(ValueError, match=match):
        replace(model.config, **change)


@pytest.mark.parametrize("dim", [0, 3, -4])
def test_invalid_image_rope_dimension(dim):
    from rapid_mlx.image.minimax_h3_image.vae import MiniMaxH3ImageRotaryPosEmbed

    with pytest.raises(ValueError, match="divisible by 4"):
        MiniMaxH3ImageRotaryPosEmbed(dim)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"pdd_step_index": -1}, "PDD head range"),
        ({"pdd_step_index": 6}, "PDD head range"),
        ({"position_ids": np.zeros((3,))}, "position_ids"),
        ({"position_ids": np.zeros((3, 2))}, "position_ids"),
        ({"token_tags": np.ones(2)}, "sequence length"),
        ({"timestep_indices": np.ones(2)}, "sequence length"),
    ],
)
def test_transformer_input_validation(change, match):
    model, _ = tiny_dit()
    args = dict(
        hidden_states=mx.zeros((1, 1, 8)),
        encoder_hidden_states=mx.zeros((1, 2, 32)),
        timestep=mx.zeros((1,)),
        timestep_indices=mx.zeros((3,), mx.int32),
        position_ids=mx.zeros((3, 3)),
        token_tags=mx.ones((3,), mx.int32),
        image_indices=mx.array([2]),
        text_indices=mx.array([0, 1]),
        pdd_step_index=0,
    )
    args.update(
        {
            key: mx.array(value) if isinstance(value, np.ndarray) else value
            for key, value in change.items()
        }
    )
    with pytest.raises(ValueError, match=match):
        model(**args)


def test_transformer_and_vae_load_local_checkpoints(tmp_path):
    import json

    from mlx.utils import tree_flatten

    from rapid_mlx.image.minimax_h3_image.transformer import (
        load_transformer,
        transformer_dtype,
    )
    from rapid_mlx.image.minimax_h3_image.vae import load_vae

    for make, folder, loader in [
        (tiny_dit, "transformer_diffusers", load_transformer),
        (tiny_vae, "vae_diffusers", load_vae),
    ]:
        original, config = make()
        randomize(original)
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
        weights = dict(tree_flatten(original.parameters()))
        source = {}
        for key, value in weights.items():
            if folder == "vae_diffusers":
                value = (
                    value.transpose(0, 4, 1, 2, 3)
                    if value.ndim == 5
                    else value.transpose(0, 3, 1, 2)
                    if value.ndim == 4
                    else value
                )
            source[key] = value
        mx.save_safetensors(str(directory / "model.safetensors"), source)
        loaded = loader(tmp_path)
        actual = dict(tree_flatten(loaded.parameters()))
        assert actual.keys() == weights.keys()
        for key, value in actual.items():
            expected = (
                transformer_dtype(key, weights[key])
                if folder == "transformer_diffusers"
                else weights[key].astype(mx.float32)
            )
            assert value.dtype == expected.dtype
            assert_close(value, np.array(expected.astype(mx.float32)))
        assert not loaded.training


def test_vae_encode_normalize_and_tiled_decode():
    model, _ = tiny_vae()
    randomize(model)
    pixels = mx.zeros((1, 3, 8, 8))
    model.use_tiling = False
    posterior = model.encode(pixels)
    assert posterior.mean.shape == (1, 2, 4, 4)
    assert posterior.logvar.shape == posterior.mean.shape
    normalized = model.normalize(posterior.mean)
    assert_close(model.denormalize(normalized), np.array(posterior.mean))
    expected = model.decode(posterior.mean)
    model.use_tiling = True
    actual = model.decode(posterior.mean)
    assert_close(actual, np.array(expected))
    # Force overlapping tiles on both spatial axes; normalization remains NCHW.
    model.tile_sample_min_height = model.tile_sample_min_width = 6
    model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = 2
    tiled = model.encode(pixels)
    assert tiled.mean.shape == posterior.mean.shape
    output = model.decode(tiled.mean)
    assert output.shape == pixels.shape
    assert bool(mx.all(mx.isfinite(output)).item())
