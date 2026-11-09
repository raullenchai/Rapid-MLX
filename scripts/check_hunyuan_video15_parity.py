#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tiny random Diffusers ↔ MLX parity, without a torch runtime dependency.

Run with an MLX Python and --reference-python pointing at a Diffusers/torch
Python. Reference execution is a separate process and never loads real weights.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRANSFORMER = dict(
    in_channels=9,
    out_channels=4,
    num_attention_heads=2,
    attention_head_dim=16,
    num_layers=2,
    num_refiner_layers=1,
    mlp_ratio=2.0,
    patch_size=1,
    patch_size_t=1,
    qk_norm="rms_norm",
    text_embed_dim=12,
    text_embed_2_dim=8,
    image_embed_dim=8,
    rope_theta=256.0,
    rope_axes_dim=[4, 6, 6],
    target_size=640,
    task_type="t2v",
    use_meanflow=False,
)
VAE = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=2,
    block_out_channels=[8, 8, 8],
    layers_per_block=1,
    spatial_compression_ratio=4,
    temporal_compression_ratio=2,
    downsample_match_channel=True,
    upsample_match_channel=True,
    scaling_factor=1.03682,
)

QWEN = dict(
    model_type="qwen2_5_vl_text",
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    num_hidden_layers=4,
    vocab_size=64,
    bos_token_id=1,
    eos_token_id=2,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
)
T5 = dict(
    vocab_size=64,
    d_model=16,
    d_ff=32,
    d_kv=4,
    num_heads=2,
    num_layers=2,
    feed_forward_proj="gated-gelu",
    dropout_rate=0.0,
    layer_norm_epsilon=1e-6,
)


def reference(directory):
    import torch
    from diffusers import (
        AutoencoderKLHunyuanVideo15,
        FlowMatchEulerDiscreteScheduler,
        HunyuanVideo15Transformer3DModel,
    )
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15RotaryPosEmbed,
        HunyuanVideo15TransformerBlock,
    )
    from safetensors.torch import save_file

    torch.manual_seed(17)
    torch.set_num_threads(2)

    def save(name, **arrays):
        np.savez(
            directory / (name + ".npz"),
            **{
                k: v.detach().float().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in arrays.items()
            },
        )

    block = HunyuanVideo15TransformerBlock(2, 16, 2).eval()
    x, context, t = torch.randn(1, 6, 32), torch.randn(1, 7, 32), torch.randn(1, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    rope = HunyuanVideo15RotaryPosEmbed(1, 1, [4, 6, 6])(torch.zeros(1, 9, 1, 2, 3))
    with torch.no_grad():
        y, cy = block(x, context, t, mask, rope)
    save(
        "block",
        x=x,
        context=context,
        t=t,
        mask=mask,
        cos=rope[0],
        sin=rope[1],
        y=y,
        cy=cy,
    )
    save_file(block.state_dict(), directory / "block.safetensors")
    model = HunyuanVideo15Transformer3DModel(**TRANSFORMER).eval()
    x = torch.randn(1, 9, 2, 2, 3)
    inputs = dict(
        hidden_states=x,
        timestep=torch.tensor([750.0]),
        encoder_hidden_states=torch.randn(1, 5, 12),
        encoder_attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
        encoder_hidden_states_2=torch.randn(1, 4, 8),
        encoder_attention_mask_2=torch.tensor([[1, 1, 0, 0]]),
        image_embeds=torch.zeros(1, 3, 8),
    )
    with torch.no_grad():
        y = model(**inputs).sample
    save("transformer", **inputs, y=y)
    save_file(model.state_dict(), directory / "transformer.safetensors")
    model.to(torch.bfloat16)
    bf16_inputs = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        y_bf16 = model(**bf16_inputs).sample
    save("transformer_bf16", **bf16_inputs, y=y_bf16)
    rope = model.rope(x)
    save("rope", cos=rope[0], sin=rope[1])
    vae = AutoencoderKLHunyuanVideo15(**VAE).eval()
    z = torch.randn(1, 2, 3, 2, 3)
    with torch.no_grad():
        y = vae.decode(z).sample
        vae.enable_tiling()
        vae.tile_sample_min_height = vae.tile_sample_min_width = 16
        vae.tile_latent_min_height = vae.tile_latent_min_width = 4
        z_tiled = torch.randn(1, 2, 2, 4, 5)
        tiled = vae.tiled_decode(z_tiled)
    save("vae", z=z, y=y, z_tiled=z_tiled, tiled=tiled)
    save_file(
        {k: v for k, v in vae.state_dict().items() if k.startswith("decoder.")},
        directory / "vae.safetensors",
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5)
    scheduler.set_timesteps(sigmas=np.linspace(1, 0, 8)[:-1])
    sample, prediction = torch.randn(1, 4, 2, 2, 3), torch.randn(1, 4, 2, 2, 3)
    outputs = []
    current = sample
    for time in scheduler.timesteps:
        current = scheduler.step(prediction, time, current).prev_sample
        outputs.append(current)
    save(
        "scheduler",
        sample=sample,
        prediction=prediction,
        sigmas=scheduler.sigmas,
        timesteps=scheduler.timesteps,
        outputs=torch.stack(outputs),
    )

    from transformers import (
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLTextModel,
        T5Config,
        T5EncoderModel,
    )

    qconfig = Qwen2_5_VLTextConfig(
        **QWEN,
        rope_scaling={"type": "default", "mrope_section": [2, 3, 3]},
        rope_parameters={"rope_type": "default", "mrope_section": [2, 3, 3]},
    )
    qwen = Qwen2_5_VLTextModel(qconfig).eval()
    ids = torch.tensor([[1, 4, 7, 12, 3, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        hidden = qwen(
            ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states[-3]
    save("qwen", ids=ids, mask=mask, hidden=hidden)
    folder = directory / "qwen_weights"
    folder.mkdir()
    save_file(qwen.state_dict(), folder / "model.safetensors")
    t5 = T5EncoderModel(T5Config(**T5)).eval()
    with torch.no_grad():
        hidden = t5(ids, attention_mask=mask).last_hidden_state
    save("t5", ids=ids, mask=mask, hidden=hidden)
    folder = directory / "t5_weights"
    folder.mkdir()
    save_file(
        {
            k: v
            for k, v in t5.state_dict().items()
            if k != "encoder.embed_tokens.weight"
        },
        folder / "model.safetensors",
    )


def compare(directory):
    import mlx.core as mx

    from rapid_mlx.video.hunyuan_video15 import FlowMatchScheduler
    from rapid_mlx.video.hunyuan_video15.transformer import (
        Transformer,
        TransformerBlock,
        rotary_embedding,
    )
    from rapid_mlx.video.hunyuan_video15.vae import VAE as MLXVAE

    report = {}

    def check(name, actual, expected, atol=3e-5, rtol=3e-4):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        diff = np.abs(actual - expected)
        report[name] = {
            "max_abs": float(diff.max()),
            "max_rel": float((diff / np.maximum(np.abs(expected), 1e-6)).max()),
        }
        print(name, json.dumps(report[name]), flush=True)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def load(model, name):
        weights = mx.load(str(directory / (name + ".safetensors")))
        weights = [
            (k, v.transpose(0, 2, 3, 4, 1) if v.ndim == 5 else v)
            for k, v in weights.items()
        ]
        model.load_weights(weights, strict=True)

    def arrays(name):
        return {k: mx.array(v) for k, v in np.load(directory / (name + ".npz")).items()}

    a = arrays("block")
    block = TransformerBlock(2, 16, 2)
    load(block, "block")
    y, cy = block(a["x"], a["context"], a["t"], a["mask"], (a["cos"], a["sin"]))
    check("block_latent", y, a["y"])
    check("block_context", cy, a["cy"])
    a = arrays("transformer")
    model = Transformer(TRANSFORMER)
    load(model, "transformer")
    y = model(**{k: v for k, v in a.items() if k != "y"})
    check("transformer", y, a["y"])
    model.set_dtype(mx.bfloat16)
    bf = arrays("transformer_bf16")
    y_bf16 = model(**{k: v.astype(mx.bfloat16) for k, v in bf.items() if k != "y"})
    check("transformer_bf16", y_bf16.astype(mx.float32), bf["y"], 2e-2, 3e-2)
    cos, sin = rotary_embedding(a["hidden_states"].shape, 1, 1, [4, 6, 6], 256)
    r = arrays("rope")
    check("rope_cos", cos, r["cos"], 1e-6)
    check("rope_sin", sin, r["sin"], 1e-6)
    a = arrays("vae")
    vae = MLXVAE(VAE)
    load(vae, "vae")
    check("vae_decode", vae.decode(a["z"]), a["y"], 1e-4, 1e-3)
    check(
        "vae_tiled_decode",
        vae.tiled_decode(a["z_tiled"], tile_sample_size=16),
        a["tiled"],
        1e-4,
        1e-3,
    )
    a = arrays("scheduler")
    scheduler = FlowMatchScheduler({"shift": 5, "num_train_timesteps": 1000}, 7)
    check("scheduler_sigmas", scheduler.sigmas, a["sigmas"], 1e-7)
    check("scheduler_timesteps", scheduler.timesteps, a["timesteps"], 1e-4)
    current = a["sample"]
    for i in range(7):
        current = scheduler.step(a["prediction"], i, current)
        check(f"scheduler_step_{i}", current, a["outputs"][i], 1e-6)

    import mlx.nn as nn
    from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model
    from transformers import T5Config

    from rapid_mlx.image.sd35_runtime._vendor.t5 import SD3T5Encoder
    from rapid_mlx.video.hunyuan_video15.loading import load_component, qwen_key, t5_key
    from rapid_mlx.video.hunyuan_video15.text import (
        encode_qwen_hidden,
        encode_t5_hidden,
    )

    qconfig = dict(QWEN, num_hidden_layers=2)
    qwen = Qwen2Model(ModelArgs.from_dict(qconfig))
    del qwen.norm
    load_component(
        qwen,
        directory / "qwen_weights",
        dtype=mx.float32,
        key_map=lambda k: qwen_key(k, 2),
    )
    a = arrays("qwen")
    check(
        "qwen_text_hidden",
        encode_qwen_hidden(qwen, a["ids"].astype(mx.int32), a["mask"]),
        a["hidden"],
        1e-5,
        1e-4,
    )
    t5 = SD3T5Encoder(T5Config(**T5), low_memory_mode=False)
    for layer in t5.encoder.layers:
        layer.dense.act = nn.gelu_approx
    load_component(t5, directory / "t5_weights", dtype=mx.float32, key_map=t5_key)
    a = arrays("t5")
    check(
        "byt5_hidden",
        encode_t5_hidden(t5, a["ids"].astype(mx.int32), a["mask"]),
        a["hidden"],
        1e-5,
        1e-4,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--write-reference", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.write_reference:
        reference(args.write_reference)
        return
    with tempfile.TemporaryDirectory(
        prefix="website-poshog-hyv15-parity-",
        dir="/private/tmp" if Path("/private/tmp").exists() else None,
    ) as tmp:
        directory = Path(tmp)
        subprocess.run(
            [
                args.reference_python,
                str(Path(__file__).resolve()),
                "--write-reference",
                str(directory),
            ],
            check=True,
        )
        report = compare(directory)
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
