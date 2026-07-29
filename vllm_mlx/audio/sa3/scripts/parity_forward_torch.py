"""§9 forward-parity harness — torch (MPS) side.

Controlled single forward of the SA3 base model, replicating underfit's
training-step forward (underfit/training/loop.py) on INJECTED inputs so the
result is deterministic and diffable against the MLX trainer's forward
(parity_forward_mlx.py). fp32, cfg_dropout=0.

Run in an env with `stable_audio_3` importable (the sa3 repo venv):

    python parity_forward_torch.py \
        --base-dir <sa3-sm-music-base>       # dir with model_config.json + model.safetensors
        --latents  <clip.npy>                # a [256,T] pre-encoded latent
        --workdir  <dir>                     # writes shared_inputs.npz + torch_out.npz

Then run parity_forward_mlx.py (MLX venv) with the same --workdir.

The harness localizes any mismatch: it saves cross/global/noised/prediction/
loss + the model's pretransform.scale (the MLX side asserts scale==1.0 for the
softnorm no-scale convention; a non-1.0 value would be a real bug). Weights on
both sides come from the same fp16 source upcast to fp32, so agreement is the
cross-framework fp32 floor (~79 dB); a convention bug reads far lower.
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.loading_utils import remap_state_dict_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="Base checkpoint dir (model_config.json + model.safetensors)")
    ap.add_argument("--latents", required=True, help="A [256,T] pre-encoded latent .npy")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--prompt", default="techno")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--noise-seed", type=int, default=12345)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    # Shared injected inputs (numpy — both engines consume these verbatim).
    clip = np.load(args.latents)[:, : args.crop]
    lat = np.stack([clip, clip, clip], 0).astype(np.float32)            # B=3
    noise = np.random.default_rng(args.noise_seed).standard_normal(lat.shape).astype(np.float32)
    t = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    np.savez(f"{args.workdir}/shared_inputs.npz", latents=lat, noise=noise, t=t,
             prompt=args.prompt, seconds_total=args.seconds)

    cfg = json.load(open(f"{args.base_dir}/model_config.json"))
    print("building base model (fp32)...", flush=True)
    model = create_diffusion_cond_from_config(cfg)
    sd = load_file(f"{args.base_dir}/model.safetensors")
    sd = remap_state_dict_keys(sd, model.state_dict())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded: {len(missing)} missing / {len(unexpected)} unexpected", flush=True)
    model = model.to(device).eval().requires_grad_(False)

    scale = float(getattr(model.pretransform, "scale", 1.0)) if model.pretransform is not None else 1.0
    print(f"pretransform_scale = {scale}   diffusion_objective = {model.diffusion_objective}", flush=True)

    latents = torch.tensor(lat).to(device)
    noise_t = torch.tensor(noise).to(device)
    tt = torch.tensor(t).to(device)
    B, C, T = latents.shape
    diffusion_input = latents / scale if scale != 1.0 else latents
    noised = diffusion_input * (1 - tt)[:, None, None] + noise_t * tt[:, None, None]
    target = noise_t - diffusion_input

    metadata = [{"prompt": args.prompt, "seconds_total": args.seconds, "seconds_start": 0.0}
                for _ in range(B)]
    conditioning = model.conditioner(metadata, device)
    conditioning["inpaint_mask"] = [torch.ones(B, 1, T, device=device)]
    conditioning["inpaint_masked_input"] = [torch.zeros(B, C, T, device=device)]
    with torch.no_grad():
        ci = model.get_conditioning_inputs(conditioning)

    # BACKWARD: gradient of the loss w.r.t. the noised input — the full DiT
    # backward sweep (every interior gradient the adapter grads are built from).
    noised.requires_grad_(True)
    output = model(noised, tt, cond=conditioning, cfg_dropout_prob=0.0,
                   padding_mask=torch.ones(B, T, dtype=torch.bool, device=device))
    loss = ((output.float() - target.float()) ** 2).mean()  # full mask → signal-only == mean
    loss.backward()
    grad_noised = noised.grad
    print(f"loss = {float(loss):.8f}   |grad_noised| = {float(grad_noised.abs().mean()):.6e}", flush=True)
    np.savez(f"{args.workdir}/torch_out.npz",
             prediction=output.detach().float().cpu().numpy(), loss=float(loss),
             cross=ci["cross_attn_cond"].float().cpu().numpy(),
             gcond=ci["global_cond"].float().cpu().numpy(),
             noised=noised.detach().float().cpu().numpy(),
             target=target.float().cpu().numpy(),
             grad_noised=grad_noised.float().cpu().numpy(), scale=scale)
    print(f"wrote {args.workdir}/torch_out.npz", flush=True)


if __name__ == "__main__":
    main()
