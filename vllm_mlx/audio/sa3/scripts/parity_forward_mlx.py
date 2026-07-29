"""§9 forward-parity harness — MLX side + diff.

Reads shared_inputs.npz + torch_out.npz (from parity_forward_torch.py) and runs
the SAME controlled forward through the MLX base DiT in fp32, then reports three
PSNRs that LOCALIZE any mismatch:

  (1) DiT-only  — feed torch's cross/global + the shared noised into the MLX DiT.
      Isolates the DiT forward convention (weights are the same fp16 source→fp32,
      so ~79 dB is the cross-framework fp32 floor; a convention bug reads ≪40 dB).
  (2) conditioning — MLX-native cross/global vs torch's (T5 + seconds; fp16-T5 bound).
  (3) end-to-end — full MLX-native prediction + loss vs torch.

Also asserts the torch model's pretransform.scale == 1.0 (the softnorm no-scale
convention the MLX trainer relies on). Run in the MLX venv:

    python parity_forward_mlx.py --base-npz <dit_sm-music-base_f16.npz> --workdir <dir>
"""
import argparse
import importlib
import sys

import numpy as np


def psnr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max(np.abs(a).max(), np.abs(b).max()) / np.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-npz", required=True, help="MLX base DiT weights (dit_<model>-base_f16.npz)")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--dit-module", default="models.defs.dit_mlx")
    ap.add_argument("--prompt", default="techno")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    import mlx.core as mx
    from sa3_mlx import T5GEMMA_NPZ_REL
    from weights import ensure_local
    from models.defs.sa3_pipeline import apply_prompt_padding, load_conditioner_from_npz
    from models.defs.t5gemma_mlx import T5Gemma

    tt = np.load(f"{args.workdir}/torch_out.npz")
    inp = np.load(f"{args.workdir}/shared_inputs.npz", allow_pickle=True)
    assert abs(float(tt["scale"]) - 1.0) < 1e-9, \
        f"pretransform.scale={float(tt['scale'])} != 1.0 — the MLX trainer's no-scale " \
        "softnorm convention is WRONG for this model (real bug)."

    latents = mx.array(inp["latents"]).astype(mx.float32)
    noise = mx.array(inp["noise"]).astype(mx.float32)
    t = mx.array(inp["t"]).astype(mx.float32)
    B, C, T = latents.shape
    dit = importlib.import_module(args.dit_module).load_dit(
        args.base_npz, T_lat=T, dtype=mx.float32, compile_=False)

    noised = latents * (1.0 - t)[:, None, None] + noise * t[:, None, None]
    target = noise - latents
    lac = mx.concatenate([mx.ones((B, 1, T)), mx.zeros((B, C, T))], axis=1).transpose(0, 2, 1).astype(mx.float32)

    # (1) DiT-only: inject torch's cross/global
    pred1 = dit(noised, t, mx.array(tt["cross"]).astype(mx.float32),
                mx.array(tt["gcond"]).astype(mx.float32), local_add_cond=lac)
    mx.eval(pred1)

    # (2) MLX-native conditioning
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs = load_conditioner_from_npz(args.base_npz, prefix="cond.")
    embeds, mask = t5.encode([args.prompt] * B, max_len=256)
    prompt_cond = apply_prompt_padding(embeds.astype(mx.float32), mask, padding_emb.astype(mx.float32))
    sec_tok = mx.stop_gradient(secs([args.seconds] * B).astype(mx.float32))
    cross_m = mx.concatenate([prompt_cond, sec_tok], axis=1)
    gcond_m = sec_tok[:, 0, :]
    mx.eval(cross_m, gcond_m)

    # (3) end-to-end
    pred3 = dit(noised, t, cross_m, gcond_m, local_add_cond=lac)
    mx.eval(pred3)
    loss_m = float(mx.mean((pred3 - target) ** 2))

    # (4) BACKWARD: ∂loss/∂noised through the DiT (with torch's cross/global so
    # the backward is compared on identical inputs — isolates the DiT backward).
    cross_t = mx.array(tt["cross"]).astype(mx.float32)
    gcond_t = mx.array(tt["gcond"]).astype(mx.float32)

    def loss_of_noised(nz):
        pred = dit(nz, t, cross_t, gcond_t, local_add_cond=lac)
        return mx.mean((pred - target) ** 2)

    grad_m = mx.grad(loss_of_noised)(noised)
    mx.eval(grad_m)
    p_grad = psnr(np.array(grad_m), tt["grad_noised"]) if "grad_noised" in tt else float("nan")

    print(f"noised sanity PSNR   = {psnr(np.array(noised), tt['noised']):.1f} dB (inf = bit-identical)")
    print(f"(1) DiT-only forward = {psnr(np.array(pred1), tt['prediction']):6.2f} dB  [conv bug ≪40; fp32 floor ~79]")
    print(f"(2) conditioning     = cross {psnr(np.array(cross_m), tt['cross']):.2f} dB / global {psnr(np.array(gcond_m), tt['gcond']):.2f} dB")
    print(f"(3) end-to-end pred  = {psnr(np.array(pred3), tt['prediction']):6.2f} dB")
    print(f"(4) BACKWARD d loss/d noised PSNR = {p_grad:6.2f} dB  [full DiT backward; conv bug ≪40]")
    print(f"loss: torch {float(tt['loss']):.6f}  MLX {loss_m:.6f}  |Δ| {abs(float(tt['loss'])-loss_m):.2e}")


if __name__ == "__main__":
    sys.exit(main())
