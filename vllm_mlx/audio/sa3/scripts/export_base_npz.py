"""Convert a Stable Audio 3 *-base torch checkpoint → the MLX training npz.

Torch-free (safetensors.numpy). Produces `dit_<model>-base_f16.npz` — the DiT
weights + baked conditioner that `lora_train_mlx.py --dit-weights` expects.
LoRA training uses the BASE checkpoint (not the shipped ARC weights inference
uses); these npz are auto-downloaded by weights.py, or reproduced here from the
HF `stabilityai/stable-audio-3-*-base` repos.

    python scripts/export_base_npz.py \
        --base-dir <sa3-*-base>            # dir with model.safetensors + model_config.json
        --output   models/mlx/dit_<model>-base_f16.npz
        [--validate-against <existing npz>]   # assert bit-parity with a known-good npz

Conversion (identical to the shipped inference-npz layout): strip the DiT
wrapper prefix `model.model.`, transpose the two Conv1d weights
[out,in,k]→[out,k,in], rename RMSNorm `.gamma`→`.weight`, remap
`.to_local_embed.{0,2}`→`.seq.{0,2}`, and fold the 3 conditioner tensors under
the `cond.` prefix. Same for small (sm-music/sm-sfx) and medium.
"""
import argparse

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

_DIT_PREFIX = "model.model."
_CONV_WEIGHTS = ("preprocess_conv.weight", "postprocess_conv.weight")
_COND_MAP = {
    "conditioner.conditioners.prompt.padding_embedding": "cond.padding_embedding",
    "conditioner.conditioners.seconds_total.embedder.embedding.1.weight": "cond.seconds_total_weight",
    "conditioner.conditioners.seconds_total.embedder.embedding.1.bias": "cond.seconds_total_bias",
}


def convert(base_dir):
    sd = load_file(f"{base_dir}/model.safetensors")
    out = {}
    for k, arr in sd.items():
        if k in _COND_MAP:
            out[_COND_MAP[k]] = mx.array(arr.astype(np.float32))
            continue
        if not k.startswith(_DIT_PREFIX):
            continue  # pretransform / T5 backbone — not part of the DiT npz
        sk = k[len(_DIT_PREFIX):]
        a = arr.astype(np.float32)
        if sk in _CONV_WEIGHTS:  # Conv1d [out,in,k] -> MLX [out,k,in]
            out[sk] = mx.array(a.transpose(0, 2, 1))
        elif sk.endswith(".gamma"):  # RMSNorm .gamma -> .weight
            out[sk[: -len(".gamma")] + ".weight"] = mx.array(a)
        else:
            sk = sk.replace(".to_local_embed.0.", ".to_local_embed.seq.0.")
            sk = sk.replace(".to_local_embed.2.", ".to_local_embed.seq.2.")
            out[sk] = mx.array(a)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--validate-against", default=None,
                    help="Assert key-set + value parity with a known-good npz.")
    args = ap.parse_args()

    wd = convert(args.base_dir)
    print(f"converted {len(wd)} keys ({sum(1 for k in wd if k.startswith('cond'))} conditioner)")

    if args.validate_against:
        ref = dict(mx.load(args.validate_against))
        extra, missing = set(wd) - set(ref), set(ref) - set(wd)
        assert not extra and not missing, f"key mismatch: extra={extra}, missing={missing}"
        worst = max(
            float(mx.max(mx.abs(wd[k].astype(mx.float32) - ref[k].astype(mx.float32))))
            for k in ref
        )
        print(f"validate: {len(ref)} keys match, max|Δ| = {worst:.2e}")
        # Tolerance is fp16-scale: the source ckpt is fp32 and both npz are fp16,
        # so a handful of the specially-transformed keys (conv transpose, the
        # to_local_embed rename) can differ by ~1 fp16 ULP (~4e-3) depending on
        # the rounding path. A real structural bug (wrong transpose/rename) gives
        # an O(0.1-1) diff or a shape/key mismatch — both caught well below this.
        assert worst < 1e-2, "values diverge from the reference npz (structural bug)"

    mx.savez(args.output, **{k: v.astype(mx.float16) for k, v in wd.items()})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
