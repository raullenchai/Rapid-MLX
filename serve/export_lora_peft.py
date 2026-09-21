# SPDX-License-Identifier: Apache-2.0
"""Convert an mlx-lm LoRA safetensors into PEFT layout for GGUF conversion.

PREP for the Vast.ai path (docs/engineering/operations/vast-ai-serving-runbook.md).
mlx-lm stores LoRA as `{module_path}.lora_A` / `{module_path}.lora_B` keys
(single fused matrices per side); PEFT/transformers expects
`base_model.model.{module_path}.lora_A.weight` / `.lora_B.weight` plus a
PEFT `adapter_config.json`. The mapping below is structural; final fidelity
is verified on the rented box by the accuracy gate (within 1.5 pts of the
mlx readout on the 192-item held-out set).

  python export_lora_peft.py --adapter <mlx adapter dir> --out peft-adapter/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="dir containing adapters.safetensors")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        raise SystemExit("pip install safetensors torch")

    src = Path(args.adapter) / "adapters.safetensors"
    tensors = load_file(str(src))
    out = {}
    keymap = {}
    for k, v in tensors.items():
        if k.endswith(".lora_A") or k.endswith(".lora_A.weight"):
            newk = f"base_model.model.{k.split('.lora_A')[0]}.lora_A.weight"
        elif k.endswith(".lora_B") or k.endswith(".lora_B.weight"):
            newk = f"base_model.model.{k.split('.lora_B')[0]}.lora_B.weight"
        else:
            newk = k
        out[newk] = v
        keymap[k] = newk

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapter_model.safetensors"))
    manifest = json.loads((Path(__file__).parent / "adapter_manifest.json").read_text())
    rank = min(t.shape[0] for k, t in tensors.items() if "lora_A" in k and t.ndim == 2)
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "r": int(rank),
        "target_modules": sorted({km[len("base_model.model."):].rsplit(".lora_A", 1)[0]
                                  for km in keymap.values() if ".lora_A." in km}),
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "bias": "none", "lora_alpha": int(rank), "lora_dropout": 0.0,
    }, indent=1))
    (out_dir / "marvin_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out_dir}/adapter_model.safetensors ({len(out)} tensors, rank {rank})")
    print("verify on GPU box: accuracy gate within 1.5 pts of mlx readout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
