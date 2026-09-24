# SPDX-License-Identifier: Apache-2.0
"""Convert an official CLM PyTorch head to a safe, MLX-native artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(input_path: str | Path, output_dir: str | Path) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "conversion requires PyTorch; install it only in the conversion environment"
        ) from exc
    import mlx.core as mx
    import numpy as np

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # weights_only prevents the checkpoint pickle from constructing arbitrary
    # Python objects. Official CLM heads contain tensors and primitive config.
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    required = {"state_head", "action_head", "logit_scale", "cfg"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"CLM checkpoint is missing: {sorted(missing)}")
    config = dict(checkpoint["cfg"])
    config["projection_dim"] = int(
        checkpoint.get("projection_dim", config.get("projection_dim", 512))
    )
    config["hidden_size"] = int(config.get("hidden_size", 4096))
    logit_scale = checkpoint["logit_scale"]
    config["logit_scale"] = float(torch.as_tensor(logit_scale).float().item())
    config.setdefault("model_name", "clm-latest")
    tensors = {}
    for prefix in ("state_head", "action_head"):
        state = checkpoint[prefix]
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{prefix}.{name} is not a tensor")
            tensors[f"{prefix}.{name}"] = mx.array(
                np.asarray(tensor.detach().float().cpu().numpy())
            )
    mx.save_safetensors(str(destination / "model.safetensors"), tensors)
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official CLM .pt checkpoint")
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()
    try:
        result = convert(args.input, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(result)


if __name__ == "__main__":
    main()
