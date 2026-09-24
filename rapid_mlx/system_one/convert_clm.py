# SPDX-License-Identifier: Apache-2.0
"""Convert an official CLM PyTorch head to a safe, MLX-native artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path


def _publish_artifact(staging: Path, destination: Path) -> None:
    """Publish a validated two-file artifact as one directory generation."""
    required = {"config.json", "model.safetensors"}
    if {item.name for item in staging.iterdir()} != required:
        raise ValueError("staged CLM artifact is incomplete")
    backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    moved_old = False
    try:
        if destination.exists():
            if not destination.is_dir():
                raise ValueError(
                    f"CLM output exists and is not a directory: {destination}"
                )
            unexpected = {item.name for item in destination.iterdir()} - required
            if unexpected:
                raise ValueError(
                    "refusing to replace CLM output containing unrelated files: "
                    f"{sorted(unexpected)}"
                )
            os.replace(destination, backup)
            moved_old = True
        os.replace(staging, destination)
    except BaseException:
        if moved_old and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


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
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        mx.save_safetensors(str(staging / "model.safetensors"), tensors)
        (staging / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Materialize both files before publishing the directory generation.
        mx.eval(mx.load(str(staging / "model.safetensors")))
        json.loads((staging / "config.json").read_text(encoding="utf-8"))
        _publish_artifact(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
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
