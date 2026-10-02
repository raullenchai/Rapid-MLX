# SPDX-License-Identifier: Apache-2.0
"""Convert an official CLM PyTorch head to a safe, MLX-native artifact."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


def _expected_head_shapes(config: Mapping) -> dict[str, tuple[int, ...]]:
    missing = {"width", "depth"} - config.keys()
    if missing:
        raise ValueError(f"CLM cfg is missing: {sorted(missing)}")
    dimensions = {
        "hidden_size": int(config.get("hidden_size", 4096)),
        "width": int(config["width"]),
        "depth": int(config["depth"]),
        "projection_dim": int(config.get("projection_dim", 512)),
    }
    if any(value < 1 for value in dimensions.values()):
        raise ValueError("CLM cfg dimensions must be positive")
    if dimensions["depth"] < 2:
        raise ValueError("CLM cfg depth must be at least 2")
    hidden = dimensions["hidden_size"]
    width = dimensions["width"]
    depth = dimensions["depth"]
    projection = dimensions["projection_dim"]
    shapes = {
        "inp.weight": (width, hidden),
        "inp.bias": (width,),
        "out.weight": (projection, width),
        "out.bias": (projection,),
    }
    for index in range(max(0, depth - 2)):
        shapes[f"hidden.{index}.weight"] = (width, width)
        shapes[f"hidden.{index}.bias"] = (width,)
        if config.get("layernorm", False):
            shapes[f"norms.{index}.weight"] = (width,)
            shapes[f"norms.{index}.bias"] = (width,)
    return shapes


@contextmanager
def _artifact_lock(destination: Path, *, exclusive: bool) -> Iterator[None]:
    """Coordinate artifact publication with readers in other processes."""
    identity = hashlib.sha256(str(destination.resolve()).encode()).hexdigest()
    lock_dir = (
        Path(tempfile.gettempdir()) / f"rapid-mlx-{os.getuid()}" / "clm-artifact-locks"
    )
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_dir / f"{identity}.lock"
    with lock_path.open("a+b") as lock_file:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_file.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _publish_artifact(staging: Path, destination: Path) -> None:
    """Publish one generation while blocking new artifact readers."""
    with _artifact_lock(destination, exclusive=True):
        _publish_artifact_unlocked(staging, destination)


def _publish_artifact_unlocked(staging: Path, destination: Path) -> None:
    required = {"config.json", "model.safetensors"}
    if {item.name for item in staging.iterdir()} != required:
        raise ValueError("staged CLM artifact is incomplete")
    backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    moved_old = False
    published = False
    restored = False
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
        published = True
    except BaseException:
        if moved_old and backup.exists() and not destination.exists():
            os.replace(backup, destination)
            restored = True
        raise
    finally:
        # If restoration itself fails, the backup is the only remaining copy.
        if backup.exists() and (published or restored):
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                warnings.warn(
                    f"published CLM artifact but retained backup {backup}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )


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
    if not isinstance(checkpoint, Mapping):
        raise ValueError("CLM checkpoint root must be a mapping")
    required = {"state_head", "action_head", "logit_scale", "cfg"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"CLM checkpoint is missing: {sorted(missing)}")
    if not isinstance(checkpoint["cfg"], Mapping):
        raise ValueError("CLM checkpoint cfg must be a mapping")
    config = dict(checkpoint["cfg"])
    config["projection_dim"] = int(
        checkpoint.get("projection_dim", config.get("projection_dim", 512))
    )
    config["hidden_size"] = int(config.get("hidden_size", 4096))
    expected_shapes = _expected_head_shapes(config)
    logit_scale = checkpoint["logit_scale"]
    logit_scale_value = float(torch.as_tensor(logit_scale).float().item())
    if not math.isfinite(logit_scale_value) or not (
        math.log(sys.float_info.min)
        <= logit_scale_value
        <= math.log(sys.float_info.max)
    ):
        raise ValueError("CLM logit_scale is outside the finite exponential range")
    config["logit_scale"] = logit_scale_value
    config.setdefault("model_name", "clm-latest")
    try:
        config_json = json.dumps(config, indent=2, sort_keys=True) + "\n"
    except TypeError as exc:
        raise ValueError("CLM cfg must contain only JSON-compatible values") from exc
    tensors = {}
    for prefix in ("state_head", "action_head"):
        state = checkpoint[prefix]
        if not isinstance(state, Mapping):
            raise ValueError(f"CLM checkpoint {prefix} must be a mapping")
        if any(not isinstance(name, str) for name in state):
            raise ValueError(f"CLM checkpoint {prefix} tensor names must be strings")
        names = set(state)
        if names != set(expected_shapes):
            missing_names = sorted(set(expected_shapes) - names)
            unexpected_names = sorted(names - set(expected_shapes))
            raise ValueError(
                f"CLM checkpoint {prefix} tensors do not match config; "
                f"missing={missing_names}, unexpected={unexpected_names}"
            )
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{prefix}.{name} is not a tensor")
            actual_shape = tuple(int(value) for value in tensor.shape)
            if actual_shape != expected_shapes[name]:
                raise ValueError(
                    f"CLM checkpoint {prefix}.{name} has shape {actual_shape}; "
                    f"expected {expected_shapes[name]}"
                )
            tensors[f"{prefix}.{name}"] = mx.array(
                np.asarray(tensor.detach().float().cpu().numpy())
            )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        mx.save_safetensors(str(staging / "model.safetensors"), tensors)
        (staging / "config.json").write_text(config_json, encoding="utf-8")
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


if __name__ == "__main__":  # pragma: no cover - console-script compatibility
    main()
