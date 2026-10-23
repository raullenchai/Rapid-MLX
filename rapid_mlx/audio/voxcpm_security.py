# SPDX-License-Identifier: Apache-2.0
"""Keep legacy VoxCPM tensor sidecars without implicit pickle deserialization."""

from pathlib import Path


def load_voxcpm_checkpoint(path: Path):
    import mlx.core as mx
    from mlx_audio.tts.utils import MODEL_REMAPPING
    from mlx_audio.utils import (
        apply_quantization,
        get_model_class,
        get_model_name_parts,
        load_config,
        load_weights,
        model_type_from_config,
    )

    from rapid_mlx.utils.torch_checkpoint import load_tensor_checkpoint

    config = load_config(path)
    names = get_model_name_parts(path)
    kind = (
        model_type_from_config(config) or config.get("architecture") or names[0].lower()
    )
    if MODEL_REMAPPING.get(kind, kind) not in {"voxcpm", "voxcpm2"}:
        raise ValueError(
            "Refusing legacy audiovae.pth for an unrecognized TTS architecture"
        )
    module, kind = get_model_class(
        model_type=kind,
        model_name=names,
        category="tts",
        model_remapping=MODEL_REMAPPING,
    )
    config["model_path"] = str(path)
    model = module.Model(module.ModelConfig.from_dict(config))
    weights = load_weights(path)
    if not any(k.startswith("audio_vae.") for k in weights):
        sidecar = path / "audiovae.safetensors"
        if kind == "voxcpm2" and sidecar.is_file():
            state = mx.load(str(sidecar))
        else:
            state = load_tensor_checkpoint(path / "audiovae.pth")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            import torch

            if (
                not isinstance(state, dict)
                or not state
                or not all(
                    isinstance(key, str) and isinstance(value, torch.Tensor)
                    for key, value in state.items()
                )
            ):
                raise ValueError(
                    "VoxCPM audio VAE checkpoint must contain tensor weights"
                )
            state = {
                k: mx.array(
                    v.detach().cpu().float().numpy()
                    if v.dtype == torch.bfloat16
                    else v.detach().cpu().numpy()
                )
                for k, v in state.items()
            }
        if not state:
            raise ValueError("VoxCPM audio VAE checkpoint is empty")
        for key, value in state.items():
            weights[f"audio_vae.{key.removeprefix('module.')}"] = value
    # Keep the native architecture's normalization/quantization. Disable only
    # its implicit disk fallback on this instance, never monkeypatch torch or
    # a process-global model class. VAE weights are sanitized exactly once.
    original_path = model.args.model_path
    try:
        model.args.model_path = None
        weights = model.sanitize(weights)
    finally:
        model.args.model_path = original_path
    apply_quantization(
        model, config, weights, getattr(model, "model_quant_predicate", None)
    )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    if hasattr(module.Model, "post_load_hook"):
        model = module.Model.post_load_hook(model, path)
    return model
