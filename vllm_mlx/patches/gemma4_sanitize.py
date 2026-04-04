# SPDX-License-Identifier: Apache-2.0
"""
Monkey-patch for mlx-vlm Gemma 4 weight loading bug.

mlx-vlm 0.4.3 has two issues with Gemma 4:
1. sanitize() doubles the 'model.' prefix for language_model keys
2. MLX-format models skip sanitize entirely, but their weight keys
   still need the prefix fix to match model parameter paths

Upstream issue: https://github.com/Blaizzy/mlx-vlm/issues/912

TODO: Remove this patch once mlx-vlm fixes the bug.
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def apply_gemma4_sanitize_patch():
    """Patch mlx-vlm's load_model to fix Gemma 4 weight key mapping."""
    global _patched
    if _patched:
        return

    try:
        import mlx_vlm.utils as vlm_utils
    except ImportError:
        return

    _original_load_model = vlm_utils.load_model

    def _patched_load_model(model_path, lazy=False, **kwargs):
        """Wrap load_model to fix Gemma 4 weight keys for MLX-format models."""
        import json
        from pathlib import Path

        model_path = Path(model_path)
        config_path = model_path / "config.json"

        # Only patch for Gemma 4 models
        is_gemma4 = False
        if config_path.exists():
            config = json.loads(config_path.read_text())
            model_type = config.get("model_type", "")
            is_gemma4 = "gemma4" in model_type

        if not is_gemma4:
            return _original_load_model(model_path, lazy, **kwargs)

        # Patch sanitize to fix the double-prefix bug
        from mlx_vlm.models.gemma4 import gemma4

        def _fixed_sanitize(self, weights):
            """Complete sanitize replacement with prefix fix + conv/MoE transforms."""
            use_clipped = getattr(
                self.config.vision_config, "use_clipped_linears", False
            )
            sanitized = {}
            for k, v in weights.items():
                # Skip clipping params when not used
                if any(s in k for s in ["input_max", "input_min", "output_max", "output_min"]):
                    if "vision_tower" in k and not use_clipped:
                        continue
                    if "vision_tower" not in k and "audio_tower" not in k:
                        continue
                if "rotary_emb" in k:
                    continue
                if self.audio_tower is None and ("audio_tower" in k or "embed_audio" in k):
                    continue

                new_key = k
                if new_key.startswith("model."):
                    new_key = new_key[len("model."):]
                # FIX: only add model. if not already present
                if new_key.startswith("language_model.model."):
                    pass
                elif new_key.startswith("language_model."):
                    rest = new_key[len("language_model."):]
                    new_key = "language_model.model." + rest

                # Conv2d transpose
                if "subsample_conv_projection" in new_key and "conv.weight" in new_key and v.ndim == 4:
                    v = v.transpose(0, 2, 3, 1)
                if "depthwise_conv1d.weight" in new_key and v.ndim == 3:
                    v = v.transpose(0, 2, 1)

                # MoE transforms
                if new_key.endswith(".experts.down_proj"):
                    new_key = new_key.replace(".experts.down_proj", ".experts.switch_glu.down_proj.weight")
                if new_key.endswith(".experts.gate_up_proj"):
                    gate_key = new_key.replace(".experts.gate_up_proj", ".experts.switch_glu.gate_proj.weight")
                    up_key = new_key.replace(".experts.gate_up_proj", ".experts.switch_glu.up_proj.weight")
                    v = v.swapaxes(-1, -2)
                    mid_dim = v.shape[-1] // 2
                    sanitized[gate_key] = v[..., :mid_dim].swapaxes(-1, -2)
                    sanitized[up_key] = v[..., mid_dim:].swapaxes(-1, -2)
                    continue

                sanitized[new_key] = v
            return sanitized

        # We can't just fix sanitize — MLX-format skips it entirely.
        # Instead, we need to also fix the `is_mlx_format` check.
        # The simplest approach: temporarily force is_mlx_format=False
        # by removing the _quantization key that triggers it.
        #
        # Actually, let's just intercept after load and re-assign weights
        # with correct sanitization.
        _orig_model_sanitize = gemma4.Model.sanitize
        gemma4.Model.sanitize = _fixed_sanitize

        # Call original load_model (returns model only)
        model = _original_load_model(model_path, lazy, **kwargs)

        # Restore sanitize
        gemma4.Model.sanitize = _orig_model_sanitize

        # If weights are still zero (MLX format skipped sanitize),
        # manually load and sanitize them
        import mlx.core as mx

        test_param = model.language_model.model.embed_tokens
        # Check scales — quantized models have non-zero scales when loaded correctly.
        # Weight itself may have packed uint32 values even when scales are missing.
        test_scales = getattr(test_param, "scales", None)
        needs_reload = test_scales is not None and mx.all(test_scales == 0).item()

        if needs_reload:
            logger.info("[gemma4] MLX format skipped sanitize, reloading weights...")

            # Load raw weights (skip macOS resource forks)
            weight_files = sorted(
                f for f in model_path.glob("*.safetensors")
                if not f.name.startswith("._")
            )
            raw_weights = {}
            for wf in weight_files:
                raw_weights.update(mx.load(str(wf)))

            # Run our fixed sanitize
            sanitized = _fixed_sanitize(model, raw_weights)

            # Apply to model
            model.load_weights(list(sanitized.items()), strict=False)
            logger.info(
                "[gemma4] Reloaded %d weights with fixed sanitization",
                len(sanitized),
            )

        return model

    vlm_utils.load_model = _patched_load_model
    _patched = True
    logger.info("[gemma4] Applied load_model monkey-patch (mlx-vlm#912)")
