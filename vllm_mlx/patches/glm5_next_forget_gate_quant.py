# SPDX-License-Identifier: Apache-2.0
"""Complete GLM-5 Next forget-gate remapping for quantized checkpoints.

The released architecture sanitizer moves ``f_a_proj.weight`` and
``f_b_proj.weight`` below ``forget_gate`` but leaves their quantization
``scales`` and ``biases`` at the old path. MLX then creates quantized modules at
the nested path and strict loading rejects the 136 orphan tensors. This wrapper
finishes the same architecture-owned rename for quantization metadata.

The correction is self-retiring: if a later runtime remaps the metadata itself,
no old-path keys remain and this wrapper is a no-op.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_INSTALLED = False
_PROJECTION_NAMES = ("f_a_proj", "f_b_proj")
_QUANTIZATION_LEAVES = ("scales", "biases")


def _remap_quantized_forget_gate_tensors(weights: dict) -> dict:
    remapped = {}
    for key, value in weights.items():
        new_key = key
        for projection in _PROJECTION_NAMES:
            old = f".self_attn.{projection}."
            if old not in key or not key.endswith(_QUANTIZATION_LEAVES):
                continue
            new_key = key.replace(
                old,
                f".self_attn.forget_gate.{projection}.",
                1,
            )
            break
        remapped[new_key] = value
    return remapped


def install_glm5_next_forget_gate_quant_fix() -> bool:
    """Wrap the released GLM-5 Next sanitizer once."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return False

        from mlx_vlm.models.glm5_next import language

        if getattr(language, "_RAPID_MLX_FORGET_GATE_QUANT_INSTALLED", False):
            _INSTALLED = True
            return False

        original = language.LanguageModel.sanitize

        def patched_sanitize(self, weights):
            return _remap_quantized_forget_gate_tensors(original(self, weights))

        language.LanguageModel.sanitize = patched_sanitize
        language._RAPID_MLX_FORGET_GATE_QUANT_INSTALLED = True
        _INSTALLED = True
        return True


def is_installed() -> bool:
    return _INSTALLED


__all__ = [
    "install_glm5_next_forget_gate_quant_fix",
    "is_installed",
]
