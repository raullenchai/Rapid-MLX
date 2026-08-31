# SPDX-License-Identifier: Apache-2.0
"""Contracts for strict loading of quantized GLM-5 Next forget gates."""

from __future__ import annotations

import subprocess
import sys

from vllm_mlx.patches.glm5_next_forget_gate_quant import (
    _remap_quantized_forget_gate_tensors,
)


def test_quantization_metadata_follows_forget_gate_weight_rename() -> None:
    weights = {
        "model.layers.0.self_attn.f_a_proj.scales": object(),
        "model.layers.0.self_attn.f_a_proj.biases": object(),
        "model.layers.0.self_attn.f_b_proj.scales": object(),
        "model.layers.0.self_attn.f_b_proj.biases": object(),
        "model.layers.0.self_attn.forget_gate.f_a_proj.weight": object(),
        "model.layers.0.self_attn.q_proj.scales": object(),
    }

    remapped = _remap_quantized_forget_gate_tensors(weights)

    for projection in ("f_a_proj", "f_b_proj"):
        for leaf in ("scales", "biases"):
            assert (
                f"model.layers.0.self_attn.forget_gate.{projection}.{leaf}" in remapped
            )
            assert f"model.layers.0.self_attn.{projection}.{leaf}" not in remapped
    assert "model.layers.0.self_attn.forget_gate.f_a_proj.weight" in remapped
    assert "model.layers.0.self_attn.q_proj.scales" in remapped


def test_installer_wraps_released_sanitizer_once_in_clean_process() -> None:
    script = """
from vllm_mlx.patches import glm5_next_forget_gate_quant as patch
from mlx_vlm.models.glm5_next import language

language.LanguageModel.sanitize = lambda self, weights: dict(weights)
assert patch.install_glm5_next_forget_gate_quant_fix() is True
assert patch.install_glm5_next_forget_gate_quant_fix() is False
assert patch.is_installed() is True

result = language.LanguageModel.sanitize(
    object(),
    {"model.layers.0.self_attn.f_a_proj.scales": "sentinel"},
)
assert result == {
    "model.layers.0.self_attn.forget_gate.f_a_proj.scales": "sentinel"
}
assert language._RAPID_MLX_FORGET_GATE_QUANT_INSTALLED is True
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
