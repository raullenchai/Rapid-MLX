#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Extract MTP weights from a HuggingFace model and save as a quantized sidecar.

mlx-lm's convert/quantize pipeline strips mtp.* weights during sanitize().
This script extracts them from the original bf16 weights, quantizes them
to match the target MLX model's quantization config, and saves them as
model-mtp.safetensors in the MLX model directory.

Usage:
    python3.12 scripts/extract_mtp_weights.py \
        --hf-model Qwen/Qwen3.5-27B \
        --mlx-model /path/to/quantized-mlx-model

The script will:
1. Download only the safetensors shard(s) containing mtp.* weights
2. Quantize them to match the MLX model's quantization config
3. Save as model-mtp.safetensors in the MLX model directory
"""

import argparse
import json
import logging
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _quantize_weight(w, group_size, bits):
    """Quantize a single weight tensor."""
    q_w, q_s, q_b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q_w, q_s, q_b)
    return q_w, q_s, q_b


def main():
    parser = argparse.ArgumentParser(description="Extract MTP weights from HF model")
    parser.add_argument(
        "--hf-model", required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-27B)"
    )
    parser.add_argument(
        "--mlx-model", required=True, help="Path to quantized MLX model directory"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Override quantization bits (default: from MLX model config)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Override group size (default: from MLX model config)",
    )
    args = parser.parse_args()

    mlx_dir = Path(args.mlx_model)
    if not mlx_dir.exists():
        logger.error(f"MLX model directory not found: {mlx_dir}")
        return 1

    # Read quantization config from MLX model
    config_path = mlx_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    quant_config = config.get("quantization", {})
    bits = args.bits or quant_config.get("bits", 4)
    group_size = args.group_size or quant_config.get("group_size", 64)
    logger.info(f"Target quantization: {bits}-bit, group_size={group_size}")

    # Find which shard files contain MTP weights
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading weight index from {args.hf_model}...")
    idx_path = hf_hub_download(args.hf_model, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    weight_map = idx.get("weight_map", {})
    mtp_keys = {k: v for k, v in weight_map.items() if k.startswith("mtp.")}

    if not mtp_keys:
        logger.error("No mtp.* weights found in model index!")
        return 1

    logger.info(f"Found {len(mtp_keys)} MTP weight keys")

    # Get unique shard files needed
    shard_files = sorted(set(mtp_keys.values()))
    logger.info(f"Need to download {len(shard_files)} shard file(s): {shard_files}")

    # Download and extract MTP weights
    all_mtp_weights = {}
    for shard_file in shard_files:
        logger.info(f"Downloading {shard_file}...")
        shard_path = hf_hub_download(args.hf_model, shard_file)
        shard_weights = mx.load(shard_path)
        for k in mtp_keys:
            if mtp_keys[k] == shard_file and k in shard_weights:
                all_mtp_weights[k] = shard_weights[k]
        del shard_weights

    logger.info(f"Extracted {len(all_mtp_weights)} MTP weight tensors")

    # mlx-lm's sanitize shifts norm weights by +1.0 when mtp weights are present.
    # Since we're extracting post-sanitize, the main model norms are already shifted.
    # We need to apply the same shift to MTP norm weights for consistency.
    # EVERY RMSNorm weight uses HF's ``(1 + w)`` convention; MLX's nn.RMSNorm
    # applies ``x * w`` directly, so each norm weight needs +1.0. This MUST
    # include the MTP-specific ``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden``
    # (the "norm" is mid-name, so an ``endswith('norm.weight')`` list silently
    # missed them — leaving the MTP head's fc-input normalization inverted and
    # producing ~0% draft acceptance). Match any 1-D norm weight instead.
    for k in list(all_mtp_weights.keys()):
        if "norm" in k and k.endswith(".weight") and all_mtp_weights[k].ndim == 1:
            all_mtp_weights[k] = all_mtp_weights[k] + 1.0
            logger.info(f"  Shifted norm: {k}")

    # Convert fused MoE experts (``experts.gate_up_proj`` / ``experts.down_proj``,
    # 3-D stacked, no ``.weight`` suffix) into the ``switch_mlp.{gate,up,down}_proj``
    # split layout the MLX qwen3_5_moe model + MTP injector expect. Mirrors
    # ``mlx_lm.models.qwen3_5_moe.Model.sanitize`` so quantization below sees the
    # canonical names. Dense-MTP models have no such keys and are unaffected.
    for gate_up_key in [
        k for k in list(all_mtp_weights) if k.endswith(".experts.gate_up_proj")
    ]:
        prefix = gate_up_key[: -len(".experts.gate_up_proj")]  # e.g. mtp.layers.0.mlp
        gate_up = all_mtp_weights.pop(gate_up_key)
        if gate_up.shape[-2] % 2 != 0:
            raise ValueError(
                f"{gate_up_key}: fused gate_up_proj has odd intermediate dim "
                f"{gate_up.shape[-2]}; cannot split into equal gate/up halves. "
                "The checkpoint's expert layout is not the expected "
                "[num_experts, 2*intermediate, hidden]."
            )
        mid = gate_up.shape[-2] // 2
        all_mtp_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
        all_mtp_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
        down_key = f"{prefix}.experts.down_proj"
        if down_key not in all_mtp_weights:
            raise ValueError(
                f"{gate_up_key} present but paired {down_key} is missing; the "
                "extracted sidecar would be incomplete. Check that the MTP shard "
                "carries the full expert set."
            )
        all_mtp_weights[f"{prefix}.switch_mlp.down_proj.weight"] = all_mtp_weights.pop(
            down_key
        )
        logger.info(
            f"  Converted MoE experts: {prefix}.experts.* -> {prefix}.switch_mlp.*"
        )

    # Quantize MTP weights
    quantized = {}
    for k, v in sorted(all_mtp_weights.items()):
        if not k.endswith(".weight"):
            continue

        # Only 1-D norm/bias vectors stay in FP. Every 2-D Linear weight —
        # including the router ``mlp.gate`` and the ``shared_expert_gate``
        # (out_features=1) — MUST be quantized, because the MTP injector
        # (``qwen3_5_inject``) quantizes the whole MTP module uniformly and
        # then loads this sidecar into it: an FP tensor where the module
        # expects a quantized (weight/scales/biases) triple is rejected as a
        # "missing required MTP tensor". Mirror the target model's scheme.
        is_norm = "norm" in k or "layernorm" in k
        if is_norm or v.ndim == 1:
            quantized[k] = v
            logger.info(f"  FP: {k} {v.shape}")
            continue

        q_bits, q_gs = bits, group_size
        q_w, q_s, q_b = _quantize_weight(v, q_gs, q_bits)
        quantized[k] = q_w
        quantized[k.replace(".weight", ".scales")] = q_s
        quantized[k.replace(".weight", ".biases")] = q_b
        logger.info(f"  Quantized {q_bits}-bit: {k} {v.shape} -> {q_w.shape}")

    # Save
    output_file = mlx_dir / "model-mtp.safetensors"
    logger.info(f"\nSaving {len(quantized)} tensors to {output_file}")
    mx.save_safetensors(str(output_file), quantized)

    total_bytes = sum(v.nbytes for v in quantized.values())
    logger.info(f"MTP weights size: {total_bytes / 1e6:.1f} MB")

    # Ensure config has mtp_num_hidden_layers (for our patch to detect)
    text_config = config.get("text_config", config)
    if text_config.get("mtp_num_hidden_layers") is None:
        # Read from HF config
        hf_cfg_path = hf_hub_download(args.hf_model, "config.json")
        with open(hf_cfg_path) as f:
            hf_cfg = json.load(f)
        hf_tc = hf_cfg.get("text_config", hf_cfg)
        mtp_layers = hf_tc.get("mtp_num_hidden_layers", 0)
        if mtp_layers:
            if "text_config" in config:
                config["text_config"]["mtp_num_hidden_layers"] = mtp_layers
            else:
                config["mtp_num_hidden_layers"] = mtp_layers
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Updated config: mtp_num_hidden_layers={mtp_layers}")

    logger.info("\nDone! MTP weights extracted and quantized.")
    logger.info("Start server with: --enable-mtp")


if __name__ == "__main__":
    exit(main() or 0)
