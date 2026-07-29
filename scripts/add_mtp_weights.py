#!/usr/bin/env python3
"""
Add MTP (Multi-Token Prediction) weights to an existing MLX Qwen3-Next model.

This script:
1. Downloads the MTP shard from the original BF16 HuggingFace model
2. Extracts MTP weights (mtp.* keys)
3. Quantizes them to match the existing MLX model's quantization
4. Adds them to the MLX model's safetensors files
5. Updates config.json with num_nextn_predict_layers=1

Usage:
    python add_mtp_weights.py [--mlx-model-path PATH] [--source-model MODEL]

Requirements:
    pip install mlx
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Default paths
DEFAULT_MLX_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3-Next-80B-A3B-Instruct-6bit"
)
DEFAULT_SOURCE_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MTP_SHARD_NAME = "model-00041-of-00041.safetensors"
MTP_SHARD_URL = (
    f"https://huggingface.co/{DEFAULT_SOURCE_MODEL}/resolve/main/{MTP_SHARD_NAME}"
)


def find_snapshot_dir(model_path: str) -> Path:
    """Find the latest snapshot directory in HF cache structure."""
    snapshots_dir = Path(model_path) / "snapshots"
    if not snapshots_dir.exists():
        # Maybe it's a direct model directory
        if (Path(model_path) / "config.json").exists():
            return Path(model_path)
        raise FileNotFoundError(f"No snapshots found in {model_path}")

    # Find the most recent snapshot
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots in {snapshots_dir}")
    return snapshots[-1]


def download_mtp_shard(dest_path: Path, source_model: str) -> Path:
    """Download the MTP shard using curl with resume support."""
    shard_url = f"https://huggingface.co/{source_model}/resolve/main/{MTP_SHARD_NAME}"
    shard_path = dest_path / MTP_SHARD_NAME

    if shard_path.exists():
        print(f"MTP shard already exists: {shard_path}")
        # Verify size (should be ~3.3 GB)
        size_gb = shard_path.stat().st_size / 1e9
        if size_gb < 3.0:
            print(
                f"WARNING: File seems too small ({size_gb:.2f} GB), re-downloading..."
            )
        else:
            print(f"Size: {size_gb:.2f} GB — OK")
            return shard_path

    print("Downloading MTP shard (~3.3 GB)...")
    print(f"URL: {shard_url}")
    result = subprocess.run(
        ["curl", "-L", "-C", "-", "-o", str(shard_path), shard_url],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed with code {result.returncode}")

    size_gb = shard_path.stat().st_size / 1e9
    print(f"Downloaded: {size_gb:.2f} GB")
    return shard_path


def extract_and_quantize_mtp_weights(
    shard_path: Path, snapshot_dir: Path, quantization_bits: int = 6
):
    """Extract MTP weights, quantize, and save to MLX model directory."""
    import mlx.core as mx

    # Force CPU — no GPU needed, avoids Metal command buffer crashes
    mx.set_default_device(mx.cpu)

    print(f"\nExtracting MTP weights from {shard_path.name}...")

    # Load MTP weights from the BF16 shard using mx.load (handles bfloat16 natively)
    all_weights = mx.load(str(shard_path))
    mtp_weights = {k: v for k, v in all_weights.items() if k.startswith("mtp.")}
    del all_weights  # Free non-MTP weights

    print(f"Found {len(mtp_weights)} MTP weight keys")

    # Read existing quantization config to match
    config_path = snapshot_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Check quantization config
    quant_config = config.get("quantization", {})
    bits = quant_config.get("bits", quantization_bits)
    group_size = quant_config.get("group_size", 64)
    print(f"Target quantization: {bits}-bit, group_size={group_size}")

    # Quantize MTP weights (matching the model's quantization scheme)
    quantized_weights = {}
    skip_quantize = {
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
    }

    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".pre_fc_norm_hidden.weight",
        ".pre_fc_norm_embedding.weight",
    )

    def _quantize_one(key, weight):
        """Quantize a single weight, apply norm adjustment, return dict entries."""
        # Norm adjustment: +1.0 for RMSNorm weights (HF -> MLX convention)
        if key == "mtp.norm.weight" or any(key.endswith(s) for s in norm_suffixes):
            if weight.ndim == 1:
                weight = weight + 1.0
                mx.eval(weight)
                print(f"  Adjusted norm: {key}")

        if key in skip_quantize:
            print(f"  Keep FP: {key} {weight.shape}")
            return {key: weight}
        elif weight.ndim >= 2 and weight.shape[-1] >= group_size:
            q_w, q_s, q_b = mx.quantize(weight, group_size=group_size, bits=bits)
            mx.eval(q_w, q_s, q_b)
            print(f"  Quantize {bits}-bit: {key} {q_w.shape}")
            return {
                key: q_w,
                key.replace(".weight", ".scales"): q_s,
                key.replace(".weight", ".biases"): q_b,
            }
        else:
            print(f"  Keep FP (small): {key} {weight.shape}")
            return {key: weight}

    # Stack + quantize expert weights ONE PROJECTION AT A TIME to minimize peak memory.
    # Each projection: pop 512 BF16 experts -> stack -> quantize -> free BF16.
    num_experts = config.get("num_experts", 512)
    for proj in ["up_proj", "down_proj", "gate_proj"]:
        expert_keys = [
            f"mtp.layers.0.mlp.experts.{e}.{proj}.weight" for e in range(num_experts)
        ]
        if all(k in mtp_weights for k in expert_keys):
            stacked = mx.stack([mtp_weights.pop(k) for k in expert_keys])
            mx.eval(stacked)
            stacked_key = f"mtp.layers.0.mlp.switch_mlp.{proj}.weight"
            print(f"  Stacked {num_experts} experts for {proj}: {stacked.shape}")
            quantized_weights.update(_quantize_one(stacked_key, stacked))
            del stacked

    # Quantize remaining (non-expert) weights — pop to free originals immediately
    for key in list(mtp_weights.keys()):
        weight = mtp_weights.pop(key)
        quantized_weights.update(_quantize_one(key, weight))
        del weight
    del mtp_weights

    # Save MTP weights as a new safetensors file (name must match model*.safetensors glob)
    mtp_output_file = snapshot_dir / "model-mtp.safetensors"
    print(
        f"\nSaving {len(quantized_weights)} quantized MTP weights to {mtp_output_file}"
    )
    mx.save_safetensors(str(mtp_output_file), quantized_weights)

    # Calculate total size
    total_bytes = sum(v.nbytes for v in quantized_weights.values())
    print(f"MTP weights size: {total_bytes / 1e6:.1f} MB (quantized)")

    return mtp_output_file, list(quantized_weights.keys())


def update_model_index(snapshot_dir: Path, mtp_keys: list):
    """Update model.safetensors.index.json to include MTP weight keys."""
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"WARNING: No index file found at {index_path}, skipping index update")
        return

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Add MTP keys pointing to the new shard
    for key in mtp_keys:
        weight_map[key] = "model-mtp.safetensors"

    index["weight_map"] = weight_map
    # Update total_size (approximate — actual size includes quantization overhead)
    # Not critical for functionality

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Updated {index_path} with {len(mtp_keys)} MTP weight entries")


def update_config(snapshot_dir: Path):
    """Update config.json to enable MTP."""
    config_path = snapshot_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Add MTP config field
    config["num_nextn_predict_layers"] = 1

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated {config_path}: num_nextn_predict_layers=1")


def main():
    parser = argparse.ArgumentParser(
        description="Add MTP weights to MLX Qwen3-Next model"
    )
    parser.add_argument(
        "--mlx-model-path",
        type=str,
        default=DEFAULT_MLX_MODEL,
        help=f"Path to MLX model directory (default: {DEFAULT_MLX_MODEL})",
    )
    parser.add_argument(
        "--source-model",
        type=str,
        default=DEFAULT_SOURCE_MODEL,
        help=f"HuggingFace model to download MTP shard from (default: {DEFAULT_SOURCE_MODEL})",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Directory to download MTP shard to (default: temp dir)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=6,
        help="Quantization bits (default: 6, matching 6-bit model)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (use existing shard)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MTP Weight Addition for Qwen3-Next MLX Model")
    print("=" * 60)

    # Find snapshot directory
    snapshot_dir = find_snapshot_dir(args.mlx_model_path)
    print(f"\nMLX model snapshot: {snapshot_dir}")

    # Verify config exists
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: No config.json found in {snapshot_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)
    print(f"Model type: {config.get('model_type', 'unknown')}")
    print(f"Hidden size: {config.get('hidden_size', '?')}")
    print(f"Num experts: {config.get('num_experts', '?')}")

    if config.get("num_nextn_predict_layers", 0) > 0:
        print("\nWARNING: Model already has num_nextn_predict_layers set!")
        # Check if MTP weights already exist
        index_path = snapshot_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            mtp_keys = [k for k in index.get("weight_map", {}) if k.startswith("mtp.")]
            if mtp_keys:
                print(f"  Found {len(mtp_keys)} existing MTP weight keys")
                print("  MTP weights already added. Nothing to do.")
                sys.exit(0)

    # Download MTP shard
    download_dir = (
        Path(args.download_dir) if args.download_dir else Path(tempfile.mkdtemp())
    )
    if not args.skip_download:
        shard_path = download_mtp_shard(download_dir, args.source_model)
    else:
        shard_path = download_dir / MTP_SHARD_NAME
        if not shard_path.exists():
            print(f"ERROR: Shard not found at {shard_path}")
            sys.exit(1)

    # Extract, quantize, and save MTP weights
    mtp_file, mtp_keys = extract_and_quantize_mtp_weights(
        shard_path, snapshot_dir, quantization_bits=args.bits
    )

    # Update model index
    update_model_index(snapshot_dir, mtp_keys)

    # Update config
    update_config(snapshot_dir)

    print("\n" + "=" * 60)
    print("SUCCESS! MTP weights added to MLX model.")
    print("=" * 60)
    print(f"\nMTP weight file: {mtp_file}")
    print(f"Total MTP keys: {len(mtp_keys)}")
    print("\nTo use MTP, start the server with --enable-mtp:")
    print("  rapid-mlx serve mlx-community/Qwen3-Next-80B-A3B-Instruct-6bit \\")
    print("      --enable-mtp --port 1239")


if __name__ == "__main__":
    main()
