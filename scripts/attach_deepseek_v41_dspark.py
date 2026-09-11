#!/usr/bin/env python3
"""Attach checkpoint-native DSpark weights to a native V4.1 target artifact.

The target checkpoint is exposed through relative symlinks; only the compact
``mtp.*`` draft network is copied from the source.  This makes an A/B artifact
without duplicating the approximately 200 GB target model or mutating either
input checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from repack_deepseek_v41_native import (
    CheckpointIndex,
    TensorPlan,
    write_safetensors,
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _mtp_plans(source: CheckpointIndex) -> list[TensorPlan]:
    plans = [
        TensorPlan(name, tensor.dtype, tensor.shape, (tensor,))
        for name, tensor in source.tensors.items()
        if name.startswith("mtp.")
    ]
    plans.sort(key=lambda plan: plan.name)
    if not plans:
        raise ValueError("source checkpoint has no mtp.* tensors")
    return plans


def _dspark_config(target_config: dict, source_config: dict) -> dict:
    source_text = source_config.get("text_config", source_config)
    target_text = target_config.get("text_config", target_config)
    config = {**target_config, **target_text}
    config["model_type"] = "deepseek_v4"
    config["architectures"] = ["DeepseekV4ForCausalLM"]
    dspark_keys = (
        "dspark_block_size",
        "dspark_markov_rank",
        "dspark_n_routed_experts",
        "dspark_noise_token_id",
        "dspark_num_experts_per_tok",
        "dspark_target_layer_ids",
        "num_nextn_predict_layers",
    )
    for key in dspark_keys:
        config[key] = source_text[key]
    config["text_config"] = {
        **target_text,
        **{key: source_text[key] for key in dspark_keys},
    }
    config["rapid_quantization"] = {
        **target_config.get("rapid_quantization", {}),
        "mtp": True,
        "mtp_source": "checkpoint-native-dspark",
    }
    return config


def _inference_config(config: dict) -> dict:
    return {
        "n_mtp_layers": int(config["num_nextn_predict_layers"]),
        "dspark_block_size": int(config["dspark_block_size"]),
        "dspark_noise_token_id": int(config["dspark_noise_token_id"]),
        "dspark_target_layer_ids": list(config["dspark_target_layer_ids"]),
        "dspark_markov_rank": int(config["dspark_markov_rank"]),
        "dspark_n_routed_experts": int(config["dspark_n_routed_experts"]),
        "dspark_num_experts_per_tok": int(config["dspark_num_experts_per_tok"]),
    }


def _copy_metadata(target: Path, destination: Path) -> None:
    for name in (
        "LICENSE",
        "NOTICE",
        "chat_template.jinja",
        "engram_token_map.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        source = target / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _safe_shard_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid target shard name: {value!r}")
    shard = Path(value)
    if shard.is_absolute() or shard.name != value:
        raise ValueError(f"target shard must be a basename: {value!r}")
    return value


def build_overlay(source: Path, target: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_config = _read_json(source / "config.json")
    target_config = _read_json(target / "config.json")
    target_index = _read_json(target / "model.safetensors.index.json")
    target_weights = target_index.get("weight_map")
    if not isinstance(target_weights, dict) or not target_weights:
        raise ValueError("target checkpoint has no indexed weights")
    target_shards = {_safe_shard_name(value) for value in target_weights.values()}
    mtp_shard = "model-mtp.safetensors"
    if mtp_shard in target_shards:
        raise ValueError(f"target checkpoint already uses reserved shard {mtp_shard}")

    plans = _mtp_plans(CheckpointIndex(source))
    target_names = set(target_weights)
    generated_names = {plan.name for plan in plans}
    collisions = sorted(target_names & generated_names)
    if collisions:
        preview = ", ".join(collisions[:3])
        raise ValueError(f"target checkpoint already contains MTP tensors: {preview}")
    mtp_bytes = sum(plan.nbytes for plan in plans)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    owns_destination = False
    try:
        for shard in sorted(target_shards):
            source_shard = target / shard
            if source_shard.is_symlink():
                raise ValueError(f"target shard must not be a symlink: {source_shard}")
            if not source_shard.is_file():
                raise FileNotFoundError(source_shard)
            relative_target = os.path.relpath(source_shard, staging)
            (staging / shard).symlink_to(relative_target)

        write_safetensors(staging / mtp_shard, plans)
        weight_map = dict(target_weights)
        weight_map.update({plan.name: mtp_shard for plan in plans})
        config = _dspark_config(target_config, source_config)
        inference = _inference_config(config)

        (staging / "inference").mkdir()
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "inference" / "config.json").write_text(
            json.dumps(inference, indent=2) + "\n"
        )
        total_size = int(target_index.get("metadata", {}).get("total_size", 0))
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_size + mtp_bytes},
                    "weight_map": weight_map,
                },
                indent=2,
            )
            + "\n"
        )
        _copy_metadata(target, staging)
        (staging / "README.md").write_text(
            "# Experimental DeepSeek V4.1 REAP + DSpark overlay\n\n"
            "This local benchmark artifact reuses the target checkpoint via "
            "relative symlinks and adds its checkpoint-native three-stage "
            "DSpark draft network. It is not a standalone distribution.\n\n"
            f"- Added DSpark tensor bytes: {mtp_bytes:,}\n"
            f"- Target artifact: `{target.name}`\n"
        )
        # mkdir is the portable atomic no-replace reservation for a directory.
        # Once it succeeds this invocation owns the directory, so a failed
        # publication can safely remove only that directory and be retried.
        destination.mkdir()
        owns_destination = True
        incomplete = destination / ".rapid-overlay-incomplete"
        incomplete.touch(exist_ok=False)
        for child in staging.iterdir():
            child.rename(destination / child.name)
        staging.rmdir()
        incomplete.unlink()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if owns_destination and destination.exists():
            shutil.rmtree(destination)
        raise
    return {
        "destination": str(destination),
        "target_shards": len(target_shards),
        "mtp_tensors": len(plans),
        "mtp_bytes": mtp_bytes,
        "mtp_gb": round(mtp_bytes / 1e9, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_overlay(
        args.source.resolve(), args.target.resolve(), args.destination.resolve()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
