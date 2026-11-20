# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 conversion with bounded native safetensors output."""

import json
import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .hy_v4_flash_config import (
    ModelArgs,
    checkpoint_key,
    dequantize_fp8,
    parameter_shapes,
    quantization_policy,
    validate_lookup,
)

_CHECKPOINT_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "NOTICE",
    "NOTICE.txt",
    "NOTICE.md",
)


def _strip_remote_code_metadata(value):
    if isinstance(value, dict):
        for key in list(value):
            if key in ("auto_map", "code_revision") or "trust_remote_code" in key:
                del value[key]
            else:
                _strip_remote_code_metadata(value[key])
    elif isinstance(value, list):
        for item in value:
            _strip_remote_code_metadata(item)


class _ShardWriter:
    """Bound output residency; retain only names after each synchronous save.

    Like the streamed build writer, publish the index only after all tensors
    have been written. An indivisible tensor larger than the limit gets its
    own shard.
    """

    def __init__(self, destination, max_bytes=5_000_000_000):
        self.destination = Path(destination)
        self.destination.mkdir(exist_ok=True)
        self.max_bytes = max_bytes
        self.pending = {}
        self.size = self.total = 0
        self.weight_map = {}
        self.shards = 0

    def put(self, name, value):
        if name in self.weight_map or name in self.pending:
            raise ValueError(f"Duplicate Hy4 Flash weight: {name}")
        if self.size + value.nbytes > self.max_bytes:
            self.flush()
        self.pending[name] = value
        self.size += value.nbytes
        self.total += value.nbytes
        if value.nbytes > self.max_bytes:
            self.flush()

    def flush(self):
        import mlx.core as mx

        if not self.pending:
            return
        self.shards += 1
        filename = f"model-{self.shards:05d}.safetensors"
        mx.save_safetensors(
            str(self.destination / filename), self.pending, metadata={"format": "mlx"}
        )
        self.weight_map.update(dict.fromkeys(self.pending, filename))
        self.pending.clear()
        self.size = 0

    def finish(self, total_parameters):
        self.flush()
        (self.destination / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": self.total,
                        "total_parameters": total_parameters,
                    },
                    "weight_map": self.weight_map,
                },
                indent=4,
                sort_keys=True,
            )
        )


def convert_fp8(source, destination, bits, group_size):
    """Return False for ordinary checkpoints; consume raw FP8 before mx.load."""
    import mlx.core as mx
    from mlx_lm.utils import create_model_card, save_config

    from ..fp8_repack import _open_shards, _raw_to_mx
    from .hy_v4_flash import Model

    source, destination = Path(source), Path(destination)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "hy_v4_flash":
        return False
    weight_map, shards = _open_shards(source)
    if not any(
        entry["dtype"] == "F8_E4M3"
        for shard in shards.values()
        for entry in shard.header.values()
    ):
        return False
    if destination.exists():
        raise ValueError("Conversion destination already exists")
    args = ModelArgs.from_dict(config)
    expected = parameter_shapes(args)
    block = tuple(
        config.get(
            "quantization_config",
            config.get("text_config", {}).get("quantization_config", {}),
        ).get("weight_block_size", [128, 128])
    )
    config.pop("quantization_config", None)
    config.get("text_config", {}).pop("quantization_config", None)
    # Match quantize_model's default metadata and per-module policy, without
    # instantiating or quantizing randomly initialized model parameters.
    quantization = {"group_size": 32, "bits": bits, "mode": "affine"}
    policies = {}
    for name, shape in expected.items():
        if name.endswith(".weight") and len(shape) >= 2 and shape[-1] % 32 == 0:
            path = name.removesuffix(".weight")
            policy = quantization_policy(path, None, group_size=group_size, bits=bits)
            if policy:
                if shape[-1] % policy["group_size"] or shape[-1] * bits % 32:
                    raise ValueError(f"Invalid affine dimensions: {name}")
                policies[path] = policy
                quantization[path] = policy
    config["quantization"] = quantization
    _strip_remote_code_metadata(config)
    cast_predicate = Model.cast_predicate.fget(None)
    lookup = None
    total_parameters = 0

    def read(name):
        raw, dtype, shape = shards[weight_map[name]].read(name)
        if dtype == "F8_E4M3":
            scale = np.array(read(name + "_scale_inv").astype(mx.float32))
            return mx.array(dequantize_fp8(raw.reshape(shape), scale, block))
        if dtype in ("I64", "I32"):
            return mx.array(
                raw.view(np.int64 if dtype == "I64" else np.int32).reshape(shape)
            )
        return _raw_to_mx(raw, dtype, shape)

    # Group names only, so arbitrarily interleaved source entries cannot retain
    # multiple unfinished expert banks. Preserve complete-bank assembly.
    per_expert = re.compile(r"(.*\.switch_mlp\.)(\d+)(\..*)")
    groups = {}
    for name in weight_map:
        target = checkpoint_key(name)
        if target is None:
            logging.getLogger(__name__).info(
                "Hy4 Flash text-only: skipping vision tower and dfly_layers weights"
            )
            continue
        if name.endswith("_scale_inv"):
            continue
        match = per_expert.fullmatch(target)
        bank = match.group(1) + match.group(3) if match else name
        groups.setdefault(bank, []).append(name)

    def convert_group(names):
        nonlocal lookup, total_parameters
        mapped = {}
        for name in names:
            target = checkpoint_key(name)
            if args.tie_word_embeddings and target == "lm_head.weight":
                continue
            value = read(name)
            if not cast_predicate(target) and mx.issubdtype(value.dtype, mx.floating):
                value = value.astype(mx.float32)
            if target.endswith(".engram_layer.lookup_table"):
                current = validate_lookup(np.array(value), args.vocab_size)
                if lookup is not None and not np.array_equal(lookup, current):
                    raise ValueError("Engram layers must share the same token map")
                lookup = current
            if target in mapped:
                raise ValueError(f"Duplicate Hy4 Flash weight: {target}")
            mapped[target] = value
        for key in list(mapped):
            if ".switch_mlp.gate_up_proj." in key:
                targets = [
                    key.replace(".gate_up_proj.", f".{projection}.")
                    for projection in ("gate_proj", "up_proj")
                ]
                for target in targets:
                    if target in mapped:
                        raise ValueError(f"Duplicate Hy4 Flash weight: {target}")
                gate, up = mx.split(mapped.pop(key), 2, axis=-2)
                mapped[targets[0]] = gate
                mapped[targets[1]] = up
        for key in list(mapped):
            match = per_expert.fullmatch(key)
            if match is None or key not in mapped:
                continue
            bank = match.group(1) + match.group(3).removeprefix(".")
            expert_keys = [
                match.group(1) + str(e) + match.group(3)
                for e in range(args.n_routed_experts)
            ]
            if any(k not in mapped for k in expert_keys):
                raise ValueError(
                    f"Hy4 checkpoint has incomplete routed-expert banks: {bank}"
                )
            mapped[bank] = mx.stack([mapped.pop(k) for k in expert_keys])
        for key in list(mapped):
            value = mapped.pop(key)
            shape = expected.get(key)
            # Frozen lookup maps may cover only the compressed vocabulary.
            valid_shape = shape == value.shape or (
                key.endswith(".engram_layer.lookup_table")
                and shape is not None
                and value.ndim == 1
                and 0 < value.shape[0] <= shape[0]
            )
            if not valid_shape:
                raise ValueError(
                    f"Unexpected Hy4 tensor name or shape: {key}: {value.shape}, expected {shape}"
                )
            total_parameters += value.size
            path = key.removesuffix(".weight")
            if key.endswith(".weight") and path in policies:
                params = policies[path]
                triple = mx.quantize(
                    value, group_size=params["group_size"], bits=params["bits"]
                )
                mx.eval(*triple)
                for leaf, tensor in zip(
                    ("weight", "scales", "biases"), triple, strict=True
                ):
                    writer.put(path + "." + leaf, tensor)
                del triple, tensor
            else:
                mx.eval(value)
                writer.put(key, value)

    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        writer = _ShardWriter(staging)
        for names in groups.values():
            convert_group(names)
        # Validate completeness using names only; saved tensors are never reloaded.
        required = set(expected)
        for path in policies:
            required.update((path + ".scales", path + ".biases"))
        available = writer.weight_map.keys() | writer.pending.keys()
        if required != available:
            raise ValueError(
                f"Missing Hy4 checkpoint tensors: {sorted(required - available)}"
            )
        writer.finish(total_parameters)
        save_config(config, config_path=staging / "config.json")
        for name in _CHECKPOINT_ARTIFACTS:
            file = source / name
            if file.is_file():
                shutil.copy(file, staging)
        create_model_card(staging, None)
        staging.rename(destination)
    return True
