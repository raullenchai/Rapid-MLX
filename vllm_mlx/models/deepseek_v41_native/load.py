"""Load a converted DeepSeek-V4.1 MLX build — strictly.

``mlx_lm.load`` uses ``strict=False`` and lets missing weights silently stay
random-initialized (the GLM-5.2 lesson); this loader refuses that:

* every parameter the model owns must come from the checkpoint;
* every checkpoint tensor must either land in a parameter or belong to the
  explicitly-declared vision passthrough (``vision.*`` / ``aligner.*`` /
  ``image_*`` — carried by the conversion for a future VL runtime, reported,
  never silently ignored);
* quantization is replayed from ``config.json``: the converter writes an
  explicit per-module map (``quantization`` carries defaults plus
  ``<module path>: {group_size, bits} | false`` entries keyed by
  runtime-internal module paths); when the map is absent the loader falls back
  to the same ``is_quant_target`` / ``bits_for`` rules the converter used.

Builds bundle **nothing executable**: no ``model_file`` mechanism exists here —
a build is loaded through this package
(``vllm_mlx.models.deepseek_v41_native.load``), never through a checkpoint
Python module.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelArgs
from .convert import VISION_PREFIXES, bits_for, is_quant_target
from .model import Model


def resolve_indexed_shard(model_path: str, filename: str) -> str:
    """Resolve an indexed shard without escaping its model/CAS repository."""
    root = Path(model_path).resolve()
    lexical = Path(os.path.abspath(os.path.join(root, filename)))
    if os.path.commonpath((root, lexical)) != str(root):
        raise ValueError("Engram shard must stay inside the model directory")
    resolved = lexical.resolve(strict=True)
    if os.path.commonpath((root, resolved)) == str(root):
        return str(resolved)
    # Hub snapshots legitimately symlink immutable files into the specific
    # repository's sibling blobs directory. Recognize the repository layout,
    # rather than trusting any parent directory merely named "snapshots".
    repository = root.parent.parent
    if root.parent.name == "snapshots" and repository.name.startswith("models--"):
        lexical_blobs = repository / "blobs"
        blobs = lexical_blobs.resolve()
        if (
            lexical_blobs.is_dir()
            and not lexical_blobs.is_symlink()
            and os.path.commonpath((repository, blobs)) == str(repository)
            and os.path.commonpath((blobs, resolved)) == str(blobs)
        ):
            return str(resolved)
    raise ValueError("Engram shard symlink escapes the model repository")


def supports_engram_ssd_offload(path: str) -> bool:
    """Return whether a checkpoint has the complete supported Engram layout."""
    try:
        with open(os.path.join(path, "config.json")) as config_file:
            config = json.load(config_file)
        args = ModelArgs.from_dict(config)
        quantization = config.get("quantization")
        if (
            not args.engram_layer_ids
            or not isinstance(quantization, dict)
            or not quantization.get("engram_bits")
        ):
            return False
        with open(os.path.join(path, "model.safetensors.index.json")) as index_file:
            index = json.load(index_file)
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict):
            return False
        for layer_id in args.engram_layer_ids:
            prefix = f"layers.{layer_id}.engram.embed"
            keys = (prefix + ".weight", prefix + ".scales", prefix + ".biases")
            filenames = [mapping.get(key) for key in keys]
            if (
                any(
                    not isinstance(filename, str) or not filename
                    for filename in filenames
                )
                or len(set(filenames)) != 1
            ):
                return False
            filename = filenames[0]
            assert isinstance(filename, str)
            resolve_indexed_shard(path, filename)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def reshape_grouped_wo_a(items, args: ModelArgs):
    """Restore the logical group axis from flat release/checkpoint storage."""
    output = []
    suffixes = (".attn.wo_a.weight", ".attn.wo_a.scales", ".attn.wo_a.biases")
    expected_rows = args.o_groups * args.o_lora_rank
    for name, value in items:
        if name.endswith(suffixes) and value.ndim == 2:
            if value.shape[0] != expected_rows:
                raise ValueError(
                    f"{name} has {value.shape[0]} rows; expected {expected_rows} "
                    "for grouped wo_a"
                )
            value = value.reshape(args.o_groups, args.o_lora_rank, -1)
        output.append((name, value))
    return output


def quant_predicate(
    group_size: int, bits: int, expert_bits: int | None, module_map: dict | None = None
):
    """Per-module predicate. With ``module_map`` (the converter's explicit map,
    keyed by runtime-internal module paths) the map is authoritative; without
    it, fall back to the converter's rule set."""

    def pred(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module_map is not None:
            entry = module_map.get(path, False)
            return dict(entry) if entry else False
        w = getattr(module, "weight", None)
        if w is None:
            return False
        name = path + ".weight"
        if not is_quant_target(name, w.shape[-1], group_size):
            return False
        return {"group_size": group_size, "bits": bits_for(name, bits, expert_bits)}

    return pred


def load_token_map(path: str, args: ModelArgs, cache_dir: str | None = None):
    """The engram compressed-token map, built from the release tokenizer.

    Cached (building it decodes+normalizes all 129,280 tokens, ~a minute) in
    ``cache_dir`` when given — use that when ``path`` must stay read-only —
    else beside the tokenizer. Returns None when no tokenizer is present.
    """
    cache_file = os.path.join(cache_dir or path, "engram_token_map.json")
    if os.path.exists(cache_file):
        return json.load(open(cache_file))
    tok_file = os.path.join(path, "tokenizer.json")
    if not os.path.exists(tok_file):
        return None
    from transformers import PreTrainedTokenizerFast

    from .engram import build_compressed_token_map

    tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    token_map, n = build_compressed_token_map(tok)
    if args.engram_compressed_vocab_size and n != args.engram_compressed_vocab_size:
        raise ValueError(
            f"compressed vocab {n} != config {args.engram_compressed_vocab_size}"
        )
    json.dump(token_map, open(cache_file, "w"))
    return token_map


def load(
    path: str,
    lazy: bool | None = None,
    strict: bool = True,
    fuse_moe_gate_up: bool = False,
    engram_ssd_offload: bool = False,
):
    """``lazy=None`` auto-selects: builds that fit comfortably in physical RAM
    are materialized at load time (CPU-side, avoiding cold-mmap page-in inside
    a GPU command buffer — that trips Metal's watchdog); builds larger than
    ~80% of RAM stay mmap-lazy, because materialized buffers are unevictable
    and drive the machine into compressor thrash, while clean file-backed
    pages stream from disk per forward (measured: a 476 GB build on 512 GiB
    stalled the box when materialized, runs lazily)."""
    try:
        requested_gb = os.environ.get("DSV41_WIRED_GB")
        if requested_gb is None:
            # Never inherit the upstream 470 GB assumption on a 256 GiB Mac.
            # Metal's own budget leaves the OS enough operational headroom.
            wired = int(mx.device_info()["max_recommended_working_set_size"])
        else:
            wired = int(float(requested_gb) * 1e9)
        mx.set_wired_limit(wired)
    except (KeyError, RuntimeError, TypeError, ValueError):
        pass
    if os.environ.get("DSV41_LAZY") in ("0", "1"):
        lazy = os.environ["DSV41_LAZY"] == "1"
        print(f"[load] lazy={lazy} forced via DSV41_LAZY", flush=True)
    if lazy is None:
        size = sum(
            os.path.getsize(p) for p in glob.glob(os.path.join(path, "*.safetensors"))
        )
        try:
            phys = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            phys = int(550e9)
        lazy = size > 0.8 * phys
        if lazy:
            print(
                f"[load] {size / 1e9:.0f} GB build vs {phys / 1e9:.0f} GB RAM: "
                f"staying mmap-lazy (materializing would thrash the compressor)",
                flush=True,
            )
            # Prewarm the page cache sequentially (clean, evictable pages) so
            # the first forward's page-ins are cache hits — cold-disk faults
            # inside a GPU command buffer trip Metal's ~watchdog.
            import time as _time

            t0 = _time.time()
            buf = bytearray(1 << 28)
            for p in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                with open(p, "rb", buffering=0) as fh:
                    while fh.readinto(buf):
                        pass
            print(
                f"[load] page-cache prewarm: {size / 1e9:.0f} GB in "
                f"{_time.time() - t0:.0f}s",
                flush=True,
            )
    cfg = json.load(open(os.path.join(path, "config.json")))
    args = ModelArgs.from_dict(cfg)
    token_map = load_token_map(path, args) if args.engram_layer_ids else None
    model = Model(args, token_map=token_map)

    q = cfg.get("quantization")
    engram_keys: set[str] = set()
    engram_mapping: dict[str, object] = {}
    if engram_ssd_offload:
        index_path = os.path.join(path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise ValueError("Engram SSD offload requires a safetensors index")
        with open(index_path) as index_file:
            index = json.load(index_file)
        raw_mapping = index.get("weight_map")
        if not isinstance(raw_mapping, dict):
            raise ValueError("Engram SSD offload requires a valid weight map")
        engram_mapping = raw_mapping
    if q:
        if q.get("bits"):
            module_map = q.get("modules")
            nn.quantize(
                model,
                group_size=q["group_size"],
                bits=q["bits"],
                class_predicate=quant_predicate(
                    q["group_size"], q["bits"], q.get("expert_bits"), module_map
                ),
            )
        if q.get("engram_bits"):
            from .engram import (
                DiskQuantizedEngramEmbedding,
                QuantizedEngramEmbedding,
            )

            for layer in model.layers:
                if layer.engram is not None:
                    e = layer.engram.embed
                    if engram_ssd_offload:
                        prefix = f"layers.{layer.layer_id}.engram.embed"
                        keys = {
                            "weight": prefix + ".weight",
                            "scales": prefix + ".scales",
                            "biases": prefix + ".biases",
                        }
                        filenames = []
                        for key in keys.values():
                            filename = engram_mapping.get(key)
                            if not isinstance(filename, str) or not filename:
                                raise ValueError(
                                    "Engram SSD offload is missing indexed tensor: "
                                    f"{key}"
                                )
                            filenames.append(filename)
                        if len(set(filenames)) != 1:
                            raise ValueError(
                                "Engram affine tensors must share one shard"
                            )
                        filename = filenames[0]
                        shard_path = resolve_indexed_shard(path, filename)
                        layer.engram.embed = DiskQuantizedEngramEmbedding(
                            shard_path,
                            weight_key=keys["weight"],
                            scales_key=keys["scales"],
                            biases_key=keys["biases"],
                            num_embeddings=e.weight.shape[0],
                            dim=e.weight.shape[1],
                            group_size=q["group_size"],
                            bits=q["engram_bits"],
                        )
                        engram_keys.update(keys.values())
                    else:
                        layer.engram.embed = QuantizedEngramEmbedding(
                            e.weight.shape[0],
                            e.weight.shape[1],
                            q["group_size"],
                            q["engram_bits"],
                        )
    if engram_ssd_offload and args.engram_layer_ids and not engram_keys:
        raise ValueError("Engram SSD offload requires affine-quantized Engram tables")

    loaded: set[str] = set()
    vision_skipped: set[str] = set()
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        w = mx.load(shard)
        items = []
        for k, v in w.items():
            if k in engram_keys:
                continue
            if k.startswith(VISION_PREFIXES):
                vision_skipped.add(k)  # declared passthrough, text-only runtime
                continue
            items.append((k, v))
        items = reshape_grouped_wo_a(items, args)
        model.load_weights(items, strict=False)
        if not lazy:
            mx.eval([v for _, v in items])
        loaded.update(k for k, _ in items)
        del w

    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing = expected - loaded
    unexpected = loaded - expected
    if strict and missing:
        raise ValueError(
            f"{len(missing)} params missing from checkpoint, e.g. {sorted(missing)[:4]}"
        )
    if strict and unexpected:
        raise ValueError(
            f"{len(unexpected)} checkpoint tensors have no home, "
            f"e.g. {sorted(unexpected)[:4]}"
        )
    model.eval()
    model._vision_passthrough = sorted(vision_skipped)
    if fuse_moe_gate_up:
        # Reuse Rapid's qualified SwitchGLU fusion instead of maintaining a
        # V4.1-only copy. Keep it opt-in until this model passes its end-to-end
        # throughput and greedy-equivalence qualification gate.
        from vllm_mlx.moe_fusion import fuse_gate_up

        fused = fuse_gate_up(model)
        if fused != args.n_layers:
            raise RuntimeError(
                f"fused {fused} MoE layers; expected all {args.n_layers}"
            )
    return model, args
