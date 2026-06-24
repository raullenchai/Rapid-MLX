#!/usr/bin/env python3
"""
Idempotent vision-strip scrubber for Tmax (Qwen3.5 multimodal) bases.

Tmax 9B/27B bases ship with `model_type: qwen3_5` (multimodal) but currently
contain ZERO vision-tower tensors in their safetensors. The plan is to publish
text-only MLX variants. This script makes a snapshot directory loadable by
`mlx_lm` (which already drops `vision_tower`/`model.visual` keys in its
`qwen3_5.Model.sanitize`) and by quantizers that don't expect a vision config.

Operations (all skipped if already done):

1. Strip `vision_config`, `image_token_id`, `video_token_id`,
   `vision_start_token_id`, `vision_end_token_id` from `config.json`.
   If `architectures` is `Qwen3_5ForConditionalGeneration`, rewrite to
   `Qwen3_5ForCausalLM` (text-only equivalent).
2. For every safetensors shard: drop any tensor whose key starts with a
   probed vision prefix (default: `vision_tower.`, `model.visual.`, `visual.`).
   In Tmax bases there are NONE — this is a defensive no-op.
3. If `model.safetensors.index.json` exists, prune the weight_map and recompute
   `metadata.total_size`.

Usage:
    python scripts/tmax_strip_vision.py <snapshot_dir> [--prefix vision_tower. ...]

The script is idempotent: running it twice is a no-op the second time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as save_numpy  # type: ignore
except ImportError:  # pragma: no cover
    safe_open = None
    save_numpy = None

DEFAULT_VISION_PREFIXES = ("vision_tower.", "model.visual.", "visual.")
VISION_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


def _strip_config(cfg_path: Path) -> bool:
    cfg = json.loads(cfg_path.read_text())
    changed = False
    for k in VISION_CONFIG_KEYS:
        if k in cfg:
            cfg.pop(k)
            changed = True
    archs = cfg.get("architectures") or []
    new_archs = []
    for a in archs:
        if a == "Qwen3_5ForConditionalGeneration":
            new_archs.append("Qwen3_5ForCausalLM")
            changed = True
        else:
            new_archs.append(a)
    if new_archs != archs:
        cfg["architectures"] = new_archs
    # language_model_only is a hint flag — drop it if present, it's no longer needed
    if "language_model_only" in cfg:
        cfg.pop("language_model_only")
        changed = True
    if changed:
        cfg_path.write_text(json.dumps(cfg, indent=2))
    return changed


def _key_has_vision_prefix(key: str, prefixes: Iterable[str]) -> bool:
    return any(key.startswith(p) for p in prefixes)


def _strip_shard(shard_path: Path, prefixes: Iterable[str]) -> tuple[int, int]:
    """Return (kept_count, dropped_count). Rewrites in place only if drops > 0.

    Uses safetensors' numpy framework so we don't need a torch dependency just
    to drop tensors. Note: bf16 round-trips via numpy's view-as-bytes path that
    safetensors handles internally, but to avoid any dtype coercion we only
    rewrite when we actually need to drop tensors.
    """
    if safe_open is None or save_numpy is None:
        raise RuntimeError("safetensors (with numpy) not installed")
    # First pass: just list keys to decide whether we even need to rewrite.
    dropped: list[str] = []
    kept_keys: list[str] = []
    with safe_open(str(shard_path), framework="numpy") as f:
        for k in f:
            if _key_has_vision_prefix(k, prefixes):
                dropped.append(k)
            else:
                kept_keys.append(k)
    if not dropped:
        return (len(kept_keys), 0)
    # Need to rewrite. Load only the kept tensors (numpy view) and save out.
    kept: dict = {}
    with safe_open(str(shard_path), framework="numpy") as f:
        for k in kept_keys:
            kept[k] = f.get_tensor(k)
    tmp = shard_path.with_suffix(".safetensors.tmp")
    save_numpy(kept, str(tmp))
    shutil.move(str(tmp), str(shard_path))
    return (len(kept_keys), len(dropped))


def _prune_index(index_path: Path, prefixes: Iterable[str], dir_path: Path) -> bool:
    idx = json.loads(index_path.read_text())
    wm = idx.get("weight_map", {})
    new_wm = {k: v for k, v in wm.items() if not _key_has_vision_prefix(k, prefixes)}
    if new_wm == wm:
        return False
    idx["weight_map"] = new_wm
    # recompute total_size
    total = 0
    seen_shards: set[str] = set()
    for shard in new_wm.values():
        if shard in seen_shards:
            continue
        seen_shards.add(shard)
        total += (dir_path / shard).stat().st_size
    if "metadata" not in idx:
        idx["metadata"] = {}
    idx["metadata"]["total_size"] = total
    index_path.write_text(json.dumps(idx, indent=2))
    return True


def strip(
    snapshot_dir: Path, prefixes: Iterable[str] = DEFAULT_VISION_PREFIXES
) -> dict:
    snapshot_dir = Path(snapshot_dir)
    report: dict = {"dir": str(snapshot_dir), "shards": []}

    cfg = snapshot_dir / "config.json"
    if cfg.exists():
        report["config_changed"] = _strip_config(cfg)
    else:
        report["config_changed"] = False

    shards = sorted(snapshot_dir.glob("*.safetensors"))
    total_dropped = 0
    for shard in shards:
        kept, dropped = _strip_shard(shard, prefixes)
        total_dropped += dropped
        report["shards"].append({"shard": shard.name, "kept": kept, "dropped": dropped})
    report["total_dropped"] = total_dropped

    idx = snapshot_dir / "model.safetensors.index.json"
    if idx.exists():
        report["index_changed"] = _prune_index(idx, prefixes, snapshot_dir)
    else:
        report["index_changed"] = False

    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("snapshot_dir")
    p.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Vision tensor key prefix (repeatable). Default: vision_tower. model.visual. visual.",
    )
    args = p.parse_args(argv)
    prefixes = args.prefix or list(DEFAULT_VISION_PREFIXES)
    rep = strip(Path(args.snapshot_dir), prefixes)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
