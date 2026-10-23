#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze existing catalog identities using HF metadata only (never weights).

This records source identity and executable-config declarations, not model
quality or a security review. Existing pins win, then cached refs/main, then the
Hub's current SHA. Re-run with --output in scratch for a reviewable refresh.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import get_session

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{40}\Z")
DEFAULTS = (
    "mlx-community/sam-audio-large-fp16",
    "mlx-community/sam-audio-large",
    "mlx-community/sam-audio-small-fp16",
    "mlx-community/sam-audio-small",
    "t5-base",
    "lucasnewman/vocos-mel-24khz",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-tiny",
)


def targets():
    records = {}

    def add(repo, pin=None):
        if not isinstance(repo, str) or not repo:
            return
        item = records.setdefault(repo, {"pins": set(), "subfolders": set()})
        if pin:
            item["pins"].add(pin)

    for filename, field in (
        ("aliases.json", "hf_path"),
        ("audio/aliases.json", "hf_id"),
    ):
        catalog = json.loads((ROOT / "rapid_mlx" / filename).read_text())
        for alias, item in catalog.items():
            if alias.startswith("_"):
                continue
            if isinstance(item, str):
                add(item)
                continue
            repo = item[field]
            pin = (
                item.get("revision")
                or item.get("dflash_target_revision")
                or item.get("tensorfold_target_revision")
            )
            add(repo, pin)
            if item.get("subfolder"):
                records[repo]["subfolders"].add(item["subfolder"])
            for key, value in item.items():
                if key.endswith("_model") and isinstance(value, str) and "/" in value:
                    add(value, item.get(key.replace("_model", "_revision")))
        for assets in catalog.get("_runtime_assets", {}).values():
            for asset in assets:
                add(asset["repo_id"], asset.get("revision"))
    # Preserve already reviewed runtime-specific image/video identities too.
    from rapid_mlx._download_gate import (
        IMAGE_MODEL_REVISIONS,
        IMAGE_MODEL_RUNTIME_ASSETS,
    )
    from rapid_mlx.video.wan import WAN_REVISIONS

    for repo, pin in (IMAGE_MODEL_REVISIONS | WAN_REVISIONS).items():
        add(repo, pin)
    for assets in IMAGE_MODEL_RUNTIME_ASSETS.values():
        for repo, pin, _patterns in assets:
            add(repo, pin)
    for repo in DEFAULTS:
        add(repo)
    return records


def inspect(repo, selection):
    pins = sorted(selection["pins"])
    if len(pins) > 1:
        raise ValueError(f"Conflicting existing pins for {repo}: {pins}")
    pin = pins[0] if pins else None
    source = "existing_catalog_pin" if pin else "hub_metadata"
    ref = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    cached = ref.read_text().strip() if ref.is_file() else None
    if not pin and cached and SHA.fullmatch(cached):
        pin, source = cached, "cached_main_ref"
    info = HfApi().model_info(repo, revision=pin, files_metadata=True)
    revision = info.sha
    if not revision or not SHA.fullmatch(revision):
        raise ValueError(f"No immutable revision for {repo}")
    filenames = {entry.rfilename for entry in info.siblings}
    configs = {}
    for folder in ["", *sorted(selection["subfolders"])]:
        for name in ("config.json", "tokenizer_config.json"):
            filename = f"{folder}/{name}" if folder else name
            if filename not in filenames:
                continue
            response = get_session().get(
                hf_hub_url(repo, filename, revision=revision), timeout=45
            )
            response.raise_for_status()
            raw = response.content
            value = json.loads(raw)
            configs[filename] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "model_type": value.get("model_type"),
                "model_file": value.get("model_file"),
                "auto_map": value.get("auto_map"),
                "text_encoder": value.get("text_encoder"),
            }
    files = {}
    for entry in info.siblings:
        if entry.lfs:
            files[entry.rfilename] = {"sha256": entry.lfs.sha256, "size": entry.size}
    return {
        "revision": revision,
        "selection": source,
        "cached_main_ref": cached,
        "configs": configs,
        "lfs_files": files,
        "git_files": {entry.rfilename: {"blob_id": entry.blob_id, "size": entry.size}
                      for entry in info.siblings if not entry.lfs},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    wanted = targets()
    previous = json.loads(args.output.read_text()) if args.output.exists() else {}
    results = previous.get("repositories", {})
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(inspect, repo, value): repo
            for repo, value in wanted.items()
            if repo not in results
        }
        for future in concurrent.futures.as_completed(pending):
            repo = pending[future]
            try:
                results[repo] = future.result()
                print(f"OK {repo} {results[repo]['revision']}", flush=True)
            except Exception as exc:
                failures[repo] = str(exc)
                print(f"FAILED {repo}: {type(exc).__name__}", flush=True)
            args.output.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "scope": "Metadata identity lock, not model-quality or executable-code approval",
                        "repositories": dict(sorted(results.items())),
                        "errors": failures,
                    },
                    indent=2,
                )
                + "\n"
            )
    print(f"Locked {len(results)}/{len(wanted)} repositories; failures={len(failures)}")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
