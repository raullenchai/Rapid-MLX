# SPDX-License-Identifier: Apache-2.0
"""Detect which model aliases have weights present locally.

The benchmark tier uses this to skip aliases whose weights would
otherwise trigger a multi-GB download mid-run.  We deliberately don't
*download* anything — the user is expected to pre-fetch what they want
benchmarked.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelAvailability:
    alias: str
    repo_id: str
    available: bool
    path: Path | None = None
    reason: str = ""  # only set when available=False


def load_aliases() -> dict[str, str]:
    """Return the alias → repo-id mapping shipped with the package.

    Thin wrapper over ``vllm_mlx.model_aliases.list_aliases`` so the
    doctor harness sees the same view regardless of whether the entry
    in ``aliases.json`` is the legacy bare-string form or the rich
    profile form (this module pre-dates the schema bump).
    """
    from ..model_aliases import list_aliases as _list

    return _list()


def _hf_cache_roots() -> list[Path]:
    """Return all candidate HF cache directories on this host.

    HF_HUB_CACHE wins (this is what the doctor sets in the spawned
    server's env).  ~/.cache/huggingface/hub is the standard fallback.
    LM Studio's cache is also checked because users often have models
    there from prior LM Studio installs.
    """
    roots: list[Path] = []
    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        roots.append(Path(env_cache))
    env_home = os.environ.get("HF_HOME")
    if env_home:
        roots.append(Path(env_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / ".lmstudio" / "models")
    # Dedupe while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r and r not in seen:
            out.append(r)
            seen.add(r)
    return out


def _repo_to_hf_dirname(repo_id: str) -> str:
    """HF cache uses 'models--{org}--{repo}' as the directory name."""
    return "models--" + repo_id.replace("/", "--")


def _is_complete_snapshot(snap: Path) -> bool:
    """A snapshot is "complete" if it has config.json AND every weight file resolves.

    Why "every" matters: HF interrupts mid-download can leave a snapshot
    with config.json, the index file, and *some* shard symlinks (the
    ones that finished) — but not all.  Loading such a snapshot crashes
    the server with "No safetensors found" or a torch/mlx-lm error
    halfway through deserialization, both of which surface in the
    scorecard as misleading runtime failures rather than the correct
    "skipped (partial download)" diagnosis.

    Strategy:

    1. If ``model.safetensors.index.json`` is present, parse its
       ``weight_map`` and require every referenced shard to exist
       (and resolve, in case it's a dangling symlink into ../blobs/).
    2. Otherwise, require at least one resolvable ``*.safetensors`` /
       ``*.npz`` / ``*.gguf`` file — a single-file model needs only one,
       and we shouldn't require an index for those.

    Either way: ``resolve(strict=True)`` is what catches dangling
    symlinks, which a plain ``.exists()`` check misses on macOS.
    """
    if not (snap / "config.json").exists():
        return False

    # 1. Sharded model — the index lists every shard we need.
    index_path = snap / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (OSError, json.JSONDecodeError):
            # The index exists but we can't trust it — could be
            # truncated mid-download, corrupted, or a non-standard
            # layout we don't understand.  Either way, falling back to
            # "any one shard counts" would re-introduce the false
            # positive codex flagged.  Treat as incomplete instead.
            return False
        weight_map = index.get("weight_map") or {}
        shard_names = set(weight_map.values())
        if shard_names:
            for shard in shard_names:
                target = snap / shard
                try:
                    target.resolve(strict=True)
                except (OSError, RuntimeError):
                    return False
            return True
        # Index parsed cleanly but has no weight_map — unusual but
        # legitimate (some templates ship empty maps).  Fall through to
        # the single-file glob check.

    # 2. Single-file or non-indexed model — at least one weight file resolves.
    for ext in ("*.safetensors", "*.npz", "*.gguf"):
        for entry in snap.glob(ext):
            try:
                entry.resolve(strict=True)
                return True
            except (OSError, RuntimeError):
                continue
    return False


def _check_alias(
    alias: str, repo_id: str, cache_roots: list[Path]
) -> ModelAvailability:
    """Look for the model in the candidate cache roots.

    Two cache layouts are checked at every root:
      - Hugging Face hub:  ``{root}/models--{org}--{repo}/snapshots/<sha>/``
      - LM Studio (and HF "raw" mode):  ``{root}/{org}/{repo}/``

    The second layout is what LM Studio uses for its model store at
    ``~/.lmstudio/models``; without it, models installed via LM Studio
    were silently invisible to discovery.

    A snapshot directory only counts when it has both config.json and
    at least one resolvable weight file (see _is_complete_snapshot).
    """
    hf_dirname = _repo_to_hf_dirname(repo_id)
    saw_partial = False
    for root in cache_roots:
        # 1. Hugging Face hub layout
        hf_candidate = root / hf_dirname
        if hf_candidate.exists():
            snapshots = hf_candidate / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    if _is_complete_snapshot(snap):
                        return ModelAvailability(
                            alias=alias,
                            repo_id=repo_id,
                            available=True,
                            path=snap,
                        )
                    elif (snap / "config.json").exists():
                        saw_partial = True

        # 2. LM Studio / HF raw layout — repo_id maps to nested dirs.
        raw_candidate = root.joinpath(*repo_id.split("/"))
        if _is_complete_snapshot(raw_candidate):
            return ModelAvailability(
                alias=alias,
                repo_id=repo_id,
                available=True,
                path=raw_candidate,
            )
        elif (raw_candidate / "config.json").exists():
            saw_partial = True

    reason = (
        "found config.json but no weight shards — partial download?"
        if saw_partial
        else "not found in HF_HUB_CACHE / ~/.cache/huggingface / ~/.lmstudio"
    )
    return ModelAvailability(
        alias=alias,
        repo_id=repo_id,
        available=False,
        reason=reason,
    )


def discover_local_models() -> list[ModelAvailability]:
    """Return availability info for every alias in aliases.json.

    Sorted: available first (alphabetical), then unavailable.
    """
    aliases = load_aliases()
    cache_roots = _hf_cache_roots()
    results = [_check_alias(a, r, cache_roots) for a, r in aliases.items()]
    results.sort(key=lambda m: (not m.available, m.alias))
    return results


def available_aliases() -> list[str]:
    """Convenience: just the aliases that are present locally."""
    return [m.alias for m in discover_local_models() if m.available]
