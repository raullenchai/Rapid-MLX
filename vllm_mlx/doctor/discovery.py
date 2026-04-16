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

from .runner import REPO_ROOT


@dataclass
class ModelAvailability:
    alias: str
    repo_id: str
    available: bool
    path: Path | None = None
    reason: str = ""  # only set when available=False


def load_aliases() -> dict[str, str]:
    """Return the alias → repo-id mapping shipped with the package."""
    aliases_path = REPO_ROOT / "vllm_mlx" / "aliases.json"
    with open(aliases_path) as f:
        return json.load(f)


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
    """A snapshot is "complete" if it has config.json AND actual model weights.

    Just probing config.json is not enough — partially-downloaded HF
    snapshots routinely have config.json + model.safetensors.index.json
    but no *.safetensors shards (the shards are symlinks into ../blobs/
    that point at non-existent files when the download was aborted).
    Loading such a snapshot crashes the server with "No safetensors
    found", which then shows up in the scorecard as a misleading FAIL
    even though the right diagnosis is "model never finished
    downloading".

    A weight file counts whether it's a single ``model.safetensors``,
    sharded ``model-NNNN-of-NNNN.safetensors``, or any other
    ``.safetensors`` file (some MLX exports use bespoke names).  We
    additionally resolve symlinks so a dangling link does not satisfy
    the check.
    """
    if not (snap / "config.json").exists():
        return False
    for entry in snap.glob("*.safetensors"):
        try:
            # resolve(strict=True) raises if the target is missing,
            # so we can rely on it to detect dangling symlinks too.
            entry.resolve(strict=True)
            return True
        except (OSError, RuntimeError):
            continue
    # Some loaders accept .npz or .gguf — be conservative and accept
    # them too rather than over-report missing weights.
    for ext in ("*.npz", "*.gguf"):
        for entry in snap.glob(ext):
            try:
                entry.resolve(strict=True)
                return True
            except (OSError, RuntimeError):
                continue
    return False


def _check_alias(alias: str, repo_id: str, cache_roots: list[Path]) -> ModelAvailability:
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
                            alias=alias, repo_id=repo_id,
                            available=True, path=snap,
                        )
                    elif (snap / "config.json").exists():
                        saw_partial = True

        # 2. LM Studio / HF raw layout — repo_id maps to nested dirs.
        raw_candidate = root.joinpath(*repo_id.split("/"))
        if _is_complete_snapshot(raw_candidate):
            return ModelAvailability(
                alias=alias, repo_id=repo_id,
                available=True, path=raw_candidate,
            )
        elif (raw_candidate / "config.json").exists():
            saw_partial = True

    reason = (
        "found config.json but no weight shards — partial download?"
        if saw_partial
        else "not found in HF_HUB_CACHE / ~/.cache/huggingface / ~/.lmstudio"
    )
    return ModelAvailability(
        alias=alias, repo_id=repo_id, available=False, reason=reason,
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
