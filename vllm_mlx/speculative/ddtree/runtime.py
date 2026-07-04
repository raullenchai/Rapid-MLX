# SPDX-License-Identifier: Apache-2.0
"""DDTree runtime boundary.

The first DDTree integration deliberately treats ``dtree_mlx`` as an
optional external runtime. That keeps the rapid-mlx MVP small and lets us
validate correctness/performance before deciding whether to vendor or
reimplement the tree verifier.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eligibility import have_runtime

logger = logging.getLogger(__name__)


@dataclass
class DDTreeRuntime:
    generator: Any
    main_model_repo: str
    drafter_repo: str
    speculative_tokens: int
    tree_budget: int


def load_runtime(
    *,
    main_model_repo: str,
    drafter_repo: str,
    speculative_tokens: int,
    tree_budget: int,
) -> DDTreeRuntime:
    if not have_runtime():
        raise RuntimeError(
            "DDTree runtime not available — install the experimental runtime with: "
            "pip install 'dtree-mlx @ git+https://github.com/DrHB/dtree-mlx.git'"
        )

    from dtree_mlx.api import DFlashGenerator

    resolved_drafter_repo = _prepare_draft_model_for_dtree(drafter_repo)

    logger.info(
        "Loading DDTree runtime: target=%s drafter=%s spec=%d tree_budget=%d",
        main_model_repo,
        resolved_drafter_repo,
        speculative_tokens,
        tree_budget,
    )
    generator = DFlashGenerator(
        target_model=main_model_repo,
        draft_model=resolved_drafter_repo,
        draft_attention_mask="auto",
    )
    return DDTreeRuntime(
        generator=generator,
        main_model_repo=main_model_repo,
        drafter_repo=drafter_repo,
        speculative_tokens=speculative_tokens,
        tree_budget=tree_budget,
    )


def _prepare_draft_model_for_dtree(draft_model: str) -> str:
    """Return a dtree-mlx-compatible draft path.

    Public Qwen3.5 DFlash draft repos use the newer transformers 5
    ``rope_parameters.rope_theta`` config shape. Current ``dtree-mlx`` expects
    the older top-level ``rope_theta`` field, so materialize a small local
    mirror with a patched config and symlinked weights when needed.
    """

    path = _resolve_model_path(draft_model)
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        return draft_model
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return draft_model

    rope_parameters = cfg.get("rope_parameters")
    if cfg.get("rope_theta") is not None or not isinstance(rope_parameters, dict):
        return draft_model
    rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        return draft_model

    patched_cfg = dict(cfg)
    patched_cfg["rope_theta"] = rope_theta
    if "rope_scaling" not in patched_cfg and "rope_scaling" in rope_parameters:
        patched_cfg["rope_scaling"] = rope_parameters["rope_scaling"]

    patched = _patched_draft_dir(path)
    patched.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        dst = patched / child.name
        if child.name == "config.json":
            continue
        target = child.resolve()
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and Path(os.readlink(dst)) == target:
                continue
            if dst.is_dir() and not dst.is_symlink():
                continue
            dst.unlink()
        dst.symlink_to(target, target_is_directory=target.is_dir())
    (patched / "config.json").write_text(json.dumps(patched_cfg, indent=2) + "\n")
    logger.info(
        "DDTree: patched draft config for dtree-mlx compatibility: %s -> %s",
        draft_model,
        patched,
    )
    return str(patched)


def _resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo))


def _patched_draft_dir(source: Path) -> Path:
    import hashlib

    root = Path(
        os.environ.get(
            "RAPID_MLX_DDTREE_PATCH_CACHE", "~/.cache/rapid-mlx/ddtree-drafts"
        )
    ).expanduser()
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
    return root / f"{source.name}-{digest}"
