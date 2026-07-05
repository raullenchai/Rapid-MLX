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
import shutil
import tempfile
import uuid
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
    _install_qwen35_split_prefill_patch(generator)
    return DDTreeRuntime(
        generator=generator,
        main_model_repo=main_model_repo,
        drafter_repo=drafter_repo,
        speculative_tokens=speculative_tokens,
        tree_budget=tree_budget,
    )


def _install_qwen35_split_prefill_patch(generator: Any) -> None:
    """Match MLX-LM prompt prefill semantics for dtree-mlx Qwen3.5 targets.

    ``mlx_lm.generate_step`` processes all prompt tokens except the final one
    into cache first, then samples the first generated token from a separate
    one-token call. Current ``dtree-mlx`` Qwen3.5 DFlash/DDTree paths prefill
    the whole prompt in one call, which is not numerically equivalent for the
    hybrid GatedDeltaNet target and can change greedy outputs. Keep the patch
    local to the optional runtime boundary until upstream exposes the behavior.
    """

    target = getattr(generator, "target", None)
    adapter = getattr(target, "adapter", None)
    if getattr(adapter, "family", None) != "qwen3_5":
        return
    if getattr(target, "_rapid_mlx_split_prefill_patch", False):
        return
    original = getattr(target, "forward_with_hidden_states", None)
    if original is None:
        return

    import mlx.core as mx

    def cache_is_empty(cache: Any) -> bool:
        if not isinstance(cache, (list, tuple)):
            return False
        for layer_cache in cache:
            if int(getattr(layer_cache, "offset", 0) or 0) > 0:
                return False
            if getattr(layer_cache, "keys", None) is not None:
                return False
            if getattr(layer_cache, "values", None) is not None:
                return False
            try:
                if layer_cache[0] is not None or layer_cache[1] is not None:
                    return False
            except (TypeError, IndexError, KeyError, AttributeError):
                pass
        return True

    def forward_with_split_prefill(
        inputs,
        cache,
        layer_ids,
        return_rollback_records: bool = False,
    ):
        if (
            not return_rollback_records
            and getattr(inputs, "ndim", None) == 2
            and int(inputs.shape[0]) == 1
            and int(inputs.shape[1]) > 1
            and cache_is_empty(cache)
        ):
            prefix_inputs = inputs[:, :-1]
            last_input = inputs[:, -1:]
            prefix_logits, prefix_hidden = original(
                prefix_inputs,
                cache,
                layer_ids,
                False,
            )
            mx.eval(prefix_logits, prefix_hidden)
            last_logits, last_hidden = original(last_input, cache, layer_ids, False)
            mx.eval(last_logits, last_hidden)
            return last_logits, mx.concatenate([prefix_hidden, last_hidden], axis=1)

        return original(inputs, cache, layer_ids, return_rollback_records)

    target.forward_with_hidden_states = forward_with_split_prefill
    target._rapid_mlx_split_prefill_patch = True


def _prepare_draft_model_for_dtree(draft_model: str) -> str:
    """Return a dtree-mlx-compatible draft path.

    The public Qwen3.5 DFlash draft repos use the newer transformers 5
    ``rope_parameters.rope_theta`` config shape. Current ``dtree-mlx``
    expects the older top-level ``rope_theta`` field, so the raw HF repo
    fails at load time before generation starts. Materialize a small local
    mirror with a patched config and symlinked weights, then pass that path
    to ``dtree-mlx``.
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
    if _patched_dir_is_usable(patched, path, patched_cfg):
        return str(patched)
    patched.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{patched.name}.tmp-", dir=patched.parent))
    completed = False
    try:
        for child in path.iterdir():
            dst = tmp / child.name
            if child.name == "config.json":
                continue
            target = child.resolve()
            dst.symlink_to(target, target_is_directory=target.is_dir())
        (tmp / "config.json").write_text(json.dumps(patched_cfg, indent=2) + "\n")
        try:
            tmp.replace(patched)
        except OSError:
            if _patched_dir_is_usable(patched, path, patched_cfg):
                completed = True
                _remove_path(tmp)
                return str(patched)
            patched = patched.with_name(f"{patched.name}-{uuid.uuid4().hex[:8]}")
            tmp.replace(patched)
        completed = True
    finally:
        if not completed:
            _remove_path(tmp)
    logger.info(
        "DDTree: patched draft config for dtree-mlx compatibility: %s -> %s",
        draft_model,
        patched,
    )
    return str(patched)


def _patched_dir_is_usable(patched: Path, source: Path, patched_cfg: dict) -> bool:
    cfg_path = patched / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        if json.loads(cfg_path.read_text()) != patched_cfg:
            return False
    except (OSError, json.JSONDecodeError):
        return False
    for child in source.iterdir():
        if child.name == "config.json":
            continue
        dst = patched / child.name
        if not dst.is_symlink() or dst.resolve() != child.resolve():
            return False
    return True


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


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
