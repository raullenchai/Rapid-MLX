# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validation for a local MLX model's ``model_file`` config.

``mlx_lm`` can import a custom model implementation named by ``model_file`` in
``config.json``.  Rapid-MLX validates that field only when the caller supplied
an already-existing *local model directory*.  Repository ids and other remote
Hugging Face loading paths deliberately remain outside this guard: they resolve
their snapshot inside ``mlx_lm``/``huggingface_hub``, after Rapid-MLX no longer
has a caller-supplied local root to contain.
"""

from __future__ import annotations

import json
from pathlib import Path


def validate_local_model_file(model_name: str | Path) -> None:
    """Validate ``config.json::model_file`` before a local MLX model load.

    Missing ``model_file`` is the normal case and is accepted.  When it is
    present, the value must name an existing regular ``.py`` file relative to
    the resolved local model directory.  Resolving both paths catches
    ``..`` traversal and symlinks that leave the model root.

    This function intentionally does nothing for a non-directory model name;
    that is a remote/Hugging Face path whose download and custom-code policy is
    owned by the upstream loader rather than this local containment boundary.
    """
    supplied_root = Path(model_name)
    if not supplied_root.is_dir():
        return

    model_root = supplied_root.resolve(strict=True)
    config_path = model_root / "config.json"
    if not config_path.is_file():
        return

    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read local model config at {config_path}") from exc

    if not isinstance(config, dict) or config.get("model_file") is None:
        return

    model_file = config["model_file"]
    if not isinstance(model_file, str):
        raise ValueError("Local model config model_file must be a relative Python file")

    relative_path = Path(model_file)
    if relative_path.is_absolute() or relative_path.suffix != ".py":
        raise ValueError(
            "Local model config model_file must be a relative Python file inside "
            "the model directory"
        )

    try:
        custom_model_path = (model_root / relative_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            "Local model config model_file must name an existing Python file inside "
            "the model directory"
        ) from exc

    if (
        not custom_model_path.is_relative_to(model_root)
        or not custom_model_path.is_file()
    ):
        raise ValueError(
            "Local model config model_file must stay inside the model directory "
            "and name a regular Python file"
        )
