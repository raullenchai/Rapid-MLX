# SPDX-License-Identifier: Apache-2.0
"""Explicit consent and path containment for checkpoint-provided Python.

Callers resolve remote repositories to a snapshot before checking config and
executing the loader on that same path. The local containment helper is separate
from consent: a Python file staying inside a model directory is still executable
untrusted code.
"""

from __future__ import annotations

import json
from pathlib import Path

from .._env import env_falsey, env_truthy


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

    contained = custom_model_path.is_relative_to(model_root)
    if not contained:
        # Hub snapshots intentionally symlink files into the same repository's
        # blobs directory. Only recognize that layout inside the configured Hub
        # cache; an arbitrary local directory named snapshots gains no exception.
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_root = Path(HF_HUB_CACHE).resolve()
        repository = model_root.parent.parent
        contained = (
            ".." not in relative_path.parts
            and model_root.parent.name == "snapshots"
            and repository.parent == cache_root
            and repository.name.startswith("models--")
            and custom_model_path.parent == (repository / "blobs").resolve()
        )
    if not contained or not custom_model_path.is_file():
        raise ValueError(
            "Local model config model_file must stay inside the model directory "
            "and name a regular Python file"
        )


def remote_code_enabled(requested: bool | None = None) -> bool:
    """Explicit API consent or process opt-in; process opt-out always wins."""
    if env_falsey("RAPID_MLX_TRUST_REMOTE_CODE"):
        return False
    return requested is True or (
        requested is None and env_truthy("RAPID_MLX_TRUST_REMOTE_CODE")
    )


def validate_model_code(config: dict, *, trust_remote_code: bool | None = None) -> None:
    """Guard upstream model_file imports, which ignore tokenizer trust flags."""
    if config.get("model_file") and not remote_code_enabled(trust_remote_code):
        raise ValueError(
            "Model declares model_file (executable Python). Loading is disabled "
            "by default. Review the repository code, then explicitly opt in with "
            "RAPID_MLX_TRUST_REMOTE_CODE=1 or trust_remote_code=True."
        )


def load_vlm_with_policy(model_name: str | Path, *, loader=None, **kwargs):
    """Load VLM weights and processors from one inspected snapshot with consent."""
    from mlx_vlm import load
    from mlx_vlm.utils import get_model_path

    trust = remote_code_enabled(kwargs.pop("trust_remote_code", None))
    revision = kwargs.pop("revision", None)
    model_path = get_model_path(str(model_name), revision=revision)
    with (model_path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Model config must contain an object")
    config = {**config, **(kwargs.get("model_config") or {})}
    validate_model_code(config, trust_remote_code=trust)
    if config.get("model_file"):
        validate_local_model_file(model_path)
    return (loader or load)(str(model_path), trust_remote_code=trust, **kwargs)
