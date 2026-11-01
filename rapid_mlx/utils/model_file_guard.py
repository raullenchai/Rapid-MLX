# SPDX-License-Identifier: Apache-2.0
"""Explicit consent and path containment for checkpoint-provided Python.

Callers resolve remote repositories to a snapshot before checking config and
executing the loader on that same path. The local containment helper is separate
from consent: a Python file staying inside a model directory is still executable
untrusted code.
"""

from __future__ import annotations

import json
from os.path import relpath
from pathlib import Path
from typing import Any

from .._env import env_falsey, env_truthy

FORCED_TRUST_VLM_MODEL_TYPES = frozenset({"inkling"})


class RemoteCodeConsentRequired(ValueError):  # noqa: N818
    """Raised when a model needs repository code and the operator has not opted in."""


def validate_local_model_file(
    model_name: str | Path, *, model_config: dict | None = None
) -> None:
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
    if model_config is None and not config_path.is_file():
        return

    try:
        if model_config is not None:
            config = model_config
        else:
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
        # Preserve the lexical cache entry while resolving storage-tier links.
        # A configured cache repository may itself point to a cold-storage disk.
        # Only cache-owned entries gain this exception, never arbitrary folders
        # that merely imitate the Hub directory names.
        lexical_root = supplied_root.absolute()
        for candidate in (lexical_root, *lexical_root.parents):
            if candidate.parent.name != "snapshots":
                continue
            repository = candidate.parent.parent
            if (
                ".." not in relative_path.parts
                and repository.parent.resolve() == cache_root
                and repository.name.startswith("models--")
                and model_root.is_relative_to(candidate.resolve())
                and custom_model_path.parent == (repository / "blobs").resolve()
            ):
                contained = True
                break
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
        raise RemoteCodeConsentRequired(
            "Model declares model_file (executable Python). Loading is disabled "
            "by default. Review the repository code, then explicitly opt in with "
            "RAPID_MLX_TRUST_REMOTE_CODE=1 or trust_remote_code=True."
        )


def _validate_processor_directory(directory: Path, *, root: Path) -> dict:
    """Check one metadata directory and return its model config for path discovery."""
    model_config = {}
    location = (
        f"Model metadata in '{relpath(directory, root)}'"
        if directory != root
        else "Model processor"
    )
    for name in (
        "config.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        try:
            with (directory / name).open(encoding="utf-8") as handle:
                config = json.load(handle)
        except FileNotFoundError:
            continue
        if not isinstance(config, dict):
            raise ValueError(f"{name} must contain an object")
        if name == "config.json":
            model_config = config
        try:
            validate_model_code(config)
        except RemoteCodeConsentRequired as exc:
            if directory == root:
                raise
            raise RemoteCodeConsentRequired(f"{location}: {exc}") from exc
        if config.get("auto_map") and not remote_code_enabled():
            raise RemoteCodeConsentRequired(
                f"{location} declares auto_map (executable Python). "
                "Review the repository code, then explicitly opt in with "
                "RAPID_MLX_TRUST_REMOTE_CODE=1."
            )
    return model_config


def validate_processor_code(model_path: str | Path) -> None:
    """Inspect metadata before loaders that force repository-code trust.

    Check the root, up to 64 visible immediate subdirectories, and directory
    paths named in the root config (including one level inside dictionaries).
    Run once at load, never from generation or transcription.
    """
    root = Path(model_path)
    config = _validate_processor_directory(root, root=root)
    validate_local_model_file(root)
    if not root.is_dir():
        return

    seen = {root.resolve()}

    def check_directory(directory: Path) -> None:
        if not directory.is_dir():
            return
        resolved = directory.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        _validate_processor_directory(directory, root=root)

    count = 0
    for directory in root.iterdir():
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        check_directory(directory)
        count += 1
        if count == 64:
            break

    for value in config.values():
        values = value.values() if isinstance(value, dict) else (value,)
        for path in values:
            if isinstance(path, str):
                check_directory(root / path)


def checked_stt_loader(loader):
    """Bind consent to native dispatch after snapshot/archive resolution.

    Use an isolated function namespace, never process-wide monkeypatches. The
    native resolver retains its source hints, cache policy and transport; only
    the final local checkpoint boundary gains a metadata check.
    """
    from functools import update_wrapper
    from types import FunctionType

    if (
        not isinstance(loader, FunctionType)
        or "base_load_model" not in loader.__code__.co_names
        or not callable(loader.__globals__.get("base_load_model"))
    ):
        raise RuntimeError("Unsupported STT loader: missing local dispatch boundary")
    native = loader.__globals__["base_load_model"]

    def checked_base(*, model_path, **kwargs):
        if not isinstance(model_path, Path):
            raise ValueError("STT dispatch must use a resolved local checkpoint")
        validate_processor_code(model_path)
        return native(model_path=model_path, **kwargs)

    adapted = FunctionType(
        loader.__code__,
        {**loader.__globals__, "base_load_model": checked_base},
        loader.__name__,
        loader.__defaults__,
        loader.__closure__,
    )
    adapted.__kwdefaults__ = loader.__kwdefaults__
    return update_wrapper(adapted, loader)


def resolve_vlm_checkpoint(
    model_name: str | Path, *, revision: str | None = None
) -> Path:
    """Honor catalog identity before handing VLM loaders a concrete snapshot."""
    from mlx_vlm.utils import get_model_path

    from ..model_aliases import resolve_hub_model, resolve_revision, resolve_subfolder

    name = str(model_name)
    local = Path(name).expanduser()
    if local.is_dir():
        return local
    pin = resolve_revision(name)
    if pin is not None:
        if revision is not None and revision != pin:
            raise ValueError(
                f"{name} is pinned to revision {pin}; conflicting revision {revision}"
            )
        # Reuse the text loader's exact-cache/mirror resolution and completeness
        # check. A failed pin resolution must not retry against repository HEAD.
        from .tokenizer import _resolve_subfolder_checkpoint

        return Path(_resolve_subfolder_checkpoint(name))
    subfolder = resolve_subfolder(name)
    options: dict[str, Any] = {"revision": revision}
    if subfolder is not None:
        options["allow_patterns"] = [f"{subfolder}/*"]
    snapshot = Path(get_model_path(resolve_hub_model(name), **options))
    return snapshot / subfolder if subfolder is not None else snapshot


def load_vlm_with_policy(model_name: str | Path, *, loader=None, **kwargs):
    """Load VLM weights and processors from one inspected snapshot with consent."""
    from mlx_vlm import load

    trust = remote_code_enabled(kwargs.pop("trust_remote_code", None))
    revision = kwargs.pop("revision", None)
    model_path = resolve_vlm_checkpoint(model_name, revision=revision)
    with (model_path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Model config must contain an object")
    config = {**config, **(kwargs.get("model_config") or {})}
    validate_model_code(config, trust_remote_code=trust)
    if not trust and config.get("model_type") in FORCED_TRUST_VLM_MODEL_TYPES:
        validate_processor_code(model_path)
    if config.get("model_file"):
        validate_local_model_file(model_path, model_config=config)
    if config.get("model_type") == "kimi_k25" and not config.get("model_file"):
        from ..models.kimi_vlm import load_kimi_vlm

        return load_kimi_vlm(model_path, **kwargs)
    return (loader or load)(str(model_path), trust_remote_code=trust, **kwargs)
