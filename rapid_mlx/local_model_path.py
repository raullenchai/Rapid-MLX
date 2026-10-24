# SPDX-License-Identifier: Apache-2.0
"""Local model-path detection and privacy-safe missing-file diagnostics."""

from __future__ import annotations

import difflib
import errno
import json
import os
import re
import shlex
from pathlib import Path, PurePosixPath

MAX_MISSING_LOCAL_MODEL_FILES = 5
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _typed_ref(model_ref: str) -> str:
    """Remove shell quotes accidentally passed literally and surrounding space."""
    if os.path.exists(os.path.expanduser(model_ref)):
        return model_ref
    value = model_ref.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inside = value[1:-1]
        value = inside if os.path.exists(os.path.expanduser(inside)) else inside.strip()
    return value


def resolved_model_ref(model_ref: str) -> str:
    """Use an absolute path for local input, preserving Hub repo ids."""
    # A literal existing directory can legitimately end in a space. Only
    # repair whitespace/quotes when that exact spelling does not resolve.
    value = _typed_ref(model_ref)
    # An unprefixed org/name is a Hub id when the whole path is missing,
    # regardless of whether an unrelated local org directory exists.
    if _HF_REPO_ID.fullmatch(value) and not os.path.exists(value):
        return value
    if is_local_model_ref(value) or ("/" in value and Path(value).parent.is_dir()):
        if "/" not in value and not value.startswith(("~", ".")):
            from .model_aliases import local_dir_shadows_alias

            if local_dir_shadows_alias(value) or os.path.exists(value):
                return value
        return str(Path(value).expanduser().absolute())
    return value


def _exception_chain(exc: BaseException | None) -> tuple[BaseException, ...]:
    """Return the bounded explicit cause chain used for load diagnostics."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 16:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
    return tuple(chain)


def _has_authoritative_load_failure(exc: BaseException | None) -> bool:
    """Whether a typed failure must retain its own classification/message."""
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    from .model_load_errors import (
        IncompatibleWeights,
        InvalidModelConfig,
        QuantizationMismatch,
        TokenizerLoadFailed,
    )
    from .runtime.optional_runtime import OptionalRuntimeMissing

    authoritative = (
        OptionalRuntimeMissing,
        InvalidModelConfig,
        TokenizerLoadFailed,
        IncompatibleWeights,
        QuantizationMismatch,
        HfHubHTTPError,
        RepositoryNotFoundError,
        ModuleNotFoundError,
    )
    return any(isinstance(current, authoritative) for current in _exception_chain(exc))


def is_local_model_ref(model_ref: object) -> bool:
    """Whether ``model_ref`` denotes a local path, including a missing one."""
    if not isinstance(model_ref, str) or not model_ref:
        return False
    from .telemetry.redact import normalize_model_path

    if normalize_model_path(model_ref) == "<local>":
        return True
    try:
        return os.path.exists(os.path.expanduser(model_ref))
    except (OSError, ValueError):
        return True


def _safe_relative_name(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        return None
    return path.as_posix()


def _missing_index_shards(
    root: Path, *, recursive: bool = True, primary_only: bool = False
) -> list[str]:
    missing: set[str] = set()
    try:
        indexes = sorted(
            root.rglob("*.safetensors.index.json")
            if recursive
            else root.glob("*.safetensors.index.json")
        )
    except OSError:
        return []
    for index_path in indexes:
        if primary_only and index_path.name != "model.safetensors.index.json":
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict):
            continue
        for raw_name in weight_map.values():
            name = _safe_relative_name(raw_name)
            if name is None:
                continue
            candidate = index_path.parent.joinpath(*PurePosixPath(name).parts)
            try:
                present = candidate.is_file() and candidate.stat().st_size > 0
            except OSError:
                present = False
            if not present:
                missing.add(candidate.relative_to(root).as_posix())
    return sorted(missing)


def _exception_filename_path(exc: BaseException) -> Path | None:
    """Return a text filename as a path without accepting bytes paths."""
    if not isinstance(exc, FileNotFoundError) or exc.filename is None:
        return None
    try:
        filename = os.fspath(exc.filename)
    except TypeError:
        return None
    if not isinstance(filename, str):
        return None
    return Path(filename)


def missing_local_model_files(
    model_ref: object,
    exc: BaseException | None = None,
    *,
    limit: int = MAX_MISSING_LOCAL_MODEL_FILES,
    scan_indexes: bool = True,
) -> tuple[str, ...]:
    """Return model-root-relative missing names, capped by ``limit``.

    HTTP callers disable ``scan_indexes`` to inspect only the exception chain
    rather than enumerate an untrusted directory tree.
    """
    if not is_local_model_ref(model_ref) or not isinstance(model_ref, str):
        return ()
    root = Path(model_ref).expanduser().absolute()
    if not root.is_dir():
        return ()

    missing = set(_missing_index_shards(root)) if scan_indexes else set()
    for current in _exception_chain(exc):
        candidate = _exception_filename_path(current)
        if candidate is None:
            continue
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            relative = None
        if relative and relative != ".":
            missing.add(relative)
    return tuple(sorted(missing)[: max(0, limit)])


def local_model_failure_message(
    model_ref: object,
    exc: BaseException | None = None,
    *,
    include_supplied_path: bool = False,
    scan_indexes: bool = True,
) -> str | None:
    """Render a local-model failure without exposing paths to HTTP clients."""
    if _has_authoritative_load_failure(exc):
        return None
    if not isinstance(model_ref, str):
        return None
    typed = _typed_ref(model_ref) if include_supplied_path else model_ref
    if not is_local_model_ref(typed) and not (
        include_supplied_path and "/" in typed and Path(typed).parent.is_dir()
    ):
        return None
    expanded = Path(typed).expanduser().absolute()
    try:
        exists = expanded.exists()
    except OSError:
        return None
    shown = f" {model_ref!r}" if include_supplied_path else ""
    if not exists:
        missing_path = False
        for current in _exception_chain(exc):
            candidate = _exception_filename_path(current)
            if candidate is not None and candidate.expanduser().absolute() == expanded:
                missing_path = True
                break
        if missing_path:
            message = f"The local model path{shown} does not exist."
            if not include_supplied_path:
                return message
            message += f" Resolved path: {expanded}."
            try:
                siblings = [p.name for p in expanded.parent.iterdir() if p.is_dir()]
            except OSError:
                siblings = []
            matches = difflib.get_close_matches(
                expanded.name, siblings, n=3, cutoff=0.0
            )
            if matches:
                message += " Nearby model folders: " + ", ".join(matches) + "."
            repo_id = typed.removeprefix("./")
            if typed.startswith("./") and _HF_REPO_ID.fullmatch(repo_id):
                message += f" If you meant a Hugging Face repo, use {repo_id}."
            elif typed.startswith("~/") or typed.startswith(("./", "../", "/")):
                message += " If you meant a Hugging Face repo, use org/model without a path prefix."
            return message
        return None
    missing = missing_local_model_files(model_ref, exc, scan_indexes=scan_indexes)
    if missing:
        return (
            f"The local model directory{shown} is missing required files: "
            f"{', '.join(missing)}."
        )
    return None


def raise_if_missing_local_model(model_ref: object) -> None:
    """Raise before a missing explicit local path can be treated as a Hub id."""
    if not is_local_model_ref(model_ref) or not isinstance(model_ref, str):
        return
    expanded = Path(model_ref).expanduser().absolute()
    exists = expanded.exists()
    if not exists:
        raise FileNotFoundError(
            errno.ENOENT,
            "Local model path does not exist",
            model_ref,
        )


def validate_local_model_for_serve(model_ref: object) -> None:
    """Reject an incomplete explicit model directory before engine startup."""
    raise_if_missing_local_model(model_ref)
    if not is_local_model_ref(model_ref) or not isinstance(model_ref, str):
        return
    root = Path(model_ref).expanduser().absolute()
    if not root.is_dir():
        raise NotADirectoryError(errno.ENOTDIR, "Expected a model directory", str(root))
    if not os.access(root, os.R_OK | os.X_OK):
        raise PermissionError(errno.EACCES, "Cannot read model directory", str(root))
    # Non-text lanes use manifests instead of a universal root config.json.
    # Check the manifest and at least one component before their loaders run.
    for name in ("split_model.json", "model_index.json", "params.json"):
        manifest_path = root / name
        if not manifest_path.is_file():
            continue
        # params.json is also used as auxiliary metadata by some text models.
        if name == "params.json" and (root / "config.json").is_file():
            continue
        if name == "model_index.json" and (root / "config.json").is_file():
            from .model_metadata import config_indicates_multimodal

            try:
                config_data = json.loads(
                    (root / "config.json").read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                config_data = {}
            if not isinstance(config_data, dict) or not config_indicates_multimodal(
                config_data
            ):
                continue
        if not os.access(manifest_path, os.R_OK):
            raise PermissionError(
                errno.EACCES, "Cannot read model manifest", str(manifest_path)
            )
        from .model_load_errors import InvalidModelConfig

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidModelConfig(f"Invalid model manifest: {name}") from exc
        if not isinstance(manifest, dict) or not manifest:
            raise InvalidModelConfig(f"Invalid model manifest: {name}")
        if name == "model_index.json" and "_class_name" not in manifest:
            raise InvalidModelConfig(f"Invalid model manifest: {name}")
        if name == "split_model.json":
            components = manifest.get("components")
            if not isinstance(components, list) or not components:
                raise FileNotFoundError(
                    errno.ENOENT, "Missing model components", str(manifest_path)
                )
            for component in components:
                if not isinstance(component, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]+", component
                ):
                    raise FileNotFoundError(
                        errno.ENOENT, "Invalid model component", str(manifest_path)
                    )
                weight = root / f"{component}.safetensors"
                if not weight.is_file() or weight.stat().st_size == 0:
                    raise FileNotFoundError(
                        errno.ENOENT, "Missing model component", str(weight)
                    )
            return
        if name == "model_index.json":
            metadata_components = {
                "scheduler",
                "processor",
                "feature_extractor",
                "image_processor",
            }
            required = [
                component
                for component, declaration in manifest.items()
                if not component.startswith("_")
                and component not in metadata_components
                and not component.startswith("tokenizer")
                and isinstance(declaration, list)
                and declaration != [None, None]
            ]
            if not required:
                raise FileNotFoundError(
                    errno.ENOENT, "Missing model components", str(manifest_path)
                )
            for component in required:
                if not re.fullmatch(r"[A-Za-z0-9_-]+", component):
                    raise FileNotFoundError(
                        errno.ENOENT, "Invalid model component", str(manifest_path)
                    )
                component_dir = root / component
                missing_shards = _missing_index_shards(component_dir, recursive=False)
                if missing_shards:
                    raise FileNotFoundError(
                        errno.ENOENT,
                        "Missing model component",
                        str(component_dir / missing_shards[0]),
                    )
                component_files = (
                    *component_dir.glob("*.safetensors"),
                    *component_dir.glob("*.npz"),
                    root / f"{component}.safetensors",
                )
                if not any(
                    path.is_file() and path.stat().st_size > 0
                    for path in component_files
                ):
                    raise FileNotFoundError(
                        errno.ENOENT, "Missing model component", str(root / component)
                    )
            return
        if not any(
            path.is_file() and path.stat().st_size > 0
            for path in root.glob("*.safetensors")
        ):
            raise FileNotFoundError(
                errno.ENOENT, "Missing model weights", str(root / "model.safetensors")
            )
        return
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(errno.ENOENT, "Missing model config", str(config))
    if not os.access(config, os.R_OK):
        raise PermissionError(errno.EACCES, "Cannot read model config", str(config))
    from .model_load_errors import validate_model_config_file

    validate_model_config_file(root)
    # mlx-lm loads model*.safetensors at the root even with an index. Native Gemma 4 and
    # EmbeddingGemma 2 loaders accept any root safetensors filename.
    # PyTorch .bin files do not make any of these checkpoints serveable.
    try:
        config_data = json.loads(config.read_text(encoding="utf-8"))
        model_type = config_data.get("model_type")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        config_data = {}
        model_type = None  # Leave malformed configs to the loader's typed error.
    if not isinstance(model_type, str):
        model_type = None
    # The normal text path checks its primary index in tokenizer.py before
    # loading. Native Gemma 4 reads root safetensors directly and ignores it.
    if model_type not in {"gemma4", "gemma4_assistant", "gemma4_unified"}:
        missing_shards = _missing_index_shards(root, recursive=False, primary_only=True)
        if missing_shards:
            raise FileNotFoundError(
                errno.ENOENT,
                "Missing indexed model shard",
                str(root / missing_shards[0]),
            )
    from .model_metadata import config_indicates_multimodal

    native_any_safetensors = config_indicates_multimodal(config_data) or model_type in {
        "gemma4",
        "gemma4_assistant",
        "gemma4_unified",
        "embedding_gemma2",
    }
    weights = list(
        root.glob("*.safetensors" if native_any_safetensors else "model*.safetensors")
    )
    if model_type == "embedding_gemma2":
        index = root / "model.safetensors.index.json"
        if index.is_file():
            try:
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                weights = [
                    root.joinpath(*PurePosixPath(relative_name).parts)
                    for raw in weight_map.values()
                    if (relative_name := _safe_relative_name(raw)) is not None
                ]
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                AttributeError,
            ):
                weights = []
    if model_type == "whisper":
        weights.extend((root / "weights.npz", root / "weights.safetensors"))
    elif model_type == "kokoro":
        weights.append(root / "kokoro-v1_0.safetensors")
    for path in weights:
        if path.is_file() and not os.access(path, os.R_OK):
            raise PermissionError(errno.EACCES, "Cannot read model weights", str(path))
    if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
        raise FileNotFoundError(
            errno.ENOENT, "Missing model weights", str(root / "model.safetensors")
        )


def local_serve_preflight_lines(model_ref: str, exc: OSError) -> tuple[str, str]:
    """Give a local serve error a concrete repair command."""
    root = Path(model_ref).expanduser().absolute()
    if not isinstance(exc, (FileNotFoundError, NotADirectoryError, PermissionError)):
        return (
            f"The filesystem could not read the local checkpoint at {root}: {exc.strerror or type(exc).__name__}.",
            f"Check the disk and file access, then retry `rapid-mlx serve {shlex.quote(str(root))}`.",
        )
    filename = Path(exc.filename) if isinstance(exc.filename, str) else None
    if isinstance(exc, PermissionError):
        return (
            f"A file in {root} cannot be read by this user.",
            f"Grant read access with `chmod -R u+rX {shlex.quote(str(root))}`, then retry.",
        )
    if not root.exists():
        cause = f"Local model path {root} does not exist."
    elif not root.is_dir():
        cause = f"{root} is a file, not a model directory."
    elif filename == root / "config.json":
        cause = f"{root} is missing config.json."
    elif filename is not None and filename.name in (
        "split_model.json",
        "model_index.json",
        "params.json",
    ):
        cause = f"{filename.name} in {root} is invalid or missing model components."
    elif exc.strerror == "Missing model component" and filename is not None:
        cause = f"{root} is missing required component {filename.name}."
    else:
        cause = f"{root} is missing model weights."
    for part in root.parts:
        if not part.startswith("models--"):
            continue
        repo = part.removeprefix("models--").replace("--", "/", 1)
        from .model_aliases import catalog_alias_for

        alias = catalog_alias_for(repo)
        if alias:
            return cause, f"Try the catalog model: `rapid-mlx serve {alias}`."
    if not root.exists():
        return cause, "Check the path, then run `rapid-mlx serve /path/to/model`."
    if not root.is_dir():
        return (
            cause,
            "Serve its containing model directory with `rapid-mlx serve /path/to/model`.",
        )
    if filename == root / "config.json":
        return (
            cause,
            f"Restore config.json from the matching checkpoint, then run `rapid-mlx serve {shlex.quote(str(root))}`.",
        )
    if filename is not None and filename.name in (
        "split_model.json",
        "model_index.json",
        "params.json",
    ):
        return (
            cause,
            f"Restore {filename.name} and its components from the matching checkpoint, then run `rapid-mlx serve {shlex.quote(str(root))}`.",
        )
    return (
        cause,
        f"Restore the checkpoint weights, then run `rapid-mlx serve {shlex.quote(str(root))}`.",
    )
