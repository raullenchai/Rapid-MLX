# SPDX-License-Identifier: Apache-2.0
"""Local-only startup diagnostics; no checkpoint details enter telemetry."""

from __future__ import annotations

import errno
import json
import os
import re
from pathlib import Path


def _local_config(model_ref: object) -> tuple[Path, dict] | None:
    if not isinstance(model_ref, str):
        return None
    root = Path(model_ref).expanduser()
    try:
        path = root / "config.json"
        if not root.is_dir() or path.stat().st_size > 1024 * 1024:
            return None
        config = json.loads(path.read_text(encoding="utf-8"))
        return (root, config) if isinstance(config, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def _chain(exc: BaseException):
    seen: set[int] = set()
    for _ in range(32):
        if id(exc) in seen:
            break
        seen.add(id(exc))
        yield exc
        if exc.__cause__ is None:
            break
        exc = exc.__cause__


def _same_local_path(value: object, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).expanduser().resolve(
        strict=False
    ) == expected.resolve(strict=False)


def truncated_local_weights(model_ref: object) -> bool:
    """A read failure belongs to a checkpoint only with local weight evidence."""
    if not isinstance(model_ref, str):
        return False
    root = Path(model_ref).expanduser()
    if not root.is_dir():
        return False
    from .local_model_path import validate_local_model_for_serve
    from .model_load_errors import InvalidModelConfig

    try:
        validate_local_model_for_serve(str(root))
    except (OSError, InvalidModelConfig):
        return False
    try:
        return any(
            path.is_file() and os.access(path, os.R_OK)
            for path in root.rglob("*.safetensors")
        )
    except OSError:
        return False


def malformed_numeric_config(exc: BaseException, model_ref: object) -> str | None:
    """Find the numeric field named by our independent config validator."""
    from .model_load_errors import InvalidModelConfig

    local = _local_config(model_ref)
    if local is None:
        return None
    messages = [
        str(error) for error in _chain(exc) if isinstance(error, InvalidModelConfig)
    ]
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    ):
        value = local[1].get(key)
        if (
            any(f"{key} must be an integer" in message for message in messages)
            and value is not None
            and (isinstance(value, bool) or not isinstance(value, int))
        ):
            return key
    return None


def local_serve_diagnostic(
    exc: BaseException, model_ref: object
) -> tuple[str, str, str] | None:
    """Return an existing serve class and two safe CLI lines when attributable.

    Only an existing local directory is inspected. Exact loader shapes and
    files constrain the diagnosis so unrelated programming errors stay other.
    """
    local = _local_config(model_ref)
    if local is None:
        return None
    root, config = local
    try:
        errors = list(_chain(exc))
        texts = [str(error) for error in errors]
    except BaseException:
        return None

    model_type = config.get("model_type")
    if field := malformed_numeric_config(exc, model_ref):
        return (
            "invalid_config",
            f"config.json has a nonnumeric {field} value.",
            f"Correct {field} in config.json, then retry `rapid-mlx serve /path/to/model`.",
        )
    if (
        isinstance(model_type, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", model_type)
        and any(
            isinstance(error, (ValueError, ModuleNotFoundError))
            and re.fullmatch(r"Model type .+ not supported\.?", message)
            for error, message in zip(errors, texts)
        )
    ):
        return (
            "unsupported_architecture",
            f"config.json requests unsupported model_type '{model_type}'.",
            "Use a supported model such as `rapid-mlx serve qwen3-0.6b-4bit`.",
        )

    if (
        any(
            isinstance(error, TypeError) and "TextConfig.__init__() missing" in message
            for error, message in zip(errors, texts)
        )
        and isinstance(model_type, str)
        and model_type.endswith("_vl")
    ):
        return (
            "invalid_config",
            "config.json is missing required VLM text_config fields.",
            "Copy a complete config.json from the matching VLM checkpoint.",
        )

    from rapid_mlx.model_load_errors import IncompatibleWeights

    quant = config.get("quantization")
    if (
        isinstance(quant, dict)
        and quant.get("mode") == "mxfp4"
        and any(isinstance(error, IncompatibleWeights) for error in errors)
        and any(
            re.search(
                r"Expected shape .+ received shape .+ for parameter "
                r".+\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight",
                message,
            )
            for message in texts
        )
    ):
        return (
            "quantization_mismatch",
            "config.json declares MXFP4, but the checkpoint weight layout differs.",
            "Use matching MLX MXFP4 weights or correct the quantization metadata.",
        )

    if any(
        isinstance(error, ValueError)
        and message.startswith("Cannot use apply_chat_template because")
        for error, message in zip(errors, texts)
    ) or any(type(error).__name__ == "TemplateSyntaxError" for error in errors):
        return (
            "invalid_config",
            "The model's chat template is missing or invalid.",
            "Fix tokenizer_config.json or chat_template.jinja for this checkpoint.",
        )

    from rapid_mlx.model_load_errors import TokenizerLoadFailed

    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "spiece.model",
        "tiktoken.model",
    )
    if (
        any(isinstance(error, TokenizerLoadFailed) for error in errors)
        and not any((root / name).is_file() for name in tokenizer_files)
        and not any(root.glob("*.tiktoken"))
    ):
        return (
            "tokenizer_load_failed",
            "The local model directory has no tokenizer files.",
            "Copy tokenizer.json or tokenizer.model and tokenizer_config.json from the matching checkpoint.",
        )

    if (
        isinstance(model_type, str)
        and model_type.endswith("_vl")
        and not (root / "preprocessor_config.json").is_file()
        and any(
            isinstance(error, OSError)
            and message.startswith("Can't load image processor for")
            and "preprocessor_config.json" in message
            for error, message in zip(errors, texts)
        )
    ):
        return (
            "local_path_missing",
            "The VLM directory is missing preprocessor_config.json.",
            "Copy preprocessor_config.json from the matching VLM checkpoint.",
        )

    return None


def startup_error_lines(
    exc: BaseException, model_ref: object
) -> tuple[str, str] | None:
    """Present specific local causes plus the existing memory remedy."""
    from rapid_mlx.telemetry.model_events import serve_error_class

    error_class = serve_error_class(exc, model_ref=model_ref)
    detail = _startup_only_diagnostic(exc, model_ref, error_class)
    if detail is not None:
        return detail[1:]
    detail = local_serve_diagnostic(exc, model_ref)
    if detail is not None and detail[0] == error_class:
        return detail[1:]
    if error_class == "insufficient_memory":
        return (
            "The model ran out of memory while loading weights.",
            "Free memory, then run `rapid-mlx serve qwen3-0.6b-4bit` for a smaller model.",
        )
    if error_class == "backend_load_failed":
        return (
            "The MLX or Metal backend could not start.",
            "Run `rapid-mlx doctor`, then retry the model after resolving the reported runtime issue.",
        )
    return None


def _startup_only_diagnostic(
    exc: BaseException, model_ref: object, error_class: str
) -> tuple[str, str, str] | None:
    """Diagnose observed loader shapes for display without changing event classes."""
    try:
        errors = list(_chain(exc))
        if (
            error_class == "corrupt_weights"
            and truncated_local_weights(model_ref)
            and any(
                isinstance(error, RuntimeError)
                and re.fullmatch(
                    r"\[read\] Unable to read \d+ bytes from file\.", str(error)
                )
                for error in errors
            )
        ):
            return (
                "incomplete_weights",
                "A model safetensors file is incomplete or unreadable.",
                "Re-download or re-export the matching model weights, then retry.",
            )
        incomplete = _incomplete_checkpoint_diagnostic(errors, model_ref, error_class)
        if incomplete is not None:
            return incomplete
        if error_class == "other" and any(
            isinstance(error, OSError) and error.errno == errno.ENOSPC
            for error in errors
        ):
            return (
                "disk_full",
                "The model could not be prepared because the disk is full.",
                "Free disk space and retry the model load.",
            )
        if not isinstance(model_ref, str):
            return None
        root = Path(model_ref).expanduser().absolute()
        if (
            error_class == "corrupt_weights"
            and root.is_file()
            and root.suffix == ".safetensors"
            and any(
                isinstance(error, NotADirectoryError)
                and _same_local_path(error.filename, root / "config.json")
                for error in errors
            )
        ):
            return (
                "single_weight_file",
                "A safetensors file was supplied where a model directory is required.",
                "Serve its parent model directory containing config.json and the matching weights.",
            )
        if not root.is_dir():
            return None
        config_path = root / "config.json"
        if any(
            isinstance(error, PermissionError)
            and error.filename is not None
            and Path(error.filename)
            .expanduser()
            .resolve(strict=False)
            .is_relative_to(root.resolve(strict=False))
            for error in errors
        ) or any(
            isinstance(error, RuntimeError)
            and str(error).startswith("[load_safetensors] Failed to open file ")
            and _same_local_path(
                str(error).removeprefix("[load_safetensors] Failed to open file "),
                weight,
            )
            and weight.is_file()
            and not os.access(weight, os.R_OK)
            for weight in root.glob("model*.safetensors")
            for error in errors
        ):
            return (
                "unreadable_model_file",
                "Rapid-MLX cannot read a file in the local model directory.",
                "Grant your user read access to the model files, then retry.",
            )
        if (
            error_class == "local_path_missing"
            and not config_path.is_file()
            and any(
                isinstance(error, FileNotFoundError)
                and _same_local_path(error.filename, config_path)
                for error in errors
            )
        ):
            return (
                "missing_config",
                "The local model directory is missing config.json.",
                "Copy config.json from the same checkpoint as the model weights.",
            )
        if (
            error_class == "local_path_missing"
            and config_path.is_file()
            and any(
                isinstance(error, FileNotFoundError)
                and str(error).startswith("No safetensors found in ")
                and _same_local_path(
                    str(error).removeprefix("No safetensors found in "), root
                )
                for error in errors
            )
        ):
            return (
                "missing_weights",
                "The local model directory has no readable model safetensors files.",
                "Complete the model download or export its MLX safetensors weights into this directory.",
            )
        if (
            error_class in ("other", "corrupt_weights")
            and truncated_local_weights(model_ref)
            and any(
                isinstance(error, RuntimeError)
                and re.fullmatch(
                    r"\[read\] Unable to read \d+ bytes from file\.", str(error)
                )
                for error in errors
            )
        ):
            return (
                "incomplete_weights",
                "A model safetensors file is incomplete or unreadable.",
                "Re-download or re-export the matching model weights, then retry.",
            )
    except BaseException:
        return None
    return None


def _redownload_step(model_ref: object) -> str:
    """Next step for a checkpoint whose own files are incomplete."""
    from rapid_mlx.local_model_path import is_local_model_ref

    if isinstance(model_ref, str) and not is_local_model_ref(model_ref):
        return (
            f"Remove the cached copy (all revisions) with `rapid-mlx rm {model_ref}`, "
            f"download it again with `rapid-mlx pull {model_ref}`, then retry."
        )
    return (
        "Re-download or re-export the complete checkpoint into this "
        "directory, then retry."
    )


def _incomplete_checkpoint_diagnostic(
    errors: list[BaseException], model_ref: object, error_class: str
) -> tuple[str, str, str] | None:
    """Name missing checkpoint data: absent shards/tensors or truncated files."""
    from rapid_mlx.model_load_errors import IncompatibleWeights, IncompleteCheckpoint

    for error in errors:
        if isinstance(error, IncompleteCheckpoint):
            return ("incomplete_checkpoint", str(error), _redownload_step(model_ref))

    if error_class == "incompatible_weights":
        from rapid_mlx.models.mllm import (
            _parse_missing_count,
            _parse_missing_param_names,
        )

        for error in errors:
            if isinstance(error, IncompatibleWeights):
                names = _parse_missing_param_names(str(error))
                if names and _parse_missing_count(str(error)) == len(names):
                    return (
                        "missing_parameters",
                        f"The checkpoint is missing {len(names)} weights this "
                        f"model needs (for example {names[0]}).",
                        "Re-download the complete checkpoint, or use the "
                        "config.json that matches these weights.",
                    )

    if (
        error_class == "corrupt_weights"
        and isinstance(model_ref, str)
        and not any(
            isinstance(error, PermissionError)
            or str(error).startswith("[load_safetensors] Failed to open file ")
            for error in errors
        )
    ):
        from rapid_mlx.checkpoint_integrity import incomplete_checkpoint_reason

        reason = incomplete_checkpoint_reason(Path(model_ref).expanduser())
        if reason is not None:
            return (
                "incomplete_checkpoint",
                f"The model checkpoint is incomplete: {reason}.",
                _redownload_step(model_ref),
            )
    return None
