# SPDX-License-Identifier: Apache-2.0
"""Local-only startup diagnostics; no checkpoint details enter telemetry."""

from __future__ import annotations

import json
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
    detail = local_serve_diagnostic(exc, model_ref)
    if detail is not None and detail[0] == error_class:
        return detail[1:]
    if error_class == "insufficient_memory":
        return (
            "The model ran out of memory while loading weights.",
            "Free memory or serve a smaller or more quantized model.",
        )
    return None
