# SPDX-License-Identifier: Apache-2.0
"""Typed failures raised at model-load boundaries.

The exception messages remain local/user-facing. Telemetry consumes only the
exception types and never sends these messages.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")


class InvalidModelConfig(ValueError):  # noqa: N818 - domain name is user-facing
    """The checkpoint's ``config.json`` cannot drive a model load."""


class TokenizerLoadFailed(RuntimeError):  # noqa: N818 - domain name is user-facing
    """The checkpoint tokenizer assets or configuration cannot be loaded."""


class IncompatibleWeights(RuntimeError):  # noqa: N818 - domain name is user-facing
    """Checkpoint parameter names or shapes do not match the model."""


class QuantizationMismatch(RuntimeError):  # noqa: N818 - domain name is user-facing
    """Checkpoint quantization metadata and tensors are incompatible."""


def validate_model_config_file(model_path: str | Path) -> dict[str, Any] | None:
    """Validate a local ``config.json`` before MLX starts loading weights.

    Remote repository identifiers deliberately return ``None``; their config
    is validated after mlx-lm resolves the repository to a snapshot.
    """

    root = Path(model_path)
    if not root.is_dir():
        return None
    config_path = root / "config.json"
    if not config_path.is_file():
        return None
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        if not isinstance(config, dict):
            raise ValueError("the top-level value must be an object")
        if config.get("model_file") is None:
            model_type = config["model_type"]
            if not isinstance(model_type, str) or not model_type:
                raise ValueError("model_type must be a non-empty string")
    except (KeyError, ValueError) as exc:
        raise InvalidModelConfig(
            f"Invalid model config at {config_path}: {exc}"
        ) from exc
    return config


def load_tokenizer_checked(loader: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run a tokenizer loader and preserve its failure as a typed cause."""

    try:
        return loader(*args, **kwargs)
    except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
        raise TokenizerLoadFailed(f"Tokenizer loading failed: {exc}") from exc


def load_weights_checked(
    model: Any,
    weights: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    strict: bool,
) -> None:
    """Apply checkpoint weights while naming parameter and shape failures."""

    items = list(weights.items()) if isinstance(weights, Mapping) else list(weights)
    try:
        model.load_weights(items, strict=strict)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise IncompatibleWeights(f"Model weights are incompatible: {exc}") from exc


def quantize_checked(quantize: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Apply checkpoint quantization while naming metadata/tensor mismatches."""

    try:
        return quantize(*args, **kwargs)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise QuantizationMismatch(
            f"Model quantization is incompatible: {exc}"
        ) from exc


def _traceback_contains(exc: BaseException, function_names: set[str]) -> bool:
    traceback = exc.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name in function_names:
            return True
        traceback = traceback.tb_next
    return False


def load_model_checked(
    loader: Callable[..., tuple[Any, dict[str, Any]]],
    model_path: str | Path,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run an mlx-lm model loader and type its internal load phase."""

    validate_model_config_file(model_path)
    try:
        return loader(model_path, **kwargs)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        if _traceback_contains(
            exc,
            {"_quantize", "quantize", "quantized_matmul", "to_quantized"},
        ):
            raise QuantizationMismatch(
                f"Model quantization is incompatible: {exc}"
            ) from exc
        if _traceback_contains(exc, {"load_weights"}):
            raise IncompatibleWeights(f"Model weights are incompatible: {exc}") from exc
        raise


def load_mlx_lm_checked(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
    *,
    model_config: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Load an eager mlx-lm model with explicit model/tokenizer phases."""

    from mlx_lm.utils import _download, load_model, load_tokenizer

    model_path = _download(model_name)
    model, config = load_model_checked(
        load_model, model_path, model_config=model_config
    )
    tokenizer = load_tokenizer_checked(
        load_tokenizer,
        model_path,
        tokenizer_config,
        eos_token_ids=config.get("eos_token_id"),
    )
    return model, tokenizer
