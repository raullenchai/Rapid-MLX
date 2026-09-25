# SPDX-License-Identifier: Apache-2.0
"""Typed failures raised at model-load boundaries.

The exception messages remain local/user-facing. Telemetry consumes only the
exception types and never sends these messages.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from .runtime.optional_runtime import OptionalRuntimeMissing

_T = TypeVar("_T")


class InvalidModelConfig(ValueError):  # noqa: N818 - domain name is user-facing
    """The checkpoint's ``config.json`` cannot drive a model load."""


class TokenizerLoadFailed(RuntimeError):  # noqa: N818 - domain name is user-facing
    """The checkpoint tokenizer assets or configuration cannot be loaded."""


class IncompatibleWeights(RuntimeError):  # noqa: N818 - domain name is user-facing
    """Checkpoint parameter names or shapes do not match the model."""


class QuantizationMismatch(RuntimeError):  # noqa: N818 - domain name is user-facing
    """Checkpoint quantization metadata and tensors are incompatible."""


_TYPED_LOAD_FAILURES = (
    OptionalRuntimeMissing,
    InvalidModelConfig,
    TokenizerLoadFailed,
    IncompatibleWeights,
    QuantizationMismatch,
    HfHubHTTPError,
    RepositoryNotFoundError,
    FileNotFoundError,
    ModuleNotFoundError,
)
_MLX_LOAD_BOUNDARIES_ACTIVE: ContextVar[bool] = ContextVar(
    "mlx_load_boundaries_active", default=False
)


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
        model_file = config.get("model_file")
        if model_file is not None:
            if not isinstance(model_file, str) or not model_file:
                raise ValueError("model_file must be a non-empty string when present")
        else:
            model_type = config["model_type"]
            if not isinstance(model_type, str) or not model_type:
                raise ValueError("model_type must be a non-empty string")
    except _TYPED_LOAD_FAILURES:
        raise
    except (KeyError, ValueError) as exc:
        raise InvalidModelConfig(
            f"Invalid model config at {config_path}: {exc}"
        ) from exc
    return config


def load_tokenizer_checked(loader: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run a tokenizer loader and preserve its failure as a typed cause."""

    try:
        return loader(*args, **kwargs)
    except (HfHubHTTPError, RepositoryNotFoundError):
        # Preserve remote resolution/download failures for the existing Hub
        # error path. Local tokenizer assets instead belong to this boundary.
        raise
    except (
        OptionalRuntimeMissing,
        InvalidModelConfig,
        TokenizerLoadFailed,
        IncompatibleWeights,
        QuantizationMismatch,
        ModuleNotFoundError,
    ):
        raise
    except (OSError, ImportError, TypeError, ValueError) as exc:
        raise TokenizerLoadFailed(f"Tokenizer loading failed: {exc}") from exc


@contextmanager
def typed_weight_boundary() -> Iterator[None]:
    """Type only MLX's documented ``load_weights`` ``ValueError`` failures."""

    try:
        yield
    except _TYPED_LOAD_FAILURES:
        raise
    except ValueError as exc:
        raise IncompatibleWeights(f"Model weights are incompatible: {exc}") from exc


@contextmanager
def typed_quantization_boundary() -> Iterator[None]:
    """Type only MLX quantization ``ValueError`` failures."""

    try:
        yield
    except _TYPED_LOAD_FAILURES:
        raise
    except ValueError as exc:
        raise QuantizationMismatch(
            f"Model quantization is incompatible: {exc}"
        ) from exc


def load_weights_checked(
    model: Any,
    weights: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    strict: bool,
) -> None:
    """Apply checkpoint weights while naming parameter and shape failures."""

    items = list(weights.items()) if isinstance(weights, Mapping) else list(weights)
    with typed_weight_boundary():
        model.load_weights(items, strict=strict)


def quantize_checked(quantize: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Apply checkpoint quantization while naming metadata/tensor mismatches."""

    with typed_quantization_boundary():
        return quantize(*args, **kwargs)


def _install_mlx_load_boundary_hooks() -> None:
    """Install context-gated wrappers at mlx-lm's two shared load call sites."""

    try:
        import mlx.nn as nn
    except ImportError:
        return

    if not getattr(nn.quantize, "_rapid_mlx_typed_boundary", False):
        original_quantize = nn.quantize

        @wraps(original_quantize)
        def checked_quantize(*args: Any, **kwargs: Any) -> Any:
            if not _MLX_LOAD_BOUNDARIES_ACTIVE.get():
                return original_quantize(*args, **kwargs)
            with typed_quantization_boundary():
                return original_quantize(*args, **kwargs)

        checked_quantize._rapid_mlx_typed_boundary = True  # type: ignore[attr-defined]
        nn.quantize = checked_quantize

    if not getattr(nn.Module.load_weights, "_rapid_mlx_typed_boundary", False):
        original_load_weights = nn.Module.load_weights

        @wraps(original_load_weights)
        def checked_load_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _MLX_LOAD_BOUNDARIES_ACTIVE.get():
                return original_load_weights(self, *args, **kwargs)
            with typed_weight_boundary():
                return original_load_weights(self, *args, **kwargs)

        checked_load_weights._rapid_mlx_typed_boundary = True  # type: ignore[attr-defined]
        nn.Module.load_weights = checked_load_weights


@contextmanager
def typed_mlx_load_boundaries() -> Iterator[None]:
    """Enable the shared mlx-lm weight and quantization API boundaries."""

    _install_mlx_load_boundary_hooks()
    token = _MLX_LOAD_BOUNDARIES_ACTIVE.set(True)
    try:
        yield
    finally:
        _MLX_LOAD_BOUNDARIES_ACTIVE.reset(token)


def load_model_checked(
    loader: Callable[..., tuple[Any, dict[str, Any]]],
    model_path: str | Path,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run an mlx-lm model loader and type its internal load phase."""

    validate_model_config_file(model_path)
    with typed_mlx_load_boundaries():
        return loader(model_path, **kwargs)


def load_mlx_lm_checked(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
    *,
    model_config: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Load an eager mlx-lm model with explicit model/tokenizer phases."""

    from mlx_lm.utils import _download, load_model, load_tokenizer

    local_path = Path(model_name).expanduser()
    model_path = local_path if local_path.exists() else _download(model_name)
    model, config = load_model_checked(
        load_model, model_path, model_config=model_config
    )
    tokenizer = load_tokenizer_checked(
        load_tokenizer,
        model_path,
        tokenizer_config or {},
        eos_token_ids=config.get("eos_token_id"),
    )
    return model, tokenizer
