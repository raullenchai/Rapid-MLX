"""``ModelMixin``: save/load behaviour shared by every network.

A model is an ``mlx.nn.Module`` that:

* takes its :class:`~mlx_diffuser.configuration.Config` as the sole ``__init__`` arg,
* declares ``config_class`` so it can be rebuilt from ``config.json``,
* stores that config on ``self.config``.

Everything else (``from_pretrained``, ``save_pretrained``, dtype casting,
quantization, parameter counting) is provided here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, TypeVar

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .configuration import Config
from .hub import push_folder, resolve
from .quantization import quantize_module
from .utils import as_dtype, get_logger

WEIGHTS_NAME = "model.safetensors"

C = TypeVar("C", bound=Config)
M = TypeVar("M", bound="ModelMixin")
logger = get_logger()


class ModelMixin(nn.Module, Generic[C]):
    """Base class for all mlx-diffuser networks (generic over its config type)."""

    #: The Config subclass this model is constructed from.
    config_class: type[C]

    config: C

    def save_pretrained(
        self, save_directory: str | Path, push_to_hub: str | None = None
    ) -> Path:
        """Write ``config.json`` + ``model.safetensors`` into ``save_directory``.

        If ``push_to_hub`` is a repo id, the folder is uploaded afterwards.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.save(save_directory)
        self.save_weights(str(save_directory / WEIGHTS_NAME))
        if push_to_hub:
            push_folder(save_directory, push_to_hub)
        return save_directory

    @classmethod
    def from_pretrained(
        cls: type[M],
        path_or_repo_id: str | Path,
        *,
        dtype: str | mx.Dtype | None = None,
        quantize: int | None = None,
        quant_group_size: int = 64,
        strict: bool = True,
        revision: str | None = None,
        **config_overrides: Any,
    ) -> M:
        """Load a model from a local directory or a Hub repo id.

        Args:
            path_or_repo_id: local dir (with ``config.json`` + weights) or Hub id.
            dtype: cast floating-point params to this dtype after loading.
            quantize: if set (2/3/4/6/8), weight-quantize after loading.
            quant_group_size: group size for quantization.
            strict: require the checkpoint to match the model's params exactly.
            revision: Hub revision (branch/tag/commit) when downloading.
            **config_overrides: override individual config fields at load time.
        """
        local = resolve(path_or_repo_id, revision=revision)
        config = cls.config_class.load(local)
        if config_overrides:
            config = config.replace(**config_overrides)

        model = cls(config)  # type: ignore[call-arg]  # subclasses take a config

        # Pre-quantized checkpoints (written by the converters) carry a
        # quantization.json sidecar: quantize the skeleton *before* loading so
        # the parameter tree matches the stored weight/scales/biases triplets.
        quant_marker = local / "quantization.json"
        if quant_marker.exists():
            import json

            spec = json.loads(quant_marker.read_text())
            quantize_module(model, bits=spec["bits"], group_size=spec["group_size"])

        weights_file = local / WEIGHTS_NAME
        if weights_file.exists():
            model.load_weights(str(weights_file), strict=strict)
        else:  # sharded checkpoint (model-*.safetensors)
            from .converters.base import load_safetensors_folder

            merged = load_safetensors_folder(local)
            model.load_weights(list(merged.items()), strict=strict)
            del merged

        resolved_dtype = as_dtype(dtype)
        if resolved_dtype is not None:
            model.set_dtype(resolved_dtype)
        if quantize is not None and not quant_marker.exists():
            quantize_module(model, bits=quantize, group_size=quant_group_size)

        from .converters.base import _eval_in_chunks  # local import avoids a cycle

        _eval_in_chunks(model.parameters())
        model.eval()
        return model

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Total number of parameter elements in the model."""
        params = self.trainable_parameters() if trainable_only else self.parameters()
        return sum(v.size for _, v in tree_flatten(params))  # type: ignore[union-attr, str-unpack]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = self.num_parameters()
        return f"{type(self).__name__}({n / 1e6:.1f}M params)"
