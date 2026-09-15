"""Converter registry and shared tensor helpers.

A :class:`Converter` knows how to turn one external component (a diffusers
subfolder such as ``vae/`` or ``transformer/``) into one of our models: it
translates the ``config.json`` into our :class:`~mlx_diffuser.configuration.Config`
and remaps the weight tensors onto the model's channels-last parameter tree.

The heavy lifting is a *build-and-fill* strategy: we instantiate the target model,
read its expected ``(key -> shape)`` tree, and assert every key is filled with a
matching shape. A converter that drifts from the architecture fails loudly here
rather than silently producing noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx.utils import tree_flatten

if TYPE_CHECKING:
    from ..modeling import ModelMixin

__all__ = [
    "Converter",
    "convert_conv_weight",
    "get_converter",
    "load_safetensors_folder",
    "register_converter",
]


def convert_conv_weight(w: mx.array) -> mx.array:
    """Reorder a PyTorch conv kernel to MLX channels-last layout.

    Conv2d ``(O, I, kH, kW) -> (O, kH, kW, I)``; Conv3d ``(O, I, kT, kH, kW) ->
    (O, kT, kH, kW, I)``. Other ranks are returned unchanged.
    """
    if w.ndim == 4:
        return w.transpose(0, 2, 3, 1)
    if w.ndim == 5:
        return w.transpose(0, 2, 3, 4, 1)
    return w


def load_safetensors_folder(folder: str | Path) -> dict[str, mx.array]:
    """Load and merge every ``*.safetensors`` shard in ``folder`` into one dict.

    MLX reads safetensors natively, so no PyTorch dependency is needed to convert
    weights. Sharded checkpoints (``*-00001-of-000NN.safetensors``) are merged.
    """
    folder = Path(folder)
    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {folder}")
    merged: dict[str, mx.array] = {}
    for shard in shards:
        merged.update(mx.load(str(shard)))  # type: ignore[arg-type]  # safetensors -> dict
    return merged


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


class Converter:
    """Base class for one component converter (one diffusers subfolder).

    Subclasses set :attr:`source_class` (the diffusers ``_class_name`` they accept)
    and implement :meth:`build_config` and :meth:`convert_weights`.
    """

    #: diffusers ``_class_name`` this converter consumes.
    source_class: str

    def build_config(self, hf_config: dict):
        """Translate a diffusers ``config.json`` dict into our ``Config``."""
        raise NotImplementedError

    def build_model(self, hf_config: dict) -> ModelMixin:
        """Instantiate the (empty) target model from a diffusers config."""
        raise NotImplementedError

    def convert_weights(
        self, weights: dict[str, mx.array], hf_config: dict
    ) -> dict[str, mx.array]:
        """Map a diffusers state dict onto our model's parameter keys."""
        raise NotImplementedError

    def convert(
        self,
        source: str | Path,
        *,
        dtype: mx.Dtype | None = None,
        quantize: int | None = None,
        quant_group_size: int = 64,
    ) -> ModelMixin:
        """Load a diffusers component and return a populated model.

        ``source`` is either a diffusers subfolder (``config.json`` + safetensors
        shards) or a single ``.safetensors`` file (config falls back to the
        converter's defaults). Weights are loaded strictly so any missing / extra /
        mis-shaped tensor raises immediately. ``dtype`` casts the floating-point
        weights; ``quantize`` (2/3/4/6/8) weight-quantizes the model.

        Conversion is memory-safe for very large encoders: safetensors are loaded
        lazily (memory-mapped) and only materialized at the final ``mx.eval``, so
        quantizing a multi-GB model never holds it all in RAM at once.
        """
        from ..quantization import quantize_module  # local import avoids a cycle

        source = Path(source)
        if source.is_dir():
            cfg = source / "config.json"
            hf_config = _load_json(cfg) if cfg.exists() else {}
            weights = load_safetensors_folder(source)
        else:
            # Standalone research checkpoints commonly use ``component.json`` next
            # to ``component.safetensors`` (TRELLIS does this), while diffusers uses
            # a generic ``config.json`` inside each component directory.
            named_sibling = source.with_suffix(".json")
            generic_sibling = source.parent / "config.json"
            sibling = named_sibling if named_sibling.exists() else generic_sibling
            hf_config = _load_json(sibling) if sibling.exists() else {}
            weights = mx.load(str(source))  # type: ignore[assignment]

        model = self.build_model(hf_config)
        converted = self.convert_weights(weights, hf_config)
        _assert_matches(model, converted)
        model.load_weights(list(converted.items()), strict=True)
        if dtype is not None:
            model.set_dtype(dtype)
        if quantize is not None:
            quantize_module(model, bits=quantize, group_size=quant_group_size)
        # Materialize the parameter tree in chunks rather than one `mx.eval`. The weights
        # are lazy (mmap'd); a single eval would force every source tensor resident at
        # once, spiking to the *unquantized* size (~24 GB for FLUX). We first drop our own
        # references to the source dicts so that, as each chunk is quantized, its bf16
        # source can be freed — then peak stays near the quantized footprint instead of
        # swapping. (For unquantized loads this is just a chunked eval, equally correct.)
        del converted, weights
        _eval_in_chunks(model.parameters())
        model.eval()
        return model


def _eval_in_chunks(params, chunk: int = 24) -> None:
    """Evaluate a parameter tree ``chunk`` leaves at a time to bound peak memory.

    Quantizing a multi-GB model in one ``mx.eval`` holds every source tensor resident at
    once; evaluating small groups (and releasing MLX's buffer cache between them) keeps
    only a handful of transient full-precision tensors live, so peak tracks the quantized
    result rather than the original. The caller must drop its references to the source
    (unquantized) weights first, or they stay resident regardless.
    """
    leaves = [v for _, v in tree_flatten(params)]  # type: ignore[union-attr, str-unpack]
    for i in range(0, len(leaves), chunk):
        mx.eval(leaves[i : i + chunk])
        mx.clear_cache()  # return the just-freed source buffers to the OS, not MLX's pool


def _assert_matches(model: ModelMixin, converted: dict[str, mx.array]) -> None:
    """Check the converted dict exactly covers the model's parameter tree."""
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}  # type: ignore[union-attr, str-unpack]
    got = {k: tuple(v.shape) for k, v in converted.items()}
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    mismatched = sorted(
        k for k in expected.keys() & got.keys() if expected[k] != got[k]
    )
    if missing or extra or mismatched:
        lines = []
        if missing:
            lines.append(f"missing {len(missing)} keys, e.g. {missing[:5]}")
        if extra:
            lines.append(f"extra {len(extra)} keys, e.g. {extra[:5]}")
        if mismatched:
            ex = [(k, expected[k], got[k]) for k in mismatched[:5]]
            lines.append(f"shape mismatch on {len(mismatched)}, e.g. {ex}")
        raise ValueError(
            "Converted weights do not match the model:\n  " + "\n  ".join(lines)
        )


_REGISTRY: dict[str, Converter] = {}


def register_converter(converter_cls: type[Converter]) -> type[Converter]:
    """Class decorator: register a converter under its ``source_class``."""
    _REGISTRY[converter_cls.source_class] = converter_cls()
    return converter_cls


def get_converter(source_class: str) -> Converter:
    """Look up a registered converter by diffusers ``_class_name``."""
    if source_class not in _REGISTRY:
        raise KeyError(
            f"No converter for '{source_class}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[source_class]
