"""MLX LoRA-family adapters compatible with Stable Audio 3 checkpoints.

This module intentionally depends only on MLX and NumPy so it can be used by
the standalone optimized MLX runtime without pulling in PyTorch or safetensors.
"""

from __future__ import annotations

import json
import math
import os
import re
import typing as tp
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


_LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.+)\.parametrizations\.weight\.(?P<index>\d+)\."
    r"(?P<param>lora_A|lora_B|M_xs|magnitude|magnitude_r|magnitude_c|U|V)$"
)
_XS_ADAPTER_TYPES = {
    "lora-xs",
    "dora-rows-xs",
    "dora-cols-xs",
    "bora-xs",
}
_SUPPORTED_ADAPTER_TYPES = {
    "lora",
    "dora-rows",
    "dora-cols",
    "bora",
    *_XS_ADAPTER_TYPES,
}
_FULL_WEIGHT_ADAPTER_TYPES = _SUPPORTED_ADAPTER_TYPES - {"lora"}
# bora/bora-xs have a selectable forward (LoRALinear ``bora_mode``): their
# column norm is taken over the row-rescaled intermediate diag(α)·V with
# α = m_r/rownorm(V), which has no rank-r expansion without also storing
# W0**2. "speed" (default) caches that W0² once at init (+1 weight copy per
# bora layer, native dtype) and uses the reformulated low-rank forward;
# "memory" keeps the original full-weight-materializing forward and
# allocates no cache. Every other adapted type always uses the reformulated
# forward (see _reformulated_linear_forward), which never builds the
# [out, in] weight.
_BORA_ADAPTER_TYPES = {"bora", "bora-xs"}
_BORA_MODES = {"speed", "memory"}
_DORA_ROW_ADAPTER_TYPES = {"dora-rows", "dora-rows-xs"}
_DORA_COL_ADAPTER_TYPES = {"dora-cols", "dora-cols-xs"}

# Ablation/fallback toggle (read once at import): SA3_LORA_NAIVE_DORA=1
# restores the original full-weight forward for the reformulated types.
_NAIVE_DORA = os.environ.get("SA3_LORA_NAIVE_DORA", "") == "1"


@dataclass(frozen=True)
class LoRAInjectionReport:
    layer_names: tuple[str, ...]
    trainable_parameters: int
    adapter_type: str

    @property
    def layer_count(self) -> int:
        return len(self.layer_names)


@dataclass(frozen=True)
class LoRAApplyReport:
    path: str
    adapter_type: str
    loaded_layers: int
    applied_layers: int
    missing_targets: tuple[str, ...] = ()
    skipped_layers: tuple[str, ...] = ()


class LoRALinear(nn.Module):
    """Trainable LoRA-family wrapper for an MLX Linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        source_name: str,
        adapter_type: str = "lora",
        checkpoint_prefix: str = "model.",
        bora_mode: str = "speed",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if bora_mode not in _BORA_MODES:
            raise ValueError(
                f"bora_mode must be one of {sorted(_BORA_MODES)}, "
                f"got {bora_mode!r}."
            )

        self.base = base
        self.base.freeze()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.source_name = str(source_name)
        self.checkpoint_name = _checkpoint_name(source_name, checkpoint_prefix)
        self.adapter_type = canonical_adapter_type(adapter_type)
        self.bora_mode = str(bora_mode)

        fan_out, fan_in = (int(value) for value in base.weight.shape)
        source_weight = _linear_source_weight_2d(base.weight)
        _validate_rank(
            self.rank,
            fan_out=fan_out,
            fan_in=fan_in,
            source_name=self.source_name,
        )
        _initialize_adapter(self, source_weight, fan_out=fan_out, fan_in=fan_in)
        # Frozen-base row/column energy Σ W0² for the reformulated DoRA/BoRA
        # norms. The base never trains, so this is a constant computed once.
        # The leading underscore keeps it out of MLX parameters()/checkpoints.
        if self.adapter_type in _DORA_ROW_ADAPTER_TYPES:
            self._w0_sq = mx.sum(source_weight * source_weight, axis=1)
        elif self.adapter_type in _DORA_COL_ADAPTER_TYPES:
            self._w0_sq = mx.sum(source_weight * source_weight, axis=0)
        elif self.adapter_type in _BORA_ADAPTER_TYPES and self.bora_mode == "speed":
            # BoRA rownorm(V) reuses the dora-rows closed form (row Σ W0²);
            # its colnorm(diag(α)·V) additionally needs the FULL element-wise
            # W0² (α² re-weights the rows), cached once in the weight's native
            # dtype — this is speed mode's memory cost: +1 weight copy per
            # bora layer. "memory" mode allocates neither.
            self._w0_sq = mx.sum(source_weight * source_weight, axis=1)
            self._w0_sq_full = self.base.weight * self.base.weight

    def __call__(self, x):
        if self.adapter_type in _FULL_WEIGHT_ADAPTER_TYPES:
            if _NAIVE_DORA or (
                self.adapter_type in _BORA_ADAPTER_TYPES
                and self.bora_mode == "memory"
            ):
                return self._full_weight_forward(x)
            return _reformulated_linear_forward(self, x)

        base_output = self.base(x)
        adapter_output = (x.astype(mx.float32) @ self.lora_A.T) @ self.lora_B.T
        return base_output + (adapter_output * self.scaling).astype(base_output.dtype)

    def _full_weight_forward(self, x):
        """Original forward: materialize the full adapted weight per call.

        Kept verbatim as the bora/bora-xs ``bora_mode="memory"`` path and as
        the SA3_LORA_NAIVE_DORA=1 ablation fallback for the reformulated
        types.
        """

        adapted_weight = _adapted_weight_2d(
            _linear_source_weight_2d(self.base.weight),
            adapter_type=self.adapter_type,
            layer=self,
        )
        output = x.astype(mx.float32) @ adapted_weight.T
        bias = getattr(self.base, "bias", None)
        if bias is not None:
            output = output + bias.astype(mx.float32)
        return output.astype(x.dtype)


class LoRAConv1d(nn.Module):
    """Trainable LoRA-family wrapper for an MLX Conv1d layer.

    Stays on the original (full-weight for DoRA/BoRA) paths: the underfit
    product exclude list removes all convs, so this wrapper is never on the
    training hot path and does not need the reformulated forward.
    ``bora_mode`` intentionally does not apply here — bora/bora-xs convs
    always use the full-weight forward (equivalent to "memory" mode).
    """

    def __init__(
        self,
        base: nn.Conv1d,
        *,
        rank: int,
        alpha: float,
        source_name: str,
        adapter_type: str = "lora",
        checkpoint_prefix: str = "model.",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        self.base.freeze()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.source_name = str(source_name)
        self.checkpoint_name = _checkpoint_name(source_name, checkpoint_prefix)
        self.adapter_type = canonical_adapter_type(adapter_type)

        fan_out, kernel_size, fan_in_per_group = (
            int(value) for value in base.weight.shape
        )
        fan_in = fan_in_per_group * kernel_size
        source_weight = _conv1d_source_weight_2d(base.weight)
        _validate_rank(
            self.rank,
            fan_out=fan_out,
            fan_in=fan_in,
            source_name=self.source_name,
        )
        _initialize_adapter(self, source_weight, fan_out=fan_out, fan_in=fan_in)

    def __call__(self, x):
        fan_out, kernel_size, fan_in_per_group = (
            int(value) for value in self.base.weight.shape
        )

        if self.adapter_type in _FULL_WEIGHT_ADAPTER_TYPES:
            adapted_source = _adapted_weight_2d(
                _conv1d_source_weight_2d(self.base.weight),
                adapter_type=self.adapter_type,
                layer=self,
            )
            adapted_weight = _conv1d_weight_from_source_2d(
                adapted_source,
                fan_out=fan_out,
                fan_in_per_group=fan_in_per_group,
                kernel_size=kernel_size,
            )
            output = mx.conv1d(
                x.astype(mx.float32),
                adapted_weight,
                self.base.stride,
                self.base.padding,
                self.base.dilation,
                self.base.groups,
            )
            bias = getattr(self.base, "bias", None)
            if bias is not None:
                output = output + bias.astype(mx.float32)
            return output.astype(x.dtype)

        base_output = self.base(x)
        delta_weight = _conv1d_weight_from_source_2d(
            self.lora_B @ self.lora_A,
            fan_out=fan_out,
            fan_in_per_group=fan_in_per_group,
            kernel_size=kernel_size,
        )
        adapter_output = mx.conv1d(
            x.astype(mx.float32),
            delta_weight,
            self.base.stride,
            self.base.padding,
            self.base.dilation,
            self.base.groups,
        )
        return base_output + (adapter_output * self.scaling).astype(base_output.dtype)


TrainableLoRALayer = LoRALinear | LoRAConv1d


class _ExpoFourierFeatures(nn.Module):
    """MLX port of stable_audio_tools ExpoFourierFeatures (fp32, no params).

    Occupies ``embedder.embedding.0`` in TrainableSecondsEmbedder so the
    trainable Linear lands at ``embedder.embedding.1``, matching underfit's
    torch Sequential(ExpoFourierFeatures, Linear) checkpoint layout. The math
    mirrors sa3_pipeline.expo_fourier_features exactly.
    """

    def __init__(
        self,
        dim: int = 256,
        min_freq: float = 0.5,
        max_freq: float = 10000.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)

    def __call__(self, t: mx.array) -> mx.array:
        t = t.astype(mx.float32).reshape(-1, 1)
        half = self.dim // 2
        ramp = mx.arange(half, dtype=mx.float32) / max(half - 1, 1)
        freqs = mx.exp(
            ramp * (math.log(self.max_freq) - math.log(self.min_freq))
            + math.log(self.min_freq)
        )
        args = t * freqs * 2 * math.pi
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class TrainableSecondsEmbedder(nn.Module):
    """Trainable seconds_total conditioner (underfit checkpoint-compatible).

    Built from the baked conditioner weights as loaded by
    sa3_pipeline.load_conditioner_from_npz: ``weight`` [768, 256] fp32 and
    ``bias`` [768] fp32. Replicates SecondsTotalEmbedder's exact math (clip to
    [min_val, max_val], normalize, expo Fourier features, Linear) and returns
    the [B, 1, 768] seconds token.

    The module tree places the Linear at the runtime name
    ``embedder.embedding.1`` so injecting with
    ``checkpoint_prefix="conditioners.seconds_total."`` reproduces underfit's
    checkpoint key root ``conditioners.seconds_total.embedder.embedding.1``.
    When injected, the base Linear is frozen and only the adapter trains.
    """

    def __init__(
        self,
        weight: mx.array,
        bias: mx.array,
        *,
        min_val: float = 0.0,
        max_val: float = 384.0,
        fourier_dim: int = 256,
    ):
        super().__init__()
        weight = mx.array(weight).astype(mx.float32)
        bias = mx.array(bias).astype(mx.float32)
        fan_out, fan_in = (int(value) for value in weight.shape)
        if fan_in != int(fourier_dim):
            raise ValueError(
                f"Conditioner weight fan-in {fan_in} does not match "
                f"fourier_dim {fourier_dim}."
            )
        linear = nn.Linear(fan_in, fan_out)
        linear.weight = weight
        linear.bias = bias
        embedder = nn.Module()
        embedder.embedding = [_ExpoFourierFeatures(dim=int(fourier_dim)), linear]
        self.embedder = embedder
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.fourier_dim = int(fourier_dim)

    def __call__(self, seconds: float | tp.Sequence[float] | mx.array) -> mx.array:
        if isinstance(seconds, (int, float)):
            seconds = [float(seconds)]
        if not isinstance(seconds, mx.array):
            seconds = mx.array([float(value) for value in seconds], dtype=mx.float32)
        values = seconds.astype(mx.float32).reshape(-1)
        values = mx.clip(values, self.min_val, self.max_val)
        norm = (values - self.min_val) / (self.max_val - self.min_val)
        features = self.embedder.embedding[0](norm)  # (B, fourier_dim)
        token = self.embedder.embedding[1](features)  # (B, 768)
        return token[:, None, :]  # (B, 1, 768)


def inject_trainable_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float | None = None,
    include: tp.Sequence[str] | None = None,
    exclude: tp.Sequence[str] | None = None,
    adapter_type: str = "lora",
    checkpoint_prefix: str = "model.",
    bora_mode: str = "speed",
) -> LoRAInjectionReport:
    """Freeze an MLX model and replace selected Linear/Conv1d layers.

    ``checkpoint_prefix`` is prepended to each layer's remapped runtime name
    to form its checkpoint key root (underfit convention: DiT layers save as
    ``model.<name>``, the seconds conditioner as
    ``conditioners.seconds_total.<name>``). Include/exclude patterns match
    against both the bare runtime name and the full checkpoint name, so
    underfit dashboard filter strings work verbatim.

    ``bora_mode`` selects the bora/bora-xs Linear forward: "speed" (default)
    caches W0² per layer for the reformulated forward, "memory" keeps the
    original full-weight forward with no cache. It only affects bora-family
    Linear layers (Conv1d always uses the full-weight path).
    """

    alpha = float(rank if alpha is None else alpha)
    adapter_type = canonical_adapter_type(adapter_type)
    model.freeze()

    replacements: list[tuple[str, TrainableLoRALayer]] = []
    for name, layer in model.named_modules():
        if not name or not _name_is_selected(
            name,
            _checkpoint_name(name, checkpoint_prefix),
            include=include,
            exclude=exclude,
        ):
            continue
        if isinstance(layer, nn.Linear):
            replacement = LoRALinear(
                layer,
                rank=rank,
                alpha=alpha,
                source_name=name,
                adapter_type=adapter_type,
                checkpoint_prefix=checkpoint_prefix,
                bora_mode=bora_mode,
            )
        elif isinstance(layer, nn.Conv1d):
            replacement = LoRAConv1d(
                layer,
                rank=rank,
                alpha=alpha,
                source_name=name,
                adapter_type=adapter_type,
                checkpoint_prefix=checkpoint_prefix,
            )
        else:
            continue
        replacements.append((name, replacement))

    if not replacements:
        raise ValueError("No MLX Linear or Conv1d layers matched the LoRA filters.")

    model.update_modules(tree_unflatten(replacements))
    trainable_parameters = sum(
        int(value.size) for _, value in tree_flatten(model.trainable_parameters())
    )
    return LoRAInjectionReport(
        layer_names=tuple(name for name, _ in replacements),
        trainable_parameters=trainable_parameters,
        adapter_type=adapter_type,
    )


def underfit_lora_config() -> dict[str, tp.Any]:
    """Effective underfit product defaults (what the dashboard writes).

    The exclude list is exactly what underfit's dashboard puts in the
    ``lora_config`` metadata of trained checkpoints (confirmed against a real
    underfit checkpoint): adapters go on every attention/FF linear, while the
    embedders/projections/convs stay frozen.
    """

    return {
        "adapter_type": "dora-rows",
        "rank": 16,
        "alpha": 16,
        "include": None,
        "exclude": [
            "to_timestep_embed",
            "to_cond_embed",
            "to_global_embed",
            "to_local_embed",
            "global_cond_embedder",
            "project_in",
            "project_out",
            "preprocess_conv",
            "postprocess_conv",
        ],
    }


def inject_from_lora_config(
    model: nn.Module,
    lora_config: dict[str, tp.Any] | None,
    *,
    checkpoint_prefix: str = "model.",
    bora_mode: str = "speed",
) -> tuple[LoRAInjectionReport, dict[str, tp.Any]]:
    """Inject adapters from an underfit ``lora_config`` dict.

    Mirrors underfit's apply_lora_from_config config-layer fallbacks: rank
    defaults to 8, alpha defaults to rank, adapter_type defaults to "lora"
    (legacy "dora" resolves to "dora-rows"). Returns the injection report and
    the saved-config dict ({rank, alpha, adapter_type, include, exclude})
    ready to pass to save metadata. ``bora_mode`` is a runtime speed-vs-memory
    knob (see inject_trainable_lora), not checkpoint semantics, so it is not
    part of the saved config.
    """

    lora_config = dict(lora_config or {})
    rank = lora_config.get("rank")
    rank = 8 if rank is None else int(rank)
    alpha = lora_config.get("alpha")
    alpha = rank if alpha is None else alpha
    adapter_type = canonical_adapter_type(lora_config.get("adapter_type") or "lora")
    include = lora_config.get("include")
    exclude = lora_config.get("exclude")

    report = inject_trainable_lora(
        model,
        rank=rank,
        alpha=float(alpha),
        include=include,
        exclude=exclude,
        adapter_type=adapter_type,
        checkpoint_prefix=checkpoint_prefix,
        bora_mode=bora_mode,
    )
    saved_config = {
        "rank": rank,
        "alpha": alpha,
        "adapter_type": adapter_type,
        "include": include,
        "exclude": exclude,
    }
    return report, saved_config


def iter_trainable_lora_layers(
    model: nn.Module,
) -> tp.Iterator[TrainableLoRALayer]:
    for _, layer in model.named_modules():
        if isinstance(layer, (LoRALinear, LoRAConv1d)):
            yield layer


def save_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    include: tp.Sequence[str] | None = None,
    exclude: tp.Sequence[str] | None = None,
    extra_config: dict[str, tp.Any] | None = None,
) -> Path:
    """Save trainable adapters using the official SA3 safetensors contract."""

    layers = list(iter_trainable_lora_layers(model))
    if not layers:
        raise ValueError("The model has no trainable MLX LoRA layers to save.")

    ranks = {layer.rank for layer in layers}
    alphas = {layer.alpha for layer in layers}
    adapter_types = {layer.adapter_type for layer in layers}
    if len(ranks) != 1 or len(alphas) != 1 or len(adapter_types) != 1:
        raise ValueError("A checkpoint must use one rank, alpha, and adapter type.")

    rank = next(iter(ranks))
    alpha = next(iter(alphas))
    adapter_type = next(iter(adapter_types))
    state_dict: dict[str, mx.array] = {}
    for layer in layers:
        prefix = f"{layer.checkpoint_name}.parametrizations.weight.0"
        if adapter_type in _XS_ADAPTER_TYPES:
            state_dict[f"{prefix}.M_xs"] = layer.M_xs.astype(mx.float16)
        else:
            state_dict[f"{prefix}.lora_A"] = layer.lora_A.astype(mx.float16)
            state_dict[f"{prefix}.lora_B"] = layer.lora_B.astype(mx.float16)

        if adapter_type in {
            "dora-rows",
            "dora-cols",
            "dora-rows-xs",
            "dora-cols-xs",
        }:
            state_dict[f"{prefix}.magnitude"] = layer.magnitude.astype(mx.float16)
        elif adapter_type in {"bora", "bora-xs"}:
            state_dict[f"{prefix}.magnitude_r"] = layer.magnitude_r.astype(mx.float16)
            state_dict[f"{prefix}.magnitude_c"] = layer.magnitude_c.astype(mx.float16)

    config: dict[str, tp.Any] = {
        "rank": rank,
        # Underfit writes integral alphas as JSON ints — keep metadata parity.
        "alpha": int(alpha) if float(alpha).is_integer() else alpha,
        "adapter_type": adapter_type,
        "include": list(include) if include else None,
        "exclude": list(exclude) if exclude else None,
    }
    if extra_config:
        protected = {"rank", "alpha", "adapter_type", "include", "exclude"}
        overlap = protected.intersection(extra_config)
        if overlap:
            raise ValueError(
                "extra_config cannot override checkpoint fields: "
                + ", ".join(sorted(overlap))
            )
        config.update(extra_config)

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output_path),
        state_dict,
        metadata={"lora_config": json.dumps(config)},
    )
    return output_path


def load_trainable_lora_state(
    model: nn.Module,
    state_dict: dict[str, tp.Any],
) -> int:
    """Restore trainable adapter parameters from an underfit checkpoint.

    ``state_dict`` uses underfit checkpoint keys
    (``<checkpoint_name>.parametrizations.weight.<N>.<param>``). Each group is
    matched to the injected trainable layer with the same checkpoint_name;
    unmatched groups and unknown params are tolerated (torch strict=False
    semantics). 2-D DoRA magnitudes are squeezed to 1-D like underfit's
    prepare_dora_state_dict. Parameters are restored in fp32. Returns the
    number of layers restored.
    """

    grouped = _group_lora_state_dict(state_dict)
    layers = {
        layer.checkpoint_name: layer for layer in iter_trainable_lora_layers(model)
    }
    restored = 0
    for checkpoint_name, params in grouped.items():
        layer = layers.get(checkpoint_name)
        if layer is None:
            continue
        applied = False
        for param_name in (
            "lora_A",
            "lora_B",
            "magnitude",
            "magnitude_r",
            "magnitude_c",
            "M_xs",
        ):
            value = params.get(param_name)
            if value is None:
                continue
            current = getattr(layer, param_name, None)
            if current is None:
                continue
            array = np.asarray(value, dtype=np.float32)
            if param_name == "magnitude" and array.ndim == 2:
                array = array.squeeze()
            if tuple(array.shape) != tuple(current.shape):
                raise ValueError(
                    f"Checkpoint tensor {checkpoint_name}.{param_name} has "
                    f"shape {tuple(array.shape)}, expected "
                    f"{tuple(current.shape)}."
                )
            setattr(layer, param_name, mx.array(array, dtype=mx.float32))
            applied = True
        restored += int(applied)
    return restored


def load_lora_checkpoint(
    path: str | Path,
) -> tuple[dict[str, mx.array], dict[str, tp.Any]]:
    """Load an SA3 LoRA safetensors checkpoint without PyTorch."""

    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError("The standalone MLX runtime supports .safetensors LoRAs.")
    state_dict, metadata = mx.load(str(checkpoint_path), return_metadata=True)
    config = {}
    if metadata and metadata.get("lora_config"):
        config = json.loads(metadata["lora_config"])
    return dict(state_dict), config


def apply_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strength: float = 1.0,
) -> LoRAApplyReport:
    """Materialize one checkpoint into a loaded MLX model at a fixed strength.

    Only the model's in-memory weights are changed; the base checkpoint on disk
    is untouched. The operation is not reversible because the original target
    weights are not retained. Reload the base model before applying a different
    strength instead of calling this function repeatedly on the same instance.
    """

    state_dict, config = load_lora_checkpoint(path)
    adapter_type = _adapter_type_from_state(
        config.get("adapter_type", "lora"),
        state_dict,
    )
    if adapter_type not in _SUPPORTED_ADAPTER_TYPES:
        raise ValueError(f"Unsupported MLX LoRA adapter type: {adapter_type!r}")

    grouped = _group_lora_state_dict(state_dict)
    target_params = dict(tree_flatten(model.parameters()))
    target_keys = tuple(target_params)
    missing_targets: list[str] = []
    skipped_layers: list[str] = []
    applied_layers = 0

    global_rank = int(config.get("rank") or _infer_global_rank(grouped) or 0)
    alpha_value = config.get("alpha", config.get("lora_alpha"))
    alpha = float(alpha_value if alpha_value is not None else (global_rank or 1))

    for source_name, params in grouped.items():
        target_key = _resolve_target_key(f"{source_name}.weight", target_keys)
        if target_key is None:
            missing_targets.append(source_name)
            continue
        try:
            adapted = _apply_checkpoint_layer(
                target_params[target_key],
                params,
                adapter_type=adapter_type,
                alpha=alpha,
                strength=float(strength),
            )
        except ValueError as exc:
            skipped_layers.append(f"{source_name}: {exc}")
            continue

        target_dtype = target_params[target_key].dtype
        updated = mx.array(adapted)
        if updated.dtype != target_dtype:
            updated = updated.astype(target_dtype)
        model.update(tree_unflatten([(target_key, updated)]))
        target_params[target_key] = updated
        applied_layers += 1

    if applied_layers:
        mx.eval(model.parameters())

    return LoRAApplyReport(
        path=str(Path(path).expanduser().resolve()),
        adapter_type=adapter_type,
        loaded_layers=len(grouped),
        applied_layers=applied_layers,
        missing_targets=tuple(sorted(set(missing_targets))),
        skipped_layers=tuple(skipped_layers),
    )


def apply_lora_checkpoints(
    model: nn.Module,
    paths: tp.Sequence[str | Path],
    *,
    strengths: float | tp.Sequence[float] = 1.0,
) -> tuple[LoRAApplyReport, ...]:
    """Materialize an ordered checkpoint stack into a loaded MLX model.

    This is a fixed-strength, in-place operation. In particular, DoRA and BoRA
    composition is order-dependent. Callers that need mutable strengths should
    retain canonical base weights and rebuild the complete ordered stack from
    those values rather than applying updates cumulatively.
    """

    if isinstance(strengths, (int, float)):
        values = [float(strengths)] * len(paths)
    else:
        values = [float(value) for value in strengths]
        if len(values) == 1:
            values *= len(paths)
        if len(values) != len(paths):
            raise ValueError(
                f"Expected 1 or {len(paths)} strengths, got {len(values)}."
            )

    return tuple(
        apply_lora_checkpoint(model, path, strength=strength)
        for path, strength in zip(paths, values, strict=True)
    )


def canonical_adapter_type(adapter_type: str) -> str:
    adapter_type = str(adapter_type or "lora").strip().lower()
    aliases = {
        "dora": "dora-rows",
        "dora-xs": "dora-rows-xs",
        "xs": "lora-xs",
    }
    adapter_type = aliases.get(adapter_type, adapter_type)
    if adapter_type not in _SUPPORTED_ADAPTER_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_ADAPTER_TYPES))
        raise ValueError(
            f"Unsupported MLX adapter type {adapter_type!r}. Expected one of: "
            f"{supported}."
        )
    return adapter_type


def _initialize_adapter(layer, source_weight, *, fan_out: int, fan_in: int) -> None:
    if layer.adapter_type in _XS_ADAPTER_TYPES:
        layer.U, layer.V = _svd_bases(source_weight, layer.rank)
        layer.M_xs = mx.zeros((layer.rank, layer.rank), dtype=mx.float32)
        layer.freeze(keys=["U", "V"], recurse=False)
    else:
        init_scale = 1.0 / math.sqrt(fan_in)
        layer.lora_A = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(layer.rank, fan_in),
            dtype=mx.float32,
        )
        layer.lora_B = mx.zeros((fan_out, layer.rank), dtype=mx.float32)

    if layer.adapter_type in {"dora-rows", "dora-rows-xs"}:
        layer.magnitude = _row_norms(source_weight)
    elif layer.adapter_type in {"dora-cols", "dora-cols-xs"}:
        layer.magnitude = _column_norms(source_weight)
    elif layer.adapter_type in {"bora", "bora-xs"}:
        layer.magnitude_r = _row_norms(source_weight)
        layer.magnitude_c = _column_norms(source_weight)


def _apply_checkpoint_layer(
    target_weight,
    params: dict[str, np.ndarray],
    *,
    adapter_type: str,
    alpha: float,
    strength: float,
) -> np.ndarray:
    target = np.asarray(target_weight, dtype=np.float32)
    if strength == 0:
        return target

    if adapter_type in _XS_ADAPTER_TYPES:
        source_shape = _source_shape_for_xs(tuple(target.shape), params)
    else:
        delta, _ = _lora_delta_2d(params)
        source_shape = _source_shape_for_delta(tuple(target.shape), delta.shape)

    source = _target_to_source_weight(target, source_shape)
    base_2d = source.reshape(source_shape[0], -1).astype(np.float32, copy=False)
    if adapter_type in _XS_ADAPTER_TYPES:
        delta, rank = _xs_delta_2d(params, base_2d)
    else:
        delta, rank = _lora_delta_2d(params)

    value = base_2d + (float(alpha) / rank) * strength * delta
    if adapter_type in {"lora", "lora-xs"}:
        adapted = value
    elif adapter_type in {"dora-rows", "dora-rows-xs"}:
        adapted = _dora_weight_2d(
            value,
            magnitude=_require_param(params, "magnitude").reshape(-1),
            norm_dim=1,
        )
    elif adapter_type in {"dora-cols", "dora-cols-xs"}:
        adapted = _dora_weight_2d(
            value,
            magnitude=_require_param(params, "magnitude").reshape(-1),
            norm_dim=0,
        )
    else:
        adapted = _bora_weight_2d(
            value,
            magnitude_r=_require_param(params, "magnitude_r").reshape(-1),
            magnitude_c=_require_param(params, "magnitude_c").reshape(-1),
        )

    return _source_to_target_weight(adapted.reshape(source_shape), target.shape)


def _adapted_weight_2d(weight_2d, *, adapter_type: str, layer):
    value = weight_2d.astype(mx.float32) + _adapter_delta_2d(layer) * float(
        layer.scaling
    )
    if adapter_type in {"lora", "lora-xs"}:
        return value
    if adapter_type in {"dora-rows", "dora-rows-xs"}:
        return _dora_weight_2d(value, magnitude=layer.magnitude, norm_dim=1)
    if adapter_type in {"dora-cols", "dora-cols-xs"}:
        return _dora_weight_2d(value, magnitude=layer.magnitude, norm_dim=0)
    return _bora_weight_2d(
        value,
        magnitude_r=layer.magnitude_r,
        magnitude_c=layer.magnitude_c,
    )


def _adapter_delta_2d(layer):
    if layer.adapter_type in _XS_ADAPTER_TYPES:
        return layer.U @ layer.M_xs.astype(mx.float32) @ layer.V.T
    return layer.lora_B @ layer.lora_A


def _effective_low_rank_factors(layer):
    """Return fp32 ``(A, B)`` with ``delta == B @ A``, folding -xs cores.

    For the -xs variants the effective factors are ``B̃ = U @ M_xs`` [out, r]
    and ``Ã = V.T`` [r, in] (U, V frozen); B̃ is recomputed per call — it is a
    cheap [out, rank] product, and gradients flow to M_xs through it.
    """

    if layer.adapter_type in _XS_ADAPTER_TYPES:
        return layer.V.T, layer.U @ layer.M_xs.astype(mx.float32)
    return layer.lora_A, layer.lora_B


def _reformulated_linear_forward(layer, x):
    """Adapted-linear forward without materializing the [out, in] weight.

    Mathematically identical to ``x @ _adapted_weight_2d(...).T + bias`` for
    lora-xs / dora-rows(-xs) / dora-cols(-xs) / bora(-xs), with
    V = W0 + s·B@A:

    * lora-xs:   y = x @ W0.T + s·((x @ Ã.T) @ B̃.T) + bias
    * dora-rows: W' = diag(m / rownorm(V)) · V, a row scale of the output:
      y = (x @ W0.T + s·((x @ A.T) @ B.T)) ⊙ c + bias,  c = m / rownorm(V)
    * dora-cols: W' = V · diag(m / colnorm(V)), a scale of the input features
      (x @ W'.T = (x ⊙ c) @ V.T):
      y = ((x ⊙ c) @ W0.T + s·(((x ⊙ c) @ A.T) @ B.T)) + bias
    * bora(-xs): W' = diag(α) · V · diag(β), both scales at once (see
      _bora_reformulated_forward)

    The bias is NOT parametrized (torch applies the parametrization to the
    weight only: y = x @ W'.T + bias), so it is added un-scaled after the
    row/column scaling. The base matmul runs in the layer's native dtype;
    the low-rank term, norms, and scaling are fp32; the result is cast back
    to x.dtype like the original path.
    """

    if layer.adapter_type in _BORA_ADAPTER_TYPES:
        return _bora_reformulated_forward(layer, x)

    a, b = _effective_low_rank_factors(layer)
    scaling = float(layer.scaling)
    weight = layer.base.weight
    bias = getattr(layer.base, "bias", None)

    x32 = x.astype(mx.float32)
    if layer.adapter_type in _DORA_COL_ADAPTER_TYPES:
        col_scale = _dora_scale_no_materialize(layer, a, b, weight, norm_dim=0)
        x32 = x32 * col_scale
        base_output = x32.astype(x.dtype) @ weight.T
    else:
        base_output = x @ weight.T
    output = base_output.astype(mx.float32) + ((x32 @ a.T) @ b.T) * scaling
    if layer.adapter_type in _DORA_ROW_ADAPTER_TYPES:
        output = output * _dora_scale_no_materialize(
            layer, a, b, weight, norm_dim=1
        )
    if bias is not None:
        output = output + bias.astype(mx.float32)
    return output.astype(x.dtype)


def _bora_reformulated_forward(layer, x):
    """BoRA speed-mode forward: W' = diag(α)·V·diag(β) without building V.

    With V = W0 + s·B@A, α = m_r / rownorm(V) and
    β = m_c / colnorm(diag(α)·V):

      y = (((x ⊙ β) @ W0.T + s·(((x ⊙ β) @ A.T) @ B.T)) ⊙ α) + bias

    α reuses the dora-rows closed-form rownorm (_v_norm_no_materialize with
    the cached row ΣW0²); β needs the α²-reweighted column norm, which is
    where the cached full W0² comes in (_bora_col_scale_no_materialize). α
    depends on m_r, A, B; β depends on everything including α — all
    differentiable, autodiff handles it. Same dtype policy as the other
    reformulated types: base matmul in the layer's native dtype, low-rank
    term and scales in fp32, bias un-scaled last, result cast to x.dtype.
    """

    a, b = _effective_low_rank_factors(layer)
    scaling = float(layer.scaling)
    weight = layer.base.weight
    bias = getattr(layer.base, "bias", None)

    row_scale = layer.magnitude_r.astype(mx.float32) / _v_norm_no_materialize(
        layer, a, b, weight, norm_dim=1
    )
    col_scale = _bora_col_scale_no_materialize(layer, a, b, weight, row_scale)

    x32 = x.astype(mx.float32) * col_scale
    base_output = x32.astype(x.dtype) @ weight.T
    output = base_output.astype(mx.float32) + ((x32 @ a.T) @ b.T) * scaling
    output = output * row_scale
    if bias is not None:
        output = output + bias.astype(mx.float32)
    return output.astype(x.dtype)


def _bora_col_scale_no_materialize(layer, a, b, weight, row_scale):
    """``m_c / colnorm(diag(α)·V)`` without materializing V or diag(α)·V.

    colnorm²(diag(α)·V) per column j is Σᵢ αᵢ²·Vᵢⱼ², expanded into three
    terms with V = W0 + s·B@A:

      1. Σᵢ αᵢ²·W0ᵢⱼ²      = α² @ W0²         (cached _w0_sq_full, native
                                               dtype — one weight-sized read)
      2. 2s·Σᵢ αᵢ²·W0ᵢⱼ·(BA)ᵢⱼ = 2s·rowsum((W0.T @ (α²[:,None] ⊙ B)) ⊙ A.T)
                                               (one W0 read into a rank-r
                                               matmul)
      3. s²·Σᵢ αᵢ²·(BA)ᵢⱼ²  = s²·colsum((M@A) ⊙ A),  M = B.T @ (α²[:,None]⊙B)
                                               [r, r] — tiny

    Term 1 runs the matvec in the cache's native dtype (fp16 for an fp16
    base — accepted accumulation precision, validated by the equivalence
    tests) and casts the result to fp32; terms 2-3 follow the existing
    _v_norm_no_materialize dtype pattern.
    """

    scaling = float(layer.scaling)
    alpha_sq = row_scale * row_scale  # [out] fp32
    w0_sq = layer._w0_sq_full
    base_term = mx.matmul(
        alpha_sq.astype(w0_sq.dtype)[None, :], w0_sq
    ).astype(mx.float32)[0]  # α² @ W0² -> [in]
    weighted_b = alpha_sq[:, None] * b  # [out, r] fp32
    cross_lhs = mx.matmul(weight.T, weighted_b.astype(weight.dtype)).astype(
        mx.float32
    )  # W0.T @ (α² ⊙ B) -> [in, r]
    cross = mx.sum(cross_lhs * a.T, axis=1)
    gram = b.T @ weighted_b
    quad = mx.sum((gram @ a) * a, axis=0)
    norm_sq = base_term + 2.0 * scaling * cross + scaling * scaling * quad
    # Same clamp semantics as _v_norm_no_materialize / _bora_weight_2d.
    norm = mx.sqrt(mx.maximum(norm_sq, 0.0))
    return layer.magnitude_c.astype(mx.float32) / mx.maximum(norm, 1e-12)


def _dora_scale_no_materialize(layer, a, b, weight, *, norm_dim: int):
    """``m / norm(V, axis=norm_dim)`` without materializing V = W0 + s·B@A."""

    return layer.magnitude.astype(mx.float32) / _v_norm_no_materialize(
        layer, a, b, weight, norm_dim=norm_dim
    )


def _v_norm_no_materialize(layer, a, b, weight, *, norm_dim: int):
    """Clamped ``norm(V, axis=norm_dim)`` without materializing V = W0 + s·B@A.

    Expands the squared norm so only rank-r matmuls touch W0 (read once, in
    its native dtype), with the constant ΣW0² cached at init:

      rownorm² = ΣW0²(rows) + 2s·rowsum((W0 @ A.T) ⊙ B) + s²·rowsum((B@G) ⊙ B)
                 with G = A @ A.T  [r, r]
      colnorm² = ΣW0²(cols) + 2s·rowsum((W0.T @ B) ⊙ A.T) + s²·colsum((M@A) ⊙ A)
                 with M = B.T @ B  [r, r]
    """

    scaling = float(layer.scaling)
    if norm_dim == 1:
        cross_lhs = mx.matmul(weight, a.T.astype(weight.dtype)).astype(
            mx.float32
        )  # W0 @ A.T -> [out, r]
        cross = mx.sum(cross_lhs * b, axis=1)
        gram = a @ a.T
        quad = mx.sum((b @ gram) * b, axis=1)
    else:
        cross_lhs = mx.matmul(weight.T, b.astype(weight.dtype)).astype(
            mx.float32
        )  # W0.T @ B -> [in, r]
        cross = mx.sum(cross_lhs * a.T, axis=1)
        gram = b.T @ b
        quad = mx.sum((gram @ a) * a, axis=0)
    norm_sq = layer._w0_sq + 2.0 * scaling * cross + scaling * scaling * quad
    # Match _dora_weight_2d semantics: clamp the NORM (not norm²) at 1e-12;
    # the max(·, 0) only guards tiny negative rounding in the expansion.
    norm = mx.sqrt(mx.maximum(norm_sq, 0.0))
    return mx.maximum(norm, 1e-12)


def _linear_source_weight_2d(weight):
    return weight.astype(mx.float32)


def _conv1d_source_weight_2d(weight):
    fan_out, kernel_size, fan_in_per_group = (int(value) for value in weight.shape)
    return (
        weight.astype(mx.float32)
        .transpose(0, 2, 1)
        .reshape(
            fan_out,
            fan_in_per_group * kernel_size,
        )
    )


def _conv1d_weight_from_source_2d(
    source,
    *,
    fan_out: int,
    fan_in_per_group: int,
    kernel_size: int,
):
    return source.reshape(
        fan_out,
        fan_in_per_group,
        kernel_size,
    ).transpose(0, 2, 1)


def _row_norms(weight_2d):
    return mx.sqrt(mx.sum(weight_2d.astype(mx.float32) ** 2, axis=1)).astype(mx.float32)


def _column_norms(weight_2d):
    return mx.sqrt(mx.sum(weight_2d.astype(mx.float32) ** 2, axis=0)).astype(mx.float32)


def _dora_weight_2d(value, *, magnitude, norm_dim: int):
    if isinstance(value, np.ndarray):
        norms = np.linalg.norm(value, axis=norm_dim, keepdims=True)
        value_hat = value / np.maximum(norms, 1e-12)
        if norm_dim == 1:
            if magnitude.shape[0] != value.shape[0]:
                raise ValueError("DoRA row magnitude does not match the weight.")
            return value_hat * magnitude[:, None]
        if magnitude.shape[0] != value.shape[1]:
            raise ValueError("DoRA column magnitude does not match the weight.")
        return value_hat * magnitude[None, :]

    norms = mx.sqrt(mx.sum(value**2, axis=norm_dim, keepdims=True))
    value_hat = value / mx.maximum(norms, 1e-12)
    if norm_dim == 1:
        return value_hat * magnitude.astype(mx.float32)[:, None]
    return value_hat * magnitude.astype(mx.float32)[None, :]


def _bora_weight_2d(value, *, magnitude_r, magnitude_c):
    if isinstance(value, np.ndarray):
        if magnitude_r.shape[0] != value.shape[0]:
            raise ValueError("BoRA row magnitude does not match the weight.")
        if magnitude_c.shape[0] != value.shape[1]:
            raise ValueError("BoRA column magnitude does not match the weight.")
        row_norms = np.linalg.norm(value, axis=1, keepdims=True)
        row_scaled = value / np.maximum(row_norms, 1e-12)
        row_scaled *= magnitude_r[:, None]
        column_norms = np.linalg.norm(row_scaled, axis=0, keepdims=True)
        return (row_scaled / np.maximum(column_norms, 1e-12)) * magnitude_c[None, :]

    row_norms = mx.sqrt(mx.sum(value**2, axis=1, keepdims=True))
    row_scaled = (value / mx.maximum(row_norms, 1e-12)) * magnitude_r.astype(
        mx.float32
    )[:, None]
    column_norms = mx.sqrt(mx.sum(row_scaled**2, axis=0, keepdims=True))
    return (row_scaled / mx.maximum(column_norms, 1e-12)) * magnitude_c.astype(
        mx.float32
    )[None, :]


def _group_lora_state_dict(
    state_dict: dict[str, tp.Any],
) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, value in state_dict.items():
        match = _LORA_KEY_RE.match(key)
        if match is None:
            continue
        grouped.setdefault(match.group("prefix"), {})[match.group("param")] = (
            np.asarray(value, dtype=np.float32)
        )
    return grouped


def _adapter_type_from_state(
    adapter_type: str,
    state_dict: dict[str, tp.Any],
) -> str:
    raw_type = str(adapter_type or "lora").strip().lower()
    keys = tuple(state_dict)
    has_xs = any(key.endswith(".M_xs") for key in keys)
    if has_xs:
        if raw_type in {"bora", "bora-xs"} or any(
            key.endswith((".magnitude_r", ".magnitude_c")) for key in keys
        ):
            return "bora-xs"
        if raw_type in {"dora-cols", "dora-cols-xs"}:
            return "dora-cols-xs"
        if raw_type in {"dora", "dora-rows", "dora-rows-xs"} or any(
            key.endswith(".magnitude") for key in keys
        ):
            return "dora-rows-xs"
        return "lora-xs"
    if raw_type == "lora":
        if any(key.endswith((".magnitude_r", ".magnitude_c")) for key in keys):
            return "bora"
        if any(key.endswith(".magnitude") for key in keys):
            return "dora-rows"
    return canonical_adapter_type(raw_type)


def _infer_global_rank(grouped: dict[str, dict[str, np.ndarray]]) -> int:
    ranks = {
        rank for params in grouped.values() if (rank := _rank_from_params(params)) > 0
    }
    if not ranks:
        return 0
    if len(ranks) > 1:
        raise ValueError(f"Multiple adapter ranks found: {sorted(ranks)}.")
    return next(iter(ranks))


def _rank_from_params(params: dict[str, np.ndarray]) -> int:
    core = params.get("M_xs")
    if core is not None and core.ndim == 2 and core.shape[0] == core.shape[1]:
        return int(core.shape[0])
    adapter_a = params.get("lora_A")
    adapter_b = params.get("lora_B")
    if adapter_a is None or adapter_b is None:
        return 0
    if adapter_b.shape[-1] == adapter_a.shape[0]:
        return int(adapter_a.shape[0])
    if adapter_a.shape[-1] == adapter_b.shape[0]:
        return int(adapter_a.shape[-1])
    return 0


def _lora_delta_2d(
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    adapter_a = _require_param(params, "lora_A").astype(np.float64)
    adapter_b = _require_param(params, "lora_B").astype(np.float64)
    if adapter_b.shape[-1] == adapter_a.shape[0]:
        delta = adapter_b @ adapter_a
        rank = adapter_a.shape[0]
    elif adapter_a.shape[-1] == adapter_b.shape[0]:
        delta = adapter_a @ adapter_b
        rank = adapter_a.shape[-1]
    else:
        raise ValueError(
            "Unable to multiply LoRA matrices with shapes "
            f"A={adapter_a.shape}, B={adapter_b.shape}."
        )
    if not np.isfinite(delta).all():
        raise ValueError("LoRA delta contains non-finite values.")
    return delta.astype(np.float32), int(rank)


def _xs_delta_2d(
    params: dict[str, np.ndarray],
    base_2d: np.ndarray,
) -> tuple[np.ndarray, int]:
    core = _require_param(params, "M_xs")
    if core.ndim != 2 or core.shape[0] != core.shape[1]:
        raise ValueError(f"LoRA-XS core must be square, got {core.shape}.")
    rank = int(core.shape[0])
    if rank > min(base_2d.shape):
        raise ValueError(
            f"LoRA-XS rank {rank} exceeds base weight shape {base_2d.shape}."
        )

    u = params.get("U")
    v = params.get("V")
    if u is None or v is None:
        u, v = _svd_bases_numpy(base_2d, rank)
    delta = u.astype(np.float64) @ core.astype(np.float64) @ v.astype(np.float64).T
    if not np.isfinite(delta).all():
        raise ValueError("LoRA-XS delta contains non-finite values.")
    return delta.astype(np.float32), rank


def _require_param(params: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = params.get(name)
    if value is None:
        raise ValueError(f"Adapter layer is missing {name}.")
    return value.astype(np.float32, copy=False)


def _resolve_target_key(
    source_weight_key: str,
    target_keys: tuple[str, ...],
) -> str | None:
    target_key_set = set(target_keys)
    candidates = _target_key_candidates(source_weight_key)
    for candidate in candidates:
        if candidate in target_key_set:
            return candidate

    suffix_matches = {
        target_key
        for candidate in candidates
        for target_key in target_keys
        if target_key.endswith(candidate)
    }
    if len(suffix_matches) == 1:
        return next(iter(suffix_matches))
    return None


def _target_key_candidates(source_weight_key: str) -> tuple[str, ...]:
    prefixes = ("model.model.", "model.")
    candidates = [source_weight_key]
    for prefix in prefixes:
        if source_weight_key.startswith(prefix):
            candidates.append(source_weight_key[len(prefix) :])

    candidates.extend(
        candidate.replace("to_local_embed.0.", "to_local_embed.seq.0.").replace(
            "to_local_embed.2.", "to_local_embed.seq.2."
        )
        for candidate in tuple(candidates)
    )
    return tuple(dict.fromkeys(candidates))


def _checkpoint_layer_name(name: str) -> str:
    return name.replace("to_local_embed.seq.0", "to_local_embed.0").replace(
        "to_local_embed.seq.2", "to_local_embed.2"
    )


def _checkpoint_name(source_name: str, checkpoint_prefix: str) -> str:
    return str(checkpoint_prefix or "") + _checkpoint_layer_name(str(source_name))


def _source_shape_for_delta(
    target_shape: tuple[int, ...],
    delta_shape: tuple[int, int],
) -> tuple[int, ...]:
    if len(target_shape) == 2:
        candidates = (target_shape, (target_shape[1], target_shape[0]))
    elif len(target_shape) == 3:
        candidates = (
            (target_shape[0], target_shape[2], target_shape[1]),
            target_shape,
        )
    else:
        candidates = (target_shape,)

    for candidate in candidates:
        if (
            candidate[0] == delta_shape[0]
            and int(np.prod(candidate[1:])) == delta_shape[1]
        ):
            return candidate
    raise ValueError(
        f"Unable to map LoRA delta {delta_shape} to target shape {target_shape}."
    )


def _source_shape_for_xs(
    target_shape: tuple[int, ...],
    params: dict[str, np.ndarray],
) -> tuple[int, ...]:
    u = params.get("U")
    v = params.get("V")
    if u is not None and v is not None:
        return _source_shape_for_delta(
            target_shape,
            (int(u.shape[0]), int(v.shape[0])),
        )
    if len(target_shape) == 3:
        return (target_shape[0], target_shape[2], target_shape[1])
    return target_shape


def _target_to_source_weight(
    target: np.ndarray,
    source_shape: tuple[int, ...],
) -> np.ndarray:
    if target.shape == source_shape:
        return target
    if target.ndim == 2 and target.T.shape == source_shape:
        return target.T
    if target.ndim == 3:
        candidate = target.transpose(0, 2, 1)
        if candidate.shape == source_shape:
            return candidate
    raise ValueError(
        f"Unable to map target shape {target.shape} to source shape {source_shape}."
    )


def _source_to_target_weight(
    source: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    if source.shape == target_shape:
        return source
    if source.ndim == 2 and source.T.shape == target_shape:
        return source.T
    if source.ndim == 3:
        candidate = source.transpose(0, 2, 1)
        if candidate.shape == target_shape:
            return candidate
    raise ValueError(
        f"Unable to map source shape {source.shape} to target shape {target_shape}."
    )


def _validate_rank(
    rank: int,
    *,
    fan_out: int,
    fan_in: int,
    source_name: str,
) -> None:
    max_rank = min(fan_out, fan_in)
    if rank > max_rank:
        raise ValueError(
            f"Adapter rank {rank} exceeds maximum rank {max_rank} for "
            f"{source_name!r} with shape ({fan_out}, {fan_in})."
        )


def _svd_bases(weight_2d, rank: int):
    u, v = _svd_bases_numpy(np.asarray(weight_2d, dtype=np.float32), rank)
    return mx.array(u, dtype=mx.float32), mx.array(v, dtype=mx.float32)


def _svd_bases_numpy(
    weight_2d: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    u, _, vh = np.linalg.svd(
        weight_2d.astype(np.float32, copy=False),
        full_matrices=False,
    )
    u, vh = _canonicalize_svd_signs(u, vh)
    return (
        u[:, :rank].astype(np.float32, copy=False),
        vh[:rank, :].T.astype(np.float32, copy=False),
    )


def _canonicalize_svd_signs(
    u: np.ndarray,
    vh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_abs_indices = np.abs(u).argmax(axis=0)
    signs = np.sign(u[max_abs_indices, np.arange(u.shape[1])])
    signs[signs == 0] = 1
    return u * signs[None, :], vh * signs[:, None]


def _name_is_selected(
    *names: str,
    include: tp.Sequence[str] | None,
    exclude: tp.Sequence[str] | None,
) -> bool:
    """Match include/exclude filters against any of the given aliases.

    Callers pass both the bare runtime name and the full checkpoint name so
    underfit dashboard filter strings (written in torch/checkpoint naming)
    select the same layers verbatim.
    """

    if include and not any(_matches_any(name, include) for name in names):
        return False
    return not (
        exclude and any(_matches_any(name, exclude) for name in names)
    )


def _matches_any(name: str, patterns: tp.Sequence[str]) -> bool:
    return any(
        expanded in name for pattern in patterns for expanded in _expand(pattern)
    )


def _expand(pattern: str) -> list[str]:
    parts = re.split(r"\[(\d+)-(\d+)\]", pattern)
    if len(parts) == 1:
        return [pattern]

    literals = parts[0::3]
    starts = parts[1::3]
    ends = parts[2::3]
    ranges = []
    for start, end in zip(starts, ends, strict=True):
        start_value = int(start)
        end_value = int(end)
        step = 1 if end_value >= start_value else -1
        ranges.append(
            [str(value) for value in range(start_value, end_value + step, step)]
        )

    expanded = []
    for values in product(*ranges):
        pieces = []
        for index, literal in enumerate(literals):
            pieces.append(literal)
            if index < len(values):
                pieces.append(values[index])
        expanded.append("".join(pieces))
    return expanded
