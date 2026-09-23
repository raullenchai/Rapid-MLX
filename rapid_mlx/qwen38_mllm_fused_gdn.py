# SPDX-License-Identifier: Apache-2.0
"""Opt-in fused GDN decode for one exact Qwen3.8 MLLM artifact.

This canary is intentionally narrower than the Qwen3.5-family text runtime:
it accepts one immutable Qwen3.8 target, the pinned mlx-vlm ABI, and the 48
GatedDeltaNet instances belonging to one already-loaded VLM.  It neither
loads weights nor creates an executor.  Unknown provenance, ABI drift, cache
metadata, speculation, batching, or geometry leave stock mlx-vlm untouched.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR = "RAPID_MLX_QWEN38_MLLM_FUSED_GDN"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

_REPO = "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"
_REVISION = "aa985c29ff5b334cbfdcbbc787d47e66e9d9e456"
_VERIFICATION_ID = (
    "hf-snapshot-sha256:"
    "360a8c76fc60c254c595442d16157eb76868033d328d5df9f41344c83bf77a66"
)
_CONFIG_SHA256 = "48e2fc64017e1d7472373ce012ba1db172319e5fc1e4f81ccc97fea523b00d1b"
_INDEX_SHA256 = "47d389bc0826b9f8d0ad495d0c4bf89d7e9398a50b77ae35771dde96e9db9229"
_LAYER_TYPES_SHA256 = "d9c34d22fdab24ccd3286b84516e34a3bfa117d228218df41dfab516e899221f"
_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(64)
)
_SHARDS = (
    (
        "model-00001-of-00003.safetensors",
        "hf_blob:6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
    ),
    (
        "model-00002-of-00003.safetensors",
        "hf_blob:83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
    ),
    (
        "model-00003-of-00003.safetensors",
        "hf_blob:31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
    ),
)

_MLX_VLM_VERSION = "0.7.1"
_MLX_VERSION = "0.32.2"
_MLX_LM_VERSION = "0.31.3"
_LANGUAGE_MODULE = "mlx_vlm.models.qwen3_5.language"
_CACHE_MODULE = "mlx_vlm.models.cache"
_KERNEL_MODULE = "rapid_mlx.kernels.qwen4_fused_gdn_decode"
_LANGUAGE_SHA256 = "df8006a2e9067e64e70b91eeaadc2ce83de0fdb54869570cc88e78f46297c5d7"
_CACHE_SHA256 = "b736c299bc576f4bdf8d6edf3e1ac6fa3019ca888a2f3cba115d4252f84d5b29"
_KERNEL_SHA256 = "73deff77202039597b77cb75cd4e91dd91dfc05575acdca53a1b2937031b2db8"

_LAYERS = 64
_GDN_LAYERS = 48
_HIDDEN_SIZE = 5120
_NUM_KEY_HEADS = 16
_NUM_VALUE_HEADS = 48
_KEY_HEAD_DIM = 128
_VALUE_HEAD_DIM = 128
_CONV_KERNEL = 4
_KEY_DIM = _NUM_KEY_HEADS * _KEY_HEAD_DIM
_VALUE_DIM = _NUM_VALUE_HEADS * _VALUE_HEAD_DIM
_CONV_DIM = 2 * _KEY_DIM + _VALUE_DIM
_THREADGROUPS = (32, 16, 8, 4)
_PROBE_STEPS = 32
_MISSING = object()
_VERIFICATION_AUTHORITY = "rapid_mlx.qwen_artifact:hub-snapshot-v1"

EXPERIMENT_COMMIT = "1c1c6573ab0353aa94bfb2b04e18208d5ddf2466"
EXPERIMENT_METHODOLOGY_SHA256 = (
    "5eec2e6dbcd67e5b9ef2df856880eef468ea5394c7d43a3fbad62150394bb5b3"
)
EXPERIMENT_RECEIPT_SHA256 = (
    "7fb5e03a8b672cb93943ff33238076b3ebad0814edf04e9059dbb1ecc14de5a2"
)


def operator_enabled() -> bool:
    """Return whether the process explicitly enabled the internal canary."""

    return os.environ.get(ENV_VAR, "").strip().lower() in _TRUE_VALUES


def contract_status() -> dict[str, Any]:
    """Immutable qualification evidence; never rewrite the raw gate result."""

    return {
        "experiment_commit": EXPERIMENT_COMMIT,
        "methodology_sha256": EXPERIMENT_METHODOLOGY_SHA256,
        "receipt_sha256": EXPERIMENT_RECEIPT_SHA256,
        "raw_aggregate_pass": False,
        "adjudication": "vlm_differential_pass_semantic_aggregate_invalid",
        "runtime_versions_required": {
            "mlx": _MLX_VERSION,
            "mlx-lm": _MLX_LM_VERSION,
            "mlx-vlm": _MLX_VLM_VERSION,
        },
        "kernel_sha256": _KERNEL_SHA256,
    }


def actual_runtime_versions() -> dict[str, str | None]:
    """Collect versions only after explicit operator opt-in."""

    return {
        package: _installed_version(package)
        for package in ("mlx", "mlx-lm", "mlx-vlm", "rapid-mlx")
    }


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_matches(value: Any, expected_module: str, expected_sha256: str) -> bool:
    try:
        source = Path(inspect.getsourcefile(value) or "").resolve()
        return bool(
            getattr(value, "__module__", None) == expected_module
            and source.is_file()
            and hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha256
        )
    except (OSError, TypeError, ValueError):
        return False


def _kernel_contract_failure(fused_kernel: Any) -> str | None:
    required = {
        "mlx": _MLX_VERSION,
        "mlx-lm": _MLX_LM_VERSION,
        "mlx-vlm": _MLX_VLM_VERSION,
    }
    if any(
        _installed_version(package) != version for package, version in required.items()
    ):
        return "runtime_version_drift"
    if not _source_matches(fused_kernel, _KERNEL_MODULE, _KERNEL_SHA256):
        return "kernel_source_drift"
    if tuple(inspect.signature(fused_kernel).parameters) != (
        "qkv",
        "z",
        "beta",
        "alpha",
        "conv_state",
        "conv_weight",
        "a_log",
        "dt_bias",
        "recurrent_state",
        "norm_weight",
        "norm_eps",
        "threadgroup_y",
        "num_key_heads",
        "num_value_heads",
        "key_head_dim",
        "value_head_dim",
        "conv_kernel",
        "qwen35_semantics",
    ):
        return "kernel_source_drift"
    return None


def _artifact_exact(truth: Any, verified_target: Any) -> bool:
    """Require B0's opaque mint, then cross-check its complete receipt."""

    try:
        from .qwen_runtime_plan import VerifiedQwenTarget
        from .runtime.qwen_artifact import (
            QwenArtifactTruth,
            to_verified_runtime_target,
        )

        if not isinstance(truth, QwenArtifactTruth) or not isinstance(
            verified_target, VerifiedQwenTarget
        ):
            return False
        if to_verified_runtime_target(truth) != verified_target:
            return False
        identity = verified_target.identity
        quantization = json.loads(identity.quantization)
        weight_layout = json.loads(identity.weight_layout)
        return bool(
            truth is not None
            and verified_target.verification_authority == _VERIFICATION_AUTHORITY
            and verified_target.verification_id == _VERIFICATION_ID
            and identity.target_repo == _REPO
            and identity.target_revision == _REVISION
            and identity.target_subfolder is None
            and identity.outer_model_type == "qwen3_5"
            and identity.language_model_type == "qwen3_5_text"
            and identity.layer_layout == _LAYER_TYPES
            and quantization
            == {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                "override_bits": [],
                "override_count": 0,
                "override_group_sizes": [],
            }
            and weight_layout["layout"] == "indexed_safetensors"
            and weight_layout["shard_count"] == 3
            and weight_layout["missing_shard_count"] == 0
            and weight_layout["index_sha256"] == _INDEX_SHA256
            and tuple(weight_layout["shards"]) == tuple(name for name, _ in _SHARDS)
            and weight_layout["file_identities"] == dict(_SHARDS)
            and truth.source_repo == _REPO
            and truth.revision == _REVISION
            and truth.target_subfolder is None
            and truth.identity_status.value == "verified_hub_snapshot"
            and truth.verification_id == _VERIFICATION_ID
            and truth.config_sha256 == _CONFIG_SHA256
            and truth.outer_model_type == "qwen3_5"
            and truth.text_model_type == "qwen3_5_text"
            and truth.geometry.hidden_size == _HIDDEN_SIZE
            and truth.geometry.num_hidden_layers == _LAYERS
            and truth.geometry.full_attention_interval == 4
            and truth.geometry.num_attention_heads == 24
            and truth.geometry.num_key_value_heads == 4
            and truth.geometry.linear_num_key_heads == _NUM_KEY_HEADS
            and truth.geometry.linear_num_value_heads == _NUM_VALUE_HEADS
            and truth.geometry.linear_key_head_dim == _KEY_HEAD_DIM
            and truth.geometry.linear_value_head_dim == _VALUE_HEAD_DIM
            and truth.geometry.layer_types == _LAYER_TYPES
            and truth.geometry.layer_types_sha256 == _LAYER_TYPES_SHA256
            and dict(truth.geometry.layer_type_counts)
            == {"full_attention": 16, "linear_attention": _GDN_LAYERS}
            and truth.quantization.bits == 4
            and truth.quantization.group_size == 64
            and truth.quantization.mode == "affine"
            and truth.target_weights.layout.value == "indexed_safetensors"
            and truth.target_weights.shard_count == 3
            and truth.target_weights.index_sha256 == _INDEX_SHA256
            and tuple(truth.target_weights.file_identities) == _SHARDS
            and truth.target_weights.missing_shard_count == 0
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _language_layers(language_model: Any) -> list[Any]:
    layers = getattr(language_model, "layers", None)
    if layers is None:
        layers = getattr(getattr(language_model, "model", None), "layers", None)
    return list(layers or ())


def _metadata_snapshot(cache: Any) -> dict[str, Any]:
    result = {}
    for name in (
        "left_padding",
        "_left_padding",
        "_left_padding_advance",
        "lengths",
        "_lengths",
        "_lengths_advance",
        "is_speculating",
        "_speculation_generation",
        "metadata_revision",
    ):
        value = getattr(cache, name, _MISSING)
        if value is _MISSING:
            raise RuntimeError(f"ArraysCache metadata missing: {name}")
        result[name] = value
    return result


def _plain_cache(cache: Any, arrays_cache_class: type, *, fresh: bool) -> bool:
    """Require exact non-speculative ArraysCache metadata, never truthiness."""

    try:
        is_speculating = cache.is_speculating
        return bool(
            type(cache) is arrays_cache_class
            and isinstance(cache.cache, list)
            and len(cache.cache) == 2
            and (not fresh or all(value is None for value in cache.cache))
            and type(is_speculating) is bool
            and is_speculating is False
            and int(cache.history_capacity) == 0
            and cache._speculation is None
            and cache.left_padding is None
            and cache.lengths is None
            and cache._left_padding is None
            and cache._lengths is None
            and (not fresh or int(cache.metadata_revision) == 0)
            and (not fresh or int(cache._left_padding_advance) == 0)
            and (not fresh or int(cache._lengths_advance) == 0)
            and (not fresh or int(cache._speculation_generation) == 0)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _metadata_equal(left: dict[str, Any], right: dict[str, Any], mx: Any) -> bool:
    if left.keys() != right.keys():
        return False
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, mx.array) or isinstance(right_value, mx.array):
            if not isinstance(left_value, mx.array) or not isinstance(
                right_value, mx.array
            ):
                return False
            if not bool(mx.array_equal(left_value, right_value).item()):
                return False
        elif left_value != right_value:
            return False
    return True


def _loaded_config_exact(args: Any) -> bool:
    expected = {
        "model_type": "qwen3_5_text",
        "hidden_size": _HIDDEN_SIZE,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_num_key_heads": _NUM_KEY_HEADS,
        "linear_num_value_heads": _NUM_VALUE_HEADS,
        "linear_key_head_dim": _KEY_HEAD_DIM,
        "linear_value_head_dim": _VALUE_HEAD_DIM,
        "linear_conv_kernel_dim": _CONV_KERNEL,
    }
    return all(getattr(args, name, None) == value for name, value in expected.items())


def _runtime_contract(language_model: Any) -> tuple[type, type, list[Any], Any, Any]:
    if importlib.metadata.version("mlx-vlm") != _MLX_VLM_VERSION:
        raise RuntimeError("mlx-vlm version is not the qualified 0.7.1 runtime")

    import mlx.core as mx
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5.language import (
        Qwen3_5DecoderLayer,
        Qwen3_5GatedDeltaNet,
        _qwen3_5_advance_left_padding_info,
        _qwen3_5_advance_lengths_info,
    )

    if not _source_matches(
        Qwen3_5GatedDeltaNet.__call__, _LANGUAGE_MODULE, _LANGUAGE_SHA256
    ) or not _source_matches(ArraysCache, _CACHE_MODULE, _CACHE_SHA256):
        raise RuntimeError("mlx-vlm Qwen3.5 language/cache source drifted")
    if tuple(inspect.signature(Qwen3_5GatedDeltaNet.__call__).parameters) != (
        "self",
        "inputs",
        "mask",
        "cache",
    ):
        raise RuntimeError("Qwen3_5GatedDeltaNet call ABI drifted")
    expected_cache_signatures = {
        "update_window": ("self", "index", "source", "width", "lengths"),
        "update_recurrent": ("self", "index", "length", "update"),
        "advance": ("self", "N"),
        "extract": ("self", "idx"),
        "filter": ("self", "batch_indices"),
        "merge": ("caches",),
    }
    for name, expected in expected_cache_signatures.items():
        method = getattr(ArraysCache, name, None)
        if (
            not callable(method)
            or tuple(inspect.signature(method).parameters) != expected
        ):
            raise RuntimeError(f"ArraysCache {name} ABI drifted")

    decoder_layers = _language_layers(language_model)
    if len(decoder_layers) != _LAYERS or any(
        type(layer) is not Qwen3_5DecoderLayer for layer in decoder_layers
    ):
        raise RuntimeError("loaded decoder is not exact 64-layer mlx-vlm Qwen3.5")
    actual_layout = tuple(
        "linear_attention"
        if bool(getattr(layer, "is_linear", False))
        else "full_attention"
        for layer in decoder_layers
    )
    if actual_layout != _LAYER_TYPES:
        raise RuntimeError("loaded Qwen3.8 decoder layer order drifted")
    args = getattr(language_model, "args", None)
    if args is None:
        args = getattr(getattr(language_model, "model", None), "args", None)
    if not _loaded_config_exact(args):
        raise RuntimeError("loaded Qwen3.8 language config geometry drifted")
    gdn_layers = [
        layer.linear_attn
        for layer in decoder_layers
        if bool(getattr(layer, "is_linear", False))
    ]
    if len(gdn_layers) != _GDN_LAYERS or any(
        type(layer) is not Qwen3_5GatedDeltaNet for layer in gdn_layers
    ):
        raise RuntimeError("loaded decoder is not exact 48-GDN Qwen3.8")
    for layer in gdn_layers:
        geometry = (
            getattr(layer, "hidden_size", None),
            getattr(layer, "num_k_heads", None),
            getattr(layer, "num_v_heads", None),
            getattr(layer, "head_k_dim", None),
            getattr(layer, "head_v_dim", None),
            getattr(layer, "conv_kernel_size", None),
        )
        if geometry != (
            _HIDDEN_SIZE,
            _NUM_KEY_HEADS,
            _NUM_VALUE_HEADS,
            _KEY_HEAD_DIM,
            _VALUE_HEAD_DIM,
            _CONV_KERNEL,
        ):
            raise RuntimeError("loaded Qwen3.8 GDN geometry drifted")
        if (
            _shape(layer.conv1d.weight) != (_CONV_DIM, _CONV_KERNEL, 1)
            or _shape(layer.A_log) != (_NUM_VALUE_HEADS,)
            or _shape(layer.dt_bias) != (_NUM_VALUE_HEADS,)
            or _shape(layer.norm.weight) != (_VALUE_HEAD_DIM,)
        ):
            raise RuntimeError("loaded Qwen3.8 GDN weight geometry drifted")
        if (
            layer.conv1d.weight.dtype != mx.bfloat16
            or layer.dt_bias.dtype != mx.bfloat16
            or layer.norm.weight.dtype != mx.bfloat16
            or layer.A_log.dtype not in {mx.bfloat16, mx.float32}
        ):
            raise RuntimeError("loaded Qwen3.8 GDN weight dtype drifted")

    prompt_cache = language_model.make_cache()
    linear_indexes = [
        index
        for index, layer in enumerate(decoder_layers)
        if bool(getattr(layer, "is_linear", False))
    ]
    for index in linear_indexes:
        entry = prompt_cache[index]
        if not _plain_cache(entry, ArraysCache, fresh=True):
            raise RuntimeError("loaded Qwen3.8 prompt cache is not exact/plain")
        _metadata_snapshot(entry)
    fresh = ArraysCache(size=2)
    if not _plain_cache(fresh, ArraysCache, fresh=True):
        raise RuntimeError("fresh ArraysCache plain semantics drifted")
    return (
        Qwen3_5GatedDeltaNet,
        ArraysCache,
        gdn_layers,
        _qwen3_5_advance_left_padding_info,
        _qwen3_5_advance_lengths_info,
    )


class Qwen38MllmFusedGdnCanary:
    """One-engine reversible exact-instance patch."""

    def __init__(
        self,
        gdn_class: type,
        arrays_cache_class: type,
        layers: list[Any],
        mx: Any,
        fused_kernel: Callable[..., tuple[Any, Any, Any]],
        advance_left_padding: Callable[[Any, int], None],
        advance_lengths: Callable[[Any, int], None],
    ) -> None:
        self.gdn_class = gdn_class
        self.arrays_cache_class = arrays_cache_class
        self.layers = layers
        self.layer_indexes = {id(layer): index for index, layer in enumerate(layers)}
        self.mx = mx
        self.fused_kernel = fused_kernel
        self.advance_left_padding = advance_left_padding
        self.advance_lengths = advance_lengths
        self.original = gdn_class.__call__
        self.threadgroup_y: int | None = None
        self.qualified = False
        self.installed = False
        self.hits = 0
        self.layer_hits = [0] * _GDN_LAYERS
        self.probe_steps_committed = 0
        self.wrapper: Any | None = None

    def _eligible(self, layer: Any, inputs: Any, mask: Any, cache: Any) -> bool:
        try:
            return bool(
                id(layer) in self.layer_indexes
                and self.qualified
                and self.threadgroup_y in _THREADGROUPS
                and _shape(inputs) == (1, 1, _HIDDEN_SIZE)
                and inputs.dtype == self.mx.bfloat16
                and mask is None
                and _plain_cache(cache, self.arrays_cache_class, fresh=False)
                and _shape(cache[0]) == (1, _CONV_KERNEL - 1, _CONV_DIM)
                and cache[0].dtype == self.mx.bfloat16
                and _shape(cache[1])
                == (1, _NUM_VALUE_HEADS, _VALUE_HEAD_DIM, _KEY_HEAD_DIM)
                and cache[1].dtype == self.mx.float32
                and not bool(getattr(layer, "training", False))
                and getattr(layer, "sharding_group", None) is None
            )
        except Exception:
            return False

    def _compute(self, layer: Any, inputs: Any, cache: Any) -> dict[str, Any]:
        qkv = layer.in_proj_qkv(inputs)
        z = layer.in_proj_z(inputs)
        beta, alpha = layer._project_gates(inputs)
        expected = (
            (qkv, (1, 1, _CONV_DIM)),
            (z, (1, 1, _VALUE_DIM)),
            (beta, (1, 1, _NUM_VALUE_HEADS)),
            (alpha, (1, 1, _NUM_VALUE_HEADS)),
        )
        if not all(
            _shape(value) == shape and value.dtype == self.mx.bfloat16
            for value, shape in expected
        ):
            raise ValueError("projected Qwen3.8 GDN geometry drifted")
        previous_conv = cache[0]
        previous_recurrent = cache[1]
        output, kernel_conv, next_recurrent = self.fused_kernel(
            qkv,
            z,
            beta,
            alpha,
            previous_conv,
            layer.conv1d.weight,
            layer.A_log,
            layer.dt_bias,
            previous_recurrent,
            layer.norm.weight,
            layer.norm.eps,
            threadgroup_y=self.threadgroup_y,
            num_key_heads=_NUM_KEY_HEADS,
            num_value_heads=_NUM_VALUE_HEADS,
            key_head_dim=_KEY_HEAD_DIM,
            value_head_dim=_VALUE_HEAD_DIM,
            conv_kernel=_CONV_KERNEL,
            qwen35_semantics=True,
        )
        return {
            "projected": layer.out_proj(output),
            "output": output,
            "kernel_conv": kernel_conv,
            "next_recurrent": next_recurrent,
            "qkv": qkv,
            "previous_conv": previous_conv,
            "previous_recurrent": previous_recurrent,
        }

    def _commit(self, layer: Any, cache: Any, computed: dict[str, Any]) -> Any:
        conv_input = self.mx.concatenate(
            [computed["previous_conv"], computed["qkv"]], axis=1
        )
        cache.update_window(0, conv_input, _CONV_KERNEL - 1, lengths=cache.lengths)

        def update_recurrent(initial: Any, state_steps: Any):
            if initial is not computed["previous_recurrent"] or state_steps is not None:
                raise RuntimeError("plain ArraysCache recurrent contract drifted")
            return computed["output"], computed["next_recurrent"]

        cache.update_recurrent(1, 1, update_recurrent)
        cache.advance(1)
        self.advance_left_padding(cache, 1)
        self.advance_lengths(cache, 1)
        index = self.layer_indexes[id(layer)]
        self.hits += 1
        self.layer_hits[index] += 1
        return computed["projected"]

    def _call(self, layer: Any, inputs: Any, mask: Any = None, cache: Any = None):
        if not self._eligible(layer, inputs, mask, cache):
            return self.original(layer, inputs, mask, cache)
        try:
            computed = self._compute(layer, inputs, cache)
        except Exception:
            # Nothing in the caller-owned cache has changed yet.
            return self.original(layer, inputs, mask, cache)
        # No per-layer synchronization: the real-weight qualification probe
        # already synchronized this exact graph.  Once commit begins, failures
        # propagate and the request cache is discarded; stock is never replayed.
        return self._commit(layer, cache, computed)

    def _new_cache(self) -> Any:
        cache = self.arrays_cache_class(size=2)
        cache[0] = self.mx.zeros(
            (1, _CONV_KERNEL - 1, _CONV_DIM), dtype=self.mx.bfloat16
        )
        cache[1] = self.mx.zeros(
            (1, _NUM_VALUE_HEADS, _VALUE_HEAD_DIM, _KEY_HEAD_DIM),
            dtype=self.mx.float32,
        )
        return cache

    def qualify(self) -> bool:
        """Synchronously prove 32 real-weight stock/candidate steps exact."""

        layer = self.layers[0]
        for threadgroup_y in _THREADGROUPS:
            self.hits = 0
            self.layer_hits = [0] * _GDN_LAYERS
            self.probe_steps_committed = 0
            try:
                self.threadgroup_y = threadgroup_y
                stock_cache = self._new_cache()
                candidate_cache = self._new_cache()
                passed = True
                for step in range(_PROBE_STEPS):
                    hidden = (
                        self.mx.random.normal(
                            (1, 1, _HIDDEN_SIZE), key=self.mx.random.key(38100 + step)
                        )
                        * 0.1
                    ).astype(self.mx.bfloat16)
                    stock = self.original(layer, hidden, None, stock_cache)
                    computed = self._compute(layer, hidden, candidate_cache)
                    self.mx.eval(
                        stock,
                        computed["projected"],
                        computed["kernel_conv"],
                        computed["next_recurrent"],
                        stock_cache[0],
                        stock_cache[1],
                    )
                    if not all(
                        bool(self.mx.array_equal(left, right).item())
                        for left, right in (
                            (stock, computed["projected"]),
                            (stock_cache[0], computed["kernel_conv"]),
                            (stock_cache[1], computed["next_recurrent"]),
                        )
                    ):
                        passed = False
                        break
                    hits_before = self.hits
                    layer_hits_before = tuple(self.layer_hits)
                    candidate = self._commit(layer, candidate_cache, computed)
                    committed_once = bool(
                        self.hits - hits_before == 1
                        and self.layer_hits[0] - layer_hits_before[0] == 1
                        and tuple(self.layer_hits[1:]) == layer_hits_before[1:]
                    )
                    self.mx.eval(candidate, candidate_cache[0], candidate_cache[1])
                    if (
                        not committed_once
                        or not all(
                            bool(self.mx.array_equal(left, right).item())
                            for left, right in (
                                (stock, candidate),
                                (stock_cache[0], candidate_cache[0]),
                                (stock_cache[1], candidate_cache[1]),
                            )
                        )
                        or not _metadata_equal(
                            _metadata_snapshot(stock_cache),
                            _metadata_snapshot(candidate_cache),
                            self.mx,
                        )
                    ):
                        passed = False
                        break
                if (
                    passed
                    and self.hits == _PROBE_STEPS
                    and self.layer_hits[0] == _PROBE_STEPS
                    and not any(self.layer_hits[1:])
                ):
                    self.qualified = True
                    self.probe_steps_committed = _PROBE_STEPS
                    self.hits = 0
                    self.layer_hits = [0] * _GDN_LAYERS
                    return True
            except Exception:
                logger.debug(
                    "Qwen3.8 MLLM fused GDN probe candidate failed: ty=%d",
                    threadgroup_y,
                    exc_info=True,
                )
        self.threadgroup_y = None
        self.qualified = False
        self.hits = 0
        self.layer_hits = [0] * _GDN_LAYERS
        self.probe_steps_committed = 0
        return False

    def install(self) -> None:
        if not self.qualified or self.installed:
            raise RuntimeError("Qwen3.8 fused GDN canary is not installable")
        owner = self

        def wrapper(layer: Any, inputs: Any, mask: Any = None, cache: Any = None):
            return owner._call(layer, inputs, mask, cache)

        self.wrapper = wrapper
        if self.gdn_class.__call__ is not self.original:
            self.wrapper = None
            raise RuntimeError("Qwen3.8 GDN class changed before canary install")
        self.gdn_class.__call__ = wrapper
        self.installed = True

    def close(self) -> None:
        if self.installed and self.gdn_class.__call__ is self.wrapper:
            self.gdn_class.__call__ = self.original
        self.installed = False

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "qualified": self.qualified,
            "installed": self.installed,
            "artifact_verification_id": _VERIFICATION_ID,
            "gdn_layers": _GDN_LAYERS,
            "probe_steps": _PROBE_STEPS,
            "probe_steps_committed": self.probe_steps_committed,
            "threadgroup_y": self.threadgroup_y,
            "hits": self.hits,
        }


def install_qwen38_mllm_fused_gdn_canary(
    language_model: Any, artifact_truth: Any, verified_target: Any
) -> tuple[Qwen38MllmFusedGdnCanary | None, str | None]:
    """Install on the model-owner thread and return ``(patch, fallback)``."""

    if not operator_enabled():
        return None, "operator_disabled"
    if not _artifact_exact(artifact_truth, verified_target):
        return None, "artifact_not_exact"
    try:
        import mlx.core as mx

        from .kernels.qwen4_fused_gdn_decode import (
            fused_gdn_decode,
            fused_gdn_runtime_supported,
        )

        if not fused_gdn_runtime_supported():
            return None, "runtime_unsupported"
        kernel_failure = _kernel_contract_failure(fused_gdn_decode)
        if kernel_failure is not None:
            return None, kernel_failure
        gdn_class, cache_class, layers, advance_padding, advance_lengths = (
            _runtime_contract(language_model)
        )
        canary = Qwen38MllmFusedGdnCanary(
            gdn_class,
            cache_class,
            layers,
            mx,
            fused_gdn_decode,
            advance_padding,
            advance_lengths,
        )
        if not canary.qualify():
            logger.warning("Qwen3.8 MLLM fused GDN parity failed; using stock")
            return None, "real_weight_probe_failed"
        canary.install()
        logger.info(
            "[qwen38_mllm_gdn] canary active: layers=%d probe_steps=%d ty=%d",
            _GDN_LAYERS,
            _PROBE_STEPS,
            canary.threadgroup_y,
        )
        return canary, None
    except Exception:
        logger.warning(
            "Qwen3.8 MLLM fused GDN qualification failed; using stock",
            exc_info=True,
        )
        return None, "qualification_exception"


__all__ = [
    "ENV_VAR",
    "Qwen38MllmFusedGdnCanary",
    "actual_runtime_versions",
    "contract_status",
    "install_qwen38_mllm_fused_gdn_canary",
    "operator_enabled",
]
