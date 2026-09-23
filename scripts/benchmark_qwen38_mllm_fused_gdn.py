#!/usr/bin/env python3
"""Offline A/B qualification for benchmark-only Qwen3.8 MLLM fused GDN.

The candidate is default-off and reversible.  It wraps only the 48 exact
``mlx-vlm==0.7.1`` Qwen3.5 GatedDeltaNet instances loaded from the pinned
Qwen3.8 artifact and delegates every ineligible call to the untouched stock
method.  Nothing in this file is imported by production runtime code.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts import benchmark_qwen38_mllm_eager_dispatch as common

SCRIPT = Path(__file__).resolve()
MIB = 1024 * 1024
EXPECTED_STRATA = 6
EXPECTED_TOKENS = 256
EXPECTED_GDN_LAYERS = 48
EXPECTED_MLX_VLM_VERSION = "0.7.1"
EXPECTED_GDN_MODULE = "mlx_vlm.models.qwen3_5.language"
EXPECTED_GDN_CLASS = "Qwen3_5GatedDeltaNet"
EXPECTED_CACHE_MODULE = "mlx_vlm.models.cache"
EXPECTED_CACHE_CLASS = "ArraysCache"
EXPECTED_LANGUAGE_SHA256 = (
    "df8006a2e9067e64e70b91eeaadc2ce83de0fdb54869570cc88e78f46297c5d7"
)
EXPECTED_CACHE_SHA256 = (
    "b736c299bc576f4bdf8d6edf3e1ac6fa3019ca888a2f3cba115d4252f84d5b29"
)
THREADGROUP_CANDIDATES = (32, 16, 8, 4)

NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
CONV_KERNEL = 4
KEY_DIM = NUM_KEY_HEADS * KEY_HEAD_DIM
VALUE_DIM = NUM_VALUE_HEADS * VALUE_HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
HIDDEN_SIZE = 5120

PROMPTS = common.PROMPTS


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


_MISSING = object()


def _required_attr(value: Any, name: str) -> Any:
    result = getattr(value, name, _MISSING)
    if result is _MISSING:
        raise RuntimeError(f"cache metadata attribute is missing: {name}")
    return result


def _metadata_snapshot(cache: Any) -> dict[str, Any]:
    """Capture the metadata stock Qwen3.5 advances around recurrent state."""
    return {
        "left_padding": _required_attr(cache, "left_padding"),
        "left_padding_raw": _required_attr(cache, "_left_padding"),
        "left_padding_advance": _required_attr(cache, "_left_padding_advance"),
        "lengths": _required_attr(cache, "lengths"),
        "lengths_raw": _required_attr(cache, "_lengths"),
        "lengths_advance": _required_attr(cache, "_lengths_advance"),
        "is_speculating": bool(_required_attr(cache, "is_speculating")),
        "speculation_generation": _required_attr(cache, "_speculation_generation"),
        "metadata_revision": _required_attr(cache, "metadata_revision"),
    }


def _metadata_equal(left: dict[str, Any], right: dict[str, Any], mx: Any) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, mx.array) or isinstance(b, mx.array):
            if not isinstance(a, mx.array) or not isinstance(b, mx.array):
                return False
            if not bool(mx.array_equal(a, b).item()):
                return False
        elif a != b:
            return False
    return True


def _gdn_layers(language_model: Any) -> list[Any]:
    return [
        layer.linear_attn
        for layer in common._language_layers(language_model)
        if bool(getattr(layer, "is_linear", False))
    ]


def _stock_method_provenance(gdn_class: type) -> dict[str, Any]:
    method = gdn_class.__call__
    source = Path(inspect.getsourcefile(method) or "").resolve()
    if method.__module__ != EXPECTED_GDN_MODULE:
        raise RuntimeError("stock GDN method module was wrapped or replaced")
    if method.__qualname__ != f"{EXPECTED_GDN_CLASS}.__call__":
        raise RuntimeError("stock GDN method qualname was wrapped or replaced")
    if inspect.unwrap(method) is not method or method.__closure__ is not None:
        raise RuntimeError("stock GDN method is not the pinned plain function")
    if (
        not source.is_file()
        or hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_LANGUAGE_SHA256
    ):
        raise RuntimeError("stock GDN method source does not match pinned mlx-vlm")
    return {
        "module": method.__module__,
        "qualname": method.__qualname__,
        "source_sha256": EXPECTED_LANGUAGE_SHA256,
        "plain_unwrapped": True,
    }


def _cache_class_provenance(arrays_cache_class: type) -> dict[str, Any]:
    source = Path(inspect.getsourcefile(arrays_cache_class) or "").resolve()
    if (
        not source.is_file()
        or hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_CACHE_SHA256
    ):
        raise RuntimeError("ArraysCache source does not match pinned mlx-vlm")
    return {
        "module": arrays_cache_class.__module__,
        "qualname": arrays_cache_class.__qualname__,
        "source_sha256": EXPECTED_CACHE_SHA256,
    }


def _qualify_gdn_layers(
    language_model: Any,
    decoder_class: type,
    gdn_class: type,
    arrays_cache_class: type,
    *,
    enforce_origin: bool = True,
) -> dict[str, Any]:
    """Fail closed on the pinned class ABI and exact loaded Qwen3.8 geometry."""
    decoder_qualification = common._qualify_loaded_model(
        language_model,
        decoder_class,
        enforce_origin=enforce_origin,
    )
    if enforce_origin and (
        gdn_class.__module__ != EXPECTED_GDN_MODULE
        or gdn_class.__name__ != EXPECTED_GDN_CLASS
    ):
        raise RuntimeError("loaded GDN class is not exact mlx-vlm Qwen3.5")
    stock_provenance = _stock_method_provenance(gdn_class) if enforce_origin else {}
    cache_provenance = (
        _cache_class_provenance(arrays_cache_class) if enforce_origin else {}
    )
    if enforce_origin:
        if tuple(inspect.signature(gdn_class.__call__).parameters) != (
            "self",
            "inputs",
            "mask",
            "cache",
        ):
            raise RuntimeError("loaded GDN call ABI does not match mlx-vlm 0.7.1")
    if enforce_origin and (
        arrays_cache_class.__module__ != EXPECTED_CACHE_MODULE
        or arrays_cache_class.__name__ != EXPECTED_CACHE_CLASS
    ):
        raise RuntimeError("loaded cache class is not exact mlx-vlm ArraysCache")
    required_cache_methods = (
        "update_window",
        "update_recurrent",
        "advance",
        "extract",
        "filter",
        "merge",
    )
    if any(
        not callable(getattr(arrays_cache_class, name, None))
        for name in required_cache_methods
    ):
        raise RuntimeError("loaded ArraysCache ABI is incomplete")
    expected_cache_signatures = {
        "update_window": ("self", "index", "source", "width", "lengths"),
        "update_recurrent": ("self", "index", "length", "update"),
        "advance": ("self", "N"),
        "extract": ("self", "idx"),
        "filter": ("self", "batch_indices"),
        "merge": ("caches",),
    }
    for name, expected in expected_cache_signatures.items():
        actual = tuple(inspect.signature(getattr(arrays_cache_class, name)).parameters)
        if actual != expected:
            raise RuntimeError(f"loaded ArraysCache {name} ABI mismatch: {actual}")
    layers = _gdn_layers(language_model)
    if len(layers) != EXPECTED_GDN_LAYERS:
        raise RuntimeError("loaded model does not contain exactly 48 GDN layers")
    if any(type(layer) is not gdn_class for layer in layers):
        raise RuntimeError("loaded GDN instances are not one exact pinned class")
    for layer in layers:
        geometry = (
            int(getattr(layer, "hidden_size", -1)),
            int(getattr(layer, "num_k_heads", -1)),
            int(getattr(layer, "num_v_heads", -1)),
            int(getattr(layer, "head_k_dim", -1)),
            int(getattr(layer, "head_v_dim", -1)),
            int(getattr(layer, "conv_kernel_size", -1)),
        )
        if geometry != (
            HIDDEN_SIZE,
            NUM_KEY_HEADS,
            NUM_VALUE_HEADS,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CONV_KERNEL,
        ):
            raise RuntimeError(f"loaded GDN geometry mismatch: {geometry}")
        if _shape(layer.conv1d.weight) != (CONV_DIM, CONV_KERNEL, 1):
            raise RuntimeError("loaded GDN convolution shape mismatch")
        if _shape(layer.A_log) != (NUM_VALUE_HEADS,):
            raise RuntimeError("loaded GDN A_log shape mismatch")
        if _shape(layer.dt_bias) != (NUM_VALUE_HEADS,):
            raise RuntimeError("loaded GDN dt_bias shape mismatch")
        if _shape(layer.norm.weight) != (VALUE_HEAD_DIM,):
            raise RuntimeError("loaded GDN norm shape mismatch")
    cache = language_model.make_cache()
    linear_indexes = [
        index
        for index, layer in enumerate(common._language_layers(language_model))
        if bool(getattr(layer, "is_linear", False))
    ]
    for index in linear_indexes:
        entry = cache[index]
        if type(entry) is not arrays_cache_class:
            raise RuntimeError("loaded GDN cache is not exact mlx-vlm ArraysCache")
        if not isinstance(getattr(entry, "cache", None), list) or len(entry.cache) != 2:
            raise RuntimeError("loaded GDN cache does not contain exactly two slots")
        _metadata_snapshot(entry)
    fresh = arrays_cache_class(size=2)
    if (
        len(fresh.cache) != 2
        or any(value is not None for value in fresh.cache)
        or fresh.left_padding is not None
        or fresh.lengths is not None
        or bool(fresh.is_speculating)
        or int(fresh.history_capacity) != 0
        or getattr(fresh, "_speculation", _MISSING) is not None
        or int(fresh.metadata_revision) != 0
    ):
        raise RuntimeError("fresh ArraysCache does not have pinned plain semantics")
    return {
        **decoder_qualification,
        "gdn_class": f"{gdn_class.__module__}.{gdn_class.__name__}",
        "gdn_layers": EXPECTED_GDN_LAYERS,
        "num_key_heads": NUM_KEY_HEADS,
        "num_value_heads": NUM_VALUE_HEADS,
        "key_head_dim": KEY_HEAD_DIM,
        "value_head_dim": VALUE_HEAD_DIM,
        "conv_kernel": CONV_KERNEL,
        "stock_method": stock_provenance,
        "cache_class": cache_provenance,
        "gdn_object_ids": [id(layer) for layer in layers],
    }


class FusedGdnPatch:
    """Reversible exact-instance adapter around the pinned mlx-vlm GDN class."""

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
        if len(layers) != EXPECTED_GDN_LAYERS or any(
            type(layer) is not gdn_class for layer in layers
        ):
            raise RuntimeError(
                "candidate allowlist must contain exact 48 GDN instances"
            )
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
        self.hits = 0
        self.layer_hits = [0] * EXPECTED_GDN_LAYERS
        self.installed = False
        self.wrapped: Any | None = None

    def _eligible(self, layer: Any, inputs: Any, mask: Any, cache: Any) -> bool:
        try:
            return bool(
                id(layer) in self.layer_indexes
                and self.qualified
                and self.threadgroup_y in THREADGROUP_CANDIDATES
                and _shape(inputs) == (1, 1, HIDDEN_SIZE)
                and inputs.dtype == self.mx.bfloat16
                and mask is None
                and type(cache) is self.arrays_cache_class
                and isinstance(cache.cache, list)
                and len(cache.cache) == 2
                and not bool(cache.is_speculating)
                and int(cache.history_capacity) == 0
                and cache.lengths is None
                and cache.left_padding is None
                and cache[0] is not None
                and cache[1] is not None
                and _shape(cache[0]) == (1, CONV_KERNEL - 1, CONV_DIM)
                and cache[0].dtype == self.mx.bfloat16
                and _shape(cache[1])
                == (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM)
                and cache[1].dtype == self.mx.float32
                and not bool(getattr(layer, "training", False))
                and getattr(layer, "sharding_group", None) is None
            )
        except Exception:
            return False

    def _projected_eligible(
        self, layer: Any, qkv: Any, z: Any, beta: Any, alpha: Any
    ) -> bool:
        expected = (
            (qkv, (1, 1, CONV_DIM)),
            (z, (1, 1, VALUE_DIM)),
            (beta, (1, 1, NUM_VALUE_HEADS)),
            (alpha, (1, 1, NUM_VALUE_HEADS)),
        )
        try:
            return bool(
                all(_shape(value) == shape for value, shape in expected)
                and all(value.dtype == self.mx.bfloat16 for value, _ in expected)
                and layer.conv1d.weight.dtype == self.mx.bfloat16
                and layer.dt_bias.dtype == self.mx.bfloat16
                and layer.norm.weight.dtype == self.mx.bfloat16
                and layer.A_log.dtype in (self.mx.bfloat16, self.mx.float32)
            )
        except Exception:
            return False

    def _compute_candidate(self, layer: Any, inputs: Any, cache: Any) -> dict[str, Any]:
        """Build every candidate result without mutating the caller's cache."""
        qkv = layer.in_proj_qkv(inputs)
        z = layer.in_proj_z(inputs)
        beta, alpha = layer._project_gates(inputs)
        if not self._projected_eligible(layer, qkv, z, beta, alpha):
            raise ValueError("projected candidate geometry is not exact")
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
            num_key_heads=NUM_KEY_HEADS,
            num_value_heads=NUM_VALUE_HEADS,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            conv_kernel=CONV_KERNEL,
            qwen35_semantics=True,
        )
        # Construct the final projection before touching cache.  The 32-step
        # real-weight probe synchronizes this and both state outputs before it
        # qualifies the patch for measured requests.
        projected = layer.out_proj(output)
        return {
            "projected": projected,
            "output": output,
            "kernel_conv": kernel_conv,
            "next_recurrent": next_recurrent,
            "qkv": qkv,
            "previous_conv": previous_conv,
            "previous_recurrent": previous_recurrent,
        }

    def _commit_candidate(
        self, layer: Any, cache: Any, computed: dict[str, Any]
    ) -> Any:
        """Commit through the exact cache APIs after candidate construction."""
        conv_input = self.mx.concatenate(
            [computed["previous_conv"], computed["qkv"]], axis=1
        )
        cache.update_window(0, conv_input, CONV_KERNEL - 1, lengths=cache.lengths)

        def commit_recurrent(initial: Any, state_steps: Any):
            if initial is not computed["previous_recurrent"] or state_steps is not None:
                raise RuntimeError("plain recurrent cache contract changed")
            return computed["output"], computed["next_recurrent"]

        output, _ = cache.update_recurrent(1, 1, commit_recurrent)
        cache.advance(1)
        self.advance_left_padding(cache, 1)
        self.advance_lengths(cache, 1)
        index = self.layer_indexes[id(layer)]
        self.hits += 1
        self.layer_hits[index] += 1
        return computed["projected"]

    def _candidate_call(
        self, layer: Any, inputs: Any, mask: Any = None, cache: Any = None
    ) -> Any:
        if not self._eligible(layer, inputs, mask, cache):
            return self.original(layer, inputs, mask, cache)

        # Before cache mutation, projection, fused kernel construction, and
        # output projection must all succeed.  Stock fallback is safe only in
        # this pre-commit region.  Post-commit failures propagate rather than
        # double-advancing state through stock.
        try:
            computed = self._compute_candidate(layer, inputs, cache)
        except Exception:
            return self.original(layer, inputs, mask, cache)
        # Do not synchronize each of 48 layers: that would serialize the
        # candidate and invalidate the throughput experiment.  The 32-step
        # probe synchronizes the same real-weight graph before qualification.
        # A later Metal/request failure aborts and discards this request cache;
        # there is intentionally no unsafe mid-request stock replay.
        return self._commit_candidate(layer, cache, computed)

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("fused GDN patch already installed")
        owner = self

        def wrapped(layer: Any, inputs: Any, mask: Any = None, cache: Any = None):
            return owner._candidate_call(layer, inputs, mask, cache)

        self.wrapped = wrapped
        self.gdn_class.__call__ = self.original
        self.installed = True

    def set_candidate(self, enabled: bool) -> None:
        if not self.installed or self.wrapped is None:
            raise RuntimeError("fused GDN patch is not installed")
        self.gdn_class.__call__ = self.wrapped if enabled else self.original

    def close(self) -> None:
        if self.installed:
            self.gdn_class.__call__ = self.original
        self.installed = False

    def reset_hits(self) -> None:
        self.hits = 0
        self.layer_hits = [0] * EXPECTED_GDN_LAYERS


def _arrays_equal(mx: Any, *pairs: tuple[Any, Any]) -> bool:
    return all(bool(mx.array_equal(left, right).item()) for left, right in pairs)


def _make_plain_cache(arrays_cache_class: type, mx: Any) -> Any:
    cache = arrays_cache_class(size=2)
    cache[0] = mx.zeros((1, CONV_KERNEL - 1, CONV_DIM), dtype=mx.bfloat16)
    cache[1] = mx.zeros(
        (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=mx.float32
    )
    return cache


def run_real_weight_parity_probe(
    patch: FusedGdnPatch,
    *,
    steps: int = 32,
) -> dict[str, Any]:
    """Select a threadgroup only after 32-step real-layer bit-exact proof."""
    mx = patch.mx
    layer = patch.layers[0]
    for threadgroup_y in THREADGROUP_CANDIDATES:
        patch.reset_hits()
        try:
            stock_cache = _make_plain_cache(patch.arrays_cache_class, mx)
            candidate_cache = _make_plain_cache(patch.arrays_cache_class, mx)
            patch.threadgroup_y = threadgroup_y
            patch.qualified = False
            passed = True
            for step in range(steps):
                hidden = (
                    mx.random.normal(
                        (1, 1, HIDDEN_SIZE), key=mx.random.key(38100 + step)
                    )
                    * 0.1
                ).astype(mx.bfloat16)
                stock = patch.original(layer, hidden, None, stock_cache)
                computed = patch._compute_candidate(layer, hidden, candidate_cache)
                mx.eval(
                    stock,
                    computed["projected"],
                    computed["kernel_conv"],
                    computed["next_recurrent"],
                    stock_cache[0],
                    stock_cache[1],
                )
                arrays_exact = _arrays_equal(
                    mx,
                    (stock, computed["projected"]),
                    (stock_cache[0], computed["kernel_conv"]),
                    (stock_cache[1], computed["next_recurrent"]),
                )
                metadata_before_commit = (
                    _metadata_equal(
                        _metadata_snapshot(
                            _make_plain_cache(patch.arrays_cache_class, mx)
                        ),
                        _metadata_snapshot(candidate_cache),
                        mx,
                    )
                    if step == 0
                    else True
                )
                # All candidate arrays and the stock comparator are synchronized
                # and exact before the candidate cache is advanced.
                if not arrays_exact or not metadata_before_commit:
                    passed = False
                    break
                hits_before = patch.hits
                layer_hits_before = list(patch.layer_hits)
                candidate = patch._commit_candidate(layer, candidate_cache, computed)
                engaged_once = bool(
                    patch.hits - hits_before == 1
                    and patch.layer_hits[0] - layer_hits_before[0] == 1
                    and patch.layer_hits[1:] == layer_hits_before[1:]
                )
                mx.eval(candidate, candidate_cache[0], candidate_cache[1])
                metadata_exact = _metadata_equal(
                    _metadata_snapshot(stock_cache),
                    _metadata_snapshot(candidate_cache),
                    mx,
                )
                committed_exact = _arrays_equal(
                    mx,
                    (stock, candidate),
                    (stock_cache[0], candidate_cache[0]),
                    (stock_cache[1], candidate_cache[1]),
                )
                if not engaged_once or not committed_exact or not metadata_exact:
                    passed = False
                    break
        except Exception:
            # A compile, synchronization, or cache-contract error rejects this
            # threadgroup.  A poisoned stream will naturally reject later
            # candidates too and produce a fail-closed NO-GO.
            passed = False
        if passed and patch.layer_hits[0] == steps and not any(patch.layer_hits[1:]):
            patch.qualified = True
            patch.reset_hits()
            return {
                "pass": True,
                "steps": steps,
                "threadgroup_y": threadgroup_y,
                "output_exact": True,
                "conv_cache_exact": True,
                "recurrent_cache_exact": True,
                "metadata_exact": True,
                "real_weight_layer_index": 0,
                "candidate_hits": steps,
            }
    patch.threadgroup_y = None
    patch.qualified = False
    patch.reset_hits()
    return {
        "pass": False,
        "steps": steps,
        "threadgroup_y": None,
        "output_exact": False,
        "conv_cache_exact": False,
        "recurrent_cache_exact": False,
        "metadata_exact": False,
        "real_weight_layer_index": 0,
        "candidate_hits": 0,
    }


def _cv(values: list[float]) -> float:
    if not values or statistics.mean(values) == 0:
        return float("inf")
    return (
        statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0
    )


def _fused_engaged(sample: dict[str, Any]) -> bool:
    hits = [int(value) for value in sample.get("fused_layer_hits", [])]
    return bool(
        len(hits) == EXPECTED_GDN_LAYERS
        and min(hits) > 0
        and len(set(hits)) == 1
        and int(sample.get("fused_hits", 0)) == sum(hits)
    )


def _fused_exact_completion(sample: dict[str, Any]) -> bool:
    expected = int(sample.get("completion_tokens", -1)) + 1
    hits = [int(value) for value in sample.get("fused_layer_hits", [])]
    return bool(
        expected > 0
        and len(hits) == EXPECTED_GDN_LAYERS
        and all(value == expected for value in hits)
        and int(sample.get("fused_hits", -1)) == EXPECTED_GDN_LAYERS * expected
    )


def _stock_unmodified(sample: dict[str, Any]) -> bool:
    hits = [int(value) for value in sample.get("fused_layer_hits", [])]
    return bool(
        len(hits) == EXPECTED_GDN_LAYERS
        and not any(hits)
        and int(sample.get("fused_hits", -1)) == 0
    )


def media_sequence_pass(media: dict[str, Any]) -> bool:
    before = media.get("stock_before", {})
    candidate = media.get("candidate_image", {})
    text = media.get("candidate_text", {})
    after = media.get("stock_after", {})
    expected = str(media.get("expected", "")).strip().casefold()
    text_expected = str(media.get("candidate_text_expected", "")).strip().casefold()
    return bool(
        expected
        and text_expected
        and all(
            expected in sample.get("text", "").casefold()
            for sample in (before, candidate, after)
        )
        and before.get("token_sha256")
        == candidate.get("token_sha256")
        == after.get("token_sha256")
        and before.get("text_sha256")
        == candidate.get("text_sha256")
        == after.get("text_sha256")
        and text_expected in text.get("text", "").casefold()
        and all(
            int(sample.get("singleton_batch_delta", 0)) > 0
            for sample in (before, candidate, text, after)
        )
        and _stock_unmodified(before)
        and _fused_engaged(candidate)
        and _fused_exact_completion(candidate)
        and _fused_engaged(text)
        and _fused_exact_completion(text)
        and _stock_unmodified(after)
    )


def evaluate_gates(
    receipt: dict[str, Any], memory_limit: int = 64 * MIB
) -> dict[str, Any]:
    strata = receipt.get("pairs", [])
    complete = len(strata) == EXPECTED_STRATA and all(
        len(s.get("baseline", [])) == 2 and len(s.get("candidate", [])) == 2
        for s in strata
    )
    ratios: list[float] = []
    wall_ratios: list[float] = []
    paired_ratios: list[float] = []
    paired_wall_ratios: list[float] = []
    ttft_ratios: list[float] = []
    active_deltas: list[int] = []
    peak_deltas: list[int] = []
    samples: list[dict[str, Any]] = []
    for stratum in strata:
        baseline = stratum.get("baseline", [])
        candidate = stratum.get("candidate", [])
        samples.extend(baseline + candidate)
        if len(baseline) != 2 or len(candidate) != 2:
            continue
        base_tps = [float(s.get("decode_tps", 0)) for s in baseline]
        cand_tps = [float(s.get("decode_tps", 0)) for s in candidate]
        if min(base_tps) > 0:
            ratios.append(statistics.median(cand_tps) / statistics.median(base_tps))
            paired_ratios.extend(cand_tps[i] / base_tps[i] for i in range(2))
        base_elapsed = [float(s.get("elapsed_s", 0)) for s in baseline]
        cand_elapsed = [float(s.get("elapsed_s", 0)) for s in candidate]
        if min(base_elapsed) > 0 and min(cand_elapsed) > 0:
            wall_ratios.append(
                statistics.median(base_elapsed) / statistics.median(cand_elapsed)
            )
            paired_wall_ratios.extend(
                base_elapsed[index] / cand_elapsed[index] for index in range(2)
            )
        base_ttft = [float(s.get("ttft_s", 0)) for s in baseline]
        cand_ttft = [float(s.get("ttft_s", 0)) for s in candidate]
        if min(base_ttft) > 0:
            ttft_ratios.append(
                statistics.median(cand_ttft) / statistics.median(base_ttft)
            )
        for index in range(2):
            active_deltas.append(
                int(candidate[index]["memory"]["active_bytes"])
                - int(baseline[index]["memory"]["active_bytes"])
            )
            peak_deltas.append(
                int(candidate[index]["memory"]["peak_bytes"])
                - int(baseline[index]["memory"]["peak_bytes"])
            )
    tokens_exact = all(
        len(
            {
                sample.get("token_sha256")
                for arm in ("baseline", "candidate")
                for sample in stratum.get(arm, [])
            }
        )
        == 1
        for stratum in strata
    )
    hits_exact = all(
        (
            _stock_unmodified(sample)
            if arm == "baseline"
            else (
                _fused_engaged(sample)
                and all(
                    int(hit) == EXPECTED_TOKENS + 1
                    for hit in sample["fused_layer_hits"]
                )
                and int(sample["fused_hits"])
                == EXPECTED_GDN_LAYERS * (EXPECTED_TOKENS + 1)
            )
        )
        for stratum in strata
        for arm in ("baseline", "candidate")
        for sample in stratum.get(arm, [])
    )
    singleton_samples = all(
        int(sample.get("singleton_batch_delta", 0)) > 0 for sample in samples
    )
    media_samples = receipt.get("media_recovery", {})
    singleton_media = all(
        int(media_samples.get(key, {}).get("singleton_batch_delta", 0)) > 0
        for key in (
            "stock_before",
            "candidate_image",
            "candidate_text",
            "stock_after",
        )
    )
    checks = {
        "six_complete_prompt_strata": complete,
        "all_twenty_four_samples_are_256_tokens": complete
        and all(
            int(s.get("completion_tokens", -1)) == EXPECTED_TOKENS for s in samples
        ),
        "median_decode_speedup_gte_1_03": len(ratios) == EXPECTED_STRATA
        and statistics.median(ratios) >= 1.03,
        "five_of_six_strata_positive": sum(ratio > 1 for ratio in ratios) >= 5,
        "median_wall_speedup_gte_1_03": len(wall_ratios) == EXPECTED_STRATA
        and statistics.median(wall_ratios) >= 1.03,
        "five_of_six_wall_strata_positive": sum(ratio > 1 for ratio in wall_ratios)
        >= 5,
        "paired_ratio_cv_lte_0_05": len(paired_ratios) == 12
        and _cv(paired_ratios) <= 0.05,
        "paired_wall_ratio_cv_lte_0_05": len(paired_wall_ratios) == 12
        and _cv(paired_wall_ratios) <= 0.05,
        "exact_token_ids_all_runs_per_stratum": complete and tokens_exact,
        "median_ttft_ratio_lte_1_10": len(ttft_ratios) == EXPECTED_STRATA
        and statistics.median(ttft_ratios) <= 1.10,
        "active_delta_lte_64_mib": bool(active_deltas)
        and max(active_deltas) <= memory_limit,
        "isolated_peak_delta_lte_64_mib": bool(peak_deltas)
        and max(peak_deltas) <= memory_limit,
        "exact_candidate_hits_per_layer_and_zero_baseline": complete and hits_exact,
        "singleton_fastpath_engaged": complete
        and singleton_samples
        and singleton_media,
        "real_weight_32_step_bit_exact": bool(
            receipt.get("real_weight_parity", {}).get("pass")
        )
        and int(receipt.get("real_weight_parity", {}).get("steps", 0)) == 32,
        "same_model_and_executor": bool(
            receipt.get("identity", {}).get("same_model")
            and receipt.get("identity", {}).get("same_executor")
        ),
        "prefix_cache_disabled": receipt.get("configuration", {}).get("prefix_cache")
        is False,
        "greedy_thinking_off": receipt.get("configuration", {}).get("temperature")
        == 0.0
        and receipt.get("configuration", {}).get("thinking") is False,
        "measured_ignore_eos_enabled": receipt.get("configuration", {}).get(
            "measured_ignore_eos"
        )
        is True,
        "artifact_exact_b0_verified": bool(receipt.get("artifact", {}).get("verified")),
        "source_clean": receipt.get("source", {}).get("dirty") is False,
        "source_tree_match": receipt.get("source", {}).get("source_tree_match") is True,
        "vlm_candidate_and_stock_recovery": bool(
            receipt.get("media_recovery", {}).get("pass")
        ),
        "no_errors": not receipt.get("errors"),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "metrics": {
            "stratum_speedup_median": statistics.median(ratios) if ratios else None,
            "positive_strata": sum(ratio > 1 for ratio in ratios),
            "wall_speedup_median": statistics.median(wall_ratios)
            if wall_ratios
            else None,
            "positive_wall_strata": sum(ratio > 1 for ratio in wall_ratios),
            "paired_ratio_cv": _cv(paired_ratios),
            "paired_wall_ratio_cv": _cv(paired_wall_ratios),
            "ttft_ratio_median": statistics.median(ttft_ratios)
            if ttft_ratios
            else None,
            "max_active_delta_bytes": max(active_deltas) if active_deltas else None,
            "max_peak_delta_bytes": max(peak_deltas) if peak_deltas else None,
        },
    }


async def _run_sample(
    engine: Any,
    patch: FusedGdnPatch,
    *,
    mode: str,
    prompt: str,
    max_tokens: int,
    messages: list[dict[str, Any]] | None = None,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    patch.set_candidate(mode == "candidate")
    hits_before = patch.hits
    layer_hits_before = list(patch.layer_hits)
    await common._worker_memory(engine, reset=True)
    stats_before = dict(engine.get_stats().get("batch_generator", {}))
    started = time.perf_counter()
    first_at: float | None = None
    token_ids: list[int] = []
    final = None
    async for output in engine.stream_chat(
        messages=messages or [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
        ignore_eos=ignore_eos,
    ):
        delta = common._delta_token_ids(output, len(token_ids))
        if delta and first_at is None:
            first_at = time.perf_counter()
        token_ids.extend(delta)
        final = output
    ended = time.perf_counter()
    if final is None or first_at is None or not token_ids:
        raise RuntimeError("request produced no timed token output")
    memory = await common._worker_memory(engine)
    stats_after = dict(engine.get_stats().get("batch_generator", {}))
    text = (
        getattr(final, "raw_text", None) or getattr(final, "text", None) or ""
    ).strip()
    decode_s = max(ended - first_at, 1e-12)
    return {
        "mode": mode,
        "completion_tokens": len(token_ids),
        "token_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "ttft_s": first_at - started,
        "elapsed_s": ended - started,
        "decode_tps": max(len(token_ids) - 1, 0) / decode_s,
        "singleton_batch_delta": int(stats_after.get("singleton_batches", 0))
        - int(stats_before.get("singleton_batches", 0)),
        "fused_hits": patch.hits - hits_before,
        "fused_layer_hits": [
            after - before for after, before in zip(patch.layer_hits, layer_hits_before)
        ],
        "memory": memory,
    }


def _without_text(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample.items() if key != "text"}


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    artifact = common._inspect_exact_artifact(args.model)
    mlx_vlm_version = importlib.metadata.version("mlx-vlm")
    if mlx_vlm_version != EXPECTED_MLX_VLM_VERSION:
        raise RuntimeError(
            f"benchmark requires mlx-vlm=={EXPECTED_MLX_VLM_VERSION}, "
            f"found {mlx_vlm_version}"
        )

    import mlx.core as mx
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5.language import (
        Qwen3_5DecoderLayer,
        Qwen3_5GatedDeltaNet,
        _qwen3_5_advance_left_padding_info,
        _qwen3_5_advance_lengths_info,
    )

    import rapid_mlx
    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.kernels.qwen4_fused_gdn_decode import fused_gdn_decode
    from rapid_mlx.scheduler import SchedulerConfig

    source_root = SCRIPT.parent.parent.resolve()
    try:
        Path(rapid_mlx.__file__).resolve().relative_to(source_root)
        source_tree_match = True
    except ValueError:
        source_tree_match = False
    if not source_tree_match:
        raise RuntimeError("imported rapid_mlx is not from this benchmark source tree")

    engine = BatchedEngine(
        str(args.model.expanduser().resolve()),
        force_mllm=True,
        no_hybrid=True,
        no_spec_decode=True,
        stream_interval=1,
        scheduler_config=SchedulerConfig(
            enable_prefix_cache=False,
            mllm_singleton_fastpath="auto",
        ),
    )
    errors: list[str] = []
    patch: FusedGdnPatch | None = None
    try:
        await engine.start()
        language_model = getattr(engine._model, "language_model", engine._model)
        loaded = _qualify_gdn_layers(
            language_model,
            Qwen3_5DecoderLayer,
            Qwen3_5GatedDeltaNet,
            ArraysCache,
        )
        patch = FusedGdnPatch(
            Qwen3_5GatedDeltaNet,
            ArraysCache,
            _gdn_layers(language_model),
            mx,
            fused_gdn_decode,
            _qwen3_5_advance_left_padding_info,
            _qwen3_5_advance_lengths_info,
        )
        patch.install()
        parity = await engine.execute_on_model_worker(
            run_real_weight_parity_probe, patch, steps=32
        )
        if not parity["pass"]:
            raise RuntimeError("32-step real-weight fused GDN parity probe failed")

        identity_before = {
            "model": id(language_model),
            "executor": id(engine._model_load_executor),
        }
        for _ in range(2):
            for mode in ("baseline", "candidate", "candidate", "baseline"):
                await _run_sample(
                    engine,
                    patch,
                    mode=mode,
                    prompt=PROMPTS[0][1],
                    max_tokens=32,
                )

        pairs = []
        for index, (category, prompt) in enumerate(PROMPTS):
            order = (
                ("baseline", "candidate", "candidate", "baseline")
                if index % 2 == 0
                else ("candidate", "baseline", "baseline", "candidate")
            )
            samples: dict[str, list[dict[str, Any]]] = {
                "baseline": [],
                "candidate": [],
            }
            for mode in order:
                samples[mode].append(
                    await _run_sample(
                        engine,
                        patch,
                        mode=mode,
                        prompt=prompt,
                        max_tokens=EXPECTED_TOKENS,
                        ignore_eos=True,
                    )
                )
            pairs.append(
                {
                    "index": index + 1,
                    "category": category,
                    "order": list(order),
                    **samples,
                }
            )

        image_url, image_identity = common._data_url(args.image_path)
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.image_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        stock_before = await _run_sample(
            engine,
            patch,
            mode="baseline",
            prompt="",
            max_tokens=64,
            messages=image_messages,
        )
        candidate_image = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt="",
            max_tokens=64,
            messages=image_messages,
        )
        candidate_text = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt=PROMPTS[3][1],
            max_tokens=64,
        )
        stock_after = await _run_sample(
            engine,
            patch,
            mode="baseline",
            prompt="",
            max_tokens=64,
            messages=image_messages,
        )
        media_full = {
            "expected": args.image_expect,
            "candidate_text_expected": "100",
            "stock_before": stock_before,
            "candidate_image": candidate_image,
            "candidate_text": candidate_text,
            "stock_after": stock_after,
        }
        media_recovery = {
            "required": True,
            "image": image_identity,
            "expected": args.image_expect,
            "candidate_text_expected": "100",
            "stock_before": _without_text(stock_before),
            "candidate_image": _without_text(candidate_image),
            "candidate_text": _without_text(candidate_text),
            "stock_after": _without_text(stock_after),
            "pass": media_sequence_pass(media_full),
        }

        identity_after = {
            "model": id(getattr(engine._model, "language_model", engine._model)),
            "executor": id(engine._model_load_executor),
        }
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        source_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=source_root,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        )
        package_versions = {}
        for package in ("mlx", "mlx-lm", "mlx-vlm", "rapid-mlx"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        try:
            physical_memory = int(os.sysconf("SC_PHYS_PAGES")) * int(
                os.sysconf("SC_PAGE_SIZE")
            )
        except (OSError, ValueError):
            physical_memory = None
        receipt = {
            "schema_version": 1,
            "methodology_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "source": {
                "git_commit": source_commit,
                "dirty": source_dirty,
                "source_tree_match": source_tree_match,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "physical_memory_bytes": physical_memory,
                "packages": package_versions,
            },
            "artifact": artifact,
            "loaded_model": {
                key: value
                for key, value in loaded.items()
                if key not in {"layer_object_ids", "gdn_object_ids"}
            },
            "adapter": {
                "default_on": False,
                "production_files_changed": False,
                "threadgroup_y": parity["threadgroup_y"],
                "kernel": "rapid_mlx.kernels.qwen4_fused_gdn_decode.fused_gdn_decode",
                "qwen35_semantics": True,
            },
            "real_weight_parity": parity,
            "configuration": {
                "strata": EXPECTED_STRATA,
                "tokens": EXPECTED_TOKENS,
                "warmup_cycles": 2,
                "warmup_order": ["baseline", "candidate", "candidate", "baseline"],
                "measured_orders": ["ABBA", "BAAB"],
                "runs_per_arm_per_stratum": 2,
                "prefix_cache": False,
                "singleton_fastpath": "auto",
                "temperature": 0.0,
                "top_p": 1.0,
                "thinking": False,
                "measured_ignore_eos": True,
                "stream_interval": 1,
            },
            "identity": {
                "before": identity_before,
                "after": identity_after,
                "same_model": identity_before["model"] == identity_after["model"],
                "same_executor": identity_before["executor"]
                == identity_after["executor"],
            },
            "pairs": pairs,
            "media_recovery": media_recovery,
            "errors": errors,
        }
        receipt["gates"] = evaluate_gates(receipt, args.max_memory_delta_mib * MIB)
        return receipt
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if patch is not None:
            patch.close()
        await engine.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="exact local canonical HF snapshot; never downloaded",
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--image-expect", required=True)
    parser.add_argument(
        "--image-prompt", default="Describe the main visible subject in one sentence."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-memory-delta-mib", type=int, default=64)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.image_expect.strip():
        parser.error("--image-expect must be non-blank")
    if not args.model.is_dir():
        parser.error("--model must be an existing local snapshot directory")
    if not args.image_path.is_file():
        parser.error("--image-path must be an existing local file")
    if args.max_memory_delta_mib < 0:
        parser.error("--max-memory-delta-mib must be non-negative")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        receipt = asyncio.run(run_benchmark(args))
        status = 0 if receipt["gates"]["pass"] else 1
    except Exception as exc:
        receipt = {
            "schema_version": 1,
            "methodology_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "artifact": {"verified": False},
            "errors": [f"{type(exc).__name__}: {exc}"],
            "gates": {"pass": False, "checks": {"setup": False}},
        }
        status = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "pass": receipt["gates"]["pass"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
