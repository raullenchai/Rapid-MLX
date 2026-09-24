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
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import benchmark_qwen38_mllm_eager_dispatch as common

SCRIPT = Path(__file__).resolve()
MIB = 1024 * 1024
GIB = 1024 * MIB
EXPECTED_STRATA = 6
EXPECTED_TOKENS = 256
EXPECTED_GDN_LAYERS = 48
EXPECTED_ABORT_CYCLES = 50
EXPECTED_LONG_TEXT_TOKENS = 512
EXPECTED_IMAGE_SIZE = (1920, 1080)
MIN_SPEEDUP = 1.05
MAX_RATIO_CV = 0.01
MIN_MEMORY_ALLOWANCE = 512 * MIB
MEMORY_ALLOWANCE_FRACTION = 0.03
MAX_EXTRA_SWAP = 256 * MIB
VALID_HARDWARE_TARGETS = {
    ("Apple M4 Pro", 48),
    ("Apple M1 Max", 64),
}
EXPECTED_MLX_VLM_VERSION = "0.7.1"
EXPECTED_MLX_VERSION = "0.32.2"
EXPECTED_MLX_LM_VERSION = "0.31.3"
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
EXPECTED_KERNEL_SHA256 = (
    "73deff77202039597b77cb75cd4e91dd91dfc05575acdca53a1b2937031b2db8"
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


class HardwareQualificationError(RuntimeError):
    def __init__(self, hardware: dict[str, Any]):
        self.hardware = hardware
        super().__init__(
            "qualification hardware mismatch: "
            f"expected {hardware['expected_chip']} with "
            f"{hardware['expected_memory_gib']} GiB; detected "
            f"chip={hardware.get('chip')!r}, "
            f"physical_memory_bytes={hardware.get('physical_memory_bytes')!r}, "
            f"probe_errors={hardware.get('errors', [])!r}"
        )


class QualificationRunError(RuntimeError):
    def __init__(self, receipt: dict[str, Any]):
        self.receipt = receipt
        primary = receipt.get("failure", {}).get("primary", {})
        super().__init__(
            f"qualification failed during {receipt.get('stage')}: "
            f"{primary.get('type', 'unknown error')}"
        )


def _valid_hardware_pair(expected_memory_gib: int, expected_chip: str) -> bool:
    return (expected_chip, expected_memory_gib) in VALID_HARDWARE_TARGETS


_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\"']+")
_CREDENTIALED_HOST_RE = re.compile(
    r"(?i)\b[^\s:@/]+:[^@\s/]+@[a-z0-9.-]+(?::\d+)?(?:/[^\s<>\"']*)?"
)
_SENSITIVE_PARAMETER_RE = re.compile(
    r"(?i)\b(token|access_token|auth_token|api[_-]?key|secret|password|passwd|"
    r"signature|sig)=([^&\s,;]+)"
)
_AUTHORIZATION_RE = re.compile(r"(?i)\b(bearer|basic)\s+[a-z0-9._~+/=-]+")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?i)\b[a-z]:\\(?:[^\\\s<>\"']+\\)*[^\\\s<>\"',;:()\[\]{}]+"
)
_POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w:])/(?:[^/\s<>\"'?,;:()\[\]{}]+/)*"
    r"[^/\s<>\"'?,;:()\[\]{}]+"
)


def _sanitize_text(value: str) -> str:
    """Remove paths and credentials while retaining useful failure context."""
    value = _URL_RE.sub("<redacted-url>", value)
    value = _CREDENTIALED_HOST_RE.sub("<redacted-url>", value)
    value = _SENSITIVE_PARAMETER_RE.sub(r"\1=<redacted>", value)
    value = _AUTHORIZATION_RE.sub(r"\1 <redacted>", value)
    value = _WINDOWS_ABSOLUTE_PATH_RE.sub("<redacted-path>", value)
    return _POSIX_ABSOLUTE_PATH_RE.sub("<redacted-path>", value)


def _sanitized_error(exc: BaseException) -> dict[str, str]:
    message = _sanitize_text(" ".join(str(exc).split()))[:500]
    return {"type": type(exc).__name__, "message": message}


def _sanitize_partial_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_partial_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_partial_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_partial_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _source_identity() -> dict[str, Any]:
    source_root = SCRIPT.parent.parent.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    return {
        "git_commit": commit,
        "dirty": dirty,
        "source_tree_match": None,
        "methodology_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    }


def _unavailable(error: str) -> dict[str, Any]:
    return {"available": False, "error": error}


def _mlx_memory_snapshot(mx: Any, *, clear_cache: bool = False) -> dict[str, Any]:
    """Capture real MLX allocator counters, or an explicit unsupported record."""
    try:
        if clear_cache:
            clear = getattr(mx, "clear_cache", None)
            if not callable(clear):
                return _unavailable("mlx.core.clear_cache is unavailable")
            clear()
        synchronize = getattr(mx, "synchronize", None)
        if callable(synchronize):
            synchronize()
        getters = {
            "active_bytes": getattr(mx, "get_active_memory", None),
            "peak_bytes": getattr(mx, "get_peak_memory", None),
            "cache_bytes": getattr(mx, "get_cache_memory", None),
        }
        missing = [name for name, getter in getters.items() if not callable(getter)]
        if missing:
            return _unavailable(
                f"MLX allocator probes unavailable: {', '.join(missing)}"
            )
        return {
            "available": True,
            **{name: int(getter()) for name, getter in getters.items()},
            "error": None,
        }
    except Exception as exc:
        return _unavailable(f"{type(exc).__name__}: {exc}")


def _parse_scaled_bytes(value: str, unit: str) -> int:
    factors = {"B": 1, "K": 1024, "KB": 1024, "M": MIB, "MB": MIB, "G": GIB, "GB": GIB}
    return int(float(value) * factors[unit.upper()])


def _process_footprint_snapshot(
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Read macOS physical footprint without substituting RSS on other systems."""
    command = shutil.which("footprint")
    if platform.system() != "Darwin" or command is None:
        return _unavailable("macOS footprint(1) is unavailable")
    try:
        result = runner(
            [command, "-p", str(os.getpid())],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
        text = result.stdout
        current = re.search(r"Footprint:\s+([0-9.]+)\s+(B|KB|MB|GB)\b", text)
        peak = re.search(r"phys_footprint_peak:\s+([0-9.]+)\s+(B|KB|MB|GB)\b", text)
        if current is None or peak is None:
            return _unavailable("could not parse footprint(1) output")
        return {
            "available": True,
            "current_bytes": _parse_scaled_bytes(*current.groups()),
            "peak_bytes": _parse_scaled_bytes(*peak.groups()),
            "error": None,
        }
    except Exception as exc:
        return _unavailable(f"{type(exc).__name__}: {exc}")


def _swap_snapshot(
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Read host swap use on macOS, failing visibly when the API is absent."""
    command = "/usr/sbin/sysctl"
    if platform.system() != "Darwin" or not Path(command).is_file():
        return _unavailable("macOS vm.swapusage sysctl is unavailable")
    try:
        result = runner(
            [command, "-n", "vm.swapusage"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
        match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", result.stdout)
        if match is None:
            return _unavailable("could not parse vm.swapusage")
        return {
            "available": True,
            "used_bytes": _parse_scaled_bytes(*match.groups()),
            "error": None,
        }
    except Exception as exc:
        return _unavailable(f"{type(exc).__name__}: {exc}")


def _system_memory_snapshot() -> dict[str, Any]:
    return {
        "physical_footprint": _process_footprint_snapshot(),
        "swap": _swap_snapshot(),
    }


def _hardware_snapshot(expected_memory_gib: int, expected_chip: str) -> dict[str, Any]:
    actual_memory: int | None
    chip: str | None = None
    errors: list[str] = []
    try:
        actual_memory = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
    except (OSError, ValueError) as exc:
        actual_memory = None
        errors.append(f"physical memory unavailable: {type(exc).__name__}: {exc}")
    sysctl = "/usr/sbin/sysctl"
    if platform.system() == "Darwin" and Path(sysctl).is_file():
        try:
            chip = subprocess.run(
                [sysctl, "-n", "machdep.cpu.brand_string"],
                check=True,
                text=True,
                capture_output=True,
                timeout=10,
            ).stdout.strip()
        except Exception as exc:
            errors.append(f"chip identity unavailable: {type(exc).__name__}: {exc}")
    else:
        errors.append("chip identity unavailable: macOS sysctl is unavailable")
    expected_bytes = expected_memory_gib * GIB
    verified = bool(
        _valid_hardware_pair(expected_memory_gib, expected_chip)
        and actual_memory is not None
        and abs(actual_memory - expected_bytes) <= GIB
        and chip is not None
        and chip.casefold() == expected_chip.casefold()
    )
    return {
        "expected_memory_gib": expected_memory_gib,
        "expected_chip": expected_chip,
        "physical_memory_bytes": actual_memory,
        "chip": chip,
        "verified": verified,
        "errors": errors,
    }


async def _memory_checkpoint(
    engine: Any | None,
    mx: Any,
    *,
    clear_cache: bool = False,
) -> dict[str, Any]:
    if engine is None:
        mlx = _mlx_memory_snapshot(mx, clear_cache=clear_cache)
    else:
        try:
            mlx = await engine.execute_on_model_worker(
                _mlx_memory_snapshot, mx, clear_cache=clear_cache
            )
        except Exception as exc:
            mlx = _unavailable(f"{type(exc).__name__}: {exc}")
    return {"mlx": mlx, **_system_memory_snapshot()}


def _track_mlx_peak(tracker: dict[str, int], memory: dict[str, Any]) -> None:
    if memory.get("available", True) is not True:
        return
    for field in ("active_bytes", "peak_bytes", "cache_bytes"):
        value = memory.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            tracker[field] = max(tracker.get(field, 0), value)


def _image_metadata(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return {"width": int(width), "height": int(height)}


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
        and int(text.get("completion_tokens", -1)) == EXPECTED_LONG_TEXT_TOKENS
        and (
            int(media.get("image", {}).get("width", -1)),
            int(media.get("image", {}).get("height", -1)),
        )
        == EXPECTED_IMAGE_SIZE
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


def _request_counts_zero(status: dict[str, Any]) -> bool:
    return bool(
        int(status.get("running_requests", -1)) == 0
        and int(status.get("queued_requests", -1)) == 0
        and int(status.get("admitted_requests", -1)) == 0
    )


def abort_recovery_pass(result: dict[str, Any]) -> bool:
    cycles = result.get("cycles", [])
    return bool(
        len(cycles) == EXPECTED_ABORT_CYCLES
        and len({int(cycle.get("cutoff_tokens", -1)) for cycle in cycles}) > 1
        and all(
            0
            < int(cycle.get("observed_tokens", 0))
            <= int(cycle.get("cutoff_tokens", -1))
            and bool(cycle.get("closed"))
            and _request_counts_zero(cycle.get("idle_status", {}))
            and _fused_engaged(cycle.get("recovery", {}))
            and _fused_exact_completion(cycle.get("recovery", {}))
            for cycle in cycles
        )
        and _request_counts_zero(result.get("final_status", {}))
    )


def cache_recovery_pass(result: dict[str, Any]) -> bool:
    checkpoint = result.get("post_clear", {})
    recovery = result.get("recovery", {})
    return bool(
        result.get("allocator_clear_attempted") is True
        and checkpoint.get("mlx", {}).get("available") is True
        and _fused_engaged(recovery)
        and _fused_exact_completion(recovery)
        and _request_counts_zero(result.get("final_status", {}))
    )


def lifecycle_pass(result: dict[str, Any]) -> bool:
    reload_result = result.get("reload", {})
    final_stop = result.get("final_stop", {})
    return bool(
        result.get("pause", {}).get("paused") is True
        and _request_counts_zero(result.get("pause", {}))
        and result.get("resume", {}).get("paused") is False
        and _request_counts_zero(result.get("resume", {}))
        and _stock_unmodified(result.get("resume_stock_recovery", {}))
        and reload_result.get("stopped") is True
        and reload_result.get("started") is True
        and reload_result.get("pause", {}).get("paused") is True
        and _request_counts_zero(reload_result.get("pause", {}))
        and reload_result.get("resume", {}).get("paused") is False
        and _request_counts_zero(reload_result.get("resume", {}))
        and reload_result.get("model_replaced") is True
        and reload_result.get("executor_replaced") is True
        and bool(reload_result.get("parity", {}).get("pass"))
        and _stock_unmodified(reload_result.get("stock_recovery", {}))
        and _fused_engaged(reload_result.get("candidate_recovery", {}))
        and _fused_exact_completion(reload_result.get("candidate_recovery", {}))
        and final_stop.get("pause", {}).get("paused") is True
        and _request_counts_zero(final_stop.get("pause", {}))
        and final_stop.get("completed") is True
        and final_stop.get("loaded") is False
    )


def _probe_available(checkpoint: dict[str, Any], probe: str) -> bool:
    return checkpoint.get(probe, {}).get("available") is True


def _probe_int(checkpoint: dict[str, Any], probe: str, field: str) -> int | None:
    value = checkpoint.get(probe, {}).get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _no_monotonic_growth(values: list[int]) -> bool:
    pairs = list(zip(values, values[1:]))
    return len(values) > 1 and not (
        all(later >= earlier for earlier, later in pairs)
        and any(later > earlier for earlier, later in pairs)
    )


def evaluate_gates(
    receipt: dict[str, Any], legacy_memory_limit: int | None = None
) -> dict[str, Any]:
    if legacy_memory_limit is None:
        legacy_memory_limit = 64 * MIB
    strata = receipt.get("pairs", [])
    complete = len(strata) == EXPECTED_STRATA and all(
        len(s.get("baseline", [])) == 2 and len(s.get("candidate", [])) == 2
        for s in strata
    )
    layout_exact = len(strata) == EXPECTED_STRATA and all(
        int(stratum.get("index", -1)) == index + 1
        and stratum.get("category") == PROMPTS[index][0]
        and stratum.get("order")
        == list(
            ("baseline", "candidate", "candidate", "baseline")
            if index % 2 == 0
            else ("candidate", "baseline", "baseline", "candidate")
        )
        for index, stratum in enumerate(strata)
    )
    ratios: list[float] = []
    wall_ratios: list[float] = []
    paired_ratios: list[float] = []
    paired_wall_ratios: list[float] = []
    ttft_ratios: list[float] = []
    active_deltas: list[int] = []
    peak_deltas: list[int] = []
    memory_limits: list[int] = []
    footprint_deltas: list[int] = []
    footprint_peak_deltas: list[int] = []
    footprint_limits: list[int] = []
    extra_swap: list[int] = []
    candidate_swap: list[int] = []
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
            baseline_active = int(baseline[index]["memory"]["active_bytes"])
            baseline_peak = int(baseline[index]["memory"]["peak_bytes"])
            candidate_active = int(candidate[index]["memory"]["active_bytes"])
            candidate_peak = int(candidate[index]["memory"]["peak_bytes"])
            active_deltas.append(candidate_active - baseline_active)
            peak_deltas.append(candidate_peak - baseline_peak)
            memory_limits.append(
                max(
                    MIN_MEMORY_ALLOWANCE,
                    int(
                        max(baseline_active, baseline_peak) * MEMORY_ALLOWANCE_FRACTION
                    ),
                )
            )
            baseline_swap = baseline[index].get("system_memory", {}).get("swap", {})
            candidate_swap_probe = (
                candidate[index].get("system_memory", {}).get("swap", {})
            )
            if (
                baseline_swap.get("available") is True
                and candidate_swap_probe.get("available") is True
            ):
                baseline_used = int(baseline_swap["used_bytes"])
                candidate_used = int(candidate_swap_probe["used_bytes"])
                extra_swap.append(candidate_used - baseline_used)
                candidate_swap.append(candidate_used)
            baseline_footprint = (
                baseline[index].get("system_memory", {}).get("physical_footprint", {})
            )
            candidate_footprint = (
                candidate[index].get("system_memory", {}).get("physical_footprint", {})
            )
            if (
                baseline_footprint.get("available") is True
                and candidate_footprint.get("available") is True
            ):
                baseline_current = int(baseline_footprint["current_bytes"])
                baseline_process_peak = int(baseline_footprint["peak_bytes"])
                footprint_deltas.append(
                    int(candidate_footprint["current_bytes"]) - baseline_current
                )
                footprint_peak_deltas.append(
                    int(candidate_footprint["peak_bytes"]) - baseline_process_peak
                )
                footprint_limits.append(
                    max(
                        MIN_MEMORY_ALLOWANCE,
                        int(
                            max(baseline_current, baseline_process_peak)
                            * MEMORY_ALLOWANCE_FRACTION
                        ),
                    )
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
        "exact_abba_baab_order_index_category": layout_exact,
        "all_twenty_four_samples_are_256_tokens": complete
        and all(
            int(s.get("completion_tokens", -1)) == EXPECTED_TOKENS for s in samples
        ),
        "median_decode_speedup_gte_1_05": len(ratios) == EXPECTED_STRATA
        and statistics.median(ratios) >= MIN_SPEEDUP,
        "all_six_decode_strata_positive": len(ratios) == EXPECTED_STRATA
        and all(ratio > 1 for ratio in ratios),
        "median_wall_speedup_gte_1_05": len(wall_ratios) == EXPECTED_STRATA
        and statistics.median(wall_ratios) >= MIN_SPEEDUP,
        "all_six_wall_strata_positive": len(wall_ratios) == EXPECTED_STRATA
        and all(ratio > 1 for ratio in wall_ratios),
        "paired_ratio_cv_lte_0_01": len(paired_ratios) == 12
        and _cv(paired_ratios) <= MAX_RATIO_CV,
        "paired_wall_ratio_cv_lte_0_01": len(paired_wall_ratios) == 12
        and _cv(paired_wall_ratios) <= MAX_RATIO_CV,
        "exact_token_ids_all_runs_per_stratum": complete and tokens_exact,
        "median_ttft_ratio_lte_1_10": len(ttft_ratios) == EXPECTED_STRATA
        and statistics.median(ttft_ratios) <= 1.10,
        "active_delta_lte_max_512_mib_or_3_percent": len(active_deltas) == 12
        and all(delta <= limit for delta, limit in zip(active_deltas, memory_limits)),
        "isolated_peak_delta_lte_max_512_mib_or_3_percent": len(peak_deltas) == 12
        and all(delta <= limit for delta, limit in zip(peak_deltas, memory_limits)),
        "physical_footprint_delta_lte_max_512_mib_or_3_percent": len(footprint_deltas)
        == 12
        and all(
            delta <= limit for delta, limit in zip(footprint_deltas, footprint_limits)
        ),
        "physical_footprint_peak_delta_lte_max_512_mib_or_3_percent": len(
            footprint_peak_deltas
        )
        == 12
        and all(
            delta <= limit
            for delta, limit in zip(footprint_peak_deltas, footprint_limits)
        ),
        "candidate_extra_swap_lte_256_mib": len(extra_swap) == 12
        and max(extra_swap) <= MAX_EXTRA_SWAP,
        "candidate_swap_has_no_monotonic_growth": len(candidate_swap) == 12
        and _no_monotonic_growth(candidate_swap),
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
        "fifty_randomized_abort_recovery_cycles": abort_recovery_pass(
            receipt.get("abort_recovery", {})
        ),
        "allocator_cache_clear_and_recovery": cache_recovery_pass(
            receipt.get("cache_recovery", {})
        ),
        "pause_resume_reload_stop_lifecycle": lifecycle_pass(
            receipt.get("lifecycle", {})
        ),
        "target_48_or_64_gib_hardware": bool(
            receipt.get("hardware", {}).get("verified") is True
            and _valid_hardware_pair(
                int(receipt.get("hardware", {}).get("expected_memory_gib", -1)),
                str(receipt.get("hardware", {}).get("expected_chip", "")),
            )
        ),
        "exact_pinned_runtime_contract": bool(
            receipt.get("runtime_contract", {}).get("packages", {}).get("mlx")
            == EXPECTED_MLX_VERSION
            and receipt.get("runtime_contract", {}).get("packages", {}).get("mlx-lm")
            == EXPECTED_MLX_LM_VERSION
            and receipt.get("runtime_contract", {}).get("packages", {}).get("mlx-vlm")
            == EXPECTED_MLX_VLM_VERSION
            and receipt.get("runtime_contract", {})
            .get("kernel", {})
            .get("source_tree_match")
            is True
            and receipt.get("runtime_contract", {})
            .get("kernel", {})
            .get("source_sha256")
            == EXPECTED_KERNEL_SHA256
            and receipt.get("runtime_contract", {})
            .get("stock_method", {})
            .get("source_sha256")
            == EXPECTED_LANGUAGE_SHA256
            and receipt.get("runtime_contract", {})
            .get("cache_class", {})
            .get("source_sha256")
            == EXPECTED_CACHE_SHA256
        ),
        "no_errors": not receipt.get("errors"),
    }
    checkpoints = receipt.get("memory_checkpoints", {})
    checkpoint_names = ("pre", "probe", "peak", "post_clear", "post_stop")
    checks["pre_memory_captured_before_engine_construction"] = (
        checkpoints.get("pre", {}).get("phase") == "before_engine_construction"
    )
    checks["aggregate_mlx_peak_covers_run"] = bool(
        checkpoints.get("peak", {}).get("aggregate_run_max")
        and _probe_int(
            {"mlx": checkpoints["peak"]["aggregate_run_max"]},
            "mlx",
            "peak_bytes",
        )
        is not None
    )
    checks["all_five_mlx_memory_checkpoints_available"] = all(
        _probe_available(checkpoints.get(name, {}), "mlx") for name in checkpoint_names
    )
    checks["all_five_physical_footprint_checkpoints_available"] = all(
        _probe_available(checkpoints.get(name, {}), "physical_footprint")
        for name in checkpoint_names
    )
    checks["all_five_swap_checkpoints_available"] = all(
        _probe_available(checkpoints.get(name, {}), "swap") for name in checkpoint_names
    )
    peak_mlx = checkpoints.get("peak", {}).get("mlx", {})
    post_clear_mlx = checkpoints.get("post_clear", {}).get("mlx", {})
    checks["allocator_cache_nonincreasing_after_clear"] = bool(
        peak_mlx.get("available") is True
        and post_clear_mlx.get("available") is True
        and int(post_clear_mlx.get("cache_bytes", -1))
        <= int(peak_mlx.get("cache_bytes", -1))
    )

    pre_stop_names = ("pre", "probe", "peak", "post_clear")
    mlx_reference = [
        value
        for name in pre_stop_names
        for field in ("active_bytes", "peak_bytes")
        if (value := _probe_int(checkpoints.get(name, {}), "mlx", field)) is not None
    ]
    for aggregate_name in ("measured_sample_max", "aggregate_run_max"):
        measured_max = checkpoints.get("peak", {}).get(aggregate_name, {})
        mlx_reference.extend(
            value
            for field in ("active_bytes", "peak_bytes")
            if isinstance((value := measured_max.get(field)), int)
            and not isinstance(value, bool)
            and value >= 0
        )
    mlx_reference_max = max(mlx_reference) if mlx_reference else None
    mlx_cleanup_limit = (
        max(
            MIN_MEMORY_ALLOWANCE,
            int(mlx_reference_max * MEMORY_ALLOWANCE_FRACTION),
        )
        if mlx_reference_max is not None
        else None
    )
    post_stop = checkpoints.get("post_stop", {})
    post_stop_active = _probe_int(post_stop, "mlx", "active_bytes")
    post_stop_cache = _probe_int(post_stop, "mlx", "cache_bytes")
    post_stop_mlx_peak = _probe_int(post_stop, "mlx", "peak_bytes")
    pre_mlx_active = _probe_int(checkpoints.get("pre", {}), "mlx", "active_bytes")
    pre_mlx_cache = _probe_int(checkpoints.get("pre", {}), "mlx", "cache_bytes")
    post_stop_active_residual = (
        max(0, post_stop_active - pre_mlx_active)
        if post_stop_active is not None and pre_mlx_active is not None
        else None
    )
    post_stop_cache_residual = (
        max(0, post_stop_cache - pre_mlx_cache)
        if post_stop_cache is not None and pre_mlx_cache is not None
        else None
    )
    checks["post_stop_mlx_active_within_cleanup_bound"] = bool(
        mlx_cleanup_limit is not None
        and post_stop_active_residual is not None
        and post_stop_active_residual <= mlx_cleanup_limit
    )
    checks["post_stop_mlx_cache_within_cleanup_bound"] = bool(
        mlx_cleanup_limit is not None
        and post_stop_cache_residual is not None
        and post_stop_cache_residual <= mlx_cleanup_limit
    )
    checks["stop_does_not_raise_mlx_peak_beyond_allowance"] = bool(
        mlx_reference_max is not None
        and mlx_cleanup_limit is not None
        and post_stop_mlx_peak is not None
        and post_stop_mlx_peak <= mlx_reference_max + mlx_cleanup_limit
    )

    footprint_reference = [
        value
        for name in pre_stop_names
        for field in ("current_bytes", "peak_bytes")
        if (value := _probe_int(checkpoints.get(name, {}), "physical_footprint", field))
        is not None
    ]
    pre_stop_system = checkpoints.get("peak", {}).get("pre_stop_system_memory", {})
    footprint_reference.extend(
        value
        for field in ("current_bytes", "peak_bytes")
        if (value := _probe_int(pre_stop_system, "physical_footprint", field))
        is not None
    )
    footprint_reference_max = max(footprint_reference) if footprint_reference else None
    footprint_cleanup_limit = (
        max(
            MIN_MEMORY_ALLOWANCE,
            int(footprint_reference_max * MEMORY_ALLOWANCE_FRACTION),
        )
        if footprint_reference_max is not None
        else None
    )
    post_stop_footprint = _probe_int(post_stop, "physical_footprint", "current_bytes")
    post_stop_footprint_peak = _probe_int(post_stop, "physical_footprint", "peak_bytes")
    pre_footprint = _probe_int(
        checkpoints.get("pre", {}), "physical_footprint", "current_bytes"
    )
    post_stop_footprint_residual = (
        max(0, post_stop_footprint - pre_footprint)
        if post_stop_footprint is not None and pre_footprint is not None
        else None
    )
    checks["post_stop_footprint_within_cleanup_bound"] = bool(
        footprint_cleanup_limit is not None
        and post_stop_footprint_residual is not None
        and post_stop_footprint_residual <= footprint_cleanup_limit
    )
    checks["stop_does_not_raise_footprint_peak_beyond_allowance"] = bool(
        footprint_reference_max is not None
        and footprint_cleanup_limit is not None
        and post_stop_footprint_peak is not None
        and post_stop_footprint_peak
        <= footprint_reference_max + footprint_cleanup_limit
    )

    pre_swap = _probe_int(checkpoints.get("pre", {}), "swap", "used_bytes")
    post_stop_swap = _probe_int(post_stop, "swap", "used_bytes")
    checks["end_to_end_extra_swap_lte_256_mib"] = bool(
        pre_swap is not None
        and post_stop_swap is not None
        and post_stop_swap - pre_swap <= MAX_EXTRA_SWAP
    )

    # Preserve the original schema-v1 semantics independently of stricter gates.
    checks["median_decode_speedup_gte_1_03"] = len(ratios) == EXPECTED_STRATA and (
        statistics.median(ratios) >= 1.03
    )
    checks["five_of_six_strata_positive"] = sum(ratio > 1 for ratio in ratios) >= 5
    checks["median_wall_speedup_gte_1_03"] = len(wall_ratios) == EXPECTED_STRATA and (
        statistics.median(wall_ratios) >= 1.03
    )
    checks["five_of_six_wall_strata_positive"] = (
        sum(ratio > 1 for ratio in wall_ratios) >= 5
    )
    checks["paired_ratio_cv_lte_0_05"] = len(paired_ratios) == 12 and (
        _cv(paired_ratios) <= 0.05
    )
    checks["paired_wall_ratio_cv_lte_0_05"] = len(paired_wall_ratios) == 12 and (
        _cv(paired_wall_ratios) <= 0.05
    )
    checks["active_delta_lte_64_mib"] = bool(active_deltas) and (
        max(active_deltas) <= legacy_memory_limit
    )
    checks["isolated_peak_delta_lte_64_mib"] = bool(peak_deltas) and (
        max(peak_deltas) <= legacy_memory_limit
    )
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
            "max_memory_allowance_bytes": max(memory_limits) if memory_limits else None,
            "max_physical_footprint_delta_bytes": max(footprint_deltas)
            if footprint_deltas
            else None,
            "max_physical_footprint_peak_delta_bytes": max(footprint_peak_deltas)
            if footprint_peak_deltas
            else None,
            "max_physical_footprint_allowance_bytes": max(footprint_limits)
            if footprint_limits
            else None,
            "max_candidate_extra_swap_bytes": max(extra_swap) if extra_swap else None,
            "legacy_memory_limit_bytes": legacy_memory_limit,
            "mlx_cleanup_limit_bytes": mlx_cleanup_limit,
            "post_stop_mlx_active_residual_bytes": post_stop_active_residual,
            "post_stop_mlx_cache_residual_bytes": post_stop_cache_residual,
            "physical_footprint_cleanup_limit_bytes": footprint_cleanup_limit,
            "post_stop_physical_footprint_residual_bytes": (
                post_stop_footprint_residual
            ),
            "end_to_end_extra_swap_bytes": (
                post_stop_swap - pre_swap
                if post_stop_swap is not None and pre_swap is not None
                else None
            ),
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
    capture_system_memory: bool = True,
    peak_tracker: dict[str, int] | None = None,
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
    if peak_tracker is not None:
        _track_mlx_peak(peak_tracker, memory)
    system_memory = (
        _system_memory_snapshot()
        if capture_system_memory
        else {
            "physical_footprint": _unavailable("not requested for this sample"),
            "swap": _unavailable("not requested for this sample"),
        }
    )
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
        "system_memory": system_memory,
    }


def _without_text(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample.items() if key != "text"}


async def _wait_for_idle(engine: Any, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        status = engine.lifecycle_status()
        if _request_counts_zero(status):
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(f"engine did not become idle: {status}")
        await asyncio.sleep(0.01)


async def run_abort_recovery(
    engine: Any,
    patch: FusedGdnPatch,
    *,
    cycles: int = EXPECTED_ABORT_CYCLES,
    seed: int = 38150,
    timeout: float = 120.0,
    peak_tracker: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Close 50 partially consumed streams and prove recovery after each one."""
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    for index in range(cycles):
        cutoff = rng.randint(1, 16)
        patch.set_candidate(index % 2 == 1)
        stream = engine.stream_chat(
            messages=[{"role": "user", "content": PROMPTS[index % len(PROMPTS)][1]}],
            max_tokens=cutoff + 64,
            temperature=0.0,
            top_p=1.0,
            enable_thinking=False,
            ignore_eos=True,
        )
        observed = 0
        closed = False
        try:
            async for output in stream:
                observed += len(common._delta_token_ids(output, observed))
                if observed >= cutoff:
                    break
        finally:
            close = getattr(stream, "aclose", None)
            if not callable(close):
                raise RuntimeError("stream does not expose async close")
            await close()
            closed = True
        idle = await _wait_for_idle(engine, timeout=timeout)
        abort_memory = await common._worker_memory(engine)
        if peak_tracker is not None:
            _track_mlx_peak(peak_tracker, abort_memory)
        recovery = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt=PROMPTS[(index + 1) % len(PROMPTS)][1],
            max_tokens=8,
            ignore_eos=True,
            capture_system_memory=False,
            peak_tracker=peak_tracker,
        )
        results.append(
            {
                "index": index + 1,
                "cutoff_tokens": cutoff,
                "observed_tokens": observed,
                "closed": closed,
                "idle_status": idle,
                "abort_memory": abort_memory,
                "recovery": _without_text(recovery),
            }
        )
    result = {
        "seed": seed,
        "requested_cycles": cycles,
        "cycles": results,
        "final_status": await _wait_for_idle(engine, timeout=timeout),
    }
    result["pass"] = abort_recovery_pass(result)
    return result


async def _run_benchmark_impl(
    args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    state["stage"] = "hardware_preflight"
    hardware = _hardware_snapshot(args.expected_memory_gib, args.expected_chip)
    state["hardware"] = hardware
    if not hardware["verified"]:
        raise HardwareQualificationError(hardware)

    state["stage"] = "source_preflight"
    source_root = SCRIPT.parent.parent.resolve()
    source = _source_identity()
    state["source"] = source
    if source["dirty"]:
        raise RuntimeError("qualification source tree must be clean")

    state["stage"] = "runtime_preflight"
    expected_versions = {
        "mlx": EXPECTED_MLX_VERSION,
        "mlx-lm": EXPECTED_MLX_LM_VERSION,
        "mlx-vlm": EXPECTED_MLX_VLM_VERSION,
    }
    package_versions: dict[str, str | None] = {}
    for package, expected in expected_versions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        package_versions[package] = actual
        if actual != expected:
            raise RuntimeError(
                f"benchmark requires {package}=={expected}, found {actual}"
            )
    try:
        package_versions["rapid-mlx"] = importlib.metadata.version("rapid-mlx")
    except importlib.metadata.PackageNotFoundError:
        package_versions["rapid-mlx"] = None

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

    try:
        Path(rapid_mlx.__file__).resolve().relative_to(source_root)
        source_tree_match = True
    except ValueError:
        source_tree_match = False
    if not source_tree_match:
        raise RuntimeError("imported rapid_mlx is not from this benchmark source tree")
    source["source_tree_match"] = source_tree_match

    stock_contract = _stock_method_provenance(Qwen3_5GatedDeltaNet)
    cache_contract = _cache_class_provenance(ArraysCache)
    kernel_source = Path(inspect.getsourcefile(fused_gdn_decode) or "").resolve()
    try:
        kernel_source.relative_to(source_root)
        kernel_source_match = kernel_source.is_file()
    except ValueError:
        kernel_source_match = False
    kernel_sha256 = (
        hashlib.sha256(kernel_source.read_bytes()).hexdigest()
        if kernel_source_match
        else None
    )
    if (
        fused_gdn_decode.__module__ != "rapid_mlx.kernels.qwen4_fused_gdn_decode"
        or fused_gdn_decode.__name__ != "fused_gdn_decode"
        or not kernel_source_match
        or kernel_sha256 != EXPECTED_KERNEL_SHA256
    ):
        raise RuntimeError("fused GDN kernel does not match the benchmark source tree")
    runtime_contract = {
        "packages": package_versions,
        "stock_method": stock_contract,
        "cache_class": cache_contract,
        "kernel": {
            "module": fused_gdn_decode.__module__,
            "qualname": fused_gdn_decode.__qualname__,
            "source_sha256": kernel_sha256,
            "source_tree_match": True,
        },
    }
    state["runtime_contract"] = runtime_contract

    state["stage"] = "artifact_preflight"
    artifact = common._inspect_exact_artifact(args.model)
    state["artifact"] = artifact

    memory_checkpoints: dict[str, Any] = {}
    state["memory_checkpoints"] = memory_checkpoints
    state["stage"] = "pre_engine_memory"
    memory_checkpoints["pre"] = await _memory_checkpoint(None, mx)
    memory_checkpoints["pre"]["phase"] = "before_engine_construction"
    aggregate_mlx_peak: dict[str, int] = {}
    _track_mlx_peak(aggregate_mlx_peak, memory_checkpoints["pre"]["mlx"])

    state["stage"] = "engine_construction"
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
    stopped = False
    primary_error: Exception | None = None

    async def qualify_and_patch() -> tuple[
        Any, dict[str, Any], FusedGdnPatch, dict[str, Any]
    ]:
        language = getattr(engine._model, "language_model", engine._model)
        qualified = _qualify_gdn_layers(
            language,
            Qwen3_5DecoderLayer,
            Qwen3_5GatedDeltaNet,
            ArraysCache,
        )
        installed = FusedGdnPatch(
            Qwen3_5GatedDeltaNet,
            ArraysCache,
            _gdn_layers(language),
            mx,
            fused_gdn_decode,
            _qwen3_5_advance_left_padding_info,
            _qwen3_5_advance_lengths_info,
        )
        installed.install()
        proof = await engine.execute_on_model_worker(
            run_real_weight_parity_probe, installed, steps=32
        )
        if not proof["pass"]:
            installed.close()
            raise RuntimeError("32-step real-weight fused GDN parity probe failed")
        return language, qualified, installed, proof

    try:
        state["stage"] = "engine_start"
        await engine.start()
        state["stage"] = "real_weight_probe"
        language_model, loaded, patch, parity = await qualify_and_patch()
        memory_checkpoints["probe"] = await _memory_checkpoint(engine, mx)
        _track_mlx_peak(aggregate_mlx_peak, memory_checkpoints["probe"]["mlx"])

        identity_before = {
            "model": id(language_model),
            "executor": id(engine._model_load_executor),
        }
        state["stage"] = "warmup"
        for _ in range(2):
            for mode in ("baseline", "candidate", "candidate", "baseline"):
                await _run_sample(
                    engine,
                    patch,
                    mode=mode,
                    prompt=PROMPTS[0][1],
                    max_tokens=32,
                    peak_tracker=aggregate_mlx_peak,
                )

        state["stage"] = "abba_baab_measurement"
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
                        peak_tracker=aggregate_mlx_peak,
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

        state["stage"] = "mixed_media_recovery"
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
            peak_tracker=aggregate_mlx_peak,
        )
        candidate_image = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt="",
            max_tokens=64,
            messages=image_messages,
            peak_tracker=aggregate_mlx_peak,
        )
        candidate_text = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt=PROMPTS[3][1],
            max_tokens=EXPECTED_LONG_TEXT_TOKENS,
            ignore_eos=True,
            peak_tracker=aggregate_mlx_peak,
        )
        stock_after = await _run_sample(
            engine,
            patch,
            mode="baseline",
            prompt="",
            max_tokens=64,
            messages=image_messages,
            peak_tracker=aggregate_mlx_peak,
        )
        media_full = {
            "image": {**image_identity, **_image_metadata(args.image_path)},
            "expected": args.image_expect,
            "candidate_text_expected": "100",
            "stock_before": stock_before,
            "candidate_image": candidate_image,
            "candidate_text": candidate_text,
            "stock_after": stock_after,
        }
        media_recovery = {
            "required": True,
            "sequence": [
                "candidate_image_1920x1080",
                f"candidate_text_{EXPECTED_LONG_TEXT_TOKENS}_tokens",
                "stock_image_recovery_1920x1080",
            ],
            "image": media_full["image"],
            "expected": args.image_expect,
            "candidate_text_expected": "100",
            "stock_before": _without_text(stock_before),
            "candidate_image": _without_text(candidate_image),
            "candidate_text": _without_text(candidate_text),
            "stock_after": _without_text(stock_after),
            "pass": media_sequence_pass(media_full),
        }

        state["stage"] = "randomized_abort_recovery"
        abort_recovery = await run_abort_recovery(
            engine,
            patch,
            timeout=args.lifecycle_timeout,
            peak_tracker=aggregate_mlx_peak,
        )

        memory_checkpoints["peak"] = await _memory_checkpoint(engine, mx)
        _track_mlx_peak(aggregate_mlx_peak, memory_checkpoints["peak"]["mlx"])
        measured_samples = [
            sample
            for pair in pairs
            for arm in ("baseline", "candidate")
            for sample in pair[arm]
        ]
        memory_checkpoints["peak"]["measured_sample_max"] = {
            "active_bytes": max(
                int(sample["memory"]["active_bytes"]) for sample in measured_samples
            ),
            "peak_bytes": max(
                int(sample["memory"]["peak_bytes"]) for sample in measured_samples
            ),
        }

        state["stage"] = "cache_clear_recovery"
        prefix_cache_cleared = engine.clear_prefix_cache(reset_stats=False)
        memory_checkpoints["post_clear"] = await _memory_checkpoint(
            engine, mx, clear_cache=True
        )
        _track_mlx_peak(aggregate_mlx_peak, memory_checkpoints["post_clear"]["mlx"])
        cache_recovery_sample = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt=PROMPTS[2][1],
            max_tokens=16,
            ignore_eos=True,
            capture_system_memory=False,
            peak_tracker=aggregate_mlx_peak,
        )
        cache_recovery = {
            "allocator_clear_attempted": True,
            "prefix_cache_clear_result": bool(prefix_cache_cleared),
            "post_clear": memory_checkpoints["post_clear"],
            "recovery": _without_text(cache_recovery_sample),
            "final_status": await _wait_for_idle(
                engine, timeout=args.lifecycle_timeout
            ),
        }
        cache_recovery["pass"] = cache_recovery_pass(cache_recovery)

        identity_measured_after = {
            "model": id(getattr(engine._model, "language_model", engine._model)),
            "executor": id(engine._model_load_executor),
        }

        state["stage"] = "pause_resume_reload"
        lifecycle: dict[str, Any] = {}
        lifecycle["pause"] = await engine.pause_generation(
            "wait", timeout=args.lifecycle_timeout
        )
        lifecycle["resume"] = await engine.resume_generation()
        lifecycle["resume_stock_recovery"] = _without_text(
            await _run_sample(
                engine,
                patch,
                mode="baseline",
                prompt=PROMPTS[1][1],
                max_tokens=16,
                ignore_eos=True,
                capture_system_memory=False,
                peak_tracker=aggregate_mlx_peak,
            )
        )

        reload_before = {
            "model": id(getattr(engine._model, "language_model", engine._model)),
            "executor": id(engine._model_load_executor),
        }
        reload_pause = await engine.pause_generation(
            "wait", timeout=args.lifecycle_timeout
        )
        state["stage"] = "reload_stop_start"
        patch.close()
        patch = None
        await engine.stop()
        await engine.start()
        reload_language, reload_loaded, patch, reload_parity = await qualify_and_patch()
        reload_probe_memory = await common._worker_memory(engine)
        _track_mlx_peak(aggregate_mlx_peak, reload_probe_memory)
        reload_resume = await engine.resume_generation()
        reload_after = {
            "model": id(reload_language),
            "executor": id(engine._model_load_executor),
        }
        reload_stock = await _run_sample(
            engine,
            patch,
            mode="baseline",
            prompt=PROMPTS[4][1],
            max_tokens=16,
            ignore_eos=True,
            capture_system_memory=False,
            peak_tracker=aggregate_mlx_peak,
        )
        reload_candidate = await _run_sample(
            engine,
            patch,
            mode="candidate",
            prompt=PROMPTS[4][1],
            max_tokens=16,
            ignore_eos=True,
            capture_system_memory=False,
            peak_tracker=aggregate_mlx_peak,
        )
        lifecycle["reload"] = {
            "pause": reload_pause,
            "stopped": True,
            "started": True,
            "resume": reload_resume,
            "before": reload_before,
            "after": reload_after,
            "model_replaced": reload_before["model"] != reload_after["model"],
            "executor_replaced": reload_before["executor"] != reload_after["executor"],
            "loaded_model": {
                key: value
                for key, value in reload_loaded.items()
                if key not in {"layer_object_ids", "gdn_object_ids"}
            },
            "parity": reload_parity,
            "probe_memory": reload_probe_memory,
            "stock_recovery": _without_text(reload_stock),
            "candidate_recovery": _without_text(reload_candidate),
        }

        identity_after = {
            "model": identity_measured_after["model"],
            "executor": identity_measured_after["executor"],
        }
        memory_checkpoints["peak"]["pre_stop_system_memory"] = _system_memory_snapshot()
        state["stage"] = "final_stop"
        patch.close()
        patch = None
        final_pause = await engine.pause_generation(
            "wait", timeout=args.lifecycle_timeout
        )
        await engine.stop()
        stopped = True
        lifecycle["final_stop"] = {
            "pause": final_pause,
            "completed": True,
            "loaded": bool(getattr(engine, "_loaded", False)),
        }
        import gc

        gc.collect()
        memory_checkpoints["post_stop"] = await _memory_checkpoint(
            None, mx, clear_cache=True
        )
        _track_mlx_peak(aggregate_mlx_peak, memory_checkpoints["post_stop"]["mlx"])
        memory_checkpoints["peak"]["aggregate_run_max"] = dict(aggregate_mlx_peak)
        lifecycle["pass"] = lifecycle_pass(lifecycle)

        state["stage"] = "receipt_evaluation"
        receipt = {
            "schema_version": 1,
            "schema_compatibility": "additive-v1",
            "methodology_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "source": source,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "physical_memory_bytes": hardware["physical_memory_bytes"],
                "packages": package_versions,
            },
            "hardware": hardware,
            "runtime_contract": runtime_contract,
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
                "invocation": list(sys.argv),
                "strata": EXPECTED_STRATA,
                "tokens": EXPECTED_TOKENS,
                "warmup_cycles": 2,
                "warmup_order": ["baseline", "candidate", "candidate", "baseline"],
                "measured_orders": ["ABBA", "BAAB"],
                "runs_per_arm_per_stratum": 2,
                "abort_cycles": EXPECTED_ABORT_CYCLES,
                "abort_seed": 38150,
                "long_text_tokens": EXPECTED_LONG_TEXT_TOKENS,
                "required_image_size": list(EXPECTED_IMAGE_SIZE),
                "max_memory_delta_mib": args.max_memory_delta_mib,
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
            "abort_recovery": abort_recovery,
            "cache_recovery": cache_recovery,
            "lifecycle": lifecycle,
            "memory_checkpoints": memory_checkpoints,
            "thresholds": {
                "all_strata_positive": True,
                "median_decode_speedup": MIN_SPEEDUP,
                "median_wall_speedup": MIN_SPEEDUP,
                "paired_ratio_cv": MAX_RATIO_CV,
                "memory_delta_min_bytes": MIN_MEMORY_ALLOWANCE,
                "memory_delta_fraction": MEMORY_ALLOWANCE_FRACTION,
                "legacy_memory_delta_bytes": args.max_memory_delta_mib * MIB,
                "candidate_extra_swap_bytes": MAX_EXTRA_SWAP,
                "candidate_swap_monotonic_growth_allowed": False,
            },
            "errors": errors,
        }
        receipt["gates"] = evaluate_gates(
            receipt, args.max_memory_delta_mib * MIB
        )
        return receipt
    except Exception as exc:
        primary_error = exc
        errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup_failure: Exception | None = None
        if patch is not None:
            try:
                patch.close()
            except Exception as exc:
                state.setdefault("cleanup_errors", []).append(_sanitized_error(exc))
                cleanup_failure = cleanup_failure or exc
        if not stopped:
            try:
                await engine.stop()
                stopped = True
            except Exception as exc:
                state.setdefault("cleanup_errors", []).append(_sanitized_error(exc))
                cleanup_failure = cleanup_failure or exc
            try:
                import gc

                gc.collect()
                if "post_stop" not in memory_checkpoints:
                    memory_checkpoints["post_stop"] = await _memory_checkpoint(
                        None, mx, clear_cache=True
                    )
                    memory_checkpoints["post_stop"]["phase"] = "failure_cleanup"
            except Exception as exc:
                state.setdefault("cleanup_errors", []).append(_sanitized_error(exc))
                cleanup_failure = cleanup_failure or exc
        if cleanup_failure is not None and primary_error is None:
            raise cleanup_failure


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    state: dict[str, Any] = {"stage": "initialization", "cleanup_errors": []}
    try:
        return await _run_benchmark_impl(args, state)
    except Exception as exc:
        primary = _sanitized_error(exc)
        cleanup = list(state.get("cleanup_errors", []))
        source = state.get("source")
        if source is None:
            try:
                source = _source_identity()
            except Exception as source_exc:
                source = _unavailable(
                    f"source identity unavailable: {_sanitized_error(source_exc)['type']}"
                )
        error_strings = [f"primary: {primary['type']}: {primary['message']}"]
        error_strings.extend(
            f"cleanup: {item['type']}: {item['message']}" for item in cleanup
        )
        receipt = {
            "schema_version": 1,
            "schema_compatibility": "additive-v1",
            "methodology_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "stage": state.get("stage", "unknown"),
            "hardware": state.get(
                "hardware", _unavailable("hardware preflight did not complete")
            ),
            "source": source,
            "runtime_contract": state.get(
                "runtime_contract",
                _unavailable("runtime preflight did not complete"),
            ),
            "artifact": state.get(
                "artifact", {"verified": False, "error": "preflight incomplete"}
            ),
            "memory_checkpoints": state.get("memory_checkpoints", {}),
            "failure": {"primary": primary, "cleanup": cleanup},
            "errors": error_strings,
            "gates": {"pass": False, "checks": {"setup": False}},
        }
        raise QualificationRunError(_sanitize_partial_value(receipt)) from exc


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
    parser.add_argument(
        "--expected-memory-gib", type=int, choices=(48, 64), required=True
    )
    parser.add_argument(
        "--expected-chip",
        choices=("Apple M4 Pro", "Apple M1 Max"),
        required=True,
    )
    parser.add_argument(
        "--max-memory-delta-mib",
        type=int,
        default=64,
        help=(
            "schema-v1 active/peak delta limit (default: 64); the additional "
            "max(512 MiB, 3%%) qualification gates remain fixed"
        ),
    )
    parser.add_argument("--lifecycle-timeout", type=float, default=120.0)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.image_expect.strip():
        parser.error("--image-expect must be non-blank")
    if not _valid_hardware_pair(args.expected_memory_gib, args.expected_chip):
        parser.error(
            "hardware target must be Apple M4 Pro + 48 GiB or Apple M1 Max + 64 GiB"
        )
    if not args.model.is_dir():
        parser.error("--model must be an existing local snapshot directory")
    if not args.image_path.is_file():
        parser.error("--image-path must be an existing local file")
    try:
        image = _image_metadata(args.image_path)
    except Exception as exc:
        parser.error(f"--image-path must be a readable image: {exc}")
    if (image["width"], image["height"]) != EXPECTED_IMAGE_SIZE:
        parser.error("--image-path must be exactly 1920x1080")
    if args.lifecycle_timeout <= 0:
        parser.error("--lifecycle-timeout must be positive")
    if args.max_memory_delta_mib < 0:
        parser.error("--max-memory-delta-mib must be non-negative")
    if args.output.exists():
        parser.error("--output already exists; raw receipts are immutable")


def _write_new_receipt(path: Path, receipt: dict[str, Any]) -> None:
    """Create, never replace, an immutable raw qualification receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        receipt = asyncio.run(run_benchmark(args))
        status = 0 if receipt["gates"]["pass"] else 1
    except QualificationRunError as exc:
        receipt = exc.receipt
        status = 2
    except Exception as exc:
        receipt = {
            "schema_version": 1,
            "schema_compatibility": "additive-v1",
            "methodology_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "artifact": {"verified": False},
            "errors": [f"{type(exc).__name__}: {exc}"],
            "gates": {"pass": False, "checks": {"setup": False}},
        }
        status = 2
    _write_new_receipt(args.output, receipt)
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
