# SPDX-License-Identifier: Apache-2.0
"""Opt-in absorbed MLA routing for short multi-token forwards.

mlx-lm 0.31.3 only uses the absorbed Multi-head Latent Attention (MLA)
factorization for a single query.  On a warm cache the same factorization is
cheaper for the small query widths used by speculative verification.  This
module carries a narrowly version-gated compatibility patch while the
equivalent upstream change is pending.

Unknown upstream implementations are left untouched.  The patch also becomes
a no-op once mlx-lm provides its own ``max_absorbed_queries`` helper.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import threading
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

logger = logging.getLogger(__name__)

_ENV = "RAPID_MLX_MLA_ABSORBED_VERIFY"
_STATS_ENV = "RAPID_MLX_MLA_ABSORBED_VERIFY_STATS"
_QUALIFIED_MLX_LM_VERSION = "0.31.3"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
MIN_CACHE_LENGTH = 1024
_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()
_INSTALLED = False
_PROVIDER = "none"
_ENABLED = False
_STATS_ENABLED = os.environ.get(_STATS_ENV, "").strip().lower() in _TRUE_VALUES
_PATCHED_TARGETS: set[tuple[str, str]] = set()

_STATS = {
    "forwards": 0,
    "absorbed": 0,
    "materialized": 0,
    "disabled": 0,
    "single_token": 0,
    "unsupported_cache": 0,
    "short_cache": 0,
}


def _increment_stat(name: str) -> None:
    if not _STATS_ENABLED:
        return
    with _STATS_LOCK:
        _STATS[name] += 1


# Exact mlx-lm 0.31.3 method bodies.  A changed body is not safe to replace
# with a local replica: it may contain a new cache, mask, or numerical rule.
_SUPPORTED_SOURCE_HASHES = {
    ("deepseek_v3", "DeepseekV3Attention"): (
        "410c4c4877a477268a04ac7198bfe5f094007f5665c9569b6dd89e17d8e576c5"
    ),
    ("glm4_moe_lite", "Glm4MoeLiteAttention"): (
        "5012a87eb0a3b3450f638c4f3f0d331ee136e1a3a013160f99763fb7798ff93b"
    ),
    ("kimi_linear", "KimiMLAAttention"): (
        "aef564308e14efa43b29813219d3906917167c415aa63ebb0ac173c3f107f7e8"
    ),
    ("longcat_flash", "LongcatFlashMLA"): (
        "740a730d8f0f68108455df90c6af0fa39fba239bdabd92396d6c20e5efa5110a"
    ),
}


def _feature_enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in _TRUE_VALUES


def _mlx_lm_version() -> str | None:
    try:
        return version("mlx-lm")
    except PackageNotFoundError:  # pragma: no cover - core dependency in production
        return None


def max_absorbed_queries(
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    cache_len: int | None = None,
) -> int:
    """Return the largest integer query width where absorbed MLA is cheaper.

    With ``r`` as the latent rank, ``d`` as the sum of the non-positional QK
    and value dimensions, and ``T`` as the post-update cache length, absorbed
    MLA wins while::

        L < r*d*T / (r*d + T*(2*r - d))

    Integer arithmetic preserves the strict inequality at exact crossover
    points.  Invalid or unsupported geometry fails closed to single-token
    decode, which mlx-lm already routes through the absorbed path.
    """
    dims = (kv_lora_rank, qk_nope_head_dim, v_head_dim)
    if any(not isinstance(value, int) or value <= 0 for value in dims):
        return 1

    d = qk_nope_head_dim + v_head_dim
    if cache_len is None:
        numerator = kv_lora_rank * d
        denominator = 2 * kv_lora_rank - d
    elif isinstance(cache_len, int) and cache_len > 0:
        numerator = kv_lora_rank * d * cache_len
        denominator = kv_lora_rank * d + cache_len * (2 * kv_lora_rank - d)
    else:
        return 1

    if numerator <= 0 or denominator <= 0:
        return 1
    return max(1, (numerator - 1) // denominator)


def latent_length(kv_latent: Any) -> int:
    """Read the sequence length from an MLA latent representation."""
    array = kv_latent[0] if isinstance(kv_latent, tuple) else kv_latent
    shape = getattr(array, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("MLA latent must expose a sequence dimension")
    return int(shape[-2])


def _use_absorbed(self: Any, query_len: int, kv_latent: Any) -> bool:
    cache_len = latent_length(kv_latent)
    if cache_len < MIN_CACHE_LENGTH:
        return False
    return query_len <= max_absorbed_queries(
        int(self.kv_lora_rank),
        int(self.qk_nope_head_dim),
        int(self.v_head_dim),
        cache_len,
    )


def _attention_call(
    self: Any,
    x: Any,
    mask: Any = None,
    cache: Any = None,
    *,
    flavor: str,
) -> Any:
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    if flavor == "kimi":
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
    else:
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        n_heads = self.num_attention_heads if flavor == "longcat" else self.num_heads
        q_head_dim = self.qk_head_dim if flavor == "longcat" else self.q_head_dim
        q = q.reshape(B, L, n_heads, q_head_dim).transpose(0, 2, 1, 3)
        if flavor == "longcat" and self.mla_scale_q_lora is not None:
            q = q * self.mla_scale_q_lora
    q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

    compressed_kv = self.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
    k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
    kv_latent = self.kv_a_layernorm(compressed_kv)
    if flavor == "longcat" and self.mla_scale_kv_lora is not None:
        kv_latent = kv_latent * self.mla_scale_kv_lora

    if flavor != "kimi":
        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
    kv_latent = mx.expand_dims(kv_latent, axis=1)
    if cache is not None:
        kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

    pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        # All four exact supported model wrappers request return_array=True.
        # Keep the same array-only contract as their hashed upstream attention
        # methods, which perform this identical mx.where operation.
        pe_scores = mx.where(
            mask,
            pe_scores,
            mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
        )

    cache_len = latent_length(kv_latent)
    absorbed = _use_absorbed(self, L, kv_latent)
    if not absorbed and cache_len < MIN_CACHE_LENGTH:
        _increment_stat("short_cache")
    _increment_stat("absorbed" if absorbed else "materialized")
    if absorbed:
        q_nope = self.embed_q(q_nope)
        k = v = kv_latent
    else:
        k = self.embed_q(kv_latent, transpose=False)
        v = self.unembed_out(kv_latent)

    output = scaled_dot_product_attention(
        q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
    )
    if absorbed:
        output = self.unembed_out(output)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


_FLAVORS = {
    ("deepseek_v3", "DeepseekV3Attention"): "standard",
    ("glm4_moe_lite", "Glm4MoeLiteAttention"): "standard",
    ("kimi_linear", "KimiMLAAttention"): "kimi",
    ("longcat_flash", "LongcatFlashMLA"): "longcat",
}


def _source_hash(call: Callable[..., Any]) -> str | None:
    try:
        source = inspect.getsource(call)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode()).hexdigest()


def install_mla_absorbed_verify() -> None:
    """Install the compatibility patch once, refusing unknown source shapes."""
    global _ENABLED, _INSTALLED, _PROVIDER

    with _LOCK:
        if _INSTALLED:
            return
        _ENABLED = _feature_enabled()
        try:
            from mlx_lm.models import mla
        except ImportError:  # pragma: no cover - mlx-lm is a core dependency
            logger.debug("[mla_absorbed_verify] mlx-lm unavailable; skipping")
            return

        if hasattr(mla, "max_absorbed_queries"):
            _PROVIDER = "upstream"
            _INSTALLED = True
            return
        if not _ENABLED:
            # Default-off must be a literal zero-overhead path: do not wrap
            # every attention layer merely to rediscover that the flag is off.
            _PROVIDER = "disabled"
            _INSTALLED = True
            return

        installed_version = _mlx_lm_version()
        if installed_version != _QUALIFIED_MLX_LM_VERSION:
            logger.warning(
                "[mla_absorbed_verify] refusing unqualified mlx-lm version %s "
                "(expected %s)",
                installed_version or "unavailable",
                _QUALIFIED_MLX_LM_VERSION,
            )
            _PROVIDER = "unsupported"
            _INSTALLED = True
            return

        originals = getattr(mla, "_RAPID_MLX_MLA_ABSORBED_ORIGINALS", None)
        if originals is None:
            originals = {}
            mla._RAPID_MLX_MLA_ABSORBED_ORIGINALS = originals

        for key, expected_hash in _SUPPORTED_SOURCE_HASHES.items():
            module_name, class_name = key
            try:
                module = __import__(
                    f"mlx_lm.models.{module_name}", fromlist=[class_name]
                )
                cls = getattr(module, class_name)
            except (ImportError, AttributeError):
                continue

            marker = f"_RAPID_MLX_MLA_ABSORBED_{class_name}"
            if getattr(module, marker, False):
                _PATCHED_TARGETS.add(key)
                continue
            original = cls.__call__
            actual_hash = _source_hash(original)
            if actual_hash != expected_hash:
                logger.warning(
                    "[mla_absorbed_verify] refusing unknown %s.%s implementation "
                    "(source hash %s)",
                    module_name,
                    class_name,
                    actual_hash or "unavailable",
                )
                continue

            originals[key] = original
            flavor = _FLAVORS[key]

            def _patched(
                self, x, mask=None, cache=None, *, _orig=original, _flavor=flavor
            ):
                _increment_stat("forwards")
                if not _ENABLED:
                    _increment_stat("disabled")
                    return _orig(self, x, mask, cache)
                if cache is not None and hasattr(cache, "bits"):
                    # mlx-lm 0.31.3 MLA attention cannot consume the quantized
                    # positional-key tuple. Preserve that upstream path rather
                    # than expanding this BF16 compatibility patch's scope.
                    _increment_stat("unsupported_cache")
                    return _orig(self, x, mask, cache)
                if int(x.shape[1]) == 1:
                    _increment_stat("single_token")
                    return _orig(self, x, mask, cache)
                return _attention_call(self, x, mask, cache, flavor=_flavor)

            _patched.__name__ = original.__name__
            _patched.__qualname__ = original.__qualname__
            cls.__call__ = _patched
            setattr(module, marker, True)
            _PATCHED_TARGETS.add(key)

        _PROVIDER = "rapid" if originals else "none"
        _INSTALLED = True


def mla_absorbed_verify_stats() -> dict[str, Any]:
    """Return mechanism counters and the active implementation provider."""
    with _STATS_LOCK:
        counters = dict(_STATS)
    return {
        **counters,
        "installed": _INSTALLED,
        "enabled": _ENABLED,
        "stats_enabled": _STATS_ENABLED,
        "provider": _PROVIDER,
        "targets": tuple(f"{module}.{cls}" for module, cls in sorted(_PATCHED_TARGETS)),
    }


def is_installed() -> bool:
    return _INSTALLED
