# SPDX-License-Identifier: Apache-2.0
"""KV-cache dtype resolution (R15 task #300).

This module centralizes the policy for when the mlx-lm
``QuantizedKVCache`` is used for the live KV cache.

Three user-facing dtypes:

* ``bf16`` — full-precision KV cache (mlx-lm ``KVCache`` /
  ``RotatingKVCache``). Safe everywhere. The default (#1853).
* ``int8`` — 8-bit ``QuantizedKVCache``. 97-98% quality retention on most
  workloads, used as the reasoning/code profile.
* ``int4`` — 4-bit ``QuantizedKVCache``. Smallest KV footprint.

The live serve path uses packed K/V with fused quantized attention. KV
quantization remains an explicit memory/quality trade-off: it reduces resident
cache bytes but is not assumed to improve latency on every model or context.

Unsupported-family safelist:

* Pure sliding-window attention — rotating buffers cannot tolerate arbitrary
  quant-block boundaries. Hybrid layouts are handled per layer: compatible
  full-attention caches are quantized while bounded rotating caches stay BF16.
* Multi-head Latent Attention (DeepSeek V3+, Kimi K2.5) — the K
  projection is already compressed; quantizing on top compounds error.

An operator-EXPLICIT int8/int4 request on a safelisted family raises
:class:`KVCacheQuantizationUnsupportedError` before the server reports
ready (#78) — never a silent bf16 fallback. Only auto/profile-selected
quantization downgrades to ``bf16``. Nested ``text_config`` dicts
(multimodal wrappers) are scanned structurally.

The ``--reasoning`` profile pins to ``int8`` regardless of the dtype
flag (AIME-class hard math collapses ~20pt at sub-4-bit on Qwen3
thinking variants).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Public dtype enum. Order is significant for the Prometheus gauge — the
# string value is stamped onto a label so a Grafana legend filter on
# ``dtype="int4"`` works without parsing.
KV_CACHE_DTYPES = ("bf16", "int8", "int4")

DEFAULT_KV_CACHE_DTYPE = "bf16"
REASONING_KV_CACHE_DTYPE = "int8"

# ---------------------------------------------------------------------------
# Safelist — architectures where int4 is known to break decode quality.
# ---------------------------------------------------------------------------
# Sliding-window attention families. Detection is layered:
#   1. ``alias.architecture`` / ``alias.family`` substring (cheap)
#   2. ``hf_config.sliding_window`` field (canonical)
#   3. ``hf_config.model_type`` matches a known family
#
# Substring patterns are case-insensitive against the *resolved* HF path
# AND the alias key, so both ``gemma-3-27b-4bit`` (alias) and
# ``mlx-community/gemma-3-27b-it-4bit`` (hf_path) trigger.
_SLIDING_WINDOW_PATTERNS: tuple[str, ...] = (
    "gemma-3",
    "gemma3",
    "gpt-oss",
    "gpt_oss",
)

# Multi-head Latent Attention families. The K projection is already
# compressed via ``q_lora_rank`` / ``kv_lora_rank``; stacking int4 on top
# is documented to compound error on reasoning workloads.
_MLA_PATTERNS: tuple[str, ...] = (
    "deepseek-v3",
    "deepseek_v3",
    "deepseek-v4",  # V4 inherits the MLA layout from V3
    "deepseek_v4",
    "kimi-k2",
    "kimi_k2",
)

# HF ``model_type`` values that imply MLA. Conservative — we'd rather
# leave a few quality wins on the table than flip a decode-breaking dtype
# default on a model that nobody benched.
_MLA_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "deepseek_v3",
        "deepseek_v4",
        # inclusionAI Ling 3.0/2.6 (vendored KDA+MLA hybrid): the 6 MLA
        # layers carry the same compressed-K layout as DeepSeek V3, so
        # quantized KV stacks error the same way. model_type is the
        # family signal — the "ling" name alone is too generic for
        # _MLA_PATTERNS.
        "bailing_hybrid",
    }
)

_SLIDING_WINDOW_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "gemma3",
        "gemma3_text",
        "gpt_oss",
    }
)


class KVCacheQuantizationUnsupportedError(ValueError):
    """An explicit int8/int4 KV-cache request cannot be honored (#78).

    The single typed rejection contract for every serving lane (text
    CLI, MLLM engine start, residency route): raised — before the
    server reports ready — instead of silently serving bf16 when the
    operator explicitly asked for a quantized KV cache that the model
    or lane has no supported path for. Only auto/profile-selected
    quantization may downgrade to bf16.

    Attributes:
        requested: The dtype the operator asked for.
        model_name: Model alias / path the request targeted.
        family_reason: Why the request cannot be honored.
    """

    def __init__(self, *, requested: str, model_name: str, family_reason: str):
        self.requested = requested
        self.model_name = model_name
        self.family_reason = family_reason
        super().__init__(
            f"--kv-cache-dtype {requested} cannot be honored for "
            f"'{model_name}': {family_reason}. Remove the flag (bf16 is "
            f"the default and safe everywhere), or pick a full-attention "
            f"model family for quantized KV."
        )


def _candidate_configs(hf_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Config dicts to scan: the top level plus a nested ``text_config``.

    Multimodal checkpoints (Gemma 4) nest the language backbone's fields
    under ``text_config``; the top-level-only read let int8/int4 through
    on every Gemma-4 alias (#78).
    """
    if hf_config is None:
        return []
    configs = [hf_config]
    text_config = hf_config.get("text_config")
    if isinstance(text_config, dict):
        configs.append(text_config)
    return configs


@dataclass(frozen=True)
class KVCacheDtypeDecision:
    """Outcome of :func:`resolve_kv_cache_dtype`.

    Attributes:
        dtype: Resolved dtype string from :data:`KV_CACHE_DTYPES`.
        reason: Human-readable explanation for the startup log line.
        downgraded: True when the requested dtype was overridden by the
            safelist or by ``--reasoning``. Operators consume this to
            decide whether to print a banner.
        requested: The dtype the operator asked for (CLI flag value).
    """

    dtype: str
    reason: str
    downgraded: bool
    requested: str


def _is_sliding_window(
    *,
    model_name: str,
    hf_path: str | None,
    hf_config: dict[str, Any] | None,
    alias_metadata: dict[str, Any] | None,
) -> bool:
    """Return True when the model uses sliding-window attention.

    Detection order: alias metadata → HF config field → name substring.
    The cheapest signal (alias metadata) is preferred so a pinned
    aliases.json entry can override an ambiguous HF config.
    """
    if alias_metadata is not None and alias_metadata.get("sliding_window"):
        return True

    for cfg in _candidate_configs(hf_config):
        # Canonical HF field — populated when the architecture rotates a
        # fixed window per layer (Gemma 3, GPT-OSS, Mistral sliding).
        # Checked on the top level AND the nested ``text_config`` so a
        # multimodal wrapper config (Gemma 4) can't hide the field (#78).
        sw = cfg.get("sliding_window")
        if isinstance(sw, int) and sw > 0:
            return True
        model_type = cfg.get("model_type")
        if isinstance(model_type, str) and model_type in _SLIDING_WINDOW_MODEL_TYPES:
            return True

    needle = f"{model_name or ''} {hf_path or ''}".lower()
    return any(pat in needle for pat in _SLIDING_WINDOW_PATTERNS)


def _supports_hybrid_partial_quantization(
    hf_config: dict[str, Any] | None,
) -> bool:
    """Whether config proves a safe full-attention + rotating split.

    Quantized rotating caches are not supported, but a hybrid model does not
    need them in order to reduce its long-context footprint: exact full-
    attention ``KVCache`` layers can be quantized independently while bounded
    sliding-window layers remain BF16.  Require an explicit per-layer layout;
    a scalar ``sliding_window`` alone cannot prove that any full-attention
    component exists.

    This is only the config-level admission check. The loaded-model probe
    separately verifies cache classes and cross-layer sharing capability
    before the server reports ready.
    """
    if not isinstance(hf_config, dict) or hf_config.get("model_type") not in {
        "gemma4",
        "gemma4_assistant",
        "gemma4_unified",
    }:
        return False

    for cfg in _candidate_configs(hf_config):
        layer_types = cfg.get("layer_types")
        if not isinstance(layer_types, list) or not layer_types:
            continue
        normalized = {value.lower() for value in layer_types if isinstance(value, str)}
        has_full = "full_attention" in normalized
        has_sliding = any("sliding" in value for value in normalized)
        if has_full and has_sliding:
            return True
    return False


def _is_mla(
    *,
    model_name: str,
    hf_path: str | None,
    hf_config: dict[str, Any] | None,
    alias_metadata: dict[str, Any] | None,
) -> bool:
    """Return True when the model uses Multi-head Latent Attention.

    MLA's hallmark is the ``q_lora_rank`` / ``kv_lora_rank`` pair in the
    HF config — the K projection is already compressed, so stacking int4
    on top compounds quantization error. Alias-level metadata wins over
    config inference for the same reason as sliding-window.

    codex r1 BLOCKING #3: the rank-pair alone is too loose. A model
    can ship both fields for reasons unrelated to MLA (some
    quantization toolkits stash LoRA adapter ranks under the same
    field names, and a hand-authored config in a future architecture
    might legitimately use both fields without implementing MLA). We
    therefore require a *family signal* (alias metadata, known
    ``model_type``, or a name-pattern hit) in ADDITION to the rank
    pair before the int4-to-bf16 downgrade fires. The unambiguous
    signals (explicit alias override + canonical ``model_type``)
    still trigger standalone.
    """
    if alias_metadata is not None and alias_metadata.get("is_mla"):
        return True

    needle = f"{model_name or ''} {hf_path or ''}".lower()
    name_hit = any(pat in needle for pat in _MLA_PATTERNS)

    for cfg in _candidate_configs(hf_config):
        model_type = cfg.get("model_type")
        if isinstance(model_type, str) and model_type in _MLA_MODEL_TYPES:
            return True

        # Rank-pair detection only fires when accompanied by a family
        # signal (name match). Avoids the false-positive class where a
        # non-DeepSeek/Kimi model ships both rank fields for unrelated
        # reasons.
        q_rank = cfg.get("q_lora_rank")
        kv_rank = cfg.get("kv_lora_rank")
        rank_pair = (
            isinstance(q_rank, int)
            and q_rank > 0
            and isinstance(kv_rank, int)
            and kv_rank > 0
        )
        if rank_pair and name_hit:
            return True

    return name_hit


def quantized_kv_unsupported_reason(
    *,
    model_name: str | None = None,
    hf_path: str | None = None,
    hf_config: dict[str, Any] | None = None,
    alias_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Return why this model cannot serve a quantized KV cache, or None.

    Single source of truth for the family-capability policy — the
    dtype resolver, the legacy ``--kv-cache-quantization`` CLI branch,
    and the runtime residency gate all consume this one predicate so
    the serving lanes cannot drift (#78).
    """
    if _is_sliding_window(
        model_name=model_name or "",
        hf_path=hf_path,
        hf_config=hf_config,
        alias_metadata=alias_metadata,
    ):
        if _supports_hybrid_partial_quantization(hf_config):
            return None
        return (
            "sliding-window attention (rotating KV buffers are "
            "incompatible with QuantizedKVCache block boundaries, and the "
            "config does not prove an independently quantizable full-attention "
            "component without cross-layer KV sharing)"
        )
    if _is_mla(
        model_name=model_name or "",
        hf_path=hf_path,
        hf_config=hf_config,
        alias_metadata=alias_metadata,
    ):
        return (
            "MLA architecture (K projection already compressed via "
            "q_lora_rank/kv_lora_rank; quantizing on top compounds error)"
        )
    return None


def resolve_kv_cache_dtype(
    requested: str,
    *,
    reasoning: bool = False,
    explicit: bool = False,
    model_name: str | None = None,
    hf_path: str | None = None,
    hf_config: dict[str, Any] | None = None,
    alias_metadata: dict[str, Any] | None = None,
) -> KVCacheDtypeDecision:
    """Resolve the effective KV cache dtype for a model load.

    Args:
        requested: One of :data:`KV_CACHE_DTYPES`. Typically the CLI
            ``--kv-cache-dtype`` value.
        reasoning: When True, pin to :data:`REASONING_KV_CACHE_DTYPE`
            regardless of ``requested`` — for AIME / hard math / code
            workloads where sub-4-bit drops -20pt on thinking variants.
        explicit: True when the operator explicitly asked for
            ``requested`` (CLI flag / control-plane request). An
            explicit int8/int4 on an unsupported family then raises
            :class:`KVCacheQuantizationUnsupportedError` pre-ready
            instead of silently falling back to bf16 (#78); auto/
            default requests keep the downgrade behavior.
        model_name: Alias key or display name. Used for substring
            detection when ``hf_config`` is unavailable.
        hf_path: Resolved HuggingFace repo path. Same as ``model_name``
            for substring detection.
        hf_config: Parsed HuggingFace ``config.json`` (or any dict-like
            with the same fields). When provided, ``sliding_window`` and
            ``q_lora_rank`` / ``kv_lora_rank`` take precedence over the
            substring patterns.
        alias_metadata: Optional dict from the alias profile carrying
            ``sliding_window`` / ``is_mla`` hints. These win over the HF
            config so a curated alias can override an ambiguous upstream
            release.

    Returns:
        A :class:`KVCacheDtypeDecision` carrying the resolved dtype,
        operator-readable reason string, and bookkeeping for the startup
        log line and the Prometheus gauge.
    """
    if requested not in KV_CACHE_DTYPES:
        raise ValueError(
            f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}, got {requested!r}"
        )

    # --reasoning wins over every other consideration — the operator
    # explicitly asked for the AIME-safe profile, so even an explicit
    # ``--kv-cache-dtype int4`` should yield int8. (Setting the dtype
    # flag to bf16 alongside --reasoning would be silently overridden;
    # we log "reasoning profile" so the operator sees what happened.)
    if reasoning:
        if requested == REASONING_KV_CACHE_DTYPE:
            reason = (
                f"reasoning profile (--reasoning) pins to "
                f"{REASONING_KV_CACHE_DTYPE}; already requested"
            )
            return KVCacheDtypeDecision(
                dtype=REASONING_KV_CACHE_DTYPE,
                reason=reason,
                downgraded=False,
                requested=requested,
            )
        reason = (
            f"reasoning profile (--reasoning) pins to "
            f"{REASONING_KV_CACHE_DTYPE} (requested {requested}) — "
            f"sub-4-bit drops -20pt on AIME-class math"
        )
        return KVCacheDtypeDecision(
            dtype=REASONING_KV_CACHE_DTYPE,
            reason=reason,
            downgraded=True,
            requested=requested,
        )

    # Safelist only kicks in for sub-bf16 requests; an operator who
    # explicitly asked for bf16 should never be silently moved.
    if requested != "bf16":
        family_reason = quantized_kv_unsupported_reason(
            model_name=model_name,
            hf_path=hf_path,
            hf_config=hf_config,
            alias_metadata=alias_metadata,
        )
        if family_reason is not None:
            if explicit:
                raise KVCacheQuantizationUnsupportedError(
                    requested=requested,
                    model_name=model_name or hf_path or "unknown",
                    family_reason=family_reason,
                )
            reason = f"{family_reason} detected; falling back from {requested} to bf16"
            return KVCacheDtypeDecision(
                dtype="bf16",
                reason=reason,
                downgraded=True,
                requested=requested,
            )

    # Pass-through for the safe-to-use cases. argparse cannot tell an
    # explicit ``--kv-cache-dtype bf16`` from the parser default, so the
    # reason makes no default-vs-override claim.
    if requested == "bf16":
        reason = "bf16 selected (no QuantizedKVCache wrap)"
    else:
        reason = (
            f"{requested} selected (operator override; packed live KV with "
            f"fused quantized attention); model={model_name or hf_path or 'unknown'}"
        )

    return KVCacheDtypeDecision(
        dtype=requested,
        reason=reason,
        downgraded=False,
        requested=requested,
    )


def dtype_to_quantization_bits(dtype: str) -> tuple[bool, int]:
    """Return ``(kv_cache_quantization, kv_cache_quantization_bits)``.

    Maps the user-facing dtype string onto the existing
    ``SchedulerConfig`` knobs that wire into
    ``mlx_lm.QuantizedKVCache``. ``bf16`` disables quantization; ``int8``
    / ``int4`` enable it with the matching bit width.

    Raises:
        ValueError: If ``dtype`` is not in :data:`KV_CACHE_DTYPES`.
    """
    if dtype == "bf16":
        return False, 8  # bits ignored when quantization=False
    if dtype == "int8":
        return True, 8
    if dtype == "int4":
        return True, 4
    raise ValueError(
        f"unknown kv_cache_dtype {dtype!r}; expected one of {KV_CACHE_DTYPES}"
    )


def log_kv_cache_decision(
    decision: KVCacheDtypeDecision, *, model_name: str | None = None
) -> None:
    """Emit the operator-facing startup log line for a decision, exactly once.

    The log line is the operator's window into what
    ``resolve_kv_cache_dtype`` decided — without it, a downgrade from
    int4 to bf16 looks like a perf regression. It is a single ``logger.info``
    emission: in a foreground ``rapid-mlx serve`` the root handler writes to
    stderr (visible on the terminal), and in log-backed deployments the INFO
    record survives into the log. The decision is surfaced exactly once —
    #2356 flagged that a previous ``logger.info`` + ``print(msg)`` pair sent
    the same line to the terminal twice (stderr and stdout), so a user's
    startup card never settled on a single dtype line.
    """
    msg = f"KV cache dtype: {decision.dtype} — {decision.reason}"
    if model_name and model_name not in decision.reason:
        msg = (
            f"KV cache dtype: {decision.dtype} (model={model_name}) — {decision.reason}"
        )
    logger.info(msg)
