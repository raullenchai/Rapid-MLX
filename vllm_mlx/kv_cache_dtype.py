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

The R15 rollout defaulted to int4 on the theory that Apple Silicon
decode is memory-bandwidth-bound, so a 4×-smaller KV cache should
speed up every decode step. The live serve path
(``QuantizedBatchKVCache``, #1197) implements quantization as
dequant-on-read — it materializes full-precision K/V every decode
step — so the bandwidth win never materializes and per-token cost
GROWS with context instead: measured on qwen3.5-4b at 16k context,
bf16 134.6 tok/s vs int4 98.2 (-27%) vs int8 86.1 (-36%); the
short-context numbers that motivated the int4 default (#910,
292-token prompts) could not see this. Until the read path uses a
fused quantized-attention kernel, quantized KV is a memory/quality
trade-off the operator must opt into, not a free speedup (#1853).

Auto-downgrade safelist (forces ``bf16``):

* Sliding-window attention (Gemma 3, GPT-OSS) — the rotating buffer
  can't tolerate arbitrary-boundary quant blocks.
* Multi-head Latent Attention (DeepSeek V3+, Kimi K2.5) — the K
  projection is already compressed; quantizing on top compounds error.

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

    if hf_config is not None:
        # Canonical HF field — populated when the architecture rotates a
        # fixed window per layer (Gemma 3, GPT-OSS, Mistral sliding).
        sw = hf_config.get("sliding_window")
        if isinstance(sw, int) and sw > 0:
            return True
        model_type = hf_config.get("model_type")
        if isinstance(model_type, str) and model_type in _SLIDING_WINDOW_MODEL_TYPES:
            return True

    needle = f"{model_name or ''} {hf_path or ''}".lower()
    return any(pat in needle for pat in _SLIDING_WINDOW_PATTERNS)


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

    if hf_config is not None:
        model_type = hf_config.get("model_type")
        if isinstance(model_type, str) and model_type in _MLA_MODEL_TYPES:
            return True

        # Rank-pair detection only fires when accompanied by a family
        # signal (name match). Avoids the false-positive class where a
        # non-DeepSeek/Kimi model ships both rank fields for unrelated
        # reasons.
        q_rank = hf_config.get("q_lora_rank")
        kv_rank = hf_config.get("kv_lora_rank")
        rank_pair = (
            isinstance(q_rank, int)
            and q_rank > 0
            and isinstance(kv_rank, int)
            and kv_rank > 0
        )
        if rank_pair and name_hit:
            return True

    return name_hit


def resolve_kv_cache_dtype(
    requested: str,
    *,
    reasoning: bool = False,
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
        if _is_sliding_window(
            model_name=model_name or "",
            hf_path=hf_path,
            hf_config=hf_config,
            alias_metadata=alias_metadata,
        ):
            reason = (
                f"sliding-window attention detected (rotating buffer is "
                f"incompatible with QuantizedKVCache block boundaries); "
                f"falling back from {requested} to bf16"
            )
            return KVCacheDtypeDecision(
                dtype="bf16",
                reason=reason,
                downgraded=True,
                requested=requested,
            )
        if _is_mla(
            model_name=model_name or "",
            hf_path=hf_path,
            hf_config=hf_config,
            alias_metadata=alias_metadata,
        ):
            reason = (
                f"MLA architecture detected (K projection already "
                f"compressed via q_lora_rank/kv_lora_rank); falling back "
                f"from {requested} to bf16 to avoid compounding error"
            )
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
            f"{requested} selected (operator override; dequant-on-read "
            f"costs O(context) per decode step — #1853); "
            f"model={model_name or hf_path or 'unknown'} not in safelist"
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
    """Emit the operator-facing startup log line for a decision.

    The log line is the operator's window into what
    ``resolve_kv_cache_dtype`` decided — without it, a downgrade from
    int4 to bf16 looks like a perf regression. Always logged at INFO so
    it survives in default deployments. Also prints to stdout for
    parity with the existing CLI banners (operators run ``rapid-mlx
    serve`` in foreground much more often than they ``journalctl`` it).
    """
    msg = f"KV cache dtype: {decision.dtype} — {decision.reason}"
    if model_name and model_name not in decision.reason:
        msg = (
            f"KV cache dtype: {decision.dtype} (model={model_name}) — {decision.reason}"
        )
    logger.info(msg)
    print(msg)
