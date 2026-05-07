"""Auto-detect optimal configuration for a model family.

This is the **per-model profile registry**. When users don't specify a
parser, throttle, or optimization flag explicitly, this module infers
the best configuration from the model name/path pattern, with optional
runtime enrichment from the loaded model object.

Two stages:

1. ``detect_model_config(model_path)`` — declarative, name-regex based.
   Runs *before* model load. Returns ``ModelConfig`` with parser
   defaults and capability gates (e.g. whether spec decoding is safe
   for this arch).

2. ``enrich_model_config(cfg, model)`` — runtime probe of the loaded
   model. Used as a safety net for unrecognized hybrid models — if the
   regex misses a new family, the ``ArraysCache`` probe still flags it
   as hybrid and disables spec decoding.

Add a new field here when you have an optimization that's safe for
some arches but not others. Keep regex entries small and ordered: most
specific first.
"""

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from .model_aliases import resolve_profile

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Auto-detected configuration for a model family.

    Includes both parser defaults (tool/reasoning) and capability gates
    (which optimizations are safe to enable). Defaults err on the side
    of "supported" — known-incompatible families set the flag explicitly.
    """

    # --- Parser defaults ---
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    default_max_tokens: int | None = (
        None  # Per-model default when user omits max_tokens
    )

    # --- Architecture / capability gates ---
    # ``is_hybrid`` = the model uses linear-attention or recurrent layers
    # (GatedDeltaNet, Mamba, Jamba, ...). Hybrid models need request
    # throttling and disable optimizations that rely on chunked-batched
    # forward — verified on Qwen3.5-4B where spec decode produces
    # corrupted output (see evals/results/SUFFIX_POC_REPORT.md).
    is_hybrid: bool = False

    # ``supports_spec_decode`` controls SuffixDecoding / draft-model
    # speculative decoding. Disabled for hybrid models because the
    # batched-verify path through GatedDeltaNet derails generation.
    # Pure-attention models (llama, qwen3, mistral, gemma3, gpt-oss,
    # phi, ...) are safe.
    supports_spec_decode: bool = True


# Model family patterns → optimal config.
# Order matters: first match wins. More specific patterns go first.
_MODEL_PATTERNS: list[tuple[re.Pattern, ModelConfig]] = [
    # DeepSeek V4 / V4-Flash — sparse MoE with sliding-window attention
    # (RotatingKVCache). Pure-attention so spec decode is safe; tool
    # parser inherits the standard DeepSeek format. Upstream chat
    # template is currently chat-only with no tools (see deepseek-ai
    # discussion #16) — when fixed, just bump the parser here.
    (
        re.compile(r"deepseek.*v4", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek",
            reasoning_parser=None,
        ),
    ),
    # DeepSeek V3.1 / R1-0528 — dedicated parser, before generic deepseek
    (
        re.compile(r"deepseek.*(v3\.1|r1[-_]?0528)", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek_v31",
            reasoning_parser="deepseek_r1",
        ),
    ),
    # DeepSeek R1 (non-0528) — has reasoning
    (
        re.compile(r"deepseek.*r1", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek",
            reasoning_parser="deepseek_r1",
        ),
    ),
    # DeepSeek (V3, V2.5, etc.) — no reasoning parser
    (
        re.compile(r"deepseek", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek",
            reasoning_parser=None,
        ),
    ),
    # Qwopus (Qwen3.5 distilled with Claude Opus reasoning) — hybrid base
    (
        re.compile(r"qwopus", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            is_hybrid=True,
            supports_spec_decode=False,
        ),
    ),
    # Qwen3-Coder-Next / Qwen3-Next — hybrid linear attention, BEFORE
    # the generic Qwen3-Coder regex (which would otherwise win and tag
    # this as pure-attention by mistake).
    (
        re.compile(r"qwen3[-_]?(coder[-_]?next|next)", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
            is_hybrid=True,
            supports_spec_decode=False,
        ),
    ),
    # Qwen3.6 — hybrid GatedDeltaNet, XML tool format
    (
        re.compile(r"qwen3\.6", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="qwen3_coder_xml",
            reasoning_parser="qwen3",
            is_hybrid=True,
            supports_spec_decode=False,
        ),
    ),
    # Qwen3.5 — hybrid GatedDeltaNet (model_type=qwen3_5). Must come
    # before the generic Qwen3 regex.
    (
        re.compile(r"qwen3\.5", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            is_hybrid=True,
            supports_spec_decode=False,
        ),
    ),
    # Qwen3-Coder (older, pure-attention) — not Coder-Next
    (
        re.compile(r"qwen3[-_]?coder", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # Qwen3 (pure attention, the original Qwen3 line)
    (
        re.compile(r"qwen3", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
        ),
    ),
    # GLM family (GLM-4.5, GLM-4.7)
    (
        re.compile(r"glm[-_]?4", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="glm47",
            reasoning_parser=None,
        ),
    ),
    # MiniMax M2.5
    (
        re.compile(r"minimax", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="minimax",
            reasoning_parser="minimax",
        ),
    ),
    # GPT-OSS
    (
        re.compile(r"gpt[-_]?oss", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="harmony",
            reasoning_parser="harmony",
        ),
    ),
    # Kimi
    (
        re.compile(r"kimi", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="kimi",
            reasoning_parser=None,
        ),
    ),
    # Magistral (Mistral reasoning variant) — must precede generic
    # mistral so the reasoning_parser is set. Magistral emits standard
    # ``<think>...</think>`` so the qwen3 reasoning parser handles it.
    (
        re.compile(r"magistral", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
        ),
    ),
    # Mistral / Devstral / Mistral-Small-3.x (model_type=mistral3)
    (
        re.compile(r"mistral|devstral", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # Gemma 4 (native tool format)
    (
        re.compile(r"gemma[-_]?4", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="gemma4",
            reasoning_parser="gemma4",
        ),
    ),
    # Gemma 2/3 (hermes format)
    (
        re.compile(r"gemma", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # Hermes (fine-tuned Llama etc.)
    (
        re.compile(r"hermes", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # Llama (Llama 3.x and earlier)
    # Note: Llama 4 Scout/Maverick (109B/400B params) deliberately NOT added —
    # too large to run on the typical Mac the project targets, so the
    # validation burden (pr_validate × all agents) is not justified.
    (
        re.compile(r"llama", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="llama",
            reasoning_parser=None,
        ),
    ),
    # Phi
    (
        re.compile(r"phi[-_]?[34]", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # ---------- 2026 model families ----------
    # IBM Granite 4 (model_type=granitemoehybrid) — Mamba2 + Transformer
    # MoE with NoPE. Hybrid arch → spec decode disabled. Tool format is
    # IBM-custom; hermes is the closest existing parser as a fallback.
    # Granite 4 does NOT emit ``<think>...</think>`` reasoning blocks
    # (verified via SSE inspection: every content delta is plain text).
    # Setting ``reasoning_parser=qwen3`` here would route ALL output
    # into ``reasoning_content`` because the qwen3 parser stays in the
    # reasoning state until it sees a ``</think>`` close tag.
    (
        re.compile(r"granite[-_]?4", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
            is_hybrid=True,
            supports_spec_decode=False,
        ),
    ),
    # SmolLM3 (HuggingFace, model_type=smollm3) — pure-attention dense
    # with /think /no_think dual modes. Best-in-class at 3B.
    (
        re.compile(r"smollm3", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
        ),
    ),
    # Note: Tencent Hy3 / Hunyuan 3 (295B params, ~150GB at 4-bit) is
    # the #1 model by OpenRouter token volume but only runs on 192GB+
    # Macs — too few users have the hardware to justify the validation
    # burden. Will revisit if mlx-community ships a smaller distilled
    # variant.
    # Pure recurrent / linear-attention families (Mamba, Jamba, RWKV).
    # Tool/reasoning parsers unknown → leave defaults; capability flags
    # block batched-verify-style optimizations.
    (
        re.compile(r"mamba|jamba|rwkv", re.IGNORECASE),
        ModelConfig(
            is_hybrid=True,
            supports_spec_decode=False,
        ),
    ),
]


def detect_model_config(model_path: str) -> ModelConfig | None:
    """Detect optimal parser config from model name/path.

    Two-stage lookup:
    1. **Alias profile** (single source of truth) — if ``model_path`` is a
       known alias name (``qwen3.5-4b``) or maps to one's HF path
       (``mlx-community/Qwen3.5-4B-MLX-4bit``), return that profile's
       config directly. This guarantees per-alias granularity for any
       optimization that varies by size/quant within a family.
    2. **Regex fallback** (``_MODEL_PATTERNS``) — for non-aliased HF
       paths the user serves directly. Coarser-grained: one pattern
       covers a whole family.

    Args:
        model_path: Model name or path (e.g. "mlx-community/Qwen3.5-9B-4bit")

    Returns:
        ModelConfig if an alias profile or regex pattern matches, None
        otherwise.
    """
    profile = resolve_profile(model_path)
    if profile is not None:
        logger.info(
            f"Resolved alias profile for '{model_path}' → "
            f"tool_call_parser={profile.tool_call_parser}, "
            f"reasoning_parser={profile.reasoning_parser}, "
            f"is_hybrid={profile.is_hybrid}, "
            f"supports_spec_decode={profile.supports_spec_decode}"
        )
        return ModelConfig(
            tool_call_parser=profile.tool_call_parser,
            reasoning_parser=profile.reasoning_parser,
            default_max_tokens=profile.default_max_tokens,
            is_hybrid=profile.is_hybrid,
            supports_spec_decode=profile.supports_spec_decode,
        )

    for pattern, config in _MODEL_PATTERNS:
        if pattern.search(model_path):
            logger.info(
                f"Auto-detected model family '{pattern.pattern}' → "
                f"tool_call_parser={config.tool_call_parser}, "
                f"reasoning_parser={config.reasoning_parser}, "
                f"is_hybrid={config.is_hybrid}, "
                f"supports_spec_decode={config.supports_spec_decode}"
            )
            return config
    return None


def enrich_model_config(cfg: ModelConfig | None, model: Any) -> ModelConfig:
    """Runtime-enrich a ``ModelConfig`` from a loaded mlx-lm model.

    This is the safety net for capability gates: if regex didn't tag a
    model as hybrid (e.g. a brand-new arch we haven't added to
    ``_MODEL_PATTERNS`` yet), the ``ArraysCache`` probe still catches
    it. Always conservative — only flips capability flags **off**, never on.

    Args:
        cfg: Initial config from ``detect_model_config``, or None when
            no name pattern matched.
        model: The loaded mlx-lm model object.

    Returns:
        Updated ``ModelConfig`` (a fresh dataclass; never mutates input).
    """
    if cfg is None:
        cfg = ModelConfig()

    # Probe for ArraysCache (used by linear-attention layers — Qwen3.5
    # GatedDeltaNet, Qwen3-Next, Mamba). Same pattern that engine_core
    # has been using; consolidate it here.
    try:
        if hasattr(model, "make_cache"):
            from mlx_lm.models.cache import ArraysCache

            test_cache = model.make_cache()
            if any(isinstance(c, ArraysCache) for c in test_cache):
                if not cfg.is_hybrid or cfg.supports_spec_decode:
                    logger.info(
                        "Runtime probe: model has ArraysCache layers — "
                        "marking as hybrid, disabling spec decode"
                    )
                cfg = replace(cfg, is_hybrid=True, supports_spec_decode=False)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"ArraysCache probe failed (non-fatal): {e!r}")

    return cfg


# --- Visibility helpers ----------------------------------------------------
#
# Three levels of profile visibility for users:
#
#   Level 1 — ``format_profile_summary(model_path, cfg)`` returns a one-line
#             string suitable for a startup log: "Model profile:
#             qwen3.5 (hybrid GatedDeltaNet) → throttle ON, spec decode OFF".
#             Always emitted on engine init.
#
#   Level 2 — ``format_profile_table(model_path, cfg)`` returns a
#             multi-line ASCII table. Emitted only when verbose logging is
#             on (server --verbose, or RAPID_MLX_PROFILE=1 env var).
#
#   Level 3 — ``rapid-mlx info <model>`` CLI subcommand wraps
#             ``detect_model_config`` + ``format_profile_table`` so a user
#             can see capabilities without launching a server.


def _arch_label(cfg: "ModelConfig") -> str:
    """One-word architecture label for human display."""
    if cfg.is_hybrid:
        return "hybrid (linear-attention/Mamba)"
    return "pure attention"


def format_profile_summary(model_path: str, cfg: "ModelConfig | None") -> str:
    """Single-line profile summary for startup logs (Level 1).

    Empty/no-match models return a generic line so the log is consistent
    across known and unknown models.
    """
    if cfg is None:
        return f"Model profile: {model_path} (unknown family — using defaults)"
    parts = [_arch_label(cfg)]
    parts.append(f"throttle {'ON' if cfg.is_hybrid else 'OFF'}")
    parts.append(f"spec decode {'OFF' if not cfg.supports_spec_decode else 'OK'}")
    if cfg.tool_call_parser:
        parts.append(f"tool={cfg.tool_call_parser}")
    if cfg.reasoning_parser:
        parts.append(f"reasoning={cfg.reasoning_parser}")
    return f"Model profile: {model_path} → " + ", ".join(parts)


def format_profile_table(model_path: str, cfg: "ModelConfig | None") -> str:
    """Multi-line ASCII capability table for verbose startup output and
    the ``rapid-mlx info`` CLI command (Level 2 + Level 3).

    Width is fixed at 64 cols so it renders cleanly in terminal logs.
    Note: Unicode check/cross marks count as 1 char each (no double-width).
    """
    inner = 60  # printable width between ``│ `` and `` │`` markers
    sep = "─" * inner

    def _row(text: str) -> str:
        return f"│ {text:<{inner}} │"

    rows: list[tuple[str, str]]
    header = f"Model: {model_path}"
    if len(header) > inner:
        header = header[: inner - 1] + "…"

    if cfg is None:
        rows = [
            ("Profile", "(no pattern matched — using defaults)"),
            ("Tool format", "(none)"),
            ("Reasoning parser", "(none)"),
            ("Architecture", "unknown"),
            ("Spec decode", "✓ default-on"),
            ("Throttle", "✗ default-off"),
        ]
    else:
        spec = "✓ supported" if cfg.supports_spec_decode else "✗ disabled (hybrid arch)"
        throttle = "✓ 200ms gap" if cfg.is_hybrid else "✗ not needed"
        rows = [
            ("Tool format", cfg.tool_call_parser or "(none)"),
            ("Reasoning parser", cfg.reasoning_parser or "(none)"),
            ("Architecture", _arch_label(cfg)),
            ("Spec decode", spec),
            ("Throttle", throttle),
        ]

    body = [_row(header), _row(sep)]
    for k, v in rows:
        body.append(_row(f"{k:<17}: {v}"))

    top = "┌" + "─" * (inner + 2) + "┐"
    bot = "└" + "─" * (inner + 2) + "┘"
    return "\n".join([top, *body, bot])


def get_profile(model_path: str, model: object | None = None) -> "ModelConfig":
    """One-shot profile lookup combining both stages.

    This is the public API for code that wants the final ModelConfig in
    one call: regex pattern match → optional runtime ArraysCache probe.
    Always returns a ``ModelConfig`` (never None) — falls back to defaults
    when nothing matches so downstream code doesn't need null checks.

    Args:
        model_path: Model name or HF repo path.
        model: Optional loaded mlx-lm model object. When provided, runtime
            probe runs as a safety net for unknown hybrid arches.

    Returns:
        Final merged ``ModelConfig``.
    """
    cfg = detect_model_config(model_path) or ModelConfig()
    if model is not None:
        cfg = enrich_model_config(cfg, model)
    return cfg
