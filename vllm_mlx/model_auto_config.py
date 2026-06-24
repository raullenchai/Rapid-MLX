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
from dataclasses import dataclass, field, replace
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

    # r6-A R6-C1: when the alias profile (or an explicit caller) pins
    # ``is_hybrid``, ``enrich_model_config``'s runtime ArraysCache probe
    # MUST NOT one-way-flip the value to True. Without this gate, dense
    # Qwen3.5 / Qwen3.6 aliases whose JSON declares ``is_hybrid=false``
    # still got promoted to hybrid at boot because ``make_cache()``
    # returns linear-attention layers — re-enabling the throttle +
    # prefix-boundary snapshot path that wedges ``rapid-mlx serve
    # qwen3.5-4b-4bit`` with ``metal::malloc`` Resource-limit (499000)
    # errors. Default ``False`` preserves the legacy safety-net behaviour
    # for aliases / serve targets that haven't opted into the explicit
    # contract; the probe still promotes ``is_hybrid`` to True when the
    # cache type indicates linear attention.
    is_hybrid_explicit: bool = False

    # ``supports_spec_decode`` controls SuffixDecoding / draft-model
    # speculative decoding. Disabled for hybrid models because the
    # batched-verify path through GatedDeltaNet derails generation.
    # Pure-attention models (llama, qwen3, mistral, gemma3, gpt-oss,
    # phi, ...) are safe.
    supports_spec_decode: bool = True

    # SuffixDecoding eligibility tier (#269). One of:
    #   "unknown"    — not benched (silent default)
    #   "agent"      — tool_loop ≥ 1.8x, no regression — recommend the flag
    #   "structured" — peak workload ≥ 1.5x, no regression — may help
    #   "neutral"    — no workload wins, no regression — silent
    #   "avoid"      — at least one workload regresses — warn
    suffix_decoding_tier: str = "unknown"
    # Per-workload speedup measured by ``scripts/bench_suffix_decoding_integrated.py``.
    # ``field(default_factory=dict)`` so each ``ModelConfig`` instance gets
    # its own fresh dict (a literal ``{}`` would silently share state).
    suffix_bench_speedup: dict[str, float] = field(default_factory=dict)

    # PFlash long-prompt compression eligibility (#287). Mirrors
    # ``AliasProfile.pflash_tier`` — the single source of truth lives in
    # ``aliases.json`` and is copied here by ``detect_model_config`` so
    # ``serve``/``bench`` can pick up the default without re-resolving
    # the profile. Values: ``"unknown"`` (engine defaults PFlash off) or
    # ``"verified"`` (engine defaults PFlash to ``always``). Explicit
    # CLI ``--pflash`` still wins. See VALID_PFLASH_TIERS for the enum.
    pflash_tier: str = "unknown"


# DEPRECATED dispatch surface — see ``vllm_mlx/reasoning/think_detector.py``.
#
# The name-regex map below is the ONLY fall-back when a serve target lacks
# an explicit alias entry in ``aliases.json``. Every entry in this map is
# a per-model regex used to dispatch parser implementations; the user has
# called this pattern out as the antipattern to avoid in PRs after #715
# (which added the ``vibethinker`` + Qwen3 non-thinking entries).
#
# Migration target: aliases declare capability booleans
# (``can_emit_think``, ``has_native_tool_format``, …) and the engine
# picks parser implementations at runtime via ``ThinkDetector`` and the
# tool-call format probe. Do NOT add new regex entries here — extend
# ``aliases.json`` instead, which is the source of truth for any model
# the project officially supports. Existing entries stay in place until
# the migration completes (tracked separately so PRs stay tight on a
# single issue).
#
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
    # DeepSeek V3.1 (thinking-channel wire shape: NAME<sep>{json}).
    # Matched before V3 / R1-0528 so the more specific pattern wins.
    (
        re.compile(r"deepseek.*v3\.1", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek_v31",
            reasoning_parser="deepseek_r1",
        ),
    ),
    # DeepSeek-R1-0528 (V3 chat template — function-typed fenced JSON).
    # R12-5: split off the V3.1 parser to its own DeepSeekV3ToolParser.
    (
        re.compile(r"deepseek.*r1[-_]?0528", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek_v3",
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
    # DeepSeek V3 (vanilla checkpoints: V3-0324 etc.) — same
    # function-typed fenced JSON wire shape as R1-0528. R12-5: route
    # to the dedicated V3 parser so vanilla V3 users get the same
    # forced-tool prefix injection as R1-0528 (codex r3 P2 — without
    # this, a direct serve of ``deepseek-ai/DeepSeek-V3-0324`` falls
    # through to the generic ``deepseek`` parser, which has neither
    # the block-wise scanner hardening nor a forced-prefix branch).
    # Matched AFTER V3.1 (above) so the more specific pattern wins.
    (
        re.compile(r"deepseek.*v3(?![._\d])", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek_v3",
            reasoning_parser=None,
        ),
    ),
    # DeepSeek (V2.5 and older) — no reasoning parser
    (
        re.compile(r"deepseek", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek",
            reasoning_parser=None,
        ),
    ),
    # UI-TARS (ByteDance) — Qwen2-VL / Qwen2.5-VL based GUI-agent VLM.
    # Wire format is the literal ``Action: verb(kwargs)`` Computer-Use
    # shape (see vllm_mlx.tool_parsers.ui_tars_tool_parser). MUST come
    # BEFORE any generic Qwen2/Qwen2.5 pattern would otherwise match —
    # full HF paths like ``mlx-community/UI-TARS-7B-DPO-4bit`` should
    # resolve here, not to the generic Qwen3 fallback.
    (
        re.compile(r"ui[-_]?tars", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="ui_tars",
            reasoning_parser="ui_tars",
            is_hybrid=False,
            # UI-TARS uses Qwen2-VL/Qwen2.5-VL mrope; spec decode hasn't
            # been benched on the VLM variant. Keep off until verified
            # to avoid silent quality regressions (mirrors the gemma 3n
            # / phi-3.5 conservative defaults).
            supports_spec_decode=False,
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
    # VibeThinker (Weibo AI reasoning derivative, base = Qwen2.5-Coder-3B).
    # Pure-attention Qwen2 architecture; chat template does NOT inject
    # ``<think>`` — the model emits ``<think>...</think>`` autonomously on
    # every response. ``deepseek_r1`` parser handles that "model decides"
    # contract (same as DeepSeek-R1 distill on Qwen base).
    #
    # 2026-06-17 VibeThinker live test (PR for #708 follow-up): although
    # the upstream model card disowns tool calling, the inherited Qwen2
    # vocab carries the ``<tool_call>`` / ``</tool_call>`` and
    # ``<function=...>`` tokens AND the live test confirmed the 3B-8bit
    # weights emit BOTH shapes when prompted with tools (Test 4 of the
    # live-test report). Wire ``hermes`` parser so the bare
    # ``<function=name>...</function>`` shape (which the OutputRouter
    # token-fallback misses) lands in ``tool_calls`` instead of leaking
    # as raw text into ``content``.
    #
    # Placed before the generic ``qwen`` regex would have been (there is
    # none today) — this pattern is the only signal for full-HF-path
    # serves of ``WeiboAI/VibeThinker-3B`` or
    # ``mlx-community/VibeThinker-3B-*`` that miss the alias lookup.
    (
        re.compile(r"vibethinker", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            # ``vibethinker`` parser — DeepSeek-R1 variant with a 1024-char
            # no-tag threshold for the preamble-before-``<think>`` shape
            # (codex r2 P2 — keeps the base ``deepseek_r1`` threshold at 64
            # for distilled-on-Qwen aliases that DO open with ``<think>``
            # immediately).
            reasoning_parser="vibethinker",
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
    # Qwen3.6 MoE (A3B / A10B / generic MoE markers) — hybrid
    # GatedDeltaNet + sparse experts, XML tool format. r6-A R6-C1: the
    # earlier bare ``qwen3\.6`` regex also fired on the DENSE 27B variant
    # (mlx-community/Qwen3.6-27B-4bit, model_type=qwen3_5), which carries
    # GatedDeltaNet layers but wedges on metal::malloc when the engine
    # opts into the hybrid throttle + prefix-boundary snapshot path. The
    # MoE-marker gate keeps the hybrid stamp ON for the A3B (35B) MoE
    # variants that actually need it, while the dense 27B falls through
    # to the generic Qwen3 fallback (pure-attention contract).
    (
        re.compile(r"qwen3\.6.*(a3b|a10b|moe)", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="qwen3_coder_xml",
            reasoning_parser="qwen3",
            is_hybrid=True,
            supports_spec_decode=False,
        ),
    ),
    # Qwen3.6 dense (non-MoE) — same XML tool format, but NOT hybrid for
    # routing purposes (see the MoE branch above for the r6-A R6-C1
    # rationale: dense GatedDeltaNet variants wedge under the hybrid
    # scheduler path on Metal).
    (
        re.compile(r"qwen3\.6", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="qwen3_coder_xml",
            reasoning_parser="qwen3",
        ),
    ),
    # Qwen3.5 MoE (A3B / A10B / generic MoE markers) — hybrid
    # GatedDeltaNet + sparse experts (model_type=qwen3_5_moe). Must come
    # before the generic Qwen3 regex. r6-A R6-C1: the prior bare
    # ``qwen3\.5`` regex also stamped DENSE variants
    # (mlx-community/Qwen3.5-4B-MLX-4bit, model_type=qwen3_5) as hybrid,
    # which surfaces as a ``metal::malloc`` Resource-limit (499000) wedge
    # on every generation step — the hybrid scheduler's allocation
    # pattern is incompatible with the dense GatedDeltaNet cache layout
    # at the 4B/9B/27B sizes. Restricting the hybrid stamp to MoE markers
    # keeps the A3B (35B) / A10B (122B) variants on the correct path
    # while dense siblings fall through to the generic Qwen3 fallback.
    (
        re.compile(r"qwen3\.5.*(a3b|a10b|moe)", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            is_hybrid=True,
            supports_spec_decode=False,
        ),
    ),
    # Qwen3.5 dense (non-MoE) — same hermes tool format, but NOT hybrid
    # for routing purposes (see the MoE branch above for the r6-A R6-C1
    # rationale).
    (
        re.compile(r"qwen3\.5", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
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
    # Qwen3 non-thinking variants — these explicitly DO NOT emit
    # ``<think>...</think>`` and the qwen3 reasoning parser's Case-4
    # fallback ("no tags + ``enable_thinking=True`` → all output is
    # reasoning", #575) duplicates the entire response into BOTH
    # ``content`` and ``reasoning_content`` when the client passes
    # ``enable_thinking=True``. The 2026-06-18 fuzz battery against PR
    # #714 caught this on the Qwen3-VL-2B-Instruct and
    # Qwen3-4B-Instruct-2507 4-bit MLX repacks.
    #
    # MUST come BEFORE the generic ``qwen3`` regex below. The Thinking
    # sibling (Qwen3-4B-Thinking-2507) takes the family default since
    # ``thinking`` won't match either of these.
    (
        re.compile(
            r"qwen3[-_]?(?:vl[-_]?2b|4b[-_]?instruct)",
            re.IGNORECASE,
        ),
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
    # Gemma 3n — on-device multimodal (text+image+audio). The chat
    # template does NOT define tool-call special tokens, and the 2026-
    # 06-18 fuzz battery against PR #714 confirmed the model ignores
    # tool prompts entirely (returns prose, not a parseable envelope).
    # ``tool_call_parser=hermes`` advertised tool capability the model
    # cannot honour. Match BEFORE the generic ``gemma`` regex so the
    # 3n variants resolve to ``tool_call_parser=None``.
    (
        re.compile(r"gemma[-_]?3n", re.IGNORECASE),
        ModelConfig(
            tool_call_parser=None,
            reasoning_parser=None,
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
    # Nanbeige 4.x (Nanbeige LLM Lab) — model_type=llama under the hood
    # at the 3B preview, but the model is NOT a vanilla LLaMA-3 chat
    # checkpoint: its chat template + tool format are upstream-Nanbeige,
    # not Meta-Llama. Letting the bare HF path fall through to the
    # generic ``llama`` regex below would mis-tag ``tool_call_parser=llama``
    # and silently break tool calls. Pin to the safer ``hermes`` fallback.
    # Smoke test (PR #715 batch): Nanbeige4.1-3B emits autonomous
    # ``<think>...</think>`` blocks on every response — verified by a
    # local ``rapid-mlx serve nanbeige4.1-3b-4bit`` + chat completion
    # where the assistant content opened with ``<think>\n...`` despite
    # no template-level injection. Use ``deepseek_r1`` reasoning parser
    # (same "model decides" contract as VibeThinker / DeepSeek-R1
    # distill on a Qwen base) so the block lands in
    # ``reasoning_content`` instead of leaking into ``content``.
    # MUST come BEFORE the ``llama`` regex below — first-match-wins.
    (
        re.compile(r"nanbeige", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="deepseek_r1",
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
    # Phi-4-mini-reasoning — Microsoft's math-tuned 3.8B reasoning
    # variant of Phi-4-mini. The chat template does NOT inject any
    # ``<think>`` tag (the only special tokens are ``<|user|>`` /
    # ``<|assistant|>`` / ``<|end|>`` / ``<|tool_call|>`` — verified
    # via tokenizer_config.json), but the model emits
    # ``<think>...</think>`` autonomously on every response (smoke-
    # verified: ``Say hi`` returned ``<think>\nOkay, I need to say hi
    # in three words...`` as the assistant content with the deepseek_r1
    # parser disabled). Use ``deepseek_r1`` — same "model decides"
    # contract as VibeThinker / R1-distill / Nanbeige4.1 — so the block
    # lands in ``reasoning_content`` instead of leaking into ``content``.
    # MUST come BEFORE the generic ``phi[-_]?[34]`` regex below.
    (
        re.compile(r"phi[-_]?4[-_]?mini[-_]?reasoning", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="deepseek_r1",
        ),
    ),
    # Phi-3.5-mini — the chat template only defines ``<|user|>`` /
    # ``<|assistant|>`` / ``<|end|>`` (no ``<tool_call>`` special token);
    # the 2026-06-18 fuzz battery against PR #714 confirmed the model
    # ignores tool prompts. Pin ``tool_call_parser=None`` BEFORE the
    # generic ``phi`` regex so the bare-HF-path serves don't advertise
    # tool capability the model cannot honour. The Phi-4 family (which
    # CAN tool-call) and Phi-4-mini-reasoning (handled above) are
    # unaffected.
    (
        re.compile(r"phi[-_]?3\.?5", re.IGNORECASE),
        ModelConfig(
            tool_call_parser=None,
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
       known alias name (``qwen3.5-4b-4bit``) or maps to one's HF path
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
            f"supports_spec_decode={profile.supports_spec_decode}, "
            f"suffix_tier={profile.suffix_decoding_tier}, "
            f"pflash_tier={profile.pflash_tier}"
        )
        # AliasProfile stores the bench dict as a sorted tuple (frozen
        # dataclasses must avoid mutable shared state). Materialize a
        # fresh dict here so each ModelConfig instance owns its copy.
        speedup = (
            dict(profile.suffix_bench_speedup) if profile.suffix_bench_speedup else {}
        )
        return ModelConfig(
            tool_call_parser=profile.tool_call_parser,
            reasoning_parser=profile.reasoning_parser,
            default_max_tokens=profile.default_max_tokens,
            is_hybrid=profile.is_hybrid,
            # r6-A R6-C1: thread the explicit-pin flag so
            # ``enrich_model_config`` can honour aliases that have
            # deliberately marked their model as non-hybrid even when
            # the upstream ``make_cache()`` returns linear-attention
            # layers (qwen3_5 dense weights).
            is_hybrid_explicit=profile.is_hybrid_explicit,
            supports_spec_decode=profile.supports_spec_decode,
            suffix_decoding_tier=profile.suffix_decoding_tier,
            suffix_bench_speedup=speedup,
            pflash_tier=profile.pflash_tier,
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
    #
    # r6-A R6-C1: when the alias profile (or an explicit caller) pinned
    # ``is_hybrid_explicit=True``, the probe's hybrid promotion is
    # suppressed — the JSON/CLI is the authoritative source of truth and
    # the boot path must not silently override it. ``supports_spec_decode``
    # is still forced off when ArraysCache is present so the drafter
    # never wires up against a linear-attention model regardless of the
    # routing decision (which is a separate safety contract). Without
    # this gate, dense Qwen3.5 / Qwen3.6 aliases that declared
    # ``is_hybrid=false`` were silently re-promoted to hybrid at boot,
    # which is the path that wedges metal::malloc on the 4B variant.
    try:
        if hasattr(model, "make_cache"):
            from mlx_lm.models.cache import ArraysCache

            test_cache = model.make_cache()
            if any(isinstance(c, ArraysCache) for c in test_cache):
                if cfg.is_hybrid_explicit:
                    if cfg.supports_spec_decode:
                        logger.info(
                            "Runtime probe: model has ArraysCache layers — "
                            "honouring is_hybrid_explicit=True (keeping "
                            "is_hybrid=%s), forcing supports_spec_decode=False",
                            cfg.is_hybrid,
                        )
                        cfg = replace(cfg, supports_spec_decode=False)
                else:
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


# --- SuffixDecoding tier classification (#269) ----------------------------
#
# Pure function so the boundary logic is unit-testable in isolation. Bench
# numbers come from ``scripts/bench_suffix_decoding_integrated.py``;
# thresholds are tuned to match the qualitative recommendation we'd give
# a user looking at the same table by eye:
#
#   - AGENT     — tool calling specifically wins big AND nothing regresses
#                 (we'd tell the user "turn it on").
#   - STRUCTURED — some workload wins meaningfully AND nothing meaningfully
#                  regresses (we'd say "try it for that workload").
#   - NEUTRAL   — within noise across the board (silent — no point
#                 suggesting either direction).
#   - AVOID     — anything regresses past 0.85x, or signal is too mixed
#                 to recommend (warn at startup).


def classify_suffix_decoding_tier(speedup: dict[str, float]) -> str:
    """Map a per-workload speedup dict to a tier string.

    Empty dict → "unknown". Single-workload dicts use the special-case
    rule that an empty ``min(others)`` is treated as +∞ (the AGENT gate
    is satisfied vacuously). See ``tests/test_suffix_decoding_tier.py``
    for boundary cases including the real Qwen3-0.6B / Qwen3-14B numbers.
    """
    if not speedup:
        return "unknown"

    lo = min(speedup.values())
    hi = max(speedup.values())

    # AVOID first: any individual workload regressing past 0.85x means
    # we don't know the user's traffic mix well enough to recommend.
    if lo < 0.85:
        return "avoid"

    # AGENT — tool_loop must be the workload winning big, AND no other
    # workload regresses past 0.95x. Tool_loop missing from the dict
    # means the bench didn't measure it; we can't claim agent then.
    tool_loop = speedup.get("tool_loop")
    if tool_loop is not None and tool_loop >= 1.8:
        others = [v for k, v in speedup.items() if k != "tool_loop"]
        if not others or min(others) >= 0.95:
            return "agent"

    # STRUCTURED — some workload wins meaningfully (≥1.5x) AND the
    # weakest workload still clears 0.90x (small regression tolerated
    # because the user is opting in for the structured win).
    if hi >= 1.5 and lo >= 0.90:
        return "structured"

    # NEUTRAL — flat across the board. Tighter than STRUCTURED's 0.90
    # floor: we want true noise here, not a near-miss STRUCTURED.
    if lo >= 0.95 and hi >= 1.0 and hi < 1.5:
        return "neutral"

    # Mixed signal that didn't fit any positive bucket — recommend AVOID
    # rather than silently shipping ambiguous data.
    return "avoid"


def suffix_decoding_hint(cfg: "ModelConfig | None") -> str | None:
    """Startup hint for the SuffixDecoding flag, or ``None`` for silent tiers.

    The hint surfaces only AGENT / STRUCTURED / AVOID tiers. UNKNOWN and
    NEUTRAL stay silent — no user-visible nudge until bench data exists
    or there's a real regression to warn about.

    Hybrid arches (``supports_spec_decode=False``) always return ``None``
    even if the tier was somehow set: spec decoding is gated off at the
    engine level, and a "recommended" hint there would just confuse.
    """
    if cfg is None:
        return None
    if not cfg.supports_spec_decode:
        return None
    tier = cfg.suffix_decoding_tier
    speedup = cfg.suffix_bench_speedup or {}
    if tier == "agent":
        peak = speedup.get("tool_loop") or (max(speedup.values()) if speedup else 0)
        return (
            f"SuffixDecoding: recommended for tool/agent traffic "
            f"(tool_loop {peak:.1f}x). Pass --suffix-decoding to enable."
        )
    if tier == "structured":
        peak_key = max(speedup, key=speedup.get) if speedup else "structured"
        peak_val = speedup.get(peak_key, 0)
        return (
            f"SuffixDecoding: may help on {peak_key} ({peak_val:.2f}x). "
            "Pass --suffix-decoding if your traffic matches."
        )
    if tier == "avoid":
        worst_key = min(speedup, key=speedup.get) if speedup else "some workloads"
        worst_val = speedup.get(worst_key, 0)
        return (
            f"SuffixDecoding: NOT recommended for this model — {worst_key} "
            f"regresses to {worst_val:.2f}x. Leave --suffix-decoding off."
        )
    return None


def _arch_label(cfg: "ModelConfig") -> str:
    """One-word architecture label for human display."""
    if cfg.is_hybrid:
        return "hybrid (linear-attention/Mamba)"
    return "pure attention"


def _suffix_tier_cell(cfg: "ModelConfig", max_width: int | None = None) -> str:
    """Format the ``Suffix tier`` row for ``rapid-mlx info``.

    AGENT/STRUCTURED — surface the peak workload speedup (the reason the
    tier was assigned). AVOID — surface the worst-regressing workload so
    the user understands the warning. UNKNOWN — point them at the bench
    script. Hybrid arches always render ``n/a`` regardless of tier
    because ``supports_spec_decode=False`` gates the flag off anyway.

    When ``max_width`` is set and the produced string would exceed it,
    the parenthetical note after the tier word (``avoid``/``prefer``/
    ``neutral``/…) is truncated so the value fits inside the caller's
    box column without breaking alignment. The tier word itself is kept
    intact because it's the load-bearing signal. Truncated notes end
    with ``…)`` instead of ``)``.
    """
    if not cfg.supports_spec_decode:
        text = "n/a (hybrid arch — spec decode off)"
    else:
        tier = cfg.suffix_decoding_tier
        speedup = cfg.suffix_bench_speedup or {}
        if tier == "unknown":
            text = "unknown — run scripts/bench_suffix_decoding_integrated"
        elif tier == "agent" and speedup:
            peak_key = (
                "tool_loop" if "tool_loop" in speedup else max(speedup, key=speedup.get)
            )
            text = (
                f"agent ({peak_key} {speedup[peak_key]:.2f}x"
                " — recommend --suffix-decoding)"
            )
        elif tier == "structured" and speedup:
            peak_key = max(speedup, key=speedup.get)
            text = (
                f"structured ({peak_key} {speedup[peak_key]:.2f}x"
                " — try if traffic matches)"
            )
        elif tier == "neutral":
            text = "neutral (within noise — leave off)"
        elif tier == "avoid" and speedup:
            worst_key = min(speedup, key=speedup.get)
            text = (
                f"avoid ({worst_key} {speedup[worst_key]:.2f}x regression — leave off)"
            )
        else:
            text = tier
    return _truncate_tier_note(text, max_width)


def _truncate_tier_note(text: str, max_width: int | None) -> str:
    """Shorten a ``tier (note)`` string to fit within ``max_width`` chars.

    Only the parenthetical note is trimmed; the leading tier word stays
    whole. If the tier word alone already overflows (shouldn't happen
    with current tiers but kept defensive), the full text is returned
    unchanged — the caller's column will visibly break, surfacing the
    bug instead of silently dropping load-bearing data.

    The ``tier — note`` (em-dash) form used by the ``unknown`` tier is
    handled as a fallback so that variant also fits inside the box.
    """
    if max_width is None or len(text) <= max_width:
        return text
    open_paren = text.find("(")
    if open_paren != -1 and text.endswith(")"):
        # ``prefix`` = ``tier (`` — keep verbatim. Available room for
        # note body = max_width − len(prefix) − len("…)").
        prefix = text[: open_paren + 1]
        available = max_width - len(prefix) - len("…)")
        if available < 1:
            return text
        note_body = text[open_paren + 1 : -1]
        return prefix + note_body[:available].rstrip() + "…)"
    em_dash = text.find(" — ")
    if em_dash != -1:
        prefix = text[: em_dash + 3]  # include the `` — `` separator
        available = max_width - len(prefix) - len("…")
        if available < 1:
            return text
        note_body = text[em_dash + 3 :]
        return prefix + note_body[:available].rstrip() + "…"
    return text


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
    # Value column = ``inner`` minus the 17-char key field and the
    # 2-char ``": "`` separator. Used by ``_suffix_tier_cell`` to keep
    # long parenthetical notes inside the box.
    value_width = inner - 17 - 2
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
            (
                "Suffix tier",
                _truncate_tier_note(
                    "unknown — run scripts/bench_suffix_decoding_integrated",
                    value_width,
                ),
            ),
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
            ("Suffix tier", _suffix_tier_cell(cfg, max_width=value_width)),
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
