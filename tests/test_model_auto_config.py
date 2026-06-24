"""Tests for model auto-config detection."""

import pytest

from vllm_mlx.model_auto_config import (
    ModelConfig,
    detect_model_config,
    enrich_model_config,
    format_profile_summary,
    format_profile_table,
    get_profile,
    warn_misbound_deepseek_v3_parser,
)


class TestDetectModelConfig:
    """Test detect_model_config with various model paths."""

    # Qwen family (non-Coder) — covers the original Qwen3 line, the
    # Qwen3.5 family, and the Qwen3-4B-Thinking-2507 small variant.
    # The ``qwen3`` regex resolves all of these to the same ``hermes``
    # + ``qwen3`` parser pair.
    #
    # Qwen3-4B-Instruct-2507 and Qwen3-VL-2B-Instruct deliberately NOT
    # listed here — they are NON-thinking variants and have their own
    # auto-config regex (above the generic ``qwen3``) that clears the
    # reasoning parser. See ``test_qwen3_non_thinking_variants`` below.
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/Qwen3.5-9B-4bit",
            "mlx-community/Qwen3-0.6B-MLX-4bit",
            "/Users/someone/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-8bit",
            "mlx-community/Qwen3-4B-Thinking-2507-4bit",
            "Qwen/Qwen3-4B-Thinking-2507",
        ],
    )
    def test_qwen_family(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "hermes"
        assert config.reasoning_parser == "qwen3"

    # Qwen3 non-thinking variants — Instruct-2507 and VL-2B. These do
    # NOT emit ``<think>...</think>`` autonomously; wiring the qwen3
    # reasoning parser duplicates the response into BOTH content and
    # reasoning_content when the client passes ``enable_thinking=True``
    # (PR #715 bundle, fuzz finding A). The dedicated regex MUST win
    # over the generic ``qwen3`` regex.
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/Qwen3-4B-Instruct-2507-4bit",
            "Qwen/Qwen3-4B-Instruct-2507",
            "mlx-community/Qwen3-VL-2B-Instruct-4bit",
            "Qwen/Qwen3-VL-2B-Instruct",
        ],
    )
    def test_qwen3_non_thinking_variants(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.tool_call_parser == "hermes", (
            f"{model_path}: tool_call_parser stays 'hermes'. "
            f"Got {cfg.tool_call_parser!r}."
        )
        assert cfg.reasoning_parser is None, (
            f"{model_path}: reasoning_parser must be None — non-thinking "
            f"Qwen3 variant. Did the specific regex get demoted below the "
            f"generic 'qwen3' regex? Got {cfg.reasoning_parser!r}."
        )

    # GLM family
    @pytest.mark.parametrize(
        "model_path",
        [
            "lmstudio-community/GLM-4.7-Flash-MLX-8bit",
            "GLM-4.5-Air-MLX-4bit",
            "glm4-9b-chat",
        ],
    )
    def test_glm_family(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "glm47"
        assert config.reasoning_parser is None

    # MiniMax
    def test_minimax(self):
        config = detect_model_config("lmstudio-community/MiniMax-M2.5-MLX-4bit")
        assert config is not None
        assert config.tool_call_parser == "minimax"
        assert config.reasoning_parser == "minimax"

    # GPT-OSS
    def test_gpt_oss(self):
        config = detect_model_config("mlx-community/gpt-oss-20b-MXFP4-Q8")
        assert config is not None
        assert config.tool_call_parser == "harmony"
        assert config.reasoning_parser == "harmony"

    # Mistral / Devstral
    @pytest.mark.parametrize(
        "model_path",
        [
            "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
            "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
        ],
    )
    def test_mistral_devstral(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "hermes"
        assert config.reasoning_parser is None

    # Qwen3-Coder (no reasoning parser)
    @pytest.mark.parametrize(
        "model_path",
        [
            "Qwen3-Coder-Next-MLX-4bit",
            "lmstudio-community/Qwen3-Coder-Next-MLX-6bit",
        ],
    )
    def test_qwen_coder(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "hermes"
        assert config.reasoning_parser is None

    # DeepSeek V3.1 thinking-channel → deepseek_v31 parser. R12-5
    # split: V3.1 is V3.1-only; R1-0528 lives on the V3 parser below
    # because its chat_template.jinja was inherited from V3.
    @pytest.mark.parametrize(
        "model_path",
        [
            "deepseek-ai/DeepSeek-V3.1-0324",
        ],
    )
    def test_deepseek_v31(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "deepseek_v31"
        assert config.reasoning_parser == "deepseek_r1"

    # R12-5: DeepSeek-R1-0528 (V3-shape chat template) → deepseek_v3
    # parser. Split off from deepseek_v31 in PR for 0.8.14 — the V3.1
    # parser was carrying V3 auto-detect logic that risked the historic
    # ``name="function"`` mis-emission on edge cases.
    @pytest.mark.parametrize(
        "model_path",
        [
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "mlx-community/DeepSeek-R1-0528-4bit",
        ],
    )
    def test_deepseek_r1_0528_routes_to_v3_parser(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "deepseek_v3"
        assert config.reasoning_parser == "deepseek_r1"

    # DeepSeek V4 / V4-Flash — sparse MoE with sliding-window attention,
    # pure-attention (spec decode safe).
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/DeepSeek-V4-Flash-8bit",
            "mlx-community/DeepSeek-V4-Flash-2bit-DQ",
            "mlx-community/DeepSeek-V4-Flash-4bit",
            "deepseek-ai/DeepSeek-V4",
        ],
    )
    def test_deepseek_v4(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "deepseek"
        # V4-Flash chat template emits `<think>...</think>` blocks gated
        # by ``thinking_mode``; ``deepseek_r1`` handles that format. The
        # base ``deepseek-ai/DeepSeek-V4`` path is resolved via family
        # detection (no aliases.json entry), so it currently gets no
        # reasoning_parser — only the MLX variants benefit from the
        # alias wiring. Track both shapes here so a refactor that flips
        # the family default has to update this test consciously.
        if model_path == "deepseek-ai/DeepSeek-V4":
            assert config.reasoning_parser is None
        else:
            assert config.reasoning_parser == "deepseek_r1"
        assert config.is_hybrid is False
        assert config.supports_spec_decode is True

    # ---- 2026 model families ----

    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/granite-4.0-h-small-4bit",
            "mlx-community/granite-4.0-h-tiny-4bit",
            "mlx-community/granite-4.0-h-micro-4bit",
            "ibm-granite/granite-4.0-h-small",
            "ibm-granite/granite-4.0-h-micro",
        ],
    )
    def test_granite4_hybrid(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.tool_call_parser == "hermes"
        # Granite 4 does NOT emit <think>...</think> reasoning. Setting
        # a reasoning parser would route all output into reasoning_content.
        assert cfg.reasoning_parser is None
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    def test_smollm3(self):
        cfg = detect_model_config("mlx-community/SmolLM3-3B-4bit")
        assert cfg is not None
        assert cfg.tool_call_parser == "hermes"
        assert cfg.reasoning_parser == "qwen3"
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True

    # VibeThinker (Weibo AI reasoning family; 1.5B base = Qwen2.5-Math-1.5B,
    # 3B base = Qwen2.5-Coder-3B). Verify both the alias paths
    # (vibethinker-{1.5b-4bit,3b-8bit} → JSON profile) and the bare-HF-path
    # regex fallback (WeiboAI/VibeThinker-{1.5B,3B}, served by full repo id
    # without an alias) wire ``deepseek_r1`` reasoning parser so
    # ``<think>...</think>`` blocks land in ``reasoning_content`` not
    # ``content``. ``tool_call_parser`` is ``hermes`` after the
    # 2026-06-17 live test confirmed the model emits both
    # ``<tool_call>{...}</tool_call>`` and bare
    # ``<function=name>...</function>`` shapes — see
    # ``test_aliases_contract.test_vibethinker_family_wires_deepseek_r1_reasoning_parser``
    # for the full rationale.
    @pytest.mark.parametrize(
        "model_path",
        [
            "vibethinker-1.5b-4bit",
            "mlx-community/VibeThinker-1.5B-mlx-4bit",
            "WeiboAI/VibeThinker-1.5B",
            "vibethinker-3b-8bit",
            "mlx-community/VibeThinker-3B-8bit",
            "WeiboAI/VibeThinker-3B",
        ],
    )
    def test_vibethinker(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        # ``vibethinker`` parser — DeepSeek-R1 variant with a larger
        # no-tag threshold for preamble-before-``<think>`` (codex r2 P2).
        assert cfg.reasoning_parser == "vibethinker"
        assert cfg.tool_call_parser == "hermes"
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True

    # Nanbeige 4.x (Nanbeige LLM Lab) — model_type=llama in config.json
    # but NOT a vanilla Meta-LLaMA-3 chat checkpoint. Pinned ahead of
    # the generic ``llama`` regex in model_auto_config.py so a bare HF
    # path serve picks up the upstream-Nanbeige tool/reasoning shape
    # (``hermes`` + ``deepseek_r1`` — the 3B preview emits autonomous
    # ``<think>...</think>`` blocks on every response, smoke-verified)
    # instead of the LLaMA tool parser.
    @pytest.mark.parametrize(
        "model_path",
        [
            "nanbeige4.1-3b-4bit",
            "mlx-community/Nanbeige4.1-3B-4bit",
            "Nanbeige/Nanbeige4.1-3B",
        ],
    )
    def test_nanbeige(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        # The Nanbeige regex must win — `tool_call_parser="llama"` here
        # would mean the generic LLaMA regex misfired, and tool calls
        # would silently fail at runtime.
        assert cfg.tool_call_parser == "hermes", (
            f"{model_path}: tool_call_parser must be 'hermes', got "
            f"{cfg.tool_call_parser!r} — did the regex order change?"
        )
        # Nanbeige4.1-3B emits autonomous ``<think>...</think>`` blocks
        # (smoke-verified). ``deepseek_r1`` parser routes the block into
        # ``reasoning_content`` so it doesn't leak into ``content``.
        assert cfg.reasoning_parser == "deepseek_r1"
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True

    @pytest.mark.parametrize(
        "model_path",
        [
            "mistralai/Magistral-Small-2509",
            "mlx-community/Magistral-Small-2509-4bit",
        ],
    )
    def test_magistral(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        # Magistral routes through its own entry, NOT generic Mistral.
        # Critical: reasoning_parser must be set.
        assert cfg.reasoning_parser == "qwen3"
        assert cfg.is_hybrid is False

    # DeepSeek R1 (non-0528) → deepseek parser + reasoning
    def test_deepseek_r1(self):
        config = detect_model_config("deepseek-ai/DeepSeek-R1")
        assert config is not None
        assert config.tool_call_parser == "deepseek"
        assert config.reasoning_parser == "deepseek_r1"

    # R12-S1 (Sven r12 HIGH-1 verdict): DeepSeek-R1-Distill-* checkpoints
    # are Qwen2 / Llama2-arch SFTs whose tokenizers do NOT carry the
    # V3 fullwidth-pipe special tokens — they emit the V2-style envelope
    # the legacy ``deepseek`` parser handles, NOT the V3 fenced-JSON
    # envelope ``deepseek_v3`` expects. The regex ordering MUST keep
    # these models on the legacy parser; if a future edit flipped them
    # to ``deepseek_v3`` (because the family name still contains "r1"),
    # Sven's HIGH-1 would re-surface as a default-config regression
    # rather than a wrong-flag misbind.
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ],
    )
    def test_deepseek_r1_distill_routes_to_legacy_parser(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        # MUST be the legacy ``deepseek`` parser, NOT ``deepseek_v3`` —
        # the distills cannot emit the V3 wire shape.
        assert config.tool_call_parser == "deepseek", (
            f"{model_path}: tool_call_parser must be 'deepseek' (legacy), "
            f"got {config.tool_call_parser!r}. R1-Distill family is "
            "Qwen2/Llama2-arch SFT and cannot emit the V3 fenced-JSON "
            "envelope. See Sven r12 dogfood HIGH-1 verdict (R12-S1)."
        )
        assert config.reasoning_parser == "deepseek_r1"

    # DeepSeek V2.5 (and older non-R1) → legacy ``deepseek`` parser.
    # R12-5: V3 vanilla checkpoints now route to ``deepseek_v3`` (the
    # dedicated parser) — see ``test_deepseek_v3_vanilla_routes_to_v3_parser``.
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/DeepSeek-V2.5-4bit",
        ],
    )
    def test_deepseek_no_reasoning(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "deepseek"
        assert config.reasoning_parser is None

    # R12-5: vanilla DeepSeek-V3 checkpoints (V3-0324 etc.) emit the
    # V3 fenced-JSON wire shape, same as R1-0528. Route them to the
    # dedicated ``deepseek_v3`` parser so they get the block-wise
    # scanner hardening and the forced-tool prefix injection (codex
    # round-3 P2 — the generic ``deepseek`` parser has neither).
    @pytest.mark.parametrize(
        "model_path",
        [
            "deepseek-ai/DeepSeek-V3-0324",
            "deepseek-v3-0324",
        ],
    )
    def test_deepseek_v3_vanilla_routes_to_v3_parser(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "deepseek_v3"
        # No ``deepseek_r1`` reasoning on vanilla V3 — only R1 family
        # ships with the thinking-channel.
        assert config.reasoning_parser is None

    # Hermes fine-tuned
    def test_hermes(self):
        config = detect_model_config("mlx-community/Hermes-3-Llama-3.1-8B-4bit")
        assert config is not None
        assert config.tool_call_parser == "hermes"

    # Llama
    def test_llama(self):
        config = detect_model_config("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
        assert config is not None
        assert config.tool_call_parser == "llama"
        assert config.reasoning_parser is None

    # Kimi
    def test_kimi(self):
        config = detect_model_config("mlx-community/Kimi-Linear-48B-A3B-Instruct-6bit")
        assert config is not None
        assert config.tool_call_parser == "kimi"
        assert config.reasoning_parser is None

    # Gemma 3 (non-3n) — text-only family; carries hermes tool format.
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/gemma-3-12b-it-4bit",
        ],
    )
    def test_gemma(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "hermes"
        assert config.reasoning_parser is None

    # Gemma 3n — on-device multimodal (text+image+audio). Chat
    # template defines no tool-call special tokens; pin
    # ``tool_call_parser=None`` so HF-path serves don't advertise tool
    # capability the model can't honour (PR #715 bundle, fuzz
    # finding D).
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/gemma-3n-E2B-it-4bit",
            "lmstudio-community/gemma-3n-E4B-it-MLX-4bit",
            "google/gemma-3n-E2B-it",
        ],
    )
    def test_gemma_3n_no_tool_calls(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.tool_call_parser is None, (
            f"{model_path}: tool_call_parser must be None — Gemma 3n chat "
            f"template carries no tool-call tokens (PR #715 fuzz finding "
            f"D). Did the gemma-3n regex get demoted below the generic "
            f"'gemma' regex? Got {cfg.tool_call_parser!r}."
        )
        assert cfg.reasoning_parser is None

    # Phi-4-mini-instruct (non-reasoning) — the only remaining Phi
    # family member with hermes tool calls. Phi-3.5-mini moved to
    # ``test_phi_3_5_no_tool_calls`` (no tool support) and
    # Phi-4-mini-reasoning has its own test below (deepseek_r1 parser).
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/Phi-4-mini-instruct-4bit",
        ],
    )
    def test_phi(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "hermes"
        assert config.reasoning_parser is None

    # Phi-3.5-mini — chat template defines no ``<tool_call>`` special
    # token; the model ignores tool prompts (PR #715 bundle, fuzz
    # finding D). Pin ``tool_call_parser=None``. The dedicated regex
    # MUST win over the generic ``phi[-_]?[34]`` regex.
    @pytest.mark.parametrize(
        "model_path",
        [
            "microsoft/Phi-3.5-mini-instruct",
            "mlx-community/Phi-3.5-mini-instruct-4bit",
        ],
    )
    def test_phi_3_5_no_tool_calls(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.tool_call_parser is None, (
            f"{model_path}: tool_call_parser must be None — Phi-3.5-mini "
            f"chat template carries no tool tokens (PR #715 fuzz finding "
            f"D). Did the phi-3.5 regex get demoted below the generic "
            f"'phi' regex? Got {cfg.tool_call_parser!r}."
        )
        assert cfg.reasoning_parser is None

    # Phi-4-mini-reasoning — Microsoft's math-tuned reasoning variant.
    # Smoke-verified to emit autonomous ``<think>...</think>`` blocks
    # despite the chat template not injecting one. The dedicated regex
    # MUST win over the generic ``phi[-_]?[34]`` regex so the block
    # lands in ``reasoning_content`` instead of leaking into
    # ``content`` (which is what happens with reasoning_parser=None).
    @pytest.mark.parametrize(
        "model_path",
        [
            "phi-4-mini-reasoning-4bit",
            "lmstudio-community/Phi-4-mini-reasoning-MLX-4bit",
            "microsoft/Phi-4-mini-reasoning",
        ],
    )
    def test_phi_4_mini_reasoning(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.tool_call_parser == "hermes"
        assert cfg.reasoning_parser == "deepseek_r1", (
            f"{model_path}: reasoning_parser must be 'deepseek_r1' — "
            f"Phi-4-mini-reasoning emits `<think>` blocks autonomously, "
            f"smoke-verified. Got {cfg.reasoning_parser!r}. Did the "
            f"phi-4-mini-reasoning regex get demoted below the generic "
            f"phi regex?"
        )

    # Unknown model → None
    def test_unknown_model(self):
        config = detect_model_config("some-random-model-xyz")
        assert config is None

    # Explicit flags override (tested at integration level, but verify None doesn't crash)
    def test_empty_path(self):
        config = detect_model_config("")
        assert config is None


class TestCapabilityGates:
    """Per-arch capability gates: is_hybrid + supports_spec_decode."""

    @pytest.mark.parametrize(
        "model_path",
        [
            # MoE Qwen3.5 — A3B / A10B / generic MoE markers stay hybrid.
            "mlx-community/Qwen3.5-122B-A10B-8bit",
            "mlx-community/Qwen3.5-35B-A3B-4bit",
        ],
    )
    def test_qwen35_moe_hybrid(self, model_path):
        """r6-A R6-C1: only the A3B / A10B / MoE Qwen3.5 variants stamp
        as hybrid in the auto-derivation regex. The dense 4B/9B/27B
        siblings now fall through to the non-hybrid generic Qwen3
        fallback — see ``test_qwen35_dense_not_hybrid`` for the
        complementary guard, and ``aliases.json`` for the matching
        per-alias declaration."""
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    @pytest.mark.parametrize(
        "model_path",
        [
            # Dense Qwen3.5 — the wedge surface. Must NOT be hybrid.
            "mlx-community/Qwen3.5-9B-4bit",
            "/Users/x/.lmstudio/models/Qwen3.5-4B-MLX-4bit",
            "mlx-community/Qwen3.5-27B-4bit",
        ],
    )
    def test_qwen35_dense_not_hybrid(self, model_path):
        """r6-A R6-C1: dense Qwen3.5 paths (no MoE marker) must NOT pick
        up ``is_hybrid=True`` from the auto-derivation regex. Pre-fix,
        the bare ``qwen3\\.5`` regex stamped every match as hybrid which
        wedged the runtime on metal::malloc 499000."""
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is False, (
            f"{model_path} resolved to is_hybrid=True — the regex is still "
            f"stamping dense paths."
        )

    @pytest.mark.parametrize(
        "model_path",
        [
            # MoE Qwen3.6 — A3B markers stay hybrid.
            "mlx-community/Qwen3.6-35B-A3B-4bit",
        ],
    )
    def test_qwen36_moe_hybrid(self, model_path):
        """Same MoE-marker contract as the Qwen3.5 siblings — see
        ``test_qwen35_moe_hybrid``."""
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    @pytest.mark.parametrize(
        "model_path",
        [
            # Dense Qwen3.6 — must NOT be hybrid post-r6-A.
            "mlx-community/Qwen3.6-27B-4bit",
            "unsloth/Qwen3.6-27B-MLX-8bit",
        ],
    )
    def test_qwen36_dense_not_hybrid(self, model_path):
        """r6-A R6-C1 sibling — same wedge story on the Qwen3.6 dense
        27B variants. The MoE-marker gate keeps them off the hybrid
        path."""
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is False

    @pytest.mark.parametrize(
        "model_path",
        [
            "lmstudio-community/Qwen3-Coder-Next-MLX-4bit",
            "mlx-community/Qwen3-Coder-Next-4bit",
            "mlx-community/Qwen3-Next-80B-A3B-Instruct",
        ],
    )
    def test_qwen3_next_hybrid(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    def test_qwopus_hybrid(self):
        cfg = detect_model_config("Jackrong/MLX-Qwopus3.5-27B-v3-4bit")
        assert cfg is not None
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    @pytest.mark.parametrize(
        "model_path",
        [
            "state-spaces/mamba-2.8b",
            "ai21labs/Jamba-v0.1",
            "fla-org/rwkv-7-1.5b",
        ],
    )
    def test_pure_recurrent_hybrid(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/Qwen3-0.6B-8bit",
            "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
            "mlx-community/gemma-3-12b-it-4bit",
            "mlx-community/gpt-oss-20b-MXFP4-Q8",
            "microsoft/Phi-3.5-mini-instruct",
        ],
    )
    def test_pure_attention_supports_spec_decode(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True

    def test_qwen3_coder_legacy_pure_attention(self):
        # The old Qwen3-Coder (without "Next") is pure-attention. The
        # regex order must catch it AFTER Coder-Next so it doesn't get
        # mis-tagged as hybrid.
        cfg = detect_model_config("Qwen/Qwen3-Coder-7B-Instruct")
        assert cfg is not None
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True


class TestEnrichModelConfig:
    """Runtime-probe safety net using the loaded model object."""

    def test_enrich_no_make_cache_method(self):
        # Models without make_cache (e.g. some VLMs) → no probe, no flip.
        class StubModel:
            pass

        cfg = enrich_model_config(None, StubModel())
        # Default config: not hybrid, supports spec decode.
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True

    def test_enrich_arrayscache_flips_to_hybrid(self):
        # Stub the import: any cache element identified as ArraysCache
        # should flip is_hybrid → True.
        from mlx_lm.models.cache import ArraysCache

        class HybridModel:
            def make_cache(self):
                # ArraysCache.__init__ requires no positional args.
                return [ArraysCache(size=1)]

        cfg = enrich_model_config(None, HybridModel())
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    def test_enrich_pure_kvcache_stays_supported(self):
        from mlx_lm.models.cache import KVCache

        class PureModel:
            def make_cache(self):
                return [KVCache()]

        cfg = enrich_model_config(None, PureModel())
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True

    def test_enrich_mixed_cache_still_flags_hybrid(self):
        # Even one ArraysCache in the layer list → hybrid.
        from mlx_lm.models.cache import ArraysCache, KVCache

        class MixedModel:
            def make_cache(self):
                return [KVCache(), ArraysCache(size=1), KVCache()]

        cfg = enrich_model_config(None, MixedModel())
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    def test_enrich_does_not_mutate_input(self):
        # Input cfg must not be modified — return a fresh dataclass.
        from mlx_lm.models.cache import ArraysCache

        class HybridModel:
            def make_cache(self):
                return [ArraysCache(size=1)]

        original = ModelConfig(tool_call_parser="hermes")
        result = enrich_model_config(original, HybridModel())
        assert original.is_hybrid is False  # unchanged
        assert original.supports_spec_decode is True  # unchanged
        assert result.is_hybrid is True  # new instance
        assert result.tool_call_parser == "hermes"  # preserved

    def test_enrich_swallows_probe_errors(self):
        # Probe failures (rare, but possible if model is half-loaded)
        # must not crash engine init.
        class BrokenModel:
            def make_cache(self):
                raise RuntimeError("probe failure")

        cfg = enrich_model_config(None, BrokenModel())
        # Defaults preserved when probe fails.
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True


class TestVisibility:
    """Level 1 / Level 2 / Level 3 visibility helpers."""

    # --- Level 1: one-line summary ---

    def test_summary_for_pure_attention(self):
        cfg = detect_model_config("mlx-community/Qwen3-0.6B-8bit")
        line = format_profile_summary("mlx-community/Qwen3-0.6B-8bit", cfg)
        # Single line, contains key facts
        assert "\n" not in line
        assert "pure attention" in line
        assert "spec decode OK" in line
        assert "throttle OFF" in line
        assert "tool=hermes" in line
        assert "reasoning=qwen3" in line

    def test_summary_for_hybrid(self):
        # r6-A R6-C1: use the A3B MoE Qwen3.5 path — dense ``Qwen3.5-4B``
        # is no longer hybrid (it was the wedge surface). MoE variants
        # remain on the hybrid path and still exercise the
        # ``throttle ON`` / ``spec decode OFF`` summary output.
        cfg = detect_model_config("mlx-community/Qwen3.5-35B-A3B-4bit")
        line = format_profile_summary("mlx-community/Qwen3.5-35B-A3B-4bit", cfg)
        assert "hybrid" in line
        assert "throttle ON" in line
        assert "spec decode OFF" in line

    def test_summary_for_unknown(self):
        line = format_profile_summary("brand-new-model", None)
        assert "unknown family" in line
        assert "brand-new-model" in line

    # --- Level 2 / Level 3: ASCII table ---

    def test_table_renders_aligned(self):
        cfg = detect_model_config("mlx-community/Qwen3.5-4B-MLX-4bit")
        table = format_profile_table("mlx-community/Qwen3.5-4B-MLX-4bit", cfg)
        lines = table.splitlines()
        # Header row + separator + 5 data rows + 2 borders
        assert len(lines) >= 8
        # Each row pipes-out at the same column for alignment
        widths = {len(line) for line in lines if line.startswith(("│", "┌", "└"))}
        assert len(widths) == 1, (
            f"All rows must be same printable width, got: {widths}\n{table}"
        )

    def test_table_for_hybrid_shows_disabled_spec(self):
        # r6-A R6-C1: use the A3B MoE Qwen3.5 path — see
        # ``test_summary_for_hybrid`` for the dense-vs-MoE rationale.
        cfg = detect_model_config("mlx-community/Qwen3.5-35B-A3B-4bit")
        table = format_profile_table("mlx-community/Qwen3.5-35B-A3B-4bit", cfg)
        assert "✗ disabled (hybrid arch)" in table
        assert "✓ 200ms gap" in table

    def test_table_for_pure_attention_shows_supported(self):
        cfg = detect_model_config("mlx-community/Qwen3-0.6B-8bit")
        table = format_profile_table("mlx-community/Qwen3-0.6B-8bit", cfg)
        assert "✓ supported" in table
        assert "✗ not needed" in table

    def test_table_for_unknown_shows_defaults(self):
        table = format_profile_table("some-new-model", None)
        assert "no pattern matched" in table

    def test_table_truncates_long_path(self):
        long = "very/long/path/" + "x" * 200
        table = format_profile_table(long, None)
        # Header line must fit inside the box border
        for line in table.splitlines():
            assert len(line) <= 80, f"line too wide: {line!r}"


class TestGetProfile:
    """``get_profile()`` is the public one-shot API."""

    def test_get_profile_without_model(self):
        # r6-A R6-C1: ``Qwen3.5-4B-MLX-4bit`` is now non-hybrid (the
        # wedge surface). Use the A3B MoE variant so this guard
        # continues exercising the hybrid-arch ``supports_spec_decode``
        # gate; ``test_dense_qwen35_get_profile_not_hybrid`` covers the
        # complementary contract on the dense path.
        cfg = get_profile("mlx-community/Qwen3.5-35B-A3B-4bit")
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    def test_dense_qwen35_get_profile_not_hybrid(self):
        """r6-A R6-C1 complementary guard: ``get_profile`` on a dense
        Qwen3.5 path must return ``is_hybrid=False`` — the alias
        profile pins it via ``is_hybrid_explicit=True`` and the
        regex fallback respects the MoE-marker gate."""
        cfg = get_profile("mlx-community/Qwen3.5-4B-MLX-4bit")
        assert cfg.is_hybrid is False
        # supports_spec_decode is forced off because the underlying
        # qwen3_5 architecture still uses linear-attention layers and
        # spec decode isn't safe — but the routing decision (hybrid
        # throttle + prefix-boundary snapshot) is off.
        assert cfg.supports_spec_decode is False

    def test_get_profile_unknown_returns_defaults(self):
        cfg = get_profile("brand-new-model-xyz")
        # Never returns None — falls back to default ModelConfig.
        assert isinstance(cfg, ModelConfig)
        assert cfg.is_hybrid is False
        assert cfg.supports_spec_decode is True

    def test_get_profile_with_model_runs_enrichment(self):
        from mlx_lm.models.cache import ArraysCache

        class HybridStub:
            def make_cache(self):
                return [ArraysCache(size=1)]

        # Even a model name we don't know about gets flipped to hybrid
        # via runtime probe.
        cfg = get_profile("mystery-model", HybridStub())
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False


class TestWarnMisboundDeepseekV3Parser:
    """R12-S1: cover the misbind-warning helper added in response to
    Sven r12 dogfood HIGH-1. The auto-detect path already routes the
    R1-distill family to the legacy ``deepseek`` parser (covered by
    ``test_deepseek_r1_distill_routes_to_legacy_parser`` above); this
    helper exists so an EXPLICIT override (``--tool-call-parser
    deepseek_v3``) on a non-V3 model surfaces a loud startup warning
    instead of silently shipping ``arguments="{}"`` tool calls.
    """

    # No warning when no parser is bound at all.
    def test_no_warn_when_parser_unset(self):
        assert (
            warn_misbound_deepseek_v3_parser(
                "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit", None
            )
            is None
        )

    # No warning for non-V3 parsers (hermes, deepseek, llama, etc.) —
    # the helper only fires on the V3-family parsers it knows about.
    @pytest.mark.parametrize(
        "parser",
        [
            "hermes",
            "deepseek",
            "deepseek_r1",
            "llama",
            "mistral",
            "kimi",
            "auto",
        ],
    )
    def test_no_warn_for_non_v3_family_parsers(self, parser):
        # Even on a model whose name would otherwise look suspicious to a
        # naive substring matcher, the helper should not warn unless the
        # bound parser is in the V3 family.
        assert (
            warn_misbound_deepseek_v3_parser(
                "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit", parser
            )
            is None
        )

    # In-spec: V3-line checkpoints (V3 / R1-0528 / V4 / V5) emit the V3
    # fenced-JSON body, matched by deepseek_v3 / deepseek_r1_0528.
    # V3.1-line checkpoints emit the V3.1 plain-JSON body, matched by
    # deepseek_v31. Matching combos warn nothing.
    @pytest.mark.parametrize(
        "model_path,parser",
        [
            ("mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit", "deepseek_v3"),
            ("mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit", "deepseek_r1_0528"),
            ("deepseek-ai/DeepSeek-V3-0324", "deepseek_v3"),
            ("mlx-community/DeepSeek-V3-0324-4bit", "deepseek_v3"),
            ("deepseek-ai/DeepSeek-V3.1-0324", "deepseek_v31"),
            ("mlx-community/DeepSeek-V3.1-MLX-4bit", "deepseek_v31"),
            # Forward-cover V4 / V5 — V3 template lineage per upstream
            # V4 card. The helper should accept these without warning.
            ("deepseek-ai/DeepSeek-V4", "deepseek_v3"),
            ("mlx-community/DeepSeek-V5-MLX-4bit", "deepseek_v3"),
        ],
    )
    def test_no_warn_for_matching_sub_family(self, model_path, parser):
        assert warn_misbound_deepseek_v3_parser(model_path, parser) is None

    # Cross-sub-family misbind (codex r1 P2): V3.1 model + V3 parser, or
    # V3 model + V3.1 parser. Both ends sit inside the V3 template
    # lineage so the outer envelope matches, but the per-block body
    # shapes are incompatible — same empty-args failure as the
    # out-of-lineage case. The warning MUST fire here too.
    @pytest.mark.parametrize(
        "model_path,parser",
        [
            # V3.1 model + V3 body parser → blocks dropped, empty args.
            ("deepseek-ai/DeepSeek-V3.1-0324", "deepseek_v3"),
            ("mlx-community/DeepSeek-V3.1-MLX-4bit", "deepseek_v3"),
            ("deepseek-ai/DeepSeek-V3.1-0324", "deepseek_r1_0528"),
            # V3-line model + V3.1 body parser → same.
            ("mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit", "deepseek_v31"),
            ("deepseek-ai/DeepSeek-V3-0324", "deepseek_v31"),
            ("deepseek-ai/DeepSeek-V4", "deepseek_v31"),
        ],
    )
    def test_warn_on_cross_sub_family_misbind(self, model_path, parser):
        msg = warn_misbound_deepseek_v3_parser(model_path, parser)
        assert msg is not None, (
            f"cross-sub-family misbind {parser} on {model_path} must warn"
        )
        # Must name the offending flag, the model, and reference the
        # body-shape mismatch so the user knows it's not the same as the
        # out-of-lineage case.
        assert parser in msg
        assert model_path in msg
        # Cross-sub-family messages mention the chat-template family
        # (V3.0 or V3.1) rather than the generic "non-V3" framing used
        # by the out-of-lineage warning.
        assert ("V3.0" in msg) or ("V3.1" in msg)

    # The Sven HIGH-1 repro: forcing deepseek_v3 on an R1-Distill (Qwen2
    # arch) MUST surface a warning. Covers every common distill flavour
    # (1.5B, 7B, 14B, 32B Qwen + 8B/70B Llama) and every V3-family
    # parser alias the user might pick.
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
            "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ],
    )
    @pytest.mark.parametrize(
        "parser",
        ["deepseek_v3", "deepseek_v31", "deepseek_r1_0528"],
    )
    def test_warn_on_r1_distill_misbind(self, model_path, parser):
        msg = warn_misbound_deepseek_v3_parser(model_path, parser)
        assert msg is not None
        # Must name the offending flag and the model so the user can
        # locate the misconfiguration without grepping the source.
        assert parser in msg
        assert model_path in msg
        # Must point at the auto-detected suggestion (``deepseek`` for
        # the R1-distill family) so the fix is one flag-flip away.
        assert "deepseek" in msg
        # Must mention the actionable remediation — drop the flag or
        # switch to hermes for Qwen/Llama-arch SFTs.
        assert "auto-detect" in msg.lower() or "hermes" in msg.lower()

    # Also warn on V2.x and bare Qwen/Llama paths that someone might
    # explicitly try to coerce to the V3 parser.
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/DeepSeek-V2.5-4bit",
            "mlx-community/Qwen3-0.6B-MLX-4bit",
            "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        ],
    )
    def test_warn_on_non_v3_explicit_override(self, model_path):
        msg = warn_misbound_deepseek_v3_parser(model_path, "deepseek_v3")
        assert msg is not None
        assert model_path in msg

    # Unknown models (no regex match) still get a warning when bound to
    # a V3-family parser, but the message degrades gracefully (no
    # auto-detect suggestion).
    def test_warn_on_unknown_model_no_suggestion(self):
        msg = warn_misbound_deepseek_v3_parser(
            "brand-new-org/MysteryModel-2026-7B", "deepseek_v3"
        )
        assert msg is not None
        # No `auto-detect would pick 'X'` blurb when there's no match.
        # The actionable nudge is still present (drop the flag).
        assert "deepseek_v3" in msg

    # Codex round-2 P2 regression: when the user serves a built-in
    # alias (``deepseek-r1-8b-4bit``) whose name itself does not carry
    # the ``0528`` / ``v3`` marker but whose ``hf_path`` resolves to a
    # V3-template checkpoint, the bare-text classifier returns ``None``
    # — and the auto-detect path sets ``args.tool_call_parser`` to
    # ``deepseek_v3``. Without resolving the alias here, the misbind
    # warning fires falsely on a perfectly correct default serve. The
    # helper MUST resolve aliases before classifying.
    def test_no_warn_on_v3_alias_with_v3_parser(self):
        # The alias name has no V3 marker — only the resolved HF path
        # (``mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit``) does.
        # The helper must do the alias lookup itself.
        assert (
            warn_misbound_deepseek_v3_parser("deepseek-r1-8b-4bit", "deepseek_v3")
            is None
        )
        # The matching ``deepseek_r1_0528`` alias on the same parser
        # family is also in-spec.
        assert (
            warn_misbound_deepseek_v3_parser("deepseek-r1-8b-4bit", "deepseek_r1_0528")
            is None
        )

    # Conversely, when the user serves the same V3 alias with a V3.1
    # parser (cross-sub-family), the warning MUST still fire — the alias
    # resolution is informational, not a free pass.
    def test_warn_on_v3_alias_with_v31_parser(self):
        msg = warn_misbound_deepseek_v3_parser("deepseek-r1-8b-4bit", "deepseek_v31")
        assert msg is not None
        # Cross-sub-family framing: the helper recognised the alias as
        # a V3.0 (not V3.1) checkpoint.
        assert "V3.0" in msg

    # Codex round-3 P3 regression: when a user serves a legitimate V3 or
    # V3.1 checkpoint from a local path whose PARENT DIRECTORY contains
    # ``distill`` / ``distillations`` / ``distilled``, the substring
    # check used to short-circuit on the full path and falsely classify
    # the model as a non-V3 checkpoint — emitting a misbind warning on
    # a perfectly correct serve. The distill reject must be scoped to
    # the model-name component (last path segment) only.
    @pytest.mark.parametrize(
        "model_path,parser",
        [
            (
                "/models/distillations/deepseek-ai/DeepSeek-V3.1-0324",
                "deepseek_v31",
            ),
            ("/tmp/distilled/DeepSeek-V3-0324", "deepseek_v3"),
            (
                "/home/me/distilled-set/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                "deepseek_v3",
            ),
        ],
    )
    def test_no_warn_when_parent_dir_only_contains_distill(self, model_path, parser):
        # Tail segment is a real V3 / V3.1 checkpoint name; the parent
        # ``distill*`` directory must NOT trip the rejection gate.
        assert warn_misbound_deepseek_v3_parser(model_path, parser) is None

    # Counterpart: when the model-name component itself carries
    # ``distill`` (the actual R1-Distill family), the rejection still
    # fires regardless of how deep the parent path goes.
    @pytest.mark.parametrize(
        "model_path",
        [
            "/home/me/random/dir/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            "/var/cache/mlx-community/DeepSeek-R1-Distill-Llama-8B",
        ],
    )
    def test_warn_when_name_component_contains_distill(self, model_path):
        msg = warn_misbound_deepseek_v3_parser(model_path, "deepseek_v3")
        assert msg is not None
        assert model_path in msg

    # Codex round-4 BLOCKING regression: the r3 fix scoped the
    # ``distill`` REJECT to the tail segment but left the V3 / V3.1 /
    # R1-0528 / V4-V5 POSITIVE classifiers running against the full
    # path. That falsely classified a non-V3 checkpoint as v3/v31 when
    # an unrelated parent directory carried the family marker — e.g.
    # ``/models/DeepSeek-V3/qwen-model`` resolved to ``"v3"`` and
    # SUPPRESSED the misbind warning even with the wrong parser. All
    # classifiers must scope to the model-name component.
    @pytest.mark.parametrize(
        "model_path,parser",
        [
            ("/models/DeepSeek-V3/qwen-model", "deepseek_v3"),
            ("/models/DeepSeek-V3.1/qwen-model", "deepseek_v31"),
            ("/models/DeepSeek-R1-0528/random-model", "deepseek_v3"),
            ("/models/DeepSeek-V4/qwen-model", "deepseek_v3"),
        ],
    )
    def test_warn_when_only_parent_dir_carries_v3_marker(self, model_path, parser):
        # The model-NAME component is a non-V3 checkpoint (``qwen-model``
        # / ``random-model``). The misbind warning must fire even when
        # the parent dir is named after a V3 family — the parent dir
        # tells us nothing about the actual checkpoint.
        msg = warn_misbound_deepseek_v3_parser(model_path, parser)
        assert msg is not None, (
            f"non-V3 checkpoint under V3-named parent ({model_path!r}) "
            f"with {parser!r} must still warn — classifier must scope to "
            "the model-name component, not the full path."
        )

    # Codex round-5 P2 regression: when auto-detect would ALSO pick a
    # V3-family parser for the same path (because its regex still runs
    # against the full path and the parent dir matches the family),
    # surfacing "Auto-detect would pick 'deepseek_v3'" + "Drop the flag
    # to let auto-detect choose" creates a contradiction — auto-detect
    # would silently keep the same wrong parser. The warning must
    # suppress the auto-detect suggestion in that case AND switch the
    # remediation to "Pass --tool-call-parser hermes" (since dropping
    # the flag wouldn't help).
    def test_no_contradiction_when_auto_detect_also_misclassifies(self):
        msg = warn_misbound_deepseek_v3_parser(
            "/models/DeepSeek-V3/qwen-model", "deepseek_v3"
        )
        assert msg is not None
        # No "Auto-detect would pick 'deepseek_v3'" — suppressed.
        assert "Auto-detect would pick" not in msg
        # No "drop the explicit ... flag to let auto-detect choose" —
        # would silently keep the wrong parser. The replacement
        # remediation pins to hermes.
        assert "Drop the explicit" not in msg
        assert "hermes" in msg

    # Sanity check that the suggestion path still fires for the common
    # case (non-V3-named parent, R1-Distill family, auto suggests the
    # legacy ``deepseek`` parser).
    def test_suggestion_still_fires_when_auto_picks_non_v3(self):
        msg = warn_misbound_deepseek_v3_parser(
            "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit", "deepseek_v3"
        )
        assert msg is not None
        assert "Auto-detect would pick 'deepseek'" in msg
        assert "Drop the explicit" in msg
