"""Tests for model auto-config detection."""

import pytest

from vllm_mlx.model_auto_config import (
    ModelConfig,
    detect_model_config,
    enrich_model_config,
    format_profile_summary,
    format_profile_table,
    get_profile,
)


class TestDetectModelConfig:
    """Test detect_model_config with various model paths."""

    # Qwen family (non-Coder)
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/Qwen3.5-9B-4bit",
            "mlx-community/Qwen3-0.6B-MLX-4bit",
            "/Users/someone/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-8bit",
        ],
    )
    def test_qwen_family(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "hermes"
        assert config.reasoning_parser == "qwen3"

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

    # DeepSeek V3.1 / R1-0528 → deepseek_v31 parser
    @pytest.mark.parametrize(
        "model_path",
        [
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "deepseek-ai/DeepSeek-V3.1-0324",
            "mlx-community/DeepSeek-R1-0528-4bit",
        ],
    )
    def test_deepseek_v31(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "deepseek_v31"
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
        assert config.reasoning_parser is None
        assert config.is_hybrid is False
        assert config.supports_spec_decode is True

    # ---- 2026 model families ----

    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/granite-4.0-h-small-4bit",
            "mlx-community/granite-4.0-h-tiny-4bit",
            "ibm-granite/granite-4.0-h-small",
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

    # DeepSeek non-R1 (V3, V2.5) → deepseek parser, no reasoning
    @pytest.mark.parametrize(
        "model_path",
        [
            "deepseek-v3-0324",
            "mlx-community/DeepSeek-V2.5-4bit",
        ],
    )
    def test_deepseek_no_reasoning(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "deepseek"
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

    # Gemma
    def test_gemma(self):
        config = detect_model_config("mlx-community/gemma-3-12b-it-4bit")
        assert config is not None
        assert config.tool_call_parser == "hermes"
        assert config.reasoning_parser is None

    # Phi
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/Phi-4-mini-instruct-4bit",
            "microsoft/Phi-3.5-mini-instruct",
        ],
    )
    def test_phi(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "hermes"
        assert config.reasoning_parser is None

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
            "mlx-community/Qwen3.5-9B-4bit",
            "mlx-community/Qwen3.5-122B-A10B-8bit",
            "/Users/x/.lmstudio/models/Qwen3.5-4B-MLX-4bit",
        ],
    )
    def test_qwen35_hybrid(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/Qwen3.6-27B-4bit",
            "unsloth/Qwen3.6-27B-MLX-8bit",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
        ],
    )
    def test_qwen36_hybrid(self, model_path):
        cfg = detect_model_config(model_path)
        assert cfg is not None
        assert cfg.is_hybrid is True
        assert cfg.supports_spec_decode is False

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
        cfg = detect_model_config("mlx-community/Qwen3.5-4B-MLX-4bit")
        line = format_profile_summary("mlx-community/Qwen3.5-4B-MLX-4bit", cfg)
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
        cfg = detect_model_config("mlx-community/Qwen3.5-4B-MLX-4bit")
        table = format_profile_table("mlx-community/Qwen3.5-4B-MLX-4bit", cfg)
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
        cfg = get_profile("mlx-community/Qwen3.5-4B-MLX-4bit")
        assert cfg.is_hybrid is True
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
