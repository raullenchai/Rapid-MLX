"""Tests for model auto-config detection."""

import pytest

from vllm_mlx.model_auto_config import detect_model_config


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

    @pytest.mark.parametrize(
        "model_path",
        [
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "samuelfaj/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
            "/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
        ],
    )
    def test_qwen36_uses_coder_xml_tools_and_qwen3_reasoning(self, model_path):
        config = detect_model_config(model_path)
        assert config is not None
        assert config.tool_call_parser == "qwen3_coder_xml"
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
