"""Tests for model auto-config detection."""

import pytest

from vllm_mlx.model_auto_config import ModelConfig, detect_model_config


class TestDetectModelConfig:
    """Test detect_model_config with various model paths."""

    # Qwen family
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/Qwen3.5-9B-4bit",
            "mlx-community/Qwen3-0.6B-MLX-4bit",
            "/Users/someone/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-8bit",
            "Qwen3-Coder-Next-MLX-4bit",
            "lmstudio-community/Qwen3-Coder-Next-MLX-6bit",
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

    # DeepSeek
    def test_deepseek(self):
        config = detect_model_config("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
        config2 = detect_model_config("deepseek-v3-0324")
        # DeepSeek should match deepseek pattern, not qwen
        assert config is not None
        assert config.reasoning_parser == "deepseek_r1"
        assert config2 is not None
        assert config2.tool_call_parser == "hermes"

    # Hermes fine-tuned
    def test_hermes(self):
        config = detect_model_config("mlx-community/Hermes-3-Llama-3.1-8B-4bit")
        assert config is not None
        assert config.tool_call_parser == "hermes"

    # Unknown model → None
    def test_unknown_model(self):
        config = detect_model_config("some-random-model-xyz")
        assert config is None

    # Explicit flags override (tested at integration level, but verify None doesn't crash)
    def test_empty_path(self):
        config = detect_model_config("")
        assert config is None
