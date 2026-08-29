# SPDX-License-Identifier: Apache-2.0
"""
Tests for API utility functions.

Tests clean_output_text, is_mllm_model, and extract_multimodal_content
from vllm_mlx/api/utils.py. No MLX dependency.
"""

import json
from pathlib import Path

import pytest

from vllm_mlx.api.models import ContentPart, ImageUrl, Message
from vllm_mlx.api.utils import (
    MLLM_PATTERNS,
    SPECIAL_TOKENS_PATTERN,
    _check_legacy_string_patterns,
    _config_indicates_vlm,
    _content_to_text,
    _local_checkpoint_has_multimodal_weights,
    _try_read_config_json,
    clean_output_text,
    extract_multimodal_content,
    is_mllm_model,
    is_vlm_model,
    validate_content_blocks_for_capabilities,
)


class TestCleanOutputText:
    """Tests for clean_output_text function."""

    def test_empty_string(self):
        assert clean_output_text("") == ""

    def test_none_returns_none(self):
        assert clean_output_text(None) is None

    def test_plain_text_unchanged(self):
        assert clean_output_text("Hello world") == "Hello world"

    def test_removes_im_end(self):
        assert clean_output_text("Hello<|im_end|>") == "Hello"

    def test_removes_im_start(self):
        assert clean_output_text("<|im_start|>Hello") == "Hello"

    def test_removes_endoftext(self):
        assert clean_output_text("Hello<|endoftext|>") == "Hello"

    def test_removes_eot_id(self):
        assert clean_output_text("Hello<|eot_id|>") == "Hello"

    def test_removes_end_token(self):
        assert clean_output_text("Hello<|end|>") == "Hello"

    def test_removes_start_header_id(self):
        result = clean_output_text("<|start_header_id|>assistant<|end_header_id|>Hello")
        assert "<|start_header_id|>" not in result
        assert "<|end_header_id|>" not in result

    def test_removes_s_tags(self):
        assert clean_output_text("<s>Hello</s>") == "Hello"

    def test_removes_pad_tokens(self):
        assert clean_output_text("[PAD]Hello[PAD]") == "Hello"

    def test_removes_sep_cls(self):
        assert clean_output_text("[CLS]Hello[SEP]") == "Hello"

    def test_removes_multiple_special_tokens(self):
        text = "<|im_start|>assistant\nHello world<|im_end|><|endoftext|>"
        result = clean_output_text(text)
        assert result == "assistant\nHello world"

    def test_preserves_think_tags(self):
        text = "<think>Let me think about this.</think>The answer is 42."
        result = clean_output_text(text)
        assert "<think>" in result
        assert "</think>" in result
        assert "The answer is 42." in result

    def test_adds_missing_opening_think_tag(self):
        text = "Some thinking content.</think>The answer is 42."
        result = clean_output_text(text)
        assert result.startswith("<think>")
        assert "</think>" in result

    def test_no_extra_think_tag_when_already_present(self):
        text = "<think>Thinking.</think>Answer."
        result = clean_output_text(text)
        assert result.count("<think>") == 1

    def test_strips_whitespace(self):
        assert clean_output_text("  Hello  ") == "Hello"

    def test_combined_special_tokens_and_think(self):
        text = "<|im_start|><think>I need to think.</think>42<|im_end|>"
        result = clean_output_text(text)
        assert "<think>" in result
        assert "</think>" in result
        assert "42" in result
        assert "<|im_start|>" not in result


class TestSpecialTokensPattern:
    """Tests for the special tokens regex pattern."""

    def test_matches_all_expected_tokens(self):
        tokens = [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<|end|>",
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "</s>",
            "<s>",
            "<pad>",
            "[PAD]",
            "[SEP]",
            "[CLS]",
        ]
        for token in tokens:
            assert SPECIAL_TOKENS_PATTERN.search(token) is not None, (
                f"Pattern should match {token}"
            )

    def test_does_not_match_think_tags(self):
        assert SPECIAL_TOKENS_PATTERN.search("<think>") is None
        assert SPECIAL_TOKENS_PATTERN.search("</think>") is None

    def test_does_not_match_normal_text(self):
        assert SPECIAL_TOKENS_PATTERN.search("Hello world") is None


class TestIsMllmModel:
    """Tests for is_mllm_model function."""

    def test_qwen_vl_models(self):
        assert is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit") is True
        assert is_mllm_model("mlx-community/Qwen2-VL-7B-Instruct-4bit") is True

    def test_llava_models(self):
        assert is_mllm_model("mlx-community/llava-1.5-7b-4bit") is True
        assert is_mllm_model("mlx-community/LLaVA-NeXT-7b") is True

    def test_idefics_models(self):
        assert is_mllm_model("mlx-community/Idefics3-8B-Llama3-4bit") is True
        assert is_mllm_model("mlx-community/idefics2-8b-4bit") is True

    def test_paligemma_models(self):
        assert is_mllm_model("mlx-community/paligemma2-3b-mix-224-4bit") is True
        assert is_mllm_model("mlx-community/PaliGemma-3b-mix") is True

    def test_gemma3_models(self):
        assert is_mllm_model("mlx-community/gemma-3-12b-it-4bit") is True
        assert is_mllm_model("mlx-community/gemma3-4b-it-4bit") is True

    def test_medgemma_models(self):
        assert is_mllm_model("mlx-community/MedGemma-4b-it-4bit") is True
        assert is_mllm_model("mlx-community/medgemma-4b") is True

    def test_pixtral_models(self):
        assert is_mllm_model("mlx-community/pixtral-12b-4bit") is True
        assert is_mllm_model("mlx-community/Pixtral-12b-8bit") is True

    def test_molmo_models(self):
        assert is_mllm_model("mlx-community/Molmo-7B-D-0924-4bit") is True
        assert is_mllm_model("mlx-community/molmo-7b") is True

    def test_phi3_vision(self):
        assert is_mllm_model("mlx-community/phi3-vision-128k") is True
        assert is_mllm_model("mlx-community/phi-3-vision-128k-instruct-4bit") is True

    def test_cogvlm(self):
        assert is_mllm_model("mlx-community/CogVLM-chat-hf") is True
        assert is_mllm_model("mlx-community/cogvlm-chat-hf") is True

    def test_internvl(self):
        assert is_mllm_model("mlx-community/InternVL2-8B") is True

    def test_deepseek_vl(self):
        assert is_mllm_model("mlx-community/deepseek-vl-7b-chat-4bit") is True
        assert is_mllm_model("mlx-community/DeepSeek-VL2-small-4bit") is True

    def test_non_mllm_models(self):
        assert is_mllm_model("mlx-community/Llama-3.2-3B-Instruct-4bit") is False
        assert is_mllm_model("mlx-community/Qwen3-8B-4bit") is False
        assert is_mllm_model("mlx-community/Mistral-7B-Instruct-v0.3-4bit") is False
        assert is_mllm_model("mlx-community/DeepSeek-R1-Distill-Qwen-7B") is False

    def test_case_insensitive(self):
        assert is_mllm_model("LLAVA-7B") is True
        assert is_mllm_model("pixtral-12b") is True

    def test_backwards_compatibility_alias(self):
        assert is_vlm_model is is_mllm_model

    def test_all_patterns_defined(self):
        assert len(MLLM_PATTERNS) > 20


class TestIsMllmModelConfigPriority:
    """Tests that config.json takes priority over the legacy substring matcher.

    Regression coverage for issue #516: a local path with a triggering
    substring (e.g. "vl-") used to be misrouted to the MLLM loader even when
    the model itself was text-only. With config.json inspection in front,
    the model's own metadata wins.
    """

    @staticmethod
    def _write_config(tmp_path, name, payload):
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(payload))
        return model_dir

    def test_text_only_config_overrides_triggering_path(self, tmp_path):
        # Path basename contains "vl-" which would match the legacy pattern,
        # but the model declares itself as text-only via config.json.
        model_dir = self._write_config(
            tmp_path,
            "qwen3-vl-derived",
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        )
        assert _check_legacy_string_patterns(str(model_dir)) is True
        assert is_mllm_model(str(model_dir)) is False

    def test_vlm_config_under_neutral_path(self, tmp_path):
        model_dir = self._write_config(
            tmp_path,
            "my-model",
            {
                "model_type": "qwen2_vl",
                "architectures": ["Qwen2VLForConditionalGeneration"],
            },
        )
        assert _check_legacy_string_patterns(str(model_dir)) is False
        assert is_mllm_model(str(model_dir)) is True

    def test_vision_config_key_indicates_vlm(self, tmp_path):
        model_dir = self._write_config(
            tmp_path,
            "exotic-vlm",
            {"model_type": "custom", "vision_config": {"hidden_size": 768}},
        )
        assert is_mllm_model(str(model_dir)) is True

    def test_audio_config_key_indicates_vlm(self, tmp_path):
        model_dir = self._write_config(
            tmp_path,
            "audio-model",
            {"model_type": "custom_audio", "audio_config": {"sample_rate": 16000}},
        )
        assert is_mllm_model(str(model_dir)) is True

    def test_missing_config_falls_back_to_string_matcher(self, tmp_path):
        # No config.json present, so detection falls back to the legacy
        # matcher against the input string.
        model_dir = tmp_path / "Qwen3-VL-7B"
        model_dir.mkdir()
        assert is_mllm_model(str(model_dir)) is True

    def test_malformed_config_falls_back_gracefully(self, tmp_path):
        model_dir = tmp_path / "broken-vl-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{not valid json")
        # Falls back to legacy matcher; basename has "vl-" → True.
        assert is_mllm_model(str(model_dir)) is True

    def test_oversized_config_falls_back_gracefully(self, tmp_path):
        model_dir = tmp_path / "huge-config-vl-model"
        model_dir.mkdir()
        # 2 MB of payload exceeds the 1 MB cap.
        (model_dir / "config.json").write_text("x" * (2 * 1024 * 1024))
        assert is_mllm_model(str(model_dir)) is True

    def test_hf_repo_id_uses_legacy_matcher(self):
        # HF repo IDs are not local dirs, so config inspection short-circuits
        # and the legacy matcher decides. This preserves prior behaviour.
        assert is_mllm_model("Qwen/Qwen3-32B") is False
        assert is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit") is True

    def test_try_read_config_json_returns_none_for_repo_id(self):
        assert _try_read_config_json("Qwen/Qwen3-32B") is None

    def test_try_read_config_json_returns_none_for_missing_dir(self, tmp_path):
        assert _try_read_config_json(str(tmp_path / "does-not-exist")) is None

    def test_config_indicates_vlm_recognises_llava(self):
        assert (
            _config_indicates_vlm({"architectures": ["LlavaForConditionalGeneration"]})
            is True
        )

    def test_config_indicates_vlm_rejects_text_only_qwen3(self):
        assert _config_indicates_vlm({"architectures": ["Qwen3ForCausalLM"]}) is False

    def test_config_indicates_vlm_handles_missing_architectures(self):
        assert _config_indicates_vlm({"model_type": "qwen3"}) is False

    def test_config_indicates_vlm_handles_non_list_architectures(self):
        assert _config_indicates_vlm({"architectures": "Qwen3ForCausalLM"}) is False


class TestIsMllmModelWeightsPresenceOverride:
    """Verifies the local weights-presence override for text-only forks of
    multimodal architectures.

    Regression coverage for issue #393: ``Qwen3.6-35B-A3B-MLX-8bit`` ships
    ``config.json`` with a populated ``vision_config`` block (because the
    base ``Qwen3_5MoeForConditionalGeneration`` architecture is multimodal-
    capable), but the user's safetensors checkpoint only contains language
    tensors — no ``vision_tower.*`` weights. Detection used to trust the
    config and route the model into the MLLM batched engine, which then
    crashed at first request because there was no vision tower to call.

    The fix: when config says VLM AND the path is a local directory with
    a ``model.safetensors.index.json``, scan the index for actual
    multimodal tensor prefixes. If none are present, override to text.
    """

    @staticmethod
    def _make_model_dir(tmp_path, name, config, weight_names):
        """Build a fake local model dir with config.json + sharded index.

        ``weight_names`` is the list of tensor names to embed in the
        ``weight_map`` of ``model.safetensors.index.json`` — controls
        whether the weights-presence override fires.
        """
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(config))
        weight_map = {name: "model-00001-of-00001.safetensors" for name in weight_names}
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )
        return model_dir

    def test_vision_config_but_no_vision_weights_is_text_only(self, tmp_path):
        """The #393 fix path. Config declares vision_config but the index
        has only language tensors — must return False (route as text)."""
        model_dir = self._make_model_dir(
            tmp_path,
            "Qwen3.6-35B-A3B-MLX-8bit",
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152, "depth": 27},
                "image_token_id": 151655,
            },
            weight_names=[
                "language_model.lm_head.weight",
                "language_model.model.embed_tokens.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
            ],
        )
        assert is_mllm_model(str(model_dir)) is False

    def test_vision_config_with_vision_weights_is_vlm(self, tmp_path):
        """Same config shape as above, but the index DOES carry
        vision_tower tensors — must remain True (route as VLM). Mirrors
        the genuine mlx-community/Qwen3.5-35B-A3B-8bit shape."""
        model_dir = self._make_model_dir(
            tmp_path,
            "Qwen3.5-35B-A3B-8bit",
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152, "depth": 27},
            },
            weight_names=[
                "language_model.lm_head.weight",
                "vision_tower.blocks.0.attn.qkv.weight",
                "vision_tower.blocks.0.mlp.linear_fc1.weight",
            ],
        )
        assert is_mllm_model(str(model_dir)) is True

    def test_vision_config_with_unrecognised_vision_namespace_stays_vlm(self, tmp_path):
        """An unknown encoder prefix is insufficient evidence to force text."""
        model_dir = self._make_model_dir(
            tmp_path,
            "repackaged-vlm",
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152},
            },
            weight_names=[
                "language_model.lm_head.weight",
                "vision_encoder.blocks.0.weight",
            ],
        )

        assert is_mllm_model(str(model_dir)) is True

    def test_audio_only_checkpoint_with_audio_weights(self, tmp_path):
        """Same principle but for the audio branch (audio_tower prefix)."""
        model_dir = self._make_model_dir(
            tmp_path,
            "some-audio-vlm",
            {"model_type": "custom_audio", "audio_config": {"sample_rate": 16000}},
            weight_names=[
                "language_model.lm_head.weight",
                "audio_tower.encoder.layer.0.weight",
            ],
        )
        assert is_mllm_model(str(model_dir)) is True

    def test_missing_index_falls_back_to_config(self, tmp_path):
        """Single-file safetensors (no sharded index) → the
        weights-presence probe returns None, and we trust the config.
        This is the conservative side: we'd rather route a small VLM
        correctly than risk wrong-False on a real one. The cost of a
        wrong-True (text path errors clearly at request time) is much
        less than a wrong-False (silent corruption)."""
        model_dir = tmp_path / "Qwen-VL-tiny"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen2VLForConditionalGeneration"],
                    "vision_config": {"hidden_size": 768},
                }
            )
        )
        # No index file.
        assert is_mllm_model(str(model_dir)) is True

    def test_unreadable_index_falls_back_to_config(self, tmp_path):
        model_dir = tmp_path / "broken-index"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["LlavaForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                }
            )
        )
        (model_dir / "model.safetensors.index.json").write_text("{ not json")
        assert is_mllm_model(str(model_dir)) is True

    def test_text_only_config_skips_weights_probe(self, tmp_path):
        """If config says text-only, the weights probe must not fire —
        irrelevant tensor names shouldn't suddenly promote a model to
        MLLM routing. Preserves the existing text-routing decision."""
        model_dir = self._make_model_dir(
            tmp_path,
            "qwen3-text",
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
            # Even if the dir somehow had vision-named tensors (shouldn't
            # happen with a text-only config, but defensive), config wins.
            weight_names=["language_model.lm_head.weight", "vision_tower.fake"],
        )
        assert is_mllm_model(str(model_dir)) is False

    def test_non_qwen_text_only_fork_is_text_architecture_agnostic(self, tmp_path):
        """Round-5 REGRESSION repro (the origin/main regression).

        A text-only fork of a NON-Qwen VLM architecture (Gemma3) — config
        declares ``vision_config`` but the weight index ships only language
        tensors — must route as TEXT (``False``).  The round-3/4 code returned
        ``None`` here (because the text-only detector recognised ONLY
        Qwen3.5-MoE), and the local-dir fallback then force-routed it to the
        MLLM engine, crashing on a missing vision tower.  The architecture-
        agnostic weight-evidence rule (no vision tensors → text) restores
        origin/main for EVERY architecture, not just Qwen.
        """
        model_dir = self._make_model_dir(
            tmp_path,
            "Gemma3-Research-Text-Fork",
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152},
            },
            weight_names=[
                "language_model.model.embed_tokens.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.norm.weight",
                "lm_head.weight",
            ],
        )
        assert is_mllm_model(str(model_dir)) is False

    def test_non_qwen_repackaged_vlm_with_vision_weights_is_vlm(self, tmp_path):
        """Round-5 #1121 guard: the SAME non-Qwen config, but the weight index
        DOES ship vision tensors → must route as VLM (``True``).  Distinguishes
        the two opposite failure modes purely by weight evidence."""
        model_dir = self._make_model_dir(
            tmp_path,
            "Gemma3-Real-VLM",
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152},
            },
            weight_names=[
                "language_model.model.embed_tokens.weight",
                "vision_tower.encoder.layers.0.self_attn.q_proj.weight",
                "multi_modal_projector.mm_input_projection_weight",
            ],
        )
        assert is_mllm_model(str(model_dir)) is True

    def test_gemma4_vision_embedder_family_is_vlm(self, tmp_path):
        """Round-5 prefix-audit guard: gemma-4-12B ships its vision stack as
        ``vision_embedder`` / ``embed_vision`` / ``embed_audio`` (NOT
        ``vision_tower``).  These prefixes were MISSING from the pre-round-5
        list, so the architecture-agnostic "absent → text" rule would have
        misrouted this real VLM to text.  The expanded
        ``MULTIMODAL_TENSOR_PREFIXES`` must catch it (True)."""
        model_dir = self._make_model_dir(
            tmp_path,
            "gemma-4-12B-it",
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152},
            },
            weight_names=[
                "language_model.model.embed_tokens.weight",
                "vision_embedder.patch_dense.weight",
                "embed_vision.embedding_projection.weight",
                "embed_audio.embedding_projection.weight",
            ],
        )
        assert is_mllm_model(str(model_dir)) is True

    def test_qwen35_moe_text_fork_still_text(self, tmp_path):
        """Round-3 behaviour preserved: a Qwen3.5-MoE text fork (no vision
        tensors) stays TEXT under the general rule (the former Qwen-specific
        special-case is now subsumed)."""
        model_dir = self._make_model_dir(
            tmp_path,
            "Qwen3.5-MoE-Text-Fork",
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152},
            },
            weight_names=[
                "language_model.model.embed_tokens.weight",
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                "language_model.model.norm.weight",
            ],
        )
        assert is_mllm_model(str(model_dir)) is False

    def test_qwen35_4b_alias_no_longer_forces_text(self):
        """Runtime capability, not a stale alias pin, now decides this VLM."""
        from vllm_mlx.model_aliases import resolve_profile

        profile = resolve_profile("qwen3.5-4b-4bit")
        assert profile.is_text_only is False
        assert profile.vision_min_memory_gb == 32.0

    def test_legacy_weights_probe_wrapper_delegates_to_shared_metadata(self, tmp_path):
        model_dir = self._make_model_dir(
            tmp_path,
            "vision-weights",
            {"vision_config": {}},
            ["vision_tower.blocks.0.weight"],
        )

        assert _local_checkpoint_has_multimodal_weights(model_dir) is True


class TestIsMllmModelCachedMetadata:
    """Cached metadata needs actual modality evidence for a new VLM route."""

    @staticmethod
    def _metadata(config):
        from vllm_mlx.model_metadata import ModelMetadata

        return ModelMetadata(config=config, chat_template=None, snapshot_dir=None)

    def test_text_only_config_overrides_substring_match(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod

        # Substring "gemma-3" would flag this as MLLM. Metadata says
        # Gemma3ForCausalLM (text-only) → route as text.
        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda name: self._metadata({"architectures": ["Gemma3ForCausalLM"]}),
        )
        assert is_mllm_model("mlx-community/gemma-3-1b-it-4bit") is False

    def test_vlm_config_keeps_substring_match(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod

        # Substring "gemma-3" matches and metadata also says VLM.
        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda name: self._metadata(
                {
                    "architectures": ["Gemma3ForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                }
            ),
        )
        assert is_mllm_model("mlx-community/gemma-3-27b-it-4bit") is True

    def test_hub_unreachable_falls_back_to_substring(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod

        # Substring matches; metadata is unavailable (offline / cold cache).
        # Should preserve the legacy True result.
        monkeypatch.setattr(utils_mod, "read_model_metadata", lambda name: None)
        assert is_mllm_model("mlx-community/gemma-3-4b-it-4bit") is True

    def test_metadata_detects_vlm_without_name_marker(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod

        # A re-packaged checkpoint need not contain a historical name marker.
        # Its cached config plus real checkpoint evidence route it correctly.
        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda name: self._metadata(
                {
                    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                }
            ),
        )
        monkeypatch.setattr(
            utils_mod,
            "checkpoint_has_multimodal_weights",
            lambda snapshot, config: True,
        )
        assert is_mllm_model("publisher/research-agent-mlx") is True

    def test_cached_inherited_vision_config_without_weights_stays_text(
        self, monkeypatch
    ):
        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda name: self._metadata(
                {
                    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                }
            ),
        )
        monkeypatch.setattr(
            utils_mod,
            "checkpoint_has_multimodal_weights",
            lambda snapshot, config: None,
        )

        assert is_mllm_model("publisher/text-only-repack") is False

    def test_text_only_alias_beats_cached_vision_evidence(self, monkeypatch):
        """An alias that POSITIVELY declares ``is_text_only`` short-circuits
        even against positive checkpoint evidence — the operator pin is
        authoritative for a checkpoint we deliberately serve text-only."""
        from vllm_mlx.api import utils as utils_mod
        from vllm_mlx.model_profile import ModelProfile

        text_only_profile = ModelProfile(
            hf_path="publisher/vision-config-served-text",
            is_text_only=True,
        )
        monkeypatch.setattr(
            utils_mod,
            "resolve_profile",
            lambda name: text_only_profile,
        )
        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda name: self._metadata(
                {
                    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                }
            ),
        )
        monkeypatch.setattr(
            utils_mod,
            "checkpoint_has_multimodal_weights",
            lambda snapshot, config: True,
        )

        assert is_mllm_model("publisher/vision-config-served-text") is False

    def test_registered_non_text_alias_yields_to_positive_vision_weights(
        self, monkeypatch
    ):
        """Regression for the #1121 routing-order bug: a registered alias that
        is NOT ``is_text_only`` and whose name carries NO legacy VLM substring
        must still route as multimodal once the checkpoint supplies positive
        vision weights. The bare alias entry must not override real evidence.

        Before the reorder, ``if profile is not None`` short-circuited to the
        (False) legacy name matcher BEFORE the ``verdict is True`` branch, so
        this repackaged VLM was misrouted to the text engine. This test fails
        against the old order and passes after evidence-priority routing.
        """
        from vllm_mlx.api import utils as utils_mod
        from vllm_mlx.model_profile import ModelProfile

        # A registered non-text alias (e.g. a text-family entry someone
        # repackaged a VLM under). ``is_text_only`` is False, and the alias
        # name has NO legacy VLM marker (``_check_legacy_string_patterns`` False).
        non_text_profile = ModelProfile(
            hf_path="publisher/research-agent-4b",
            is_text_only=False,
        )
        monkeypatch.setattr(
            utils_mod,
            "resolve_profile",
            lambda name: non_text_profile,
        )
        assert _check_legacy_string_patterns("publisher/research-agent-4b") is False
        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda name: self._metadata(
                {
                    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                }
            ),
        )
        monkeypatch.setattr(
            utils_mod,
            "checkpoint_has_multimodal_weights",
            lambda snapshot, config: True,
        )

        assert is_mllm_model("publisher/research-agent-4b") is True

    def test_curated_text_alias_beats_real_vision_weights(self, monkeypatch):
        """FIX 1 regression: a curated ``is_text_only`` alias whose CHECKPOINT
        genuinely ships ``vision_tower`` weights must still route as TEXT.

        ``mlx-community/Qwen3.5-4B-MLX-4bit`` (the default smoke alias) is
        curated text-only, yet its real ``model.safetensors.index.json`` carries
        BOTH ``language_model.*`` AND ``vision_tower.*`` tensors — so
        ``checkpoint_has_multimodal_weights`` correctly returns True on the raw
        bytes.  A prior fix let that True win over the curated alias, flipping
        the model to the MLLM lane and tripping the ``--pflash not supported for
        multimodal`` guard (7 CLI serve tests went red).  The curated
        ``is_text_only`` pin must short-circuit to text BEFORE the weight-
        evidence path.  Fails before FIX 1 (returns True), passes after.
        """
        from vllm_mlx.api import utils as utils_mod
        from vllm_mlx.model_profile import ModelProfile

        curated_text_profile = ModelProfile(
            hf_path="mlx-community/Qwen3.5-4B-MLX-4bit",
            is_text_only=True,
        )
        monkeypatch.setattr(
            utils_mod, "resolve_profile", lambda name: curated_text_profile
        )
        # Real checkpoint evidence: a genuine ``vision_tower.`` tensor is present
        # (positive multimodal verdict) — the curated pin must still win.
        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda name: self._metadata(
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                    "image_token_id": 248056,
                }
            ),
        )
        monkeypatch.setattr(
            utils_mod,
            "checkpoint_has_multimodal_weights",
            lambda snapshot, config: True,
        )

        assert is_mllm_model("mlx-community/Qwen3.5-4B-MLX-4bit") is False

    def test_qwen35_4b_positive_checkpoint_evidence_routes_as_vision(self, monkeypatch):
        """The former text pin no longer masks complete vision weights."""
        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda name: self._metadata(
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                    "text_config": {"layer_types": ["linear_attention"]},
                }
            ),
        )
        monkeypatch.setattr(
            utils_mod,
            "checkpoint_has_multimodal_weights",
            lambda snapshot, config: True,
        )

        assert is_mllm_model("qwen3.5-4b-4bit") is True

    def test_inconclusive_verdict_is_not_promoted_by_file_existence(
        self, monkeypatch, tmp_path
    ):
        """FIX 2 regression: an inconclusive checkpoint verdict (``None``) must
        NOT be promoted to multimodal just because checkpoint files exist on
        disk.  A cached (non-local) VLM-config snapshot whose weights are
        inconclusive and whose name carries NO legacy VLM substring must fall
        through to the name matcher (False), not be promoted to True.
        """
        from vllm_mlx.api import utils as utils_mod
        from vllm_mlx.model_metadata import ModelMetadata

        # Snapshot dir that HAS checkpoint files (so a file-existence probe
        # would say "evidence available") but the modality verdict is None.
        (tmp_path / "model.safetensors").write_bytes(b"\x00\x00")

        def _cached_meta(name):
            return ModelMetadata(
                config={
                    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "vision_config": {"hidden_size": 1024},
                },
                chat_template=None,
                snapshot_dir=tmp_path,
                is_local=False,
            )

        monkeypatch.setattr(utils_mod, "resolve_profile", lambda name: None)
        monkeypatch.setattr(utils_mod, "read_model_metadata", _cached_meta)
        monkeypatch.setattr(
            utils_mod,
            "checkpoint_has_multimodal_weights",
            lambda snapshot, config: None,
        )
        # Name has NO legacy VLM substring.
        assert _check_legacy_string_patterns("publisher/plain-repack") is False

        # Must NOT be promoted on bare file existence — stays text (False).
        assert is_mllm_model("publisher/plain-repack") is False

    def test_hub_helper_rejects_local_path_lookalikes(self):
        from vllm_mlx.api.utils import _try_read_hub_config_json

        # Defensive: hub helper must not interpret local-path-like strings
        # as repo IDs and trigger a network call.
        assert _try_read_hub_config_json("/abs/path/model") is None
        assert _try_read_hub_config_json("./relative/model") is None
        assert _try_read_hub_config_json("../up/model") is None
        assert _try_read_hub_config_json("~/cache/model") is None
        # No slash at all → not a repo ID shape.
        assert _try_read_hub_config_json("just-a-name") is None


class TestMllmBackboneIsHybrid:
    """A multimodal checkpoint whose language backbone is linear-attention /
    recurrent (ArraysCache) cannot be served by the MLLM continuous-batching
    engine (#352); the routing layer must be able to detect that offline."""

    @staticmethod
    def _meta(config):
        from vllm_mlx.model_metadata import ModelMetadata

        return ModelMetadata(config=config, chat_template=None, snapshot_dir=None)

    def _patch(self, monkeypatch, config):
        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(
            utils_mod, "read_model_metadata", lambda name: self._meta(config)
        )

    def test_gated_deltanet_layer_types_is_hybrid(self, monkeypatch):
        from vllm_mlx.api.utils import mllm_backbone_is_hybrid

        # Qwen3.5/3.6 shape: linear_attention layers interleaved with
        # full_attention, config nested under text_config, vision tower present.
        self._patch(
            monkeypatch,
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"hidden_size": 1024},
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                },
            },
        )
        assert mllm_backbone_is_hybrid("any/qwen3.5-like") is True

    def test_user_alias_load_resolves_exact_qwen35_checkpoint_before_routing(
        self, monkeypatch, tmp_path
    ):
        """Issue #2329: residency callers bypass ``cli.main`` alias resolution.

        The exact cached checkpoint selected by Desktop must be the common
        source for config materialization, automatic lane selection, and the
        loader. The immutable snapshot remains trusted through ``refs/main``;
        the fix must not guess among unreferenced revisions.
        """
        import json

        import huggingface_hub

        from vllm_mlx import server

        alias = "qwen3.5-2b-4bit"
        repo = "mlx-community/Qwen3.5-2B-MLX-4bit"
        revision = "93760be4f1f69842a46bc13dbdc0f19e291392a3"
        repo_root = tmp_path / "hub" / "models--mlx-community--Qwen3.5-2B-MLX-4bit"
        snapshot = repo_root / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "vision_config": {"model_type": "qwen3_5_vision"},
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "layer_types": [
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention",
                        ],
                    },
                }
            )
        )
        (snapshot / "model.safetensors").write_bytes(b"complete")
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "language_model.layers.0.weight": "model.safetensors",
                        "vision_tower.blocks.0.weight": "model.safetensors",
                    }
                }
            )
        )
        alias_file = tmp_path / "user-aliases.json"
        alias_file.write_text(
            json.dumps({"version": 1, "aliases": {alias: repo}}) + "\n"
        )
        monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", str(alias_file))
        monkeypatch.setattr(
            huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub")
        )
        from vllm_mlx import cli

        def fail_on_network(_name):
            raise AssertionError("complete singleton snapshot must stay offline")

        monkeypatch.setattr(
            cli,
            "_ensure_model_downloaded",
            fail_on_network,
        )

        class StubEngine:
            is_mllm = False
            preserve_native_tool_format = False
            _tokenizer = None
            _tool_logits_processor_factory = None

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        monkeypatch.setattr(server, "BatchedEngine", StubEngine)
        monkeypatch.setattr(server, "_engine", None, raising=False)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False, raising=False)
        monkeypatch.setattr(server, "_tool_call_parser", None, raising=False)
        monkeypatch.setattr(server, "_reasoning_parser_name", None, raising=False)
        monkeypatch.setattr(server, "_reasoning_parser", None, raising=False)
        monkeypatch.setattr(server, "_tool_parser_instance", None, raising=False)
        monkeypatch.setattr(server, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(server, "_enable_tool_logits_bias", False, raising=False)
        # Simulate a Desktop residency replacement after an image model. The
        # old process-global alias must not lend its image profile to Qwen.
        monkeypatch.setattr(server, "_model_alias", "z-image-turbo", raising=False)

        from vllm_mlx.model_metadata import read_model_metadata
        from vllm_mlx.utils.tokenizer import _local_snapshot_if_cached

        metadata = read_model_metadata(repo)
        assert metadata is not None
        assert metadata.snapshot_dir == snapshot
        assert _local_snapshot_if_cached(repo) == str(snapshot)

        server.load_model(alias)

        assert server._model_path == repo
        # Only an explicit ``served_model_name`` replaces the primary served
        # id. The canonical checkpoint remains primary by default, while the
        # requested short name is exposed separately as its alias.
        assert server._model_name == repo
        assert server._model_alias == alias
        assert server._engine.kwargs["model_name"] == str(snapshot)
        assert server._engine.kwargs["force_text"] is True

        # CLI startup arrives with the canonical repo plus a matching saved
        # alias. That same-source identity remains valid and keeps its profile.
        server.load_model(repo)
        assert server._model_alias == alias
        assert server._engine.kwargs["model_name"] == str(snapshot)
        assert server._engine.kwargs["force_text"] is True

        # A removed/corrupt prior identity is stale process state, not a reason
        # to reject a valid current canonical request.
        from vllm_mlx import model_aliases
        from vllm_mlx.user_aliases import UserAliasError

        real_resolve_model = model_aliases.resolve_model

        def resolve_with_stale_prior(name):
            if name == "removed-prior-alias":
                raise UserAliasError("prior alias no longer exists")
            return real_resolve_model(name)

        monkeypatch.setattr(model_aliases, "resolve_model", resolve_with_stale_prior)
        server._model_alias = "removed-prior-alias"
        server.load_model(repo)
        assert server._model_alias is None
        assert server._engine.kwargs["model_name"] == str(snapshot)
        assert server._engine.kwargs["force_text"] is True

    def test_unreferenced_snapshot_resolution_fails_closed(self, monkeypatch, tmp_path):
        """Ambiguous, partial, and malformed cache shapes must stay offline-safe."""
        import huggingface_hub

        from vllm_mlx import model_metadata

        repo = "publisher/model"
        repo_root = tmp_path / "models--publisher--model"
        snapshots = repo_root / "snapshots"
        snapshot = snapshots / "immutable-revision"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
        )
        (snapshot / "model.safetensors").write_bytes(b"complete")
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

        refs = repo_root / "refs"
        refs.mkdir()
        (refs / "main").write_text("missing-revision", encoding="utf-8")
        assert model_metadata.resolve_unreferenced_cached_snapshot(repo) is None
        (refs / "main").unlink()

        second = snapshots / "second-revision"
        second.mkdir()
        assert model_metadata.resolve_unreferenced_cached_snapshot(repo) is None
        second.rmdir()

        with monkeypatch.context() as scoped:
            scoped.setattr(
                "vllm_mlx.model_aliases.checkpoint_prefix", lambda _name: "nested/"
            )
            assert model_metadata.resolve_unreferenced_cached_snapshot(repo) is None

        (snapshot / "model.safetensors").unlink()
        assert model_metadata.resolve_unreferenced_cached_snapshot(repo) is None

        def fail_to_list(_path):
            raise OSError("stale cache mount")

        with monkeypatch.context() as scoped:
            scoped.setattr(Path, "iterdir", fail_to_list)
            assert model_metadata.resolve_unreferenced_cached_snapshot(repo) is None

    def test_sliding_and_full_attention_is_not_hybrid(self, monkeypatch):
        from vllm_mlx.api.utils import mllm_backbone_is_hybrid

        # Gemma-4 shape: sliding_attention + full_attention → RotatingKVCache,
        # which the MLLM engine handles. NOT a linear-attention backbone.
        self._patch(
            monkeypatch,
            {
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "vision_config": {"hidden_size": 1024},
                "text_config": {
                    "model_type": "gemma4_unified_text",
                    "layer_types": [
                        "sliding_attention",
                        "sliding_attention",
                        "full_attention",
                    ],
                },
            },
        )
        assert mllm_backbone_is_hybrid("any/gemma4-like") is False

    def test_recurrent_model_type_without_layer_types_is_hybrid(self, monkeypatch):
        from vllm_mlx.api.utils import mllm_backbone_is_hybrid

        # A pure Mamba/recurrent backbone that does not enumerate layer_types.
        self._patch(
            monkeypatch,
            {"architectures": ["SomeVLM"], "model_type": "mamba2_vlm"},
        )
        assert mllm_backbone_is_hybrid("any/mamba-vlm") is True

    def test_missing_metadata_returns_false(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod
        from vllm_mlx.api.utils import mllm_backbone_is_hybrid

        # No config → unknown → False (never removes multimodal routing from a
        # checkpoint we cannot positively classify as hybrid).
        monkeypatch.setattr(utils_mod, "read_model_metadata", lambda name: None)
        assert mllm_backbone_is_hybrid("any/unknown") is False

    def test_plain_attention_top_level_config_is_not_hybrid(self, monkeypatch):
        from vllm_mlx.api.utils import mllm_backbone_is_hybrid

        # Non-nested config, ordinary transformer → not hybrid.
        self._patch(
            monkeypatch,
            {"model_type": "llama", "num_hidden_layers": 32},
        )
        assert mllm_backbone_is_hybrid("any/llama-like") is False


class TestResolveServingLane:
    """The single lane-decision orchestrator: given the two offline probes and
    the explicit flags, decide the FINAL (is_mllm_lane, auto_text_fallback)."""

    def _patch_probes(
        self, monkeypatch, *, is_mllm, hybrid, hybrid_runtime_supported=False
    ):
        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(utils_mod, "is_mllm_model", lambda n: is_mllm)
        monkeypatch.setattr(
            utils_mod,
            "mllm_backbone_cache_mode",
            lambda n: "arrays" if hybrid else None,
        )
        monkeypatch.setattr(
            utils_mod,
            "mllm_hybrid_runtime_supported",
            lambda: hybrid_runtime_supported,
        )
        monkeypatch.setattr(utils_mod, "physical_ram_gb", lambda: 64.0)

    def test_hybrid_vlm_auto_downgrades_to_text(self, monkeypatch):
        from vllm_mlx.api.utils import resolve_serving_lane

        # Multimodal checkpoint + hybrid backbone, no flags → text lane, marked
        # as an AUTOMATIC fallback (Qwen3.6-27B shape).
        self._patch_probes(monkeypatch, is_mllm=True, hybrid=True)
        assert resolve_serving_lane("any/qwen36-27b") == (False, True)

    @pytest.mark.parametrize(
        ("architecture_unavailable", "cache_mode", "reason"),
        [
            (True, None, "vision_architecture_unavailable"),
            (False, "other", "vision_hybrid_cache_unsupported"),
        ],
    )
    def test_unsupported_vision_contracts_fail_closed(
        self, monkeypatch, architecture_unavailable, cache_mode, reason
    ):
        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(utils_mod, "is_mllm_model", lambda _name: True)
        monkeypatch.setattr(
            utils_mod,
            "mllm_arch_unsupported_but_text_vendored",
            lambda _name: architecture_unavailable,
        )
        monkeypatch.setattr(
            utils_mod, "mllm_backbone_cache_mode", lambda _name: cache_mode
        )

        assert utils_mod.resolve_serving_lane_decision(
            "local/checkpoint"
        ) == utils_mod.ServingLaneDecision(False, reason, auto_text_fallback=True)

    def test_hybrid_vlm_uses_vision_when_runtime_supports_it(self, monkeypatch):
        from vllm_mlx.api.utils import (
            resolve_serving_lane,
            resolve_serving_lane_decision,
        )

        self._patch_probes(
            monkeypatch,
            is_mllm=True,
            hybrid=True,
            hybrid_runtime_supported=True,
        )
        assert resolve_serving_lane("any/qwen35-9b") == (True, False)
        assert (
            resolve_serving_lane_decision("any/qwen35-9b").reason
            == "vision_hybrid_runtime_supported"
        )

    def test_hybrid_vlm_falls_back_when_measured_vision_floor_does_not_fit(
        self, monkeypatch
    ):
        from vllm_mlx.api import utils as utils_mod

        self._patch_probes(
            monkeypatch,
            is_mllm=True,
            hybrid=True,
            hybrid_runtime_supported=True,
        )
        monkeypatch.setattr(utils_mod, "physical_ram_gb", lambda: 16.0)

        decision = utils_mod.resolve_serving_lane_decision(
            "local/qwen35", vision_min_memory_gb=32.0
        )

        assert decision == utils_mod.ServingLaneDecision(
            False, "vision_memory_insufficient", auto_text_fallback=True
        )

    def test_hybrid_vlm_uses_vision_at_measured_memory_floor(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod

        self._patch_probes(
            monkeypatch,
            is_mllm=True,
            hybrid=True,
            hybrid_runtime_supported=True,
        )
        monkeypatch.setattr(utils_mod, "physical_ram_gb", lambda: 32.0)

        assert utils_mod.resolve_serving_lane_decision(
            "local/qwen35", vision_min_memory_gb=32.0
        ) == utils_mod.ServingLaneDecision(True, "vision_hybrid_runtime_supported")

    def test_explicit_vision_honors_measured_memory_floor(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod

        self._patch_probes(
            monkeypatch,
            is_mllm=True,
            hybrid=True,
            hybrid_runtime_supported=True,
        )
        monkeypatch.setattr(utils_mod, "physical_ram_gb", lambda: 16.0)

        assert utils_mod.resolve_serving_lane_decision(
            "local/qwen35",
            force_mllm=True,
            vision_min_memory_gb=32.0,
        ) == utils_mod.ServingLaneDecision(
            False, "vision_memory_insufficient", auto_text_fallback=True
        )

    def test_automatic_vision_still_honors_measured_memory_floor(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod

        self._patch_probes(
            monkeypatch,
            is_mllm=True,
            hybrid=True,
            hybrid_runtime_supported=True,
        )
        monkeypatch.setattr(utils_mod, "physical_ram_gb", lambda: 16.0)

        auto = utils_mod.resolve_serving_lane_decision(
            "qwen3.5-4b-4bit",
            vision_min_memory_gb=32.0,
        )

        assert auto == utils_mod.ServingLaneDecision(
            False, "vision_memory_insufficient", auto_text_fallback=True
        )

    @pytest.mark.parametrize(
        ("architecture_unavailable", "cache_mode"),
        [(True, None), (False, "other"), (False, "arrays")],
    )
    def test_explicit_vision_precedes_automatic_capability_fallbacks(
        self, monkeypatch, architecture_unavailable, cache_mode
    ):
        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(utils_mod, "is_mllm_model", lambda _name: True)
        monkeypatch.setattr(
            utils_mod,
            "mllm_arch_unsupported_but_text_vendored",
            lambda _name: architecture_unavailable,
        )
        monkeypatch.setattr(
            utils_mod, "mllm_backbone_cache_mode", lambda _name: cache_mode
        )
        monkeypatch.setattr(utils_mod, "mllm_hybrid_runtime_supported", lambda: False)
        monkeypatch.setattr(utils_mod, "physical_ram_gb", lambda: 64.0)

        assert utils_mod.resolve_serving_lane_decision(
            "local/checkpoint",
            force_mllm=True,
            vision_min_memory_gb=32.0,
        ) == utils_mod.ServingLaneDecision(True, "vision_lane_forced")

    def test_unknown_physical_memory_does_not_invent_a_fallback(self, monkeypatch):
        from vllm_mlx.api import utils as utils_mod

        self._patch_probes(
            monkeypatch,
            is_mllm=True,
            hybrid=True,
            hybrid_runtime_supported=True,
        )
        monkeypatch.setattr(utils_mod, "physical_ram_gb", lambda: 0.0)

        assert utils_mod.resolve_serving_lane_decision(
            "local/qwen35", vision_min_memory_gb=32.0
        ) == utils_mod.ServingLaneDecision(True, "vision_hybrid_runtime_supported")

    @pytest.mark.parametrize(
        ("installed", "supported"),
        [("0.6.15", False), ("0.6.16", True), ("0.7.0", True)],
    )
    def test_hybrid_runtime_version_contract(self, monkeypatch, installed, supported):
        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(utils_mod, "version", lambda _distribution: installed)
        assert utils_mod.mllm_hybrid_runtime_supported() is supported

    def test_missing_hybrid_runtime_fails_closed(self, monkeypatch):
        from importlib.metadata import PackageNotFoundError

        from vllm_mlx.api import utils as utils_mod

        def missing(_distribution):
            raise PackageNotFoundError

        monkeypatch.setattr(utils_mod, "version", missing)
        assert utils_mod.mllm_hybrid_runtime_supported() is False

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            ({"text_config": {"layer_types": ["mamba"]}}, "other"),
            ({"text_config": {"model_type": "qwen3_next"}}, "arrays"),
        ],
    )
    def test_hybrid_cache_mode_uses_checkpoint_metadata(
        self, monkeypatch, config, expected
    ):
        from types import SimpleNamespace

        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda _name: SimpleNamespace(config=config),
        )

        assert utils_mod.mllm_backbone_cache_mode("local/checkpoint") == expected

    def test_missing_vendored_vision_module_fails_closed(self, monkeypatch):
        from types import SimpleNamespace

        from vllm_mlx.api import utils as utils_mod

        monkeypatch.setattr(
            utils_mod,
            "read_model_metadata",
            lambda _name: SimpleNamespace(config={"model_type": "muse_glimmer"}),
        )
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda _name: (_ for _ in ()).throw(ImportError("vision extra missing")),
        )

        assert utils_mod.mllm_arch_unsupported_but_text_vendored("local/model") is True

    def test_genuine_vlm_stays_on_mllm_lane(self, monkeypatch):
        from vllm_mlx.api.utils import resolve_serving_lane

        # Multimodal + sliding/full attention (gemma-4) → MLLM lane, no fallback.
        self._patch_probes(monkeypatch, is_mllm=True, hybrid=False)
        assert resolve_serving_lane("any/gemma4-12b") == (True, False)

    def test_pure_text_model_is_text_lane_no_fallback(self, monkeypatch):
        from vllm_mlx.api.utils import resolve_serving_lane

        # Not multimodal at all → text lane by nature, not an auto-fallback.
        self._patch_probes(monkeypatch, is_mllm=False, hybrid=False)
        assert resolve_serving_lane("any/plain-llm") == (False, False)

    def test_explicit_force_text_wins_no_auto_marker(self, monkeypatch):
        from vllm_mlx.api.utils import resolve_serving_lane

        # Even a hybrid VLM: an explicit --text-only is NOT an auto-fallback, so
        # diagnostics don't conflate the two. Probes are not consulted.
        self._patch_probes(monkeypatch, is_mllm=True, hybrid=True)
        assert resolve_serving_lane("any/qwen36-27b", force_text=True) == (
            False,
            False,
        )

    def test_explicit_force_mllm_keeps_mllm_lane(self, monkeypatch):
        from vllm_mlx.api.utils import resolve_serving_lane

        # Explicit --mllm on a hybrid VLM keeps the MLLM lane (the engine will
        # raise its own #352 error naming --mllm — the flag the user set).
        self._patch_probes(
            monkeypatch,
            is_mllm=True,
            hybrid=True,
            hybrid_runtime_supported=True,
        )
        monkeypatch.setattr(
            "vllm_mlx.api.utils.physical_ram_gb",
            lambda: 64.0,
        )
        assert resolve_serving_lane("any/qwen36-27b", force_mllm=True) == (
            True,
            False,
        )

    def test_force_mllm_on_text_checkpoint_stays_text_forced(self, monkeypatch):
        from vllm_mlx.api.utils import (
            ServingLaneDecision,
            resolve_serving_lane_decision,
        )

        self._patch_probes(monkeypatch, is_mllm=False, hybrid=False)

        assert resolve_serving_lane_decision(
            "local/text-model", force_mllm=True
        ) == ServingLaneDecision(True, "vision_lane_forced")


class TestServingLaneDecisionValidation:
    """The SSOT membership checks on ``ServingLaneDecision`` fail fast."""

    def test_unknown_reason_is_rejected(self):
        from vllm_mlx.api.utils import ServingLaneDecision

        with pytest.raises(ValueError, match="unknown serving_lane_reason"):
            ServingLaneDecision(False, "operator_forced_text")

    def test_auto_text_fallback_with_non_downgrade_reason_is_rejected(self):
        from vllm_mlx.api.utils import ServingLaneDecision

        # A vision-lane reason is not a silent downgrade, so pairing it with
        # auto_text_fallback=True must fail (it would reach the generic copy).
        with pytest.raises(ValueError, match="requires auto_text_fallback=False"):
            ServingLaneDecision(True, "vision_supported", auto_text_fallback=True)

    def test_downgrade_reason_without_auto_text_fallback_is_rejected(self):
        from vllm_mlx.api.utils import ServingLaneDecision

        with pytest.raises(ValueError, match="requires auto_text_fallback=True"):
            ServingLaneDecision(False, "vision_memory_insufficient")

    def test_reason_must_match_selected_lane(self):
        from vllm_mlx.api.utils import ServingLaneDecision

        with pytest.raises(ValueError, match="requires is_mllm=True"):
            ServingLaneDecision(False, "vision_supported")
        with pytest.raises(ValueError, match="requires is_mllm=False"):
            ServingLaneDecision(True, "text_checkpoint")

    def test_auto_text_fallback_with_downgrade_reason_is_accepted(self):
        from vllm_mlx.api.utils import ServingLaneDecision

        decision = ServingLaneDecision(
            False, "vision_memory_insufficient", auto_text_fallback=True
        )
        assert decision.reason == "vision_memory_insufficient"
        assert decision.auto_text_fallback is True

    def test_ssot_members_are_all_constructible(self):
        from vllm_mlx.api.utils import (
            AUTO_TEXT_FALLBACK_REASONS,
            SERVING_LANE_REASONS,
            VISION_SERVING_LANE_REASONS,
            ServingLaneDecision,
        )

        # Every reason the SSOT names must be a valid construction — this would
        # catch the set drifting from what the decision function emits. Every
        # auto-fallback reason must also pass the flag.
        for reason in SERVING_LANE_REASONS:
            ServingLaneDecision(
                reason in VISION_SERVING_LANE_REASONS,
                reason,
                auto_text_fallback=reason in AUTO_TEXT_FALLBACK_REASONS,
            )
        # Auto-fallback members are a strict subset of all reasons.
        assert AUTO_TEXT_FALLBACK_REASONS <= SERVING_LANE_REASONS
        assert VISION_SERVING_LANE_REASONS <= SERVING_LANE_REASONS
        assert not (AUTO_TEXT_FALLBACK_REASONS & VISION_SERVING_LANE_REASONS)


class TestExtractMultimodalContent:
    """Tests for extract_multimodal_content function."""

    def test_simple_text_messages(self):
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]
        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 2
        assert processed[0] == {"role": "system", "content": "You are helpful."}
        assert processed[1] == {"role": "user", "content": "Hello"}
        assert images == []
        assert videos == []

    def test_none_content(self):
        messages = [Message(role="assistant", content=None)]
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0] == {"role": "assistant", "content": ""}

    def test_multimodal_with_image_url(self):
        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="What is this?"),
                    ContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="https://example.com/img.png"),
                    ),
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "What is this?"
        assert images == ["https://example.com/img.png"]
        assert videos == []

    def test_responses_text_blocks_are_extracted_as_text(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "input_text", "text": "question"},
                    {"type": "output_text", "text": "prior answer"},
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert processed == [{"role": "user", "content": "question\nprior answer"}]
        assert images == []
        assert videos == []

    def test_multimodal_with_dict_image_url(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert images == ["data:image/png;base64,abc"]

    def test_multimodal_with_input_image(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "input_text", "text": "Describe this"},
                    {
                        "type": "input_image",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert processed == [{"role": "user", "content": "Describe this"}]
        assert images == ["data:image/png;base64,abc"]
        assert videos == []

    def test_multimodal_with_string_image_url_rejected(self):
        """F-065: the bare-string ``image_url`` shorthand was
        previously accepted at the Message layer and silently
        dropped by the multimodal preprocessor. Per the new
        OpenAI-spec contract the wire form must be the object
        shape ``{"url": "..."}``; constructing a Message with
        the bare-string form now raises ``ValidationError``
        upfront so the silent-drop hazard is closed."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as ei:
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Look"},
                    {
                        "type": "image_url",
                        "image_url": "https://example.com/img.png",
                    },
                ],
            )
        assert "image_url must be an object" in str(ei.value)

    def test_multimodal_with_video(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What happens?"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert videos == ["/path/to/video.mp4"]

    def test_multimodal_with_video_url(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/v.mp4"},
                    },
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert videos == ["https://example.com/v.mp4"]

    def test_unknown_content_block_rejected_not_empty_prompt(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Use this block"},
                    {"type": "document_url", "document_url": {"url": "doc.pdf"}},
                ],
            )
        ]

        with pytest.raises(ValueError, match="Unsupported content block type"):
            extract_multimodal_content(messages)

    def test_chat_image_url_string_rejected_not_responses_shorthand(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": "data:image/png;base64,abc"},
                ],
            }
        ]

        with pytest.raises(ValueError, match="image_url must be an object"):
            extract_multimodal_content(messages)

    def test_malformed_image_block_rejected_not_empty_prompt(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {}},
                ],
            )
        ]

        with pytest.raises(ValueError, match="image_url.url"):
            extract_multimodal_content(messages)

    def test_audio_content_block_rejected_not_silently_dropped(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Transcribe"},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/a.wav"},
                    },
                ],
            )
        ]

        with pytest.raises(ValueError, match="Audio content blocks"):
            extract_multimodal_content(messages)

    def test_multimodal_with_string_video_url_rejected(self):
        """F-065 mirror surface: bare-string ``video_url`` was
        also previously accepted and silently dropped. Now → 422."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as ei:
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Look"},
                    {
                        "type": "video_url",
                        "video_url": "https://example.com/v.mp4",
                    },
                ],
            )
        assert "video_url must be an object" in str(ei.value)

    def test_multiple_images(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Compare these"},
                    {"type": "image_url", "image_url": {"url": "img1.png"}},
                    {"type": "image_url", "image_url": {"url": "img2.png"}},
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert len(images) == 2

    def test_tool_response_message(self):
        messages = [
            Message(role="tool", content="72F and sunny", tool_call_id="call_1")
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0]["role"] == "user"
        assert "Tool Result" in processed[0]["content"]
        assert "call_1" in processed[0]["content"]

    def test_tool_response_preserve_native(self):
        messages = [
            Message(role="tool", content="72F and sunny", tool_call_id="call_1")
        ]
        processed, images, videos = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["role"] == "tool"
        assert processed[0]["tool_call_id"] == "call_1"
        assert processed[0]["content"] == "72F and sunny"

    def test_assistant_with_tool_calls(self):
        messages = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0]["role"] == "assistant"
        assert "get_weather" in processed[0]["content"]

    def test_assistant_with_tool_calls_preserve_native(self):
        messages = [
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["role"] == "assistant"
        assert processed[0]["content"] == "Let me check."
        assert "tool_calls" in processed[0]
        assert len(processed[0]["tool_calls"]) == 1

    def test_dict_messages(self):
        messages = [
            Message(role="user", content="Hello"),
        ]
        # Also test with raw dicts (the function handles both)
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0]["content"] == "Hello"

    def test_image_type_content_with_raw_dicts(self):
        # type="image" path handles raw dict content (not Pydantic ContentPart)
        # Pass a raw dict message to avoid Pydantic stripping unknown fields
        raw_messages = [
            type(
                "Msg",
                (),
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "image": "https://example.com/img.png"},
                    ],
                    "tool_calls": None,
                    "tool_call_id": None,
                },
            )()
        ]
        processed, images, videos = extract_multimodal_content(raw_messages)
        assert images == ["https://example.com/img.png"]

    def test_empty_messages(self):
        processed, images, videos = extract_multimodal_content([])
        assert processed == []
        assert images == []
        assert videos == []

    def test_tool_response_none_content(self):
        messages = [Message(role="tool", content=None, tool_call_id="call_1")]
        processed, images, videos = extract_multimodal_content(messages)
        assert processed[0]["role"] == "user"
        assert "call_1" in processed[0]["content"]

    def test_multiple_text_parts_combined(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            )
        ]
        processed, images, videos = extract_multimodal_content(messages)
        assert "First part." in processed[0]["content"]
        assert "Second part." in processed[0]["content"]

    def test_assistant_tool_calls_with_list_content(self):
        """Regression test for issue #61: list content + tool_calls causes TypeError."""
        messages = [
            Message(
                role="assistant",
                content=[ContentPart(type="text", text="Let me check.")],
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Prague"}',
                        },
                    }
                ],
            )
        ]
        result, images, videos = extract_multimodal_content(messages)
        assert isinstance(result[0]["content"], str)
        assert "Let me check." in result[0]["content"]
        assert "get_weather" in result[0]["content"]

    def test_assistant_tool_calls_with_list_content_native(self):
        """Regression test for issue #61: list content + tool_calls with native format."""
        messages = [
            Message(
                role="assistant",
                content=[ContentPart(type="text", text="Checking now.")],
                tool_calls=[
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            )
        ]
        result, images, videos = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert isinstance(result[0]["content"], str)
        assert "Checking now." in result[0]["content"]
        assert "tool_calls" in result[0]


class TestContentToText:
    """Tests for the _content_to_text helper."""

    def test_none(self):
        assert _content_to_text(None) == ""

    def test_string(self):
        assert _content_to_text("hello") == "hello"

    def test_empty_string(self):
        assert _content_to_text("") == ""

    def test_list_of_content_parts(self):
        parts = [
            ContentPart(type="text", text="Hello"),
            ContentPart(type="text", text="World"),
        ]
        assert _content_to_text(parts) == "Hello\nWorld"

    def test_list_of_dicts(self):
        parts = [
            {"type": "text", "text": "foo"},
            {"type": "image_url", "image_url": "http://img"},
        ]
        assert _content_to_text(parts) == "foo"

    def test_list_of_unknown_dicts_ignored(self):
        parts = [
            {},
            {"type": "future_block", "text": "ignored"},
            {"type": "text", "text": "foo"},
        ]
        assert _content_to_text(parts) == "foo"

    def test_list_of_responses_text_dicts(self):
        parts = [
            {"type": "input_text", "text": "foo"},
            {"type": "output_text", "text": "bar"},
            {"type": "input_image", "image_url": "http://img"},
        ]
        assert _content_to_text(parts) == "foo\nbar"

    def test_list_with_no_text_parts(self):
        parts = [{"type": "image_url", "image_url": "http://img"}]
        assert _content_to_text(parts) == ""

    def test_empty_list(self):
        assert _content_to_text([]) == ""


class TestValidateContentBlocksForCapabilities:
    @pytest.mark.parametrize(
        "messages",
        [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": "not-an-object"}
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "abc", "format": "exe"},
                        }
                    ],
                }
            ],
        ],
    )
    def test_invalid_input_audio_payload_is_rejected(self, messages):
        with pytest.raises(ValueError):
            validate_content_blocks_for_capabilities(
                messages,
                model_name="audio-model",
                allow_image=False,
                allow_video=False,
                allow_audio=True,
            )

    def test_enabled_image_and_video_capabilities_accept_media(self):
        validate_content_blocks_for_capabilities(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": "https://example.com/video.mp4"},
                        },
                    ],
                }
            ],
            model_name="vision-model",
            allow_image=True,
            allow_video=True,
        )

    def test_text_lane_image_error_has_stable_machine_readable_code(self):
        from vllm_mlx.api.utils import UnsupportedContentBlockError

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/img"}}
                ],
            }
        ]

        with pytest.raises(UnsupportedContentBlockError) as caught:
            validate_content_blocks_for_capabilities(
                messages,
                model_name="vision-model",
                allow_image=False,
                allow_video=False,
            )

        assert caught.value.openai_detail(
            serving_lane_reason="vision_hybrid_runtime_unsupported"
        ) == {
            "error": {
                "message": (
                    "Model 'vision-model' is serving text-only; image input "
                    "is unsupported."
                ),
                "type": "invalid_request_error",
                "code": "image_input_unsupported",
                "param": "messages.content",
                "serving_lane_reason": "vision_hybrid_runtime_unsupported",
            }
        }
        assert (
            "serving_lane_reason"
            not in caught.value.openai_detail(serving_lane_reason=object())["error"]
        )

    def test_chat_route_preserves_typed_text_lane_image_error(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.routes.chat import router

        class TextLaneEngine:
            is_mllm = False
            serving_lane_reason = "vision_hybrid_runtime_unsupported"
            preserve_native_tool_format = False
            supports_guided_generation = False
            tokenizer = None

        cfg = reset_config()
        cfg.engine = TextLaneEngine()
        cfg.model_name = "vision-model"
        cfg.model_registry = None
        cfg.no_thinking = True
        cfg.tool_call_parser = None
        cfg.reasoning_parser_name = None
        app = FastAPI()
        app.include_router(router)

        try:
            response = TestClient(app).post(
                "/v1/chat/completions",
                json={
                    "model": "vision-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,abc"},
                                }
                            ],
                        }
                    ],
                    "max_tokens": 8,
                },
            )
        finally:
            reset_config()

        assert response.status_code == 400
        assert response.json()["detail"]["error"]["code"] == ("image_input_unsupported")

    def test_chat_text_block_rejects_missing_text(self):
        messages = [{"role": "user", "content": [{"type": "text"}]}]

        with pytest.raises(ValueError, match="text\\.text is required"):
            validate_content_blocks_for_capabilities(
                messages,
                model_name="chat-model",
                allow_image=False,
                allow_video=False,
            )

    def test_chat_text_block_allows_empty_text(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]

        validate_content_blocks_for_capabilities(
            messages,
            model_name="chat-model",
            allow_image=False,
            allow_video=False,
        )

    def test_chat_text_block_rejects_explicit_null_text(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": None}]}]

        with pytest.raises(ValueError, match="content\\[\\]\\.text must be"):
            validate_content_blocks_for_capabilities(
                messages,
                model_name="chat-model",
                allow_image=False,
                allow_video=False,
            )

    def test_responses_text_blocks_are_valid_text_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "question"},
                    {"type": "output_text", "text": "prior answer"},
                ],
            }
        ]

        validate_content_blocks_for_capabilities(
            messages,
            model_name="chat-model",
            allow_image=False,
            allow_video=False,
        )

    def test_input_audio_requires_format_even_when_audio_allowed(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "base64data"},
                    }
                ],
            }
        ]

        with pytest.raises(ValueError, match="input_audio\\.format"):
            validate_content_blocks_for_capabilities(
                messages,
                model_name="audio-model",
                allow_image=False,
                allow_video=False,
                allow_audio=True,
            )

    def test_input_audio_allowed_when_audio_capability_enabled(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "base64data", "format": "wav"},
                    }
                ],
            }
        ]

        validate_content_blocks_for_capabilities(
            messages,
            model_name="audio-model",
            allow_image=False,
            allow_video=False,
            allow_audio=True,
        )

    @pytest.mark.parametrize(
        "content_part",
        [
            {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}},
            {"type": "audio", "audio": "base64data"},
        ],
    )
    def test_audio_url_and_audio_rejected_even_when_audio_allowed(self, content_part):
        messages = [{"role": "user", "content": [content_part]}]

        with pytest.raises(ValueError, match="only input_audio is supported"):
            validate_content_blocks_for_capabilities(
                messages,
                model_name="audio-model",
                allow_image=False,
                allow_video=False,
                allow_audio=True,
            )


class TestGptOssSpecialTokens:
    """Tests for GPT-OSS channel token handling in utils."""

    def test_pattern_matches_channel_token(self):
        assert SPECIAL_TOKENS_PATTERN.search("<|channel|>") is not None

    def test_pattern_matches_message_token(self):
        assert SPECIAL_TOKENS_PATTERN.search("<|message|>") is not None

    def test_pattern_matches_start_token(self):
        assert SPECIAL_TOKENS_PATTERN.search("<|start|>") is not None

    def test_pattern_matches_return_token(self):
        assert SPECIAL_TOKENS_PATTERN.search("<|return|>") is not None

    def test_pattern_matches_call_token(self):
        assert SPECIAL_TOKENS_PATTERN.search("<|call|>") is not None

    def test_clean_output_extracts_final_channel(self):
        text = (
            "<|channel|>analysis<|message|>Thinking about it"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>"
        )
        result = clean_output_text(text)
        assert result == "The answer is 42"
        assert "<|" not in result

    def test_clean_output_final_only(self):
        text = "<|channel|>final<|message|>Just the answer<|return|>"
        result = clean_output_text(text)
        assert result == "Just the answer"

    def test_clean_output_strips_return_token(self):
        text = "<|channel|>final<|message|>Hello world<|return|>"
        result = clean_output_text(text)
        assert "<|return|>" not in result
        assert result == "Hello world"

    def test_clean_output_no_channel_tokens_passthrough(self):
        text = "Normal text without any channel tokens."
        result = clean_output_text(text)
        assert result == text

    def test_pattern_matches_constrain_token(self):
        assert SPECIAL_TOKENS_PATTERN.search("<|constrain|>") is not None

    def test_clean_output_constrain_format(self):
        """Should extract final content from extended constrain format."""
        text = (
            "<|channel|>analysis<|message|>Thinking"
            "<|end|><|channel|>final <|constrain|>JSON<|message|>"
            '{"hello":"world"}<|return|>'
        )
        result = clean_output_text(text)
        assert result == '{"hello":"world"}'
        assert "<|constrain|>" not in result
        assert "<|channel|>" not in result

    def test_clean_output_constrain_final_only(self):
        """Should handle constrain format with only final channel."""
        text = '<|channel|>final <|constrain|>JSON<|message|>{"key":"value"}<|return|>'
        result = clean_output_text(text)
        assert result == '{"key":"value"}'

    def test_clean_output_no_final_strips_constrain(self):
        """When no final channel found, constrain tokens should be stripped."""
        text = "<|channel|>analysis<|message|>Just thinking <|constrain|>something"
        result = clean_output_text(text)
        assert "<|constrain|>" not in result


def test_dflash_runtime_probe_uses_symbol_level_discovery(monkeypatch):
    """The lightweight probe checks the required drafter package itself."""
    import importlib.util

    from vllm_mlx.speculative.dflash.eligibility import have_runtime

    probes: list[str] = []

    def find_spec(name: str):
        probes.append(name)
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert have_runtime() is True
    assert probes == ["mlx_vlm.speculative.drafters"]
