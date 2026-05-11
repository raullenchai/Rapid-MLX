# SPDX-License-Identifier: Apache-2.0
"""Tests for lightning-mlx CLI model presets."""

import json
from argparse import Namespace

from vllm_mlx.cli import (
    _apply_ornstein_mtplx_preset,
    _apply_qwen36_35b_defaults,
    _apply_qwen36_mtplx_preset,
)
from vllm_mlx.model_aliases import resolve_model


def _serve_args(model: str, original_alias: str | None = None) -> Namespace:
    return Namespace(
        command="serve",
        model=model,
        _original_alias=original_alias,
        enable_mtp=False,
        disable_mtp=False,
        served_model_name=None,
        port=8000,
        default_temperature=None,
        default_top_p=None,
        enable_prefix_cache=True,
        disable_prefix_cache=False,
        max_num_seqs=256,
        prefill_batch_size=8,
        completion_batch_size=32,
        stream_interval=1,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        reasoning_parser=None,
        no_thinking=False,
        log_level="INFO",
        enable_tool_logits_bias=False,
    )


def test_qwen36_alias_applies_agentic_mtplx_preset():
    args = _serve_args(
        "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        original_alias="qwen3.6-27b",
    )

    _apply_qwen36_mtplx_preset(args, ["serve", "qwen3.6-27b"])

    assert args.enable_mtp is True
    assert args.served_model_name == "local"
    assert args.port == 8010
    assert args.default_temperature == 0.6
    assert args.default_top_p == 0.95
    assert args.disable_prefix_cache is True
    assert args.max_num_seqs == 1
    assert args.prefill_batch_size == 1
    assert args.completion_batch_size == 1
    assert args.stream_interval == 1
    assert args.enable_auto_tool_choice is True
    assert args.tool_call_parser == "qwen3_coder_xml"
    assert args.reasoning_parser == "qwen3"
    assert args.no_thinking is False
    assert args.enable_tool_logits_bias is False


def test_qwen36_local_path_applies_same_preset():
    args = _serve_args("/models/Qwen3.6-27B-MTPLX-Optimized-Speed")

    _apply_qwen36_mtplx_preset(
        args, ["serve", "/models/Qwen3.6-27B-MTPLX-Optimized-Speed"]
    )

    assert args.enable_mtp is True
    assert args.served_model_name == "local"
    assert args.port == 8010
    assert args.tool_call_parser == "qwen3_coder_xml"
    assert args.reasoning_parser == "qwen3"
    assert args.no_thinking is False
    assert args.enable_tool_logits_bias is False


def test_verified_local_mtplx_qwen35b_path_uses_no_thinking_default(tmp_path):
    model = tmp_path / "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"
    model.mkdir()
    (model / "mtp.safetensors").write_bytes(b"")
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"arch_id": "qwen3-next-mtp"}), encoding="utf-8"
    )
    args = _serve_args(str(model))

    _apply_qwen36_mtplx_preset(args, ["serve", str(model)])

    assert args.enable_mtp is True
    assert args.served_model_name == "local"
    assert args.port == 8010
    assert args.default_temperature == 0.6
    assert args.default_top_p == 0.95
    assert args.disable_prefix_cache is True
    assert args.max_num_seqs == 1
    assert args.prefill_batch_size == 1
    assert args.completion_batch_size == 1
    assert args.stream_interval == 1
    assert args.enable_auto_tool_choice is True
    assert args.tool_call_parser == "qwen3_coder_xml"
    assert args.reasoning_parser == "qwen3"
    # Thinking is now ENABLED by default for 35B (was True before).
    # With no_thinking=True the model emits <|im_end|> immediately
    # after tool results and the agent loop dies; thinking enabled
    # restores 100% agentic functionality with pi/Claude Code/etc.
    assert args.no_thinking is False
    assert args.log_level == "WARNING"
    assert args.enable_tool_logits_bias is False


def test_verified_local_mtplx_qwen35b_path_respects_no_thinking(tmp_path):
    model = tmp_path / "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"
    model.mkdir()
    (model / "mtp.safetensors").write_bytes(b"")
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"arch_id": "qwen3-next-mtp"}), encoding="utf-8"
    )
    args = _serve_args(str(model))
    args.no_thinking = True

    _apply_qwen36_mtplx_preset(args, ["serve", str(model), "--no-thinking"])

    assert args.no_thinking is True


def test_verified_local_mtplx_qwen35b_path_respects_enable_thinking(tmp_path):
    model = tmp_path / "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"
    model.mkdir()
    (model / "mtp.safetensors").write_bytes(b"")
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"arch_id": "qwen3-next-mtp"}), encoding="utf-8"
    )
    args = _serve_args(str(model))

    _apply_qwen36_mtplx_preset(args, ["serve", str(model), "--enable-thinking"])

    assert args.no_thinking is False


def test_verified_local_mtplx_qwen35b_path_respects_log_level(tmp_path):
    model = tmp_path / "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"
    model.mkdir()
    (model / "mtp.safetensors").write_bytes(b"")
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"arch_id": "qwen3-next-mtp"}), encoding="utf-8"
    )
    args = _serve_args(str(model))
    args.log_level = "INFO"

    _apply_qwen36_mtplx_preset(args, ["serve", str(model), "--log-level", "INFO"])

    assert args.log_level == "INFO"


def test_qwen36_35b_base_path_uses_temperature_defaults():
    args = _serve_args("/models/Qwen3.6-35B-A3B-4bit")

    _apply_qwen36_35b_defaults(args, ["serve", "/models/Qwen3.6-35B-A3B-4bit"])

    assert args.default_temperature == 0.6
    assert args.default_top_p == 0.95
    assert args.enable_mtp is False


def test_qwen36_35b_base_path_respects_temperature_override():
    args = _serve_args("/models/Qwen3.6-35B-A3B-4bit")
    args.default_temperature = 0.2

    _apply_qwen36_35b_defaults(
        args,
        [
            "serve",
            "/models/Qwen3.6-35B-A3B-4bit",
            "--default-temperature",
            "0.2",
        ],
    )

    assert args.default_temperature == 0.2


def test_qwen36_preset_respects_explicit_overrides():
    args = _serve_args(
        "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        original_alias="qwen3.6-27b",
    )
    args.port = 9000
    args.served_model_name = "custom"
    args.enable_prefix_cache = True

    _apply_qwen36_mtplx_preset(
        args,
        [
            "serve",
            "qwen3.6-27b",
            "--port",
            "9000",
            "--served-model-name",
            "custom",
            "--enable-prefix-cache",
        ],
    )

    assert args.port == 9000
    assert args.served_model_name == "custom"
    assert args.disable_prefix_cache is False
    assert args.enable_mtp is True


def _ornstein_serve_args(model: str, original_alias: str | None = None) -> Namespace:
    return Namespace(
        command="serve",
        model=model,
        _original_alias=original_alias,
        enable_mtp=False,
        disable_mtp=False,
        enable_ngram=False,
        disable_ngram=False,
        prefill_step_size=8192,
        max_concurrent=1,
        max_num_seqs=256,
        prefill_batch_size=8,
        completion_batch_size=32,
        stream_interval=1,
        default_temperature=None,
        default_top_p=None,
        enable_auto_tool_choice=False,
        ngram_num_draft_tokens=0,
        ngram_min_occurrences=0,
        ngram_acceptance_mode=None,
        ngram_hybrid_verify=False,
        ngram_only_in_think=True,
        ngram_skip_tool_calls=False,
        ngram_self_tune=False,
        ngram_self_tune_disable_threshold=0.0,
        ngram_auto_disable_mtp_threshold=0.0,
        ngram_auto_disable_min_ngram=0.0,
    )


def test_ornstein_aliases_resolve_to_hf_paths():
    assert (
        resolve_model("ornstein3.6-35-saber-4bit")
        == "samuelfaj/Ornstein3.6-35B-A3B-SABER-4bit-MTPLX-Optimized-Speed"
    )
    assert (
        resolve_model("ornstein3.6-35-saber")
        == "samuelfaj/Ornstein3.6-35B-A3B-SABER-6bit-MTPLX-Optimized-Speed"
    )
    assert (
        resolve_model("ornstein3.6-35-saber-8bit")
        == "samuelfaj/Ornstein3.6-35B-A3B-SABER-8bit-MTPLX-Optimized-Speed"
    )


def test_ornstein_alias_applies_full_mtplx_ngram_preset():
    args = _ornstein_serve_args(
        "samuelfaj/Ornstein3.6-35B-A3B-SABER-6bit-MTPLX-Optimized-Speed",
        original_alias="ornstein3.6-35-saber",
    )

    _apply_ornstein_mtplx_preset(args, ["serve", "ornstein3.6-35-saber"])

    assert args.enable_mtp is True
    assert args.prefill_step_size == 32768
    assert args.max_concurrent == 3
    assert args.max_num_seqs == 1
    assert args.prefill_batch_size == 1
    assert args.completion_batch_size == 1
    assert args.stream_interval == 1
    assert args.default_temperature == 0.6
    assert args.default_top_p == 0.95
    assert args.enable_auto_tool_choice is True
    assert args.enable_ngram is True
    assert args.ngram_num_draft_tokens == 6
    assert args.ngram_min_occurrences == 2
    assert args.ngram_acceptance_mode == "greedy"
    assert args.ngram_hybrid_verify is True
    assert args.ngram_only_in_think is False
    assert args.ngram_skip_tool_calls is True
    assert args.ngram_self_tune is True
    assert args.ngram_self_tune_disable_threshold == 0.30
    assert args.ngram_auto_disable_mtp_threshold == 0.85
    assert args.ngram_auto_disable_min_ngram == 0.50


def test_ornstein_preset_respects_user_overrides():
    args = _ornstein_serve_args(
        "samuelfaj/Ornstein3.6-35B-A3B-SABER-8bit-MTPLX-Optimized-Speed",
        original_alias="ornstein3.6-35-saber-8bit",
    )
    args.prefill_step_size = 4096
    args.max_concurrent = 8
    args.default_temperature = 0.2

    _apply_ornstein_mtplx_preset(
        args,
        [
            "serve",
            "ornstein3.6-35-saber-8bit",
            "--prefill-step-size",
            "4096",
            "--max-concurrent",
            "8",
            "--default-temperature",
            "0.2",
            "--disable-mtp",
            "--disable-ngram",
        ],
    )

    assert args.prefill_step_size == 4096
    assert args.max_concurrent == 8
    assert args.default_temperature == 0.2
    assert args.enable_mtp is False
    assert args.enable_ngram is False


def test_ornstein_preset_triggers_on_hf_marker_without_alias():
    args = _ornstein_serve_args(
        "samuelfaj/Ornstein3.6-35B-A3B-SABER-4bit-MTPLX-Optimized-Speed"
    )

    _apply_ornstein_mtplx_preset(
        args,
        ["serve", "samuelfaj/Ornstein3.6-35B-A3B-SABER-4bit-MTPLX-Optimized-Speed"],
    )

    assert args.enable_mtp is True
    assert args.enable_ngram is True
    assert args.prefill_step_size == 32768


def test_qwen36_preset_respects_disable_mtp():
    args = _serve_args(
        "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        original_alias="qwen3.6-27b",
    )

    _apply_qwen36_mtplx_preset(args, ["serve", "qwen3.6-27b", "--disable-mtp"])

    assert args.enable_mtp is False
