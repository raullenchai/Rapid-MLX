# SPDX-License-Identifier: Apache-2.0
"""Tests for lightning-mlx CLI model presets."""

from argparse import Namespace
import json

from vllm_mlx.cli import _apply_qwen36_mtplx_preset


def _serve_args(model: str, original_alias: str | None = None) -> Namespace:
    return Namespace(
        command="serve",
        model=model,
        _original_alias=original_alias,
        enable_mtp=False,
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
    assert args.no_thinking is True
    assert args.enable_tool_logits_bias is True


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
    assert args.no_thinking is True
    assert args.enable_tool_logits_bias is True


def test_verified_local_mtplx_qwen35b_path_keeps_thinking_enabled(tmp_path):
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
    assert args.no_thinking is False
    assert args.log_level == "WARNING"
    assert args.enable_tool_logits_bias is True


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
