import argparse

from vllm_mlx.cli import (
    _QWEN36_MTPLX_MODEL,
    _apply_qwen36_mtplx_preset,
)
from vllm_mlx.scheduler import SchedulerConfig


def _serve_args(**overrides):
    values = {
        "command": "serve",
        "model": _QWEN36_MTPLX_MODEL,
        "_original_alias": None,
        "enable_mtp": False,
        "served_model_name": None,
        "port": 8000,
        "default_temperature": None,
        "default_top_p": None,
        "disable_prefix_cache": False,
        "max_num_seqs": 256,
        "prefill_batch_size": 8,
        "completion_batch_size": 32,
        "prefill_step_size": 2048,
        "stream_interval": 1,
        "enable_auto_tool_choice": False,
        "tool_call_parser": None,
        "reasoning_parser": None,
        "no_thinking": False,
        "log_level": "INFO",
        "enable_tool_logits_bias": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_qwen36_mtplx_preset_uses_sustained_prefill_step_size():
    args = _serve_args()

    _apply_qwen36_mtplx_preset(args, ["serve", _QWEN36_MTPLX_MODEL])

    assert args.enable_mtp is True
    assert args.prefill_step_size == 8192
    assert args.max_num_seqs == 1
    assert args.prefill_batch_size == 1
    assert args.completion_batch_size == 1


def test_qwen36_mtplx_preset_keeps_explicit_prefill_step_size():
    args = _serve_args(prefill_step_size=4096)

    _apply_qwen36_mtplx_preset(
        args,
        ["serve", _QWEN36_MTPLX_MODEL, "--prefill-step-size", "4096"],
    )

    assert args.prefill_step_size == 4096


def test_scheduler_default_prefill_step_size_is_sustained():
    assert SchedulerConfig().prefill_step_size == 8192
