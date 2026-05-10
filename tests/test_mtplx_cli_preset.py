import argparse

from vllm_mlx.cli import (
    _QWEN36_35B_8BIT_MTPLX_MODEL,
    _QWEN36_35B_MTPLX_MODEL,
    _QWEN36_MTPLX_MODEL,
    _apply_qwen36_35b_defaults,
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
        "mtp_num_draft_tokens": 1,
        "mtp_optimistic": False,
        "stream_interval": 1,
        "enable_auto_tool_choice": False,
        "tool_call_parser": None,
        "reasoning_parser": None,
        "no_thinking": False,
        "log_level": "INFO",
        "enable_tool_logits_bias": False,
        # N-gram defaults (preset overrides these for 35B-A3B).
        "enable_ngram": False,
        "ngram_num_draft_tokens": 4,
        "ngram_size": 3,
        "ngram_min_matches": 2,
        "ngram_only_in_think": True,
        "ngram_acceptance_mode": "greedy",
        "ngram_min_occurrences": 1,
        "ngram_adaptive_k": True,
        "ngram_auto_disable_mtp_threshold": 0.0,
        "ngram_auto_disable_min_ngram": 0.50,
        "ngram_hybrid_verify": False,
        "ngram_skip_tool_calls": True,
        "ngram_self_tune": True,
        "ngram_self_tune_disable_threshold": 0.30,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _bench_args(**overrides):
    values = {
        "command": "bench",
        "model": _QWEN36_MTPLX_MODEL,
        "_original_alias": None,
        "enable_mtp": False,
        "disable_prefix_cache": False,
        "max_num_seqs": 32,
        "prefill_batch_size": 8,
        "completion_batch_size": 16,
        "prefill_step_size": 2048,
        "mtp_num_draft_tokens": 1,
        "mtp_optimistic": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_qwen36_mtplx_serve_preset_uses_agentic_defaults():
    args = _serve_args()

    _apply_qwen36_mtplx_preset(args, ["serve", _QWEN36_MTPLX_MODEL])

    assert args.enable_mtp is True
    assert args.prefill_step_size == 2048
    assert args.mtp_num_draft_tokens == 3
    assert args.mtp_optimistic is False
    assert args.max_num_seqs == 1
    assert args.prefill_batch_size == 1
    assert args.completion_batch_size == 1
    assert args.no_thinking is False


def test_qwen36_mtplx_bench_preset_enables_mtp():
    args = _bench_args()

    _apply_qwen36_mtplx_preset(args, ["bench", _QWEN36_MTPLX_MODEL])

    assert args.enable_mtp is True
    assert args.disable_prefix_cache is True
    assert args.prefill_step_size == 8192
    assert args.mtp_num_draft_tokens == 3
    assert args.mtp_optimistic is True
    assert args.max_num_seqs == 1
    assert args.prefill_batch_size == 1
    assert args.completion_batch_size == 1


def test_qwen36_35b_mtplx_serve_preset_uses_agentic_defaults():
    args = _serve_args(model=_QWEN36_35B_MTPLX_MODEL)

    _apply_qwen36_mtplx_preset(args, ["serve", _QWEN36_35B_MTPLX_MODEL])

    assert args.enable_mtp is True
    assert args.disable_prefix_cache is True
    assert args.max_num_seqs == 1
    assert args.prefill_batch_size == 1
    assert args.completion_batch_size == 1
    assert args.prefill_step_size == 2048
    assert args.mtp_num_draft_tokens == 1
    assert args.mtp_optimistic is False
    # Thinking is enabled by default for 35B agentic correctness.
    assert args.no_thinking is False


def test_qwen36_mtplx_bench_preset_keeps_disabled_mtp():
    args = _bench_args()

    _apply_qwen36_mtplx_preset(
        args,
        ["bench", _QWEN36_MTPLX_MODEL, "--disable-mtp"],
    )

    assert args.enable_mtp is False


def test_qwen36_mtplx_bench_preset_keeps_explicit_mtp_options():
    args = _bench_args(mtp_num_draft_tokens=2, mtp_optimistic=False)

    _apply_qwen36_mtplx_preset(
        args,
        [
            "bench",
            _QWEN36_MTPLX_MODEL,
            "--mtp-num-draft-tokens",
            "2",
            "--mtp-optimistic",
        ],
    )

    assert args.mtp_num_draft_tokens == 2
    assert args.mtp_optimistic is False


def test_qwen36_mtplx_preset_keeps_explicit_prefill_step_size():
    args = _serve_args(prefill_step_size=4096)

    _apply_qwen36_mtplx_preset(
        args,
        ["serve", _QWEN36_MTPLX_MODEL, "--prefill-step-size", "4096"],
    )

    assert args.prefill_step_size == 4096


def test_scheduler_default_prefill_step_size_is_sustained():
    assert SchedulerConfig().prefill_step_size == 8192


def test_qwen36_35b_serve_preset_enables_ngram_with_tuned_defaults():
    args = _serve_args(model=_QWEN36_35B_MTPLX_MODEL)

    _apply_qwen36_mtplx_preset(args, ["serve", _QWEN36_35B_MTPLX_MODEL])

    # N-gram is auto-enabled for 35B-A3B with the validated agentic
    # configuration.
    assert args.enable_ngram is True
    assert args.ngram_num_draft_tokens == 6
    assert args.ngram_min_occurrences == 2
    assert args.ngram_acceptance_mode == "greedy"
    assert args.ngram_hybrid_verify is True
    assert args.ngram_only_in_think is False  # everywhere
    assert args.ngram_skip_tool_calls is True
    assert args.ngram_self_tune is True
    assert args.ngram_self_tune_disable_threshold == 0.30
    assert args.ngram_auto_disable_mtp_threshold == 0.85
    assert args.ngram_auto_disable_min_ngram == 0.50


def test_qwen36_35b_serve_preset_disable_ngram_flag_overrides():
    args = _serve_args(model=_QWEN36_35B_MTPLX_MODEL)

    _apply_qwen36_mtplx_preset(
        args,
        ["serve", _QWEN36_35B_MTPLX_MODEL, "--disable-ngram"],
    )

    assert args.enable_ngram is False


def test_qwen36_35b_serve_preset_keeps_explicit_ngram_overrides():
    args = _serve_args(
        model=_QWEN36_35B_MTPLX_MODEL,
        ngram_num_draft_tokens=8,
        ngram_min_occurrences=4,
        ngram_hybrid_verify=False,
    )

    _apply_qwen36_mtplx_preset(
        args,
        [
            "serve",
            _QWEN36_35B_MTPLX_MODEL,
            "--ngram-num-draft-tokens",
            "8",
            "--ngram-min-occurrences",
            "4",
        ],
    )

    assert args.ngram_num_draft_tokens == 8
    assert args.ngram_min_occurrences == 4
    # User did NOT pass --ngram-hybrid-verify, so the preset still flips
    # it on (the existing hybrid_verify=False in args is the parser's
    # default, not an explicit override).
    assert args.ngram_hybrid_verify is True


def test_qwen36_35b_serve_preset_no_hybrid_verify_overrides():
    args = _serve_args(
        model=_QWEN36_35B_MTPLX_MODEL,
        ngram_hybrid_verify=False,
    )

    _apply_qwen36_mtplx_preset(
        args,
        [
            "serve",
            _QWEN36_35B_MTPLX_MODEL,
            "--no-ngram-hybrid-verify",
        ],
    )

    assert args.ngram_hybrid_verify is False


def test_qwen36_35b_8bit_alias_matches_4bit_preset():
    """8bit alias must apply identical defaults — only model differs."""
    a = _serve_args(
        model=_QWEN36_35B_MTPLX_MODEL, _original_alias="qwen3.6-35b"
    )
    b = _serve_args(
        model=_QWEN36_35B_8BIT_MTPLX_MODEL,
        _original_alias="qwen3.6-35b-8bit",
    )

    _apply_qwen36_mtplx_preset(a, ["serve", "qwen3.6-35b"])
    _apply_qwen36_35b_defaults(a, ["serve", "qwen3.6-35b"])
    _apply_qwen36_mtplx_preset(b, ["serve", "qwen3.6-35b-8bit"])
    _apply_qwen36_35b_defaults(b, ["serve", "qwen3.6-35b-8bit"])

    da, db = vars(a), vars(b)
    diffs = {k: (da[k], db[k]) for k in da if da[k] != db[k]}
    assert diffs == {
        "model": (_QWEN36_35B_MTPLX_MODEL, _QWEN36_35B_8BIT_MTPLX_MODEL),
        "_original_alias": ("qwen3.6-35b", "qwen3.6-35b-8bit"),
    }


def test_qwen36_27b_serve_preset_does_not_enable_ngram():
    """27B model should not get the 35B-only ngram preset."""
    args = _serve_args(model=_QWEN36_MTPLX_MODEL)

    _apply_qwen36_mtplx_preset(args, ["serve", _QWEN36_MTPLX_MODEL])

    assert args.enable_ngram is False
    assert args.ngram_num_draft_tokens == 4  # parser default unchanged
    assert args.ngram_min_occurrences == 1
    assert args.ngram_hybrid_verify is False
