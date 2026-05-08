# SPDX-License-Identifier: Apache-2.0
"""Tests for `_apply_alias_profile_to_args` — the helper that bridges
``AliasProfile`` (and the regex fallback ``ModelConfig``) into the serve
command's argparse Namespace.

Pins the contract:
- tier ``agent`` or ``structured`` + non-hybrid + user didn't override
  → ``args.suffix_decoding`` becomes True
- tier ``neutral`` / ``avoid`` / ``unknown`` → no change
- ``supports_spec_decode=False`` (hybrid) → no change even if tier=agent
- ``--no-suffix-decoding`` opt-out → no change even on agent tier
- ``--suffix-decoding`` already set → no change (already enabled)
- ``auto_config=None`` (unknown alias) → no-op, no crash

Plus the existing parser auto-detect contract (tool_call + reasoning).
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx import cli
from vllm_mlx.model_auto_config import ModelConfig


def _args(**overrides) -> SimpleNamespace:
    """Minimal Namespace covering only the fields the helper reads."""
    base = SimpleNamespace(
        tool_call_parser=None,
        reasoning_parser=None,
        enable_auto_tool_choice=False,
        no_thinking=False,
        suffix_decoding=False,
        no_suffix_decoding=False,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        tool_call_parser="hermes",
        reasoning_parser=None,
        is_hybrid=False,
        supports_spec_decode=True,
        suffix_decoding_tier="unknown",
    )
    base.update(overrides)
    return ModelConfig(**base)


_logger = logging.getLogger("test")


# ---------------------------------------------------------------------- happy path


def test_agent_tier_auto_enables_suffix_decoding():
    args = _args()
    cli._apply_alias_profile_to_args(args, _cfg(suffix_decoding_tier="agent"), _logger)
    assert args.suffix_decoding is True


def test_structured_tier_auto_enables_suffix_decoding():
    args = _args()
    cli._apply_alias_profile_to_args(
        args, _cfg(suffix_decoding_tier="structured"), _logger
    )
    assert args.suffix_decoding is True


# ---------------------------------------------------------------------- silent tiers


@pytest.mark.parametrize("tier", ["neutral", "avoid", "unknown"])
def test_quiet_tiers_do_not_auto_enable(tier):
    args = _args()
    cli._apply_alias_profile_to_args(args, _cfg(suffix_decoding_tier=tier), _logger)
    assert args.suffix_decoding is False


# ---------------------------------------------------------------------- hybrid gate


def test_hybrid_arch_blocks_auto_enable_even_at_agent_tier():
    """Belt + suspenders: a hybrid model (supports_spec_decode=False) must
    keep the flag off even if tier somehow became ``agent``. The framework
    gate is the floor; the CLI must not fight it."""
    args = _args()
    cfg = _cfg(
        suffix_decoding_tier="agent",
        is_hybrid=True,
        supports_spec_decode=False,
    )
    cli._apply_alias_profile_to_args(args, cfg, _logger)
    assert args.suffix_decoding is False


# ---------------------------------------------------------------------- explicit overrides


def test_no_suffix_decoding_opts_out_of_auto_enable():
    args = _args(no_suffix_decoding=True)
    cli._apply_alias_profile_to_args(args, _cfg(suffix_decoding_tier="agent"), _logger)
    assert args.suffix_decoding is False


def test_already_enabled_remains_enabled_for_any_tier():
    """If user passed --suffix-decoding, the auto-apply must NOT toggle it
    off for any tier value (it would override explicit user intent)."""
    for tier in ["agent", "structured", "neutral", "avoid", "unknown"]:
        args = _args(suffix_decoding=True)
        cli._apply_alias_profile_to_args(args, _cfg(suffix_decoding_tier=tier), _logger)
        assert args.suffix_decoding is True, f"tier={tier} flipped suffix_decoding off"


# ---------------------------------------------------------------------- robustness


def test_none_config_is_noop():
    """If detect_model_config returned None (unknown alias / non-string),
    helper must not crash and must leave args untouched."""
    args = _args()
    cli._apply_alias_profile_to_args(args, None, _logger)
    assert args.suffix_decoding is False
    assert args.tool_call_parser is None
    assert args.reasoning_parser is None
    assert args.enable_auto_tool_choice is False


# ---------------------------------------------------------------------- parser auto-detect (regression)


def test_parser_autodetect_sets_tool_call_and_reasoning():
    args = _args()
    cli._apply_alias_profile_to_args(
        args, _cfg(tool_call_parser="hermes", reasoning_parser="qwen3"), _logger
    )
    assert args.tool_call_parser == "hermes"
    assert args.reasoning_parser == "qwen3"
    assert args.enable_auto_tool_choice is True


def test_parser_autodetect_respects_existing_tool_call():
    """If user explicitly passed --tool-call-parser, helper must not
    override it."""
    args = _args(tool_call_parser="custom")
    cli._apply_alias_profile_to_args(args, _cfg(tool_call_parser="hermes"), _logger)
    assert args.tool_call_parser == "custom"
    # enable_auto_tool_choice must NOT be flipped on by the helper when the
    # parser was user-set; the existing serve flow handles that.
    assert args.enable_auto_tool_choice is False


def test_parser_autodetect_respects_no_thinking():
    """--no-thinking must suppress the reasoning_parser auto-set."""
    args = _args(no_thinking=True)
    cli._apply_alias_profile_to_args(args, _cfg(reasoning_parser="qwen3"), _logger)
    assert args.reasoning_parser is None


# ---------------------------------------------------------------------- argparse smoke


def test_help_advertises_no_suffix_decoding_flag():
    """The CLI exposes --no-suffix-decoding as a registered flag."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0


def test_no_suffix_decoding_argparse_stores_true():
    """`rapid-mlx serve <model> --no-suffix-decoding` parses to
    args.no_suffix_decoding=True (argparse hyphen→underscore)."""
    parser_args = [
        "rapid-mlx",
        "serve",
        "qwen3.5-4b",
        "--no-suffix-decoding",
    ]

    captured: dict = {}

    def fake_serve(args):
        captured["ns"] = args

    with (
        patch.object(sys, "argv", parser_args),
        patch.object(cli, "serve_command", side_effect=fake_serve),
    ):
        # main() resolves the alias and dispatches; we capture the args.
        cli.main()

    assert captured["ns"].no_suffix_decoding is True
    assert captured["ns"].suffix_decoding is False
