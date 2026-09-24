"""Speculative-selection provenance for the Qwen auto-runtime planner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.qwen_runtime_plan import SpeculativeIntent
from rapid_mlx.spec_decode import config as spec_config


def _args(model: str, **overrides) -> SimpleNamespace:
    values = {
        "model": model,
        "speculative_config": None,
        "enable_ddtree": False,
        "enable_dflash": False,
        "enable_mtp": False,
        "no_spec_decode": False,
        "spec_decode": "none",
        "dflash_drafter_path": "",
        "mtp_num_draft_tokens": 1,
        "mtp_optimistic": False,
        "mtp_sidecar": None,
        "mtp_max_k": None,
        "mtp_disable_auto_k": False,
        "mtp_backend": None,
        "force_spec_decode": False,
        "suffix_decoding": False,
        "suffix_max_draft": None,
        "suffix_max_suffix_len": None,
        "suffix_min_confidence": None,
        "suffix_min_draft_len": None,
        "mllm": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("alias", "expected_k"),
    [
        ("qwen3.6-35b-4bit", 2),
        ("qwen3.8-27b-4bit", 3),
    ],
)
def test_qwen_alias_default_records_implicit_source_without_behavior_change(
    alias: str, expected_k: int
) -> None:
    args = _args(alias)

    cli._normalize_speculative_config_or_exit(args)

    assert args._speculative_config_source == "alias_default"
    assert args._speculative_config.method == "mtp"
    assert args.spec_decode == "mtp"
    assert args.mtp_max_k == expected_k


def test_native_mtp_ready_alias_records_implicit_native_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args("qwen3.6-35b-4bit")
    monkeypatch.setattr(cli, "_alias_native_mtp_capable", lambda _model: True)
    monkeypatch.setattr(cli, "_native_mtp_runtime_ready", lambda _model: True)

    cli._normalize_speculative_config_or_exit(args)

    assert args._speculative_config_source == "alias_default"
    assert args._speculative_config.method == "mtp"
    assert args._speculative_config.backend == "native"


def test_explicit_speculative_config_records_operator_source() -> None:
    args = _args(
        "qwen3.6-35b-4bit",
        speculative_config=(
            '{"method":"mtp","num_speculative_tokens":1,"continuous_batching":false}'
        ),
    )

    cli._normalize_speculative_config_or_exit(args)

    assert args._speculative_config_source == "explicit_config"
    assert args._speculative_config.method == "mtp"
    assert args.mtp_max_k == 1
    assert args.mtp_continuous_batching is False


def test_legacy_enable_mtp_records_legacy_source() -> None:
    args = _args("qwen3.6-35b-4bit", enable_mtp=True, mtp_max_k=1)

    cli._normalize_speculative_config_or_exit(args)

    assert args._speculative_config_source == "legacy_flags"
    assert args._speculative_config.method == "mtp"
    assert args.enable_mtp is True
    assert args.mtp_max_k == 1


@pytest.mark.parametrize("alias", ["qwen3.6-35b-4bit", "qwen3.8-27b-4bit"])
def test_explicit_mllm_suppresses_alias_default_and_records_none(alias: str) -> None:
    args = _args(alias, mllm=True)

    cli._normalize_speculative_config_or_exit(args)

    assert args._speculative_config_source == "none"
    assert args._speculative_config is None
    assert args.spec_decode == "none"


@pytest.mark.parametrize("alias", ["qwen3.6-35b-4bit", "qwen3.8-27b-4bit"])
def test_no_spec_decode_suppresses_alias_default_and_records_none(alias: str) -> None:
    args = _args(alias, no_spec_decode=True)

    cli._normalize_speculative_config_or_exit(args)

    assert args._speculative_config_source == "none"
    assert args._speculative_config is None
    assert args.spec_decode == "none"


def test_unknown_alias_records_none_and_preserves_plain_decode() -> None:
    args = _args("someone/unknown-qwen")

    cli._normalize_speculative_config_or_exit(args)

    assert args._speculative_config_source == "none"
    assert args._speculative_config is None
    assert args.spec_decode == "none"


def test_parser_none_result_normalizes_provenance_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args("someone/unknown-qwen", speculative_config='{"method":"mtp"}')
    monkeypatch.setattr(spec_config, "parse_speculative_config", lambda _raw: None)

    cli._normalize_speculative_config_or_exit(args)

    assert args._speculative_config_source == "none"
    assert args._speculative_config is None


@pytest.mark.parametrize(
    ("source", "no_spec_decode", "expected"),
    [
        ("none", False, SpeculativeIntent.NONE),
        ("none", True, SpeculativeIntent.EXPLICIT_DISABLED),
        ("alias_default", False, SpeculativeIntent.ALIAS_DEFAULT),
        ("explicit_config", False, SpeculativeIntent.EXPLICIT_ENABLED),
        ("legacy_flags", False, SpeculativeIntent.EXPLICIT_ENABLED),
    ],
)
def test_qwen_planner_intent_maps_normalized_provenance(
    source: str,
    no_spec_decode: bool,
    expected: SpeculativeIntent,
) -> None:
    args = _args("qwen3.6-35b-4bit", no_spec_decode=no_spec_decode)
    args._speculative_config_source = source

    assert cli._qwen_speculative_intent(args) is expected


def test_qwen_planner_intent_rejects_unknown_internal_source() -> None:
    args = _args("qwen3.6-35b-4bit")
    args._speculative_config_source = "future_unmapped_source"

    with pytest.raises(ValueError, match="unknown speculative config source"):
        cli._qwen_speculative_intent(args)
