# SPDX-License-Identifier: Apache-2.0
"""Tests for the vLLM-style speculative config frontend."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from vllm_mlx.spec_decode.config import (
    SpeculativeConfigError,
    parse_speculative_config,
    require_migrated_speculative_config,
)
from vllm_mlx.spec_decode.registry import get_spec_decoder, iter_spec_decoders


def test_parse_speculative_config_accepts_vllm_common_keys() -> None:
    cfg = parse_speculative_config(
        '{"method":"mtp","model":"local/draft","num_speculative_tokens":4,'
        '"disable_auto_k":true}'
    )

    assert cfg is not None
    assert cfg.method == "mtp"
    assert cfg.model == "local/draft"
    assert cfg.num_speculative_tokens == 4
    assert cfg.disable_auto_k is True
    assert cfg.tree_budget is None


def test_parse_ddtree_speculative_config_accepts_method_keys() -> None:
    cfg = parse_speculative_config(
        '{"method":"ddtree","model":"local/draft",'
        '"num_speculative_tokens":8,"tree_budget":24}'
    )

    assert cfg is not None
    assert cfg.method == "ddtree"
    assert cfg.model == "local/draft"
    assert cfg.num_speculative_tokens == 8
    assert cfg.tree_budget == 24


def test_parse_dflash_speculative_config_accepts_drafter_model() -> None:
    cfg = parse_speculative_config(
        '{"method":"dflash","model":"z-lab/Qwen3.5-27B-DFlash"}'
    )

    assert cfg is not None
    assert cfg.method == "dflash"
    assert cfg.model == "z-lab/Qwen3.5-27B-DFlash"
    assert cfg.num_speculative_tokens is None
    assert cfg.tree_budget is None


def test_parse_speculative_config_normalizes_registered_alias() -> None:
    cfg = parse_speculative_config('{"method":"ngram"}')

    assert cfg is not None
    assert cfg.method == "suffix"


def test_parse_suffix_speculative_config_accepts_existing_knobs() -> None:
    cfg = parse_speculative_config(
        '{"method":"suffix","num_speculative_tokens":6,'
        '"max_suffix_len":5,"min_confidence":0.4,"min_draft_len":3}'
    )

    assert cfg is not None
    assert cfg.method == "suffix"
    assert cfg.num_speculative_tokens == 6
    assert cfg.max_suffix_len == 5
    assert cfg.min_confidence == 0.4
    assert cfg.min_draft_len == 3


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("", "cannot be empty"),
        ("[]", "JSON object"),
        ("{bad", "valid JSON"),
        ('{"model":"x"}', "requires string key 'method'"),
        ('{"method":"mtp","num_speculative_tokens":0}', "positive integer"),
        ('{"method":"mtp","num_speculative_tokens":true}', "positive integer"),
        ('{"method":"mtp","model":""}', "non-empty string"),
        ('{"method":"mtp","tree_budget":24}', "unsupported speculative-config"),
        ('{"method":"mtp","disable_auto_k":1}', "boolean"),
        ('{"method":"ddtree","tree_budget":0}', "positive integer"),
        ('{"method":"ddtree","tree_budget":true}', "positive integer"),
        ('{"method":"ddtree","unknown":1}', "unsupported speculative-config"),
        ('{"method":"suffix","max_suffix_len":0}', "positive integer"),
        ('{"method":"suffix","min_confidence":0}', "positive number"),
        ('{"method":"suffix","min_confidence":true}', "positive number"),
        ('{"method":"suffix","min_confidence":NaN}', "positive number"),
        ('{"method":"suffix","min_confidence":Infinity}', "positive number"),
        ('{"method":"suffix","min_confidence":1.5}', "between 0 and 1"),
        ('{"method":"suffix","min_draft_len":0}', "positive integer"),
        ('{"method":"suffix","model":"x"}', "unsupported speculative-config"),
        ('{"method":"suffix","tree_budget":24}', "unsupported speculative-config"),
        (
            '{"method":"dflash","num_speculative_tokens":4}',
            "unsupported speculative-config",
        ),
        ('{"method":"dflash","tree_budget":24}', "unsupported speculative-config"),
        ('{"method":"unknown"}', "unsupported speculative decoding method"),
    ],
)
def test_parse_speculative_config_rejects_bad_payloads(raw: str, match: str) -> None:
    with pytest.raises(SpeculativeConfigError, match=match):
        parse_speculative_config(raw)


def test_require_migrated_speculative_config_accepts_mtp() -> None:
    cfg = parse_speculative_config('{"method":"mtp"}')
    assert cfg is not None

    require_migrated_speculative_config(cfg)


def test_require_migrated_speculative_config_accepts_ddtree() -> None:
    cfg = parse_speculative_config('{"method":"ddtree"}')
    assert cfg is not None

    require_migrated_speculative_config(cfg)


def test_require_migrated_speculative_config_accepts_dflash() -> None:
    cfg = parse_speculative_config('{"method":"dflash"}')
    assert cfg is not None

    require_migrated_speculative_config(cfg)


def test_require_migrated_speculative_config_accepts_suffix() -> None:
    cfg = parse_speculative_config('{"method":"suffix"}')
    assert cfg is not None

    require_migrated_speculative_config(cfg)


def test_legacy_config_helpers_warn_and_return_configs() -> None:
    from vllm_mlx.spec_decode.config import (
        legacy_ddtree_config,
        legacy_dflash_config,
        legacy_mtp_config,
        legacy_suffix_config,
    )

    with pytest.warns(DeprecationWarning, match="legacy_ddtree_config"):
        ddtree = legacy_ddtree_config()
    with pytest.warns(DeprecationWarning, match="legacy_dflash_config"):
        dflash = legacy_dflash_config("local/draft")
    with pytest.warns(DeprecationWarning, match="legacy_mtp_config"):
        mtp = legacy_mtp_config(
            model="local/assistant",
            num_speculative_tokens=2,
            disable_auto_k=True,
        )
    with pytest.warns(DeprecationWarning, match="legacy_suffix_config"):
        suffix = legacy_suffix_config(
            num_speculative_tokens=6,
            max_suffix_len=5,
            min_confidence=0.4,
            min_draft_len=3,
        )

    assert ddtree.method == "ddtree"
    assert dflash.method == "dflash"
    assert dflash.model == "local/draft"
    assert mtp.method == "mtp"
    assert mtp.model == "local/assistant"
    assert mtp.num_speculative_tokens == 2
    assert mtp.disable_auto_k is True
    assert suffix.method == "suffix"
    assert suffix.num_speculative_tokens == 6
    assert suffix.max_suffix_len == 5
    assert suffix.min_confidence == 0.4
    assert suffix.min_draft_len == 3


def test_spec_decoder_registry_lists_existing_backends() -> None:
    methods = {plugin.method for plugin in iter_spec_decoders()}

    assert {"ddtree", "dflash", "mtp", "suffix"}.issubset(methods)
    assert get_spec_decoder("ddtree").config_enabled is True
    assert get_spec_decoder("dflash").config_enabled is True
    assert get_spec_decoder("mtp").config_enabled is True
    assert get_spec_decoder("suffix").config_enabled is True
    assert get_spec_decoder("ngram") == get_spec_decoder("suffix")


def test_serve_help_exposes_speculative_config() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--speculative-config" in proc.stdout


def _spec_config_args(**overrides):
    data = {
        "speculative_config": None,
        "enable_ddtree": False,
        "enable_dflash": False,
        "spec_decode": "none",
        "dflash_drafter_path": "",
        "mtp_num_draft_tokens": 1,
        "mtp_optimistic": False,
        "mtp_sidecar": None,
        "mtp_max_k": None,
        "mtp_disable_auto_k": False,
        "suffix_decoding": False,
        "suffix_max_draft": None,
        "suffix_max_suffix_len": None,
        "suffix_min_confidence": None,
        "suffix_min_draft_len": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_speculative_config_mtp_normalizes_to_legacy_spec_decode() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(
        speculative_config=(
            '{"method":"mtp","model":"google/gemma-4-12B-it-assistant",'
            '"num_speculative_tokens":2,"disable_auto_k":true}'
        )
    )

    _normalize_speculative_config_or_exit(args)

    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar == "google/gemma-4-12B-it-assistant"
    assert args.mtp_max_k == 2
    assert args.mtp_disable_auto_k is True
    assert args._speculative_config.method == "mtp"
    assert args._speculative_config.model == "google/gemma-4-12B-it-assistant"
    assert args._speculative_config.num_speculative_tokens == 2
    assert args._speculative_config.disable_auto_k is True


def test_speculative_config_mtp_populates_runtime_args() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    config_args = _spec_config_args(
        speculative_config=(
            '{"method":"mtp","model":"google/gemma-4-12B-it-assistant",'
            '"num_speculative_tokens":2,"disable_auto_k":true}'
        )
    )

    _normalize_speculative_config_or_exit(config_args)

    assert config_args.spec_decode == "mtp"
    assert config_args.mtp_sidecar == "google/gemma-4-12B-it-assistant"
    assert config_args.mtp_max_k == 2
    assert config_args.mtp_disable_auto_k is True
    assert config_args.suffix_decoding is False
    assert config_args.enable_dflash is False
    assert config_args.enable_ddtree is False


def test_speculative_config_mtp_without_token_count_keeps_legacy_one_token() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(speculative_config='{"method":"mtp"}')

    _normalize_speculative_config_or_exit(args)

    assert args.spec_decode == "mtp"
    assert args.mtp_max_k == 1
    assert args.mtp_disable_auto_k is False


def test_speculative_config_parse_none_cleanly_disables(monkeypatch) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit
    from vllm_mlx.spec_decode import config as config_mod

    monkeypatch.setattr(config_mod, "parse_speculative_config", lambda _raw: None)
    args = _spec_config_args(speculative_config='{"method":"mtp"}')

    _normalize_speculative_config_or_exit(args)

    assert args._speculative_config is None
    assert args.spec_decode == "none"
    assert args.enable_mtp is False
    assert args.mtp_max_k == 1


def test_no_speculative_config_fills_suffix_runtime_defaults() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args()

    _normalize_speculative_config_or_exit(args)

    assert args._speculative_config is None
    assert args.mtp_max_k == 1
    assert args.suffix_max_draft == 8
    assert args.suffix_max_suffix_len == 4
    assert args.suffix_min_confidence == 0.3
    assert args.suffix_min_draft_len == 2


def test_no_speculative_config_preserves_programmatic_runtime_fields() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(
        enable_dflash=True,
        dflash_drafter_path="local/draft",
    )

    _normalize_speculative_config_or_exit(args)

    assert args._speculative_config.method == "dflash"
    assert args.enable_dflash is True
    assert args.dflash_drafter_path == "local/draft"


@pytest.mark.parametrize(
    ("overrides", "method"),
    [
        ({"enable_dflash": True}, "dflash"),
        ({"enable_ddtree": True}, "ddtree"),
        ({"spec_decode": "mtp"}, "mtp"),
        ({"enable_mtp": True}, "mtp"),
        ({"suffix_decoding": True}, "suffix"),
    ],
)
def test_hidden_legacy_aliases_normalize_to_speculative_config(
    overrides, method
) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(**overrides)

    _normalize_speculative_config_or_exit(args)

    assert args._speculative_config is not None
    assert args._speculative_config.method == method


def test_hidden_legacy_enable_mtp_preserves_compat_marker() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(enable_mtp=True, mtp_max_k=2)

    _normalize_speculative_config_or_exit(args)

    assert args._speculative_config.method == "mtp"
    assert args.spec_decode == "mtp"
    assert args.enable_mtp is True
    assert args.mtp_num_draft_tokens == 2
    assert args.mtp_max_k == 2


def test_hidden_legacy_aliases_reject_multiple_methods(capsys) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(enable_dflash=True, suffix_decoding=True)

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    assert "select multiple methods" in capsys.readouterr().err


def test_hidden_legacy_mtp_optimistic_rejects_enable_mtp(capsys) -> None:
    """PR #1050 hard-reject: ``--enable-mtp --mtp-optimistic`` is no longer
    accepted (previously silently ignored under the vendored installer)."""
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(enable_mtp=True, mtp_optimistic=True)

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "mtp_optimistic" in captured.err
    assert "not supported under the unified spec-decode" in captured.err


def test_hidden_legacy_mtp_optimistic_rejects_migrated_mtp(capsys) -> None:
    """PR #1050 hard-reject: ``--spec-decode=mtp --mtp-optimistic`` fails."""
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(spec_decode="mtp", mtp_optimistic=True)

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "mtp_optimistic" in captured.err
    assert "not supported under the unified spec-decode" in captured.err


def test_hidden_legacy_mtp_token_count_aliases_reject_conflict(capsys) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(enable_mtp=True, mtp_max_k=2, mtp_num_draft_tokens=3)

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "mtp_max_k and mtp_num_draft_tokens conflict" in captured.err


@pytest.mark.parametrize(
    ("overrides", "knob"),
    [
        ({"dflash_drafter_path": "local/draft"}, "dflash_drafter_path"),
        ({"mtp_sidecar": "local/assistant"}, "mtp_sidecar"),
        ({"mtp_max_k": 2}, "mtp_max_k"),
        ({"mtp_num_draft_tokens": 2}, "mtp_num_draft_tokens"),
        ({"mtp_disable_auto_k": True}, "mtp_disable_auto_k"),
        ({"mtp_optimistic": True}, "mtp_optimistic"),
        ({"suffix_max_draft": 6}, "suffix_max_draft"),
        ({"suffix_max_suffix_len": 5}, "suffix_max_suffix_len"),
        ({"suffix_min_confidence": 0.4}, "suffix_min_confidence"),
        ({"suffix_min_draft_len": 3}, "suffix_min_draft_len"),
    ],
)
def test_hidden_legacy_tuning_knobs_require_method_selector(
    overrides, knob, capsys
) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(**overrides)

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "requires" in captured.err
    assert knob in captured.err


@pytest.mark.parametrize(
    ("overrides", "conflict"),
    [
        ({"enable_ddtree": True}, "enable_ddtree"),
        ({"enable_dflash": True}, "enable_dflash"),
        ({"spec_decode": "mtp"}, "spec_decode=mtp"),
        ({"dflash_drafter_path": "local/draft"}, "dflash_drafter_path"),
        ({"enable_mtp": True}, "enable_mtp"),
        ({"mtp_sidecar": "local/assistant"}, "mtp_sidecar"),
        ({"mtp_max_k": 2}, "mtp_max_k"),
        ({"mtp_num_draft_tokens": 2}, "mtp_num_draft_tokens"),
        ({"mtp_disable_auto_k": True}, "mtp_disable_auto_k"),
        ({"mtp_optimistic": True}, "mtp_optimistic"),
        ({"suffix_decoding": True}, "suffix_decoding"),
        ({"suffix_max_draft": 6}, "suffix_max_draft"),
        ({"suffix_max_suffix_len": 5}, "suffix_max_suffix_len"),
        ({"suffix_min_confidence": 0.4}, "suffix_min_confidence"),
        ({"suffix_min_draft_len": 3}, "suffix_min_draft_len"),
    ],
)
def test_speculative_config_rejects_legacy_alias_conflicts(
    overrides, conflict, capsys
) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(speculative_config='{"method":"mtp"}', **overrides)

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "--speculative-config is mutually exclusive" in captured.err
    assert conflict in captured.err


@pytest.mark.parametrize(
    ("overrides", "conflict"),
    [
        ({"enable_ddtree": True}, "enable_ddtree"),
        ({"enable_dflash": True}, "enable_dflash"),
        ({"spec_decode": "mtp"}, "spec_decode=mtp"),
        ({"dflash_drafter_path": "local/draft"}, "dflash_drafter_path"),
        ({"enable_mtp": True}, "enable_mtp"),
        ({"mtp_sidecar": "local/assistant"}, "mtp_sidecar"),
        ({"mtp_max_k": 2}, "mtp_max_k"),
        ({"mtp_num_draft_tokens": 2}, "mtp_num_draft_tokens"),
        ({"mtp_disable_auto_k": True}, "mtp_disable_auto_k"),
        ({"mtp_optimistic": True}, "mtp_optimistic"),
        ({"suffix_decoding": True}, "suffix_decoding"),
        ({"suffix_max_draft": 6}, "suffix_max_draft"),
        ({"suffix_max_suffix_len": 5}, "suffix_max_suffix_len"),
        ({"suffix_min_confidence": 0.4}, "suffix_min_confidence"),
        ({"suffix_min_draft_len": 3}, "suffix_min_draft_len"),
    ],
)
def test_no_spec_decode_rejects_programmatic_runtime_fields(
    overrides, conflict, capsys
) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(no_spec_decode=True, **overrides)

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "--no-spec-decode is mutually exclusive" in captured.err
    assert conflict in captured.err


def test_speculative_config_malformed_reports_clean_error(capsys) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(speculative_config="")

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "cannot be empty" in captured.err
    assert "AttributeError" not in captured.err


def test_speculative_config_rejects_no_spec_decode(capsys) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(
        speculative_config='{"method":"mtp"}',
        no_spec_decode=True,
    )

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err
    assert "--no-spec-decode" in captured.err


def test_speculative_config_suffix_normalizes_to_legacy_suffix_args() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(
        speculative_config=(
            '{"method":"suffix","num_speculative_tokens":6,'
            '"max_suffix_len":5,"min_confidence":0.4,"min_draft_len":3}'
        )
    )

    _normalize_speculative_config_or_exit(args)

    assert args.suffix_decoding is True
    assert args.suffix_max_draft == 6
    assert args.suffix_max_suffix_len == 5
    assert args.suffix_min_confidence == 0.4
    assert args.suffix_min_draft_len == 3
    assert args._speculative_config.method == "suffix"
