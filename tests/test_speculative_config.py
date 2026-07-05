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
        '{"method":"mtp","model":"local/draft","num_speculative_tokens":4}'
    )

    assert cfg is not None
    assert cfg.method == "mtp"
    assert cfg.model == "local/draft"
    assert cfg.num_speculative_tokens == 4
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
        ('{"method":"ddtree","tree_budget":0}', "positive integer"),
        ('{"method":"ddtree","tree_budget":true}', "positive integer"),
        ('{"method":"ddtree","unknown":1}', "unsupported speculative-config"),
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


def test_spec_decoder_registry_lists_existing_backends() -> None:
    methods = {plugin.method for plugin in iter_spec_decoders()}

    assert {"ddtree", "dflash", "mtp", "suffix"}.issubset(methods)
    assert get_spec_decoder("ddtree").config_enabled is True
    assert get_spec_decoder("dflash").config_enabled is True
    assert get_spec_decoder("mtp").config_enabled is True
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
        "mtp_sidecar": None,
        "mtp_num_draft_tokens": 1,
        "mtp_max_k": 3,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_speculative_config_mtp_normalizes_to_legacy_spec_decode() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(
        speculative_config=(
            '{"method":"mtp","model":"google/gemma-4-12B-it-assistant",'
            '"num_speculative_tokens":2}'
        )
    )

    _normalize_speculative_config_or_exit(args)

    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar == "google/gemma-4-12B-it-assistant"
    assert args.mtp_max_k == 2
    assert args._speculative_config.method == "mtp"
    assert args._speculative_config.model == "google/gemma-4-12B-it-assistant"
    assert args._speculative_config.num_speculative_tokens == 2


def test_spec_decode_mtp_legacy_flag_is_speculative_config_shorthand() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(
        spec_decode="mtp",
        mtp_sidecar="google/gemma-4-12B-it-assistant",
        mtp_max_k=2,
    )

    _normalize_speculative_config_or_exit(args)

    assert args.spec_decode == "mtp"
    assert args._speculative_config.method == "mtp"
    assert args._speculative_config.model == "google/gemma-4-12B-it-assistant"
    assert args._speculative_config.num_speculative_tokens == 2


def test_speculative_config_mtp_rejects_legacy_sidecar_flag(capsys) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _spec_config_args(
        speculative_config='{"method":"mtp"}',
        mtp_sidecar="google/gemma-4-12B-it-assistant",
    )

    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err
    assert "--mtp-sidecar" in captured.err
