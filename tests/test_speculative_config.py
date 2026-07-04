# SPDX-License-Identifier: Apache-2.0
"""Tests for the vLLM-style speculative config frontend."""

from __future__ import annotations

import subprocess
import sys

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
        ('{"method":"unknown"}', "unsupported speculative decoding method"),
    ],
)
def test_parse_speculative_config_rejects_bad_payloads(raw: str, match: str) -> None:
    with pytest.raises(SpeculativeConfigError, match=match):
        parse_speculative_config(raw)


def test_require_migrated_speculative_config_rejects_unwired_method() -> None:
    cfg = parse_speculative_config('{"method":"mtp"}')
    assert cfg is not None

    with pytest.raises(SpeculativeConfigError, match="not wired yet"):
        require_migrated_speculative_config(cfg)


def test_spec_decoder_registry_lists_existing_backends() -> None:
    methods = {plugin.method for plugin in iter_spec_decoders()}

    assert {"dflash", "mtp", "suffix"}.issubset(methods)
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


def test_serve_rejects_unwired_speculative_config_before_model_load() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.cli",
            "serve",
            "qwen3.5-9b-8bit",
            "--speculative-config",
            '{"method":"mtp"}',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2
    assert "not wired yet" in proc.stderr
    assert "use --spec-decode mtp" in proc.stderr
    assert "Fetching" not in proc.stderr
