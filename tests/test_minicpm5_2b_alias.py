# SPDX-License-Identifier: Apache-2.0
"""Product contracts for the official MiniCPM5 2B MLX checkpoint."""

from __future__ import annotations

from vllm_mlx import model_sizes
from vllm_mlx.catalog import build_legacy_catalog_snapshot
from vllm_mlx.model_aliases import list_profiles
from vllm_mlx.model_auto_config import detect_model_config

ALIAS = "minicpm5-2b-4bit"
REPO = "openbmb/MiniCPM5-2B-MLX"
DOWNLOAD_BYTES = 1_425_999_742


def test_alias_pins_the_qualified_checkpoint_and_protocols() -> None:
    profile = list_profiles()[ALIAS]
    assert profile.hf_path == REPO
    assert profile.is_text_only is True
    assert profile.tool_call_parser == "minicpm"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid is False
    assert profile.is_moe is False
    assert profile.supports_spec_decode is False
    assert detect_model_config(ALIAS) == profile
    assert detect_model_config(REPO) == profile


def test_alias_is_available_to_server_and_desktop_as_text_chat() -> None:
    snapshot = build_legacy_catalog_snapshot()
    record = next(item for item in snapshot["aliases"] if item["alias"] == ALIAS)

    assert record["availability"]["server"] is True
    assert record["availability"]["desktop"] is True
    assert record["capabilities"]["task_types"] == ["text_generation"]
    assert record["capabilities"]["operation_modes"] == ["chat"]
    assert record["capabilities"]["runtime_adapter"] == "mlx_lm"
    assert record["capabilities"]["is_text_only"] is True
    assert record["capabilities"]["tool_call_parser"] == "minicpm"
    assert record["capabilities"]["reasoning_parser"] == "qwen3"


def test_download_manifest_pins_the_qualified_footprint() -> None:
    assert model_sizes.size_bytes(REPO) == DOWNLOAD_BYTES
