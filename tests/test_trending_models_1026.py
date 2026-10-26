"""Catalog contracts for the October 26 model qualification wave."""

from types import SimpleNamespace

import pytest

from rapid_mlx.api import utils
from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.model_sizes import size_bytes


@pytest.mark.parametrize(
    "alias,repo,revision,size,tool_parser,reasoning_parser,is_hybrid,is_moe",
    [
        (
            "nanbeige4.3-33b-4bit",
            "pratikgx/Nanbeige4.3-33B-MLX-4bit",
            "883c8bda53b2062ff7e37f178ccf503c79674797",
            18649200689,
            "hermes",
            "deepseek_r1",
            False,
            False,
        ),
        (
            "spark-x2.6-4b-4bit",
            "abenzerps/Spark-X2.6-4B-MLX-4bit",
            "253181296bd0c643880b0f64daa0049a42148d4a",
            2330273898,
            "glm47",
            "qwen3",
            False,
            False,
        ),
        (
            "sage-a1.5-4bit",
            "mlx-community/Sage-A1.5-OptiQ-4bit",
            "efad5741d354e38b3429433c5144331f8fb86cce",
            23536584172,
            "hermes",
            "qwen3",
            True,
            True,
        ),
    ],
)
def test_qualified_alias_contract(
    alias, repo, revision, size, tool_parser, reasoning_parser, is_hybrid, is_moe
):
    profile = resolve_profile(alias)
    assert profile is not None
    assert profile.hf_path == repo
    assert profile.revision == revision
    assert size_bytes(repo) == size
    assert profile.tool_call_parser == tool_parser
    assert profile.reasoning_parser == reasoning_parser
    assert profile.is_hybrid is is_hybrid
    assert profile.is_moe is is_moe
    assert (
        profile.min_memory_gb
        == {
            "nanbeige4.3-33b-4bit": 32,
            "spark-x2.6-4b-4bit": 16,
            "sage-a1.5-4bit": 48,
        }[alias]
    )
    assert profile.supports_image_input is (alias == "sage-a1.5-4bit")
    assert profile.is_text_only is (alias == "nanbeige4.3-33b-4bit")
    assert profile.vision_min_memory_gb == (48 if alias == "sage-a1.5-4bit" else None)


def test_spark_uses_vision_runtime_for_text_checkpoint(monkeypatch):
    monkeypatch.setattr(
        utils,
        "read_model_metadata",
        lambda _: SimpleNamespace(config={"model_type": "spark2_5"}),
    )
    assert utils.is_mllm_model("spark-x2.6-4b-4bit") is True
