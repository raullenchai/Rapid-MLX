# SPDX-License-Identifier: Apache-2.0
"""Pinned catalog and tool-wire contracts for the October 27 model wave."""

import json
from pathlib import Path

import pytest

from rapid_mlx.model_aliases import resolve_profile, resolve_revision
from rapid_mlx.model_auto_config import detect_model_config
from rapid_mlx.model_sizes import size_bytes
from rapid_mlx.tool_parsers import ToolParserManager

ALIAS = "ornith-1.5-122b-a10b-4bit"
REPO = "ornith-ai/Ornith-1.5-122B-MLX-4bit"
REVISION = "06a57aba4fbae218efc9716bcdf7d6b3035331c8"
ROOT = Path(__file__).resolve().parents[1]


def test_ornith_122b_pinned_catalog_contract():
    profile = resolve_profile(ALIAS)
    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.revision == REVISION
    assert resolve_revision(REPO) == REVISION
    assert profile.modality == "text"
    assert profile.min_memory_gb == 96
    assert profile.tool_call_parser == "qwen3_coder_xml"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid is True
    assert profile.is_moe is True
    assert profile.supports_spec_decode is False
    assert profile.supports_image_input is False
    assert size_bytes(REPO) == 69_642_922_462


def test_ornith_122b_alias_routing_preserves_hybrid_contract():
    cfg = detect_model_config(ALIAS)
    assert cfg is not None
    assert cfg.is_hybrid is True
    assert cfg.is_moe is True
    assert cfg.supports_spec_decode is False
    assert cfg.tool_call_parser == "qwen3_coder_xml"
    assert cfg.reasoning_parser == "qwen3"


def test_ornith_122b_packaged_integrity_covers_exact_checkpoint():
    payload = json.loads((ROOT / "rapid_mlx/model_digests.json").read_text())
    entry = payload["aliases"][ALIAS]
    assert entry["repo"] == REPO
    assert entry["revision"] == REVISION
    files = {file["path"]: file for file in entry["files"]}
    # All language shards and the separate vision tower are needed even
    # though the alias does not advertise qualified image input.
    assert {f"model-{i:05d}-of-00014.safetensors" for i in range(1, 15)} <= set(files)
    assert files["model-vision.safetensors"]["size"] == 902_618_785
    assert "chat_template.jinja" in files
    assert not any(path.endswith(".py") for path in files)


@pytest.mark.parametrize("city", ["Paris", "007"])
def test_ornith_template_xml_uses_schema_aware_parser(city):
    # Checkpoint template emits named XML parameters, rather than the JSON
    # tool body used by some sibling templates. Keep numeric-looking cities
    # as strings under the declared OpenAI schema.
    profile = resolve_profile(ALIAS)
    parser = ToolParserManager.get_tool_parser(profile.tool_call_parser)(None)
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
    }
    wire = (
        "<tool_call>\n<function=get_weather>\n"
        f"<parameter=city>\n{city}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = parser.extract_tool_calls(wire, request=request)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "get_weather"
    assert json.loads(result.tool_calls[0]["arguments"]) == {"city": city}
    assert not result.content
