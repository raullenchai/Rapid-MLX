import json

from rapid_mlx import model_sizes
from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.tool_parsers import ToolParserManager


def test_lfm3_alias_profile():
    profile = resolve_profile("lfm3-1.7b-8bit")
    assert profile.hf_path == "LiquidAI/LFM3-1.7B-MLX-8bit"
    assert profile.revision == "ee7667f3cfccb2a41725f91820531823fcabc6b7"
    assert profile.tool_call_parser == "nemotron"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid and not profile.supports_spec_decode
    assert model_sizes.size_bytes(profile.hf_path) == 1_855_254_842


def test_lfm3_template_tool_calls():
    parser = ToolParserManager.get_tool_parser("nemotron")(None)
    text = (
        "Before <tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis & Lyon\n</parameter>\n"
        '<parameter=options>\n{"units": ["C", "F"]}\n</parameter>\n'
        "</function>\n</tool_call>\n\n<tool_call>\n<function=other>\n"
        '<parameter=message>\nhello "world"\n</parameter>\n'
        "</function>\n</tool_call> After"
    )
    result = parser.extract_tool_calls(text)
    assert result.tools_called and len(result.tool_calls) == 2
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "city": "Paris & Lyon",
        "options": {"units": ["C", "F"]},
    }
    assert json.loads(result.tool_calls[1]["arguments"]) == {"message": 'hello "world"'}
    assert result.content == "Before  After"
    parser.reset()
    current = ""
    content = ""
    calls = []
    for char in text:
        previous, current = current, current + char
        delta = parser.extract_tool_calls_streaming(previous, current, char) or {}
        content += delta.get("content", "") or ""
        calls.extend(delta.get("tool_calls", []))
    assert len(calls) == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "city": "Paris & Lyon",
        "options": {"units": ["C", "F"]},
    }
    assert json.loads(calls[1]["function"]["arguments"]) == {"message": 'hello "world"'}
    assert "<" not in content
    assert "Before" in content and "After" in content


def test_lfm3_vl_alias_profile():
    profile = resolve_profile("lfm3-vl-2b-8bit")
    assert profile.hf_path == "LiquidAI/LFM3-VL-2B-MLX-8bit"
    assert profile.revision == "83336d5be3874a2bdd3297bde77b2114dbd3a816"
    assert profile.supports_image_input and profile.is_hybrid
    assert profile.vision_min_memory_gb == 8
    assert profile.tool_call_parser == "nemotron"
    assert profile.reasoning_parser == "qwen3"
    assert model_sizes.size_bytes(profile.hf_path) == 2_234_045_143
