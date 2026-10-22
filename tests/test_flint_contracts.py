# SPDX-License-Identifier: Apache-2.0
"""Flint catalog, parser, template, and KV policy contracts without MLX."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

from rapid_mlx.kv_cache_dtype import (
    KVCacheQuantizationUnsupportedError,
    resolve_kv_cache_dtype,
)
from rapid_mlx.tool_parsers import ToolParserManager
from rapid_mlx.utils.chat_template import _normalize_assistant_tool_call_arguments

# Byte-exact copies from this validated checkpoint, hashed from the original
# files in its HF snapshot. Keep these pins independent of the local fixtures;
# tests need neither the checkpoint cache nor network access.
FLINT_SOURCE = "RecursiveSI/flint-int4"
FLINT_REVISION = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
# (fixture name, original checkpoint file name, SHA-256)
FLINT_FIXTURES = (
    (
        "flint_chat_template.jinja",
        "chat_template.jinja",
        "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466",
    ),
    (
        "flint_config.json",
        "config.json",
        "80f577a984db1bd40dfff88457b611d4c13e807704a945616be27f28809eed21",
    ),
)

ARGS = {
    "number": 42,
    "flag": True,
    "obj": {"x": 1},
    "items": [1, "two"],
    "text": "hello there",
    "quoted": "123",
}
CALL = (
    "<tool_call>lookup"
    + "".join(
        f"<arg_key>{k}</arg_key><arg_value>{json.dumps(v) if not isinstance(v, str) or k == 'quoted' else v}</arg_value>"
        for k, v in ARGS.items()
    )
    + "</tool_call>"
)


@pytest.mark.parametrize("separator", ["", "\n"])
@pytest.mark.parametrize("step", [1, 7, 10000])
def test_flint_tool_values_parallel_streaming_parity(separator, step):
    parser = ToolParserManager.get_tool_parser("glm47")(None)
    text = (
        "I will look that up. " + CALL + separator + CALL.replace("lookup", "save", 1)
    )
    result = parser.extract_tool_calls(text)
    assert result.content is None
    expected = [(tc["name"], json.loads(tc["arguments"])) for tc in result.tool_calls]
    assert expected == [("lookup", ARGS), ("save", ARGS)]
    stream = []
    for i in range(0, len(text), step):
        delta = parser.extract_tool_calls_streaming(
            text[:i], text[: i + step], text[i : i + step]
        )
        if delta:
            stream.extend(
                (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                for tc in delta.get("tool_calls", [])
            )
    assert stream == expected


def test_flint_reasoning_tags_emitted_by_model():
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    parser = Qwen3ReasoningParser()
    reasoning, content = parser.extract_reasoning(
        "<think>Check the facts.</think>Answer."
    )
    assert reasoning == "Check the facts."
    assert content == "Answer."


def test_flint_fixtures_match_validated_checkpoint():
    root = Path(__file__).resolve().parents[1]
    for fixture, original, digest in FLINT_FIXTURES:
        data = (root / "tests/fixtures" / fixture).read_bytes()
        assert sha256(data).hexdigest() == digest, (
            f"{fixture} drifted from {FLINT_SOURCE}@{FLINT_REVISION}/{original}"
        )
    alias = json.loads((root / "rapid_mlx/aliases.json").read_text())[
        "flint-40b-a2.5b-4bit"
    ]
    assert alias["hf_path"] == FLINT_SOURCE
    assert (
        f"`{FLINT_SOURCE}@{FLINT_REVISION}`"
        in (root / "docs/reference/models.md").read_text()
    )


def test_flint_template_replays_string_arguments_and_reasoning():
    template = (
        Path(__file__).parent / "fixtures/flint_chat_template.jinja"
    ).read_text()
    messages = [
        {"role": "user", "content": "Look up the record."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Check first.",
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": json.dumps(ARGS)}}
            ],
        },
        {"role": "tool", "content": "Found it.", "tool_call_id": "call_1"},
    ]
    normalized = _normalize_assistant_tool_call_arguments(messages)
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == ARGS
    assert isinstance(messages[1]["tool_calls"][0]["function"]["arguments"], str)
    rendered = _compile_jinja_template(template).render(
        messages=normalized, tools=[], add_generation_prompt=True
    )
    assert "<think>Check first.</think>" in rendered
    assert "<arg_key>number</arg_key><arg_value>42</arg_value>" in rendered
    assert "<|observation|><tool_response>Found it.</tool_response>" in rendered
    assert rendered.endswith("<|assistant|>")
    assert not rendered.endswith("<think>")


@pytest.mark.parametrize("dtype", ["int4", "int8"])
def test_flint_config_kv_policy(dtype):
    config = {
        "model_type": "flint",
        "sliding_window": 513,
        "layer_types": ["full_attention", "sliding_attention"],
    }
    with pytest.raises(KVCacheQuantizationUnsupportedError):
        resolve_kv_cache_dtype(
            dtype, explicit=True, model_name="flint", hf_config=config
        )
    decision = resolve_kv_cache_dtype(dtype, model_name="flint", hf_config=config)
    assert decision.dtype == "bf16"


def test_flint_alias_contract():
    root = Path(__file__).resolve().parents[1] / "rapid_mlx"
    alias = json.loads((root / "aliases.json").read_text())["flint-40b-a2.5b-4bit"]
    assert alias == dict(
        hf_path="RecursiveSI/flint-int4",
        revision="3b93f9053b530488405ae9fca51d4900d6ce21f0",
        tool_call_parser="glm47",
        reasoning_parser="qwen3",
        is_moe=True,
        is_hybrid=False,
        is_hybrid_explicit=True,
        supports_spec_decode=False,
        min_memory_gb=48,
    )
    assert (
        json.loads((root / "model_sizes.json").read_text())["sizes"][alias["hf_path"]]
        == 22568912600
    )
    assert "flint" not in (root / "model_recommendations.json").read_text().lower()
    from rapid_mlx.model_aliases import resolve_profile

    profile = resolve_profile("flint-40b-a2.5b-4bit")
    assert profile.is_hybrid_explicit is True
    assert profile.enforce_min_memory is False
    assert profile.supports_native_mtp is False
