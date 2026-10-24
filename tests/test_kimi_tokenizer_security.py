# SPDX-License-Identifier: Apache-2.0
import base64
import hashlib
import json
from pathlib import Path

import pytest

from rapid_mlx.utils.kimi_tokenizer import load_kimi_tokenizer
from rapid_mlx.utils.kimi_tokenizer.vocabulary import load_local_bpe

FIXTURES = Path(__file__).parent / "fixtures" / "kimi_k26"


@pytest.fixture
def tokenizer(tmp_path):
    config = json.loads((FIXTURES / "tokenizer_config.json").read_text())
    config["added_tokens_decoder"] = {
        str(int(index) - 163584 + 256): value
        for index, value in config["added_tokens_decoder"].items()
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    (tmp_path / "tiktoken.model").write_bytes(
        b"".join(
            base64.b64encode(bytes([i])) + b" " + str(i).encode() + b"\n"
            for i in range(256)
        )
    )
    # Never imported, even with config's AutoTokenizer mapping present.
    (tmp_path / "tokenization_kimi.py").write_text(
        "raise RuntimeError('executed repo')"
    )
    result = load_kimi_tokenizer(tmp_path)
    result.chat_template = (FIXTURES / "chat_template.jinja").read_text()
    return result


@pytest.mark.parametrize(
    "text", ["", "Hello 世界! مرحبا 👋", "é\n\t12345", "x" * 26000, " " * 26000]
)
def test_data_only_roundtrip(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text


def test_special_tokens_and_skip(tokenizer):
    assert tokenizer.encode("[BOS]hello[EOS]")[0] == 256
    assert tokenizer.encode("[BOS]hello[EOS]")[-1] == 257
    assert (
        tokenizer.decode(tokenizer.encode("[BOS]hello[EOS]"), skip_special_tokens=True)
        == "hello"
    )
    assert tokenizer.encode("[BOS]", allow_special_tokens=False) == list(b"[BOS]")


def test_base_tokenizer_positional_and_paired_contracts(tokenizer):
    assert tokenizer.decode(tokenizer.encode("left", "right", False)) == "leftright"
    assert tokenizer.encode(
        "left", text_pair="right", add_special_tokens=False
    ) == list(b"leftright")
    assert tokenizer.encode([97, 98], [99], False) == [97, 98, 99]
    ids = tokenizer.encode("[BOS]hello[EOS]")
    assert tokenizer.decode(ids, True) == "hello"
    assert tokenizer.decode(ids, False, False) == "[BOS]hello[EOS]"
    assert (
        tokenizer.decode(
            tokenizer.encode("[BOS]", "[EOS]", False, allow_special_tokens=False), True
        )
        == "[BOS][EOS]"
    )
    assert tokenizer.encode([97], [98], allow_special_tokens=False) == [97, 98]


def test_decode_normalizes_numpy_inputs(tokenizer):
    import numpy as np

    assert tokenizer.decode(np.array([97, 98], dtype=np.int64)) == "ab"
    assert tokenizer.decode(np.array(97, dtype=np.int64)) == "a"
    assert tokenizer.decode(np.int64(97)) == "a"


def test_constructor_requires_local_vocabulary(tmp_path):
    from rapid_mlx.utils.kimi_tokenizer.tokenization import TikTokenTokenizer

    with pytest.raises(ValueError, match="local tiktoken.model"):
        TikTokenTokenizer(str(tmp_path / "missing"))


@pytest.mark.parametrize("value", [None, 42, ["text"]])
def test_raw_tokenizer_rejects_non_text_explicitly(tokenizer, value):
    with pytest.raises(TypeError, match="must be a string"):
        tokenizer._encode_text(value)


@pytest.mark.parametrize("boundary", [25_000, 400_000])
@pytest.mark.parametrize("padding", ["x", " "])
def test_special_tokens_survive_long_text_chunk_boundaries(
    tokenizer, boundary, padding
):
    marker = "<|im_end|>"
    text = padding * (boundary - 5) + marker + "tail"
    tokens = tokenizer.encode(text)
    assert tokens.count(tokenizer.special_tokens[marker]) == 1
    assert tokenizer.decode(tokens) == text
    ordinary = tokenizer.encode(text, allow_special_tokens=False)
    assert tokenizer.special_tokens[marker] not in ordinary
    assert tokenizer.decode(ordinary) == text


@pytest.mark.parametrize(
    "thinking,suffix", [(True, "<think>"), (False, "<think></think>")]
)
def test_exact_pinned_chat_template(tokenizer, thinking, suffix):
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], thinking=thinking
    )
    assert (
        result
        == "<|im_user|>user<|im_middle|>Hi<|im_end|><|im_assistant|>assistant<|im_middle|>"
        + suffix
    )
    assert (
        tokenizer.decode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hi"}],
                thinking=thinking,
                tokenize=True,
                return_dict=False,
            )
        )
        == result
    )


def test_tool_schema_rendering_and_thinking_history(tokenizer):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Weather?"}], tools=tools, thinking=False
    )
    declaration = result.split("<|im_middle|>", 1)[1].split("<|im_end|>", 1)[0]
    assert json.loads(declaration) == tools
    conversation = [
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "Answer", "reasoning_content": "Reason"},
        {"role": "user", "content": "Two"},
    ]
    assert "Reason" not in tokenizer.apply_chat_template(conversation)
    assert "Reason" in tokenizer.apply_chat_template(
        conversation, preserve_thinking=True
    )


@pytest.mark.parametrize(
    "name", ["weather", "get-weather", "default", 'a"\\b\nc', "天気"]
)
@pytest.mark.parametrize("tokenize", [False, True])
def test_json_tool_format_preserves_complete_schemas(tokenizer, name, tokenize):
    # Shared and nested definitions, escaped JSON Pointers, recursive roots,
    # closed objects and scalar enums remain data.
    schema = {
        "type": "object",
        "$defs": {
            "a/b~c": {
                "type": "object",
                "$defs": {"Nested": {"type": "string"}},
                "properties": {"value": {"$ref": "#/$defs/Nested"}},
            }
        },
        "properties": {
            "child": {"$ref": "#"},
            "postal code": {"$ref": "#/$defs/a~1b~0c"},
            "user-id": {"enum": [None, True, False, 1, 1.5, 'a"\\b\nc']},
        },
        "required": ["user-id"],
        "additionalProperties": False,
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": "first\rsecond\u2028third\u2029fourth",
                "parameters": schema,
            },
        }
        for tool_name in [name, "second-tool"]
    ]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tools=tools,
        tokenize=tokenize,
        return_dict=False,
        tools_ts_str="must not replace the JSON schema",
    )
    if tokenize:
        rendered = tokenizer.decode(rendered)
    declaration = rendered.split("<|im_middle|>", 1)[1].split("<|im_end|>", 1)[0]
    assert json.loads(declaration) == tools
    assert rendered.endswith("<think>")


def test_vocab_refuses_url_and_invalid_ranks(tmp_path):
    with pytest.raises(ValueError, match="local"):
        load_local_bpe("https://example.invalid/model")
    path = tmp_path / "vocab"
    path.write_bytes(b"YQ== 0\nYQ== 1\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_local_bpe(path)


@pytest.mark.real_hf_cache
def test_real_pinned_vocabulary_offline_if_cached():
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        "mlx-community/Kimi-K2.6-mlx-DQ3_K_M-q8",
        "tiktoken.model",
        revision="9cb5961bf22facced678b894ee5e9eb1538f7cba",
    )
    if not isinstance(cached, str):
        pytest.skip(
            "Pinned tokenizer data not cached; synthetic-vocabulary tests remain mandatory"
        )
    tokenizer = load_kimi_tokenizer(Path(cached).parent)
    text = "Hello 世界! مرحبا 👋"
    expected = [19180, 220, 2243, 0, 61561, 7570, 4265, 1063, 130732, 233]
    assert tokenizer.encode(text) == expected
    assert tokenizer.decode(expected) == text
    assert tokenizer.encode("[BOS]hello[EOS]") == [163584, 22931, 163585]


def test_reviewed_source_provenance():
    root = Path(__file__).parents[1] / "rapid_mlx/utils/kimi_tokenizer"
    for filename, expected in {
        "tokenization.py": "21bfbe6f53235954c1cc17db42aa5a5de86004943fb3dffb20afc818bd656781",
    }.items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected


@pytest.mark.requires_mlx
@pytest.mark.parametrize("trust", [None, "0"])
@pytest.mark.parametrize("lazy", [False, True])
def test_public_loader_uses_native_model_and_data_tokenizer(
    tokenizer, monkeypatch, trust, lazy
):
    import mlx_lm.utils

    from rapid_mlx.utils import tokenizer as loading

    path = Path(tokenizer.name_or_path)
    config = {
        "model_type": "kimi_k25",
        "text_config": {"model_type": "deepseek_v3"},
        "auto_map": {"AutoModel": "evil.Model"},
        "model_file": "evil.py",
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "evil.py").write_text("raise RuntimeError('repo code executed')")
    (path / "chat_template.jinja").write_text(tokenizer.chat_template)
    if trust is None:
        monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", trust)
    calls = []
    model = object()

    def load_model(model_path, **kwargs):
        calls.append(kwargs)
        assert kwargs["model_config"]["model_file"] is None
        assert kwargs["model_config"]["auto_map"] is None
        return model, config

    monkeypatch.setattr(mlx_lm.utils, "load_model", load_model)
    monkeypatch.setattr(loading, "_try_inject_mtp_post_load", lambda *a: None)
    monkeypatch.setattr(loading, "_post_load_ubc_evict", lambda *a: None)
    result = loading.load_model_with_fallback(str(path), lazy=lazy, return_config=lazy)
    assert result[0] is model
    assert result[1].encode("[BOS]hello")[0] == 256
    assert (
        result[1]
        .apply_chat_template([{"role": "user", "content": "Hi"}])
        .endswith("<think>")
    )
    assert calls[0].get("lazy", False) is lazy
    if lazy:
        assert result[2] == config


@pytest.mark.requires_mlx
def test_tiny_native_kimi_model_executes_with_owned_tokenizer(tokenizer, monkeypatch):
    import mlx.core as mx
    from mlx_lm.models.kimi_k25 import Model, ModelArgs

    from rapid_mlx.utils.tokenizer import load_model_with_fallback

    path = Path(tokenizer.name_or_path)
    config = {
        "model_type": "kimi_k25",
        "text_config": {
            "model_type": "deepseek_v3",
            "vocab_size": 512,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "q_lora_rank": 8,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "qk_nope_head_dim": 4,
            "v_head_dim": 4,
            "intermediate_size": 32,
            "first_k_dense_replace": 1,
        },
        "auto_map": {"AutoModel": "evil.Model"},
        "model_file": "evil.py",
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "evil.py").write_text("raise RuntimeError('repo code executed')")
    (path / "chat_template.jinja").write_text(tokenizer.chat_template)
    original = Model(ModelArgs.from_dict(config))
    original.save_weights(str(path / "model.safetensors"))
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    model, loaded = load_model_with_fallback(str(path))
    inputs = mx.array([loaded.encode("hello")])
    assert model(inputs).shape == (1, 5, 512)
    assert loaded.decode(inputs[0].tolist()) == "hello"


def test_tokenizer_utility_contract(tokenizer, tmp_path):
    from rapid_mlx.utils.kimi_tokenizer.tokenization import TikTokenTokenizer

    token = tokenizer.encode("a")[0]
    assert tokenizer.decode(token) == "a"
    assert tokenizer.vocab_size == 512
    assert tokenizer.clean_up_tokenization(" a ") == " a "
    target = tmp_path / "saved"
    target.mkdir()
    (saved,) = tokenizer.save_vocabulary(str(target), filename_prefix="copy")
    assert Path(saved).read_bytes() == Path(tokenizer.vocab_file).read_bytes()
    assert tokenizer.save_vocabulary(str(Path(tokenizer.vocab_file).parent)) == (
        tokenizer.vocab_file,
    )
    with pytest.raises(ValueError, match="directory"):
        tokenizer.save_vocabulary(str(target / "missing"))
    assert "Hi" in tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tools=[{"type": "function", "function": {"name": "bad"}}],
    )
    # Default special-token construction remains usable for supported vocabularies.
    kwargs = {
        "bos_token": "<|reserved_token_256|>",
        "eos_token": "<|reserved_token_257|>",
        "pad_token": "<|reserved_token_258|>",
        "unk_token": "<|reserved_token_259|>",
    }
    native = TikTokenTokenizer(tokenizer.vocab_file, added_tokens_decoder={}, **kwargs)
    assert native.decode(native.encode("hello")) == "hello"


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"tokenizer_class": "Unknown"},
        {
            "tokenizer_class": "TikTokenTokenizer",
            "added_tokens_decoder": {"1": {"content": 2}},
        },
    ],
)
def test_invalid_tokenizer_configuration(tmp_path, config):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_kimi_tokenizer(tmp_path)


@pytest.mark.parametrize(
    "content,message",
    [
        (b"\nYQ== 1\n", "contiguous"),
        (b"YQ== 0\n", "every byte"),
        (b"YQ== -1\n", "duplicate"),
    ],
)
def test_invalid_vocabulary_data(tmp_path, content, message):
    path = tmp_path / "tiktoken.model"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        load_local_bpe(path)
