# SPDX-License-Identifier: Apache-2.0
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_mlx

from rapid_mlx.models.kimi_vlm import load_kimi_processor, load_kimi_vlm


@pytest.fixture
def checkpoint(tmp_path):
    fixtures = Path(__file__).parent / "fixtures/kimi_k26"
    config = json.loads((fixtures / "tokenizer_config.json").read_text())
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
    (tmp_path / "chat_template.jinja").write_text(
        (fixtures / "chat_template.jinja").read_text()
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k25",
                "auto_map": {"AutoModel": "modeling_kimi.Model"},
                "vision_config": {"patch_size": 14, "merge_kernel_size": [2, 2]},
            }
        )
    )
    for filename in [
        "tokenization_kimi.py",
        "modeling_kimi.py",
        "kimi_k25_processor.py",
    ]:
        (tmp_path / filename).write_text(
            "raise AssertionError('repository code executed')"
        )
    return tmp_path


@pytest.mark.parametrize(
    "extra_config,expected",
    [
        ({}, []),
        ({"eos_token_id": [123]}, [123]),
        ({"text_config": {"eos_token_id": 124}}, [124]),
    ],
)
def test_native_stopping_uses_tokenizer_and_config_sources(
    checkpoint, extra_config, expected
):
    path = checkpoint / "config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | extra_config))
    (checkpoint / "generation_config.json").write_text('{"eos_token_id": [125]}')
    tokenizer = load_kimi_processor(checkpoint).tokenizer
    criteria = tokenizer.stopping_criteria
    for token in [tokenizer.eos_token_id, 125, *expected]:
        assert criteria(token)
    assert not criteria(97)
    criteria.reset()
    assert criteria(tokenizer.eos_token_id)
    assert criteria(125)


def test_native_processor_preserves_image_tokens_tools_and_thinking(checkpoint):
    from PIL import Image

    processor = load_kimi_processor(checkpoint)
    assert processor.tokenizer.stopping_criteria(processor.tokenizer.eos_token_id)
    output = processor(images=[Image.new("RGB", (28, 28))], text="<|media_pad|>hello")
    assert output["pixel_values"].size > 0
    assert output["image_grid_hws"].shape[-1] == 2
    assert output["image_token_id"] in output["input_ids"][0].tolist()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    text = processor.apply_chat_template(
        [{"role": "user", "content": "Hi"}], tools=tools, thinking=False
    )
    declaration = text.split("<|im_middle|>", 1)[1].split("<|im_end|>", 1)[0]
    assert json.loads(declaration) == tools
    assert text.endswith("<think></think>")
    processor.detokenizer.add_token(processor.tokenizer.encode("Hi")[0])
    processor.detokenizer.finalize()
    assert processor.detokenizer.text


@pytest.mark.parametrize("trust", [None, "0"])
def test_native_loader_uses_same_snapshot_explicit_false_and_preserves_options(
    checkpoint, monkeypatch, trust
):
    import mlx_vlm.utils

    if trust is None:
        monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", trust)
    seen = []
    fake = SimpleNamespace()
    monkeypatch.setattr(
        mlx_vlm.utils, "load_model", lambda path, **kw: seen.append((path, kw)) or fake
    )
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    model, processor = load_vlm_with_policy(checkpoint, lazy=True, strict=False)
    assert model is fake
    assert seen == [
        (checkpoint, {"lazy": True, "strict": False, "trust_remote_code": False})
    ]
    assert processor.tokenizer.decode(processor.tokenizer.encode("你好")) == "你好"


def test_native_loader_never_accepts_custom_model_file(checkpoint):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["model_file"] = "modeling_kimi.py"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="model_file"):
        load_kimi_vlm(checkpoint, trust_remote_code=True)


@pytest.mark.parametrize("trust", [None, "0"])
def test_public_multimodal_entry_uses_native_processor(checkpoint, monkeypatch, trust):
    import mlx_vlm
    import mlx_vlm.utils
    from PIL import Image

    from rapid_mlx.models import mllm
    from rapid_mlx.patches import (
        glm5_next_forget_gate_quant,
        glm5_next_processor,
        glm5_next_runtime,
    )

    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    monkeypatch.setattr(
        glm5_next_runtime, "install_glm5_next_runtime_fix", lambda: None
    )
    monkeypatch.setattr(
        glm5_next_processor, "install_glm5_next_processor_patch", lambda: None
    )
    monkeypatch.setattr(
        glm5_next_forget_gate_quant,
        "install_glm5_next_forget_gate_quant_fix",
        lambda: None,
    )
    monkeypatch.setattr(
        mlx_vlm, "load", lambda *a, **k: pytest.fail("unchecked processor dispatch")
    )
    monkeypatch.setattr(
        mlx_vlm.utils,
        "load_model",
        lambda *a, **k: SimpleNamespace(config=SimpleNamespace()),
    )
    if trust is None:
        monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", trust)
    engine = mllm.MLXMultimodalLM(str(checkpoint), enable_cache=False)
    engine.load()
    assert engine._loaded
    result = engine.processor(
        images=[Image.new("RGB", (28, 28))], text="<|media_pad|>what?"
    )
    assert result["pixel_values"].size > 0
    assert result["image_token_id"] in result["input_ids"][0].tolist()


def test_native_media_config_and_adapter(checkpoint, monkeypatch):
    import mlx_vlm.utils

    (checkpoint / "preprocessor_config.json").write_text(
        json.dumps({"media_proc_cfg": {"patch_size": 14, "merge_kernel_size": 2}})
    )
    model = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(mlx_vlm.utils, "load_model", lambda *a, **k: model)
    calls = []
    monkeypatch.setattr(
        mlx_vlm.utils,
        "apply_lora_layers",
        lambda model, path: calls.append(path) or model,
    )
    loaded, processor = load_kimi_vlm(checkpoint, adapter_path="adapter")
    assert loaded is model
    assert calls == ["adapter"]
    assert processor.image_processor.merge_kernel_size == [2, 2]
    (checkpoint / "config.json").write_text('{"model_type": "other"}')
    with pytest.raises(ValueError, match="requires model_type"):
        load_kimi_vlm(checkpoint)
