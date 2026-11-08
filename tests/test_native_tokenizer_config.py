# SPDX-License-Identifier: Apache-2.0
"""Native loaders must not ask HF to instantiate an unsupported model config."""

import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_mlx
pytest.importorskip("mlx.core", exc_type=ImportError)


@pytest.mark.parametrize("model_type", ["lfm3_vl", "prism_hadamard_qwen35"])
def test_native_tokenizer_has_no_architecture_mismatch(
    tmp_path, monkeypatch, caplog, model_type
):
    import socket

    import mlx_lm.utils
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    def no_network(*args, **kwargs):
        pytest.fail("Native tokenizer loading must stay offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    backend = Tokenizer(
        WordLevel({"<unk>": 0, "<pad>": 1, "</s>": 2, "hello": 3}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    template = "{{ messages }}"
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        model_max_length=8192,
        chat_template=template,
    ).save_pretrained(tmp_path)
    config = {"model_type": model_type}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "chat_template.jinja").write_text(template)
    caplog.set_level("WARNING", logger="transformers")

    if model_type == "lfm3_vl":
        from rapid_mlx.models.media_backend import load_native_media

        monkeypatch.setattr(mlx_lm.utils, "load_model", lambda *a, **kw: (object(), {}))
        _, processor = load_native_media(tmp_path, config)
    else:
        import mlx_vlm.models.qwen3_5 as qwen
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil,
        )

        from rapid_mlx.models.prism_hadamard_qwen35 import _build_processor

        monkeypatch.setattr(
            Qwen2VLImageProcessorPil, "from_pretrained", lambda *a, **kw: object()
        )
        monkeypatch.setattr(
            qwen, "Qwen3VLProcessor", lambda **kw: SimpleNamespace(**kw)
        )
        processor = _build_processor(tmp_path)

    assert not any("instantiate a model of type" in r.message for r in caplog.records)
    tokenizer = processor.tokenizer
    assert tokenizer.encode("hello", add_special_tokens=False) == [3]
    assert tokenizer.pad_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.model_max_length == 8192
    assert tokenizer.chat_template == template
