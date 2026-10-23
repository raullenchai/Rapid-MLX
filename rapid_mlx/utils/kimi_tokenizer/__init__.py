# SPDX-License-Identifier: Apache-2.0
"""Reviewed Kimi tokenizer: executable implementation ships with Rapid.

Only local vocabulary/config/template data is read from the checkpoint. Never
import its Python, AutoTokenizer mapping, or processor declarations.
"""

import json
from pathlib import Path


def load_kimi_tokenizer(model_path):
    from transformers import AddedToken

    from .tokenization import TikTokenTokenizer

    path = Path(model_path)
    config = json.loads((path / "tokenizer_config.json").read_text())
    if (
        not isinstance(config, dict)
        or config.get("tokenizer_class") != "TikTokenTokenizer"
    ):
        raise ValueError("Unsupported Kimi tokenizer configuration")
    added = {}
    for index, value in config.get("added_tokens_decoder", {}).items():
        if not isinstance(value, dict) or not isinstance(value.get("content"), str):
            raise ValueError("Invalid Kimi added token")
        added[int(index)] = AddedToken(
            **{
                key: value[key]
                for key in (
                    "content",
                    "single_word",
                    "lstrip",
                    "rstrip",
                    "normalized",
                    "special",
                )
                if key in value
            }
        )
    kwargs = {
        key: config[key]
        for key in (
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "additional_special_tokens",
            "model_max_length",
            "clean_up_tokenization_spaces",
            "chat_template",
        )
        if key in config
    }
    tokenizer = TikTokenTokenizer(
        str(path / "tiktoken.model"), added_tokens_decoder=added, **kwargs
    )
    tokenizer.name_or_path = str(path)
    return tokenizer
