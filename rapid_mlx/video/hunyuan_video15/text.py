# SPDX-License-Identifier: Apache-2.0
"""Reuse MLX Qwen2 and the vendored T5 primitives for dual prompt conditioning."""

from __future__ import annotations

import gc
import json
import re

import mlx.core as mx
import mlx.nn as nn

from .loading import load_component, qwen_key, t5_key

# The spaces in this template are significant: upstream's continuation indentation
# is part of the string, and its fixed 108-token crop depends on that template.
SYSTEM_MESSAGE = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    + "        ".join(
        [
            "",
            "1. The main content and theme of the video. ",
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. ",
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. ",
            "4. background environment, light, style and atmosphere. ",
            "5. camera angles, movements, and transitions used in the video.",
        ]
    )
)


def extract_glyph_text(prompt):
    matches = re.findall(r'"(.*?)"|“(.*?)”', prompt)
    texts = list(dict.fromkeys(a or b for a, b in matches))
    return ". ".join(f'Text "{t}"' for t in texts) + ". " if texts else None


def encode_prompts(root, prompts, config, cancel_check):
    from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model
    from transformers import AutoTokenizer, ByT5Tokenizer, T5Config

    from ...image.sd35_runtime._vendor.t5 import SD3T5Encoder

    qconfig = json.loads((root / "text_encoder/config.json").read_text())
    # hidden_states[-3] is the output of layer N-2 before the final RMSNorm.
    qconfig["num_hidden_layers"] -= 2
    qconfig["rope_scaling"] = (
        None  # Text-only mRoPE has identical positions on all axes.
    )
    model = Qwen2Model(ModelArgs.from_dict(qconfig))
    del model.norm
    load_component(
        model,
        root / "text_encoder",
        dtype=mx.bfloat16,
        key_map=lambda key: qwen_key(key, qconfig["num_hidden_layers"]),
        cancel_check=cancel_check,
    )
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    results = []
    for prompt in prompts:
        cancel_check()
        tokens = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt or " "},
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=1108,
            truncation=True,
            return_tensors="np",
        )
        ids, valid = mx.array(tokens["input_ids"]), mx.array(tokens["attention_mask"])
        h = encode_qwen_hidden(model, ids, valid, cancel_check)
        results.append([h[:, 108:], valid[:, 108:]])
        mx.eval(*results[-1])
    del model, h
    gc.collect()
    mx.clear_cache()

    tconfig = T5Config.from_pretrained(root / "text_encoder_2", local_files_only=True)
    # Use the existing configurable T5 layers. Its top-level forward omits padding
    # masks, so run the layers with position bias plus the tokenizer mask here.
    model = SD3T5Encoder(tconfig, low_memory_mode=False)
    for layer in model.encoder.layers:
        layer.dense.act = nn.gelu_approx  # T5 gated-gelu uses gelu_new (tanh).
    load_component(
        model,
        root / "text_encoder_2",
        dtype=mx.float32,
        key_map=t5_key,
        cancel_check=cancel_check,
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        root / "tokenizer_2", local_files_only=True
    )
    for result, prompt in zip(results, prompts):
        cancel_check()
        glyph = extract_glyph_text(prompt)
        if glyph is None:
            result.extend(
                [
                    mx.zeros((1, 256, tconfig.d_model), dtype=mx.bfloat16),
                    mx.zeros((1, 256), dtype=mx.int32),
                ]
            )
            continue
        tokens = tokenizer(
            glyph,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        valid = mx.array(tokens["attention_mask"])
        h = encode_t5_hidden(
            model, mx.array(tokens["input_ids"]), valid, cancel_check
        ).astype(mx.bfloat16)
        mx.eval(h)
        result.extend([h, valid])
    del model
    gc.collect()
    mx.clear_cache()
    return results


def encode_qwen_hidden(model, ids, valid, cancel_check=lambda: None):
    h = model.embed_tokens(ids)
    causal = mx.arange(ids.shape[1])[:, None] >= mx.arange(ids.shape[1])[None]
    mask = causal[None, None] & valid[:, None, None, :].astype(mx.bool_)
    for layer in model.layers:
        cancel_check()
        h = layer(h, mask=mask)
        mx.eval(h)
    return h


def encode_t5_hidden(model, ids, valid, cancel_check=lambda: None):
    h = model.wte(ids)
    mask = model.encoder.relative_attention_bias(h.shape[1], h.shape[1])[
        None
    ] + mx.where(valid[:, None, None, :] != 0, 0.0, -3.4028234663852886e38)
    for layer in model.encoder.layers:
        cancel_check()
        h = layer(h, mask)
        mx.eval(h)
    return model.encoder.ln(h)
