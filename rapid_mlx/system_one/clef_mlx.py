# SPDX-License-Identifier: MIT
#
# Adapted from mlx_vlm/models/clef/ (clef.py, config.py, __init__.py) in
# https://github.com/Blaizzy/mlx-vlm at revision
# fdd94f39552a011e298f5d4160ef001238943c1b, which is distributed under the MIT
# License:
#
#   Copyright © 2025 Prince Canuma
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
# Taken from there: the joint schema head and its layers, the forward pass over
# text, image and video inputs, the mapping of the release's head weights onto
# MLX modules, the quantization predicate, the option ordering and the prompt
# and span layout. The head and prompt in turn follow the Cloudflare Clef
# release (Apache-2.0), whose notice is in rapid_mlx/clef/vendor/NOTICE.
#
# Changes: imports point at the installed mlx-vlm Qwen3.5 classes; the module
# registers itself as the `clef` model type for mlx-vlm's loader; checkpoint
# metadata is validated before loading; every weight is materialized at load so
# requests can run on worker threads; scoring returns unrounded per-option
# probabilities and answers are built in the Clef release's wire shape; video
# frame sampling keeps the processor's defaults and does not edit the caller's
# lists.
"""Cloudflare Clef joint-schema decisions on native MLX.

Clef reads the state and every question in one prompt, then a trained head
scores each question's options from the backbone's hidden states. Nothing is
generated. The head, the prompt layout and the option ordering are part of
what the checkpoint was trained on, so they follow the release.

**Registration:** the mlx-vlm this project pins has no ``clef`` model type, so
``load_clef`` installs this module as ``sys.modules["mlx_vlm.models.clef"]``
and lets mlx-vlm's own loader build, quantize and fill the model.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.base import install_auto_processor_patch
from mlx_vlm.models.qwen3_5 import (
    LanguageModel,
    TextConfig,
    VisionConfig,
    VisionModel,
)
from mlx_vlm.models.qwen3_5 import Model as Qwen3_5Model
from mlx_vlm.models.qwen3_5.config import ModelConfig as Qwen3_5ModelConfig
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from mlx_vlm.models.qwen3_vl.qwen3_vl import masked_scatter

__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]

MODEL_TYPE = "clef"
MAX_LENGTH = 16384
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
_HEAD_FIELDS = (
    "hidden_size",
    "width",
    "routing_layers",
    "layers",
    "heads",
    "feedforward",
)


def _span_means(spans, length: int) -> mx.array:
    weights = np.zeros((len(spans), length), dtype=np.float32)
    for row, (start, end) in enumerate(spans):
        weights[row, start:end] = 1.0 / (end - start)
    return mx.array(weights)


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiHeadAttention(width, heads, bias=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = [
            nn.Linear(width, feedforward),
            nn.Linear(feedforward, width),
        ]

    def __call__(self, queries: mx.array, memory: mx.array) -> mx.array:
        memory = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), memory, memory)
        hidden = nn.gelu(self.feedforward[0](self.feedforward_norm(queries)))
        return queries + self.feedforward[1](hidden)


class FieldDecoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(width, heads, bias=True)
        self.multihead_attn = nn.MultiHeadAttention(width, heads, bias=True)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, x: mx.array, memory: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)
        x = x + self.multihead_attn(self.norm2(x), memory, memory)
        return x + self.linear2(nn.gelu(self.linear1(self.norm3(x))))


class JointSchemaHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [
            FieldDecoderLayer(width, heads, feedforward) for _ in range(layers)
        ]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = [nn.Linear(width * 4, width), nn.Linear(width, 1)]
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    def __call__(self, hidden, question_spans, option_spans, lexical, types, counts):
        length = hidden.shape[0]
        memory = self.memory_projection(hidden)[None]
        global_vector = hidden[-1]
        questions = (_span_means(question_spans, length) @ hidden).astype(hidden.dtype)
        contexts = (_span_means(option_spans, length) @ hidden).astype(hidden.dtype)
        owner = mx.array([i for i, count in enumerate(counts) for _ in range(count)])

        routed = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(questions)[owner]
        )[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        routed = routed[0]

        base = self.question_projection(questions)
        scores = mx.sum(routed * base[owner], axis=-1) / math.sqrt(routed.shape[-1])
        bounds = np.cumsum([0, *counts]).tolist()
        summaries = mx.stack(
            [
                mx.sum(
                    mx.softmax(scores[s:e], precise=True)[:, None] * routed[s:e], axis=0
                )
                for s, e in zip(bounds[:-1], bounds[1:])
            ]
        )
        fields = (
            base
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vector)
            + self.type_embedding(types)
        )[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])[owner]

        anchor = questions + global_vector
        anchor = anchor / mx.maximum(
            mx.linalg.norm(anchor, axis=-1, keepdims=True), 1e-12
        )
        lexical = lexical / mx.maximum(
            mx.linalg.norm(lexical, axis=-1, keepdims=True), 1e-12
        )
        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        prior = prior_scale * mx.sum(lexical * anchor[owner], axis=-1)
        options = self.option_norm(routed)
        cosine = mx.sum(fields * options, axis=-1) / mx.maximum(
            mx.linalg.norm(fields, axis=-1) * mx.linalg.norm(options, axis=-1), 1e-8
        )
        features = mx.concatenate(
            [fields, options, fields * options, mx.abs(fields - options)], axis=-1
        )
        residual = self.residual_scorer[1](
            nn.gelu(self.residual_scorer[0](features))
        ).squeeze(-1)
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, math.log(100.0)))
        joint = joint_scale * cosine + residual
        return prior + mx.sigmoid(self.residual_gate) * joint


@dataclass
class ModelConfig(Qwen3_5ModelConfig):
    head_config: dict = field(default_factory=dict)


class Model(Qwen3_5Model):
    """The Qwen3.5 backbone with Clef's joint schema head."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.head = JointSchemaHead(**config.head_config)

    def __call__(self, input_ids, question_spans, option_spans, qtype, **media):
        embeds = self.language_model.model.embed_tokens(input_ids)
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        for pixels, grid, token in (
            ("pixel_values", "image_grid_thw", self.config.image_token_index),
            ("pixel_values_videos", "video_grid_thw", self.config.video_token_index),
        ):
            if media.get(pixels) is not None:
                states, _ = self.vision_tower(media[pixels].astype(dtype), media[grid])
                embeds = masked_scatter(
                    embeds,
                    mx.broadcast_to((input_ids == token)[..., None], embeds.shape),
                    states,
                )
        position_ids, _ = self.language_model.get_rope_index(
            input_ids, media.get("image_grid_thw"), media.get("video_grid_thw")
        )
        hidden = self.language_model.model(
            input_ids, inputs_embeds=embeds, position_ids=position_ids
        )
        hidden = self.head.hidden_norm(hidden[0])
        flat = np.asarray(input_ids[0])
        lexical_ids = np.concatenate([flat[s:e] for s, e in option_spans])
        lexical_spans = np.cumsum([0, *(e - s for s, e in option_spans)])
        lm_head, ids = self.language_model.lm_head, mx.array(lexical_ids)
        embeddings = lm_head.weight[ids]
        if "scales" in lm_head:
            biases = lm_head.get("biases")
            embeddings = mx.dequantize(
                embeddings,
                lm_head.scales[ids],
                None if biases is None else biases[ids],
                group_size=lm_head.group_size,
                bits=lm_head.bits,
                mode=lm_head.mode,
            )
        lexical = (
            _span_means(
                list(zip(lexical_spans[:-1], lexical_spans[1:])), len(lexical_ids)
            )
            @ embeddings
        )
        return self.head(
            hidden,
            [span for span, _ in question_spans],
            option_spans,
            lexical.astype(hidden.dtype),
            qtype,
            [count for _, count in question_spans],
        )

    def sanitize(self, weights):
        backbone, head = {}, {}
        for key, value in weights.items():
            if key.startswith(
                ("model.", "lm_head", "language_model.", "vision_tower.")
            ):
                backbone[key] = value
                continue
            key = "head." + key.removeprefix("head.")
            if key.endswith(("in_proj_weight", "in_proj_bias")):
                prefix, suffix = key.rsplit(".in_proj_", 1)
                for name, part in zip(
                    ("query_proj", "key_proj", "value_proj"), mx.split(value, 3, axis=0)
                ):
                    head[f"{prefix}.{name}.{suffix}"] = part
                continue
            for layer in ("feedforward", "residual_scorer"):
                key = key.replace(f"{layer}.3.", f"{layer}.1.")
            head[key] = value
        return {**super().sanitize(backbone), **head}

    @property
    def quant_predicate(self):
        base = super().quant_predicate

        def predicate(path, module):
            if path.startswith("head."):
                return False
            return True if base is None else base(path, module)

        return predicate


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    kind = question["type"]
    if kind == "noul":
        defaults = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        defaults.update(question.get("criteria") or {})
        return [(key, defaults[key]) for key in ("true", "false")]
    if kind == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


def clef_answer(
    question: dict[str, Any], probabilities: dict[str, float]
) -> dict[str, Any]:
    """Build one answer in the shape and rounding of the Clef release."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {
                option: round(probabilities[option], 4) for option in options
            },
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(
            sum(index * probabilities[level] for index, level in enumerate(levels)), 4
        ),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


def load_clef(path: str | Path) -> tuple[Any, Any]:
    """Load a prepared Clef checkpoint as ``(model, processor)``.

    A prepared checkpoint is a Qwen3.5 checkpoint whose root ``config.json``
    says ``model_type: "clef"`` and carries the published joint head
    configuration under ``head_config``.
    """
    root = Path(path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Clef checkpoint has no config.json: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Clef needs model_type {MODEL_TYPE!r} in config.json, "
            f"got {config.get('model_type')!r}"
        )
    head = config.get("head_config")
    if not isinstance(head, dict) or set(head) != set(_HEAD_FIELDS):
        raise ValueError(
            "Clef checkpoint config.json needs a head_config with exactly "
            f"{', '.join(_HEAD_FIELDS)}"
        )
    if any(
        isinstance(head[name], bool)
        or not isinstance(head[name], int)
        or head[name] < 1
        for name in _HEAD_FIELDS
    ):
        raise ValueError("Clef head_config values must be positive integers")

    import mlx_vlm

    sys.modules[f"mlx_vlm.models.{MODEL_TYPE}"] = sys.modules[__name__]
    install_auto_processor_patch(MODEL_TYPE, Qwen3VLProcessor)
    model, processor = mlx_vlm.load(str(root))
    # Requests run on worker threads. A lazy array stays bound to the thread
    # that built it, so materialize every weight now.
    mx.eval(model.parameters())
    return model, processor


class ClefScorer:
    """Scores typed questions jointly with a loaded Clef model."""

    def __init__(self, model: Any, processor: Any):
        self._model = model
        self._processor = processor
        self._tokenizer = getattr(processor, "tokenizer", processor)

    def _tokens(self, text: str) -> list[int]:
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def _sample_videos(self, videos: list[list[Any]]) -> list[dict[str, Any]]:
        """Pick evenly spaced frames in place and describe them for the processor."""
        video_processor = self._processor.video_processor
        metadata = []
        for index, frames in enumerate(videos):
            total = len(frames)
            count = min(
                max(int(total / 24 * video_processor.fps), video_processor.min_frames),
                video_processor.max_frames,
                total,
            )
            indices = np.linspace(0, total - 1, count).round().astype(int)
            videos[index] = [frames[i] for i in indices]
            metadata.append({"frames_indices": indices.tolist(), "fps": 24})
        return metadata

    def _sequence(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        images: list[Any] | None,
        videos: list[list[Any]] | None,
    ):
        tokens = self._tokens
        schema = tokens("\n\nSCHEMA FIELDS:\n")
        rows = []
        for index, (name, question) in enumerate(questions.items()):
            schema += tokens(
                f"\nFIELD {index + 1}\nID: {name}\nTYPE: {question['type']}\n"
                "INSTRUCTION: "
            )
            start = len(schema)
            schema += tokens(_render(question.get("instructions") or str(name)))
            question_span = (start, len(schema))
            schema += tokens("\nALLOWED OPTIONS:\n")
            spans, labels = [], []
            for number, (option, description) in enumerate(_options(question), 1):
                schema += tokens(f"OPTION {number}: ")
                start = len(schema)
                semantics = {"option_id": option}
                if description is not None:
                    semantics["description"] = description
                schema += tokens(_render(semantics))
                spans.append((start, len(schema)))
                labels.append(option)
                schema += tokens("\n")
            schema += tokens("END FIELD\n")
            rows.append((name, question, question_span, spans, labels))

        prefix = tokens(
            "<|im_start|>system\nRead the complete state and schema. Decide every "
            "field jointly. Each answer must be exactly one of that field's allowed "
            "options.<|im_end|>\n<|im_start|>user\nSTATE:\n"
        )
        suffix = tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "JOINT SCHEMA DECISIONS:"
        )
        media: dict[str, mx.array] = {}
        images, videos = list(images or []), [list(v) for v in videos or []]
        if images or videos:
            media_kwargs = (
                {"video_metadata": self._sample_videos(videos)} if videos else {}
            )
            encoded = self._processor(
                text=[
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    * len(images)
                    # transformers 5 keeps the outer vision tokens around
                    # timestamped frames
                    + "<|vision_start|><|vision_start|><|video_pad|>"
                    "<|vision_end|><|vision_end|>" * len(videos) + "\n"
                ],
                images=images or None,
                videos=videos or None,
                **media_kwargs,
            )
            prefix += np.asarray(encoded["input_ids"])[0].tolist()
            media = {
                key: mx.array(np.asarray(encoded[key]))
                for key in (
                    "pixel_values",
                    "image_grid_thw",
                    "pixel_values_videos",
                    "video_grid_thw",
                )
                if encoded.get(key) is not None
            }
        fixed = len(prefix) + len(schema) + len(suffix)
        if fixed > MAX_LENGTH:
            raise ValueError(
                f"schema requires {fixed} tokens before state; maximum is {MAX_LENGTH}"
            )
        state_ids = tokens(_render(state))[: MAX_LENGTH - fixed]
        offset = len(prefix) + len(state_ids)
        rows = [
            (
                name,
                question,
                (span[0] + offset, span[1] + offset),
                [(s + offset, e + offset) for s, e in spans],
                labels,
            )
            for name, question, span, spans, labels in rows
        ]
        return prefix + state_ids + schema + suffix, rows, media

    def score(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        images: list[Any] | None = None,
        videos: list[list[Any]] | None = None,
    ) -> tuple[dict[str, dict[str, float]], int]:
        """Return ``{question_id: {option: probability}}`` and the token count."""
        ids, rows, media = self._sequence(state, questions, images, videos)
        logits = self._model(
            mx.array([ids]),
            [(row[2], len(row[3])) for row in rows],
            [span for row in rows for span in row[3]],
            mx.array([QUESTION_TYPES[row[1]["type"]] for row in rows]),
            **media,
        )
        mx.eval(logits)
        answers = {}
        start = 0
        for name, _, _, spans, labels in rows:
            values = logits[start : start + len(spans)].astype(mx.float32)
            start += len(spans)
            probabilities = np.asarray(mx.softmax(values)).astype(float).tolist()
            answers[name] = dict(zip(labels, probabilities))
        return answers, len(ids)
