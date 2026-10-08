# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# The prompt layout, label tokens, row batching, calibration and isolated score
# levels below are adapted from mlx_vlm/models/decider2/decider2.py in
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
# Changes from that file: the backbone is the mlx-lm Qwen3.5 text model, so
# position ids are left to it; questions arrive as the System One wire schema
# and raw probabilities are returned to the System One response builder; the
# packed (non-independent) scoring mode is not included; calibration values are
# validated at load; the label readout rows are materialized at construction so
# requests can run on worker threads.
"""Decider typed-decision scoring on the mlx-lm Qwen3.5 text backbone.

The prompt layout, label tokens, row batching and calibration follow the
``decider2`` model in mlx-vlm (MIT, see the notice above), which in turn
follows the published decider-2b release. They are part of what the checkpoint
was tuned and calibrated on, so they are reproduced here rather than
redesigned. Decider reads the prompt once and scores answer labels at the last
position; it never generates text.
"""

from __future__ import annotations

import json
import math
import re
import string
from pathlib import Path
from typing import Any

from .schema import Question

MODEL_TYPE = "decider2"
MAX_STATE_TOKENS = 32768
# Rows of one request are batched while rows x padded length stays under this.
_BATCH_TOKEN_BUDGET = 8192
_PAD_MULTIPLE = 64
_MAX_LABELS = 255
_INLINE_OPTIONS = 10
_MAX_SCORE_LEVELS = 10


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _annotate(value: Any) -> Any:
    """Number the items of long lists, as the release does for its state."""
    if isinstance(value, list):
        if len(value) >= 8:
            return [
                {"_index": index, **_annotate(item)}
                if isinstance(item, dict)
                else {"_index": index, "value": _annotate(item)}
                for index, item in enumerate(value)
            ]
        return [_annotate(item) for item in value]
    if isinstance(value, dict):
        return {key: _annotate(item) for key, item in value.items()}
    return value


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(_annotate(state), ensure_ascii=False)


def render_question(question: Question) -> dict[str, Any]:
    """Turn one wire question into Decider's instruction, keys and options."""
    instruction = _text(question.instructions)
    if question.type == "choice":
        assert isinstance(question.criteria, dict)
        if not 2 <= len(question.criteria) <= _MAX_LABELS:
            raise ValueError(
                f"Decider choice questions need 2 to {_MAX_LABELS} options"
            )
        keys = list(question.criteria)
        options = [
            key
            if question.criteria[key] in (None, "")
            else f"{key}: {_text(question.criteria[key])}"
            for key in keys
        ]
        levels: list[str] | None = None
    elif question.type == "score":
        assert isinstance(question.criteria, list)
        if not 2 <= len(question.criteria) <= _MAX_SCORE_LEVELS:
            raise ValueError(
                f"Decider score questions need 2 to {_MAX_SCORE_LEVELS} levels"
            )
        keys = [str(index) for index in range(len(question.criteria))]
        levels = [_text(item) for item in question.criteria]
        options = [f"{index}: {level}" for index, level in enumerate(levels)]
    else:
        criteria = question.criteria if isinstance(question.criteria, dict) else {}
        false, true = criteria.get("false"), criteria.get("true")
        keys = ["false", "true"]
        options = [
            "no" if false in (None, "") else f"no: {_text(false)}",
            "yes" if true in (None, "") else f"yes: {_text(true)}",
        ]
        levels = None
    return {
        "instruction": instruction,
        "keys": keys,
        "options": options,
        "levels": levels,
    }


def load_decider(path: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load a prepared Decider checkpoint as ``(text_model, tokenizer, settings)``.

    A prepared checkpoint is a Qwen3.5 text checkpoint whose root
    ``config.json`` says ``model_type: "decider2"`` and carries the published
    calibration under ``decision_config``.
    """
    root = Path(path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Decider checkpoint has no config.json: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Decider needs model_type {MODEL_TYPE!r} in config.json, "
            f"got {config.get('model_type')!r}"
        )
    settings = config.get("decision_config")
    if not isinstance(settings, dict) or "temperature" not in settings:
        raise ValueError(
            "Decider checkpoint config.json has no decision_config.temperature"
        )
    limit = settings.get("max_state_tokens", MAX_STATE_TOKENS)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Decider max_state_tokens must be a positive integer")
    by_type = settings.get("temperature_by_type") or {}
    if not isinstance(by_type, dict):
        raise ValueError("Decider temperature_by_type must be an object")
    for name, value in {"temperature": settings["temperature"], **by_type}.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"Decider calibration {name!r} must be a positive finite number"
            )

    from mlx_lm.models import qwen3_5
    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer

    model, _ = load_model(
        root, get_model_classes=lambda config: (qwen3_5.Model, qwen3_5.ModelArgs)
    )
    tokenizer = AutoTokenizer.from_pretrained(str(root))
    return model.language_model, tokenizer, settings


class DeciderScorer:
    """Scores typed questions with a loaded Decider backbone.

    ``text_model.model`` maps token ids to final hidden states and owns the
    input embedding table, which is also the label readout (tied weights).
    """

    def __init__(self, text_model: Any, tokenizer: Any, settings: dict[str, Any]):
        self._backbone = text_model.model
        self._tokenizer = tokenizer
        self._settings = settings
        names = list(string.ascii_uppercase) + [
            first + second
            for first in string.ascii_uppercase
            for second in string.ascii_uppercase
        ]
        labels = [
            (name, tokens[0])
            for name in names
            if len(tokens := self._encode(name)) == 1
        ][:_MAX_LABELS]
        if len(labels) != _MAX_LABELS or len({token for _, token in labels}) != len(
            labels
        ):
            raise ValueError(
                f"the Decider tokenizer does not provide {_MAX_LABELS} distinct "
                "label tokens"
            )
        self._labels = [name for name, _ in labels]
        self._label_token_ids = [token for _, token in labels]

        import mlx.core as mx

        self._label_weights = self._backbone.embed_tokens(
            mx.array(self._label_token_ids)
        )
        # Requests run on worker threads. A lazy array stays bound to the
        # thread that built it, so materialize the readout rows now.
        mx.eval(self._label_weights)
        pad = getattr(tokenizer, "pad_token_id", None)
        self._pad_token_id = int(pad) if pad is not None else 0

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _prompt(self, context_ids: list[int], question: str, options: list[str]):
        ids = list(context_ids)
        head = f"\n\nQuestion: {question}\nOptions:"
        tail = "\nAnswer: ("
        if len(options) <= _INLINE_OPTIONS:
            listing = "".join(
                f"\n({self._labels[index]}) {value}"
                for index, value in enumerate(options)
            )
            return ids + self._encode(head + listing + tail)
        # Past ten options the release writes each label as its own token so
        # two-letter labels cannot merge with the surrounding punctuation.
        ids += self._encode(head)
        open_ids = self._encode("\n(")
        for index, value in enumerate(options):
            ids += open_ids + [self._label_token_ids[index]]
            ids += self._encode(f") {value}")
        return ids + self._encode(tail)

    def _score_rows(
        self, prompts: list[list[int]], widths: list[int], temperatures: list[float]
    ) -> list[list[float]]:
        import mlx.core as mx
        import numpy as np

        results: list[list[float]] = []
        start = 0
        while start < len(prompts):
            end = start + 1
            while (
                end < len(prompts)
                and (end - start + 1) * max(len(p) for p in prompts[start : end + 1])
                <= _BATCH_TOKEN_BUDGET
            ):
                end += 1
            group = prompts[start:end]
            longest = max(len(prompt) for prompt in group)
            length = -(-longest // _PAD_MULTIPLE) * _PAD_MULTIPLE
            ids = np.full((len(group), length), self._pad_token_id, dtype=np.int32)
            for row, prompt in enumerate(group):
                ids[row, : len(prompt)] = prompt
            hidden = self._backbone(mx.array(ids))
            last = hidden[
                mx.arange(len(group)), mx.array([len(prompt) - 1 for prompt in group])
            ]
            logits = last @ self._label_weights.T
            for row in range(len(group)):
                index = start + row
                values = (
                    logits[row, : widths[index]].astype(mx.float32)
                    / temperatures[index]
                )
                results.append(np.asarray(mx.softmax(values)).astype(float).tolist())
            start = end
        return results

    def score(
        self, state: Any, questions: dict[str, Question]
    ) -> tuple[dict[str, tuple[list[str], list[float]]], int]:
        """Return ``{question_id: (keys, probabilities)}`` and the token count."""
        rendered = {key: render_question(value) for key, value in questions.items()}
        limit = min(
            int(self._settings.get("max_state_tokens", MAX_STATE_TOKENS)),
            MAX_STATE_TOKENS,
        )
        context_ids = self._encode("Context:\n" + render_state(state))[:limit]
        isolate = bool(self._settings.get("isolated_levels", False))
        default_temperature = float(self._settings["temperature"])
        by_type = self._settings.get("temperature_by_type") or {}

        rows: list[tuple[str, list[str]]] = []
        temperatures: list[float] = []
        spans: list[tuple[str, int, int, bool]] = []
        for key, item in rendered.items():
            kind = questions[key].type
            temperature = float(by_type.get(kind, default_temperature))
            if isolate and kind == "score":
                # Each level is judged on its own as a yes/no fit, then the
                # fits are normalized into a distribution over levels.
                spans.append((key, len(rows), len(item["levels"]), True))
                for level in item["levels"]:
                    level = re.sub(r"^\s*-?\d+\s*:\s*", "", level)
                    rows.append(
                        (
                            f"{item['instruction']}\nProposed answer: {level}\n"
                            "Does the proposed answer fit?",
                            ["no", "yes"],
                        )
                    )
                    temperatures.append(temperature)
            else:
                spans.append((key, len(rows), 1, False))
                rows.append((item["instruction"], item["options"]))
                temperatures.append(temperature)

        prompts = [
            self._prompt(context_ids, question, options) for question, options in rows
        ]
        probabilities = self._score_rows(
            prompts, [len(options) for _, options in rows], temperatures
        )
        answers: dict[str, tuple[list[str], list[float]]] = {}
        for key, start, count, isolated in spans:
            if isolated:
                fit = [probabilities[start + index][1] for index in range(count)]
                mass = sum(fit) or 1e-9
                distribution = [value / mass for value in fit]
            else:
                distribution = probabilities[start]
            answers[key] = (rendered[key]["keys"], distribution)
        # Rows repeat the state, which is one prompt to the caller: count the
        # prefix every row shares once, as the release does.
        shared = 0
        shortest = min(len(prompt) for prompt in prompts)
        while shared < shortest and all(
            prompt[shared] == prompts[0][shared] for prompt in prompts
        ):
            shared += 1
        return answers, shared + sum(len(prompt) - shared for prompt in prompts)
