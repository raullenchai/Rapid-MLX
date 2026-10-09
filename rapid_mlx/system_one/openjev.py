# SPDX-License-Identifier: Apache-2.0
"""OpenJev typed decisions through mlx-lm.

OpenJev is a fine-tuned Qwen3.8 chat model; mlx-lm loads it with its
``qwen3_5`` model class. Each readout renders one question
as a chat prompt and reads the option-letter logits at the first output
position; nothing is generated. The prompt text, the option letters, the
splitting of long option lists and the calibration constants follow the
OpenJev release helper (``helper/shim.py`` in ``openjev/openjev`` at revision
``1c341f65bfe5d50fdb935c71e9739c9e0938d6c4``) and its port in oMLX
(``omlx/models/openjev.py`` at ``79f4488e2658``), both Apache-2.0, because the
checkpoint was tuned and calibrated on them.

Every readout of a request starts with the same chat header and state, so that
shared token prefix is run once and each readout continues from a copy of its
cache. The model emits bfloat16 logits, whose step near the option letters is
about 0.125; continuing from a cache can move one logit by one step, which
shifts an undecided probability by about 0.02 against an uncached readout.

The published MLX conversion carries the language model only, so this backend
answers questions about text and JSON, not images.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, cast

from .schema import Question

LETTERS = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [
    chr(code) for code in range(ord("a"), ord("z") + 1)
]
READOUT_TEMPERATURE = 0.85
# Yes/no calibration of the MLX release (READOUT_NOUL_T, bias 0): the yes/no
# log-odds are divided by this temperature, as in the release helper.
NOUL_TEMPERATURE = 1.829074
NOUL_CLIP = 1e-4
SCORE_SUFFIX = " Rate along the ordered levels below (lowest first)."
# The release reads state.screenshot / state.image as an image in exactly two
# cases: a ``data:image`` URL, or a string longer than this (raw base64). Any
# other value is ordinary state text there, and stays text here.
_RAW_IMAGE_MIN_CHARS = 2000
# Tokens per forward pass while reading a long prompt, to bound peak memory.
_PREFILL_CHUNK = 2048
# One readout of this many tokens takes about two minutes on an M3 Ultra.
MAX_PROMPT_TOKENS = 32768


def render_state(state: Any) -> str:
    if isinstance(state, dict):
        for key in ("screenshot", "image"):
            value = state.get(key)
            if isinstance(value, str) and (
                value.startswith("data:image") or len(value) > _RAW_IMAGE_MIN_CHARS
            ):
                raise ValueError(
                    f"state.{key} holds an image; the OpenJev MLX build reads text only"
                )
        return json.dumps(state, ensure_ascii=False)
    return state if isinstance(state, str) else str(state)


def _description(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_question(question: Question) -> tuple[str, list[tuple[str, str]]]:
    """Turn one wire question into OpenJev's instruction and ``(key, text)`` options."""
    if question.instructions is None:
        raise ValueError("OpenJev questions need instructions")
    # The release was tuned on Python-literal text for structured instructions.
    instructions = (
        question.instructions
        if isinstance(question.instructions, str)
        else str(question.instructions)
    )
    if question.type == "choice":
        # Question's own validator guarantees the criteria shape per type.
        choices = cast(dict[str, Any], question.criteria)
        return instructions, [
            (str(key), _description(value)) for key, value in choices.items()
        ]
    if question.type == "score":
        levels = cast(list[Any], question.criteria)
        return instructions + SCORE_SUFFIX, [
            (str(index), _description(value)) for index, value in enumerate(levels)
        ]
    criteria = question.criteria if isinstance(question.criteria, dict) else {}
    return instructions, [
        ("yes", _description(criteria.get("true")) or "The statement is true."),
        ("no", _description(criteria.get("false")) or "The statement is false."),
    ]


def render_prompt(state: str, instructions: str, options: list[tuple[str, str]]) -> str:
    """Return the user message of one readout."""
    lines = "\n".join(
        f"[{LETTERS[index]}] {key}: {text}" for index, (key, text) in enumerate(options)
    )
    return (
        f"State:\n{state}\n\nQuestion: {instructions}\nOptions:\n{lines}"
        "\n\nAnswer with the letter of the best option only."
    )


def option_groups(count: int) -> list[range]:
    """Split options into near-equal readouts of at most one letter each."""
    if count < 1:
        raise ValueError("OpenJev questions need at least one option")
    if count > len(LETTERS) ** 2:
        # The readout over the group winners has one letter per group.
        raise ValueError(f"OpenJev reads at most {len(LETTERS) ** 2} options")
    if count <= len(LETTERS):
        return [range(count)]
    groups = -(-count // len(LETTERS))
    size = -(-count // groups)
    return [range(start, min(count, start + size)) for start in range(0, count, size)]


def compose_groups(parts: list[list[float]], final: list[float]) -> list[float]:
    """Merge per-group readouts through the readout over the group winners.

    Each winner's probability in the final readout anchors the mass of its
    group, so every option keeps a non-zero share and the result sums to 1.
    """
    raw: list[float] = []
    for group, probabilities in enumerate(parts):
        winner = max(probabilities)
        raw.extend(final[group] * value / winner for value in probabilities)
    total = sum(raw)
    return [value / total for value in raw]


def choice_confidence(probabilities: list[float]) -> float:
    if len(probabilities) == 1:
        return 1.0
    uniform = 1.0 / len(probabilities)
    return max(0.0, (max(probabilities) - uniform) / (1.0 - uniform))


def score_confidence(probabilities: list[float]) -> float:
    count = len(probabilities)
    if count == 1:
        return 1.0
    mode = max(range(count), key=probabilities.__getitem__)
    spread = sum(value * abs(index - mode) for index, value in enumerate(probabilities))
    center = (count - 1) / 2
    uniform_spread = sum(abs(index - center) for index in range(count)) / count
    return max(0.0, 1.0 - spread / uniform_spread)


def noul_probability(p_yes: float) -> float:
    p_yes = min(max(p_yes, NOUL_CLIP), 1.0 - NOUL_CLIP)
    z = math.log(p_yes / (1.0 - p_yes)) / NOUL_TEMPERATURE
    return 1.0 / (1.0 + math.exp(-z))


def openjev_answer(
    question: Question, options: list[tuple[str, str]], probabilities: list[float]
) -> dict[str, Any]:
    """Build one answer in the shape and rounding of the OpenJev release."""
    if question.type == "choice":
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": options[best][0],
            "probabilities": {
                key: round(value, 4) for (key, _), value in zip(options, probabilities)
            },
            "confidence": round(choice_confidence(probabilities), 4),
        }
    if question.type == "score":
        assert isinstance(question.criteria, list)
        return {
            "type": "score",
            "score": round(
                sum(index * value for index, value in enumerate(probabilities)), 4
            ),
            "legend": {
                str(index): level for index, level in enumerate(question.criteria)
            },
            "probabilities": {
                str(index): round(value, 4) for index, value in enumerate(probabilities)
            },
            "confidence": round(score_confidence(probabilities), 4),
        }
    return {"type": "noul", "noul": round(noul_probability(probabilities[0]), 4)}


def load_openjev(path: str) -> tuple[Any, Any, int]:
    """Load an OpenJev MLX checkpoint as ``(model, tokenizer, context_limit)``."""
    import mlx.core as mx
    from mlx_lm import load

    loaded = load(path)
    model, tokenizer = loaded[0], loaded[1]
    # Requests run on worker threads. A lazy array stays bound to the thread
    # that built it, so materialize every weight now.
    mx.eval(model.parameters())
    limit = None
    config_path = Path(path) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        value = (config.get("text_config") or config).get("max_position_embeddings")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            limit = value
    return model, tokenizer, min(limit or MAX_PROMPT_TOKENS, MAX_PROMPT_TOKENS)


class OpenJevScorer:
    """Scores typed questions with a loaded OpenJev model."""

    def __init__(
        self, model: Any, tokenizer: Any, context_limit: int = MAX_PROMPT_TOKENS
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._limit = context_limit
        encoded = [self._encode(letter) for letter in LETTERS]
        letter_ids = [ids[0] for ids in encoded if len(ids) == 1]
        if len(set(letter_ids)) != len(LETTERS):
            # The readout compares one logit per option letter.
            raise ValueError(
                "this tokenizer does not give each OpenJev option letter its "
                "own single token"
            )
        self._letter_ids = letter_ids

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _prompt_ids(
        self, state: str, instructions: str, options: list[tuple[str, str]]
    ) -> list[int]:
        rendered = self._tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": render_prompt(state, instructions, options),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self._encode(rendered)
        if len(ids) > self._limit:
            raise ValueError(
                f"the OpenJev prompt needs {len(ids)} tokens; the limit is {self._limit}"
            )
        return ids

    def _run(self, ids: list[int], cache: list) -> Any:
        """Feed ``ids`` through the model and return the last position's logits."""
        import mlx.core as mx

        logits = None
        for start in range(0, len(ids), _PREFILL_CHUNK):
            chunk = mx.array(ids[start : start + _PREFILL_CHUNK])[None]
            logits = self._model(chunk, cache=cache)[0, -1]
            mx.eval(logits, [entry.state for entry in cache])
        return logits

    def _new_cache(self) -> list[Any]:
        from mlx_lm.models.cache import make_prompt_cache

        cache: list[Any] = make_prompt_cache(self._model)
        return cache

    def _readout(
        self, ids: list[int], count: int, shared: tuple[list[int], list] | None
    ) -> list[float]:
        """Return the probabilities of the first ``count`` option letters."""
        import mlx.core as mx

        if shared is not None and ids[: len(shared[0])] == shared[0]:
            # Caches update in place; each readout continues from its own copy.
            cache = copy.deepcopy(shared[1])
            tail = ids[len(shared[0]) :]
        else:
            cache, tail = self._new_cache(), ids
        logits = self._run(tail, cache).astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits)
        scores = [
            float(value) for value in logprobs[mx.array(self._letter_ids[:count])]
        ]
        if not all(math.isfinite(value) for value in scores):
            raise RuntimeError("OpenJev option letter scores are not finite")
        top = max(scores)
        weights = [math.exp((value - top) / READOUT_TEMPERATURE) for value in scores]
        total = sum(weights)
        return [value / total for value in weights]

    def _shared_prefix(
        self, readouts: list[list[int]]
    ) -> tuple[list[int], list] | None:
        """Run the longest token prefix every first readout shares, once."""
        if len(readouts) < 2:
            return None
        # Leave at least one token for each readout to continue from.
        length = min(len(ids) for ids in readouts) - 1
        shared = 0
        while shared < length and all(
            ids[shared] == readouts[0][shared] for ids in readouts
        ):
            shared += 1
        if shared == 0:
            return None
        prefix = readouts[0][:shared]
        cache = self._new_cache()
        self._run(prefix, cache)
        return prefix, cache

    def score(
        self, state: Any, questions: dict[str, Question]
    ) -> tuple[dict[str, tuple[list[tuple[str, str]], list[float]]], int]:
        """Return ``{question_id: (options, probabilities)}`` and the token count."""
        text = render_state(state)
        plans = []
        for question_id, question in questions.items():
            instructions, options = render_question(question)
            groups = option_groups(len(options))
            group_ids = [
                self._prompt_ids(text, instructions, [options[i] for i in group])
                for group in groups
            ]
            plans.append((question_id, instructions, options, groups, group_ids))

        shared = self._shared_prefix([ids for plan in plans for ids in plan[4]])
        answers = {}
        tokens = 0
        for question_id, instructions, options, groups, group_ids in plans:
            parts = []
            for group, ids in zip(groups, group_ids):
                parts.append(self._readout(ids, len(group), shared))
                tokens += len(ids)
            if len(parts) == 1:
                probabilities = parts[0]
            else:
                winners = [
                    options[group[max(range(len(part)), key=part.__getitem__)]]
                    for group, part in zip(groups, parts)
                ]
                ids = self._prompt_ids(text, instructions, winners)
                final = self._readout(ids, len(winners), shared)
                tokens += len(ids)
                probabilities = compose_groups(parts, final)
            answers[question_id] = (options, probabilities)
        return answers, tokens
