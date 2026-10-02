# SPDX-License-Identifier: Apache-2.0
"""Wire models and shared rendering for the System One API."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul", "choice", "score"]
    instructions: Any
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def validate_criteria(self) -> Question:
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise ValueError("choice criteria must be a non-empty object")
            if len(self.criteria) > 255:
                raise ValueError("choice questions support at most 255 options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise ValueError(
                    "score criteria must contain at least two ordered levels"
                )
            if len(self.criteria) > 255:
                raise ValueError("score questions support at most 255 levels")
        elif self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError("noul criteria must be an object when provided")
        return self


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Any
    questions: dict[str, Question] = Field(min_length=1, max_length=64)
    model: str | None = None
    temperature: float = Field(default=1.0, ge=1e-6, le=100.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def bounded_candidates(self) -> SystemOneRequest:
        # CLM encodes every candidate independently. Bound the aggregate, not
        # only each question, so a schema-valid request cannot multiply 64
        # individually-valid 255-option questions into 16k encoder passes.
        total = 0
        for question in self.questions.values():
            if question.type == "noul":
                total += 2
            else:
                total += len(question.criteria or [])
        if total > 255:
            raise ValueError("a request supports at most 255 total answer candidates")
        return self


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: Any = ""
    question: str | None = None
    answers: list[str] = Field(min_length=1, max_length=255)
    model: str | None = None
    temperature: float = Field(default=1.0, ge=1e-6, le=100.0, allow_inf_nan=False)

    @field_validator("answers")
    @classmethod
    def non_empty_answers(cls, values: list[str]) -> list[str]:
        if any(not value for value in values):
            raise ValueError("answers must be non-empty strings")
        return values


def to_text(value: Any, indent: int = 0) -> str:
    """Render structured state using CLM's training-time prose layout."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    pad = " " * indent
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                parts.append(f"{pad}{key}:\n{to_text(item, indent + 2)}")
            else:
                parts.append(f"{pad}{key}: {to_text(item)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(value, (list, tuple)):
        return "\n".join(
            f"{pad}-\n{to_text(item, indent + 2)}"
            if isinstance(item, (dict, list)) and item
            else f"{pad}- {to_text(item)}"
            for item in value
        )
    return json.dumps(value, ensure_ascii=False)


def clm_pairs(
    state: Any, questions: dict[str, Question]
) -> dict[str, tuple[str, list[str], list[str]]]:
    pairs = {}
    state_value = to_text(state).strip()
    for question_id, question in questions.items():
        instructions = to_text(question.instructions).strip()
        state_text = (
            f"{state_value}\n\n{instructions}"
            if state_value and instructions
            else state_value or instructions
        )
        if question.type == "choice":
            assert isinstance(question.criteria, dict)
            keys = list(question.criteria)
            candidates = [
                to_text(question.criteria[key])
                if question.criteria[key] not in (None, "")
                else key
                for key in keys
            ]
        elif question.type == "score":
            assert isinstance(question.criteria, list)
            keys = [str(index) for index in range(len(question.criteria))]
            candidates = [to_text(item) for item in question.criteria]
        else:
            criteria = question.criteria if isinstance(question.criteria, dict) else {}
            keys = ["false", "true"]
            candidates = []
            for key in keys:
                description = criteria.get(key)
                if description in (None, ""):
                    description = (
                        f"Yes. This is true: {instructions}"
                        if key == "true"
                        else f"No. This is false: {instructions}"
                    )
                candidates.append(f"{key}: {to_text(description)}")
        pairs[question_id] = (state_text, keys, candidates)
    return pairs


def answer_from_probabilities(
    question: Question, keys: list[str], probabilities: list[float]
) -> dict[str, Any]:
    distribution = dict(zip(keys, (float(p) for p in probabilities)))
    winner = max(range(len(probabilities)), key=probabilities.__getitem__)
    rest = [p for index, p in enumerate(probabilities) if index != winner]
    confidence = probabilities[winner] - (sum(rest) / len(rest) if rest else 0.0)
    if question.type == "noul":
        return {"type": "noul", "noul": distribution["true"]}
    if question.type == "choice":
        return {
            "type": "choice",
            "choice": keys[winner],
            "confidence": max(0.0, min(1.0, confidence)),
            "probabilities": distribution,
        }
    return {
        "type": "score",
        "score": sum(
            index * probability for index, probability in enumerate(probabilities)
        ),
        "confidence": max(0.0, min(1.0, confidence)),
        "probabilities": distribution,
        "legend": {
            str(index): item for index, item in enumerate(question.criteria or [])
        },
    }
