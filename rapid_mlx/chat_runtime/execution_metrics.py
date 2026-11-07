# SPDX-License-Identifier: Apache-2.0
"""Bounded facts for one final model response, separate from turn usage."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIStreamedResponse

from ..api.models import ExecutionHostMetrics, ExecutionReferenceMetrics

METRICS_KEY = "rapid_execution_metrics"


class GenerationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    execution_reference: ExecutionReferenceMetrics
    execution_host: ExecutionHostMetrics | None = None
    time_to_first_token_ms: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    mean_itl_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)


def bounded_metrics(value):
    """Select only supported facts; provider-specific fields never enter storage."""
    if not isinstance(value, dict):
        return None
    selected = {
        key: value[key] for key in GenerationMetrics.model_fields if key in value
    }
    try:
        return GenerationMetrics.model_validate(selected).model_dump(exclude_none=True)
    except (ValueError, TypeError):
        return None


class ExecutionStreamedResponse(OpenAIStreamedResponse):
    async def _validate_response(self):
        terminal_seen = False
        async for chunk in super()._validate_response():
            if not terminal_seen and any(
                choice.finish_reason is not None for choice in chunk.choices
            ):
                terminal_seen = True
                metrics = bounded_metrics(getattr(chunk, "metrics", None))
                if metrics is not None:
                    self.provider_details = {
                        **(self.provider_details or {}),
                        METRICS_KEY: metrics,
                    }
            yield chunk


class ExecutionChatModel(OpenAIChatModel):
    @property
    def _streamed_response_cls(self):
        return ExecutionStreamedResponse

    def _process_provider_details(self, response):
        details = super()._process_provider_details(response) or {}
        metrics = bounded_metrics(getattr(response, "metrics", None))
        if metrics is not None:
            details[METRICS_KEY] = metrics
        return details or None


def final_generation_metrics(messages):
    """The last response's counters and facts must describe that one generation."""
    if not messages or not isinstance(messages[-1], ModelResponse):
        return None
    response = messages[-1]
    if response.finish_reason not in {"stop", "length"} or not any(
        isinstance(part, TextPart) and part.content.strip() for part in response.parts
    ):
        return None
    metrics = bounded_metrics((response.provider_details or {}).get(METRICS_KEY))
    if (
        metrics is None
        or metrics["execution_reference"]["completion_tokens"]
        != response.usage.output_tokens
    ):
        return None
    return metrics
