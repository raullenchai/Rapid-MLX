# SPDX-License-Identifier: Apache-2.0
"""Wire events and in-process state for the Rapid Agent Runtime."""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Literal, cast

from jsonschema import validators
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_serializer,
    model_validator,
)

AgentEventType = Literal[
    "run.created",
    "model.requested",
    "tool.requested",
    "approval.required",
    "approval.resolved",
    "tool.completed",
    "synthesis.required",
    "run.completed",
    "run.failed",
    "run.cancelled",
]


def _contains_schema_reference(value: JsonValue) -> bool:
    if isinstance(value, dict):
        if any(key in value for key in ("$ref", "$dynamicRef", "$recursiveRef")):
            return True
        return any(_contains_schema_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_schema_reference(item) for item in value)
    return False


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AgentRunStatus(str, Enum):
    READY = "ready"
    AWAITING_MODEL = "awaiting_model"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_TOOL_RESULT = "awaiting_tool_result"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    LOCAL_CHANGE = "local_change"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"

    @property
    def requires_approval(self) -> bool:
        return self is ToolRisk.EXTERNAL_SIDE_EFFECT


class AgentProfile(_WireModel):
    """Immutable model-specific runtime limits bound at run creation."""

    model_config = ConfigDict(frozen=True)

    name: StrictStr = Field(min_length=1, max_length=128)
    max_visible_tools: StrictInt = Field(ge=0, le=64)
    max_tool_rounds: StrictInt = Field(ge=0, le=128)
    repeated_call_limit: StrictInt = Field(ge=0, le=16)
    attach_ledger_to_tool_results: StrictBool = True


class ToolSpec(_WireModel):
    """The small, explicitly classified surface projected from a tool registry."""

    model_config = ConfigDict(frozen=True)

    name: StrictStr = Field(min_length=1, max_length=128)
    description: StrictStr = Field(default="", max_length=4096)
    parameters_json: StrictStr = Field(default="{}", repr=False)
    risk: ToolRisk

    @field_validator("parameters_json")
    @classmethod
    def validate_parameters_json(cls, value: str) -> str:
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("tool parameters must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("tool parameters must be a JSON object")
        if _contains_schema_reference(decoded):
            raise ValueError("tool parameters must use an inline JSON Schema")
        try:
            validators.validator_for(decoded).check_schema(decoded)
        except Exception as exc:
            raise ValueError("tool parameters must be a valid JSON Schema") from exc
        return json.dumps(
            decoded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def parameters(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], json.loads(self.parameters_json))

    @model_validator(mode="before")
    @classmethod
    def accept_wire_parameters(cls, value):
        if isinstance(value, dict) and "parameters" in value:
            normalized = dict(value)
            if "parameters_json" in normalized:
                raise ValueError(
                    "tool spec must not contain both parameters and parameters_json"
                )
            try:
                normalized["parameters_json"] = json.dumps(
                    normalized.pop("parameters"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("tool parameters must be JSON serializable") from exc
            return normalized
        return value

    @model_serializer(mode="plain")
    def serialize_wire(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "risk": self.risk.value,
        }


class AgentToolCall(_WireModel):
    model_config = ConfigDict(frozen=True)

    id: StrictStr = Field(min_length=1, max_length=256)
    name: StrictStr = Field(min_length=1, max_length=128)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class RedactedPendingCall(_WireModel):
    """In-process call identity; raw arguments stay in transient runtime state."""

    model_config = ConfigDict(frozen=True)

    id: StrictStr = Field(min_length=1, max_length=256)
    name: StrictStr = Field(min_length=1, max_length=128)


class AgentToolResult(_WireModel):
    call_id: StrictStr = Field(min_length=1, max_length=256)
    content: StrictStr = Field(max_length=262_144)
    is_error: StrictBool = False
    executed: StrictBool = True
    safe_summary: StrictStr | None = Field(default=None, max_length=1024)


class AgentModelTurn(_WireModel):
    content: StrictStr = Field(default="", max_length=262_144)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)


class AgentEvent(_WireModel):
    """Immutable wire event consumed by both headless and Desktop clients."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    sequence: StrictInt = Field(ge=1)
    type: AgentEventType
    created_at: float = Field(ge=0, allow_inf_nan=False)
    payload_json: StrictStr = Field(default="{}", repr=False)

    @field_validator("payload_json")
    @classmethod
    def validate_payload_json(cls, value: str) -> str:
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("event payload must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("event payload must be a JSON object")
        return json.dumps(
            decoded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def data(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], json.loads(self.payload_json))

    @model_validator(mode="before")
    @classmethod
    def accept_wire_data(cls, value):
        if isinstance(value, dict) and "data" in value:
            normalized = dict(value)
            if "payload_json" in normalized:
                raise ValueError("event must not contain both data and payload_json")
            try:
                normalized["payload_json"] = json.dumps(
                    normalized.pop("data"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("event data must be JSON serializable") from exc
            return normalized
        return value

    @model_serializer(mode="plain")
    def serialize_wire(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "type": self.type,
            "created_at": self.created_at,
            "data": self.data,
        }


class AgentRun(_WireModel):
    """Frozen, process-local reducer state. It is not a public restore format."""

    model_config = ConfigDict(frozen=True)

    id: StrictStr = Field(
        default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=256
    )
    model: StrictStr = Field(min_length=1)
    goal: StrictStr = Field(min_length=1, max_length=65_536)
    profile: AgentProfile
    status: AgentRunStatus = AgentRunStatus.READY
    model_turns: StrictInt = Field(default=0, ge=0)
    tool_rounds: StrictInt = Field(default=0, ge=0)
    final_synthesis: StrictBool = False
    visible_tools: tuple[ToolSpec, ...] = Field(default_factory=tuple)
    pending_call: RedactedPendingCall | None = None
    pending_risk: ToolRisk | None = None
    used_call_ids: tuple[StrictStr, ...] = Field(default_factory=tuple)
    failure_code: StrictStr | None = Field(default=None, max_length=128)
    events: tuple[AgentEvent, ...] = Field(default_factory=tuple)
