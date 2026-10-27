"""Version-one chat wire types, independent of model-provider events."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Usage(WireModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)


class TurnStarted(WireModel):
    type: Literal["turn.started"] = "turn.started"
    message_id: str
    parent_message_id: str | None = None
    branch_id: str
    model: str
    thinking: bool


class TextDelta(WireModel):
    type: Literal["text.delta"] = "text.delta"
    part_id: str
    text: str


class ReasoningDelta(WireModel):
    type: Literal["reasoning.delta"] = "reasoning.delta"
    part_id: str
    text: str = ""
    omitted: bool = False


class ToolStarted(WireModel):
    type: Literal["tool.started"] = "tool.started"
    call_id: str
    name: str
    label: str = ""
    target: str = ""
    summary: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(WireModel):
    type: Literal["tool.result"] = "tool.result"
    call_id: str
    result: str
    label: str = ""
    target: str = ""
    summary: str = ""
    details: list[str] = Field(default_factory=list)
    details_kind: Literal["diff", "output", "text"] = "text"
    omitted_lines: int = Field(default=0, ge=0)
    outcome: Literal["success", "error"] = "success"


class PermissionRequest(WireModel):
    type: Literal["permission.request"] = "permission.request"
    op_id: str
    revision: int = Field(ge=0)
    tool: str
    title: str = "Allow this action?"
    preview_language: Literal["python", "bash", "diff", "text"] = "text"
    summary: str
    preview: str
    risk: str
    reason: str
    options: list[Literal["once", "conversation", "deny"]]
    expires_at: float
    subagent_id: str | None = None


class SubagentEvent(WireModel):
    id: str
    call_id: str
    title: str
    status: Literal[
        "queued",
        "running",
        "completed",
        "error",
        "cancelled",
        "budget_exhausted",
        "timeout",
    ]
    tool_count: int = Field(default=0, ge=0)
    summary: str = ""


class SubagentStarted(SubagentEvent):
    type: Literal["subagent.started"] = "subagent.started"


class SubagentProgress(SubagentEvent):
    type: Literal["subagent.progress"] = "subagent.progress"


class SubagentFinished(SubagentEvent):
    type: Literal["subagent.finished"] = "subagent.finished"


class ToolOutput(WireModel):
    type: Literal["tool.output"] = "tool.output"
    call_id: str
    text: str


class PermissionResponse(WireModel):
    op_id: str
    revision: int = Field(ge=0)
    decision: Literal["allow", "deny"]
    scope: Literal["once", "conversation"] = "once"
    reason: str = Field(default="", max_length=512)


class Source(WireModel):
    """Provenance from a read-only tool; URLs are references, never fetched."""

    type: Literal["source"] = "source"
    source_id: str
    call_id: str
    title: str
    attachment_id: str | None = None
    url: str | None = None


class UsageEvent(Usage):
    type: Literal["usage"] = "usage"


class ContextEvent(WireModel):
    type: Literal["context"] = "context"
    used_tokens: int = Field(ge=0)
    limit_tokens: int = Field(ge=1)
    system_tools_tokens: int = Field(ge=0)
    conversation_tokens: int = Field(ge=0)
    tool_output_tokens: int = Field(ge=0)
    free_tokens: int = Field(ge=0)
    percent: int = Field(ge=0, le=100)
    estimated: bool = False
    compacted: bool = False
    notice: str = ""


class TurnTerminal(WireModel):
    type: Literal["turn.terminal"] = "turn.terminal"
    outcome: Literal["completed", "cancelled", "error"]
    final_message_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    error_code: str | None = None


Event = Annotated[
    TurnStarted
    | TextDelta
    | ReasoningDelta
    | ToolStarted
    | ToolResult
    | ToolOutput
    | PermissionRequest
    | SubagentStarted
    | SubagentProgress
    | SubagentFinished
    | Source
    | UsageEvent
    | ContextEvent
    | TurnTerminal,
    Field(discriminator="type"),
]


class Envelope(WireModel):
    wire_version: Literal[1] = 1
    turn_id: str
    seq: int = Field(ge=1)
    event: Event

    @property
    def event_id(self) -> str:
        return f"{self.turn_id}:{self.seq}"

    def sse(self) -> bytes:
        return (
            f"id: {self.event_id}\nevent: {self.event.type}\n"
            f"data: {self.model_dump_json()}\n\n"
        ).encode()


def json_schema() -> dict:
    from .runtime import TurnRequest

    schema = Envelope.model_json_schema()
    schema["$defs"]["PermissionResponse"] = PermissionResponse.model_json_schema()
    schema["$defs"]["TurnRequest"] = TurnRequest.model_json_schema(mode="validation")
    schema["x-turn-request"] = {"$ref": "#/$defs/TurnRequest"}
    schema["x-permission-response"] = {"$ref": "#/$defs/PermissionResponse"}
    return schema
