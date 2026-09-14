# SPDX-License-Identifier: Apache-2.0
"""Small-model agent orchestration contracts.

The runtime kernel owns bounded progress and wire-visible state. Model I/O and
tool execution stay behind adapters so the headless server and Rapid Desktop
can share one state machine without sharing implementation language.
"""

from .models import (
    AgentEvent,
    AgentModelTurn,
    AgentProfile,
    AgentRun,
    AgentRunStatus,
    AgentToolCall,
    AgentToolResult,
    ToolRisk,
    ToolSpec,
)
from .profiles import resolve_agent_profile
from .runtime import AgentRuntime, AgentRuntimeError, AgentRuntimeOutput

__all__ = [
    "AgentEvent",
    "AgentModelTurn",
    "AgentProfile",
    "AgentRun",
    "AgentRunStatus",
    "AgentRuntime",
    "AgentRuntimeError",
    "AgentRuntimeOutput",
    "AgentToolCall",
    "AgentToolResult",
    "ToolRisk",
    "ToolSpec",
    "resolve_agent_profile",
]
