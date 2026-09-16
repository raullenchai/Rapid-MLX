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
from .profiles import (
    PERSONAL_INTELLIGENCE_QUALIFICATIONS,
    PersonalIntelligenceQualification,
    resolve_agent_profile,
    resolve_personal_intelligence_profile,
    resolve_personal_intelligence_qualification,
)
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
    "PERSONAL_INTELLIGENCE_QUALIFICATIONS",
    "PersonalIntelligenceQualification",
    "ToolRisk",
    "ToolSpec",
    "resolve_agent_profile",
    "resolve_personal_intelligence_qualification",
    "resolve_personal_intelligence_profile",
]
