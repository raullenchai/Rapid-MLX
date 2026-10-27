# SPDX-License-Identifier: Apache-2.0
"""Small wire-v1 client reducer; no model or tool execution lives here."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class ReplayRequiredError(ValueError):
    """The next event is missing; reconnect at the last applied cursor."""


def sse_events(lines: Iterable[str]):
    """Decode bounded SSE frames, accepting comments and multiline data."""
    data: list[str] = []
    event_id = event_type = ""
    size = 0
    for line in lines:
        if not line:
            if data:
                event = json.loads("\n".join(data))
                expected_id = f"{event['turn_id']}:{event['seq']}"
                if event_id != expected_id or event_type != event["event"]["type"]:
                    raise ValueError("Invalid chat event identity")
                yield event
            data, event_id, event_type, size = [], "", "", 0
            continue
        size += len(line.encode("utf-8"))
        if size > 2 * 1024 * 1024:
            raise ValueError("Chat event exceeds size limit")
        if line.startswith(":"):
            continue
        key, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if key == "data":
            data.append(value)
        elif key == "id":
            event_id = value
        elif key == "event":
            event_type = value
    # An incomplete frame is never applied. The caller reconnects from cursor.


@dataclass
class TurnState:
    turn_id: str
    cursor: int = 0
    parts: dict[str, str] = field(default_factory=dict)
    final_text: str = ""
    reasoning: str = ""
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    subagents: dict[str, dict[str, Any]] = field(default_factory=dict)
    timeline: list[tuple[str, str]] = field(default_factory=list)
    notices: dict[str, str] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    terminal: dict[str, Any] | None = None
    reasoning_since: float | None = None
    reasoning_elapsed: float = 0.0

    @property
    def reasoning_seconds(self) -> float:
        elapsed = self.reasoning_elapsed
        if self.reasoning_since is not None:
            elapsed += time.monotonic() - self.reasoning_since
        return elapsed

    def end_reasoning(self) -> None:
        if self.reasoning_since is not None:
            self.reasoning_elapsed += time.monotonic() - self.reasoning_since
            self.reasoning_since = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts.values())

    def apply(self, envelope: dict[str, Any]) -> bool:
        if envelope.get("wire_version") != 1 or envelope.get("turn_id") != self.turn_id:
            raise ValueError("Incompatible chat event")
        seq = envelope["seq"]
        if not isinstance(seq, int) or seq < 1:
            raise ValueError("Invalid chat sequence")
        if seq <= self.cursor:
            return False
        if seq != self.cursor + 1:
            raise ReplayRequiredError("Chat stream has a gap")
        if self.terminal is not None:
            raise ValueError("Event after terminal")
        data = envelope["event"]
        kind = data["type"]
        if kind in {"text.delta", "tool.started", "tool.result", "turn.terminal"}:
            self.end_reasoning()
        if kind == "text.delta":
            part = data["part_id"]
            if part not in self.parts:
                self.timeline.append(("text", part))
            self.parts[part] = self.parts.get(part, "") + data["text"]
            self.final_text += data["text"]
        elif kind == "reasoning.delta" and not data.get("omitted"):
            if data.get("text") and self.reasoning_since is None:
                self.reasoning_since = time.monotonic()
            # Volatile only, with a bounded preview even on very long turns.
            self.reasoning = (self.reasoning + data.get("text", ""))[-1_000_000:]
        elif kind == "tool.started":
            self.final_text = ""
            self.timeline.append(("tool", data["call_id"]))
            self.tools[data["call_id"]] = {
                "name": data["name"],
                "args": data.get("args", {}),
                "summary": data.get("summary", ""),
                "label": data.get("label", ""),
                "target": data.get("target", ""),
                "status": "running",
            }
        elif kind == "tool.result":
            tool = self.tools.setdefault(data["call_id"], {"name": "Tool"})
            tool["status"] = data.get("outcome", "completed")
            for key in (
                "label",
                "target",
                "summary",
                "details",
                "details_kind",
                "omitted_lines",
            ):
                if key in data:
                    tool[key] = data[key]
            tool["has_result"] = True
            self.timeline.append(("result", data["call_id"]))
        elif kind in {"subagent.started", "subagent.progress", "subagent.finished"}:
            self.end_reasoning()
            self.subagents[data["id"]] = {
                key: data[key]
                for key in ("id", "title", "status", "call_id", "tool_count", "summary")
            }
        elif kind == "source":
            self.sources[data["source_id"]] = data
        elif kind in {"context", "usage"}:
            self.context.update(data)
            notice = data.get("notice")
            if notice:
                key = str(seq)
                self.notices[key] = notice
                self.timeline.append(("notice", key))
        elif kind == "turn.terminal":
            if data["outcome"] not in {"completed", "cancelled", "error"}:
                raise ValueError("Unknown turn outcome; upgrade the client")
            if data["outcome"] == "completed" and not self.final_text.strip():
                raise ValueError("Completed chat turn has no visible answer")
            self.terminal = data
            for tool in self.tools.values():
                if tool["status"] == "running":
                    tool["status"] = data["outcome"]
            for child in self.subagents.values():
                if child["status"] in {"queued", "running"}:
                    child["status"] = (
                        "cancelled" if data["outcome"] == "cancelled" else "error"
                    )
        self.cursor = seq
        return True
