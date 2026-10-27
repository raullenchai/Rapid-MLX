"""Bounded, owner-scoped helper runs with isolated histories and shared authority."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.usage import RunUsage, UsageLimits

from .presentation import ToolPresentation
from .progress import tool_progress
from .store import Conflict
from .tools import TurnDeps, available_tools


@dataclass(frozen=True)
class ChildLimits:
    requests: int = 6
    tool_calls: int = 8
    tokens: int = 12000
    timeout: float = 120
    summary_bytes: int = 4000


def _usage(usage):
    return {
        key: getattr(usage, key, 0)
        for key in ("input_tokens", "output_tokens", "requests")
    }


_CITATIONS = re.compile(r"\[(\s*\d+(?:\s*,\s*\d+)*\s*)\]")
_SUMMARY_MARKER = (
    "\n[Summary truncated; see the helper transcript for omitted details.]"
)


def _summary_fragment(text, end):
    if end >= len(text):
        return text
    prefix = text[:end]
    # Prefer complete words when a whitespace boundary is available. The marker
    # still makes clipping explicit for scripts that do not separate words.
    if end and not text[end].isspace():
        boundary = max(prefix.rfind(" "), prefix.rfind("\n"), prefix.rfind("\t"))
        if boundary >= 0:
            prefix = prefix[:boundary]
    return prefix.rstrip() + _SUMMARY_MARKER


def _clip(text, size):
    if len(text.encode("utf-8")) <= size:
        return text
    available = max(0, size - len(_SUMMARY_MARKER.encode()))
    prefix = text.encode("utf-8")[:available].decode("utf-8", errors="ignore")
    result = _summary_fragment(text, len(prefix))
    return result.encode("utf-8")[: max(0, size)].decode("utf-8", errors="ignore")


def child_instructions(ctx):
    from .prompts import build_instructions

    return build_instructions(ctx.deps, available_tools(ctx.deps), child=True)


def _snapshot_messages(messages):
    from .runtime import _durable_messages

    rows = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                rows.append(
                    {"role": "user", "parts": [{"type": "text", "text": part.content}]}
                )
        rows.extend(_durable_messages([message]))
    return rows


class DelegationManager:
    def __init__(self, runtime, *, limits=None):
        self.runtime = runtime
        self.store = runtime.store
        self.limits = limits or ChildLimits()
        self._slots = asyncio.Semaphore(3)
        self._runs = {}
        self._closed = False

    async def run(self, ctx, task, context=None, tools=None):
        parent = ctx.deps
        if parent.subagent_id or self._closed:
            return "Delegation unavailable: helpers cannot delegate and a closing owner cannot start work."
        if not isinstance(task, str) or not task.strip() or len(task.encode()) > 8192:
            return "Delegate a nonempty focused task of at most 8192 bytes."
        if context is not None and len(context.encode()) > 16384:
            return "Delegation context exceeds 16384 bytes; provide a focused excerpt."
        permitted = available_tools(parent) - {"delegate_task"}
        selected = permitted if tools is None else set(tools)
        if not selected.issubset(permitted):
            return "Unavailable helper tools: " + ", ".join(
                sorted(selected - permitted)
            )
        call_id = ctx.tool_call_id or ""
        key = (parent.turn_id, call_id)
        signature = (task, context, tuple(sorted(selected)))
        existing = self._runs.get(key)
        if existing:
            if existing[0] != signature:
                return "Delegation call ID was reused with different arguments; no helper was started."
            work = existing[1]
        else:
            work = asyncio.create_task(self._execute(ctx, task, context, selected))
            self._runs[key] = (signature, work)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            if not work.cancelling():
                work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            raise

    async def _execute(self, ctx, task, context, selected):
        from .runtime import chat_agent, turn_clock

        parent = ctx.deps
        call_id = ctx.tool_call_id or ""
        child_id = "sa-" + uuid.uuid4().hex
        title = " ".join(task.split())[:120]
        row = await self.store.create_subrun(
            id=child_id,
            turn_id=parent.turn_id,
            call_id=call_id,
            title=title,
            task=task,
            context=context,
            tools=sorted(selected),
        )
        if row["id"] != child_id:
            if row["status"] not in {"queued", "running"}:
                return json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "summary": row["summary"],
                        "sources": row["sources"],
                    },
                    ensure_ascii=False,
                )
            raise Conflict("delegation already running")
        usage = RunUsage()
        tool_count = 0
        evidence = []
        status = "error"
        summary = "The helper did not complete its task."
        cancelled = False
        limits = self.limits
        common = {"id": child_id, "title": title, "call_id": call_id}

        async def progress(kind, state, activity=""):
            await parent.emit(
                {
                    "type": kind,
                    **common,
                    "status": state,
                    "tool_count": tool_count,
                    "summary": activity,
                }
            )

        async def snapshot(items, consumed):
            await self.store.update_subrun(
                child_id,
                messages=_snapshot_messages(items),
                usage=_usage(consumed),
                tool_count=tool_count,
            )

        presenter = ToolPresentation(parent.workspace)

        async def emit(event):
            nonlocal tool_count
            operation = None
            if event["type"] == "tool.started" and event.get("args", {}).get("op_id"):
                operation = await self.store.get_operation(event["args"]["op_id"])
            event = presenter.event(event, operation)
            await self.store.append_subrun_event(child_id, event)
            kind = event["type"]
            if kind == "permission.request":
                await parent.emit({**event, "subagent_id": child_id})
            elif kind == "tool.started":
                tool_count += 1
                activity = tool_progress(event["name"], event.get("args", {}))
                await self.store.update_subrun(child_id, tool_count=tool_count)
                await progress("subagent.progress", "running", activity)
            elif kind == "tool.result":
                evidence.append(event.get("result", ""))
                await progress("subagent.progress", "running", "Tool completed")
            return event

        deps = TurnDeps(
            self.store,
            parent.conversation_id,
            parent.attachment_ids,
            emit,
            workspace=parent.workspace,
            model=parent.model,
            surface=parent.surface,
            prompt_workspace=parent.prompt_workspace,
            turn_id=parent.turn_id,
            actions=parent.actions,
            action_tools=parent.action_tools,
            approvals_always=parent.approvals_always,
            web=parent.web,
            subagent_id=child_id,
            allowed_tools=frozenset(selected),
            context_budget=parent.context_budget,
            run_usage=usage,
            usage_limits=UsageLimits(
                request_limit=limits.requests,
                tool_calls_limit=limits.tool_calls,
                total_tokens_limit=limits.tokens,
            ),
            snapshot=snapshot,
        )
        prompt = f"{turn_clock()}\nTask: {task}"
        if context:
            prompt += "\nProvided context:\n" + context
        history = [ModelRequest(parts=[UserPromptPart(prompt)])]
        settings = dict(ctx.model_settings or {})
        settings["max_tokens"] = min(
            settings.get("max_tokens") or 1024, 1024, max(1, limits.tokens)
        )
        agent = chat_agent(child_instructions)
        try:
            await parent.emit(
                {
                    "type": "tool.started",
                    "call_id": call_id,
                    "name": "delegate_task",
                    "args": {"task": task},
                }
            )
            await progress("subagent.started", "queued")
            await snapshot(history, usage)
            async with asyncio.timeout(limits.timeout):
                async with self._slots:
                    await self.store.update_subrun(child_id, status="running")
                    await progress("subagent.progress", "running", "Working")
                    final, additions, _ = await self.runtime._attempt(
                        agent,
                        ctx.model,
                        history,
                        deps,
                        settings,
                        0,
                    )
                    history.extend(additions)
                    if not final.strip():
                        # One tools-disabled synthesis uses the SAME cumulative
                        # budget; a spent budget falls back to recorded evidence.
                        final, additions, _ = await self.runtime._attempt(
                            self.runtime.synthesis_agent,
                            ctx.model,
                            history,
                            deps,
                            settings,
                            1,
                        )
                        history.extend(additions)
                    if final.strip():
                        summary = final
                        status = "completed"
                    else:
                        status = "error"
        except UsageLimitExceeded:
            status = "budget_exhausted"
        except TimeoutError:
            status = "timeout"
        except asyncio.CancelledError:
            status = "cancelled"
            cancelled = True
        except Exception:
            # Never serialize provider exception strings, credentials or reasoning.
            status = "error"
        finally:
            if status != "completed":
                label = {
                    "budget_exhausted": "reached its budget",
                    "timeout": "timed out",
                    "cancelled": "was cancelled",
                }.get(status, "could not finish")
                summary = f"Partial result: the helper {label}. "
                if evidence:
                    summary += "Retrieved evidence (incomplete):\n" + "\n\n".join(
                        evidence[-3:]
                    )
                else:
                    summary += (
                        "No tool evidence was retrieved; the task remains unverified."
                    )
            summary = _clip(summary, limits.summary_bytes)
            await self.store.update_subrun(
                child_id,
                usage=_usage(usage),
                status=status,
                summary=summary,
                sources=list(deps.sources.values()),
                tool_count=tool_count,
            )
            parent.actions.release_child(parent.turn_id, child_id)
            for name, value in _usage(usage).items():
                parent.child_usage[name] += value
            activity = {
                "completed": "Completed",
                "timeout": "Timed out after",
                "cancelled": "Cancelled after",
                "budget_exhausted": "Budget reached after",
                "error": "Stopped after",
            }[status]
            await progress(
                "subagent.finished", status, f"{activity} {tool_count} tools"
            )
        if cancelled:
            raise asyncio.CancelledError
        summary, sources = await self._promote_sources(
            parent, call_id, summary, deps.sources
        )
        # Three summaries fit within the parent's 16 KB allowance, including
        # JSON escaping and source metadata. Full sources remain in the journal.
        result = self._result(child_id, status, summary, sources, limits.summary_bytes)
        await parent.emit(
            {
                "type": "tool.result",
                "call_id": call_id,
                "result": result,
                "outcome": "success" if status == "completed" else "error",
            }
        )
        return result

    @staticmethod
    def _result(child_id, status, summary, sources, budget):
        payload = {"id": child_id, "status": status, "summary": "", "sources": []}
        cited = {
            number
            for group in _CITATIONS.findall(summary)
            for number in re.findall(r"\d+", group)
        }
        ordered = sorted(sources, key=lambda source: source["source_id"] not in cited)
        for source in ordered:
            item = {
                key: source[key]
                for key in ("source_id", "title", "url")
                if key in source
            }
            candidate = {**payload, "sources": [*payload["sources"], item]}
            if len(json.dumps(candidate, ensure_ascii=False).encode()) > budget // 2:
                continue
            payload["sources"].append(item)
        if len(payload["sources"]) < len(sources):
            payload["source_note"] = (
                "Additional source details are in the helper transcript and source list."
            )
        # JSON escapes can expand the text, so fit the encoded envelope, not just
        # the UTF-8 summary. Even the smallest configured budget keeps a status.
        payload["summary"] = summary
        if len(json.dumps(payload, ensure_ascii=False).encode()) <= max(512, budget):
            return json.dumps(payload, ensure_ascii=False)
        lo, hi = 0, max(0, len(summary) - 1)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            payload["summary"] = _summary_fragment(summary, mid)
            if len(json.dumps(payload, ensure_ascii=False).encode()) <= max(
                512, budget
            ):
                lo = mid
            else:
                hi = mid - 1
        payload["summary"] = _summary_fragment(summary, lo)
        return json.dumps(payload, ensure_ascii=False)

    async def _promote_sources(self, parent, call_id, summary, sources):
        mapping = {}
        pending = []
        promoted = []
        # Reserve IDs without yielding so concurrently completed helpers cannot
        # choose the same number for different URLs.
        for url, source in sources.items():
            item = parent.sources.get(url)
            if item is None:
                item = {
                    **source,
                    "source_id": str(len(parent.sources) + 1),
                    "call_id": call_id,
                }
                parent.sources[url] = item
                pending.append(item)
            mapping[source["source_id"]] = item["source_id"]
            promoted.append(item)
        summary = _CITATIONS.sub(
            lambda match: (
                "["
                + re.sub(
                    r"\d+", lambda number: mapping.get(number[0], number[0]), match[1]
                )
                + "]"
            ),
            summary,
        )
        for item in pending:
            await parent.emit(item)
        return summary, promoted

    async def cancel_turn(self, turn_id):
        tasks = [
            work
            for (tid, _), (_, work) in self._runs.items()
            if tid == turn_id and not work.done()
        ]
        for work in tasks:
            if not work.cancelling():
                work.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def release_turn(self, turn_id):
        for key in list(self._runs):
            if key[0] == turn_id:
                self._runs.pop(key)

    async def close(self):
        self._closed = True
        for turn_id in {key[0] for key in self._runs}:
            await self.cancel_turn(turn_id)
        self._runs.clear()
