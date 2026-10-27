"""Background turn ownership, model execution, and durable event publication."""

from __future__ import annotations

import asyncio
import locale
import logging
import os
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import (
    IncompleteToolCall,
    ModelHTTPError,
    UsageLimitExceeded,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import RunUsage, UsageLimits

from .context import (
    ContextManager,
    ContextSynthesisUnavailableError,
    RuntimeNoticePart,
    context_event,
)
from .events import Envelope
from .history import _project_messages
from .model_client import ModelClients, inference_url
from .presentation import ToolPresentation
from .tools import TurnDeps, available_tools, toolset
from .tools.action_tools import ActionTools
from .tools.web import WebTools

logger = logging.getLogger(__name__)
_TRUNCATED_CALL = object()


class _SynthesisRequiredError(Exception):
    """Stop before inference when context preparation ends the tool phase."""


def _localized(text, zh, en):
    return zh if re.search(r"[\u3400-\u9fff]", text) else en


class _AttemptError(Exception):
    """Carry completed work out of an unsuccessful agent run without logging inputs."""

    def __init__(self, error, messages, usage):
        super().__init__(type(error).__name__)
        self.error, self.messages, self.usage = error, messages, usage


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    v: Literal[1] = 1
    turn_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    conversation_id: str
    workspace: str | None = Field(default=None, max_length=4096)
    branch_id: str
    expected_revision: int = Field(ge=0)
    text: str = Field(default="", max_length=262144)
    attachment_ids: list[str] = Field(default_factory=list, max_length=32)
    inference_url: str
    model: str = Field(min_length=1, max_length=512)
    surface: Literal["desktop", "cli", "plain"] = "desktop"
    thinking: bool = False
    action_tools: bool = True
    approvals: Literal["default", "always"] = "default"
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=32768, ge=1, le=32768)
    context_tokens: int = Field(default=32768, ge=512, le=131072)
    mode: Literal["append", "regenerate"] = "append"
    inference_api_key: SecretStr | None = Field(default=None, exclude=True)

    @field_validator("inference_url")
    @classmethod
    def local_endpoint(cls, value: str) -> str:
        return inference_url(value)


# Re-exported for callers that share the chat identity.
from .prompts import BASE_INSTRUCTIONS as BASE_INSTRUCTIONS
from .prompts import build_instructions


def turn_clock(timestamp: datetime | None = None) -> str:
    """Local clock goes beside the user turn, after the stable system/tool prefix."""
    now = (timestamp or datetime.now().astimezone()).astimezone()
    zone: str | None = os.environ.get("TZ", "")
    if not zone:
        target = str(Path("/etc/localtime").resolve())
        zone = target.split("zoneinfo/")[-1] if "zoneinfo/" in target else now.tzname()
    language = locale.getlocale()[0] or "unknown"
    return (
        f"[Local time: {now:%A %Y-%m-%d %H:%M %Z} (UTC{now:%z}); "
        f"timezone: {zone}; locale: {language} (not a precise location).]"
    )


def _dated_history(messages):
    result = []
    for message in messages:
        parts = list(message.parts)
        for index, part in enumerate(parts):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                parts[index] = replace(
                    part, content=f"{turn_clock(part.timestamp)}\n{part.content}"
                )
                break
        result.append(replace(message, parts=parts))
    return result


def latest_prompt(history):
    return next(
        (
            p.content
            for m in reversed(history)
            for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        ),
        "",
    )


class BoundedHistory(AbstractCapability[TurnDeps]):
    async def before_model_request(self, ctx, request_context):
        if ctx.deps.prepare_context:
            had_tools = bool(request_context.model_request_parameters.function_tools)
            prepared = await ctx.deps.prepare_context(ctx, request_context)
            if (
                had_tools
                and ctx.deps.result_budget_exhausted
                and not prepared.model_request_parameters.function_tools
            ):
                # Context hooks run inside stream setup, after the outer budget
                # check. Start the summary agent before any generation with the
                # now-stale action instructions can occur.
                raise _SynthesisRequiredError()
            return prepared
        from .context import prepare_helper_context

        return await prepare_helper_context(ctx, request_context)


def chat_instructions(ctx: RunContext[TurnDeps]) -> str:
    return build_instructions(ctx.deps, available_tools(ctx.deps))


def synthesis_instructions(ctx: RunContext[TurnDeps]) -> str:
    instructions = build_instructions(ctx.deps, set(), synthesis=True)
    if ctx.deps.result_budget_exhausted:
        # Adapted summary request; see LICENSE-PROMPTS. Keep this in the system
        # instructions so the human's language remains the conversation language.
        instructions += (
            "\n\nYou've reached the tool budget for this turn. Please provide a final response "
            "summarizing what you've found and accomplished so far, without calling any more tools. "
            "Distinguish completed work, verified results, and what remains unfinished. "
            "Use the user's language."
        )
    return instructions


def chat_agent(instructions, *, tools=True):
    return Agent(
        deps_type=TurnDeps,
        instructions=instructions,
        toolsets=[toolset()] if tools else [],
        output_type=[str, DeferredToolRequests] if tools else str,
        capabilities=[BoundedHistory()],
        retries=0,
    )


class Runtime:
    def __init__(
        self,
        store,
        *,
        model: Model | None = None,
        web_tools: WebTools | None = None,
        web_enabled: bool | None = None,
        usage_limits: UsageLimits | None = None,
    ):
        self.store = store
        self.actions = ActionTools(store)
        self.clients = ModelClients()
        self.model_override = model
        # Parent turns have no default iteration/tool-count ceiling. Embedders
        # may supply a finite limit; result bytes and context remain bounded.
        self.usage_limits = usage_limits or UsageLimits(request_limit=None)
        from .delegation import DelegationManager

        self.delegation = DelegationManager(self)
        if web_enabled is None:
            web_enabled = os.environ.get(
                "RAPID_MLX_CHAT_WEB_TOOLS", "1"
            ).lower() not in {"0", "false", "off"}
        self.web_tools = (web_tools or WebTools()) if web_enabled else None
        self.agent = chat_agent(chat_instructions)
        self.synthesis_agent = chat_agent(synthesis_instructions, tools=False)
        self._presentations: dict[str, ToolPresentation] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._signals: dict[str, asyncio.Event] = {}
        # Transient reasoning can be observed live but is never replayed from disk.
        self._volatile: dict[str, dict[int, dict]] = {}
        self._submit_lock = asyncio.Lock()
        self._admissions: set[asyncio.Task] = set()
        self._cancellations: dict[str, asyncio.Task] = {}
        self._readers: dict[str, int] = {}
        self._closed = False

    @property
    def active_turns(self) -> int:
        return (
            sum(not task.done() for task in self._tasks.values())
            + len(self._admissions)
            + len(self._cancellations)
            + len(self.actions.undo_tasks)
        )

    async def submit(self, request: TurnRequest) -> tuple[dict, bool]:
        if self._closed:
            raise RuntimeError("runtime_closed")
        # Admission outlives the HTTP request. Cancellation after the writer
        # commits must never leave a durable running turn without an owner.
        admission = asyncio.create_task(self._admit(request))
        self._admissions.add(admission)

        def admitted(task):
            self._admissions.discard(task)
            if not task.cancelled():
                task.exception()  # Observe detached conflicts without logging inputs.

        admission.add_done_callback(admitted)
        return await asyncio.shield(admission)

    async def _admit(self, request: TurnRequest) -> tuple[dict, bool]:
        async with self._submit_lock:
            if self._closed:
                raise RuntimeError("runtime_closed")
            turn, created = await self.store.begin_turn(request.model_dump())
            if created:
                self._signals[request.turn_id] = asyncio.Event()
                self._volatile[request.turn_id] = {}
                task = asyncio.create_task(self._run(request, turn))
                self._tasks[request.turn_id] = task
                task.add_done_callback(
                    lambda done: self._turn_done(request.turn_id, done)
                )
            return turn, created

    def _release_turn(self, turn_id):
        task = self._tasks.get(turn_id)
        if not self._readers.get(turn_id) and (task is None or task.done()):
            self._signals.pop(turn_id, None)
            self._volatile.pop(turn_id, None)
            self._readers.pop(turn_id, None)
            self._presentations.pop(turn_id, None)

    def _turn_done(self, turn_id, task):
        self._tasks.pop(turn_id, None)
        signal = self._signals.get(turn_id)
        if signal:
            signal.set()
        if not task.cancelled():
            task.exception()
        self._release_turn(turn_id)

    async def _emit(self, turn_id: str, event: dict) -> dict:
        presenter = self._presentations.setdefault(turn_id, ToolPresentation())
        operation = None
        op_id = (
            event.get("args", {}).get("op_id")
            if event["type"] == "tool.started"
            else None
        )
        if op_id:
            operation = await self.store.get_operation(op_id)
        event = presenter.event(event, operation=operation)
        envelope: dict = await self.store.append_event(turn_id, event)
        # Validate before publication, including store-produced sequencing.
        Envelope.model_validate(envelope)
        if event["type"] == "reasoning.delta":
            volatile = self._volatile.setdefault(turn_id, {})
            # Bound volatile memory independently of the durable event journal.
            if len(volatile) >= 4096:
                volatile.pop(next(iter(volatile)))
            volatile[envelope["seq"]] = {**envelope, "event": event}
        self._signals.setdefault(turn_id, asyncio.Event()).set()
        return envelope

    async def events(self, turn_id: str, after_seq: int = 0):
        task = self._tasks.get(turn_id)
        live = task is not None and not task.done()
        self._readers[turn_id] = self._readers.get(turn_id, 0) + 1
        try:
            async for envelope in self._events(turn_id, after_seq, live):
                yield envelope
        finally:
            self._readers[turn_id] -= 1
            self._release_turn(turn_id)

    def _observed_event(self, turn_id, envelope, live):
        if live:
            return self._volatile.get(turn_id, {}).get(envelope["seq"], envelope)
        return envelope

    async def _events(self, turn_id, after_seq, live):
        signal = self._signals.setdefault(turn_id, asyncio.Event())
        while True:
            signal.clear()
            events = await self.store.replay_events(turn_id, after_seq)
            for envelope in events:
                after_seq = envelope["seq"]
                yield self._observed_event(turn_id, envelope, live)
                if envelope["event"]["type"] == "turn.terminal":
                    return
            turn = await self.store.get_turn(turn_id)
            if turn["status"] != "running":
                # A terminal may have committed between the replay and status reads.
                for envelope in await self.store.replay_events(turn_id, after_seq):
                    yield self._observed_event(turn_id, envelope, live)
                return
            try:
                await asyncio.wait_for(signal.wait(), 10)
            except asyncio.TimeoutError:
                yield None  # Transport heartbeat; never an event/cursor.

    async def cancel(self, turn_id: str):
        # Stop owns cleanup independently of the requesting connection. In
        # particular a second disconnected Stop must not interrupt stream close.
        operation = self._cancellations.get(turn_id)
        if operation is None:
            operation = asyncio.create_task(self._cancel(turn_id))
            self._cancellations[turn_id] = operation

            def finished(task):
                self._cancellations.pop(turn_id, None)
                if not task.cancelled():
                    task.exception()

            operation.add_done_callback(finished)
        return await asyncio.shield(operation)

    async def _cancel(self, turn_id: str):
        await self.store.get_turn(turn_id)
        task = self._tasks.get(turn_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Cancellation can arrive before the task executes its first instruction.
            await self.store.finish_turn(turn_id, "cancelled")
            signal = self._signals.get(turn_id)
            if signal:
                signal.set()
        return await self.store.get_turn(turn_id)

    async def close(self):
        self._closed = True
        await asyncio.gather(*self._admissions, return_exceptions=True)
        await asyncio.gather(*self._cancellations.values(), return_exceptions=True)
        active = dict(self._tasks)
        tasks = [task for task in active.values() if not task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for turn_id in active:
            await self.store.finish_turn(turn_id, "cancelled")
            signal = self._signals.get(turn_id)
            if signal:
                signal.set()
        await self.delegation.close()
        await self.actions.close()
        await self.clients.close()
        self._volatile.clear()

    async def _context_manager(self, request, *, running=False):
        rows = await self.store.branch_messages(request.branch_id)
        conversation = await self.store.get_conversation(
            request.conversation_id, request.branch_id
        )
        deps = TurnDeps(
            self.store,
            request.conversation_id,
            frozenset(
                p["attachment_id"]
                for row in rows
                for p in row["parts"]
                if p["type"] == "attachment"
            ),
            lambda event: self._emit(request.turn_id, event),
            workspace=conversation.get("workspace"),
            model=request.model,
            surface=request.surface,
            web=self.web_tools,
            action_tools=request.action_tools,
            delegation=self.delegation,
            turn_id=request.turn_id if running else "",
            actions=self.actions if running else None,
            approvals_always=request.approvals == "always",
            usage_limits=self.usage_limits,
        )
        model = self.model_override or self.clients.model(
            request.inference_url,
            request.model,
            (
                request.inference_api_key.get_secret_value()
                if request.inference_api_key
                else None
            ),
        )
        return ContextManager(self, request, deps, rows, model)

    async def context_info(self, request):
        manager = await self._context_manager(request)
        # Build exactly the exposed function schema and dynamic instructions.
        from types import SimpleNamespace

        from pydantic_ai.models import ModelRequestParameters
        from pydantic_ai.tools import Tool

        from .tools import REGISTRY

        names = available_tools(manager.deps)
        definitions = [
            Tool(spec.function, name=spec.name).tool_def
            for spec in REGISTRY
            if spec.name in names
        ]
        parameters = ModelRequestParameters(function_tools=definitions)
        from pydantic_ai.messages import ModelRequest, SystemPromptPart

        history = [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        chat_instructions(SimpleNamespace(deps=manager.deps))
                    )
                ]
            )
        ] + _project_messages(manager.rows)
        measured = await manager.counter.measure(history, parameters)
        return context_event(measured, request.context_tokens)

    async def compact(self, request, instructions=""):
        manager = await self._context_manager(request)
        changed = await manager.checkpoint_older(manual=True, instructions=instructions)
        return {
            "context": await self.context_info(request),
            "notice": (
                "Context compacted — summary checkpoint saved."
                if changed
                else "Nothing to compact yet."
            ),
        }

    async def _run(self, request: TurnRequest, turn: dict):
        tid = request.turn_id
        usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        deps = None
        new_messages: list = []
        human_text = request.text

        def combined_usage():
            children = deps.child_usage if deps else {}
            return {key: value + children.get(key, 0) for key, value in usage.items()}

        try:
            manager = await self._context_manager(request, running=True)
            rows, deps, model = manager.rows, manager.deps, manager.model
            human_text = next(
                (
                    "\n".join(
                        part.get("text", "")
                        for part in row["parts"]
                        if part["type"] == "text"
                    )
                    for row in reversed(rows)
                    if row["role"] == "user"
                ),
                request.text,
            )
            history = _project_messages(rows)
            deps.context_budget = max(
                128,
                request.context_tokens
                - min(request.max_tokens, request.context_tokens // 4)
                - 128,
            )
            if self.web_tools is not None:
                history = _dated_history(history)
                if request.mode == "regenerate":
                    # Preserve the original user timestamp, but make the current
                    # answer date explicit when regenerating a dated question.
                    for message in reversed(history):
                        for index, part in enumerate(message.parts):
                            if isinstance(part, UserPromptPart) and isinstance(
                                part.content, str
                            ):
                                message.parts[index] = replace(
                                    part,
                                    content=part.content
                                    + "\nAnswering now: "
                                    + turn_clock(),
                                )
                                break
                        else:
                            continue
                        break
            # Output byte quotas are independent of token context accounting.
            deps.read_budget_bytes = 16000
            deps.prepare_context = manager.prepare
            deps.token_counter = manager.counter
            self._presentations[tid] = ToolPresentation(deps.workspace)
            settings = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": request.thinking}
                },
            }
            final = ""
            for attempt in range(3):
                # An exhausted turn gets one retry if its summary is empty.
                # Ordinary empty answers retain their single recovery attempt.
                if attempt == 2 and not deps.result_budget_exhausted:
                    break
                selected = self.agent if attempt == 0 else self.synthesis_agent
                summary_failed = False
                try:
                    final, messages, consumed = await self._attempt(
                        selected,
                        model,
                        history,
                        deps,
                        settings,
                        attempt,
                    )
                except _AttemptError as error:
                    if not attempt or not deps.result_budget_exhausted:
                        raise
                    # A summary transport failure cannot discard completed work
                    # or hide the exhausted budget. Cancellation still propagates.
                    logger.error(
                        "Chat work summary failed: %s", type(error.error).__name__
                    )
                    final, messages, consumed = "", error.messages, error.usage
                    summary_failed = True
                new_messages.extend(messages)
                history.extend(messages)
                for key in usage:
                    usage[key] += getattr(consumed, key, 0)
                if final.strip() or summary_failed:
                    break
            if deps.result_budget_exhausted:
                notice = _localized(
                    human_text,
                    "\n\n本轮工具预算已用尽。可以说“继续”来继续未完成的工作。",
                    '\n\nThe tool budget for this turn is exhausted. Say "continue" to resume unfinished work.',
                )
                await deps.emit(
                    {"type": "text.delta", "part_id": "tool-budget", "text": notice}
                )
                if (
                    final.strip()
                    and new_messages
                    and isinstance(new_messages[-1], ModelResponse)
                ):
                    new_messages[-1] = replace(
                        new_messages[-1],
                        parts=[*new_messages[-1].parts, TextPart(notice)],
                    )
                    history[-1] = new_messages[-1]
                else:
                    message = ModelResponse(parts=[TextPart(notice)])
                    new_messages.append(message)
                    history.append(message)
            if not final.strip():
                fallback = _localized(
                    human_text,
                    (
                        "模型未能生成工作总结。已完成的操作和结果已保存。"
                        if deps.result_budget_exhausted
                        else "我无法根据现有信息确认完整答案。已执行的操作和结果已保存，请缩小问题范围后继续。"
                    ),
                    (
                        "The model could not produce a work summary. Completed actions and results are saved."
                        if deps.result_budget_exhausted
                        else "I couldn't verify an answer from the available evidence. Completed actions and results are saved; please try a narrower question."
                    ),
                )
                budget_recovery = (
                    deps.result_budget_exhausted
                    or deps.context_compacted
                    or any(
                        part.part_kind == "tool-return"
                        for message in new_messages
                        for part in message.parts
                    )
                )
                await deps.emit(
                    {"type": "text.delta", "part_id": "fallback", "text": fallback}
                )
                new_messages.append(ModelResponse(parts=[TextPart(fallback)]))
                await self.store.finish_turn(
                    tid,
                    "completed" if budget_recovery else "error",
                    messages=_durable_messages(new_messages),
                    usage=combined_usage(),
                    error_code=None if budget_recovery else "empty_final_answer",
                )
            else:
                await manager.report(history)
                await self._emit(tid, {"type": "usage", **combined_usage()})
                await self.store.finish_turn(
                    tid,
                    "completed",
                    messages=_durable_messages(new_messages),
                    usage=combined_usage(),
                )
        except asyncio.CancelledError:
            await self.delegation.cancel_turn(tid)
            # Agent/model context managers have already closed the HTTP stream.
            await self.store.finish_turn(tid, "cancelled", usage=combined_usage())
        except Exception as error:
            await self.delegation.cancel_turn(tid)
            if isinstance(error, _AttemptError):
                new_messages.extend(error.messages)
                for key in usage:
                    usage[key] += getattr(error.usage, key, 0)
                error = error.error
            # Exception messages may contain credentials, prompts, or reasoning.
            logger.error("Chat generation failed: %s", type(error).__name__)
            fallback = _localized(
                human_text,
                "我未能完成这次回答。已完成的操作和已有回答已保存，请继续或重试。",
                "I couldn't complete this answer. Completed actions and partial answers are saved; please continue or try again.",
            )
            await self._emit(
                tid,
                {
                    "type": "text.delta",
                    "part_id": "fallback",
                    "text": fallback,
                },
            )
            new_messages.append(ModelResponse(parts=[TextPart(fallback)]))
            await self.store.finish_turn(
                tid,
                "error",
                messages=_durable_messages(new_messages),
                usage=combined_usage(),
                error_code="generation_failed",
            )
        finally:
            await self.delegation.cancel_turn(tid)
            self.delegation.release_turn(tid)
            self.actions.release_turn(tid)
            self._signals.setdefault(tid, asyncio.Event()).set()

    async def _attempt(self, agent, model, history, deps, settings, attempt):
        history = list(history)
        initial = len(history)
        usage = deps.run_usage if deps.run_usage is not None else RunUsage()
        deferred = None
        continuation = 0
        truncations = 0
        human_text = latest_prompt(history)
        while True:
            progress = []
            previous_snapshot = deps.snapshot

            async def snapshot(
                items,
                consumed,
                progress=progress,
                previous_snapshot=previous_snapshot,
                segment_start=len(history),
            ):
                progress[:] = items[segment_start:]
                if previous_snapshot:
                    await previous_snapshot(items, consumed)

            deps.snapshot = snapshot
            try:
                final, messages, _ = await self._segment(
                    agent,
                    model,
                    history,
                    deps,
                    settings,
                    attempt,
                    usage,
                    deferred,
                    continuation,
                )
            except Exception as error:
                if deps.subagent_id:
                    # Helpers already persist snapshots and classify their own
                    # quota/error outcomes using the original exception type.
                    raise
                raise _AttemptError(
                    error,
                    _completed_messages(
                        history[initial:] + progress, history[:initial]
                    ),
                    usage,
                ) from None
            finally:
                deps.snapshot = previous_snapshot
            history.extend(messages)
            if final is _TRUNCATED_CALL:
                truncations += 1
                request_limit = (
                    deps.usage_limits or UsageLimits(request_limit=None)
                ).request_limit
                budget_exhausted = (
                    request_limit is not None and usage.requests >= request_limit
                ) or deps.result_budget_exhausted
                exhausted = truncations > 4 or budget_exhausted
                notice = _localized(
                    human_text,
                    (
                        "工具调用因输出长度限制被截断，该次调用未执行。请缩小操作范围后继续。"
                        if exhausted
                        else "工具调用因输出长度限制被截断，该次调用未执行。正在重试。"
                    ),
                    (
                        "The tool call was truncated by the output limit and was not executed. Please continue with a smaller operation."
                        if exhausted
                        else "The tool call was truncated by the output limit and was not executed. Retrying."
                    ),
                )
                await deps.emit(
                    {
                        "type": "text.delta",
                        "part_id": f"truncation:{attempt}:{continuation}",
                        "text": notice,
                    }
                )
                if messages and isinstance(history[-1], ModelResponse):
                    history[-1] = replace(
                        history[-1], parts=[*history[-1].parts, TextPart(notice)]
                    )
                else:
                    history.append(ModelResponse(parts=[TextPart(notice)]))
                if budget_exhausted:
                    deps.result_budget_exhausted = True
                    return "", history[initial:], usage
                if exhausted:
                    return notice, history[initial:], usage
                history.append(
                    ModelRequest(
                        parts=[
                            RuntimeNoticePart(
                                _localized(
                                    human_text,
                                    "上次工具调用被截断，未执行。请重新发出完整且更短的调用；如有必要，将内容拆成较小的操作。此前已完成的操作无需重复。",
                                    "Your last tool call was truncated and was not executed. Retry with complete, shorter arguments; split the work into smaller operations if necessary. Do not repeat previously completed operations.",
                                )
                            )
                        ]
                    )
                )
                deferred = None
                continuation += 1
                continue
            if not isinstance(final, DeferredToolRequests):
                return final, history[initial:], usage
            try:
                deferred = await self.actions.deferred_results(final, deps)
            except Exception as error:
                if deps.subagent_id:
                    raise
                raise _AttemptError(
                    error,
                    _completed_messages(history[initial:], history[:initial]),
                    usage,
                ) from None
            continuation += 1

    async def _segment(
        self,
        agent,
        model,
        history,
        deps,
        settings,
        attempt,
        usage,
        deferred,
        continuation=0,
    ):
        round_no = 0
        pending_text = ""
        segment_id = f"{attempt}:{continuation}"
        async with agent.iter(
            # Synthesis instructions belong in the system configuration. An
            # English synthetic user turn would override the real user's language.
            user_prompt=None,
            message_history=history,
            deferred_tool_results=deferred,
            usage=usage,
            model=model,
            deps=deps,
            model_settings=settings,
            usage_limits=(
                UsageLimits(request_limit=1, tool_calls_limit=0)
                if attempt and not deps.subagent_id
                else deps.usage_limits or UsageLimits(request_limit=None)
            ),
        ) as run:
            try:
                async for node in run:
                    if deps.snapshot:
                        await deps.snapshot(run.all_messages(), run.usage)
                    if Agent.is_model_request_node(node):
                        pending_text = ""
                        if attempt == 0 and (
                            deps.result_budget_exhausted
                            or deps.result_bytes >= 16000
                            or (deps.web is not None and deps.web_calls >= 4)
                            or (round_no and deps.read_budget_bytes <= 768)
                        ):
                            deps.result_budget_exhausted = True
                            new_messages = list(run.new_messages())
                            if not any(
                                message is node.request
                                for message in run.all_messages()
                            ):
                                new_messages.append(node.request)
                            return "", new_messages, run.usage
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                if event.event_kind == "part_start":
                                    kind = event.part.part_kind
                                    text = getattr(event.part, "content", "")
                                elif event.event_kind == "part_delta":
                                    kind = event.delta.part_delta_kind
                                    text = getattr(event.delta, "content_delta", "")
                                else:
                                    continue
                                if (
                                    kind in {"text", "thinking"}
                                    and text
                                    and attempt == 0
                                ):
                                    if kind == "text":
                                        pending_text += text
                                    await deps.emit(
                                        {
                                            "type": "text.delta"
                                            if kind == "text"
                                            else "reasoning.delta",
                                            "part_id": f"{segment_id}:{round_no}:{event.index}",
                                            "text": text,
                                        }
                                    )
                        response = stream.response
                        pending_text = ""
                        if deps.token_counter and self.model_override is None:
                            deps.token_counter.calibrate(response.usage.input_tokens)
                        round_no += 1
                        visible = "\n\n".join(
                            p.content for p in response.parts if p.part_kind == "text"
                        )
                        if attempt and re.search(
                            r"<(?:tool_call|function[=>]|invoke[ >])", visible, re.I
                        ):
                            # Reject tool-shaped summary text before length-stop
                            # display, and omit it from both durable and retry
                            # history. The bounded summary retry can then recover.
                            messages = list(run.new_messages())
                            if messages and isinstance(messages[-1], ModelResponse):
                                messages.pop()
                            return "", _completed_messages(messages, history), run.usage
                        if response.finish_reason == "length":
                            if response.tool_calls:
                                # Gate the entire response before dispatch, even valid
                                # JSON: a provider may have cut off further arguments/calls.
                                return (
                                    _TRUNCATED_CALL,
                                    _completed_messages(run.new_messages(), history),
                                    run.usage,
                                )
                            human_text = latest_prompt(history)
                            notice = _localized(
                                human_text,
                                "\n\n回答因输出长度限制被截断，请说“继续”。",
                                '\n\nThis answer was truncated by the output limit. Say "continue" to resume.',
                            )
                            await deps.emit(
                                {
                                    "type": "text.delta",
                                    "part_id": f"{segment_id}:{round_no}:truncated",
                                    "text": (visible if attempt else "") + notice,
                                }
                            )
                            messages = list(run.new_messages())
                            # Keep the visible fragment and notice in the same final
                            # response so final context accounting includes both.
                            messages[-1] = replace(
                                messages[-1],
                                parts=[*messages[-1].parts, TextPart(notice)],
                            )
                            return visible + notice, messages, run.usage
                        if (
                            attempt or deps.result_budget_exhausted
                        ) and response.tool_calls:
                            # A provider can ignore the disabled tool schema. Do not
                            # dispatch or persist its unexecuted synthesis calls.
                            return (
                                "",
                                _completed_messages(run.new_messages(), history),
                                run.usage,
                            )
                        if not response.tool_calls:
                            if not visible.strip():
                                # Avoid the harness's generic empty-output retry. The only
                                # recovery is our single tools-disabled synthesis attempt.
                                return "", run.new_messages(), run.usage
                            if attempt:
                                await deps.emit(
                                    {
                                        "type": "text.delta",
                                        "part_id": f"{segment_id}:{round_no}:answer",
                                        "text": visible,
                                    }
                                )
                result = run.result
                if result is None:
                    return "", run.new_messages(), run.usage
                return result.output, result.new_messages(), result.usage
            except IncompleteToolCall:
                pending_text = ""
                return (
                    _TRUNCATED_CALL,
                    _completed_messages(run.new_messages(), history),
                    run.usage,
                )
            except (
                UsageLimitExceeded,
                ModelHTTPError,
                ContextSynthesisUnavailableError,
                _SynthesisRequiredError,
            ) as error:
                if deps.subagent_id and isinstance(error, UsageLimitExceeded):
                    raise
                if isinstance(error, ModelHTTPError) and not (
                    error.status_code == 400
                    and any(
                        word in str(error).lower()
                        for word in ("context", "max_tokens", "prompt too long")
                    )
                ):
                    raise
                deps.result_budget_exhausted = True
                return (
                    "",
                    _completed_messages(run.new_messages(), history),
                    run.usage,
                )
            finally:
                if deps.snapshot:
                    messages = list(run.all_messages())
                    if pending_text:
                        messages.append(ModelResponse(parts=[TextPart(pending_text)]))
                    await deps.snapshot(messages, run.usage)


def _completed_messages(messages, prior_messages=()):
    """Retain only complete tool pairs when a generation/tool quota ends a segment."""
    calls = {
        p.tool_call_id
        for m in [*prior_messages, *messages]
        for p in m.parts
        if p.part_kind == "tool-call"
    }
    returns = {
        p.tool_call_id
        for m in messages
        for p in m.parts
        if p.part_kind == "tool-return"
    }
    paired = calls & returns
    result = []
    for message in messages:
        parts = [
            p
            for p in message.parts
            if p.part_kind not in {"tool-call", "tool-return"}
            or p.tool_call_id in paired
        ]
        if parts:
            result.append(replace(message, parts=parts))
    return result


def _durable_messages(messages) -> list[dict]:
    """Whitelist domain parts; reasoning and provider metadata never reach SQLite."""
    rows = []
    for message in messages:
        parts = []
        for part in message.parts:
            if part.part_kind == "text":
                parts.append({"type": "text", "text": part.content})
            elif part.part_kind == "tool-call":
                parts.append(
                    {
                        "type": "tool_call",
                        "call_id": part.tool_call_id,
                        "name": part.tool_name,
                        "args": part.args_as_dict(),
                    }
                )
            elif part.part_kind == "tool-return":
                parts.append(
                    {
                        "type": "tool_result",
                        "call_id": part.tool_call_id,
                        "name": part.tool_name,
                        "result": part.model_response_str(),
                    }
                )
        if parts:
            rows.append(
                {
                    "role": (
                        "assistant" if isinstance(message, ModelResponse) else "tool"
                    ),
                    "parts": parts,
                }
            )
    return rows
