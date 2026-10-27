# SPDX-License-Identifier: Apache-2.0
"""Local action adapters, fenced by the operation ledger before every effect."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import os
import shutil
import signal
import stat
import sys
import time
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic_ai import ApprovalRequired, RunContext
from pydantic_ai.tools import DeferredToolResults, ToolDenied

from ..path_policy import resolve_path as file_path
from ..path_policy import workspace_root
from ..store import Conflict
from ..store.actions import TERMINAL_STATES
from . import TurnDeps, emit_result
from .action_policy import (
    Policy,
    actions_enabled,
    configured_policy,
    execution_policy,
    file_policy,
)

MAX_FILE_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 8192
APPROVAL_TIMEOUT = 120


def digest(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def parent_fd(path, *, create=False):
    """Open each canonical parent without following a newly substituted symlink."""
    path = Path(path)
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def read_image(path):
    try:
        with parent_fd(path) as directory:
            fd = os.open(
                Path(path).name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
                    raise ValueError(
                        "Only regular UTF-8 files up to 1 MiB can be edited"
                    )
                data = stream.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError("File grew beyond the edit limit")
                if b"\x00" in data:
                    raise ValueError("Binary files cannot be edited")
                data.decode("utf-8")
                return data, stat.S_IMODE(info.st_mode)
    except FileNotFoundError:
        return None, None


def replace_image(path, data, mode, expected):
    """Compare pre-image, then atomically replace via the opened directory."""
    with parent_fd(path, create=data is not None) as directory:
        name = Path(path).name
        try:
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
        except FileNotFoundError:
            current = None
        else:
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Target is no longer a regular file")
                current = stream.read(MAX_FILE_BYTES + 1)
        if current != expected:
            raise ValueError("File changed since inspection; refusing to overwrite it")
        if data is None:
            os.unlink(name, dir_fd=directory)
        else:
            temporary = ".rapid-edit-" + uuid.uuid4().hex
            try:
                fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    mode or 0o644,
                    dir_fd=directory,
                )
                with os.fdopen(fd, "wb") as stream:
                    if mode is not None:
                        os.fchmod(stream.fileno(), mode)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
        os.fsync(directory)


class BoundedOutput:
    def __init__(self, limit=MAX_OUTPUT_BYTES):
        self.limit = limit
        self.head = b""
        self.tail = b""
        self.total = 0

    def add(self, chunk):
        self.total += len(chunk)
        head_room = self.limit // 2 - len(self.head)
        self.head += chunk[:head_room]
        self.tail = (self.tail + chunk[head_room:])[-(self.limit // 2) :]

    def text(self):
        omitted = self.total - len(self.head) - len(self.tail)
        marker = f"\n[… {omitted} bytes omitted …]\n".encode() if omitted > 0 else b""
        return (self.head + marker + self.tail).decode("utf-8", errors="replace")


async def kill_group(process):
    # Kill the owned group even if its shell leader has already exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            await asyncio.sleep(0.1)
    await process.wait()


async def run_process(argv, workspace, timeout, emit, call_id):
    output = BoundedOutput()
    started = time.monotonic()
    # No stdin, no interactive shell, and a session/process group owned by this call.
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            word in k.upper()
            for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL")
        )
    }
    env.update(PYTHONUNBUFFERED="1", GIT_TERMINAL_PROMPT="0", PAGER="cat")
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    )
    try:
        process = await asyncio.shield(spawning)
    except asyncio.CancelledError:
        process = await spawning
        await kill_group(process)
        raise
    streamed = 0
    timed_out = False
    try:
        async with asyncio.timeout(timeout):
            while chunk := await process.stdout.read(2048):
                output.add(chunk)
                if streamed < MAX_OUTPUT_BYTES // 2:
                    visible = chunk[: MAX_OUTPUT_BYTES // 2 - streamed]
                    streamed += len(visible)
                    await emit(
                        {
                            "type": "tool.output",
                            "call_id": call_id,
                            "text": visible.decode("utf-8", errors="replace"),
                        }
                    )
            await process.wait()
    except TimeoutError:
        timed_out = True
    finally:
        await kill_group(process)
    if output.tail:
        omitted = output.total - len(output.head) - len(output.tail)
        tail = (
            f"\n[… {omitted} bytes omitted …]\n" if omitted > 0 else ""
        ) + output.tail.decode("utf-8", errors="replace")
        await emit({"type": "tool.output", "call_id": call_id, "text": tail})
    return {
        "exit_code": process.returncode,
        "duration": round(time.monotonic() - started, 3),
        "output": output.text(),
        "output_bytes": output.total,
        "timed_out": timed_out,
    }


@dataclass
class Prepared:
    name: str
    args: dict
    summary: str
    preview: str
    policy: object
    path: Path | None = None
    before: bytes | None = None
    mode: int | None = None
    after: bytes | None = None


class ActionTools:
    """One runtime's volatile waiters; durable decisions always come from SQLite."""

    def __init__(self, store, *, approval_timeout=APPROVAL_TIMEOUT):
        self.store = store
        self.approval_timeout = approval_timeout
        self.prepared = {}
        self.signals = {}
        self.undo_tasks = {}
        self.locks = weakref.WeakValueDictionary()

    def lock(self, conversation_id):
        return self.locks.setdefault(conversation_id, asyncio.Lock())

    async def respond(self, op_id, revision, decision, scope="once", reason=""):
        op = await self.store.decide_operation(
            op_id, revision, decision, scope, reason=reason
        )
        self.signals.setdefault(op_id, asyncio.Event()).set()
        return op

    async def wait(self, op_id):
        signal = self.signals.setdefault(op_id, asyncio.Event())
        while True:
            signal.clear()
            op = await self.store.get_operation(op_id)
            if op["state"] != "intent":
                return op
            remaining = op["expires_at"] - time.time()
            if remaining <= 0:
                try:
                    return await self.store.decide_operation(
                        op_id, op["revision"], "deny", who="timeout"
                    )
                except Conflict:
                    continue
            try:
                await asyncio.wait_for(signal.wait(), remaining)
            except TimeoutError:
                pass

    async def deferred_results(self, requests, deps):
        results = DeferredToolResults()
        for call in requests.approvals:
            op_id = requests.metadata[call.tool_call_id]["op_id"]
            op = await self.wait(op_id)
            allowed = op["state"] == "approved"
            denial = "Action denied or approval expired."
            if op["decision_reason"]:
                denial += " Reason: " + op["decision_reason"]
            results.approvals[call.tool_call_id] = (
                True if allowed else ToolDenied(denial + " Do not retry it.")
            )
            if not allowed:
                await emit_result(deps, deps.call_id(call.tool_call_id), denial)
        return results

    def release_turn(self, turn_id):
        for key in list(self.prepared):
            if key[0] == turn_id:
                _, op_id = self.prepared.pop(key)
                self.signals.pop(op_id, None)

    def release_child(self, turn_id, child_id):
        """Release this child's waiters without disturbing sibling operations."""
        for key in list(self.prepared):
            if key[0] == turn_id and str(key[1]).startswith(child_id + ":"):
                _, op_id = self.prepared.pop(key)
                self.signals.pop(op_id, None)

    def prepare(self, name, args, deps):
        workspace = deps.workspace
        workspace_root(workspace)
        if name in {"write_file", "patch"}:
            _, path = file_path(workspace, args["path"])
            if path.is_relative_to(self.store.directory.resolve()):
                raise ValueError("The chat profile cannot be modified by action tools")
            policy = file_policy(workspace, args["path"])
            before, mode = read_image(path)
            old = (before or b"").decode("utf-8")
            if name == "patch":
                if (
                    before is None
                    or not args["old_text"]
                    or old.count(args["old_text"]) != 1
                ):
                    raise ValueError(
                        "patch requires one exact, nonempty old_text match in an existing file"
                    )
                new = old.replace(args["old_text"], args["new_text"], 1)
            else:
                new = args["content"]
            after = new.encode("utf-8")
            if len(after) > MAX_FILE_BYTES or b"\x00" in after:
                raise ValueError("Only UTF-8 text up to 1 MiB can be edited")
            diff = list(
                difflib.unified_diff(
                    old.splitlines(),
                    new.splitlines(),
                    fromfile=str(path),
                    tofile=str(path),
                    lineterm="",
                )
            )
            added = sum(
                line.startswith("+") and not line.startswith("+++") for line in diff
            )
            removed = sum(
                line.startswith("-") and not line.startswith("---") for line in diff
            )
            preview = "\n".join(diff)
            # Never include existing credential contents in an approval preview.
            if policy.risk == "secret":
                preview = f"{path}\n[Secrets-like file contents hidden; {len(after)} new bytes]"
            return Prepared(
                name,
                args,
                f"Edited {path} (+{added} −{removed})",
                preview,
                policy,
                path,
                before,
                mode,
                after,
            )
        timeout = args["timeout"]
        if not 0 < timeout <= 600:
            raise ValueError(
                "timeout must be greater than zero and at most 600 seconds"
            )
        code = args["command"] if name == "terminal" else args["code"]
        if not code.strip() or len(code.encode()) > 65536:
            raise ValueError("Use nonempty code up to 64 KiB")
        policy = execution_policy(code, workspace, python=name == "execute_code")
        summary = code if name == "terminal" else "python3 code"
        return Prepared(name, args, summary, code, policy)

    async def call(self, ctx, name, args):
        deps = ctx.deps
        if not actions_enabled() or not deps.action_tools or not deps.workspace:
            return "Action tools are disabled or this conversation has no workspace."
        async with self.lock(deps.conversation_id):
            key = (deps.turn_id, ctx.tool_call_id)
            try:
                if key in self.prepared:
                    prepared, op_id = self.prepared[key]
                    if prepared.name != name or prepared.args != args:
                        raise Conflict("deferred arguments changed")
                    op = await self.store.get_operation(op_id)
                else:
                    try:
                        prepared = self.prepare(name, args, deps)
                    except (ValueError, OSError, UnicodeError):
                        prepared = Prepared(
                            name,
                            args,
                            "Action not dispatched",
                            "",
                            Policy(
                                True,
                                "invalid",
                                "Action arguments could not be safely prepared.",
                                False,
                            ),
                        )
                    invalid = prepared.policy.risk == "invalid"
                    policy = (
                        prepared.policy
                        if invalid
                        else configured_policy(prepared.policy)
                    )
                    if deps.approvals_always and not invalid:
                        policy = Policy(
                            True,
                            policy.risk,
                            "All side-effect calls require approval. " + policy.reason,
                            False,
                        )
                    op, created = await self.store.create_operation(
                        deps.turn_id,
                        ctx.tool_call_id,
                        name,
                        1,
                        {**args, "path": str(prepared.path)} if prepared.path else args,
                        risk=policy.risk,
                        reason=policy.reason,
                        summary=prepared.summary,
                        preview=prepared.preview,
                        prompt=policy.prompt,
                        allow_conversation=policy.allow_conversation,
                        timeout=self.approval_timeout,
                    )
                    op_id = op["id"]
                    if invalid:
                        if op["state"] == "intent":
                            await self.store.transition_operation(
                                op_id,
                                op["revision"],
                                "cancelled_before_effect",
                                {
                                    "summary": "Invalid arguments or file changed; no effect dispatched."
                                },
                            )
                    else:
                        self.prepared[key] = (prepared, op_id)
                    if created:
                        await deps.emit(
                            {
                                "type": "tool.started",
                                "call_id": ctx.tool_call_id,
                                "name": name,
                                "args": {"op_id": op_id},
                            }
                        )
                        if op["state"] == "intent" and not invalid:
                            await deps.emit(
                                {
                                    "type": "permission.request",
                                    "op_id": op_id,
                                    "revision": op["revision"],
                                    "tool": name,
                                    "summary": op["summary"],
                                    "preview": op["preview"],
                                    "risk": op["risk"],
                                    "reason": op["reason"],
                                    "options": (
                                        ["once", "conversation", "deny"]
                                        if op["allow_conversation"]
                                        else ["once", "deny"]
                                    ),
                                    "expires_at": op["expires_at"],
                                }
                            )
                    if invalid:
                        return await emit_result(
                            deps,
                            ctx.tool_call_id,
                            "Action not dispatched: check path, UTF-8 size, timeout, and exact unique patch match.",
                        )
                if op["state"] == "intent":
                    raise ApprovalRequired(metadata={"op_id": op_id})
                if op["state"] in TERMINAL_STATES or op["state"] == "dispatched":
                    return (op["receipt"] or {}).get(
                        "result", f"Operation {op['state']}; it will not be retried."
                    )
                if str(Path(deps.workspace).resolve()) != deps.workspace:
                    await self.store.transition_operation(
                        op_id,
                        op["revision"],
                        "cancelled_before_effect",
                        {"summary": "Workspace changed before dispatch"},
                    )
                    result = "Workspace changed before dispatch; no action was run."
                    return await emit_result(deps, ctx.tool_call_id, result, "error")
                # Recheck a file after human approval, without silently changing its preview.
                if prepared.path and (
                    file_path(deps.workspace, args["path"])[1] != prepared.path
                    or read_image(prepared.path)[0] != prepared.before
                ):
                    await self.store.transition_operation(
                        op_id,
                        op["revision"],
                        "cancelled_before_effect",
                        {"summary": "File changed while awaiting approval"},
                    )
                    result = (
                        "File changed while awaiting approval; no edit was applied."
                    )
                    return await emit_result(deps, ctx.tool_call_id, result, "error")
                checkpoint = (
                    (
                        str(prepared.path),
                        prepared.before,
                        prepared.mode,
                        digest(prepared.after),
                    )
                    if prepared.path
                    else None
                )
                op = await self.store.dispatch_operation(
                    op_id, op["revision"], 1, checkpoint
                )
                try:
                    if prepared.path:
                        replace_image(
                            prepared.path,
                            prepared.after,
                            prepared.mode,
                            prepared.before,
                        )
                        receipt = {
                            "summary": prepared.summary,
                            "result": prepared.summary,
                            "path": str(prepared.path),
                        }
                        state = "succeeded"
                    else:
                        argv = (
                            ["/bin/sh", "-c", args["command"]]
                            if name == "terminal"
                            else [
                                shutil.which("python3") or sys.executable,
                                "-u",
                                "-c",
                                args["code"],
                            ]
                        )
                        receipt = await run_process(
                            argv,
                            deps.workspace,
                            args["timeout"],
                            deps.emit,
                            ctx.tool_call_id,
                        )
                        state = (
                            "outcome_unknown"
                            if receipt["timed_out"]
                            else (
                                "succeeded"
                                if receipt["exit_code"] == 0
                                else "failed_known"
                            )
                        )
                        command_label = " ".join(prepared.summary.splitlines())[:240]
                        summary = f"Ran `{command_label}` (exit {receipt['exit_code']}, {receipt['duration']:.1f}s)"
                        if receipt["timed_out"]:
                            summary += " — timed out; effects unknown, do not retry automatically."
                        receipt.update(
                            summary=summary, result=summary + "\n" + receipt["output"]
                        )
                    await self.store.transition_operation(
                        op_id, op["revision"], state, receipt
                    )
                except Exception:
                    receipt = {
                        "summary": "Action interrupted; effects unknown. Do not retry automatically.",
                        "result": "Action interrupted; effects unknown. Do not retry automatically.",
                    }
                    await self.store.transition_operation(
                        op_id, op["revision"], "outcome_unknown", receipt
                    )
                    state = "outcome_unknown"
                return await emit_result(
                    deps,
                    ctx.tool_call_id,
                    receipt["result"],
                    "success" if state == "succeeded" else "error",
                )
            except (ValueError, OSError, UnicodeError) as error:
                # Validation fails before dispatch. Do not expose arbitrary exception text.
                result = (
                    str(error)
                    if type(error) is ValueError
                    else "Action could not inspect the requested file or command."
                )
                return await emit_result(deps, ctx.tool_call_id, result, "error")

    async def close(self):
        await asyncio.gather(*self.undo_tasks.values(), return_exceptions=True)

    async def undo(self, conversation_id, operation_id):
        # A disconnected HTTP caller cannot abandon a committed undo dispatch.
        key = (conversation_id, operation_id)
        task = self.undo_tasks.get(key)
        if task is None:
            task = asyncio.create_task(self._undo(conversation_id, operation_id))
            self.undo_tasks[key] = task

            def finished(done):
                self.undo_tasks.pop(key, None)
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def _undo(self, conversation_id, operation_id):
        async with self.lock(conversation_id):
            op, checkpoint = await self.store.begin_undo(conversation_id, operation_id)
            if checkpoint is None:
                return op
            try:
                path = Path(checkpoint["path"])
                current, _ = read_image(path)
                if current is None or digest(current) != checkpoint["post_hash"]:
                    result = (
                        "File changed after the edit; undo refused to overwrite it."
                    )
                    state = "failed_known"
                else:
                    replace_image(
                        path, checkpoint["pre_image"], checkpoint["pre_mode"], current
                    )
                    result, state = f"Restored {path}", "succeeded"
                return await self.store.transition_operation(
                    op["id"],
                    op["revision"],
                    state,
                    {"summary": result, "result": result},
                )
            except BaseException:
                await self.store.transition_operation(
                    op["id"],
                    op["revision"],
                    "outcome_unknown",
                    {
                        "summary": "Undo interrupted; inspect the file before another change."
                    },
                )
                raise


# Path and reader-prefix guidance adapted from MIT-licensed tool descriptions;
# see /NOTICE and ../LICENSE-PROMPTS. File content itself is never rewritten.
async def write_file(ctx: RunContext[TurnDeps], path: str, content: str) -> str:
    """Create or replace a UTF-8 file; records an undo checkpoint. Path may be absolute, relative to the workspace, or ~/path; it must resolve inside the workspace. Write actual file content without the LINE_NUM| display prefixes added by read_file."""
    return cast(
        str,
        await ctx.deps.actions.call(
            ctx, "write_file", {"path": path, "content": content}
        ),
    )


async def patch(
    ctx: RunContext[TurnDeps], path: str, old_text: str, new_text: str
) -> str:
    """Replace one exact, unique old_text match in a UTF-8 file; records an undo checkpoint. Path may be absolute, relative to the workspace, or ~/path; it must resolve inside the workspace. Preserve indentation after read_file's LINE_NUM| prefix; never include that display prefix in old_text or new_text."""
    return cast(
        str,
        await ctx.deps.actions.call(
            ctx, "patch", {"path": path, "old_text": old_text, "new_text": new_text}
        ),
    )


async def terminal(
    ctx: RunContext[TurnDeps], command: str, timeout: float = 120
) -> str:
    """Run a non-interactive shell command in the workspace. Timeout is at most 600 seconds."""
    return cast(
        str,
        await ctx.deps.actions.call(
            ctx, "terminal", {"command": command, "timeout": timeout}
        ),
    )


async def execute_code(
    ctx: RunContext[TurnDeps],
    language: Literal["python"],
    code: str,
    timeout: float = 120,
) -> str:
    """Run Python using python3 on PATH in the workspace; bounded output and timeout (max 600s)."""
    return cast(
        str,
        await ctx.deps.actions.call(
            ctx,
            "execute_code",
            {"language": language, "code": code, "timeout": timeout},
        ),
    )
