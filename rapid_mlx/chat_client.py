# SPDX-License-Identifier: Apache-2.0
"""Interactive client for the authenticated, persistent chat owner."""

from __future__ import annotations

import codecs
import html
import json
import os
import shlex
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from rapid_mlx.chat_harness_render import HarnessRenderer, safe_markdown, source_lines
from rapid_mlx.chat_input import ChatInput
from rapid_mlx.chat_render import (
    _console,
    render_markdown,
    supports_rich_output,
    terminal_safe_text,
)
from rapid_mlx.chat_runtime.loopback_transport import LoopbackTransport
from rapid_mlx.chat_state import ReplayRequiredError, TurnState, sse_events

PREFIX = "/v1/rapid/chat"
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class ChatCommand:
    name: str
    aliases: tuple[str, ...]
    argument_hint: str
    description: str


COMMANDS = (
    ChatCommand("/new", ("/reset", "/clear"), "", "Start a conversation"),
    ChatCommand("/context", (), "", "Show context token usage"),
    ChatCommand(
        "/compact", (), "[instructions]", "Summarize this branch to free context"
    ),
    ChatCommand("/history", (), "[number|id]", "List conversations or resume one"),
    ChatCommand("/edit", (), "[text]", "Edit the last user message on a new branch"),
    ChatCommand("/regen", (), "", "Generate a sibling answer"),
    ChatCommand("/branch", (), "[number|id]", "List branches or switch"),
    ChatCommand("/think", (), "on|off", "Turn thinking on or off"),
    ChatCommand(
        "/show-thinking", (), "", "Expand/collapse thinking (this session only)"
    ),
    ChatCommand(
        "/attach", (), "PATH", "Attach text, PDF, or Office documents (up to 20 MiB)"
    ),
    ChatCommand("/undo", (), "", "Restore the last file edit in this conversation"),
    ChatCommand("/export", ("/save",), "[PATH]", "Export this branch as Markdown"),
    ChatCommand("/help", ("/?",), "", "Show commands"),
    ChatCommand("/quit", ("/exit", "/bye"), "", "Exit (also Ctrl-D)"),
)
COMMAND_BY_NAME = {
    name: command for command in COMMANDS for name in (command.name, *command.aliases)
}
HELP = (
    "Commands\n"
    + "".join(
        f"  {command.name + (' ' + command.argument_hint if command.argument_hint else ''):<25} "
        f"{command.description}\n"
        for command in COMMANDS
    )
    + """Ctrl-C stops a response and returns to the prompt. At an empty prompt,
press Ctrl-C twice to exit. Conversations are saved automatically.
"""
)


def history_time(value: str) -> str:
    """Render stored UTC timestamps in the terminal user's local timezone."""
    return datetime.fromisoformat(value).astimezone().strftime("%m-%d %H:%M")


def select_item(items: list[dict], choice: str, error: str) -> dict:
    """Select a numbered row or an unambiguous ID prefix."""
    if choice.isdigit() and 1 <= int(choice) <= len(items):
        return items[int(choice) - 1]
    matches = [item for item in items if item["id"].startswith(choice)]
    if len(matches) != 1:
        raise ValueError(error)
    return matches[0]


def print_status(text: str, style: str = "dim") -> None:
    text = terminal_safe_text(text)
    if supports_rich_output(sys.stdout):
        _console(sys.stdout).print(text, style=style, markup=False, highlight=False)
    else:
        print(text)


class ChatClient:
    def __init__(self, args: Any, inference_url: str, model: str):
        self.args = args
        self.workspace = str(Path.cwd().resolve())
        self.profile = getattr(args, "profile", "default")
        self.inference_url = inference_url
        self.model = model
        self.thinking = bool(getattr(args, "think", False))
        self.expanded = False
        self.client_id = uuid.uuid4().hex
        self.conversation_id: str | None = None
        self.branch_id: str | None = None
        self.snapshot: dict = {}
        self.attachments: list[str] = []
        self.last_state: TurnState | None = None
        self._last_context_tokens: int | None = None
        self.stop_armed = False
        self.input = ChatInput(COMMANDS)
        self.prompt_surface = False
        self.endpoint = ""
        self.headers: dict[str, str] = {}
        self.client = httpx.Client(
            timeout=httpx.Timeout(getattr(args, "response_timeout", 600), connect=5),
            trust_env=False,
            transport=LoopbackTransport(trust_env=False),
        )
        self.stopping = threading.Event()
        self.lease_thread: threading.Thread | None = None

    def connect(self):
        from rapid_mlx.chat_runtime.owner import discover_or_start

        discovered = discover_or_start(self.profile)
        self.endpoint = discovered["endpoint"].rstrip("/")
        self.headers = {"Authorization": f"Bearer {discovered['token']}"}
        capabilities = self.request("GET", "/capabilities", retry=False)
        if capabilities.get("wire_version") != 1:
            raise ValueError("Chat owner requires a client upgrade")
        self.prompt_surface = capabilities.get("capabilities", {}).get(
            "prompt_surface", False
        )
        self.request(
            "POST", "/clients", json={"client_id": self.client_id}, retry=False
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        retry: bool = True,
        idempotent: bool = False,
        **kwargs,
    ):
        for attempt in range(2):
            try:
                response = self.client.request(
                    method,
                    self.endpoint + PREFIX + path,
                    headers=self.headers,
                    **kwargs,
                )
                if retry and attempt == 0 and response.status_code == 401:
                    self.connect()
                    continue
                response.raise_for_status()
                return response.json() if response.content else {}
            except httpx.TransportError:
                if not retry or attempt:
                    raise
                if method.upper() not in {"GET", "DELETE"} and not idempotent:
                    raise
                self.connect()
        raise RuntimeError(  # pragma: no cover - both attempts return or raise
            "Could not reconnect to chat owner"
        )

    def heartbeat(self):
        while not self.stopping.wait(30):
            try:
                self.request(
                    "POST",
                    "/clients",
                    json={"client_id": self.client_id},
                    idempotent=True,
                )
            except (httpx.HTTPError, OSError, ValueError, RuntimeError):
                # Interactive operations report errors and reconnect as needed.
                pass

    def close(self):
        self.stopping.set()
        self.input.close()
        if self.lease_thread:
            self.lease_thread.join(timeout=6)
        try:
            self.request("DELETE", f"/clients/{self.client_id}", retry=False)
        except (httpx.HTTPError, OSError):
            pass
        self.client.close()

    def refresh(self):
        self.snapshot = self.request(
            "GET",
            f"/conversations/{self.conversation_id}",
            params={"branch_id": self.branch_id} if self.branch_id else {},
        )
        if not self.branch_id:
            self.branch_id = (
                self.snapshot.get("branch_id") or self.snapshot["branches"][0]["id"]
            )
        return self.snapshot

    def branch(self):
        return next(
            branch
            for branch in self.snapshot["branches"]
            if branch["id"] == self.branch_id
        )

    def new(self):
        self.input.context_percent = 0
        result = self.request(
            "POST", "/conversations", json={"workspace": self.workspace}
        )
        self.conversation_id = result["id"]
        self.branch_id = result.get("branch_id") or result["branches"][0]["id"]
        self.attachments.clear()
        self.refresh()
        self.show_chat_status()

    def show_chat_status(self, title: str | None = None):
        label = (
            f"Resumed: {terminal_safe_text(title)}" if title is not None else "New chat"
        )
        print_status(f"{label} · Type / for commands")

    def conversations(self):
        result = self.request("GET", "/conversations")
        return result if isinstance(result, list) else result["conversations"]

    def select_conversation(self, choice: str, *, display: bool = True):
        items = self.conversations()
        if not items:
            print("No saved conversations yet.")
            return False
        if choice == "last":
            selected = items[0]
        else:
            selected = select_item(
                items, choice, "Choose a conversation number or unique ID from /history"
            )
        self.conversation_id = selected["id"]
        self.input.context_percent = 0
        self.branch_id = None
        self.attachments.clear()
        self.refresh()
        self.show_chat_status(
            str(
                selected.get("display_title") or selected.get("title") or "Conversation"
            )
        )
        if display:
            self.show_history()
            for operation in self.snapshot.get("uncertain_operations", []):
                print(
                    f"Outcome unknown: {terminal_safe_text(operation['tool'])} ({operation['id']}). Inspect effects before running again."
                )
        return True

    @staticmethod
    def message_text(message: dict) -> str:
        return "\n".join(
            part.get("text", "")
            for part in message.get("parts", [])
            if part.get("type", part.get("kind")) == "text"
        )

    def show_history(self):
        for message in self.snapshot.get("messages", []):
            text = self.message_text(message)
            if text:
                print(f"\n{message['role'].capitalize()}:")
                render_markdown(safe_markdown(text))
                for line in source_lines(message.get("sources", [])):
                    print(line)
                print()

    def permission(self, data, renderer):
        current = self.request("GET", f"/operations/{data['op_id']}")
        if current["state"] != "intent" or current["revision"] != data["revision"]:
            return
        renderer.pause()
        try:
            approval_data = dict(data)
            if child_id := data.get("subagent_id"):
                child = renderer.state.subagents.get(child_id, {})
                title = terminal_safe_text(child.get("title") or child_id)[:120]
                approval_data["reason"] = data["reason"] + f" (helper: {title})"
            key = self.input.approval(
                allow_conversation="conversation" in data["options"],
                timeout=max(0, data["expires_at"] - time.time()),
                data=approval_data,
            )
            decision = "allow" if key in {"y", "a"} else "deny"
            try:
                self.request(
                    "POST",
                    "/permission.response",
                    retry=False,
                    json={
                        "op_id": data["op_id"],
                        "revision": data["revision"],
                        "decision": decision,
                        "scope": "conversation" if key == "a" else "once",
                        "reason": self.input.approval_reason,
                    },
                )
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 409:
                    raise
                print("Approval expired or was already resolved.")
        finally:
            renderer.resume()

    def export_subruns(self) -> list[str]:
        """Export hidden owner transcripts without changing visible branch history."""
        pieces = []
        turn_ids = dict.fromkeys(
            message["turn_id"]
            for message in self.snapshot.get("messages", [])
            if message.get("turn_id")
        )
        for turn_id in turn_ids:
            for child in self.request("GET", f"/turns/{turn_id}/subruns"):
                run = self.request("GET", f"/subruns/{child['id']}")
                label = html.escape(
                    f"Helper: {run['title']} — {run['status']} "
                    f"({run['tool_count']} tools)"
                )
                transcript = {
                    key: run.get(key)
                    for key in (
                        "id",
                        "turn_id",
                        "call_id",
                        "task",
                        "context",
                        "summary",
                        "tools",
                        "usage",
                        "messages",
                        "events",
                    )
                }
                payload = html.escape(
                    json.dumps(transcript, ensure_ascii=False, indent=2)
                )
                sources = "\n\n".join(
                    source_lines(run.get("sources", []), markdown=True)
                )
                pieces.append(
                    f"<details>\n<summary>{label}</summary>\n\n"
                    f"<pre>{payload}</pre>\n\n{sources}\n\n</details>\n"
                )
        return pieces

    def context_tokens(self) -> int:
        """Bound chat history by checkpoint and active serving context limits."""
        from rapid_mlx.chat_runtime.context_limits import context_limit
        from rapid_mlx.chat_runtime.model_client import inference_url

        requested = getattr(self.args, "context_length", None)
        fallback = context_limit(requested, previous=self._last_context_tokens)
        key = os.environ.get("RAPID_MLX_API_KEY")
        try:
            response = self.client.get(
                inference_url(self.inference_url) + "/models",
                headers={"Authorization": f"Bearer {key}"} if key else {},
                timeout=2,
                follow_redirects=False,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return fallback
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return fallback
        entries = [entry for entry in entries if isinstance(entry, dict)]
        selected = next(
            (entry for entry in entries if entry.get("id") == self.model), None
        )
        if selected is None and len(entries) == 1:
            selected = entries[0]
        if selected is None:
            return fallback
        budget = context_limit(
            requested, model=selected, previous=self._last_context_tokens
        )
        self._last_context_tokens = budget
        return budget

    def context_request(self, text: str = "", *, mode: str = "append") -> dict:
        """Send model settings to the owner for both turns and context operations."""
        self.refresh()
        body = {
            "v": 1,
            "turn_id": uuid.uuid4().hex,
            "conversation_id": self.conversation_id,
            "workspace": self.snapshot.get("workspace") or self.workspace,
            "branch_id": self.branch_id,
            "expected_revision": self.branch()["revision"],
            "text": text,
            "attachment_ids": list(self.attachments) if mode == "append" else [],
            "inference_url": self.inference_url,
            "model": self.model,
            "thinking": self.thinking,
            "temperature": getattr(self.args, "temperature", 0.7),
            "max_tokens": getattr(self.args, "max_tokens", None) or 32768,
            "context_tokens": self.context_tokens(),
            "mode": mode,
            "action_tools": getattr(self.args, "action_tools", True)
            and os.environ.get("RAPID_MLX_CHAT_ACTION_TOOLS", "1").lower()
            not in {"0", "false", "off"},
            "approvals": getattr(self.args, "approvals", None)
            or (
                "always"
                if os.environ.get("RAPID_MLX_CHAT_APPROVALS") == "always"
                else "default"
            ),
        }
        if self.prompt_surface:
            body["surface"] = "cli" if supports_rich_output(sys.stdout) else "plain"
        key = os.environ.get("RAPID_MLX_API_KEY")
        if key:
            body["inference_api_key"] = key
        return body

    def turn(self, text: str, *, mode: str = "append"):
        body = self.context_request(text, mode=mode)
        state = TurnState(body["turn_id"])
        self.last_state = state
        renderer = HarnessRenderer(state, expanded=self.expanded)
        cancelled = False
        failures = 0
        submitted = False
        try:
            while state.terminal is None and failures < 4:
                try:
                    method = "GET" if submitted else "POST"
                    path = f"/turns/{state.turn_id}/events" if submitted else "/turns"
                    headers = dict(self.headers)
                    if submitted:
                        headers["Last-Event-ID"] = f"{state.turn_id}:{state.cursor}"
                    with self.client.stream(
                        method,
                        self.endpoint + PREFIX + path,
                        headers=headers,
                        **({"json": body} if not submitted else {}),
                    ) as response:
                        if response.status_code == 401:
                            raise httpx.ConnectError("Owner restarted")
                        response.raise_for_status()
                        submitted = True

                        for event in sse_events(response.iter_lines()):
                            previous_cursor = state.cursor
                            if state.apply(event):
                                failures = 0
                                renderer.update(event)
                                if event["event"]["type"] in {"context", "usage"}:
                                    context = state.context
                                    if context.get("limit_tokens"):
                                        self.input.context_percent = context.get(
                                            "percent", 0
                                        )
                                if event["event"]["type"] == "permission.request":
                                    try:
                                        self.permission(event["event"], renderer)
                                    except httpx.TransportError:
                                        state.cursor = previous_cursor
                                        raise
                    if state.terminal is None:
                        failures += 1
                        self.connect()
                except KeyboardInterrupt:
                    cancelled = True
                    print("\nStopping…", flush=True)
                    for attempt in range(3):
                        try:
                            self.request(
                                "POST",
                                f"/turns/{state.turn_id}/cancel",
                                json={},
                                idempotent=True,
                            )
                            break
                        except httpx.TransportError:
                            if attempt < 2:
                                try:
                                    self.connect()
                                except httpx.TransportError:
                                    pass
                    else:
                        print(
                            "Couldn't confirm the stop — the owner may still be running this turn."
                        )
                        return
                    # Cancellation acknowledgement follows cleanup. Replay its terminal.
                    submitted = True
                    failures = 0
                except (httpx.TransportError, ReplayRequiredError):
                    failures += 1
                    time.sleep(0.15)
                    self.connect()
            if state.terminal is None:
                print(
                    "\nConnection interrupted. Your conversation is saved; use /history to resume."
                )
            elif state.terminal["outcome"] == "error":
                renderer.finish()
                message = (
                    state.terminal.get("message")
                    or "The response could not finish. You can continue or /regen."
                )
                if supports_rich_output(sys.stdout):
                    from rich.text import Text

                    _console(sys.stdout).print(
                        Text(terminal_safe_text(message), style="red")
                    )
                else:
                    print("\n" + terminal_safe_text(message))
            elif state.terminal["outcome"] == "cancelled" or cancelled:
                renderer.finish()
                self.stop_armed = True
                print("\nStopped.")
            if mode == "append":
                self.attachments.clear()
        finally:
            renderer.finish()
        self.refresh()

    def command(self, line: str) -> bool:
        command, _, arg = line.partition(" ")
        arg = arg.strip()
        registered = COMMAND_BY_NAME.get(command)
        if registered:
            command = registered.name
        if command == "/quit":
            return False
        if command == "/help":
            print(HELP)
        elif command == "/new":
            self.new()
        elif command in {"/context", "/compact"}:
            body = self.context_request()
            if command == "/compact":
                body = {"request": body, "instructions": arg}
            result = self.request(
                "POST",
                f"/conversations/{self.conversation_id}{command}",
                json=body,
                retry=command != "/compact",
            )
            if command == "/compact":
                self.refresh()
                print(terminal_safe_text(result.get("notice", "Context compacted.")))
            context = result.get("context", result)
            self.input.context_percent = context.get("percent", 0)
            if notice := context.get("notice"):
                print_status(notice)
            print(
                f"Context: {context['used_tokens']:,} / {context['limit_tokens']:,} tokens"
            )
            for label, key in (
                ("System + tools", "system_tools_tokens"),
                ("Conversation", "conversation_tokens"),
                ("Tool output", "tool_output_tokens"),
                ("Free", "free_tokens"),
            ):
                print(f"  {label:<16} {context.get(key, 0):>8,}")
        elif command == "/history":
            if arg:
                self.select_conversation(arg)
            else:
                for index, item in enumerate(self.conversations(), 1):
                    title = (
                        item.get("display_title") or item.get("title") or "Conversation"
                    )
                    timestamp = history_time(
                        item.get("updated_at") or item["created_at"]
                    )
                    print(
                        f"{index}. {item['id'][:8]}  {terminal_safe_text(str(title))}  "
                        f"{timestamp}  {item.get('message_count', 0)} messages"
                    )
                print("Use /history NUMBER to resume.")
        elif command == "/branch":
            self.refresh()
            branches = self.snapshot["branches"]
            if arg:
                chosen = select_item(
                    branches, arg, "Choose a branch number or unique ID"
                )
                self.branch_id = chosen["id"]
                self.input.context_percent = 0
                self.attachments.clear()
                self.refresh()
                self.show_history()
            else:
                for index, item in enumerate(branches, 1):
                    mark = "*" if item["id"] == self.branch_id else " "
                    print(f"{mark} {index}. {item['id'][:8]}")
                print("Use /branch NUMBER to switch.")
        elif command == "/think":
            if arg not in {"on", "off"}:
                raise ValueError("Use /think on or /think off")
            self.thinking = arg == "on"
            print("Thinking on." if self.thinking else "Thinking off.")
        elif command == "/show-thinking":
            self.expanded = not self.expanded
            print("Thinking expanded." if self.expanded else "Thinking collapsed.")
            if self.expanded and self.last_state and self.last_state.reasoning:
                print(safe_markdown(self.last_state.reasoning))
        elif command == "/attach":
            paths = shlex.split(arg)
            if len(paths) != 1:
                raise ValueError("Use /attach PATH (quote paths containing spaces)")
            path = Path(paths[0]).expanduser()
            with path.open("rb") as file:
                content = file.read(MAX_ATTACHMENT_BYTES + 1)
            if len(content) > MAX_ATTACHMENT_BYTES:
                raise ValueError("Attachment exceeds the 20 MiB limit")
            try:
                result = self.request(
                    "POST",
                    f"/conversations/{self.conversation_id}/attachments",
                    params={"name": path.name},
                    content=content,
                )
            except httpx.HTTPStatusError as error:
                try:
                    message = error.response.json().get("detail")
                except ValueError:
                    message = None
                if isinstance(message, dict):
                    message = message.get("message")
                if isinstance(message, str):
                    raise ValueError(message) from None
                raise
            self.attachments.append(result["id"])
            print(
                f"Attached {terminal_safe_text(path.name)}. Ask a question to send it."
            )
        elif command in {"/edit", "/regen"}:
            self.refresh()
            users = [
                message
                for message in self.snapshot.get("messages", [])
                if message["role"] == "user"
            ]
            if not users:
                raise ValueError("Send a message first")
            body = {"branch_id": self.branch_id, "operation_id": uuid.uuid4().hex}
            if command == "/edit":
                replacement = (
                    arg
                    or self.input.prompt(
                        "Edited message> ", complete_commands=False
                    ).strip()
                )
                if not replacement:
                    raise ValueError("Edited message cannot be empty")
                body.update(message_id=users[-1]["id"], text=replacement)
            operation = "edit" if command == "/edit" else "regenerate"
            result = self.request(
                "POST", f"/conversations/{self.conversation_id}/{operation}", json=body
            )
            self.branch_id = result.get("branch_id") or result["id"]
            self.turn("", mode="regenerate")
        elif command == "/undo":
            result = self.request(
                "POST",
                f"/conversations/{self.conversation_id}/undo",
                json={"operation_id": uuid.uuid4().hex},
            )
            print(
                terminal_safe_text(
                    (result.get("receipt") or {}).get("summary", result["state"])
                )
            )
        elif command == "/export":
            self.refresh()
            pieces = ["# Rapid chat\n"]
            for message in self.snapshot.get("messages", []):
                text = self.message_text(message)
                if text:
                    pieces.append(f"## {message['role'].capitalize()}\n\n{text}\n")
                    sources = source_lines(message.get("sources", []), markdown=True)
                    if sources:
                        pieces.append("\n\n".join(sources) + "\n")
            pieces.extend(self.export_subruns())
            exported = "\n".join(pieces)
            if arg:
                paths = shlex.split(arg)
                if len(paths) != 1:
                    raise ValueError("Use /export PATH")
                with Path(paths[0]).expanduser().open("x", encoding="utf-8") as file:
                    file.write(exported)
                print("Export saved.")
            else:
                print(safe_markdown(exported))
        else:
            print("Unknown command. Type /help for commands.")
        return True

    def prompt_text(self) -> str:
        encoding = sys.stdout.encoding or "ascii"
        marker = "›" if codecs.lookup(encoding).name == "utf-8" else ">"
        return f"{'Thinking ' if self.thinking else ''}{marker} "

    def run(self):
        self.connect()
        self.lease_thread = threading.Thread(
            target=self.heartbeat, name="chat-client-lease", daemon=True
        )
        self.lease_thread.start()
        resume = getattr(self.args, "resume", None)
        if not resume or not self.select_conversation(resume):
            self.new()
        interrupts = 0
        while True:
            try:
                line = self.input.prompt(self.prompt_text()).strip()
                if not line:
                    continue
                interrupts = 0
                self.stop_armed = False
                if line in {"exit", "quit"}:
                    break
                if line.startswith("/"):
                    if not self.command(line):
                        break
                else:
                    self.turn(line)
                interrupts = int(self.stop_armed)
            except EOFError:
                break
            except KeyboardInterrupt:
                interrupts += 1
                print("\nPress Ctrl-C again to exit." if interrupts < 2 else "")
                if interrupts >= 2:
                    break
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    print(
                        f"\nChat request was rejected ({exc.response.status_code}). Refresh with /history and try again."
                    )
                elif isinstance(exc, httpx.HTTPError):
                    print("\nChat connection unavailable. Try again to reconnect.")
                else:
                    print(f"\n{terminal_safe_text(str(exc))}")


def run_chat(args: Any, inference_url: str, model: str):
    if getattr(args, "mcp_config", None):
        print(
            "The experimental harness supports attachments; MCP is not available yet.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if getattr(args, "system", None):
        print(
            "Custom system prompts are not available in the experimental harness yet.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    client = ChatClient(args, inference_url, model)
    try:
        client.run()
    except (ValueError, RuntimeError) as exc:
        print(terminal_safe_text(str(exc)), file=sys.stderr)
        raise SystemExit(1) from None
    except (httpx.HTTPError, OSError):
        print(
            "Could not connect to the chat owner. Check the profile and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        client.close()
