# SPDX-License-Identifier: Apache-2.0
"""One reusable terminal reader, independent of response rendering."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession

    from rapid_mlx.chat_client import ChatCommand


def command_completer(commands: tuple[ChatCommand, ...]):
    """Build the editor lazily so piped input does not import prompt-toolkit."""
    import shlex

    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document

    class CommandCompleter(Completer):
        paths = PathCompleter(expanduser=True)

        def get_completions(self, document, complete_event):
            before = document.text_before_cursor
            if not before.startswith("/"):
                return
            word, separator, argument = before.partition(" ")
            if not separator:
                if any(char.isspace() for char in before):
                    return
                query = word[1:]
                matches = [command for command in commands if query in command.name[1:]]
                # Stable sorting retains the help order within each rank.
                matches.sort(key=lambda command: not command.name[1:].startswith(query))
                for command in matches:
                    yield Completion(
                        command.name + (" " if command.argument_hint else ""),
                        start_position=-len(before),
                        display=command.name,
                        display_meta=command.description,
                    )
            elif word == "/think" and document.is_cursor_at_the_end:
                for value in ("on", "off"):
                    if value.startswith(argument) and value != argument:
                        yield Completion(value, start_position=-len(argument))
            elif word == "/attach" and document.is_cursor_at_the_end:
                # Decode a single shell argument, including an open quote, then
                # quote the completed path so filenames with spaces still attach.
                try:
                    parts = shlex.split(argument)
                except ValueError:
                    for quote in ("'", '"'):
                        try:
                            parts = shlex.split(argument + quote)
                            break
                        except ValueError:
                            continue
                    else:
                        return
                if len(parts) > 1:
                    return
                path = parts[0] if parts else ""
                for completion in self.paths.get_completions(
                    Document(path), complete_event
                ):
                    suffix = completion.text
                    if completion.display_text.endswith("/"):
                        suffix += "/"
                    if suffix:
                        yield Completion(
                            shlex.quote(path + suffix),
                            start_position=-len(argument),
                            display=completion.display,
                        )

    return CommandCompleter()


class ChatInput:
    """Preserve editing and typeahead without libedit's terminal drain barrier.

    On macOS, libedit waits for pending echoed output to drain before returning
    a submitted line. A paused PTY reader can therefore prevent admission until
    Ctrl-C discards the line. The terminal editor below restores modes with
    TCSANOW, so reading a submitted line does not depend on display consumption.
    """

    def __init__(self, commands: tuple[ChatCommand, ...] = ()):
        self._session: PromptSession[str] | None = None
        self._commands = commands
        self._complete_commands = True
        self._approval_options = ()
        self._approval_selected = 0
        self._approval_data = None
        self._approval_expanded = False
        self.context_percent = 0
        self.approval_reason = ""

    def prompt(
        self,
        text: str,
        *,
        complete_commands: bool = True,
        timeout: float | None = None,
        timeout_result: str = "n",
    ) -> str:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return input(text)
        self._complete_commands = complete_commands
        if self._session is None:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.application import get_app
            from prompt_toolkit.completion import (
                CompleteEvent,
                Completion,
                ConditionalCompleter,
            )
            from prompt_toolkit.filters import Condition, has_completions
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.output.vt100 import Vt100_Output
            from prompt_toolkit.styles import Style

            @Condition
            def slash_input():
                return (
                    self._complete_commands
                    and get_app().current_buffer.text.startswith("/")
                )

            bindings = KeyBindings()

            @Condition
            def approval_mode():
                return bool(self._approval_options)

            @bindings.add("y", filter=approval_mode, eager=True)
            @bindings.add("a", filter=approval_mode, eager=True)
            @bindings.add("n", filter=approval_mode, eager=True)
            @bindings.add("N", filter=approval_mode, eager=True)
            def decide_approval(event):
                key = event.key_sequence[0].key
                if key in self._approval_options:
                    event.app.exit(result=key)

            @bindings.add("up", filter=approval_mode, eager=True)
            @bindings.add("down", filter=approval_mode, eager=True)
            def select_approval(event):
                step = -1 if event.key_sequence[0].key == "up" else 1
                self._approval_selected = (self._approval_selected + step) % len(
                    self._approval_options
                )

            @bindings.add("enter", filter=approval_mode, eager=True)
            def accept_approval(event):
                event.app.exit(result=self._approval_options[self._approval_selected])

            @bindings.add("1", filter=approval_mode, eager=True)
            @bindings.add("2", filter=approval_mode, eager=True)
            @bindings.add("3", filter=approval_mode, eager=True)
            @bindings.add("4", filter=approval_mode, eager=True)
            def numbered_approval(event):
                index = int(event.key_sequence[0].key) - 1
                if index < len(self._approval_options):
                    event.app.exit(result=self._approval_options[index])

            @bindings.add("v", filter=approval_mode, eager=True)
            def expand_approval(event):
                from prompt_toolkit.application import run_in_terminal

                self._approval_expanded = not self._approval_expanded
                run_in_terminal(self._print_approval_card)

            completer = command_completer(self._commands)

            @bindings.add("down", filter=has_completions)
            @bindings.add("up", filter=has_completions)
            def navigate(event):
                state = event.current_buffer.complete_state
                index = state.complete_index
                backwards = event.key_sequence[0].key == "up"
                if index is None:
                    index = len(state.completions) - 1 if backwards else 0
                else:
                    index = (index + (-1 if backwards else 1)) % len(state.completions)
                # Highlight without replacing the typed query until acceptance.
                state.go_to_index(index)

            @bindings.add("tab", filter=slash_input)
            @bindings.add("enter", filter=has_completions)
            def accept_completion(event):
                buffer = event.current_buffer
                state = buffer.complete_state
                if event.key_sequence[0].key == "c-m":
                    # Fully typed commands with no required argument retain
                    # their one-Enter behavior, including aliases and /history.
                    exact = next(
                        (
                            command
                            for command in self._commands
                            if buffer.text in (command.name, *command.aliases)
                        ),
                        None,
                    )
                    if exact and (
                        not exact.argument_hint or exact.argument_hint.startswith("[")
                    ):
                        buffer.complete_state = None
                        buffer.validate_and_handle()
                        return
                if state:
                    completion = state.current_completion or state.completions[0]
                else:
                    # Tab can arrive before the asynchronous menu, or after a
                    # cursor move. Use the same replacement rules in both cases.
                    completion = next(
                        iter(
                            completer.get_completions(
                                buffer.document,
                                CompleteEvent(completion_requested=True),
                            )
                        ),
                        None,
                    )
                    if completion is None:
                        return
                buffer.complete_state = None
                # Completion replaces the entire command even when editing its
                # middle; keep any existing arguments and their separator.
                before = buffer.document.text_before_cursor
                skip_separator = False
                if before.startswith("/") and not any(c.isspace() for c in before):
                    after = buffer.document.text_after_cursor
                    word_tail = next(
                        (i for i, c in enumerate(after) if c.isspace()), len(after)
                    )
                    buffer.delete(word_tail)
                    if completion.text.endswith(" ") and after[word_tail:]:
                        completion = Completion(
                            completion.text.rstrip(), completion.start_position
                        )
                        skip_separator = True
                buffer.apply_completion(completion)
                if skip_separator:
                    buffer.cursor_position += 1

            output = Vt100_Output.from_pty(sys.stdout)
            # A line prompt does not need cursor-position replies. Besides
            # supporting simple PTYs, this prevents another input handshake.
            output.enable_cpr = False
            self._session = PromptSession(
                output=output,
                enable_history_search=~slash_input,
                completer=ConditionalCompleter(completer, slash_input),
                complete_while_typing=True,
                key_bindings=bindings,
                reserve_space_for_menu=len(self._commands) + 1,
                style=Style.from_dict(
                    {
                        "completion-menu": "bg:default fg:default",
                        "completion-menu.completion.current": "bg:default fg:default reverse",
                        "completion-menu.meta.completion": (
                            "bg:default"
                            if "NO_COLOR" in os.environ
                            else "bg:default fg:ansibrightblack"
                        ),
                        "completion-menu.meta.completion.current": (
                            "bg:default reverse"
                            if "NO_COLOR" in os.environ
                            else "bg:default fg:ansibrightblack reverse"
                        ),
                        "scrollbar.background": "bg:default",
                        "scrollbar.button": "bg:default",
                    }
                ),
            )

        def formatted_prompt():
            from prompt_toolkit.application import get_app

            if self._approval_options:
                fragments = []
                labels = {
                    "y": "Yes, run once",
                    "a": "Yes, for this conversation",
                    "n": "No",
                    "N": "No, and tell it why…",
                }
                for index, key in enumerate(self._approval_options):
                    selected = index == self._approval_selected
                    fragments.append(
                        (
                            "bold" if selected else "",
                            f" {'❯' if selected else ' '} {index + 1}. {labels[key]}\n",
                        )
                    )
                return fragments

            # Keep the same visual prompt on acceptance, but distinguish the
            # final repaint from a new ready prompt (› / Thinking ›, or the
            # ASCII fallback) for terminal automation.
            # Explicit default foreground also preserves the trailing space
            # instead of replacing it with a cursor-forward instruction.
            if get_app().is_done and text.endswith(" "):
                return [
                    ("fg:default", text[:-1]),
                    ("[ZeroWidthEscape]", "\x1b[0m"),
                    ("fg:default", " "),
                ]
            return [("fg:default", text)]

        def start_timeout():
            if timeout is not None:
                import asyncio

                from prompt_toolkit.application import get_app

                app = get_app()

                async def expire():
                    await asyncio.sleep(max(0, timeout))
                    if not app.is_done:
                        app.exit(result=timeout_result)

                app.create_background_task(expire())

        def context_prompt():
            if (
                self._approval_options
                or not self._complete_commands
                or self.context_percent < 50
            ):
                return []
            return [
                (
                    "" if "NO_COLOR" in os.environ else "fg:ansibrightblack",
                    f"context {self.context_percent}%",
                )
            ]

        assert self._session is not None
        answer: str = self._session.prompt(
            formatted_prompt, pre_run=start_timeout, rprompt=context_prompt
        )
        return answer

    def _print_approval_card(self):
        import shutil

        from rapid_mlx.chat_harness_render import approval_card

        # Keep the static card out of the editor's acceptance repaint. A large
        # syntax preview otherwise fills the PTY output queue before the
        # submitted decision can return, violating input-drain independence.
        sys.stdout.write(
            approval_card(
                self._approval_data or {},
                expanded=self._approval_expanded,
                width=shutil.get_terminal_size().columns,
            )
        )
        sys.stdout.flush()

    def approval(self, *, allow_conversation=False, timeout=120, data=None):
        self.approval_reason = ""
        self._approval_data = data
        self._approval_selected = 0
        self._approval_expanded = False
        self._approval_options = (
            ("y", "a", "n", "N") if allow_conversation else ("y", "n", "N")
        )
        try:
            # Piped clients fail closed; tool output cannot consume the next chat
            # line as consent. Interactive approval uses the existing editor.
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                return "n"
            self._print_approval_card()
            started = time.monotonic()
            result = self.prompt("", complete_commands=False, timeout=timeout)
            if result == "N":
                self._approval_options = ()
                self.approval_reason = self.prompt(
                    "Denial reason (Enter skips)> ",
                    complete_commands=False,
                    timeout=max(0, timeout - (time.monotonic() - started)),
                    timeout_result="",
                ).strip()[:512]
                return "n"
            return result
        finally:
            self._approval_options = ()
            self._approval_data = None

    def close(self) -> None:
        if self._session is not None:
            self._session.app.input.close()
            self._session = None
