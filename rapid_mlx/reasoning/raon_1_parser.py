# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for the Raon-1 chat template (``--reasoning-parser raon_1``).

The chat template (``chat_template.jinja`` in the model repository) ends the
generation prompt with ``<think>\\n`` (or ``<think></think>\\n\\n`` when
``enable_thinking`` is false). The model then writes::

    REASONING </think>\\n\\n CONTENT <tool_call>\\n<function=NAME>\\n
    <parameter=KEY>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>

``<think>``, ``</think>``, ``<tool_call>`` and ``</tool_call>`` are single
special tokens. This module also holds the markup tokenizer shared with
``raon_1_tool_parser``: it splits a stream of (text, token ids) into markup tags
and text, using token ids for the special tokens and text for the tags the
template spells out (``<function=``, ``<parameter=`` ...). A trailing partial tag
is held back until the next delta shows what it is.

Adapted from the vendor parser: only parser interfaces and lifecycle hooks changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from .base import DeltaMessage, ReasoningParser

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
FUNCTION_OPEN = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "</parameter>"
TAG_CLOSE = ">"
SPECIAL_MARKERS = (THINK_START, THINK_END, TOOL_CALL_START, TOOL_CALL_END)
TEXT_TAGS = (FUNCTION_OPEN, FUNCTION_CLOSE, PARAMETER_OPEN, PARAMETER_CLOSE, TAG_CLOSE)
# Tokenizer special tokens outside the markup (end-of-turn, padding) are removed.
DISCARD = "<discard>"


class RaonMarkupTokens:
    """Token ids of the template's special markers, looked up once per tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        vocab = tokenizer.get_vocab()
        missing = [text for text in SPECIAL_MARKERS if text not in vocab]
        if missing:
            raise ValueError(
                "raon_1 parsers require the Raon chat-template special tokens "
                f"{list(SPECIAL_MARKERS)}; missing from the tokenizer: {missing}"
            )
        self.ids = {text: vocab[text] for text in SPECIAL_MARKERS}
        self.kind_by_id: dict[int, str] = {
            token_id: text for text, token_id in self.ids.items()
        }
        self.literals: dict[str, str] = {
            text: text for text in SPECIAL_MARKERS + TEXT_TAGS
        }
        try:
            specials = list(
                zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)
            )
        except (AttributeError, NotImplementedError):
            specials = []
        for text, token_id in specials:
            if text not in self.literals:
                self.kind_by_id[token_id] = DISCARD
                self.literals[text] = DISCARD
        # Text matches of id-backed kinds are plain text once real token ids are seen.
        self.id_kinds = frozenset(self.kind_by_id.values())
        self.first_chars = frozenset(literal[0] for literal in self.literals)
        self.partials = frozenset(
            literal[:end] for literal in self.literals for end in range(1, len(literal))
        )
        self.by_first: dict[str, list[str]] = {}
        for literal in sorted(self.literals, key=len, reverse=True):
            self.by_first.setdefault(literal[0], []).append(literal)
        if any(
            other != literal and other.startswith(literal)
            for literal in self.literals
            for other in self.literals
        ):
            raise ValueError("raon_1 markup tags must not prefix one another")
        self._tokenizer = tokenizer
        self._texts: dict[int, str] = {}

    def text(self, token_id: int) -> str:
        if token_id not in self._texts:
            text = self._tokenizer.decode([token_id])
            self._texts[token_id] = text if isinstance(text, str) else ""
        return self._texts[token_id]


class _Text:
    """Text from the detokenizer with the number of tokens that produced it."""

    __slots__ = ("text", "token_texts", "count")

    def __init__(
        self, text: str, token_texts: tuple[str, ...] = (), count: int = 0
    ) -> None:
        self.text, self.token_texts, self.count = text, token_texts, count


class _Special:
    """A special token located by its id."""

    __slots__ = ("kind", "text")

    def __init__(self, kind: str, text: str) -> None:
        self.kind, self.text = kind, text


class Piece:
    """One markup tag (``kind`` set) or one run of text (``kind`` is None).

    ``always`` marks text that is emitted even when empty, so token counts of
    text the detokenizer has not produced yet are still attributed.
    """

    __slots__ = ("kind", "value", "count", "always")

    def __init__(
        self, kind: str | None, value: str, count: int = 0, always: bool = False
    ) -> None:
        self.kind, self.value, self.count, self.always = kind, value, count, always


class RaonMarkupStream:
    """Incremental splitter of model output into markup ``Piece`` objects."""

    def __init__(self, tokens: RaonMarkupTokens) -> None:
        self.tokens = tokens
        self.reset()

    def reset(self) -> None:
        self.saw_token_ids = False
        self._buffer = ""
        self._buffer_counts: list[int] = []
        self._waiting: list[_Special] = []
        self._waiting_counts: list[int] = []
        self._waiting_tail_count = 0
        self._waiting_text = ""

    # -- public ---------------------------------------------------------------

    def feed(self, text: str, token_ids: Sequence[int]) -> list[Piece]:
        tokens = self.tokens
        if token_ids:
            self.saw_token_ids = True
        if (
            text
            and not self._buffer
            and not self._waiting
            and tokens.first_chars.isdisjoint(text)
            and not any(token_id in tokens.kind_by_id for token_id in token_ids)
        ):
            return [Piece(None, text, len(token_ids), always=True)]
        items = self._split_specials(text, token_ids)
        if len(items) == 1 and isinstance(items[0], _Text):
            item = items[0]
            lexed = self._lex(item.text, item.token_texts, item.count)
            if len(lexed) == 1 and lexed[0][0] is None:
                return [Piece(None, lexed[0][1], lexed[0][2], always=True)]
            return self._pieces(lexed)
        return self._items(items)

    def finish(self) -> list[Piece]:
        pieces = self._items(self._release_waiting())
        pieces.extend(self._pieces(self._lex_flush()))
        return pieces

    # -- special tokens -------------------------------------------------------

    def _items(self, items: list[_Text | _Special]) -> list[Piece]:
        pieces: list[Piece] = []
        for item in items:
            if isinstance(item, _Special):
                pieces.extend(self._pieces(self._lex_flush()))
                pieces.append(Piece(item.kind, item.text))
            elif not item.text and item.count:
                pieces.append(Piece(None, "", item.count, always=True))
            else:
                pieces.extend(
                    self._pieces(self._lex(item.text, item.token_texts, item.count))
                )
        return pieces

    def _pieces(self, lexed: list[tuple[str | None, str, int]]) -> list[Piece]:
        strict = self.saw_token_ids
        id_kinds = self.tokens.id_kinds
        return [
            Piece(
                None if kind is None or (strict and kind in id_kinds) else kind,
                value,
                count,
            )
            for kind, value, count in lexed
        ]

    def _split_specials(
        self, text: str, token_ids: Sequence[int]
    ) -> list[_Text | _Special]:
        tokens = self.tokens
        leading: list[_Text | _Special] = []
        tail_count = 0
        if self._waiting:
            leading, text, tail_count = self._resolve_waiting(text, len(token_ids))
        if not any(token_id in tokens.kind_by_id for token_id in token_ids):
            if text:
                texts = tuple(tokens.text(token_id) for token_id in token_ids)
                if not leading:
                    return [_Text(text, texts, len(texts))]
                leading.append(_Text(text, texts, tail_count + len(texts)))
            return leading
        items: list[_Text | _Special] = []
        run: list[str] = []
        for token_id in token_ids:
            kind = tokens.kind_by_id.get(token_id)
            if kind is None:
                run.append(tokens.text(token_id))
                continue
            if run and "".join(run):
                items.append(_Text("".join(run), tuple(run), len(run)))
            run = []
            items.append(_Special(kind, tokens.text(token_id)))
        if run and "".join(run):
            items.append(_Text("".join(run), tuple(run), len(run)))
        if tail_count:
            items.insert(0, _Text("", (), tail_count))
        if not text:
            # No detokenized text yet: wait before acting on the special tokens.
            count = 0
            for item in items:
                if isinstance(item, _Text):
                    count += item.count
                else:
                    self._waiting.append(item)
                    self._waiting_counts.append(count)
                    count = 0
            self._waiting_tail_count += count
            return leading
        return leading + self._align(text, items)

    def _align(
        self, text: str, items: list[_Text | _Special]
    ) -> list[_Text | _Special]:
        """Place token-decoded items onto the detokenizer's text."""
        if not items:
            return [_Text(text)]
        decoded = "".join(item.text for item in items)
        if not decoded:
            return [_Text(text)] + items
        start = text.find(decoded)
        if start > 0:
            return [_Text(text[:start])] + items
        if start == 0:
            return items
        # Per-token decoding differs from the stream text; split at the special tokens.
        specials = [item for item in items if isinstance(item, _Special)]
        if not specials:
            return [_Text(text)]
        gaps: list[list[str]] = [[] for _ in range(len(specials) + 1)]
        gap_counts = [0] * (len(specials) + 1)
        gap = 0
        for item in items:
            if isinstance(item, _Special):
                gap += 1
            else:
                gaps[gap].extend(item.token_texts)
                gap_counts[gap] += item.count
        positions = [-1] * len(specials)
        limit = len(text)
        for index in range(len(specials) - 1, -1, -1):
            found = text.rfind(specials[index].text, 0, limit)
            if found >= 0:
                positions[index] = limit = found
        aligned: list[_Text | _Special] = []
        used = 0
        for index, special in enumerate(specials):
            found = positions[index]
            if found >= used:
                if found > used:
                    aligned.append(
                        _Text(text[used:found], tuple(gaps[index]), gap_counts[index])
                    )
                aligned.append(special)
                used = found + len(special.text)
                continue
            later = any(position >= 0 for position in positions[index + 1 :])
            if not later and used < len(text) and special.kind != DISCARD:
                self._waiting_text += text[used:]
                used = len(text)
            self._waiting.append(special)
            self._waiting_counts.append(gap_counts[index])
        if used < len(text):
            aligned.append(_Text(text[used:], tuple(gaps[-1]), gap_counts[-1]))
        return aligned

    def _resolve_waiting(
        self, text: str, new_count: int
    ) -> tuple[list[_Text | _Special], str, int]:
        waiting, counts = self._waiting, self._waiting_counts
        tail_count = self._waiting_tail_count
        self._waiting, self._waiting_counts, self._waiting_tail_count = [], [], 0
        resolved: list[_Text | _Special] = []
        rest = self._waiting_text + text
        self._waiting_text = ""
        for special, count in zip(waiting, counts):
            found = rest.find(special.text)
            if found > 0:
                resolved += [_Text(rest[:found], (), count), special]
                rest = rest[found + len(special.text) :]
            elif found == 0:
                resolved.append(special)
                rest = rest[len(special.text) :]
            else:
                if rest:
                    self._waiting_text += rest
                    count += new_count
                    rest = ""
                self._waiting.append(special)
                self._waiting_counts.append(count)
        if self._waiting:
            self._waiting_tail_count = tail_count
            tail_count = 0
        return resolved, rest, tail_count

    def _release_waiting(self) -> list[_Text | _Special]:
        if not self._waiting and not self._waiting_text:
            return []
        released: list[_Text | _Special] = []
        if self._waiting_text:
            released.append(
                _Text(
                    self._waiting_text,
                    (),
                    self._waiting_counts[0] if self._waiting_counts else 0,
                )
            )
        released.extend(self._waiting)
        if self._waiting_tail_count:
            released.append(_Text("", (), self._waiting_tail_count))
        self._waiting, self._waiting_counts = [], []
        self._waiting_tail_count, self._waiting_text = 0, ""
        return released

    # -- text tags ------------------------------------------------------------

    @staticmethod
    def _char_counts(text: str, token_texts: tuple[str, ...], count: int) -> list[int]:
        counts = [0] * len(text)
        if not text:
            return counts
        if not token_texts:
            if count:
                counts[0] = count
            return counts
        position = placed = 0
        for token_text in token_texts:
            if not token_text:
                continue
            found = text.find(token_text, position)
            if found < 0:
                continue
            counts[found] += 1
            placed += 1
            position = found + len(token_text)
        if count > placed:
            counts[0] += count - placed
        return counts

    def _take(self, length: int) -> tuple[str, int]:
        value, self._buffer = self._buffer[:length], self._buffer[length:]
        count = sum(self._buffer_counts[:length])
        del self._buffer_counts[:length]
        return value, count

    def _lex(
        self, text: str, token_texts: tuple[str, ...], count: int
    ) -> list[tuple[str | None, str, int]]:
        counts = self._char_counts(text, token_texts, count)
        if not self._buffer and self.tokens.first_chars.isdisjoint(text):
            return [(None, text, sum(counts))]
        self._buffer += text
        self._buffer_counts.extend(counts)
        return self._drain(final=False)

    def _lex_flush(self) -> list[tuple[str | None, str, int]]:
        return self._drain(final=True) if self._buffer else []

    def _drain(self, final: bool) -> list[tuple[str | None, str, int]]:
        tokens = self.tokens
        out: list[tuple[str | None, str, int]] = []
        while self._buffer:
            buffer = self._buffer
            if tokens.first_chars.isdisjoint(buffer):
                out.append((None, *self._take(len(buffer))))
                break
            if not final and buffer in tokens.partials:
                break
            tag = next(
                (
                    literal
                    for literal in tokens.by_first.get(buffer[0], ())
                    if buffer.startswith(literal)
                ),
                None,
            )
            if tag is not None:
                out.append((tokens.literals[tag], *self._take(len(tag))))
            else:
                out.append((None, *self._take(self._text_end(buffer))))
        return out

    def _text_end(self, buffer: str) -> int:
        """Length of the text run before the next position that may start a tag."""
        literals = self.tokens.literals
        first_chars = self.tokens.first_chars
        for index in range(1, len(buffer)):
            if buffer[index] not in first_chars:
                continue
            tail = buffer[index:]
            if any(
                tail[: len(literal)] == literal[: len(tail)] for literal in literals
            ):
                return index
        return len(buffer)


# Reasoning state machine ------------------------------------------------------

_REASONING, _CONTENT = "reasoning", "content"


class RaonReasoningParser(ReasoningParser):
    """Splits ``REASONING</think>CONTENT`` into ``reasoning`` and ``content``.

    Reasoning ends at ``</think>``; a ``<tool_call>`` written before
    ``</think>`` also ends it and starts the content handed to the tool parser.
    Further ``</think>`` tokens in content are dropped. Trailing whitespace of
    the reasoning is removed. With ``enable_thinking`` false everything is
    content.
    """

    engine_based_streaming = True

    def __init__(self, tokenizer: Any = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(tokenizer)
        self.model_tokenizer = tokenizer
        tokenizer = tokenizer or _TextTokenizer()
        self.markup = RaonMarkupTokens(tokenizer)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking = chat_kwargs.get("enable_thinking", True)
        self._stream = RaonMarkupStream(self.markup)
        self._started = False
        self._count_from_stream = False
        self._counter: RaonReasoningParser | None = None
        self._reset()

    # -- state ----------------------------------------------------------------

    def _reset(self) -> None:
        self._stream.reset()
        self._state = _REASONING if self.thinking else _CONTENT
        self._ended = False
        self._held_content = ""
        self._content_started = False
        self._held_reasoning = ""
        self._reasoning_count = 0
        self._streamed_reasoning: list[str] = []

    def _events(self, pieces: list[Piece]) -> list[tuple[str, str]]:
        """Apply the reasoning grammar; tool markup stays text for the tool parser."""
        events: list[tuple[str, str]] = []
        for piece in pieces:
            kind, value = piece.kind, piece.value
            if kind is None:
                if not value and not piece.always:
                    continue
            elif kind == DISCARD or kind == THINK_START and self._state == _REASONING:
                continue
            elif kind == THINK_END:
                if self._state == _REASONING:
                    self._state = _CONTENT
                    events.append(("end", ""))
                continue
            elif kind == TOOL_CALL_START:
                self._state = _CONTENT
                events += [("end", value), ("content", value)]
                continue
            elif kind == FUNCTION_OPEN and self._state == _CONTENT:
                events.append(("content", value))
                continue
            if self._state == _REASONING:
                self._reasoning_count += piece.count
                events.append(("reasoning", value))
            else:
                events.append(("content", value))
        return events

    def _finish_events(self) -> list[tuple[str, str]]:
        events = self._events(self._stream.finish())
        if self._state == _REASONING:
            self._state = _CONTENT
            events.append(("end", ""))
        return events

    def _delta(
        self, events: list[tuple[str, str]], finished: bool = False
    ) -> DeltaMessage | None:
        if not events and not self._held_content:
            return None
        reasoning = (
            "".join(value for kind, value in events if kind == "reasoning") or None
        )
        if any(kind == "end" for kind, _ in events):
            self._ended = True
        content = self._held_content + "".join(
            value for kind, value in events if kind == "content"
        )
        self._held_content = ""
        if not self._content_started and content:
            if content.strip():
                self._content_started = True
            elif not finished:
                self._held_content, content = content, ""
        if not content and not reasoning:
            return None
        delta = DeltaMessage(
            **({"content": content} if content else {}),
            **({"reasoning": reasoning} if reasoning else {}),
        )
        if reasoning is not None:
            self._streamed_reasoning.append(reasoning)
        return delta

    def _trim_reasoning(self, delta: DeltaMessage | None) -> DeltaMessage | None:
        """Hold trailing reasoning whitespace until more reasoning follows."""
        if delta is not None and delta.reasoning is not None:
            combined = self._held_reasoning + delta.reasoning
            trimmed = combined.rstrip()
            self._held_reasoning = combined[len(trimmed) :]
            delta.reasoning = trimmed or None
            if delta.reasoning is None and delta.content is None:
                return None
        elif self._held_reasoning and self._ended:
            self._held_reasoning = ""
        return delta

    @staticmethod
    def _force_content(request: Any) -> bool:
        kwargs = getattr(request, "chat_template_kwargs", None)
        return bool(
            kwargs
            and (
                kwargs.get("enable_thinking") is False
                or kwargs.get("force_nonempty_content") is True
            )
        )

    # -- ReasoningParser interface ---------------------------------------------

    @property
    def reasoning_start_str(self) -> str:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return THINK_END

    def adjust_request(self, request: Any) -> Any:
        # The markers are special tokens; keep them in the text the parsers see.
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        ids = self.markup.ids
        if not input_ids:
            return not self.thinking
        for token_id in reversed(input_ids):
            if token_id == ids[THINK_END]:
                return True
            if token_id == ids[THINK_START]:
                return False
        # A tool call without its closing tag also ends the reasoning.
        for index in range(len(input_ids) - 1, -1, -1):
            if input_ids[index] == ids[THINK_START]:
                return False
            if input_ids[index] == ids[TOOL_CALL_START]:
                if ids[TOOL_CALL_END] in input_ids[index + 1 :]:
                    continue
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end = self.markup.ids[THINK_END]
        for index in range(len(input_ids) - 1, -1, -1):
            if input_ids[index] == end:
                return input_ids[index + 1 :]
        return input_ids

    def _extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        self._count_from_stream = False
        if not self.thinking:
            reasoning, content = None, model_output
        else:
            self._reset()
            events = (
                self._events(self._stream.feed(model_output, []))
                + self._finish_events()
            )
            reasoning = (
                "".join(value for kind, value in events if kind == "reasoning").rstrip()
                or None
            )
            content = (
                "".join(value for kind, value in events if kind == "content") or None
            )
        if self._force_content(request) and (content is None or not content.strip()):
            reasoning, content = content, reasoning
        return reasoning, content

    def _extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self._count_from_stream = True
        if not self._started:
            self._started = True
            self._reset()
        events = self._events(self._stream.feed(delta_text, delta_token_ids))
        return self._trim_reasoning(self._delta(events))

    def has_engine_confirmed_reasoning_end(self) -> bool:
        return self._ended

    def finish_streaming(self) -> DeltaMessage | None:
        events = self._finish_events()
        if events or self._held_content:
            return self._delta(events, finished=True)
        return None

    def get_streaming_fallback_content(self, text: str, request: Any) -> str | None:
        if not self._force_content(request):
            return None
        return "".join(self._streamed_reasoning) or None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if self._count_from_stream:
            return self._reasoning_count
        if not token_ids:
            return 0
        if self._counter is None:
            self._counter = type(self)(
                self.model_tokenizer,
                chat_template_kwargs={"enable_thinking": self.thinking},
            )
        counter = self._counter
        counter._reset()
        counter._events(
            counter._stream.feed(self.model_tokenizer.decode(token_ids), token_ids)
        )
        counter._finish_events()
        return counter._reasoning_count

    def set_enable_thinking(self, enabled: bool | None) -> None:
        self.thinking = enabled is not False
        self.reset_state()

    def reset_state(self) -> None:
        self._started = False
        self._reset()

    def extract_reasoning(self, model_output: str, enable_thinking: bool | None = None):
        self.thinking = enable_thinking is not False
        return self._extract_reasoning(
            model_output, SimpleNamespace(chat_template_kwargs={})
        )

    def extract_reasoning_streaming(
        self, previous_text: str, current_text: str, delta_text: str
    ):
        return self._extract_reasoning_streaming(
            previous_text, current_text, delta_text, [], [], []
        )

    def finish_stream(self):
        return self.finish_streaming()

    def is_open_in_think(self, accumulated_text: str) -> bool:
        return (
            self.thinking
            and THINK_END not in accumulated_text
            and TOOL_CALL_START not in accumulated_text
        )


class _TextTokenizer:
    """The public parser API also permits text-only callers without a tokenizer."""

    all_special_tokens = ()
    all_special_ids = ()

    def get_vocab(self):
        return {marker: index for index, marker in enumerate(SPECIAL_MARKERS)}
