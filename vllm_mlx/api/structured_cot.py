# SPDX-License-Identifier: Apache-2.0
"""Structured CoT logits processor.

Implements the constrained-decoding technique from
https://andthattoo.dev/blog/structured_cot. The constraint is applied **only**
inside the model's reasoning block (between ``<think>`` and ``</think>``); the
final answer is left unconstrained. The default grammar is vendored from
https://github.com/andthattoo/structured-cot under
``vllm_mlx/api/grammars/structured_cot.gbnf``.

Two modes are supported:

* ``outlines`` (preferred): compiles the grammar via ``outlines`` CFG support
  and masks logits against the FSM for every token emitted inside ``<think>``.
* ``sectioned`` (fallback): a purpose-built parser that handles GBNFs of the
  shape used by the upstream repo (``<think>\\n`` + ordered ``HEADER: line\\n``
  sections + ``</think>\\n\\n``). This avoids requiring ``outlines`` at runtime
  while still constraining the reasoning block.

The processor exposes ``__call__(token_context, logits) -> logits`` so it plugs
straight into the scheduler's per-request ``logits_processors`` list.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_GRAMMAR_PATH = (
    Path(__file__).resolve().parent / "grammars" / "structured_cot.gbnf"
)
LCB_PLAN_GRAMMAR_PATH = (
    Path(__file__).resolve().parent / "grammars" / "structured_cot_lcb_plan.gbnf"
)

THINK_OPEN = "<think>\n"
THINK_CLOSE = "</think>\n\n"


def load_grammar(path: str | Path | None) -> str:
    """Load a GBNF grammar from disk; ``None`` returns the bundled default."""
    grammar_path = Path(path) if path else DEFAULT_GRAMMAR_PATH
    return grammar_path.read_text(encoding="utf-8")


def _strip_comments(grammar_text: str) -> str:
    return "\n".join(
        line for line in grammar_text.splitlines() if not line.lstrip().startswith("#")
    )


_THINK_RULE_RE = re.compile(r'^\s*think\s*::=\s*(.+)$', re.MULTILINE)
_LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _unescape(literal: str) -> str:
    return (
        literal.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def parse_sectioned_grammar(grammar_text: str) -> list[str] | None:
    """Best-effort parse of the upstream ``<think>``/section/``</think>`` shape.

    Returns the ordered list of section headers (e.g. ``["GOAL: ", "APPROACH: ",
    "EDGE: "]``) when the grammar matches, else ``None``.
    """
    body = _strip_comments(grammar_text)
    match = _THINK_RULE_RE.search(body)
    if not match:
        return None
    literals = [_unescape(m.group(1)) for m in _LITERAL_RE.finditer(match.group(1))]
    if len(literals) < 3:
        return None
    if literals[0] != THINK_OPEN or literals[-1] != THINK_CLOSE:
        return None
    sections = literals[1:-1]
    if not sections or any(not s.endswith(": ") for s in sections):
        return None
    return sections


class _SectionedFSM:
    """Tracks position inside the structured think block.

    The expected stream is::

        <think>\\n
        SECTION_1: <free line>\\n
        SECTION_2: <free line>\\n
        ...
        </think>\\n\\n

    ``feed(text)`` advances state; ``allowed_prefix()`` returns the substring
    the next emission must begin with (empty string means free text up to a
    newline). ``done`` flips to ``True`` once ``</think>\\n\\n`` is consumed.
    """

    def __init__(self, sections: list[str]):
        self._template = (
            THINK_OPEN
            + "".join(f"{header}\x00\n" for header in sections)
            + THINK_CLOSE
        )
        self._pos = 0
        self.done = False

    def feed(self, text: str) -> None:
        for ch in text:
            self._consume(ch)
            if self.done:
                return

    def _consume(self, ch: str) -> None:
        if self._pos >= len(self._template):
            self.done = True
            return
        expected = self._template[self._pos]
        if expected == "\x00":
            if ch == "\n":
                self._pos += 2  # advance past placeholder + the newline
                if self._pos >= len(self._template):
                    self.done = True
            return  # any non-newline char stays in the free-line region
        if ch == expected:
            self._pos += 1
            if self._pos >= len(self._template):
                self.done = True

    def required_next(self) -> str | None:
        """Required literal continuation, or ``None`` if free-line region."""
        if self.done or self._pos >= len(self._template):
            return None
        ch = self._template[self._pos]
        if ch == "\x00":
            return None
        end = self._pos
        while end < len(self._template) and self._template[end] != "\x00":
            end += 1
        return self._template[self._pos:end]


class StructuredCoTLogitsProcessor:
    """Per-request logits processor enforcing structured CoT.

    Activates when ``<think>`` is observed in the decoded output and releases
    after ``</think>\\n\\n``. Outside that window, logits pass through
    unchanged.
    """

    def __init__(
        self,
        tokenizer: Any,
        grammar_text: str,
        bias_floor: float = -1e9,
    ) -> None:
        sections = parse_sectioned_grammar(grammar_text)
        if sections is None:
            raise ValueError(
                "structured_cot: only the bundled <think>/section/</think> "
                "grammar shape is supported by the built-in FSM. Use "
                "--structured-cot=<path> with a compatible grammar."
            )
        self.tokenizer = tokenizer
        self.sections = sections
        self.bias_floor = float(bias_floor)
        self._fsm = _SectionedFSM(sections)
        self._seen_text = ""
        self._think_seen = False
        self._last_token_count = 0
        self._vocab_strings = self._build_vocab_strings()
        self._eos_ids = self._collect_eos_ids()
        self._prefix_cache: dict[str, list[int]] = {}
        self._free_line_cache: list[int] | None = None

    def reset(self) -> None:
        self._fsm = _SectionedFSM(self.sections)
        self._seen_text = ""
        self._think_seen = False
        self._last_token_count = 0

    def _build_vocab_strings(self) -> list[str]:
        size = getattr(self.tokenizer, "vocab_size", None)
        if size is None and hasattr(self.tokenizer, "get_vocab"):
            size = len(self.tokenizer.get_vocab())
        if size is None:
            raise ValueError("tokenizer must expose vocab_size or get_vocab()")
        strings: list[str] = []
        for tid in range(int(size)):
            try:
                strings.append(self.tokenizer.decode([tid]))
            except Exception:
                strings.append("")
        return strings

    def _collect_eos_ids(self) -> set[int]:
        ids: set[int] = set()
        for attr in ("eos_token_id", "eos_token_ids"):
            value = getattr(self.tokenizer, attr, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                ids.update(int(v) for v in value if v is not None)
            else:
                ids.add(int(value))
        return ids

    def _decode_new(self, token_context: Any) -> str:
        token_ids = list(token_context) if not isinstance(token_context, list) else token_context
        if len(token_ids) <= self._last_token_count:
            return ""
        new_ids = token_ids[self._last_token_count :]
        self._last_token_count = len(token_ids)
        try:
            return self.tokenizer.decode(new_ids)
        except Exception:
            return ""

    def __call__(self, token_context: Any, logits: mx.array) -> mx.array:
        new_text = self._decode_new(token_context)
        if new_text:
            self._seen_text += new_text
            if self._think_seen:
                self._fsm.feed(new_text)

        if not self._think_seen:
            if THINK_OPEN in self._seen_text:
                self._think_seen = True
                idx = self._seen_text.index(THINK_OPEN)
                # Feed the FSM from the start of <think> so it advances past
                # the open marker into the first section header.
                self._fsm.feed(self._seen_text[idx:])
            else:
                return logits

        if self._fsm.done:
            return logits

        required = self._fsm.required_next()
        if required is None:
            allowed = self._tokens_allowed_in_free_line()
        else:
            allowed = self._tokens_matching_prefix(required)

        if not allowed:
            return logits  # nothing to enforce; let the model proceed

        vocab_size = logits.shape[-1]
        bias = np.full((vocab_size,), self.bias_floor, dtype=np.float32)
        bias[np.fromiter(allowed, dtype=np.int64)] = 0.0
        bias_arr = mx.array(bias).astype(logits.dtype)
        return logits + bias_arr

    def _tokens_matching_prefix(self, required: str) -> list[int]:
        if not required:
            return []
        cached = self._prefix_cache.get(required)
        if cached is not None:
            return cached
        out: list[int] = []
        for tid, text in enumerate(self._vocab_strings):
            if not text:
                continue
            if required.startswith(text) or text.startswith(required):
                out.append(tid)
        # Always allow EOS to terminate even when the prefix is unmet, so the
        # constraint can never deadlock a finished response.
        out.extend(self._eos_ids)
        self._prefix_cache[required] = out
        return out

    def _tokens_allowed_in_free_line(self) -> list[int]:
        if self._free_line_cache is not None:
            return self._free_line_cache
        out: list[int] = []
        for tid, text in enumerate(self._vocab_strings):
            if not text:
                continue
            if "\n" not in text or text == "\n" or text.endswith("\n"):
                out.append(tid)
        out.extend(self._eos_ids)
        self._free_line_cache = out
        return out


def make_structured_cot_factory(
    tokenizer: Any,
    grammar_path: str | Path | None,
) -> Callable[[], StructuredCoTLogitsProcessor]:
    """Return a zero-arg factory that mints fresh per-request processors."""
    grammar_text = load_grammar(grammar_path)

    def factory() -> StructuredCoTLogitsProcessor:
        return StructuredCoTLogitsProcessor(tokenizer, grammar_text)

    return factory
