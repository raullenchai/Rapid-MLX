# SPDX-License-Identifier: Apache-2.0
"""Tests for structured CoT logits processor."""

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_mlx.api.structured_cot import (
    DEFAULT_GRAMMAR_PATH,
    LCB_PLAN_GRAMMAR_PATH,
    StructuredCoTLogitsProcessor,
    _SectionedFSM,
    load_grammar,
    parse_sectioned_grammar,
)


class FakeTokenizer:
    """Minimal byte-level fake tokenizer.

    Each character (plus the literal multi-char chunks we register) maps to a
    distinct token id. ``decode([id])`` returns the original chunk.
    """

    def __init__(self, chunks: list[str]) -> None:
        seen: dict[str, int] = {}
        for chunk in chunks:
            if chunk in seen:
                continue
            seen[chunk] = len(seen)
        self._chunks = list(seen.keys())
        self._ids = seen
        self.vocab_size = len(self._chunks)
        self.eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        i = 0
        out: list[int] = []
        max_chunk = max((len(c) for c in self._chunks), default=1)
        while i < len(text):
            for length in range(min(len(text) - i, max_chunk), 0, -1):
                candidate = text[i : i + length]
                if candidate in self._ids:
                    out.append(self._ids[candidate])
                    i += length
                    break
            else:
                raise KeyError(f"unknown char: {text[i]!r}")
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self._chunks[i] for i in ids)


def test_parse_sectioned_grammar_default():
    text = load_grammar(DEFAULT_GRAMMAR_PATH)
    sections = parse_sectioned_grammar(text)
    assert sections == ["GOAL: ", "APPROACH: ", "EDGE: "]


def test_parse_sectioned_grammar_lcb_plan():
    text = load_grammar(LCB_PLAN_GRAMMAR_PATH)
    sections = parse_sectioned_grammar(text)
    assert sections == ["GOAL: ", "STATE: ", "ALGO: ", "EDGE: ", "VERIFY: "]


def test_sectioned_fsm_advances_through_template():
    fsm = _SectionedFSM(["GOAL: ", "APPROACH: ", "EDGE: "])
    fsm.feed("<think>\n")
    assert fsm.required_next() == "GOAL: "

    fsm.feed("GOAL: ")
    # Now in the free-line region, awaiting newline.
    assert fsm.required_next() is None
    fsm.feed("solve it\n")
    assert fsm.required_next() == "APPROACH: "

    fsm.feed("APPROACH: think harder\n")
    assert fsm.required_next() == "EDGE: "

    fsm.feed("EDGE: empty input\n")
    assert fsm.required_next() == "</think>\n\n"

    fsm.feed("</think>\n\n")
    assert fsm.done is True


def test_processor_passes_through_before_think():
    chunks = ["<think>\n", "GOAL: ", "hi", "\n", "</think>\n\n", "x"]
    tokenizer = FakeTokenizer(chunks)
    grammar = load_grammar(DEFAULT_GRAMMAR_PATH)
    processor = StructuredCoTLogitsProcessor(tokenizer, grammar)

    logits = mx.zeros((1, tokenizer.vocab_size))
    out = processor([], logits)
    # Without <think> seen yet, processor must not bias logits.
    assert mx.array_equal(out, logits)


def test_processor_constrains_after_think_open():
    chunks = ["<think>\n", "GOAL: ", "APPROACH: ", "hi", "\n", "</think>\n\n"]
    tokenizer = FakeTokenizer(chunks)
    grammar = load_grammar(DEFAULT_GRAMMAR_PATH)
    processor = StructuredCoTLogitsProcessor(tokenizer, grammar)

    think_id = tokenizer.encode("<think>\n")[0]
    goal_id = tokenizer.encode("GOAL: ")[0]
    approach_id = tokenizer.encode("APPROACH: ")[0]

    logits = mx.zeros((1, tokenizer.vocab_size))
    out = processor([think_id], logits)
    # GOAL: must be allowed (~0 bias) and APPROACH: must not (very negative).
    assert float(out[0, goal_id]) > -1.0
    assert float(out[0, approach_id]) < -1e6
