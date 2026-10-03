# SPDX-License-Identifier: Apache-2.0
"""Drive qwen3_coder_xml output through the real streaming stack in tests.

Used by the #4037 / #4038 tests to compare streaming with non-streaming at
any delta granularity.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

from rapid_mlx.service.postprocessor import StreamingPostProcessor
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

# Single special tokens on the Qwen3-Coder tokenizer: they always arrive whole.
_SPECIAL = re.compile(r"(<tool_call>|</tool_call>)")


def deltas(text: str, size: int) -> list[str]:
    """Cut ``text`` into ``size``-character deltas, special tokens kept whole.

    ``size=0`` sends the whole text as ONE delta (batched / speculative
    decoding can deliver several special tokens at once).
    """
    if size == 0:
        return [text]
    out: list[str] = []
    for part in _SPECIAL.split(text):
        if _SPECIAL.fullmatch(part):
            out.append(part)
        else:
            out.extend(part[i : i + size] for i in range(0, len(part), size))
    return out


def non_stream(text: str, request: dict | None) -> tuple[list[tuple[str, str]], str]:
    """``extract_tool_calls`` as ``[(name, arguments)]`` plus content."""
    result = Qwen3CoderToolParser(None).extract_tool_calls(text, request=request)
    calls = [(c["name"], c["arguments"]) for c in result.tool_calls]
    return calls, result.content or ""


def _output(text: str, finished: bool) -> MagicMock:
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = None
    out.finish_reason = "stop" if finished else None
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


def stream(
    text: str, request: dict | None, size: int
) -> tuple[list[tuple[str, str]], str]:
    """Run ``text`` through ``StreamingPostProcessor`` (chunks + finalize)."""
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    cfg.tool_parser_instance = None
    pp = StreamingPostProcessor(cfg, request=request)
    pp.reset()
    events = []
    pieces = deltas(text, size)
    for i, piece in enumerate(pieces):
        events.extend(pp.process_chunk(_output(piece, i == len(pieces) - 1)))
    events.extend(pp.finalize())

    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    content: list[str] = []
    for event in events:
        if event.type in ("content", "finish") and event.content:
            content.append(event.content)
        for call in event.tool_calls or []:
            index = call.get("index", 0)
            fn = call.get("function") or {}
            if fn.get("name"):
                names[index] = fn["name"]
            arguments[index] = arguments.get(index, "") + (fn.get("arguments") or "")
    calls = [(names[i], arguments.get(i, "")) for i in sorted(names)]
    return calls, "".join(content)
