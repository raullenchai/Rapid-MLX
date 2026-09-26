# SPDX-License-Identifier: Apache-2.0
"""User ``stop`` strings match the answer of a ``<think>`` model, not its
reasoning.

The ``<think>`` counterpart of #1049 (harmony final-channel scoping). A
reasoning model that writes a client stop string while it thinks used to
end the request inside the ``<think>`` block, so ``content`` came back
empty: ``stop=["10"]`` on "count to 10" fires while the model counts in its
reasoning, and an agent's ``</execute_bash>`` stop fires while the model
plans which action to take.

This module runs without MLX:

1. The pure helpers in ``rapid_mlx.reasoning.think_stop``.
2. Routes: chat / Anthropic attach the scope when a ``<think>`` reasoning
   parser is configured and the request sets ``stop``.

The scheduler integration lives in ``test_think_stop_scheduler.py`` (MLX
lane), and engine dispatch of the scope to both lanes in
``test_lane_parity_contract.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rapid_mlx.reasoning.think_stop import (
    ReasoningStopScope,
    answer_start,
    build_reasoning_stop_scope,
    find_stop_in_answer,
)

PROMPT_OPENED = ReasoningStopScope("<think>", "</think>", starts_in_reasoning=True)
MODEL_OPENED = ReasoningStopScope("<think>", "</think>", starts_in_reasoning=False)

# Qwen3.5-style: the template opens <think> in the prompt, the model counts
# in its reasoning, closes it, and answers.
COUNTING = "I will count 1 to 10 and stop at 10.\n</think>\n\n1\n2\n9\n10\n11"
COUNTING_ANSWER_BEFORE_STOP = (
    "I will count 1 to 10 and stop at 10.\n</think>\n\n1\n2\n9\n"
)


# ---------------------------------------------------------------------------
# Layer 1: helpers
# ---------------------------------------------------------------------------


def test_prompt_opened_reasoning_is_stop_agnostic():
    text = "The action ends with </execute_bash>, so I will"
    assert answer_start(text, PROMPT_OPENED) is None
    assert find_stop_in_answer(text, ["</execute_bash>"], PROMPT_OPENED) is None


def test_stop_matches_the_answer_occurrence_not_the_reasoning_one():
    match = find_stop_in_answer(COUNTING, ["10"], PROMPT_OPENED)
    assert match is not None
    stop_str, idx = match
    assert stop_str == "10"
    assert COUNTING[:idx] == COUNTING_ANSWER_BEFORE_STOP


def test_model_opened_reasoning_is_stop_agnostic_until_close():
    reasoning = "<think>\nmention STOP while thinking"
    assert find_stop_in_answer(reasoning, ["STOP"], MODEL_OPENED) is None
    text = reasoning + "</think>answer STOP tail"
    stop_str, idx = find_stop_in_answer(text, ["STOP"], MODEL_OPENED)
    assert text[:idx].endswith("</think>answer ")


@pytest.mark.parametrize("partial", ["", "  ", "<", "<thi", "\n<think"])
def test_partial_opener_waits(partial):
    """Output that may still become ``<think>`` is not classified yet."""
    assert answer_start(partial, MODEL_OPENED) is None


def test_no_reasoning_keeps_raw_stream_semantics():
    """A thinking-off answer (no markers) matches exactly as before,
    including the first-listed-stop-wins iteration order."""
    text = "B then A"
    assert answer_start(text, MODEL_OPENED) == 0
    assert find_stop_in_answer(text, ["A", "B"], MODEL_OPENED) == ("A", 7)


def test_stop_spanning_the_close_marker_is_not_an_answer_stop():
    text = "ends with 1</think>0 ok"
    assert find_stop_in_answer(text, ["10"], PROMPT_OPENED) is None


def test_separator_after_close_is_not_answer_text():
    """The template's whitespace after ``</think>`` is stripped from
    ``content`` by the parser, so a ``\\n`` stop applies inside the answer."""
    text = "why</think>\n\nA\nB"
    stop_str, idx = find_stop_in_answer(text, ["\n"], PROMPT_OPENED)
    assert text[:idx] == "why</think>\n\nA"
    # Only the separator so far: nothing to match yet.
    assert find_stop_in_answer("why</think>\n\n", ["\n"], PROMPT_OPENED) is None


def test_empty_stop_strings_are_ignored():
    assert find_stop_in_answer("x</think>a", ["", "a"], PROMPT_OPENED) == ("a", 9)


def test_scope_built_only_for_think_tag_parsers():
    from rapid_mlx.reasoning import get_parser

    for name in ("qwen3", "deepseek_r1"):
        scope = build_reasoning_stop_scope(get_parser(name)(), starts_in_reasoning=True)
        assert scope == PROMPT_OPENED
    assert build_reasoning_stop_scope(None, starts_in_reasoning=True) is None
    # Harmony models are scoped by the scheduler's family gate (#1049).
    harmony = get_parser("gpt_oss")()
    assert build_reasoning_stop_scope(harmony, starts_in_reasoning=True) is None


# ---------------------------------------------------------------------------
# Layer 2: routes attach the scope
# ---------------------------------------------------------------------------

_THINK_TEMPLATE = (
    "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
)


class _RecordingEngine:
    preserve_native_tool_format = False
    supports_guided_generation = False
    is_mllm = False

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(chat_template=_THINK_TEMPLATE)
        self.calls: list[dict[str, Any]] = []

    def build_prompt(self, messages, **_kwargs):
        return "PROMPT"

    @staticmethod
    def _output():
        from rapid_mlx.engine.base import GenerationOutput

        return GenerationOutput(
            text="reasoning</think>answer",
            new_text="reasoning</think>answer",
            tokens=[1],
            prompt_tokens=2,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    async def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self._output()

    async def stream_chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        yield self._output()


def _client(surface: str, engine: _RecordingEngine, *, parser_name: str | None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.reasoning import get_parser

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "think-stop-test"
    cfg.model_path = "think-stop-test"
    cfg.model_registry = None
    cfg.reasoning_parser = get_parser(parser_name)() if parser_name else None
    cfg.reasoning_parser_name = parser_name
    cfg.tool_parser = None
    cfg.tool_call_parser = None
    if surface == "chat":
        from rapid_mlx.routes.chat import router
    else:
        from rapid_mlx.routes.anthropic import router
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    return TestClient(app)


def _post(
    surface: str,
    *,
    parser_name: str | None,
    stop: list[str] | None,
    stream: bool,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    """Send one request and return the kwargs the engine received."""
    from rapid_mlx.config import reset_config

    engine = _RecordingEngine()
    client = _client(surface, engine, parser_name=parser_name)
    body: dict[str, Any] = {
        "model": "think-stop-test",
        "messages": [{"role": "user", "content": "count to 10"}],
        "max_tokens": 64,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if stop is not None:
        body["stop" if surface == "chat" else "stop_sequences"] = stop
    path = "/v1/chat/completions" if surface == "chat" else "/v1/messages"
    try:
        if stream:
            with client.stream("POST", path, json=body) as response:
                assert response.status_code == 200, response.text
                list(response.iter_lines())
        else:
            response = client.post(path, json=body)
            assert response.status_code == 200, response.text
        return engine.calls[-1]
    finally:
        reset_config()


_SURFACES = pytest.mark.parametrize("surface", ["chat", "anthropic"])
_STREAM = pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])


@_SURFACES
@_STREAM
def test_routes_attach_the_scope_for_a_think_parser(surface, stream):
    kwargs = _post(surface, parser_name="qwen3", stop=["10"], stream=stream)
    assert kwargs["reasoning_stop_scope"] == PROMPT_OPENED


@_SURFACES
def test_thinking_off_scope_does_not_start_in_reasoning(surface):
    """With thinking disabled the template does not open ``<think>``, so an
    answer without markers is matched from its first character, as before."""
    kwargs = _post(
        surface, parser_name="qwen3", stop=["10"], stream=False, enable_thinking=False
    )
    assert kwargs["reasoning_stop_scope"] == MODEL_OPENED


@_SURFACES
@pytest.mark.parametrize(
    ("parser_name", "stop"),
    [("qwen3", None), (None, ["10"]), ("gpt_oss", ["10"])],
    ids=["no-stop", "no-reasoning-parser", "harmony-parser"],
)
def test_routes_leave_other_requests_unscoped(surface, parser_name, stop):
    kwargs = _post(surface, parser_name=parser_name, stop=stop, stream=False)
    assert "reasoning_stop_scope" not in kwargs


def test_multimodel_scope_uses_the_selected_entry_not_the_global_parser():
    """A reasoning sidecar gets its parser even when the default is plain."""
    from rapid_mlx.config import reset_config
    from rapid_mlx.service.helpers import reasoning_stop_scope_kwargs

    engine = _RecordingEngine()
    entry = SimpleNamespace(engine=engine, reasoning_parser="qwen3")
    registry = SimpleNamespace(get_entry=lambda _model: entry)
    request = SimpleNamespace(
        model="reasoning-sidecar",
        stop=["10"],
        tools=None,
        enable_thinking=None,
        chat_template_kwargs={"enable_thinking": True},
    )
    cfg = reset_config()
    cfg.model_registry = registry
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    try:
        assert reasoning_stop_scope_kwargs(engine, request) == {
            "reasoning_stop_scope": PROMPT_OPENED
        }
    finally:
        reset_config()


def test_multimodel_plain_entry_does_not_borrow_the_global_parser():
    """A plain sidecar keeps raw stops when the default model reasons."""
    from rapid_mlx.config import reset_config
    from rapid_mlx.reasoning import get_parser
    from rapid_mlx.service.helpers import reasoning_stop_scope_kwargs

    engine = _RecordingEngine()
    entry = SimpleNamespace(engine=engine, reasoning_parser=None)
    registry = SimpleNamespace(get_entry=lambda _model: entry)
    request = SimpleNamespace(
        model="plain-sidecar",
        stop=["10"],
        tools=None,
        enable_thinking=None,
        chat_template_kwargs={"enable_thinking": True},
    )
    cfg = reset_config()
    cfg.model_registry = registry
    cfg.reasoning_parser = get_parser("qwen3")()
    cfg.reasoning_parser_name = "qwen3"
    try:
        assert reasoning_stop_scope_kwargs(engine, request) == {}
    finally:
        reset_config()


def test_multimodel_scope_fails_closed_on_registry_replacement():
    """Do not attach a replacement entry's parser to a retired engine."""
    from rapid_mlx.config import reset_config
    from rapid_mlx.service.helpers import reasoning_stop_scope_kwargs

    captured_engine = _RecordingEngine()
    replacement = _RecordingEngine()
    entry = SimpleNamespace(engine=replacement, reasoning_parser="qwen3")
    registry = SimpleNamespace(get_entry=lambda _model: entry)
    request = SimpleNamespace(
        model="replaced-sidecar",
        stop=["10"],
        tools=None,
        enable_thinking=None,
        chat_template_kwargs={"enable_thinking": True},
    )
    cfg = reset_config()
    cfg.model_registry = registry
    try:
        assert reasoning_stop_scope_kwargs(captured_engine, request) == {}
    finally:
        reset_config()
