"""User stop strings follow the same prompt-derived state as PromptOutputOwner."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.config import reset_config
from rapid_mlx.reasoning import get_parser
from rapid_mlx.reasoning.think_stop import find_stop_in_answer
from rapid_mlx.service.helpers import reasoning_stop_scope_kwargs

FLINT_TEMPLATE = (
    Path(__file__).parents[1] / "fixtures" / "flint_chat_template.jinja"
).read_text()
# LFM2.5 opens reasoning in its generation prompt.
LFM_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>"
    "{% endif %}"
)


def _scope(template, model, thinking, tools=None):
    cfg = reset_config()
    cfg.reasoning_parser = get_parser("qwen3")()
    cfg.model_path = model
    engine = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=template))
    request = SimpleNamespace(
        stop=["Observation:"],
        chat_template_kwargs={"enable_thinking": thinking},
        tools=tools,
    )
    return reasoning_stop_scope_kwargs(engine, request)["reasoning_stop_scope"]


@pytest.mark.parametrize("thinking", [False, None, True])
def test_lfm_stop_waits_for_prompt_opened_reasoning_to_close(thinking):
    scope = _scope(LFM_TEMPLATE, "LiquidAI/LFM2.5-24B-A2B-MLX-4bit", thinking)
    assert scope.starts_in_reasoning
    raw = "I should not write Observation: here</think>Answer. Observation: x"
    hit = find_stop_in_answer(raw, ["Observation:"], scope)
    assert hit is not None and raw[: hit[1]].endswith("Answer. ")


@pytest.mark.parametrize("thinking", [False, None, True])
def test_flint_stop_applies_to_plain_answers(thinking):
    scope = _scope(FLINT_TEMPLATE, "RecursiveSI/flint-int4", thinking)
    assert not scope.starts_in_reasoning
    raw = "Step one. Observation: done. More text after stop."
    hit = find_stop_in_answer(raw, ["Observation:"], scope, terminal=True)
    assert hit is not None and raw[: hit[1]] == "Step one. "


def test_flint_stop_waits_for_model_written_reasoning():
    scope = _scope(FLINT_TEMPLATE, "RecursiveSI/flint-int4", None)
    raw = "<think>Observation: private</think>Answer. Observation: x"
    hit = find_stop_in_answer(raw, ["Observation:"], scope, terminal=True)
    assert hit is not None and raw[: hit[1]].endswith("Answer. ")


@pytest.mark.parametrize(
    "owned,contract,suffix,expected",
    [
        (True, "flint", "<think>", True),
        (True, "flint", "", False),
        (True, "chatml", "<think>", True),
        (True, "chatml", "<think>x</think>", False),
        (False, "chatml", "", True),
    ],
)
def test_engine_scopes_stop_by_rendered_suffix(owned, contract, suffix, expected):
    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.prompt_parse_context import PromptParseContext
    from rapid_mlx.reasoning.think_stop import ReasoningStopScope

    engine = object.__new__(BatchedEngine)
    # Non-owned families keep the request-time scope unchanged.
    scope = ReasoningStopScope("<think>", "</think>", True, prompt_owned=owned)
    kwargs = {"reasoning_stop_scope": scope}
    engine._align_owned_stop_scope(
        kwargs, PromptParseContext(suffix, None, None, False, contract)
    )
    assert kwargs["reasoning_stop_scope"].starts_in_reasoning is expected


def test_local_path_keeps_lfm_ownership_from_served_name():
    cfg = reset_config()
    cfg.reasoning_parser = get_parser("qwen3")()
    cfg.model_path = "/models/local-snapshot"
    cfg.model_name = "lfm2.5-24b-a2b-4bit"
    engine = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=LFM_TEMPLATE))
    request = SimpleNamespace(
        stop=["Observation:"],
        chat_template_kwargs={"enable_thinking": False},
        tools=None,
    )
    scope = reasoning_stop_scope_kwargs(engine, request)["reasoning_stop_scope"]
    assert scope.prompt_owned and scope.starts_in_reasoning
