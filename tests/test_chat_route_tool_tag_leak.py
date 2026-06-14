# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the chat-route tool-tag leak bug.

Bug: when the tool parser successfully extracted tool_calls from output that
contained an unclosed `<tool_call>` block (model omitted the closing tag), it
returned `content=None` (i.e. fully stripped). routes/chat.py then fell back
to the RAW output for the reasoning parser via `cleaned_text or output.text`,
and the reasoning parser strips `<think>` but not `<tool_call>` — so the
opening tag + JSON survived all the way to user-facing `content`.

Detected by `rapid-mlx agents hermes --test` (no_tool_leak + stress_no_leak)
even on a strong 27B model, ruling out model weakness. Fix: when tool_calls
were extracted, trust the tool parser's cleaned_text and only run the
reasoning parser to recover reasoning_text from the raw output.
"""

from types import SimpleNamespace

from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from vllm_mlx.service.helpers import _finalize_content_and_reasoning
from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser


def _make_request_stub() -> SimpleNamespace:
    """Match the shape of ChatCompletionRequest the parser may inspect.

    HermesToolParser may walk request.tools / request.model on
    schema-driven type-conversion paths; passing None lets a future
    parser change pass `request=None` silently while users would crash
    in production. SimpleNamespace gives the parser the same attribute
    access surface it sees in the live route.
    """
    return SimpleNamespace(
        model="test-model",
        tools=None,
        tool_choice=None,
        messages=[],
    )


def _drive_chat_route_pipeline(
    raw_output: str,
) -> tuple[str | None, list, str | None]:
    """Drive the REAL post-parse helper from routes/chat.py.

    Wraps the production tool parser + reasoning parser around the
    extracted ``_finalize_content_and_reasoning`` helper so the
    regression suite tests the exact orchestration the route runs —
    no parallel reimplementation that can silently drift from prod.

    Returns (final_content, tool_calls, reasoning_text). final_content
    is what the user receives in `choices[0].message.content`.
    """
    parser = HermesToolParser(tokenizer=None)
    parser.reset()
    result = parser.extract_tool_calls(raw_output, request=_make_request_stub())
    cleaned_text = result.content or ""
    tool_calls = list(result.tool_calls) if result.tools_called else []

    reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw_output,
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        reasoning_parser=reasoning_parser,
    )

    final_content = cleaned_text if cleaned_text else None
    return final_content, tool_calls, reasoning_text


_LEAK_MARKERS = ("<tool_call>", "<function=", "<|im_end|>", "<|tool_call|>")


def _assert_no_leak(content: str | None) -> None:
    if not content:
        return
    leaks = [m for m in _LEAK_MARKERS if m in content]
    assert not leaks, (
        f"Tool tags leaked into user content: {leaks!r} (content={content!r})"
    )


class TestToolTagLeakRegression:
    """The specific cases that fired in the agent test suite."""

    def test_unclosed_tool_call_does_not_leak(self):
        # Real Qwopus 27B output: model omitted </tool_call> closing tag.
        raw = '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo test"}}'
        content, tool_calls, reasoning = _drive_chat_route_pipeline(raw)
        assert tool_calls and tool_calls[0]["name"] == "terminal"
        _assert_no_leak(content)
        # Qwen3's implicit-think heuristic could otherwise reroute a bare
        # tool_call into reasoning_content where it would also be visible
        # to the user. Guard both sinks.
        _assert_no_leak(reasoning)

    def test_unclosed_tool_call_with_thinking_does_not_leak(self):
        # The exact shape that fires in stress_no_leak — reasoning + tool_call.
        raw = (
            "<think>The user wants me to use the terminal tool.</think>\n"
            '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo test"}}'
        )
        content, tool_calls, reasoning = _drive_chat_route_pipeline(raw)
        assert tool_calls and tool_calls[0]["name"] == "terminal"
        assert reasoning and "terminal tool" in reasoning
        _assert_no_leak(content)
        _assert_no_leak(reasoning)

    def test_well_formed_tool_call_still_passes(self):
        # Control: properly closed tag. Should still extract cleanly.
        raw = (
            "<think>I should use the terminal.</think>\n"
            '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo test"}}\n'
            "</tool_call>"
        )
        content, tool_calls, reasoning = _drive_chat_route_pipeline(raw)
        assert tool_calls and tool_calls[0]["name"] == "terminal"
        assert reasoning and "terminal" in reasoning
        _assert_no_leak(content)

    def test_no_tool_call_path_preserves_content(self):
        # When no tool_calls fire, plain text content should pass through
        # unchanged (regression guard for the else branch). Note: Hermes
        # parser strips <think> tags itself before the reasoning parser
        # would see them, so reasoning_text from the route's reasoning
        # parser call is expected to be None in this branch — that is
        # pre-existing behavior unrelated to this fix.
        raw = "<think>Just thinking.</think>The actual answer is 42."
        content, tool_calls, _ = _drive_chat_route_pipeline(raw)
        assert not tool_calls
        assert content and "answer is 42" in content
        _assert_no_leak(content)

    def test_parser_finds_nothing_preserves_existing_cleaned_text(self):
        # Regression for the v0.6.64 gpt-oss-20b-mxfp4-q8 empty-TextBlock bug:
        # ``engine.generate()`` runs ``clean_output_text`` on harmony
        # output, which strips channel markup and returns just the
        # final-channel content ("4"). The non-streaming route then
        # called ``_finalize_content_and_reasoning`` with this
        # pre-cleaned string, and the HarmonyReasoningParser — looking
        # for ``<|channel|>analysis``/``<|channel|>final`` markers —
        # found none and returned ``(None, None)``. The helper then
        # silently overwrote the perfectly valid ``cleaned_text="4"``
        # with ``None``, so anthropic_sdk / langchain / pydantic_ai
        # received empty TextBlocks for fully-formed answers.
        #
        # Fix: when the parser returns ``(None, None)``, keep the
        # original cleaned_text. Validate here with the harmony
        # reasoning parser because that is the parser that exhibits
        # the pattern, but the guard applies to any parser that
        # legitimately reports "I found no markers I understand."
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

        cleaned, reasoning = _finalize_content_and_reasoning(
            raw_text="4",
            cleaned_text="4",
            tool_calls=None,
            reasoning_parser=HarmonyReasoningParser(),
        )
        assert cleaned == "4"
        assert reasoning is None

    def test_parser_returns_reasoning_only_preserves_cleaned_text(self):
        # DeepSeek review on PR #436 flagged that the initial
        # ``(None, None)``-only guard still clobbered ``cleaned_text``
        # whenever the parser returned ``(reasoning, None)`` — same
        # regression class by a different route. Concrete case: a
        # ``<think>thinking</think>`` payload with no actual content
        # after the closing tag. The Qwen3 reasoning parser pulls
        # ``"thinking"`` out as reasoning and returns ``content=None``;
        # the helper must NOT overwrite the caller's ``cleaned_text``
        # with that ``None``. (Downstream ``strip_thinking_tags`` will
        # collapse this to an empty content for the wire response,
        # which is the right outcome — but the helper's job is to
        # respect the contract "only update cleaned_text when the
        # parser explicitly produced new content.")
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        raw = "<think>just thinking, no answer</think>"
        cleaned, reasoning = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=None,
            reasoning_parser=Qwen3ReasoningParser(tokenizer=None),
        )
        assert cleaned == raw, (
            f"expected original cleaned_text preserved (sentinel for "
            f"downstream strip_thinking_tags), got {cleaned!r}"
        )
        assert reasoning is not None and "thinking" in reasoning


class TestBareThinkingProcessLeakRegression:
    """Regression for issue #570.

    The Qwen3 family chat template injects ``<think>\\n`` after the
    assistant generation marker when ``enable_thinking=True``. The model
    is supposed to emit its chain-of-thought followed by ``</think>`` and
    then the user-facing answer. In practice the model occasionally
    restates the channel boundary inline with a bare-text preamble
    (``Here's a thinking process:\\n\\n1. **Analyze...``); when the
    request also runs out of ``max_tokens`` before the model reaches
    ``</think>``, neither tag appears anywhere in the output.

    Previously the reasoning parser's "no end token, no start token"
    branch returned ``(None, model_output)`` — routing the whole
    chain-of-thought into the user-facing ``content`` field while
    ``reasoning_content`` stayed empty. Any OpenAI-compatible client
    consuming ``/v1/chat/completions`` saw reasoning leaking into the
    answer.

    Fix: extend ``Qwen3ReasoningParser.extract_reasoning`` (and the
    streaming finalizer) to recognise bare-text "thinking process"
    preambles and route them to ``reasoning_content`` while clearing
    the cleaned content. Match conservatively at the very start of the
    output so a normal answer that merely mentions "let me think" mid-
    response is not reclassified.
    """

    # Real text captured from a failing curl against ``qwen3.6-35b-4bit``.
    _LEAKED_PREAMBLE = (
        "Here's a thinking process:\n\n"
        "1.  **Analyze User Input:**\n"
        "   - **Route:** Seattle to San Diego\n"
        "   - **Duration:** 7 days\n\n"
        "2.  **Evaluate Each Option (Food Scene Reputation):**\n"
        "   - **Portland, OR:** World-renowned food scene. Innovative, diverse.\n"
        "   - **San Francisco, CA:** Iconic global food destination."
    )

    def test_bare_thinking_preamble_routes_to_reasoning_not_content(self):
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        cleaned, reasoning = _finalize_content_and_reasoning(
            raw_text=self._LEAKED_PREAMBLE,
            cleaned_text=self._LEAKED_PREAMBLE,
            tool_calls=None,
            reasoning_parser=Qwen3ReasoningParser(tokenizer=None),
        )
        # ``reasoning_content`` must contain the chain-of-thought.
        assert reasoning is not None
        assert "thinking process" in reasoning
        assert "Portland" in reasoning
        # ``content`` must be cleared so the leak does not reach the
        # user-facing ``choices[0].message.content`` field.
        assert not cleaned, (
            f"chain-of-thought leaked into user-facing content: {cleaned!r}"
        )

    def test_bare_thinking_preamble_via_full_chat_route_pipeline(self):
        # Drive the same pipeline the OpenAI ``/v1/chat/completions``
        # route uses end-to-end (tool parser + reasoning parser +
        # finalize helper). ``final_content`` here is what the client
        # sees in ``choices[0].message.content``.
        content, tool_calls, reasoning = _drive_chat_route_pipeline(
            self._LEAKED_PREAMBLE
        )
        assert not tool_calls
        assert reasoning is not None and "thinking process" in reasoning
        assert content is None, (
            f"expected empty content (full output was reasoning), got {content!r}"
        )

    def test_normal_answer_with_let_me_think_mid_sentence_not_reclassified(self):
        # Mid-sentence mentions of "let me think" must not trigger the
        # bare-text fallback — the regex anchors to the very start of
        # the output.
        answer = (
            "Portland has the best food scene of those options. Many people think "
            "it's world-class — let me think of an example... Pok Pok was iconic."
        )
        cleaned, reasoning = _finalize_content_and_reasoning(
            raw_text=answer,
            cleaned_text=answer,
            tool_calls=None,
            reasoning_parser=__import__(
                "vllm_mlx.reasoning.qwen3_parser", fromlist=["Qwen3ReasoningParser"]
            ).Qwen3ReasoningParser(tokenizer=None),
        )
        assert reasoning is None
        assert cleaned == answer

    def test_bare_thinking_preamble_with_tool_call_does_not_leak_markup(self):
        # Codex r2 BLOCKING on PR #572: when the model embeds a tool
        # call inside what looks like a thinking preamble, the bare-text
        # fallback would otherwise echo the raw output (including
        # ``<tool_call>`` markup) into ``reasoning_content`` because
        # ``_finalize_content_and_reasoning`` calls
        # ``extract_reasoning(raw_text)`` on the unstripped raw output
        # when ``tool_calls`` is non-empty.
        raw = (
            "Here's a thinking process:\n\n"
            "I need to call the weather tool.\n"
            '<tool_call>\n{"name": "weather", "arguments": '
            '{"city": "Seattle"}}\n</tool_call>'
        )
        content, tool_calls, reasoning = _drive_chat_route_pipeline(raw)
        # Tool parser succeeded.
        assert tool_calls and tool_calls[0]["name"] == "weather"
        # Tool markup must not leak via either sink.
        _assert_no_leak(content)
        _assert_no_leak(reasoning)
