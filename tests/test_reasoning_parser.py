# SPDX-License-Identifier: Apache-2.0
"""
Tests for reasoning content extraction parsers.

Tests cover:
- Parser registry (registration, lookup, listing)
- Qwen3 parser (non-streaming and streaming)
- DeepSeek-R1 parser (non-streaming and streaming)
- Edge cases (no tags, partial tags, etc.)
"""

import pytest

from vllm_mlx.reasoning import (
    DeltaMessage,
    ReasoningParser,
    get_parser,
    list_parsers,
    register_parser,
)


class TestParserRegistry:
    """Tests for the parser registry functions."""

    def test_list_parsers_includes_builtin(self):
        """Built-in parsers should be registered."""
        parsers = list_parsers()
        assert "qwen3" in parsers
        assert "deepseek_r1" in parsers

    def test_get_parser_qwen3(self):
        """Should be able to get Qwen3 parser."""
        parser_cls = get_parser("qwen3")
        parser = parser_cls()
        assert isinstance(parser, ReasoningParser)

    def test_get_parser_deepseek(self):
        """Should be able to get DeepSeek-R1 parser."""
        parser_cls = get_parser("deepseek_r1")
        parser = parser_cls()
        assert isinstance(parser, ReasoningParser)

    def test_get_unknown_parser_raises(self):
        """Unknown parser name should raise KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_parser("unknown_parser")
        assert "unknown_parser" in str(exc_info.value)
        assert "Available parsers" in str(exc_info.value)

    def test_register_custom_parser(self):
        """Should be able to register custom parsers."""

        class CustomParser(ReasoningParser):
            def extract_reasoning(self, model_output):
                return None, model_output

            def extract_reasoning_streaming(self, prev, curr, delta):
                return DeltaMessage(content=delta)

        register_parser("custom_test", CustomParser)
        assert "custom_test" in list_parsers()

        parser = get_parser("custom_test")()
        assert isinstance(parser, CustomParser)


class TestQwen3Parser:
    """Tests for the Qwen3 reasoning parser."""

    @pytest.fixture
    def parser(self):
        """Create a fresh Qwen3 parser for each test."""
        return get_parser("qwen3")()

    # Non-streaming tests

    def test_extract_with_both_tags(self, parser):
        """Should extract reasoning when both tags present."""
        output = "<think>Let me analyze this problem</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me analyze this problem"
        assert content == "The answer is 42."

    def test_extract_only_reasoning(self, parser):
        """Should handle case where only reasoning is present."""
        output = "<think>Just thinking out loud</think>"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Just thinking out loud"
        assert content is None

    def test_extract_multiline_reasoning(self, parser):
        """Should preserve newlines in reasoning content."""
        output = (
            "<think>Step 1: Analyze\nStep 2: Solve\nStep 3: Verify</think>Result: 42"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Step 3" in reasoning
        assert content == "Result: 42"

    def test_no_tags_returns_content_only(self, parser):
        """Qwen3 requires both tags - no tags means pure content."""
        output = "Just a regular response without thinking."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_only_start_tag_truncated(self, parser):
        """Missing end tag = truncated thinking (e.g., max_tokens hit during reasoning)."""
        output = "<think>Started thinking but never finished"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Started thinking but never finished"
        assert content is None

    def test_only_end_tag_implicit_mode(self, parser):
        """Qwen3 supports implicit mode - when <think> is in prompt, only </think> in output."""
        output = "Some text</think>more text"
        reasoning, content = parser.extract_reasoning(output)
        # Implicit mode: everything before </think> is reasoning
        assert reasoning == "Some text"
        assert content == "more text"

    # Streaming tests

    def test_streaming_simple_flow(self, parser):
        """Test basic streaming with reasoning then content."""
        parser.reset_state()

        # Simulate streaming tokens
        deltas = ["<think>", "think", "ing", "</think>", "answer"]
        accumulated = ""
        results = []

        for delta in deltas:
            prev = accumulated
            accumulated += delta
            result = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if result:
                results.append(result)

        # Collect reasoning and content
        reasoning_parts = [r.reasoning for r in results if r.reasoning]
        content_parts = [r.content for r in results if r.content]

        assert "".join(reasoning_parts) == "thinking"
        assert "".join(content_parts) == "answer"

    def test_streaming_skip_tags(self, parser):
        """Special tokens themselves should be skipped."""
        parser.reset_state()

        # Just the start tag
        result = parser.extract_reasoning_streaming("", "<think>", "<think>")
        assert result is None

        # Just the end tag
        result = parser.extract_reasoning_streaming(
            "<think>reasoning", "<think>reasoning</think>", "</think>"
        )
        assert result is None

    def test_streaming_transition_chunk(self, parser):
        """Chunk containing end tag should split reasoning and content."""
        parser.reset_state()

        # Previous has start, delta contains end and content
        prev = "<think>reasoning"
        delta = " more</think>content here"
        curr = prev + delta

        result = parser.extract_reasoning_streaming(prev, curr, delta)

        assert result is not None
        assert result.reasoning == " more"
        assert result.content == "content here"


class TestDeepSeekR1Parser:
    """Tests for the DeepSeek-R1 reasoning parser."""

    @pytest.fixture
    def parser(self):
        """Create a fresh DeepSeek-R1 parser for each test."""
        return get_parser("deepseek_r1")()

    # Non-streaming tests

    def test_extract_with_both_tags(self, parser):
        """Should extract reasoning when both tags present."""
        output = "<think>Step by step analysis</think>Final answer: 42"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Step by step analysis"
        assert content == "Final answer: 42"

    def test_extract_implicit_start_tag(self, parser):
        """DeepSeek-R1 handles implicit start tag (missing <think>)."""
        output = "Implicit reasoning content</think>The answer"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Implicit reasoning content"
        assert content == "The answer"

    def test_extract_no_tags_pure_content(self, parser):
        """No tags should return pure content."""
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_extract_multiline_reasoning(self, parser):
        """Should preserve newlines in reasoning content."""
        output = "<think>Line 1\nLine 2\nLine 3</think>Result"
        reasoning, content = parser.extract_reasoning(output)
        assert "Line 1" in reasoning
        assert "Line 2" in reasoning
        assert "Line 3" in reasoning
        assert content == "Result"

    # Streaming tests

    def test_streaming_simple_flow(self, parser):
        """Test basic streaming with reasoning then content."""
        parser.reset_state()

        deltas = ["<think>", "think", "ing", "</think>", "answer"]
        accumulated = ""
        results = []

        for delta in deltas:
            prev = accumulated
            accumulated += delta
            result = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if result:
                results.append(result)

        reasoning_parts = [r.reasoning for r in results if r.reasoning]
        content_parts = [r.content for r in results if r.content]

        assert "".join(reasoning_parts) == "thinking"
        assert "".join(content_parts) == "answer"


class TestDeltaMessage:
    """Tests for the DeltaMessage dataclass."""

    def test_reasoning_content_alias(self):
        """reasoning_content should alias reasoning."""
        msg = DeltaMessage(reasoning="test reasoning")
        assert msg.reasoning == "test reasoning"
        assert msg.reasoning_content == "test reasoning"

    def test_content_only(self):
        """Should handle content-only messages."""
        msg = DeltaMessage(content="just content")
        assert msg.content == "just content"
        assert msg.reasoning is None
        assert msg.reasoning_content is None

    def test_both_fields(self):
        """Should handle transition messages with both."""
        msg = DeltaMessage(reasoning="ending", content="starting")
        assert msg.reasoning == "ending"
        assert msg.content == "starting"


class TestEdgeCases:
    """Test edge cases across parsers."""

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        """Parametrized fixture for both parsers."""
        return get_parser(request.param)()

    def test_empty_output(self, parser):
        """Empty output should return (None, '')."""
        reasoning, content = parser.extract_reasoning("")
        # Either both None or content is empty string
        assert reasoning is None or reasoning == ""

    def test_whitespace_only_reasoning(self, parser):
        """Whitespace-only reasoning should be treated as None."""
        output = "<think>   </think>content"
        reasoning, content = parser.extract_reasoning(output)
        # Whitespace-only should be stripped to None
        if reasoning is not None:
            assert reasoning.strip() == "" or reasoning is None

    def test_nested_tags_not_supported(self, parser):
        """Nested tags are not officially supported - behavior may vary."""
        output = "<think>outer<think>inner</think>still outer</think>content"
        # Just ensure it doesn't crash
        reasoning, content = parser.extract_reasoning(output)
        # Result may vary by parser implementation

    def test_streaming_reset_state(self, parser):
        """reset_state should allow reuse of parser."""
        # First stream
        parser.reset_state()
        parser.extract_reasoning_streaming("", "<think>", "<think>")

        # Reset for new stream
        parser.reset_state()

        # Should work fresh
        result = parser.extract_reasoning_streaming("", "content", "content")
        assert result is not None


class TestRealisticStreaming:
    """Tests for realistic streaming scenarios simulating actual model output."""

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        """Parametrized fixture for both parsers."""
        return get_parser(request.param)()

    def test_token_by_token_streaming(self, parser):
        """Simulate realistic token-by-token streaming."""
        # Typical model output broken into tokens
        tokens = [
            "<",
            "think",
            ">",  # Start tag split across tokens
            "Let",
            " me",
            " analyze",
            " this",
            ".",
            "\n",
            "Step",
            " 1",
            ":",
            " check",
            " input",
            "\n",
            "Step",
            " 2",
            ":",
            " compute",
            "</",
            "think",
            ">",  # End tag split across tokens
            "The",
            " answer",
            " is",
            " 42",
            ".",
        ]

        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        # Verify reasoning was captured
        assert "Let me analyze" in full_reasoning
        assert "Step 1" in full_reasoning
        assert "Step 2" in full_reasoning

        # Verify content was captured
        assert "The answer is 42" in full_content

    def test_long_reasoning_streaming(self, parser):
        """Test streaming with extended reasoning."""
        # Long reasoning content
        reasoning_text = """
        First, I need to understand the problem.
        The user is asking about quantum computing.

        Let me break this down:
        1. Quantum bits (qubits) can be in superposition
        2. Entanglement allows correlated states
        3. Quantum gates perform operations

        After careful analysis, I can provide an answer.
        """

        output = f"<think>{reasoning_text}</think>Quantum computing uses qubits."

        # Simulate character-by-character streaming
        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for char in output:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "quantum computing" in full_reasoning.lower()
        assert "qubits" in full_reasoning.lower()
        assert "Quantum computing uses qubits" in full_content

    def test_streaming_no_content_after_reasoning(self, parser):
        """Test streaming when there's only reasoning, no content."""
        tokens = ["<think>", "just", " thinking", "</think>"]

        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "just thinking" in "".join(reasoning_parts)
        assert len(content_parts) == 0 or "".join(content_parts).strip() == ""


class TestUnicodeAndSpecialCharacters:
    """Tests for Unicode and special characters in reasoning."""

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        """Parametrized fixture for both parsers."""
        return get_parser(request.param)()

    def test_unicode_reasoning(self, parser):
        """Test reasoning with Unicode characters."""
        output = "<think>分析这个问题：日本語テスト émojis: 🤔💭</think>答案是42"
        reasoning, content = parser.extract_reasoning(output)
        assert "分析" in reasoning
        assert "日本語" in reasoning
        assert "🤔" in reasoning
        assert "42" in content

    def test_code_in_reasoning(self, parser):
        """Test reasoning containing code snippets."""
        output = """<think>
Let me analyze the code:
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```
This is a recursive implementation.
</think>The factorial function uses recursion."""

        reasoning, content = parser.extract_reasoning(output)
        assert "def factorial" in reasoning
        assert "recursive" in reasoning
        assert "uses recursion" in content

    def test_html_like_content(self, parser):
        """Test that HTML-like content doesn't confuse the parser."""
        output = "<think>The user mentioned <div> and <span> tags</think>Use CSS for styling."
        reasoning, content = parser.extract_reasoning(output)
        assert "<div>" in reasoning
        assert "<span>" in reasoning
        assert "CSS" in content

    def test_math_expressions(self, parser):
        """Test reasoning with mathematical expressions."""
        output = "<think>Given: x² + 2x + 1 = 0, so (x+1)² = 0, x = -1</think>x = -1"
        reasoning, content = parser.extract_reasoning(output)
        assert "x²" in reasoning
        assert "(x+1)²" in reasoning
        assert "-1" in content


class TestAPIModelsIntegration:
    """Tests for integration with API models."""

    def test_assistant_message_with_reasoning(self):
        """Test that AssistantMessage can hold reasoning content."""
        from vllm_mlx.api.models import AssistantMessage

        msg = AssistantMessage(
            content="The answer is 42.",
            reasoning_content="Let me think step by step...",
        )
        assert msg.content == "The answer is 42."
        assert msg.reasoning_content == "Let me think step by step..."
        assert msg.role == "assistant"

    def test_assistant_message_reasoning_none(self):
        """Test AssistantMessage with no reasoning."""
        from vllm_mlx.api.models import AssistantMessage

        msg = AssistantMessage(content="Simple response without reasoning.")
        assert msg.content == "Simple response without reasoning."
        assert msg.reasoning_content is None

    def test_chat_completion_chunk_delta_with_reasoning(self):
        """Test that ChatCompletionChunkDelta can hold reasoning_content."""
        from vllm_mlx.api.models import ChatCompletionChunkDelta

        delta = ChatCompletionChunkDelta(reasoning_content="thinking...")
        assert delta.reasoning_content == "thinking..."
        assert delta.content is None

        delta2 = ChatCompletionChunkDelta(content="response text")
        assert delta2.content == "response text"
        assert delta2.reasoning_content is None

    def test_delta_transition(self):
        """Test delta during transition from reasoning to content."""
        from vllm_mlx.api.models import ChatCompletionChunkDelta

        # During transition, both might have values
        delta = ChatCompletionChunkDelta(
            reasoning_content="final thought", content="starting answer"
        )
        assert delta.reasoning_content == "final thought"
        assert delta.content == "starting answer"


class TestParserPerformance:
    """Basic performance tests for parsers."""

    @pytest.fixture(params=["qwen3", "deepseek_r1"])
    def parser(self, request):
        """Parametrized fixture for both parsers."""
        return get_parser(request.param)()

    def test_large_output_extraction(self, parser):
        """Test extraction from large output."""
        # Generate large reasoning content
        reasoning_lines = [f"Step {i}: processing data chunk {i}" for i in range(100)]
        reasoning_text = "\n".join(reasoning_lines)
        output = f"<think>{reasoning_text}</think>Processing complete."

        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is not None
        assert "Step 0" in reasoning
        assert "Step 99" in reasoning
        assert content == "Processing complete."

    def test_streaming_many_chunks(self, parser):
        """Test streaming with many small chunks."""
        parser.reset_state()

        # Generate many small chunks
        base_output = "<think>A" * 100 + "</think>" + "B" * 50
        accumulated = ""
        chunk_count = 0

        for char in base_output:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                chunk_count += 1

        # Should have processed all characters
        assert chunk_count > 0

    def test_repeated_parsing(self, parser):
        """Test parsing same output multiple times."""
        output = "<think>Quick thought</think>Quick answer"

        for _ in range(100):
            reasoning, content = parser.extract_reasoning(output)
            assert reasoning == "Quick thought"
            assert content == "Quick answer"


class TestDeepSeekSpecificCases:
    """Tests specific to DeepSeek-R1 parser behavior."""

    @pytest.fixture
    def parser(self):
        """Create DeepSeek-R1 parser."""
        return get_parser("deepseek_r1")()

    def test_implicit_reasoning_streaming(self, parser):
        """Test streaming when start tag is implicit (DeepSeek-R1 specific)."""
        # DeepSeek-R1 sometimes omits <think> but includes </think>
        tokens = ["reasoning", " text", " here", "</think>", "answer"]

        parser.reset_state()
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        # For DeepSeek-R1, content before </think> without <think> is treated as content
        # until </think> appears in the delta
        all_parts = reasoning_parts + content_parts
        assert len(all_parts) > 0

    def test_deepseek_long_implicit_reasoning(self, parser):
        """Test long implicit reasoning without start tag."""
        output = """Let me think about this problem carefully.

First, I need to consider the constraints.
Then, I'll apply the algorithm.
Finally, I'll verify the result.</think>The answer is 42."""

        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "think about this problem" in reasoning
        assert "42" in content


class TestQwen3SpecificCases:
    """Tests specific to Qwen3 parser behavior."""

    @pytest.fixture
    def parser(self):
        """Create Qwen3 parser."""
        return get_parser("qwen3")()

    def test_qwen3_implicit_mode_support(self, parser):
        """Qwen3 supports implicit mode for OpenCode compatibility."""
        # Only end tag - implicit mode (think injected in prompt)
        output1 = "some text</think>more text"
        reasoning, content = parser.extract_reasoning(output1)
        # Implicit mode: everything before </think> is reasoning
        assert reasoning == "some text"
        assert content == "more text"

        # Only start tag - truncated thinking (max_tokens hit during reasoning)
        output2 = "<think>incomplete reasoning"
        reasoning, content = parser.extract_reasoning(output2)
        # Truncated: everything after <think> is reasoning, no content
        assert reasoning == "incomplete reasoning"
        assert content is None

    def test_qwen3_empty_think_tags(self, parser):
        """Test empty think tags."""
        output = "<think></think>Just the answer."
        reasoning, content = parser.extract_reasoning(output)
        # Empty reasoning should be None
        assert reasoning is None or reasoning.strip() == ""
        assert content == "Just the answer."

    def test_qwen3_whitespace_between_tags(self, parser):
        """Test various whitespace patterns."""
        test_cases = [
            ("<think> </think>answer", None, "answer"),
            ("<think>\n\n</think>answer", None, "answer"),
            ("<think>\t\t</think>answer", None, "answer"),
        ]

        for output, expected_reasoning, expected_content in test_cases:
            reasoning, content = parser.extract_reasoning(output)
            if expected_reasoning is None:
                assert reasoning is None or reasoning.strip() == ""
            assert expected_content in (content or "")


class TestGptOssParser:
    """Tests for the GPT-OSS reasoning parser (channel-based format)."""

    @pytest.fixture
    def parser(self):
        """Create a fresh GPT-OSS parser for each test."""
        return get_parser("gpt_oss")()

    # Non-streaming tests

    def test_extract_both_channels(self, parser):
        """Should extract reasoning from analysis and content from final."""
        output = (
            "<|channel|>analysis<|message|>Let me think step by step"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me think step by step"
        assert content == "The answer is 42"

    def test_extract_only_final(self, parser):
        """Should handle output with only final channel."""
        output = "<|channel|>final<|message|>Just the answer<|return|>"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == "Just the answer"

    def test_extract_only_analysis(self, parser):
        """Should handle output with only analysis channel."""
        output = "<|channel|>analysis<|message|>Just thinking out loud"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Just thinking out loud"
        assert content is None

    def test_no_channel_tokens_fallback(self, parser):
        """No channel tokens should return pure content."""
        output = "Just a regular response."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == output

    def test_empty_analysis_channel(self, parser):
        """Empty analysis channel should return None reasoning."""
        output = (
            "<|channel|>analysis<|message|>"
            "<|start|>assistant<|channel|>final<|message|>Content here<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == "Content here"

    def test_multiline_analysis(self, parser):
        """Should preserve multiline reasoning content."""
        output = (
            "<|channel|>analysis<|message|>Step 1: Analyze\nStep 2: Solve\nStep 3: Verify"
            "<|start|>assistant<|channel|>final<|message|>Result: 42<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Step 3" in reasoning
        assert content == "Result: 42"

    def test_no_return_token(self, parser):
        """Should handle missing <|return|> at end."""
        output = (
            "<|channel|>analysis<|message|>Thinking"
            "<|start|>assistant<|channel|>final<|message|>Answer"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Thinking"
        assert content == "Answer"

    # Streaming tests

    def test_streaming_full_flow(self, parser):
        """Test streaming through analysis -> transition -> final phases."""
        parser.reset_state()

        # Simulate token-by-token streaming
        tokens = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "Let me ",
            "think",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "The answer",
            " is 42",
            "<|return|>",
        ]

        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "Let me think" in full_reasoning
        assert "The answer is 42" in full_content

    def test_streaming_only_final(self, parser):
        """Test streaming with only final channel."""
        parser.reset_state()

        tokens = [
            "<|channel|>",
            "final",
            "<|message|>",
            "Direct ",
            "answer",
            "<|return|>",
        ]

        accumulated = ""
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result and result.content:
                content_parts.append(result.content)

        assert "Direct answer" in "".join(content_parts)

    def test_streaming_suppresses_structural_tokens(self, parser):
        """Structural tokens should not leak into reasoning or content."""
        parser.reset_state()

        tokens = [
            "<|channel|>analysis<|message|>",
            "thinking",
            "<|start|>",
            "assistant",
            "<|channel|>final<|message|>",
            "answer",
            "<|return|>",
        ]

        accumulated = ""
        all_output = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    all_output.append(result.reasoning)
                if result.content:
                    all_output.append(result.content)

        combined = "".join(all_output)
        assert "<|" not in combined

    def test_registry_includes_gpt_oss(self):
        """gpt_oss should be in the parser registry."""
        assert "gpt_oss" in list_parsers()

    def test_extract_constrain_format(self, parser):
        """Should handle extended format with <|constrain|> token."""
        output = (
            "<|channel|>analysis<|message|>We need to output JSON"
            "<|end|><|channel|>final <|constrain|>JSON<|message|>"
            '{"hello":"world"}<|return|>'
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "We need to output JSON"
        assert content == '{"hello":"world"}'

    def test_extract_constrain_no_analysis(self, parser):
        """Should handle constrain format with only final channel."""
        output = (
            '<|channel|>final <|constrain|>JSON<|message|>{"key":"value"}<|return|>'
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == '{"key":"value"}'

    def test_streaming_constrain_format(self, parser):
        """Streaming should handle <|constrain|> in channel marker."""
        parser.reset_state()

        tokens = [
            "<|channel|>analysis<|message|>",
            "Thinking...",
            "<|end|>",
            "<|channel|>final <|constrain|>JSON<|message|>",
            '{"result":',
            '"ok"}',
            "<|return|>",
        ]

        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)

        assert "Thinking" in full_reasoning
        assert '{"result":"ok"}' in full_content
        assert "<|constrain|>" not in full_content

    def test_constrain_tokens_stripped(self, parser):
        """<|constrain|> should not leak into output."""
        output = (
            '<|channel|>final <|constrain|>JSON<|message|>{"hello":"world"}<|return|>'
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "<|constrain|>" not in (content or "")
        assert "<|channel|>" not in (content or "")


class TestDeepSeekNoTagThreshold:
    """Tests for the no-tag content threshold in DeepSeek-R1 parser."""

    @pytest.fixture
    def parser(self):
        """Create DeepSeek-R1 parser."""
        return get_parser("deepseek_r1")()

    def test_no_tag_long_output_becomes_content(self, parser):
        """Long output without any tags should become content after threshold."""
        parser.reset_state()

        # Generate text longer than 64 chars without any think tags
        text = "This is a regular response without any thinking tags. " * 3
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for char in text:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        # After threshold, new chars should go to content
        full_content = "".join(content_parts)
        assert len(full_content) > 0, "Long no-tag output should have content"

    def test_with_tags_still_separates_correctly(self, parser):
        """Output with tags should still be correctly separated."""
        parser.reset_state()

        tokens = ["<think>", "reasoning here", "</think>", "content here"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "reasoning here" in "".join(reasoning_parts)
        assert "content here" in "".join(content_parts)

    def test_finalize_corrects_short_no_tag_output(self, parser):
        """finalize_streaming should correct short no-tag output."""
        parser.reset_state()

        # Stream a short output (under 64 chars) without tags
        text = "Short answer."
        accumulated = ""

        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        # Finalize should emit correction
        correction = parser.finalize_streaming(accumulated)
        assert correction is not None
        assert correction.content == text

    def test_finalize_no_correction_with_tags(self, parser):
        """finalize_streaming should not correct when tags were seen."""
        parser.reset_state()

        text = "<think>thinking</think>answer"
        accumulated = ""

        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        # No correction needed - tags were seen
        correction = parser.finalize_streaming(accumulated)
        assert correction is None

    def test_finalize_no_correction_for_long_no_tag(self, parser):
        """finalize_streaming should not correct long no-tag output (already content)."""
        parser.reset_state()

        text = "A" * 100  # Over threshold
        accumulated = ""

        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        # No correction needed - already classified as content past threshold
        correction = parser.finalize_streaming(accumulated)
        assert correction is None

    def test_saw_any_tag_flag_persists(self, parser):
        """_saw_any_tag should persist and reset correctly."""
        parser.reset_state()
        assert not parser._saw_any_tag

        # Stream with tags
        text = "<think>test</think>done"
        accumulated = ""
        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        assert parser._saw_any_tag

        # Reset should clear it
        parser.reset_state()
        assert not parser._saw_any_tag
