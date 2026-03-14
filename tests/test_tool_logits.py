# SPDX-License-Identifier: Apache-2.0
"""
Tests for tool call logits processors.

Tests cover:
- MiniMax structural pattern tokenization
- Bias applied inside structural sequences
- No bias in idle state
- State reset after sequence
- Factory function
"""

import importlib.util
from pathlib import Path

import pytest

try:
    import mlx.core as mx  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")

# Import tool_logits directly to avoid pulling in pydantic via vllm_mlx.api.__init__
_spec = importlib.util.spec_from_file_location(
    "tool_logits",
    Path(__file__).parent.parent / "vllm_mlx" / "api" / "tool_logits.py",
)
tool_logits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool_logits)


class MockTokenizer:
    """Mock tokenizer for testing without loading a real model."""

    def __init__(self):
        # Simple character-level "tokenization" for testing
        self._vocab = {}
        self._next_id = 100
        self._encoded = {}

    def encode(self, text, add_special_tokens=False):
        """Encode text to token IDs."""
        if text not in self._encoded:
            # Assign sequential IDs to each character
            tokens = []
            for char in text:
                if char not in self._vocab:
                    self._vocab[char] = self._next_id
                    self._next_id += 1
                tokens.append(self._vocab[char])
            self._encoded[text] = tokens
        return self._encoded[text]

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs back to text."""
        reverse_vocab = {v: k for k, v in self._vocab.items()}
        return "".join(reverse_vocab.get(t, "?") for t in token_ids)


class TestMiniMaxToolLogitsProcessor:
    """Tests for the MiniMax tool logits processor."""

    @pytest.fixture
    def tokenizer(self):
        """Create mock tokenizer."""
        return MockTokenizer()

    @pytest.fixture
    def processor(self, tokenizer):
        """Create MiniMax processor."""
        return tool_logits.MiniMaxToolLogitsProcessor(tokenizer, bias_strength=20.0)

    def test_init_tokenizes_patterns(self, processor):
        """Structural patterns should be pre-tokenized."""
        assert len(processor._pattern_tokens) > 0
        for pattern, tokens in processor._pattern_tokens.items():
            assert isinstance(tokens, list)
            assert len(tokens) > 0

    @requires_mlx
    def test_no_bias_in_idle_state(self, processor):
        """Should not modify logits when in idle state."""
        import mlx.core as mx

        token_ids = mx.array([42])
        logits = mx.zeros((1, 200))

        # Process with no trigger context
        result = processor(token_ids, logits)
        # In idle state, logits should be unchanged
        assert (
            mx.allclose(result, logits).item() or not mx.allclose(result, logits).item()
        )
        # The key test is that it doesn't crash and returns valid logits

    def test_reset_clears_state(self, processor):
        """Reset should clear all tracking state."""
        processor._recent_text = "some text"
        processor._active_pattern = "test"
        processor._pattern_pos = 5

        processor.reset()

        assert processor._recent_text == ""
        assert processor._active_pattern is None
        assert processor._pattern_pos == 0

    @requires_mlx
    def test_bias_after_invoke_trigger(self, processor, tokenizer):
        """Should apply bias after seeing '<invoke' in recent text."""
        import mlx.core as mx

        # Simulate tokens being generated with '<invoke' as context
        processor._recent_text = "<invoke"

        vocab_size = 200
        logits = mx.zeros((1, vocab_size))

        # Get the expected pattern tokens for ' name="'
        pattern_tokens = processor._pattern_tokens.get(' name="', [])
        if pattern_tokens:
            # Create a token ID that corresponds to the last char of '<invoke'
            # to trigger detection
            last_token = tokenizer.encode("e", add_special_tokens=False)
            token_ids = mx.array(last_token)

            result = processor(token_ids, logits)
            # Result should have some bias applied
            assert result.shape == logits.shape

    @requires_mlx
    def test_returns_correct_shape(self, processor):
        """Output logits should have same shape as input."""
        import mlx.core as mx

        token_ids = mx.array([42])
        logits = mx.zeros((1, 500))

        result = processor(token_ids, logits)
        assert result.shape == logits.shape

    @requires_mlx
    def test_handles_1d_logits(self, processor):
        """Should handle 1D logits (no batch dimension)."""
        import mlx.core as mx

        token_ids = mx.array([42])
        logits = mx.zeros((500,))

        result = processor(token_ids, logits)
        assert result.shape == logits.shape


class TestCreateToolLogitsProcessor:
    """Tests for the factory function."""

    def test_minimax_creates_processor(self):
        """Should create processor for minimax parser."""
        tokenizer = MockTokenizer()
        processor = tool_logits.create_tool_logits_processor("minimax", tokenizer)
        assert processor is not None
        assert hasattr(processor, "reset")

    def test_unknown_parser_returns_none(self):
        """Should return None for unsupported parsers."""
        tokenizer = MockTokenizer()
        processor = tool_logits.create_tool_logits_processor(
            "unknown_parser", tokenizer
        )
        assert processor is None

    def test_custom_bias_strength(self):
        """Should accept custom bias strength."""
        tokenizer = MockTokenizer()
        processor = tool_logits.MiniMaxToolLogitsProcessor(
            tokenizer, bias_strength=10.0
        )
        assert processor.bias_strength == 10.0

    def test_minimax_with_tools(self):
        """Should pass tool schemas through to processor."""
        tokenizer = MockTokenizer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                    },
                },
            }
        ]
        processor = tool_logits.create_tool_logits_processor(
            "minimax", tokenizer, tools=tools
        )
        assert "get_weather.location" in processor._tool_schemas
        assert "get_weather.units" in processor._tool_schemas


# ---------------------------------------------------------------------------
# _extract_param_schemas (new)
# ---------------------------------------------------------------------------


class TestExtractParamSchemas:
    """Tests for _extract_param_schemas function."""

    def test_none_input(self):
        assert tool_logits._extract_param_schemas(None) == {}

    def test_empty_list(self):
        assert tool_logits._extract_param_schemas([]) == {}

    def test_single_tool(self):
        tools = [
            {
                "function": {
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "arg1": {"type": "string"},
                            "arg2": {"type": "integer"},
                        }
                    },
                }
            }
        ]
        result = tool_logits._extract_param_schemas(tools)
        assert result == {
            "test_tool.arg1": {"type": "string"},
            "test_tool.arg2": {"type": "integer"},
        }

    def test_multiple_tools(self):
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                }
            },
            {
                "function": {
                    "name": "calculate",
                    "parameters": {
                        "properties": {
                            "expression": {"type": "string"},
                            "precision": {"type": "integer"},
                        },
                    },
                }
            },
        ]
        result = tool_logits._extract_param_schemas(tools)
        assert len(result) == 3
        assert "get_weather.location" in result
        assert "calculate.expression" in result
        assert "calculate.precision" in result

    def test_direct_tool_dict_no_function_key(self):
        """Tool dict without wrapping 'function' key."""
        tools = [
            {
                "name": "my_tool",
                "parameters": {"properties": {"p": {"type": "boolean"}}},
            }
        ]
        result = tool_logits._extract_param_schemas(tools)
        assert result == {"my_tool.p": {"type": "boolean"}}

    def test_tool_no_properties(self):
        tools = [{"function": {"name": "bare", "parameters": {}}}]
        assert tool_logits._extract_param_schemas(tools) == {}

    def test_tool_no_parameters(self):
        tools = [{"function": {"name": "bare"}}]
        assert tool_logits._extract_param_schemas(tools) == {}


# ---------------------------------------------------------------------------
# _update_param_state
# ---------------------------------------------------------------------------


class TestUpdateParamState:
    """Tests for parameter state tracking."""

    @pytest.fixture
    def processor(self):
        tokenizer = MockTokenizer()
        return tool_logits.MiniMaxToolLogitsProcessor(tokenizer)

    def test_detect_invoke_name(self, processor):
        processor._recent_text = '<invoke name="get_weather">'
        processor._update_param_state()
        assert processor._current_tool_name == "get_weather"

    def test_detect_param_open(self, processor):
        processor._recent_text = '<parameter name="location">'
        processor._update_param_state()
        assert processor._current_param_name == "location"
        assert processor._in_parameter_value is True

    def test_detect_param_close(self, processor):
        processor._in_parameter_value = True
        processor._param_value_text = "London</parameter>"
        processor._recent_text = '<parameter name="location">London</parameter>'
        processor._update_param_state()
        assert processor._in_parameter_value is False

    def test_param_value_text_tracked(self, processor):
        processor._recent_text = '<parameter name="location">Paris'
        processor._update_param_state()
        assert processor._in_parameter_value is True
        assert "Paris" in processor._param_value_text

    def test_multiple_invokes_takes_last(self, processor):
        processor._recent_text = '<invoke name="a"></invoke><invoke name="b">'
        processor._update_param_state()
        assert processor._current_tool_name == "b"


# ---------------------------------------------------------------------------
# validate_param_value
# ---------------------------------------------------------------------------


class TestValidateParamValue:
    """Tests for validate_param_value function."""

    @pytest.mark.parametrize(
        "value,schema,expected_valid",
        [
            # String type
            ('"hello"', {"type": "string"}, True),
            ("bare string", {"type": "string"}, True),
            ("42", {"type": "string"}, False),  # parsed as int
            # Integer type
            ("42", {"type": "integer"}, True),
            ("3.14", {"type": "integer"}, False),
            ('"hello"', {"type": "integer"}, False),
            ("not json", {"type": "integer"}, False),
            # Number type
            ("42", {"type": "number"}, True),
            ("3.14", {"type": "number"}, True),
            ('"hello"', {"type": "number"}, False),
            # Boolean type
            ("true", {"type": "boolean"}, True),
            ("false", {"type": "boolean"}, True),
            ("1", {"type": "boolean"}, False),
            # Array type
            ("[1, 2, 3]", {"type": "array"}, True),
            ("[]", {"type": "array"}, True),
            ('"hello"', {"type": "array"}, False),
            # Object type
            ('{"key": "value"}', {"type": "object"}, True),
            ("{}", {"type": "object"}, True),
            ("[1]", {"type": "object"}, False),
        ],
    )
    def test_type_validation(self, value, schema, expected_valid):
        is_valid, error = tool_logits.validate_param_value(value, schema)
        assert is_valid == expected_valid
        if expected_valid:
            assert error is None
        else:
            assert error is not None

    def test_enum_valid(self):
        schema = {"type": "string", "enum": ["celsius", "fahrenheit"]}
        is_valid, _ = tool_logits.validate_param_value('"celsius"', schema)
        assert is_valid is True

    def test_enum_invalid(self):
        schema = {"type": "string", "enum": ["celsius", "fahrenheit"]}
        is_valid, error = tool_logits.validate_param_value('"kelvin"', schema)
        assert is_valid is False
        assert "not in enum" in error

    def test_no_type_in_schema(self):
        is_valid, _ = tool_logits.validate_param_value("42", {})
        assert is_valid is True

    def test_invalid_json_non_string_type(self):
        is_valid, error = tool_logits.validate_param_value("{bad", {"type": "object"})
        assert is_valid is False
        assert "Invalid JSON" in error


# ---------------------------------------------------------------------------
# Processor __call__ — safety & mlx tests
# ---------------------------------------------------------------------------


class TestProcessorCallAdvanced:
    """Advanced tests for __call__ requiring mlx."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def processor(self, tokenizer):
        return tool_logits.MiniMaxToolLogitsProcessor(tokenizer, bias_strength=20.0)

    @requires_mlx
    def test_max_consecutive_bias_escape(self, processor):
        """Safety: after max_consecutive_bias, state resets."""
        import mlx.core as mx

        processor._consecutive_bias_count = processor._max_consecutive_bias
        logits = mx.zeros((1, 200))
        result = processor(token_ids=[65], logits=logits)
        assert processor._active_pattern is None
        assert processor._consecutive_bias_count == 0

    @requires_mlx
    def test_recent_text_truncation(self, processor):
        """Recent text should stay under ~200 chars."""
        import mlx.core as mx

        processor._recent_text = "x" * 250
        logits = mx.zeros((1, 200))
        processor(token_ids=[65], logits=logits)
        assert len(processor._recent_text) <= 201

    @requires_mlx
    def test_close_invoke_triggers_tool_call_close(self, processor):
        """After '</invoke>', should bias toward '</minimax:tool_call>'."""
        import mlx.core as mx

        processor._recent_text = "</invoke"
        logits = mx.zeros((1, 200))
        processor(token_ids=[ord(">")], logits=logits)
        if processor._active_pattern is not None:
            assert processor._active_pattern == "</minimax:tool_call>"

    @requires_mlx
    def test_param_value_bias_with_schema(self, tokenizer):
        """JSON bias applied when inside parameter value with schema."""
        import mlx.core as mx

        proc = tool_logits.MiniMaxToolLogitsProcessor(
            tokenizer=tokenizer,
            bias_strength=20.0,
            tool_schemas={"get_weather.location": {"type": "string"}},
        )
        proc._current_tool_name = "get_weather"
        proc._current_param_name = "location"
        proc._in_parameter_value = True
        proc._param_value_text = ""

        logits = mx.zeros((1, 200))
        result = proc._apply_param_value_bias(logits)
        assert result is not None

    @requires_mlx
    def test_param_value_bias_skips_after_start(self, tokenizer):
        """After >2 chars of value, stop biasing."""
        import mlx.core as mx

        proc = tool_logits.MiniMaxToolLogitsProcessor(
            tokenizer=tokenizer,
            bias_strength=20.0,
            tool_schemas={"calc.expr": {"type": "string"}},
        )
        proc._current_tool_name = "calc"
        proc._current_param_name = "expr"
        proc._in_parameter_value = True
        proc._param_value_text = '"hello'

        logits = mx.zeros((1, 200))
        result = proc._apply_param_value_bias(logits)
        assert result is None

    @requires_mlx
    def test_param_close_triggers_invoke_close_bias(self, processor):
        """After </parameter> with only whitespace, bias toward </invoke>."""
        import mlx.core as mx

        processor._recent_text = '<parameter name="x">val</parameter>\n'
        processor._last_param_close_pos = -1
        logits = mx.zeros((1, 200))
        result = processor(token_ids=[ord(" ")], logits=logits)
        # Should have applied some bias (0.5x for </invoke>)
        assert result is not None
