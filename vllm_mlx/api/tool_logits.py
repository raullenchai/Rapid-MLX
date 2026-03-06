# SPDX-License-Identifier: Apache-2.0
"""
Logits processors for jump-forward decoding of tool call structural tokens.

When models generate tool calls in XML format (e.g., MiniMax's
<minimax:tool_call>), many tokens are predictable structural markup.
By biasing logits toward the expected next token, we accelerate generation
of these structural sequences without constraining the model's free choices
for argument values.

Usage:
    processor = create_tool_logits_processor("minimax", tokenizer)
    if processor:
        # Pass to BatchGenerator via logits_processors
        ...
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def _extract_param_schemas(tools: list[dict] | None) -> dict[str, dict]:
    """
    Extract parameter JSON schemas from tool definitions.

    Returns a dict mapping "tool_name.param_name" -> JSON schema for that parameter.
    """
    if not tools:
        return {}

    schemas: dict[str, dict] = {}
    for tool in tools:
        func = tool.get("function", tool)
        tool_name = func.get("name", "")
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        for param_name, param_schema in properties.items():
            key = f"{tool_name}.{param_name}"
            schemas[key] = param_schema
    return schemas


class ToolLogitsProcessor(Protocol):
    """Protocol for tool call logits processors."""

    def __call__(self, token_ids: Any, logits: Any) -> Any:
        """Apply logits bias based on current generation state."""
        ...

    def reset(self) -> None:
        """Reset state for a new generation."""
        ...


class MiniMaxToolLogitsProcessor:
    """
    Logits processor that biases structural tokens in MiniMax tool calls.

    MiniMax tool call format:
        <minimax:tool_call>
        <invoke name="function_name">
        <parameter name="param">value</parameter>
        </invoke>
        </minimax:tool_call>

    After detecting the start of a structural sequence, biases the logits
    toward the expected continuation tokens. Does not constrain the model
    for free-form content (function names, parameter names, values).

    State machine:
        idle -> after_invoke (saw "<invoke")
        after_invoke -> idle (saw ' name="')
        idle -> after_param_value (saw '">...something not starting with <')
        ... etc.

    The processor uses a simpler approach: it pre-tokenizes known structural
    patterns and when the recent tokens match a pattern prefix, biases toward
    the next token in the pattern.
    """

    # Structural patterns that follow predictable sequences
    PATTERNS = [
        # After <invoke → expect ' name="'
        (' name="', "<invoke"),
        # After param value closing → expect </parameter>
        ("</parameter>", None),  # Triggered by seeing '">' after param value
        # After </parameter> block → could be another <parameter or </invoke>
        ("</invoke>", None),  # Triggered contextually
        # After </invoke> → expect </minimax:tool_call>
        ("</minimax:tool_call>", "</invoke>"),
    ]

    def __init__(
        self,
        tokenizer: Any,
        bias_strength: float = 20.0,
        tool_schemas: dict[str, dict] | None = None,
    ):
        """
        Initialize the MiniMax tool logits processor.

        Args:
            tokenizer: The tokenizer to use for encoding patterns.
            bias_strength: Logits bias to add to expected tokens.
            tool_schemas: Map of "tool.param" -> JSON schema for parameter value constraint.
        """
        self.tokenizer = tokenizer
        self.bias_strength = bias_strength
        self._tool_schemas = tool_schemas or {}

        # Pre-tokenize structural fragments
        self._pattern_tokens: dict[str, list[int]] = {}
        for pattern, _ in self.PATTERNS:
            tokens = tokenizer.encode(pattern, add_special_tokens=False)
            if tokens:
                self._pattern_tokens[pattern] = tokens

        # Pre-tokenize common JSON structural tokens for parameter value bias
        self._json_tokens: dict[str, list[int]] = {}
        for char in ['"', "{", "[", "]", "}", ",", ":", "true", "false", "null"]:
            toks = tokenizer.encode(char, add_special_tokens=False)
            if toks:
                self._json_tokens[char] = toks

        # State tracking
        self._recent_text = ""
        self._active_pattern: str | None = None
        self._pattern_pos = 0  # Position within active pattern's token sequence
        self._last_param_close_pos = (
            -1
        )  # Track last </parameter> position to avoid re-triggering
        self._consecutive_bias_count = 0  # Safety: escape hatch for stuck patterns
        self._max_consecutive_bias = 50  # Max tokens to bias before force-resetting

        # Parameter value tracking for structural constraint
        self._current_tool_name: str | None = None
        self._current_param_name: str | None = None
        self._in_parameter_value = False
        self._param_value_text = ""  # Accumulated text of current param value

    def reset(self) -> None:
        """Reset state for a new generation."""
        self._recent_text = ""
        self._active_pattern = None
        self._pattern_pos = 0
        self._last_param_close_pos = -1
        self._consecutive_bias_count = 0
        self._current_tool_name = None
        self._current_param_name = None
        self._in_parameter_value = False
        self._param_value_text = ""

    # Regex patterns for detecting tool/parameter context
    _INVOKE_RE = re.compile(r'<invoke\s+name="([^"]+)"')
    _PARAM_OPEN_RE = re.compile(r'<parameter\s+name="([^"]+)">')
    _PARAM_CLOSE_RE = re.compile(r"</parameter>")

    def _update_param_state(self) -> None:
        """Update parameter value tracking state from recent text."""
        text = self._recent_text

        # Detect <invoke name="tool_name">
        for m in self._INVOKE_RE.finditer(text):
            self._current_tool_name = m.group(1)

        # Detect <parameter name="param_name"> → entering value
        for m in self._PARAM_OPEN_RE.finditer(text):
            self._current_param_name = m.group(1)
            end_pos = m.end()
            # Only activate if this is the latest unclosed parameter
            close_after = text.find("</parameter>", end_pos)
            if close_after == -1:
                # No close tag after this open → we're inside value
                self._in_parameter_value = True
                self._param_value_text = text[end_pos:]

        # Detect </parameter> → leaving value
        if self._in_parameter_value:
            if "</parameter>" in self._param_value_text or text.rstrip().endswith(
                "</parameter>"
            ):
                self._in_parameter_value = False
                self._param_value_text = ""

    def _apply_param_value_bias(self, logits: Any) -> Any | None:
        """
        Apply JSON structural bias when generating a parameter value.

        Uses the schema type to bias toward valid JSON tokens:
        - string: bias toward quote characters
        - number/integer: bias toward digit tokens
        - boolean: bias toward 'true'/'false'
        - object/array: bias toward opening braces/brackets

        Returns biased logits, or None to skip bias (let model generate freely).
        """
        import mlx.core as mx

        if not self._current_tool_name or not self._current_param_name:
            return None

        schema_key = f"{self._current_tool_name}.{self._current_param_name}"
        schema = self._tool_schemas.get(schema_key)
        if not schema:
            return None

        param_type = schema.get("type", "")
        value_text = self._param_value_text.strip()

        # Only bias at the START of a value (first meaningful token)
        # Once the model has started generating, let it continue freely
        if len(value_text) > 2:
            return None

        bias_tokens: list[int] = []
        weak_bias = self.bias_strength * 0.3  # Lighter bias for value guidance

        if param_type == "string":
            # Strings should start with "
            if not value_text:
                bias_tokens = self._json_tokens.get('"', [])
        elif param_type in ("number", "integer"):
            # Numbers: bias toward digit tokens (0-9, -, .)
            for ch in "0123456789-.":
                toks = self.tokenizer.encode(ch, add_special_tokens=False)
                if toks:
                    bias_tokens.extend(toks)
        elif param_type == "boolean":
            # Bias toward 'true' and 'false'
            for val in ["true", "false"]:
                toks = self._json_tokens.get(val, [])
                bias_tokens.extend(toks)
        elif param_type == "object":
            if not value_text:
                bias_tokens = self._json_tokens.get("{", [])
        elif param_type == "array":
            if not value_text:
                bias_tokens = self._json_tokens.get("[", [])

        if not bias_tokens:
            return None

        bias = mx.zeros_like(logits)
        for tok in bias_tokens:
            if logits.ndim == 2:
                bias[0, tok] = weak_bias
            else:
                bias[tok] = weak_bias
        return logits + bias

    def __call__(self, token_ids: Any, logits: Any) -> Any:
        """
        Apply logits bias for structural tool call tokens.

        Args:
            token_ids: Previously generated token IDs.
            logits: Current logits tensor (1, vocab_size).

        Returns:
            Modified logits tensor.
        """
        import mlx.core as mx

        # Decode last few tokens to track context
        if hasattr(token_ids, "tolist"):
            id_list = token_ids.tolist()
        else:
            id_list = list(token_ids)

        if not id_list:
            return logits

        # Safety: escape hatch if stuck in a bias loop
        if self._consecutive_bias_count >= self._max_consecutive_bias:
            logger.warning(
                "Tool logits processor hit max consecutive bias limit "
                f"({self._max_consecutive_bias}), resetting state"
            )
            self._active_pattern = None
            self._pattern_pos = 0
            self._consecutive_bias_count = 0
            return logits

        # Decode last token to update recent text
        last_token_text = self.tokenizer.decode(
            [id_list[-1]], skip_special_tokens=False
        )
        self._recent_text += last_token_text
        # Keep only last 200 chars for matching
        if len(self._recent_text) > 200:
            self._recent_text = self._recent_text[-200:]

        # --- Parameter value state tracking ---
        self._update_param_state()

        # If inside a parameter value, apply JSON structural bias
        if self._in_parameter_value and self._tool_schemas:
            biased = self._apply_param_value_bias(logits)
            if biased is not None:
                return biased

        # If we're tracking an active pattern, bias toward next token
        if self._active_pattern is not None:
            pattern_tokens = self._pattern_tokens.get(self._active_pattern, [])
            if self._pattern_pos < len(pattern_tokens):
                target_token = pattern_tokens[self._pattern_pos]
                self._pattern_pos += 1
                self._consecutive_bias_count += 1

                # Add bias to the expected token
                bias = mx.zeros_like(logits)
                if logits.ndim == 2:
                    bias[0, target_token] = self.bias_strength
                else:
                    bias[target_token] = self.bias_strength
                return logits + bias
            else:
                # Pattern complete — skip trigger check this call to avoid
                # re-activating on stale _recent_text
                self._active_pattern = None
                self._pattern_pos = 0
                self._consecutive_bias_count = 0
                return logits

        # Not biasing — reset counter
        self._consecutive_bias_count = 0

        # Check if we should start tracking a pattern
        for pattern, trigger in self.PATTERNS:
            if trigger and self._recent_text.rstrip().endswith(trigger):
                pattern_tokens = self._pattern_tokens.get(pattern, [])
                if pattern_tokens:
                    self._active_pattern = pattern
                    self._pattern_pos = 0
                    # Bias first token
                    target_token = pattern_tokens[0]
                    self._pattern_pos = 1
                    self._consecutive_bias_count = 1

                    bias = mx.zeros_like(logits)
                    if logits.ndim == 2:
                        bias[0, target_token] = self.bias_strength
                    else:
                        bias[target_token] = self.bias_strength
                    return logits + bias

        # Check for </invoke> trigger: after seeing </parameter>\n or similar
        # Only trigger once per </parameter> occurrence to avoid repeated bias
        param_close_pos = self._recent_text.rfind("</parameter>")
        if param_close_pos > self._last_param_close_pos:
            after_param = self._recent_text[param_close_pos + len("</parameter>") :]
            # If the text after </parameter> is whitespace only, we might
            # be about to see </invoke> or another <parameter
            stripped = after_param.strip()
            if not stripped:
                self._last_param_close_pos = param_close_pos
                pattern = "</invoke>"
                pattern_tokens = self._pattern_tokens.get(pattern, [])
                if pattern_tokens:
                    target_token = pattern_tokens[0]
                    bias = mx.zeros_like(logits)
                    if logits.ndim == 2:
                        bias[0, target_token] = self.bias_strength * 0.5
                    else:
                        bias[target_token] = self.bias_strength * 0.5
                    return logits + bias

        return logits


def create_tool_logits_processor(
    parser_name: str,
    tokenizer: Any,
    bias_strength: float = 20.0,
    tools: list[dict] | None = None,
) -> ToolLogitsProcessor | None:
    """
    Factory function to create a tool logits processor for a given parser.

    Args:
        parser_name: Name of the tool call parser (e.g., "minimax").
        tokenizer: The tokenizer instance.
        bias_strength: Logits bias strength.
        tools: Optional tool definitions for parameter value schema constraint.

    Returns:
        A logits processor instance, or None if not supported for this parser.
    """
    tool_schemas = _extract_param_schemas(tools)
    if parser_name == "minimax":
        return MiniMaxToolLogitsProcessor(
            tokenizer,
            bias_strength=bias_strength,
            tool_schemas=tool_schemas,
        )
    # Future: add support for other parsers (hermes, llama, etc.)
    return None


def validate_param_value(value: str, schema: dict) -> tuple[bool, str | None]:
    """
    Validate a parameter value against its JSON schema (lightweight).

    Used by SimpleEngine for post-generation validation of tool call parameters.

    Args:
        value: The parameter value string.
        schema: JSON schema for the parameter.

    Returns:
        (is_valid, error_message) tuple.
    """
    param_type = schema.get("type", "")

    # Try to parse as JSON first
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        # Not valid JSON — check if it's a bare string (common for string params)
        if param_type == "string":
            return True, None  # Bare strings are acceptable for string params
        return False, f"Invalid JSON value: {value!r}"

    # Type check
    if param_type == "string" and not isinstance(parsed, str):
        return False, f"Expected string, got {type(parsed).__name__}"
    elif param_type == "integer" and not isinstance(parsed, int):
        return False, f"Expected integer, got {type(parsed).__name__}"
    elif param_type == "number" and not isinstance(parsed, (int, float)):
        return False, f"Expected number, got {type(parsed).__name__}"
    elif param_type == "boolean" and not isinstance(parsed, bool):
        return False, f"Expected boolean, got {type(parsed).__name__}"
    elif param_type == "array" and not isinstance(parsed, list):
        return False, f"Expected array, got {type(parsed).__name__}"
    elif param_type == "object" and not isinstance(parsed, dict):
        return False, f"Expected object, got {type(parsed).__name__}"

    # Enum check
    if "enum" in schema and parsed not in schema["enum"]:
        return False, f"Value {parsed!r} not in enum {schema['enum']}"

    return True, None
