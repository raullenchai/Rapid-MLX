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

import datetime
import json
import logging
import math
import re
import uuid
from collections.abc import Callable
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
        if not isinstance(tool, dict):
            continue
        func = tool.get("function", tool)
        if not isinstance(func, dict):
            continue
        tool_name = func.get("name", "")
        params = func.get("parameters", {})
        # Tolerate malformed tool schemas: ``parameters`` may legally be
        # omitted (treated as no-args) but client JSON sometimes ships it
        # as ``null``, a bare string, a list, or a number. Skip cleanly
        # instead of letting ``params.get(...)`` raise ``AttributeError``
        # — that bubbles up as an unmapped 500 and leaks raw Python error
        # text. ``properties`` has the same exposure (F-140 / retires
        # F-031).
        if not isinstance(params, dict):
            continue
        properties = params.get("properties", {})
        if not isinstance(properties, dict):
            continue
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


# JSON-Schema → Python type predicate. Booleans are explicitly excluded
# from ``integer`` / ``number`` because Python treats ``True``/``False``
# as ``int`` subclasses (``isinstance(True, int) is True``); JSON-Schema
# and pydantic both consider them their own type.
def _matches_single_json_type(parsed: object, t: str) -> bool:
    if t == "string":
        return isinstance(parsed, str)
    if t == "integer":
        return isinstance(parsed, int) and not isinstance(parsed, bool)
    if t == "number":
        return isinstance(parsed, (int, float)) and not isinstance(parsed, bool)
    if t == "boolean":
        return isinstance(parsed, bool)
    if t == "array":
        return isinstance(parsed, list)
    if t == "object":
        return isinstance(parsed, dict)
    if t == "null":
        return parsed is None
    # Unknown type name — treat as no constraint (skip), matching the
    # `allowed_types` normalisation in ``validate_param_value`` which
    # already drops non-string entries.
    return False


def _value_matches_any_type(parsed: object, allowed_types: set[str]) -> bool:
    """Return True if ``parsed`` matches at least one type in
    ``allowed_types``.

    Codex round 2 BLOCKING on PR #736 caught that the original
    branch tree never visited union-typed schemas (``"type": [...]``).
    Splitting the predicate into a single-type helper lets the union
    case fall out as a simple ``any()`` over the set.
    """
    if not allowed_types:
        return True  # No declared type → no constraint
    return any(_matches_single_json_type(parsed, t) for t in allowed_types)


def validate_param_value(value: str, schema: dict) -> tuple[bool, str | None]:
    """
    Validate a parameter value against its JSON schema (lightweight).

    Enforced constraints (F-141 scoped fix): ``type`` (strict — booleans
    are NOT accepted for ``integer``/``number`` even though
    ``isinstance(True, int)`` is True in Python), ``enum``, ``minimum`` /
    ``maximum`` (numeric only), ``minLength`` / ``maxLength`` (string
    only). Other JSON-schema keywords (``pattern``, ``format``,
    ``multipleOf``, ``uniqueItems``, ``required``, ``oneOf``/``anyOf``,
    ``additionalProperties``) are intentionally left pass-through and
    tracked as a follow-up — see TODO below.

    Args:
        value: The parameter value string.
        schema: JSON schema for the parameter.

    Returns:
        (is_valid, error_message) tuple.
    """
    # Defensive guard: ``_extract_param_schemas`` may publish a non-dict
    # entry when a future change to the extraction shape is bug-prone
    # (e.g. earlier ``properties`` of mixed types reaching here as the
    # leaf schema). Treat non-dict as "no constraint" rather than letting
    # ``schema.get(...)`` raise ``AttributeError`` and 500. (F-140)
    if not isinstance(schema, dict):
        return True, None
    param_type = schema.get("type", "")

    # JSON Schema lets ``type`` be either a single string or a list of
    # strings (a "union" type — e.g. ``{"type": ["string", "null"]}``).
    # Normalise to a set of allowed scalar names; an empty set means
    # "no type constraint declared" (the schema may still carry an
    # ``enum`` / range / length constraint which we honour below).
    # Anything we don't recognise (e.g. a stray int that survived the
    # F-140 guards upstream) is dropped from the allowed set so it
    # doesn't accidentally permit every type — codex round 2 BLOCKING
    # on PR #736 caught the original branch leaving union schemas
    # unchecked.
    if isinstance(param_type, str):
        allowed_types: set[str] = {param_type} if param_type else set()
    elif isinstance(param_type, list):
        allowed_types = {t for t in param_type if isinstance(t, str)}
    else:
        allowed_types = set()

    # Try to parse as JSON first
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        # Not valid JSON — check if it's a bare string (common for string params)
        if "string" in allowed_types:
            return True, None  # Bare strings are acceptable for string params
        return False, f"Invalid JSON value: {value!r}"

    # Type check (strict — exclude bool from integer/number; pydantic
    # treats ``True``/``False`` as their own type, not as ``1``/``0``).
    # For union ``type`` ([...]) the value is valid if it matches ANY
    # allowed type. ``allowed_types == set()`` means the schema omitted
    # ``type`` entirely — we skip this branch and fall through to the
    # enum / range / length checks. Unrecognised type names (anything
    # outside the six JSON-schema scalars + ``"null"``) are ignored
    # rather than silently permitting; if the schema lists ONLY
    # unknown types we skip the type check entirely.
    if allowed_types:
        type_matches = _value_matches_any_type(parsed, allowed_types)
        if not type_matches:
            display = (
                next(iter(allowed_types))
                if len(allowed_types) == 1
                else f"one of {sorted(allowed_types)}"
            )
            return (
                False,
                f"Expected {display}, got {type(parsed).__name__}",
            )

    # Enum check
    if "enum" in schema and parsed not in schema["enum"]:
        return False, f"Value {parsed!r} not in enum {schema['enum']}"

    # Numeric range (only when the parsed value is actually numeric —
    # the type-check branch above already rejected non-numeric values
    # for declared ``integer``/``number`` types, but a schema may omit
    # ``type`` and still carry ``minimum``/``maximum``).
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
            if parsed < minimum:
                return False, f"Value {parsed} below minimum {minimum}"
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
            if parsed > maximum:
                return False, f"Value {parsed} above maximum {maximum}"

    # String length
    if isinstance(parsed, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and not isinstance(min_length, bool):
            if len(parsed) < min_length:
                return (
                    False,
                    f"String length {len(parsed)} below minLength {min_length}",
                )
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and not isinstance(max_length, bool):
            if len(parsed) > max_length:
                return (
                    False,
                    f"String length {len(parsed)} above maxLength {max_length}",
                )

    # F-141a: pattern / format / multipleOf / uniqueItems
    #
    # These were deferred from PR #736 (F-141 scoped). The follow-up
    # set is intentionally narrow — only constraints with an
    # unambiguous JSON-schema semantics. ``required``, ``oneOf``,
    # ``anyOf``, ``additionalProperties`` are still deferred (compound
    # semantics, multiple violation classes per call, schema-traversal
    # rework).

    # pattern — regex match (``re.fullmatch`` per operator note,
    # NOT ``re.match`` which only anchors the start). Only string
    # parsed values are checked; non-string parsed values fall
    # through (the upstream ``type`` branch already rejected those
    # when ``type:"string"`` was declared).
    if isinstance(parsed, str):
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and pattern:
            try:
                if re.fullmatch(pattern, parsed) is None:
                    return (
                        False,
                        f"String {parsed!r} does not match pattern {pattern!r}",
                    )
            except re.error:
                # Bogus regex in the *schema* — fall back to advisory
                # (the schema author is at fault, not the model).
                pass

    # format — best-effort, narrow allow-list. Unknown formats
    # intentionally pass through as advisory (operator note:
    # "reject unknown formats as 200 OK passthrough rather than
    # 400, to stay loose"). Only string parsed values are checked.
    if isinstance(parsed, str):
        fmt = schema.get("format")
        if isinstance(fmt, str) and fmt in _FORMAT_VALIDATORS:
            if not _FORMAT_VALIDATORS[fmt](parsed):
                return (
                    False,
                    f"String {parsed!r} is not a valid {fmt}",
                )

    # multipleOf — integer / float. ``value % multipleOf == 0`` for
    # integers; ``math.isclose`` for floats (operator note) so a
    # 0.1 + 0.2 == 0.3 float-drift case doesn't 400.
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        multiple_of = schema.get("multipleOf")
        if isinstance(multiple_of, (int, float)) and not isinstance(multiple_of, bool):
            if multiple_of <= 0:
                # JSON-schema requires multipleOf > 0; bogus schema,
                # treat as advisory rather than 400.
                pass
            elif isinstance(parsed, int) and isinstance(multiple_of, int):
                if parsed % multiple_of != 0:
                    return (
                        False,
                        f"Value {parsed} is not a multiple of {multiple_of}",
                    )
            else:
                # Float arithmetic — use ``math.isclose`` on the
                # remainder. ``abs(parsed / multiple_of -
                # round(parsed / multiple_of))`` is the canonical
                # "distance to nearest multiple" measure.
                #
                # codex r3 BLOCKING: the default ``rel_tol=1e-9`` lets
                # large non-multiples through because the relative
                # error stays small (e.g. ``quotient = 1e10 +
                # epsilon`` rounds to ``1e10`` with relative drift
                # well under 1e-9). Pin ``rel_tol=0.0`` so only the
                # absolute tolerance applies, which is the operator's
                # intent for "the model emitted a value that *should*
                # be a multiple but is off by a few floating-point
                # ulps".
                quotient = parsed / multiple_of
                if not math.isclose(
                    quotient, round(quotient), rel_tol=0.0, abs_tol=1e-9
                ):
                    return (
                        False,
                        f"Value {parsed} is not a multiple of {multiple_of}",
                    )

    # uniqueItems — for arrays. JSON-schema uses *value* equality, not
    # JSON-text equality, so we need a canonicalisation that respects
    # number equality (``1`` and ``1.0`` are duplicates per spec) and
    # nested-dict structural equality. codex r2 BLOCKING: the previous
    # ``json.dumps`` keying treated ``1`` and ``1.0`` as different and
    # let real duplicates through.
    if isinstance(parsed, list) and schema.get("uniqueItems") is True:
        try:
            canonical = [_uniqueitems_canonical(item) for item in parsed]
        except TypeError:
            # Non-JSON-serialisable item somehow leaked in (parsed came
            # from ``json.loads``, so this is unreachable in practice
            # but defensive). Skip the uniqueness check.
            canonical = None
        if canonical is not None and len(set(canonical)) != len(canonical):
            return False, f"Array {parsed!r} has duplicate items (uniqueItems)"

    return True, None


# ---------------------------------------------------------------------
# F-141a — narrow ``format`` validators.
#
# Per operator scope: ``email`` / ``uri`` / ``uuid`` / ``date`` /
# ``date-time``. Unknown ``format`` values pass through (advisory) so
# we stay loose for real-world schemas that ship bespoke format hints.
# ---------------------------------------------------------------------


def _is_email(value: str) -> bool:
    # RFC 5321 / 5322 compliance is impossible without a real parser.
    # Use a defensive regex: one ``@``, non-empty local part, non-empty
    # domain part with at least one dot, no whitespace, no control
    # chars. Mirrors what the openapi-style "loose email" check does.
    if not value or len(value) > 254:
        return False
    if any(c.isspace() or ord(c) < 0x20 for c in value):
        return False
    return bool(_EMAIL_RE.fullmatch(value))


def _is_uri(value: str) -> bool:
    # Loose URI shape: ``scheme:rest`` with a non-empty scheme that
    # starts with a letter and only contains letters / digits / ``+`` /
    # ``-`` / ``.`` (RFC 3986 §3.1 scheme grammar). Doesn't validate
    # the path / query — schema-level guards are best-effort.
    if not value or any(c.isspace() for c in value):
        return False
    return bool(_URI_RE.match(value))


def _is_uuid(value: str) -> bool:
    try:
        # ``UUID(str)`` accepts hyphenated 8-4-4-4-12, urn:uuid:..., and
        # raw 32-char hex. JSON-schema's ``format: uuid`` is canonically
        # the hyphenated form, so verify after a normalising round-trip.
        u = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return str(u) == value.lower().removeprefix("urn:uuid:")


def _is_date(value: str) -> bool:
    # JSON-schema ``date`` = RFC 3339 ``full-date`` = ``YYYY-MM-DD``.
    try:
        datetime.date.fromisoformat(value)
    except (ValueError, TypeError):
        return False
    # ``fromisoformat`` accepts e.g. ``2024-01-01T00:00:00`` on Python
    # 3.11+; reject anything carrying a time component.
    return len(value) == 10 and value[4] == "-" and value[7] == "-"


def _is_date_time(value: str) -> bool:
    # JSON-schema ``date-time`` = RFC 3339 ``date-time``. ``Z`` is the
    # UTC shorthand; ``datetime.fromisoformat`` doesn't accept it on
    # pre-3.11 Pythons, so swap it for ``+00:00`` first.
    candidate = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        dt = datetime.datetime.fromisoformat(candidate)
    except (ValueError, TypeError):
        return False
    # codex r4 BLOCKING: JSON-schema ``date-time`` is RFC 3339 ``date-time``
    # whose grammar requires the literal ``T`` separator between
    # ``full-date`` and ``full-time``. The space form is a NOTE in the
    # spec ("Applications that generate this format SHOULD use uppercase
    # T") and is NOT part of the production rule we enforce here.
    # ``fromisoformat`` accepts a bare ``YYYY-MM-DD`` on 3.11+ (wrong
    # format — date, not date-time) and ``"2024-01-15 10:30:00+00:00"``
    # (RFC-non-conformant separator), so reject both.
    if "T" not in value:
        return False
    # codex r2 BLOCKING: RFC 3339 also requires a timezone offset (``Z``
    # or ``±HH:MM``). ``datetime.fromisoformat`` accepts a bare
    # ``2024-01-15T10:30:00`` as a naive datetime, which JSON-schema's
    # ``date-time`` format does NOT permit. Reject naive results.
    if dt.tzinfo is None:
        return False
    return True


_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+\-.]*:")
_FORMAT_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "email": _is_email,
    "uri": _is_uri,
    "uuid": _is_uuid,
    "date": _is_date,
    "date-time": _is_date_time,
}


def _uniqueitems_canonical(value: object) -> object:
    """Build a hashable key for ``value`` that respects JSON-schema's
    value-equality rules.

    codex r2 BLOCKING fix: ``json.dumps`` keying made ``1`` and ``1.0``
    distinct, but JSON-schema considers numerically-equal numbers as
    duplicates. Walk the structure recursively, normalising every
    numeric leaf to a ``float`` (with bools kept separate) and every
    dict / list to a tuple that hashable ``set()`` can compare.
    """
    if isinstance(value, bool):
        # Booleans are their own JSON type — keep separate from ints.
        return ("bool", value)
    if isinstance(value, (int, float)):
        # Normalise int / float to a single canonical numeric form so
        # ``1`` and ``1.0`` hash equal (per JSON-schema value equality).
        #
        # codex r3 BLOCKING: ``float(value)`` is lossy for large ints
        # (``9007199254740992`` and ``9007199254740993`` collapse to
        # the same float), so two genuinely distinct JSON integers
        # would be falsely flagged as duplicates. Use ``Decimal`` to
        # preserve the full numeric value while still hashing
        # ``Decimal("1")`` and ``Decimal("1.0")`` equal (their
        # internal coefficients differ but ``__hash__`` is computed
        # from the numeric value).
        from decimal import Decimal

        # ``Decimal(str(value))`` is the round-trip-safe constructor
        # — going through ``str`` preserves int exactness and reads
        # float values at full ``repr`` precision.
        return ("num", Decimal(str(value)))
    if isinstance(value, str):
        return ("str", value)
    if value is None:
        return ("null",)
    if isinstance(value, list):
        return ("arr", tuple(_uniqueitems_canonical(v) for v in value))
    if isinstance(value, dict):
        # Sort by key for structural-equality semantics regardless of
        # JSON object key order.
        return (
            "obj",
            tuple(sorted((k, _uniqueitems_canonical(v)) for k, v in value.items())),
        )
    # Unknown type — fall back to repr so ``set()`` still works without
    # raising ``TypeError``. Practically unreachable from ``json.loads``.
    return ("other", repr(value))
