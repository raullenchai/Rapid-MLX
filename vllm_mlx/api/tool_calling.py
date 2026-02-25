# SPDX-License-Identifier: Apache-2.0
"""
Tool calling parsing and conversion utilities.

Supports parsing tool calls from multiple model formats:
- Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- Llama: <function=name>{"arg": "value"}</function>

Also includes structured output (JSON Schema) utilities:
- parse_json_output: Extract JSON from model output
- validate_json_schema: Validate JSON against a schema
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from jsonschema import validate, ValidationError

from .models import FunctionCall, ResponseFormat, ToolCall


def _is_tool_call_json(obj: dict) -> bool:
    """
    Check if a JSON object looks like a tool call.

    A tool call must have:
    - "name" key with a string value (function name)
    - "arguments" key (the function arguments)

    This prevents false positives where regular JSON like {"name": "John", "age": 30}
    would be incorrectly parsed as a tool call.

    Args:
        obj: JSON object to check

    Returns:
        True if object appears to be a tool call
    """
    if not isinstance(obj, dict):
        return False

    # Must have both "name" and "arguments" keys
    if "name" not in obj or "arguments" not in obj:
        return False

    # "name" must be a non-empty string (function name)
    if not isinstance(obj["name"], str) or not obj["name"].strip():
        return False

    # "arguments" must be a dict or string
    args = obj["arguments"]
    if not isinstance(args, (dict, str)):
        return False

    return True


def _parse_raw_json_tool_calls(text: str) -> Optional[List[dict]]:
    """
    Parse raw JSON tool calls from model output.

    Handles:
    - Single JSON object: {"name": "func", "arguments": {...}}
    - Multiple objects separated by commas: {...}, {...}
    - JSON array: [{...}, {...}]

    Only objects with BOTH "name" AND "arguments" keys are considered tool calls.
    This prevents false positives with regular JSON objects.

    Args:
        text: Raw model output text

    Returns:
        List of tool call dicts with 'name' and 'arguments', or None if no valid tool calls found
    """
    if not text:
        return None

    text = text.strip()

    # Try JSON array first
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(
                _is_tool_call_json(item) for item in parsed
            ):
                return [
                    {"name": item["name"], "arguments": item.get("arguments", {})}
                    for item in parsed
                ]
        except json.JSONDecodeError:
            pass

    # Find JSON objects with balanced braces
    tool_calls = []
    depth = 0
    start = None

    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                json_str = text[start : i + 1]
                try:
                    obj = json.loads(json_str)
                    # Only consider as tool call if it has both "name" AND "arguments"
                    if _is_tool_call_json(obj):
                        tool_calls.append(
                            {"name": obj["name"], "arguments": obj.get("arguments", {})}
                        )
                except json.JSONDecodeError:
                    pass
                start = None

    return tool_calls if tool_calls else None


def parse_tool_calls(
    text: str, request: dict[str, Any] | None = None
) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse tool calls from model output.

    Supports multiple formats:
    - Qwen3 bracket: [Calling tool: function_name({"arg": "value"})]
    - Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - Llama: <function=name>{"arg": "value"}</function>
    - Nemotron: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    - Raw JSON: {"name": "...", "arguments": {...}} (single or multiple)

    Args:
        text: Raw model output text

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
        - cleaned_text: Text with tool call tags removed
        - tool_calls: List of ToolCall objects, or None if no tool calls found
    """
    tool_calls = []
    cleaned_text = text

    # Pattern for Qwen3 bracket-style: [Calling tool: function_name({...})]
    bracket_pattern = r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]"
    bracket_matches = re.findall(bracket_pattern, text, re.DOTALL)

    for name, args_str in bracket_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Remove bracket tool calls from cleaned text
    if bracket_matches:
        cleaned_text = re.sub(
            r"\[Calling tool:\s*\w+\(\{.*?\}\)\]", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Pattern for Nemotron-style: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    nemotron_pattern = (
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>"
    )
    nemotron_matches = re.findall(nemotron_pattern, text, re.DOTALL)

    for name, params_block in nemotron_matches:
        # Parse parameters from <parameter=name>value</parameter> format
        param_pattern = r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>"
        params = re.findall(param_pattern, params_block, re.DOTALL)
        arguments = {}
        for p_name, p_value in params:
            val = p_value.strip()
            try:
                arguments[p_name.strip()] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                arguments[p_name.strip()] = val

        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name.strip(), arguments=json.dumps(arguments)
                ),
            )
        )

    # Remove Nemotron tool call tags from cleaned text
    if nemotron_matches:
        cleaned_text = re.sub(
            r"<tool_call>\s*<function=[^>]+>.*?</function>\s*</tool_call>",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

    # Pattern for Qwen-style tool calls: <tool_call>{"json"}</tool_call>
    qwen_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    qwen_matches = re.findall(qwen_pattern, cleaned_text, re.DOTALL)

    for match in qwen_matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Remove Qwen tool call tags from cleaned text
    if qwen_matches:
        cleaned_text = re.sub(
            r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Pattern for Llama-style: <function=name>{"json"}</function>
    llama_pattern = r"<function=([^>]+)>(\{.*?\})</function>"
    llama_matches = re.findall(llama_pattern, cleaned_text, re.DOTALL)

    for name, args_str in llama_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    if llama_matches:
        cleaned_text = re.sub(
            r"<function=[^>]+>\{.*?\}</function>", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Note: We keep <think>...</think> tags for reasoning models
    # The user may want to see the model's reasoning process

    # Fallback: Raw JSON tool calls (lowest priority)
    # Only try if no other formats matched
    if not tool_calls:
        raw_json_calls = _parse_raw_json_tool_calls(cleaned_text)
        if raw_json_calls:
            for call_data in raw_json_calls:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=call_data["name"],
                            arguments=(
                                json.dumps(call_data["arguments"])
                                if isinstance(call_data["arguments"], dict)
                                else str(call_data["arguments"])
                            ),
                        ),
                    )
                )
            # Clean the JSON from text since we parsed it as tool calls
            cleaned_text = ""

    return cleaned_text, tool_calls if tool_calls else None


def convert_tools_for_template(tools: Optional[List]) -> Optional[List[dict]]:
    """
    Convert OpenAI tools format to format expected by tokenizer.apply_chat_template.

    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Template format (commonly used by models):
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Args:
        tools: List of ToolDefinition objects or dicts in OpenAI format

    Returns:
        List of tool definitions in template format, or None if no tools
    """
    if not tools:
        return None

    converted = []
    for tool in tools:
        # Handle both Pydantic models and dicts
        if isinstance(tool, dict):
            tool_type = tool.get("type")
            tool_func = tool.get("function")
        else:
            tool_type = getattr(tool, "type", None)
            tool_func = getattr(tool, "function", None)

        if tool_type == "function" and tool_func:
            # Handle function as dict or Pydantic model
            if isinstance(tool_func, dict):
                func_name = tool_func.get("name", "")
                func_desc = tool_func.get("description", "")
                func_params = tool_func.get(
                    "parameters", {"type": "object", "properties": {}}
                )
            else:
                func_name = getattr(tool_func, "name", "")
                func_desc = getattr(tool_func, "description", "")
                func_params = getattr(
                    tool_func, "parameters", {"type": "object", "properties": {}}
                )

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": func_desc,
                        "parameters": func_params,
                    },
                }
            )

    return converted if converted else None


def format_tool_call_for_message(tool_call: ToolCall) -> dict:
    """
    Format a ToolCall object for inclusion in a message.

    Args:
        tool_call: ToolCall object

    Returns:
        Dict representation suitable for message content
    """
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


# =============================================================================
# Structured Output (JSON Schema) Utilities
# =============================================================================


def validate_json_schema(
    data: Any, schema: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON data against a JSON Schema.

    Args:
        data: The JSON data to validate (dict, list, etc.)
        schema: JSON Schema specification

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if data matches schema
        - error_message: Error description if invalid, None if valid
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e.message)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model output text.

    Tries multiple strategies:
    1. Parse entire text as JSON
    2. Extract JSON from markdown code blocks
    3. Find JSON object/array in text

    Args:
        text: Raw model output text

    Returns:
        Parsed JSON data, or None if no valid JSON found
    """
    text = text.strip()

    # Strategy 1: Try to parse entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON object or array in text
    # Look for { ... } or [ ... ]
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def parse_json_output(
    text: str, response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
) -> Tuple[str, Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Parse JSON from model output when response_format is set.

    Args:
        text: Raw model output text
        response_format: ResponseFormat specification (optional)
            - If type="json_object", extracts any valid JSON
            - If type="json_schema", extracts and validates against schema

    Returns:
        Tuple of (cleaned_text, parsed_json, is_valid, error_message)
        - cleaned_text: Original text (preserved for reference)
        - parsed_json: Extracted JSON data, or None if extraction failed
        - is_valid: True if JSON is valid (and matches schema if specified)
        - error_message: Error description if invalid, None if valid
    """
    # Handle None or text format - just return original
    if response_format is None:
        return text, None, True, None

    # Normalize response_format to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    # text format - no JSON extraction
    if format_type == "text":
        return text, None, True, None

    # json_object or json_schema - extract JSON
    parsed = extract_json_from_text(text)

    if parsed is None:
        return text, None, False, "Failed to extract valid JSON from output"

    # json_object - just verify it's valid JSON (already done by extraction)
    if format_type == "json_object":
        return text, parsed, True, None

    # json_schema - validate against schema
    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})

        if schema:
            is_valid, error = validate_json_schema(parsed, schema)
            if not is_valid:
                return text, parsed, False, f"JSON Schema validation failed: {error}"

        return text, parsed, True, None

    # Unknown format type - treat as text
    return text, None, True, None


def build_json_system_prompt(
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Build a system prompt instruction for JSON output.

    For models without native JSON mode support, this adds instructions
    to the prompt to encourage proper JSON formatting.

    Args:
        response_format: ResponseFormat specification

    Returns:
        System prompt instruction string, or None if not needed
    """
    if response_format is None:
        return None

    # Normalize to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    if format_type == "text":
        return None

    if format_type == "json_object":
        return (
            "⚠️ JSON OUTPUT REQUIRED ⚠️\n\n"
            "You MUST respond with ONLY a valid JSON object.\n\n"
            "RULES:\n"
            "- Start response with { or [\n"
            "- NO text before or after JSON\n"
            "- NO thinking or explanations\n"
            "- NO markdown code blocks (```)\n"
            "- ONLY the raw JSON object"
        )

    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})
        name = json_schema_spec.get("name", "response")
        description = json_schema_spec.get("description", "")
        strict = json_schema_spec.get("strict", False)

        # If strict mode is enabled and guided generation is available,
        # the server should use guided decoding instead of prompt injection
        if strict:
            # Return stronger instruction for strict mode
            prompt = (
                f"⚠️ STRICT JSON OUTPUT REQUIRED ⚠️\n\n"
                f"You MUST respond with ONLY a valid JSON object matching the '{name}' schema.\n"
            )
            if description:
                prompt += f"Purpose: {description}\n"
            prompt += (
                f"\nJSON Schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
                "STRICT RULES:\n"
                "- Start response with {{ or [\n"
                "- NO text before or after JSON\n"
                "- NO thinking, reasoning, or explanations\n"
                "- NO markdown code blocks\n"
                "- ONLY the JSON object"
            )
            return prompt

        # Standard (non-strict) mode - still strong but less aggressive
        prompt = (
            f"⚠️ JSON OUTPUT REQUIRED ⚠️\n\n"
            f"Respond with a valid JSON object matching the '{name}' schema.\n"
        )
        if description:
            prompt += f"Purpose: {description}\n"
        prompt += (
            f"\nJSON Schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            "RULES:\n"
            "- Start response with { or [\n"
            "- NO text before or after JSON\n"
            "- NO markdown code blocks\n"
            "- ONLY the JSON object"
        )
        return prompt

    return None


def extract_json_schema_for_guided(
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract JSON schema from response_format for guided generation.

    Returns the schema dict if response_format specifies json_schema type,
    otherwise returns None.

    Args:
        response_format: ResponseFormat specification

    Returns:
        JSON schema dict or None
    """
    if response_format is None:
        return None

    # Normalize to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    if format_type != "json_schema":
        return None

    json_schema_spec = rf_dict.get("json_schema", {})
    schema = json_schema_spec.get("schema", {})

    if not schema:
        return None

    return schema
