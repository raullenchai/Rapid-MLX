# SPDX-License-Identifier: Apache-2.0
"""
Abstract tool parser base class and manager for rapid-mlx.

Inspired by vLLM's tool parser architecture but simplified for MLX backend.
"""

import importlib
import json
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from transformers import PreTrainedTokenizerBase

# Pattern to match and strip think tags
# Handles two cases:
# 1. Full tags: <think>...</think>
# 2. Only closing tag: ...content before...</think> (when <think> is in prompt)
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
IMPLICIT_THINK_PATTERN = re.compile(r"^.*?</think>", re.DOTALL)

# General fallback: model outputs tool calls as text instead of structured format.
# Common degradation at low quantization after multiple tool rounds.
# Variant 1: [Calling tool="name" param1="val1" param2="val2"]
TEXT_TOOL_CALL_KV_PATTERN = re.compile(
    r'\[Calling\s+tool="([^"]+)"((?:\s+\w+="(?:[^"\\]|\\.)*")*)\s*\]'
)
TEXT_TOOL_CALL_KV_PARAM = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
# Variant 2: [Calling tool: name(json_args)]  or  [Calling tool: name({...})]
TEXT_TOOL_CALL_FN_PATTERN = re.compile(r"\[Calling\s+tool:\s*(\w+)\((\{.*?\})\)\s*\]")
# Combined check for either variant
TEXT_TOOL_CALL_ANY = re.compile(r"\[Calling\s+tool[=:]")


@dataclass
class ExtractedToolCallInformation:
    """Information extracted from model output about tool calls."""

    tools_called: bool
    """Whether any tool calls were detected."""

    tool_calls: list[dict[str, Any]]
    """List of tool calls with 'name' and 'arguments' fields."""

    content: str | None = None
    """Any content that wasn't part of tool calls."""


# Canonical wire-format labels each ToolParser subclass declares it handles.
# Adding a new label here is a deliberate act — the audit script and the
# parity test (test_tool_call_streaming_parity.py) cross-reference these
# strings, so reusing existing labels keeps the surfaces aligned. Define
# new labels only when an actually-novel wire format is introduced.
#
# Canonical formats (and which parsers handle them):
#   tool_call_json        — <tool_call>{"name":...,"arguments":...}</tool_call>
#                           (Hermes, Qwen, Qwen3-Coder JSON variant)
#   tool_call_xml_body    — <tool_call><function=name><parameter=p>v</parameter>
#                           </function></tool_call>  (Qwen3.6, Nemotron, Hermes
#                           fallback, Qwen3-Coder XML variant)
#   function_bare         — <function=name>...</function>  without <tool_call>
#                           wrapper (Hermes BARE_FUNCTION_PATTERN)
#   raw_json              — bare {"name":...,"arguments":...} (Hermes fallback,
#                           xLAM, Llama JSON variant)
#   calling_tool_text     — [Calling tool="name" k="v"]  text fallback for
#                           low-quant degradation
#   gemma4_native         — <|tool_call>call:name{k:v}<tool_call|>  (Gemma 4)
#   harmony_commentary    — <|channel|>commentary to=functions.X<|message|>
#                           {...}<|call|>  (GPT-OSS / Harmony)
#   mistral_tool_calls    — [TOOL_CALLS][{"name":...,"arguments":...}]
#   llama_python_tag      — <|python_tag|>{"name":..."parameters":...}
#   glm_named_tool_call   — <tool_call>name\n{json}</tool_call>  (GLM-4.5/4.7)
#   functionary_native    — <|from|>assistant<|recipient|>name<|content|>...
#   granite_native        — <|tool_call|>[{...}]  (Granite 3/4)
#   kimi_native           — <|tool_calls_section_begin|>...<|tool_calls_section_end|>
#   minimax_native        — <tool_calls>[...]</tool_calls>  (MiniMax M2/M2.5)
#   seed_oss_native       — Seed-OSS specific (TBD; placeholder)
#   deepseek_native       — DeepSeek V3 specific
#   deepseek_v31_native   — DeepSeek V3.1 / R1-0528 specific
#   qwen3_coder_xml_named — Qwen3-Coder XML variant with named function tags
WIRE_FORMAT_LABELS: frozenset[str] = frozenset(
    {
        "tool_call_json",
        "tool_call_xml_body",
        "function_bare",
        "raw_json",
        "calling_tool_text",
        "gemma4_native",
        "harmony_commentary",
        "mistral_tool_calls",
        "llama_python_tag",
        "glm_named_tool_call",
        "functionary_native",
        "granite_native",
        "kimi_native",
        "minimax_native",
        "seed_oss_native",
        "deepseek_native",
        "deepseek_v31_native",
        "qwen3_coder_xml_named",
    }
)


class ToolParser(ABC):
    """
    Abstract base class for tool call parsers.

    Each parser implementation handles a specific model's tool calling format.
    """

    # Class attribute to declare native format support.
    # Set to True in subclasses whose corresponding model chat templates
    # can handle role="tool" messages and tool_calls fields directly,
    # without needing conversion to text format.
    SUPPORTS_NATIVE_TOOL_FORMAT: bool = False

    # Declarative list of wire formats this parser handles. Every concrete
    # subclass MUST override this with at least one label from
    # ``WIRE_FORMAT_LABELS``. The structural test
    # ``tests/test_tool_parser_wire_formats.py::test_every_parser_declares_formats``
    # enforces this. Forcing function for the #425-class meta-fix: when a
    # new parser ships, the wire format(s) it handles MUST be documented
    # at the class level rather than buried in the regex patterns. This is
    # what makes the audit + parity matrices machine-checkable rather than
    # reading-the-source archeology.
    EXPECTED_WIRE_FORMATS: tuple[str, ...] = ()

    @classmethod
    def supports_native_format(cls) -> bool:
        """
        Check if this parser supports native tool message format.

        Native format means the parser's corresponding model chat template
        can handle:
        - role="tool" messages directly (not converted to role="user")
        - tool_calls field on assistant messages (not converted to text)

        Returns:
            True if native format is supported
        """
        return cls.SUPPORTS_NATIVE_TOOL_FORMAT

    @staticmethod
    def strip_think_tags(text: str) -> str:
        """
        Strip think tags from text.

        Handles two scenarios:
        1. Full tags: <think>...</think> in output
        2. Only closing tag: ...</think> when <think> was in prompt

        Used as fallback when no reasoning parser is configured but the model
        produces thinking tags. This prevents tool parsing failures with
        models that use thinking tags (e.g., Ring-Mini-Linear-2.0 with hermes).

        Args:
            text: Model output that may contain think tags

        Returns:
            Text with think tags removed
        """
        # First try to strip full tags
        result = THINK_TAG_PATTERN.sub("", text)

        # If no full tags found but </think> exists, strip implicit think
        # (when <think> was injected in the prompt)
        if result == text and "</think>" in text:
            result = IMPLICIT_THINK_PATTERN.sub("", text)

        return result.strip()

    def __init__(self, tokenizer: PreTrainedTokenizerBase | None = None):
        """
        Initialize the tool parser.

        Args:
            tokenizer: The tokenizer for the model (optional, some parsers need it)
        """
        self.model_tokenizer = tokenizer
        # State for streaming parsing
        self.current_tool_id: int = -1
        self.prev_tool_call_arr: list[dict] = []

    @cached_property
    def vocab(self) -> dict[str, int]:
        """Get the tokenizer vocabulary."""
        if self.model_tokenizer is None:
            return {}
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model response.

        Args:
            model_output: The complete model output string
            request: Optional request context (for tool definitions, etc.)

        Returns:
            ExtractedToolCallInformation with parsed tool calls
        """
        raise NotImplementedError

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract tool calls from streaming model output.

        Override this method for streaming support. Default implementation
        returns None (no streaming support).

        Args:
            previous_text: Text before this delta
            current_text: Complete text so far
            delta_text: New text in this chunk
            previous_token_ids: Token IDs before this delta
            current_token_ids: All token IDs so far
            delta_token_ids: New token IDs in this chunk
            request: Optional request context

        Returns:
            Delta message dict with content and/or tool_calls, or None
        """
        return None

    def has_pending_tool_call(self, text: str) -> bool:
        """Check if text contains incomplete tool call markup.

        Used as a fallback when streaming ends before the parser's closing
        tag arrives.  Subclasses should override for non-standard markers.
        """
        return "<tool_call>" in text or self.has_text_format_tool_call(text)

    def flush_held_content(self, full_text: str) -> str:
        """Return any prefix-held content suffix to release at stream end.

        Streaming parsers that prefix-hold partial tool-call sentinels
        (e.g. ``<``, ``<|``, ``<func``...) can end the stream still
        holding bytes that turned out not to be sentinel openers. Those
        bytes are ordinary content but were never emitted via
        ``extract_tool_calls_streaming`` because the parser couldn't be
        sure no further matching char would arrive.

        Subclasses that prefix-hold MUST override this to return the
        held suffix of ``full_text`` (typically computed as
        ``full_text[len(_safe_content_prefix(full_text)):]``). The
        postprocessor calls this in ``finalize()`` when no tool calls
        fired and emits the result as a final content event so the
        last few characters of plain content aren't dropped (codex
        round-3 CRITICAL).

        Default returns empty string — parsers that don't prefix-hold
        have nothing held to release.
        """
        return ""

    # -----------------------------------------------------------------
    # General text-format tool call fallback
    # -----------------------------------------------------------------
    # Models at low quantization sometimes degrade after multiple tool
    # rounds and output tool calls as plain text instead of structured
    # format.  These methods detect and convert the common
    # [Calling tool="name" key="value" ...] pattern.

    @staticmethod
    def has_text_format_tool_call(text: str) -> bool:
        """Check if text contains a text-format tool call.

        Detects two common degradation patterns:
          [Calling tool="name" key="value" ...]
          [Calling tool: name({json})]
        """
        return TEXT_TOOL_CALL_ANY.search(text) is not None

    @staticmethod
    def extract_text_format_tool_calls(text: str) -> list[dict[str, Any]]:
        """Extract tool calls from text-format patterns.

        Handles two variants:
          Variant 1: [Calling tool="name" key="value" key2="value2"]
          Variant 2: [Calling tool: name({"key": "value"})]

        Returns list of dicts with 'id', 'name', 'arguments' keys.
        """
        tool_calls: list[dict[str, Any]] = []

        # Variant 1: key="value" pairs
        for match in TEXT_TOOL_CALL_KV_PATTERN.finditer(text):
            func_name = match.group(1)
            params_str = match.group(2)
            arguments: dict[str, Any] = {}
            for pm in TEXT_TOOL_CALL_KV_PARAM.finditer(params_str):
                key = pm.group(1)
                value = pm.group(2).replace('\\"', '"')
                try:
                    arguments[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = value
            if arguments:
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": func_name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )

        # Variant 2: name({json})
        for match in TEXT_TOOL_CALL_FN_PATTERN.finditer(text):
            func_name = match.group(1)
            json_str = match.group(2)
            try:
                arguments = json.loads(json_str)
                if isinstance(arguments, dict) and arguments:
                    tool_calls.append(
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": func_name.strip(),
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        }
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        return tool_calls

    def reset(self) -> None:
        """Reset parser state for a new request."""
        self.current_tool_id = -1
        self.prev_tool_call_arr = []


class ToolParserManager:
    """
    Central registry for ToolParser implementations.

    Supports both eager and lazy registration of tool parsers.
    """

    tool_parsers: dict[str, type[ToolParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def get_tool_parser(cls, name: str) -> type[ToolParser]:
        """
        Retrieve a registered ToolParser class by name.

        Args:
            name: Parser name (e.g., 'mistral', 'qwen', 'llama')

        Returns:
            The ToolParser class

        Raises:
            KeyError: If parser not found
        """
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        raise KeyError(
            f"Tool parser '{name}' not found. "
            f"Available parsers: {cls.list_registered()}"
        )

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ToolParser]:
        """Import and register a lazily loaded parser."""
        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ToolParser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a ToolParser subclass."
                )
            cls.tool_parsers[name] = parser_cls
            return parser_cls
        except Exception as e:
            raise ImportError(
                f"Failed to import tool parser '{name}' from {module_path}: {e}"
            ) from e

    @classmethod
    def register_module(
        cls,
        name: str | list[str],
        module: type[ToolParser] | None = None,
        force: bool = True,
    ) -> type[ToolParser] | None:
        """
        Register a ToolParser class.

        Can be used as a decorator or direct call.

        Usage:
            @ToolParserManager.register_module("my_parser")
            class MyToolParser(ToolParser):
                ...

            # Or direct registration:
            ToolParserManager.register_module("my_parser", MyToolParser)
        """
        names = [name] if isinstance(name, str) else name

        if module is not None:
            # Direct registration
            if not issubclass(module, ToolParser):
                raise TypeError(
                    f"module must be subclass of ToolParser, got {type(module)}"
                )
            for n in names:
                if not force and n in cls.tool_parsers:
                    raise KeyError(f"Parser '{n}' is already registered")
                cls.tool_parsers[n] = module
            return module

        # Decorator usage
        def decorator(parser_cls: type[ToolParser]) -> type[ToolParser]:
            for n in names:
                if not force and n in cls.tool_parsers:
                    raise KeyError(f"Parser '{n}' is already registered")
                cls.tool_parsers[n] = parser_cls
            return parser_cls

        return decorator  # type: ignore

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        Register a lazy module mapping for deferred loading.

        Args:
            name: Parser name to register
            module_path: Full module path (e.g., 'vllm_mlx.tool_parsers.mistral')
            class_name: Class name within the module
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all registered tool parsers."""
        return sorted(set(cls.tool_parsers.keys()) | set(cls.lazy_parsers.keys()))
