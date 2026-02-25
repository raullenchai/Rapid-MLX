# SPDX-License-Identifier: Apache-2.0
"""
Abstract tool parser base class and manager for vllm-mlx.

Inspired by vLLM's tool parser architecture but simplified for MLX backend.
"""

import importlib
import re
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


@dataclass
class ExtractedToolCallInformation:
    """Information extracted from model output about tool calls."""

    tools_called: bool
    """Whether any tool calls were detected."""

    tool_calls: list[dict[str, Any]]
    """List of tool calls with 'name' and 'arguments' fields."""

    content: str | None = None
    """Any content that wasn't part of tool calls."""


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
        return "<tool_call>" in text

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
