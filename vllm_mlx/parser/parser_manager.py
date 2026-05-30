# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vLLM's vllm/parser/parser_manager.py (Apache-2.0). Trimmed
# to drop the plugin-loader (``import_parser`` for arbitrary file paths)
# and the Llama 3.2 pythonic warning that doesn't apply to MLX targets.
"""
Registry + resolver for unified ``Parser`` implementations.

``ParserManager.get_parser(tool_parser_name, reasoning_parser_name, ...)``
tries three strategies in order:

1. A unified ``Parser`` registered under the same name as both the tool
   and reasoning parser (channel-routed models like Harmony land here).
2. A unified ``Parser`` registered under EITHER name.
3. Fall back to ``_WrappedParser`` composing the two individual parser
   classes.

Most rapid-mlx aliases hit strategy 3 today; strategy 1/2 are the
extension points for the Phase 2 channel-routed migration of
``OutputRouter``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_mlx.parser.abstract_parser import Parser
    from vllm_mlx.reasoning.base import ReasoningParser
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


class ParserManager:
    """Central registry for unified ``Parser`` implementations."""

    parsers: dict[str, type[Parser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}

    @classmethod
    def get_parser_internal(cls, name: str) -> type[Parser]:
        if name in cls.parsers:
            return cls.parsers[name]
        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)
        registered = ", ".join(cls.list_registered())
        raise KeyError(f"Parser '{name}' not found. Available parsers: {registered}")

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[Parser]:
        from vllm_mlx.parser.abstract_parser import Parser

        module_path, class_name = cls.lazy_parsers[name]
        mod = importlib.import_module(module_path)
        parser_cls = getattr(mod, class_name)
        if not issubclass(parser_cls, Parser):
            raise TypeError(f"{class_name} in {module_path} is not a Parser subclass.")
        cls.parsers[name] = parser_cls
        return parser_cls

    @classmethod
    def _register_module(
        cls,
        module: type[Parser],
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        from vllm_mlx.parser.abstract_parser import Parser

        if not issubclass(module, Parser):
            raise TypeError(
                f"module must be subclass of Parser, but got {type(module)}"
            )
        if module_name is None:
            module_names = [module.__name__]
        elif isinstance(module_name, str):
            module_names = [module_name]
        else:
            module_names = list(module_name)
        for name in module_names:
            if not force and name in cls.parsers:
                existed = cls.parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.parsers[name] = module

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[Parser] | None = None,
    ) -> type[Parser] | Callable[[type[Parser]], type[Parser]]:
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        def _decorator(obj: type[Parser]) -> type[Parser]:
            module_path = obj.__module__
            class_name = obj.__name__
            if isinstance(name, str):
                names = [name]
            elif name is not None:
                names = list(name)
            else:
                names = [class_name]
            for n in names:
                cls.lazy_parsers[n] = (module_path, class_name)
            return obj

        return _decorator

    @classmethod
    def list_registered(cls) -> list[str]:
        return sorted(set(cls.parsers.keys()) | set(cls.lazy_parsers.keys()))

    @classmethod
    def get_tool_parser(
        cls,
        tool_parser_name: str | None,
        enable_auto_tools: bool = False,
    ) -> type[ToolParser] | None:
        from vllm_mlx.tool_parsers import ToolParserManager

        if not enable_auto_tools or tool_parser_name is None:
            return None
        try:
            return ToolParserManager.get_tool_parser(tool_parser_name)
        except KeyError as e:
            raise TypeError(
                f"Tool parser '{tool_parser_name}' is not registered"
            ) from e

    @classmethod
    def get_reasoning_parser(
        cls, reasoning_parser_name: str | None
    ) -> type[ReasoningParser] | None:
        if not reasoning_parser_name:
            return None
        # Local import to avoid hard dep at module import time on the
        # reasoning package (keeps the parser layer importable in
        # contexts that load tool_parsers only).
        from vllm_mlx.reasoning import get_parser

        cls_ = get_parser(reasoning_parser_name)
        if cls_ is None:
            raise TypeError(
                f"Reasoning parser '{reasoning_parser_name}' is not registered"
            )
        return cls_

    @classmethod
    def get_parser(
        cls,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        enable_auto_tools: bool = False,
    ) -> type[Parser] | None:
        """Three-strategy resolver — see module docstring."""
        from vllm_mlx.parser.abstract_parser import _WrappedParser

        if not tool_parser_name and not reasoning_parser_name:
            return None

        # Strategy 1: same name registered as a unified Parser
        if tool_parser_name and tool_parser_name == reasoning_parser_name:
            try:
                return cls.get_parser_internal(tool_parser_name)
            except KeyError:
                pass

        # Strategy 2: either name registered as a unified Parser
        for name in (tool_parser_name, reasoning_parser_name):
            if name:
                try:
                    return cls.get_parser_internal(name)
                except KeyError:
                    pass

        # Strategy 3: compose a _WrappedParser subclass from the
        # individual classes. A fresh subclass per resolve avoids the
        # class-attribute-mutation race: if two callers resolved
        # different (reasoning, tool) pairs before either instantiated
        # their result, mutating ``_WrappedParser`` itself would make
        # both pick up whichever pair was assigned last.
        reasoning_cls = cls.get_reasoning_parser(reasoning_parser_name)
        tool_cls = cls.get_tool_parser(tool_parser_name, enable_auto_tools)
        if reasoning_cls is None and tool_cls is None:
            return None

        subclass_name = (
            f"_WrappedParser__"
            f"{reasoning_parser_name or 'none'}__{tool_parser_name or 'none'}"
        )
        return type(
            subclass_name,
            (_WrappedParser,),
            {
                "reasoning_parser_cls": reasoning_cls,
                "tool_parser_cls": tool_cls,
            },
        )


__all__ = ["ParserManager"]
