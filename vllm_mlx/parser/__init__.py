# SPDX-License-Identifier: Apache-2.0
"""
Unified Parser layer — orchestrates reasoning + tool parsing for one
stream behind a single ``parse_delta`` entry point. Adapted from vLLM's
``vllm.parser`` package (Apache-2.0) — see ``abstract_parser.py`` header
for the adaptation notes.
"""

from vllm_mlx.parser.abstract_parser import (
    DelegatingParser,
    Parser,
    StreamState,
    _WrappedParser,
)
from vllm_mlx.parser.parser_manager import ParserManager

__all__ = [
    "DelegatingParser",
    "Parser",
    "ParserManager",
    "StreamState",
    "_WrappedParser",
]
