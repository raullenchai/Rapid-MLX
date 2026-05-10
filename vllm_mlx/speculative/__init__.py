# SPDX-License-Identifier: Apache-2.0
"""
Speculative decoding utilities for vllm-mlx.
"""

from .ngram_drafter import (
    NgramRequestState,
    ThinkStateTracker,
    lookup_think_token_ids,
)
from .prompt_lookup import PromptLookupDecoder

__all__ = [
    "NgramRequestState",
    "PromptLookupDecoder",
    "ThinkStateTracker",
    "lookup_think_token_ids",
]
