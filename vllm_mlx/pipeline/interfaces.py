# SPDX-License-Identifier: Apache-2.0
"""Pipeline stage interfaces — the contracts that plugins implement.

Each interface defines a single responsibility in the inference pipeline.
Default implementations use mlx-lm's public API as a black box.
Optimizations (MTP, speculative decode, TurboQuant) implement the same
interface with enhanced behavior.

Architecture::

    Request → Tokenize → CacheStrategy → PrefillStrategy → DecodeStrategy → Detokenize → Response
                              ↑                ↑                  ↑
                         - LRU cache      - Standard         - Standard
                         - TurboQuant     - Chunked          - MTP
                         - RadixTree      - Paged            - Speculative
                                          - Offload          - Medusa
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ── Data types ─────────────────────────────────────────────────────


@dataclass
class TokenResult:
    """Result from one decode step for one sequence."""

    uid: int
    token: int
    logprobs: Any = None  # mx.array or None
    finish_reason: str | None = None
    prompt_cache: list | None = None  # extractable cache for prefix reuse


@dataclass
class CacheResult:
    """Result from prefix cache lookup."""

    hit: bool = False
    cache: Any = None  # KV cache state to pass to prefill
    cached_tokens: int = 0
    remaining_tokens: list[int] = field(default_factory=list)


@dataclass
class DecodeRequest:
    """Request to insert into the decode stage.

    The ``uid`` field is ignored by StandardDecode — BatchGenerator
    assigns its own UIDs.  The assigned UID is returned by
    ``DecodeStrategy.insert()``.
    """

    tokens: list[int]
    max_tokens: int
    sampler: Callable | None = None
    logits_processors: list[Callable] | None = None
    cache: Any = None  # pre-computed KV cache from prefill/cache lookup


# ── Stage interfaces ───────────────────────────────────────────────


class CacheStrategy(ABC):
    """Prefix cache — look up and store KV state by token prefix.

    Implementations: LRUPrefixCache, BlockAwarePrefixCache,
    TurboQuantCache (future), RadixTreeCache (future).
    """

    @abstractmethod
    def lookup(self, token_ids: list[int]) -> CacheResult:
        """Find cached KV state for a token prefix."""
        ...

    @abstractmethod
    def store(self, token_ids: list[int], cache: Any) -> None:
        """Store KV state for future reuse."""
        ...

    def invalidate(self, token_ids: list[int]) -> bool:
        """Invalidate a specific cache entry. Returns True if found."""
        return False

    def evict(self, n: int = 1) -> int:
        """Evict entries to free memory. Returns count evicted."""
        return 0


class DecodeStrategy(ABC):
    """Token generation — the core decode loop.

    Default implementation wraps mlx-lm BatchGenerator's public API
    (insert/next/remove). Optimized implementations add MTP, speculative
    decode, Medusa heads, etc. on top.
    """

    @abstractmethod
    def insert(self, request: DecodeRequest) -> int:
        """Insert a request for generation. Returns assigned UID."""
        ...

    @abstractmethod
    def step(self) -> list[TokenResult]:
        """Run one decode step. Returns tokens for all active sequences."""
        ...

    @abstractmethod
    def remove(self, uid: int) -> Any | None:
        """Remove a sequence. Returns its prompt cache if available."""
        ...

    @abstractmethod
    def has_active(self) -> bool:
        """Whether there are active sequences being decoded."""
        ...

    def close(self) -> None:  # noqa: B027
        """Release resources."""


class DecodePlugin(ABC):
    """Optional decode-time optimization that wraps a DecodeStrategy.

    Plugins can intercept step() to add speculative tokens, modify
    sampling, or implement custom decode logic. They compose via
    wrapping: MTPPlugin(StandardDecode(...)).

    Lifecycle: on_insert → wrap_step (repeated) → on_remove → on_close
    """

    @abstractmethod
    def wrap_step(
        self, base_step: Callable[[], list[TokenResult]]
    ) -> list[TokenResult]:
        """Wrap the base decode step with custom logic."""
        ...

    def on_insert(self, request: DecodeRequest, uid: int) -> None:  # noqa: B027
        """Called when a new sequence is inserted. Set up per-sequence state."""

    def on_remove(self, uid: int) -> None:  # noqa: B027
        """Called when a sequence is removed. Clean up per-sequence state."""

    def on_close(self) -> None:  # noqa: B027
        """Called when the decode strategy is closed. Release plugin resources."""
