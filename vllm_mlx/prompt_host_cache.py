# SPDX-License-Identifier: Apache-2.0
"""Bounded process-local cache for immutable prompt host work.

Chat rendering and tokenization are CPU work that is repeated before the
device prefix cache can even be queried.  This cache keeps those two exact
results independent from KV/GDN/QSA state: values are immutable strings or
integer tuples, and any identity uncertainty is a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

_DEFAULT_MAX_ENTRIES = 64
_DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def prompt_host_cache_enabled() -> bool:
    return os.environ.get("RAPID_MLX_PROMPT_HOST_CACHE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _fingerprint(value: Any) -> str | None:
    """Return an order-preserving JSON digest, or ``None`` when unsafe."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            # Template mappings can be observably order-sensitive.  Sorting
            # would alias two inputs that a Jinja template may render apart.
            sort_keys=False,
        ).encode()
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _HostEntry:
    kind: str
    value: str | tuple[int, ...]
    size_bytes: int


class PromptHostCache:
    """Thread-safe LRU for exact render and tokenize results.

    Both an entry cap and a byte cap are enforced.  Oversized prompts simply
    bypass retention, so a large document cannot evict the process into host
    memory pressure merely because it was submitted repeatedly.
    """

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        *,
        enabled: bool | None = None,
    ) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("prompt host cache limits must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.enabled = prompt_host_cache_enabled() if enabled is None else bool(enabled)
        self._entries: OrderedDict[tuple[str, str], _HostEntry] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.RLock()
        self._stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "oversize_bypasses": 0,
            "uncacheable_bypasses": 0,
            "invalidations": 0,
        }
        self._hits_by_kind = {"render": 0, "tokens": 0}

    @staticmethod
    def fingerprint(value: Any) -> str | None:
        return _fingerprint(value)

    def _get(self, kind: str, fingerprint: str | None) -> str | tuple[int, ...] | None:
        if not self.enabled or fingerprint is None:
            if self.enabled and fingerprint is None:
                with self._lock:
                    self._stats["uncacheable_bypasses"] += 1
            return None
        key = (kind, fingerprint)
        with self._lock:
            self._stats["lookups"] += 1
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            self._entries.move_to_end(key)
            self._stats["hits"] += 1
            self._hits_by_kind[kind] += 1
            return entry.value

    def _put(
        self,
        kind: str,
        fingerprint: str | None,
        value: str | tuple[int, ...],
        size_bytes: int,
    ) -> bool:
        if not self.enabled or fingerprint is None:
            return False
        size_bytes = int(size_bytes)
        if size_bytes > self.max_bytes:
            with self._lock:
                self._stats["oversize_bypasses"] += 1
            return False
        key = (kind, fingerprint)
        entry = _HostEntry(kind, value, size_bytes)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._current_bytes -= previous.size_bytes
            self._entries[key] = entry
            self._current_bytes += size_bytes
            self._stats["stores"] += 1
            while (
                len(self._entries) > self.max_entries
                or self._current_bytes > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._current_bytes -= evicted.size_bytes
                self._stats["evictions"] += 1
        return True

    def get_render(self, fingerprint: str | None) -> str | None:
        value = self._get("render", fingerprint)
        return value if isinstance(value, str) else None

    def put_render(self, fingerprint: str | None, prompt: str) -> bool:
        return self._put("render", fingerprint, prompt, len(prompt.encode("utf-8")))

    def get_tokens(self, fingerprint: str | None) -> list[int] | None:
        value = self._get("tokens", fingerprint)
        return list(value) if isinstance(value, tuple) else None

    def put_tokens(self, fingerprint: str | None, tokens: list[int]) -> bool:
        frozen = tuple(int(token) for token in tokens)
        # A tuple holds one pointer per item and Python integer objects are
        # typically 28 bytes on 64-bit CPython.  Forty bytes/token is a
        # conservative process-memory ledger, not a wire-format claim.
        return self._put("tokens", fingerprint, frozen, 40 * len(frozen))

    def clear(self, *, reset_stats: bool = False) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._current_bytes = 0
            if reset_stats:
                for name in self._stats:
                    self._stats[name] = 0
                for kind in self._hits_by_kind:
                    self._hits_by_kind[kind] = 0
            else:
                self._stats["invalidations"] += count
            return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "enabled": self.enabled,
                "entries": len(self._entries),
                "current_bytes": self._current_bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "hits_by_kind": dict(self._hits_by_kind),
            }
