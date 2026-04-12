"""Model Registry — manage multiple loaded models with request-level routing.

Supports the multi-model pattern needed by modern AI agents:
  - Hermes: cheap model + strong model routing
  - Aider: architect mode (plan model + edit model)
  - OpenClaude: per-agent model routing

Usage:
    registry = ModelRegistry()
    registry.add("qwen3.5-4b", engine, is_default=True)
    registry.add("qwen3.5-27b", engine2)

    # Request routing
    engine = registry.get_engine("qwen3.5-27b")  # specific
    engine = registry.get_engine("default")       # default
    engine = registry.get_engine(None)             # default
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """A loaded model with its metadata."""

    engine: object  # BaseEngine — use object to avoid circular import
    model_name: str  # canonical name (e.g., full HF path)
    model_path: str  # actual path on disk or HF ID
    aliases: set[str] = field(default_factory=set)  # alternative names
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    is_mllm: bool = False
    max_tokens: int = 4096

    def matches(self, name: str) -> bool:
        """Check if a request model name matches this entry."""
        if name == self.model_name or name == self.model_path:
            return True
        return name in self.aliases


class ModelRegistry:
    """Registry of loaded models for multi-model serving.

    Thread-safe for reads (dict lookup). Writes (add/remove) should only
    happen during server startup or explicit model load/unload.
    """

    def __init__(self):
        self._entries: dict[str, ModelEntry] = {}  # canonical_name -> entry
        self._default: str | None = None
        # Lookup index: any name/alias -> canonical_name
        self._index: dict[str, str] = {}

    def add(self, entry: ModelEntry, is_default: bool = False):
        """Register a model. First model added is default unless specified."""
        canonical = entry.model_name
        self._entries[canonical] = entry

        # Build lookup index
        self._index[canonical] = canonical
        self._index[entry.model_path] = canonical
        for alias in entry.aliases:
            self._index[alias] = canonical

        if is_default or self._default is None:
            self._default = canonical

        logger.info(
            f"Registered model '{canonical}' "
            f"(aliases={entry.aliases or 'none'}, "
            f"default={canonical == self._default})"
        )

    def remove(self, name: str) -> ModelEntry | None:
        """Unregister a model. Returns the entry if found."""
        canonical = self._index.get(name)
        if not canonical or canonical not in self._entries:
            return None

        entry = self._entries.pop(canonical)

        # Clean up index
        keys_to_remove = [k for k, v in self._index.items() if v == canonical]
        for k in keys_to_remove:
            del self._index[k]

        # Reset default if we removed it
        if self._default == canonical:
            self._default = next(iter(self._entries), None)

        logger.info(f"Unregistered model '{canonical}'")
        return entry

    def get_engine(self, model_name: str | None = None) -> object:
        """Get the engine for a model name. Falls back to default.

        Args:
            model_name: Model name from request. None or "default" uses default.

        Returns:
            BaseEngine instance

        Raises:
            KeyError: If no matching model found and no default set.
        """
        if not self._entries:
            raise KeyError("No models loaded")

        # None, empty, or "default" → default engine
        if not model_name or model_name == "default":
            if self._default and self._default in self._entries:
                return self._entries[self._default].engine
            raise KeyError("No default model set")

        # Exact lookup via index
        canonical = self._index.get(model_name)
        if canonical and canonical in self._entries:
            return self._entries[canonical].engine

        # Fallback: default
        if self._default and self._default in self._entries:
            return self._entries[self._default].engine

        raise KeyError(f"Model '{model_name}' not found")

    def get_entry(self, model_name: str | None = None) -> ModelEntry:
        """Get the full ModelEntry (engine + metadata) for a model name."""
        if not self._entries:
            raise KeyError("No models loaded")

        if not model_name or model_name == "default":
            if self._default and self._default in self._entries:
                return self._entries[self._default]
            raise KeyError("No default model set")

        canonical = self._index.get(model_name)
        if canonical and canonical in self._entries:
            return self._entries[canonical]

        if self._default and self._default in self._entries:
            return self._entries[self._default]

        raise KeyError(f"Model '{model_name}' not found")

    def validate_model_name(self, model_name: str) -> None:
        """Validate that a model name is served. Raises KeyError if not."""
        if not model_name:
            return
        if model_name == "default":
            return
        if model_name not in self._index:
            available = ", ".join(self.list_model_names())
            raise KeyError(
                f"Model '{model_name}' not found. Available: {available}"
            )

    def list_model_names(self) -> list[str]:
        """List all canonical model names + aliases."""
        names = []
        for entry in self._entries.values():
            names.append(entry.model_name)
            for alias in sorted(entry.aliases):
                if alias != entry.model_name:
                    names.append(alias)
        return names

    def list_entries(self) -> list[ModelEntry]:
        """List all model entries."""
        return list(self._entries.values())

    @property
    def default_name(self) -> str | None:
        """The default model's canonical name."""
        return self._default

    @property
    def default_entry(self) -> ModelEntry | None:
        """The default model's entry."""
        if self._default:
            return self._entries.get(self._default)
        return None

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._index
