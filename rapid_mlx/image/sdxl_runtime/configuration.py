"""Configuration base class.

A ``Config`` is a plain ``@dataclass`` that round-trips to ``config.json``. It is
deliberately tiny: no validation framework, no schema registry — just dataclass
fields plus JSON (de)serialization. Subclasses look like::

    @dataclass
    class UNet2DConfig(Config):
        in_channels: int = 4
        hidden_size: int = 320
        ...
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, TypeVar

CONFIG_NAME = "config.json"

T = TypeVar("T", bound="Config")


@dataclasses.dataclass
class Config:
    """Base for all model / scheduler configs."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict, tagged with the concrete class name."""
        data = dataclasses.asdict(self)
        data["_class_name"] = type(self).__name__
        return data

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Build a config, ignoring unknown keys (forward compatibility)."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        known = {k: v for k, v in data.items() if k in field_names}
        unknown = set(data) - field_names - {"_class_name"}
        if unknown:
            from .utils import get_logger

            get_logger().debug(
                "%s: ignoring unknown config keys %s", cls.__name__, sorted(unknown)
            )
        return cls(**known)

    def save(self, save_directory: str | Path) -> Path:
        """Write ``config.json`` into ``save_directory`` and return its path."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        path = save_directory / CONFIG_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        """Load from a ``config.json`` file or a directory containing one."""
        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_NAME
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def replace(self: T, **changes: Any) -> T:
        """Return a copy with ``changes`` applied (like ``dataclasses.replace``)."""
        return dataclasses.replace(self, **changes)
