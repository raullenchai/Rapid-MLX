"""Local MiniMax-H3-Image runtime; layout remains importable without MLX."""

from typing import Any

__all__ = ["MiniMaxH3Image"]


def __getattr__(name: str) -> Any:
    if name == "MiniMaxH3Image":
        from .pipeline import MiniMaxH3Image

        return MiniMaxH3Image
    raise AttributeError(name)
