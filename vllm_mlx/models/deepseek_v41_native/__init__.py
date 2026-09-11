"""MLX runtime for deepseek-ai/DeepSeek-V4.1-Flash (model_type: deepseek_v41).

Keep this package entry lightweight: artifact admission is imported by the
shared CLI even when an unrelated model is served. The architecture itself is
loaded only when a caller asks for its public classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ModelArgs
    from .model import Model

__all__ = ["ModelArgs", "Model"]


def __getattr__(name: str):
    if name == "ModelArgs":
        from .config import ModelArgs

        return ModelArgs
    if name == "Model":
        from .model import Model

        return Model
    raise AttributeError(name)
