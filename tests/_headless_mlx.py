"""Inert MLX import seam for lifecycle-only tests on Linux."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from unittest.mock import MagicMock


def install_headless_mlx_import_stubs() -> None:
    """Permit importing engine classes when tensor operations are never run."""

    if importlib.util.find_spec("mlx") is not None:
        return

    module_names = (
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx.utils",
        "mlx_lm",
        "mlx_lm.generate",
        "mlx_lm.sample_utils",
        "mlx_lm.tokenizer_utils",
        "mlx_lm.models",
        "mlx_lm.models.cache",
        "mlx_lm.models.deepseek_v32",
    )
    for name in module_names:
        module = MagicMock(name=name)
        module.__name__ = name
        module.__path__ = []
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = module

    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.modules["mlx_lm.generate"].BatchGenerator = type("BatchGenerator", (), {})
    sys.modules["mlx_lm.tokenizer_utils"].NaiveStreamingDetokenizer = type(
        "NaiveStreamingDetokenizer", (), {}
    )
    cache_module = sys.modules["mlx_lm.models.cache"]
    for class_name in ("MambaCache", "ArraysCache", "KVCache", "RotatingKVCache"):
        setattr(cache_module, class_name, type(class_name, (), {}))
    sys.modules["mlx_lm.models"].deepseek_v32 = sys.modules[
        "mlx_lm.models.deepseek_v32"
    ]
