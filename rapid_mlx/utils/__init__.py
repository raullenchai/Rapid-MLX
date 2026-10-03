# SPDX-License-Identifier: Apache-2.0
"""Utility modules for rapid-mlx.

Keep the tokenizer import lazy: importing JSON body-depth helpers in the
standalone decision server must not initialize MLX (and Metal) first.
"""


def __getattr__(name: str):
    if name == "load_model_with_fallback":
        from .tokenizer import load_model_with_fallback

        return load_model_with_fallback
    raise AttributeError(name)


__all__ = ["load_model_with_fallback"]
