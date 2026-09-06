# SPDX-License-Identifier: Apache-2.0
"""Generation dispatch for DFlash runtime backends."""

from __future__ import annotations

from typing import Any


def stream_generate(
    runtime: Any | None,
    model: Any,
    processor: Any,
    prompt: str,
    **kwargs: Any,
):
    if runtime is not None and runtime.backend == "dflash-mlx":
        return runtime.backend_state.stream_generate(prompt, **kwargs)
    from mlx_vlm import stream_generate as mlx_vlm_stream_generate

    return mlx_vlm_stream_generate(model, processor, prompt, **kwargs)


def generate(
    runtime: Any | None,
    model: Any,
    processor: Any,
    prompt: str,
    **kwargs: Any,
):
    if runtime is not None and runtime.backend == "dflash-mlx":
        return runtime.backend_state.generate(prompt, **kwargs)
    from mlx_vlm import generate as mlx_vlm_generate

    return mlx_vlm_generate(model, processor, prompt, **kwargs)
