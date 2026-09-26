#!/usr/bin/env python3
"""Measure Qwen-Image 2.1 MLX memory with an experimental q4 encoder.

This is an engineering probe, not a supported conversion path.  mflux 0.20
deliberately keeps the Qwen3-VL encoder in bf16.  ``--quantize-text-encoder``
temporarily removes that guard so we can measure whether fully q4 weights fit
the 8/16 GiB product targets before changing a production loader.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import psutil


def _gib(value: int) -> float:
    return round(value / 1024**3, 3)


def _memory(phase: str) -> dict[str, object]:
    process = psutil.Process()
    row = {
        "phase": phase,
        "mlx_active_gib": _gib(mx.get_active_memory()),
        "mlx_cache_gib": _gib(mx.get_cache_memory()),
        "mlx_peak_gib": _gib(mx.get_peak_memory()),
        "rss_gib": _gib(process.memory_info().rss),
    }
    print(json.dumps(row), flush=True)
    return row


def _enable_text_encoder_quantization() -> None:
    from mflux.models.qwen21.weights.qwen21_weight_definition import (
        Qwen21WeightDefinition,
    )

    original = Qwen21WeightDefinition.get_components

    def components():
        result = original()
        for component in result:
            if component.name == "text_encoder":
                component.skip_quantization = False
        return result

    Qwen21WeightDefinition.get_components = staticmethod(components)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--quantize", type=int, default=4)
    parser.add_argument("--quantize-text-encoder", action="store_true")
    parser.add_argument(
        "--materialize-prompt",
        action="store_true",
        help="Evaluate prompt embeddings before MemorySaver releases the encoder",
    )
    parser.add_argument(
        "--materialize-model",
        action="store_true",
        help="Evaluate every model parameter immediately after loading",
    )
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="A red vintage bicycle leaning against a white wall, morning light",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mx.set_cache_limit(0)
    if args.memory_limit_gib:
        mx.set_memory_limit(int(args.memory_limit_gib * 1024**3))
        mx.set_wired_limit(int(args.memory_limit_gib * 1024**3))
    mx.reset_peak_memory()
    _memory("start")

    if args.quantize_text_encoder:
        _enable_text_encoder_quantization()

    if args.materialize_prompt:
        from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import (
            Qwen21PromptEncoder,
        )

        original_encode_prompt = Qwen21PromptEncoder.encode_prompt

        def encode_prompt(*encode_args, **encode_kwargs):
            embeds, mask = original_encode_prompt(*encode_args, **encode_kwargs)
            mx.eval(embeds, mask)
            return embeds, mask

        Qwen21PromptEncoder.encode_prompt = staticmethod(encode_prompt)

    from mflux.callbacks.callback import BeforeLoopCallback, InLoopCallback
    from mflux.callbacks.instances.memory_saver import MemorySaver
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.tiling_config import TilingConfig
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    class Probe(BeforeLoopCallback, InLoopCallback):
        def call_before_loop(self, **kwargs) -> None:
            _memory("before_denoise_after_encoder_eviction")

        def call_in_loop(self, **kwargs) -> None:
            _memory("denoise_step")

    started = time.perf_counter()
    model = QwenImage21(
        quantize=args.quantize,
        model_path=str(args.model_path),
        model_config=ModelConfig.qwen_image_21(),
    )
    if args.materialize_model:
        mx.eval(model.parameters())
    _memory("model_loaded")
    if args.save_model:
        model.save_model(str(args.save_model))
        _memory("model_saved")
        if args.save_only:
            return

    model.tiling_config = TilingConfig()
    model.callbacks.register(
        MemorySaver(
            model=model,
            keep_transformer=False,
            cache_limit_bytes=0,
            num_seeds=1,
        )
    )
    model.callbacks.register(Probe())
    image = model.generate_image(
        seed=42,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
    )
    _memory("generated")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.image.save(args.output)
    del image, model
    gc.collect()
    mx.clear_cache()
    final = _memory("released")
    final["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
