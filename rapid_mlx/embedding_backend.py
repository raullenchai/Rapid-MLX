# SPDX-License-Identifier: Apache-2.0
"""Dependency-free dispatch for the native EmbeddingGemma 2 encoder."""

import json
from pathlib import Path


def is_native_embedding_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    if model_name in {
        "embeddinggemma-2-bf16",
        "embeddinggemma-2-4bit",
        "google/embeddinggemma-2",
        "mlx-community/embeddinggemma-2-bf16",
        "mlx-community/embeddinggemma-2-4bit",
    }:
        return True
    config = Path(model_name) / "config.json"
    if config.is_file():
        return bool(
            json.loads(config.read_text()).get("model_type") == "embedding_gemma2"
        )
    return False
