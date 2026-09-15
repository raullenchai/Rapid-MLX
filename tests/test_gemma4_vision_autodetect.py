# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for #1126 Gemma 4 repo-ID vision routing."""

import json

from rapid_mlx import model_metadata
from rapid_mlx.api import utils as api_utils


def _cached_snapshot(tmp_path, config, weights):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights}), encoding="utf-8"
    )
    return snapshot


def test_gemma4_repo_id_with_cached_vision_weights_routes_to_mllm(
    tmp_path, monkeypatch
):
    """A Gemma 4 repo ID must use its cached config and vision tensors."""
    snapshot = _cached_snapshot(
        tmp_path,
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "vision_config": {"hidden_size": 128},
        },
        {"vision_tower.encoder.0.weight": "model.safetensors"},
    )
    monkeypatch.setattr(
        model_metadata,
        "_cached_file",
        lambda repo_id, filename: (
            snapshot / filename if filename == "config.json" else None
        ),
    )

    assert api_utils.is_mllm_model("mlx-community/gemma-4-26b-a4b-it-4bit")


def test_gemma4_configured_text_fork_stays_on_text_route(tmp_path, monkeypatch):
    """#393 guard: config alone cannot promote a text-only checkpoint."""
    snapshot = _cached_snapshot(
        tmp_path,
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "vision_config": {"hidden_size": 128},
        },
        {"language_model.layers.0.weight": "model.safetensors"},
    )
    monkeypatch.setattr(
        model_metadata,
        "_cached_file",
        lambda repo_id, filename: (
            snapshot / filename if filename == "config.json" else None
        ),
    )

    assert not api_utils.is_mllm_model("publisher/gemma4-text-fork")
