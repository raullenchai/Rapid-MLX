"""Hermes 0.21.6 no longer reads model.max_tokens from config.yaml."""

import yaml

from rapid_mlx.agents import get_profile


def test_hermes_current_config_omits_removed_output_limit():
    profile = get_profile("hermes")
    config = yaml.safe_load(
        profile.render_config(
            "http://127.0.0.1:18710/v1", "qwen3.5-4b-4bit", context_length=262144
        )
    )

    assert config["model"]["provider"] == "custom"
    assert config["model"]["context_length"] == 262144
    assert "max_tokens" not in config["model"]
