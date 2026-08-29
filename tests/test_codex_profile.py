# SPDX-License-Identifier: Apache-2.0
"""Sanity tests for the codex CLI agent profile.

Confirms the YAML loads, the template substitutes correctly, and the
config shape matches what Codex CLI's TOML parser actually expects —
catches refactor-time breakage of `rapid-mlx agents codex --setup`.
"""

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from vllm_mlx.agents import get_profile, list_profiles


def test_codex_profile_is_loadable():
    profile = get_profile("codex")
    assert profile is not None
    assert profile.name == "codex"
    assert profile.display_name == "Codex CLI"


def test_codex_profile_is_listed():
    """`rapid-mlx agents` (no args) lists profiles; codex must show up."""
    names = {p.name for p in list_profiles()}
    assert "codex" in names


def test_codex_capabilities_match_supported_surface():
    profile = get_profile("codex")
    # rapid-mlx >= 0.7.10 implements /v1/responses streaming + tool calls;
    # the profile must advertise those so `--test` plans the right checks.
    assert profile.needs_function_calling is True
    assert profile.needs_streaming is True


def test_codex_recommended_models_are_known_aliases():
    """The recommended models must exist in aliases.json. If an alias is
    renamed/removed, this test catches the dead reference before users do."""
    from vllm_mlx.model_aliases import list_aliases

    aliases = set(list_aliases())
    profile = get_profile("codex")
    missing = [m for m in profile.recommended_models if m not in aliases]
    assert not missing, (
        f"codex.yaml references model aliases that don't exist in "
        f"aliases.json: {missing}. Update the YAML or add the alias."
    )


def test_codex_template_renders_to_valid_toml():
    """The substituted template must parse as TOML so Codex CLI accepts it."""
    profile = get_profile("codex")
    rendered = profile.render_config(
        base_url="http://localhost:8000/v1",
        model_id="qwen3.6-35b-4bit",
    )
    assert isinstance(rendered, str)
    parsed = tomllib.loads(rendered)
    # Top-level keys Codex reads.
    assert parsed["model"] == "qwen3.6-35b-4bit"
    assert parsed["model_provider"] == "rapid-mlx"
    assert parsed["model_context_window"] == 32768
    assert parsed["model_catalog_json"] == "{model_catalog_path}"
    # Provider block.
    providers = parsed["model_providers"]
    assert "rapid-mlx" in providers
    rmlx = providers["rapid-mlx"]
    assert rmlx["base_url"] == "http://localhost:8000/v1"
    # name is a display-only field, must be a string.
    assert isinstance(rmlx["name"], str)
    # Codex CLI >= 0.135 rejects an inline `api_key = "..."` literal under
    # `--strict-config` (the field is unknown; credentials come from
    # `env_key = "VAR_NAME"` indirection instead). rapid-mlx is unauthed
    # by default so the template omits both — verify neither variant
    # sneaks back in via a future refactor.
    assert "api_key" not in rmlx, (
        "Inline `api_key` is rejected by `codex --strict-config`. "
        'If you need to ship credentials, use `env_key = "VAR_NAME"`.'
    )


def test_codex_setup_writes_startup_static_model_catalog(monkeypatch, tmp_path):
    import json

    from vllm_mlx.agents.adapter import setup_agent_config

    monkeypatch.setenv("HOME", str(tmp_path))
    profile = get_profile("codex")
    summary = setup_agent_config(
        profile,
        base_url="http://127.0.0.1:8000/v1",
        model_id="deepseek-v4-flash-0731",
        context_length=1_048_576,
    )

    assert "config" in summary.lower()
    config_path = tmp_path / ".codex" / "config.toml"
    config = tomllib.loads(config_path.read_text())
    catalog_path = tmp_path / ".codex" / "rapid-mlx-model-catalog.json"
    assert config["model_catalog_json"] == str(catalog_path.resolve())
    catalog = json.loads(catalog_path.read_text())
    model = catalog["models"][0]
    assert model["slug"] == "deepseek-v4-flash-0731"
    assert model["context_window"] == 1_048_576
    assert model["default_reasoning_level"] == "none"


@pytest.mark.parametrize(
    "base_url,expected_in_rendered",
    [
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("https://my.host:9000/v1", "https://my.host:9000/v1"),
    ],
)
def test_codex_template_passes_through_base_url(base_url, expected_in_rendered):
    profile = get_profile("codex")
    rendered = profile.render_config(base_url=base_url, model_id="default")
    assert expected_in_rendered in rendered
