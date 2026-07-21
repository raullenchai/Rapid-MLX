"""Regression tests for agent --setup config generation (#1120).

Covers:
  1. context_length placeholder is resolved from server-reported context_window
  2. hermes template includes image + computer_use in platform_toolsets
  3. merge-on-write preserves existing user config keys
  4. fresh write works when no config file exists
"""

from __future__ import annotations

import json
import textwrap

import pytest
import yaml

from vllm_mlx.agents.adapter import (
    _deep_merge,
    _merge_file_config,
    _MergeParseError,
    _valid_context_window,
    fetch_context_window,
    setup_agent_config,
)
from vllm_mlx.agents.base import AgentConfigSpec, AgentProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hermes_profile(**overrides) -> AgentProfile:
    """Build a minimal hermes-like profile for testing."""
    defaults = dict(
        name="hermes",
        display_name="Hermes Agent",
        config=AgentConfigSpec(
            type="yaml",
            path="~/.hermes/config.yaml",
            template=textwrap.dedent("""\
                model:
                  provider: "custom"
                  default: "{model_id}"
                  base_url: "{base_url}"
                  context_length: {context_length}
                  max_tokens: 4096
                platform_toolsets:
                  cli: [terminal, file, code_execution, web, browser, skills, image, computer_use]
            """),
        ),
    )
    defaults.update(overrides)
    return AgentProfile(**defaults)


# ---------------------------------------------------------------------------
# 1. context_length placeholder resolution
# ---------------------------------------------------------------------------


class TestContextLengthPlaceholder:
    def test_default_fallback(self):
        """When no context_length is provided, falls back to 32768."""
        profile = _hermes_profile()
        rendered = profile.render_config("http://localhost:8000/v1", "qwen3.5-4b-4bit")
        parsed = yaml.safe_load(rendered)
        assert parsed["model"]["context_length"] == 32768

    def test_explicit_context_length(self):
        """When context_length is provided, it's used in the template."""
        profile = _hermes_profile()
        rendered = profile.render_config(
            "http://localhost:8000/v1",
            "gemma-4-26b-4bit",
            context_length=131072,
        )
        parsed = yaml.safe_load(rendered)
        assert parsed["model"]["context_length"] == 131072

    def test_setup_passes_context_length(self, tmp_path, monkeypatch):
        """setup_agent_config forwards context_length to render_config."""
        config_path = tmp_path / "config.yaml"
        profile = _hermes_profile(
            config=AgentConfigSpec(
                type="yaml",
                path=str(config_path),
                template=textwrap.dedent("""\
                    model:
                      context_length: {context_length}
                """),
            ),
        )
        setup_agent_config(
            profile,
            "http://localhost:8000/v1",
            "test-model",
            context_length=65536,
        )
        parsed = yaml.safe_load(config_path.read_text())
        assert parsed["model"]["context_length"] == 65536


# ---------------------------------------------------------------------------
# 2. hermes toolsets completeness
# ---------------------------------------------------------------------------


class TestHermesToolsets:
    def test_hermes_yaml_includes_image_and_computer_use(self):
        """Hermes profile template must include image + computer_use tools."""
        from vllm_mlx.agents import get_profile, load_profiles

        load_profiles()
        profile = get_profile("hermes")
        assert profile is not None, "hermes profile not found"

        rendered = profile.render_config(
            "http://localhost:8000/v1", "test-model", context_length=32768
        )
        parsed = yaml.safe_load(rendered)
        toolsets = parsed.get("platform_toolsets", {}).get("cli", [])
        assert "image" in toolsets, f"'image' missing from cli toolsets: {toolsets}"
        assert "computer_use" in toolsets, (
            f"'computer_use' missing from cli toolsets: {toolsets}"
        )


# ---------------------------------------------------------------------------
# 3. merge-on-write
# ---------------------------------------------------------------------------


class TestMergeOnWrite:
    def test_deep_merge_preserves_existing_keys(self):
        base = {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
        override = {"b": {"x": 99, "z": 30}, "d": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": {"x": 99, "y": 20, "z": 30}, "c": 3, "d": 4}

    def test_deep_merge_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}

    def test_deep_merge_replaces_lists(self):
        """Template list values win unconditionally (no union merge)."""
        base = {"tools": ["terminal", "file", "my_custom_tool"]}
        override = {"tools": ["terminal", "file", "image", "web"]}
        result = _deep_merge(base, override)
        assert result["tools"] == ["terminal", "file", "image", "web"]
        assert "my_custom_tool" not in result["tools"]

    def test_yaml_merge_preserves_user_keys(self, tmp_path):
        """Existing YAML keys not in the template are preserved."""
        existing = tmp_path / "config.yaml"
        existing.write_text(
            textwrap.dedent("""\
                model:
                  provider: "custom"
                  default: "old-model"
                  base_url: "http://old:8000/v1"
                  context_length: 32768
                  max_tokens: 4096
                my_custom_setting: true
                platform_toolsets:
                  cli: [terminal, file, image, my_custom_tool]
            """)
        )

        new_template = textwrap.dedent("""\
            model:
              provider: "custom"
              default: "new-model"
              base_url: "http://new:8000/v1"
              context_length: 131072
              max_tokens: 4096
            platform_toolsets:
              cli: [terminal, file, code_execution, web, browser, skills, image, computer_use]
        """)

        result = _merge_file_config(existing, new_template, "yaml")
        parsed = yaml.safe_load(result)

        # Template values win
        assert parsed["model"]["default"] == "new-model"
        assert parsed["model"]["context_length"] == 131072
        # User's custom key is preserved
        assert parsed["my_custom_setting"] is True
        # platform_toolsets.cli is replaced by template (authoritative)
        cli = parsed["platform_toolsets"]["cli"]
        assert "code_execution" in cli  # from template
        assert "my_custom_tool" not in cli  # template list wins

    def test_json_merge_preserves_user_keys(self, tmp_path):
        existing = tmp_path / "config.json"
        existing.write_text(json.dumps({"base_url": "old", "custom": 42}))
        new_template = json.dumps({"base_url": "new", "model": "test"})
        result = _merge_file_config(existing, new_template, "json")
        parsed = json.loads(result)
        assert parsed["base_url"] == "new"
        assert parsed["custom"] == 42
        assert parsed["model"] == "test"

    def test_fresh_write_when_no_existing(self, tmp_path):
        """When no existing file, rendered template is returned as-is."""
        missing = tmp_path / "does_not_exist.yaml"
        template = "model:\n  default: test\n"
        result = _merge_file_config(missing, template, "yaml")
        assert result == template

    def test_setup_merges_existing_yaml(self, tmp_path):
        """Full integration: setup_agent_config merges into existing file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent("""\
                model:
                  provider: "custom"
                  default: "old"
                  base_url: "http://old:8000/v1"
                  context_length: 8192
                user_preference: dark_mode
            """)
        )

        profile = _hermes_profile(
            config=AgentConfigSpec(
                type="yaml",
                path=str(config_path),
                template=textwrap.dedent("""\
                    model:
                      provider: "custom"
                      default: "{model_id}"
                      base_url: "{base_url}"
                      context_length: {context_length}
                """),
            ),
        )
        summary = setup_agent_config(
            profile,
            "http://localhost:8000/v1",
            "new-model",
            context_length=131072,
        )
        assert "Merged" in summary

        parsed = yaml.safe_load(config_path.read_text())
        assert parsed["model"]["default"] == "new-model"
        assert parsed["model"]["context_length"] == 131072
        assert parsed["user_preference"] == "dark_mode"

    def test_setup_reports_failure_on_malformed(self, tmp_path):
        """setup_agent_config reports failure when existing config is malformed."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("[[[[not yaml")
        profile = _hermes_profile(
            config=AgentConfigSpec(
                type="yaml",
                path=str(config_path),
                template="model:\n  default: new\n",
            ),
        )
        summary = setup_agent_config(profile, "http://x/v1", "m")
        assert summary.startswith("Cannot")

    def test_empty_yaml_treated_as_fresh(self, tmp_path):
        """Empty existing YAML file is treated as fresh write."""
        existing = tmp_path / "config.yaml"
        existing.write_text("")
        template = "model:\n  default: new\n"
        result = _merge_file_config(existing, template, "yaml")
        assert result == template

    def test_unreadable_file_raises(self, tmp_path, monkeypatch):
        """Unreadable existing file raises OSError (caller must not overwrite)."""
        from pathlib import Path

        existing = tmp_path / "config.yaml"
        existing.write_text("model:\n  default: old\n")
        original_read = Path.read_text

        def _fail_read(self, *a, **kw):
            if self == existing:
                raise OSError("permission denied")
            return original_read(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", _fail_read)
        with pytest.raises(OSError):
            _merge_file_config(existing, "model:\n  default: new\n", "yaml")

    def test_malformed_yaml_raises(self, tmp_path):
        """Malformed existing YAML raises _MergeParseError."""
        existing = tmp_path / "config.yaml"
        existing.write_text("this: is: not: valid: yaml: [[")
        with pytest.raises(_MergeParseError):
            _merge_file_config(existing, "model:\n  default: new\n", "yaml")

    def test_malformed_json_raises(self, tmp_path):
        """Malformed existing JSON raises _MergeParseError."""
        existing = tmp_path / "config.json"
        existing.write_text("{not valid json")
        with pytest.raises(_MergeParseError):
            _merge_file_config(existing, '{"model": "new"}', "json")


# ---------------------------------------------------------------------------
# 4. context_window validation
# ---------------------------------------------------------------------------


class TestContextWindowValidation:
    def test_positive_int_accepted(self):
        assert _valid_context_window(131072) == 131072

    def test_zero_rejected(self):
        assert _valid_context_window(0) is None

    def test_negative_rejected(self):
        assert _valid_context_window(-1) is None

    def test_bool_rejected(self):
        assert _valid_context_window(True) is None
        assert _valid_context_window(False) is None

    def test_none_rejected(self):
        assert _valid_context_window(None) is None

    def test_string_rejected(self):
        assert _valid_context_window("131072") is None


class TestFetchContextWindow:
    def test_exact_model_match(self, monkeypatch):
        models = [
            {"id": "model-a", "context_window": 8192},
            {"id": "model-b", "context_window": 131072},
        ]
        monkeypatch.setattr(
            "vllm_mlx.agents.adapter._fetch_models", lambda _url: models
        )
        assert fetch_context_window("http://x/v1", "model-b") == 131072

    def test_fallback_single_model_serve(self, monkeypatch):
        """Single-model serve: fallback to the only entry when no exact match."""
        models = [{"id": "only-model", "context_window": 65536}]
        monkeypatch.setattr(
            "vllm_mlx.agents.adapter._fetch_models", lambda _url: models
        )
        assert fetch_context_window("http://x/v1", "unknown") == 65536

    def test_no_fallback_multi_model_serve(self, monkeypatch):
        """Multi-model serve: no fallback to avoid wrong context window."""
        models = [
            {"id": "model-a", "context_window": 8192},
            {"id": "model-b", "context_window": 131072},
        ]
        monkeypatch.setattr(
            "vllm_mlx.agents.adapter._fetch_models", lambda _url: models
        )
        assert fetch_context_window("http://x/v1", "model-c") is None

    def test_empty_models(self, monkeypatch):
        monkeypatch.setattr("vllm_mlx.agents.adapter._fetch_models", lambda _url: [])
        assert fetch_context_window("http://x/v1", "any") is None


# ---------------------------------------------------------------------------
# 5. env-type profiles are unaffected
# ---------------------------------------------------------------------------


class TestEnvProfileUnchanged:
    def test_env_config_returns_dict(self):
        profile = AgentProfile(
            name="test-env",
            display_name="Test",
            config=AgentConfigSpec(
                type="env",
                env_vars={
                    "BASE_URL": "{base_url}",
                    "MODEL": "{model_id}",
                },
            ),
        )
        rendered = profile.render_config("http://localhost:8000/v1", "my-model")
        assert isinstance(rendered, dict)
        assert rendered["BASE_URL"] == "http://localhost:8000/v1"
        assert rendered["MODEL"] == "my-model"
