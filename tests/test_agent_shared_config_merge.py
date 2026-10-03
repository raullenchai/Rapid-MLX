# SPDX-License-Identifier: Apache-2.0
"""Agent configs shared with the user's own entries must merge, never replace.

Regression coverage for the PR #4056 review (FIX-FIRST):

* dsh's ``cordis.patch.yml`` ``llm-pi-ai`` layer is the plugin-level provider
  registry. Replacing the same-id layer wholesale deleted every other provider
  the user had configured there.
* pi's ``models.json`` keeps a LIST of models per provider. The generic
  deep-merge replaced the whole ``providers.rapid-mlx.models`` list, deleting
  the user's other models under that provider — with no preview and no backup.
* A fresh pi ``models.json`` (a file that carries API keys) was created 0644.
* ``agents opencode --test`` now launches the real client, and an inherited
  ``XDG_*_HOME`` / ``OPENCODE_CONFIG`` escaped the throwaway HOME.

Every test seeds a realistic existing user config in a tmp dir; nothing here
touches the real ~/.pi, ~/.dsh or ~/.config.
"""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from rapid_mlx.agents import get_profile, load_profiles
from rapid_mlx.agents.adapter import _atomic_write, setup_agent_config
from rapid_mlx.agents.config_merge import deep_merge, merge_patch_layers
from rapid_mlx.agents.setup import (
    FIRST_CLASS_SETUP_AGENTS,
    apply_setup_plan,
    build_setup_plan,
)

BASE_URL = "http://127.0.0.1:8153/v1"
MODEL = "mlx-community/Qwen3.5-9B-4bit"  # a full HF path: the CLI keeps it verbatim


def setup_function():
    load_profiles()


# --------------------------------------------------------------------------
# dsh: cordis.patch.yml
# --------------------------------------------------------------------------

DSH_EXISTING = """\
- id: user-custom-layer
  config:
    theme: dark
- id: llm-pi-ai
  config:
    timeoutMs: 60000
    providers:
      openai:
        apiKeyEnv: OPENAI_API_KEY
      custom-local:
        baseURL: http://host/v1
      rapid-mlx:
        baseURL: http://old-host:8000/v1
        models:
          - id: other-local
            name: Other local model
          - id: mlx-community/Qwen3.5-9B-4bit
            name: stale name
            userNote: keep me
- id: agent-default-model
  config:
    provider: openai
    model: gpt-x
"""


def _assert_dsh_merge_kept_user_entries(layers: list) -> None:
    ids = [layer["id"] for layer in layers]
    # Layers stay in place; nothing is duplicated.
    assert ids == ["user-custom-layer", "llm-pi-ai", "agent-default-model"]
    by_id = {layer["id"]: layer["config"] for layer in layers}
    assert by_id["user-custom-layer"] == {"theme": "dark"}
    pi_ai = by_id["llm-pi-ai"]
    assert pi_ai["timeoutMs"] == 60000
    providers = pi_ai["providers"]
    # The user's other providers in the shared layer survive.
    assert providers["openai"] == {"apiKeyEnv": "OPENAI_API_KEY"}
    assert providers["custom-local"] == {"baseURL": "http://host/v1"}
    rapid = providers["rapid-mlx"]
    assert rapid["baseURL"] == BASE_URL
    # Models merge by id: the other model survives, ours is updated in place.
    assert [m["id"] for m in rapid["models"]] == ["other-local", MODEL]
    ours = rapid["models"][1]
    assert ours["name"] == f"{MODEL} (Rapid-MLX)"
    assert ours["userNote"] == "keep me"
    assert by_id["agent-default-model"] == {"provider": "rapid-mlx", "model": MODEL}


def test_dsh_first_class_setup_keeps_other_providers_in_the_shared_layer(
    tmp_path, monkeypatch
):
    dsh_home = tmp_path / "dsh"
    dsh_home.mkdir()
    patch_file = dsh_home / "cordis.patch.yml"
    patch_file.write_text(DSH_EXISTING)
    monkeypatch.setenv("DSH_HOME", str(dsh_home))

    plan = build_setup_plan("deepseek-harness", BASE_URL, MODEL, context_length=65536)
    _assert_dsh_merge_kept_user_entries(plan.after)

    apply_setup_plan(plan)
    _assert_dsh_merge_kept_user_entries(yaml.safe_load(patch_file.read_text()))
    assert list(dsh_home.glob("cordis.patch.yml.bak.*"))


def test_dsh_generic_writer_keeps_other_providers_in_the_shared_layer(
    tmp_path, monkeypatch
):
    """The harness's generic writer shares the merge with the first-class flow."""
    dsh_home = tmp_path / "dsh"
    dsh_home.mkdir()
    patch_file = dsh_home / "cordis.patch.yml"
    patch_file.write_text(DSH_EXISTING)
    monkeypatch.setenv("DSH_HOME", str(dsh_home))
    profile = get_profile("deepseek-harness")
    assert profile is not None

    summary = setup_agent_config(profile, BASE_URL, MODEL, context_length=65536)

    assert not summary.startswith("Cannot"), summary
    _assert_dsh_merge_kept_user_entries(yaml.safe_load(patch_file.read_text()))


def test_patch_layer_merge_edge_cases():
    existing = [
        "a stray scalar entry",
        {"note": "an entry without an id"},
        {"id": "llm-pi-ai", "config": None, "enabled": True},
    ]
    incoming = [
        {"id": "llm-pi-ai", "config": {"providers": {"rapid-mlx": {}}}},
        {"id": "agent-default-model", "config": {"model": "m"}},
    ]

    merged = merge_patch_layers(existing, incoming)

    assert merged[:2] == existing[:2]
    # A non-mapping config cannot be merged and takes ours; other keys stay.
    assert merged[2] == {
        "id": "llm-pi-ai",
        "config": {"providers": {"rapid-mlx": {}}},
        "enabled": True,
    }
    assert merged[3] == incoming[1]
    # Inputs are not mutated.
    assert existing[2]["config"] is None


def test_deep_merge_only_merges_id_keyed_models_lists():
    base = {
        "models": [{"id": "a", "x": 1}],
        "tags": ["user"],
        "other": {"models": ["plain", "strings"]},
    }
    override = {
        "models": [{"id": "b"}],
        "tags": ["ours"],
        "other": {"models": ["ours"]},
    }

    merged = deep_merge(base, override)

    assert merged["models"] == [{"id": "a", "x": 1}, {"id": "b"}]
    # Lists that are not id-keyed keep replace semantics.
    assert merged["tags"] == ["ours"]
    assert merged["other"]["models"] == ["ours"]


# --------------------------------------------------------------------------
# pi: models.json
# --------------------------------------------------------------------------

PI_EXISTING = {
    "providers": {
        "rapid-mlx": {
            "baseUrl": "http://old-host:8000/v1",
            "api": "openai-completions",
            "apiKey": "not-needed",
            "headers": {"X-Team": "infra"},
            "models": [
                {"id": "custom-a", "name": "Custom A", "contextWindow": 32768},
                {"id": MODEL, "name": "stale", "cost": {"input": 0}},
                {"id": "custom-b"},
            ],
        },
        "glm-spark": {
            "baseUrl": "http://elsewhere:8888/v1",
            "api": "openai-completions",
            "apiKey": "sk-real-secret",
            "models": [{"id": "GLM-5.3"}],
        },
    }
}


def _assert_pi_merge_kept_user_entries(written: dict) -> None:
    providers = written["providers"]
    assert providers["glm-spark"] == PI_EXISTING["providers"]["glm-spark"]
    rapid = providers["rapid-mlx"]
    assert rapid["baseUrl"] == BASE_URL
    assert rapid["headers"] == {"X-Team": "infra"}
    assert [m["id"] for m in rapid["models"]] == ["custom-a", MODEL, "custom-b"]
    assert rapid["models"][0] == {
        "id": "custom-a",
        "name": "Custom A",
        "contextWindow": 32768,
    }
    ours = rapid["models"][1]
    assert ours["name"] == f"{MODEL} (Rapid-MLX)"
    assert ours["contextWindow"] == 65536
    assert ours["cost"] == {"input": 0}


@pytest.fixture
def pi_dir(tmp_path, monkeypatch):
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    return agent_dir


def test_pi_is_a_first_class_setup_flow():
    assert "pi" in FIRST_CLASS_SETUP_AGENTS


def test_pi_plan_merges_models_by_id_previews_and_backs_up(pi_dir):
    pi_dir.mkdir()
    models_json = pi_dir / "models.json"
    original = json.dumps(PI_EXISTING, indent=2)
    models_json.write_text(original)

    plan = build_setup_plan("pi", BASE_URL, MODEL, context_length=65536)

    # Planning is side-effect free and the preview names the change.
    assert models_json.read_text() == original
    assert plan.path == models_json
    assert plan.changed
    assert BASE_URL in plan.diff()
    _assert_pi_merge_kept_user_entries(plan.after)

    apply_setup_plan(plan)

    _assert_pi_merge_kept_user_entries(json.loads(models_json.read_text()))
    (backup,) = pi_dir.glob("models.json.bak.*")
    assert backup.read_text() == original


def test_pi_plan_fresh_file_is_owner_only(pi_dir):
    plan = build_setup_plan("pi", BASE_URL, MODEL, context_length=65536)
    assert plan.before == {}

    apply_setup_plan(plan)

    models_json = pi_dir / "models.json"
    assert stat.S_IMODE(models_json.stat().st_mode) == 0o600
    written = json.loads(models_json.read_text())
    assert [m["id"] for m in written["providers"]["rapid-mlx"]["models"]] == [MODEL]


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_pi_plan_refuses_an_unmergeable_models_json(pi_dir, content):
    pi_dir.mkdir()
    (pi_dir / "models.json").write_text(content)

    with (
        patch("rapid_mlx.agents.setup.track_agent_configure_failed") as failed,
        pytest.raises(ValueError),
    ):
        build_setup_plan("pi", BASE_URL, MODEL)

    failed.assert_called_once_with("config_invalid", "pi")


def test_pi_generic_writer_merges_models_by_id(pi_dir):
    """The harness's generic writer must not drop sibling models either."""
    pi_dir.mkdir()
    (pi_dir / "models.json").write_text(json.dumps(PI_EXISTING))
    profile = get_profile("pi")
    assert profile is not None

    summary = setup_agent_config(profile, BASE_URL, MODEL, context_length=65536)

    assert not summary.startswith("Cannot"), summary
    _assert_pi_merge_kept_user_entries(json.loads((pi_dir / "models.json").read_text()))


def _run_pi_cli(monkeypatch, *flags):
    import rapid_mlx.cli as cli

    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.fetch_context_window", lambda *_a, **_k: 65536
    )
    monkeypatch.setattr(
        "sys.argv",
        ["rapid-mlx", "agents", "pi", "--setup", "--model", MODEL]
        + ["--base-url", BASE_URL, *flags],
    )
    cli.main()


def test_cli_pi_setup_previews_and_needs_consent(pi_dir, monkeypatch, capsys):
    pi_dir.mkdir()
    models_json = pi_dir / "models.json"
    original = json.dumps(PI_EXISTING)
    models_json.write_text(original)

    _run_pi_cli(monkeypatch, "--dry-run")
    out = capsys.readouterr().out
    assert "(proposed)" in out
    assert "Dry run only; nothing was written." in out
    assert models_json.read_text() == original

    # Non-interactive without --yes: consent is refused, nothing is written.
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    _run_pi_cli(monkeypatch)
    assert "Setup cancelled; nothing was written." in capsys.readouterr().out
    assert models_json.read_text() == original
    assert not list(pi_dir.glob("models.json.bak.*"))

    _run_pi_cli(monkeypatch, "--yes", "--no-check")
    assert "Configured Pi Coding Agent" in capsys.readouterr().out
    _assert_pi_merge_kept_user_entries(json.loads(models_json.read_text()))
    assert list(pi_dir.glob("models.json.bak.*"))


# --------------------------------------------------------------------------
# Fresh generic config files are owner-only
# --------------------------------------------------------------------------


def test_atomic_write_creates_fresh_files_owner_only(tmp_path):
    previous = os.umask(0o022)
    try:
        target = tmp_path / "agent" / "models.json"
        assert _atomic_write(target, '{"apiKey": "x"}\n') is True
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.read_text() == '{"apiKey": "x"}\n'
    assert not list(target.parent.glob(".rapid-mlx-*"))


def test_atomic_write_keeps_an_existing_files_mode(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("a: 1\n")
    target.chmod(0o640)

    assert _atomic_write(target, "a: 1\n") is False
    assert _atomic_write(target, "a: 2\n") is True

    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert target.read_text() == "a: 2\n"


# --------------------------------------------------------------------------
# opencode --test: XDG / OPENCODE_CONFIG isolation
# --------------------------------------------------------------------------

_PATCHED_RUNNER_TESTS = (
    "_test_single_tool_call",
    "_test_tool_choice",
    "_test_multi_turn_tool",
    "_test_no_tool_leak",
    "_test_no_tool_needed",
    "_test_streaming_tool_call",
    "_test_many_tools",
    "_test_tag_suppression",
    "_test_streaming_basic",
    "_test_stress_no_leak",
    "_test_e2e_file_read",
    "_test_e2e_terminal",
)


def _run_harness(profile_name, isolated_home, model_id=MODEL, *, setup_fails=False):
    """Drive AgentTestRunner.run() with every API/e2e probe mocked out.

    Returns the ``_test_e2e_chat`` mock so a test can read the argv template
    and child environment the real client would have been launched with.
    """
    from contextlib import ExitStack

    temporary_home = SimpleNamespace(name=str(isolated_home), cleanup=lambda: None)
    profile = get_profile(profile_name)
    assert profile is not None
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "rapid_mlx.agents.testing.tempfile.TemporaryDirectory",
                return_value=temporary_home,
            )
        )
        for target, value in (
            ("rapid_mlx.agents.testing.AgentTestRunner._server_available", True),
            ("rapid_mlx.agents.testing.AgentTestRunner._agent_binary_available", True),
            ("rapid_mlx.agents.adapter.fetch_context_window", 65536),
        ):
            stack.enter_context(patch(target, return_value=value))
        if setup_fails:
            stack.enter_context(
                patch(
                    "rapid_mlx.agents.adapter.setup_agent_config",
                    side_effect=RuntimeError("boom"),
                )
            )
        for name in _PATCHED_RUNNER_TESTS:
            stack.enter_context(patch("rapid_mlx.agents.testing." + name))
        plain_chat = stack.enter_context(
            patch("rapid_mlx.agents.testing._test_plain_chat")
        )
        e2e_chat = stack.enter_context(patch("rapid_mlx.agents.testing._test_e2e_chat"))
        from rapid_mlx.agents.testing import AgentTestRunner, TestResult, TestStatus

        plain_chat.return_value = TestResult("plain_chat", TestStatus.PASS)
        e2e_chat.return_value = TestResult("e2e_chat", TestStatus.PASS)
        AgentTestRunner(profile, base_url=BASE_URL, model_id=model_id).run()
    return e2e_chat


@pytest.fixture
def operator_env(tmp_path, monkeypatch):
    """A realistic operator shell: XDG dirs and OpenCode overrides exported."""
    real = tmp_path / "operator"
    for key, rel in (
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_DATA_HOME", ".local/share"),
        ("XDG_STATE_HOME", ".local/state"),
        ("XDG_CACHE_HOME", ".cache"),
    ):
        monkeypatch.setenv(key, str(real / rel))
    monkeypatch.setenv("OPENCODE_CONFIG", str(real / "custom-opencode.json"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(real / "opencode-dir"))
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT", '{"model": "anthropic/claude-remote"}'
    )
    return real


def test_opencode_e2e_child_cannot_escape_the_throwaway_home(tmp_path, operator_env):
    isolated_home = tmp_path / "isolated-opencode-home"
    isolated_home.mkdir()

    env = _run_harness("opencode", isolated_home).call_args.kwargs["env_overrides"]

    assert env["HOME"] == str(isolated_home)
    assert env["XDG_CONFIG_HOME"] == str(isolated_home / ".config")
    assert env["XDG_DATA_HOME"] == str(isolated_home / ".local/share")
    assert env["XDG_STATE_HOME"] == str(isolated_home / ".local/state")
    assert env["XDG_CACHE_HOME"] == str(isolated_home / ".cache")
    generated = isolated_home / ".config" / "opencode" / "opencode.json"
    assert env["OPENCODE_CONFIG"] == str(generated)
    assert env["OPENCODE_CONFIG_DIR"] == str(generated.parent)
    # The inline config outranks the file config: it carries ours, not the
    # operator's remote model.
    assert env["OPENCODE_CONFIG_CONTENT"] == generated.read_text()
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["rapid-mlx"]
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    # Nothing was written to the operator's real XDG tree.
    assert not operator_env.exists()


def test_opencode_inline_config_is_neutralised_when_setup_refresh_fails(
    tmp_path, operator_env
):
    isolated_home = tmp_path / "isolated-opencode-home"
    isolated_home.mkdir()

    env = _run_harness("opencode", isolated_home, setup_fails=True).call_args.kwargs[
        "env_overrides"
    ]

    assert env["OPENCODE_CONFIG_CONTENT"] == "{}"


def test_pi_e2e_pins_the_local_provider_and_model(tmp_path, monkeypatch):
    # An operator with a remote provider authenticated: without the pin pi
    # may pick it and send the test prompt to a paid account.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-operator-real")
    isolated_home = tmp_path / "isolated-pi-home"
    isolated_home.mkdir()

    e2e_chat = _run_harness("pi", isolated_home, model_id="odd model'id")

    query_cmd = e2e_chat.call_args.args[1]
    import shlex

    argv = shlex.split(query_cmd)
    assert argv[argv.index("--provider") + 1] == "rapid-mlx"
    assert argv[argv.index("--model") + 1] == "odd model'id"
    assert "{model_id}" not in query_cmd
    env = e2e_chat.call_args.kwargs["env_overrides"]
    assert env["PI_TELEMETRY"] == "0"
    assert env["PI_OFFLINE"] == "1"


def test_pi_setup_writes_through_a_symlinked_models_json(pi_dir, tmp_path):
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    real_target = dotfiles / "pi-models.json"
    original = json.dumps(PI_EXISTING)
    real_target.write_text(original)
    pi_dir.mkdir()
    link = pi_dir / "models.json"
    link.symlink_to(real_target)

    plan = build_setup_plan("pi", BASE_URL, MODEL, context_length=65536)
    apply_setup_plan(plan)

    # The link survives and still points at the managed file, which now
    # carries the merged config; the backup sits beside the real target.
    assert link.is_symlink()
    assert link.resolve() == real_target.resolve()
    _assert_pi_merge_kept_user_entries(json.loads(real_target.read_text()))
    (backup,) = dotfiles.glob("pi-models.json.bak.*")
    assert backup.read_text() == original


def test_merge_yaml_null_document_and_invalid_template_edges():
    from rapid_mlx.agents.adapter import _merge_yaml, _MergeParseError

    # An explicit YAML null document is a fresh write, not a merge.
    assert _merge_yaml("~\n", "- id: a\n") == "- id: a\n"
    # A template that does not parse is reported, never written over a file.
    with pytest.raises(_MergeParseError, match="rendered template"):
        _merge_yaml("- id: a\n", "key: [unclosed\n")
