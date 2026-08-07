# SPDX-License-Identifier: Apache-2.0
"""``agents <x> --setup`` must not touch the operator's real config.

The Tier-1 release gate runs `agents codex --setup` and `agents hermes --setup`
on the machine that also happens to be someone's daily driver. It protected
the real config by backing it up and restoring it afterwards, which failed in
two ways: the restore is skipped on SIGKILL, and once a config had been
clobbered by any run, every later run faithfully backed up and restored the
*damaged* file — so a codex install stayed pointed at a local rapid-mlx server
for weeks while each run's restore appeared to work.

Both CLIs relocate their whole config directory with an environment variable
(``CODEX_HOME`` / ``HERMES_HOME``). Honouring it means the gate can redirect
the write instead of trusting itself to put things back.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vllm_mlx.agents import get_profile
from vllm_mlx.agents.adapter import _resolve_config_path, setup_agent_config


@pytest.mark.parametrize(
    ("agent", "home_env", "filename"),
    [("codex", "CODEX_HOME", "config.toml"), ("hermes", "HERMES_HOME", "config.yaml")],
)
def test_profile_declares_its_home_env(agent: str, home_env: str, filename: str):
    """Each agent profile names the variable its own CLI honours."""
    cfg = get_profile(agent).config
    assert cfg.home_env == home_env
    assert Path(os.path.expanduser(cfg.path)).name == filename


@pytest.mark.parametrize(
    ("agent", "home_env"), [("codex", "CODEX_HOME"), ("hermes", "HERMES_HOME")]
)
def test_home_env_redirects_the_config_path(agent, home_env, tmp_path, monkeypatch):
    cfg = get_profile(agent).config
    default = Path(os.path.expanduser(cfg.path))

    monkeypatch.delenv(home_env, raising=False)
    assert _resolve_config_path(cfg) == default

    monkeypatch.setenv(home_env, str(tmp_path))
    redirected = _resolve_config_path(cfg)
    assert redirected == tmp_path / default.name
    assert redirected != default
    # Only the file name carries over — a relocated home must not inherit the
    # real one's directory layout.
    assert redirected.parent == tmp_path


@pytest.mark.parametrize(
    ("agent", "home_env"), [("codex", "CODEX_HOME"), ("hermes", "HERMES_HOME")]
)
def test_empty_or_blank_home_env_falls_back(agent, home_env, monkeypatch):
    """An exported-but-empty variable must not send the config to the cwd."""
    cfg = get_profile(agent).config
    default = Path(os.path.expanduser(cfg.path))
    for blank in ("", "   "):
        monkeypatch.setenv(home_env, blank)
        assert _resolve_config_path(cfg) == default


def test_setup_writes_to_the_redirected_home_and_leaves_the_real_one_alone(
    tmp_path, monkeypatch
):
    """The end-to-end property the release gate depends on."""
    real_home = tmp_path / "real-home"
    (real_home / ".codex").mkdir(parents=True)
    sentinel = real_home / ".codex" / "config.toml"
    sentinel.write_text('model = "gpt-5.6"\nmodel_provider = "openai"\n')
    before = sentinel.read_text()

    redirected = tmp_path / "throwaway"
    redirected.mkdir()

    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("CODEX_HOME", str(redirected))

    setup_agent_config(get_profile("codex"), base_url="http://localhost:8000/v1")

    assert (redirected / "config.toml").exists(), "setup did not write to CODEX_HOME"
    assert "rapid-mlx" in (redirected / "config.toml").read_text()
    # The operator's file is untouched, byte for byte.
    assert sentinel.read_text() == before


@pytest.mark.parametrize(
    ("agent", "home_env", "version"),
    [
        ("hermes", "HERMES_HOME", "0.8.5"),
        ("hermes", "HERMES_HOME", "0.9.1"),
        ("codex", "CODEX_HOME", "0.8.5"),
    ],
)
def test_version_specific_config_still_honours_the_home_env(agent, home_env, version):
    """``--agent-version`` must not walk around the redirect.

    A ``versions:`` block in the profile YAML is a *complete replacement* config,
    so one that simply does not spell out ``home_env`` used to resolve straight
    back to the operator's real file even with the variable set. Hermes 0.8.x
    was exactly that shape: ``HERMES_HOME=/tmp/x agents hermes --setup
    --agent-version 0.8.5`` rewrote the real ``~/.hermes/config.yaml``. The
    redirect is a safety property of the agent, not a per-version detail.
    """
    profile = get_profile(agent)
    versioned = profile.get_config_for_version(version)
    assert versioned.home_env == profile.config.home_env == home_env


def test_no_resolvable_version_escapes_the_redirect():
    """Sweep the version axis: every resolution keeps the redirect.

    Structural rather than example-based, so a *future* ``versions:`` block that
    forgets ``home_env`` fails here instead of silently rewriting someone's real
    config during a release gate.
    """
    candidates = [
        "0.1.0",
        "0.7.9",
        "0.8.0",
        "0.8.5",
        "0.9.0",
        "0.9.1",
        "1.0.0",
        "2.5.3",
    ]
    for agent in ("codex", "hermes"):
        profile = get_profile(agent)
        expected = profile.config.home_env
        assert expected, f"{agent} base config lost its home_env"
        for version in [None, *candidates]:
            resolved = profile.get_config_for_version(version)
            assert resolved.home_env == expected, (
                f"{agent} --agent-version {version} resolves to a config with "
                f"home_env={resolved.home_env!r}, escaping {expected}"
            )


def test_a_user_profile_cannot_drop_the_redirect():
    """`~/.rapid-mlx/agents/codex.yaml` replaces the built-in wholesale.

    A hand-written profile that never mentions ``home_env`` would otherwise
    remove the redirect entirely and send ``--setup`` back to the operator's
    real config — reintroduced by a file whose author was thinking about
    models, not about config safety.
    """
    from vllm_mlx.agents import _keep_home_env
    from vllm_mlx.agents.base import AgentConfigSpec

    builtin = get_profile("codex")
    assert builtin.config.home_env == "CODEX_HOME"

    from dataclasses import replace as _replace

    silent = _replace(
        builtin,
        config=AgentConfigSpec(type="toml", path="~/.codex/config.toml"),
    )
    assert silent.config.home_env is None  # what the loader would have produced
    assert _keep_home_env(silent, builtin).config.home_env == "CODEX_HOME"

    # Choosing a *different* variable is still the profile author's call.
    relocated = _replace(
        builtin,
        config=AgentConfigSpec(
            type="toml", path="~/.codex/config.toml", home_env="MY_CODEX_HOME"
        ),
    )
    assert _keep_home_env(relocated, builtin).config.home_env == "MY_CODEX_HOME"


def test_a_version_that_declares_its_own_home_env_keeps_it():
    """Inheritance fills a hole; it never overrides a deliberate choice.

    A version block that genuinely relocates says so, and that has to win —
    otherwise the redirect would point at the *old* agent's home and the new
    one's real config would be written instead.
    """
    from dataclasses import replace as _replace

    from vllm_mlx.agents.base import AgentConfigSpec, AgentVersionSpec

    profile = get_profile("hermes")
    relocated = AgentConfigSpec(
        type="yaml", path="~/.new/settings.yaml", home_env="NEW_AGENT_HOME"
    )
    probe = _replace(
        profile,
        versions=[AgentVersionSpec(version_range=">=99", config=relocated)],
    )
    assert probe.get_config_for_version("99.1").home_env == "NEW_AGENT_HOME"
    # ...and a version with no block of its own still falls back to the base.
    assert probe.get_config_for_version("1.0").home_env == profile.config.home_env
