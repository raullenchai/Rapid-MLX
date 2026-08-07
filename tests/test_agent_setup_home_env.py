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
