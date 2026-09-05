# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for ``rapid-mlx start`` (#150 one-command agent startup).

``start`` is an orchestrator: it resolves a profile + model, consents to a
download, spawns ``serve`` as a foreground parent-owned child (or reuses a
compatible server), then prints/applies agent config. These tests cover the
orchestration decisions without touching the network, the HF cache, or
spawning real servers — every side effect is monkeypatched at the module
boundary of ``vllm_mlx.run.cli`` (and the source modules it imports from
inside each helper).
"""

from __future__ import annotations

import argparse
import signal
import types

import pytest

import vllm_mlx.recommendations as rec
from vllm_mlx import _download_gate as dg
from vllm_mlx.run import cli as run_cli


def _make_args(**overrides) -> argparse.Namespace:
    """Build a ``start`` argument namespace with quiet defaults."""
    base = dict(
        profile="hermes",
        model=None,
        port=8000,
        host="127.0.0.1",
        no_download=False,
        dry_run=False,
        yes=False,
        no_setup=False,
        ready_timeout=600,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class _FakeProfile:
    """Minimal stand-in for ``agents.AgentProfile`` (fields ``start`` reads)."""

    def __init__(self, name="hermes", recommended_models=None, config=None):
        self.name = name
        self.display_name = name.title()
        if recommended_models is None:
            recommended_models = ["qwen3.5-9b-4bit"]
        self.recommended_models = recommended_models
        self.kind = "agent"
        self.config = config

    def get_config_for_version(self, version):  # pragma: no cover - trivial
        return self.config


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _patch_select_resources(monkeypatch, cached=None, fit=True, ram=64.0):
    """Patch the symbols ``_select_model`` imports (freshly, at call time)."""
    monkeypatch.setattr(rec, "physical_ram_gb", lambda: ram)
    if fit is True:
        monkeypatch.setattr(rec, "is_recommended_alias", lambda alias, r: True)
    elif fit is False:
        monkeypatch.setattr(rec, "is_recommended_alias", lambda alias, r: False)
    else:
        monkeypatch.setattr(rec, "is_recommended_alias", fit)
    if cached is not None:
        monkeypatch.setattr(dg, "is_repo_cached", cached)
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: a)


def test_select_explicit_model_wins(monkeypatch):
    """``--model`` is served verbatim, skipping all recommended logic."""
    picked = run_cli._select_model(
        explicit="mlx-community/Explicit-4bit", profile=None, no_download=True
    )
    assert picked == "mlx-community/Explicit-4bit"


def test_select_cached_recommended_picked(monkeypatch):
    """First recommended model that fits RAM and is cached wins."""
    profile = _FakeProfile(recommended_models=["qwen3.5-9b-4bit", "qwen3.6-27b-4bit"])
    _patch_select_resources(
        monkeypatch, cached=lambda alias: alias == "qwen3.6-27b-4bit"
    )

    picked = run_cli._select_model(explicit=None, profile=profile, no_download=False)
    assert picked == "qwen3.6-27b-4bit"


def test_select_first_fit_when_nothing_cached(monkeypatch):
    """With downloads allowed, the first recommended (declaration order) fits."""
    profile = _FakeProfile(recommended_models=["qwen3.6-27b-4bit", "qwen3.5-9b-4bit"])
    _patch_select_resources(monkeypatch, cached=lambda alias: False)

    picked = run_cli._select_model(explicit=None, profile=profile, no_download=False)
    assert picked == "qwen3.6-27b-4bit"


def test_select_no_download_refuses_when_nothing_cached(monkeypatch, capsys):
    """--no-download with no cached fitting model returns None + message."""
    profile = _FakeProfile(recommended_models=["qwen3.5-9b-4bit"])
    _patch_select_resources(monkeypatch, cached=lambda alias: False)

    picked = run_cli._select_model(explicit=None, profile=profile, no_download=True)
    assert picked is None
    assert "--no-download" in capsys.readouterr().out


def test_select_nothing_fits(monkeypatch, capsys):
    """Every recommended model rejected by RAM -> None + message."""
    profile = _FakeProfile(recommended_models=["qwen3.6-27b-4bit"])
    _patch_select_resources(monkeypatch, cached=lambda alias: False, fit=False)
    monkeypatch.setattr(run_cli, "_fits_host", lambda alias, ram: False)

    picked = run_cli._select_model(explicit=None, profile=profile, no_download=False)
    assert picked is None
    assert "fit" in capsys.readouterr().out


def test_select_cached_agent_model_with_unknown_fit_is_not_rejected(monkeypatch):
    """Agent-only recommendations outside the global tiers remain eligible."""
    alias = "qwen3-coder-30b-4bit"
    profile = _FakeProfile(recommended_models=[alias])
    _patch_select_resources(monkeypatch, cached=lambda candidate: candidate == alias)

    assert (
        run_cli._select_model(explicit=None, profile=profile, no_download=True) == alias
    )


def test_select_no_recommended_models_errors(monkeypatch, capsys):
    """Profile with an empty recommended list -> clear error, pass --model."""
    profile = _FakeProfile(recommended_models=[])
    picked = run_cli._select_model(explicit=None, profile=profile, no_download=False)
    assert picked is None
    assert "--model" in capsys.readouterr().out


def test_select_generic_uses_starter(monkeypatch):
    """No profile and no model -> the first-run chat starter alias."""
    from vllm_mlx import first_run as fr

    monkeypatch.setattr(fr, "select_chat_default", lambda: ("qwen3.5-4b-4bit", True))
    picked = run_cli._select_model(explicit=None, profile=None, no_download=False)
    assert picked == "qwen3.5-4b-4bit"


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


def test_consent_cached_passes(monkeypatch):
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: a)
    monkeypatch.setattr(dg, "is_repo_cached", lambda alias: True)
    assert run_cli._confirm_download("m", no_download=False, yes=False) is True


def test_consent_no_download_uncached_refuses(monkeypatch, capsys):
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: a)
    monkeypatch.setattr(dg, "is_repo_cached", lambda alias: False)
    assert run_cli._confirm_download("m", no_download=True, yes=False) is False
    assert "refusing" in capsys.readouterr().out


def test_consent_yes_bypasses_prompt(monkeypatch):
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: a)
    monkeypatch.setattr(dg, "is_repo_cached", lambda alias: False)

    def fail_confirm(*a, **k):
        raise AssertionError("should not prompt with --yes")

    monkeypatch.setattr(dg, "confirm_or_abort", fail_confirm)
    assert run_cli._confirm_download("m", no_download=False, yes=True) is True


def test_consent_routes_to_confirm_or_abort(monkeypatch):
    """uncached + interactive + no --yes -> confirm_or_abort decides."""
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: a)
    monkeypatch.setattr(dg, "is_repo_cached", lambda alias: False)
    monkeypatch.setattr(dg, "estimate_download_size_bytes", lambda alias: 123)
    calls = {}

    def fake_confirm(repo_id, size):
        calls["repo"] = repo_id
        calls["size"] = size
        return False

    monkeypatch.setattr(dg, "confirm_or_abort", fake_confirm)
    # Force a TTY stdin so the gate reaches confirm_or_abort.
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))

    result = run_cli._confirm_download("m", no_download=False, yes=False)
    assert result is False
    assert calls.get("repo") == "m"
    assert calls.get("size") == 123


def test_consent_auto_pull_env_skips(monkeypatch):
    """RAPID_MLX_AUTO_PULL=1 bypasses the prompt (CI path)."""
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: a)
    monkeypatch.setattr(dg, "is_repo_cached", lambda alias: False)
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "1")
    assert run_cli._confirm_download("m", no_download=False, yes=False) is True


# ---------------------------------------------------------------------------
# Spawn + reuse
# ---------------------------------------------------------------------------


def test_spawn_foreground_child_env(monkeypatch):
    """Serve child: isolated signal group + gate + watchdog env."""
    import os
    import subprocess as real_subprocess

    args = _make_args()
    captured = {}

    class FakeProc:
        pass

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        captured["start_new_session"] = kw.get("start_new_session", False)
        return FakeProc()

    monkeypatch.setattr(real_subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_cli, "subprocess", real_subprocess)

    proc = run_cli._spawn_foreground_serve("qwen3.5-9b-4bit", args)
    assert isinstance(proc, FakeProc)
    assert captured["start_new_session"] is True  # parent is sole signal relay
    assert captured["cmd"][:6] == [
        real_subprocess.sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        "qwen3.5-9b-4bit",
        "--host",
    ]
    assert captured["env"]["RAPID_MLX_CHAT_SPAWN"] == "1"
    assert captured["env"]["RAPID_MLX_WATCHDOG_PPID"] == str(os.getpid())


def test_spawn_no_download_forces_child_offline(monkeypatch):
    """--no-download remains strict inside the canonical serve child."""
    import subprocess as real_subprocess

    captured = {}
    monkeypatch.setattr(
        real_subprocess,
        "Popen",
        lambda cmd, **kwargs: captured.update(kwargs) or types.SimpleNamespace(),
    )
    run_cli._spawn_foreground_serve("cached", _make_args(no_download=True))
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_reuse_compatible_server_attaches_no_spawn(monkeypatch, capsys):
    """Port serving the chosen model -> attach, never spawn."""
    args = _make_args()
    toggles = {"attached": False}

    from vllm_mlx.agents import adapter as ad

    fetched = []
    monkeypatch.setattr(
        ad,
        "_fetch_models",
        lambda base: fetched.append(base) or [{"id": "qwen3.5-9b-4bit"}],
    )
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: "qwen3.5-9b-4bit")
    monkeypatch.setattr(
        run_cli,
        "_attach_and_configure",
        lambda base_url, model, profile, a: toggles.update(attached=True) or 0,
    )

    result = run_cli._reuse_or_refuse(
        "http://127.0.0.1:8000",
        "qwen3.5-9b-4bit",
        "qwen3.5-9b-4bit",
        _FakeProfile(),
        args,
    )
    assert result == 0
    assert toggles["attached"] is True
    assert fetched == ["http://127.0.0.1:8000/v1"]
    assert "Reusing" in capsys.readouterr().out


def test_reuse_incompatible_server_refuses(monkeypatch, capsys):
    """Port serving a different model -> clean refusal (exit 1)."""
    args = _make_args()
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(ad, "_fetch_models", lambda base: [{"id": "other-model"}])
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: "qwen3.5-9b-4bit")

    result = run_cli._reuse_or_refuse(
        "http://127.0.0.1:8000",
        "qwen3.5-9b-4bit",
        "qwen3.5-9b-4bit",
        _FakeProfile(),
        args,
    )
    assert result == 1
    assert "already serves" in capsys.readouterr().out


def test_reuse_occupied_not_rapidmlx_refuses(monkeypatch, capsys):
    """Port occupied by a non-rapid-mlx listener -> clean refusal."""
    args = _make_args()
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(ad, "_fetch_models", lambda base: [])
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: "qwen3.5-9b-4bit")

    result = run_cli._reuse_or_refuse(
        "http://127.0.0.1:8000",
        "qwen3.5-9b-4bit",
        "qwen3.5-9b-4bit",
        _FakeProfile(),
        args,
    )
    assert result == 1
    assert "not a healthy" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# start_command orchestration
# ---------------------------------------------------------------------------


def test_start_command_dry_run_no_side_effects(monkeypatch, capsys):
    """--dry-run never spawns, downloads, or writes config."""
    args = _make_args(dry_run=True)
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "qwen3.5-9b-4bit")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(
        run_cli, "_spawn_foreground_serve", lambda *a: pytest.fail("spawned")
    )
    monkeypatch.setattr(
        run_cli, "_confirm_download", lambda *a, **k: pytest.fail("consented")
    )

    assert run_cli.start_command(args) == 0
    assert "Dry run" in capsys.readouterr().out


def _patch_profile_lookup(monkeypatch, profile):
    """Patch ``vllm_mlx.agents.get_profile`` (imported inside start_command)."""
    import vllm_mlx.agents as agents_mod

    monkeypatch.setattr(agents_mod, "get_profile", lambda n: profile)


def test_start_command_unknown_profile(monkeypatch, capsys):
    """Unknown profile -> clear error, return 1, never selected/spawned."""
    args = _make_args(profile="nope")
    _patch_profile_lookup(monkeypatch, None)
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: pytest.fail("selected"))

    assert run_cli.start_command(args) == 1
    assert "Unknown agent" in capsys.readouterr().out


def test_start_command_port_reuse_path(monkeypatch, capsys):
    """Occupied compatible port -> reuse handler runs, no spawn path."""
    args = _make_args()
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "qwen3.5-9b-4bit")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: True)
    monkeypatch.setattr(run_cli, "_reuse_or_refuse", lambda *a, **k: 0)
    monkeypatch.setattr(
        run_cli, "_spawn_foreground_serve", lambda *a: pytest.fail("spawned")
    )

    assert run_cli.start_command(args) == 0


def test_start_command_spawn_then_wait(monkeypatch):
    """Normal path: consent -> spawn -> wait-ready -> configure -> wait-child."""
    args = _make_args()
    order = []
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(
        run_cli, "_select_model", lambda **kw: order.append("select") or "m"
    )
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(
        run_cli, "_confirm_download", lambda *a, **k: order.append("confirm") or True
    )
    monkeypatch.setattr(
        run_cli, "_spawn_foreground_serve", lambda *a: order.append("spawn") or object()
    )
    monkeypatch.setattr(
        run_cli, "_wait_ready", lambda *a, **k: order.append("ready") or "ready"
    )
    monkeypatch.setattr(
        run_cli, "_attach_and_configure", lambda *a, **k: order.append("configure") or 0
    )
    monkeypatch.setattr(run_cli, "_wait_child", lambda p: order.append("wait") or 3)

    assert run_cli.start_command(args) == 3
    assert order == ["select", "confirm", "spawn", "ready", "configure", "wait"]


def test_start_command_preserves_explicit_alias_identity(monkeypatch):
    """Resolved repos gate downloads while serve/config retain the alias."""
    args = _make_args(model="org/model")
    args._original_alias = "friendly-alias"
    seen = {}
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "org/model")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(
        run_cli,
        "_confirm_download",
        lambda model, **kw: seen.update(download=model) or True,
    )
    monkeypatch.setattr(
        run_cli,
        "_spawn_foreground_serve",
        lambda model, a: seen.update(spawn=model) or object(),
    )
    monkeypatch.setattr(run_cli, "_wait_ready", lambda *a, **k: "ready")
    monkeypatch.setattr(
        run_cli,
        "_attach_and_configure",
        lambda base, model, *a: seen.update(config=model) or 0,
    )
    monkeypatch.setattr(run_cli, "_wait_child", lambda proc: 0)

    assert run_cli.start_command(args) == 0
    assert seen == {
        "download": "org/model",
        "spawn": "friendly-alias",
        "config": "friendly-alias",
    }


def test_start_command_setup_failure_stops_owned_child(monkeypatch):
    """Failed setup exits nonzero and cannot leave a spawned server behind."""
    args = _make_args()
    proc = object()
    stopped = []
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(run_cli, "_confirm_download", lambda *a, **k: True)
    monkeypatch.setattr(run_cli, "_spawn_foreground_serve", lambda *a: proc)
    monkeypatch.setattr(run_cli, "_wait_ready", lambda *a, **k: "ready")
    monkeypatch.setattr(run_cli, "_attach_and_configure", lambda *a, **k: 1)
    monkeypatch.setattr(run_cli, "_terminate_child", lambda p: stopped.append(p))
    monkeypatch.setattr(
        run_cli, "_wait_child", lambda p: pytest.fail("must not wait indefinitely")
    )

    assert run_cli.start_command(args) == 1
    assert stopped == [proc]


def test_start_command_spawn_after_consent_refused(monkeypatch):
    """Consent refused -> return 1, never spawn."""
    args = _make_args()
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(run_cli, "_confirm_download", lambda *a, **k: False)
    monkeypatch.setattr(
        run_cli, "_spawn_foreground_serve", lambda *a: pytest.fail("spawned")
    )

    assert run_cli.start_command(args) == 1


def test_start_command_spawn_error_is_clean(monkeypatch, capsys):
    """Popen failures become a concise command failure, not a traceback."""
    args = _make_args()
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(run_cli, "_confirm_download", lambda *a, **k: True)
    monkeypatch.setattr(
        run_cli,
        "_spawn_foreground_serve",
        lambda *a: (_ for _ in ()).throw(OSError("fork failed")),
    )

    assert run_cli.start_command(args) == 1
    assert "Could not start the server process: fork failed" in capsys.readouterr().out


def test_start_command_readiness_exit(monkeypatch):
    """Serve child that exits before ready -> reap, return its code, no config.

    Exercise the ``outcome == \"exited\"`` branch: the child died before
    /health/ready returned, so start reaps it (its code) rather than
    configuring the agent or printing a traceback.
    """
    args = _make_args()
    order = []
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(run_cli, "_confirm_download", lambda *a, **k: True)
    monkeypatch.setattr(
        run_cli, "_spawn_foreground_serve", lambda *a: order.append("spawn") or object()
    )
    monkeypatch.setattr(run_cli, "_wait_ready", lambda *a, **k: "exited")
    monkeypatch.setattr(
        run_cli, "_attach_and_configure", lambda *a, **k: order.append("configure") or 0
    )
    monkeypatch.setattr(run_cli, "_wait_child", lambda p: order.append("wait") or 6)

    assert run_cli.start_command(args) == 6
    # configure must NOT run when the serve child never became ready.
    assert order == ["spawn", "wait"]


def test_start_command_readiness_timeout_terminates(monkeypatch):
    """Ready-timeout with a STILL-ALIVE child -> terminate, reap, fail.

    This is the HIGH hang fix: a timeout must NOT turn into an indefinite
    ``_wait_child`` wait. The child (alive, still loading) is terminated,
    then reaped; configure never runs.
    """
    args = _make_args()
    order = []
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(run_cli, "_confirm_download", lambda *a, **k: True)
    monkeypatch.setattr(
        run_cli, "_spawn_foreground_serve", lambda *a: order.append("spawn") or object()
    )
    monkeypatch.setattr(run_cli, "_wait_ready", lambda *a, **k: "timeout")
    terminated = []
    monkeypatch.setattr(run_cli, "_terminate_child", lambda p: terminated.append(True))
    monkeypatch.setattr(
        run_cli, "_attach_and_configure", lambda *a, **k: order.append("configure") or 0
    )
    monkeypatch.setattr(
        run_cli, "_wait_child", lambda p: pytest.fail("cleanup is already bounded")
    )

    assert run_cli.start_command(args) == 124
    assert terminated == [True]  # child was terminated on timeout
    assert order == ["spawn"]


def test_start_command_readiness_keyboard(monkeypatch):
    """Ctrl-C during readiness -> reap, no configure."""
    args = _make_args()
    order = []
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: False)
    monkeypatch.setattr(run_cli, "_confirm_download", lambda *a, **k: True)
    monkeypatch.setattr(
        run_cli, "_spawn_foreground_serve", lambda *a: order.append("spawn") or object()
    )
    monkeypatch.setattr(run_cli, "_wait_ready", lambda *a, **k: "interrupted")
    monkeypatch.setattr(
        run_cli, "_attach_and_configure", lambda *a, **k: order.append("configure") or 0
    )
    stopped = []
    monkeypatch.setattr(run_cli, "_terminate_child", lambda p: stopped.append(p))
    monkeypatch.setattr(
        run_cli, "_wait_child", lambda p: pytest.fail("cleanup is already bounded")
    )

    assert run_cli.start_command(args) == 128 + signal.SIGINT
    assert len(stopped) == 1
    assert order == ["spawn"]


# ---------------------------------------------------------------------------
# Child lifecycle
# ---------------------------------------------------------------------------


def test_wait_child_propagates_exit_code(monkeypatch):
    """Child's nonzero exit code is returned as start's exit code."""
    import time as real_time

    class FakeChild:
        def __init__(self):
            self.returncode = None
            self.n = 0

        def poll(self):
            self.n += 1
            if self.n >= 3:
                self.returncode = 7
            return self.returncode

        def send_signal(self, signum):  # pragma: no cover - not reached
            pass

    monkeypatch.setattr(real_time, "sleep", lambda s: None)
    assert run_cli._wait_child(FakeChild()) == 7


# ---------------------------------------------------------------------------
# Dispatch + registration
# ---------------------------------------------------------------------------


def test_start_registers_subcommand():
    """``start`` is a registered subcommand of the top-level CLI."""
    from vllm_mlx.cli import build_parser

    choices = build_parser()._subparsers._group_actions[0].choices
    assert "start" in choices


def test_start_main_dispatch_routes(monkeypatch):
    """``main()`` routes ``rapid-mlx start ...`` to ``start_command``.

    Covers the ``elif args.command == "start"`` dispatch branch in cli.py
    (registration is covered above; the runtime routing needs a full
    ``main()`` round-trip so changed-lines coverage sees it).
    """
    import sys

    from vllm_mlx import cli

    captured = {}

    def fake_start(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "start", "hermes", "--dry-run"])
    # The dispatch does ``from vllm_mlx.run.cli import start_command`` inside
    # main(); patch the true source module so that import picks up the fake.
    monkeypatch.setattr("vllm_mlx.run.cli.start_command", fake_start)
    cli.main()
    assert captured["args"].profile == "hermes"
    assert captured["args"].dry_run is True


def test_start_main_dispatch_propagates_exit_code(monkeypatch):
    """A nonzero start_command return raises SystemExit(code) at dispatch."""
    import sys

    from vllm_mlx import cli

    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "start", "nope", "--dry-run"])
    monkeypatch.setattr("vllm_mlx.run.cli.start_command", lambda args: 3)
    # The dispatch's ``if code: raise SystemExit(code)`` fires -> SystemExit(3).
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 3


def test_run_alias_still_routes_to_chat(monkeypatch):
    """Regression guard: #150 must NOT hijack ``run`` (stays a chat alias)."""
    from vllm_mlx.cli import build_parser

    # ``run`` resolves to the chat parser: it accepts chat-only flags.
    ns = build_parser().parse_args(["run", "qwen3.5-4b-4bit", "--think"]).__dict__
    assert ns.get("command") == "run"
    assert "think" in ns
    assert "max_tokens" in ns


def test_start_help_is_distinct_from_chat():
    """``start`` help must not advertise chat flags (distinct verb)."""
    from vllm_mlx.cli import build_parser

    choices = build_parser()._subparsers._group_actions[0].choices
    start_p = choices["start"]
    seen = {a.dest for a in start_p._actions}
    assert "profile" in seen
    assert "no_download" in seen
    assert "dry_run" in seen
    # Chat's model positional dest is ``model``; start profiles under
    # ``profile`` so the two verbs never share semantics.
    assert "model" in seen


# ---------------------------------------------------------------------------
# Coverage hardening: real helper bodies + wrappers + attach/configure
# ---------------------------------------------------------------------------


def test_start_command_model_none_returns_1(monkeypatch):
    """start_command returns 1 (with no spawn) when selection yields None."""
    args = _make_args()
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: None)
    monkeypatch.setattr(
        run_cli, "_spawn_foreground_serve", lambda *a: pytest.fail("spawned")
    )
    assert run_cli.start_command(args) == 1


def test_hf_id_real_resolves(monkeypatch):
    """Real ``_hf_id`` maps an alias via resolve_model (and degrades on error)."""

    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_model",
        lambda name: "mlx-community/Resolved-4bit",
    )
    assert run_cli._hf_id("anything") == "mlx-community/Resolved-4bit"

    # Exception path: resolve_model raising -> falls back to the alias.
    def boom(name):
        raise RuntimeError("boom")

    monkeypatch.setattr("vllm_mlx.model_aliases.resolve_model", boom)
    assert run_cli._hf_id("flat-name") == "flat-name"


def test_consent_no_download_cached_returns_true(monkeypatch):
    """--no-download with a cached model passes (nothing to download)."""
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: a)
    monkeypatch.setattr(dg, "is_repo_cached", lambda alias: True)
    assert run_cli._confirm_download("m", no_download=True, yes=False) is True


def test_consent_no_download_accepts_existing_local_model(tmp_path):
    """A local checkpoint path never needs a Hugging Face cache entry."""
    model = tmp_path / "model"
    model.mkdir()
    assert run_cli._confirm_download(str(model), no_download=True, yes=False) is True


def test_consent_size_estimate_raises_handled(monkeypatch):
    """A size-estimate network failure is swallowed; gate still runs."""
    monkeypatch.setattr(run_cli, "_hf_id", lambda a: a)
    monkeypatch.setattr(dg, "is_repo_cached", lambda alias: False)
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
    calls = {}
    monkeypatch.setattr(
        dg,
        "estimate_download_size_bytes",
        lambda alias: (_ for _ in ()).throw(RuntimeError("network")),
    )
    monkeypatch.setattr(
        dg,
        "confirm_or_abort",
        lambda repo_id, size: calls.update(repo=repo_id, size=size) or False,
    )
    assert run_cli._confirm_download("m", no_download=False, yes=False) is False
    assert calls.get("repo") == "m"
    assert calls.get("size") is None  # estimate failed -> None passed through


def test_wait_ready_wrapper(monkeypatch):
    """_wait_ready delegates to cli._wait_for_chat_server, returns "ready"."""
    from vllm_mlx import cli as cli_mod

    got = {}
    monkeypatch.setattr(
        cli_mod,
        "_wait_for_chat_server",
        lambda b, p, timeout_s=None: got.update(b=b, p=p, t=timeout_s) or None,
    )
    assert run_cli._wait_ready("http://x", "proc", 30) == "ready"
    assert got == {"b": "http://x", "p": "proc", "t": 30}


def test_wait_ready_reports_exit(monkeypatch, capsys):
    """A serve child that exits early -> "exited" + clean message."""
    from vllm_mlx import cli as cli_mod

    class DeadProc:
        returncode = 1

        def poll(self):
            return 1

    def fail(base_url, proc, timeout_s=None):
        raise RuntimeError("server exited early")

    monkeypatch.setattr(cli_mod, "_wait_for_chat_server", fail)
    assert run_cli._wait_ready("http://x", DeadProc(), 5) == "exited"
    out = capsys.readouterr().out
    assert "before becoming ready" in out


def test_wait_ready_reports_timeout(monkeypatch, capsys):
    """A readiness timeout -> "timeout" (child STILL alive) + clean message."""
    from vllm_mlx import cli as cli_mod

    class AliveProc:
        def poll(self):
            return None

    def timeout(base_url, proc, timeout_s=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(cli_mod, "_wait_for_chat_server", timeout)
    assert run_cli._wait_ready("http://x", AliveProc(), 5) == "timeout"
    assert "did not become ready" in capsys.readouterr().out


def test_wait_ready_keyboard_interrupt(monkeypatch, capsys):
    """A Ctrl-C during readiness -> "exited" without a raw traceback."""
    from vllm_mlx import cli as cli_mod

    def intr(base_url, proc, timeout_s=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "_wait_for_chat_server", intr)
    assert run_cli._wait_ready("http://x", object(), 5) == "interrupted"
    assert "Interrupted during startup" in capsys.readouterr().out


def test_port_is_busy_wrapper(monkeypatch):
    """_port_is_busy delegates to cli._port_is_busy."""
    from vllm_mlx import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_port_is_busy", lambda h, p: h == "127.0.0.1")
    assert run_cli._port_is_busy("127.0.0.1", 8000) is True
    assert run_cli._port_is_busy("0.0.0.0", 8000) is False


def test_reuse_dry_run_branch(monkeypatch, capsys):
    """Reuse handler under --dry-run probes before promising an attach."""
    args = _make_args(dry_run=True)
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(ad, "_fetch_models", lambda base: [{"id": "m"}])
    monkeypatch.setattr(run_cli, "_hf_id", lambda model: model)
    previews = []
    monkeypatch.setattr(
        run_cli,
        "_attach_and_configure",
        lambda base_url, model, profile, seen_args: (
            previews.append((base_url, model, seen_args.dry_run)) or 0
        ),
    )
    result = run_cli._reuse_or_refuse(
        "http://127.0.0.1:8000", "m", "m", _FakeProfile(), args
    )
    assert result == 0
    assert previews == [("http://127.0.0.1:8000", "m", True)]
    assert "would reuse" in capsys.readouterr().out


def test_reuse_dry_run_refuses_incompatible_server(monkeypatch, capsys):
    """Dry-run reports the same incompatible-port failure as a real start."""
    args = _make_args(dry_run=True)
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(ad, "_fetch_models", lambda base: [{"id": "other"}])
    monkeypatch.setattr(run_cli, "_hf_id", lambda model: model)
    assert (
        run_cli._reuse_or_refuse(
            "http://127.0.0.1:8000", "m", "m", _FakeProfile(), args
        )
        == 1
    )
    assert "not m" in capsys.readouterr().out


# --- _attach_and_configure + _print_instructions ---------------------------


class _FakeCfg:
    def __init__(self, template=None, config_type="json"):
        self.template = template
        self.type = config_type


class _SetupPlanFake:
    def __init__(self, path="p", changed=True, diff="@@ diff"):
        self.path = path
        self.changed = changed
        self._diff = diff

    def diff(self):
        return self._diff


def _first_class_profile():
    p = _FakeProfile(name="claude-code", recommended_models=["qwen3.5-9b-4bit"])
    p.display_name = "Claude Code"
    return p


def test_attach_generic_profile_is_none(monkeypatch, capsys):
    """profile None -> prints generic endpoint, no config write."""
    args = _make_args()
    rc = run_cli._attach_and_configure("http://127.0.0.1:8000", "m", None, args)
    assert rc == 0
    assert "OpenAI-compatible endpoint" in capsys.readouterr().out


def test_attach_no_setup_prints_instructions(monkeypatch, capsys):
    """--no-setup prints instructions without writing config."""
    args = _make_args(no_setup=True)
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(
        ad, "get_setup_instructions", lambda *a, **k: "  instructions here"
    )
    rc = run_cli._attach_and_configure(
        "http://127.0.0.1:8000", "m", _FakeProfile(), args
    )
    assert rc == 0
    assert "instructions here" in capsys.readouterr().out


def test_attach_first_class_template_fetch_context(monkeypatch, capsys):
    """First-class profile with {context_length} template fetches it."""
    args = _make_args(yes=True)
    prof = _first_class_profile()
    prof.config = _FakeCfg(template="  {context_length}")

    import vllm_mlx.agents.setup as setup_mod
    from vllm_mlx.agents import adapter as ad

    fetches = {}
    monkeypatch.setattr(
        ad, "fetch_context_window", lambda b, m: fetches.update(b=b, m=m) or 32768
    )
    monkeypatch.setattr(
        setup_mod,
        "build_setup_plan",
        lambda *a, **k: _SetupPlanFake(changed=True, diff="+model = x"),
    )
    monkeypatch.setattr(setup_mod, "apply_setup_plan", lambda plan: plan.path)
    monkeypatch.setattr(setup_mod, "confirm_plan", lambda plan: True)

    rc = run_cli._attach_and_configure("http://b", "m", prof, args)
    assert rc == 0
    assert fetches.get("m") == "m"
    out = capsys.readouterr().out
    assert "Configured Claude Code" in out


def test_attach_first_class_context_fetch_fails(monkeypatch, capsys):
    """A failed context-length fetch degrades gracefully (context_length=None)."""
    args = _make_args(yes=True)
    prof = _first_class_profile()
    prof.config = _FakeCfg(template="  {context_length}")

    import vllm_mlx.agents.setup as setup_mod
    from vllm_mlx.agents import adapter as ad

    def boom(base_url, model):
        raise RuntimeError("server not ready")

    monkeypatch.setattr(ad, "fetch_context_window", boom)
    planned = {}
    monkeypatch.setattr(
        setup_mod,
        "build_setup_plan",
        lambda *a, **k: planned.update(k=k) or _SetupPlanFake(changed=True),
    )
    monkeypatch.setattr(setup_mod, "apply_setup_plan", lambda plan: plan.path)
    monkeypatch.setattr(setup_mod, "confirm_plan", lambda plan: True)

    rc = run_cli._attach_and_configure("http://b", "m", prof, args)
    assert rc == 0
    # context_length was requested (template) but fetch failed -> None passed.
    assert planned["k"].get("context_length") is None
    assert "Configured Claude Code" in capsys.readouterr().out


def test_attach_deepseek_reasoning_probe_fails_open(monkeypatch):
    """A transient reasoning probe failure does not block agent setup."""
    args = _make_args(yes=True)
    prof = _FakeProfile(name="deepseek-harness", recommended_models=["qwen3.5-9b-4bit"])
    prof.display_name = "DeepSeek Harness"
    prof.config = _FakeCfg(template=None)

    import vllm_mlx.agents.setup as setup_mod
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(
        ad,
        "fetch_reasoning_support",
        lambda *a, **k: (_ for _ in ()).throw(OSError("probe unavailable")),
    )
    planned = {}
    monkeypatch.setattr(
        setup_mod,
        "build_setup_plan",
        lambda *a, **k: planned.update(k=k) or _SetupPlanFake(changed=False),
    )

    assert run_cli._attach_and_configure("http://b", "m", prof, args) == 0
    assert planned["k"]["supports_reasoning"] is None


def test_attach_first_class_unchanged(monkeypatch, capsys):
    """First-class profile already configured: no-file-changes path."""
    args = _make_args()
    prof = _first_class_profile()
    prof.config = _FakeCfg(template=None)

    import vllm_mlx.agents.setup as setup_mod

    monkeypatch.setattr(
        setup_mod,
        "build_setup_plan",
        lambda *a, **k: _SetupPlanFake(changed=False),
    )
    rc = run_cli._attach_and_configure("http://b", "m", prof, args)
    assert rc == 0
    assert "Already configured" in capsys.readouterr().out


def test_attach_first_class_build_fails(monkeypatch, capsys):
    """First-class profile whose plan build raises -> instructions, rc 1."""
    args = _make_args()
    prof = _first_class_profile()
    prof.config = _FakeCfg(template=None)

    import vllm_mlx.agents.setup as setup_mod
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(
        setup_mod,
        "build_setup_plan",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no config")),
    )
    monkeypatch.setattr(
        ad, "get_setup_instructions", lambda *a, **k: "  fallback instructions"
    )
    rc = run_cli._attach_and_configure("http://b", "m", prof, args)
    assert rc == 1
    out = capsys.readouterr().out
    assert "setup failed" in out
    assert "fallback instructions" in out


def test_attach_first_class_apply_fails(monkeypatch, capsys):
    """First-class profile whose apply raises RuntimeError -> error print."""
    args = _make_args(yes=True)
    prof = _first_class_profile()
    prof.config = _FakeCfg(template=None)

    import vllm_mlx.agents.setup as setup_mod

    monkeypatch.setattr(
        setup_mod, "build_setup_plan", lambda *a, **k: _SetupPlanFake(changed=True)
    )
    monkeypatch.setattr(
        setup_mod,
        "apply_setup_plan",
        lambda p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(setup_mod, "confirm_plan", lambda p: True)
    rc = run_cli._attach_and_configure("http://b", "m", prof, args)
    assert rc == 1
    assert "setup failed" in capsys.readouterr().out


def test_attach_first_class_consent_cancelled(monkeypatch, capsys):
    """Consent declined for a changed first-class plan -> nothing written."""
    args = _make_args()  # yes=False
    prof = _first_class_profile()
    prof.config = _FakeCfg(template=None)

    import vllm_mlx.agents.setup as setup_mod

    monkeypatch.setattr(
        setup_mod, "build_setup_plan", lambda *a, **k: _SetupPlanFake(changed=True)
    )
    monkeypatch.setattr(setup_mod, "confirm_plan", lambda p: False)
    monkeypatch.setattr(
        setup_mod, "apply_setup_plan", lambda p: pytest.fail("should not apply")
    )
    rc = run_cli._attach_and_configure("http://b", "m", prof, args)
    assert rc == 0
    assert "Setup cancelled" in capsys.readouterr().out


def test_attach_generic_writer_success(monkeypatch, capsys):
    """Generic (non-first-class) profile writer success path."""
    args = _make_args(yes=True)
    prof = _FakeProfile(name="hermes", recommended_models=["qwen3.5-9b-4bit"])
    prof.config = _FakeCfg(template=None)

    from vllm_mlx.agents import adapter as ad

    called = {}
    monkeypatch.setattr(
        ad,
        "setup_agent_config",
        lambda profile, base_url, model, **kwargs: (
            called.update(base_url=base_url, model=model) or "wrote config"
        ),
    )
    rc = run_cli._attach_and_configure("http://b", "m", prof, args)
    assert rc == 0
    assert called == {"base_url": "http://b/v1", "model": "m"}
    assert "configured!" in capsys.readouterr().out


def test_start_command_renders_ipv6_base_url(monkeypatch):
    """IPv6 bind hosts use a valid bracketed URL for readiness checks."""
    args = _make_args(host="::1")
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: True)
    reused = {}
    monkeypatch.setattr(
        run_cli,
        "_reuse_or_refuse",
        lambda base_url, *rest: reused.update(base_url=base_url) or 0,
    )

    assert run_cli.start_command(args) == 0
    assert reused["base_url"] == "http://[::1]:8000"


@pytest.mark.parametrize(
    ("bind_host", "expected"),
    [("0.0.0.0", "http://127.0.0.1:8000"), ("::", "http://[::1]:8000")],
)
def test_start_command_probes_wildcard_bind_via_loopback(
    monkeypatch, bind_host, expected
):
    """Wildcard bind addresses are never emitted as client destinations."""
    args = _make_args(host=bind_host)
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda h, p: True)
    seen = {}
    monkeypatch.setattr(
        run_cli,
        "_reuse_or_refuse",
        lambda base_url, *rest: seen.update(base_url=base_url) or 0,
    )
    assert run_cli.start_command(args) == 0
    assert seen["base_url"] == expected


def test_attach_generic_writer_cannot(monkeypatch, capsys):
    """Generic writer returning 'Cannot...' -> failure path, instructions."""
    args = _make_args()
    prof = _FakeProfile(name="hermes", recommended_models=["qwen3.5-9b-4bit"])
    prof.config = _FakeCfg(template=None)

    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(
        ad, "setup_agent_config", lambda *a, **k: "Cannot find the agent binary"
    )
    monkeypatch.setattr(
        ad, "get_setup_instructions", lambda *a, **k: "  manual instructions"
    )
    rc = run_cli._attach_and_configure("http://b", "m", prof, args)
    assert rc == 1
    out = capsys.readouterr().out
    assert "setup failed" in out
    assert "manual instructions" in out


@pytest.mark.parametrize("dry_run", [True, False])
def test_attach_generic_writer_exception_falls_back(monkeypatch, capsys, dry_run):
    """Generic config renderer/writer failures leave the server usable."""
    args = _make_args(yes=True, dry_run=dry_run)
    prof = _FakeProfile(name="hermes", recommended_models=["qwen3.5-9b-4bit"])
    prof.config = _FakeCfg(template=None)

    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(
        ad,
        "setup_agent_config",
        lambda *a, **k: (_ for _ in ()).throw(OSError("config unavailable")),
    )
    monkeypatch.setattr(
        ad, "get_setup_instructions", lambda *a, **k: "  manual instructions"
    )

    assert run_cli._attach_and_configure("http://b", "m", prof, args) == 1
    out = capsys.readouterr().out
    assert "setup failed: config unavailable" in out
    assert "manual instructions" in out


def test_attach_env_profile_prints_exports_without_write_consent(monkeypatch, capsys):
    """Environment-only profiles give instructions and never claim a write."""
    args = _make_args()
    prof = _FakeProfile(name="langchain")
    prof.config = _FakeCfg(config_type="env")
    from vllm_mlx.agents import adapter as ad

    calls = []
    monkeypatch.setattr(
        ad,
        "setup_agent_config",
        lambda *a, **kwargs: (
            calls.append(kwargs.get("dry_run"))
            or "Run these commands in your shell:\n  export OPENAI_API_BASE=http://b/v1"
        ),
    )
    monkeypatch.setattr(
        run_cli,
        "_confirm_config_write",
        lambda: pytest.fail("env instructions do not write configuration"),
    )

    assert run_cli._attach_and_configure("http://b", "m", prof, args) == 0
    out = capsys.readouterr().out
    assert calls == [True]
    assert "uses shell environment variables" in out
    assert "configured!" not in out


def test_attach_env_profile_exception_falls_back(monkeypatch, capsys):
    """Environment-profile rendering failures retain manual instructions."""
    args = _make_args()
    prof = _FakeProfile(name="langchain")
    prof.config = _FakeCfg(config_type="env")
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(
        ad,
        "setup_agent_config",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad template")),
    )
    monkeypatch.setattr(
        ad, "get_setup_instructions", lambda *a, **k: "  manual instructions"
    )

    assert run_cli._attach_and_configure("http://b", "m", prof, args) == 1
    out = capsys.readouterr().out
    assert "setup failed: bad template" in out
    assert "manual instructions" in out


def test_cached_context_window_reads_text_config(monkeypatch):
    """Dry-run can preview the eventual context value without network access."""
    import vllm_mlx.model_metadata as metadata_mod

    requested = []
    monkeypatch.setattr(run_cli, "_hf_id", lambda alias: "org/resolved-model")
    monkeypatch.setattr(
        metadata_mod,
        "read_model_metadata",
        lambda model: (
            requested.append(model)
            or types.SimpleNamespace(
                config={"text_config": {"max_position_embeddings": 262_144}}
            )
        ),
    )
    assert run_cli._cached_context_window("cached-model") == 262_144
    assert requested == ["org/resolved-model"]


def test_cached_context_window_metadata_error_is_unknown(monkeypatch):
    """Broken cache metadata defers setup preview rather than crashing start."""
    import vllm_mlx.model_metadata as metadata_mod

    monkeypatch.setattr(
        metadata_mod,
        "read_model_metadata",
        lambda model: (_ for _ in ()).throw(OSError("corrupt cache")),
    )
    assert run_cli._cached_context_window("cached-model") is None


def test_print_instructions_first_class(monkeypatch, capsys):
    """_print_instructions renders first-class instructions."""
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(ad, "get_setup_instructions", lambda *a, **k: "  render me")
    run_cli._print_instructions(_FakeProfile(), "http://b", "m")
    assert "render me" in capsys.readouterr().out


def test_print_instructions_exception(monkeypatch, capsys):
    """_print_instructions tolerates a rendering failure."""
    from vllm_mlx.agents import adapter as ad

    monkeypatch.setattr(
        ad,
        "get_setup_instructions",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")),
    )
    run_cli._print_instructions(_FakeProfile(), "http://b", "m")
    assert "Could not render" in capsys.readouterr().out


def test_print_next_steps(monkeypatch, capsys):
    run_cli._print_next_steps("hermes", "http://b", "m")
    assert "hermes" in capsys.readouterr().out


def test_print_dry_run_generic(monkeypatch, capsys):
    run_cli._print_dry_run(None, "m", _make_args())
    out = capsys.readouterr().out
    assert "generic OpenAI-compatible" in out
    # physical_ram_gb() is a macOS-only probe (sysctl hw.memsize) that returns
    # 0.0 on non-darwin. The `RAM:` prints are gated on `if ram_gb:` (a real
    # runtime line that must stay covered on the Linux no-MLX matrix, which
    # runs this hermetic file). Patch the recommendation module's attribute so
    # both branches are exercised deterministically on every host.
    monkeypatch.setattr(rec, "physical_ram_gb", lambda: 256.0)
    run_cli._print_dry_run("codex", "m", _make_args())
    out = capsys.readouterr().out
    assert "codex" in out and "RAM:" in out and "256.0 GB" in out


def test_print_dry_run_ram_unavailable_skips_line(monkeypatch, capsys):
    """Falsy RAM probe → the RAM prints are skipped (no negative output)."""
    monkeypatch.setattr(rec, "physical_ram_gb", lambda: 0.0)
    run_cli._print_dry_run("codex", "m", _make_args())
    out = capsys.readouterr().out
    assert "RAM:" not in out


def test_print_unknown_agent(monkeypatch, capsys):
    import vllm_mlx.agents as agents_mod

    monkeypatch.setattr(
        agents_mod, "list_profiles", lambda: [_FakeProfile(name="codex")]
    )
    run_cli._print_unknown_agent("nope")
    out = capsys.readouterr().out
    assert "Unknown agent: nope" in out
    assert "codex" in out


def test_print_nothing_cached_fit_variants(monkeypatch, capsys):
    """_print_nothing_cached with & without a fitting candidate."""
    monkeypatch.setattr(rec, "is_recommended_alias", lambda alias, ram: False)
    run_cli._print_nothing_cached(["qwen3.5-9b-4bit"], 64.0)
    out = capsys.readouterr().out
    assert "✗" in out
    run_cli._print_nothing_cached(["qwen3.5-9b-4bit"], 0.0)
    assert "Drop --no-download" in capsys.readouterr().out


def test_print_nothing_fits(monkeypatch, capsys):
    run_cli._print_nothing_fits(["qwen3.5-9b-4bit"], 12.0)
    assert "fit" in capsys.readouterr().out


def test_wait_child_returns_exit_code(monkeypatch):
    """A positive child exit code is returned verbatim."""
    import time as real_time

    class FakeChild:
        def __init__(self):
            self.returncode = None
            self._polls = 0

        def poll(self):
            self._polls += 1
            if self._polls >= 2:
                self.returncode = 5
            return self.returncode

    monkeypatch.setattr(real_time, "sleep", lambda s: None)
    assert run_cli._wait_child(FakeChild()) == 5


def test_wait_child_none_code_returns_1(monkeypatch):
    """A child whose poll() ends with returncode None -> exit 1."""
    import time as real_time

    class FakeChild:
        returncode = None
        _polls = 0

        def poll(self):
            self._polls += 1
            # Break the loop on the second call (return non-None) while keeping
            # returncode None -> exercises the ``code is None`` -> 1 path.
            return -1 if self._polls >= 2 else None

    monkeypatch.setattr(real_time, "sleep", lambda s: None)
    assert run_cli._wait_child(FakeChild()) == 1


def test_wait_child_maps_signal_death_to_posix(monkeypatch):
    """A signal-death returncode (-SIGTERM) maps to 128 + signum."""
    import time as real_time

    class FakeChild:
        def __init__(self):
            self.returncode = None
            self._polls = 0

        def poll(self):
            self._polls += 1
            if self._polls >= 2:
                self.returncode = -15  # killed by SIGTERM
            return self.returncode

    monkeypatch.setattr(real_time, "sleep", lambda s: None)
    assert run_cli._wait_child(FakeChild()) == 128 + 15  # 143


def test_terminate_child_is_quiet(monkeypatch):
    """_terminate_child swallows ProcessLookupError (child already gone)."""
    import subprocess as real_subprocess

    class FakeChild:
        def poll(self):
            return None

        def terminate(self):
            raise ProcessLookupError()

    monkeypatch.setattr(real_subprocess, "Popen", lambda *a, **k: FakeChild())
    # No exception propagates.
    run_cli._terminate_child(FakeChild())


def test_terminate_child_escalates_after_grace_period():
    """A SIGTERM-ignoring child is killed and reaped within bounded waits."""
    calls = []

    class FakeChild:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")
            self.returncode = -signal.SIGKILL

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if self.returncode is None:
                raise run_cli.subprocess.TimeoutExpired("serve", timeout)
            return self.returncode

    run_cli._terminate_child(FakeChild(), grace_s=0.01)
    assert calls == ["terminate", ("wait", 0.01), "kill", ("wait", 0.01)]


def test_foreground_child_relays_signals(monkeypatch):
    """_foreground_child installs a relay forwarding SIGINT/SIGTERM to the child
    and restores prior handlers on exit (HIGH 2: relay active during ready-wait)."""
    sent = []

    class FakeChild:
        def poll(self):
            return None

        def send_signal(self, signum):
            sent.append(signum)
            if signum == signal.SIGINT:
                raise ProcessLookupError()  # already gone -> swallowed

    reaped = []
    monkeypatch.setattr(
        run_cli, "_reap_forwarded_child", lambda proc: reaped.append(proc)
    )
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    for signum in (signal.SIGINT, signal.SIGTERM):
        with (
            pytest.raises(run_cli._ForwardedSignalError) as exc,
            run_cli._foreground_child(FakeChild()),
        ):
            # Handler forwards to the child, then forces the parent out of a
            # potentially blocked prompt/wait so cleanup can complete.
            signal.getsignal(signum)(signum, None)
        assert exc.value.signum == signum
    assert signal.SIGINT in sent
    assert signal.SIGTERM in sent
    assert len(reaped) == 2
    # Restored.
    assert signal.getsignal(signal.SIGINT) == prev_int
    assert signal.getsignal(signal.SIGTERM) == prev_term


def test_reap_forwarded_child_waits_before_escalating():
    """A relayed signal gets a grace period before any second signal."""
    calls = []

    class FakeChild:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            calls.append(("wait", timeout))
            self.returncode = -signal.SIGINT
            return self.returncode

    run_cli._reap_forwarded_child(FakeChild(), grace_s=0.01)
    assert calls == [("wait", 0.01)]


# Coverage contracts for defensive orchestration branches.  These paths are
# deliberately hermetic: they exercise ownership and exit-code semantics
# without starting a server, touching an agent config, or reading the network.


def test_start_dry_run_without_profile_never_attaches(monkeypatch):
    args = _make_args(profile=None, dry_run=True)
    _patch_profile_lookup(monkeypatch, None)
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda *a: False)
    monkeypatch.setattr(
        run_cli,
        "_attach_and_configure",
        lambda *a, **k: pytest.fail("generic dry-run must not configure an agent"),
    )
    assert run_cli.start_command(args) == 0


def test_start_maps_forwarded_signal_to_shell_exit(monkeypatch):
    args = _make_args()
    _patch_profile_lookup(monkeypatch, _FakeProfile())
    monkeypatch.setattr(run_cli, "_select_model", lambda **kw: "m")
    monkeypatch.setattr(run_cli, "_port_is_busy", lambda *a: False)
    monkeypatch.setattr(run_cli, "_confirm_download", lambda *a, **k: True)
    monkeypatch.setattr(run_cli, "_spawn_foreground_serve", lambda *a: object())

    class ForwardOnEnter:
        def __init__(self, proc):
            pass

        def __enter__(self):
            raise run_cli._ForwardedSignalError(signal.SIGTERM)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(run_cli, "_foreground_child", ForwardOnEnter)
    assert run_cli.start_command(args) == 128 + signal.SIGTERM


def test_fits_host_uses_alias_minimum(monkeypatch):
    import vllm_mlx.model_aliases as aliases

    monkeypatch.setattr(rec, "recommendation_footprint_gb", lambda alias: None)
    monkeypatch.setattr(
        aliases,
        "resolve_profile",
        lambda alias: types.SimpleNamespace(min_memory_gb=24),
    )
    assert run_cli._fits_host("agent-model", 32) is True


def test_foreground_child_cleans_up_on_unrelated_exception(monkeypatch):
    class FakeChild:
        def poll(self):
            return None

    child = FakeChild()
    terminated = []
    monkeypatch.setattr(run_cli, "_terminate_child", terminated.append)
    with pytest.raises(RuntimeError), run_cli._foreground_child(child):
        raise RuntimeError("prompt failed")
    assert terminated == [child]


class _CleanupChild:
    def __init__(self, *, exited=False, terminate_error=None, kill_error=None):
        self.returncode = 0 if exited else None
        self.terminate_error = terminate_error
        self.kill_error = kill_error
        self.wait_calls = 0
        self.calls = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        if self.terminate_error:
            raise self.terminate_error()

    def kill(self):
        self.calls.append("kill")
        if self.kill_error:
            raise self.kill_error()

    def wait(self, timeout):
        self.calls.append(("wait", timeout))
        self.wait_calls += 1
        if self.wait_calls == 1 and self.returncode is None:
            raise run_cli.subprocess.TimeoutExpired("serve", timeout)
        if self.returncode is None:
            raise run_cli.subprocess.TimeoutExpired("serve", timeout)
        return self.returncode


def test_terminate_child_already_exited_and_graceful_paths():
    exited = _CleanupChild(exited=True)
    run_cli._terminate_child(exited)
    assert exited.calls == []

    child = _CleanupChild()
    child.wait = lambda timeout: child.calls.append(("wait", timeout)) or 0
    run_cli._terminate_child(child)
    assert child.calls == ["terminate", ("wait", 5.0)]


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
def test_terminate_child_ignores_kill_races(error):
    child = _CleanupChild(kill_error=error)
    run_cli._terminate_child(child, grace_s=0.01)
    assert child.calls == ["terminate", ("wait", 0.01), "kill"]


def test_terminate_child_bounds_post_kill_wait():
    child = _CleanupChild()
    run_cli._terminate_child(child, grace_s=0.01)
    assert child.calls == [
        "terminate",
        ("wait", 0.01),
        "kill",
        ("wait", 0.01),
    ]


def test_reap_forwarded_child_already_exited_and_escalates(monkeypatch):
    run_cli._reap_forwarded_child(_CleanupChild(exited=True))

    child = _CleanupChild()
    escalated = []
    monkeypatch.setattr(
        run_cli,
        "_terminate_child",
        lambda proc, grace_s: escalated.append((proc, grace_s)),
    )
    run_cli._reap_forwarded_child(child, grace_s=0.01)
    assert escalated == [(child, 0.01)]


def test_attach_deepseek_dry_run_uses_cached_reasoning_profile(monkeypatch):
    import vllm_mlx.agents.setup as setup_mod
    import vllm_mlx.model_aliases as aliases

    args = _make_args(dry_run=True)
    prof = _FakeProfile(name="deepseek-harness")
    prof.config = _FakeCfg()
    monkeypatch.setattr(
        aliases,
        "resolve_profile",
        lambda model: types.SimpleNamespace(reasoning_parser="deepseek_r1"),
    )
    planned = {}
    monkeypatch.setattr(
        setup_mod,
        "build_setup_plan",
        lambda *a, **kw: planned.update(kw) or _SetupPlanFake(changed=False),
    )
    assert run_cli._attach_and_configure("http://b", "m", prof, args) == 0
    assert planned["supports_reasoning"] is True


def test_attach_first_class_dry_run_does_not_write(monkeypatch, capsys):
    import vllm_mlx.agents.setup as setup_mod

    args = _make_args(dry_run=True)
    prof = _first_class_profile()
    prof.config = _FakeCfg()
    monkeypatch.setattr(
        setup_mod, "build_setup_plan", lambda *a, **k: _SetupPlanFake(changed=True)
    )
    monkeypatch.setattr(
        setup_mod,
        "apply_setup_plan",
        lambda plan: pytest.fail("dry-run must not apply the setup plan"),
    )
    assert run_cli._attach_and_configure("http://b", "m", prof, args) == 0
    assert "Dry run only" in capsys.readouterr().out


def test_attach_generic_dry_run_and_cancel_paths(monkeypatch, capsys):
    from vllm_mlx.agents import adapter as ad

    prof = _FakeProfile(name="hermes", config=_FakeCfg())
    calls = []
    monkeypatch.setattr(
        ad,
        "setup_agent_config",
        lambda *a, **k: calls.append(k["dry_run"]) or "preview",
    )
    assert (
        run_cli._attach_and_configure("http://b", "m", prof, _make_args(dry_run=True))
        == 0
    )
    assert calls == [True]

    monkeypatch.setattr(run_cli, "_confirm_config_write", lambda: False)
    assert run_cli._attach_and_configure("http://b", "m", prof, _make_args()) == 0
    assert calls == [True, True]
    assert "Setup cancelled" in capsys.readouterr().out


@pytest.mark.parametrize(
    "result", [RuntimeError("write failed"), "Cannot write config"]
)
def test_attach_generic_write_failure_paths(monkeypatch, capsys, result):
    from vllm_mlx.agents import adapter as ad

    prof = _FakeProfile(name="hermes", config=_FakeCfg())
    calls = 0

    def setup(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "preview"
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(ad, "setup_agent_config", setup)
    assert (
        run_cli._attach_and_configure("http://b", "m", prof, _make_args(yes=True)) == 1
    )
    assert "setup failed" in capsys.readouterr().out


def test_confirm_config_write_noninteractive_yes_and_interrupt(monkeypatch):
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: False))
    assert run_cli._confirm_config_write() is False

    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert run_cli._confirm_config_write() is True

    monkeypatch.setattr(
        "builtins.input", lambda prompt: (_ for _ in ()).throw(EOFError())
    )
    assert run_cli._confirm_config_write() is False


def test_cached_context_window_rejects_invalid_candidates(monkeypatch):
    import vllm_mlx.model_metadata as metadata_mod

    monkeypatch.setattr(
        metadata_mod,
        "read_model_metadata",
        lambda model: types.SimpleNamespace(
            config={"max_position_embeddings": True, "text_config": {}}
        ),
    )
    assert run_cli._cached_context_window("m") is None
