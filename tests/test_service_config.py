from __future__ import annotations

import io
import json
import os
import stat
import types
from pathlib import Path

import pytest

from vllm_mlx.cli import build_parser
from vllm_mlx.headless_service.config import (
    SCHEMA_VERSION,
    ServiceConfig,
    ServiceConfigError,
    assert_private_file,
    atomic_write,
    config_bytes,
    config_digest,
    load_config,
    pending_config_path,
)
from vllm_mlx.headless_service.plist import build_plist_dict


def _config(**updates) -> ServiceConfig:
    values = {
        "schema_version": SCHEMA_VERSION,
        "label": "com.rapidmlx.server",
        "service_user": "serveuser",
        "executable": "/Users/serveuser/.local/bin/rapid-mlx",
        "model": "qwen3.5-4b-4bit",
        "host": "127.0.0.1",
        "port": 8000,
        "serve_args": ("--max-num-seqs", "4"),
    }
    values.update(updates)
    return ServiceConfig(**values).validated()


def test_service_config_round_trip_and_digest(tmp_path):
    path = tmp_path / "service.json"
    config = _config()
    atomic_write(path, (json.dumps(config.to_dict()) + "\n").encode())
    loaded = load_config(path)
    assert loaded == config
    assert config_digest(loaded) == config_digest(config)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": 999}, "unsupported"),
        ({"port": 0}, "port"),
        ({"executable": "rapid-mlx"}, "absolute"),
        ({"serve_args": ("--api-key=leak",)}, "API key"),
        ({"log_retention_days": 0}, "at least 1"),
    ],
)
def test_service_config_rejects_unsafe_values(updates, message):
    with pytest.raises(ServiceConfigError, match=message):
        _config(**updates)


def test_service_config_rejects_unknown_fields():
    raw = _config().to_dict()
    raw["typo_port"] = 9000
    with pytest.raises(ServiceConfigError, match="unknown"):
        ServiceConfig.from_dict(raw)


def test_service_config_rejects_string_serve_args():
    raw = _config().to_dict()
    raw["serve_args"] = "--max-num-seqs 4"
    with pytest.raises(ServiceConfigError, match="array of strings"):
        ServiceConfig.from_dict(raw)


def test_atomic_write_replaces_whole_file(tmp_path):
    path = tmp_path / "config.json"
    atomic_write(path, b"old\n")
    atomic_write(path, b"new\n")
    assert path.read_bytes() == b"new\n"
    assert not list(tmp_path.glob(".config.json.*"))


def test_private_file_guard(tmp_path):
    path = tmp_path / "credential"
    path.write_text("secret\n")
    path.chmod(0o600)
    assert_private_file(path, expected_uid=os.getuid())
    path.chmod(0o640)
    with pytest.raises(ServiceConfigError, match="group or others"):
        assert_private_file(path)


def test_config_backed_plist_contains_no_model_flags_or_secret(tmp_path):
    config_path = tmp_path / "service.json"
    plist = build_plist_dict(
        label="com.rapidmlx.server",
        user="serveuser",
        executable="/Users/serveuser/.local/bin/rapid-mlx",
        model="qwen3.5-4b-4bit",
        home=Path("/Users/serveuser"),
        log_dir=Path("/Users/serveuser/Library/Logs/Rapid-MLX"),
        config_path=config_path,
    )
    assert plist["ProgramArguments"] == [
        "/Users/serveuser/.local/bin/rapid-mlx",
        "service",
        "run",
        "--config",
        str(config_path),
    ]
    serialized = repr(plist)
    assert "qwen3.5-4b-4bit" not in serialized
    assert "RAPID_MLX_API_KEY" not in serialized


def test_configure_cli_parses_candidate_fields():
    args = build_parser().parse_args(
        [
            "service",
            "configure",
            "--model",
            "mlx-community/new-model",
            "--port",
            "9000",
            "--",
            "--max-num-seqs",
            "8",
        ]
    )
    assert args.model == "mlx-community/new-model"
    assert args.port == 9000
    assert args.serve_args == ["--", "--max-num-seqs", "8"]


def test_runtime_loads_private_credential_and_supervises(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import runtime

    credential = tmp_path / "api-key"
    credential.write_text("sk-test\n")
    credential.chmod(0o600)
    config = _config(
        service_user="runner",
        executable="/opt/rapid-mlx",
        credential_file=str(credential),
    )
    config_file = tmp_path / "service.json"
    atomic_write(config_file, (json.dumps(config.to_dict()) + "\n").encode())
    monkeypatch.setattr(
        runtime.pwd,
        "getpwnam",
        lambda _user: types.SimpleNamespace(pw_uid=os.getuid()),
    )
    observed = {}

    def fake_supervise(argv, env, runtime_config):
        observed.update(argv=argv, env=env, config=runtime_config)
        return 23

    monkeypatch.setattr(runtime, "_supervise", fake_supervise)
    assert runtime.run_command(types.SimpleNamespace(config=str(config_file))) == 23
    assert observed["argv"][-2:] == ["--max-num-seqs", "4"]
    assert observed["env"]["RAPID_MLX_API_KEY"] == "sk-test"


def test_rotating_log_bounds_backups(tmp_path):
    from vllm_mlx.headless_service.rotating_logs import RotatingLog

    path = tmp_path / "server.stdout.log"
    sink = RotatingLog(path, max_bytes=8, backup_count=2, retention_days=7)
    for payload in (b"12345678", b"abcdefgh", b"ABCDEFGH", b"87654321"):
        sink.write(payload)
    sink.close()
    assert path.read_bytes() == b"87654321"
    backups = list(tmp_path.glob("server.stdout.log.*"))
    assert len(backups) <= 2
    assert all(item.stat().st_size == 8 for item in backups)


def test_runtime_refuses_world_readable_credential(monkeypatch, tmp_path, capsys):
    from vllm_mlx.headless_service import runtime

    credential = tmp_path / "api-key"
    credential.write_text("sk-test\n")
    credential.chmod(0o644)
    config = _config(service_user="runner", credential_file=str(credential))
    config_file = tmp_path / "service.json"
    atomic_write(config_file, (json.dumps(config.to_dict()) + "\n").encode())
    monkeypatch.setattr(
        runtime.pwd,
        "getpwnam",
        lambda _user: types.SimpleNamespace(pw_uid=os.getuid()),
    )
    assert runtime.run_command(types.SimpleNamespace(config=str(config_file))) == 78
    assert "group or others" in capsys.readouterr().err


def _installed_config(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import config as config_module
    from vllm_mlx.headless_service import configure

    home = tmp_path / "home"
    monkeypatch.setattr(
        config_module, "SERVICE_CONFIG_ROOT", tmp_path / "system-config"
    )
    current_path = config_module.config_path(home)
    current = _config(service_user="runner")
    atomic_write(current_path, config_bytes(current))
    monkeypatch.setattr(
        configure,
        "installed_identity",
        lambda _label: ("runner", home, current_path),
    )
    monkeypatch.setattr(
        configure,
        "_account",
        lambda _user: types.SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid()),
    )
    return configure, home, current_path, current


def test_configure_stages_without_touching_active(monkeypatch, tmp_path):
    configure, home, current_path, current = _installed_config(monkeypatch, tmp_path)
    monkeypatch.setattr(configure, "is_root", lambda: True)
    args = types.SimpleNamespace(
        label=None,
        dry_run=False,
        model="new-model",
        host=None,
        port=9000,
        log_retention_days=None,
        log_max_mb=None,
        log_backup_count=None,
        clear_serve_args=False,
        serve_args=["--", "--max-num-seqs", "8"],
    )
    assert configure.configure_command(args) == 0
    assert load_config(current_path) == current
    candidate = load_config(pending_config_path(home))
    assert candidate.model == "new-model"
    assert candidate.port == 9000
    assert candidate.serve_args == ("--max-num-seqs", "8")


def test_apply_promotes_healthy_candidate(monkeypatch, tmp_path):
    configure, home, current_path, _ = _installed_config(monkeypatch, tmp_path)
    pending = pending_config_path(home)
    candidate = _config(service_user="runner", model="new-model", port=9000)
    atomic_write(pending, config_bytes(candidate))
    monkeypatch.setattr(configure, "is_root", lambda: True)
    monkeypatch.setattr(configure, "_port_busy", lambda _host, _port: False)
    monkeypatch.setattr(configure, "_bootout", lambda _label: None)
    monkeypatch.setattr(
        configure,
        "_bootstrap",
        lambda _label: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(configure, "_wait_ready", lambda _host, _port: True)
    assert (
        configure.apply_command(types.SimpleNamespace(label=None, dry_run=False)) == 0
    )
    assert load_config(current_path) == candidate
    assert not pending.exists()


def test_apply_restores_previous_config_when_candidate_is_unhealthy(
    monkeypatch, tmp_path, capsys
):
    configure, home, current_path, current = _installed_config(monkeypatch, tmp_path)
    candidate = _config(service_user="runner", model="bad-model", port=9000)
    atomic_write(pending_config_path(home), config_bytes(candidate))
    monkeypatch.setattr(configure, "is_root", lambda: True)
    monkeypatch.setattr(configure, "_port_busy", lambda _host, _port: False)
    monkeypatch.setattr(configure, "_bootout", lambda _label: None)
    boot_count = 0

    def bootstrap(_label):
        nonlocal boot_count
        boot_count += 1
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(configure, "_bootstrap", bootstrap)
    monkeypatch.setattr(
        configure,
        "_wait_ready",
        lambda _host, port: port == current.port,
    )
    assert (
        configure.apply_command(types.SimpleNamespace(label=None, dry_run=False)) == 2
    )
    assert load_config(current_path) == current
    assert boot_count == 2
    assert "previous service restored" in capsys.readouterr().err


def test_credential_set_never_prints_secret(monkeypatch, tmp_path, capsys):
    configure, home, _, _ = _installed_config(monkeypatch, tmp_path)
    monkeypatch.setattr(configure, "is_root", lambda: True)
    monkeypatch.setattr(configure.sys, "stdin", io.StringIO("sk-sensitive\n"))
    args = types.SimpleNamespace(
        label=None,
        credential_command="set",
    )
    assert configure.credential_command(args) == 0
    captured = capsys.readouterr()
    assert "sk-sensitive" not in captured.out + captured.err
    path = home / ".rapid-mlx-secrets" / "com.rapidmlx.server.credential"
    assert path.read_text() == "sk-sensitive\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("version", "extras", "expected"),
    [
        (None, None, "rapid-mlx"),
        ("0.13.5", "vision,embeddings", "rapid-mlx[vision,embeddings]==0.13.5"),
    ],
)
def test_upgrade_target_is_constrained(version, extras, expected):
    from vllm_mlx.headless_service.upgrade import _target

    assert _target(version, extras) == expected
    with pytest.raises(ServiceConfigError):
        _target("1.0 --index-url evil", extras)


def test_upgrade_success_snapshots_before_mutation(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import upgrade

    configure, home, _, _ = _installed_config(monkeypatch, tmp_path)
    python = home / ".rapid-mlx" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o700)
    monkeypatch.setattr(upgrade, "is_root", lambda: True)
    monkeypatch.setattr(upgrade, "_account", configure._account)
    events = []

    def as_user(_user, argv, *, timeout):
        del timeout
        events.append(tuple(argv))
        if argv[-2:] == ["pip", "freeze"]:
            return types.SimpleNamespace(
                returncode=0, stdout="rapid-mlx==0.13.4\n", stderr=""
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(upgrade, "_as_user", as_user)
    monkeypatch.setattr(upgrade, "_bootout", lambda _label: events.append(("bootout",)))
    monkeypatch.setattr(
        upgrade,
        "_bootstrap",
        lambda _label: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(upgrade, "_wait_ready", lambda _host, _port: True)
    args = types.SimpleNamespace(
        label=None,
        dry_run=False,
        version="0.13.5",
        extras="vision",
        pre=False,
    )
    assert upgrade.upgrade_command(args) == 0
    assert events[0][-2:] == ("pip", "freeze")
    assert events[1] == ("bootout",)
    assert any("rapid-mlx[vision]==0.13.5" in event for event in events)
    snapshot = (
        tmp_path / "system-config" / "com.rapidmlx.server.previous-requirements.txt"
    )
    assert snapshot.read_text() == "rapid-mlx==0.13.4\n"


def test_upgrade_rolls_back_after_readiness_failure(monkeypatch, tmp_path, capsys):
    from vllm_mlx.headless_service import upgrade

    configure, home, _, _ = _installed_config(monkeypatch, tmp_path)
    python = home / ".rapid-mlx" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o700)
    monkeypatch.setattr(upgrade, "is_root", lambda: True)
    monkeypatch.setattr(upgrade, "_account", configure._account)

    def as_user(_user, argv, *, timeout):
        del timeout
        stdout = "rapid-mlx==0.13.4\n" if argv[-2:] == ["pip", "freeze"] else ""
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(upgrade, "_as_user", as_user)
    monkeypatch.setattr(upgrade, "_bootout", lambda _label: None)
    monkeypatch.setattr(
        upgrade,
        "_bootstrap",
        lambda _label: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(upgrade, "_wait_ready", lambda _host, _port: False)
    monkeypatch.setattr(upgrade, "_restore", lambda **_kwargs: True)
    args = types.SimpleNamespace(
        label=None,
        dry_run=False,
        version=None,
        extras=None,
        pre=False,
    )
    assert upgrade.upgrade_command(args) == 2
    assert "previous environment restored" in capsys.readouterr().err
