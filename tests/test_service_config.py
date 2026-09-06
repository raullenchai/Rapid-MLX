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
    private_file_present,
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


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"service_user": "root"}, "non-root"),
        ({"model": ""}, "model"),
        ({"host": "bad host"}, "host"),
        ({"log_max_mb": 0}, "at least 1"),
        ({"log_backup_count": 0}, "at least 1"),
        ({"serve_args": ("bad\0arg",)}, "NUL"),
        ({"credential_file": "relative"}, "absolute"),
    ],
)
def test_service_config_rejects_remaining_invalid_values(updates, message):
    with pytest.raises(ServiceConfigError, match=message):
        _config(**updates)


@pytest.mark.parametrize("payload", ["[]", "{"])
def test_load_config_rejects_invalid_json_shapes(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(ServiceConfigError):
        load_config(path)


def test_atomic_write_owner_and_cleanup_paths(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import config as config_module

    path = tmp_path / "owned"
    ownership = []
    monkeypatch.setattr(
        config_module.os,
        "fchown",
        lambda fd, uid, gid: ownership.append((fd, uid, gid)),
    )
    atomic_write(path, b"ok", uid=123, gid=None)
    assert ownership[0][1:] == (123, -1)

    real_replace = config_module.os.replace
    monkeypatch.setattr(
        config_module.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        atomic_write(tmp_path / "failed", b"no")
    assert not list(tmp_path.glob(".failed.*"))
    monkeypatch.setattr(config_module.os, "replace", real_replace)


def test_definition_and_secret_directory_helpers(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import config as config_module

    monkeypatch.setattr(config_module, "SERVICE_CONFIG_ROOT", tmp_path / "definitions")
    monkeypatch.setattr(config_module.os, "geteuid", lambda: 0)
    chowns = []
    monkeypatch.setattr(
        config_module.os,
        "chown",
        lambda path, uid, gid: chowns.append((path, uid, gid)),
    )
    target = config_module.ensure_config_dir(tmp_path, uid=501, gid=20)
    assert target.is_dir() and stat.S_IMODE(target.stat().st_mode) == 0o755
    secret_dir = config_module.ensure_credential_dir(tmp_path, uid=501, gid=20)
    assert secret_dir.is_dir() and stat.S_IMODE(secret_dir.stat().st_mode) == 0o700
    assert chowns == [(target, 0, 0), (secret_dir, 501, 20)]

    calls = []
    monkeypatch.setattr(
        config_module, "atomic_write", lambda *a, **k: calls.append((a, k))
    )
    config_module.atomic_write_definition(tmp_path / "x", b"data")
    assert calls[0][1]["uid"] == calls[0][1]["gid"] == 0


def test_private_file_guards_and_presence(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ServiceConfigError, match="cannot inspect"):
        assert_private_file(missing)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ServiceConfigError, match="regular"):
        assert_private_file(directory)

    owned = tmp_path / "owned"
    owned.write_text("secret")
    owned.chmod(0o600)
    with pytest.raises(ServiceConfigError, match="owned by"):
        assert_private_file(owned, expected_uid=os.getuid() + 1)
    assert private_file_present(owned) is True
    assert private_file_present(missing) is False

    monkeypatch.setattr(
        Path, "stat", lambda _self: (_ for _ in ()).throw(PermissionError())
    )
    assert private_file_present(owned) is None


def _configure_args(**updates):
    values = dict(
        label=None,
        dry_run=False,
        model=None,
        host=None,
        port=None,
        log_retention_days=None,
        log_max_mb=None,
        log_backup_count=None,
        clear_serve_args=False,
        serve_args=None,
    )
    values.update(updates)
    return types.SimpleNamespace(**values)


def test_configure_dry_run_clear_and_error_paths(monkeypatch, tmp_path, capsys):
    from vllm_mlx.headless_service import configure

    module, _, _, _ = _installed_config(monkeypatch, tmp_path)
    assert (
        module.configure_command(_configure_args(dry_run=True, clear_serve_args=True))
        == 0
    )
    assert '"serve_args": []' in capsys.readouterr().out

    monkeypatch.setattr(module, "is_root", lambda: False)
    assert module.configure_command(_configure_args()) == 1
    monkeypatch.setattr(module, "is_root", lambda: True)
    monkeypatch.setattr(
        module,
        "atomic_write_definition",
        lambda *_a: (_ for _ in ()).throw(OSError("disk")),
    )
    assert module.configure_command(_configure_args()) == 2

    monkeypatch.setattr(configure, "installed_identity", lambda _label: None)
    assert configure.configure_command(_configure_args()) == 1


def test_configure_launchctl_wrappers(monkeypatch):
    from vllm_mlx.headless_service import configure

    calls = []
    monkeypatch.setattr(
        configure.subprocess,
        "run",
        lambda argv, **kwargs: (
            calls.append((argv, kwargs)) or types.SimpleNamespace(returncode=0)
        ),
    )
    assert configure._bootstrap("com.rapidmlx.server").returncode == 0
    configure._bootout("com.rapidmlx.server")
    assert calls[0][0][1] == "bootstrap" and calls[1][0][1] == "bootout"


def test_apply_validation_dry_run_and_nonroot(monkeypatch, tmp_path, capsys):
    configure, home, _, current = _installed_config(monkeypatch, tmp_path)
    pending = pending_config_path(home)

    atomic_write(pending, config_bytes(current.updated(service_user="other")))
    assert configure.apply_command(_configure_args()) == 1
    atomic_write(pending, config_bytes(current.updated(executable="/other/rapid-mlx")))
    assert configure.apply_command(_configure_args()) == 1
    atomic_write(pending, config_bytes(current.updated(port=9000)))
    monkeypatch.setattr(configure, "_port_busy", lambda *_a: True)
    assert configure.apply_command(_configure_args()) == 1
    monkeypatch.setattr(configure, "_port_busy", lambda *_a: False)
    assert configure.apply_command(_configure_args(dry_run=True)) == 0
    assert "restore previous" in capsys.readouterr().out
    monkeypatch.setattr(configure, "is_root", lambda: False)
    assert configure.apply_command(_configure_args()) == 1


def test_apply_bootstrap_and_rollback_failure_paths(monkeypatch, tmp_path, capsys):
    configure, home, current_path, current = _installed_config(monkeypatch, tmp_path)
    atomic_write(pending_config_path(home), config_bytes(current.updated(model="next")))
    monkeypatch.setattr(configure, "is_root", lambda: True)
    monkeypatch.setattr(configure, "_bootout", lambda _label: None)
    monkeypatch.setattr(
        configure,
        "_bootstrap",
        lambda _label: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    assert configure.apply_command(_configure_args()) == 2
    assert "ROLLBACK FAILED" in capsys.readouterr().err
    assert load_config(current_path) == current


def test_config_show_and_credential_command_paths(monkeypatch, tmp_path, capsys):
    configure, home, _, current = _installed_config(monkeypatch, tmp_path)
    atomic_write(pending_config_path(home), config_bytes(current.updated(model="next")))
    assert configure.config_show_command(types.SimpleNamespace(label=None)) == 0
    assert json.loads(capsys.readouterr().out)["pending"]["model"] == "next"

    assert (
        configure.credential_command(
            types.SimpleNamespace(label=None, credential_command="status")
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["configured"] is False

    monkeypatch.setattr(configure, "is_root", lambda: False)
    assert (
        configure.credential_command(
            types.SimpleNamespace(label=None, credential_command="unset")
        )
        == 1
    )
    monkeypatch.setattr(configure, "is_root", lambda: True)
    assert (
        configure.credential_command(
            types.SimpleNamespace(label=None, credential_command="unset")
        )
        == 0
    )
    assert (
        configure.credential_command(
            types.SimpleNamespace(label=None, credential_command="bogus")
        )
        == 2
    )


@pytest.mark.parametrize("secret", ["", "one\ntwo\n"])
def test_credential_rejects_invalid_input(monkeypatch, tmp_path, secret):
    configure, _, _, _ = _installed_config(monkeypatch, tmp_path)
    monkeypatch.setattr(configure, "is_root", lambda: True)
    monkeypatch.setattr(configure.sys, "stdin", io.StringIO(secret))
    assert (
        configure.credential_command(
            types.SimpleNamespace(label=None, credential_command="set")
        )
        == 1
    )


def test_credential_rejects_tty_and_write_failure(monkeypatch, tmp_path):
    configure, _, _, _ = _installed_config(monkeypatch, tmp_path)
    monkeypatch.setattr(configure, "is_root", lambda: True)
    tty = io.StringIO("secret\n")
    tty.isatty = lambda: True
    monkeypatch.setattr(configure.sys, "stdin", tty)
    assert (
        configure.credential_command(
            types.SimpleNamespace(label=None, credential_command="set")
        )
        == 1
    )

    stream = io.StringIO("secret\n")
    monkeypatch.setattr(configure.sys, "stdin", stream)
    monkeypatch.setattr(
        configure,
        "ensure_credential_dir",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk")),
    )
    assert (
        configure.credential_command(
            types.SimpleNamespace(label=None, credential_command="set")
        )
        == 2
    )


def test_runtime_stream_copy_success_and_log_failure(monkeypatch, capsys):
    from vllm_mlx.headless_service import runtime

    class Source:
        def __init__(self, chunks):
            self.chunks = iter(chunks)
            self.closed = False

        def read(self, _size):
            return next(self.chunks, b"")

        def close(self):
            self.closed = True

    class Child:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    writes = []
    source = Source([b"a", b""])
    runtime._copy_stream(source, types.SimpleNamespace(write=writes.append), Child())
    assert writes == [b"a"] and source.closed

    child = Child()
    source = Source([b"bad", b"drain", b""])
    sink = types.SimpleNamespace(
        write=lambda _data: (_ for _ in ()).throw(OSError("full"))
    )
    runtime._copy_stream(source, sink, child)
    assert child.terminated and source.closed
    assert "log write failed" in capsys.readouterr().err


def test_runtime_supervisor_forwards_signals_and_closes(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import runtime

    monkeypatch.setattr(runtime, "log_dir_for", lambda _user: tmp_path)
    logs = []

    class Log:
        def __init__(self, path, **options):
            self.path, self.options, self.closed = path, options, False
            logs.append(self)

        def write(self, _data):
            pass

        def close(self):
            self.closed = True

    handlers = {}

    def fake_signal(signum, handler):
        previous = handlers.get(signum, f"old-{signum}")
        handlers[signum] = handler
        if callable(handler) and not any(
            callable(value) for value in handlers.values() if value is not handler
        ):
            handler(signum, None)
        return previous

    class Pipe(io.BytesIO):
        pass

    class Child:
        def __init__(self, *_args, **_kwargs):
            self.stdout, self.stderr = Pipe(b"out"), Pipe(b"err")
            self.signals = []

        def poll(self):
            return None

        def send_signal(self, signum):
            self.signals.append(signum)

        def wait(self):
            handlers[__import__("signal").SIGTERM](__import__("signal").SIGTERM, None)
            return -15

    class Thread:
        def __init__(self, target, args, daemon):
            del daemon
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

        def join(self, timeout):
            assert timeout == 5

    monkeypatch.setattr(runtime, "RotatingLog", Log)
    monkeypatch.setattr(runtime.signal, "signal", fake_signal)
    monkeypatch.setattr(runtime.subprocess, "Popen", Child)
    monkeypatch.setattr(runtime.threading, "Thread", Thread)
    assert runtime._supervise(["rapid-mlx"], {}, _config(service_user="runner")) == 143
    assert all(log.closed for log in logs)


def test_runtime_supervisor_requires_log_home(monkeypatch):
    from vllm_mlx.headless_service import runtime

    monkeypatch.setattr(runtime, "log_dir_for", lambda _user: None)
    with pytest.raises(ServiceConfigError, match="log directory"):
        runtime._supervise([], {}, _config())


@pytest.mark.parametrize("credential", [None, "bad\nvalue\n"])
def test_runtime_rejects_uid_or_bad_credential(monkeypatch, tmp_path, credential):
    from vllm_mlx.headless_service import runtime

    path = tmp_path / "credential"
    if credential is not None:
        path.write_text(credential)
        path.chmod(0o600)
    config = _config(
        service_user="runner", credential_file=str(path) if credential else None
    )
    config_file = tmp_path / "service.json"
    atomic_write(config_file, config_bytes(config))
    uid = os.getuid() + 1 if credential is None else os.getuid()
    monkeypatch.setattr(
        runtime.pwd, "getpwnam", lambda _user: types.SimpleNamespace(pw_uid=uid)
    )
    assert runtime.run_command(types.SimpleNamespace(config=str(config_file))) == 78


def test_rotating_log_defensive_paths(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import rotating_logs

    sink = rotating_logs.RotatingLog(
        tmp_path / "server.log", max_bytes=8, backup_count=1, retention_days=1
    )
    sink.write(b"")
    backup = tmp_path / "server.log.old"
    backup.write_bytes(b"old")
    os.utime(backup, (0, 0))
    sink._purge()
    assert not backup.exists()

    class BadTell(io.BytesIO):
        def tell(self):
            raise OSError("unknown")

    sink.path.write_bytes(b"1234")
    sink._handle = BadTell()
    sink.write(b"x")
    sink.close()

    real_fstat = rotating_logs.os.fstat
    monkeypatch.setattr(
        rotating_logs.os,
        "fstat",
        lambda _fd: types.SimpleNamespace(st_mode=stat.S_IFDIR),
    )
    with pytest.raises(OSError, match="non-regular"):
        sink._open()
    monkeypatch.setattr(rotating_logs.os, "fstat", real_fstat)


@pytest.mark.parametrize(
    ("plist", "home", "expected"),
    [
        ({"UserName": ""}, None, None),
        ({"UserName": "runner"}, None, None),
        (
            {
                "UserName": "runner",
                "EnvironmentVariables": {"HOME": "/Users/runner"},
                "ProgramArguments": ["rapid-mlx", "run", "--config", "/cfg.json"],
            },
            None,
            Path("/cfg.json"),
        ),
    ],
)
def test_installed_identity_variants(monkeypatch, plist, home, expected):
    from vllm_mlx.headless_service import definition

    monkeypatch.setattr(definition, "installed_plist", lambda _label: plist)
    monkeypatch.setattr(definition, "home_for_user", lambda _user: home)
    result = definition.installed_identity()
    assert (result[2] if result else None) == expected


def test_restart_reads_config_backed_bind(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import restart

    path = tmp_path / "service.json"
    atomic_write(path, config_bytes(_config(host="0.0.0.0", port=9000)))
    monkeypatch.setattr(
        "vllm_mlx.headless_service.definition.installed_identity",
        lambda _label: ("runner", tmp_path, path),
    )
    assert restart._declared_bind("com.rapidmlx.server") == ("0.0.0.0", 9000)


def test_configure_remaining_error_paths(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import configure

    monkeypatch.setattr(
        configure.pwd, "getpwnam", lambda _user: (_ for _ in ()).throw(KeyError())
    )
    with pytest.raises(ServiceConfigError, match="no longer exists"):
        configure._account("gone")

    module, home, _, current = _installed_config(monkeypatch, tmp_path)
    atomic_write(pending_config_path(home), config_bytes(current.updated(model="next")))
    monkeypatch.setattr(module, "is_root", lambda: True)
    monkeypatch.setattr(
        module,
        "_account",
        lambda _user: (_ for _ in ()).throw(ServiceConfigError("gone")),
    )
    assert module.apply_command(_configure_args()) == 1

    monkeypatch.setattr(module, "installed_identity", lambda _label: None)
    assert module.config_show_command(types.SimpleNamespace(label=None)) == 1
    assert (
        module.credential_command(
            types.SimpleNamespace(label=None, credential_command="status")
        )
        == 1
    )


def test_apply_restore_and_pending_unlink_oserrors(monkeypatch, tmp_path):
    configure, home, current_path, current = _installed_config(monkeypatch, tmp_path)
    pending = pending_config_path(home)
    atomic_write(pending, config_bytes(current.updated(model="next")))
    monkeypatch.setattr(configure, "is_root", lambda: True)
    monkeypatch.setattr(configure, "_bootout", lambda _label: None)
    calls = 0

    def write(path, data):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("restore disk failure")
        atomic_write(path, data)

    monkeypatch.setattr(configure, "atomic_write_definition", write)
    monkeypatch.setattr(
        configure,
        "_bootstrap",
        lambda _label: types.SimpleNamespace(returncode=1, stdout="", stderr="bad"),
    )
    assert configure.apply_command(_configure_args()) == 2

    atomic_write(current_path, config_bytes(current))
    atomic_write(pending, config_bytes(current.updated(model="good")))
    monkeypatch.setattr(
        configure,
        "atomic_write_definition",
        lambda path, data: atomic_write(path, data),
    )
    monkeypatch.setattr(
        configure,
        "_bootstrap",
        lambda _label: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(configure, "_wait_ready", lambda *_a: True)
    real_unlink = Path.unlink
    monkeypatch.setattr(
        Path, "unlink", lambda *_a, **_k: (_ for _ in ()).throw(OSError("busy"))
    )
    assert configure.apply_command(_configure_args()) == 0
    monkeypatch.setattr(Path, "unlink", real_unlink)


def test_credential_unset_oserror(monkeypatch, tmp_path):
    configure, _, _, _ = _installed_config(monkeypatch, tmp_path)
    monkeypatch.setattr(configure, "is_root", lambda: True)
    monkeypatch.setattr(
        Path, "unlink", lambda *_a, **_k: (_ for _ in ()).throw(OSError("busy"))
    )
    assert (
        configure.credential_command(
            types.SimpleNamespace(label=None, credential_command="unset")
        )
        == 2
    )


def test_install_service_config_helper(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import config as config_module
    from vllm_mlx.headless_service import install

    monkeypatch.setattr(config_module, "SERVICE_CONFIG_ROOT", tmp_path / "definitions")
    monkeypatch.setattr(
        "pwd.getpwnam",
        lambda _user: types.SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid()),
    )
    path = tmp_path / "definitions" / "service.json"
    install._install_service_config(user="runner", home=tmp_path, path=path, data=b"{}")
    assert path.read_bytes() == b"{}"


def test_status_config_and_human_diagnostics(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import definition, status

    config_file = tmp_path / "service.json"
    credential = tmp_path / "credential"
    credential.write_text("secret\n")
    credential.chmod(0o600)
    effective = _config(
        service_user="runner",
        host="127.0.0.2",
        port=9000,
        credential_file=str(credential),
    )
    atomic_write(config_file, config_bytes(effective))
    plist = {
        "UserName": "runner",
        "ProgramArguments": ["/bin/rapid-mlx", "run", "--config", str(config_file)],
    }
    print_out = "state = running\nruns = 4\nlast exit code = 7\n"
    monkeypatch.setattr(status, "_launchctl_print", lambda _label: print_out)
    monkeypatch.setattr(status, "_read_installed_plist", lambda _label: plist)
    monkeypatch.setattr(
        definition,
        "installed_identity",
        lambda _label: ("runner", tmp_path, config_file),
    )
    monkeypatch.setattr(status, "_endpoint_health", lambda *_a: (True, True))
    monkeypatch.setattr(status, "_port_busy", lambda *_a: True)
    data = status.collect_status()
    assert data["config_sha256"] == config_digest(effective)
    assert data["credential_configured"] is True

    data.update(
        pid=None,
        crash_loop_suspected=True,
        config_error="bad config",
        pending_config=True,
        credential_configured=None,
        log_dir=str(tmp_path),
    )
    rendered = status._render_human(data)
    assert "startup failure" in rendered
    assert "PENDING changes" in rendered
    assert "unknown (run status with sudo)" in rendered

    config_file.write_text("{")
    broken = status.collect_status()
    assert broken["config_error"]


def _upgrade_fixture(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import upgrade

    configure, home, _, _ = _installed_config(monkeypatch, tmp_path)
    python = home / ".rapid-mlx" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o700)
    monkeypatch.setattr(upgrade, "_account", configure._account)
    return upgrade, python


def test_upgrade_helpers(monkeypatch, tmp_path):
    upgrade, python = _upgrade_fixture(monkeypatch, tmp_path)
    with pytest.raises(ServiceConfigError, match="extras"):
        upgrade._target(None, "bad,extra!")
    calls = []
    monkeypatch.setattr(
        upgrade.subprocess,
        "run",
        lambda argv, **kwargs: (
            calls.append((argv, kwargs)) or types.SimpleNamespace(returncode=0)
        ),
    )
    upgrade._as_user("runner", ["cmd"], timeout=7)
    assert calls[0][0][:4] == ["/usr/bin/sudo", "-u", "runner", "-H"]

    monkeypatch.setattr(upgrade, "_bootout", lambda label: calls.append((label, {})))
    monkeypatch.setattr(
        upgrade, "_as_user", lambda *_a, **_k: types.SimpleNamespace(returncode=1)
    )
    assert not upgrade._restore(
        user="runner",
        python=python,
        requirements=tmp_path / "r",
        label="label",
        host="h",
        port=1,
    )
    monkeypatch.setattr(
        upgrade, "_as_user", lambda *_a, **_k: types.SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        upgrade, "_bootstrap", lambda _label: types.SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(upgrade, "_wait_ready", lambda *_a: True)
    assert upgrade._restore(
        user="runner",
        python=python,
        requirements=tmp_path / "r",
        label="label",
        host="h",
        port=1,
    )


def test_upgrade_preflight_failure_paths(monkeypatch, tmp_path):
    upgrade, python = _upgrade_fixture(monkeypatch, tmp_path)
    args = types.SimpleNamespace(
        label=None, dry_run=False, version=None, extras=None, pre=False
    )
    real_identity = upgrade._identity_or_error
    monkeypatch.setattr(
        upgrade,
        "_identity_or_error",
        lambda _label: (_ for _ in ()).throw(ServiceConfigError("missing")),
    )
    assert upgrade.upgrade_command(args) == 1
    monkeypatch.setattr(upgrade, "_identity_or_error", real_identity)

    upgrade, python = _upgrade_fixture(monkeypatch, tmp_path / "second")
    python.unlink()
    assert upgrade.upgrade_command(args) == 1

    upgrade, _ = _upgrade_fixture(monkeypatch, tmp_path / "third")
    assert (
        upgrade.upgrade_command(
            types.SimpleNamespace(**{**vars(args), "dry_run": True})
        )
        == 0
    )
    monkeypatch.setattr(upgrade, "is_root", lambda: False)
    assert upgrade.upgrade_command(args) == 1


def test_upgrade_snapshot_and_install_failure_paths(monkeypatch, tmp_path, capsys):
    upgrade, _ = _upgrade_fixture(monkeypatch, tmp_path)
    args = types.SimpleNamespace(
        label=None, dry_run=False, version=None, extras=None, pre=True
    )
    monkeypatch.setattr(upgrade, "is_root", lambda: True)
    monkeypatch.setattr(
        upgrade,
        "_as_user",
        lambda *_a, **_k: types.SimpleNamespace(
            returncode=1, stdout="", stderr="freeze"
        ),
    )
    assert upgrade.upgrade_command(args) == 2

    responses = iter(
        [
            types.SimpleNamespace(returncode=0, stdout="pkg==1\n", stderr=""),
            types.SimpleNamespace(returncode=1, stdout="", stderr="install"),
        ]
    )
    monkeypatch.setattr(upgrade, "_as_user", lambda *_a, **_k: next(responses))
    monkeypatch.setattr(upgrade, "_bootout", lambda _label: None)
    monkeypatch.setattr(upgrade, "_restore", lambda **_k: False)
    assert upgrade.upgrade_command(args) == 2
    assert "ROLLBACK FAILED" in capsys.readouterr().err


def test_upgrade_snapshot_write_and_doctor_warning(monkeypatch, tmp_path, capsys):
    upgrade, _ = _upgrade_fixture(monkeypatch, tmp_path)
    args = types.SimpleNamespace(
        label=None, dry_run=False, version=None, extras=None, pre=False
    )
    monkeypatch.setattr(upgrade, "is_root", lambda: True)
    monkeypatch.setattr(
        upgrade,
        "_as_user",
        lambda *_a, **_k: types.SimpleNamespace(
            returncode=0, stdout="pkg==1\n", stderr=""
        ),
    )
    real_atomic_write = upgrade.atomic_write
    monkeypatch.setattr(
        upgrade,
        "atomic_write",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk")),
    )
    assert upgrade.upgrade_command(args) == 2
    monkeypatch.setattr(upgrade, "atomic_write", real_atomic_write)

    upgrade, _ = _upgrade_fixture(monkeypatch, tmp_path / "doctor")
    monkeypatch.setattr(upgrade, "is_root", lambda: True)
    responses = iter(
        [
            types.SimpleNamespace(returncode=0, stdout="pkg==1\n", stderr=""),
            types.SimpleNamespace(returncode=0, stdout="", stderr=""),
            types.SimpleNamespace(returncode=1, stdout="", stderr="doctor"),
        ]
    )
    monkeypatch.setattr(upgrade, "_as_user", lambda *_a, **_k: next(responses))
    monkeypatch.setattr(upgrade, "_bootout", lambda _label: None)
    monkeypatch.setattr(
        upgrade, "_bootstrap", lambda _label: types.SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(upgrade, "_wait_ready", lambda *_a: True)
    assert upgrade.upgrade_command(args) == 0
    assert "doctor" in capsys.readouterr().err


def test_atomic_write_cleanup_tolerates_unlink_failure(monkeypatch, tmp_path):
    from vllm_mlx.headless_service import config as config_module

    monkeypatch.setattr(
        config_module.os,
        "replace",
        lambda *_a: (_ for _ in ()).throw(OSError("replace")),
    )
    monkeypatch.setattr(
        Path, "unlink", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unlink"))
    )
    with pytest.raises(OSError, match="replace"):
        atomic_write(tmp_path / "target", b"data")


def test_rotating_log_purge_tolerates_filesystem_error(monkeypatch, tmp_path):
    from vllm_mlx.headless_service.rotating_logs import RotatingLog

    sink = RotatingLog(tmp_path / "log", max_bytes=1, backup_count=1, retention_days=1)
    backup = tmp_path / "log.old"
    backup.write_bytes(b"old")
    os.utime(backup, (0, 0))
    monkeypatch.setattr(
        Path, "unlink", lambda *_a, **_k: (_ for _ in ()).throw(OSError("busy"))
    )
    sink._purge()


def test_install_rejects_invalid_generated_definition(monkeypatch):
    from vllm_mlx.headless_service import common, install

    monkeypatch.setattr(install, "validate_service_account", lambda _user: None)
    monkeypatch.setattr(common, "home_for_user", lambda _user: Path("/Users/runner"))
    monkeypatch.setattr(install, "resolve_executable", lambda _home: "/bin/rapid-mlx")
    args = types.SimpleNamespace(
        label="invalid/label",
        model="model",
        service_user="runner",
        host="127.0.0.1",
        port=8000,
        serve_args=[],
        dry_run=True,
    )
    assert install.install_command(args) == 1
