"""Disk-write switches: --disable-disk-caches, --log-file, log level, version check."""

import argparse
import asyncio
import os
import socket
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from rapid_mlx import disk_caches, log_file
from rapid_mlx._env import env_falsey, env_truthy, is_falsey, is_truthy
from rapid_mlx.headless_service.install import (
    ServiceInstallError,
    log_file_from_serve_args,
    refuse_unbootable_log_file,
)


@pytest.fixture(autouse=True)
def _clean_switches(monkeypatch):
    for name in (
        disk_caches.ENV_VAR,
        log_file.ENV_VAR,
        "APC_DISK_ENABLED",
        "RAPID_MLX_PREFIX_CACHE_AUTOLOAD",
        "RAPID_MLX_LOG_LEVEL",
        "RAPID_MLX_DISABLE_VERSION_CHECK",
        "CI",
    ):
        monkeypatch.delenv(name, raising=False)
    disk_caches.configure(False)
    yield
    disk_caches.configure(False)


@pytest.mark.parametrize(
    ("value", "truthy", "falsey"),
    [
        ("1", True, False),
        ("true", True, False),
        ("YES", True, False),
        ("on", True, False),
        ("Enable", True, False),
        ("enabled", True, False),
        (" 1 ", True, False),
        ("0", False, True),
        ("false", False, True),
        ("NO", False, True),
        ("off", False, True),
        ("disable", False, True),
        (" Disabled ", False, True),
        ("", False, False),
        ("maybe", False, False),
        ("2", False, False),
    ],
)
def test_env_helpers_classify_values(monkeypatch, value, truthy, falsey):
    monkeypatch.setenv("RAPID_MLX_TEST_SWITCH", value)
    assert env_truthy("RAPID_MLX_TEST_SWITCH") is truthy
    assert env_falsey("RAPID_MLX_TEST_SWITCH") is falsey
    assert is_truthy(value) is truthy
    assert is_falsey(value) is falsey


def test_env_helpers_treat_unset_as_neither(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_TEST_SWITCH", raising=False)
    assert env_truthy("RAPID_MLX_TEST_SWITCH") is False
    assert env_falsey("RAPID_MLX_TEST_SWITCH") is False
    assert is_truthy(None) is False
    assert is_falsey(None) is False


def test_disk_caches_enabled_by_default():
    assert disk_caches.disabled() is False
    assert disk_caches.source() is None


def test_disk_caches_flag_wins_and_names_its_source(monkeypatch):
    monkeypatch.setenv(disk_caches.ENV_VAR, "1")
    disk_caches.configure(True)
    assert disk_caches.source() == "--disable-disk-caches"


def test_disk_caches_env_switch(monkeypatch):
    monkeypatch.setenv(disk_caches.ENV_VAR, "yes")
    assert disk_caches.source() == disk_caches.ENV_VAR
    monkeypatch.setenv(disk_caches.ENV_VAR, "0")
    assert disk_caches.disabled() is False


def test_overruled_settings_name_explicit_disk_writers(monkeypatch):
    monkeypatch.setenv("APC_DISK_ENABLED", "1")
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_AUTOLOAD", "1")
    assert disk_caches.overruled_settings(kv_disk_checkpoint_interval=256) == [
        "--kv-disk-checkpoint-interval 256",
        "APC_DISK_ENABLED=1",
        "RAPID_MLX_PREFIX_CACHE_AUTOLOAD=1",
    ]


def test_overruled_settings_ignore_settings_that_are_off(monkeypatch):
    monkeypatch.setenv("APC_DISK_ENABLED", "0")
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_AUTOLOAD", "0")
    assert disk_caches.overruled_settings(kv_disk_checkpoint_interval=0) == []
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_AUTOLOAD", "disabled")
    assert disk_caches.overruled_settings(kv_disk_checkpoint_interval=0) == []


def test_disk_cache_policy_logged_with_overruled_settings(monkeypatch):
    from rapid_mlx.cli import _log_disk_cache_policy

    logger = MagicMock()
    args = argparse.Namespace(kv_disk_checkpoint_interval=64)
    _log_disk_cache_policy(args, logger)
    logger.info.assert_not_called()
    logger.warning.assert_not_called()

    monkeypatch.setenv("APC_DISK_ENABLED", "1")
    disk_caches.configure(True)
    _log_disk_cache_policy(args, logger)
    assert logger.info.call_args.args[1] == "--disable-disk-caches"
    warned = [call.args[1] for call in logger.warning.call_args_list]
    assert warned == ["--kv-disk-checkpoint-interval 64", "APC_DISK_ENABLED=1"]


@pytest.mark.parametrize(
    ("func", "word"),
    [("_shutdown_save_prefix_cache", "save"), ("_deferred_load_prefix_cache", "load")],
)
def test_prefix_cache_lifespan_steps_report_why_they_skip(
    monkeypatch, caplog, func, word
):
    server = pytest.importorskip("rapid_mlx.server")
    step = getattr(server, func)

    monkeypatch.setenv(disk_caches.ENV_VAR, "1")
    with caplog.at_level("INFO", logger="rapid_mlx.server"):
        asyncio.run(step())
    assert (
        f"Prefix-cache auto-{word} skipped: disk caches disabled by "
        f"{disk_caches.ENV_VAR}" in caplog.text
    )

    caplog.clear()
    monkeypatch.delenv(disk_caches.ENV_VAR)
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_AUTOLOAD", "off")
    with caplog.at_level("INFO", logger="rapid_mlx.server"):
        asyncio.run(step())
    assert "RAPID_MLX_PREFIX_CACHE_AUTOLOAD" in caplog.text
    assert "disk caches disabled" not in caplog.text


def test_prefix_cache_persistence_respects_disk_caches():
    server = pytest.importorskip("rapid_mlx.server")
    assert server._automatic_prefix_cache_persistence_enabled() is True
    disk_caches.configure(True)
    assert server._automatic_prefix_cache_persistence_enabled() is False


def test_kv_checkpoint_skipped_when_disk_caches_disabled():
    scheduler = pytest.importorskip("rapid_mlx.scheduler")
    fake = argparse.Namespace(
        config=argparse.Namespace(kv_disk_checkpoint_interval=256)
    )
    disk_caches.configure(True)
    # A request without attributes proves the hook returns before reading it.
    assert scheduler.Scheduler._maybe_disk_checkpoint(fake, object(), None) is None


def test_log_file_flag_wins_over_environment(monkeypatch):
    monkeypatch.setenv(log_file.ENV_VAR, "/tmp/env.log")
    assert log_file.resolve("-") == ("-", "--log-file")
    assert log_file.resolve(None) == ("/tmp/env.log", log_file.ENV_VAR)


def test_log_file_unset_means_stderr():
    assert log_file.resolve(None) == (None, None)


def test_log_file_validation(tmp_path):
    assert log_file.validate("-", "--log-file") == "-"
    assert log_file.validate("/dev/null", "--log-file") == "/dev/null"
    target = tmp_path / "server.log"
    assert log_file.validate(str(target), "--log-file") == str(target)
    with pytest.raises(log_file.LogFileError, match="does not exist"):
        log_file.validate(str(tmp_path / "missing" / "x.log"), "--log-file")
    with pytest.raises(log_file.LogFileError, match="is a directory"):
        log_file.validate(str(tmp_path), log_file.ENV_VAR)


def test_log_file_apply_captures_print_and_native_output(tmp_path):
    target = tmp_path / "server.log"
    target.write_text("previous\n")
    script = (
        "import os, sys\n"
        "from rapid_mlx import log_file\n"
        f"log_file.apply({str(target)!r})\n"
        "print('from python stdout')\n"
        "print('from python stderr', file=sys.stderr)\n"
        "sys.stdout.flush()\n"
        "os.write(2, b'from fd 2\\n')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout == ""
    assert result.stderr == ""
    content = target.read_text()
    assert content.startswith("previous\n")
    for line in ("from python stdout", "from python stderr", "from fd 2"):
        assert line in content


def test_log_file_apply_dash_moves_stderr_to_stdout():
    script = (
        "import sys\n"
        "from rapid_mlx import log_file\n"
        "log_file.apply('-')\n"
        "print('err line', file=sys.stderr)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert "err line" in result.stdout
    assert result.stderr == ""


def test_json_command_rejects_stdout_log_target_from_environment():
    env = {**os.environ, log_file.ENV_VAR: "-"}
    result = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "recipe", "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "RAPID_MLX_LOG_FILE=-" in result.stderr


def test_serve_log_level_resolution(monkeypatch):
    from rapid_mlx.cli import _resolve_serve_log_level

    assert _resolve_serve_log_level(None) == "INFO"
    monkeypatch.setenv("RAPID_MLX_LOG_LEVEL", "warning")
    assert _resolve_serve_log_level(None) == "WARNING"
    assert _resolve_serve_log_level("DEBUG") == "DEBUG"
    monkeypatch.setenv("RAPID_MLX_LOG_LEVEL", "trace")
    with pytest.raises(SystemExit) as exc:
        _resolve_serve_log_level(None)
    assert exc.value.code == 2


def test_version_check_zero_does_not_disable(monkeypatch):
    from rapid_mlx import _version_check as vc

    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "0")
    assert vc._explicitly_disabled() is False
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    assert vc._explicitly_disabled() is True


def test_disable_version_check_flag_sets_environment_for_children(monkeypatch):
    from rapid_mlx import cli

    monkeypatch.setattr(
        sys, "argv", ["rapid-mlx", "--no-banner", "--disable-version-check", "version"]
    )
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)
    assert os.environ["RAPID_MLX_DISABLE_VERSION_CHECK"] == "1"


class _StopServeError(Exception):
    pass


def test_serve_command_applies_log_file_and_disk_caches(monkeypatch, tmp_path):
    from rapid_mlx import cli
    from rapid_mlx.runtime import optional_runtime

    applied = []
    monkeypatch.setattr(log_file, "apply", lambda t, s: applied.append((t, s)))

    def _stop(_yes):
        raise _StopServeError

    monkeypatch.setattr(optional_runtime, "set_assume_yes", _stop)
    target = tmp_path / "serve.log"
    args = argparse.Namespace(
        log_level=None, log_file=str(target), disable_disk_caches=True
    )
    with pytest.raises(_StopServeError):
        cli.serve_command(args)
    assert applied == [(str(target), "--log-file")]
    assert disk_caches.source() == "--disable-disk-caches"
    assert args.log_level == "INFO"


def test_serve_command_rejects_unusable_log_file(monkeypatch, tmp_path, capsys):
    from rapid_mlx import cli

    monkeypatch.setattr(log_file, "apply", lambda *_: pytest.fail("must not apply"))
    args = argparse.Namespace(
        log_level=None,
        log_file=str(tmp_path / "missing" / "serve.log"),
        disable_disk_caches=False,
    )
    with pytest.raises(SystemExit) as exc:
        cli.serve_command(args)
    assert exc.value.code == 2
    assert "does not exist" in capsys.readouterr().err


def test_serve_command_reports_unwritable_log_file(tmp_path, capsys):
    from rapid_mlx import cli

    target = tmp_path / "read-only.log"
    target.touch(mode=0o400)
    args = argparse.Namespace(
        log_level=None, log_file=str(target), disable_disk_caches=False
    )
    with pytest.raises(SystemExit) as exc:
        cli.serve_command(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"error: --log-file {str(target)!r} cannot be opened for writing" in err
    assert "Traceback" not in err


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_serve_sends_all_output_to_log_file(tmp_path):
    pytest.importorskip("mlx.core")
    model_dir = tmp_path / "empty-model"
    model_dir.mkdir()
    target = tmp_path / "serve.log"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in (disk_caches.ENV_VAR, log_file.ENV_VAR, "RAPID_MLX_LOG_LEVEL")
    }
    # The CLI's own telemetry notice is printed before serve redirects its
    # output and depends on this host's saved consent.
    env["RAPID_MLX_TELEMETRY"] = "0"
    # The empty model directory makes the server stop during startup.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rapid_mlx.cli",
            "--no-banner",
            "--disable-version-check",
            "serve",
            str(model_dir),
            "--port",
            str(_free_port()),
            "--disable-disk-caches",
            "--kv-disk-checkpoint-interval",
            "64",
            "--log-file",
            str(target),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == ""
    content = target.read_text()
    assert "Disk caches disabled by --disable-disk-caches" in content
    assert (
        "--kv-disk-checkpoint-interval 64 ignored: disk caches disabled by "
        "--disable-disk-caches" in content
    )


def _parse(argv):
    pytest.importorskip("websockets")
    from rapid_mlx.cli import build_parser

    return build_parser().parse_args(argv)


def test_parser_accepts_new_options():
    args = _parse(
        [
            "--disable-version-check",
            "serve",
            "m",
            "--disable-disk-caches",
            "--log-file",
            "/dev/null",
        ]
    )
    assert args.disable_version_check is True
    assert args.disable_disk_caches is True
    assert args.log_file == "/dev/null"
    assert args.log_level is None
    for command in ("chat", "start", "share"):
        parsed = _parse([command, "m", "--disable-disk-caches", "--log-file", "-"])
        assert parsed.disable_disk_caches is True
        assert parsed.log_file == "-"


def test_chat_spawn_forwards_settings_and_inherits_stdio(monkeypatch, tmp_path):
    from rapid_mlx import cli

    captured = {}

    class FakeProc:
        pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    log_path = tmp_path / "chat.log"
    proc, _ = cli._spawn_chat_server(
        "m",
        str(log_path),
        disable_disk_caches=True,
        log_target="/dev/null",
    )
    assert "--disable-disk-caches" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--log-level") + 1] == "WARNING"
    assert captured["kwargs"]["stdout"] is None
    assert captured["kwargs"]["stderr"] is None
    assert captured["kwargs"]["env"][log_file.ENV_VAR] == "/dev/null"
    assert proc._rapid_mlx_log is None
    assert not log_path.exists()


def test_chat_spawn_leaves_level_to_environment(monkeypatch, tmp_path):
    from rapid_mlx import cli

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return argparse.Namespace()

    monkeypatch.setenv("RAPID_MLX_LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    proc, _ = cli._spawn_chat_server("m", str(tmp_path / "chat.log"))
    proc._rapid_mlx_log.close()
    assert "--log-level" not in captured["cmd"]


@pytest.mark.parametrize(
    ("serve_args", "expected"),
    [
        ((), None),
        (("--max-num-seqs", "4"), None),
        (("--log-file", "/var/log/x.log"), "/var/log/x.log"),
        (("--log-file=/dev/null",), "/dev/null"),
    ],
)
def test_service_log_file_extraction(serve_args, expected):
    assert log_file_from_serve_args(serve_args) == expected


def test_service_refuses_log_files_unavailable_at_boot(tmp_path):
    refuse_unbootable_log_file(("--log-file", "-"))
    refuse_unbootable_log_file(("--log-file", "/dev/null"))
    refuse_unbootable_log_file(
        ("--log-file", str(tmp_path / "server.log")), require_directory=True
    )
    with pytest.raises(ServiceInstallError, match="absolute"):
        refuse_unbootable_log_file(("--log-file", "server.log"))
    with pytest.raises(ServiceInstallError, match="/Volumes"):
        refuse_unbootable_log_file(("--log-file", "/Volumes/RAMDisk/server.log"))
    with pytest.raises(ServiceInstallError, match="does not exist"):
        refuse_unbootable_log_file(
            ("--log-file", str(tmp_path / "missing" / "server.log")),
            require_directory=True,
        )
