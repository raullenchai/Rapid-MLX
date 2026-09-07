"""Tests for ``rapid-mlx service`` — the headless macOS service lifecycle.

These tests are hermetic: they never touch launchd, the real account
database (``pwd``/``grp``), or the real ``/Library/LaunchDaemons``. User
accounts are simulated by monkeypatching ``pwd``/``grp``/``home_for_user``;
mutations are exercised only through the pure generators and the
``--dry-run`` surface.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vllm_mlx.cli import build_parser
from vllm_mlx.headless_service import common
from vllm_mlx.headless_service import install as ins_mod
from vllm_mlx.headless_service.plist import (
    build_plist_dict,
    parse_plist,
    serialize_plist,
)


@pytest.fixture
def plist_kwargs():
    return dict(
        label="com.rapidmlx.server",
        user="serveuser",
        executable="/Users/serveuser/.local/bin/rapid-mlx",
        model="qwen3.5-4b-4bit",
        home=Path("/Users/serveuser"),
        log_dir=Path("/Users/serveuser/Library/Logs/Rapid-MLX"),
        host="127.0.0.1",
        port=8000,
        serve_args=(),
    )


# ---------------------------------------------------------------------------
# plist generation — determinism + the documented safety contract.
# ---------------------------------------------------------------------------


def test_plist_is_deterministic(plist_kwargs):
    a = serialize_plist(build_plist_dict(**plist_kwargs))
    b = serialize_plist(build_plist_dict(**plist_kwargs))
    assert a == b


def test_plist_matches_documented_safety_contract(plist_kwargs):
    """Same safety assertions as the static-template test
    ``test_headless_service_assets`` but for the generated plist."""
    parsed = build_plist_dict(**plist_kwargs)
    assert parsed["Label"] == "com.rapidmlx.server"
    assert parsed["UserName"] == "serveuser"
    assert parsed["EnvironmentVariables"]["HOME"] == "/Users/serveuser"
    assert parsed["ProgramArguments"][0].startswith("/Users/serveuser/")
    host_idx = parsed["ProgramArguments"].index("--host")
    assert parsed["ProgramArguments"][host_idx + 1] == "127.0.0.1"
    assert parsed["KeepAlive"] is True
    assert parsed["ThrottleInterval"] >= 10
    assert parsed["Umask"] == 0o27
    assert "ProcessType" not in parsed
    assert "RAPID_MLX_API_KEY" not in parsed["EnvironmentVariables"]
    assert parsed["StandardOutPath"] != parsed["StandardErrorPath"]


def test_plist_argv_round_trips_serve_options(plist_kwargs):
    plist_kwargs["serve_args"] = ("--max-num-seqs", "4")
    argv = build_plist_dict(**plist_kwargs)["ProgramArguments"]
    i = argv.index("--max-num-seqs")
    assert argv[i + 1] == "4"


def test_plist_xml_parse_round_trip(plist_kwargs):
    parsed = parse_plist(serialize_plist(build_plist_dict(**plist_kwargs)))
    assert parsed["Label"] == "com.rapidmlx.server"
    assert parsed["UserName"] == "serveuser"


# ---------------------------------------------------------------------------
# CLI registration + dispatch.
# ---------------------------------------------------------------------------


def test_service_subcommand_registered():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert "service" in choices


def test_service_install_parses_passthrough_after_separator():
    args = build_parser().parse_args(
        [
            "service",
            "install",
            "--service-user",
            "serveuser",
            "--model",
            "qwen3.5-4b-4bit",
            "--",
            "--max-num-seqs",
            "4",
        ]
    )
    assert args.serve_args == ["--", "--max-num-seqs", "4"]


def test_service_macos_guard():
    """On non-darwin the service dispatch must refuse with a clear message."""

    class FakeArgs:
        command = "service"
        service_command = "status"

    # Simulate non-darwin by forcing sys.platform in the module check.
    import vllm_mlx.headless_service.cli as svc_cli

    real_platform = sys.platform
    try:
        sys.platform = "linux"
        with pytest.raises(SystemExit) as ei:
            svc_cli.service_command(FakeArgs())
        assert ei.value.code == 2
    finally:
        sys.platform = real_platform


@pytest.mark.parametrize(
    ("verb", "module_name", "handler_name"),
    [
        ("configure", "configure", "configure_command"),
        ("apply", "configure", "apply_command"),
        ("config", "configure", "config_show_command"),
        ("credential", "configure", "credential_command"),
        ("run", "runtime", "run_command"),
        ("upgrade", "upgrade", "upgrade_command"),
    ],
)
def test_new_service_verbs_dispatch(monkeypatch, verb, module_name, handler_name):
    import importlib

    import vllm_mlx.headless_service.cli as svc_cli

    module = importlib.import_module(f"vllm_mlx.headless_service.{module_name}")
    called = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(module, handler_name, lambda args: called.append(args) or 0)
    args = types.SimpleNamespace(service_command=verb, label=None)
    svc_cli.service_command(args)
    assert called == [args]


# ---------------------------------------------------------------------------
# Service-account validation (monkeypatched account DB).
# ---------------------------------------------------------------------------


def test_validate_rejects_missing_user(monkeypatch):
    monkeypatch.setattr(ins_mod, "user_uid", staticmethod(lambda u: None))
    with pytest.raises(ins_mod.ServiceInstallError, match="does not exist"):
        ins_mod.validate_service_account("nope")


def test_validate_rejects_root():
    with pytest.raises(ins_mod.ServiceInstallError, match="root"):
        ins_mod.validate_service_account("root")


def test_validate_rejects_system_uid(monkeypatch):
    monkeypatch.setattr(ins_mod, "user_uid", staticmethod(lambda u: 33))
    # Patch home too: on some CI hosts the `_www` system user has no home dir,
    # which would make the "does not exist" branch run first and change the
    # error text. Keeping the home deterministic forces the system-uid branch.
    monkeypatch.setattr(
        common, "home_for_user", staticmethod(lambda u: Path(f"/var/{u}"))
    )
    with pytest.raises(ins_mod.ServiceInstallError, match="system account"):
        ins_mod.validate_service_account("_www")


def test_validate_rejects_admin(monkeypatch):
    monkeypatch.setattr(ins_mod, "user_uid", staticmethod(lambda u: 503))
    monkeypatch.setattr(ins_mod, "is_admin_user", staticmethod(lambda u: True))
    monkeypatch.setattr(
        common, "home_for_user", staticmethod(lambda u: Path("/Users/u"))
    )
    with pytest.raises(ins_mod.ServiceInstallError, match="admin"):
        ins_mod.validate_service_account("bob")


def test_validate_accepts_least_privilege(monkeypatch):
    monkeypatch.setattr(ins_mod, "user_uid", staticmethod(lambda u: 503))
    monkeypatch.setattr(ins_mod, "is_admin_user", staticmethod(lambda u: False))
    monkeypatch.setattr(
        common, "home_for_user", staticmethod(lambda u: Path("/Users/serveuser"))
    )
    # Should not raise.
    ins_mod.validate_service_account("serveuser")


# ---------------------------------------------------------------------------
# Secret / port-race guards.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--api-key", "--api-key=secret", "--api_key=secret"])
def test_refuse_secret_flags(flag):
    with pytest.raises(ins_mod.ServiceInstallError, match="API key"):
        ins_mod.refuse_secret_flags((flag, "value"))


def test_allow_non_secret_flags():
    # No raise.
    ins_mod.refuse_secret_flags(("--max-num-seqs", "4", "--use-paged-cache"))


@pytest.mark.parametrize("flag", ["--host", "--port=9000", "--listen-fd"])
def test_refuse_passthrough_bind_overrides(flag):
    with pytest.raises(ins_mod.ServiceInstallError, match="bind option"):
        ins_mod.refuse_secret_flags((flag, "value"))


# ---------------------------------------------------------------------------
# install --dry-run (no system mutation).
# ---------------------------------------------------------------------------


def _valid_user_monkeypatch(monkeypatch):
    monkeypatch.setattr(
        ins_mod, "user_uid", staticmethod(lambda u: 503 if u == "serveuser" else None)
    )
    monkeypatch.setattr(ins_mod, "is_admin_user", staticmethod(lambda u: False))
    monkeypatch.setattr(
        common,
        "home_for_user",
        staticmethod(lambda u: Path("/Users/serveuser") if u == "serveuser" else None),
    )
    # Existing install transaction tests exercise launchd/plist rollback, not
    # account-owned config persistence. Keep their fake /Users path hermetic;
    # dedicated config tests below cover the atomic writer itself.
    monkeypatch.setattr(
        ins_mod,
        "_install_service_config",
        staticmethod(lambda **_kwargs: None),
    )
    # validate_service_account requires the service-user binary to exist.
    monkeypatch.setattr(
        ins_mod,
        "resolve_executable",
        staticmethod(lambda home: "/Users/serveuser/.local/bin/rapid-mlx"),
    )


def _ns(**over):
    import types

    base = dict(
        service_command="install",
        label=None,
        model="qwen3.5-4b-4bit",
        service_user="serveuser",
        host="127.0.0.1",
        port=8000,
        serve_args=[],
        dry_run=True,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_install_dry_run_prints_mutations_and_touches_nothing(monkeypatch, capsys):
    _valid_user_monkeypatch(monkeypatch)
    # Ensure no real port probe fires during dry-run.
    monkeypatch.setattr(ins_mod, "_port_busy", staticmethod(lambda h, p: False))
    code = ins_mod.install_command(_ns())
    assert code == 0
    out = capsys.readouterr().out
    assert "[DRY-RUN]" in out
    assert "launchctl bootstrap system" in out
    assert "install -o root -g wheel -m 644" in out
    # Nothing must actually exist.
    assert not Path("/Library/LaunchDaemons/com.rapidmlx.server.plist").exists()


def test_install_without_service_user_errors(monkeypatch):
    _valid_user_monkeypatch(monkeypatch)
    code = ins_mod.install_command(_ns(service_user=None))
    assert code == 1


def test_install_refuses_admin_user(monkeypatch, capsys):
    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins_mod, "is_admin_user", staticmethod(lambda u: True))
    code = ins_mod.install_command(_ns())
    assert code == 1
    assert "administrator" in capsys.readouterr().err


def test_install_refuses_port_race(monkeypatch, capsys):
    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins_mod, "_port_busy", staticmethod(lambda h, p: True))
    code = ins_mod.install_command(_ns())
    assert code == 1
    assert "already answers" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# uninstall --dry-run (no system mutation).
# ---------------------------------------------------------------------------


def test_uninstall_dry_run_prints_removal_and_touches_nothing(monkeypatch, capsys):
    code = ins_mod.uninstall_command(_ns())
    assert code == 0
    out = capsys.readouterr().out
    assert "launchctl bootout system/com.rapidmlx.server" in out
    assert "rm -f /Library/LaunchDaemons/com.rapidmlx.server.plist" in out
    assert not Path("/Library/LaunchDaemons/com.rapidmlx.server.plist").exists()


# ---------------------------------------------------------------------------
# status --json shape.
# ---------------------------------------------------------------------------


def test_status_json_shape(monkeypatch, capsys):
    from vllm_mlx.headless_service import status as st

    monkeypatch.setattr(st, "_launchctl_print", staticmethod(lambda _label: None))
    monkeypatch.setattr(st, "_read_installed_plist", staticmethod(lambda _label: None))
    monkeypatch.setattr(
        st, "_endpoint_health", staticmethod(lambda h, p: (False, False))
    )
    monkeypatch.setattr(st, "_port_busy", staticmethod(lambda h, p: False))

    import types

    ns = types.SimpleNamespace(
        service_command="status",
        label=None,
        service_user=None,
        host="127.0.0.1",
        port=8000,
        json=True,
    )
    code = st.status_command(ns)
    assert code == 1  # not registered
    data = json.loads(capsys.readouterr().out)
    assert data["registered"] is False
    assert data["pid"] is None
    assert data["plist_present"] is False
    assert set(data) >= {
        "label",
        "domain",
        "registered",
        "pid",
        "model",
        "livez",
        "readyz",
        "port",
        "plist",
        "log_dir",
    }


# ---------------------------------------------------------------------------
# Readiness probe correctness (matches the /readyz contract).
# ---------------------------------------------------------------------------


class _HttpResponder:
    """A thread that answers raw HTTP on 127.0.0.1:0 the way rapid-mlx does."""

    def __init__(self, status_line: bytes, body: bytes):
        import socket as _s
        import threading

        self.sock = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        self.sock.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.status_line = status_line
        self.body = body
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        conn, _ = self.sock.accept()
        conn.settimeout(2)
        try:
            conn.recv(4096)
            conn.sendall(
                self.status_line
                + b"\r\nContent-Length: "
                + str(len(self.body)).encode()
                + b"\r\n\r\n"
                + self.body
            )
        finally:
            conn.close()


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (b"HTTP/1.1 200 OK", b'{"status":"healthy","ready":true}', True),
        (
            b"HTTP/1.1 503 Service Unavailable",
            b'{"status":"draining","ready":false}',
            False,
        ),
        # A bare 200 while the model is still loading must NOT count as ready.
        (b"HTTP/1.1 200 OK", b'{"status":"healthy","ready":false}', False),
        # Whitespace-tolerant: FastAPI compact vs pretty JSON.
        (b"HTTP/1.1 200 OK", b'{"status": "healthy", "ready": true}', True),
    ],
)
def test_readyz_ready_semantics(monkeypatch, status, body, expected):
    from vllm_mlx.headless_service.install import _readyz_ready

    srv = _HttpResponder(status, body)
    try:
        assert _readyz_ready("127.0.0.1", srv.port) is expected
    finally:
        srv.sock.close()


# ---------------------------------------------------------------------------
# launchd Label validation (path-traversal / injection hardening).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../evil", "a b", "x/y", "x;rm", "$(x)", ""])
def test_validate_label_rejects_unsafe(bad):
    from vllm_mlx.headless_service.common import validate_label

    with pytest.raises(ValueError):
        validate_label(bad)


@pytest.mark.parametrize("good", ["com.rapidmlx.server", "com.example-a1.b", "simple"])
def test_validate_label_accepts_safe(good):
    from vllm_mlx.headless_service.common import validate_label

    assert validate_label(good) == good


def test_service_dispatch_rejects_bad_label():
    from vllm_mlx.headless_service.cli import service_command

    class FakeArgs:
        command = "service"
        service_command = "status"
        label = "../../etc/evil"

    # Force darwin so the label-validation path (not the macOS-only guard)
    # is what actually raises here, on every host.
    real_platform = sys.platform
    try:
        sys.platform = "darwin"
        with pytest.raises(SystemExit) as ei:
            service_command(FakeArgs())
        assert ei.value.code == 2
    finally:
        sys.platform = real_platform


def test_service_dispatch_good_label_iterates_each_subcommand(monkeypatch):
    """Dispatch forwards to the right *_command for each subcommand."""
    import vllm_mlx.headless_service.cli as svc
    from vllm_mlx.headless_service import install as disp_ins
    from vllm_mlx.headless_service import logs as disp_logs
    from vllm_mlx.headless_service import restart as disp_rest
    from vllm_mlx.headless_service import status as disp_st

    seen = {}

    def _mk(name):
        def _cmd(args):
            seen[name] = args
            return 0

        return _cmd

    monkeypatch.setattr(disp_ins, "install_command", _mk("install"))
    monkeypatch.setattr(disp_st, "status_command", _mk("status"))
    monkeypatch.setattr(disp_logs, "logs_command", _mk("logs"))
    monkeypatch.setattr(disp_rest, "restart_command", _mk("restart"))
    monkeypatch.setattr(disp_ins, "uninstall_command", _mk("uninstall"))

    class FakeArgs:
        label = "com.rapidmlx.server"

    # Force darwin: on Linux CI the macOS-only guard fires before dispatch.
    real_platform = sys.platform
    try:
        sys.platform = "darwin"
        for name in ("install", "status", "logs", "restart", "uninstall"):
            seen.clear()
            fake = FakeArgs()
            fake.service_command = name
            svc.service_command(fake)  # code 0 → no SystemExit
            assert seen.get(name) is fake, f"dispatch failed for {name}"
    finally:
        sys.platform = real_platform


def test_service_dispatch_unknown_subcommand_errors(monkeypatch, capsys):
    import vllm_mlx.headless_service.cli as svc

    class FakeArgs:
        label = None
        service_command = "bogus"

    # Force darwin so the macOS-only guard doesn't shadow the bad-command error.
    real_platform = sys.platform
    try:
        sys.platform = "darwin"
        with pytest.raises(SystemExit) as ei:
            svc.service_command(FakeArgs())
        assert ei.value.code == 2
        assert (
            "expected one of install/status/logs/restart/uninstall"
            in capsys.readouterr().err
        )
    finally:
        sys.platform = real_platform


# ---------------------------------------------------------------------------
# Reviewer fixes: rollback cleanliness, reinstall tolerance, exec requirement.
# ---------------------------------------------------------------------------


def test_resolve_executable_requires_service_user_binary(monkeypatch):
    """The daemon must run the SERVICE account's binary, not the operator's."""
    from vllm_mlx.headless_service.install import resolve_executable

    # No /Users/serveuser/.local/bin/rapid-mlx exists on this machine.
    with pytest.raises(ins_mod.ServiceInstallError, match="no Rapid-MLX binary"):
        resolve_executable(Path("/Users/serveuser"))


def test_serve_argv_splices_module_invocation(plist_kwargs):
    """A ``-m vllm_mlx`` executable must be separate argv entries, never one
    space-joined argv[0] (which launchd would fail to exec)."""
    from vllm_mlx.headless_service.plist import serve_argv

    argv = serve_argv(["/opt/venv/bin/python", "-m", "vllm_mlx"], "qwen3.5-4b-4bit")
    assert argv[0] == "/opt/venv/bin/python"
    assert argv[1] == "-m"
    assert argv[2] == "vllm_mlx"


def test_install_rollback_removes_plist_on_readiness_failure(monkeypatch, tmp_path):
    """A failed readiness wait must remove the persistent plist so nothing
    auto-starts on reboot (the 'rolled back' guarantee)."""
    import vllm_mlx.headless_service.install as ins

    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "_port_busy", staticmethod(lambda h, p: False))
    monkeypatch.setattr(ins, "is_root", staticmethod(lambda: True))
    # Point the plist directory at a temp dir (raw module constant patch).
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)
    monkeypatch.setattr(ins, "_wait_ready", staticmethod(lambda h, p, **k: False))

    # _run: emulate the real shell-outs enough that the rollback's `rm -f`
    # actually deletes the persistent plist (the readiness wait is faked to
    # fail). plutil/install/bootstrap return success so the only failure is
    # the readiness wait.
    def _fake_run(args, check=True):
        if args and args[0] == "install":
            import shutil

            shutil.copyfile(args[-2], args[-1])
        if args and args[0] == "rm":
            import os

            for path in args[2:]:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        return _FakeResult()

    monkeypatch.setattr(ins, "_run", _fake_run)
    plist = tmp_path / "com.rapidmlx.server.plist"

    code = ins.install_command(_ns(dry_run=False))
    assert code == 2
    # The persistent plist must be gone after the rollback so nothing
    # auto-starts on reboot.
    assert not plist.exists()


class _FakeResult:
    returncode = 0
    stdout = ""
    stderr = "service already loaded"


# ---------------------------------------------------------------------------
# logs command — log-path resolution + tail invocation (subprocess patched).
# ---------------------------------------------------------------------------


def test_log_paths_from_installed_plist(monkeypatch, plist_kwargs, tmp_path):
    """Prefers the plist-declared StandardOut/StandardError paths."""
    import vllm_mlx.headless_service.logs as logs

    plist = tmp_path / "com.rapidmlx.server.plist"
    plist.write_bytes(serialize_plist(build_plist_dict(**plist_kwargs)))
    monkeypatch.setattr(ins_mod, "_plist_path", staticmethod(lambda _l: plist))
    out, err = logs._log_paths("com.rapidmlx.server", "serveuser")
    assert out == Path("/Users/serveuser/Library/Logs/Rapid-MLX/server.stdout.log")
    assert err == Path("/Users/serveuser/Library/Logs/Rapid-MLX/server.stderr.log")


def test_log_paths_fallback_to_service_default(monkeypatch, tmp_path):
    """No plist → fall back to the service account's default log dir."""
    import vllm_mlx.headless_service.logs as logs

    monkeypatch.setattr(
        ins_mod, "_plist_path", staticmethod(lambda _l: tmp_path / "missing.plist")
    )
    monkeypatch.setattr(
        logs,
        "log_dir_for",
        staticmethod(lambda u: Path(f"/Users/{u}/Library/Logs/Rapid-MLX")),
    )
    out, err = logs._log_paths("com.rapidmlx.server", "serveuser")
    assert out.name == "server.stdout.log"
    assert err.name == "server.stderr.log"


def test_log_paths_returns_none_when_unknown(monkeypatch, tmp_path):
    """No plist and no resolvable service default → None (→ hard error)."""
    import vllm_mlx.headless_service.logs as logs

    monkeypatch.setattr(
        ins_mod, "_plist_path", staticmethod(lambda _l: tmp_path / "missing.plist")
    )
    monkeypatch.setattr(logs, "log_dir_for", staticmethod(lambda u: None))
    assert logs._log_paths("com.rapidmlx.server", "serveuser") is None


def test_logs_no_paths_errors(monkeypatch, capsys, tmp_path):
    import vllm_mlx.headless_service.logs as logs

    monkeypatch.setattr(logs, "_log_paths", staticmethod(lambda *a: None))
    code = logs.logs_command(_ns())
    assert code == 1
    assert "no log paths known" in capsys.readouterr().err


def _patched_logged_files(monkeypatch, tmp_path):
    """Point _log_paths at two temp files; record subprocess.run calls."""
    import vllm_mlx.headless_service.logs as logs

    out_f = tmp_path / "out.log"
    err_f = tmp_path / "err.log"
    out_f.write_text("hello out\n")
    err_f.write_text("hello err\n")
    monkeypatch.setattr(
        logs,
        "_log_paths",
        staticmethod(lambda *a: (out_f, err_f)),
    )
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(argv)
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    return out_f, err_f, calls


def test_logs_tails_both_files(monkeypatch, tmp_path, capsys):
    import vllm_mlx.headless_service.logs as logs

    _, _, calls = _patched_logged_files(monkeypatch, tmp_path)
    code = logs.logs_command(_ns())
    assert code == 0
    # Two tail invocations: stdout then stderr.
    assert len(calls) == 2
    assert calls[0][1] == "-n"
    out = capsys.readouterr().out
    assert "=== stdout:" in out
    assert "=== stderr:" in out


def test_logs_skips_missing_log_file(monkeypatch, tmp_path, capsys):
    import vllm_mlx.headless_service.logs as logs

    out_f, err_f, calls = _patched_logged_files(monkeypatch, tmp_path)
    # Make stderr file absent → "not present" note, only one tail call.
    err_f.unlink()
    code = logs.logs_command(_ns())
    assert code == 0
    assert len(calls) == 1
    assert "(stderr: not present" in capsys.readouterr().out


def test_logs_tail_oserror(monkeypatch, tmp_path, capsys):
    import subprocess as sp

    import vllm_mlx.headless_service.logs as logs

    out_f, err_f, _ = _patched_logged_files(monkeypatch, tmp_path)

    def _boom(argv, **kwargs):
        raise OSError("perm denied")

    monkeypatch.setattr(sp, "run", _boom)
    code = logs.logs_command(_ns())
    assert code == 0
    assert "cannot read" in capsys.readouterr().err


def test_logs_follow_streams(monkeypatch, tmp_path, capsys):
    import vllm_mlx.headless_service.logs as logs

    _, _, calls = _patched_logged_files(monkeypatch, tmp_path)
    code = logs.logs_command(_ns(follow=True))
    assert code == 0
    # --follow runs `tail -F <stdout> <stderr>` once.
    assert len(calls) == 1
    assert calls[0][1] == "-F"
    assert len(calls[0]) == 4  # tail -F out err


# ---------------------------------------------------------------------------
# restart command — kickstart orchestration (subprocess/_wait_ready patched).
# ---------------------------------------------------------------------------


def test_declared_bind_parses_plist(monkeypatch, plist_kwargs, tmp_path):
    import vllm_mlx.headless_service.restart as rest

    plist = tmp_path / "com.rapidmlx.server.plist"
    plist.write_bytes(serialize_plist(build_plist_dict(**plist_kwargs)))
    monkeypatch.setattr(ins_mod, "_plist_path", staticmethod(lambda _l: plist))
    assert rest._declared_bind("com.rapidmlx.server") == ("127.0.0.1", 8000)


def test_declared_bind_none_when_missing_plist(monkeypatch, tmp_path):
    import vllm_mlx.headless_service.restart as rest

    monkeypatch.setattr(
        ins_mod, "_plist_path", staticmethod(lambda _l: tmp_path / "nope.plist")
    )
    assert rest._declared_bind("com.rapidmlx.server") is None


def test_declared_bind_none_on_bad_plist(monkeypatch, tmp_path):
    import vllm_mlx.headless_service.restart as rest

    plist = tmp_path / "com.rapidmlx.server.plist"
    plist.write_bytes(b"<not a plist")
    monkeypatch.setattr(ins_mod, "_plist_path", staticmethod(lambda _l: plist))
    assert rest._declared_bind("com.rapidmlx.server") is None


def test_restart_dry_run(monkeypatch, capsys):
    import vllm_mlx.headless_service.restart as rest

    monkeypatch.setattr(rest, "_declared_bind", staticmethod(lambda _l: None))
    monkeypatch.setattr(rest, "_kickstart_status", staticmethod(lambda _l: 0))
    code = rest.restart_command(_ns())
    assert code == 0
    assert "[DRY-RUN] would run: launchctl kickstart -k" in capsys.readouterr().out


def test_restart_not_registered_errors(monkeypatch, capsys):
    import vllm_mlx.headless_service.restart as rest

    monkeypatch.setattr(rest, "_declared_bind", staticmethod(lambda _l: None))
    monkeypatch.setattr(rest, "_kickstart_status", staticmethod(lambda _l: 1))
    code = rest.restart_command(_ns())
    assert code == 1
    assert "is not registered" in capsys.readouterr().err


def test_restart_happy_path(monkeypatch, capsys):
    import vllm_mlx.headless_service.restart as rest

    monkeypatch.setattr(
        rest, "_declared_bind", staticmethod(lambda _l: ("127.0.0.1", 8123))
    )
    monkeypatch.setattr(rest, "_kickstart_status", staticmethod(lambda _l: 0))
    monkeypatch.setattr(rest, "_wait_ready", staticmethod(lambda h, p, **k: True))
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(argv)
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    code = rest.restart_command(_ns(dry_run=False))
    assert code == 0
    assert "healthy on 127.0.0.1:8123" in capsys.readouterr().out
    assert calls[0][1] == "kickstart"


def test_restart_kickstart_failure(monkeypatch, capsys):
    import vllm_mlx.headless_service.restart as rest

    monkeypatch.setattr(rest, "_declared_bind", staticmethod(lambda _l: None))
    monkeypatch.setattr(rest, "_kickstart_status", staticmethod(lambda _l: 0))

    def _fake_run(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    code = rest.restart_command(_ns(dry_run=False))
    assert code == 1
    assert "kickstart failed" in capsys.readouterr().err


def test_restart_ready_failure(monkeypatch, capsys):
    import vllm_mlx.headless_service.restart as rest

    monkeypatch.setattr(rest, "_declared_bind", staticmethod(lambda _l: None))
    monkeypatch.setattr(rest, "_kickstart_status", staticmethod(lambda _l: 0))
    monkeypatch.setattr(rest, "_wait_ready", staticmethod(lambda h, p, **k: False))
    monkeypatch.setattr(subprocess, "run", staticmethod(lambda *a, **k: _FakeResult()))
    code = rest.restart_command(_ns(dry_run=False))
    assert code == 1
    assert "did not become ready" in capsys.readouterr().err


def test_kickstart_status_subprocess_error(monkeypatch):
    import vllm_mlx.headless_service.restart as rest

    def _boom(argv, **kwargs):
        raise OSError("no launchctl")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert rest._kickstart_status("com.rapidmlx.server") == 1


# ---------------------------------------------------------------------------
# status command — richer aggregation + human rendering (parsers exercised).
# ---------------------------------------------------------------------------


def _fake_launchctl_print(pid=None, last_exit=None, state=None, runs=None):
    lines = ["com.rapidmlx.server = {", "    active count = 1"]
    if pid is not None:
        lines.append(f"    pid = {pid}")
    if last_exit is not None:
        lines.append(f"    last exit code = {last_exit}")
    if state is not None:
        lines.append(f"    state = {state}")
    if runs is not None:
        lines.append(f"    runs = {runs}")
    lines.append("}")
    return "\n".join(lines)


def test_parse_pid_and_last_exit():
    import vllm_mlx.headless_service.status as st

    out = _fake_launchctl_print(pid=4242, last_exit=0)
    assert st._parse_pid(out) == 4242
    assert st._parse_last_exit(out) == 0
    assert st._parse_pid(None) is None
    assert st._parse_pid("no pid here") is None
    assert st._parse_last_exit(None) is None
    assert st._parse_last_exit(_fake_launchctl_print(last_exit=-9)) == -9
    details = _fake_launchctl_print(state="running", runs=3)
    assert st._parse_launchd_field(details, "state") == "running"
    assert st._parse_launchd_field(details, "runs") == "3"


def test_collect_status_full_branch(monkeypatch, plist_kwargs, tmp_path):
    """Registered + running + plist present + endpoint healthy."""
    import vllm_mlx.headless_service.status as st

    monkeypatch.setattr(
        st,
        "_launchctl_print",
        staticmethod(lambda _l: _fake_launchctl_print(pid=4242, last_exit=0)),
    )
    plist = tmp_path / "com.rapidmlx.server.plist"
    plist.write_bytes(serialize_plist(build_plist_dict(**plist_kwargs)))
    monkeypatch.setattr(st, "_plist_path", staticmethod(lambda _l: plist))
    monkeypatch.setattr(st, "_endpoint_health", staticmethod(lambda h, p: (True, True)))
    monkeypatch.setattr(st, "_port_busy", staticmethod(lambda h, p: True))
    # Stub `ps -o user=` so the owner lookup is hermetic.
    monkeypatch.setattr(
        subprocess,
        "run",
        staticmethod(
            lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="serveuser\n")
        ),
    )
    monkeypatch.setattr(
        st,
        "log_dir_for",
        staticmethod(lambda u: Path("/Users/u/Library/Logs/Rapid-MLX")),
    )
    data = st.collect_status(
        label="com.rapidmlx.server", user="serveuser", host="127.0.0.1", port=8000
    )
    assert data["registered"] is True
    assert data["pid"] == 4242
    assert data["last_exit"] == 0
    assert data["model"] == "qwen3.5-4b-4bit"
    assert data["owner"] == "serveuser"
    assert data["livez"] is True
    assert data["readyz"] is True
    assert data["port_open"] is True
    assert data["plist_present"] is True
    assert data["host"] == "127.0.0.1"
    assert data["port"] == 8000
    assert data["log_dir"]


def test_status_command_json_and_exit_codes(monkeypatch, capsys):
    import vllm_mlx.headless_service.status as st

    monkeypatch.setattr(
        st, "_launchctl_print", staticmethod(lambda _l: _fake_launchctl_print(pid=9))
    )
    monkeypatch.setattr(st, "_read_installed_plist", staticmethod(lambda _l: None))
    monkeypatch.setattr(st, "_endpoint_health", staticmethod(lambda h, p: (True, True)))
    monkeypatch.setattr(st, "_port_busy", staticmethod(lambda h, p: True))
    ns = _ns(json=True)
    assert st.status_command(ns) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["pid"] == 9

    # Ready but no PID → not "up" → exit 1.
    monkeypatch.setattr(
        st, "_launchctl_print", staticmethod(lambda _l: _fake_launchctl_print(pid=None))
    )
    assert st.status_command(_ns(json=True)) == 1


def test_render_human_lines():
    import vllm_mlx.headless_service.status as st

    s = {
        "label": "com.rapidmlx.server",
        "domain": "system",
        "registered": True,
        "pid": 42,
        "owner": "serveuser",
        "last_exit": 0,
        "model": "qwen3.5-4b-4bit",
        "host": "127.0.0.1",
        "port": 8000,
        "livez": True,
        "readyz": True,
        "port_open": True,
        "plist": "/Library/LaunchDaemons/com.rapidmlx.server.plist",
        "log_dir": "/Users/u/Library/Logs/Rapid-MLX",
        "plist_present": True,
    }
    text = st._render_human(s)
    assert "registered" in text
    assert "pid:                   42 (as serveuser)" in text
    assert "healthy" not in text

    # Down + no pid + hint.
    s2 = dict(
        s,
        pid=None,
        owner=None,
        registered=False,
        last_exit=None,
        model=None,
        port_open=False,
    )
    text2 = st._render_human(s2)
    assert "(no live process)" in text2
    assert "not registered" in text2
    assert "not registered — `rapid-mlx service install --dry-run`" in text2


# ---------------------------------------------------------------------------
# common helpers — account-DB fallthrough branches.
# ---------------------------------------------------------------------------


def test_common_home_for_user_missing(monkeypatch):
    import pwd

    class _NoPwd:
        def getpwnam(self, user):
            raise KeyError(user)

    monkeypatch.setattr(pwd, "getpwnam", _NoPwd().getpwnam)
    assert common.home_for_user("ghost") is None
    assert common.user_uid("ghost") is None


def test_common_log_dir_for_none_home(monkeypatch):
    monkeypatch.setattr(common, "home_for_user", staticmethod(lambda u: None))
    assert common.log_dir_for("ghost") is None


def test_is_admin_user_missing_group(monkeypatch):
    import grp

    def _raise(*a):
        raise KeyError("no group 80")

    monkeypatch.setattr(grp, "getgrgid", _raise)
    assert ins_mod.is_admin_user("serveuser") is False


def test_logs_follow_tail_failure(monkeypatch, tmp_path, capsys):
    import vllm_mlx.headless_service.logs as logs

    _, _, _ = _patched_logged_files(monkeypatch, tmp_path)

    def _boom(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", _boom)
    code = logs.logs_command(_ns(follow=True))
    assert code == 1
    assert "tail failed" in capsys.readouterr().err


def test_log_paths_ignores_corrupt_plist(monkeypatch, plist_kwargs, tmp_path):
    """A corrupt installed plist must not crash _log_paths (falls through)."""
    import vllm_mlx.headless_service.logs as logs

    plist = tmp_path / "com.rapidmlx.server.plist"
    plist.write_bytes(b"<not-a-plist")
    monkeypatch.setattr(ins_mod, "_plist_path", staticmethod(lambda _l: plist))
    monkeypatch.setattr(
        logs,
        "log_dir_for",
        staticmethod(lambda u: Path(f"/Users/{u}/Library/Logs/Rapid-MLX")),
    )
    out, err = logs._log_paths("com.rapidmlx.server", "serveuser")
    assert out.name == "server.stdout.log"


def test_declared_bind_none_when_no_port(monkeypatch, plist_kwargs, tmp_path):
    """A plist without a --port pair yields None (no reliable bind)."""
    import vllm_mlx.headless_service.restart as rest

    cfg = build_plist_dict(**plist_kwargs)
    cfg["ProgramArguments"] = ["/bin/rapid-mlx", "serve", "qwen3.5-4b-4bit"]
    plist = tmp_path / "com.rapidmlx.server.plist"
    plist.write_bytes(serialize_plist(cfg))
    monkeypatch.setattr(ins_mod, "_plist_path", staticmethod(lambda _l: plist))
    assert rest._declared_bind("com.rapidmlx.server") is None


def test_launchctl_print_returns_none_on_nonzero(monkeypatch):
    import vllm_mlx.headless_service.status as st

    monkeypatch.setattr(
        subprocess,
        "run",
        staticmethod(lambda *a, **k: types.SimpleNamespace(returncode=5, stdout="x")),
    )
    assert st._launchctl_print("com.rapidmlx.server") is None


def test_launchctl_print_returns_none_on_error(monkeypatch):
    import vllm_mlx.headless_service.status as st

    def _boom(argv, **kwargs):
        raise OSError("no launchctl")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert st._launchctl_print("com.rapidmlx.server") is None


def test_read_installed_plist_none_when_missing(monkeypatch, tmp_path):
    import vllm_mlx.headless_service.status as st

    monkeypatch.setattr(
        st, "_plist_path", staticmethod(lambda _l: tmp_path / "x.plist")
    )
    assert st._read_installed_plist("com.rapidmlx.server") is None


def test_read_installed_plist_none_when_corrupt(monkeypatch, tmp_path):
    import vllm_mlx.headless_service.status as st

    plist = tmp_path / "x.plist"
    plist.write_bytes(b"<not-a-plist")
    monkeypatch.setattr(st, "_plist_path", staticmethod(lambda _l: plist))
    assert st._read_installed_plist("com.rapidmlx.server") is None


def test_endpoint_health_connection_error(monkeypatch):
    import socket

    import vllm_mlx.headless_service.status as st

    def _boom(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(ins_mod, "_readyz_ready", staticmethod(lambda h, p: False))
    live, ready = st._endpoint_health("127.0.0.1", 1)
    assert live is False
    assert ready is False


def test_status_owner_ps_error(monkeypatch, plist_kwargs, tmp_path):
    """A failing `ps` owner lookup degrades to owner=None, not a crash."""
    import vllm_mlx.headless_service.status as st

    monkeypatch.setattr(
        st, "_launchctl_print", staticmethod(lambda _l: _fake_launchctl_print(pid=7))
    )
    monkeypatch.setattr(st, "_endpoint_health", staticmethod(lambda h, p: (True, True)))
    monkeypatch.setattr(st, "_port_busy", staticmethod(lambda h, p: False))
    plist = tmp_path / "com.rapidmlx.server.plist"
    plist.write_bytes(serialize_plist(build_plist_dict(**plist_kwargs)))
    monkeypatch.setattr(st, "_plist_path", staticmethod(lambda _l: plist))

    def _boom(argv, **kwargs):
        raise OSError("ps failed")

    monkeypatch.setattr(subprocess, "run", _boom)
    data = st.collect_status(label="com.rapidmlx.server")
    assert data["pid"] == 7
    assert data["owner"] is None


def test_status_command_human_branch(monkeypatch, capsys):
    """Non-json status prints the human table (and returns 1 when down)."""
    import vllm_mlx.headless_service.status as st

    monkeypatch.setattr(
        st, "_launchctl_print", staticmethod(lambda _l: _fake_launchctl_print())
    )
    monkeypatch.setattr(st, "_read_installed_plist", staticmethod(lambda _l: None))
    monkeypatch.setattr(
        st, "_endpoint_health", staticmethod(lambda h, p: (False, False))
    )
    monkeypatch.setattr(st, "_port_busy", staticmethod(lambda h, p: False))
    code = st.status_command(_ns(json=False))
    assert code == 1
    text = capsys.readouterr().out
    assert "launcher registration:" in text
    # Down state surfaces in the human table.
    assert "(no live process)" in text
    assert "livez=down readyz=down port=closed" in text


def test_resolve_executable_success(monkeypatch, tmp_path):
    """A real executable under the service home resolves to that path."""
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    exe = bin_dir / "rapid-mlx"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    resolved = ins_mod.resolve_executable(Path(tmp_path))
    assert resolved == str(exe)


# ---------------------------------------------------------------------------
# install.py remaining helpers — cache, build, port-busy, wait, uninstall.
# ---------------------------------------------------------------------------


def test_cache_root_present_branches(tmp_path):
    home = tmp_path / "serveuser"
    cache = home / ".cache" / "huggingface" / "hub"
    assert ins_mod._cache_root_present(home, "org/model") is False
    cache.mkdir(parents=True)
    assert ins_mod._cache_root_present(home, "org/model") is False
    (cache / "models--org--model").mkdir()
    assert ins_mod._cache_root_present(home, "org/model") is True


def test_build_plist_bytes_requires_resolvable_home(monkeypatch):
    monkeypatch.setattr(common, "log_dir_for", staticmethod(lambda u: None))
    with pytest.raises(ins_mod.ServiceInstallError, match="cannot resolve home"):
        ins_mod._build_plist_bytes(
            label="com.rapidmlx.server",
            user="serveuser",
            executable="/bin/rapid-mlx",
            model="qwen3.5-4b-4bit",
            home=Path("/Users/serveuser"),
            host="127.0.0.1",
            port=8000,
            serve_args=(),
        )


def test_port_busy_true_when_answering(monkeypatch):
    import socket

    class _Sock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        socket, "create_connection", staticmethod(lambda *a, **k: _Sock())
    )
    assert ins_mod._port_busy("127.0.0.1", 8000) is True


def test_port_busy_false_on_connect_error(monkeypatch):
    import socket

    def _boom(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", _boom)
    assert ins_mod._port_busy("127.0.0.1", 1) is False


def test_wait_ready_immediate(monkeypatch):
    monkeypatch.setattr(ins_mod, "_readyz_ready", staticmethod(lambda h, p: True))
    assert ins_mod._wait_ready("127.0.0.1", 1, timeout_s=5) is True


def test_wait_ready_timeout(monkeypatch):
    monkeypatch.setattr(ins_mod, "_readyz_ready", staticmethod(lambda h, p: False))
    state = {"t": 0.0}

    def _tick():
        state["t"] += 1.0
        return state["t"]

    monkeypatch.setattr(
        ins_mod, "time", types.SimpleNamespace(time=_tick, sleep=lambda s: None)
    )
    # time() advances past the (5s) deadline → loop ends → False.
    assert ins_mod._wait_ready("127.0.0.1", 1, timeout_s=5) is False


def test_uninstall_non_root_errors(monkeypatch, capsys):
    monkeypatch.setattr(ins_mod, "is_root", staticmethod(lambda: False))
    code = ins_mod.uninstall_command(_ns(dry_run=False))
    assert code == 1
    assert "requires root" in capsys.readouterr().err


def test_uninstall_real_removes(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ins_mod, "is_root", staticmethod(lambda: True))
    monkeypatch.setattr(
        ins_mod, "_plist_path", staticmethod(lambda _l: tmp_path / "x.plist")
    )
    monkeypatch.setattr(subprocess, "run", staticmethod(lambda *a, **k: _FakeResult()))
    code = ins_mod.uninstall_command(_ns(dry_run=False))
    assert code == 0
    assert "uninstalled com.rapidmlx.server" in capsys.readouterr().out


def test_uninstall_subprocess_failure(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ins_mod, "is_root", staticmethod(lambda: True))
    monkeypatch.setattr(
        ins_mod, "_plist_path", staticmethod(lambda _l: tmp_path / "x.plist")
    )

    def _boom(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", _boom)
    code = ins_mod.uninstall_command(_ns(dry_run=False))
    assert code == 2
    assert "uninstall failed" in capsys.readouterr().err


def test_install_non_root_errors(monkeypatch, capsys):
    """Real (non-dry-run) install without root refuses cleanly."""
    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins_mod, "_port_busy", staticmethod(lambda h, p: False))
    monkeypatch.setattr(ins_mod, "is_root", staticmethod(lambda: False))
    code = ins_mod.install_command(_ns(dry_run=False))
    assert code == 1
    assert "requires root" in capsys.readouterr().err


def test_is_root_direct():
    # Non-root on CI hosts just exercises the comparison expression.
    assert ins_mod.is_root() in (True, False)


def test_is_admin_user_membership_false(monkeypatch):
    import grp

    def _group(*a):
        return types.SimpleNamespace(gr_mem=["alice", "bob"])

    monkeypatch.setattr(grp, "getgrgid", _group)
    assert ins_mod.is_admin_user("serveuser") is False


def test_is_admin_user_checks_effective_group_list(monkeypatch):
    import grp
    import pwd

    monkeypatch.setattr(
        grp, "getgrgid", lambda _gid: types.SimpleNamespace(gr_gid=80, gr_mem=[])
    )
    monkeypatch.setattr(pwd, "getpwnam", lambda _u: types.SimpleNamespace(pw_gid=20))
    monkeypatch.setattr(ins_mod.os, "getgrouplist", lambda _u, _gid: [20, 80])
    assert ins_mod.is_admin_user("serveuser") is True


def test_is_admin_user_group_lookup_fallback(monkeypatch):
    import grp
    import pwd

    monkeypatch.setattr(
        grp, "getgrgid", lambda _gid: types.SimpleNamespace(gr_gid=80, gr_mem=[])
    )
    monkeypatch.setattr(pwd, "getpwnam", lambda _u: types.SimpleNamespace(pw_gid=80))
    monkeypatch.setattr(
        ins_mod.os,
        "getgrouplist",
        lambda *_args: (_ for _ in ()).throw(OSError("directory unavailable")),
    )
    assert ins_mod.is_admin_user("serveuser") is True


def test_validate_service_account_empty_user():
    with pytest.raises(ins_mod.ServiceInstallError, match="required"):
        ins_mod.validate_service_account("")


def test_run_forwards_to_subprocess(monkeypatch):
    monkeypatch.setattr(subprocess, "run", staticmethod(lambda *a, **k: "ran"))
    assert ins_mod._run(["echo", "hi"]) == "ran"


def test_kickstart_status_success(monkeypatch):
    import vllm_mlx.headless_service.restart as rest

    monkeypatch.setattr(
        subprocess,
        "run",
        staticmethod(lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="")),
    )
    assert rest._kickstart_status("com.rapidmlx.server") == 0


def test_launchctl_print_success_returns_stdout(monkeypatch):
    import vllm_mlx.headless_service.status as st

    monkeypatch.setattr(
        subprocess,
        "run",
        staticmethod(
            lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="pid = 5")
        ),
    )
    assert st._launchctl_print("com.rapidmlx.server") == "pid = 5"


def test_endpoint_health_live_200(monkeypatch):
    """_probe_live reads a 200 from /livez (socket sendall/recv path)."""
    import vllm_mlx.headless_service.status as st

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def settimeout(self, *a):
            pass

        def sendall(self, *a):
            pass

        def recv(self, n):
            return b"HTTP/1.1 200 OK\r\n\r\n"

    monkeypatch.setattr(
        socket, "create_connection", staticmethod(lambda *a, **k: _Conn())
    )
    monkeypatch.setattr(ins_mod, "_readyz_ready", staticmethod(lambda h, p: True))
    live, ready = st._endpoint_health("127.0.0.1", 8000)
    assert live is True
    assert ready is True


def test_endpoint_health_probes_wildcard_bind_via_loopback(monkeypatch):
    import vllm_mlx.headless_service.status as st

    seen = []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def settimeout(self, *a):
            pass

        def sendall(self, *a):
            pass

        def recv(self, n):
            return b"HTTP/1.1 200 OK\r\n\r\n"

    def _connect(address, **_kwargs):
        seen.append(address)
        return _Conn()

    monkeypatch.setattr(socket, "create_connection", _connect)
    monkeypatch.setattr(
        ins_mod,
        "_readyz_ready",
        lambda host, port: seen.append((host, port)) or True,
    )
    assert st._endpoint_health("0.0.0.0", 8000) == (True, True)
    assert seen == [("127.0.0.1", 8000), ("127.0.0.1", 8000)]


def test_logs_follow_keyboard_interrupt(monkeypatch, tmp_path, capsys):
    import vllm_mlx.headless_service.logs as logs

    _, _, _ = _patched_logged_files(monkeypatch, tmp_path)

    def _kbi(argv, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(subprocess, "run", _kbi)
    assert logs.logs_command(_ns(follow=True)) == 0


def test_readyz_ready_connection_error():
    """A refused /readyz probe is NOT ready (OSError branch)."""
    from vllm_mlx.headless_service.install import _readyz_ready

    # Port 1 on loopback is never listening → connection refused → False.
    assert _readyz_ready("127.0.0.1", 1) is False


@pytest.mark.parametrize(
    ("bind", "probe"),
    [("0.0.0.0", "127.0.0.1"), ("::", "::1"), ("[::]", "::1")],
)
def test_wildcard_bind_uses_connectable_probe_address(bind, probe):
    from vllm_mlx.headless_service.install import _probe_host

    assert _probe_host(bind) == probe


def test_install_plutil_lint_failure_rolls_back(monkeypatch, tmp_path, capsys):
    """A plutil -lint failure aborts the install cleanly (rollback)."""
    ins = ins_mod
    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "_port_busy", staticmethod(lambda h, p: False))
    monkeypatch.setattr(ins, "is_root", staticmethod(lambda: True))
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)

    def _fake_run(args, check=True):
        if args[0] == "plutil":

            class _Bad(_FakeResult):
                returncode = 1
                stderr = "invalid property list"

            return _Bad()
        return _FakeResult()

    monkeypatch.setattr(ins, "_run", _fake_run)
    code = ins.install_command(_ns(dry_run=False))
    assert code == 2
    assert "plutil -lint failed" in capsys.readouterr().err


def test_install_bootstrap_hard_failure(monkeypatch, tmp_path, capsys):
    """A bootstrap failure that is NOT 'already loaded' is a hard error."""
    ins = ins_mod
    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "_port_busy", staticmethod(lambda h, p: False))
    monkeypatch.setattr(ins, "is_root", staticmethod(lambda: True))
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)

    def _fake_run(args, check=True):
        if args[0] == "plutil":
            return _FakeResult()
        if args[0] == "install":
            return _FakeResult()
        if args[0] == "launchctl" and args[1] == "bootstrap":

            class _Bad(_FakeResult):
                returncode = 8
                stderr = "Bootstrap failed: 5: Input/output error"

            return _Bad()
        return _FakeResult()

    monkeypatch.setattr(ins, "_run", _fake_run)
    code = ins.install_command(_ns(dry_run=False))
    assert code == 2
    assert "bootstrap failed" in capsys.readouterr().err


def test_install_generic_step_failure_rolls_back(monkeypatch, tmp_path, capsys):
    """A non-ServiceInstallError during install prints info + returns 2."""
    ins = ins_mod
    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "_port_busy", staticmethod(lambda h, p: False))
    monkeypatch.setattr(ins, "is_root", staticmethod(lambda: True))
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)
    monkeypatch.setattr(ins, "_run", staticmethod(lambda *a, **k: _FakeResult()))

    # _wait_ready raises a GENERIC exception (not ServiceInstallError) →
    # the generic rollback branch (info: removed + install step failed).
    def _boom(h, p, **k):
        raise RuntimeError("readiness probe crashed")

    monkeypatch.setattr(ins, "_wait_ready", _boom)
    code = ins.install_command(_ns(dry_run=False))
    assert code == 2
    err = capsys.readouterr().err
    assert "info: removed" in err
    assert "install step failed" in err


def test_install_rollback_unlink_oserror(monkeypatch, tmp_path, capsys):
    """staged.unlink() failing in rollback is tolerated (except OSError)."""
    from pathlib import Path as _Path

    ins = ins_mod
    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "_port_busy", staticmethod(lambda h, p: False))
    monkeypatch.setattr(ins, "is_root", staticmethod(lambda: True))
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)
    monkeypatch.setattr(ins, "_run", staticmethod(lambda *a, **k: _FakeResult()))
    monkeypatch.setattr(ins, "_wait_ready", staticmethod(lambda h, p, **k: False))

    real_unlink = _Path.unlink

    def _unlink_boom(self, *a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(_Path, "unlink", _unlink_boom)
    try:
        code = ins.install_command(_ns(dry_run=False))
    finally:
        monkeypatch.setattr(_Path, "unlink", real_unlink)
    assert code == 2
    assert "rolled back" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Reviewer fixes: rollback cleanliness, reinstall tolerance, exec requirement.
# ---------------------------------------------------------------------------


def test_install_real_path_rejects_already_loaded(monkeypatch, tmp_path, capsys):
    """A hidden loaded job must not make a new plist look successfully active."""
    import vllm_mlx.headless_service.install as ins

    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "_port_busy", staticmethod(lambda h, p: False))
    monkeypatch.setattr(ins, "is_root", staticmethod(lambda: True))
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)
    monkeypatch.setattr(ins, "_wait_ready", staticmethod(lambda h, p, **k: True))

    def _fake_run(args, check=True):
        if args and args[0] == "launchctl" and args[1] == "bootstrap":
            # launchd exits non-zero with "already loaded" on reinstall.
            class _Already(_FakeResult):
                returncode = 5
                stderr = "service already loaded"

            return _Already()
        if args and args[0] == "rm":
            import os

            for path in args[2:]:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        return _FakeResult()

    monkeypatch.setattr(ins, "_run", _fake_run)
    code = ins.install_command(_ns(dry_run=False))
    assert code == 2
    assert "bootstrap failed" in capsys.readouterr().err


def test_install_refuses_existing_plist_before_mutation(monkeypatch, tmp_path, capsys):
    """Replacing a plist while launchd retains old argv creates split-brain."""
    import vllm_mlx.headless_service.install as ins

    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)
    (tmp_path / "com.rapidmlx.server.plist").write_text("existing")
    code = ins.install_command(_ns(dry_run=False))
    assert code == 1
    assert "already exists" in capsys.readouterr().err


def test_stage_plist_uses_private_unpredictable_file(tmp_path, monkeypatch):
    import stat

    import vllm_mlx.headless_service.install as ins

    monkeypatch.setattr(ins.tempfile, "tempdir", str(tmp_path))
    first = ins._stage_plist(b"one")
    second = ins._stage_plist(b"two")
    try:
        assert first != second
        assert first.read_bytes() == b"one"
        assert stat.S_IMODE(first.stat().st_mode) == 0o600
    finally:
        first.unlink()
        second.unlink()


@pytest.mark.parametrize("unlink_fails", [False, True])
def test_stage_plist_write_failure_closes_and_cleans(
    tmp_path, monkeypatch, unlink_fails
):
    import os

    import vllm_mlx.headless_service.install as ins

    raw_path = tmp_path / "staged.plist"
    fd = os.open(raw_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    monkeypatch.setattr(ins.tempfile, "mkstemp", lambda **_kwargs: (fd, str(raw_path)))

    class _BadHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            os.close(fd)

        def write(self, _data):
            raise RuntimeError("write failed")

    monkeypatch.setattr(ins.os, "fdopen", lambda *_args, **_kwargs: _BadHandle())
    real_unlink = ins.os.unlink
    if unlink_fails:
        monkeypatch.setattr(
            ins.os, "unlink", lambda _path: (_ for _ in ()).throw(OSError("busy"))
        )
    with pytest.raises(RuntimeError, match="write failed"):
        ins._stage_plist(b"data")
    if unlink_fails:
        monkeypatch.setattr(ins.os, "unlink", real_unlink)
        raw_path.unlink()
    else:
        assert not raw_path.exists()


def test_install_success_cleans_secure_staging_file(monkeypatch, tmp_path, capsys):
    import shutil

    import vllm_mlx.headless_service.install as ins

    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "_port_busy", lambda _h, _p: False)
    monkeypatch.setattr(ins, "is_root", lambda: True)
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)
    monkeypatch.setattr(ins, "_wait_ready", lambda _h, _p, **_k: True)
    monkeypatch.setattr(ins.tempfile, "tempdir", str(tmp_path))

    def _fake_run(argv, check=True):
        if argv[0] == "install":
            shutil.copyfile(argv[-2], argv[-1])
        return _FakeResult()

    monkeypatch.setattr(ins, "_run", _fake_run)
    code = ins.install_command(
        _ns(dry_run=False, serve_args=["--", "--max-num-seqs", "4"])
    )
    assert code == 0
    assert "Installed and running" in capsys.readouterr().out
    assert not list(tmp_path.glob("rapid-mlx-service-*.plist"))


def test_install_success_tolerates_staging_cleanup_failure(monkeypatch, tmp_path):
    import vllm_mlx.headless_service.install as ins

    _valid_user_monkeypatch(monkeypatch)
    monkeypatch.setattr(ins, "_port_busy", lambda _h, _p: False)
    monkeypatch.setattr(ins, "is_root", lambda: True)
    monkeypatch.setattr(ins, "LAUNCH_DAEMONS_DIR", tmp_path)
    monkeypatch.setattr(ins, "_wait_ready", lambda _h, _p, **_k: True)
    monkeypatch.setattr(ins, "_run", lambda *_a, **_k: _FakeResult())
    staged = tmp_path / "private-stage.plist"
    staged.write_bytes(b"plist")
    monkeypatch.setattr(ins, "_stage_plist", lambda _buf: staged)
    real_unlink = Path.unlink

    def _fail_only_for_stage(self, *args, **kwargs):
        if self == staged:
            raise OSError("busy")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _fail_only_for_stage)
    assert ins.install_command(_ns(dry_run=False)) == 0
    assert staged.exists()
    real_unlink(staged)
