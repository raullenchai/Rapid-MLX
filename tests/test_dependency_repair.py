# SPDX-License-Identifier: Apache-2.0
"""No-model simulations of broken installs and the recovery consent boundary."""

import io
import json
import subprocess
import sys
import types
from unittest.mock import Mock

import pytest

from rapid_mlx import _upgrade_runner as upgrade_runner
from rapid_mlx import _version_check as version_check
from rapid_mlx import dependency_repair as repair
from rapid_mlx.runtime import optional_runtime


@pytest.fixture
def environment(monkeypatch):
    # Existing unit tests stub subprocess.run; keep pip installs isolated too.
    monkeypatch.setattr(
        upgrade_runner,
        "_run_filtered",
        lambda argv: subprocess.run(argv, check=False).returncode,
    )
    monkeypatch.setenv("RAPID_MLX_DEPENDENCY_REPAIR_RETRIED", "")
    monkeypatch.delenv("RAPID_MLX_DEPENDENCY_REPAIR_RETRIED", raising=False)
    monkeypatch.setattr(
        repair.metadata, "requires", lambda _: ["tiktoken>=0.5", "pyyaml>=6"]
    )
    monkeypatch.setattr(optional_runtime, "_running_in_desktop_sidecar", lambda: False)
    monkeypatch.setattr(
        version_check,
        "detect_install_method",
        lambda: version_check.InstallInfo(
            "pip",
            "python -m pip install --upgrade rapid-mlx",
            [sys.executable, "-m", "pip", "install", "--upgrade", "rapid-mlx"],
        ),
    )
    outcomes = []
    monkeypatch.setattr(repair, "record_outcome", outcomes.append)
    monkeypatch.delenv(repair._RETRY, raising=False)
    return outcomes


def stdin(monkeypatch, answer="y\n", tty=True):
    stream = io.StringIO(answer)
    stream.isatty = lambda: tty
    monkeypatch.setattr(sys, "stdin", stream)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("uv", ["uv", "tool", "upgrade", "--reinstall", "rapid-mlx"]),
        ("pipx", ["pipx", "reinstall", "rapid-mlx"]),
        ("brew", ["brew", "reinstall", "rapid-mlx"]),
        ("pip", None),
        ("install_sh", None),
        ("unknown", None),
    ],
)
def test_missing_module_prints_one_method_command(
    environment, monkeypatch, capsys, method, expected
):
    monkeypatch.setattr(
        version_check,
        "detect_install_method",
        lambda: version_check.InstallInfo(
            method,
            "",
            [sys.executable, "-m", "pip", "install", "--upgrade", "rapid-mlx"],
        ),
    )
    stdin(monkeypatch, tty=False)
    fake = types.ModuleType("rapid_mlx.cli")
    fake.cli_entrypoint = Mock(
        side_effect=ModuleNotFoundError("missing", name="tiktoken")
    )
    monkeypatch.setitem(sys.modules, "rapid_mlx.cli", fake)
    with pytest.raises(SystemExit) as error:
        repair.cli_entrypoint()
    assert error.value.code == 1
    assert environment == ["command_shown"]
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    import shlex

    command = shlex.split(lines[1])
    if expected:
        assert command == expected
    else:
        assert command[:3] == [sys.executable, "-m", "pip"]
        assert "--force-reinstall" in command


def test_desktop_never_runs_installer(environment, monkeypatch, capsys):
    monkeypatch.setattr(optional_runtime, "_running_in_desktop_sidecar", lambda: True)
    stdin(monkeypatch)
    monkeypatch.setattr(
        repair.subprocess, "run", Mock(side_effect=AssertionError("mutation"))
    )
    assert not repair.recover()
    assert "reinstall Rapid-MLX Desktop" in capsys.readouterr().err
    assert environment == ["command_shown"]


@pytest.mark.parametrize("answer", ["\n", "n\n", "yes\n"])
def test_prompt_requires_single_y(environment, monkeypatch, answer):
    stdin(monkeypatch, answer)
    assert not repair.recover()
    assert environment == ["declined"]


def test_eof_prompt_is_no_answer(environment, monkeypatch):
    stdin(monkeypatch, "")
    assert not repair.recover()
    assert environment == ["no_answer"]


def test_interrupted_prompt_preserves_ctrl_c(environment, monkeypatch):
    stdin(monkeypatch)
    monkeypatch.setattr(sys.stdin, "readline", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        repair.recover()
    assert environment == ["interrupted"]


@pytest.mark.parametrize("failure", [OSError(), subprocess.CompletedProcess([], 1)])
def test_install_failure(environment, monkeypatch, failure):
    stdin(monkeypatch)
    run = (
        Mock(side_effect=failure)
        if isinstance(failure, Exception)
        else Mock(return_value=failure)
    )
    monkeypatch.setattr(repair.subprocess, "run", run)
    assert not repair.recover(module="tiktoken")
    assert environment == ["failed"]


@pytest.mark.parametrize(
    "verify",
    [
        subprocess.CompletedProcess([], 0),
        subprocess.CompletedProcess([], 1),
        subprocess.TimeoutExpired([], 60),
        OSError(),
    ],
)
def test_installer_success_requires_fresh_import(environment, monkeypatch, verify):
    stdin(monkeypatch)
    run = Mock(side_effect=[subprocess.CompletedProcess([], 0), verify])
    monkeypatch.setattr(repair.subprocess, "run", run)
    success = isinstance(verify, subprocess.CompletedProcess) and verify.returncode == 0
    assert repair.recover(module="tiktoken") is success
    assert environment == ["repaired" if success else "failed"]
    assert run.call_args.args[0][-1] == "import tiktoken"


def test_missing_symbol_verification_reproduces_from_import(environment, monkeypatch):
    stdin(monkeypatch)
    verify = subprocess.CompletedProcess([], 1)
    run = Mock(side_effect=[subprocess.CompletedProcess([], 0), verify])
    monkeypatch.setattr(repair.subprocess, "run", run)
    assert not repair.recover(module="package", symbol="removed_symbol")
    assert run.call_args.args[0] == [
        sys.executable,
        "-c",
        "from package import removed_symbol",
    ]
    assert environment == ["failed"]

    stdin(monkeypatch)
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
        ]
    )
    monkeypatch.setattr(repair.subprocess, "run", run)
    assert repair.recover(module="package", symbol="removed_symbol")
    assert environment == ["failed", "repaired"]


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            ImportError("cannot import name 'removed_symbol' from 'package'"),
            "removed_symbol",
        ),
        (ModuleNotFoundError("No module named 'package'", name="package"), None),
        (ImportError("broken", name="tiktoken"), None),
    ],
)
def test_missing_symbol_extraction(error, expected):
    assert repair.missing_symbol(error) == expected


def test_cli_entrypoint_verifies_missing_symbol(environment, monkeypatch):
    fake = types.ModuleType("rapid_mlx.cli")
    error = ImportError("cannot import name 'removed_symbol' from 'tiktoken'")
    error.name = "tiktoken"
    fake.cli_entrypoint = Mock(side_effect=error)
    monkeypatch.setitem(sys.modules, "rapid_mlx.cli", fake)
    calls = []
    monkeypatch.setattr(
        repair, "recover", lambda **kwargs: calls.append(kwargs) or False
    )
    with pytest.raises(SystemExit):
        repair.cli_entrypoint()
    assert calls == [
        {"allow_run": True, "module": "tiktoken", "symbol": "removed_symbol"}
    ]


def test_reexec_once(environment, monkeypatch):
    stdin(monkeypatch)
    fake = types.ModuleType("rapid_mlx.cli")
    fake.cli_entrypoint = Mock(side_effect=ImportError("broken", name="tiktoken"))
    monkeypatch.setitem(sys.modules, "rapid_mlx.cli", fake)
    monkeypatch.setattr(
        repair.subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0))
    )
    execute = Mock(side_effect=SystemExit(17))
    monkeypatch.setattr(repair.os, "execv", execute)
    from rapid_mlx.telemetry import posthog_sender

    monkeypatch.setattr(posthog_sender, "get_sender", lambda: Mock())
    with pytest.raises(SystemExit, match="17"):
        repair.cli_entrypoint()
    assert execute.call_args.args == (
        sys.executable,
        [sys.executable, *sys.orig_argv[1:]],
    )
    assert environment == ["repaired"]
    environment.clear()
    with pytest.raises(SystemExit, match="1"):
        repair.cli_entrypoint()
    assert execute.call_count == 1
    assert environment[0] == "failed"


def test_exec_failure(environment, monkeypatch):
    fake = types.ModuleType("rapid_mlx.cli")
    fake.cli_entrypoint = Mock(side_effect=ImportError(name="tiktoken"))
    monkeypatch.setitem(sys.modules, "rapid_mlx.cli", fake)
    monkeypatch.setattr(repair, "recover", lambda **_: True)
    monkeypatch.setattr(repair.os, "execv", Mock(side_effect=OSError()))
    from rapid_mlx.telemetry import posthog_sender

    monkeypatch.setattr(posthog_sender, "get_sender", lambda: Mock())
    with pytest.raises(SystemExit, match="1"):
        repair.cli_entrypoint()
    assert environment == ["failed"]


def test_unrelated_import_error_is_not_repaired(environment, monkeypatch):
    fake = types.ModuleType("rapid_mlx.cli")
    fake.cli_entrypoint = Mock(side_effect=ImportError(name="custom_plugin"))
    monkeypatch.setitem(sys.modules, "rapid_mlx.cli", fake)
    with pytest.raises(ImportError):
        repair.cli_entrypoint()
    assert not repair.dependency_failure(
        ModuleNotFoundError(name="mlx_lm.models.unknown")
    )
    assert environment == []


def test_declared_names_and_platform_markers(monkeypatch):
    monkeypatch.setattr(
        repair.metadata,
        "requires",
        lambda _: [
            "pyyaml>=6",
            "pillow",
            "mlx-video-with-audio",
            "foo; extra == 'audio'",
            "tomli; python_version < '0'",
            "numpy; python_version >= '3'",
            "",
        ],
    )
    assert repair.declared_modules(active_only=True) == {
        "yaml",
        "PIL",
        "mlx_video",
        "numpy",
    }
    assert "tomli" in repair.declared_modules()


def test_missing_modules_respects_brew_omissions(environment, monkeypatch):
    monkeypatch.setattr(
        repair.metadata, "requires", lambda _: ["mlx-vlm", "mflux", "tiktoken"]
    )
    monkeypatch.setattr(
        version_check,
        "detect_install_method",
        lambda: version_check.InstallInfo("brew", "", []),
    )
    monkeypatch.setattr(repair.importlib.util, "find_spec", lambda _: None)
    assert repair.missing_modules() == ["tiktoken"]


def test_pip_fallback_and_global_pipx(environment, monkeypatch):
    monkeypatch.setattr(
        version_check,
        "detect_install_method",
        lambda: version_check.InstallInfo("unknown", "", []),
    )
    assert repair.repair_argv()[:3] == [sys.executable, "-m", "pip"]
    monkeypatch.setattr(
        version_check,
        "detect_install_method",
        lambda: version_check.InstallInfo(
            "unknown", "sudo pipx upgrade --global rapid-mlx", []
        ),
    )
    stdin(monkeypatch)
    assert repair.repair_argv() == [
        "sudo",
        "pipx",
        "reinstall",
        "--global",
        "rapid-mlx",
    ]
    assert not repair.recover()


def test_install_sh_uv_pip_preserved(environment, monkeypatch):
    monkeypatch.setattr(
        version_check,
        "detect_install_method",
        lambda: version_check.InstallInfo(
            "install_sh",
            "",
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--upgrade",
                "rapid-mlx",
            ],
        ),
    )
    assert repair.repair_argv()[:5] == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
    ]


def test_telemetry_closed_and_opt_out(monkeypatch):
    from rapid_mlx.telemetry import registry, track

    emit = Mock()
    monkeypatch.setattr(track, "track", emit)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "--no-telemetry"])
    repair.record_outcome("repaired")
    emit.assert_not_called()
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve"])
    for outcome in sorted(repair._OUTCOMES):
        repair.record_outcome(outcome)
        event, props = emit.call_args.args
        registry.validate(event, props)
        assert set(props) == {
            "error_class",
            "failure_subtype",
            "extra_recovery",
            "failure_stage",
        }
    repair.record_outcome("arbitrary")
    assert emit.call_count == len(repair._OUTCOMES)
    emit.side_effect = ImportError()
    repair.record_outcome("failed")


def test_recover_without_verifier(environment, monkeypatch):
    stdin(monkeypatch)
    monkeypatch.setattr(
        repair.subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0))
    )
    assert repair.recover()


def test_doctor_plans_and_verifies_dependency_repair(environment, monkeypatch):
    from rapid_mlx.doctor.env_health import Check, CheckStatus, Report, Section
    from rapid_mlx.doctor.repairs import apply_repairs, plan_repairs

    report = Report(
        sections=[
            Section(
                "Required",
                [
                    Check(
                        "missing", CheckStatus.FAIL, id="packages.required.dependencies"
                    )
                ],
            )
        ]
    )
    actions = plan_repairs(report)
    assert len(actions) == 1 and actions[0].id == "repair.dependencies"
    assert apply_repairs(actions, dry_run=True)[0].status == "planned"
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    assert apply_repairs(actions, run=run)[0].status == "verified"
    assert run.call_count == 2
    assert environment == ["repaired"]
    for result in (subprocess.CompletedProcess([], 1, "", ""), OSError()):
        run = (
            Mock(side_effect=result)
            if isinstance(result, Exception)
            else Mock(return_value=result)
        )
        assert apply_repairs(actions, run=run)[0].status == "failed"
    # Installer succeeds but the fresh-process verifier still fails.
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", ""),
        ]
    )
    assert apply_repairs(actions, run=run)[0].status == "failed"
    monkeypatch.setattr(optional_runtime, "_running_in_desktop_sidecar", lambda: True)
    assert plan_repairs(report) == []


def test_doctor_plans_dependency_then_service_repair(environment):
    from rapid_mlx.doctor.env_health import Check, CheckStatus, Report, Section
    from rapid_mlx.doctor.repairs import plan_repairs

    report = Report(
        sections=[
            Section(
                "Required",
                [
                    Check(
                        "missing", CheckStatus.FAIL, id="packages.required.dependencies"
                    )
                ],
            ),
            Section(
                "Always-on Service",
                [
                    Check("registered", CheckStatus.OK, id="service.registration"),
                    Check("process", CheckStatus.FAIL, id="service.process"),
                ],
            ),
        ]
    )
    assert [action.id for action in plan_repairs(report)] == [
        "repair.dependencies",
        "repair.service.kickstart",
    ]


def test_doctor_required_probe_uses_shared_repair(environment, monkeypatch):
    from rapid_mlx.doctor import env_health as eh

    monkeypatch.setattr(eh, "REQUIRED_PACKAGES", [])
    monkeypatch.setattr(eh, "_runtime_uses_context", lambda _: False)
    monkeypatch.setattr(repair, "missing_modules", lambda: ["tiktoken"])
    section = eh.section_required_packages()
    assert section.checks[0].id == "packages.required.dependencies"
    assert "--force-reinstall" in section.checks[0].label
    monkeypatch.setattr(optional_runtime, "_running_in_desktop_sidecar", lambda: True)
    assert (
        "reinstall Rapid-MLX Desktop" in eh.section_required_packages().checks[0].label
    )


def test_doctor_decline_records_outcome(environment, monkeypatch):
    from argparse import Namespace

    from rapid_mlx.doctor import cli
    from rapid_mlx.doctor.env_health import Check, CheckStatus, Report, Section

    report = Report(
        sections=[
            Section(
                "Required",
                [
                    Check(
                        "missing", CheckStatus.FAIL, id="packages.required.dependencies"
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(cli, "run_all", lambda **_: report)
    stdin(monkeypatch, "n\n")
    with pytest.raises(SystemExit):
        cli.doctor_command(Namespace(fix=True))
    assert environment == ["declined"]


def test_damaged_venv_bootstraps_missing_pip(environment, monkeypatch):
    monkeypatch.setattr(repair.importlib.util, "find_spec", lambda _: None)
    argv = repair.repair_argv()
    assert argv[:2] == [sys.executable, "-c"]
    assert "ensurepip.bootstrap" in argv[2]
    assert "sys.executable" in argv[2]


def test_retry_marker_consumed_on_success(environment, monkeypatch):
    fake = types.ModuleType("rapid_mlx.cli")
    fake.cli_entrypoint = Mock()
    monkeypatch.setitem(sys.modules, "rapid_mlx.cli", fake)
    monkeypatch.setenv(repair._RETRY, "1")
    repair.cli_entrypoint()
    assert repair._RETRY not in repair.os.environ


def test_telemetry_flush_cannot_block_restart(environment, monkeypatch):
    fake = types.ModuleType("rapid_mlx.cli")
    fake.cli_entrypoint = Mock(side_effect=ImportError(name="tiktoken"))
    monkeypatch.setitem(sys.modules, "rapid_mlx.cli", fake)
    monkeypatch.setattr(repair, "recover", lambda **_: True)
    from rapid_mlx.telemetry import posthog_sender

    monkeypatch.setattr(posthog_sender, "get_sender", Mock(side_effect=ImportError()))
    execute = Mock(side_effect=SystemExit(17))
    monkeypatch.setattr(repair.os, "execv", execute)
    with pytest.raises(SystemExit, match="17"):
        repair.cli_entrypoint()
    execute.assert_called_once()


def test_missing_distribution_metadata(environment, monkeypatch):
    monkeypatch.setattr(
        repair.metadata,
        "requires",
        Mock(side_effect=repair.metadata.PackageNotFoundError()),
    )
    assert repair.dependency_failure(ImportError(name="tiktoken"))
    assert repair.missing_modules() == ["rapid-mlx distribution metadata"]
    import rapid_mlx

    monkeypatch.setattr(rapid_mlx, "__version__", "0.0.0")
    assert repair.repair_argv()[-1] == "rapid-mlx"


@pytest.mark.parametrize("module", ["yaml._yaml", "tiktoken._tiktoken"])
def test_declared_package_internals_qualify(environment, module):
    assert repair.dependency_failure(ImportError(name=module))


@pytest.mark.parametrize("module", ["mlx_lm.models.newarch", "mlx_vlm.models.x"])
def test_unknown_model_architecture_is_not_repaired(monkeypatch, module):
    monkeypatch.setattr(repair.metadata, "requires", lambda _: ["mlx-lm", "mlx-vlm"])
    assert not repair.dependency_failure(ImportError(name=module))
    assert repair.dependency_failure(ImportError(name="mlx_lm.generate"))


def test_retry_guard_survives_startup_handoff(environment, monkeypatch):
    fake = types.ModuleType("rapid_mlx.cli")
    observed = []
    fake.cli_entrypoint = lambda: observed.append(repair.os.environ.get(repair._RETRY))
    monkeypatch.setitem(sys.modules, "rapid_mlx.cli", fake)
    monkeypatch.setenv(repair._RETRY, "1")
    repair.cli_entrypoint()
    assert observed == ["1"]
    assert repair._RETRY not in repair.os.environ


def test_doctor_names_failed_checks_when_nothing_is_missing(environment, monkeypatch):
    from rapid_mlx.doctor import env_health as eh

    monkeypatch.setattr(eh, "REQUIRED_PACKAGES", [("mlx", "mlx")])
    monkeypatch.setattr(eh, "_runtime_uses_context", lambda _: False)
    monkeypatch.setattr(eh, "_module_available", lambda *_, **__: False)
    monkeypatch.setattr(repair, "missing_modules", lambda: [])
    section = eh.section_required_packages()
    summary = next(
        c for c in section.checks if c.id == "packages.required.dependencies"
    )
    assert "broken: ;" not in summary.label
    assert "mlx" in summary.label.split(";")[0]


def test_doctor_global_pipx_repair_is_never_run_with_double_sudo(
    environment, monkeypatch
):
    from rapid_mlx.doctor.env_health import Check, CheckStatus, Report, Section
    from rapid_mlx.doctor.repairs import apply_repairs, plan_repairs

    monkeypatch.setattr(
        version_check,
        "detect_install_method",
        lambda: version_check.InstallInfo(
            "unknown", "sudo pipx upgrade --global rapid-mlx", []
        ),
    )
    report = Report(
        sections=[
            Section(
                "Required",
                [Check("x", CheckStatus.FAIL, id="packages.required.dependencies")],
            )
        ]
    )
    (action,) = plan_repairs(report)
    assert action.requires_root and action.command[0] == "pipx"
    run = Mock()
    (result,) = apply_repairs([action], run=run, geteuid=lambda: 501)
    run.assert_not_called()
    assert result.status == "not_applied"
    assert result.command == ["sudo", "pipx", "reinstall", "--global", "rapid-mlx"]


def test_declared_modules_survives_missing_packaging(monkeypatch):
    import builtins

    from rapid_mlx import dependency_repair

    monkeypatch.setattr(
        dependency_repair.metadata,
        "requires",
        lambda _name: ["packaging>=23", "mlx>=0.30; sys_platform == 'darwin'"],
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("packaging"):
            raise ImportError("No module named 'packaging'", name="packaging")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert dependency_repair.declared_modules(active_only=True) == {
        "packaging",
        "mlx",
    }


def test_declared_modules_survives_malformed_requirement(monkeypatch):
    monkeypatch.setattr(
        repair.metadata,
        "requires",
        lambda _name: ["mlx>=0.30; sys_platform ==", "numpy>=1"],
    )
    assert repair.declared_modules(active_only=True) == {"mlx", "numpy"}


def test_verification_argv_never_interpolates_unsafe_names():
    argv = repair._verification_argv("bad name; rm", "x")
    assert argv[-1] == "bad name; rm"
    assert "importlib.import_module" in argv[2]
    assert repair._verification_argv("pkg.mod", "Name")[2] == "from pkg.mod import Name"
    assert repair._verification_argv("pkg", "bad name")[2] == "import pkg"


def test_unreadable_prompt_is_no_answer(environment, monkeypatch):
    stdin(monkeypatch)
    monkeypatch.setattr(sys.stdin, "readline", Mock(side_effect=OSError()))
    assert not repair.recover()
    assert environment == ["no_answer"]


def test_base_requirement_import_names_match_installed_distributions(monkeypatch):
    import re
    import tomllib
    from pathlib import Path

    from packaging.requirements import Requirement

    dependencies = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )["project"]["dependencies"]
    owners = repair.metadata.packages_distributions()
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    for text in dependencies:
        req = Requirement(text)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        try:
            repair.metadata.distribution(req.name)
        except repair.metadata.PackageNotFoundError:
            continue
        modules = {
            module
            for module, distributions in owners.items()
            if any(normalize(dist) == normalize(req.name) for dist in distributions)
        }
        monkeypatch.setattr(repair.metadata, "requires", lambda _, text=text: [text])
        declared = repair.declared_modules()
        assert len(declared) == 1, text
        assert declared <= modules, text


def test_pydantic_ai_slim_import_alias(environment, monkeypatch):
    monkeypatch.setattr(
        repair.metadata, "requires", lambda _: ["pydantic-ai-slim[openai]==2.61.0"]
    )
    monkeypatch.setattr(
        repair.importlib.util,
        "find_spec",
        lambda name: object() if name == "pydantic_ai" else None,
    )
    assert repair.missing_modules() == []


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize(
    "direct,expected",
    [
        (
            '{"url":"https://example.org/repo.git","vcs_info":{"vcs":"git","commit_id":"abc123","requested_revision":"main"},"subdirectory":"python"}',
            ["rapid-mlx @ git+https://example.org/repo.git@abc123#subdirectory=python"],
        ),
        (
            '{"url":"file:///tmp/source%20tree","dir_info":{"editable":true}}',
            ["-e", "/tmp/source tree"],
        ),
        (
            '{"url":"https://example.org/rapid.whl","archive_info":{}}',
            ["rapid-mlx @ https://example.org/rapid.whl"],
        ),
        (None, None),
        ("{broken", None),
        (
            '{"url":"https://example.org/repo.git","vcs_info":{"vcs":"git","requested_revision":"release"}}',
            ["rapid-mlx @ git+https://example.org/repo.git@release"],
        ),
        ('{"url":"https://example.org/repo.git","vcs_info":{"vcs":"git"}}', None),
        ('{"url":7}', None),
        (
            '{"url":"file:///tmp/source","dir_info":{}}',
            ["rapid-mlx @ file:///tmp/source"],
        ),
    ],
)
def test_pip_repair_preserves_source(
    environment, monkeypatch, direct, expected, bootstrap
):
    import rapid_mlx

    monkeypatch.setattr(rapid_mlx, "__version__", "9.8.7")
    distribution = Mock()
    distribution.read_text.return_value = direct
    monkeypatch.setattr(
        repair.metadata, "distribution", Mock(return_value=distribution)
    )
    monkeypatch.setattr(
        repair.importlib.util, "find_spec", lambda _: None if bootstrap else object()
    )
    expected = expected or ["rapid-mlx==9.8.7"]
    argv = repair.repair_argv()
    distribution.read_text.assert_called_once_with("direct_url.json")
    if bootstrap:
        assert argv[3:] == expected
        ensurepip = types.ModuleType("ensurepip")
        ensurepip.bootstrap = Mock()
        monkeypatch.setitem(sys.modules, "ensurepip", ensurepip)
        monkeypatch.setattr(sys, "argv", ["-c", *expected])
        call = Mock()
        monkeypatch.setattr(subprocess, "check_call", call)
        exec(argv[2], {})
        call.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", *expected]
        )
    else:
        assert argv == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            *expected,
        ]


@pytest.mark.parametrize(
    "source,url,info,prefix",
    [
        (
            "archive_info",
            "https://example.org/x.zip",
            {},
            "rapid-mlx @ https://example.org/x.zip",
        ),
        (
            "dir_info",
            "file:///tmp/source%20tree",
            {},
            "rapid-mlx @ file:///tmp/source%20tree",
        ),
        ("dir_info", "file:///tmp/source%20tree", {"editable": True}, None),
        (
            "vcs_info",
            "https://example.org/repo.git",
            {"vcs": "git", "commit_id": "abc123"},
            "rapid-mlx @ git+https://example.org/repo.git@abc123",
        ),
    ],
)
@pytest.mark.parametrize(
    "subdirectory,encoded",
    [
        ("python", "python"),
        ("python/pkg #&%", "python/pkg%20%23%26%25"),
        (42, None),
        (None, None),
    ],
)
def test_reinstall_spec_preserves_subdirectory(
    monkeypatch, source, url, info, prefix, subdirectory, encoded
):
    import rapid_mlx

    monkeypatch.setattr(rapid_mlx, "__version__", "9.8.7")
    distribution = Mock()
    distribution.read_text.return_value = json.dumps(
        {"url": url, source: info, "subdirectory": subdirectory}
    )
    monkeypatch.setattr(repair.metadata, "distribution", lambda _: distribution)

    if encoded is None:
        expected = ["rapid-mlx==9.8.7"]
    elif prefix is None:
        expected = ["-e", f"/tmp/source tree/{subdirectory}"]
    else:
        expected = [f"{prefix}#subdirectory={encoded}"]
    assert repair._reinstall_spec() == expected


@pytest.mark.parametrize(
    "archive,subdirectory,fragment",
    [
        ({"hashes": {"sha256": "ab" * 32}}, "", "sha256=" + "ab" * 32),
        (
            {"hashes": {"sha256": "ab" * 32}},
            "python",
            "sha256=" + "ab" * 32 + "&subdirectory=python",
        ),
        (
            {"hashes": {"sha256": "ab" * 32}},
            "python/pkg #&%",
            "sha256=" + "ab" * 32 + "&subdirectory=python/pkg%20%23%26%25",
        ),
        ({"hash": "sha256=" + "ab" * 32}, "", "sha256=" + "ab" * 32),
        (
            {"hashes": {"sha512": "cd" * 64, "sha256": "ab" * 32}},
            "",
            "sha256=" + "ab" * 32,
        ),
        (
            {"hashes": {"unknown": "ab", "sha512": "cd" * 64, "sha1": "ef" * 20}},
            "",
            "sha512=" + "cd" * 64,
        ),
        (None, "", None),
        ([], "", None),
        ({"hashes": None}, "", None),
        ({"hashes": []}, "", None),
        ({"hashes": {}}, "", None),
        ({"hashes": {"unknown": "ab"}}, "", None),
        ({"hashes": {"sha256": None}}, "", None),
        ({"hashes": {"sha256": 42}}, "", None),
        ({"hashes": {"sha256": ""}}, "", None),
        ({"hashes": {"sha256": "not-hex"}}, "", None),
        ({"hashes": {"sha256": "ab&subdirectory=other"}}, "", None),
        ({"hash": None}, "", None),
        ({"hash": 42}, "", None),
        ({"hash": "sha256"}, "", None),
        ({"hash": "unknown=ab"}, "", None),
        ({"hash": "sha256="}, "", None),
        ({"hash": "sha256=not-hex"}, "", None),
    ],
)
def test_reinstall_spec_preserves_archive_hash(
    monkeypatch, archive, subdirectory, fragment
):
    import rapid_mlx

    monkeypatch.setattr(rapid_mlx, "__version__", "9.8.7")
    distribution = Mock()
    distribution.read_text.return_value = json.dumps(
        {
            "url": "https://example.org/rapid.whl",
            "archive_info": archive,
            "subdirectory": subdirectory,
        }
    )
    monkeypatch.setattr(repair.metadata, "distribution", lambda _: distribution)

    expected = (
        f"rapid-mlx @ https://example.org/rapid.whl#{fragment}"
        if fragment is not None
        else "rapid-mlx==9.8.7"
    )
    assert repair._reinstall_spec() == [expected]


def test_pip_repair_unreadable_source_uses_running_version(environment, monkeypatch):
    import rapid_mlx

    monkeypatch.setattr(rapid_mlx, "__version__", "9.8.7")
    distribution = Mock()
    distribution.read_text.side_effect = OSError("unreadable")
    monkeypatch.setattr(repair.metadata, "distribution", lambda _: distribution)
    assert repair.repair_argv()[-1] == "rapid-mlx==9.8.7"


def test_reinstall_spec_missing_direct_url_uses_running_version(monkeypatch):
    import rapid_mlx

    monkeypatch.setattr(rapid_mlx, "__version__", "9.8.7")
    distribution = Mock()
    distribution.read_text.return_value = None
    monkeypatch.setattr(repair.metadata, "distribution", lambda _: distribution)
    assert repair._reinstall_spec() == ["rapid-mlx==9.8.7"]
    distribution.read_text.assert_called_once_with("direct_url.json")


def test_reinstall_spec_unknown_version_uses_bare_spec(monkeypatch):
    import rapid_mlx

    monkeypatch.setattr(rapid_mlx, "__version__", "0.0.0")
    distribution = Mock()
    distribution.read_text.return_value = None
    monkeypatch.setattr(repair.metadata, "distribution", lambda _: distribution)
    assert repair._reinstall_spec() == ["rapid-mlx"]


def test_pip_fallback_preserves_editable_source(environment, monkeypatch):
    distribution = Mock()
    distribution.read_text.return_value = (
        '{"url":"file:///tmp/source","dir_info":{"editable":true}}'
    )
    monkeypatch.setattr(repair.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(
        version_check,
        "detect_install_method",
        lambda: version_check.InstallInfo("unknown", "", []),
    )
    assert repair.repair_argv() == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "-e",
        "/tmp/source",
    ]
