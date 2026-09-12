# SPDX-License-Identifier: Apache-2.0
"""Always-on service checks exposed through ``rapid-mlx doctor``."""

from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

from vllm_mlx import __version__
from vllm_mlx.doctor import env_health as eh


def _status(**changes):
    value = {
        "plist_present": True,
        "registered": True,
        "pid": 42,
        "owner": "serveuser",
        "declared_user": "serveuser",
        "last_exit": 0,
        "launchd_state": "running",
        "runs": 1,
        "crash_loop_suspected": False,
        "model": "qwen3.5-4b-4bit",
        "host": "127.0.0.1",
        "port": 8000,
        "livez": True,
        "readyz": True,
        "plist": "/Library/LaunchDaemons/com.rapidmlx.server.plist",
        "config_file": "/Library/Application Support/Rapid-MLX/Services/config.json",
        "config_error": None,
        "config_valid": True,
        "endpoint_configured": True,
        "endpoint_attributable": True,
        "pending_config": False,
        "executable": None,
    }
    value.update(changes)
    return value


def _runtime_with_metadata(tmp_path, version=__version__):
    executable = tmp_path / "bin" / "rapid-mlx"
    executable.parent.mkdir(parents=True, exist_ok=True)
    interpreter = tmp_path / "bin" / "python3"
    if not interpreter.exists():
        interpreter.symlink_to(sys.executable)
    executable.write_text(
        f"#!{interpreter}\nimport re\n"
        "import sys\n"
        "from vllm_mlx.cli import cli_entrypoint\n"
        "if __name__ == '__main__':\n"
        r"    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])"
        "\n"
        "    sys.exit(cli_entrypoint())\n"
    )
    executable.chmod(0o755)
    metadata = (
        tmp_path
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / f"rapid_mlx-{version}.dist-info"
        / "METADATA"
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"Name: rapid-mlx\nVersion: {version}\n")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(executable.read_bytes()).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    relative_executable = executable.relative_to(tmp_path)
    (metadata.parent / "RECORD").write_text(
        f"../../../{relative_executable},sha256={digest},{executable.stat().st_size}\n"
    )
    return executable


def _refresh_runtime_record(tmp_path, executable):
    record = next(
        tmp_path.glob("lib/python*/site-packages/rapid_mlx-*.dist-info/RECORD")
    )
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(executable.read_bytes()).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    relative_executable = executable.relative_to(tmp_path)
    record.write_text(
        f"../../../{relative_executable},sha256={digest},{executable.stat().st_size}\n"
    )


def test_absent_optional_service_is_healthy():
    section = eh.section_always_on_service(
        status_data=_status(plist_present=False, registered=False),
        platform_name="darwin",
    )
    assert len(section.checks) == 1
    assert section.checks[0].status is eh.CheckStatus.OK
    assert section.checks[0].id == "service.installation"


def test_healthy_service_checks_every_layer(tmp_path):
    executable = _runtime_with_metadata(tmp_path)

    section = eh.section_always_on_service(
        status_data=_status(executable=str(executable)),
        platform_name="darwin",
    )
    assert all(check.status is eh.CheckStatus.OK for check in section.checks)
    assert {check.id for check in section.checks} >= {
        "service.definition",
        "service.registration",
        "service.process",
        "service.owner",
        "service.config",
        "service.runtime",
        "service.endpoint.liveness",
        "service.endpoint.readiness",
    }


def test_crash_loop_is_a_confirmed_failure():
    section = eh.section_always_on_service(
        status_data=_status(
            pid=None,
            owner=None,
            last_exit=78,
            crash_loop_suspected=True,
            livez=False,
            readyz=False,
        ),
        platform_name="darwin",
    )
    by_id = {check.id: check for check in section.checks}
    assert by_id["service.process"].status is eh.CheckStatus.FAIL
    assert "repeated startup failure" in by_id["service.process"].label
    assert by_id["service.endpoint.liveness"].status is eh.CheckStatus.SKIPPED


def test_owner_mismatch_is_a_confirmed_failure():
    section = eh.section_always_on_service(
        status_data=_status(pid=42, owner="root", declared_user="serveuser"),
        platform_name="darwin",
    )

    owner = next(check for check in section.checks if check.id == "service.owner")
    assert owner.status is eh.CheckStatus.FAIL
    assert "expected serveuser" in owner.label


def test_running_service_requires_a_declared_non_root_owner():
    for declared_user in (None, "", "root"):
        section = eh.section_always_on_service(
            status_data=_status(
                pid=42,
                owner="root" if declared_user == "root" else "serveuser",
                declared_user=declared_user,
            ),
            platform_name="darwin",
        )
        owner = next(check for check in section.checks if check.id == "service.owner")
        assert owner.status is eh.CheckStatus.FAIL


def test_live_but_unready_is_warning_not_broken():
    section = eh.section_always_on_service(
        status_data=_status(readyz=False), platform_name="darwin"
    )
    readiness = next(
        check for check in section.checks if check.id == "service.endpoint.readiness"
    )
    assert readiness.status is eh.CheckStatus.WARN
    assert "loading" in readiness.detail


def test_endpoint_probe_errors_are_skipped_not_confirmed_failures():
    section = eh.section_always_on_service(
        status_data=_status(livez=None, readyz=None), platform_name="darwin"
    )

    by_id = {check.id: check for check in section.checks}
    assert by_id["service.endpoint.liveness"].status is eh.CheckStatus.SKIPPED
    assert by_id["service.endpoint.readiness"].status is eh.CheckStatus.SKIPPED


def test_invalid_config_and_pending_change_are_distinct():
    section = eh.section_always_on_service(
        status_data=_status(
            config_error="bad schema", config_valid=False, pending_config=True
        ),
        platform_name="darwin",
    )
    by_id = {check.id: check for check in section.checks}
    assert by_id["service.config"].status is eh.CheckStatus.FAIL
    assert by_id["service.config.pending"].status is eh.CheckStatus.WARN


def test_unverified_config_and_missing_runtime_are_failures():
    section = eh.section_always_on_service(
        status_data=_status(config_valid=False, executable=None),
        platform_name="darwin",
    )

    by_id = {check.id: check for check in section.checks}
    assert by_id["service.config"].status is eh.CheckStatus.FAIL
    assert by_id["service.runtime"].status is eh.CheckStatus.FAIL


def test_zero_launch_count_is_not_rendered_as_unknown(tmp_path):
    executable = _runtime_with_metadata(tmp_path)
    section = eh.section_always_on_service(
        status_data=_status(runs=0, executable=str(executable)),
        platform_name="darwin",
    )

    process = next(check for check in section.checks if check.id == "service.process")
    assert "runs=0" in process.detail


def test_non_executable_runtime_is_a_confirmed_failure(tmp_path):
    executable = tmp_path / "rapid-mlx"
    executable.write_text("")
    executable.chmod(0o644)

    section = eh.section_always_on_service(
        status_data=_status(executable=str(executable)),
        platform_name="darwin",
    )

    runtime = next(check for check in section.checks if check.id == "service.runtime")
    assert runtime.status is eh.CheckStatus.FAIL
    assert "not executable" in runtime.label


def test_runtime_metadata_io_error_is_unverified(monkeypatch, tmp_path):
    executable = _runtime_with_metadata(tmp_path)

    def inaccessible(_self, _pattern):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "glob", inaccessible)

    assert eh._service_runtime_version(str(executable)) is None


def test_runtime_metadata_rejects_unrelated_launcher_and_stale_versions(tmp_path):
    executable = _runtime_with_metadata(tmp_path)
    executable.write_text("#!/bin/sh\necho impostor\n")
    assert eh._service_runtime_version(str(executable)) is None


def test_runtime_metadata_rejects_extra_top_level_guard(tmp_path):
    executable = _runtime_with_metadata(tmp_path)
    executable.write_text(
        executable.read_text()
        + "\nif True:\n"
        + "    print('unexpected top-level conditional')\n"
    )

    assert eh._service_runtime_version(str(executable)) is None


def test_runtime_metadata_rejects_shadowed_entrypoint(tmp_path):
    executable = _runtime_with_metadata(tmp_path)
    executable.write_text(
        executable.read_text().replace(
            "if __name__ == '__main__':",
            "from impostor import cli_entrypoint\nif __name__ == '__main__':",
        )
    )

    assert eh._service_runtime_version(str(executable)) is None


def test_runtime_metadata_ignores_other_python_version(tmp_path):
    executable = _runtime_with_metadata(tmp_path)
    stale = tmp_path / "lib/python9.9/site-packages/rapid_mlx-9.9.9.dist-info/METADATA"
    stale.parent.mkdir(parents=True)
    stale.write_text("Name: rapid-mlx\nVersion: 9.9.9\n")

    assert eh._service_runtime_version(str(executable)) == __version__


def test_runtime_metadata_rejects_multiple_active_versions(tmp_path):
    executable = _runtime_with_metadata(tmp_path)
    active_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    stale = (
        tmp_path / f"lib/{active_tag}/site-packages/rapid_mlx-9.9.9.dist-info/METADATA"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("Name: rapid-mlx\nVersion: 9.9.9\n")

    assert eh._service_runtime_version(str(executable)) is None


def test_launchctl_probe_error_is_unverified_not_confirmed_absence():
    section = eh.section_always_on_service(
        status_data=_status(
            registered=False,
            pid=None,
            launchctl_error="TimeoutExpired: launchctl",
        ),
        platform_name="darwin",
    )

    by_id = {check.id: check for check in section.checks}
    assert by_id["service.registration"].status is eh.CheckStatus.WARN
    assert by_id["service.process"].status is eh.CheckStatus.WARN


def test_unattributable_endpoint_cannot_make_service_look_healthy():
    section = eh.section_always_on_service(
        status_data=_status(
            pid=None,
            livez=True,
            readyz=True,
            endpoint_configured=False,
        ),
        platform_name="darwin",
    )

    by_id = {check.id: check for check in section.checks}
    assert by_id["service.endpoint.liveness"].status is eh.CheckStatus.SKIPPED
    assert by_id["service.endpoint.readiness"].status is eh.CheckStatus.SKIPPED


def test_missing_endpoint_attribution_evidence_fails_closed():
    status = _status()
    status.pop("endpoint_attributable")
    section = eh.section_always_on_service(status_data=status, platform_name="darwin")

    by_id = {check.id: check for check in section.checks}
    assert by_id["service.endpoint.liveness"].status is eh.CheckStatus.SKIPPED
    assert by_id["service.endpoint.readiness"].status is eh.CheckStatus.SKIPPED


def test_runtime_metadata_rejects_oversized_or_headerless_launchers(tmp_path):
    executable = _runtime_with_metadata(tmp_path)
    executable.write_bytes(b"x" * (32 * 1024 + 1))
    assert eh._service_runtime_version(str(executable)) is None

    executable.write_text("not a console script\n")
    assert eh._service_runtime_version(str(executable)) is None


def test_runtime_metadata_rejects_missing_or_non_executable_interpreter(
    monkeypatch, tmp_path
):
    executable = _runtime_with_metadata(tmp_path)
    lines = executable.read_text().splitlines()
    lines[0] = f"#!{tmp_path / 'bin' / 'python-missing'}"
    executable.write_text("\n".join(lines) + "\n")
    assert eh._service_runtime_version(str(executable)) is None

    executable = _runtime_with_metadata(tmp_path / "access")
    monkeypatch.setattr(eh.os, "access", lambda *_a, **_k: False)
    assert eh._service_runtime_version(str(executable)) is None


def test_runtime_metadata_uses_pyvenv_version_for_unversioned_interpreter(tmp_path):
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    executable = _runtime_with_metadata(tmp_path)
    executable.write_text(executable.read_text().replace("python3", "python", 1))
    _refresh_runtime_record(tmp_path, executable)
    (tmp_path / "pyvenv.cfg").write_text(
        f"version = {sys.version_info.major}.{sys.version_info.minor}.0\n"
    )

    assert eh._service_runtime_version(str(executable)) == __version__


def test_runtime_metadata_rejects_unversioned_environment_without_version(tmp_path):
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    executable = _runtime_with_metadata(tmp_path)
    executable.write_text(executable.read_text().replace("python3", "python", 1))

    assert eh._service_runtime_version(str(executable)) is None


def test_runtime_metadata_skips_wrong_distribution_and_missing_record_entry(tmp_path):
    executable = _runtime_with_metadata(tmp_path)
    metadata = next(
        tmp_path.glob("lib/python*/site-packages/rapid_mlx-*.dist-info/METADATA")
    )
    metadata.write_text("Name: another-project\nVersion: 1.0\n")
    assert eh._service_runtime_version(str(executable)) is None

    executable = _runtime_with_metadata(tmp_path / "record")
    record = next(
        (tmp_path / "record").glob(
            "lib/python*/site-packages/rapid_mlx-*.dist-info/RECORD"
        )
    )
    record.write_text(f"missing-file,sha256=ignored,1\n{record.read_text()}")
    assert eh._service_runtime_version(str(executable)) == __version__

    executable_row = record.read_text().splitlines()[-1]
    record.write_text(executable_row.replace("sha256=", "missing=") + "\n")
    assert eh._service_runtime_version(str(executable)) is None


def test_service_section_platform_and_collector_paths(monkeypatch):
    section = eh.section_always_on_service(platform_name="linux")
    assert section.checks[0].id == "service.platform"

    import vllm_mlx.headless_service.status as service_status

    monkeypatch.setattr(
        service_status,
        "collect_status",
        lambda **_kwargs: _status(plist_present=False, registered=False),
    )
    section = eh.section_always_on_service(platform_name="darwin")
    assert section.checks[0].id == "service.installation"


def test_uninstalled_service_launchctl_error_is_unverified():
    section = eh.section_always_on_service(
        status_data=_status(
            plist_present=False,
            registered=False,
            launchctl_error="permission denied",
        ),
        platform_name="darwin",
    )
    assert section.checks[0].status is eh.CheckStatus.WARN


def test_service_section_covers_runtime_and_owner_failure_variants(
    tmp_path, monkeypatch
):
    # Doctor suppresses the version comparison entirely when it cannot
    # determine its OWN version (``__version__ == "0.0.0"``), which is exactly
    # what happens in a bare checkout that was never pip-installed — the
    # mismatch below then renders OK and the assertion fails for a reason that
    # has nothing to do with the service. Pin doctor's version so the branch
    # under test is the one that runs.
    #
    # Patched on ``vllm_mlx``, not on ``env_health``: the comparison does
    # ``from vllm_mlx import __version__`` inside the function body, so it
    # re-reads the package attribute on every call and ``env_health`` has no
    # ``__version__`` of its own to patch.
    monkeypatch.setattr("vllm_mlx.__version__", "1.2.3")
    missing = tmp_path / "missing-rapid-mlx"
    section = eh.section_always_on_service(
        status_data=_status(owner=None, executable=str(missing)),
        platform_name="darwin",
    )
    by_id = {check.id: check for check in section.checks}
    assert "could not be verified" in by_id["service.owner"].label
    assert by_id["service.runtime"].status is eh.CheckStatus.FAIL

    unverified = tmp_path / "rapid-mlx"
    unverified.write_text("not a launcher\n")
    unverified.chmod(0o755)
    section = eh.section_always_on_service(
        status_data=_status(executable=str(unverified)), platform_name="darwin"
    )
    runtime = next(check for check in section.checks if check.id == "service.runtime")
    assert runtime.status is eh.CheckStatus.WARN

    different = _runtime_with_metadata(tmp_path / "different", version="9.9.9")
    section = eh.section_always_on_service(
        status_data=_status(executable=str(different)), platform_name="darwin"
    )
    runtime = next(check for check in section.checks if check.id == "service.runtime")
    assert runtime.status is eh.CheckStatus.WARN
    assert "differs from Doctor" in runtime.label
