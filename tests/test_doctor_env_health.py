# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the env-health probes in ``vllm_mlx.doctor.env_health``.

These tests are the safety net for the user-facing ``rapid-mlx doctor``
contract:

* Apple-Silicon detection works on macOS-arm64 and falls back gracefully.
* Required-package matrix flags missing packages as ✗ (fail), missing
  optional packages as ⚠ (warn).
* HF cache writability is correctly probed.
* Network timeout produces a ⚠, never a ✗ — air-gapped CI must not break.
* Exit code is 0 with only ✓/⚠; 1 if any ✗.

Every test is dependency-injected (no real subprocess / network / disk
mutation) so the suite runs identically on every Python and every OS.
"""

from __future__ import annotations

import importlib
import json
import os
import plistlib
import shlex
import subprocess
import sys
import time

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test matrix
    import tomli as tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from vllm_mlx.doctor import env_health as eh


@pytest.fixture(autouse=True)
def clean_runtime_probe_state(clean_doctor_runtime_state, monkeypatch):
    original_import_probe = eh._runtime_module_importable

    def import_from_probe_when_available(
        runtime, module, sidecar_root, *, exercise=False, **kwargs
    ):
        probe = eh._probe_runtime(runtime, sidecar_root)
        package = eh._probe_package_by_module(probe, module) if probe else None
        if package is not None and package.get("importable") is not None:
            return bool(package["importable"])
        return original_import_probe(
            runtime,
            module,
            sidecar_root,
            exercise=exercise,
            **kwargs,
        )

    monkeypatch.setattr(
        eh, "_runtime_module_importable", import_from_probe_when_available
    )
    yield


# ---------------------------------------------------------------------------
# Section: System
# ---------------------------------------------------------------------------


def test_apple_silicon_detected():
    """On a real arm64 macOS box the System section should report the chip."""
    with (
        mock.patch.object(eh.platform, "system", return_value="Darwin"),
        mock.patch.object(eh.platform, "machine", return_value="arm64"),
        mock.patch.object(
            eh.platform, "mac_ver", return_value=("14.3", ("", "", ""), "arm64")
        ),
        mock.patch.object(eh.platform, "release", return_value="23.3.0"),
        mock.patch.object(
            eh, "_detect_apple_silicon", return_value=("Apple M3 Pro", 36)
        ),
        mock.patch.object(eh, "_disk_free_gb", return_value=162.0),
        mock.patch.object(eh, "_dir_size_gb", return_value=12.0),
    ):
        section = eh.section_system()

    apple_row = next(c for c in section.checks if "Apple Silicon" in c.label)
    assert apple_row.status is eh.CheckStatus.OK
    assert "Apple M3 Pro" in apple_row.label
    assert "36 GB" in apple_row.label


def test_python_section_identifies_application_environment(tmp_path, monkeypatch):
    home = tmp_path / "home"
    runtime = home / ".rapid-mlx" / "bin" / "python3"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    monkeypatch.setattr(eh.Path, "home", lambda: home)
    monkeypatch.setattr(eh.sys, "executable", str(runtime))
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(runtime))

    section = eh.section_python()
    runtime_row = next(
        c for c in section.checks if c.label.startswith("Active runtime")
    )

    assert runtime_row.status is eh.CheckStatus.OK
    assert "Rapid-MLX application environment" in runtime_row.label
    assert str(runtime.resolve()) in runtime_row.detail
    assert "relevant_sys.path" in runtime_row.detail


def test_python_section_preserves_symlinked_application_environment(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    application_bin = home / ".rapid-mlx" / "bin"
    base_python = tmp_path / "base" / "python3"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("")
    application_runtime = application_bin / "python3"
    application_bin.mkdir(parents=True)
    application_runtime.symlink_to(base_python)
    monkeypatch.setattr(eh.Path, "home", lambda: home)
    monkeypatch.setattr(eh.sys, "executable", str(application_runtime))
    monkeypatch.setattr(eh.sys, "prefix", str(home / ".rapid-mlx"))

    section = eh.section_python()
    runtime_row = next(
        c for c in section.checks if c.label.startswith("Active runtime")
    )

    assert runtime_row.status is eh.CheckStatus.OK
    assert "Rapid-MLX application environment" in runtime_row.label


def test_python_section_does_not_trust_symlink_target_directory(tmp_path, monkeypatch):
    home = tmp_path / "home"
    application_prefix = home / ".rapid-mlx"
    base_runtime = tmp_path / "base" / "bin" / "python3"
    base_runtime.parent.mkdir(parents=True)
    base_runtime.write_text("")
    monkeypatch.setattr(eh.Path, "home", lambda: home)
    monkeypatch.setattr(eh.sys, "executable", str(base_runtime))
    monkeypatch.setattr(eh.sys, "prefix", str(application_prefix))

    kind = eh._runtime_environment(
        base_runtime.absolute(),
        Path(application_prefix),
    )

    assert kind == "virtual environment"


def test_python_section_does_not_classify_base_target_as_application(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    application_prefix = home / ".rapid-mlx"
    application_bin = application_prefix / "bin"
    application_bin.mkdir(parents=True)
    base_runtime = tmp_path / "base" / "bin" / "python3"
    base_runtime.parent.mkdir(parents=True)
    base_runtime.write_text("")
    (application_bin / "python3").symlink_to(base_runtime)
    monkeypatch.setattr(eh.Path, "home", lambda: home)
    monkeypatch.setattr(eh.sys, "executable", str(base_runtime))
    monkeypatch.setattr(eh.sys, "prefix", str(application_prefix))

    kind = eh._runtime_environment(base_runtime.absolute(), Path(application_prefix))

    assert kind != "Rapid-MLX application environment"


@pytest.fixture(name="allow_rapid_mlx_module_servers")
def allow_rapid_mlx_module_servers(monkeypatch):
    monkeypatch.setattr(
        eh,
        "_runtime_has_rapid_mlx_distribution",
        lambda runtime, cwd, env: True,
    )
    monkeypatch.setattr(
        eh,
        "_filesystem_runtime_has_rapid_mlx_distribution",
        lambda runtime: True,
    )


def test_stopped_runtime_override_accepts_system_python_and_resolves_relative_path(
    tmp_path,
    monkeypatch,
):
    system_runtime = tmp_path / "opt" / "homebrew" / "bin" / "python3"
    system_runtime.parent.mkdir(parents=True)
    system_runtime.write_text("")
    relative_runtime = tmp_path / "bin" / "python3"
    relative_runtime.parent.mkdir(parents=True)
    relative_runtime.write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", "bin/python3")

    assert eh._is_diagnostic_python_override(system_runtime)
    assert eh._runtime_python_path() == relative_runtime.resolve()


def test_runtime_authentication_uses_filesystem_distribution_without_launch(tmp_path):
    runtime = tmp_path / "arbitrary-venv" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    site_root = (
        runtime.parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    package_root = site_root / "vllm_mlx"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    (package_root / "cli.py").write_text("from vllm_mlx.server import app\n")
    (site_root / "rapid_mlx-0.0.0.dist-info").mkdir()
    (site_root / "rapid_mlx-0.0.0.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rapid-mlx\nVersion: 0.0.0\n"
    )

    assert eh._filesystem_runtime_has_rapid_mlx_distribution(runtime) is True
    assert eh._is_trusted_runtime_executable(runtime) is True

    untrusted_runtime = tmp_path / "not-rapid-mlx" / "bin" / "python"
    untrusted_runtime.parent.mkdir(parents=True)
    untrusted_runtime.write_text("")

    assert eh._is_trusted_runtime_executable(untrusted_runtime) is False


def test_system_layout_runtime_is_authenticated_by_isolated_distribution_probe(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    system_runtime = tmp_path / "usr" / "local" / "bin" / "python3"
    system_runtime.parent.mkdir(parents=True)
    system_runtime.write_text("#!/bin/sh\nprintf '[\"rapid-mlx\"]\\n'\n")
    system_runtime.chmod(0o755)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(system_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    "test-model",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(system_runtime)

        def environ(self):
            return {"PATH": str(system_runtime.parent)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._filesystem_runtime_has_rapid_mlx_distribution(system_runtime) is False
    assert eh._runtime_python_path() == system_runtime.absolute()


def test_local_real_import_uses_trusted_dynamic_sys_path(tmp_path, monkeypatch):
    module_root = tmp_path / "layered-site"
    module_root.mkdir()
    (module_root / "probe_dynamic_trusted.py").write_text("probe_loaded = True\n")
    monkeypatch.syspath_prepend(module_root)
    monkeypatch.setattr(
        eh,
        "_TRUSTED_SYS_PATH_ROOTS",
        (module_root.resolve(),),
    )

    assert eh._module_available("probe_dynamic_trusted", real_import=True) is True


def test_import_probe_cache_separates_trust_policies(tmp_path, monkeypatch):
    runtime = Path(sys.executable)
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    (trusted_root / "probe_cache_policy.py").write_text("probe_loaded = True\n")

    trusted = eh._runtime_module_importable(
        runtime,
        "probe_cache_policy",
        None,
        trusted_roots=(trusted_root,),
    )
    untrusted = eh._runtime_module_importable(runtime, "probe_cache_policy", None)

    assert trusted is True
    assert untrusted is False


def test_run_all_clears_invocation_scoped_caches(monkeypatch):
    def section():
        cached_section = eh.Section("Cached")
        cached_section.add("cached", eh.CheckStatus.OK)
        return cached_section

    monkeypatch.setattr(eh, "_SECTION_BUILDERS", (section,))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: Path(sys.executable))
    eh._RUNTIME_PROBE_CACHE[Path("/stale")] = {}
    eh._RUNTIME_IMPORT_CACHE[Path("/stale"), "module", "", False, True, ()] = True
    eh._RUNTIME_IMPORT_TIMEOUTS.add((Path("/stale"), "module", "", False, True, ()))
    eh._RUNTIME_DISTRIBUTION_CACHE[Path("/stale")] = True
    eh._RUNTIME_CONTEXTS[Path("/stale")] = (Path("/"), {})

    try:
        eh.run_all()
    finally:
        eh._RUNTIME_SELECTION_DONE = False

    assert eh._RUNTIME_PROBE_CACHE == {}
    assert eh._RUNTIME_IMPORT_CACHE == {}
    assert not eh._RUNTIME_IMPORT_TIMEOUTS
    assert eh._RUNTIME_DISTRIBUTION_CACHE == {}
    assert eh._RUNTIME_CONTEXTS == {}


def test_module_server_runtime_must_register_rapid_mlx_distribution(tmp_path):
    def make_runtime(name: str) -> Path:
        runtime = tmp_path / name / "bin" / "python3"
        runtime.parent.mkdir(parents=True)
        runtime.write_text("#!/bin/sh\nprintf '[\"" + name + "\"]\\n'\n")
        runtime.chmod(0o755)
        return runtime

    assert eh._runtime_has_rapid_mlx_distribution(
        make_runtime("rapid-mlx"),
        tmp_path,
        {"PATH": "/usr/bin:/bin"},
    )
    assert not eh._runtime_has_rapid_mlx_distribution(
        make_runtime("not-rapid-mlx"),
        tmp_path,
        {"PATH": "/usr/bin:/bin"},
    )


def test_runtime_validation_rejects_fake_system_python_distribution(tmp_path):
    server_cwd = tmp_path / "server-cwd"
    server_cwd.mkdir()
    fake_dist_info = server_cwd / "vllm_mlx-999.fake.dist-info" / "METADATA"
    fake_dist_info.parent.mkdir(parents=True)
    fake_dist_info.write_text(
        "Metadata-Version: 2.1\nName: rapid-mlx\nVersion: 999.fake\n"
    )

    assert not eh._runtime_has_rapid_mlx_distribution(
        Path("/usr/bin/python3"),
        server_cwd,
        {"PATH": "/usr/bin:/bin"},
    )


def test_runtime_validation_requires_system_python_to_register_distribution():
    assert not eh._runtime_has_rapid_mlx_distribution(
        Path("/usr/bin/python3"),
        Path("/"),
        {"PATH": "/usr/bin:/bin"},
    )


def test_runtime_validation_ignores_context_distribution_metadata(tmp_path):
    context_root = tmp_path / "server-context"
    dist_info = context_root / "rapid_mlx-999.fake.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rapid-mlx\nVersion: 999.fake\n"
    )
    runtime = tmp_path / "runtime" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text('#!/bin/sh\nexec /usr/bin/python3 "$@"\n')
    runtime.chmod(0o755)

    assert not eh._runtime_has_rapid_mlx_distribution(
        runtime,
        context_root,
        {"PATH": "/usr/bin:/bin"},
    )


def test_runtime_distribution_probe_preserves_venv_symlinks(tmp_path):
    base_bin = tmp_path / "base" / "bin"
    base_bin.mkdir(parents=True)
    probe_script = base_bin / "python3"
    probe_script.write_text(
        '#!/bin/sh\ncase "$0" in *venv-a*) printf \'["rapid-mlx"]\\n\' ;; '
        "*) printf '[]\\n' ;; esac\n"
    )
    probe_script.chmod(0o755)

    venv_a = tmp_path / "venv-a" / "bin" / "python3"
    venv_a.parent.mkdir(parents=True)
    venv_a.symlink_to(probe_script)
    venv_b = tmp_path / "venv-b" / "bin" / "python3"
    venv_b.parent.mkdir(parents=True)
    venv_b.symlink_to(probe_script)

    assert eh._RUNTIME_DISTRIBUTION_CACHE == {}

    assert eh._runtime_has_rapid_mlx_distribution(
        venv_a, tmp_path, {"PATH": "/usr/bin:/bin"}
    )
    assert eh._RUNTIME_DISTRIBUTION_CACHE[venv_a] is True
    assert not eh._runtime_has_rapid_mlx_distribution(
        venv_b, tmp_path, {"PATH": "/usr/bin:/bin"}
    )
    assert eh._RUNTIME_DISTRIBUTION_CACHE[venv_b] is False


def test_remote_probe_reads_sanitized_context_metadata_without_trusting_imports(
    tmp_path,
    monkeypatch,
):
    runtime = Path(sys.executable).resolve()
    context_root = tmp_path / "server-context"
    context_root.mkdir()
    (context_root / "probe_metadata_module.py").write_text("probe_loaded = True\n")
    dist_info = context_root / "transformers-5.12.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: transformers\nVersion: 5.12.1\n"
    )
    (dist_info / "RECORD").write_text("probe_metadata_module.py,,\n")
    monkeypatch.setattr(eh, "_RUNTIME_CONTEXTS", {runtime: (context_root, {})})
    monkeypatch.setattr(
        eh, "_RUNTIME_PACKAGES", {"transformers": "probe_metadata_module"}
    )

    probe = eh._probe_runtime(
        runtime,
    )

    assert probe is not None
    package = cast("dict[str, object]", probe["packages"]["transformers"])
    assert package["version"] == "5.12.1"
    assert package["discoverable"] is True
    assert package["trusted_origin"] is False
    assert package["importable"] is None
    assert not eh._runtime_module_importable(runtime, "probe_metadata_module", None)


def test_remote_probe_rejects_metadata_from_unowned_context(
    tmp_path,
    monkeypatch,
):
    runtime = Path(sys.executable).resolve()
    context_root = tmp_path / "server-context"
    context_root.mkdir()
    module_root = tmp_path / "runtime-packages"
    site_root = module_root / "site-packages"
    module_root.mkdir()
    site_root.mkdir()
    (site_root / "probe_unrelated_module.py").write_text("probe_loaded = True\n")
    dist_info = context_root / "transformers-5.12.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: transformers\nVersion: 5.12.1\n"
    )
    monkeypatch.setattr(eh, "_RUNTIME_CONTEXTS", {runtime: (context_root, {})})
    monkeypatch.setattr(
        eh,
        "_RUNTIME_PACKAGES",
        {"transformers": "probe_unrelated_module"},
    )
    probe = eh._probe_runtime(runtime, sidecar_root=module_root)

    assert probe is not None
    package = cast("dict[str, object]", probe["packages"]["transformers"])
    assert package["version"] is None
    assert package["discoverable"] is True
    assert package["trusted_origin"] is True
    assert package["importable"] is None


def test_metadata_missing_remote_module_is_not_claimed_importable(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server-runtime" / "bin" / "python"
    report = {
        "executable": str(runtime),
        "base_prefix": str(tmp_path / "base"),
        "packages": {
            "mlx-vlm": {
                "importable": None,
                "discoverable": True,
                "trusted_origin": False,
                "module": "mlx_vlm",
                "version": None,
            }
        },
        "path": [],
        "prefix": str(tmp_path / "runtime"),
    }
    runtime.parent.mkdir(parents=True)
    runtime.write_text(f"#!/bin/sh\ncat <<'JSON'\n{json.dumps(report)}\nJSON\n")
    runtime.chmod(0o755)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_RUNTIME_PROBE_CACHE", {})
    monkeypatch.setattr(eh, "_RUNTIME_IMPORT_CACHE", {})
    monkeypatch.setattr(
        eh,
        "_module_visibility",
        lambda dist, runtime=None: (True, False),
    )

    section = eh.section_optional_packages()

    row = next(check for check in section.checks if check.label.startswith("mlx-vlm"))
    assert "importability is not verified" in row.label
    assert "importable in" not in row.label


def test_remote_probe_preserves_context_path_order(tmp_path, monkeypatch):
    first_context = tmp_path / "first"
    second_context = tmp_path / "second"
    first_context.mkdir()
    second_context.mkdir()
    monkeypatch.setattr(
        eh,
        "_RUNTIME_CONTEXTS",
        {
            Path(sys.executable).resolve(): (
                first_context,
                {"PYTHONPATH": str(second_context)},
            )
        },
    )
    monkeypatch.setattr(eh, "_RUNTIME_PACKAGES", {"transformers": "probe_single_file"})

    probe = eh._probe_runtime(
        Path(sys.executable).resolve(),
    )

    assert probe is not None
    path = cast("list[object]", probe["path"])
    assert path[:2] == [str(first_context.resolve()), str(second_context.resolve())]


def test_apple_silicon_warn_on_non_arm64_mac():
    """Intel Mac should produce a WARN row, not a FAIL."""
    with (
        mock.patch.object(eh.platform, "system", return_value="Darwin"),
        mock.patch.object(eh.platform, "machine", return_value="x86_64"),
        mock.patch.object(
            eh.platform, "mac_ver", return_value=("14.3", ("", "", ""), "x86_64")
        ),
        mock.patch.object(eh.platform, "release", return_value="23.3.0"),
        mock.patch.object(eh, "_disk_free_gb", return_value=200.0),
        mock.patch.object(eh, "_dir_size_gb", return_value=None),
    ):
        section = eh.section_system()

    assert any(
        c.status is eh.CheckStatus.WARN and "Non-Apple-Silicon" in c.label
        for c in section.checks
    )


def test_low_disk_marks_fail():
    """< 5 GB free disk is a hard FAIL (next download will fail)."""
    with (
        mock.patch.object(eh.platform, "system", return_value="Linux"),
        mock.patch.object(eh.platform, "machine", return_value="x86_64"),
        mock.patch.object(eh.platform, "mac_ver", return_value=("", ("", "", ""), "")),
        mock.patch.object(eh.platform, "release", return_value="6.5.0"),
        mock.patch.object(eh, "_disk_free_gb", return_value=2.0),
        mock.patch.object(eh, "_dir_size_gb", return_value=None),
    ):
        section = eh.section_system()

    fail_rows = [c for c in section.checks if c.status is eh.CheckStatus.FAIL]
    assert any("Free disk" in c.label for c in fail_rows), [
        c.label for c in section.checks
    ]


def test_huge_hf_cache_marks_warn(tmp_path):
    """> 100 GB HF cache → WARN with cleanup hint.

    The cache dir is pointed at a fresh ``tmp_path`` that EXISTS so the
    ``cache.exists()`` branch is taken deterministically on every platform
    (a fresh Linux CI runner has no ``~/.cache/huggingface`` yet, which would
    otherwise short-circuit to "HF cache: not present" and skip the WARN). The
    reported size comes from the mocked ``_dir_size_gb``, not host state.
    """
    (tmp_path / "dummy").mkdir()
    with (
        mock.patch.object(eh.platform, "system", return_value="Darwin"),
        mock.patch.object(eh.platform, "machine", return_value="arm64"),
        mock.patch.object(
            eh.platform, "mac_ver", return_value=("14.3", ("", "", ""), "arm64")
        ),
        mock.patch.object(eh.platform, "release", return_value="23.3.0"),
        mock.patch.object(eh, "_detect_apple_silicon", return_value=("Apple M3", 36)),
        mock.patch.object(eh, "_disk_free_gb", return_value=200.0),
        mock.patch.object(eh, "_dir_size_gb", return_value=246.0),
        mock.patch.object(eh, "_hf_cache_dir", return_value=tmp_path),
    ):
        section = eh.section_system()

    warn_rows = [c for c in section.checks if c.status is eh.CheckStatus.WARN]
    assert any("HF cache size: 246 GB" in c.label for c in warn_rows)
    assert any("rapid-mlx rm" in c.label for c in warn_rows)


# ---------------------------------------------------------------------------
# Section: Python
# ---------------------------------------------------------------------------


def test_python_version_reported():
    """The Python section always reports the running interpreter version."""
    section = eh.section_python()
    py_row = section.checks[0]
    assert py_row.label.startswith("Python ")
    # Anything >= 3.10 is OK, < 3.10 is FAIL — the running interpreter is
    # whatever pytest is using, which is always supported by our matrix.
    assert py_row.status is eh.CheckStatus.OK


def test_install_location_reported():
    section = eh.section_python()
    # Second row is always the install-location classifier.
    loc_row = next(c for c in section.checks if "Install location" in c.label)
    assert "Install location" in loc_row.label


# ---------------------------------------------------------------------------
# Section: Agent integrations
# ---------------------------------------------------------------------------


class _Connection:
    def close(self):
        pass


def test_agent_integrations_are_quiet_when_no_client_is_configured(tmp_path):
    section = eh.section_agent_integrations(home=tmp_path)
    assert section.checks[0].status is eh.CheckStatus.OK


def test_agent_integrations_report_config_and_server_reachability(tmp_path):
    claude = tmp_path / ".claude/settings.json"
    claude.parent.mkdir(parents=True)
    claude.write_text('{"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:8000"}}')
    cont = tmp_path / ".continue/config.json"
    cont.parent.mkdir(parents=True)
    cont.write_text(
        '{"models":[{"title":"rapid-mlx","provider":"openai",'
        '"apiBase":"http://localhost:8001/v1"}]}'
    )

    def connect(address, *, timeout):
        assert timeout == 0.25
        if address[1] == 8001:
            raise ConnectionRefusedError
        return _Connection()

    section = eh.section_agent_integrations(home=tmp_path, connect=connect)

    assert [check.status for check in section.checks] == [
        eh.CheckStatus.OK,
        eh.CheckStatus.WARN,
    ]
    assert [check.label for check in section.checks] == [
        "Claude Code server is reachable",
        "Continue.dev server is not reachable",
    ]


@pytest.mark.parametrize("claude_config", ["not json", "null", "[]"])
def test_agent_integrations_warn_for_malformed_or_inactive_config(
    tmp_path, claude_config
):
    claude = tmp_path / ".claude/settings.json"
    claude.parent.mkdir(parents=True)
    claude.write_text(claude_config)
    cline = (
        tmp_path
        / "Library/Application Support/Code/User/globalStorage"
        / "saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
    )
    cline.parent.mkdir(parents=True)
    cline.write_text(
        '{"apiProvider":"anthropic","openAiBaseUrl":"http://localhost:8000/v1"}'
    )

    section = eh.section_agent_integrations(
        home=tmp_path,
        connect=lambda *_args, **_kwargs: pytest.fail("must not connect"),
    )

    assert len(section.checks) == 2
    assert all(check.status is eh.CheckStatus.WARN for check in section.checks)


# ---------------------------------------------------------------------------
# Section: Required + optional packages
# ---------------------------------------------------------------------------


def test_required_packages_all_present_marks_ok():
    """When every required dist is installed, every row is OK."""
    supported = {
        "mlx": "0.32.5",
        "mlx-lm": "0.31.4",
        "transformers": "5.12.1",
    }

    def fake_ver(dist: str, runtime=None) -> str:
        return str(supported.get(dist, "9.9.9"))

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_module_available", return_value=True),
    ):
        section = eh.section_required_packages()
    assert all(c.status is eh.CheckStatus.OK for c in section.checks)
    # Each row carries the version returned for that distribution.
    assert all(c.status is eh.CheckStatus.OK for c in section.checks)


def test_runtime_compatibility_policy_matches_project():
    """Doctor's runtime contract must follow pyproject, not drift by hand."""
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    dependencies = [
        Requirement(dependency)
        for dependency in project["project"]["dependencies"]
        + project["project"]["optional-dependencies"]["vision"]
    ]
    expected = {
        "mlx": ">=0.32.1,<0.33",
        "mlx-lm": ">=0.31.3,<0.32",
        "transformers": ">=5.0.0,!=5.13.0,<5.16",
        "mlx-vlm": "==0.6.17",
    }
    policy = {
        requirement.name.lower(): str(requirement.specifier)
        for requirement in dependencies
        if requirement.name.lower() in expected
    }
    policy = {name: str(SpecifierSet(specifier)) for name, specifier in policy.items()}
    assert set(policy) == set(expected)
    assert set(eh._SUPPORTED_VERSIONS) == set(expected)
    assert all(
        SpecifierSet(policy[name]) == SpecifierSet(expected[name]) for name in expected
    )
    assert all(
        SpecifierSet(eh._SUPPORTED_VERSIONS[name]) == SpecifierSet(expected[name])
        for name in expected
    )


def test_required_package_missing_marks_fail():
    def fake_ver(dist: str, runtime=None) -> str | None:
        return None if dist == "transformers" else "1.2.3"

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_module_available", return_value=False),
    ):
        section = eh.section_required_packages()

    transformers_row = next(c for c in section.checks if "transformers" in c.label)
    assert transformers_row.status is eh.CheckStatus.FAIL
    assert "not installed" in transformers_row.label


def test_local_import_health_check_ignores_untrusted_shadow_modules(
    tmp_path,
    monkeypatch,
):
    shadow = tmp_path / "doctor_untrusted_shadow_probe.py"
    shadow.write_text("raise RuntimeError('untrusted shadow module ran')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    assert not eh._module_available("doctor_untrusted_shadow_probe", real_import=True)


def test_local_import_health_check_catches_system_exit(tmp_path):
    site_root = tmp_path / "site-packages"
    site_root.mkdir()
    (site_root / "damaged_dependency.py").write_text("raise SystemExit(2)\n")

    assert not eh._runtime_module_importable(
        Path(sys.executable).resolve(), "damaged_dependency", tmp_path
    )


def test_local_pillow_probe_catches_system_exit(monkeypatch):
    broken_image = SimpleNamespace(new=mock.Mock(side_effect=SystemExit(2)))
    monkeypatch.setattr(eh, "_module_origin_is_trusted", lambda _module: True)
    with mock.patch.dict(
        eh.sys.modules,
        {"PIL": SimpleNamespace(Image=broken_image)},
    ):
        assert not eh._pil_importable()


def test_remediation_command_quotes_interpreter_and_requirements():
    runtime = Path("/tmp/space dir/bad'python")

    command = eh._runtime_pip_command(
        "rapid-mlx[vision]", "transformers>=5.0,<5.16", runtime=runtime
    )

    assert command == (
        "'/tmp/space dir/bad'\"'\"'python' -m pip install --upgrade "
        "'rapid-mlx[vision]' 'transformers>=5.0,<5.16'"
    )


def test_signed_sidecar_required_dependency_gets_safe_repair(tmp_path):
    runtime = _stage_sidecar_bundle(tmp_path)

    def fake_ver(dist: str, runtime=None) -> str | None:
        return None if dist == "transformers" else "1.2.3"

    with (
        mock.patch.object(eh.sys, "executable", str(runtime)),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_module_available", return_value=False),
    ):
        section = eh.section_required_packages()

    row = next(c for c in section.checks if c.label.startswith("transformers"))
    assert row.status is eh.CheckStatus.FAIL
    assert "reinstall Rapid-MLX Desktop.app" in row.label
    assert "pip install" not in row.label
    assert "-m pip" not in row.label


def test_importable_layered_package_is_not_reported_missing():
    """A layered runtime may expose modules while omitting dist-info.

    Doctor must describe the unverifiable metadata, not contradict the server
    by claiming the package is absent.
    """
    with (
        mock.patch.object(eh, "_safe_version", return_value=None),
        mock.patch.object(
            eh,
            "_module_available",
            side_effect=lambda module, _runtime=None, *, real_import=False: (
                module == "transformers"
            ),
        ),
    ):
        section = eh.section_required_packages()

    row = next(c for c in section.checks if c.label.startswith("transformers"))
    assert row.status is eh.CheckStatus.WARN
    assert "importable" in row.label
    assert "not installed" not in row.label


def test_incompatible_transformers_fails_with_runtime_specific_repair(tmp_path):
    runtime = tmp_path / "app-venv" / "bin" / "python3"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "5.16.0" if dist == "transformers" else "1.2.3"

    with (
        mock.patch.object(eh.sys, "executable", str(runtime)),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
    ):
        section = eh.section_required_packages()

    row = next(c for c in section.checks if c.label.startswith("transformers"))
    assert row.status is eh.CheckStatus.FAIL
    assert "requires >=5.0.0,!=5.13.0,<5.16" in row.label
    assert str(runtime.resolve()) in row.label
    assert "-m pip" in row.label


def test_supported_transformers_remains_ok():
    def fake_ver(dist: str, runtime=None) -> str | None:
        return "5.12.1" if dist == "transformers" else "1.2.3"

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_module_available", return_value=True),
    ):
        section = eh.section_required_packages()

    row = next(c for c in section.checks if c.label.startswith("transformers"))
    assert row.status is eh.CheckStatus.OK


def test_supported_transformers_with_broken_import_is_not_ok():
    def fake_ver(dist: str, runtime=None) -> str | None:
        return "5.12.1" if dist == "transformers" else "1.2.3"

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh,
            "_module_available",
            side_effect=lambda module, _runtime=None, *, real_import=False: (
                module != "transformers"
            ),
        ),
    ):
        section = eh.section_required_packages()

    row = next(c for c in section.checks if c.label.startswith("transformers"))
    assert row.status is eh.CheckStatus.FAIL
    assert "cannot import" in row.label
    assert "5.12.1" in row.label


def test_supported_transformers_excluded_release_is_rejected():
    def fake_ver(dist: str, runtime=None) -> str | None:
        return "5.13.0" if dist == "transformers" else "1.2.3"

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_required_packages()

    row = next(c for c in section.checks if c.label.startswith("transformers"))
    assert row.status is eh.CheckStatus.FAIL
    assert "!=5.13.0" in row.label


def test_doctor_follows_the_running_server_runtime(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")
    report = {
        "executable": str(server_runtime),
        "prefix": str(server_runtime.parents[1]),
        "path": [str(server_runtime.parents[1]), "/runtime-site-packages"],
        "packages": {
            dist: {"importable": False, "version": "1.2.3"}
            for dist in eh._DISTRIBUTION_MODULES
        },
    }
    report["packages"]["transformers"] = {
        "importable": True,
        "version": "5.16.0",
    }
    script = f"#!/bin/sh\ncat <<'JSON'\n{json.dumps(report)}\nJSON\n"
    server_runtime.write_text(script)
    server_runtime.chmod(0o755)

    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(server_runtime))

    python_section = eh.section_python()
    packages = eh.section_required_packages()

    runtime_row = next(
        c for c in python_section.checks if "Selected diagnostic runtime" in c.label
    )
    transformers_row = next(
        c for c in packages.checks if c.label.startswith("transformers")
    )
    assert runtime_row.status is eh.CheckStatus.WARN
    assert "package checks use" in runtime_row.label
    assert "/runtime-site-packages" in runtime_row.detail
    assert transformers_row.status is eh.CheckStatus.FAIL
    assert "5.16.0" in transformers_row.label
    assert str(server_runtime) in transformers_row.label
    assert f"{shlex.quote(str(server_runtime))} -m pip" in transformers_row.label
    assert str(doctor_exe) not in transformers_row.label


def test_running_server_runtime_outranks_runtime_override(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    override_runtime = tmp_path / "override-runtime" / "bin" / "python"
    override_runtime.parent.mkdir(parents=True)
    override_runtime.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.write_text("")
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")
    entrypoint = tmp_path / "bin" / "rapid-mlx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        f"#!{server_runtime}\nfrom vllm_mlx.cli import main\nsys.exit(main())\n"
    )
    entrypoint.chmod(0o755)
    report = {
        "executable": str(server_runtime),
        "prefix": str(server_runtime.parents[1]),
        "path": [str(server_runtime.parents[1]), "/runtime-site-packages"],
        "packages": {
            dist: {"importable": True, "version": "1.2.3"}
            for dist in eh._DISTRIBUTION_MODULES
        },
    }
    report["packages"]["transformers"] = {"importable": True, "version": "5.16.0"}
    script = f"#!/bin/sh\ncat <<'JSON'\n{json.dumps(report)}\nJSON\n"
    server_runtime.write_text(script)
    server_runtime.chmod(0o755)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [str(server_runtime), str(entrypoint), "serve"],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return dict(os.environ)

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError

    eh._RUNTIME_PROBE_CACHE.clear()
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(override_runtime))

    python_section = eh.section_python()
    packages = eh.section_required_packages()

    runtime_row = next(
        c for c in python_section.checks if "Active server runtime" in c.label
    )
    transformers_row = next(
        c for c in packages.checks if c.label.startswith("transformers")
    )
    assert str(server_runtime) in runtime_row.detail
    assert str(override_runtime) not in runtime_row.detail
    assert transformers_row.status is eh.CheckStatus.FAIL
    assert str(server_runtime) in transformers_row.label
    assert str(override_runtime) not in transformers_row.label


def test_discovered_system_python_is_not_restricted_to_runtime_override_layouts(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "opt" / "homebrew" / "bin" / "python3"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(server_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    "test-model",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return dict(os.environ)

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == server_runtime.resolve()


def test_relative_module_server_uses_process_executable(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.write_text("")
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(server_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    "test-model",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return dict(os.environ)

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == server_runtime.resolve()


def test_module_command_accepts_python_flags_before_dash_m(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(doctor_exe),
                    "-O",
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    "test-model",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(doctor_exe)

        def environ(self):
            return {"PATH": str(doctor_exe.parent)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == doctor_exe.absolute()


def test_module_command_accepts_python_value_flags_before_dash_m(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(doctor_exe),
                    "-X",
                    "dev",
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    "test-model",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(doctor_exe)

        def environ(self):
            return {"PATH": str(doctor_exe.parent)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == doctor_exe.absolute()


def test_arbitrary_server_interpreter_requires_explicit_override(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    arbitrary_runtime = tmp_path / "uv" / "bin" / "python"
    arbitrary_runtime.parent.mkdir(parents=True)
    arbitrary_runtime.write_text("")
    monkeypatch.setattr(eh, "_is_trusted_runtime_executable", lambda runtime: False)
    monkeypatch.setattr(
        eh,
        "_runtime_has_rapid_mlx_distribution",
        lambda runtime, cwd, env: False,
    )
    monkeypatch.setattr(eh, "_is_trusted_runtime_executable", lambda runtime: False)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(arbitrary_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    "test-model",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(arbitrary_runtime)

        def environ(self):
            return {"PATH": str(arbitrary_runtime.parent)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == doctor_exe.absolute()
    assert eh._SELECTED_SERVER_RUNTIME is False


def test_running_server_same_interpreter_is_still_server_context(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_context = tmp_path / "server-context"
    server_context.mkdir()

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(doctor_exe),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    "test-model",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(doctor_exe)

        def environ(self):
            return {"PYTHONPATH": str(server_context)}

        def cwd(self):
            return str(server_context)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    selected = eh._runtime_python_path()

    assert selected == doctor_exe.absolute()
    assert eh._SELECTED_SERVER_RUNTIME is True
    assert eh._RUNTIME_CONTEXTS[selected] == (
        server_context.resolve(),
        {"PYTHONPATH": str(server_context)},
    )


def test_doctor_uses_one_cached_runtime_selection(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime" / "bin" / "python"
    other_runtime = tmp_path / "changed" / "bin" / "python"
    selections = []

    def select_runtime():
        selections.append("selected")
        return runtime

    def section():
        section_section = eh.Section("Cached")
        section_section.add("cached", eh.CheckStatus.OK)
        return section_section

    monkeypatch.setattr(eh, "_runtime_python_path", select_runtime)
    monkeypatch.setattr(eh, "_SECTION_BUILDERS", (section,))
    monkeypatch.setattr(eh, "_DOCTOR_DEADLINE", None)
    eh._RUNTIME_SELECTION_DONE = False
    try:
        eh.run_all()
    finally:
        eh._RUNTIME_SELECTION_DONE = False

    assert selections == ["selected"]
    assert eh._selected_runtime()[0] == runtime
    assert other_runtime not in selections


def test_running_server_runtime_preserves_venv_executable_symlink(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    base_python = tmp_path / "base" / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("")
    server_runtime = tmp_path / "server-venv" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.symlink_to(base_python)
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(server_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return dict(os.environ)

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    selected = eh._runtime_python_path()

    assert selected == server_runtime
    assert selected != server_runtime.resolve()


def test_stopped_runtime_override_preserves_venv_executable_symlink(
    tmp_path,
    monkeypatch,
):
    base_python = tmp_path / "base" / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("")
    override_runtime = tmp_path / "override" / "bin" / "python"
    override_runtime.parent.mkdir(parents=True)
    override_runtime.symlink_to(base_python)
    (override_runtime.parents[1] / "pyvenv.cfg").write_text("")
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(override_runtime))

    selected = eh._runtime_python_path()

    assert selected == override_runtime
    assert selected != override_runtime.resolve()


def test_module_command_wins_over_resolved_process_executable(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    base_python = tmp_path / "base" / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("")
    venv_python = tmp_path / "server-venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    (venv_python.parents[1] / "pyvenv.cfg").write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(venv_python),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(base_python)

        def environ(self):
            return dict(os.environ)

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == venv_python


def test_entrypoint_command_derives_venv_python_sibling(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    base_python = tmp_path / "base" / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("")
    venv_root = tmp_path / "server-venv"
    venv_python = venv_root / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    entrypoint = venv_root / "bin" / "rapid-mlx"
    entrypoint.write_text(f"#!{venv_python}\nfrom vllm_mlx.cli import main\nmain()\n")
    entrypoint.chmod(0o755)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [str(entrypoint), "serve"],
                "create_time": 123.0,
            }

        def exe(self):
            return str(base_python)

        def environ(self):
            return dict(os.environ)

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == venv_python


def test_entrypoint_direct_shebang_wins_over_sibling_python(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    venv_root = tmp_path / "server-venv"
    shebang_python = venv_root / "bin" / "python"
    sibling_python = venv_root / "bin" / "python3"
    shebang_python.parent.mkdir(parents=True)
    shebang_python.write_text("")
    sibling_python.write_text("")
    entrypoint = venv_root / "bin" / "rapid-mlx"
    entrypoint.write_text(
        f"#!{shebang_python}\nfrom vllm_mlx.cli import main\nmain()\n"
    )
    entrypoint.chmod(0o755)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [str(entrypoint), "serve"],
                "create_time": 123.0,
            }

        def exe(self):
            return str(doctor_exe)

        def environ(self):
            return dict(os.environ)

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == shebang_python
    assert eh._runtime_python_path() != sibling_python


def test_entrypoint_non_python_shebang_uses_process_executable(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    shell_target = tmp_path / "not-python.sh"
    shell_target.write_text("#!/bin/sh\nexit 0\n")
    shell_target.chmod(0o755)
    entrypoint = tmp_path / "bin" / "rapid-mlx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(f"#!{shell_target}\nfrom vllm_mlx.cli import main\nmain()\n")
    entrypoint.chmod(0o755)
    dist_info = tmp_path / "dist" / "rapid_mlx-0.0.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "RECORD").write_text("rapid-mlx,,\n")
    runtime = tmp_path / "server-runtime" / "bin" / "python3"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    (runtime.parent.parent / "pyvenv.cfg").write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [str(entrypoint), "serve"],
                "create_time": 123.0,
            }

        def exe(self):
            return str(runtime)

        def environ(self):
            return {"PATH": str(runtime.parent)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == runtime.absolute()


def test_local_import_rejects_source_tree_shadow_module(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    shadow = tmp_path / "transformers.py"
    shadow.write_text("raise AssertionError('doctor executed a shadow module')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert eh._module_origin_is_trusted("transformers") is False
    assert not eh._module_available("transformers", real_import=True)


def test_trusted_sys_path_roots_exclude_dynamic_paths(
    tmp_path,
    monkeypatch,
):
    python_path_root = tmp_path / "python-path"
    python_path_root.mkdir()
    source_root = Path(eh.__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(python_path_root))
    monkeypatch.syspath_prepend(tmp_path)

    trusted_roots = eh._trusted_sys_path_roots()

    assert python_path_root.resolve() not in trusted_roots
    assert tmp_path.resolve() not in trusted_roots
    assert source_root not in trusted_roots


def test_local_pillow_probe_rejects_source_tree_shadow_module(
    tmp_path,
    monkeypatch,
):
    shadow = tmp_path / "PIL"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")
    (shadow / "Image.py").write_text(
        "raise AssertionError('doctor executed a Pillow shadow module')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    saved_modules = {name: sys.modules.get(name) for name in ("PIL", "PIL.Image")}
    try:
        for name in saved_modules:
            sys.modules.pop(name, None)
        assert eh._module_origin_is_trusted("PIL.Image") is False
        assert not eh._pil_importable()
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_runtime_import_probe_rejects_non_object_json(tmp_path):
    runtime = tmp_path / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\nprintf '[]\\n'\n")
    runtime.chmod(0o755)
    eh._RUNTIME_IMPORT_CACHE.clear()

    assert not eh._runtime_module_importable(runtime, "transformers", None)


def test_runtime_probes_stop_after_deadline(tmp_path, monkeypatch):
    runtime = Path(sys.executable)
    monkeypatch.setattr(eh, "_DOCTOR_DEADLINE", 0.0)
    monkeypatch.setattr(eh, "_RUNTIME_PROBE_CACHE", {})
    monkeypatch.setattr(eh, "_RUNTIME_IMPORT_CACHE", {})

    assert eh._probe_runtime(runtime) is None
    assert not eh._runtime_module_importable(runtime, "transformers", None)
    assert eh._import_probe_was_interrupted(runtime, "transformers", None)


def test_timed_out_required_import_is_unknown_not_failed(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server-runtime" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    report = {
        "executable": str(runtime),
        "prefix": str(tmp_path),
        "base_prefix": str(tmp_path),
        "path": [],
        "packages": {},
    }
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: report)
    monkeypatch.setattr(eh, "_safe_version", lambda dist, runtime=None: "5.12.1")
    monkeypatch.setattr(
        eh,
        "_module_visibility",
        lambda dist, runtime=None: (False, False),
    )
    eh._RUNTIME_IMPORT_TIMEOUTS.add(
        eh._import_probe_cache_key(runtime, "transformers", None)
    )

    section = eh.section_required_packages()

    row = next(check for check in section.checks if "transformers" in check.label)
    assert row.status is eh.CheckStatus.WARN
    assert "importability unknown" in row.label


def test_timeout_during_visibility_is_unknown_not_failed(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server-runtime" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    report = {
        "executable": str(runtime),
        "prefix": str(tmp_path),
        "base_prefix": str(tmp_path),
        "path": [],
        "packages": {},
    }

    def time_out_during_visibility(dist, runtime=None):
        eh._RUNTIME_IMPORT_TIMEOUTS.add(
            eh._import_probe_cache_key(runtime, "transformers", None)
        )
        return False, False

    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: report)
    monkeypatch.setattr(eh, "_safe_version", lambda dist, runtime=None: "5.12.1")
    monkeypatch.setattr(eh, "_module_visibility", time_out_during_visibility)

    section = eh.section_required_packages()

    row = next(check for check in section.checks if "transformers" in check.label)
    assert row.status is eh.CheckStatus.WARN
    assert "importability unknown" in row.label


def test_unrelated_vllm_mlx_module_server_is_not_selected(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.write_text("")
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(server_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return {
                "PYTHONPATH": os.environ.get(
                    "RAPID_MLX_TEST_SERVER_PYTHONPATH", "/attacker-controlled"
                )
            }

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(
        eh,
        "_runtime_has_rapid_mlx_distribution",
        lambda runtime, cwd, env: False,
    )
    monkeypatch.setattr(eh, "_RUNTIME_PROBE_CACHE", {})

    assert eh._runtime_python_path() == doctor_exe.resolve()


def test_unrelated_process_with_similar_arguments_is_not_a_server(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    override_runtime = tmp_path / "override-runtime" / "bin" / "python"
    override_runtime.parent.mkdir(parents=True)
    (override_runtime.parents[1] / "pyvenv.cfg").write_text("")
    report = {
        "executable": str(override_runtime),
        "base_prefix": str(override_runtime.parents[1]),
        "prefix": str(override_runtime.parents[1]),
        "path": [str(override_runtime.parents[1])],
        "packages": {
            dist: {"importable": True, "version": "1.2.3"}
            for dist in eh._DISTRIBUTION_MODULES
        },
    }
    script = f"#!/bin/sh\ncat <<'JSON'\n{json.dumps(report)}\nJSON\n"
    override_runtime.write_text(script)
    override_runtime.chmod(0o755)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": ["vi", "rapid-mlx", "serve"],
                "create_time": 123.0,
            }

        def exe(self):
            return "/bin/sh"

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(override_runtime))

    section = eh.section_python()
    runtime_row = next(
        c for c in section.checks if "Selected diagnostic runtime" in c.label
    )

    assert str(override_runtime) in runtime_row.detail
    assert "system environment" in runtime_row.label


def test_module_serve_process_is_selected(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.write_text("")
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")
    report = {
        "executable": str(server_runtime),
        "base_prefix": str(server_runtime.parents[1]),
        "prefix": str(server_runtime.parents[1]),
        "path": [str(server_runtime.parents[1])],
        "packages": {},
    }
    script = f"#!/bin/sh\ncat <<'JSON'\n{json.dumps(report)}\nJSON\n"
    server_runtime.write_text(script)
    server_runtime.chmod(0o755)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(server_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                    "test-model",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return dict(os.environ)

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    section = eh.section_python()
    runtime_row = next(c for c in section.checks if "Active server runtime" in c.label)

    assert str(server_runtime) in runtime_row.detail


def test_path_launched_entrypoint_process_is_selected(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.write_text("")
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")
    entrypoint = tmp_path / "bin" / "rapid-mlx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        f"#!{server_runtime}\nfrom vllm_mlx.cli import main\nsys.exit(main())\n"
    )
    entrypoint.chmod(0o755)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [str(server_runtime), "rapid-mlx", "serve"],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return {"PATH": str(entrypoint.parent)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == server_runtime


def test_indented_generated_entrypoint_process_is_selected(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.write_text("")
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")
    entrypoint = tmp_path / "bin" / "rapid-mlx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        f"#!{server_runtime}\nfrom vllm_mlx.cli import main\n\n"
        'if __name__ == "__main__":\n    sys.exit(main())\n'
    )
    entrypoint.chmod(0o755)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [str(server_runtime), str(entrypoint), "serve"],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return {"PATH": str(entrypoint.parent)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == server_runtime


def test_env_shebang_entrypoint_process_is_selected(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    entrypoint = tmp_path / "bin" / "rapid-mlx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        "#!/usr/bin/env python3\nfrom vllm_mlx.cli import main\nmain()\n"
    )
    runtime = tmp_path / "bin" / "python3"
    runtime.write_text("")
    runtime.chmod(0o755)
    (runtime.parent.parent / "pyvenv.cfg").write_text("")

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [str(entrypoint), "serve"],
                "create_time": 123.0,
            }

        def exe(self):
            return str(runtime)

        def environ(self):
            return {"PATH": str(runtime.parent)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == runtime


def test_newest_server_context_is_selected_for_a_shared_runtime(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    server_runtime.write_text("")
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")

    class FakeProcess:
        def __init__(self, create_time, context_id, cwd):
            self.info = {
                "pid": os.getpid() + 1 + int(create_time),
                "cmdline": [
                    str(server_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                ],
                "create_time": create_time,
            }
            self.context_id = context_id
            self.cwd_path = cwd

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return {"RAPID_MLX_TEST_CONTEXT": self.context_id}

        def cwd(self):
            return self.cwd_path

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    old_cwd = tmp_path / "old"
    new_cwd = tmp_path / "new"
    old_cwd.mkdir()
    new_cwd.mkdir()
    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [
        FakeProcess(1.0, "old", old_cwd),
        FakeProcess(2.0, "new", new_cwd),
    ]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_python_path() == server_runtime


def test_remote_runtime_probe_uses_an_allowlisted_environment(
    tmp_path,
    monkeypatch,
    allow_rapid_mlx_module_servers,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = Path(sys.executable)
    server_env_path = tmp_path / "server-context-packages"
    script = (
        "import json, os, sys; [sys.path.insert(0, root) for root in "
        "json.loads(sys.argv[2])['trusted'] + json.loads(sys.argv[2])['context']]; "
        "print(json.dumps({'executable': sys.executable, "
        "'base_prefix': sys.base_prefix, 'prefix': sys.prefix, 'path': sys.path, "
        "'packages': {}, 'environment': dict(os.environ)}))"
    )
    monkeypatch.setattr(eh, "_PROBE_SCRIPT", script)

    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": [
                    str(server_runtime),
                    "-m",
                    "vllm_mlx.cli",
                    "serve",
                ],
                "create_time": 123.0,
            }

        def exe(self):
            return str(server_runtime)

        def environ(self):
            return {"PYTHONPATH": str(server_env_path)}

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return SimpleNamespace(real=os.getuid())

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setenv("PYTHONPATH", "/attacker-controlled")
    monkeypatch.setenv("PYTHONHOME", "/attacker-home")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/attacker.dylib")
    server_env_path = tmp_path / "server-context-packages"
    monkeypatch.setenv("RAPID_MLX_TEST_SERVER_PYTHONPATH", str(server_env_path))
    monkeypatch.setattr(
        eh,
        "_RUNTIME_CONTEXTS",
        {server_runtime: (tmp_path, {"PYTHONPATH": str(server_env_path)})},
    )

    eh.section_python()
    probe = eh._probe_runtime(server_runtime)

    assert probe is not None
    environment = probe["environment"]
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "DYLD_INSERT_LIBRARIES" not in environment
    assert str(server_env_path) in probe["path"]
    assert str(tmp_path) in probe["path"]


def test_server_context_relative_pythonpath_uses_server_cwd(tmp_path, monkeypatch):
    server_runtime = Path(sys.executable).resolve()
    server_cwd = tmp_path / "server"
    relative_path = tmp_path / "relative"
    server_cwd.mkdir()
    relative_path.mkdir()
    monkeypatch.setattr(
        eh,
        "_RUNTIME_CONTEXTS",
        {
            server_runtime: (
                server_cwd,
                {"PYTHONPATH": f"relative{os.pathsep}/absolute"},
            )
        },
    )

    assert eh._server_import_paths(server_runtime) == [
        server_cwd.resolve(),
        (server_cwd / "relative").resolve(),
        Path("/absolute"),
    ]


def test_remote_system_runtime_is_not_mislabeled_as_a_virtual_environment(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "runtime" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    (runtime.parents[1] / "pyvenv.cfg").write_text("")
    report = {
        "executable": str(runtime),
        "base_prefix": str(runtime.parents[1]),
        "prefix": str(runtime.parents[1]),
        "path": [str(runtime.parents[1])],
        "packages": {},
    }
    script = f"#!/bin/sh\ncat <<'JSON'\n{json.dumps(report)}\nJSON\n"
    runtime.write_text(script)
    runtime.chmod(0o755)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(runtime))

    section = eh.section_python()
    runtime_row = next(
        c for c in section.checks if "Selected diagnostic runtime" in c.label
    )

    assert "system environment" in runtime_row.label


def test_remote_importable_package_without_metadata_is_a_warning(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")
    packages = {
        dist: {"importable": False, "version": None}
        for dist in eh._DISTRIBUTION_MODULES
    }
    packages["transformers"] = {"importable": True, "version": None}
    report = {
        "executable": str(server_runtime),
        "prefix": str(server_runtime.parents[1]),
        "path": [str(server_runtime.parents[1]), "/runtime-site-packages"],
        "packages": packages,
    }
    script = f"#!/bin/sh\ncat <<'JSON'\n{json.dumps(report)}\nJSON\n"
    server_runtime.write_text(script)
    server_runtime.chmod(0o755)

    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(server_runtime))
    section = eh.section_required_packages()
    row = next(c for c in section.checks if c.label.startswith("transformers"))
    assert row.status is eh.CheckStatus.WARN
    assert "importability" in row.label
    assert "not installed" not in row.label
    assert str(server_runtime) in row.detail


def test_remote_probe_reports_context_module_without_safe_import_verification(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    server_runtime = tmp_path / "server-runtime" / "bin" / "python"
    server_runtime.parent.mkdir(parents=True)
    context_root = tmp_path / "server-context"
    package_root = context_root / "probe-package"
    package_root.mkdir(parents=True)
    (package_root / "probe_runtime_package.py").write_text(
        "transformers_imported = True\n"
    )
    base_python = (
        Path(sys.base_prefix)
        / "bin"
        / f"python{sys.version_info[0]}.{sys.version_info[1]}"
    )
    if not base_python.is_file() or base_python == Path(sys.executable):
        pytest.skip("test requires a non-virtual baseline Python")
    (server_runtime.parents[1] / "pyvenv.cfg").write_text("")
    server_runtime.write_text(f'#!/bin/sh\nexec {str(base_python)!r} "$@"\n')
    server_runtime.chmod(0o755)
    monkeypatch.setitem(
        eh._DISTRIBUTION_MODULES, "transformers", "probe_runtime_package"
    )
    monkeypatch.setitem(eh._RUNTIME_PACKAGES, "transformers", "probe_runtime_package")
    monkeypatch.setattr(
        eh,
        "_RUNTIME_CONTEXTS",
        {
            server_runtime.resolve(): (
                context_root,
                {"PYTHONPATH": str(package_root)},
            )
        },
    )
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(server_runtime))
    monkeypatch.setattr(
        eh,
        "_module_visibility",
        lambda dist, runtime=None: (True, False),
    )

    section = eh.section_required_packages()
    row = next(c for c in section.checks if c.label.startswith("transformers"))

    assert row.status is eh.CheckStatus.WARN
    assert "importability cannot be verified" in row.label
    assert "not installed" not in row.label
    assert str(server_runtime) in row.detail


def test_remote_probe_accepts_trusted_single_file_module(
    tmp_path,
    monkeypatch,
):
    sidecar_root = tmp_path / "sidecar"
    site_root = sidecar_root / "site-packages"
    site_root.mkdir(parents=True)
    (site_root / "probe_single_file.py").write_text("probe_loaded = True\n")
    runtime = Path(sys.executable).resolve()
    monkeypatch.setattr(eh, "_RUNTIME_PACKAGES", {"transformers": "probe_single_file"})
    probe = eh._probe_runtime(
        runtime,
        sidecar_root=sidecar_root,
    )

    assert probe is not None
    package = cast("dict[str, object]", probe["packages"]["transformers"])
    assert package["discoverable"] is True
    assert package["trusted_origin"] is True
    assert package["importable"] is None
    assert eh._runtime_module_importable(runtime, "probe_single_file", sidecar_root)
    assert package["version"] is None


def test_remote_import_probe_ignores_dependency_stdout_banners(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = Path(sys.executable)
    sidecar_root = tmp_path / "sidecar"
    site_root = sidecar_root / "site-packages"
    site_root.mkdir(parents=True)
    (site_root / "probe_banner.py").write_text(
        "print('dependency startup banner')\nprobe_loaded = True\n"
    )
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    assert eh._runtime_module_importable(runtime, "probe_banner", sidecar_root)


def test_remote_probe_does_not_report_unbound_spec_on_bad_finder(tmp_path):
    finder_path = tmp_path / "bad_finder.py"
    finder_path.write_text(
        "class BadFinder:\n"
        "    def find_spec(self, _name, _path=None, _target=None):\n"
        "        raise RuntimeError('malformed package finder')\n"
        "\n"
        "def install():\n"
        "    import sys\n"
        "    sys.meta_path.insert(0, BadFinder())\n"
    )
    wrapper = (
        "import json\n"
        "from pathlib import Path\n"
        "import sys\n"
        "import importlib.metadata\n"
        "import importlib.util\n"
        "import bad_finder\n"
        "bad_finder.install()\n"
        f"exec(compile({eh._PROBE_SCRIPT!r}, '<probe>', 'exec'))\n"
    )
    result = json.loads(
        subprocess.check_output(  # noqa: S603
            [
                sys.executable,
                "-c",
                wrapper,
                '{"bad-package": "probe_bad_finder"}',
                '{"trusted": [], "context": []}',
            ],
            text=True,
            cwd="/",
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
        )
    )
    assert result["packages"]["bad-package"] == {
        "importable": None,
        "discoverable": False,
        "trusted_origin": False,
        "module": "probe_bad_finder",
        "version": None,
    }


def test_context_module_without_metadata_is_visible_but_unverified(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "runtime" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    context_root = tmp_path / "server-context"
    context_root.mkdir()
    (context_root / "probe_context_module.py").write_text("probe_loaded = True\n")
    report = {
        "executable": str(runtime),
        "base_prefix": str(tmp_path),
        "prefix": str(tmp_path),
        "path": [str(context_root)],
        "packages": {
            "transformers": {
                "importable": False,
                "discoverable": True,
                "trusted_origin": False,
                "module": "probe_context_module",
                "version": None,
            }
        },
    }
    runtime.write_text(f"#!/bin/sh\ncat <<'JSON'\n{json.dumps(report)}\nJSON\n")
    runtime.chmod(0o755)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_RUNTIME_PROBE_CACHE", {})

    visible = eh._visible_without_metadata(
        "transformers",
        runtime,
    )

    assert visible is True


def test_failed_remote_runtime_probe_is_one_explicit_failure(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    unavailable_runtime = tmp_path / "server-runtime" / "bin" / "python"

    eh._RUNTIME_PROBE_CACHE.clear()
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: unavailable_runtime)

    section = eh.section_required_packages()

    assert len(section.checks) == 1
    assert section.checks[0].status is eh.CheckStatus.FAIL
    assert "Could not inspect the active server runtime" in section.checks[0].label
    assert str(unavailable_runtime) in section.checks[0].detail
    assert "RAPID_MLX_RUNTIME_PYTHON" in section.checks[0].detail
    assert all("not installed" not in check.label for check in section.checks)


def test_failed_remote_runtime_probe_is_one_optional_failure(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    unavailable_runtime = tmp_path / "server-runtime" / "bin" / "python"
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: unavailable_runtime)

    section = eh.section_optional_packages()

    assert len(section.checks) == 1
    assert section.checks[0].status is eh.CheckStatus.FAIL
    assert "Could not inspect the active server runtime" in section.checks[0].label
    assert str(unavailable_runtime) in section.checks[0].detail
    assert all("not installed" not in check.label for check in section.checks)


def test_remote_pillow_probe_rejects_a_module_that_cannot_import(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = Path(sys.executable)
    package_root = tmp_path / "broken-pillow"
    (package_root / "site-packages" / "PIL").mkdir(parents=True)
    (package_root / "site-packages" / "PIL" / "__init__.py").write_text("")
    (package_root / "site-packages" / "PIL" / "Image.py").write_text(
        "raise ImportError('PIL broken')\n"
    )
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_bundled_sidecar_root", lambda _runtime=None: package_root)
    monkeypatch.setattr(
        eh,
        "_safe_version",
        lambda distribution, runtime=None: (
            "0.6.17" if distribution == "mlx-vlm" else None
        ),
    )

    try:
        section = eh.section_optional_packages()
        row = next(c for c in section.checks if c.label.startswith("mlx-vlm (vision"))
    finally:
        eh._RUNTIME_PROBE_CACHE.pop(runtime, None)

    assert row.status is eh.CheckStatus.WARN
    assert "Pillow (PIL) missing or broken" in row.label


def test_remote_pillow_probe_accepts_a_successful_image_exercise(
    tmp_path,
    monkeypatch,
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = Path(sys.executable)
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(
        eh,
        "_safe_version",
        lambda distribution, runtime=None: (
            "0.6.17" if distribution == "mlx-vlm" else None
        ),
    )
    monkeypatch.setattr(
        eh,
        "_module_visibility",
        lambda dist, runtime=None: (True, True),
    )

    try:
        section = eh.section_optional_packages()
        row = next(c for c in section.checks if c.label.startswith("mlx-vlm (vision"))
    finally:
        eh._RUNTIME_PROBE_CACHE.pop(runtime, None)

    assert row.status is eh.CheckStatus.OK
    assert "Pillow" not in row.label


def test_missing_optional_package_marks_warning():
    """A missing optional package is ⚠ with an install hint, never ✗."""

    def fake_ver(dist: str, runtime=None) -> str | None:
        # mlx-audio missing; the rest present.
        return None if dist == "mlx-audio" else "1.0.0"

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_module_available", return_value=False),
    ):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if "mlx-audio" in c.label)
    assert audio_row.status is eh.CheckStatus.WARN
    assert "pip install" in audio_row.label  # hint preserved


def test_importable_mlx_vlm_without_metadata_is_not_reported_missing():
    with (
        mock.patch.object(eh, "_safe_version", return_value=None),
        mock.patch.object(
            eh,
            "_module_available",
            side_effect=lambda module, _runtime=None, *, real_import=False: (
                module == "mlx_vlm"
            ),
        ),
    ):
        section = eh.section_optional_packages()

    row = next(c for c in section.checks if c.label.startswith("mlx-vlm (vision"))
    assert row.status is eh.CheckStatus.WARN
    assert "importable" in row.label
    assert "not installed" not in row.label


def test_incompatible_mlx_vlm_names_bounded_extension_repair(tmp_path):
    runtime = tmp_path / ".rapid-mlx" / "bin" / "python3"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.7.1" if dist == "mlx-vlm" else None

    with (
        mock.patch.object(eh.sys, "executable", str(runtime)),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
    ):
        section = eh.section_optional_packages()

    row = next(
        c
        for c in section.checks
        if c.label.startswith("mlx-vlm (vision") and "incompatible" in c.label
    )
    assert row.status is eh.CheckStatus.FAIL
    assert "requires ==0.6.17" in row.label
    assert "rapid-mlx[vision]" in row.label
    assert "transformers>=5.0.0,!=5.13.0,<5.16" in row.label
    assert str(runtime.resolve()) in row.label


def test_compatible_mlx_vlm_is_accepted():
    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.6.17" if dist == "mlx-vlm" else None

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_pil_importable", return_value=True),
        mock.patch.object(eh, "_module_visibility", return_value=(True, True)),
    ):
        section = eh.section_optional_packages()

    row = next(c for c in section.checks if c.label.startswith("mlx-vlm (vision"))
    assert row.status is eh.CheckStatus.OK


def test_incompatible_mlx_vlm_does_not_mark_dflash_ok():
    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.7.1" if dist == "mlx-vlm" else None

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_pil_importable", return_value=True),
    ):
        section = eh.section_optional_packages()

    dflash_row = next(c for c in section.checks if "dflash" in c.label)
    assert dflash_row.status is not eh.CheckStatus.OK
    assert "incompatible" in dflash_row.label


def test_run_all_bounds_remote_probe_budget_and_clears_deadline(monkeypatch):
    seen_deadlines = []

    def inspect_deadline():
        seen_deadlines.append(eh._DOCTOR_DEADLINE)
        return eh.Section("Test")

    monkeypatch.setattr(eh, "_SECTION_BUILDERS", (inspect_deadline,))

    report = eh.run_all()

    assert report.sections[0].title == "Test"
    assert len(seen_deadlines) == 1
    assert seen_deadlines[0] is not None
    assert eh._DOCTOR_DEADLINE is None


def test_runtime_process_scan_stops_when_doctor_budget_expires(tmp_path, monkeypatch):
    runtime_override = tmp_path / "override" / "python3"
    runtime_override.parent.mkdir(parents=True)
    runtime_override.write_text("")
    monkeypatch.setenv("RAPID_MLX_RUNTIME_PYTHON", str(runtime_override))
    monkeypatch.setattr(eh, "_DOCTOR_DEADLINE", time.monotonic() - 1)

    class FakeProcess:
        info = {"pid": os.getpid() + 1, "cmdline": [], "create_time": 1.0}

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FakeProcess()]
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert eh._runtime_python_path() == runtime_override.absolute()
    fake_psutil.process_iter.assert_called_once()


def test_signed_sidecar_receives_no_pip_remediation(tmp_path):
    runtime = _stage_sidecar_bundle(tmp_path)

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.7.1" if dist == "mlx-vlm" else None

    with (
        mock.patch.object(eh.sys, "executable", str(runtime)),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
    ):
        section = eh.section_optional_packages()

    row = next(
        c
        for c in section.checks
        if c.label.startswith("mlx-vlm (vision") and "incompatible" in c.label
    )
    assert row.status is eh.CheckStatus.FAIL
    assert "reinstall Rapid-MLX Desktop.app" in row.label
    assert "pip install" not in row.label
    assert "-m pip" not in row.label
    assert str(runtime.resolve()) not in row.label


def test_unsupported_mlx_audio_version_marks_warning():
    """A transitive mlx-audio outside Rapid-MLX's pin is not healthy."""

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.4.6" if dist == "mlx-audio" else None

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if "mlx-audio" in c.label)
    assert audio_row.status is eh.CheckStatus.WARN
    assert "0.4.6" in audio_row.label
    assert "requires mlx-audio>=0.2.9,<0.4.4" in audio_row.label


def test_supported_mlx_audio_version_marks_ok():
    """A version inside the declared audio range remains healthy."""

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(eh, "_module_available", return_value=True),
    ):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if "mlx-audio" in c.label)
    assert audio_row.status is eh.CheckStatus.OK


def test_absent_audio_dependency_stack_marks_warning():
    """An absent audio extra remains one concise optional-package warning."""
    with mock.patch.object(eh, "_safe_version", return_value=None):
        section = eh.section_optional_packages()

    audio_rows = [c for c in section.checks if "mlx-audio" in c.label]
    assert len(audio_rows) == 1
    assert audio_rows[0].status is eh.CheckStatus.WARN


def test_healthy_complete_audio_dependency_stack_marks_ok():
    """When every audio dependency imports and mlx-audio is in range, all
    audio rows are OK."""
    with (
        mock.patch.object(eh, "_safe_version", return_value="0.4.3"),
        mock.patch.object(eh, "_module_available", return_value=True),
    ):
        section = eh.section_optional_packages()

    audio_rows = [
        c
        for c in section.checks
        if "audio" in c.label.lower() or "mlx-audio" in c.label
    ]
    assert audio_rows
    assert all(c.status is eh.CheckStatus.OK for c in audio_rows), [
        (c.label, c.status) for c in audio_rows
    ]


def test_incomplete_audio_dependency_import_stack_marks_warning():
    """A present mlx-audio with a missing/broken audio dependency import is
    WARN, not OK — the audio feature set is not actually usable."""

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    with (
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh,
            "_module_available",
            side_effect=lambda module, _runtime=None, *, real_import=False: (
                module != "f5_tts_mlx"
            ),
        ),
    ):
        section = eh.section_optional_packages()

    broken = next(c for c in section.checks if "f5-tts-mlx" in c.label)
    assert broken.status is eh.CheckStatus.WARN


def _stage_sidecar_bundle(tmp_path: Path, *, slot: str = "embedded") -> Path:
    """Build the on-disk shape ``build-sidecar.sh`` produces and return the
    interpreter.

    ``slot`` picks which managed location it sits in — ``embedded`` mirrors
    ``Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/`` and
    ``runtime-override`` mirrors ``~/Library/Application Support/Rapid/
    runtime-override/rapid-mlx/``. Both share the identical bundle layout.
    """
    if slot == "runtime-override":
        parent = (
            tmp_path / "Library" / "Application Support" / "Rapid" / "runtime-override"
        )
    else:
        parent = tmp_path / "Rapid-MLX Desktop.app" / "Contents" / "Resources"
        parent.parent.mkdir(parents=True)
        with (parent.parent / "Info.plist").open("wb") as handle:
            plistlib.dump({"CFBundleIdentifier": "com.rapidmlx.rapid"}, handle)
    root = parent / "rapid-mlx"
    (root / "site-packages").mkdir(parents=True)
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "rapid-mlx").write_text("#!/bin/sh\n")
    exe = root / "python" / "bin" / "python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    return exe


def test_bundled_sidecar_detected_from_layout(tmp_path: Path):
    """The bundle is fingerprinted off ``sys.executable``'s directory shape."""
    exe = _stage_sidecar_bundle(tmp_path)
    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() == exe.parents[2].resolve()


def test_ordinary_install_is_not_treated_as_bundled_sidecar(tmp_path: Path):
    """A venv interpreter has no ``python/bin`` + ``site-packages`` sibling
    shape, so CLI installs keep the full-extra contract."""
    exe = tmp_path / "venv" / "bin" / "python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() is None


def test_custom_install_with_sidecar_shape_is_not_treated_as_desktop(
    tmp_path: Path,
):
    """The three internal bundle paths are user-creatable and therefore do
    not prove that Desktop owns the environment without a managed location."""
    root = tmp_path / "custom-python" / "rapid-mlx"
    (root / "site-packages").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "bin" / "rapid-mlx").write_text("#!/bin/sh\n")
    exe = root / "python" / "bin" / "python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")

    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() is None


def test_unrelated_app_with_sidecar_shape_is_not_treated_as_desktop(
    tmp_path: Path,
):
    """A generic app-bundle suffix is not Rapid-MLX provenance."""
    exe = _stage_sidecar_bundle(tmp_path)
    info = exe.parents[4] / "Info.plist"
    with info.open("wb") as handle:
        plistlib.dump({"CFBundleIdentifier": "com.example.other-app"}, handle)

    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() is None


def test_non_mapping_info_plist_is_not_treated_as_desktop(tmp_path: Path):
    """A valid plist may have a non-dictionary root and must not crash doctor."""
    exe = _stage_sidecar_bundle(tmp_path)
    info = exe.parents[4] / "Info.plist"
    with info.open("wb") as handle:
        plistlib.dump(["not", "a", "bundle", "dictionary"], handle)

    with mock.patch.object(eh.sys, "executable", str(exe)):
        assert eh._bundled_sidecar_root() is None


def test_runtime_override_must_belong_to_active_home(tmp_path: Path):
    """A matching suffix under another home is not Desktop provenance."""
    foreign_home = tmp_path / "foreign-home"
    exe = _stage_sidecar_bundle(foreign_home, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path / "active-home")}),
    ):
        assert eh._bundled_sidecar_root() is None


def test_bundled_sidecar_grades_audio_against_desktop_extra(tmp_path: Path):
    """RC 0.12.18: the signed bundle installs ``[audio-desktop]`` (mlx-audio +
    soundfile only). Grading it against the full ``[audio]`` closure reported a
    healthy build as incomplete."""

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    # Everything outside the audio-desktop extra is absent, exactly like the
    # real bundle.
    desktop_modules = {module for _, module in eh._AUDIO_DESKTOP_IMPORTS}
    exe = _stage_sidecar_bundle(tmp_path)

    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh,
            "_module_available",
            side_effect=lambda m, _runtime=None, *, real_import=False: (
                m in desktop_modules
            ),
        ),
    ):
        section = eh.section_optional_packages()

    audio_row = next(c for c in section.checks if c.label.startswith("mlx-audio"))
    assert audio_row.status is eh.CheckStatus.OK
    assert "incomplete" not in audio_row.label


def test_bundled_sidecar_never_recommends_pip_install(tmp_path: Path):
    """Mutating a managed sidecar's Python environment is never the right
    advice, so no row may hand the user a ``pip install`` line."""
    exe = _stage_sidecar_bundle(tmp_path)
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", return_value=None),
    ):
        section = eh.section_optional_packages()

    offenders = [
        (c.label, c.detail)
        for c in section.checks
        if "pip install 'rapid-mlx[" in c.label + c.detail
    ]
    assert not offenders, offenders


def test_bundle_does_not_ask_user_to_reinstall_for_an_extra_it_never_ships(
    tmp_path: Path,
):
    """mlx-embeddings is outside the desktop product surface — ``build-
    sidecar.sh`` never installs it. Flagging it ⚠ with a "reinstall" hint
    reported a healthy bundle as broken and the warning would survive every
    reinstall."""
    exe = _stage_sidecar_bundle(tmp_path)
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", return_value=None),
    ):
        section = eh.section_optional_packages()

    row = next(c for c in section.checks if c.label.startswith("mlx-embeddings"))
    assert row.status is eh.CheckStatus.OK
    assert "not bundled" in row.label
    assert "reinstall" not in row.label.lower()


def test_cli_install_still_warns_about_missing_embeddings(tmp_path: Path):
    """The exclusion is bundle-only: an ordinary pip install that lacks
    mlx-embeddings must still get the ⚠ + install hint."""
    exe = tmp_path / "venv" / "bin" / "python3.12"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", return_value=None),
    ):
        section = eh.section_optional_packages()

    row = next(c for c in section.checks if c.label.startswith("mlx-embeddings"))
    assert row.status is eh.CheckStatus.WARN
    assert "pip install" in row.label
    assert "rapid-mlx[embeddings]" in row.label


def test_embedded_bundle_hint_names_the_shipping_app(tmp_path: Path):
    """``Rapid.app`` is a legacy name that release verification rejects; the
    shipping product is ``Rapid-MLX Desktop.app``."""
    exe = _stage_sidecar_bundle(tmp_path, slot="embedded")
    with mock.patch.object(eh.sys, "executable", str(exe)):
        hint = eh._sidecar_repair_hint(eh._bundled_sidecar_root())

    assert "Rapid-MLX Desktop.app" in hint


def test_runtime_override_is_not_told_to_reinstall_the_app(tmp_path: Path):
    """The runtime override lives outside the app bundle, survives desktop
    upgrades, and the bootstrapper skips its download while the cache exists —
    so "reinstall the .app" alone repairs nothing."""
    exe = _stage_sidecar_bundle(tmp_path, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path)}),
    ):
        root = eh._bundled_sidecar_root()
        hint = eh._sidecar_repair_hint(root)

    assert "outside the app bundle" in hint
    # The step has to be actionable, so it names the exact directory to remove.
    assert str(root) in hint


def test_runtime_override_hint_avoids_the_nonexistent_update_entry_point(
    tmp_path: Path,
):
    """Settings → check for updates hands off to Sparkle, which updates the app
    bundle; ``UpdateChecker`` explicitly does not act on the manifest's
    ``sidecar_*`` fields. Pointing users there for a broken runtime override is
    a dead-end recovery flow."""
    exe = _stage_sidecar_bundle(tmp_path, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path)}),
    ):
        hint = eh._sidecar_repair_hint(eh._bundled_sidecar_root())

    assert "check for updates" not in hint.lower()


def test_runtime_override_hint_does_not_promise_an_automatic_reinstall(
    tmp_path: Path,
):
    """Nothing re-downloads a sidecar today: the bootstrapper module is not in
    the source tree and the release workflow builds only the full DMG. A slim
    install that merely deletes the override lands on the missing-runtime
    overlay, whose only actions are Recheck and an *app* update download. So
    the hint must order "install the app that ships a sidecar" BEFORE the
    removal, and must not claim relaunching reinstalls the runtime."""
    exe = _stage_sidecar_bundle(tmp_path, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path)}),
    ):
        root = eh._bundled_sidecar_root()
        hint = eh._sidecar_repair_hint(root)

    # No promise of self-repair.
    assert "reinstalls it" not in hint
    assert "then reinstalls" not in hint
    # Install-first ordering: the app install must precede the removal step,
    # otherwise a slim user is stranded with no sidecar at all.
    assert hint.index("Rapid-MLX Desktop.app") < hint.index(f"remove {root}")


def test_runtime_override_broken_audio_row_uses_the_runtime_hint(tmp_path: Path):
    """End-to-end: a genuinely broken override bundle is still ⚠, and the
    remediation the user reads is the runtime one, not the app one."""

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    exe = _stage_sidecar_bundle(tmp_path, slot="runtime-override")
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.dict(eh.os.environ, {"HOME": str(tmp_path)}),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh,
            "_module_available",
            side_effect=lambda m, _runtime=None, *, real_import=False: m != "soundfile",
        ),
    ):
        section = eh.section_optional_packages()

    broken = next(c for c in section.checks if "soundfile" in c.label)
    assert broken.status is eh.CheckStatus.WARN
    assert "outside the app bundle" in broken.label


def test_bundled_sidecar_still_flags_a_genuinely_broken_audio_install(
    tmp_path: Path,
):
    """Narrowing the contract must not make the bundle unfalsifiable — a
    missing soundfile is still inside the audio-desktop contract."""

    def fake_ver(dist: str, runtime=None) -> str | None:
        return "0.4.3" if dist == "mlx-audio" else None

    exe = _stage_sidecar_bundle(tmp_path)
    with (
        mock.patch.object(eh.sys, "executable", str(exe)),
        mock.patch.object(eh, "_safe_version", side_effect=fake_ver),
        mock.patch.object(
            eh,
            "_module_available",
            side_effect=lambda m, _runtime=None, *, real_import=False: m != "soundfile",
        ),
    ):
        section = eh.section_optional_packages()

    broken = next(c for c in section.checks if "soundfile" in c.label)
    assert broken.status is eh.CheckStatus.WARN
    assert "reinstall Rapid-MLX Desktop.app" in broken.label


# ---------------------------------------------------------------------------
# Section: HuggingFace cache
# ---------------------------------------------------------------------------


def test_hf_cache_writable_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A writable cache dir produces OK; a missing one produces WARN."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "hub").mkdir()
    section = eh.section_hf_cache()
    writable_row = section.checks[0]
    assert writable_row.status is eh.CheckStatus.OK
    assert "writable" in writable_row.label

    # Now point HF_HOME at a non-existent dir; first row should WARN.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "doesnotexist"))
    section = eh.section_hf_cache()
    missing_row = section.checks[0]
    assert missing_row.status is eh.CheckStatus.WARN
    assert "does not exist" in missing_row.label


def test_hf_cache_readonly_marks_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A readonly cache dir is a hard FAIL — downloads can't proceed."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    cache_root = tmp_path / "ro"
    cache_root.mkdir()
    (cache_root / "hub").mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_root))
    # mock os.access so the test doesn't depend on chmod semantics across CI.
    with mock.patch.object(eh.os, "access", return_value=False):
        section = eh.section_hf_cache()
    first = section.checks[0]
    assert first.status is eh.CheckStatus.FAIL
    assert "NOT writable" in first.label


def test_hf_cache_resolves_hf_hub_cache_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``$HF_HUB_CACHE`` wins over ``$HF_HOME`` over the default — matches
    huggingface_hub itself. Codex review round 1 caught the previous
    revision returning ``~/.cache/huggingface`` instead of the actual
    ``~/.cache/huggingface/hub`` subdir where downloads land."""
    hub = tmp_path / "external_ssd_hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    # HF_HOME is *also* set, to a different (writable) path — HF_HUB_CACHE
    # must take precedence.
    other_home = tmp_path / "wrong_home"
    other_home.mkdir()
    (other_home / "hub").mkdir()
    monkeypatch.setenv("HF_HOME", str(other_home))

    resolved = eh._hf_cache_dir()
    assert resolved == hub, (
        f"HF_HUB_CACHE should win over HF_HOME; resolved {resolved} != {hub}"
    )


def test_hf_cache_default_includes_hub_subdir(monkeypatch: pytest.MonkeyPatch):
    """Default (no env vars) resolution must include the trailing ``hub``
    segment — that's where huggingface_hub actually writes downloads."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    resolved = eh._hf_cache_dir()
    assert resolved.name == "hub", (
        f"default cache dir should end in .../huggingface/hub; got {resolved}"
    )


def test_dir_size_walk_aborts_on_budget(tmp_path: Path):
    """A walk that exceeds ``budget_s`` returns None rather than running
    indefinitely — keeps doctor under its 5 s wall-clock contract on
    network-mounted caches. Codex review round 1 flagged the unbounded
    walk as a contract violation."""
    # Populate a tiny tree so there's something to walk.
    for i in range(3):
        (tmp_path / f"f{i}.bin").write_bytes(b"\0" * 1024)
    # budget_s=0 forces an immediate abort on the first deadline check.
    result = eh._dir_size_gb(tmp_path, budget_s=0.0)
    assert result is None, (
        f"zero-budget walk should return None, got {result!r} — "
        "the deadline guard isn't firing"
    )


def test_dir_size_walk_aborts_inside_flat_directory(tmp_path: Path):
    """Flat directory with many files must respect the per-file deadline,
    not just the per-directory one. HF cache's ``blobs/`` subdir is the
    real-world example: thousands of files in a single dir; a per-dir
    deadline check would let a single cold-cache stat() storm blow past
    the budget. Codex review round 2 caught this; the per-file check
    fixes it."""
    # 500 files in one flat dir.
    for i in range(500):
        (tmp_path / f"f{i:04d}.bin").write_bytes(b"\0")
    # budget_s=0 must abort *before* iterating all 500 files.
    import time as _time

    t0 = _time.monotonic()
    result = eh._dir_size_gb(tmp_path, budget_s=0.0)
    elapsed = _time.monotonic() - t0
    assert result is None, "zero-budget flat-dir walk should return None"
    # Tight ceiling: must abort within a small multiple of one file's worth.
    assert elapsed < 0.5, (
        f"flat-dir walk took {elapsed:.3f}s with budget_s=0 — "
        "per-file deadline check isn't firing"
    )


def test_hf_cache_non_directory_marks_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``HF_HUB_CACHE`` pointing at a writable regular file must FAIL —
    ``os.access`` returns True for writable files too, so the previous
    revision would have shipped ✓ here. Codex review round 2 fixed."""
    file_target = tmp_path / "i_am_a_file_not_a_dir"
    file_target.write_text("oops")
    monkeypatch.setenv("HF_HUB_CACHE", str(file_target))

    section = eh.section_hf_cache()
    first = section.checks[0]
    assert first.status is eh.CheckStatus.FAIL
    assert "NOT a directory" in first.label


def test_hf_cache_missing_with_readonly_parent_marks_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Missing cache + readonly nearest-existing-parent → FAIL, not WARN.
    Codex review round 2: previous unconditional WARN exited 0 even when
    the first download was guaranteed to fail."""
    readonly_root = tmp_path / "readonly"
    readonly_root.mkdir()
    target = readonly_root / "nonexistent_hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(target))

    # Mock os.access so the test doesn't depend on chmod semantics across
    # CI runners (where the actual chmod may not stick in tmpfs).
    real_access = eh.os.access

    def fake_access(p, mode):
        if str(p) == str(readonly_root):
            return False  # parent isn't writable
        return real_access(p, mode)

    with mock.patch.object(eh.os, "access", side_effect=fake_access):
        section = eh.section_hf_cache()
    first = section.checks[0]
    assert first.status is eh.CheckStatus.FAIL
    assert "parent" in first.label and "NOT writable" in first.label


# ---------------------------------------------------------------------------
# Section: Network
# ---------------------------------------------------------------------------


def test_network_probe_timeout_is_warning_not_failure():
    """An unreachable huggingface.co must produce ⚠, never ✗.

    This is the contract that lets air-gapped CI runners still get a
    green ``rapid-mlx doctor`` (the spec is explicit about this).
    """

    def fake_probe() -> tuple[eh.CheckStatus, str]:
        return eh.CheckStatus.WARN, "TimeoutError: timed out after 2.0s"

    section = eh.section_network(probe=fake_probe)
    hf_row = next(c for c in section.checks if "huggingface.co" in c.label)
    assert hf_row.status is eh.CheckStatus.WARN
    # Spec: WARN never contributes to exit code, only FAIL does.
    assert hf_row.status is not eh.CheckStatus.FAIL


def test_network_probe_ok():
    def fake_probe() -> tuple[eh.CheckStatus, str]:
        return eh.CheckStatus.OK, "HTTP 200"

    section = eh.section_network(probe=fake_probe)
    hf_row = next(c for c in section.checks if "huggingface.co" in c.label)
    assert hf_row.status is eh.CheckStatus.OK


# ---------------------------------------------------------------------------
# Section: Shell integration
# ---------------------------------------------------------------------------


def test_argcomplete_not_in_rc_marks_warning(tmp_path: Path):
    """No rc file contains the argcomplete hook → WARN with activation hint."""
    fake_rc = tmp_path / ".zshrc"
    fake_rc.write_text("export PATH=$PATH:~/bin\n")  # no argcomplete hook

    section = eh.section_shell_integration(
        which=lambda name: "/usr/local/bin/rapid-mlx" if name == "rapid-mlx" else None,
        rcs=[fake_rc],
    )
    argc_row = next(c for c in section.checks if "argcomplete" in c.label)
    assert argc_row.status is eh.CheckStatus.WARN
    assert "register-python-argcomplete rapid-mlx" in argc_row.label


def test_argcomplete_present_marks_ok(tmp_path: Path):
    fake_rc = tmp_path / ".zshrc"
    fake_rc.write_text('eval "$(register-python-argcomplete rapid-mlx)"\n')

    section = eh.section_shell_integration(
        which=lambda name: "/usr/local/bin/rapid-mlx" if name == "rapid-mlx" else None,
        rcs=[fake_rc],
    )
    argc_row = next(c for c in section.checks if "argcomplete" in c.label)
    assert argc_row.status is eh.CheckStatus.OK


def test_rapid_mlx_not_on_path_marks_fail(tmp_path: Path):
    section = eh.section_shell_integration(
        which=lambda name: None,
        rcs=[tmp_path / "missing.zshrc"],
    )
    path_row = section.checks[0]
    assert path_row.status is eh.CheckStatus.FAIL
    assert "NOT on $PATH" in path_row.label


def test_running_exe_mismatch_with_path_warns(tmp_path: Path):
    """Issue #2352: PATH resolves a different rapid-mlx than the one actually
    running this doctor (e.g. a venv/bin that precedes a global install) →
    WARN with BOTH paths and an actionable fix, rather than a silently-green
    PATH row that points troubleshooting at the wrong install."""
    section = eh.section_shell_integration(
        which=lambda name: (
            "/Users/x/.local/bin/rapid-mlx" if name == "rapid-mlx" else None
        ),
        rcs=[tmp_path / "missing.zshrc"],
        running_exe="/private/tmp/env/bin/rapid-mlx",
    )
    mismatch = next(c for c in section.checks if "differs from the $PATH" in c.label)
    assert mismatch.status is eh.CheckStatus.WARN
    assert "/private/tmp/env/bin/rapid-mlx" in mismatch.label
    assert "/Users/x/.local/bin/rapid-mlx" in mismatch.label
    assert "activate this environment" in mismatch.label
    # The PATH row itself is still OK (it IS on PATH); only the divergence warns.
    path_row = next(c for c in section.checks if "in $PATH" in c.label)
    assert path_row.status is eh.CheckStatus.OK


def test_running_exe_matching_path_has_no_mismatch_warn(tmp_path: Path):
    """When the running CLI and the PATH-resolved CLI agree, no mismatch warn."""
    section = eh.section_shell_integration(
        which=lambda name: (
            "/private/tmp/env/bin/rapid-mlx" if name == "rapid-mlx" else None
        ),
        rcs=[tmp_path / "missing.zshrc"],
        running_exe="/private/tmp/env/bin/rapid-mlx",
    )
    assert not any("differs from the $PATH" in c.label for c in section.checks)


def test_running_cli_exe_prefers_argv0(monkeypatch):
    """``_running_cli_exe`` uses sys.argv[0] (the console-script path) when it
    resolves to a `rapid-mlx` entry point."""
    monkeypatch.setattr(eh.sys, "argv", ["/opt/venv/bin/rapid-mlx", "doctor"])
    assert eh._running_cli_exe() == "/opt/venv/bin/rapid-mlx"


def test_running_cli_exe_falls_back_to_executable_sibling(monkeypatch):
    """When sys.argv[0] isn't a `rapid-mlx` script (e.g. python -m), fall back
    to the `rapid-mlx` console script beside sys.executable."""
    monkeypatch.setattr(eh.sys, "argv", ["-m", "rapid_mlx.doctor"])
    monkeypatch.setattr(eh.sys, "executable", "/opt/venv/bin/python")
    monkeypatch.setattr(eh.os.path, "exists", lambda p: p == "/opt/venv/bin/rapid-mlx")
    assert eh._running_cli_exe() == "/opt/venv/bin/rapid-mlx"


def test_running_cli_exe_uses_executable_when_named_rapid_mlx(monkeypatch):
    """If sys.executable itself is `rapid-mlx` (rare direct exec), return it."""
    monkeypatch.setattr(eh.sys, "argv", ["-m", "rapid_mlx.doctor"])
    monkeypatch.setattr(eh.sys, "executable", "/opt/venv/bin/rapid-mlx")
    assert eh._running_cli_exe() == "/opt/venv/bin/rapid-mlx"


# ---------------------------------------------------------------------------
# Exit-code aggregation
# ---------------------------------------------------------------------------


def _make_report(*statuses: eh.CheckStatus) -> eh.Report:
    report = eh.Report()
    section = eh.Section("test")
    for i, st in enumerate(statuses):
        section.add(f"check-{i}", st)
    report.sections.append(section)
    return report


def test_overall_exit_code_zero_when_no_issues():
    report = _make_report(eh.CheckStatus.OK, eh.CheckStatus.OK, eh.CheckStatus.WARN)
    assert report.exit_code == 0
    assert report.n_warn == 1
    assert report.n_fail == 0


def test_overall_exit_code_one_when_any_issue():
    report = _make_report(eh.CheckStatus.OK, eh.CheckStatus.WARN, eh.CheckStatus.FAIL)
    assert report.exit_code == 1
    assert report.n_fail == 1


def test_overall_exit_code_zero_with_only_warnings():
    report = _make_report(eh.CheckStatus.WARN, eh.CheckStatus.WARN)
    # Spec rule: warnings never affect exit code.
    assert report.exit_code == 0


# ---------------------------------------------------------------------------
# Top-level run_all + render smoke
# ---------------------------------------------------------------------------


def test_run_all_returns_all_sections():
    """run_all() must emit exactly the sections the spec mandates, in the spec
    order. Test pins the order so future drift is loud."""
    report = eh.run_all()
    titles = [s.title for s in report.sections]
    expected = [
        "System",
        "Python",
        "Required Packages",
        "Updates",
        "Optional Packages",
        "HuggingFace Cache",
        "Network",
        "Shell Integration",
        "Optional Tools",
        "Agent Integrations",
    ]
    assert titles == expected, (
        f"sections drifted from spec order. got {titles}, expected {expected}"
    )


def test_run_all_crashing_probe_does_not_abort_report(monkeypatch):
    """If a section builder crashes, run_all() records it as a single ✗
    row but keeps going. A buggy probe must not blank the whole report."""

    def boom() -> eh.Section:
        raise RuntimeError("synthetic")

    # Patch the second builder so we can confirm earlier sections still rendered.
    builders = list(eh._SECTION_BUILDERS)
    builders[1] = boom
    monkeypatch.setattr(eh, "_SECTION_BUILDERS", tuple(builders))

    report = eh.run_all()
    assert len(report.sections) == len(builders)
    crashed = report.sections[1]
    assert any("probe crashed" in c.label for c in crashed.checks)
    assert any(c.status is eh.CheckStatus.FAIL for c in crashed.checks)


def test_render_outputs_section_headers(capsys):
    """Sanity-check the renderer: every section title appears in the output."""
    report = _make_report(eh.CheckStatus.OK)
    report.sections[0].title = "MySection"

    import io

    from vllm_mlx.doctor.cli import render

    buf = io.StringIO()
    render(report, stream=buf)
    out = buf.getvalue()
    assert "MySection" in out
    assert "Summary:" in out
    assert "Rapid-MLX Doctor" in out


def test_render_verbose_includes_detail():
    from vllm_mlx.doctor.cli import render

    report = eh.Report()
    section = eh.Section("X")
    section.add("the-label", eh.CheckStatus.OK, detail="the-detail-string")
    report.sections.append(section)

    import io

    buf = io.StringIO()
    render(report, verbose=True, stream=buf)
    assert "the-detail-string" in buf.getvalue()

    buf2 = io.StringIO()
    render(report, verbose=False, stream=buf2)
    assert "the-detail-string" not in buf2.getvalue()


# ---------------------------------------------------------------------------
# Importable sanity — guarantees the module is wired into the public surface.
# ---------------------------------------------------------------------------


def test_env_health_public_exports():
    pkg = importlib.import_module("vllm_mlx.doctor")
    for name in ("run_all", "Report", "Section", "Check", "CheckStatus"):
        assert hasattr(pkg, name), f"vllm_mlx.doctor missing {name}"


# ---------------------------------------------------------------------------
# Section: Updates (version freshness)
# ---------------------------------------------------------------------------


def test_updates_up_to_date_marks_ok():
    section = eh.section_updates(
        installed=lambda: "0.10.15",
        fetch_latest=lambda: "0.10.15",
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.OK
    assert "up to date" in row.label


def test_update_available_marks_warn_with_upgrade_command():
    info = mock.Mock(upgrade_command="brew upgrade rapid-mlx", method="brew")
    section = eh.section_updates(
        installed=lambda: "0.10.12",
        fetch_latest=lambda: "0.10.15",
        install_info=info,
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    assert "update available: 0.10.15" in row.label
    assert "brew upgrade rapid-mlx" in row.label


def test_update_available_from_rc_to_final_marks_warn():
    """Doctor reuses the shared parser, so its existing Updates section is
    the single user-visible RC-to-final notice; the command entry point must
    not perform a duplicate network fetch before ``run_all()``."""
    info = mock.Mock(upgrade_command="rapid-mlx upgrade", method="pip")
    section = eh.section_updates(
        installed=lambda: "0.13.2rc2",
        fetch_latest=lambda: "0.13.2",
        install_info=info,
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    assert "update available: 0.13.2" in row.label
    assert "installed 0.13.2rc2" in row.label


def test_updates_offline_marks_warn_never_fail():
    section = eh.section_updates(
        installed=lambda: "0.10.15",
        fetch_latest=lambda: None,
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    # Air-gapped doctor must never escalate the update check to a hard fail.
    assert all(c.status is not eh.CheckStatus.FAIL for c in section.checks)


def test_updates_unknown_installed_version_marks_warn():
    section = eh.section_updates(
        installed=lambda: None,
        fetch_latest=lambda: "0.10.15",
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    assert "unknown" in row.label


@pytest.mark.parametrize(
    "installed,latest",
    [
        ("0.10", "0.10.15"),  # installed unparseable (too few components)
        ("0.10.15", "0.11.0a1"),  # latest alpha remains unsupported
        ("0.10.15+local", "0.10.16"),  # local version remains unsupported
    ],
)
def test_updates_unparseable_version_marks_warn_not_up_to_date(installed, latest):
    """A version ``_parse_version`` can't order (alpha/local/short) must
    NOT silently green-light "up to date" — that falsely reassures a user
    who may well be behind. It downgrades to ⚠ like every other uncertain
    branch, and never to a hard fail."""
    section = eh.section_updates(
        installed=lambda: installed,
        fetch_latest=lambda: latest,
    )
    row = section.checks[0]
    assert row.status is eh.CheckStatus.WARN
    assert "up to date" not in row.label
    assert all(c.status is not eh.CheckStatus.FAIL for c in section.checks)


# ---------------------------------------------------------------------------
# Section: Shell Integration — shadowed / duplicate installs
# ---------------------------------------------------------------------------


def test_shadowed_install_marks_warn(tmp_path: Path):
    section = eh.section_shell_integration(
        which=lambda name: (
            "/opt/homebrew/bin/rapid-mlx" if name == "rapid-mlx" else None
        ),
        rcs=[tmp_path / "missing.zshrc"],
        find_all=lambda: [
            "/opt/homebrew/bin/rapid-mlx",
            "/Users/x/.local/bin/rapid-mlx",
        ],
    )
    shadow_row = next(c for c in section.checks if "shadowed" in c.label)
    assert shadow_row.status is eh.CheckStatus.WARN
    assert "2 places" in shadow_row.label
    assert ".local/bin/rapid-mlx" in shadow_row.label


def test_single_install_has_no_shadow_warn(tmp_path: Path):
    section = eh.section_shell_integration(
        which=lambda name: (
            "/opt/homebrew/bin/rapid-mlx" if name == "rapid-mlx" else None
        ),
        rcs=[tmp_path / "missing.zshrc"],
        find_all=lambda: ["/opt/homebrew/bin/rapid-mlx"],
    )
    assert not any("shadowed" in c.label for c in section.checks)


def test_rapid_mlx_on_path_dedupes_by_resolved_target(tmp_path: Path):
    """A symlink to the same binary is one install; a distinct binary is two."""
    d1, d2, d3 = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (d1, d2, d3):
        d.mkdir()
    real = d1 / "rapid-mlx"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    (d2 / "rapid-mlx").symlink_to(real)  # same target → deduped
    other = d3 / "rapid-mlx"
    other.write_text("#!/bin/sh\n")
    other.chmod(0o755)

    path_env = os.pathsep.join(str(d) for d in (d1, d2, d3))
    found = eh._rapid_mlx_on_path(path_env=path_env)
    assert found == [str(real), str(other)]


def test_rapid_mlx_on_path_empty_component_is_cwd(tmp_path: Path, monkeypatch):
    """An empty PATH component means the current directory (shutil.which
    semantics). A ``./rapid-mlx`` shadow must be surfaced, not skipped."""
    cli = tmp_path / "rapid-mlx"
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    # Leading empty component ("" before the sep) resolves to cwd.
    found = eh._rapid_mlx_on_path(path_env=os.pathsep + "/nonexistent-doctor-dir")
    resolved = [os.path.realpath(p) for p in found]
    assert os.path.realpath(str(cli)) in resolved


def test_runtime_distribution_probe_honors_cache_deadline_and_errors(tmp_path):
    runtime = tmp_path / "python"
    runtime.write_text("")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            eh, "_RUNTIME_DISTRIBUTION_CACHE", {runtime.absolute(): True}
        )
        monkeypatch.setattr(
            eh.subprocess,
            "run",
            mock.Mock(side_effect=AssertionError("cached probe must not run")),
        )
        assert eh._runtime_has_rapid_mlx_distribution(runtime, tmp_path, {})
    finally:
        monkeypatch.undo()

    with mock.patch.object(eh, "_DOCTOR_DEADLINE", 0.0):
        assert not eh._runtime_has_rapid_mlx_distribution(runtime, tmp_path, {})

    with mock.patch.object(
        eh.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd=str(runtime), timeout=1),
    ):
        assert not eh._runtime_has_rapid_mlx_distribution(runtime, tmp_path, {})


def test_runtime_distribution_probe_validates_usr_bin_python(tmp_path):
    result = SimpleNamespace(stdout='["rapid-mlx"]\n')
    with mock.patch.object(eh.subprocess, "run", return_value=result):
        assert eh._runtime_has_rapid_mlx_distribution(
            Path("/usr/bin/python3"),
            tmp_path,
            {"PATH": str(tmp_path)},
        )


def test_module_probe_rejects_unprobeable_runtime(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: None)

    assert not eh._module_available("transformers", runtime)

    probe = {"packages": {"transformers": {"importable": True}}}
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: probe)
    monkeypatch.setattr(eh, "_runtime_module_importable", lambda *args, **kwargs: True)
    assert eh._module_available("transformers", runtime)


def test_module_probe_catches_all_unsafe_failures(monkeypatch):
    with mock.patch.object(eh._iu, "find_spec", side_effect=SystemExit):
        assert not eh._module_available("transformers")


def test_module_origin_uses_bundled_sidecar_as_trusted_root(tmp_path, monkeypatch):
    module = tmp_path / "trusted_probe_module.py"
    module.write_text("probe_loaded = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(eh, "_bundled_sidecar_root", lambda python=None: tmp_path)

    assert eh._module_origin_is_trusted("trusted_probe_module")


def test_trusted_sys_path_roots_ignore_empty_components(monkeypatch):
    monkeypatch.setattr(eh.sys, "path", ["", "relative"])

    assert eh._trusted_sys_path_roots() == set()


def test_filesystem_runtime_authentication_uses_sidecar_shape(tmp_path, monkeypatch):
    runtime = tmp_path / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    site_root = tmp_path / "site-packages"
    site_root.mkdir()
    (site_root / "vllm_mlx").mkdir()
    monkeypatch.setattr(eh, "_bundled_sidecar_root", lambda python=None: tmp_path)

    assert eh._filesystem_runtime_has_rapid_mlx_distribution(runtime)

    monkeypatch.setattr(eh, "_bundled_sidecar_root", lambda python=None: None)
    assert not eh._filesystem_runtime_has_rapid_mlx_distribution(runtime)


def test_bundled_sidecar_detection_fails_closed_on_resolve_error():
    with mock.patch.object(Path, "resolve", side_effect=OSError):
        assert eh._bundled_sidecar_root(Path(sys.executable)) is None


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (Path("/tmp/missing-python"), False),
        (Path("/tmp/not-python"), False),
        (Path(sys.executable), True),
    ],
)
def test_diagnostic_python_override_validates_shape(candidate, expected):
    assert eh._is_diagnostic_python_override(candidate) is expected


@pytest.mark.parametrize(
    "sidecar_root,exe,prefix,expected",
    [
        (None, Path("/opt/python"), Path("/opt"), "system environment"),
        (Path("/sidecar"), Path("/opt/python"), Path("/opt"), "desktop sidecar"),
    ],
)
def test_runtime_environment_classifies_sidecar_and_system(
    sidecar_root, exe, prefix, expected
):
    with mock.patch.object(eh, "_bundled_sidecar_root", return_value=sidecar_root):
        assert eh._runtime_environment(exe, prefix, prefix) == expected


def test_runtime_environment_classifies_all_layouts(tmp_path, monkeypatch):
    home = tmp_path / "home"
    application_bin = home / ".rapid-mlx" / "bin"
    runtime_root = home / ".rapid-mlx-python"
    application_python = application_bin / "python3"
    runtime_python = runtime_root / "bin" / "python3"
    developer_python = Path(eh.__file__).resolve().parents[2] / "bin" / "python3"
    application_bin.mkdir(parents=True)
    runtime_python.parent.mkdir(parents=True)
    monkeypatch.setattr(eh.Path, "home", lambda: home)

    assert (
        eh._runtime_environment(application_python, Path(sys.prefix))
        == "Rapid-MLX application environment"
    )
    assert (
        eh._runtime_environment(runtime_python, runtime_root)
        == "Rapid-MLX runtime environment"
    )
    assert (
        eh._runtime_environment(developer_python, Path(sys.prefix))
        == "developer installation"
    )


def test_selected_runtime_retries_null_cached_value(tmp_path, monkeypatch):
    runtime = tmp_path / "python"
    calls = []

    def select_runtime():
        calls.append("selected")
        return runtime

    monkeypatch.setattr(eh, "_runtime_python_path", select_runtime)
    monkeypatch.setattr(eh, "_SELECTED_RUNTIME", None)
    monkeypatch.setattr(eh, "_RUNTIME_SELECTION_DONE", False)

    assert eh._selected_runtime() == (runtime, False)
    monkeypatch.setattr(eh, "_SELECTED_RUNTIME", None)
    assert eh._selected_runtime() == (runtime, False)
    assert len(calls) == 2


def test_probe_package_helpers_reject_invalid_reports():
    assert eh._probe_package({"packages": []}, "transformers") is None
    assert eh._probe_package_by_module({"packages": []}, "transformers") is None


def test_runtime_import_probe_rejects_non_object_and_timeout(tmp_path):
    runtime = tmp_path / "python"
    runtime.write_text("")
    result = SimpleNamespace(stdout=f"{eh._PROBE_RESULT_PREFIX}{json.dumps([])}\n")
    with mock.patch.object(eh.subprocess, "run", return_value=result):
        assert not eh._runtime_module_importable(runtime, "transformers", None)

    with mock.patch.object(
        eh.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd=str(runtime), timeout=1),
    ):
        assert not eh._runtime_module_importable(runtime, "transformers", None)
        assert not eh._import_probe_was_interrupted(runtime, "transformers", None)
        assert not eh._runtime_module_importable(runtime, "mlx_vlm", None)
        assert eh._import_probe_was_interrupted(runtime.absolute(), "mlx_vlm", None)


def test_python_section_warns_when_selected_runtime_cannot_be_probed(
    tmp_path, monkeypatch
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: None)

    section = eh.section_python()
    row = next(c for c in section.checks if "could not be inspected" in c.label)

    assert row.status is eh.CheckStatus.WARN
    assert str(runtime) in row.detail
    assert "RAPID_MLX_RUNTIME_PYTHON" in row.detail


def test_visibility_helpers_distinguish_unknown_and_context_modules(monkeypatch):
    assert not eh._visible_without_metadata("unknown-distribution")

    with mock.patch.object(eh, "_module_available", return_value=True):
        assert eh._visible_without_metadata("transformers", None)

    runtime = Path("/do-not-exist/python")
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: None)
    assert eh._module_visibility("transformers", runtime) == (False, False)

    package = {
        "module": "trusted_probe_module",
        "importable": True,
        "trusted_origin": True,
    }
    report = {"packages": {"transformers": package}}
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: report)
    monkeypatch.setattr(eh, "_runtime_module_importable", lambda *args, **kwargs: True)
    assert eh._module_visibility("transformers", runtime) == (True, True)


def test_invalid_dependency_version_is_not_supported():
    assert not eh._version_supported("transformers", "not-a-version")


def test_remote_pillow_probe_fails_closed_without_runtime_probe(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: None)

    assert not eh._pil_importable(runtime)


def _server_process(tmp_path, cmdline, exe, environ, uids=None):
    class FakeProcess:
        def __init__(self):
            self.info = {
                "pid": os.getpid() + 1,
                "cmdline": cmdline,
                "create_time": 123.0,
            }

        def exe(self):
            return str(exe)

        def environ(self):
            return environ

        def cwd(self):
            return str(tmp_path)

        def uids(self):
            return uids or SimpleNamespace(real=os.getuid())

    return FakeProcess()


class _FailingUidProbe:
    def __call__(self):
        raise OSError


def _install_fake_process_runtime(monkeypatch, process):
    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [process]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)


def test_runtime_selection_rejects_relative_module_command_with_no_interpreter(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    runtime.chmod(0o755)
    (runtime.parents[1] / "pyvenv.cfg").write_text("")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            ["python", "-m", "vllm_mlx.cli", "serve"],
            runtime,
            {"PATH": ""},
        ),
    )

    assert eh._runtime_python_path() == runtime.absolute()

    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            ["python", "-m", "vllm_mlx.cli", "serve"],
            runtime,
            {"PATH": str(runtime.parent)},
        ),
    )
    assert eh._runtime_python_path() == runtime.absolute()

    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            ["python", "-m", "vllm_mlx.cli", "serve"],
            runtime,
            {"PATH": str(runtime.parent)},
        ),
    )
    assert eh._runtime_python_path() == runtime.absolute()


def test_runtime_selection_rejects_non_python_uid_probe(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    fake_process = _server_process(
        tmp_path,
        [str(runtime), "-m", "vllm_mlx.cli", "serve"],
        runtime,
        {},
        uids=SimpleNamespace(real=os.getuid() + 1),
    )
    _install_fake_process_runtime(
        monkeypatch,
        fake_process,
    )
    assert eh._runtime_python_path() == doctor_exe.absolute()

    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(runtime), "-m", "vllm_mlx.cli", "serve"],
            runtime,
            {},
            uids=_FailingUidProbe(),
        ),
    )
    assert eh._runtime_python_path() == doctor_exe.absolute()


def test_runtime_selection_rejects_empty_server_command(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(tmp_path, ["serve"], runtime, {}),
    )

    assert eh._runtime_python_path() == doctor_exe.absolute()


def test_runtime_selection_resolves_relative_entrypoint_from_server_cwd(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    entrypoint = tmp_path / "tools" / "rapid-mlx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(f"#!{runtime}\nfrom vllm_mlx.cli import main\nmain()\n")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(runtime), "tools/rapid-mlx", "serve"],
            runtime,
            {"PATH": str(runtime.parent)},
        ),
    )

    assert eh._runtime_python_path() == runtime.absolute()


def test_runtime_selection_rejects_missing_relative_entrypoint(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(runtime), "rapid-mlx", "serve"],
            runtime,
            {"PATH": ""},
        ),
    )

    assert eh._runtime_python_path() == doctor_exe.absolute()


def test_runtime_selection_rejects_invalid_entrypoint_and_unknown_entry(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    entrypoint = tmp_path / "bin" / "rapid-mlx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# not a valid entrypoint\n")
    missing_entrypoint = tmp_path / "missing" / "rapid-mlx"
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(runtime), str(missing_entrypoint), "serve"],
            runtime,
            {},
        ),
    )
    assert eh._runtime_python_path() == doctor_exe.absolute()

    with mock.patch.object(
        Path,
        "read_bytes",
        side_effect=OSError,
    ):
        _install_fake_process_runtime(
            monkeypatch,
            _server_process(
                tmp_path,
                [str(runtime), str(entrypoint), "serve"],
                runtime,
                {},
            ),
        )
        assert eh._runtime_python_path() == doctor_exe.absolute()

    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(runtime), "not-rapid-mlx", "serve"],
            runtime,
            {},
        ),
    )
    assert eh._runtime_python_path() == doctor_exe.absolute()


def test_runtime_selection_supports_sibling_python_and_env_fallback(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python3"
    sibling = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    sibling.write_text("")
    entrypoint = runtime.parent / "rapid-mlx"
    entrypoint.write_text("echo no shebang\nfrom vllm_mlx.cli import main\nmain()\n")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(entrypoint), "serve"],
            runtime,
            {},
        ),
    )
    assert eh._runtime_python_path() == runtime.absolute()

    entrypoint.write_text(
        "#!/usr/bin/env python\nfrom vllm_mlx.cli import main\nmain()\n"
    )
    process_runtime = tmp_path / "process" / "bin" / "python"
    process_runtime.parent.mkdir(parents=True)
    process_runtime.write_text("")
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(entrypoint), "serve"],
            process_runtime,
            {},
        ),
    )
    assert eh._runtime_python_path() == process_runtime.absolute()

    shell_process = tmp_path / "process" / "bin" / "sh"
    shell_process.write_text("")
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(entrypoint), "serve"],
            shell_process,
            {},
        ),
    )
    assert eh._runtime_python_path() == runtime.absolute()


def test_runtime_selection_handles_entrypoint_read_errors(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    entrypoint = runtime.parent / "rapid-mlx"
    entrypoint.write_text(f"#!{runtime}\nfrom vllm_mlx.cli import main\nmain()\n")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))

    with (
        mock.patch.object(
            Path,
            "read_bytes",
            return_value=b"from vllm_mlx.cli import main\nmain()\n",
        ),
        mock.patch.object(
            Path,
            "read_text",
            side_effect=OSError,
        ),
    ):
        _install_fake_process_runtime(
            monkeypatch,
            _server_process(
                tmp_path,
                [str(entrypoint), "serve"],
                runtime,
                {},
            ),
        )
        assert eh._runtime_python_path() == runtime.absolute()


def test_runtime_selection_reads_installed_module_marker_files(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    site_root = tmp_path / "site"
    package_root = site_root / "vllm_mlx"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    (package_root / "cli.py").write_text("from vllm_mlx.cli import main\n")
    (site_root / "rapid_mlx-0.0.0.dist-info").mkdir()
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(runtime), str(package_root / "cli.py"), "serve"],
            runtime,
            {},
        ),
    )

    assert eh._runtime_python_path() == runtime.absolute()

    package_root.joinpath("cli.py").unlink()
    package_root.joinpath("cli.py").mkdir()
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(
            tmp_path,
            [str(runtime), str(package_root / "cli.py"), "serve"],
            runtime,
            {},
        ),
    )
    assert eh._runtime_python_path() == doctor_exe.absolute()


def test_runtime_selection_swallows_process_iteration_failures(tmp_path, monkeypatch):
    class FailingProcess:
        @property
        def info(self):
            raise psutil_access_denied

    psutil_access_denied = type("AccessDenied", (RuntimeError,), {})
    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [FailingProcess()]
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = psutil_access_denied
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert eh._runtime_python_path() == Path(sys.executable).absolute()


def test_runtime_selection_ignores_non_server_and_self_processes(
    tmp_path, monkeypatch, allow_rapid_mlx_module_servers
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    _install_fake_process_runtime(
        monkeypatch,
        _server_process(tmp_path, [str(runtime), "not-serve"], runtime, {}),
    )
    assert eh._runtime_python_path() == doctor_exe.absolute()

    fake_psutil = mock.Mock()
    fake_psutil.process_iter.return_value = [
        _server_process(tmp_path, [str(runtime), "serve"], runtime, {})
    ]
    fake_psutil.process_iter.return_value[0].info["pid"] = os.getpid()
    fake_psutil.NoSuchProcess = RuntimeError
    fake_psutil.AccessDenied = RuntimeError
    fake_psutil.ZombieProcess = RuntimeError
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert eh._runtime_python_path() == doctor_exe.absolute()


def test_required_package_timeout_is_warn_for_installed_and_visible_states(
    tmp_path, monkeypatch
):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    probe = {
        "executable": str(runtime),
        "prefix": str(runtime.parents[1]),
        "base_prefix": str(runtime.parents[1]),
        "path": [],
        "packages": {},
    }
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: probe)
    monkeypatch.setattr(eh, "_safe_version", lambda dist, runtime=None: "5.12.1")
    monkeypatch.setattr(
        eh, "_import_probe_was_interrupted", lambda *args, **kwargs: True
    )

    section = eh.section_required_packages()
    row = next(c for c in section.checks if c.label.startswith("transformers"))

    assert row.status is eh.CheckStatus.WARN
    assert "probe timed out" in row.label

    monkeypatch.setattr(eh, "_safe_version", lambda dist, runtime=None: None)
    monkeypatch.setattr(
        eh, "_module_visibility", lambda dist, runtime=None: (True, False)
    )
    visibility_counts: dict[str, int] = {}

    def interrupt_after_visibility(runtime, module, sidecar_root):
        count = visibility_counts.get(module, 0)
        visibility_counts[module] = count + 1
        return count > 0

    monkeypatch.setattr(eh, "_import_probe_was_interrupted", interrupt_after_visibility)
    section = eh.section_required_packages()
    row = next(c for c in section.checks if c.label.startswith("transformers"))
    assert row.status is eh.CheckStatus.WARN
    assert "probe timed out" in row.label


def test_required_package_timeout_before_visibility(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    probe = {
        "executable": str(runtime),
        "prefix": str(runtime.parents[1]),
        "base_prefix": str(runtime.parents[1]),
        "path": [],
        "packages": {},
    }
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: probe)
    monkeypatch.setattr(eh, "_safe_version", lambda dist, runtime=None: None)
    monkeypatch.setattr(
        eh, "_import_probe_was_interrupted", lambda *args, **kwargs: True
    )

    section = eh.section_required_packages()
    row = next(c for c in section.checks if c.label.startswith("transformers"))

    assert row.status is eh.CheckStatus.WARN
    assert "probe timed out" in row.label


def test_required_package_timeout_after_visibility(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    probe = {
        "executable": str(runtime),
        "prefix": str(runtime.parents[1]),
        "base_prefix": str(runtime.parents[1]),
        "path": [],
        "packages": {},
    }
    counts: dict[str, int] = {}

    def interrupt_after_visibility(runtime, module, sidecar_root):
        counts[module] = counts.get(module, 0) + 1
        return counts[module] > 1

    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: probe)
    monkeypatch.setattr(eh, "_safe_version", lambda dist, runtime=None: None)
    monkeypatch.setattr(
        eh, "_module_visibility", lambda dist, runtime=None: (True, False)
    )
    monkeypatch.setattr(eh, "_import_probe_was_interrupted", interrupt_after_visibility)

    section = eh.section_required_packages()

    assert section.checks
    assert all("probe timed out" in check.label for check in section.checks)


def test_optional_package_timeout_is_warn(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    probe = {
        "executable": str(runtime),
        "prefix": str(runtime.parents[1]),
        "base_prefix": str(runtime.parents[1]),
        "path": [],
        "packages": {},
    }
    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: probe)
    monkeypatch.setattr(eh, "_safe_version", lambda dist, runtime=None: None)
    monkeypatch.setattr(
        eh, "_import_probe_was_interrupted", lambda *args, **kwargs: True
    )

    section = eh.section_optional_packages()

    assert section.checks
    timeout_rows = [
        check for check in section.checks if "probe timed out" in check.label
    ]
    assert timeout_rows
    assert all(check.status is eh.CheckStatus.WARN for check in timeout_rows)


def test_optional_package_probe_can_interrupt_after_visibility(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    probe = {
        "executable": str(runtime),
        "prefix": str(runtime.parents[1]),
        "base_prefix": str(runtime.parents[1]),
        "path": [],
        "packages": {},
    }
    interruption_counts: dict[str, int] = {}

    def interrupt_after_visibility(runtime, module, sidecar_root):
        count = interruption_counts.get(module, 0)
        interruption_counts[module] = count + 1
        return count > 0

    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: probe)
    monkeypatch.setattr(eh, "_safe_version", lambda dist, runtime=None: None)
    monkeypatch.setattr(
        eh, "_module_visibility", lambda dist, runtime=None: (True, False)
    )
    monkeypatch.setattr(eh, "_import_probe_was_interrupted", interrupt_after_visibility)

    section = eh.section_optional_packages()

    assert section.checks
    timeout_rows = [
        check for check in section.checks if "probe timed out" in check.label
    ]
    assert timeout_rows
    assert all(check.status is eh.CheckStatus.WARN for check in timeout_rows)


def test_dflash_reports_supported_vlm_with_unverified_import(tmp_path, monkeypatch):
    doctor_exe = tmp_path / "doctor" / "bin" / "python"
    doctor_exe.parent.mkdir(parents=True)
    doctor_exe.write_text("")
    runtime = tmp_path / "server" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("")
    probe = {
        "executable": str(runtime),
        "prefix": str(runtime.parents[1]),
        "base_prefix": str(runtime.parents[1]),
        "path": [],
        "packages": {},
    }

    def safe_version(dist, runtime=None):
        return "0.6.17" if dist == "mlx-vlm" else None

    monkeypatch.setattr(eh.sys, "executable", str(doctor_exe))
    monkeypatch.setattr(eh, "_runtime_python_path", lambda: runtime)
    monkeypatch.setattr(eh, "_probe_runtime", lambda *args, **kwargs: probe)
    monkeypatch.setattr(eh, "_safe_version", safe_version)
    monkeypatch.setattr(eh, "_pil_importable", lambda runtime=None: True)
    monkeypatch.setattr(
        eh, "_module_visibility", lambda dist, runtime=None: (True, False)
    )

    section = eh.section_optional_packages()
    row = next(c for c in section.checks if c.label.startswith("mlx-vlm 0.5.0+"))

    assert row.status is eh.CheckStatus.WARN
    assert "broken or unverified" in row.label
    assert str(runtime) in row.detail
