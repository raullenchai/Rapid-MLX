#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the non-inference helpers in release_artifact_matrix.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "release_artifact_matrix.py"


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location("release_artifact_matrix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_release_wheel_requires_exactly_one_candidate(matrix, tmp_path):
    wheel = tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl"
    wheel.write_bytes(b"placeholder")
    assert matrix.find_release_wheel(tmp_path) == wheel.resolve()


def test_find_release_wheel_rejects_missing_candidate(matrix, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        matrix.find_release_wheel(tmp_path)


def test_find_release_wheel_rejects_ambiguous_candidates(matrix, tmp_path):
    (tmp_path / "rapid_mlx-0.10.8-py3-none-any.whl").write_bytes(b"one")
    (tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl").write_bytes(b"two")
    with pytest.raises(ValueError, match="0.10.8.*0.10.9"):
        matrix.find_release_wheel(tmp_path)


def test_clean_env_drops_source_injection_variables(matrix, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unsafe/source")
    monkeypatch.setenv("PYTHONHOME", "/unsafe/home")
    monkeypatch.setenv("PIP_TARGET", "/unsafe/target")
    env = matrix._clean_env()
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PIP_TARGET" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["RAPID_MLX_DISABLE_VERSION_CHECK"] == "1"
    assert env["RAPID_MLX_TELEMETRY"] == "0"


def test_validate_families_json_allows_a_nonempty_diagnostic_subset(matrix):
    assert matrix.validate_families_json('["qwen36", "gptoss"]') == (
        "qwen36",
        "gptoss",
    )


@pytest.mark.parametrize(
    "value, message",
    [
        ("[]", "non-empty"),
        ('["qwen36", "qwen36"]', "duplicates"),
        ('["qwen36", "unknown"]', "unknown family"),
        ('["qwen36", 3]', "only family-name strings"),
        ('{"family": "qwen36"}', "JSON array"),
        ("not-json", "JSON array"),
    ],
)
def test_validate_families_json_rejects_invalid_selection(matrix, value, message):
    with pytest.raises(ValueError, match=message):
        matrix.validate_families_json(value)


def test_validate_families_json_requires_all_families_for_publication(matrix):
    all_families = list(matrix.FAMILY_CONFIGS)
    assert matrix.validate_families_json(
        str(all_families).replace("'", '"'), require_all_families=True
    ) == tuple(all_families)

    with pytest.raises(ValueError, match="publication requires every release family"):
        matrix.validate_families_json('["qwen36"]', require_all_families=True)


def test_family_configs_cover_the_release_eligible_families(matrix):
    assert set(matrix.FAMILY_CONFIGS) == {"qwen36", "gemma4", "deepseek", "gptoss"}
    assert matrix.FAMILY_CONFIGS["gemma4"].extras == ("vision",)


def test_cli_smoke_covers_base_commands_but_not_optional_chat(matrix):
    assert set(matrix.CLI_SMOKE_SCRIPTS) == {
        "rapid-mlx",
        "rapid-mlx-bench",
        "vllm-mlx",
        "vllm-mlx-bench",
    }


def test_parser_rejects_unknown_family(matrix):
    parser = matrix._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dist-dir", "dist", "--family", "unknown"])


def test_assert_port_available_rejects_a_busy_port(matrix):
    """A stale listener on the release port must be caught BEFORE spawn.

    Guards the readiness-probe integrity fix: if the port is already held
    (e.g. a stale same-family server), the runner must refuse rather than let
    that other process answer /v1/models and mask a broken candidate.
    """
    import socket

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    busy_port = listener.getsockname()[1]
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            matrix._assert_port_available(busy_port)
    finally:
        listener.close()


def test_assert_port_available_accepts_a_free_port(matrix):
    """A genuinely-free port must pass so a normal boot is not blocked."""
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()  # release it so the helper can re-bind

    # Should not raise.
    matrix._assert_port_available(free_port)


def _canonical_pkg_name(spec: str) -> str:
    """Extract + PEP 503-normalize the package name from a requirement spec.

    ``"langchain-openai>=0.2.0"`` -> ``"langchain-openai"``. Normalization
    lower-cases and collapses ``-``/``_``/``.`` runs to a single ``-`` so a
    runtime dep declared as ``Langchain_OpenAI`` still compares equal.
    """
    import re

    # Strip version/marker/extras noise: name ends at the first char that is
    # not part of a PEP 508 distribution name.
    name = re.split(r"[<>=!~;\[ ]", spec.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def test_matrix_test_dependencies_are_client_only(matrix):
    """The SDK/test clients must never leak into the wheel's RUNTIME deps.

    They belong only in the CLIENT venv, isolated from the server venv, so a
    missing runtime dep in the wheel cannot be masked by a client's transitive
    closure. A presence check alone is not enough (it stays green even if a
    client is ALSO added to ``[project].dependencies``); assert the client set
    is disjoint from the released package's declared runtime dependencies.
    """
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover — 3.10 fallback
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        except ModuleNotFoundError:
            pytest.skip("tomllib/tomli required to parse pyproject.toml")

    pyproject_path = _REPO_ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as fp:
        pyproject = tomllib.load(fp)

    runtime_deps = {
        _canonical_pkg_name(spec)
        for spec in pyproject.get("project", {}).get("dependencies", [])
    }
    client_pkgs = {
        _canonical_pkg_name(spec) for spec in matrix.MATRIX_TEST_DEPENDENCIES
    }

    # Sanity: the known client SDKs really are in the client tuple.
    assert {"openai", "langchain-openai", "aider-chat"} <= client_pkgs

    leaked = client_pkgs & runtime_deps
    assert not leaked, (
        "matrix client/test package(s) also appear in the released package's "
        "runtime [project].dependencies — this defeats the server/client venv "
        f"split and lets a client mask a missing runtime dep: {sorted(leaked)}"
    )
