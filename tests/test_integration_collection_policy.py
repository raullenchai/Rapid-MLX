"""Exercise the real collection hook without contacting an integration server."""

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("arguments", "addopts", "skipped", "url"),
    [
        ([], "", True, "http://localhost:8000"),
        (["--server-url", "http://localhost:8000"], "", False, "http://localhost:8000"),
        (["--server-url=http://127.0.0.1:1"], "", False, "http://127.0.0.1:1"),
        ([], "--server-url=http://127.0.0.1:1", False, "http://127.0.0.1:1"),
        (["--server-url="], "", True, ""),
    ],
)
def test_integration_requires_explicit_server_url(
    tmp_path: Path, arguments: list[str], addopts: str, skipped: bool, url: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "tests/conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_probe.py").write_text(
        "import pytest\n"
        f"def test_default_url(server_url):\n    assert server_url == {url!r}\n"
        "@pytest.mark.integration\n"
        "def test_integration_body():\n    assert True\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(root / "pytest.ini"),
            "-o",
            "addopts=",
            f"--junitxml={tmp_path / 'results.xml'}",
            str(tmp_path / "test_probe.py"),
            *arguments,
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTEST_ADDOPTS": addopts,
            "PYTHONPATH": str(root),
            "RAPID_MLX_TELEMETRY": "0",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases = ET.parse(tmp_path / "results.xml").findall(".//testcase")
    assert len(cases) == 2
    integration = next(
        case for case in cases if case.get("name") == "test_integration_body"
    )
    assert (integration.find("skipped") is not None) == skipped
    assert all(
        case.find("failure") is None and case.find("error") is None for case in cases
    )
