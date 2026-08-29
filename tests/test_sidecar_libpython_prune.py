from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRUNER = ROOT / "apps/rapid-mac/scripts/prune-unused-libpython.sh"


def _fixture(tmp_path: Path, *, consumer: bool = False) -> tuple[Path, dict[str, str]]:
    stage = tmp_path / "stage"
    libpython = stage / "python/lib/libpython3.12.dylib"
    interpreter = stage / "python/bin/python3.12"
    extension = stage / "site-packages/example.so"
    for path in (libpython, interpreter, extension):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    tools = tmp_path / "tools"
    tools.mkdir()
    file_tool = tools / "file"
    file_tool.write_text("#!/bin/sh\necho 'Mach-O 64-bit bundle arm64'\n")
    file_tool.chmod(0o755)
    otool = tools / "otool"
    otool.write_text(
        "#!/bin/sh\n"
        'echo "$2:"\n'
        'if [ "${FAKE_LIBPYTHON_CONSUMER:-}" = "$2" ]; then\n'
        "  echo '    @rpath/libpython3.12.dylib (compatibility version 3.12.0)'\n"
        "else\n"
        "  echo '    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0)'\n"
        "fi\n"
    )
    otool.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tools}:{env['PATH']}"
    if consumer:
        env["FAKE_LIBPYTHON_CONSUMER"] = str(extension)
    return stage, env


def _run(stage: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PRUNER), str(stage)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_pruner_removes_libpython_only_after_complete_no_consumer_scan(tmp_path):
    stage, env = _fixture(tmp_path)

    result = _run(stage, env)

    assert result.returncode == 0, result.stderr
    assert not (stage / "python/lib/libpython3.12.dylib").exists()


def test_pruner_refuses_a_linked_libpython_and_names_the_consumer(tmp_path):
    stage, env = _fixture(tmp_path, consumer=True)

    result = _run(stage, env)

    assert result.returncode != 0
    assert "refusing to drop it" in result.stderr
    assert "example.so" in result.stderr
    assert (stage / "python/lib/libpython3.12.dylib").exists()


def test_pruner_refuses_an_incomplete_macho_scan(tmp_path):
    stage, env = _fixture(tmp_path)
    (stage / "site-packages/example.so").unlink()

    result = _run(stage, env)

    assert result.returncode != 0
    assert "incomplete scan" in result.stderr
    assert (stage / "python/lib/libpython3.12.dylib").exists()


def test_pruner_refuses_an_otool_failure(tmp_path):
    stage, env = _fixture(tmp_path)
    broken_otool = Path(env["PATH"].split(":", 1)[0]) / "otool"
    broken_otool.write_text("#!/bin/sh\nexit 1\n")
    broken_otool.chmod(0o755)

    result = _run(stage, env)

    assert result.returncode != 0
    assert "otool failed" in result.stderr
    assert (stage / "python/lib/libpython3.12.dylib").exists()
