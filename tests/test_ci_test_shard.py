import configparser
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the 3.10 lane
    import tomli as tomllib

from scripts.ci_test_shard import TestFile, discover, ignored_paths, partition

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_discovery_pattern_matches_pytest_config() -> None:
    parser = configparser.ConfigParser()
    parser.read(REPO_ROOT / "pytest.ini")
    assert parser["pytest"]["python_files"].split() == ["test_*.py"]


def test_pytest_ini_is_the_only_pytest_config() -> None:
    # pytest.ini wins over pyproject.toml, so a [tool.pytest.ini_options]
    # table there is silently ignored and drifts from the real config.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "pytest" not in pyproject.get("tool", {})


def _run_warning_policy_probe(
    tmp_path: Path, name: str, source: str
) -> subprocess.CompletedProcess[str]:
    probe = tmp_path / f"test_{name}.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(source, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(REPO_ROOT / "pytest.ini"),
            "-o",
            "addopts=",
            str(probe),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_pytest_warning_policy_contract_in_isolated_processes(tmp_path: Path) -> None:
    rapid_owned = _run_warning_policy_probe(
        tmp_path / "rapid-owned",
        "rapid_owned",
        """
import warnings

def test_warning():
    warnings.warn_explicit(
        "rapid-owned warning",
        DeprecationWarning,
        __file__,
        1,
        module="rapid_mlx.warning_policy_probe",
    )
""",
    )
    assert rapid_owned.returncode != 0
    assert "DeprecationWarning: rapid-owned warning" in (
        rapid_owned.stdout + rapid_owned.stderr
    )

    third_party = _run_warning_policy_probe(
        tmp_path / "third-party",
        "third_party",
        """
import warnings

def test_warning():
    warnings.warn_explicit(
        "third-party warning",
        DeprecationWarning,
        __file__,
        1,
        module="third_party.warning_policy_probe",
    )
""",
    )
    third_party_output = third_party.stdout + third_party.stderr
    assert third_party.returncode == 0, third_party_output
    assert "warnings summary" in third_party_output
    assert "DeprecationWarning: third-party warning" in third_party_output

    collection = _run_warning_policy_probe(
        tmp_path / "collection",
        "collection",
        """
class TestUncollectable:
    def __init__(self):
        pass

def test_control():
    pass
""",
    )
    assert collection.returncode != 0
    assert "PytestCollectionWarning" in collection.stdout + collection.stderr

    pytest_deprecation = _run_warning_policy_probe(
        tmp_path / "pytest-deprecation",
        "pytest_deprecation",
        """
import warnings
import pytest

def test_warning():
    warnings.warn(pytest.PytestDeprecationWarning("pytest deprecation"))
""",
    )
    assert pytest_deprecation.returncode != 0
    assert "PytestDeprecationWarning" in (
        pytest_deprecation.stdout + pytest_deprecation.stderr
    )


def _write(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("pass\n" * lines)


def test_partition_is_deterministic_balanced_and_file_atomic() -> None:
    files = [
        TestFile("tests/test_slow.py", 11),
        TestFile("tests/test_medium.py", 7),
        TestFile("tests/test_small_a.py", 4),
        TestFile("tests/test_small_b.py", 3),
        TestFile("tests/test_tiny.py", 1),
    ]

    first = partition(files, 3)
    second = partition(reversed(files), 3)

    assert first == second
    flattened = [item.path for shard in first for item in shard]
    assert sorted(flattened) == sorted(item.path for item in files)
    assert len(flattened) == len(set(flattened))
    weights = [sum(item.weight for item in shard) for shard in first]
    assert max(weights) - min(weights) <= max(item.weight for item in files)


def test_discovery_separates_ordinary_headless_and_integration(tmp_path: Path) -> None:
    _write(tmp_path / "tests/test_root.py", 2)
    _write(tmp_path / "tests/unit/test_nested.py", 3)
    _write(tmp_path / "tests/headless_mlx/test_fake.py", 4)
    _write(tmp_path / "tests/integrations/test_live.py", 5)
    _write(tmp_path / "tests/helper.py", 6)

    ordinary = discover(tmp_path, "ordinary")
    headless = discover(tmp_path, "headless")

    assert [(item.path, item.weight) for item in ordinary] == [
        ("tests/test_root.py", 2),
        ("tests/unit/test_nested.py", 3),
    ]
    assert [(item.path, item.weight) for item in headless] == [
        ("tests/headless_mlx/test_fake.py", 4)
    ]


def test_ignored_paths_make_shards_disjoint_and_complete(tmp_path: Path) -> None:
    for index, lines in enumerate((12, 8, 5, 3, 2, 1), start=1):
        _write(tmp_path / f"tests/test_{index}.py", lines)
    all_paths = {item.path for item in discover(tmp_path, "ordinary")}

    selected_sets = []
    for shard_index in (1, 2, 3):
        ignored = {
            value.removeprefix("--ignore=")
            for value in ignored_paths(tmp_path, "ordinary", shard_index, 3)
        }
        selected_sets.append(all_paths - ignored)

    assert set.union(*selected_sets) == all_paths
    assert all(selected_sets)
    assert all(
        selected_sets[left].isdisjoint(selected_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


@pytest.mark.parametrize("index", (0, 4))
def test_invalid_shard_index_fails(tmp_path: Path, index: int) -> None:
    _write(tmp_path / "tests/test_one.py", 1)
    with pytest.raises(ValueError, match="shard index"):
        ignored_paths(tmp_path, "ordinary", index, 3)
