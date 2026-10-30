"""Ordinary source collection and coverage enrollment contracts."""

from pathlib import Path

import yaml

from scripts.ci_test_shard import discover, partition

ROOT = Path(__file__).resolve().parents[1]


def jobs():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]


def test_cli_snapshot_fixture_has_two_cpu_consumers_each_in_one_shard():
    fixture_read = "SNAPSHOT.read_text()"
    consumers = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if path.name != Path(__file__).name and fixture_read in path.read_text()
    ]
    assert sorted(consumers) == [
        "tests/test_cli_parser_snapshot.py",
        "tests/test_serve_command_characterization.py",
    ]

    shards = partition(discover(ROOT, "ordinary"), 3)
    for consumer in consumers:
        selected = [
            index
            for index, shard in enumerate(shards, 1)
            if any(item.path == consumer for item in shard)
        ]
        assert len(selected) == 1


def test_classifier_changes_have_real_hosted_coverage_in_existing_shards():
    steps = jobs()["test-matrix"]["steps"]
    ordinary = next(
        s["run"] for s in steps if "--cov=scripts.release_control" in s.get("run", "")
    )
    assert "--cov=scripts.classify_ci_changes" in ordinary
    assert "tests" in ordinary
    # Keep the combined coverage and mandatory diff gate; enrollment is not a waiver.
    gate = jobs()["changed-lines-coverage"]["steps"][-1]["run"]
    assert "--fail-under 100" in gate
    assert "coverage-data/linux/coverage-linux-3.12.data" in gate
