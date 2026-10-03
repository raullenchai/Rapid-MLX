# SPDX-License-Identifier: Apache-2.0
"""Leaderboard deep links: rapidmlx.com/leaderboard?mac=... and ?run=...

The installer (shell) and the CLI (Python) must print the same ?mac= slug
for the same Mac, or the "what runs on my Mac" link lands on different rows
depending on where the user saw it.
"""

from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from rapid_mlx import cli
from rapid_mlx.community_bench import hardware
from rapid_mlx.leaderboard_links import (
    LEADERBOARD_URL,
    mac_slug,
    mac_url,
    run_url,
    this_mac_url,
)

INSTALL_SH = Path(__file__).resolve().parents[1] / "install.sh"

CASES = [
    ("Apple M4 Pro", 48, "m4-pro-48"),
    ("Apple M4", 16, "m4-16"),
    ("Apple M3 Ultra", 256, "m3-ultra-256"),
    ("Apple M1 Max", 64, "m1-max-64"),
    ("Apple M5 Max", 128, "m5-max-128"),
    ("Apple M2", 8, "m2-8"),
    ("apple m3 pro", 18, "m3-pro-18"),
    ("Apple M2 Max (Virtual)", 32, None),
    ("M4 Pro", 48, None),
    ("Apple M4 Pro extra", 48, None),
    (" Apple M4 Pro", 48, None),
    ("Apple M4 Pro ", 48, None),
    ("Apple  M4 Pro", 48, None),
    ("Apple M4 Mega", 48, None),
    ("Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz", 32, None),
    ("Apple Silicon", 32, None),
    ("", 16, None),
    ("Apple M4 Pro", 0, None),
]


@pytest.mark.parametrize(("chip", "memory", "slug"), CASES)
def test_mac_slug(chip: str, memory: int, slug: str | None) -> None:
    assert mac_slug(chip, memory) == slug
    expected = f"{LEADERBOARD_URL}?mac={slug}" if slug else LEADERBOARD_URL
    assert mac_url(chip, memory) == expected


def test_mac_slug_rejects_anything_but_a_positive_whole_gib() -> None:
    for memory in (None, True, 47.9, 48.0, float("nan"), float("inf"), -8, "48"):
        assert mac_slug("Apple M4 Pro", memory) is None, memory
    assert mac_slug(None, 48) is None


@pytest.mark.parametrize(("chip", "memory", "slug"), CASES)
def test_installer_prints_the_same_link_as_the_cli(
    chip: str, memory: int, slug: str | None
) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; RAPID_INSTALL_LIB=1 source "$1"; '
            'leaderboard_mac_url "$2" "$3"',
            "links",
            str(INSTALL_SH),
            chip,
            str(memory),
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    assert result.stdout.strip() == mac_url(chip, memory)


def test_installer_link_without_memory_falls_back_to_the_board() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; RAPID_INSTALL_LIB=1 source "$1"; '
            'leaderboard_mac_url "Apple M4 Pro" ""',
            "links",
            str(INSTALL_SH),
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    assert result.stdout.strip() == LEADERBOARD_URL


def test_run_url_accepts_only_server_issued_ids() -> None:
    run_id = "301acc89-9185-4ee2-9f11-e3846e3139ab"
    assert run_url(run_id) == f"{LEADERBOARD_URL}?run={run_id}"
    assert run_url(f" {run_id.upper()} ") == f"{LEADERBOARD_URL}?run={run_id}"
    for bad in (None, 7, "", "../admin", "301acc89-9185-1ee2-9f11-e3846e3139ab"):
        assert run_url(bad) is None


def test_this_mac_url_uses_the_host_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_chip", lambda: "Apple M4 Pro")
    monkeypatch.setattr(hardware, "host_memory_gib", lambda: 48)
    assert this_mac_url() == f"{LEADERBOARD_URL}?mac=m4-pro-48"


def test_this_mac_url_reuses_known_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_chip", lambda: "Apple M4 Pro")

    def no_probe() -> int:
        raise AssertionError("memory was already known; must not probe again")

    monkeypatch.setattr(hardware, "host_memory_gib", no_probe)
    assert this_mac_url(48) == f"{LEADERBOARD_URL}?mac=m4-pro-48"


def test_this_mac_url_degrades_to_the_board(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> str:
        raise RuntimeError("no sysctl here")

    monkeypatch.setattr(hardware, "_chip", unavailable)
    monkeypatch.setattr(hardware, "host_memory_gib", lambda: None)
    assert this_mac_url() == LEADERBOARD_URL


def test_recipe_text_ends_with_this_macs_link(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rapid_mlx import recommendations

    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 100.0)
    monkeypatch.setattr(recommendations, "physical_ram_gb", lambda: 48.0)
    monkeypatch.setattr(hardware, "_chip", lambda: "Apple M4 Pro")
    cli.recipe_command(Namespace(max_ram=None, json=False))
    lines = capsys.readouterr().out.rstrip().splitlines()
    assert lines[0].startswith("Recommended for this 48.0 GB Mac")
    assert lines[-1] == (
        f"Measured speeds on Macs like this one: {LEADERBOARD_URL}?mac=m4-pro-48"
    )


def test_recipe_with_max_ram_links_the_board_not_this_mac(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 100.0)

    def no_probe() -> str:
        raise AssertionError("a simulated Mac must not probe this host")

    monkeypatch.setattr(hardware, "_chip", no_probe)
    cli.recipe_command(Namespace(max_ram=32, json=False))
    lines = capsys.readouterr().out.rstrip().splitlines()
    assert lines[-1] == f"Measured speeds on Macs like this one: {LEADERBOARD_URL}"


def test_recipe_json_shape_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 100.0)
    cli.recipe_command(Namespace(max_ram=48, json=True))
    out = capsys.readouterr().out
    assert "leaderboard" not in out
