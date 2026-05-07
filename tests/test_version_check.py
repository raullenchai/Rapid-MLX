# SPDX-License-Identifier: Apache-2.0
"""Tests for the staleness-warning helper.

The helper is opt-in (TTY+no-CI), cache-aware, and fail-silent on
network errors. Tests pin those guarantees so a future "let's add a
real call" change can't accidentally break the CLI on an offline
laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx import _version_check as vc

# --- _parse_version ---------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.6.14", (0, 6, 14)),
        ("v0.6.14", (0, 6, 14)),  # leading v stripped
        ("1.0.0", (1, 0, 0)),
        ("0.6.14.dev3", (0, 6, 14)),  # dev suffix tolerated, takes patch
    ],
)
def test_parse_version_accepts_typical(raw, expected):
    assert vc._parse_version(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0.6",  # missing patch
        "abc",
        "0.6.x",
    ],
)
def test_parse_version_rejects_garbage(raw):
    assert vc._parse_version(raw) is None


# --- staleness_warning logic (no network) -----------------------------


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at tmp + force interactive mode + no fetch."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(vc, "_cache_path", lambda: cache_dir / "version_check.json")
    # Disable the disabled() short-circuit so logic runs.
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    # Block real network — every test MUST stub _fetch_latest_from_github.
    monkeypatch.setattr(
        vc,
        "_fetch_latest_from_github",
        lambda: pytest.fail("real network call leaked into test"),
    )
    return cache_dir


def _seed_cache(cache_dir: Path, latest: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "version_check.json").write_text(
        json.dumps({"latest": latest, "ts": 9999})
    )


def test_warns_when_2_or_more_patch_behind(isolated_cache, monkeypatch):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.14")
    _seed_cache(isolated_cache, "0.6.16")

    msg = vc.staleness_warning()
    assert msg is not None
    assert "0.6.14" in msg
    assert "0.6.16" in msg
    assert "brew upgrade" in msg


def test_silent_when_only_1_patch_behind(isolated_cache, monkeypatch):
    """1 patch behind is normal noise — minor bug-fix releases happen.
    We only want to nag when feature releases are missed (≥2 lag).
    """
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.15")
    _seed_cache(isolated_cache, "0.6.16")

    assert vc.staleness_warning() is None


def test_silent_when_current(isolated_cache, monkeypatch):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.16")
    _seed_cache(isolated_cache, "0.6.16")

    assert vc.staleness_warning() is None


def test_silent_when_dev_ahead(isolated_cache, monkeypatch):
    """Devs running their own builds ahead of main shouldn't get a
    warning that confuses them about phantom 'latest' releases."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.7.0")
    _seed_cache(isolated_cache, "0.6.16")

    assert vc.staleness_warning() is None


def test_silent_across_minor_boundary(isolated_cache, monkeypatch):
    """If user is on 0.6.x and 0.7.x is out, that's a minor bump — they
    might be intentionally pinning the 0.6 line. Don't auto-suggest a
    cross-minor upgrade."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.10")
    _seed_cache(isolated_cache, "0.7.0")

    assert vc.staleness_warning() is None


def test_silent_when_offline(tmp_path, monkeypatch):
    """No cache + GitHub fetch fails → no warning, no exception."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(vc, "_cache_path", lambda: cache_dir / "version_check.json")
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.14")
    monkeypatch.setattr(vc, "_fetch_latest_from_github", lambda: None)

    assert vc.staleness_warning() is None


def test_silent_when_disabled(monkeypatch):
    monkeypatch.setattr(vc, "_disabled", lambda: True)
    # Even with stub installed/cache that would warn, disabled wins.
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.14")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.16")

    assert vc.staleness_warning() is None


def test_silent_when_dev_build_unparseable(isolated_cache, monkeypatch):
    """``rapid-mlx`` not installed (running from source tree without
    install) → ``pkg_version`` raises and we return None — no warning."""
    monkeypatch.setattr(vc, "_installed_version", lambda: None)

    assert vc.staleness_warning() is None


# --- _disabled honors RAPID_MLX_DISABLE_VERSION_CHECK ----------------


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    assert vc._disabled() is True


def test_disabled_in_ci(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.setenv("CI", "true")
    assert vc._disabled() is True


# --- print_staleness_warning_if_any never raises ---------------------


def test_print_helper_swallows_all_exceptions(monkeypatch, capsys):
    def boom():
        raise RuntimeError("simulated GitHub outage")

    monkeypatch.setattr(vc, "staleness_warning", boom)
    # Must not raise — the CLI must never break because of a staleness
    # check. capsys just makes sure we don't pollute stdout either.
    vc.print_staleness_warning_if_any()
    captured = capsys.readouterr()
    assert captured.out == ""
