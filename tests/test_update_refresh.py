# SPDX-License-Identifier: Apache-2.0
"""Update polling contracts: no network, sleeping, MLX, or live threads."""

import asyncio
from unittest.mock import Mock

import pytest

from rapid_mlx import _version_check as vc
from rapid_mlx import update_refresh as refresh

EMPTY = {"latest_version": None, "update_available": False}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in ("CI", "RAPID_MLX_DISABLE_VERSION_CHECK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(refresh, "_latest", None)
    monkeypatch.setattr(refresh, "_seen_versions", set())
    monkeypatch.setattr(refresh, "_available", False)
    monkeypatch.setattr(refresh, "_thread", None)
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(vc, "get_latest_version", Mock(return_value="1.1.0"))
    monkeypatch.setattr(refresh.threading, "Thread", Mock())


def test_start_is_idempotent_and_does_no_io():
    assert refresh.snapshot() == EMPTY
    assert refresh.start() is True
    thread = refresh.threading.Thread
    thread.assert_called_once_with(
        target=refresh._run,
        args=(vc.CACHE_TTL_SECONDS,),
        daemon=True,
        name="rapid-mlx-update-check",
    )
    thread.return_value.start.assert_called_once_with()
    assert refresh.start() is False
    vc.get_latest_version.assert_not_called()


def test_seed_then_announce_each_new_version_once(capsys):
    refresh.start()
    refresh._refresh(announce=False)
    assert refresh.snapshot() == {"latest_version": "1.1.0", "update_available": True}
    assert capsys.readouterr().err == ""
    for version in ("1.1.0", "1.2.0", "1.1.0", "1.2.0", "1.3.0", "1.3.0"):
        vc.get_latest_version.return_value = version
        refresh._refresh(announce=True)
    assert capsys.readouterr().err == "".join(
        vc.format_staleness_warning("1.0.0", version) + "\n"
        for version in ("1.2.0", "1.3.0")
    )


@pytest.mark.parametrize(
    "installed,latest",
    [
        ("1.1.0", "1.1.0"),
        ("2.0.0", "1.1.0"),
        ("dev", "1.1.0"),
        (None, "1.1.0"),
        ("1.0.0", "dev"),
    ],
)
def test_not_newer_or_unparseable(monkeypatch, installed, latest, capsys):
    refresh.start()
    monkeypatch.setattr(vc, "_installed_version", lambda: installed)
    vc.get_latest_version.return_value = latest
    refresh._refresh(announce=True)
    assert refresh.snapshot() == {"latest_version": latest, "update_available": False}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("initial_failure", [False, True])
def test_loop_survives_failures_and_retains_status(monkeypatch, initial_failure):
    class StopLoop(BaseException):
        pass

    refresh.start(interval=17)
    values = ([RuntimeError()] if initial_failure else []) + [
        "1.1.0",
        None,
        RuntimeError(),
        "1.2.0",
    ]
    vc.get_latest_version.side_effect = values
    observed = []

    def tick(interval):
        assert interval == 17
        observed.append(refresh.snapshot())
        if len(observed) == len(values):
            raise StopLoop

    monkeypatch.setattr(refresh.time, "sleep", tick)
    with pytest.raises(StopLoop):
        refresh._run(17)
    if initial_failure:
        assert observed.pop(0) == EMPTY
    assert [s["latest_version"] for s in observed] == [
        "1.1.0",
        "1.1.0",
        "1.1.0",
        "1.2.0",
    ]


@pytest.mark.parametrize("env", ["RAPID_MLX_DISABLE_VERSION_CHECK", "CI"])
def test_opt_out_before_and_after_start(monkeypatch, env):
    monkeypatch.setenv(env, "1")
    assert refresh.start() is False
    refresh.threading.Thread.assert_not_called()
    monkeypatch.delenv(env)
    assert refresh.start() is True
    refresh._refresh(announce=False)
    monkeypatch.setenv(env, "1")
    assert refresh.snapshot() == EMPTY
    refresh._refresh(announce=True)
    monkeypatch.delenv(env)
    assert refresh.snapshot() == EMPTY
    assert refresh.start() is False


def test_health_reads_snapshot_without_fetching(monkeypatch):
    from rapid_mlx.config import ServerConfig
    from rapid_mlx.routes import health

    monkeypatch.setattr(health, "get_config", ServerConfig)
    result = asyncio.run(health.health())
    assert {key: result[key] for key in EMPTY} == EMPTY
    refresh.start()
    refresh._refresh(announce=False)
    vc.get_latest_version.reset_mock()
    result = asyncio.run(health.health())
    assert result["latest_version"] == "1.1.0"
    assert result["update_available"] is True
    vc.get_latest_version.assert_not_called()


def test_notice_text_is_unchanged():
    assert vc.format_staleness_warning("1.0.0", "1.1.0") == (
        "⚠ rapid-mlx 1.0.0 is behind latest 1.1.0 — "
        "run `rapid-mlx upgrade` to pick up new model aliases / flags."
    )
