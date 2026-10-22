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
    monkeypatch.setattr(refresh, "_lock", refresh.threading.Lock())
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


@pytest.mark.parametrize("announced", [None, "1.1.0"])
def test_announce_each_new_version_once(capsys, announced):
    def check_announced():
        if announced is not None:
            assert announced in refresh._seen_versions

    refresh.threading.Thread.return_value.start.side_effect = check_announced
    refresh.start(announced=announced)
    refresh._refresh()
    assert refresh.snapshot() == {"latest_version": "1.1.0", "update_available": True}
    assert capsys.readouterr().err == (
        "" if announced else vc.format_staleness_warning("1.0.0", "1.1.0") + "\n"
    )
    for version in ("1.1.0", "1.2.0", "1.1.0", "1.2.0", "1.3.0", "1.3.0"):
        vc.get_latest_version.return_value = version
        refresh._refresh()
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
    refresh._refresh()
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
    refresh._refresh()
    monkeypatch.setenv(env, "1")
    assert refresh.snapshot() == EMPTY
    refresh._refresh()
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
    refresh._refresh()
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


@pytest.mark.parametrize("already_seen", [False, True])
def test_thread_start_failure_can_retry(already_seen):
    if already_seen:
        refresh._seen_versions.add("1.1.0")
    refresh.threading.Thread.return_value.start.side_effect = RuntimeError("no thread")
    assert refresh.start(announced="1.1.0") is False
    assert refresh.snapshot() == EMPTY
    assert refresh._thread is None
    assert refresh._seen_versions == ({"1.1.0"} if already_seen else set())
    refresh.threading.Thread.return_value.start.side_effect = None
    assert refresh.start() is True


def test_child_ignores_parent_state_and_can_start(capsys):
    assert refresh.start(announced="1.1.0") is True
    refresh._refresh()
    parent_lock = refresh._lock
    parent_seen_versions = refresh._seen_versions
    with parent_lock:
        refresh._reset_after_fork()
        assert parent_lock.locked()
        assert refresh._lock is not parent_lock
        assert not refresh._lock.locked()
        assert refresh._thread is None
        assert refresh._latest is None
        assert refresh._available is False
        assert refresh._seen_versions == set()
        assert refresh._seen_versions is not parent_seen_versions
        assert parent_seen_versions == {"1.1.0"}
    assert refresh.snapshot() == EMPTY
    assert refresh.start() is True
    assert refresh.snapshot() == EMPTY
    refresh._refresh()
    assert (
        capsys.readouterr().err == vc.format_staleness_warning("1.0.0", "1.1.0") + "\n"
    )
    assert refresh.start() is False


def test_audio_serve_passes_announced_version(monkeypatch):
    import sys
    from types import SimpleNamespace

    import rapid_mlx
    from rapid_mlx import cli

    class StopServe(BaseException):
        pass

    server = SimpleNamespace(
        app=object(),
        configure_logging=lambda level: level,
        _sync_config=lambda: None,
        register_audio_routes_if_enabled=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", server)
    monkeypatch.setattr(rapid_mlx, "server", server, raising=False)
    monkeypatch.setattr(
        cli, "_configure_engineless_server_security", lambda *args: None
    )
    monkeypatch.setattr(
        vc, "print_staleness_warning_if_any", Mock(return_value="1.1.0")
    )
    start = Mock(side_effect=StopServe)
    monkeypatch.setattr(refresh, "start", start)
    args = cli.build_parser().parse_args(["serve", "audio-model"])
    entry = SimpleNamespace(
        alias="audio-model", hf_id="org/audio-model", type="stt", languages=None
    )
    with pytest.raises(StopServe):
        cli._serve_audio_mode(args, entry)
    vc.print_staleness_warning_if_any.assert_called_once_with(allow_non_tty=True)
    start.assert_called_once_with(announced="1.1.0")
