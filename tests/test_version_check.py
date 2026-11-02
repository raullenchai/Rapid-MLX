# SPDX-License-Identifier: Apache-2.0
"""Tests for the staleness-warning helper.

The helper is TTY-gated for normal commands, while ``serve`` may emit a
passive notice to non-TTY startup logs. Both paths are cache-aware,
explicitly opt-out, and fail-silent on network errors. Tests pin those
guarantees so a future "let's add a real call" change can't accidentally
break the CLI on an offline laptop.
"""

from __future__ import annotations

import json
import os
import site
import subprocess
import sys
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rapid_mlx import _version_check as vc


@pytest.fixture(autouse=True)
def no_real_network_or_cache(monkeypatch, tmp_path):
    import urllib.error

    monkeypatch.setattr(vc, "_cache_path", lambda: tmp_path / "version_check.json")
    monkeypatch.setattr(vc, "_prompt_floor_announced", None)

    def offline(*args, **kwargs):
        raise urllib.error.URLError("network disabled in tests")

    monkeypatch.setattr(vc.urllib.request, "urlopen", offline)


# --- _parse_version ---------------------------------------------------

INF = float("inf")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.6.14", (0, 6, 14, INF)),  # final
        ("v0.6.14", (0, 6, 14, INF)),  # leading v stripped
        ("1.0.0", (1, 0, 0, INF)),
        ("0.6.14.dev3", (0, 6, 14, INF)),  # dev: base tuple, unchanged
        ("0.6.14.post1", (0, 6, 14, INF)),  # dotted suffix: base tuple
    ],
)
def test_parse_version_accepts_typical(raw, expected):
    assert vc._parse_version(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Attached rcN parses to its rc ordinal, so it sorts BELOW the
        # matching final (which carries ``inf``).
        ("0.13.0rc1", (0, 13, 0, 1)),
        ("0.13.0rc2", (0, 13, 0, 2)),
        ("0.6.62rc1", (0, 6, 62, 1)),
    ],
)
def test_parse_version_accepts_release_candidates(raw, expected):
    assert vc._parse_version(raw) == expected


def test_rc_sorts_below_its_final():
    """An rc of X.Y.Z must compare less than the X.Y.Z final, so once the
    final ships an rc user reports as behind (staleness warning fires)."""
    rc = vc._parse_version("0.13.0rc1")
    final = vc._parse_version("0.13.0")
    assert rc is not None and final is not None
    assert rc < final


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0.6",  # missing patch
        "abc",
        "0.6.x",
        "0.6.62a1",  # attached alpha stays unparseable
        "0.6.62+local.build",  # local-version metadata stays unparseable
    ],
)
def test_parse_version_rejects_garbage(raw):
    assert vc._parse_version(raw) is None


# --- _fetch_latest routes through the countable landing worker --------


class _FakeResp:
    """Minimal urlopen() context-manager stand-in."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_fetch_latest_targets_cli_update_endpoint_with_version(monkeypatch):
    """The update poll must go to rapidmlx.com/api/cli-update (countable),
    NOT api.github.com directly, and carry the installed version as the
    ``v`` query param so the server can bucket active-install counts."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return _FakeResp(json.dumps({"tag_name": "v0.6.70"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = vc._fetch_latest()

    assert result == "0.6.70"  # leading v stripped, parse unchanged
    # Exact URL — parse it so a near-miss path like ``/api/cli-update-legacy``
    # (which ``startswith`` would wave through) fails the test.
    parsed = urllib.parse.urlparse(captured["url"])
    assert parsed.scheme == "https"
    assert parsed.netloc == "rapidmlx.com"  # never api.github.com
    assert parsed.path == "/api/cli-update"  # exact path, no suffix
    assert urllib.parse.parse_qs(parsed.query) == {"v": ["0.6.61"]}
    # Timeout guard preserved.
    assert captured["timeout"] == vc.NETWORK_TIMEOUT_SECONDS


def test_fetch_latest_pins_nonidentifying_user_agent(monkeypatch):
    """The poll must send a fixed, non-identifying User-Agent — NOT
    urllib's default ``Python-urllib/<x.y.z>`` (which would leak the
    interpreter patch version). This keeps the on-the-wire footprint to
    the ``v`` param + unavoidable IP, matching the privacy docstring."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    captured = {}

    def fake_urlopen(req, timeout=None):
        # urllib normalizes header keys to ``User-agent``.
        captured["ua"] = req.get_header("User-agent")
        return _FakeResp(json.dumps({"tag_name": "v0.6.70"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    vc._fetch_latest()

    assert captured["ua"] == vc.USER_AGENT
    assert captured["ua"] == "rapid-mlx-cli"
    # Never the interpreter-leaking default.
    assert "Python-urllib" not in (captured["ua"] or "")


def test_fetch_latest_url_encodes_version(monkeypatch):
    """A version with URL-special chars (local build metadata carries
    ``+``) must be percent-encoded so the query string stays well-formed
    and nothing but the version leaks."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61+local.build")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResp(json.dumps({"tag_name": "v0.6.70"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    vc._fetch_latest()
    # ``+`` percent-encoded (urlencode uses quote_plus → %2B), no raw +.
    assert "v=0.6.61%2Blocal.build" in captured["url"]


def test_fetch_latest_sends_empty_version_when_uninstalled(monkeypatch):
    """Running from an uninstalled source tree → ``_installed_version``
    is None → still send ``v=`` (empty), never crash."""
    monkeypatch.setattr(vc, "_installed_version", lambda: None)
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResp(json.dumps({"tag_name": "v0.6.70"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert vc._fetch_latest() == "0.6.70"
    assert "v=" in captured["url"]
    assert captured["url"].startswith("https://rapidmlx.com/api/cli-update")


def test_fetch_latest_fail_open_on_urlerror(monkeypatch):
    """Any network error → None, silently (never break the CLI)."""
    import urllib.error

    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")

    def boom(req, timeout=None):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert vc._fetch_latest() is None


def test_fetch_latest_fail_open_on_incomplete_read(monkeypatch):
    """Truncated chunked response (http.client.IncompleteRead, an
    HTTPException — not an OSError) → None, never a crash. Regression:
    ``rapid-mlx upgrade`` crashed with a raw traceback on a flaky
    connection instead of failing open."""
    import http.client

    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")

    class _TruncatedResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: _TruncatedResp()
    )
    assert vc._fetch_latest() is None


def test_fetch_latest_fail_open_on_bad_json(monkeypatch):
    """Malformed worker response → None, no exception."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")

    def fake_urlopen(req, timeout=None):
        return _FakeResp(b"not json{")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert vc._fetch_latest() is None


def test_fetch_latest_returns_none_when_tag_missing(monkeypatch):
    """Passthrough JSON without ``tag_name`` → None (unchanged parse)."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")

    responses = iter(
        [
            _FakeResp(json.dumps({"other": "field"}).encode()),
            _FakeResp(json.dumps([]).encode()),
        ]
    )

    def fake_urlopen(req, timeout=None):
        return next(responses)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert vc._fetch_latest() is None


@pytest.mark.parametrize("worker_body", [[], "valid-json", 42])
def test_fetch_latest_fails_open_on_non_object_worker_json(monkeypatch, worker_body):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.12.7")
    responses = iter(
        [
            _FakeResp(json.dumps(worker_body).encode()),
            _FakeResp(json.dumps([]).encode()),
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: next(responses))
    assert vc._fetch_latest() is None


def test_fetch_latest_ignores_desktop_release_and_selects_engine_tag(monkeypatch):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.12.7")
    urls = []
    responses = iter(
        [
            _FakeResp(json.dumps({"tag_name": "rapid-mac-v0.12.5"}).encode()),
            _FakeResp(
                json.dumps(
                    [
                        {"tag_name": "rapid-mac-v0.12.5"},
                        {"tag_name": "v0.12.7"},
                        {"tag_name": "v0.12.6"},
                        {"tag_name": "99.0.0"},
                        {"tag_name": 999},
                        {"tag_name": f"v{'9' * 5000}.0.0"},
                    ]
                ).encode()
            ),
        ]
    )

    def fake_urlopen(req, timeout=None):
        urls.append(req.full_url)
        return next(responses)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert vc._fetch_latest() == "0.12.7"
    assert urls == [
        "https://rapidmlx.com/api/cli-update?v=0.12.7",
        f"{vc.GITHUB_RELEASES_ENDPOINT}&page=1",
    ]


def test_desktop_fallback_paginates_until_engine_release(monkeypatch):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.12.7")
    desktop_page = [{"tag_name": f"rapid-mac-v0.11.{index}"} for index in range(100)]
    responses = iter(
        [
            _FakeResp(json.dumps({"tag_name": "rapid-mac-v0.12.5"}).encode()),
            _FakeResp(json.dumps(desktop_page).encode()),
            _FakeResp(json.dumps([{"tag_name": "v0.12.7"}]).encode()),
        ]
    )
    urls = []

    def fake_urlopen(req, timeout=None):
        urls.append(req.full_url)
        return next(responses)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert vc._fetch_latest() == "0.12.7"
    assert urls[-2:] == [
        f"{vc.GITHUB_RELEASES_ENDPOINT}&page=1",
        f"{vc.GITHUB_RELEASES_ENDPOINT}&page=2",
    ]


def test_disabled_short_circuits_before_any_fetch(monkeypatch):
    """``_disabled()`` must gate the whole check — no network, even the
    countable one, when running in CI / non-TTY / opted out."""
    monkeypatch.setattr(vc, "_disabled", lambda: True)
    monkeypatch.setattr(
        vc,
        "_fetch_latest",
        lambda: pytest.fail("fetch leaked despite _disabled()"),
    )
    assert vc.staleness_warning() is None


# --- staleness_warning logic (no network) -----------------------------


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at tmp + force interactive mode + no fetch."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(vc, "_cache_path", lambda: cache_dir / "version_check.json")
    # Disable the disabled() short-circuit so logic runs.
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    # Block real network — every test MUST stub _fetch_latest.
    monkeypatch.setattr(
        vc,
        "_fetch_latest",
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
    assert "rapid-mlx upgrade" in msg


def test_warns_when_1_patch_behind(isolated_cache, monkeypatch):
    """Any newer release warns — including a single patch. There is no
    ≥2-lag threshold anymore: we nudge on every update so no straggler is
    silently left behind.
    """
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.15")
    _seed_cache(isolated_cache, "0.6.16")

    msg = vc.staleness_warning()
    assert msg is not None
    assert "0.6.15" in msg
    assert "0.6.16" in msg
    assert "rapid-mlx upgrade" in msg


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


def test_rc_user_sees_warning_once_final_ships(isolated_cache, monkeypatch):
    """The 0.13.1 behaviour: a user on ``0.13.0rc1`` must be told they are
    behind now that final ``0.13.0`` is out. Previously rc parsed to None
    and stayed silent forever."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.13.0rc1")
    _seed_cache(isolated_cache, "0.13.0")

    msg = vc.staleness_warning()
    assert msg is not None
    assert "0.13.0rc1" in msg
    assert "0.13.0" in msg
    assert "rapid-mlx upgrade" in msg


def test_dev_build_behavior_unchanged(isolated_cache, monkeypatch):
    """A ``X.Y.Z.devN`` build keeps its exact historical comparison: behind
    a released final it warns (old ``_parse_version`` gave ``(0, 6, 15)``),
    on/equal to final it is silent — the rc change must not alter devN."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.15.dev1")
    _seed_cache(isolated_cache, "0.6.16")

    msg = vc.staleness_warning()
    assert msg is not None
    assert "0.6.15.dev1" in msg
    assert "0.6.16" in msg

    # And on the same base final, a dev build is not treated as ahead.
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.16.dev1")
    assert vc.staleness_warning() is None


def test_warns_across_minor_boundary(isolated_cache, monkeypatch):
    """Cross-minor lag now warns. Users pinned on an old minor are exactly
    the fragmentation we want to nudge; the passive one-liner is opt-out
    (``RAPID_MLX_DISABLE_VERSION_CHECK``) for anyone deliberately staying."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.10")
    _seed_cache(isolated_cache, "0.7.0")

    msg = vc.staleness_warning()
    assert msg is not None
    assert "0.6.10" in msg
    assert "0.7.0" in msg


@pytest.mark.parametrize(
    "installed,latest",
    [
        ("0.10.8", "0.12.10"),  # the real straggler cohort (0.10.x panel bucket)
        ("0.11.0", "0.12.10"),  # the real straggler cohort (0.11.x panel bucket)
        ("0.9.9", "1.0.0"),  # behind by a whole major — still warns
    ],
)
def test_warns_for_cross_line_stragglers(
    isolated_cache, monkeypatch, installed, latest
):
    """Regression for the fragmentation goal: an install several minors
    (or a major) behind the latest release MUST get the banner. The old
    same-major.minor gate returned None for exactly these users, leaving
    the fragmented base un-nudged on every non-``serve`` command."""
    monkeypatch.setattr(vc, "_installed_version", lambda: installed)
    _seed_cache(isolated_cache, latest)

    msg = vc.staleness_warning()
    assert msg is not None
    assert installed in msg
    assert latest in msg
    assert "rapid-mlx upgrade" in msg


def test_silent_when_offline(tmp_path, monkeypatch):
    """No cache + GitHub fetch fails → no warning, no exception."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(vc, "_cache_path", lambda: cache_dir / "version_check.json")
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.14")
    monkeypatch.setattr(vc, "_fetch_latest", lambda: None)

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


def test_non_tty_serve_warning_bypasses_only_tty_gate(monkeypatch):
    """Daemon startup may check freshness, but explicit opt-outs still win."""
    monkeypatch.setattr(vc, "_disabled", lambda: True)
    monkeypatch.setattr(vc, "_explicitly_disabled", lambda: False)
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.14")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.16")

    msg = vc.staleness_warning(allow_non_tty=True)

    assert msg is not None
    assert "0.6.14" in msg
    assert "0.6.16" in msg


@pytest.mark.parametrize(
    "env_name",
    ["CI", "RAPID_MLX_DISABLE_VERSION_CHECK"],
)
def test_non_tty_serve_warning_respects_explicit_opt_out(monkeypatch, env_name):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("RAPID_MLX_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.setenv(env_name, "1")
    monkeypatch.setattr(
        vc,
        "get_latest_version",
        lambda force_refresh=False: pytest.fail("network leaked despite opt-out"),
    )

    assert vc.staleness_warning(allow_non_tty=True) is None


def test_non_tty_serve_warning_prints_only_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr(
        vc,
        "_staleness_warning_versions",
        lambda *, allow_non_tty=False: ("1.0.0", "1.1.0") if allow_non_tty else None,
    )

    assert vc.print_staleness_warning_if_any(allow_non_tty=True) == "1.1.0"

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == vc.format_staleness_warning("1.0.0", "1.1.0")


def test_all_serve_lanes_opt_into_non_tty_warning():
    """Keep daemon logging scoped to all serve lanes, not every CLI command."""
    cli_source = Path(vc.__file__).with_name("cli.py").read_text()
    call = "print_staleness_warning_if_any(allow_non_tty=True)"

    assert cli_source.count(call) == 4
    audio_source = cli_source.split("def _serve_audio_mode", 1)[1].split("\ndef ", 1)[0]
    text_source = cli_source.split("def serve_command", 1)[1].split("\ndef ", 1)[0]
    retrieval_source = cli_source.split("def _serve_retrieval_only_mode", 1)[1].split(
        "\ndef ", 1
    )[0]
    cua_source = cli_source.split("def _serve_cua_only_mode", 1)[1].split("\ndef ", 1)[
        0
    ]
    assert call in audio_source
    assert call in text_source
    assert call in retrieval_source
    assert call in cua_source


# --- print_staleness_warning_if_any never raises ---------------------


def test_print_helper_swallows_all_exceptions(monkeypatch, capsys):
    def boom(**_kwargs):
        raise RuntimeError("simulated GitHub outage")

    monkeypatch.setattr(vc, "_staleness_warning_versions", boom)
    # Must not raise — the CLI must never break because of a staleness
    # check. capsys just makes sure we don't pollute stdout either.
    assert vc.print_staleness_warning_if_any() is None
    captured = capsys.readouterr()
    assert captured.out == ""


# --- staleness warning recommends `rapid-mlx upgrade` ----------------


def test_warning_message_recommends_upgrade_subcommand(isolated_cache, monkeypatch):
    """The banner must point users at our own upgrade subcommand.

    We centralise install-method detection in ``rapid-mlx upgrade`` (it also
    handles pip / install.sh users, who a raw ``brew upgrade`` would strand),
    so the warning just needs to point there.
    """
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.20")
    _seed_cache(isolated_cache, "0.6.30")

    msg = vc.staleness_warning()
    assert msg is not None
    assert "rapid-mlx upgrade" in msg


# --- detect_install_method() -----------------------------------------


def test_detect_install_method_brew(monkeypatch):
    """A brew install resolves through realpath into ``/opt/homebrew/Cellar/``.

    The detector must spot that and return the core formula upgrade command
    — ``brew upgrade rapid-mlx`` (rapid-mlx is in homebrew/core, no tap needed).
    """
    fake_binary = "/opt/homebrew/bin/rapid-mlx"
    fake_realpath = "/opt/homebrew/Cellar/rapid-mlx/0.6.20/bin/rapid-mlx"
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: fake_realpath if p == fake_binary else p,
    )
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)

    info = vc.detect_install_method()
    assert info.method == "brew"
    assert info.upgrade_command == "brew upgrade rapid-mlx"
    assert info.upgrade_argv == ["brew", "upgrade", "rapid-mlx"]
    assert info.binary_path == fake_binary


def test_detect_install_method_brew_linux(monkeypatch):
    """Linux Homebrew installs to ``/home/linuxbrew/.linuxbrew/`` — must
    detect there too, otherwise Linux-via-brew users get the pip command."""
    fake_binary = "/home/linuxbrew/.linuxbrew/bin/rapid-mlx"
    fake_realpath = "/home/linuxbrew/.linuxbrew/Cellar/rapid-mlx/0.6.20/bin/rapid-mlx"
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: fake_realpath if p == fake_binary else p,
    )
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)

    info = vc.detect_install_method()
    assert info.method == "brew"


def test_unrelated_path_launcher_cannot_override_running_interpreter(monkeypatch):
    """Module entrypoints must ignore another installation's PATH launcher."""
    fake_binary = "/opt/homebrew/bin/rapid-mlx"
    fake_realpath = "/opt/homebrew/Cellar/rapid-mlx/0.15.2/bin/rapid-mlx"
    monkeypatch.setattr(vc.sys, "executable", "/tmp/active-venv/bin/python")
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: fake_realpath if str(p) == fake_binary else str(p),
    )
    monkeypatch.setattr("sysconfig.get_path", lambda _name: "/tmp/active-venv/bin")

    info = vc.detect_install_method()

    assert info.method == "pip"
    assert info.binary_path is None
    assert info.upgrade_argv[:3] == ["/tmp/active-venv/bin/python", "-m", "pip"]


def test_launcher_ownership_requires_distribution_and_active_interpreter(
    tmp_path, monkeypatch
):
    launcher = tmp_path / "foreign" / "rapid-mlx"
    launcher.parent.mkdir()
    monkeypatch.setattr(vc, "distribution", lambda _name: MagicMock(entry_points=[]))
    assert vc._launcher_belongs_to_running_install(str(launcher)) is False

    monkeypatch.setattr(
        vc,
        "distribution",
        lambda _name: MagicMock(entry_points=[SimpleNamespace(name="rapid-mlx")]),
    )
    monkeypatch.setattr("sysconfig.get_path", lambda _name: str(tmp_path / "scripts"))

    launcher.write_text(f"#!{vc.sys.executable}\n")
    assert vc._launcher_belongs_to_running_install(str(launcher)) is True

    launcher.write_text("not a script\n")
    assert vc._launcher_belongs_to_running_install(str(launcher)) is False

    launcher.write_text("#!/different/python\n")
    assert vc._launcher_belongs_to_running_install(str(launcher)) is False

    monkeypatch.setattr("sysconfig.get_path", lambda _name: str(launcher.parent))
    assert vc._launcher_belongs_to_running_install(str(launcher)) is False


def test_launcher_ownership_distinguishes_sibling_venvs(tmp_path, monkeypatch):
    for name in ("a", "b"):
        subprocess.run(
            [
                sys._base_executable or sys.executable,
                "-m",
                "venv",
                "--without-pip",
                str(tmp_path / name),
            ],
            check=True,
        )
        launcher = tmp_path / name / "bin" / "rapid-mlx"
        launcher.write_text(f"#!{launcher.parent / 'python'}\n")
        launcher.chmod(0o755)
    bin_b = tmp_path / "b" / "bin"
    python3 = bin_b / "python3"
    if not python3.exists():
        python3.symlink_to("python")
    monkeypatch.setattr(sys, "executable", str(bin_b / "python"))
    monkeypatch.setattr(
        vc,
        "distribution",
        lambda _: SimpleNamespace(entry_points=[SimpleNamespace(name="rapid-mlx")]),
    )
    assert not vc._launcher_belongs_to_running_install(
        str(tmp_path / "a" / "bin" / "rapid-mlx")
    )
    launcher = bin_b / "rapid-mlx"
    assert vc._launcher_belongs_to_running_install(str(launcher))
    launcher.write_text(f"#!{python3}\n")
    assert vc._launcher_belongs_to_running_install(str(launcher))


def test_python_environment_falls_back_when_readlink_fails(tmp_path, monkeypatch):
    target = tmp_path / "python-real"
    target.touch()
    path = tmp_path / "python"
    path.symlink_to(target)

    expected = os.path.normcase(os.path.realpath(path))
    # Fail the environment probe, then allow realpath to resolve the symlink.
    real = os.readlink
    first_call = True

    def readlink(*args, **kwargs):
        nonlocal first_call
        if first_call:
            first_call = False
            raise OSError("cannot read symlink")
        return real(*args, **kwargs)

    monkeypatch.setattr(vc.os, "readlink", readlink)
    assert vc._python_environment(str(path)) == expected


def test_active_scripts_symlink_must_target_active_interpreter(tmp_path, monkeypatch):
    scripts = tmp_path / "active" / "bin"
    scripts.mkdir(parents=True)
    foreign = tmp_path / "pipx" / "bin" / "rapid-mlx"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/foreign/python\n")
    launcher = scripts / "rapid-mlx"
    launcher.symlink_to(foreign)
    monkeypatch.setattr(
        vc,
        "distribution",
        lambda _name: MagicMock(entry_points=[SimpleNamespace(name="rapid-mlx")]),
    )
    monkeypatch.setattr("sysconfig.get_path", lambda _name: str(scripts))

    assert vc._launcher_belongs_to_running_install(str(launcher)) is False

    foreign.write_text(f"#!{vc.sys.executable}\n")
    assert vc._launcher_belongs_to_running_install(str(launcher)) is True


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (
            "#!/bin/sh",
            "'''exec' '/path with space/python' \"$0\" \"$@\"",
            "/path with space/python",
        ),
        (
            "#!/bin/sh",
            '\'\'\'exec\' "/path with space/python" "$0" "$@"',
            "/path with space/python",
        ),
        ("#!/bin/sh", '\'\'\'exec\' "/path/$var/python" "$0" "$@"', None),
        ("#!/bin/sh", '\'\'\'exec\' "/path/`cmd`/python" "$0" "$@"', None),
        ("#!/bin/sh", '\'\'\'exec\' "/path/\\python" "$0" "$@"', None),
        ("#!/bin/sh", "'''exec' /path/python \"$0\" \"$@\"", "/path/python"),
        ("#!/bin/sh -", "'''exec' /path/python \"$0\" \"$@\"", "/path/python"),
        ("#!/x/python", "", "/x/python"),
        ("#!/usr/bin/env python3", "", "/usr/bin/env"),
        ("not a shebang", "", None),
        ("#!/bin/sh", "invalid", None),
    ],
)
def test_launcher_interpreter(first, second, expected):
    assert vc.launcher_interpreter(first, second) == expected


@pytest.mark.parametrize("shell", ["#!/bin/sh", "#!/bin/sh -"])
@pytest.mark.parametrize("quoting", ["single", "double", "none"])
def test_macos_pipx_trampoline_ownership(tmp_path, monkeypatch, shell, quoting):
    home = tmp_path / "home"
    venv = home / "Library" / "Application Support" / "pipx" / "venvs" / "rapid-mlx"
    launcher = venv / "bin" / "rapid-mlx"
    launcher.parent.mkdir(parents=True)
    (venv / "pipx_metadata.json").write_text("managed")
    python = tmp_path / "python" if quoting == "none" else launcher.parent / "python"
    interpreter = {
        "single": f"'{python}'",
        "double": f'"{python}"',
        "none": str(python),
    }[quoting]
    launcher.write_text(f"{shell}\n'''exec' {interpreter} \"$0\" \"$@\"\n' '''\n")
    monkeypatch.setattr(vc.sys, "executable", str(python))
    monkeypatch.setattr(vc.sys, "platform", "darwin")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: str(launcher))
    monkeypatch.setattr(
        vc,
        "distribution",
        lambda _name: MagicMock(entry_points=[SimpleNamespace(name="rapid-mlx")]),
    )
    for name in ("PIPX_HOME", "PIPX_GLOBAL_HOME", "UV_TOOL_DIR", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)

    info = vc.detect_install_method()
    assert info.method == "pipx"
    assert info.upgrade_argv == ["pipx", "upgrade", "rapid-mlx"]

    monkeypatch.setattr(vc.sys, "executable", str(tmp_path / "different-python"))
    assert vc._launcher_belongs_to_running_install(str(launcher)) is False


def test_launcher_ownership_fails_closed_without_distribution(monkeypatch) -> None:
    monkeypatch.setattr(
        vc,
        "distribution",
        lambda _name: (_ for _ in ()).throw(vc.PackageNotFoundError()),
    )

    assert vc._launcher_belongs_to_running_install("/tmp/rapid-mlx") is False


def test_local_bin_launcher_alone_does_not_imply_install_sh(tmp_path, monkeypatch):
    """uv, pipx, and user-site pip all share ``~/.local/bin``.

    A launcher located there but not resolving into a known managed venv must
    conservatively use the running interpreter instead of letting install.sh
    replace an unrelated tool manager's entry point.
    """
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    fake_binary = str(local_bin / "rapid-mlx")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr("os.path.realpath", lambda p: p)

    info = vc.detect_install_method()
    assert info.method == "pip"
    assert info.upgrade_argv[:3] == [vc.sys.executable, "-m", "pip"]


def test_detect_install_method_install_sh_via_symlink(tmp_path, monkeypatch):
    """install.sh actually creates a venv under ``~/.rapid-mlx/`` and
    symlinks the entry point into ``~/.local/bin/rapid-mlx``. ``realpath``
    resolves through the symlink, so a check that *only* looked at the
    resolved path classified install.sh users as 'pip' and silently
    suggested the wrong upgrade command. Pin the symlink case explicitly.
    """
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    venv_bin = home / ".rapid-mlx" / "bin"
    venv_bin.mkdir(parents=True)
    fake_binary = str(local_bin / "rapid-mlx")
    fake_realpath = str(venv_bin / "rapid-mlx")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: fake_realpath if p == fake_binary else p,
    )
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)

    info = vc.detect_install_method()
    assert info.method == "install_sh"
    assert info.upgrade_argv == [
        vc.sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "rapid-mlx",
    ]
    assert "curl" not in info.upgrade_command
    assert "bash" not in info.upgrade_argv


@pytest.mark.parametrize(
    "manager,resolved_parts,expected_argv",
    [
        (
            "uv",
            (".local", "share", "uv", "tools", "rapid-mlx", "bin", "rapid-mlx"),
            ["uv", "tool", "upgrade", "rapid-mlx"],
        ),
        (
            "pipx",
            (".local", "share", "pipx", "venvs", "rapid-mlx", "bin", "rapid-mlx"),
            ["pipx", "upgrade", "rapid-mlx"],
        ),
    ],
)
def test_detects_tool_manager_from_resolved_launcher(
    tmp_path, monkeypatch, manager, resolved_parts, expected_argv
):
    home = tmp_path / "home"
    binary = home / ".local" / "bin" / "rapid-mlx"
    resolved = home.joinpath(*resolved_parts)
    venv = resolved.parent.parent
    venv.mkdir(parents=True)
    receipt = "uv-receipt.toml" if manager == "uv" else "pipx_metadata.json"
    (venv / receipt).write_text("managed")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: str(binary))
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: str(resolved) if Path(p) in {binary, resolved} else str(p),
    )
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.delenv("PIPX_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)

    info = vc.detect_install_method()

    assert info.method == manager
    assert info.upgrade_argv == expected_argv
    assert "install.sh" not in info.upgrade_command


@pytest.mark.parametrize(
    "manager,env_name,relative_parts,expected_argv",
    [
        (
            "uv",
            "UV_TOOL_DIR",
            ("rapid-mlx",),
            ["uv", "tool", "upgrade", "rapid-mlx"],
        ),
        (
            "pipx",
            "PIPX_HOME",
            ("venvs", "rapid-mlx"),
            ["pipx", "upgrade", "rapid-mlx"],
        ),
    ],
)
def test_detects_configured_tool_manager_root(
    tmp_path, monkeypatch, manager, env_name, relative_parts, expected_argv
):
    home = tmp_path / "home"
    binary = home / ".local" / "bin" / "rapid-mlx"
    manager_root = tmp_path / "custom-tools"
    resolved = manager_root.joinpath(*relative_parts, "bin", "rapid-mlx")
    venv = resolved.parent.parent
    venv.mkdir(parents=True)
    receipt = "uv-receipt.toml" if manager == "uv" else "pipx_metadata.json"
    (venv / receipt).write_text("managed")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: str(binary))
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: str(resolved) if Path(p) in {binary, resolved} else str(p),
    )
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.delenv("PIPX_HOME", raising=False)
    monkeypatch.setenv(env_name, str(manager_root))
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)

    info = vc.detect_install_method()

    assert info.method == manager
    assert info.upgrade_argv == expected_argv


def test_global_pipx_requires_manual_privileged_upgrade(tmp_path, monkeypatch):
    home = tmp_path / "home"
    binary = "/usr/local/bin/rapid-mlx"
    pipx_home = tmp_path / "global-pipx"
    resolved = pipx_home / "venvs" / "rapid-mlx" / "bin" / "rapid-mlx"
    resolved.parent.parent.mkdir(parents=True)
    (resolved.parent.parent / "pipx_metadata.json").write_text("managed")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: binary)
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: str(resolved) if Path(p) in {Path(binary), resolved} else str(p),
    )
    monkeypatch.setenv("PIPX_GLOBAL_HOME", str(pipx_home))
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)

    info = vc.detect_install_method()

    assert info.method == "unknown"
    assert info.upgrade_argv == []
    assert info.upgrade_command == "sudo pipx upgrade --global rapid-mlx"


def test_manager_like_path_outside_known_root_falls_back_to_pip(tmp_path, monkeypatch):
    """Directory names and even a receipt are insufficient outside manager roots."""
    home = tmp_path / "home"
    binary = home / ".local" / "bin" / "rapid-mlx"
    venv = tmp_path / "ordinary" / "uv" / "tools" / "rapid-mlx"
    resolved = venv / "bin" / "rapid-mlx"
    venv.mkdir(parents=True)
    (venv / "uv-receipt.toml").write_text("coincidental")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: str(binary))
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: str(resolved) if Path(p) in {binary, resolved} else str(p),
    )
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    info = vc.detect_install_method()

    assert info.method == "pip"
    assert info.upgrade_argv[:3] == [vc.sys.executable, "-m", "pip"]


def test_manager_root_symlink_is_normalized_before_comparison(tmp_path, monkeypatch):
    """Normalize both sides when a manager root itself is reached via symlink."""
    home = tmp_path / "home"
    binary = tmp_path / "uv-bin" / "rapid-mlx"
    configured_root = tmp_path / "uv-tools"
    venv = configured_root / "rapid-mlx"
    expected = venv / "bin" / "rapid-mlx"
    canonical = Path("/private") / expected.relative_to("/")
    venv.mkdir(parents=True)
    (venv / "uv-receipt.toml").write_text("managed")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: str(binary))

    def fake_realpath(path):
        return str(canonical) if Path(path) in {binary, expected} else str(path)

    monkeypatch.setattr("os.path.realpath", fake_realpath)
    monkeypatch.setenv("UV_TOOL_DIR", str(configured_root))
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)

    info = vc.detect_install_method()

    assert info.method == "uv"
    assert info.upgrade_argv == ["uv", "tool", "upgrade", "rapid-mlx"]


def test_install_sh_root_symlink_is_normalized_before_comparison(tmp_path, monkeypatch):
    """Normalize the install.sh root as well as its resolved launcher."""
    home = tmp_path / "home"
    binary = home / ".local/bin/rapid-mlx"
    expected_root = home / ".rapid-mlx"
    canonical_root = Path("/private") / expected_root.relative_to("/")
    canonical_binary = canonical_root / "bin/rapid-mlx"
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: str(binary))

    def fake_realpath(path):
        if Path(path) == binary:
            return str(canonical_binary)
        if Path(path) == expected_root:
            return str(canonical_root)
        return str(path)

    monkeypatch.setattr("os.path.realpath", fake_realpath)
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)

    info = vc.detect_install_method()

    assert info.method == "install_sh"
    assert info.upgrade_argv[:3] == [vc.sys.executable, "-m", "pip"]


def test_detect_install_method_pip_uses_sys_executable(monkeypatch):
    """When the binary path doesn't match brew or install.sh, fall back to
    pip — and use ``sys.executable -m pip`` so the upgrade lands in the
    same Python env that's currently running the CLI (matters when the
    user has multiple python3 installs).
    """
    import sys

    monkeypatch.setattr("shutil.which", lambda _name: "/some/other/path/rapid-mlx")
    monkeypatch.setattr("os.path.realpath", lambda p: p)

    info = vc.detect_install_method()
    assert info.method == "pip"
    assert info.upgrade_command.startswith(sys.executable)
    assert info.upgrade_command.endswith("-m pip install --upgrade rapid-mlx")
    # argv form is shell-safe even if sys.executable contains spaces — that
    # was a P0 in deepseek review (subprocess shell=True path injection).
    assert info.upgrade_argv == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "rapid-mlx",
    ]


def test_detect_install_method_no_binary_falls_back_to_pip(monkeypatch):
    """When ``rapid-mlx`` isn't on PATH (e.g. invoked via
    ``python -m rapid_mlx.cli``), default to pip so the upgrade subcommand
    still works."""
    monkeypatch.setattr("shutil.which", lambda _name: None)

    info = vc.detect_install_method()
    assert info.method == "pip"
    assert info.binary_path is None


# --- prompt_upgrade_if_available ------------------------------------------


@pytest.fixture
def interactive(monkeypatch):
    """Enable the prompt path: TTY on stdin+stderr, not disabled, not in CI."""
    monkeypatch.delenv("RAPID_MLX_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(vc.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(vc.sys.stderr, "isatty", lambda: True)


def test_prompt_returns_false_when_disabled(monkeypatch, interactive):
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    # Disabled MUST short-circuit before fetching anything.
    monkeypatch.setattr(
        vc,
        "get_latest_version",
        lambda force_refresh=False: pytest.fail("network leaked on disabled"),
    )
    assert vc.prompt_upgrade_if_available() is False


def test_prompt_returns_false_when_stdin_not_tty(monkeypatch, interactive):
    monkeypatch.setattr(vc.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        vc,
        "get_latest_version",
        lambda force_refresh=False: pytest.fail("network leaked on non-TTY"),
    )
    assert vc.prompt_upgrade_if_available() is False


def test_prompt_returns_false_when_already_current(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.62")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    with patch("builtins.input") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()


def test_prompt_returns_false_when_local_ahead(monkeypatch, interactive):
    """Dev build one bump ahead of latest — never prompt downward."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.63")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    with patch("builtins.input") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()


@pytest.mark.parametrize(
    "dev_version",
    [
        "0.6.62.dev1+gabcdef",  # editable dev build
        "0.6.61.dev1",  # dev base of in-progress next bump
        "0.6.62rc1",  # release candidate
        "0.6.62a1",  # alpha
        "0.6.62b1",  # beta
        "0.6.62.post1",  # post-release
        "0.6.62+local.build",  # PEP 440 local version
    ],
)
def test_prompt_returns_false_for_pep440_non_final_release(
    monkeypatch, interactive, dev_version
):
    """Real ``_parse_version`` tolerates dev/rc/+ suffixes and returns a tuple,
    which would otherwise let a dev on ``0.6.61.dev1`` get a false prompt for
    ``0.6.62``. The dev-build guard must skip BEFORE parsing, using the real
    parser unmocked so a future regression of the parser doesn't silently
    bypass the guard. DeepSeek finding #3 on PR #428.
    """
    monkeypatch.setattr(vc, "_installed_version", lambda: dev_version)
    # Real _parse_version intentionally NOT mocked — guard must fire first.
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.7.0")
    with patch("builtins.input") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()


def test_prompt_returns_false_when_upgrade_subprocess_fails(monkeypatch, interactive):
    """Brew/pip failure (network, conflict, sudo prompt) must NOT cause
    serve to exit silently. Return False so the caller continues booting
    with the current version. DeepSeek finding #2 on PR #428.
    """
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="brew",
            upgrade_command="brew upgrade rapid-mlx",
            upgrade_argv=["brew", "upgrade", "rapid-mlx"],
        ),
    )
    fake_result = MagicMock(returncode=1)
    with (
        patch("builtins.input", return_value="y"),
        patch("subprocess.run", return_value=fake_result),
    ):
        # Failed upgrade → return False so serve continues with the
        # current installed version. The user sees the exit code and can
        # retry manually.
        assert vc.prompt_upgrade_if_available() is False


def test_prompt_does_not_auto_run_privileged_global_pipx(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="unknown",
            upgrade_command="sudo pipx upgrade --global rapid-mlx",
            upgrade_argv=[],
        ),
    )
    with (
        patch("builtins.input") as inp,
        patch("subprocess.run") as run,
    ):
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()
        run.assert_not_called()


def test_prompt_returns_false_when_offline(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: None)
    with patch("builtins.input") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()


def test_prompt_returns_false_when_user_declines(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip",
            upgrade_command="pip install -U rapid-mlx",
            upgrade_argv=["pip", "install", "-U", "rapid-mlx"],
        ),
    )
    with (
        patch("builtins.input", return_value="n"),
        patch("subprocess.run") as run,
    ):
        assert vc.prompt_upgrade_if_available() is False
        run.assert_not_called()


def test_prompt_returns_true_and_runs_upgrade_on_accept(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="brew",
            upgrade_command="brew upgrade rapid-mlx",
            upgrade_argv=["brew", "upgrade", "rapid-mlx"],
        ),
    )
    fake_result = MagicMock(returncode=0)
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda _: "0.6.62")
    # Empty answer == default Y.
    with (
        patch("builtins.input", return_value=""),
        patch("subprocess.run", return_value=fake_result) as run,
    ):
        assert vc.prompt_upgrade_if_available() is True
        run.assert_called_once_with(["brew", "upgrade", "rapid-mlx"], check=False)


def test_prompt_successful_command_without_upgrade_continues(
    monkeypatch, interactive, capsys, tmp_path
):
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _: True)
    launcher = tmp_path / "rapid-mlx"
    launcher.write_text('#!/bin/sh\necho "rapid-mlx 0.6.61"\n')
    launcher.chmod(0o755)
    monkeypatch.setattr(vc.sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "99.0.0")
    monkeypatch.setattr(vc, "get_release_info", lambda **_: None)
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip",
            upgrade_command="no-op",
            upgrade_argv=[sys.executable, "-c", "pass"],
            binary_path=str(launcher),
        ),
    )
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert vc.prompt_upgrade_if_available() is False
    output = capsys.readouterr().out
    assert "is still 0.6.61" in output
    assert "expected 99.0.0" in output
    assert "Upgrade complete" not in output


def test_prompt_reports_verified_upgrade(monkeypatch, interactive, capsys):
    monkeypatch.setattr(vc.sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "99.0.0")
    monkeypatch.setattr(vc, "get_release_info", lambda **_: None)
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip",
            upgrade_command="no-op",
            upgrade_argv=[sys.executable, "-c", "pass"],
            binary_path=None,
        ),
    )
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda _: "99.0.0")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert vc.prompt_upgrade_if_available() is True
    assert "Upgrade complete: rapid-mlx 99.0.0" in capsys.readouterr().out


def test_prompt_reports_unverified_upgrade(monkeypatch, interactive, capsys):
    monkeypatch.setattr(vc.sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "99.0.0")
    monkeypatch.setattr(vc, "get_release_info", lambda **_: None)
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip",
            upgrade_command="no-op",
            upgrade_argv=[sys.executable, "-c", "pass"],
            binary_path=None,
        ),
    )
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert vc.prompt_upgrade_if_available() is False
    assert "could not be verified" in capsys.readouterr().out


def test_prompt_crosses_minor_boundary(monkeypatch, interactive):
    """The interactive prompt fires across minor bumps (as does the passive
    banner now) — user can still say no."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.62")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.7.0")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip",
            upgrade_command="pip",
            upgrade_argv=["pip"],
        ),
    )
    with patch("builtins.input", return_value="n") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_called_once()


def test_prompt_never_raises(monkeypatch, interactive):
    """A bug in detect_install_method or _installed_version must never crash
    the CLI — silently skip and return False so serve continues to boot.
    """

    def boom():
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(vc, "_installed_version", boom)
    # Must not raise.
    assert vc.prompt_upgrade_if_available() is False


def test_prompt_returns_false_on_keyboard_interrupt(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip", upgrade_command="pip", upgrade_argv=["pip"]
        ),
    )
    with (
        patch("builtins.input", side_effect=KeyboardInterrupt()),
        patch("subprocess.run") as run,
    ):
        assert vc.prompt_upgrade_if_available() is False
        run.assert_not_called()


@pytest.mark.parametrize("disabled", [False, True])
def test_print_helper_returns_none_without_notice(monkeypatch, disabled):
    monkeypatch.setattr(vc, "_explicitly_disabled", lambda: disabled)
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(vc, "get_latest_version", lambda: "1.0.0")
    assert vc.print_staleness_warning_if_any(allow_non_tty=True) is None


def test_print_helper_returns_none_on_print_error(monkeypatch):
    monkeypatch.setattr(
        vc, "_staleness_warning_versions", lambda **kw: ("1.0.0", "1.1.0")
    )
    monkeypatch.setattr(
        "builtins.print", MagicMock(side_effect=OSError("closed stderr"))
    )
    assert vc.print_staleness_warning_if_any(allow_non_tty=True) is None


@pytest.mark.parametrize("uv", [None, "/tools/uv"])
def test_install_sh_without_pip_uses_available_installer(tmp_path, monkeypatch, uv):
    home = tmp_path / "home"
    binary = str(home / ".rapid-mlx/bin/rapid-mlx")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(
        "shutil.which", lambda name: binary if name == "rapid-mlx" else uv
    )
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _p: True)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    info = vc.detect_install_method()
    expected = (
        [uv, "pip", "install", "--python", vc.sys.executable, "--upgrade", "rapid-mlx"]
        if uv
        else [vc.sys.executable, "-m", "pip", "install", "--upgrade", "rapid-mlx"]
    )
    assert info.upgrade_argv == expected
    import shlex

    assert shlex.split(info.upgrade_command) == expected


@pytest.mark.parametrize(
    "spec,version,expected",
    [
        (">=3.12", (3, 11, 9), False),
        (">=3.12", (3, 12, 0), True),
        ("garbage", (3, 11, 0), True),
        (None, (3, 11, 0), True),
    ],
)
def test_python_supports(spec, version, expected):
    assert vc._python_supports(spec, version) is expected


@pytest.mark.parametrize(
    "fetch,body,expected",
    [
        ("python", {"info": {"requires_python": ">=3.12"}}, ">=3.12"),
        ("python", {"info": {"requires_python": "x" * 201}}, None),
        ("brew", {"versions": {"stable": "1.0.1"}}, "1.0.1"),
        ("brew", {"versions": {"stable": "bad"}}, None),
        ("brew", {}, None),
    ],
)
def test_metadata_fetches(monkeypatch, fetch, body, expected):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(body).encode()
    opener = MagicMock(return_value=response)
    monkeypatch.setattr(vc.urllib.request, "urlopen", opener)
    result = (
        vc._fetch_requires_python("1.0.1")
        if fetch == "python"
        else vc._fetch_brew_formula_version()
    )
    assert result == expected
    request = opener.call_args.args[0]
    assert request.get_header("User-agent") == vc.USER_AGENT
    assert opener.call_args.kwargs["timeout"] == vc.NETWORK_TIMEOUT_SECONDS


def test_metadata_fetches_fail_open(monkeypatch):
    assert vc._fetch_requires_python("1.0.1") is vc._REQUIRES_PYTHON_FAILED
    assert vc._fetch_brew_formula_version() is None
    monkeypatch.setattr(
        vc.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("invalid version reached network"),
    )
    assert vc._fetch_requires_python("../bad") is vc._REQUIRES_PYTHON_FAILED


def test_cache_optional_fields(monkeypatch):
    vc._write_cache("1.0.1", ">=3.12", "1.0.0")
    assert vc._read_cache()["requires_python"] == ">=3.12"
    assert vc._read_cache()["brew_stable"] == "1.0.0"
    vc._cache_path().write_text(
        json.dumps(
            {
                "latest": "1.0.1",
                "requires_python": [],
                "brew_stable": "bad",
                "requires_python_checked": "yes",
                "requires_python_failed_at": True,
            }
        )
    )
    cache = vc._read_cache()
    assert cache == {"latest": "1.0.1"}
    monkeypatch.setattr(
        vc,
        "_fetch_requires_python",
        lambda version: None,
    )
    monkeypatch.setattr(vc, "_fetch_brew_formula_version", lambda: "1.0.1")
    release = vc.get_release_info(need_brew=True)
    assert release.requires_python is None
    assert release.brew_stable == "1.0.1"
    assert vc._read_cache()["brew_stable"] == "1.0.1"


@pytest.mark.parametrize("spec", [None, ">=3.12"])
def test_cached_missing_python_metadata_fetched_once(monkeypatch, spec):
    import os
    import time

    path = vc._cache_path()
    timestamp = int(time.time()) - 100
    path.write_text(
        json.dumps({"latest": "1.0.1", "brew_stable": "1.0.0", "ts": timestamp})
    )
    os.utime(path, (timestamp, timestamp))
    fetch = MagicMock(return_value=spec)
    monkeypatch.setattr(vc, "_fetch_requires_python", fetch)
    assert vc.get_release_info().requires_python == spec
    assert vc.get_release_info().requires_python == spec
    fetch.assert_called_once_with("1.0.1")
    cache = vc._read_cache()
    assert cache["ts"] == timestamp
    assert path.stat().st_mtime == timestamp
    assert cache["brew_stable"] == "1.0.0"
    assert cache["requires_python_checked"] is True


def test_fresh_cache_marks_python_metadata_checked(monkeypatch):
    monkeypatch.setattr(vc, "_fetch_latest", lambda: "1.0.1")
    fetch = MagicMock(return_value=None)
    monkeypatch.setattr(vc, "_fetch_requires_python", fetch)
    assert vc.get_release_info().requires_python is None
    assert vc.get_release_info().requires_python is None
    fetch.assert_called_once_with("1.0.1")
    assert vc._read_cache()["requires_python_checked"] is True


@pytest.mark.parametrize("fresh", [True, False])
def test_failed_python_metadata_is_retried(monkeypatch, fresh):
    import os
    import time

    path = vc._cache_path()
    timestamp = int(time.time()) - 100
    if not fresh:
        path.write_text(json.dumps({"latest": "1.0.1", "ts": timestamp}))
        os.utime(path, (timestamp, timestamp))
    monkeypatch.setattr(vc, "_fetch_latest", lambda: "1.0.1")
    monkeypatch.setattr(vc, "_fetch_brew_formula_version", lambda: "1.0.0")
    fetch = MagicMock(side_effect=[vc._REQUIRES_PYTHON_FAILED, ">=3.12"])
    monkeypatch.setattr(vc, "_fetch_requires_python", fetch)

    release = vc.get_release_info(need_brew=True)
    assert (release.latest, release.requires_python, release.brew_stable) == (
        "1.0.1",
        None,
        "1.0.0",
    )
    cache = vc._read_cache()
    assert "requires_python_checked" not in cache
    original_ts, original_mtime = cache["ts"], path.stat().st_mtime
    if not fresh:
        assert original_ts == original_mtime == timestamp

    assert vc.get_release_info(need_brew=True).requires_python is None
    assert fetch.call_count == 1
    failed_at = cache["requires_python_failed_at"]
    monkeypatch.setattr(vc.time, "time", lambda: failed_at + 3600)
    assert vc.get_release_info(need_brew=True).requires_python == ">=3.12"
    assert fetch.call_count == 2
    cache = vc._read_cache()
    assert cache["requires_python_checked"] is True
    assert cache["requires_python"] == ">=3.12"
    assert "requires_python_failed_at" not in cache
    assert cache["brew_stable"] == "1.0.0"
    assert cache["ts"] == original_ts
    assert path.stat().st_mtime == original_mtime


@pytest.mark.parametrize(
    "spec,expected",
    [
        (">=3.10,>=3.12", "3.12"),
        (">3.11", "3.11"),
        ("==3.13.*", "3.13"),
        ("<4", "3.10"),
        (">3.11.1", "3.11"),
        (">=3.13.0", "3.13"),
        (">=3.12,!=3.12.*", "3.13"),
        ("invalid", "3.12"),
        (">=4", "3.12"),
    ],
)
def test_python_upgrade_instructions_highest_floor(spec, expected):
    info = vc.InstallInfo("uv", "upgrade", ["upgrade"])
    assert vc.python_upgrade_instructions(info, spec) == [
        f"uv tool upgrade --python {expected} rapid-mlx"
    ]


@pytest.mark.parametrize(
    "method,spec,brew,expected",
    [
        ("pip", ">=3.12", None, "requires Python >=3.12"),
        ("brew", None, "1.0.0", None),
        ("brew", None, "1.0.1", "latest 1.0.1"),
    ],
)
def test_eligibility_warning(monkeypatch, method, spec, brew, expected):
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc, "get_release_info", lambda *a, **kw: vc.ReleaseInfo("1.0.2", spec, brew)
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(method, "upgrade", ["upgrade"]),
    )
    monkeypatch.setattr(vc, "_python_supports", lambda spec: False)
    notice = vc.staleness_warning()
    if expected is None:
        assert notice is None
    else:
        assert expected in notice
        if spec:
            assert "run `rapid-mlx upgrade`" not in notice


@pytest.mark.parametrize("method,block", [("pip", "python"), ("brew", "brew_lag")])
def test_prompt_blocks_unavailable_upgrade(
    monkeypatch, interactive, capsys, method, block
):
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    monkeypatch.setattr(vc.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda *a, **kw: vc.ReleaseInfo("1.0.1", ">=3.12", "1.0.0"),
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(method, "upgrade", ["upgrade"]),
    )
    monkeypatch.setattr(vc, "_python_supports", lambda spec: False)
    with patch("builtins.input") as inp, patch("subprocess.run") as run:
        assert not vc.prompt_upgrade_if_available()
    inp.assert_not_called()
    run.assert_not_called()
    output = capsys.readouterr().err
    assert ("requires Python" in output) == (block == "python")
    if block == "python":
        assert vc.print_staleness_warning_if_any(allow_non_tty=True) == "1.0.1"
        assert capsys.readouterr().err == ""


def test_fresh_process_version_uses_launcher(monkeypatch):
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _: True)
    info = vc.InstallInfo("uv", "upgrade", ["upgrade"], "/tools/rapid-mlx")
    monkeypatch.setattr(vc.os, "access", lambda *args: True)
    with patch(
        "subprocess.run",
        return_value=SimpleNamespace(
            returncode=0, stdout="noise rapid-mlx 1.0.0\nrapid-mlx 1.0.1\n"
        ),
    ) as run:
        assert vc.installed_version_in_fresh_process(info) == "1.0.1"
    run.assert_called_once_with(
        ["/tools/rapid-mlx", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_fresh_process_version_falls_back_to_python(monkeypatch):
    info = vc.InstallInfo("pip", "upgrade", ["upgrade"])
    with patch(
        "subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="1.0.1\n")
    ) as run:
        assert vc.installed_version_in_fresh_process(info) == "1.0.1"
    assert run.call_args.args[0][0:3] == [vc.sys.executable, "-I", "-c"]
    with patch("subprocess.run", side_effect=OSError):
        assert vc.installed_version_in_fresh_process(info) is None


@pytest.mark.parametrize(
    "prefix,base_prefix,enabled,expected_paths",
    [
        ("/base", "/base", True, ["/user/site"]),
        ("/venv", "/base", True, []),
        ("/base", "/base", False, []),
    ],
)
def test_fresh_process_user_site_conditions(
    monkeypatch, prefix, base_prefix, enabled, expected_paths
):
    monkeypatch.setattr(sys, "prefix", prefix)
    monkeypatch.setattr(sys, "base_prefix", base_prefix)
    monkeypatch.setattr(site, "ENABLE_USER_SITE", enabled)
    monkeypatch.setattr(site, "getusersitepackages", lambda: "/user/site")
    info = vc.InstallInfo("pip", "upgrade", ["upgrade"])
    with patch(
        "subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="\n")
    ) as run:
        assert vc.installed_version_in_fresh_process(info) is None
    assert run.call_args.args[0][4:] == expected_paths


def test_fresh_process_finds_user_site_without_launcher(tmp_path):
    base_python = os.path.realpath(getattr(sys, "_base_executable", sys.executable))
    env = os.environ.copy()
    env.update(
        PYTHONUSERBASE=str(tmp_path / "ub"), PATH="/usr/bin:/bin", HOME=str(tmp_path)
    )
    env.pop("PYTHONPATH", None)
    is_base = subprocess.run(
        [base_python, "-c", "import sys; print(sys.prefix == sys.base_prefix)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    if is_base.stdout.strip() != "True":
        pytest.skip("base interpreter is a virtual environment")
    result = subprocess.run(
        [base_python, "-c", "import site; print(site.getusersitepackages())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    dist_info = Path(result.stdout.strip()) / "rapid_mlx-9.9.9.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rapid-mlx\nVersion: 9.9.9\n"
    )
    paths = [str(Path(__file__).resolve().parents[1]), *site.getsitepackages()]
    script = (
        f"import sys; sys.path[:0] = {paths!r}; "
        "from rapid_mlx._version_check import InstallInfo, installed_version_in_fresh_process; "
        "print(installed_version_in_fresh_process(InstallInfo("
        "method='pip', upgrade_command='x', upgrade_argv=['x'], binary_path=None)))"
    )
    result = subprocess.run(
        [base_python, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "9.9.9"


def test_python_supports_without_packaging(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_packaging(name, *args, **kwargs):
        if name == "packaging.specifiers":
            raise ImportError("packaging unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_packaging)
    assert vc._python_supports(">=99", (3, 10, 0)) is True


def test_cache_rewrite_survives_missing_cache_file(monkeypatch):
    monkeypatch.setattr(vc, "_read_cache", lambda: {"latest": "1.0.1"})
    monkeypatch.setattr(vc, "_fetch_requires_python", lambda version: ">=3.12")
    with patch.object(vc, "_write_cache") as write:
        release = vc.get_release_info()
    assert (release.latest, release.requires_python) == ("1.0.1", ">=3.12")
    write.assert_not_called()


def test_notices_fail_open_when_install_detection_fails(monkeypatch):
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(vc, "get_latest_version", lambda: "1.0.1")
    monkeypatch.setattr(
        vc, "detect_install_method", MagicMock(side_effect=OSError("offline"))
    )
    assert vc.staleness_warning() == vc.format_staleness_warning("1.0.0", "1.0.1")
    assert vc._format_available_warning(
        "1.0.0", "1.0.1"
    ) == vc.format_staleness_warning("1.0.0", "1.0.1")


def test_prompt_offers_available_brew_formula(monkeypatch, interactive, capsys):
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(vc, "get_latest_version", lambda: "1.0.2")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda **kw: vc.ReleaseInfo("1.0.2", brew_stable="1.0.1"),
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo("brew", "brew upgrade", ["brew"]),
    )
    with patch("builtins.input", return_value="n"), patch("subprocess.run") as run:
        assert vc.prompt_upgrade_if_available() is False
    run.assert_not_called()
    assert "available: 1.0.1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "method,command,expected",
    [
        ("brew", "brew upgrade rapid-mlx", "brew upgrade rapid-mlx"),
        (
            "unknown",
            "sudo pipx upgrade --global rapid-mlx",
            "sudo pipx reinstall --global --python python3.12 rapid-mlx",
        ),
    ],
)
def test_python_upgrade_instructions_for_managed_installs(method, command, expected):
    assert vc.python_upgrade_instructions(
        vc.InstallInfo(method, command, []), ">=3.12"
    ) == [expected]


def test_fresh_process_failure_does_not_accept_output():
    info = vc.InstallInfo("pip", "upgrade", ["upgrade"])
    with patch(
        "subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="1.0.1")
    ):
        assert vc.installed_version_in_fresh_process(info) is None


def test_shell_launcher_without_valid_trampoline_is_rejected(tmp_path, monkeypatch):
    launcher = tmp_path / "rapid-mlx"
    launcher.write_text("#!/bin/sh\necho hello\n")
    monkeypatch.setattr(
        vc,
        "distribution",
        lambda name: SimpleNamespace(entry_points=[SimpleNamespace(name="rapid-mlx")]),
    )
    assert vc._launcher_belongs_to_running_install(str(launcher)) is False


def test_corrupt_cache_is_ignored():
    vc._cache_path().write_text("{")
    assert vc._read_cache() is None


def test_unparseable_installed_version_has_no_staleness_warning(monkeypatch):
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    monkeypatch.setattr(vc, "_installed_version", lambda: "???")
    assert vc.staleness_warning() is None


@pytest.mark.parametrize("version", [None, "???"])
def test_prompt_skips_missing_or_unparseable_version(monkeypatch, interactive, version):
    monkeypatch.setattr(vc, "_installed_version", lambda: version)
    assert vc.prompt_upgrade_if_available() is False


def test_python_upgrade_instructions_without_packaging(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def no_packaging(name, *args, **kwargs):
        if name == "packaging.specifiers":
            raise ImportError("unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_packaging)
    info = vc.InstallInfo("uv", "upgrade", ["upgrade"])
    assert vc.python_upgrade_instructions(info, ">3.11.1") == [
        "uv tool upgrade --python 3.12 rapid-mlx"
    ]


def test_user_pip_install_upgrade_and_python_instructions(tmp_path, monkeypatch):
    import site
    import sys

    import rapid_mlx

    monkeypatch.setattr(vc, "_externally_managed", lambda: False)
    user_site = tmp_path / "site-packages"
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(
        rapid_mlx, "__file__", str(user_site / "rapid_mlx" / "__init__.py")
    )
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr("shutil.which", lambda _: None)
    info = vc.detect_install_method()
    assert info.upgrade_argv == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "--upgrade",
        "rapid-mlx",
    ]
    assert "pip install --user --upgrade rapid-mlx" in info.upgrade_command
    instructions = vc.python_upgrade_instructions(info, ">=3.12")
    assert any("python3.12 -m venv" in line for line in instructions)
    assert all("--user" not in line for line in instructions)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    assert "--user" not in vc.detect_install_method().upgrade_argv


@pytest.mark.parametrize(
    ("method", "argv", "label"),
    [
        ("brew", [], "Homebrew"),
        ("install_sh", [], "one-line installer"),
        ("uv", [], "uv tool"),
        ("pipx", [], "pipx"),
        ("pip", [], "pip"),
        ("pip", ["pip", "install", "--user"], "pip --user"),
        ("other", [], "unknown"),
    ],
)
def test_install_method_label(method, argv, label):
    assert vc.install_method_label(vc.InstallInfo(method, "", argv)) == label


def test_pip_python_upgrade_uses_versioned_environment():
    instructions = vc.python_upgrade_instructions(
        vc.InstallInfo("pip", "", []), ">=3.12"
    )
    assert instructions[1] == (
        "python3.12 -m venv ~/.venvs/rapid-mlx-py312 && "
        "~/.venvs/rapid-mlx-py312/bin/pip install rapid-mlx"
    )
    assert instructions[2] == (
        "Add ~/.venvs/rapid-mlx-py312/bin to PATH (or call it directly); "
        "remove the old environment only after the new one works."
    )


@pytest.mark.parametrize(
    "marker,venv,user",
    [
        (True, False, True),
        (True, False, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_pip_externally_managed_marker(tmp_path, monkeypatch, marker, venv, user):
    import site
    import sysconfig

    import rapid_mlx

    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    if marker:
        (stdlib / "EXTERNALLY-MANAGED").touch()
    user_site = tmp_path / "site-packages"
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(stdlib))
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(
        rapid_mlx,
        "__file__",
        str((user_site if user else tmp_path) / "rapid_mlx" / "__init__.py"),
    )
    monkeypatch.setattr(
        vc.sys, "prefix", str(tmp_path / "venv") if venv else vc.sys.base_prefix
    )
    monkeypatch.setattr("shutil.which", lambda _: None)
    argv = vc.detect_install_method().upgrade_argv
    assert ("--break-system-packages" in argv) == (marker and user and not venv)
    assert ("--user" in argv) == (user and not venv)
    if marker and user and not venv:
        assert argv[argv.index("install") + 1] == "--break-system-packages"


def test_externally_managed_handles_oserror(monkeypatch):
    def fail(_name):
        raise OSError("unavailable")

    monkeypatch.setattr(vc.sysconfig, "get_path", fail)
    assert vc._externally_managed() is False


def test_installation_docs_identify_running_python():
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[1] / "docs/getting-started/installation.md"
    ).read_text()
    assert (
        "Check `rapid-mlx --version`, then run `rapid-mlx doctor --only python`: its first check line (for example `✓ Python 3.11.9`) is the Python your install actually runs."
        in text
    )


@pytest.mark.parametrize(
    "version", ["1.0.1a1", "1.0.1b1", "1.0.1.dev0", "1.0.1rc1", "1.0.1.post1"]
)
def test_fresh_process_preserves_version_suffix(monkeypatch, version):
    info = vc.InstallInfo("pip", "upgrade", ["upgrade"], "/tools/rapid-mlx")
    monkeypatch.setattr(vc.os, "access", lambda *args: True)
    monkeypatch.setattr(vc, "_launcher_belongs_to_running_install", lambda _: True)
    with patch(
        "subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout=f"rapid-mlx {version}\n"),
    ):
        assert vc.installed_version_in_fresh_process(info) == version


def test_fresh_process_rejects_foreign_venv_launcher(tmp_path, monkeypatch):
    for name in ("a", "b"):
        venv = tmp_path / name
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(f"home = {sys.base_prefix}\n")
        (venv / "bin" / "python").symlink_to(sys._base_executable or sys.executable)
    python_a = str(tmp_path / "a" / "bin" / "python")
    launcher = tmp_path / "b" / "bin" / "rapid-mlx"
    launcher.write_text(f"#!{launcher.parent / 'python'}\n")
    launcher.chmod(0o755)
    monkeypatch.setattr(sys, "executable", python_a)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "a"))
    monkeypatch.setattr(
        vc,
        "distribution",
        lambda _: SimpleNamespace(entry_points=[SimpleNamespace(name="rapid-mlx")]),
    )
    assert not vc._launcher_belongs_to_running_install(str(launcher))
    info = vc.InstallInfo("pip", "upgrade", [python_a, "-m", "pip"], str(launcher))

    def probe(argv, **kwargs):
        output = "rapid-mlx 1.0.1\n" if argv[0] == str(launcher) else "1.0.1\n"
        return SimpleNamespace(returncode=0, stdout=output)

    with patch("subprocess.run", side_effect=probe) as run:
        assert vc.installed_version_in_fresh_process(info) == "1.0.1"
    assert run.call_args.args[0][:3] == [python_a, "-I", "-c"]


@pytest.mark.parametrize(
    "version",
    ["0.6.62a1", "0.6.62b1", "0.6.62.dev0", "0.6.62rc1", "0.6.62.post1", "0.6.63"],
)
def test_prompt_requires_exact_upgrade_target(monkeypatch, interactive, version):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda **_: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo("pip", "upgrade", ["upgrade"]),
    )
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda _: version)
    with (
        patch("builtins.input", return_value=""),
        patch("subprocess.run", return_value=SimpleNamespace(returncode=0)),
    ):
        assert vc.prompt_upgrade_if_available() is False


@pytest.mark.parametrize("without_packaging", [False, True])
@pytest.mark.parametrize(
    "version,expected",
    [
        ("1.0.1", True),
        ("1.0.1.dev0", False),
        ("1.0.1.post1", False),
        ("1.0.1b1", False),
        ("garbage", False),
    ],
)
def test_verification_version_equality(
    monkeypatch, without_packaging, version, expected
):
    import builtins

    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if without_packaging and name == "packaging.version":
            raise ImportError("packaging unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    assert vc._versions_equal(version, "1.0.1") is expected


def test_verification_version_equality_normalizes_release():
    assert vc._versions_equal("1.0.1.0", "v1.0.1")
