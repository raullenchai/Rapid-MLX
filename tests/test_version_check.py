# SPDX-License-Identifier: Apache-2.0
"""Tests for the staleness-warning helper.

The helper is opt-in (TTY+no-CI), cache-aware, and fail-silent on
network errors. Tests pin those guarantees so a future "let's add a
real call" change can't accidentally break the CLI on an offline
laptop.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    info = vc.detect_install_method()
    assert info.method == "brew"


def test_detect_install_method_install_sh(tmp_path, monkeypatch):
    """install.sh drops the binary in ``~/.local/bin`` — re-running the
    script is the only sane upgrade path for this install class.
    """
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    fake_binary = str(local_bin / "rapid-mlx")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr("os.path.realpath", lambda p: p)

    info = vc.detect_install_method()
    assert info.method == "install_sh"
    assert "install.sh" in info.upgrade_command
    # Upgrade prefers the canonical rapidmlx.com host and falls back to the
    # raullenchai.github.io Pages mirror — both present, primary first, wired
    # with ``||`` so the mirror is only reached if rapidmlx.com fails.
    cmd = info.upgrade_command
    assert "https://rapidmlx.com/install.sh" in cmd
    assert "https://raullenchai.github.io/Rapid-MLX/install.sh" in cmd
    assert cmd.index("rapidmlx.com") < cmd.index("github.io")
    assert "||" in cmd
    # Fetch to a temp file and run only a complete download (never ``| bash``):
    # guards against a false-success on total outage and a partial/hybrid script.
    assert "mktemp" in cmd
    assert "set -e" in cmd
    assert "| bash" not in cmd
    # Bounded timeouts so a blackholed primary fails fast into the fallback
    # instead of hanging forever.
    assert "--connect-timeout" in cmd
    assert "--max-time" in cmd


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

    info = vc.detect_install_method()
    assert info.method == "install_sh"
    assert "install.sh" in info.upgrade_command
    cmd = info.upgrade_command
    assert "https://rapidmlx.com/install.sh" in cmd
    assert "https://raullenchai.github.io/Rapid-MLX/install.sh" in cmd
    assert cmd.index("rapidmlx.com") < cmd.index("github.io")
    assert "||" in cmd
    # Pipe needs a shell — wrapped as ``bash -c <pipe>``, never `shell=True`.
    assert info.upgrade_argv[:2] == ["bash", "-c"]
    # The executed argv carries the same primary-first fallback pipeline.
    argv_cmd = info.upgrade_argv[2]
    assert "https://rapidmlx.com/install.sh" in argv_cmd
    assert argv_cmd.index("rapidmlx.com") < argv_cmd.index("github.io")


def _run_install_sh_upgrade(
    tmp_path, monkeypatch, *, primary_ok, mirror_ok, primary_stall=False
):
    """Execute the real install_sh upgrade_argv with a stub ``curl`` on PATH.

    The stub writes a host-tagged marker script to the ``-o`` target and exits 0
    for an "ok" host, or exits non-zero (writing nothing) for a "down" host —
    letting us assert the shell's fetch-to-temp-then-run semantics end to end.
    With ``primary_stall`` the primary first asserts the bounded-timeout flags
    reached it, then exits 28 (curl's timeout code) after a short sleep,
    simulating a blackholed host hitting ``--max-time``.
    Returns ``(returncode, sentinel_text_or_None)``.
    """
    import subprocess

    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    fake_binary = str(home / ".local" / "bin" / "rapid-mlx")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr("os.path.realpath", lambda p: p)
    argv = vc.detect_install_method().upgrade_argv

    sentinel = tmp_path / "ran.txt"
    primary_flags = tmp_path / "primary_flags.txt"
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    curl = stub_dir / "curl"
    # ``curl -fsSL --connect-timeout N --max-time N <url> -o <file>`` — capture
    # the whole argv, find the url + -o target, decide ok/fail by host, and
    # (on ok) drop a script that records which host served it. Every primary
    # invocation records whether the bounded-timeout flags were present to
    # ``primary_flags`` — a signal independent of the fallback path, so a test
    # can assert the flags actually reached curl rather than inferring it from a
    # mirror run (which happens on *any* primary failure). In stall mode the
    # primary then times out (exit 28), emulating a blackholed host.
    curl.write_text(
        "#!/bin/bash\n"
        'argv="$*"; url=""; out=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in -o) out="$2"; shift 2;; https://*) url="$1"; shift;;'
        " *) shift;; esac\n"
        "done\n"
        f'primary_ok="{int(primary_ok)}"; mirror_ok="{int(mirror_ok)}"\n'
        f'primary_stall="{int(primary_stall)}"\n'
        'case "$url" in\n'
        "  *rapidmlx.com*)\n"
        '    case "$argv" in *--connect-timeout*--max-time*)'
        f" echo yes > {primary_flags};; *) echo no > {primary_flags};; esac\n"
        '    if [ "$primary_stall" = 1 ]; then sleep 0.2; exit 28; fi\n'
        '    [ "$primary_ok" = 1 ] || exit 22;'
        f" printf 'printf primary > {sentinel}\\n' > \"$out\";;\n"
        '  *github.io*) [ "$mirror_ok" = 1 ] || exit 22;'
        f" printf 'printf mirror > {sentinel}\\n' > \"$out\";;\n"
        "  *) exit 22;;\n"
        "esac\n"
    )
    curl.chmod(0o755)

    env = dict(os.environ, PATH=f"{stub_dir}:{os.environ['PATH']}")
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
    got = sentinel.read_text() if sentinel.exists() else None
    return proc.returncode, got


def test_install_sh_upgrade_prefers_primary(tmp_path, monkeypatch):
    """Primary reachable → run the rapidmlx.com script; mirror untouched."""
    rc, ran = _run_install_sh_upgrade(
        tmp_path, monkeypatch, primary_ok=True, mirror_ok=True
    )
    assert rc == 0
    assert ran == "primary"


def test_install_sh_upgrade_falls_back_to_mirror(tmp_path, monkeypatch):
    """Primary down → transparently run the github.io mirror script."""
    rc, ran = _run_install_sh_upgrade(
        tmp_path, monkeypatch, primary_ok=False, mirror_ok=True
    )
    assert rc == 0
    assert ran == "mirror"


def test_install_sh_upgrade_total_outage_fails_loudly(tmp_path, monkeypatch):
    """Both hosts down → non-zero exit and NO installer runs. Regression for a
    naive ``curl A || curl B | bash`` that returns bash's status and would
    falsely report success (running nothing) when both fetches fail.
    """
    rc, ran = _run_install_sh_upgrade(
        tmp_path, monkeypatch, primary_ok=False, mirror_ok=False
    )
    assert rc != 0
    assert ran is None


def test_install_sh_upgrade_stalled_primary_times_out_to_mirror(tmp_path, monkeypatch):
    """A blackholed primary that hangs (rather than returning a clean HTTP
    error) must not stall the upgrade forever: the bounded ``--connect-timeout``
    /``--max-time`` flags reach the primary curl, it aborts (exit 28), and the
    github.io mirror serves the install.
    """
    rc, ran = _run_install_sh_upgrade(
        tmp_path, monkeypatch, primary_ok=True, mirror_ok=True, primary_stall=True
    )
    assert rc == 0
    assert ran == "mirror"
    # Assert the flags reached the *primary* invocation directly — not inferred
    # from the mirror run, which fires on any primary failure. Drop either flag
    # from the production command and this records "no" and the test fails.
    flags = (tmp_path / "primary_flags.txt").read_text().strip()
    assert flags == "yes"


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
    ``python -m vllm_mlx.cli``), default to pip so the upgrade subcommand
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
    # Empty answer == default Y.
    with (
        patch("builtins.input", return_value=""),
        patch("subprocess.run", return_value=fake_result) as run,
    ):
        assert vc.prompt_upgrade_if_available() is True
        run.assert_called_once_with(["brew", "upgrade", "rapid-mlx"], check=False)


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
