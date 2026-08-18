# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified server ready/connect output SSOT.

``vllm_mlx.connect`` is the single source of truth for how a running server's
endpoints are rendered — both the serve lifespan banner and ``rapid-mlx
connect`` (human + ``--json``) consume it. These tests lock:

* the endpoint URL derivations (base vs OpenAI ``/v1`` vs Anthropic),
* the rendered banner shape (Ready / OpenAI / Anthropic / Model / Connect),
* the machine-readable JSON shape (for the desktop / other tooling),
* the socket-activation (inherited-fd) fallback shape,
* the ``connect_command`` plumbing for all three target forms.
"""

from __future__ import annotations

import argparse
import io
import json
import shlex
from contextlib import redirect_stdout

import pytest

from vllm_mlx import connect
from vllm_mlx.cli import connect_command


def _endpoints(host="localhost", port=8000, model="qwen3.6-35b-4bit"):
    return connect.ServerEndpoints(host, port, model=model)


def _run_connect(*, target=None, json_=False, host=None, port=None, model=None):
    """Invoke ``connect_command`` the way argparse drives it, capturing stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        connect_command(
            argparse.Namespace(
                target=target,
                json=json_,
                host=host,
                port=port,
                model=model,
            )
        )
    return buf.getvalue()


# --- SSOT endpoint derivations ----------------------------------------------
def test_endpoint_urls():
    ep = _endpoints(host="127.0.0.1", port=9000)
    assert ep.base_url == "http://127.0.0.1:9000"
    assert ep.openai_url == "http://127.0.0.1:9000/v1"
    # Anthropic base has NO trailing /v1 (the SDK appends /v1/messages itself).
    assert ep.anthropic_url == "http://127.0.0.1:9000"


def test_json_shape_matches_banner_endpoints():
    ep = _endpoints()
    d = ep.to_dict()
    assert d == {
        "ready": "http://localhost:8000",
        "openai": "http://localhost:8000/v1",
        "anthropic": "http://localhost:8000",
        "model": "qwen3.6-35b-4bit",
    }


def test_render_banner_matches_spec():
    out = connect.render_banner(_endpoints())
    assert "Ready: http://localhost:8000" in out
    assert "OpenAI:    http://localhost:8000/v1" in out
    assert "Anthropic: http://localhost:8000" in out
    assert "Model:     qwen3.6-35b-4bit" in out
    # First-class agent setup commands carry the real endpoint so a user
    # copying from the banner wires up the actual server, not localhost.
    assert (
        "rapid-mlx agents claude-code --setup --base-url http://localhost:8000/v1"
        in out
    )
    assert (
        "rapid-mlx agents continue --setup --base-url http://localhost:8000/v1" in out
    )
    assert "rapid-mlx connect openai-python" in out


def test_listen_fd_shape():
    ep = connect.ServerEndpoints("", 0, model=None, listen_fd=7)
    out = connect.render_banner(ep)
    assert "Ready: inherited fd 7" in out
    assert "OpenAI:" not in out  # no known address → no endpoint rows
    d = ep.to_dict()
    assert d["ready"] == "inherited fd 7"
    assert d["openai"] is None
    assert d["anthropic"] is None
    assert d["listen_fd"] == 7


# --- endpoints_from_bind (serve-lifespan wiring) ----------------------------
def test_endpoints_from_bind_host_port():
    ep = connect.endpoints_from_bind("localhost", 9123, model="gpt-oss-20b")
    assert ep.base_url == "http://localhost:9123"
    assert ep.model == "gpt-oss-20b"
    assert ep.listen_fd is None


def test_endpoints_from_bind_prefers_fd_when_no_host_port():
    # Mirrors the serve superisor handoff: fd set but no host/port known.
    ep = connect.endpoints_from_bind(None, None, listen_fd=5)
    assert ep.listen_fd == 5


def test_endpoints_from_bind_host_port_wins_over_fd():
    ep = connect.endpoints_from_bind("localhost", 8000, listen_fd=5)
    assert ep.listen_fd is None
    assert ep.base_url == "http://localhost:8000"


# --- connect_command plumbing ------------------------------------------------
def test_connect_no_target_renders_banner(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model="m1"),
    )
    # A live server is present for this shape check.
    monkeypatch.setattr(connect, "probe_server_alive", lambda *a, **k: True)
    out = _run_connect()
    assert "Ready: http://localhost:8000" in out
    assert "Connect:" in out
    assert "OpenAI:" in out


def test_probe_server_alive_keys_on_healthz_status_code(monkeypatch):
    """#1999: /healthz is rapid-mlx-specific, so a non-404 response (2xx, or a
    DDTree auth 401/403) counts as alive across all serving modes; a 404 (an
    unrelated HTTP server) or a refused/timed-out connection does not. The body
    is not inspected — the standard / DFlash / DDTree bodies differ."""
    import urllib.error
    import urllib.request

    def _ok(url, timeout=None):
        return object()  # urlopen returns without raising on 2xx

    def _raise(code):
        def _open(url, timeout=None):
            raise urllib.error.HTTPError(url, code, "err", {}, None)

        return _open

    def _conn(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    # 2xx (any serving mode) → alive.
    monkeypatch.setattr(urllib.request, "urlopen", _ok)
    assert connect.probe_server_alive("localhost", 8000) is True

    # Auth-gated DDTree /healthz answers 401/403 → still a server there.
    for code in (401, 403):
        monkeypatch.setattr(urllib.request, "urlopen", _raise(code))
        assert connect.probe_server_alive("localhost", 8000) is True, code

    # 404 (unrelated server), 503 (up but draining, refusing new work), and
    # 500 (broken) are all "don't hand a client this endpoint".
    for code in (404, 500, 503):
        monkeypatch.setattr(urllib.request, "urlopen", _raise(code))
        assert connect.probe_server_alive("localhost", 8000) is False, code

    # Nothing listening: connection refused.
    monkeypatch.setattr(urllib.request, "urlopen", _conn)
    assert connect.probe_server_alive("localhost", 8000) is False

    # A port occupied by a non-HTTP service (SSH, a DB) answers with garbage,
    # which urllib surfaces as http.client.BadStatusLine — must not crash.
    import http.client

    def _bad_status(url, timeout=None):
        raise http.client.BadStatusLine("\x00nonsense")

    monkeypatch.setattr(urllib.request, "urlopen", _bad_status)
    assert connect.probe_server_alive("localhost", 8000) is False


def test_connect_reports_no_server_when_nothing_listening(monkeypatch):
    """#1999: connect must not announce Ready: for an address that refuses
    connections; it says there is no server and drops the Connect cheat-sheet
    that would otherwise wire a client to a dead endpoint."""
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    monkeypatch.setattr(connect, "probe_server_alive", lambda *a, **k: False)
    out = _run_connect()
    assert "Ready:" not in out
    assert "No rapid-mlx server on http://localhost:8000" in out
    assert "rapid-mlx serve <model>" in out
    # No cheat-sheet that points a client at the dead endpoint.
    assert "Connect:" not in out
    assert "--setup" not in out
    # The prospective addresses are still shown, just not as "Ready".
    assert "http://localhost:8000/v1" in out


def test_connect_json_is_valid_and_stable(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model="m1"),
    )
    out = _run_connect(json_=True)
    payload = json.loads(out)
    assert payload["ready"] == "http://localhost:8000"
    assert payload["openai"] == "http://localhost:8000/v1"
    assert payload["anthropic"] == "http://localhost:8000"
    assert payload["model"] == "m1"


def test_connect_openai_python_snippet(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model="m1"),
    )
    out = _run_connect(target="openai-python")
    assert "http://localhost:8000/v1" in out
    assert "OpenAI(" in out
    assert "m1" in out


def test_connect_claude_code_points_at_setup(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    out = _run_connect(target="claude-code")
    assert "rapid-mlx agents claude-code --setup" in out


def test_connect_continue_points_at_setup(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    out = _run_connect(target="continue")
    assert "rapid-mlx agents continue --setup" in out


def test_connect_unknown_target_exits(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc, redirect_stdout(buf):
        connect_command(
            argparse.Namespace(
                target="nope", json=False, host=None, port=None, model=None
            )
        )
    assert exc.value.code == 1
    # The helpful supported-target list is printed before exiting.
    assert "Supported: claude-code, continue, openai-python" in buf.getvalue()


# --- P1 fixes applied after #1872 revert ------------------------------------
# (Reviewer feedback on #1871: remote `--host/--port` setup commands must
# carry `--base-url`; IPv6 literals/scoped addresses must be bracket-wrapped;
# `connect --port` must reuse the validated `_port_arg`.)


def test_claude_remote_endpoint_passthrough(monkeypatch):
    """`--host/--port` must flow into the suggested `--base-url` for claude."""
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("mini.local", 9000, model=None),
    )
    out = _run_connect(target="claude-code")
    assert "http://mini.local:9000/v1" in out
    assert "rapid-mlx agents claude-code --setup" in out
    assert "--base-url http://mini.local:9000/v1" in out


def test_continue_remote_endpoint_passthrough(monkeypatch):
    """`--host/--port` must flow into the suggested `--base-url` for continue."""
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("mini.local", 9000, model=None),
    )
    out = _run_connect(target="continue")
    assert "http://mini.local:9000/v1" in out
    assert "--base-url http://mini.local:9000/v1" in out


def test_claude_localhost_keeps_default_base_url(monkeypatch):
    """Default localhost:8000 setup command still emits an explicit base url."""
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    out = _run_connect(target="claude-code")
    assert "--base-url http://localhost:8000/v1" in out


def test_ipv6_literal_bracketing():
    ep = _endpoints(host="::1", port=8000)
    assert ep.base_url == "http://[::1]:8000"
    assert ep.openai_url == "http://[::1]:8000/v1"
    assert ep.anthropic_url == "http://[::1]:8000"


def test_ipv6_scoped_address_bracketing():
    # Scoped (zone-id) address must be bracket-wrapped AND the zone-id `%`
    # percent-encoded as %25 per RFC 6874.
    ep = _endpoints(host="fe80::1%en0", port=8000)
    assert ep.base_url == "http://[fe80::1%25en0]:8000"
    assert ep.openai_url == "http://[fe80::1%25en0]:8000/v1"


def test_ipv6_banner_renders_bracketed(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("::1", 8000, model=None),
    )
    monkeypatch.setattr(connect, "probe_server_alive", lambda *a, **k: True)
    out = _run_connect()
    assert "Ready: http://[::1]:8000" in out
    assert "OpenAI:    http://[::1]:8000/v1" in out


def test_ipv6_json_renders_bracketed(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("::1", 8000, model="m1"),
    )
    payload = json.loads(_run_connect(json_=True))
    assert payload["ready"] == "http://[::1]:8000"
    assert payload["openai"] == "http://[::1]:8000/v1"
    assert payload["anthropic"] == "http://[::1]:8000"


def test_connect_invalid_port_rejected():
    """`connect --port` must reuse `_port_arg`, rejecting 0 / out-of-range."""
    from vllm_mlx.cli import build_parser

    parser = build_parser()
    for bad in ("0", "70000", "-1", "65536"):
        with pytest.raises(SystemExit):
            parser.parse_args(["connect", "--port", bad])


def test_connect_valid_port_accepted():
    """`connect --port` accepts a legitimate in-range value."""
    from vllm_mlx.cli import build_parser

    args = build_parser().parse_args(["connect", "--port", "9000", "--host", "x"])
    assert args.port == 9000
    assert args.host == "x"


# --- Review round 2: banner must carry endpoint; explicit --model must win ----
def test_render_banner_connect_carries_remote_endpoint():
    """Copying the Ready banner's setup command must target the real server."""
    out = connect.render_banner(connect.ServerEndpoints("mini.local", 9000, model="m1"))
    assert (
        "rapid-mlx agents claude-code --setup "
        "--base-url http://mini.local:9000/v1" in out
    )
    assert (
        "rapid-mlx agents continue --setup --base-url http://mini.local:9000/v1" in out
    )


def test_render_banner_connect_carries_ipv6_endpoint():
    out = connect.render_banner(connect.ServerEndpoints("::1", 8000, model="m1"))
    # IPv6 literals contain '['/']', which shlex.quote wraps in single quotes.
    assert "--base-url 'http://[::1]:8000/v1'" in out


def test_resolve_endpoints_preserves_explicit_model(monkeypatch):
    """An explicit --model must never be overwritten by a live probe."""
    from vllm_mlx.config import get_config

    probe_called = []

    def fake_probe(host, port):
        probe_called.append((host, port))
        return "server-says-other-model"

    monkeypatch.setattr(connect, "_probe_running_model", fake_probe)

    cfg = get_config()
    cfg.bind_host = None
    cfg.bind_port = None
    cfg.model_alias = None
    cfg.model_name = None

    ep = connect.resolve_endpoints(model="explicitly-requested")
    assert ep.model == "explicitly-requested"
    assert probe_called == []  # probe must not run when --model is explicit


def test_resolve_endpoints_probes_when_no_model(monkeypatch):
    """Live probe still fills in the model when none was supplied anywhere."""
    from vllm_mlx.config import get_config

    probe_called = []

    def fake_probe(host, port):
        probe_called.append((host, port))
        return "probed-model"

    monkeypatch.setattr(connect, "_probe_running_model", fake_probe)

    cfg = get_config()
    cfg.bind_host = None
    cfg.bind_port = None
    cfg.model_alias = None
    cfg.model_name = None

    ep = connect.resolve_endpoints()
    assert ep.model == "probed-model"
    assert probe_called == [("localhost", 8000)]


def test_ipv6_scoped_zone_id_config_and_banner_agree():
    """Banner Connect command and endpoint rows both use the %25 form."""
    out = connect.render_banner(connect.ServerEndpoints("fe80::1%en0", 8000, model="m"))
    assert "Ready: http://[fe80::1%25en0]:8000" in out
    assert "--base-url 'http://[fe80::1%25en0]:8000/v1'" in out


# --- Review round 3: dynamic URL in copied commands must be shell-quoted ----
def test_banner_connect_base_url_is_shell_quoted_for_special_hosts():
    """A crafted host must not smuggle a second shell command into the banner.

    ``render_banner`` builds the Connect: rows that a user copies and pastes
    into a shell. Every dynamic ``--base-url`` value must be ``shlex.quote``-
    ed so characters like ``;``, ``$()`` and spaces cannot be interpreted as
    shell syntax.
    """

    cases = [
        "mini.local; touch /tmp/pwned",
        "ha x",
        "$(rm -rf /)",
        "it's",
    ]
    for host in cases:
        out = connect.render_banner(connect.ServerEndpoints(host, 8000, model="m"))
        expected_url = connect.ServerEndpoints(host, 8000, model="m").openai_url
        expected = f"--base-url {shlex.quote(expected_url)}"
        assert expected in out, (
            f"host {host!r}: expected {expected!r} to be in banner output:\n{out}"
        )


def test_banner_ipv6_zone_id_quoted_and_not_double_encoded():
    """Scoped IPv6 with a raw zone-id is encoded %25 and then shell-quoted."""
    out = connect.render_banner(connect.ServerEndpoints("fe80::1%en0", 8000, model="m"))
    assert "--base-url 'http://[fe80::1%25en0]:8000/v1'" in out


def test_banner_already_encoded_zone_id_not_double_encoded():
    """An already-encoded %25 zone-id is left alone, not turned into %2525."""
    out = connect.render_banner(
        connect.ServerEndpoints("fe80::1%25en0", 8000, model="m")
    )
    assert "http://[fe80::1%25en0]:8000" in out
    assert "%2525" not in out


def test_banner_already_bracketed_host_not_double_bracketed():
    """A pre-bracketed [::1] host is normalized, not rendered as [[::1]]."""
    out = connect.render_banner(connect.ServerEndpoints("[::1]", 8000, model="m"))
    assert "Ready: http://[::1]:8000" in out
    assert "[[::1]]" not in out


def test_point_command_base_url_is_shell_quoted():
    """``rapid-mlx connect <agent>`` output must shell-quote --base-url.

    ``_print_point_command`` is what ``connect claude-code`` / ``connect
    continue`` render, and it carries the same copy-paste security contract as
    the banner.
    """
    from vllm_mlx.cli import _print_point_command

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_point_command(
            "Claude Code",
            "agents claude-code --setup",
            "http://mini.local; touch /tmp/pwned:8000/v1",
        )
    out = buf.getvalue()
    expected = shlex.quote("http://mini.local; touch /tmp/pwned:8000/v1")
    # The copy-paste command's --base-url is shell-quoted. (The human-facing
    # ``→  <url>`` echo line above it is informational, not a command.)
    assert f"--base-url {expected}" in out
