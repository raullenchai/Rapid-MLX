# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified server ready/connect output SSOT.

``rapid_mlx.connect`` is the single source of truth for how a running server's
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
import sys
from contextlib import redirect_stdout

import pytest

from rapid_mlx import connect
from rapid_mlx.cli import connect_command


def _endpoints(host="localhost", port=8000, model="qwen3.6-35b-4bit"):
    return connect.ServerEndpoints(host, port, model=model)


def _run_connect(
    *,
    target=None,
    json_=False,
    host=None,
    port=None,
    model=None,
    base_url=None,
    allow_insecure_api_key=False,
):
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
                base_url=base_url,
                allow_insecure_api_key=allow_insecure_api_key,
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
    assert "rapid-mlx launch codex" in out
    # Python connects to this server, so its command carries the live endpoint.
    assert "rapid-mlx connect openai-python --base-url http://localhost:8000/v1" in out


def test_serve_next_uses_installed_python_for_sdk():
    out = connect.render_serve_next(_endpoints())
    assert connect.package_install_command("openai") in out


def test_package_install_command_supports_uv_and_pip_less_tools(monkeypatch):
    monkeypatch.setattr(connect.shutil, "which", lambda _name: "/usr/bin/uv")
    assert connect.package_install_command("openai") == (
        f"uv pip install --python {shlex.quote(sys.executable)} openai"
    )
    monkeypatch.setattr(connect.shutil, "which", lambda _name: None)
    assert connect.package_install_command("mlx-vlm==0.7.2", no_deps=True) == (
        f"{shlex.quote(sys.executable)} -m pip install --no-deps mlx-vlm==0.7.2"
    )


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

    responses = []

    def _ok(url, timeout=None):
        response = io.BytesIO(b"healthy")
        responses.append(response)
        return response

    def _raise(code):
        def _open(url, timeout=None):
            response = io.BytesIO(b"error")
            responses.append(response)
            raise urllib.error.HTTPError(url, code, "err", {}, response)

        return _open

    def _conn(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    # 2xx (any serving mode) → alive.
    monkeypatch.setattr(urllib.request, "urlopen", _ok)
    assert connect.probe_server_alive("localhost", 8000) is True
    assert responses[-1].closed

    # Auth-gated DDTree /healthz answers 401/403 → still a server there.
    for code in (401, 403):
        monkeypatch.setattr(urllib.request, "urlopen", _raise(code))
        assert connect.probe_server_alive("localhost", 8000) is True, code
        assert responses[-1].closed

    # 404 (unrelated server), 503 (up but draining, refusing new work), and
    # 500 (broken) are all "don't hand a client this endpoint".
    for code in (404, 500, 503):
        monkeypatch.setattr(urllib.request, "urlopen", _raise(code))
        assert connect.probe_server_alive("localhost", 8000) is False, code
        assert responses[-1].closed

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
    assert connect.package_install_command("openai") in out
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
                target="nope",
                json=False,
                host=None,
                port=None,
                model=None,
                base_url=None,
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
    from rapid_mlx.cli import build_parser

    parser = build_parser()
    for bad in ("0", "70000", "-1", "65536"):
        with pytest.raises(SystemExit):
            parser.parse_args(["connect", "--port", bad])


def test_connect_valid_port_accepted():
    """`connect --port` accepts a legitimate in-range value."""
    from rapid_mlx.cli import build_parser

    args = build_parser().parse_args(["connect", "--port", "9000", "--host", "x"])
    assert args.port == 9000
    assert args.host == "x"


# --- Review round 2: banner must carry endpoint; explicit --model must win ----
def test_render_banner_connect_carries_remote_endpoint():
    """The SDK command targets the live server; launch starts its own."""
    out = connect.render_banner(connect.ServerEndpoints("mini.local", 9000, model="m1"))
    assert "rapid-mlx launch codex" in out
    assert "rapid-mlx connect openai-python --base-url http://mini.local:9000/v1" in out


def test_render_banner_connect_carries_ipv6_endpoint():
    out = connect.render_banner(connect.ServerEndpoints("::1", 8000, model="m1"))
    # IPv6 literals contain '['/']', which shlex.quote wraps in single quotes.
    assert "--base-url 'http://[::1]:8000/v1'" in out


def test_resolve_endpoints_preserves_explicit_model(monkeypatch):
    """An explicit --model must never be overwritten by a live probe."""
    from rapid_mlx.config import get_config

    probe_called = []

    def fake_probe(host, port, **kwargs):
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
    from rapid_mlx.config import get_config

    probe_called = []

    def fake_probe(host, port, **kwargs):
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


def test_serve_next_uses_bound_endpoint_served_name_and_auth():
    ep = connect.ServerEndpoints("::1", 9123, model="my served model")
    out = connect.render_serve_next(ep, api_key="key with ' quote")
    assert out.count("Next:") == 1
    assert out.count("rapid-mlx launch ") == 7
    assert "curl -i -sS " in out
    assert "http://[::1]:9123/v1/chat/completions" in out
    assert "my served model" in out
    assert '-H "Authorization: Bearer $RAPID_MLX_API_KEY"' in out
    assert "rapid-mlx launch opencode" in out
    assert "Set RAPID_MLX_API_KEY to the server key first." in out
    assert "key with ' quote" not in out
    for agent in ("codex", "claude-code", "opencode", "pi", "hermes", "crush", "goose"):
        assert f"rapid-mlx launch {agent}" in out
    assert "OpenAI SDK base_url: http://[::1]:9123/v1; model: my served model" in out
    assert "from openai import OpenAI" in out
    assert 'environ["RAPID_MLX_API_KEY"]' in out


def test_serve_next_has_no_auth_header_without_key():
    out = connect.render_serve_next(connect.ServerEndpoints("localhost", 8765, "m"))
    assert "Authorization" not in out
    assert out.count("rapid-mlx launch ") == 7
    assert (
        connect.render_serve_next(connect.ServerEndpoints("", 0, "m", listen_fd=3))
        == ""
    )
    assert connect.render_serve_next(connect.ServerEndpoints("localhost", 8765)) == ""


def test_serve_next_lists_agents_when_no_setup_profile_is_available(monkeypatch):
    monkeypatch.setattr("rapid_mlx.agents.get_profile", lambda _name: None)
    out = connect.render_serve_next(connect.ServerEndpoints("localhost", 8765, "m"))
    assert "Coding agent: rapid-mlx launch codex" in out


# --- #2348: connect must accept/use the live instance base URL ---------------
# The serve banner advertises ``connect openai-python --base-url <url>`` so the
# pasted command points at the *actual* running server (its port + model),
# not the localhost:8000 default the standalone process would otherwise print.


def test_parse_base_url_derives_host_and_port():
    assert connect._parse_base_url("http://localhost:8123/v1") == ("localhost", 8123)
    assert connect._parse_base_url("http://127.0.0.1:9000") == ("127.0.0.1", 9000)
    # Trailing /v1 is the OpenAI-style form the banner prints; it's stripped.
    assert connect._parse_base_url("http://localhost:8123/v1")[1] == 8123


def test_parse_base_url_ipv6_bracketing():
    # Bracket-wrapped / scoped IPv6 pasted verbatim from the banner.
    assert connect._parse_base_url("http://[::1]:8000/v1") == ("::1", 8000)
    assert connect._parse_base_url("http://[fe80::1%25en0]:8123/v1") == (
        "fe80::1%en0",
        8123,
    )


def test_parse_base_url_does_not_decode_non_zone_host_escapes():
    # In an ordinary (non-IPv6) hostname every percent-escape is a genuine
    # octet and must survive untouched — including `%25` — else the generated
    # URL would be malformed (codex #2348-R2/R3).
    assert connect._parse_base_url("http://%2Fexample.com:8000/v1") == (
        "%2Fexample.com",
        8000,
    )
    assert connect._parse_base_url("http://%20host:8000/v1") == ("%20host", 8000)
    assert connect._parse_base_url("http://%25example.com:8000/v1") == (
        "%25example.com",
        8000,
    )
    # Only in a scoped IPv6 literal is `%25` a zone-id separator, decoded back
    # to `%`.
    assert connect._parse_base_url("http://[fe80::1%25en0]:8123/v1") == (
        "fe80::1%en0",
        8123,
    )


def test_parse_base_url_default_port_and_errors():
    # No port → the rapid-mlx default.
    assert connect._parse_base_url("http://server.local/v1") == ("server.local", 8000)
    # Unsupported scheme, malformed URL, no host, an out-of-range/zero
    # explicit port, and an unexpected path-prefix (a proxied endpoint the
    # SSOT does not model) are all rejected rather than retargeting the snippet.
    for bad in (
        "ftp://localhost:8123",
        "not-a-url",
        "http://:8123",
        "http://localhost:0/v1",
        "http://localhost:70000/v1",
        "http://server/proxy/v1",
        "http://localhost:8123/custom",
        "http://localhost:8123//",
        "http://localhost:8123/v1//",
    ):
        with pytest.raises(ValueError):
            connect._parse_base_url(bad)
    # The SSOT-renderable paths — empty, bare/trailing slash, `/v1` (+ one
    # trailing slash) — are all accepted losslessly.
    assert connect._parse_base_url("http://localhost:8123/v1") == ("localhost", 8123)
    assert connect._parse_base_url("http://localhost:8123/") == ("localhost", 8123)
    assert connect._parse_base_url("http://localhost:8123/v1/") == ("localhost", 8123)


def test_connect_base_url_partial_override_keeps_other_coordinate(monkeypatch):
    """`--host` and `--port` are independent overrides of the base URL: giving
    only `--host` must keep the base URL's port (not fall back to 8000)."""
    probed = []

    def fake_probe(host, port, **kwargs):
        probed.append((host, port))
        return "m"

    monkeypatch.setattr(connect, "_probe_running_model", fake_probe)
    out = _run_connect(
        target="openai-python",
        base_url="http://localhost:8123/v1",
        host="127.0.0.1",
    )
    assert "http://127.0.0.1:8123/v1" in out
    assert probed == [("127.0.0.1", 8123)]


def test_connect_openai_python_uses_base_url_for_snippet(monkeypatch):
    """`--base-url` (the banner's pasted command) must make the standalone
    snippet target the live host/port and probe THAT server for the model, not
    default to localhost:8000."""
    probed = []

    def fake_probe(host, port, **kwargs):
        probed.append((host, port))
        return "lfm2.5-1b-4bit"

    monkeypatch.setattr(connect, "_probe_running_model", fake_probe)
    out = _run_connect(target="openai-python", base_url="http://localhost:8123/v1")
    assert "http://localhost:8123/v1" in out
    assert "lfm2.5-1b-4bit" in out
    assert "http://localhost:8000" not in out
    # The probe hit the base-url's port, not the 8000 default.
    assert probed == [("localhost", 8123)]


def test_connect_base_url_yields_to_explicit_host_port(monkeypatch):
    """Explicit --host/--port flags win over --base-url (flags > base-url)."""
    probed = []

    def fake_probe(host, port, **kwargs):
        probed.append((host, port))
        return "m"

    monkeypatch.setattr(connect, "_probe_running_model", fake_probe)
    out = _run_connect(
        target="openai-python",
        base_url="http://localhost:8123/v1",
        host="10.0.0.5",
        port=9999,
    )
    assert "http://10.0.0.5:9999/v1" in out
    assert probed == [("10.0.0.5", 9999)]


def test_connect_invalid_base_url_exits_cleanly(monkeypatch):
    """A malformed --base-url must not crash; it prints a clear message."""
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc, redirect_stdout(buf):
        connect_command(
            argparse.Namespace(
                target="openai-python",
                json=False,
                host=None,
                port=None,
                model=None,
                base_url="ftp://localhost:8123",
            )
        )
    assert exc.value.code == 1
    assert "invalid --base-url" in buf.getvalue()


def test_point_command_base_url_is_shell_quoted():
    """``rapid-mlx connect <agent>`` output must shell-quote --base-url.

    ``_print_point_command`` is what ``connect claude-code`` / ``connect
    continue`` render, and it carries the same copy-paste security contract as
    the banner.
    """
    from rapid_mlx.cli import _print_point_command

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


@pytest.fixture
def auth_connect_server(monkeypatch):
    """Exercise real HTTP discovery without loading a model or using the GPU."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from rapid_mlx.config import get_config

    cfg = get_config()
    for name in ("bind_host", "bind_port", "model_alias", "model_name"):
        monkeypatch.setattr(cfg, name, None)
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    state = {"status": 401, "requests": [], "models": [{"id": "loaded-model"}]}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append((self.path, self.headers.get("Authorization")))
            if self.path == "/v1/models" and state.get("redirect"):
                self.send_response(302)
                self.send_header("Location", "/redirected-models")
            elif (
                self.path in ("/healthz", "/redirected-models")
                or self.headers.get("Authorization") == "Bearer test-key"
            ):
                self.send_response(200)
            else:
                self.send_response(state["status"])
            self.end_headers()
            self.wfile.write(json.dumps({"data": state["models"]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_connect_authenticated_discovery(auth_connect_server, monkeypatch):
    port, state = auth_connect_server
    monkeypatch.setenv("RAPID_MLX_API_KEY", "test-key")
    payload = json.loads(_run_connect(host="127.0.0.1", port=port, json_=True))
    assert payload["model"] == "loaded-model"
    assert ("/v1/models", "Bearer test-key") in state["requests"]
    assert "test-key" not in json.dumps(payload)
    banner = _run_connect(host="127.0.0.1", port=port)
    assert "loaded-model" in banner
    assert "RAPID_MLX_API_KEY" in banner
    assert "test-key" not in banner


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_connect_rejected_auth_is_explicit(
    auth_connect_server, monkeypatch, status, key
):
    port, state = auth_connect_server
    state["status"] = status
    if key:
        monkeypatch.setenv("RAPID_MLX_API_KEY", key)
    payload = json.loads(_run_connect(host="127.0.0.1", port=port, json_=True))
    assert payload["model"] is None
    assert payload["auth_required"] is True
    banner = _run_connect(host="127.0.0.1", port=port)
    assert "(authentication required)" in banner
    assert "(not yet loaded)" not in banner
    assert "export RAPID_MLX_API_KEY=" in banner
    assert "wrong-key" not in banner


@pytest.mark.parametrize("key", [None, "test-key"])
def test_connect_python_snippet_reads_environment(
    auth_connect_server, monkeypatch, key
):
    port, _ = auth_connect_server
    if key:
        monkeypatch.setenv("RAPID_MLX_API_KEY", key)
    out = _run_connect(target="python", host="127.0.0.1", port=port, model="explicit")
    assert "import os" in out
    assert 'os.environ.get("RAPID_MLX_API_KEY") or "sk-noop"' in out
    assert "RAPID_MLX_API_KEY" in out
    assert "test-key" not in out


@pytest.mark.parametrize(
    "status,models,expected",
    [
        (200, [{"id": "public-model"}], "public-model"),
        (200, [], None),
        (500, [{"id": "not-available"}], None),
    ],
)
def test_connect_non_auth_responses(auth_connect_server, status, models, expected):
    port, state = auth_connect_server
    state.update(status=status, models=models)
    payload = json.loads(_run_connect(host="127.0.0.1", port=port, json_=True))
    assert payload["model"] == expected
    assert "auth_required" not in payload
    assert state["requests"] == [("/v1/models", None)]


def test_connect_does_not_forward_key_on_redirect(auth_connect_server, monkeypatch):
    port, state = auth_connect_server
    state["redirect"] = True
    monkeypatch.setenv("RAPID_MLX_API_KEY", "test-key")
    assert connect._probe_running_model("127.0.0.1", port) == "loaded-model"
    assert state["requests"] == [
        ("/v1/models", "Bearer test-key"),
        ("/redirected-models", None),
    ]


@pytest.mark.parametrize("status", [401, 403, 500])
def test_connect_probe_closes_rejected_response(monkeypatch, status):
    import urllib.error
    import urllib.request

    body = io.BytesIO(b"rejected")
    error = urllib.error.HTTPError(
        "http://localhost/v1/models", status, "error", {}, body
    )

    def reject(*args, **kwargs):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", reject)
    if status in (401, 403):
        with pytest.raises(PermissionError):
            connect._probe_running_model("localhost", 12345)
    else:
        assert connect._probe_running_model("localhost", 12345) is None
    assert body.closed


@pytest.mark.parametrize(
    "url,opt_in,expected_auth",
    [
        ("http://localhost:12345/v1", False, True),
        ("http://127.0.0.2:12345/v1", False, True),
        ("http://[::1]:12345/v1", False, True),
        ("https://remote.example/v1", False, True),
        ("https://remote.example:9443/v1", False, True),
        ("http://remote.example:12345/v1", False, False),
        ("http://192.0.2.1:12345/v1", False, False),
        ("http://localhost.example:12345/v1", False, False),
        ("http://remote.example:12345/v1", True, True),
    ],
)
@pytest.mark.parametrize("key", [None, "test-key"])
def test_connect_transport_policy(
    auth_connect_server, monkeypatch, capsys, url, opt_in, expected_auth, key
):
    import urllib.request

    if key:
        monkeypatch.setenv("RAPID_MLX_API_KEY", key)
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b'{"data": [{"id": "loaded-model"}]}')

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    output = _run_connect(base_url=url, json_=True, allow_insecure_api_key=opt_in)
    payload = json.loads(output)
    expected_base = url.removesuffix("/v1")
    if expected_base == "https://remote.example":
        expected_base += ":443"
    assert payload["ready"] == expected_base
    assert payload["model"] == "loaded-model"
    assert requests[0].full_url == expected_base + "/v1/models"
    assert requests[0].get_header("Authorization") == (
        "Bearer test-key" if expected_auth and key else None
    )
    stderr = capsys.readouterr().err
    assert ("unencrypted" in stderr) == (bool(key) and not expected_auth)
    if key and not expected_auth:
        assert "--allow-insecure-api-key" in stderr
    snippet = _run_connect(
        base_url=url, target="python", model="explicit", allow_insecure_api_key=opt_in
    )
    assert ('api_key=os.environ.get("RAPID_MLX_API_KEY")' in snippet) == expected_auth
    assert ("export RAPID_MLX_API_KEY=" in snippet) == expected_auth
    if not expected_auth:
        assert "rerun with an HTTPS URL or --allow-insecure-api-key" in snippet
    assert "test-key" not in snippet + output + stderr
    banner = _run_connect(base_url=url, allow_insecure_api_key=opt_in)
    assert f"Ready: {expected_base}" in banner
    assert requests[-1] == expected_base + "/healthz"


def test_connect_insecure_key_flag_is_explicit():
    from rapid_mlx.cli_parser import build_parser

    parser = build_parser()
    assert not parser.parse_args(["connect"]).allow_insecure_api_key
    assert parser.parse_args(
        ["connect", "--allow-insecure-api-key"]
    ).allow_insecure_api_key
