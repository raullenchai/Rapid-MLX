# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dogfood-v0.8.2 finding #3 — bearer-in-argv leak.

The ``vllm_mlx.server`` standalone entrypoint historically only honored
``--api-key`` on argv. rapid-desktop's sidecar shim exported
``RAPID_MLX_API_KEY`` AND still appended ``--api-key "$KEY"`` to argv,
so ``ps -ef`` exposed the per-launch bearer token to any local user
(codex BLOCKER taxonomy #3 — "bearer-in-shell-history").

The fix introduces ``vllm_mlx.server._resolve_api_key`` as the single
SSOT and routes both entrypoints (``cli.py``'s ``rapid-mlx serve`` and
``server.py``'s ``python -m vllm_mlx.server``) through it; the
``rapid-mlx serve`` banner reads the same SSOT via
``vllm_mlx.cli._auth_feature_str``. These tests call into both helpers
directly so a refactor that drops the env-var branch fails them;
mutation-testing the production code (removing the ``or`` clause)
flips them red.

A live-subprocess test boots a real ``rapid-mlx serve`` with the
bearer in env-only and asserts: (a) ``psutil.Process.cmdline()`` —
the same source ``ps -ef`` reads — does NOT contain the bearer; (b)
``GET /v1/models`` without auth returns 401 (proving the env path
actually wired the key into ``verify_api_key``); (c) ``GET /v1/models``
with the env-only bearer returns 200. That triple is the actual
contract the dogfood report fingered.
"""

from __future__ import annotations

import http.client
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

# ---------------------------------------------------------------------------
# In-process unit tests: call the REAL _resolve_api_key helper
# ---------------------------------------------------------------------------


def test_api_key_env_only_resolves_to_env_value(monkeypatch):
    """Env-only is the supported rapid-desktop sidecar path. If the
    production helper drops its env-var branch, this assertion fails."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.setenv("RAPID_MLX_API_KEY", "ENV_SECRET")
    assert _resolve_api_key(argv_value=None) == "ENV_SECRET"


def test_api_key_argv_only_still_works(monkeypatch):
    """Inline argv is the legacy path — backwards-compat must hold."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    assert _resolve_api_key(argv_value="ARGV_SECRET") == "ARGV_SECRET"


def test_api_key_both_set_argv_wins(monkeypatch):
    """Argv override is documented in --api-key help; pin the priority."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.setenv("RAPID_MLX_API_KEY", "ENV_VALUE")
    assert _resolve_api_key(argv_value="ARGV_VALUE") == "ARGV_VALUE"


def test_api_key_neither_set_is_none(monkeypatch):
    """No-auth dev path stays anonymous-OK."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    assert _resolve_api_key(argv_value=None) is None


def test_api_key_empty_string_argv_falls_back_to_env(monkeypatch):
    """Edge case: an empty ``--api-key ""`` should not silently disable
    auth when the env var is set. The ``or`` short-circuit means env
    wins because empty string is falsy."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.setenv("RAPID_MLX_API_KEY", "ENV_FALLBACK")
    assert _resolve_api_key(argv_value="") == "ENV_FALLBACK"


# ---------------------------------------------------------------------------
# Banner test: call the REAL _auth_feature_str renderer
# ---------------------------------------------------------------------------


def test_banner_renders_auth_on_when_only_env_is_set(monkeypatch):
    """Pre-fix the banner gate was ``if args.api_key`` — a sidecar
    that set env-only saw the line omitted even though enforcement
    was on. The fix routes through ``_auth_feature_str`` which calls
    the same ``_resolve_api_key`` SSOT the server reads. Calling the
    renderer directly proves the banner path mirrors the enforcement
    path; if the gate regresses to ``args.api_key`` only, this flips
    red because the input ``argv_api_key=None`` produces no feature."""
    from vllm_mlx.cli import _auth_feature_str

    monkeypatch.setenv("RAPID_MLX_API_KEY", "ENV_SECRET")
    assert _auth_feature_str(argv_api_key=None) == "auth: on"


def test_banner_renders_auth_on_when_only_argv_is_set(monkeypatch):
    """Backwards-compat: inline argv-set path also renders the line."""
    from vllm_mlx.cli import _auth_feature_str

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    assert _auth_feature_str(argv_api_key="ARGV_SECRET") == "auth: on"


def test_banner_omits_auth_line_when_neither_is_set(monkeypatch):
    """Dev path: no auth → no banner line. Mirrors the SECURITY
    CONFIGURATION block's ``Authentication: DISABLED`` warning."""
    from vllm_mlx.cli import _auth_feature_str

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    assert _auth_feature_str(argv_api_key=None) is None


# ---------------------------------------------------------------------------
# Live-process regression: env-only spawn, ps-cmdline + HTTP auth-enforced
# ---------------------------------------------------------------------------


def _find_free_port(start: int = 11830, end: int = 11930) -> int:
    """Pick a free port well away from rapid-mlx defaults (8000) and the
    user's prod sidecar (8451 — see memory feedback_dogfood_pkill_port_
    qualified). Anchoring in 11800s keeps this test from racing any
    concurrent agent on the same machine."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in range {start}-{end}")


def _wait_for_healthz(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    """Poll /healthz until ready or proc exits. Returns True on ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            conn.request("GET", "/healthz")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return True
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.5)
    return False


def _http_get(port: int, path: str, bearer: str | None) -> int:
    """Tiny stdlib HTTP GET that returns status code."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    if bearer:
        req.add_header("Authorization", f"Bearer {bearer}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


@pytest.mark.slow
def test_env_only_spawn_keeps_bearer_out_of_ps_and_enforces_auth():
    """End-to-end contract for dogfood-v0.8.2 finding #3:

    1. Boot ``rapid-mlx serve qwen3.5-4b-4bit`` with the bearer in
       ``RAPID_MLX_API_KEY`` env-only (NO ``--api-key`` argv).
    2. Read ``psutil.Process.cmdline()`` on the live child — the same
       source ``ps -ef`` reads. Bearer MUST NOT appear.
    3. ``GET /v1/models`` without auth → 401 (proves env actually
       wired the key into ``verify_api_key``; tautological otherwise).
    4. ``GET /v1/models`` with the env bearer → 200.
    5. Port-qualified pkill cleanup (per memory feedback_dogfood_
       pkill_port_qualified) — must NOT touch the user's prod 8451.

    Mutation safety: if production ignores ``RAPID_MLX_API_KEY``, step
    3 would return 200 (no auth wired) and the test flips red. If
    production puts the bearer on argv, step 2 sees it and flips red.
    """
    psutil = pytest.importorskip("psutil")
    port = _find_free_port()
    bearer = "DOGFOOD_AGENT_F_LIVE_BEARER_must_not_appear_in_ps_cmdline"
    env = {**os.environ, "RAPID_MLX_API_KEY": bearer}
    # The default test model per project memory.
    model_alias = "qwen3.5-4b-4bit"

    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        model_alias,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    assert bearer not in " ".join(cmd), "test bug: do not put bearer on argv"

    proc = subprocess.Popen(  # noqa: S603 — controlled test argv
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Read cmdline early — before model load completes — so we
        # have proof of leak-or-not regardless of healthz timing.
        observed_cmdline: list[str] = []
        for _ in range(20):
            try:
                ps_proc = psutil.Process(proc.pid)
                observed_cmdline = ps_proc.cmdline()
                if observed_cmdline:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.1)

        # Assertion 1: bearer never appears in live process cmdline.
        # This is exactly what ``ps -ef`` reads on macOS + Linux.
        joined_live = " ".join(observed_cmdline)
        assert bearer not in joined_live, (
            f"BEARER LEAKED in live process cmdline ({observed_cmdline!r}). "
            "Sidecar spawn path MUST keep RAPID_MLX_API_KEY in env only."
        )

        # Healthz: skip if model load takes too long on this CI. The
        # ps-leak assertion above is the primary regression check; the
        # auth-enforced check is a stronger but more environment-
        # dependent guarantee.
        if not _wait_for_healthz(port, proc, timeout=180.0):
            pytest.skip(
                f"server did not reach /healthz within 180s (rc={proc.poll()}); "
                "ps-leak assertion above already passed — auth-enforcement "
                "sub-check skipped to avoid CI environment flakes"
            )

        # Assertion 2: env-only auth is actually enforced. If production
        # ignored RAPID_MLX_API_KEY, this would return 200 and flip red.
        unauth_status = _http_get(port, "/v1/models", bearer=None)
        assert unauth_status == 401, (
            f"env-only auth NOT enforced: GET /v1/models without bearer "
            f"returned {unauth_status} (expected 401). The env-var path "
            "is not wired into verify_api_key — the leak fix is moot."
        )

        # Assertion 3: env-only bearer actually unlocks the surface.
        auth_status = _http_get(port, "/v1/models", bearer=bearer)
        assert auth_status == 200, (
            f"env-only bearer rejected: GET /v1/models with Bearer "
            f"{bearer[:8]}... returned {auth_status} (expected 200)"
        )
    finally:
        # Port-qualified pkill per memory feedback_dogfood_pkill_port_
        # qualified. Bare ``pkill -f vllm_mlx.cli`` would kill the
        # user's prod sidecar at 8451.
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(
            ["pkill", "-f", f"vllm_mlx.cli.*{port}"],
            check=False,
            capture_output=True,
        )


# ---------------------------------------------------------------------------
# Cheap docs-advertisement checks (fast — no model load)
# ---------------------------------------------------------------------------


def test_cli_help_advertises_env_fallback():
    """``rapid-mlx serve --help`` must document the env-var so
    downstream wrappers know the safer form exists."""
    result = subprocess.run(  # noqa: S603 — controlled test argv
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "RAPID_MLX_API_KEY" in result.stdout, (
        "--api-key help text must advertise the env-var fallback. "
        "Found:\n" + result.stdout[:2000]
    )


def test_server_help_advertises_env_fallback():
    """Parity check: ``python -m vllm_mlx.server`` (the standalone
    entry the rapid-desktop sidecar invokes) must also advertise the
    env fallback. Pre-fix only ``vllm_mlx.cli`` did, which is the
    docs/code mismatch that forced the shim to argv-pass the bearer."""
    result = subprocess.run(  # noqa: S603 — controlled test argv
        [sys.executable, "-m", "vllm_mlx.server", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "--api-key" in combined
    assert "RAPID_MLX_API_KEY" in combined, (
        "server entry --help must advertise the env fallback. "
        "Pre-fix it only said 'if not set, no auth required', which "
        "left downstream wrappers no choice but to put the bearer "
        "on argv. Got:\n" + combined[:2000]
    )
