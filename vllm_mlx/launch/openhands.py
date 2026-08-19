# SPDX-License-Identifier: Apache-2.0
"""OpenHands (agent-canvas desktop) launch adapter.

Unlike every other adapter in this package, OpenHands' provider settings
are NOT patchable as a file. They live in ``~/.openhands/settings.json``,
but the ``api_key`` field there is Fernet-encrypted with the key in
``~/.openhands/agent-canvas/secret-key.txt``, so a plaintext write lands
a credential the app cannot decrypt. The supported route is the running
agent-server's REST API, which performs the encryption for us:

    PATCH /api/settings
    X-Session-API-Key: <~/.openhands/agent-canvas/api-key.txt>
    {"agent_settings_diff": {"llm": {...}}}

Verified against agent-canvas 1.14.0 / openhands-sdk 1.42.1.

Two consequences the other adapters do not have:

* **OpenHands must be running.** There is no offline write. :func:`detect`
  therefore reports "installed" from the on-disk data directory, and the
  write path fails with an actionable message when nothing answers.
* **The ingress port is the front door.** agent-canvas runs the
  agent-server on 18000 and a Node ingress on 8000 that proxies ``/api``
  to it. We talk to the ingress because its port is the one the user's
  browser is already on; ``OPENHANDS_URL`` overrides it for a
  non-default layout.

The model name is written with LiteLLM's ``openai/`` provider prefix.
OpenHands routes every completion through LiteLLM, which cannot resolve a
bare non-catalog name like ``qwen3.5-4b-4bit`` and errors before any
request reaches us.

Only the three keys we own (``model``, ``base_url``, ``api_key``) are
sent; the diff shape leaves the rest of the user's LLM block — retries,
reasoning effort, condenser and tool config — untouched.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

# Ports probed, in order, when ``OPENHANDS_URL`` is unset. agent-canvas
# defaults its ingress to 8000 but users move it (``--port``), most often
# because rapid-mlx is already there — 8000 is *our* default serve port
# too. Probing a small range keeps the copy-paste command portable
# instead of making the user discover and pass the port themselves.
_PROBE_PORTS = (8000, 8010, 3000, 8001, 8002, 8003, 8080)

# Session credential the agent-server requires on every ``/api`` call.
# Written by agent-canvas at first launch, 0600.
_API_KEY_FILE = "api-key.txt"


def _data_dir() -> Path:
    """Resolve OpenHands' data root (``OPENHANDS_DIR`` overrides)."""
    override = os.environ.get("OPENHANDS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".openhands"


def _is_openhands(url: str, session_key: str) -> bool:
    """Return True when ``url`` answers ``/api/settings`` as OpenHands.

    The probe authenticates rather than merely checking that something
    listens, which is what makes it safe to sweep a port range: an
    rapid-mlx server on the same port answers 404, and any other service
    answers 404/401/garbage. Only a real agent-server holding this
    session key returns 200.
    """
    request = urllib.request.Request(
        f"{url}/api/settings",
        method="GET",
        headers={"X-Session-API-Key": session_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200
    except (urllib.error.HTTPError, OSError):
        return False


def _base_url(session_key: str | None = None) -> str:
    """Resolve the ingress base URL, without a trailing slash.

    ``OPENHANDS_URL`` wins outright and is never probed — an explicit
    address must not be silently second-guessed. Otherwise we sweep
    :data:`_PROBE_PORTS` for one that authenticates, falling back to the
    first entry so the "could not reach" error names a plausible address.
    """
    override = os.environ.get("OPENHANDS_URL")
    if override:
        return override.rstrip("/")
    default = f"http://127.0.0.1:{_PROBE_PORTS[0]}"
    if session_key is None:
        return default
    for port in _PROBE_PORTS:
        candidate = f"http://127.0.0.1:{port}"
        if _is_openhands(candidate, session_key):
            return candidate
    return default


def _session_key() -> str | None:
    """Read the agent-server session key, or None when it isn't there."""
    try:
        key = (_data_dir() / "agent-canvas" / _API_KEY_FILE).read_text().strip()
    except OSError:
        return None
    return key or None


def detect() -> bool:
    """Return True when OpenHands appears installed on this machine.

    Keyed on the data directory rather than the running server: a user
    who has OpenHands installed but closed should be told to start it,
    not told it isn't installed.
    """
    return (_data_dir() / "agent-canvas").is_dir()


def current_config_path() -> Path | None:
    """Return the settings file our PATCH ultimately persists to.

    We never write this file ourselves — the agent-server does, after
    encrypting the key — but it is the honest answer to "where did my
    configuration go", and the launch CLI prints it.
    """
    if not detect():
        return None
    return _data_dir() / "settings.json"


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Point OpenHands at the local rapid-mlx OpenAI-compatible server.

    ``server_url`` is the origin (``http://127.0.0.1:8001``); OpenHands
    wants the OpenAI base URL, so ``/v1`` is appended when absent.

    The ``config_path`` arg is a test/dry-run hook naming the settings
    file; production callers let :func:`current_config_path` resolve it.
    """
    settings_path = config_path or current_config_path()
    if settings_path is None:
        raise FileNotFoundError(
            "OpenHands does not appear to be installed (no ~/.openhands/"
            "agent-canvas). Install it with `npx @openhands/agent-canvas`, "
            "open it once, and try again."
        )

    session_key = _session_key()
    if session_key is None:
        raise FileNotFoundError(
            f"OpenHands session key not found at {_data_dir()}/agent-canvas/"
            f"{_API_KEY_FILE}. Start OpenHands once so it generates one."
        )

    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    # Resolved once: the sweep costs a round trip per candidate port and
    # both the request and the error messages below need the same answer.
    openhands_url = _base_url(session_key)

    payload = json.dumps(
        {
            "agent_settings_diff": {
                "llm": {
                    # LiteLLM cannot route a bare non-catalog model name.
                    "model": f"openai/{model}",
                    "base_url": base_url,
                    "api_key": api_key,
                }
            }
        }
    ).encode()

    request = urllib.request.Request(
        f"{openhands_url}/api/settings",
        data=payload,
        method="PATCH",
        headers={
            "Content-Type": "application/json",
            "X-Session-API-Key": session_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10):
            pass
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        # Reaching a 404 means the sweep found nothing and fell back, or
        # OPENHANDS_URL points somewhere else. Either way the fix is a
        # port, not something inside OpenHands — and :8000 is rapid-mlx's
        # own default serve port, so this is a realistic mix-up.
        if exc.code == 404:
            raise RuntimeError(
                f"{openhands_url} answered, but not as OpenHands (HTTP 404 on "
                f"/api/settings). Something else is on that port — it is also "
                f"rapid-mlx's own default serve port. Start OpenHands, or set "
                f"OPENHANDS_URL to the port its ingress listens on."
            ) from exc
        raise RuntimeError(
            f"OpenHands rejected the settings update (HTTP {exc.code}): {detail}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not reach OpenHands at {openhands_url}. Its settings are only "
            "writable through the running app (the stored API key is encrypted "
            "with a key we don't hold), so start OpenHands and re-run. Set "
            "OPENHANDS_URL if it listens elsewhere."
        ) from exc

    return settings_path
