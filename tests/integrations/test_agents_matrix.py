# SPDX-License-Identifier: Apache-2.0
"""Tier-1 agents × 4 families integration matrix (0.10.2 PR-2 pilot).

Eleven Tier-1 agents — the pilot finalized the top 10 via pre-flight
verification of five commercial CLIs (cursor / droid / kimi-code /
qodercli / copilot) against the "can be pointed at a custom OpenAI
base_url" bar; kilo-code retained from #1030 pending an operator scope
call on top-10-vs-top-11:

Existing wire cells (from #1030 scaffold):

* codex-cli (/v1/responses)
* claude-code (/v1/messages via Anthropic SDK — covered by
  ``test_anthropic_sdk.py``; the matrix cell here proves the SDK still
  drives an end-to-end tool loop on the running server)
* opencode (/v1/chat/completions)
* qwen-code (/v1/chat/completions, promoted from fallback pool since
  Cursor CLI's agent path is locked to Cursor's backend)
* openhands (/v1/chat/completions)
* hermes-agent (covered end-to-end by ``test_hermes.py`` — this file
  smokes the wire in a lightweight cell; promoted from fallback pool
  since Alibaba Qoder's CLI has no first-party OpenAI-compat base_url
  hook, only proxy wrappers)
* aider (real bash-CLI drive via ``test_aider.sh`` — matrix cell shells
  out to the harness, asserts add.py rewritten to ``return a + b``)
* kilo-code (/v1/chat/completions)

New wire cells (0.10.2 PR-2 pilot):

* copilot (GitHub Copilot CLI — /v1/chat/completions via BYOK env vars
  ``COPILOT_PROVIDER_BASE_URL`` + ``COPILOT_PROVIDER_API_KEY``, docs
  https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models)
* droid (Factory AI Droid CLI — /v1/chat/completions via
  ``~/.factory/settings.json`` ``customModels`` array with
  ``provider: generic-chat-completion-api``)
* kimi-code (Moonshot Kimi Code CLI — /v1/chat/completions via
  ``~/.kimi/config.toml`` provider block with ``type = "openai"``
  and ``base_url``)

Each cell is a **smoke** — connect via the agent's wire, run a single
tool-calling exchange, verify the response envelope is well-formed and
no channel-marker leak (``<think>``, ``<|channel|>analysis``, etc.).
Deeper flows (multi-turn, sustained fuzz) live in the dedicated
integration files (``test_hermes.py``, ``test_anthropic_sdk.py``).

Rationale for lightweight cells + heavy dedicated files:
1. Matrix runs fast enough for per-PR (< 60 s for 24 cells).
2. Regressions in a single agent's wire still surface (dedicated file
   would give a false-green if the CI skipped it).
3. New agents added to the Tier-1 list get a smoke automatically; the
   deep test file follows at whatever cadence the agent's stability
   deserves.

**Cells not exercised in this file — see the "🔲 pending" cells of the
README matrix:** claude-code needs an installed ``anthropic`` package (in
optional extras, not core). Aider previously deferred to ``test_aider.sh``
for its edit-and-write flow; as of 0.10.3-window the matrix cell now
shells out to that harness directly (see ``TestAider`` below) so the four
Tier-1 family cells run for real instead of xfail'ing structurally.
OpenHands — previously a strict-xfail placeholder pending "Docker E2E
harness required for OpenHands' text-action format" — has landed the
same shape: ``TestOpenHands`` now shells out to ``test_openhands.sh``
which drives the pinned OpenHands 0.9.0 app + runtime images against
``rapid-mlx serve`` and asserts add.py corrected. Correctness gate is a **strict AST whitelist on ``add.py``'s return
expression** (parses the file, requires the module top level to be a
single ``def add`` optionally preceded by a docstring, the ``def add``
itself to have no decorators / defaults / annotations, and the return
to be one of ``a + b`` / ``b + a`` / ``sum([a, b])`` / ``sum((a, b))``
after an optional function-level docstring, with the signature pinned
to ``(a, b)``). Non-executing, so the LLM's output never touches the
host process; strictly stronger than a runtime pair-sweep (no
``a - b + k``, ``return CONST``, or ``if …: return 5`` cheat can
satisfy the AST shape). Docker-daemon skip guard so non-Docker CI
stays green.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.integrations.conftest import (
    FamilyAlias,
    assert_content_nonempty,
    assert_no_analysis_channel_leak,
    assert_no_think_tag_leak,
    assert_tool_call_shape,
    strict_skip_or_fail,
)

# --------------------------------------------------------------------------- #
# Shared per-cell tool-call payload
# --------------------------------------------------------------------------- #


_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_TOOL_PROMPT = "What's the weather in Tokyo? Use the get_weather tool."


def _openai_client_and_errors(base_url: str):
    """Lazy openai import — the pkg is optional in the base venv.

    Codex #1030 round-5 findings 1-3: return the exception classes AND
    the client instance from a single import site so callers can never
    accidentally trigger a raw ImportError before the skip guard fires.
    Every OpenAI-wire cell in this module goes through this helper.
    """
    try:
        from openai import (
            APIStatusError,
            BadRequestError,
            NotFoundError,
            OpenAI,
        )
    except ImportError:
        pytest.skip("openai package not installed — agent matrix skipped")
    client = OpenAI(base_url=base_url, api_key="not-needed")
    return client, (BadRequestError, NotFoundError, APIStatusError)


def _openai_client(base_url: str):
    """Back-compat single-value client accessor (thin wrapper).

    Preserved for cells that only want the client and use a bare
    ``except Exception`` handler; the tool-call helper below uses the
    tuple-returning ``_openai_client_and_errors`` for typed catching.
    """
    client, _errs = _openai_client_and_errors(base_url)
    return client


def _run_openai_tool_smoke(
    rapid_mlx_server: dict[str, Any],
    family_alias: FamilyAlias,
    *,
    agent_label: str,
) -> None:
    """Run one tool-call cell against ``/v1/chat/completions``.

    Shared by every Tier-1 agent that speaks the OpenAI wire (opencode,
    qwen-code, openhands, kilo-code, and — degraded — hermes).
    """
    client, wire_errors = _openai_client_and_errors(rapid_mlx_server["base_url"])
    model_id = rapid_mlx_server["model_id"]

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": _TOOL_PROMPT}],
            tools=_TOOL_SCHEMA,
            temperature=0.0,
            max_tokens=384,
        )
    except wire_errors as exc:
        # Codex #1030 finding 3: server rejecting a tool-call request is a
        # regression, not a degraded-cell condition. In strict mode we fail
        # the cell so CI can catch the wire break; non-strict skips so a
        # local dev on a still-booting server doesn't get spurious reds.
        strict_skip_or_fail(
            f"{agent_label}/{family_alias.family}: server rejected tool request "
            f"on {model_id!r}: {exc}"
        )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    content = msg.content or ""

    if not tool_calls:
        # A model may answer inline for small aliases — still assert wire
        # cleanliness so we catch channel leaks even without a tool call.
        assert_content_nonempty(content, ctx=f"{agent_label}/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)
        # Codex #1030 round-2 finding 1: an empty tool_calls slot on a Tier-1
        # agent cell is a real regression signal in CI (server may have
        # dropped tool-call plumbing) — strict mode fails; local dev on a
        # small alias still skips.
        strict_skip_or_fail(
            f"{agent_label}/{family_alias.family}: {model_id} returned no "
            f"tool_calls (content={content[:100]!r}); strict CI treats this "
            f"as a wire regression on the tool-call path."
        )
        return  # unreachable in strict; explicit for clarity in local runs

    tc = tool_calls[0]
    # openai SDK returns a Pydantic model — normalize to dict for the helper.
    tc_dict = {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
    assert_tool_call_shape(tc_dict)
    assert tc.function.name == "get_weather", tc.function.name
    args = json.loads(tc.function.arguments)
    assert "city" in args, args
    assert "tokyo" in args["city"].lower(), args


# --------------------------------------------------------------------------- #
# Cells — one per Tier-1 agent
# --------------------------------------------------------------------------- #


class TestCodexCLI:
    """Codex CLI /v1/responses — stateless shim (see codex.yaml)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        import httpx

        base_url = rapid_mlx_server["base_url"]
        model_id = rapid_mlx_server["model_id"]

        # Minimal /v1/responses envelope that Codex CLI would send.
        payload = {
            "model": model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply with just SHIPPED."}
                    ],
                }
            ],
            "stream": False,
            "max_output_tokens": 64,
        }
        try:
            r = httpx.post(f"{base_url}/responses", json=payload, timeout=90)
        except httpx.HTTPError as exc:
            # Codex #1030 round-3 finding 1: a transport failure to a wired
            # server AFTER the session-scope /v1/models probe succeeded is a
            # regression signal in strict CI (the server tore down mid-test
            # or the codex-shape route was pulled). Strict fails; local dev
            # on a flaky loopback still skips.
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: transport error hitting "
                f"/v1/responses after session probe was healthy: {exc!r}"
            )
        if r.status_code in (404, 405):
            # Codex #1030 round-2 finding 3: RAPID_MLX_MATRIX_STRICT=1 is meant
            # to gate exactly this — the /v1/responses route MUST be wired for
            # Codex CLI Tier-1 support. Strict CI fails; local dev on an older
            # server without the shim skips.
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: /v1/responses returned "
                f"{r.status_code} — route not wired on this server."
            )
        if r.status_code >= 400:
            # Codex #1030 finding 2: a 4xx / 5xx from a wired route IS a
            # regression. Strict CI must fail so the codex-shape SSE break
            # can't hide behind a skipped cell.
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: server returned {r.status_code} "
                f"({r.text[:200]!r})"
            )
        data = r.json()
        # Codex #1030 round-6 finding 4: walk the /v1/responses envelope,
        # extract the first output_text block, assert it is non-empty and
        # channel-clean. A blank output_text or a leaked ``<|channel|>``
        # marker now fails instead of passing on envelope-only truthiness.
        outputs = data.get("output") or []
        assert outputs, f"empty output envelope: {data}"
        text = ""
        for output_msg in outputs:
            for block in output_msg.get("content", []) or []:
                if block.get("type") in ("output_text", "text"):
                    text = block.get("text", "") or ""
                    if text:
                        break
            if text:
                break
        assert_content_nonempty(text, ctx=f"codex-cli/{family_alias.family}")
        assert_no_think_tag_leak(text)
        assert_no_analysis_channel_leak(text)


class TestClaudeCode:
    """Claude Code /v1/messages — Anthropic SDK route (see test_anthropic_sdk.py)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from anthropic import (
                Anthropic,
                APIStatusError,
                BadRequestError,
                NotFoundError,
            )
        except ImportError:
            pytest.skip("anthropic SDK not installed — cell deferred")

        base_no_v1 = rapid_mlx_server["base_url"].rstrip("/").removesuffix("/v1")
        client = Anthropic(base_url=base_no_v1, api_key="not-needed")

        try:
            resp = client.messages.create(
                model=rapid_mlx_server["model_id"],
                max_tokens=128,
                messages=[{"role": "user", "content": "Reply with just SHIPPED."}],
            )
        except NotFoundError:
            # Codex #1030 round-2 finding 4: strict CI must fail when the
            # Anthropic /v1/messages route is missing — that's exactly the
            # regression the Claude Code Tier-1 matrix cell is here to catch.
            # Local dev on a mock or older server still skips.
            strict_skip_or_fail(
                f"claude-code/{family_alias.family}: /v1/messages returned 404 "
                f"on {rapid_mlx_server['base_url']} — Anthropic route not wired."
            )
        except (BadRequestError, APIStatusError) as exc:
            # Codex #1030 finding 2: a wired-but-broken /v1/messages IS a
            # regression. Strict CI fails; local dev skips.
            strict_skip_or_fail(
                f"claude-code/{family_alias.family}: server rejected request: {exc}"
            )

        # Walk content blocks and find the first text — reasoning models emit a
        # thinking block first (see test_anthropic_sdk.py _first_text).
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        assert_content_nonempty(text, ctx=f"claude-code/{family_alias.family}")
        assert_no_think_tag_leak(text)
        assert_no_analysis_channel_leak(text)


class TestOpenCode:
    """OpenCode /v1/chat/completions with tool call."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="opencode")


class TestQwenCode:
    """Qwen Code /v1/chat/completions with tool call.

    Qwen Code speaks plain OpenAI wire via ``openaiCompatible.baseUrl``
    (see ``qwen-code.yaml``) — the smoke here is the same wire as the
    agent itself would drive.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="qwen-code")


class TestOpenHands:
    """OpenHands — real Docker E2E harness (``test_openhands.sh``).

    OpenHands' native wire is a text-action format, NOT OpenAI function
    calling — the CodeActAgent parses the model's plaintext reply,
    extracts actions (``IPythonRunCellAction``, ``CmdRunAction``, edit
    blocks), and applies file edits through its sandbox runtime.
    So the correctness signal is **did OpenHands actually rewrite the
    file the way we asked**, not **did tool_calls fire**.

    The bash harness at ``tests/integrations/test_openhands.sh`` takes
    ``--model <alias>`` + ``--base-url <url>``, seeds a scratch
    ``add.py`` with ``return a - b  # BUG``, runs
    ``docker run ... python -m openhands.core.main -t "Fix the bug ..."``
    with pinned OpenHands 0.9.0 app + runtime images, then asserts
    (a) the container exited 0 and (b) the file now contains
    ``return a + b``. Full family-by-family empirical verification is
    documented in the PR that un-xfailed this cell.

    Skipped (not failed) when Docker isn't available so non-Docker CI
    stays green; the real coverage is Apple-silicon local + Docker CI
    only.
    """

    _HARNESS_TIMEOUT_SECONDS = 600  # generous — OpenHands boots a sandbox

    @staticmethod
    def _docker_available() -> bool:
        """Return True iff the Docker daemon is reachable.

        ``docker info`` is the canonical daemon-alive probe (``docker
        version`` returns client-side even when the socket is dead). We
        cap the probe at 5 s so a hung / half-crashed daemon doesn't
        turn every OpenHands cell into a 30-second hang.
        """
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            return False
        try:
            result = subprocess.run(
                [docker_bin, "info"],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return result.returncode == 0

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        harness = Path(__file__).parent / "test_openhands.sh"
        assert harness.exists(), f"openhands harness missing: {harness}"

        # Docker daemon is a hard prerequisite for OpenHands' sandbox
        # runtime (docker-in-docker sock passthrough). Skip cleanly on
        # environments without Docker so non-Docker CI still passes;
        # Apple-silicon local + Docker CI cover the real coverage.
        if not self._docker_available():
            pytest.skip(
                "Docker daemon required for OpenHands E2E harness "
                "(start Docker Desktop / dockerd — the harness runs "
                "the pinned OpenHands app + sandbox runtime images)."
            )

        # Pass the FULL parsed base_url to the harness — the harness
        # itself rewrites the host to ``host.docker.internal`` for the
        # inside-container view, so a fixture pointed at a non-localhost
        # host (CI shard on a remote-serve node) still tests the right
        # server. ``--base-url`` is authoritative; ``--port`` is only
        # kept for standalone local invocations.
        base_url = rapid_mlx_server["base_url"]
        model_id = rapid_mlx_server["model_id"]

        env = os.environ.copy()

        try:
            result = subprocess.run(
                [
                    "bash",
                    str(harness),
                    "--model",
                    model_id,
                    "--base-url",
                    base_url,
                    "--timeout",
                    str(self._HARNESS_TIMEOUT_SECONDS),
                ],
                capture_output=True,
                text=True,
                # +60 s over the harness's internal ``timeout(1)`` so
                # cleanup + docker rm -f can run without the pytest
                # wall-timeout tripping first.
                timeout=self._HARNESS_TIMEOUT_SECONDS + 60,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"openhands/{family_alias.family}: harness wall-timeout "
                f"({self._HARNESS_TIMEOUT_SECONDS + 60}s) exceeded. "
                # Codex #1048 round-8 nit: with ``text=True``,
                # ``TimeoutExpired.stdout``/``.stderr`` are ``str | None``,
                # so a bytes-literal ``or`` fallback would mix ``str`` and
                # ``bytes`` in the tail slice. Use ``""`` and take the
                # tail off a homogeneous ``str``.
                f"stdout(tail)={(exc.stdout or '')[-800:]!r} "
                f"stderr(tail)={(exc.stderr or '')[-800:]!r}"
            )

        # Assert exit 0 with the tail of stdout+stderr for diagnostics
        # — the harness itself already prints BEFORE/AFTER add.py and
        # the last 60 lines of the openhands log, so this is enough to
        # root-cause any empirical failure without re-running.
        if result.returncode != 0:
            tail_out = result.stdout[-2000:] if result.stdout else ""
            tail_err = result.stderr[-1000:] if result.stderr else ""
            pytest.fail(
                f"openhands/{family_alias.family}: harness exited "
                f"{result.returncode} on model {model_id!r}\n"
                f"--- stdout tail ---\n{tail_out}\n"
                f"--- stderr tail ---\n{tail_err}"
            )


class TestHermesAgent:
    """Hermes Agent — deep flow in ``test_hermes.py``; matrix cell smokes wire."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(
            rapid_mlx_server, family_alias, agent_label="hermes-agent"
        )


class TestAider:
    """Aider — real bash-CLI harness (``test_aider.sh``).

    Aider does not speak OpenAI tool_calls — it sends the file + user
    instruction as plain messages, expects the LLM to emit
    ``SEARCH ... REPLACE ...`` blocks, and applies those edits locally.
    So the correctness signal is **did aider actually rewrite the file
    the way we asked**, not **did tool_calls fire**.

    The bash harness at ``tests/integrations/test_aider.sh`` takes
    ``--model <alias>`` + ``--base-url <url>``, seeds a scratch ``add.py``
    with ``return a - b  # BUG``, runs aider one-shot with
    ``--message "Fix the bug ... this function should add, not subtract"``,
    then asserts (a) aider exited 0 and (b) the file now contains
    ``return a + b``. Full family-by-family empirical verification is
    documented in the PR that un-xfailed this cell.
    """

    _HARNESS_TIMEOUT_SECONDS = 300

    @staticmethod
    def _resolve_aider_bin() -> str | None:
        """Return a usable aider binary path, or ``None`` if none present.

        Codex #1047 nit: previously the pytest skip guard only checked
        ``shutil.which("aider")``, ignoring the ``AIDER_BIN`` env var that
        the bash harness already honors. A CI operator that pins a
        non-standard binary via ``AIDER_BIN`` would see the cell skip even
        though the harness would happily run. Centralize the lookup so
        Python and bash agree.

        Lookup order (no machine-specific fallback — operators that install
        aider outside ``PATH`` point ``AIDER_BIN`` at it):
          1. ``AIDER_BIN`` env var (an executable file), then
          2. ``shutil.which("aider")`` on ``PATH``.
        """
        env_pin = os.environ.get("AIDER_BIN")
        if env_pin and Path(env_pin).is_file() and os.access(env_pin, os.X_OK):
            return env_pin
        which = shutil.which("aider")
        if which is not None:
            return which
        return None

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        harness = Path(__file__).parent / "test_aider.sh"
        assert harness.exists(), f"aider harness missing: {harness}"

        # Aider is not installable as an importable pkg here — it's the
        # end-user CLI. Skip cleanly if it's not on disk instead of
        # rebooting install policy from the test.
        if self._resolve_aider_bin() is None:
            pytest.skip(
                "aider CLI not found (checked AIDER_BIN env var and PATH) — "
                "install `pip install aider-chat` or set AIDER_BIN"
            )

        # Codex #1047 blocking: pass the FULL parsed base_url to the
        # harness, not just the port. The old ``--port`` path silently
        # rewrote host to ``127.0.0.1``, which would test the wrong
        # server if the fixture was pointed at a non-localhost host
        # (CI shard on a remote-serve node, or a devcontainer where
        # the app runs on ``host.docker.internal``). ``--base-url`` is
        # authoritative; the harness still accepts ``--port`` for
        # standalone local invocations.
        base_url = rapid_mlx_server["base_url"]

        # Drive aider against the actual served model_id — this ensures
        # LiteLLM's ``openai/<model>`` prefix in the harness lines up
        # with what the /v1/models probe returned.
        model_id = rapid_mlx_server["model_id"]

        env = os.environ.copy()
        # Belt-and-braces: also set the analytics/update opt-out env vars
        # in the parent process so a nested aider subprocess sees them
        # even if the harness accidentally strips them.
        env.setdefault("AIDER_ANALYTICS_ASKED", "1")
        env.setdefault("AIDER_CHECK_UPDATE", "false")

        try:
            result = subprocess.run(
                [
                    "bash",
                    str(harness),
                    "--model",
                    model_id,
                    "--base-url",
                    base_url,
                    "--timeout",
                    str(self._HARNESS_TIMEOUT_SECONDS),
                ],
                capture_output=True,
                text=True,
                timeout=self._HARNESS_TIMEOUT_SECONDS + 30,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"aider/{family_alias.family}: harness wall-timeout "
                f"({self._HARNESS_TIMEOUT_SECONDS + 30}s) exceeded. "
                # Codex #1048 round-8 nit: with ``text=True``,
                # ``TimeoutExpired.stdout``/``.stderr`` are ``str | None``,
                # so a bytes-literal ``or`` fallback would mix ``str`` and
                # ``bytes`` in the tail slice. Use ``""`` and take the
                # tail off a homogeneous ``str``.
                f"stdout(tail)={(exc.stdout or '')[-800:]!r} "
                f"stderr(tail)={(exc.stderr or '')[-800:]!r}"
            )

        # Assert exit 0 with the tail of stdout+stderr for diagnostics
        # — the harness itself already prints BEFORE/AFTER add.py and
        # the last 40 lines of aider's log, so this is enough to
        # root-cause any empirical failure without re-running.
        if result.returncode != 0:
            tail_out = result.stdout[-2000:] if result.stdout else ""
            tail_err = result.stderr[-1000:] if result.stderr else ""
            pytest.fail(
                f"aider/{family_alias.family}: harness exited "
                f"{result.returncode} on model {model_id!r}\n"
                f"--- stdout tail ---\n{tail_out}\n"
                f"--- stderr tail ---\n{tail_err}"
            )


class TestKiloCode:
    """Kilo Code /v1/chat/completions with tool call.

    Kilo Code is a Cline fork; wire is standard OpenAI-compat
    (see ``kilo-code.yaml``). Cell smokes a tool call the same way
    Kilo's file-read + shell tools would.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="kilo-code")


class TestCopilot:
    """GitHub Copilot CLI wire smoke.

    Pre-flight verdict: **PASS** — Copilot CLI supports BYOK via the
    env vars ``COPILOT_PROVIDER_BASE_URL``, ``COPILOT_PROVIDER_API_KEY``,
    ``COPILOT_MODEL``, and ``COPILOT_PROVIDER_TYPE=openai``. Docs:
    https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models

    Cell shape: **wire-smoke only** — the plain
    ``/v1/chat/completions`` tool-call round-trip. Driving the real
    ``copilot`` CLI as a non-interactive subprocess is blocked on
    ``gh auth login`` OAuth (interactive TTY, no ``--no-tty`` escape
    hatch as of 2026-07). A follow-up sibling PR can add a real-CLI
    subprocess cell once a token-flow harness is agreed with raullen —
    the wire-smoke here still catches server-side tool-call regressions
    that would break Copilot BYOK users.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="copilot")


class TestDroid:
    """Factory AI Droid CLI wire smoke.

    Pre-flight verdict: **PASS** — Droid CLI supports custom models via
    ``~/.factory/settings.json`` ``customModels`` array with fields
    ``model`` / ``displayName`` / ``baseUrl`` / ``apiKey`` /
    ``provider = "generic-chat-completion-api"``. Docs:
    https://docs.factory.ai/cli/byok/overview

    Cell shape: **wire-smoke only** — same rationale as ``TestCopilot``.
    Real subprocess driving requires ``droid`` first-run onboarding and
    a Factory session token; wire-smoke catches the /v1/chat/completions
    tool-call regressions that would break Droid BYOK users.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="droid")


class TestKimiCode:
    """Moonshot Kimi Code CLI wire smoke.

    Pre-flight verdict: **PASS** — Kimi Code CLI supports OpenAI-compat
    providers via ``~/.kimi/config.toml`` provider blocks with
    ``type = "openai"`` + ``base_url``. Docs:
    https://moonshotai.github.io/kimi-cli/en/configuration/providers.html

    Cell shape: **wire-smoke only** — same rationale as ``TestCopilot``.
    Real subprocess driving requires kimi-cli first-run auth flow;
    wire-smoke catches the /v1/chat/completions tool-call regressions
    that would break Kimi-Code BYOK users.
    """

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="kimi-code")
