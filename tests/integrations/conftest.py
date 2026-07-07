# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the 0.10.2 agent + framework integration matrices.

Two matrices share this harness:

* ``test_agents_matrix.py`` — 11 Tier-1 agents × 4 families (Qwen 3.6,
  Gemma 4, DeepSeek V4, gpt-oss) = 44 cells. The three commercial-CLI
  cells added in the 0.10.2 pilot (copilot / droid / kimi-code) run as
  **wire-smoke via the shared OpenAI SDK helper** — driving the actual
  CLI binaries as subprocesses in <60 s is blocked by vendor OAuth /
  first-run onboarding flows (documented in the pre-flight verdict at
  the top of ``README.md``).
* ``test_frameworks_matrix.py`` — 3 Tier-1 frameworks × 4 families =
  12 cells.

Both matrices reuse the same server fixture, cheap-alias-per-family fixture,
and assertion helpers. The fixtures never boot the server themselves — the
operator (or CI) must have a rapid-mlx server already listening on
``RAPID_MLX_BASE_URL`` (default ``http://localhost:8000/v1``) before running
these tests. If no server is reachable, every test in the matrix ``skip``s
so ``pytest tests/integrations`` never produces a false red on a clean box.

Environment overrides
---------------------

* ``RAPID_MLX_BASE_URL`` — where to point clients (default: localhost:8000/v1).
* ``RAPID_MLX_AGENT_MATRIX_FAMILY`` — restrict matrix to one family
  (``qwen36`` / ``gemma4`` / ``deepseek`` / ``gptoss``). Handy for CI
  shards, and mandatory in Golden-Path runs so the CI job knows which
  server alias to boot.
* ``RAPID_MLX_MATRIX_STRICT`` — if ``1``, missing-server / model-mismatch
  raise instead of skipping. Off by default so a naive ``pytest`` run stays
  green.

Cheap-alias policy (W5 OOM budget + G11 disk)
---------------------------------------------

The matrix boots against **small aliases only** (≤ 8B). The full 27-35B
alias sweep is reserved for the weekly Golden Path job — never per-PR.
This keeps the per-CI-run resident footprint under the ~50 GB-per-process
ceiling agreed in workflow.md ``## W5`` step 4 (see the operator-services
baseline note).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


DEFAULT_BASE_URL = "http://localhost:8000/v1"


@dataclass(frozen=True)
class FamilyAlias:
    """A cheap-per-family alias used across the matrices."""

    family: str  # matrix column key: "qwen36" / "gemma4" / "deepseek" / "gptoss"
    alias: str  # rapid-mlx alias string (positional model arg)
    reason: str  # why this alias — used in skip messages


# Cheap per-family aliases. Kept intentionally small so the server can boot
# them under the W5 OOM budget without evicting operator services on the
# M3 Ultra. The 27-35B family flagships live in the Golden Path weekly job.
#
# ``qwen35-4b`` stands in for the Qwen 3.6 family in the small-alias matrix
# because the smallest 3.6 SKU is 27B (a 3.6 "4B" does not exist —
# see 0.10-TODO "Don't do list"). Qwen 3.5-4B shares tool + reasoning
# parser families (``hermes`` / ``qwen3``) with 3.6, so it exercises the
# same wire without loading a 15 GB weight blob per test process.
_FAMILY_ALIASES: dict[str, FamilyAlias] = {
    "qwen36": FamilyAlias(
        family="qwen36",
        # 4B stand-in for the family — see docstring above.
        alias="qwen3.5-4b-4bit",
        reason="Qwen 3.6 has no <8B SKU; qwen3.5-4b shares parsers",
    ),
    "gemma4": FamilyAlias(
        family="gemma4",
        alias="gemma-4-12b-4bit",
        reason="smallest Gemma 4 text-only alias (12B fits in ~7 GB @ 4-bit)",
    ),
    "deepseek": FamilyAlias(
        family="deepseek",
        # 0.10.2 Tier-1 rep — swapped from V4-Flash-8bit (~155 GB
        # weights per HF-API `repo_info(files_metadata=True)`) to the
        # R1-Distill-Qwen 32B 4-bit checkpoint (~16 GB) because
        # V4-Flash is single-node-infeasible on an M3 Ultra (155 GB
        # weights + G11 100 GB free floor = 255 GB required, 156 GB
        # available at this PR's execution). The V4-Flash Tier-1 slot
        # needs a hardware plan — tracked in a follow-up issue linked
        # from the PR body.
        #
        # R1-Distill-Qwen-32B-4bit stays above the "no cheap-alias"
        # bar (32B params is comfortably Tier-1) and exercises the
        # same ``deepseek`` tool-call parser + ``deepseek_r1``
        # reasoning parser V4-Flash would have (per
        # ``vllm_mlx/aliases.json``). No parser-coverage loss.
        alias="deepseek-r1-32b-4bit",
        reason=(
            "V4-Flash-8bit is 155 GB single-node-infeasible; "
            "R1-Distill-Qwen-32B-4bit exercises the same tool_call + "
            "reasoning parsers at ~16 GB"
        ),
    ),
    "gptoss": FamilyAlias(
        family="gptoss",
        alias="gpt-oss-20b-mxfp4-q8",
        reason="smallest gpt-oss (20B MXFP4-Q8 ~11 GB); no <20B in the family",
    ),
}


def _families_in_scope() -> tuple[str, ...]:
    """Return the families to parametrize over.

    Honours ``RAPID_MLX_AGENT_MATRIX_FAMILY`` for CI sharding — set that
    env to one family key to restrict the run to a single column.
    """
    only = os.environ.get("RAPID_MLX_AGENT_MATRIX_FAMILY", "").strip()
    if only:
        if only not in _FAMILY_ALIASES:
            raise ValueError(
                f"RAPID_MLX_AGENT_MATRIX_FAMILY={only!r} unknown; "
                f"valid: {sorted(_FAMILY_ALIASES)}"
            )
        return (only,)
    return tuple(_FAMILY_ALIASES.keys())


def _strict() -> bool:
    return os.environ.get("RAPID_MLX_MATRIX_STRICT", "").strip() == "1"


def matrix_strict_mode() -> bool:
    """Public accessor for ``RAPID_MLX_MATRIX_STRICT``.

    Cells use this to decide whether to skip on a server / route / SDK
    failure (default, non-strict) or fail the CI job (strict). CI shards
    that want per-cell coverage enforcement set ``RAPID_MLX_MATRIX_STRICT=1``
    before running the matrix.
    """
    return _strict()


def strict_skip_or_fail(reason: str) -> None:
    """Skip in non-strict mode; fail in strict mode.

    Consolidates the "cell degraded, not red" pattern so a broken
    server-side route or a regressed SDK doesn't quietly hide behind a
    green skipped cell when the operator asked for enforcement via
    ``RAPID_MLX_MATRIX_STRICT=1``. Codex #1030 flagged the earlier all-skip
    pattern as regression-hiding.
    """
    if _strict():
        pytest.fail(reason)
    pytest.skip(reason)


# --------------------------------------------------------------------------- #
# Server fixture
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def rapid_mlx_base_url() -> str:
    """Return the base URL of the rapid-mlx server under test."""
    return os.environ.get("RAPID_MLX_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


@pytest.fixture(scope="session")
def rapid_mlx_server(rapid_mlx_base_url: str) -> dict[str, Any]:
    """Verify a rapid-mlx server is reachable; return metadata.

    Yields a dict with ``base_url`` and ``model_id`` (the first entry from
    ``/v1/models``). If no server is reachable, this fixture ``skip``s the
    dependent test unless ``RAPID_MLX_MATRIX_STRICT=1`` is set — that flag
    turns the miss into a hard fail so CI can enforce coverage.
    """
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed — matrix skipped")

    try:
        resp = httpx.get(f"{rapid_mlx_base_url}/models", timeout=3.0)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            raise RuntimeError("empty /v1/models response")
        model_id = data[0]["id"]
    except Exception as exc:  # noqa: BLE001 — surface the underlying error
        message = (
            f"No rapid-mlx server reachable at {rapid_mlx_base_url}: {exc!r}. "
            "Start one with `rapid-mlx serve <alias>` before running the "
            "matrix, or set RAPID_MLX_MATRIX_STRICT=1 to hard-fail instead."
        )
        if _strict():
            pytest.fail(message)
        pytest.skip(message)

    return {"base_url": rapid_mlx_base_url, "model_id": model_id}


# --------------------------------------------------------------------------- #
# Family parametrization
# --------------------------------------------------------------------------- #


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Auto-parametrize any test that requests a ``family_alias`` argument."""
    if "family_alias" in metafunc.fixturenames:
        families = _families_in_scope()
        aliases = [_FAMILY_ALIASES[f] for f in families]
        metafunc.parametrize(
            "family_alias",
            aliases,
            ids=[a.family for a in aliases],
        )


@pytest.fixture(scope="session")
def family_alias_for_active_server(
    rapid_mlx_server: dict[str, Any],
) -> FamilyAlias | None:
    """Best-effort mapping from the running server's model_id → family.

    Returns ``None`` if the running model doesn't match any known family
    prefix — matrix tests then skip themselves so we never assert against
    a wire the operator's booted server isn't actually running.
    """
    mid = rapid_mlx_server["model_id"].lower()
    if mid.startswith("qwen3.6") or "qwen3.6" in mid:
        return _FAMILY_ALIASES["qwen36"]
    if mid.startswith("gemma-4") or "gemma-4" in mid:
        return _FAMILY_ALIASES["gemma4"]
    if mid.startswith("gpt-oss") or "gpt-oss" in mid:
        return _FAMILY_ALIASES["gptoss"]
    # DeepSeek family — 0.10.2 Tier-1. Match both:
    #  * the R1-Distill-Qwen 32B 4-bit variant used as Tier-1 rep in
    #    the PR-2 pilot (served id = "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit");
    #  * and any future ``deepseek-v4-*`` variants — reserved for the
    #    hardware-plan follow-up.
    if (
        mid.startswith("deepseek-v4")
        or "deepseek-v4" in mid
        or mid.startswith("deepseek-r1")
        or "deepseek-r1" in mid
    ):
        return _FAMILY_ALIASES["deepseek"]
    # Qwen 3.5 stands in for Qwen 3.6 in the small-alias matrix (see
    # ``_FAMILY_ALIASES['qwen36'].reason``). Substring match — not just
    # startswith — because the served id may carry a repo prefix
    # (``mlx-community/Qwen3.5-4B-MLX-4bit``), matching how the qwen3.6
    # / gemma-4 / gpt-oss / deepseek arms above resolve full HF paths.
    if mid.startswith("qwen3.5") or "qwen3.5" in mid:
        return _FAMILY_ALIASES["qwen36"]
    return None


# --------------------------------------------------------------------------- #
# Family-specific strict-xfail: DeepSeek R1-Distill tool_call cells
# --------------------------------------------------------------------------- #
#
# Root-cause (G8, verified 2026-07-06 on both 4bit + 8bit weights): the
# 0.10.2 Tier-1 DeepSeek rep — ``mlx-community/DeepSeek-R1-Distill-Qwen-
# 32B-4bit`` — architecturally cannot emit OpenAI-shape ``tool_calls``.
# R1's post-training (SFT + RLHF) was reasoning-only per DeepSeek's own
# paper (arXiv 2501.12948 §2.3.3), and distillation into Qwen 32B
# preserved that gap: the refusal pattern
#
#   "I cannot provide the current weather in Tokyo as I cannot access
#    the get_weather tool."
#
# is deterministic across 4bit (16 GB) and 8bit (34.8 GB) variants —
# not a quantization artifact, not a rapid-mlx parser bug. The base
# Qwen 2.5-32B tool-emission capability was lost during distillation.
#
# Rather than downgrade these 9 cells to skips (G8: "root-cause failures,
# do not hide behind skips") we mark them ``xfail(strict=True)`` with
# the architectural reason surfaced in test output. This matches the
# existing OpenHands / Aider strict-xfail pattern in
# ``test_agents_matrix.py``. If a future rapid-mlx change (or a fine-
# tune, or a V4-Flash upgrade) DID unlock tool_calls for the DeepSeek
# family, the strict marker would XPASS and force a revisit.
#
# Full V4-Flash coverage for the family (which was tool-trained) is
# tracked in follow-up issue **#1041** ("hardware plan needed") — the
# quant weights are 155 GB and single-node-infeasible on the M3 Ultra
# under the G11 100 GB free-disk floor.

_DEEPSEEK_R1_TOOLCALL_XFAIL_NODEIDS = frozenset(
    {
        # test_agents_matrix.py — OpenAI-wire agents that require true
        # ``tool_calls`` emission (opencode, qwen-code, hermes, kilo-code,
        # copilot, droid, kimi-code). CodexCLI + ClaudeCode use text-only
        # routes and PASS on R1-Distill, so they are NOT in this list.
        "test_agents_matrix.py::TestOpenCode",
        "test_agents_matrix.py::TestQwenCode",
        "test_agents_matrix.py::TestHermesAgent",
        "test_agents_matrix.py::TestKiloCode",
        "test_agents_matrix.py::TestCopilot",
        "test_agents_matrix.py::TestDroid",
        "test_agents_matrix.py::TestKimiCode",
        # test_frameworks_matrix.py — LangChain bind_tools + PydanticAI
        # @agent.tool_plain both require ``tool_calls`` emission. Smolagents
        # uses ToolCallingAgent's code-execution style which routes without
        # the OpenAI tool_call shape, so smolagents PASSES on R1-Distill
        # and is NOT in this list.
        "test_frameworks_matrix.py::TestLangChain",
        "test_frameworks_matrix.py::TestPydanticAI",
    }
)

_DEEPSEEK_R1_XFAIL_REASON = (
    "DeepSeek-R1-Distill-Qwen-32B (0.10.2 Tier-1 DeepSeek rep, both 4bit "
    "and 8bit verified) architecturally cannot emit OpenAI-shape "
    "tool_calls: R1 post-training was reasoning-only (arXiv 2501.12948 "
    "§2.3.3), and distillation into Qwen 32B preserved that gap — the "
    "refusal 'I cannot access the get_weather tool' reproduces "
    "deterministically at both quant levels. Root-caused (G8), not a "
    "parser bug. V4-Flash (which was tool-trained) is 155 GB and "
    "single-node-infeasible on M3 Ultra under the G11 100 GB floor — "
    "tracked in follow-up issue #1041 (hardware plan)."
)


# --------------------------------------------------------------------------- #
# Family-specific strict-xfail: gpt-oss × OpenHands (harmony format mismatch)
# --------------------------------------------------------------------------- #
#
# Note (2026-07-07 revised): gpt-oss + OpenHands remains strict-xfail'd, but
# the ROOT CAUSE has changed. PR #1051 (channel-scoped user stops in the
# harmony scheduler path; see ``vllm_mlx/reasoning/harmony_stop.py``) fixed
# the rapid-mlx-side wire bug — a wire-level probe against the running
# server with ``stop=["</execute_bash>", "</execute_ipython>"]`` now returns
# non-empty ``content`` and ``finish_reason=stop`` instead of the pre-fix
# empty-content premature stop. That fix is pinned at the unit level by
# ``tests/test_harmony_stop_final_channel_only.py``.
#
# End-to-end still XFAILs because gpt-oss's native harmony output format
# (analysis + final channels, plain markdown code in the final channel)
# does not emit the ``<execute_bash>…</execute_bash>`` /
# ``<execute_ipython>…</execute_ipython>`` text-action XML tags that
# OpenHands' CodeActAgent parses. CodeActAgent treats the reply as an
# empty ``MessageAction`` → prompts "Request user input >>" → EOFError on
# the non-interactive stdin → 300 s wall-clock timeout, ``add.py`` never
# rewritten. This is an upstream OpenHands CodeActAgent parser gap
# (harmony format not yet supported), not a rapid-mlx bug. Filed as an
# informational note in
# https://github.com/All-Hands-AI/OpenHands/issues/15167.
#
# If a future OpenHands release DOES add harmony-format parsing (or if a
# CodeActAgent variant learns to project harmony's final-channel markdown
# fences into text-action tags), this strict-xfail will XPASS and force
# the marker to be removed.

_GPTOSS_OPENHANDS_XFAIL_NODEID = "test_agents_matrix.py::TestOpenHands"
_GPTOSS_OPENHANDS_XFAIL_FAMILY = "gptoss"  # matches the parametrize id
_GPTOSS_OPENHANDS_XFAIL_REASON = (
    "gpt-oss's native harmony output format (analysis + final channels, "
    "plain markdown code in the final channel) does not emit the "
    "<execute_bash>/<execute_ipython> text-action XML tags that OpenHands' "
    "CodeActAgent parses. The rapid-mlx-side wire-level fix from PR #1051 "
    "is confirmed via ``tests/test_harmony_stop_final_channel_only.py`` "
    "(harmony parser now channel-scopes user stops), but the end-to-end "
    "harness still times out at 300 s because CodeActAgent treats the "
    "final-channel markdown reply as an empty MessageAction and blocks on "
    "user input. Requires upstream OpenHands harmony parser support; "
    "tracked at https://github.com/All-Hands-AI/OpenHands/issues/15167."
)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply strict-xfail markers to family-specific cells.

    Two independent xfail sets are applied here:

    * The 9 DeepSeek R1-Distill tool-call cells (block above), driven by
      the R1 architectural tool-emission gap.
    * The single gpt-oss × OpenHands cell (block just above), driven by
      the harmony ↔ CodeActAgent text-action format mismatch documented
      upstream.
    """
    del config  # unused — items already carry the config context.
    for item in items:
        # DeepSeek R1-Distill tool-call gap.
        if "[deepseek]" in item.nodeid:
            for prefix in _DEEPSEEK_R1_TOOLCALL_XFAIL_NODEIDS:
                if prefix in item.nodeid:
                    item.add_marker(
                        pytest.mark.xfail(
                            reason=_DEEPSEEK_R1_XFAIL_REASON,
                            strict=True,
                        )
                    )
                    break
        # gpt-oss × OpenHands harmony-format mismatch.
        if (
            _GPTOSS_OPENHANDS_XFAIL_NODEID in item.nodeid
            and f"[{_GPTOSS_OPENHANDS_XFAIL_FAMILY}]" in item.nodeid
        ):
            item.add_marker(
                pytest.mark.xfail(
                    reason=_GPTOSS_OPENHANDS_XFAIL_REASON,
                    strict=True,
                )
            )


@pytest.fixture(autouse=True)
def _guard_family_matches_server(request: pytest.FixtureRequest) -> None:
    """Autouse guard: skip / fail a family cell when the server model doesn't match.

    Codex #1030 flagged that parametrizing every cell over all three
    families lets a single-family server (e.g. Qwen 3.6 only) silently
    "cover" Gemma 4 / gpt-oss cells. This guard fires per cell:

    * cell fixture requests ``family_alias`` — the parametrized target;
    * ``family_alias_for_active_server`` maps the running model → family;
    * mismatch → ``strict_skip_or_fail`` (fail in strict, skip otherwise).

    Codex #1030 round-4 finding 1: ``rapid_mlx_server`` and
    ``family_alias_for_active_server`` are fetched **lazily** — only when
    a cell has actually opted in by requesting ``family_alias``. Fetching
    them unconditionally would force every existing deep-flow test in
    ``tests/integrations/`` through the /v1/models probe, changing their
    behavior. The lazy fetch keeps this fixture scoped to matrix cells.
    """
    if "family_alias" not in request.fixturenames:
        return
    try:
        cell_family: FamilyAlias = request.getfixturevalue("family_alias")
    except pytest.FixtureLookupError:
        return
    # Lazy fetch — only after we know this cell opted into the family matrix.
    server_info: dict[str, Any] = request.getfixturevalue("rapid_mlx_server")
    active: FamilyAlias | None = request.getfixturevalue(
        "family_alias_for_active_server"
    )
    if active is None:
        strict_skip_or_fail(
            f"cell {cell_family.family}: running model "
            f"{server_info['model_id']!r} doesn't map to any known family "
            f"(qwen36 / gemma4 / deepseek / gptoss)."
        )
        return
    if active.family != cell_family.family:
        strict_skip_or_fail(
            f"cell {cell_family.family}: running server is {active.family} "
            f"({server_info['model_id']!r}) — coverage for {cell_family.family} "
            f"belongs in a separate matrix run "
            f"(RAPID_MLX_AGENT_MATRIX_FAMILY={cell_family.family})."
        )


# --------------------------------------------------------------------------- #
# Assertion helpers — shared across both matrices
# --------------------------------------------------------------------------- #


def assert_content_nonempty(text: str, ctx: str = "") -> None:
    """Assert the model produced non-empty visible content."""
    assert isinstance(text, str), f"{ctx}: expected str, got {type(text)!r}"
    assert text.strip(), f"{ctx}: empty content"


def assert_tool_call_shape(tool_call: dict[str, Any]) -> None:
    """Assert an OpenAI-shape tool call dict is valid."""
    assert isinstance(tool_call, dict), f"tool_call not a dict: {tool_call!r}"
    assert tool_call.get("id"), f"tool_call missing id: {tool_call!r}"
    assert tool_call.get("type") == "function", tool_call
    fn = tool_call.get("function")
    assert isinstance(fn, dict), f"tool_call.function not a dict: {fn!r}"
    assert fn.get("name"), f"tool_call.function missing name: {fn!r}"
    args = fn.get("arguments")
    assert isinstance(args, str), (
        f"tool_call.function.arguments must be JSON string: {args!r}"
    )
    # arguments should parse as JSON (may be an empty object for no-arg tools)
    try:
        json.loads(args)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"tool_call.function.arguments not JSON-parseable: {args!r} ({exc})"
        ) from exc


def assert_stream_deltas_valid(events: list[dict[str, Any]]) -> None:
    """Assert an OpenAI streaming response yielded well-formed deltas."""
    assert events, "no streaming events collected"
    assert any(
        ev.get("choices", [{}])[0].get("delta", {}).get("content") for ev in events
    ), f"no content deltas in {len(events)} events"


def assert_no_analysis_channel_leak(text: str) -> None:
    """Assert the openai-harmony ``analysis`` channel didn't leak into content.

    gpt-oss models emit ``<|channel|>analysis`` / ``<|channel|>final`` markers
    around chain-of-thought; the server must strip / route the analysis
    channel to ``reasoning_content``, not the visible answer. Regression
    fixture referenced by Continue #8990.
    """
    for marker in ("<|channel|>analysis", "analysis<|message|>", "<|start|>analysis"):
        assert marker not in text, (
            f"gpt-oss analysis-channel leak into content: found {marker!r} in {text!r}"
        )


def assert_no_think_tag_leak(text: str) -> None:
    """Assert ``<think>...</think>`` traces don't leak into visible content."""
    for marker in ("<think>", "</think>", "<|thinking|>", "<|/thinking|>"):
        assert marker not in text, (
            f"think-tag leak into content: found {marker!r} in {text!r}"
        )


# --------------------------------------------------------------------------- #
# Public API — what matrix files import
# --------------------------------------------------------------------------- #


__all__ = [
    "FamilyAlias",
    "assert_content_nonempty",
    "assert_no_analysis_channel_leak",
    "assert_no_think_tag_leak",
    "assert_stream_deltas_valid",
    "assert_tool_call_shape",
    "matrix_strict_mode",
    "strict_skip_or_fail",
]
