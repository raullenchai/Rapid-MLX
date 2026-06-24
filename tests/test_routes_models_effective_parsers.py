# SPDX-License-Identifier: Apache-2.0
"""R12 MED-1 — ``/v1/models`` surfaces EFFECTIVE parsers, not static alias defaults.

Both 0.8.15 dogfood reports (Vlad r12 MED-1, Sven r12 MED-1) observed
``GET /v1/models`` advertising ``tool_call_parser: null`` and
``reasoning_parser: null`` for raw HF ids even when the runtime had
parsers actively bound — either via explicit ``--tool-call-parser`` /
``--reasoning-parser`` CLI flags, or via the auto-detect outcome.
Agentic SDKs that route on the declared parsers (and human operators
debugging tool-call issues) see misleading nulls.

This file pins the contract directly against
:mod:`vllm_mlx.routes.models`:

  * Lookup order = explicit CLI flag > auto-detect outcome > alias
    profile default > ``null``. The runtime stores the EFFECTIVE value
    (CLI override OR auto-detect) on ``ModelEntry.tool_call_parser`` /
    ``ModelEntry.reasoning_parser`` after :func:`server.load_model`
    has finished — i.e. by the time ``/v1/models`` runs, the per-entry
    field already encodes both override and auto-detect.
  * Raw HF ids (no alias profile) must surface live parsers, not
    ``null``.
  * Aliased ids must let the CLI override beat the alias default —
    aliasing must not lie.
  * Discovery-listing of an alias the server isn't currently serving
    falls back to the alias profile (preserves pre-fix behavior).
  * Truly-no-parser case still reports ``null``.

A regression here is a wire-shape break, not a unit-level bug, so we
mount a real :class:`FastAPI` app with the ``vllm_mlx.routes.models``
router and inspect the JSON response — same shape every SDK that hits
``/v1/models`` sees.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


@contextmanager
def _mounted(
    *,
    model_name: str | None,
    model_alias: str | None = None,
    tool_call_parser: str | None = None,
    reasoning_parser_name: str | None = None,
    model_registry=None,
    embedding_model_locked: str | None = None,
):
    """Mount a TestClient on the models router with the given live state.

    Snapshots + restores every mutated ``ServerConfig`` field AND the
    server-module globals the bridge fallback reads, so cases never
    leak state across each other.
    """
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import models as models_route

    app = FastAPI()
    app.include_router(models_route.router)

    cfg = get_config()
    saved = {
        k: getattr(cfg, k, None)
        for k in (
            "model_name",
            "model_alias",
            "model_registry",
            "tool_call_parser",
            "reasoning_parser_name",
            "embedding_model_locked",
            "api_key",
        )
    }
    cfg.model_name = model_name
    cfg.model_alias = model_alias
    cfg.model_registry = model_registry
    cfg.tool_call_parser = tool_call_parser
    cfg.reasoning_parser_name = reasoning_parser_name
    cfg.embedding_model_locked = embedding_model_locked
    cfg.api_key = None

    import vllm_mlx.server as srv

    saved_srv = {
        "_tool_call_parser": srv._tool_call_parser,
        "_reasoning_parser_name": srv._reasoning_parser_name,
        "_embedding_model_locked": srv._embedding_model_locked,
    }
    srv._tool_call_parser = tool_call_parser
    srv._reasoning_parser_name = reasoning_parser_name
    srv._embedding_model_locked = embedding_model_locked

    try:
        yield TestClient(app)
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
        for k, v in saved_srv.items():
            setattr(srv, k, v)


def _make_registry(*entries):
    """Build a ``ModelRegistry`` populated with the given entries."""
    from vllm_mlx.runtime.model_registry import ModelRegistry

    registry = ModelRegistry()
    for i, entry in enumerate(entries):
        registry.add(entry, is_default=(i == 0))
    return registry


def _make_entry(
    *,
    model_name: str,
    model_path: str | None = None,
    aliases=None,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
):
    from vllm_mlx.runtime.model_registry import ModelEntry

    return ModelEntry(
        engine=object(),  # opaque — none of these tests call into the engine
        model_name=model_name,
        model_path=model_path or model_name,
        aliases=set(aliases or []),
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        is_mllm=False,
        max_tokens=4096,
    )


def _by_id(body, model_id):
    for entry in body["data"]:
        if entry["id"] == model_id:
            return entry
    raise AssertionError(f"id {model_id!r} not in /v1/models: {body}")


# ---------------------------------------------------------------------------
# R12 MED-1 — raw HF id surfaces EFFECTIVE parsers
# ---------------------------------------------------------------------------


def test_raw_hf_id_surfaces_explicit_cli_parsers():
    """Single-model serve of a raw HF id (no alias profile) with
    explicit ``--tool-call-parser hermes`` / ``--reasoning-parser qwen``.

    Pre-fix this returned ``tool_call_parser: null`` / ``reasoning_parser: null``
    because the route only read the alias profile. The runtime was
    actively using the CLI-supplied parsers and the listing lied.
    """
    raw_id = "mlx-community/Qwen3-0.6B-bf16"
    with _mounted(
        model_name=raw_id,
        tool_call_parser="hermes",
        reasoning_parser_name="qwen",
    ) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, raw_id)
    assert entry["tool_call_parser"] == "hermes", (
        f"raw HF id must surface the explicit CLI tool-call parser; got {entry!r}"
    )
    assert entry["reasoning_parser"] == "qwen", (
        f"raw HF id must surface the explicit CLI reasoning parser; got {entry!r}"
    )


def test_raw_hf_id_surfaces_autodetected_parsers():
    """Auto-detect outcome lands on the server globals exactly the same
    way an explicit flag does (``server._tool_call_parser`` is set by
    ``args.tool_call_parser`` whether it came from the operator or from
    ``model_auto_config.detect_model_config``). The route can't tell
    the two paths apart — both flow through the same per-server live
    state — so this test pins that the auto-detected parser also
    surfaces, exactly like the explicit flag.
    """
    raw_id = "mlx-community/Qwen3-0.6B-bf16"
    with _mounted(
        model_name=raw_id,
        tool_call_parser="hermes",  # populated by auto-detect path
        reasoning_parser_name="qwen",
    ) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, raw_id)
    assert entry["tool_call_parser"] == "hermes"
    assert entry["reasoning_parser"] == "qwen"


def test_raw_hf_id_no_parser_bound_stays_null():
    """Truly-no-parser case must preserve the pre-fix shape — ``null``
    on both fields. The fix surfaces EFFECTIVE values; absence is
    still ``null``, not a string.
    """
    raw_id = "some-vendor/SomeModel-bf16"
    with _mounted(model_name=raw_id) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, raw_id)
    assert entry["tool_call_parser"] is None
    assert entry["reasoning_parser"] is None


# ---------------------------------------------------------------------------
# Aliasing must not lie — CLI override beats alias default
# ---------------------------------------------------------------------------


def test_alias_cli_override_beats_alias_default():
    """When an aliased model has a default parser in ``aliases.json``
    and the operator passes ``--tool-call-parser Y`` / ``--reasoning-parser Y``,
    ``/v1/models`` must report Y (the live runtime value), not the
    static alias default.
    """
    # Pick a known alias from aliases.json that has a parser default.
    from vllm_mlx.model_aliases import resolve_profile

    alias = "qwen3-0.6b-4bit"
    profile = resolve_profile(alias)
    if profile is None or not (profile.tool_call_parser or profile.reasoning_parser):
        pytest.skip(
            "Sentinel alias lost its parser default; pick another alias to pin override."
        )
    # Force CLI override to values that DIFFER from the alias default.
    override_tool = (
        "deepseek_v3" if profile.tool_call_parser != "deepseek_v3" else "hermes"
    )
    override_reasoning = (
        "deepseek_r1" if profile.reasoning_parser != "deepseek_r1" else "qwen"
    )
    with _mounted(
        model_name=profile.hf_path,
        model_alias=alias,
        tool_call_parser=override_tool,
        reasoning_parser_name=override_reasoning,
    ) as client:
        body = client.get("/v1/models").json()
    for served_id in (profile.hf_path, alias):
        entry = _by_id(body, served_id)
        assert entry["tool_call_parser"] == override_tool, (
            f"alias {served_id!r} must surface CLI override {override_tool!r}, "
            f"not its static alias default {profile.tool_call_parser!r}; got {entry!r}"
        )
        assert entry["reasoning_parser"] == override_reasoning, (
            f"alias {served_id!r} must surface CLI override {override_reasoning!r}, "
            f"not its static alias default {profile.reasoning_parser!r}; got {entry!r}"
        )


def test_alias_no_cli_override_keeps_alias_default():
    """When no explicit CLI override is passed for an aliased serve,
    the runtime still ends up with the alias profile parsers bound
    (``model_auto_config.detect_model_config`` applies the profile
    to ``args.tool_call_parser`` / ``args.reasoning_parser`` BEFORE
    the server boots — by the time ``effective_parsers_for`` runs,
    ``_tool_call_parser`` is populated with the alias profile value).

    This pins that the listing surfaces those parsers, matching what
    the runtime is actually using.

    Note: the live runtime state is AUTHORITATIVE for served ids
    (Tier 2). The alias profile default at Tier 3 only applies to
    ids the server is NOT actively serving (discovery clients
    pre-flighting an alias they might switch to). Pinning that
    Tier 3 behavior is covered by
    :func:`test_effective_parsers_helper_lookup_order` Tier 3.
    """
    from vllm_mlx.model_aliases import resolve_profile

    alias = "qwen3-0.6b-4bit"
    profile = resolve_profile(alias)
    if profile is None or not (profile.tool_call_parser or profile.reasoning_parser):
        pytest.skip(
            "Sentinel alias lost its parser default; pick another alias to pin fallback."
        )
    # Real-world post-CLI state: the alias-resolution path has
    # populated the server globals from the alias profile. The
    # listing should surface those (which happen to match the
    # alias profile default — but the source of truth is the live
    # runtime, NOT the profile read directly).
    with _mounted(
        model_name=profile.hf_path,
        model_alias=alias,
        tool_call_parser=profile.tool_call_parser,
        reasoning_parser_name=profile.reasoning_parser,
    ) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, alias)
    assert entry["tool_call_parser"] == profile.tool_call_parser
    assert entry["reasoning_parser"] == profile.reasoning_parser


# ---------------------------------------------------------------------------
# Multi-model registry: per-entry live state wins
# ---------------------------------------------------------------------------


def test_multi_model_per_entry_parsers_surface():
    """In multi-model serve, each ``ModelEntry`` carries its own live
    parsers. The listing must surface each entry's own parsers, not
    the (single) server-global value. Pre-fix the route read the
    alias profile only; the per-entry runtime state was invisible.
    """
    entry_a = _make_entry(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        tool_call_parser="hermes",
        reasoning_parser="qwen",
    )
    entry_b = _make_entry(
        model_name="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        tool_call_parser="deepseek_v3",
        reasoning_parser="deepseek_r1",
    )
    registry = _make_registry(entry_a, entry_b)
    with _mounted(model_name=None, model_registry=registry) as client:
        body = client.get("/v1/models").json()
    a = _by_id(body, "mlx-community/Qwen3-0.6B-bf16")
    b = _by_id(body, "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit")
    assert a["tool_call_parser"] == "hermes"
    assert a["reasoning_parser"] == "qwen"
    assert b["tool_call_parser"] == "deepseek_v3"
    assert b["reasoning_parser"] == "deepseek_r1"


def test_multi_model_unrelated_id_not_polluted_by_server_global():
    """The server-global parsers must NOT paint themselves onto entries
    they aren't wired for. The route's :func:`_is_served_model` gate
    is the same guard :func:`_tools_capable` already uses (Codex r4
    BLOCKING on PR #804); this pins that the parser surface respects
    the same gate. Pre-fix discovery clients reading the listing
    could see the wrong parser for an alias the server doesn't run.
    """
    entry_a = _make_entry(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        tool_call_parser=None,  # this entry has no live parser
        reasoning_parser=None,
    )
    registry = _make_registry(entry_a)
    with _mounted(
        model_name=None,
        model_registry=registry,
        tool_call_parser="hermes",  # server global is set but doesn't apply
        reasoning_parser_name="qwen",
    ) as client:
        body = client.get("/v1/models").json()
    entry = _by_id(body, "mlx-community/Qwen3-0.6B-bf16")
    # No alias profile for this raw HF id, no per-entry parser, and the
    # server global only applies to the served entry — the registry
    # surface uses per-entry state, not the cross-registry global.
    assert entry["tool_call_parser"] is None
    assert entry["reasoning_parser"] is None


# ---------------------------------------------------------------------------
# Direct helper exercise — single source of truth
# ---------------------------------------------------------------------------


def test_effective_parsers_helper_lookup_order():
    """Pin the lookup-order contract on the helper directly so a
    refactor that splits the call sites can't silently regress one
    branch. Order: per-entry live state > per-server live state >
    profile default > None.
    """
    from vllm_mlx.routes.models import effective_parsers_for

    # Tier 1 — per-entry live wins, even when both server-global and
    # profile would prefer something else.
    entry = _make_entry(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        tool_call_parser="hermes",
        reasoning_parser="qwen",
    )
    registry = _make_registry(entry)
    with _mounted(
        model_name=None,
        model_registry=registry,
        tool_call_parser="server-global-tool",
        reasoning_parser_name="server-global-reasoning",
    ):
        tool, reasoning = effective_parsers_for(
            "mlx-community/Qwen3-0.6B-bf16",
            "profile-tool",
            "profile-reasoning",
        )
        assert tool == "hermes"
        assert reasoning == "qwen"

    # Tier 2 — per-server live wins when no entry is bound (single-model
    # serve without registry).
    with _mounted(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        tool_call_parser="hermes",
        reasoning_parser_name="qwen",
    ):
        tool, reasoning = effective_parsers_for(
            "mlx-community/Qwen3-0.6B-bf16",
            "profile-tool",
            "profile-reasoning",
        )
        assert tool == "hermes"
        assert reasoning == "qwen"

    # Tier 3 — alias profile default wins when nothing live is bound.
    with _mounted(model_name="some/other-model"):
        tool, reasoning = effective_parsers_for(
            "some-aliased-id-not-served",
            "profile-tool",
            "profile-reasoning",
        )
        assert tool == "profile-tool"
        assert reasoning == "profile-reasoning"

    # Tier 4 — None falls through when neither live nor profile has it.
    with _mounted(model_name="some/other-model"):
        tool, reasoning = effective_parsers_for(
            "some-aliased-id-not-served", None, None
        )
        assert tool is None
        assert reasoning is None


def test_entry_with_none_parser_does_not_fall_back_to_profile():
    """Codex round-1 BLOCKING regression test.

    When a multi-model registry entry exists for a model whose
    operator deliberately bound NO tool/reasoning parser (e.g.
    ``--no-tool-call-parser``, or the auto-detector saw a model
    without native tool support), the entry's ``tool_call_parser`` /
    ``reasoning_parser`` are ``None`` BY DESIGN — that ``None`` is
    the live runtime state.

    The pre-fix Tier-1 branch fell back to the alias profile
    default when the entry field was falsy, which would re-introduce
    the original V-1/S-2 misreport: ``/v1/models`` would advertise
    a parser the runtime is NOT using. Clients tool-calling against
    the advertised parser would then mis-attribute or miss tool
    output entirely.

    Pin: when the registry entry is present, its parser fields are
    authoritative — ``None`` on the entry must surface as ``null``
    on the wire, NEVER as the alias profile default.
    """
    from vllm_mlx.routes.models import effective_parsers_for

    entry = _make_entry(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        tool_call_parser=None,  # operator deliberately bound no parser
        reasoning_parser=None,
    )
    registry = _make_registry(entry)
    with _mounted(model_name=None, model_registry=registry):
        # Profile default is non-None — simulates an alias whose
        # static aliases.json default would otherwise leak through.
        tool, reasoning = effective_parsers_for(
            "mlx-community/Qwen3-0.6B-bf16",
            "profile-tool",
            "profile-reasoning",
        )
        assert tool is None, (
            f"entry's live ``tool_call_parser=None`` must surface as null; "
            f"got {tool!r} (alias profile leak — codex r1 blocker)"
        )
        assert reasoning is None, (
            f"entry's live ``reasoning_parser=None`` must surface as null; "
            f"got {reasoning!r} (alias profile leak — codex r1 blocker)"
        )


def test_tier2_one_sided_live_parser_does_not_backfill_from_profile():
    """Codex round-2 BLOCKING regression test.

    Operator runs an alias and binds ONLY one parser side at the
    server-global level (e.g. ``--tool-call-parser hermes`` but no
    ``--reasoning-parser``). Pre-fix Tier 2 backfilled the missing
    side from the alias profile (``live_reasoning or
    profile_reasoning_parser``), so the listing falsely advertised
    a reasoning parser the runtime is NOT using. Reasoning-aware
    clients would then mis-attribute streamed reasoning content.

    Pin: when Tier 2 fires, both sides come from the live server
    state independently — the unbound side surfaces as ``null``,
    NEVER as the alias profile default.
    """
    from vllm_mlx.routes.models import effective_parsers_for

    with _mounted(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        tool_call_parser="hermes",  # only tool-call bound
        reasoning_parser_name=None,  # reasoning explicitly unbound
    ):
        tool, reasoning = effective_parsers_for(
            "mlx-community/Qwen3-0.6B-bf16",
            "profile-tool-default",
            "profile-reasoning-default",
        )
        assert tool == "hermes"
        assert reasoning is None, (
            f"Tier 2 must NOT backfill reasoning from profile default "
            f"when the live server state has it unbound; got {reasoning!r}"
        )

    # Symmetric case — reasoning bound, tool-call unbound.
    with _mounted(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        tool_call_parser=None,
        reasoning_parser_name="qwen",
    ):
        tool, reasoning = effective_parsers_for(
            "mlx-community/Qwen3-0.6B-bf16",
            "profile-tool-default",
            "profile-reasoning-default",
        )
        assert tool is None, (
            f"Tier 2 must NOT backfill tool from profile default "
            f"when the live server state has it unbound; got {tool!r}"
        )
        assert reasoning == "qwen"


def test_tier1_matches_guard_rejects_non_bool_truthy():
    """Codex round-2 BLOCKING regression test.

    The Tier 1 guard is ``candidate.matches(model_id) is True`` —
    strict identity on the literal ``True``. Pre-fix the guard was
    ``bool(candidate.matches(model_id)) is True``, which collapses
    any truthy non-bool return (``MagicMock``, ``"yes"``, ``1``) to
    ``True``, lets the default entry leak through, and reports its
    parsers for an unrelated id.

    Pin: a candidate whose ``matches`` returns a truthy non-``True``
    sentinel must be rejected — Tier 1 must NOT engage, and the
    helper must fall through to the profile default.
    """
    from vllm_mlx.routes.models import effective_parsers_for

    class _FakeRegistryEntry:
        # Simulates a test-double / partial-init entry whose ``matches``
        # returns a truthy non-bool (closest real-world analogue: a
        # ``MagicMock`` whose default truthiness is ``True``).
        tool_call_parser = "default-entry-tool"
        reasoning_parser = "default-entry-reasoning"

        def matches(self, name):
            return "truthy-but-not-True"  # truthy str, but not ``is True``

    class _FakeRegistry:
        # Mimics ``ModelRegistry.get_entry``'s fallback-to-default
        # behavior: returns the default entry on miss, NEVER raises.
        def __init__(self, default_entry):
            self._default = default_entry

        def get_entry(self, name):
            return self._default

    fake_default = _FakeRegistryEntry()
    fake_registry = _FakeRegistry(fake_default)

    with _mounted(model_name=None, model_registry=fake_registry):
        tool, reasoning = effective_parsers_for(
            "some-unrelated-id",
            "profile-tool",
            "profile-reasoning",
        )
        # Default-entry parsers must NOT have leaked through the guard.
        assert tool != "default-entry-tool", (
            f"strict ``is True`` guard must reject truthy-non-bool ``matches`` "
            f"return; default-entry parser leaked: {tool!r}"
        )
        assert reasoning != "default-entry-reasoning", (
            f"strict ``is True`` guard must reject truthy-non-bool ``matches`` "
            f"return; default-entry parser leaked: {reasoning!r}"
        )
        # Fell through to profile default per the lookup-order contract.
        assert tool == "profile-tool"
        assert reasoning == "profile-reasoning"


def test_tier2_served_with_both_sides_unbound_does_not_fall_back_to_profile():
    """Codex round-3 BLOCKING regression test.

    For a served alias whose runtime has BOTH parser sides explicitly
    unbound (operator passed ``--no-tool-call-parser`` AND
    ``--no-reasoning-parser``, OR auto-detect found neither), Tier 2
    must NOT fall through to the alias profile default. The runtime
    is NOT using parsers; advertising the alias's static defaults
    would lie.

    Pre-fix the gate was ``if live_tool or live_reasoning:`` —
    falsey on the (None, None) case → fell through to Tier 3 →
    advertised the alias profile defaults. Now: ``_is_served_model``
    True is authoritative, including for the all-off case.
    """
    from vllm_mlx.routes.models import effective_parsers_for

    with _mounted(
        model_name="mlx-community/Qwen3-0.6B-bf16",
        tool_call_parser=None,
        reasoning_parser_name=None,
    ):
        tool, reasoning = effective_parsers_for(
            "mlx-community/Qwen3-0.6B-bf16",
            "profile-tool-default",  # would lie if backfilled
            "profile-reasoning-default",
        )
        assert tool is None, (
            f"Tier 2 must NOT fall back to profile default for a served "
            f"id with both sides unbound; got {tool!r}"
        )
        assert reasoning is None, (
            f"Tier 2 must NOT fall back to profile default for a served "
            f"id with both sides unbound; got {reasoning!r}"
        )


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
