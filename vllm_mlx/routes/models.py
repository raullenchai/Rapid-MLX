# SPDX-License-Identifier: Apache-2.0
"""Model listing endpoints.

The OpenAI-canonical `/v1/models` and `/v1/models/{id}` endpoints
serve ``ModelInfo`` shapes that carry Rapid-MLX vendor extensions
(see ``api/models.ModelInfo``). The extensions surface per-alias
profile data — curated sampling, hybrid/MoE flags, parser pair,
modality — pulled from ``model_aliases.resolve_profile``. OpenAI
clients ignore the extra fields per spec; rapid-desktop reads
them to auto-apply calibrated defaults so a user opening a chat
on ``qwen3.5-9b-4bit`` doesn't have to hand-tune sliders.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..api.models import ModelInfo, ModelsResponse
from ..api.utils import is_mllm_model
from ..config import get_config
from ..middleware.auth import verify_api_key
from ..model_aliases import resolve_profile

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_context_window(model_id: str) -> int | None:
    """Return the engine-advertised max prompt-token context window for
    ``model_id`` when an engine is loaded for it, else ``None``.

    The currently-loaded engine knows the real cap — it's the same
    chain the request-time context-length guard consults
    (``service.helpers.get_model_max_context``), so advertising it on
    ``/v1/models`` keeps the client's "max tokens" slider lined up
    with what the server will actually enforce. Issue #363:
    rapid-desktop's PR #318 consumer needs this to auto-scale the
    chat-input cap; absent the field the consumer fell through to a
    desktop-side per-family heuristic that drifted out of sync with
    every long-context release.

    Resolution:
      * Single-model serve — ``cfg.engine`` is THE loaded engine and
        the route's only entries are ``cfg.model_name`` and
        ``cfg.model_alias``; both surface the same window.
      * Multi-model serve — look up the matching ``ModelEntry`` via
        the registry's index (NOT ``get_engine`` which falls back to
        the default engine on miss; that would advertise the wrong
        cap for an unloaded alias).
      * No live engine for ``model_id`` — return ``None`` so the
        client falls back to its own per-family default. The desktop
        carries that fallback as defense-in-depth.

    Failures inside ``get_model_max_context`` (missing attributes,
    tokenizer probe raises) must NOT 500 the listing endpoint — they
    fall through to ``None`` and the request still completes. The
    helper's own fallback (``_FALLBACK_MAX_CONTEXT_TOKENS = 4 Mi``)
    is a DoS sentinel for the request-time guard, NOT a number worth
    advertising to clients; we suppress it here by treating any
    integer ≥ ``_DOS_SENTINEL_FLOOR`` as "no useful value" so the
    desktop's per-family heuristic still wins for un-introspectable
    models.
    """
    # The DoS sentinel inside ``get_model_max_context`` is 4 MiB
    # (4_194_304). Anything at or above that floor is the sentinel,
    # not a real context window — no production LLM today exposes a
    # 4M-token window on Apple Silicon. Hoisted as a local constant
    # so the comparison is explicit and the relationship to the
    # helper's fallback constant is obvious to future readers.
    _DOS_SENTINEL_FLOOR = 4_194_304
    engine = None
    cfg = get_config()
    if cfg.model_registry is not None:
        try:
            entry = cfg.model_registry.get_entry(model_id)
        except KeyError:
            entry = None
        # ``get_entry`` falls back to the default entry on miss — guard
        # that the entry we got actually matches ``model_id`` so the
        # listing doesn't advertise the default engine's cap for every
        # unloaded alias.
        if entry is not None and entry.matches(model_id):
            engine = entry.engine
    else:
        candidate = getattr(cfg, "engine", None)
        if candidate is not None:
            served = {cfg.model_name, cfg.model_alias} - {None}
            if model_id in served:
                engine = candidate
    if engine is None:
        return None
    try:
        # Imported lazily to keep this module's import surface small;
        # ``service.helpers`` pulls in the request lifecycle which we
        # don't need at module-load time.
        from ..service.helpers import get_model_max_context
    except Exception:  # noqa: BLE001
        return None
    try:
        window = get_model_max_context(engine)
    except Exception as exc:  # noqa: BLE001
        # A failed probe MUST NOT 500 ``/v1/models``; log once and let
        # the client fall back to its per-family heuristic.
        logger.debug(
            "context_window probe failed for %s: %s", model_id, exc, exc_info=False
        )
        return None
    if not isinstance(window, int) or window <= 0:
        return None
    if window >= _DOS_SENTINEL_FLOOR:
        return None
    return window


def _reported_modality(model_id: str, profile_modality: str) -> str:
    """Return the modality the wire-level ``/v1/models`` should advertise.

    ``AliasProfile.modality`` is an engine-routing discriminator:
    ``text`` selects the AR ``BatchedEngine`` lane, ``text-diffusion``
    selects the diffusion lane. Vision-Language aliases (qwen3-vl-*,
    gemma-3n-*, etc.) deliberately keep ``modality="text"`` internally
    because their language backbone IS routed through the AR lane —
    the multimodal path is layered on top via ``MLLMBatchGenerator``,
    not a separate engine. Reusing the routing discriminator as the
    externally-reported modality therefore mislabels VL models as
    text-only and downstream clients skip the image content shapes
    they otherwise would have sent (F-067).

    The fix derives the reported value from the same ``is_mllm_model``
    detector the rest of the codebase already trusts to gate VLM
    routing (see ``cli.py``/``server.py``: ``is_mllm=getattr(args,
    "mllm", False) or is_mllm_model(args.model)``). This keeps engine
    routing untouched while reporting an accurate capability hint on
    the wire. Non-``text`` profile modalities (e.g. ``text-diffusion``)
    bypass the detector entirely so existing dispatched lanes still
    advertise their canonical value.
    """
    if profile_modality != "text":
        return profile_modality
    try:
        if is_mllm_model(model_id):
            return "image"
    except Exception:  # noqa: BLE001
        # ``is_mllm_model`` reads local config / HF cache; any IO or
        # parse failure must NOT block ``/v1/models``. Fall back to
        # the profile's declared modality so the endpoint stays
        # available even when the cache layer is misbehaving.
        pass
    return profile_modality


def _locked_embedding_id() -> str | None:
    """Return the configured embedding model id, if any.

    Reads from ``ServerConfig.embedding_model_locked`` first and falls
    back to the ``server._embedding_model_locked`` global so the
    capability shows up even before ``_sync_config`` has bridged the
    value (mirrors the same bridge the embeddings route uses).
    """
    cfg = get_config()
    locked = cfg.embedding_model_locked
    if locked is not None:
        return locked
    try:
        from ..server import _embedding_model_locked as _server_locked

        return _server_locked
    except Exception:  # noqa: BLE001
        return None


def _is_vlm(model_id: str, profile_modality: str | None) -> bool:
    """Return True when ``model_id`` accepts image input.

    Single source of truth for VLM detection on the wire. Combines
    two signals:

    * ``profile_modality != "text"`` — explicit alias registration
      (``aliases.json``) wins when the profile flags a non-text
      modality. Catches diffusion / audio / future modalities the
      raw HF-id heuristic can't see.
    * :func:`vllm_mlx.api.utils.is_mllm_model` — the same detector
      ``cli.py`` and ``server.py`` use to route requests through
      ``MLLMBatchGenerator``. Covers VLM aliases that internally
      keep ``modality="text"`` (their language backbone IS the AR
      lane; the multimodal path layers on top — see
      :func:`_reported_modality`) and raw HF VLM repos that have no
      alias entry yet.

    Any exception inside ``is_mllm_model`` (corrupt local config,
    HF cache I/O failure) collapses to ``False`` so the
    ``/v1/models`` endpoint stays available — losing the ``"vision"``
    capability tag is far less harmful than 500'ing the whole listing.
    """
    if profile_modality is not None and profile_modality != "text":
        # Non-text profile modalities are authoritative (e.g.
        # ``image``, ``text-diffusion``). The wire-reported modality
        # may still flip to ``image`` via ``_reported_modality``
        # below, but the capability tag is decided independently.
        if profile_modality == "image":
            return True
    try:
        return bool(is_mllm_model(model_id))
    except Exception:  # noqa: BLE001
        return False


def _is_served_model(model_id: str) -> bool:
    """Return True when ``model_id`` is the (or one of the) model(s)
    the server is currently serving.

    The server-level tool parser flag (``--tool-call-parser`` or the
    auto-detected value) applies ONLY to the model the server is
    actually serving — it's a per-server flag, not a per-registry
    setting. Without this guard a single configured parser would
    paint ``"tools"`` onto every entry returned by ``/v1/models``,
    including unrelated registry / discovery entries the parser
    isn't wired for. Codex r4 BLOCKING on PR #804.

    Multi-model serve (``model_registry``) lists every served entry;
    membership in the registry is the served-set. Single-model serve
    uses ``model_name`` / ``model_alias``. Both surfaces are
    consulted because the registry is set on multi-model serve only.
    """
    cfg = get_config()
    if cfg.model_registry is not None:
        try:
            return model_id in cfg.model_registry
        except Exception:  # noqa: BLE001
            return False
    return model_id in {cfg.model_name, cfg.model_alias} - {None}


def _tools_capable(model_id: str, profile_tool_parser: str | None) -> bool:
    """Return True when ``model_id`` exposes a tool-call surface.

    Two-tier decision:

    * **Per-entry alias signal** — the alias profile carries a
      non-empty ``tool_call_parser`` (qwen, hermes, mistral, …) set
      in ``aliases.json`` for the tool-capable families. This is
      authoritative for every registered alias regardless of which
      model the server is currently serving — discovery clients
      use it to pre-flight tool-call support on aliases they're
      considering switching to.
    * **Server-global fallback** — ``ServerConfig.tool_call_parser``
      OR ``vllm_mlx.server._tool_call_parser``. ONLY applied when
      ``model_id`` IS the currently served model (or one of them in
      multi-model serve). Without this gate a single configured
      parser would paint ``"tools"`` onto every unrelated registry
      entry — codex r4 BLOCKING on PR #804. The dual read (config +
      server global) mirrors :func:`_locked_embedding_id`'s
      bridge-order fallback so the capability shows up even before
      ``_sync_config`` has plumbed the value.
    """
    if profile_tool_parser:
        return True
    if not _is_served_model(model_id):
        return False
    cfg = get_config()
    if getattr(cfg, "tool_call_parser", None):
        return True
    try:
        from ..server import _tool_call_parser as _server_tool_parser

        return bool(_server_tool_parser)
    except Exception:  # noqa: BLE001
        return False


def _detect_capabilities(
    model_id: str,
    profile_modality: str | None = None,
    profile_tool_parser: str | None = None,
) -> list[str]:
    """Compute the ``capabilities`` tag list for ``model_id``.

    F-D01: pre-fix only the configured embedding model carried a tag
    (``"embedding"``) — every other entry returned ``[]`` even when
    it accepted image input or exposed a tool-call surface.
    Downstream clients that route on capabilities couldn't tell a VLM
    from a text-only model from the wire.

    The unified detector emits the full set in a stable order:

    * ``"embedding"`` — exactly when this id is the
      ``--embedding-model`` locked at startup. The chat surface is
      still 400'd by the embeddings route guard for non-locked ids
      (H-09), so we don't combine ``"text"`` with ``"embedding"``.
    * ``"text"`` — every non-embedding model accepts text input.
    * ``"vision"`` — :func:`_is_vlm` returns True (profile flags
      ``image`` or :func:`is_mllm_model` matches).
    * ``"tools"`` — :func:`_tools_capable` returns True (alias
      profile has a parser, or the server is running with one).

    Order is fixed: ``text → vision → tools`` (or just
    ``["embedding"]`` for the embedding entry). Tests pin this so
    a future addition (e.g. ``"audio"``) is a deliberate, reviewed
    change rather than a silent reordering.
    """
    locked = _locked_embedding_id()
    if locked is not None and model_id == locked:
        # H-09 invariant preserved: the embedding model carries
        # ``"embedding"`` exclusively. Combining it with ``"text"``
        # would mislead clients into routing chat traffic at the
        # embedding model id — the chat surface is not wired.
        return ["embedding"]

    caps: list[str] = ["text"]
    if _is_vlm(model_id, profile_modality):
        caps.append("vision")
    if _tools_capable(model_id, profile_tool_parser):
        caps.append("tools")
    return caps


def _reported_modality_for_embedding(locked_id: str) -> str:
    """Return the ``modality`` field for the embedding entry.

    F-D01 cosmetic fix: pre-fix the embedding entry advertised
    ``modality=None`` while VLMs advertised ``modality="image"``.
    Clients reading ``modality`` to distinguish lanes saw a
    three-way (text / image / null) shape instead of the documented
    text / image / embedding axis. Embedding models accept text
    input, so the on-wire modality is ``"text"`` — the
    ``capabilities=["embedding"]`` tag is what distinguishes the
    lane, not the modality.
    """
    return "text"


def _resolve_audio_entry(model_id: str):
    """Return the audio registry entry for ``model_id`` (alias OR HF id).

    R11-B-F4 (Bo 0.8.12 dogfood): the wire-level ``/v1/models`` listing
    for an audio-only alias (``rapid-mlx serve kokoro``) advertised
    ``capabilities=["text"]`` and ``modality=null`` because the
    capability detector was wired only for text / VLM / embedding
    lanes. Drop-in OpenAI clients couldn't tell an audio alias from a
    text model from the wire — and the ``audio_lanes`` field they
    DID get back was the server-wide lane-health snapshot, not the
    per-entry capability hint.

    The fix introspects the audio registry (the SAME source of truth
    that drives ``serve_command``'s audio-mode fork — see
    :mod:`vllm_mlx.audio.registry`) and returns the resolved entry
    when ``model_id`` is a known audio alias OR a registered audio HF
    id. Otherwise returns ``None`` so the text / VLM / embedding path
    continues to own the entry.

    The lookup is intentionally read-only and exception-tolerant: any
    failure (``[audio]`` extra not installed, registry JSON parse
    error) falls through to ``None`` so ``/v1/models`` stays 200.
    """
    try:
        from ..audio.registry import resolve_audio_alias
    except Exception:  # noqa: BLE001
        return None
    try:
        return resolve_audio_alias(model_id)
    except Exception:  # noqa: BLE001
        return None


def _audio_routes_mounted() -> bool:
    """Task #292: True iff the canonical ``/v1/audio/*`` router is attached.

    On text-only servers (Bo R13/R14 fuzz wave) the audio router is
    NOT registered, so ``/v1/audio/transcriptions`` etc. return a stock
    FastAPI 404. ``/v1/models`` should reflect that — clients shouldn't
    see ``audio_lanes={"stt":"degraded"}`` on a server that wouldn't
    answer ``/v1/audio/transcriptions`` at all.

    Codex r0 BLOCKING #1: the predicate ONLY inspects the live ASGI
    app — never the ``ServerConfig.enable_audio_lane`` flag. The flag
    is the upstream INPUT to the gate; the route table is the downstream
    OUTPUT. A boot path that sets the flag but hasn't yet called the
    registration hook would otherwise advertise ``audio_lanes`` while
    ``/v1/audio/*`` still 404s.

    Codex r2 BLOCKING: the previous prefix-scan implementation
    false-positived on operator-added subpaths (e.g.
    ``/v1/audio/health`` probes) — same shape as the
    ``register_audio_routes`` NIT that codex r0 caught. Now we check
    the app-local sentinel attribute that
    :func:`vllm_mlx.routes.audio.register_audio_routes` stamps on the
    app on a successful registration. The sentinel is the single source
    of truth for "the canonical audio router is mounted on this app".

    Falls through to False on any inspection failure (defensive — the
    listing must stay 200 even if the app/router shape changes).
    """
    try:
        from ..routes.audio import _AUDIO_REGISTRATION_SENTINEL
        from ..server import app as _server_app
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(getattr(_server_app, _AUDIO_REGISTRATION_SENTINEL, False))
    except Exception:  # noqa: BLE001
        return False


def _audio_lane_snapshot() -> dict[str, str] | None:
    """Return the current per-lane audio status, or ``None`` when no
    deep probe has run.

    F-K-CAPABILITIES-OMIT-AUDIO: surfaces the recorded outcome of
    :func:`vllm_mlx.audio.probe.deep_probe_audio_lane` on every
    ``ModelInfo`` returned by ``/v1/models``. Pre-fix, audio lane
    health was invisible — a Whisper backend that 500'd on every
    request still advertised the same ``capabilities`` list as a
    healthy one. Now a degraded lane shows up as
    ``audio_lanes: {"stt": "degraded", ...}`` so dashboards / the
    desktop can warn before sending real traffic.

    Task #292 (Bo R13/R14): if the audio router is NOT mounted on the
    live ASGI app (text-only server boot, no ``--enable-audio``), this
    returns ``None`` so ``/v1/models`` doesn't advertise lane health
    for routes that would answer 404. The pre-fix shape was
    misleading: a text-only Qwen3-7B-4bit server reported
    ``audio_lanes={"stt":"missing","tts":"missing"}`` even though
    ``/v1/audio/transcriptions`` was about to 500.

    The function is intentionally tolerant: any failure resolving
    the probe module (e.g. ``[audio]`` extra not installed, so the
    probe module isn't even reachable) returns ``None`` rather than
    raising. ``/v1/models`` MUST stay 200 even when the audio probe
    is broken.
    """
    if not _audio_routes_mounted():
        return None
    try:
        from ..audio import probe as _audio_probe
    except Exception:  # noqa: BLE001
        return None
    snapshot: dict[str, str] = {}
    for lane in ("stt", "tts"):
        try:
            entry = _audio_probe.audio_lane_status(lane)
        except Exception:  # noqa: BLE001
            continue
        status = entry.get("status") if isinstance(entry, dict) else None
        if status and status != "unknown":
            snapshot[lane] = status
    return snapshot or None


def effective_parsers_for(
    model_id: str, profile_tool_parser: str | None, profile_reasoning_parser: str | None
) -> tuple[str | None, str | None]:
    """Return the EFFECTIVE ``(tool_call_parser, reasoning_parser)`` pair
    for ``model_id`` — i.e. the parsers actually in use by the live
    runtime, not the static alias profile defaults.

    Lookup order (highest precedence first):

    1. **Per-entry live state** — when a ``ModelEntry`` is loaded for
       ``model_id`` (multi-model serve, or single-model serve that
       populated the registry), the entry's ``tool_call_parser`` /
       ``reasoning_parser`` fields hold what the runtime is actually
       using. These already encode both the explicit CLI flag
       (``--tool-call-parser``, ``--reasoning-parser``) and the
       auto-detect outcome — :func:`server.load_model` writes them
       AFTER ``_detect_native_tool_support`` has resolved the final
       values.
    2. **Per-server live state (single-model)** — for single-model
       serves that don't go through the registry, fall back to the
       ``ServerConfig`` globals (which mirror ``server._tool_call_parser``
       and ``server._reasoning_parser_name``). Same gating as
       :func:`_tools_capable`: the server-global parsers ONLY apply
       to the model the server is currently serving, never to
       unrelated registry / discovery ids.
    3. **Alias profile default** — the values declared in
       ``aliases.json`` and surfaced via ``AliasProfile``. Used for
       discovery clients pre-flighting tool-call support on aliases
       they're considering switching to (the runtime isn't actually
       running them yet).
    4. **None** — preserves the existing wire shape for ids with no
       live runtime AND no alias profile entry.

    R12 MED-1 (Vlad + Sven dogfood, rapid-mlx 0.8.15): pre-fix the
    ``/v1/models`` handler read ``profile.tool_call_parser`` /
    ``profile.reasoning_parser`` directly, so a server booted with
    a raw HF id (no alias profile) returned ``null`` for both fields
    even when the runtime was actively using auto-detected or
    CLI-supplied parsers. Agentic SDKs that route on the declared
    parsers (and human operators debugging tool-call issues) saw
    misleading nulls. This helper is the single source of truth for
    "what parsers does the runtime actually have bound for this id".

    Exception-tolerant by design: the route must stay 200 even when
    the registry mutates mid-iteration or the server module fails to
    import (test isolation, partial-init). Any failure collapses to
    the profile default so the listing never regresses below the
    pre-fix shape.
    """
    cfg = get_config()

    def _coerce(value):
        """Return ``value`` only when it's a non-empty string.

        Defensive against test doubles (``MagicMock`` registry entries
        used in ``tests/test_routes.py``) and any future entry shape
        that stores a non-string sentinel. The wire field is
        ``str | None``; anything else is treated as "no value bound"
        and reported as ``null`` on the wire (the entry branch is
        authoritative — non-string entry parser fields collapse to
        ``None`` here rather than falling back to the alias profile
        default; for unrelated ids the registry guard above already
        prevents this branch from running). Keeps the route 200 even
        when an entry has a malformed parser field.
        """
        return value if isinstance(value, str) and value else None

    # Tier 1 — per-entry live state
    entry = None
    try:
        if cfg.model_registry is not None:
            try:
                candidate = cfg.model_registry.get_entry(model_id)
            except KeyError:
                candidate = None
            # ``get_entry`` falls back to the default entry on miss; guard
            # that the entry actually corresponds to ``model_id`` so we
            # never report the default entry's parsers for an unrelated id.
            #
            # Strict identity check on the boolean: ``candidate.matches(...)
            # is True`` (no ``bool()`` wrap). The real
            # :meth:`ModelEntry.matches` already returns the literal
            # ``True`` / ``False``. Codex review (round 2) flagged that
            # wrapping a non-bool sentinel — e.g. a ``MagicMock`` whose
            # default truthiness is ``True`` — in ``bool(...)`` would
            # collapse to ``True`` and let a default-entry leak through
            # the guard. The strict-identity check rejects anything that
            # isn't the literal ``True`` (including ``MagicMock``, ``1``,
            # ``"yes"``, etc.), so test doubles can't shortcut the guard
            # and report the default entry's parsers for an unrelated id.
            if candidate is not None and candidate.matches(model_id) is True:
                entry = candidate
    except Exception:  # noqa: BLE001
        entry = None
    if entry is not None:
        # Per-entry live state is AUTHORITATIVE for the registry case.
        # ``server.load_model`` writes ``tool_call_parser`` /
        # ``reasoning_parser`` onto the ``ModelEntry`` from the resolved
        # globals AFTER ``_detect_native_tool_support`` — so an entry
        # field of ``None`` means the runtime is deliberately running
        # this model WITHOUT that parser (operator passed
        # ``--no-tool-call-parser`` or the auto-detector saw a model
        # without native tool support). Falling back to the alias
        # profile default in that case would lie about the live state
        # and re-introduce the very V-1/S-2 bug this PR was opened
        # to fix — clients would tool-call against a parser that isn't
        # actually bound, and tool-output extraction would silently
        # mis-attribute on the response. Codex review (round 1) flagged
        # this as a blocking regression: when the entry exists, return
        # the entry's fields directly; do NOT use the profile default
        # as a backstop. The profile default only applies to ids with
        # NO live entry / NO server-global binding (Tier 3 below).
        return (
            _coerce(getattr(entry, "tool_call_parser", None)),
            _coerce(getattr(entry, "reasoning_parser", None)),
        )

    # Tier 2 — per-server live state (single-model serve without registry)
    if _is_served_model(model_id):
        live_tool = getattr(cfg, "tool_call_parser", None)
        live_reasoning = getattr(cfg, "reasoning_parser_name", None)
        # ``ServerConfig`` was wired in PR #225; pre-bridge installs may
        # still carry the values only on the legacy server module
        # globals. Mirror :func:`_locked_embedding_id`'s bridge-order
        # fallback so the values surface even before ``_sync_config``
        # has plumbed them.
        if live_tool is None:
            try:
                from ..server import _tool_call_parser as _server_tool_parser

                live_tool = _server_tool_parser
            except Exception:  # noqa: BLE001
                live_tool = None
        if live_reasoning is None:
            try:
                from ..server import _reasoning_parser_name as _server_reasoning_name

                live_reasoning = _server_reasoning_name
            except Exception:  # noqa: BLE001
                live_reasoning = None
        # Server-global live state is AUTHORITATIVE for the single-model
        # serve case — same Tier 1 reasoning. Once ``_is_served_model``
        # has confirmed this id IS the model the server is currently
        # serving, the per-server globals describe its live binding
        # exhaustively — including the case where BOTH sides are
        # ``None`` (operator passed ``--no-tool-call-parser`` and
        # ``--no-reasoning-parser``, or auto-detect found neither).
        # Codex review (rounds 2 + 3) flagged two related leaks:
        #   - r2: ``live_tool or profile_tool_parser`` backfilled a
        #     missing live side from the alias profile (one-sided bind
        #     case) — falsely advertising a parser the runtime is NOT
        #     using.
        #   - r3: gating Tier 2 on ``live_tool or live_reasoning``
        #     fell through to the profile default when BOTH sides were
        #     unbound for a served alias — same lie, just the all-off
        #     case.
        # Fix: return the coerced live fields authoritatively whenever
        # ``_is_served_model`` is True. Never backfill from the profile
        # for an id we're actively serving — the alias profile default
        # only applies to ids with NO live binding at all (Tier 3).
        return (_coerce(live_tool), _coerce(live_reasoning))

    # Tier 3 / 4 — alias profile default (which may itself be None)
    return profile_tool_parser, profile_reasoning_parser


def _build_model_info(model_id: str) -> ModelInfo:
    """Construct a ``ModelInfo`` for ``model_id``, filling vendor
    extension fields from the alias registry when the id resolves.

    ``model_id`` may be a known alias (``qwen3.5-4b-4bit``) or a raw
    HF path (``mlx-community/Qwen3.5-4B-MLX-4bit``); ``resolve_profile``
    handles both. Unknown ids (operator-supplied custom paths, models
    not yet in ``aliases.json``) get the OpenAI baseline shape with
    every extension field at ``None`` — except modality and
    capabilities, which still run through the detectors so an
    unregistered VLM repo id still advertises ``image`` modality
    plus the ``"vision"`` capability tag (F-D01 + F-067 layer fix
    covers raw HF paths too, not just registered aliases).
    """
    profile = resolve_profile(model_id)
    # ``context_window`` is engine-derived (not profile-derived) so it
    # surfaces even for unregistered operator-supplied ids when an
    # engine is loaded for them. Resolution is best-effort: probe
    # failures fall through to ``None`` and the client uses its own
    # per-family fallback. See ``_resolve_context_window`` docstring.
    context_window = _resolve_context_window(model_id)
    # F-K-CAPABILITIES-OMIT-AUDIO: per-lane audio status snapshot, or
    # ``None`` when the deep probe never ran (e.g.
    # ``RAPID_MLX_AUDIO_DEEP_PROBE`` unset). Identical value is
    # attached to every entry — the listing's role is to advertise
    # SERVER-WIDE backend health, not per-model audio capability
    # (which would require a separate dry-run per audio alias).
    audio_lanes = _audio_lane_snapshot()

    # R11-B-F4 (Bo 0.8.12 dogfood): audio aliases get an audio-shaped
    # ModelInfo regardless of whether the registry has a text profile
    # for the id (it never does — audio aliases are NOT in
    # ``model_aliases.aliases.json``). The lookup is via
    # :func:`_resolve_audio_entry` so both the short alias (``kokoro``)
    # and the registered HF id (``mlx-community/Kokoro-82M-bf16``) land
    # on the same shape. Pre-fix both forms came back as
    # ``capabilities=["text"]`` / ``modality=null`` and drop-in OpenAI
    # clients couldn't tell audio aliases apart from text models.
    audio_entry = _resolve_audio_entry(model_id)
    if audio_entry is not None:
        # The per-entry ``audio.<kind>`` capability + ``modality="audio"``
        # is the wire-level distinction. ``audio_lanes`` (server-wide
        # health) is still attached so dashboards see degraded backends
        # for the audio aliases too.
        if audio_entry.type == "tts":
            audio_caps = ["audio.speech"]
        else:
            audio_caps = ["audio.transcription"]
        return ModelInfo(
            id=model_id,
            modality="audio",
            capabilities=audio_caps,
            audio_lanes=audio_lanes,
        )

    locked = _locked_embedding_id()
    if locked is not None and model_id == locked:
        # The embedding entry: ``capabilities=["embedding"]`` plus an
        # explicit ``modality="text"`` (F-D01 cosmetic — pre-fix this
        # came back as ``null`` and clients couldn't tell embedding
        # apart from "unset" on the wire). Pass through the
        # ``context_window`` so embedding entries also carry the
        # engine-advertised cap (PR #808 contract) when an engine is
        # actually loaded for the embedding id.
        if profile is None:
            return ModelInfo(
                id=model_id,
                modality=_reported_modality_for_embedding(locked),
                capabilities=["embedding"],
                context_window=context_window,
                audio_lanes=audio_lanes,
            )
        sampling = (
            dict(profile.recommended_sampling)
            if profile.recommended_sampling is not None
            else None
        )
        eff_tool, eff_reasoning = effective_parsers_for(
            model_id, profile.tool_call_parser, profile.reasoning_parser
        )
        return ModelInfo(
            id=model_id,
            recommended_sampling=sampling,
            is_hybrid=profile.is_hybrid,
            is_moe=profile.is_moe,
            tool_call_parser=eff_tool,
            reasoning_parser=eff_reasoning,
            modality=_reported_modality_for_embedding(locked),
            capabilities=["embedding"],
            context_window=context_window,
            audio_lanes=audio_lanes,
        )

    if profile is None:
        # Preserve the prior unknown-id wire shape (``ModelInfo(id=model_id)``
        # with the schema-default ``modality``) and only override when the
        # multimodal detector matches. Codex round-2 BLOCKING on PR #743
        # flagged that passing ``modality=None`` explicitly is technically
        # the same value today but would silently regress if
        # ``ModelInfo.modality``'s default ever flipped from ``None``.
        # Branch on the detector instead so the default path keeps using
        # the schema default.
        #
        # R12 MED-1 (Vlad + Sven 0.8.15 dogfood): raw HF ids (no alias
        # profile) used to advertise ``tool_call_parser=null`` /
        # ``reasoning_parser=null`` even when the runtime was actively
        # using auto-detected or CLI-supplied parsers. Surface the live
        # values via :func:`effective_parsers_for` so the listing
        # reports what the runtime is actually doing — not what the
        # (missing) alias profile would have said.
        eff_tool, eff_reasoning = effective_parsers_for(model_id, None, None)
        capabilities = _detect_capabilities(
            model_id, profile_modality=None, profile_tool_parser=eff_tool
        )
        try:
            if is_mllm_model(model_id):
                return ModelInfo(
                    id=model_id,
                    modality="image",
                    capabilities=capabilities,
                    tool_call_parser=eff_tool,
                    reasoning_parser=eff_reasoning,
                    context_window=context_window,
                    audio_lanes=audio_lanes,
                )
        except Exception:  # noqa: BLE001
            pass
        return ModelInfo(
            id=model_id,
            capabilities=capabilities,
            tool_call_parser=eff_tool,
            reasoning_parser=eff_reasoning,
            context_window=context_window,
            audio_lanes=audio_lanes,
        )
    # ``recommended_sampling`` lives on the dataclass as a tuple of
    # ``(key, value)`` pairs (frozen-dataclass requirement); convert
    # back to a dict for JSON serialization. ``None`` stays ``None``
    # and serializes as JSON ``null`` on the wire (we deliberately do
    # NOT set ``exclude_none`` on ``ModelInfo`` so the shape is
    # predictable for clients; see the ``ModelInfo`` docstring).
    sampling = (
        dict(profile.recommended_sampling)
        if profile.recommended_sampling is not None
        else None
    )
    # R12 MED-1: the EFFECTIVE parsers may override the alias profile
    # defaults when the operator passed ``--tool-call-parser`` /
    # ``--reasoning-parser`` on the CLI, or when auto-detect chose a
    # different parser than the alias default. Surface the live values
    # so ``/v1/models`` never lies about what the runtime is doing.
    # When no live runtime is bound for ``model_id`` (discovery listing
    # of an alias the server isn't currently running), the helper falls
    # back to the profile default — pre-fix shape is preserved.
    eff_tool, eff_reasoning = effective_parsers_for(
        model_id, profile.tool_call_parser, profile.reasoning_parser
    )
    capabilities = _detect_capabilities(
        model_id,
        profile_modality=profile.modality,
        profile_tool_parser=eff_tool,
    )
    return ModelInfo(
        id=model_id,
        recommended_sampling=sampling,
        is_hybrid=profile.is_hybrid,
        is_moe=profile.is_moe,
        tool_call_parser=eff_tool,
        reasoning_parser=eff_reasoning,
        modality=_reported_modality(model_id, profile.modality),
        capabilities=capabilities,
        context_window=context_window,
        audio_lanes=audio_lanes,
    )


@router.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models (supports multi-model).

    Each entry carries the Rapid-MLX vendor extension fields when
    its id resolves to a known alias. OpenAI-spec clients ignore
    unknown fields, so the wire shape stays backward-compatible.
    """
    cfg = get_config()

    models = []
    seen_ids: set[str] = set()

    def _append(info: ModelInfo) -> None:
        if info.id in seen_ids:
            return
        seen_ids.add(info.id)
        models.append(info)

    if cfg.model_registry:
        for entry in cfg.model_registry.list_entries():
            _append(_build_model_info(entry.model_name))
            for alias in sorted(entry.aliases):
                if alias != entry.model_name:
                    _append(_build_model_info(alias))
    elif cfg.model_name:
        _append(_build_model_info(cfg.model_name))
        if cfg.model_alias and cfg.model_alias != cfg.model_name:
            _append(_build_model_info(cfg.model_alias))

    # Surface the dedicated embedding model id (when configured) so
    # clients discover the ``/v1/embeddings``-capable id from the same
    # ``/v1/models`` listing. H-09 sub-fix: when no embedding model is
    # configured the route guard already 400s on ``/v1/embeddings``,
    # so nothing is added here — capability advertisement matches
    # actual behavior.
    locked = _locked_embedding_id()
    if locked:
        _append(_build_model_info(locked))

    return ModelsResponse(data=models)


@router.get("/v1/models/{model_id:path}", dependencies=[Depends(verify_api_key)])
async def retrieve_model(model_id: str) -> ModelInfo:
    """Retrieve a specific model by ID.

    Same vendor-extension shape as `/v1/models` for callers that
    only want the profile for the active alias (rapid-desktop's
    SamplingConfig-bootstrap path).

    Uses Starlette's ``:path`` converter so HF-style ids containing
    ``/`` (e.g. ``mlx-community/all-MiniLM-L6-v2-4bit``) match the
    route without forcing clients to URL-encode the slash — every
    other rapid-mlx endpoint accepts the bare HF id, this one
    should too. Slashes in alias ids are still safe: the lookup is
    a string-equality match against the registry / cfg, not a path
    parse.
    """
    cfg = get_config()

    if cfg.model_registry and model_id in cfg.model_registry:
        return _build_model_info(model_id)
    if model_id in (cfg.model_name, cfg.model_alias):
        return _build_model_info(model_id)
    # The dedicated embedding model id is addressable too so callers
    # can hydrate per-model state from ``/v1/models/{id}`` without
    # extra wire heuristics.
    locked = _locked_embedding_id()
    if locked and model_id == locked:
        return _build_model_info(model_id)
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
