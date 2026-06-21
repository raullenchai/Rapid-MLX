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


def _embedding_capabilities(model_id: str) -> list[str]:
    """Return capability tags for ``model_id``.

    H-09 sub-fix: only the explicitly-configured embedding model is
    tagged ``"embedding"``. When the server boots without
    ``--embedding-model``, the route guard rejects ``/v1/embeddings``
    requests with a 400, so advertising any chat-model id as
    embedding-capable would be lying to the client.

    Reads from ``ServerConfig.embedding_model_locked`` first and falls
    back to the ``server._embedding_model_locked`` global so the
    capability shows up even before ``_sync_config`` has bridged the
    value (mirrors the same bridge the embeddings route uses).
    """
    cfg = get_config()
    locked = cfg.embedding_model_locked
    if locked is None:
        try:
            from ..server import _embedding_model_locked as _server_locked

            locked = _server_locked
        except Exception:  # noqa: BLE001
            locked = None
    if locked is not None and model_id == locked:
        return ["embedding"]
    return []


def _build_model_info(model_id: str) -> ModelInfo:
    """Construct a ``ModelInfo`` for ``model_id``, filling vendor
    extension fields from the alias registry when the id resolves.

    ``model_id`` may be a known alias (``qwen3.5-4b-4bit``) or a raw
    HF path (``mlx-community/Qwen3.5-4B-MLX-4bit``); ``resolve_profile``
    handles both. Unknown ids (operator-supplied custom paths, models
    not yet in ``aliases.json``) get the OpenAI baseline shape with
    every extension field at ``None`` — except modality, which still
    runs through the multimodal detector so an unregistered VLM repo
    id still advertises ``image`` (F-067 layer fix covers raw HF paths
    too, not just registered aliases).
    """
    capabilities = _embedding_capabilities(model_id)
    profile = resolve_profile(model_id)
    # ``context_window`` is engine-derived (not profile-derived) so it
    # surfaces even for unregistered operator-supplied ids when an
    # engine is loaded for them. Resolution is best-effort: probe
    # failures fall through to ``None`` and the client uses its own
    # per-family fallback. See ``_resolve_context_window`` docstring.
    context_window = _resolve_context_window(model_id)
    if profile is None:
        # Preserve the prior unknown-id wire shape (``ModelInfo(id=model_id)``
        # with the schema-default ``modality``) and only override when the
        # multimodal detector matches. Codex round-2 BLOCKING on PR #743
        # flagged that passing ``modality=None`` explicitly is technically
        # the same value today but would silently regress if
        # ``ModelInfo.modality``'s default ever flipped from ``None``.
        # Branch on the detector instead so the default path keeps using
        # the schema default.
        try:
            if is_mllm_model(model_id):
                return ModelInfo(
                    id=model_id,
                    modality="image",
                    capabilities=capabilities,
                    context_window=context_window,
                )
        except Exception:  # noqa: BLE001
            pass
        return ModelInfo(
            id=model_id,
            capabilities=capabilities,
            context_window=context_window,
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
    return ModelInfo(
        id=model_id,
        recommended_sampling=sampling,
        is_hybrid=profile.is_hybrid,
        is_moe=profile.is_moe,
        tool_call_parser=profile.tool_call_parser,
        reasoning_parser=profile.reasoning_parser,
        modality=_reported_modality(model_id, profile.modality),
        capabilities=capabilities,
        context_window=context_window,
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
    locked = cfg.embedding_model_locked
    if locked is None:
        try:
            from ..server import _embedding_model_locked as _server_locked

            locked = _server_locked
        except Exception:  # noqa: BLE001
            locked = None
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
    locked = cfg.embedding_model_locked
    if locked is None:
        try:
            from ..server import _embedding_model_locked as _server_locked

            locked = _server_locked
        except Exception:  # noqa: BLE001
            locked = None
    if locked and model_id == locked:
        return _build_model_info(model_id)
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
