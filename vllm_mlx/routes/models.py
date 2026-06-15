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

from fastapi import APIRouter, Depends, HTTPException

from ..api.models import ModelInfo, ModelsResponse
from ..config import get_config
from ..middleware.auth import verify_api_key
from ..model_aliases import resolve_profile

router = APIRouter()


def _build_model_info(model_id: str) -> ModelInfo:
    """Construct a ``ModelInfo`` for ``model_id``, filling vendor
    extension fields from the alias registry when the id resolves.

    ``model_id`` may be a known alias (``qwen3.5-4b-4bit``) or a raw
    HF path (``mlx-community/Qwen3.5-4B-MLX-4bit``); ``resolve_profile``
    handles both. Unknown ids (operator-supplied custom paths, models
    not yet in ``aliases.json``) get the OpenAI baseline shape with
    every extension field at ``None``.
    """
    profile = resolve_profile(model_id)
    if profile is None:
        return ModelInfo(id=model_id)
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
        modality=profile.modality,
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
    if cfg.model_registry:
        for entry in cfg.model_registry.list_entries():
            models.append(_build_model_info(entry.model_name))
            for alias in sorted(entry.aliases):
                if alias != entry.model_name:
                    models.append(_build_model_info(alias))
    elif cfg.model_name:
        models.append(_build_model_info(cfg.model_name))
        if cfg.model_alias and cfg.model_alias != cfg.model_name:
            models.append(_build_model_info(cfg.model_alias))
    return ModelsResponse(data=models)


@router.get("/v1/models/{model_id}", dependencies=[Depends(verify_api_key)])
async def retrieve_model(model_id: str) -> ModelInfo:
    """Retrieve a specific model by ID.

    Same vendor-extension shape as `/v1/models` for callers that
    only want the profile for the active alias (rapid-desktop's
    SamplingConfig-bootstrap path).
    """
    cfg = get_config()

    if cfg.model_registry and model_id in cfg.model_registry:
        return _build_model_info(model_id)
    if model_id in (cfg.model_name, cfg.model_alias):
        return _build_model_info(model_id)
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
