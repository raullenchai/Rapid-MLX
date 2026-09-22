# SPDX-License-Identifier: Apache-2.0
"""Best-effort telemetry-v2 inference and capability emitters."""

from __future__ import annotations

from rapid_mlx.telemetry import emit, model_id, redact, store
from rapid_mlx.telemetry import track as track_module

_MODEL_TYPES = frozenset(
    {
        "llm",
        "vlm",
        "embedding",
        "image-gen",
        "video-gen",
        "text-diffusion",
        "audio",
        "other",
    }
)


def model_type_token(source: object | None) -> str:
    """Classify a resolved engine/profile into the registry vocabulary."""
    try:
        if source is None:
            return "other"
        if isinstance(source, str):
            return source if source in _MODEL_TYPES else "other"

        modality = getattr(source, "modality", None)
        if modality == "text":
            return "vlm" if getattr(source, "supports_image_input", False) else "llm"
        if isinstance(modality, str) and modality in _MODEL_TYPES - {
            "llm",
            "vlm",
            "other",
        }:
            return modality
        if getattr(source, "is_image_gen", False):
            return "image-gen"
        if getattr(source, "is_video_gen", False):
            return "video-gen"
        if getattr(source, "is_embedding", False):
            return "embedding"
        if getattr(source, "is_audio", False):
            return "audio"
        return "vlm" if getattr(source, "is_mllm", False) else "llm"
    except Exception:
        return "other"


def emit_completed_request(
    *,
    model: str,
    endpoint: str,
    caller_agent: str | None,
    caller_client: str | None,
    result: str,
) -> None:
    """Record one completed request and emit only a crossed milestone.

    Callers pass the resolved telemetry model id, never the request's model
    field. Every operation is best effort because this runs beside the
    response-finalization telemetry on the generation path.
    """
    try:
        safe_model = model_id.telemetry_model_id(model)
        safe_endpoint = emit._normalize_endpoint(endpoint)
        caller = redact.normalize_caller_agent(caller_agent, caller_client)
        outcome = result if result in ("ok", "failed") else "failed"
        crossing = store.record(f"inf|{safe_model}|{safe_endpoint}|{caller}|{outcome}")
        if crossing is not None:
            track_module.track(
                "inference_bucket_reached",
                {
                    "model": safe_model,
                    "endpoint": safe_endpoint,
                    "caller": caller,
                    "result": outcome,
                    "count_bucket": crossing.bucket,
                    "bucket_source": crossing.bucket_source,
                },
            )
        if outcome == "ok":
            track_module.emit_active_day()
    except Exception:
        return


def emit_capability_rejected(capability: str, *, model_type: str = "other") -> None:
    """Emit one closed-vocabulary capability rejection without raising."""
    try:
        track_module.track(
            "capability_rejected",
            {
                "capability": capability,
                "model_type": model_type_token(model_type),
            },
        )
    except Exception:
        return
