# SPDX-License-Identifier: Apache-2.0
"""Machine-facing adapter for the existing metadata and catalog fit checks."""

from __future__ import annotations

import json
from typing import Any

from rapid_mlx.byom import preflight as pf
from rapid_mlx.model_reference import (
    InvalidModelReferenceError,
    normalize_model_reference,
    reference_form,
)


def _cached_inspection(repo: str) -> pf.Inspection | None:
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache

    from rapid_mlx.model_metadata import (
        resolve_offline_cached_snapshot,
        resolve_unreferenced_cached_snapshot,
    )

    path = try_to_load_from_cache(repo, "config.json")
    snapshot = (
        Path(path).parent
        if isinstance(path, str)
        else resolve_offline_cached_snapshot(repo)
        or resolve_unreferenced_cached_snapshot(repo)
    )
    return pf.inspect_local(str(snapshot)) if snapshot is not None else None


def _format(inspection: pf.Inspection, verdict: pf.Verdict) -> str | None:
    if verdict.format_label == "GGUF":
        return "gguf"
    if verdict.format_label == "PyTorch .bin":
        return "pytorch"
    # Config-only (unfinished) repos have quantization metadata but no weights.
    if not any(name.endswith(".safetensors") for name in inspection.files):
        return None
    bits = pf._quant_bits(inspection.config)
    if bits is not None:
        return f"mlx-{bits}bit"
    return inspection.dtype or "safetensors"


def _check_disk(result: dict[str, Any]) -> dict[str, Any]:
    from rapid_mlx.cli import _download_disk_space_status

    if result["cached"]:
        result["disk_fits"] = True
        return result
    fits, free_bytes, message = _download_disk_space_status(
        result["download_bytes"] or 0
    )
    result.update(disk_fits=fits, free_disk_bytes=free_bytes)
    if fits is False:
        if result["verdict"] == "ok":
            result.update(verdict="too_big", reason="insufficient_disk")
            result["message"] = message
        else:
            result["message"] = f"{result['message']}\n{message}"
    return result


def check_reference(reference: str) -> dict[str, Any]:
    """Return schema v1; metadata failures are completed, conservative checks."""
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    from rapid_mlx.cli import _cache_runnability
    from rapid_mlx.model_aliases import (
        catalog_alias_for,
        resolve_hub_model,
        resolve_profile,
    )
    from rapid_mlx.model_metadata import hub_offline_mode_active
    from rapid_mlx.model_sizes import size_bytes

    result: dict[str, Any] = {
        "schema": 1,
        "reference": reference,
        "repo": None,
        "alias": None,
        "in_catalog": False,
        "verdict": "unknown",
        "reason": None,
        "message": "Model compatibility could not be verified.",
        "download_bytes": None,
        "format": None,
        "capabilities": [],
        "required_memory_bytes": None,
        "fits": None,
        "cached": False,
        "disk_fits": None,
        "free_disk_bytes": None,
    }
    try:
        normalized = normalize_model_reference(reference)
        repo = resolve_hub_model(normalized)
        if "/" not in repo:
            raise HFValidationError("Expected an alias or org/name.")
        validate_repo_id(repo)
    except (ValueError, HFValidationError) as exc:
        result.update(
            verdict="invalid_reference",
            reason=exc.reason
            if isinstance(exc, InvalidModelReferenceError)
            else "invalid_model_ref",
            message=str(exc)
            if isinstance(exc, InvalidModelReferenceError)
            else "Enter a model alias, org/name, or Hugging Face model URL.",
        )
        return result
    result["repo"] = repo
    from rapid_mlx.audio.registry import resolve_audio_alias

    audio = resolve_audio_alias(repo)
    alias = catalog_alias_for(normalized) or catalog_alias_for(repo)
    if alias is None and audio is not None:
        alias = audio.alias
    result.update(alias=alias, in_catalog=alias is not None)
    cached = _cache_runnability(repo) is True
    result["cached"] = cached
    profile = resolve_profile(alias or repo)
    if profile is not None:
        fit = pf.catalog_memory_fit(
            alias or repo,
            ram_bytes=pf.physical_ram_bytes(),
            recommended_working_set_bytes=pf.mlx_recommended_working_set_bytes(),
        )
        caps: list[str] = ["chat"] if profile.modality == "text" else [profile.modality]
        if profile.supports_image_input:
            caps.append("vision")
        if profile.tool_call_parser:
            caps.append("tools")
        from rapid_mlx.telemetry.quant import quant_token

        quant = quant_token(repo)
        result.update(
            download_bytes=size_bytes(repo),
            required_memory_bytes=fit.estimate_bytes,
            fits=fit.fits,
            capabilities=caps,
            format=f"mlx-{quant}" if quant not in {"unknown", "other"} else None,
            verdict="too_big" if fit.fits is False else "ok",
            reason=pf.INSUFFICIENT_MEMORY if fit.fits is False else None,
            message="This model does not fit this Mac's memory."
            if fit.fits is False
            else "This catalog model is supported.",
        )
        return _check_disk(result)
    if audio is not None:
        result.update(
            download_bytes=size_bytes(repo),
            capabilities=[audio.type],
            verdict="ok",
            message="This catalog model is supported.",
        )
        return _check_disk(result)
    if cached:
        inspection = _cached_inspection(repo)
    elif hub_offline_mode_active():
        result.update(
            reason="offline",
            message="The Hub is offline; model compatibility could not be verified.",
        )
        return result
    else:
        try:
            inspection = pf.inspect_hub(repo, raise_errors=True)
        except Exception as exc:
            from rapid_mlx.telemetry.model_events import pull_error_class

            reason = pull_error_class(exc)
            if reason in {"gated", "not_found"}:
                from rapid_mlx.telemetry.model_id import _revoke_proof

                _revoke_proof(repo)
            reason = (
                "offline"
                if reason == "network" or isinstance(exc, TimeoutError)
                else reason
            )
            result.update(
                reason=reason,
                message={
                    "not_found": "The model repository was not found.",
                    "gated": "Access to this model requires Hugging Face authorization.",
                    "offline": "The Hub is unavailable; model compatibility could not be verified.",
                }.get(reason, "Model metadata could not be read."),
            )
            return result
    if inspection is None:
        return result
    inspection, verdict = pf.evaluate_inspection(inspection, refuse_oversize=True)
    if verdict is None:
        return result
    result.update(
        download_bytes=inspection.download_bytes or inspection.weight_bytes or None,
        format=_format(inspection, verdict),
        required_memory_bytes=verdict.est_memory_bytes,
        fits={"yes": True, "tight": True, "no": False}.get(verdict.fit),
    )
    if inspection.has_chat_template is True:
        result["capabilities"].append("chat")
    if inspection.config.get("vision_config"):
        result["capabilities"].append("vision")
    if verdict.failure:
        result.update(
            verdict="too_big"
            if verdict.failure == pf.INSUFFICIENT_MEMORY
            else "unsupported",
            reason=verdict.failure,
            message={
                pf.UNSUPPORTED_FORMAT: (
                    "This repo only has GGUF files, which Rapid-MLX can't run."
                    if verdict.format_label == "GGUF"
                    else "This repo only has PyTorch .bin weights, which Rapid-MLX can't run."
                    if verdict.format_label == "PyTorch .bin"
                    else "This model's weight format is not supported."
                ),
                pf.UNSUPPORTED_ARCHITECTURE: "This model's architecture is not supported.",
                pf.INSUFFICIENT_MEMORY: "This model does not fit this Mac's memory.",
            }[verdict.failure],
        )
    elif verdict.architecture_supported is True and result["format"] is not None:
        result.update(verdict="ok", message="This model is supported.")
    return _check_disk(result)


def check_command(args: Any) -> None:
    """Print one verdict and emit one consent-gated check event."""
    import sys
    from contextlib import redirect_stdout

    from rapid_mlx.telemetry.model_events import emit_model_ref_checked

    # Optional dependencies may log diagnostics; reserve stdout for the payload.
    with redirect_stdout(sys.stderr):
        result = check_reference(args.model)
        emit_model_ref_checked(result, ref_form=reference_form(args.model))
    print(json.dumps(result) if getattr(args, "json", False) else result["message"])
    if result["disk_fits"] is False:
        sys.exit(1)
