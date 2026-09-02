# SPDX-License-Identifier: Apache-2.0
"""Read-only adapters from current alias/recommendation files to atomic records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from .canonical import rcj_digest
from .validation import ContractValidator


def _registry_model_id(repo_id: str, subfolder: str | None) -> str:
    source_key = f"{repo_id}\0{subfolder or ''}".encode()
    return f"legacy/hf/{hashlib.sha256(source_key).hexdigest()[:24]}"


def _availability(*, desktop: bool = True) -> dict[str, bool]:
    return {"cli": True, "server": True, "desktop": desktop, "website": True}


def _image_operations(repo_id: str) -> list[str]:
    folded = repo_id.casefold().replace("_", "-")
    if "flux2" in folded or "flux.2" in folded or "klein" in folded:
        return ["text_to_image", "image_to_image"]
    if "qwen-image-edit" in folded:
        return ["image_to_image"]
    return ["text_to_image"]


def _main_capabilities(profile: Any) -> dict[str, Any]:
    modality = getattr(profile, "modality", "text") or "text"
    if modality == "image-gen":
        tasks, operations, adapter = (
            ["image_generation"],
            _image_operations(profile.hf_path),
            "mflux",
        )
    elif modality == "video-gen":
        tasks = ["video_generation"]
        operations = [
            mode.replace("-", "_")
            for mode in (profile.video_modes or ("text-to-video",))
        ]
        adapter = "rapid_mlx/video"
    elif modality == "vision":
        tasks, operations, adapter = (
            ["vision_language"],
            ["chat", "image_understanding"],
            "mlx_vlm",
        )
    else:
        tasks, operations = ["text_generation"], ["chat"]
        if bool(getattr(profile, "supports_image_input", False)):
            tasks.append("vision_language")
            operations.append("image_understanding")
        adapter = (
            "rapid_mlx/text_diffusion" if modality == "text-diffusion" else "mlx_lm"
        )
    capabilities: dict[str, Any] = {
        "task_types": tasks,
        "is_text_only": bool(getattr(profile, "is_text_only", False)),
        "operation_modes": operations,
        "runtime_adapter": adapter,
        "experimental": bool(getattr(profile, "experimental", False)),
    }
    for field in ("tool_call_parser", "reasoning_parser", "chat_template_id"):
        value = getattr(profile, field, None)
        if value:
            capabilities[field] = value
    return capabilities


def _audio_capabilities(entry: Any) -> dict[str, Any]:
    alias = entry.alias.casefold()
    if entry.type == "stt":
        if entry.family == "qwen3_aligner":
            # Alignment is an operation of the speech-recognition pipeline,
            # not a distinct model/runtime identity kind. Keeping the atomic
            # task broad lets one recognizer expose transcription and/or
            # alignment without inventing an unreachable ExecutionConfig.
            tasks, operations = ["speech_recognition"], ["forced_alignment"]
        else:
            tasks, operations = ["speech_recognition"], ["transcription"]
            if entry.family == "whisper":
                operations.append("translation")
    else:
        tasks = ["speech_synthesis"]
        if "voicedesign" in alias:
            operations = ["voice_design"]
        elif entry.family in {"indextts", "f5"} or alias == "qwen3-tts-clone":
            operations = ["voice_cloning"]
        else:
            operations = ["preset_voice"]
    return {
        "task_types": tasks,
        "is_text_only": False,
        "operation_modes": operations,
        "runtime_adapter": f"mlx_audio/{entry.family}",
        "experimental": entry.family in {"voxcpm", "dia"},
    }


def _alias_record(
    alias: str,
    model_id: str,
    capabilities: dict[str, Any],
    *,
    origin: str,
    desktop: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "alias": alias,
        "origin": origin,
        "target": {"registry_model_id": model_id, "resolution_status": "unresolved"},
        "capabilities": capabilities,
        "availability": _availability(desktop=desktop),
        "default_execution_preset_id": None,
        "execution_presets": [],
    }


def build_legacy_catalog_snapshot() -> dict[str, Any]:
    """Project both legacy alias registries into one deterministic snapshot."""

    from vllm_mlx.audio.registry import list_audio_aliases
    from vllm_mlx.model_aliases import list_builtin_aliases, list_profiles
    from vllm_mlx.model_sizes import size_bytes

    profiles = list_profiles()
    builtin = set(list_builtin_aliases())
    models: dict[str, dict[str, Any]] = {}
    aliases: list[dict[str, Any]] = []

    # `models --json` exposes the complete profile view, including user aliases.
    # Project that same surface rather than truncating the shadow graph to the
    # built-in catalog.
    for alias in sorted(profiles):
        profile = profiles[alias]
        model_id = _registry_model_id(profile.hf_path, profile.subfolder)
        model = {
            "schema_version": 1,
            "registry_model_id": model_id,
            "source": {
                "provider": "huggingface",
                "repo_id": profile.hf_path,
                **({"subfolder": profile.subfolder} if profile.subfolder else {}),
            },
            "resolution_status": "unresolved",
        }
        estimated_size = size_bytes(profile.hf_path)
        if isinstance(estimated_size, int) and estimated_size > 0:
            model["estimated_download_size_bytes"] = estimated_size
        models.setdefault(model_id, model)
        aliases.append(
            _alias_record(
                alias,
                model_id,
                _main_capabilities(profile),
                origin="builtin" if alias in builtin else "user",
            )
        )

    for entry in list_audio_aliases():
        model_id = _registry_model_id(entry.hf_id, None)
        model = {
            "schema_version": 1,
            "registry_model_id": model_id,
            "source": {"provider": "huggingface", "repo_id": entry.hf_id},
            "resolution_status": "unresolved",
        }
        estimated_size = size_bytes(entry.hf_id)
        if isinstance(estimated_size, int) and estimated_size > 0:
            model["estimated_download_size_bytes"] = estimated_size
        models.setdefault(model_id, model)
        aliases.append(
            _alias_record(
                entry.alias,
                model_id,
                _audio_capabilities(entry),
                origin="builtin",
                desktop=entry.alias != "whisper-tiny",
            )
        )

    snapshot: dict[str, Any] = {
        "schema_version": 2,
        "models": sorted(models.values(), key=lambda item: item["registry_model_id"]),
        "aliases": sorted(aliases, key=lambda item: item["alias"]),
        "recommendation_policy_digests": [],
    }
    snapshot["catalog_digest"] = rcj_digest(snapshot)
    ContractValidator().validate_catalog_snapshot(snapshot)
    return snapshot


def load_product_recommendation_policy(
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the atomic RAM policy and validate it against the current catalog."""

    snapshot = snapshot or build_legacy_catalog_snapshot()
    path = Path(__file__).resolve().parents[1] / "model_recommendations.json"
    policy = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    aliases = {item["alias"]: item for item in snapshot["aliases"]}
    ContractValidator().validate_recommendation_policy(policy, aliases=aliases)
    return policy


def build_legacy_recommendation_policy(
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility name for callers introduced during the shadow phase."""

    return load_product_recommendation_policy(snapshot)


def build_shadow_report(
    snapshot: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic migration coverage; never changes resolution behavior."""

    from vllm_mlx.audio.registry import list_audio_aliases
    from vllm_mlx.model_aliases import list_profiles

    snapshot = snapshot or build_legacy_catalog_snapshot()
    policy = policy or load_product_recommendation_policy(snapshot)
    legacy_aliases = set(list_profiles()) | {
        entry.alias for entry in list_audio_aliases()
    }
    projected_aliases = {entry["alias"] for entry in snapshot["aliases"]}
    task_counts: dict[str, int] = {}
    for alias in snapshot["aliases"]:
        for task in alias["capabilities"]["task_types"]:
            task_counts[task] = task_counts.get(task, 0) + 1
    failures = []
    if legacy_aliases != projected_aliases:
        failures.append("alias_set_mismatch")
    policy_aliases = {
        pick["alias"] for tier in policy["tiers"] for pick in tier["picks"]
    }
    if not policy_aliases <= projected_aliases:
        failures.append("recommendation_alias_missing")
    return {
        "schema_version": 1,
        "mode": "shadow",
        "equivalent": not failures,
        "failures": failures,
        "legacy_alias_count": len(legacy_aliases),
        "projected_alias_count": len(projected_aliases),
        "registry_model_count": len(snapshot["models"]),
        "task_counts": {key: task_counts[key] for key in sorted(task_counts)},
        "catalog_digest": snapshot["catalog_digest"],
        "recommendation_policy_digest": policy["policy_digest"],
    }


def build_catalog_bundle() -> dict[str, Any]:
    """Build the complete shadow bundle exposed to Server and Desktop."""

    snapshot = build_legacy_catalog_snapshot()
    policy = load_product_recommendation_policy(snapshot)
    snapshot["recommendation_policy_digests"] = [policy["policy_digest"]]
    snapshot["catalog_digest"] = rcj_digest(
        {
            key: snapshot[key]
            for key in (
                "schema_version",
                "models",
                "aliases",
                "recommendation_policy_digests",
            )
        }
    )
    ContractValidator().validate_catalog_snapshot(snapshot)
    return {
        "snapshot": snapshot,
        "recommendation_policies": [policy],
        "shadow_report": build_shadow_report(snapshot, policy),
    }
