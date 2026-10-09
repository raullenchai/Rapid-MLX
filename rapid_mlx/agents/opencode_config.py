"""Keep the OpenCode provider usable when switching between major versions."""

from __future__ import annotations


def reconcile(config: dict, version: str | None) -> dict:
    """Reconcile Rapid-MLX's two provider shapes without dropping user config."""
    if version and version.startswith("1."):
        native = config.get("providers")
        if native is None:
            return config
        if not isinstance(native, dict) or any(key != "rapid-mlx" for key in native):
            raise ValueError(
                "OpenCode 1.x cannot read the remaining 2.x `providers` entries. "
                "Migrate or remove them manually, then re-run --setup."
            )
        del config["providers"]
        return config

    legacy = config.get("provider")
    native = config.get("providers")
    if not isinstance(legacy, dict) or not isinstance(native, dict):
        return config
    source = legacy.get("rapid-mlx")
    target = native.get("rapid-mlx")
    if not isinstance(source, dict) or not isinstance(target, dict):
        return config

    options = source.get("options")
    settings = target.get("settings")
    if isinstance(options, dict) and isinstance(settings, dict):
        for key, value in options.items():
            settings.setdefault(key, value)

    old_models = source.get("models")
    new_models = target.get("models")
    if isinstance(old_models, dict) and isinstance(new_models, dict):
        for model_id, old_model in old_models.items():
            if model_id not in new_models:
                new_models[model_id] = _migrate_model(model_id, old_model)
    return config


def _migrate_model(model_id: str, old_model: object) -> dict:
    if not isinstance(old_model, dict):
        raise ValueError(
            f"Cannot migrate legacy OpenCode model {model_id!r}: invalid entry"
        )
    supported = {
        "id",
        "name",
        "family",
        "limit",
        "tool_call",
        "modalities",
        "options",
        "headers",
        "status",
    }
    unsupported = old_model.keys() - supported
    if unsupported or old_model.get("status") not in (None, "deprecated"):
        raise ValueError(
            f"Cannot migrate legacy OpenCode model {model_id!r} automatically; "
            "convert its custom fields to the 2.x schema, then re-run --setup."
        )
    result = {
        key: old_model[key]
        for key in ("name", "family", "limit", "headers")
        if key in old_model
    }
    if "id" in old_model:
        result["modelID"] = old_model["id"]
    if "options" in old_model:
        result["settings"] = old_model["options"]
    if old_model.get("status") == "deprecated":
        result["disabled"] = True
    capabilities = {}
    if "tool_call" in old_model:
        capabilities["tools"] = old_model["tool_call"]
    modalities = old_model.get("modalities")
    if isinstance(modalities, dict):
        for key in ("input", "output"):
            if key in modalities:
                capabilities[key] = modalities[key]
    if capabilities:
        result["capabilities"] = capabilities
    return result
