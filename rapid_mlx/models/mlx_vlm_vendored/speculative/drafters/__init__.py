import importlib
import importlib.machinery
import json
import logging
import sys
from types import ModuleType
from typing import Any, Optional, Tuple

from .dflash2 import DFlash2DraftModel
from mlx_vlm.speculative.drafters.dspark import DSparkDraftModel
from mlx_vlm.speculative.drafters.laguna_dflash import LagunaDFlashDraftModel
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    MuseGlimmerAssistantDraftModel,
)
from .qwen3_dflash import DFlashDraftModel

KNOWN_DRAFTER_KINDS = {"dflash", "mtp", "eagle3"}

# Drafter HF ``model_type`` → required round-loop kind. Anything not listed
# here falls back to ``DEFAULT_DRAFTER_KIND`` when the caller didn't pass one.
DRAFTER_KIND_BY_MODEL_TYPE = {
    "deepseek_v4_mtp": "mtp",
    "deepseek_v4_dspark": "dflash",
    "dspark": "dflash",
    "gemma4_dspark": "dflash",
    "eagle3": "eagle3",
    "gemma4_assistant": "mtp",
    "gemma4_unified_assistant": "mtp",
    "glm4_moe_lite_mtp": "mtp",
    "glm5_next_mtp": "mtp",
    "glm_moe_dsa_mtp": "mtp",
    "hy_v4_mtp": "mtp",
    "inkling_mtp": "mtp",
    "qwen3_5_mtp": "mtp",
    "qwen4_exp_mtp": "mtp",
    "laguna": "dflash",
    "muse_glimmer_assistant": "dflash",
    "qwen3_dspark": "dflash",
    # Rapid upstream-bugfix (documented deviation): pinned 0.7.1 omits the
    # served DFlash families' model types, so an explicit wrong --draft-kind
    # (e.g. "mtp") dispatched them through the wrong round loop instead of
    # being overridden here.
    "dflash2": "dflash",
    "qwen3_dflash": "dflash",
}

DEFAULT_DRAFTER_KIND = "dflash"

# Rapid binding hook (documented deviation): the served drafter families'
# checkpoint ``model_type`` values. pinned ``load_model`` resolves sidecar
# architectures through ``mlx_vlm.models.<model_type>``; pre-registering a
# package-compatible ``sys.modules`` shim that exposes the vendored
# package's ``Model``/``ModelConfig`` (preserving any existing exports,
# ``__path__`` and ``__spec__``) makes the pinned loader construct the
# vendored classes, so the documented runtime fixes reach production
# drafters. Bindings install lazily, one family per load, and existing
# entries are re-bound when they do not match the vendored classes — a
# pinned module imported earlier, or a shim bound before the GLM
# compatibility swap, must not silently serve a stale implementation.
# Unvendored families fall through to the pinned modules.
_SERVED_ARCHITECTURE_FAMILIES = (
    "glm5_next_mtp",
    "qwen3_5_mtp",
    "qwen3_dflash",
    "dflash2",
)


def install_served_architecture_bindings(model_type: Optional[str] = None) -> None:
    # Bind one served family's architecture module (lazily, per load).
    if model_type not in _SERVED_ARCHITECTURE_FAMILIES:
        return
    target = f"mlx_vlm.models.{model_type}"
    package = importlib.import_module(f"{__name__}.{model_type}")
    existing = sys.modules.get(target)
    if (
        existing is not None
        and getattr(existing, "Model", None) is package.Model
        and getattr(existing, "ModelConfig", None) is package.ModelConfig
    ):
        return
    # Package-compatible shim: preserve an existing canonical module's
    # exports (including ``__path__``/``__spec__``) so submodule imports
    # keep working; only ``Model``/``ModelConfig`` are overridden.
    shim = ModuleType(target)
    if existing is not None:
        setattr(shim, "__path__", getattr(existing, "__path__", []))
        for name, value in vars(existing).items():
            if name not in ("Model", "ModelConfig"):
                setattr(shim, name, value)  # noqa: B010
    else:
        setattr(shim, "__path__", [])
    setattr(
        shim,
        "__spec__",
        getattr(existing, "__spec__", None)
        or importlib.machinery.ModuleSpec(target, loader=None, is_package=True),
    )
    setattr(shim, "Model", package.Model)  # noqa: B010
    setattr(shim, "ModelConfig", package.ModelConfig)  # noqa: B010
    sys.modules[target] = shim
    # a previously imported pinned child leaves a stale attribute on
    # the parent package; ``from mlx_vlm.models import <model_type>``
    # resolves through that attribute, so it must be updated too.
    parent = sys.modules.get("mlx_vlm.models")
    if parent is not None:
        setattr(parent, model_type, shim)


logger = logging.getLogger(__name__)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _hidden_size(config: Any) -> Any:
    return _cfg_get(_cfg_get(config, "text_config", config), "hidden_size")


def _noop_target_compatibility(target_model: Any) -> None:
    del target_model


def _validate_model_specific_compatibility(target_model: Any, draft_model: Any) -> None:
    validator = getattr(
        draft_model,
        "validate_target_compatibility",
        _noop_target_compatibility,
    )
    validator(target_model)


def validate_drafter_compatibility(
    target_model: Any,
    draft_model: Any,
    draft_kind: Optional[str],
) -> None:
    """Validate that a loaded drafter can safely pair with a target model.

    This intentionally uses architecture/config fields instead of repository
    names, so quantized MLX conversions and local checkpoints remain accepted.
    """
    draft_cfg = getattr(draft_model, "config", None)
    if draft_cfg is None:
        return

    model_type = _cfg_get(draft_cfg, "model_type")
    expected_kind = _expected_drafter_kind(model_type, draft_cfg)
    if expected_kind is not None and draft_kind != expected_kind:
        raise ValueError(
            f"Drafter model_type={model_type!r} requires draft_kind={expected_kind!r}. "
            f"Got draft_kind={draft_kind!r}."
        )

    _validate_model_specific_compatibility(target_model, draft_model)

    if draft_kind != "mtp":
        return

    draft_hidden_size = (
        _cfg_get(draft_cfg, "backbone_hidden_size")
        or _cfg_get(draft_cfg, "target_hidden_size")
        or _hidden_size(draft_cfg)
    )
    target = getattr(target_model, "language_model", target_model)
    target_hidden_size = _hidden_size(getattr(target, "config", None))

    if (
        draft_hidden_size is not None
        and target_hidden_size is not None
        and draft_hidden_size != target_hidden_size
    ):
        raise ValueError(
            "Drafter is incompatible with the target model. "
            "Use the drafter checkpoint for the same target family and size. "
            f"Drafter target hidden_size={draft_hidden_size!r}, "
            f"target hidden_size={target_hidden_size!r}."
        )


def _read_drafter_config(model_path) -> dict:
    """Read the drafter's HF ``config.json`` without loading weights. Returns an
    empty dict when the config can't be read."""
    try:
        with open(model_path / "config.json") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    # Rapid upstream-bugfix (documented deviation): pinned 0.7.1 returns
    # any decoded JSON value; a non-object config crashes resolve_drafter_kind
    # on config.get(). Degrade to the documented empty-dict contract.
    return config if isinstance(config, dict) else {}


def _peek_drafter_model_type(model_path) -> Optional[str]:
    config = _read_drafter_config(model_path)
    model_type = config.get("model_type") or config.get("speculators_model_type")
    # Rapid upstream-bugfix (documented deviation): sidecar checkpoints
    # declare the backbone model type (e.g. "qwen3") and carry the drafter
    # settings in a nested ``dflash_config`` object — the served type is
    # normalized only later by the family's ``Config.from_dict``, so
    # binding on the raw type would skip the vendored shim and let pinned
    # ``load_model`` construct the backbone architecture instead. The
    # DFlash2-exclusive selector/conv keys discriminate DFlash2 from the
    # Qwen3 DFlash layout sharing the same nested object.
    if model_type not in _SERVED_ARCHITECTURE_FAMILIES and isinstance(
        config.get("dflash_config"), dict
    ):
        dflash_config = config["dflash_config"]
        dflash2_keys = (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
            "input_embedding_scale",
            "output_multiplier",
        )
        if any(key in dflash_config for key in dflash2_keys):
            return "dflash2"
        return "qwen3_dflash"
    return model_type


def _declares_mtp_layers(config: Any) -> bool:
    """True when the config declares next-token-prediction layers, the marker of
    a native checkpoint that carries an embedded ``mtp.*`` head."""
    for cfg in (config, _cfg_get(config, "text_config")):
        if cfg is None:
            continue
        count = _cfg_get(cfg, "num_nextn_predict_layers")
        if isinstance(count, int) and count > 0:
            return True
    return False


def _expected_drafter_kind(model_type: Any, config: Any = None) -> Optional[str]:
    """Round-loop kind a drafter requires, or ``None`` when it can't be inferred:
    an explicit ``model_type`` mapping first, then an ``mtp`` model_type name,
    then a config that declares next-token-prediction layers."""
    expected = DRAFTER_KIND_BY_MODEL_TYPE.get(model_type)
    if expected is not None:
        return expected
    if "mtp" in str(model_type).lower():
        return "mtp"
    if config is not None and _declares_mtp_layers(config):
        return "mtp"
    return None


def resolve_drafter_kind(model_path, kind: Optional[str] = None) -> str:
    """Reconcile the caller's ``kind`` with the drafter's actual model type.

    When ``kind`` is None, auto-detect from the drafter's HF ``model_type`` or,
    for a native checkpoint that declares next-token-prediction layers, ``mtp``;
    if neither applies, fall back to :data:`DEFAULT_DRAFTER_KIND`.

    When the caller passes a ``kind`` that disagrees with the drafter's
    ``model_type``, we override (and warn). This avoids the trap where a
    user points ``--draft-model`` at e.g. a ``gemma4_assistant`` checkpoint
    but forgets ``--draft-kind mtp``: rather than crashing deep inside
    ``draft_block`` with an opaque error, we pick the right kind for them.
    """
    config = _read_drafter_config(model_path)
    model_type = config.get("model_type") or config.get("speculators_model_type")
    expected = _expected_drafter_kind(model_type, config)

    if kind is None:
        resolved = expected or DEFAULT_DRAFTER_KIND
        logger.info(
            "Auto-detected --draft-kind=%r for drafter %r (model_type=%r).",
            resolved,
            str(model_path),
            model_type,
        )
        return resolved

    if expected is not None and expected != kind:
        logger.warning(
            "Drafter %r has model_type=%r which requires --draft-kind=%r; "
            "got --draft-kind=%r. Overriding to %r.",
            str(model_path),
            model_type,
            expected,
            kind,
            expected,
        )
        return expected
    return kind


def load_drafter(
    path_or_repo: str, kind: Optional[str] = None, **kwargs
) -> Tuple[object, str]:
    """Load a speculative drafter and return ``(model, resolved_kind)``.

    ``kind`` defaults to ``None``, which triggers auto-detection from the
    drafter's HF ``model_type`` (see :func:`resolve_drafter_kind`). Callers
    should use ``resolved_kind`` for downstream dispatch instead of trusting
    their original ``kind`` arg.
    """
    if kind is not None and kind not in KNOWN_DRAFTER_KINDS:
        raise ValueError(
            f"Unknown drafter kind {kind!r}. Known: {sorted(KNOWN_DRAFTER_KINDS)}"
        )
    from mlx_vlm.utils import get_model_path, load_model

    path = get_model_path(path_or_repo)
    install_served_architecture_bindings(_peek_drafter_model_type(path))
    resolved = resolve_drafter_kind(path, kind)
    return load_model(path, **kwargs), resolved


__all__ = [
    "DEFAULT_DRAFTER_KIND",
    "DRAFTER_KIND_BY_MODEL_TYPE",
    "KNOWN_DRAFTER_KINDS",
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "DSparkDraftModel",
    "LagunaDFlashDraftModel",
    "MuseGlimmerAssistantDraftModel",
    "load_drafter",
    "resolve_drafter_kind",
    "validate_drafter_compatibility",
]
