# SPDX-License-Identifier: Apache-2.0
"""Production assembly for Rapid's continuous self-MTP runtime.

The engine and MLX backend deliberately accept injected protocols.  This
module is the narrow production bridge from an MTP-injected loaded model to
those protocols.  It validates the injector's capability descriptor before it
constructs any runtime object and imports ``mlx_lm`` lazily when a target cache
is actually requested.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from .continuous_engine import (
    ContinuousSelfMTPCapabilities,
    ContinuousSelfMTPConfig,
    ContinuousSelfMTPRuntime,
    ContinuousSelfMTPUnsupportedError,
    RapidForwardSeams,
)
from .mlx_backend import RapidMLXSelfMTPBackend, RapidRaggedCacheAdapter
from .ragged_cache import preflight_ragged_cache, trim_ragged_cache

_SUPPORTED_FAMILIES = frozenset({"qwen3_5"})


def _unsupported(message: str) -> ContinuousSelfMTPUnsupportedError:
    return ContinuousSelfMTPUnsupportedError(
        f"cannot assemble continuous self-MTP runtime: {message}"
    )


def _make_prompt_cache(model: Any) -> Any:
    """Construct a target-trunk cache without making module import eager."""

    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def _descriptor_for(model: Any) -> Mapping[str, Any]:
    candidate = getattr(model, "language_model", None)
    found: list[Mapping[str, Any]] = []
    for owner in (candidate, model):
        descriptor = getattr(owner, "batched_mtp_capability", None)
        if isinstance(descriptor, Mapping):
            found.append(descriptor)
    if not found:
        raise _unsupported("model has no batched_mtp_capability descriptor")
    if any(dict(descriptor) != dict(found[0]) for descriptor in found[1:]):
        raise _unsupported("outer and inner capability descriptors disagree")
    return found[0]


def _resolve_inner(model: Any, family: str) -> Any:
    if family == "qwen3_5":
        # Use the injector's resolver rather than growing a subtly different
        # list of supported wrapper shapes here.
        from .qwen3_5_inject import _resolve_inner_text_model

        inner = _resolve_inner_text_model(model)
    else:  # Guarded by descriptor validation; retained for direct testability.
        inner = None
    if inner is None:
        raise _unsupported(f"cannot resolve injected {family} text model")
    return inner


def _require_descriptor(
    descriptor: Mapping[str, Any],
) -> tuple[str, str]:
    family = descriptor.get("model_family")
    if not isinstance(family, str) or family not in _SUPPORTED_FAMILIES:
        raise _unsupported(f"unsupported model family: {family!r}")

    required = {
        "protocol_version": 1,
        "recursive_draft_depth": 2,
        "fixed_membership": True,
        "target_return_hidden": True,
        "mtp_return_hidden": True,
        "confirmed_target_forward": True,
        "ragged_rollback": True,
        "atomic_cache_commit": True,
        "quantized_cache": False,
        "windowed_cache": False,
        "xtc": False,
    }
    for name, expected in required.items():
        if descriptor.get(name) != expected:
            raise _unsupported(
                f"capability descriptor mismatch: {name} must be {expected!r}"
            )

    batch_forward_name = descriptor.get("batch_forward")
    if not isinstance(batch_forward_name, str) or not batch_forward_name:
        raise _unsupported("capability descriptor has no batch_forward method")
    return family, batch_forward_name


def _require_target_abi(inner: Any) -> None:
    if not callable(inner):
        raise _unsupported("resolved text model is not callable")
    try:
        signature = inspect.signature(inner.__call__)
    except (TypeError, ValueError) as exc:
        raise _unsupported("cannot inspect target forward ABI") from exc
    missing = tuple(
        name
        for name in ("return_hidden", "n_confirmed")
        if name not in signature.parameters
    )
    if missing:
        raise _unsupported("target forward ABI is missing " + ", ".join(missing))


def assemble_continuous_self_mtp_runtime(
    model: Any,
    *,
    allow_dynamic_membership: bool = False,
    array_ops: Any = None,
    logits_processor: Any = None,
    prefill_step_size: int = 512,
) -> ContinuousSelfMTPRuntime:
    """Build a ready runtime from an MTP-injected Rapid model.

    Fixed-core capabilities are admitted only after the versioned descriptor,
    target ABI, injected batch-forward seam, and cache factories are all
    present.  Dynamic membership is an additional conjunction of caller policy
    and descriptor attestation; requesting it cannot manufacture a capability.
    """

    descriptor = _descriptor_for(model)
    family, batch_forward_name = _require_descriptor(descriptor)
    inner = _resolve_inner(model, family)
    _require_target_abi(inner)

    inner_descriptor = getattr(inner, "batched_mtp_capability", None)
    if not isinstance(inner_descriptor, Mapping) or dict(inner_descriptor) != dict(
        descriptor
    ):
        raise _unsupported("resolved text model does not carry the same descriptor")

    batch_forward = getattr(inner, batch_forward_name, None)
    if not callable(batch_forward):
        raise _unsupported(f"injected method {batch_forward_name!r} is not callable")
    make_mtp_cache = getattr(inner, "make_mtp_cache", None)
    if not callable(make_mtp_cache):
        raise _unsupported("injected make_mtp_cache is not callable")

    from .ragged_cache import install_ragged_cache_rollback

    install_ragged_cache_rollback(qwen4_state_cls=None, qsa_cls=None)

    def mtp_forward(hidden: Any, token_ids: Any, cache: Any, *, return_hidden: bool):
        # RapidForwardSeams always asks for hidden state.  The injected batched
        # method bakes that request into its contract and accepts no flag.
        if return_hidden is not True:
            raise _unsupported("batched MTP forward must return hidden state")
        return batch_forward(hidden, token_ids, cache)

    dynamic_membership = (
        allow_dynamic_membership and descriptor.get("dynamic_join") is True
    )
    capabilities = ContinuousSelfMTPCapabilities(
        target_return_hidden=descriptor.get("target_return_hidden") is True,
        mtp_return_hidden=descriptor.get("mtp_return_hidden") is True,
        confirmed_target_forward=descriptor.get("confirmed_target_forward") is True,
        ragged_rollback=descriptor.get("ragged_rollback") is True,
        atomic_cache_commit=descriptor.get("atomic_cache_commit") is True,
        dynamic_membership=dynamic_membership,
        flash_dynamic_membership_attested=False,
    )
    missing = capabilities.missing_fixed_core()
    if missing:  # Defensive: future capability additions remain fail-closed.
        raise _unsupported("missing fixed-core capability: " + ", ".join(missing))

    return ContinuousSelfMTPRuntime(
        config=ContinuousSelfMTPConfig(
            enabled=True,
            allow_dynamic_membership=allow_dynamic_membership,
            architecture=family,
        ),
        capabilities=capabilities,
        forwards=RapidForwardSeams(inner, mtp_forward),
        compute=RapidMLXSelfMTPBackend(
            target_cache_factory=lambda: _make_prompt_cache(inner),
            draft_cache_factory=make_mtp_cache,
            array_ops=array_ops,
            logits_processor=logits_processor,
            prefill_step_size=prefill_step_size,
        ),
        caches=RapidRaggedCacheAdapter(
            preflight=preflight_ragged_cache,
            trim=trim_ragged_cache,
        ),
    )


__all__ = ["assemble_continuous_self_mtp_runtime"]
