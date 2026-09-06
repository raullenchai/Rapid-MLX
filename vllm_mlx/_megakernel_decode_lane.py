# SPDX-License-Identifier: Apache-2.0
"""Opt-in plain-decode lane on the whole-token persistent megakernel.

Background
----------
Some mlx-lm builds ship a *plain-decode megakernel*: after a stock prefill,
every further token of a width-1, non-speculative completion runs as one
persistent Metal dispatch (the layer chain plus ``lm_head`` inside the kernel),
instead of the per-op eager graph mlx-lm normally replays each step. The lane
lives entirely inside ``mlx_lm.generate.generate_step`` — when the build's
``mlx_lm.megakernel_lane`` module is present, the master switch is on, and the
request is width-1 plain (no KV quantization, no rotating/paged cache, no
caller-owned session cache, batch=1), ``generate_step`` attaches the lane on
its own and drives it through its normal decode loop. Rapid never has to
reimplement the loop; it only has to (a) recognise a megakernel-capable model,
(b) flip the build's master switch, (c) keep speculative / tool-constrained /
batched requests off the lane, and (d) observe engagement for logging.

Value, honestly
---------------
The megakernel is Rapid's *fastest plain (non-speculative) decode lane* — it is
for when self-MTP is off or ineffective (temperature sampling, tool-constrained
generation, low draft acceptance). It is **not** a speculative-decode
replacement: on the Flash-Next (qwen4_exp) geometry the lane decodes plain at
~59.5 tok/s (1.68-1.81x the stock plain path) but does **not** beat that model's
own self-MTP (66-70 tok/s); on the dense 35B (qwen3_5_moe) geometry it decodes
plain at ~118 tok/s but does not beat a compiled-replay decode. So the lane is
routed only for width-1 plain requests, and speculative requests fall through to
their existing path untouched.

Why a build dependency, not vendored code
-----------------------------------------
The kernel, its geometries, and the mixed-precision (4/6-bit) expert decode the
dense checkpoint needs are ~8k lines that live in the mlx-lm build, not here.
Rapid depends on a build that exposes ``mlx_lm.megakernel_lane`` and the
``megakernel_geometry`` registry; when the installed mlx-lm has neither (the
stock wheel), this module reports "unavailable" and every entry point is a
silent no-op — the runner uses the ordinary ``generate_step`` path exactly as
before. See the PR description for the required build.

Fail-closed contract
--------------------
Every decision here defaults to OFF. The lane is enabled for a request only
when ALL hold:

* the operator opted in (``RAPID_MEGAKERNEL_DECODE_LANE=1`` / CLI flag);
* the installed mlx-lm exposes the lane (import probe succeeds);
* the loaded model's ``model_type`` maps to a ``MegakernelGeometry`` (and, if
  the operator pinned a geometry, it matches);
* the request is width-1 plain: batch=1, no speculative config, no
  logit-constraining tools/guided decoding;
* the projected end position (context + max_tokens) is within the geometry's
  profile.

Any failure returns a reason and leaves the stock decode path in place. The
lane itself is additionally self-gating and fail-closed inside mlx-lm: it
declines (never crashes the request) on a poisoned pack, an unsupported device,
or an out-of-profile context, and the runner then decodes eagerly.

Counters mirror the ``_mxfp4_moe_guardrail`` pattern — plain module-level ints
behind a lock, snapshot exposed for ``routes/metrics.py`` — so the metrics
surface stays uniform and no default Prometheus registry is touched.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---- Operator-facing environment switches ----------------------------
# The CLI flags (``--megakernel-decode-lane`` / ``--megakernel-geometry``)
# set these before the engine loads the model; the runner reads them at
# construction. Keeping the CLI→runner hop in env vars keeps the flag
# plumbing shallow and the default OFF everywhere it is not set.
ENV_ENABLE = "RAPID_MEGAKERNEL_DECODE_LANE"
ENV_GEOMETRY = "RAPID_MEGAKERNEL_GEOMETRY"
ENV_MAX_CONTEXT = "RAPID_MEGAKERNEL_MAX_CONTEXT"

# mlx-lm build switches the lane reads. We flip these (in addition to the
# in-process ``set_qwen4_megakernel`` toggle) so a build that captured the
# master flag at import still honours a late enable.
_MLX_MASTER_ENV = "MLX_QWEN4_MEGAKERNEL"
_MLX_LANE_ENV = "MLX_QWEN4_MEGAKERNEL_LANE"
_MLX_MAX_WIDTH_ENV = "MLX_QWEN4_MEGAKERNEL_MAX_WIDTH"

# ---- Process-local counters (Prometheus-style, never decrease) --------
_lock = threading.Lock()
_lane_engaged_total = 0
_lane_declined_total = 0
_lane_unavailable_total = 0


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "on", "yes"}


@dataclass(frozen=True)
class MegakernelLaneConfig:
    """Operator intent for the plain-decode megakernel lane.

    Constructed once at runner init from the environment the CLI populated.
    ``geometry`` is ``"auto"`` (accept whichever geometry the model maps to)
    or a pinned geometry name (``"qwen4_exp"`` / ``"qwen36_35b_a3b"``) that
    the loaded model must match, else the lane stays off and logs why.
    ``max_context`` optionally caps the context the lane is allowed for
    (0 = defer entirely to the geometry's own profile).
    """

    enabled: bool = False
    geometry: str = "auto"
    max_context: int = 0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "MegakernelLaneConfig":
        env = env if env is not None else os.environ
        try:
            max_context = int(env.get(ENV_MAX_CONTEXT, "0") or "0")
        except ValueError:
            max_context = 0
        return cls(
            enabled=_truthy(env.get(ENV_ENABLE)),
            geometry=(env.get(ENV_GEOMETRY) or "auto").strip().lower(),
            max_context=max(0, max_context),
        )


@dataclass(frozen=True)
class LaneDecision:
    """Outcome of a per-request routing decision."""

    route: bool
    reason: str


def snapshot_counters() -> dict[str, int]:
    """Point-in-time counter values for ``routes/metrics.py``."""
    with _lock:
        return {
            "rapid_mlx_megakernel_lane_engaged_total": _lane_engaged_total,
            "rapid_mlx_megakernel_lane_declined_total": _lane_declined_total,
            "rapid_mlx_megakernel_lane_unavailable_total": _lane_unavailable_total,
        }


def reset_for_tests() -> None:
    """Zero the counters between test cases (mirrors the guardrail helper)."""
    global _lane_engaged_total, _lane_declined_total, _lane_unavailable_total
    with _lock:
        _lane_engaged_total = 0
        _lane_declined_total = 0
        _lane_unavailable_total = 0


def _bump_engaged() -> None:
    global _lane_engaged_total
    with _lock:
        _lane_engaged_total += 1


def _bump_declined() -> None:
    global _lane_declined_total
    with _lock:
        _lane_declined_total += 1


def _bump_unavailable() -> None:
    global _lane_unavailable_total
    with _lock:
        _lane_unavailable_total += 1


def lane_available() -> bool:
    """Whether the installed mlx-lm exposes the plain-decode megakernel lane.

    Pure import probe — no pack, no device touch, no engine mutation. The
    stock mlx-lm wheel has neither module and returns ``False`` so every
    caller degrades to the ordinary decode path.
    """
    try:
        import mlx_lm.megakernel_lane  # noqa: F401
        import mlx_lm.models.megakernel_geometry  # noqa: F401
    except Exception:
        return False
    return True


def geometry_name_for_model(model: Any) -> str | None:
    """The megakernel geometry name the loaded model maps to, or ``None``.

    Resolves the text model's ``model_type`` through the build's geometry
    registry (``geometry_for_model_type``). Returns ``None`` for any model
    the megakernel does not describe, and for any build without the registry.
    """
    try:
        from mlx_lm.models.megakernel_geometry import geometry_for_model_type
    except Exception:
        return None
    text_model = getattr(model, "language_model", model)
    args = getattr(text_model, "args", None)
    model_type = getattr(args, "model_type", None)
    if not isinstance(model_type, str):
        return None
    geometry = geometry_for_model_type(model_type)
    return geometry.name if geometry is not None else None


def geometry_profile_capacity(model: Any) -> int | None:
    """Max position count the mapped geometry admits (``max_position_embeddings``)."""
    try:
        from mlx_lm.models.megakernel_geometry import geometry_for_model_type
    except Exception:
        return None
    text_model = getattr(model, "language_model", model)
    args = getattr(text_model, "args", None)
    model_type = getattr(args, "model_type", None)
    if not isinstance(model_type, str):
        return None
    geometry = geometry_for_model_type(model_type)
    if geometry is None:
        return None
    return int(getattr(geometry, "max_position_embeddings", 0)) or None


def configure_process_env(config: MegakernelLaneConfig) -> None:
    """Flip the mlx-lm build's master switches for an opted-in run.

    Idempotent and only ever *enables* — it never disables a switch the
    operator set by hand. Called once, before the model loads, so a build
    that captures ``MLX_QWEN4_MEGAKERNEL`` at import sees it on. A separate
    in-process ``set_qwen4_megakernel(True)`` in :func:`enable_for_model`
    covers builds already imported by the time we run.
    """
    if not config.enabled:
        return
    os.environ.setdefault(_MLX_MASTER_ENV, "1")
    os.environ.setdefault(_MLX_LANE_ENV, "1")
    # The plain lane is width-1 by construction; pin the build's width so a
    # stray dual-width default can never widen this lane.
    os.environ.setdefault(_MLX_MAX_WIDTH_ENV, "1")


def enable_for_model(config: MegakernelLaneConfig, model: Any) -> LaneDecision:
    """Decide whether the lane can serve this loaded model, and arm the build.

    Returns a :class:`LaneDecision`. On ``route=True`` the build's master
    switch is armed (env + ``set_qwen4_megakernel(True)``) and the runner may
    pass a status dict to ``generate_step`` for width-1 plain requests. Every
    negative path is a clean no-op that leaves the stock decode in place.
    """
    if not config.enabled:
        return LaneDecision(False, "megakernel decode lane not enabled")
    if not lane_available():
        _bump_unavailable()
        return LaneDecision(
            False,
            "installed mlx-lm does not expose the megakernel lane "
            "(stock wheel); using generate_step",
        )
    name = geometry_name_for_model(model)
    if name is None:
        return LaneDecision(
            False, "loaded model has no registered megakernel geometry"
        )
    if config.geometry != "auto" and config.geometry != name:
        return LaneDecision(
            False,
            f"pinned geometry {config.geometry!r} does not match the loaded "
            f"model's geometry {name!r}",
        )
    # Arm the in-process master flag for builds imported before
    # ``configure_process_env`` ran. Import-guarded: a build without the
    # toggle simply relies on the env switch.
    try:
        from mlx_lm.models.qwen4_megakernel import set_qwen4_megakernel

        set_qwen4_megakernel(True)
    except Exception:
        pass
    return LaneDecision(True, f"megakernel lane armed for geometry {name!r}")


def _has_constraining_tools(sampling_params: Any) -> bool:
    """Whether the request constrains logits mid-generation (tools/guided).

    Such a request must NOT ride the lane: the megakernel emits its own
    logits per step and does not expose the per-step logit-processor hook a
    grammar/tool constraint needs. Detection is conservative — any sign of a
    guided-decoding / grammar / tool-choice constraint routes to the eager
    path. Unknown shapes are treated as constraining (fail-closed).
    """
    if sampling_params is None:
        return False
    for attr in (
        "guided_decoding",
        "guided_grammar",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "grammar",
        "logits_processors",
        "response_format",
    ):
        value = getattr(sampling_params, attr, None)
        if value:
            return True
    return False


def route_decision(
    config: MegakernelLaneConfig,
    *,
    armed: bool,
    context_len: int,
    max_tokens: int,
    batch_size: int,
    is_speculative: bool,
    sampling_params: Any,
    capacity: int | None,
) -> LaneDecision:
    """Per-request width-1-plain gate. Returns whether to route to the lane.

    ``armed`` is the result of :func:`enable_for_model` for the loaded model.
    Every guard below fails closed to the stock decode path.
    """
    if not armed:
        return LaneDecision(False, "lane not armed for this model")
    if batch_size != 1:
        _bump_declined()
        return LaneDecision(False, f"batch size {batch_size} != 1")
    if is_speculative:
        _bump_declined()
        return LaneDecision(False, "speculative request keeps its own path")
    if _has_constraining_tools(sampling_params):
        _bump_declined()
        return LaneDecision(False, "logit-constraining tools/guided decoding")
    # Context-in-profile: the projected end position must fit the geometry's
    # own capacity and any operator-set cap.
    projected = context_len + (max_tokens if max_tokens and max_tokens > 0 else 0) + 1
    if capacity is not None and projected > capacity:
        _bump_declined()
        return LaneDecision(
            False,
            f"projected end position {projected} exceeds geometry capacity {capacity}",
        )
    if config.max_context and context_len > config.max_context:
        _bump_declined()
        return LaneDecision(
            False,
            f"context {context_len} exceeds operator cap {config.max_context}",
        )
    return LaneDecision(True, "width-1 plain request within profile")


def new_status_dict() -> dict[str, Any]:
    """A fresh status dict to hand ``generate_step`` as ``_megakernel_status``."""
    return {}


def note_engagement(status: dict[str, Any] | None) -> bool:
    """Record whether the lane actually engaged for a completed request.

    ``generate_step`` writes ``used=True`` into the status dict when the lane
    attaches, or ``decline_reason`` when it self-declines (out-of-profile,
    poisoned, another lane active). We bump the matching counter and return
    whether it engaged, for the runner's debug log.
    """
    if not status:
        _bump_declined()
        return False
    if status.get("used"):
        _bump_engaged()
        return True
    _bump_declined()
    return False
