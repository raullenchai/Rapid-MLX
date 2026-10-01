# SPDX-License-Identifier: MIT
#
# Row-invariant lane matmul: the design and the original Metal kernels are
# TensorFold's (github.com/ashhart/TensorFold, MIT, (c) 2026 TensorFold
# contributors); this implementation is adapted from mlx2
# (github.com/pierre427/mlx2, src/mlx2/runtime/lane/). See LICENSE-LANE-MATMUL.

"""Row-invariant small-M matmul for multi-row verify and batched decode.

A speculative verify (MTP, copy-draft) or a batched decode step runs several
rows through each projection.  MLX's quantized matmul picks a different
kernel by row count, and its cost grows with the rows.  The lane matmul uses
one arithmetic for every row count and reads each weight once for all rows,
so 16 rows cost about as much as one:

* ``mpp`` (M5, Metal 4 tensor units): affine 2-8-bit and bf16/fp16 weights;
* ``simd`` (M1-M4, simdgroup matrix units): affine 2-8-bit weights, bf16
  activations (TensorFold's row-exact ``simd_qmm`` family).

Opt-in with ``RAPID_MLX_LANE_MATMUL``:

* unset / ``0`` / ``off`` (default): not installed;
* ``crossover``: the lane arithmetic from 8 rows (16 for bf16/fp16 weights);
  fewer rows (one-token decode, a K<=6 MTP verify) keep stock kernels;
* ``exact``: every call of 1-32 rows, so a verify row's projections equal the
  same row decoded alone.

Mixture-of-experts models are skipped: their expert layers use ``gather_qmm``,
which is not covered.  The installer walks the backbone as loaded; projections
attached later (a native MTP head injected after load) keep stock kernels, so
``exact`` makes the backbone's projections row-invariant, not the draft head's.
On M1-M4 only bf16 checkpoints are covered (the simd kernels read bf16
activations).
"""

from __future__ import annotations

import logging
import os

from .installer import LAW_IDS, format_class, install, law_id, stats, uninstall
from .matmul import LaneUnsupportedError, available, backend, force_backend

logger = logging.getLogger(__name__)

ENABLE_ENV = "RAPID_MLX_LANE_MATMUL"
MODES = ("off", "crossover", "exact")
# Crossovers per weight format, from a sweep of 34 dense and MoE checkpoints
# at 8K and 32K context on an M5 Max (lane beats stock from 8 rows on 2-8-bit
# weights and from 16 on bf16); the simd backend's per-projection crossover on
# an M3 Pro is also between 4 and 8 rows.
CROSSOVER = {
    "q2": 8,
    "q3": 8,
    "q4": 8,
    "q5": 8,
    "q6": 8,
    "q8": 8,
    "bf16": 16,
    "fp16": 16,
}


def mode_from_env() -> str:
    raw = os.environ.get(ENABLE_ENV, "").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return "off"
    if raw not in MODES:
        raise ValueError(f"{ENABLE_ENV} must be one of {MODES}, got {raw!r}")
    return raw


_EXPERT_KEYS = (
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "moe_num_experts",
)


def _config_moe(config) -> bool:
    """Whether a model config (mlx-lm ``model.args``, or a dict) routes to experts."""
    if config is None:
        return False
    scopes = [config]
    for nested in ("text_config", "llm_config"):
        inner = (
            config.get(nested)
            if isinstance(config, dict)
            else getattr(config, nested, None)
        )
        if inner is not None:
            scopes.append(inner)
    for scope in scopes:
        for key in _EXPERT_KEYS:
            value = (
                scope.get(key) if isinstance(scope, dict) else getattr(scope, key, None)
            )
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return True
    return False


def is_moe(model) -> bool:
    """Routed experts, from the config's expert count or an expert module.

    The config is authoritative (a checkpoint declaring experts is MoE
    whatever its module names); the module scan catches expert containers of
    models without an args object (mlx-lm's SwitchGLU / SwitchLinear, and
    the usual ``*MoE`` / ``*SparseMoeBlock`` / ``*Experts`` names).
    """
    for holder in (model, getattr(model, "language_model", None)):
        if holder is not None and _config_moe(getattr(holder, "args", None)):
            return True
    for _name, module in model.named_modules():
        kind = type(module).__name__
        if "Switch" in kind or kind.endswith(("MoE", "SparseMoeBlock", "Experts")):
            return True
    return False


def install_lane_matmul(model, mode: str | None = None) -> dict | None:
    """Install per ``mode`` (default: from the environment); None when not installed."""
    mode = mode_from_env() if mode is None else mode
    if mode == "off":
        return None
    if mode not in MODES:
        raise ValueError(f"lane matmul mode must be one of {MODES}")
    if not available():
        logger.warning(
            "[lane_matmul] %s requested but this device has no lane backend", mode
        )
        return None
    if is_moe(model):
        logger.info(
            "[lane_matmul] skipped: mixture-of-experts model (expert layers not covered)"
        )
        return None
    thresholds = {fmt: 1 for fmt in CROSSOVER} if mode == "exact" else dict(CROSSOVER)
    try:
        receipt = install(model, min_rows_by_format=thresholds)
    except Exception:
        # A kernel that fails to compile or launch in an install-time probe
        # must not abort serving: restore every projection to stock.
        uninstall(model)
        logger.warning(
            "[lane_matmul] install failed; serving with stock kernels", exc_info=True
        )
        return None
    receipt["mode"] = mode
    if not receipt["covered"]:
        uninstall(model)
        logger.info("[lane_matmul] nothing covered: %s", receipt["refused"])
        return None
    logger.info(
        "[lane_matmul] %s: law=%s covered=%s groups=%s stock_stacked=%s refused=%s",
        mode,
        receipt["law_id"],
        receipt["covered"],
        receipt["groups"],
        receipt.get("stock_stacked"),
        receipt["refused"],
    )
    return receipt


__all__ = [
    "CROSSOVER",
    "ENABLE_ENV",
    "LAW_IDS",
    "LaneUnsupportedError",
    "available",
    "backend",
    "force_backend",
    "format_class",
    "install",
    "install_lane_matmul",
    "is_moe",
    "law_id",
    "mode_from_env",
    "stats",
    "uninstall",
]
