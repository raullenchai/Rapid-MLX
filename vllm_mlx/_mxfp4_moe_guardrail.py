# SPDX-License-Identifier: Apache-2.0
"""Load-time guardrail for the MoE + MXFP4 + multi-device throughput cliff.

Background — task #297, R15 quantization + Metal + MoE scout findings:

* **mlx#3402** (still open as of 2026-06): the three-tuple MoE
  architecture + MXFP4 quantization + multi-device dispatch hits a 60×
  throughput cliff on M3 Ultra distributed runs. GLM-5.1 / DeepSeek-V3.2
  measured **0.27 tok/s vs expected ~16 tok/s**. The bug lives in
  ``mlx.core``'s MXFP4 quantization kernels — the gather/scatter pattern
  used by MoE expert routing fights the inter-device packing scheme.

* **mlx#2962** (still open): MLX NVFP4 uses signed E4M3 scales instead
  of Blackwell's unsigned UE4M3, costing ~137× dynamic range. MoE
  weights stored as NVFP4 silently fall off the cliff long before the
  user notices token-quality regressions.

Without a guardrail, the next big MoE drop (GLM-5.x, DeepSeek V4 family,
Kimi K2.7) will surface as a customer-visible throughput regression
that operators cannot self-diagnose — the symptom is "serving is slow"
and the only person who can correlate it back to the three-tuple is
someone who's already read both upstream issues.

Decision (v1) — **warn only**. Auto-routing to a non-MXFP4 variant is
deferred because we don't yet have a clean "fallback alias for X" map
across the registry. Operators see the warning, click the issue link,
and can pick a fallback themselves. The follow-up to auto-route lands
once we have a curated quant-variant map per alias family.

What the guardrail does at load time:

1. Detect MoE-ness from the alias profile (``is_moe`` flag from
   ``aliases.json``).
2. Detect quant format from the HF path string (``mxfp4`` / ``nvfp4``
   token, case-insensitive). The path-based heuristic matches the way
   ``aliases.json`` already names these variants
   (``qwen3.5-122b-mxfp4``, ``minimax-m2.7-mxfp4``, etc.) and stays
   robust without parsing each ``config.json`` quant block at load time.
3. Detect multi-device dispatch via launcher state set BEFORE the
   worker spawns. We use a multi-signal probe because upstream
   ``mlx.distributed_run`` uses different vars per backend:
   * ``MLX_WORLD_SIZE`` — NCCL backend only (CUDA hosts).
   * ``OMPI_COMM_WORLD_SIZE`` / ``PMI_SIZE`` — MPI launchers.
   * ``MLX_HOSTFILE`` — ring backend (Apple Silicon default,
     and the precise target of the mlx#3402 cliff). We open the
     hostfile and count its JSON entries — read-only, no side
     effects.

   We deliberately do NOT call ``mlx.distributed.init()`` for the
   probe because it creates the global communication group, can
   block while a backend handshakes, and ``backend="any"`` would
   make the first successful backend the *global* group. A
   warning-only guardrail must never mutate engine state.
4. On the full three-tuple match, log a loud WARNING with the issue
   link and bump ``rapid_mlx_mxfp4_moe_distributed_warnings_total``.
5. On a MoE + NVFP4 match (any device count — mlx#2962 dynamic-range
   loss bites even single-device), log a separate WARNING and bump
   ``rapid_mlx_nvfp4_moe_warnings_total``.

Process-local counters mirror the pattern already used by
``api/response_format_metrics.py`` — hand-rolled module-level ints
behind a ``threading.Lock`` rather than ``prometheus_client``, so we
keep the metrics surface uniform with the rest of rapid-mlx and avoid
the global default registry that fights multi-engine tests.

Counters never decrease for the lifetime of the process. Tests use
:func:`reset_for_tests` to zero them between cases.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---- Module-level Prometheus counters --------------------------------
# Same pattern as ``api/response_format_metrics.py``: plain ints behind
# a lock, snapshot exposed to ``routes/metrics.py``. Counters never
# decrease for the process lifetime — Prometheus contract.
_lock = threading.Lock()
_mxfp4_moe_distributed_warnings_total = 0
_nvfp4_moe_warnings_total = 0


# Single canonical issue link, hoisted to a constant so tests can assert
# the warning carries the upstream pointer and so a future link change
# touches one line.
MLX_3402_URL = "https://github.com/ml-explore/mlx/issues/3402"
MLX_2962_URL = "https://github.com/ml-explore/mlx/issues/2962"


@dataclass(frozen=True)
class GuardrailSignal:
    """Detection inputs used by :func:`check_load_time_guardrails`.

    Captured as a small frozen dataclass so the call site in
    ``server.load_model()`` constructs it once and the detection helper
    stays pure / unit-testable without poking at server globals.
    """

    is_moe: bool
    quant_format: Optional[str]  # "mxfp4", "nvfp4", or None
    distributed_world_size: int


def _detect_quant_format(hf_path: str | None) -> Optional[str]:
    """Best-effort quant-format detection from the HF path string.

    rapid-mlx's ``aliases.json`` names mxfp4 / nvfp4 variants with the
    quant token embedded in the alias and the HF path (e.g.
    ``mlx-community/MiniMax-M2.7-4bit-mxfp4``). That naming convention
    is the single load-time signal we trust — parsing every model's
    ``config.json`` quant block at load time would be both heavier and
    less reliable (uploaders are inconsistent about that field).

    Returns ``"mxfp4"`` / ``"nvfp4"`` if the corresponding token appears
    anywhere in the path (case-insensitive). Returns ``None`` when the
    path is missing or carries neither token. ``mxfp4`` wins over
    ``nvfp4`` if both somehow appear — the three-tuple cliff is the
    more severe of the two issues so the warning that fires should be
    the louder one.
    """
    if not hf_path:
        return None
    lowered = hf_path.lower()
    if "mxfp4" in lowered:
        return "mxfp4"
    if "nvfp4" in lowered:
        return "nvfp4"
    return None


def _detect_distributed_world_size() -> int:
    """Return the active MLX distributed world size, or 1 as a safe default.

    **Non-mutating probe.** We deliberately do NOT call
    ``mlx.core.distributed.init()`` here even though it would give the
    exact group size, because ``init()`` is documented as creating the
    global communication group — calling it purely for a metric/warning
    probe would either (a) initialize an unwanted singleton group on
    single-host runs or (b) block during startup if the distributed
    environment is present but not yet ready (codex review #297 round 1).
    Worse, ``backend="any"`` makes the first successful backend the
    *global* group, which is a real side-effect on the engine that
    follows.

    Instead, we read launcher state via env vars set by ``mlx-launcher``
    / ``mlx.distributed_run`` and common MPI launchers BEFORE they spawn
    the worker. Multiple signals are checked because the upstream
    launcher uses a *different* set of vars per backend (codex review
    #297 round 2):

    * ``MLX_WORLD_SIZE`` — set ONLY for the NCCL backend (CUDA hosts).
      The ring backend, which is the Apple Silicon default and the
      precise target of the mlx#3402 cliff, does NOT set this.
    * ``OMPI_COMM_WORLD_SIZE`` — OpenMPI's ``mpirun``.
    * ``PMI_SIZE`` — MPICH / Intel MPI launchers.
    * ``MLX_HOSTFILE`` — set by the ring backend (Apple Silicon
      default). Points to a JSON array of host descriptors; its length
      IS the world size. We open the file and count entries — read-only,
      cannot mutate engine state.

    If any of these signals reports a multi-device run we return it;
    otherwise we report 1 (single device).
    """
    import json
    import os

    # Direct integer env vars (NCCL + MPI launchers).
    for env_var in ("MLX_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        raw = os.environ.get(env_var)
        if not raw:
            continue
        try:
            size = int(raw)
        except ValueError:
            continue
        if size >= 1:
            return size

    # Ring backend (Apple Silicon default) — count entries in the
    # JSON hostfile. The file is a JSON array of ``{ssh, ips}`` host
    # objects; ``len(hosts)`` is the world size as constructed by
    # ``mlx/distributed_run.py``. Read-only — no side effects.
    hostfile_path = os.environ.get("MLX_HOSTFILE")
    if hostfile_path:
        try:
            with open(hostfile_path) as fp:
                hosts = json.load(fp)
            if isinstance(hosts, list) and len(hosts) >= 1:
                return len(hosts)
        except Exception:  # pragma: no cover — defensive
            logger.debug(
                "MLX_HOSTFILE=%s present but unreadable; treating as size 1",
                hostfile_path,
                exc_info=True,
            )

    return 1


def _emit_mxfp4_moe_distributed_warning(
    *,
    hf_path: str | None,
    alias: str | None,
    world_size: int,
) -> None:
    """Log the WARNING and bump the counter for the three-tuple cliff."""
    global _mxfp4_moe_distributed_warnings_total
    with _lock:
        _mxfp4_moe_distributed_warnings_total += 1
    logger.warning(
        "MoE + MXFP4 + multi-device throughput cliff detected "
        "(mlx#3402, ~0.27 tok/s observed on M3 Ultra distributed vs "
        "~16 tok/s expected). model=%s alias=%s world_size=%d. "
        "Consider switching to a non-MXFP4 variant (e.g. 4-bit integer) "
        "or single-device serving until upstream fix lands. "
        "Details: %s",
        hf_path or "<unknown>",
        alias or "<none>",
        world_size,
        MLX_3402_URL,
    )


def _emit_nvfp4_moe_warning(
    *,
    hf_path: str | None,
    alias: str | None,
) -> None:
    """Log the WARNING and bump the counter for MoE + NVFP4 dynamic-range loss."""
    global _nvfp4_moe_warnings_total
    with _lock:
        _nvfp4_moe_warnings_total += 1
    logger.warning(
        "MoE + NVFP4 dynamic-range loss detected "
        "(mlx#2962, signed E4M3 scales instead of Blackwell unsigned UE4M3 "
        "→ ~137x dynamic-range loss). model=%s alias=%s. "
        "MoE expert routing is especially sensitive to the lost range; "
        "expect silent token-quality regressions before throughput drops. "
        "Consider a 4-bit integer or non-NVFP4 variant. Details: %s",
        hf_path or "<unknown>",
        alias or "<none>",
        MLX_2962_URL,
    )


def check_load_time_guardrails(
    signal: GuardrailSignal,
    *,
    hf_path: str | None = None,
    alias: str | None = None,
) -> list[str]:
    """Run the load-time guardrails against ``signal``.

    Returns the list of guardrail names that fired (empty when none did).
    Always returns — never raises and never blocks the load. The intent
    is operator visibility, not enforcement; an over-eager enforcer
    would refuse to load a model that works fine outside the upstream
    bug's three-tuple corner.

    Guardrails:
        * ``"mxfp4_moe_distributed"`` — the full three-tuple cliff
          (mlx#3402).
        * ``"nvfp4_moe"`` — MoE + NVFP4 dynamic-range loss (mlx#2962).
          Fires regardless of device count because the dynamic-range
          loss bites even on single-device serving.

    Importantly, the three-tuple gate strictly requires all three
    inputs — any 2-of-3 leaves the cliff inactive and we stay silent.
    That's covered by the test matrix; see
    ``tests/test_mxfp4_moe_guardrail.py``.
    """
    fired: list[str] = []

    if not signal.is_moe:
        # Both guardrails gate on MoE. Bail early — keeps the path
        # noise-free for the >95% of aliases that aren't MoE.
        return fired

    quant = signal.quant_format
    is_distributed = signal.distributed_world_size > 1

    if quant == "mxfp4" and is_distributed:
        _emit_mxfp4_moe_distributed_warning(
            hf_path=hf_path,
            alias=alias,
            world_size=signal.distributed_world_size,
        )
        fired.append("mxfp4_moe_distributed")
    elif quant == "nvfp4":
        _emit_nvfp4_moe_warning(hf_path=hf_path, alias=alias)
        fired.append("nvfp4_moe")

    return fired


def check_from_profile(
    *,
    model_name: str,
    profile,  # AliasProfile | None — annotated loosely to avoid an import cycle
    alias: str | None = None,
) -> list[str]:
    """Convenience adapter that bridges ``server.load_model()`` to the guardrail.

    Pulls ``is_moe`` from the resolved ``AliasProfile`` (False when no
    profile was found — bare HF paths are conservatively treated as
    non-MoE because we have no metadata to infer expert count without
    cracking ``config.json``). The HF path used for quant detection is
    ``profile.hf_path`` when available, else the raw ``model_name`` the
    operator passed (which is the HF path itself in the no-alias case).

    Kept in this module so ``server.py`` stays a one-line call site —
    matches the seam style used by other defensive load-time checks
    (e.g. ``_version_check``, ``_download_gate``).
    """
    is_moe = bool(getattr(profile, "is_moe", False))
    hf_path = (
        getattr(profile, "hf_path", None)
        if profile is not None
        else model_name
    )
    quant_format = _detect_quant_format(hf_path or model_name)
    world_size = _detect_distributed_world_size()
    signal = GuardrailSignal(
        is_moe=is_moe,
        quant_format=quant_format,
        distributed_world_size=world_size,
    )
    return check_load_time_guardrails(
        signal,
        hf_path=hf_path or model_name,
        alias=alias,
    )


def snapshot() -> dict[str, int]:
    """Return a consistent snapshot of the guardrail counters for ``/metrics``."""
    with _lock:
        return {
            "mxfp4_moe_distributed_warnings_total": (
                _mxfp4_moe_distributed_warnings_total
            ),
            "nvfp4_moe_warnings_total": _nvfp4_moe_warnings_total,
        }


# Help/TYPE/sample lines for the two counters, exposed as a pure helper
# so unit tests can verify rendering WITHOUT importing
# ``vllm_mlx.routes.metrics`` (which transitively imports the whole
# engine stack via ``vllm_mlx.config`` → ``BaseEngine``).
#
# ``routes/metrics.py:_render_mxfp4_moe_guardrail_counters`` is a thin
# wrapper that calls into this helper plus the shared formatting
# primitives — so the rendered output stays bit-identical regardless of
# which entry point the test uses.
_MXFP4_MOE_HELP = (
    "Load-time warnings fired for the MoE + MXFP4 + multi-device "
    "throughput cliff (upstream mlx#3402). Any non-zero value means an "
    "operator started a model matching the three-tuple; expect "
    "~0.27 tok/s vs ~16 tok/s until upstream lands a fix."
)
_NVFP4_MOE_HELP = (
    "Load-time warnings fired for the MoE + NVFP4 dynamic-range loss "
    "(upstream mlx#2962, signed E4M3 scales instead of Blackwell "
    "unsigned UE4M3 -> ~137x dynamic-range loss). Fires regardless of "
    "device count because the dynamic-range loss bites even on "
    "single-device serving."
)


def render_prometheus_lines() -> list[str]:
    """Render the guardrail counters as Prometheus text-exposition lines.

    Pure helper — does not touch the engine, does not import the route
    module, does not depend on the global ``ServerConfig`` singleton.
    Returns the standard HELP / TYPE / sample triplet for each counter
    in stable order so dashboards / tests can pattern-match against the
    output.
    """
    stats = snapshot()
    mxfp4 = int(stats.get("mxfp4_moe_distributed_warnings_total", 0))
    nvfp4 = int(stats.get("nvfp4_moe_warnings_total", 0))
    return [
        f"# HELP rapid_mlx_mxfp4_moe_distributed_warnings_total {_MXFP4_MOE_HELP}",
        "# TYPE rapid_mlx_mxfp4_moe_distributed_warnings_total counter",
        f"rapid_mlx_mxfp4_moe_distributed_warnings_total {mxfp4}",
        f"# HELP rapid_mlx_nvfp4_moe_warnings_total {_NVFP4_MOE_HELP}",
        "# TYPE rapid_mlx_nvfp4_moe_warnings_total counter",
        f"rapid_mlx_nvfp4_moe_warnings_total {nvfp4}",
    ]


def reset_for_tests() -> None:
    """Test-only hook: zero the counters between cases.

    Production code MUST NOT call this — Prometheus counters are
    contractually monotonic for the process lifetime.
    """
    global _mxfp4_moe_distributed_warnings_total
    global _nvfp4_moe_warnings_total
    with _lock:
        _mxfp4_moe_distributed_warnings_total = 0
        _nvfp4_moe_warnings_total = 0
