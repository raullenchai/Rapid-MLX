# SPDX-License-Identifier: Apache-2.0
"""Build atomic identities and the BenchmarkRun v1 envelope."""

from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from vllm_mlx import __version__
from vllm_mlx.catalog import rcj_digest

from .benchmark_contracts import BenchmarkRunValidator, registered_workload
from .hardware import Hardware, Software, run_conditions


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


#: ``model-identity.schema.json#/$defs/quantization/properties/method``.
_METHOD_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_BASE_DTYPES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


def _cached_config(
    repo_id: str, subfolder: str | None = None
) -> tuple[dict[str, Any], str | None] | None:
    """Return ``(config.json, resolved_revision)`` from the local HF cache.

    The benchmark just loaded this repo, so its snapshot is on disk. Only the
    cache is consulted — never the network — and any failure means "no
    facts", never an exception: identity facts are best-effort provenance.
    ``subfolder`` selects a nested variant (``4bit/config.json``) for repos
    that ship several quantisations side by side.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        filename = "config.json"
        if subfolder:
            filename = f"{subfolder.strip('/')}/config.json"
        path = try_to_load_from_cache(repo_id, filename)
        if not isinstance(path, str):
            return None
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:  # noqa: BLE001 — provenance must never fail a run
        return None
    if not isinstance(config, dict):
        return None
    revision = None
    parts = Path(path).parts
    if "snapshots" in parts:
        candidate = parts[parts.index("snapshots") + 1]
        if len(candidate) == 40 and all(c in "0123456789abcdef" for c in candidate):
            revision = candidate
    return config, revision


def quantization_facts(config: dict[str, Any]) -> dict[str, Any]:
    """Project an MLX ``config.json`` onto the atomic quantization contract.

    MLX converters write ``{"quantization": {"bits": 4, "group_size": 64,
    "mode": "affine", "<layer>": {"bits": 8, ...}}}``; mflux writes
    ``{"quantization": {"method": "mflux", "bits": 4, ...}}``. Uniform bits
    are a ``weights`` quantization; per-layer overrides with different bits
    are ``mixed``; no block means the weights are unquantized (``none``).

    The file is publisher-controlled, so every value is type- and
    range-checked against the contract and anything unexpected degrades to
    ``unknown`` — provenance must never make a run unsavable.
    """
    try:
        return _quantization_facts(config)
    except Exception:  # noqa: BLE001 — see docstring
        return {"kind": "unknown", "base_dtype": "unknown"}


def _bits_x2(value: Any) -> int | None:
    """``bits`` as the contract's x2 integer (2..64), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    doubled = float(value) * 2
    if not doubled.is_integer():
        # 3.3 bpw is not a contract value; recording it as 3.5 would invent
        # a quantization fact.
        return None
    x2 = int(doubled)
    return x2 if 2 <= x2 <= 64 else None


def _quantization_facts(config: dict[str, Any]) -> dict[str, Any]:
    dtype = config.get("torch_dtype")
    if dtype is None:
        dtype = config.get("dtype")
    base_dtype = _BASE_DTYPES.get(str(dtype), "unknown")
    block = config.get("quantization")
    if block is None:
        block = config.get("quantization_config")
    if block is None:
        return {"kind": "none", "base_dtype": base_dtype}
    if not isinstance(block, dict):
        # A declaration that exists but cannot be read is not "unquantized".
        return {"kind": "unknown", "base_dtype": base_dtype}
    bits_x2 = _bits_x2(block.get("bits"))
    if bits_x2 is None:
        return {"kind": "unknown", "base_dtype": base_dtype}
    override_bits_x2 = {
        _bits_x2(value.get("bits"))
        for value in block.values()
        if isinstance(value, dict) and value.get("bits") is not None
    }
    facts: dict[str, Any] = {"base_dtype": base_dtype}
    if override_bits_x2 - {bits_x2}:
        # Per-layer overrides at other bit widths (or unparsable ones):
        # the artifact is not uniformly quantized.
        facts["kind"] = "mixed"
    else:
        facts["kind"] = "weights"
        facts["weight_bits_x2"] = bits_x2
    # mlx-lm writes the scheme as ``mode`` ("affine", "mxfp4", ...); mflux
    # image models write ``method`` ("mflux"). Honour whichever the artifact
    # declares; only a block that names neither is assumed to be mlx-lm's
    # historical affine default.
    declared = None
    for field in ("method", "quant_method", "mode"):
        candidate = block.get(field)
        if isinstance(candidate, str) and candidate:
            declared = candidate
            break
    if isinstance(declared, str) and declared:
        method = declared.strip().lower()
        facts["method"] = method if _METHOD_PATTERN.fullmatch(method) else "other"
    elif facts["kind"] == "weights":
        facts["method"] = "affine"
    group_size = block.get("group_size")
    if (
        isinstance(group_size, int)
        and not isinstance(group_size, bool)
        and 1 <= group_size <= 4096
    ):
        facts["group_size"] = group_size
    return facts


def unresolved_model_identity(
    repo_id: str, task_type: str, subfolder: str | None = None
) -> dict[str, Any]:
    """Identity with every fact the local cache can vouch for.

    ``identity_strength`` stays ``unresolved`` (no manifest digest is
    computed here), but the quantization block and the resolved snapshot
    revision are filled from the cached ``config.json`` so a
    ``...-4bit`` model no longer reports ``quantization.kind: unknown``.
    """
    source: dict[str, Any] = {"kind": "huggingface", "repo_id": repo_id}
    if subfolder:
        source["subfolder"] = subfolder
    quantization: dict[str, Any] = {"kind": "unknown", "base_dtype": "unknown"}
    cached = _cached_config(repo_id, subfolder)
    if cached is not None:
        config, revision = cached
        quantization = quantization_facts(config)
        if revision is not None:
            source["resolved_revision"] = revision
    identity = {
        "schema_version": 1,
        "identity_strength": "unresolved",
        "pipeline_kind": task_type,
        "components": [
            {
                "component_id": "primary",
                "role": "primary",
                "source": source,
                "artifact": {"format": "mlx-safetensors"},
                "quantization": quantization,
            }
        ],
    }
    return identity


def _unknown_conditions() -> dict[str, Any]:
    """Schema-valid placeholder when no snapshot was taken."""
    return {
        "power_source": "unknown",
        "low_power_mode": None,
        "thermal_state": "unknown",
        "memory_pressure": "unknown",
        "available_memory_mib": None,
    }


def machine_observation(
    hardware: Hardware,
    software: Software,
    *,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the atomic machine observation.

    ``before``/``after`` are ``run_conditions()`` snapshots taken by the
    runner around the measured work. A missing ``before`` is probed now (the
    observation is being built, so "now" is the best available "before");
    a missing ``after`` is recorded as unknown rather than re-probed, because
    a snapshot taken after result construction would misreport the run.
    """
    profile = {
        "chip": hardware.chip,
        "memory_gib": hardware.ram_gb,
        "cpu_cores": hardware.cpu_cores,
        "gpu_cores": hardware.gpu_cores,
    }
    return {
        "schema_version": 1,
        "profile_completeness": "partial",
        "profile": profile,
        "profile_digest": rcj_digest(profile),
        "os": {"name": "macOS", "version": software.macos, "architecture": "arm64"},
        "conditions_before": (before if before is not None else run_conditions()),
        "conditions_after": after if after is not None else _unknown_conditions(),
    }


def _installed(name: str, fallback: str | None = None) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return fallback


def _source_checkout_revision(start: Path | None = None) -> str | None:
    """Return HEAD when the imported runtime lives inside a Git checkout."""

    location = (start or Path(__file__)).resolve()
    root = next(
        (
            parent
            for parent in (location.parent, *location.parents)
            if (parent / ".git").exists()
        ),
        None,
    )
    if root is None:
        return None
    try:
        relative = location.relative_to(root)
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                "--",
                str(relative),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        # A wheel installed into a repository-local virtualenv can still have
        # a .git ancestor. It is a release unless this exact imported module
        # belongs to the checkout index.
        if tracked.returncode != 0:
            return None
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("could not resolve the Rapid-MLX source revision") from exc
    revision = result.stdout.strip().lower()
    if (
        result.returncode != 0
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError("could not resolve the Rapid-MLX source revision")
    return revision


def execution_config(
    task_type: str, *, context_length: int | None = None
) -> dict[str, Any]:
    source_revision = _source_checkout_revision()
    runtime: dict[str, Any] = {
        "distribution": "source" if source_revision is not None else "release",
        "rapid_mlx": __version__,
        "mlx": _installed("mlx", "unknown"),
        "python": platform.python_version(),
    }
    if source_revision is not None:
        runtime["rapid_mlx_revision"] = source_revision
    for package, field in (
        ("mlx-lm", "mlx_lm"),
        ("mlx-vlm", "mlx_vlm"),
        ("mflux", "mflux"),
    ):
        installed = _installed(package)
        if installed is not None:
            runtime[field] = installed
    resources = {"max_concurrency": 1, "compute_dtype": "unknown"}
    diffusion = {
        "attention_backend": "unknown",
        "compilation": "unknown",
        "vae_tiling": None,
        "vae_slicing": None,
    }
    if task_type == "text_generation":
        task = {
            "kind": task_type,
            "language": {
                "context_length": context_length,
                "speculative_decoding": {"method": "none"},
                "kv_cache": {"mode": "unknown", "dtype": "unknown"},
                "prefix_cache_enabled": False,
                "prefill_backend": "gpu",
                "prefill_chunk_size": None,
            },
        }
    elif task_type == "image_generation":
        task = {"kind": task_type, "diffusion": diffusion}
    elif task_type == "video_generation":
        task = {
            "kind": task_type,
            "diffusion": diffusion,
            "temporal_chunking": {"enabled": None},
        }
    else:
        raise ValueError(f"unsupported task type {task_type!r}")
    projection = {"task_type": task_type, "resources": resources, "task": task}
    return {
        "schema_version": 1,
        "config_digest": rcj_digest(projection),
        "runtime": runtime,
        **projection,
    }


def build_run(
    *,
    repo_id: str,
    task_type: str,
    hardware: Hardware | None,
    subfolder: str | None = None,
    software: Software | None,
    started_at: str,
    measurements: list[dict[str, Any]] | None = None,
    status: str = "completed",
    failure_code: str | None = None,
    context_length: int | None = None,
    execution: dict[str, Any] | None = None,
    conditions_before: dict[str, Any] | None = None,
    conditions_after: dict[str, Any] | None = None,
    model_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = {"status": status}
    if failure_code is not None:
        outcome["failure_code"] = failure_code
    run: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "started_at": started_at,
        "completed_at": utc_now(),
        "collector": {"name": "rapid-mlx-community-bench", "version": __version__},
        # Callers that resolved the identity before loading pass it in, so
        # the record describes the snapshot that was measured even if the
        # cache moves on (another pull) while the benchmark runs.
        "model": (
            model_identity
            if model_identity is not None
            else unresolved_model_identity(repo_id, task_type, subfolder)
        ),
        "execution": (
            execution
            if execution is not None
            else execution_config(task_type, context_length=context_length)
        ),
        "workload": registered_workload(task_type),
        "outcome": outcome,
    }
    if hardware is not None and software is not None:
        run["machine"] = machine_observation(
            hardware, software, before=conditions_before, after=conditions_after
        )
    if measurements:
        run["measurements"] = measurements
    BenchmarkRunValidator().validate(run)
    return run


__all__ = ["build_run", "execution_config", "machine_observation", "utc_now"]
