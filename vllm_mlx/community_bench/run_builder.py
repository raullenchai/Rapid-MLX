# SPDX-License-Identifier: Apache-2.0
"""Build atomic identities and the BenchmarkRun v1 envelope."""

from __future__ import annotations

import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from vllm_mlx import __version__
from vllm_mlx.catalog import rcj_digest

from .benchmark_contracts import BenchmarkRunValidator, registered_workload
from .hardware import Hardware, Software


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def unresolved_model_identity(repo_id: str, task_type: str) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "identity_strength": "unresolved",
        "pipeline_kind": task_type,
        "components": [
            {
                "component_id": "primary",
                "role": "primary",
                "source": {"kind": "huggingface", "repo_id": repo_id},
                "artifact": {"format": "mlx-safetensors"},
                "quantization": {"kind": "unknown", "base_dtype": "unknown"},
            }
        ],
    }
    return identity


def _conditions() -> dict[str, Any]:
    return {
        "power_source": "unknown",
        "low_power_mode": None,
        "thermal_state": "unknown",
        "memory_pressure": "unknown",
        "available_memory_mib": None,
    }


def machine_observation(
    hardware: Hardware, software: Software, *, after: dict[str, Any] | None = None
) -> dict[str, Any]:
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
        "conditions_before": _conditions(),
        "conditions_after": after or _conditions(),
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
    software: Software | None,
    started_at: str,
    measurements: list[dict[str, Any]] | None = None,
    status: str = "completed",
    failure_code: str | None = None,
    context_length: int | None = None,
    execution: dict[str, Any] | None = None,
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
        "model": unresolved_model_identity(repo_id, task_type),
        "execution": (
            execution
            if execution is not None
            else execution_config(task_type, context_length=context_length)
        ),
        "workload": registered_workload(task_type),
        "outcome": outcome,
    }
    if hardware is not None and software is not None:
        run["machine"] = machine_observation(hardware, software)
    if measurements:
        run["measurements"] = measurements
    BenchmarkRunValidator().validate(run)
    return run


__all__ = ["build_run", "execution_config", "machine_observation", "utc_now"]
