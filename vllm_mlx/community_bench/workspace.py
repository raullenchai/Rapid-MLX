# SPDX-License-Identifier: Apache-2.0
"""Model-first planning and private local storage for Community Benchmark."""

from __future__ import annotations

import heapq
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_mlx.catalog import build_legacy_catalog_snapshot

from .benchmark_contracts import BenchmarkRunValidator, registered_workload

_TASK_PROTOCOL = {
    "text_generation": "rapid-community-speed",
    "image_generation": "rapid-image-speed",
    "video_generation": "rapid-video-speed",
}
_FOCUS_ALIASES = frozenset(
    {
        "qwen3.8-27b-4bit",
        "qwen3.5-9b-4bit",
        "gemma-4-e4b-4bit",
        "flux2-klein-4b",
        "z-image-turbo",
        "qwen-image",
        "wan2.2-ti2v-5b-q8",
    }
)
_REGISTERED_WAN_ALIASES = frozenset(
    {
        "wan2.2-t2v-a14b-bf16",
        "wan2.2-ti2v-5b-bf16",
        "wan2.2-ti2v-5b-q8",
    }
)


def _primary_task(task_types: list[str]) -> str | None:
    # A multimodal chat alias also advertises vision_language. Text is the
    # registered output-speed protocol until a VLM workload is published.
    for task in ("image_generation", "video_generation", "text_generation"):
        if task in task_types:
            return task
    return None


def benchmark_catalog(*, memory_gib: int | None = None) -> dict[str, Any]:
    """Project the atomic catalog into the model-first benchmark picker."""

    snapshot = build_legacy_catalog_snapshot()
    # This is the shadow-migration bridge until minimum-memory evidence is
    # projected into the atomic recommendation layer for every alias.
    from vllm_mlx.model_aliases import list_profiles

    profiles = list_profiles()
    models = {item["registry_model_id"]: item for item in snapshot["models"]}
    entries: list[dict[str, Any]] = []
    for alias in snapshot["aliases"]:
        task_types = list(alias["capabilities"]["task_types"])
        task = _primary_task(task_types)
        if task is None:
            continue
        operations = alias["capabilities"]["operation_modes"]
        if task == "image_generation" and "text_to_image" not in operations:
            continue
        # rapid-video-speed-v1 is deliberately a Wan protocol. LTX and
        # CogVideoX use different native step/resolution contracts and need
        # their own registered workload before they can produce comparable rows.
        if task == "video_generation" and alias["alias"] not in _REGISTERED_WAN_ALIASES:
            continue
        model = models[alias["target"]["registry_model_id"]]
        workload = registered_workload(task)
        size = model.get("estimated_download_size_bytes")
        estimated_memory_gib = None
        estimate_source = "unknown"
        profile = profiles.get(alias["alias"])
        minimum_memory = getattr(profile, "min_memory_gb", None)
        if isinstance(minimum_memory, int | float) and minimum_memory > 0:
            estimated_memory_gib = int(minimum_memory + 0.999999)
            estimate_source = "profile_minimum"
        elif isinstance(size, int):
            # This is deliberately a planning estimate, not benchmark evidence.
            estimated_memory_gib = max(1, (size + (1 << 30) - 1) // (1 << 30) + 2)
            estimate_source = "artifact_size_fallback"
        fit = "unknown"
        if memory_gib is not None and estimated_memory_gib is not None:
            fit = "fits" if estimated_memory_gib <= memory_gib else "does_not_fit"
        entries.append(
            {
                "alias": alias["alias"],
                "repo_id": model["source"]["repo_id"],
                "task_type": task,
                "protocol_id": _TASK_PROTOCOL[task],
                "protocol_version": workload["protocol_version"],
                "focus": alias["alias"] in _FOCUS_ALIASES,
                "estimated_memory_gib": estimated_memory_gib,
                "memory_estimate_source": estimate_source,
                "memory_fit": fit,
                "identity_strength": alias["target"]["resolution_status"],
                "comparable": alias["target"]["resolution_status"] != "unresolved",
            }
        )
    entries.sort(key=lambda item: (not item["focus"], item["alias"]))
    return {
        "schema_version": 1,
        "catalog_digest": snapshot["catalog_digest"],
        "models": entries,
    }


def plan_for_alias(alias_name: str) -> dict[str, Any]:
    catalog = benchmark_catalog()
    for entry in catalog["models"]:
        if entry["alias"] == alias_name:
            return {
                "schema_version": 1,
                "model": entry,
                "workload": registered_workload(entry["task_type"]),
                "privacy": {"storage": "local", "uploads": False},
            }
    raise ValueError(f"unknown or unsupported benchmark model {alias_name!r}")


@dataclass(frozen=True)
class LocalRunArchive:
    """Private, atomic JSON run archive. Reading never executes or uploads."""

    root: Path

    @classmethod
    def default(cls) -> LocalRunArchive:
        override = os.environ.get("RAPID_MLX_BENCHMARK_HOME")
        root = (
            Path(override).expanduser()
            if override
            else Path.home() / ".rapid-mlx" / "benchmarks"
        )
        return cls(root)

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def save(self, run: dict[str, Any]) -> Path:
        BenchmarkRunValidator().validate(run)
        self.runs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.runs_dir, 0o700)
        target = self.runs_dir / f"{run['run_id']}.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{run['run_id']}.", suffix=".tmp", dir=self.runs_dir
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(run, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, target)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        return target

    def get(self, run_id: str) -> dict[str, Any]:
        if not run_id or any(
            character not in "0123456789abcdef-" for character in run_id
        ):
            raise ValueError("invalid run id")
        run = json.loads((self.runs_dir / f"{run_id}.json").read_text(encoding="utf-8"))
        if not isinstance(run, dict):
            raise ValueError("benchmark run archive entry is not a JSON object")
        BenchmarkRunValidator().validate(run)
        return run

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and limit < 1:
            raise ValueError("result limit must be positive")
        if not self.runs_dir.exists():
            return []
        runs: list[dict[str, Any]] = []
        latest: list[tuple[str, str, dict[str, Any]]] = []
        for path in self.runs_dir.glob("*.json"):
            try:
                run = self.get(path.stem)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if limit is None:
                runs.append(run)
                continue
            # The filename is unique even if a hand-edited archive duplicates
            # the embedded run_id, so heap comparisons never fall through to
            # comparing the unorderable run dictionaries.
            item = (run["started_at"], path.stem, run)
            if len(latest) < limit:
                heapq.heappush(latest, item)
            elif item[:2] > latest[0][:2]:
                heapq.heapreplace(latest, item)
        if limit is not None:
            return [item[2] for item in sorted(latest, reverse=True)]
        return sorted(runs, key=lambda run: run["started_at"], reverse=True)
