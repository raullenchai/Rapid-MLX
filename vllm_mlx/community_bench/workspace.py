# SPDX-License-Identifier: Apache-2.0
"""Model-first planning and private local storage for Community Benchmark."""

from __future__ import annotations

import copy
import heapq
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_mlx.catalog import build_legacy_catalog_snapshot

from .benchmark_contracts import (
    BenchmarkRunValidator,
    SubmissionReceiptValidator,
    registered_workload,
)

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


#: Unified memory the benchmark must leave to macOS, the display server and
#: the KV cache of the 2048-token case. A 27B 4-bit model whose weights alone
#: are 18 GB used to be reported as "fits" on an 18 GB Mac; running it swaps
#: the machine to a halt, which is the one outcome a planning column exists
#: to prevent.
_HEADROOM_FLOOR_GIB = 2
_HEADROOM_FRACTION = 0.10

#: Alias fragments that mark a paired sidecar rather than a standalone model.
_SIDECAR_MARKERS = ("assistant", "draft")

_PARAM_COUNT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)b(?![a-z0-9])")
_BIT_WIDTH = re.compile(r"(\d+(?:\.\d+)?)(?:bit|bpw)")


def _parameter_floor_gib(alias: str) -> int | None:
    """Lower bound on the working set from the alias's own name.

    ``qwen3.6-35b-mtp-4bit`` names 35 B parameters at 4 bits: at least
    35 × 0.5 GB of weights, before any activations or cache. Catalog
    download sizes are sometimes wrong for variant repos (an MTP head listed
    as 3 GB for a 35 B model); this floor keeps such rows from being called
    a fit. An alias that names no bit width is assumed 4-bit (the catalog's
    usual default variant) so the floor stays a lower bound; ``bf16``/``fp16``
    names count as 16-bit. Where an alias names several sizes
    (``lfm2.5-8b-a1b``) the largest wins: all experts must be resident for
    a benchmark.
    """
    lowered = alias.lower()
    # Speculative-decoding drafters are named after the model they pair with
    # (``gemma-4-31b-assistant`` is a 0.4 B four-layer drafter), so the size
    # in their name says nothing about their own weights.
    if any(marker in lowered for marker in _SIDECAR_MARKERS):
        return None
    counts = [float(m) for m in _PARAM_COUNT.findall(lowered)]
    if not counts:
        return None
    params_b = max(counts)
    bits_match = _BIT_WIDTH.search(lowered)
    bits = float(bits_match.group(1)) if bits_match else 4.0
    if "bf16" in lowered or "fp16" in lowered or "-f16" in lowered:
        bits = 16.0
    weights_gib = params_b * bits / 8.0
    return int(math.ceil(weights_gib * 1.1 + 1))


def curated_footprints() -> dict[str, float]:
    """``alias -> working-set GB`` from the recommendation tiers, read once.

    The picker's curated footprints are the best planning number we have
    for the models they cover; reading the tiers once per catalog build
    avoids re-validating them for every alias.
    """
    try:
        from vllm_mlx.recommendations import load_recommendation_tiers

        tiers = load_recommendation_tiers()
    except Exception:  # noqa: BLE001 — planning data must never fail the catalog
        return {}
    footprints: dict[str, float] = {}
    for tier in tiers:
        for pick in tier.picks:
            footprint = getattr(pick, "footprint_gb", None)
            if isinstance(footprint, int | float) and footprint > 0:
                footprints.setdefault(pick.alias.casefold(), float(footprint))
    return footprints


def estimate_memory_gib(
    alias: str,
    *,
    minimum_memory_gb: float | None,
    download_size_bytes: int | None,
    footprints: dict[str, float] | None = None,
) -> tuple[int | None, str]:
    """Planning estimate with its provenance.

    Precedence: an explicit profile minimum (``profile_minimum`` — the
    smallest *total* machine memory the profile admits, so it is compared
    against the host directly), then the curated recommendation footprint
    (the working set the model picker already shows), then the artifact size
    plus activations, never below the parameter-count floor. The last three
    are working-set estimates and get OS/KV headroom in ``memory_fit``.
    """
    if isinstance(minimum_memory_gb, int | float) and minimum_memory_gb > 0:
        return int(minimum_memory_gb + 0.999999), "profile_minimum"
    if footprints is None:
        footprints = curated_footprints()
    floor = _parameter_floor_gib(alias)
    footprint = footprints.get(alias.casefold())
    if isinstance(footprint, int | float) and footprint > 0:
        estimate = int(math.ceil(footprint))
        # Curated numbers are hand-maintained; the parameter floor guards
        # against a stale one just like it guards the artifact size.
        if floor is not None and floor > estimate:
            return floor, "parameter_count_floor"
        return estimate, "curated_footprint"
    if isinstance(download_size_bytes, int):
        # This is deliberately a planning estimate, not benchmark evidence.
        estimate = max(1, (download_size_bytes + (1 << 30) - 1) // (1 << 30) + 2)
        if floor is not None and floor > estimate:
            return floor, "parameter_count_floor"
        return estimate, "artifact_size_fallback"
    if floor is not None:
        return floor, "parameter_count_floor"
    return None, "unknown"


def memory_fit(
    estimated_memory_gib: int | None,
    memory_gib: int | None,
    source: str = "artifact_size_fallback",
) -> str:
    """``fits`` only when the estimate leaves the required headroom.

    A ``profile_minimum`` is already a whole-machine floor (existing launch
    logic admits ``total_ram >= min_memory_gb``), so it is compared directly;
    every working-set estimate must additionally leave room for macOS and
    the 2048-token KV cache.
    """
    if memory_gib is None or estimated_memory_gib is None:
        return "unknown"
    if source == "profile_minimum":
        return "fits" if estimated_memory_gib <= memory_gib else "does_not_fit"
    headroom = max(_HEADROOM_FLOOR_GIB, math.ceil(memory_gib * _HEADROOM_FRACTION))
    return "fits" if estimated_memory_gib + headroom <= memory_gib else "does_not_fit"


def benchmark_catalog(*, memory_gib: int | None = None) -> dict[str, Any]:
    """Project the atomic catalog into the model-first benchmark picker."""

    snapshot = build_legacy_catalog_snapshot()
    # This is the shadow-migration bridge until minimum-memory evidence is
    # projected into the atomic recommendation layer for every alias.
    from vllm_mlx.model_aliases import list_profiles

    profiles = list_profiles()
    footprints = curated_footprints()
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
        profile = profiles.get(alias["alias"])
        estimated_memory_gib, estimate_source = estimate_memory_gib(
            alias["alias"],
            minimum_memory_gb=getattr(profile, "min_memory_gb", None),
            download_size_bytes=model.get("estimated_download_size_bytes"),
            footprints=footprints,
        )
        fit = memory_fit(estimated_memory_gib, memory_gib, estimate_source)
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
                "privacy": {
                    "storage": "local",
                    "uploads": False,
                    "upload": "explicit_consent_only",
                },
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

    @property
    def receipts_dir(self) -> Path:
        return self.root / "receipts"

    @staticmethod
    def _atomic_save(directory: Path, name: str, value: dict[str, Any]) -> Path:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        target = directory / f"{name}.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
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

    def save(self, run: dict[str, Any]) -> Path:
        BenchmarkRunValidator().validate(run)
        return self._atomic_save(self.runs_dir, run["run_id"], run)

    def save_receipt(self, receipt: dict[str, Any], *, install_id: str) -> Path:
        SubmissionReceiptValidator().validate(receipt)
        run_id = receipt.get("submission_id")
        if not isinstance(run_id, str):
            raise ValueError("receipt is missing a submission id")
        # A receipt may only exist for a locally archived run. This prevents a
        # forged receipt file from making an unrelated row look shared.
        run = self.get(run_id)
        from .atomic_upload import atomic_run_digest

        wire = copy.deepcopy(run)
        wire["install_id"] = install_id
        if atomic_run_digest(wire) != receipt["run_digest"]:
            raise ValueError("receipt does not identify the current archived run")
        envelope = {
            "schema_version": 1,
            "install_id": install_id,
            "receipt": receipt,
        }
        return self._atomic_save(self.receipts_dir, run_id, envelope)

    def receipt(self, run_id: str) -> dict[str, Any] | None:
        # Reuse the run-id validation and require that the result still exists.
        self.get(run_id)
        path = self.receipts_dir / f"{run_id}.json"
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
            return None
        install_id = envelope.get("install_id")
        value = envelope.get("receipt")
        if (
            not isinstance(install_id, str)
            or not isinstance(value, dict)
            or value.get("submission_id") != run_id
        ):
            return None
        try:
            SubmissionReceiptValidator().validate(value)
        except ValueError:
            return None
        from .atomic_upload import atomic_run_digest

        wire = copy.deepcopy(self.get(run_id))
        wire["install_id"] = install_id
        if atomic_run_digest(wire) != value["run_digest"]:
            return None
        return value

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
