# SPDX-License-Identifier: Apache-2.0
"""Model-first planning and private local storage for Community Benchmark."""

from __future__ import annotations

import copy
import heapq
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapid_mlx.byom.preflight import _working_set_budget, catalog_memory_fit
from rapid_mlx.catalog import build_legacy_catalog_snapshot
from rapid_mlx.model_aliases import draft_only_conflict, raise_if_draft_only_model

from .benchmark_contracts import (
    BenchmarkRunValidator,
    SubmissionReceiptValidator,
    registered_workload,
)
from .provenance_schema import ProvenanceInvalid
from .provenance_schema import validate as validate_provenance_document

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
        "qwen-image-2.1",
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


def benchmark_runtime_readiness(alias: str, task_type: str) -> dict[str, Any]:
    """Probe whether this process can execute a registered workload.

    The serving lanes expose their side-effect-free dependency probes so
    planning and execution stay on one contract without loading a model,
    downloading anything, or duplicating dependency rules. The registered
    video protocol admits only Wan aliases; other video families never reach
    this function through the catalog.

    Probe faults are ``unknown`` rather than ``unavailable``. A diagnostic
    must not invent a missing dependency, and ``benchmark run`` retains the
    authoritative fail-closed guard if the environment changes after planning.
    """

    if task_type == "text_generation":
        from rapid_mlx.model_aliases import resolve_profile
        from rapid_mlx.models.deepseek_v41_native.artifacts import (
            REAP,
            is_product_target,
            product_artifact,
            require_pins,
            require_product_memory,
        )

        profile = resolve_profile(alias)
        if profile is not None and is_product_target(profile.hf_path):
            try:
                artifact = product_artifact(profile.hf_path)
                assert artifact is not None
                if artifact == REAP:
                    require_product_memory()
                else:
                    require_product_memory(artifact)
                    require_pins(artifact)
            except RuntimeError as exc:
                return {"status": "unavailable", "message": str(exc)}
            except Exception:
                return {
                    "status": "unknown",
                    "message": (
                        "Runtime readiness could not be verified; run will check again."
                    ),
                }
        return {"status": "ready", "message": None}
    if task_type == "image_generation":
        from rapid_mlx.runtime.image_lane import image_runtime_issue
    elif task_type == "video_generation" and alias in _REGISTERED_WAN_ALIASES:
        from rapid_mlx.runtime.video_lane import registered_wan_runtime_issue
    else:
        return {
            "status": "unknown",
            "message": "Runtime readiness could not be verified; run will check again.",
        }

    try:
        if task_type == "image_generation":
            issue = image_runtime_issue(alias)
        else:
            issue = registered_wan_runtime_issue(alias)
    except Exception:
        return {
            "status": "unknown",
            "message": "Runtime readiness could not be verified; run will check again.",
        }
    if issue:
        return {"status": "unavailable", "message": issue}
    return {"status": "ready", "message": None}


def _primary_task(task_types: list[str]) -> str | None:
    # A multimodal chat alias also advertises vision_language. Text is the
    # registered output-speed protocol until a VLM workload is published.
    for task in ("image_generation", "video_generation", "text_generation"):
        if task in task_types:
            return task
    return None


def benchmark_catalog(
    *,
    memory_gib: int | None = None,
    recommended_working_set_bytes: int | None = None,
    check_cache: bool = False,
) -> dict[str, Any]:
    """Project the atomic catalog into the model-first benchmark picker."""

    snapshot = build_legacy_catalog_snapshot()
    ram_bytes = memory_gib * (1 << 30) if memory_gib else None
    budget = _working_set_budget(ram_bytes, recommended_working_set_bytes)
    models = {item["registry_model_id"]: item for item in snapshot["models"]}
    entries: list[dict[str, Any]] = []
    for alias in snapshot["aliases"]:
        # Serve refuses draft-only checkpoints, so the benchmark catalog never offers them.
        if draft_only_conflict(alias["alias"]) is not None:
            continue
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
        result = catalog_memory_fit(
            alias["alias"],
            ram_bytes=ram_bytes,
            recommended_working_set_bytes=recommended_working_set_bytes,
        )
        estimated_memory_gib = (
            math.ceil(result.estimate_bytes / (1 << 30))
            if result.estimate_bytes is not None
            else None
        )
        fit = (
            "unknown"
            if result.fits is None
            else ("fits" if result.fits else "does_not_fit")
        )
        entries.append(
            {
                "alias": alias["alias"],
                "repo_id": model["source"]["repo_id"],
                # Some repos host several quantisations in subfolders
                # (``LiquidAI/LFM2.5-2.6B-MLX`` -> ``4bit/``); the identity
                # facts must be read from that folder's config, not the root.
                "subfolder": model["source"].get("subfolder"),
                "task_type": task,
                "protocol_id": _TASK_PROTOCOL[task],
                "protocol_version": workload["protocol_version"],
                "focus": alias["alias"] in _FOCUS_ALIASES,
                "estimated_memory_gib": estimated_memory_gib,
                "estimated_memory_bytes": result.estimate_bytes,
                "estimated_memory_gb": round(result.estimate_bytes / (1 << 30), 1)
                if result.estimate_bytes is not None
                else None,
                "memory_estimate_source": result.estimate_source,
                "memory_fit": fit,
                "memory_fit_basis": result.fit_basis,
                "memory_floor_gib": result.floor_gib,
                "identity_strength": alias["target"]["resolution_status"],
                "comparable": alias["target"]["resolution_status"] != "unresolved",
                "runtime": benchmark_runtime_readiness(alias["alias"], task),
            }
        )
        if check_cache:
            entries[-1]["model_cached"] = model_is_cached(entries[-1]["repo_id"])
    entries.sort(key=lambda item: (not item["focus"], item["alias"]))
    return {
        "schema_version": 1,
        "catalog_digest": snapshot["catalog_digest"],
        "memory_budget_gib": round(budget / (1 << 30), 1)
        if budget is not None
        else None,
        "models": entries,
    }


def describe_case(case: dict[str, Any]) -> str:
    """One human-readable line for a registered workload case.

    Text cases read ``pp512-tg128   512 prompt -> 128 output tokens``; image
    and video cases describe their geometry. Every line ends with the round
    plan so a user knows how much work one case is before starting it.
    """

    case_id = str(case.get("case_id", "?"))
    if "target_prompt_tokens" in case:
        detail = (
            f"{case['target_prompt_tokens']} prompt tokens -> "
            f"{case['target_output_tokens']} output tokens"
        )
    elif "frames" in case:
        fps = case.get("fps_milli", 0) / 1000
        detail = (
            f"{case['width']}x{case['height']}, {case['frames']} frames @ {fps:g} fps, "
            f"{case['steps']} steps"
        )
    elif "width" in case:
        count = case.get("image_count", 1)
        detail = (
            f"{case['width']}x{case['height']}, {case['steps']} steps, "
            f"{count} image{'s' if count != 1 else ''}"
        )
    else:  # pragma: no cover - every registered protocol has a known shape
        detail = "registered case"
    warmup = int(case.get("warmup_rounds", 0))
    measured = int(case.get("measured_rounds", 0))
    return f"{case_id:<16} {detail}   ({warmup} warmup + {measured} measured)"


def model_is_cached(repo_id: str) -> bool | None:
    """Tri-state: the weights are complete in the local cache (``True``),
    definitively absent (``False``), or the probe was inconclusive (``None``).

    Delegates to the modality-aware probe behind ``rapid-mlx models
    --cached`` so mflux image and Wan/split video checkpoints, whose weights
    live in component subdirectories, are judged by their own layout rather
    than by the text-only ``model*.safetensors`` rule. Never downloads.
    """

    try:
        from rapid_mlx.cli import _cache_runnability

        verdict = _cache_runnability(repo_id)
    except Exception:
        return None
    return verdict if isinstance(verdict, bool) else None


def plan_for_alias(
    alias_name: str,
    *,
    memory_gib: int | None = None,
    check_cache: bool = False,
    recommended_working_set_bytes: int | None = None,
) -> dict[str, Any]:
    """Describe the exact local workload for one alias.

    ``memory_gib`` fills the model's ``memory_fit`` the same way the catalog
    does. ``check_cache`` adds ``model_cached`` (``True``/``False``, or
    ``None`` when the cache could not be probed); it touches only the local
    filesystem and never downloads anything.
    """

    raise_if_draft_only_model(alias_name, action="Benchmark")
    catalog = benchmark_catalog(
        memory_gib=memory_gib,
        recommended_working_set_bytes=recommended_working_set_bytes,
    )
    for entry in catalog["models"]:
        if entry["alias"] == alias_name:
            plan = {
                "schema_version": 1,
                "memory_budget_gib": catalog["memory_budget_gib"],
                "model": entry,
                "workload": registered_workload(entry["task_type"]),
                "privacy": {
                    "storage": "local",
                    "uploads": False,
                    "upload": "explicit_consent_only",
                },
            }
            if memory_gib is not None:
                plan["memory_gib"] = memory_gib
            if check_cache:
                plan["model_cached"] = model_is_cached(entry["repo_id"])
            return plan
    raise ValueError(f"unknown or unsupported benchmark model {alias_name!r}")


class ProvenanceUnreadable(RuntimeError):  # noqa: N818 - stable domain error name
    """A run's build provenance exists but cannot be trusted.

    Deliberately distinct from ``None``. ``None`` means the file is not there,
    which for a run archived before provenance existed is an honest answer.
    This means the file *is* there and is damaged, unreadable or malformed —
    and a damaged file must never be quietly downgraded to "no claim", because
    that is how corruption becomes permission to publish.
    """


def validate_provenance(
    provenance: Any, *, run_id: str | None = None
) -> dict[str, Any]:
    """Check a stored provenance document against the closed schema.

    Delegates to :mod:`provenance_schema` rather than restating the rules:
    this reader and the one in ``run_builder`` used to disagree about unknown
    fields and about a revision on a release, so a document one refused the
    other accepted. One definition, applied at every boundary.
    """

    where = f" for {run_id}" if run_id else ""
    try:
        return validate_provenance_document(
            provenance, label=f"the build provenance{where}"
        )
    except ProvenanceInvalid as exc:
        raise ProvenanceUnreadable(str(exc)) from exc


@dataclass(frozen=True)
class LocalRunArchive:
    """Private, atomic JSON run archive. Reading never executes or uploads."""

    root: Path
    fallback_root: Path | None = None

    @classmethod
    def default(cls) -> LocalRunArchive:
        override = os.environ.get("RAPID_MLX_BENCHMARK_HOME")
        legacy = Path.home() / ".rapid-mlx" / "benchmarks"
        if override:
            return cls(Path(override).expanduser())
        home = os.environ.get("RAPID_MLX_HOME", "").strip()
        if home:
            root = Path(home).expanduser() / "benchmarks"
            return cls(
                root,
                fallback_root=legacy if root.resolve() != legacy.resolve() else None,
            )
        return cls(legacy)

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not run_id or any(
            character not in "0123456789abcdef-" for character in run_id
        ):
            raise ValueError("invalid run id")

    def _read_text(self, directory: Path, name: str, category: str) -> str:
        try:
            return (directory / f"{name}.json").read_text(encoding="utf-8")
        except FileNotFoundError:
            if self.fallback_root is None:
                raise
            return (self.fallback_root / category / f"{name}.json").read_text(
                encoding="utf-8"
            )

    def _run_metadata_dir(self, run_id: str, category: str) -> Path:
        if self.fallback_root is not None:
            self._validate_run_id(run_id)
            if (
                not (self.runs_dir / f"{run_id}.json").exists()
                and (self.fallback_root / "runs" / f"{run_id}.json").exists()
            ):
                return self.fallback_root / category
        return self.root / category

    def _read_run_metadata(self, run_id: str, category: str) -> str:
        if self.fallback_root is not None:
            self._validate_run_id(run_id)
        try:
            return (self.root / category / f"{run_id}.json").read_text(encoding="utf-8")
        except FileNotFoundError:
            return (
                self._run_metadata_dir(run_id, category) / f"{run_id}.json"
            ).read_text(encoding="utf-8")

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def receipts_dir(self) -> Path:
        return self.root / "receipts"

    @property
    def provenance_dir(self) -> Path:
        """How the runtime that produced each run was built.

        Kept beside the run rather than inside it: the run document is a closed
        schema, and the wire contract it projects to has no field for a tree
        that is not a commit. It has to be stored *per run* so the fact travels
        with the result — rebuilding the app cleanly tomorrow must not make a
        benchmark measured by a modified build retroactively publishable.
        """

        return self.root / "provenance"

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
        """Archive a run with no provenance of its own.

        Only for callers that genuinely have none to record — tests and the
        legacy path. Everything that measures uses
        :meth:`save_with_provenance`, because a run visible without its
        provenance is a run that can be published without one.
        """

        BenchmarkRunValidator().validate(run)
        return self._atomic_save(self.runs_dir, run["run_id"], run)

    def save_with_provenance(
        self, run: dict[str, Any], provenance: dict[str, Any]
    ) -> Path:
        """Archive a run and the build that produced it, provenance first.

        Ordering is the whole point. A run file is what makes a result visible
        to ``benchmark results`` and publishable by ``benchmark share``; the
        provenance file is what can forbid publishing it. Writing the run
        first leaves a window — and, if the provenance write then fails, a
        permanent state — in which a result produced by a modified build looks
        exactly like a legacy run and is allowed to publish.

        So provenance is written and fsynced first, and any failure propagates
        before the run is written at all. The result is that a missing
        provenance file can only mean "archived before this existed", never
        "we tried and could not".

        The reverse leftover — a provenance file with no run — is harmless:
        nothing reads it without a run, and it is overwritten when that run id
        is next archived.
        """

        BenchmarkRunValidator().validate(run)
        run_id = run["run_id"]
        # Validated before either write, so a malformed provenance document
        # cannot be the thing that is persisted.
        validate_provenance(provenance)
        self.save_provenance(run_id, provenance)
        return self._atomic_save(self.runs_dir, run_id, run)

    def save_provenance(self, run_id: str, provenance: dict[str, Any]) -> Path:
        """Record how the runtime that produced ``run_id`` was built."""

        return self._atomic_save(self.provenance_dir, run_id, provenance)

    def provenance(self, run_id: str) -> dict[str, Any] | None:
        """The stored build provenance for a run.

        ``None`` means the file is genuinely absent — a run archived before
        provenance existed. That is the one case a caller may treat as "no
        claim either way".

        Anything else raises. A file that exists but cannot be read, is not
        JSON, is not an object, or does not satisfy the provenance shape is
        **corruption**, and corruption must not be indistinguishable from a
        legacy run: that turns a damaged or tampered file into permission to
        publish.
        """

        try:
            raw = self._read_run_metadata(run_id, "provenance")
        except FileNotFoundError:
            return None
        except OSError as exc:
            # Permissions, an I/O error, a directory where a file should be.
            raise ProvenanceUnreadable(
                f"the build provenance for {run_id} could not be read: {exc}"
            ) from exc
        try:
            stored = json.loads(raw)
        except ValueError as exc:
            raise ProvenanceUnreadable(
                f"the build provenance for {run_id} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(stored, dict):
            raise ProvenanceUnreadable(
                f"the build provenance for {run_id} is not a JSON object"
            )
        validate_provenance(stored, run_id=run_id)
        return stored

    def save_receipt(self, receipt: dict[str, Any], *, install_id: str) -> Path:
        SubmissionReceiptValidator().validate(receipt)
        run_id = receipt.get("submission_id")
        if not isinstance(run_id, str):
            raise ValueError("receipt is missing a submission id")
        # A receipt may only exist for a locally archived run. This prevents a
        # forged receipt file from making an unrelated row look shared.
        run = self.get(run_id)
        from .atomic_upload import atomic_run_digest
        from .publication import project_run_for_publication

        wire, _ = project_run_for_publication(copy.deepcopy(run))
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
        try:
            envelope = json.loads(self._read_run_metadata(run_id, "receipts"))
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
        from .publication import project_run_for_publication

        wire, _ = project_run_for_publication(copy.deepcopy(self.get(run_id)))
        wire["install_id"] = install_id
        if atomic_run_digest(wire) != value["run_digest"]:
            return None
        return value

    def get(self, run_id: str) -> dict[str, Any]:
        self._validate_run_id(run_id)
        run = json.loads(self._read_text(self.runs_dir, run_id, "runs"))
        if not isinstance(run, dict):
            raise ValueError("benchmark run archive entry is not a JSON object")
        BenchmarkRunValidator().validate(run)
        return run

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and limit < 1:
            raise ValueError("result limit must be positive")
        runs: list[dict[str, Any]] = []
        latest: list[tuple[str, str, dict[str, Any]]] = []
        directories = [self.runs_dir]
        if self.fallback_root is not None:
            directories.append(self.fallback_root / "runs")
        seen: set[str] = set()
        paths = (path for directory in directories for path in directory.glob("*.json"))
        for path in paths:
            if path.stem in seen:
                continue
            seen.add(path.stem)
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
