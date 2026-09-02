# SPDX-License-Identifier: Apache-2.0
"""Installed Community Benchmark contracts and registered workload lookup."""

from __future__ import annotations

import copy
import json
from importlib import resources
from typing import Any

import jsonschema
import referencing

from vllm_mlx.catalog import rcj_digest
from vllm_mlx.catalog.validation import CatalogValidationError, ContractValidator

_PROTOCOL_FILES = {
    "text_generation": "rapid-community-speed-v2.json",
    "image_generation": "rapid-image-speed-v1.json",
    "video_generation": "rapid-video-speed-v1.json",
}
_PROTOCOL_HISTORY = {
    "text_generation": (
        "rapid-community-speed-v1.json",
        "rapid-community-speed-v2.json",
    ),
    "image_generation": ("rapid-image-speed-v1.json",),
    "video_generation": ("rapid-video-speed-v1.json",),
}


def _read_json(name: str) -> dict[str, Any]:
    # Installed projections share the already-packaged atomic contract
    # resource bundle. Top-level ``proto/`` remains the cross-product SSOT;
    # tests pin every projection byte-for-byte to that source.
    root = resources.files("vllm_mlx.catalog.schemas")
    loaded = json.loads(root.joinpath(name).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"packaged contract {name!r} is not a JSON object")
    return loaded


def registered_workload(task_type: str) -> dict[str, Any]:
    """Return an isolated copy of the registered workload for ``task_type``."""

    try:
        filename = _PROTOCOL_FILES[task_type]
    except KeyError as exc:
        raise ValueError(
            f"no registered community benchmark for {task_type!r}"
        ) from exc
    return copy.deepcopy(_read_json(filename)["workload"])


def registered_workload_history(task_type: str) -> list[dict[str, Any]]:
    """All immutable protocol versions accepted for local archive reads."""

    try:
        filenames = _PROTOCOL_HISTORY[task_type]
    except KeyError as exc:
        raise ValueError(
            f"no registered community benchmark for {task_type!r}"
        ) from exc
    return [copy.deepcopy(_read_json(name)["workload"]) for name in filenames]


def public_prompt(case_id: str) -> str:
    dataset = _read_json("rapid-public-prompts-v1.json")
    for case in dataset["cases"]:
        if case["case_id"] == case_id:
            return str(case["prompt"])
    raise ValueError(f"no registered prompt for case {case_id!r}")


class BenchmarkRunValidator:
    """Validate a run and pin registered claims to their packaged workload."""

    def __init__(self) -> None:
        schema = _read_json("benchmark-run.schema.json")
        catalog_root = resources.files("vllm_mlx.catalog.schemas")
        atomic = [
            json.loads(catalog_root.joinpath(name).read_text(encoding="utf-8"))
            for name in (
                "model-identity.schema.json",
                "machine-observation.schema.json",
                "execution-config.schema.json",
            )
        ]
        registry = referencing.Registry().with_resources(
            (item["$id"], referencing.Resource.from_contents(item))
            for item in [schema, *atomic]
        )
        self._validator = jsonschema.Draft202012Validator(
            schema,
            registry=registry,
            format_checker=jsonschema.FormatChecker(),
        )
        self._atomic = ContractValidator()

    def validate(self, run: dict[str, Any]) -> None:
        failures = sorted(
            self._validator.iter_errors(run),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        if failures:
            failure = failures[0]
            path = "/".join(str(part) for part in failure.absolute_path)
            raise CatalogValidationError("benchmark_run", path, failure.message)

        self._atomic.validate_model_identity(run["model"])
        if "machine" in run:
            self._atomic.validate("machine_observation", run["machine"])
        self._atomic.validate("execution_config", run["execution"])

        if "machine" in run:
            machine = run["machine"]
            if machine["profile_digest"] != rcj_digest(machine["profile"]):
                raise CatalogValidationError(
                    "benchmark_run", "machine/profile_digest", "does not match profile"
                )
        execution = run["execution"]
        execution_projection = {
            "task_type": execution["task_type"],
            "resources": execution["resources"],
            "task": execution["task"],
        }
        if execution["config_digest"] != rcj_digest(execution_projection):
            raise CatalogValidationError(
                "benchmark_run",
                "execution/config_digest",
                "does not match effective task and resources",
            )

        workload = run["workload"]
        if workload["protocol_strength"] == "registered":
            expected = registered_workload_history(workload["task_type"])
            if workload not in expected:
                raise CatalogValidationError(
                    "benchmark_run",
                    "workload",
                    "registered workload differs from every packaged protocol version",
                )

        if run["outcome"]["status"] == "completed":
            measurements = run["measurements"]
            pairs = [(item["case_id"], item["round_index"]) for item in measurements]
            if len(pairs) != len(set(pairs)):
                raise CatalogValidationError(
                    "benchmark_run", "measurements", "case/round pairs must be unique"
                )
            by_case = {case["case_id"]: case for case in workload["cases"]}
            expected_pairs = {
                (case["case_id"], round_index)
                for case in workload["cases"]
                for round_index in range(1, case["measured_rounds"] + 1)
            }
            if set(pairs) != expected_pairs:
                raise CatalogValidationError(
                    "benchmark_run",
                    "measurements",
                    "does not contain exactly the declared measured rounds",
                )
            for index, measurement in enumerate(measurements):
                case = by_case[measurement["case_id"]]
                for field in ("width", "height", "frames", "image_count"):
                    if field in measurement and measurement[field] != case[field]:
                        raise CatalogValidationError(
                            "benchmark_run",
                            f"measurements/{index}/{field}",
                            "does not match the registered case",
                        )
                for measured_field, target_field in (
                    ("prompt_tokens", "target_prompt_tokens"),
                    ("output_tokens", "target_output_tokens"),
                ):
                    if (
                        measured_field in measurement
                        and target_field in case
                        and measurement[measured_field] != case[target_field]
                    ):
                        raise CatalogValidationError(
                            "benchmark_run",
                            f"measurements/{index}/{measured_field}",
                            f"does not match registered {target_field}",
                        )
                phases = sum(
                    float(measurement.get(field, 0))
                    for field in (
                        "ttft_ms",
                        "decode_duration_ms",
                        "media_encode_duration_ms",
                        "text_encode_duration_ms",
                        "denoise_duration_ms",
                        "vae_decode_duration_ms",
                    )
                )
                if phases > float(measurement["total_duration_ms"]) + 1:
                    raise CatalogValidationError(
                        "benchmark_run",
                        f"measurements/{index}/total_duration_ms",
                        "is shorter than its measured phases",
                    )
