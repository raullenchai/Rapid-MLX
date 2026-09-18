# SPDX-License-Identifier: Apache-2.0
"""A Python mirror of the deployed ingestion validator.

The service that accepts Community Benchmark submissions is a Cloudflare
Worker (``landing/src/index.js`` in the rapidmlx.com repository); its rules
cannot be imported from here. This module restates the parts that decide
whether a submission is accepted, so a test can validate the **exact** bytes
``preview_run`` would send rather than the more permissive local schema.

Mirrored from ``atomicValidateModel`` / ``atomicValidateRun`` at commit
``519fd61`` ("site: redesign community leaderboard and contributor profiles").
The load-bearing detail is ``atomicExactObject(value, allowed, required)``:
``allowed`` and ``required`` are the same list at every call site, so a key
that is not listed is rejected outright — which is how a perfectly good local
record became unpublishable.

:func:`cross_check_against_worker` re-derives the allowlists from the worker
source when a checkout happens to be present, so this mirror cannot drift
unnoticed on a machine that has one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

#: Checkouts a developer may have. Absent on CI, which is why the mirror
#: below is the primary authority for the test suite.
_WORKER_CANDIDATES = (
    Path.home()
    / "Documents/Github/rapidmlx.com/.claude/worktrees/codex+leaderboard-redesign"
    / "landing/src/index.js",
    Path.home() / "Documents/Github/rapidmlx.com/landing/src/index.js",
)


class IngestionRejected(Exception):  # noqa: N818 - mirrors the service contract name
    """The worker would refuse this submission, with its own message."""


def _exact_object(
    value: Any,
    allowed: tuple[str, ...],
    label: str,
    required: tuple[str, ...] | None = None,
) -> None:
    """Mirror of ``atomicExactObject(value, allowed, required, label)``.

    ``required`` defaults to ``allowed``, which is how every model-block call
    site uses it; ``execution.runtime`` is the one that passes a shorter
    required list.
    """

    if not isinstance(value, dict):
        raise IngestionRejected(f"{label} must be an object")
    unknown = next((key for key in value if key not in allowed), None)
    if unknown is not None:
        raise IngestionRejected(f"{label}.{unknown} is not upload-allowlisted")
    missing = next((key for key in (required or allowed) if key not in value), None)
    if missing is not None:
        raise IngestionRejected(f"{label}.{missing} is required")


#: ``atomicValidateModel``: the model document.
MODEL_KEYS = ("schema_version", "identity_strength", "pipeline_kind", "components")
#: …its single primary component.
COMPONENT_KEYS = ("component_id", "role", "source", "artifact", "quantization")
#: …the component's Hugging Face source. **No subfolder, no revision.**
SOURCE_KEYS = ("kind", "repo_id")
ARTIFACT_KEYS = ("format",)
QUANTIZATION_KEYS = ("kind", "base_dtype")

#: ``atomicValidateExecution``. Unlike the model block these two lists differ:
#: the optional package versions may be absent, but nothing outside `allowed`
#: may appear.
RUNTIME_ALLOWED = (
    "distribution",
    "rapid_mlx",
    "mlx",
    "mlx_lm",
    "mlx_vlm",
    "mflux",
    "python",
    "rapid_mlx_revision",
)
RUNTIME_REQUIRED = ("distribution", "rapid_mlx", "mlx", "python")
EXECUTION_KEYS = (
    "schema_version",
    "config_digest",
    "runtime",
    "task_type",
    "resources",
    "task",
)
RESOURCES_KEYS = ("max_concurrency", "compute_dtype")

DISTRIBUTIONS = ("release", "source")

#: ``atomicVersion`` — every runtime value except the distribution and the
#: revision must look like a version string.
_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")

_REPO_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

ATOMIC_BENCH_TASKS = frozenset(
    {"text_generation", "image_generation", "video_generation"}
)


def validate_model(model: Any, task_type: str) -> None:
    """Mirror of ``atomicValidateModel``. Raises :class:`IngestionRejected`."""

    _exact_object(model, MODEL_KEYS, "model")
    if model["schema_version"] != 1 or model["identity_strength"] != "unresolved":
        raise IngestionRejected(
            "model identity must be an unresolved v1 identity in this internal beta"
        )
    if model["pipeline_kind"] != task_type or task_type not in ATOMIC_BENCH_TASKS:
        raise IngestionRejected("model.pipeline_kind does not match the workload")
    components = model["components"]
    if not isinstance(components, list) or len(components) != 1:
        raise IngestionRejected(
            "model.components must contain exactly the primary component"
        )
    component = components[0]
    _exact_object(component, COMPONENT_KEYS, "model.components[0]")
    if component["component_id"] != "primary" or component["role"] != "primary":
        raise IngestionRejected("model primary component is invalid")
    _exact_object(component["source"], SOURCE_KEYS, "model.components[0].source")
    if component["source"]["kind"] != "huggingface" or not _REPO_ID.match(
        str(component["source"]["repo_id"])
    ):
        raise IngestionRejected(
            "model source must be a public Hugging Face repository id"
        )
    _exact_object(component["artifact"], ARTIFACT_KEYS, "model.components[0].artifact")
    if component["artifact"]["format"] != "mlx-safetensors":
        raise IngestionRejected("model artifact format")
    _exact_object(
        component["quantization"],
        QUANTIZATION_KEYS,
        "model.components[0].quantization",
    )
    if (
        component["quantization"]["kind"] != "unknown"
        or component["quantization"]["base_dtype"] != "unknown"
    ):
        raise IngestionRejected("model quantization")


def validate_execution(execution: Any, task_type: str) -> None:
    """Mirror of ``atomicValidateExecution``, as far as ``runtime``.

    The conditional rule is the one that matters to this client:
    ``distribution: source`` REQUIRES a 40-character lowercase
    ``rapid_mlx_revision``; ``distribution: release`` forbids it entirely. A
    clean source build is therefore publishable today, with no website change.
    """

    _exact_object(execution, EXECUTION_KEYS, "execution")
    if (
        execution["schema_version"] != 1
        or execution["task_type"] != task_type
        or not _SHA256.match(str(execution["config_digest"]))
    ):
        raise IngestionRejected("execution identity")

    runtime = execution["runtime"]
    _exact_object(runtime, RUNTIME_ALLOWED, "execution.runtime", RUNTIME_REQUIRED)
    if runtime["distribution"] not in DISTRIBUTIONS:
        raise IngestionRejected("execution.runtime.distribution")
    for key, value in runtime.items():
        if key in ("distribution", "rapid_mlx_revision"):
            continue
        if not isinstance(value, str) or not _VERSION.match(value):
            raise IngestionRejected(f"execution.runtime.{key}")
    if runtime["distribution"] == "source":
        if not _SHA1.match(str(runtime.get("rapid_mlx_revision", ""))):
            raise IngestionRejected("execution.runtime.rapid_mlx_revision")
    elif "rapid_mlx_revision" in runtime:
        raise IngestionRejected("execution.runtime.rapid_mlx_revision")

    _exact_object(execution["resources"], RESOURCES_KEYS, "execution.resources")


def validate_submission(payload: Any) -> None:
    """Validate the parts of a submission this mirror covers.

    Covers the model identity and ``execution`` (through ``runtime`` and
    ``resources``) — the two blocks this client decides the contents of. The
    machine and workload blocks have their own allowlists in the worker and are
    not mirrored here; a mirror that claimed more than it checks would be worse
    than one that states its scope.
    """

    if not isinstance(payload, dict):
        raise IngestionRejected("submission must be an object")
    workload = payload.get("workload")
    if not isinstance(workload, dict):
        raise IngestionRejected("workload must be an object")
    task_type = str(workload.get("task_type"))
    validate_model(payload.get("model"), task_type)
    validate_execution(payload.get("execution"), task_type)


def worker_source() -> Path | None:
    """A local worker checkout, when one exists."""

    return next((path for path in _WORKER_CANDIDATES if path.is_file()), None)


def _atomic_exact_object_calls(text: str) -> dict[str, tuple[str, ...]]:
    """Every ``atomicExactObject(value, allowed, required, "label")`` call.

    Parsed by balancing parentheses rather than by a regex over the whole file:
    the label is the call's last argument, and a pattern anchored only on the
    label happily swallows a thousand lines of unrelated source on its way
    there.
    """

    found: dict[str, tuple[str, ...]] = {}
    needle = "atomicExactObject("
    index = text.find(needle)
    while index != -1:
        cursor = index + len(needle)
        depth = 1
        while cursor < len(text) and depth:
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
            cursor += 1
        arguments = text[index + len(needle) : cursor - 1]
        # The label is the final string literal in the argument list.
        labels = re.findall(r'"([^"]*)"\s*,?\s*$', arguments.strip())
        if labels:
            arrays = re.findall(r"\[([^\]]*)\]", arguments)
            if arrays:
                allowed = tuple(re.findall(r'"([^"]+)"', arrays[0]))
                found[labels[0]] = allowed
        index = text.find(needle, cursor)
    return found


def cross_check_against_worker() -> list[str]:
    """Re-derive the allowlists from the worker and report any disagreement.

    Returns an empty list when the mirror matches, or when no checkout is
    present to compare against.
    """

    source = worker_source()
    if source is None:
        return []
    calls = _atomic_exact_object_calls(
        source.read_text(encoding="utf-8", errors="replace")
    )
    expectations = {
        "model": MODEL_KEYS,
        "model.components[0]": COMPONENT_KEYS,
        "model.components[0].source": SOURCE_KEYS,
        "model.components[0].artifact": ARTIFACT_KEYS,
        "model.components[0].quantization": QUANTIZATION_KEYS,
        "execution": EXECUTION_KEYS,
        "execution.runtime": RUNTIME_ALLOWED,
        "execution.resources": RESOURCES_KEYS,
    }
    problems: list[str] = []
    for label, expected in expectations.items():
        allowed = calls.get(label)
        if allowed is None:
            problems.append(f"{label}: no atomicExactObject call found in the worker")
        elif tuple(sorted(allowed)) != tuple(sorted(expected)):
            problems.append(
                f"{label}: worker allows {sorted(allowed)}, mirror expects "
                f"{sorted(expected)}"
            )
    return problems


__all__ = [
    "IngestionRejected",
    "COMPONENT_KEYS",
    "EXECUTION_KEYS",
    "RESOURCES_KEYS",
    "RUNTIME_ALLOWED",
    "RUNTIME_REQUIRED",
    "validate_execution",
    "MODEL_KEYS",
    "QUANTIZATION_KEYS",
    "SOURCE_KEYS",
    "cross_check_against_worker",
    "validate_model",
    "validate_submission",
    "worker_source",
]
