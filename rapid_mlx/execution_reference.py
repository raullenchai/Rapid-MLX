# SPDX-License-Identifier: Apache-2.0
"""Bounded terminal execution facts from one loaded handle and request."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def public_repositories() -> frozenset[str]:
    """The bundled, versioned catalog; never user-overridden alias resolution."""
    try:
        payload = json.loads(
            Path(__file__).with_name("decode_expectations.json").read_text()
        )
        return frozenset(row["repo_id"] for row in payload["models"].values())
    except (OSError, ValueError, KeyError, TypeError):
        return frozenset()


@lru_cache(maxsize=1)
def _catalog_artifacts() -> tuple[tuple[str, str], ...]:
    aliases = json.loads(Path(__file__).with_name("aliases.json").read_text())
    repo_profiles = json.loads(
        Path(__file__).with_name("repo_profiles.json").read_text()
    )
    aliases = {k: v for k, v in repo_profiles.items() if k != "_comment"} | aliases
    public = public_repositories()
    return tuple(
        sorted(
            {
                (row["hf_path"], (row.get("subfolder") or ""))
                for row in aliases.values()
                if row.get("hf_path") in public
            }
        )
    )


@dataclass(frozen=True)
class LoadReceipt:
    repo_id: str | None = None
    assistant_source: str = "unknown"
    backend: str = "ordinary"
    comparable: bool = True


def freeze_load_receipt(
    checkpoint_source: str, *, head_container: str | None = None
) -> LoadReceipt:
    """Map the selected artifact to the trusted cache, without hashing weights.

    This is provenance mapping, not cryptographic attestation. A copied local
    directory or mutable custom alias cannot supply a public repository name.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    try:
        source = Path(checkpoint_source).expanduser().resolve(strict=True)
        if not source.is_dir():
            return LoadReceipt()
        matches: set[str] = set()
        roots: set[Path] = set()
        artifacts = _catalog_artifacts()
        # Consider only the selected path's cache coordinates, then confirm
        # the canonical trusted-cache match. No cache-wide traversal or IO
        # over weight files is needed, and a lookalike directory is not proof.
        for candidate in (Path(checkpoint_source).expanduser().absolute(), source):
            parts = candidate.parts
            for index, name in enumerate(parts[:-2]):
                if parts[index + 1] != "snapshots" or not re.fullmatch(
                    r"[0-9a-f]{40}", parts[index + 2]
                ):
                    continue
                suffix = "/".join(parts[index + 3 :])
                for repo, subfolder in artifacts:
                    if (
                        name != "models--" + repo.replace("/", "--")
                        or suffix != subfolder
                    ):
                        continue
                    snapshot = (
                        Path(HF_HUB_CACHE) / name / "snapshots" / parts[index + 2]
                    )
                    if (snapshot / subfolder).resolve(strict=True) == source:
                        matches.add(repo)
                        roots.add(snapshot.resolve())
        repo_id = next(iter(matches)) if len(matches) == 1 else None
        artifact_root = next(iter(roots)) if len(roots) == 1 else source
        assistant = "unknown"
        if head_container is not None:
            head = Path(head_container).expanduser().resolve(strict=True)
            assistant = (
                "embedded_head"
                if head.is_relative_to(artifact_root)
                else "external_model"
            )
        # The cohort vocabulary cannot distinguish repository subfolder
        # quantizations. Keep the public repo fact without inventing a match.
        comparable = not any(
            repo == repo_id and subfolder for repo, subfolder in artifacts
        )
        return LoadReceipt(
            repo_id=repo_id, assistant_source=assistant, comparable=comparable
        )
    except (OSError, ValueError, KeyError, TypeError):
        return LoadReceipt()


@dataclass(frozen=True)
class _Observation:
    receipt: LoadReceipt
    method: str
    max_draft_tokens: int | None = None


@dataclass
class _Trace:
    observations: set[_Observation] = field(default_factory=set)


def _observe(
    request: Any, receipt: LoadReceipt, method: str, maximum: int | None = None
) -> None:
    trace = getattr(request, "_execution_reference_trace", None)
    if not isinstance(trace, _Trace):
        trace = _Trace()
        request._execution_reference_trace = trace
    trace.observations.add(_Observation(receipt, method, maximum))


def _requests_for_step(gb: Any, requests: dict, uid_map: dict) -> list[Any]:
    return [requests.get(uid_map.get(uid)) for uid in list(getattr(gb, "uids", ()))]


def _verifications(request: Any) -> int:
    counter = getattr(request, "_mtp_accept_counter", None)
    return counter.snapshot().verify_calls if counter is not None else 0


def observe_plain_steps(
    bg: Any, requests: dict, uid_map: dict, receipt: LoadReceipt | None
) -> dict | None:
    """Mark only calls that actually execute the ordinary generation step."""
    if not isinstance(receipt, LoadReceipt):
        return None
    gb = bg._generation_batch
    original = gb._step
    state = {"method": "unknown"}

    def observed_step():
        owners = _requests_for_step(gb, requests, uid_map)
        result = original()
        for request in owners[: len(result[0])]:
            if request is not None:
                method = "none" if state["method"] in ("none", "mtp") else "unknown"
                _observe(request, receipt, method)
        return result

    gb._step = observed_step
    return state


def observe_mtp_steps(
    step: Any,
    gb: Any,
    requests: dict,
    uid_map: dict,
    receipt: LoadReceipt | None,
    maximum: int,
):
    """Positive verifier work binds the executing strategy, not process flags."""
    if not isinstance(receipt, LoadReceipt):
        return step

    def observed_step():
        owners = [
            (request, _verifications(request))
            for request in _requests_for_step(gb, requests, uid_map)
            if request is not None
        ]
        result = step()
        for request, before in owners:
            if _verifications(request) > before:
                _observe(request, receipt, "mtp", maximum)
        return result

    return observed_step


def terminal_reference(
    request: Any, completion_tokens: int, *, runtime_method: str | None
) -> dict | None:
    trace = getattr(request, "_execution_reference_trace", None)
    if not isinstance(trace, _Trace) or not trace.observations:
        return None
    records = trace.observations
    repos = {row.receipt.repo_id for row in records}
    repo = next(iter(repos)) if len(repos) == 1 else None
    mixed = len(records) > 1
    row = next(iter(records))
    method = "mixed" if mixed else row.method
    backends = {entry.receipt.backend for entry in records}
    backend = next(iter(backends)) if len(backends) == 1 else "mixed"
    if runtime_method not in (None, "mtp"):
        method = "mixed" if mixed else "unknown"
        backend = "unknown"
    if method == "none" and _verifications(request) > 0:
        method = "unknown"  # Unattributed verification contradicts plain-only proof.
    source = row.receipt.assistant_source if method == "mtp" else None
    maximum = row.max_draft_tokens if method == "mtp" else None
    observed = None
    counter = getattr(request, "_mtp_accept_counter", None)
    if method == "mtp" and counter is not None:
        histogram = counter.snapshot().response_metrics()["drafted_by_depth"]
        observed = max(
            (i + 1 for i, count in enumerate(histogram) if count > 0), default=None
        )
    key = None
    if backend == "ordinary" and row.receipt.comparable:
        if method == "none":
            key = "none"
        elif (
            method == "mtp"
            and source == "embedded_head"
            and type(maximum) is int
            and maximum > 0
        ):
            key = f"mtp:embedded_head:k={maximum}:ordinary"
    return {
        "version": 1,
        "status": "mixed"
        if mixed
        else "known"
        if repo is not None and key is not None
        else "unknown",
        "repo_id": repo,
        "method": method,
        "assistant_source": source,
        "backend": backend,
        "max_draft_tokens": maximum,
        "observed_max_draft_tokens": observed,
        "config_key": key,
        "completion_tokens": completion_tokens,
    }
