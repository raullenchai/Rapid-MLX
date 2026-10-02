#!/usr/bin/env python3
"""Capture sanitized GLM-5.3 TensorFold product-path qualification evidence.

Run this whole program through ``scripts/large-model-run.py``.  The harness
starts the shipped Rapid alias, records one request derived from the tracked
fixture, stops it, then starts the pinned direct runtime for a drafted/serial
comparison while the same command-lifetime host lock remains held.

Model resolution is local-only.  This program never opts into a Hub download.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs/engineering/performance/fixtures/glm53-tensorfold-ttlcache.json"
ALIASES = ROOT / "rapid_mlx/aliases.json"
ALIAS = "glm5.3-flash-tensorfold"
DIRECT_SERVED_NAME = "glm53-tf-v06"
HASH_MAP_NAME = "artifact-hashes.json"
SCHEMA = "rapid-mlx/glm53-tensorfold-product-capture/v1"
HASH_MAP_SCHEMA = "rapid-mlx/artifact-sha256-map/v1"
REQUIRED_FIXTURE_SHA256 = (
    "1e688842b5ff93702f83f0a7b5dac4eeca9c6cdc1734371ed74c19f6602f1eca"
)
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
CHILD_ENV_OVERRIDES = {
    **OFFLINE_ENV,
    "RAPID_MLX_TELEMETRY": "0",
    "RAPID_MLX_DISABLE_VERSION_CHECK": "1",
    "DO_NOT_TRACK": "1",
}
TENSORFOLD_CHILD_ENV_ALLOWLIST: frozenset[str] = frozenset()
TOKEN_FIELDS = ("token_ids", "output_token_ids", "generated_token_ids")
SENSITIVE_ENV_NAME = re.compile(
    r"(?:^|_)(?:TOKEN|PASSWORD|SECRET|API_KEY)(?:$|_)", re.IGNORECASE
)


class CaptureError(RuntimeError):
    """The capture contract could not be satisfied."""


class TerminationSignalError(CaptureError):
    """A terminating signal received while owned processes need cleanup."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        self.exit_code = 128 + signum
        super().__init__(f"interrupted by signal {signal.Signals(signum).name}")


class ShutdownError(CaptureError):
    """An owned server did not complete a clean, expected shutdown."""

    def __init__(self, message: str, facts: dict[str, Any]) -> None:
        self.facts = facts
        super().__init__(message)


@contextlib.contextmanager
def termination_signal_handlers():
    """Turn terminating signals into cleanup-aware exceptions.

    The first signal unwinds through ``run_capture`` so its ``finally`` block
    can terminate the owned server and verify the listener is gone. Further
    terminating signals are ignored during that bounded cleanup. The caller
    then exits with the conventional ``128 + signal`` status.
    """

    watched = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous = {item: signal.getsignal(item) for item in watched}
    handling = False

    def terminate(signum: int, _frame: object) -> None:
        nonlocal handling
        if handling:
            return
        handling = True
        for item in watched:
            signal.signal(item, signal.SIG_IGN)
        raise TerminationSignalError(signum)

    for item in watched:
        signal.signal(item, terminate)
    try:
        yield
    finally:
        if not handling:
            for item, handler in previous.items():
                signal.signal(item, handler)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_json(value))


class Sanitizer:
    """Scrub machine/user/process data before bytes reach retained artifacts."""

    def __init__(self, *private_paths: Path) -> None:
        replacements: list[tuple[str, str]] = []
        candidates = [str(path.resolve()) for path in private_paths]
        candidates.extend(
            value
            for value in (
                str(Path.home()),
                os.environ.get("HOME"),
                os.environ.get("TMPDIR"),
            )
            if value
        )
        for value in sorted(set(candidates), key=len, reverse=True):
            replacements.append((value.rstrip("/"), "<private-path>"))
        users = {getpass.getuser(), os.environ.get("USER", "")}
        self._users = tuple(sorted((x for x in users if x), key=len, reverse=True))
        self._replacements = tuple(replacements)

    def text(self, value: str) -> str:
        for source, replacement in self._replacements:
            value = value.replace(source, replacement)
        for username in self._users:
            value = re.sub(
                rf"(?<=/Users/){re.escape(username)}(?=/|\b)",
                "<user>",
                value,
            )
            value = re.sub(
                rf"(?i)(\buser(?:name)?\s*[=:]\s*){re.escape(username)}\b",
                r"\1<user>",
                value,
            )
            value = re.sub(rf"\b{re.escape(username)}\b", "<user>", value)
        value = re.sub(r"/Users/[^/\s\"']+(?:/[^\s\"']+)*", "<private-path>", value)
        value = re.sub(r"/private/(?:tmp|var)/[^\s\"']+", "<private-path>", value)
        value = re.sub(
            r"(?i)\b(authorization\b[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?|"
            r"(?:api[_-]?key|token|password|secret)\b[\"']?\s*[:=]\s*[\"']?)"
            r"[^\\\s,;\"']+",
            r"\1<redacted>",
            value,
        )
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", value)
        value = re.sub(
            r"https?://(?!127\.0\.0\.1(?::\d+)?(?:/|\b)|localhost(?::\d+)?(?:/|\b))[^\s\"']+",
            "<url>",
            value,
        )
        value = re.sub(
            r"(?i)(\bpid\b[\"']?\s*[=:]?\s*|server process \[|process \[)"
            r"\d+(\]?)",
            r"\1<pid>\2",
            value,
        )
        return value

    def bytes(self, value: bytes) -> bytes:
        return self.text(value.decode("utf-8", errors="replace")).encode()

    def argv(self, argv: Sequence[str], target: Path | None = None) -> list[str]:
        target_text = str(target.resolve()) if target is not None else None
        normalized = []
        for item in argv:
            if target_text and item == target_text:
                normalized.append("<local-qualified-snapshot>")
            else:
                normalized.append(self.text(item))
        if normalized:
            normalized[0] = Path(normalized[0]).name
        return normalized


def run_text(command: Sequence[str], *, timeout: float = 15) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "exit_code": None, "stdout": ""}
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True,
            "exit_code": None,
            "stdout": "",
            "error": type(exc).__name__,
        }
    return {
        "available": True,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def parse_swap_used_bytes(text: str) -> int | None:
    match = re.search(r"\bused\s*=\s*([0-9.]+)([KMGTP])", text, re.IGNORECASE)
    if not match:
        return None
    powers = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}
    return round(float(match.group(1)) * 1024 ** powers[match.group(2).upper()])


def listener_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def allocate_loopback_port(exclude: set[int] | None = None) -> int:
    excluded = exclude or set()
    for _attempt in range(32):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if (
            port not in excluded
            and port not in {8080, 8891}
            and not listener_open(port)
        ):
            return port
    raise CaptureError("could not allocate a collision-free loopback port")


def host_snapshot(ports: Sequence[int], sanitizer: Sanitizer) -> dict[str, Any]:
    memory_pressure = run_text(["memory_pressure"])
    vm_stat = run_text(["vm_stat"])
    swap = run_text(["sysctl", "-n", "vm.swapusage"])
    for record in (memory_pressure, vm_stat, swap):
        for field in ("stdout", "stderr"):
            record[field] = sanitizer.text(str(record.get(field, "")))
    loads = os.getloadavg()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "memory_pressure": memory_pressure,
        "vm_stat": vm_stat,
        "swap": {
            **swap,
            "used_bytes": parse_swap_used_bytes(str(swap.get("stdout", ""))),
        },
        "load_average": {"one": loads[0], "five": loads[1], "fifteen": loads[2]},
        "listeners": {str(port): listener_open(port) for port in ports},
    }


def package_provenance(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}
    record: dict[str, Any] = {"installed": True, "version": distribution.version}
    try:
        raw = distribution.read_text("direct_url.json")
        direct = json.loads(raw) if raw else {}
    except (OSError, ValueError):
        direct = {}
    if isinstance(direct, dict):
        vcs = direct.get("vcs_info")
        if isinstance(vcs, dict):
            record["vcs"] = {
                "type": vcs.get("vcs"),
                "commit_id": vcs.get("commit_id"),
                "requested_revision": vcs.get("requested_revision"),
            }
        record["editable"] = bool((direct.get("dir_info") or {}).get("editable"))
    return record


@dataclass(frozen=True)
class LaunchBinding:
    command: str
    distribution: str
    executable: Path
    version: str
    entry_point: str
    executable_sha256: str

    def provenance(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "distribution": self.distribution,
            "version": self.version,
            "entry_point": self.entry_point,
            "executable_sha256": self.executable_sha256,
            "interpreter": Path(sys.executable).name,
            "interpreter_sha256": sha256_bytes(Path(sys.executable).read_bytes()),
        }


def resolve_console_binding(command: str, distribution_name: str) -> LaunchBinding:
    """Bind a console script to the distribution in this Python interpreter."""

    executable_text = shutil.which(command)
    if executable_text is None:
        raise CaptureError(f"required console command is unavailable: {command}")
    executable = Path(executable_text).resolve()
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise CaptureError(
            f"required distribution is unavailable: {distribution_name}"
        ) from exc
    entry_points = [
        item
        for item in distribution.entry_points
        if item.group == "console_scripts" and item.name == command
    ]
    if len(entry_points) != 1:
        raise CaptureError(
            f"{distribution_name} does not expose exactly one {command} console script"
        )
    try:
        source = executable.read_text()
    except (OSError, UnicodeError) as exc:
        raise CaptureError(f"cannot inspect console command: {command}") from exc
    first_line = source.splitlines()[0] if source else ""
    if not first_line.startswith("#!"):
        raise CaptureError(f"{command} console script has no interpreter shebang")
    try:
        shebang = shlex.split(first_line[2:])
    except ValueError as exc:
        raise CaptureError(f"{command} console script has an invalid shebang") from exc
    if (
        len(shebang) != 1
        or Path(shebang[0]).resolve() != Path(sys.executable).resolve()
    ):
        raise CaptureError(
            f"{command} is not bound to the capture interpreter {Path(sys.executable).name}"
        )
    entry_point = entry_points[0]
    expected_import = f"from {entry_point.module} import {entry_point.attr}"
    if entry_point.attr is None or expected_import not in source:
        raise CaptureError(
            f"{command} wrapper does not match {distribution_name} entry-point metadata"
        )
    return LaunchBinding(
        command=command,
        distribution=distribution_name,
        executable=executable,
        version=distribution.version,
        entry_point=entry_point.value,
        executable_sha256=sha256_bytes(executable.read_bytes()),
    )


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git("status", "--short")),
    }


def require_clean_source() -> dict[str, Any]:
    provenance = git_provenance()
    if provenance["dirty"]:
        raise CaptureError("capture source tree must be clean")
    return provenance


def model_contract() -> dict[str, Any]:
    aliases = json.loads(ALIASES.read_text())
    profile = aliases.get(ALIAS)
    if not isinstance(profile, dict):
        raise CaptureError(f"missing built-in alias {ALIAS}")
    result = {
        "alias": ALIAS,
        "repository": profile.get("hf_path"),
        "target_revision": profile.get("tensorfold_target_revision"),
        "runtime_revision": profile.get("tensorfold_runtime_revision"),
    }
    for field in ("repository", "target_revision", "runtime_revision"):
        if not isinstance(result[field], str) or not result[field]:
            raise CaptureError(f"alias has no {field}")
    return result


def resolve_local_target(model: dict[str, Any]) -> Path:
    from huggingface_hub import snapshot_download

    try:
        resolved = snapshot_download(
            repo_id=model["repository"],
            revision=model["target_revision"],
            local_files_only=True,
        )
    except Exception as exc:
        raise CaptureError(
            "qualified target is not complete in the default local Hub cache"
        ) from exc
    path = Path(resolved).resolve()
    if model["target_revision"] not in path.parts:
        raise CaptureError("local target did not resolve to the qualified revision")
    return path


def product_payload(fixture: dict[str, Any]) -> dict[str, Any]:
    """Translate the direct runtime's budget name onto Rapid's public API."""

    payload = dict(fixture)
    budget = payload.pop("thinking_budget", None)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise CaptureError("fixture must carry a positive integer thinking_budget")
    if "reasoning_max_tokens" in payload:
        raise CaptureError("fixture unexpectedly carries both reasoning budget fields")
    payload["reasoning_max_tokens"] = budget
    payload["model"] = ALIAS
    return payload


def direct_payload(fixture: dict[str, Any], *, drafted: bool) -> dict[str, Any]:
    payload = dict(fixture)
    if not drafted:
        payload["draft"] = False
    return payload


@dataclass(frozen=True)
class HTTPResult:
    status: int
    headers: dict[str, str]
    raw: bytes
    started_offset_ns: int
    first_byte_offset_ns: int | None
    ended_offset_ns: int
    ttft_ns: int | None
    ttft_basis: str | None


def _visible_sse_line(line: bytes) -> bool:
    if not line.startswith(b"data:"):
        return False
    value = line[5:].strip()
    if not value or value == b"[DONE]":
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    for choice in parsed.get("choices") or []:
        delta = choice.get("delta") or {}
        if any(
            delta.get(key) not in (None, "", [], {})
            for key in ("content", "reasoning_content", "reasoning", "tool_calls")
        ):
            return True
    return False


def http_request(
    url: str,
    *,
    origin_ns: int,
    payload: bytes | None = None,
    timeout: float = 1800,
) -> HTTPResult:
    request = urllib.request.Request(url, data=payload)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    started = time.monotonic_ns()
    streamed = False
    if payload is not None:
        try:
            body = json.loads(payload)
            streamed = body.get("stream") is True
        except (json.JSONDecodeError, AttributeError):
            pass
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            chunks: list[bytes] = []
            first_byte: int | None = None
            ttft: int | None = None
            if streamed:
                while line := response.readline():
                    observed = time.monotonic_ns()
                    if first_byte is None:
                        first_byte = observed
                    chunks.append(line)
                    if ttft is None and _visible_sse_line(line):
                        ttft = observed - started
            else:
                first = response.read(1)
                if first:
                    first_byte = time.monotonic_ns()
                    chunks.append(first)
                chunks.append(response.read())
            ended = time.monotonic_ns()
            return HTTPResult(
                status=response.status,
                headers=headers,
                raw=b"".join(chunks),
                started_offset_ns=started - origin_ns,
                first_byte_offset_ns=(
                    first_byte - origin_ns if first_byte is not None else None
                ),
                ended_offset_ns=ended - origin_ns,
                ttft_ns=ttft,
                ttft_basis=("first_visible_sse_delta" if streamed else None),
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise CaptureError(
            f"HTTP {exc.code} from {urllib.parse.urlsplit(url).path}: "
            f"{raw[:300].decode(errors='replace')}"
        ) from exc


def wait_models(
    server: ManagedServer, port: int, *, origin_ns: int, timeout: float
) -> HTTPResult:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    last_error = "not ready"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise CaptureError(
                f"server exited before readiness (status {server.process.returncode})"
            )
        try:
            result = http_request(url, origin_ns=origin_ns, timeout=2)
            if result.status == 200:
                return result
        except (OSError, CaptureError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
        time.sleep(0.25)
    raise CaptureError(f"server readiness timed out ({last_error})")


def parsed_json(result: HTTPResult, label: str) -> dict[str, Any]:
    try:
        value = json.loads(result.raw)
    except json.JSONDecodeError as exc:
        raise CaptureError(f"{label} did not return JSON") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"{label} returned a non-object")
    return value


def require_models_identity(
    value: dict[str, Any], expected: str, *, require_tensorfold_mtp: bool = False
) -> dict[str, Any]:
    models = value.get("data")
    if not isinstance(models, list) or len(models) != 1:
        raise CaptureError("/v1/models did not expose exactly one model")
    model = models[0]
    if not isinstance(model, dict) or model.get("id") != expected:
        raise CaptureError(f"/v1/models did not expose {expected!r}")
    projection: dict[str, Any] = {"id": expected}
    if require_tensorfold_mtp:
        speculative = model.get("speculative_decoding")
        require_speculative_identity(speculative, label="/v1/models")
        projection["speculative_decoding"] = expected_speculative_identity()
    return projection


def expected_speculative_identity() -> dict[str, Any]:
    return {
        "configured": True,
        "method": "mtp",
        "runtime_state": "active",
        "backend": "tensorfold",
    }


def require_speculative_identity(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise CaptureError(f"{label} omitted speculative-decoding identity")
    expected = expected_speculative_identity()
    if any(value.get(key) != item for key, item in expected.items()):
        raise CaptureError(f"{label} did not expose active TensorFold MTP")


def require_product_identity(
    health: dict[str, Any], status: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    if (
        health.get("status") != "ok"
        or health.get("algorithm") != "mtp"
        or health.get("engine") != "tensorfold-glm-5.3-flash"
    ):
        raise CaptureError("product /healthz did not expose ready TensorFold MTP")
    if status.get("status") != "ready" or status.get("model") != ALIAS:
        raise CaptureError("product /v1/status readiness or model identity mismatch")
    require_speculative_identity(
        status.get("speculative_decoding"), label="product /v1/status"
    )
    profile = status.get("profile")
    if not isinstance(profile, dict) or profile.get("id") != ALIAS:
        raise CaptureError("product profile identity mismatch")
    compatibility = profile.get("compatibility")
    runtime = profile.get("runtime")
    profile_models = profile.get("models")
    target = profile_models.get("target") if isinstance(profile_models, dict) else None
    if profile.get("mode") != "accelerated":
        raise CaptureError("product profile mode is not accelerated")
    if not isinstance(compatibility, dict) or (
        compatibility.get("state"),
        compatibility.get("reason"),
        compatibility.get("action"),
    ) != ("ready", None, None):
        raise CaptureError("product profile compatibility is not ready")
    if not isinstance(runtime, dict) or (
        runtime.get("extra") != "manual-source-install"
        or runtime.get("installed") is not True
    ):
        raise CaptureError("product TensorFold runtime is not installed and qualified")
    if not isinstance(target, dict) or (
        target.get("repository") != model["repository"]
        or target.get("revision") != model["target_revision"]
        or target.get("ready") is not True
    ):
        raise CaptureError("product target provenance or readiness mismatch")
    return {
        "health": {
            "status": "ok",
            "engine": "tensorfold-glm-5.3-flash",
            "algorithm": "mtp",
        },
        "status": "ready",
        "model": ALIAS,
        "speculative_decoding": expected_speculative_identity(),
        "profile": {
            "id": ALIAS,
            "mode": "accelerated",
            "compatibility": {"state": "ready", "reason": None, "action": None},
            "runtime": {"extra": "manual-source-install", "installed": True},
            "target": {
                "repository": model["repository"],
                "revision": model["target_revision"],
                "ready": True,
            },
        },
    }


def token_sha256(token_ids: Sequence[int]) -> str:
    encoded = ",".join(str(token) for token in token_ids).encode()
    return sha256_bytes(encoded)


def _find_token_ids(value: dict[str, Any]) -> tuple[list[int], str] | None:
    candidates: list[tuple[Any, str]] = []
    for field in TOKEN_FIELDS:
        candidates.append((value.get(field), field))
    tensorfold = value.get("tensorfold")
    if isinstance(tensorfold, dict):
        for field in TOKEN_FIELDS:
            candidates.append((tensorfold.get(field), f"tensorfold.{field}"))
    choices = value.get("choices") or []
    if choices and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get("message") or {}
        for field in TOKEN_FIELDS:
            candidates.append((choice.get(field), f"choices[0].{field}"))
            if isinstance(message, dict):
                candidates.append((message.get(field), f"choices[0].message.{field}"))
    for candidate, source in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, list) or not all(
            type(item) is int and item >= 0 for item in candidate
        ):
            raise CaptureError(f"explicit token IDs at {source} were malformed")
        return candidate, source
    return None


def load_audit_token_ids(path: Path) -> tuple[list[int], str] | None:
    if not path.is_file():
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != 1:
        raise CaptureError(
            f"expected one TensorFold audit record, found {len(records)}"
        )
    record = records[0]
    tokens = record.get("token_ids")
    digest = record.get("token_sha256")
    if not isinstance(tokens, list) or not all(
        type(item) is int and item >= 0 for item in tokens
    ):
        raise CaptureError("TensorFold audit did not expose integer token IDs")
    if digest != token_sha256(tokens):
        raise CaptureError("TensorFold audit token SHA-256 did not verify")
    return tokens, "RAPID_MLX_TENSORFOLD_AUDIT_PATH"


def response_summary(
    value: dict[str, Any],
    timing: HTTPResult,
    *,
    expected_model: str,
    audit_tokens: tuple[list[int], str] | None = None,
    require_opaque_fingerprint: bool = False,
) -> dict[str, Any]:
    if value.get("object") != "chat.completion":
        raise CaptureError("completion response object was not chat.completion")
    if value.get("model") != expected_model:
        raise CaptureError("completion response model identity mismatch")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise CaptureError("completion response did not contain a response ID")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise CaptureError("completion response did not contain exactly one choice")
    if not isinstance(choices[0], dict):
        raise CaptureError("completion response choice was not an object")
    choice = choices[0]
    if choice.get("index") != 0:
        raise CaptureError("completion response choice index was not zero")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise CaptureError("completion response did not contain a message object")
    if message.get("role") != "assistant":
        raise CaptureError("completion response message role was not assistant")
    content = message.get("content")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    if content is not None and not isinstance(content, str):
        raise CaptureError("completion content was not text or null")
    if reasoning is not None and not isinstance(reasoning, str):
        raise CaptureError("completion reasoning was not text or null")
    if not isinstance(content, str) and not isinstance(reasoning, str):
        raise CaptureError("completion response exposed neither content nor reasoning")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise CaptureError("completion response exposed an unexpected finish reason")
    usage_raw = value.get("usage")
    if not isinstance(usage_raw, dict):
        raise CaptureError("completion response did not expose usage")
    usage: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = usage_raw.get(field)
        if type(item) is not int or item < 0:
            raise CaptureError(
                f"completion usage {field} was not a nonnegative integer"
            )
        usage[field] = item
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise CaptureError(
            "completion usage total did not match prompt plus completion"
        )
    response_tokens = _find_token_ids(value)
    if (
        response_tokens is not None
        and audit_tokens is not None
        and response_tokens[0] != audit_tokens[0]
    ):
        raise CaptureError("HTTP and audit token IDs did not match")
    exposed = audit_tokens or response_tokens
    if exposed is None:
        tokens: dict[str, Any] = {
            "availability": "unavailable",
            "reason": "complete token IDs were not exposed by this HTTP/runtime path",
            "ids": None,
            "sha256": None,
            "sha256_encoding": None,
        }
    else:
        ids, source = exposed
        tokens = {
            "availability": "available",
            "source": source,
            "ids": ids,
            "sha256": token_sha256(ids),
            "sha256_encoding": "ascii comma-separated decimal token IDs",
        }
        if len(ids) != usage["completion_tokens"]:
            raise CaptureError("complete token ID count did not match completion usage")
    opaque = None
    tensorfold = value.get("tensorfold")
    if tensorfold is not None and not isinstance(tensorfold, dict):
        raise CaptureError("TensorFold response metadata was not an object")
    if isinstance(tensorfold, dict) and tensorfold.get("token_sha") is not None:
        fingerprint = tensorfold.get("token_sha")
        if (
            not isinstance(fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{12}", fingerprint) is None
        ):
            raise CaptureError("TensorFold opaque token fingerprint was not 12-hex")
        opaque = {
            "value": fingerprint,
            "kind": "tensorfold_opaque_token_fingerprint",
            "is_sha256": False,
        }
    if require_opaque_fingerprint and opaque is None:
        raise CaptureError("direct TensorFold response omitted its opaque fingerprint")
    return {
        "content_sha256": (
            sha256_bytes(content.encode()) if isinstance(content, str) else None
        ),
        "reasoning_sha256": (
            sha256_bytes(reasoning.encode()) if isinstance(reasoning, str) else None
        ),
        "content_present": isinstance(content, str),
        "reasoning_present": isinstance(reasoning, str),
        "finish_reason": finish_reason,
        "usage": usage,
        "full_token_ids": tokens,
        "opaque_token_fingerprint": opaque,
        "timing": {
            "clock": "time.monotonic_ns",
            "started_offset_ns": timing.started_offset_ns,
            "first_byte_offset_ns": timing.first_byte_offset_ns,
            "ended_offset_ns": timing.ended_offset_ns,
            "duration_ns": timing.ended_offset_ns - timing.started_offset_ns,
            "ttft_ns": timing.ttft_ns,
            "ttft_basis": timing.ttft_basis,
            "ttft_unavailable_reason": (
                None if timing.ttft_ns is not None else "request_was_not_streamed"
            ),
        },
    }


def retain_http(
    directory: Path,
    name: str,
    result: HTTPResult,
    sanitizer: Sanitizer,
) -> dict[str, Any]:
    retained = sanitizer.bytes(result.raw)
    write_bytes(directory / f"{name}.sanitized.body", retained)
    headers = {
        key: sanitizer.text(value)
        for key, value in result.headers.items()
        if key not in {"set-cookie", "authorization", "proxy-authorization"}
    }
    write_json(directory / f"{name}.headers.json", headers)
    return {
        "status": result.status,
        "received_bytes": len(result.raw),
        "received_sha256": sha256_bytes(result.raw),
        "retained_bytes": len(retained),
        "retained_sha256": sha256_bytes(retained),
        "sanitization_changed_bytes": retained != result.raw,
    }


@dataclass
class ManagedServer:
    name: str
    process: subprocess.Popen[bytes]
    port: int
    command: list[str]
    raw_log: Path
    started_offset_ns: int


def start_server(
    name: str,
    command: Sequence[str],
    *,
    port: int,
    env: dict[str, str],
    raw_dir: Path,
    origin_ns: int,
) -> ManagedServer:
    if listener_open(port):
        raise CaptureError(f"refusing to start {name}: loopback port is occupied")
    raw_log = raw_dir / f"{name}.log"
    handle = raw_log.open("wb")
    try:
        process = subprocess.Popen(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except Exception:
        handle.close()
        raise
    handle.close()
    return ManagedServer(
        name=name,
        process=process,
        port=port,
        command=list(command),
        raw_log=raw_log,
        started_offset_ns=time.monotonic_ns() - origin_ns,
    )


def stop_server(server: ManagedServer, *, timeout: float = 120) -> dict[str, Any]:
    forced = False
    was_running = server.process.poll() is None
    termination_sent = False
    if was_running:
        try:
            os.killpg(server.process.pid, signal.SIGTERM)
            termination_sent = True
        except ProcessLookupError:
            pass
        try:
            server.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            forced = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.process.pid, signal.SIGKILL)
            server.process.wait(timeout=15)
    deadline = time.monotonic() + 5
    while listener_open(server.port) and time.monotonic() < deadline:
        time.sleep(0.1)
    gone = not listener_open(server.port)
    if not gone:
        # The leader can exit while a child retains the listener. The whole
        # session is ours, so force the remaining owned process group down,
        # retain that fact, and still fail the capture.
        forced = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.process.pid, signal.SIGKILL)
        deadline = time.monotonic() + 15
        while listener_open(server.port) and time.monotonic() < deadline:
            time.sleep(0.1)
        gone = not listener_open(server.port)
    facts = {
        "process_was_running_at_shutdown": was_running,
        "process_exited": True,
        "exit_code": server.process.returncode,
        "termination_signal_sent": termination_sent,
        "forced_kill_of_owned_process_group": forced,
        "listener_gone": gone,
    }
    if not was_running or not termination_sent:
        raise ShutdownError(f"{server.name} exited before requested shutdown", facts)
    if forced:
        raise ShutdownError(f"{server.name} required a forced kill", facts)
    if server.process.returncode != 0:
        raise ShutdownError(
            f"{server.name} exited with unexpected status {server.process.returncode}",
            facts,
        )
    if not gone:
        raise ShutdownError(
            f"{server.name} listener remained after process shutdown", facts
        )
    return facts


def retain_log(server: ManagedServer, destination: Path, sanitizer: Sanitizer) -> None:
    raw = server.raw_log.read_bytes() if server.raw_log.is_file() else b""
    write_bytes(destination, sanitizer.bytes(raw))


def product_command(executable: Path, port: int) -> list[str]:
    return [
        str(executable),
        "serve",
        ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def direct_command(executable: Path, port: int, target: Path) -> list[str]:
    return [
        str(executable),
        "serve",
        str(target),
        "--name",
        DIRECT_SERVED_NAME,
        "--context",
        "8192",
        "--max-tokens",
        "4096",
        "--parallel",
        "1",
        "--mtp-drafts",
        "3",
        "--prefill-pass",
        "8",
        "--pass-cache-gib",
        "16",
        "--snapshot-dir",
        "none",
        "--port",
        str(port),
        "--no-update-check",
    ]


def compare_summaries(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    result = {
        field: left.get(field) == right.get(field)
        for field in ("content_sha256", "reasoning_sha256", "finish_reason")
    }
    left_usage = left.get("usage") or {}
    right_usage = right.get("usage") or {}
    result["usage_counts"] = all(
        left_usage.get(field) == right_usage.get(field)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    left_tokens = left.get("full_token_ids") or {}
    right_tokens = right.get("full_token_ids") or {}
    if (
        left_tokens.get("availability") == "available"
        and right_tokens.get("availability") == "available"
    ):
        result["full_token_ids_sha256"] = left_tokens.get("sha256") == right_tokens.get(
            "sha256"
        )
    left_opaque = left.get("opaque_token_fingerprint") or {}
    right_opaque = right.get("opaque_token_fingerprint") or {}
    if left_opaque.get("value") is not None and right_opaque.get("value") is not None:
        result["opaque_token_fingerprint"] = left_opaque.get(
            "value"
        ) == right_opaque.get("value")
    return result


def write_hash_map(output: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != HASH_MAP_NAME
    )
    hashes = {
        path.relative_to(output).as_posix(): sha256_bytes(path.read_bytes())
        for path in files
    }
    write_json(
        output / HASH_MAP_NAME,
        {
            "schema": HASH_MAP_SCHEMA,
            "algorithm": "sha256",
            "excludes": [HASH_MAP_NAME],
            "files": hashes,
        },
    )
    return hashes


def verify_hash_map(output: Path) -> None:
    record = json.loads((output / HASH_MAP_NAME).read_text())
    expected = record.get("files")
    if not isinstance(expected, dict):
        raise CaptureError("artifact hash map has no file mapping")
    actual_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != HASH_MAP_NAME
    }
    if actual_paths != set(expected):
        raise CaptureError("artifact hash map does not cover the retained file set")
    for relative, digest in expected.items():
        if sha256_bytes((output / relative).read_bytes()) != digest:
            raise CaptureError(f"artifact hash mismatch: {relative}")


def runtime_provenance() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rapid-mlx": package_provenance("rapid-mlx"),
        "tensorfold": package_provenance("tensorfold"),
        "mlx": package_provenance("mlx"),
        "offline_environment": OFFLINE_ENV,
        "child_environment_overrides": CHILD_ENV_OVERRIDES,
        "child_environment_removed": [
            "PYTHONHOME",
            "PYTHONPATH",
            "RAPID_MLX_* inherited values",
            "TENSORFOLD_* inherited values except explicit allowlist",
            "credential-named inherited values",
        ],
    }


def offline_child_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if SENSITIVE_ENV_NAME.search(name) is None
        and name not in {"PYTHONHOME", "PYTHONPATH"}
        and not name.startswith("RAPID_MLX_")
        and (
            not name.startswith("TENSORFOLD_") or name in TENSORFOLD_CHILD_ENV_ALLOWLIST
        )
    }
    environment.update(CHILD_ENV_OVERRIDES)
    return environment


def require_qualified_runtime(model: dict[str, Any]) -> None:
    runtime = package_provenance("tensorfold")
    vcs = runtime.get("vcs") or {}
    if (
        runtime.get("version") != "0.6.0"
        or vcs.get("type") != "git"
        or vcs.get("commit_id") != model["runtime_revision"]
        or runtime.get("editable") is True
    ):
        raise CaptureError(
            "installed TensorFold runtime is not the exact qualified, non-editable "
            "0.6.0 git revision"
        )


ProductBuilder = Callable[[int], list[str]]
DirectBuilder = Callable[[int, Path], list[str]]
HostProbe = Callable[[Sequence[int], Sanitizer], dict[str, Any]]


def run_capture(
    output: Path,
    *,
    target: Path,
    product_builder: ProductBuilder,
    direct_builder: DirectBuilder,
    launch_provenance: dict[str, Any],
    source_provenance: dict[str, Any],
    load_timeout: float = 1800,
    request_timeout: float = 1800,
    probe: HostProbe = host_snapshot,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise CaptureError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    origin_ns = time.monotonic_ns()
    model = model_contract()
    sanitizer = Sanitizer(ROOT, output, target)
    fixture_bytes = FIXTURE.read_bytes()
    if sha256_bytes(fixture_bytes) != REQUIRED_FIXTURE_SHA256:
        raise CaptureError("tracked qualification fixture SHA-256 changed")
    fixture = json.loads(fixture_bytes)
    product_body = canonical_json(product_payload(fixture))
    drafted_body = canonical_json(direct_payload(fixture, drafted=True))
    serial_body = canonical_json(direct_payload(fixture, drafted=False))
    product_port = allocate_loopback_port()
    direct_port = allocate_loopback_port({product_port})
    ports = [product_port, direct_port]
    write_bytes(output / "requests/frozen-fixture.json", fixture_bytes)
    write_bytes(output / "requests/product-submitted.json", product_body)
    write_bytes(output / "requests/direct-drafted-submitted.json", drafted_body)
    write_bytes(output / "requests/direct-serial-submitted.json", serial_body)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "clock": "time.monotonic_ns",
        "main": source_provenance,
        "model": model,
        "runtime": runtime_provenance(),
        "launch_bindings": launch_provenance,
        "host_lock": {
            "wrapper": "scripts/large-model-run.py",
            "command_lifetime_marker_present": (
                os.environ.get("RAPID_LARGE_MODEL_LOCK_HELD") == "1"
            ),
        },
        "fixture": {
            "path": FIXTURE.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(fixture_bytes),
            "product_translation": {
                "thinking_budget": "reasoning_max_tokens",
                "model": ALIAS,
            },
        },
        "argv": sanitizer.argv(sys.argv),
        "ports": {
            "selection": "kernel-assigned collision-free loopback ports",
            "product": product_port,
            "direct": direct_port,
            "explicitly_excluded": [8080, 8891],
        },
        "phases": {},
    }
    product_server: ManagedServer | None = None
    direct_server: ManagedServer | None = None
    product_shutdown: dict[str, Any] | None = None
    direct_shutdown: dict[str, Any] | None = None
    capture_error: BaseException | None = None

    child_env = offline_child_environment()
    with tempfile.TemporaryDirectory(prefix="rapid-mlx-glm53-capture-raw-") as raw:
        raw_dir = Path(raw)
        audit_path = raw_dir / "product-token-audit.jsonl"
        try:
            before = probe(ports, sanitizer)
            write_json(output / "host/pre.json", before)
            swap_used = (before.get("swap") or {}).get("used_bytes")
            if swap_used is None:
                raise CaptureError("could not establish pre-capture swap usage")
            if swap_used != 0:
                raise CaptureError("discarded: pre-capture swap usage is nonzero")
            if any((before.get("listeners") or {}).values()):
                raise CaptureError(
                    "selected loopback port became occupied before launch"
                )

            product_env = dict(
                child_env,
                RAPID_MLX_TENSORFOLD_AUDIT_PATH=str(audit_path),
            )
            product_command = product_builder(product_port)
            product_server = start_server(
                "product",
                product_command,
                port=product_port,
                env=product_env,
                raw_dir=raw_dir,
                origin_ns=origin_ns,
            )
            models_result = wait_models(
                product_server,
                product_port,
                origin_ns=origin_ns,
                timeout=load_timeout,
            )
            models = parsed_json(models_result, "product /v1/models")
            models_identity = require_models_identity(
                models, ALIAS, require_tensorfold_mtp=True
            )
            health_result = http_request(
                f"http://127.0.0.1:{product_port}/healthz",
                origin_ns=origin_ns,
                timeout=10,
            )
            status_result = http_request(
                f"http://127.0.0.1:{product_port}/v1/status",
                origin_ns=origin_ns,
                timeout=10,
            )
            health = parsed_json(health_result, "product /healthz")
            status = parsed_json(status_result, "product /v1/status")
            phase_dir = output / "product"
            raw_contract = {
                "models": retain_http(phase_dir, "models", models_result, sanitizer),
                "health": retain_http(phase_dir, "health", health_result, sanitizer),
                "status": retain_http(phase_dir, "status", status_result, sanitizer),
            }
            manifest["phases"]["product"] = {
                "command": sanitizer.argv(product_command),
                "raw_responses": raw_contract,
            }
            product_identity = require_product_identity(health, status, model)
            completion_result = http_request(
                f"http://127.0.0.1:{product_port}/v1/chat/completions",
                origin_ns=origin_ns,
                payload=product_body,
                timeout=request_timeout,
            )
            completion = parsed_json(completion_result, "product completion")
            audit_tokens = load_audit_token_ids(audit_path)
            if audit_tokens is None:
                raise CaptureError(
                    "product completion omitted required full-token audit evidence"
                )
            product_summary = response_summary(
                completion,
                completion_result,
                expected_model=ALIAS,
                audit_tokens=audit_tokens,
            )
            raw_contract["completion"] = retain_http(
                phase_dir, "completion", completion_result, sanitizer
            )
            write_json(phase_dir / "completion.summary.json", product_summary)
            manifest["phases"]["product"].update(
                {
                    "identity": {"models": models_identity, **product_identity},
                    "completion": product_summary,
                }
            )
            try:
                product_shutdown = stop_server(product_server)
            except ShutdownError as exc:
                product_shutdown = exc.facts
                raise
            finally:
                manifest["phases"]["product"]["process"] = {
                    "started_offset_ns": product_server.started_offset_ns,
                    **(product_shutdown or {}),
                }
                retain_log(
                    product_server, phase_dir / "server.sanitized.log", sanitizer
                )

            direct_command = direct_builder(direct_port, target)
            direct_server = start_server(
                "direct",
                direct_command,
                port=direct_port,
                env=child_env,
                raw_dir=raw_dir,
                origin_ns=origin_ns,
            )
            direct_models_result = wait_models(
                direct_server,
                direct_port,
                origin_ns=origin_ns,
                timeout=load_timeout,
            )
            direct_models = parsed_json(direct_models_result, "direct /v1/models")
            direct_identity = require_models_identity(direct_models, DIRECT_SERVED_NAME)
            drafted_result = http_request(
                f"http://127.0.0.1:{direct_port}/v1/chat/completions",
                origin_ns=origin_ns,
                payload=drafted_body,
                timeout=request_timeout,
            )
            serial_result = http_request(
                f"http://127.0.0.1:{direct_port}/v1/chat/completions",
                origin_ns=origin_ns,
                payload=serial_body,
                timeout=request_timeout,
            )
            drafted = response_summary(
                parsed_json(drafted_result, "direct drafted completion"),
                drafted_result,
                expected_model=DIRECT_SERVED_NAME,
                require_opaque_fingerprint=True,
            )
            serial = response_summary(
                parsed_json(serial_result, "direct serial completion"),
                serial_result,
                expected_model=DIRECT_SERVED_NAME,
                require_opaque_fingerprint=True,
            )
            direct_dir = output / "direct"
            direct_raw_contract = {
                "models": retain_http(
                    direct_dir, "models", direct_models_result, sanitizer
                ),
                "drafted": retain_http(
                    direct_dir, "drafted", drafted_result, sanitizer
                ),
                "serial": retain_http(direct_dir, "serial", serial_result, sanitizer),
            }
            write_json(direct_dir / "drafted.summary.json", drafted)
            write_json(direct_dir / "serial.summary.json", serial)
            comparisons = {
                "product_vs_direct_drafted": compare_summaries(
                    product_summary, drafted
                ),
                "direct_drafted_vs_serial": compare_summaries(drafted, serial),
            }
            manifest["phases"]["direct"] = {
                "command": sanitizer.argv(direct_command, target),
                "identity": {"models": direct_identity},
                "raw_responses": direct_raw_contract,
                "drafted": drafted,
                "serial": serial,
                "comparisons": comparisons,
            }
            try:
                direct_shutdown = stop_server(direct_server)
            except ShutdownError as exc:
                direct_shutdown = exc.facts
                raise
            finally:
                manifest["phases"]["direct"]["process"] = {
                    "started_offset_ns": direct_server.started_offset_ns,
                    **(direct_shutdown or {}),
                }
                retain_log(
                    direct_server, direct_dir / "server.sanitized.log", sanitizer
                )
            if not all(comparisons["product_vs_direct_drafted"].values()):
                raise CaptureError("product and direct drafted outputs differ")
            if not all(comparisons["direct_drafted_vs_serial"].values()):
                raise CaptureError("direct drafted and serial outputs differ")
        except BaseException as exc:
            capture_error = exc
        finally:
            for server, stopped, destination in (
                (
                    product_server,
                    product_shutdown,
                    output / "product/server.sanitized.log",
                ),
                (
                    direct_server,
                    direct_shutdown,
                    output / "direct/server.sanitized.log",
                ),
            ):
                if server is not None:
                    if stopped is None:
                        try:
                            stopped = stop_server(server)
                        except ShutdownError as stop_exc:
                            stopped = stop_exc.facts
                            if capture_error is None:
                                capture_error = stop_exc
                        except BaseException as stop_exc:
                            if capture_error is None:
                                capture_error = stop_exc
                    if stopped is not None:
                        phase = manifest["phases"].setdefault(server.name, {})
                        phase["process"] = {
                            "started_offset_ns": server.started_offset_ns,
                            **stopped,
                        }
                    retain_log(server, destination, sanitizer)
            after = probe(ports, sanitizer)
            write_json(output / "host/post.json", after)
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["duration_ns"] = time.monotonic_ns() - origin_ns
            manifest["final_listeners_gone"] = not any(
                (after.get("listeners") or {}).values()
            )
            post_swap = (after.get("swap") or {}).get("used_bytes")
            manifest["swap_gate"] = {
                "rule": "discard if pre- or post-capture used swap is nonzero",
                "pre_used_bytes": (
                    (
                        json.loads((output / "host/pre.json").read_text()).get("swap")
                        or {}
                    ).get("used_bytes")
                    if (output / "host/pre.json").is_file()
                    else None
                ),
                "post_used_bytes": post_swap,
                "passed": False,
            }
            manifest["swap_gate"]["passed"] = (
                manifest["swap_gate"]["pre_used_bytes"] == 0 and post_swap == 0
            )
            if capture_error is None and post_swap != 0:
                capture_error = CaptureError(
                    "discarded: post-capture swap usage is nonzero"
                )
            if capture_error is None and not manifest["final_listeners_gone"]:
                capture_error = CaptureError("capture left a loopback listener behind")
            manifest["status"] = "complete" if capture_error is None else "invalid"
            if capture_error is not None:
                manifest["error"] = sanitizer.text(str(capture_error))
            write_json(output / "manifest.json", manifest)
            write_hash_map(output)
            verify_hash_map(output)

    if capture_error is not None:
        raise capture_error
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new/empty artifact directory under /private/tmp",
    )
    parser.add_argument("--load-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.environ.get("RAPID_LARGE_MODEL_LOCK_HELD") != "1":
        raise CaptureError(
            "run the harness through scripts/large-model-run.py so one host lock "
            "covers product and direct phases"
        )
    output = args.output.resolve()
    if Path("/private/tmp") not in (output, *output.parents):
        raise CaptureError("capture output must be under /private/tmp")
    with termination_signal_handlers():
        source_provenance = require_clean_source()
        model = model_contract()
        require_qualified_runtime(model)
        rapid_binding = resolve_console_binding("rapid-mlx", "rapid-mlx")
        tensorfold_binding = resolve_console_binding("tensorfold", "tensorfold")
        target = resolve_local_target(model)
        run_capture(
            output,
            target=target,
            product_builder=lambda port: product_command(
                rapid_binding.executable, port
            ),
            direct_builder=lambda port, local_target: direct_command(
                tensorfold_binding.executable, port, local_target
            ),
            launch_provenance={
                "rapid-mlx": rapid_binding.provenance(),
                "tensorfold": tensorfold_binding.provenance(),
            },
            source_provenance=source_provenance,
            load_timeout=args.load_timeout,
            request_timeout=args.request_timeout,
        )
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TerminationSignalError as exc:
        print(f"glm53-product-capture: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except CaptureError as exc:
        print(f"glm53-product-capture: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
