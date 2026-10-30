# SPDX-License-Identifier: Apache-2.0
"""Cache successful GPU matrix results for an exact validation configuration."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .base import StepResult
from .context import env_truthy


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


def _repo_state(repo_root: Path) -> tuple[str, bool]:
    try:
        return _git(repo_root, "rev-parse", "HEAD"), not _git(
            repo_root, "status", "--porcelain", "--untracked-files=no"
        )
    except (OSError, subprocess.SubprocessError):
        return "", False


def _cache_dir(ctx) -> Path | None:
    override = os.environ.get("PR_VALIDATE_GPU_CACHE_DIR")
    if override:
        return Path(override)
    try:
        common = _git(ctx.repo_root, "rev-parse", "--git-common-dir")
        if not common:
            return None
        return (ctx.repo_root / common).resolve() / "pr-validate-gpu-cache"
    except (OSError, subprocess.SubprocessError):
        return None


def cache_key(ctx) -> str | None:
    if env_truthy("PR_VALIDATE_NO_GPU_CACHE"):
        ctx.run_log("GPU cache disabled by PR_VALIDATE_NO_GPU_CACHE")
        return None
    if not re.fullmatch(r"[0-9a-f]{40}", ctx.head_sha):
        ctx.run_log("GPU cache disabled: invalid head SHA")
        return None
    head, clean = _repo_state(ctx.repo_root)
    if not head or not clean:
        ctx.run_log("GPU cache disabled: repository state is dirty or unavailable")
        return None
    if _cache_dir(ctx) is None:
        ctx.run_log("GPU cache disabled: git common directory unavailable")
        return None
    package = Path(__file__).parent
    try:
        registry = hashlib.sha256(
            (package / "golden_models.yaml").read_bytes()
        ).hexdigest()
        step = hashlib.sha256(
            (package / "steps/stress_e2e_bench.py").read_bytes()
        ).hexdigest()
    except OSError as exc:
        ctx.run_log(f"GPU cache disabled: {exc}")
        return None
    data = {
        "schema": 1,
        "head_sha": ctx.head_sha,
        "base_sha": ctx.base_sha,
        "repo_head": head,
        "registry": registry,
        "step": step,
        "ram_gb": os.environ.get("PR_VALIDATE_RAM_GB", ""),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load(key, ctx) -> StepResult | None:
    directory = _cache_dir(ctx)
    if directory is None:
        return None
    try:
        data = json.loads((directory / f"{key}.json").read_text())
        if (
            data["schema"] != 1
            or data["key"] != key
            or data["head_sha"] != ctx.head_sha
            or data["status"] != "pass"
        ):
            return None
        if not all(
            isinstance(data[field], str)
            for field in ("summary", "details", "recorded_at")
        ):
            return None
        if not all(
            isinstance(data[field], list)
            and all(isinstance(x, str) for x in data[field])
            for field in ("findings", "artifacts")
        ):
            return None
        return StepResult(
            name="stress_e2e_bench",
            status="pass",
            summary=f"cached pass for head {ctx.head_sha[:12]}: {data['summary']}",
            details=f"Reused the passing gpu result recorded at {data['recorded_at']} for this exact head and config (set PR_VALIDATE_NO_GPU_CACHE=1 to force a fresh run).\n\n{data['details']}",
            findings=data["findings"],
            artifacts=data["artifacts"],
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None


def store(key, ctx, result: StepResult) -> None:
    if result.status != "pass":
        return
    directory = _cache_dir(ctx)
    if directory is None:
        return
    data = {
        "schema": 1,
        "key": key,
        "head_sha": ctx.head_sha,
        "status": "pass",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "summary": result.summary,
        "details": result.details,
        "findings": result.findings,
        "artifacts": result.artifacts,
        "duration_seconds": result.duration_seconds,
    }
    tmp = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as f:
            tmp = f.name
            json.dump(data, f)
        os.replace(tmp, directory / f"{key}.json")
    except OSError as exc:
        ctx.run_log(f"warning: could not store GPU cache: {exc}")
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            except OSError:
                pass
