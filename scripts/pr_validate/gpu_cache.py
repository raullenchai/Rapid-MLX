# SPDX-License-Identifier: Apache-2.0
"""Cache successful GPU matrix results for an exact validation configuration."""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .base import StepResult
from .context import env_truthy


def _log(ctx, msg: str) -> None:
    log = getattr(ctx, "run_log", None)
    if callable(log):
        log(msg)


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
            repo_root, "status", "--porcelain"
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


def _installed_fingerprint() -> str:
    packages = sorted(
        f"{name.lower()}=={dist.version}"
        for dist in metadata.distributions()
        if (name := dist.metadata.get("Name"))
    )
    return hashlib.sha256("\n".join(packages).encode()).hexdigest()


def model_fingerprint(model_id: str, cache_dir: Path | None = None) -> dict:
    """Identify the loaded files without modifying the Hub cache."""
    from rapid_mlx.model_aliases import (
        resolve_hub_model,
        resolve_model,
        resolve_revision,
        resolve_subfolder,
    )

    resolved = resolve_model(model_id)
    repo_id = resolve_hub_model(model_id)
    revision = None
    repo_cache = None
    if Path(resolved).is_dir():
        root = Path(resolved)
    else:
        if cache_dir is None:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_dir = Path(HF_HUB_CACHE)
        repo_cache = cache_dir / f"models--{repo_id.replace('/', '--')}"
        revision = resolve_revision(model_id)
        if revision is None:
            revision = (repo_cache / "refs" / "main").read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("invalid snapshot revision")
        root = repo_cache / "snapshots" / revision
        subfolder = resolve_subfolder(model_id)
        if subfolder:
            root = root / subfolder
    if not root.is_dir():
        raise ValueError("model directory unavailable")

    entries = []

    def onerror(exc):
        raise exc

    for directory, dirs, files in os.walk(root, onerror=onerror, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in dirs):
            raise ValueError("linked model directory")
        for name in files:
            path = base / name
            if path.is_symlink():
                if not path.resolve(strict=True).is_file():
                    raise ValueError("invalid model link")
                link = Path(os.path.abspath(path.parent / os.readlink(path)))
                if repo_cache is not None and link.parent == repo_cache / "blobs":
                    stat = path.stat()
                    identity = [
                        "blob",
                        link.name,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ino,
                        stat.st_ctime_ns,
                    ]
                else:
                    stat = path.stat()
                    identity = [
                        "file",
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ino,
                        stat.st_ctime_ns,
                    ]
            else:
                stat = path.stat()
                if not path.is_file():
                    raise ValueError("invalid model file")
                identity = [
                    "file",
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ino,
                    stat.st_ctime_ns,
                ]
            entries.append((path.relative_to(root).as_posix(), identity))
    if not entries:
        raise ValueError("empty model directory")
    digest = hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()
    return {"repo_id": repo_id, "revision": revision, "sha256": digest}


def cache_key(ctx) -> str | None:
    head_sha = getattr(ctx, "head_sha", "") or ""
    base_sha = getattr(ctx, "base_sha", "") or ""
    if env_truthy("PR_VALIDATE_NO_GPU_CACHE"):
        _log(ctx, "GPU cache disabled by PR_VALIDATE_NO_GPU_CACHE")
        return None
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        _log(ctx, "GPU cache disabled: invalid head SHA")
        return None
    head, clean = _repo_state(ctx.repo_root)
    if not head or not clean:
        _log(ctx, "GPU cache disabled: repository state is dirty or unavailable")
        return None
    if _cache_dir(ctx) is None:
        _log(ctx, "GPU cache disabled: git common directory unavailable")
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
        _log(ctx, f"GPU cache disabled: {exc}")
        return None
    # The clean checkout commit covers in-tree scripts and baselines; the
    # interpreter, host, packages and model weights cover inputs outside the tree.
    # PR/base identity, registry, step and RAM override complete the run config.
    from .steps.stress_e2e_bench import _selected_choices

    try:
        _, choices = _selected_choices()
        if not choices:
            raise ValueError("no selected models")
    except Exception:  # noqa: BLE001
        _log(ctx, "GPU cache disabled: model weights not resolvable (selection)")
        return None
    models = []
    for choice in choices:
        try:
            fingerprint = model_fingerprint(choice.model_id)
        except Exception:  # noqa: BLE001
            _log(
                ctx,
                f"GPU cache disabled: model weights not resolvable ({choice.model_id})",
            )
            return None
        models.append(
            {
                "family": choice.family,
                "model_id": choice.model_id,
                "fingerprint": fingerprint,
            }
        )
    data = {
        "schema": 2,
        "models": models,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "repo_head": head,
        "registry": registry,
        "step": step,
        "ram_gb": os.environ.get("PR_VALIDATE_RAM_GB", ""),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "node": platform.node(),
        "installed": _installed_fingerprint(),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load(key, ctx) -> StepResult | None:
    head_sha = getattr(ctx, "head_sha", "") or ""
    directory = _cache_dir(ctx)
    if directory is None:
        return None
    try:
        data = json.loads((directory / f"{key}.json").read_text())
        if (
            data["schema"] != 1
            or data["key"] != key
            or data["head_sha"] != head_sha
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
            summary=f"cached pass for head {head_sha[:12]}: {data['summary']}",
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
        "head_sha": getattr(ctx, "head_sha", "") or "",
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
        _log(ctx, f"warning: could not store GPU cache: {exc}")
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            except OSError:
                pass
