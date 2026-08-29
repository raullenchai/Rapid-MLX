# SPDX-License-Identifier: Apache-2.0
"""Prepare catalog-declared audio runtime requirements.

Model servers should enter inference with every external artifact already
materialized.  This module keeps that preparation in the explicit
``rapid-mlx pull`` workflow and leaves request handling local-only.
"""

from __future__ import annotations

import importlib
import logging
import os
import signal
import subprocess
import sys

from vllm_mlx.audio.registry import AudioRuntimeRequirement

logger = logging.getLogger(__name__)


class AudioRuntimePreparationError(RuntimeError):
    """A declared runtime requirement could not be prepared safely."""


def spacy_pipeline_available(package: str) -> bool:
    """Return whether ``package`` is installed for this interpreter.

    ``spacy.util.is_package`` is also the predicate used by the downstream G2P
    runtime before it attempts its own download, so matching it prevents a
    hidden first-request network path.
    """

    import spacy.util

    return bool(spacy.util.is_package(package))


def _installer_env(
    environ: dict[str, str], prefix: str, prefix_is_venv: bool
) -> dict[str, str]:
    """Return a child environment targeting the running interpreter."""

    env = dict(environ)
    if prefix_is_venv:
        env["VIRTUAL_ENV"] = prefix
    else:
        env.pop("VIRTUAL_ENV", None)
    return env


def _run_installer(cmd: list[str], env: dict[str, str], timeout: int) -> None:
    """Run an installer in its own process group and reap it on timeout."""

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        try:
            proc.communicate(timeout=10)
        except Exception:  # noqa: BLE001 - preserve the original timeout
            pass
        for pipe in (proc.stdout, proc.stderr, proc.stdin):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:  # noqa: BLE001
                    pass
        raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)


def _prepare_spacy_pipeline(package: str) -> None:
    try:
        if spacy_pipeline_available(package):
            return
    except Exception as exc:  # noqa: BLE001 - normalize broken environments
        raise AudioRuntimePreparationError(
            f"spaCy runtime is not importable while preparing '{package}'"
        ) from exc

    env = _installer_env(
        dict(os.environ),
        sys.prefix,
        os.path.exists(os.path.join(sys.prefix, "pyvenv.cfg")),
    )
    try:
        _run_installer(
            [sys.executable, "-m", "spacy", "download", package], env, timeout=300
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ) as exc:
        status = getattr(exc, "returncode", None)
        logger.error(
            "Audio runtime preparation failed for spaCy pipeline %s: %s%s",
            package,
            type(exc).__name__,
            f" (exit {status})" if status is not None else "",
        )
        raise AudioRuntimePreparationError(
            f"Could not prepare required spaCy pipeline '{package}'"
        ) from exc

    importlib.invalidate_caches()
    try:
        available = spacy_pipeline_available(package)
    except Exception as exc:  # noqa: BLE001 - keep CLI errors path-safe
        raise AudioRuntimePreparationError(
            f"spaCy pipeline '{package}' was installed but cannot be imported"
        ) from exc
    if not available:
        raise AudioRuntimePreparationError(
            f"spaCy reported success but pipeline '{package}' is not available "
            "to the rapid-mlx interpreter"
        )


def prepare_runtime_requirement(requirement: AudioRuntimeRequirement) -> None:
    """Materialize one validated catalog requirement for offline inference."""

    if requirement.kind == "spacy_pipeline":
        _prepare_spacy_pipeline(requirement.name)
        return
    raise AudioRuntimePreparationError(
        f"Unsupported audio runtime requirement kind: {requirement.kind}"
    )
