#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the live agent/framework acceptance matrix against a release wheel.

This is deliberately separate from ``release_smoke.py``:

* ``release_smoke.py`` proves that each distribution installs and imports in
  a clean virtual environment.
* This runner proves that the *wheel produced for the release* can boot a
  real server and satisfy the strict agent + framework matrix for one model
  family.

The runner is intended for an isolated Apple-Silicon macOS release machine.
It installs the wheel into a fresh SERVER venv (release artifact + declared
extras only), starts the console script from an empty working directory, then
drives the integration clients from a SEPARATE CLIENT venv (test/SDK deps)
against the running server over HTTP.  Two isolations matter here: keeping the
server out of the source checkout (otherwise Python can import ``vllm_mlx/``
from the tree and falsely validate the source instead of the candidate
artifact), and keeping the client SDK dependencies out of the server venv
(otherwise a client's transitive dependency could satisfy a runtime import the
wheel forgot to declare, masking a missing runtime dependency).

Run one family per invocation so every matrix cell is attributable to the
model actually loaded by that server.  A release workflow fans this script
out over the supported families.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_FILES = (
    REPO_ROOT / "tests/integrations/test_agents_matrix.py",
    REPO_ROOT / "tests/integrations/test_frameworks_matrix.py",
)

# These are the client libraries that the strict matrix imports.  They remain
# test-only: none are added to the released package's runtime dependency set.
MATRIX_TEST_DEPENDENCIES = (
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.24.0",
    "openai>=1.0.0",
    "anthropic>=0.30.0",
    "langchain-openai>=0.2.0",
    "pydantic-ai>=0.0.20",
    "smolagents>=1.0.0",
    "aider-chat>=0.80.0",
)

# The base distribution exposes two canonical commands and two deprecated
# compatibility aliases. ``rapid-mlx-chat`` and ``vllm-mlx-chat`` belong to
# the optional ``chat`` extra, so testing them here would incorrectly reject a
# valid base wheel.
CLI_SMOKE_SCRIPTS = (
    "rapid-mlx",
    "rapid-mlx-bench",
    "vllm-mlx",
    "vllm-mlx-bench",
)


@dataclass(frozen=True)
class FamilyConfig:
    """One independently booted, release-eligible matrix family."""

    model: str
    server_args: tuple[str, ...] = ("--no-thinking",)
    extras: tuple[str, ...] = ()


# Keep this map aligned with tests/integrations/conftest.py::_FAMILY_ALIASES.
# The cheap aliases make the full four-family release matrix feasible on an
# isolated M3/Ultra runner; the larger golden models remain a separate perf
# and weekly-path concern.
FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    "qwen36": FamilyConfig("qwen3.5-4b-4bit"),
    "gemma4": FamilyConfig("gemma-4-12b-4bit", extras=("vision",)),
    "deepseek": FamilyConfig("deepseek-r1-32b-4bit"),
    "gptoss": FamilyConfig("gpt-oss-20b-mxfp4-q8"),
}


def validate_families_json(
    value: str, *, require_all_families: bool = False
) -> tuple[str, ...]:
    """Parse workflow matrix input and reject coverage-bypassing values.

    A dispatch-only diagnostic run may select a non-empty subset of the
    supported families. A publishing run must cover every feasible family;
    otherwise a caller could green-light a release after validating only the
    cheapest or most convenient model.
    """

    try:
        families = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"families must be a JSON array: {exc.msg}") from exc

    if not isinstance(families, list) or not families:
        raise ValueError("families must be a non-empty JSON array")
    if not all(isinstance(family, str) for family in families):
        raise ValueError("families must contain only family-name strings")
    if len(set(families)) != len(families):
        raise ValueError("families must not contain duplicates")

    unknown = sorted(set(families) - FAMILY_CONFIGS.keys())
    if unknown:
        raise ValueError(
            "unknown family name(s): "
            + ", ".join(unknown)
            + f"; choices: {', '.join(sorted(FAMILY_CONFIGS))}"
        )
    if require_all_families and set(families) != set(FAMILY_CONFIGS):
        raise ValueError(
            "publication requires every release family exactly once: "
            + ", ".join(FAMILY_CONFIGS)
        )
    return tuple(families)


def _clean_env() -> dict[str, str]:
    """Return an environment that cannot import source-tree Python packages."""

    env = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PIP_TARGET",
        "PIP_PREFIX",
    ):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    # Artifact acceptance must not contact the update endpoint or emit
    # telemetry while booting the candidate wheel.  Apart from avoiding
    # release-runner side effects, this makes the live matrix independent of
    # an operator's existing telemetry consent or transient network failures.
    env["RAPID_MLX_DISABLE_VERSION_CHECK"] = "1"
    env["RAPID_MLX_TELEMETRY"] = "0"
    return env


def _run(cmd: Sequence[str], *, cwd: Path, env: dict[str, str]) -> None:
    """Print and run a command, preserving actionable failure output."""

    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(list(cmd), cwd=cwd, env=env, check=True)


def find_release_wheel(dist_dir: Path) -> Path:
    """Return the sole rapid-mlx wheel in ``dist_dir`` or raise clearly."""

    wheels = sorted(path for path in dist_dir.glob("rapid_mlx-*.whl") if path.is_file())
    if len(wheels) != 1:
        rendered = ", ".join(path.name for path in wheels) or "<none>"
        raise ValueError(
            f"expected exactly one rapid-mlx wheel under {dist_dir}, found: {rendered}"
        )
    return wheels[0].resolve()


def _assert_port_available(port: int) -> None:
    """Fail if anything is already listening on ``port`` before we spawn.

    Without this, a stale same-family server left on the fixed release port
    could answer ``/v1/models`` with 200 while the candidate process is still
    loading (or has failed to bind), letting a broken candidate pass the
    readiness gate on another process's response. Binding the port ourselves
    proves it is genuinely free; we release it immediately so the candidate
    server can claim it.

    There is a narrow TOCTOU window between releasing this probe socket and the
    candidate server binding. It is not exploitable in practice: if a foreign
    listener grabbed the port in that window, the candidate ``rapid-mlx serve``
    would fail to bind and exit, and ``_wait_for_server``'s ``process.poll()``
    liveness check surfaces that as "candidate server exited early" rather than
    accepting the foreign listener's response. The alternative — handing the
    bound socket to the child — is not possible because ``rapid-mlx serve``
    opens its own listening socket.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # No SO_REUSEADDR: we want the bind to FAIL if the port is in use,
        # not to share it with a lingering listener.
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(
            f"release port {port} is already in use before the candidate "
            f"server is started ({exc}). Refusing to run — a stale listener "
            f"could answer the readiness probe and mask a broken candidate. "
            f"Free the port on the release runner and retry."
        ) from exc
    finally:
        sock.close()


def _wait_for_server(
    *, port: int, process: subprocess.Popen[str], log: Path, timeout: int
) -> None:
    """Wait until the candidate artifact's server exposes ``/v1/models``.

    Readiness is only accepted when (a) the spawned candidate process is
    still alive and (b) it answers ``/v1/models`` with 200. The caller has
    already proven the port was free before the spawn (see
    ``_assert_port_available``), so a 200 here can only come from the
    candidate process we launched — not a stale server that happened to
    hold the port.
    """

    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log.read_text(errors="replace")[-4000:] if log.exists() else ""
            raise RuntimeError(
                f"candidate server exited early ({process.returncode}); log tail:\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    # Re-confirm the candidate is still the live process after a
                    # successful probe — closes the window where it crashes just
                    # as (or just after) it starts answering.
                    if process.poll() is not None:
                        tail = (
                            log.read_text(errors="replace")[-4000:]
                            if log.exists()
                            else ""
                        )
                        raise RuntimeError(
                            f"candidate server exited ({process.returncode}) "
                            f"immediately after readiness; log tail:\n{tail}"
                        )
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    tail = log.read_text(errors="replace")[-4000:] if log.exists() else ""
    raise RuntimeError(
        f"candidate server did not answer {url} within {timeout}s ({last_error}); "
        f"log tail:\n{tail}"
    )


def _terminate(process: subprocess.Popen[str]) -> None:
    """Stop the server without leaving a model process on the release runner."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_family(
    *,
    dist_dir: Path,
    family: str,
    port: int,
    server_timeout: int,
    keep_venv: bool,
) -> None:
    """Install one release wheel and execute its strict matrix family slice."""

    try:
        config = FAMILY_CONFIGS[family]
    except KeyError as exc:
        raise ValueError(
            f"unknown family {family!r}; choices: {', '.join(sorted(FAMILY_CONFIGS))}"
        ) from exc

    wheel = find_release_wheel(dist_dir)
    root = Path(tempfile.mkdtemp(prefix=f"rapid-mlx-release-{family}-"))
    # Two isolated venvs, never one. The SERVER venv holds ONLY the release
    # wheel (+ its declared extras) so the boot proves the wheel's own runtime
    # dependency set is complete — nothing masks a missing runtime dep. The
    # CLIENT venv holds the matrix test/SDK dependencies; if these lived in the
    # server venv, one of their transitive deps could satisfy a runtime import
    # the wheel forgot to declare and the server would boot on borrowed deps,
    # silently defeating the clean-room guarantee this runner exists to give.
    server_venv = root / "server-venv"
    client_venv = root / "client-venv"
    server_cwd = root / "server-cwd"
    log = root / "server.log"
    process: subprocess.Popen[str] | None = None
    env = _clean_env()

    try:
        server_cwd.mkdir()

        # --- Server venv: release wheel + extras ONLY ---------------------- #
        _run([sys.executable, "-m", "venv", str(server_venv)], cwd=root, env=env)
        server_python = server_venv / "bin" / "python"
        rapid_mlx = server_venv / "bin" / "rapid-mlx"

        _run(
            [str(server_python), "-m", "pip", "install", "--upgrade", "pip"],
            cwd=root,
            env=env,
        )
        artifact_spec = str(wheel)
        if config.extras:
            artifact_spec = f"{artifact_spec}[{','.join(config.extras)}]"
        _run(
            [
                str(server_python),
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                artifact_spec,
            ],
            cwd=root,
            env=env,
        )
        if not rapid_mlx.is_file():
            raise RuntimeError(
                f"release wheel did not install rapid-mlx at {rapid_mlx}"
            )
        for script in CLI_SMOKE_SCRIPTS:
            executable = server_venv / "bin" / script
            if not executable.is_file():
                raise RuntimeError(
                    f"release wheel did not install console script {script!r}"
                )
            _run([str(executable), "--help"], cwd=root, env=env)

        # --- Client venv: matrix test + SDK deps (NEVER in the server venv) - #
        _run([sys.executable, "-m", "venv", str(client_venv)], cwd=root, env=env)
        client_python = client_venv / "bin" / "python"
        _run(
            [str(client_python), "-m", "pip", "install", "--upgrade", "pip"],
            cwd=root,
            env=env,
        )
        _run(
            [
                str(client_python),
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                *MATRIX_TEST_DEPENDENCIES,
            ],
            cwd=root,
            env=env,
        )

        # The matrix drives the running server over HTTP from the CLIENT venv.
        # Aider is a client too, so its binary comes from the client venv.
        runner_env = env | {"PATH": f"{client_venv / 'bin'}:{env.get('PATH', '')}"}

        # A release matrix must fail rather than silently omit Docker/Aider
        # cells.  The tests themselves convert missing prerequisites into
        # skips, and RAPID_MLX_MATRIX_NO_SKIPS below upgrades those skips to
        # failures.  Fail before the expensive model load when the obvious
        # host prerequisites are absent.
        missing = [
            binary
            for binary in ("docker", "aider")
            if shutil.which(binary, path=runner_env["PATH"]) is None
        ]
        if missing:
            raise RuntimeError(
                "release matrix runner is missing required executable(s): "
                + ", ".join(missing)
            )
        docker = subprocess.run(
            ["docker", "info"],
            cwd=root,
            env=runner_env,
            capture_output=True,
            text=True,
        )
        if docker.returncode != 0:
            raise RuntimeError(
                "release matrix requires a reachable Docker daemon for the "
                f"OpenHands cell: {docker.stderr[-1000:]}"
            )

        # Prove the port is free BEFORE spawning, so the readiness probe can
        # only ever be answered by the candidate process we launch (never a
        # stale same-family server lingering on the fixed release port).
        _assert_port_available(port)
        with log.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [
                    str(rapid_mlx),
                    "serve",
                    config.model,
                    "--port",
                    str(port),
                    *config.server_args,
                ],
                cwd=server_cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        _wait_for_server(port=port, process=process, log=log, timeout=server_timeout)

        matrix_env = runner_env | {
            "RAPID_MLX_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "RAPID_MLX_AGENT_MATRIX_FAMILY": family,
            "RAPID_MLX_MATRIX_STRICT": "1",
            "RAPID_MLX_MATRIX_NO_SKIPS": "1",
            "AIDER_BIN": str(client_venv / "bin" / "aider"),
        }
        _run(
            [
                str(client_python),
                "-m",
                "pytest",
                *map(str, MATRIX_FILES),
                "-v",
                "--tb=short",
            ],
            cwd=REPO_ROOT,
            env=matrix_env,
        )
    finally:
        if process is not None:
            _terminate(process)
        if keep_venv:
            print(f"[release-matrix] preserved workdir: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        help="Directory containing exactly one rapid_mlx-*.whl release candidate.",
    )
    parser.add_argument(
        "--family",
        choices=sorted(FAMILY_CONFIGS),
        help="One model family to boot and validate.",
    )
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument(
        "--server-timeout-seconds",
        type=int,
        default=900,
        help="Maximum time for a cold model download and server boot.",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Keep the temporary venv and server log for debugging.",
    )
    parser.add_argument(
        "--validate-families-json",
        help="Validate a workflow families JSON input without running a model.",
    )
    parser.add_argument(
        "--require-all-families",
        action="store_true",
        help="Require every release family exactly once (publication mode).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_families_json is not None:
        if args.dist_dir is not None or args.family is not None:
            raise ValueError(
                "--validate-families-json cannot be combined with --dist-dir or --family"
            )
        families = validate_families_json(
            args.validate_families_json,
            require_all_families=args.require_all_families,
        )
        print("[release-matrix] valid families: " + ", ".join(families))
        return 0
    if args.require_all_families:
        raise ValueError("--require-all-families requires --validate-families-json")
    if args.dist_dir is None or args.family is None:
        raise ValueError(
            "--dist-dir and --family are required to run the release matrix"
        )
    if not args.dist_dir.is_dir():
        raise ValueError(f"--dist-dir is not a directory: {args.dist_dir}")
    if args.server_timeout_seconds <= 0:
        raise ValueError("--server-timeout-seconds must be positive")
    run_family(
        dist_dir=args.dist_dir.resolve(),
        family=args.family,
        port=args.port,
        server_timeout=args.server_timeout_seconds,
        keep_venv=args.keep_venv,
    )
    print(f"[release-matrix] {args.family}: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[release-matrix] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
