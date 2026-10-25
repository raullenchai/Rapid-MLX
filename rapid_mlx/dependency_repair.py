# SPDX-License-Identifier: Apache-2.0
"""Import-light recovery for damaged CLI environments; never loads a model."""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import os
import re
import shlex
import subprocess
import sys

_RETRY = "RAPID_MLX_DEPENDENCY_REPAIR_RETRIED"
_OUTCOMES = frozenset(
    {"repaired", "command_shown", "declined", "no_answer", "interrupted", "failed"}
)
_MISSING_SYMBOL = re.compile(r"cannot import name ['\"]([^'\"]+)['\"] from")
_DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_SYMBOL_NAME = re.compile(r"[A-Za-z_]\w*")
_ALIASES = {
    "pyyaml": "yaml",
    "pillow": "PIL",
    "huggingface-hub": "huggingface_hub",
    "opencv-python": "cv2",
    "tomli-w": "tomli_w",
    "mlx-video-with-audio": "mlx_video",
}


def declared_modules(*, active_only: bool = False) -> set[str]:
    """Use installed requirements, including absent packages without metadata."""
    requirements = metadata.requires("rapid-mlx") or []
    names = set()
    for requirement in requirements:
        # Optional extras are not broken base dependencies. Import failures for
        # platform-specific base requirements still qualify when actually used.
        if re.search(r"\bextra\s*[=!]", requirement):
            continue
        if active_only and ";" in requirement:
            try:
                from packaging.requirements import Requirement
            except ImportError:
                # packaging may itself be the broken dependency; keep the
                # requirement rather than crash while auditing it.
                Requirement = None
            if Requirement is not None:
                try:
                    marker = Requirement(requirement).marker
                except Exception:
                    # Malformed metadata: keep the name so the damage is
                    # diagnosed instead of crashing the audit.
                    marker = None
                if marker is not None and not marker.evaluate():
                    continue
        match = re.match(r"[\w.-]+", requirement)
        if match:
            name = match[0].lower().replace("_", "-")
            names.add(_ALIASES.get(name, name.replace("-", "_")))
    return names


def dependency_failure(exc: ImportError) -> bool:
    """Only declared imports qualify; model/plugin import bugs are not repaired."""
    try:
        # Unknown model architectures are plugin selection failures, not a
        # damaged dependency. Native and other package internals still qualify.
        if exc.name and exc.name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
            return False
        return bool(exc.name and exc.name.partition(".")[0] in declared_modules())
    except metadata.PackageNotFoundError:
        # An import failure plus absent own metadata is a damaged installation.
        return True


def missing_symbol(exc: ImportError) -> str | None:
    """Return the failed ``from module import name`` target, when recorded."""
    message = getattr(exc, "msg", None) or str(exc)
    match = _MISSING_SYMBOL.search(message)
    return match.group(1) if match else None


def _verification_argv(module: str, symbol: str | None) -> list[str]:
    if not _DOTTED_NAME.fullmatch(module):
        return [
            sys.executable,
            "-c",
            "import importlib,sys; importlib.import_module(sys.argv[1])",
            module,
        ]
    if symbol is not None and _SYMBOL_NAME.fullmatch(symbol):
        return [sys.executable, "-c", f"from {module} import {symbol}"]
    return [sys.executable, "-c", f"import {module}"]


def repair_argv() -> list[str] | None:
    from rapid_mlx import __version__
    from rapid_mlx._version_check import detect_install_method
    from rapid_mlx.runtime.optional_runtime import _running_in_desktop_sidecar

    if _running_in_desktop_sidecar():
        return None
    info = detect_install_method()
    spec = f"rapid-mlx=={__version__}" if __version__ != "0.0.0" else "rapid-mlx"
    if info.method == "brew":
        return ["brew", "reinstall", "rapid-mlx"]
    if info.method == "uv":
        return ["uv", "tool", "upgrade", "--reinstall", "rapid-mlx"]
    if info.method == "pipx":
        return ["pipx", "reinstall", "rapid-mlx"]
    if info.upgrade_command.startswith("sudo pipx "):
        return ["sudo", "pipx", "reinstall", "--global", "rapid-mlx"]
    # Preserve install.sh's uv-pip selection when its venv has no pip.
    argv = list(info.upgrade_argv)
    if "install" in argv and "pip" in argv:
        if (
            argv[:3] == [sys.executable, "-m", "pip"]
            and importlib.util.find_spec("pip") is None
        ):
            # A damaged venv may have lost pip too. Bootstrap it in this same
            # interpreter before reinstalling; never fall back to PATH's pip.
            return [
                sys.executable,
                "-c",
                "import ensurepip,subprocess,sys; ensurepip.bootstrap(upgrade=True); "
                "subprocess.check_call([sys.executable,'-m','pip','install',"
                "'--force-reinstall',sys.argv[1]])",
                spec,
            ]
        return [*argv[:-1], "--force-reinstall", spec]
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        spec,
    ]


def record_outcome(outcome: str) -> None:
    """Reuse failure telemetry, without paths, exception text or commands."""
    if "--no-telemetry" in sys.argv:
        return
    try:
        from rapid_mlx.telemetry.track import track

        if outcome in _OUTCOMES:
            track(
                "model_serve_failed",
                {
                    "error_class": "other",
                    "failure_subtype": "import_error",
                    "extra_recovery": outcome,
                    "failure_stage": "preflight",
                },
            )
    except Exception:
        # Missing telemetry dependencies must not prevent recovery.
        pass


def recover(
    *, allow_run: bool = True, module: str | None = None, symbol: str | None = None
) -> bool:
    """Return true only after an explicitly accepted installer succeeds."""
    argv = repair_argv()
    print(
        "Rapid-MLX's Python environment is missing or has broken dependencies.",
        file=sys.stderr,
    )
    if argv is None:
        print(
            "Update or reinstall Rapid-MLX Desktop to restore its bundled runtime.",
            file=sys.stderr,
        )
        record_outcome("command_shown")
        return False
    print(shlex.join(argv), file=sys.stderr)
    if not allow_run or not sys.stdin.isatty() or argv[0] == "sudo":
        record_outcome("command_shown" if allow_run else "failed")
        return False
    print("Run this repair now? [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        reply = sys.stdin.readline()
    except KeyboardInterrupt:
        record_outcome("interrupted")
        raise
    except (OSError, ValueError):
        reply = ""
    if not reply:
        record_outcome("no_answer")
        return False
    if reply.strip().lower() != "y":
        record_outcome("declined")
        return False
    try:
        result = subprocess.run(argv, check=False)
    except OSError:
        result = None
    if result is None or result.returncode:
        print(
            "Dependency repair failed; run the command above to inspect the error.",
            file=sys.stderr,
        )
        record_outcome("failed")
        return False
    if module is not None:
        try:
            verified = (
                subprocess.run(
                    _verification_argv(module, symbol),
                    capture_output=True,
                    timeout=60,
                    check=False,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            verified = False
        record_outcome("repaired" if verified else "failed")
        if not verified:
            print("The dependency still cannot import after repair.", file=sys.stderr)
            return False
    return True


def missing_modules() -> list[str]:
    """Check discoverability without importing native/GPU packages."""
    from rapid_mlx._version_check import detect_install_method

    try:
        modules = declared_modules(active_only=True)
    except metadata.PackageNotFoundError:
        return ["rapid-mlx distribution metadata"]
    if detect_install_method().method == "brew":
        # The text-only formula intentionally excludes these base runtimes.
        modules -= {"mlx_vlm", "mlx_video", "mflux", "torch", "torchvision", "cv2"}
    return sorted(
        module for module in modules if importlib.util.find_spec(module) is None
    )


def cli_entrypoint() -> None:
    """Catch failures even while importing the CLI, then retry in fresh Python."""
    retried = os.environ.get(_RETRY) == "1"
    try:
        from rapid_mlx.cli import cli_entrypoint as run

        run()
        os.environ.pop(_RETRY, None)
    except ImportError as exc:
        if not dependency_failure(exc):
            raise
        if recover(
            allow_run=not retried,
            module=exc.name,
            symbol=missing_symbol(exc),
        ):
            os.environ[_RETRY] = "1"
            # The tool manager may replace the running venv. Exec its lexical
            # interpreter path, preserving the original invocation and flags.
            try:
                from rapid_mlx.telemetry.posthog_sender import get_sender

                get_sender().flush(2.0)
            except Exception:
                pass
            try:
                os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
            except OSError:
                record_outcome("failed")
                print(
                    "Could not restart Python; rerun the startup command.",
                    file=sys.stderr,
                )
        raise SystemExit(1) from None
