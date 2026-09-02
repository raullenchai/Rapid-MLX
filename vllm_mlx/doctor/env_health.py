# SPDX-License-Identifier: Apache-2.0
"""Environment-health probes for ``rapid-mlx doctor``.

The whole module is a tree of cheap, side-effect-free checks: chip / OS / disk,
Python interpreter, packages installed, HF cache, network, shell integration,
optional dev tools. The user runs ``rapid-mlx doctor`` to answer one question
— "is my install/env broken?" — so every probe must:

* run in well under a second (no model load, no engine init, no server boot);
* never escalate to sudo or read user data outside ``~/.cache/huggingface``;
* report a deterministic status (✓ / ⚠ / ✗) with a one-line label.

Total wall-clock for ``rapid-mlx doctor`` ≤ 5 s on a warm cache, dominated by
the single 2-second network HEAD against ``huggingface.co`` (which downgrades
to ⚠ on timeout — never ✗).

The CLI in ``doctor/cli.py`` consumes ``run_all()`` and renders the report.
Tests in ``tests/test_doctor_env_health.py`` cover each section's probe.
"""

from __future__ import annotations

import importlib.metadata as _im
import importlib.util as _iu
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Check:
    """One row in a section. ``detail`` is shown under ``--verbose``."""

    label: str
    status: CheckStatus
    detail: str = ""


@dataclass
class Section:
    title: str
    checks: list[Check] = field(default_factory=list)

    def add(self, label: str, status: CheckStatus, detail: str = "") -> None:
        self.checks.append(Check(label=label, status=status, detail=detail))


@dataclass
class Report:
    sections: list[Section] = field(default_factory=list)

    def all_checks(self) -> list[Check]:
        return [c for s in self.sections for c in s.checks]

    @property
    def n_ok(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.OK)

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.WARN)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.all_checks() if c.status is CheckStatus.FAIL)

    @property
    def exit_code(self) -> int:
        # Spec: warnings never fail the exit code — only ✗ items do. CI scripts
        # that gate on doctor want a strict "is anything broken" signal, not
        # "is anything not perfect".
        return 1 if self.n_fail else 0


# ---------------------------------------------------------------------------
# Required + optional package matrices
# ---------------------------------------------------------------------------

# Each tuple: (PyPI distribution name, human label). The doctor doesn't
# enforce version pins here — pyproject already does that at install time.
# Showing the version is enough for the user to grep "old transformers?".
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("mlx", "mlx"),
    ("mlx-lm", "mlx-lm"),
    ("transformers", "transformers"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("rapid-mlx", "rapid-mlx"),
]

_DISTRIBUTION_MODULES: dict[str, str] = {
    "mlx": "mlx",
    "mlx-lm": "mlx_lm",
    "transformers": "transformers",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "rapid-mlx": "vllm_mlx",
    "mlx-vlm": "mlx_vlm",
    "mlx-audio": "mlx_audio",
    "mlx-embeddings": "mlx_embeddings",
    "pillow": "PIL",
}

# These are runtime compatibility contracts, not update recommendations. A
# version outside them can import successfully and still make the next server
# start fail. Keep them aligned with pyproject.toml.
_SUPPORTED_VERSIONS: dict[str, str] = {
    "mlx": ">=0.32.1,<0.33",
    "mlx-lm": ">=0.31.3,<0.32",
    "transformers": ">=5.0.0,!=5.13.0,<5.16",
    "mlx-vlm": "==0.6.17",
}

# Each tuple: (distribution, label, install hint). Missing optionals are ⚠
# (warning) not ✗ — that's the whole point of "optional". The hint is
# echoed verbatim in the report so the user can copy-paste.
OPTIONAL_PACKAGES: list[tuple[str, str, str]] = [
    ("mlx-vlm", "mlx-vlm (vision extras)", "rapid-mlx[vision]"),
    ("mlx-audio", "mlx-audio (audio extras)", "rapid-mlx[audio]"),
    (
        "mlx-embeddings",
        "mlx-embeddings (embeddings extras)",
        "rapid-mlx[embeddings]",
    ),
]

# ``find_spec`` checks discoverability without importing heavyweight audio
# packages such as scipy, numba, and spaCy.  Keep this aligned with the
# runtime dependencies declared by the ``audio`` extra in pyproject.toml.
_AUDIO_IMPORTS: tuple[tuple[str, str], ...] = (
    ("mlx-audio", "mlx_audio"),
    ("f5-tts-mlx", "f5_tts_mlx"),
    ("sounddevice", "sounddevice"),
    ("soundfile", "soundfile"),
    ("scipy", "scipy"),
    ("numba", "numba"),
    ("tiktoken", "tiktoken"),
    ("misaki", "misaki"),
    ("spacy", "spacy"),
    ("num2words", "num2words"),
    ("loguru", "loguru"),
    ("espeakng-loader", "espeakng_loader"),
    ("phonemizer-fork", "phonemizer"),
    ("cn2an", "cn2an"),
)

# The macOS desktop sidecar deliberately installs ``rapid-mlx[audio-desktop]``
# (``apps/rapid-mac/scripts/build-sidecar.sh``), a bounded extra that is just
# ``mlx-audio`` + ``soundfile`` — the general-purpose TTS stack (f5-tts-mlx,
# spaCy, misaki, numba, …) is excluded to stay under the bundle size gate.
# Grading that bundle against ``_AUDIO_IMPORTS`` reported a perfectly healthy
# signed 0.12.18 build as "incomplete — missing: f5-tts-mlx, numba, tiktoken,
# …", i.e. the exact set difference between the two extras. Keep this aligned
# with the ``audio-desktop`` extra in pyproject.toml (locked by
# ``tests/test_audio_desktop_extra.py``).
_AUDIO_DESKTOP_IMPORTS: tuple[tuple[str, str], ...] = (
    ("mlx-audio", "mlx_audio"),
    ("soundfile", "soundfile"),
)

# Distributions the desktop sidecar deliberately does NOT ship. The bundle is
# built to a hard size cap (``.github/workflows/rapid-mac-release.yml`` gates on
# BUNDLE_SIZE_CAP_MB), so ``build-sidecar.sh`` installs the bounded
# ``[audio-desktop]`` extra plus ``mlx-vlm --no-deps`` + Pillow, and nothing
# else. ``mlx-embeddings`` is simply not part of the desktop product surface.
# Reporting it as "not installed — reinstall the app" was doubly wrong: the
# install is healthy, and reinstalling reproduces the identical warning
# forever. These rows are informational, never ⚠.
_DESKTOP_EXCLUDED_DISTS: frozenset[str] = frozenset({"mlx-embeddings"})

# Where a bundled sidecar came from decides what "repair" even means, so the
# two sources must not share one hint.
#
# * ``embedded`` — ``Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/``.
#   Inside the code-signed, notarized app; pip-installing into it breaks the
#   signature seal and Gatekeeper then rejects the app ("a sealed resource is
#   missing or invalid"). Reinstalling the app genuinely replaces it.
# * ``runtime-override`` — ``~/Library/Application Support/Rapid/
#   runtime-override/rapid-mlx/``, written by the bootstrapper on first launch
#   of the slim DMG. It lives OUTSIDE the app bundle and deliberately "survives
#   desktop upgrades" (``ServerLocator.swift``), and the bootstrapper
#   short-circuits its download while the cache is present
#   (``build-bootstrapper-dmg.sh``). So "reinstall the .app" is actively wrong
#   here — it typically changes nothing.
#
#   There is no in-app "repair the runtime" action to point at, and no
#   automatic re-download either. Settings → check for updates hands off to
#   Sparkle, which updates the *app* bundle, and ``UpdateChecker`` explicitly
#   does not act on the manifest's ``sidecar_*`` fields. The bootstrapper that
#   originally populated this slot is not part of the source tree any more, and
#   the release workflow builds only the full DMG. So "remove it and relaunch"
#   is NOT self-repairing on its own: with no bundled sidecar the desktop just
#   lands on the missing-runtime overlay, whose only actions are Recheck and an
#   app-update download.
#
#   What does work today, in this order: install the current
#   Rapid-MLX Desktop.app — its DMG ships a sidecar at
#   ``Contents/Resources/rapid-mlx/`` (``build.sh``) — and only then remove the
#   override, so ``ServerLocator.find()`` resolves the bundled slot. Doing it
#   the other way round leaves a slim install stranded. Note that reinstalling
#   alone is not enough when the override's VERSION is equal or newer: it keeps
#   winning over the bundled copy (``shouldPreferBundled``), which is exactly
#   why the removal step is required.
_EMBEDDED_REPAIR_HINT = (
    "reinstall Rapid-MLX Desktop.app — the bundled sidecar's Python "
    "environment is code-signed and must not be pip-installed into"
)
_RUNTIME_OVERRIDE_REPAIR_HINT_TEMPLATE = (
    "this runtime at {root} lives outside the app bundle and no app update "
    "replaces it — install the current Rapid-MLX Desktop.app (its DMG ships a "
    "sidecar), then remove {root} and relaunch so the bundled sidecar is used"
)
_DOCTOR_BUDGET_S = 5.0
# Leave enough wall-clock headroom for subprocess timeout cleanup, report
# assembly, and CLI rendering.  ``subprocess.run(timeout=...)`` only starts
# terminating the child at its timeout and can return a few milliseconds
# later; consuming the full user-facing budget inside probes therefore makes
# the end-to-end command exceed its own contract.
_DOCTOR_COMPLETION_HEADROOM_S = 0.1
_DOCTOR_DEADLINE: float | None = None
_DOCTOR_RUN_LOCK = threading.Lock()
_RUNTIME_CONTEXTS: dict[Path, tuple[Path, dict[str, str]]] = {}
_RUNTIME_DISTRIBUTION_CACHE: dict[Path, bool] = {}
_TRUSTED_SYS_PATH_ROOTS: tuple[Path, ...] = ()
_SELECTED_RUNTIME: Path | None = None
_SELECTED_SERVER_RUNTIME = False
_RUNTIME_SELECTION_DONE = False


def _runtime_uses_context(runtime: Path) -> bool:
    """Whether a runtime requires its captured server context for probing."""
    return runtime != Path(sys.executable).absolute() or runtime in _RUNTIME_CONTEXTS


def _runtime_has_rapid_mlx_distribution(
    runtime: Path,
    cwd: Path,
    env: dict[str, str],
) -> bool:
    """Return whether *runtime* has ``vllm_mlx`` registered by a Rapid-MLX wheel.

    ``python -m vllm_mlx.cli`` is a useful process signature, but by itself it
    would also match an unrelated top-level module. This probe imports only
    packaging metadata; it never imports the server or a user's module.
    """
    cache_key = runtime.absolute()
    if cache_key in _RUNTIME_DISTRIBUTION_CACHE:
        return _RUNTIME_DISTRIBUTION_CACHE[cache_key]
    if _DOCTOR_DEADLINE is not None and time.monotonic() >= _DOCTOR_DEADLINE:
        return False
    if runtime == Path("/usr/bin/python3") and cwd != Path("/"):
        if _runtime_has_rapid_mlx_distribution(
            Path("/usr/bin/python3"),
            Path("/"),
            {"PATH": env.get("PATH", os.environ.get("PATH", "/usr/bin:/bin"))},
        ):
            return True
    try:
        result = subprocess.run(  # noqa: S603 — runtime path is validated by caller
            [
                str(runtime),
                "-I",
                "-c",
                "import importlib.metadata, json, sys; "
                "print(json.dumps(importlib.metadata.packages_distributions().get("
                "'vllm_mlx', [])))",
            ],
            capture_output=True,
            text=True,
            timeout=_bounded_timeout(3),
            cwd=str(cwd),
            env={"PATH": env.get("PATH", os.environ.get("PATH", "/usr/bin:/bin"))},
            check=True,
        )
        names = json.loads(result.stdout)
        installed = isinstance(names, list) and bool(
            {str(name).lower().replace("_", "-") for name in names}
            & {"rapid-mlx", "vllm-mlx"}
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        installed = False
    _RUNTIME_DISTRIBUTION_CACHE[cache_key] = installed
    return installed


def _runtime_environment(
    exe: Path,
    prefix: Path | None = None,
    base_prefix: Path | None = None,
) -> str:
    if _bundled_sidecar_root(exe) is not None:
        return "desktop sidecar"
    home = Path.home()
    application_bin = home / ".rapid-mlx" / "bin"
    runtime_root = (home / ".rapid-mlx-python").resolve()
    if (
        exe in {application_bin / "python", application_bin / "python3"}
        or exe.parent == application_bin
    ):
        return "Rapid-MLX application environment"
    effective_prefix = prefix if prefix is not None else Path(sys.prefix).resolve()
    effective_base_prefix = Path(
        str(base_prefix or getattr(sys, "base_prefix", "") or sys.prefix)
    ).resolve()
    if exe == runtime_root / "bin" / "python3" or effective_prefix == runtime_root:
        return "Rapid-MLX runtime environment"
    project = Path(__file__).resolve().parents[2]
    if (project / "pyproject.toml").is_file() and project in exe.parents:
        return "developer installation"
    if effective_prefix != effective_base_prefix:
        return "virtual environment"
    return "system environment"


def _module_available(
    module: str,
    runtime: Path | None = None,
    *,
    real_import: bool = False,
) -> bool:
    """Return whether *module* is discoverable, without importing it."""
    if runtime is not None and _runtime_uses_context(runtime):
        probe = _probe_runtime(
            runtime,
            _bundled_sidecar_root(runtime),
        )
        if not probe:
            return False
        return _runtime_module_importable(
            runtime,
            module,
            _bundled_sidecar_root(runtime),
        )
    try:
        if real_import:
            if not _module_origin_is_trusted(module):
                return False
            return _runtime_module_importable(
                Path(sys.executable).absolute(),
                "PIL.Image" if module == "PIL" else module,
                None,
                trusted_roots=_TRUSTED_SYS_PATH_ROOTS,
                exercise=module == "PIL",
            )
        return _iu.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False
    except (Exception, SystemExit):
        return False


def _module_origin_is_trusted(module: str) -> bool:
    """Reject shadow modules outside the active runtime/package roots."""
    try:
        spec = _iu.find_spec(module)
        if spec is None:
            return False
        locations: list[Path] = []
        if spec.origin:
            locations.append(Path(spec.origin).parent)
        locations.extend(
            Path(location) for location in spec.submodule_search_locations or []
        )
        trusted_roots = set(_TRUSTED_SYS_PATH_ROOTS)
        sidecar_root = _bundled_sidecar_root()
        if sidecar_root is not None:
            trusted_roots.add(sidecar_root.resolve())
        vllm_mlx_source = Path(__file__).resolve().parents[1]
        trusted_roots.add(vllm_mlx_source)
        return any(
            location.resolve().is_relative_to(trusted_root)
            for location in locations
            for trusted_root in trusted_roots
        )
    except (OSError, ValueError, ImportError):
        return False


def _trusted_sys_path_roots() -> set[Path]:
    """Return safe active roots from which local real imports may execute."""
    unsafe_roots: set[Path] = set()
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry:
            unsafe_roots.add(Path(entry).expanduser().resolve())
    unsafe_roots.add(Path.cwd().resolve())
    unsafe_roots.add(Path(__file__).resolve().parents[2])

    trusted_roots: set[Path] = set()
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            continue
        resolved = candidate.resolve()
        if resolved in unsafe_roots:
            continue
        trusted_roots.add(resolved)
    return trusted_roots


_TRUSTED_SYS_PATH_ROOTS = tuple(_trusted_sys_path_roots())


def _is_diagnostic_python_override(candidate: Path) -> bool:
    """Accept an explicit Python path; runtime probing reports real failures."""
    if not candidate.is_file() or not candidate.name.lower().startswith("python"):
        return False
    return True


def _filesystem_runtime_has_rapid_mlx_distribution(runtime: Path) -> bool:
    """Validate a runtime's Rapid-MLX install without executing it."""
    runtime = runtime.absolute()
    if _bundled_sidecar_root(runtime) is not None:
        return True
    for parent in (runtime.parent, *runtime.parents):
        candidates = [
            *parent.glob("lib/python3.*/site-packages"),
            *parent.glob("lib/python3/dist-packages"),
            *parent.glob("lib/site-packages"),
            *parent.glob("site-packages"),
        ]
        for site_root in candidates:
            package_root = site_root / "vllm_mlx"
            if not (package_root / "cli.py").is_file():
                continue
            if any(site_root.glob("rapid_mlx-*.dist-info")):
                return True
    return False


def _is_trusted_runtime_executable(runtime: Path) -> bool:
    """Authenticate a candidate before doctor executes any probe."""
    return _filesystem_runtime_has_rapid_mlx_distribution(runtime)


def _runtime_python_path() -> Path:
    """Return the authoritative Python executable for runtime checks.

    A running Rapid-MLX server wins because its interpreter defines the
    production dependency set. An explicit override covers a stopped server;
    the CLI interpreter is the fallback for the common single-runtime setup.
    """
    global _SELECTED_SERVER_RUNTIME
    _SELECTED_SERVER_RUNTIME = False
    try:
        import psutil

        def _is_installed_rapid_mlx_entrypoint(entry: Path) -> bool:
            if not entry.is_file():
                return False
            try:
                content = entry.read_bytes()[:8192].decode(errors="ignore")
            except OSError:
                return False
            imported_names = re.findall(
                r"^from\s+vllm_mlx\.cli\s+import\s+([^#\n]+)$",
                content,
                re.M,
            )
            imported = {
                name.strip()
                for part in imported_names
                for name in part.split(",")
                if name.strip()
            }
            return bool(
                imported
                and imported <= {"main", "cli_entrypoint"}
                and re.search(
                    r"^[ \t]*(?:(?:main|cli_entrypoint)\(\)|"
                    r"sys\.exit\((?:main|cli_entrypoint)\(\)\))[ \t]*$",
                    content,
                    re.M,
                )
            )

        def _is_installed_rapid_mlx_module(entry: Path) -> bool:
            package_root = entry.parent
            site_root = package_root.parent
            try:
                content = (
                    package_root.joinpath("cli.py")
                    .read_bytes()[:8192]
                    .decode(errors="ignore")
                )
            except OSError:
                return False
            return (
                package_root.joinpath("__init__.py").is_file()
                and "from vllm_mlx." in content
                and any(site_root.glob("rapid_mlx-*.dist-info"))
            )

        def _python_sibling(entry: Path) -> Path | None:
            for name in ("python3.12", "python3", "python"):
                candidate = entry.with_name(name)
                if candidate.is_file():
                    return candidate.absolute()
            return None

        def _python_from_entrypoint(entry: Path, process: Any) -> Path | None:
            try:
                shebang = entry.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, UnicodeDecodeError, IndexError):
                return None
            if not shebang.startswith("#!"):
                return _python_sibling(entry)
            shebang_parts = shebang[2:].strip().split()
            if len(shebang_parts) >= 1 and Path(shebang_parts[0]).name.startswith(
                "env"
            ):
                process_exe = Path(process.exe()).absolute()
                if process_exe.is_file() and process_exe.name.lower().startswith(
                    "python"
                ):
                    return process_exe
                return _python_sibling(entry)
            if shebang_parts:
                interpreter = Path(shebang_parts[0]).absolute()
                if interpreter.is_file() and interpreter.name.lower().startswith(
                    "python"
                ):
                    return interpreter
            return _python_sibling(entry)

        def _python_from_module_command(
            command: list[str], context_env: dict[str, str]
        ) -> Path | None:
            interpreter = Path(command[0])
            if not interpreter.is_absolute():
                found = shutil.which(command[0], path=context_env.get("PATH", ""))
                if found is None:
                    return None
                interpreter = Path(found)
            return interpreter.absolute()

        def _module_command_entry(command: list[str]) -> str | None:
            value_options = {"-X", "-W", "--check-hash-based-pycs"}
            index = 1
            while index < len(command):
                argument = command[index]
                if argument == "-m":
                    return command[index + 1] if index + 1 < len(command) else None
                if argument in value_options:
                    index += 2
                    continue
                if argument.startswith("-"):
                    index += 1
                    continue
                return None
            return None

        def _runtime_candidate(cmdline: list[str], process: Any) -> Path | None:
            if hasattr(os, "getuid") and hasattr(process, "uids"):
                try:
                    if process.uids().real != os.getuid():
                        return None
                except Exception:
                    return None
            try:
                context_env = {
                    str(key): str(value) for key, value in process.environ().items()
                }
                context_cwd = Path(process.cwd()).resolve()
            except Exception:
                return None
            try:
                serve_index = cmdline.index("serve")
            except ValueError:
                return None
            command = cmdline[:serve_index]
            if not command:
                return None
            entry_argument = command[-1]
            entry = Path(entry_argument)
            module_entry = _module_command_entry(command)
            if module_entry is not None:
                entry_argument = module_entry
                entry = Path(module_entry)
            if entry.name == "rapid-mlx" and not entry.is_absolute():
                if "/" in entry_argument:
                    entry = context_cwd / entry
                else:
                    found = shutil.which(
                        entry_argument, path=context_env.get("PATH", "")
                    )
                    if found is None:
                        return None
                    entry = Path(found)
            if entry.name == "rapid-mlx":
                if not _is_installed_rapid_mlx_entrypoint(entry):
                    return None
                candidate = _python_from_entrypoint(entry, process)
            elif (module_entry is not None and entry.name == "vllm_mlx.cli") or (
                len(command) >= 2
                and entry.name == "cli.py"
                and entry.parent.name == "vllm_mlx"
                and _is_installed_rapid_mlx_module(entry)
            ):
                candidate = _python_from_module_command(command, context_env)
            else:
                return None
            if candidate is None:
                candidate = Path(process.exe()).absolute()
            if (
                not candidate.is_absolute()
                or not candidate.is_file()
                or not candidate.name.lower().startswith("python")
                or not (
                    _is_trusted_runtime_executable(candidate)
                    or _runtime_has_rapid_mlx_distribution(
                        candidate,
                        context_cwd,
                        context_env,
                    )
                )
            ):
                return None
            return candidate

        candidates: list[tuple[float, Path, Path, dict[str, str]]] = []
        for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                if (
                    _DOCTOR_DEADLINE is not None
                    and time.monotonic() >= _DOCTOR_DEADLINE
                ):
                    break
                cmdline = process.info.get("cmdline") or []
                if process.info["pid"] == os.getpid():
                    continue
                candidate = _runtime_candidate(cmdline, process)
                if candidate is None:
                    continue
                context_env = {
                    str(key): str(value) for key, value in process.environ().items()
                }
                context_cwd = Path(process.cwd()).resolve()
                candidates.append(
                    (
                        float(process.info["create_time"]),
                        candidate,
                        context_cwd,
                        context_env,
                    )
                )
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
                OSError,
                TypeError,
            ):
                continue
        if candidates:
            _created, selected, selected_cwd, selected_env = max(
                candidates, key=lambda item: item[0]
            )
            _RUNTIME_CONTEXTS[selected] = (selected_cwd, selected_env)
            _SELECTED_SERVER_RUNTIME = True
            return selected
    except (Exception, SystemExit):
        pass

    override = os.environ.get("RAPID_MLX_RUNTIME_PYTHON", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if _is_diagnostic_python_override(candidate):
            return candidate.absolute()

    return Path(sys.executable).absolute()


def _selected_runtime() -> tuple[Path, bool]:
    """Return the cached runtime selection for one doctor invocation."""
    global _SELECTED_RUNTIME, _RUNTIME_SELECTION_DONE
    if not _RUNTIME_SELECTION_DONE:
        _SELECTED_RUNTIME = _runtime_python_path()
        _RUNTIME_SELECTION_DONE = True
    selected_runtime = _SELECTED_RUNTIME
    if selected_runtime is None:
        selected_runtime = _runtime_python_path()
        _SELECTED_RUNTIME = selected_runtime
    return selected_runtime, _SELECTED_SERVER_RUNTIME


_PROBE_SCRIPT = """\
import importlib
import importlib.util
import importlib.metadata
import json
import sys
from pathlib import Path

probe_paths = json.loads(sys.argv[2])
trusted_roots = [root for root in probe_paths["trusted"] if root]
trusted_metadata_roots = list(trusted_roots)
for site_root in trusted_roots:
    sys.path.insert(0, site_root)
trusted_metadata_roots.extend(sys.path)
module_trusted_roots = [*trusted_roots, *sys.path]
for site_root in reversed(probe_paths["context"]):
    sys.path.insert(0, site_root)
context_metadata_roots = [root for root in probe_paths["context"] if root]
trusted_roots = [
    str(Path(root).resolve()) for root in module_trusted_roots
]

def _path_is_trusted(path):
    resolved = Path(path).resolve()
    return any(resolved == Path(root) or resolved.is_relative_to(root) for root in trusted_roots)

def _module_path_is_trusted(spec):
    locations = []
    if spec.origin:
        locations.append(str(Path(spec.origin).parent))
    if spec.submodule_search_locations:
        locations.extend(spec.submodule_search_locations)
    return any(_path_is_trusted(location) for location in locations)

distributions = json.loads(sys.argv[1])

def distribution_version(name, module_path):
    for metadata_roots in (trusted_metadata_roots, context_metadata_roots):
        for distribution in importlib.metadata.distributions(path=metadata_roots):
            dist_name = (distribution.metadata.get("Name") or "").lower()
            if dist_name != name.lower() or not _distribution_owns_module(
                distribution, module_path
            ):
                continue
            return distribution.version
    return None

def _distribution_owns_module(distribution, module_path):
    if module_path is None:
        return False
    module_path = Path(module_path).resolve()
    owned_paths = [module_path]
    if module_path.name == "__init__.py":
        owned_paths.append(module_path.parent)
    for installed_file in distribution.files or []:
        try:
            installed_path = Path(distribution.locate_file(installed_file)).resolve()
        except (Exception, SystemExit):
            continue
        for owned_path in owned_paths:
            if installed_path == owned_path or installed_path.is_relative_to(
                owned_path
            ):
                return True
    return False

packages = {}
for distribution, module_name in distributions.items():
    version = None
    discoverable = False
    spec = None
    try:
        spec = importlib.util.find_spec(module_name)
        version = None if spec is None else distribution_version(
            distribution,
            spec.origin,
        )
    except importlib.metadata.PackageNotFoundError:
        version = None
    except (Exception, SystemExit):
        version = None
    discoverable = spec is not None
    try:
        trusted_origin = spec is not None and _module_path_is_trusted(spec)
    except (Exception, SystemExit):
        spec = None
        discoverable = False
        trusted_origin = False
    packages[distribution] = {
        "importable": None,
        "discoverable": discoverable,
        "trusted_origin": trusted_origin,
        "module": module_name,
        "version": version,
    }
print(json.dumps({
    "executable": sys.executable,
    "base_prefix": sys.base_prefix,
    "packages": packages,
    "path": sys.path,
    "prefix": sys.prefix,
}))
"""

_RUNTIME_PACKAGES: dict[str, str] = dict(_DISTRIBUTION_MODULES)
for _audio_dist, _audio_module in (*_AUDIO_IMPORTS, *_AUDIO_DESKTOP_IMPORTS):
    _RUNTIME_PACKAGES.setdefault(_audio_dist, _audio_module)
_RUNTIME_PROBE_CACHE: dict[
    tuple[Path, Path | None, tuple[Path, ...]], dict[str, object] | None
] = {}
_PROBE_RESULT_PREFIX = "__RAPID_MLX_IMPORT_RESULT__"
_RUNTIME_IMPORT_SCRIPT = """\
import importlib
import importlib.util
import json
import sys
from pathlib import Path

_PROBE_SENTINEL = "__RAPID_MLX_IMPORT_RESULT__"
module_name = sys.argv[1]
probe_paths = json.loads(sys.argv[2])
exercise = module_name == "PIL.Image"
trusted_roots = [root for root in probe_paths["trusted"] if root]
trusted_roots = [str(Path(root).resolve()) for root in [*trusted_roots, *sys.path]]
for root in reversed(trusted_roots):
    sys.path.insert(0, root)

def _path_is_trusted(path):
    resolved = Path(path).resolve()
    return any(
        resolved == Path(root) or resolved.is_relative_to(root)
        for root in trusted_roots
    )

def _module_path_is_trusted(spec):
    locations = []
    if spec.origin:
        locations.append(str(Path(spec.origin).parent))
    if spec.submodule_search_locations:
        locations.extend(spec.submodule_search_locations)
    return any(_path_is_trusted(location) for location in locations)

def _emit_import_result(payload):
    print(_PROBE_SENTINEL + json.dumps(payload))

spec = importlib.util.find_spec(module_name)
if spec is None or not _module_path_is_trusted(spec):
    _emit_import_result({"importable": False, "trusted_origin": False})
else:
    if exercise:
        import PIL.Image as Image

        Image.new("RGB", (1, 1))
        _emit_import_result({"importable": True, "trusted_origin": True})
    else:
        try:
            importlib.import_module(module_name)
        except (Exception, SystemExit):
            _emit_import_result({"importable": False, "trusted_origin": True})
            sys.exit(0)
        _emit_import_result({"importable": True, "trusted_origin": True})
"""
_RUNTIME_IMPORT_CACHE: dict[
    tuple[Path, str, str, bool, bool, tuple[str, ...]],
    bool,
] = {}
_RUNTIME_IMPORT_TIMEOUTS: set[tuple[Path, str, str, bool, bool, tuple[str, ...]]] = (
    set()
)


def _bounded_timeout(default_s: float) -> float:
    """Return a positive timeout that never meaningfully exceeds the budget."""
    if _DOCTOR_DEADLINE is None:
        return default_s
    return max(0.001, min(default_s, _DOCTOR_DEADLINE - time.monotonic()))


def _import_probe_cache_key(
    runtime: Path,
    module: str,
    sidecar_root: Path | None,
    *,
    trusted_roots: tuple[Path, ...] = (),
    exercise: bool = False,
    isolated: bool = True,
) -> tuple[Path, str, str, bool, bool, tuple[str, ...]]:
    return (
        runtime,
        module,
        str(sidecar_root.resolve()) if sidecar_root else "",
        exercise,
        isolated,
        tuple(sorted(str(root.resolve()) for root in trusted_roots)),
    )


def _import_probe_was_interrupted(
    runtime: Path,
    module: str,
    sidecar_root: Path | None,
    *,
    trusted_roots: tuple[Path, ...] = (),
    exercise: bool = False,
    isolated: bool = True,
) -> bool:
    cache_key = _import_probe_cache_key(
        runtime,
        module,
        sidecar_root,
        trusted_roots=trusted_roots,
        exercise=exercise,
        isolated=isolated,
    )
    return cache_key in _RUNTIME_IMPORT_TIMEOUTS


def _probe_package(
    probe: dict[str, object],
    distribution: str,
) -> dict[str, object] | None:
    packages = probe.get("packages")
    if not isinstance(packages, dict):
        return None
    package = packages.get(distribution)
    if not isinstance(package, dict):
        return None
    return cast("dict[str, object]", package)


def _probe_package_by_module(
    probe: dict[str, object],
    module: str,
) -> dict[str, object] | None:
    packages = probe.get("packages")
    if not isinstance(packages, dict):
        return None
    for package in packages.values():
        if isinstance(package, dict) and package.get("module") == module:
            return cast("dict[str, object]", package)
    return None


def _runtime_module_importable(
    runtime: Path,
    module: str,
    sidecar_root: Path | None,
    *,
    trusted_roots: tuple[Path, ...] = (),
    exercise: bool = False,
    isolated: bool = True,
) -> bool:
    """Import one trusted module in *runtime*, independently of other probes."""
    cache_key = _import_probe_cache_key(
        runtime,
        module,
        sidecar_root,
        trusted_roots=trusted_roots,
        exercise=exercise,
        isolated=isolated,
    )
    if cache_key in _RUNTIME_IMPORT_CACHE:
        return _RUNTIME_IMPORT_CACHE[cache_key]
    _RUNTIME_IMPORT_TIMEOUTS.discard(cache_key)
    if _DOCTOR_DEADLINE is not None and time.monotonic() >= _DOCTOR_DEADLINE:
        _RUNTIME_IMPORT_CACHE[cache_key] = False
        _RUNTIME_IMPORT_TIMEOUTS.add(cache_key)
        return False
    importable = False
    try:
        command = [str(runtime)]
        if isolated:
            command.append("-I")
        command.extend(
            [
                "-c",
                _RUNTIME_IMPORT_SCRIPT,
                "PIL.Image" if exercise else module,
                json.dumps(
                    {
                        "trusted": [str(sidecar_root / "site-packages")]
                        if sidecar_root
                        else [str(root) for root in trusted_roots],
                    }
                ),
            ]
        )
        result = subprocess.run(  # noqa: S603 — runtime path is caller-validated
            command,
            capture_output=True,
            text=True,
            timeout=_bounded_timeout(10),
            env={
                "HOME": os.environ.get("HOME", str(Path.home())),
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            },
            cwd="/",
            check=True,
        )
        probe_lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith(_PROBE_RESULT_PREFIX)
        ]
        if not probe_lines:
            importable = False
            _RUNTIME_IMPORT_CACHE[cache_key] = importable
            return importable
        result_json = json.loads(probe_lines[-1].removeprefix(_PROBE_RESULT_PREFIX))
        if not isinstance(result_json, dict):
            importable = False
        else:
            importable = bool(result_json.get("importable"))
    except subprocess.TimeoutExpired:
        importable = False
        _RUNTIME_IMPORT_TIMEOUTS.add(cache_key)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        importable = False
    _RUNTIME_IMPORT_CACHE[cache_key] = importable
    return importable


def _probe_runtime(
    runtime: Path,
    sidecar_root: Path | None = None,
) -> dict[str, object] | None:
    """Inspect one interpreter without importing the server runtime."""
    sanitized_context = tuple(_server_import_paths(runtime))
    cache_key = (
        runtime,
        sidecar_root.resolve() if sidecar_root else None,
        sanitized_context,
    )
    cache = _RUNTIME_PROBE_CACHE
    if cache_key in cache:
        return cache[cache_key]
    try:
        if _DOCTOR_DEADLINE is not None and time.monotonic() >= _DOCTOR_DEADLINE:
            cache[cache_key] = None
            return None
        env = {
            "HOME": os.environ.get("HOME", str(Path.home())),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        result = subprocess.run(  # noqa: S603 — runtime path is resolved above
            [
                str(runtime),
                "-I",
                "-c",
                _PROBE_SCRIPT,
                json.dumps(_RUNTIME_PACKAGES),
                json.dumps(
                    {
                        "trusted": [str(sidecar_root / "site-packages")]
                        if sidecar_root
                        else [],
                        "context": [
                            str(path) for path in _server_import_paths(runtime)
                        ],
                    }
                ),
            ],
            capture_output=True,
            text=True,
            timeout=_bounded_timeout(20),
            env=env,
            cwd="/",
            check=True,
        )
        probe: object = json.loads(result.stdout)
        if not isinstance(probe, dict):
            cache[cache_key] = None
            return None
        typed_probe = cast("dict[str, object]", probe)
        cache[cache_key] = typed_probe
        return typed_probe
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        cache[cache_key] = None
        return None


def _server_import_paths(runtime: Path) -> list[Path]:
    """Return sanitized server-context paths for a selected remote runtime."""
    context = _RUNTIME_CONTEXTS.get(runtime)
    if not context:
        return []
    cwd, context_env = context
    paths = [cwd]
    pythonpath = context_env.get("PYTHONPATH", "")
    paths.extend(
        (cwd / entry).resolve() for entry in pythonpath.split(os.pathsep) if entry
    )
    unique_paths: list[Path] = []
    for path in paths:
        if path.is_absolute() and path not in unique_paths:
            unique_paths.append(path)
    return unique_paths


def _bundled_sidecar_root(python: Path | None = None) -> Path | None:
    """Return the sidecar bundle root when doctor is running from a managed
    sidecar's embedded interpreter, else ``None``.

    The bundle layout produced by ``build-sidecar.sh`` is fixed, and both
    managed slots share it (``ServerLocator.swift`` resolves each through the
    same ``rapid-mlx/bin/rapid-mlx`` suffix)::

        <root>/bin/rapid-mlx        # shell shim (sidecar-shim.sh)
        <root>/python/bin/python3.12
        <root>/site-packages/

    We fingerprint that shape off ``sys.executable`` rather than trusting an
    env var: the desktop is not the only thing that spawns the sidecar (the
    user can run the shim by hand), so any env-var marker would be absent in
    exactly the cases that matter.  The shape alone is not provenance,
    though: a custom Python installation can reproduce it.  Require one of
    the two locations ``ServerLocator`` owns before changing doctor contracts.
    """
    try:
        exe = (python or Path(sys.executable)).resolve()
    except OSError:
        return None
    if len(exe.parents) < 3:
        return None
    if exe.parent.name != "bin" or exe.parents[1].name != "python":
        return None
    root = exe.parents[2]
    if not (root / "site-packages").is_dir():
        return None
    if not (root / "bin" / "rapid-mlx").exists():
        return None

    # Full DMG: <any app name>.app/Contents/Resources/rapid-mlx.  Users may
    # rename the app after installation, so the stable macOS bundle suffix is
    # the provenance signal rather than the display name.
    embedded_shape = (
        root.parent.name == "Resources"
        and root.parents[1].name == "Contents"
        and root.parents[2].suffix == ".app"
    )
    embedded = False
    if embedded_shape:
        try:
            with (root.parents[1] / "Info.plist").open("rb") as handle:
                plist = plistlib.load(handle)
        except (OSError, plistlib.InvalidFileException, ValueError, TypeError):
            plist = None
        bundle_id = plist.get("CFBundleIdentifier") if isinstance(plist, dict) else None
        # Production is exact; dogfood isolation appends a suffix so multiple
        # personas can coexist without sharing preferences.
        embedded = isinstance(bundle_id, str) and (
            bundle_id == "com.rapidmlx.rapid"
            or bundle_id.startswith("com.rapidmlx.rapid.dogfood-")
        )

    # Runtime override: ApplicationSupportLocator may root this under a
    # dogfood HOME, but the suffix below is invariant.  Match the complete
    # product-owned suffix so an arbitrary ``runtime-override/rapid-mlx``
    # directory is not enough to suppress ordinary CLI diagnostics.
    runtime_override = False
    home = os.environ.get("HOME", "").strip()
    if home:
        try:
            expected_override = (
                Path(home).expanduser().resolve()
                / "Library"
                / "Application Support"
                / "Rapid"
                / "runtime-override"
                / "rapid-mlx"
            ).resolve()
        except OSError:
            expected_override = None
        runtime_override = root == expected_override
    return root if embedded or runtime_override else None


def _sidecar_repair_hint(root: Path) -> str:
    """Pick the repair hint matching which managed slot *root* sits in.

    ``runtime-override`` is detected by its parent directory name rather than
    by a full absolute-path match against ``~/Library/Application Support`` —
    the desktop resolves Application Support through ``ApplicationSupportLocator``
    which honours an overridden ``$HOME`` (dogfood / test launches), so a
    hardcoded home-relative path would misclassify exactly those runs. Anything
    that is not the override slot is the in-bundle copy.

    The override hint embeds the resolved path because doctor already knows it
    exactly and the recovery is a manual removal — a generic "remove the cached
    runtime" would leave the user guessing at a location that moves with
    ``$HOME``.
    """
    if root.parent.name == "runtime-override":
        return _RUNTIME_OVERRIDE_REPAIR_HINT_TEMPLATE.format(root=root)
    return _EMBEDDED_REPAIR_HINT


# ---------------------------------------------------------------------------
# Section: System
# ---------------------------------------------------------------------------


def _detect_apple_silicon() -> tuple[str | None, int | None]:
    """Return (chip_brand, ram_gb) or (None, None) on non-Mac / sysctl failure.

    ``sysctl`` is a system binary that's always present on macOS; we use it
    instead of ``platform.processor()`` because the latter returns ``''`` on
    arm64 macOS in some Python builds (CPython issue #97965, present in 3.10+).
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None, None
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_bounded_timeout(2),
            check=False,
        )
        memsize = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_bounded_timeout(2),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    brand_str = brand.stdout.strip() or None
    try:
        ram_gb: int | None = round(int(memsize.stdout.strip()) / (1024**3))
    except (TypeError, ValueError):
        ram_gb = None
    return brand_str, ram_gb


def _disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except (OSError, FileNotFoundError):
        return None


def _hf_cache_dir() -> Path:
    """Return the HuggingFace **hub** cache dir.

    Resolution order matches huggingface_hub itself:

      1. ``$HF_HUB_CACHE`` (the most specific override; some users point this
         at an external SSD while leaving ``HF_HOME`` alone).
      2. ``$HF_HOME/hub`` (the canonical sub-path under a custom HF_HOME).
      3. ``~/.cache/huggingface/hub`` (the default the hub library writes to).

    Earlier revisions returned ``~/.cache/huggingface`` (no ``hub`` suffix).
    That was wrong: real downloads land in the ``hub`` subdir, so a missing
    or unwritable hub would have been masked by a probe that checked the
    parent. Codex-review round 1 caught this; the env-var fall-through plus
    the trailing ``hub`` segment fix both problems at once.
    """
    env_hub_cache = os.environ.get("HF_HUB_CACHE")
    if env_hub_cache:
        return Path(env_hub_cache).expanduser()
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


# Wall-clock budget for the recursive HF-cache size walk. The whole doctor
# run is contracted at ≤ 5 s and the network probe alone can spend 2 s, so
# the cache walk must finish in ~1 s on a hot FS / abort cleanly on a cold
# or network-mounted cache. Codex-review round 1 flagged the previous
# unbounded walk as a contract violation on TB-scale caches.
_CACHE_WALK_BUDGET_S = 1.5


def _dir_size_gb(path: Path, *, budget_s: float = _CACHE_WALK_BUDGET_S) -> float | None:
    """Sum file sizes under ``path``.

    Returns:
        ``None`` if the directory doesn't exist OR the walk hit the wall-clock
        budget (which is itself useful signal — "cache is too large/slow to
        size", which the caller renders as "unknown").

        Otherwise the total in GB.

    Walks with ``os.walk(..., followlinks=False)`` so LM-Studio-style symlinks
    don't double-count. The deadline is checked **inside** the per-file loop,
    not just between directories: HF cache's ``blobs/`` subdir is flat with
    thousands of entries, so a per-directory deadline would let a single
    cold-cache stat() storm blow past the 1.5 s budget. Codex review round 2
    caught the per-directory variant as a contract violation; this version
    aborts on the very next file once the deadline expires.
    """
    import time as _time

    if not path.exists():
        return None
    deadline = _time.monotonic() + budget_s
    if _DOCTOR_DEADLINE is not None:
        deadline = min(deadline, _DOCTOR_DEADLINE)
    total = 0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for f in files:
                if _time.monotonic() >= deadline:
                    # Budget exhausted mid-directory; partial total isn't
                    # useful (lower bound only). Caller renders "unknown".
                    return None
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    # Broken symlink, permission denied — skip silently;
                    # this probe is "is the cache enormous?", not "audit
                    # every file".
                    continue
            if _time.monotonic() >= deadline:
                return None
    except OSError:
        return None
    return total / (1024**3)


def section_system() -> Section:
    """Hardware + OS section.

    ⚠ on:
      * non-Apple-Silicon (rapid-mlx targets M-series; works elsewhere but
        with Metal-fallback caveats).
      * < 20 GB free disk (model weights are big).
      * HF cache > 100 GB (suggest ``rapid-mlx rm`` cleanup).

    ✗ on:
      * < 5 GB free disk (next download will fail).
    """
    s = Section("System")

    chip, ram_gb = _detect_apple_silicon()
    if chip:
        ram_str = f"{ram_gb} GB" if ram_gb else "unknown RAM"
        s.add(
            f"Apple Silicon ({chip}, {ram_str})",
            CheckStatus.OK,
            detail=f"chip={chip} ram_gb={ram_gb}",
        )
    elif platform.system() == "Darwin":
        s.add(
            "Non-Apple-Silicon Mac — MLX requires arm64",
            CheckStatus.WARN,
            detail=f"machine={platform.machine()}",
        )
    else:
        s.add(
            f"Not macOS ({platform.system()}) — MLX is Apple-only",
            CheckStatus.WARN,
            detail=f"system={platform.system()} machine={platform.machine()}",
        )

    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        s.add(
            f"macOS {mac_ver} (Darwin {platform.release()})",
            CheckStatus.OK,
            detail=f"mac_ver={mac_ver} darwin={platform.release()}",
        )
    else:
        s.add(
            f"OS: {platform.system()} {platform.release()}",
            CheckStatus.OK,
            detail=f"system={platform.system()} release={platform.release()}",
        )

    free_gb = _disk_free_gb(Path.home())
    if free_gb is None:
        s.add(
            "Free disk: unknown",
            CheckStatus.WARN,
            detail="shutil.disk_usage($HOME) failed",
        )
    elif free_gb < 5:
        s.add(
            f"Free disk: {free_gb:.0f} GB (very low — next download may fail)",
            CheckStatus.FAIL,
            detail=f"free_gb={free_gb:.1f}",
        )
    elif free_gb < 20:
        s.add(
            f"Free disk: {free_gb:.0f} GB (low — large models need 20+ GB)",
            CheckStatus.WARN,
            detail=f"free_gb={free_gb:.1f}",
        )
    else:
        s.add(
            f"Free disk: {free_gb:.0f} GB",
            CheckStatus.OK,
            detail=f"free_gb={free_gb:.1f}",
        )

    cache = _hf_cache_dir()
    if not cache.exists():
        s.add(
            f"HF cache: not present ({cache})",
            CheckStatus.OK,
            detail=f"path={cache}",
        )
    else:
        cache_size_gb = _dir_size_gb(cache)
        if cache_size_gb is None:
            # Walk hit the time budget — likely a very large or network-
            # mounted cache. Don't penalise the user; just say so.
            s.add(
                f"HF cache size: too large to size in {_CACHE_WALK_BUDGET_S:.1f}s "
                "(consider `rapid-mlx rm` if unused models accumulated)",
                CheckStatus.WARN,
                detail=f"path={cache} budget_s={_CACHE_WALK_BUDGET_S}",
            )
        elif cache_size_gb > 100:
            s.add(
                f"HF cache size: {cache_size_gb:.0f} GB "
                "(consider `rapid-mlx rm` for unused models)",
                CheckStatus.WARN,
                detail=f"cache_gb={cache_size_gb:.1f} path={cache}",
            )
        else:
            s.add(
                f"HF cache size: {cache_size_gb:.1f} GB",
                CheckStatus.OK,
                detail=f"cache_gb={cache_size_gb:.1f} path={cache}",
            )

    return s


# ---------------------------------------------------------------------------
# Section: Python
# ---------------------------------------------------------------------------


def _install_location(exe: Path | None = None) -> tuple[str, Path]:
    """Classify where ``rapid-mlx`` is installed: ``uv tool``, ``pipx``,
    ``virtualenv``, ``system``. Returned label is for display; the path
    is shown in --verbose."""
    exe = (exe or Path(sys.executable)).resolve()
    parts = exe.parts
    lower = str(exe).lower()
    if "uv/tools" in lower or "/uv/tools/" in lower:
        return "uv tool", exe
    if "pipx" in lower:
        return "pipx", exe
    # site-packages under a venv-style structure
    if (
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        or "VIRTUAL_ENV" in os.environ
    ):
        return "virtualenv", exe
    if "Cellar" in parts or "/homebrew/" in lower:
        return "Homebrew", exe
    return "system", exe


def section_python() -> Section:
    s = Section("Python")

    py_ver = ".".join(str(x) for x in sys.version_info[:3])
    exe = Path(sys.executable).absolute()
    # Defensive: pyproject pins ``requires-python = ">=3.10"`` so install-
    # time pip would already have refused — but doctor should still tell the
    # user clearly if they somehow got rapid-mlx onto an older interpreter
    # (e.g. a hand-copied wheel). Ruff's UP036 flags this as dead under our
    # support matrix; that's the point of the defensive branch.
    if sys.version_info >= (3, 10):  # noqa: UP036
        s.add(
            f"Python {py_ver}",
            CheckStatus.OK,
            detail=f"executable={exe}; prefix={Path(sys.prefix).resolve()}",
        )
    else:  # pragma: no cover — only reachable on unsupported interpreters
        s.add(
            f"Python {py_ver} (rapid-mlx requires >= 3.10)",
            CheckStatus.FAIL,
            detail=f"executable={exe}; prefix={Path(sys.prefix).resolve()}",
        )

    selected_runtime, server_runtime_selected = _selected_runtime()
    server_differs = server_runtime_selected or selected_runtime != exe
    runtime_noun = (
        "Active server runtime"
        if server_runtime_selected
        else "Selected diagnostic runtime"
    )
    runtime_probe = (
        _probe_runtime(selected_runtime, _bundled_sidecar_root(selected_runtime))
        if server_differs
        else None
    )
    runtime = _runtime_environment(exe, Path(sys.prefix))
    detail = (
        f"runtime_type={runtime}; sys.executable={exe}; "
        f"sys.prefix={Path(sys.prefix).resolve()}; "
        f"relevant_sys.path={json.dumps([entry for entry in sys.path if entry])}"
    )
    if server_differs:
        if runtime_probe is not None:
            selected_exe = runtime_probe.get("executable", selected_runtime)
            selected_prefix = runtime_probe.get("prefix", selected_runtime.parent)
            selected_base_prefix = runtime_probe.get("base_prefix", selected_prefix)
            selected_path = runtime_probe.get("path", [])
            selected_kind = _runtime_environment(
                Path(str(selected_exe)),
                Path(str(selected_prefix)),
                Path(str(selected_base_prefix)),
            )
            label = (
                f"{runtime_noun} differs from the doctor CLI; "
                f"package checks use {selected_kind}"
            )
            detail += (
                f"; server_sys.executable={selected_exe}; "
                f"server_sys.prefix={selected_prefix}; "
                f"server_sys.path={json.dumps(selected_path)}"
            )
        else:
            label = (
                f"{runtime_noun} differs from the doctor CLI and could not be inspected"
            )
            detail += (
                f"; server_runtime={selected_runtime}; "
                "override=RAPID_MLX_RUNTIME_PYTHON"
            )
        s.add(label, CheckStatus.WARN, detail=detail)
    else:
        s.add(
            f"Active runtime: {runtime}",
            CheckStatus.OK,
            detail=detail,
        )

    install_label, path = _install_location(exe)
    s.add(
        f"Install location: {install_label} ({path})",
        CheckStatus.OK,
        detail=(
            f"sys.executable={path}; all package checks use this runtime's "
            "sys.path (or the running server's equivalent runtime)"
        ),
    )

    return s


# ---------------------------------------------------------------------------
# Section: packages
# ---------------------------------------------------------------------------


def _safe_version(
    dist: str,
    runtime: Path | None = None,
) -> str | None:
    runtime = runtime or Path(sys.executable).absolute()
    if _runtime_uses_context(runtime):
        probe = _probe_runtime(
            runtime,
            _bundled_sidecar_root(runtime),
        )
        package = _probe_package(probe, dist) if probe else None
        if package is not None:
            version = package.get("version")
            return str(version) if version else None
        return None
    try:
        return _im.version(dist)
    except _im.PackageNotFoundError:
        return None


def _visible_without_metadata(
    dist: str,
    runtime: Path | None = None,
) -> bool:
    """Whether this runtime can import *dist* despite missing dist metadata.

    Layered/relocatable runtimes can expose a package directory on ``sys.path``
    without exposing its sibling ``*.dist-info`` directory. Calling that
    package "not installed" is a false negative: the server launched by this
    interpreter can see it. We report the metadata defect separately instead.
    """
    runtime = runtime or Path(sys.executable).absolute()
    module = _DISTRIBUTION_MODULES.get(dist)
    if module is None:
        return False
    if _runtime_uses_context(runtime):
        return _module_visibility(dist, runtime)[0]
    return _module_available(module, real_import=True)


def _module_visibility(
    dist: str,
    runtime: Path | None = None,
) -> tuple[bool, bool]:
    """Return (module visible, doctor verified a real import).

    Context paths supplied by a running server can make a module discoverable,
    but doctor must not execute code from those paths merely to diagnose it.
    Such a module is reported as visible but unverified rather than missing.
    """
    runtime = runtime or Path(sys.executable).absolute()
    if _runtime_uses_context(runtime):
        probe = _probe_runtime(
            runtime,
            _bundled_sidecar_root(runtime),
        )
        package = _probe_package(probe, dist) if probe else None
        if package is None:
            return False, False
        importable = package.get("importable")
        trusted_origin = bool(
            package.get("trusted_origin", isinstance(importable, bool) and importable)
        )
        if trusted_origin and _runtime_module_importable(
            runtime,
            str(package.get("module", "")),
            _bundled_sidecar_root(runtime),
        ):
            return True, True
        if bool(
            package.get("discoverable", isinstance(importable, bool) and importable)
        ):
            return True, False
        return False, False
    importable = _module_available(_DISTRIBUTION_MODULES[dist], real_import=True)
    return importable, importable


def _version_supported(dist: str, version: str) -> bool:
    spec = _SUPPORTED_VERSIONS.get(dist)
    if spec is None:
        return True
    try:
        return Version(version) in SpecifierSet(spec)
    except (InvalidVersion, InvalidSpecifier):
        return False


def _runtime_pip_command(
    *requirements: str,
    runtime: Path | None = None,
) -> str:
    """Return an unambiguous repair command for the interpreter under test."""
    quoted = " ".join(shlex.quote(requirement) for requirement in requirements)
    selected_runtime = runtime or _runtime_python_path()
    return f"{shlex.quote(str(selected_runtime))} -m pip install --upgrade {quoted}"


def _pil_importable(
    runtime: Path | None = None,
) -> bool:
    """Lightweight probe: does mlx-vlm's ``from PIL import Image`` actually
    work?

    mlx-vlm does ``from PIL import Image`` at module load, so a present
    mlx-vlm with an absent PIL — the Homebrew ``pip install --no-deps
    mlx-vlm`` state (#1126) — is a FALSE positive: metadata says installed,
    but every vision path crashes on import.

    We perform the EXACT lightweight import mlx-vlm uses rather than a
    ``find_spec('PIL')`` probe: ``find_spec`` only proves *something named
    PIL is discoverable*, so a shadowed namespace dir or a damaged Pillow
    whose real ``from PIL import Image`` raises would still get a green row.
    We then run a minimal native-backed op (``Image.new``) so a Pillow whose
    Python layer imports but whose ``_imaging`` C extension is missing or
    ABI-mismatched — present-but-broken, not merely absent — is caught too,
    without relying on the version-specific moment Pillow first touches
    ``_imaging``. ``PIL.Image`` + a 1×1 allocation is microsecond-cheap and
    does NOT pull torch the way a real ``import mlx_vlm`` would, so it stays
    well within doctor's ≤5 s budget. ANY failure (missing, shadowed, broken
    native ext) ⇒ not importable."""
    if runtime is not None and _runtime_uses_context(runtime):
        probe = _probe_runtime(
            runtime,
            _bundled_sidecar_root(runtime),
        )
        if not probe:
            return False
        return _runtime_module_importable(
            runtime,
            "PIL.Image",
            _bundled_sidecar_root(runtime),
            exercise=True,
        )

    try:
        if not _module_origin_is_trusted("PIL.Image"):
            return False
        from PIL import Image

        # Force the native ``_imaging`` backend to actually initialise — a
        # broken/ABI-mismatched C ext otherwise slips through as healthy.
        Image.new("RGB", (1, 1))
    except (Exception, SystemExit):
        # ImportError (absent/shadowed), OSError (broken native ext), or any
        # other load-time failure — all mean the real vision import can't run.
        return False
    return True


def _version_at_least(ver: str, minimum: tuple[int, ...]) -> bool:
    """Compare a PEP 440 version string against a ``(major, minor, patch)``
    floor using leading-numeric tuple semantics.

    Pre-release/local-version suffixes are stripped — ``"0.5.0rc1"`` and
    ``"0.5.0+local"`` both compare equal to ``(0, 5, 0)``. We don't need
    full PEP 440 ordering here; the only floor we care about is the DFlash
    bump (``mlx-vlm >= 0.5.0``). A malformed version returns ``False`` —
    safer to nudge the user to upgrade than to silently pass.
    """
    parts: list[int] = []
    for raw in ver.split("."):
        # Stop at the first non-numeric chunk (e.g. "0rc1", "5+local").
        digits = ""
        for ch in raw:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        return False
    # Pad to the shape of ``minimum`` so the tuple comparison is well-defined.
    while len(parts) < len(minimum):
        parts.append(0)
    return tuple(parts[: len(minimum)]) >= minimum


def section_required_packages() -> Section:
    s = Section("Required Packages")
    runtime = _selected_runtime()[0]
    sidecar_root = _bundled_sidecar_root(runtime)
    sidecar_hint = _sidecar_repair_hint(sidecar_root) if sidecar_root else None
    runtime_probe = (
        _probe_runtime(
            runtime,
            sidecar_root,
        )
        if _runtime_uses_context(runtime)
        else None
    )
    if _runtime_uses_context(runtime) and runtime_probe is None:
        s.add(
            "Could not inspect the active server runtime",
            CheckStatus.FAIL,
            detail=(
                f"runtime={runtime}; set RAPID_MLX_RUNTIME_PYTHON to its "
                "Python executable, ensure that interpreter is readable and "
                "runnable, then run doctor again"
            ),
        )
        return s
    for dist, label in REQUIRED_PACKAGES:
        ver = (
            _safe_version(dist, runtime)
            if _runtime_uses_context(runtime)
            else _safe_version(dist)
        )
        if ver and not _version_supported(dist, ver):
            supported = _SUPPORTED_VERSIONS[dist]
            if sidecar_hint:
                repair = sidecar_hint
            elif dist == "transformers":
                repair = _runtime_pip_command(
                    "rapid-mlx",
                    f"transformers{supported}",
                    runtime=runtime,
                )
            else:
                repair = _runtime_pip_command(
                    "rapid-mlx",
                    f"{dist}{supported}",
                    runtime=runtime,
                )
            s.add(
                f"{label} {ver} is incompatible (requires {supported}) — "
                f"run `{repair}`",
                CheckStatus.FAIL,
                detail=(
                    f"distribution={dist} version={ver} supported={supported} "
                    f"runtime={runtime}"
                ),
            )
        elif ver:
            repair = sidecar_hint or _runtime_pip_command("rapid-mlx", runtime=runtime)
            module = _DISTRIBUTION_MODULES[dist]
            if _import_probe_was_interrupted(
                runtime,
                module,
                sidecar_root,
            ):
                s.add(
                    f"{label} {ver} importability unknown — doctor probe timed out",
                    CheckStatus.WARN,
                    detail=f"distribution={dist} module={module} timeout=true",
                )
                continue
            visible, import_verified = _module_visibility(
                dist,
                runtime if _runtime_uses_context(runtime) else None,
            )
            if _import_probe_was_interrupted(runtime, module, sidecar_root):
                s.add(
                    f"{label} {ver} importability unknown — doctor probe timed out",
                    CheckStatus.WARN,
                    detail=f"distribution={dist} module={module} timeout=true",
                )
                continue
            if not visible:
                s.add(
                    f"{label} {ver} has broken metadata or cannot import in "
                    f"{runtime} — run `{repair}`",
                    CheckStatus.FAIL,
                    detail=(
                        f"distribution={dist} version={ver} "
                        f"module={_DISTRIBUTION_MODULES[dist]} "
                        f"importable=False runtime={runtime}"
                    ),
                )
                continue
            if not import_verified:
                s.add(
                    f"{label} {ver} is visible but importability cannot be "
                    "verified safely",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} version={ver} "
                        f"module={_DISTRIBUTION_MODULES[dist]} "
                        f"runtime={runtime}; module is in server context, "
                        "but doctor does not execute server context paths"
                    ),
                )
                continue
            s.add(
                f"{label} {ver}",
                CheckStatus.OK,
                detail=f"distribution={dist} version={ver}",
            )
        else:
            module = _DISTRIBUTION_MODULES[dist]
            if _import_probe_was_interrupted(
                runtime,
                module,
                sidecar_root,
            ):
                s.add(
                    f"{label} importability unknown — doctor probe timed out",
                    CheckStatus.WARN,
                    detail=f"distribution={dist} module={module} timeout=true",
                )
                continue
            visible, import_verified = _module_visibility(
                dist,
                runtime if _runtime_uses_context(runtime) else None,
            )
            if _import_probe_was_interrupted(runtime, module, sidecar_root):
                s.add(
                    f"{label} importability unknown — doctor probe timed out",
                    CheckStatus.WARN,
                    detail=f"distribution={dist} module={module} timeout=true",
                )
                continue
            if not visible:
                repair = sidecar_hint or _runtime_pip_command(
                    "rapid-mlx", runtime=runtime
                )
                s.add(
                    f"{label} not installed in {runtime} — run `{repair}`",
                    CheckStatus.FAIL,
                    detail=(f"distribution={dist} missing runtime={runtime}"),
                )
                continue
            if not import_verified:
                s.add(
                    f"{label} is visible but importability cannot be verified safely",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} "
                        f"module={_DISTRIBUTION_MODULES[dist]} "
                        f"runtime={runtime}; module is in server context, "
                        "but doctor does not execute server context paths"
                    ),
                )
                continue
            s.add(
                f"{label} is importable but version metadata is unavailable",
                CheckStatus.WARN,
                detail=(
                    f"distribution={dist} module={_DISTRIBUTION_MODULES[dist]} "
                    f"runtime={runtime}; reinstall this "
                    "runtime to restore compatibility checks"
                ),
            )
    return s


def section_updates(
    *,
    installed: Callable[[], str | None] | None = None,
    fetch_latest: Callable[[], str | None] | None = None,
    install_info: object | None = None,
) -> Section:
    """Is the installed rapid-mlx the latest release?

    Best-effort and network-bounded — reuses ``_version_check``'s 2 s timeout and
    cache-first fetch, and warms the same cache the CLI's staleness banner reads.
    Never fatal: an unknown installed version or an unreachable endpoint
    downgrades to ⚠ (like the HF network probe), so ``doctor`` never exits
    non-zero just because the machine is offline.
    """
    s = Section("Updates")
    from vllm_mlx import _version_check as vc

    cur = (installed or vc._installed_version)()
    if not cur:
        s.add(
            "installed rapid-mlx version unknown",
            CheckStatus.WARN,
            detail="_installed_version() returned None",
        )
        return s

    latest = (fetch_latest or vc.get_latest_version)()
    if latest is None:
        s.add(
            f"rapid-mlx {cur} — couldn't reach the update endpoint (offline?)",
            CheckStatus.WARN,
            detail="version endpoint unreachable; freshness check skipped",
        )
        return s

    pc, pl = vc._parse_version(cur), vc._parse_version(latest)
    if pc is None or pl is None:
        # One side is an unsupported alpha/beta/local/git-describe build
        # (e.g. ``0.11.0a1``, ``0.10.15+local``, ``0.10``). We can't order
        # it, so DON'T fall through to a green "up to date" — that would
        # falsely reassure a user who might well be behind. Downgrade to ⚠
        # like every other uncertain branch in this section.
        s.add(
            f"rapid-mlx {cur} — can't compare against latest {latest} "
            "(unrecognized version format); freshness check skipped",
            CheckStatus.WARN,
            detail=(
                f"installed={cur} latest={latest} "
                f"parsed_installed={pc} parsed_latest={pl}"
            ),
        )
    elif pl > pc:
        info = install_info if install_info is not None else vc.detect_install_method()
        cmd = getattr(info, "upgrade_command", None) or "rapid-mlx upgrade"
        s.add(
            f"update available: {latest} (installed {cur}) — run `{cmd}`",
            CheckStatus.WARN,
            detail=(
                f"installed={cur} latest={latest} method={getattr(info, 'method', '?')}"
            ),
        )
    else:
        s.add(
            f"rapid-mlx {cur} is up to date",
            CheckStatus.OK,
            detail=f"installed={cur} latest={latest}",
        )
    return s


def section_optional_packages() -> Section:
    s = Section("Optional Packages")
    runtime = _selected_runtime()[0]
    # RC 0.12.18: the signed desktop bundle installs the bounded
    # ``[audio-desktop]`` extra, but doctor graded it against the full
    # ``[audio]`` contract and reported a healthy build as "incomplete", then
    # told the user to pip-install into a code-signed app. Detect the managed
    # sidecar once and swap the contract, the remediation hint, and the
    # treatment of extras the bundle intentionally omits.
    sidecar_root = _bundled_sidecar_root(runtime)
    bundled = sidecar_root is not None
    audio_contract = _AUDIO_DESKTOP_IMPORTS if bundled else _AUDIO_IMPORTS
    repair_hint = _sidecar_repair_hint(sidecar_root) if sidecar_root else None
    runtime_probe = (
        _probe_runtime(
            runtime,
            sidecar_root,
        )
        if _runtime_uses_context(runtime)
        else None
    )
    if _runtime_uses_context(runtime) and runtime_probe is None:
        s.add(
            "Could not inspect the active server runtime",
            CheckStatus.FAIL,
            detail=(
                f"runtime={runtime}; set RAPID_MLX_RUNTIME_PYTHON to its "
                "Python executable, ensure that interpreter is readable and "
                "runnable, then run doctor again"
            ),
        )
        return s
    for dist, label, install_hint in OPTIONAL_PACKAGES:
        # ``pip`` on PATH may belong to ~/.rapid-mlx-python or Homebrew while
        # the server runs from ~/.rapid-mlx. Bind every CLI remediation to the
        # interpreter whose package visibility doctor just inspected.
        hint = (
            repair_hint
            if repair_hint
            else _runtime_pip_command(install_hint, runtime=runtime)
        )
        ver = (
            _safe_version(dist, runtime)
            if _runtime_uses_context(runtime)
            else _safe_version(dist)
        )
        if bundled and not ver and dist in _DESKTOP_EXCLUDED_DISTS:
            # Not a defect: this extra is outside the desktop product surface,
            # so there is nothing for the user to repair. ⚠ + "reinstall" here
            # would survive any reinstall and train users to ignore the
            # section.
            s.add(
                f"{label} not bundled with Rapid-MLX Desktop (not required)",
                CheckStatus.OK,
                detail=f"distribution={dist} bundled=false reason=excluded-from-desktop",
            )
            continue
        if ver and not _version_supported(dist, ver):
            supported = _SUPPORTED_VERSIONS[dist]
            if repair_hint:
                repair = repair_hint
            else:
                repair = _runtime_pip_command(
                    "rapid-mlx[vision]",
                    f"transformers{_SUPPORTED_VERSIONS['transformers']}",
                    runtime=runtime,
                )
            s.add(
                f"{label} {ver} is incompatible (requires {supported}) — {repair}",
                CheckStatus.FAIL,
                detail=(
                    f"distribution={dist} version={ver} supported={supported} "
                    f"runtime={runtime}"
                ),
            )
            continue
        if ver:
            # #1255: mlx-vlm can install mlx-audio transitively without
            # respecting Rapid-MLX's supported audio range. Presence alone
            # must not make doctor report that environment as healthy.
            if dist == "mlx-audio" and not (
                _version_at_least(ver, (0, 2, 9))
                and not _version_at_least(ver, (0, 4, 4))
            ):
                s.add(
                    f"{label} {ver} unsupported — rapid-mlx requires "
                    f"mlx-audio>=0.2.9,<0.4.4 (`{hint}`)",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} version={ver} "
                        f"supported=>=0.2.9,<0.4.4 hint={hint}"
                    ),
                )
                continue
            if dist == "mlx-audio":
                missing = [
                    distribution
                    for distribution, module in audio_contract
                    if not _module_available(
                        module,
                        runtime if _runtime_uses_context(runtime) else None,
                    )
                ]
                if missing:
                    missing_text = ", ".join(missing)
                    s.add(
                        f"{label} {ver} incomplete — missing: {missing_text} "
                        f"(`{hint}`)",
                        CheckStatus.WARN,
                        detail=(
                            f"distribution={dist} version={ver} "
                            f"missing={missing_text} hint={hint} "
                            f"contract={'audio-desktop' if bundled else 'audio'}"
                        ),
                    )
                    continue
            # #1126: mlx-vlm imports Pillow (PIL) at load. A present
            # mlx-vlm with an absent PIL (Homebrew `pip install --no-deps
            # mlx-vlm`) is a FALSE positive — metadata says "installed" but
            # every vision path crashes on `from PIL import Image` deep in
            # the FastAPI lifespan. Report it honestly and name the real
            # gap so the user fixes the right thing.
            if dist == "mlx-vlm" and not _pil_importable(
                runtime if _runtime_uses_context(runtime) else None,
            ):
                s.add(
                    f"{label} {ver} present but Pillow (PIL) missing or "
                    f"broken — vision paths will fail (`{hint}`)",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} version={ver} "
                        f"pil=missing-or-broken hint={hint}"
                    ),
                )
                continue
            visible, import_verified = _module_visibility(
                dist,
                runtime if _runtime_uses_context(runtime) else None,
            )
            if not visible or not import_verified:
                s.add(
                    f"{label} {ver} present but importability is broken or unverified",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} version={ver} "
                        f"module={_DISTRIBUTION_MODULES[dist]} "
                        f"visible={visible} verified={import_verified} "
                        f"runtime={runtime}"
                    ),
                )
                continue
            s.add(
                f"{label} {ver}",
                CheckStatus.OK,
                detail=f"distribution={dist} version={ver}",
            )
        else:
            module = _DISTRIBUTION_MODULES[dist]
            if _import_probe_was_interrupted(
                runtime,
                module,
                sidecar_root,
            ):
                s.add(
                    f"{label} importability unknown — doctor probe timed out",
                    CheckStatus.WARN,
                    detail=f"distribution={dist} module={module} timeout=true",
                )
                continue
            visible, import_verified = _module_visibility(
                dist,
                runtime if _runtime_uses_context(runtime) else None,
            )
            if _import_probe_was_interrupted(runtime, module, sidecar_root):
                s.add(
                    f"{label} importability unknown — doctor probe timed out",
                    CheckStatus.WARN,
                    detail=f"distribution={dist} module={module} timeout=true",
                )
                continue
            if not visible:
                s.add(
                    f"{label} not installed (`{hint}`)",
                    CheckStatus.WARN,
                    detail=f"distribution={dist} hint={hint}",
                )
            elif import_verified:
                s.add(
                    f"{label} is importable in {runtime} but version metadata "
                    "is unavailable",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} "
                        f"module={_DISTRIBUTION_MODULES[dist]} "
                        f"visible=true verified=true runtime={runtime}; "
                        "package is not missing, but compatibility cannot be "
                        "verified"
                    ),
                )
            else:
                s.add(
                    f"{label} is visible in {runtime} but importability is "
                    "not verified and version metadata is unavailable",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution={dist} "
                        f"module={_DISTRIBUTION_MODULES[dist]} "
                        f"visible=true verified=false runtime={runtime}; "
                        "package is not missing, but health cannot be verified"
                    ),
                )
    # DFlash is the headline 0.9.x feature and is gated on mlx-vlm >= 0.5.0.
    # The plain optional-package row above only reports presence/version —
    # it cannot say "you have mlx-vlm but it's too old for [dflash]". This
    # extra row makes the gate explicit so a fresh-install user knows whether
    # `pip install 'rapid-mlx[dflash]'` will actually work.
    #
    # The desktop bundle DOES ship mlx-vlm (pinned 0.6.3, ``--no-deps`` + Pillow
    # — the gemma-4 family needs it even in text-only mode), so unlike
    # mlx-embeddings this row is a real contract on a bundled sidecar and stays
    # gradeable; only the remediation wording changes.
    dflash_min = (0, 5, 0)
    dflash_hint = repair_hint or _runtime_pip_command(
        "rapid-mlx[dflash]", runtime=runtime
    )
    vision_hint = repair_hint or _runtime_pip_command(
        "rapid-mlx[vision]", runtime=runtime
    )
    vlm_ver = (
        _safe_version("mlx-vlm", runtime)
        if _runtime_uses_context(runtime)
        else _safe_version("mlx-vlm")
    )
    if (
        vlm_ver
        and _version_supported("mlx-vlm", vlm_ver)
        and _version_at_least(vlm_ver, dflash_min)
    ):
        # #1126: same PIL honesty as the vision row — a version-adequate
        # mlx-vlm whose Pillow dep is missing can't actually run the
        # dflash/vision runtime, so don't paint it green.
        if not _pil_importable(
            runtime if _runtime_uses_context(runtime) else None,
        ):
            s.add(
                "mlx-vlm 0.5.0+ (dflash extras) present but Pillow (PIL) "
                "missing or broken — dflash/vision paths will fail",
                CheckStatus.WARN,
                detail=(
                    f"distribution=mlx-vlm version={vlm_ver} "
                    f"pil=missing-or-broken hint={vision_hint}"
                ),
            )
        else:
            _, vlm_verified = _module_visibility(
                "mlx-vlm",
                runtime if _runtime_uses_context(runtime) else None,
            )
            if not vlm_verified:
                s.add(
                    "mlx-vlm 0.5.0+ (dflash extras) present but importability "
                    "is broken or unverified",
                    CheckStatus.WARN,
                    detail=(
                        f"distribution=mlx-vlm version={vlm_ver} "
                        f"verified=False runtime={runtime}"
                    ),
                )
            else:
                s.add(
                    "mlx-vlm 0.5.0+ (dflash extras)",
                    CheckStatus.OK,
                    detail=f"distribution=mlx-vlm version={vlm_ver}",
                )
    else:
        current = vlm_ver or "not installed"
        s.add(
            f"mlx-vlm 0.5.0+ (dflash extras) not installed, too old, or "
            f"incompatible (current: {current}, need: 0.5.0+ and "
            f"{_SUPPORTED_VERSIONS['mlx-vlm']})",
            CheckStatus.WARN,
            detail=dflash_hint,
        )

    return s


# ---------------------------------------------------------------------------
# Section: HuggingFace cache
# ---------------------------------------------------------------------------


def _nearest_existing_parent(p: Path) -> Path | None:
    """Walk up ``p``'s ancestors until we find one that exists, or return
    ``None`` if even the filesystem root has somehow disappeared."""
    for ancestor in (p, *p.parents):
        if ancestor.exists():
            return ancestor
    return None


def section_hf_cache() -> Section:
    s = Section("HuggingFace Cache")

    cache = _hf_cache_dir()
    if cache.exists():
        # Codex review round 2: ``os.access`` returns True for a writable
        # regular file too, so a user who set ``HF_HUB_CACHE`` to a path
        # that's now a file (typo, mv accident, …) would see ✓ here and
        # then fail on the first download with a confusing error.
        if not cache.is_dir():
            s.add(
                f"{cache} exists but is NOT a directory",
                CheckStatus.FAIL,
                detail=f"path={cache} type=non-directory",
            )
        elif os.access(cache, os.W_OK):
            s.add(
                f"{cache} exists, writable",
                CheckStatus.OK,
                detail=f"path={cache}",
            )
        else:
            s.add(
                f"{cache} exists but NOT writable",
                CheckStatus.FAIL,
                detail=f"path={cache}",
            )
    else:
        # Missing cache isn't *always* a soft warning — if the nearest
        # existing parent isn't writable either, the first download will
        # fail trying to create ``cache``. Codex review round 2 caught
        # the previous unconditional WARN as silently green for
        # ``HF_HOME=/readonly/hf``.
        parent = _nearest_existing_parent(cache.parent)
        if parent is None or not os.access(parent, os.W_OK):
            s.add(
                f"{cache} does not exist and parent {parent} is NOT writable "
                "— next download will fail",
                CheckStatus.FAIL,
                detail=f"path={cache} parent={parent}",
            )
        else:
            s.add(
                f"{cache} does not exist yet (will be created on first download)",
                CheckStatus.WARN,
                detail=f"path={cache} parent={parent}",
            )

    # Disk free for the partition the cache lives on (or would live on).
    probe_dir = cache if cache.exists() else cache.parent
    if not probe_dir.exists():
        probe_dir = Path.home()
    free_gb = _disk_free_gb(probe_dir)
    if free_gb is None:
        s.add(
            "Free space on cache partition: unknown",
            CheckStatus.WARN,
            detail=f"probe_dir={probe_dir}",
        )
    elif free_gb < 5:
        s.add(
            f"Free space: {free_gb:.0f} GB (very low — model downloads will fail)",
            CheckStatus.FAIL,
            detail=f"free_gb={free_gb:.1f} probe_dir={probe_dir}",
        )
    else:
        s.add(
            f"Free space: {free_gb:.0f} GB",
            CheckStatus.OK,
            detail=f"free_gb={free_gb:.1f} probe_dir={probe_dir}",
        )

    return s


# ---------------------------------------------------------------------------
# Section: Network
# ---------------------------------------------------------------------------


# Single, time-boxed network probe. The whole point is to catch "user is
# behind a proxy / offline / DNS broken" early — not to audit reachability
# of every endpoint we ever talk to. A 2 s budget keeps the worst-case
# doctor runtime under the 5 s contract even when the resolver hangs.
_HF_PROBE_URL = "https://huggingface.co"
_HF_PROBE_TIMEOUT_S = 2.0
_HF_PROBE_SCRIPT = r"""
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
timeout = float(sys.argv[2])
request = urllib.request.Request(url, method="HEAD")
try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        print(json.dumps({"code": response.status}))
except urllib.error.HTTPError as exc:
    print(json.dumps({"code": exc.code}))
except (urllib.error.URLError, TimeoutError) as exc:
    print(json.dumps({"error": type(exc).__name__}))
"""


def _probe_hf(
    timeout: float = _HF_PROBE_TIMEOUT_S,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[CheckStatus, str]:
    """Return HEAD reachability while bounding DNS resolution in a child."""
    parent_timeout = _bounded_timeout(timeout)
    child_timeout = max(0.001, parent_timeout - min(0.1, parent_timeout / 2))
    try:
        result = run(  # noqa: S603 — fixed interpreter and script
            [
                sys.executable,
                "-I",
                "-c",
                _HF_PROBE_SCRIPT,
                _HF_PROBE_URL,
                str(child_timeout),
            ],
            capture_output=True,
            text=True,
            timeout=parent_timeout,
            check=True,
        )
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("network probe returned a non-object")
        code = payload.get("code")
        if code in (200, 301, 302, 405):
            return CheckStatus.OK, f"HEAD {_HF_PROBE_URL} → HTTP {code}"
        if isinstance(code, int):
            return CheckStatus.WARN, f"HTTP {code} (rate-limited?)"
        error = payload.get("error")
        return CheckStatus.WARN, f"unreachable ({error or 'network error'})"
    except subprocess.TimeoutExpired:
        return CheckStatus.WARN, "unreachable (network probe timed out)"
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        return CheckStatus.WARN, "unreachable (network probe failed)"


def section_network(
    *, probe: Callable[[], tuple[CheckStatus, str]] | None = None
) -> Section:
    """Network reachability probe.

    ``probe`` is injected by tests to avoid hitting the real internet.
    """
    s = Section("Network")
    fn = probe or _probe_hf
    status, detail = fn()
    if status is CheckStatus.OK:
        s.add("huggingface.co reachable", CheckStatus.OK, detail=detail)
    else:
        s.add(
            f"huggingface.co not reachable ({detail})",
            CheckStatus.WARN,
            detail=detail,
        )

    return s


# ---------------------------------------------------------------------------
# Section: Shell integration
# ---------------------------------------------------------------------------


_ARGCOMPLETE_HOOK_NEEDLE = "register-python-argcomplete rapid-mlx"

# Bound per-rc read so a 50 MB hand-edited zshrc, a named pipe, or a
# block device pointed-to via symlink can't make doctor hang or eat RAM.
# 256 KB is roughly 4000 lines of shell config, which is far above any
# real-world rc file's footprint. Codex review round 2 caught the
# previous unbounded ``read_text`` as a DoS / hang vector.
_RC_READ_LIMIT_BYTES = 256 * 1024


def _candidate_shell_rcs() -> list[Path]:
    """Return the rc files we look at for argcomplete activation."""
    home = Path.home()
    return [
        home / ".zshrc",
        home / ".bashrc",
        home / ".bash_profile",
        home / ".profile",
    ]


def _read_rc_prefix(rc: Path, limit: int = _RC_READ_LIMIT_BYTES) -> str | None:
    """Read up to ``limit`` bytes from ``rc``. Skips non-regular files
    (pipes / devices / symlinks-to-non-files) and decoding errors."""
    try:
        # ``stat`` follows symlinks, which is the right behavior for shell
        # rc files (people symlink their dotfiles all the time) — but we
        # refuse non-regular targets (S_IFREG missing).
        st = rc.stat()
    except OSError:
        return None
    import stat as _stat

    if not _stat.S_ISREG(st.st_mode):
        return None
    try:
        with rc.open("rb") as f:
            return f.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return None


def _argcomplete_hook_present(
    rcs: list[Path] | None = None,
) -> tuple[bool, Path | None]:
    """Return (present, rc_file_with_hook). ``rcs`` is injected by tests."""
    rcs = rcs if rcs is not None else _candidate_shell_rcs()
    for rc in rcs:
        content = _read_rc_prefix(rc)
        if content and _ARGCOMPLETE_HOOK_NEEDLE in content:
            return True, rc
    return False, None


def _rapid_mlx_on_path(path_env: str | None = None) -> list[str]:
    """Every ``rapid-mlx`` executable on PATH, in PATH order, deduped by resolved
    target. More than one distinct target means competing installs — e.g. a
    ``curl | bash`` copy in ``~/.local/bin`` shadowing a Homebrew one — where the
    first on PATH silently wins and upgrades to the others do nothing.
    """
    raw = os.environ.get("PATH", "") if path_env is None else path_env
    seen: set[str] = set()
    out: list[str] = []
    for d in raw.split(os.pathsep):
        # An empty PATH component means the current directory on POSIX, which
        # is exactly how ``shutil.which()`` (the function that picks the
        # *active* rapid-mlx above) resolves it. Skipping it would let a
        # ``./rapid-mlx`` shadow slip past this very check — the kind of
        # competing install it exists to surface — so map "" → os.curdir
        # instead of dropping it.
        directory = d or os.curdir
        cand = os.path.join(directory, "rapid-mlx")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            target = os.path.realpath(cand)
            if target not in seen:
                seen.add(target)
                out.append(cand)
    return out


def _running_cli_exe() -> str | None:
    """Resolve the ``rapid-mlx`` executable that launched this process.

    When invoked as a console script, ``sys.argv[0]`` is the path to the
    ``rapid-mlx`` entry point (``<venv>/bin/rapid-mlx``). Fall back to the
    interpreter's sibling of the same name (a console script installed in the
    same bin dir as ``sys.executable``) so ``doctor`` still reports
    something sensible under unusual launchers (e.g. ``python -m …``).
    Returns ``None`` only if neither yields a plausible path (not expected).
    """
    argv0 = os.path.realpath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if argv0 and os.path.basename(argv0) == "rapid-mlx":
        return argv0
    sibling = os.path.join(os.path.dirname(sys.executable), "rapid-mlx")
    if os.path.basename(sys.executable) == "rapid-mlx":
        return os.path.realpath(sys.executable)
    return sibling if os.path.exists(sibling) else None


def section_shell_integration(
    *,
    which: Callable[[str], str | None] | None = None,
    rcs: list[Path] | None = None,
    find_all: Callable[[], list[str]] | None = None,
    running_exe: str | None = None,
) -> Section:
    """Verify the CLI is on PATH and argcomplete is wired up.

    ``which``, ``rcs``, ``find_all`` and ``running_exe`` are dependency-injected
    for tests. ``running_exe`` is the executable that actually launched this
    doctor; it defaults to the real running CLI so a venv/bin that precedes a
    global install is surfaced (issue #2352).
    """
    s = Section("Shell Integration")
    which_fn = which or shutil.which

    if running_exe is None:
        running_exe = _running_cli_exe()

    cli_path = which_fn("rapid-mlx")
    if cli_path:
        s.add(
            f"rapid-mlx in $PATH ({cli_path})",
            CheckStatus.OK,
            detail=f"path={cli_path}",
        )
        # Issue #2352: the PATH-checked executable can be a different install
        # from the one actually running this doctor (e.g. inside a venv whose
        # bin/ precedes a global ~/.local/bin install). Surface the divergence
        # with both paths and an actionable fix instead of a silently-green
        # "PATH OK" that points troubleshooting at the wrong install. Compare
        # with realpath + normcase so a console-script symlink (Homebrew/PyPI)
        # that resolves to the same binary is NOT a false mismatch — the
        # running-CLI detector already realpaths sys.argv[0].
        if running_exe and (
            os.path.normcase(os.path.realpath(running_exe))
            != os.path.normcase(os.path.realpath(cli_path))
        ):
            s.add(
                f"running rapid-mlx ({running_exe}) differs from the $PATH "
                f"rapid-mlx ({cli_path}) — activate this environment or reorder "
                f"$PATH",
                CheckStatus.WARN,
                detail=f"running={running_exe} path={cli_path}",
            )
        installs = (find_all or _rapid_mlx_on_path)()
        if len(installs) > 1:
            active, shadowed = installs[0], installs[1:]
            s.add(
                f"rapid-mlx installed in {len(installs)} places — {active} wins on "
                f"$PATH, {', '.join(shadowed)} shadowed; remove the extra install(s)",
                CheckStatus.WARN,
                detail=f"active={active} shadowed={shadowed}",
            )
    else:
        s.add(
            "rapid-mlx NOT on $PATH",
            CheckStatus.FAIL,
            detail="shutil.which('rapid-mlx') returned None",
        )

    present, rc = _argcomplete_hook_present(rcs=rcs)
    if present:
        s.add(
            f"argcomplete activated in {rc.name if rc else 'shell rc'}",
            CheckStatus.OK,
            detail=f"hook found in {rc}",
        )
    else:
        s.add(
            "argcomplete not activated — add "
            '`eval "$(register-python-argcomplete rapid-mlx)"` to your shell rc',
            CheckStatus.WARN,
            detail="no shell rc contains the activation snippet",
        )

    return s


# ---------------------------------------------------------------------------
# Section: Optional Tools
# ---------------------------------------------------------------------------


def section_optional_tools(
    *, which: Callable[[str], str | None] | None = None
) -> Section:
    """Probe for development tools that improve the rapid-mlx experience but
    are never required to run inference. Missing → ✗ (issue) because the user
    explicitly opted into a workflow that needs them — phrasing makes it
    clear they're only relevant if you're using those harnesses."""
    s = Section("Optional Tools")
    which_fn = which or shutil.which

    codex = which_fn("codex")
    if codex:
        s.add(
            f"codex CLI ({codex})",
            CheckStatus.OK,
            detail="@openai/codex on PATH",
        )
    else:
        s.add(
            "codex CLI not installed (relevant if using codex agent harness)",
            CheckStatus.WARN,
            detail="npm install -g @openai/codex",
        )

    return s


# ---------------------------------------------------------------------------
# Section: Agent Integrations
# ---------------------------------------------------------------------------


def _agent_integrations(home: Path) -> list[tuple[str, Path, str | None]]:
    """Read the local endpoint selected by each supported agent client."""
    configs = [
        ("Claude Code", home / ".claude/settings.json"),
        ("Continue.dev", home / ".continue/config.json"),
    ]
    cline_roots = (
        home / "Library/Application Support/Code/User/globalStorage",
        home / "Library/Application Support/Code - Insiders/User/globalStorage",
        home / "Library/Application Support/VSCodium/User/globalStorage",
        home / ".config/Code/User/globalStorage",
        home / ".config/Code - Insiders/User/globalStorage",
        home / ".config/VSCodium/User/globalStorage",
    )
    configs.extend(
        ("Cline", root / "saoudrizwan.claude-dev/settings/cline_mcp_settings.json")
        for root in cline_roots
    )

    integrations: list[tuple[str, Path, str | None]] = []
    for name, path in configs:
        if not path.is_file():
            continue
        url = None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                integrations.append((name, path, None))
                continue
            if name == "Claude Code" and isinstance(data.get("env"), dict):
                url = data["env"].get("ANTHROPIC_BASE_URL")
            elif name == "Continue.dev" and isinstance(data.get("models"), list):
                url = next(
                    (
                        model.get("apiBase")
                        for model in data["models"]
                        if isinstance(model, dict)
                        and model.get("title") == "rapid-mlx"
                        and model.get("provider") == "openai"
                    ),
                    None,
                )
            elif name == "Cline" and data.get("apiProvider") == "openai":
                url = data.get("openAiBaseUrl")
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        integrations.append((name, path, url if isinstance(url, str) else None))
    return integrations


_TCP_PROBE_SCRIPT = r"""
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])
try:
    connection = socket.create_connection((host, port), timeout=timeout)
    connection.close()
    print("1")
except OSError:
    print("0")
"""


def _server_reachable(
    url: str,
    *,
    connect: Callable[..., object] | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    """Perform a deadline-bounded TCP check without interpreting engine APIs."""
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        timeout = _bounded_timeout(0.25)
        if connect is not None:
            connection = connect((parsed.hostname, port), timeout=timeout)
            close = getattr(connection, "close", None)
            if close:
                close()
            return True
        child_timeout = max(0.001, timeout - min(0.05, timeout / 2))
        result = run(  # noqa: S603 — fixed interpreter and script
            [
                sys.executable,
                "-I",
                "-c",
                _TCP_PROBE_SCRIPT,
                parsed.hostname,
                str(port),
                str(child_timeout),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "1"
    except (OSError, TypeError, ValueError, subprocess.TimeoutExpired):
        return False


def section_agent_integrations(
    *,
    home: Path | None = None,
    connect: Callable[..., object] | None = None,
) -> Section:
    """Report whether installed agent configs point to a reachable server."""
    section = Section("Agent Integrations")
    integrations = _agent_integrations(home or Path.home())
    if not integrations:
        section.add(
            "No Claude Code, Cline, or Continue.dev config found", CheckStatus.OK
        )
        return section

    for name, path, url in integrations:
        if not url:
            section.add(
                f"{name} config found, but Rapid-MLX is not configured",
                CheckStatus.WARN,
                detail=f"path={path}",
            )
        elif _server_reachable(url, connect=connect):
            section.add(
                f"{name} server is reachable",
                CheckStatus.OK,
                detail=f"path={path}",
            )
        else:
            section.add(
                f"{name} server is not reachable",
                CheckStatus.WARN,
                detail=f"path={path}",
            )
    return section


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Sections fixed in spec order. Adding a new probe means appending to one
# of these lists, not adding a new section midway — keeps the user's mental
# model stable across rapid-mlx versions.
_SECTION_BUILDERS = (
    section_system,
    section_python,
    section_required_packages,
    section_updates,
    section_optional_packages,
    section_hf_cache,
    section_network,
    section_shell_integration,
    section_optional_tools,
    section_agent_integrations,
)

_SECTION_TITLES = {
    section_system: "System",
    section_python: "Python",
    section_required_packages: "Required Packages",
    section_updates: "Updates",
    section_optional_packages: "Optional Packages",
    section_hf_cache: "HuggingFace Cache",
    section_network: "Network",
    section_shell_integration: "Shell Integration",
    section_optional_tools: "Optional Tools",
    section_agent_integrations: "Agent Integrations",
}


def _budget_exhausted_report() -> Report:
    report = Report()
    for builder in _SECTION_BUILDERS:
        skipped = Section(
            _SECTION_TITLES.get(
                builder, builder.__name__.replace("section_", "").title()
            )
        )
        skipped.add(
            "Skipped: doctor time budget exhausted",
            CheckStatus.WARN,
            detail="probe did not start before the shared deadline",
        )
        report.sections.append(skipped)
    return report


def _run_all_serialized(caller_deadline: float) -> Report:
    """Run every section and return the aggregate report.

    Each section builder is wrapped in a try/except so a single buggy probe
    cannot abort the whole report. If a section crashes, it lands in the
    report as a single ✗ row labelled with the exception class — that's
    still a useful signal ("doctor is broken, file a bug").
    """
    global _DOCTOR_DEADLINE, _RUNTIME_SELECTION_DONE
    report = Report()
    try:
        _DOCTOR_DEADLINE = caller_deadline
        _RUNTIME_SELECTION_DONE = False
        _RUNTIME_PROBE_CACHE.clear()
        _RUNTIME_IMPORT_CACHE.clear()
        _RUNTIME_IMPORT_TIMEOUTS.clear()
        _RUNTIME_DISTRIBUTION_CACHE.clear()
        _RUNTIME_CONTEXTS.clear()
        if time.monotonic() >= _DOCTOR_DEADLINE:
            return _budget_exhausted_report()
        _selected_runtime()
        for index, builder in enumerate(_SECTION_BUILDERS):
            if time.monotonic() >= _DOCTOR_DEADLINE:
                for skipped_builder in _SECTION_BUILDERS[index:]:
                    skipped = Section(
                        _SECTION_TITLES.get(
                            skipped_builder,
                            skipped_builder.__name__.replace("section_", "").title(),
                        )
                    )
                    skipped.add(
                        "Skipped: doctor time budget exhausted",
                        CheckStatus.WARN,
                        detail="probe did not start before the shared deadline",
                    )
                    report.sections.append(skipped)
                break
            try:
                report.sections.append(builder())
            except Exception as e:  # noqa: BLE001 — see docstring above
                crashed = Section(builder.__name__.replace("section_", "").title())
                crashed.add(
                    f"probe crashed: {type(e).__name__}: {e}",
                    CheckStatus.FAIL,
                    detail=f"{type(e).__module__}.{type(e).__name__}: {e}",
                )
                report.sections.append(crashed)
    finally:
        _DOCTOR_DEADLINE = None
    return report


def run_all() -> Report:
    """Run one coherent probe set; serialize access to process-global caches."""
    caller_deadline = time.monotonic() + (
        _DOCTOR_BUDGET_S - _DOCTOR_COMPLETION_HEADROOM_S
    )
    remaining = max(0.0, caller_deadline - time.monotonic())
    if not _DOCTOR_RUN_LOCK.acquire(timeout=remaining):
        return _budget_exhausted_report()
    try:
        return _run_all_serialized(caller_deadline)
    finally:
        _DOCTOR_RUN_LOCK.release()
