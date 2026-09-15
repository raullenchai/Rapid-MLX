# SPDX-License-Identifier: Apache-2.0
"""Adapter for the standalone LTX-2.5 MLX runtime."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

LTX25_RUNTIME_COMMIT = "fcbd6f31e5a80d513c45550d960c2f598a5c3ffb"
LTX25_RUNTIME_REPOSITORY = "https://github.com/raullenchai/ltx-2-mlx.git"
LTX25_RUNTIME_VERSION = "0.14.15"
# Stamped into each embedded distribution's .dist-info by build-sidecar.sh;
# its content is the audited source commit the packages were built from.
LTX25_PROVENANCE_FILE = "RAPID_LTX25_PROVENANCE"
_DEFAULT_TIMEOUT_SECONDS = 7200
_TERMINATE_GRACE_SECONDS = 10
_RUNTIME_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE: tempfile.TemporaryDirectory[str] | None = None
_FAST_MODEL_ENV = "RAPID_MLX_LTX25_FAST_MODEL"
_FAST_STAGE1_MANIFEST = "fast-stage1-exact-prefix.json"
_FAST_STAGE2_MANIFEST = "fast-stage2.json"
_FAST_DEQUANT_MATMUL_MIN_TOKENS = "1024"
_MAX_FAST_MANIFEST_BYTES = 64 * 1024
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FAST_STAGE1_PROFILES = {
    "ltx_stage1_exact_high_noise_two_middle_spans_v1": {
        "execution_spans": [[0, 1], [1, 2], [2, 3], [3, 5], [5, 7]],
        "learned_spans": [[3, 5], [5, 7]],
        "schedule": [1.0, 0.99375, 0.9875, 0.98125, 0.909375, 0.421875, 0.0],
        "evaluations": 6,
    },
}
_INNER_PROMPT_RUNNER = """\
import signal
import sys
from ltx_pipelines_mlx.cli import main

signal.signal(signal.SIGTERM, signal.SIG_DFL)
sys.argv = ["ltx-2-mlx", *sys.argv[1:], "--prompt", sys.stdin.read()]
main()
"""
_STDIN_PROMPT_RUNNER = f"""\
import os
import signal
import subprocess
import sys

prompt = sys.stdin.read()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen(
    [sys.executable, "-c", {_INNER_PROMPT_RUNNER!r}, *sys.argv[1:]],
    stdin=subprocess.PIPE,
    text=True,
)
try:
    child.communicate(input=prompt)
finally:
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except ProcessLookupError:
        pass
raise SystemExit(child.returncode)
"""


def _isolated_python_environment() -> dict[str, str]:
    """Return an environment that does not pin a child to Rapid's Python.

    The signed desktop sidecar deliberately exports ``PYTHONHOME`` and
    ``PYTHONPATH`` so its own interpreter cannot import host packages.  Those
    variables must not cross the runtime boundary: uv's build-isolation
    interpreters and the provisioned LTX interpreter otherwise try to start
    with Rapid's stdlib/site-packages and fail before importing ``encodings``.
    """
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def is_ltx25_model(model_name: str | None) -> bool:
    """Return whether a model identifier explicitly selects LTX-2.5."""
    if not model_name:
        return False
    normalized = model_name.casefold().replace("_", "-")
    return normalized.rsplit("/", 1)[-1] == "ltx-2.5-mlx-q8"


def resolve_ltx25_runtime() -> str | None:
    """Resolve the CLI only when its checkout is at the audited revision."""
    override = os.environ.get("RAPID_MLX_LTX25_RUNTIME", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            absolute = str(candidate.absolute())
            return (
                absolute
                if _runtime_revision(absolute) == LTX25_RUNTIME_COMMIT
                else None
            )
        return None
    executable = shutil.which("ltx-2-mlx")
    if executable is None:
        return None
    absolute = str(Path(executable).absolute())
    return absolute if _runtime_revision(absolute) == LTX25_RUNTIME_COMMIT else None


def embedded_ltx25_interpreter() -> str | None:
    """Return this interpreter when the audited LTX packages are embedded.

    Desktop cannot provision a Git checkout or an ``uv`` environment after
    signing.  A version number alone is not proof of provenance — any
    same-version distribution from an index would satisfy it — so the sidecar
    build stamps each embedded distribution with the audited source commit,
    and both stamps must match ``LTX25_RUNTIME_COMMIT`` exactly.
    """
    try:
        for name in ("ltx-core-mlx", "ltx-pipelines-mlx"):
            distribution = importlib.metadata.distribution(name)
            if distribution.version != LTX25_RUNTIME_VERSION:
                return None
            provenance = distribution.read_text(LTX25_PROVENANCE_FILE)
            if provenance is None or provenance.strip() != LTX25_RUNTIME_COMMIT:
                return None
        cli_present = importlib.util.find_spec("ltx_pipelines_mlx.cli") is not None
    except (
        OSError,
        ImportError,
        ModuleNotFoundError,
        importlib.metadata.PackageNotFoundError,
    ):
        return None
    return sys.executable if cli_present else None


def _runtime_revision(executable: str) -> str | None:
    """Verify the documented workspace points at the pinned Git revision."""
    path = Path(executable)
    try:
        repository = path.parents[2]
    except IndexError:
        return None
    if not (repository / ".git").exists():
        return None
    if path.absolute() != (repository / ".venv" / "bin" / "ltx-2-mlx").absolute():
        return None
    try:

        def git(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
                env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
            )

        result = git("rev-parse", "HEAD")
        git("ls-files", "--error-unmatch", "uv.lock")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _materialize_runtime(repository: Path, destination: Path) -> None:
    """Extract only tracked files from the audited commit into a fresh tree."""
    try:
        archive = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                LTX25_RUNTIME_COMMIT,
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        ).stdout
        root = destination.resolve()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            members = source.getmembers()
            for member in members:
                target = (destination / member.name).resolve()
                if not target.is_relative_to(root) or not (
                    member.isfile() or member.isdir()
                ):
                    raise LTX25BackendError(
                        "The pinned LTX-2.5 source archive contains an unsafe entry."
                    )
            source.extractall(destination, members=members)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        tarfile.TarError,
    ) as exc:
        raise LTX25BackendError(
            "The pinned LTX-2.5 source snapshot could not be materialized."
        ) from exc


# URLs with userinfo (https://user:token@index.example, including
# percent-encoded forms like https://build%40corp:s3cret@…) can appear in
# uv's package-index error output; redact the ENTIRE userinfo before
# display — anything between ``scheme://`` and ``@`` is treated as
# potentially credential-bearing.
_CREDENTIAL_URL_RE = re.compile(r"(\w[\w+.-]*://)[^/@\s]+@")
# Token-bearing query parameters / assignments (``?token=…``,
# ``access_key=…``, ``Authorization: Bearer …``) that package-index
# errors can echo back.
# Any identifier ENDING in a credential noun (token / secret / password /
# passwd / key / credential, optional plural) — covers access_token,
# client_secret, api-key, HF_TOKEN, AWS_SECRET_ACCESS_KEY, … without
# enumerating prefixes. Values may be bare or quoted.
_CREDENTIAL_PARAM_RE = re.compile(
    # Suffix set includes signed-URL auth params (codex on #2166):
    # ``X-Amz-Signature=``, ``sig=`` (Azure SAS), ``sas=``, ``auth=``,
    # ``jwt=`` — none of which end in the classic credential suffixes,
    # and uv stderr can echo a full index URL query string verbatim.
    r"(?i)((?<![\w-])[\w-]*"
    r"(?:token|secret|password|passwd|key|credential|signature|sig|auth|sas|jwt)s?"
    r"\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^&\s\"']+)"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[a-z0-9._~+/=-]+")
# Any ``Authorization:``-style header value regardless of scheme (Basic,
# Digest, custom) — the scheme word is kept, the credential is redacted.
_AUTH_HEADER_RE = re.compile(r"(?i)\b(authorization\s*[=:]\s*[a-z0-9_-]+\s+)[^\s\"']+")


def _sanitize_diagnostic(text: str) -> str:
    """Make subprocess output safe to print: no control sequences (terminal
    escape injection), no embedded index credentials (URL userinfo,
    token-bearing query params, bearer headers)."""
    text = _CREDENTIAL_URL_RE.sub(r"\1***@", text)
    text = _AUTH_HEADER_RE.sub(r"\1***", text)
    text = _CREDENTIAL_PARAM_RE.sub(r"\1***", text)
    text = _BEARER_RE.sub(r"\1***", text)
    return "".join(ch if ch.isprintable() or ch in " \t\n" else " " for ch in text)


def _stderr_tail(stream: io.BufferedRandom) -> str:
    """Read a bounded tail from a spooled stderr file (never the whole body)."""
    try:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 4096))
        return stream.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _bounded(text: str) -> str:
    return text[:300] + "…" if len(text) > 300 else text


def _provisioning_failure_detail(exc: Exception) -> str:
    """Compress a provisioning failure into one actionable, sanitized line.

    EVERY exception-derived string passes through ``_sanitize_diagnostic``
    and the length bound — ``OSError`` messages can embed paths or
    environment-derived text with the same control-sequence/credential
    exposure as uv's stderr.
    """
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"`uv sync --frozen` timed out after {int(exc.timeout)}s"
    if isinstance(exc, subprocess.CalledProcessError):
        detail = f"`uv sync --frozen` failed with exit code {exc.returncode}"
        stderr = _sanitize_diagnostic((exc.stderr or "").strip())
        if stderr:
            detail += f" ({_bounded(' | '.join(stderr.splitlines()[-3:]))})"
        return detail
    return _bounded(_sanitize_diagnostic(str(exc))) or type(exc).__name__


def prepare_ltx25_runtime(executable: str) -> Path:
    """Provision the pinned runtime once into a process-private workspace."""
    global _RUNTIME_CACHE

    with _RUNTIME_CACHE_LOCK:
        if _RUNTIME_CACHE is not None:
            return Path(_RUNTIME_CACHE.name)
        uv = shutil.which("uv")
        if uv is None:
            raise LTX25BackendError(
                "LTX-2.5 support requires uv to build its pinned runtime."
            )
        cache: tempfile.TemporaryDirectory[str] | None = None
        try:
            cache = tempfile.TemporaryDirectory(prefix="rapidmlx-ltx25-runtime-")
            workspace = Path(cache.name)
            _materialize_runtime(Path(executable).parents[2], workspace)
            # stderr goes to a temp file, not PIPE: a noisy dependency build
            # must never buffer unbounded output in the server's memory. Only
            # a bounded tail is read back, and only on failure.
            with tempfile.TemporaryFile() as uv_stderr:
                try:
                    subprocess.run(
                        [
                            str(Path(uv).absolute()),
                            "sync",
                            "--frozen",
                            "--project",
                            str(workspace),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=uv_stderr,
                        timeout=1800,
                        env=_isolated_python_environment(),
                    )
                except subprocess.CalledProcessError as run_exc:
                    # Real runs route stderr to the file, so the exception
                    # carries none; keep any stderr already attached.
                    if not run_exc.stderr:
                        run_exc.stderr = _stderr_tail(uv_stderr)
                    raise
            interpreter = workspace / ".venv" / "bin" / "python"
            if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                raise LTX25BackendError(
                    "The pinned LTX-2.5 runtime did not create its Python environment."
                )
        except LTX25BackendError:
            if cache is not None:
                cache.cleanup()
            raise
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            if cache is not None:
                cache.cleanup()
            raise LTX25BackendError(
                "The pinned LTX-2.5 runtime could not be provisioned: "
                + _provisioning_failure_detail(exc)
            ) from exc
        _RUNTIME_CACHE = cache
        return workspace


def _prepared_ltx25_runtime() -> Path | None:
    with _RUNTIME_CACHE_LOCK:
        return Path(_RUNTIME_CACHE.name) if _RUNTIME_CACHE is not None else None


def _generation_timeout_seconds() -> int:
    raw = os.environ.get("RAPID_MLX_LTX25_TIMEOUT_SEC", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise LTX25BackendError(
            "The LTX-2.5 timeout must be an integer number of seconds."
        ) from exc
    if value < 60:
        raise LTX25BackendError("The LTX-2.5 timeout must be at least 60 seconds.")
    return value


class LTX25BackendError(RuntimeError):
    """Safe, public-facing error from the LTX-2.5 backend."""


@dataclass(frozen=True)
class LTX25FastConfig:
    """Validated operator-selected portable fast model package."""

    model_dir: Path
    base_model_id: str
    qualification_revision: str
    base_revision: str
    stage1_evaluations: int


def _local_fast_file(model_dir: Path, value: object, *, suffix: str) -> Path:
    if not isinstance(value, str):
        raise LTX25BackendError(
            "The configured LTX-2.5 fast manifest is missing a package file."
        )
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.suffix != suffix:
        raise LTX25BackendError(
            "The configured LTX-2.5 fast manifest contains an unsafe package path."
        )
    path = model_dir / relative
    if not path.is_file():
        raise LTX25BackendError(
            "The configured LTX-2.5 fast model is missing a package file."
        )
    return path


def _read_fast_manifest(path: Path) -> dict:
    try:
        if not path.is_file():
            raise ValueError
        with path.open("rb") as source:
            raw = source.read(_MAX_FAST_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_FAST_MANIFEST_BYTES:
            raise ValueError
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise LTX25BackendError(
            "The configured LTX-2.5 fast model has a missing or invalid manifest."
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise LTX25BackendError(
            "The configured LTX-2.5 fast model requires schema_version=1 manifests."
        )
    return value


def resolve_ltx25_fast_config() -> LTX25FastConfig | None:
    """Resolve a model-contained fast package without inspecting host identity."""
    raw = os.environ.get(_FAST_MODEL_ENV, "").strip()
    if not raw:
        return None
    try:
        model_dir = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise LTX25BackendError(
            f"{_FAST_MODEL_ENV} does not name an available fast model directory."
        ) from exc
    if not model_dir.is_dir():
        raise LTX25BackendError(
            f"{_FAST_MODEL_ENV} does not name an available fast model directory."
        )

    stage1 = _read_fast_manifest(model_dir / _FAST_STAGE1_MANIFEST)
    stage2 = _read_fast_manifest(model_dir / _FAST_STAGE2_MANIFEST)
    capability = stage1.get("capability")
    profile = (
        _FAST_STAGE1_PROFILES.get(capability) if isinstance(capability, str) else None
    )
    if profile is None:
        raise LTX25BackendError(
            "The configured LTX-2.5 fast model does not advertise the qualified two-middle exact-prefix capability."
        )
    if (
        stage1.get("execution_spans") != profile["execution_spans"]
        or stage1.get("learned_spans") != profile["learned_spans"]
        or stage1.get("clean_final_span") != [7, 8]
        or stage1.get("schedule") != profile["schedule"]
    ):
        raise LTX25BackendError(
            "The configured LTX-2.5 fast model does not use the qualified exact-prefix schedule."
        )
    qualification = stage1.get("qualification_revision")
    if not isinstance(qualification, str) or not qualification:
        raise LTX25BackendError(
            "The configured LTX-2.5 fast model has no qualification revision."
        )
    shared_fields = ("base_model_id", "base_revision", "transformer_sha256")
    if any(
        not isinstance(stage1.get(field), str)
        or not stage1[field]
        or stage1[field] != stage2.get(field)
        for field in shared_fields
    ):
        raise LTX25BackendError(
            "The configured LTX-2.5 fast manifests do not bind the same base model."
        )
    if not _IMMUTABLE_REVISION_RE.fullmatch(stage1["base_revision"]):
        raise LTX25BackendError(
            "The configured LTX-2.5 fast model does not bind an immutable base revision."
        )
    digest_fields = (
        stage1.get("transformer_sha256"),
        stage2.get("transformer_sha256"),
        stage2.get("adapter_sha256"),
    )
    if any(
        not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
        for value in digest_fields
    ):
        raise LTX25BackendError(
            "The configured LTX-2.5 fast manifests require lowercase SHA-256 digests."
        )
    transformer = _local_fast_file(
        model_dir, stage1.get("transformer_file"), suffix=".safetensors"
    )
    if stage2.get("transformer_file") != transformer.name:
        raise LTX25BackendError(
            "The configured LTX-2.5 fast manifests do not select the same transformer."
        )
    segments = stage1.get("segments")
    learned_spans = profile["learned_spans"]
    if not isinstance(segments, list) or len(segments) != len(learned_spans):
        raise LTX25BackendError(
            "The configured LTX-2.5 fast model has the wrong learned middle segment count."
        )
    for segment, expected_span in zip(segments, learned_spans, strict=True):
        if not isinstance(segment, dict):
            raise LTX25BackendError(
                "The configured LTX-2.5 fast model has an invalid learned middle segment."
            )
        segment_digest = segment.get("adapter_sha256")
        if (
            segment.get("span") != expected_span
            or not isinstance(segment_digest, str)
            or not _SHA256_RE.fullmatch(segment_digest)
        ):
            raise LTX25BackendError(
                "The configured LTX-2.5 fast model has an invalid learned middle segment."
            )
        _local_fast_file(model_dir, segment.get("adapter_file"), suffix=".safetensors")
    _local_fast_file(model_dir, stage2.get("adapter_file"), suffix=".safetensors")
    return LTX25FastConfig(
        model_dir=model_dir,
        base_model_id=stage1["base_model_id"],
        qualification_revision=qualification,
        base_revision=stage1["base_revision"],
        stage1_evaluations=profile["evaluations"],
    )


class LTX25VideoEngine:
    """Run LTX-2.5 through its native MLX command-line runtime."""

    native_fps = 24

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stopping = False

    def fast_mode_details(self) -> dict | None:
        """Return advertised fast-mode metadata, or None when not configured."""
        try:
            config = resolve_ltx25_fast_config()
        except LTX25BackendError:
            return None
        if config is None:
            return None
        if config.base_model_id != self.model_name:
            return None
        return {
            "qualification_revision": config.qualification_revision,
            "base_revision": config.base_revision,
            "experimental": config.qualification_revision.startswith("diagnostic-"),
            "operation_modes": ["text-to-video"],
            "stage1_evaluations": config.stage1_evaluations,
            "stage2_evaluations": 1,
        }

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        leader_running = process.poll() is None
        if not leader_running:
            # The runner is a group supervisor: before it exits it signals the
            # still-owned process group, so no stale PGID needs to be reused.
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
            return
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            # The unreaped leader still owns this PID/PGID, so reuse is
            # impossible and escalation is safe.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise LTX25BackendError(
                "The LTX-2.5 runtime process group could not be reaped."
            ) from exc

    def stop(self) -> None:
        """Stop an active external generation during bounded shutdown."""
        with self._process_lock:
            self._stopping = True
            process = self._process
        if process is not None:
            self._terminate_process(process)

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
        image: Path | None,
        conditioning_strength: float | None = None,
        generation_mode: str = "standard",
    ) -> None:
        if generation_mode not in {"standard", "fast"}:
            raise LTX25BackendError("generation_mode must be standard or fast.")
        fast_config = resolve_ltx25_fast_config() if generation_mode == "fast" else None
        if generation_mode == "fast" and fast_config is None:
            raise LTX25BackendError(
                "LTX-2.5 fast mode is not configured; set RAPID_MLX_LTX25_FAST_MODEL to a qualified model package."
            )
        if fast_config is not None and fast_config.base_model_id != self.model_name:
            raise LTX25BackendError(
                "The configured LTX-2.5 fast model does not match the served model."
            )
        timeout = _generation_timeout_seconds()
        interpreter = embedded_ltx25_interpreter()
        environment = os.environ.copy()
        workspace = None
        if interpreter is None:
            workspace = _prepared_ltx25_runtime()
        if interpreter is None and workspace is None:
            executable = resolve_ltx25_runtime()
            if executable is None:
                raise LTX25BackendError(
                    "LTX-2.5 support requires the pinned ltx-2-mlx runtime. "
                    "See the LTX-2.5 setup in the video generation guide."
                )
            workspace = prepare_ltx25_runtime(executable)
        if interpreter is None:
            assert workspace is not None
            interpreter = str(workspace / ".venv" / "bin" / "python")
            environment = _isolated_python_environment()
        try:
            staging = tempfile.TemporaryDirectory(
                prefix=f".{output_path.name}.",
                dir=output_path.parent,
            )
            staged_output = Path(staging.name) / "output.mp4"
        except OSError as exc:
            raise LTX25BackendError(
                "LTX-2.5 generation could not create its temporary output."
            ) from exc

        command = [
            interpreter,
            "-c",
            _STDIN_PROMPT_RUNNER,
            "generate",
            "--model",
            str(fast_config.model_dir) if fast_config is not None else self.model_name,
            "--distilled",
            "--low-ram",
            "--quiet",
            "--height",
            str(height),
            "--width",
            str(width),
            "--frames",
            str(num_frames),
            "--frame-rate",
            str(fps),
            "--seed",
            str(seed),
            "--output",
            str(staged_output),
        ]
        if fast_config is not None:
            command.extend(
                [
                    "--fast-stage1-segmented-manifest",
                    _FAST_STAGE1_MANIFEST,
                    "--fast-stage2-manifest",
                    _FAST_STAGE2_MANIFEST,
                ]
            )
            environment["LTX2_DEQUANT_MATMUL_MIN_TOKENS"] = (
                _FAST_DEQUANT_MATMUL_MIN_TOKENS
            )
        if image is not None:
            command.extend(
                [
                    "--image",
                    str(image),
                    "0",
                    str(
                        1.0 if conditioning_strength is None else conditioning_strength
                    ),
                ]
            )
        process: subprocess.Popen[str] | None = None
        try:
            # Prompts may contain private user data. Keep them out of argv and
            # local process listings by feeding the isolated runtime over stdin.
            with self._process_lock:
                if self._stopping:
                    raise LTX25BackendError(
                        "LTX-2.5 generation cannot start while the server is stopping."
                    )
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    start_new_session=True,
                    env=environment,
                )
                self._process = process
            process.communicate(input=prompt, timeout=timeout)
            if process.returncode:
                raise LTX25BackendError(
                    f"LTX-2.5 runtime exited with code {process.returncode}; "
                    "runtime output is not retained because it may contain request data."
                )
            if not staged_output.is_file() or staged_output.stat().st_size == 0:
                raise LTX25BackendError(
                    "LTX-2.5 generation completed without an MP4 output."
                )
            os.replace(staged_output, output_path)
        except subprocess.TimeoutExpired as exc:
            if process is not None:
                self._terminate_process(process)
            raise LTX25BackendError(
                "LTX-2.5 generation exceeded its configured time limit."
            ) from exc
        except LTX25BackendError:
            if process is not None:
                self._terminate_process(process)
            raise
        except BaseException as exc:
            if process is not None:
                self._terminate_process(process)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise LTX25BackendError(
                "LTX-2.5 generation failed while running its isolated runtime."
            ) from exc
        finally:
            with self._process_lock:
                if self._process is process:
                    self._process = None
            staging.cleanup()
