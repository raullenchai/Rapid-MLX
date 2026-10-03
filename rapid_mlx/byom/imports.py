# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx import``: explicit, cancel-safe bf16 → MLX quantization.

Wraps ``mlx_lm.convert`` for a Hugging Face repo or local directory holding
unquantized safetensors of an architecture this install supports. It is an
explicit command only — ``serve`` and ``pull`` never convert.

Guarantees:

* **Preflight first.** Format, architecture and existing quantization are
  checked from metadata; free disk (source download + output + 10 %) and memory
  (served output + max(4 GiB, 20 % of RAM) headroom) are checked before any
  download or conversion starts.
* **Keyed cache.** An import is identified by (source, revision, mlx-lm
  version, recipe). Re-running the same import is a no-op; a different key
  under the same name is refused unless ``--force``.
* **Atomic + cancel-safe.** Conversion and the one-token smoke test run in
  child processes inside a temporary directory next to the cache; only a
  converted, smoke-tested model is renamed into place. Ctrl-C, a crash or a
  failed smoke test leaves the cache untouched, and a per-name lock keeps two
  imports of the same name apart. Hub sources download into the normal
  Hugging Face cache, so an interrupted download resumes; the source is never
  deleted.
* **Visible.** Imports are listed by ``rapid-mlx models --cached``, served by
  name (``rapid-mlx serve my-ft-4bit``) and removed by ``rapid-mlx rm``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapid_mlx.byom import preflight as pf

MANIFEST = "rapid-mlx-import.json"
GROUP_SIZE = 64
SUPPORTED_BITS = (2, 3, 4, 6, 8)
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SOURCE_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
]
WORKER_BOOTSTRAP = (
    "import sys; from rapid_mlx.byom._import_worker import main; "
    "sys.exit(main(sys.argv[1:]))"
)
_GIB = 1 << 30


class ImportRefusedError(Exception):
    """A precondition failed; the message is shown to the user as-is."""


def imports_root() -> Path:
    base = os.environ.get("RAPID_MLX_HOME", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".rapid-mlx"
    return root / "imports"


def _valid_name(name: str) -> bool:
    return bool(_NAME_RE.fullmatch(name)) and ".." not in name


def imported_model_path(name: str) -> str | None:
    """Directory of a completed import called ``name``, else ``None``."""
    if not isinstance(name, str) or "/" in name or not _valid_name(name):
        return None
    candidate = imports_root() / name
    return str(candidate) if (candidate / MANIFEST).is_file() else None


@dataclass(frozen=True)
class ImportedModel:
    name: str
    path: Path
    size: int
    source: str
    bits: int


def list_imports() -> list[ImportedModel]:
    root = imports_root()
    found: list[ImportedModel] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return found
    for entry in entries:
        # Internal recovery/lock directories are not published imports.  A
        # process can die after a successful replacement but before its
        # ``.old-*`` backup is removed, so filtering by manifest alone would
        # expose that backup as another model in ``models --cached``.
        if not _valid_name(entry.name) or entry.is_symlink() or not entry.is_dir():
            continue
        try:
            manifest = json.loads((entry / MANIFEST).read_text(encoding="utf-8"))
            size = sum(f.stat().st_size for f in entry.iterdir() if f.is_file())
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        found.append(
            ImportedModel(
                name=entry.name,
                path=entry,
                size=size,
                source=str(manifest.get("source", "?")),
                bits=int(manifest.get("bits", 0)),
            )
        )
    return found


# --------------------------------------------------------------------------
# Planning


@dataclass(frozen=True)
class Plan:
    source: str
    is_local: bool
    revision: str | None
    name: str
    bits: int
    model_type: str
    dtype: str
    source_bytes: int
    output_bytes: int
    download_bytes: int
    key: str


def _default_name(source: str, bits: int) -> str:
    base = source.rstrip("/").rsplit("/", 1)[-1].lower()
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-.") or "model"
    for suffix in ("-bf16", "-fp16", "-f16", "-hf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].strip("-.") or "model"
    return f"{base[:56]}-{bits}bit"


def _name_is_taken(name: str) -> bool:
    """Whether ``name`` already resolves to a catalog, audio or user alias."""
    from rapid_mlx.audio.registry import resolve_audio_alias
    from rapid_mlx.model_aliases import (
        _RETIRED_MODEL_ALIASES,
        list_aliases,
        user_alias_reserved_names,
    )

    lowered = name.lower()
    taken = {alias.lower() for alias in list_aliases()}
    taken |= {alias.lower() for alias in _RETIRED_MODEL_ALIASES}
    taken |= {alias.lower() for alias in user_alias_reserved_names()}
    return lowered in taken or resolve_audio_alias(name) is not None


_HASHED_FILE_MAX_BYTES = 64 << 20


def _local_revision(path: Path) -> str:
    """A change fingerprint for a local source tree.

    Every file (recursively, by relative path) contributes its size, mtime and
    inode; files up to 64 MiB (configs, tokenizers, templates) also contribute
    their content. Weight shards are not re-read: hashing tens of GB on every
    run would cost minutes, and an in-place rewrite that preserves size,
    mtime and inode is not a realistic edit.
    """
    digest = hashlib.sha256()
    for entry in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = entry.stat()
        rel = entry.relative_to(path).as_posix()
        digest.update(
            f"{rel}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino};".encode()
        )
        if stat.st_size <= _HASHED_FILE_MAX_BYTES:
            digest.update(hashlib.sha256(entry.read_bytes()).digest())
    return "local:" + digest.hexdigest()[:16]


def cache_key(source: str, revision: str | None, bits: int) -> str:
    payload = {
        "source": source,
        "revision": revision,
        "mlx_lm": pf._mlx_lm_version(),
        "recipe": f"affine-q{bits}-g{GROUP_SIZE}",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def plan_import(
    source: str,
    *,
    bits: int,
    name: str | None,
    inspection: pf.Inspection,
    supported: frozenset[str] | None,
    hub_cached: bool,
) -> Plan:
    """Validate the source from metadata and size the job. Raises ImportRefusedError."""
    if pf.classify_format(inspection.files) is not None:
        raise ImportRefusedError(
            f"{source} has no safetensors weights (GGUF or PyTorch .bin only); "
            "import converts safetensors checkpoints."
        )
    config = inspection.config
    quant = config.get("quantization") or config.get("quantization_config")
    if quant:
        raise ImportRefusedError(
            f"{source} is already quantized; serve it directly: "
            f"rapid-mlx serve {source}"
        )
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ImportRefusedError(f"{source} has no config.json model_type to convert.")
    if config.get("model_file") or config.get("auto_map"):
        raise ImportRefusedError(
            f"{source} ships its own model code (model_file/auto_map); import "
            "only converts architectures built into mlx-lm and never runs "
            "repository code."
        )
    if pf.architecture_supported(config, supported) is not True:
        raise ImportRefusedError(
            f"Architecture {model_type} has no mlx-lm converter in this install, "
            "so it cannot be imported."
        )
    if inspection.weight_bytes <= 0:
        raise ImportRefusedError(
            f"{source} has no model*.safetensors weights at its root."
        )
    dtype = (inspection.dtype or "bf16").lower().replace("bfloat16", "bf16")
    bytes_per_param = 4 if dtype in ("f32", "float32") else 2
    params = inspection.weight_bytes / bytes_per_param
    # Affine quantization stores a scale and bias per group of 64 weights.
    output_bytes = int(params * (bits + 32 / GROUP_SIZE) / 8)
    revision = inspection.revision
    if inspection.is_local:
        revision = _local_revision(Path(source))
    elif not revision:
        raise ImportRefusedError(
            f"Could not resolve {source}'s current revision; try again."
        )
    if name is None:
        name = _default_name(source, bits)
        if _name_is_taken(name):
            # Catalog and user aliases win name resolution; never shadow one.
            name = f"{name}-local"
    if not _valid_name(name):
        raise ImportRefusedError(
            f"'{name}' is not a valid import name (letters, digits, '.', '_', '-')."
        )
    if _name_is_taken(name):
        raise ImportRefusedError(
            f"'{name}' is already a model alias; pick another --name."
        )
    return Plan(
        source=source,
        is_local=inspection.is_local,
        revision=revision,
        name=name,
        bits=bits,
        model_type=model_type,
        dtype=dtype,
        source_bytes=inspection.weight_bytes,
        output_bytes=output_bytes,
        download_bytes=0
        if inspection.is_local or hub_cached
        else inspection.weight_bytes,
        key=cache_key(source, revision, bits),
    )


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def convertible_model_types() -> frozenset[str] | None:
    """Model types ``mlx_lm.convert`` itself can build (no vision/audio)."""
    types_ = pf._installed_types("mlx_lm", ("models",))
    return frozenset(types_) if types_ is not None else None


def revision_cached(repo: str, inspection: pf.Inspection) -> bool:
    """Whether the inspected revision's weights are already in the HF cache."""
    if not inspection.revision:
        return False
    owner, _, name = repo.partition("/")
    snapshot = _hf_cache_dir() / f"models--{owner}--{name}" / "snapshots"
    snapshot = snapshot / inspection.revision
    shards = [f for f in inspection.files if pf.is_runtime_weight(f)]
    return bool(shards) and all((snapshot / shard).is_file() for shard in shards)


def _hf_cache_dir() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def check_resources(plan: Plan, ram_bytes: int | None) -> list[str]:
    """Disk and memory preflight; returns the plan lines or raises."""
    root = imports_root()
    output_need = int(plan.output_bytes * 1.1)
    free_out = _free_bytes(root)
    lines = [
        f"Source   {plan.dtype} · {pf._gb(plan.source_bytes)} · "
        f"{plan.model_type} (supported)"
    ]
    if plan.download_bytes:
        cache = _hf_cache_dir()
        download_need = int(plan.download_bytes * 1.1)
        free_cache = _free_bytes(cache)
        same_fs = _same_filesystem(cache, root)
        if same_fs:
            output_need += download_need
        elif free_cache < download_need:
            raise ImportRefusedError(
                f"Downloading the source needs ~{pf._gb(download_need)} free in "
                f"the Hugging Face cache; {pf._gb(free_cache)} is free."
            )
    if free_out < output_need:
        raise ImportRefusedError(
            f"Import needs ~{pf._gb(output_need)} free disk; "
            f"{pf._gb(free_out)} is free."
        )
    lines.append(
        f"Needs    ~{pf._gb(output_need)} free disk  (you have {pf._gb(free_out)})"
    )
    if ram_bytes:
        headroom = max(4 * _GIB, int(ram_bytes * 0.2))
        if plan.output_bytes + headroom > ram_bytes:
            raise ImportRefusedError(
                f"The {plan.bits}-bit output (~{pf._gb(plan.output_bytes)}) would "
                f"not leave enough memory to serve it on this "
                f"{pf._gb(ram_bytes)} Mac. Try fewer bits (--quantize 3 or 2)."
            )
        lines.append(
            f"Output   ~{pf._gb(plan.output_bytes)}, fits in {pf._gb(ram_bytes)} ✓"
        )
    else:
        lines.append(f"Output   ~{pf._gb(plan.output_bytes)}")
    return lines


def _same_filesystem(a: Path, b: Path) -> bool:
    def _dev(path: Path) -> int:
        while not path.exists():
            path = path.parent
        return path.stat().st_dev

    return _dev(a) == _dev(b)


# --------------------------------------------------------------------------
# Execution


class _Lock:
    """Per-name exclusive lock; released on exit even after a crash."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> _Lock:
        import fcntl

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ImportRefusedError(
                "Another `rapid-mlx import` of this name is running."
            ) from None
        self._fd = fd
        return self

    def __exit__(self, *exc: object) -> None:
        import fcntl

        assert self._fd is not None
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)


# Conversion time scales with model size and has no useful bound; the
# one-token smoke test must finish quickly or the output is not usable.
SMOKE_TIMEOUT_SECONDS = 600
_STDERR_TAIL_LINES = 5


def _run_worker(*argv: str) -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER_BOOTSTRAP, *argv],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=SMOKE_TIMEOUT_SECONDS if argv[0] == "smoke" else None,
        )
    except subprocess.TimeoutExpired:
        raise ImportRefusedError(
            f"{argv[0]} step did not finish within {SMOKE_TIMEOUT_SECONDS}s."
        ) from None
    if result.returncode != 0:
        lines = (result.stderr or "").strip().splitlines()[-_STDERR_TAIL_LINES:]
        detail = "\n    ".join(lines) if lines else "(no output)"
        raise ImportRefusedError(f"{argv[0]} step failed:\n    {detail}")


def _download_source(plan: Plan) -> str:
    from huggingface_hub import snapshot_download

    path: str = snapshot_download(
        plan.source, revision=plan.revision, allow_patterns=_SOURCE_PATTERNS
    )
    return path


def _existing_key(final: Path) -> str | None:
    try:
        manifest = json.loads((final / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = manifest.get("key") if isinstance(manifest, dict) else None
    return key if isinstance(key, str) else None


def execute(
    plan: Plan,
    *,
    force: bool,
    spinner_factory: Callable[[str], Any],
    run_worker: Callable[..., None] = _run_worker,
    download: Callable[[Plan], str] = _download_source,
) -> tuple[Path, bool]:
    """Build, smoke-test and atomically publish the import.

    Returns ``(path, reused)``; ``reused`` means an identical import existed.
    """
    root = imports_root()
    root.mkdir(parents=True, exist_ok=True)
    final = root / plan.name
    with _Lock(root / ".locks" / f"{plan.name}.lock"):
        temps = list(root.glob(f".tmp-{plan.name}-*"))
        backups = sorted(root.glob(f".old-{plan.name}-*"))
        existing = _existing_key(final)
        if existing == plan.key:
            # The published model won the previous atomic replacement.  A
            # crash may still have left its full-size old copy or temp dir.
            for stale in [*temps, *backups]:
                shutil.rmtree(stale, ignore_errors=True)
            return final, True
        if final.exists() and not force:
            raise ImportRefusedError(
                f"An import named '{plan.name}' already exists from a different "
                "source or recipe. Pass --name to pick another name, or --force "
                "to replace it."
            )
        # We hold the lock: any temp dir for this name is an abandoned run,
        # and a backup without a published import is a --force replacement
        # that died between its two renames — put it back.
        if backups and not final.exists():
            os.rename(backups.pop(), final)
            existing = _existing_key(final)
            if existing == plan.key:
                for stale in [*temps, *backups]:
                    shutil.rmtree(stale, ignore_errors=True)
                return final, True
        for stale in [*temps, *backups]:
            shutil.rmtree(stale, ignore_errors=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{plan.name}-", dir=root))
        try:
            source_path = plan.source if plan.is_local else download(plan)
            out = tmp / "model"
            with spinner_factory("Quantizing … · Ctrl-C safe"):
                run_worker(
                    "convert",
                    str(source_path),
                    str(out),
                    str(plan.bits),
                    str(GROUP_SIZE),
                )
            with spinner_factory("Smoke test …"):
                run_worker("smoke", str(out))
            manifest = {
                "key": plan.key,
                "source": plan.source,
                "revision": plan.revision,
                "mlx_lm": pf._mlx_lm_version(),
                "bits": plan.bits,
                "group_size": GROUP_SIZE,
                "created_at": int(time.time()),
            }
            (out / MANIFEST).write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            if final.exists():
                # The backup sits outside the temp dir so a crash between the
                # two renames leaves a recoverable copy (restored above).
                backup = root / f".old-{plan.name}-{os.getpid()}-{time.time_ns()}"
                os.rename(final, backup)
                try:
                    os.rename(out, final)
                except BaseException:
                    # Restore immediately for ordinary I/O failures and
                    # Ctrl-C in the narrow publish window.  SIGKILL cannot run
                    # this handler; the next invocation recovers the backup.
                    try:
                        if not final.exists() and backup.exists():
                            os.rename(backup, final)
                    except OSError:
                        pass
                    raise
                shutil.rmtree(backup, ignore_errors=True)
            else:
                os.rename(out, final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return final, False


def _import_dir_for(target: str) -> Path | None:
    """The import a resolved ``rm`` target points at (name or its directory)."""
    named = imported_model_path(target)
    if named is not None:
        return Path(named)
    try:
        candidate = Path(target).resolve()
        root = imports_root().resolve()
    except (OSError, ValueError):
        return None
    if candidate.parent == root and (candidate / MANIFEST).is_file():
        return candidate
    return None


def remove_import(target: str, *, assume_yes: bool) -> bool:
    """``rapid-mlx rm`` for an import. Returns False if not an import.

    ``target`` is the alias-resolved model: an import's bare name resolves to
    its directory, while a catalog alias resolves elsewhere and is left alone.
    """
    found = _import_dir_for(target)
    if found is None:
        return False
    name = found.name
    path = str(found)
    from rapid_mlx.cli import _format_bytes

    size = sum(f.stat().st_size for f in Path(path).iterdir() if f.is_file())
    if not assume_yes:
        try:
            answer = input(
                f"Remove imported model {name} ({_format_bytes(size)})? [y/N] "
            )
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("  Cancelled.")
            return True
    shutil.rmtree(path)
    print(f"  Removed imported model {name} ({_format_bytes(size)}).")
    return True


def print_imports_section() -> None:
    """Extra block for ``models --cached`` (kept out of the parsed table)."""
    imported = list_imports()
    if not imported:
        return
    from rapid_mlx.cli import _format_bytes

    print("  ── Imported with `rapid-mlx import` (serve or rm by name) ──")
    for item in imported:
        print(
            f"  • {item.name}  {_format_bytes(item.size)}  "
            f"from {item.source} · {item.bits}-bit"
        )
    print()


# --------------------------------------------------------------------------
# CLI


def import_command(args: Any, *, spinner_factory: Callable[[str], Any]) -> None:
    source = args.source
    bits = int(args.quantize)
    is_local = os.path.exists(source)
    if not is_local and "/" not in source:
        print(
            f"\n  Error: '{source}' is neither a local directory nor a Hugging "
            "Face repo id (org/name).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        if is_local:
            inspection = pf.inspect_local(source)
        else:
            with spinner_factory(f"Checking {source.split('/')[-1]} …"):
                inspection = pf.inspect_hub(source)
        if inspection is None:
            raise ImportRefusedError(
                f"Could not read {source}'s metadata (check the name, your "
                "network, or `huggingface-cli login` for gated repos)."
            )
        plan = plan_import(
            source,
            bits=bits,
            name=args.name,
            inspection=inspection,
            supported=convertible_model_types(),
            hub_cached=not is_local and revision_cached(source, inspection),
        )
        lines = check_resources(plan, pf.physical_ram_bytes())
        print()
        for line in lines:
            print(f"  {line}")
        final, reused = execute(plan, force=args.force, spinner_factory=spinner_factory)
    except ImportRefusedError as exc:
        print(f"\n  ✗ {exc}\n", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("\n  Cancelled. The import cache is unchanged.\n", file=sys.stderr)
        raise SystemExit(130) from None
    print("  ✓ Already imported" if reused else "  ✓ Smoke test passed")
    print(f"  rapid-mlx serve {plan.name}")
    print(f"  (stored at {final})\n")
