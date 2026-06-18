# SPDX-License-Identifier: Apache-2.0
"""Wire protocol for the KV cache export/import HTTP API (issue #476).

Two concepts that look similar but are NOT the same — keep them straight:

* ``PROTOCOL_VERSION`` (this module) — the version of the *manifest* the
  HTTP API agrees on. Bumped when the manifest schema below changes shape
  in a way old clients can't read.
* ``index["version"]`` (in ``vllm_mlx/memory_cache.py``) — the on-disk
  format of the engine's prefix-cache directory itself (entry layout,
  safetensors keys, etc.). Bumped independently when the engine changes
  how it serializes entries.

A manifest sits **alongside** the engine's ``index.json`` at the export
root and describes "what model produced this blob, with what quant /
paged-cache / turboquant-kv settings, and at what protocol version" so a
peer instance can refuse a mismatched import before touching tensors.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROTOCOL_VERSION = "1"
MANIFEST_FILENAME = "manifest.json"

# Default sandbox root for export/import paths. Overridable via the
# ``RAPID_MLX_CACHE_EXPORT_DIR`` env var. All caller-supplied paths must
# resolve inside this directory after symlink expansion — otherwise a
# bearer-token holder could write arbitrary files anywhere on disk.
_DEFAULT_EXPORT_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "rapid-mlx", "cache_exports"
)
_EXPORT_DIR_ENV = "RAPID_MLX_CACHE_EXPORT_DIR"


class InvalidExportPathError(ValueError):
    """Raised when a caller-supplied path escapes the sandbox root."""


class ManifestNotFoundError(FileNotFoundError):
    """Raised when ``read_manifest`` is called on a path without one."""


class ManifestMismatchError(ValueError):
    """Raised when a manifest doesn't match caller expectations.

    Carries both sides so routes can surface a structured 409 body.
    """

    def __init__(self, field: str, expected: str, actual: str) -> None:
        super().__init__(
            f"manifest {field} mismatch: expected {expected!r}, got {actual!r}"
        )
        self.field = field
        self.expected = expected
        self.actual = actual


@dataclass
class Manifest:
    """Header describing the engine-cache blob at an export root.

    Additive-only: new fields MUST default to a value old readers will
    treat as "unknown / unset". Removing or renaming a field is a
    breaking change and requires bumping ``PROTOCOL_VERSION``.
    """

    protocol_version: str = PROTOCOL_VERSION
    model_id: str = ""
    quantization: str = ""
    paged_cache: bool = False
    turboquant_kv: bool = False
    # The engine's on-disk index format version (``index["version"]``)
    # at the time of export. Separate from protocol_version above —
    # see the module docstring.
    index_format_version: int = 0
    entries: int = 0
    total_bytes: int = 0
    # Free-form provenance — exporting instance's rapid-mlx version
    # and a timestamp. Importers MUST NOT gate on these (they're
    # informational), but they're invaluable for debugging.
    rapid_mlx_version: str = ""
    created_at: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        # Drop any unknown keys so a future writer adding fields doesn't
        # break an older reader — additive evolution is the whole point.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def write_manifest(root: Path, manifest: Manifest) -> Path:
    """Write ``manifest.json`` under ``root``. Returns the path written."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / MANIFEST_FILENAME
    target.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return target


def read_manifest(root: Path) -> Manifest:
    """Read and parse ``manifest.json`` at ``root``.

    Raises ``ManifestNotFoundError`` if missing — distinct from the
    generic ``FileNotFoundError`` so route handlers can map it to a 404
    without misclassifying other I/O failures.
    """
    target = root / MANIFEST_FILENAME
    if not target.is_file():
        raise ManifestNotFoundError(f"manifest.json not found in {root}")
    return Manifest.from_dict(json.loads(target.read_text()))


def default_export_root() -> Path:
    """The sandbox root all caller-supplied paths must resolve inside."""
    raw = os.environ.get(_EXPORT_DIR_ENV) or _DEFAULT_EXPORT_DIR
    # realpath here too — if the operator points the env at a symlink,
    # we sandbox to its target so commonpath comparisons stay sound.
    return Path(os.path.realpath(os.path.expanduser(raw)))


def resolve_cache_dir(caller_path: str | None) -> Path:
    """Resolve ``caller_path`` into a vetted absolute path under the sandbox.

    Rules (all must hold):

    1. ``None`` → the sandbox root itself.
    2. Absolute paths must resolve under the sandbox root after realpath
       (catches symlink escape).
    3. Relative paths are resolved against the sandbox root.
    4. The literal ``..`` segment is rejected even before resolution so
       a CVE in ``realpath`` can't bypass the sandbox.
    5. No segment may be a symlink pointing outside the sandbox — walk
       and check each ancestor of the final path.

    Raises ``InvalidExportPathError`` on any violation.
    """
    root = default_export_root()
    root.mkdir(parents=True, exist_ok=True)

    if caller_path is None or caller_path == "":
        return root

    if ".." in Path(caller_path).parts:
        raise InvalidExportPathError(
            f"path component '..' is not allowed: {caller_path!r}"
        )

    candidate = Path(caller_path)
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = Path(os.path.realpath(candidate))

    try:
        common = Path(os.path.commonpath([str(root), str(resolved)]))
    except ValueError as exc:
        # commonpath raises on cross-drive paths (e.g. Windows). Treat
        # as escape — we're not crossing drives intentionally on macOS.
        raise InvalidExportPathError(
            f"path {caller_path!r} could not be compared to sandbox root"
        ) from exc

    if common != root:
        raise InvalidExportPathError(
            f"path {caller_path!r} resolves outside sandbox {root}"
        )

    # Symlink walk — even if realpath landed us inside the sandbox, an
    # intermediate symlink that *points* outside lets a future write
    # follow it. Reject any symlink ancestor whose realpath escapes.
    cursor = candidate
    while cursor != cursor.parent:
        if cursor.is_symlink():
            target = Path(os.path.realpath(cursor))
            try:
                if Path(os.path.commonpath([str(root), str(target)])) != root:
                    raise InvalidExportPathError(
                        f"symlink {cursor} points outside sandbox"
                    )
            except ValueError as exc:
                raise InvalidExportPathError(
                    f"symlink {cursor} could not be compared to sandbox root"
                ) from exc
        cursor = cursor.parent

    return resolved
