"""Immutable artifact contract for the qualified V4.1/DSpark product lane."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

TARGET_REPO = "rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX"
TARGET_REVISION = "a25fec277b9e7cedc0e9f3f15da874a5cf9d491b"
MTP_REPO = "Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP"
MTP_REVISION = "802f1a00982705d81b79ad1c83aa0ccc0b863ebc"


@dataclass(frozen=True)
class ArtifactFile:
    name: str
    size: int
    sha256: str


MTP_FILES = (
    ArtifactFile(
        "config.json",
        23_617,
        "c615d9c9469bc66273306b08c5a92670abdbefd7edc31de7e01dd7d9719f489e",
    ),
    ArtifactFile(
        "model-00044-of-00048.safetensors",
        1_496_184_720,
        "772bdc5044bc8acfdb38e38749d784d50019b1846ed2b85337993a5fba530663",
    ),
    ArtifactFile(
        "model-00045-of-00048.safetensors",
        1_471_598_024,
        "b200f6ce5e726e38a684a04f941e1321fc58a1c0ba6a2b7ab747f7ddb753ff3e",
    ),
    ArtifactFile(
        "model-00046-of-00048.safetensors",
        1_492_295_832,
        "6890e30fcb9b6fa759f43ab21424cd748242a734f263b469b148da0834449352",
    ),
    ArtifactFile(
        "model.safetensors.index.json",
        11_268_410,
        "796eb8ceeeec2cba865fa1986238aa5214d9479d8225151b8b79b04126b33d62",
    ),
)
MTP_ALLOW_PATTERNS = tuple(file.name for file in MTP_FILES)
MIN_UNIFIED_MEMORY_GB = 224.0


def is_product_target(model_name: str | None) -> bool:
    return model_name == TARGET_REPO


def require_product_memory() -> float:
    """Fail closed for programmatic callers that bypass the CLI catalog gate."""
    import psutil

    # Keep this lightweight artifact gate importable on non-MLX hosts (notably
    # Linux CI). Importing the general hardware module eagerly imports MLX even
    # though this check only needs physical memory.
    total = float(psutil.virtual_memory().total) / (1024**3)
    if total < MIN_UNIFIED_MEMORY_GB:
        raise RuntimeError(
            "DeepSeek V4.1 DSpark K4 requires at least "
            f"{MIN_UNIFIED_MEMORY_GB:.0f} GiB unified memory; this Mac reports "
            f"{total:.1f} GiB"
        )
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_mtp_snapshot(snapshot: str | Path) -> Path:
    """Fail closed unless the pinned sidecar is the qualified byte set.

    Hugging Face stores LFS files as snapshot symlinks whose blob basename is
    the SHA-256. Trust that content-addressed identity only when the target is
    contained by this repository's ``blobs`` directory. Regular files (and
    non-CAS symlinks) are hashed directly.
    """
    root = Path(snapshot).resolve()
    blobs = (
        (root.parents[1] / "blobs").resolve()
        if root.parent.name == "snapshots"
        else None
    )
    for expected in MTP_FILES:
        link = Path(snapshot) / expected.name
        if not link.is_file():
            raise FileNotFoundError(f"missing pinned DSpark artifact: {expected.name}")
        resolved = link.resolve(strict=True)
        if resolved.stat().st_size != expected.size:
            raise ValueError(f"DSpark artifact size mismatch: {expected.name}")
        cas_verified = False
        if link.is_symlink() and blobs is not None:
            try:
                resolved.relative_to(blobs)
            except ValueError:
                pass
            else:
                cas_verified = resolved.name == expected.sha256
        if not cas_verified and _sha256(resolved) != expected.sha256:
            raise ValueError(f"DSpark artifact hash mismatch: {expected.name}")
    return Path(snapshot)


def download_mtp_snapshot() -> Path:
    """Resolve only the 4.47 GB data subset needed by the owned runtime."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MTP_REPO,
        revision=MTP_REVISION,
        allow_patterns=list(MTP_ALLOW_PATTERNS),
    )
    return verify_mtp_snapshot(path)


def download_target_snapshot() -> Path:
    """Resolve the exact target checkpoint revision; never follow moving main."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(TARGET_REPO, revision=TARGET_REVISION))
