"""Immutable artifact contract for the qualified V4.1/DSpark product lane."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

TARGET_REPO = "rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX"
TARGET_REVISION = "a25fec277b9e7cedc0e9f3f15da874a5cf9d491b"
MTP_REPO = "rapid-mlx/DeepSeek-V4.1-Flash-DSpark-4d2e-MLX"
MTP_REVISION = "9530d6d2bf59e0d05177bd538095d5704ded1488"


@dataclass(frozen=True)
class ArtifactFile:
    name: str
    size: int
    sha256: str


MTP_FILES = (
    ArtifactFile(
        "config.json",
        138_476,
        "3518b608fd8b90bc518c7d4533c751367fe3c2416aa3a4f7fde40bc6e498f012",
    ),
    ArtifactFile(
        "dspark-mixed-stage-0.safetensors",
        1_556_346_856,
        "381a7fbc8758cd86baab55e8f1ae3020e49e2977f067d4269416889c1aee8118",
    ),
    ArtifactFile(
        "dspark-mixed-stage-1.safetensors",
        1_512_099_424,
        "2ca7dc77528ebe3220e6e1633733925ca3420e346b4475163e1639541333876e",
    ),
    ArtifactFile(
        "dspark-mixed-stage-2.safetensors",
        1_549_346_368,
        "1ed4663f0487e13372e753b2a5e8b760ae8fd43ba661fae891dfdff62384e64b",
    ),
    ArtifactFile(
        "model.safetensors.index.json",
        265_016,
        "70cbf70324c1b8d7d7c3d5f7f522f38d0be6cba673e90430a0f9a1d8b1ab8cff",
    ),
    ArtifactFile(
        "rapid-dspark-manifest.json",
        1_261,
        "5b50b23fa1d30f445eab71de601075a3dab1f4463b5a4930c1d7ac4451f70b9e",
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
    """Resolve only the 4.62 GB data subset needed by the owned runtime."""
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
