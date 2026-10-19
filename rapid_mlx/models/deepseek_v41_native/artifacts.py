"""Immutable artifact contract for the qualified V4.1/DSpark product lane."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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


@dataclass(frozen=True)
class ProductArtifact:
    alias: str
    target_repo: str
    target_revision: str | None
    mtp_repo: str
    mtp_revision: str | None
    mtp_files: tuple[ArtifactFile, ...] | None
    draft_subfolder: str = ""
    target_files: tuple[ArtifactFile, ...] | None = None
    min_memory_gb: float = MIN_UNIFIED_MEMORY_GB


REAP = ProductArtifact(
    "deepseek-v41-flash-reap-2bit",
    TARGET_REPO,
    TARGET_REVISION,
    MTP_REPO,
    MTP_REVISION,
    MTP_FILES,
)
# UPLOAD OWNER: replace BOTH revision placeholders with the uploaded commit
# and populate complete target/draft size + SHA-256 manifests. Never use main.
FLASH_1017 = ProductArtifact(
    alias="deepseek-v41-flash-1017",
    target_repo="rapid-mlx/DeepSeek-V4.1-Flash-1017-MLX",
    target_revision=None,  # TODO(upload): immutable 40-character commit
    mtp_repo="rapid-mlx/DeepSeek-V4.1-Flash-1017-MLX",
    mtp_revision=None,  # TODO(upload): same commit as target
    mtp_files=None,  # TODO(upload): config, index, model-mtp.safetensors
    target_files=None,  # TODO(upload): metadata and ALL indexed target shards
    draft_subfolder="dspark",
)
PRODUCT_ARTIFACTS = {item.alias: item for item in (REAP, FLASH_1017)}
LOCAL_MODEL_DIR_ENV = "RAPID_MLX_V41_MODEL_DIR"


def product_artifact(model_name: str | None) -> ProductArtifact | None:
    return next(
        (
            item
            for item in PRODUCT_ARTIFACTS.values()
            if model_name in (item.alias, item.target_repo)
        ),
        None,
    )


def require_pins(artifact: ProductArtifact) -> None:
    """Reject unpublished/incomplete contracts before any Hub or disk probe."""
    if not artifact.draft_subfolder:
        return  # The original released contract is unchanged.
    revisions = (artifact.target_revision, artifact.mtp_revision)
    manifests = (artifact.target_files, artifact.mtp_files)
    valid = (
        all(re.fullmatch(r"[0-9a-f]{40}", rev or "") for rev in revisions)
        and revisions[0] == revisions[1]
        and all(manifests)
    )
    for files in manifests:
        for item in files or ():
            valid = valid and (
                item.size > 0
                and bool(re.fullmatch(r"[0-9a-f]{64}", item.sha256))
                and Path(item.name).name == item.name
                and item.name not in (".", "..")
            )
    target_names = {item.name for item in artifact.target_files or ()}
    draft_names = {item.name for item in artifact.mtp_files or ()}
    if (
        not valid
        or not {
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "engram_token_map.json",
        }
        <= target_names
        or not {
            "config.json",
            "model.safetensors.index.json",
            "model-mtp.safetensors",
        }
        <= draft_names
    ):
        raise RuntimeError(
            f"{artifact.alias}: upload revision/file pins are unset or invalid"
        )


def local_product_path(artifact: ProductArtifact | None) -> Path | None:
    """Explicit qualification override; never turns an unknown model into V4.1."""
    value = os.environ.get(LOCAL_MODEL_DIR_ENV)
    if value is None:
        return None
    if artifact is None or not artifact.draft_subfolder:
        raise ValueError(f"{LOCAL_MODEL_DIR_ENV} requires deepseek-v41-flash-1017")
    root = Path(value).expanduser().resolve()
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "engram_token_map.json",
        "dspark/config.json",
        "dspark/model.safetensors.index.json",
        "dspark/model-mtp.safetensors",
    )
    if not value.strip() or not all((root / name).is_file() for name in required):
        raise ValueError(
            f"{LOCAL_MODEL_DIR_ENV}: incomplete local target/dspark build: {root}"
        )
    logging.getLogger(__name__).warning(
        "UNPINNED LOCAL QUALIFICATION: %s=%s; bypassing downloads and revision/hash "
        "verification. Memory and greedy/text/input limits still apply.",
        LOCAL_MODEL_DIR_ENV,
        root,
    )
    return root


def resolve_product_paths(
    artifact: ProductArtifact, *, allow_local: bool = True
) -> tuple[Path, Path]:
    if not allow_local and LOCAL_MODEL_DIR_ENV in os.environ:
        raise ValueError("local qualification override is only supported by serve")
    local = local_product_path(artifact) if allow_local else None
    if local is not None:
        return local, local / artifact.draft_subfolder
    require_pins(artifact)
    if not artifact.draft_subfolder:
        return download_target_snapshot(), download_mtp_snapshot()
    from ..._mirror import pinned_snapshot_download

    assert artifact.target_revision is not None
    assert artifact.target_files is not None and artifact.mtp_files is not None
    logging.getLogger(__name__).info(
        "Resolving pinned V4.1 target + co-located draft: %s @ %s (%.2f GB pinned files)",
        artifact.target_repo,
        artifact.target_revision,
        sum(item.size for item in (*artifact.target_files, *artifact.mtp_files)) / 1e9,
    )
    target = Path(
        pinned_snapshot_download(artifact.target_repo, artifact.target_revision)
    )
    draft = target / artifact.draft_subfolder
    for root, files in ((target, artifact.target_files), (draft, artifact.mtp_files)):
        assert files is not None
        names = {item.name for item in files}
        index = json.loads((root / "model.safetensors.index.json").read_text())
        if not set(index["weight_map"].values()) <= names:
            raise ValueError("indexed shards are missing from the pinned file manifest")
        _verify_files(root, files)
    return target, draft


def is_product_target(model_name: str | None) -> bool:
    return product_artifact(model_name) is not None


def mtp_model_identity_digest(artifact: ProductArtifact = REAP) -> str:
    """Stable identity digest for the exact DSpark sidecar byte set.

    Community Benchmark's execution contract requires an external speculative
    model identity.  The product runtime already verifies every permitted MTP
    file by size and SHA-256, so this digest binds the same pinned revision and
    allow-list without reading or downloading the files.
    """
    from rapid_mlx.catalog import rcj_digest

    require_pins(artifact)
    files = artifact.mtp_files or ()

    components = [
        {
            "component_id": "primary",
            "role": "primary",
            "source": {
                "kind": "huggingface",
                "repo_id": artifact.mtp_repo,
                "resolved_revision": artifact.mtp_revision,
                **(
                    {"subfolder": artifact.draft_subfolder}
                    if artifact.draft_subfolder
                    else {}
                ),
            },
            "artifact": {
                "format": "mlx-safetensors",
                "manifest_digest": rcj_digest(
                    [
                        {"name": item.name, "size": item.size, "sha256": item.sha256}
                        for item in files
                    ]
                ),
                "manifest_basis": "content_sha256",
                "total_size_bytes": sum(item.size for item in files),
            },
            "quantization": {"kind": "unknown", "base_dtype": "unknown"},
        }
    ]
    projection = {
        "schema_version": 1,
        "pipeline_kind": "text_generation",
        "components": components,
    }
    return rcj_digest(projection)


def require_product_memory(artifact: ProductArtifact = REAP) -> float:
    """Fail closed for programmatic callers that bypass the CLI catalog gate."""
    import psutil

    # Keep this lightweight artifact gate importable on non-MLX hosts (notably
    # Linux CI). Importing the general hardware module eagerly imports MLX even
    # though this check only needs physical memory.
    total = float(psutil.virtual_memory().total) / (1024**3)
    if total < artifact.min_memory_gb:
        raise RuntimeError(
            "DeepSeek V4.1 DSpark K4 requires at least "
            f"{artifact.min_memory_gb:.0f} GiB unified memory; this Mac reports "
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
    return _verify_files(snapshot, MTP_FILES)


def _verify_files(snapshot: str | Path, files: tuple[ArtifactFile, ...]) -> Path:
    root = Path(snapshot).resolve()
    if root.name == "dspark":
        root = root.parent
    blobs = (
        (root.parents[1] / "blobs").resolve()
        if root.parent.name == "snapshots"
        else None
    )
    for expected in files:
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
    from ..._mirror import pinned_snapshot_download

    path = pinned_snapshot_download(
        MTP_REPO,
        MTP_REVISION,
        allow_patterns=list(MTP_ALLOW_PATTERNS),
    )
    return verify_mtp_snapshot(path)


def download_target_snapshot() -> Path:
    """Resolve the exact target checkpoint revision; never follow moving main."""
    from ..._mirror import pinned_snapshot_download

    return Path(pinned_snapshot_download(TARGET_REPO, TARGET_REVISION))
