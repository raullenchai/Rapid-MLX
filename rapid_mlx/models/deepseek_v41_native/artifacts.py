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
FLASH_1017_TARGET_FILES = (
    ArtifactFile(
        "config.json",
        419_339,
        "658f6e64d32bc9fa5b3d1fb6ef230ffd8d7483f856623258b034c369cb2bc845",
    ),
    ArtifactFile(
        "engram_token_map.json",
        874_691,
        "de40bcdd71f24430ba82752ef385093be962adbf845a1c97db39161597d5dc28",
    ),
    ArtifactFile(
        "model-engram-1-affine.safetensors",
        12_288_197_632,
        "ebccac5290a467126ceda51935b4fb0789fa1ac1bde5b774bece3568b578719e",
    ),
    ArtifactFile(
        "model-engram-1-weight.safetensors",
        49_152_789_648,
        "e5fc6037e69be764548919bf984bf3c794ff9e31286945dd3e97bf9b5e3cfb7d",
    ),
    ArtifactFile(
        "model-engram-14-affine.safetensors",
        12_288_534_080,
        "2b4ca3e721879356ce3d14158b934230d564474d3462ef609c30ee537b875fbc",
    ),
    ArtifactFile(
        "model-engram-14-weight.safetensors",
        49_154_135_440,
        "1e837d260d9b1678bdbcea07bfdc48ebfa3ea92b3c33a4b1df5ab96e5f740c64",
    ),
    ArtifactFile(
        "model-layer-0.safetensors",
        4_424_793_580,
        "119d6b0f4fdc2e1834db55a6d38b6499b7fc257efec81b674459c46b14e63905",
    ),
    ArtifactFile(
        "model-layer-1.safetensors",
        4_591_992_804,
        "4598a22019900fdbdb26a19707a71775cb9ffc475a2a4619bf2e230d40ae3347",
    ),
    ArtifactFile(
        "model-layer-10.safetensors",
        4_424_793_628,
        "5f746a18f056d8c5f51debf8bc38368e0ccc46e1f8ec9b8ea3657ddbc8efec8d",
    ),
    ArtifactFile(
        "model-layer-11.safetensors",
        4_424_793_628,
        "2b3346b0af2a4e7579c04c4561d5265252bd1754002e0ad71d5efa736fc13f26",
    ),
    ArtifactFile(
        "model-layer-12.safetensors",
        4_424_793_628,
        "6fbd849b97a92ee9d219c9f3fd68f3d58d05f2b3c7ab99fb42319a7bb35ef333",
    ),
    ArtifactFile(
        "model-layer-13.safetensors",
        4_424_793_628,
        "7bd783d9415787f46a729fd135d7fe7dc6ddc7299e456b4e00e4d8a23aa28f2b",
    ),
    ArtifactFile(
        "model-layer-14.safetensors",
        4_603_380_796,
        "31999d27c047ccaebb399ae7894f190b41d89cb588cb530234206c0be2d3c358",
    ),
    ArtifactFile(
        "model-layer-15.safetensors",
        6_123_486_748,
        "dd89752ce7711a33a4316c2dac698fc03334e85c19a57d349eb2b5c50e53c4da",
    ),
    ArtifactFile(
        "model-layer-16.safetensors",
        4_424_793_628,
        "740310a619bf33e6a4e94f8afc512fe98b89ed5f10e8ba5da8011c4d0630a396",
    ),
    ArtifactFile(
        "model-layer-17.safetensors",
        4_424_793_628,
        "bfd96bd307b611a4a38bc529f3ee401434187f24c4f86ba9986d18a7fce7808f",
    ),
    ArtifactFile(
        "model-layer-18.safetensors",
        4_424_793_628,
        "a3374bdb64a623a856695e10aa1a74c44d6c6ba0a5deda7e6b22be1d68bbee52",
    ),
    ArtifactFile(
        "model-layer-19.safetensors",
        4_424_793_628,
        "f775dc7a7fb5ecc37645370082b8d8e0c80b7b12b02a10029986782fe75bcf9e",
    ),
    ArtifactFile(
        "model-layer-2.safetensors",
        4_436_181_508,
        "7960a0b050d94ad9e55c2e25183e752ce5ab9b888fd66261e9ec7b421d0c91a7",
    ),
    ArtifactFile(
        "model-layer-20.safetensors",
        4_999_626_988,
        "6606da1761de4e8bfd4a65cef995fc5c7e6da6d1e74395bcaa0041bebcce1a7e",
    ),
    ArtifactFile(
        "model-layer-21.safetensors",
        4_991_024_668,
        "b3980fca697ca6a4092136efe7d87d59178ea0c1d0bd1404ba4b2de85989db92",
    ),
    ArtifactFile(
        "model-layer-22.safetensors",
        4_991_024_668,
        "6546300a03b74f72c525550f74b67e09288c2ac999f35208b1ee9843bdcaa664",
    ),
    ArtifactFile(
        "model-layer-23.safetensors",
        4_991_024_668,
        "cd8017cf95247a5e7c3f356877ef02847979d10af3fb563237445d615526861a",
    ),
    ArtifactFile(
        "model-layer-24.safetensors",
        4_996_769_964,
        "893e167b85d6217a98609e44677632b33bcdd0c744b4dcc3fb3578c5ca8984e9",
    ),
    ArtifactFile(
        "model-layer-25.safetensors",
        4_991_024_668,
        "9cde07f2d219d20908bf219f2cd6de858a15d694ac28edc36b329b8f0ea7adc4",
    ),
    ArtifactFile(
        "model-layer-26.safetensors",
        4_991_024_668,
        "340e9242387bc83b35f6236cfc6fbc2c08feb876230ed038aea493cd150c2628",
    ),
    ArtifactFile(
        "model-layer-27.safetensors",
        4_991_024_668,
        "b3aa497e5d24ad6406a18d86c10cf6d1203b1027bc55b2c38a18c25ab7113cf3",
    ),
    ArtifactFile(
        "model-layer-28.safetensors",
        4_996_769_964,
        "8954e1651028aadfcab8e757a993e28780b474ded102b2e8af765fcc289af9b3",
    ),
    ArtifactFile(
        "model-layer-29.safetensors",
        4_991_024_668,
        "8c138b9c8198efac85425f6027ac566228f9397adfc5f10d0a9b8238c0cc5d85",
    ),
    ArtifactFile(
        "model-layer-3.safetensors",
        4_424_793_580,
        "db4d51882c66bdbb3f3d40d421f8aaca865b4f7cac60ca586c9a4dd21b0597bb",
    ),
    ArtifactFile(
        "model-layer-30.safetensors",
        4_991_024_668,
        "c3ef03a3f16828e5e71ed5f04210c1b8ad3355682bd5c3cf667bc3255bb95ceb",
    ),
    ArtifactFile(
        "model-layer-31.safetensors",
        4_991_024_668,
        "7da8dcde32bfacdf4f6f883a5b0e5c3a45bad7f77961a27d0fdd7ad8d45997af",
    ),
    ArtifactFile(
        "model-layer-32.safetensors",
        4_430_538_924,
        "31dd2a5aa06291fa4cffd3208a9cdc5e6b6194188455673a471445296495d0c6",
    ),
    ArtifactFile(
        "model-layer-33.safetensors",
        4_424_793_628,
        "117e9751f49a53c01bba21fffcb5e76d627653f1711a9d482f64e43b37c9e41c",
    ),
    ArtifactFile(
        "model-layer-34.safetensors",
        4_424_793_628,
        "d0ce172568fc97fefe89aed194ba32ceb5a06f73ecfd8dc8755ba105e101fef9",
    ),
    ArtifactFile(
        "model-layer-35.safetensors",
        6_123_486_748,
        "5d79f588a8d3e13d39ccb8aade49555dfc63b317e403676753b8c2f8e3c7f9ef",
    ),
    ArtifactFile(
        "model-layer-36.safetensors",
        6_129_232_044,
        "ffa2e56934455d5d40186853350dde679a100fbef962f223f6f8461db71026b2",
    ),
    ArtifactFile(
        "model-layer-37.safetensors",
        6_123_486_748,
        "2efcdc0fc2b9f7fe1e3db84f33f097ab0a54a2522a1983276b62c7d069f02c7d",
    ),
    ArtifactFile(
        "model-layer-38.safetensors",
        6_123_486_748,
        "9731d95d89cb125811c225fb5ec85e285c526f1c9bb6f0706df11e2fcee625fd",
    ),
    ArtifactFile(
        "model-layer-39.safetensors",
        7_255_948_828,
        "6c460994353a3b5372a8eeff9711184534f1292c9ae2a8d656d4250b47bf0cad",
    ),
    ArtifactFile(
        "model-layer-4.safetensors",
        4_424_793_580,
        "d62205f8d41a54216a6654a101c50c3cd970a06fccc6a146e6952c3ab8e8250e",
    ),
    ArtifactFile(
        "model-layer-5.safetensors",
        4_424_793_580,
        "0bc6279d26a2d9a9eea0072255905066047673c4c82a406215b760b9851b4855",
    ),
    ArtifactFile(
        "model-layer-6.safetensors",
        4_424_793_580,
        "2af545bca3a4ca0627fae7b2672917f3e6856fa0d6d4321011d040ca028653f9",
    ),
    ArtifactFile(
        "model-layer-7.safetensors",
        4_424_793_580,
        "040bfae99f90dc165c7b380e7fa6eb2d9c5592d0034fc97be22694d1eda80336",
    ),
    ArtifactFile(
        "model-layer-8.safetensors",
        4_436_181_508,
        "c31705de0fa2cb09b8af25bce3e04dbdc6c1e2c98a23ee56b24617e36eb440ab",
    ),
    ArtifactFile(
        "model-layer-9.safetensors",
        4_424_793_580,
        "d2c75e51e956bfc8201993cadd952896da4b0e574200a8e881d5c943feba31f3",
    ),
    ArtifactFile(
        "model-top.safetensors",
        2_647_664_944,
        "d2bd2ca2c87e2ee591b739f92b8cadf37ce302ffadf55444122ef21d3541ee2d",
    ),
    ArtifactFile(
        "model.safetensors.index.json",
        134_294,
        "65fbf52d2214751ffbf7394cea4db712135753c91e73f78cdfe8f80d140f6e1f",
    ),
    ArtifactFile(
        "tokenizer.json",
        6_367_257,
        "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    ),
    ArtifactFile(
        "tokenizer_config.json",
        801,
        "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    ),
)
FLASH_1017_DRAFT_FILES = (
    ArtifactFile(
        "config.json",
        110_204,
        "89a53611c4d9a420bcac142a21f1dc458855e58fe7d33f4e31c383f8c7ba7156",
    ),
    ArtifactFile(
        "model-mtp.safetensors",
        8_386_765_492,
        "bcd27c69db4b47632db2591a39657dbf01145f597e1fd22cb022318e1a87d1fe",
    ),
    ArtifactFile(
        "model.safetensors.index.json",
        225_324,
        "deae44f42d4402330cda6220b6346e5d4defd6368807ea7af263d1970bae50b6",
    ),
)
FLASH_1017_REVISION = "c95bb8ad2bf9b2bf65fcab9b9c3dbbdec24dcb39"
FLASH_1017 = ProductArtifact(
    alias="deepseek-v41-flash-1017",
    target_repo="rapid-mlx/DeepSeek-V4.1-Flash-1017-MLX",
    target_revision=FLASH_1017_REVISION,
    mtp_repo="rapid-mlx/DeepSeek-V4.1-Flash-1017-MLX",
    mtp_revision=FLASH_1017_REVISION,
    mtp_files=FLASH_1017_DRAFT_FILES,
    target_files=FLASH_1017_TARGET_FILES,
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
    if value is None or not value.strip():
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
    complete = all((root / name).is_file() for name in required)
    for directory in (root, root / "dspark") if complete else ():
        try:
            index = json.loads((directory / "model.safetensors.index.json").read_text())
            shards = set(index["weight_map"].values()) or {"<empty weight_map>"}
        except (ValueError, AttributeError, KeyError, TypeError):
            shards = {"<unreadable index>"}
        # Shard names must be plain files inside the selected build.
        complete = complete and all(
            isinstance(s, str) and Path(s).name == s and (directory / s).is_file()
            for s in shards
        )
        # The loader globs every shard, so an unindexed one would load too.
        complete = (
            complete and {p.name for p in directory.glob("*.safetensors")} == shards
        )
    if not complete:
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


# Verified resolutions, so serve reuses the CLI's prefetch without accepting
# caller-supplied paths: (target, draft, is_local_override) per alias.
_RESOLVED: dict[str, tuple[Path, Path, bool]] = {}


def take_resolved_product_paths(
    artifact: ProductArtifact,
) -> tuple[Path, Path, bool] | None:
    """Pop this process's last verified resolution of ``artifact``, if any."""
    return _RESOLVED.pop(artifact.alias, None)


def resolve_product_paths(
    artifact: ProductArtifact, *, allow_local: bool = True
) -> tuple[Path, Path]:
    if not allow_local and os.environ.get(LOCAL_MODEL_DIR_ENV, "").strip():
        raise ValueError("local qualification override is only supported by serve")
    local = (
        local_product_path(artifact)
        if allow_local and artifact.draft_subfolder
        else None
    )
    if local is not None:
        _RESOLVED[artifact.alias] = (local, local / artifact.draft_subfolder, True)
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
        # The loader reads every top-level shard, so an unlisted one could
        # silently override verified tensors.
        unpinned = sorted({p.name for p in root.glob("*.safetensors")} - names)
        if unpinned:
            raise ValueError(f"unpinned safetensors in the pinned snapshot: {unpinned}")
        _verify_files(root, files)
    _RESOLVED[artifact.alias] = (target, draft, False)
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
