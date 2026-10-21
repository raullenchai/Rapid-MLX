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
        314_694,
        "541dfd95c0a1eeb2d3b38b129912d8858ccae112af4959d17797e831a41a7843",
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
        4_926_799_340,
        "a5a9cb9c5384bedefdee5e62e4836511b5c21dbf0e56d144d8635b88d886e249",
    ),
    ArtifactFile(
        "model-layer-1.safetensors",
        5_093_998_580,
        "484e538ab4da9d063f1decd67ce88c87eb5a2cf7fe388e00061bbf8a6d0ce383",
    ),
    ArtifactFile(
        "model-layer-10.safetensors",
        4_926_799_388,
        "76b2e57d444ea569e4e4ba0071cac6b4e719c3ec62e385797ece27aa9a295e57",
    ),
    ArtifactFile(
        "model-layer-11.safetensors",
        4_926_799_388,
        "403028e6497199898830fb6aa84b5a3b86be4711c75d0a28f3de6021bb19f17a",
    ),
    ArtifactFile(
        "model-layer-12.safetensors",
        4_926_799_388,
        "b29a05d47f5e36e0047cd0974817e8239352bec06b0595ff8c19b5a6a0e62369",
    ),
    ArtifactFile(
        "model-layer-13.safetensors",
        4_926_799_388,
        "f10609b8cfeafe20a75e340fbc39b23cb8ec4452394a3995c6e3690a323e1137",
    ),
    ArtifactFile(
        "model-layer-14.safetensors",
        5_169_611_868,
        "5949fe5878e5d49713ed82e3a2f0eb500bda6a85f421da2dee00b7abb4aa9e5b",
    ),
    ArtifactFile(
        "model-layer-15.safetensors",
        6_123_486_772,
        "c3770ea352ecf209eeec4b4a8d1c853c637b076d46b9c7dbb1907d87e0763c07",
    ),
    ArtifactFile(
        "model-layer-16.safetensors",
        4_991_024_692,
        "1c3599cf319a0291650c3724a926b2183c71b92a5fd25949c01e3599a4838d87",
    ),
    ArtifactFile(
        "model-layer-17.safetensors",
        4_991_024_692,
        "addeec4ed54bee018bf990cb086ab828cb7efe18a39d06f84d11bee93a14413a",
    ),
    ArtifactFile(
        "model-layer-18.safetensors",
        4_991_024_692,
        "03f34116a79e64da7e0eda54efdeabf6a02b5c5b2e4eac8657d5f39a73f366dd",
    ),
    ArtifactFile(
        "model-layer-19.safetensors",
        4_926_799_388,
        "01a7f6f39b9554e92f2ae995fd224de6c0f5aac2c3a5420a4826ef16e43ee365",
    ),
    ArtifactFile(
        "model-layer-2.safetensors",
        4_938_187_268,
        "cf2d8f3993dff016767c645fb7ce161bac5fbc3e9df72b659773cc20b7763cd5",
    ),
    ArtifactFile(
        "model-layer-20.safetensors",
        4_935_401_716,
        "d89dfe82eb28a31976f5c5b9e221f45f3dc0c48dab0393ceaa4798f376acfefb",
    ),
    ArtifactFile(
        "model-layer-21.safetensors",
        4_926_799_388,
        "bb3303ff6bfbbb384f00382f37e21a2f907deddc2b8893af9a47496d085e5e1e",
    ),
    ArtifactFile(
        "model-layer-22.safetensors",
        4_926_799_388,
        "b7b430520373fc45539e043b6efa23e4d7bbf05ec5060d92c73eccd69573941b",
    ),
    ArtifactFile(
        "model-layer-23.safetensors",
        4_926_799_388,
        "1150d2f6b52bae40a4ba9ae8534303dbdd9a05639aff08ebfacf8cba89279693",
    ),
    ArtifactFile(
        "model-layer-24.safetensors",
        4_932_544_692,
        "d8b335512f76c64ac8532e0c48af3c048323f100eec3a016447f56743453a356",
    ),
    ArtifactFile(
        "model-layer-25.safetensors",
        4_926_799_388,
        "13ddf68f7015592fec2e452e4e4f7f018c9e132a7815f40f74a9e879a87446d8",
    ),
    ArtifactFile(
        "model-layer-26.safetensors",
        4_991_024_692,
        "8d06462f4e3136def0f659357598355ecdaf65ddfae6f61643b8a067d1b419dc",
    ),
    ArtifactFile(
        "model-layer-27.safetensors",
        4_926_799_388,
        "8dd625c354c72bf611c6c913ea3e90a0b98f3197c2374a28aeead013bbe5c4cc",
    ),
    ArtifactFile(
        "model-layer-28.safetensors",
        4_932_544_692,
        "08123df0cd49906e6effeb578a0c98aeffade9271c22b9a041b54127e59e72d3",
    ),
    ArtifactFile(
        "model-layer-29.safetensors",
        4_926_799_388,
        "84ee67724d32d9af56853cd42a7ed75e5b9a54f62236ccc275e92c5eb126a3ab",
    ),
    ArtifactFile(
        "model-layer-3.safetensors",
        4_926_799_340,
        "d71cb697c06defcda7f3f77a57d7edd5b47b4e46a6345d0fecf1458431511bb1",
    ),
    ArtifactFile(
        "model-layer-30.safetensors",
        4_926_799_388,
        "2a02bc43f0f8a3ddfcc49efa435d87ec9c7310e1115a01ad81054518e912b466",
    ),
    ArtifactFile(
        "model-layer-31.safetensors",
        4_926_799_388,
        "f47df6ce1e321c1a608a429552952895935dabe72d94ec9aa8f27ddb3c451ac9",
    ),
    ArtifactFile(
        "model-layer-32.safetensors",
        4_932_544_692,
        "321b23d4cdd6b08d2d78df1500ac5b98f9c54b645d3d77b1615832e780b53942",
    ),
    ArtifactFile(
        "model-layer-33.safetensors",
        4_926_799_388,
        "aefdddf19e181b43beda06ccd48af858a0776736312b5005e08a9948287f1cec",
    ),
    ArtifactFile(
        "model-layer-34.safetensors",
        4_926_799_388,
        "ad194b487a1fd89a05a82f4950cb86c7ce48ac0bfaf3ac43b5e29e51e44f45a4",
    ),
    ArtifactFile(
        "model-layer-35.safetensors",
        6_059_261_468,
        "7cb333066f357f177549d4f78cdd7a6efa96c7d1d82144262f34a8c8e06b82b3",
    ),
    ArtifactFile(
        "model-layer-36.safetensors",
        6_065_006_772,
        "3a2b3c3525388fd6d81962171c01c98a2ec620e0f600cf661f0ff1c7c10d06a7",
    ),
    ArtifactFile(
        "model-layer-37.safetensors",
        6_059_261_468,
        "5d176699e7d62ab16b4781a10ea5dba75c322a688a8abe2e9cdf1882347c7376",
    ),
    ArtifactFile(
        "model-layer-38.safetensors",
        6_059_261_468,
        "63ba71745be94a05e3c334f9bfb1814ad54410ded980dc379bb4b6cf28fae816",
    ),
    ArtifactFile(
        "model-layer-39.safetensors",
        6_123_486_772,
        "4d606bf7d5c1cf4cce2cae1247987c8236fd1008b450d8ddab1e30187705d38a",
    ),
    ArtifactFile(
        "model-layer-4.safetensors",
        4_926_799_340,
        "647ad03ab908e91e205283e659388c6bd5f20a88398b0d5694eb5088ab776529",
    ),
    ArtifactFile(
        "model-layer-5.safetensors",
        4_926_799_340,
        "04eb0c27399fedc44b3d55b2250e5ca0663c41257b636be7e1899b3291585b03",
    ),
    ArtifactFile(
        "model-layer-6.safetensors",
        4_991_024_644,
        "acae753a197c24952db354aa67b419db5337c0c2d0bab23775b4b6ca74de9633",
    ),
    ArtifactFile(
        "model-layer-7.safetensors",
        4_926_799_340,
        "9b33f536d4d6868190abc09887ed9fd051e8a0b669f0498d63abeb10ab46abee",
    ),
    ArtifactFile(
        "model-layer-8.safetensors",
        4_938_187_268,
        "a9f1d39e9661a51a7335ab6d68d063f294d14221dfda9cfca6f47581b6c2c6c4",
    ),
    ArtifactFile(
        "model-layer-9.safetensors",
        4_926_799_340,
        "319c7b8555f5abb4277d4fec2891b6dd84bb47f462e92b8392234aaa8501df00",
    ),
    ArtifactFile(
        "model-top.safetensors",
        2_647_664_944,
        "d2bd2ca2c87e2ee591b739f92b8cadf37ce302ffadf55444122ef21d3541ee2d",
    ),
    ArtifactFile(
        "model.safetensors.index.json",
        134_294,
        "a75907861bb5cdb2e8a36cff287a1fa0c267cbe7e15a331848dac0e1fd6d9af3",
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
        378_261,
        "baeffad53b1f5722b0fb0b860aef4fd6e569df9aefe2e2910f53785ec79e8ea2",
    ),
    ArtifactFile(
        "model-mtp.safetensors",
        8_386_765_500,
        "47848a39e226cbad590922e28c9764c3aa4484da77f3dded34bab685691531be",
    ),
    ArtifactFile(
        "model.safetensors.index.json",
        225_324,
        "5571f22a8cc2826b5b9a167cfdd04a660226377046c6733b81cfe43f3ae54fc7",
    ),
)
FLASH_1017_REVISION = "c521050aba97719393865adf611ad745c7f4f551"
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
