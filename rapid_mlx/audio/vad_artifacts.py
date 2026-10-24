# SPDX-License-Identifier: Apache-2.0
"""Resolve the reviewed Silero VAD data before the native loader consumes it."""

from pathlib import Path

VAD_REPO = "mlx-community/silero-vad"
VAD_REVISION = "7bc17f22d3c0451bd3a6cd71e759b009271ff49a"
_VAD_FILES = ("config.json", "model.safetensors")
# mlx-audio loads every root safetensors file, or NPZ when none are present.
# Reject unexpected tensor files as well as changes to the two expected files.
_VAD_VERIFY_PATTERNS = ("config.json", "*.safetensors", "*.npz")


def resolve_vad_snapshot() -> Path:
    """Return verified local data, preserving offline and disabled-download policy.

    The packaged manifest permits verification of an existing snapshot without
    a Hub request or a previous receipt. Invalid files are retained for repair,
    but never returned to the model loader, including on a subsequent retry.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    from rapid_mlx.model_downloads import snapshot_download
    from rapid_mlx.model_integrity import verify_snapshot

    snapshot = (
        Path(HF_HUB_CACHE)
        / "models--mlx-community--silero-vad"
        / "snapshots"
        / VAD_REVISION
    )
    if not all((snapshot / name).is_file() for name in _VAD_FILES):
        snapshot = Path(
            snapshot_download(
                VAD_REPO, revision=VAD_REVISION, allow_patterns=list(_VAD_FILES)
            )
        )
    verify_snapshot(
        VAD_REPO,
        snapshot,
        revision=VAD_REVISION,
        allow_patterns=list(_VAD_VERIFY_PATTERNS),
        local_files_only=True,
    )
    return snapshot
