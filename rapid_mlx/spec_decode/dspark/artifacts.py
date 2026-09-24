# SPDX-License-Identifier: Apache-2.0
"""Revision-pinned artifact resolution for companion DSpark."""

from __future__ import annotations

from dataclasses import dataclass

from .eligibility import CompanionDSparkPair, validate_companion_artifacts


@dataclass(frozen=True)
class CompanionDSparkArtifacts:
    target_path: str
    drafter_path: str


def download_companion_artifacts(
    pair: CompanionDSparkPair,
) -> CompanionDSparkArtifacts:
    """Resolve both qualified snapshots at immutable Hub revisions."""

    from huggingface_hub import snapshot_download

    target_path = snapshot_download(pair.target_repo, revision=pair.target_revision)
    drafter_path = snapshot_download(pair.drafter_repo, revision=pair.drafter_revision)
    validate_companion_artifacts(
        pair, target_path=target_path, drafter_path=drafter_path
    )
    return CompanionDSparkArtifacts(
        target_path=str(target_path), drafter_path=str(drafter_path)
    )


__all__ = ["CompanionDSparkArtifacts", "download_companion_artifacts"]
