# SPDX-License-Identifier: Apache-2.0
"""Immutable repository identities for shipped catalogs and runtime assets."""

import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def locked_repositories() -> dict:
    with Path(__file__).with_name("model_catalog_lock.json").open() as handle:
        lock = json.load(handle)
    if lock.get("schema_version") != 1 or lock.get("errors"):
        raise ValueError("Model catalog identity lock is incomplete")
    repositories = lock["repositories"]
    for repo, record in repositories.items():
        if not re.fullmatch(r"[0-9a-f]{40}", record.get("revision", "")):
            raise ValueError(f"Invalid locked model revision for {repo}")
    return repositories


def resolve_repository_identity(model_name: str) -> tuple[str, str | None]:
    """Resolve text/audio aliases and nested assets without following mutable refs."""
    from .model_aliases import local_dir_shadows_alias

    if Path(model_name).expanduser().is_dir() and not local_dir_shadows_alias(model_name):
        return model_name, None
    from .model_aliases import resolve_hub_model, resolve_revision

    repo = resolve_hub_model(model_name)
    pin = resolve_revision(model_name)
    if pin is not None:
        return repo, pin
    from .audio.registry import resolve_audio_alias

    audio = resolve_audio_alias(model_name)
    if audio is not None:
        repo, pin = audio.hf_id, audio.revision
    for canonical, record in locked_repositories().items():
        if canonical.lower() == repo.lower():
            locked = record["revision"]
            if pin is not None and pin != locked:
                raise ValueError(f"Audio catalog and identity lock disagree for {repo}")
            return canonical, locked
    return repo, pin
