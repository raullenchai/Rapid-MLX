"""Offline inspection of explicitly selected, user-owned chat checkpoints."""

from __future__ import annotations

import os

MAX_LOCAL_CHECKPOINTS = 32


def inspect_local_checkpoints(paths: list[str]) -> list[dict]:
    """Certify exact directories without loading weights or checkpoint code.

    Registration grants read access to that directory, not its parent. Existing
    discovered checkpoints retain their already established read boundaries and
    identity. No inspection failure is converted to a Hub lookup or download.
    """
    from rapid_mlx._download_gate import _snapshot_is_complete
    from rapid_mlx.byom.preflight import physical_ram_bytes, supported_model_types
    from rapid_mlx.local_models import (
        _checkpoint_dir,
        _judge_checkpoint,
        discover_runnable_chat_models,
    )
    from rapid_mlx.model_aliases import _external_model_tree_is_contained

    if not 1 <= len(paths) <= MAX_LOCAL_CHECKPOINTS:
        raise ValueError("inspect-local accepts between 1 and 32 directories")

    # This is the same offline inventory used by the existing cached catalog.
    # Match exact canonical checkpoints, never repo names, basenames or parents.
    known: dict[str, str] = {}
    environment_error: str | None = "runtime_inventory_unavailable"
    supported = None
    ram = None
    try:
        supported = supported_model_types()
        if supported is not None:
            environment_error = "memory_unavailable"
            ram = physical_ram_bytes()
            if type(ram) is int and ram > 0:
                environment_error = "inventory_unavailable"
                for model in discover_runnable_chat_models(ram_bytes=ram, strict=True):
                    directory = _checkpoint_dir(model.repo, model.external)
                    if directory:
                        known.setdefault(os.path.realpath(directory), model.model)
                environment_error = None
    except Exception:
        # A broken environment is not an empty, authoritative inventory.
        pass
    results = []
    for requested in paths:
        row: dict = {"requested_path": requested, "path": None, "ready": False}
        results.append(row)
        if (
            not os.path.isabs(requested)
            or any(
                ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF
                for char in requested
            )
            or len(requested.encode("utf-8")) > 4096
        ):
            row["reason"] = "invalid_path"
            continue
        try:
            directory = os.path.realpath(requested)
            row["path"] = directory
            if not os.path.isdir(directory):
                row["reason"] = "missing_directory"
                continue
            if environment_error:
                row["reason"] = environment_error
                continue
            # A mistaken collection/home-folder selection must not trigger a
            # recursive tree audit. This names-only prefilter reads no linked
            # metadata; the shared completeness gate below remains authoritative.
            with os.scandir(directory) as entries:
                looks_like_checkpoint = any(
                    entry.name == "model.safetensors.index.json"
                    or (
                        entry.name.startswith("model")
                        and entry.name.endswith(".safetensors")
                    )
                    for entry in entries
                )
            if not looks_like_checkpoint:
                row["reason"] = "incomplete_checkpoint"
                continue
            if directory not in known and not _external_model_tree_is_contained(
                directory, [directory]
            ):
                row["reason"] = "unsafe_symlink"
                continue
            if not _snapshot_is_complete(directory):
                row["reason"] = "incomplete_checkpoint"
                continue
            reasons: list[str] = []
            candidate = _judge_checkpoint(
                directory,
                None,
                external=True,
                mtime=os.path.getmtime(directory),
                supported=supported,
                ram_bytes=ram,
                checkpoint_directory=directory,
                rejection=reasons,
            )
            if candidate is None:
                row["reason"] = reasons[0] if reasons else "inspection_failed"
                continue
            row.update(
                ready=True,
                launch_path=directory,
                known_model=known.get(directory),
                size_bytes=candidate.size_bytes,
                fit=candidate.fit,
                mlx_native=candidate.mlx_native,
                modified_epoch=candidate.mtime,
            )
        except Exception:
            row["reason"] = "inspection_failed"
    return results
