# SPDX-License-Identifier: Apache-2.0
"""Non-destructive merges for agent configs that share files with the user.

Two config shapes we write are *shared* with the user's own entries:

* pi's ``models.json`` keeps every provider in one file, and a provider's
  ``models`` is a LIST of ``{"id": ...}`` entries. Replacing that list
  wholesale (the generic deep-merge rule for lists) deletes every model the
  user registered under the same provider.
* dsh's ``cordis.patch.yml`` is a LIST of ``{id, config}`` patch layers, and
  the ``llm-pi-ai`` layer is the plugin-level provider registry — it holds the
  user's other providers too. Replacing a same-id layer wholesale deletes
  them.

Both are fixed the same way: mappings merge recursively, and an id-keyed
``models`` list merges entry-by-entry on ``id`` (our entry is deep-merged over
the same-id entry in place; new ids are appended; nothing else is touched).
"""

from __future__ import annotations

from typing import Any


def is_id_list(value: Any) -> bool:
    """True for a non-empty list whose every entry is a mapping with an ``id``."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and "id" in item for item in value)
    )


def merge_by_id(existing: list[Any], incoming: list[Any]) -> list[Any]:
    """Merge *incoming* id-keyed entries into *existing*, keeping everything else.

    A same-id entry is deep-merged in place (our keys win, the user's extra
    keys survive); an unseen id is appended. Entries without an ``id`` and
    entries whose id we do not write are left exactly where they were.
    """
    merged = list(existing)
    for entry in incoming:  # callers pass an ``is_id_list`` here
        matched = False
        for index, current in enumerate(merged):
            if isinstance(current, dict) and current.get("id") == entry["id"]:
                merged[index] = deep_merge(current, entry)
                matched = True
        if not matched:
            merged.append(entry)
    return merged


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive mapping merge where an id-keyed ``models`` list merges by id.

    Every other list and every scalar in *override* wins, matching the generic
    adapter merge. Neither input is mutated.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        elif key == "models" and is_id_list(value) and isinstance(current, list):
            merged[key] = merge_by_id(current, value)
        else:
            merged[key] = value
    return merged


def merge_patch_layers(
    existing: list[Any], incoming: list[dict[str, Any]]
) -> list[Any]:
    """Merge Cordis patch layers (``[{id, config}, ...]``) without data loss.

    A layer whose id we write is merged in place: its ``config`` mapping is
    deep-merged with ours (so the user's other providers inside ``llm-pi-ai``
    survive) and any other keys on the entry are kept. A same-id layer whose
    ``config`` is not a mapping cannot be merged and takes our config. Layers
    we do not write — and entries without an id — are left untouched, in
    place. Ids we write that the file lacks are appended.
    """
    merged = list(existing)
    for layer in incoming:
        layer_id = layer["id"]  # every layer we write carries an id
        matched = False
        for index, current in enumerate(merged):
            if not (isinstance(current, dict) and current.get("id") == layer_id):
                continue
            matched = True
            updated = {**current, **layer}
            if isinstance(current.get("config"), dict) and isinstance(
                layer.get("config"), dict
            ):
                updated["config"] = deep_merge(current["config"], layer["config"])
            merged[index] = updated
        if not matched:
            merged.append(layer)
    return merged
