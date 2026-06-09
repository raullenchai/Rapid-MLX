# SPDX-License-Identifier: Apache-2.0
"""Read sampling defaults from a HuggingFace ``generation_config.json``.

Model authors ship recommended sampling parameters (``temperature``,
``top_p``, ``top_k``, ``min_p``, ``repetition_penalty``, …) inside
``generation_config.json`` next to the safetensors weights. The HF
transformers loader auto-applies these; mlx-lm's generators do not, and
neither does Rapid-MLX — until now. This module gives the server an
opinionated, well-typed view of just the sampling subset, so the request
→ CLI → alias → generation_config → fallback cascade in
``service/helpers.py`` has a real value to read for layer 3 instead of
always falling through.

We deliberately do **not** consume non-sampling keys here (eos_token_id,
pad_token_id, bos_token_id, transformers_version, do_sample, _from_model_config).
Those either belong to the engine/tokenizer plumbing or are HF-internal
provenance flags. Keeping the surface small means a future field added to
``generation_config.json`` upstream won't silently leak into a sampling
default it was never meant to drive.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Subset of HF ``generation_config.json`` keys that map onto our
# server-side sampling resolve chain. Order doesn't matter — this is a
# membership filter.
_SAMPLING_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)


def load_generation_config_sampling(model_path: str | None) -> dict[str, float | int]:
    """Return the sampling subset of ``<model_path>/generation_config.json``.

    Resolution rules:

    * Local directory: read ``<model_path>/generation_config.json`` directly.
    * HuggingFace repo id (``mlx-community/Qwen3.5-4B-MLX-4bit``): probe the
      local HF hub snapshots directory for a downloaded copy. Returns ``{}``
      if no snapshot is on disk — we don't trigger a network fetch from a
      sampling-defaults helper.
    * Missing file, unreadable JSON, or non-dict payload: return ``{}``
      (silent — this is best-effort enrichment, not a critical path).

    Values are kept as their JSON-decoded types (``float``/``int``). Any
    key whose value isn't a finite number is dropped — never crash the
    server on a stray null/string in someone's hand-edited config.
    """
    if not model_path:
        return {}

    config_path = _resolve_config_path(model_path)
    if config_path is None:
        return {}

    try:
        with open(config_path) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(
            "generation_config: skip %s (read/parse failed: %s)", config_path, exc
        )
        return {}

    if not isinstance(raw, dict):
        logger.debug("generation_config: skip %s (not a JSON object)", config_path)
        return {}

    out: dict[str, float | int] = {}
    for key in _SAMPLING_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool):
            # Defensive: JSON booleans are also ``int`` in Python. We
            # never want ``temperature=True`` to land as ``1`` here.
            continue
        if not isinstance(value, (int, float)):
            continue
        if value != value or value in (float("inf"), float("-inf")):
            # NaN / +-inf
            continue
        if key == "top_k":
            # ``top_k`` is an integer count; silently truncating a fractional
            # value (e.g. 20.5 → 20) hides a malformed config. Drop it
            # unless it parses as a whole number.
            if isinstance(value, float) and not value.is_integer():
                continue
            out[key] = int(value)
            continue
        out[key] = value

    if out:
        logger.info(
            "generation_config: loaded sampling defaults from %s: %s",
            config_path,
            out,
        )
    return out


def load_generation_config_eos_ids(model_path: str | None) -> tuple[int, ...]:
    """Return the ``eos_token_id`` list from ``<model_path>/generation_config.json``.

    Many chat-tuned models ship a primary ``<eos>`` token in
    ``tokenizer_config.json`` (e.g. id 1 for Gemma 3 / 3n) but
    declare the wider stop set — including the chat-template
    terminator like ``<end_of_turn>`` (id 106) — only in
    ``generation_config.json``'s ``eos_token_id`` array. mlx-lm's
    tokenizer wrapper exposes only the primary id, so without this
    helper the scheduler never adds the chat-template terminator to
    its stop set and the model emits ``<end_of_turn>`` as a literal
    token until ``max_tokens`` is hit.

    The helper is intentionally conservative:

    * Returns ``()`` for missing path, missing file, parse error,
      non-dict payload, or no ``eos_token_id`` key.
    * Returns ``()`` when the value is a single int — that case is
      already covered by the tokenizer's own ``eos_token_id``.
    * Filters out booleans (JSON ``True``/``False`` decode to
      ``int``) and non-finite values.
    """
    if not model_path:
        return ()
    config_path = _resolve_config_path(model_path)
    if config_path is None:
        return ()
    try:
        with open(config_path) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(
            "generation_config: eos read failed for %s: %s", config_path, exc
        )
        return ()
    if not isinstance(raw, dict):
        return ()
    value = raw.get("eos_token_id")
    if not isinstance(value, list):
        return ()
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if not isinstance(item, int):
            continue
        out.append(item)
    if out:
        logger.info(
            "generation_config: loaded extra EOS token ids from %s: %s",
            config_path,
            out,
        )
    return tuple(out)


def _resolve_config_path(model_path: str) -> str | None:
    """Best-effort locate ``generation_config.json`` for ``model_path``.

    ``model_path`` may be either a filesystem path (absolute or relative)
    or a ``org/repo`` HuggingFace identifier. We check the filesystem
    first, then fall back to the local HF hub cache layout.
    """
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, "generation_config.json")
        return candidate if os.path.isfile(candidate) else None

    # HF hub cache: ``<hub>/models--<org>--<repo>/snapshots/<sha>/generation_config.json``
    if "/" in model_path and ":" not in model_path:
        hub = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
            "~/.cache/huggingface/hub"
        )
        cache_root = os.path.join(hub, "models--" + model_path.replace("/", "--"))
        # Prefer the canonical revision from ``refs/main`` so a stale
        # snapshot pulled months ago doesn't shadow the current
        # generation_config. Falls through to a snapshot scan if the
        # ref file is missing or the resolved SHA is no longer on disk.
        ref_path = os.path.join(cache_root, "refs", "main")
        if os.path.isfile(ref_path):
            try:
                with open(ref_path) as fh:
                    sha = fh.read().strip()
            except OSError:
                sha = ""
            if sha:
                candidate = os.path.join(
                    cache_root, "snapshots", sha, "generation_config.json"
                )
                if os.path.isfile(candidate):
                    return candidate

        # Snapshot scan fallback — multiple snapshots can co-exist
        # (different revisions pulled over time). Each is a symlink farm
        # so generation_config.json is identical across them in practice;
        # any one with the file is acceptable.
        repo_dir = os.path.join(cache_root, "snapshots")
        if os.path.isdir(repo_dir):
            try:
                snapshots = sorted(os.listdir(repo_dir))
            except OSError:
                return None
            for snap in snapshots:
                candidate = os.path.join(repo_dir, snap, "generation_config.json")
                if os.path.isfile(candidate):
                    return candidate
    return None
