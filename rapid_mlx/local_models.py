"""Offline discovery and policy-ranked defaults for complete local chat models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapid_mlx.byom.preflight import Inspection

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalModel:
    model: str
    repo: str
    alias: str | None
    external: bool
    size_bytes: int
    mtime: float
    fit: str
    # Converted for MLX (quantized MLX config or an MLX-named repo). A plain
    # Transformers checkpoint can load too, but ranks after MLX builds.
    mlx_native: bool = True

    @property
    def cataloged(self) -> bool:
        return self.alias is not None


_NON_CHAT_TOKENS = (
    "embed",
    "rerank",
    "whisper",
    "tts",
    "vad",
    "tokenizer",
    "asr",
    "stt",
)


def _checkpoint_dir(repo: str, external: bool) -> str | None:
    """The exact local checkpoint directory serve would load, or ``None``."""
    from rapid_mlx import model_aliases
    from rapid_mlx.model_metadata import read_cached_model_metadata

    if external:
        return model_aliases._resolve_external_model_path(repo)
    metadata = read_cached_model_metadata(repo)
    if metadata is None or metadata.snapshot_dir is None:
        return None
    return str(metadata.snapshot_dir)


_TOKENIZER_VOCAB_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer.model",
        "vocab.txt",
        "spiece.model",
        "tiktoken.model",
    }
)


def _has_tokenizer(files: tuple[str, ...]) -> bool:
    """A tokenizer vocabulary, not just ``tokenizer_config.json`` metadata.

    A byte-level BPE ``vocab.json`` is only usable together with its
    ``merges.txt``.
    """
    names = set(files)
    return (
        any(
            name in _TOKENIZER_VOCAB_FILES or name.endswith(".tiktoken")
            for name in names
        )
        or {"vocab.json", "merges.txt"} <= names
    )


def _uncataloged_chat(repo: str, inspection: Inspection) -> bool | None:
    """``None`` when not a runnable chat model, else whether it is MLX-native."""
    from rapid_mlx.byom.alternatives import _mlx_bits
    from rapid_mlx.byom.preflight import evaluate, supported_model_types

    lowered = repo.lower()
    if any(token in lowered for token in _NON_CHAT_TOKENS):
        return None
    if lowered.endswith(("-draft", "-drafter", "_draft", "_drafter")):
        return None
    if inspection.has_chat_template is not True:
        return None
    config = inspection.config
    architectures = config.get("architectures")
    if not isinstance(architectures, (list, tuple)) or not any(
        isinstance(name, str)
        and name.endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration"))
        for name in architectures
    ):
        return None
    # AWQ/GPTQ/other packed formats declare a ``quant_method``; the MLX loader
    # cannot run them, so they are never auto-selected.
    for key in ("quantization", "quantization_config"):
        block = config.get(key)
        if isinstance(block, dict) and "quant_method" in block:
            return None
    verdict = evaluate(
        inspection,
        refuse_oversize=True,
        supported=supported_model_types(),
        ram_bytes=None,
    )
    if verdict.failure is not None or verdict.architecture_supported is not True:
        return None
    return "mlx" in lowered or _mlx_bits(config) is not None


def discover_runnable_chat_models(ram_bytes: int | None = None) -> list[LocalModel]:
    """Read local metadata only; broken caches never interrupt the CLI."""
    try:
        from rapid_mlx.byom.preflight import evaluate, inspect_local, physical_ram_bytes
        from rapid_mlx.cli import (
            _cache_entry_is_runnable,
            _scan_external_model_dirs,
            _scan_hf_cache_models,
        )
        from rapid_mlx.first_run import _is_chat_alias
        from rapid_mlx.model_aliases import list_profiles, raise_if_draft_only_model

        if ram_bytes is None:
            ram_bytes = physical_ram_bytes()
        aliases: dict[str, str] = {}
        for name, profile in list_profiles().items():
            aliases.setdefault(profile.hf_path, name)
        hub = [
            row for row in _scan_hf_cache_models() if _cache_entry_is_runnable(row[0])
        ]
        hub_repos = {row[0] for row in hub}
        rows = [(row, False) for row in hub]
        rows.extend(
            (row, True)
            for row in _scan_external_model_dirs()
            if row[0] not in hub_repos
        )
        found = []
        for (repo, _size, mtime), external in rows:
            try:
                alias = aliases.get(repo)
                raise_if_draft_only_model(alias or repo)
                if alias is not None and not _is_chat_alias(alias):
                    continue
                # Judge the exact checkpoint serve would load: a config, a
                # tokenizer and runtime weights must all be on disk.
                directory = _checkpoint_dir(repo, external)
                inspection = inspect_local(directory) if directory else None
                if (
                    inspection is None
                    or not inspection.config
                    or inspection.weight_bytes <= 0
                    or not _has_tokenizer(inspection.files)
                ):
                    continue
                mlx_native = True
                if alias is None:
                    native = _uncataloged_chat(repo, inspection)
                    if native is None:
                        continue
                    mlx_native = native
                # Memory fit uses the preflight rule on this checkpoint's own
                # weights, not the whole repo directory (old revisions etc.).
                verdict = evaluate(
                    inspection,
                    refuse_oversize=True,
                    supported=None,
                    ram_bytes=ram_bytes,
                )
                if verdict.fit == "no":
                    continue
                found.append(
                    LocalModel(
                        alias or repo,
                        repo,
                        alias,
                        external,
                        inspection.weight_bytes,
                        mtime,
                        verdict.fit,
                        mlx_native,
                    )
                )
            except Exception:
                log.debug("Could not inspect local model %s", repo, exc_info=True)
        return found
    except Exception:
        log.debug("Could not discover local models", exc_info=True)
        return []


def select_default_local_model(ram_gb: float | None = None) -> LocalModel | None:
    """Mirror Desktop Quickstart: baseline, tier candidates, other catalog, then BYOM.

    Within the latter groups comfortable fits win; catalog aliases sort by name,
    uncataloged models put MLX-native builds first, then recency across hub and
    external stores. Unknown RAM uses
    the 16 GB policy while retaining unknown memory fit.
    """
    try:
        from rapid_mlx.recommendations import (
            physical_ram_gb,
            starter_baseline_for_ram,
            starter_model_candidates,
        )

        if ram_gb is None:
            ram_gb = physical_ram_gb()
        models = discover_runnable_chat_models(
            int(ram_gb * (1 << 30)) if ram_gb > 0 else 0
        )
        policy_ram = ram_gb if ram_gb > 0 else 16
        catalog = {model.alias: model for model in models if model.cataloged}
        for alias in (
            starter_baseline_for_ram(policy_ram),
            *starter_model_candidates(policy_ram),
        ):
            if alias in catalog:
                return catalog[alias]
        if catalog:
            return min(
                catalog.values(), key=lambda model: (model.fit != "yes", model.alias)
            )
        return (
            min(
                models,
                key=lambda model: (
                    model.fit != "yes",
                    not model.mlx_native,
                    -model.mtime,
                ),
            )
            if models
            else None
        )
    except Exception:
        log.debug("Could not select a local model", exc_info=True)
        return None
