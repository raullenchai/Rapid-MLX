"""Offline discovery and policy-ranked defaults for complete local chat models."""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

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


# Whole name segments that mark a non-chat role (embedding, reranking, speech).
_NON_CHAT_TOKENS = frozenset(
    {
        "embed",
        "embedding",
        "embeddings",
        "rerank",
        "reranker",
        "whisper",
        "tts",
        "vad",
        "tokenizer",
        "asr",
        "stt",
    }
)


def _checkpoint_dir(repo: str, external: bool) -> str | None:
    """The exact local checkpoint directory serve would load, or ``None``."""
    from rapid_mlx import model_aliases
    from rapid_mlx.model_metadata import read_cached_model_metadata

    if external:
        # The external scanner keeps one row per repo id (the first configured
        # root holding a complete copy), and serve resolves the id to that same
        # first copy. Judge exactly that directory: a later root's copy of the
        # same id is never what serve would load.
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


def _packed_quantization(config: dict) -> bool:
    """AWQ/GPTQ/other packed formats declare a ``quant_method``; the MLX
    loader cannot run them, so they are never auto-selected."""
    return any(
        isinstance(config.get(key), dict) and "quant_method" in config[key]
        for key in ("quantization", "quantization_config")
    )


def serve_startup_note(args: Any, line: str) -> None:
    """Print a ``serve`` startup note on stderr, or hold it for the log file.

    With ``--log-file`` (or its environment equivalent) all serve output
    belongs in the log, which ``serve_command`` opens later; it flushes the
    held notes right after redirecting.
    """
    from rapid_mlx import log_file

    target, _source = log_file.resolve(getattr(args, "log_file", None))
    if target is None:
        print(line, file=sys.stderr)
        return
    notes = getattr(args, "_startup_notes", None)
    if notes is None:
        notes = []
        args._startup_notes = notes
    notes.append(line)


def flush_serve_startup_notes(args: Any) -> None:
    for line in getattr(args, "_startup_notes", None) or ():
        print(line, file=sys.stderr)


def _uncataloged_chat(
    repo: str, inspection: Inspection, supported: frozenset[str] | None
) -> bool | None:
    """``None`` when not a runnable chat model, else whether it is MLX-native."""
    from rapid_mlx.byom.alternatives import _mlx_bits
    from rapid_mlx.byom.preflight import chat_model_types, evaluate

    lowered = repo.lower()
    # Name heuristics look at the repository name's own segments only, never
    # the publisher, so ``org/embedded-chat`` or a publisher named "...asr..."
    # is not excluded by accident.
    segments = re.split(r"[^a-z0-9]+", lowered.rsplit("/", 1)[-1])
    if any(segment in _NON_CHAT_TOKENS for segment in segments):
        return None
    if segments[-1] in ("draft", "drafter"):
        return None
    config = inspection.config
    # Only text and vision loaders give a chat model; speech checkpoints with a
    # causal or conditional-generation head are never a chat default.
    model_type = config.get("model_type")
    chat_types = chat_model_types()
    if (
        not isinstance(model_type, str)
        or chat_types is None
        or (model_type not in chat_types and model_type.lower() not in chat_types)
    ):
        return None
    architectures = config.get("architectures")
    if not isinstance(architectures, (list, tuple)) or not any(
        isinstance(name, str)
        and name.endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration"))
        for name in architectures
    ):
        return None
    verdict = evaluate(
        inspection,
        refuse_oversize=True,
        supported=supported,
        ram_bytes=None,
    )
    if verdict.failure is not None or verdict.architecture_supported is not True:
        return None
    return "mlx" in lowered or _mlx_bits(config) is not None


def _judge_checkpoint(
    repo: str,
    alias: str | None,
    *,
    external: bool,
    mtime: float,
    supported: frozenset[str] | None,
    ram_bytes: int | None,
) -> LocalModel | None:
    """The checkpoint as a runnable local chat model, or ``None``."""
    from rapid_mlx.byom.preflight import evaluate, inspect_local
    from rapid_mlx.first_run import _is_chat_alias
    from rapid_mlx.model_aliases import raise_if_draft_only_model

    raise_if_draft_only_model(alias or repo)
    if alias is not None and not _is_chat_alias(alias):
        return None
    # Judge the exact checkpoint serve would load: a config, a chat template,
    # a tokenizer vocabulary and runtime weights must all be on disk.
    directory = _checkpoint_dir(repo, external)
    inspection = inspect_local(directory) if directory else None
    if (
        inspection is None
        or not inspection.config
        or inspection.weight_bytes <= 0
        or inspection.has_chat_template is not True
        or not _has_tokenizer(inspection.files)
        or _packed_quantization(inspection.config)
    ):
        return None
    mlx_native = True
    if alias is None:
        native = _uncataloged_chat(repo, inspection, supported)
        if native is None:
            return None
        mlx_native = native
    # Memory fit uses the preflight rule on this checkpoint's own weights, not
    # the whole repo directory (old revisions etc.). A failure (proven
    # unsupported architecture or oversize) excludes cataloged checkpoints too.
    verdict = evaluate(
        inspection,
        refuse_oversize=True,
        supported=supported,
        ram_bytes=ram_bytes,
    )
    if verdict.failure is not None or verdict.fit == "no":
        return None
    return LocalModel(
        alias or repo,
        repo,
        alias,
        external,
        inspection.weight_bytes,
        mtime,
        verdict.fit,
        mlx_native,
    )


def catalog_model_ready(alias: str) -> bool:
    """Whether a cataloged chat alias's cached Hub checkpoint passes the same
    checks as discovery. Fail-silent: any probe error means not ready."""
    try:
        from rapid_mlx.byom.preflight import physical_ram_bytes, supported_model_types
        from rapid_mlx.model_aliases import list_profiles

        profile = list_profiles().get(alias)
        if profile is None:
            return False
        return (
            _judge_checkpoint(
                profile.hf_path,
                alias,
                external=False,
                mtime=0.0,
                supported=supported_model_types(),
                ram_bytes=physical_ram_bytes(),
            )
            is not None
        )
    except Exception:
        log.debug("Could not check local model %s", alias, exc_info=True)
        return False


def discover_runnable_chat_models(ram_bytes: int | None = None) -> list[LocalModel]:
    """Read local metadata only; broken caches never interrupt the CLI."""
    try:
        from rapid_mlx.byom.preflight import physical_ram_bytes, supported_model_types
        from rapid_mlx.cli import (
            _cache_entry_is_runnable,
            _scan_external_model_dirs,
            _scan_hf_cache_models,
        )
        from rapid_mlx.model_aliases import list_profiles

        if ram_bytes is None:
            ram_bytes = physical_ram_bytes()
        supported = supported_model_types()
        aliases: dict[str, str] = {}
        for name, profile in list_profiles().items():
            aliases.setdefault(profile.hf_path, name)
        hub = [
            row for row in _scan_hf_cache_models() if _cache_entry_is_runnable(row[0])
        ]
        hub_repos = {row[0] for row in hub}
        rows = [(row, False) for row in hub]
        rows.extend((row, True) for row in _scan_external_model_dirs())
        found: list[LocalModel] = []
        accepted_hub: set[str] = set()
        for (repo, _size, mtime), external in rows:
            # The managed Hub copy wins when it passes; an external copy of the
            # same repo is only a fallback for a Hub copy that did not.
            if external and repo in accepted_hub:
                continue
            try:
                model = _judge_checkpoint(
                    repo,
                    aliases.get(repo),
                    external=external,
                    mtime=mtime,
                    supported=supported,
                    ram_bytes=ram_bytes,
                )
                if model is not None and external and repo in hub_repos:
                    # The id would resolve to the rejected Hub copy; serve the
                    # verified external directory by path instead.
                    directory = _checkpoint_dir(repo, True)
                    model = replace(model, model=directory) if directory else None
            except Exception:
                log.debug("Could not inspect local model %s", repo, exc_info=True)
                continue
            if model is not None:
                if not external:
                    accepted_hub.add(repo)
                found.append(model)
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
