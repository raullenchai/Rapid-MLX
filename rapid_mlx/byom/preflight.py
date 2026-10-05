# SPDX-License-Identifier: Apache-2.0
"""Metadata-only preflight for models that are not in the Rapid-MLX catalog.

``rapid-mlx serve`` and ``rapid-mlx pull`` accept any Hugging Face repo id or
local path. Before this module, a repo that could never load here (GGUF-only,
PyTorch ``.bin``-only, an architecture no installed loader implements, or a
checkpoint bigger than the Mac) was discovered only after the full download.

The preflight reads metadata only — one ``model_info`` call, plus a
``config.json`` fetch into a temporary directory when the summary suggests an
unsupported architecture — and stops before any weight byte is fetched when it
can PROVE the model cannot run. Every verdict is conservative: anything it
cannot establish (network trouble, gated repo, unusual layout, unknown loader
inventory) is "no verdict", and the existing download/load paths run exactly
as before. Cached repos and catalog models never reach this module.

Telemetry: a rejection is raised as :class:`PreflightRejectedError`, whose closed
``failure_class`` the existing ``model_serve_failed`` / ``model_pull_failed``
classifiers map onto registry enums. No repo name, file name or architecture
string leaves the machine through this module.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ``failure_class`` values. Each maps onto an existing registry enum value in
# ``rapid_mlx.telemetry.model_events`` — keep the two in sync.
UNSUPPORTED_FORMAT = "unsupported_format"
UNSUPPORTED_ARCHITECTURE = "unsupported_architecture"
INSUFFICIENT_MEMORY = "insufficient_memory"

# Architecture names that positively identify a text-generating checkpoint.
# The architecture verdict is only ever given for these: an encoder, embedding,
# speech or detection model is not something ``serve`` runs as a chat model,
# but it may be a perfectly good ``pull`` target for another lane.
_DECODER_ARCH_SUFFIXES = ("ForCausalLM", "LMHeadModel")
# Encoder-decoder / multimodal names are shared by speech models (Whisper)
# and VLMs; they count only together with a chat template.
_CHAT_ARCH_SUFFIXES = ("ForConditionalGeneration",)

# Model types Rapid-MLX registers into ``mlx_lm.models`` itself (see
# ``rapid_mlx/utils/tokenizer.py::_register_vendored_archs`` and the
# self-registering modules under ``rapid_mlx/models``), plus the natively
# converted DeepSeek V4.1 checkpoint. ``tests/test_byom_preflight.py`` scans the
# package for registrations so this cannot fall behind.
RAPID_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "bailing_hybrid",
        "cohere2_moe",
        "deepseek_v4",
        "deepseek_v41",
        "g9v3",
        "gpt_oss_puzzle",
        "hy_v3",
        "k2_horizon",
        "muse_glimmer",
        "nemotron_labs_diffusion",
        "qwen4_exp",
    }
)

_GGUF_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])(I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Hub metadata deadline. A slow Hub must never hold up a model that would have
# worked: on timeout the preflight gives no verdict and the normal path runs.
_HUB_TIMEOUT_SECONDS = 10.0

# Rough served footprint: weights plus KV cache / activations / runtime.
_SERVE_OVERHEAD_FRACTION = 0.2
_SERVE_OVERHEAD_BYTES = 1 << 30
# Above this share of unified memory the fit is reported as "tight".
_COMFORTABLE_FRACTION = 0.75

_GIB = float(1 << 30)
_MAX_QUANTS_SHOWN = 4


class PreflightRejectedError(Exception):
    """A model was rejected before download; ``failure_class`` is closed."""

    def __init__(self, failure_class: str) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class


@dataclass(frozen=True)
class Inspection:
    """What the metadata says about one repo or local path."""

    ref: str
    is_local: bool
    files: tuple[str, ...]
    weight_bytes: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    has_chat_template: bool | None = None
    dtype: str | None = None
    revision: str | None = None
    # Hub-only provenance used to look for an MLX build of the same model:
    # parameter count (safetensors or GGUF header), the repos this one is a
    # QUANTIZATION of (``base_model:quantized:<repo>`` tags only — finetunes
    # and merges are different models), its license and content tags.
    params: int | None = None
    quantized_from: tuple[str, ...] = ()
    license: str | None = None
    tags: tuple[str, ...] = ()
    # Anonymously readable (not gated, not private): only such repos may be
    # named in a support request.
    public: bool = False


@dataclass(frozen=True)
class Verdict:
    """The preflight's conclusion; ``failure`` is ``None`` when it may run."""

    failure: str | None
    format_label: str
    model_type: str | None
    architecture_supported: bool | None
    gguf_quants: tuple[str, ...] = ()
    est_memory_bytes: int | None = None
    ram_bytes: int | None = None
    fit: str = "unknown"


# --------------------------------------------------------------------------
# File-list helpers


def _root_files(files: Iterable[str]) -> list[str]:
    return [name for name in files if "/" not in name]


def is_runtime_weight(name: str) -> bool:
    """A root shard the text loader reads (mlx-lm globs ``model*.safetensors``).

    Adapters, consolidated copies and other side files at the root are not
    loaded, so they never count toward the memory estimate.
    """
    lower = name.lower()
    return (
        "/" not in name and lower.startswith("model") and lower.endswith(".safetensors")
    )


def _has_suffix(files: Iterable[str], suffix: str) -> bool:
    return any(name.lower().endswith(suffix) for name in files)


def gguf_quants(files: Iterable[str]) -> tuple[str, ...]:
    """Quantization labels named by GGUF file names, e.g. ``("Q4_K_M",)``."""
    found: list[str] = []
    for name in files:
        if not name.lower().endswith(".gguf"):
            continue
        stem = name.rsplit("/", 1)[-1][: -len(".gguf")]
        match = _GGUF_QUANT_RE.search(stem)
        label = match.group(1).upper() if match else None
        if label and label not in found:
            found.append(label)
    return tuple(found)


def classify_format(files: Iterable[str]) -> str | None:
    """Return a failure format (``"gguf"`` / ``"pytorch"``) or ``None``.

    Only a repo with NO safetensors anywhere is a format failure: weights in
    subfolders (one folder per quantization) are selected by other paths, and
    ``.npz`` repos belong to the audio lanes.
    """
    files = list(files)
    if _has_suffix(files, ".safetensors") or _has_suffix(files, ".npz"):
        return None
    if _has_suffix(files, ".gguf"):
        return "gguf"
    if any(
        name.lower().endswith(".bin") and "pytorch_model" in name.lower()
        for name in _root_files(files)
    ):
        return "pytorch"
    return None


# --------------------------------------------------------------------------
# Loader inventory


def _dict_literal_keys(source_file: Path, name: str) -> set[str] | None:
    """Keys of a module-level ``name = {...}`` dict literal, or ``None``."""
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return None
            return {str(key) for key in value} if isinstance(value, dict) else None
    return None


def _module_names(directory: Path) -> set[str]:
    names: set[str] = set()
    try:
        entries = list(directory.iterdir())
    except OSError:
        return names
    for entry in entries:
        if entry.name.startswith(("_", ".")):
            continue
        if entry.suffix == ".py":
            names.add(entry.stem)
        elif (entry / "__init__.py").is_file():
            names.add(entry.name)
    return names


def _package_dir(package: str) -> Path | None:
    """Locate an installed package WITHOUT importing it (no MLX import)."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    locations = list(spec.submodule_search_locations or []) if spec else []
    return Path(locations[0]) if locations else None


def _installed_types(package: str, model_dirs: tuple[str, ...]) -> set[str] | None:
    root = _package_dir(package)
    if root is None:
        return None
    remapped = _dict_literal_keys(root / "utils.py", "MODEL_REMAPPING")
    if remapped is None:
        return None
    names: set[str] = set(remapped)
    for sub in model_dirs:
        names |= _module_names(root / sub)
    return names


def supported_model_types() -> frozenset[str] | None:
    """Every ``model_type`` some installed (or installable) loader constructs.

    ``None`` means the inventory cannot be established (mlx-lm missing or its
    layout unreadable); callers must then give no architecture verdict.
    """
    text_types = _installed_types("mlx_lm", ("models",))
    if text_types is None:
        return None
    vision_types = _installed_types(
        "mlx_vlm", ("models", os.path.join("speculative", "drafters"))
    )
    if vision_types is None:
        from rapid_mlx.byom._vlm_model_types import MLX_VLM_MODEL_TYPES

        vision_types = set(MLX_VLM_MODEL_TYPES)
    # Speech runtimes (the optional [audio] extra), when installed. Speech
    # checkpoints are rarely decoder-only, but a TTS built on a causal LM is.
    audio_dirs = (
        os.path.join("stt", "models"),
        os.path.join("tts", "models"),
        os.path.join("sts", "models"),
    )
    audio_root = _package_dir("mlx_audio")
    audio_types: set[str] = set()
    if audio_root is not None:
        for sub in audio_dirs:
            audio_types |= _module_names(audio_root / sub)
    return frozenset(text_types | vision_types | audio_types | RAPID_MODEL_TYPES)


def _mlx_lm_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("mlx-lm")
    except Exception:
        return None


def architecture_supported(
    config: dict[str, Any],
    supported: frozenset[str] | None,
    *,
    chat_template: bool | None = None,
) -> bool | None:
    """``False`` only when the config PROVES no loader here can build it."""
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type or supported is None:
        return None
    if model_type.lower() in supported or model_type in supported:
        return True
    # Repo-supplied model code and speculative-drafter configs are resolved
    # by other mechanisms; never call them unsupported.
    if config.get("model_file") or config.get("dflash_config") is not None:
        return None
    architectures = config.get("architectures") or ()
    if not isinstance(architectures, (list, tuple)):
        return None
    names = [arch for arch in architectures if isinstance(arch, str)]
    if any(arch.endswith(_DECODER_ARCH_SUFFIXES) for arch in names):
        return False
    if chat_template is True and any(
        arch.endswith(_CHAT_ARCH_SUFFIXES) for arch in names
    ):
        return False
    return None


# --------------------------------------------------------------------------
# Inspection


def _quant_bits(config: dict[str, Any]) -> int | None:
    for key in ("quantization", "quantization_config"):
        block = config.get(key)
        if isinstance(block, dict) and isinstance(block.get("bits"), int):
            return int(block["bits"])
    return None


def _call_with_deadline(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    from rapid_mlx._download_gate import call_with_deadline

    return call_with_deadline(fn, _HUB_TIMEOUT_SECONDS, *args, **kwargs)


def _fetch_hub_config(repo_id: str, revision: str | None) -> dict[str, Any] | None:
    """Download ONLY ``config.json`` into a throwaway directory.

    A temp dir (not the HF cache) so a rejected repo leaves no config-only
    "stub" behind for later cache probes to trip over.
    """
    import json
    import shutil

    from huggingface_hub import hf_hub_download

    def _fetch() -> Any:
        # The worker owns its temp dir end to end: on a deadline the caller
        # returns while this thread may still be writing, so cleanup must not
        # happen on the caller's side.
        tmp = tempfile.mkdtemp(prefix="rapid-mlx-preflight-")
        try:
            path = hf_hub_download(
                repo_id, "config.json", revision=revision, local_dir=tmp
            )
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    try:
        parsed = _call_with_deadline(_fetch)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _hub_dtype(info: Any) -> str | None:
    params = getattr(getattr(info, "safetensors", None), "parameters", None)
    if not isinstance(params, dict) or not params:
        return None
    floats: list[str] = [
        str(key).lower() for key in params if str(key).upper() in {"BF16", "F16", "F32"}
    ]
    if len(floats) == len(params):
        best: str = max(floats, key=lambda key: params[key.upper()])
        return best
    return None


def _hub_params(info: Any) -> int | None:
    """Parameter count from the safetensors index or the GGUF header."""
    total = getattr(getattr(info, "safetensors", None), "total", None)
    if not isinstance(total, int):
        gguf = getattr(info, "gguf", None)
        total = gguf.get("total") if isinstance(gguf, dict) else None
    return total if isinstance(total, int) and total > 0 else None


def _card_get(card: Any, key: str) -> Any:
    try:
        return card.get(key) if card is not None else None
    except Exception:
        return None


def hub_tags(info: Any) -> tuple[str, ...]:
    tags = getattr(info, "tags", None)
    return (
        tuple(t for t in tags if isinstance(t, str)) if isinstance(tags, list) else ()
    )


def quantized_from(tags: tuple[str, ...]) -> tuple[str, ...]:
    """Repos a Hub repo declares itself a quantization of."""
    prefix = "base_model:quantized:"
    return tuple(t[len(prefix) :] for t in tags if t.startswith(prefix) and "/" in t)


def card_license(card: Any) -> str | None:
    value = _card_get(card, "license")
    return value if isinstance(value, str) and value else None


def inspect_hub(repo_id: str) -> Inspection | None:
    """Metadata for a Hub repo, or ``None`` when it cannot be read."""
    try:
        from huggingface_hub import model_info

        info = _call_with_deadline(model_info, repo_id, files_metadata=True)
    except Exception:
        return None
    siblings = getattr(info, "siblings", None) or []
    files = tuple(
        name
        for name in (getattr(s, "rfilename", None) for s in siblings)
        if isinstance(name, str)
    )
    from rapid_mlx._download_gate import _sibling_size

    weight_bytes = sum(
        _sibling_size(s) for s in siblings if is_runtime_weight(s.rfilename)
    )
    summary = getattr(info, "config", None)
    config = dict(summary) if isinstance(summary, dict) else {}
    tokenizer_config = config.pop("tokenizer_config", None)
    has_template: bool | None = None
    if any(
        name == "chat_template.jinja" or name == "chat_template.json" for name in files
    ):
        has_template = True
    elif isinstance(tokenizer_config, dict):
        has_template = bool(tokenizer_config.get("chat_template"))
    return Inspection(
        ref=repo_id,
        is_local=False,
        files=files,
        weight_bytes=weight_bytes,
        config=config,
        has_chat_template=has_template,
        dtype=_hub_dtype(info),
        revision=getattr(info, "sha", None),
        params=_hub_params(info),
        quantized_from=quantized_from(hub_tags(info)),
        license=card_license(getattr(info, "card_data", None)),
        tags=hub_tags(info),
        public=getattr(info, "private", None) is False
        and getattr(info, "gated", None) is False,
    )


def inspect_local(path: str) -> Inspection | None:
    """Metadata for a local model directory or a single weight file."""
    target = Path(path)
    try:
        if target.is_file():
            return Inspection(
                ref=path,
                is_local=True,
                files=(target.name,),
                weight_bytes=0,
            )
        if not target.is_dir():
            return None
        files = tuple(sorted(entry.name for entry in target.iterdir()))
        weight_bytes = sum(
            (target / name).stat().st_size for name in files if is_runtime_weight(name)
        )
    except OSError:
        return None
    from rapid_mlx.model_metadata import read_local_model_metadata

    metadata = read_local_model_metadata(path)
    config = dict(metadata.config or {}) if metadata is not None else {}
    has_template = metadata.chat_template is not None if metadata is not None else None
    dtype = config.get("torch_dtype") or config.get("dtype")
    return Inspection(
        ref=path,
        is_local=True,
        files=files,
        weight_bytes=weight_bytes,
        config=config,
        has_chat_template=has_template if config else None,
        dtype=dtype if isinstance(dtype, str) else None,
    )


# --------------------------------------------------------------------------
# Verdict


def _format_label(inspection: Inspection, bad_format: str | None) -> str:
    if bad_format == "gguf":
        return "GGUF"
    if bad_format == "pytorch":
        return "PyTorch .bin"
    bits = _quant_bits(inspection.config)
    if bits is not None:
        return f"MLX · {bits}-bit"
    if inspection.dtype:
        return f"safetensors · {inspection.dtype.lower().replace('bfloat16', 'bf16')}"
    return "safetensors"


def physical_ram_bytes() -> int | None:
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
    except Exception:
        return None
    return total if total > 0 else None


def evaluate(
    inspection: Inspection,
    *,
    refuse_oversize: bool,
    supported: frozenset[str] | None,
    ram_bytes: int | None,
) -> Verdict:
    """Decide whether ``inspection`` can run; never raises."""
    bad_format = classify_format(inspection.files)
    label = _format_label(inspection, bad_format)
    model_type = inspection.config.get("model_type")
    model_type = model_type if isinstance(model_type, str) else None
    if bad_format is not None:
        return Verdict(
            failure=UNSUPPORTED_FORMAT,
            format_label=label,
            model_type=model_type,
            architecture_supported=None,
            gguf_quants=gguf_quants(inspection.files),
        )
    arch_ok = architecture_supported(
        inspection.config, supported, chat_template=inspection.has_chat_template
    )
    est = None
    fit = "unknown"
    if inspection.weight_bytes > 0 and ram_bytes:
        est = int(
            inspection.weight_bytes * (1 + _SERVE_OVERHEAD_FRACTION)
            + _SERVE_OVERHEAD_BYTES
        )
        if inspection.weight_bytes >= ram_bytes:
            fit = "no"
        elif est <= ram_bytes * _COMFORTABLE_FRACTION:
            fit = "yes"
        else:
            fit = "tight"
    failure = None
    if arch_ok is False:
        failure = UNSUPPORTED_ARCHITECTURE
    elif fit == "no" and refuse_oversize:
        failure = INSUFFICIENT_MEMORY
    return Verdict(
        failure=failure,
        format_label=label,
        model_type=model_type,
        architecture_supported=arch_ok,
        est_memory_bytes=est,
        ram_bytes=ram_bytes,
        fit=fit,
    )


# --------------------------------------------------------------------------
# Rendering


def _gb(num_bytes: int) -> str:
    value = num_bytes / _GIB
    return f"{value:.1f} GB" if value < 10 else f"{value:.0f} GB"


def _hf_search_url(ref: str) -> str:
    from urllib.parse import quote

    name = ref.rstrip("/").rsplit("/", 1)[-1]
    name = re.sub(r"[-_.]?gguf$", "", name, flags=re.IGNORECASE) or name
    return f"https://huggingface.co/models?search={quote(name)}%20mlx"


def render_failure(
    inspection: Inspection, verdict: Verdict, *, search_hint: bool = True
) -> list[str]:
    nothing = "" if inspection.is_local else " Nothing was downloaded."
    shown = inspection.ref
    if verdict.failure == UNSUPPORTED_FORMAT and verdict.format_label == "GGUF":
        shown_quants = list(verdict.gguf_quants[:_MAX_QUANTS_SHOWN])
        if len(verdict.gguf_quants) > _MAX_QUANTS_SHOWN:
            shown_quants.append("…")
        quants = f" ({', '.join(shown_quants)})" if shown_quants else ""
        lines = [
            f"! {shown} only has GGUF files{quants}.",
            f"  Rapid-MLX runs MLX and safetensors weights, not GGUF.{nothing}",
        ]
        if not inspection.is_local and search_hint:
            lines += [
                "  Look for an MLX build of the same model:",
                f"    {_hf_search_url(shown)}",
            ]
        return lines
    if verdict.failure == UNSUPPORTED_FORMAT:
        return [
            f"! {shown} only has PyTorch .bin weights.",
            f"  Rapid-MLX loads safetensors (MLX or Hugging Face format).{nothing}",
            "  Look for a safetensors or MLX build of the same model.",
        ]
    if verdict.failure == UNSUPPORTED_ARCHITECTURE:
        version = _mlx_lm_version()
        runtime = f" (mlx-lm {version})" if version else ""
        return [
            f"✗ Architecture {verdict.model_type} is not supported yet.",
            f"  This Rapid-MLX install{runtime} has no loader for it.{nothing}",
            "  Models that run here: rapid-mlx models",
        ]
    assert verdict.est_memory_bytes is not None and verdict.ram_bytes is not None
    return [
        f"✗ {shown} needs ~{_gb(verdict.est_memory_bytes)} of memory; "
        f"this Mac has {_gb(verdict.ram_bytes)}.",
        f"  Its weights alone are {_gb(inspection.weight_bytes)}.{nothing}",
        "  Pick a smaller or more quantized build: rapid-mlx models",
    ]


def render_pass(inspection: Inspection, verdict: Verdict) -> list[str]:
    size = f" · {_gb(inspection.weight_bytes)}" if inspection.weight_bytes else ""
    lines = [
        "✓ Checked before download",
        f"  Format        {verdict.format_label}{size}",
    ]
    if verdict.model_type:
        state = "supported" if verdict.architecture_supported else "not verified"
        lines.append(f"  Architecture  {verdict.model_type} ({state})")
    if inspection.has_chat_template is True:
        lines.append("  Chat template found")
    elif inspection.has_chat_template is False:
        lines.append(
            "  Chat template none — likely a base model; prefer /v1/completions"
        )
    if verdict.fit != "unknown":
        assert verdict.est_memory_bytes is not None and verdict.ram_bytes is not None
        lines.append(
            f"  Fits your Mac {verdict.fit} · "
            f"~{_gb(verdict.est_memory_bytes)} of {_gb(verdict.ram_bytes)}"
        )
    return lines


# --------------------------------------------------------------------------
# CLI hook


def _cli_version() -> str:
    try:
        from importlib.metadata import version

        return version("rapid-mlx")
    except Exception:
        return "dev"


def gguf_format_requested(fmt: object) -> bool:
    return isinstance(fmt, str) and fmt.strip().lower() == "gguf"


GGUF_FORMAT_MESSAGE = (
    "Rapid-MLX cannot run GGUF files, so `--format gguf` is not available. "
    "Pick an MLX variant (e.g. --bits 4) or pull the repo without a selector."
)


def _is_catalog_or_registry_model(name: str) -> bool:
    from rapid_mlx.audio.registry import resolve_audio_alias
    from rapid_mlx.model_aliases import resolve_profile

    return resolve_profile(name) is not None or resolve_audio_alias(name) is not None


def _target_status(args: Any) -> tuple[str | None, tuple[str, bool] | None]:
    """``(unchecked outcome, target)`` for these args.

    ``target`` is ``(ref, is_local)`` when the preflight applies. Otherwise
    the outcome names why a bring-your-own model was NOT checked (a
    ``byom_preflight`` telemetry value: ``skipped`` / ``no_verdict`` /
    ``cached``), or is ``None`` for catalog models and selector pulls.
    """
    command = getattr(args, "command", None)
    model = getattr(args, "model", None)
    if command not in ("serve", "pull") or not isinstance(model, str) or not model:
        return None, None
    skipped = bool(getattr(args, "no_preflight", False))
    if command == "pull" and (
        getattr(args, "bits", None) is not None
        or getattr(args, "format", None) is not None
    ):
        return None, None
    if os.path.exists(model):
        return ("skipped", None) if skipped else (None, (model, True))
    if "/" not in model or _is_catalog_or_registry_model(model):
        return None, None
    if skipped:
        return "skipped", None
    from rapid_mlx._download_gate import is_repo_cached
    from rapid_mlx.model_metadata import hub_offline_mode_active

    # Offline first, exactly like origin/main: offline mode never probes the
    # cache, so an offline run is "no_verdict" even for a cached repo.
    if hub_offline_mode_active():
        return "no_verdict", None
    if is_repo_cached(model):
        return "cached", None
    return None, (model, False)


def preflight_target(args: Any) -> tuple[str, bool] | None:
    """``(ref, is_local)`` when the preflight applies to these args."""
    return _target_status(args)[1]


def _funnel() -> Any:
    from rapid_mlx.telemetry import byom_funnel

    return byom_funnel


def _suggestion_kind(hints: list[str], found_build: bool) -> str:
    if found_build:
        return "mlx_build"
    return "catalog" if hints else "none"


def _emit_rejection(args: Any, exc: PreflightRejectedError) -> None:
    shown = getattr(args, "_original_alias", None) or args.model
    if args.command == "serve":
        from rapid_mlx.telemetry.model_events import emit_model_serve_failed
        from rapid_mlx.telemetry.server_start import set_failure_stage

        set_failure_stage("preflight")
        emit_model_serve_failed(exc, alias_or_path=shown, failure_stage="preflight")
    else:
        from rapid_mlx.telemetry.model_events import emit_model_pull_failed

        emit_model_pull_failed(exc, model_ref=shown)


def run_cli_preflight(args: Any, *, spinner_factory: Callable[[str], Any]) -> None:
    """Run the preflight for ``serve``/``pull`` args; exit 1 on a rejection.

    Silent (no output, no state change) for every model it does not apply to
    and whenever metadata cannot be read. On an interactive terminal a passing
    check prints a short summary before the download starts.
    """
    funnel = _funnel()
    funnel.begin((getattr(args, "_original_alias", None), getattr(args, "model", None)))
    if args.command == "pull" and gguf_format_requested(getattr(args, "format", None)):
        print(f"\n  Error: {GGUF_FORMAT_MESSAGE}", file=sys.stderr)
        _emit_rejection(args, PreflightRejectedError(UNSUPPORTED_FORMAT))
        raise SystemExit(1)
    unchecked, target = _target_status(args)
    if target is None:
        if unchecked is not None:
            funnel.set_preflight(unchecked)
        return
    ref, is_local = target
    # Until a verdict lands, an interrupted or unreadable check is no verdict.
    funnel.set_preflight("no_verdict")
    if is_local:
        inspection = inspect_local(ref)
    else:
        with spinner_factory(f"Checking {ref.split('/')[-1]} …"):
            inspection = inspect_hub(ref)
    if inspection is None:
        return
    supported = supported_model_types()
    # ``pull`` is storage-only and may target another Mac, and ``--disk-stream``
    # serves MoE experts from disk; only a plain ``serve`` refuses a checkpoint
    # whose weights alone exceed this machine's memory.
    verdict = evaluate(
        inspection,
        refuse_oversize=args.command == "serve"
        and not getattr(args, "disk_stream", False),
        supported=supported,
        ram_bytes=physical_ram_bytes(),
    )
    if verdict.failure == UNSUPPORTED_ARCHITECTURE and not inspection.is_local:
        # The Hub's config summary omits fields that route a model to other
        # loaders; confirm against the real config.json before refusing.
        full = _fetch_hub_config(ref, inspection.revision)
        confirmed = (
            None
            if full is None
            else architecture_supported(
                full, supported, chat_template=inspection.has_chat_template
            )
        )
        if confirmed is not False:
            if confirmed is True:
                funnel.set_preflight("passed")
            return
    if verdict.failure is not None:
        from rapid_mlx.byom.alternatives import suggest

        suggested: list[str] = []
        with spinner_factory("Looking for a model that runs here …"):
            hints, found_build = suggest(
                inspection,
                verdict,
                command=args.command,
                supported=supported,
                ram_bytes=verdict.ram_bytes or physical_ram_bytes(),
                targets=suggested,
            )
        print("", file=sys.stderr)
        lines = render_failure(inspection, verdict, search_hint=not found_build)
        for line in lines + hints:
            print(f"  {line}", file=sys.stderr)
        print(
            "    (Think this is wrong? Re-run with --no-preflight to skip this check.)\n",
            file=sys.stderr,
        )
        from rapid_mlx.byom.support_request import offer

        outcome = offer(args, inspection, verdict, _cli_version())
        funnel.note_refusal(
            suggestion=_suggestion_kind(hints, found_build),
            support_request=outcome,
            suggested_refs=suggested,
        )
        _emit_rejection(args, PreflightRejectedError(verdict.failure))
        raise SystemExit(1)
    funnel.set_preflight("passed")
    if not is_local and sys.stdout.isatty():
        print()
        for line in render_pass(inspection, verdict):
            print(f"  {line}")
