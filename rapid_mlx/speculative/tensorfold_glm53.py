# SPDX-License-Identifier: Apache-2.0
"""Qualified GLM-5.3-Flash bridge to TensorFold's embedded-MTP lane."""

from __future__ import annotations

import functools
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .tensorfold_qwen27 import TensorFoldQwen27Backend, TensorFoldUnavailable

SUPPORTED_VERSION = "0.6.0"
SUPPORTED_MLX_VERSION = "0.32.3"
SUPPORTED_TARGET = "Vontra/GLM-5.3-Flash-MLX-4bit-MTP"
SUPPORTED_TARGET_REVISION = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"
SUPPORTED_RUNTIME_REVISION = "c4646171139ee8a3c38103eaa1699dad226ec12b"
INSTALL_HINT = (
    "Install the qualified TensorFold runtime from its vetted revision with:\n"
    '    python -m pip install "tensorfold @ '
    "git+https://github.com/ashhart/TensorFold.git@"
    f'{SUPPORTED_RUNTIME_REVISION}"'
)


def download_qualified_target() -> str:
    """Resolve the immutable target through the process-wide Hub cache."""

    from huggingface_hub import snapshot_download

    target = snapshot_download(SUPPORTED_TARGET, revision=SUPPORTED_TARGET_REVISION)
    validate_target(Path(target))
    return target


def _snapshot_revision(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.parts[resolved.parts.index("snapshots") + 1]
    except (ValueError, IndexError) as exc:
        raise TensorFoldUnavailable(
            f"checkpoint is not a pinned Hugging Face snapshot: {resolved}"
        ) from exc


def validate_target(target: Path) -> None:
    """Reject anything outside the measured target revision and weight ABI."""

    if _snapshot_revision(target) != SUPPORTED_TARGET_REVISION:
        raise TensorFoldUnavailable("unqualified GLM-5.3-Flash target revision")
    try:
        config = json.loads((target / "config.json").read_text())
        index = json.loads((target / "model.safetensors.index.json").read_text())
    except (OSError, ValueError) as exc:
        raise TensorFoldUnavailable(
            "GLM-5.3-Flash requires readable config and weight index files"
        ) from exc
    text = config.get("text_config") or config
    quant = config.get("quantization_config") or config.get("quantization") or {}
    mtp_prefix = f"model.language_model.layers.{text.get('num_hidden_layers')}."
    valid = (
        config.get("model_type") == "glm5_next"
        and text.get("num_hidden_layers") == 45
        and text.get("num_nextn_predict_layers") == 1
        and text.get("hidden_size") == 4096
        and quant.get("bits") == 4
        and quant.get("group_size") == 64
        and quant.get("mode") == "affine"
        and any(name.startswith(mtp_prefix) for name in index.get("weight_map", {}))
    )
    if not valid:
        raise TensorFoldUnavailable(
            "target is not the qualified GLM-5.3-Flash 4-bit/group-64 MTP layout"
        )


def _runtime_direct_url() -> dict[str, Any]:
    """Return pip's immutable VCS provenance for the installed runtime."""

    try:
        distribution = importlib.metadata.distribution("tensorfold")
        raw = distribution.read_text("direct_url.json")
        return json.loads(raw) if raw else {}
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        return {}


def require_runtime(
    version: str | None = None, *, direct_url: dict[str, Any] | None = None
) -> None:
    try:
        found = version or importlib.metadata.version("tensorfold")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TensorFoldUnavailable(
            f"tensorfold-glm53 requires tensorfold=={SUPPORTED_VERSION}"
        ) from exc
    if found != SUPPORTED_VERSION:
        raise TensorFoldUnavailable(
            f"tensorfold-glm53 requires tensorfold=={SUPPORTED_VERSION}; found {found}"
        )
    provenance = _runtime_direct_url() if direct_url is None else direct_url
    vcs = provenance.get("vcs_info") or {}
    if (
        vcs.get("vcs") != "git"
        or vcs.get("commit_id") != SUPPORTED_RUNTIME_REVISION
        or (provenance.get("dir_info") or {}).get("editable") is True
    ):
        raise TensorFoldUnavailable(
            "tensorfold-glm53 requires the exact qualified TensorFold git revision"
        )


def require_environment(
    *,
    mlx_version: str | None = None,
    machine: str | None = None,
    memory_gb: float | None = None,
) -> None:
    machine = machine or platform.machine()
    if sys.platform != "darwin" or machine != "arm64":
        raise TensorFoldUnavailable("tensorfold-glm53 requires Apple Silicon macOS")
    try:
        found = mlx_version or importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TensorFoldUnavailable(
            f"tensorfold-glm53 requires mlx=={SUPPORTED_MLX_VERSION}"
        ) from exc
    if found != SUPPORTED_MLX_VERSION:
        raise TensorFoldUnavailable(
            f"tensorfold-glm53 requires mlx=={SUPPORTED_MLX_VERSION}; found {found}"
        )
    if memory_gb is None:
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    if memory_gb < 256:
        raise TensorFoldUnavailable("tensorfold-glm53 requires a 256 GB Mac")


class TensorFoldGLM53Backend(TensorFoldQwen27Backend):
    """One-lane Rapid boundary around the pinned GLM TensorFold runtime."""

    @classmethod
    def load(
        cls,
        target_dir: str,
        _drafter_dir: str = "",
        *,
        served_name: str,
        context_window: int = 8192,
        max_tokens: int = 4096,
    ) -> TensorFoldGLM53Backend:
        require_runtime()
        require_environment()
        target = Path(target_dir)
        validate_target(target)

        from tensorfold.families import detect

        family = detect(target)
        package = family.package
        if family.model_type != "glm5_next":
            raise TensorFoldUnavailable("TensorFold did not select the GLM-5.3 family")
        for key, value in getattr(package, "MLX_ENV", {}).items():
            os.environ.setdefault(key, value)

        import mlx.core as mx
        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.prefill_plan import PrefillPlan, message_markers
        from tensorfold.server.app import ChatApp
        from tensorfold.server.memory_budget import PROCESS_BYTES, configure_mlx
        from tensorfold.server.residency import wire_resident

        memory_limit = configure_mlx(mx, 8 * 1024**3, fraction=0.85)
        model, tokenizer = package.load(target, mtp_drafts=3)
        settings = dict(package.engine_settings(model))
        getattr(model, "release_rounds", lambda: None)()
        wire_resident(mx, memory_limit - PROCESS_BYTES)
        openers, assistant = message_markers(tokenizer)
        plan = PrefillPlan(2048, openers, 256, assistant)
        engine_factory = functools.partial(
            LaneEngine,
            prefill_plan=plan,
            prefill_pass=8,
            pass_cache=16 * 1024**3,
        )
        app = ChatApp(
            model,
            tokenizer,
            served_name=served_name,
            engine_factory=engine_factory,
            lanes=1,
            max_rows=int(settings.get("max_rows", 16)),
            max_draft=int(settings.get("max_draft", 15)),
            default_max_tokens=int(max_tokens),
            context_window=int(context_window),
            enable_thinking=True,
            checkpoint_slots=0,
            checkpoint_budget_bytes=None,
            memory_budget_bytes=memory_limit,
            use_proposer=True,
            snapshot_dir=None,
            model_id=f"{target.resolve()}|tensorfold={SUPPORTED_VERSION}",
            model_dir=target,
        )
        return cls(app)


def run_tensorfold_glm53_server(**kwargs: Any) -> None:
    """Serve the qualified GLM profile through Rapid's shared text boundary."""

    from .tensorfold_qwen27_server import run_tensorfold_qwen27_server

    run_tensorfold_qwen27_server(
        **kwargs,
        backend_class=TensorFoldGLM53Backend,
        profile_id="glm5.3-flash-tensorfold",
        backend_label="TensorFold GLM-5.3-Flash",
        method="mtp",
        algorithm="mtp",
        fallback_model="glm5.3-flash-4bit",
        min_memory_gb=256,
        runtime_extra="manual-source-install",
        target_repository=SUPPORTED_TARGET,
        target_revision=SUPPORTED_TARGET_REVISION,
        paired_repository=None,
        paired_revision=None,
        supports_reasoning_budget=True,
    )
