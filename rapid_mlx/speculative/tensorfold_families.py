# SPDX-License-Identifier: Apache-2.0
"""Data-driven Rapid profiles for the remaining TensorFold Mac families.

Each profile pins one measured target (and, where the family drafts with a
separate model, one measured drafter) to the shared TensorFold runtime.  The
loader follows TensorFold's own ``serve`` construction so a profile behaves the
way the upstream server does, while Rapid keeps the HTTP and output boundary.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tensorfold_qwen27 import (
    TensorFoldQwen27Backend,
    TensorFoldUnavailable,
    _snapshot_revision,
    require_environment,
    require_runtime,
)
from .tensorfold_runtime import SUPPORTED_VERSION

_GIB = 1024**3
_MIN_CHUNK = 256


@dataclass(frozen=True)
class TensorFoldFamilyProfile:
    """One catalog-qualified target (plus optional drafter) on the shared runtime."""

    profile_id: str
    label: str
    model_type: str
    target: str
    target_revision: str
    method: str
    algorithm: str
    fallback_model: str
    min_memory_gb: int
    quantization: tuple[int, int]
    # False when the alias cannot serve its target through the ordinary
    # engine, so there is no in-place opt-out.
    ordinary_engine: bool = True
    drafter: str | None = None
    drafter_revision: str | None = None
    drafter_bits: int = 4
    drafter_architecture: str | None = None


PROFILES: dict[str, TensorFoldFamilyProfile] = {
    profile.profile_id: profile
    for profile in (
        TensorFoldFamilyProfile(
            profile_id="bonsai2-27b-tensorfold",
            label="TensorFold Ternary Bonsai 2",
            model_type="prism_hadamard_qwen35",
            target="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
            target_revision="fcba37d2117a7077eac6b613b2668d14d9779edd",
            method="dflash",
            algorithm="dflash2",
            fallback_model="bonsai2-27b-2bit",
            min_memory_gb=96,
            quantization=(2, 128),
            ordinary_engine=False,
            drafter="z-lab/Qwen3.8-27B-DFlash2",
            drafter_revision="50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
            drafter_bits=4,
            drafter_architecture="DFlash2DraftModel",
        ),
        TensorFoldFamilyProfile(
            profile_id="nemotron-3.5-lightning-tensorfold",
            label="TensorFold Nemotron 3.5 Lightning",
            model_type="nemotron_h",
            target="TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",
            target_revision="d9d758fb83953437f7263256b0d96157e2a348b8",
            method="mtp",
            algorithm="mtp",
            fallback_model="nemotron-3.5-lightning-30b-4bit",
            min_memory_gb=48,
            quantization=(4, 64),
        ),
        TensorFoldFamilyProfile(
            profile_id="qwen3.8-flash-next-tensorfold",
            label="TensorFold Qwen3.8 Flash Next",
            model_type="qwen4_exp",
            target="TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP",
            target_revision="2b170fa6309d5d1ee380b35636075fac7945f286",
            method="mtp",
            algorithm="mtp",
            fallback_model="qwen3.8-flash-next-4bit",
            min_memory_gb=192,
            quantization=(4, 32),
            ordinary_engine=False,
        ),
    )
}


def profile_for(alias: str | None) -> TensorFoldFamilyProfile | None:
    return PROFILES.get(alias) if alias else None


def _read_config(path: Path, what: str) -> dict[str, Any]:
    try:
        parsed = json.loads((path / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise TensorFoldUnavailable(f"{what} requires a readable config.json") from exc
    if not isinstance(parsed, dict):
        raise TensorFoldUnavailable(f"{what} config.json is not an object")
    return parsed


def validate_artifacts(
    profile: TensorFoldFamilyProfile, target: Path, drafter: Path | None
) -> None:
    """Reject every checkpoint outside the measured revisions before loading."""

    if _snapshot_revision(target) != profile.target_revision:
        raise TensorFoldUnavailable(f"unqualified {profile.label} target revision")
    config = _read_config(target, "target")
    quant = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quant, dict):
        quant = {}
    if (
        config.get("model_type") != profile.model_type
        or (quant.get("bits"), quant.get("group_size")) != profile.quantization
    ):
        raise TensorFoldUnavailable(
            f"target is not the qualified {profile.label} checkpoint layout"
        )
    if profile.drafter is None:
        return
    if drafter is None:
        raise TensorFoldUnavailable(f"{profile.label} requires its qualified drafter")
    if _snapshot_revision(drafter) != profile.drafter_revision:
        raise TensorFoldUnavailable(f"unqualified {profile.label} drafter revision")
    draft_config = _read_config(drafter, "drafter")
    if draft_config.get("architectures") != [profile.drafter_architecture]:
        raise TensorFoldUnavailable(
            f"drafter is not the qualified {profile.label} draft model"
        )


@dataclass(frozen=True)
class FamilyArtifacts:
    target_path: str
    drafter_path: str


def download_qualified_artifacts(profile: TensorFoldFamilyProfile) -> FamilyArtifacts:
    """Resolve the profile's immutable snapshots through the shared Hub cache."""

    from .._mirror import pinned_snapshot_download

    target = pinned_snapshot_download(profile.target, profile.target_revision)
    drafter = (
        pinned_snapshot_download(profile.drafter, profile.drafter_revision)
        if profile.drafter is not None and profile.drafter_revision is not None
        else ""
    )
    validate_artifacts(profile, Path(target), Path(drafter) if drafter else None)
    return FamilyArtifacts(target_path=str(target), drafter_path=str(drafter))


def require_memory(
    profile: TensorFoldFamilyProfile, *, memory_gb: float | None = None
) -> None:
    if memory_gb is None:
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _GIB
    if memory_gb < profile.min_memory_gb:
        raise TensorFoldUnavailable(
            f"{profile.profile_id} requires a {profile.min_memory_gb} GB Mac"
        )


class TensorFoldFamilyBackend(TensorFoldQwen27Backend):
    """One-lane Rapid boundary around a profile's pinned TensorFold family."""

    profile: TensorFoldFamilyProfile

    @classmethod
    def load(
        cls,
        target_dir: str,
        drafter_dir: str = "",
        *,
        served_name: str,
        context_window: int = 0,
        max_tokens: int = 4096,
    ) -> TensorFoldFamilyBackend:
        profile = cls.profile
        require_runtime()
        require_environment()
        require_memory(profile)
        target = Path(target_dir)
        drafter = Path(drafter_dir) if drafter_dir else None
        validate_artifacts(profile, target, drafter)

        from tensorfold.families import detect

        family = detect(target)
        package = family.package
        if family.model_type != profile.model_type:
            raise TensorFoldUnavailable(
                f"TensorFold did not select the {profile.label} family"
            )
        # MLX reads these once, at import. Nothing above imports MLX (the
        # probes read package metadata only), so they still take effect here.
        # Like upstream's serve, an operator's explicit value takes precedence.
        for key, value in getattr(package, "MLX_ENV", {}).items():
            os.environ.setdefault(key, value)

        import mlx.core as mx
        from tensorfold.engine import prefill_step
        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.prefill_plan import PrefillPlan, message_markers
        from tensorfold.server.app import ChatApp
        from tensorfold.server.memory_budget import (
            PROCESS_BYTES,
            configure_mlx,
            model_fraction,
        )
        from tensorfold.server.prompt_memory import probe_tokens
        from tensorfold.server.residency import wire_resident

        memory_limit = configure_mlx(mx, 8 * _GIB, fraction=model_fraction(package))
        options: dict[str, Any] = {"lane_kernels": "auto", "parallel": 1}
        if drafter is not None:
            options["drafter"] = str(drafter)
            options["drafter_bits"] = profile.drafter_bits
        model = app = backend = None
        try:
            model, tokenizer = package.load(target, **options)
            settings = dict(package.engine_settings(model))
            getattr(model, "release_rounds", lambda: None)()
            wire_resident(mx, memory_limit - PROCESS_BYTES)
            openers, assistant = message_markers(tokenizer)
            steps = settings.pop("prefill_steps", None) or (LaneEngine.prefill_step,)
            step = prefill_step.choose(
                lambda grid: LaneEngine(model, prefill_plan=PrefillPlan(grid)),
                steps,
                memory_limit - PROCESS_BYTES,
                probe_tokens(tokenizer),
                int(context_window),
            )
            plan = PrefillPlan(step, openers, _MIN_CHUNK, assistant)
            engine_factory = functools.partial(
                LaneEngine, prefill_plan=plan, prefill_pass=8, pass_cache=16 * _GIB
            )
            resolve_prefill = getattr(model, "resolve_prefill_identity", None)
            if resolve_prefill is not None:
                resolve_prefill()
            app = ChatApp(
                model,
                tokenizer,
                served_name=served_name,
                engine_factory=engine_factory,
                lanes=1,
                max_rows=int(settings.get("max_rows", 16)),
                max_draft=int(settings.get("max_draft", 32)),
                default_max_tokens=int(max_tokens),
                context_window=int(context_window),
                enable_thinking=True,
                checkpoint_slots=0,
                checkpoint_budget_bytes=None,
                memory_budget_bytes=memory_limit,
                fit_context=not context_window,
                use_proposer=True,
                snapshot_dir=None,
                model_id=f"{target.resolve()}|tensorfold={SUPPORTED_VERSION}",
                model_dir=target,
            )
            backend = cls(app)
            hook = getattr(package, "setup", None)
            if hook is not None:
                hook(app, model, **options)
        except BaseException:
            # Startup failed with weights possibly allocated: stop an app that
            # runs its scheduler, then hand the memory back. The startup error
            # is the one to report, so a failing shutdown must not replace it.
            try:
                if backend is not None:
                    backend.close()
                elif app is not None:
                    app.scheduler.stop()
            except Exception:
                pass
            finally:
                # Drop every local owner first, or the cache clear frees nothing.
                model = app = backend = None
                mx.clear_cache()
            raise
        return backend


def backend_class_for(
    profile: TensorFoldFamilyProfile,
) -> type[TensorFoldFamilyBackend]:
    """A backend class bound to one profile for the shared server runner."""

    return type(
        f"TensorFold{profile.model_type.title().replace('_', '')}Backend",
        (TensorFoldFamilyBackend,),
        {"profile": profile},
    )


def run_tensorfold_family_server(
    profile: TensorFoldFamilyProfile, **kwargs: Any
) -> None:
    """Serve one qualified family profile through Rapid's shared text boundary."""

    from .tensorfold_qwen27_server import run_tensorfold_qwen27_server

    run_tensorfold_qwen27_server(
        **kwargs,
        backend_class=backend_class_for(profile),
        profile_id=profile.profile_id,
        backend_label=profile.label,
        method=profile.method,
        algorithm=profile.algorithm,
        fallback_model=profile.fallback_model,
        min_memory_gb=profile.min_memory_gb,
        runtime_extra="manual-source-install",
        target_repository=profile.target,
        target_revision=profile.target_revision,
        paired_repository=profile.drafter,
        paired_revision=profile.drafter_revision,
        supports_reasoning_budget=True,
    )
