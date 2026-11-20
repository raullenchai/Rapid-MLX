# SPDX-License-Identifier: Apache-2.0
"""Conservative, pre-spawn serving-mode allowlist (stdlib only)."""

import json
import os
from pathlib import Path

from . import RUNNER_ENV


def supervised_mode(args, env=None) -> str | None:
    """Return an allowed mode, or leave the existing serve path untouched.

    Unknown checkpoints fall back: no network or inference imports are needed
    to decide. Catalog targets establish text/VLM intent.
    """
    env = os.environ if env is None else env
    if (
        "RAPID_MLX_WATCHDOG_PPID" in env
        or getattr(args, "watchdog_ppid", None) is not None
        or env.get("RAPID_MLX_SINGLE_PROCESS") == "1"
        or RUNNER_ENV in env
    ):
        return None
    if any(
        getattr(args, name, None)
        for name in (
            "cua_only",
            "enable_dflash",
            "enable_ddtree",
            "enable_mtp",
            "force_spec_decode",
            "speculative_config",
            "draft_model",
            "dflash_drafter_path",
            "mtp_sidecar",
            "mtp_backend",
            "tensorfold_lanes",
            "enable_audio",
            "video_output_dir",
            "specprefill_draft_model",
        )
    ) or getattr(args, "spec_decode", "none") not in (None, "none"):
        return None
    model = getattr(args, "model", None)
    if not model:
        return (
            "retrieval"
            if (
                getattr(args, "embedding_model", None)
                or getattr(args, "rerank_model", None)
            )
            else None
        )
    catalog = json.loads((Path(__file__).parents[1] / "aliases.json").read_text())
    profile = catalog.get(model)
    if profile is None:
        profile = next(
            (
                p
                for p in catalog.values()
                if isinstance(p, dict) and p.get("hf_path") == model
            ),
            None,
        )
    if not isinstance(profile, dict):
        # Do not infer modality from a repository's name. An unrecognized remote
        # target can use the existing single-process path until qualified.
        return None
    if profile.get("modality", "text") not in ("text", "vision"):
        return None
    if any(
        profile.get(name)
        for name in (
            "audio_model_type",
            "dflash_draft_model",
            "ddtree_draft_model",
            "tensorfold_mtp",
            "tensorfold_dflash",
            "supports_native_mtp",
            "native_mtp_draft_model",
        )
    ):
        return None
    if (
        profile.get("mtp_draft_model")
        and profile.get("mtp_default_enabled", True)
        and not getattr(args, "no_spec_decode", False)
    ):
        return None
    # Native product lanes own alternate servers even without a spec flag.
    if (
        profile.get("model_type") in ("deepseek_v4", "deepseek_v41")
        or "dspark" in str(profile).lower()
        or str(profile.get("hf_path", "")).startswith("rapid-mlx/DeepSeek-V4.1-")
    ):
        return None
    return "lazy" if getattr(args, "lazy_load", False) else "text-vlm"
