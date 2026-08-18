# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stability AI — vendored from github.com/Stability-AI/stable-audio-3
"""Shared weights manifest + downloader.

Maps every weight file the runtime needs to its position in the
`stabilityai/stable-audio-3-optimized` HuggingFace repo.

`install.py` calls `ensure_local` upfront for the bundles the user picks.
`sa3_mlx.py` calls `ensure_local` lazily, just before each model loads — so
a fresh checkout with no weights still works if the user is willing to
wait for the first run to download them.
"""

from __future__ import annotations

from pathlib import Path

REPO_ID = "stabilityai/stable-audio-3-optimized"
# weights.py lives in <project>/scripts/; SCRIPT_DIR points at the project
# root so the local rel paths in the manifest ("models/mlx/foo.npz") resolve
# against the actual project layout.
SCRIPT_DIR = Path(__file__).resolve().parent.parent


# Bundles the install script offers to the user. Each maps to a list of
# weight files (local relative path on the left, HF repo path on the right).
# T5Gemma is in SHARED because every bundle needs it.

DIT_BUNDLES: dict[str, list[tuple[str, str]]] = {
    "medium": [
        ("models/mlx/dit_medium_f16.npz",     "MLX/dit_medium_f16.npz"),
        ("models/mlx/same_l_decoder_f32.npz", "MLX/same_l_decoder_f32.npz"),
        ("models/mlx/same_l_encoder_f32.npz", "MLX/same_l_encoder_f32.npz"),
    ],
    "sm-music": [
        ("models/mlx/dit_sm-music_f16.npz",   "MLX/dit_sm-music_f16.npz"),
        ("models/mlx/same_s_decoder_f32.npz", "MLX/same_s_decoder_f32.npz"),
        ("models/mlx/same_s_encoder_f32.npz", "MLX/same_s_encoder_f32.npz"),
    ],
    "sm-sfx": [
        ("models/mlx/dit_sm-sfx_f16.npz",     "MLX/dit_sm-sfx_f16.npz"),
        ("models/mlx/same_s_decoder_f32.npz", "MLX/same_s_decoder_f32.npz"),
        ("models/mlx/same_s_encoder_f32.npz", "MLX/same_s_encoder_f32.npz"),
    ],
}

SHARED: list[tuple[str, str]] = [
    ("models/mlx/t5gemma_f16.npz", "MLX/t5gemma_f16.npz"),
]

# Base-model DiT weights — needed only for LoRA *training* (lora_train_mlx.py
# trains on the BASE checkpoint, not the ARC weights inference uses). Not part
# of any install bundle; auto-downloaded lazily by the trainer via ensure_local.
TRAINING_BASE: list[tuple[str, str]] = [
    ("models/mlx/dit_sm-music-base_f16.npz", "MLX/dit_sm-music-base_f16.npz"),
    ("models/mlx/dit_sm-sfx-base_f16.npz",   "MLX/dit_sm-sfx-base_f16.npz"),
    ("models/mlx/dit_medium-base_f16.npz",   "MLX/dit_medium-base_f16.npz"),
]

# Human-friendly bundle sizes (for the install prompt).
BUNDLE_SIZES = {
    "medium":   "5.9 GB  (medium DiT + SAME-L codec)",
    "sm-music": "1.3 GB  (small music DiT + SAME-S codec)",
    "sm-sfx":   "1.3 GB  (small sfx DiT + SAME-S codec)",
}

# Flat (local_rel_path → hf_path) lookup — used by sa3_mlx.py for lazy
# auto-download at load time.
FLAT_MANIFEST: dict[str, str] = {}
for _items in DIT_BUNDLES.values():
    for _rel, _hf in _items:
        FLAT_MANIFEST[_rel] = _hf
for _rel, _hf in SHARED:
    FLAT_MANIFEST[_rel] = _hf
for _rel, _hf in TRAINING_BASE:
    FLAT_MANIFEST[_rel] = _hf


def _hf_token_configured() -> bool:
    """True if any HF token is set — env var or cached login on disk."""
    import os
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token  # huggingface_hub ≥ 0.19
        return bool(get_token())
    except ImportError:
        try:
            from huggingface_hub import HfFolder
            return bool(HfFolder.get_token())
        except Exception:
            return False
    except Exception:
        return False


_LOGIN_TIP_SHOWN = False

def _show_hf_login_tip_once() -> None:
    """Print a one-time login suggestion if no HF token is configured.

    Anonymous downloads work but have a ~50 GB/day soft cap on HF's LFS CDN
    and lower aggregate bandwidth — a free token effectively removes both.
    Stays silent if a token is already in place.
    """
    global _LOGIN_TIP_SHOWN
    if _LOGIN_TIP_SHOWN:
        return
    _LOGIN_TIP_SHOWN = True
    if _hf_token_configured():
        return
    import sys
    YEL  = "\033[1;33m" if sys.stdout.isatty() else ""
    BOLD = "\033[1m"    if sys.stdout.isatty() else ""
    DIM  = "\033[2m"    if sys.stdout.isatty() else ""
    RST  = "\033[0m"    if sys.stdout.isatty() else ""
    print()
    print(f"  {YEL}⚠  not logged in to HuggingFace{RST} — anonymous downloads work but are")
    print(f"     rate-limited (~50 GB/day cap on the LFS CDN). For faster, higher-limit")
    print(f"     downloads, log in once with a free read-only token:")
    print()
    print(f"       1. create an account at {BOLD}https://huggingface.co/join{RST}")
    print(f"       2. generate a token at {BOLD}https://huggingface.co/settings/tokens{RST}")
    print(f"          {DIM}('Read' scope is enough){RST}")
    print(f"       3. save it on this machine — pick one:")
    print(f"            {BOLD}hf auth login{RST}              {DIM}# modern (huggingface_hub ≥ 1.0){RST}")
    print(f"            {BOLD}huggingface-cli login{RST}      {DIM}# classic; still works{RST}")
    print(f"            {BOLD}export HF_TOKEN=hf_xxx{RST}     {DIM}# one-off / scripts{RST}")
    print()


def ensure_local(local_rel_path: str, verbose: bool = True) -> Path:
    """Resolve a weight file to an absolute local path, downloading if missing.

    Resolution order:
      1. A real (materialized) file already vendored at ``local_rel_path`` — a
         dev checkout that ran the old bundle flow. Used as-is.
      2. Otherwise the file is streamed into the writable HuggingFace cache
         (``~/.cache/huggingface/hub/``) and that cache path is returned
         DIRECTLY.

    The runtime loads weights straight from the returned path. We deliberately
    do NOT symlink/copy the download into the vendored package directory: when
    rapid-mlx is pip/brew-installed that directory lives under a read-only
    ``site-packages`` tree, so writing there raises ``PermissionError`` and
    generation would crash on first use. Keeping the download in the HF cache
    keeps first-run generation working regardless of install layout.
    """
    target = SCRIPT_DIR / local_rel_path
    # ``exists()`` follows symlinks, so a dangling committed pointer is treated
    # as absent and falls through to the (idempotent) download below.
    if target.exists():
        return target

    if local_rel_path not in FLAT_MANIFEST:
        raise FileNotFoundError(
            f"{local_rel_path} is not in the weights manifest — can't auto-download."
        )

    # First-download tip: nudge users toward logging in to HF for better limits.
    # No-op if a token is already configured.
    _show_hf_login_tip_once()

    hf_filename = FLAT_MANIFEST[local_rel_path]
    if verbose:
        print(f"  ↓ downloading {hf_filename}  (from {REPO_ID})")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to auto-download weights.\n"
            "Run:  pip install huggingface_hub\n"
            "Or run the install.py script in this directory."
        ) from e

    # Lands in the writable HF cache; load straight from there (no write into
    # the possibly read-only vendored package dir).
    cached = hf_hub_download(repo_id=REPO_ID, filename=hf_filename)
    return Path(cached)


def is_present(local_rel_path: str) -> bool:
    """True if the file exists locally (does not trigger a download)."""
    p = SCRIPT_DIR / local_rel_path
    return p.exists() or p.is_symlink()


def bundle_status(bundle: str) -> tuple[int, int]:
    """Returns (present_count, total_count) for the bundle (including SHARED)."""
    items = DIT_BUNDLES[bundle] + SHARED
    present = sum(1 for rel, _ in items if is_present(rel))
    return present, len(items)
