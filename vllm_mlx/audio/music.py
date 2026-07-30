# SPDX-License-Identifier: Apache-2.0
"""
Music (and SFX) generation engine using a vendored, MLX-native Stable Audio 3.

This is the third audio lane alongside ``TTSEngine`` (text→speech) and
``STTEngine`` (speech→text): ``MusicEngine`` does text→music/SFX, fully local on
Apple Silicon. The model (Stability AI's Stable Audio 3) ships an official
torch-free MLX implementation, vendored under ``vllm_mlx/audio/sa3/`` — so this
adds **no** heavy dependency (only ``mlx``/``numpy``/``sentencepiece``/
``soundfile``, all already required) and the weights are fetched from HF at
first use (Stability Community License; free commercial use under $1M rev).

MVP note: ``generate`` drives the vendored ``sa3_mlx.py`` in a subprocess (its
prompt→audio flow currently lives in that script's ``main``). Lifting that flow
into an in-process call is a follow-up; the public API here is stable so the
internals can change without touching callers.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

_SA3_DIR = Path(__file__).parent / "sa3"
_SA3_CLI = _SA3_DIR / "scripts" / "sa3_mlx.py"
_SA3_MLX_DIR = _SA3_DIR / "models" / "mlx"

# DiT / decoder pairings (from the SA3 spec). sm-music/same-s = fast small
# (~4x realtime, ~1.7GB); medium/same-l = higher quality (~3.9GB peak).
DEFAULT_DIT = "medium"
DEFAULT_DECODER = "same-l"

# HuggingFace repo holding the real (LFS) SA3 weights. The vendored
# ``sa3/models/mlx/`` dir ships no tensors — they are fetched from here on first
# use into the writable HF cache and loaded from there (never written back into
# the possibly read-only package dir). Repo layout puts every MLX file under
# ``MLX/`` (mirrors the repo→local map in ``sa3/scripts/weights.py``).
_SA3_REPO_ID = "stabilityai/stable-audio-3-optimized"

# Selected-component → weight filename (basename shared by the local vendored
# path ``sa3/models/mlx/<name>`` and the repo path ``MLX/<name>``).
_DIT_NPZ = {
    "medium": "dit_medium_f16.npz",
    "sm-music": "dit_sm-music_f16.npz",
    "sm-sfx": "dit_sm-sfx_f16.npz",
}
_DECODER_NPZ = {
    "same-l": "same_l_decoder_f32.npz",
    "same-s": "same_s_decoder_f32.npz",
}
# T5Gemma text conditioner — shared by every dit/decoder combination.
_SHARED_NPZ = ("t5gemma_f16.npz",)

# Stable Audio 3 is trained/served for clips up to ~47 s (the DiT positional
# grid and the schedule are sized for it); longer requests degrade or blow up
# memory. Enforce a sane positive range before doing any work.
_MAX_SECONDS = 47.0


class MusicEngine:
    """Text-to-music / text-to-SFX via MLX-native Stable Audio 3.

    Usage:
        eng = MusicEngine()                       # medium/same-l
        eng.generate("epic cinematic war drums", "bgm.wav", seconds=36)
        # SFX: eng = MusicEngine(dit="sm-sfx", decoder="same-s")
    """

    def __init__(self, dit: str = DEFAULT_DIT, decoder: str = DEFAULT_DECODER):
        self.dit = dit
        self.decoder = decoder

    def _ensure_weights(self) -> dict[str, Path]:
        """Resolve the SA3 weights for the selected dit/decoder, returning the
        absolute path of each component file.

        Any component not already present as a real vendored file is fetched
        from HuggingFace into the writable HF cache (``~/.cache/huggingface``)
        and its cache path is returned. We deliberately do NOT copy/symlink the
        download into the vendored ``sa3/models/mlx/`` directory: under a
        pip/brew install that directory lives inside a read-only
        ``site-packages`` tree, so writing there raises ``PermissionError`` and
        first generation would crash. The SA3 runner loads the weights from the
        returned cache paths (it re-resolves them via the same manifest, an
        idempotent ``hf_hub_download`` cache hit), so nothing is ever written
        into the package directory.
        """
        try:
            dit_npz = _DIT_NPZ[self.dit]
        except KeyError as e:
            raise ValueError(
                f"unknown dit {self.dit!r}; expected one of {sorted(_DIT_NPZ)}"
            ) from e
        try:
            decoder_npz = _DECODER_NPZ[self.decoder]
        except KeyError as e:
            raise ValueError(
                f"unknown decoder {self.decoder!r}; "
                f"expected one of {sorted(_DECODER_NPZ)}"
            ) from e

        needed = (dit_npz, decoder_npz, *_SHARED_NPZ)

        resolved: dict[str, Path] = {}
        to_fetch = []
        for name in needed:
            vendored = _SA3_MLX_DIR / name
            # A real (materialized) vendored file — a dev checkout — is used as
            # is. ``exists()`` follows symlinks, so a dangling committed pointer
            # counts as absent and is fetched from the cache instead.
            if vendored.exists():
                resolved[name] = vendored
            else:
                to_fetch.append(name)

        if to_fetch:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise RuntimeError(
                    "huggingface_hub is required to auto-download SA3 weights.\n"
                    "Run:  pip install huggingface_hub"
                ) from e
            for name in to_fetch:
                cached = hf_hub_download(repo_id=_SA3_REPO_ID, filename=f"MLX/{name}")
                resolved[name] = Path(cached)

        return resolved

    def generate(
        self,
        prompt: str,
        out_path: str | Path,
        seconds: float = 30.0,
        steps: int = 8,
        negative_prompt: str | None = None,
        seed: int | None = None,
        timeout: int = 900,
    ) -> Path:
        """Generate ``seconds`` of audio for ``prompt`` → ``out_path`` (wav).

        Args:
            prompt: Natural-language description of the music/SFX.
            out_path: Output wav path. Relative paths are resolved against the
                caller's cwd (see the ``--out`` note below).
            seconds: Length in seconds (SA3 supports up to ~47s).
            steps: Pingpong sampling steps (8 is a good fast default).
            negative_prompt: CFG negative branch (e.g. "vocals, singing").
            seed: Fixed seed for reproducibility (None = random per run).
            timeout: Seconds to wait for the generation subprocess.
        Returns:
            The output Path (always absolute).

        Raises:
            ValueError: if ``seconds`` is non-finite, <= 0, or > ~47 (the SA3
                supported maximum).
        """
        # Validate duration up front — before deleting the destination or
        # launching the subprocess. ``float(... )`` also rejects non-numeric
        # input; NaN/inf slip past a plain range check, so test finiteness
        # explicitly (Pydantic's ``ge=/le=`` has the same NaN gap).
        try:
            seconds = float(seconds)
        except (TypeError, ValueError) as e:
            raise ValueError(f"seconds must be a number, got {seconds!r}") from e
        if not math.isfinite(seconds) or seconds <= 0 or seconds > _MAX_SECONDS:
            raise ValueError(
                f"seconds must be in (0, {_MAX_SECONDS:g}] "
                f"(Stable Audio 3's supported range); got {seconds!r}"
            )
        # ``sa3_mlx.py`` re-roots any RELATIVE ``--out`` under its own vendored
        # ``sa3/output/`` directory, which for an installed wheel is inside
        # site-packages. That both hides the wav from the caller and makes the
        # post-run existence check below fail on a successful generation, so
        # resolve to an absolute path against the caller's cwd up front.
        #
        # Resolve only the PARENT: ``Path.resolve()`` on the full path would
        # dereference a final-component symlink, so a subsequent ``unlink()``
        # would delete the symlink's *target* (and generation would write
        # through the link) instead of replacing the link itself.
        _p = Path(out_path).expanduser()
        out_path = _p.parent.resolve() / _p.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear any previous entry at the target so the post-run existence check
        # below cannot mistake a stale wav for a successful generation (the
        # child could exit 0 without writing anything). ``is_symlink()`` is
        # checked first so we unlink the LINK, never follow it to a target.
        if out_path.is_symlink() or out_path.exists():
            out_path.unlink()
        self._ensure_weights()
        # ``--flag=value`` form so a prompt that begins with "-" is never
        # mistaken for an option by the child's argparse.
        cmd = [
            sys.executable,
            str(_SA3_CLI),
            f"--prompt={prompt}",
            f"--dit={self.dit}",
            f"--decoder={self.decoder}",
            f"--steps={steps}",
            # Forward the full float (round-trippable ``str(float)``) — a fixed
            # 2-decimal format would collapse small valid durations (e.g.
            # 0.003s) to ``0.00`` and generate an empty clip.
            f"--seconds={seconds!r}",
            f"--out={out_path}",
        ]
        if negative_prompt:
            cmd.append(f"--negative-prompt={negative_prompt}")
        if seed is not None:
            cmd.append(f"--seed={seed}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"MusicEngine (SA3) generation exceeded {timeout}s "
                f"(prompt={prompt[:60]!r}, seconds={seconds})"
            ) from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(
                f"MusicEngine (SA3) generation failed "
                f"(exit {e.returncode}): {stderr[-400:] or '<no stderr>'}"
            ) from e
        if not out_path.exists():
            raise RuntimeError(f"MusicEngine produced no output at {out_path}")
        return out_path
