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

import subprocess
import sys
from pathlib import Path

_SA3_CLI = Path(__file__).parent / "sa3" / "scripts" / "sa3_mlx.py"

# DiT / decoder pairings (from the SA3 spec). sm-music/same-s = fast small
# (~4x realtime, ~1.7GB); medium/same-l = higher quality (~3.9GB peak).
DEFAULT_DIT = "medium"
DEFAULT_DECODER = "same-l"


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
            out_path: Output wav path (absolute recommended).
            seconds: Length in seconds (SA3 supports up to ~47s).
            steps: Pingpong sampling steps (8 is a good fast default).
            negative_prompt: CFG negative branch (e.g. "vocals, singing").
            seed: Fixed seed for reproducibility (None = random per run).
            quantize: Weight quantization (3/4/5/6/8).
        Returns:
            The output Path.
        """
        out_path = Path(out_path)
        cmd = [
            sys.executable, str(_SA3_CLI),
            "--prompt", prompt,
            "--dit", self.dit, "--decoder", self.decoder,
            "--steps", str(steps), "--seconds", f"{seconds:.2f}",
            "--out", str(out_path),
        ]
        if negative_prompt:
            cmd += ["--negative-prompt", negative_prompt]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"MusicEngine (SA3) generation failed: "
                f"{e.stderr.decode()[-400:] if e.stderr else e}"
            ) from e
        if not out_path.exists():
            raise RuntimeError(f"MusicEngine produced no output at {out_path}")
        return out_path
