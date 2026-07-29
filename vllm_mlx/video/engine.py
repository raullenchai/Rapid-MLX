# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 (MLX-native) backend for the rapid-mlx video lane.

``VideoEngine`` is the fourth generation lane alongside ``TTSEngine``
(text→speech), ``STTEngine`` (speech→text) and ``MusicEngine``
(text→music/SFX): it does **text→video** AND **image→video**, fully local
on Apple Silicon. The backend is `dgrauet/ltx-2-mlx <https://github.com/dgrauet/ltx-2-mlx>`_
— a pure-MLX port of LTX-2.3 with **no torch dependency** — which emits an
mp4 with a natively-synchronized audio track. It runs free-human-motion t2v
clips and identity-locking i2v (animate a still) on a 32 GB Mac mini using the
q4 distilled variant.

This concrete engine is meant to satisfy the ``VideoEngine`` Protocol defined
by the ``/v1/video/generations`` route contract (PR #1300 / branch
``feat/openai-routes-content-farm``): the route layer depends only on that
Protocol, so registering this backend takes the lane from contract-only to
live with zero handler changes. See ``REQUIREMENTS_rapid.md`` sections B1/B2
for the design intent.

MVP note (mirrors ``MusicEngine``): ``generate`` drives the ``ltx-2-mlx``
CLI in a **subprocess** rather than calling the pipeline in-process. Lifting
that flow into an in-process (vendored) call — and wiring the registry factory
so ``resolve_video_engine`` returns this backend — is the follow-up. The
public API here is stable so the internals can change without touching
callers.

The ``ltx-2-mlx`` package is an OPTIONAL dependency: install it with
``pip install '.[video]'``. It is not part of the base install (the weights
and Metal pipelines are heavy), so importing this module is cheap and the
missing-dependency error is raised only when :meth:`VideoEngine.generate`
is actually called.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# LTX-2.3 model variants published by the upstream port. The q4 distilled
# variant is the sensible default for a 32 GB Mac mini (fits with block
# streaming + distilled two-stage, no CFG). Point ``model`` at a q8 repo /
# local dir for higher quality on machines with more RAM.
DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q4"

# The console-script the ``ltx-2-mlx`` package installs (see its
# ``[project.scripts]``). We shell out to this rather than importing the
# pipeline module so the heavy import cost is paid only per generation and
# only when the extra is installed.
_LTX_CLI = "ltx-2-mlx"

_INSTALL_HINT = (
    "The 'ltx-2-mlx' CLI was not found on PATH. The LTX-2.3 video backend is "
    "an optional dependency — install it with:  pip install '.[video]'  (or "
    "`pip install ltx-2-mlx`). It is a pure-MLX package (no torch) and pulls "
    "heavy weights, so it is deliberately kept out of the base install."
)


class VideoEngine:
    """Text→video / image→video via the MLX-native LTX-2.3 backend.

    Usage::

        eng = VideoEngine()                                  # q4 distilled
        # text-to-video
        eng.generate("a fox trotting through snow", "clip.mp4")
        # image-to-video (animate a still, locking identity)
        eng.generate("the same woman turns and smiles", "shot.mp4",
                     image="lead_ref.png")

    Satisfies the ``VideoEngine`` Protocol used by ``/v1/video/generations``.
    """

    def __init__(self, model: str = DEFAULT_MODEL, distilled: bool = True):
        """Args:
        model: HF repo id or local dir of the LTX-2.3 MLX weights.
        distilled: Use the distilled two-stage pipeline (fastest, no CFG) —
            the practical default on a 32 GB mini. Set ``False`` to let the
            CLI's own pipeline default apply.
        """
        self.model = model
        self.distilled = distilled

    def generate(
        self,
        prompt: str,
        out_path: str | Path,
        *,
        image: str | None = None,
        height: int = 768,
        width: int = 448,
        num_frames: int = 97,
        frame_rate: float = 24.0,
        steps: int | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        low_ram: bool = True,
        timeout: int = 3600,
    ) -> Path:
        """Render a clip for ``prompt`` → ``out_path`` (mp4). Returns the path.

        Args:
            prompt: Natural-language description of the video.
            out_path: Output mp4 path (absolute recommended).
            image: Conditioning first frame for **image-to-video** — a local
                path to a still. ``None`` selects **text-to-video**. (The
                subprocess CLI takes a filesystem path; a route layer that
                receives a base64/URL image is responsible for materialising
                it to a temp file before calling this method.)
            height: Output height in pixels (default 768 — portrait 竖屏).
            width: Output width in pixels (default 448).
            num_frames: Number of frames to render (default 97 ≈ 4 s @ 24 fps).
            frame_rate: Playback frame rate; LTX-2.3 is trained at 24 fps —
                values far from that drift out of distribution.
            steps: Denoising steps (``None`` = backend default).
            negative_prompt: CFG negative branch. Accepted for Protocol
                compatibility but NOT yet forwarded — the ``ltx-2-mlx``
                ``generate`` subcommand exposes no ``--negative-prompt`` flag,
                and the distilled default pipeline runs without CFG. Wiring
                this (via a CFG mode) is a follow-up.
            seed: Fixed seed for reproducibility (``None`` = random per run).
            low_ram: Stream transformer blocks from disk (``--low-ram``) to
                cap peak Metal memory — required to fit a 32 GB mini.
            timeout: Subprocess timeout in seconds (video gen is minutes-long).

        Returns:
            The output ``Path`` (an mp4 with a native-audio track).
        """
        if shutil.which(_LTX_CLI) is None:
            raise ImportError(_INSTALL_HINT)

        out_path = Path(out_path)
        cmd: list[str] = [
            _LTX_CLI,
            "generate",
            "--prompt",
            prompt,
            "--model",
            self.model,
            "--frame-rate",
            f"{frame_rate:g}",
            "--height",
            str(height),
            "--width",
            str(width),
            "--frames",
            str(num_frames),
            "--output",
            str(out_path),
        ]
        if self.distilled:
            cmd.append("--distilled")
        if low_ram:
            cmd.append("--low-ram")
        if image is not None:
            # I2V: anchor the first frame on the reference still (locks
            # identity). PATH alone => FRAME_IDX=0 STRENGTH=1.0.
            cmd += ["--image", str(image)]
        if steps is not None:
            cmd += ["--steps", str(steps)]
        # NOTE: ``negative_prompt`` is intentionally not forwarded — the
        # ``ltx-2-mlx generate`` subcommand has no --negative-prompt flag and
        # the distilled pipeline is CFG-free. Kept in the signature for
        # VideoEngine-Protocol compatibility.
        if seed is not None:
            cmd += ["--seed", str(seed)]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except FileNotFoundError as e:  # CLI vanished between which() and run()
            raise ImportError(_INSTALL_HINT) from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode()[-600:] if e.stderr else str(e)
            raise RuntimeError(
                f"VideoEngine (LTX-2.3) generation failed:\n{stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"VideoEngine (LTX-2.3) timed out after {timeout}s for {out_path}"
            ) from e

        if not out_path.exists():
            raise RuntimeError(f"VideoEngine produced no output at {out_path}")
        return out_path


def generate_video(
    prompt: str,
    out_path: str | Path,
    *,
    image: str | None = None,
    model: str = DEFAULT_MODEL,
    **kwargs,
) -> Path:
    """One-shot convenience wrapper: build a :class:`VideoEngine` and run it.

    Mirrors ``generate_speech`` / ``transcribe_audio`` in the audio lane.
    """
    return VideoEngine(model=model).generate(prompt, out_path, image=image, **kwargs)
