# SPDX-License-Identifier: Apache-2.0
"""Wan 2.1 / 2.2 video backend (content-farm lane).

Implements :class:`vllm_mlx.video.engine.VideoEngine` on top of
`mlx-video <https://github.com/Blaizzy/mlx-video>`_, which carries an
MLX-native port of Alibaba's Wan DiT + UMT5 + 3D-VAE pipeline. That
package is by the same author as ``mlx-vlm`` and ``mlx-audio``, both of
which rapid-mlx already depends on.

Why Wan and not Wan 2.7: open weights stop at **Wan 2.2**. Wan 2.5, 2.6
and 2.7 are API-only — no HuggingFace weights, no GitHub repo, no
self-hosting. Verified against the ``Wan-AI`` HF org and the
``Wan-Video`` GitHub org. A local inference server can only serve what
has weights, so this backend covers 2.1 and 2.2 and stops there.

Supported (whatever the converted checkpoint contains — mlx-video
auto-detects from ``config.json`` + weight shapes):

===================== ========= ============ ==========================
variant               params    pipeline     native
===================== ========= ============ ==========================
Wan2.1 T2V-1.3B       1.3B      single        480p, 16 fps
Wan2.1 T2V-14B        14B       single        720p, 16 fps
Wan2.2 TI2V-5B        5B        single        720p, 24 fps
Wan2.2 T2V-A14B       27B/14B   dual (MoE)    720p, 24 fps
Wan2.2 I2V-A14B       27B/14B   dual (MoE)    720p, 24 fps
===================== ========= ============ ==========================

DEPENDENCY WARNING — do NOT add ``mlx-video`` to a pip extra. The name
``mlx-video`` on PyPI belongs to an UNRELATED project
(``AmiraniLabs/mlx-video``, a 5 KB video *loading* utility). Installing
that satisfies the import name and then fails at call time in a
thoroughly confusing way. The package this backend needs is only
installable from git, which PyPI forbids as a direct reference in
published metadata — hence the runtime probe below instead of an extra.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Env var pointing at a converted MLX Wan model directory (the layout
#: mlx-video's ``convert.py`` produces: ``model.safetensors`` or
#: ``{high,low}_noise_model.safetensors``, plus ``t5_encoder.safetensors``,
#: ``vae.safetensors`` and ``config.json``).
#:
#: Deliberately explicit opt-in rather than an alias table pointing at
#: pre-converted community uploads. Those exist and work, but silently
#: fetching multi-GB weights from an unvetted third-party HF account on a
#: user's first request is a supply-chain decision, not a convenience.
#: The operator names the directory they trust.
ENV_MODEL_DIR = "RAPID_MLX_WAN_MODEL_DIR"

#: Optional generation overrides. Unset → the checkpoint's own config
#: defaults (Wan2.1: 50 steps / shift 5.0 / guide 5.0; Wan2.2: 40 steps /
#: shift 12.0 / guide 3.0,4.0; TI2V-5B: 40 / 5.0 / 5.0).
ENV_STEPS = "RAPID_MLX_WAN_STEPS"
ENV_SCHEDULER = "RAPID_MLX_WAN_SCHEDULER"  # euler | dpm++ | unipc
ENV_TILING = "RAPID_MLX_WAN_TILING"  # auto | none | aggressive | ...
#: Wan2.2-Lightning step-distilled LoRAs. These are the single biggest
#: lever on wall-clock: measured on an M3 Ultra, 832x480 x 2 s went from
#: 295 s at the default 40 steps to 77 s at 8 steps. A 4-step Lightning
#: LoRA takes it further. Dual-model checkpoints need both halves.
ENV_LORA = "RAPID_MLX_WAN_LORA"  # path[:strength][,path[:strength]]
ENV_LORA_HIGH = "RAPID_MLX_WAN_LORA_HIGH"
ENV_LORA_LOW = "RAPID_MLX_WAN_LORA_LOW"

#: Wan's latent temporal stride is 4, so a clip's frame count must be
#: ``4n+1`` — one anchor frame plus n groups of 4. Anything else makes
#: mlx-video raise deep in latent packing.
_FRAME_MULTIPLE = 4


def _valid_frame_counts_around(n: int) -> tuple[int, int]:
    """Nearest valid ``4n+1`` frame counts at or below / above ``n``."""
    k = max(0, (n - 1) // _FRAME_MULTIPLE)
    lower = k * _FRAME_MULTIPLE + 1
    upper = (k + 1) * _FRAME_MULTIPLE + 1
    return lower, upper


def _parse_loras(spec: str | None) -> list[tuple[str, float]] | None:
    """Parse ``path[:strength][,path[:strength]]`` into mlx-video's form.

    Strength defaults to 1.0. Windows-style drive letters aren't a
    concern here (macOS-only runtime), but a path CAN contain ``:`` — so
    split from the right and only treat the tail as a strength when it
    parses as a float.
    """
    if not spec:
        return None
    out: list[tuple[str, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        path, sep, tail = part.rpartition(":")
        if sep and path:
            try:
                out.append((path, float(tail)))
                continue
            except ValueError:
                pass  # ':' was part of the path, not a strength
        out.append((part, 1.0))
    return out or None


def probe_mlx_video() -> str | None:
    """Return ``None`` if mlx-video is importable, else why it isn't.

    Distinguishes "not installed" from "the WRONG mlx-video is installed"
    — the PyPI name collision described in the module docstring is the
    single most likely way this backend fails on a user's machine, and a
    bare ``ModuleNotFoundError`` would send them to install exactly the
    wrong package.
    """
    try:
        import mlx_video  # noqa: F401
    except ImportError:
        return (
            "mlx-video is not installed. Install the video-generation "
            "package from git:\n"
            "    pip install git+https://github.com/Blaizzy/mlx-video.git\n"
            "NOTE: do NOT `pip install mlx-video` — that PyPI name is an "
            "unrelated video-loading utility, not the generation package."
        )
    try:
        from mlx_video.models.wan_2.generate import generate_video  # noqa: F401
    except ImportError:
        return (
            "an `mlx_video` module is importable but has no Wan pipeline "
            "(`mlx_video.models.wan_2`). The PyPI package named "
            "`mlx-video` is an unrelated video-loading utility; this "
            "backend needs Prince Canuma's generation package:\n"
            "    pip uninstall mlx-video && "
            "pip install git+https://github.com/Blaizzy/mlx-video.git"
        )
    return None


class WanVideoEngine:
    """:class:`~vllm_mlx.video.engine.VideoEngine` over mlx-video's Wan pipeline.

    One instance wraps one converted checkpoint directory. Construction is
    cheap — mlx-video loads weights inside ``generate_video``, so there is
    no persistent model held here. (That also means every request pays the
    load; a resident-model variant is a follow-up once we know whether the
    lane sees enough traffic to justify pinning ~18 GB.)
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        steps: int | None = None,
        scheduler: str = "unipc",
        tiling: str = "auto",
        loras: list[tuple[str, float]] | None = None,
        loras_high: list[tuple[str, float]] | None = None,
        loras_low: list[tuple[str, float]] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise ValueError(
                f"Wan model directory does not exist: {self.model_dir}. "
                f"Set ${ENV_MODEL_DIR} to a converted MLX Wan checkpoint."
            )
        self._steps = steps
        self._scheduler = scheduler
        self._tiling = tiling
        self._loras = loras
        self._loras_high = loras_high
        self._loras_low = loras_low
        self._config = self._read_config()

    def _read_config(self) -> dict:
        """Read the checkpoint's ``config.json``, tolerating its absence.

        mlx-video can auto-detect the variant from weight shapes when the
        file is missing, so a missing config is not fatal — it only costs
        us the native-fps hint below.
        """
        path = self.model_dir / "config.json"
        if not path.is_file():
            logger.info(
                "Wan checkpoint %s has no config.json; mlx-video will "
                "auto-detect the variant from weight shapes",
                self.model_dir,
            )
            return {}
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read %s: %s", path, e)
            return {}

    @property
    def native_frame_rate(self) -> float:
        """The fps the checkpoint was TRAINED at (16 for 2.1, 24 for 2.2).

        Wan does not take fps as a generation parameter — the model emits
        frames at a fixed rate and fps is purely a container property. The
        route reads this so the response reports the clip's real playback
        rate instead of echoing back a ``frame_rate`` the generator never
        honoured.
        """
        return float(self._config.get("sample_fps") or 24)

    @property
    def served_model(self) -> str:
        """Identifier for the checkpoint that actually ran, e.g. ``wan2.2-ti2v``.

        The route echoes this instead of the request's ``model`` field.
        Reporting back ``"ltx-2.3"`` (our schema default) on a clip that a
        Wan checkpoint rendered is worse than useless to a client trying to
        attribute a result — same reasoning as :attr:`native_frame_rate`.

        Falls back to the directory name when the checkpoint has no
        ``config.json`` to describe itself.
        """
        version = self._config.get("model_version")
        kind = self._config.get("model_type")
        if version and kind:
            return f"wan{version}-{kind}"
        if version:
            return f"wan{version}"
        return self.model_dir.name

    @property
    def max_area(self) -> int:
        """Pixel-area ceiling from the checkpoint (0 = unconstrained).

        TI2V-5B declares 901120 (= 704x1280). Exceeding it doesn't fail
        cleanly inside the pipeline, so the guard lives here.
        """
        try:
            return int(self._config.get("max_area") or 0)
        except (TypeError, ValueError):
            return 0

    def generate(
        self,
        prompt: str,
        out_path: str | Path,
        *,
        image: str | None = None,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 25.0,
        steps: int | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Path:
        """Render ``prompt`` to an mp4 at ``out_path``. Returns the path.

        ``frame_rate`` is accepted for Protocol conformance but NOT used:
        see :attr:`native_frame_rate`. Everything else maps onto
        mlx-video's ``generate_video``.

        Raises:
            ValueError: for caller-fixable requests — a frame count that
                isn't ``4n+1``, or a resolution over the checkpoint's
                area ceiling. The route turns these into a 400 rather
                than letting them surface as an opaque 500.
        """
        err = probe_mlx_video()
        if err is not None:
            raise ImportError(err)

        if num_frames % _FRAME_MULTIPLE != 1:
            lower, upper = _valid_frame_counts_around(num_frames)
            raise ValueError(
                f"num_frames must be 4n+1 for Wan (latent temporal stride "
                f"is 4); got {num_frames}. Nearest valid values: "
                f"{lower} or {upper}."
            )

        area_cap = self.max_area
        if area_cap and width * height > area_cap:
            raise ValueError(
                f"{width}x{height} is {width * height} pixels, over this "
                f"checkpoint's {area_cap}-pixel ceiling "
                f"(e.g. 1280x704). Reduce width/height."
            )

        if abs(frame_rate - self.native_frame_rate) > 0.5:
            # Once per request, at info: the caller asked for something the
            # generator structurally cannot vary, and silently ignoring it
            # would leave them wondering why playback speed never changes.
            logger.info(
                "Wan generates at a fixed %.0f fps; requested frame_rate=%.1f "
                "is ignored (fps is a container property, not a generation "
                "parameter for this model family)",
                self.native_frame_rate,
                frame_rate,
            )

        from mlx_video.models.wan_2.generate import generate_video

        out_path = Path(out_path)
        # mlx-video treats seed=-1 as "random"; our Protocol uses None.
        generate_video(
            model_dir=str(self.model_dir),
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps if steps is not None else self._steps,
            seed=-1 if seed is None else seed,
            output_path=str(out_path),
            scheduler=self._scheduler,
            tiling=self._tiling,
            loras=self._loras,
            loras_high=self._loras_high,
            loras_low=self._loras_low,
        )
        return out_path


def build_engine_from_env(model: str) -> WanVideoEngine:
    """Construct a :class:`WanVideoEngine` from environment configuration.

    ``model`` (the request's ``model`` field) is accepted for the factory
    signature and logged, but does NOT select a checkpoint: rapid-mlx
    serves one model per process, matching how the LLM lane works. The
    served checkpoint is whatever ``$RAPID_MLX_WAN_MODEL_DIR`` names.
    """
    model_dir = os.environ.get(ENV_MODEL_DIR)
    if not model_dir:
        raise NotImplementedError(
            f"no video backend configured. Set ${ENV_MODEL_DIR} to a "
            "converted MLX Wan checkpoint directory to serve "
            "/v1/video/generations. See docs/content_farm_api.md."
        )
    steps_raw = os.environ.get(ENV_STEPS)
    steps = None
    if steps_raw:
        try:
            steps = int(steps_raw)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r", ENV_STEPS, steps_raw)
    logger.info("Serving video request for model=%r from %s", model, model_dir)
    return WanVideoEngine(
        model_dir,
        steps=steps,
        scheduler=os.environ.get(ENV_SCHEDULER, "unipc"),
        tiling=os.environ.get(ENV_TILING, "auto"),
        loras=_parse_loras(os.environ.get(ENV_LORA)),
        loras_high=_parse_loras(os.environ.get(ENV_LORA_HIGH)),
        loras_low=_parse_loras(os.environ.get(ENV_LORA_LOW)),
    )


def register() -> bool:
    """Install this backend as the video-lane factory. Returns whether it took.

    Registration is intentionally unconditional on mlx-video being
    present: the factory itself probes at call time so an operator who
    configured ``$RAPID_MLX_WAN_MODEL_DIR`` but hasn't installed
    mlx-video gets the actionable 503 from :func:`probe_mlx_video`
    instead of a bare 501 that reads as "rapid-mlx has no video support".
    """
    from . import engine as engine_mod

    if not os.environ.get(ENV_MODEL_DIR):
        return False
    engine_mod._VIDEO_ENGINE_FACTORY = build_engine_from_env
    return True
