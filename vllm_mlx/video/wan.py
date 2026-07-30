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
import math
import os
from pathlib import Path

from .engine import InvalidVideoRequestError, VideoBackendUnavailableError

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

#: Alias → converted MLX Wan checkpoint on the Hub, with a PINNED revision.
#:
#: Without this table the lane is unusable without hand-converting weights
#: (download the PyTorch release, run mlx-video's ``convert.py``, point an
#: env var at the output) — a bar high enough that nobody would.
#:
#: Every entry is pinned to an exact commit, which is the part that makes
#: this defensible. These are COMMUNITY conversions, not first-party
#: artifacts: Alibaba publishes PyTorch weights, mlx-video needs its own
#: layout, and third parties did that work. A pin means an account cannot
#: silently swap the bytes under a pinned rapid-mlx release, and it makes a
#: download reproducible. Upgrading a pin is a reviewable diff.
#:
#: All entries are Apache-2.0 with ``base_model`` attribution to
#: ``Wan-AI/*``, and the layout of each was verified against mlx-video's
#: expected file set. Only ``wan2.2-ti2v-5b`` has been rendered end-to-end
#: by us (M3 Ultra, real weights) — the others are structurally verified,
#: not quality-verified, and the table says which is which.
WAN_ALIASES: dict[str, dict[str, str]] = {
    # VERIFIED end-to-end on an M3 Ultra: t2v and i2v both render.
    "wan2.2-ti2v-5b": {
        "repo": "Anes1032/Wan2.2-TI2V-5B-mlx-q8",
        "revision": "9624723c94ddf509832555c45e223a035baa7d1c",
        "size": "18 GB",
        "notes": "8-bit, 720p/24fps, single-model. Verified end-to-end.",
    },
    "wan2.2-ti2v-5b-bf16": {
        "repo": "rickylin20260522/Wan2.2-TI2V-5B-mlx",
        "revision": "592b2473f27cd6f466cdd9f2c0f5750a77b37b59",
        "size": "23 GB",
        "notes": "bf16, 720p/24fps, single-model. Layout verified.",
    },
    "wan2.2-i2v-a14b": {
        "repo": "Anes1032/Wan2.2-I2V-A14B-mlx-q8",
        "revision": "633f50fc3e16e7faf76713dcf07b0bea730f02c9",
        "size": "40 GB",
        "notes": "8-bit dual-model MoE, image-to-video. Layout verified.",
    },
    "wan2.2-t2v-a14b": {
        "repo": "rickylin20260522/Wan2.2-T2V-A14B-mlx",
        "revision": "225358452f995a6807acaebff9dfc4976c39c8c8",
        "size": "64 GB",
        "notes": "bf16 dual-model MoE, text-to-video. Layout verified.",
    },
}

#: Env var naming an alias from :data:`WAN_ALIASES`, or a bare HF repo id
#: (optionally ``repo@revision``). Downloaded on first use into the normal
#: HuggingFace cache. Mutually exclusive with :data:`ENV_MODEL_DIR`, which
#: still takes a local directory and wins when both are set.
ENV_MODEL = "RAPID_MLX_WAN_MODEL"


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

#: Solver names mlx-video's Wan pipeline accepts.
_SCHEDULERS: frozenset[str] = frozenset({"euler", "dpm++", "unipc"})

#: VAE-decode tiling modes mlx-video's Wan pipeline accepts.
_TILING_MODES: frozenset[str] = frozenset(
    {"auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"}
)


def _validated_choice(
    value: str, allowed: frozenset[str], env_var: str, label: str
) -> str:
    """Return ``value`` if it's an accepted option, else fail at construction.

    A typo in these reaches the renderer and becomes a generic 500 —
    potentially *after* a multi-GB weight load, which is an expensive way to
    learn about a misspelling. Checking here surfaces it as a 503 naming the
    variable, before anything is loaded.
    """
    if value not in allowed:
        raise VideoBackendUnavailableError(
            f"invalid {label} {value!r} in ${env_var}: expected one of "
            f"{', '.join(sorted(allowed))}"
        )
    return value


#: Largest value ``_parse_loras`` will read as a strength rather than as part
#: of the path. LoRA scales are conventionally 0..2; anything larger in a
#: trailing ``:N`` is much more likely to be a date or port in a directory
#: name.
_MAX_LORA_STRENGTH = 4.0

#: Wan's latent temporal stride is 4, so a clip's frame count must be
#: ``4n+1`` — one anchor frame plus n groups of 4. Anything else makes
#: mlx-video raise deep in latent packing.
_FRAME_MULTIPLE = 4

#: Upper bound on ``steps``, mirroring ``VideoGenerationRequest.steps`` so an
#: env override can't smuggle in a value the HTTP contract would reject.
_MAX_STEPS = 500

#: Mirrors ``VideoGenerationRequest.num_frames``' ceiling, so the "nearest
#: valid frame count" hint never suggests a value the schema would reject.
_MAX_FRAMES = 4096


def _valid_frame_counts_around(n: int) -> tuple[int, int | None]:
    """Nearest valid ``4n+1`` frame counts at or below / above ``n``.

    The upper suggestion is ``None`` when it would exceed the request
    schema's own ceiling — recommending 4097 to a caller whose request can
    never carry more than 4096 is advice they cannot take.
    """
    k = max(0, (n - 1) // _FRAME_MULTIPLE)
    lower = k * _FRAME_MULTIPLE + 1
    upper = (k + 1) * _FRAME_MULTIPLE + 1
    return lower, (upper if upper <= _MAX_FRAMES else None)


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
                strength = float(tail)
            except ValueError:
                pass  # ':' was part of the path, not a strength
            else:
                # Bound it: LoRA strengths are conventionally 0..2, so a
                # trailing number outside that range is far more likely to be
                # part of the path (``/models/run:2026``, a port, a date) than
                # a scale factor. Without this, that path silently becomes
                # ``/models/run`` at strength 2026.
                if 0.0 <= strength <= _MAX_LORA_STRENGTH:
                    out.append((path, strength))
                    continue
                logger.warning(
                    "Treating %r as part of the LoRA path: %s is outside the "
                    "0..%g strength range",
                    part,
                    tail,
                    _MAX_LORA_STRENGTH,
                )
        out.append((part, 1.0))
    return out or None


#: Canonical install target. Pinned because ``mlx-video``'s PyPI name
#: belongs to an unrelated project, so there is no versioned release to
#: depend on and an unpinned ``main`` could change ``generate_video``'s
#: signature under a working install. Kept in one place so the docs and
#: every runtime message agree — they drifted once already.
MLX_VIDEO_PIN = "87db56a51758fefb748a359b90a5283bb8ba4837"
MLX_VIDEO_INSTALL = (
    f"pip install 'git+https://github.com/Blaizzy/mlx-video.git@{MLX_VIDEO_PIN}'"
)

#: Pixel ceiling for a conditioning frame, checked from the header BEFORE
#: decoding. The 12 MB cap on the base64 string bounds the *compressed*
#: size, not the decompressed one — a small PNG can declare a 30000x30000
#: canvas and cost gigabytes on ``load()``. 64 MP is far above any sane
#: input (Wan crops and resizes to the requested output size regardless)
#: and far below anything that threatens the process.
_MAX_IMAGE_PIXELS = 64 * 1024 * 1024

#: Magic-byte prefixes for the raster formats PIL will open. Used instead of
#: ``PIL.Image.open`` because Pillow lives in the ``[vision]`` extra and this
#: check must work on a base install.
_IMAGE_MAGIC: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"GIF87a",
    b"GIF89a",
    b"BM",  # BMP
    b"II*\x00",  # TIFF little-endian
    b"MM\x00*",  # TIFF big-endian
)


def _looks_like_image(raw: bytes) -> bool:
    """True if ``raw`` starts with a known raster-image signature."""
    if any(raw.startswith(sig) for sig in _IMAGE_MAGIC):
        return True
    # WebP: 'RIFF' <4-byte size> 'WEBP'
    return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"


def _decode_error(raw: bytes) -> str | None:
    """Return why ``raw`` isn't loadable image data, or ``None`` if it is.

    A signature check alone is not enough: a TRUNCATED PNG has a perfectly
    valid 8-byte header and then fails inside PIL's chunk reader, which
    surfaced as a `500 video_generation_failed` — blaming the server for a
    corrupt upload. Decoding here moves that to a `400`.

    Uses PIL when importable and falls back to the signature check when it
    isn't. Pillow lives in the ``[vision]`` extra, but mlx-video (this
    backend's own runtime dependency) requires it, so in any environment
    that can actually render, the strong check is the one that runs.
    """
    try:
        import io

        from PIL import Image
    except ImportError:
        return (
            None
            if _looks_like_image(raw)
            else (
                "image payload decoded successfully but is not a recognised "
                "image format (expected PNG, JPEG, GIF, BMP, TIFF or WebP)"
            )
        )

    try:
        # Read the header FIRST and bound the pixel count before decoding
        # anything. A conditioning frame is size-capped at 12 MB of base64,
        # but compressed formats expand: a small PNG can carry a
        # 30000x30000 canvas that costs gigabytes once decompressed. Opening
        # is header-only and cheap; ``load()`` is what allocates.
        probe = Image.open(io.BytesIO(raw))
        w, h = probe.size
        if w * h > _MAX_IMAGE_PIXELS:
            return (
                f"image is {w}x{h} ({w * h} pixels), over the "
                f"{_MAX_IMAGE_PIXELS}-pixel limit for a conditioning frame. "
                "Downscale it — Wan crops and resizes to the requested "
                "output size anyway."
            )
        # verify() checks structure without decoding pixels but leaves the
        # file object unusable, so re-open to force a full load — that is
        # what actually catches truncation.
        Image.open(io.BytesIO(raw)).verify()
        Image.open(io.BytesIO(raw)).load()
    except Exception as e:  # noqa: BLE001 — PIL raises many types here
        return f"image payload is not loadable image data: {e}"
    return None


def _materialise_image(image: str) -> tuple[str, bool]:
    """Turn a contract ``image`` value into a local file path for mlx-video.

    mlx-video's I2V path does ``PIL.Image.open(image)`` — a **filesystem
    path**, nothing else. It does not fetch URLs and does not decode
    base64. Handing it the wire formats our contract advertises
    (``data:image/*;base64,...`` or a bare base64 payload) would have PIL
    treat a multi-megabyte string as a filename and fail, so i2v was
    advertised and broken. This decodes inline payloads to a temp file.

    Returns ``(path, is_temp)`` — the caller unlinks when ``is_temp``.

    Remote ``http(s)`` URLs are REFUSED here rather than fetched. Fetching
    them would make this the server's only outbound-request primitive and
    therefore an SSRF vector (loopback, RFC1918, link-local metadata
    endpoints, DNS rebinding between validation and connect, redirects to
    any of those). Doing that safely needs socket-level control — resolve
    and re-check on every connection and every redirect hop — which is a
    subsystem, not a helper, and one this backend has no way to exercise.
    The schema still accepts URLs because another backend may implement a
    safe loader; this one asks the client to inline the frame instead,
    which it can always do.

    Raises:
        InvalidVideoRequestError: for a remote URL, or a payload that
            isn't decodable image data.
    """
    import base64
    import binascii
    import tempfile
    from urllib.parse import urlsplit

    scheme = urlsplit(image).scheme.lower()
    if scheme in ("http", "https"):
        raise InvalidVideoRequestError(
            "this backend does not fetch remote images: pass the "
            "conditioning frame inline as a data:image/*;base64,... URI or a "
            "bare base64 payload. (Fetching caller-supplied URLs server-side "
            "would be an SSRF vector; see vllm_mlx/video/wan.py.)"
        )

    payload = image
    if image.lower().startswith("data:"):
        # Schema already guaranteed data:image/*;base64, — take the tail.
        _, _, payload = image.partition(",")

    try:
        raw = base64.b64decode("".join(payload.split()), validate=True)
    except (binascii.Error, ValueError) as e:
        raise InvalidVideoRequestError(
            f"image is not decodable base64 image data: {e}"
        ) from e
    if not raw:
        raise InvalidVideoRequestError("image decoded to zero bytes")
    # Valid base64 is not the same as loadable image data: `aGVsbG8=`
    # decodes cleanly to b"hello", and a truncated PNG has a perfectly
    # valid header. Both used to reach mlx-video and come back as a 500
    # video_generation_failed — blaming the server for a caller-fixable
    # input.
    problem = _decode_error(raw)
    if problem is not None:
        raise InvalidVideoRequestError(problem)

    # Suffix matters: PIL sniffs content, but a sensible extension keeps the
    # temp file legible in logs and to anything else that inspects it.
    # Capture the name before writing: NamedTemporaryFile(delete=False)
    # creates the file immediately, so a failing write (disk full, most
    # likely on the machine this runs on) would otherwise strand a
    # rapidmlx-i2v-* file that nobody owns — the caller only learns the
    # path on success.
    fh = tempfile.NamedTemporaryFile(
        delete=False, prefix="rapidmlx-i2v-", suffix=".png"
    )
    try:
        with fh:
            fh.write(raw)
    except Exception:
        try:
            os.unlink(fh.name)
        except OSError:
            logger.warning("Could not remove partial i2v frame %s", fh.name)
        raise
    return fh.name, True


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
            f"    {MLX_VIDEO_INSTALL}\n"
            "NOTE: do NOT `pip install mlx-video` — that PyPI name is an "
            "unrelated video-loading utility, not the generation package."
        )
    try:
        from mlx_video.models.wan_2.generate import generate_video  # noqa: F401
    except ImportError as e:
        # Two very different causes land here and the advice differs, so
        # distinguish them instead of always blaming the PyPI collision:
        #   * the Wan package itself is absent -> wrong distribution
        #   * a TRANSITIVE dependency is absent -> right package, broken env
        # Telling someone to uninstall the correct package because PIL is
        # missing would be actively harmful.
        missing = getattr(e, "name", "") or ""
        if missing.split(".")[0] not in ("", "mlx_video"):
            return (
                f"mlx-video is installed but a dependency is missing: {e}. "
                f"Install it (e.g. `pip install {missing.split('.')[0]}`) — the "
                "Wan pipeline itself is present, so do NOT reinstall mlx-video."
            )
        return (
            "an `mlx_video` module is importable but has no Wan pipeline "
            f"(`mlx_video.models.wan_2`): {e}. The PyPI package named "
            "`mlx-video` is an unrelated video-loading utility; this "
            "backend needs Prince Canuma's generation package:\n"
            f"    pip uninstall mlx-video && {MLX_VIDEO_INSTALL}"
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
            # Raised while the ROUTE is resolving the engine, i.e. outside
            # the generation try/except — so it must be a type the route
            # already maps, or a typo'd env var becomes an unstructured 500.
            #
            # The path goes to the LOG, not the message: the route returns
            # this text to the client, and disclosing the server's
            # filesystem layout is the same leak we refuse for ``url``.
            # Naming the env var is enough for the operator to act, and
            # they can read their own logs for the value.
            logger.error(
                "Configured Wan model directory does not exist: %s", self.model_dir
            )
            raise VideoBackendUnavailableError(
                f"the configured Wan model directory does not exist. Check "
                f"${ENV_MODEL_DIR} points at a converted MLX Wan checkpoint "
                f"(the resolved path is in the server log)."
            )
        self._steps = steps
        self._scheduler = _validated_choice(
            scheduler, _SCHEDULERS, ENV_SCHEDULER, "scheduler"
        )
        self._tiling = _validated_choice(tiling, _TILING_MODES, ENV_TILING, "tiling")
        self._loras = loras
        self._loras_high = loras_high
        self._loras_low = loras_low
        self._config = self._read_config()
        self._warn_if_unguarded()

    def _warn_if_unguarded(self) -> None:
        """Say so, once, when checkpoint metadata leaves a guard inactive.

        Both the fps report and the resolution ceiling come from
        ``config.json``. A checkpoint without it still renders (mlx-video
        auto-detects the variant from weight shapes), so this is a warning
        and not a refusal — but an operator should know that two safety
        rails are off rather than discovering it from a wrong ``frame_rate``
        in a response.
        """
        if self.native_frame_rate is None:
            logger.warning(
                "Checkpoint %s declares no sample_fps: responses will report "
                "frame_rate as null, because the real rate is unknowable here "
                "(Wan2.1 is 16 fps, Wan2.2 is 24, and weight shapes don't "
                "distinguish them)",
                self.model_dir,
            )
        if self.max_area == 0:
            logger.warning(
                "Checkpoint %s declares no max_area: the resolution ceiling "
                "guard is inactive, so an oversized request will fail inside "
                "the pipeline instead of as a clean 400",
                self.model_dir,
            )

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
        # A PRESENT-but-broken config.json is fatal, not ignorable.
        #
        # Ignoring it here was wrong: mlx-video reads the same file from the
        # same directory, and its fallback to weight-shape auto-detection
        # only triggers when config.json is ABSENT. So "ignore and carry on"
        # meant we sailed past a file that would crash the loader seconds
        # later — and the fake-backed tests stayed green because the fake
        # never reads it. Fail here, at construction, with something the
        # operator can act on.
        try:
            with open(path) as fh:
                loaded = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Unreadable Wan config.json at %s: %s", path, e)
            raise VideoBackendUnavailableError(
                "the checkpoint's config.json is present but unreadable "
                f"({type(e).__name__}); repair or remove it. mlx-video "
                "auto-detects the variant from weight shapes when the file "
                "is absent, but cannot recover from a malformed one."
            ) from e
        if not isinstance(loaded, dict):
            # `[]`, `null` and bare scalars all decode as valid JSON and then
            # blow up on `.get()` — here and, identically, inside mlx-video.
            logger.error(
                "Wan config.json at %s is valid JSON but not an object (%s)",
                path,
                type(loaded).__name__,
            )
            raise VideoBackendUnavailableError(
                "the checkpoint's config.json is valid JSON but not an object "
                f"(got {type(loaded).__name__}); repair or remove it."
            )
        return loaded

    @property
    def native_frame_rate(self) -> float | None:
        """The fps the checkpoint was TRAINED at, or ``None`` if unknown.

        Wan does not take fps as a generation parameter — the model emits
        frames at a fixed rate and fps is purely a container property. The
        route reads this so the response reports the clip's real playback
        rate instead of echoing back a ``frame_rate`` the generator never
        honoured.

        Returns ``None`` rather than guessing when ``config.json`` is
        absent or has no ``sample_fps``. Wan2.1 emits 16 fps and Wan2.2
        emits 24, and the two are NOT distinguishable from weight shapes
        alone (both ship a 14B variant), so a default would be wrong half
        the time — and asserting a wrong rate is worse than admitting we
        don't know, which is the whole reason this property exists. The
        route serialises ``None`` as a ``null`` ``frame_rate``; it does NOT
        substitute the requested value, since Wan never honoured that
        either.
        """
        fps = self._config.get("sample_fps")
        if fps is None:
            return None
        try:
            value = float(fps)
        except (TypeError, ValueError):
            logger.warning("Checkpoint declares a non-numeric sample_fps: %r", fps)
            return None
        # ``> 0`` alone lets inf through (``float("inf") > 0`` is True), and an
        # infinite fps reaches the response where JSON serialisation rejects
        # it — *after* a multi-minute render. Same non-finite guard the request
        # schema already applies to ``frame_rate`` and ``seconds``; it belongs
        # on checkpoint metadata too, which is equally untrusted input.
        if not math.isfinite(value) or value <= 0:
            logger.warning("Checkpoint declares an unusable sample_fps: %r", fps)
            return None
        return value

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
        # NOT the directory name. This value is returned to every API caller,
        # and a checkpoint dir is named by the operator — it can carry a
        # customer name, an internal project, a home path fragment. Callers
        # gain nothing from it. Same reasoning as keeping the model path out
        # of the 503 body.
        return "wan"

    @property
    def max_area(self) -> int:
        """Pixel-area ceiling from the checkpoint (0 = unconstrained).

        TI2V-5B declares 901120 (= 704x1280). Exceeding it doesn't fail
        cleanly inside the pipeline, so the guard lives here.

        0 means "no ceiling declared", which matches mlx-video's own
        ``WanModelConfig`` default — we are not inventing a laxer rule than
        upstream. But a checkpoint whose config was stripped therefore
        loses the guard entirely, so that case is logged once at
        construction rather than passing silently; see
        :meth:`_warn_if_unguarded`.
        """
        try:
            return int(self._config.get("max_area") or 0)
        except (TypeError, ValueError):
            logger.warning(
                "Checkpoint declares a non-numeric max_area (%r); treating as "
                "unconstrained",
                self._config.get("max_area"),
            )
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
            # The OPERATOR-fault type, not bare ImportError: the route's
            # render-time handler must not treat every ImportError as
            # "your install is wrong". mlx-video does lazy imports during
            # rendering, and one of those failing is an internal fault whose
            # raw message should not reach the client.
            raise VideoBackendUnavailableError(err)

        if num_frames % _FRAME_MULTIPLE != 1:
            lower, upper = _valid_frame_counts_around(num_frames)
            hint = f"{lower} or {upper}" if upper is not None else str(lower)
            raise InvalidVideoRequestError(
                f"num_frames must be 4n+1 for Wan (latent temporal stride "
                f"is 4); got {num_frames}. Nearest valid: {hint}."
            )

        area_cap = self.max_area
        if area_cap and width * height > area_cap:
            raise InvalidVideoRequestError(
                f"{width}x{height} is {width * height} pixels, over this "
                f"checkpoint's {area_cap}-pixel ceiling "
                f"(e.g. 1280x704). Reduce width/height."
            )

        native = self.native_frame_rate
        if native is not None and abs(frame_rate - native) > 0.5:
            # Once per request, at info: the caller asked for something the
            # generator structurally cannot vary, and silently ignoring it
            # would leave them wondering why playback speed never changes.
            # Skipped when the rate is unknown — there is nothing to compare
            # against, and _warn_if_unguarded already said so at construction.
            logger.info(
                "Wan generates at a fixed %.0f fps; requested frame_rate=%.1f "
                "is ignored (fps is a container property, not a generation "
                "parameter for this model family)",
                native,
                frame_rate,
            )

        self._check_mode_supports_image(image)

        from mlx_video.models.wan_2.generate import generate_video

        out_path = Path(out_path)
        image_path: str | None = None
        image_is_temp = False
        if image is not None:
            image_path, image_is_temp = _materialise_image(image)

        try:
            # mlx-video treats seed=-1 as "random"; our Protocol uses None.
            generate_video(
                model_dir=str(self.model_dir),
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_path,
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
        finally:
            if image_is_temp and image_path:
                try:
                    os.unlink(image_path)
                except OSError as e:
                    logger.warning(
                        "Failed to unlink temp i2v frame %s: %s", image_path, e
                    )
        return out_path

    def _check_mode_supports_image(self, image: str | None) -> None:
        """Reject an image/no-image combination the checkpoint can't do.

        ``model_type`` says which modes a checkpoint supports: ``t2v`` is
        text-only, ``i2v`` requires a conditioning frame, ``ti2v`` does
        both. Without this check, handing an image to a T2V checkpoint (or
        omitting one for I2V) reaches the pipeline and surfaces as an
        opaque 500 after a weight load, instead of a 400 the caller can act
        on immediately.

        Unknown/absent ``model_type`` permits either — we don't guess.
        """
        kind = str(self._config.get("model_type") or "").lower()
        if kind == "t2v" and image is not None:
            raise InvalidVideoRequestError(
                "this checkpoint is text-to-video only (model_type=t2v) and "
                "cannot take a conditioning frame; omit `image`, or serve a "
                "ti2v/i2v checkpoint for image-to-video."
            )
        if kind == "i2v" and image is None:
            raise InvalidVideoRequestError(
                "this checkpoint is image-to-video only (model_type=i2v) and "
                "requires a conditioning frame; supply `image`, or serve a "
                "ti2v/t2v checkpoint for text-to-video."
            )


def resolve_checkpoint(spec: str) -> str:
    """Resolve an alias / repo id / ``repo@revision`` to a local directory.

    Downloads into the ordinary HuggingFace cache on first use, so a second
    request is free and the operator can pre-warm with ``hf download``.

    An alias always carries a pinned revision (see :data:`WAN_ALIASES`). A
    bare repo id the operator typed themselves is downloaded at its current
    ``main`` unless they pinned it with ``@revision``; that is their call to
    make, and it is logged either way so the resolved source is auditable.
    """
    entry = WAN_ALIASES.get(spec)
    if entry is not None:
        repo, revision = entry["repo"], entry["revision"]
        logger.info(
            "Wan alias %r -> %s @ %s (%s)", spec, repo, revision[:12], entry["size"]
        )
    elif "@" in spec:
        repo, _, revision = spec.rpartition("@")
        logger.info("Wan checkpoint %s @ %s (operator-pinned)", repo, revision[:12])
    else:
        repo, revision = spec, None
        logger.warning(
            "Wan checkpoint %s requested without a revision — resolving to "
            "current main. Pin it as '%s@<sha>' for reproducible downloads.",
            repo,
            repo,
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - hub ships with the engine
        raise VideoBackendUnavailableError(
            f"huggingface_hub is required to resolve {spec!r}: {e}"
        ) from e

    try:
        return snapshot_download(repo, revision=revision)
    except Exception as e:
        # Network, auth, gated repo, wrong revision — all operator-fixable,
        # and the repo id is public information so it's safe to name.
        logger.error("Could not download Wan checkpoint %s: %s", repo, e)
        raise VideoBackendUnavailableError(
            f"could not download the Wan checkpoint {repo!r}: "
            f"{type(e).__name__}. Check network access and that the "
            f"repo/revision exists."
        ) from e


def build_engine_from_env(model: str) -> WanVideoEngine:
    """Construct a :class:`WanVideoEngine` from environment configuration.

    ``model`` (the request's ``model`` field) is accepted for the factory
    signature and logged, but does NOT select a checkpoint: rapid-mlx
    serves one model per process, matching how the LLM lane works. The
    served checkpoint is whatever ``$RAPID_MLX_WAN_MODEL_DIR`` names.
    """
    # A local directory wins: an operator who converted their own weights
    # should not have that silently overridden by a stale alias setting.
    model_dir = os.environ.get(ENV_MODEL_DIR)
    if not model_dir:
        spec = os.environ.get(ENV_MODEL)
        if spec:
            model_dir = resolve_checkpoint(spec)
    if not model_dir:
        raise NotImplementedError(
            f"no video backend configured. Set ${ENV_MODEL} to one of "
            f"{', '.join(sorted(WAN_ALIASES))} (downloaded on first use), or "
            f"${ENV_MODEL_DIR} to a locally-converted MLX Wan checkpoint. "
            "See docs/content_farm_api.md."
        )
    steps_raw = os.environ.get(ENV_STEPS)
    steps = None
    if steps_raw:
        # Hold the env override to the SAME 1..500 bound the public request
        # contract enforces. Without this, `RAPID_MLX_WAN_STEPS=0` silently
        # forwards a value the API would have rejected, and the failure
        # surfaces from inside the sampler instead of at configuration time.
        try:
            candidate = int(steps_raw)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r", ENV_STEPS, steps_raw)
        else:
            if 1 <= candidate <= _MAX_STEPS:
                steps = candidate
            else:
                logger.warning(
                    "Ignoring out-of-range %s=%r (must be 1..%d)",
                    ENV_STEPS,
                    steps_raw,
                    _MAX_STEPS,
                )
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

    if not (os.environ.get(ENV_MODEL_DIR) or os.environ.get(ENV_MODEL)):
        return False
    engine_mod._VIDEO_ENGINE_FACTORY = build_engine_from_env
    return True
