# Video generation

Rapid-MLX exposes LTX-2.3, LTX-2.5, CogVideoX-Fun and Wan through the asynchronous
OpenAI-compatible Videos API.

Video generation requires Python 3.11 or newer because the upstream
`mlx-video-with-audio` runtime does not support Python 3.10. Rapid-MLX's core
text and audio features continue to support Python 3.10.

## Keep completed videos across restarts

By default, video jobs and MP4 files use a process-temporary directory and are
removed when the server exits. For a desktop app or another long-lived client,
pass an explicit artifact directory when starting the server:

```bash
rapid-mlx serve ltx-2.3-mlx-q4 \
  --video-output-dir /path/to/rapid-mlx-video-artifacts
```

Every completed job and its manifest are committed durably and atomically.
Starting a later server with the same directory restores up to the newest 100
completed jobs, so they remain available through `GET /v1/videos`,
`GET /v1/videos/{id}` and `GET /v1/videos/{id}/content`.
`DELETE /v1/videos/{id}` removes both the record and its MP4. Queued,
interrupted, failed, incomplete or malformed records are never restored as
completed work.

The metadata includes the prompt and generation settings. Choose a
user-private directory with enough free space; Rapid-MLX creates new job
directories with owner-only permissions but does not change the permissions of
an existing parent directory.

## Discover capabilities before serving

`rapid-mlx models --json` lists video aliases separately from other model
types. Each video entry includes `video_modes` (`text-to-video`,
`image-to-video`, or both) and `min_memory_gb`, so clients can choose a
compatible checkpoint before downloading or starting it. After the model is
serving, `GET /v1/videos/capabilities` remains the source for its live size,
duration, frame-rate, workload, reference-image, and optional-control limits.

## Motion and conditioning controls

`POST /v1/videos` accepts optional controls in addition to `seconds`:

- `fps` selects the output frame rate for LTX and CogVideoX-Fun. Wan uses the
  checkpoint's native frame rate and rejects a different value.
- `frames` overrides the frame count derived from `seconds`. LTX requires
  `8n+1` with a minimum of 9; Wan and CogVideoX-Fun require `4n+1` with a
  minimum of 5.
- `guidance_scale` (1–30) and `negative_prompt` are passed to backends with
  classifier-free guidance. LTX-2.5's distilled pipeline rejects both.
- `conditioning_strength` (0–1) controls how closely LTX image-to-video follows
  `input_reference`; it is rejected without a reference image and is not
  currently supported by Wan or CogVideoX-Fun.

For example, request a shorter, lower-frame-rate LTX image-to-video result with
weaker reference conditioning:

```bash
curl http://127.0.0.1:8000/v1/videos \
  -F model=ltx-2.3-mlx-q4 \
  -F 'prompt=the camera sweeps around the subject' \
  -F input_reference=@start.png \
  -F fps=12 \
  -F frames=17 \
  -F guidance_scale=4.5 \
  -F conditioning_strength=0.35 \
  -F 'negative_prompt=static camera, frozen subject'
```

## Wan 2.1 / 2.2

Wan uses the `mlx-video-with-audio` runtime included in the video extra. Four
converted Wan 2.2 checkpoints are registered; the 5B Q8 TI2V checkpoint is the
recommended starting point and supports both text-to-video and
image-to-video.

```bash
pip install 'rapid-mlx[video]'
brew install ffmpeg
rapid-mlx serve wan2.2-ti2v-5b-q8
```

Create a one-second text-to-video job:

```bash
curl http://127.0.0.1:8000/v1/videos \
  -F model=wan2.2-ti2v-5b-q8 \
  -F 'prompt=a fox running through fresh snow, cinematic tracking shot' \
  -F seconds=1 \
  -F size=832x512 \
  -F seed=42
```

Add `-F input_reference=@start.png` for image-to-video when the served
checkpoint is TI2V or I2V. Poll and download the result through the same
`GET /v1/videos/{id}` and `GET /v1/videos/{id}/content` endpoints shown
below.

The backend reads the checkpoint's native frame rate (16 fps for Wan 2.1,
24 fps for Wan 2.2), enforces Wan's `4n+1` temporal shape and honors the
checkpoint's pixel-area ceiling. Converted local Wan 2.1/2.2 checkpoints can
override the selected Wan alias with `RAPID_MLX_WAN_MODEL_DIR`.

Optional process-level tuning:

```bash
RAPID_MLX_WAN_STEPS=8 \
RAPID_MLX_WAN_SCHEDULER=unipc \
RAPID_MLX_WAN_TILING=auto \
rapid-mlx serve wan2.2-ti2v-5b-q8
```

LoRAs use `path[:strength]` entries through `RAPID_MLX_WAN_LORA`; dual-model
checkpoints additionally accept `RAPID_MLX_WAN_LORA_HIGH` and
`RAPID_MLX_WAN_LORA_LOW`.

## LTX-2.3

MLX-native LTX-2.3. The Q4 checkpoint is a 22.8 GB download and wants at least
24 GB of unified memory; 32 GB or more is comfortable.

The checkpoint is an audio-video one and the pipeline runs
`generate_video_with_audio`, but the track it produces is silent. Rapid-MLX
remuxes it away, so **the MP4 you get back is video-only** — a silent audio
track is worse than none, since it makes downstream tools believe the clip has
sound. Expect no audio from any of the three backends.

```bash
pip install 'rapid-mlx[video]'
brew install ffmpeg
rapid-mlx serve ltx-2.3-mlx-q4
```

```bash
curl http://localhost:8000/v1/videos \
  -F model=ltx-2.3-mlx-q4 \
  -F 'prompt=A fox running through fresh snow, cinematic tracking shot' \
  -F seconds=4 \
  -F size=768x512
```

LTX honours the `fps`, `frames` and `conditioning_strength` controls described
above; `frames` must satisfy LTX's own frame-count rule.

## LTX-2.5

LTX-2.5 uses a new Gemma 4 text encoder and is served through the standalone
`ltx-2-mlx` research runtime. That runtime is not published on PyPI yet, so it
must be installed from its audited source commit. The registered Q8 checkpoint
is a 67.7 GB download and uses the low-RAM distilled path by default.
The weights are distributed under the LTX-2.x Community License rather than
an open-source license; review the checkpoint's `LICENSE.md` before use,
especially its commercial-use and generated-content disclosure terms.

```bash
git clone --branch ltx25 https://github.com/MrMoferFRAN/ltx-2-mlx.git
git -C ltx-2-mlx checkout 57952288076766abe27dda3a774b2c24f7346977
uv sync --project ltx-2-mlx
brew install ffmpeg

RAPID_MLX_LTX25_RUNTIME="$PWD/ltx-2-mlx/.venv/bin/ltx-2-mlx" \
  rapid-mlx serve ltx-2.5-mlx-q8
```

Rapid uses the executable only to locate and verify the source checkout: it
must be the expected workspace entry point, with a tracked lockfile and the
exact HEAD above. For every generation, Rapid materializes only the tracked
files from that exact commit into a process-private temporary tree and runs
`uv sync --frozen` during startup preflight. Generations reuse that provisioned
environment, so checkout modifications, untracked files and the mutable
checkout `.venv` are never executed or rebuilt per request. Generations have a
two-hour safety deadline; set
`RAPID_MLX_LTX25_TIMEOUT_SEC` to a larger value (minimum 60) for unusually
large jobs.

Create a job with the same API. LTX-2.5 generates synchronized audio and Rapid
preserves that audio track:

```bash
curl http://localhost:8000/v1/videos \
  -F model=ltx-2.5-mlx-q8 \
  -F 'prompt=A red fox trots across fresh snow at golden hour' \
  -F frames=97 \
  -F fps=24 \
  -F size=704x480
```

Text-to-video and image-to-video are supported. Frame counts must be `8n+1`;
dimensions are rounded up to the runtime's required 32-pixel boundary and
cropped back to the requested OpenAI size when necessary. The distilled path
does not expose `guidance_scale` or `negative_prompt`.

## CogVideoX-Fun

Rapid-MLX can serve CogVideoX-Fun as an experimental, single-worker video
generation backend on Apple Silicon. The MVP supports one-second,
672×384 text-to-video jobs.

### Install

```bash
brew install ffmpeg
pip install 'rapid-mlx[video]'
```

The pinned CogVideoX-Fun MLX runtime ships with Rapid-MLX; no source checkout
or `PYTHONPATH` modification is required.

Start the recommended q4 checkpoint:

```bash
rapid-mlx serve cogvideox-fun-5b-q4 --port 8000
```

The pipeline and weights load on the first job. The q4 checkpoint uses about
14.5 GB peak RSS; a Mac with at least 24 GB unified memory is recommended.

### Create and download a video

```bash
curl http://127.0.0.1:8000/v1/videos \
  -F model=cogvideox-fun-5b-q4 \
  -F 'prompt=a beautiful sunset over the ocean' \
  -F seconds=1 \
  -F size=672x384 \
  -F seed=42
```

The response contains a job ID. Poll it and download the completed MP4:

```bash
curl http://127.0.0.1:8000/v1/videos/video_ID
curl http://127.0.0.1:8000/v1/videos/video_ID/content --output result.mp4
```

Generation is serialized and defaults to 50 diffusion steps. On an M3 Ultra,
the q4 model produced a one-second 672×384 clip in about 338 seconds. Static
scenes were usable in testing; fast subject motion remains experimental.
