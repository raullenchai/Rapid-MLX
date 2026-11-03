# Video generation

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a video model:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve ltx-2.3-mlx-q4
```

The server prints the local endpoint. Submit a video request as shown below.

Rapid-MLX exposes LTX-2.3, LTX-2.5, CogVideoX-Fun and Wan through the asynchronous
OpenAI-compatible Videos API.

The video runtime ships with the official installer and requires Python 3.12 or newer.

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

LTX-2.5 uses a new Gemma 4 text encoder. Its runtime (`ltx_core_mlx` and
`ltx_pipelines_mlx`) ships inside rapid-mlx, so no separate checkout is needed:

```bash
brew install ffmpeg
rapid-mlx serve ltx-2.5-mlx-q8
```

`rapid-mlx pull ltx-2.5-mlx-q8` runs the same runtime preflight before it
downloads the weights, and `rapid-mlx models` marks video aliases that cannot
run on the current host.

The registered Q8 checkpoint is a 67.7 GB download and uses the low-RAM
distilled path by default. The weights are distributed under the LTX-2.x
Community License rather than an open-source license; review the checkpoint's
`LICENSE.md` before use, especially its commercial-use and generated-content
disclosure terms.

`RAPID_MLX_LTX25_RUNTIME` remains available for development against a pinned
`ltx-2-mlx` checkout. When set, it must point at an executable at the audited
revision and replaces the bundled runtime; an invalid override is reported as
an error instead of silently falling back. The bundled runtime carries
provenance metadata for its repository, commit and version, so a missing or
damaged copy is reported as a broken install rather than replaced by an
unverified copy.

Generations have a two-hour safety deadline; set
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

### Extend an LTX-2.5 video

`POST /v1/videos/extend` uses the existing MP4 as temporal context and returns
one MP4 containing the original and appended frames. It runs the runtime's
video extension pipeline with the source video as temporal context.

```bash
curl http://localhost:8000/v1/videos/extend \
  -F model=ltx-2.5-mlx-q8 \
  -F 'prompt=The camera continues tracking the rider as the dragon banks left' \
  -F input_video=@source.mp4 \
  -F extend_frames=16 \
  -F seed=42
```

The source must be an MP4 at exactly 24 fps, with dimensions from 256 to 1920
pixels in multiples of 32 and a decoded frame count of `8n+1` (at least 9).
Variable-rate files whose nominal and average frame rates disagree, corrupt
videos, and inconsistent frame-count metadata are rejected before generation.
`extend_frames` counts **output frames**, must be a multiple of 8, and accepts
8–48. The endpoint accepts up to 97 total output frames, 24 million
pixel-frames, and a 20 MB upload. The result uses the same asynchronous job and
download endpoints as `POST /v1/videos`. `GET /v1/videos/capabilities` reports
these extension limits. The uploaded source is removed after generation.

The server's `--max-request-bytes` also applies to uploads. To allow the full
20 MiB source plus multipart overhead, start the server with
`--max-request-bytes 23068672`.

## CogVideoX-Fun

Rapid-MLX can serve CogVideoX-Fun as an experimental, single-worker video
generation backend on Apple Silicon. The MVP supports one-second,
672×384 text-to-video jobs.

### Install

```bash
brew install ffmpeg
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

## Pika H3 Flex

Serve `pika-h3-flex` with at least 160 GB of unified memory and the same runtime
requirements as MiniMax-H3. It reuses the MiniMax-H3 base download and adds
FL2VA and Ref2VA adapters plus a second transformer partition. Generation uses
a fixed six-step renoise schedule, 24 fps video, and synchronized 32 kHz stereo
audio. Set `RAPID_MLX_URL` to your server's base URL for the examples below.

Use text alone, a first frame through `input_reference`, a last frame through
`last_frame`, or both fields for first+last-frame conditioning. Recommended
sizes are `832x480` (480p 16:9) and `480x832`. Durations are 5–15 seconds, rounded
up to the 17n+5 frame grid: 5 seconds gives 124 frames. The maximum is 345 frames,
including requests for 15 seconds. Explicit `frames` must follow the same grid.

```sh
rapid-mlx serve pika-h3-flex
curl "$RAPID_MLX_URL/v1/videos" \
  -F model=pika-h3-flex \
  -F 'prompt=A quiet shoreline at sunrise, with gentle waves' \
  -F seconds=5 -F size=832x480 \
  -F input_reference=@first.png -F last_frame=@last.png
```

Ref2VA generates video and synchronized audio from a prompt and ordered reference
files. Repeat the `references` multipart field; upload order is the packed
conditioning order:

```sh
curl "$RAPID_MLX_URL/v1/videos" \
  -F model=pika-h3-flex \
  -F 'prompt=The character walks along the shoreline as the waves break' \
  -F seconds=5 -F size=832x480 \
  -F references=@character.png \
  -F references=@motion.mp4 \
  -F references=@sound.wav
```

Supplying `references` selects Ref2VA. Without references, text alone selects T2VA,
and `input_reference` or `last_frame` selects FL2VA. References cannot be combined
with either frame field. Reference limits are:

- Up to 9 images, 3 videos, and 3 standalone audio clips, with 12 files total.
  At least one image or video is required.
- Images: JPEG, PNG, or WebP, up to 20 MB and 16 megapixels, with aspect ratios
  from 1:4 to 4:1.
- Videos: MP4, MOV, or WebM, up to 100 MB and 3840 × 2160 source pixels.
- Audio: WAV, MP3, M4A, AAC, FLAC, or OGG, up to 20 MB.
- Each video and standalone audio clip must be 2–15 seconds. Video durations
  total at most 15 seconds; standalone audio durations independently total at
  most 15 seconds. Video soundtracks count toward neither duration total.

Media types are detected from file contents. Video and audio uploads require
`ffprobe`. Reference video is resampled to 24 fps, and audio to 32 kHz stereo,
before truncation to the requested output length. Reference image and video
sizes follow the generated canvas tier; image references may be upscaled.

Switching between FL2VA/T2VA and Ref2VA unloads one 33B transformer and loads the
other (about a minute; the text encoder and VAEs stay loaded); peak memory stays
in the same class as a single mode.

### License and attribution

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

FL2VA and Ref2VA adapters Copyright © 2026 Mellis, Inc. (Pika). It is a Model Derivative of MiniMax H3 distributed under the MiniMax H3 Community License Agreement.

The weights are downloaded from the upstream Hugging Face repositories and are never mirrored.
