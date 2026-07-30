# Video generation

Rapid-MLX exposes LTX-2.3, CogVideoX-Fun and Wan through the asynchronous
OpenAI-compatible Videos API.

Video generation requires Python 3.11 or newer because the upstream
`mlx-video-with-audio` runtime does not support Python 3.10. Rapid-MLX's core
text and audio features continue to support Python 3.10.

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
