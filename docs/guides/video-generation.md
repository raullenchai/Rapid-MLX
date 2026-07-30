# CogVideoX-Fun video generation

Rapid-MLX can serve CogVideoX-Fun as an experimental, single-worker video
generation backend on Apple Silicon. The MVP supports one-second,
672×384 text-to-video jobs.

## Install

The CogVideoX MLX port is currently source-only:

```bash
brew install ffmpeg
pip install 'rapid-mlx[video]'
git clone https://github.com/dgrauet/VideoX-Fun-mlx.git
git -C VideoX-Fun-mlx checkout 26326e7d52e6762375227b320d77003dac764d14
export PYTHONPATH="$PWD/VideoX-Fun-mlx:$PYTHONPATH"
```

Start the recommended q4 checkpoint:

```bash
rapid-mlx serve cogvideox-fun-5b-q4 --port 8000
```

The pipeline and weights load on the first job. The q4 checkpoint uses about
14.5 GB peak RSS; a Mac with at least 24 GB unified memory is recommended.

## Create and download a video

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
