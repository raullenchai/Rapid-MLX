# Local batch travel posters with ComfyUI and Rapid-MLX

Generate six imaginary travel posters with Qwen-Image 2.1 on Apple Silicon.
ComfyUI handles graph execution and saving; Rapid-MLX owns diffusion inference
through MLX. The included reel builder animates those still images with FFmpeg.

## Install

Requires an Apple Silicon Mac, Python 3.11+, Git and FFmpeg. The demonstrated
host is an M3 Ultra with 256 GB unified memory; this demo does not establish
performance or usability on smaller Macs. Qwen's default q4 download is about
8.9 GiB. Follow your machine's model storage policy and reuse its HF cache.

From the Rapid-MLX repository root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[image]'
brew install ffmpeg

mkdir -p /private/tmp/demo-twitter-comfy
# Use a separate environment so ComfyUI's Torch dependencies do not change MLX's runtime.
git clone https://github.com/Comfy-Org/ComfyUI.git /private/tmp/demo-twitter-comfy/ComfyUI
git -C /private/tmp/demo-twitter-comfy/ComfyUI checkout 0df64eb242b7c5759c3e86afd5d1846d923b1033
python3.12 -m venv /private/tmp/demo-twitter-comfy/venv
/private/tmp/demo-twitter-comfy/venv/bin/pip install -r /private/tmp/demo-twitter-comfy/ComfyUI/requirements.txt
ln -s "$PWD/examples/comfyui-travel/custom_nodes/rapid_mlx" \
  /private/tmp/demo-twitter-comfy/ComfyUI/custom_nodes/rapid_mlx
```

Run each server in its own terminal:

```bash
.venv/bin/rapid-mlx serve qwen-image-2.1 --host 127.0.0.1 --port 18427
```

```bash
/private/tmp/demo-twitter-comfy/venv/bin/python /private/tmp/demo-twitter-comfy/ComfyUI/main.py \
  --cpu --listen 127.0.0.1 --port 8189 \
  --output-directory /private/tmp/demo-twitter-media/comfy-output
```

Open <http://127.0.0.1:8189> and drag `workflow.json` onto the canvas. The graph is
`Rapid-MLX · Local Image → Save Image`. ComfyUI's `--cpu` flag applies to its own
Torch process; the Rapid-MLX server still uses Metal through MLX. No diffusion
weights need to be installed in ComfyUI. If your server requires authentication,
set `RAPID_MLX_API_KEY` in the ComfyUI process environment; the node reads it
without putting the key in the workflow or saved PNG metadata.

## Generate the batch

```bash
python3 examples/comfyui-travel/batch.py \
  --output examples/comfyui-travel/media/posters
```

The script submits each job to ComfyUI's `/prompt`, waits for a successful
`/history/{prompt_id}` result, then downloads SaveImage's PNG through `/view`.
Each output has a JSON sidecar containing the prompt, seed, request settings,
ComfyUI history, elapsed wall time and image SHA256. Start with `--limit 1` to
check the setup. Repeating the same command skips verified completed files;
changed settings require a fresh output directory. A timed-out job may still
be running: inspect the queue before resubmitting.

The six prompts in `destinations.json` share a visual direction and vary by
location. This is sequential batch automation, not simultaneous GPU batching.
Each request uses `n=1`, a fixed seed, 1024×1024 pixels and 40 steps. A seed
records the experiment; exact bytes can change with runtime and hardware.
Different prompts generate independent pictures, so this does not promise
character identity consistency. Lettering quality still needs visual review.

## Build shareable media

```bash
/private/tmp/demo-twitter-comfy/venv/bin/python examples/comfyui-travel/render.py \
  --input examples/comfyui-travel/media/posters \
  --output examples/comfyui-travel/media
```

This creates a 3×2 contact sheet and a silent 1920×1080 H.264 MP4 with six
three-second scenes, subtle camera movement, fades and a two-second brand outro. These are animated stills,
not Qwen-generated motion. The original lettering in each poster comes from
Qwen; the contact-sheet and video labels are added by `render.py`.

`media/` is ignored by Git. Keep generated outputs there for review and sharing.
The temporary servers and intermediate files under `/private/tmp` follow local
scratch retention; preserve wanted originals in the workspace before cleanup.

## Validate and troubleshoot

```bash
/private/tmp/demo-twitter-comfy/venv/bin/python examples/comfyui-travel/test_node.py
```

The node tests cover RGB tensor layout, request seed, HTTP failure details,
cancelled generation and connection failure. Actual generation evidence belongs
in `media/posters/*.json`; tests alone do not establish model quality or speed.

If the node is missing, check the symlink and restart ComfyUI. If generation fails,
read the Rapid-MLX server log; the node forwards HTTP error details. Keep the
ComfyUI and Rapid-MLX model names aligned. To try another supported generation
model, serve it first and pass `--model` and the appropriate `--steps` to the
batch script. Do not use an edit-only model with this generation node.

ComfyUI can reuse cached node outputs when inputs are identical. Change the seed
to request a new sample. Interrupting ComfyUI does not cancel an HTTP request
already running in Rapid-MLX; wait for completion before restarting that job.

## Verified run

See [VERIFIED.md](VERIFIED.md) and [verified-run.json](verified-run.json) for the
real run, exact runtime versions, selected seeds, image hashes and measured times.

## Files

- `custom_nodes/rapid_mlx/`: HTTP bridge, returns a standard ComfyUI IMAGE tensor.
- `workflow.json`: UI-importable example graph.
- `destinations.json`: all six destination prompts and fixed seeds.
- `batch.py`: sequential execution and resumable evidence capture.
- `render.py`: contact sheet and animated-still video.
- `blog.en.md`, `blog.zh.md`, `tweets.md`: publication drafts.

The implementation uses the official [ComfyUI image tensor contract](https://docs.comfy.org/custom-nodes/backend/images_and_masks)
and [queue/history API](https://docs.comfy.org/development/comfyui-server/comms_routes).
See [Rapid-MLX Qwen-Image 2.1 behavior](../../docs/models/families/qwen-image-2.1.md)
for model precision and img2img limitations. Review the upstream model's license
and usage terms before distributing commercial assets.
