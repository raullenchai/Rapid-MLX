# Generate local travel posters with ComfyUI and Rapid MLX

I built a travel agency for places that do not exist: a lunar night market, a resort above Saturn's rings, a train through the deep ocean. The posters were generated locally on a Mac with Qwen-Image 2.1, with ComfyUI organizing the workflow and Rapid-MLX running inference through MLX.

![Six imaginary travel destinations](media/contact-sheet.jpg)

The useful part is the repeatable pipeline. Replace the destinations with product concepts, event themes or story settings, and the same workflow becomes a small local creative production line.

## Connect the workflow to local inference

The demo runs two Python processes. ComfyUI handles its graph, execution queue and image saving. A custom node calls Rapid-MLX's local Images API and converts the returned PNG into a standard ComfyUI IMAGE tensor.

```text
Destination prompts and seeds
          ↓
ComfyUI queue → Rapid-MLX Image node
                        ↓ POST /v1/images/generations
                   Rapid-MLX → MLX → Qwen-Image 2.1
                        ↓ base64 PNG
                   ComfyUI Save Image
                        ↓
                  PNG + execution record
```

Rapid-MLX integrates mflux in its image runtime; this demo uses that existing API. The diffusion weights live in the Rapid-MLX model cache. ComfyUI needs no second copy. Keeping the environments separate also prevents ComfyUI's Torch dependencies from changing the MLX inference environment.

## Start Rapid MLX

On an Apple Silicon Mac with Python 3.11 or later:

```bash
pip install 'rapid-mlx[image]'
rapid-mlx serve qwen-image-2.1 --host 127.0.0.1 --port 18427
```

The `qwen-image-2.1` alias uses a prequantized checkpoint with native MLX q4 denoiser and text encoder weights, an approximately 8.9 GiB download and 40 default denoising steps. This demo runs on an M3 Ultra with 256 GB unified memory; its results do not establish performance on smaller Macs. See the [model guide](../../docs/models/families/qwen-image-2.1.md) for precision and memory details.

A request to the server looks like this:

```bash
curl http://127.0.0.1:18427/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1","prompt":"A cinematic lunar night market travel poster, with the headline LUNAR NIGHT MARKET","size":"1024x1024","steps":40,"seed":4300,"response_format":"b64_json"}'
```

The response contains PNG bytes encoded in `data[0].b64_json`. The node checks for cancellation, missing images and HTTP errors before passing pixels downstream.

## Add the ComfyUI node

Link or copy `custom_nodes/rapid_mlx` from the demo into ComfyUI's `custom_nodes` directory, restart ComfyUI and import `workflow.json`. The [README](README.md) contains complete installation commands and the tested ComfyUI commit.

![Imported ComfyUI graph with a completed image loaded from execution history](media/workflow.png)

ComfyUI can run in CPU mode for this workflow:

```bash
python main.py --cpu --listen 127.0.0.1 --port 8189
```

That flag applies to ComfyUI's Torch process. Rapid-MLX still runs diffusion through MLX and Metal in its own process.

After decoding the API response, the bridge converts the image into RGB float pixels:

```python
pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
return (torch.from_numpy(pixels).unsqueeze(0),)
```

The tensor has shape `[batch, height, width, channels]`, as specified by the [ComfyUI image contract](https://docs.comfy.org/custom-nodes/backend/images_and_masks). Torch carries these pixels; it does not run the diffusion model in this setup. Connect the node's IMAGE output to the built-in Save Image node.

If the server requires authentication, set `RAPID_MLX_API_KEY` in ComfyUI's environment. The node reads it at runtime so it does not enter the workflow or PNG metadata.

## Turn six prompts into a batch

`destinations.json` contains the scene, requested lettering and fixed seed for each destination. All prompts share the same direction: cinematic retro-futurist travel illustration, detailed architecture and tiny travelers for scale.

```bash
python3 examples/comfyui-travel/batch.py \
  --output examples/comfyui-travel/media/posters
```

The script submits one graph to ComfyUI's `/prompt`, polls `/history/{prompt_id}`, downloads the saved PNG through `/view`, then starts the next job. These are standard [ComfyUI server APIs](https://docs.comfy.org/development/comfyui-server/comms_routes).

This is sequential batch automation. Rapid-MLX's image lane runs one image at a time so diffusion pipelines do not compete for unified memory. Each image uses a separate prompt, `n=1`, a fixed seed, 1024×1024 pixels and 40 steps.

A JSON sidecar records each request, its seed, ComfyUI history, elapsed wall time and image SHA256. Running the same command again verifies and skips completed files. Changed settings require a new output directory, preserving the earlier experiment.

A seed makes comparisons easier; it does not guarantee identical bytes across runtime versions or hardware. ComfyUI can also cache identical node inputs, so change the seed when you want a new sample. Interrupting ComfyUI does not cancel an HTTP request already running in Rapid-MLX; inspect the active job before resubmitting after a timeout.

## Measured demo run

The six selected images took about 189–205 seconds each in Rapid-MLX server completion logs, at 1024×1024 and 40 steps on the M3 Ultra. Client elapsed time also includes ComfyUI queue waits and polling. This is one demo session with cached weights, not a controlled hardware comparison; [the run record](VERIFIED.md) includes exact versions, seeds and per-image measurements.

## Review generated lettering

The poster headline, subtitle and `RAPID TRAVEL` footer are requested inside the image prompt. They are generated by Qwen rather than added over the original poster afterward. This makes visual review part of the workflow: inspect spelling, repeated words and readability before publishing.

The first lunar poster rendered `DARK` as `DANK`. I retained that result and reran it with a shorter subtitle and a new seed. The Saturn subtitle also needed a shorter prompt and another sample. The execution records distinguish those attempts. Shared art direction can also vary between independent generations; it does not guarantee character or object identity across a series.

## Make a shareable reel

```bash
python examples/comfyui-travel/render.py \
  --input examples/comfyui-travel/media/posters \
  --output examples/comfyui-travel/media
```

The renderer creates a contact sheet and a silent 1080p H.264 video, with three seconds per destination and a two-second Rapid-MLX outro. FFmpeg adds a gentle push-in and fades to each still; the renderer adds the surrounding brand labels.

The reel is animated stills, not diffusion-generated motion. A separate image-to-video stage could use one of Rapid-MLX's video models, but the reproducible scope here is batch image generation.

To adapt the demo, change the scenes and text in `destinations.json`, run `--limit 1` to inspect a sample, then generate the full set. Rapid-MLX provides the local inference service, and ComfyUI turns it into a workflow you can inspect, queue and repeat.
