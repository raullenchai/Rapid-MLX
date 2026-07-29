# Video generation (LTX-2.3, MLX-native) — POC

`VideoEngine` is the video lane of rapid-mlx: **text→video** (t2v) and
**image→video** (i2v), fully local on Apple Silicon. It completes the turnkey
local content spine — script → voice → timing → music → **motion** — entirely
on one Mac.

> **Status: draft / POC.** `VideoEngine` currently shells out to the
> `ltx-2-mlx` CLI in a subprocess (mirroring how `MusicEngine` wraps
> Stable Audio 3). Proper in-process vendoring and wiring to the
> `/v1/video/generations` route (the `VideoEngine` Protocol from PR #1300) is
> the follow-up.

## Backend

The backend is [`dgrauet/ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx), a
**pure-MLX** port of LTX-2.3 (no torch). It emits an mp4 with a
natively-synchronized audio track. The **q4 distilled** variant runs
free-human-motion t2v clips and identity-locking i2v on a 32 GB Mac mini.

## Install

The `ltx-2-mlx` dependency is an **optional** extra — it is not part of the
base install (heavy diffusion pipelines + multi-GB weights):

```bash
pip install '.[video]'
```

If the CLI is missing, `VideoEngine.generate` raises a clear install-hint
error rather than failing at import time.

## Usage

```python
from vllm_mlx.video import VideoEngine

eng = VideoEngine()  # q4 distilled, sensible defaults for a 32 GB mini

# text-to-video
eng.generate("a fox trotting through fresh snow", "clip.mp4")

# image-to-video — animate a still, locking identity (the key path for
# multi-shot same-face consistency: still → i2v lock-face + motion)
eng.generate("the same woman turns and smiles", "shot.mp4",
             image="lead_ref.png")
```

Knobs exposed by `generate`: `height`, `width`, `num_frames`, `frame_rate`
(LTX-2.3 is trained at 24 fps), `steps`, `negative_prompt`, `seed`, and
`low_ram` (block-streaming, on by default to fit 32 GB).

## Route contract

This concrete engine is designed to satisfy the `VideoEngine` Protocol behind
the OpenAI-style `POST /v1/video/generations` route (PR #1300 /
`feat/openai-routes-content-farm`). Once the backend is registered with that
route's factory, the video lane goes live with zero handler changes.
