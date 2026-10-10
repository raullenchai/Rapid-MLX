# SPDX-License-Identifier: Apache-2.0
"""ComfyUI image node backed by a separately running Rapid-MLX server.

PyTorch only carries the returned pixels; diffusion runs in Rapid-MLX/MLX.
"""

import base64
import io
import json
import os
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image


class RapidMLXImage:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI node protocol
        return {
            "required": {
                "base_url": ("STRING", {"default": "http://127.0.0.1:18427"}),
                "model": ("STRING", {"default": "qwen-image-2.1"}),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "A moonlit floating city"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Rapid-MLX"
    DESCRIPTION = "Generate locally through Rapid-MLX. Use the server root URL without /v1. Set RAPID_MLX_API_KEY in the ComfyUI process if authentication is enabled."

    def generate(self, base_url, model, prompt, width, height, steps, seed):
        payload = {
            "model": model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "steps": steps,
            "seed": seed,
            "n": 1,
            "response_format": "b64_json",
        }
        headers = {
            "Content-Type": "application/json",
            "X-Rapid-Client": "comfyui-travel",
        }
        if key := os.environ.get("RAPID_MLX_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v1/images/generations",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode(errors="replace")
            raise RuntimeError(f"Rapid-MLX HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Cannot reach Rapid-MLX. Start rapid-mlx serve and check base_url."
            ) from exc
        if result.get("cancelled"):
            raise RuntimeError(
                "Rapid-MLX generation was cancelled; no completed image returned."
            )
        if len(result.get("data", [])) != 1 or "b64_json" not in result["data"][0]:
            raise RuntimeError("Rapid-MLX did not return one base64 image.")
        raw = base64.b64decode(result["data"][0]["b64_json"], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(pixels).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"RapidMLXImage": RapidMLXImage}
NODE_DISPLAY_NAME_MAPPINGS = {"RapidMLXImage": "Rapid-MLX · Local Image"}
