# SPDX-License-Identifier: Apache-2.0
"""Queue reproducible poster jobs through ComfyUI, download outputs and evidence."""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def api(base, route, payload=None):
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def graph(job, args):
    return {
        "1": {
            "class_type": "RapidMLXImage",
            "inputs": {
                "base_url": args.rapid_url,
                "model": args.model,
                "prompt": job["prompt"],
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "seed": job["seed"],
            },
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["1", 0],
                "filename_prefix": "rapid-travel/" + job["id"],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8189")
    parser.add_argument("--rapid-url", default="http://127.0.0.1:18427")
    parser.add_argument("--model", default="qwen-image-2.1")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((ROOT / "destinations.json").read_text())[: args.limit]
    for job in jobs:
        workflow = graph(job, args)
        fingerprint = hashlib.sha256(
            json.dumps(workflow, sort_keys=True).encode()
        ).hexdigest()
        evidence_file = args.output / (job["id"] + ".json")
        target = args.output / (job["id"] + ".png")
        if evidence_file.exists() and target.exists():
            prior = json.loads(evidence_file.read_text())
            if (
                prior.get("request_sha256") == fingerprint
                and prior.get("image_sha256")
                == hashlib.sha256(target.read_bytes()).hexdigest()
            ):
                print("Resume: already saved", target, flush=True)
                continue
            raise RuntimeError(
                f"Existing output has different settings or bytes: {target}. Use a fresh output directory."
            )
        started = time.monotonic()
        queued = api(args.comfy_url, "/prompt", {"prompt": workflow})
        prompt_id = queued["prompt_id"]
        print("Queued", job["id"], prompt_id, flush=True)
        while time.monotonic() - started < args.timeout:
            history = api(args.comfy_url, "/history/" + prompt_id).get(prompt_id)
            if history:
                if history.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(json.dumps(history["status"]))
                if history.get("status", {}).get("completed"):
                    break
            time.sleep(2)
        else:
            raise TimeoutError(
                f"Job {prompt_id} still running; inspect ComfyUI history before resubmitting."
            )
        outputs = history.get("outputs", {}).get("2", {}).get("images", [])
        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected one saved image for {prompt_id}, got {outputs}"
            )
        with urllib.request.urlopen(
            args.comfy_url.rstrip("/") + "/view?" + urllib.parse.urlencode(outputs[0]),
            timeout=30,
        ) as response:
            image = response.read()
        target.write_bytes(image)
        evidence = {
            "destination": job["id"],
            "prompt_id": prompt_id,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "request_sha256": fingerprint,
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "workflow": workflow,
            "history": history,
        }
        evidence_file.write_text(json.dumps(evidence, indent=2) + "\n")
        print("Saved", target, evidence["elapsed_seconds"], "seconds", flush=True)


if __name__ == "__main__":
    main()
