#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Marvin /v1/classify proxy — the executable serving contract.

Two backends, one response shape:
  --backend mlx        local Apple Silicon (mlx-lm), for Mac serving / dev
  --backend llama_cpp  remote llama-server (CUDA box, e.g. Vast.ai), the
                       NVIDIA serving path from the runbook

Response: {"decision": "<candidate>", "confidence": p, "probabilities": {...},
           "latency_ms": ms, "model": ..., "adapter": ...}

Readout semantics (must stay identical across backends): one forward pass,
softmax restricted to the candidate letter tokens (A..H), argmax decides,
zero generated tokens. Self-check validates the backend before any claim.

  python classify_proxy.py --backend mlx --serve --port 8123
  python classify_proxy.py --backend llama_cpp --base http://localhost:8080 --selfcheck
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "bench" / "marvins_garden"
sys.path.insert(0, str(BENCH))

import render  # noqa: E402

MANIFEST = json.loads((HERE / "adapter_manifest.json").read_text())
# Serving backend: mlx (Mac, main-thread worker) or llama_cpp (NVIDIA box).
BACKEND = os.environ.get("MARVIN_BACKEND", "mlx")
LLM_BASE = os.environ.get("MARVIN_LLM_BASE", "http://127.0.0.1:8080")


class MlxBackend:
    def __init__(self):
        sys.path.insert(0, str(HERE.parents[1] / "demos" / "slay_the_spire"))
        import marvin_spire
        marvin_spire.load()
        self._m = marvin_spire

    def letter_probs(self, prompt: str, n_candidates: int) -> dict[str, float]:
        probs, _ = self._m.read_letter_probs(
            self._m.load()["model"], self._m.load()["tokenizer"],
            prompt, [f"c{i}" for i in range(n_candidates)], MANIFEST["think_mode"])
        return {render.letter_for(i): probs[render.letter_for(i)] for i in range(n_candidates)}


class LlamaCppBackend:
    """Next-token logprobs from llama-server; letters must be single tokens."""

    def __init__(self, base: str, api_key: str | None):
        self.base = base.rstrip("/")
        self.key = api_key or os.environ.get("MARVIN_SERVE_KEY", "")

    def _chat(self, prompt: str, logprobs: int) -> dict:
        body = json.dumps({
            "model": "marvin", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": logprobs,
        }).encode()
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def letter_probs(self, prompt: str, n_candidates: int) -> dict[str, float]:
        import math
        letters = [render.letter_for(i) for i in range(n_candidates)]
        resp = self._chat(prompt, 20)
        items = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        tok2lp = {it["token"].strip(): it["logprob"] for it in items}
        lps = {l: tok2lp[l] for l in letters if l in tok2lp}
        if len(lps) != len(letters):
            missing = sorted(set(letters) - set(lps))
            raise RuntimeError(f"llama-server logprobs missing letters {missing}; "
                               f"increase top_logprobs or check tokenization")
        mx = max(lps.values())
        exps = {l: math.exp(v - mx) for l, v in lps.items()}
        z = sum(exps.values())
        return {l: exps[l] / z for l in letters}


def classify(backend, family: str, fields: dict, candidates: list[str], option_lines: list[str]) -> dict:
    prompt = render.render_prompt(family, fields, candidates, option_lines)
    t0 = time.perf_counter()
    probs = backend.letter_probs(prompt, len(candidates))
    letters = [render.letter_for(i) for i in range(len(candidates))]
    best = max(letters, key=lambda l: probs[l])
    return {"decision": candidates[letters.index(best)],
            "confidence": probs[best],
            "probabilities": {candidates[render.letter_for(i)]: probs[render.letter_for(i)]
                              for i in range(len(candidates))},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": MANIFEST["model"], "adapter": MANIFEST["name"]}


SELFCHECK = {
    "family": "spire_play",
    "fields": {"your_hp": "42/80", "energy": "2/3", "your_block": 0,
               "incoming_attack_damage_this_turn": 12,
               "hand": ["Defend (cost 1)", "Strike (cost 1)"],
               "draw_pile": 4, "discard_pile": 1,
               "monsters": ["Jaw Worm hp 30/40 block 0"]},
    "candidates": ["play Defend", "play Strike -> Jaw Worm", "end turn"],
    "option_lines": ["Defend — cost 1", "Strike — cost 1, targets Jaw Worm",
                     "End turn — no more plays this turn"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("mlx", "llama_cpp"), default="mlx")
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    backend = MlxBackend() if args.backend == "mlx" else LlamaCppBackend(args.base, None)
    result = classify(backend, **SELFCHECK)
    print(json.dumps(result, indent=1))
    if args.serve:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                out = classify(backend, req["family"], req["fields"],
                               req["candidates"], req["option_lines"])
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
