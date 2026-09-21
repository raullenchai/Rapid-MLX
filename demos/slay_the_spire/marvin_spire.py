# SPDX-License-Identifier: Apache-2.0
"""Marvin's Garden — Slay the Spire decision engine (demo serving core).

Loads the Bonsai 27B 2-bit base + LoRA adapter once, then answers every
decision with ONE forward pass whose softmax is restricted to the candidate
letter tokens (the executable /v1/classify spec in eval_label_readout.py).

The adapter⇄template contract holds here exactly as in eval: adapters
continued from v15c are evaluated with think-mode "enabled" (the mlx-lm
ChatDataset training view).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[2] / "bench" / "marvins_garden"
sys.path.insert(0, str(BENCH))

from eval_label_readout import read_letter_probs  # noqa: E402
import render  # noqa: E402

MODEL = os.environ.get(
    "MARVIN_MODEL",
    "/Volumes/NVMe-4T/huggingface/hub/models--prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/70f75f3ad081ab840a42f3304c02c27e7f89bfb7",
)
ADAPTER = os.environ.get("MARVIN_ADAPTER", str(BENCH / "adapters" / "spire"))
THINK_MODE = os.environ.get("MARVIN_THINK_MODE", "enabled")  # v15c-lineage contract

from concurrent.futures import ThreadPoolExecutor

_state: dict = {}
# MLX streams are thread-bound; ThreadingHTTPServer handles each request in
# its own thread — so every forward pass runs on ONE dedicated worker thread.
_pool = ThreadPoolExecutor(max_workers=1)


def load():
    return _pool.submit(_load).result()


def _load():
    if _state:
        return _state
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(MODEL, adapter_path=ADAPTER or None)
    _state.update(model=model, tokenizer=tokenizer)
    return _state


def decide(prompt: str, candidates: list[str]) -> dict:
    """One forward pass → per-letter probabilities, picked letter, latency ms."""
    return _pool.submit(_decide, prompt, candidates).result()


def _decide(prompt: str, candidates: list[str]) -> dict:
    s = load()
    letters = [render.letter_for(i) for i in range(len(candidates))]
    t0 = time.perf_counter()
    probs, n_tokens = read_letter_probs(s["model"], s["tokenizer"], prompt, candidates, THINK_MODE)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    chosen_idx = int(np.argmax([probs[l] for l in letters]))
    return {
        "probabilities": {candidates[i]: probs[letters[i]] for i in range(len(candidates))},
        "chosen": candidates[chosen_idx],
        "confidence": float(probs[letters[chosen_idx]]),
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": n_tokens,
    }
