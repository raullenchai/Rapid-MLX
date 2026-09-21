# SPDX-License-Identifier: Apache-2.0
"""Marvin's Garden — Slay the Spire decision engine (demo serving core).

Loads the Bonsai 27B 2-bit base + LoRA adapter once, then answers every
decision with ONE forward pass whose softmax is restricted to the candidate
letter tokens (the executable /v1/classify spec in eval_label_readout.py).

Thread model (empirically pinned down on M3 Ultra / macOS 15):
- MLX's Metal command buffers are thread-affine. `load` and every forward
  pass must run on the SAME thread, and that thread must be the MAIN one:
  load on a pool thread + forward on the same pool thread deadlocks
  (0% CPU forever), while forward on any other thread raises
  "There is no Stream(cpu, 0) in current thread".
- Therefore: call start_main_worker() from the process main thread. All
  other threads (ThreadingHTTPServer handlers) go through decide()/load(),
  which marshal the call onto the main thread via a job queue. Scripts that
  never start the worker get plain in-thread execution.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
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
ADAPTER = os.environ.get("MARVIN_ADAPTER", str(BENCH / "adapters" / "spire_r1"))
# Empirical: this lineage was trained with the mlx-lm ChatDataset view where
# the label letter directly follows the think opener — the "disabled"
# rendering in eval_label_readout (enabled vs disabled scored within 0.4pt).
THINK_MODE = os.environ.get("MARVIN_THINK_MODE", "disabled")

_state: dict = {}
_jobs: queue.Queue = queue.Queue()
_worker_started = threading.Event()


def start_main_worker() -> None:
    """MUST run on the process main thread: load once, serve requests forever."""
    _load()
    _worker_started.set()
    while True:
        fn, args, box = _jobs.get()
        try:
            box["value"] = fn(*args)
        except BaseException as e:  # surfaced to the caller
            box["error"] = e
        box["done"].set()


def _submit(fn, *args):
    if _worker_started.is_set():
        box: dict = {"done": threading.Event()}
        _jobs.put((fn, args, box))
        box["done"].wait()
        if "error" in box:
            raise box["error"]
        return box["value"]
    return fn(*args)  # no worker (headless scripts): run in current thread


def load():
    return _submit(_load)


def _load():
    if _state:
        return _state
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(MODEL, adapter_path=ADAPTER or None)
    _state.update(model=model, tokenizer=tokenizer)
    return _state


def decide(prompt: str, candidates: list[str]) -> dict:
    """One forward pass → per-letter probabilities, picked letter, latency ms."""
    return _submit(_decide, prompt, candidates)


def run(fn, *args):
    """Run fn(*args) on the inference (main) thread — for callers that need
    engine/MLX calls serialized with the model (e.g. HTTP handler threads)."""
    return _submit(fn, *args)


def _decide(prompt: str, candidates: list[str]) -> dict:
    s = _load()
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
