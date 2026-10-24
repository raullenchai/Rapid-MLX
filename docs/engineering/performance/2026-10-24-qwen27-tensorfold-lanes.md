# Qwen27 TensorFold: single-request lanes 1 versus 4

## Scope and method

Issue #4847 promotes the lane setting to `serve --tensorfold-lanes` and aligns
the Qwen27 pool's advertised capacity, child configuration and HTTP admission
at four. Standalone serving remains serial by default. Desktop has no new
control. No running service is changed by this work.

The measurement uses the cached qualified Qwen3.8-27B target at
`70ae7fac63274ff2eac54152031433374cb80f2f` and drafter at
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. All snapshot resolution is offline;
no models are downloaded. Environment: Apple M3 Ultra, 256 GiB unified memory,
macOS 26.5.2, Python 3.12.14, MLX 0.32.3, TensorFold 0.6.6 at
`cb2ebf0540f42604e2759b2ddef497861e928248`, the supporting `mlx-lm` 0.31.3 and Transformers 5.15.1, and the Rapid-MLX adapter in this PR.

Each cell is a fresh process with one untimed 32-token warmup, then three
sequential requests capped at 256 tokens each. The same Python, science and
creative-writing prompts are used in every cell; the complete text is below.
Sampling is greedy (`temperature=0`, seed 1234), thinking is off, DFlash2 is on,
and prefix caching is disabled to prevent reuse from biasing later requests.
The order is 1/4, 4/1, 1/4 across three repetitions. The outer GPU lock covers
all six cells. There is never more than one request in flight. Model loading
and warmup are excluded from timing but included in peak MLX allocation.
The 8,192-token context limit is held constant.

## Results

All 18 measured requests produced 256 tokens, with zero cached tokens. For each
prompt, the SHA-256 of the returned content matched in all six cells. Times
below include prefill and decode but exclude startup and the warmup. Throughput
is 768 tokens divided by the sum of three sequential request times, **not**
concurrent aggregate throughput.

| Repeat | Lanes | Python s | Tides s | Story s | Total s | End-to-end tok/s | Median TTFT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 3.393 | 4.328 | 6.379 | 14.100 | 54.47 | 259.4 |
| 1 | 4 | 3.460 | 4.276 | 5.750 | 13.486 | 56.95 | 252.5 |
| 2 | 1 | 3.446 | 4.196 | 5.623 | 13.265 | 57.90 | 250.2 |
| 2 | 4 | 3.274 | 4.121 | 5.432 | 12.827 | 59.87 | 248.7 |
| 3 | 1 | 3.359 | 4.226 | 5.625 | 13.209 | 58.14 | 251.8 |
| 3 | 4 | 3.336 | 4.103 | 5.377 | 12.817 | 59.92 | 249.8 |

Median three-request total: **13.265 s at one lane versus 12.827 s at
four lanes** (3.3% shorter in this sample). This sample does not reproduce
the earlier single-request slowdown. Three repetitions and the observed timing
variation do not establish a general speedup; the purpose is to expose the
single-request cost of enabling lanes on the same workload.

| Lanes | Peak MLX GiB, including load/probe | Maximum sampled process footprint GiB |
| --- | --- | --- |
| 1 | 18.175 | 27.840 |
| 4 | 18.699 | 20.903 |

Content SHA-256 (identical across repeats and lane settings):

- Python: `e299fdb9b8ee9fb17e9f06217e97e628ce3f6a873c910c739a487f7cd694118a`
- Tides: `c08c05063172b6f856222b9e1b784d389aa04ae599ab2482acacfed907bcbe17`
- Story: `6f0b6b7037bad1674006c1ede4060b2ddee2ab51002a5b743b1df52a8adf754d`


## Memory headroom at the 48 GiB floor

TensorFold's concurrent admission uses 70% of physical RAM, subtracting memory
held elsewhere. A 48 GiB host starts from a 33.6 GiB share, leaving 14.4 GiB
outside that share. The run's memory probe reports a stream of about 166 MiB
at 64 tokens and 391 MiB at 2,112 tokens, growing by 64 KiB per token, with a
shared round of up to 2.75 GiB at four streams. Weights are shared; four lanes
do not mean four model copies.

The [earlier concurrent pilot](2026-10-12-tensorfold-concurrency-pilot.md)
measured 16.81–18.55 GiB peak MLX allocations at four active lanes. The new
single-request measurements above include the four-lane startup memory probe.
Use the observed process footprint as well as MLX allocation when planning:
non-MLX memory is material, and an assumed 3 GiB allowance alone does not
explain every measured process footprint. Neither metric should be added to
the other; process footprint already includes Metal allocations.

The largest sampled four-lane process footprint was 20.903 GiB, leaving
about 12.70 GiB under a 33.6 GiB share and 27.10 GiB of physical RAM on a
48 GiB host before other applications. Conservatively using the largest
footprint across **either** setting (27.840 GiB) leaves 5.76 GiB under that
share. Together with the earlier four-active-lane measurements, this supports
the short-request pool workload, with runtime memory admission still needed
for larger prompts and co-resident application pressure.

These measurements do **not** qualify four maximum-length contexts on physical
48 GiB hardware. `TENSORFOLD_MEMORY_LIMIT_GB=33.6` was exported identically for
all cells, but source inspection and the startup diagnostic show that this
adapter's concurrent admission uses physical RAM when no explicit prompt
memory budget is supplied. It reported about 169 GiB available on this host,
not 33.6 GiB. The environment setting therefore does not make this a simulated
48 GiB test. The 48 GiB figures above are headroom reasoning, not a measured
capacity guarantee. Background operator services were not stopped; timings
must be read with their observed variation. Footprint is sampled after each
request, not continuously during loading.

Existing memory admission remains responsible for refusing requests that do
not fit. Pool overflow is separately covered by stream and non-stream tests:
four requests run, the fifth receives `503` / `Retry-After: 1`, and admitted
requests finish after capacity is released.

## Reproduction

Use the qualified runtime above and save the following harness as
`qwen27-lanes-ab.py` in a checkout of this PR. It requires the two pinned
snapshots already in the standard Hugging Face cache.

```python
import asyncio, hashlib, importlib.metadata, json, os, platform, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
import mlx.core as mx
from tensorfold.server.memory_budget import process_footprint
from rapid_mlx.speculative.tensorfold_qwen27 import TensorFoldQwen27Backend

PROMPTS = [
    "Write a complete Python LRU cache implementation with type annotations, a detailed explanation, and at least five unit test examples. Explain every method.",
    "Write a detailed essay of at least 1500 words explaining how tides work, including lunar and solar effects and coastal geography.",
    "Write a detailed story of at least 1500 words about a botanist discovering a forgotten garden on a mountain. Include dialogue and vivid descriptions.",
]


async def main():
    lanes = int(sys.argv[1])
    target = snapshot_download(
        "Vontra/Qwen3.8-27B-MLX-4bit",
        revision="70ae7fac63274ff2eac54152031433374cb80f2f",
        local_files_only=True,
    )
    draft = snapshot_download(
        "z-lab/Qwen3.8-27B-DFlash2",
        revision="50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
        local_files_only=True,
    )
    mx.reset_peak_memory()
    backend = TensorFoldQwen27Backend.load(
        target,
        draft,
        served_name="qwen27-ab",
        context_window=8192,
        max_tokens=256,
        lanes=lanes,
        enable_prefix_cache=False,
    )

    async def one(i, text, count):
        prompt = list(
            backend._app.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        start = time.perf_counter()
        first = None
        reply = None
        async for event in backend.stream(
            str(i),
            prompt,
            max_tokens=count,
            sampling={"temperature": 0.0, "seed": 1234, "draft": True},
        ):
            if event.error is not None:
                raise event.error
            if event.delta is not None and first is None:
                first = time.perf_counter()
            if event.reply is not None:
                reply = event.reply
        elapsed = time.perf_counter() - start
        return {
            "prompt": i,
            "seconds": elapsed,
            "ttft": first - start,
            "tokens": reply["completion_tokens"],
            "sha256": hashlib.sha256(reply["content"].encode()).hexdigest(),
            "cached_tokens": reply["cached_tokens"],
            "peak_gib": mx.get_peak_memory() / 2**30,
            "footprint_gib": process_footprint() / 2**30,
        }

    try:
        await one("warmup", "Explain binary search in detail.", 32)
        results = [await one(i, p, 256) for i, p in enumerate(PROMPTS)]
        output = {
            "lanes": lanes,
            "macos": platform.mac_ver()[0],
            "mlx": importlib.metadata.version("mlx"),
            "tensorfold": importlib.metadata.version("tensorfold"),
            "results": results,
        }
        Path(sys.argv[2]).write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output), flush=True)
    finally:
        await asyncio.get_running_loop().run_in_executor(None, backend.close)


asyncio.run(main())
```

Run from the checkout using that environment's Python:

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
export TENSORFOLD_MEMORY_LIMIT_GB=33.6
lockf -k /var/tmp/rapid-mlx-large-model.lock sh -c '
  for cell in 1:1 1:4 2:4 2:1 3:1 3:4; do
    repeat=${cell%:*}
    lanes=${cell#*:}
    python qwen27-lanes-ab.py "$lanes" "run-${repeat}-lanes-${lanes}.json"
  done
'
```
