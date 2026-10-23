# Bolt E4B text qualification — 2026-10-23

Environment: Mac Studio M3 Ultra, 256 GiB, macOS 26.5.2; Python 3.12.13,
MLX 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2, Transformers 5.15.1.
All model execution and conversions used the shared GPU lock. No other process
was stopped. This is a shared-host measurement, not an isolated speed claim.
Artifact identities and boundaries are in the
[decision record](../decisions/2026-10-23-nemotron-bolt-e4b.md).

## Correctness and serving

Three prompts, each with thinking false/true: “Say hello in one short sentence.”,
“What is 17 times 23?”, and “Write a Python function that adds two numbers.”
Temperature zero, at most 48 output tokens, fresh cache, prefill step 256.
Both pinned aliases match an independent native implementation on **198/198
emitted token IDs** (six cases each). Four cases emit 48 tokens and two stop
at three tokens, including EOS. This is same-quant implementation parity;
quantization is not expected to preserve every BF16 greedy token.

BF16, both initial CPU conversions, and both temporary artifacts each score 12/12 on
an exact-answer smoke set: multiplication/division/addition, two capitals,
triangle sides, Python `len`/`sum`, oxygen symbol, hexadecimal conversion,
integer sorting, and a secret-code lookup. This small set establishes functional
parity only. Final local GPU conversions match all 1,355 tensors in each pinned artifact.
CPU/GPU raw-text NLL differences are disclosed in the decision record.

Both aliases pass ten HTTP requests each through `rapid-mlx serve <alias>`:
hello; arithmetic with thinking off/on (391, reasoning absent/present);
three automatic `get_weather(city)` calls for Paris/Tokyo/London plus their
synthetic tool-result continuations; and a **8,234-token** prompt returning
`JADE-731`. All requests return 200; all six tool trials return one structured
call, valid JSON arguments, and `finish_reason=tool_calls`. No forced tool choice
or tool grammar was needed. OpenCode dry-run resolves the four-bit alias.
Multimodal inputs and streaming are not qualified by this workload.

## Throughput and memory

Batch one, no draft, no prefix reuse, default cache precision; exactly 64 output
tokens. EOS stopping is disabled only for this synthetic throughput workload.
Prompts repeat/token-trim `The quick brown fox jumps over the lazy dog. ` to
exactly 1,024 or 8,192 tokens. Two sequential measurements, no discarded warmup. The first prefill includes
lazy weight materialization; the second is warm.
Peak is MLX allocator GB, not total process/system memory.

| Artifact | Prompt | Prefill tok/s (runs 1 / 2) | Decode tok/s (runs 1 / 2) | Peak GB |
| --- | ---: | ---: | ---: | ---: |
| Local GPU 4-bit | 1,024 | 1079.64 / 2748.74 | 114.59 / 114.31 | 4.763 |
| Pinned 4-bit | 1,024 | 1129.63 / 2752.79 | 115.52 / 115.33 | 4.763 |
| Local GPU 4-bit | 8,192 | 2755.26 / 2752.17 | 99.47 / 101.00 | 4.938 |
| Pinned 4-bit | 8,192 | 2759.43 / 2765.59 | 100.41 / 100.60 | 4.938 |
| Local GPU 8-bit | 1,024 | 672.18 / 2708.25 | 85.23 / 85.28 | 8.452 |
| Pinned 8-bit | 1,024 | 694.48 / 2681.72 | 84.62 / 84.57 | 8.452 |
| Local GPU 8-bit | 8,192 | 2731.97 / 2725.11 | 76.44 / 74.90 | 8.612 |
| Pinned 8-bit | 8,192 | 2732.75 / 2744.57 | 76.54 / 77.43 | 8.612 |

Final local and pinned weights are tensor-identical; timing differences remain
shared-host variation. All reported runs assert 64 emitted tokens. Earlier
EOS-enabled measurements terminated after 5–64 tokens and are superseded by
this fixed-length table; they must not be compared as 64-token runs.

The CPU conversion experiments were also rerun for exactly 64 output tokens:

- 4-bit: 1,024-token prompt, 115.45 / 114.57 decode tok/s; 8,192-token prompt, 99.84 / 100.75 decode tok/s.
- 8-bit: 1,024-token prompt, 84.08 / 85.00 decode tok/s; 8,192-token prompt, 76.38 / 75.16 decode tok/s.

DFlash has no measured speedup: its available formats fail the existing loader.

## Reproduction

Use the exact source/artifact revisions in the decision record. Download into
the normal Hugging Face cache with `snapshot_download(repo, revision=sha)`.
For conversion, set `SOURCE` to the pinned BF16 snapshot and `OUTPUT` to a new
output directory; run this Python under the shared model lock:

```python
import json, os, shutil
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.gemma4 import Model, ModelArgs
from mlx_lm.utils import quantize_model, save_model, save_config

mx.set_default_device(mx.gpu)
source, output = Path(os.environ["SOURCE"]), Path(os.environ["OUTPUT"])
config = json.loads((source / "config.json").read_text())
config["eos_token_id"] = json.loads((source / "generation_config.json").read_text())["eos_token_id"]
config.pop("audio_config", None)
model = Model(ModelArgs.from_dict(config))
weights = model.sanitize(mx.load(str(source / "model-00001-of-00001.safetensors")))
model.load_weights(list(weights.items()), strict=True)
del weights
model, config = quantize_model(model, config, 64, int(os.environ["BITS"]))
output.mkdir(exist_ok=False)
save_model(output, model, donate_model=True)
save_config(config, output / "config.json")
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy2(source / name, output / name)
```

Run twice with `BITS=4` and `BITS=8`. Preserve the governing license and base
model license with attribution in any staged model card before publication.
For the native BF16 comparison, create a temporary view containing symlinks to
the main shard and tokenizer/config files only; exclude the frozen-KV sidecar.

For benchmarks, set `MODEL` to an artifact directory and run:

```python
import os
import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from rapid_mlx.utils.tokenizer import load_model_with_fallback
model, tokenizer = load_model_with_fallback(os.environ["MODEL"])
tokenizer.eos_token_ids = set()  # Throughput only; retain EOS for real chat.
mx.reset_peak_memory()
seed = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
for count in (1024, 8192):
    prompt = (seed * (count // len(seed) + 1))[:count]
    for repeat in range(2):
        for result in stream_generate(model, tokenizer, prompt, max_tokens=64,
                                      sampler=make_sampler(0), prefill_step_size=256):
            pass
        assert result.generation_tokens == 64
        print(count, repeat, result.prompt_tps, result.generation_tps, result.peak_memory)
```

Serve each alias on an unused port, retaining default preflight:

```sh
rapid-mlx serve nemotron-3.5-bolt-e4b --host 127.0.0.1 --port "$PORT" \
  --disable-disk-caches --disable-prefix-cache --prefill-step-size 256 --max-tokens 768
rapid-mlx launch opencode --model nemotron-3.5-bolt-e4b --dry-run
```

Send `/v1/chat/completions` requests with temperature 0 and
`chat_template_kwargs.enable_thinking` explicitly false/true. Define
`get_weather` with a required string `city`; return synthetic 18°C/sunny tool
responses. For the long prompt, repeat the benchmark sentence 820 times and
append `The secret code is JADE-731. What is the secret code? Reply with only the code.`
Repeat serving with `nemotron-3.5-bolt-e4b-8bit`.
