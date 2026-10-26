# Prefer a longer restorable divergent prefix

A changed question after a long document can unnecessarily repeat most of its
prefill when a short stored prefix is returned before examining an existing,
longer divergent cache entry. Compare the candidate's safely restorable length,
not just its literal common-prefix length. Keep the short prefix available if
restoration refuses. Exact and supersequence priority, candidate discovery,
checkpoint recording, model precision and sampling are unchanged.

This is a selection fix, not a new cache index or a universal prefill speedup.
It does not claim to find the globally best checkpoint among all stored entries.
The metadata-only probe uses the same precedence on successful restoration;
actual restoration can still refuse and fetch then falls back to the short
prefix. Neither probing nor restoring a branch mutates the stored source state.

## Measurement

Baseline: `0002bf9bea60c3fe768cdfa276fd4b2e5a7f4f72`.
Target: `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX`, snapshot
`eb4e4dd38cf2f0a34045347e7a4cbfb60487df2a`; BF16 KV, native MTP max K=3,
prefill step 2048, greedy, thinking disabled. Inputs use deterministic varied
configuration records; expected answers extract a configuration value. Completed
responses only: the harness rejects missing terminal events, missing usage,
errors and length-truncated output. TTFT is client time to first nonempty content.

The benchmark loads one model and alternates the baseline and candidate
selection methods. Everything outside those two methods uses the candidate
checkout. Before **every** measured request it clears the cache and restores the
same saved seed inventory; the preceding measured request cannot warm the next.
Seeds are excluded. The optional `long-only` control runs the baseline selector
with only the *same* longest seed entry: it diagnoses existing restore-path
behavior and is not included in the equal-inventory A/B timing comparison.

Results and source hashes are recorded in the accompanying fixture. Timing is
specific to the stated hardware, package versions and cache inventory. There is
no concurrency, broad task-quality, decode-throughput or cold-cache speed claim.
Small differences in controls are not treated as improvements.

| Hardware | Scenario | A/B pairs | Median TTFT, seconds | Reduction | Cached tokens |
|---|---|---:|---:|---:|---:|
| M3 Ultra | plain-8k | 3 | 19.080 → 0.602 | 96.8% | 1,792 → 7,936 |
| M3 Ultra | middle-edit-8k | 2 | 19.090 → 13.027 | 31.8% | 1,792 → 3,840 |
| M5 Max | plain-8k | 3 | 2.585 → 2.583 | 0.1% | 6,144 → 6,144 |
| M5 Max | middle-edit-8k | 2 | 7.908 → 5.445 | 31.1% | 1,792 → 3,840 |
| M5 Max | plain-32k | 2 | 42.324 → 12.436 | 70.6% | 1,792 → 24,576 |
| M4 Pro | plain-8k | 3 | 61.129 → 1.458 | 97.6% | 1,792 → 7,936 |
| M4 Pro | middle-edit-8k | 2 | 61.131 → 41.569 | 32.0% | 1,792 → 3,840 |
| M4 Pro | plain-32k | 2 | 331.536 → 94.271 | 71.6% | 1,792 → 24,576 |

The M5 8K tail-change control retains only the long cache after prefill-memory
reclamation, so both arms already resume at 6144; there is no material gain.
M3 8K and both middle-edit samples have identical A/B output hashes. M5 32K
candidate and long-only control have identical output hashes and resume at
24576. Its baseline uses 1792 and emits fenced JSON; the candidate and control
emit plain JSON. All measured answers have the expected key/value semantics.
This does not establish broad quality equivalence.

The final 8-file regression suite passes 244 tests.

### M4 Pro follow-up

The same implementation at `230fa59023c7a0fb7b2c577ed4fd9c204f7fe042`
was measured on an M4 Pro (12 CPU / 16 GPU cores, 48 GiB, macOS 26.5.1).
Python 3.12.13, MLX 0.32.3, MLX-LM 0.31.3 and MLX-VLM 0.7.2 match
both earlier hosts. All 17 model files were verified against the same snapshot.
The 8K campaign uses the original two-seed cases; the separate 32K campaign
uses the same short-long-short seed order as M5. Each A/B trial restores an
identical per-case inventory. Both campaigns completed under the cooperative
GPU lock with a 27 GiB working-set estimate and the default 4 GiB reserve.

All 14 measured A/B responses and both 32K long-only controls have the expected
strict JSON values. Within each scenario every output hash is identical,
including the 32K controls. The long-only control median is 94.126 seconds,
resuming at the same 24,576 tokens as the candidate. Seed requests, model loading
and file transfer are excluded from these medians. System swap was already
354.62 MiB before the campaign, observed at 466.25 MiB during 32K, and 458.25 MiB
afterward; this is not a swap-free or exclusive CPU/power measurement.

This extends the hardware evidence to M4 Pro; it does not qualify other models,
MTP-disabled operation, or other inference backends. The optimization selects an existing
cache restoration path rather than changing the MTP algorithm. Its gains remain
conditional on a longer restorable entry competing with a shorter strict prefix.

## Output and compatibility limits

Changing how much prefill is replayed can change floating-point execution and
greedy output. A prior 32K experiment returned the same JSON key/value answers
but added Markdown fences on the longer-cache path. Plain-language JSON-only
instructions are not a schema guarantee; token-identical output is measured,
not assumed. The diagnostic long-only control distinguishes selector changes
from the pre-existing long-cache restoration path.

Strict JSON-schema requests use a separate guided generation path. The four
8K schema checks returned valid, identical JSON, but do not exercise this prefix
selector and are not evidence of its performance or output equivalence.

Regression coverage includes ordinary KV and recurrent caches, shorter/equal
checkpoints, absent or unreachable checkpoints, restore refusal, source
immutability, LRU and accounting, probe parity on success, sliding caches,
wrapper/pooling rewind metadata and existing exact/supersequence behavior.
A refused longer restore must preserve the valid shorter prefix. Unknown or
unsafe rewind metadata must not displace it.

## Reproduce

Use a local model snapshot with its MTP sidecar and the package versions in the
fixture. The benchmark starts an isolated loopback server, disables downloads
and disk caches, and cleans up its own server. Run on an otherwise idle host
under the cooperative GPU lock. No production service needs to be stopped.

```sh
export PREFIX_BENCH=/private/tmp/rapid-prefix-selection
export PREFIX_MODEL="$HOME/.cache/huggingface/hub/models--rapid-mlx--Qwen3.8-27B-4bit-MTP-MLX/snapshots/eb4e4dd38cf2f0a34045347e7a4cbfb60487df2a"
mkdir -p "$PREFIX_BENCH"
git show 0002bf9bea60c3fe768cdfa276fd4b2e5a7f4f72:rapid_mlx/memory_cache.py > "$PREFIX_BENCH/baseline_memory_cache.py"
```

Generate the input cases (the 8K and 32K old instructions deliberately differ;
short old instructions can end exactly at an already reusable boundary):

```python
import json, os
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ["PREFIX_MODEL"], local_files_only=True)
records = "".join(
    f"Record {i}: region zone-{i%17}; timeout {10+i%90}; workers {1+i%8}; status ready.\n"
    for i in range(5000)
)
ids = tok.encode(records, add_special_tokens=False)
header = "Configuration reference. Read CONFIG_COLOR from these notes.\n"
old = "\nExtract the CONFIG_COLOR value. Output only JSON with a single key color."
long_old = old + (
    " Read the configuration above carefully. Preserve the exact value. Do not add"
    " commentary, explanation, code fences, other keys, or any text outside the JSON object."
)
new = "\nAnswer with JSON only: keys color containing CONFIG_COLOR and verified set to true."
short = header + tok.decode(ids[:1800]) + '\nReturn only {"seed":1}.'

def body(text):
    return {"messages": [{"role": "user", "content": text}]}

def case(name, seed, query, color, repeats):
    return dict(name=name, seeds=[body(short), body(seed)], query=body(query),
                expected={"color": color, "verified": True}, repeats=repeats)

background = header + tok.decode(ids[:8192-180]) + "\nCONFIG_COLOR=blue\n"
cases = [case("plain-8k", background + long_old, background + new, "blue", 3)]
before = header + tok.decode(ids[:4000]) + "\nCONFIG_COLOR="
after = "\n" + tok.decode(ids[4000:8000])
cases.append(case("middle-edit-8k", before + "blue" + after + old,
                  before + "green" + after + new, "green", 2))
background = header + tok.decode(ids[:32768-180]) + "\nCONFIG_COLOR=blue\n"
cases.append(case("plain-32k", background + old, background + new, "blue", 2))
cases[-1]["modes"] = ["baseline", "candidate", "long-only"]
Path(os.environ["PREFIX_BENCH"], "cases.json").write_text(json.dumps(cases))
```

The M5 8K campaign uses only the first two cases. The separate M5 32K
campaign uses the last case and adds `cases[-1]["seeds"].append(body(short))`:
after reading the long document, send the short request again, then return to
the long document. This leaves both short and long entries resident; seed order
is part of the workload, not a hardware-controlled constant. Do not compare
numbers across hosts as a hardware-only speed ratio.

```sh
python scripts/large-model-run.py --working-set-gb 64 --lock-timeout 120 -- \
  python scripts/benchmark_prefix_selection.py \
  --model "$PREFIX_MODEL" \
  --baseline-source "$PREFIX_BENCH/baseline_memory_cache.py" \
  --cases "$PREFIX_BENCH/cases.json" --output "$PREFIX_BENCH/results"
```

The output directory must be new. `results.json` contains per-request timings,
usage, output hashes and raw strict-JSON checks; `inventories.json` records every
restored cache inventory. `command.json` binds the two selector files by SHA256.
Use `--working-set-gb 20 --threshold-gb 0` for the stated 8K-only M5 run while
retaining the memory precheck and cooperative lock. The M5 32K run used a
27 GiB estimate plus the default 4 GiB reserve; it started only after that
precheck passed. M3 retained an idle desktop service; neither run is an
exclusive CPU/power experiment.
