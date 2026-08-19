# Gemma 4 12B on an 18 GB Mac

Measured on a MacBook Pro M3 Pro (12-core, 18 GB, macOS 15.6.1) against
`mlx-community/gemma-4-12B-it-4bit` (6.3 GiB on disk), rapid-mlx 0.11.9,
mlx 0.31.2, mlx-vlm 0.6.3. B=1, temperature 0, 3 reps, medians reported.

**Not validated on 16 GB, and the numbers here argue against assuming it
transfers.** The D-METAL-CAP admission cap scales from the device's
`max_recommended_working_set_size`, so the 10.3 GB cap this guide works
against becomes roughly 9.1 GB on a 16 GB machine — while the steady
state recorded below is already 8.4–8.7 GB before a request's own KV
projection is added. On 16 GB, expect to have to give something up —
shorter contexts, a smaller `--cache-memory-mb`, or dropping
`--hybrid-cache-entries` back to its default 0. This recipe as written
was not measured there.

## The command

```bash
rapid-mlx serve gemma-4-12b-4bit \
  --no-mllm \
  --speculative-config '{"method":"suffix","num_speculative_tokens":8}' \
  --hybrid-cache-entries 2 \
  --cache-memory-mb 768 \
  --max-tokens 2048 \
  --gpu-memory-utilization 0.85
```

Result: **35.5 tok/s** on a code-edit workload (vs 16.3 tok/s with none of
the flags), TTFT 0.54 s, steady-state Metal 8.4–8.7 GB, and five
consecutive `aider` edit rounds landing clean with the project's test
suite green after each.

## Why each flag

### `--no-mllm` — required, not optional

`gemma-4-12b-*` ships `model_type: gemma4_unified` with vision *and*
audio sub-configs, so auto-detection routes it to the MLLM lane
(`BatchedEngine loaded: … (mllm=True)`). For text serving that lane costs
you:

- **PFlash is unavailable** — `validate_model_support` rejects the
  MLLM/VLM lane outright.
- **Prompts hard-cap at 8192 tokens** — `Total prompt tokens (21174)
  exceeds the per-batch cap (8192 = prefill_step_size 8192 × 1
  request(s))`, on a model advertising 262144 context.
- **Higher TTFT** — 0.611 s vs 0.352 s on the same code-generation
  prompt (−42 %).

Pass `--no-mllm` unless you are actually sending images or audio.

### `--speculative-config '{"method":"suffix",…}'` — workload-specific

SuffixDecoding is a *workload* flag, not a model accelerator. Measured
here:

| workload | suffix off | suffix on | |
|---|---:|---:|---|
| code edit (re-emit a file with one change) | 16.3 tok/s | **35.5 tok/s** | 2.18× |
| chat (distinct prompt per rep) | 17.9 tok/s | 16.7 tok/s | 0.93× |
| pure generation | 17.8 tok/s | 17.4 tok/s | 0.98× |

The same A/B on a Mac mini M2 Pro 32GB: code edit 19.6 → 22.2 (1.13×),
chat 21.7 → 18.6 (0.86×). Acceptance is identical on both machines
(0.823); the spread is entirely how the two chips scale a K-wide verify
forward — 3.44× a 1-wide forward on M3 Pro, 7.07× on M2 Pro. Measure your
own before assuming the M3 numbers transfer.

Turn it on for agent / code-edit traffic, which re-emits most of its
input. Leave it off for chat. This reproduces the split already recorded
in [suffix_decoding_eligibility.md](../suffix_decoding_eligibility.md).

### `--max-tokens 2048` — the one that makes OpenAI clients work

What matters here is not Gemma 4 12B's real KV footprint but the one the
admission gate projects, and they differ by 4x.

The real growing cost is **16 KB per token**: only 8 of 48 layers are
full-attention, and those use `num_global_key_value_heads` (1) with
`global_head_dim` (512), not the `num_key_value_heads` (8) /
`head_dim` (256) the 40 sliding layers use. The sliding layers are
window-bounded at 1024 tokens, so past that they stop growing entirely —
per-token arithmetic overstates long contexts for this architecture.

The gate charges **64 KB per token** instead. `kv_estimation.py` reads the
global dims correctly and then clamps with
`max(global_per_layer, uniform_per_layer)`, a floor meant to protect
against configs whose global dims exceed the base. Gemma 4 is the inverse
case — its global layers are *cheaper* per layer than its local ones — so
the floor rounds 2048 B/layer up to 8192 B/layer. The projection is
conservative by design and never under-counts, which is the right default
against a Metal cliff that aborts the process rather than raising. It is
also why the flags below matter more on this model than the real
footprint would suggest.

The gate projects `(prompt_tokens + max_tokens) × 64 KB` plus a
per-request sliding term, so both halves of that sum matter. With a
client that omits `max_tokens` (aider does — the server logs
`max_tokens=None`, and the server then substitutes its own default)
every request was rejected before generating a token:

```
503: Metal active 7.0GB + reserved KV 0.0GB + projected KV 5.7GB
     would exceed gpu_memory_utilization cap 10.3GB (D-METAL-CAP)
```

Setting `--max-tokens 2048` was what turned that into a working server —
measured, on the run this guide records. The exact arithmetic behind the
5.7 GB figure is not reconstructed here: the projection also carries the
prompt and the sliding term, and the server's own omitted-`max_tokens`
default participates, so quoting a single cause for that number would be
guesswork. What is reproducible is the shape — the projection scales with
the token budget you let a request reserve, and bounding it fixes the
rejections. 2048 is far above what edit traffic actually uses (the aider
rounds above returned 341–583 tokens).

### `--hybrid-cache-entries 2` and `--cache-memory-mb 768` — budget, not leak

Retained prefix-cache entries are not free, and on this architecture the
sliding layers dominate an entry rather than the per-token term: 40 of 48
layers hold a 1024-slot window at 320 KB per slot, so an entry converges
to a few hundred MB regardless of how short the prompt was. With
`--hybrid-cache-entries 8`, steady-state Metal climbed from 7.0 GB to
8.3 GB — measured, not derived — and the *next* request was then rejected
on a 2.1 GB projection against a 10.3 GB cap.

Memory does **plateau** — five identical requests in a row held at
8.60 GB, so this is a budgeting problem, not a leak.

Note the direction: `gemma-4-12b-4bit` is `is_hybrid: false`, so
`_resolve_hybrid_cache_entries` leaves the count at the parser default of
**0**, and passing `--hybrid-cache-entries 2` *opts in* to retaining two
otherwise non-trimmable entries. It buys prefix reuse across turns on a
sliding-window model at the cost of some headroom — a trade, not a
reduction from a wasteful default. 8 was measurably too many on 18 GB
(the rejection above); 2 was the largest value that stayed clear of the
cap on this machine. If you are not running repeat-prefix traffic, the
default 0 is the cheaper choice.

### `--gpu-memory-utilization 0.85`

Leaves the cap above the ~8.6 GB plateau with room for the projection.
Do not raise this much further on 18 GB: a long-context run was observed
peaking at 17.19 GB of Metal, and Apple Silicon firmware can panic rather
than raise OOM (issue #324).

## The persisted prefix cache will break long requests

Long prompts appeared to fail on their own — a needle-in-haystack at
10358 tokens returned `'4yclycl+"'`, and later crashed the server
outright with

```
libc++abi: [METAL] Command buffer execution failed: Insufficient Memory
Fatal Python error: Aborted
```

It is not a context-length limit and not a quantization artifact. Driving
the *same* checkpoint through `mlx_lm.generate` with a plain
`make_prompt_cache` — no scheduler, no prefix cache — answers correctly
at **13221 tokens** (cache types `['KVCache', 'RotatingKVCache']`, peak
9.43 GB). The engine is what fails, and the cause is on disk:

```
[cache_persist] LOADED 5 entries from ~/.cache/rapid-mlx/prefix_cache/... (1921MB total)
```

The prefix cache is restored at startup and consumed **1.9 GB before the
first request** — the bulk of the "steady state" you observe — and none
of it is visible to the pre-flight memory warning. Combined with the
gate's 64 KB/token projection, that is what pushed a 10k-token request
past the Metal allocation.

**This was observed on the default cache budget, which is why
`--cache-memory-mb 768` is in the recipe above.** With that flag set,
`MemoryAwarePrefixCache.load_from_disk` stops staging once it would
exceed the limit, so a 1.9 GB restore cannot happen in the first place —
a reader following the recommended command should not hit this. It is
recorded because the symptom (garbage output and a hard abort at ~10k
tokens, on a run that works fine at 13k through `mlx_lm.generate`) is
easy to misread as a context-length or quantization failure, and because
the persisted cache survives restarts: a directory grown under earlier,
looser settings is still on disk.

Any one of these makes 10358 tokens pass:

```bash
rm -rf ~/.cache/rapid-mlx/prefix_cache/<model-dir>   # clear the persisted cache
rapid-mlx serve … --disable-prefix-cache
rapid-mlx serve … --no-memory-aware-cache
```

Clearing the persisted cache unblocks the request in front of you at the
cost of one cold prefill, but it is **not a durable fix**: under a loose
budget `MemoryAwarePrefixCache.store()` grows the cache straight back and
shutdown persists it again, so the next long request — or the next
restart — can reproduce the abort. Treat `rm -rf` as the immediate
unblock and `--cache-memory-mb 768` (already in the recipe above) as the
fix that keeps it from recurring; `--disable-prefix-cache` is the
belt-and-braces option if you would rather pay full prefill every turn
than think about the budget.

Note that the failure mode is a **process abort, not a 503**: MLX raises
an uncaught C++ `std::runtime_error` on Metal OOM, so exceeding the real
allocation kills the server rather than degrading. The D-METAL-CAP
admission gate is the only thing standing between you and that, which is
why under-setting `--metal-cap-kv-bytes-per-token` is dangerous.

## The admission gate models KV only, not the prefill working set

This is the hardening gap worth knowing about, and it is easy to make
worse rather than better.

`_enforce_metal_cap_at_admission` projects a request's **KV cache** and
compares it against the Metal cap. It does not model the transient
working set of the prefill itself. On Gemma 4 12B — 48 layers,
`global_head_dim` 512 — prefilling a few thousand tokens in one go
allocates well beyond its own KV, and none of that is in the projection.

There is also a dead band, though it needs both triggers stated to be
diagnosed correctly. `evict_prefix_cache_under_pressure` fires on
*either* Metal active exceeding `metal_pressure_evict_fraction` (default
0.9) of the cap, **or** the memory-aware cache ledger reaching 0.9 of its
own budget — with `--cache-memory-mb 768` that second one trips near
691 MiB, independent of the Metal ratio. Admission, meanwhile, rejects as
soon as `active + reserved + projected` reaches the cap.

So a server can sit in a band where admission refuses everything and
neither eviction trigger has been reached. Measured here: after four
aider rounds the server settled at active 9.17 GB against a 10.6 GB cap
— 0.865, under the Metal trigger — and 503'd every request from then on.
Only a restart cleared it.

If you land in that state, check the cache ledger before concluding the
eviction path cannot run: a cache well under its own budget is what puts
you in the dead band, and a cache near it means something else is wrong.
Widening the dead band is not the fix — an earlier attempt at exactly
that is in the "what did not work" section below, and it traded a
recoverable 503 for a process abort.

**Closing the dead band naively makes things worse.** Evicting at
admission and re-admitting was tried: it fired 7 times, correctly
reclaimed memory (active 9.17 → 8.6 GB, cap 11.0 GB), admitted a
4643-token request whose KV projected to well under the 2.4 GB of
headroom — and the process died in prefill:

```
libc++abi: [METAL] Command buffer execution failed: Insufficient Memory
Fatal Python error: Aborted
```

The conservative dead band was accidentally protective: by keeping
`active` low it left slack that the unmodeled prefill spike happened to
fit inside. Trading a recoverable 503 for a process abort is a bad trade,
so that change was reverted. Making admission more permissive is only
safe *after* the prefill working set is modelled — chunking the prefill
and charging it per chunk would be the honest fix, and it is a real piece
of work, not a flag.

Two things follow for anyone tuning this:

- Treat a rising `rapid_mlx_metal_active_memory_bytes` as the signal to
  restart the server, not as headroom to spend.
- Do not lower `--metal-cap-kv-bytes-per-token` to "recover" the
  headroom a KV codec should have bought. The gate's over-estimate is
  covering for the prefill it cannot see.

## Other known limits

- **TurboQuant runs, but does not buy you stability.** The fused Metal
  kernel is available on mlx 0.31.2
  (`rapid_mlx_turboquant_fused_kernel{status="available"}`; it reported
  `fallback` on 0.32.0, so check the gauge rather than assuming). Four
  consecutive aider rounds passed with `--kv-cache-turboquant k8v4` and
  the test suite stayed green. But the codec is invisible to the
  admission gate, which keeps projecting fp16 — so once Metal active
  climbs, requests are rejected exactly as if the codec were off. The
  soak ended with the server 503'ing a 4k-token needle it had served
  minutes earlier. Compression you cannot spend at admission time is not
  headroom. Note that MLX core has no
  quantized-KV `scaled_dot_product_attention` (ml-explore/mlx#3404), so
  every implementation is either a full dequant or custom Metal kernels.
  For reference, `lovelacemadeline/gemma4-turboquant-mlx` gets PolarQuant
  2-bit KV working on Gemma 4 by dispatching *per layer type* — a fused
  polar MMA kernel on the D=256 sliding layers, and dequant into Apple's
  `steel_attention` on the D=512 full-attention layers, which it reports
  as ~6× faster than polar MMA at that dimension. Gemma 4's two head dims
  appear to need two different kernels.
- **MTP cannot be enabled on Gemma 4 at all.** Not "slower at 4-bit" —
  the server refuses to start. Both
  `--speculative-config '{"method":"mtp","model":"…-assistant-4bit",…}'`
  and the same with `--force-spec-decode` exit with

      error: MTP speculative-config requires either a Qwen3.5 / Qwen3.6
      checkpoint with mtp_num_hidden_layers >= 1 in config.json.
      Assistant sidecars are reserved for future validated support and do
      not make this model eligible.

  This is the fail-closed gate from
  [`docs/specdecoding-validation-notes.md`](../specdecoding-validation-notes.md)
  (cli.py), and no
  flag bypasses it. Enabling MTP here is a feature to implement, not a
  flag to set. Two upstream reference points for whoever does:
  Google's design has the MTP head cross-attend the *main model's* KVs
  and share its embedding table (not an independent sidecar, which is
  what the failed rapid-mlx attempt used), and Ollama's implementation
  (ollama/ollama#15980) calls out "changes to the rotating cache to be
  able to handle MTP correctly" — Gemma 4's sliding-window cache is the
  likely shape of the recorded greedy divergence. Note also that Ollama
  reports BF16 as the operating point and a *regression* at int4
  (60 → 45 tok/s at 41 % acceptance); the ~90 % aider-polyglot number is
  on nvfp4, so the payoff at 4-bit is unproven even once it works.
- **This guide was validated with the `[vision]` extra installed.** The
  `gemma4_unified` architecture classes live in mlx-vlm and mlx-lm ships
  no `gemma4_unified` module of its own, so that is the configuration
  everything here was measured on. `load_gemma4_unified_text` does carry
  a vendored fallback for installs without mlx-vlm, and `bench_command`
  routes through `load_model_with_fallback` rather than binding
  `mlx_lm.load` (#1408), so a bare install may well work — it simply was
  not exercised here, so treat `[vision]` as the tested path rather than
  a hard requirement.

## Reference points

A well-optimized Gemma 12B on M2/M3-class hardware is generally reported
in the 30–50 tok/s range, and Ollama's MLX multi-token-prediction work
reports ~90 % faster generation on the aider polyglot benchmark for
Gemma 4 12B nvfp4 on an M5 Max. The 36.0 tok/s here is within that band
for an M3 Pro at 4-bit, reached with SuffixDecoding rather than MTP —
rapid-mlx's Gemma 4 assistant-sidecar MTP path is documented as failing
greedy-lossless validation and fails closed (see
[`docs/specdecoding-validation-notes.md`](../specdecoding-validation-notes.md)).
