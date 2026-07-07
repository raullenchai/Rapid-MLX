# SuffixDecoding eligibility — explicit workload flag

> **Default**: `--speculative-config '{"method":"suffix"}'` is **OFF** and must stay opt-in.
> SuffixDecoding is a workload-specific optimization, not a general model
> accelerator. It helps only when the request generates long, repetitive
> continuations with high prompt/output n-gram overlap.
>
> Prefer the unified form for new usage:
>
> ```bash
> rapid-mlx serve gemma-4-12b-4bit \
>   --speculative-config '{"method":"suffix","num_speculative_tokens":8}'
> ```

## When To Use It

Use SuffixDecoding only when the user explicitly knows their traffic is
high-overlap, for example:

- code editing that re-emits most of the input file/function;
- prompt-copy or template-fill tasks;
- tool-call XML loops with repeated structure;
- agent loops that emit repeated scaffolding across many turns.

Leave it off for ordinary chat, diverse prose, low-overlap JSONL, and
unknown traffic mixes. Rapid-MLX does not infer the user's workload yet;
future adaptive gating may sample early acceptance rates per request, but
that is not implemented today.

## Current Local Validation

Deep local benches on 2026-07-06 showed the expected split: Suffix helps
some high-overlap Gemma 4 workloads, but regresses GPT-OSS and Qwen.

| Model | Workload | Median speedup | Recommendation |
|---|---|---:|---|
| `gemma-4-12b-4bit` | copy code | 1.54x | Candidate |
| `gemma-4-12b-4bit` | long code edit | 1.56x | Candidate |
| `gemma-4-12b-4bit` | repeated tool XML | 1.49x | Candidate |
| `gemma-4-12b-4bit` | JSONL repeat | 0.77x | Avoid |
| `gemma-4-31b-4bit` | copy code | 1.54x | Candidate |
| `gemma-4-31b-4bit` | long code edit | 1.44x | Candidate |
| `gemma-4-31b-4bit` | repeated tool XML | 1.52x | Candidate |
| `gemma-4-31b-4bit` | JSONL repeat | 0.76x | Avoid |
| `gpt-oss-20b` | deep high-overlap sweep | ~0.86x | Avoid |
| `qwen3.6-27b-8bit` | forced deep high-overlap sweep | 0.88-0.95x | Avoid |

This is why SuffixDecoding is exposed as an explicit flag only. It is
reasonable to try on Gemma 4 code-edit / copy / repeated tool-XML traffic;
it is not recommended for GPT-OSS, Qwen3.6, or general chat.

## Tier definitions

| Tier | Trigger | Startup hint | What you should do |
|---|---|---|---|
| **AGENT** | `tool_loop ≥ 1.8x` AND `min(others) ≥ 0.95x` | Explicit flag hint | Try only for agent/tool-heavy traffic matching the bench. |
| **STRUCTURED** | `max ≥ 1.5x` AND `min ≥ 0.90x` | Explicit flag hint | Try only for the winning structured workload. |
| **NEUTRAL** | `min ≥ 0.95x` AND `max ≥ 1.0x` | (silent) | Leave OFF; it neither helps nor hurts. |
| **AVOID** | any workload `< 0.85x` | `⚠️ SuffixDecoding is known to regress this model — avoid --speculative-config '{"method":"suffix"}'` | Leave OFF. It will measurably slow chat/tool/etc. |
| **UNKNOWN** | not benched | (silent) | Leave OFF, or run the bench (below) and update the profile. |

The hint is informational only — there is no auto-enable. The actual
workload mix at *your* startup is unknown, so the user owns the flag.

## Why this matters

SuffixDecoding exploits repetition in the prompt and generated token
stream. It wins only when many drafted tokens are accepted. When the
prompt/output overlap is weak, the verify forwards become pure overhead
and the request slows down.

The same model can be positive on one workload and negative on another:

| Model | Tier | tool_loop speedup | Why |
|---|---|---|---|
| Gemma 4 12B | **mixed** | 1.49x on repeated tool XML | Repeated XML scaffolding creates high n-gram reuse. |
| Gemma 4 12B | **avoid** | 0.77x on JSONL repeat | The generated token path did not match drafts well enough to pay for verify. |

In other words: **model family alone is not enough**. Bench the workload
you ship, and leave the flag off unless the traffic is known.

## How to bench a new model

```bash
python3.12 scripts/bench_suffix_decoding_integrated.py \
    --model gemma-4-12b-4bit \
    --runs 3 \
    --max-tokens 256
```

This runs four workloads (chat, json_array, tool_loop, code_edit) at
N=3 runs each, in vanilla and SuffixDecoding ON modes, takes the
median, and computes the tier. Wall-clock ~10–20 min per model on M3
Ultra; longer on slower hardware.

The result file (`evals/results/suffix_<model>.json`) shows raw TPS,
speedup ratios, and the resulting tier:

```json
{
  "model": "gemma-4-12b-4bit",
  "vanilla_tps": {"chat": 58.0, "json_array": 61.0, "tool_loop": 64.0, "code_edit": 57.0},
  "suffix_tps":  {"chat": 53.0, "json_array": 47.0, "tool_loop": 95.0, "code_edit": 88.0},
  "speedup":     {"chat": 0.91, "json_array": 0.77, "tool_loop": 1.48, "code_edit": 1.54},
  "tier": "avoid"
}
```

Pass `--update-profile` to also print the patch for the corresponding
`ModelConfig` entry in `vllm_mlx/model_auto_config.py`. Paste it
manually — the script never auto-edits source.

## Currently classified models

> **Status**: framework just landed; classifications are still being
> filled in. All models default to `unknown` until benched.

The first sweep (issue #269 acceptance criteria) covers:

- mlx-community/Qwen3-0.6B-8bit
- mlx-community/Qwen3-4B-4bit
- mlx-community/Qwen3-8B-4bit
- mlx-community/Qwen3-14B-4bit
- mlx-community/Llama-3.2-1B-Instruct-4bit
- mlx-community/Llama-3.1-8B-Instruct-4bit
- mlx-community/SmolLM3-3B-4bit

Hybrid arches (Qwen3.5/3.6 A3B/A10B, Granite4, Qwopus) are gated
upstream by `supports_spec_decode=False` and stay `n/a (hybrid arch)`.

## FAQ

**Why not auto-enable positive models?**
Because the *traffic mix* at startup is unknown. Even Gemma 4 can be
positive on repeated tool XML and negative on JSONL. Default-OFF is
conservative and predictable.

**Why does model size not determine the tier?**
Tier is determined by prompt/output token reuse and acceptance rate, not
parameter count.

**When do tiers re-bench?**
On meaningful upstream version bumps (`mlx`, `mlx-lm`, the model
release itself). There's no automatic schedule; CI doesn't gate on
tier values changing. Track via the result JSON timestamps under
`evals/results/`.

**Will the boundaries (`1.8x`, `1.5x`, `0.85x`, ...) change?**
They're tunable in `vllm_mlx/model_auto_config.py::classify_suffix_decoding_tier`.
If you adjust them, also re-classify every model — the boundaries are
fixed across the registry by design (fairness + stability).
