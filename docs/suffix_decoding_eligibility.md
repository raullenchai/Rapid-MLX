# SuffixDecoding eligibility — per-model classification

> **Default**: `--suffix-decoding` is **OFF**. The flag is opt-in because
> the win is workload-dependent and some models regress on plain chat.
>
> This page tells you whether flipping it on for *your* model is a good
> idea.

## Tier definitions

| Tier | Trigger | Startup hint | What you should do |
|---|---|---|---|
| **AGENT** | `tool_loop ≥ 1.8x` AND `min(其余) ≥ 0.95x` | `💡 SuffixDecoding recommended for this model (tool ~Nx) — try --suffix-decoding` | Enable `--suffix-decoding` for agentic / tool-heavy traffic. |
| **STRUCTURED** | `max ≥ 1.5x` AND `min ≥ 0.90x` | `💡 SuffixDecoding may help on structured output (~Nx peak) — try --suffix-decoding` | Worth trying if you do JSON / code generation; benchmark your own traffic. |
| **NEUTRAL** | `min ≥ 0.95x` AND `max ≥ 1.0x` | (silent) | Leave OFF; it neither helps nor hurts. |
| **AVOID** | any workload `< 0.85x` | `⚠️ SuffixDecoding is known to regress this model — avoid --suffix-decoding` | Leave OFF. It will measurably slow chat/tool/etc. |
| **UNKNOWN** | not benched | (silent) | Leave OFF, or run the bench (below) and update the profile. |

The hint is informational only — there is no auto-enable. The actual
workload mix at *your* startup is unknown, so the user owns the flag.

## Why this matters

`--suffix-decoding` exploits repetition in the *output* token stream. It
wins big when output is structured (JSON), agentic (tool-call loops), or
code-edit-like (lots of token reuse from the prompt). It loses on
free-form chat where repetition is incidental — every wasted draft is a
small but real overhead.

The same model size can land in opposite tiers depending on training
mix:

| Model | Tier | tool_loop speedup | Why |
|---|---|---|---|
| Qwen3-0.6B-8bit | **AGENT** | 4.6x | RLHF tuned for tool calling — output token vocabulary tightly clusters around the tool-format tokens, so the suffix tree hits often. |
| Llama-3.2-1B-Instruct-4bit | **NEUTRAL/STRUCTURED** | 1.27x | Same size, but no tool-RLHF. Outputs more diverse, fewer suffix hits. |

In other words: **size doesn't predict tier**. Bench the model you ship.

## How to bench a new model

```bash
python3.12 scripts/bench_suffix_decoding_integrated.py \
    --model mlx-community/Qwen3-0.6B-8bit \
    --runs 3 \
    --max-tokens 256
```

This runs four workloads (chat, json_array, tool_loop, code_edit) at
N=3 runs each, in vanilla and `--suffix-decoding ON` modes, takes the
median, and computes the tier. Wall-clock ~10–20 min per model on M3
Ultra; longer on slower hardware.

The result file (`evals/results/suffix_<model>.json`) shows raw TPS,
speedup ratios, and the resulting tier:

```json
{
  "model": "mlx-community/Qwen3-0.6B-8bit",
  "vanilla_tps": {"chat": 280.0, "json_array": 290.0, "tool_loop": 305.0, "code_edit": 240.0},
  "suffix_tps":  {"chat": 295.0, "json_array": 410.0, "tool_loop": 1402.0, "code_edit": 380.0},
  "speedup":     {"chat": 1.05,  "json_array": 1.41,  "tool_loop": 4.60,    "code_edit": 1.58},
  "tier": "agent"
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

Hybrid arches (Qwen3.5/3.6, Granite4, Qwopus) are gated upstream by
`supports_spec_decode=False` and stay `n/a (hybrid arch)`.

## FAQ

**Why not auto-enable AGENT-tier models?**
Because the *traffic mix* at your startup is unknown. A model classified
AGENT may still see chat-only requests in your deployment, where the
tier-defining `tool_loop` win never applies. Default-OFF is conservative
and predictable.

**Why does model size not determine the tier?**
See the Qwen3-0.6B vs Llama-3.2-1B example above. Tier is determined by
output-token *predictability*, which is a function of training data and
RLHF — not parameter count.

**When do tiers re-bench?**
On meaningful upstream version bumps (`mlx`, `mlx-lm`, the model
release itself). There's no automatic schedule; CI doesn't gate on
tier values changing. Track via the result JSON timestamps under
`evals/results/`.

**Will the boundaries (`1.8x`, `1.5x`, `0.85x`, ...) change?**
They're tunable in `vllm_mlx/model_auto_config.py::classify_suffix_decoding_tier`.
If you adjust them, also re-classify every model — the boundaries are
fixed across the registry by design (fairness + stability).
