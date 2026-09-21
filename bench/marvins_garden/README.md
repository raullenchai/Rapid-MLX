# Marvin's Garden — fast decisions from a smart base

Marvin's Garden turns **Bonsai 27B 2-bit** (`prism-ml/Ternary-Bonsai-27B-mlx-2bit`)
into a high-frequency decision model: routing, tool gating, and injection
guarding, answered with **one forward pass and zero generated tokens**.

The bet: the base is already smarter than the mid-size models others
post-train for this job (Ternary-Bonsai-27B > Qwen3.5-9B class). We are not
chasing intelligence; we are removing latency. The recipe is the
contrastive-data-curation loop popularized by Bespoke Nimble, rebuilt
against Rapid-MLX's own product decisions — no distillation, fully
synthetic data, labels derived from audited rules.

## Pipeline

```
generate_contrastive.py   deterministic synthetic pairs (3 families, flip_key audited)
        │
        ▼
validate.py               fail-closed schema + pair-integrity gate (jsonschema mandatory)
        │
        ▼
to_chat_sft.py            pairs -> mlx-lm chat SFT (assistant = one letter, no CoT)
        │
        ▼
train_lora.sh             mlx_lm.lora on the quantized 27B base
        │
        ▼
eval_label_readout.py     single-forward letter readout: accuracy, flip robustness,
                          calibration, abstention, ms/decision
```

## Quickstart

```bash
# 1. data (deterministic; same seed -> byte-identical files)
python bench/marvins_garden/generate_contrastive.py --seed 20260919

# 2. validate (exit code = failed files; jsonschema>=4.0 required, fail-closed)
python bench/marvins_garden/validate.py

# 3. SFT conversion
python bench/marvins_garden/to_chat_sft.py

# 4. train (see env knobs in the script header)
bash bench/marvins_garden/train_lora.sh

# 5. evaluate — "marvin mode" is the 3-style paraphrase ensemble
python bench/marvins_garden/eval_label_readout.py \
  --model prism-ml/Ternary-Bonsai-27B-mlx-2bit \
  --adapter bench/marvins_garden/adapters/marvins-garden \
  --styles base,concise,spec \
  --output bench/marvins_garden/results/eval_marvin_27b.json

# 6. IQ-tax probe (general capability, base vs adapter)
python bench/marvins_garden/iq_probe.py \
  --model prism-ml/Ternary-Bonsai-27B-mlx-2bit \
  [--adapter bench/marvins_garden/adapters/marvins-garden]
```

## Data contract

One JSON object per line (`schema.json`, draft 2020-12,
`additionalProperties: false`). Key fields:

- `contrast_group` / `pair_id`: two members (`-a`, `-b`) differ in exactly
  one audited fact (`flip_key`); their correct labels differ by construction.
- `label` must be one of `candidates`; `meta.scenario` stores the
  pre-render fields so the label is **derivable** from the audit copy
  (unit-tested, not trusted).
- Families: `model_routing` (8-option alias menu mirroring the Rapid-MLX
  alias table), `tool_gate` (call_tool vs answer_directly with decoy tools),
  `injection_guard` (allow vs block with separable attack lines).

Regenerating with the committed seed must reproduce the committed files —
`tests/test_marvins_garden.py` enforces determinism and contrast integrity.

## Metrics that matter

| Metric | Meaning |
| --- | --- |
| `accuracy` | heldout, single-forward readout |
| `flip_both_members_correct` | fraction of contrast groups with BOTH members right — the discrimination metric contrastive curation targets |
| `ece_15bin` | calibration of the readout confidence |
| `abstain` | coverage/accuracy at a confidence threshold (fallback routing) |
| `ms_per_decision` | wall time per decision, `generated_tokens: 0` by construction |

## Serving contract (planned)

The eval script is the executable spec for a future `/v1/classify` endpoint:
chat-template prefill, one forward pass, letter-token softmax. **Each
adapter is paired with a serving template mode** (see perf doc, Finding 1):
serve `marvins-garden` with `--think-mode disabled`, serve
`marvins-garden-v15` with `--think-mode enabled` (the mlx-lm training
view). The mode must travel with the adapter. Until the endpoint exists,
`max_tokens=1` + `logprobs` approximates it with a known top-k failure
mode.

## Training gotcha (durable lesson)

`mlx_lm.lora` computes loss over the FULL sequence by default. With ~300
prompt tokens and a 1-letter completion, the decision signal is ~1/300 of
the gradient and the adapter learns to model prompts while the readout
stays at chance (observed: val loss 0.048, heldout accuracy 41% ≈ base
40.1%). `--mask-prompt` is MANDATORY for this pipeline: val loss then
starts at ~ln(n_candidates) and the letter itself is optimized.

## Non-goals / honesty

- No standard public benchmark exists for this task class; heldout numbers
  here measure OUR distribution (same caveat Nimble published).
- No RL, no distillation from any teacher in v1. Roadmap: rubric-SFT for
  two-step "look-then-commit" decisions, then lightweight preference tuning
  on calibration loss.
