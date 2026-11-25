# LoRA fine-tuning on your Mac

Use `rapid-mlx finetune` (alias `rapid-mlx lora`) for local LoRA fine-tuning of
a text model on Apple Silicon. Quantized MLX checkpoints use QLoRA.
Use checkpoints from sources you trust, since mlx-lm allows custom tokenizer code.

## Which model should I fine-tune?

Choose a text-only checkpoint that supports serving LoRA adapters, such as
`qwen3-0.6b-4bit`. Qwen3.5-family multimodal checkpoints do not support adapters
yet. Fine-tuning checks serve compatibility before loading weights. See the
[serve CLI reference](../reference/cli.md#rapid-mlx-serve).

## Prepare data and run

Supply a JSONL file or a directory with `train.jsonl` and `valid.jsonl`.
Use one format within each file, with one JSON object per line:

- Chat: `{"messages":[{"role":"user","content":"Say hello"},{"role":"assistant","content":"Hello!"}]}`
- Prompt and completion: `{"prompt":"Say hello","completion":"Hello!"}`
- Plain text: `{"text":"Hello from a local model."}`

A single file is split reproducibly, holding out about 10% for validation.
Rapid-MLX holds out at least one batch for validation. Each split needs a full
batch (four rows by default). Lower `--batch-size` for small datasets.

If you omit `--iters`, Rapid-MLX prints its choice of three iterations per
full training batch, for about three epochs with a minimum of 3 iterations and a
cap of 1,000. The learning-rate default is `1e-5`. If the final reported
validation loss equals or exceeds the initial loss, Rapid-MLX suggests trying more
training data or a different `--learning-rate`.

This example checks the workflow with at least two rows in `examples.jsonl`.
For fine-tuning on your task, use representative data.

```bash
rapid-mlx finetune qwen3-0.6b-4bit --data ./examples.jsonl --out ./my-adapter --iters 10 --batch-size 1 --num-layers 4 --max-seq-length 512
rapid-mlx serve qwen3-0.6b-4bit --adapter ./my-adapter --port 8613
# In a second terminal, with Codex installed:
rapid-mlx agents codex --setup --base-url http://127.0.0.1:8613/v1
codex
```

Agent setup writes the local server and adapter model ID to the Codex config.
Use a fresh output directory for each run. The default is a timestamped path under
`~/.rapid-mlx/adapters/`. The memory estimate is advisory. The tuner prints
measured `Peak mem` during training. On success, Rapid-MLX
prints the serve command. To start a new run from saved LoRA weights, pass
`--resume path/to/adapters.safetensors` with the same base model and layer settings.

When telemetry is enabled, the completion event contains the outcome, elapsed
time and row-count buckets, and whether validation data was prepared.

## Evaluate your adapter

Keep a test dataset separate from training and validation. For writing style,
put the outline in the user turn and the author's original passage in the
assistant turn. Eval accepts one user/assistant pair per JSONL row, or a
`prompt` and `completion` pair.

```bash
rapid-mlx eval qwen3-0.6b-4bit --adapter ./my-adapter --data ./test.jsonl \
  --few-shot ./train.jsonl --task style --samples 20 --seed 0 --out ./eval-report
```

LoRA learns how you write, not what you know; facts and claims must come from
the outline. With little text, try few-shot prompting first; eval tells you
whether fine-tuning was worth it.

Eval uses the same sampled outlines for the base model, the LoRA adapter, and
the optional three-shot baseline. It chooses three distinct examples from
`--few-shot` training rows using the seed and excludes exact matches to held-out
prompts. The JSONL has no row-level source document IDs, so eval cannot verify
document separation. Use a document-level train/eval split when preparing your
data. The model can still borrow topics from training examples, even across
documents. Outputs use greedy decoding with a 512-token limit.

- **Loss and perplexity.** Eval gives each arm the reference passage token by
  token and averages negative log probabilities over reference tokens. Perplexity
  is the exponential of that average. Lower values indicate that the model
  assigns more probability to the held-out writing. Prompt tokens and turn
  terminators are excluded, except when tokenization merges the prompt boundary
  into the first scored token.
- **Style distance** (lower is closer) is a surface measure. It compares
  sentence length and variance, type-token ratio, mean word length, first- and
  second-person pronoun use, contractions, sentence-initial conjunctions,
  list and markup lines, punctuation (including questions and dashes), and
  English function words. Each feature uses its population standard deviation
  across the sampled references, with a minimum scale for near-constant features.
  Eval bounds each scaled difference to the same range, averages across
  features and samples, and reports three decimals. English-specific features
  and word counts are coarse for other languages, especially those without
  word spacing.
- **Indicative style preference.** The base model sees the reference and two
  anonymous outputs in both orders, with the same excerpt of each text in both.
  The rendered prompt and answer token fit within 2,048 tokens or the smaller
  context limit declared by the model or tokenizer.
  Eval scores the next-token log probabilities of A and B and averages the
  adapter's log-odds across orderings to offset position bias. An absolute
  average at or below 0.1 counts as a tie. Empty candidates, tokenizer options
  that cannot be scored, prompts that cannot fit, and non-finite scores count
  as invalid comparisons. The JSON reports decided comparisons, ties by margin,
  and invalid comparisons for each baseline. Small judges may prefer their own
  writing. The preference measures perceived style and cannot verify facts.

Use `--task generic` for loss, perplexity, and samples alone. With a style judge
and three-shot baseline, the headline is the adapter's win rate across all
sampled prompts, including ties and invalid comparisons in the denominator.
The serve recommendation requires a majority of wins in that case and lower
perplexity than both the base model and the three-shot baseline. When the
headline is perplexity change, the recommendation requires lower perplexity
than the available baselines.

Eval starts stdout and `report.md` with a one-line perplexity scorecard and
adapter decision, and prints elapsed time and up to three truncated samples.
It writes `report.md` and `report.json` to `--out`, which defaults to
`<adapter>-eval` beside the adapter.
Both files stay local. The JSON contains aggregate results and the adapter hash.
The Markdown includes excerpts from your dataset and model outputs, so review it
before sharing.
