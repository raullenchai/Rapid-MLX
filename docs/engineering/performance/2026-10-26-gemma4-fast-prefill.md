# Gemma 4 last-position prefill contract

The Gemma 4 text wrapper accepts `logits_to_keep=1` as permission to discard
positions before the final query in trailing KV-sharing layers. The default
call returns all positions. The optimization is limited to one sequence and
requires a nonempty producer section followed by borrower layers. Models with
no KV sharing retain the original forward.

Set `RAPID_MLX_GEMMA4_FAST_PREFILL=0` before starting `rapid-mlx serve` to
suppress the text optimization and the media lane's existing last-position
opt-in. This does not change the producer cache format. No cache migration is
needed when enabling or disabling it.

## Consumer audit

| Consumer | Contract |
| --- | --- |
| Text scheduler, including bounded chunks and prefix-cache resumes | Only `PromptProcessingBatch.prompt` opts in. It discards outputs and evaluates producer cache state. |
| Text sampling and generated-token logprobs | The original model runs the separately retained final prompt token and subsequent decode tokens. |
| Multi-sequence prefill, including padded batch caches | Full forward, even with an explicit last-position request. |
| Speculative verification, suffix verification, assistant drafting | Full forward. The scheduler does not install the prompt hook when speculation is configured. Verify calls retain the original model. |
| Hidden-state capture, preallocated hidden sinks, shared-KV capture, deferred normalization | Full forward, even with an explicit last-position request. |
| Prompt scoring / prompt logprobs | No implicit opt-in. The completions API rejects `echo` together with `logprobs`; generated-token logprobs are separate. |
| Embeddings / pooling | Separate model consumer; no implicit opt-in. Ordinary trunk calls keep all hidden positions. |
| Gemma 4 image/audio media lane | Already qualifies the pinned dependency's `logits_to_keep=1` forwarding contract for ordinary cached prefill. Unknown metadata, videos, and output/capture options keep the full path. The same environment switch suppresses this opt-in. |

The text hook belongs to a generator's persistent prompt batch. It temporarily
supplies the explicit signal while invoking the output-discarding prompt
method, and restores the model on success or failure. It does not patch model
classes or generation/verification batches. A split prompt batch inherits the
original model and has no hook.

## Cache and position invariants

All producer layers process the complete chunk. Borrowers receive the original
producer K/V and cache metadata, with query RoPE advanced by `chunk_length - 1`.
Per-layer inputs and array masks retain their last query row. A string causal
mask becomes `None` for the final query, which can attend every preceding key.
Sliding-window array masks retain their window restrictions. The stored
producer start offset is preserved for borrower chains.

Weight-free tests cover both attention types, per-layer embeddings, double-wide
MLPs, rotated caches across several chunks, full-position consumers, cache
identity, greedy continuation, and real batch-generator prefix resumes and
per-generated-token logprobs. Numeric comparison uses `atol=rtol=1e-5` on the
small float32 model. This tolerance is not a claim of bitwise equivalence for
quantized real weights.

## Measurement boundary

The ordinary text batch generator discards prompt logits and evaluates only
cache state. With lazy evaluation, borrower computation whose result is
unconsumed can already be eliminated. The final prompt token is evaluated
separately at sequence length one. Therefore model-call timings that evaluate
every chunk's logits cannot establish an end-to-end serving speedup.

Serving qualification must compare streaming HTTP TTFT against the exact base
commit, with prefix caching disabled, on a leased machine. Use ABAB pairs for
1K/4K/16K real-text prompts, median TTFT, and 128-token decode throughput. Record
first-token KL against the base and its own 2048-versus-512 chunking noise,
plus greedy differences and the margins at their first divergent positions.
No serving performance claim is made by the numerical tests.

Gemma 3n, multi-row query pruning, speculative acceleration, prompt-scoring API
support, and changes to other model families are outside this change.
