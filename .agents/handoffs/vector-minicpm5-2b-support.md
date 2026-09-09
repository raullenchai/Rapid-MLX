# Vector handoff — MiniCPM5 2B support

## PR-start FYI

- Owner/host: Vector, Studio
- Branch/worktree: `vector/minicpm5-2b-support`,
  `/private/tmp/rapid-mlx-minicpm5-2b-support`
- Intention: add the official MiniCPM5 2B MLX 4-bit checkpoint to the shared
  Server/Desktop catalog with the already-supported native XML tool protocol.
- Scope: alias metadata, offline size manifest, exact Desktop tool-capability
  verdict, focused tests, user documentation, and reproducible qualification.
- Non-goals: default-model changes, new model kernels, speculative/DSpark
  support, broad MiniCPM family promotion, or sampling-policy changes.
- Verification: alias/catalog/size tests, focused Swift capability tests,
  real server load and 31-case tool suite, self-adversarial diff review, PR CI.

The Orca agent messaging channel was unavailable in this session. This tracked
handoff carries the equivalent non-blocking start FYI for Atlas, Pixel, Harbor,
Echo, and ds0731.

## Reference-first check

The serving precedents use the checkpoint's standard `LlamaForCausalLM`
implementation without a model-code fork. The MLX-native release has the same
llama model type, 4-bit affine quantization, ChatML-style template, MiniCPM XML
tool wire format, and Qwen-style reasoning envelope as mechanisms already
present in Rapid. Reuse was selected; no new runtime or parser is justified.

The separately published draft checkpoint targets a speculative algorithm not
currently qualified in Rapid. It remains explicitly outside this PR.

## PR-complete FYI

- PR: https://github.com/raullenchai/Rapid-MLX/pull/3285
- Outcome: Server and Desktop now expose the official 2B MLX 4-bit checkpoint
  through `minicpm5-2b-4bit`, with exact size and parser metadata.
- Evidence: 3,590 focused Python contract tests and 42 Desktop capability tests
  passed; the real alias booted, became ready, and reproduced 24/31 on the
  tool-calling suite.
- Quantitative result: 7.17 s summed request time versus 32.54 s for the 4B
  comparison on the recorded M3 Ultra setup (78% less, about 4.5x faster), with
  a 24/31 versus 26/31 correctness trade-off.
- Risk/rollout: not a Smart default; speculative decoding stays disabled. One
  failed selection case triggered the loop breaker and recovered, so repetition
  stability remains something to monitor rather than a release blocker.
- Follow-up owner: none required for this scoped support PR. A future draft-model
  integration needs separate architecture and qualification work.

As with the start FYI, this completion notice is recorded here because the Orca
agent messaging channel was unavailable.
