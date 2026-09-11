# Vector handoff: DeepSeek V4.1 K4 128-token suite

- Owner: Vector
- Host: Studio (M3 Ultra, 256 GiB unified memory)
- Branch: `vector/deepseek-v41-dspark-adapter`
- Base: PR #3309
- Worktree: `/private/tmp/rapid-mlx-deepseek-v41-dspark-adapter`
- Status: product qualification failed; ambiguity-rescue experiment next

## Verified facts

- The policy-approved Hugging Face cache still contains the checkpoint runtime;
  reconstruction and model copying were unnecessary.
- A four-domain 128-token run reaches 11.78 weighted tok/s at K4 and peaks at
  218.07 GB.
- Per-domain K4 throughput is 16.24 code, 9.87 reasoning, 10.74 structured, and
  12.03 Chinese tok/s.
- Complete sequential-greedy equivalence is 0/4. This is target batch numerical
  behavior, not an unverified draft token bypassing target verification.
- Mean accepted draft tokens/block is 0.79. The low-acceptance reasoning and
  structured prompts erase most speculative gains.
- The target artifact itself repeats on reasoning and Chinese prompts, which is
  a separate model-quality blocker.

## Next concrete action

Measure top-two target logit margins at first divergence and prototype bounded
singleton replay only for ambiguous blocks. Retain it only if four-domain
exactness reaches 4/4 and weighted throughput remains at least 12 tok/s.
