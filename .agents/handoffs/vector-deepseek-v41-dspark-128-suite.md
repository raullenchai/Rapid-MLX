# Vector handoff: DeepSeek V4.1 K4 128-token suite

- Owner: Vector
- Host: Studio (M3 Ultra, 256 GiB unified memory)
- Branch: `vector/deepseek-v41-dspark-adapter`
- Base: PR #3309
- Worktree: `/private/tmp/rapid-mlx-deepseek-v41-dspark-adapter`
- Status: product qualification and margin-rescue experiment failed

## Verified facts

- The policy-approved Hugging Face cache still contains the checkpoint runtime;
  reconstruction and model copying were unnecessary.
- A four-domain 128-token run reaches 11.78 weighted tok/s at K4 and peaks at
  218.07 GB.
- Per-domain K4 throughput is 16.24 code, 9.87 reasoning, 10.74 structured, and
  12.03 Chinese tok/s.
- Complete sequential-greedy equivalence is 0/4. This is target batch numerical
  behavior, not an unverified draft token bypassing target verification.
- First divergence occurs at token 79/4/15/7. Minimum batched top-two margins
  are 0.0183/0.0099/0.0046/0.0052 for code/reasoning/structured/Chinese.
- Mean accepted draft tokens/block is 0.79. The low-acceptance reasoning and
  structured prompts erase most speculative gains.
- The target artifact itself repeats on reasoning and Chinese prompts, which is
  a separate model-quality blocker.
- A 0.05-margin sequential replay hybrid was rejected and removed: 0/4 exact,
  11.21 weighted tok/s, and an incorrect early Chinese EOS.

## Next concrete action

Improve draft acceptance or make target batch numerics stable by construction.
Do not reintroduce margin replay without new evidence; the bounded 0.05 trial
failed both exactness and throughput gates.
