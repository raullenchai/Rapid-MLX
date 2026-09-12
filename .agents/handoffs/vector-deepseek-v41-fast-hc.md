# Vector handoff — DeepSeek V4.1 fast hyper-connections

Date: 2026-09-11

Owner: Vector  
Host: Studio  
Branch: `vector/deepseek-v41-fast-hc`  
Worktree: `/private/tmp/rapid-mlx-deepseek-v41-fast-hc`  
Base dependency: PR #3338 (`vector/deepseek-v41-engram-offload`)

## Intention and boundary

Raise DeepSeek V4.1 fixed-K4 decode throughput toward 40 tok/s by optimizing
only the native target's mHC mix, residual expansion, and collapse-normalize
path. Non-goals are drafter weights/training, quantization, catalog behavior,
adaptive verification, and other model families.

## Reference-first check

- vLLM: searched the current DeepSeek V4.1 implementation. It contains the
  architecture and speculative configuration but no reusable Apple Metal mHC
  kernel.
- SGLang: searched the current tree; no DeepSeek V4.1 mHC implementation was
  available to adapt.
- MLX-LM: Rapid already carries its older V4 fused Sinkhorn/collapse precedent,
  but V4.1 staggers the `pre` coefficient across sublayers, so that same-cycle
  kernel cannot be reused safely.
- oMLX: reviewed the current V4.1 single-pass mHC implementation and adapted its
  compiled mix graph, four-way Sinkhorn, post-mix, and collapse-normalize Metal
  pattern to Rapid's staggered runtime. Its file-level MIT license and source
  revision are retained in the modified source and native-runtime NOTICE.

## Verified facts

- Focused tests: 11 passed; related native load tests: 29 passed.
- Release-width BF16 kernel parity: elementwise equal at row counts 1 and 5.
- FP32 maximum focused error: `2.38e-7`.
- Real model, fixed K4, 16-token paired probe: 9.60 tok/s legacy versus 34.90
  tok/s warm fast path, identical emitted tokens.
- Four-domain 128-token suite, two repeats: 21.46 tok/s all-repeat weighted;
  26.39 tok/s warm-repeat weighted; all four outputs repeat-stable.
- Warm domain results: code 34.88, reasoning 25.04, structured 21.45, Chinese
  27.53 tok/s.
- Peak MLX memory: 171.13 GB with Engram SSD offload.

## Author-owned adversarial review

- Round 1, numerical/layout/licensing: corrected the initial repository-level
  Apache attribution to the kernel file's actual MIT boundary; retained the
  exact source revision and bundled license text. No numerical defect found.
- Round 2, compatibility/fallback: exercised the release shape on CPU and
  unsupported/empty layouts on Metal. All remained on the portable path; no
  change required beyond the added CPU regression test.
- Round 3, benchmark/scope: checked that cold and warm numbers are reported
  separately, that the prior 19.39 tok/s comparison uses the same four-domain
  K4 shape, and that no drafter, quantization, or catalog change entered the
  diff. No in-contract defect remained.

## PR validation

- PR: #3343, stacked on #3338.
- `pr_validate` exact head/base override: description, supply chain, test
  environment, vocabulary, lint, and 93.1% patch coverage passed. The external
  Codex step was intentionally skipped because this task uses the recorded
  author-owned review.
- Full unit: 23,868 passed, 132 skipped, 18 failed. All 18 failures reproduce
  unchanged on base #3338: 16 extraction tests require undeclared `torch`, one
  image precision test requires undeclared `mflux`, and one existing disk-stream
  CLI assertion receives no warning. There is no head-only failure; these are
  not fixed here to preserve the PR boundary.

## Remaining work

1. Complete author-owned adversarial review and fix only in-contract findings.
2. Run PR validation, commit, push, and open the stacked PR.
3. Rebase onto `main` after #3338 lands, then queue.
4. Separately re-evaluate the full 4-bit DSpark head under Engram offload; its
   metadata currently needs explicit 2-bit overrides for target-shared embed
   and head tensors. That is the next 40 tok/s experiment, not part of this PR.

## Risks

- First use includes Metal compilation and page warming; publish both cold and
  warm measurements rather than quoting only 34.9 tok/s.
- Low-acceptance domains remain below 30 tok/s. Kernel work alone does not prove
  a universal 40 tok/s claim.
- The PR is stacked on #3338 and must not merge first.
