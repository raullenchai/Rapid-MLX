# DeepSeek V4 Codex stability plan

Status: **substantially delivered** · Statused: 2026-08-18

> Since this plan was written, dsh became a Tier-1 agent in the
> integration smoke loop (`tests/integrations/agent_smoke.sh`), and
> DSpark shipped as a speculative-draft method. The document is kept as
> the original stability plan.

This document tracks the work required to make DeepSeek-V4-Flash-0731 a
reliable Codex engineering backend before speculative decoding is enabled for
coding workloads.

## Current baseline

- A real implementation of GitHub issue #707 completed without a repetition
  reconnect. The run used roughly 12 tool rounds, grew to a 17K-token prompt,
  recovered from two genuine test failures, and produced a reviewable patch.
- Independent validation passed 61 related tests (with one existing XPASS),
  Ruff, and `git diff --check`.
- The Rapid-MLX Codex/repetition/tool-call regression slice passes 194 tests.
- DeepSeek Codex requests use a low-entropy stochastic default when the client
  does not explicitly choose a temperature. This is currently the correctness
  baseline.
- Greedy DSpark is not part of the baseline: it improves decode performance,
  but has reproduced deterministic malformed tool arguments.

## Workstream boundaries

### Stability baseline

- Responses API event ordering and terminal failure envelopes
- DeepSeek DSML schema canonicalization and tool-call validation
- reasoning/think-tag containment
- repetition detection and recovery
- Codex action/progress guidance
- disconnect and abort semantics

Primary files include `routes/responses.py`, `service/helpers.py`,
`repetition_guard.py`, `output_collector.py`, the DeepSeek tool parser, and
their focused tests.

### Long-context correctness and memory

- prefix-boundary parity
- pooling-cache trimming and rollback invariants
- adaptive prefill and process-memory guards
- progressive context soak tooling
- safe recovery from Metal allocation failures

Primary files include `deepseek_v4_cache.py`, `engine/batched.py`,
`engine_core.py`, the non-speculative portions of `scheduler.py`, and the
progressive-context tests and script.

### Performance experiments (gated off for Codex)

- stochastic DSpark verification and rollback
- adaptive speculative depth
- multi-row QMV verification
- DeepSeek verify-attention kernels
- switch/attention kernel experiments

Primary files include `deepseek_v4_verify*.py`,
`deepseek_v4_rollback.py`, speculative sections of `deepseek_v4.py` and
`scheduler.py`, and `test_dspark_scheduler.py`.

Performance work must remain eligible for an immediate baseline fallback. It
must not become the implicit Codex path until every stability gate below is
green.

## Stability gates

### Gate S0: deterministic regression suite

- all focused Responses, parser, tool validation, repetition, and DeepSeek
  scheduler tests pass
- no lint errors or whitespace errors
- explicit sampling parameters continue to override server defaults

### Gate S1: medium real engineering task

- 60–120 minutes or at least 30 tool rounds
- natural context growth to at least 50K tokens
- changes span at least five production/test files
- at least one real failing test is diagnosed and fixed
- zero repetition reconnects
- zero malformed tool calls
- no unchanged failing command is repeatedly executed
- final patch passes independent review and focused regression tests

### Gate S2: long progressive-context task

- natural context growth to 150K–250K tokens
- prefix-cache reuse across normal Codex turns
- no Metal resource-limit failure or silent request truncation
- zero repetition reconnects and malformed tool calls
- successful compact/resume or equivalent long-session boundary test

### Gate P1: performance reintroduction

Starting from the S2 baseline, enable one optimization at a time:

1. stochastic DSpark correctness
2. prompt priming and exact rollback
3. adaptive speculative depth
4. multi-row QMV
5. long-context cache/kernel optimizations

Each step must replay the same engineering workload and preserve S0–S2. Raw
decode throughput, end-to-end tool-round latency, TTFT, acceptance depth,
fallback count, and memory peak must all be recorded. A faster result that
reduces task success or tool-call correctness is rejected.

## Immediate execution order

1. Split and validate the stability baseline independently from speculative
   performance code.
2. Select a medium, cross-file Rapid-MLX issue and run Gate S1 against the
   local-wheel server.
3. Review and fix every failure exposed by S1, then repeat until clean.
4. Run Gate S2 with progressively growing context.
5. Freeze the stable configuration and begin P1 experiments.
