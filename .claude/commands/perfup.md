---
name: perfup
description: "Autonomous performance optimization: research, PoC, benchmark, implement, review, PR"
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Task
  - WebSearch
  - WebFetch
  - Skill
  - TaskCreate
  - TaskUpdate
  - TaskList
  - TaskGet
  - EnterPlanMode
  - ExitPlanMode
  - AskUserQuestion
---

# /perfup — Autonomous Performance Optimization

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): you are an autonomous performance researcher for vllm-mlx. You propose optimizations, benchmark them, keep what works, discard what doesn't, and ship a production PR.

## Key Files

- **Results log**: `reports/perfup-results.tsv` — append-only experiment log (commit, metric, status, description)
- **Optimization queue**: `memory/knowledge/perf_optimization_queue.md` — ranked list of candidates
- **Memory index**: `memory/MEMORY.md` — what's been done, what's known
- **Benchmark script**: `scripts/benchmark_engines.py`
- **Model for benchmarking**: Check memory for current model path. If unavailable, ask user.

## The 6 Phases

### Phase 1: Research

Read existing state, then discover new opportunities.

1. Read `memory/knowledge/perf_optimization_queue.md` and `memory/MEMORY.md`
2. If `$ARGUMENTS` is provided (e.g. `/perfup decode`), focus on that area. Otherwise broad search.
3. Scan codebase for optimization opportunities:
   - Use Task(subagent_type=Explore) on critical paths
   - Search for TODO/FIXME/PERF/HACK comments
   - Check ml-explore/mlx-lm recent releases (`gh release list --repo ml-explore/mlx-lm --limit 5`)
4. WebSearch for latest MLX inference optimizations if needed
5. Produce candidate list, each with: problem, solution, estimated impact, effort, coverage, risk

### Phase 2: Prioritize

Score and rank. Persist to memory.

1. Score each candidate (1-5 per axis):
   - **Impact**: Performance gain magnitude (5 = >2x)
   - **Ease**: Implementation effort (5 = <1 day)
   - **Coverage**: Models that benefit (5 = all)
   - **Safety**: Regression risk (5 = zero)
2. Sort by composite = Impact x Ease x Coverage x Safety
3. Update `memory/knowledge/perf_optimization_queue.md`:
   - Completed items → "Completed" section (date + results)
   - Failed/rejected → "Rejected" section (reason)
   - Active queue → "Queue" section with [P0]-[P3] tags
4. Present top 3 to user. Wait for confirmation before proceeding.

### Phase 3: PoC Experiment Loop

**This is the core loop. Inspired by autoresearch: try, measure, keep or discard. Repeat.**

```
SETUP:
  git checkout -b perfup/<optimization-name>
  Record baseline metrics (run benchmark on current code)
  Initialize reports/perfup-results.tsv if not exists

LOOP:
  1. Implement minimal PoC change in code
  2. git commit -m "perfup: <brief description>"
  3. Run benchmark: python3.12 scripts/benchmark_engines.py (or custom)
     Redirect output: > reports/perfup-run.log 2>&1
  4. Extract metrics from log (TTFT, decode tok/s, etc.)
  5. Record to reports/perfup-results.tsv:
     commit<TAB>decode_tps<TAB>ttft_ms<TAB>status<TAB>description
  6. DECISION:
     - If metric improved: KEEP. Log "keep" status. This is the new baseline.
     - If metric same or worse: DISCARD. Log "discard". git reset --hard to previous keep.
     - If crashed: Log "crash". Try to fix (1-2 attempts). If unfixable, discard and move on.
  7. If improvement confirmed and significant (>5%): break loop → Phase 4
  8. If no candidate works after trying top 3: inform user and stop.
```

**Rules for the loop:**
- Each PoC should be MINIMAL — smallest change that tests the hypothesis
- Benchmark must run on a REAL model (not mocks)
- If benchmark takes too long or model not loaded, ask user
- Do NOT ask "should I continue?" between iterations — just keep going
- DO stop and ask if you need user action (download model, start server, etc.)

### Phase 4: Full Implementation

PoC validated. Now build it properly.

1. Clean up or rewrite the PoC code for production quality
2. Enter plan mode — design clean architecture, tests, docs
3. Implement:
   - Clean code, proper error handling, logging
   - Unit tests matching existing patterns in `tests/`
   - No hacks, no dead code
4. Run full test suite: `python3.12 -m pytest tests/ -v`
5. Run benchmark again — confirm improvement matches PoC

### Phase 5: Review Loop

Independent review via Codex.

1. Invoke: `/review-loop <description of optimization>`
2. Address all findings (P0 = blocker, P1 = should fix, P2 = nice to have)
3. After review passes, run final benchmarks on all relevant models
4. Update README/docs with new benchmark numbers if applicable

### Phase 6: PR & Ship

1. Ensure all changes are on `perfup/<name>` or `feat/<name>` branch
2. Push to `raullenchai` remote (NEVER origin, NEVER main directly)
3. Create PR:
   ```
   gh pr create --repo raullenchai/vllm-mlx --base main
   ```
   PR body must include:
   - **Summary**: What was optimized and why
   - **Benchmark results**: Before/after table from perfup-results.tsv
   - **Test plan**: How to verify
4. Update memory:
   - Move optimization to "Completed" in `perf_optimization_queue.md` with PR#, date, confirmed speedup
   - Remove from todo if applicable
5. Present PR URL to user

---

## Results TSV Format

```
commit	decode_tps	ttft_ms	status	description
a1b2c3d	68.4	245	baseline	current main branch
b2c3d4e	72.1	240	keep	reduce redundant mx.eval in decode loop
c3d4e5f	67.9	248	discard	speculative prefill chunking
d4e5f6g	0.0	0	crash	fused MoE kernel (import error)
```

## Focus Areas

If `$ARGUMENTS` provided:
- `ttft` — Time to first token (prefill optimization)
- `decode` — Decode throughput (tok/s)
- `tools` — Tool calling accuracy/reliability
- `accuracy` — Model output quality
- `memory` — Memory usage / longer contexts
- `prefill` — Prefill speed
- `cache` — Cache hit rate / prompt reuse
- No argument → broad research across all areas

## Important Rules

1. **Benchmark proves everything.** No optimization ships without measured improvement.
2. **Memory is truth.** `perf_optimization_queue.md` is the canonical record of what's tried/works/failed.
3. **Git discipline.** Feature branch → PR on raullenchai/vllm-mlx. Never push to main.
4. **Keep it simple.** A small improvement with clean code beats a large improvement with ugly code. Removing code for equal performance is a win.
5. **Ask only when blocked.** Don't ask "should I continue?" — just keep iterating. Ask only for user actions (model download, server restart, etc.).
