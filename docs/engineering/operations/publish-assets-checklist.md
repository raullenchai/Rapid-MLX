# Publish-assets checklist — Trio-Flash research release (astra session 5)

Status legend: [x] done · [ ] open (owner: Atlas unless noted)

## Core assets
- [x] Research post (`docs/blog/index.html`) — machinefi tone, mystery
      boundaries respected (no base family, no size, no quantization).
- [x] Demo frames from real gameplay (`docs/blog/assets/spire_{1,2,3}.png`).
- [x] Golden functional gate (`scripts/run_golden.py`, 7/7) + parity gate
      tightened to production thresholds (−1.0 pt / agreement ≥98% /
      no guard regression / no family −2pt / ECE +≤0.02).
- [ ] **Method appendix** (versioned page, beta participants): task
      definitions, family counts, seeds, splits, leakage controls, prompt/
      candidate mapping, candidate-letters-are-single-tokens confirmation,
      comparator API revision + access date, McNemar contingency table +
      exact-test definition, ECE binning + reliability plot + bootstrap CI,
      risk–coverage + threshold-selection split, warm/cold latency protocol
      + hardware + retry policy, model card (intended/unsupported uses,
      abstention semantics), artifact hashes + changelog.
- [ ] **Sanitized item-level outputs** (CSV/JSONL): item_id, split, family,
      seed, candidate_count, gold_index, candidate_order_hash,
      trio_prediction, trio_probabilities, trio_confidence, trio_correct,
      trio_disposition, latency_ms, comparator_prediction, comparator_
      probabilities, comparator_correct, model_versions. No raw prompts;
      publish a hash of the frozen eval manifest.
- [ ] Standalone metrics script for beta participants (repo stays private;
      without data+analysis code the post is a "technical research note").
- [ ] Demo video hosting decision (frames embedded; video on request).
- [ ] Terms page naming the operating legal entity (Harbor), beta-acceptance
      before token issuance (astra clause list in session-4/5 notes).

## Server / deploy (this repo)
- [x] Pluggable serving backend: `MARVIN_BACKEND=mlx|llama_cpp`
      (`MARVIN_LLM_BASE`), same /v1/classify contract; 502 on backend error.
- [x] Vast one-command init (`serve/vast/setup_vast.sh`, llama.cpp locked).
- [ ] **Q4_K_M primary path**: convert BF16 originals (30.9GB FP8 / 55.6GB
      BF16 do NOT fit a 24GB card) via llama.cpp convert_hf_to_gguf.py →
      llama-quantize Q4_K_M (~17GB). Ternary→GGUF custom converter is
      fallback-only. Expected loss 0–1pp (astra); decision margins can be
      non-monotonic — hence the tightened parity gate.
- [ ] Recovery drill: destroy + re-provision the instance from zero using
      only off-host artifacts (scripts + GGUF + tunnel credential) — must
      succeed before publishing API availability.
- [ ] Watchdogs: llama-server AND api process (healthy frontend must not
      mask a dead inference process); named-tunnel credential stored
      off-host; verify $200/mo is fixed price + host uptime history.

## Launch sequencing (astra-confirmed)
- Week 1: blog + waitlist only (no public API) while conversion/parity/soak
  runs on Vast. · Week 2: gates pass → named tunnel + tokens to 5 users.
  · Week 3: expand after labels/abstentions/5xx/p95 review.
