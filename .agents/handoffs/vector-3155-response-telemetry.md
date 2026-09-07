# Vector handoff: #3155 request-scoped MTP response telemetry

## PR contract

- Owner: Vector
- Host: Mac Studio
- Branch: `vector/3155-response-telemetry`
- Worktree: `/private/tmp/vector-3155-response-telemetry`
- Goal: expose request-scoped MTP accepted/drafted depth counts, verify calls,
  correction tokens, and bonus tokens in the final response telemetry so a
  user can diagnose why speculative decoding did or did not improve latency.
- Scope: request-local accounting for both the singleton and continuous MTP
  paths; final-output propagation; OpenAI-compatible chat, completion, and
  Responses serialization; streaming/non-streaming parity; regression tests.
- Non-goals: no MTP algorithm, depth policy, cache behavior, model/catalog,
  downloader, or unrelated response-API change.
- Verification: pure-Python counter and concurrency tests, scheduler/output
  propagation tests, route serialization tests, local API dogfood, scoped
  adversarial review, PR validation, and CI.

## Reference-first notes (internal only)

- Rapid-MLX: reviewed the process-global `MTPAcceptCounter`, continuous MTP
  proposal accounting, request/output lifecycle, and the JSON serializers that
  already use `exclude_none=True`.
- vLLM: reviewed its current request-owned speculative accumulator and its
  final-output-only `metrics.speculative_decoding` response contract. Adopt the
  request ownership and final-only transport pattern; do not derive a request
  delta from process-global counters because concurrent requests would mix.
- SGLang: reviewed its per-request speculative counts and `sgl_ext` response
  extension. Adopt the explicit extension-envelope principle, while using the
  more neutral `metrics.speculative_decoding` envelope that also accommodates
  Rapid's future timing metrics.
- MLX-LM: no corresponding public per-request response contract was found; its
  generator response remains the lower-level token delivery surface, so the
  accounting must remain in Rapid's scheduler/request layer.

## Storage policy

No model download is required for this PR. Before any later model download or
checkpoint conversion, read `~/STORAGE-POLICY.md`, inspect the existing global
Hugging Face cache, and never override or redirect its configured cache paths.

## Verification evidence

- Scoped route/engine/counter suite: 318 passed, 2 skipped, 3 deselected.
- Real server dogfood used only the already-cached
  `mlx-community/Qwen3.5-4B-MLX-4bit` base and
  `mlx-community/Qwen3.5-4B-MTP-4bit` sidecar; no model bytes were downloaded.
- Singleton non-stream response: 1 verify call, 1/1 depth-1 accepted; streaming
  response: exactly 1 of 12 JSON events carried metrics, on the terminal chunk.
- Four simultaneous continuous-MTP requests (prefix cache disabled to exercise
  that lane) returned four distinct request-owned histograms: 40, 16, 39, and
  39 verify calls. The service log confirmed a four-lane continuous cohort.
- A deliberately prefix-cache-routed concurrent run returned no metrics because
  it performed no MTP verification; this matches the documented omission rule.
- Microbenchmark, CPython 3.11 on this Studio, 7 repeats x 200,000
  `record_round(3, 2)` calls: existing single counter median 568.7 ns/call;
  process + request fan-out median 1089.7 ns/call; incremental bookkeeping
  521.0 ns per verifier round. This is accounting overhead only, not a claimed
  end-to-end inference speedup.

## Self-adversarial review

- Round 1 found cached response replay could leak the original request's MTP
  metrics into a cache hit. Fixed by clearing metrics on cache replay and added
  a route regression test.
- Round 2 checked singleton/continuous ownership, dynamic joins, terminal output
  aggregation, stream finalization, and multi-prompt Completions. Multi-prompt
  metrics now sum per-depth counts across choices instead of dropping them.
- Round 3 dogfood checked ordinary, streaming, concurrent continuous-MTP, and
  no-verification fallback behavior. No further in-scope correctness finding.
- Round 4 full-smoke probing found open-ended test doubles can synthesize a
  `MagicMock.spec_decode_metrics` attribute. The serializer now accepts only a
  dict or the typed schema and otherwise fails closed; the three affected
  completions logging tests pass. Full smoke then reached an unrelated optional
  image test whose local environment lacks `mflux` (isolated: 22 passed, 1
  environment failure); no image-scope change was made.
- Round 5 hosted CI found one new mypy diagnostic at the request-id lookup in
  continuous-cohort attachment. The optional id is now narrowed before
  `dict.get`; the pinned CI type environment reports the existing 701-error
  grandfathered budget with no growth, and 162 related tests pass.
- Round 6 changed-lines coverage reached 98% and identified exactly two missing
  branches: empty counter-group construction and JSON-buffered Completions
  terminal metrics. Added direct regression tests; the focused suite is 26/26.
