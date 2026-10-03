# Vector/Harbor handoff: GLM-5.3 TensorFold product capture

- Owners: Vector for the server/evidence contract and Harbor for reproducible
  operations and artifact provenance. Atlas retains release priority.
- Branch/worktree: `harbor/glm53-product-capture` in a fresh `/private/tmp`
  clone.
- Base: exact main `f4ad2c7b974497f8763b3ff8a386155475d0925c`.
- Integration: current main `a94821ea02fe1331048284bcdfb1ab6fc426b865`
  was merged normally before final verification; history was not rewritten.
- Scope: a local-only capture harness, synthetic artifact-contract tests, and
  the qualification reproduction note. No engine optimization, release
  dispatch, benchmark claim, cache mutation, or host remediation is included.

## Verified implementation contract

`scripts/capture_glm53_tensorfold_product.py` must itself run under
`scripts/large-model-run.py`, so the product server and subsequent direct
drafted/serial comparison share the same command-lifetime host lock. Main first
resolves the target with `snapshot_download(..., local_files_only=True)` from
the default cache; `run_capture` then rejects nonzero used swap before either
server starts. It chooses collision-free loopback ports other than 8080/8891
and sets all supported Hub consumers offline.
The reproduction command uses an explicit 190 GiB working set: 168.5 GiB
observed resident weights plus the 16 GiB pass cache and bounded process
overhead. The registry's generic 1.5x multiplier would exceed this host's
physical capacity and cannot admit the already-qualified configuration.

The product phase runs the shipped
`rapid-mlx serve glm5.3-flash-tensorfold` command, proves health/model/profile
and pinned target identity, and reuses the existing qualification-only
`RAPID_MLX_TENSORFOLD_AUDIT_PATH` channel for complete token IDs. The frozen
fixture is retained byte-for-byte. Its direct-runtime `thinking_budget=256` is
translated only at the Rapid wire boundary to the equivalent public
`reasoning_max_tokens=256`; the submitted body and translation are explicit in
the artifacts.

The direct phase uses the already-qualified settings (context 8192, max tokens
4096, one lane, three MTP drafts, prefill pass 8, 16 GiB pass cache, update check
disabled) and sends drafted then `draft:false` requests. Content, reasoning,
finish, and any mutually available token evidence are compared. TensorFold's
12-hex `token_sha` stays labeled as an opaque fingerprint and is never called
SHA-256. Complete token IDs are either retained with a verified comma-decimal
SHA-256 or explicitly marked unavailable.

Raw process output is staged outside the artifact directory and only sanitized
logs are retained. Artifacts scrub usernames, private absolute paths,
credentials/private URLs, and PID contexts. Pre/post `memory_pressure`,
`vm_stat`, swap, load, and listener facts are retained; owned servers must exit
and their listeners must disappear. The final sorted SHA-256 map covers every
retained file except itself and is verified before success.

Production launch is bound to inspected `rapid-mlx` and `tensorfold` console
scripts whose shebangs resolve to the invoking interpreter and whose imports
match that interpreter's distribution metadata. The source checkout must be
clean. Product readiness is accepted only when models, health, status, active
TensorFold MTP, accelerated compatibility, installed runtime, and target
revision/readiness all match the qualified profile. Only validated projections
of these server-derived values enter the manifest; sanitized raw bytes remain
separate. SIGTERM, SIGHUP, and SIGINT unwind through cleanup, and forced,
nonzero, premature, or listener-leaking shutdowns invalidate the result.
Child environments drop Python module-path overrides and all inherited Rapid
or TensorFold runtime overrides, then explicitly force offline mode, version
checks off, and telemetry off. Only the product process receives its private
temporary token-audit destination.
Product completion requires the expected response object/model and full-token
audit evidence; direct completion requires its exact served model. The direct
argv pins `--snapshot-dir none` to prevent persistent snapshot/cache writes.

## Reference check

Private implementation research reviewed the existing Rapid TensorFold audit
boundary and large-model lock first. vLLM and SGLang serving benchmarks retain
per-request TTFT/detail only when streaming actually exposes a token and keep
request configuration alongside results. Open WebUI and Jan expose explicit
health/model or service-status checks. Cherry Studio treats confirmed process
exit, rather than a sent shutdown signal, as lifecycle truth and separates
local diagnostics from upload-safe projections. LM Studio's local-server flow
checks a selected loopback endpoint before use. The harness adapts those
patterns through Rapid's existing HTTP and process boundaries without copying
code or assets.

## Verification and remaining work

- Synthetic product and direct HTTP servers cover the complete lifecycle and
  artifact contract without loading a model. Focused adversarial cases include
  wrong health/model/profile/revision/status/backend/mode/readiness, malformed
  completion/usage/token evidence, port collision, mid-phase failure, premature
  exit, nonzero exit, forced shutdown, and SIGTERM/SIGHUP/SIGINT process-group
  cleanup.
- Focused verification currently passes 84 tests across the capture contract,
  GLM profile contract, and shared TensorFold HTTP contract, plus Ruff,
  `compileall`, diff checking, and a private-path/credential/noise scan.
- The first test attempt used system Python 3.9 and failed during repository
  fixture setup because Rapid requires Python 3.10+. Verification uses the
  existing Python 3.12 interpreter; this was an environment mismatch, not a
  harness failure.
- The actual product capture remains intentionally unrun and blocked until a
  swap-clear or explicitly approved stable host window. Do not clear swap,
  reboot, acquire the lock, or load the model without human authorization.
- Orca's agent-to-agent messaging command/channel was unavailable in the
  delegated shell. The PR-start FYI was sent to the parent agent for relay to
  Atlas, Pixel, Echo, and ds0731; send the completion FYI the same way.
- A local independent exact-head review requested lifecycle, provenance,
  identity/schema validation, clean-shutdown, and adversarial-test hardening.
  Those findings are implemented. The previously requested remote review host
  could not authenticate; Atlas will arrange a fresh local exact-head review,
  so no additional remote review round should be requested from this branch.

Next action for Harbor after approval: run the documented command in a stable
zero-swap window, verify `manifest.json` says `complete`, independently verify
`artifact-hashes.json`, and attach only the sanitized artifact directory to the
qualification record. Atlas decides any release action.
