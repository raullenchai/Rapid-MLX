# Vector: Qwen3.6 35B MTP product default

## PR intention

Make the qualified standard MTP path the safe text-generation default for
`qwen3.6-35b-4bit` in CLI and Desktop, while preserving an explicit ordinary
decode escape hatch and keeping the incompatible Desktop vision lane out of
the same process.

Owner: Vector. Host: Studio. Branch: `vector/qwen36-mtp-auto`. Worktree:
`/private/tmp/rapid-mlx-qwen36-mtp-auto`.

Scope: exact-artifact alias qualification/default metadata, Desktop launch
conflict resolution and user-facing trade-off copy, regression tests, and
reproducible performance/quality evidence. Non-goals: no new MTP algorithm,
no dynamic model/lane switching, no changes to other model defaults, and no
model downloads or quantization.

Verification: alias/CLI routing contracts, Desktop Swift tests, mixed coding,
reasoning, creative-writing, JSON, and tool-call dogfood, self adversarial
review, and `pr-validate` at the exact pushed head.

## Reference check (private)

The serving/configuration paths in vLLM and SGLang were checked first; both
keep speculative decoding an explicit capability-scoped server configuration.
MLX-LM, MLX-VLM, and oMLX were then checked for Apple-specific MTP and cache
constraints. The adopted Rapid pattern is exact-artifact qualification plus a
clear escape hatch, with vision and speculative decoding treated as mutually
exclusive process lanes. This avoids architecture-name inference and silent
fallback.

## Coordination

The Orca PR-start message was attempted before implementation, but the current
run rejected the broadcast target. This tracked handoff records the required
FYI until the messaging channel is available again. Pixel impact is limited to
the existing Performance control and launch flags; Atlas owns final product
default/release disposition.

## Verified facts

- The exact alias now boots standard continuous MTP without an extra flag;
  explicit `--mllm` suppresses only the automatic MTP choice.
- A cached-artifact HTTP launch loaded the target and sidecar, selected BF16
  KV cache and the text-only continuous scheduler, completed warmup, and
  returned valid sampled structured JSON for `17 * 19` with answer `323`.
- The mixed task gate observed 88.6 to 121.7 tok/s coding, 99.8 to 139.9 tok/s
  reasoning, 94.8 to 95.0 tok/s creative writing, 100.6 to 113.1 tok/s JSON,
  and 46.4 to 50.1 tok/s tool arguments. All five outputs passed their task
  contract.
- Python alias/routing suites passed 3,546 tests; the focused follow-up passed
  54 tests. Desktop spawn-argument tests passed 27/27. Ruff, format, JSON, and
  diff checks are clean.
- Adversarial review found and fixed the default-off transition: removing MTP
  now also removes its generated `--text-only` flag and restores `--mllm` for
  a vision-capable checkpoint, while respecting an explicit user text-only
  override.
