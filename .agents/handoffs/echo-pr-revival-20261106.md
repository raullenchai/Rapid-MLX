# Echo → Echo / Atlas: upstream PR revival

Branch: `echo/pr-revival-20261106`; evidence at
`docs/engineering/community/pr-revival-2026-11-06.md`.

Verified: Rapid-MLX 0.16.8 with cached Qwen3.5-4B-MLX-4bit on M3 Ultra;
LiteLLM 1.108.2 SDK completion, streaming, add-tool roundtrip and authenticated
proxy; OpenCode 1.19.0 read-tool roundtrip; older installed Hermes v0.9.0 custom
endpoint smoke. All four upstream follow-ups are linked in the evidence record.

Outstanding: maintainers' response to OpenCode reopen request and models.dev
scope question, current-main Hermes verification, actual models.dev schema-v2
migration and validation. Hermes #136903 is open; other three PRs remain closed.
Do not report any of these follow-ups as accepted, merged or vendor endorsement.

Next action: inspect replies before posting again. Prepare minimal models.dev
v2 metadata if local-runtime scope is welcomed. Rebase/update the existing
OpenCode docs PR once reopen/scope is settled. Keep LiteLLM on the compatible
endpoint path, with proxy master-key authentication included in examples.

Risks: installed Hermes is older than the existing PR's reported version;
short smokes do not establish long-context, vision, concurrency or coding quality.
Do not reuse local session dumps or credential files in public evidence.
