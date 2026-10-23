# Supply-chain audit disposition, 2026-10-22

Owner: Atlas, with Vector owning model/runtime compatibility and Harbor owning
artifact and CI integrity. Branch: `atlas/security-audit`. The supplied audit
used baseline `4a52b5939`; findings against that baseline must not be confused
with verification of this branch. No production release or deployment is
included in this work.

## Finding map

| Finding | Implemented control | Compatibility evidence or boundary |
| --- | --- | --- |
| F-01 MLLM repository Python | Resolve and inspect the exact snapshot before the shared VLM loader; separate code consent from path containment | Main and auxiliary model loaders have refusal tests; built-in overlays remain supported |
| F-02 text repository Python | Explicit consent independent of cache symlink layout; local paths inspected too | Cache blob links are supported after consent; external model roots retain precedence |
| F-03 tokenizer auto-map | Default remote-code denial with explicit process/API policy | Repository code is never implicitly authorized by a catalog pin |
| F-04 catalog identity | Immutable text/audio revisions and nested-asset identity lock with metadata evidence | Existing pins and cached revisions are retained where available; pins establish identity, not model quality |
| F-05 installer/update | Remove mutable Git fallback and downloaded shell update; exact commit required for source installs | Stable/versioned package installs and current-environment updates retained |
| F-06 desktop runtime | Automatic discovery uses bundled runtime; explicit developer override remains deliberate consent | Historical unsigned slim payloads need full-app migration; see desktop-runtime-trust.md |
| F-07 Python runtime archive | Fixed SHA-256 checked for new and cached build artifacts | Existing validated archive remains reusable |
| F-08 audio dependency bootstrap | Code-owned spaCy pipeline/version allowlist with wheel-only, no-dependency install | Arbitrary PyPI installation was not demonstrated by the original report; known pipelines preserved |
| F-09 vendored URL loading | Shared DNS-pinned, redirect-checked, size-limited media downloads | Local data paths and decoding retained; direct library helpers covered |
| F-10 SA3 legacy objects | Restricted loading first; unrestricted conversion needs both explicit consent and the exact file SHA-256 | A private copy is hashed and loaded to avoid path replacement; ordinary tensor checkpoints need no unsafe consent |
| F-11 weight integrity | Verify selected checkpoint bytes before successful pull, including mirror and Hub paths | Packaged hashes support offline checks; unchanged stat-bound receipts avoid repeated full reads; failures retain resumable files |
| F-12 dependency drift | Align mlx-arsenal CLI and sidecar version at 0.12.1 | Deliberate version bumps require review |
| F-13 catalog review gates | Identity revisions, complete file inventory, semantic catalog/control review and current-head human approval | Trusted isolated controller; stale or bot approval does not authorize automatic merge |
| F-14 auxiliary loaders | Shared guarded loaders including Clef, rerank, DFlash, DSpark, native MTP, HiDream, OpenJev and Decider | Clef native path independently tested; no bypass through the upstream model_file selector |
| F-15 VoxCPM legacy weights | Owned loader with patched explicit restricted deserialization, followed by native sanitization | Tensor-only audiovae.pth retained; embedded/safetensors VAE weights preferred; no global monkeypatch |
| F-16 SAM nested encoder | Owned SAM/T5 tensor loaders and native tokenizer with remote code disabled | Safetensors preferred; numeric .pt/.bin supported through restricted loading; corrupt safetensors never falls back |
| F-17 vendored provenance | Required CI checks complete 58-file inventory and fixed upstream wheel source | Portable static verification needs no MLX installation or wheel execution |
| F-18 accidental public access | Non-loopback bind requires API key or explicit public-unauthenticated consent | Localhost normalized to literal loopback; actual inherited sockets inspected; serve/module/System One covered |

## Explicit compatibility transitions

The audit changes executable trust decisions. Arbitrary checkpoint Python is
not a supported implicit side effect: use a built-in implementation or explicitly
consent after reviewing the checkpoint. Such consent is not a sandbox.

Legacy tensor deserialization requires stable PyTorch >=2.10.0 because
[GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
affects restricted loading through 2.9.1. The relevant dependency floors are
raised; stricter existing floors remain unchanged. `weights_only=True` is passed
explicitly, so the environment cannot silently disable it. Safetensors paths do
not require this legacy deserialization step.

For SA3 conversion of a reviewed checkpoint containing Python objects, both
`RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE=1` and
`RAPID_MLX_SA3_TRUSTED_SHA256=<exact-file-sha256>` are required. The latter is
per-artifact operator consent, not a publisher signature. Prefer tensor-only
checkpoints and do not set unsafe conversion flags on a long-running server.

An intentionally unauthenticated server behind an independently authenticated
proxy can pass `--allow-unauthenticated-public`; normal public servers should set
`RAPID_MLX_API_KEY`. Local search retains its stricter authentication requirement
regardless of that flag. Authentication and network configuration should be
validated before exposing the proxy.

Historical unsigned slim desktop runtimes cannot be retroactively authenticated
by a nearby checksum or a signature on only the interpreter. The available safe
migration is the current full app. This is a known migration requirement and
prevents an unconditional claim that every historical installation is unaffected.

## Verification limits

Negative security tests are accompanied by positive tensor-loading, tokenizer,
CLI, download-resume, native sanitization, and audio-route regression tests.
Model metadata collection does not download or execute repository Python or
model weights. Full inference for every catalog entry is not established by
these tests. Exact test results are recorded after the final combined run.

Use the final branch commit, rather than the original report's baseline, for a
follow-up audit. Do not restore mutable references, raw URL fetches, or implicit
pickle execution to resolve a compatibility issue; repair the affected adapter
and retain a regression test.
