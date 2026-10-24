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
| F-03 tokenizer auto-map | Default remote-code denial with explicit process/API policy | Native GLM and reviewed Kimi text/vision adapters preserve shipped compatibility; a pin never implicitly authorizes repository code |
| F-04 catalog identity | 241 text and 44 audio aliases pinned; 282 repository identities with full file metadata | Existing pins and cached revisions are retained where available; pins establish identity, not model quality |
| F-05 installer/update | Remove mutable Git fallback and downloaded shell update; exact commit required for source installs | Stable/versioned package installs and current-environment updates retained |
| F-06 desktop runtime | Automatic discovery uses bundled runtime; explicit developer override remains deliberate consent | Historical unsigned slim payloads need full-app migration; see desktop-runtime-trust.md |
| F-07 Python runtime archive | Fixed SHA-256 checked for new and cached build artifacts | Existing validated archive remains reusable |
| F-08 audio dependency bootstrap | Code-owned spaCy pipeline/version allowlist with wheel-only, no-dependency install | Arbitrary PyPI installation was not demonstrated by the original report; known pipelines preserved |
| F-09 vendored URL loading | Shared DNS-pinned, redirect-checked, size-limited media downloads | Local data paths and decoding retained; direct library helpers covered |
| F-10 SA3 legacy objects | Restricted loading first; unrestricted conversion needs both explicit consent and the exact file SHA-256 | A private copy is hashed and loaded to avoid path replacement; ordinary tensor checkpoints need no unsafe consent |
| F-11 weight integrity | Verify selected bytes in pull, automatic serving, shared snapshot/file downloads, routing metadata, music and import paths | Packaged hashes support offline checks; unchanged stat-bound receipts avoid repeated full reads; failures retain resumable files |
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
these tests. Kimi text tests include a real tiny native MLX model forward and
fixed token IDs from the actual pinned vocabulary; Kimi vision tests use the
real image processor with stub model weights. GLM REAP tests use real tiny
safetensors and tokenizer data with a repository-code execution sentinel.

An isolated validation environment uses the project-required mlx-vlm 0.7.2.
The built wheel was inspected for the model integrity lock, native adapters,
Kimi license/provenance, and the direct tokenizer dependency. No model weights
were downloaded, and no installed production environment was modified.

Use the final branch commit, rather than the original report's baseline, for a
follow-up audit. Do not restore mutable references, raw URL fetches, or implicit
pickle execution to resolve a compatibility issue; repair the affected adapter
and retain a regression test.

## Final validation checkpoint

Integration checkpoint: `dfb36adbb`, including current-main mirror recovery and
reviewed vendor provenance. On Studio,
Python 3.12 with mlx-vlm 0.7.2 and patched PyTorch 2.10.0:

- Combined integration: **2,191 passed, zero failures/skips/deselections**,
  across the 44 files below.
- Earlier security suite plus integrity/catalog checks: **919 passed, one
  skipped**, zero deselections. The skip is the opt-in full-model DFlash test,
  which requires approximately 30 GB of weights.
- Ruff passed for the changed authored Python files. The imported tokenizer
  source and tokenizer-template fixture retain upstream whitespace; their
  provenance hashes bind those exact bytes.
- Wheel and sdist builds succeeded. Packaged catalog lock, adapters, tokenizer
  dependency and license/provenance files were inspected. Extracted-wheel LTX
  runtime imports passed with the pinned runtime dependency, and nine focused
  Swift sidecar tests passed. Hardware-qualified affine8 tests were skipped on
  this validation host.
- Three existing cached model configurations passed actual offline content
  verification. Full-size model weights were not downloaded for this work.

The following combined command assumes an environment with the project's test
dependencies and pinned vision runtime. Put pytest scratch **outside** the
validation virtualenv: local-search tests intentionally exclude directories
under a `pyvenv.cfg` ancestor.

```sh
mkdir -p /private/tmp/rapid-mlx-security-integration
HF_HUB_OFFLINE=1 RAPID_MLX_TELEMETRY=0 \
TMPDIR=/private/tmp/rapid-mlx-security-integration \
python -m pytest -q \
  tests/install/test_install_security.py \
  tests/install/test_install_stdin.py \
  tests/test_alias_revision.py \
  tests/test_byom_imports.py \
  tests/test_byom_preflight.py \
  tests/test_catalog_trust.py \
  tests/test_cli_offline_serve.py \
  tests/test_deepseek_v32_indexer_gate.py \
  tests/test_disk_stream_cli_wiring.py \
  tests/test_dspark_companion.py \
  tests/test_f5_checkpoint_security.py \
  tests/test_hidream_o1_alias.py \
  tests/test_kimi_tokenizer_security.py \
  tests/test_kimi_vlm_security.py \
  tests/test_local_search.py \
  tests/test_merge_on_green.py \
  tests/test_model_catalog_lock.py \
  tests/test_model_downloads.py \
  tests/test_model_file_guard.py \
  tests/test_model_integrity.py \
  tests/test_music_engine.py \
  tests/test_pull_audio_alias_resolution.py \
  tests/test_pull_audio_runtime_assets.py \
  tests/test_pull_recovery.py \
  tests/test_pull_summary.py \
  tests/test_pull_variant_selection.py \
  tests/test_pull_variant_serve_resolution.py \
  tests/test_qwen3_tts_voicedesign.py \
  tests/test_sam_audio_security.py \
  tests/test_security_hardening.py \
  tests/test_sensevoice_stt.py \
  tests/test_serve_command_characterization.py \
  tests/test_serve_listen_fd.py \
  tests/test_server_exposure.py \
  tests/test_server_load_model_order.py \
  tests/test_server_start_state.py \
  tests/test_system_one_clef_mlx.py \
  tests/test_system_one_cli.py \
  tests/test_system_one_convert_clm.py \
  tests/test_tensor_checkpoint_security.py \
  tests/test_tts_checkpoint_security.py \
  tests/test_upgrade_cli.py \
  tests/test_vendored_provenance_gate.py \
  tests/test_version_check.py
```

Atlas integration handoff: the branch is ready for code review with the
compatibility transitions above. Harbor must retain the required provenance
and catalog checks. Pixel owns the historical slim-to-full-app migration.
Production merge/release/deployment is not performed by this audit. Orca role
messaging was unavailable; implementation and independent review were
coordinated with the three task subagents and recorded here.

## Local review and CI follow-up

The owner selected an independent Codex process on Studio for formal review.
Round 1 found two valid regressions: cached loading could reuse previously
rejected bytes, and an alias resolved to an external local directory could
enter Hub downloading. Both now have positive and negative regression coverage.
Text/image warm loading checks selected content offline against packaged
manifests or prior snapshot-scoped verification evidence. External directories
retain intentional local semantics.

CI follow-up also preserves modality-specific cache completeness, warm pinned
image disk checks, typed Kimi loading errors, and Linux temporary-file behavior.
Synthetic transport tests now mock the owned download boundary; actual integrity
tests continue to reject altered bytes. Portable tests keep running on Linux,
while tests requiring MLX are classified in the Apple lane. The CLI snapshot
includes the intentional public-listening consent flag.

The expanded combined selection passed **3,824 tests**, with one existing
opt-in full-model DFlash skip and zero failures (86 files, 223.76 seconds).
Repository Ruff lint/format checks passed; the type-error budget did not grow.
This evidence does not substitute for required GitHub CI or the next review.

After integrating the landed installer, audio, validation and primary-loader
changes through main `9d2ef9e45`, the expanded regression run passed **4,103**
tests with the same single opt-in DFlash skip (95 files, 264.23 seconds).
Upstream Kimi token API, special-token chunking, JSON tool-schema and EOS fixes
are retained alongside cached-content verification and explicit external-root
resolution. Final mypy, dependency renderer, wheel/sdist license inclusion and
vendored provenance checks passed. The catalog now includes main's two reviewed
Bolt publications without changing their immutable revisions.
