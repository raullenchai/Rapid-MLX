# Executable-input security audit handoff

Receiving owners: Atlas (integration), Vector (model compatibility), Harbor and
Pixel (desktop migration). Host: Studio. Branch: `atlas/security-audit`.
State: implementation complete, ready for review; no release or deployment.

All five reported categories were confirmed against baseline
`366975c0f438d83d305df694e6a2cbf106880f88`. The fixes require explicit repository
Python consent, cover primary and auxiliary loaders, remove automatic mutable
installer execution and Git fallback, retire automatic writable desktop runtime
discovery, verify embedded Python archives, and align arsenal pins. See
`docs/engineering/decisions/2026-10-22-executable-input-trust.md` and operational
guides `model-code-trust.md`, `installer-trust.md`, `desktop-runtime-trust.md`.

Verification on macOS/Python 3.12.14:

```sh
HF_HUB_OFFLINE=1 python3.12 -m pytest -q \
  tests/test_model_file_guard.py tests/test_security_hardening.py \
  tests/test_alternate_model_load_security.py tests/test_byom_imports.py \
  tests/test_system_one_decider.py tests/test_mlx_vlm_vendored_drafters.py \
  tests/test_diffusion_engine.py tests/test_version_check.py \
  tests/install/test_install_security.py tests/install/test_install_stdin.py \
  tests/test_sidecar_supply_chain.py tests/test_rerank.py \
  tests/test_system_one_openjev.py tests/test_dspark_companion.py \
  tests/test_hidream_o1_alias.py tests/test_native_mtp.py \
  tests/test_dflash_integration.py tests/test_chat_template_registry.py \
  tests/test_chat_template_type_guard.py tests/test_glm5_shardwise_load.py \
  tests/test_minicpmv4_7_contract.py tests/test_prism_hadamard_qwen35.py
```

Result: 924 passed, 2 skipped. Skips: native MTP requires the qualified mlx-vlm
release (installed 0.7.1, qualified 0.7.2), and opt-in DFlash full-model inference.
No model weights were downloaded. Harbor additionally verified 78 native Swift
tests and 33 shell installer checks. Ruff passed for all changed Python files;
shell syntax and diff whitespace checks passed. Independent subagent review
checked actual installed upstream loader signatures and execution ordering.

Known limitations: this is not an artifact-signing or complete dependency-lock
rollout. Initial installer bootstrap and package-index trust remain. Explicit
model-code consent permits unsandboxed execution. Entire Hugging Face repository
directories symlinked to cold storage may conservatively fail containment when
custom model code is explicitly enabled. Existing slim desktop installs require
the full app; developer builds without a sidecar require explicit RAPID_BIN.

Next action: Atlas reviews/integrates this branch, with Vector validating custom
model compatibility and Harbor verifying the qualified runtime before a release.
Any production deployment requires human authorization. Original workspace
conflicts were preserved; all changes were made in the isolated task worktree.
