# Catalog and media trust follow-up

Owner: Atlas; Vector owns loading/media, Harbor owns catalog review automation.
Branch: `atlas/security-audit`. Follow-up baseline: `4d309dff3`.

## Findings and evidence boundaries

- The reported MLLM `model_file` execution was a real baseline issue. The first
  audit fixed it before this follow-up. Direct `MLXMultimodalLM.load` regression
  tests now exercise a local executable-selector config and prove the upstream
  loader is never called without consent, including process opt-out overriding
  an API opt-in. The supplied existing PoC was inspected, not executed.
- The report's installer/Python archive findings described the pre-fix code.
  This branch already removed mutable installer execution and Git fallback and
  added the recorded Python archive digest check.
- Catalog-only changes were not exempt from all CI, but catalog source and
  runtime changes lacked a dedicated trust review boundary. The community
  submission validator covers benchmark submissions, not model provenance.
- Audio catalog fields selected a spaCy pipeline to install. The original
  spaCy downloader checks the publisher's compatibility catalog and constructs
  the publisher's model URL; arbitrary attacker-controlled PyPI installation
  was not established. The new code-owned allowlist selects reviewed pipeline
  versions, disallows dependency/source-package installation, and rejects
  unknown names before invoking the installer. Installed valid pipelines remain
  usable. New pipeline families or unsupported runtime versions need review.
- Vendored remote image/audio/video decoders bypassed the shared media network
  guard. These entry points now use DNS-pinned requests, per-redirect address
  checks, transfer limits and cleanup. The current chat API does not generally
  accept arbitrary `audio_url` blocks; direct vendored/processor entry points
  were nevertheless unsafe and are covered. Local decoder behavior is retained.

The original local model path guard was not wholly absent: ordinary Hugging Face
cache symlinks could fail containment. That incidental rejection does not govern
copied/local Python files or establish consent. The current policy separates
explicit execution consent from path containment and permits ordinary trusted
same-repository blob links only after consent.

## Catalog migration and residual risk

At the follow-up baseline, the text catalog contains 239 entries and only one
general `revision` pin. A pinned commit fixes artifact identity; it does not
prove the model's quality or publisher trust. Automatically recording today's
remote HEAD for every historical alias would falsely imply an audit and is not
part of this change. Existing unpinned identities remain compatible and require
a separate reviewed migration. Run the offline unpinned report to plan that
migration; it does not download model artifacts.

New or changed text checkpoint identities require a full commit revision in the
catalog audit. Semantic catalog/control changes require a maintainer's human approval of
the exact current PR head before automatic merge. The merger runs trusted base
code and rechecks eligibility immediately before its SHA-bound merge action.
Audio source changes require human review; no nonfunctional audio revision field
has been invented. Direct VLM loading now honors existing catalog pins using the
same resolution as text loading, rejecting conflicting revisions and avoiding
HEAD fallback if a pinned checkpoint cannot be resolved.

These controls do not certify checkpoint contents, retroactively pin the existing
catalog, or sandbox explicitly trusted model code. They do not replace repository
permissions/branch protection or protect arbitrary user-invoked Python sessions
from malicious local import paths.

## Verification

The combined offline regression run passed 841 tests, with 2 skips and 1
deselection. The two skips are the qualified-runtime native MTP check and
opt-in full-model DFlash inference. The deselected upstream provenance test
expects mlx-vlm 0.7.2, while this host has 0.7.1 installed. The exact pinned
0.7.2 wheel's static source region and the modified vendored module were checked
separately against the recorded SHA-256 values; no wheel code was executed or
installed. The media agent also ran the provenance assertion with that static
source path. Ruff passed across all Python files changed in the follow-up.

Reproduce the combined behavior checks from the task worktree:

```sh
HF_HUB_OFFLINE=1 python3.12 -m pytest -q \
  tests/test_catalog_trust.py tests/test_merge_on_green.py \
  tests/test_pr_validate_supply_chain.py tests/test_alias_revision.py \
  tests/test_model_file_guard.py tests/test_security_hardening.py \
  tests/test_alternate_model_load_security.py tests/test_mllm_model_file_consent.py \
  tests/test_kokoro_g2p_bootstrap.py tests/test_audio_alias_registry.py \
  tests/test_pull_audio_runtime_assets.py tests/test_kokoro_espeak_repair.py \
  tests/test_remote_audio_security.py tests/test_mlx_vlm_vendored_inputs.py \
  tests/test_mllm.py::TestMediaSecurity tests/test_native_mtp.py \
  tests/test_dflash_integration.py tests/test_hidream_o1_alias.py \
  tests/test_dspark_companion.py \
  -k 'not test_vendored_region_matches_reviewed_sources'
python3.12 -I scripts/merge_on_green.py --help
python3.12 scripts/catalog_trust.py --report-unpinned
```

Independent review closed catalog ordering, symlink, control-file coverage,
stale/bot approval and controller import-isolation findings. Orca role messaging
was unavailable in this tool session; the scoped start/completion handoff is
recorded here and was shared among the three task subagents.

## Integration and recovery

No production deployment, dependency installation, or model-weight download was
performed for this follow-up. Atlas should integrate only after validation and
Harbor should confirm the merge workflow invokes the isolated trusted controller.
Vector owns reviewed migration of historical catalog revisions. Keep the
default model-code denial while investigating compatibility failures; do not
restore unguarded URL fetching or catalog-selected arbitrary install targets.
