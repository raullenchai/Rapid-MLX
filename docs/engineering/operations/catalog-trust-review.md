# Catalog source review and revision migration

Alias data controls which checkpoint, configuration, tokenizer and runtime users
load. Successful benchmark-schema validation does not establish the provenance
of those artifacts. The community-submission workflow remains a benchmark-data
validator. Ordinary alias changes already select the engine CI lane; they are
not exempted from CI as documentation-only changes.

The offline `scripts/catalog_trust.py` policy compares the trusted base catalog
with the proposed catalog. Every text, image, video, embedding, audio and runtime-asset identity
requires a full lowercase 40-hex commit SHA, even when unchanged. A SHA identifies content; it is not a publisher signature or proof
that someone reviewed the content. Unpinned identities fail CI; there is no grandfather exemption.

`pr_validate` reports these violations as blocking supply-chain findings. The
trusted default-branch merge controller independently reads the base/head JSON
through the GitHub API, rejects symlink catalogs and incomplete inventories, and
applies the same identity policy. It does not trust the PR's modified validator
or the GitHub diff patch. Before automatically merging semantic catalog changes
or trust-control changes, it requires an approval from a human with repository
write/maintain/admin permission on the exact current head. Bot/self approvals,
stale approvals, dismissals and unresolved human change requests do not satisfy
that requirement. Changes to the policy, workflow, schema, package/runtime
allowlists and model-code guards also require this approval. The token-bearing
merge controller runs with Python isolated mode and loads its adjacent reviewed
policy explicitly, so checkout files cannot shadow its Python imports.

Only `notes`, `description`, `display_name` and the top-level `_comment` are
cosmetic. Runtime defaults, capabilities, parser selection, alias removal and
alias ordering require review. Ordering matters because repository reverse
lookup uses the first matching alias. Audio runtime package/source edits also
require review. This gate does not change loaders or certify that a proposed pin
is consumed by every runtime. Reviewers must verify runtime pin consumption before
approving a new audio or runtime-asset identity.

## Bumping a pin

Resolve metadata only (no weights) with the Hugging Face API:

```python
from huggingface_hub import model_info
info = model_info("owner/repository", revision="main")
print(info.sha)
```

Review the repository and commit before changing `revision` in
`rapid_mlx/aliases.json` or `rapid_mlx/audio/aliases.json`. Update all aliases
sharing that repository consistently. Keep image/video loader pins and
speculative-decoding target pins in agreement. Audio `_runtime_assets` also
carry `revision`; their consumers use the shared download gate.

Check the project mirror catalog and any exact-revision manifest before choosing
the commit. Preserve a recorded served revision rather than inferring it from
upload timestamps. Report disagreements; do not silently switch runtime pins to
a newer upstream tip. A pin does not certify model quality or publisher identity.
The catalog migration preserves existing runtime pins where upstream has advanced.
The public mirror catalog currently does not publish revision fields; per-file
integrity checks still decide whether mirror bytes match the requested revision.

Run the offline catalog policy and focused download tests:

```sh
python scripts/catalog_trust.py --base <exact-base-commit-sha>
python -m pytest tests/test_catalog_trust.py tests/test_alias_revision.py
```

Pinned catalog names select their exact snapshot unless a runnable checkpoint
is selected from a configured local model folder. Explicit filesystem paths also
remain local. Hub downloads always use the catalog pin. Uncataloged repository
IDs retain their existing behavior. No catalog migration downloads weights or
publishes mirror data.

The policy takes effect for automatic merges after the controller changes are
merged to the trusted default branch. It does not change GitHub branch-protection
settings or prevent administrators from manually merging/pushing. Repository
owners retain that governance responsibility. Rollback must retain an explicit
human-review path for catalog and trust changes; do not restore unattended
source switching to get around a failing validation gate.

A current-head approval should record the reviewed repository and commit, why the
publisher/source is appropriate, the selected files and runtime configuration,
and positive loading/cache tests for the affected family. Check that all aliases
for the identity agree, including subfolders and quantization variants. Pin format
validation alone does not supply this evidence or certify existing entries.
