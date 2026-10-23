# Catalog source review and revision migration

Alias data controls which checkpoint, configuration, tokenizer and runtime users
load. Successful benchmark-schema validation does not establish the provenance
of those artifacts. The community-submission workflow remains a benchmark-data
validator. Ordinary alias changes already select the engine CI lane; they are
not exempted from CI as documentation-only changes.

The offline `scripts/catalog_trust.py` policy compares the trusted base catalog
with the proposed catalog. New text/audio aliases, changed repository/subfolder
sources, and changed or removed revision pins require a full lowercase 40-hex
commit SHA. A SHA identifies content; it is not a publisher signature or proof
that someone reviewed the content. Existing unpinned identities are grandfathered
so a cosmetic edit does not require blindly pinning the entire catalog.

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

To inventory the remaining text-catalog migration offline:

```sh
python scripts/catalog_trust.py --report-unpinned
python scripts/catalog_trust.py --base <exact-base-commit-sha>
```

Existing unpinned entries remain unreviewed by this policy. For each source migration, review the actual repository
commit/config/code, choose that immutable SHA, update all aliases sharing its
repository consistently, and verify the appropriate runtime consumes the pin.
Do not bulk resolve current remote HEAD and label the result reviewed. These
commands download no weights and make no network requests.

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
