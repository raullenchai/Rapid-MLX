# Vendored source provenance gate

The universal `lint` job in `.github/workflows/ci.yml` runs
`python -S scripts/check_vendored_provenance.py`. The required `tests` verdict
rejects failed or skipped lint, including documentation-only PRs. This closes
F-17 without requiring Linux runners to import the Apple-only MLX runtime.

`scripts/vendored-provenance.json` records the complete reviewed vendor file
inventory and each local SHA-256. For sources from mlx-vlm 0.7.2 it also records
the corresponding upstream wheel member and digest. The wheel URL and digest
are fixed independently of package resolution. The checker downloads only that
source artifact, verifies its SHA-256 before parsing, and reads it as ZIP data;
it never extracts, installs, or imports dependency code. Network failure fails
the check rather than silently omitting upstream validation.

The MiniCPM 4.7 addition has a different immutable upstream commit, recorded in
its existing `provenance.json`. The universal gate checks every vendored digest,
complete file coverage, and embedded commit markers against that inventory;
the inventory itself is covered by the complete-tree manifest. Existing native
behavior/parity tests remain useful and are not replaced by byte verification.

For an offline check using an already available upstream wheel:

```sh
python -S scripts/check_vendored_provenance.py --wheel /path/to/mlx_vlm-0.7.2-py3-none-any.whl
```

When intentionally changing vendored code, review the diff against its recorded
upstream source, document each deviation in the package inventory and source,
run behavior tests, then update the affected local digests. Update the nested
MiniCPM inventory too when that subtree changes. A dependency upgrade also
requires reviewing the new immutable artifact identity and upstream member
hashes. CI deliberately has no automatic manifest regeneration or
"accept current files" mode.

The manifest, checker, vendor tree, and workflow are protected trust-control
paths: a current-head human review is required before automated merge. A hash
update documents reviewed bytes; it does not itself prove source safety.
