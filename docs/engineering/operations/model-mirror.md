# Model mirror operations

The scheduled `.github/workflows/mirror-drift-check.yml` job runs
`scripts/mirror_drift_check.py --only-used` to compare every shipped alias with
its Hugging Face source and the public R2 mirror. Run the same command locally
with the audit dependencies installed when investigating a failure. A normal
missing or mismatched mirror object is an error; an unavailable Hugging Face
source is also an error because fallback downloads cannot work.

Checkpoints that clients pull at a pinned commit are audited at that commit,
not at the default branch. If the pinned commit no longer exists upstream, the
audit reports `hf_unavailable` with the pinned revision: every cold pull of that
alias fails, and the fix is a client re-pin, not a mirror re-sync.

## Intentionally unmirrored repositories

`scripts/mirror_unmirrored.json` is the single source of truth for aliases that
are served from Hugging Face on purpose. The drift audit puts these aliases in
the `unmirrored (intentional)` bucket, skips R2 catalog and object drift checks,
and still lists the repository through Hugging Face. An upstream authorization
or not-found error therefore remains a real audit failure. The uploader refuses
an entry in this list and prints its `reason` and `since` date; an authorized
one-off restoration must explicitly pass `--force-unmirrored`.

To add an entry, add exactly `hf_path`, `reason`, and an ISO `since` date to the
JSON file. The loader rejects duplicates, unknown keys, invalid dates, and any
stale `hf_path` that is not defined by `rapid_mlx/aliases.json` or
`rapid_mlx/audio/aliases.json`. To remove an entry, first restore and verify the
repository while the protection is still present:

```bash
python scripts/mirror_to_r2.py owner/repo --force-unmirrored
```

Only after that command succeeds should the entry be removed and the drift
audit rerun. Mirroring writes production data, so it requires explicit human
authorization and valid R2 credentials. If restoration fails, leave the entry
in place; the uploader is resumable and a retry skips already verified objects.

### Checkpoint pins

The drift audit compares a revision-pinned alias with that exact commit, so a
moving upstream default branch does not create drift. `mirror_to_r2.py` uses
the commit a pinned client pulls — the catalog alias pin, or the image loader
pin in `IMAGE_MODEL_REVISIONS` when the alias has none — when `--revision` is
omitted, and rejects an explicit conflicting revision before listing or
uploading files. The existing per-file mirror layout and verification remain
unchanged. Recovery after upstream deletes the commit is separate work (#4994);
an unavailable pin still fails closed.

### Missing upstream image revisions

When upstream returns `RevisionNotFoundError` for an immutable pin, the client
requires `/<owner>/<repo>/.rapid-mlx/revisions/<commit>.json` on the configured
mirror. The manifest must identify the exact `repo_id` and `revision`, include
`file_inventory` (the complete list of paths at that revision), and list
`files`, each with a safe repository-relative `path`, nonnegative integer `size`,
and either a lowercase `sha256` (file bytes) or `git_oid` (Git blob hash).
Every inventory path must have exactly one hashed file entry, with no extras or
duplicates; partial hash manifests are rejected even for wildcard selections.
The configured mirror is the trust root for this recovery metadata; mutable
catalog entries and object headers cannot substitute for it. Operators must
preserve manifests from the original reviewed revision, never reconstruct them
from a newer upstream revision. This change does not publish manifests or write
to the mirror.

The existing mirror downloader checks each selected file against these hashes,
including cached files. Missing or invalid manifests, unavailable mirror files,
and integrity mismatches fail closed. No moving upstream branch is substituted.
The daily mirror workflow separately checks all image checkpoint and auxiliary
runtime-asset pins upstream in the `image-pin-resolution` job; mirror availability
cannot hide a dangling upstream pin.
