# Model mirror operations

The scheduled `.github/workflows/mirror-drift-check.yml` job runs
`scripts/mirror_drift_check.py --only-used` to compare every shipped alias with
its Hugging Face source and the public R2 mirror. Run the same command locally
with the audit dependencies installed when investigating a failure. A normal
missing or mismatched mirror object is an error; an unavailable Hugging Face
source is also an error because fallback downloads cannot work.

Checkpoints that clients pull at a pinned commit are audited at that commit,
not at the default branch. If the pinned commit no longer exists upstream, the
audit reports `hf_unavailable` with the pinned revision. A packaged digest can
still authorize an exact-byte mirror pull; an unlisted pin needs a client
re-pin or restored Hub access.

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
`rapid_mlx/audio/aliases.json` or the declared runtime assets. To remove an
entry, first restore and verify the
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
unchanged. Packaged digests allow an exact-byte mirror pull when the Hub pin is
unavailable; a missing or mismatched mirror file still fails closed.

### Packaged pinned-model digests

`rapid_mlx/model_digests.json` ships with the wheel. For each mirrored pinned
alias and selected runtime asset, it records the exact commit, selected file
paths, sizes, and LFS SHA-256 or Git blob IDs. The downloader reads this file
before any Hub metadata request and verifies every R2 byte against it. The
mirror catalog still determines the download URL, but neither the catalog nor
bucket objects may supply expected hashes. `.gitattributes` and root
`README.md` model cards are omitted because they are not runtime data. Existing
Hub caches may retain them without invalidating a verified snapshot.
The puller's disk and interactive size checks use the same package inventory
for pinned aliases, so those preflights do not need a Hub request either.

After changing an alias pin, file selection, or mirror exclusion, regenerate
while the Hub is reachable:

```bash
python scripts/build_model_digests.py
python scripts/build_model_digests.py --check
python scripts/build_model_digests.py --check-r2
```

CI runs `--check` offline and compares selected file paths with
`scripts/model_digest_inventory.json`; it fails on missing files, missing
entries, mismatched revisions/selections, or invalid digests. Regeneration
updates both files from the same Hub snapshot. Run `--check-r2` outside CI
before release: it sends paced HEAD requests for every unique selected mirror
object and fails if any object is absent or unreachable. Run it after uploading
missing files and before building a release. An unlisted or unpinned repository
keeps the normal Hub metadata path. A packaged
pinned pull with a missing, corrupt, or incomplete mirror file fails closed
when the Hub is unavailable.

The daily mirror workflow separately checks all image checkpoint and auxiliary
runtime-asset pins upstream in the `image-pin-resolution` job; mirror availability
cannot hide a dangling upstream pin.
