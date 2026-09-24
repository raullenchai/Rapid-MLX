# Model mirror operations

The scheduled `.github/workflows/mirror-drift-check.yml` job runs
`scripts/mirror_drift_check.py --only-used` to compare every shipped alias with
its Hugging Face source and the public R2 mirror. Run the same command locally
with the audit dependencies installed when investigating a failure. A normal
missing or mismatched mirror object is an error; an unavailable Hugging Face
source is also an error because fallback downloads cannot work.

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
