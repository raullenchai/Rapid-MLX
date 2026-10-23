# Snapshot content verification

`rapid_mlx.model_integrity.verify_snapshot` checks selected snapshot files against
metadata for an exact commit. Large files use SHA-256; ordinary Git files use the
Git blob hash, including its size header. Callers may select patterns, excluded
patterns or exact filenames. A `local_dir` destination must supply its resolved
commit explicitly.

The verifier's `revision` and `resolved_revision` arguments require full lowercase
40-character commit SHAs. Symbolic revisions must be resolved before calling it;
the explicit CLI pull path supplies validated catalog pins or a resolved snapshot.

Successful verification writes an atomic receipt keyed by repository, commit,
selection, destination and any packaged manifest. Unchanged verified files can be
checked offline without another metadata request. File identity, size and timestamp
changes invalidate the receipt. An unknown offline snapshot requires a trusted
packaged inventory or an online verification first. Cached files and partial
transfers are retained on failure so download resume remains possible.
Selected local files absent from the immutable manifest are refused, including
files added after a receipt was created. Unselected variants and local download
bookkeeping are excluded from this inventory comparison.
If verified content resides in a read-only cache or a receipt cannot be saved,
the pull still succeeds with a diagnostic. A later pull may need online
verification again because no reusable receipt was persisted.

When an explicitly requested commit is unavailable upstream or the Hub transport
cannot be reached, the verifier reuses
the mirror's existing complete exact-revision manifest validation, then hashes the
selected local bytes. This preserves the existing missing-revision recovery path.
Content identity does not prove publisher authenticity. Cache directories and
receipts remain locally trusted; this is not protection against a local account
that can rewrite both payloads and receipts.

Explicit `rapid-mlx pull` now verifies the primary snapshot and declared runtime
assets on both Hub and mirror paths before publishing completion summaries or
variant markers. Selected subfolders and explicit quantization variants use the
same file filters for transfer and verification. A previously verified cached
pull works offline; an older cached pull without a receipt needs one online
verification before it can report verified completion offline.

Ordinary local-directory loading, cached serving, MTP setup and automatic startup
are unchanged. This is an explicit-pull boundary, not universal loader enforcement.
Full catalog pin migration, implicit download wrappers and separately downloaded
auxiliary files remain follow-up work.
