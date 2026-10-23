# Snapshot content verification

`rapid_mlx.model_integrity.verify_snapshot` checks selected snapshot files against
metadata for an exact commit. Large files use SHA-256; ordinary Git files use the
Git blob hash, including its size header. Callers may select patterns, excluded
patterns or exact filenames. A `local_dir` destination must supply its resolved
commit explicitly.

Successful verification writes an atomic receipt keyed by repository, commit,
selection, destination and any packaged manifest. Unchanged verified files can be
checked offline without another metadata request. File identity, size and timestamp
changes invalidate the receipt. An unknown offline snapshot requires a trusted
packaged inventory or an online verification first. Cached files and partial
transfers are retained on failure so download resume remains possible.

When an explicitly requested commit is unavailable upstream, the verifier reuses
the mirror's existing complete exact-revision manifest validation, then hashes the
selected local bytes. This preserves the existing missing-revision recovery path.
Content identity does not prove publisher authenticity. Cache directories and
receipts remain locally trusted; this is not protection against a local account
that can rewrite both payloads and receipts.

This preparatory module does not change download or startup callers. Integration
must verify before recording a successful pull, preserve selected subfolders and
variants, and leave ordinary local directories and offline cached startup working.
Catalog migration and broader loader enforcement remain separate changes.
