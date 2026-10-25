# Runtime model downloads

Implicit model and auxiliary asset downloads use `rapid_mlx.model_downloads`.
The helpers select the catalog revision (including audio), or a reviewed
auxiliary revision from `runtime_model_pins.py`, and refuse conflicting or
mutable revisions. Callers with an explicitly resolved full commit can pass
`revision=`. Unregistered auxiliary repositories require a verified local
checkpoint, provisioned with `rapid-mlx pull`, or a reviewed pin addition.

Both snapshot and individual-file helpers verify the selected bytes through
`model_integrity.verify_snapshot` before returning. Mirror-backed runtime
snapshots use the same boundary. Offline or disabled-download operation requires
an unchanged verification receipt or a packaged integrity manifest. An old cache
without either must first be verified by an online pull.

Pins establish immutable content identity, not publisher authenticity. Most
integrity metadata comes from the Hub; the packaged VAD manifest is independent.
Explicit local model paths remain operator-controlled.

When updating an auxiliary pin, inspect the upstream configuration and its
consumer, update the full commit, and run the runtime download and integrity
tests. Catalog pins belong in the existing catalogs, not the auxiliary map.
Nested dependency loaders must receive verified local paths or an isolated
adapter to these helpers. Do not replace process-wide Hub functions.

The AST guard in `tests/test_verified_runtime_downloads.py` rejects new direct
Hub downloads outside the documented transport/provisioning allowlist. Vendored
LTX adaptations are reproduced by `scripts/vendor_ltx25.py`; the source archive
pin and the adapted-tree digest remain separately recorded.
