# Candidate routing shadow

Tracking: #2252. This observer audits the next candidate-routing boundary while
all integration candidates continue running their existing complete checks.
It never authorizes reduced CI or emits reusable complete evidence.

The independent `Candidate routing shadow` workflow follows successful queue
attestation. It checks out default-branch code and downloads exactly one
non-expired complete CI artifact from that trusted attestation run. The source
attestation run ID is bound again in the artifact. Read-only permissions prohibit
publishing statuses or changing rollout settings. The original full attestation
workflow, indexing, main reuse and merge conditions are unchanged, so observation
cannot delay the completion of their workflows.

The observer checks the trusted queue PR identity, candidate tree and first
parent against the exact main base. Real queue merge commits have two parents;
the first-parent binding identifies the base. Both names in a rename participate
in the actual combined diff. Empty, truncated, unmapped, mixed or control changes
remain full. A mapped diff is provisionally eligible only when the latest push
CI on the exact current main base completed successfully and actually executed
all nine CPU jobs, five model jobs, Apple and required static/coverage jobs. An
older success cannot override a newer red, pending or cancelled attempt. The
main tip and latest run/attempt are rechecked after inspection.

The separate artifact uses schema `rapid-mlx/candidate-route-shadow/v1`, scope
`advisory-candidate-path-and-base`, and `authorizes_reduced_ci=false`. Mapped
test execution is not inferred from this observation: it explicitly records
`execution_proof_checked=false`. API misses, malformed identities and skipped
full jobs produce a full-route observation. This initial shadow does not accept
main whose complete jobs were skipped for authenticated candidate reuse; that
proof must be explicitly revalidated before actual candidate routing accepts it.

Observations run after attestation, so main may already have advanced when they
start. Such a snapshot remains full and must not be counted as evidence that an
earlier candidate was ineligible. These snapshots are not before/after latency
benchmarks or permission to turn on a candidate canary.

Actual reduced routing remains a subsequent reviewed change: require complete
mapped execution evidence, an authenticated reduced qualification namespace,
fresh base qualification at the merge gate, a default-off rollout switch and
full main execution after each reduced landing. Complete namespace consumers
must continue rejecting reduced artifacts. Releases keep complete qualification.
