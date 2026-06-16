# SPDX-License-Identifier: Apache-2.0
"""Community-submitted benchmark database (`rapid-mlx bench --submit`).

Layered to keep the privacy surface auditable:

- ``hardware`` — non-privileged Apple Silicon probes. The full set of
  shell commands invoked is the union of constants at module top. No
  network, no privileged sysctls, no file reads outside ``/usr/bin``.
- ``runner`` — standardized bench loop. Two buckets, 5 measured rounds
  + 1 warmup, greedy decoding by default. Locks every comparability
  parameter; users who want to tune them have to drop ``--submit``.
- ``submission`` — builds the JSON payload (matching
  ``community-benchmarks/schema.json``), prompts for explicit y/N
  consent, and only then invokes the user's local ``gh`` CLI.

Each layer can be unit-tested independently; nothing imports model
weights or MLX state until ``runner`` runs.
"""

SCHEMA_VERSION: int = 2
"""Bump in lockstep with ``community-benchmarks/schema.json``'s
``schema_version`` enum. Submissions carry this so the aggregator can
ignore rows from a schema it doesn't understand instead of failing.

v1 → v2 (additive only): three optional top-level fields — ``tier``,
``smoke_result``, ``harness_result``. A v2 submission with only
``schema_version`` bumped and none of the new fields populated is the
SAME wire shape as v1 minus the version integer, so the aggregator can
treat it interchangeably. The new fields kick in when the CLI is run
with ``bench --tier {smoke,harness,all} --submit`` — the dispatch for
``--tier all --submit`` itself is wired in a follow-up PR; this module
just exposes the kwargs so the payload builder can carry the data when
the CLI calls it."""

# Synthetic prompt seed. Locked per schema_version so the prompt_hash
# field in submissions stays bit-stable across all submitters — that's
# how the GHA validator detects "this number wasn't measured with the
# standardized prompt" tampering. The seed is just an integer; the
# prompt itself is generated from it deterministically in runner.py.
PROMPT_SEED: int = 0xC0FFEE
