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

SCHEMA_VERSION: int = 1
"""Bump in lockstep with ``community-benchmarks/schema.json``'s
``schema_version`` const. Submissions carry this so the aggregator can
ignore rows from a schema it doesn't understand instead of failing."""

# Synthetic prompt seed. Locked per schema_version so the prompt_hash
# field in submissions stays bit-stable across all submitters — that's
# how the GHA validator detects "this number wasn't measured with the
# standardized prompt" tampering. The seed is just an integer; the
# prompt itself is generated from it deterministically in runner.py.
PROMPT_SEED: int = 0xC0FFEE
