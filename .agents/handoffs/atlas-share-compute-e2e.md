# Share Compute end-to-end acceptance

- Receiving role: Atlas for node/gateway/ledger acceptance; Harbor for signed release validation.
- Branch: `atlas/share-compute-e2e`, based on official `main` `8a9e46fc0` (UI PR #3923).
- Host: Studio Mac, Apple Silicon arm64, macOS 26.5.2, 256 GiB memory. Docker Desktop 29.2.0, Linux arm64 containers.

## Verified on 2026-09-30

- `cd apps/rapid-mac && SKIP_SIDECAR=1 bash scripts/build.sh` produced a 55 MiB ad-hoc-signed app; `codesign --verify --deep --strict` passed. This is a developer build, not a signed/notarized release artifact.
- `python3.12 -m pytest tests/test_share_quicksilver.py tests/test_share_cli.py -q` passed 298 tests.
- A native `rapid_mlx.cli:cli_entrypoint` serve of `qwen3.8-27b-4bit` on `127.0.0.1:18888` loaded cached weights and reached `/healthz` ready. From `python:3.12-alpine` in Docker via `host.docker.internal:18888`, a non-streaming chat request returned HTTP 200, `LOCAL_OK`, and `usage` (18 input / 3 output tokens); streaming returned HTTP 200, `STREAM_OK`, and a final usage event (16 input / 3 output tokens). The native server was stopped after the probes.
- From that Docker image, unauthenticated `GET /v1/pool/summary` returned 2 ready `qwen3.8-27b` nodes; unauthenticated gateway inference and ledger each returned HTTP 401. These checks prove network reachability and auth rejection, not routing to this Mac.

## Access needed for live acceptance

- A temporary `qsppk-` provider key for a test account to register this Studio Mac under a unique worker name.
- A limited-credit `sk-` consumer inference key for `qwen3.8-27b`, plus a `qsprk-` read-only ledger key for the same provider account.
- Read-only request tracing that maps gateway request ID to node ID and accounting window, or a supported test-only route pin to this node. With multiple ready nodes, a successful response alone cannot establish which node served it.
- For a real 429 check, a staging/test quota that can be exhausted safely; do not deliberately saturate production limits.

Once access is available, start the pool node from this source worktree, send one ordinary and one streaming request from Docker through the QuickSilver gateway, match request IDs to the new node, query ledger before/after with `limit=1` and `next_cursor`, and inspect the accounting status. Keep credentials out of Git, shell history, logs, and this handoff. Signed/notarized release validation remains Harbor-owned and requires release authorization.
