# Vector → Atlas: CLI probe access logs (#5639)

- Branch: `vector/quiet-probes-5639`, based on main `aa4d679ec`.
- Owner/host: Vector, Studio. Scope: shared CLI Uvicorn runner; no API or response changes.
- Finding: Desktop already filters selected successful polling logs in its drawer;
  the CLI previously emitted every successful health probe at INFO.
- Change: During `AcceptingConnectionsServer.run`, suppress only standard Uvicorn
  INFO access records for 2xx GET/HEAD `/healthz` and `/livez`, including query
  strings. DEBUG retains them. Failures, ordinary requests, and application logs
  pass through. Remove the filter on exit, including exceptions. Installation
  happens in the serving process after worker logging configuration.
- Verification: `venv/bin/python -m pytest tests/test_probe_access_logging.py
  tests/test_uvicorn_mcp_startup.py -q` — 63 passed on macOS/Python 3.12.14.
  Includes real Uvicorn HTTP subprocesses at INFO and DEBUG, GET/HEAD/query,
  draining 503, ordinary traffic, and filter cleanup. No model downloads.
- Limits: Access log status alone cannot distinguish a 200 health response with
  `ready: false`; the existing readiness/response semantics remain unchanged.
  Direct third-party `uvicorn rapid_mlx.server:app` invocation does not use this
  CLI runner. Nonstandard access record formats fail open.
- Next: Atlas reviews/integrates the PR after independent review and CI. No merge,
  release, or production deployment is part of this task. Reply to discussion
  #4784 with release availability only after a release includes the change.
