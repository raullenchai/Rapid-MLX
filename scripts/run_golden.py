#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch-day functional acceptance against a running /v1/classify server.

Runs serve/golden_requests.jsonl: golden decisions (with confidence floors),
edge rejections (413 oversize / 400 schema / 401 auth). Prints a table and
exits non-zero on any failure. Works against BOTH serving targets:

  python scripts/run_golden.py --base http://localhost:8123 --token $T      # Mac/MLX
  python scripts/run_golden.py --base http://localhost:8123 --token $T      # Vast/proxy (same shape)

Usage in the launch runbook: T-45 functional gate. All rows must pass.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "serve" / "golden_requests.jsonl"


def call(base, token, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/classify", body,
                                 {"Content-Type": "application/json",
                                  "User-Agent": "trio-flash-golden/1.0",
                                  **({"Authorization": f"Bearer {token}"} if token else {})},
                                 method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8123")
    ap.add_argument("--token", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    failures = []
    print(f"{'name':<28} {'status':>6} {'check':<38} {'ms':>6}")
    for row in rows:
        name, typ = row["name"], row["type"]
        req = dict(row.get("request", {}))
        if "pad_to_chars" in req:
            req["prompt"] = (req.pop("prompt") + " pad") * (req.pop("pad_to_chars") // 5)
        status, body, dt = call(args.base, "" if typ == "reject_401" else args.token, req)
        if typ == "golden":
            ok = status == 200 and body.get("decision") == row["expect"]["decision"] \
                and body.get("confidence", 0) >= row["expect"]["confidence_min"]
            check = f"decision={body.get('decision','-')[:24]} conf={body.get('confidence',0):.3f}"
        elif typ == "reject_401":
            ok = status == 401
            check = "401 without token"
        else:
            expect = 413 if typ == "reject_413" else 400
            ok = status == expect
            check = f"expected {expect}"
        print(f"{name:<28} {status:>6} {check:<38} {dt*1000:>5.0f} {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
