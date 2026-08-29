#!/usr/bin/env bash
# Report-first shared-host cleanup. Mutations require the explicit --apply flag.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/studio_hygiene.py" "$@"
