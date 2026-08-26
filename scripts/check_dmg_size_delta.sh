#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 CURRENT_DMG PREVIOUS_BYTES CAP_MIB" >&2
  exit 2
fi

CURRENT_DMG="$1"
PREVIOUS_BYTES="$2"
CAP_MIB="$3"

[[ -f "$CURRENT_DMG" ]] || { echo "missing DMG: $CURRENT_DMG" >&2; exit 1; }
[[ "$PREVIOUS_BYTES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid previous byte size" >&2; exit 1; }
[[ "$CAP_MIB" =~ ^[0-9]+$ ]] || { echo "invalid MiB delta cap" >&2; exit 1; }

CURRENT_BYTES=$(wc -c < "$CURRENT_DMG" | tr -d '[:space:]')
[[ "$CURRENT_BYTES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid current byte size" >&2; exit 1; }

python3 - "$CURRENT_BYTES" "$PREVIOUS_BYTES" "$CAP_MIB" <<'PY'
import sys

current, previous, cap_mib = map(int, sys.argv[1:])
delta = current - previous
cap_bytes = cap_mib * 1024 * 1024
print(
    f"current_bytes={current} previous_bytes={previous} "
    f"delta_bytes={delta} cap_bytes={cap_bytes}"
)
raise SystemExit(0 if delta <= cap_bytes else 1)
PY
