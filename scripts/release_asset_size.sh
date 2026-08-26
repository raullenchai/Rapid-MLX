#!/usr/bin/env bash
# Resolve one canonical GitHub Release asset size without masking API failures.
# Output is either `present<TAB><positive bytes>` or `absent`. A missing asset
# is a verified state; auth/API/malformed/duplicate/invalid-size responses fail.
set -euo pipefail

TAG="${1:?usage: release_asset_size.sh TAG ASSET_NAME}"
ASSET_NAME="${2:?usage: release_asset_size.sh TAG ASSET_NAME}"
GH_BIN="${GH:-gh}"

if ! ASSETS_JSON="$("$GH_BIN" release view "$TAG" --json assets --jq '.assets')"; then
  echo "release_asset_size: failed to read release $TAG" >&2
  exit 1
fi

printf '%s' "$ASSETS_JSON" | python3 -c '
import json
import sys

name = sys.argv[1]
try:
    assets = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError) as error:
    raise SystemExit(f"release_asset_size: malformed assets JSON: {error}")
if not isinstance(assets, list) or any(not isinstance(asset, dict) for asset in assets):
    raise SystemExit("release_asset_size: assets response is not an array of objects")
matches = [asset for asset in assets if asset.get("name") == name]
if not matches:
    print("absent")
    raise SystemExit(0)
if len(matches) != 1:
    raise SystemExit(f"release_asset_size: expected one {name!r} asset, found {len(matches)}")
size = matches[0].get("size")
if type(size) is not int or size <= 0:
    raise SystemExit(f"release_asset_size: {name!r} has invalid size {size!r}")
print(f"present\t{size}")
' "$ASSET_NAME"
