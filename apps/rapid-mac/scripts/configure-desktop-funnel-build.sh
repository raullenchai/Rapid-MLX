#!/usr/bin/env bash
# Stamp the anonymous first-run funnel's release-only eligibility bit.
# `build.sh` calls this for every packaged app: canonical public releases pass
# 1, while local/ad-hoc, dogfood, CI dry-run, and ordinary developer builds
# pass 0 and have any inherited marker removed.
set -euo pipefail

PLIST="${1:-}"
OFFICIAL_RELEASE="${2:-}"
SIGNING_IDENTITY="${3:--}"
TEAM_ID="${4:-}"
KEY="RapidDesktopFunnelReleaseBuild"

if [[ ! -f "$PLIST" ]]; then
    echo "ERR: desktop funnel build marker requires an Info.plist" >&2
    exit 1
fi
if [[ "$OFFICIAL_RELEASE" != "0" && "$OFFICIAL_RELEASE" != "1" ]]; then
    echo "ERR: desktop funnel release flag must be exactly 0 or 1" >&2
    exit 1
fi

if [[ "$OFFICIAL_RELEASE" == "1" ]]; then
    if [[ "$SIGNING_IDENTITY" == "-" || -z "$SIGNING_IDENTITY" || -z "$TEAM_ID" ]]; then
        echo "ERR: official desktop funnel builds require Developer ID signing and a Team ID" >&2
        exit 1
    fi
    plutil -replace "$KEY" -bool true "$PLIST" 2>/dev/null \
        || plutil -insert "$KEY" -bool true "$PLIST"
else
    plutil -remove "$KEY" "$PLIST" 2>/dev/null || true
fi
