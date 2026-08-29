#!/usr/bin/env bash
# Remove python-build-standalone's shared libpython only after proving that no
# bundled Mach-O links it. A dependency update must fail the build rather than
# turn this size optimization into a runtime import failure.
set -euo pipefail

STAGE="${1:?usage: prune-unused-libpython.sh <sidecar-stage>}"
LIBPYTHON="$STAGE/python/lib/libpython3.12.dylib"

[[ -f "$LIBPYTHON" ]] || exit 0

FILE_LIST="$(mktemp "${TMPDIR:-/tmp}/rapid-libpython-files.XXXXXX")"
MACHO_LIST="$(mktemp "${TMPDIR:-/tmp}/rapid-libpython-machos.XXXXXX")"
CONSUMERS="$(mktemp "${TMPDIR:-/tmp}/rapid-libpython-consumers.XXXXXX")"
trap 'rm -f "$FILE_LIST" "$MACHO_LIST" "$CONSUMERS"' EXIT

if ! find "$STAGE" -type f -print0 > "$FILE_LIST"; then
    echo "::error::could not enumerate the sidecar while checking libpython consumers" >&2
    exit 1
fi

scanned=0
while IFS= read -r -d '' candidate; do
    [[ "$candidate" == "$LIBPYTHON" ]] && continue
    if ! description="$(file -b "$candidate")"; then
        echo "::error::could not classify $candidate while checking libpython consumers" >&2
        exit 1
    fi
    case "$description" in
        *Mach-O*)
            printf '%s\0' "$candidate" >> "$MACHO_LIST"
            scanned=$((scanned + 1))
            ;;
    esac
done < "$FILE_LIST"

# A production sidecar contains hundreds of Mach-Os. Requiring at least two
# makes an empty, truncated, or misclassified fixture fail closed without
# coupling this guard to the separate release signing baseline.
if [[ "$scanned" -lt 2 ]]; then
    echo "::error::libpython consumer scan found only $scanned bundled Mach-Os; refusing to trim on an incomplete scan" >&2
    exit 1
fi

while IFS= read -r -d '' candidate; do
    if ! dependencies="$(otool -L "$candidate")"; then
        echo "::error::otool failed for $candidate while checking libpython consumers" >&2
        exit 1
    fi
    if grep -q 'libpython3\.12\.dylib' <<< "$dependencies"; then
        printf '%s\n' "$candidate" >> "$CONSUMERS"
    fi
done < "$MACHO_LIST"

if [[ -s "$CONSUMERS" ]]; then
    echo "::error::libpython3.12.dylib is linked by bundled Mach-Os; refusing to drop it:" >&2
    cat "$CONSUMERS" >&2
    exit 1
fi

echo "==> dropping unused libpython3.12.dylib ($(du -m "$LIBPYTHON" | cut -f1) MB)"
rm -f "$LIBPYTHON"
