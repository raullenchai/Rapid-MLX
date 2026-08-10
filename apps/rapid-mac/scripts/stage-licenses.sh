#!/usr/bin/env bash
# stage-licenses.sh — stage third-party Swift license texts into a bundle so the
# notices travel with the binary (#1596).
#
# The shipped .app (and DMG) is the "distribution" that swift-cmark's
# BSD-2-Clause and the linked MIT packages ask their notice to accompany;
# assembling only the repo's THIRD_PARTY.md does not satisfy that. This script
# stages the notices into <out-dir> (build.sh points that at
# Contents/Resources/Licenses/).
#
# It is a standalone script — not an inline block — so the test suite can drive
# it against fixtures and assert both the success staging and the fail-closed
# behavior deterministically, without depending on a resolved checkout cache.
#
# Usage:
#   stage-licenses.sh <package-resolved> <checkouts-dir> <vendor-swiftmath-license> <out-dir>
#
# Fails (non-zero) when a linked package has no license file, when a resolved
# pin has no checkout, when no remote pins parse, or when the vendored notice is
# missing — a package must never ship without its notice.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <package-resolved> <checkouts-dir> <vendor-swiftmath-license> <out-dir>" >&2
    exit 2
fi

RESOLVED="$1"
CHECKOUTS="$2"
VENDOR_SWIFTMATH_LICENSE="$3"
OUT="$4"

mkdir -p "$OUT"

# Locate a license file inside a package directory. Echoes the path on success,
# returns non-zero when none of the conventional names is present.
find_license_file() {
    local dir="$1" name
    for name in LICENSE LICENSE.txt LICENSE.md LICENCE COPYING COPYING.txt \
        COPYRIGHT NOTICE; do
        if [[ -f "$dir/$name" ]]; then
            printf '%s\n' "$dir/$name"
            return 0
        fi
    done
    return 1
}

# Stage one license as ``<label>-<original-filename>.txt`` so provenance is
# obvious in the shipped Licenses/ folder. Hard-fails when the source is missing.
stage_license() {
    local label="$1" src="$2"
    if [[ ! -f "$src" ]]; then
        echo "ERR: license for '$label' not found at: $src" >&2
        echo "     A linked dependency must ship its license text (#1596)." >&2
        exit 1
    fi
    cp "$src" "$OUT/${label}-$(basename "$src").txt"
}

# (a) Vendored Swift source compiled into the binary — the notice lives in-tree
#     (a local-path package, so it is not represented in Package.resolved and
#     must not rely on an ephemeral checkout).
stage_license "SwiftMath" "$VENDOR_SWIFTMATH_LICENSE"

# (b) Remote SPM packages linked into the binary. Drive off the *resolved pins*,
#     not a scan of the checkout cache: that stages exactly the current
#     dependency set, ignores stale checkouts left by removed dependencies, and
#     fails closed when a pin's checkout or license is missing.
if [[ ! -f "$RESOLVED" ]]; then
    echo "ERR: Package.resolved not found: $RESOLVED" >&2
    exit 1
fi
if [[ ! -d "$CHECKOUTS" ]]; then
    echo "ERR: resolved-checkouts directory not found: $CHECKOUTS" >&2
    echo "     Run 'swift build' (or 'swift package resolve') first." >&2
    exit 1
fi

# Each remote pin carries a "location" URL; its on-disk checkout name is that
# URL's basename with a trailing ``.git`` stripped (SPM preserves upstream
# casing, e.g. NetworkImage vs the lowercased identity). Parse with grep/sed so
# the script needs no JSON tooling.
locations="$(sed -nE 's/.*"location"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "$RESOLVED")"
if [[ -z "$locations" ]]; then
    echo "ERR: no remote package pins parsed from $RESOLVED" >&2
    echo "     Expected at least one linked SPM dependency (#1596)." >&2
    exit 1
fi

remote_count=0
while IFS= read -r loc; do
    [[ -n "$loc" ]] || continue
    name="$(basename "$loc")"
    name="${name%.git}"
    dir="$CHECKOUTS/$name"
    if [[ ! -d "$dir" ]]; then
        echo "ERR: resolved package '$name' has no checkout at $dir" >&2
        echo "     Its license cannot be staged; the bundle would ship without" >&2
        echo "     the notice (#1596)." >&2
        exit 1
    fi
    if lic="$(find_license_file "$dir")"; then
        stage_license "$name" "$lic"
        remote_count=$((remote_count + 1))
    else
        echo "ERR: no license file found for Swift package '$name' in $dir" >&2
        echo "     Add its notice or exclude it — a linked dep cannot ship" >&2
        echo "     without its license text (#1596)." >&2
        exit 1
    fi
done <<<"$locations"

echo "staged $((remote_count + 1)) third-party license file(s) into $OUT"
