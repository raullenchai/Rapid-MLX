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

# Guard the destination before touching the filesystem: this script clears the
# notices it manages, so a stray argument must never let that escape onto an
# arbitrary path. Require a non-empty target whose final component is
# ``Licenses`` (the bundle contract), rejecting root/`.`/mislabeled dirs.
case "$OUT" in
    "" | "/" | ".") echo "ERR: refusing unsafe output path: '$OUT'" >&2; exit 2 ;;
esac
if [[ "$(basename "$OUT")" != "Licenses" ]]; then
    echo "ERR: output dir must be named 'Licenses', got: $OUT" >&2
    exit 2
fi

# Clear only the notices we manage (``*.txt``) rather than ``rm -rf`` the
# directory, so a package removed from Package.resolved leaves no stale notice
# yet no recursive delete is ever issued against the caller's path.
mkdir -p "$OUT"
rm -f "$OUT"/*.txt

# Conventional license / notice filenames. A single package may legitimately
# split its terms across more than one (e.g. an Apache-2.0 LICENSE alongside a
# required NOTICE, or a dual-licensed LICENSE-MIT + LICENSE-APACHE), so every
# one that exists is staged — not just the first.
LICENSE_FILENAMES=(
    LICENSE LICENSE.txt LICENSE.md LICENSE.rst LICENCE
    LICENSE-MIT LICENSE-APACHE LICENSE-BSD
    COPYING COPYING.txt COPYING.md COPYRIGHT
    NOTICE NOTICE.txt NOTICE.md
)

# Stage every conventional notice file found directly in a package directory
# under the given label. Echoes how many it staged; returns non-zero when none
# is present. Symlinks are rejected by stage_license.
stage_package_licenses() {
    local label="$1" dir="$2" name staged=0
    for name in "${LICENSE_FILENAMES[@]}"; do
        # ``-f`` follows symlinks; the ``! -L`` here keeps the loop from feeding
        # a symlinked notice to stage_license (which also refuses it).
        if [[ -f "$dir/$name" && ! -L "$dir/$name" ]]; then
            stage_license "$label" "$dir/$name"
            staged=$((staged + 1))
        fi
    done
    printf '%s\n' "$staged"
    [[ "$staged" -gt 0 ]]
}

# Stage one license as ``<label>-<original-filename>.txt`` so provenance is
# obvious in the shipped Licenses/ folder. Hard-fails when the source is missing
# or is a symlink: a remote package could point its ``LICENSE`` at a CI secret
# (an SSH key, an env file) and ``cp`` would dereference it into the signed app.
# Only a real, regular file is copied.
stage_license() {
    local label="$1" src="$2"
    if [[ ! -f "$src" ]]; then
        echo "ERR: license for '$label' not found at: $src" >&2
        echo "     A linked dependency must ship its license text (#1596)." >&2
        exit 1
    fi
    if [[ -L "$src" ]]; then
        echo "ERR: license for '$label' is a symlink, refusing to copy: $src" >&2
        echo "     A symlinked notice could exfiltrate a file outside the" >&2
        echo "     checkout into the signed bundle (#1596)." >&2
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

# Parse Package.resolved *structurally* with plutil (always present on macOS,
# which is the only platform this .app builds on) rather than a line-oriented
# regex — a compact single-line JSON would defeat the latter. Each remote pin
# carries a "location" URL; its on-disk checkout name is that URL's basename
# with a trailing ``.git`` stripped (SPM preserves upstream casing, e.g.
# NetworkImage vs the lowercased identity).
pin_count="$(plutil -extract pins raw -o - "$RESOLVED" 2>/dev/null || true)"
if ! [[ "$pin_count" =~ ^[0-9]+$ ]] || [[ "$pin_count" -eq 0 ]]; then
    echo "ERR: no remote package pins parsed from $RESOLVED" >&2
    echo "     Expected at least one linked SPM dependency (#1596)." >&2
    exit 1
fi

remote_count=0
for ((i = 0; i < pin_count; i++)); do
    loc="$(plutil -extract "pins.$i.location" raw -o - "$RESOLVED" 2>/dev/null || true)"
    if [[ -z "$loc" ]]; then
        echo "ERR: pin #$i in $RESOLVED has no 'location'; unsupported shape" >&2
        echo "     (registry pins are not handled). Update stage-licenses.sh." >&2
        exit 1
    fi
    name="$(basename "$loc")"
    name="${name%.git}"
    dir="$CHECKOUTS/$name"
    if [[ ! -d "$dir" ]]; then
        echo "ERR: resolved package '$name' has no checkout at $dir" >&2
        echo "     Its license cannot be staged; the bundle would ship without" >&2
        echo "     the notice (#1596)." >&2
        exit 1
    fi
    if staged="$(stage_package_licenses "$name" "$dir")"; then
        remote_count=$((remote_count + staged))
    else
        echo "ERR: no license file found for Swift package '$name' in $dir" >&2
        echo "     Add its notice or exclude it — a linked dep cannot ship" >&2
        echo "     without its license text (#1596)." >&2
        exit 1
    fi
done

echo "staged $((remote_count + 1)) third-party license file(s) into $OUT"
