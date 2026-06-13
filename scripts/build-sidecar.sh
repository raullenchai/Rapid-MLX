#!/usr/bin/env bash
#
# build-sidecar.sh — produces the rapid-mlx sidecar artifact that
# rapid-desktop stages under Rapid.app/Contents/Resources/rapid-mlx/.
#
# Codifies the recipe validated by Phase 2 spike on 2026-06-13. See
# docs/sidecar-bundle-build.md for design + measurements.
#
# Usage:
#   scripts/build-sidecar.sh [--out OUT_DIR] [--developer-id ID]
#
#   --out OUT_DIR        Staging directory. Default: build/sidecar-stage
#   --developer-id ID    Apple Developer ID for codesigning. Default: -
#                        (adhoc — for local testing only; CI passes a
#                        real "Developer ID Application: <Team>".)
#   --skip-codesign      Skip the codesign sweep entirely (smoke tests).
#   --skip-verify        Skip the post-build smoke (no system Python
#                        guarantees) — for CI staging steps where the
#                        smoke runs in a separate job.
#
# Outputs:
#   $OUT_DIR/rapid-mlx/               # the bundle root
#   $OUT_DIR/rapid-mlx/bin/rapid-mlx  # entrypoint shim
#   $OUT_DIR/rapid-mlx/python/        # embedded python 3.12
#   $OUT_DIR/rapid-mlx/site-packages/ # rapid-mlx + deps
#   $OUT_DIR/rapid-mlx-sidecar.tar.gz # packaged artifact
#   $OUT_DIR/rapid-mlx-sidecar.sha256 # SHA-256 of the tarball
#
# Exit codes:
#   0 = success
#   1 = generic failure (build step error)
#   2 = Mach-O count mismatch (signing baseline drift)
#   3 = smoke test failure (bundle can't load mlx or import rapid-mlx)

set -euo pipefail

# ----- configuration ---------------------------------------------------

# Pin python-build-standalone to a known-signing-clean release. Bump
# carefully — every tag bump needs a re-run of the Phase 2 spike to
# confirm the .so / .dylib count is unchanged. The hardcoded baseline
# in MACHO_BASELINE_COUNT below depends on this version.
PBS_TAG="${PBS_TAG:-20260610}"
PBS_VERSION="${PBS_VERSION:-3.12.13}"

# How many Mach-Os we expect to sign. A drift here means a new wheel
# added a .so OR a dependency moved a binary, both of which need
# re-baselining. Re-locked on the first authoritative CI run on
# GitHub-hosted macos-15 (run 27472544784). The original Phase 2 spike
# value of 77 was measured on a developer M3 Ultra and included
# build-time artifacts that the strip step removes on a fresh runner
# — 51 is the canonical "what actually ships" number.
MACHO_BASELINE_COUNT="${MACHO_BASELINE_COUNT:-51}"
# Allow modest drift without blocking — wheel updates sometimes shift
# 1-2 .so files. Bigger drift means a new dependency, needs review.
# Widened from 3 → 5 since we're now at a smaller baseline and the
# proportional sensitivity is the same.
MACHO_TOLERANCE="${MACHO_TOLERANCE:-5}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/build/sidecar-stage}"
DEVELOPER_ID="${DEVELOPER_ID:--}"
SKIP_CODESIGN=0
SKIP_VERIFY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT_DIR="$2"; shift 2 ;;
        --developer-id) DEVELOPER_ID="$2"; shift 2 ;;
        --skip-codesign) SKIP_CODESIGN=1; shift ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

STAGE="${OUT_DIR}/rapid-mlx"
ENTITLEMENTS="${REPO_ROOT}/scripts/sidecar-entitlements.plist"

# ----- preflight -------------------------------------------------------

require() {
    command -v "$1" > /dev/null 2>&1 || {
        echo "ERR: missing required binary: $1" >&2
        exit 1
    }
}

require curl
require tar
require python3.12
require codesign
require shasum

if [ ! -f "$ENTITLEMENTS" ] && [ "$SKIP_CODESIGN" != "1" ]; then
    echo "ERR: entitlements file missing at $ENTITLEMENTS" >&2
    exit 1
fi

ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
    echo "ERR: sidecar is arm64-only (mlx requires Apple Silicon); got $ARCH" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin"

# ----- step 1: embedded python interpreter -----------------------------

PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/cpython-${PBS_VERSION}+${PBS_TAG}-aarch64-apple-darwin-install_only_stripped.tar.gz"
PBS_TAR="/tmp/rapid-pbs-${PBS_TAG}.tar.gz"

if [ ! -f "$PBS_TAR" ]; then
    echo "==> downloading python-build-standalone $PBS_VERSION ($PBS_TAG)"
    curl -fsSL --retry 3 -o "$PBS_TAR" "$PBS_URL"
fi
echo "==> extracting python interpreter"
tar -xzf "$PBS_TAR" -C "$STAGE"
test -x "$STAGE/python/bin/python3.12" \
    || { echo "ERR: extracted python is missing executable" >&2; exit 1; }

# ----- step 2: install rapid-mlx + runtime deps ------------------------

echo "==> installing rapid-mlx into site-packages (no [vision] extras)"
# Drive with host python because bundled has ensurepip stripped.
python3.12 -m pip install \
    --target "$STAGE/site-packages" \
    --no-warn-script-location \
    --no-compile \
    --upgrade \
    "$REPO_ROOT"

# ----- step 3: strip dev / unused artifacts ----------------------------

echo "==> stripping dev artifacts"
rm -rf \
    "$STAGE/site-packages/pip" \
    "$STAGE/site-packages/pip-"*.dist-info \
    "$STAGE/site-packages/bin" \
    "$STAGE/site-packages/mlx/include" \
    "$STAGE/site-packages/mlx/lib/cmake"
rm -rf \
    "$STAGE/python/lib/python3.12/ensurepip" \
    "$STAGE/python/lib/python3.12/idlelib" \
    "$STAGE/python/lib/python3.12/turtledemo" \
    "$STAGE/python/lib/python3.12/tkinter" \
    "$STAGE/python/lib/python3.12/test" \
    "$STAGE/python/include" \
    "$STAGE/python/share/man"
find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf {} +

# ----- step 4: shim entrypoint -----------------------------------------

cp "${REPO_ROOT}/scripts/sidecar-shim.sh" "$STAGE/bin/rapid-mlx"
chmod +x "$STAGE/bin/rapid-mlx"

# ----- step 5: count + sign Mach-Os ------------------------------------

echo "==> enumerating Mach-Os"
MACHOS_LIST="$(mktemp)"
# Catch INT/TERM in addition to normal exit so Ctrl-C in interactive
# runs doesn't leak the tmpfile (codex r1 NIT).
trap 'rm -f "$MACHOS_LIST"' EXIT INT TERM
{
    find "$STAGE" -type f \( -name '*.so' -o -name '*.dylib' \)
    echo "$STAGE/python/bin/python3.12"
} > "$MACHOS_LIST"
MACHO_COUNT="$(wc -l < "$MACHOS_LIST" | tr -d ' ')"
echo "    found $MACHO_COUNT Mach-Os (baseline $MACHO_BASELINE_COUNT, tolerance $MACHO_TOLERANCE)"

# Sanity guard (codex r1 B2): a partial pip install can leave us with
# 30-50 Mach-Os instead of 77 and we'd report "drift" pointing the
# operator at re-baselining when the real fix is reading the pip log.
# Anything below half the baseline is almost certainly an install bug,
# not a wheel-set evolution.
MACHO_FLOOR=$(( MACHO_BASELINE_COUNT / 2 ))
if [ "$MACHO_COUNT" -lt "$MACHO_FLOOR" ]; then
    cat >&2 <<EOF
ERR: only $MACHO_COUNT Mach-Os found (< floor $MACHO_FLOOR ≈ half baseline
$MACHO_BASELINE_COUNT). This almost always means pip install above
silently dropped wheels — re-read the pip output, do NOT bump
MACHO_BASELINE_COUNT to paper over this.
EOF
    exit 1
fi

DIFF=$(( MACHO_COUNT - MACHO_BASELINE_COUNT ))
ABS_DIFF=${DIFF#-}
if [ "$ABS_DIFF" -gt "$MACHO_TOLERANCE" ]; then
    cat >&2 <<EOF
ERR: Mach-O count drift ($MACHO_COUNT vs baseline $MACHO_BASELINE_COUNT,
diff $DIFF > tolerance $MACHO_TOLERANCE). A new wheel added or moved a
binary. Re-run Phase 2 spike to confirm signing is still safe, then
bump MACHO_BASELINE_COUNT in this script. See the docs/sidecar-bundle-build.md
'Bump MACHO_BASELINE_COUNT' section.

Full Mach-O list (relative to \$STAGE) for forensic diff:
EOF
    sed "s#$STAGE/##" "$MACHOS_LIST" | sort >&2
    exit 2
fi

if [ "$SKIP_CODESIGN" = "1" ]; then
    echo "==> SKIPPING codesign sweep (--skip-codesign)"
else
    echo "==> codesigning $MACHO_COUNT Mach-Os with identity '$DEVELOPER_ID'"
    while IFS= read -r f; do
        codesign --force --options runtime --timestamp \
            --entitlements "$ENTITLEMENTS" \
            --sign "$DEVELOPER_ID" "$f" \
            > /dev/null 2>&1 || {
                echo "ERR: codesign failed on $f" >&2
                exit 1
            }
    done < "$MACHOS_LIST"
fi

# ----- step 6: smoke test (codex r1 B1: BEFORE packaging) --------------
#
# Order matters: smoke must run BEFORE we package the tarball so a smoke
# failure prevents the Upload artifact / Release upload steps from ever
# seeing a bundle. Running smoke after packaging would still block the
# release (set -e halts the workflow), but you'd waste minutes of CI
# producing an artifact you immediately throw away.

if [ "$SKIP_VERIFY" = "1" ]; then
    echo "==> SKIPPING smoke (--skip-verify)"
else
    echo "==> smoke test (env-stripped, no system Python)"
    SMOKE_OUT="$(env -i HOME="$HOME" PATH=/usr/bin:/bin \
        "$STAGE/bin/rapid-mlx" --version 2>&1)" || {
        echo "ERR: bundle --version failed:" >&2
        echo "$SMOKE_OUT" >&2
        exit 3
    }
    echo "    $SMOKE_OUT"

    env -i HOME="$HOME" PATH=/usr/bin:/bin \
        "$STAGE/python/bin/python3.12" -s -c \
        'import mlx.core as mx; mx.eval(mx.zeros((4,4)))' \
        > /dev/null 2>&1 || {
        echo "ERR: bundled mlx import / Metal JIT failed" >&2
        exit 3
    }
    echo "    mlx import + Metal JIT: OK"
fi

# ----- step 7: package --------------------------------------------------

TARBALL="${OUT_DIR}/rapid-mlx-sidecar.tar.gz"
echo "==> packaging $TARBALL"
( cd "$OUT_DIR" && tar -czf "$TARBALL" rapid-mlx )
shasum -a 256 "$TARBALL" | awk '{print $1}' > "${OUT_DIR}/rapid-mlx-sidecar.sha256"

RAW_SIZE="$(du -sh "$STAGE" | cut -f1)"
TAR_SIZE="$(du -sh "$TARBALL" | cut -f1)"
echo "==> raw bundle:    $RAW_SIZE"
echo "==> tarball:       $TAR_SIZE  ($TARBALL)"
echo "==> sha256:        $(cat "${OUT_DIR}/rapid-mlx-sidecar.sha256")"

echo "==> sidecar build complete"
