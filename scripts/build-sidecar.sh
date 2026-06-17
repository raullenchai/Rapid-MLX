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
# Resolve to absolute so paths derived from $OUT_DIR survive the
# `cd "$OUT_DIR"` we do later when invoking tar — otherwise a relative
# `--out build/sidecar-stage` (CI passes this) makes tar look for
# `./build/sidecar-stage/rapid-mlx-sidecar.tar.gz` from inside its own
# target directory and fail with "no such file or directory".
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# Belt-and-suspenders (codex r3 N5): if the absolutise step above ever
# silently produced an empty OUT_DIR (impossible under set -e for the
# realistic failure modes, but the consequence of getting it wrong is
# `rm -rf "/rapid-mlx"` two lines down — same family as the famous
# Steam shell-script bug). Guard explicitly.
if [ -z "$OUT_DIR" ] || [ ! -d "$OUT_DIR" ]; then
    echo "ERR: OUT_DIR resolution produced an invalid path: '$OUT_DIR'" >&2
    exit 1
fi

STAGE="${OUT_DIR}/rapid-mlx"

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

# ----- step 3.5: aggressive trim (post-strip, pre-compileall) ----------
#
# rapid-desktop's `.app` bundle hit 858 MB (machinefi/rapid-desktop#242
# CI gate caps at 500 MB). The sidecar contributes ~498 MB raw; this
# step trims ~80 MB of code we never load at runtime.
#
# Two safe drops:
#   1. transformers/models/*/modeling_*.py — our inference path goes
#      through mlx-lm's own model classes; we never instantiate
#      transformers' AutoModel/PreTrainedModel. The bundle also has
#      no PyTorch (`torch` not installed), so transformers' lazy
#      `from transformers import AutoModel` already returns a
#      "PyTorch was not found" placeholder — third parties calling
#      `AutoModel.from_pretrained(...)` against this sidecar hit
#      the placeholder error path long before any missing-module
#      lookup, so deleting modeling_*.py is a no-op for the
#      already-broken AutoModel surface. We DO still need the
#      tokenizer + config dispatch in transformers/models/auto/, so
#      that path is pruned out of the find.
#   2. image_processing_*.py + feature_extraction_*.py — text-only
#      sidecar; the vision stack ships via the `[vision]` extras
#      which aren't installed here.

echo "==> trimming transformers PyTorch model implementations"
find "$STAGE/site-packages/transformers/models" \
    -path "*/auto/*" -prune -o \
    -name "modeling_*.py" -not -name "modeling_tf_*" -not -name "modeling_flax_*" \
    -print -delete
find "$STAGE/site-packages/transformers/models" \
    -path "*/auto/*" -prune -o \
    \( -name "image_processing_*.py" -o -name "feature_extraction_*.py" \) \
    -print -delete

echo "==> trimming numpy dev/test detritus"
rm -rf \
    "$STAGE/site-packages/numpy/random/_examples" \
    "$STAGE/site-packages/numpy/typing/tests" \
    "$STAGE/site-packages/numpy/f2py/tests" \
    "$STAGE/site-packages/numpy/testing/_private/extbuild.py" \
    "$STAGE/site-packages/numpy/distutils/tests" 2>/dev/null || true

# Pre-compile every .py in the bundled stdlib + site-packages BEFORE
# we codesign. Otherwise CPython's import machinery writes .pyc files
# into __pycache__/ at runtime on first use, those additions are
# unsealed (codesign --verify --deep reports "a sealed resource is
# missing or invalid"), and any `spctl --assess` after first launch
# rejects the bundle:
#
#   * Migration Assistant copy to a new Mac → first launch fails
#     "App is damaged, move to Trash".
#   * macOS major upgrade re-evaluates Gatekeeper → same.
#   * User moves /Applications/Rapid-MLX Desktop.app and back → same.
#
# rapid-desktop issue #230 — confirmed in v0.6.14 with 1008 stray
# .pyc files post-launch. Notarisation is unaffected (the ticket lives
# in xattr metadata, not the sealed resource directory), only the
# spctl-assess re-check path.
#
# SOURCE_DATE_EPOCH freezes the .pyc magic-number timestamp so builds
# stay byte-reproducible — without it every CI re-run produces a
# different bundle and the upstream `MACHO_BASELINE_COUNT` drift
# heuristic becomes meaningless. Falls back to 0 when the build runs
# outside a git checkout (smoke test fixtures).
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$REPO_ROOT" log -1 --format=%ct HEAD 2>/dev/null || echo 0)}"
export SOURCE_DATE_EPOCH
echo "==> pre-compiling .pyc cache (SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH)"
"$STAGE/python/bin/python3.12" -m compileall -q -f -j 0 \
    "$STAGE/python/lib/python3.12" \
    "$STAGE/site-packages" \
    || { echo "ERR: compileall failed; bundle would seal-break on first launch" >&2; exit 1; }

# ----- step 3.6: drop .py sources in site-packages, hoist .pyc ---------
#
# Scope: site-packages only. The bundled stdlib under
# $STAGE/python/lib/python3.12/ is left in source form on purpose — the
# stdlib is only ~30 MB on disk after the step-3 strip, and its
# __pycache__/ entries are already populated by the compileall step
# above so import latency is identical either way.
#
# After compileall, every imported module has a sibling .pyc under
# __pycache__/<name>.<cache-tag>.pyc. Python's regular-package loader
# only treats __pycache__/<name>.<cache-tag>.pyc as a CACHE for an
# adjacent <name>.py; if the .py is missing, that .pyc is ignored
# (verified empirically — websockets.imports, every submodule, broke).
#
# For sourceless loading we hoist the .pyc up to the package directory
# and rename it to plain <name>.pyc — that IS a real module file from
# Python's perspective (see PEP 488 / importlib._bootstrap_external
# SourcelessFileLoader). Then we can safely delete the .py.
#
# IMPORTANT EXCLUSIONS:
#   * __init__.py — we keep the source to guarantee the package is
#     treated as a regular package (sourceless __init__.pyc also works
#     in theory, but keeping the source is ~negligible cost and avoids
#     a class of namespace-package downgrade bugs in tools that probe
#     __file__).
#   * Anything under site-packages/transformers/models/ — transformers
#     scans that subtree at import time (define_import_structure /
#     create_import_structure_from_path) and only recognises .py
#     files when building its lazy-load registry; sourceless .pyc
#     makes the registry empty and the top-level `import transformers`
#     raises `KeyError: frozenset()`. Other transformers subpackages
#     (utils/, generation/, models/auto/) are hoisted normally.
#
# The sidecar-shim exports PYTHONDONTWRITEBYTECODE=1 so Python won't
# attempt to write new .pyc on startup (which would also break the
# codesign seal — see step 3 above).
#
# Caveat: crash tracebacks lose source-line CONTENT for non-__init__
# modules (file:line is still shown, but the actual line of source is
# not). Acceptable for the sidecar — we capture structured logs via
# rapid-mlx telemetry. Set SKIP_SOURCE_DROP=1 to keep .py sources for
# local sidecar debugging.

if [[ "${SKIP_SOURCE_DROP:-0}" != "1" ]]; then
    # Derive the cpython cache tag from the bundled interpreter so a
    # future PBS bump to 3.13 / 3.14 doesn't silently skip this whole
    # step (the .pyc filenames embed the major.minor in the tag —
    # `cpython-312` today, `cpython-313` tomorrow). Falls back to the
    # build-host interpreter if the bundled one can't be invoked, which
    # is fine in practice (PBS_VERSION pins major.minor).
    CACHE_TAG="$("$STAGE/python/bin/python3.12" -c \
        'import sys; print(sys.implementation.cache_tag)' \
        2>/dev/null || echo "cpython-312")"
    PYC_SUFFIX=".${CACHE_TAG}.pyc"
    echo "==> hoisting .pyc out of __pycache__/ and dropping .py sources (cache tag: $CACHE_TAG)"
    # Strategy: walk every __pycache__/ in site-packages, for each
    # <name>.<cache-tag>.pyc whose adjacent <parent>/<name>.py exists,
    # `mv` the .pyc up to <parent>/<name>.pyc and delete the .py.
    # Skip __init__ so packages stay regular packages.
    #
    # EXCLUSION: transformers/models/ is left in source form (see
    # comment block above for why).
    find "$STAGE/site-packages" -type d -name __pycache__ \
        -not -path "*/transformers/models/*" -print | \
    while read -r cachedir; do
        parent="$(dirname "$cachedir")"
        for pyc in "$cachedir"/*"$PYC_SUFFIX"; do
            [[ -f "$pyc" ]] || continue
            base="$(basename "$pyc" "$PYC_SUFFIX")"
            [[ "$base" == "__init__" ]] && continue
            src="$parent/$base.py"
            [[ -f "$src" ]] || continue
            mv -f "$pyc" "$parent/$base.pyc"
            rm -f "$src"
        done
        # Drop the cache dir entirely if it's now empty. Keep a
        # non-empty __pycache__ otherwise so packages that still
        # have remaining .py (e.g. __init__.py) cache normally.
        rmdir "$cachedir" 2>/dev/null || true
    done
else
    echo "==> SKIP_SOURCE_DROP=1, keeping .py sources for sidecar debug"
fi

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

# Sanity guard (codex r1 B2 + r2 N4): a partial pip install can leave us
# with 30-50 Mach-Os instead of 77 and we'd report "drift" pointing the
# operator at re-baselining when the real fix is reading the pip log.
# Anything below half the baseline is almost certainly an install bug,
# not a wheel-set evolution. For very small baselines (test fixtures
# overriding via env) we clamp the floor to baseline-2 so a 5-mach-o
# baseline doesn't end up with a floor of 2 that masks real drops.
if [ "$MACHO_BASELINE_COUNT" -gt 20 ]; then
    MACHO_FLOOR=$(( MACHO_BASELINE_COUNT / 2 ))
else
    MACHO_FLOOR=$(( MACHO_BASELINE_COUNT - 2 ))
fi
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
    # codex r2 N1: use a throwaway HOME for the smoke so the JIT cache
    # mlx writes (under HOME/Library/Caches/mlx) doesn't pollute the
    # caller's real cache during interactive runs. CI has $HOME set;
    # local devs running this script repeatedly should not see their
    # personal mlx cache grow by a few KB each call.
    SMOKE_HOME="$(mktemp -d -t rapid-sidecar-smoke.XXXXXX)"
    trap 'rm -rf "$MACHOS_LIST" "$SMOKE_HOME"' EXIT INT TERM

    SMOKE_OUT="$(env -i HOME="$SMOKE_HOME" PATH=/usr/bin:/bin \
        "$STAGE/bin/rapid-mlx" --version 2>&1)" || {
        echo "ERR: bundle --version failed:" >&2
        echo "$SMOKE_OUT" >&2
        exit 3
    }
    echo "    $SMOKE_OUT"

    # Two-stage mlx smoke. First the import-only check (works in
    # virtualized macos-15 runners that don't expose a Metal GPU);
    # then the Metal JIT eval which we make best-effort because GHA
    # macOS runners are virtualized and may lack working Metal.
    #
    # NOTE: must mirror sidecar-shim.sh env vars (PYTHONHOME +
    # PYTHONPATH + PYTHONNOUSERSITE) — without them the bundled
    # python3.12 can't find `mlx` in site-packages because the install
    # used `pip --target site-packages/` which isn't on the default
    # interpreter path.
    IMPORT_OUT="$(env -i HOME="$SMOKE_HOME" PATH=/usr/bin:/bin \
        PYTHONHOME="$STAGE/python" \
        PYTHONPATH="$STAGE/site-packages" \
        PYTHONNOUSERSITE=1 \
        "$STAGE/python/bin/python3.12" -s -c \
        'import mlx.core as mx; print("mlx", mx.__version__)' 2>&1)" || {
        echo "ERR: bundled mlx import failed (this is a hard bundling bug):" >&2
        echo "$IMPORT_OUT" >&2
        exit 3
    }
    echo "    mlx import: $IMPORT_OUT"

    # codex r3 B1: capture inside an `if` instead of separate
    # `X=$(...); RC=$?` lines — under `set -e`, a command-substitution
    # assignment that exits non-zero aborts the script BEFORE the
    # `$?` capture runs, making the soft-skip branch below dead code.
    # `if X="$(...)" ; then` lets `set -e` see the explicit guard and
    # falls through normally on both success and failure.
    METAL_RC=0
    if METAL_OUT="$(env -i HOME="$SMOKE_HOME" PATH=/usr/bin:/bin \
        PYTHONHOME="$STAGE/python" \
        PYTHONPATH="$STAGE/site-packages" \
        PYTHONNOUSERSITE=1 \
        "$STAGE/python/bin/python3.12" -s -c \
        'import mlx.core as mx; mx.eval(mx.zeros((4,4))); print("ok")' 2>&1)"; then
        METAL_RC=0
    else
        METAL_RC=$?
    fi
    if [ "$METAL_RC" -eq 0 ]; then
        echo "    mlx Metal JIT: OK"
    elif [ -n "${CI:-}" ]; then
        # Best-effort on CI. GitHub-hosted macos-15 runners are
        # virtualized and the Metal device may not be exposed; we
        # don't want to fail the build for a runner-environment
        # constraint. Real Metal exercise happens in rapid-desktop's
        # post-notary smoke (Phase 5).
        echo "    mlx Metal JIT: SKIPPED on CI (rc=$METAL_RC) — $METAL_OUT" >&2
    else
        echo "ERR: bundled mlx Metal JIT failed (local run, this is a real regression):" >&2
        echo "$METAL_OUT" >&2
        exit 3
    fi
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
