#!/usr/bin/env bash
# build-bootstrapper-dmg.sh — produce a slim "bootstrapper-only" DMG
# from an already-built Rapid-MLX Desktop.app, for the bootstrapper
# architecture's P3 cutover (see .claude/loop/bootstrapper-plan.md).
#
# What it does:
#   1. Copies the input .app to a scratch directory (the source .app
#      is never mutated — the full DMG produced by scripts/dmg.sh
#      still ships from the same .app on the same release run).
#   2. Strips Contents/Resources/rapid-mlx/ from the copy. The
#      bootstrapper SwiftUI module (Sources/Rapid/Bootstrapper/) is
#      already wired to download the sidecar from dl.rapidmlx.com at
#      first launch and install it to
#      ~/Library/Application Support/Rapid/runtime-override/, so a
#      stripped .app behaves identically on first launch (splash UI
#      → install pipeline → ChatView) and on subsequent launches
#      (cached runtime-override short-circuits the splash). Verified
#      end-to-end on real network 2026-06-24 (see bootstrapper-plan
#      P2-DONE row).
#   3. Re-codesigns the stripped .app — the strip invalidates the
#      .app's _CodeSignature/CodeResources seal because the sidecar
#      hashes were sealed in.
#   4. Wraps the stripped .app in a DMG using the same layout as
#      scripts/dmg.sh: HFS+, UDZO compression, "Rapid-MLX Desktop"
#      volume name, Applications drop-target symlink, custom install
#      background and left-to-right icon positions.
#   5. Gates the output size (≥ 1 MB so an empty .app fails; ≤ 50 MB
#      so a regression that re-bundles a heavy dep fails — target
#      shape is 5-8 MB).
#   6. Verifies codesign on the .app inside the produced DMG (mount
#      → codesign -v --deep → detach).
#
# Output:
#   build/rapid-mlx-desktop-bootstrapper.dmg
#
# Scope (load-bearing on the release path since v0.8.9 ε.2 cutover):
#   - When NOTARIZE_INLINE_APP=1 is set (CI release path, see
#     .github/workflows/release.yml "Build bootstrapper DMG" step),
#     this script inline-notarises the stripped scratch .app via
#     ditto zip submission BEFORE wrapping it in the DMG. The DMG
#     payload then carries a stapled .app whose ticket survives
#     into the HFS+ xattrs. Local-dev builds without AC_API_*
#     credentials skip the inline-notarise; the resulting DMG is
#     a Quarantine-flagged artifact (testers can use `xattr -d
#     com.apple.quarantine` or codesign --remove-signature locally).
#   - The DMG envelope codesign is intentionally OMITTED on the
#     slim path (v0.8.12 fix for rapid-desktop#427 — see the
#     comment block above the dropped step further down). The
#     canonical full DMG path (scripts/dmg.sh) keeps its envelope
#     codesign because its larger payload doesn't trip the
#     trailer-overlap edge case.
#   - The outer release.yml "Notarise + staple bootstrapper DMG"
#     step submits the wrapped DMG to Apple Notary; on Accepted it
#     staples a DMG-level ticket. With the 2-step UDRW→UDZO pack
#     below (v0.8.12, verify-run 28196766421) Apple accepts both
#     the inline .app submit AND the outer DMG submit — the slim
#     DMG ends up with two tickets (belt-and-braces).
#   - The R2 mirror + GH Release attach are owned by release.yml
#     (slice ε.2 — `Pre-publish slim bootstrapper DMG to
#     dl.rapidmlx.com (R2)` + `Attach bootstrapper DMG to GitHub
#     Release`). Both gates validate the DMG-level ticket via
#     `xcrun stapler validate "$DMG"` and skip the slim leg on a
#     failed staple (canonical-DMG fallback path preserves user-
#     facing UpdateChecker behaviour).
#
# Determinism (NOT an invariant):
#   Repeated runs of this script over the same input .app DO NOT
#   produce byte-identical DMGs. Two sources of non-determinism:
#     1. hdiutil create writes a fresh HFS+ image with fresh
#        per-volume metadata (UUIDs, creation timestamps).
#     2. The Developer ID re-codesign branch on the scratch .app
#        passes ``--timestamp``, which embeds a signed TSA (RFC 3161
#        timestamp authority) stamp from Apple's timestamp server.
#        That stamp moves second-to-second. (The ad-hoc local-dev
#        branch does NOT pass --timestamp, so local rebuilds are
#        closer to byte-stable but still diverge on point 1.)
#   latest.json publishes the DMG's sha256 so byte-stability is
#   not required — the manifest just records whatever bytes shipped
#   from this run. If a future caller needs byte-deterministic DMGs
#   (e.g. a content-addressed update path), the levers are:
#   SOURCE_DATE_EPOCH for mtimes, drop ``--timestamp`` on the inner
#   .app re-codesign, pin volume name + UDIF block size for hdiutil.
#
# Usage:
#   bash scripts/build-bootstrapper-dmg.sh [/path/to/Rapid-MLX Desktop.app]
#   (defaults to ${ROOT}/build/Rapid-MLX Desktop.app)
#
# Env:
#   CODESIGN_IDENTITY  — same semantics as scripts/build.sh and
#                        scripts/dmg.sh: a Developer ID common-name or
#                        SHA-1 hash for CI, or "-" (default) for ad-hoc
#                        local builds.
#   BOOTSTRAPPER_DMG_MIN_MB  — override lower size gate (default 1)
#   BOOTSTRAPPER_DMG_MAX_MB  — override upper size gate (default 50)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
INPUT_APP="${1:-$BUILD/Rapid-MLX Desktop.app}"

# Output paths. The DMG name embeds "bootstrapper" so it can never be
# confused with the canonical $BUILD/rapid-mlx-desktop.dmg that the
# main release ships (and that scripts/dmg.sh produces). The scratch
# .app lives under a uniquely-named subdir so re-runs don't collide
# and so the cleanup trap can blow it away safely.
# Lowercase + hyphenated to match the canonical ``rapid-mlx-desktop.dmg``
# naming and to dodge a class of pipeline tools (Apple's notary
# extractor was the trip-wire on run 28186939707) that mis-handle
# DMG paths containing spaces. See rapid-desktop#427 fix candidate
# #2.
DMG="$BUILD/rapid-mlx-desktop-bootstrapper.dmg"
SCRATCH="$BUILD/bootstrapper-dmg-scratch"
SCRATCH_APP="$SCRATCH/Rapid-MLX Desktop.app"
STAGING="$BUILD/bootstrapper-dmg-staging"
VOL_NAME="Rapid-MLX Desktop"

# Size gates — defaults sized for the bootstrapper architecture's
# plan (target 5-8 MB; 50 MB headroom catches a gross regression
# like a future maintainer accidentally re-bundling site-packages).
# The lower gate catches an empty / stub .app.
#
# Gates are evaluated against BYTES (not du -sm output) — see the
# size-check block at the bottom of the script. Reason: macOS's
# `du -sm` reports whole-MiB disk-usage units rounded UP, so a 100 KB
# DMG can report `1` and silently pass a `>= 1 MB` lower gate. Byte-
# precise comparison (1 MiB = 1048576 bytes; 50 MiB = 52428800 bytes)
# is the correct floor/ceiling. The MB env vars are kept for human
# readability of the override interface and for the log line; the
# actual gate keys off the precise byte derivation below.
MIN_MB="${BOOTSTRAPPER_DMG_MIN_MB:-1}"
MAX_MB="${BOOTSTRAPPER_DMG_MAX_MB:-50}"

# Sanity: don't accept a 0 / negative / non-int range. Without this
# a typo'd env override could silently disable a gate.
# Upper bound on the MB envs (1 TB = 1_048_576 MB) is far above any
# plausible DMG and well below bash's signed-64-bit arithmetic limit
# (2^62 / 1_048_576 ≈ 4.4e12 MB), so the `MIN_MB * 1048576`
# multiplication below cannot overflow. Codex r2 MINOR.
GATE_MB_MAX=1048576  # 1 TB ceiling on the env override itself
if ! [[ "$MIN_MB" =~ ^[1-9][0-9]*$ ]]; then
    echo "==> ERR: BOOTSTRAPPER_DMG_MIN_MB must be a positive integer (got: '$MIN_MB')" >&2
    exit 1
fi
if ! [[ "$MAX_MB" =~ ^[1-9][0-9]*$ ]]; then
    echo "==> ERR: BOOTSTRAPPER_DMG_MAX_MB must be a positive integer (got: '$MAX_MB')" >&2
    exit 1
fi
if [[ "$MIN_MB" -gt "$GATE_MB_MAX" ]]; then
    echo "==> ERR: BOOTSTRAPPER_DMG_MIN_MB ($MIN_MB) exceeds the sanity ceiling ${GATE_MB_MAX} (1 TB). If you actually need a >1 TB floor, edit the script." >&2
    exit 1
fi
if [[ "$MAX_MB" -gt "$GATE_MB_MAX" ]]; then
    echo "==> ERR: BOOTSTRAPPER_DMG_MAX_MB ($MAX_MB) exceeds the sanity ceiling ${GATE_MB_MAX} (1 TB). If you actually need a >1 TB ceiling, edit the script." >&2
    exit 1
fi
if [[ "$MIN_MB" -ge "$MAX_MB" ]]; then
    echo "==> ERR: BOOTSTRAPPER_DMG_MIN_MB ($MIN_MB) must be < BOOTSTRAPPER_DMG_MAX_MB ($MAX_MB)" >&2
    exit 1
fi
# Convert to byte thresholds for the precise gate at the bottom of
# the script. 1 MiB = 1048576 bytes (we use MiB == "MB" here to
# match the du -sm reporting convention; the boundary semantics are
# the load-bearing bit). Overflow ruled out by the GATE_MB_MAX
# sanity check above.
MIN_BYTES=$(( MIN_MB * 1048576 ))
MAX_BYTES=$(( MAX_MB * 1048576 ))

if [[ ! -d "$INPUT_APP" ]]; then
    echo "==> ERR: input .app not found: $INPUT_APP" >&2
    echo "    Run 'bash scripts/build.sh' (or build via CI) first." >&2
    exit 1
fi

# Always blow away the scratch + staging dirs on exit. Without the
# trap, a mid-flight failure leaves a half-stripped .app on disk
# that a later run could try to wrap as if it were the main release
# bundle.
cleanup() {
    rm -rf "$SCRATCH" "$STAGING"
}
trap cleanup EXIT

echo "==> staging scratch copy at $SCRATCH"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
# -R preserves symlinks + executable bits; -p preserves mtimes. The
# source .app stays byte-identical on disk so the parallel
# scripts/dmg.sh / scripts/build-sidecar-tarball.sh steps in
# release.yml continue to see the same bytes (no race, no double-
# carve).
cp -Rp "$INPUT_APP" "$SCRATCH_APP"

# Strip the sidecar tree. The bootstrapper publishes a tree at
# ~/Library/Application Support/Rapid/runtime-override/rapid-mlx/
# at first launch (see Sources/Rapid/Bootstrapper/BootstrapInstaller
# .swift), so the .app's own Contents/Resources/rapid-mlx/ slot is
# allowed to be empty for the slim build. The ``-f`` is intentional
# — if a future build skips bundling for some reason and the dir
# never existed, the slim DMG build should still succeed (the strip
# is the goal, not the precondition).
SIDECAR_INSIDE_SCRATCH="$SCRATCH_APP/Contents/Resources/rapid-mlx"
if [[ -d "$SIDECAR_INSIDE_SCRATCH" ]]; then
    echo "==> stripping Contents/Resources/rapid-mlx/ from scratch copy"
    rm -rf "$SIDECAR_INSIDE_SCRATCH"
else
    echo "==> no Contents/Resources/rapid-mlx/ in input .app — nothing to strip"
fi
# Defensive post-condition: the path MUST be gone for the bootstrapper
# launch contract to hold (ServerLocator.swift's .bundled probe checks
# this exact path; a stale tree would short-circuit the splash flow).
if [[ -e "$SIDECAR_INSIDE_SCRATCH" ]]; then
    echo "==> ERR: sidecar tree still present after strip: $SIDECAR_INSIDE_SCRATCH" >&2
    exit 1
fi

# Strip extended attributes (especially ``com.apple.notary.ticket``
# inherited from the canonical .app's prior notarisation+staple) so
# Apple's notary extractor sees a clean bundle on the slim DMG path.
# Without this, the scratch .app carries a stale staple ticket whose
# CodeDirectory hash refers to the pre-strip bundle layout — Apple
# Notary rejects with "could not be extracted" + "no signed
# executables or bundles" (rapid-desktop#427 root cause confirmed
# via run 28187803479 notary log).
#
# ``xattr -cr`` clears all xattrs recursively. Cheap (<1s on a 400MB
# bundle) and safe — xattrs are metadata only, never code.
echo "==> stripping inherited xattrs (esp. notary ticket) from scratch .app"
xattr -cr "$SCRATCH_APP"

# Stripping a sealed bundle leaves _CodeSignature/CodeResources
# referencing hashes of the deleted files — `codesign -v` fails with
# "resource added that did not exist at signing time" or similar. The
# fix is to re-sign.
#
# Use the SAME signing flags as scripts/build.sh's Developer ID
# branch (lines 311-323 — "Apple discourages --deep for distribution
# signing"). v0.8.11 added ``--deep`` defensively; the rapid-desktop
# .app has no nested signed Mach-Os post-strip (Contents/Resources/
# rapid-mlx/ was the only nested signed tree, and the strip removes
# it), so ``--deep`` was a no-op at best and a gratuitous departure
# from the canonical path at worst. Drop it for symmetry with the
# canonical path that already notarises cleanly.
#
# Ad-hoc by default, Developer ID when CODESIGN_IDENTITY is set,
# with --options runtime + --timestamp + entitlements on the CI
# release path to match build.sh / dmg.sh.
SIGN_IDENTITY="${CODESIGN_IDENTITY:--}"
ENTITLEMENTS="$ROOT/Resources/Rapid.entitlements"
if [[ "$SIGN_IDENTITY" == "-" ]]; then
    echo "==> ad-hoc re-codesign scratch .app"
    codesign --force --sign - "$SCRATCH_APP"
else
    echo "==> Developer ID re-codesign scratch .app ($SIGN_IDENTITY)"
    if [[ ! -f "$ENTITLEMENTS" ]]; then
        echo "==> ERR: entitlements file missing: $ENTITLEMENTS" >&2
        exit 1
    fi
    codesign --force --options runtime --timestamp \
        --entitlements "$ENTITLEMENTS" \
        --sign "$SIGN_IDENTITY" "$SCRATCH_APP"
fi
# Verify the re-sign is well-formed BEFORE we wrap it. A bad
# signature inside a DMG would only surface on the end-user's Mac.
codesign --verify --strict "$SCRATCH_APP"

# v0.8.12 belt-and-braces: also inline-notarise the .app via zip
# submission BEFORE wrapping in the DMG. The LOAD-BEARING fix for
# rapid-desktop#427 is the 2-step UDRW → UDZO pack below (see the
# "DMG packing" comment block) — Apple's DMG-extractor was
# rejecting genuinely corrupt bytes from the 1-step UDZO codepath,
# not refusing on policy. Inline-notarising the .app costs ~30s of
# Apple-server time per release and gives the released slim DMG a
# second stapled ticket (the inner .app's) on top of the DMG-level
# ticket the outer release.yml notarise step installs. Either
# ticket alone satisfies Gatekeeper; both is belt-and-braces.
#
# Why a zip and not the DMG: notarize.sh's own header documents
# that notarytool accepts ``.zip / .dmg / .pkg`` and that bare .app
# notarisation requires zipping first (``ditto -c -k --keepParent
# App.app App.zip``). Submitting the .app via zip exercises Apple's
# zip-extraction path, which is a documented alternate to
# DMG-extraction. We pair this with the defensive
# ``xcrun stapler validate`` post-check below so a notary/staple
# failure aborts the build instead of silently wrapping an
# un-stapled .app into the DMG.
#
# The outer "Notarise + staple bootstrapper DMG" step in
# release.yml ALSO submits the wrapped DMG and (with the
# 2-step UDRW→UDZO pack below — verify-run 28196766421)
# Apple accepts; stapler attaches a DMG-level ticket on top
# of the inner .app ticket installed here. The released slim
# DMG thus ships with two tickets (belt-and-braces — either
# alone satisfies Gatekeeper).
#
# Gated by NOTARIZE_INLINE_APP=1 so the local-dev build path (no
# AC_API_KEY_*) stays untouched.
if [[ "${NOTARIZE_INLINE_APP:-0}" == "1" ]]; then
    if [[ -z "${AC_API_KEY_ID:-}" || -z "${AC_API_ISSUER_ID:-}" || -z "${AC_API_KEY_PATH:-}" || ! -f "${AC_API_KEY_PATH:-/dev/null}" ]]; then
        echo "==> NOTARIZE_INLINE_APP=1 but AC_API_* credentials missing — skipping inline app notarise (will rely on outer DMG notarise step)" >&2
    else
        echo "==> inline-notarise scratch .app via zip submission (rapid-desktop#427 Candidate D)"
        APP_ZIP="$SCRATCH/rapid-mlx-desktop-bootstrapper-app.zip"
        # ditto -c -k --keepParent preserves bundle structure +
        # xattrs and is the form notarytool's docs prescribe for
        # .app submission. Output to SCRATCH so cleanup() removes it
        # on script exit.
        /usr/bin/ditto -c -k --keepParent "$SCRATCH_APP" "$APP_ZIP"
        # notarize.sh handles submit-with-status-parse, log-on-failure,
        # and `stapler staple` of the staple-target (the .app, not the
        # zip — stapler can't staple zip files per notarize.sh's
        # header comment). On success SCRATCH_APP carries
        # ``com.apple.notary.ticket``.
        bash "$ROOT/scripts/notarize.sh" "$APP_ZIP" "$SCRATCH_APP"
        # Defensive post-condition: the staple must be readable.
        # `stapler validate` exits non-zero if the ticket is missing
        # or doesn't match the bundle's CodeDirectory — either would
        # break the user-facing Gatekeeper read.
        if ! xcrun stapler validate "$SCRATCH_APP" >/dev/null 2>&1; then
            echo "==> ERR: stapler validate failed on $SCRATCH_APP after inline notarise — refusing to wrap an unverified .app into the DMG" >&2
            exit 1
        fi
        echo "==> stapler validate OK on scratch .app — ticket persisted"
    fi
fi

# DMG staging mirrors scripts/dmg.sh exactly so the slim DMG has the
# same Finder shape (.app + Applications drop-target) end users
# already recognise.
echo "==> staging DMG layout at $STAGING"
rm -rf "$STAGING"
mkdir -p "$STAGING"
# Use plain ``cp -R`` (NOT ``cp -Rp``) to match scripts/dmg.sh's
# canonical-DMG path which is known to roundtrip a stapled .app
# through HFS+ + UDZO without corruption. macOS ``cp -R`` preserves
# extended attributes by default (incl. the inline-notarise step's
# ``com.apple.notary.ticket`` xattr that staples the .app), so the
# DMG payload still carries the notary ticket. ``cp -Rp`` ALSO
# preserves file flags + ACLs; verify-run 28194821136 proved that
# ``cp -Rp`` of a stapled .app deterministically produces a UDZO
# DMG that ``hdiutil verify`` rejects as "corrupt image" even
# though the bytes look structurally sound (koly trailer + XML
# offsets check out — see PR #429 comment thread). Switching to
# ``cp -R`` is the minimum change to align with the
# known-good canonical path.
cp -R "$SCRATCH_APP" "$STAGING/Rapid-MLX Desktop.app"
ln -s /Applications "$STAGING/Applications"

# DMG packing — 2-step UDRW → UDZO. v0.8.10 / v0.8.11 / v0.8.12-{a,b,c}
# all tried the 1-step ``hdiutil create -srcfolder -fs HFS+ -format
# UDZO`` form. On a stapled slim .app (strip + re-codesign + inline-
# notarise + stapler staple → ~5 MB) it deterministically produces a
# UDIF whose koly trailer is intact but whose internal BLKX (compressed
# block) table is unreadable: ``hdiutil verify`` / ``hdiutil imageinfo``
# fail with "corrupt image", Apple Notary then rejects with
# "could not be extracted" (because the bytes really ARE corrupt — not
# a server-side policy as v0.8.12-a/b/c CHANGELOG/PR commentary
# initially claimed). The canonical full DMG (scripts/dmg.sh) escapes
# this because its 156 MB payload doesn't trip the same UDZO codepath.
#
# The 2-step pattern (industry standard — ``dmgbuild``, ``create-dmg``,
# Apple TN3119) creates a UDRW (uncompressed read-write) image first,
# mounts it, copies the staged tree into the mount, detaches, then
# converts UDRW → UDZO. This isolates the staged-tree-to-HFS+ layout
# step from the UDZO compression step, sidestepping the 1-step
# UDZO codepath that fails on stapled .app metadata. Verify-run
# 28195966086 proved that the 1-step UDZO form fails on a stapled
# .app REGARDLESS of envelope codesign / cp -Rp vs cp -R / --force.
#
# Sizing: ``hdiutil makehybrid`` could autosize but we use the simpler
# ``hdiutil create -size`` route. The slim .app is ~5 MB, scratch
# leeway 64 MB covers staging + HFS+ catalog overhead. Compression
# shrinks the final UDZO back to the same ~5–6 MB the v0.8.10 1-step
# path produced.
echo "==> hdiutil create UDRW scratch image (v0.8.12 — 2-step UDZO pack)"
UDRW="$BUILD/rapid-mlx-desktop-bootstrapper.udrw.dmg"
rm -f "$UDRW" "$DMG"
# NOTE: ``-format UDRW`` requires ``-srcfolder`` (hdiutil rejects
# -format UDRW for empty-image creation with "-format requires
# -srcfolder or -srcdevice"). Omitting -format lets hdiutil default
# to UDRW for sizespec-based creates, which is what we want.
hdiutil create \
    -size 64m \
    -fs HFS+ \
    -volname "$VOL_NAME" \
    -ov \
    "$UDRW" \
    >/dev/null

echo "==> mount UDRW scratch + copy staged tree"
SCRATCH_MOUNT="$(mktemp -d "${TMPDIR:-/tmp}/bootstrapper-dmg-mount-XXXXXX")"
hdiutil attach "$UDRW" -nobrowse -mountpoint "$SCRATCH_MOUNT" -quiet
# trap to detach on early exit
trap_cleanup_mount() {
    if mount | grep -q "$SCRATCH_MOUNT"; then
        hdiutil detach "$SCRATCH_MOUNT" -quiet 2>/dev/null \
            || hdiutil detach "$SCRATCH_MOUNT" -force -quiet 2>/dev/null \
            || true
    fi
    rmdir "$SCRATCH_MOUNT" 2>/dev/null || true
}
trap trap_cleanup_mount EXIT
# cp -R preserves the inline-notarise stapler ticket via macOS's
# default xattr-preserve behaviour. Symlink to /Applications mirrors
# scripts/dmg.sh's drop-target convention.
cp -R "$STAGING/Rapid-MLX Desktop.app" "$SCRATCH_MOUNT/Rapid-MLX Desktop.app"
ln -s /Applications "$SCRATCH_MOUNT/Applications"
bash "$ROOT/scripts/configure-dmg-layout.sh" "$SCRATCH_MOUNT"

echo "==> detach UDRW scratch"
hdiutil detach "$SCRATCH_MOUNT" -quiet \
    || hdiutil detach "$SCRATCH_MOUNT" -force -quiet
rmdir "$SCRATCH_MOUNT" 2>/dev/null || true
trap - EXIT

echo "==> hdiutil convert UDRW → UDZO ($DMG)"
hdiutil convert "$UDRW" -format UDZO -ov -o "$DMG" >/dev/null
rm -f "$UDRW"

# v0.8.12 envelope-codesign DROP rationale: scripts/dmg.sh (canonical)
# DOES envelope-codesign its UDZO because the ~156 MB payload absorbs
# the trailing CMS blob without corrupting the koly trailer. On a
# ~5 MB slim payload the CMS append OVERLAPS the koly trailer and
# deterministically corrupts the image. We don't envelope-codesign the
# slim DMG; the outer release.yml "Notarise + staple bootstrapper DMG"
# step (with NOTARYTOOL_FORCE=1 to bypass notarytool's local validator,
# which gates on envelope codesign presence) issues the DMG-level
# ticket. Apple's server accepts the unsigned-but-cleanly-packed bytes
# (verify-run 28196766421).
echo "==> hdiutil verify (CRC + structure)"
if ! hdiutil verify "$DMG" >/dev/null 2>&1; then
    echo "==> ERR: hdiutil verify failed for $DMG — last-attempt diagnostic:" >&2
    hdiutil verify "$DMG" >&2 || true
    exit 1
fi

# Size gates. Run AFTER the DMG is produced so we measure the
# compressed shipping artifact, not the staging tree. The gate keys
# off PRECISE BYTES (``stat -f%z``) rather than du -sm: macOS du
# reports whole-MiB disk-usage rounded UP, so a 100 KB DMG can
# report `1` and silently pass a ``>= 1 MB`` lower gate. Comparing
# bytes against MIN_BYTES / MAX_BYTES (derived above) gives a hard
# floor/ceiling regardless of FS block size. The ``du -sm`` value
# stays for the human-readable log line.
DMG_BYTES="$(stat -f%z "$DMG" 2>/dev/null || stat -c%s "$DMG")"
DMG_MB="$(du -sm "$DMG" | awk '{print $1}')"
echo "==> bootstrapper DMG size: ${DMG_MB} MB (${DMG_BYTES} bytes)"

if [[ "$DMG_BYTES" -lt "$MIN_BYTES" ]]; then
    echo "==> ERR: bootstrapper DMG is ${DMG_BYTES} bytes (${DMG_MB} MB) which is below the ${MIN_MB} MB (${MIN_BYTES} bytes) floor." >&2
    echo "    This usually means the stripped .app is empty (no Mach-O / no resources)." >&2
    echo "    Inspect the input .app at: $INPUT_APP" >&2
    exit 1
fi
if [[ "$DMG_BYTES" -gt "$MAX_BYTES" ]]; then
    echo "==> ERR: bootstrapper DMG is ${DMG_BYTES} bytes (${DMG_MB} MB) which exceeds the ${MAX_MB} MB (${MAX_BYTES} bytes) ceiling." >&2
    echo "    Target shape per .claude/loop/bootstrapper-plan.md is 5-8 MB." >&2
    echo "    A future regression that re-bundles a heavy dep into the .app would land here;" >&2
    echo "    inspect Contents/Resources/ inside the input .app and ensure only the SwiftUI" >&2
    echo "    binary + flat asset files (cheetah PNGs / Localizable / benchmark-scores.json)" >&2
    echo "    remain. Override with BOOTSTRAPPER_DMG_MAX_MB if the jump is intentional." >&2
    exit 1
fi

# Verify codesign on the .app inside the produced DMG by mounting it
# read-only at its normal /Volumes path and running codesign -v --deep. This
# catches any mid-flight corruption between the scratch sign step and
# the hdiutil pack. A random mountpoint makes Finder cache that directory name
# as the volume identity and can poison the immediate final presentation check.
# Mirrors scripts/validate-dmg.sh's device-based cleanup pattern.
echo "==> mounting $DMG for codesign verification"
MOUNT=""
VERIFY_DEVICE=""
ATTACHED=0
verify_cleanup() {
    if [[ "$ATTACHED" -eq 1 ]]; then
        local detach_target="${VERIFY_DEVICE:-$MOUNT}"
        if [[ -n "$detach_target" ]]; then
            hdiutil detach "$detach_target" -quiet \
                || hdiutil detach "$detach_target" -force -quiet \
                || true
        fi
    fi
    # also run the original cleanup
    rm -rf "$SCRATCH" "$STAGING"
}
trap verify_cleanup EXIT
VERIFY_ATTACH_OUTPUT="$(hdiutil attach "$DMG" -nobrowse -readonly)"
ATTACHED=1
VERIFY_DEVICE="$(printf '%s\n' "$VERIFY_ATTACH_OUTPUT" | awk '$1 ~ /^\/dev\// { print $1; exit }')"
MOUNT="$(printf '%s\n' "$VERIFY_ATTACH_OUTPUT" | awk -F '\t' 'NF >= 3 && $3 != "" { print $3 }' | tail -1)"
if [[ -z "$MOUNT" || ! -d "$MOUNT" ]]; then
    echo "$VERIFY_ATTACH_OUTPUT" >&2
    echo "==> ERR: could not determine mounted bootstrapper DMG path" >&2
    exit 1
fi

MOUNTED_APP="$MOUNT/Rapid-MLX Desktop.app"
if [[ ! -d "$MOUNTED_APP" ]]; then
    echo "==> ERR: Rapid-MLX Desktop.app not found inside mounted DMG at $MOUNT" >&2
    exit 1
fi
codesign --verify --deep --strict "$MOUNTED_APP"
echo "==> codesign verification: OK"

# Sanity post-condition: the sidecar tree must NOT exist inside the
# produced DMG's .app — duplicates the strip-time check at the
# mount-time stage so a future bug that re-stages the sidecar back in
# fails here even if the strip itself succeeded.
if [[ -e "$MOUNTED_APP/Contents/Resources/rapid-mlx" ]]; then
    echo "==> ERR: stripped sidecar tree is somehow present in mounted DMG:" >&2
    echo "    $MOUNTED_APP/Contents/Resources/rapid-mlx" >&2
    exit 1
fi

# Detach the codesign inspection mount, then validate the final compressed DMG
# through the same cold-mount Finder path used by the canonical artifact. This
# catches presentation metadata lost during UDRW -> UDZO conversion.
echo "==> detach codesign verification mount"
hdiutil detach "${VERIFY_DEVICE:-$MOUNT}" -quiet \
    || hdiutil detach "${VERIFY_DEVICE:-$MOUNT}" -force -quiet
ATTACHED=0
VERIFY_DEVICE=""
MOUNT=""

echo "==> validate final bootstrapper DMG presentation"
bash "$ROOT/scripts/validate-dmg.sh" "$DMG"

echo
echo "bootstrapper DMG ready at: $DMG"
echo "  size:        ${DMG_MB} MB (${DMG_BYTES} bytes)"
echo "  inner .app:  re-signed by $SIGN_IDENTITY (DMG envelope intentionally unsigned for the slim path — see v0.8.12 root-cause comment above)"
echo "  contents:    Rapid-MLX Desktop.app (sidecar stripped — bootstrapper installs at runtime)"
echo
echo "Inner .app inline-notarisation: ${NOTARIZE_INLINE_APP:-0}=1 means the .app inside this DMG is already stapled."
echo "Outer DMG notarisation runs in the release.yml \"Notarise + staple bootstrapper DMG\" step."
echo "User-facing latest.json points at this DMG since v0.8.9 slice ε.2 (canonical fallback if the slim publish gate fails)."

# CI-friendly outputs — mirror the pattern in scripts/build-sidecar-tarball.sh
# so a future job that needs to wire latest.json / R2 keys against
# this artifact has the exact paths surfaced.
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
        echo "dmg_path=${DMG}"
        echo "dmg_name=$(basename "$DMG")"
        echo "dmg_size_bytes=${DMG_BYTES}"
        echo "dmg_size_mb=${DMG_MB}"
    } >> "$GITHUB_OUTPUT"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Bootstrapper DMG (slice α — artifact only)"
        echo ""
        echo "| field | value |"
        echo "|-------|-------|"
        echo "| dmg | \`$(basename "$DMG")\` |"
        echo "| size | ${DMG_MB} MB |"
        echo "| bytes | ${DMG_BYTES} |"
        echo "| signed_by | \`${SIGN_IDENTITY}\` |"
        echo "| codesign_verify | OK |"
        echo "| stripped_path | \`Contents/Resources/rapid-mlx/\` |"
        echo ""
        echo "Not yet notarised, not yet mirrored, not a release asset."
        echo "Slice ε cutover will promote this shape to the main DMG."
    } >> "$GITHUB_STEP_SUMMARY"
fi
