#!/bin/sh
#
# sidecar-shim.sh — entrypoint for the bundled rapid-mlx sidecar.
# Installed as $STAGE/bin/rapid-mlx by scripts/build-sidecar.sh.
#
# Job: resolve our own absolute path (even when called via symlink),
# pin PYTHONHOME / PYTHONPATH / PYTHONNOUSERSITE so a host
# `pip install --user mlx==<other>` cannot leak a different mlx.so,
# then exec the bundled python against the rapid-mlx CLI module.
#
# Why a shell shim instead of a Python entrypoint? The .dist-info
# entry_points wrapper that pip generates uses a host-Python shebang,
# which would break the moment the bundle moves machines.
#
# Compatibility notes:
#   * BSD readlink (macOS default) has no -f. We hand-roll the link
#     resolution loop to stay portable.
#   * `-s` is doubly belt-and-suspenders with PYTHONNOUSERSITE — a
#     wrapper that strips the env var would still leave -s active.
#   * `-P` (Python 3.11+) is the canonical fix for `-m` mode prepending
#     the caller's cwd to sys.path[0]. Without it, running the bundled
#     sidecar from a directory containing a sibling `vllm_mlx/` lets
#     that sibling hijack the bundled import path. PYTHONSAFEPATH=1
#     env var below is the static-analysis-friendly belt — a future
#     shim rewrite that drops the -P arg still inherits the safe path
#     via env. See rapid-desktop #361.

SELF="$0"
case "$SELF" in
    /*) ;;
    *) SELF="$(pwd)/$SELF" ;;
esac

# Resolve one level of symlink so a user-scope runtime-override
# symlink at ~/Library/Application Support/Rapid/runtime-override/
# rapid-mlx/bin/rapid-mlx → bundled rapid-mlx still finds the bundled
# python alongside the symlink target. (The ``rapid-mlx/`` wrapper
# is the top-level entry of scripts/build-sidecar-tarball.sh's
# tarball, preserved through extract + atomic publish; fixed in #430.)
if [ -L "$SELF" ]; then
    LINK="$(readlink "$SELF")"
    case "$LINK" in
        /*) SELF="$LINK" ;;
        *) SELF="$(dirname "$SELF")/$LINK" ;;
    esac
fi

BIN_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$BIN_DIR/.." && pwd)"

export PYTHONHOME="$ROOT/python"
export PYTHONPATH="$ROOT/site-packages"
export PYTHONNOUSERSITE=1
# Equivalent of `python -P`: refuse to prepend cwd / script-dir to
# sys.path[0]. Belt-and-braces with the `-P` flag on the exec line
# below — if a future shim rewrite loses the flag, the env var still
# pins the safe path. rapid-desktop #361 cwd-poison hijack closure.
export PYTHONSAFEPATH=1
# Belt-and-braces on top of build-sidecar.sh's pre-compile pass:
# refuse to write any new .pyc at runtime even if a downstream import
# path slips through that the pre-compile didn't cover. Without this,
# ANY post-build .pyc write would break codesign's seal and any
# subsequent `spctl --assess` (Migration Assistant copy, macOS major
# upgrade re-evaluation, fresh quarantine after move/rename) would
# reject the bundle as "a sealed resource is missing or invalid".
# rapid-desktop #230.
export PYTHONDONTWRITEBYTECODE=1
# Desktop spawns the sidecar with stdout/stderr piped (non-TTY), so
# CPython's default block-buffered fd1 holds the R2 puller's per-file
# completion lines until ~4 KB accumulates — which never happens
# during a multi-GB first-time pull where only ~50 bytes/file lands.
# Result: ``DownloadProgress.ingest`` never sees ``[N/M] file R2 (X
# MB)`` and the chat surface sits on ``Spinning up rapid-mlx…`` for
# the entire download. Forcing unbuffered stdout/stderr (or the
# python ``-u`` flag below) makes every print flush immediately so
# the matchers actually fire. v0.7.10 fix for the v0.7.9 regression.
export PYTHONUNBUFFERED=1
if [ -x "$ROOT/bin/ffmpeg" ]; then
    export FFMPEG_BINARY="$ROOT/bin/ffmpeg"
fi
unset PYTHONSTARTUP

exec "$ROOT/python/bin/python3.12" -P -u -s -m vllm_mlx.cli "$@"
