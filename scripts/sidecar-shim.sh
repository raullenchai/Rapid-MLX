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

SELF="$0"
case "$SELF" in
    /*) ;;
    *) SELF="$(pwd)/$SELF" ;;
esac

# Resolve one level of symlink so a user-scope runtime-override
# symlink at ~/Library/Application Support/Rapid/runtime-override/
# bin/rapid-mlx → bundled rapid-mlx still finds the bundled python
# alongside the symlink target.
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
unset PYTHONSTARTUP

exec "$ROOT/python/bin/python3.12" -s -m vllm_mlx.cli "$@"
