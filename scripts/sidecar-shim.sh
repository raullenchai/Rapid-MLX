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
# Belt-and-braces on top of build-sidecar.sh's pre-compile pass:
# refuse to write any new .pyc at runtime even if a downstream import
# path slips through that the pre-compile didn't cover. Without this,
# ANY post-build .pyc write would break codesign's seal and any
# subsequent `spctl --assess` (Migration Assistant copy, macOS major
# upgrade re-evaluation, fresh quarantine after move/rename) would
# reject the bundle as "a sealed resource is missing or invalid".
# rapid-desktop #230.
export PYTHONDONTWRITEBYTECODE=1

# HuggingFace download path. hf_xet 1.5.1 (the chunked-download client
# huggingface_hub 1.19 selects by default) stalls at ~6 MB transferred
# and emits no error, leaving model downloads frozen at "0 of N files"
# on residential macOS networks. Diagnosed 2026-06-16 against a v0.7.0
# install: raw curl through cas-bridge.xethub.hf.co / us.aws.cdn.hf.co
# pulls 200 MB at a steady 2.3 MB/s; the Python client on the same
# socket hangs zero-progress for >5 min. Disabling Xet routes the
# downloader back onto plain HTTPS range-GETs against the same CDN,
# which works. HF_HUB_DOWNLOAD_TIMEOUT=300 is cheap insurance on the
# slower link. Both honor ${VAR:-default} so power-users can override.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

unset PYTHONSTARTUP

exec "$ROOT/python/bin/python3.12" -s -m vllm_mlx.cli "$@"
