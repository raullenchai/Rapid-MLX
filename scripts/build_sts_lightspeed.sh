#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the sts_lightspeed engine + our combat bindings patch (MIT upstream).
# Produces build/slaythespire.cpython-*.so and build/main (console binary).
#
#   bash scripts/build_sts_lightspeed.sh          # uses current venv python
#
# The demos/slay_the_spire and bench/marvins_garden/generate_spire.py expect
# the build at /tmp/sts_lightspeed/build (override with STS_BUILD).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${STS_DIR:-/tmp/sts_lightspeed}"
PIN_PYBIND11=v2.13.6

if [[ ! -d "$DEST/.git" ]]; then
  git clone --depth 1 https://github.com/gamerpuppy/sts_lightspeed "$DEST"
fi
cd "$DEST"
git -C pybind11 fetch -q --tags 2>/dev/null || git submodule update --init --depth 1
git -C pybind11 checkout -q "$PIN_PYBIND11"
git apply "$HERE/patches/sts_lightspeed-bindings.patch"

# Newer CMake refuses the vendored json's ancient minimum version; the flag
# is the supported bypass. Python_EXECUTABLE pins the binding to this venv.
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DPython_EXECUTABLE="$(command -v python)" \
      -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build build -j 8 --target main slaythespire
echo "engine ready: $DEST/build/slaythespire.$(python -c 'import sys;print(f"cpython-{sys.version_info[0]}{sys.version_info[1]}")')-darwin.so"
