#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Keep streamed-installer regressions in the existing offline installer CI loop.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$HERE/test_install_stdin.py" -v
