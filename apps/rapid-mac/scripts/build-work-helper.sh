#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTENTS="$1"
PHASE="${2:-all}"
[[ "$PHASE" == stage || "$PHASE" == dependencies || "$PHASE" == all ]] || { echo "invalid helper build phase" >&2; exit 1; }
if [[ "$PHASE" != dependencies ]]; then
HELPERS="$CONTENTS/Resources/Helpers"
mkdir -p "$HELPERS/local-documents" "$CONTENTS/Resources/Licenses"
cp "$ROOT/Work/local-documents/SKILL.md" "$HELPERS/local-documents/SKILL.md"
ARCHIVE="$ROOT/build/goose-aarch64-apple-darwin-v1.54.0.tar.gz"
EXPECTED=feda29744cd4e13f9a5211a968ed47c67fb8fa79647f4d271d92d564998a393c
if [[ ! -f "$ARCHIVE" ]]; then
    curl -fL --retry 3 -o "$ARCHIVE" \
        'https://github.com/aaif-goose/goose/releases/download/v1.54.0/goose-aarch64-apple-darwin.tar.gz'
fi
ACTUAL="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo "goose archive SHA-256 mismatch" >&2; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$ARCHIVE" -C "$TMP"
GOOSE_BIN="$(find "$TMP" -type f -name goose -print -quit)"
[[ -n "$GOOSE_BIN" ]] || { echo "goose executable missing" >&2; exit 1; }
cp "$GOOSE_BIN" "$HELPERS/goose"
chmod 755 "$HELPERS/goose"
cp "$ROOT/Work/work-inference-proxy.py" "$HELPERS/work-inference-proxy.py"
cp "$ROOT/Work/work-web-search-mcp.py" "$HELPERS/work-web-search-mcp.py"
cp "$ROOT/../../rapid_mlx/goose_runtime/mini-prefix.json" "$HELPERS/goose-mini-prefix.json"
cp "$ROOT/Work/goose-prefix.json" "$HELPERS/goose-prefix.json"
cp "$ROOT/Work/work-helper-watchdog.py" "$HELPERS/work-helper-watchdog.py"
# Goose exposes the GOOSE_SHELL basename in the model's shell tool schema.
# Keep stock Goose's `bash` description while running the credential scrubber.
cp "$ROOT/Work/work-helper-watchdog.py" "$HELPERS/bash"
chmod 755 "$HELPERS/bash"
fi
if [[ "$PHASE" != stage ]]; then
mkdir -p "$CONTENTS/Resources/Licenses"
curl -fLsS 'https://raw.githubusercontent.com/aaif-goose/goose/v1.54.0/LICENSE' \
    -o "$CONTENTS/Resources/Licenses/goose-APACHE-2.0.txt"
cp "$ROOT/Resources/Licenses/goose-third-party-licenses.txt" "$CONTENTS/Resources/Licenses/"
PYTHON="$CONTENTS/Resources/rapid-mlx/python/bin/python3.12"
SITE="$CONTENTS/Resources/rapid-mlx/site-packages"
if [[ "${SKIP_SIDECAR:-0}" == 1 ]]; then
    exit 0
fi
[[ -x "$PYTHON" && -d "$SITE" ]] || { echo "Goose Chat needs the bundled Python sidecar" >&2; exit 1; }
python3 -m pip install --disable-pip-version-check --no-deps --target "$SITE" \
    'openpyxl==3.1.5' 'et_xmlfile==2.0.0'
PYTHONPATH="$SITE" "$PYTHON" -c 'import pypdf, openpyxl, csv, decimal'
# The sidecar already bundles pypdf. Preserve its tested version and ship the
# exact wheel notices for the helper's document readers.
for entry in \
    "$SITE"/pypdf-*.dist-info/licenses/LICENSE \
    "$SITE"/openpyxl-3.1.5.dist-info/LICENCE.rst \
    "$SITE"/et_xmlfile-2.0.0.dist-info/LICENCE.rst \
    "$SITE"/et_xmlfile-2.0.0.dist-info/LICENCE.python; do
    [[ -s "$entry" ]] || { echo "missing Python dependency notice: $entry" >&2; exit 1; }
    package="$(basename "$(dirname "$entry")")"
    if [[ "$package" == licenses ]]; then package="$(basename "$(dirname "$(dirname "$entry")")")"; fi
    cp "$entry" "$CONTENTS/Resources/Licenses/${package}-$(basename "$entry")"
done

fi
