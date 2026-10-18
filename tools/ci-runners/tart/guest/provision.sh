#!/usr/bin/env bash
# Provision the golden Ubuntu 24.04 arm64 CI image. Runs INSIDE the VM as root.
# Idempotent enough to re-run; contains no credentials.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

RUNNER_VERSION="${RUNNER_VERSION:-2.338.0}"
RUNNER_SHA256="${RUNNER_SHA256:-628b4a7258487b80c1d3c221095a7ded349f5ae0175fd3dfab708d07428f041b}"
NODE_VERSION="${NODE_VERSION:-22.23.2}"
# Python builds published for setup-python (linux-24.04-arm64). setup-python
# resolves "3.12" etc. to these toolcache entries without any download.
TOOLCACHE=/opt/hostedtoolcache
RUNNER_USER=admin
RUNNER_HOME=/home/$RUNNER_USER/actions-runner

# Boot-time apt activity would race job steps that call apt-get.
systemctl disable --now unattended-upgrades.service apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
apt-get -y purge unattended-upgrades 2>/dev/null || true

apt-get update
apt-get install -y --no-install-recommends \
  build-essential git git-lfs curl wget ca-certificates jq zstd unzip zip xz-utils \
  pkg-config libssl-dev libffi-dev zlib1g-dev libbz2-dev libreadline-dev \
  libsqlite3-dev liblzma-dev libncurses-dev tk-dev uuid-dev libicu74 \
  python3 python3-venv python3-pip gnupg lsb-release sudo rsync file \
  python-is-python3 libgl1 libglib2.0-0t64 libcurl4-openssl-dev libxml2-dev libstdc++-13-dev libncurses6 binutils

# GitHub CLI (the hosted image ships it; workflow steps call `gh`).
if ! command -v gh >/dev/null; then
  install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /tmp/githubcli-archive-keyring.gpg
  echo "6084d5d7bd8e288441e0e94fc6275570895da18e6751f70f057485dc2d1a811b  /tmp/githubcli-archive-keyring.gpg" | sha256sum -c -
  install -m 0644 /tmp/githubcli-archive-keyring.gpg /etc/apt/keyrings/githubcli-archive-keyring.gpg
  rm -f /tmp/githubcli-archive-keyring.gpg
  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
  apt-get update && apt-get install -y --no-install-recommends gh
fi

# Node.js (official tarball) for steps that call node/npm directly; JS actions
# use the runner's bundled node.
if ! /usr/local/bin/node --version 2>/dev/null | grep -q "v$NODE_VERSION"; then
  curl -fsSL -o /tmp/node.tar.xz "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-arm64.tar.xz"
  curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/SHASUMS256.txt" | grep " node-v$NODE_VERSION-linux-arm64.tar.xz\$" | sed 's#node-v.*#/tmp/node.tar.xz#' | sha256sum -c -
  tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1
  rm -f /tmp/node.tar.xz
fi

# Pre-populated toolcache in the hosted-runner layout. Each archive is pinned
# by URL and SHA-256 before its bundled setup.sh runs.
mkdir -p "$TOOLCACHE"
PY_BASE=https://github.com/actions/python-versions/releases/download
while read -r v tag sha; do
  [ -n "$v" ] || continue
  if [ -f "$TOOLCACHE/Python/$v/arm64.complete" ]; then continue; fi
  d=$(mktemp -d)
  curl -fsSL -o "$d/py.tgz" "$PY_BASE/$tag/python-$v-linux-24.04-arm64.tar.gz"
  echo "$sha  $d/py.tgz" | sha256sum -c -
  tar -xzf "$d/py.tgz" -C "$d"
  (cd "$d" && RUNNER_TOOL_CACHE="$TOOLCACHE" AGENT_TOOLSDIRECTORY="$TOOLCACHE" bash ./setup.sh)
  rm -rf "$d"
done <<'PYTHONS'
3.10.22 3.10.22-36876129702 678191f49348bcaf1d420ad634202d9e5210a145eb9eb84f01e3b221211f0aa2
3.11.17 3.11.17-36876195594 a7a7bb9e6b50483f393d1c44c87aa87677a0464a9e3b07f33ee5b31ce155c6ef
3.12.15 3.12.15-36805895057 c537f7852633f9a6d909181528e263087ad815cc44d52ed226992b70b2648f44
3.13.16 3.13.16-36805956071 0e923697b278f1a21ac403838a4238b659200726d8983c3f271a120e76090f0e
PYTHONS
chown -R "$RUNNER_USER:$RUNNER_USER" "$TOOLCACHE"

# Swift toolchain: the hosted ubuntu image ships one and some unit tests key
# off `swift` being on PATH.
SWIFT_VERSION=6.4.1
SWIFT_SHA256=b85dd70e204bc363ce62f1e35bd1887810d0e04487d13de8fd9d042635972f45
if ! /usr/local/swift/usr/bin/swift --version 2>/dev/null | grep -q "$SWIFT_VERSION"; then
  curl -fsSL -o /tmp/swift.tgz "https://download.swift.org/swift-$SWIFT_VERSION-release/ubuntu2404-aarch64/swift-$SWIFT_VERSION-RELEASE/swift-$SWIFT_VERSION-RELEASE-ubuntu24.04-aarch64.tar.gz"
  echo "$SWIFT_SHA256  /tmp/swift.tgz" | sha256sum -c -
  rm -rf /usr/local/swift; mkdir -p /usr/local/swift
  tar -xzf /tmp/swift.tgz -C /usr/local/swift --strip-components=1; rm -f /tmp/swift.tgz
  ln -sf /usr/local/swift/usr/bin/swift /usr/local/bin/swift
fi

# Pre-seeded tiktoken vocab files: openai-harmony's arm64 Linux wheel cannot
# download them itself (its TLS stack rejects the blob endpoint), so point it
# at a local copy, verified against the published digests.
ENC=/opt/tiktoken-encodings; mkdir -p "$ENC"
fetch_enc() {
  [ -f "$ENC/$1" ] && echo "$2  $ENC/$1" | sha256sum -c - >/dev/null 2>&1 && return 0
  curl -fsSL -o "$ENC/$1" "https://openaipublic.blob.core.windows.net/encodings/$1"
  echo "$2  $ENC/$1" | sha256sum -c -
}
fetch_enc o200k_base.tiktoken 446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d
fetch_enc cl100k_base.tiktoken 223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7
chmod -R a+rX "$ENC"

# uv (fast installs; optional for jobs). Pinned release, checksum-verified.
UV_VERSION=0.13.1
UV_SHA256=9041c73c99fdc5892cf128140beb354eca9231d502068cc36e6cb53a8e59ad14
if ! uv --version 2>/dev/null | grep -q " $UV_VERSION "; then
  curl -fsSL -o /tmp/uv.tgz "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-unknown-linux-gnu.tar.gz"
  echo "$UV_SHA256  /tmp/uv.tgz" | sha256sum -c -
  tar -xzf /tmp/uv.tgz -C /tmp
  install -m 0755 /tmp/uv-aarch64-unknown-linux-gnu/uv /tmp/uv-aarch64-unknown-linux-gnu/uvx /usr/local/bin/
  rm -rf /tmp/uv.tgz /tmp/uv-aarch64-unknown-linux-gnu
fi

# GitHub Actions runner. Registration happens per boot with a JIT config
# supplied by the host; nothing is registered in the image.
if [ ! -x "$RUNNER_HOME/run.sh" ] || ! grep -q "$RUNNER_VERSION" "$RUNNER_HOME/.runner_version" 2>/dev/null; then
  rm -rf "$RUNNER_HOME"; mkdir -p "$RUNNER_HOME"
  curl -fsSL -o /tmp/runner.tgz "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-linux-arm64-$RUNNER_VERSION.tar.gz"
  echo "$RUNNER_SHA256  /tmp/runner.tgz" | sha256sum -c -
  tar -xzf /tmp/runner.tgz -C "$RUNNER_HOME"; rm -f /tmp/runner.tgz
  "$RUNNER_HOME/bin/installdependencies.sh"
  echo "$RUNNER_VERSION" > "$RUNNER_HOME/.runner_version"
fi
cat > "$RUNNER_HOME/.env" <<EOF
LANG=C.UTF-8
RUNNER_TOOL_CACHE=$TOOLCACHE
AGENT_TOOLSDIRECTORY=$TOOLCACHE
ImageOS=ubuntu24
TIKTOKEN_ENCODINGS_BASE=$ENC
EOF
chown -R "$RUNNER_USER:$RUNNER_USER" "/home/$RUNNER_USER"

# Hygiene: nothing secret or host-specific in the image.
apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache
rm -f /home/$RUNNER_USER/.bash_history /root/.bash_history
journalctl --rotate >/dev/null 2>&1 || true; journalctl --vacuum-time=1s >/dev/null 2>&1 || true
sync; fstrim -av >/dev/null 2>&1 || true; sync
echo "golden provisioned: runner $RUNNER_VERSION, node $NODE_VERSION, python $(ls "$TOOLCACHE/Python" | tr "\n" " ")"
