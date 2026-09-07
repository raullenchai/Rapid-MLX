#!/usr/bin/env bash
#
# train_gates.sh <base-sha>
#
# Reproduce, LOCALLY on one machine, the same 5 validation gates the hosted CI
# matrix runs for an engine train, so a train can be "frozen" without waiting
# for (or burning) hosted runners. On full success it prints exactly:
#
#     GATES OK <sha> <gates-hash>  (python X.Y.Z)
#
# and exits 0. If any non-skippable gate fails it prints a per-gate FAILURE
# block and exits 1. A run on a DIRTY working tree (tracked files modified or
# staged — untracked files are ignored by the check) still runs every gate but
# NEVER prints the literal `GATES OK`: its receipt is
#
#     GATES DIRTY <sha> <gates-hash>  (python X.Y.Z)
#
# and it exits 3, because the validated bytes are not the committed <sha>.
#
# Exit codes:
#   0  GATES OK — all 5 gates passed on a clean tree (the freeze receipt)
#   1  a gate failed, or fewer than 5 gates passed (a skip is not a pass)
#   2  usage error, unresolvable base, base == HEAD, base not an ancestor, or
#      an unusable control interpreter (no PyYAML, or python < 3.10)
#   3  GATES DIRTY — all 5 gates passed but the tree has modified tracked files
#
# The gate definitions are NOT hardcoded here: they are parsed at runtime from
# .github/workflows/ci.yml and .github/workflows/rapid-mac-ci.yml by
# scripts/train_gates_parser.py (the single source of truth). The drift test
# tests/test_train_gates_matches_ci.py guards that this parser stays in sync
# with the workflows.
#
# Environment (all optional):
#   TRAIN_GATES_PYTHON           control interpreter: must be able to import
#                                yaml (PyYAML); every gate builds its own fresh
#                                venv FROM this interpreter. The hosted coverage
#                                union is 3.11-only, so use a python3.11 for a
#                                faithful freeze (a WARNING is printed for any
#                                other version). Defaults to python3, then falls
#                                back to a repo .venv.
#   TRAIN_GATES_APPLE_VENV       path to an existing Apple-Silicon venv that
#                                already has the package installed (with mlx),
#                                to reuse for Gate 4 instead of reinstalling.
#                                Gate 4 verifies that its `vllm_mlx` imports
#                                from THIS checkout; if not (stale editable
#                                install pinned to another worktree, or a
#                                wheel) it re-points the venv with
#                                `pip install -e "$ROOT" --no-deps`, and fails
#                                the gate if that still does not resolve here.
#   TRAIN_GATES_ALLOW_APPLE_INSTALL=1
#                                if no apple venv is provided, create a fresh
#                                one and `pip install -e "$ROOT[vision]"` (slow).
#   TRAIN_GATES_SKIP_APPLE=1     skip Gate 4 with a clear SKIPPED message
#                                (a skip is NOT a pass).
#   TRAIN_GATES_SKIP_SWIFT=1     skip Gate 5 with a clear SKIPPED message
#                                (a skip is NOT a pass).
#
# The 5 gates (hosted equivalents in parens):
#   1. Linux no-MLX pytest  (ci.yml test-matrix)   — fresh venv, the hosted
#      install verbatim (`pip install -e . --no-deps` + the synced
#      config/requirements-ci-linux.txt), assert `import mlx` FAILS, then run
#      the parsed Linux pytest invocations (one process per `pytest` block in
#      ci.yml — automatic unit discovery and the fake-MLX lifecycle set run in
#      separate processes, mirroring the hosted split, with the second
#      `--cov-append`ing into the same data).
#   2. mypy error budget     (ci.yml type-check)    — `pip install -r
#      config/mypy-requirements.txt` then `python scripts/check_mypy_error_budget.py`.
#   3. coverage union        (ci.yml changed-lines-coverage) — fresh venv with
#      `coverage` + the exact `diff-cover==X.Y.Z` pin parsed from ci.yml,
#      combine the Linux+Apple coverage .data produced by gates 1+4, emit
#      coverage.xml, then diff-cover --compare-branch <base-sha> --fail-under 100.
#   4. Apple-MLX pytest      (ci.yml test-apple-silicon) — run the parsed Apple
#      pytest roster (with ci.yml's -m / -k filters) in an mlx-capable venv.
#   5. Desktop swift test    (rapid-mac-ci.yml build, the `swift test` step
#      only) — `swift test --no-parallel` in apps/ when Desktop sources changed
#      vs <base-sha>; when apps/ is unchanged the gate is PASS-BY-N/A (counts
#      toward the 5-pass contract).
#
# gates-hash: a deterministic hash over the exact gate definitions (the parsed
# Linux/Apple pytest args, the Linux lane's requirements file, the mypy script
# + budget files, the diff-cover inputs/flags/pin, the swift invocation, and
# the relevant workflow step text). A CI-definition edit (a test added to a
# pytest roster, a mypy budget pin change, a diff_cover knob) changes the hash.
# Host state (venv paths, timestamps, machine id) is never included.
#
# Scratch: ALL transient state — the per-gate coverage artifacts AND the four
# per-gate venvs (Linux, mypy, coverage, and the Apple venv when
# TRAIN_GATES_ALLOW_APPLE_INSTALL=1 creates one) — lives in ONE TMPDIR-honoring
# run dir, $RUN_TMP (never the repo root, never loose in TMPDIR). The run dir is
# deleted on success and KEPT (path printed) when a gate fails, so the per-gate
# coverage inputs, the combined data, the xml and the venvs can be inspected.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARSER_MODULE="scripts.train_gates_parser"

usage() {
  cat <<'EOF'
usage: scripts/train_gates.sh <base-sha>
       scripts/train_gates.sh -h | --help

  <base-sha>  the commit to diff/validate against (any git rev: sha, tag,
              branch). It must be a strict ancestor of HEAD — typically
              $(git merge-base origin/main HEAD). HEAD itself is refused: an
              empty diff would pass diff-cover at 100% by construction.

Receipt (last line) and exit code:
  GATES OK <head-sha> <gates-hash>  (python X.Y.Z)      0  frozen
  GATES DIRTY <head-sha> <gates-hash>  (python X.Y.Z)   3  all gates passed
        but tracked files are modified/staged — NOT a freeze (untracked files
        are ignored by the dirty check)
  GATES FAILED / GATES INCOMPLETE                        1
  usage error, unresolvable base, base == HEAD, base not an ancestor   2
  unusable control interpreter (no PyYAML, or python < 3.10)           2

Environment (optional): TRAIN_GATES_PYTHON (PyYAML-capable python3.11
  recommended), TRAIN_GATES_APPLE_VENV, TRAIN_GATES_ALLOW_APPLE_INSTALL=1,
  TRAIN_GATES_SKIP_APPLE=1, TRAIN_GATES_SKIP_SWIFT=1 — see the script header.
EOF
}

# ---------------------------------------------------------------------------
# Argument handling — BEFORE any interpreter resolution, workflow parsing or
# hashing, so `--help` and bad invocations are cheap and never touch the
# environment.
# ---------------------------------------------------------------------------
if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  echo "ERROR: expected exactly one argument <base-sha>, got $#" >&2
  usage >&2
  exit 2
fi
if [[ -z "$1" || "$1" == -* ]]; then
  echo "ERROR: unknown option or empty base '$1'" >&2
  usage >&2
  exit 2
fi
BASE_SHA="$1"

# Resolve the base in THIS repo. `--quiet` keeps git's own `fatal:` off the
# terminal so an unresolvable base yields exactly one clear line + usage.
if ! BASE_SHA_RESOLVED="$(git -C "$ROOT" rev-parse --verify --quiet "${BASE_SHA}^{commit}")"; then
  echo "ERROR: cannot resolve base '$BASE_SHA' to a commit in $ROOT (unfetched? typo?)" >&2
  usage >&2
  exit 2
fi

# The validated head is the current checkout HEAD (what we're actually testing).
HEAD_RESOLVED="$(git -C "$ROOT" rev-parse HEAD)"

# Refuse base == HEAD: diff-cover would compare an EMPTY diff and pass 100% by
# construction — a receipt that validates nothing.
if [[ "$BASE_SHA_RESOLVED" == "$HEAD_RESOLVED" ]]; then
  echo "ERROR: base $BASE_SHA_RESOLVED IS the current HEAD; there are no changed lines to validate." >&2
  echo "       diff-cover would compare an empty diff and pass 100% by construction." >&2
  echo "       Pass the merge-base instead, e.g. \$(git merge-base origin/main HEAD)." >&2
  exit 2
fi

# B4: reject a wrong/forward base. diff-cover compares against <base-sha>; it
# must be a real ancestor of HEAD, else a forward/foreign base yields a trivial
# (often empty) diff and a guaranteed 100% pass — a false green.
if ! git -C "$ROOT" merge-base --is-ancestor "$BASE_SHA_RESOLVED" "$HEAD_RESOLVED"; then
  echo "ERROR: base $BASE_SHA_RESOLVED is NOT an ancestor of HEAD." >&2
  echo "       A forward/wrong base would make diff-cover compare an empty/trivial diff and pass 100% by construction." >&2
  exit 2
fi

# Dirty check — tracked files only. `--untracked-files=no` deliberately ignores
# untracked files (scratch notes, build output) because they are not part of
# the validated <sha> either way; a modified or staged TRACKED file means the
# bytes under test differ from the commit, so the receipt becomes GATES DIRTY.
GIT_DIRTY=""
DIRTY_STATUS="$(git -C "$ROOT" status --porcelain --untracked-files=no)"
if [[ -n "$DIRTY_STATUS" ]]; then
  GIT_DIRTY=1
  echo "train-gates: WARNING: working tree is DIRTY (modified/staged tracked files; untracked files are ignored):"
  printf '%s\n' "$DIRTY_STATUS" | sed 's/^/    /'
  echo "train-gates:          the gates still run, but the receipt will be GATES DIRTY (exit 3), never the OK receipt."
fi

# ---------------------------------------------------------------------------
# Scratch dir (honors TMPDIR) + exit cleanup. All transient coverage artifacts
# live here — NEVER the repo root — so a second run on a new head cannot
# silently union stale coverage from the previous head. Kept on failure.
# ---------------------------------------------------------------------------
RUN_TMP="$(mktemp -d "${TMPDIR:-/tmp}/rapid-train-gates-run-XXXXXX")"
COV_DIR="$RUN_TMP/cov"
mkdir -p "$COV_DIR"
KEEP_RUN_TMP=""
cleanup() {
  if [[ -n "$KEEP_RUN_TMP" ]]; then
    echo "train-gates: run artifacts kept for inspection: $RUN_TMP" >&2
  else
    rm -rf "$RUN_TMP"
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Control interpreter (needs only PyYAML: every gate builds its own venv).
# ---------------------------------------------------------------------------
py_version_of() {
  "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown"
}

resolve_python() {
  local candidate=""
  if [[ -n "${TRAIN_GATES_PYTHON:-}" ]]; then
    if ! "${TRAIN_GATES_PYTHON}" -c 'import yaml' >/dev/null 2>&1; then
      echo "ERROR: TRAIN_GATES_PYTHON=${TRAIN_GATES_PYTHON} cannot import yaml (pip install pyyaml)" >&2
      return 1
    fi
    candidate="${TRAIN_GATES_PYTHON}"
  elif python3 -c 'import yaml' >/dev/null 2>&1; then
    candidate="python3"
  elif [[ -x "$ROOT/.venv/bin/python" ]] \
    && "$ROOT/.venv/bin/python" -c 'import yaml' >/dev/null 2>&1; then
    candidate="$ROOT/.venv/bin/python"
  else
    echo "ERROR: no interpreter with PyYAML found; set TRAIN_GATES_PYTHON (a python3.11 with pyyaml)" >&2
    return 1
  fi
  # HARD floor: pyproject's requires-python and the hosted matrix (3.10/3.11/
  # 3.12) never go below 3.10, and every gate venv is created FROM this
  # interpreter — below the floor the package cannot even install, so this is
  # an error (exit 2), not the 3.11 WARNING below.
  if ! "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    echo "ERROR: control interpreter $candidate is python $(py_version_of "$candidate"), below the 3.10 floor" >&2
    echo "       (pyproject requires-python >= 3.10; the hosted matrix is 3.10/3.11/3.12)." >&2
    echo "       Set TRAIN_GATES_PYTHON=<python3.11 with pyyaml> for a faithful freeze." >&2
    return 1
  fi
  echo "$candidate"
}

if ! PYTHON_BIN="$(resolve_python)"; then
  exit 2
fi
PY_VERSION="$(py_version_of "$PYTHON_BIN")"
echo "train-gates: control interpreter: $PYTHON_BIN (python $PY_VERSION)"
if [[ "${PY_VERSION%.*}" != "3.11" ]]; then
  echo "train-gates: WARNING: control interpreter is python $PY_VERSION, but the hosted coverage union" >&2
  echo "                      (changed-lines-coverage) and the Linux data it consumes are 3.11-only." >&2
  echo "                      Every gate venv is created from this interpreter, so line coverage can" >&2
  echo "                      diverge on version-conditional code. Set TRAIN_GATES_PYTHON=<python3.11>" >&2
  echo "                      for a faithful freeze." >&2
fi

# ---------------------------------------------------------------------------
# Parse the gate surface from the workflows (single source of truth).
# ---------------------------------------------------------------------------
json_parse() {
  # $PYTHON_BIN -m scripts.train_gates_parser <target>
  ( cd "$ROOT" && "$PYTHON_BIN" -m "$PARSER_MODULE" "$1" )
}

LINUX_JSON="$(json_parse linux)"
APPLE_JSON="$(json_parse apple)"
DIFF_JSON="$(json_parse diff_cover)"
SWIFT_JSON="$(json_parse swift_test)"

# ---------------------------------------------------------------------------
# gates-hash (deterministic; no host state).
# ---------------------------------------------------------------------------
compute_gates_hash() {
  local mypy_script_sha
  mypy_script_sha="$(git -C "$ROOT" hash-object "$ROOT/scripts/check_mypy_error_budget.py")"
  # (a)+(b) parsed Linux + Apple pytest args; (c) mypy script committed hash;
  # (d) the workflow step text (captures ANY knob change). Feed the whole
  # deterministic payload through `git hash-object --stdin`.
  {
    echo "linux-pytest"
    "$PYTHON_BIN" -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$LINUX_JSON"
    echo "linux-requirements"
    git -C "$ROOT" hash-object "$ROOT/config/requirements-ci-linux.txt"
    echo "apple-pytest"
    "$PYTHON_BIN" -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$APPLE_JSON"
    echo "mypy-script-sha"
    echo "$mypy_script_sha"
    echo "mypy-requirements"
    git -C "$ROOT" hash-object "$ROOT/config/mypy-requirements.txt"
    echo "mypy-baseline"
    git -C "$ROOT" hash-object "$ROOT/config/mypy-error-baseline.txt"
    echo "diff-cover"
    "$PYTHON_BIN" -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$DIFF_JSON"
    echo "swift-test"
    "$PYTHON_BIN" -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$SWIFT_JSON"
    echo "workflow-step-text"
    git -C "$ROOT" hash-object .github/workflows/ci.yml
    git -C "$ROOT" hash-object .github/workflows/rapid-mac-ci.yml
  } | git -C "$ROOT" hash-object --stdin
}

GATES_HASH="$(compute_gates_hash)"

# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------
PASSED=()
SKIPPED=()
FAILED=()

note()  { printf '    %s\n' "$*"; }
passed(){ PASSED+=("$1"); if [[ $# -ge 2 ]]; then printf 'GATE %s: PASS — %s\n' "$1" "$2"; else printf 'GATE %s: PASS\n' "$1"; fi; }
passed_na(){ PASSED+=("$1"); if [[ $# -ge 2 ]]; then printf 'GATE %s: PASS (N/A) — %s\n' "$1" "$2"; else printf 'GATE %s: PASS (N/A)\n' "$1"; fi; }
skip()  { SKIPPED+=("$1"); printf 'GATE %s: SKIPPED — %s\n' "$1" "${2:-}"; }
fail()  { FAILED+=("$1"); if [[ $# -ge 2 ]]; then printf 'GATE %s: FAILURE — %s\n' "$1" "$2"; else printf 'GATE %s: FAILURE\n' "$1"; fi; }

# run_gate <label> <fn> [args...] — run one gate, never abort the run on its
# failure (every gate records its own PASSED/FAILED/SKIPPED entry), and print
# its wall time so a freeze transcript carries per-gate timings.
run_gate() {
  local label="$1"
  shift
  local t0=$SECONDS
  "$@" || true
  echo "  gate $label wall time: $((SECONDS - t0))s"
}

# ---------------------------------------------------------------------------
# Gate 1 — Linux no-MLX pytest (fresh venv, no mlx), one process per ci.yml
# pytest block.
# ---------------------------------------------------------------------------
gate1_linux() {
  echo
  echo "== Gate 1: Linux no-MLX pytest =="
  # Under $RUN_TMP (never loose in TMPDIR) so the EXIT trap removes the venv on
  # success and keeps it next to the coverage inputs on failure.
  local venv="$RUN_TMP/venv-linux"
  local py="$venv/bin/python"
  echo "  fresh venv: $venv"

  # run_gate invokes this body under `|| true`, which disables errexit for the
  # whole function: every environment step is guarded explicitly so a failed
  # venv/pip aborts THIS gate with a truthful message instead of cascading into
  # a misleading pytest failure.
  "$PYTHON_BIN" -m venv "$venv" \
    || { fail 1 "python -m venv $venv failed (control interpreter $PYTHON_BIN)"; return 1; }
  "$py" -m pip install --quiet --upgrade pip \
    || { fail 1 "pip upgrade in the fresh Linux venv failed"; return 1; }
  # Mirror ci.yml test-matrix "Install dependencies" line for line:
  #     pip install -e . --no-deps
  #     pip install --requirement config/requirements-ci-linux.txt
  # The package goes in editable and WITHOUT deps, so the base deps' mlx /
  # mlx-lm never land (the "no MLX" contract), and the lane's test surface
  # comes from the same synced requirements file the hosted lane installs —
  # never an ad hoc package list here (a pin change there also changes the
  # gates-hash). Remaining, documented divergence from hosted: this runs on
  # the local OS/arch with the control interpreter's version, not on
  # ubuntu-latest x86-64 across the 3.10/3.11/3.12 matrix; discovery, flags
  # and the coverage data name (coverage-linux-3.11.data — the leg the hosted
  # union consumes) are identical.
  "$py" -m pip install --quiet -e "$ROOT" --no-deps \
    || { fail 1 "pip install -e . --no-deps into the fresh Linux venv failed"; return 1; }
  "$py" -m pip install --quiet --requirement "$ROOT/config/requirements-ci-linux.txt" \
    || { fail 1 "pip install -r config/requirements-ci-linux.txt into the fresh Linux venv failed"; return 1; }

  if "$py" -c "import mlx" >/dev/null 2>&1; then
    fail 1 "import mlx unexpectedly SUCCEEDED in the fresh --no-deps venv; the hosted Linux gate expects no MLX"
    return 1
  fi
  note "mlx correctly absent from fresh --no-deps venv"

  # ci.yml runs TWO separate pytest processes in this step (automatic unit
  # discovery and the fake-MLX lifecycle set — see train_gates_parser.py).
  # Reproduce that split: run each parsed invocation in its OWN pytest process,
  # in ci.yml's order, each writing into the SAME coverage file with the
  # declared --cov-append semantics so the union equals the hosted combined
  # Linux coverage.
  local n_inv
  n_inv="$("$PYTHON_BIN" -c 'import sys,json; print(len(json.loads(sys.argv[1])))' "$LINUX_JSON")"
  echo "  running $n_inv Linux pytest process(es) parsed from ci.yml"
  local covfile="$COV_DIR/coverage-linux-3.11.data"
  for (( i=0; i<n_inv; i++ )); do
    local n_files cov_append paths_str ignore_str deselect_str marker_str m_str
    n_files="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(len(inv["paths"]))
PY
)"
    cov_append="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print("1" if inv["cov_declaration"]["cov_append"] else "0")
PY
)"
    echo "    pytest process $((i+1))/$n_inv ($n_files test file tokens, cov_append=$cov_append)"

    # Split the parsed space-joined token string into an array so the paths
    # reach pytest verbatim (no accidental globbing/word-splitting).
    paths_str="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(" ".join(inv["paths"]))
PY
)"
    ignore_str="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(" ".join("--ignore=%s" % d for d in inv["ignore"]))
PY
)"
    deselect_str="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(" ".join("--deselect=%s" % d for d in inv["deselect"]))
PY
)"
    marker_str="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(inv["marker"] or "")
PY
)"
    m_str="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(inv["m"] or "")
PY
)"

    # Split the parsed space-joined strings into arrays so no accidental
    # word-splitting/globbing ever occurs when they are passed to pytest.
    # NOTE (bash-3.2 + set -u): `read -a` on an EMPTY string leaves the array
    # unbound, and `${arr[@]}` on an unbound array errors under `set -u`. So
    # every array expansion below uses the `${arr[@]+"${arr[@]}"}` guard, which
    # is a no-op when the array holds no elements.
    local -a pytest_args=() ignore_args=() deselect_args=()
    read -r -a pytest_args <<<"$paths_str"
    read -r -a ignore_args <<<"$ignore_str"
    read -r -a deselect_args <<<"$deselect_str"
    local -a aux_args=()
    if [[ "$cov_append" == "1" ]]; then aux_args+=(--cov-append); fi
    if [[ -n "$marker_str" ]]; then aux_args+=(-k "$marker_str"); fi
    if [[ -n "$m_str" ]]; then aux_args+=(-m "$m_str"); fi

    if ! ( cd "$ROOT" \
        && COVERAGE_FILE="$covfile" \
        "$py" -m pytest \
          ${pytest_args[@]+"${pytest_args[@]}"} \
          ${ignore_args[@]+"${ignore_args[@]}"} \
          ${deselect_args[@]+"${deselect_args[@]}"} \
          ${aux_args[@]+"${aux_args[@]}"} \
          -v --tb=short \
          --cov=vllm_mlx \
          --cov-report=term-missing ); then
      fail 1 "Linux no-MLX pytest process $((i+1)) failed (see output above)"
      return 1
    fi
  done
  note "Linux coverage data written to $covfile"
  passed 1
}

# ---------------------------------------------------------------------------
# Gate 2 — pinned mypy error budget.
# ---------------------------------------------------------------------------
gate2_mypy() {
  echo
  echo "== Gate 2: mypy error budget =="
  if [[ ! -f "$ROOT/config/mypy-requirements.txt" ]]; then
    fail 2 "config/mypy-requirements.txt missing"
    return 1
  fi
  local venv="$RUN_TMP/venv-mypy"
  local py="$venv/bin/python"
  echo "  mypy venv: $venv"
  "$PYTHON_BIN" -m venv "$venv" \
    || { fail 2 "python -m venv $venv failed (control interpreter $PYTHON_BIN)"; return 1; }
  "$py" -m pip install --quiet --upgrade pip \
    || { fail 2 "pip upgrade in the mypy venv failed"; return 1; }
  "$py" -m pip install --quiet --requirement "$ROOT/config/mypy-requirements.txt" \
    || { fail 2 "pip install -r config/mypy-requirements.txt into the mypy venv failed"; return 1; }
  note "installed pinned mypy environment"
  if ! ( cd "$ROOT" && "$py" "$ROOT/scripts/check_mypy_error_budget.py" ); then
    fail 2 "mypy error budget overrun (see output above); freeze blocked"
    return 1
  fi
  passed 2
}

# ---------------------------------------------------------------------------
# Gate 3 — coverage union + diff-cover.
# ---------------------------------------------------------------------------
gate3_diffcover() {
  echo
  echo "== Gate 3: coverage union + diff-cover =="
  local base_sha="$1"
  local linux_data="$COV_DIR/coverage-linux-3.11.data"
  local apple_data="$COV_DIR/coverage-apple.data"
  if [[ ! -f "$linux_data" ]]; then
    fail 3 "missing $linux_data (Gate 1 must produce it)"
    return 1
  fi
  if [[ ! -f "$apple_data" ]]; then
    fail 3 "missing $apple_data (Gate 4 must produce it)"
    return 1
  fi

  # Mirror the hosted job's "Install coverage tools" step in a fresh venv:
  # `pip install coverage "diff-cover==X.Y.Z"`, with the pin PARSED from
  # ci.yml (never hardcoded here) so a hosted pin bump moves this gate — and
  # the gates-hash — with it.
  local pin
  pin="$("$PYTHON_BIN" -c 'import sys,json; print(json.loads(sys.argv[1])["diff_cover_pin"])' "$DIFF_JSON")"
  local venv="$RUN_TMP/venv-cov"
  local py="$venv/bin/python"
  echo "  coverage venv: $venv (coverage + diff-cover==$pin, the ci.yml pin)"
  "$PYTHON_BIN" -m venv "$venv" \
    || { fail 3 "python -m venv $venv failed (control interpreter $PYTHON_BIN)"; return 1; }
  "$py" -m pip install --quiet --upgrade pip \
    || { fail 3 "pip upgrade in the coverage venv failed"; return 1; }
  "$py" -m pip install --quiet coverage "diff-cover==$pin" \
    || { fail 3 "pip install coverage diff-cover==$pin into the coverage venv failed"; return 1; }

  # Reproduce the hosted changed-lines-coverage job exactly: combine + xml +
  # diff-cover all run FROM the repo root (so `.coveragerc` applies, the
  # relative_files coverage paths resolve, and `--compare-branch <base-sha>`
  # resolves against `.git`). The .data inputs live under COV_DIR (never the
  # repo root), and the transient combined `.coverage` + `coverage.xml` are
  # pointed into COV_DIR too so no coverage artifact ever lands in the repo
  # root (a stale one there would silently union on the next run's new head).
  #
  # COVERAGE_FILE is exported for the WHOLE subshell: `coverage xml` reads the
  # combined data file, which lives at $combined — with the variable scoped to
  # `combine` only, `xml` would look for a nonexistent ./.coverage and fail
  # "No data to report." `--keep` leaves the per-gate inputs in place (the run
  # dir is kept on failure) so they can be inspected.
  local combined="$COV_DIR/.coverage"
  local work_xml="$COV_DIR/coverage.xml"
  rm -f "$combined" "$work_xml"
  if ! ( cd "$ROOT" \
      && export COVERAGE_FILE="$combined" \
      && "$py" -m coverage combine --keep "$linux_data" "$apple_data" \
      && "$py" -m coverage xml -o "$work_xml" ); then
    fail 3 "coverage combine/xml failed (see output above)"
    return 1
  fi
  if ! ( cd "$ROOT" \
      && "$py" -m diff_cover.diff_cover_tool \
          "$work_xml" \
          --compare-branch "$base_sha" \
          --show-uncovered \
          --fail-under 100 ); then
    fail 3 "diff-cover --compare-branch $base_sha failed (see output above)"
    return 1
  fi
  passed 3
}

# ---------------------------------------------------------------------------
# Gate 4 — Apple-MLX pytest.
# ---------------------------------------------------------------------------
gate4_apple() {
  echo
  echo "== Gate 4: Apple-MLX pytest =="
  if [[ "${TRAIN_GATES_SKIP_APPLE:-0}" == "1" ]]; then
    skip 4 "TRAIN_GATES_SKIP_APPLE=1"
    return 0
  fi

  local apple_py=""
  if [[ -n "${TRAIN_GATES_APPLE_VENV:-}" ]]; then
    apple_py="$TRAIN_GATES_APPLE_VENV/bin/python"
    if [[ ! -x "$apple_py" ]]; then
      fail 4 "TRAIN_GATES_APPLE_VENV=$TRAIN_GATES_APPLE_VENV has no bin/python"
      return 1
    fi
    # A reused venv must import THIS checkout's package: an editable install
    # pinned to another worktree, or a wheel in site-packages, would make Gate
    # 4 (and its coverage data) test bytes that are not HEAD. Probe from a
    # neutral cwd ($RUN_TMP) so the repo root is not silently on sys.path.
    local root_real pkg_file
    root_real="$(cd "$ROOT" && pwd -P)"
    pkg_file="$(cd "$RUN_TMP" && "$apple_py" -c 'import os, vllm_mlx; print(os.path.realpath(vllm_mlx.__file__))' 2>/dev/null || true)"
    if [[ "$pkg_file" != "$root_real/vllm_mlx/"* ]]; then
      echo "  reused Apple venv imports vllm_mlx from '${pkg_file:-<not importable>}', not $root_real;"
      echo "  re-pointing it: pip install -e \"\$ROOT\" --no-deps"
      "$apple_py" -m pip install --quiet -e "$ROOT" --no-deps \
        || { fail 4 "pip install -e . --no-deps into TRAIN_GATES_APPLE_VENV=$TRAIN_GATES_APPLE_VENV failed"; return 1; }
      pkg_file="$(cd "$RUN_TMP" && "$apple_py" -c 'import os, vllm_mlx; print(os.path.realpath(vllm_mlx.__file__))' 2>/dev/null || true)"
      if [[ "$pkg_file" != "$root_real/vllm_mlx/"* ]]; then
        fail 4 "TRAIN_GATES_APPLE_VENV still imports vllm_mlx from '${pkg_file:-<not importable>}' after pip install -e . --no-deps; expected $root_real/vllm_mlx/"
        return 1
      fi
    fi
    note "reused Apple venv imports vllm_mlx from $pkg_file"
  elif [[ "${TRAIN_GATES_ALLOW_APPLE_INSTALL:-0}" == "1" ]]; then
    local venv="$RUN_TMP/venv-apple"
    apple_py="$venv/bin/python"
    echo "  creating Apple venv: $venv"
    "$PYTHON_BIN" -m venv "$venv" \
      || { fail 4 "python -m venv $venv failed (control interpreter $PYTHON_BIN)"; return 1; }
    "$apple_py" -m pip install --quiet --upgrade pip \
      || { fail 4 "pip upgrade in the fresh Apple venv failed"; return 1; }
    # Mirror ci.yml test-apple-silicon "Install project and dependencies":
    #     pip install -e ".[vision]"   then   pip install -e ".[ci-apple]"
    "$apple_py" -m pip install --quiet -e "${ROOT}[vision]" \
      || { fail 4 "pip install -e .[vision] into the fresh Apple venv failed"; return 1; }
    "$apple_py" -m pip install --quiet -e "${ROOT}[ci-apple]" \
      || { fail 4 "pip install -e .[ci-apple] into the fresh Apple venv failed"; return 1; }
  else
    fail 4 "no Apple venv provided; set TRAIN_GATES_APPLE_VENV=<venv with the package + mlx>, TRAIN_GATES_ALLOW_APPLE_INSTALL=1 to create one, or TRAIN_GATES_SKIP_APPLE=1"
    return 1
  fi

  if ! "$apple_py" -c "import mlx.core as mx" >/dev/null 2>&1; then
    fail 4 "Apple venv cannot import mlx (not an Apple-Silicon runtime?)"
    return 1
  fi

  # ci.yml's Apple -m / -k filters, parsed (not hardcoded) — if ci.yml changes
  # them, Gate 4 follows.
  local paths_str m_str k_str
  paths_str="$("$PYTHON_BIN" - "$APPLE_JSON" <<'PY'
import sys, json
print(" ".join(json.loads(sys.argv[1])["paths"]))
PY
)"
  m_str="$("$PYTHON_BIN" - "$APPLE_JSON" <<'PY'
import sys, json
print(json.loads(sys.argv[1]).get("m") or "")
PY
)"
  k_str="$("$PYTHON_BIN" - "$APPLE_JSON" <<'PY'
import sys, json
print(json.loads(sys.argv[1]).get("k") or "")
PY
)"

  echo "  running Apple-MLX pytest roster ($("$PYTHON_BIN" -c 'import sys,json; print(len(json.loads(sys.argv[1])["paths"]))' "$APPLE_JSON") test files)"
  local -a apple_args=() m_args=() k_args=()
  read -r -a apple_args <<<"$paths_str"
  if [[ -n "$m_str" ]]; then m_args=(-m "$m_str"); fi
  if [[ -n "$k_str" ]]; then k_args=(-k "$k_str"); fi
  if ! ( cd "$ROOT" \
      && COVERAGE_FILE="$COV_DIR/coverage-apple.data" \
        "$apple_py" -m pytest \
          ${apple_args[@]+"${apple_args[@]}"} \
        -v --tb=short \
        ${m_args[@]+"${m_args[@]}"} \
        ${k_args[@]+"${k_args[@]}"} \
        --cov=vllm_mlx \
        --cov-report=term-missing ); then
    fail 4 "Apple-MLX pytest failed (see output above)"
    return 1
  fi
  note "Apple coverage data written to $COV_DIR/coverage-apple.data"
  passed 4
}

# ---------------------------------------------------------------------------
# Gate 5 — Desktop swift test.
# ---------------------------------------------------------------------------
gate5_swift() {
  echo
  echo "== Gate 5: Desktop swift test =="
  if [[ "${TRAIN_GATES_SKIP_SWIFT:-0}" == "1" ]]; then
    skip 5 "TRAIN_GATES_SKIP_SWIFT=1"
    return 0
  fi
  local base_sha="$1"
  local desktop_dir="$ROOT/apps/rapid-mac"
  if [[ ! -d "$desktop_dir" ]]; then
    skip 5 "Desktop app dir $desktop_dir absent"
    return 0
  fi
  if ! command -v swift >/dev/null 2>&1; then
    skip 5 "swift toolchain not on PATH"
    return 0
  fi
  if ! git -C "$ROOT" diff --quiet "$base_sha" -- apps/; then
    note "apps/ changed vs $base_sha — running Desktop tests"
  else
    # apps/ unchanged => there is nothing for the build job's swift test step
    # to exercise. Count as a PASS-BY-N/A (it satisfies the "all five must
    # pass" contract) rather than a skip: unlike an environment skip (no swift
    # toolchain, no apps/ dir, TRAIN_GATES_SKIP_*), this is a *deterministic*
    # property of the diff and always yields the same true outcome.
    passed_na 5 "apps/ unchanged vs $base_sha; the build's swift test step has nothing to run"
    return 0
  fi
  ( cd "$desktop_dir" && swift test --no-parallel ) || {
    fail 5 "swift test --no-parallel failed (see output above)"
    return 1
  }
  passed 5 "Desktop swift test passed"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "train-gates: base-sha resolved -> $BASE_SHA_RESOLVED"
echo "train-gates: head under test   -> $HEAD_RESOLVED"
echo "train-gates: gates-hash        -> $GATES_HASH"

# B2: never tolerate stale coverage artifacts from a previous run in the repo
# root — the next run against a NEW head would silently union old coverage into
# a false green. (All current-run artifacts live under $COV_DIR.)
for stale in "$ROOT"/coverage-*.data "$ROOT"/coverage.xml "$ROOT"/.coverage; do
  [[ -e "$stale" ]] && rm -f "$stale" && echo "train-gates: removed stale $stale"
done

# Design: even if one gate fails we keep going so the operator sees every
# failure in one pass. Each gate writes its own PASSED/FAILED/SKIPPED entry
# and returns non-zero on failure; run_gate ignores that code (set -e would
# otherwise abort the run) and the exit status comes from the FAILED array.
#
# Order matters: Gate 4 (Apple) must produce coverage-apple.data BEFORE Gate 3
# (coverage union + diff-cover) combines it, so the run order is 1 -> 2 -> 4 ->
# 3 -> 5.
run_gate 1 gate1_linux
run_gate 2 gate2_mypy
run_gate 4 gate4_apple
run_gate 3 gate3_diffcover "$BASE_SHA_RESOLVED"
run_gate 5 gate5_swift "$BASE_SHA_RESOLVED"

echo
printf 'PASSED (%d): %s\n' "${#PASSED[@]}" "${PASSED[*]:-none}"
if [[ "${#SKIPPED[@]}" -gt 0 ]]; then
  printf 'SKIPPED (%d): %s\n' "${#SKIPPED[@]}" "${SKIPPED[*]}"
fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  KEEP_RUN_TMP=1
  printf 'FAILED (%d): %s\n' "${#FAILED[@]}" "${FAILED[*]}"
  echo
  echo "GATES FAILED — a gate below blocked the freeze:"
  for g in "${FAILED[@]}"; do
    echo "  FAILURE gate $g"
  done
  exit 1
fi

# The freeze contract requires all 5 gates to have PASSED (skips are NOT
# passes; PASS-BY-N/A counts).
if [[ "${#PASSED[@]}" -ne 5 ]]; then
  echo "GATES INCOMPLETE — ${#PASSED[@]}/5 passed (${#SKIPPED[@]} skipped). A skip is not a pass; rerun to exercise all gates." >&2
  exit 1
fi

# A dirty tree never earns the literal `GATES OK`: the bytes that were tested
# are not the committed HEAD. Distinct receipt, distinct exit code (3).
if [[ -n "$GIT_DIRTY" ]]; then
  echo
  echo "GATES DIRTY ${HEAD_RESOLVED} ${GATES_HASH}  (python ${PY_VERSION})"
  echo "train-gates: all 5 gates passed, but tracked files are modified/staged (see the WARNING above);" >&2
  echo "             this is NOT a freeze receipt — commit or stash, then rerun for the OK receipt." >&2
  exit 3
fi

echo
echo "GATES OK ${HEAD_RESOLVED} ${GATES_HASH}  (python ${PY_VERSION})"
