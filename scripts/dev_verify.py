#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One local verification command: from a diff or a Desktop report to evidence.

Turns a worktree diff (``--diff-base`` / ``--paths-file``) or a Desktop issue
form area (``--area``) into a reviewable verification plan, optionally executes
the selected existing checks, and records a commit-bound machine-readable
result. It is the local, pre-PR counterpart of the CI lane classifiers and the
Desktop GUI golden gate; it never invents new checks of its own.

Selection is delegated to the production routers so local and CI agreement is
structural, not aspirational:

- ``scripts/classify_ci_changes.py`` decides the engine/desktop/docs lanes.
- ``scripts/select_gui_flows.py`` decides which Desktop GUI journeys a changed
  path implicates (fail-closed: unknown paths select everything).
- ``apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml`` is the journey source
  of truth; Desktop issue areas map onto it through ``AREA_JOURNEYS`` below.

Fail-closed rules (a partial or blocked run is never green):

- A missing/invalid diff, an unknown area, or an unknown journey is a usage
  error (exit 2), never an empty plan.
- Any selected check that fails, is blocked by a missing prerequisite, or was
  not run makes the overall result ``fail`` (exit 1).
- Plan-only mode writes the same result with ``status: "plan-only"``; a plan
  is never reported as a pass.

Evidence layout (default under ``rapid-dev-verify/``, git-ignored)::

    <out>/
      verify-result.json          this run's contract artifact
      logs/<check-id>.log         full stdout/stderr per check
      journeys/<name>/            the GUI harness artifact dir, which
        result.json                 contains the harness's own verdict

Ordinary verification never downloads model weights: GUI journeys run against
the bundled fake sidecar, and the pytest selection excludes the ``slow``,
``integration``, and ``needle`` markers via ``pytest.ini``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# Direct execution (`python scripts/dev_verify.py`) puts scripts/ on sys.path
# instead of the repo root; the production routers are imported as
# `scripts.*` in both execution styles.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml"
DESKTOP_BUG_FORM = ROOT / ".github/ISSUE_TEMPLATE/desktop_bug.yml"
DESKTOP_DIR = "apps/rapid-mac"
GUI_APP = f"{DESKTOP_DIR}/build/Rapid-MLX Desktop.app"
EVIDENCE_PARENT = ROOT / "rapid-dev-verify"

# The engine commands mirror the existing documented dev entry point
# (scripts/dev_test.py lint/unit), which mirrors the CI lint/unit jobs.
LINT_COMMAND = "ruff check . && ruff format --check ."
UNIT_COMMAND = (
    f"{sys.executable} -m pytest tests/ -q --ignore=tests/integrations"
    " --deselect tests/test_event_loop.py"
    " --deselect tests/test_batching_deterministic.py"
    " --deselect tests/test_reasoning_parsers.py"
)

# The Desktop contract gates the mac CI runs as fast pytest jobs. Kept in one
# check because they are seconds-fast and share one pytest process.
DESKTOP_CONTRACT_TESTS = [
    "tests/test_rapid_mac_ax_identifiers.py",
    "tests/test_ax_baseline_os_variance.py",
    "tests/test_gui_preflight_contract.py",
    "tests/test_gui_control_behavior_contract.py",
    "tests/test_gui_golden_ci_coverage.py",
    "tests/test_gui_flow_routing.py",
    "tests/test_gui_walk_completeness.py",
    "tests/test_fake_sidecar_image_catalog.py",
]
SWIFT_TEST_WRAPPER = f"{DESKTOP_DIR}/scripts/desktop-test-timeout.sh"

# CI parity for GUI journeys: the mac CI runners always carry a CI marker and
# DO_NOT_TRACK=1, which turns app telemetry (and its first-run notice banner)
# off. A local shell has neither, so the banner would appear in the AX tree and
# fail structural baselines that were captured without it. Journeys whose
# personas deliberately exercise telemetry lift this themselves (the harness
# sets RAPID_MLX_TELEMETRY=1/DO_NOT_TRACK=0 for those personas).
TELEMETRY_NEUTRAL_ENV = {
    "RAPID_MLX_TELEMETRY": "0",
    "DO_NOT_TRACK": "1",
}

# Environment keys that must not leak from the invoking shell into a journey:
# a stale GUI_FLOWS makes the harness SKIP and exit 0 without evidence, and a
# stale RAPID_GUI_SOURCE_APP / RAPID_GUI_BASELINE_DIR would bind evidence to an
# app or baseline that is not this HEAD's. The plan's own --flow / --build
# inputs are the only routing authority here.
SHELL_LEAK_ENV_KEYS = (
    "GUI_FLOWS",
    "RAPID_GUI_SOURCE_APP",
    "RAPID_GUI_BASELINE_DIR",
)
SWIFT_CONTRACT_SCRIPTS = [
    f"swift {DESKTOP_DIR}/scripts/verify-recommendation-tiers.swift",
    f"{DESKTOP_DIR}/scripts/verify-conversation-ordering.sh",
    f"swift {DESKTOP_DIR}/scripts/verify-chat-timeout.swift",
]

SERVER_CHAT_SOURCE_PREFIXES = (
    f"{DESKTOP_DIR}/Sources/Rapid/Server/",
    f"{DESKTOP_DIR}/Sources/Rapid/Chat/",
)


@dataclass(frozen=True)
class AreaMapping:
    """Desktop issue-form area to candidate journeys plus coverage gaps.

    Areas yield *candidates*: an issue form choice is a hint about where in
    the app a report belongs, not proof of a unique cause. ``gaps`` names what
    the automated journeys cannot demonstrate for that area so an agent never
    mistakes candidate coverage for proof.
    """

    journeys: tuple[str, ...]
    gaps: tuple[str, ...] = ()


# Keys must equal the `area` dropdown options in .github/ISSUE_TEMPLATE/
# desktop_bug.yml exactly; tests/test_dev_verify_contracts.py enforces both
# directions, and that every referenced journey exists in journeys.yaml.
AREA_JOURNEYS: dict[str, AreaMapping] = {
    "First-run setup wizard": AreaMapping(
        ("fresh-install", "cached-quickstart", "cached-curated-tradeup"),
    ),
    "Chat window (asking, answering, tools, formatting)": AreaMapping(
        (
            "chat-restore",
            "chat-document-attachment",
            "chat-multimodal-attachments",
            "math-rendering",
            "slow-stream-stop",
            "restored-tools",
            "tool-loop-budget",
            "message-actions",
        ),
    ),
    "Conversation list / sidebar (rename, pin, archive, delete)": AreaMapping(
        ("chat-restore",),
        gaps=(
            "no journey drives sidebar rename/pin/archive/delete directly;"
            " chat-restore covers persisted conversation state only",
        ),
    ),
    "Models — downloading, picking, or starting one": AreaMapping(
        (
            "download-progress",
            "cached-variant-collapse",
            "model-switch-active-request",
            "low-memory-choice",
            "catalog-integrity",
            "resident-load-rejected",
            "model-crash-recovery",
        ),
    ),
    "Settings": AreaMapping(
        (
            "settings-persistence",
            "settings-mtp",
            "no-dead-controls",
            "browse-all-destination",
        ),
    ),
    "Menu-bar icon / Dock / window behaviour": AreaMapping(
        ("window-close-prompt", "campaign-banner", "launch-integrations"),
    ),
    "Updating the app": AreaMapping(("update-state", "update-busy")),
    "Installing or launching for the first time": AreaMapping(
        ("fresh-install", "launch-integrations", "cached-quickstart"),
        gaps=(
            "no automated journey validates a DMG's code signature or"
            " notarisation; release validation is a manual procedure",
        ),
    ),
    "Speed — answers arrive slowly, or the app feels sluggish": AreaMapping(
        ("model-switch-active-request", "slow-stream-stop"),
        gaps=(
            "no automated latency journey; reproduce with the bench harness"
            " and record timings instead of assuming a UI cause",
        ),
    ),
    "Somewhere else / not sure": AreaMapping(
        (  # broad coverage: every PR-tier bash-driver journey
            "fresh-install",
            "cached-quickstart",
            "cached-curated-tradeup",
            "cached-variant-collapse",
            "download-progress",
            "model-switch-active-request",
            "settings-persistence",
            "settings-mtp",
            "chat-restore",
            "chat-document-attachment",
            "chat-multimodal-attachments",
            "image-generation",
            "dictation",
            "dictation-rc2-upgrade",
            "audio-readiness",
            "model-crash-recovery",
            "low-memory-choice",
            "update-state",
            "update-busy",
            "campaign-banner",
            "window-close-prompt",
            "no-dead-controls",
            "catalog-integrity",
            "browse-all-destination",
            "resident-load-rejected",
            "launch-integrations",
        ),
        gaps=(
            "broad candidate coverage by design; narrow with --journey once"
            " reproduction points at a specific journey",
        ),
    ),
}


@dataclass
class Check:
    """One executable verification command with its selection contract."""

    id: str
    kind: str
    command: str
    why: str
    cwd: str = "."
    env: dict[str, str] = field(default_factory=dict)
    prerequisites: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    # Filled by the executor:
    status: str = "planned"  # planned|pass|fail|blocked
    exit_code: int | None = None
    duration_seconds: float | None = None
    log_path: str | None = None
    blocked_reason: str | None = None


@dataclass
class Plan:
    lanes: dict[str, bool]
    flows: list[str]
    flow_reasons: dict[str, str]
    area: str | None
    area_gaps: list[str]
    notes: list[str]
    checks: list[Check]
    input_mode: str
    paths: list[str] = field(default_factory=list)
    evidence_dir: Path = Path(".")
    explicit_journeys: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Repository inputs
# --------------------------------------------------------------------------


def load_manifest() -> list[dict[str, object]]:
    payload = yaml.safe_load(MANIFEST.read_text())
    assert payload["version"] == 1, "unsupported GUI journey manifest version"
    journeys: list[dict[str, object]] = list(payload["journeys"])
    if not journeys:
        raise ValueError("GUI journey manifest contains no journeys")
    return journeys


def journey_index() -> dict[str, dict[str, object]]:
    return {str(j["name"]): j for j in load_manifest()}


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def head_sha() -> str:
    return git_output("rev-parse", "HEAD").strip()


def resolve_diff(diff_base: str) -> tuple[list[str], str]:
    """Changed paths a PR from this worktree would carry, plus the base SHA.

    Includes committed changes since the merge base plus uncommitted worktree
    changes and untracked (non-ignored) files: a repair loop usually edits
    before committing, and an untracked new Desktop source file must still
    fail closed in routing.
    """
    base_sha = git_output("merge-base", diff_base, "HEAD").strip()
    paths: set[str] = set()
    for args in (
        ("diff", "--name-only", f"{base_sha}..HEAD"),
        ("diff", "--name-only", "HEAD"),
        ("ls-files", "--others", "--exclude-standard"),
    ):
        paths.update(line.strip() for line in git_output(*args).splitlines())
    paths.discard("")
    if not paths:
        raise RuntimeError(
            f"no changed paths between {diff_base} and HEAD (including the"
            " worktree); refusing to produce a green no-op"
        )
    return sorted(paths), base_sha


def resolve_paths_file(paths_file: Path) -> list[str]:
    try:
        raw = paths_file.read_text()
    except OSError as exc:
        raise RuntimeError(f"cannot read paths file: {exc}") from exc
    paths = sorted(
        {line.strip().removeprefix("./") for line in raw.splitlines() if line.strip()}
    )
    if not paths:
        raise RuntimeError(
            f"paths file {paths_file} lists no paths; refusing to produce a green no-op"
        )
    return paths


# --------------------------------------------------------------------------
# Plan construction
# --------------------------------------------------------------------------


def _bash_flow_journeys(flows: list[str]) -> list[str]:
    """PR-tier journeys the bash GUI harness can drive (excludes swift)."""
    index = journey_index()
    return [
        name
        for name in flows
        if index[name]["ci_tier"] == "pr" and index[name]["driver"] != "swift"
    ]


def _implicated_swift_journeys(flows: list[str]) -> list[str]:
    """PR-tier in-process journeys owned by the groups the selection hit.

    The bash router never emits `driver: swift` journeys (they run inside
    `swift test`, not the harness), so ownership must be derived from the
    selected manifest groups: a changed Chat file implicates the chat group's
    in-process suites exactly as much as its harness flows.
    """
    index = journey_index()
    groups = {str(index[name]["group"]) for name in flows}
    return sorted(
        name
        for name, journey in index.items()
        if journey["driver"] == "swift"
        and journey["ci_tier"] == "pr"
        and str(journey["group"]) in groups
    )


def _broad_selection(flows: list[str]) -> bool:
    return set(flows) == set(select_all_flows())


def swift_filter_command(journeys: list[str]) -> str:
    """A `swift test --filter` regex for the named golden suites.

    Journey names are contractually `[a-z0-9-]+` (journeys.yaml naming and
    the harness dispatcher), so they are regex-literal as-is; `re.escape`
    would quote the hyphen and Swift Testing rejects `\\-`.
    """
    for name in journeys:
        if not re.fullmatch(r"[a-z0-9-]+", name):
            raise RuntimeError(f"journey name is not filter-safe: {name}")
    pattern = "|".join(journeys)
    return f'swift test --filter "Golden journey: ({pattern})"'


def journey_command(name: str) -> str:
    """The harness invocation for one journey, injection-safe.

    journeys.yaml is PR-controlled content: a malicious journey name must
    never reach a shell string, where it would execute before the harness's
    own `case` validation could reject it. Fail closed on anything outside
    the manifest's contractual `[a-z0-9-]+` namespace.
    """
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise RuntimeError(f"journey name is not a safe command argument: {name!r}")
    return f"./scripts/gui-golden-flows.sh --flow {name}"


def select_all_flows() -> list[str]:
    """Import lazily so the router stays the single selection authority."""
    from scripts.select_gui_flows import all_flows

    return all_flows()


def _checks_for_diff_mode(
    lanes: dict[str, bool], flows: list[str], evidence_dir: Path
) -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    notes: list[str] = []

    if lanes["engine"] or lanes["docs_only"]:
        why = (
            "docs-only diff: the CI lint job still runs for this head"
            if lanes["docs_only"]
            else "engine lane changed; mirrors the CI lint job"
        )
        checks.append(
            Check(
                id="engine:lint",
                kind="lint",
                command=LINT_COMMAND,
                why=why,
                prerequisites=["ruff installed in the active python env"],
            )
        )
    if lanes["engine"]:
        checks.append(
            Check(
                id="engine:unit",
                kind="engine-tests",
                command=UNIT_COMMAND,
                why="engine lane changed; mirrors the dev_test.py unit tier"
                " (slow/integration/needle stay excluded via pytest.ini)",
                prerequisites=["python env with dev dependencies installed"],
            )
        )

    if lanes["desktop"]:
        checks.append(
            Check(
                id="desktop:contracts",
                kind="gui-contracts",
                command=f"{sys.executable} -m pytest {' '.join(DESKTOP_CONTRACT_TESTS)} -q",
                why="Desktop lane changed; these are the mac CI contract gates"
                " for routing, AX identifiers, preflight, and snapshots",
                prerequisites=["pytest and pyyaml installed"],
            )
        )

        swift_journeys = _implicated_swift_journeys(flows)
        broad = _broad_selection(flows)
        swift_relevant = broad or bool(swift_journeys)
        if swift_relevant:
            if broad:
                command = "./scripts/desktop-test-timeout.sh"
                why = (
                    "broad/fail-closed Desktop selection; runs the full"
                    " in-process suite exactly like the mac CI swift job"
                )
            else:
                command = swift_filter_command(swift_journeys)
                why = (
                    "changed paths implicate in-process golden journeys:"
                    f" {', '.join(swift_journeys)}"
                )
            checks.append(
                Check(
                    id="desktop:swift-test",
                    kind="swift-test",
                    command=command,
                    why=why,
                    cwd=DESKTOP_DIR,
                    prerequisites=["Xcode swift toolchain", "several minutes"],
                )
            )

        index = journey_index()
        touches_server_or_chat = broad or any(
            any(
                str(prefix).startswith(SERVER_CHAT_SOURCE_PREFIXES)
                for prefix in index[name]["source_paths"]  # type: ignore[attr-defined]
            )
            for name in flows
        )
        if touches_server_or_chat:
            checks.append(
                Check(
                    id="desktop:swift-contracts",
                    kind="swift-contracts",
                    command=" && ".join(SWIFT_CONTRACT_SCRIPTS),
                    why="selection touches Server/Chat surfaces; these are the"
                    " standalone recommendation/ordering/timeout contracts the"
                    " mac CI swift job runs",
                    prerequisites=["Xcode swift toolchain"],
                )
            )

        for name in _bash_flow_journeys(flows):
            out_dir = evidence_dir / "journeys" / name
            checks.append(
                Check(
                    id=f"journey:{name}",
                    kind="gui-journey",
                    command=journey_command(name),
                    why=f"changed paths route here via journeys.yaml"
                    f" ({index[name]['group']} group)",
                    cwd=DESKTOP_DIR,
                    env={
                        "RAPID_GUI_GOLDEN_OUT": str(out_dir),
                        **TELEMETRY_NEUTRAL_ENV,
                    },
                    prerequisites=[
                        f"built app at {GUI_APP} (or --build)",
                        "jq on PATH",
                        "Accessibility permission for the driving terminal",
                    ],
                    evidence_paths=[str(out_dir / "result.json")],
                )
            )
        if not any(check.kind == "gui-journey" for check in checks):
            notes.append(
                "Desktop lane selected but no bash-driver GUI journey is"
                " implicated (swift-driver journeys run under"
                " desktop:swift-test)"
            )
    return checks, notes


def _checks_for_area_mode(
    flows: list[str], explicit_journeys: list[str], evidence_dir: Path
) -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    checks.append(
        Check(
            id="desktop:contracts",
            kind="gui-contracts",
            command=f"{sys.executable} -m pytest {' '.join(DESKTOP_CONTRACT_TESTS)} -q",
            why="a Desktop report is being verified; contract gates must hold"
            " before any journey is trusted",
            prerequisites=["pytest and pyyaml installed"],
        )
    )

    swift_journeys = _implicated_swift_journeys(flows)
    if swift_journeys:
        checks.append(
            Check(
                id="desktop:swift-test",
                kind="swift-test",
                command=swift_filter_command(swift_journeys),
                why="selection includes in-process golden journeys:"
                f" {', '.join(swift_journeys)}",
                cwd=DESKTOP_DIR,
                prerequisites=["Xcode swift toolchain", "several minutes"],
            )
        )

    index = journey_index()
    for name in _bash_flow_journeys(flows):
        out_dir = evidence_dir / "journeys" / name
        reason = "candidate for the reported area"
        if name in explicit_journeys:
            reason = "explicitly requested via --journey"
        checks.append(
            Check(
                id=f"journey:{name}",
                kind="gui-journey",
                command=journey_command(name),
                why=f"{reason} ({index[name]['group']} group)",
                cwd=DESKTOP_DIR,
                env={
                    "RAPID_GUI_GOLDEN_OUT": str(out_dir),
                    **TELEMETRY_NEUTRAL_ENV,
                },
                prerequisites=[
                    f"built app at {GUI_APP} (or --build)",
                    "jq on PATH",
                    "Accessibility permission for the driving terminal",
                ],
                evidence_paths=[str(out_dir / "result.json")],
            )
        )
    return checks, []


def build_plan(args: argparse.Namespace, evidence_dir: Path | None = None) -> Plan:
    explicit_journeys = list(args.journey or [])
    known = journey_index()
    for name in explicit_journeys:
        if name not in known:
            raise RuntimeError(
                f"unknown journey: {name}; known journeys: {', '.join(sorted(known))}"
            )

    area: str | None = None
    area_gaps: list[str] = []
    paths: list[str] = []
    lanes = {"engine": False, "desktop": False, "docs_only": False}
    flows: list[str] = []
    flow_reasons: dict[str, str] = {}
    notes: list[str] = []

    if args.area is not None:
        area = args.area
        if area not in AREA_JOURNEYS:
            raise RuntimeError(
                f"unknown Desktop area: {area}; valid areas: "
                + "\n  ".join(sorted(AREA_JOURNEYS))
            )
        mapping = AREA_JOURNEYS[area]
        flows = list(mapping.journeys)
        area_gaps = list(mapping.gaps)
        lanes["desktop"] = True
        flow_reasons = {name: f"candidate for area '{area}'" for name in flows}
        notes.append(
            "An issue area yields candidates, not a proven cause; narrow with"
            " --journey once reproduction points at one journey."
        )
    elif args.diff_base or args.paths_file:
        if args.diff_base:
            paths, _base_sha = resolve_diff(args.diff_base)
        else:
            paths = resolve_paths_file(args.paths_file)

        # Local/CI agreement is structural: the production classifiers and
        # router decide lanes and journeys, exactly as the workflows do.
        from scripts.classify_ci_changes import classify
        from scripts.select_gui_flows import select

        lanes_raw = classify(paths)
        lanes = {
            "engine": lanes_raw.engine,
            "desktop": lanes_raw.desktop,
            "docs_only": lanes_raw.docs_only,
        }
        flows = select(paths) if lanes["desktop"] else []
        flow_reasons = {
            name: "routed from changed paths by select_gui_flows" for name in flows
        }
        if lanes["docs_only"] and not lanes["engine"] and not lanes["desktop"]:
            notes.append(
                "docs-only diff: no product lanes; CI still runs the lint"
                " job for this head."
            )
        if explicit_journeys:
            # An explicit journey request is a promise the plan must keep;
            # a docs-only lane must not silently swallow it.
            lanes["desktop"] = True
    else:
        # Explicit-journey-only mode still verifies a Desktop surface.
        lanes["desktop"] = True

    for name in explicit_journeys:
        if name not in flows:
            flows.append(name)
        flow_reasons[name] = "explicitly requested via --journey"
        if known[name]["driver"] == "swift":
            notes.append(
                f"{name} is a driver:swift journey; it is verified by the"
                " in-process swift suite, not the bash GUI harness."
            )
        elif known[name]["ci_tier"] == "local":
            notes.append(
                f"{name} is a local-tier journey (not part of the CI GUI"
                " shards); running it locally is valid extra evidence."
            )

    evidence_dir = evidence_dir or (
        Path(args.out) if args.out else default_evidence_dir()
    )
    if not evidence_dir.is_absolute():
        # A relative --out must resolve against the repo root, not the
        # process cwd: journey checks run with cwd=apps/rapid-mac, and a
        # cwd-relative artifact path would strand the evidence inside the
        # app build tree with a wrong recorded location.
        evidence_dir = ROOT / evidence_dir
    if args.diff_base or args.paths_file:
        checks, diff_notes = _checks_for_diff_mode(lanes, flows, evidence_dir)
        notes.extend(diff_notes)
    else:
        # Area mode and explicit-journey mode share the same shape: Desktop
        # contract gates plus candidate/explicit journeys, no engine checks.
        checks, area_notes = _checks_for_area_mode(
            flows, explicit_journeys, evidence_dir
        )
        notes.extend(area_notes)

    if args.build and any(check.kind == "gui-journey" for check in checks):
        checks.insert(
            0,
            Check(
                id="desktop:build-app",
                kind="build-app",
                command="./scripts/build.sh",
                why="journey checks need a commit-bound built app",
                cwd=DESKTOP_DIR,
                env={
                    "SKIP_SIDECAR": "1",
                    "RAPID_BUILD_CONFIG": "release",
                    "RAPID_MLX_TELEMETRY": "0",
                    "DO_NOT_TRACK": "1",
                },
                prerequisites=["Xcode swift toolchain", "several minutes"],
            ),
        )

    plan = Plan(
        lanes=lanes,
        flows=flows,
        flow_reasons=flow_reasons,
        area=area,
        area_gaps=area_gaps,
        notes=notes,
        checks=checks,
        input_mode=(
            "area"
            if area is not None
            else "diff-base"
            if args.diff_base
            else "paths-file"
            if args.paths_file
            else "explicit-journey"
        ),
        paths=paths,
        evidence_dir=evidence_dir,
        explicit_journeys=explicit_journeys,
    )
    return plan


def default_evidence_dir() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    short = head_sha()[:12]
    return EVIDENCE_PARENT / f"{stamp}-{short}"


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _tool_missing(check: Check) -> str | None:
    if check.kind == "gui-journey" and not shutil.which("jq"):
        return "jq is not on PATH (required by the GUI harness)"
    if check.kind in ("swift-test", "swift-contracts", "build-app") and not (
        shutil.which("swift") or shutil.which("xcodebuild")
    ):
        return "no swift toolchain on PATH"
    if check.kind == "gui-journey" and not (ROOT / GUI_APP).exists():
        return (
            f"no built app at {GUI_APP}; run scripts/dev_verify.py --build"
            " or apps/rapid-mac/scripts/build.sh first"
        )
    return None


def parse_journey_result(result_path: Path) -> tuple[str, str]:
    """Return (status, detail) from the GUI harness's own result.json."""
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return "fail", f"unreadable harness result.json: {exc}"
    status = str(payload.get("status", ""))
    if status not in ("pass", "fail"):
        return "fail", f"harness reported non-executable status: {status!r}"
    return status, f"harness status={status} flow={payload.get('flow')}"


def execute_checks(
    checks: list[Check], evidence_dir: Path, log_dir: Path | None = None
) -> None:
    """Run checks sequentially, writing incremental evidence.

    A missing tool or artifact is ``blocked``, never skipped silently: the
    overall result stays red whenever anything is not a verified pass.
    """
    logs = log_dir or (evidence_dir / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    build_not_verified = False
    for check in checks:
        if check.kind == "gui-journey" and build_not_verified:
            check.status = "blocked"
            check.blocked_reason = (
                "app build did not verify in this run; refusing to record"
                " journey evidence against a possibly-stale build"
            )
            continue
        blocked = _tool_missing(check)
        if blocked:
            check.status = "blocked"
            check.blocked_reason = blocked
            if check.kind == "build-app":
                # A build the caller asked for but that could not run leaves
                # whatever binary is on disk unverified — same hazard as a
                # failing build.
                build_not_verified = True
            continue
        if check.kind == "gui-journey":
            out_dir = Path(check.env["RAPID_GUI_GOLDEN_OUT"])
            shutil.rmtree(out_dir, ignore_errors=True)
        log_path = logs / f"{check.id.replace(':', '__')}.log"
        started = time.monotonic()
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in SHELL_LEAK_ENV_KEYS
        }
        env.update(check.env)
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.run(
                    check.command,
                    shell=True,
                    cwd=ROOT / check.cwd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            check.exit_code = proc.returncode
        except OSError as exc:
            check.status = "fail"
            check.blocked_reason = f"failed to launch: {exc}"
            check.log_path = str(log_path)
            continue
        check.duration_seconds = round(time.monotonic() - started, 3)
        check.log_path = str(log_path)
        if check.exit_code != 0:
            check.status = "fail"
            if check.kind == "build-app":
                build_not_verified = True
            continue
        if check.kind == "gui-journey":
            status, detail = parse_journey_result(
                Path(check.env["RAPID_GUI_GOLDEN_OUT"]) / "result.json"
            )
            check.status = status
            if status != "pass":
                check.blocked_reason = detail
        else:
            check.status = "pass"
        # A blocked build (for example no swift toolchain) is just as stale
        # as a failed one: the caller asked for THIS head's app, so journeys
        # must not proceed on whatever binary happens to be on disk.
        if check.kind == "build-app" and check.status != "pass":
            build_not_verified = True


def _mark_interrupted(plan: Plan, reason: str = "interrupted before running") -> None:
    """Leave no check in a limbo "planned" state after an aborted run."""
    for check in plan.checks:
        if check.status == "planned":
            check.status = "fail"
            check.blocked_reason = reason


def result_payload(
    plan: Plan,
    evidence_dir: Path,
    sha: str,
    diff_base: str,
    base_sha: str,
    mode: str,
) -> dict[str, object]:
    overall = (
        "plan-only"
        if mode == "plan"
        else (
            "pass"
            if plan.checks and all(c.status == "pass" for c in plan.checks)
            else "fail"
        )
    )
    return {
        "schema": "rapid-mlx/dev-verify/result/v1",
        "status": overall,
        "mode": mode,
        "head_sha": sha,
        "head_subject": git_output("log", "-1", "--format=%s").strip(),
        "diff_base": diff_base,
        "base_sha": base_sha,
        "input": {
            "mode": plan.input_mode,
            "area": plan.area,
            "paths": plan.paths,
            "explicit_journeys": plan.explicit_journeys,
        },
        "lanes": plan.lanes,
        "selection": {
            "flows": plan.flows,
            "flow_reasons": plan.flow_reasons,
            "area_gaps": plan.area_gaps,
            "notes": plan.notes,
        },
        "evidence_dir": str(evidence_dir),
        "checks": [
            {
                "id": c.id,
                "kind": c.kind,
                "status": c.status,
                "exit_code": c.exit_code,
                "duration_seconds": c.duration_seconds,
                "command": c.command,
                "why": c.why,
                "prerequisites": c.prerequisites,
                "log_path": c.log_path,
                "blocked_reason": c.blocked_reason,
                "artifact_paths": (
                    c.evidence_paths + ([c.log_path] if c.log_path else [])
                ),
            }
            for c in plan.checks
        ],
        "artifacts": ["verify-result.json", "logs/"],
    }


def write_result(payload: dict[str, object], evidence_dir: Path) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / "verify-result.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(target)
    return target


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def render_plan(plan: Plan) -> str:
    lines: list[str] = []
    lanes = ", ".join(name for name, enabled in plan.lanes.items() if enabled) or "none"
    lines.append(f"input mode : {plan.input_mode}")
    if plan.area:
        lines.append(f"area       : {plan.area}")
    if plan.paths:
        shown = plan.paths if len(plan.paths) <= 12 else plan.paths[:12] + ["…"]
        lines.append(f"paths      : {', '.join(shown)}")
    lines.append(f"lanes      : {lanes}")
    if plan.flows:
        lines.append("GUI journeys:")
        for name in plan.flows:
            lines.append(f"  - {name}: {plan.flow_reasons[name]}")
    if plan.area_gaps:
        lines.append("coverage gaps (candidates are not proof):")
        for gap in plan.area_gaps:
            lines.append(f"  ! {gap}")
    for note in plan.notes:
        lines.append(f"note: {note}")
    lines.append("checks:")
    for check in plan.checks:
        lines.append(f"  [{check.id}] {check.status}")
        lines.append(f"      cmd: {check.command}")
        lines.append(f"      why: {check.why}")
        for prereq in check.prerequisites:
            lines.append(f"      needs: {prereq}")
        for artifact in check.evidence_paths:
            lines.append(f"      evidence: {artifact}")
    return "\n".join(lines)


def render_summary(payload: dict[str, object]) -> str:
    lines = [f"result: {payload['status']}  ({payload['evidence_dir']})"]
    checks = payload["checks"]
    for check in checks if isinstance(checks, list) else []:  # type: ignore[index]
        line = f"  {check['status']:>7}  {check['id']}"
        if check["blocked_reason"]:
            line += f" — {check['blocked_reason']}"
        if check["log_path"]:
            line += f"  [log: {check['log_path']}]"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--diff-base",
        help="git ref to diff against (merge base + worktree changes)",
    )
    inputs.add_argument(
        "--paths-file", type=Path, help="file of changed repository paths"
    )
    inputs.add_argument(
        "--area",
        help="Desktop issue form area (see .github/ISSUE_TEMPLATE/desktop_bug.yml)",
    )
    parser.add_argument(
        "--journey",
        action="append",
        help="explicit journey to include (repeatable; validates against"
        " journeys.yaml)",
    )
    parser.add_argument(
        "--plan", action="store_true", help="print the plan and run nothing"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="build the Desktop app first when a GUI journey is selected",
    )
    parser.add_argument("--out", type=Path, help="evidence directory")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)

    if not (args.diff_base or args.paths_file or args.area or args.journey):
        parser.error(
            "one of --diff-base, --paths-file, --area, or --journey is"
            " required; an empty plan can never be verified"
        )

    try:
        plan = build_plan(args)
    except (
        RuntimeError,
        ValueError,
        AssertionError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        # A malformed manifest or unreadable repo state is a usage error too:
        # the agent gets a clean, actionable message, never a traceback that
        # could be misread as a verification result.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    evidence_dir = plan.evidence_dir
    diff_base = args.diff_base or ""
    base_sha = ""
    if args.diff_base:
        base_sha = git_output("merge-base", args.diff_base, "HEAD").strip()

    if args.plan:
        payload = result_payload(
            plan, evidence_dir, head_sha(), diff_base, base_sha, "plan"
        )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(render_plan(plan))
            target = write_result(payload, evidence_dir)
            print(f"\nplan recorded: {target}")
        return 0

    result_target = write_result(
        result_payload(plan, evidence_dir, head_sha(), diff_base, base_sha, "running"),
        evidence_dir,
    )
    try:
        execute_checks(plan.checks, evidence_dir)
    except KeyboardInterrupt:
        _mark_interrupted(plan)
        payload = result_payload(
            plan, evidence_dir, head_sha(), diff_base, base_sha, "execute"
        )
        payload["status"] = "fail"
        write_result(payload, evidence_dir)
        return 1
    except Exception:
        # An unexpected executor crash must still leave a red, inspectable
        # result — never a result frozen at "running".
        _mark_interrupted(plan, reason="command crashed; see logs")
        payload = result_payload(
            plan, evidence_dir, head_sha(), diff_base, base_sha, "execute"
        )
        payload["status"] = "fail"
        write_result(payload, evidence_dir)
        raise
    payload = result_payload(
        plan, evidence_dir, head_sha(), diff_base, base_sha, "execute"
    )
    write_result(payload, evidence_dir)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_summary(payload))
        print(f"evidence: {result_target}")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
