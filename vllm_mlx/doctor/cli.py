# SPDX-License-Identifier: Apache-2.0
"""CLI entry points for ``rapid-mlx doctor``."""

from __future__ import annotations

import sys

from .baseline import (
    DeltaStatus,
    compare,
    has_regression,
    load_baseline,
    load_thresholds,
    render_deltas_md,
    safe_model_slug,
    save_baseline,
)
from .runner import REPO_ROOT, CheckResult, DoctorRunner, Status

# Default model used by the check tier.  Tier 3 (full) loops a wider list.
DEFAULT_CHECK_MODEL = "qwen3.5-4b"

# Model list for the full tier: small/medium/large + a non-Qwen template
# path so we cover both Hermes-style (Qwen) and Gemma's distinct chat
# template + tool-call format.
DEFAULT_FULL_MODELS = ["qwen3.5-4b", "qwen3.5-35b", "qwen3.6-35b", "gemma-4-26b"]

# Agent profiles to exercise per-model in the full tier.  None ⇒ all
# loaded profiles.  Limit here if a particular profile is too slow to
# include in the regular full sweep.
DEFAULT_FULL_AGENT_PROFILES = None


def _require_source_checkout() -> None:
    """Doctor depends on tests/ + harness/ + pyproject.toml."""
    sentinels = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "tests",
        REPO_ROOT / "harness",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in sentinels if not p.exists()]
    if missing:
        print(
            "[doctor] this command requires a source checkout of Rapid-MLX.\n"
            f"        missing: {', '.join(missing)}\n"
            "        Clone https://github.com/raullenchai/Rapid-MLX and run "
            "from the repo root.",
            file=sys.stderr,
        )
        sys.exit(2)


def doctor_command(args) -> None:
    """Dispatch to the requested tier."""
    _require_source_checkout()

    tier = getattr(args, "tier", None) or "smoke"
    update_baselines = getattr(args, "update_baselines", False)

    # --update-baselines only makes sense for tiers that diff against a
    # baseline.  Warn explicitly when it's used with smoke (no perf
    # metrics) or benchmark (scorecard, no baseline contract) so users
    # don't think they recorded something they didn't.
    if update_baselines and tier in ("smoke", "benchmark"):
        print(
            f"[doctor] --update-baselines has no effect for tier '{tier}' "
            "(only check / full record baselines); ignoring.",
            file=sys.stderr,
        )
        update_baselines = False

    if tier == "smoke":
        result = run_smoke_tier()
    elif tier == "check":
        result = run_check_tier(
            model=getattr(args, "model", None) or DEFAULT_CHECK_MODEL,
            update_baselines=update_baselines,
        )
    elif tier == "full":
        result = run_full_tier(
            models=getattr(args, "models", None) or DEFAULT_FULL_MODELS,
            update_baselines=update_baselines,
        )
    elif tier == "benchmark":
        result = run_benchmark_tier(
            models=getattr(args, "models", None),
        )
    else:
        print(f"[doctor] unknown tier: {tier}", file=sys.stderr)
        sys.exit(2)

    sys.exit(result.exit_code)


# ---------------------------------------------------------------------
# Smoke tier
# ---------------------------------------------------------------------


def run_smoke_tier():
    """Static + import + CLI sanity. No model required."""
    from .checks import smoke

    print("Rapid-MLX Doctor — smoke tier")
    print("=" * 60)

    runner = DoctorRunner(tier="smoke")
    runner.run_check("repo_layout", smoke.check_repo_layout)
    runner.run_check("imports", smoke.check_imports)
    runner.run_check("ruff", smoke.check_ruff)
    runner.run_check("cli_sanity", smoke.check_cli_sanity)
    runner.run_check("pytest", smoke.check_pytest_unit)
    return runner.finalize()


# ---------------------------------------------------------------------
# Check tier
# ---------------------------------------------------------------------


def run_check_tier(model: str, update_baselines: bool = False):
    """Boot a server with ``model`` and run API + perf + agent checks."""
    from .checks import smoke

    print(f"Rapid-MLX Doctor — check tier (model={model})")
    print("=" * 60)

    runner = DoctorRunner(tier="check")

    # Cheap static checks first — fail fast on broken syntax / missing files.
    runner.run_check("repo_layout", smoke.check_repo_layout)
    runner.run_check("imports", smoke.check_imports)

    _run_per_model_block(
        runner=runner,
        model=model,
        tier="check",
        update_baselines=update_baselines,
        agent_profiles=[],  # check tier keeps it lean — agents only in full
    )
    return runner.finalize()


# ---------------------------------------------------------------------
# Full tier
# ---------------------------------------------------------------------


def run_full_tier(models: list[str], update_baselines: bool = False):
    """Loop check-tier work across multiple models + run all agent profiles."""
    from .checks import smoke

    print(f"Rapid-MLX Doctor — full tier (models={', '.join(models)})")
    print("=" * 60)

    runner = DoctorRunner(tier="full")

    runner.run_check("repo_layout", smoke.check_repo_layout)
    runner.run_check("imports", smoke.check_imports)

    # Resolve agent profile list once so a missing profile fails fast.
    profile_names = _resolve_agent_profiles(DEFAULT_FULL_AGENT_PROFILES)

    for model in models:
        print(f"\n  ── model: {model} ──")
        _run_per_model_block(
            runner=runner,
            model=model,
            tier="full",
            update_baselines=update_baselines,
            agent_profiles=profile_names,
            # boot_timeout_s=None → _suggested_boot_timeout picks 600s for
            # the 27B+ models (qwen3.5-35b, gemma-4-26b) and 180s for
            # smaller ones, so the same logic applies regardless of which
            # tier called us.
        )

    return runner.finalize()


def _resolve_agent_profiles(explicit: list[str] | None) -> list[str]:
    """Return the list of agent profile names to exercise.

    Loud failure on profile-loading errors: silently degrading to
    ``["generic"]`` made full-tier reports look successful while
    actually exercising 1/11 of the documented profile coverage.
    """
    if explicit:
        return explicit
    try:
        from .. import agents

        return [p.name for p in agents.list_profiles()]
    except Exception as e:
        print(
            f"[doctor] WARNING: agent profile loading failed "
            f"({type(e).__name__}: {e}); falling back to 'generic' only. "
            "Full-tier results will not reflect the documented 11-profile sweep.",
            file=sys.stderr,
        )
        return ["generic"]


# ---------------------------------------------------------------------
# Shared per-model block (used by check + full)
# ---------------------------------------------------------------------

# Default boot timeout for any model.  600s is generous enough for cold
# loads of 27B-122B models from a slow external SSD, which is the realistic
# worst case on a developer Mac.  Server crashes (port-bind failure, missing
# weights, import error, ...) are detected within <1s by proc.poll() in
# server._wait_for_health regardless of this value, so the only scenario
# where 600s actually waits 600s is a genuinely slow legitimate load — at
# which point we want the long budget.
#
# Earlier iterations tried to pick a tier-/alias-aware shorter budget for
# the small-model case, but every heuristic missed at least one supported
# large model (Qwen3-Coder lacks a 'NNb' hint in its alias, MiniMax M2.5
# is huge but named 'minimax-m2.5', etc.).  The optimisation isn't worth
# the false-fail risk.
DEFAULT_BOOT_TIMEOUT_S = 600


def _run_per_model_block(
    runner: DoctorRunner,
    model: str,
    tier: str,
    update_baselines: bool,
    agent_profiles: list[str],
    boot_timeout_s: int | None = None,
) -> None:
    """Boot a server for one model and run all model-bound checks against it.

    Failures here do NOT abort the rest of the tier — the runner just
    records the failure and moves on, so a single broken model in the
    full sweep still yields a complete report.

    ``boot_timeout_s`` defaults to ``DEFAULT_BOOT_TIMEOUT_S`` (600s).
    See the constant's docstring for why a single generous value beats
    every per-model / per-tier heuristic we tried.
    """
    from .checks import agent, api, perf, stress
    from .server import ServerStartFailed, serve

    if boot_timeout_s is None:
        boot_timeout_s = DEFAULT_BOOT_TIMEOUT_S

    server_log = runner.run_dir / f"server-{safe_model_slug(model)}.log"
    try:
        with serve(
            model=model,
            log_path=server_log,
            boot_timeout_s=boot_timeout_s,
        ) as info:
            port = info["port"]
            print(f"  [server] {model} up on port {port}, log → {server_log.name}")
            runner.run_check(
                f"regression_suite[{model}]",
                lambda: api.check_regression_suite(port),
            )
            runner.run_check(
                f"smoke_matrix[{model}]",
                lambda: api.check_smoke_matrix(port),
            )
            runner.run_check(
                f"stress[{model}]",
                lambda: stress.check_stress(port),
            )
            perf_result = runner.run_check(
                f"autoresearch[{model}]",
                lambda: perf.check_autoresearch(port, runs=1),
            )
            for profile_name in agent_profiles:
                # Bind per-iteration values so the lambda doesn't close
                # over the loop variables.
                runner.run_check(
                    f"agent[{profile_name}@{model}]",
                    lambda p=profile_name, port=port: agent.check_agent_profile(
                        p, port, model_id=model
                    ),
                )
    except ServerStartFailed as exc:
        err_msg = f"{exc}\nlog: {server_log}"
        runner.run_check(
            f"server_boot[{model}]",
            lambda: CheckResult(
                name=f"server_boot[{model}]",
                status=Status.FAIL,
                duration_s=0.0,
                detail=err_msg,
            ),
        )
        return

    # Compare perf metrics against baseline (if one exists).
    if perf_result.metrics:
        _apply_baseline(runner, tier, model, perf_result.metrics, update_baselines)


# NOTE: per-model artefact filenames (diff-{model}.md, server-{model}.log)
# use safe_model_slug from baseline.py so the mapping is injective —
# a non-injective scheme reintroduces the same overwrite bug per-model
# diffs were meant to fix.


# ---------------------------------------------------------------------
# Benchmark tier
# ---------------------------------------------------------------------


def run_benchmark_tier(models: list[str] | None = None):
    """Cross-model scorecard.  Auto-skips models not present locally.

    For each model:
      1. Boot a server (Simple engine — broadest compatibility)
      2. Run autoresearch_bench, capture decode/ttft/tool-call metrics
      3. Tear down

    Aggregates into harness/scorecard/scorecard-{ts}.md plus a
    'latest.md' alias.  Designed to be runnable overnight — no agent
    profile sweep, no baseline diff, just numbers for the table.
    """
    from .checks import benchmark, smoke
    from .discovery import discover_local_models, load_aliases
    from .scorecard import render_scorecard, write_scorecard
    from .server import ServerStartFailed, serve

    print("Rapid-MLX Doctor — benchmark tier")
    print("=" * 60)

    runner = DoctorRunner(tier="benchmark")
    runner.run_check("repo_layout", smoke.check_repo_layout)
    runner.run_check("imports", smoke.check_imports)

    # Resolve which models to run.  Without --models, sweep every alias
    # whose weights are already on disk.  Explicit --models still get
    # an availability check so a typo doesn't trigger a multi-GB
    # background download mid-sweep — explicit-but-missing aliases land
    # in the skipped list with a clear reason instead.
    skipped: list[tuple[str, str]] = []
    discovery = {m.alias: m for m in discover_local_models()}
    aliases_set = set(load_aliases())

    if models:
        run_list = []
        for alias in models:
            if alias not in aliases_set:
                skipped.append((alias, "unknown alias (not in aliases.json)"))
            elif alias in discovery and discovery[alias].available:
                run_list.append(alias)
            else:
                reason = (
                    discovery[alias].reason
                    if alias in discovery
                    else "weights not on disk"
                )
                skipped.append((alias, f"skipped (no auto-download): {reason}"))
    else:
        run_list = [m.alias for m in discovery.values() if m.available]
        skipped = [(m.alias, m.reason) for m in discovery.values() if not m.available]

    if not run_list:
        if skipped:
            print("\n  No models to benchmark.  Skipped:")
            for alias, reason in skipped:
                print(f"    - {alias}: {reason}")
        else:
            print(
                "\n  no models available locally — pre-fetch with "
                "`huggingface-cli download`,"
            )
            print("  or pass --models <alias>,<alias> to force.")
        skipped_summary = "; ".join(f"{a} ({r})" for a, r in skipped[:5])
        detail = f"no runnable models — {len(skipped)} skipped" + (
            f": {skipped_summary}" if skipped_summary else ""
        )
        runner.run_check(
            "discovery",
            lambda: CheckResult(
                name="discovery",
                status=Status.FAIL,
                duration_s=0.0,
                detail=detail,
            ),
        )
        return runner.finalize()

    print(
        f"\n  Will benchmark {len(run_list)} model(s); "
        f"{len(skipped)} skipped (not local)"
    )

    cells: list[tuple[str, CheckResult]] = []
    for model in run_list:
        print(f"\n  ── model: {model} ──")
        server_log = runner.run_dir / f"server-{safe_model_slug(model)}.log"
        # Pass the discovered local path so LM Studio / non-HF layouts
        # don't trigger a re-download via mlx_lm.load(alias).
        local_path = (
            discovery[model].path
            if model in discovery and discovery[model].available
            else None
        )
        try:
            # Cold load of a 27B+ model can take several minutes when
            # the OS has to page weights from a slow disk; default
            # 180s timeout is too tight for an overnight sweep.
            with serve(
                model=model,
                model_path=local_path,
                log_path=server_log,
                boot_timeout_s=600,
            ) as info:
                port = info["port"]
                print(f"  [server] {model} up on port {port}")
                result = runner.run_check(
                    f"bench[{model}]",
                    lambda port=port, model=model: benchmark.benchmark_one_cell(
                        model, port, runs=1
                    ),
                )
                cells.append((model, result))
        except ServerStartFailed as exc:
            err_msg = f"{exc}\nlog: {server_log}"
            result = runner.run_check(
                f"bench[{model}]",
                lambda err_msg=err_msg, model=model: CheckResult(
                    name=f"bench[{model}]",
                    status=Status.FAIL,
                    duration_s=0.0,
                    detail=f"server boot failed: {err_msg.splitlines()[0]}",
                ),
            )
            cells.append((model, result))
        except Exception as exc:  # noqa: BLE001 — never abort the whole sweep
            result = runner.run_check(
                f"bench[{model}]",
                lambda exc=exc, model=model: CheckResult(
                    name=f"bench[{model}]",
                    status=Status.FAIL,
                    duration_s=0.0,
                    detail=f"unexpected error: {type(exc).__name__}: {exc}",
                ),
            )
            cells.append((model, result))

    # Aggregate scorecard.
    scorecard_md = render_scorecard(cells, skipped=skipped)
    scorecard_path = write_scorecard(scorecard_md)
    # Also drop a copy in the run_dir for self-containment.
    (runner.run_dir / "scorecard.md").write_text(scorecard_md)

    print(f"\n  Scorecard: {scorecard_path}")
    return runner.finalize()


def _apply_baseline(
    runner: DoctorRunner,
    tier: str,
    model: str,
    metrics: dict,
    update_baselines: bool,
) -> None:
    """Diff against baseline; flag regressions; optionally update baseline.

    Baselines are per-model — comparing decode TPS for a 4B and a 35B
    model is meaningless.  ``--update-baselines`` writes the model-
    specific file, so each model accumulates its own history.
    """
    baseline = load_baseline(tier, model)
    thresholds = load_thresholds()

    if update_baselines:
        path = save_baseline(tier, model, metrics)
        runner.run_check(
            f"baseline_update[{model}]",
            lambda: CheckResult(
                name=f"baseline_update[{model}]",
                status=Status.PASS,
                duration_s=0.0,
                detail=f"wrote {path.relative_to(REPO_ROOT)}",
            ),
        )
        return

    if baseline is None:
        # First run with no baseline — record what we saw, don't fail.
        # Still write a diff.md (and per-model variant) so external
        # automation that always reads run_dir/diff.md gets a stable
        # artefact, not a missing-file error.
        notice = (
            f"_no baseline yet for model={model} — "
            "run with --update-baselines to record one_\n"
        )
        per_model_diff = runner.run_dir / f"diff-{safe_model_slug(model)}.md"
        per_model_diff.write_text(notice)
        combined_diff = runner.run_dir / "diff.md"
        section = f"## {model}\n\n{notice}\n"
        if combined_diff.exists():
            with open(combined_diff, "a") as f:
                f.write(section)
        else:
            combined_diff.write_text(section)

        runner.run_check(
            f"baseline_diff[{model}]",
            lambda: CheckResult(
                name=f"baseline_diff[{model}]",
                status=Status.SKIP,
                duration_s=0.0,
                detail=f"no baseline found for model={model}; "
                "run --update-baselines to create",
            ),
        )
        return

    # Defence-in-depth: per-model file paths should already prevent this,
    # but a manually copied/renamed file could mix model identities.
    baseline_model = baseline.get("model")
    if baseline_model and baseline_model != model:
        runner.run_check(
            f"baseline_diff[{model}]",
            lambda: CheckResult(
                name=f"baseline_diff[{model}]",
                status=Status.FAIL,
                duration_s=0.0,
                detail=(
                    f"baseline model mismatch: file has model={baseline_model!r} "
                    f"but current run is model={model!r}. Refusing to compare."
                ),
            ),
        )
        return

    deltas = compare(metrics, baseline, thresholds)
    deltas_md = render_deltas_md(deltas)

    # In multi-model tiers (full / benchmark), each call would clobber a
    # shared diff.md and we'd lose all earlier models' delta tables.
    # Always write per-model files; also append to a combined diff.md
    # so single-model tiers (check) keep their existing artefact name.
    per_model_diff = runner.run_dir / f"diff-{safe_model_slug(model)}.md"
    per_model_diff.write_text(deltas_md)

    combined_diff = runner.run_dir / "diff.md"
    section = f"## {model}\n\n{deltas_md}\n"
    if combined_diff.exists():
        with open(combined_diff, "a") as f:
            f.write(section)
    else:
        combined_diff.write_text(section)

    n_regress = sum(1 for d in deltas if d.status == DeltaStatus.REGRESSION)
    n_improve = sum(1 for d in deltas if d.status == DeltaStatus.IMPROVEMENT)
    detail = (
        f"{len(deltas)} metrics: {n_regress} regression(s), {n_improve} improvement(s)"
    )

    status = Status.REGRESSION if has_regression(deltas) else Status.PASS
    # Include model in check name so multi-model tiers don't collide.
    runner.run_check(
        f"baseline_diff[{model}]",
        lambda: CheckResult(
            name=f"baseline_diff[{model}]",
            status=status,
            duration_s=0.0,
            detail=detail,
        ),
    )

    # Stash the full delta table on the runner so finalize() can append
    # it to report.md as a section.  Stuffing it into CheckResult.detail
    # would only survive in result.json — _render_markdown truncates
    # long detail strings to 120 chars.
    if not hasattr(runner, "_pending_diff_sections"):
        runner._pending_diff_sections = []
    runner._pending_diff_sections.append((model, deltas_md))
