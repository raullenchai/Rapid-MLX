"""Adversarial operator contracts: remote state, ambiguity and source freshness."""

import argparse
import copy
import json
import plistlib
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import release_control as target

SOURCE, BUMP = "a" * 40, "b" * 40


@pytest.fixture
def plan(tmp_path):
    return dict(
        schema=1,
        repository=target.REPOSITORY,
        version="0.16.1",
        source_sha=SOURCE,
        bump_sha=BUMP,
        base="main",
        branch="release/prepare-0.16.1",
        worktree=str(tmp_path / "worktree"),
        dispatches={},
        pr=42,
    )


def workflow_run(stage="parent-dry-run", **kwargs):
    result = dict(
        id=10,
        run_attempt=1,
        path=".github/workflows/" + target.WORKFLOWS[stage][0],
        event="workflow_dispatch",
        head_sha=SOURCE if stage == "parent-dry-run" else BUMP,
        head_branch="main" if stage == "parent-dry-run" else "release/prepare-0.16.1",
        status="completed",
        conclusion="success",
        created_at="2026-10-09T00:00:00Z",
        html_url=f"https://github.com/{target.REPOSITORY}/actions/runs/10",
    )
    result.update(kwargs)
    return result


def workflow_jobs(stage="parent-dry-run"):
    jobs = [
        dict(
            name=n,
            status="completed",
            conclusion="success",
            html_url="https://github.com/job",
        )
        for n in target.WORKFLOWS[stage][1]
    ]
    if stage == "parent-dry-run":
        jobs.extend(
            dict(
                name=n,
                status="completed",
                conclusion="skipped",
                html_url="https://github.com/job",
            )
            for n in ("release", "release-prep")
        )
    return jobs


class FakeGitHub:
    def __init__(self, plan):
        self.plan = plan
        self.source = SOURCE
        self.items = {"auto-release.yml": [], "release-preflight.yml": []}
        self.calls = []
        self.fail_dispatch = False
        self.prs = []
        self.pr = dict(
            number=42,
            state="open",
            title="chore: bump version to 0.16.1",
            commits=1,
            head=dict(
                sha=BUMP, ref=plan["branch"], repo=dict(full_name=target.REPOSITORY)
            ),
            base=dict(ref="main", sha=SOURCE),
            body="## Why\nCurated rationale.\n",
        )

    def ref(self, branch):
        return self.source if branch == "main" else BUMP

    def runs(self, workflow, commit):
        return self.items[workflow]

    def jobs(self, item):
        return workflow_jobs(
            "preflight" if "preflight" in item["path"] else "parent-dry-run"
        )

    def dispatch(self, workflow, ref, inputs):
        self.calls.append((workflow, ref, inputs))
        if self.fail_dispatch:
            raise target.OperatorError("transport timeout, result unknown")

    def get(self, path, **kwargs):
        if path.startswith("pulls?"):
            return self.prs
        if path == "pulls":
            self.calls.append(("create", kwargs))
        return copy.deepcopy(self.pr)


def test_required_job_names_match_canonical_workflows():
    root = Path(__file__).parents[1]
    for stage, (workflow, names) in target.WORKFLOWS.items():
        data = yaml.safe_load((root / ".github/workflows" / workflow).read_text())
        assert names <= data["jobs"].keys(), stage


def test_successful_evidence_and_skipped_publication():
    assert (
        target.inspect_run("parent-dry-run", workflow_run(), workflow_jobs())["state"]
        == "success"
    )
    assert (
        target.inspect_run(
            "preflight", workflow_run("preflight"), workflow_jobs("preflight")
        )["state"]
        == "success"
    )


@pytest.mark.parametrize("state", ["queued", "in_progress", "action_required"])
def test_pending_and_security_approval_are_not_green(state):
    result = target.inspect_run(
        "preflight", workflow_run("preflight", status=state, conclusion=None), []
    )
    assert result["state"] == (
        "approval-required" if state == "action_required" else "waiting"
    )


@pytest.mark.parametrize(
    "conclusion", ["failure", "cancelled", "timed_out", "skipped", None]
)
def test_terminal_negative_never_green(conclusion):
    result = target.inspect_run(
        "preflight", workflow_run("preflight", conclusion=conclusion), []
    )
    assert result["state"] == "failed"


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "skipped", "running", "publication"]
)
def test_aggregate_green_is_insufficient(mutation):
    jobs = workflow_jobs()
    if mutation == "missing":
        jobs = jobs[1:]
    elif mutation == "duplicate":
        jobs.append(jobs[0].copy())
    elif mutation == "skipped":
        jobs[0]["conclusion"] = "skipped"
    elif mutation == "running":
        jobs[0]["status"] = "in_progress"
    else:
        next(j for j in jobs if j["name"] == "release")["conclusion"] = "success"
    assert (
        target.inspect_run("parent-dry-run", workflow_run(), jobs)["state"]
        == "invalid-evidence"
    )


def test_wrong_workflow_rejects():
    with pytest.raises(target.OperatorError, match="workflow"):
        target.inspect_run("preflight", workflow_run(), [])


def test_dispatch_intent_survives_ambiguous_transport_and_never_replays(plan, tmp_path):
    gh = FakeGitHub(plan)
    gh.fail_dispatch = True
    path = tmp_path / "state.json"
    with pytest.raises(target.OperatorError, match="timeout"):
        target.ensure_stage(path, plan, "parent-dry-run", gh)
    persisted = json.loads(path.read_text())
    assert persisted["dispatches"]["parent-dry-run"]["sha"] == SOURCE
    assert (
        target.ensure_stage(path, persisted, "parent-dry-run", gh)["state"]
        == "dispatch-unconfirmed"
    )
    assert len(gh.calls) == 1
    gh.items["auto-release.yml"] = [workflow_run()]
    assert (
        target.ensure_stage(path, persisted, "parent-dry-run", gh)["state"] == "success"
    )
    assert len(gh.calls) == 1


def test_normal_dispatch_does_not_publish_or_force(plan, tmp_path):
    gh = FakeGitHub(plan)
    result = target.ensure_stage(tmp_path / "state.json", plan, "preflight", gh)
    assert result["state"] == "dispatched"
    assert gh.calls == [
        (
            "release-preflight.yml",
            plan["branch"],
            {"pr_number": "42", "expected_sha": BUMP, "target_branch": "main"},
        )
    ]


def test_newest_red_is_not_hidden_by_old_green(plan):
    gh = FakeGitHub(plan)
    gh.items["auto-release.yml"] = [
        workflow_run(id=11, conclusion="failure"),
        workflow_run(),
    ]
    assert target.stage_status(plan, "parent-dry-run", gh)["state"] == "failed"


def test_dispatched_cycle_cannot_reuse_prior_run(plan):
    gh = FakeGitHub(plan)
    plan["dispatches"]["parent-dry-run"] = {"prior_run_ids": [10]}
    gh.items["auto-release.yml"] = [workflow_run()]
    assert (
        target.stage_status(plan, "parent-dry-run", gh)["state"]
        == "dispatch-unconfirmed"
    )


def test_foreign_branch_is_not_adopted(plan):
    gh = FakeGitHub(plan)
    gh.items["auto-release.yml"] = [workflow_run(head_branch="foreign")]
    assert target.stage_status(plan, "parent-dry-run", gh)["state"] == "not-started"


def test_main_advance_blocks_every_mutation(plan, tmp_path):
    gh = FakeGitHub(plan)
    gh.source = "c" * 40
    with pytest.raises(target.OperatorError, match="advanced"):
        target.resume(tmp_path / "state.json", plan, gh)
    assert not gh.calls


@pytest.mark.parametrize(
    "field,value", [("state", "closed"), ("commits", 2), ("title", "other")]
)
def test_bad_bump_pr_rejects(plan, field, value):
    gh = FakeGitHub(plan)
    gh.pr[field] = value
    with pytest.raises(target.OperatorError, match="PR"):
        target.checked_pr(plan, gh)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("head", "sha", "c" * 40),
        ("head", "ref", "other"),
        ("base", "ref", "foreign"),
        ("base", "sha", "c" * 40),
    ],
)
def test_changed_head_and_base_reject(plan, section, field, value):
    gh = FakeGitHub(plan)
    gh.pr[section][field] = value
    with pytest.raises(target.OperatorError):
        target.checked_pr(plan, gh)


def test_fork_pr_rejects(plan):
    gh = FakeGitHub(plan)
    gh.pr["head"]["repo"]["full_name"] = "foreign/repo"
    with pytest.raises(target.OperatorError):
        target.checked_pr(plan, gh)


def test_existing_pr_is_reconciled_without_push_or_create(plan, tmp_path):
    gh = FakeGitHub(plan)
    gh.prs = [gh.pr]
    assert target.publish_pr(tmp_path / "state.json", plan, gh)["number"] == 42
    assert not gh.calls
    gh.prs.append(gh.pr.copy())
    with pytest.raises(target.OperatorError, match="multiple"):
        target.publish_pr(tmp_path / "state.json", plan, gh)


def test_ambiguous_pr_creation_never_repeats(plan, tmp_path):
    gh = FakeGitHub(plan)
    plan["pr_issuance"] = {"issued_at": "earlier"}
    with pytest.raises(target.OperatorError, match="unconfirmed"):
        target.publish_pr(tmp_path / "state.json", plan, gh)
    assert not gh.calls


def test_resume_advances_existing_successes_only(plan, tmp_path, monkeypatch):
    gh = FakeGitHub(plan)
    gh.items["auto-release.yml"] = [workflow_run()]
    gh.items["release-preflight.yml"] = [workflow_run("preflight")]
    gh.prs = [gh.pr]
    attachments = []
    monkeypatch.setattr(
        target, "attach_preflight", lambda *args: attachments.append(args)
    )
    assert (
        target.resume(tmp_path / "state.json", plan, gh)["stage"]
        == "preparation-complete"
    )
    assert len(attachments) == 1 and not gh.calls
    gh.items["release-preflight.yml"] = []
    assert target.resume(tmp_path / "state.json", plan, gh)["stage"] == "preflight"


def test_resume_does_not_create_pr_until_parent_dry_run_completes(plan, tmp_path):
    gh = FakeGitHub(plan)
    result = target.resume(tmp_path / "state.json", plan, gh)
    assert result["stage"] == "parent-dry-run" and result["state"] == "dispatched"
    assert gh.calls[0][2] == {"dry_run": "true"}


def test_attach_preserves_rationale_and_replaces_stale_line(plan, monkeypatch):
    gh = FakeGitHub(plan)
    gh.pr["body"] += "Release-Preflight: https://github.com/old/actions/runs/1\n"
    calls = []

    def invoke(args, **kwargs):
        calls.append((args, kwargs))
        gh.pr["body"] = json.loads(kwargs["input"])["body"]
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(target.subprocess, "run", invoke)
    status = dict(
        state="success",
        run_id=10,
        url=f"https://github.com/{target.REPOSITORY}/actions/runs/10",
    )
    target.attach_preflight(plan, gh, status)
    assert (
        "Curated rationale." in gh.pr["body"]
        and gh.pr["body"].count("Release-Preflight:") == 1
    )
    target.attach_preflight(plan, gh, status)
    assert len(calls) == 1
    with pytest.raises(target.OperatorError):
        target.attach_preflight(plan, gh, dict(status, state="failed"))


def test_plan_identity_and_sha_rejects(plan):
    target.validate_plan(plan, "0.16.1")
    for key, value in [
        ("schema", 2),
        ("version", "0.16.2"),
        ("repository", "foreign/repo"),
        ("branch", "other"),
        ("source_sha", "short"),
    ]:
        bad = dict(plan, **{key: value})
        with pytest.raises(target.OperatorError):
            target.validate_plan(bad, "0.16.1")
    assert target.sha(SOURCE) == SOURCE
    assert target.utcnow().endswith("Z")


def test_github_get_and_post_use_explicit_methods(monkeypatch):
    gh = target.GitHub()
    commands = []
    monkeypatch.setattr(
        target, "run", lambda *args: commands.append(args) or '[[],[{"id":1}]]'
    )
    assert gh.get("tags", pages=True) == [{"id": 1}]
    assert "--paginate" in commands[0] and "GET" in commands[0]
    monkeypatch.setattr(
        target.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(
            args, 0, stdout='{"id": 2}', stderr=""
        ),
    )
    assert gh.get("pulls", payload={"title": "test"}) == {"id": 2}
    with pytest.raises(target.OperatorError):
        target.GitHub("foreign/repo")


def test_pagination_includes_duplicate_required_jobs_on_page_two(monkeypatch):
    gh = target.GitHub()
    jobs = workflow_jobs()
    monkeypatch.setattr(
        target, "run", lambda *args: json.dumps([{"jobs": jobs}, {"jobs": [jobs[0]]}])
    )
    result = gh.jobs(workflow_run())
    assert len(result) == len(jobs) + 1
    assert (
        target.inspect_run("parent-dry-run", workflow_run(), result)["state"]
        == "invalid-evidence"
    )


def test_run_queries_validate_identity_and_latest_order(monkeypatch):
    gh = target.GitHub()
    rows = [workflow_run(id=10), workflow_run(id=11)]
    monkeypatch.setattr(
        target, "run", lambda *args: json.dumps([{"workflow_runs": rows}])
    )
    assert gh.runs("auto-release.yml", SOURCE)[0]["id"] == 11
    for key, value in [
        ("head_sha", BUMP),
        ("event", "push"),
        ("path", ".github/workflows/foreign.yml"),
    ]:
        rows[0][key] = value
        with pytest.raises(target.OperatorError):
            gh.runs("auto-release.yml", SOURCE)
        rows[0] = workflow_run()


@pytest.fixture
def git_case(tmp_path):
    """Use a real bare remote and two published tags; never contact GitHub."""
    root = tmp_path / "repository"
    root.mkdir()

    def git(*args):
        return target.run("git", *args, cwd=root)

    git("init", "-b", "seed")
    git("config", "user.name", "Release Test")
    git("config", "user.email", "release-test@example.invalid")
    files = {
        "pyproject.toml": '[project]\nname = "rapid-mlx"\nversion = "0.15.7"\n',
        target.PLIST: '<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundleVersion</key><string>182</string><key>CFBundleShortVersionString</key><string>0.15.7</string></dict></plist>\n',
        target.CHANGELOG: "# Changelog\n\n## [Unreleased]\n\n## [0.15.7] — 2026-10-07\n\nOld notes.\n\n[Unreleased]: https://github.com/raullenchai/Rapid-MLX/compare/rapid-mac-v0.15.7...HEAD\n[0.15.7]: https://github.com/raullenchai/Rapid-MLX/compare/rapid-mac-v0.15.6...rapid-mac-v0.15.7\n",
        "product.py": "VALUE = 1\n",
    }
    for name, content in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    git("add", ".")
    git("commit", "-m", "seed")
    source = git("rev-parse", "HEAD")
    git("tag", "rapid-mac-v0.15.7")
    new_files = target.render(
        project=files["pyproject.toml"],
        plist=files[target.PLIST],
        changelog=files[target.CHANGELOG],
        published_changelog=files[target.CHANGELOG],
        previous_version="0.15.7",
        version="0.16.0",
        previous_builds=[182],
        notes="# Frozen release\nNotes",
        highlights="### Highlights\n- Frozen release.",
        date="2026-10-09",
    )
    target.write_metadata(root, new_files, "0.16.0")
    git("add", ".")
    git("commit", "-m", "frozen release")
    git("tag", "rapid-mac-v0.16.0")
    git("checkout", "-b", "main", source)
    remote = tmp_path / "remote.git"
    git("clone", "--bare", str(root), str(remote))
    git("remote", "add", "origin", str(remote))
    args = argparse.Namespace(
        source=source,
        version="0.16.1",
        worktree=tmp_path / "bump",
        notes=tmp_path / "notes.md",
        highlights=tmp_path / "highlights.md",
    )
    args.notes.write_text("# Release 0.16.1\nCurated notes.\n")
    args.highlights.write_text("### Highlights\n- A user-visible change.\n")
    gh = FakeGitHub(dict(branch="release/prepare-0.16.1"))
    gh.source = source
    gh.tags = [{"name": "rapid-mac-v0.15.7"}, {"name": "rapid-mac-v0.16.0"}]
    gh.releases = [dict(draft=False, tag_name=t["name"]) for t in gh.tags]
    original_get = gh.get

    def get(path, **kwargs):
        if path.startswith("tags?"):
            return gh.tags
        if path.startswith("releases?"):
            return gh.releases
        return original_get(path, **kwargs)

    gh.get = get
    return args, tmp_path / "plan.json", gh, root, files


def test_prepare_real_git_single_commit_product_unchanged_and_resume(git_case):
    args, path, gh, root, source_files = git_case
    plan = target.prepare(args, path, gh, root)
    assert target.run("git", "rev-parse", "HEAD^", cwd=args.worktree) == args.source
    assert (
        target.run("git", "show", "-s", "--format=%s", cwd=args.worktree)
        == "chore: bump version to 0.16.1"
    )
    assert not target.run("git", "status", "--porcelain", cwd=args.worktree)
    assert (
        (root / "product.py").read_text()
        == (args.worktree / "product.py").read_text()
        == source_files["product.py"]
    )
    assert "Frozen release." in (args.worktree / target.CHANGELOG).read_text()
    assert (
        plistlib.loads((args.worktree / target.PLIST).read_bytes())["CFBundleVersion"]
        == "184"
    )
    assert target.prepare(args, path, gh, root) == plan
    args.notes.write_text("Different release intent")
    with pytest.raises(target.OperatorError, match="differs"):
        target.prepare(args, path, gh, root)


@pytest.mark.parametrize(
    "problem",
    [
        "stale",
        "engine-tag",
        "desktop-tag",
        "no-baseline",
        "existing-worktree",
        "invalid-build",
        "draft",
        "newer-engine",
    ],
)
def test_prepare_rejects_unsafe_state(git_case, problem, monkeypatch):
    args, path, gh, root, _ = git_case
    if problem == "stale":
        gh.source = "c" * 40
    elif problem.endswith("-tag"):
        gh.tags.append(
            {"name": ("v" if problem == "engine-tag" else "rapid-mac-v") + args.version}
        )
    elif problem == "draft":
        gh.releases.append(dict(draft=True, tag_name="v0.16.1"))
    elif problem == "newer-engine":
        gh.releases.append(dict(draft=False, tag_name="v0.17.0"))
    elif problem == "no-baseline":
        gh.releases = []
    elif problem == "existing-worktree":
        args.worktree.mkdir()
    elif problem == "invalid-build":
        real = target.run

        def bad(*cmd, **kwargs):
            out = real(*cmd, **kwargs)
            if cmd[1:3] == ("show", "rapid-mac-v0.15.7:" + target.PLIST):
                return out.replace("<string>182</string>", "<string>bad</string>")
            return out

        monkeypatch.setattr(target, "run", bad)
    with pytest.raises(target.OperatorError):
        target.prepare(args, path, gh, root)
    assert not path.exists()


def test_changed_staging_inventory_cannot_make_product_bump(git_case, monkeypatch):
    args, path, gh, root, _ = git_case
    real = target.run

    def injected(*cmd, **kwargs):
        out = real(*cmd, **kwargs)
        if cmd[1:4] == ("diff", "--cached", "--name-only"):
            return out + "\nproduct.py"
        return out

    monkeypatch.setattr(target, "run", injected)
    with pytest.raises(target.OperatorError, match="inventory"):
        target.prepare(args, path, gh, root)
    assert not path.exists()


def test_publish_real_local_branch_and_single_pr(git_case):
    args, path, gh, root, _ = git_case
    plan = target.prepare(args, path, gh, root)
    gh.pr["head"]["sha"] = plan["bump_sha"]
    gh.pr["base"]["sha"] = plan["source_sha"]
    gh.ref = lambda branch: plan["source_sha"] if branch == "main" else plan["bump_sha"]
    gh.items["auto-release.yml"] = [workflow_run(head_sha=plan["source_sha"])]
    assert target.publish_pr(path, plan, gh)["number"] == 42
    assert len(gh.calls) == 1 and gh.calls[0][0] == "create"
    assert (
        target.run(
            "git",
            "ls-remote",
            "origin",
            "refs/heads/" + plan["branch"],
            cwd=args.worktree,
        ).split()[0]
        == plan["bump_sha"]
    )
    gh.prs = [gh.pr]
    target.publish_pr(path, plan, gh)
    assert len(gh.calls) == 1


@pytest.mark.parametrize(
    "problem", ["dirty", "remote-conflict", "branch-moved", "dry-failed"]
)
def test_publication_rejects_drift_and_no_valid_dryrun(git_case, problem):
    args, path, gh, root, _ = git_case
    plan = target.prepare(args, path, gh, root)
    gh.pr["head"]["sha"] = plan["bump_sha"]
    gh.pr["base"]["sha"] = plan["source_sha"]
    gh.items["auto-release.yml"] = [workflow_run(head_sha=plan["source_sha"])]
    gh.ref = lambda branch: plan["source_sha"] if branch == "main" else plan["bump_sha"]
    if problem == "dirty":
        (args.worktree / "product.py").write_text("changed")
    elif problem == "remote-conflict":
        target.run(
            "git",
            "push",
            "origin",
            plan["source_sha"] + ":refs/heads/" + plan["branch"],
            cwd=root,
        )
    elif problem == "branch-moved":
        gh.ref = lambda branch: plan["source_sha"]
    elif problem == "dry-failed":
        gh.items["auto-release.yml"] = [workflow_run(conclusion="failure")]
    with pytest.raises(target.OperatorError):
        target.publish_pr(path, plan, gh)
    assert not gh.calls


def test_attach_body_race_transport_and_readback(plan, monkeypatch):
    status = dict(
        state="success",
        run_id=10,
        url=f"https://github.com/{target.REPOSITORY}/actions/runs/10",
    )
    gh = FakeGitHub(plan)
    count = 0
    old_get = gh.get

    def race(path, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            gh.pr["body"] = "Concurrent human edit"
        return old_get(path, **kwargs)

    gh.get = race
    with pytest.raises(target.OperatorError, match="concurrently"):
        target.attach_preflight(plan, gh, status)
    gh.get = old_get
    monkeypatch.setattr(
        target.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1),
    )
    with pytest.raises(target.OperatorError, match="update failed"):
        target.attach_preflight(plan, gh, status)
    monkeypatch.setattr(
        target.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 0),
    )
    with pytest.raises(target.OperatorError, match="readback"):
        target.attach_preflight(plan, gh, status)


def test_dispatch_ref_drift(plan, tmp_path):
    gh = FakeGitHub(plan)
    gh.ref = lambda branch: SOURCE
    with pytest.raises(target.OperatorError, match="dispatch branch"):
        target.ensure_stage(tmp_path / "state.json", plan, "preflight", gh)
    assert not gh.calls


def test_subprocess_and_api_errors_fail_closed(monkeypatch):
    monkeypatch.setattr(
        target.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, stderr="denied"),
    )
    with pytest.raises(target.OperatorError, match="denied"):
        target.run("git", "status")
    with pytest.raises(target.OperatorError, match="denied"):
        target.GitHub().get("pulls", payload={"title": "x"})
    monkeypatch.setattr(
        target, "run", lambda *args: json.dumps({"object": {"sha": SOURCE}})
    )
    assert target.GitHub().ref("main") == SOURCE
    monkeypatch.setattr(
        target.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 0, stdout="", stderr=""),
    )
    assert target.GitHub().api("dispatch", payload={}) is None
    calls = []
    gh = target.GitHub()
    monkeypatch.setattr(gh, "get", lambda *a, **kw: calls.append((a, kw)))
    gh.dispatch("preflight.yml", "main", {"sha": SOURCE})
    assert calls[0][1]["payload"]["inputs"] == {"sha": SOURCE}


@pytest.mark.parametrize("command", ["prepare", "resume", "status"])
def test_cli_routes_use_shared_git_state_and_lock(
    plan, tmp_path, monkeypatch, capsys, command
):
    common = tmp_path / "common"
    common.mkdir()
    states = common / "release-preparation"
    states.mkdir()
    target.save(states / "0.16.1.json", plan)
    gh = FakeGitHub(plan)
    monkeypatch.setattr(target, "GitHub", lambda: gh)
    monkeypatch.setattr(
        target,
        "run",
        lambda *a, **kw: (
            f"https://github.com/{target.REPOSITORY}.git"
            if "get-url" in a
            else str(common)
            if "--git-common-dir" in a
            else str(tmp_path)
        ),
    )
    monkeypatch.setattr(target, "prepare", lambda *a: plan)
    monkeypatch.setattr(target, "resume", lambda *a: {"stage": "waiting"})
    args = [command, "--version", "0.16.1"]
    if command == "prepare":
        args += [
            "--source",
            SOURCE,
            "--notes",
            "notes",
            "--highlights",
            "highlights",
            "--worktree",
            "bump",
        ]
    assert target.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result
    assert (states / "0.16.1.lock").is_file()


def test_cli_errors_are_actionable(capsys):
    assert target.main(["status", "--version", "../bad"]) == 1
    assert "blocked" in capsys.readouterr().err


def test_direct_cli_import_and_entrypoint(monkeypatch, capsys):
    import runpy
    import sys

    script = Path(__file__).parents[1] / "scripts" / "release_control.py"
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, "argv", [str(script), "status", "--version", "../bad"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")
    assert result.value.code == 1
    assert "invalid release version" in capsys.readouterr().err


def test_source_advance_during_baseline_fetch_does_not_create_worktree(git_case):
    args, path, gh, root, _ = git_case
    count = 0

    def ref(branch):
        nonlocal count
        count += 1
        return args.source if count == 1 else "c" * 40

    gh.ref = ref
    with pytest.raises(target.OperatorError, match="during preparation"):
        target.prepare(args, path, gh, root)
    assert not args.worktree.exists()


def test_cli_refuses_noncanonical_origin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        target,
        "run",
        lambda *args, **kwargs: (
            "https://github.com/foreign/repo.git"
            if "get-url" in args
            else str(tmp_path)
        ),
    )
    assert target.main(["status", "--version", "0.16.1"]) == 1
    assert "canonical release repository" in capsys.readouterr().err
