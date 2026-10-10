#!/usr/bin/env python3
"""Prepare a release bump and resume its canonical non-publishing checks.

This operator does not merge, tag, approve deployments or publish releases.
State is in the repository's common Git directory, shared across worktrees.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

try:
    from release_prepare import CHANGELOG, PLIST, render, write_metadata
    from release_version import VERSION_RE, parse_version
except ModuleNotFoundError:
    from scripts.release_prepare import CHANGELOG, PLIST, render, write_metadata
    from scripts.release_version import VERSION_RE, parse_version

REPOSITORY = "raullenchai/Rapid-MLX"
WORKFLOWS = {
    "parent-dry-run": (
        "auto-release.yml",
        {"tier1-agent-gate", "desktop-candidate-gate", "dry-run-summary"},
    ),
    "preflight": (
        "release-preflight.yml",
        {
            "bind-bump-pr",
            "pf1-release-contract",
            "pf2-release-secrets",
            "pf3-tag-environment",
            "g1-release-smoke",
            "g11-escape-hatch",
            "preflight-summary",
        },
    ),
}


class OperatorError(RuntimeError):
    pass


def run(*args: str, cwd: Path | None = None, raw: bool = False) -> str:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=120)
    if proc.returncode:
        raise OperatorError(
            proc.stderr.strip() or f"{args[0]} failed: {proc.returncode}"
        )
    return proc.stdout if raw else proc.stdout.strip()


class GitHub:
    def __init__(self, repository: str = REPOSITORY):
        if repository != REPOSITORY:
            raise OperatorError(f"release authority is restricted to {REPOSITORY}")
        self.repository = repository

    def api(self, path: str, *, payload: dict | None = None, pages: bool = False):
        args = ["gh", "api", "--method", "GET" if payload is None else "POST", path]
        if pages:
            args += ["--paginate", "--slurp"]
        if payload is None:
            raw = run(*args)
        else:
            proc = subprocess.run(
                [*args, "--input", "-"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=120,
            )
            if proc.returncode:
                raise OperatorError(proc.stderr.strip() or "GitHub mutation failed")
            raw = proc.stdout.strip()
        data = json.loads(raw) if raw else None
        if pages:
            return [item for page in data for item in page]
        return data

    def get(self, path: str, **kwargs):
        return self.api(f"repos/{self.repository}/{path}", **kwargs)

    def ref(self, branch: str) -> str:
        return sha(self.get(f"git/ref/heads/{quote(branch, safe='')}")["object"]["sha"])

    def runs(self, workflow: str, commit: str) -> list[dict]:
        # Slurp workflow-runs objects, not arrays. Never select an older green
        # result over a more recent failed/incomplete attempt.
        raw = run(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{self.repository}/actions/workflows/{workflow}/runs?head_sha={commit}&event=workflow_dispatch&per_page=100",
        )
        result = []
        for page in json.loads(raw):
            for item in page["workflow_runs"]:
                if item["head_sha"] != commit or item["event"] != "workflow_dispatch":
                    raise OperatorError(
                        "workflow query returned a foreign source/event"
                    )
                if item["path"].split("@")[0] != f".github/workflows/{workflow}":
                    raise OperatorError("workflow query returned a foreign workflow")
                result.append(item)
        return sorted(result, key=lambda r: (r["created_at"], r["id"]), reverse=True)

    def jobs(self, item: dict) -> list[dict]:
        raw = run(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{self.repository}/actions/runs/{item['id']}/attempts/{item['run_attempt']}/jobs?per_page=100",
        )
        return [job for page in json.loads(raw) for job in page["jobs"]]

    def dispatch(self, workflow: str, ref: str, inputs: dict) -> None:
        self.get(
            f"actions/workflows/{workflow}/dispatches",
            payload={"ref": ref, "inputs": inputs},
        )


def sha(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise OperatorError("expected a full lowercase 40-character source SHA")
    return value


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save(path: Path, plan: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n")
    temporary.replace(path)


def validate_plan(plan: dict, version: str) -> None:
    if (
        plan.get("schema") != 1
        or plan.get("repository") != REPOSITORY
        or plan.get("version") != version
    ):
        raise OperatorError("release plan identity does not match this invocation")
    sha(plan["source_sha"])
    sha(plan["bump_sha"])
    if plan["branch"] != f"release/prepare-{version}" or plan["base"] != "main":
        raise OperatorError("unexpected release branch/base")


def inspect_run(stage: str, item: dict, jobs: list[dict]) -> dict:
    workflow, required = WORKFLOWS[stage]
    if item["path"].split("@")[0] != f".github/workflows/{workflow}":
        raise OperatorError("selected run has wrong workflow")
    if item["status"] != "completed":
        state = (
            "approval-required" if item["status"] == "action_required" else "waiting"
        )
    elif item["conclusion"] != "success":
        state = "failed"
    else:
        names = [job["name"] for job in jobs]
        valid = all(
            names.count(name) == 1
            and next(j for j in jobs if j["name"] == name)["conclusion"] == "success"
            and next(j for j in jobs if j["name"] == name)["status"] == "completed"
            for name in required
        )
        if stage == "parent-dry-run":
            # Positive dry-run evidence must also prove no publication jobs ran.
            valid = valid and all(
                names.count(name) == 1
                and next(j for j in jobs if j["name"] == name)["conclusion"]
                == "skipped"
                for name in ("release-prep", "release")
            )
        state = "success" if valid else "invalid-evidence"
    return {
        "state": state,
        "run_id": item["id"],
        "attempt": item["run_attempt"],
        "url": item["html_url"],
        "failed_jobs": [
            {"name": j["name"], "conclusion": j["conclusion"], "url": j["html_url"]}
            for j in jobs
            if j["conclusion"] not in (None, "success", "skipped")
        ],
    }


def stage_status(plan: dict, stage: str, gh: GitHub) -> dict:
    commit = plan["source_sha"] if stage == "parent-dry-run" else plan["bump_sha"]
    branch = plan["base"] if stage == "parent-dry-run" else plan["branch"]
    candidates = [
        r for r in gh.runs(WORKFLOWS[stage][0], commit) if r["head_branch"] == branch
    ]
    issuance = plan.get("dispatches", {}).get(stage)
    if issuance:
        candidates = [r for r in candidates if r["id"] not in issuance["prior_run_ids"]]
    if candidates:
        selected = candidates[0]
        return inspect_run(stage, selected, gh.jobs(selected))
    return {"state": "dispatch-unconfirmed" if issuance else "not-started"}


def ensure_stage(path: Path, plan: dict, stage: str, gh: GitHub) -> dict:
    status = stage_status(plan, stage, gh)
    if status["state"] != "not-started":
        return status
    assert_source(plan, gh)
    commit = plan["source_sha"] if stage == "parent-dry-run" else plan["bump_sha"]
    branch = plan["base"] if stage == "parent-dry-run" else plan["branch"]
    if gh.ref(branch) != commit:
        raise OperatorError("dispatch branch no longer matches the pinned source")
    if stage == "preflight":
        checked_pr(plan, gh)
    # Write intent before calling a mutation; if transport is ambiguous, the
    # next resume only reconciles actual runs and NEVER repeats the request.
    prior = gh.runs(WORKFLOWS[stage][0], commit)
    plan.setdefault("dispatches", {})[stage] = {
        "issued_at": utcnow(),
        "prior_run_ids": [r["id"] for r in prior],
        "sha": commit,
        "branch": branch,
    }
    save(path, plan)
    inputs = (
        {"dry_run": "true"}
        if stage == "parent-dry-run"
        else {
            "pr_number": str(plan["pr"]),
            "expected_sha": commit,
            "target_branch": plan["base"],
        }
    )
    gh.dispatch(WORKFLOWS[stage][0], branch, inputs)
    return {"state": "dispatched", "sha": commit}


def assert_source(plan: dict, gh: GitHub) -> None:
    if gh.ref(plan["base"]) != plan["source_sha"]:
        raise OperatorError(
            "source branch advanced; pinned release is stale, no automatic rebase or dispatch"
        )


def checked_pr(plan: dict, gh: GitHub) -> dict:
    pr = gh.get(f"pulls/{plan['pr']}")
    if (
        pr["state"] != "open"
        or pr["head"]["sha"] != plan["bump_sha"]
        or pr["head"]["ref"] != plan["branch"]
        or pr["head"]["repo"]["full_name"] != REPOSITORY
        or pr["base"]["ref"] != plan["base"]
        or pr["base"]["sha"] != plan["source_sha"]
        or pr["title"] != f"chore: bump version to {plan['version']}"
        or pr["commits"] != 1
    ):
        raise OperatorError(
            "bump PR is closed, moved, foreign, or no longer the prepared single commit"
        )
    return pr


def publish_pr(path: Path, plan: dict, gh: GitHub) -> dict:
    assert_source(plan, gh)
    prs = gh.get(
        f"pulls?state=all&head=raullenchai:{quote(plan['branch'], safe='')}&base=main&per_page=100",
        pages=True,
    )
    if len(prs) > 1:
        raise OperatorError(
            "multiple release PRs found; resolve identity before continuing"
        )
    if prs:
        plan["pr"] = prs[0]["number"]
        save(path, plan)
        return checked_pr(plan, gh)
    if plan.get("pr_issuance"):
        raise OperatorError(
            "PR creation remains unconfirmed; inspect GitHub, do not blindly repeat"
        )
    root = Path(plan["worktree"])
    if run("git", "rev-parse", "HEAD", cwd=root) != plan["bump_sha"] or run(
        "git", "status", "--porcelain", cwd=root
    ):
        raise OperatorError("prepared worktree is dirty or has moved")
    remote = run("git", "ls-remote", "origin", f"refs/heads/{plan['branch']}", cwd=root)
    if remote:
        if remote.split()[0] != plan["bump_sha"]:
            raise OperatorError("remote release branch differs; refusing to force-push")
    else:
        run(
            "git",
            "push",
            "origin",
            f"{plan['bump_sha']}:refs/heads/{plan['branch']}",
            cwd=root,
        )
    if gh.ref(plan["branch"]) != plan["bump_sha"]:
        raise OperatorError("published branch identity mismatch")
    dry = stage_status(plan, "parent-dry-run", gh)
    if dry["state"] != "success":
        raise OperatorError("parent dry run no longer provides valid evidence")
    plan["pr_issuance"] = {"issued_at": utcnow()}
    save(path, plan)
    body = (
        "## Why\nPrepare a versioned release from an explicitly pinned source.\n\n"
        f"Source: {plan['source_sha']}\nParent dry run: {dry['url']}\n\n"
        "## Scope\nVersion/build metadata and curated release notes only.\n\n"
        "## Non-goals\nNo product changes, protection edits, tagging or publication in this PR.\n\n"
        "## Acceptance\nOne metadata-only commit, a strictly increasing published build baseline, "
        "synchronized notes and exact-head preflight evidence.\n\n"
        "## Verification\nMetadata validators and the exact-parent canonical dry run passed.\n\n"
        "## Behaviour delta\nRelease version/build and curated notes advance; product code is unchanged.\n\n"
        "## AI assistance disclosure\nThe release operator generated metadata and this PR body; "
        "release prose was supplied explicitly by the operator.\n\n"
        "## Test plan\n- [x] Metadata validators and published build baseline\n"
        "- [x] Exact-parent canonical dry run\n"
        "\nExact-head preflight: pending; operator attaches after success.\n"
        "Required source/candidate checks remain enforced by repository policy.\n"
    )
    pr = gh.get(
        "pulls",
        payload={
            "title": f"chore: bump version to {plan['version']}",
            "head": plan["branch"],
            "base": plan["base"],
            "body": body,
        },
    )
    plan["pr"] = pr["number"]
    save(path, plan)
    return checked_pr(plan, gh)


def attach_preflight(plan: dict, gh: GitHub, status: dict) -> None:
    pr = checked_pr(plan, gh)
    url = f"https://github.com/{REPOSITORY}/actions/runs/{status['run_id']}"
    if status["state"] != "success" or status["url"] != url:
        raise OperatorError("cannot attach unverified preflight evidence")
    lines = (pr["body"] or "").splitlines()
    lines = [line for line in lines if not line.startswith("Release-Preflight:")]
    body = "\n".join(lines).rstrip() + "\n\nRelease-Preflight: " + url + "\n"
    if body != pr["body"]:
        # Re-read immediately before the edit so unrelated PR body edits survive.
        if checked_pr(plan, gh)["body"] != pr["body"]:
            raise OperatorError(
                "PR body changed concurrently; resume to preserve that edit"
            )
        proc = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "PATCH",
                f"repos/{REPOSITORY}/pulls/{plan['pr']}",
                "--input",
                "-",
            ],
            input=json.dumps({"body": body}),
            text=True,
            capture_output=True,
            timeout=120,
        )
        if proc.returncode:
            raise OperatorError(
                "preflight body update failed; resume reconciles without redispatch"
            )
        updated = checked_pr(plan, gh)
        if updated["body"] != body:
            raise OperatorError("preflight evidence body readback mismatch")


def prepare(args, path: Path, gh: GitHub, root: Path) -> dict:
    source = sha(args.source)
    notes, highlights = args.notes.read_text(), args.highlights.read_text()
    identity = hashlib.sha256(json.dumps([notes, highlights]).encode()).hexdigest()
    if path.exists():
        plan = json.loads(path.read_text())
        validate_plan(plan, args.version)
        if plan["source_sha"] != source or plan["copy_sha256"] != identity:
            raise OperatorError("existing plan differs; no implicit replacement/rebase")
        return plan
    if gh.ref("main") != source:
        raise OperatorError(
            "prepare requires the exact live main SHA; arbitrary/frozen refs are not authorized by this operator"
        )
    tags = gh.get("tags?per_page=100", pages=True)
    for name in (f"v{args.version}", f"rapid-mac-v{args.version}"):
        if any(t["name"] == name for t in tags):
            raise OperatorError(f"release tag is already reserved: {name}")
    releases = gh.get("releases?per_page=100", pages=True)
    reserved = {f"v{args.version}", f"rapid-mac-v{args.version}"}
    if any(r["tag_name"] in reserved for r in releases):
        raise OperatorError("target version already has a release or draft")
    engine = [
        r["tag_name"][1:]
        for r in releases
        if not r["draft"]
        and r["tag_name"].startswith("v")
        and VERSION_RE.fullmatch(r["tag_name"][1:])
    ]
    if engine and parse_version(args.version) <= max(map(parse_version, engine)):
        raise OperatorError(
            "release version must exceed every published Engine version"
        )
    desktop = [
        r
        for r in releases
        if not r["draft"] and r["tag_name"].startswith("rapid-mac-v")
    ]
    if not desktop:
        raise OperatorError("no published Desktop baseline")
    latest = max(
        desktop, key=lambda r: parse_version(r["tag_name"].removeprefix("rapid-mac-v"))
    )
    previous = latest["tag_name"].removeprefix("rapid-mac-v")
    run("git", "fetch", "origin", source, cwd=root)
    # Explicit tag refspecs: fetch published baselines without overwriting tags.
    builds = []
    published_changelog = ""
    for release in desktop:
        tag = release["tag_name"]
        run("git", "fetch", "origin", f"refs/tags/{tag}:refs/tags/{tag}", cwd=root)
        info = plistlib.loads(run("git", "show", f"{tag}:{PLIST}", cwd=root).encode())
        value = info["CFBundleVersion"]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
            raise OperatorError(f"published tag {tag} has a nonnumeric build")
        builds.append(int(value))
        if tag == latest["tag_name"]:
            published_changelog = run("git", "show", f"{tag}:{CHANGELOG}", cwd=root)
    content = lambda p: run("git", "show", f"{source}:{p}", cwd=root, raw=True)
    files = render(
        project=content("pyproject.toml"),
        plist=content(PLIST),
        changelog=content(CHANGELOG),
        published_changelog=published_changelog,
        previous_version=previous,
        version=args.version,
        previous_builds=builds,
        notes=notes,
        highlights=highlights,
        date=dt.date.today().isoformat(),
    )
    if gh.ref("main") != source:
        raise OperatorError("source advanced during preparation; no worktree created")
    worktree = args.worktree.resolve()
    branch = f"release/prepare-{args.version}"
    if worktree.exists():
        raise OperatorError(
            "preparation worktree already exists; refusing to overwrite it"
        )
    run("git", "worktree", "add", "-b", branch, str(worktree), source, cwd=root)
    write_metadata(worktree, files, args.version)
    run("git", "diff", "--check", cwd=worktree)
    run("git", "add", "--", *files, cwd=worktree)
    changed = set(
        run("git", "diff", "--cached", "--name-only", cwd=worktree).splitlines()
    )
    if changed != set(files):
        raise OperatorError("metadata-only bump inventory mismatch")
    run("git", "commit", "-m", f"chore: bump version to {args.version}", cwd=worktree)
    plan = {
        "schema": 1,
        "repository": REPOSITORY,
        "version": args.version,
        "source_sha": source,
        "bump_sha": sha(run("git", "rev-parse", "HEAD", cwd=worktree)),
        "base": "main",
        "branch": branch,
        "worktree": str(worktree),
        "copy_sha256": identity,
        "created_at": utcnow(),
        "dispatches": {},
        "previous_desktop_tag": latest["tag_name"],
    }
    save(path, plan)
    return plan


def resume(path: Path, plan: dict, gh: GitHub) -> dict:
    assert_source(plan, gh)
    dry = ensure_stage(path, plan, "parent-dry-run", gh)
    if dry["state"] != "success":
        return {"stage": "parent-dry-run", **dry}
    publish_pr(path, plan, gh)
    preflight = ensure_stage(path, plan, "preflight", gh)
    if preflight["state"] != "success":
        return {"stage": "preflight", **preflight}
    attach_preflight(plan, gh, preflight)
    return {
        "stage": "preparation-complete",
        "pr": plan["pr"],
        "next": "Review and required checks; no merge/publication performed",
    }


def drive(path: Path, version: str, gh: GitHub, interval: int, timeout: int) -> int:
    """Advance the existing transaction while only ordinary asynchronous work waits.

    Release the shared lock before sleeping: status and another operator may
    inspect/reconcile the same plan. Reload on every iteration so once receipts
    written by either process are always authoritative.
    """
    if interval < 1 or timeout < 1:
        raise OperatorError("poll interval and timeout must be positive seconds")
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        with path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = json.loads(path.read_text())
            validate_plan(plan, version)
            result = resume(path, plan, gh)
        if result != previous:
            print(json.dumps(result), flush=True)
            previous = result
        if result["stage"] == "preparation-complete":
            return 0
        state = result.get("state")
        if state not in ("waiting", "dispatched", "dispatch-unconfirmed"):
            print(
                json.dumps(
                    {
                        "state": "operator-action-required",
                        "reason": state,
                        "next": "Inspect the reported run; no approval, rerun or publication performed",
                    }
                ),
                flush=True,
            )
            return 2
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                json.dumps(
                    {
                        "state": "wait-timeout",
                        "next": f"Restart run --version {version}; existing dispatch receipts are retained",
                    }
                ),
                flush=True,
            )
            return 3
        time.sleep(min(interval, remaining))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--version", required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--notes", type=Path, required=True)
    prep.add_argument("--highlights", type=Path, required=True)
    prep.add_argument("--worktree", type=Path, required=True)
    for cmd in ("status", "resume"):
        sub.add_parser(cmd).add_argument("--version", required=True)
    auto = sub.add_parser("run", help="advance preparation without manual polling")
    auto.add_argument("--version", required=True)
    auto.add_argument("--poll-seconds", type=int, default=30)
    auto.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args(argv)
    try:
        parse_version(args.version)
        root = Path(run("git", "rev-parse", "--show-toplevel"))
        origin = run("git", "remote", "get-url", "origin", cwd=root)
        if origin not in (
            f"https://github.com/{REPOSITORY}.git",
            f"https://github.com/{REPOSITORY}",
            f"git@github.com:{REPOSITORY}.git",
            f"ssh://git@github.com/{REPOSITORY}.git",
        ):
            raise OperatorError(
                "origin does not match the canonical release repository"
            )
        common = Path(
            run("git", "rev-parse", "--path-format=absolute", "--git-common-dir")
        )
        states = common / "release-preparation"
        states.mkdir(exist_ok=True)
        path = states / f"{args.version}.json"
        gh = GitHub()
        if args.command == "run":
            return drive(
                path, args.version, gh, args.poll_seconds, args.timeout_seconds
            )
        with (states / f"{args.version}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == "prepare":
                result = prepare(args, path, gh, root)
            else:
                plan = json.loads(path.read_text())
                validate_plan(plan, args.version)
                if args.command == "resume":
                    result = resume(path, plan, gh)
                else:
                    result = {
                        "version": args.version,
                        "source_sha": plan["source_sha"],
                        "bump_sha": plan["bump_sha"],
                        "source_current": gh.ref(plan["base"]) == plan["source_sha"],
                        "stages": {
                            name: stage_status(plan, name, gh) for name in WORKFLOWS
                        },
                        "pr": plan.get("pr"),
                        "publication": "not managed by this preparation command",
                    }
            print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print(
            "release preparation interrupted; restart run to reconcile saved state",
            file=sys.stderr,
        )
        return 130
    except (
        OperatorError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"release preparation blocked: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
