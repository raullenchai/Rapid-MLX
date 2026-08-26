#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify exact Desktop tagged-publication evidence BEFORE the engine release.

R3 (#2301): ``tag_desktop_app.sh`` is idempotent — a ``rapid-mac-v*`` tag that
already exists at the validated candidate SHA no-ops (exit 0). That alone does
NOT prove the tagged Desktop publish happened: rapid-mac-release.yml could have
failed, been missed, or never run, and the engine release would still proceed.
This helper is the gate that closes that gap. It runs AFTER the tag claim and
BEFORE ``scripts/create_release.sh`` (the engine release) and fails closed unless
all of the following hold:

  1. ``app_tag`` is a valid release version (via the shared release_version
     parser) and the tag resolves (peeling annotated objects) to exactly
     ``accepted_sha`` — the SHA the candidate lane validated.
  2. An EXACT `rapid-mac-release.yml` run reaches SUCCESS within the deadline. A
     run is exact when its ref/branch is the tag, its head_sha equals
     ``accepted_sha``, and its event is the tag push (or an explicit
     ``workflow_dispatch`` at the tag ref — a recovery we never create here).
     Poll policy: prefer any exact success; keep waiting while an exact run is
     active (queued/in_progress); fail only when no exact run is active and a
     failed one remains.
  3. The exact successful run's immutable artifact manifest verifies the DMG
     bytes, source SHA, tag, embedded version, signing status and completed gate
     set. A published GitHub Release for the tag must expose that exact DMG
     SHA-256 and byte size, and the tag must still resolve to ``accepted_sha``.
  4. Every API discontinuity (non-zero gh, auth failure, malformed response)
     fails immediately, not after a long retry; the whole poll is bounded by the
     deadline.

If the tag exists but no successful exact run/release is reached, the diagnostic
directs recovery to: rerun the failed exact workflow, or dispatch
``rapid-mac-release.yml`` at ``--ref <tag>``, then re-run auto-release. It never
moves or deletes the tag, and it never auto-dispatches or broadens publication.

All logic is exercised with raw ``gh`` output (never text/regex on structured
fields) and a deadline + sleep that are injectable, so the poll behaviour is
testable offline with a mock ``gh`` and --sleep-sec 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The immutable asset the tagged lane must publish for this to count as the
# Desktop having shipped this tag's artifact.
DMG_ASSET = "rapid-mlx-desktop.dmg"
MANIFEST_ASSET = "rapid-mlx-desktop.manifest.json"
WORKFLOW_ARTIFACT = "rapid-mlx-desktop-dmg"

try:
    from release_version import parse_version  # run from scripts/
except ModuleNotFoundError:  # imported under a tests/ runner or scripts.* path
    from scripts.release_version import parse_version


class PublishGateError(Exception):
    """Raised on any fail-closed publish-evidence condition."""


def _assert_sha(sha: str) -> None:
    if (
        not isinstance(sha, str)
        or len(sha) != 40
        or any(ch not in "0123456789abcdef" for ch in sha)
    ):
        raise PublishGateError(
            f"accepted-sha must be a 40-character lowercase Git commit SHA; got {sha!r}"
        )


def _assert_app_tag(app_tag: str) -> None:
    if not isinstance(app_tag, str) or not app_tag.startswith("rapid-mac-v"):
        raise PublishGateError(
            f"app-tag must start with 'rapid-mac-v'; got {app_tag!r}"
        )
    try:
        parse_version(app_tag[len("rapid-mac-v") :])
    except ValueError as exc:
        raise PublishGateError(
            f"app-tag has an invalid release version: {exc}"
        ) from exc


def _run(gh: str, args: list[str], *, repo: str, timeout_sec: int = 120) -> str:
    """Run a gh subcommand; return stdout or fail closed on ANY error.

    ``gh api`` has no ``--repo`` flag — the repo is resolved from the ``GH_REPO``
    environment variable (or the cwd's git remote), so we pass it via env while
    preserving the caller's environment. An API/auth/timeout failure here is a
    hard stop — never folded into the poll retry loop, because a broken
    credential or endpoint will not heal on its own.
    """
    env = dict(os.environ)
    env["GH_REPO"] = repo
    try:
        proc = subprocess.run(
            [gh, *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublishGateError(f"failed to run gh {args[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise PublishGateError(
            f"gh {' '.join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def _resolve_tag(gh: str, repo: str, app_tag: str) -> str:
    """Resolve the tag to a commit, peeling annotated tag objects.

    Mirrors tag_desktop_app.sh's resolver: start in the explicit tag namespace
    and peel ``tag`` objects (bounded) to a ``commit``. Any API failure or a
    non-commit result fails closed.
    """
    MAX_PEEL = 10
    obj = _run(gh, ["api", f"repos/{repo}/git/ref/tags/{app_tag}"], repo=repo).strip()
    try:
        parsed = json.loads(obj)
    except json.JSONDecodeError as exc:
        raise PublishGateError(
            f"malformed tag-object JSON for {app_tag}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise PublishGateError(f"tag ref JSON for {app_tag} is not an object")
    object_sha = None
    object_type = None
    obj_node = parsed.get("object")
    if isinstance(obj_node, dict):
        object_sha = obj_node.get("sha")
        object_type = obj_node.get("type")
    depth = 0
    while True:
        if object_type == "commit":
            if not object_sha:
                raise PublishGateError(f"tag {app_tag} peels to an empty commit sha")
            return object_sha
        if object_type == "tag":
            depth += 1
            if depth > MAX_PEEL:
                raise PublishGateError(f"tag {app_tag} exceeds annotated-peel depth")
            tag_obj = _run(
                gh, ["api", f"repos/{repo}/git/tags/{object_sha}"], repo=repo
            ).strip()
            try:
                tag_parsed = json.loads(tag_obj)
            except json.JSONDecodeError as exc:
                raise PublishGateError(
                    f"malformed annotated tag JSON for {app_tag}: {exc}"
                ) from exc
            inner = tag_parsed.get("object") if isinstance(tag_parsed, dict) else None
            if not isinstance(inner, dict):
                raise PublishGateError(
                    f"annotated tag JSON for {app_tag} has no object"
                )
            object_type = inner.get("type")
            object_sha = inner.get("sha")
        else:
            raise PublishGateError(
                f"tag {app_tag} resolves to type {object_type!r}, not a commit"
            )


def _qualifying_runs(
    gh: str, repo: str, workflow: str, app_tag: str, accepted_sha: str
) -> list[dict]:
    """List rapid-mac-release.yml runs; return those BOUND exactly to this tag+candidate.

    Uses the official workflow-runs API filtered by ``branch == app_tag``, then
    requires, in Python: ``head_branch == app_tag`` (exact ref), ``head_sha ==
    accepted_sha`` (exact candidate), and ``event`` in {push, workflow_dispatch}
    (push is the expected trigger; dispatch-at-tag is an explicit recovery).
    API discontinuity (call fails, malformed JSON) fails immediately here.
    """
    runs_json = _run(
        gh,
        [
            "api",
            f"repos/{repo}/actions/workflows/{workflow}/runs",
            "-X",
            "GET",
            "-f",
            f"branch={app_tag}",
            "-f",
            "per_page=50",
            "--jq",
            ".workflow_runs",
        ],
        repo=repo,
    ).strip()
    try:
        runs = json.loads(runs_json)
    except json.JSONDecodeError as exc:
        raise PublishGateError(
            f"workflow-runs API returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(runs, list):
        raise PublishGateError("workflow-runs API did not return an array")
    allowed_events = {"push", "workflow_dispatch"}
    bound = []
    for r in runs:
        if not isinstance(r, dict):
            raise PublishGateError(
                f"workflow-runs API returned malformed record: {r!r}"
            )
        run_id = r.get("id")
        event = r.get("event")
        head_sha = r.get("head_sha")
        head_branch = r.get("head_branch")
        status = r.get("status")
        conclusion = r.get("conclusion")
        if (
            type(run_id) is not int
            or not isinstance(event, str)
            or not isinstance(head_sha, str)
            or not isinstance(head_branch, str)
            or not isinstance(status, str)
            or (conclusion is not None and not isinstance(conclusion, str))
        ):
            raise PublishGateError(
                f"workflow-runs API returned malformed record: {r!r}"
            )
        if (
            head_branch == app_tag
            and head_sha == accepted_sha
            and event in allowed_events
        ):
            bound.append(
                {
                    "databaseId": run_id,
                    "event": event,
                    "headSha": head_sha,
                    "status": status,
                    "conclusion": conclusion,
                }
            )
    return bound


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_artifact_identity(
    gh: str, repo: str, run_id: int, app_tag: str, accepted_sha: str
) -> tuple[str, int]:
    """Download the exact run artifact and return its verified DMG hash/size."""

    with tempfile.TemporaryDirectory(prefix="rapid-desktop-publish-") as tmp:
        root = Path(tmp)
        _run(
            gh,
            [
                "run",
                "download",
                str(run_id),
                "--repo",
                repo,
                "--name",
                WORKFLOW_ARTIFACT,
                "--dir",
                str(root),
            ],
            repo=repo,
            timeout_sec=600,
        )
        dmgs = list(root.rglob(DMG_ASSET))
        manifests = list(root.rglob(MANIFEST_ASSET))
        if len(dmgs) != 1 or len(manifests) != 1:
            raise PublishGateError(
                f"exact run {run_id} artifact must contain one {DMG_ASSET} and one "
                f"{MANIFEST_ASSET}; found {len(dmgs)} and {len(manifests)}"
            )
        dmg = dmgs[0]
        try:
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublishGateError(
                f"exact run {run_id} has an unreadable Desktop manifest: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise PublishGateError(
                f"exact run {run_id} Desktop manifest is not an object"
            )
        version = app_tag[len("rapid-mac-v") :]
        embedded = manifest.get("embedded_version")
        delta_compared = manifest.get("dmg_size_delta_compared")
        expected_gates = [
            "signed-build",
            "bundle-size",
            "app-notarize",
            "dmg-build",
        ]
        if delta_compared is True:
            expected_gates.append("dmg-size-delta")
        expected_gates.extend(["validate-dmg", "dmg-notarize", "final-validate-dmg"])
        if (
            manifest.get("schema") != 1
            or manifest.get("project") != "rapid-mlx"
            or manifest.get("artifact_kind") != "desktop-dmg"
            or manifest.get("version") != version
            or manifest.get("source_sha") != accepted_sha
            or manifest.get("app_tag") != app_tag
            or manifest.get("signed") is not True
            or type(delta_compared) is not bool
            or manifest.get("validation_gate") != "|".join(expected_gates)
            or not isinstance(embedded, dict)
            or embedded.get("CFBundleShortVersionString") != version
            or not isinstance(embedded.get("CFBundleVersion"), str)
            or not embedded.get("CFBundleVersion")
        ):
            raise PublishGateError(
                f"exact run {run_id} Desktop manifest identity/signing does not "
                "match the accepted tagged candidate"
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            raise PublishGateError(
                f"exact run {run_id} Desktop manifest must contain one artifact"
            )
        item = artifacts[0]
        if not isinstance(item, dict) or item.get("filename") != DMG_ASSET:
            raise PublishGateError(
                f"exact run {run_id} Desktop manifest does not name {DMG_ASSET}"
            )
        size = item.get("size")
        digest = item.get("sha256")
        if type(size) is not int or size <= 0 or not _is_sha256_hex(digest):
            raise PublishGateError(
                f"exact run {run_id} Desktop manifest has invalid DMG size/digest"
            )
        if dmg.stat().st_size != size or _sha256(dmg) != digest:
            raise PublishGateError(
                f"exact run {run_id} DMG bytes do not match its Desktop manifest"
            )
        return digest, size


def _published_release_dmg(gh: str, repo: str, app_tag: str) -> tuple[str, int] | None:
    """Return release DMG digest/size iff published, uploaded and non-empty.

    Uses the release-by-tag REST endpoint. Requires ``draft == false`` and an
    asset named ``rapid-mlx-desktop.dmg`` whose ``state == "uploaded"`` and
    ``size > 0`` and a valid SHA-256 ``digest``. The digest is mandatory because
    it is the byte identity compared with the exact run's manifest.
    """
    out = _run(gh, ["api", f"repos/{repo}/releases/tags/{app_tag}"], repo=repo).strip()
    try:
        rel = json.loads(out)
    except json.JSONDecodeError as exc:
        raise PublishGateError(f"malformed release JSON for {app_tag}: {exc}") from exc
    if not isinstance(rel, dict):
        raise PublishGateError(f"release by tag {app_tag} is not an object")
    if rel.get("tag_name") != app_tag:
        raise PublishGateError(
            f"release-by-tag response is not bound to {app_tag}: "
            f"tag_name={rel.get('tag_name')!r}"
        )
    # The release-by-tag REST endpoint names this field "draft" (not
    # "isDraft"); a draft is not published evidence.
    if rel.get("draft") is not False:
        return None
    assets = rel.get("assets")
    if not isinstance(assets, list):
        raise PublishGateError(f"release assets for {app_tag} is not a list")
    for asset in assets:
        if not isinstance(asset, dict):
            raise PublishGateError(
                f"release assets for {app_tag} contain a malformed record"
            )
        if asset.get("name") != DMG_ASSET:
            continue
        if asset.get("state") != "uploaded":
            return None
        size = asset.get("size")
        # bool is an int subclass in Python; malformed ``size: true`` must not
        # count as a non-empty artifact.
        if type(size) is not int or size <= 0:
            return None
        digest = asset.get("digest")
        if not _is_sha256_digest(digest):
            return None
        return digest[len("sha256:") :], size
    return None


def _is_sha256_digest(digest: object) -> bool:
    return (
        isinstance(digest, str)
        and digest.startswith("sha256:")
        and len(digest) == len("sha256:") + 64
        and all(ch in "0123456789abcdef" for ch in digest[len("sha256:") :])
    )


def _is_sha256_hex(digest: object) -> bool:
    return (
        isinstance(digest, str)
        and len(digest) == 64
        and all(ch in "0123456789abcdef" for ch in digest)
    )


def verify(
    *,
    app_tag: str,
    accepted_sha: str,
    repo: str,
    workflow: str,
    gh: str,
    deadline_sec: int,
    sleep_sec: int,
) -> list[str]:
    """Return evidence lines or raise PublishGateError."""
    _assert_app_tag(app_tag)
    _assert_sha(accepted_sha)

    evidence: list[str] = []

    # 1) Tag must resolve to the validated candidate.
    resolved = _resolve_tag(gh, repo, app_tag)
    if resolved != accepted_sha:
        raise PublishGateError(
            f"{app_tag} resolves to {resolved}, not the validated candidate "
            f"{accepted_sha}. A published tag is never moved: supersede with the "
            "next RC on its own validated commit."
        )
    evidence.append(f"tag {app_tag} resolves to validated candidate {accepted_sha}")

    # 2) Poll the exact rapid-mac-release.yml run(s) bound to app_tag+accepted_sha.
    #    Prefer any exact success; wait while an exact run is active; fail only
    #    when no exact run is active and a failed one remains. API discontinuities
    #    are fatal immediately (raised by _qualifying_runs/_run).
    deadline = time.monotonic() + deadline_sec
    success_run = None
    while True:
        bound = _qualifying_runs(gh, repo, workflow, app_tag, accepted_sha)
        completed = [r for r in bound if r.get("status") == "completed"]
        active = [
            r
            for r in bound
            if r.get("status") in ("queued", "in_progress", "waiting", "pending")
        ]

        # Once the set settles, its newest exact run is authoritative. A newer
        # failed rerun invalidates an older success; otherwise engine/Desktop
        # identity could be certified from stale publication evidence.
        if bound and not active:
            newest = max(bound, key=lambda run: run["databaseId"])
            if (
                newest.get("status") == "completed"
                and newest.get("conclusion") == "success"
            ):
                success_run = newest
                break
        # Still waiting: an exact run is active (e.g. a rerun in progress), or
        # the tag run hasn't appeared yet. Fail only when nothing exact is
        # active and a failure remains.
        failed = [r for r in completed if r.get("conclusion") != "success"]
        if not active and failed:
            raise PublishGateError(
                f"{app_tag}: rapid-mac-release.yml run(s) on the tag ref completed "
                f"without success and no exact run is active. The tagged Desktop "
                "publish failed. Recovery: re-run the failed exact workflow run, or "
                f"dispatch rapid-mac-release.yml at --ref {app_tag}, then re-run "
                "auto-release. Do NOT move or delete the tag."
            )
        if time.monotonic() >= deadline:
            raise PublishGateError(
                f"{app_tag}: no successful, exact rapid-mac-release.yml run on the tag "
                "ref within the deadline. The tagged Desktop publish did not "
                "complete. Recovery: re-run the failed exact workflow, or dispatch "
                f"rapid-mac-release.yml at --ref {app_tag}, then re-run auto-release. "
                "Do NOT move or delete the tag."
            )
        time.sleep(sleep_sec)

    evidence.append(
        f"exact rapid-mac-release.yml run {success_run.get('databaseId')} succeeded "
        f"(event {success_run.get('event')}) for {app_tag} at {accepted_sha}"
    )

    # 3) Bind the exact successful run's workflow artifact + manifest to the
    #    published GitHub Release asset bytes, then re-resolve the tag.
    run_digest, run_size = _run_artifact_identity(
        gh,
        repo,
        success_run["databaseId"],
        app_tag,
        accepted_sha,
    )
    evidence.append(
        f"exact run {success_run.get('databaseId')} artifact manifest binds "
        f"{DMG_ASSET} sha256:{run_digest} ({run_size} bytes)"
    )
    release_identity = _published_release_dmg(gh, repo, app_tag)
    if release_identity is None:
        raise PublishGateError(
            f"{app_tag}: the run succeeded but no published (non-draft) GitHub "
            "Release with an uploaded non-zero rapid-mlx-desktop.dmg asset exists. "
            "The Desktop artifact was not published; the engine must not release. "
            "Recovery: re-run the failed exact workflow, or dispatch "
            f"rapid-mac-release.yml at --ref {app_tag}, then re-run auto-release."
        )
    release_digest, release_size = release_identity
    if (release_digest, release_size) != (run_digest, run_size):
        raise PublishGateError(
            f"{app_tag}: published {DMG_ASSET} identity "
            f"sha256:{release_digest}/{release_size} does not match exact run "
            f"{success_run.get('databaseId')} manifest "
            f"sha256:{run_digest}/{run_size}. The engine must not release."
        )
    evidence.append(
        f"GitHub Release {app_tag} publishes the exact run DMG "
        f"sha256:{release_digest} ({release_size} bytes)"
    )

    resolved_after = _resolve_tag(gh, repo, app_tag)
    if resolved_after != accepted_sha:
        raise PublishGateError(
            f"{app_tag} no longer resolves to the validated candidate {accepted_sha} "
            f"right before the engine release (got {resolved_after}). Refusing."
        )
    evidence.append(
        f"tag re-verified at validated candidate {accepted_sha} immediately before engine release"
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--app-tag", required=True)
    p.add_argument("--accepted-sha", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--workflow", default="rapid-mac-release.yml")
    p.add_argument("--gh", default="gh")
    p.add_argument("--deadline-min", type=int, default=55)
    p.add_argument("--sleep-sec", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = verify(
            app_tag=args.app_tag,
            accepted_sha=args.accepted_sha,
            repo=args.repo,
            workflow=args.workflow,
            gh=args.gh,
            deadline_sec=args.deadline_min * 60,
            sleep_sec=args.sleep_sec,
        )
    except PublishGateError as exc:
        print(f"desktop publish gate: {exc}", file=sys.stderr)
        return 1
    print("\n".join(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
