# SPDX-License-Identifier: Apache-2.0
"""Submission flow for ``rapid-mlx bench --submit``.

Three responsibilities, in order:

1. **Build** — assemble the JSON payload from the
   ``hardware``/``runner`` outputs in a shape that exactly matches
   ``community-benchmarks/schema.json``. Pure function; no I/O.
2. **Consent** — pretty-print the payload to the terminal and require
   an explicit ``y`` keystroke. Default is no. The bytes that get
   shown ARE the bytes that get written; we don't decorate-then-strip.
3. **Open PR** — write the file to ``community-benchmarks/submissions/``
   in the user's local checkout, create a branch, commit, and shell out
   to ``gh pr create``. A contributor who cloned upstream directly gets
   a fork created/reused via ``gh repo fork``; the branch is never pushed
   to upstream unless the authenticated GitHub user owns it. If ``gh``
   isn't installed or the user is offline, print fork-first recovery
   commands. No silent failure — the file is always on disk before any
   git work, so the user can always recover from the generated JSON.

Imports are deferred inside functions so loading the module on a
non-Apple-Silicon dev box (for unit testing) doesn't drag in MLX.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import urllib.parse
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .hardware import Hardware, Software
from .runner import SCHEMA_VERSION, BenchResult, standardized_config_dict


def _new_submission_id() -> str:
    """First 12 lowercase hex chars of a fresh uuid4.

    12 chars = 48 bits of entropy ⇒ collision probability is negligible
    at the scale of a community DB. Schema pins this exact format with
    a regex so any drift fails CI.
    """
    return uuid.uuid4().hex[:12]


def _slugify(s: str) -> str:
    """Lowercase + non-alnum → ``-``, collapse runs, strip ends.

    Used only for filenames (``20260615-apple-m3-ultra-qwen3.5-9b-4bit-abc.json``).
    The schema fields themselves keep the original case ("Apple M3 Ultra").
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def build_submission_payload(
    hardware: Hardware,
    software: Software,
    alias: str,
    hf_path: str,
    bench: BenchResult,
    notes: str | None,
    now: datetime | None = None,
    *,
    tier: str | None = None,
    smoke_result: dict | None = None,
    harness_result: dict | None = None,
) -> dict:
    """Build the full JSON payload for one submission.

    Pure: no I/O, no clock reads unless ``now`` is None (in which case
    we stamp ``datetime.now(timezone.utc)`` — the only place the wall
    clock enters the submission). All other fields come from the caller.

    The returned dict's key order matches the schema's ``required`` list
    so that ``json.dumps(indent=2)`` produces a stable, readable layout
    when shown to the user for consent.

    Schema v2 optional kwargs:

    - ``tier`` — which bench tier produced this submission
      (``"speed"`` | ``"smoke"`` | ``"harness"`` | ``"all"``). If
      ``None``, the field is omitted entirely so the aggregator treats
      the row as v1-equivalent. This keeps the byte-for-byte equivalence
      with v1 that the existing ``--submit`` flow relies on, modulo the
      version integer itself.
    - ``smoke_result`` — required iff ``tier in ("smoke", "all")``. The
      schema enforces the same invariant via a top-level ``allOf``
      conditional, and we ALSO ``ValueError`` here so a misuse from a
      future caller surfaces immediately rather than as a schema
      validation failure two layers up.
    - ``harness_result`` — required iff ``tier in ("harness", "all")``.
      Same coupling as ``smoke_result``.

    The ``schema_version`` field on the wire is always
    ``SCHEMA_VERSION`` (currently 2). v2 with no new fields is a
    superset of v1 — the aggregator can ignore the bump and treat the
    row as a speed-only submission, which is the design contract.
    """
    submitted_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    # The schema expects ``date-time`` format; the ``+00:00`` suffix is
    # the canonical ISO 8601 UTC form (NOT bare 'Z', NOT naive). Strip
    # any sub-second precision so two clean submissions a moment apart
    # don't look like noise.

    # Validate the tier ↔ result coupling at the boundary. The schema's
    # ``allOf`` block enforces the same thing in CI, but failing here
    # with a clear Python-side error message is friendlier for the
    # future CLI code that wires this up — schema errors surface as
    # opaque jsonschema messages with full property paths.
    if tier is not None and tier not in ("speed", "smoke", "harness", "all"):
        raise ValueError(f"tier must be one of speed/smoke/harness/all, got {tier!r}")
    if tier in ("smoke", "all") and smoke_result is None:
        raise ValueError(f"tier={tier!r} requires smoke_result to be populated")
    if tier in ("harness", "all") and harness_result is None:
        raise ValueError(f"tier={tier!r} requires harness_result to be populated")
    # Inverse: passing a result without the matching tier would land an
    # ambiguous payload in the corpus (aggregator doesn't know which
    # tier produced it). Cheaper to reject here than to debug a
    # mis-labelled row in the dashboard later.
    if smoke_result is not None and tier not in ("smoke", "all"):
        raise ValueError(
            f"smoke_result was provided but tier={tier!r} does not include "
            f"the smoke bucket (must be 'smoke' or 'all')"
        )
    if harness_result is not None and tier not in ("harness", "all"):
        raise ValueError(
            f"harness_result was provided but tier={tier!r} does not "
            f"include the harness bucket (must be 'harness' or 'all')"
        )

    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": _new_submission_id(),
        "submitted_at": submitted_at,
        "hardware": asdict(hardware),
        "software": asdict(software),
        "model": {"alias": alias, "hf_path": hf_path},
        "config": standardized_config_dict(bench.sampling, bench.prompt_hash),
        "buckets": {
            "short": bench.short.to_schema_dict(),
            "long": bench.long.to_schema_dict(),
        },
    }
    if notes is not None:
        payload["notes"] = notes
    if bench.peak_ram_mb is not None:
        payload["peak_ram_mb"] = bench.peak_ram_mb
    # v2 optional fields — only emit when populated so the wire shape
    # for a default (tier=None) v2 submission is byte-equivalent to a
    # v1 submission modulo the version integer. The snapshot test for
    # this lives in tests/test_payload_builder_v2_kwargs.py.
    if tier is not None:
        payload["tier"] = tier
    if smoke_result is not None:
        payload["smoke_result"] = smoke_result
    if harness_result is not None:
        payload["harness_result"] = harness_result
    return payload


def _submission_filename(payload: dict) -> str:
    """``<YYYYMMDD>-<chip-slug>-<alias-slug>-<id>.json``.

    Ordering chosen so ``ls`` sorts by date naturally — a reviewer
    scanning the directory by hand sees newest at the bottom on every
    standard ``ls`` output.
    """
    date = payload["submitted_at"].split("T")[0].replace("-", "")
    chip = _slugify(payload["hardware"]["chip"])
    alias = _slugify(payload["model"]["alias"])
    sid = payload["submission_id"]
    return f"{date}-{chip}-{alias}-{sid}.json"


def _pretty(payload: dict) -> str:
    """Stable indent=2 JSON. Same encoding as what gets written to disk
    so the user reviews exactly the bytes they're submitting."""
    return json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False)


def _ask_consent(payload: dict, *, stdin=None, stdout=None) -> bool:
    """Show the payload and read a single y/N line.

    Default = N. Only ``y`` or ``yes`` (case-insensitive, stripped)
    counts as consent. EOF (piped non-interactive stdin) also counts
    as N — refusing to submit silently in CI is the safer default,
    even though ``--submit`` should never be run non-interactively
    anyway.
    """
    out = stdout or sys.stdout
    inp = stdin or sys.stdin

    print("", file=out)
    print(
        "About to submit the following payload to community-benchmarks:",
        file=out,
    )
    print("=" * 72, file=out)
    print(_pretty(payload), file=out)
    print("=" * 72, file=out)
    print(
        "Nothing has left your machine yet. Pressing [y] consents to GitHub "
        "network operations: `git fetch` of upstream `main`, creating or "
        "reusing your fork when `origin` points at upstream, `git push` to a "
        "GitHub remote you can write, then `gh pr create` against "
        "raullenchai/Rapid-MLX. They run under your existing git/gh "
        "credentials. Press [Enter] to cancel.",
        file=out,
    )
    out.flush()

    try:
        answer = inp.readline()
    except EOFError:
        return False
    if not answer:  # EOF
        return False
    return answer.strip().lower() in {"y", "yes"}


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run ``git`` in ``repo`` and capture output.

    Plumbing in its own function so test code can monkeypatch one
    callsite instead of every ``subprocess.run``.
    """
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


UPSTREAM_OWNER_REPO = "raullenchai/rapid-mlx"
UPSTREAM_REPO_FOR_GH = "raullenchai/Rapid-MLX"
FORK_REMOTE_BASENAME = "community-bench-fork"
GITHUB_OWNER_RE = re.compile(
    r"^(?!-)(?!.*--)[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?$",
    re.IGNORECASE,
)


def _list_remotes(repo: Path) -> dict[str, tuple[str | None, str | None]]:
    """Return ``{remote_name: (host, owner/repo)}`` for every git remote.

    Parsing each line of ``git remote -v`` (which emits per-fetch/push
    rows for each remote) and deduplicating to one row per remote name.
    Values are produced by ``_parse_git_remote``; unparseable URLs map
    to ``(None, None)`` and are filtered out by the caller.
    """
    r = _run_git(repo, "remote", "-v")
    if r.returncode != 0:
        return {}
    out: dict[str, tuple[str | None, str | None]] = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, url = parts[0], parts[1]
        if name in out:
            continue
        out[name] = _parse_git_remote(url)
    return out


def _find_upstream_remote(repo: Path) -> str | None:
    """Return the name of the remote pointing at ``raullenchai/Rapid-MLX``.

    Searches every remote — not just ``origin`` — so the standard
    GitHub fork workflow works:

      - ``origin`` = ``<contributor>/Rapid-MLX`` (their fork)
      - ``upstream`` = ``raullenchai/Rapid-MLX``

    Earlier revisions only accepted ``origin = raullenchai/Rapid-MLX``,
    which locked community contributors out: they don't have write
    access to upstream, so they fork, and their origin is the fork.
    (Codex PR #582 round-6 BLOCKING.) Returning the *name* (not just
    a bool) lets the caller fetch the canonical base while independently
    choosing a writable push remote (origin or a generated fork remote).
    Host check is exact-match on ``github.com`` to defeat the
    ``evilgithub.com`` spoofing surface (round-6 BLOCKING, separate).
    """
    for name, (host, path) in _list_remotes(repo).items():
        if host == "github.com" and path == UPSTREAM_OWNER_REPO:
            return name
    return None


def _safe_github_push_target(repo: Path, remote: str) -> tuple[str, str] | None:
    """Return the unique ``(owner, owner/repo)`` push target for a remote.

    Every effective push URL must point at the same github.com repository.
    The full path lets GitHub metadata, rather than a mutable repository
    name, determine whether a target is in Rapid-MLX's fork network.
    """
    r = _run_git(repo, "remote", "get-url", "--push", "--all", remote)
    if r.returncode != 0:
        return None
    urls = [line for line in r.stdout.splitlines() if line.strip()]
    if not urls:
        return None
    targets: set[str] = set()
    for url in urls:
        host, path = _parse_git_remote(url.strip())
        if host != "github.com" or not path or path.count("/") != 1:
            return None
        owner, repo_name = path.split("/", 1)
        if GITHUB_OWNER_RE.fullmatch(owner) is None or not repo_name:
            return None
        targets.add(path)
    if len(targets) != 1:
        return None
    path = targets.pop()
    return path.split("/", 1)[0], path


def _remote_is_safe_github(
    repo: Path,
    remote: str,
    *,
    expected_path: str | None = None,
) -> tuple[bool, str | None]:
    """Validate the unique GitHub push target for ``remote``."""
    target = _safe_github_push_target(repo, remote)
    if target is None:
        return False, None
    owner, path = target
    if expected_path is not None and path != expected_path.lower():
        return False, None
    return True, owner


def _origin_is_safe_github(repo: Path) -> tuple[bool, str | None]:
    """Validate origin's GitHub fetch URL and unique GitHub push target."""
    host, path = _list_remotes(repo).get("origin", (None, None))
    if host != "github.com" or not path or path.count("/") != 1:
        return False, None
    fetch_owner, fetch_repo = path.split("/", 1)
    if GITHUB_OWNER_RE.fullmatch(fetch_owner) is None or not fetch_repo:
        return False, None
    return _remote_is_safe_github(repo, "origin")


def _github_repo_is_writable_upstream_fork(repo: Path, repo_path: str) -> bool:
    """Return whether the active gh user can push this upstream-network fork."""
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo_path}",
            "--jq",
            "[.fork, .source.full_name, .parent.full_name, .permissions.push] | @tsv",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo),
    )
    if result.returncode != 0:
        return False
    fields = result.stdout.strip().split("\t")
    if len(fields) != 4 or fields[0] != "true" or fields[3] != "true":
        return False
    source, parent = fields[1], fields[2]
    return (
        source.lower() == UPSTREAM_OWNER_REPO or parent.lower() == UPSTREAM_OWNER_REPO
    )


def _find_fork_remote(repo: Path, owner: str) -> str | None:
    """Find a safe, GitHub-verified upstream fork remote for ``owner``.

    Both the fetch URL and every effective push URL must point at GitHub and
    agree on the owner. This lets us reuse a contributor's existing fork
    without trusting a fetch-only URL whose ``pushurl`` was redirected.
    """
    for name, (host, path) in _list_remotes(repo).items():
        if host != "github.com" or not path or "/" not in path:
            continue
        remote_owner, _ = path.split("/", 1)
        if remote_owner != owner.lower():
            continue
        safe, push_owner = _remote_is_safe_github(repo, name, expected_path=path)
        if (
            safe
            and push_owner == owner.lower()
            and _github_repo_is_writable_upstream_fork(repo, path)
        ):
            return name
    return None


def _unused_remote_name(repo: Path, base: str = FORK_REMOTE_BASENAME) -> str:
    """Return a deterministic remote name without overwriting user config."""
    names = set(_list_remotes(repo))
    if base not in names:
        return base
    suffix = 2
    while f"{base}-{suffix}" in names:
        suffix += 1
    return f"{base}-{suffix}"


def _github_login(repo: Path) -> tuple[str | None, str | None]:
    """Return the authenticated GitHub login, or an actionable error."""
    result = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo),
    )
    login = result.stdout.strip()
    if result.returncode != 0 or GITHUB_OWNER_RE.fullmatch(login) is None:
        error = result.stderr.strip() or "`gh` is not authenticated"
        return None, error
    return login, None


def _ensure_fork_remote(
    repo: Path, owner: str, *, stdout
) -> tuple[str | None, str | None]:
    """Create/reuse ``owner``'s fork and return its safe git remote."""
    existing = _find_fork_remote(repo, owner)
    if existing is not None:
        return existing, None

    remote_name = _unused_remote_name(repo)
    cmd = [
        "gh",
        "repo",
        "fork",
        UPSTREAM_REPO_FOR_GH,
        "--remote",
        "--remote-name",
        remote_name,
        "--clone=false",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo),
    )
    if result.returncode != 0:
        return None, result.stderr.strip() or "`gh repo fork` failed"
    if result.stdout.strip():
        print(f"  fork: {result.stdout.strip()}", file=stdout)

    remote = _find_fork_remote(repo, owner)
    if remote is None:
        return None, (
            f"fork was created but no safe git remote for {owner}/Rapid-MLX was added"
        )
    return remote, None


def _parse_git_remote(url: str) -> tuple[str | None, str | None]:
    """Extract (host, ``owner/repo``) from a git remote URL.

    Handles the three forms ``git remote get-url`` emits:
      - ``https://host/owner/repo(.git)``
      - ``ssh://git@host/owner/repo(.git)``
      - ``git@host:owner/repo(.git)``  (the scp-style form)

    Returns ``(None, None)`` for anything we can't classify, which
    causes the caller to fail-closed.
    """
    s = url.strip().lower().removesuffix(".git")
    # scp-style: ``git@host:owner/repo``. The colon is NOT a port —
    # SSH would use ``ssh://...`` for that. We treat the part before
    # ``:`` as host and the part after as path.
    if s.startswith("git@") and ":" in s and "://" not in s:
        host_part, _, path = s.partition(":")
        host = host_part[len("git@") :]
        return host or None, path or None
    # http(s):// and ssh://.
    if "://" in s:
        _, _, rest = s.partition("://")
        # Strip user@ if present in ssh URLs.
        if "@" in rest.split("/", 1)[0]:
            _, _, rest = rest.partition("@")
        host, _, path = rest.partition("/")
        return host or None, path or None
    return None, None


def _git_is_clean(repo: Path) -> bool:
    """True iff there are no uncommitted changes / untracked files.

    We refuse to commit-and-push if the working tree has unrelated
    changes — accidentally including the user's other work in a
    community-benchmark PR would be embarrassing and hard to undo.
    """
    r = _run_git(repo, "status", "--porcelain")
    return r.returncode == 0 and r.stdout.strip() == ""


def _write_payload_file(repo: Path, payload: dict) -> Path:
    """Write the JSON payload to ``submissions/<filename>`` and return path.

    Always writes with a trailing newline (Unix convention; the
    aggregator's ``json.load`` doesn't care, but ``git`` is happier
    with newline-terminated files and ``cat`` won't double-prompt).
    """
    sub_dir = repo / "community-benchmarks" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    path = sub_dir / _submission_filename(payload)
    path.write_text(_pretty(payload) + "\n", encoding="utf-8")
    return path


def _make_pr_via_gh(
    repo: Path,
    submission_path: Path,
    payload: dict,
    *,
    stdout,
    origin_owner: str,
    upstream_remote: str,
) -> tuple[bool, set[str], str | None, str | None]:
    """Return success, completed steps, head owner, and failed push remote.

    ``completed_steps`` is the set of step labels that succeeded
    before any failure (or all of them on success). The caller uses
    that set to render *state-aware* recovery instructions when
    ``success=False`` — telling the user "re-create the branch" when
    we've already pushed it would leave them confused and the branch
    in a half-committed state. (Codex PR #582 round-5 BLOCKING.)

    Strategy if anything goes wrong: bail to the manual-fallback path
    in ``submit_interactive`` so the file on disk isn't orphaned. We
    never ``git reset`` the user's repo — destructive recovery in a
    CLI we're asking strangers to run is worse than the inconvenience
    of finishing the PR by hand.
    """
    branch = f"community-bench/{payload['submission_id']}"
    rel_path = submission_path.relative_to(repo).as_posix()

    if not shutil.which("gh"):
        print(
            "\n  Note: `gh` CLI not found on PATH — falling back to "
            "manual instructions below.",
            file=stdout,
        )
        return False, set(), None, None

    # Decide where the branch can be pushed before any git mutation. Repository
    # names are not identity: GitHub permits forks to be renamed, so validate
    # non-upstream origins through their authoritative parent metadata.
    upstream_owner = UPSTREAM_OWNER_REPO.split("/", 1)[0]
    push_remote = "origin"
    origin_target = _safe_github_push_target(repo, "origin")
    if origin_target is None:
        print(
            "\n  Step failed: inspect_origin\n"
            "    stderr:  origin has no unique safe GitHub push target",
            file=stdout,
        )
        return False, set(), None, None
    head_owner, origin_path = origin_target

    origin_is_upstream = origin_path == UPSTREAM_OWNER_REPO
    origin_is_fork = not origin_is_upstream and _github_repo_is_writable_upstream_fork(
        repo, origin_path
    )
    if not origin_is_fork:
        login, login_error = _github_login(repo)
        if login is None:
            print(
                f"\n  Step failed: identify_github_user\n    stderr:  {login_error}",
                file=stdout,
            )
            return False, set(), head_owner, None
        head_owner = login
        if login.lower() == upstream_owner:
            upstream_ok, _ = _remote_is_safe_github(
                repo, upstream_remote, expected_path=UPSTREAM_OWNER_REPO
            )
            if not upstream_ok:
                print(
                    "\n  Step failed: prepare_upstream\n"
                    "    stderr:  no safe canonical upstream push remote",
                    file=stdout,
                )
                return False, set(), head_owner, None
            push_remote = upstream_remote
        else:
            push_remote, fork_error = _ensure_fork_remote(repo, login, stdout=stdout)
            if push_remote is None:
                print(
                    f"\n  Step failed: prepare_fork\n    stderr:  {fork_error}",
                    file=stdout,
                )
                return False, set(), head_owner, None

    # Branch from the upstream's default branch tip, not whatever
    # commit the user happens to have checked out. Without this, a
    # contributor with a feature branch checked out (their own work)
    # would create the community-bench branch on top of those commits,
    # and the resulting PR would carry their unrelated work too —
    # potentially leaking private code into a public bench PR. (Codex
    # PR #582 round-7 BLOCKING.) ``git fetch <upstream> main`` writes
    # FETCH_HEAD; the next ``checkout -b ... FETCH_HEAD`` branches
    # from THAT, regardless of HEAD. Fail closed if the fetch errors —
    # we won't silently fall back to the local HEAD.
    base_ref = "main"
    steps: list[tuple[str, list[str]]] = [
        (
            "fetch_base",
            [
                "git",
                "-C",
                str(repo),
                "fetch",
                "--quiet",
                upstream_remote,
                base_ref,
            ],
        ),
        (
            "checkout",
            ["git", "-C", str(repo), "checkout", "-b", branch, "FETCH_HEAD"],
        ),
        ("stage", ["git", "-C", str(repo), "add", rel_path]),
        (
            "commit",
            [
                "git",
                "-C",
                str(repo),
                "commit",
                "-m",
                f"community-bench: {payload['model']['alias']} on "
                f"{payload['hardware']['chip']} ({payload['submission_id']})",
            ],
        ),
        (
            "push",
            ["git", "-C", str(repo), "push", "-u", push_remote, branch],
        ),
        (
            "pr_create",
            [
                "gh",
                "pr",
                "create",
                # Force the PR target to canonical upstream regardless
                # of which remote ``origin`` points at. Without
                # ``--repo`` ``gh`` picks the default repo for the cwd,
                # which on a contributor's fork is the FORK — the PR
                # would open against their own repo and never reach the
                # community DB. (Codex PR #582 round-6 BLOCKING.)
                "--repo",
                UPSTREAM_REPO_FOR_GH,
                # ``--head <owner>:<branch>`` is required when origin
                # is a fork — ``gh`` otherwise looks for ``<branch>``
                # inside the target repo (raullenchai/Rapid-MLX), which
                # doesn't exist because we pushed to the contributor's
                # fork. (Codex PR #582 round-7 BLOCKING.) For a
                # maintainer's direct checkout the ``owner`` equals
                # ``raullenchai`` so the prefix is harmless.
                "--head",
                f"{head_owner}:{branch}",
                "--title",
                f"community-bench: {payload['model']['alias']} on "
                f"{payload['hardware']['chip']}",
                "--body",
                _pr_body(payload),
            ],
        ),
    ]

    completed: set[str] = set()
    for label, cmd in steps:
        # ``cwd=repo`` is critical for the ``gh`` step: ``gh pr create``
        # reads the remote / branch state from the *current working
        # directory's* git repo, not from any flag. Without it, a user
        # passing ``--repo-root /path/to/checkout`` would git-commit
        # into ``/path/to/checkout`` but then open the PR against
        # whatever repo their shell happened to be in. (Codex PR #582
        # round-2 BLOCKING.) Setting cwd for the git steps is
        # redundant since ``git -C <repo>`` already routes them, but
        # using a uniform cwd keeps the failure mode predictable.
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, cwd=str(repo)
        )
        if result.returncode != 0:
            print(
                f"\n  Step failed: {label}\n"
                f"    command: {' '.join(cmd)}\n"
                f"    stderr:  {result.stderr.strip() or '(empty)'}",
                file=stdout,
            )
            failed_push_remote = push_remote if label == "push" else None
            return False, completed, head_owner, failed_push_remote
        completed.add(label)
        if result.stdout.strip():
            print(f"  {label}: {result.stdout.strip()}", file=stdout)
    return True, completed, head_owner, None


def _pr_body(payload: dict) -> str:
    """One-paragraph PR body summarizing the submission.

    Keep it short — the file diff is the actual content, not the body.
    The body is for the human reviewer scanning the PR queue.
    """
    short = payload["buckets"]["short"]["decode_tps"]["median"]
    long_ = payload["buckets"]["long"]["decode_tps"]["median"]
    notes = payload.get("notes") or "_none_"
    return (
        f"Community benchmark submission.\n\n"
        f"- **chip**: {payload['hardware']['chip']} ({payload['hardware']['ram_gb']} GB)\n"
        f"- **model**: `{payload['model']['alias']}` "
        f"({payload['model']['hf_path']})\n"
        f"- **rapid-mlx**: {payload['software']['rapid_mlx']} / "
        f"mlx {payload['software']['mlx']}\n"
        f"- **short bucket decode_tps (median)**: {short:.2f}\n"
        f"- **long bucket decode_tps (median)**: {long_:.2f}\n"
        f"- **sampling**: {payload['config']['sampling']}\n"
        f"- **notes**: {notes}\n\n"
        f"Auto-generated by `rapid-mlx bench --submit`. The full payload "
        f"is in `{Path(_submission_filename(payload)).name}`.\n"
    )


def _find_contributor_push_target(
    repo: Path,
    *,
    verify_fork: bool = False,
    excluded_remote: str | None = None,
) -> tuple[str, str] | None:
    """Return ``(remote, owner)`` for a safe non-upstream fork target.

    Remotes are reused only when GitHub can verify their fork source. Without
    metadata, recovery creates a new collision-free fork remote instead of
    guessing that an arbitrary GitHub repository belongs to the fork network.
    """
    if not verify_fork:
        return None

    upstream_owner = UPSTREAM_OWNER_REPO.split("/", 1)[0]
    origin_safe, origin_push_owner = _origin_is_safe_github(repo)
    origin_target = _safe_github_push_target(repo, "origin")
    if (
        origin_safe
        and excluded_remote != "origin"
        and origin_push_owner
        and origin_push_owner != upstream_owner
        and origin_target is not None
        and _github_repo_is_writable_upstream_fork(repo, origin_target[1])
    ):
        return "origin", origin_push_owner

    remotes = _list_remotes(repo)
    dedicated = sorted(
        name
        for name in remotes
        if name == FORK_REMOTE_BASENAME
        or (
            name.startswith(f"{FORK_REMOTE_BASENAME}-")
            and name.removeprefix(f"{FORK_REMOTE_BASENAME}-").isdigit()
        )
    )
    for name in dedicated:
        if name == excluded_remote:
            continue
        host, path = remotes.get(name, (None, None))
        if host != "github.com" or not path or "/" not in path:
            continue
        owner, _ = path.split("/", 1)
        if owner == upstream_owner:
            continue
        safe, push_owner = _remote_is_safe_github(repo, name, expected_path=path)
        if (
            safe
            and push_owner == owner
            and _github_repo_is_writable_upstream_fork(repo, path)
        ):
            return name, owner
    return None


def _print_manual_fallback(
    repo: Path,
    submission_path: Path,
    payload: dict,
    *,
    stdout,
    completed: set[str] | None = None,
    selected_head_owner: str | None = None,
    excluded_push_remote: str | None = None,
) -> None:
    """Tell the user exactly which commands to run to finish the PR.

    Triggered when ``gh`` isn't installed, the git state is dirty, or
    a step in the auto-PR sequence failed. The submission file is
    already on disk at this point — they don't need to re-run the bench.

    ``completed`` is the set of step labels that ``_make_pr_via_gh``
    successfully ran before bailing. Without it, the fallback would
    print ``git checkout -b <branch>`` even when the branch already
    exists and is already pushed, leaving the user confused. (Codex PR
    #582 round-5 BLOCKING.) Step labels match the literals in
    ``_make_pr_via_gh.steps``: checkout / stage / commit / push /
    pr_create. ``None`` (the gh-missing or dirty-tree path) means
    nothing has happened in git yet, so the full sequence applies.
    """
    branch = f"community-bench/{payload['submission_id']}"
    rel_path = submission_path.relative_to(repo).as_posix()
    done = completed or set()
    gh_installed = shutil.which("gh") is not None
    gh_login = _github_login(repo)[0] if gh_installed else None
    gh_available = gh_login is not None
    contributor_target = _find_contributor_push_target(
        repo,
        verify_fork=gh_available,
        excluded_remote=excluded_push_remote,
    )
    base_source = f"https://github.com/{UPSTREAM_REPO_FOR_GH}.git"
    manual_fork_remote = _unused_remote_name(repo)
    origin_target = _safe_github_push_target(repo, "origin")
    origin_is_canonical = (
        origin_target is not None and origin_target[1] == UPSTREAM_OWNER_REPO
    )

    print("\n  The JSON file is on disk at:", file=stdout)
    print(f"    {submission_path}", file=stdout)

    # Lead with where we got to so the user knows what to skip.
    if done:
        already = " → ".join(
            s for s in ("checkout", "stage", "commit", "push") if s in done
        )
        print(f"  Already completed: {already}", file=stdout)
        print(
            "  Resume from where it stopped — these are the commands "
            "for the steps that still need to run:",
            file=stdout,
        )
    else:
        print(
            "  To finish the submission, run these commands from the repo root:",
            file=stdout,
        )

    if "checkout" not in done:
        if "fetch_base" not in done:
            print(f"    git fetch {base_source} main", file=stdout)
        print(f"    git checkout -b {branch} FETCH_HEAD", file=stdout)
    if "stage" not in done:
        print(f"    git add {rel_path}", file=stdout)
    if "commit" not in done:
        message = (
            f"community-bench: {payload['model']['alias']} "
            f"on {payload['hardware']['chip']}"
        )
        print(f"    git commit -m {shlex.quote(message)}", file=stdout)
    if "push" not in done:
        if contributor_target is not None:
            push_remote, _ = contributor_target
            print(f"    git push -u {push_remote} {branch}", file=stdout)
        else:
            print("", file=stdout)
            print(
                "  Your origin points at upstream (or could not be verified).",
                file=stdout,
            )
            print(
                "  Create your fork before pushing; do not push this branch "
                "to upstream:",
                file=stdout,
            )
            if gh_available:
                print(
                    f"    gh repo fork {UPSTREAM_REPO_FOR_GH} --remote "
                    f"--remote-name {manual_fork_remote} --clone=false",
                    file=stdout,
                )
            else:
                print(
                    f"    https://github.com/{UPSTREAM_REPO_FOR_GH}/fork",
                    file=stdout,
                )
                print(
                    "    # Copy your fork's complete HTTPS clone URL below",
                    file=stdout,
                )
                print(
                    f"    git remote add {manual_fork_remote} YOUR_FORK_CLONE_URL",
                    file=stdout,
                )
            print(
                f"    git push -u {manual_fork_remote} {branch}",
                file=stdout,
            )
            if not gh_available and origin_is_canonical:
                print("", file=stdout)
                print(
                    "  Maintainers only: if your credentials have confirmed "
                    "upstream write access, you may instead run:",
                    file=stdout,
                )
                print(f"    git push -u origin {branch}", file=stdout)
    # The PR-create step has two paths depending on whether ``gh`` is on
    # PATH. If we got here because gh is missing (the common newcomer
    # case), recommending ``gh pr create`` is useless — point them at
    # the GitHub web UI and at the "paste the file into a new issue"
    # fallback instead. If gh is available (this branch only hits when
    # git steps failed mid-sequence), surface gh as the resume command.
    if gh_available:
        if "push" in done and selected_head_owner is not None:
            head_arg = shlex.quote(f"{selected_head_owner}:{branch}")
        elif contributor_target is not None:
            _, head_owner = contributor_target
            head_arg = shlex.quote(f"{head_owner}:{branch}")
        elif "push" in done:
            # A state-aware recovery can reach this branch after a direct
            # upstream push by a maintainer. Keep the same-repo form.
            head_arg = shlex.quote(branch)
        else:
            # The fork command above resolves the authenticated owner. Shell
            # substitution keeps the printed recovery command copy/pasteable
            # without guessing the user's GitHub login.
            head_arg = f'"$(gh api user --jq .login):{branch}"'
        print(
            f"    gh pr create --repo {UPSTREAM_REPO_FOR_GH} --head {head_arg}",
            file=stdout,
        )
    else:
        print("", file=stdout)
        print(
            "  Then open the PR via the GitHub web UI (no `gh` CLI needed):",
            file=stdout,
        )
        # Quote both halves before joining with the literal ``:`` GitHub
        # expects between owner and branch in the compare path. Owner is
        # the more constrained piece (GitHub usernames are ``[a-zA-Z0-9-]``
        # by policy) but we still ``quote(safe="")`` defensively in case
        # _origin_is_safe_github ever loosens. The branch ref allows ``/``
        # — that's how we construct ``community-bench/<id>`` to begin with
        # — so we keep ``/`` unescaped via ``safe="/"``. Without this any
        # branch ref carrying ``#``, ``?``, or ``%`` would split the URL.
        # (Codex PR #600 round-2 BLOCKING.)
        branch_quoted = urllib.parse.quote(branch, safe="/")
        upstream_owner = UPSTREAM_OWNER_REPO.split("/", 1)[0]
        if "push" in done and selected_head_owner is not None:
            if selected_head_owner.lower() != upstream_owner:
                head_ref = (
                    f"{urllib.parse.quote(selected_head_owner, safe='')}:"
                    f"{branch_quoted}"
                )
            else:
                head_ref = branch_quoted
        elif contributor_target is not None:
            _, head_owner = contributor_target
            head_ref = f"{urllib.parse.quote(head_owner, safe='')}:{branch_quoted}"
        else:
            head_ref = f"YOUR_GITHUB_USERNAME:{branch_quoted}"
        print(
            f"    https://github.com/{UPSTREAM_REPO_FOR_GH}/compare/main...{head_ref}?expand=1",
            file=stdout,
        )
        print("", file=stdout)
        print(
            "  If you'd rather skip git entirely, paste the submission JSON",
            file=stdout,
        )
        print(
            "  contents (above path) into a new issue and we'll convert it",
            file=stdout,
        )
        print("  to a PR for you:", file=stdout)
        # ``urlencode`` over the whole querystring handles spaces, ``&``,
        # ``#``, ``%``, and any other special chars that might appear
        # in a model alias or in the chip name. Bare ``.replace(' ', '%20')``
        # produced malformed URLs for aliases like ``qwen3.6/27b`` where
        # the slash breaks GitHub's title parser. (Codex PR #600 round-1.)
        title = (
            f"community-bench: {payload['model']['alias']} "
            f"on {payload['hardware']['chip']}"
        )
        query = urllib.parse.urlencode({"title": title})
        print(
            f"    https://github.com/{UPSTREAM_REPO_FOR_GH}/issues/new?{query}",
            file=stdout,
        )


def _print_thanks(payload: dict, *, stdout) -> None:
    """Closing UX. The user just gave us real data — say so."""
    print("", file=stdout)
    print("  Thank you for contributing to the Rapid-MLX community", file=stdout)
    print(
        "  performance database! Every submission tightens the median",
        file=stdout,
    )
    print("  for everyone running this combo:", file=stdout)
    print(
        f"    {payload['hardware']['chip']} ({payload['hardware']['ram_gb']} GB) "
        f"× {payload['model']['alias']}",
        file=stdout,
    )
    print(
        "  Once the PR merges, your numbers will show up at "
        "https://rapidmlx.com/#models.",
        file=stdout,
    )


def submit_interactive(
    payload: dict,
    repo_root: Path,
    *,
    stdin=None,
    stdout=None,
) -> int:
    """End-to-end interactive submission flow. Returns exit code.

    Returns 0 on success or graceful user-cancel, non-zero only on
    setup errors (not a valid git repo, etc.) where we want CI / the
    caller to notice. A user typing 'n' is not an error.
    """
    out = stdout or sys.stdout

    # Use ``git rev-parse --show-toplevel`` instead of probing for a
    # ``.git`` directory: ``.git`` is a *file* (not a dir) in linked
    # worktrees (``git worktree add``), and refusing those would shut
    # out a legitimate workflow. (Codex PR #582 round-2 NIT.) The
    # subprocess returns the canonical repo root which we then use as
    # the cwd for every subsequent git/gh call.
    probe = subprocess.run(
        ["git", "-C", str(repo_root.resolve()), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        print(
            f"  Error: {repo_root} is not a git repository root. "
            f"--submit needs to commit the submission file into a "
            f"checkout of github.com/raullenchai/Rapid-MLX.",
            file=out,
        )
        return 2
    repo = Path(probe.stdout.strip())

    # Verify the resolved repo is associated with raullenchai/Rapid-MLX
    # before we touch any branches or open a PR. Accepted shapes:
    #   - ``origin`` = raullenchai/Rapid-MLX (maintainer's direct checkout)
    #   - ``origin`` = <user>/Rapid-MLX fork + some other remote (typically
    #     ``upstream``) pointing at raullenchai/Rapid-MLX (standard
    #     community fork workflow)
    # Without the fork case the only people who could submit are users
    # with write access to upstream — the entire community contribution
    # path was unreachable. (Codex PR #582 round-6 BLOCKING.) We also
    # require origin's host to be exactly ``github.com`` so an attacker
    # can't redirect the push by setting a malicious origin while
    # leaving a real upstream remote as a decoy.
    upstream_remote = _find_upstream_remote(repo)
    origin_ok, origin_owner = _origin_is_safe_github(repo)
    if upstream_remote is None or not origin_ok or origin_owner is None:
        print(
            f"  Error: {repo} is a git repo but no remote points at "
            f"github.com/{UPSTREAM_OWNER_REPO}, or 'origin' (including any "
            f"pushurl override) is not a single GitHub repo. --submit "
            f"needs either a direct checkout of raullenchai/Rapid-MLX or "
            f"a fork with an upstream remote: "
            f"\n    git remote add upstream https://github.com/{UPSTREAM_REPO_FOR_GH}",
            file=out,
        )
        return 2

    if not _ask_consent(payload, stdin=stdin, stdout=out):
        print("\n  Submission cancelled. Nothing was written or sent.", file=out)
        return 0

    # Snapshot the working-tree state BEFORE writing — otherwise the
    # newly-created submission file shows up as untracked in `git status`
    # and every clean checkout looks dirty, making the auto-PR path
    # unreachable. (Codex PR #582 BLOCKING.)
    tree_was_clean = _git_is_clean(repo)

    submission_path = _write_payload_file(repo, payload)
    print(f"\n  Wrote submission to {submission_path}", file=out)

    if not tree_was_clean:
        # User has other uncommitted work — don't sweep it into the PR.
        # The submission file IS on disk; we just stop short of git ops.
        print(
            "\n  Your working tree had other uncommitted changes before "
            "this submission was written; the automated PR step is "
            "skipped to avoid mixing your work into the community-bench "
            "commit.",
            file=out,
        )
        _print_manual_fallback(repo, submission_path, payload, stdout=out)
        _print_thanks(payload, stdout=out)
        return 0

    (
        pr_ok,
        completed_steps,
        selected_head_owner,
        failed_push_remote,
    ) = _make_pr_via_gh(
        repo,
        submission_path,
        payload,
        stdout=out,
        origin_owner=origin_owner,
        upstream_remote=upstream_remote,
    )
    if pr_ok:
        print("\n  PR opened successfully.", file=out)
    else:
        _print_manual_fallback(
            repo,
            submission_path,
            payload,
            stdout=out,
            completed=completed_steps,
            selected_head_owner=selected_head_owner,
            excluded_push_remote=failed_push_remote,
        )

    _print_thanks(payload, stdout=out)
    return 0


__all__ = [
    "build_submission_payload",
    "submit_interactive",
]
