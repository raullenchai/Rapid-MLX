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
   to ``gh pr create``. If ``gh`` isn't installed or the user is
   offline, fall back to printing the exact commands they need to run.
   No silent failure — the file is always on disk before any git work,
   so the user can always recover by running the commands themselves.

No network calls anywhere in this module except the one ``gh pr create``
the user has just consented to. Imports are deferred inside functions
so loading the module on a non-Apple-Silicon dev box (for unit testing)
doesn't drag in MLX.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
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
) -> dict:
    """Build the full JSON payload for one submission.

    Pure: no I/O, no clock reads unless ``now`` is None (in which case
    we stamp ``datetime.now(timezone.utc)`` — the only place the wall
    clock enters the submission). All other fields come from the caller.

    The returned dict's key order matches the schema's ``required`` list
    so that ``json.dumps(indent=2)`` produces a stable, readable layout
    when shown to the user for consent.
    """
    submitted_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    # The schema expects ``date-time`` format; the ``+00:00`` suffix is
    # the canonical ISO 8601 UTC form (NOT bare 'Z', NOT naive). Strip
    # any sub-second precision so two clean submissions a moment apart
    # don't look like noise.

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
        "Nothing has left your machine yet. Press [y] to open a PR with "
        "the JSON above, or [Enter] to cancel.",
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
    a bool) lets the caller decide where to push (always origin) and
    where to target the PR (always upstream, regardless of name).
    Host check is exact-match on ``github.com`` to defeat the
    ``evilgithub.com`` spoofing surface (round-6 BLOCKING, separate).
    """
    for name, (host, path) in _list_remotes(repo).items():
        if host == "github.com" and path == UPSTREAM_OWNER_REPO:
            return name
    return None


def _origin_host_is_github(repo: Path) -> bool:
    """True iff ``origin`` resolves to a host of ``github.com``.

    We push to ``origin``, so origin must be on GitHub — otherwise
    a contributor with ``origin = some-random-host.com/x/y`` would
    have us push their payload to an arbitrary third-party host on
    the strength of an ``upstream`` remote pointing at the canonical
    repo. Defeats the variant of the round-6 URL-spoofing BLOCKING
    where the bad URL hides in ``origin`` while ``upstream`` looks
    legitimate.
    """
    r = _run_git(repo, "remote", "get-url", "origin")
    if r.returncode != 0:
        return False
    host, _ = _parse_git_remote(r.stdout.strip())
    return host == "github.com"


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
) -> tuple[bool, set[str]]:
    """Branch + commit + push + ``gh pr create``. Returns (success, completed_steps).

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
        return False, set()

    steps: list[tuple[str, list[str]]] = [
        ("checkout", ["git", "-C", str(repo), "checkout", "-b", branch]),
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
        ("push", ["git", "-C", str(repo), "push", "-u", "origin", branch]),
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
                "--head",
                branch,
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
            return False, completed
        completed.add(label)
        if result.stdout.strip():
            print(f"  {label}: {result.stdout.strip()}", file=stdout)
    return True, completed


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


def _print_manual_fallback(
    repo: Path,
    submission_path: Path,
    payload: dict,
    *,
    stdout,
    completed: set[str] | None = None,
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
        print(f"    git checkout -b {branch}", file=stdout)
    if "stage" not in done:
        print(f"    git add {rel_path}", file=stdout)
    if "commit" not in done:
        print(
            f"    git commit -m 'community-bench: {payload['model']['alias']} "
            f"on {payload['hardware']['chip']}'",
            file=stdout,
        )
    if "push" not in done:
        print(f"    git push -u origin {branch}", file=stdout)
    # If we already pushed but pr_create failed, just retry pr_create —
    # the user shouldn't re-do the branch ops. ``--repo`` forces the
    # PR target to upstream regardless of whether origin is a fork.
    print(
        f"    gh pr create --repo {UPSTREAM_REPO_FOR_GH} --head {branch}",
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
    if upstream_remote is None or not _origin_host_is_github(repo):
        print(
            f"  Error: {repo} is a git repo but no remote points at "
            f"github.com/{UPSTREAM_OWNER_REPO}, or 'origin' is not on "
            f"github.com. --submit needs either a direct checkout of "
            f"raullenchai/Rapid-MLX or a fork with an upstream remote: "
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

    pr_ok, completed_steps = _make_pr_via_gh(repo, submission_path, payload, stdout=out)
    if pr_ok:
        print("\n  PR opened successfully.", file=out)
    else:
        _print_manual_fallback(
            repo,
            submission_path,
            payload,
            stdout=out,
            completed=completed_steps,
        )

    _print_thanks(payload, stdout=out)
    return 0


__all__ = [
    "build_submission_payload",
    "submit_interactive",
]
