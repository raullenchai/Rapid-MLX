# SPDX-License-Identifier: Apache-2.0
"""Gate: a PR that changes an ``mlx`` / ``mlx-lm`` / ``mlx-vlm`` version bound must
carry proof that a full-family output-coherence sweep was run (#1247, #1248).

Why (incident-grounded): the Qwen3.6 garbage (#1234) came from a heuristic change
*inside* ``mlx_lm`` under an existing model, not from our code. Our ``mlx*`` deps
are now capped (pyproject.toml), so a plain ``pip install rapid-mlx`` resolves to
the validated minor. Touching one of those pins — raising a cap, dropping an
exclusion, moving the floor — can re-open the exact door that shipped garbage, so
it must be a deliberate, evidenced event, not a silent diff.

Design — fail CLOSED, coarse but sound. Version-set math over PEP 440 (epochs,
pre/post/dev/local, ``===``, wildcards, ``~=``) is subtle enough that a "flag only
loosening" classifier is easy to get subtly wrong in the unsafe direction. So this
gate flags ANY change to a guarded package's normalized version specifier and
requires an explicit human attestation. Tightening / adding a cap is thus also
gated, but its attestation is a one-line note — cheap, and every change to these
coherence-critical pins deserves a conscious sign-off. (Reordering a specifier or
changing only an environment marker is NOT a change — the specifier normalizes.)

This runs on every PR (see .github/workflows/ci.yml). No-op unless a guarded
specifier actually changes. When it does, it fails UNLESS the PR attests via:

  * a label ``mlx-coherence-swept`` on the PR, or
  * a ``Coherence-Sweep: <url-or-note>`` trailer line in the PR body.

Until the always-on coherence gate (#1247) exists to attach a machine-verified
artifact, the attestation is the human checkpoint. When #1247 lands, tighten the
``_attestation_ok`` branch to verify the real artifact instead of trusting a
label/trailer.

Pure-logic core (``extract_mlx_bounds`` / ``detect_bound_changes``) is unit-tested
in tests/test_mlx_bound_guard.py — no network, no GPU.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# The packages whose version bounds are coherence-sensitive. ``mlx`` and
# ``mlx-lm`` drive text decode; ``mlx-vlm`` drives vision + spec-decode. These
# are already PEP 503-canonical; ``canonicalize_name`` is applied to parsed
# requirement names so alternate spellings (``mlx_vlm``, ``MLX.VLM``) still match.
GUARDED_PACKAGES = frozenset({"mlx", "mlx-lm", "mlx-vlm"})

# Leading distribution-name token of a requirement string, used to classify a
# requirement that ``packaging`` could not parse (so a malformed *guarded* pin
# still fails closed).
_NAME_PREFIX = re.compile(r"^\s*([A-Za-z0-9._-]+)")

ATTEST_LABEL = "mlx-coherence-swept"
ATTEST_TRAILER = "Coherence-Sweep:"


class MalformedGuardedRequirementError(ValueError):
    """A guarded (``mlx*``) dependency string that packaging cannot parse."""


def _iter_declared_dependencies(data: dict):
    """Yield every requirement string declared anywhere pip/build reads them.

    Covers ``project.dependencies``, every ``project.optional-dependencies``
    group, PEP 735 ``dependency-groups`` (string entries only — ``include-group``
    tables carry no requirement), and ``build-system.requires``. Parsing real
    TOML tables means unrelated quoted strings — notably ``keywords = ["mlx"]`` —
    are never mistaken for a dependency.
    """
    project = data.get("project", {})
    if isinstance(project, dict):
        deps = project.get("dependencies", [])
        if isinstance(deps, list):
            yield from (d for d in deps if isinstance(d, str))
        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for group in optional.values():
                if isinstance(group, list):
                    yield from (d for d in group if isinstance(d, str))

    groups = data.get("dependency-groups", {})
    if isinstance(groups, dict):
        for group in groups.values():
            if isinstance(group, list):
                yield from (d for d in group if isinstance(d, str))

    build_system = data.get("build-system", {})
    if isinstance(build_system, dict):
        requires = build_system.get("requires", [])
        if isinstance(requires, list):
            yield from (r for r in requires if isinstance(r, str))


def extract_mlx_bounds(
    pyproject_text: str, *, strict: bool = False
) -> dict[str, set[str]]:
    """Map each guarded package to the SET of version specifiers it is pinned at.

    A package can appear in several dependency groups (core / [vision] / [spec] /
    audio), each potentially with its own specifier; we collect the distinct
    normalized specifier strings so a change in ANY occurrence is caught.
    Environment markers (``; platform_system == 'Darwin'``) are ignored — only the
    version specifier is coherence-relevant.

    ``strict`` (used for the PR-head pyproject) raises rather than silently
    skipping when the TOML is malformed or a *guarded* requirement won't parse, so
    a syntax slip can't quietly disable the gate. Non-guarded parse failures are
    always ignored — other CI gates own those.
    """
    try:
        data = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError:
        if strict:
            raise
        return {pkg: set() for pkg in GUARDED_PACKAGES}

    bounds: dict[str, set[str]] = {pkg: set() for pkg in GUARDED_PACKAGES}
    for dep in _iter_declared_dependencies(data):
        try:
            req = Requirement(dep)
        except Exception as exc:
            # Couldn't parse — if the leading name token canonicalizes to a
            # guarded package, fail closed rather than let a syntax slip hide it.
            if strict:
                match = _NAME_PREFIX.match(dep)
                if match and canonicalize_name(match.group(1)) in GUARDED_PACKAGES:
                    raise MalformedGuardedRequirementError(dep) from exc
            continue
        # PEP 503 canonicalization so alternate spellings (``mlx_vlm``,
        # ``MLX.VLM``) map to the same guarded key and can't slip past.
        name = canonicalize_name(req.name)
        if name in GUARDED_PACKAGES:
            # ``str(SpecifierSet)`` normalizes ordering, so ">=0.6.3,<0.7" and
            # "<0.7,>=0.6.3" compare equal — only real specifier changes flag.
            bounds[name].add(str(req.specifier))
    return bounds


def detect_bound_changes(old_text: str, new_text: str) -> list[str]:
    """Descriptions of every guarded-package version-specifier change.

    Fail-closed: ANY change to a guarded package's normalized specifier set is
    reported (see module docstring for why we don't try to classify loosening vs
    tightening). Reordering / marker-only edits normalize away and are silent.
    """
    old = extract_mlx_bounds(old_text)
    new = extract_mlx_bounds(new_text, strict=True)
    changes: list[str] = []
    for pkg in sorted(GUARDED_PACKAGES):
        if old[pkg] == new[pkg]:
            continue
        before = ", ".join(sorted(old[pkg])) or "(absent)"
        after = ", ".join(sorted(new[pkg])) or "(absent)"
        changes.append(f"{pkg}: {before}  ->  {after}")
    return changes


def _attestation_ok(pr_body: str, pr_labels: str, forced: bool) -> bool:
    """True if the PR carries a coherence-sweep attestation (label or trailer)."""
    if forced:
        return True
    labels = {label.strip().lower() for label in pr_labels.split(",") if label.strip()}
    if ATTEST_LABEL in labels:
        return True
    for line in pr_body.splitlines():
        if line.strip().lower().startswith(ATTEST_TRAILER.lower()):
            # Require a non-empty note after the trailer (a URL or run id).
            note = line.split(":", 1)[1].strip()
            if note:
                return True
    return False


def _git_show(ref_path: str) -> str | None:
    """Return the file content at ``ref_path`` (``ref:path``), or None if absent."""
    try:
        return subprocess.check_output(
            ["git", "show", ref_path], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("MLX_BOUND_BASE_REF", "origin/main"),
        help="Git ref of the base branch to diff pyproject.toml against.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to the working-tree pyproject.toml (the PR head version).",
    )
    args = parser.parse_args(argv)

    # Fail CLOSED on infrastructure problems: a required merge gate must never be
    # silently disabled by a ref-plumbing mistake or a syntax slip.
    old_text = _git_show(f"{args.base_ref}:{args.pyproject}")
    if old_text is None:
        print(
            f"[mlx-bound-guard] ERROR: cannot read base {args.base_ref}:{args.pyproject}. "
            "The gate needs the base branch fetched (git fetch origin <base>). "
            "Failing closed.",
            file=sys.stderr,
        )
        return 1

    try:
        with open(args.pyproject, encoding="utf-8") as fh:
            new_text = fh.read()
    except OSError as exc:
        print(
            f"[mlx-bound-guard] ERROR: cannot read {args.pyproject}: {exc}. "
            "Failing closed.",
            file=sys.stderr,
        )
        return 1

    try:
        changes = detect_bound_changes(old_text, new_text)
    except tomllib.TOMLDecodeError as exc:
        print(
            f"[mlx-bound-guard] ERROR: {args.pyproject} is not valid TOML ({exc}). "
            "Failing closed.",
            file=sys.stderr,
        )
        return 1
    except MalformedGuardedRequirementError as exc:
        print(
            f"[mlx-bound-guard] ERROR: unparseable mlx* requirement {str(exc)!r} in "
            f"{args.pyproject}. Failing closed.",
            file=sys.stderr,
        )
        return 1

    if not changes:
        print("[mlx-bound-guard] no mlx/mlx-lm/mlx-vlm bound change — OK.")
        return 0

    forced = os.environ.get("MLX_BOUND_ATTESTED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    attested = _attestation_ok(
        os.environ.get("PR_BODY", ""),
        os.environ.get("PR_LABELS", ""),
        forced,
    )

    print("[mlx-bound-guard] detected mlx* version-bound change(s):")
    for change in changes:
        print(f"    {change}")

    if attested:
        print(
            "[mlx-bound-guard] coherence-sweep attestation present "
            f"(label '{ATTEST_LABEL}' or '{ATTEST_TRAILER}' trailer) — allowed."
        )
        return 0

    print(
        "\n[mlx-bound-guard] BLOCKED (#1248): a change to an mlx/mlx-lm/mlx-vlm "
        "version pin can re-open the upstream-heuristic break that shipped garbage "
        "(#1234).\n"
        "If this LOOSENS a bound (raise/remove a cap, drop an != exclusion, move "
        "the floor), first run the full-family output-coherence sweep (#1247) "
        "across ALL model families — the break lands under EXISTING models, not "
        "only new ones. If it only tightens / adds a cap, a one-line note is "
        "enough. Attest either way by:\n"
        f"  * adding the '{ATTEST_LABEL}' label to this PR, or\n"
        f"  * adding a '{ATTEST_TRAILER} <sweep run url / summary / note>' line to "
        "the PR body.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
