# SPDX-License-Identifier: Apache-2.0
"""Shared state passed to every pipeline step.

A single Context lives for the duration of one ``pr_validate`` run. Each
step reads what it needs and may append to ``results``. Steps must NOT
mutate fields other than ``results`` — anything else is shared input.

Blast-radius classification is what gates the expensive steps (full unit
suite, stress, e2e, bench). Keep the rule simple and grep-able rather
than learned: a small fixed set of paths is "high blast" because a
regression there hits every request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .base import StepResult


BlastRadius = Literal["low", "medium", "high"]


# Files / directories that affect every request → any change requires
# the heavy gate (full unit + stress + e2e + bench). Order doesn't
# matter; we use ``startswith`` matching against repo-relative paths.
HIGH_BLAST_PATHS = (
    "vllm_mlx/scheduler.py",
    "vllm_mlx/server.py",
    "vllm_mlx/cli.py",
    "vllm_mlx/engine_core.py",
    "vllm_mlx/memory_cache.py",
    "vllm_mlx/prefix_cache.py",
    "vllm_mlx/engine/",
    "vllm_mlx/runtime/",
    "vllm_mlx/routes/",
    "vllm_mlx/middleware/",
    "vllm_mlx/turboquant.py",
    "vllm_mlx/mllm_scheduler.py",
    "pyproject.toml",  # version, deps, build config
)

# Paths that are "code but isolated" — touching them needs unit tests
# but not the full stress/e2e battery. Parsers, models, agents.
MEDIUM_BLAST_PATHS = (
    "vllm_mlx/",  # anything under vllm_mlx not caught by high
    "tests/",  # test changes themselves get the unit suite
)

# Paths considered safe — docs, scripts (except validation itself),
# examples. Drive-by typo PRs land here.
LOW_BLAST_PATHS = (
    "docs/",
    "README",
    ".github/ISSUE_TEMPLATE/",  # NOT .github/workflows — that's a supply-chain risk
    "examples/",
    "evals/",
    "harness/baselines/",
)


@dataclass
class Context:
    """Per-run state. Constructed by the fetch step, read by everyone."""

    pr_number: int
    repo: str = "raullenchai/Rapid-MLX"
    base_branch: str = "main"

    # Populated by the fetch step:
    pr_title: str = ""
    pr_body: str = ""
    pr_author: str = ""
    # True iff the PR head is on a fork. Used as the "external author"
    # proxy because gh's --json output doesn't expose authorAssociation.
    pr_is_external: bool = False
    head_sha: str = ""
    head_branch: str = ""
    base_sha: str = ""
    diff_path: str = ""  # path to full diff on disk (lazy — large diffs OK)
    files_changed: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0

    # Working directory for artifacts (one per run, kept for inspection).
    work_dir: Path = field(default_factory=lambda: Path("/tmp/pr_validate"))

    # Repo root (set in __post_init__).
    repo_root: Path = field(init=False)

    # Step results accumulate here in execution order.
    results: list[StepResult] = field(default_factory=list)

    # CLI flags / config knobs — keep small.
    verbose: bool = False

    def __post_init__(self) -> None:
        # Repo root = current working directory by convention. Validator
        # is meant to be run from the repo root; bail loudly otherwise.
        cwd = Path.cwd()
        if not (cwd / "pyproject.toml").exists():
            raise RuntimeError(
                f"pr_validate must run from the repo root (no pyproject.toml in {cwd})"
            )
        self.repo_root = cwd

    # ------------------------------------------------------------------
    # Derived properties — computed from files_changed once it's set
    # ------------------------------------------------------------------

    @property
    def blast_radius(self) -> BlastRadius:
        """Classify the PR's blast radius.

        Highest match wins: a PR that touches both docs and scheduler.py
        is HIGH (the docs change is irrelevant — the scheduler change
        decides the gating). LOW only if EVERY changed file is in the
        low-risk allowlist; anything outside that bumps to medium.
        """
        if not self.files_changed:
            return "low"
        for path in self.files_changed:
            if any(path.startswith(p) for p in HIGH_BLAST_PATHS):
                return "high"
        # If we reach here, no high-blast file. Check if everything is
        # in the low set; otherwise medium.
        all_low = all(
            any(path.startswith(p) for p in LOW_BLAST_PATHS)
            for path in self.files_changed
        )
        return "low" if all_low else "medium"

    @property
    def is_external_author(self) -> bool:
        """True for non-collaborator authors. External PRs get extra
        scrutiny in the supply-chain step (since maintainers can't be
        assumed to recognize the contributor). Approximated via
        ``pr_is_external`` (PR head is on a fork) — this misses the
        rare case of a collaborator pushing a fork-based PR, but the
        false-positive is in the safe direction."""
        return self.pr_is_external

    def artifact_path(self, name: str) -> Path:
        """Return a path inside the run's work_dir, creating parent
        dirs. Steps use this for log files etc."""
        p = self.work_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def run_log(self, message: str) -> None:
        """Lightweight progress log — goes to stderr so it doesn't
        contaminate the scorecard markdown on stdout."""
        import sys

        print(f"  · {message}", file=sys.stderr)


def env_truthy(name: str) -> bool:
    """Helper for env-var feature flags (``PR_VALIDATE_NO_CODEX=1``
    style)."""
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")
