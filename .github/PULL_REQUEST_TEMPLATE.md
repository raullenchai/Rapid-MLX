## Why

<!--
Required — the maintainer will request fill-in if missing before review begins.
Link the issue or describe the user-visible problem (or concrete maintenance value for typo / docs / alias bookkeeping PRs).
Strong: "fixes #123 (parser drops tool_call deltas)", "restores N% TPS regression on model M", "patches CVE-XXXX-NNNN in dep X", "adds alias for Qwen3.5-9B (someone's trying to serve it)".
Not on their own: "improves coverage", "cleaner code", "good practice", "future-proofs for possible refactor".
PRs without a concrete necessity may be closed. See docs/development/pr_merge_sop.md §Step 0 for the carveout list.
-->

## Scope

<!-- What this PR actually changes — the concrete behavior, files, or surfaces in scope. -->

## Non-goals

<!--
Required — part of the six-field PR contract (Why / Scope / Non-goals / Acceptance /
Verification / Behaviour delta). What this PR deliberately does NOT do, so reviewers
don't ask "why didn't you also fix X". Say it plainly even when it seems obvious;
"none" is a valid answer for a tiny change.
-->

## Acceptance

<!--
The observable contract this PR is expected to satisfy — the user-visible outcomes a reviewer can check.
Prefer concrete, testable statements tied to the tests/verification below.
-->

## Verification

<!-- Bullet list of what you ran and what it showed (tests, lint, self-validation). -->

## Behaviour delta

<!--
Required when this PR changes a DEFAULT, a running lane, or a policy that users or the release pipeline depend on.
Otherwise delete this section. Name the before → after so a reader can see the intended behaviour shift at a glance.
Example: "`rapid-mlx serve` default port 8000 → 8080 on `qwen3.5-*` models."
-->

## AI assistance disclosure

<!--
Required — the maintainer will request fill-in if missing before review begins.
Tell us: which files were AI-touched, the AI's role (wrote / reviewed / suggested fix), and how you verified the output.
We don't ask for prompt transcripts.
Examples:
- "Fully human."
- "Claude wrote tests in tests/test_foo.py; I wrote the implementation in vllm_mlx/foo.py and reviewed each test against the spec."
- "Codex generated the parser skeleton; I rewrote ~30% by hand, ran the targeted tests, verified output against 5 sample inputs."
-->

> By submitting this PR I confirm I can explain the intent, risk, and behavior of every non-generated change in this PR. For any generated / boilerplate / scaffolded sections, I've identified them above and can describe how I verified them.

## Checklist

- [ ] Tests pass locally (`python3 -m pytest tests/ -x`)
- [ ] Lint passes (`ruff check && ruff format --check`)
- [ ] Self-validated with `python3 -m scripts.pr_validate.pr_validate <PR#>` — see [CONTRIBUTING.md](../CONTRIBUTING.md#self-validating-your-pr) (opt out heavy steps with `PR_VALIDATE_NO_CODEX=1 PR_VALIDATE_NO_STRESS=1` if you don't have the hardware / a codex login)
- [ ] If new tests touch a critical code path (parser / scheduler / security), I've spot-checked that they fail when the corresponding production line is broken (see SOP §Step 3)
- [ ] Updated README/docs if applicable
- [ ] No breaking changes to existing API
