## What does this PR do?

<!-- Brief description of the change. -->

## Why is this needed?

<!--
Required. Link the issue or describe the user-visible problem.
Acceptable: "fixes #123 (parser drops tool_call deltas)", "restores N% TPS regression on model M", "patches CVE-XXXX-NNNN in dep X".
NOT acceptable on its own: "improves coverage", "cleaner code", "good practice", "future-proofs for possible refactor".
PRs without a concrete necessity may be closed. See docs/development/pr_merge_sop.md §Step 0.
-->

## AI assistance disclosure

<!--
Required. Be honest — silence is treated more cautiously than disclosure.
Examples:
- "Fully human."
- "Claude wrote tests; I wrote the implementation."
- "Codex generated the parser; I rewrote ~30% by hand and verified the rest."
-->

> By submitting this PR I confirm I can explain every line of code in it on demand.

## Test plan

<!-- Bullet list of what you ran. -->

## Checklist

- [ ] Tests pass locally (`python3 -m pytest tests/ -x`)
- [ ] Lint passes (`ruff check && ruff format --check`)
- [ ] Self-validated with `python3 -m scripts.pr_validate.pr_validate <PR#>` — see [CONTRIBUTING.md](../CONTRIBUTING.md#self-validating-your-pr) (opt out heavy steps with `PR_VALIDATE_NO_DEEPSEEK=1 PR_VALIDATE_NO_STRESS=1` if you don't have the hardware/keys)
- [ ] If new tests touch a critical code path (parser / scheduler / security), I've spot-checked that they fail when the corresponding production line is broken (see SOP §Step 3)
- [ ] Updated README/docs if applicable
- [ ] No breaking changes to existing API
