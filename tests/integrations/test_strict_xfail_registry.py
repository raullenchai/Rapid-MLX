# SPDX-License-Identifier: Apache-2.0
"""Pin the exact set of strict-xfail cells in the integration matrix.

``test_xfail_audit.py`` (issue #320) enforces that every ``xfail`` marker
is *justified* — ``strict=True`` or a ``strict=False`` reason. It does
NOT enforce *how many* strict-xfails exist, nor *which* cells carry them.
That leaves a coverage-shrinkage hole in the release-artifact acceptance
gate: the 56-cell agent/framework matrix is a required release gate, and a
new genuinely-failing cell could be silently muted by adding one more
strict-xfail. Because ``test_xfail_audit`` only checks strict-vs-not, that
new xfail would pass the audit and quietly shrink the number of cells that
must actually PASS — exactly the class of regression the matrix gate is
supposed to prevent.

This test closes that hole with an **independent, checked-in snapshot** of
the exact strict-xfail nodeids (``_EXPECTED_STRICT_XFAIL_NODEIDS`` below).
It collects the two integration-matrix modules, computes the set of cells
that ``conftest.pytest_collection_modifyitems`` marks ``xfail(strict=
True)``, and asserts that set equals the snapshot exactly.

The snapshot is deliberately NOT derived from the conftest registration
constants. An earlier draft rebuilt the expected set from those same
constants; codex flagged (correctly) that this is tautological — swapping
one strict-xfail for another *within the same family* preserves the count
AND keeps the derived expected set in lock-step with the hook, so the test
stays green while coverage silently shifts. A hardcoded snapshot removes
that blind spot: ANY membership change (add, remove, or swap) diverges
from the snapshot and fails here, forcing an explicit, reviewed edit to
this file.

The snapshot pins:

  1. **Exact membership** — every nodeid must match. Adding, removing, or
     swapping a strict-xfail cell fails the set-equality assertion.
  2. **Total count** (24) and **per-family breakdown** (9 / 1 / 14) — a
     redundant tripwire that makes the coverage delta obvious in the
     failure message even before the reviewer diffs the nodeid set.

This is intentionally a pure-collection audit — no server, no model boot,
no test bodies execute — so it runs in every ``make smoke`` / pr_validate
cycle alongside ``test_xfail_audit``.
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------- #
# Independent, checked-in snapshot of every strict-xfail matrix cell.
# --------------------------------------------------------------------------- #
#
# 24 cells total, one strict-xfail per line. Grouped by the reason family
# (see tests/integrations/conftest.py for the root-cause of each group).
# Editing this set is the ONLY sanctioned way to add/remove a strict-xfail
# in the matrix — the assertion below fails until this snapshot matches the
# markers the conftest hook actually applies.

# 9 × DeepSeek R1-Distill tool-call cells — R1 distillation dropped OpenAI
# tool_call emission (conftest ``_DEEPSEEK_R1_TOOLCALL_XFAIL_NODEIDS``).
_DEEPSEEK_STRICT_XFAIL = frozenset(
    {
        "tests/integrations/test_agents_matrix.py::TestOpenCode::test_smoke[deepseek]",
        "tests/integrations/test_agents_matrix.py::TestQwenCode::test_smoke[deepseek]",
        "tests/integrations/test_agents_matrix.py::TestHermesAgent::test_smoke[deepseek]",
        "tests/integrations/test_agents_matrix.py::TestKiloCode::test_smoke[deepseek]",
        "tests/integrations/test_agents_matrix.py::TestCopilot::test_smoke[deepseek]",
        "tests/integrations/test_agents_matrix.py::TestDroid::test_smoke[deepseek]",
        "tests/integrations/test_agents_matrix.py::TestKimiCode::test_smoke[deepseek]",
        "tests/integrations/test_frameworks_matrix.py::TestLangChain::test_smoke[deepseek]",
        "tests/integrations/test_frameworks_matrix.py::TestPydanticAI::test_smoke[deepseek]",
    }
)

# 1 × gpt-oss × OpenHands cell — harmony ↔ CodeActAgent text-action mismatch.
_GPTOSS_STRICT_XFAIL = frozenset(
    {
        "tests/integrations/test_agents_matrix.py::TestOpenHands::test_smoke[gptoss]",
    }
)

# 14 × Hy3 cells — 166 GB single-node-infeasible, Ultra-only (family-wide:
# every agent + framework class × [hy3]).
_HY3_STRICT_XFAIL = frozenset(
    {
        "tests/integrations/test_agents_matrix.py::TestCodexCLI::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestClaudeCode::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestOpenCode::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestQwenCode::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestOpenHands::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestHermesAgent::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestAider::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestKiloCode::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestCopilot::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestDroid::test_smoke[hy3]",
        "tests/integrations/test_agents_matrix.py::TestKimiCode::test_smoke[hy3]",
        "tests/integrations/test_frameworks_matrix.py::TestLangChain::test_smoke[hy3]",
        "tests/integrations/test_frameworks_matrix.py::TestPydanticAI::test_smoke[hy3]",
        "tests/integrations/test_frameworks_matrix.py::TestSmolagents::test_smoke[hy3]",
    }
)

_EXPECTED_STRICT_XFAIL_NODEIDS = (
    _DEEPSEEK_STRICT_XFAIL | _GPTOSS_STRICT_XFAIL | _HY3_STRICT_XFAIL
)

# Redundant count tripwires — make the coverage delta obvious even before
# the reviewer diffs the nodeid set. Guarded against snapshot typos below.
_EXPECTED_DEEPSEEK_COUNT = 9
_EXPECTED_GPTOSS_COUNT = 1
_EXPECTED_HY3_COUNT = 14
_EXPECTED_STRICT_XFAIL_COUNT = 24


class _StrictXfailCollector:
    """Pytest plugin that records collected + strict-xfail-marked nodeids."""

    def __init__(self) -> None:
        self.all_nodeids: set[str] = set()
        self.strict_xfail_nodeids: set[str] = set()

    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        # Runs AFTER conftest.pytest_collection_modifyitems (plugin order),
        # so the strict-xfail markers the conftest applies are visible here.
        del config
        for item in items:
            self.all_nodeids.add(item.nodeid)
            for marker in item.iter_markers(name="xfail"):
                if marker.kwargs.get("strict") is True:
                    self.strict_xfail_nodeids.add(item.nodeid)
                    break


def _collect_matrix() -> _StrictXfailCollector:
    collector = _StrictXfailCollector()
    ret = pytest.main(
        [
            "tests/integrations/test_agents_matrix.py",
            "tests/integrations/test_frameworks_matrix.py",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        plugins=[collector],
    )
    # Collection-only must succeed (0 = OK; 5 = no tests collected is a bug here).
    assert ret == 0, f"matrix collection failed with pytest exit code {ret}"
    assert collector.all_nodeids, "matrix collected zero cells — harness broke"
    return collector


def test_snapshot_internal_consistency():
    """The checked-in snapshot itself must be internally consistent.

    Catches a typo in ``_EXPECTED_STRICT_XFAIL_NODEIDS`` (e.g. a duplicate
    line silently collapsed by ``frozenset``) before it can mask a real
    coverage drift in the assertion below.
    """
    assert len(_DEEPSEEK_STRICT_XFAIL) == _EXPECTED_DEEPSEEK_COUNT
    assert len(_GPTOSS_STRICT_XFAIL) == _EXPECTED_GPTOSS_COUNT
    assert len(_HY3_STRICT_XFAIL) == _EXPECTED_HY3_COUNT
    assert len(_EXPECTED_STRICT_XFAIL_NODEIDS) == _EXPECTED_STRICT_XFAIL_COUNT
    # The three family sets must be disjoint (a nodeid belongs to one family).
    assert not (_DEEPSEEK_STRICT_XFAIL & _GPTOSS_STRICT_XFAIL)
    assert not (_DEEPSEEK_STRICT_XFAIL & _HY3_STRICT_XFAIL)
    assert not (_GPTOSS_STRICT_XFAIL & _HY3_STRICT_XFAIL)


def test_strict_xfail_set_is_pinned():
    """The applied strict-xfail set must equal the checked-in snapshot exactly."""
    collector = _collect_matrix()
    applied = collector.strict_xfail_nodeids
    expected = set(_EXPECTED_STRICT_XFAIL_NODEIDS)

    # Every snapshot nodeid must actually be collectable — else the snapshot
    # references a renamed/deleted cell and the audit is silently vacuous.
    stale = expected - collector.all_nodeids
    assert not stale, (
        "snapshot references nodeid(s) that no longer exist in the matrix — "
        "a class was renamed/removed without updating this snapshot:\n"
        + "\n".join(f"  ? {n}" for n in sorted(stale))
    )

    # --- Membership: exact set equality (catches add / remove / swap) ---- #
    unexpected = applied - expected
    missing = expected - applied
    assert not unexpected, (
        "NEW strict-xfail cell(s) not in the checked-in snapshot — a "
        "genuinely-failing matrix cell may have been silently muted, "
        "shrinking the required-PASS coverage of the release-artifact matrix "
        "gate. Add it to _EXPECTED_STRICT_XFAIL_NODEIDS (and bump the counts) "
        "in test_strict_xfail_registry.py, and justify the coverage delta in "
        "the PR:\n" + "\n".join(f"  + {n}" for n in sorted(unexpected))
    )
    assert not missing, (
        "Snapshot strict-xfail cell(s) are NOT being applied by the conftest "
        "hook — either a registration constant drifted or a cell now passes. "
        "Reconcile the snapshot with conftest.pytest_collection_modifyitems:\n"
        + "\n".join(f"  - {n}" for n in sorted(missing))
    )

    # --- Count: redundant total + per-family tripwire -------------------- #
    assert len(applied) == _EXPECTED_STRICT_XFAIL_COUNT, (
        f"strict-xfail cell count changed: expected "
        f"{_EXPECTED_STRICT_XFAIL_COUNT}, found {len(applied)}. Adding or "
        f"removing a strict-xfail must be an explicit, reviewed change — "
        f"update _EXPECTED_STRICT_XFAIL_NODEIDS (and the per-family counts) "
        f"in test_strict_xfail_registry.py, and justify the coverage delta."
    )
    applied_deepseek = {n for n in applied if "[deepseek]" in n}
    applied_gptoss = {n for n in applied if "[gptoss]" in n}
    applied_hy3 = {n for n in applied if "[hy3]" in n}
    assert len(applied_deepseek) == _EXPECTED_DEEPSEEK_COUNT, (
        f"DeepSeek strict-xfail count changed: expected "
        f"{_EXPECTED_DEEPSEEK_COUNT}, found {len(applied_deepseek)}."
    )
    assert len(applied_gptoss) == _EXPECTED_GPTOSS_COUNT, (
        f"gpt-oss strict-xfail count changed: expected "
        f"{_EXPECTED_GPTOSS_COUNT}, found {len(applied_gptoss)}."
    )
    assert len(applied_hy3) == _EXPECTED_HY3_COUNT, (
        f"Hy3 strict-xfail count changed: expected {_EXPECTED_HY3_COUNT}, "
        f"found {len(applied_hy3)}."
    )
