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

This test closes that hole. It collects the two integration-matrix
modules, computes the set of cells that ``conftest.pytest_collection_
modifyitems`` marks ``xfail(strict=True)``, and asserts it equals an
*expected* set derived from the conftest registration constants. The
expected set is pinned three ways:

  1. **Membership** — the exact nodeids must match. A new strict-xfail on
     a cell that is not in a registration constant fails here.
  2. **Count** — the total is asserted against ``_EXPECTED_STRICT_XFAIL_
     COUNT``. Adding or removing any strict-xfail cell (even a legitimate
     one) forces an explicit edit to that number, so the change is
     reviewed rather than silent.
  3. **Per-family breakdown** — the DeepSeek / gpt-oss / Hy3 sub-counts
     are pinned individually, so a shift *between* families (e.g. a new
     DeepSeek cell balanced by a dropped Hy3 cell that keeps the total
     the same) is still caught.

Net effect: adding or removing ANY strict-xfail in the matrix now requires
an explicit change to this test, which surfaces the coverage delta in
review. This is intentionally a pure-collection audit — no server, no
model boot, no test bodies execute — so it runs in every ``make smoke`` /
pr_validate cycle alongside ``test_xfail_audit``.
"""

from __future__ import annotations

import pytest

from tests.integrations import conftest as matrix_conftest

# --------------------------------------------------------------------------- #
# Expected strict-xfail set, derived from the conftest registration constants.
# --------------------------------------------------------------------------- #
#
# The matrix parametrizes every cell over a single ``family`` argument, so a
# strict-xfail nodeid is always ``<module>::<Class>::test_smoke[<family>]``.
# We rebuild the expected set from the same constants the conftest hook
# consumes, so the two cannot drift: if a registration constant changes,
# the expected set changes with it — but the pinned counts below still force
# an explicit acknowledgement of the coverage delta.

# The matrix modules that carry family-parametrized cells (mirrors the
# conftest ``_INTEGRATION_MATRIX_MODULES`` gate).
_MATRIX_MODULES = ("test_agents_matrix.py", "test_frameworks_matrix.py")

# All matrix classes, per module — the universe the Hy3 family-wide rule
# spans. Sourced by collecting the modules (below), NOT hardcoded, so a new
# agent/framework class is automatically included in the Hy3 expectation.


def _expected_deepseek_nodeids(all_nodeids: set[str]) -> set[str]:
    """9 DeepSeek R1-Distill tool-call cells.

    Registered in ``conftest._DEEPSEEK_R1_TOOLCALL_XFAIL_NODEIDS`` as
    ``<module>::<Class>`` prefixes; the applied marker lands on the
    ``[deepseek]`` param of each.
    """
    out: set[str] = set()
    for nodeid in all_nodeids:
        if "[deepseek]" not in nodeid:
            continue
        if any(
            prefix in nodeid
            for prefix in matrix_conftest._DEEPSEEK_R1_TOOLCALL_XFAIL_NODEIDS
        ):
            out.add(nodeid)
    return out


def _expected_gptoss_nodeids(all_nodeids: set[str]) -> set[str]:
    """1 gpt-oss × OpenHands cell (harmony ↔ CodeActAgent format mismatch)."""
    nodeid_frag = matrix_conftest._GPTOSS_OPENHANDS_XFAIL_NODEID
    family = matrix_conftest._GPTOSS_OPENHANDS_XFAIL_FAMILY
    return {
        nodeid
        for nodeid in all_nodeids
        if nodeid_frag in nodeid and f"[{family}]" in nodeid
    }


def _expected_hy3_nodeids(all_nodeids: set[str]) -> set[str]:
    """Every Hy3 matrix cell (family-wide, 166 GB Ultra-only)."""
    family = matrix_conftest._HY3_XFAIL_FAMILY
    return {
        nodeid
        for nodeid in all_nodeids
        if f"[{family}]" in nodeid
        and any(module in nodeid for module in _MATRIX_MODULES)
    }


# Pinned counts. These are the tripwire: changing the strict-xfail set
# without editing these numbers fails the test.
_EXPECTED_DEEPSEEK_COUNT = 9
_EXPECTED_GPTOSS_COUNT = 1
_EXPECTED_HY3_COUNT = 14
_EXPECTED_STRICT_XFAIL_COUNT = (
    _EXPECTED_DEEPSEEK_COUNT + _EXPECTED_GPTOSS_COUNT + _EXPECTED_HY3_COUNT
)  # 24


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
    # Collection-only must succeed (0 = OK, 5 = no tests collected is a bug here).
    assert ret == 0, f"matrix collection failed with pytest exit code {ret}"
    assert collector.all_nodeids, "matrix collected zero cells — harness broke"
    return collector


def test_strict_xfail_set_is_pinned():
    """The applied strict-xfail set must equal the registered set exactly."""
    collector = _collect_matrix()
    applied = collector.strict_xfail_nodeids

    expected_deepseek = _expected_deepseek_nodeids(collector.all_nodeids)
    expected_gptoss = _expected_gptoss_nodeids(collector.all_nodeids)
    expected_hy3 = _expected_hy3_nodeids(collector.all_nodeids)
    expected = expected_deepseek | expected_gptoss | expected_hy3

    # --- Membership: exact set equality --------------------------------- #
    unexpected = applied - expected
    missing = expected - applied
    assert not unexpected, (
        "NEW strict-xfail cell(s) not registered in conftest — a genuinely-"
        "failing matrix cell may have been silently muted, shrinking the "
        "required-PASS coverage of the release-artifact matrix gate. Register "
        "it in the appropriate conftest constant and update the pinned counts "
        "in test_strict_xfail_registry.py:\n"
        + "\n".join(f"  + {n}" for n in sorted(unexpected))
    )
    assert not missing, (
        "Registered strict-xfail cell(s) are NOT being applied — the conftest "
        "marker hook or a registration constant drifted. Reconcile "
        "conftest.pytest_collection_modifyitems with the registration "
        "constants:\n" + "\n".join(f"  - {n}" for n in sorted(missing))
    )

    # --- Count: total tripwire ------------------------------------------ #
    assert len(applied) == _EXPECTED_STRICT_XFAIL_COUNT, (
        f"strict-xfail cell count changed: expected "
        f"{_EXPECTED_STRICT_XFAIL_COUNT}, found {len(applied)}. Adding or "
        f"removing a strict-xfail must be an explicit, reviewed change — "
        f"update _EXPECTED_STRICT_XFAIL_COUNT (and the per-family count) in "
        f"test_strict_xfail_registry.py, and justify the coverage delta in "
        f"the PR."
    )

    # --- Per-family breakdown: catch same-total shifts ------------------ #
    assert len(expected_deepseek) == _EXPECTED_DEEPSEEK_COUNT, (
        f"DeepSeek strict-xfail count changed: expected "
        f"{_EXPECTED_DEEPSEEK_COUNT}, found {len(expected_deepseek)}."
    )
    assert len(expected_gptoss) == _EXPECTED_GPTOSS_COUNT, (
        f"gpt-oss strict-xfail count changed: expected "
        f"{_EXPECTED_GPTOSS_COUNT}, found {len(expected_gptoss)}."
    )
    assert len(expected_hy3) == _EXPECTED_HY3_COUNT, (
        f"Hy3 strict-xfail count changed: expected {_EXPECTED_HY3_COUNT}, "
        f"found {len(expected_hy3)}."
    )
