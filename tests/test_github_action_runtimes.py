# SPDX-License-Identifier: Apache-2.0
"""Keep bundled official JavaScript actions on reviewed Node 24 releases."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
# Capture every ref first. Restricting the regex to a SHA would make a mutable
# tag invisible and could let another valid occurrence satisfy ``seen``.
USES_RE = re.compile(r"uses:\s*(actions/[\w-]+)@([^\s#]+)")

# These immutable SHAs were verified against the official action manifests;
# each declares ``runs.using: node24``.  Updating one is an explicit dependency
# review: verify its manifest/runtime and breaking changes, then update this
# allowlist together with the workflow pins.
NODE24_ACTIONS = {
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",  # v7.0.1
    "actions/download-artifact": "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",  # v8.0.1
    "actions/github-script": "3a2844b7e9c422d3c10d287c895573f7108da1b3",  # v9.0.0
}


def assert_reviewed_node24_refs(path: Path, text: str, seen: set[str]) -> None:
    for action, ref in USES_RE.findall(text):
        expected = NODE24_ACTIONS.get(action)
        if expected is None:
            continue
        seen.add(action)
        assert ref == expected, (
            f"{path} uses unreviewed {action}@{ref}; "
            f"expected reviewed Node 24 pin {expected}"
        )


def test_reviewed_node24_action_pins_are_used_consistently() -> None:
    seen: set[str] = set()
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        assert_reviewed_node24_refs(path.relative_to(ROOT), path.read_text(), seen)

    assert seen == set(NODE24_ACTIONS)


@pytest.mark.parametrize(
    "ref",
    (
        "v7",
        "043FB46D1A93C77AAE656E7C1C64A875D1FC6A0A",
        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0",
    ),
)
def test_mutable_or_malformed_targeted_action_ref_fails(ref: str) -> None:
    with pytest.raises(AssertionError, match="uses unreviewed actions/upload-artifact"):
        assert_reviewed_node24_refs(
            Path("workflow.yml"),
            f"- uses: actions/upload-artifact@{ref}",
            set(),
        )
