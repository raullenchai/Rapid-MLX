# SPDX-License-Identifier: Apache-2.0
"""Keep bundled official JavaScript actions on reviewed Node 24 releases."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
USES_RE = re.compile(r"uses:\s*(actions/[\w-]+)@([0-9a-f]{40})")

# These immutable SHAs were verified against the official action manifests;
# each declares ``runs.using: node24``.  Updating one is an explicit dependency
# review: verify its manifest/runtime and breaking changes, then update this
# allowlist together with the workflow pins.
NODE24_ACTIONS = {
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",  # v7.0.1
    "actions/download-artifact": "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",  # v8.0.1
    "actions/github-script": "3a2844b7e9c422d3c10d287c895573f7108da1b3",  # v9.0.0
}


def test_reviewed_node24_action_pins_are_used_consistently() -> None:
    seen: set[str] = set()
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        for action, sha in USES_RE.findall(path.read_text()):
            expected = NODE24_ACTIONS.get(action)
            if expected is None:
                continue
            seen.add(action)
            assert sha == expected, (
                f"{path.relative_to(ROOT)} uses unreviewed {action}@{sha}; "
                f"expected reviewed Node 24 pin {expected}"
            )

    assert seen == set(NODE24_ACTIONS)
