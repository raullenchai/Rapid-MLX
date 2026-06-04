# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for the harmony control-token allowlist.

Issue cluster #444 / #455 / #468 / #480 all turn on "did a harmony
control marker leak into a content delta?". Each regression file needs
the same allowlist; duplicating it inline would let a partial fix slip
through when a sibling file forgets to add a newly-leaking marker.

The canonical list is in ``vllm_mlx/tool_parsers/harmony_tool_parser.py``
inside ``_strip_control_tokens``. Adding a new token there without
updating this list would mask a downstream regression — the test
:func:`tests/parsers/test_infra_smoke.py::test_harmony_markers_match_source`
pins that the lists agree.
"""

from __future__ import annotations

# Control tokens stripped by `_strip_control_tokens` (the authoritative
# list lives at harmony_tool_parser.py:225-233).
HARMONY_CONTROL_TOKENS: tuple[str, ...] = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)

# Channel labels — also stripped by `_strip_control_tokens` (lines
# 240-241) but worth pinning separately because some bugs leak the
# channel label without the surrounding `<|channel|>...<|message|>`
# brackets.
HARMONY_CHANNEL_LABELS: tuple[str, ...] = ("analysis", "commentary", "final")

# Tool-call recipient prefix — stripped at harmony_tool_parser.py:242.
# Used as a leak marker: any content delta containing this prefix means
# the harmony recipient header bled through.
HARMONY_RECIPIENT_PREFIX: str = "to=functions."

# Union of all leakable substrings used by regression-test invariants.
# Order doesn't matter — the assert is "no marker appears in content".
HARMONY_LEAK_MARKERS: tuple[str, ...] = (
    *HARMONY_CONTROL_TOKENS,
    HARMONY_RECIPIENT_PREFIX,
)


def assert_no_harmony_marker_leak(content: str | None, *, context: str = "") -> None:
    """Assert that no harmony control marker leaked into ``content``.

    A single helper used by every harmony regression test (issues
    #444 / #455 / #468 / #480) so a fix that closes 5 of the 6 by
    being stricter on one marker but loose on another can't pass
    silently.

    Args:
        content: The accumulated content delta string (None is fine —
            "no content emitted" is always leak-free).
        context: Optional context string (test name, stream_interval,
            etc.) included in the failure message to make the
            parametrize id obvious.
    """
    if content is None or content == "":
        return
    leaked = [m for m in HARMONY_LEAK_MARKERS if m in content]
    assert not leaked, (
        f"Harmony control marker(s) leaked into content delta: "
        f"{leaked!r}. context={context!r} content={content!r}"
    )
