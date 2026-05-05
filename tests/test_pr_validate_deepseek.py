# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek review step's pure helpers.

The DeepSeek API call itself is integration-level (requires a key, hits
the network) and lives in ``scripts/pr_validate/steps/deepseek_review.py``.
We test only the helpers that have isolated logic:

- ``_truncate_diff_at_file_boundary`` — file-boundary aware diff cap
- ``_gather_directory_context`` path filter (via the inline normalization
  block; we exercise it through a small repro)

These were the surfaces that PR review (round 2 of #202's fix) caught
real bugs in: regex missing git's quoted-filename form, ``startswith("..")``
over-filter, and the ``.``-current-dir leak.
"""

from __future__ import annotations

import os

import pytest

from scripts.pr_validate.steps.deepseek_review import (
    _is_safe_listing_path,
    _truncate_diff_at_file_boundary,
)


def _block(name: str, lines: int = 2000) -> str:
    """A fake unified diff for a single file. Each line is ~60 bytes so a
    2000-line block is ~120KB."""
    body = "\n".join(f"+line {i} " + "x" * 50 for i in range(lines))
    return (
        f"diff --git a/{name} b/{name}\n"
        f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n{body}\n"
    )


def _quoted_block(name: str, lines: int = 2000) -> str:
    """Same as ``_block`` but emits git's quoted-filename header form
    that the original regex (``a/(.+?) b/``) failed to match."""
    body = "\n".join(f"+line {i} " + "x" * 50 for i in range(lines))
    return (
        f'diff --git "a/{name}" "b/{name}"\n'
        f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n{body}\n"
    )


class TestTruncateDiffAtFileBoundary:
    """``_truncate_diff_at_file_boundary`` returns ``(kept, omitted, truncated)``.

    Truncation must happen at file boundaries (``diff --git`` headers) so
    DeepSeek never sees a half-cut file diff. Files that don't fit must be
    listed by name in ``omitted`` so the prompt can name them.
    """

    def test_short_diff_returned_untouched(self):
        diff = _block("foo.py", 10)
        kept, omitted, truncated = _truncate_diff_at_file_boundary(diff, 120_000)
        assert kept == diff
        assert omitted == []
        assert truncated is False

    def test_truncates_at_file_boundary_not_byte(self):
        # Exercise the file-boundary branch: small first file fits cleanly,
        # large second file overflows. We expect file A to be returned in
        # full (ending exactly at file B's header), file B fully omitted.
        a = _block("scripts/small.py", 200)  # ~12KB
        b = _block("vllm_mlx/anthropic.py", 3000)  # ~180KB
        diff = a + b

        kept, omitted, truncated = _truncate_diff_at_file_boundary(diff, 100_000)

        assert truncated is True
        assert omitted == ["vllm_mlx/anthropic.py"]
        # Kept content must end at the boundary — last byte is the newline
        # that terminates file A's last hunk line, just before file B's
        # ``diff --git`` header.
        assert kept.endswith("\n"), f"kept tail: {kept[-50:]!r}"
        # File A's complete diff is in there; file B is not.
        assert kept.count("diff --git ") == 1
        assert "anthropic.py" not in kept

    def test_quoted_filename_recognized(self):
        """Bug fixed in #209: regex was ``a/(.+?) b/`` which doesn't match
        ``"a/foo bar.py" "b/foo bar.py"``. Files with spaces would be invisible
        to the boundary detector → could cut mid-file silently."""
        a = _block("scripts/regular.py", 2000)
        b = _quoted_block("vllm_mlx/file with space.py", 2000)
        diff = a + b

        _kept, omitted, truncated = _truncate_diff_at_file_boundary(diff, 120_000)

        assert truncated is True
        assert "vllm_mlx/file with space.py" in omitted

    def test_first_file_overflows_falls_back_to_raw_slice(self):
        """If the first (and only) file is bigger than the limit, we have
        no boundary to cut at — raw-slice and signal truncation. omitted
        is empty because there are no fully-skipped files."""
        huge = _block("only.py", 3000)  # ~180KB
        kept, omitted, truncated = _truncate_diff_at_file_boundary(huge, 120_000)

        assert truncated is True
        assert omitted == []
        # Raw-sliced near the byte limit.  Use ``<=`` rather than ``==``
        # because ``errors="ignore"`` will drop a trailing incomplete UTF-8
        # sequence (1-3 bytes) if the cap lands mid-codepoint.  Test data
        # here is pure ASCII so today the equality holds, but the contract
        # is "≤ max_bytes", not "exactly max_bytes".
        kept_bytes = len(kept.encode())
        assert kept_bytes <= 120_000
        assert kept_bytes >= 120_000 - 3  # never drop more than a code point

    def test_first_file_overflows_with_more_files_lists_them_omitted(self):
        """First file alone overflows AND there are subsequent files —
        first is partially shown, rest are listed as omitted."""
        a = _block("scripts/big.py", 3000)  # ~180KB on its own
        b = _block("vllm_mlx/anthropic.py", 5)
        c = _block("vllm_mlx/completions.py", 5)
        diff = a + b + c

        kept, omitted, truncated = _truncate_diff_at_file_boundary(diff, 120_000)

        assert truncated is True
        assert omitted == ["vllm_mlx/anthropic.py", "vllm_mlx/completions.py"]
        # First file is partially shown; we don't promise its boundary.
        assert len(kept.encode()) == 120_000

    def test_unicode_path_byte_count(self):
        """``len(str)`` counts code points; the API budget is bytes. Make
        sure a diff with multi-byte chars doesn't silently exceed."""
        # Each char is 3 UTF-8 bytes. 50000 chars = 150000 bytes > 120K.
        body = "\n".join(f"+行 {i}" + "汉" * 100 for i in range(500))
        diff = (
            f"diff --git a/cjk.py b/cjk.py\n"
            f"--- a/cjk.py\n+++ b/cjk.py\n@@ -1 +1 @@\n{body}\n"
        )
        # Diff string length is small in chars; byte length is what matters.
        assert len(diff.encode()) > 120_000

        kept, omitted, truncated = _truncate_diff_at_file_boundary(diff, 120_000)
        assert truncated is True
        # Kept must fit inside the byte budget.
        assert len(kept.encode()) <= 120_000

    def test_no_diff_headers_at_all(self):
        """Defensive: input that doesn't look like a unified diff (e.g.
        someone passed plain text). We raw-slice, no crash, no omitted."""
        garbage = "x" * 200_000  # not a diff
        kept, omitted, truncated = _truncate_diff_at_file_boundary(garbage, 120_000)

        assert truncated is True
        assert omitted == []
        assert len(kept.encode()) == 120_000


class TestPathFilter:
    """``_is_safe_listing_path`` must reject path-traversal attempts and
    ``.``/``..`` while accepting legitimate names that happen to start with
    two dots (``..hidden``, ``..env``).  We test the production helper
    directly so production-side changes can't drift away from these
    expectations silently."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Accepted — pass dirname through to gh api.
            ("scripts/foo.py", True),
            ("vllm_mlx/routes/anthropic.py", True),
            ("..hidden/foo.py", True),  # legitimate name starting with ..
            ("..env/x.py", True),
            ("foo/..hidden/bar.py", True),
            # Rejected — would either traverse, hit an invalid endpoint, or
            # be silently dropped because there's no dirname to feed.
            ("../escape/foo.py", False),
            ("../../etc/passwd", False),
            ("/etc/passwd", False),
            ("./foo.py", False),  # dirname='.', normpath='.'
            ("..", False),  # all-traversal
            ("foo.py", False),  # no dirname at all
        ],
    )
    def test_filter(self, path, expected):
        assert _is_safe_listing_path(os.path.dirname(path)) is expected
