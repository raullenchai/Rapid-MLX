# SPDX-License-Identifier: Apache-2.0
"""Tests for the codex review step's pure helpers.

The ``codex exec`` call itself is integration-level (requires the
codex CLI to be installed and logged in, hits the network) and lives
in ``scripts/pr_validate/steps/codex_review.py``. We test only the
helpers that have isolated logic plus the JSONL parser:

- ``_truncate_diff_at_file_boundary`` — file-boundary aware diff cap
- ``_is_safe_listing_path`` — path-traversal filter for the dir-listing
  enhancement that feeds ``gh api``
- ``_parse_codex_jsonl`` — codex exec stdout parser (agent_message
  concatenation + ``turn.completed`` usage extraction)

The diff-cap and path-filter tests were carried over from the prior
DeepSeek step verbatim: regex missing git's quoted-filename form,
``startswith("..")`` over-filter, and the ``.``-current-dir leak are
all real bugs surfaced by PR review on the original implementation.
"""

from __future__ import annotations

import json
import os

import pytest

from scripts.pr_validate.steps.codex_review import (
    CODEX_MODEL,
    CodexReviewStep,
    _is_safe_listing_path,
    _is_transient_codex_failure,
    _parse_codex_jsonl,
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


class TestParseCodexJsonl:
    """``_parse_codex_jsonl`` reads ``codex exec --json`` stdout (one
    JSON object per line) and returns ``(reply_text, usage_dict)``.

    The contract is: only ``item.completed`` events whose ``item.type``
    is ``agent_message`` contribute to the reply (concatenated in
    stream order); ``turn.completed`` carries the token usage; every
    other event type is ignored without crashing. Malformed lines are
    silently dropped so a half-streamed reply is still reviewable.
    """

    @staticmethod
    def _stream(*events: dict) -> str:
        return "\n".join(json.dumps(e) for e in events)

    def test_extracts_agent_message_and_usage(self):
        stdout = self._stream(
            {"type": "thread.started", "thread_id": "abc"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "i0",
                    "type": "agent_message",
                    "text": "1. [BLOCKING] x.py:1 — bug.",
                },
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )
        text, usage = _parse_codex_jsonl(stdout)
        assert text == "1. [BLOCKING] x.py:1 — bug."
        assert usage == {"input_tokens": 100, "output_tokens": 20}

    def test_concatenates_multiple_agent_messages_in_order(self):
        """gpt-5.5 streams in chunks; the parser must keep order."""
        stdout = self._stream(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "1. First."},
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "2. Second."},
            },
            {"type": "turn.completed", "usage": {}},
        )
        text, _ = _parse_codex_jsonl(stdout)
        # The parser joins chunks with a blank line so numbered list
        # entries don't collide visually in the artifact.
        assert text == "1. First.\n\n2. Second."

    def test_ignores_non_agent_item_types(self):
        """``item.completed`` also fires for reasoning, tool_use, etc.
        Only ``agent_message`` should contribute."""
        stdout = self._stream(
            {
                "type": "item.completed",
                "item": {"type": "reasoning", "text": "thinking…"},
            },
            {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
            {
                "type": "item.completed",
                "item": {"type": "tool_use", "name": "read_file"},
            },
        )
        text, usage = _parse_codex_jsonl(stdout)
        assert text == "ok"
        # No turn.completed → usage is empty dict (defensive default).
        assert usage == {}

    def test_skips_malformed_lines(self):
        """Codex can emit a partial line on SIGINT / network blip. The
        parser must not crash; everything else is still extracted."""
        stdout = "\n".join(
            [
                "not json at all",
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "kept"},
                    }
                ),
                "{broken json",
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5}}),
            ]
        )
        text, usage = _parse_codex_jsonl(stdout)
        assert text == "kept"
        assert usage == {"input_tokens": 5}

    def test_empty_stdout_returns_empty(self):
        """Codex exited 0 but emitted nothing (rare; policy refusal
        sometimes lands here). Parser returns falsy values so the
        caller can surface a 'no agent message' skip."""
        text, usage = _parse_codex_jsonl("")
        assert text == ""
        assert usage == {}

    def test_empty_text_chunks_dropped(self):
        """An ``agent_message`` with empty/missing ``text`` should not
        contribute a stray separator to the concatenated reply."""
        stdout = self._stream(
            {"type": "item.completed", "item": {"type": "agent_message", "text": ""}},
            {"type": "item.completed", "item": {"type": "agent_message"}},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "real"},
            },
        )
        text, _ = _parse_codex_jsonl(stdout)
        assert text == "real"


class TestModelPinning:
    """The README + step description promise ``gpt-5.5``; the
    invocation must pass ``--model`` explicitly so a change to the
    caller's ``~/.codex/config.toml`` default can't silently swap the
    reviewer underneath the gate (codex round-1 BLOCKER on PR #505).
    """

    def test_codex_model_constant_matches_documented(self):
        assert CODEX_MODEL == "gpt-5.5"

    def test_codex_command_includes_explicit_model_flag(self, monkeypatch, tmp_path):
        """Drive the step with a fake ``codex`` binary that records the
        argv it was called with, then assert ``--model gpt-5.5`` is
        present. This pins the contract at the subprocess boundary."""
        captured: dict = {}

        class _FakeProc:
            returncode = 0
            stderr = ""
            stdout = json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "No blocking issues found.",
                    },
                }
            )

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _FakeProc()

        # Stub the subprocess + binary resolution. shutil.which returns
        # a non-None path so the step proceeds to the codex_exec call.
        monkeypatch.setattr(
            "scripts.pr_validate.steps.codex_review.shutil.which",
            lambda _: "/usr/bin/codex-stub",
        )
        monkeypatch.setattr(
            "scripts.pr_validate.steps.codex_review.subprocess.run", fake_run
        )

        # Minimal context — a tmp diff is enough; we just want the
        # command to be assembled and ``subprocess.run`` invoked.
        from scripts.pr_validate.context import Context

        # Context's __post_init__ requires the cwd to be a repo root.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("")

        ctx = Context(pr_number=505)
        ctx.work_dir = tmp_path
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n+x\n")
        ctx.diff_path = diff_path

        CodexReviewStep().run(ctx)

        cmd = captured["cmd"]
        # Adjacent ``--model`` + value pair must appear together.
        assert "--model" in cmd, f"missing --model in {cmd}"
        idx = cmd.index("--model")
        assert cmd[idx + 1] == CODEX_MODEL, (
            f"expected --model {CODEX_MODEL}, got {cmd[idx + 1]}"
        )


class TestBackwardsCompatOptOut:
    """The deepseek→codex swap renamed ``PR_VALIDATE_NO_DEEPSEEK`` to
    ``PR_VALIDATE_NO_CODEX``. Honor the old name as a deprecation alias
    for a migration window so existing CI/local workflows that disabled
    the paid LLM review don't silently re-enable codex (codex round-1
    BLOCKER on PR #505)."""

    def test_old_env_var_still_disables(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("")
        monkeypatch.setenv("PR_VALIDATE_NO_DEEPSEEK", "1")
        monkeypatch.delenv("PR_VALIDATE_NO_CODEX", raising=False)

        from scripts.pr_validate.context import Context

        ctx = Context(pr_number=1)
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text("diff --git a/x b/x\n")
        ctx.diff_path = diff_path

        assert CodexReviewStep().should_run(ctx) is False

    def test_new_env_var_disables(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("")
        monkeypatch.setenv("PR_VALIDATE_NO_CODEX", "1")
        monkeypatch.delenv("PR_VALIDATE_NO_DEEPSEEK", raising=False)

        from scripts.pr_validate.context import Context

        ctx = Context(pr_number=1)
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text("diff --git a/x b/x\n")
        ctx.diff_path = diff_path

        assert CodexReviewStep().should_run(ctx) is False

    def test_neither_env_var_runs(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("")
        monkeypatch.delenv("PR_VALIDATE_NO_CODEX", raising=False)
        monkeypatch.delenv("PR_VALIDATE_NO_DEEPSEEK", raising=False)

        from scripts.pr_validate.context import Context

        ctx = Context(pr_number=1)
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text("diff --git a/x b/x\n")
        ctx.diff_path = diff_path

        assert CodexReviewStep().should_run(ctx) is True

    def test_old_env_var_emits_deprecation_warning(self, monkeypatch, tmp_path, capsys):
        """The deprecation nudge must actually go to stderr so callers
        notice — otherwise the alias becomes a hidden permanent API."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("")
        monkeypatch.setenv("PR_VALIDATE_NO_DEEPSEEK", "1")
        monkeypatch.delenv("PR_VALIDATE_NO_CODEX", raising=False)

        from scripts.pr_validate.context import Context

        ctx = Context(pr_number=1)
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text("diff --git a/x b/x\n")
        ctx.diff_path = diff_path

        CodexReviewStep().should_run(ctx)
        captured = capsys.readouterr()
        assert "deprecated" in captured.err.lower()
        assert "PR_VALIDATE_NO_CODEX" in captured.err


class TestPromptInjectionGuards:
    """The codex prompt and the PR diff share one ``codex exec`` prompt
    slot — they are not naturally role-separated. A malicious diff could
    inject ``ignore previous instructions`` or invoke tools. We mitigate
    by (a) fencing the diff with explicit ``UNTRUSTED USER INPUT``
    boundary markers and (b) appending a final-instruction block AFTER
    the diff that re-asserts the no-tool-use rule (codex round-2 BLOCKER
    on PR #505).

    We pin the prompt assembly by capturing what gets sent to codex via
    monkeypatched ``subprocess.run`` and asserting the marker strings
    are present in the right relative order.
    """

    @staticmethod
    def _capture_combined_prompt(monkeypatch, tmp_path, diff_body: str) -> str:
        """Drive the step and return whatever combined prompt was passed
        to ``subprocess.run``'s ``input=`` kwarg."""
        captured: dict = {}

        class _FakeProc:
            returncode = 0
            stderr = ""
            stdout = json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "No blocking issues found.",
                    },
                }
            )

        def fake_run(cmd, **kwargs):
            captured["input"] = kwargs.get("input", "")
            return _FakeProc()

        monkeypatch.setattr(
            "scripts.pr_validate.steps.codex_review.shutil.which",
            lambda _: "/usr/bin/codex-stub",
        )
        monkeypatch.setattr(
            "scripts.pr_validate.steps.codex_review.subprocess.run", fake_run
        )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("")

        from scripts.pr_validate.context import Context

        ctx = Context(pr_number=505)
        ctx.work_dir = tmp_path
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(diff_body)
        ctx.diff_path = diff_path

        CodexReviewStep().run(ctx)
        return captured["input"]

    def test_diff_is_fenced_with_untrusted_input_markers(self, monkeypatch, tmp_path):
        """The diff block must sit between explicit BEGIN/END markers so
        the model can identify the boundary even when the diff content
        contains markdown-looking sequences."""
        diff = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n+content\n"
        prompt = self._capture_combined_prompt(monkeypatch, tmp_path, diff)

        begin_idx = prompt.find("BEGIN UNTRUSTED USER INPUT")
        end_idx = prompt.find("END UNTRUSTED USER INPUT")
        diff_idx = prompt.find("+content")
        assert begin_idx >= 0, "missing BEGIN marker"
        assert end_idx > begin_idx, "missing/misordered END marker"
        assert begin_idx < diff_idx < end_idx, (
            "diff content must sit between BEGIN and END markers"
        )

    def test_codex_subprocess_runs_in_isolated_cwd(self, monkeypatch, tmp_path):
        """Defence-in-depth: ``--sandbox read-only`` still permits reads,
        so if a prompt-injection bypasses the in-prompt guards and the
        model runs ``ls`` / ``cat *`` / ``find``, we want it to land in
        an empty directory rather than the repo root (codex round-3
        BLOCKER on PR #505). The cwd kwarg must NOT be the repo root
        or any user dir — it must be an isolated temp dir."""
        captured: dict = {}

        class _FakeProc:
            returncode = 0
            stderr = ""
            stdout = json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "No blocking issues found.",
                    },
                }
            )

        def fake_run(cmd, **kwargs):
            captured["cwd"] = kwargs.get("cwd")
            return _FakeProc()

        monkeypatch.setattr(
            "scripts.pr_validate.steps.codex_review.shutil.which",
            lambda _: "/usr/bin/codex-stub",
        )
        monkeypatch.setattr(
            "scripts.pr_validate.steps.codex_review.subprocess.run", fake_run
        )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("")

        from scripts.pr_validate.context import Context

        ctx = Context(pr_number=505)
        ctx.work_dir = tmp_path
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text("diff --git a/x b/x\n")
        ctx.diff_path = diff_path

        CodexReviewStep().run(ctx)

        cwd = captured["cwd"]
        assert cwd is not None, "codex subprocess MUST be given a cwd= kwarg"
        # The cwd must not be the repo root / pyproject parent — it has
        # to be an isolated tempdir so the model can't `ls` into anything
        # useful. We check by listing the dir contents *while it still
        # exists* — but TemporaryDirectory has already cleaned up by the
        # time run() returns. Instead, assert the path looks like a temp
        # dir (system tempdir prefix) and contains the marker prefix
        # we picked.
        assert "codex-review-cwd-" in cwd, (
            f"cwd should be a TemporaryDirectory with the codex-review prefix, "
            f"got {cwd!r}"
        )

    def test_final_instructions_appear_after_the_diff(self, monkeypatch, tmp_path):
        """Prompt-injection mitigation hinges on the no-tool-use rule
        getting the *last word*. An attacker writing 'ignore previous
        instructions' inside the diff fails because the model also sees
        the same rule re-asserted AFTER the diff block."""
        diff = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n+x\n"
        prompt = self._capture_combined_prompt(monkeypatch, tmp_path, diff)

        end_marker_idx = prompt.find("END UNTRUSTED USER INPUT")
        final_block_idx = prompt.find("FINAL INSTRUCTIONS")
        assert final_block_idx > end_marker_idx, (
            "FINAL INSTRUCTIONS block must come AFTER the diff so it "
            "gets the last word over any in-diff injection attempt"
        )
        # The final block must re-assert the no-tool-use rule (the
        # specific defence against 'invoke a shell tool to read repo'
        # injection attempts).
        final_section = prompt[final_block_idx:].lower()
        assert "do not call shell tools" in final_section
        assert "do not read files" in final_section
        assert "untrusted" in final_section


class TestNonZeroExitDiscrimination:
    """Codex round-4 BLOCKER on PR #505: mapping every non-zero exit to
    ``skip`` lets a malicious diff bypass the review gate by inducing a
    crash. The discriminator must distinguish "backend transiently
    broken" (skip) from "diff plausibly caused this" (fail).
    """

    @pytest.mark.parametrize(
        "stderr,expected",
        [
            # Transient — should skip
            ("error: not logged in to ChatGPT", True),
            ("HTTP 401 Unauthorized", True),
            ("rate limit exceeded (429)", True),
            ("upstream returned 502 Bad Gateway", True),
            ("503 Service Unavailable", True),
            ("504 Gateway Timeout", True),
            ("connection refused", True),
            ("connection reset by peer", True),
            ("Could not resolve host: api.openai.com", True),
            ("network is unreachable", True),
            ("SSL handshake failed", True),
            ("tls handshake timeout", True),
            ("request timed out after 600s", True),
            # Non-transient — should fail
            ("panic: runtime error: index out of range", False),
            ("model returned malformed response", False),
            ("Error: prompt exceeds context window", False),
            # Empty stderr — no evidence of transience, must fail
            ("", False),
            ("   \n", False),
        ],
    )
    def test_discriminator(self, stderr, expected):
        assert _is_transient_codex_failure(stderr) is expected

    def test_codex_skip_on_transient_backend(self, monkeypatch, tmp_path):
        """Network-down stderr → skip (don't block PRs on flaky API)."""

        class _FakeProc:
            returncode = 1
            stderr = "error: could not resolve host: api.openai.com"
            stdout = ""

        self._drive_and_assert(monkeypatch, tmp_path, _FakeProc(), expected="skip")

    def test_codex_fail_on_content_induced_crash(self, monkeypatch, tmp_path):
        """Non-transient stderr → fail (a malicious diff might be the cause)."""

        class _FakeProc:
            returncode = 1
            stderr = "panic: runtime error in model inference"
            stdout = ""

        self._drive_and_assert(monkeypatch, tmp_path, _FakeProc(), expected="fail")

    def test_codex_fail_on_empty_stderr_nonzero_exit(self, monkeypatch, tmp_path):
        """Silent crash → fail. Without stderr evidence of transience we
        must NOT default to skip — that's the exact bypass the round-4
        BLOCKER identified."""

        class _FakeProc:
            returncode = 137  # SIGKILL — could be OOM induced by diff
            stderr = ""
            stdout = ""

        self._drive_and_assert(monkeypatch, tmp_path, _FakeProc(), expected="fail")

    @staticmethod
    def _drive_and_assert(monkeypatch, tmp_path, fake_proc, *, expected: str):
        monkeypatch.setattr(
            "scripts.pr_validate.steps.codex_review.shutil.which",
            lambda _: "/usr/bin/codex-stub",
        )
        monkeypatch.setattr(
            "scripts.pr_validate.steps.codex_review.subprocess.run",
            lambda *a, **kw: fake_proc,
        )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("")

        from scripts.pr_validate.context import Context

        ctx = Context(pr_number=1)
        ctx.work_dir = tmp_path
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text("diff --git a/x b/x\n")
        ctx.diff_path = diff_path

        result = CodexReviewStep().run(ctx)
        assert result.status == expected, (
            f"expected {expected}, got {result.status}: {result.summary}"
        )
