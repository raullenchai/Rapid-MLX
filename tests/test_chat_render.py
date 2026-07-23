# SPDX-License-Identifier: Apache-2.0
"""Terminal rendering tests for ``rapid-mlx chat``."""

from __future__ import annotations

import io
import os
import re
from unittest.mock import patch

from rich.cells import cell_len

from vllm_mlx.chat_render import (
    StreamingMarkdownRenderer,
    render_markdown,
    terminal_safe_text,
)


class _Tty(io.StringIO):
    def isatty(self):
        return True


def _visible(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return text.replace("\r", "")


def test_render_markdown_formats_common_agent_output():
    output = _Tty()
    markdown = """\
# Result

Use **care** with `rm`.

- first item is intentionally long enough to wrap instead of disappearing at the terminal edge WRAP-END
- second

```python
print("hello")
```

| Name | Value |
| --- | --- |
| answer | 42 |
"""

    with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
        os.environ.pop("NO_COLOR", None)
        os.environ.pop("CI", None)
        render_markdown(markdown, stream=output)

    rendered = output.getvalue()
    visible = _visible(rendered)
    assert "\x1b[" in rendered
    assert visible.splitlines()[0].startswith("Result")
    assert "# Result" not in visible
    assert "**care**" not in visible
    assert "`rm`" not in visible
    assert "Result" in visible
    assert "care" in visible
    assert "rm" in visible
    assert "first" in visible and "second" in visible
    assert "WRAP-END" in visible
    assert 'print("hello")' in visible
    assert "answer" in visible and "42" in visible


def test_streaming_renderer_keeps_plain_output_for_pipes():
    output = io.StringIO()
    renderer = StreamingMarkdownRenderer(output)

    renderer.write("A **bold")
    renderer.write("** answer.")
    renderer.finish()

    assert output.getvalue() == "A **bold** answer."


def test_streaming_renderer_renders_complete_markdown_document():
    output = _Tty()
    with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
        os.environ.pop("NO_COLOR", None)
        os.environ.pop("CI", None)
        renderer = StreamingMarkdownRenderer(output)
        renderer.write("# Res")
        assert "Res" in _visible(output.getvalue())
        renderer.write("ult\n\nUse **care**")
        assert "Use **care**" in _visible(output.getvalue())
        renderer.write(" and `code`.")
        renderer.finish()

    rendered = output.getvalue().rsplit("\x1b[2K", 1)[-1]
    visible = _visible(rendered)
    assert "Result" in visible
    assert "care" in visible
    assert "code" in visible
    assert "# Result" not in visible
    assert "**care**" not in visible


def test_streaming_renderer_flushes_partial_block_after_error():
    output = _Tty()
    with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
        os.environ.pop("NO_COLOR", None)
        os.environ.pop("CI", None)
        try:
            with StreamingMarkdownRenderer(output) as renderer:
                renderer.write("partial **answer**")
                raise RuntimeError("stream failed")
        except RuntimeError as exc:
            assert str(exc) == "stream failed"

    final_render = output.getvalue().rsplit("\x1b[2K", 1)[-1]
    visible = _visible(final_render)
    assert "partial" in visible
    assert "answer" in visible
    assert "**answer**" not in visible


def test_streaming_preview_respects_terminal_cell_width():
    output = _Tty()
    with patch.dict(
        os.environ,
        {"TERM": "xterm-256color", "COLUMNS": "8"},
        clear=False,
    ):
        os.environ.pop("NO_COLOR", None)
        os.environ.pop("CI", None)
        renderer = StreamingMarkdownRenderer(output)
        renderer.write("prefix 你好世界")

    preview = output.getvalue().rsplit("\x1b[2m", 1)[-1].split("\x1b[0m", 1)[0]
    assert cell_len(preview) <= 7


def test_streaming_preview_shows_live_token_and_elapsed():
    output = _Tty()
    with patch.dict(
        os.environ,
        {"TERM": "xterm-256color", "COLUMNS": "100"},
        clear=False,
    ):
        os.environ.pop("NO_COLOR", None)
        os.environ.pop("CI", None)
        renderer = StreamingMarkdownRenderer(output)
        renderer.write("Hello")
        renderer.write(" there")

    # The live (pre-finish) preview line leads with a token/elapsed badge so a
    # long answer shows progress instead of a silent ticker.
    preview = output.getvalue().rsplit("\x1b[2m", 1)[-1].split("\x1b[0m", 1)[0]
    assert preview.startswith("2 tok · ")
    assert "s  " in preview
    assert "Hello there" in preview


def test_terminal_safe_text_removes_control_sequences():
    text = "tool\x1b]52;c;payload\x07\r\nname"

    assert terminal_safe_text(text) == "tool]52;c;payloadname"


def test_render_markdown_preserves_unfenced_html():
    output = _Tty()
    with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
        os.environ.pop("NO_COLOR", None)
        os.environ.pop("CI", None)
        render_markdown("<div>hello</div>", stream=output)

    visible = _visible(output.getvalue())
    assert "<div>hello</div>" in visible


def test_streaming_renderer_preserves_cross_block_markdown_context():
    output = _Tty()
    with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
        os.environ.pop("NO_COLOR", None)
        os.environ.pop("CI", None)
        renderer = StreamingMarkdownRenderer(output)
        renderer.write("[Rapid][project]\n\n")
        assert "[Rapid][project]" in _visible(output.getvalue())
        renderer.write("[project]: https://rapidmlx.com\n")
        renderer.finish()

    visible = _visible(output.getvalue().rsplit("\x1b[2K", 1)[-1])
    assert "Rapid" in visible
    assert "[Rapid][project]" not in visible


def test_no_color_disables_markdown_rendering():
    output = _Tty()
    with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True):
        render_markdown("**plain**", stream=output)

    assert output.getvalue() == "**plain**"


def test_dumb_terminal_keeps_incremental_plain_output():
    output = _Tty()
    with patch.dict(os.environ, {"TERM": "dumb"}, clear=True):
        renderer = StreamingMarkdownRenderer(output)
        renderer.write("first ")
        assert output.getvalue() == "first "
        renderer.write("**second**")
        renderer.finish()

    assert output.getvalue() == "first **second**"


def test_ci_pseudo_terminal_keeps_plain_output():
    output = _Tty()
    with patch.dict(
        os.environ,
        {"CI": "true", "TERM": "xterm-256color"},
        clear=False,
    ):
        os.environ.pop("NO_COLOR", None)
        renderer = StreamingMarkdownRenderer(output)
        renderer.write("A **captured** answer.")
        renderer.finish()

    assert output.getvalue() == "A **captured** answer."
