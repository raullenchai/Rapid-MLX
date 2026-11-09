# SPDX-License-Identifier: Apache-2.0
"""Phone UI accessibility contract and incremental reasoning boundaries."""

import shutil
import subprocess
from html.parser import HTMLParser
from importlib.resources import files

import pytest

PAGE = files("rapid_mlx").joinpath("phone.html").read_text(encoding="utf-8")


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.by_id = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.by_id
            self.by_id[attrs["id"]] = (tag, attrs)


def test_phone_accessibility_contract():
    elements = Elements()
    elements.feed(PAGE)
    for name, tag, label in [
        ("composer", "textarea", "Message"),
        ("send", "button", "Send message"),
        ("stop", "button", "Stop generating"),
        ("new-chat", "button", "New chat"),
        ("model-button", "button", "Choose model"),
        ("apps-button", "button", "Use an app"),
    ]:
        actual_tag, attrs = elements.by_id[name]
        assert actual_tag == tag
        assert attrs["aria-label"] == label
    for name in ["model-sheet", "apps-sheet"]:
        tag, attrs = elements.by_id[name]
        assert tag == "dialog"
        assert attrs["aria-labelledby"] in elements.by_id


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 100])
def test_thinking_tags_survive_stream_boundaries(chunk_size):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the page's JavaScript parser")
    parser = PAGE.split("function thinkingParser(", 1)[1].split(
        "async function send(", 1
    )[0]
    script = "function thinkingParser(" + parser
    script += f"""
const assert = require('node:assert/strict');
const answer = {{content: '', thought: '', seconds: 0}};
const parser = thinkingParser(answer);
parser.reason('Separate reasoning. ');
const text = 'Before<think>Hidden <b>thought</b></think>After';
for (let i = 0; i < text.length; i += {chunk_size})
    parser.content(text.slice(i, i + {chunk_size}));
parser.content('', true);
assert.equal(answer.content, 'BeforeAfter');
assert.equal(answer.thought, 'Separate reasoning. Hidden <b>thought</b>');
assert.ok(answer.seconds >= 1);
const incomplete = {{content: '', thought: ''}};
const other = thinkingParser(incomplete);
other.content('Plain <thi');
other.content('', true);
assert.equal(incomplete.content, 'Plain <thi');
const stopped = {{content: '', thought: ''}};
const last = thinkingParser(stopped);
last.content('<think>Unfinished thought');
last.content('', true);
assert.equal(stopped.content, '');
assert.equal(stopped.thought, 'Unfinished thought');
"""
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_secret_is_read_from_fragment_not_query():
    # A query key would reach server access logs. Keep bootstrap fragment-only.
    assert "new URLSearchParams(location.hash.slice(1))" in PAGE
    assert "location.search" not in PAGE
    assert "innerHTML" not in PAGE
    assert "insertAdjacentHTML" not in PAGE
    assert 'credentials: "omit"' in PAGE or 'credentials:"omit"' in PAGE
