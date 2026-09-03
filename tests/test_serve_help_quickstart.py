# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx serve --help`` must lead with the common serve journey (#2354).

The serve parser's option dump is 500+ lines of advanced tuning; a new user
must be able to see how to actually start a server without reading the whole
help. This gating test locks in the concise quick-start (command + the few
options that matter for a first server) and the first-time tips epilog.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from vllm_mlx import cli


@pytest.fixture
def serve_help(capsys):
    """Capture the rendered ``rapid-mlx serve --help`` text."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_serve_help_leads_with_a_command_line(serve_help):
    """#2354: the help must open with a copyable serve command, not bury it
    under 500 lines of options."""
    assert "rapid-mlx serve <model>" in serve_help


def test_serve_help_shows_the_essential_options(serve_help):
    """#2354: the quick-start must name the few flags a first-time user needs
    (port, host, auth) in plain terms."""
    assert "--port" in serve_help
    assert "--host" in serve_help
    assert "--api-key" in serve_help


def test_serve_help_points_to_discovery_commands(serve_help):
    """#2354: the epilog must point a new user at the discoverability entry
    points (listing models + RAM-fit recommendation)."""
    assert "rapid-mlx models" in serve_help
    assert "rapid-mlx recipe" in serve_help


def test_serve_help_explains_the_ready_url(serve_help):
    """#2354: the description must tell the user what success looks like —
    the 'Ready:' URL / OpenAI-compatible base they connect to."""
    assert "Ready:" in serve_help


def test_serve_parser_still_parses_a_normal_invocation():
    """#2354 regression guard: adding the description/epilog must not change
    how an ordinary ``serve <model> --port N`` invocation parses."""
    parser = cli.build_parser()
    args = parser.parse_args(
        ["serve", "qwen3.5-4b-4bit", "--port", "8123", "--log-level", "INFO"]
    )
    assert args.command == "serve"
    assert args.model == "qwen3.5-4b-4bit"
    assert args.port == 8123
