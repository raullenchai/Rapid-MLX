# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx serve --help`` must lead with the common serve journey (#2354).

The serve parser's option dump is 500+ lines of advanced tuning; a new user
must be able to see how to actually start a server without reading the whole
help. This gating test locks in the concise quick-start (command + the few
options that matter for a first server) and the first-time tips epilog.

The assertions are scoped to the introduction block (the RawDescription body
between the usage line and the first section header) rather than the whole
help dump: the option flags (``--port``/``--host``/``--api-key``) also appear
as real rows in the 500-line options section later, so a grep across the whole
output would stay green even if the quick-start explanation lines were
deleted (codex r1).
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


def _intro(serve_help: str) -> str:
    """The description block between the usage line and the first section.

    argparse orders ``usage`` -> ``description`` -> sections -> ``epilog``.
    The description body is what sits between the usage block and the first
    ``positional arguments:`` / ``options:`` header. Scoping the assertions
    here (instead of the whole 500-line dump) is what makes them bite: every
    quick-start string must live in the intro, not merely coexist with the
    real option rows later (codex r1).
    """
    start = serve_help.index("Start a local OpenAI")
    end = serve_help.index("options:")
    return serve_help[start:end]


def _epilog(serve_help: str) -> str:
    """The tips epilog block, which appears after the options section."""
    return serve_help[serve_help.rindex("First-time tips:") :]


def test_serve_help_leads_with_a_command_line(serve_help):
    """#2354: the help must open with a copyable serve command — in the
    description block ahead of the options section, not merely present
    somewhere in the 500-line dump.

    codex r5/r6/r7: the leading command must be genuinely RUNNABLE when a
    new user pastes it — a concrete real model name, no shell metacharacters
    (``<model>`` would read as input redirection, ``[--port N]`` as a glob).
    The ``<model>`` placeholder is still explained on the legend line below,
    so the template intent is preserved without polluting the copyable line.

    codex r8 BLOCKING: 'concrete' must mean more than 'no metacharacters' —
    the named model has to actually exist in the alias catalog, or an
    invalid placeholder like ``not-a-real-model`` would pass. Resolve the
    extracted model through ``list_profiles()``.
    """
    intro = _intro(serve_help)
    # The copyable command is the indented '  rapid-mlx serve ...' line.
    cmd = next(
        ln.strip() for ln in intro.splitlines() if ln.startswith("  rapid-mlx serve")
    )
    for metachar in ("<", ">", "["):
        assert metachar not in cmd, (
            f"copyable command must not contain {metachar!r}: {cmd!r}"
        )
    assert cmd.startswith("rapid-mlx serve ")
    model = cmd[len("rapid-mlx serve ") :].strip()
    # Concrete real alias (not a placeholder token) *and* present in the
    # actual catalog, so a removed or invented model fails (codex r8).
    assert model, "command line must name a model"
    assert not any(ch in model for ch in "<[]>"), (
        f"model arg must be concrete: {model!r}"
    )
    from vllm_mlx.model_aliases import list_profiles

    assert model in list_profiles(), (
        f"help example model {model!r} is not a served alias"
    )
    # The placeholder is still documented in the legend.
    assert "<model>" in intro
    # The command must precede the options section, not follow it.
    assert serve_help.index(cmd) < serve_help.index("options:")


def test_serve_help_shows_the_essential_options(serve_help):
    """#2354 (codex r1): the quick-start must explain the few flags a first
    user needs in plain terms, and those explanations must live in the intro
    block — the flags also appear as real option rows later, so asserting
    bare ``--port`` across the whole help would pass even with the
    quick-start lines deleted."""
    intro = _intro(serve_help)
    for flag, phrase in (
        ("--port", "bind port"),
        ("--host", "bind host"),
        ("--api-key", "bearer token"),
    ):
        assert flag in intro
        assert phrase in intro, f"intro must explain {flag} in plain terms: {phrase!r}"
    # codex r7 nit: --port is optional (default 8000); the intro must not
    # imply a first server needs to pass it (or --host) explicitly.
    assert "default" in intro


def test_serve_help_points_to_discovery_commands(serve_help):
    """#2354: the epilog must point a new user at the discoverability entry
    points (listing models + RAM-fit recommendation), in the tips block after
    the options."""
    epilog = _epilog(serve_help)
    assert "rapid-mlx models" in epilog
    assert "rapid-mlx recipe" in epilog


def test_serve_help_explains_the_ready_url(serve_help):
    """#2354: the description must tell the user what success looks like —
    the 'Ready:' URL / OpenAI-compatible base they connect to. The audio
    endpoints are listed as a route FAMILY (``/v1/audio/*``), not a bare
    ``/v1/audio`` which 404s (the actual routes are transcriptions/
    translations/speech/voices — codex r2 nit)."""
    intro = _intro(serve_help)
    assert "Ready:" in intro
    assert "/v1/audio/*" in intro
    assert " /v1/audio," not in intro  # a bare /v1/audio route does not exist


def test_serve_help_documented_defaults_match_the_parser(serve_help):
    """#2354 (codex r2 nit): the quick-start hard-codes ``(default 8000)`` and
    ``(default 127.0.0.1...`` for the two flags a first user touches. Those
    strings must stay in lock-step with the parser's actual defaults, or the
    help silently lies when a default changes. Assert the documented numbers
    equal the serve subparser's resolved defaults, and that the intro
    presents them."""
    parser = cli.build_parser()
    serve = next(
        a.choices["serve"]
        for a in parser._actions
        if a.dest == "command" and "serve" in a.choices
    )
    port = str(serve.get_default("port"))
    host = str(serve.get_default("host"))
    intro = _intro(serve_help)
    assert f"(default {port})" in intro
    assert f"(default {host}," in intro


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
