# SPDX-License-Identifier: Apache-2.0
"""First-run consent prompt.

Fires at most once per machine, only when:

- The user has never been prompted (``get_consent_state`` returns None).
- ``RAPID_MLX_TELEMETRY`` is not set (env var already determines state,
  no need to ask).
- ``--no-telemetry`` is not set on this run.
- ``stdin`` is a tty (we are not in a pipe / CI / daemon-spawn).
- The current subcommand is interactive (``serve``, ``chat``, etc.) —
  not ``version``, ``help``, ``models``, ``ps``, ``info``, or any
  read-only / one-shot command where a prompt would be intrusive.

The disclosure copy is intentionally short: 6 lines, links to README,
defaults to NO. We never nag — declining writes ``consent=False`` so
this prompt does not fire again.
"""

from __future__ import annotations

import sys

from vllm_mlx import __version__ as _rapid_mlx_version  # noqa: N811
from vllm_mlx.telemetry.state import (
    ENV_VAR,
    consent_path,
    get_consent_state,
    record_consent,
)

# Subcommands where prompting would be intrusive (one-shot info commands,
# pipe-friendly outputs, or commands explicitly about telemetry config).
# Everything else (serve, chat, agents, bench, doctor, pull, rm, upgrade)
# is interactive enough that a one-time disclosure is OK.
_NON_INTERACTIVE_SUBCOMMANDS = frozenset(
    {
        "version",
        "help",
        "models",
        "ps",
        "info",
        "telemetry",
    }
)

_DISCLOSURE = """\
Rapid-MLX can send anonymous usage data so we can prioritise the right
models and catch regressions across the user base.

We collect (only if you say yes): subcommand names, model alias names,
bucketed token/latency counts, error categories, OS + chip + RAM.

We never collect: prompts, completions, file paths, IPs, env-var values,
or any user-generated text. Source: vllm_mlx/telemetry/.

Type 'y' to opt in, anything else to decline. You can change this anytime
with `rapid-mlx telemetry {{enable,disable,reset}}`. To force-disable in
scripts: set {env}=0.
"""


def maybe_prompt_for_consent(
    subcommand: str | None, *, cli_no_telemetry: bool = False
) -> None:
    """Show the first-run prompt if (and only if) all guards pass.

    Returns silently when any guard skips. Never raises — a broken stdin
    or unwritable home directory must not crash the user's serve / chat
    invocation just because we couldn't ask about telemetry.
    """
    try:
        if cli_no_telemetry:
            return
        # Env var already decides — no need to prompt.
        import os

        if os.environ.get(ENV_VAR) is not None:
            return
        if subcommand in _NON_INTERACTIVE_SUBCOMMANDS:
            return
        if subcommand is None:
            return
        if get_consent_state() is not None:
            return
        if not sys.stdin.isatty():
            return
        if not sys.stdout.isatty():
            return

        version = _rapid_mlx_version
        print()
        print(_DISCLOSURE.format(env=ENV_VAR))
        print("  [opt-in to anonymous telemetry?  y/N]  ", end="", flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Treat interruption as decline. Don't record — let the user
            # be re-prompted next interactive run, since they didn't
            # actually answer.
            print()
            return
        consent = answer in ("y", "yes")
        record_consent(consent, rapid_mlx_version=version)
        if consent:
            print(
                "Thanks — telemetry enabled. Disable anytime with "
                "`rapid-mlx telemetry disable`."
            )
        else:
            print(
                "Got it — telemetry stays off. Enable later with "
                f"`rapid-mlx telemetry enable` (or delete {consent_path()} "
                "to be re-prompted)."
            )
        print()
    except (OSError, EOFError, KeyboardInterrupt):
        # Telemetry consent is *never* a reason for the CLI to fail —
        # but only swallow the failure modes we actually expect: I/O on
        # an unwritable home, terminal weirdness, or user Ctrl-C during
        # input. Programming errors (AttributeError, TypeError, ...)
        # propagate so they get noticed in development.
        return
