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
    client_id_path,
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
Rapid-MLX is open source and built on what its users report. We do not
ship analytics SDKs, third-party trackers, or ads — and we never will.
With your help, we can do three concrete things better:

  • Fix the crashes you actually hit, with anonymous error fingerprints
    instead of waiting for someone to file an issue.
  • Tune performance on the Apple Silicon variants people actually run,
    instead of whatever happens to sit on our desks.
  • Surface the models the community runs most, so onboarding effort
    goes where it matters.

WHAT WE SEND (only after you say yes):
  • Your chip family + RAM tier — "Apple M3 Ultra, 256 GB", never serial
  • OS family + major.minor version ("darwin 25.3"), arch ("arm64"),
    Python major.minor ("3.12"), Rapid-MLX version ("0.6.79")
  • Which subcommand you ran ("serve" / "chat") and its duration
  • A UTC timestamp on each event (no timezone, second precision)
  • Crash fingerprints — file:line:exception_class, no message text
  • A random UUID at {client_id_path}, which you can rotate or wipe
  • A per-process random UUID (a session id) so we can group your
    session_start + session_end without correlating across runs

LATER (when per-request instrumentation lands, behind the same gate):
  • Which alias you load, decode tokens/sec and latency in coarse buckets

WHAT WE NEVER SEND, EVER:
  • Prompts. Generated text. File paths. API keys.
  • Anything from before this prompt or from a session you opted out of.

ABOUT YOUR IP:
  Any HTTPS request reveals your IP to the receiver at the network
  layer — we cannot change that. What we control is what we record.
  Our Worker never writes your IP to the stored event; it is used
  only for a transient per-minute rate-limit counter (the counter
  key is sha256(IP), the raw IP is discarded the same request).

You can see the exact bytes that would leave your machine right now:
  rapid-mlx telemetry preview

You can pause, resume, or reset your identity anytime:
  rapid-mlx telemetry {{status,disable,enable,reset}}

To force-disable in scripts or CI: set {env}=0.
"""


def maybe_prompt_for_consent(
    subcommand: str | None, *, cli_no_telemetry: bool = False
) -> bool:
    """Show the first-run prompt if (and only if) all guards pass.

    Returns ``True`` IFF the prompt was actually shown AND the user
    just answered for the first time (regardless of yes/no). Returns
    ``False`` for every skip path. The caller uses this to suppress
    lifecycle emit on the same invocation that just collected consent
    — round 3 codex review caught that emitting ``session_start`` for
    the argv that ran BEFORE the prompt contradicts the disclosure's
    "nothing from before this prompt" promise.

    Never raises — a broken stdin or unwritable home directory must
    not crash the user's serve / chat invocation just because we
    couldn't ask about telemetry.
    """
    # ``just_collected`` is owned at function scope (round 14 codex
    # review fix): the previous structure flipped this back to False
    # whenever a post-``record_consent`` ``print()`` raised ``OSError``
    # (e.g. SIGPIPE from a pipe closed by the parent shell), so the
    # CLI dropped ``_just_collected_consent`` and emitted same-run
    # lifecycle telemetry — violating the "nothing from before this
    # prompt" promise. Owning the flag at the outermost scope means
    # the final ``return`` honours whatever was already persisted, even
    # if the thank-you / opt-out chatter blew up.
    just_collected = False
    try:
        if cli_no_telemetry:
            return False
        # Env var already decides — no need to prompt.
        import os

        if os.environ.get(ENV_VAR) is not None:
            return False
        if subcommand in _NON_INTERACTIVE_SUBCOMMANDS:
            return False
        if subcommand is None:
            return False
        if get_consent_state() is not None:
            return False
        if not sys.stdin.isatty():
            return False
        if not sys.stdout.isatty():
            return False

        version = _rapid_mlx_version
        print()
        print(
            _DISCLOSURE.format(
                env=ENV_VAR,
                client_id_path=client_id_path(),
            )
        )
        print(
            "Contribute anonymous telemetry to make rapid-mlx better "
            "for everyone? [y/N]  ",
            end="",
            flush=True,
        )
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Treat interruption as decline. Don't record — let the user
            # be re-prompted next interactive run, since they didn't
            # actually answer.
            print()
            return False
        consent = answer in ("y", "yes")
        record_consent(consent, rapid_mlx_version=version)
        # From here on, consent IS persisted to disk. The flag must
        # survive any subsequent print failure so the CLI knows not to
        # emit lifecycle telemetry for the argv that ran before the
        # prompt.
        just_collected = True
        if consent:
            print()
            print(
                "Thank you for contributing. "
                "rapid-mlx will get measurably better because you said yes."
            )
            print(
                "Audit anytime: `rapid-mlx telemetry status` / `... preview`. "
                "Stop anytime: `rapid-mlx telemetry disable`."
            )
        else:
            print()
            print(
                "Got it — telemetry stays off and we will not ask again. "
                "You can always opt in later with "
                "`rapid-mlx telemetry enable`,"
            )
            print(f"or delete {consent_path()} to be re-prompted.")
        print()
        return just_collected
    except (OSError, EOFError, KeyboardInterrupt):
        # Telemetry consent is *never* a reason for the CLI to fail —
        # but only swallow the failure modes we actually expect: I/O on
        # an unwritable home, terminal weirdness, or user Ctrl-C during
        # input. Programming errors (AttributeError, TypeError, ...)
        # propagate so they get noticed in development.
        #
        # Return ``just_collected`` (not ``False``) so a post-record
        # OSError still reports "yes, we just collected consent" — the
        # round 14 codex blocker.
        return just_collected
