# SPDX-License-Identifier: Apache-2.0
"""Machine-readable half of the activation/engagement spec.

The human spec lives in ``docs/telemetry-activation.md``; this module is
the code contract it references. The growth dashboard and the repository
tests both key off ``ACTIVATION_SPEC_VERSION`` — bump it (and update the
doc) whenever the rules below change.

Nothing here touches the network or reads consent; it is pure definition
so both ``emit`` (producer) and the tests (verifier) import the same
constants instead of re-declaring them.
"""

from __future__ import annotations

# Bump in lockstep with docs/telemetry-activation.md whenever what counts
# as engaged/activated changes (new kind, changed success predicate, ...).
ACTIVATION_SPEC_VERSION = 2

# The funnel milestones. Each fires at most once per install (client_id).
ACTIVATION_FIRST_INFERENCE = "first_inference"
ACTIVATION_MODEL_PULL = "model_pull"
ACTIVATION_AGENT_SETUP = "agent_setup"
ACTIVATION_FIRST_CHAT_REPLY = "first_chat_reply"
ACTIVATION_FIRST_VISION_REPLY = "first_vision_reply"
ACTIVATION_FIRST_DICTATION = "first_dictation"
ACTIVATION_FIRST_IMAGE = "first_image"

# Where the milestone happened. ``cli`` = the interactive REPL / a CLI
# subcommand; ``api`` = the HTTP server serving an external caller;
# ``desktop`` = the native Mac app.
SURFACE_CLI = "cli"
SURFACE_API = "api"
SURFACE_DESKTOP = "desktop"
ACTIVATION_KIND_SURFACE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        (ACTIVATION_FIRST_INFERENCE, SURFACE_CLI),
        (ACTIVATION_FIRST_INFERENCE, SURFACE_API),
        (ACTIVATION_MODEL_PULL, SURFACE_CLI),
        (ACTIVATION_AGENT_SETUP, SURFACE_CLI),
        (ACTIVATION_FIRST_CHAT_REPLY, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_VISION_REPLY, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_DICTATION, SURFACE_DESKTOP),
        (ACTIVATION_FIRST_IMAGE, SURFACE_DESKTOP),
    }
)
ACTIVATION_KINDS: frozenset[str] = frozenset(
    kind for kind, _ in ACTIVATION_KIND_SURFACE_PAIRS
)
ACTIVATION_SURFACES: frozenset[str] = frozenset(
    surface for _, surface in ACTIVATION_KIND_SURFACE_PAIRS
)
DESKTOP_ACTIVATION_KINDS: frozenset[str] = frozenset(
    kind
    for kind, surface in ACTIVATION_KIND_SURFACE_PAIRS
    if surface == SURFACE_DESKTOP
)


def is_allowed_activation(activation_kind: str, surface: str) -> bool:
    """Return whether a milestone is valid on the supplied product surface."""
    return (activation_kind, surface) in ACTIVATION_KIND_SURFACE_PAIRS


# ``rapid-mlx chat`` spawns its own ephemeral ``serve`` and drives it over
# HTTP, so first_inference is emitted at the server-side success chokepoint
# for BOTH surfaces. Rather than invent a new env var, we reuse the marker
# the chat front-end ALREADY sets on the server it spawns
# (``RAPID_MLX_CHAT_SPAWN=1``): a chat-spawned server is the ``cli`` surface,
# a standalone ``serve`` is ``api``. One emission site, no double-counting.
CHAT_SPAWN_ENV = "RAPID_MLX_CHAT_SPAWN"

# Generative endpoints whose successful, non-empty completion counts as
# engagement.
#
# The engine inference scope remains chat-completions engagement in spec v2;
# v2 adds Desktop milestone kinds and does not expand engine endpoints. This
# is an explicit, versioned scope decision, not an accidental omission.
# ``/v1/chat/completions`` (streaming + non-streaming) is the dominant surface
# (all CLI ``chat`` traffic auto-spawns a server that loops through it, plus
# the bulk of direct API usage) and is the single endpoint instrumented with
# both a ``request`` event and the ``activation`` emit.
#
# ``/v1/completions`` (routes/completions.py) and ``/v1/messages``
# (routes/anthropic.py) are separate, generative, and NOT part of the v1
# engagement contract: an install that inferences exclusively through them is
# out of scope for the engine's engaged metric by definition. Listing them here without
# wiring would over-promise coverage the code doesn't deliver; wiring them is a
# deliberate future item that MUST bump ``ACTIVATION_SPEC_VERSION`` and update the
# dashboard in lockstep. See docs/telemetry-activation.md.
INFERENCE_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/v1/chat/completions",
    }
)


def is_successful_inference(status: int, completion_tokens: int) -> bool:
    """The single success predicate shared by call sites and tests.

    A request is a successful inference iff the HTTP status is 2xx AND the
    generation was non-empty. An error response or an empty completion is
    explicitly NOT engagement (see docs/telemetry-activation.md).
    """
    try:
        status_ok = 200 <= int(status) < 300
        nonempty = int(completion_tokens) > 0
    except (TypeError, ValueError):
        return False
    return status_ok and nonempty
