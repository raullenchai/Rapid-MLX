# SPDX-License-Identifier: Apache-2.0
"""Shared activation names, surfaces, and inference-success predicate.

Nothing here transmits or reads consent. Engine and desktop v2 emitters use
these definitions so the event registry and local marker names stay aligned.
"""

from __future__ import annotations

# Local milestones. Each marker may be claimed at most once per install id.
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
# The engine inference scope is chat-completions engagement.
# ``/v1/chat/completions`` (streaming + non-streaming) is the dominant surface
# (all CLI ``chat`` traffic auto-spawns a server that loops through it, plus
# the bulk of direct API usage) and is the single endpoint instrumented with
# for the inference milestone.
#
# ``/v1/completions`` (routes/completions.py) and ``/v1/messages``
# (routes/anthropic.py) are separate, generative, and not part of this
# engagement contract: an install that inferences exclusively through them is
# out of scope for the engine's engaged metric by definition. Listing them here without
# wiring would over-promise coverage the code doesn't deliver; wiring them is a
# deliberate future item.
INFERENCE_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/v1/chat/completions",
    }
)


def is_successful_inference(status: int, completion_tokens: int) -> bool:
    """The single success predicate shared by call sites and tests.

    A request is a successful inference iff the HTTP status is 2xx AND the
    generation was non-empty. An error response or an empty completion is
    explicitly NOT engagement.
    """
    try:
        status_ok = 200 <= int(status) < 300
        nonempty = int(completion_tokens) > 0
    except (TypeError, ValueError):
        return False
    return status_ok and nonempty
