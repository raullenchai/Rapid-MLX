# SPDX-License-Identifier: Apache-2.0
"""JSON nesting-depth helpers (D-TOOL-RECUR / D-DEEP-JSON).

A client-supplied JSON body with extreme structural nesting
(``[[[…1…]]]`` or ``{"a":{"a":…}}`` ~1000 levels deep, ~10 KB on the
wire) crashed the worker with HTTP 500 because the Python validation
stack ascended one frame per nesting level and hit the recursion
limit. Pydantic v2 body validation on every ``/v1/...`` route is
recursive over the input tree; the chat-template tool sanitiser at
:mod:`vllm_mlx.utils.chat_template` was recursive over
``tools[].function.parameters`` (D-TOOL-RECUR, cross-confirmed on five
parsers). Same payload as a non-validated field returned 200 — proving
the surface was validator-recursion, not parser-bytes.

This module owns the structural-depth measurement used by:

* The per-tool depth validator on :class:`ToolDefinition.function.parameters`
  (env override ``RAPID_MLX_MAX_TOOL_SCHEMA_DEPTH``, default 64) — rejects
  with the canonical 400 envelope before the chat-template sanitiser
  runs.
* The whole-body depth guard wired into
  :class:`vllm_mlx.middleware.body_depth.RequestBodyDepthMiddleware`
  (env override ``RAPID_MLX_MAX_BODY_DEPTH``, default 64) — rejects any
  body whose JSON nesting depth exceeds the cap before FastAPI / Pydantic
  recurses over it.

Both guards use :func:`json_nesting_depth_exceeds`, which walks
iteratively with an explicit work stack so the depth measurement
itself cannot crash the worker.

The defaults (64) are deliberately generous: real-world tool schemas
top out around depth 8–12 (a function with ``properties.x.items.
properties.y.…`` is already a stretch); legitimate
``messages[].content`` shapes are flat lists of dicts (depth 3–4). 64
gives headroom for adversarial-but-benign nested ``$ref``-heavy
schemas while staying well under the Python recursion limit (default
1000) that broke pre-fix. Operators can raise / lower via the env
overrides if they have a workload that needs different bounds.
"""

from __future__ import annotations

import os
from typing import Any

# Env knobs. Resolved at request time (via :func:`resolve_max_*`) so a
# test fixture that mutates ``os.environ`` between cases doesn't need
# to rebuild the FastAPI app.
MAX_TOOL_SCHEMA_DEPTH_ENV = "RAPID_MLX_MAX_TOOL_SCHEMA_DEPTH"
MAX_BODY_DEPTH_ENV = "RAPID_MLX_MAX_BODY_DEPTH"

# Default caps. See module docstring for the "64 vs the Python
# recursion limit" rationale. Centralised so the request-model
# validator, the middleware, and the regression suite all read the
# same number.
DEFAULT_MAX_TOOL_SCHEMA_DEPTH = 64
DEFAULT_MAX_BODY_DEPTH = 64


def _resolve_env_int(name: str, default: int) -> int:
    """Read ``os.environ[name]`` as a positive int with a sane fallback.

    A non-integer / empty value falls back to ``default`` (NOT 0 —
    silently disabling the depth gate on a typo would mask the real
    cause; mirrors :func:`vllm_mlx.middleware.body_size._resolve_limit`).
    A non-positive value (``0``, ``-1``) is honoured as "disable the
    gate" so operators can opt out via ``RAPID_MLX_MAX_BODY_DEPTH=0``.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_max_tool_schema_depth() -> int:
    """Resolve the per-tool-schema depth cap from env, defaulting to 64."""
    return _resolve_env_int(MAX_TOOL_SCHEMA_DEPTH_ENV, DEFAULT_MAX_TOOL_SCHEMA_DEPTH)


def resolve_max_body_depth() -> int:
    """Resolve the whole-body depth cap from env, defaulting to 64."""
    return _resolve_env_int(MAX_BODY_DEPTH_ENV, DEFAULT_MAX_BODY_DEPTH)


def json_nesting_depth_exceeds(obj: Any, max_depth: int) -> bool:
    """Return ``True`` iff ``obj``'s nesting depth strictly exceeds ``max_depth``.

    Counts containers only: each ``dict`` / ``list`` / ``tuple`` adds
    one level. Scalars (``str``/``int``/``float``/``bool``/``None``)
    are leaves at depth 0. So:

      * ``1`` → depth 0
      * ``{"a": 1}`` → depth 1
      * ``[[[1]]]`` → depth 3
      * ``{"a": {"b": [1, 2]}}`` → depth 3

    ``max_depth <= 0`` disables the check (returns ``False`` for any
    input). The walk uses an explicit work stack so the depth
    measurement itself cannot crash the worker on adversarial input —
    if the C JSON parser already materialised the tree, this function
    can always measure it.

    Early-exits the moment the stack reaches ``max_depth + 1`` so an
    obviously-too-deep payload doesn't pay the O(N) full traversal
    cost.
    """
    if max_depth <= 0:
        return False
    if not isinstance(obj, (dict, list, tuple)):
        return False
    # Stack entries: (node, depth_at_node). The root container sits at
    # depth 1; one level of nesting adds one to the count.
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            return True
        # Only descend into the values that can themselves nest. Scalar
        # keys (str / int) do not extend the depth.
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, (dict, list, tuple)):
                    stack.append((v, depth + 1))
        elif isinstance(node, (list, tuple)):
            for v in node:
                if isinstance(v, (dict, list, tuple)):
                    stack.append((v, depth + 1))
    return False
