# SPDX-License-Identifier: Apache-2.0
"""Dependency-free error types + envelope builders for structured output.

This module deliberately imports NOTHING heavy (no ``mlx`` / ``llguidance`` /
``jsonschema``) so that lightweight consumers — the FastAPI exception handler
registered at app startup, route modules, and the shared route-boundary
validator — can import ``GuidedSchemaCompileError`` and its 400-envelope builder
WITHOUT triggering native MLX / llguidance module initialization on apps that
never touch guided decoding.

``vllm_mlx.api.guided`` re-exports both names for backward compatibility.
"""

from __future__ import annotations

from typing import Any

# Per-surface locator for the offending field in the 400 body. The chat/
# completions API nests the schema under ``response_format.json_schema.schema``;
# the responses API under ``text.format.schema``. The chat param is the DEFAULT
# because it is both the most common surface and the historical value.
CHAT_RESPONSE_FORMAT_PARAM = "response_format.json_schema.schema"
RESPONSES_TEXT_FORMAT_PARAM = "text.format.schema"


class GuidedSchemaCompileError(Exception):
    """A caller-supplied structured-output schema/grammar failed to compile.

    Raised by the guided layer (``GuidedGenerator._decode_constrained``) when
    llguidance rejects a grammar at matcher construction. It is CAUGHT inside
    ``generate_json`` and degraded to ``None`` (the operational path): the
    structural validity of a caller schema is settled ONCE, up front, at the
    route boundary (``nonstrict_json_schema_boundary_error`` for non-strict
    json_schema; the strict ``check_schema_validity`` pre-flight for strict), so
    any failure that reaches the guided layer is operational — an
    unsupported-but-valid construct, a tokenizer/model-compat issue, an internal
    compiler limit, or a truncated parse — NOT a client fault. The class is also
    used as a lightweight carrier by the boundary validator to build the
    canonical 400 envelope via :func:`guided_schema_compile_error_detail`.
    """


class GuidedGenerationCancelledError(Exception):
    """Internal control signal for cooperatively cancelled guided work.

    This is deliberately distinct from an operational guided-decoder failure:
    callers must terminate the request instead of falling back to unconstrained
    generation after the client has cancelled its schema-constrained request.
    """

    def __init__(self, *, lifecycle_task: object | None = None) -> None:
        super().__init__()
        # When shutdown initiated the stop, this is the exact task entered in
        # the engine's lifecycle ledger.  The guided worker may be nested under
        # a disconnect guard, so consumers must not substitute current_task().
        self.lifecycle_task = lifecycle_task


def guided_schema_compile_error_detail(
    exc: BaseException,
    param: str | None = None,
) -> dict[str, Any]:
    """Build the canonical OpenAI-shaped 400 envelope for an invalid schema.

    Shared by every route-boundary validator so the 400 body is byte-identical
    across ``/v1/chat/completions`` and ``/v1/responses``. ``param`` locates the
    offending field per surface: ``response_format.json_schema.schema`` on the
    chat/completions API (the default), ``text.format.schema`` on the responses
    API. The message embeds only the schema-level diagnostic carried by ``exc``
    (the caller's own malformed schema, confirmed by an independent validator) —
    never a server-internal exception. The boundary validator only runs
    ``check_schema_validity`` (a meta-schema check); it never invokes the
    llguidance compiler, so the wording says the schema is INVALID rather than
    that it "failed to compile" (which is reserved for an actual llguidance
    rejection and no longer flows through this envelope).
    """
    resolved = param if param is not None else CHAT_RESPONSE_FORMAT_PARAM
    return {
        "error": {
            "message": f"{resolved} is not a valid JSON schema: {exc}",
            "type": "invalid_request_error",
            "code": "invalid_response_format_schema",
            "param": resolved,
        }
    }
