# SPDX-License-Identifier: Apache-2.0
"""R12-4 — strict ``response_format.json_schema`` enforcement helpers.

This module implements the post-generate validation + auto-repair retry
path that drives strict-mode enforcement when the engine does NOT have
the ``[guided]`` extra installed (the path that previously surfaced a
400 ``guided_extra_required`` and broke pydantic-ai end-to-end — see
the H-06 carry / R12-4 PR body for the design rationale).

The design choice (option a in the R12-4 design doc):
    1. Run the engine UNCONSTRAINED (no outlines guidance).
    2. Strip any markdown / prose wrapper from the model output.
    3. Validate the parsed JSON against the supplied schema.
    4. If invalid AND repair is enabled: re-prompt the engine ONCE
       with a system-prompt-injected hint that names the validation
       error and demands ONLY valid JSON.
    5. If invalid after the repair attempt: surface 422 with a
       structured envelope (``code=json_schema_violation``,
       ``param=response_format.json_schema``, ``details`` carrying the
       failing path / expected / got).

The disable flag ``RAPID_MLX_STRICT_JSON_SCHEMA=off`` (default ``on``)
short-circuits step 3+ — strict requests fall through to the legacy
prompt-injection behavior. Operators who rely on the pre-R12-4
silent-pass-through behavior get the legacy code path back without
having to pin to an older release.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-variable feature flags
# ---------------------------------------------------------------------------

_DISABLE_FLAG = "RAPID_MLX_STRICT_JSON_SCHEMA"
_REPAIR_FLAG = "RAPID_MLX_STRICT_JSON_SCHEMA_REPAIR"

# Values that disable a feature flag (matches the
# ``RAPID_MLX_AUTO_PULL`` / ``RAPID_MLX_PROFILE_VERBOSE`` convention).
_OFF_VALUES = {"0", "off", "false", "no", "disable", "disabled"}


def strict_enforcement_enabled() -> bool:
    """Return ``True`` iff post-generate strict enforcement should run.

    Default: enabled. The env var ``RAPID_MLX_STRICT_JSON_SCHEMA=off``
    (``0`` / ``false`` / ``no`` / ``disable`` / ``disabled`` are also
    accepted as falsy) short-circuits enforcement so requests fall
    through to the legacy prompt-injection-only behavior.

    The disable flag is intended ONLY for operators who relied on the
    silent-pass-through behavior pre-R12-4 and need time to adapt their
    clients. It will likely be removed in a future release.
    """
    raw = os.environ.get(_DISABLE_FLAG, "").strip().lower()
    return raw not in _OFF_VALUES


def repair_retry_enabled() -> bool:
    """Return ``True`` iff the single auto-repair retry should run.

    Default: enabled. The env var
    ``RAPID_MLX_STRICT_JSON_SCHEMA_REPAIR=off`` disables ONLY the
    retry; the post-decode validation + 422 envelope still runs (so
    strict mode remains hard-contract; only the retry is skipped).
    Useful for cost-sensitive deployments that prefer to fail fast.
    """
    raw = os.environ.get(_REPAIR_FLAG, "").strip().lower()
    return raw not in _OFF_VALUES


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------

# Models trained on chat formats commonly wrap JSON in a ```json ... ```
# code fence even when the system prompt explicitly forbids it. We strip
# at MOST one outer fence — anything inside is left for ``json.loads``
# to either accept or reject. Conservatively scoped: we never recurse,
# and never reach for substring matches that could chew off legitimate
# string content.
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(?P<body>.*?)\n?```\s*$",
    re.DOTALL,
)


def _strip_markdown_fence(text: str) -> str:
    """Strip a single outer ```json ... ``` fence if present.

    Returns the input unchanged if no fence is detected. Conservatively
    matches at the start/end of the text — does not strip fences that
    appear mid-stream (those would surface as ``invalid JSON`` from
    ``json.loads`` and route to the 422 envelope, which is correct).
    """
    if not text:
        return text
    m = _FENCE_RE.match(text)
    if not m:
        return text
    return m.group("body")


def extract_json_payload(output_text: str) -> str:
    """Pull the JSON payload out of a model's raw output.

    Strategy:
        1. Strip leading/trailing whitespace.
        2. Strip a single outer ```json ... ``` fence.
        3. Strip leading/trailing whitespace again.

    The returned string is what we hand to ``json.loads``. If the model
    emitted prose on either side of the JSON, this helper returns the
    full text and ``json.loads`` will reject — which is the correct
    failure mode (the schema contract was violated).
    """
    if not output_text:
        return ""
    text = output_text.strip()
    text = _strip_markdown_fence(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Validation with structured error envelope
# ---------------------------------------------------------------------------


def validate_and_envelope(
    output_text: str, json_schema: dict[str, Any]
) -> tuple[bool, dict[str, Any] | None]:
    """Validate ``output_text`` against ``json_schema``.

    Returns ``(True, None)`` on success and ``(False, details)`` on
    failure. The ``details`` dict mirrors the R12-4 design envelope:

        {
            "failing_path": "/age",            # JSON pointer
            "expected":    "minimum: 18",      # validator + value
            "got":         5,                  # offending instance
            "message":     "5 is less than ...",
            "reason":      "schema_violation"  # or "invalid_json" / "empty"
        }

    The route layer wraps this in the OpenAI-shaped 422 envelope so
    SDKs can decode ``error.details.failing_path`` programmatically.
    """
    if not output_text or not output_text.strip():
        return False, {
            "reason": "empty",
            "message": "model emitted no content",
        }
    payload = extract_json_payload(output_text)
    if not payload:
        return False, {
            "reason": "empty",
            "message": "model emitted no content after fence-stripping",
        }
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return False, {
            "reason": "invalid_json",
            "message": f"output is not valid JSON: {exc}",
        }
    try:
        # Lazy-import so the helper module stays import-light for
        # routes that never touch json_schema.
        from jsonschema import Draft202012Validator, FormatChecker
        from jsonschema.exceptions import ValidationError
        from jsonschema.validators import validator_for
    except ImportError as exc:  # pragma: no cover
        # ``jsonschema`` is a base dep — this branch is dead code under
        # normal installs. Surface as 500-shape (no envelope) so the
        # route maps it to a generic 500, not a validation 422.
        raise RuntimeError(f"jsonschema not available: {exc}") from exc

    try:
        validator_cls = validator_for(json_schema)
    except TypeError:
        validator_cls = Draft202012Validator

    # Codex r1 BLOCKING #1: pass a ``FormatChecker`` so JSON Schema
    # ``format`` constraints (``"email"``, ``"uri"``, ``"date"``, …)
    # are enforced rather than treated as annotations. Pre-fix,
    # ``{"format":"email"}`` validated any string that satisfied
    # ``type:"string"`` — the format keyword was effectively ignored
    # and a violating output would surface a confusing 200 (or, if
    # ``type`` happened to also fail, a misleading ``type`` error).
    # The default ``FormatChecker`` covers the common formats out of
    # the box; specialised formats that require optional dependencies
    # (e.g. ``regex``) are best-effort but never throw — see the
    # jsonschema docs.
    validator = validator_cls(json_schema, format_checker=FormatChecker())
    # Codex r2 #1: take the FIRST error from ``iter_errors`` directly
    # rather than ``sorted(...)`` with a list-typed key. Pre-fix,
    # ``key=lambda e: list(e.absolute_path)`` would raise ``TypeError``
    # under Python 3 when two errors had ``absolute_path`` lists that
    # compared int-vs-str at the same index (e.g. one path
    # ``["users", 0]`` and another ``["users", "ids"]``) — turning a
    # legitimate schema violation into an HTTP 500. ``iter_errors``
    # already yields in deterministic depth-first order, so the FIRST
    # entry is the right one for the envelope; pulling it via
    # ``next()`` avoids materializing every error AND avoids the
    # mixed-type-comparison hazard.
    try:
        first: ValidationError = next(validator.iter_errors(parsed))
    except StopIteration:
        return True, None
    failing_path = "/" + "/".join(str(p) for p in first.absolute_path)
    if failing_path == "/":
        failing_path = "/" if list(first.absolute_path) else ""
    # Compact ``expected`` summary: ``"<validator>: <validator_value>"``
    # (e.g. ``"minimum: 18"``, ``"required: ['age', 'name']"``).
    try:
        expected_value = first.validator_value
    except AttributeError:
        expected_value = None
    expected = f"{first.validator}: {expected_value!r}" if first.validator else "schema"
    # ``instance`` is the offending value at the failing path; keep it
    # JSON-roundtripable for the envelope.
    try:
        got = json.loads(json.dumps(first.instance, default=str))
    except (TypeError, ValueError):
        got = repr(first.instance)
    return False, {
        "reason": "schema_violation",
        "failing_path": failing_path or "/",
        "expected": expected,
        "got": got,
        "message": first.message,
    }


# ---------------------------------------------------------------------------
# Repair hint injection
# ---------------------------------------------------------------------------


def build_repair_messages(
    original_messages: list[dict[str, Any]],
    failed_output: str,
    schema: dict[str, Any],
    failure_details: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the message list for the single repair retry.

    Strategy: append the failed assistant output as an assistant turn,
    then a SYSTEM turn with a hint that names the specific validation
    error and demands ONLY valid JSON. Keeping the failure visible to
    the model is important — models trained on RLHF respond much better
    to "your previous output was wrong because X; produce ONLY valid
    JSON" than to a silent re-run of the original prompt.

    The schema is included in the hint at most once (we de-duplicate
    against any prior schema injection — see
    ``build_json_system_prompt`` — so a strict request that already
    embedded the schema in the system prompt doesn't pay a 2x token
    cost on the repair turn).
    """
    try:
        schema_str = json.dumps(schema, indent=2)
    except (TypeError, ValueError):
        schema_str = str(schema)

    reason = failure_details.get("reason", "schema_violation")
    if reason == "invalid_json":
        hint_body = (
            "Your previous response was not valid JSON: "
            f"{failure_details.get('message', 'parse failed')}. "
            "Emit ONLY a single JSON object that conforms to the schema. "
            "Do NOT wrap it in markdown fences. Do NOT add any prose "
            "before or after the JSON."
        )
    elif reason == "empty":
        hint_body = (
            "Your previous response was empty. Emit ONLY a single JSON "
            "object that conforms to the schema. Do NOT wrap it in "
            "markdown fences. Do NOT add any prose before or after the JSON."
        )
    else:
        # ``schema_violation`` — name the failing path so the model has
        # concrete feedback to act on.
        path = failure_details.get("failing_path") or "/"
        expected = failure_details.get("expected", "schema constraint")
        got_repr = json.dumps(failure_details.get("got"), default=str)[:200]
        hint_body = (
            "Your previous response failed JSON Schema validation at "
            f"path '{path}': expected {expected}, got {got_repr}. "
            f"Detail: {failure_details.get('message', 'schema violation')}. "
            "Emit ONLY a single JSON object that conforms to the schema. "
            "Do NOT wrap it in markdown fences. Do NOT add any prose "
            "before or after the JSON."
        )

    hint = (
        "STRICT JSON SCHEMA REPAIR RETRY\n\n"
        f"{hint_body}\n\n"
        f"JSON Schema:\n```json\n{schema_str}\n```"
    )

    # The repair turn is a fresh system + user pair appended to the
    # original conversation. Including the failed output as a literal
    # assistant turn lets the model see exactly what it did wrong.
    repair = list(original_messages)
    if failed_output and failed_output.strip():
        repair.append({"role": "assistant", "content": failed_output})
    repair.append({"role": "system", "content": hint})
    repair.append(
        {
            "role": "user",
            "content": (
                "Retry the previous request. Emit ONLY the JSON object."
            ),
        }
    )
    return repair


# ---------------------------------------------------------------------------
# 422 envelope construction
# ---------------------------------------------------------------------------


def build_violation_envelope(
    details: dict[str, Any],
    *,
    param: str = "response_format.json_schema",
    attempts: int = 1,
) -> dict[str, Any]:
    """Build the 422 envelope body for a strict-mode validation failure.

    The shape is intentionally OpenAI-compatible:

        {
            "error": {
                "message": "...",
                "type":    "validation_error",
                "code":    "json_schema_violation",
                "param":   "response_format.json_schema",
                "details": {
                    "failing_path": "...",
                    "expected":     "...",
                    "got":          ...,
                    "attempts":     2
                }
            }
        }

    Clients keying off ``error.code`` can detect this case
    programmatically; SDK consumers (pydantic-ai etc.) can extract
    ``error.details.failing_path`` to drive an in-process retry.
    """
    reason = details.get("reason", "schema_violation")
    short_message = details.get("message", "schema violation")
    if reason == "invalid_json":
        prefix = "model output is not valid JSON"
    elif reason == "empty":
        prefix = "model output is empty"
    else:
        path = details.get("failing_path", "/")
        prefix = f"strict response_format violated at '{path}'"
    # Codex r3 NIT: derive the "the request set X=true" field name
    # from ``param`` so the message stays accurate on every surface.
    # /v1/responses passes ``param="text.format"`` and the legacy
    # message ("response_format.json_schema.strict=true") would
    # mislead clients reading the Responses-surface envelope into
    # thinking they need to fix a different field. Use the
    # surface-correct field name: chat uses
    # ``response_format.json_schema.strict``, Responses uses
    # ``text.format.strict``. We synthesize from ``param`` by
    # appending ``.strict`` when missing — both production callers
    # already pass the right shape.
    field_label = param if param.endswith(".strict") else f"{param}.strict"
    msg = (
        f"{prefix}: {short_message}. "
        f"The request set {field_label}=true; "
        f"the server made {attempts} attempt(s) to honor the contract "
        "and the final output did not validate. See error.details for "
        "the failing path / expected / got triple."
    )
    envelope_details: dict[str, Any] = {
        "attempts": attempts,
    }
    for key in ("failing_path", "expected", "got", "reason"):
        if key in details:
            envelope_details[key] = details[key]
    return {
        "error": {
            "message": msg,
            "type": "validation_error",
            "code": "json_schema_violation",
            "param": param,
            "details": envelope_details,
        }
    }
