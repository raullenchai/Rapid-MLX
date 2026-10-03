# SPDX-License-Identifier: Apache-2.0
""" "Ask us to support it?" — an opt-in support request after a refusal.

When the BYOM preflight refuses a PUBLIC Hub model because no loader here
builds its architecture or because it ships only GGUF / PyTorch ``.bin``
weights, the CLI offers to file a support request. It is sent only after the
user answers ``y`` on an interactive terminal, or when ``--request`` was
passed (for non-interactive use). Never for gated/private repos or local
paths, and never for "too big for this Mac".

Exactly five fields go to ``rapidmlx.com/api/model-request``: the repo id,
``model_type``, weight format, failure class and Rapid-MLX version. The site
deduplicates requests by failure class + architecture/format into ONE GitHub
issue per key and returns its URL and vote count. If the endpoint cannot be
reached, the CLI prints a prefilled GitHub issue link instead, so the user can
still file it themselves.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode

from rapid_mlx.byom import preflight as pf

ENDPOINT = "https://rapidmlx.com/api/model-request"
ISSUE_FORM = "https://github.com/raullenchai/Rapid-MLX/issues/new"
TIMEOUT_SECONDS = 8.0
_REQUESTABLE = (pf.UNSUPPORTED_ARCHITECTURE, pf.UNSUPPORTED_FORMAT)

CONSENT_PROMPT = (
    "  Ask us to support it? This sends the repo id, architecture,\n"
    "  format and your Rapid-MLX version. Nothing else. [y/N] "
)


def request_payload(
    inspection: pf.Inspection, verdict: pf.Verdict, version: str
) -> dict[str, Any] | None:
    """The five-field request, or ``None`` when this refusal is not eligible."""
    if (
        verdict.failure not in _REQUESTABLE
        or inspection.is_local
        or not inspection.public
    ):
        return None
    if verdict.failure == pf.UNSUPPORTED_FORMAT:
        fmt = "gguf" if verdict.format_label == "GGUF" else "pytorch"
    else:
        fmt = "mlx" if verdict.format_label.startswith("MLX") else "safetensors"
    model_type = verdict.model_type.lower() if verdict.model_type else None
    return {
        "repo": inspection.ref,
        "model_type": model_type,
        "format": fmt,
        "failure": verdict.failure,
        "version": version,
    }


BUSY = "busy"


def _post(payload: dict[str, Any]) -> dict[str, Any] | str | None:
    """The site's answer, :data:`BUSY` while another request files the same
    issue, or ``None`` when the site could not take the request."""
    from rapid_mlx._version_check import USER_AGENT

    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read(64 * 1024).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return BUSY if exc.code == 409 else None
    except Exception:
        return None
    votes = body.get("votes") if isinstance(body, dict) else None
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("issue_url"), str)
        or not body["issue_url"].startswith("https://github.com/")
        or not (votes is None or (isinstance(votes, int) and votes > 0))
    ):
        return None
    return body


def fallback_link(payload: dict[str, Any]) -> str:
    """A prefilled ``model_support.yml`` issue form for the same request."""
    subject = payload["model_type"] or payload["format"]
    details = (
        f"Refused by the pre-download check: {payload['failure']} "
        f"(format: {payload['format']}, model_type: {payload['model_type']}, "
        f"Rapid-MLX {payload['version']})."
    )
    query = urlencode(
        {
            "template": "model_support.yml",
            "title": f"Model support request: {subject}",
            "model_name": payload["repo"].split("/", 1)[-1],
            "hf_id": payload["repo"],
            "details": details,
        }
    )
    return f"{ISSUE_FORM}?{query}"


def _wants_request(args: Any) -> bool:
    if getattr(args, "request", False):
        return True
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "  Ask us to support it: re-run with --request "
            "(sends the repo id, architecture, format and version).",
            file=sys.stderr,
        )
        return False
    try:
        answer = input(CONSENT_PROMPT)
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return False
    return answer.strip().lower() in {"y", "yes"}


def offer(
    args: Any, inspection: pf.Inspection, verdict: pf.Verdict, version: str
) -> None:
    """Offer (or, with ``--request``, send) a support request. Never raises."""
    payload = request_payload(inspection, verdict, version)
    if payload is None or not _wants_request(args):
        return
    result = _post(payload)
    if result == BUSY:
        print(
            "  Someone is filing this request right now; re-run with --request "
            "in a minute to add your vote.",
            file=sys.stderr,
        )
        return
    if not isinstance(result, dict):
        print(
            "  Couldn't reach rapidmlx.com. Open the request yourself (prefilled):",
            file=sys.stderr,
        )
        print(f"    {fallback_link(payload)}", file=sys.stderr)
        return
    votes = result["votes"]
    if result.get("created"):
        print("  ✓ Opened a support request:", file=sys.stderr)
    elif votes is None:
        print("  ✓ Added your vote to an existing request:", file=sys.stderr)
    else:
        people = "person" if votes == 1 else "people"
        print(
            f"  ✓ Added your vote to an existing request ({votes} {people} so far):",
            file=sys.stderr,
        )
    print(f"    {result['issue_url']}", file=sys.stderr)
    print("  Follow it there to hear when it ships.", file=sys.stderr)
