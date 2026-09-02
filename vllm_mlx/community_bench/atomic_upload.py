# SPDX-License-Identifier: Apache-2.0
"""Explicit upload flow for atomic Community Benchmark runs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from .benchmark_contracts import BenchmarkRunValidator, SubmissionReceiptValidator
from .upload import (
    SubmitError,
    board_url,
    commit_install_id,
    peek_install_id,
    post_submission,
    submission_body,
    validate_board_url,
)


@dataclass(frozen=True)
class AtomicUploadAcceptance:
    receipt: dict[str, Any]
    install_id: str
    payload_digest: str


def _ask_consent(
    payload: dict[str, Any], *, target: str, stdin: TextIO, stdout: TextIO
) -> bool:
    print("", file=stdout)
    print(f"About to upload this benchmark to {target}:", file=stdout)
    print("=" * 72, file=stdout)
    print(submission_body(payload).decode("utf-8"), file=stdout)
    print("=" * 72, file=stdout)
    print(
        "Everything shown above leaves this Mac. It includes model source, "
        "Mac configuration, OS/runtime versions, execution settings, timings, "
        "and a random resettable install id. It does not include your name, "
        "hostname, serial number, hardware UUID, IP address, prompts, outputs, "
        "or file paths. The JSON payload has no IP-address field. Like any "
        "HTTPS service, the destination observes the source IP; this endpoint "
        "uses it for short-lived request rate limiting and does not put it in "
        "the benchmark record.",
        file=stdout,
    )
    print("Upload this result? [y/N] ", end="", flush=True, file=stdout)
    answer = stdin.readline()
    return answer.strip().lower() in {"y", "yes"}


def _validated_receipt(
    response: dict[str, Any], run_id: str, run_digest: str
) -> dict[str, Any]:
    receipt = response.get("receipt")
    if not isinstance(receipt, dict):
        raise SubmitError("the board accepted the request without a submission receipt")
    try:
        SubmissionReceiptValidator().validate(receipt)
    except ValueError as exc:
        raise SubmitError(
            f"the board returned an invalid submission receipt: {exc}"
        ) from exc
    if receipt["submission_id"] != run_id:
        raise SubmitError("the board receipt does not identify the uploaded run")
    if receipt["run_digest"] != run_digest:
        raise SubmitError("the board receipt does not identify the uploaded payload")
    return copy.deepcopy(receipt)


def _ecmascript_number(value: float) -> str:
    """Render one finite float like JSON.stringify / ECMAScript Number::toString."""

    if not math.isfinite(value):
        raise ValueError("benchmark payload contains a non-finite number")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    raw = repr(abs(value)).lower()
    mantissa, _, exponent_text = raw.partition("e")
    exponent = int(exponent_text or "0")
    whole, dot, fraction = mantissa.partition(".")
    combined = whole + (fraction if dot else "")
    leading_zeroes = len(combined) - len(combined.lstrip("0"))
    digits = combined.lstrip("0").rstrip("0")
    decimal_position = len(whole) + exponent - leading_zeroes
    if -6 < decimal_position <= 21:
        if decimal_position <= 0:
            return sign + "0." + "0" * (-decimal_position) + digits
        if decimal_position >= len(digits):
            return sign + digits + "0" * (decimal_position - len(digits))
        return sign + digits[:decimal_position] + "." + digits[decimal_position:]
    coefficient = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    scientific_exponent = decimal_position - 1
    exponent_sign = "+" if scientific_exponent >= 0 else ""
    return f"{sign}{coefficient}e{exponent_sign}{scientific_exponent}"


def _atomic_canonical(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _ecmascript_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_atomic_canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                f"{_atomic_canonical(key)}:{_atomic_canonical(value[key])}"
                for key in sorted(value)
            )
            + "}"
        )
    raise TypeError(f"unsupported benchmark payload value: {type(value).__name__}")


def atomic_run_digest(run: dict[str, Any]) -> str:
    """Digest a run with the ingestion service's sorted JSON canonical form."""

    canonical = _atomic_canonical(run).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def preview_run(
    run: dict[str, Any], *, install_id: str | None = None, url: str | None = None
) -> dict[str, Any]:
    """Build the exact wire payload without writing or sending anything."""

    BenchmarkRunValidator().validate(run)
    base = (validate_board_url(url) if url is not None else board_url()).rstrip("/")
    target = base if url or base.endswith("/atomic") else f"{base}/atomic"
    candidate = install_id or peek_install_id()
    wire = copy.deepcopy(run)
    wire["install_id"] = candidate
    BenchmarkRunValidator().validate(wire)
    body = submission_body(wire)
    return {
        "target": target,
        "install_id": candidate,
        "payload_digest": atomic_run_digest(wire),
        "body_digest": f"sha256:{hashlib.sha256(body).hexdigest()}",
        "payload_json": body.decode("utf-8"),
        "payload": wire,
    }


def upload_run(
    run: dict[str, Any],
    *,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    url: str | None = None,
    approved_install_id: str | None = None,
    approved_payload_digest: str | None = None,
    approved_body_digest: str | None = None,
    approved_target: str | None = None,
) -> AtomicUploadAcceptance | None:
    """Upload one validated run, returning its server receipt.

    ``None`` means an interactive user declined. ``assume_yes`` exists for a
    caller such as Rapid Desktop that presents its own native confirmation.
    """

    preview = preview_run(run, install_id=approved_install_id, url=url)
    target = preview["target"]
    candidate = preview["install_id"]
    wire = preview["payload"]
    wire_digest = preview["payload_digest"]
    body_digest = preview["body_digest"]
    if approved_target is not None and approved_target != target:
        raise SubmitError(
            "the upload destination changed after preview; review the payload again"
        )
    if approved_payload_digest is not None and approved_payload_digest != wire_digest:
        raise SubmitError(
            "the archived benchmark changed after preview; review the payload again"
        )
    if approved_body_digest is not None and approved_body_digest != body_digest:
        raise SubmitError(
            "the serialized benchmark changed after preview; review the payload again"
        )

    out = stdout or sys.stdout
    inp = stdin or sys.stdin
    if not assume_yes and not _ask_consent(wire, target=target, stdin=inp, stdout=out):
        return None

    settled = commit_install_id(candidate)
    if settled != candidate:
        raise SubmitError(
            "the install id changed before upload; review the payload and try again"
        )
    response = post_submission(wire, url=target)
    receipt = _validated_receipt(response, run["run_id"], wire_digest)
    return AtomicUploadAcceptance(receipt, candidate, wire_digest)


__all__ = [
    "AtomicUploadAcceptance",
    "atomic_run_digest",
    "preview_run",
    "upload_run",
]
