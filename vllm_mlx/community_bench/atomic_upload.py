# SPDX-License-Identifier: Apache-2.0
"""Explicit upload flow for atomic Community Benchmark runs."""

from __future__ import annotations

import copy
import json
import sys
from typing import Any, TextIO

from .benchmark_contracts import BenchmarkRunValidator, SubmissionReceiptValidator
from .upload import (
    SubmitError,
    board_url,
    commit_install_id,
    peek_install_id,
    post_submission,
)


def _ask_consent(
    payload: dict[str, Any], *, target: str, stdin: TextIO, stdout: TextIO
) -> bool:
    print("", file=stdout)
    print(f"About to upload this benchmark to {target}:", file=stdout)
    print("=" * 72, file=stdout)
    print(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=stdout
    )
    print("=" * 72, file=stdout)
    print(
        "Everything shown above leaves this Mac. It includes model source, "
        "Mac configuration, OS/runtime versions, execution settings, timings, "
        "and a random resettable install id. It does not include your name, "
        "hostname, serial number, hardware UUID, IP address, prompts, outputs, "
        "or file paths.",
        file=stdout,
    )
    print("Upload this result? [y/N] ", end="", flush=True, file=stdout)
    answer = stdin.readline()
    return answer.strip().lower() in {"y", "yes"}


def _validated_receipt(response: dict[str, Any], run_id: str) -> dict[str, Any]:
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
    return copy.deepcopy(receipt)


def upload_run(
    run: dict[str, Any],
    *,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    url: str | None = None,
) -> dict[str, Any] | None:
    """Upload one validated run, returning its server receipt.

    ``None`` means an interactive user declined. ``assume_yes`` exists for a
    caller such as Rapid Desktop that presents its own native confirmation.
    """

    BenchmarkRunValidator().validate(run)
    base = (url or board_url()).rstrip("/")
    target = base if url or base.endswith("/atomic") else f"{base}/atomic"
    candidate = peek_install_id()
    wire = copy.deepcopy(run)
    wire["install_id"] = candidate
    BenchmarkRunValidator().validate(wire)

    out = stdout or sys.stdout
    inp = stdin or sys.stdin
    if not assume_yes and not _ask_consent(wire, target=target, stdin=inp, stdout=out):
        return None

    settled = commit_install_id(candidate)
    wire["install_id"] = settled
    BenchmarkRunValidator().validate(wire)
    response = post_submission(wire, url=target)
    return _validated_receipt(response, run["run_id"])


__all__ = ["upload_run"]
