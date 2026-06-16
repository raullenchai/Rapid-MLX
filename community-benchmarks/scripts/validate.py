#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate one or more community benchmark submission files.

Run modes:

    # Validate every JSON file under submissions/
    python community-benchmarks/scripts/validate.py

    # Validate specific files (used by the GHA on PR diff)
    python community-benchmarks/scripts/validate.py path/to/sub.json ...

Layered checks, in order:

1. **JSON parse** — file must decode.
2. **Schema** — must match ``community-benchmarks/schema.json``
   (draft 2020-12, ``additionalProperties: false`` everywhere). The
   const fields on ``config.rounds`` etc. are enforced here.
3. **Whitelist** — ``model.alias`` must exist in
   ``vllm_mlx/aliases.json`` and ``model.hf_path`` must match the
   value stored there. We re-check after the CLI's own whitelist
   guard because the JSON file in a PR is the authoritative artifact;
   anything else is just history.
4. **Sanity** — decode_tps > 0, ttft_ms < 30 s, peak_ram_mb <= chip's
   RAM, chip non-empty, etc. These don't catch every fraud, but they
   cut the noise floor so a real outlier still gets reviewer attention.

Exit code is the number of failed files (capped at 125 so it fits in a
shell exit status). 0 = all clean. The GHA fails the job on non-zero.

Designed to run with stdlib only when ``jsonschema`` isn't installed —
in that case schema validation is skipped with a clear warning. The
GHA installs ``jsonschema`` explicitly, so CI always runs the full
check; local invocations stay friction-free.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


def _reject_non_finite(constant: str) -> None:
    """Reject ``NaN`` / ``Infinity`` / ``-Infinity`` during JSON decode.

    Python's stdlib ``json`` accepts these by default (it emits and
    parses them as a permissive extension). Every comparison against
    NaN returns False, so a hand-edited submission with
    ``decode_tps: NaN`` would silently pass the ``> 0`` / ``< MAX``
    sanity bounds. (Codex PR #582 round-7 BLOCKING.) Hooking
    ``parse_constant`` here makes the parser raise instead.
    """
    raise _IssueError(f"json: non-finite number ({constant}) is not permitted")


def _has_non_finite(obj) -> bool:
    """Recursively scan a decoded payload for ``NaN`` / ``inf``.

    Defence in depth alongside ``_reject_non_finite``: a serializer that
    emits non-finite floats outside the standard tokens (e.g. some
    library encoding ``inf`` as a literal Python repr) would not trip
    ``parse_constant``. This catches anything the parser missed.
    """
    if isinstance(obj, float):
        return not math.isfinite(obj)
    if isinstance(obj, dict):
        return any(_has_non_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_non_finite(v) for v in obj)
    return False


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"
ALIASES_PATH = REPO_ROOT / "vllm_mlx" / "aliases.json"
SUBMISSIONS_DIR = REPO_ROOT / "community-benchmarks" / "submissions"

# Sanity bounds. Wider than any realistic Apple Silicon number so a
# legitimate outlier still passes — these only catch obvious tampering
# (1e9 decode_tps, negative TTFT) and unit confusion.
MAX_DECODE_TPS = 2_000.0  # M3 Ultra single-line tops out ~140 in the wild
MAX_PREFILL_TPS = 50_000.0  # llama.cpp pp can hit ~10k on Ultra; cap well above
MAX_TTFT_MS = 30_000.0  # 30 s is well past "the model failed to load"
MAX_RAM_GB = 1024
# Filename pattern: <YYYYMMDD>-<chip-slug>-<alias-slug>-<id>.json. We
# don't enforce the exact slugs (chip names change) — just the shape.
FILENAME_RE = re.compile(r"^[0-9]{8}-[a-z0-9-]+-[a-z0-9.-]+-[0-9a-f]{12}\.json$")
# The CLI gates on ``is_apple_silicon()`` before benching, but the
# submission file in a PR is the authoritative artifact — a hand-edited
# JSON for non-Apple hardware would otherwise bypass the
# Apple-Silicon-only contract. Pattern matches the strings
# ``sysctl -n machdep.cpu.brand_string`` actually emits: "Apple M1",
# "Apple M3 Pro", "Apple M4 Max", "Apple M3 Ultra", etc. (Codex PR
# #582 round-3 BLOCKING.)
APPLE_CHIP_RE = re.compile(r"^Apple M\d+(?:\s+(?:Pro|Max|Ultra))?$")


class _IssueError(Exception):
    """Validation failure with a single human-readable line."""


def _load_schema() -> dict | None:
    if not SCHEMA_PATH.exists():
        print(f"  WARN: schema not found at {SCHEMA_PATH}; skipping schema check")
        return None
    return json.loads(SCHEMA_PATH.read_text())


def _load_aliases() -> dict[str, dict]:
    """Read ``vllm_mlx/aliases.json`` directly — no engine import needed."""
    if not ALIASES_PATH.exists():
        return {}
    raw = json.loads(ALIASES_PATH.read_text())
    return raw if isinstance(raw, dict) else {}


def _check_schema(payload: dict, schema: dict | None) -> None:
    """Raise ``_IssueError`` if the payload doesn't match the JSON Schema.

    ``jsonschema`` is mandatory. The previous fallback ("just warn and
    skip the schema check if the lib is missing") silently demoted the
    most-load-bearing gate in the validator to a no-op — a contributor
    or a CI mis-config without ``jsonschema`` installed would pass
    schema validation purely because the lib wasn't there. (Codex PR
    #582 round-7 BLOCKING.) The GHA pins ``jsonschema>=4.0`` so CI
    always has it; local invocations get a clear install hint instead
    of a silent pass.
    """
    if schema is None:
        return
    try:
        import jsonschema
    except ImportError as exc:
        raise _IssueError(
            "schema: jsonschema package is required for validation but is "
            "not installed. Install it with `pip install 'jsonschema>=4.0'` "
            "and re-run."
        ) from exc
    # ``jsonschema.validate()`` ignores ``format`` by default — it
    # advertises but doesn't enforce. Even when the FORMAT_CHECKER is
    # wired in (as below), ``date-time`` resolves to a no-op stub unless
    # ``rfc3339-validator`` is also installed — a transitive that
    # neither this validator nor jsonschema declares. The previous code
    # passed ``FORMAT_CHECKER`` and assumed it covered date-time, so on
    # any install without rfc3339-validator (which is most installs)
    # ``"submitted_at": "definitely not a timestamp"`` passed silently.
    # Defense in depth: build a FormatChecker that delegates to Python's
    # stdlib ``datetime.fromisoformat`` so the gate works regardless of
    # which optional jsonschema extras the host environment carries.
    # (Codex PR #582 round-3 NIT; resurfaced by PR #600 full_unit sweep.)
    checker = jsonschema.FormatChecker()

    # RFC 3339 ``date-time`` grammar (the strict subset of ISO 8601 the
    # schema promises): ``YYYY-MM-DDThh:mm:ss[.frac](Z|±hh:mm)``. We
    # require the time component (``T`` or space separator + ``hh:mm:ss``)
    # AND a timezone offset. ``datetime.fromisoformat`` alone is too
    # loose: it accepts date-only strings (``"2026-06-16"``) and
    # naïve timestamps without offset (``"2026-06-16T12:00:00"``) — both
    # are rejected by RFC 3339 but would pass schema validation
    # silently. (Codex PR #602 round-1 BLOCKING.) Belt-and-suspenders:
    # match the wire shape with a regex, THEN call fromisoformat to
    # confirm semantic validity (real month / day / leap-year).
    # Strict RFC 3339 disallows the space separator that ISO 8601
    # permits as a "readability" alternative — the ABNF in §5.6 only
    # admits ``T``/``t`` between date and time. JSON Schema's
    # ``format: date-time`` references RFC 3339, so we follow it.
    # (Codex PR #602 round-2 BLOCKING.)
    # The offset is also case-insensitive in RFC 3339 §5.6 — both ``Z``
    # and ``z`` are valid for UTC. Earlier rounds accepted ``[Tt]`` but
    # only ``Z``, which silently rejected payloads emitting ``...z``.
    # (Codex PR #602 round-3 BLOCKING.)
    _RFC3339_DATETIME = re.compile(
        r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}"
        r"(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$"
    )

    @checker.checks("date-time", raises=(ValueError, TypeError))
    def _is_rfc3339_datetime(value: object) -> bool:
        if not isinstance(value, str):
            return False
        if not _RFC3339_DATETIME.match(value):
            return False
        # ``fromisoformat`` doesn't accept the trailing ``Z`` shorthand
        # for UTC until 3.11+; normalise it (case-insensitive) to
        # ``+00:00`` so payloads generated by the CLI (which writes
        # ``...Z``) pass on the full 3.10–3.13 matrix that the project
        # supports. The lowercase ``z`` variant is admitted by RFC 3339
        # §5.6 and must be normalised the same way.
        normalized = re.sub(r"[Zz]$", "+00:00", value)
        parsed = datetime.fromisoformat(normalized)
        # Final belt: a passing regex with an implausible date
        # (``9999-99-99T...``) would still be caught by fromisoformat,
        # but a tz-less ISO 8601 that the regex somehow matched (it
        # can't) would fall through silently. Re-assert tzinfo.
        return parsed.tzinfo is not None

    validator = jsonschema.Draft202012Validator(schema, format_checker=checker)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    if errors:
        e = errors[0]
        loc = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise _IssueError(f"schema: {loc}: {e.message}") from None


def _check_alias_whitelist(payload: dict, aliases: dict[str, dict]) -> None:
    alias = payload.get("model", {}).get("alias")
    hf_path = payload.get("model", {}).get("hf_path")
    if not alias:
        raise _IssueError("alias: missing model.alias")
    if alias not in aliases:
        raise _IssueError(
            f"alias: '{alias}' is not on the whitelist "
            f"(vllm_mlx/aliases.json). Register it there first."
        )
    # Fail closed if the alias entry doesn't carry a usable ``hf_path``.
    # The previous version skipped the comparison whenever
    # ``aliases[alias]["hf_path"]`` was missing or empty, which meant a
    # malformed whitelist entry silently accepted arbitrary submitted
    # paths — including ones the contributor never ran. (Codex PR #582
    # round-7 BLOCKING.) A whitelist with a missing ``hf_path`` is a
    # data error worth surfacing, not a permission to skip the check.
    entry = aliases[alias]
    expected_path = entry.get("hf_path") if isinstance(entry, dict) else None
    if not isinstance(expected_path, str) or not expected_path:
        raise _IssueError(
            f"alias: whitelist entry for '{alias}' has no usable hf_path — "
            f"fix vllm_mlx/aliases.json before any submission for this "
            f"alias can be validated."
        )
    if hf_path != expected_path:
        raise _IssueError(
            f"alias: model.hf_path '{hf_path}' does not match registered "
            f"hf_path for '{alias}' (expected '{expected_path}')"
        )


def _check_sanity(payload: dict) -> None:
    """Plausibility bounds. Each violation raises ``_IssueError``."""
    hw = payload.get("hardware", {})
    chip = hw.get("chip", "")
    if not chip:
        raise _IssueError("hardware: chip is empty")
    # Enforce Apple Silicon at the validator boundary so a hand-edited
    # JSON for non-Apple hardware can't slip past — the CLI gate runs
    # on the submitter's machine and is bypassable by anyone willing to
    # edit JSON. Allowlist by pattern rather than enumeration so a
    # newly-released Apple chip ("Apple M5") doesn't need a code change
    # to be acceptable.
    if not APPLE_CHIP_RE.match(chip):
        raise _IssueError(
            f"hardware: chip {chip!r} does not match the Apple Silicon "
            f"pattern 'Apple M<n>[ Pro|Max|Ultra]'. The community DB is "
            f"Apple-Silicon-only by contract."
        )
    if not (1 <= hw.get("ram_gb", 0) <= MAX_RAM_GB):
        raise _IssueError(f"hardware: ram_gb out of range: {hw.get('ram_gb')}")

    peak = payload.get("peak_ram_mb")
    if peak is not None:
        # 1 GB = 1024 MiB. peak ≤ total RAM (some slack for shared GPU).
        if peak > hw["ram_gb"] * 1024 * 2:
            raise _IssueError(
                f"hardware: peak_ram_mb={peak} exceeds 2× total RAM ({hw['ram_gb']} GB)"
            )

    for bucket_name in ("short", "long"):
        b = payload["buckets"][bucket_name]
        # ``tps > 0`` is the documented contract; the previous check
        # only rejected negatives, so a maliciously crafted file with
        # ``decode_tps=0`` would slip through. (Codex PR #582 round-2
        # BLOCKING.) Zero TTFT is OK (legitimate floor when prefill is
        # cached); zero throughput is not.
        for tps_field in ("decode_tps", "prefill_tps"):
            stat = b[tps_field]
            if stat["median"] <= 0:
                raise _IssueError(
                    f"buckets.{bucket_name}.{tps_field}: median "
                    f"{stat['median']} must be > 0 (per the validator "
                    f"contract — zero throughput indicates a failed run)"
                )
        if b["ttft_ms"]["median"] < 0:
            raise _IssueError(f"buckets.{bucket_name}.ttft_ms: median < 0")
        if b["decode_tps"]["median"] > MAX_DECODE_TPS:
            raise _IssueError(
                f"buckets.{bucket_name}.decode_tps: median "
                f"{b['decode_tps']['median']:.1f} > {MAX_DECODE_TPS} (unrealistic)"
            )
        if b["prefill_tps"]["median"] > MAX_PREFILL_TPS:
            raise _IssueError(
                f"buckets.{bucket_name}.prefill_tps: median "
                f"{b['prefill_tps']['median']:.1f} > {MAX_PREFILL_TPS}"
            )
        if b["ttft_ms"]["median"] > MAX_TTFT_MS:
            raise _IssueError(
                f"buckets.{bucket_name}.ttft_ms: median "
                f"{b['ttft_ms']['median']:.1f} > {MAX_TTFT_MS} ms (likely a stuck request)"
            )
        # Apply the same bounds to every raw round, not just the
        # summary median. Without per-round checks, ``rounds_raw`` can
        # carry zero/negative throughput or 60-second TTFTs as long as
        # the median lands plausible — and the summary recomputation
        # check on the next line would still pass because the bogus
        # rounds DO produce the bogus median. (Codex PR #582 round-7
        # BLOCKING.) Validating each row directly closes that gap.
        _check_rounds_raw(bucket_name, b)
        # Recompute every summary stat from ``rounds_raw`` and refuse
        # the file if the precomputed value disagrees. Without this
        # check, a contributor could ship arbitrary median numbers
        # alongside a plausible-looking ``rounds_raw`` and the
        # aggregator (which trusts ``median``) would publish the lie.
        # (Codex PR #582 round-2 BLOCKING.)
        _check_summary_matches_rounds(bucket_name, b)


def _check_rounds_raw(bucket_name: str, bucket: dict) -> None:
    """Per-round sanity bounds — every entry must individually pass.

    The summary-median checks only look at the median, so a payload
    can park 4 round values inside the realistic band and bury an
    arbitrary 5th (negative, zero, 1e9) and the median is unaffected.
    Per-round checks here force every single value into the same
    realistic envelope, matching what the runner actually produces.
    """
    rounds = bucket.get("rounds_raw", [])
    for i, r in enumerate(rounds):
        for tps_field in ("decode_tps", "prefill_tps"):
            v = r.get(tps_field)
            if v is None or v <= 0:
                raise _IssueError(
                    f"buckets.{bucket_name}.rounds_raw[{i}].{tps_field}: "
                    f"{v!r} must be > 0"
                )
        max_tps = {"decode_tps": MAX_DECODE_TPS, "prefill_tps": MAX_PREFILL_TPS}
        for tps_field, ceiling in max_tps.items():
            if r[tps_field] > ceiling:
                raise _IssueError(
                    f"buckets.{bucket_name}.rounds_raw[{i}].{tps_field}: "
                    f"{r[tps_field]:.1f} > {ceiling} (unrealistic)"
                )
        ttft = r.get("ttft_ms")
        if ttft is None or ttft < 0 or ttft > MAX_TTFT_MS:
            raise _IssueError(
                f"buckets.{bucket_name}.rounds_raw[{i}].ttft_ms: "
                f"{ttft!r} out of range [0, {MAX_TTFT_MS}]"
            )


def _check_summary_matches_rounds(bucket_name: str, bucket: dict) -> None:
    """Recompute median/min/max/stddev from ``rounds_raw`` and verify.

    Tolerance is loose (1e-6 absolute, 1e-3 relative) — float
    serialization round-trips through JSON aren't bit-exact and we
    don't want to reject submissions over a 1 ULP drift. But anything
    larger means the summary was tampered with or hand-edited.
    """
    import statistics

    rounds = bucket.get("rounds_raw", [])
    if len(rounds) != 5:
        # Schema enforces exactly 5 — but if we got here without
        # jsonschema (older Python or skipped install), guard anyway.
        return
    for metric in ("decode_tps", "prefill_tps", "ttft_ms"):
        values = [r[metric] for r in rounds]
        sorted_v = sorted(values)
        computed = {
            "median": float(statistics.median(values)),
            "min": float(sorted_v[0]),
            "max": float(sorted_v[-1]),
            "stddev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }
        claimed = bucket[metric]
        for k, want in computed.items():
            got = float(claimed[k])
            tol = max(1e-6, abs(want) * 1e-3)
            if abs(got - want) > tol:
                raise _IssueError(
                    f"buckets.{bucket_name}.{metric}.{k}: claimed {got} "
                    f"but rounds_raw computes to {want:.6f} (Δ={abs(got - want):.6f}, "
                    f"tol={tol:.6f}). Summary stats must be derivable "
                    f"from rounds_raw — re-run the bench or recompute "
                    f"from the raw column."
                )


def _check_filename(path: Path) -> None:
    name = path.name
    if not FILENAME_RE.match(name):
        raise _IssueError(
            f"filename: '{name}' does not match "
            f"<YYYYMMDD>-<chip-slug>-<alias-slug>-<12hex>.json"
        )


def _check_path_in_submissions(path: Path) -> None:
    """Refuse files that aren't inside ``community-benchmarks/submissions/``.

    PRs editing other files in the same diff are fine — they just
    don't get fed to us. We're called explicitly with each submission
    file path, but a buggy GHA filter could feed us something else;
    this is the cheap belt-and-braces guard.

    The check looks at the file path's own ancestry — ``.../community-benchmarks/submissions/<file>`` —
    rather than comparing against this script's ``SUBMISSIONS_DIR``. The
    GHA trust-gate copies a frozen base validator to ``/tmp/base-validator/``
    and runs it against files in ``/home/runner/work/...``; comparing
    against the validator's own ``REPO_ROOT`` would always fail there
    even though the path is structurally correct.
    """
    resolved = path.resolve()
    parent = resolved.parent
    grandparent = parent.parent if parent != parent.parent else parent
    if parent.name != "submissions" or grandparent.name != "community-benchmarks":
        raise _IssueError(
            f"path: {path} is not inside community-benchmarks/submissions/"
        )


def _check_no_duplicate_submission_id(
    path: Path, payload: dict, existing_ids: set[str]
) -> None:
    """Refuse a submission whose ``submission_id`` already exists.

    ``submission_id`` is generated locally as the first 12 hex of a
    uuid4 — naturally unique. A duplicate means the contributor
    copied an existing file and renamed it (intentionally or not),
    which would inflate the contributing machine's vote in the
    aggregator. (Codex PR #582 round-3 BLOCKING.)

    Note: this only fires when ``existing_ids`` was passed in by the
    caller; single-file validation (CI on a PR diff) populates it
    from the full submissions/ corpus, so the new file is compared
    against history. The aggregator also has its own de-dup pass as
    defense in depth.
    """
    sid = payload.get("submission_id")
    if sid is None:
        return
    if sid in existing_ids:
        raise _IssueError(
            f"submission_id: {sid} already exists in the corpus. "
            f"Each submission must have a unique uuid4-derived id; "
            f"copying an existing file under a new name is not how "
            f"to add a second sample (re-run the bench instead)."
        )


def _read_submission_id(path: Path) -> str | None:
    """Read just the ``submission_id`` from a file, or ``None`` on parse error.

    Cheap helper for the up-front corpus-id scan: we don't need to
    re-validate to know which id to subtract from the "is this a
    duplicate?" check.
    """
    try:
        return json.loads(path.read_text()).get("submission_id")
    except (OSError, json.JSONDecodeError):
        return None


def _load_submission_id_index(
    submissions_dir: Path | None = None,
) -> dict[str, set[Path]]:
    """Walk submissions/ and return ``submission_id -> set of paths``.

    Returning paths (not just ids) lets the caller subtract the target
    file's own path when validating, so we still detect a true
    duplicate (two different files sharing one id) without false-
    flagging a file as a duplicate of itself.

    ``submissions_dir`` is an explicit override for the corpus location.
    Without it we fall back to the module-level ``SUBMISSIONS_DIR``,
    which is derived from the validator script's own path. The GHA
    trust-gate relocates ``validate.py`` to ``/tmp/base-validator/``,
    where ``SUBMISSIONS_DIR`` points to an empty/non-existent directory
    — meaning the duplicate-id gate silently no-op'd on the trusted
    pass. Letting the caller derive the corpus from the target file's
    own parent (which the structural ``_check_path_in_submissions``
    guard already validated as a real ``community-benchmarks/submissions/``
    folder) closes that hole. (Codex PR #587 BLOCKING.)
    """
    index: dict[str, set[Path]] = {}
    corpus = submissions_dir if submissions_dir is not None else SUBMISSIONS_DIR
    if not corpus.exists():
        return index
    for existing in corpus.glob("*.json"):
        try:
            payload = json.loads(existing.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sid = payload.get("submission_id")
        if sid is not None:
            index.setdefault(sid, set()).add(existing.resolve())
    return index


def _ids_with_other_owners(
    index: dict[str, set[Path]],
    target: Path,
    target_sid: str | None,
    in_run: set[str],
) -> set[str]:
    """Compose the "id is already used elsewhere" set for one target.

    O(1) per target: we only need to know whether *the target's own
    ``submission_id``* is also carried by another file. The earlier
    version scanned every entry of ``index`` to populate a set that
    was then membership-tested for one value, making the validator
    O(targets × corpus) — which Codex round-7 flagged as the round-2
    fix not actually fixing the asymptote. (PR #582 round-7 BLOCKING.)
    By only checking ``index.get(target_sid)`` and ``in_run``, the
    per-target cost is constant.
    """
    out: set[str] = set(in_run)
    if target_sid is None:
        return out
    owners = index.get(target_sid)
    if owners is None:
        return out
    target_resolved = target.resolve()
    if any(p != target_resolved for p in owners):
        out.add(target_sid)
    return out


def validate_one(
    path: Path,
    schema: dict | None,
    aliases: dict[str, dict],
    existing_ids: set[str] | None = None,
) -> list[str]:
    """Return the list of issues found for one file. Empty = OK."""
    issues: list[str] = []

    try:
        # Resolve symlinks ONCE and reuse for both the path-shape check
        # and the file read. The previous version called
        # ``_check_path_in_submissions(path.resolve())`` but then
        # ``path.read_text()`` on the original (unresolved) path, leaving
        # a symlink/TOCTOU gap: a symlink under submissions/ pointing
        # outside the directory could pass the location check on its
        # target but be read from the symlink source. (Codex PR #582
        # round-7 BLOCKING.) ``path.resolve()`` is non-strict to avoid
        # a confusing "file not found" raise from inside resolve when
        # the caller passed a typo'd path; the explicit ``is_file()``
        # check below produces a cleaner error message in that case.
        resolved = path.resolve()
        # Reject symlinks outright. Allowing them lets a contributor add
        # ``submissions/twin.json`` pointing at an existing file, which
        # the dedup check (keyed on resolved paths) would collapse into
        # the original — so the symlink "submission" would not be
        # validated as a fresh row but would still be merged into the
        # corpus, inflating a single machine's contribution. (Codex
        # round-7 BLOCKING, validate.py v2.)
        if path.is_symlink():
            raise _IssueError(
                f"path: {path} is a symlink; community submissions must "
                f"be regular files (a symlink could collapse dedup checks "
                f"and inflate a contributor's row count)."
            )
        if not resolved.is_file():
            raise _IssueError(f"path: {path} is not a regular file")
        _check_path_in_submissions(resolved)
        _check_filename(path)
        text = resolved.read_text(encoding="utf-8")
        # ``parse_constant`` hooks ``NaN`` / ``Infinity`` / ``-Infinity``
        # so the comparisons in ``_check_sanity`` can't be bypassed by
        # a hand-edited file with a non-finite throughput. Defensive
        # recursive scan after parse catches any non-finite that
        # slipped past the parser (e.g. via a numeric literal that
        # decoded to ``inf`` through float overflow). (Codex PR #582
        # round-7 BLOCKING.)
        payload = json.loads(text, parse_constant=_reject_non_finite)
        if _has_non_finite(payload):
            raise _IssueError(
                "json: payload contains a non-finite number; only finite "
                "floats are accepted (NaN/Infinity bypass sanity checks)."
            )
        _check_schema(payload, schema)
        _check_alias_whitelist(payload, aliases)
        _check_sanity(payload)
        if existing_ids is not None:
            _check_no_duplicate_submission_id(path, payload, existing_ids)
    except _IssueError as e:
        issues.append(str(e))
    except json.JSONDecodeError as e:
        issues.append(f"json: parse error: {e}")
    except (OSError, KeyError) as e:
        # KeyError surfaces "schema passed but we then asked for a field
        # the schema doesn't require" — that's a validator bug, not user
        # error, but it shouldn't silently pass either.
        issues.append(f"internal: {type(e).__name__}: {e}")
    return issues


def main(argv: list[str]) -> int:
    targets = (
        [Path(p) for p in argv[1:]]
        if len(argv) > 1
        else sorted(SUBMISSIONS_DIR.glob("*.json"))
    )
    if not targets:
        print("  No submission files to validate.")
        return 0

    schema = _load_schema()
    aliases = _load_aliases()
    if not aliases:
        print("  ERROR: aliases.json is empty or missing — every file will fail.")
        return min(125, len(targets))

    # Cross-file uniqueness check: build the id→paths index ONCE
    # up-front, then for each target derive the "ids owned by some
    # OTHER file" set rather than rescanning the entire submissions/
    # directory per target. The previous version was O(n²) over the
    # corpus. (Codex PR #582 round-5 NIT.) Tracking by path (not just
    # id) so two files with the same id are correctly flagged: removing
    # the target's id from a plain set would mask both.
    #
    # Derive the corpus from the first target's parent rather than the
    # validator-location-derived ``SUBMISSIONS_DIR``: the
    # ``_check_path_in_submissions`` guard already validated that the
    # target lives in a real ``community-benchmarks/submissions/``
    # folder, so its parent is the authoritative corpus location. This
    # makes the duplicate-id gate work both for local CLI invocations
    # (where validator and corpus share a repo) and for the GHA trust
    # gate's relocated-validator setup (where they don't). (Codex PR
    # #587 BLOCKING.)
    corpus_dir = targets[0].resolve().parent if targets else None
    id_index = _load_submission_id_index(submissions_dir=corpus_dir)
    seen_in_run: set[str] = set()
    failures = 0
    for path in targets:
        target_sid = _read_submission_id(path)
        existing = _ids_with_other_owners(id_index, path, target_sid, seen_in_run)
        issues = validate_one(path, schema, aliases, existing_ids=existing)
        if issues:
            failures += 1
            print(f"  FAIL  {path.name}")
            for issue in issues:
                print(f"        {issue}")
        else:
            print(f"  OK    {path.name}")
            sid_self = _read_submission_id(path)
            if sid_self:
                # Track passes so a second ADDED file in the same PR
                # with the same id (within this run, not yet on disk
                # in the merge-base) is flagged.
                seen_in_run.add(sid_self)

    print()
    print(f"  {len(targets) - failures}/{len(targets)} files passed.")
    return min(125, failures)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
