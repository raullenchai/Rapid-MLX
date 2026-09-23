#!/usr/bin/env python3
"""Audit shipped aliases against Hugging Face and the public model mirror.

Rapid-MLX ships two alias catalogs with different schemas:
``rapid_mlx/aliases.json`` maps text/vision/image/video aliases via ``hf_path``;
``rapid_mlx/audio/aliases.json`` maps STT/TTS aliases via ``hf_id``. This
read-only, credential-free auditor always loads both and resolves repository
ids from those files rather than guessing them from alias names.

The allow-list mirrors the client's selection contract in ``rapid_mlx/_mirror.py``:
``.gitattributes`` is not needed at runtime, and an alias with ``subfolder``
causes the client to pass ``allow_patterns=["<subfolder>/*"]``. Consequently,
non-selected quantisation folders such as ``5bit/``, ``6bit/`` and ``8bit/``
must not be reported missing when the selected/default quant is elsewhere.

Public requests are cache-busted after every redirect and carry
``Cache-Control: no-cache``. This is essential because the CDN has served an
old body and a cached 404 for up to its advertised ``max-age=3600``; the first
``models.rapidmlx.com`` redirect also drops the incoming query string.

When boto3 and R2 credentials happen to be available, object metadata is also
compared with Hugging Face's LFS SHA-256. Without credentials that optional
check is reported as skipped; the normal scheduled audit needs no secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CATALOG_URL = "https://models.rapidmlx.com/api/models"
HF_API_BASE = "https://huggingface.co/api/models"
MIRROR_BASE = "https://models.rapidmlx.com"
R2_ENDPOINT_URL = "https://f25478810829faf5ccc86f4ed9a96ef1.r2.cloudflarestorage.com"
R2_BUCKET = "rapid-mlx-models"
ROOT = Path(__file__).resolve().parents[1]
ALIASES_PATH = ROOT / "rapid_mlx" / "aliases.json"
AUDIO_ALIASES_PATH = ROOT / "rapid_mlx" / "audio" / "aliases.json"
MAX_WORKERS = 8
HF_MIN_INTERVAL_SECONDS = 1.0
_USER_AGENT = "rapid-mlx mirror-drift-auditor"
_SEVERITY = {"info": 10, "warning": 20, "error": 30, "never": 10_000}
_HF_GATE_LOCK = threading.Lock()
_hf_next_request = 0.0


@dataclass(frozen=True)
class AliasSpec:
    alias: str
    hf_path: str
    subfolder: str | None
    source: str


@dataclass(frozen=True)
class HfFile:
    path: str
    size: int | None
    sha256: str | None


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: str
    path: str | None = None
    detail: str | None = None


@dataclass
class AliasReport:
    alias: str
    source: str
    hf_path: str
    catalog_present: bool
    catalog_hf_path: str | None
    catalog_status: str | None
    checked_files: int = 0
    sha_check: str = "skipped_no_credentials"
    findings: list[Finding] = field(default_factory=list)

    @property
    def state(self) -> str:
        return "ok" if not self.findings else "findings"


def _cache_busted(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.append(("mirror_drift", uuid.uuid4().hex))
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urllib.parse.urlencode(query),
            parts.fragment,
        )
    )


class _FinalUrlRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        # Incident 2026-09-23: the first CDN redirect dropped its query. Add a
        # fresh cache-buster to *each destination*, including the final URL.
        redirected = super().redirect_request(
            req, fp, code, msg, headers, _cache_busted(newurl)
        )
        if redirected is not None:
            redirected.add_header("Cache-Control", "no-cache")
        return redirected


_OPENER = urllib.request.build_opener(_FinalUrlRedirectHandler())


def _request(url: str, *, method: str = "GET", timeout: float = 30.0) -> Any:
    """Make a polite cache-busted request, retrying transient failures."""
    last: BaseException | None = None
    for attempt in range(5):
        delay = float(2**attempt)
        request = urllib.request.Request(
            _cache_busted(url),
            method=method,
            headers={"Cache-Control": "no-cache", "User-Agent": _USER_AGENT},
        )
        try:
            return _OPENER.open(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                return error
            last = error
            retry_after = error.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(float(retry_after), 30.0))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = error
        if attempt < 4:
            time.sleep(delay)
    assert last is not None
    raise last


def _get_json(url: str) -> Any:
    response = _request(url)
    with response:
        status = int(getattr(response, "status", response.getcode()))
        if status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {status}")
        return json.loads(response.read())


def _load_aliases(main_path: Path, audio_path: Path) -> list[AliasSpec]:
    specs: list[AliasSpec] = []
    for path, key, source in (
        (main_path, "hf_path", "main"),
        (audio_path, "hf_id", "audio"),
    ):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        for alias, profile in payload.items():
            if not isinstance(profile, dict):
                continue
            repo_id = profile.get(key)
            if not isinstance(repo_id, str) or not _valid_repo_id(repo_id):
                continue
            subfolder = profile.get("subfolder")
            specs.append(
                AliasSpec(
                    alias=alias,
                    hf_path=repo_id,
                    subfolder=subfolder if isinstance(subfolder, str) else None,
                    source=source,
                )
            )
    return specs


def _valid_repo_id(repo_id: str) -> bool:
    owner, separator, name = repo_id.partition("/")
    return bool(owner and separator and name and "/" not in name)


def _hf_files(repo_id: str) -> list[HfFile]:
    _throttle_hf()
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    payload = _get_json(f"{HF_API_BASE}/{encoded}?blobs=true")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Hugging Face returned an invalid listing for {repo_id}")
    files: list[HfFile] = []
    for sibling in payload.get("siblings", []):
        if not isinstance(sibling, dict):
            continue
        path = sibling.get("rfilename")
        if (
            not isinstance(path, str)
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            continue
        size = sibling.get("size")
        lfs = sibling.get("lfs")
        sha = lfs.get("sha256") if isinstance(lfs, dict) else None
        files.append(
            HfFile(
                path=path,
                size=size if isinstance(size, int) else None,
                sha256=sha if isinstance(sha, str) and len(sha) == 64 else None,
            )
        )
    return files


def _throttle_hf() -> None:
    """Globally pace anonymous HF listings while mirror HEADs remain parallel."""
    global _hf_next_request
    with _HF_GATE_LOCK:
        now = time.monotonic()
        delay = max(0.0, _hf_next_request - now)
        if delay:
            time.sleep(delay)
        _hf_next_request = max(now, _hf_next_request) + HF_MIN_INTERVAL_SECONDS


def _required_files(files: list[HfFile], subfolder: str | None) -> list[HfFile]:
    """Apply the runtime client's allow-pattern and ignorable-file contract."""
    selected = subfolder.strip("/") if subfolder else None
    return [
        item
        for item in files
        if item.path != ".gitattributes"
        and (selected is None or item.path.startswith(f"{selected}/"))
    ]


def _mirror_url(repo_id: str, path: str) -> str:
    key = f"{repo_id}/{path}"
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in key.split("/"))
    return f"{MIRROR_BASE}/{encoded}"


def _public_head(repo_id: str, item: HfFile) -> tuple[int, int | None]:
    response = _request(_mirror_url(repo_id, item.path), method="HEAD")
    with response:
        status = int(getattr(response, "status", response.getcode()))
        raw_size = response.headers.get("Content-Length")
        return status, int(raw_size) if raw_size and raw_size.isdigit() else None


def _maybe_r2_client() -> Any | None:
    try:
        import boto3
    except ImportError:
        return None
    session = boto3.Session()
    if session.get_credentials() is None:
        return None
    return session.client(
        "s3",
        endpoint_url=os.environ.get("RAPID_MLX_R2_ENDPOINT_URL", R2_ENDPOINT_URL),
    )


def _r2_metadata(client: Any, repo_id: str, path: str) -> dict[str, str] | None:
    try:
        response = client.head_object(Bucket=R2_BUCKET, Key=f"{repo_id}/{path}")
    except Exception as error:
        response_data = getattr(error, "response", {})
        code = str(response_data.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    metadata = response.get("Metadata")
    return metadata if isinstance(metadata, dict) else {}


def _catalog_entries() -> list[dict[str, Any]]:
    payload = _get_json(CATALOG_URL)
    models = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        raise RuntimeError("mirror catalog response has no models list")
    return [entry for entry in models if isinstance(entry, dict)]


def _audit_alias(
    spec: AliasSpec,
    entry: dict[str, Any] | None,
    r2_client: Any | None,
) -> AliasReport:
    catalog_hf_path = entry.get("hf_path") if entry else None
    catalog_hf_path = catalog_hf_path if isinstance(catalog_hf_path, str) else None
    catalog_status = entry.get("status") if entry else None
    report = AliasReport(
        alias=spec.alias,
        source=spec.source,
        hf_path=spec.hf_path,
        catalog_present=entry is not None,
        catalog_hf_path=catalog_hf_path,
        catalog_status=str(catalog_status) if catalog_status is not None else None,
        sha_check="checked" if r2_client is not None else "skipped_no_credentials",
    )
    if entry is None:
        report.findings.append(Finding("not_in_catalog", "error"))
    elif catalog_hf_path != spec.hf_path:
        report.findings.append(
            Finding(
                "hf_path_mismatch",
                "error",
                detail=f"catalog={catalog_hf_path or 'missing'} shipped={spec.hf_path}",
            )
        )

    files = _required_files(_hf_files(spec.hf_path), spec.subfolder)
    report.checked_files = len(files)
    missing = 0
    for item in files:
        status, mirror_size = _public_head(spec.hf_path, item)
        if status < 200 or status >= 300:
            missing += 1
            report.findings.append(
                Finding("missing_file", "error", item.path, f"HTTP {status}")
            )
            continue
        if item.size is not None and mirror_size != item.size:
            report.findings.append(
                Finding(
                    "size_mismatch",
                    "error",
                    item.path,
                    f"mirror={mirror_size} hf={item.size}",
                )
            )
        if r2_client is not None and item.sha256 is not None:
            metadata = _r2_metadata(r2_client, spec.hf_path, item.path)
            actual = metadata.get("hf-sha256") if metadata is not None else None
            if actual != item.sha256:
                report.findings.append(
                    Finding(
                        "sha_mismatch",
                        "error",
                        item.path,
                        f"r2={actual or 'missing'} hf={item.sha256}",
                    )
                )
    if entry and entry.get("status") == "mirrored" and files and missing == len(files):
        report.findings.append(
            Finding(
                "false_mirrored",
                "error",
                detail="catalog says mirrored; no file is reachable",
            )
        )
    return report


def audit(
    main_aliases_path: Path = ALIASES_PATH,
    audio_aliases_path: Path = AUDIO_ALIASES_PATH,
    *,
    aliases: set[str] | None = None,
    only_used: bool = False,
    workers: int = MAX_WORKERS,
) -> list[AliasReport]:
    """Audit aliases; ``only_used`` omits bucket-only catalog inventory rows."""
    specs = _load_aliases(main_aliases_path, audio_aliases_path)
    selected = [spec for spec in specs if aliases is None or spec.alias in aliases]
    if aliases is not None:
        unknown = aliases - {spec.alias for spec in selected}
        if unknown:
            raise ValueError(f"unknown alias(es): {', '.join(sorted(unknown))}")

    entries = _catalog_entries()
    by_alias = {
        str(entry["alias"]).lower(): entry for entry in entries if entry.get("alias")
    }
    r2_client = _maybe_r2_client()
    reports: list[AliasReport] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futures = {
            pool.submit(
                _audit_alias, spec, by_alias.get(spec.alias.lower()), r2_client
            ): spec
            for spec in selected
        }
        for future in as_completed(futures):
            reports.append(future.result())

    if not only_used and aliases is None:
        shipped_aliases = {spec.alias.lower() for spec in specs}
        for entry in entries:
            alias = str(entry.get("alias", ""))
            repo_id = str(entry.get("hf_path", ""))
            if alias and repo_id and alias.lower() not in shipped_aliases:
                reports.append(
                    AliasReport(
                        alias=alias,
                        source="catalog_only",
                        hf_path=repo_id,
                        catalog_present=True,
                        catalog_hf_path=repo_id,
                        catalog_status=str(entry.get("status"))
                        if entry.get("status")
                        else None,
                        sha_check="not_applicable",
                    )
                )
    return sorted(reports, key=lambda report: (report.source, report.alias))


def _fails(reports: list[AliasReport], fail_on: str) -> bool:
    threshold = _SEVERITY[fail_on]
    return any(
        _SEVERITY[finding.severity] >= threshold
        for report in reports
        for finding in report.findings
    )


def _summary_counts(reports: list[AliasReport]) -> dict[str, int]:
    counts: dict[str, int] = {"ok": 0}
    for report in reports:
        if not report.findings:
            counts["ok"] += 1
        for finding in report.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
    return counts


def _render_text(reports: list[AliasReport]) -> str:
    headings = ("ALIAS", "SOURCE", "CATALOG", "FILES", "STATE")
    rows: list[tuple[str, str, str, str, str]] = []
    for report in reports:
        catalog = report.catalog_status if report.catalog_present else "ABSENT"
        rows.append(
            (
                report.alias,
                report.source,
                catalog or "unknown",
                str(report.checked_files),
                report.state,
            )
        )
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    ]
    lines = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(headings))
    ]
    lines.append("  ".join("-" * width for width in widths))
    for report, row in zip(reports, rows, strict=True):
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
        for finding in report.findings:
            where = f" {finding.path}" if finding.path else ""
            detail = f" ({finding.detail})" if finding.detail else ""
            lines.append(f"  {finding.severity.upper()} {finding.kind}{where}{detail}")
    counts = _summary_counts(reports)
    summary = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    lines.append(f"SUMMARY aliases={len(reports)} {summary}")
    return "\n".join(lines)


def _report_dict(report: AliasReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["state"] = report.state
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alias", action="append", default=[], help="audit one alias; repeatable"
    )
    parser.add_argument(
        "--only-used",
        action="store_true",
        help="show only aliases used by shipped alias files (omit catalog-only inventory)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument(
        "--fail-on",
        choices=tuple(_SEVERITY),
        default="error",
        help="lowest severity that makes the command fail (default: error)",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS, help="concurrency, capped at 8"
    )
    parser.add_argument(
        "--aliases-path", type=Path, default=ALIASES_PATH, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--audio-aliases-path",
        type=Path,
        default=AUDIO_ALIASES_PATH,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        reports = audit(
            args.aliases_path,
            args.audio_aliases_path,
            aliases=set(args.alias) or None,
            only_used=args.only_used,
            workers=args.workers,
        )
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"mirror drift audit failed: {error}", file=sys.stderr)
        return 2
    failed = _fails(reports, args.fail_on)
    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": not failed,
                    "fail_on": args.fail_on,
                    "summary": _summary_counts(reports),
                    "aliases": [_report_dict(report) for report in reports],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_render_text(reports))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
