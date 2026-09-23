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
import hashlib
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
from datetime import datetime, timezone
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
SNAPSHOT_PATH = ROOT / "scripts" / "mirror_drift_snapshot.json"
MAX_WORKERS = 8
SMALL_NON_LFS_MAX_BYTES = 1024 * 1024
HF_MIN_INTERVAL_SECONDS = 1.0
_USER_AGENT = "rapid-mlx mirror-drift-auditor"
_SEVERITY = {"info": 10, "warning": 20, "error": 30, "never": 10_000}
_HF_GATE_LOCK = threading.Lock()
_hf_next_request = 0.0
_COUNT_LOCK = threading.Lock()
_request_counts = {"hf": 0, "mirror": 0}


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
    oid: str | None = None


@dataclass(frozen=True)
class HfRepo:
    revision: str | None
    files: list[HfFile]


@dataclass(frozen=True)
class MirrorProbe:
    status: int
    size: int | None
    etag: str | None
    blob_oid: str | None


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
            redirected = urllib.request.Request(
                redirected.full_url,
                headers=dict(redirected.headers),
                origin_req_host=redirected.origin_req_host,
                unverifiable=redirected.unverifiable,
                method=req.get_method(),
            )
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


def _model_info(repo_id: str) -> Any:
    from huggingface_hub import model_info

    return model_info(repo_id, files_metadata=True)


def _hf_repo(repo_id: str) -> HfRepo:
    """Fetch one complete HF repository description, with paced retries."""
    from huggingface_hub.errors import HfHubHTTPError

    last: BaseException | None = None
    for attempt in range(5):
        _throttle_hf()
        with _COUNT_LOCK:
            _request_counts["hf"] += 1
        delay = float(2**attempt)
        try:
            info = _model_info(repo_id)
            break
        except HfHubHTTPError as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", 0)
            if status < 500 and status != 429:
                raise
            last = error
            retry_after = response.headers.get("Retry-After") if response else None
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(float(retry_after), 30.0))
        except (OSError, TimeoutError) as error:
            last = error
        if attempt < 4:
            time.sleep(delay)
    else:
        assert last is not None
        raise last

    siblings = getattr(info, "siblings", None)
    if not isinstance(siblings, list):
        raise RuntimeError(f"Hugging Face returned an invalid listing for {repo_id}")
    files: list[HfFile] = []
    for sibling in siblings:
        path = getattr(sibling, "rfilename", None)
        if (
            not isinstance(path, str)
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            continue
        size = getattr(sibling, "size", None)
        lfs = getattr(sibling, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        oid = getattr(sibling, "blob_id", None)
        files.append(
            HfFile(
                path=path,
                size=size if isinstance(size, int) else None,
                sha256=sha if isinstance(sha, str) and len(sha) == 64 else None,
                oid=oid if isinstance(oid, str) else None,
            )
        )
    revision = getattr(info, "sha", None)
    return HfRepo(revision if isinstance(revision, str) else None, files)


def _hf_files(repo_id: str) -> list[HfFile]:
    """Compatibility helper used by focused callers and tests."""
    return _hf_repo(repo_id).files


def _throttle_hf() -> None:
    """Globally pace anonymous HF listings while mirror HEADs remain parallel."""
    global _hf_next_request
    with _HF_GATE_LOCK:
        now = time.monotonic()
        delay = max(0.0, _hf_next_request - now)
        if delay:
            time.sleep(delay)
        _hf_next_request = max(now, _hf_next_request) + HF_MIN_INTERVAL_SECONDS


def _selected_files(files: list[HfFile], subfolder: str | None) -> list[HfFile]:
    selected = subfolder.strip("/") if subfolder else None
    return [
        item
        for item in files
        if selected is None or item.path.startswith(f"{selected}/")
    ]


def _optional_asset(path: str) -> bool:
    """Match documentation/metadata assets that the downloader may HF-fallback."""
    name = Path(path).name.lower()
    return (
        name == ".gitattributes"
        or name.startswith(("readme", "license", "notice", "citation", "authors"))
        or Path(name).suffix in {".md", ".rst", ".png", ".jpg", ".jpeg", ".gif", ".svg"}
    )


def _required_files(files: list[HfFile], subfolder: str | None) -> list[HfFile]:
    """Return runtime-required files after the downloader's selection contract."""
    return [
        item
        for item in _selected_files(files, subfolder)
        if not _optional_asset(item.path)
    ]


def _mirror_url(repo_id: str, path: str) -> str:
    key = f"{repo_id}/{path}"
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in key.split("/"))
    return f"{MIRROR_BASE}/{encoded}"


def _git_blob_oid(body: bytes) -> str:
    prefix = f"blob {len(body)}\0".encode()
    return hashlib.sha1(prefix + body, usedforsecurity=False).hexdigest()


def _public_probe(repo_id: str, item: HfFile) -> MirrorProbe:
    fetch_body = (
        item.sha256 is None
        and item.size is not None
        and item.size <= SMALL_NON_LFS_MAX_BYTES
    )
    with _COUNT_LOCK:
        _request_counts["mirror"] += 1
    response = _request(
        _mirror_url(repo_id, item.path), method="GET" if fetch_body else "HEAD"
    )
    with response:
        status = int(getattr(response, "status", response.getcode()))
        raw_size = response.headers.get("Content-Length")
        size = int(raw_size) if raw_size and raw_size.isdigit() else None
        body = (
            response.read(SMALL_NON_LFS_MAX_BYTES + 1)
            if fetch_body and 200 <= status < 300
            else None
        )
        if body is not None and size is None:
            size = len(body)
        return MirrorProbe(
            status=status,
            size=size,
            etag=response.headers.get("ETag"),
            blob_oid=_git_blob_oid(body) if body is not None else None,
        )


def _public_head(repo_id: str, item: HfFile) -> tuple[int, int | None]:
    """Backward-compatible status/size view for callers that only need HEAD data."""
    probe = _public_probe(repo_id, item)
    return probe.status, probe.size


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


def _snapshot_aliases(path: Path = SNAPSHOT_PATH) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    aliases = payload.get("aliases") if isinstance(payload, dict) else None
    if not isinstance(aliases, dict):
        return {}
    return {key: value for key, value in aliases.items() if isinstance(value, dict)}


def _snapshot_matches(
    alias: str, entry: dict[str, Any] | None, snapshot: dict[str, dict[str, Any]]
) -> bool:
    saved = snapshot.get(alias.lower())
    if entry is None or saved is None:
        return False
    fields = ("hf_path", "status", "total_bytes", "file_count", "latest_uploaded")
    return all(saved.get(field) == entry.get(field) for field in fields)


def _new_report(
    spec: AliasSpec, entry: dict[str, Any] | None, has_r2: bool
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
        sha_check="checked" if has_r2 else "skipped_no_credentials",
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

    return report


def _recent_timestamp(value: Any, now: datetime) -> bool:
    if isinstance(value, (int, float)):
        stamp = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
    else:
        return False
    age = (now - stamp.astimezone(timezone.utc)).total_seconds()
    return -300 <= age <= 3600


def _sync_in_progress(
    entry: dict[str, Any] | None, now: datetime | None = None
) -> bool:
    if not entry:
        return False
    if str(entry.get("status", "")).lower() == "partial":
        return True
    return _recent_timestamp(
        entry.get("latest_uploaded"), now or datetime.now(timezone.utc)
    )


def _file_finding(
    kind: str,
    item: HfFile,
    detail: str,
    *,
    sync_in_progress: bool,
) -> Finding:
    if _optional_asset(item.path):
        severity = "info"
    elif sync_in_progress:
        severity = "warning"
        detail = f"{detail}; sync in progress"
    else:
        severity = "error"
    return Finding(kind, severity, item.path, detail)


def _etag_sha256(etag: str | None) -> str | None:
    if not etag:
        return None
    value = etag.removeprefix("W/").strip('"').lower()
    return (
        value
        if len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        else None
    )


def _probe_with_metadata(
    repo_id: str, item: HfFile, r2_client: Any | None
) -> tuple[MirrorProbe, dict[str, str] | None]:
    probe = _public_probe(repo_id, item)
    metadata = (
        _r2_metadata(r2_client, repo_id, item.path)
        if r2_client is not None and item.sha256 is not None
        else None
    )
    return probe, metadata


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

    with _COUNT_LOCK:
        _request_counts.update(hf=0, mirror=0)

    entries = _catalog_entries()
    by_alias = {
        str(entry["alias"]).lower(): entry for entry in entries if entry.get("alias")
    }
    snapshot = _snapshot_aliases()
    r2_client = _maybe_r2_client()
    pool_size = max(1, min(workers, MAX_WORKERS))
    specs_by_repo: dict[str, list[AliasSpec]] = {}
    for spec in selected:
        specs_by_repo.setdefault(spec.hf_path, []).append(spec)
    skip_repo_probes = {
        repo_id
        for repo_id, repo_specs in specs_by_repo.items()
        if all(
            _snapshot_matches(spec.alias, by_alias.get(spec.alias.lower()), snapshot)
            and not _sync_in_progress(by_alias.get(spec.alias.lower()))
            for spec in repo_specs
        )
    }

    # One paced model_info call per unique repository, fanned out to every alias.
    repos: dict[str, HfRepo] = {}
    repo_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=pool_size) as hf_pool:
        hf_futures = {
            hf_pool.submit(_hf_repo, repo_id): repo_id
            for repo_id in {spec.hf_path for spec in selected}
        }
        for hf_future in as_completed(hf_futures):
            repo_id = hf_futures[hf_future]
            try:
                repos[repo_id] = hf_future.result()
            except Exception as error:
                repo_errors[repo_id] = str(error).splitlines()[0]

    reports = []
    report_context: list[
        tuple[AliasReport, AliasSpec, dict[str, Any] | None, list[HfFile], bool]
    ] = []
    probes: dict[tuple[str, str], HfFile] = {}
    for spec in selected:
        entry = by_alias.get(spec.alias.lower())
        report = _new_report(spec, entry, r2_client is not None)
        files: list[HfFile]
        if spec.hf_path in repo_errors:
            report.findings.append(
                Finding("hf_unavailable", "error", detail=repo_errors[spec.hf_path])
            )
            files = []
        elif spec.hf_path in skip_repo_probes:
            files = []
        else:
            files = _selected_files(repos[spec.hf_path].files, spec.subfolder)
        report.checked_files = len(files)
        in_progress = _sync_in_progress(entry)
        report_context.append((report, spec, entry, files, in_progress))
        reports.append(report)
        for item in files:
            probes.setdefault((spec.hf_path, item.path), item)

    # Mirror I/O is globally deduplicated and isolated in its own bounded pool.
    probe_results: dict[tuple[str, str], tuple[MirrorProbe, dict[str, str] | None]] = {}
    with ThreadPoolExecutor(max_workers=pool_size) as mirror_pool:
        mirror_futures = {
            mirror_pool.submit(_probe_with_metadata, repo_id, item, r2_client): key
            for key, item in probes.items()
            for repo_id in [key[0]]
        }
        for mirror_future in as_completed(mirror_futures):
            probe_results[mirror_futures[mirror_future]] = mirror_future.result()

    for report, spec, entry, files, in_progress in report_context:
        required = _required_files(files, None)
        missing_required = 0
        for item in files:
            probe, metadata = probe_results[(spec.hf_path, item.path)]
            if probe.status < 200 or probe.status >= 300:
                if not _optional_asset(item.path):
                    missing_required += 1
                report.findings.append(
                    _file_finding(
                        "missing_file",
                        item,
                        f"HTTP {probe.status}",
                        sync_in_progress=in_progress,
                    )
                )
                continue
            if item.size is not None and probe.size != item.size:
                report.findings.append(
                    _file_finding(
                        "size_mismatch",
                        item,
                        f"mirror={probe.size} hf={item.size} etag={probe.etag or 'missing'}",
                        sync_in_progress=in_progress,
                    )
                )
            if item.sha256 is None and item.oid and probe.blob_oid != item.oid:
                report.findings.append(
                    _file_finding(
                        "content_mismatch",
                        item,
                        f"mirror_blob={probe.blob_oid or 'missing'} hf_blob={item.oid}",
                        sync_in_progress=in_progress,
                    )
                )
            if item.sha256 is not None:
                public_sha = _etag_sha256(probe.etag)
                metadata_sha = (
                    metadata.get("hf-sha256") if metadata is not None else None
                )
                actual_sha = metadata_sha or public_sha
                if actual_sha is not None and actual_sha != item.sha256:
                    report.findings.append(
                        _file_finding(
                            "content_mismatch",
                            item,
                            f"mirror_sha256={actual_sha} hf_sha256={item.sha256}",
                            sync_in_progress=in_progress,
                        )
                    )
        if (
            entry
            and entry.get("status") == "mirrored"
            and required
            and missing_required == len(required)
        ):
            detail = "catalog says mirrored; no required file is reachable"
            severity = "warning" if in_progress else "error"
            if in_progress:
                detail += "; sync in progress"
            report.findings.append(Finding("false_mirrored", severity, detail=detail))

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
    lines.append(
        f"REQUESTS hf={_request_counts['hf']} mirror={_request_counts['mirror']}"
    )
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
                    "requests": dict(_request_counts),
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
