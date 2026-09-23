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
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
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
MAX_WORKERS = 32
HF_MAX_WORKERS = 8
DEFAULT_DEADLINE_SECONDS = 1500.0
MAX_RETRY_AFTER_SECONDS = 300.0
PROGRESS_INTERVAL_SECONDS = 30.0
PROGRESS_REPO_INTERVAL = 10
PROGRESS_PROBE_INTERVAL = 100
SMALL_NON_LFS_MAX_BYTES = 1024 * 1024
HF_MIN_INTERVAL_SECONDS = 1.0
_USER_AGENT = "rapid-mlx mirror-drift-auditor"
_REAL_SLEEP = time.sleep
_SEVERITY = {"info": 10, "warning": 20, "error": 30, "never": 10_000}
_HF_GATE_LOCK = threading.Lock()
_hf_next_request = 0.0
_COUNT_LOCK = threading.Lock()
_mirror_retry_causes: dict[str, int] = {}
_mirror_retry_after_values: set[str] = set()
_request_counts = {"hf": 0, "mirror": 0}
_profile_counts: dict[str, float | int] = {
    "hf_throttle_seconds": 0.0,
    "hf_retry_seconds": 0.0,
    "mirror_retry_seconds": 0.0,
    "mirror_head_calls": 0,
    "mirror_head_seconds": 0.0,
    "mirror_get_calls": 0,
    "mirror_get_seconds": 0.0,
    "mirror_redirects": 0,
}
_REQUEST_CONTEXT = threading.local()


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


@dataclass
class AuditProgress:
    started: float = field(default_factory=time.monotonic)
    repos_total: int = 0
    repos_done: int = 0
    probes_total: int = 0
    probes_done: int = 0
    catalog_seconds: float = 0.0
    hf_seconds: float = 0.0
    mirror_seconds: float = 0.0
    reports: list[AliasReport] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def emit(self, reason: str) -> None:
        with _COUNT_LOCK:
            hf_calls = _request_counts["hf"]
            mirror_calls = _request_counts["mirror"]
        elapsed = self.elapsed()
        rate = self.probes_done / elapsed if elapsed else 0.0
        print(
            "PROGRESS"
            f" reason={reason} repos={self.repos_done}/{self.repos_total}"
            f" probes={self.probes_done}/{self.probes_total}"
            f" rate={rate:.2f}/s hf_calls={hf_calls}"
            f" mirror_calls={mirror_calls} elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    def start_periodic(self) -> None:
        def run() -> None:
            while not self._stop.wait(PROGRESS_INTERVAL_SECONDS):
                self.emit("timer")

        self._thread = threading.Thread(target=run, name="audit-progress", daemon=True)
        self._thread.start()

    def stop_periodic(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def metrics_line(self) -> str:
        elapsed = self.elapsed()
        rate = self.probes_done / elapsed if elapsed else 0.0
        mirror_rate = (
            self.probes_done / self.mirror_seconds if self.mirror_seconds else 0.0
        )
        with _COUNT_LOCK:
            counts = dict(_request_counts)
            profile = dict(_profile_counts)
            retry_causes = dict(_mirror_retry_causes)
            retry_after_values = set(_mirror_retry_after_values)
        cause_summary = (
            ",".join(
                f"{cause}:{count}" for cause, count in sorted(retry_causes.items())
            )
            or "none"
        )
        retry_after_summary = ",".join(sorted(retry_after_values, key=float)) or "none"
        return (
            "METRICS"
            f" repos={self.repos_done}/{self.repos_total}"
            f" probes={self.probes_done}/{self.probes_total}"
            f" probes_per_second={rate:.2f} hf_calls={counts['hf']}"
            f" mirror_calls={counts['mirror']} elapsed={elapsed:.1f}s"
            f" catalog={self.catalog_seconds:.1f}s hf={self.hf_seconds:.1f}s"
            f" mirror={self.mirror_seconds:.1f}s"
            f" mirror_probes_per_second={mirror_rate:.2f}"
            f" hf_throttle={profile['hf_throttle_seconds']:.1f}s"
            f" hf_retry={profile['hf_retry_seconds']:.1f}s"
            f" head={profile['mirror_head_calls']}"
            f" head_time={profile['mirror_head_seconds']:.1f}s"
            f" get={profile['mirror_get_calls']}"
            f" get_time={profile['mirror_get_seconds']:.1f}s"
            f" redirects={profile['mirror_redirects']}"
            f" mirror_retry={profile['mirror_retry_seconds']:.1f}s"
            f" mirror_retry_causes={cause_summary}"
            f" mirror_retry_after={retry_after_summary}"
        )


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
        if getattr(_REQUEST_CONTEXT, "kind", None) == "mirror":
            with _COUNT_LOCK:
                _profile_counts["mirror_redirects"] += 1
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


class _AuditAbortError(RuntimeError):
    """Stop work that was already admitted when another probe exhausted."""


class _MirrorAdmission:
    """Shared AIMD admission gate with a single post-cooldown canary."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.abort_event = threading.Event()
        self.ceiling = MAX_WORKERS
        self.limit = MAX_WORKERS
        self.active = 0
        self.cooldown_until = 0.0
        self.canary_pending = False
        self.canary_active = False
        self.admission_epoch = 0
        self.last_halved_epoch: int | None = None
        self.successes = 0
        self.deadline: float | None = None

    def reset(self, workers: int, deadline: float | None) -> None:
        with self._condition:
            self.abort_event.clear()
            self.ceiling = max(1, min(workers, MAX_WORKERS))
            self.limit = self.ceiling
            self.active = 0
            self.cooldown_until = 0.0
            self.canary_pending = False
            self.canary_active = False
            self.admission_epoch = 0
            self.last_halved_epoch = None
            self.successes = 0
            self.deadline = deadline
            self._condition.notify_all()

    def abort(self) -> None:
        self.abort_event.set()
        with self._condition:
            self._condition.notify_all()

    def _check_abort(self) -> None:
        if self.abort_event.is_set():
            raise _AuditAbortError("mirror audit aborted after an exhausted probe")

    def _check_deadline(self, end: float) -> None:
        if self.deadline is not None and end > self.deadline:
            self.abort()
            raise RuntimeError("required mirror wait exceeds audit deadline")

    def wait_delay(
        self,
        delay: float,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        self._check_abort()
        self._check_deadline(clock() + delay)
        if sleeper is _REAL_SLEEP:
            if self.abort_event.wait(delay):
                self._check_abort()
        else:
            sleeper(delay)
            self._check_abort()

    def acquire(
        self,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> tuple[bool, int]:
        """Return the canary flag and rate-limit epoch for this admission."""
        while True:
            self._check_abort()
            delay = 0.0
            with self._condition:
                self._check_abort()
                now = clock()
                if self.deadline is not None and now >= self.deadline:
                    self.abort()
                    raise RuntimeError("mirror audit deadline reached")
                delay = max(0.0, self.cooldown_until - now)
                if not delay:
                    if self.canary_pending:
                        if not self.canary_active and self.active == 0:
                            self.canary_active = True
                            self.active += 1
                            self.admission_epoch += 1
                            return True, self.admission_epoch
                    elif self.active < self.limit:
                        self.active += 1
                        return False, self.admission_epoch
                    self._condition.wait(timeout=0.1)
                    continue
            self.wait_delay(delay, clock, sleeper)

    def rate_limited(
        self, delay: float, clock: Callable[[], float], admission_epoch: int
    ) -> None:
        end = clock() + delay
        self._check_deadline(end)
        with self._condition:
            if (
                self.last_halved_epoch is None
                or admission_epoch > self.last_halved_epoch
            ):
                self.limit = max(1, self.limit // 2)
                self.last_halved_epoch = admission_epoch
            self.cooldown_until = max(self.cooldown_until, end)
            self.canary_pending = True
            self.successes = 0
            self._condition.notify_all()

    def finished(self, *, canary: bool, success: bool) -> None:
        with self._condition:
            self.active -= 1
            if canary:
                self.canary_active = False
                if success:
                    self.canary_pending = False
            elif success and not self.canary_pending and self.limit < self.ceiling:
                self.successes += 1
                if self.successes >= self.limit:
                    self.limit += 1
                    self.successes = 0
            self._condition.notify_all()


_MIRROR_ADMISSION = _MirrorAdmission()


def _reset_retry_state(
    *, workers: int = MAX_WORKERS, deadline: float | None = None
) -> None:
    _MIRROR_ADMISSION.reset(workers, deadline)
    with _COUNT_LOCK:
        _mirror_retry_causes.clear()
        _mirror_retry_after_values.clear()


def _record_mirror_retry(cause: str, delay: float, retry_after: str | None) -> None:
    with _COUNT_LOCK:
        _profile_counts["mirror_retry_seconds"] += delay
        _mirror_retry_causes[cause] = _mirror_retry_causes.get(cause, 0) + 1
        if retry_after is not None:
            _mirror_retry_after_values.add(retry_after)


def _request(
    url: str,
    *,
    method: str = "GET",
    timeout: float = 30.0,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> Any:
    """Make a polite cache-busted request, retrying transient failures."""
    clock = clock or time.monotonic
    sleeper = sleeper or time.sleep
    is_mirror = getattr(_REQUEST_CONTEXT, "kind", None) == "mirror"
    last: BaseException | None = None
    for attempt in range(5):
        if is_mirror:
            canary, admission_epoch = _MIRROR_ADMISSION.acquire(clock, sleeper)
        else:
            canary, admission_epoch = False, 0
        delay = float(2**attempt)
        cause: str
        retry_after: str | None = None
        request = urllib.request.Request(
            _cache_busted(url),
            method=method,
            headers={"Cache-Control": "no-cache", "User-Agent": _USER_AGENT},
        )
        try:
            response = _OPENER.open(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                if is_mirror:
                    _MIRROR_ADMISSION.finished(canary=canary, success=True)
                return error
            last = error
            cause = str(error.code)
            retry_after = error.headers.get("Retry-After")
            try:
                if (
                    retry_after is None
                    or not retry_after.isascii()
                    or not retry_after.isdigit()
                ):
                    raise ValueError
                retry_after_seconds = int(retry_after)
                if retry_after_seconds <= 0:
                    raise ValueError
            except (AttributeError, ValueError):
                retry_after = None
            else:
                delay = max(delay, min(retry_after_seconds, MAX_RETRY_AFTER_SECONDS))
        except TimeoutError as error:
            last = error
            cause = "timeout"
        except urllib.error.URLError as error:
            last = error
            cause = "url_error"
        except OSError as error:
            last = error
            cause = "os_error"
        else:
            if is_mirror:
                _MIRROR_ADMISSION.finished(canary=canary, success=True)
            return response

        if attempt < 4:
            if is_mirror:
                _record_mirror_retry(cause, delay, retry_after)
                if cause == "429":
                    try:
                        _MIRROR_ADMISSION.rate_limited(delay, clock, admission_epoch)
                    except RuntimeError as error:
                        value = retry_after or f"{delay:g}"
                        raise RuntimeError(
                            f"mirror Retry-After {value}s exceeds audit deadline"
                        ) from error
                    finally:
                        _MIRROR_ADMISSION.finished(canary=canary, success=False)
                    continue
                _MIRROR_ADMISSION.finished(canary=canary, success=False)
                _MIRROR_ADMISSION.wait_delay(delay, clock, sleeper)
            else:
                sleeper(delay)
        elif is_mirror:
            _MIRROR_ADMISSION.finished(canary=canary, success=False)
    if is_mirror:
        _MIRROR_ADMISSION.abort()
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
            with _COUNT_LOCK:
                _profile_counts["hf_retry_seconds"] += delay
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
            with _COUNT_LOCK:
                _profile_counts["hf_throttle_seconds"] += delay
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
    method = "get" if fetch_body else "head"
    started = time.monotonic()
    with _COUNT_LOCK:
        _request_counts["mirror"] += 1
        _profile_counts[f"mirror_{method}_calls"] += 1
    _REQUEST_CONTEXT.kind = "mirror"
    try:
        response = _request(_mirror_url(repo_id, item.path), method=method.upper())
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
    finally:
        _REQUEST_CONTEXT.kind = None
        with _COUNT_LOCK:
            _profile_counts[f"mirror_{method}_seconds"] += time.monotonic() - started


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
    _MIRROR_ADMISSION._check_abort()
    probe = _public_probe(repo_id, item)
    metadata = (
        _r2_metadata(r2_client, repo_id, item.path)
        if r2_client is not None and item.sha256 is not None
        else None
    )
    return probe, metadata


def _apply_probe_result(
    report: AliasReport,
    item: HfFile,
    probe: MirrorProbe,
    metadata: dict[str, str] | None,
    *,
    has_r2: bool,
    sync_in_progress: bool,
) -> bool:
    """Apply one completed probe and return whether a required file is missing."""
    if probe.status < 200 or probe.status >= 300:
        report.findings.append(
            _file_finding(
                "missing_file",
                item,
                f"HTTP {probe.status}",
                sync_in_progress=sync_in_progress,
            )
        )
        return not _optional_asset(item.path)
    if item.size is not None and probe.size != item.size:
        report.findings.append(
            _file_finding(
                "size_mismatch",
                item,
                f"mirror={probe.size} hf={item.size} etag={probe.etag or 'missing'}",
                sync_in_progress=sync_in_progress,
            )
        )
    if item.sha256 is None and item.oid:
        if probe.blob_oid is None:
            report.findings.append(
                Finding(
                    "content_check",
                    "info",
                    item.path,
                    "unverified: body exceeds "
                    f"{SMALL_NON_LFS_MAX_BYTES}-byte probe limit",
                )
            )
        elif probe.blob_oid != item.oid:
            report.findings.append(
                _file_finding(
                    "content_mismatch",
                    item,
                    f"mirror_blob={probe.blob_oid} hf_blob={item.oid}",
                    sync_in_progress=sync_in_progress,
                )
            )
    if item.sha256 is not None:
        public_sha = _etag_sha256(probe.etag)
        metadata_sha = metadata.get("hf-sha256") if metadata is not None else None
        if has_r2 and metadata_sha is None:
            report.findings.append(
                _file_finding(
                    "content_mismatch",
                    item,
                    f"no checksum metadata; hf_sha256={item.sha256}",
                    sync_in_progress=sync_in_progress,
                )
            )
        else:
            actual_sha = metadata_sha if has_r2 else public_sha
            if actual_sha is not None and actual_sha != item.sha256:
                report.findings.append(
                    _file_finding(
                        "content_mismatch",
                        item,
                        f"mirror_sha256={actual_sha} hf_sha256={item.sha256}",
                        sync_in_progress=sync_in_progress,
                    )
                )
    return False


def audit(
    main_aliases_path: Path = ALIASES_PATH,
    audio_aliases_path: Path = AUDIO_ALIASES_PATH,
    *,
    aliases: set[str] | None = None,
    only_used: bool = False,
    workers: int = MAX_WORKERS,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    progress: AuditProgress | None = None,
) -> list[AliasReport]:
    """Audit aliases; ``only_used`` omits bucket-only catalog inventory rows."""
    specs = _load_aliases(main_aliases_path, audio_aliases_path)
    selected = [spec for spec in specs if aliases is None or spec.alias in aliases]
    if aliases is not None:
        unknown = aliases - {spec.alias for spec in selected}
        if unknown:
            raise ValueError(f"unknown alias(es): {', '.join(sorted(unknown))}")

    pool_size = max(1, min(workers, MAX_WORKERS))
    deadline = time.monotonic() + max(0.0, deadline_seconds)
    with _COUNT_LOCK:
        _request_counts.update(hf=0, mirror=0)
        for metric_name, value in _profile_counts.items():
            _profile_counts[metric_name] = 0.0 if isinstance(value, float) else 0
    _reset_retry_state(workers=pool_size, deadline=deadline)

    catalog_started = time.monotonic()
    entries = _catalog_entries()
    if progress is not None:
        progress.catalog_seconds = time.monotonic() - catalog_started
    by_alias = {
        str(entry["alias"]).lower(): entry for entry in entries if entry.get("alias")
    }
    r2_client = _maybe_r2_client()

    # One paced model_info call per unique repository, fanned out to every alias.
    repos: dict[str, HfRepo] = {}
    repo_errors: dict[str, str] = {}
    unique_repos = {spec.hf_path for spec in selected}
    if progress is not None:
        progress.repos_total = len(unique_repos)
        progress.emit("catalog-listed")
    hf_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(pool_size, HF_MAX_WORKERS)) as hf_pool:
        hf_futures = {
            hf_pool.submit(_hf_repo, repo_id): repo_id for repo_id in unique_repos
        }
        for hf_future in as_completed(hf_futures):
            repo_id = hf_futures[hf_future]
            try:
                repos[repo_id] = hf_future.result()
            except Exception as error:
                repo_errors[repo_id] = str(error).splitlines()[0]
            if progress is not None:
                progress.repos_done += 1
                if progress.repos_done % PROGRESS_REPO_INTERVAL == 0:
                    progress.emit("repo-batch")
    if progress is not None:
        progress.hf_seconds = time.monotonic() - hf_started

    reports = []
    report_context: list[
        tuple[AliasReport, AliasSpec, dict[str, Any] | None, list[HfFile], bool]
    ] = []
    probes: dict[tuple[str, str], HfFile] = {}
    probe_targets: dict[tuple[str, str], list[tuple[AliasReport, HfFile, bool]]] = {}
    for spec in selected:
        entry = by_alias.get(spec.alias.lower())
        report = _new_report(spec, entry, r2_client is not None)
        files: list[HfFile]
        if spec.hf_path in repo_errors:
            report.findings.append(
                Finding("hf_unavailable", "error", detail=repo_errors[spec.hf_path])
            )
            files = []
        else:
            files = _selected_files(repos[spec.hf_path].files, spec.subfolder)
        report.checked_files = len(files)
        in_progress = _sync_in_progress(entry)
        report_context.append((report, spec, entry, files, in_progress))
        reports.append(report)
        for item in files:
            probe_key = (spec.hf_path, item.path)
            probes.setdefault(probe_key, item)
            probe_targets.setdefault(probe_key, []).append((report, item, in_progress))
    if progress is not None:
        progress.reports = reports
        progress.probes_total = len(probes)
        progress.emit("mirror-start")

    # Mirror I/O is globally deduplicated and isolated in its own bounded pool.
    # Keeping at most two worker-windows submitted makes an exhausted transient
    # promptly cancellable instead of leaving thousands of queued futures.
    missing_required_by_report: dict[int, int] = {}
    mirror_started = time.monotonic()
    mirror_pool = ThreadPoolExecutor(max_workers=pool_size)
    probe_items = iter(probes.items())
    mirror_futures: dict[Future[Any], tuple[str, str]] = {}

    def submit_next() -> bool:
        try:
            key, item = next(probe_items)
        except StopIteration:
            return False
        mirror_futures[
            mirror_pool.submit(_probe_with_metadata, key[0], item, r2_client)
        ] = key
        return True

    for _ in range(min(len(probes), pool_size * 2)):
        submit_next()
    try:
        while mirror_futures:
            completed, _pending = wait(mirror_futures, return_when=FIRST_COMPLETED)
            completed_count = 0
            for mirror_future in completed:
                probe_key = mirror_futures.pop(mirror_future)
                probe, metadata = mirror_future.result()
                completed_count += 1
                for report, item, in_progress in probe_targets[probe_key]:
                    missing_required_by_report[id(report)] = (
                        missing_required_by_report.get(id(report), 0)
                        + _apply_probe_result(
                            report,
                            item,
                            probe,
                            metadata,
                            has_r2=r2_client is not None,
                            sync_in_progress=in_progress,
                        )
                    )
                if progress is not None:
                    progress.probes_done += 1
                    if progress.probes_done % PROGRESS_PROBE_INTERVAL == 0:
                        progress.emit("probe-batch")
            for _ in range(completed_count):
                submit_next()
    except BaseException:
        _MIRROR_ADMISSION.abort()
        for mirror_future in mirror_futures:
            mirror_future.cancel()
        raise
    finally:
        mirror_pool.shutdown(
            wait=True, cancel_futures=_MIRROR_ADMISSION.abort_event.is_set()
        )
    if progress is not None:
        progress.mirror_seconds = time.monotonic() - mirror_started

    for report, spec, entry, files, in_progress in report_context:
        required = _required_files(files, None)
        missing_required = missing_required_by_report.get(id(report), 0)
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
        max([len(headings[index]), *(len(row[index]) for row in rows)])
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


def _render_partial(progress: AuditProgress, reason: str) -> str:
    return f"PARTIAL REPORT ({reason})\n{_render_text(list(progress.reports))}\n"


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
        "--workers", type=int, default=MAX_WORKERS, help="concurrency, capped at 32"
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=DEFAULT_DEADLINE_SECONDS,
        help="fail before a required wait exceeds this job budget (default: 1500)",
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
    progress = AuditProgress()
    terminating = False

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        nonlocal terminating
        if terminating:
            return
        terminating = True
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        payload = _render_partial(progress, "terminated before completion").encode(
            errors="replace"
        )
        os.write(2, payload)
        os._exit(124)

    previous_sigterm = signal.signal(signal.SIGTERM, handle_sigterm)
    progress.start_periodic()
    try:
        reports = audit(
            args.aliases_path,
            args.audio_aliases_path,
            aliases=set(args.alias) or None,
            only_used=args.only_used,
            workers=args.workers,
            deadline_seconds=args.deadline_seconds,
            progress=progress,
        )
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(
            _render_partial(progress, "failed before completion"),
            file=sys.stderr,
            end="",
        )
        print(f"mirror drift audit failed: {error}", file=sys.stderr)
        return 2
    finally:
        progress.stop_periodic()
        signal.signal(signal.SIGTERM, previous_sigterm)
    progress.emit("complete")
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
            ),
            flush=True,
        )
    else:
        print(_render_text(reports), flush=True)
    print(progress.metrics_line(), file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
