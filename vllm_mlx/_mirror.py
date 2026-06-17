"""R2-first / HuggingFace-fallback model downloader.

``rapid-mlx pull <alias>`` (and the implicit prefetch invoked by
``rapid-mlx serve <alias>`` / ``rapid-mlx chat <alias>`` when the model
isn't cached) tries the project's Cloudflare R2 mirror at
``https://models.rapidmlx.com`` first and falls back to HuggingFace
**per file** on any miss. R2 is edge-cached and substantially faster
than the HF CDN for paths it has; per-file fallback keeps users
unblocked when the mirror is partial (some aliases have ``config.json``
mirrored but weight shards still uploading).

Design constraints (from PR #649 spec):

* Per-file fallback — each file is tried on R2; on any non-2xx (404 in
  practice) we fall back to ``hf_hub_download`` for *that file only*.
  We never abort the whole pull on the first R2 miss.
* Catalog-aware — we hit ``GET /api/models`` once to learn whether the
  alias's HF repo is mirrored. If ``status != "mirrored"``, we skip R2
  entirely. If the catalog fetch fails (network, 5xx), we transparently
  fall through to HF for everything.
* HF-cache-compatible — files land at
  ``~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<rev>/<file>``
  with ``refs/main`` pinned. The next ``hf_hub_download`` /
  ``snapshot_download`` call sees a cache hit. We do NOT invent a
  parallel cache.
* Default ON — set ``RAPID_MLX_MODEL_MIRROR=""`` to disable.
* No new third-party deps — stdlib ``urllib`` + ``huggingface_hub``.
* Concurrency capped at 4 to stay polite to Cloudflare.
* Resume — interrupted ``.part`` files are completed via ``Range`` requests.
* Integrity — ``Content-Length`` from R2 is compared against the size
  HF advertises. Mismatch → delete the R2 byte stream and fall back.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Cloudflare's edge fronts the R2 bucket at this hostname. The catalog
# lives at ``/api/models`` and per-file objects at
# ``/<owner>/<repo>/<filename>``. Override / disable with the
# ``RAPID_MLX_MODEL_MIRROR`` env var.
MIRROR_DEFAULT = "https://models.rapidmlx.com"

# Cloudflare 403s the default ``Python-urllib/*`` UA — verified by the
# maintainer. Any plausible browser-ish UA works. Keep ``rapid-mlx`` in
# the string so the maintainer can spot our traffic in R2 logs.
_USER_AGENT = "Mozilla/5.0 (rapid-mlx mirror client)"

# Catalog responses are tiny (a few hundred KB at most) and Cloudflare
# caches them ``public, max-age=300``. 10 s is plenty.
_CATALOG_TIMEOUT = 10.0

# Per-file timeout for R2 connect + initial response. Large shards stream
# via ``resp.read()`` after this point, which uses the socket-level
# default timeout (no per-read clock). 60 s matches the old
# ``_try_mirror_prefetch`` value.
_FILE_TIMEOUT = 60.0

# Polite cap. Cloudflare can take more, but four parallel connections to
# the same edge host already saturate a typical home connection and we
# don't want to look like a scraper.
_MAX_WORKERS = 4

# Chunk size for streaming reads. 8 MiB matches the old prefetch path
# and keeps tqdm-free progress redraws coarse enough to not flood the
# terminal.
_CHUNK_BYTES = 8 * 1024 * 1024


def _mirror_base() -> str:
    """Return the configured mirror base URL, or ``""`` if disabled.

    Empty string means "force HF" — distinct from "unset" which means
    "use the project default". This is the documented opt-out knob.
    """
    return os.environ.get("RAPID_MLX_MODEL_MIRROR", MIRROR_DEFAULT).strip()


def fetch_catalog(
    base: str, timeout: float = _CATALOG_TIMEOUT
) -> dict[str, Any] | None:
    """Fetch ``GET <base>/api/models`` and return the parsed JSON.

    Returns ``None`` on any failure (network, 5xx, malformed JSON) — the
    caller treats this as "skip R2, go straight to HF".
    """
    url = f"{base.rstrip('/')}/api/models"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            raw = resp.read()
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ):
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def find_catalog_entry(catalog: dict[str, Any], hf_path: str) -> dict[str, Any] | None:
    """Find the catalog entry for ``hf_path`` (case-insensitive on hf_path).

    Returns ``None`` if the catalog doesn't list this repo. The caller
    should treat this as "not mirrored, go to HF".
    """
    models = catalog.get("models")
    if not isinstance(models, list):
        return None
    needle = hf_path.lower()
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("hf_path", "")).lower() == needle:
            return entry
    return None


def _is_mirrored(entry: dict[str, Any]) -> bool:
    return str(entry.get("status", "")).lower() == "mirrored"


def _hf_cache_root() -> Path:
    """Resolve the HF cache root, honoring ``HF_HUB_CACHE`` / ``HF_HOME``."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _validate_relative_filename(fname: str) -> bool:
    """Reject path-traversal / absolute paths in catalog or sibling listings.

    A maliciously-crafted entry like ``../../etc/passwd`` would otherwise
    let ``snap_dir / fname`` resolve outside the snapshot directory. Same
    guard as the original ``_try_mirror_prefetch`` shipped with PR #647.
    """
    if not fname or fname.startswith("/") or Path(fname).is_absolute():
        return False
    parts = Path(fname).parts
    if ".." in parts:
        return False
    return True


def _build_r2_url(base: str, download_url_base: str, fname: str) -> str:
    """Compose the per-file R2 URL.

    The catalog gives ``download_url_base`` as ``/<owner>/<repo>/`` —
    we append a URL-encoded filename. Encoding each segment individually
    handles spaces, ``#``, ``?``, ``%`` in filenames.
    """
    fname_parts = Path(fname).parts
    encoded = "/".join(urllib.parse.quote(p, safe="") for p in fname_parts)
    # Normalize the join — strip trailing slash on base, leading slash on
    # path, then rejoin with one separator. Avoids ``//`` and missing
    # ``/`` cases.
    base = base.rstrip("/")
    path = download_url_base.strip("/")
    return f"{base}/{path}/{encoded}"


def _download_one_from_r2(
    url: str,
    target: Path,
    expected_size: int | None,
) -> tuple[bool, str]:
    """Download a single file from R2 into ``target``.

    Returns ``(True, "")`` on success. On any failure returns
    ``(False, <reason>)`` where reason is a short tag used for the
    summary line. Cleans up partial ``.part`` files on failure.

    Supports resume via ``Range: bytes=<offset>-`` when a non-empty
    ``.part`` already exists from a prior aborted run.
    """
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"mkdir:{type(e).__name__}"

    # Resume offset — pick up where a prior run left off. Codex
    # round-3 NIT #2: if the existing .part is unstatable (directory,
    # permission, etc.) we drop it and start fresh rather than letting
    # the OSError propagate out of the worker.
    existing = 0
    if tmp.exists():
        try:
            existing = tmp.stat().st_size
        except OSError:
            _safe_unlink(tmp)
            existing = 0

    headers = {"User-Agent": _USER_AGENT}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_FILE_TIMEOUT) as resp:
            # 200 = full body; 206 = partial (Range honored). Anything
            # else is a miss — including 416 (range not satisfiable),
            # which means the .part is already complete or the file
            # shrank server-side. Safer to wipe and refetch via HF.
            if resp.status not in (200, 206):
                _safe_unlink(tmp)
                return False, f"status:{resp.status}"

            content_length = resp.headers.get("Content-Length")
            try:
                length = int(content_length) if content_length else 0
            except ValueError:
                _safe_unlink(tmp)
                return False, "bad-content-length"

            # Total final size: resume bytes + body bytes.
            total_size = existing + length if resp.status == 206 else length

            # Integrity precheck — if HF told us the size, R2 must agree.
            # Mismatch means the mirror has a different (possibly stale)
            # build of this file. Fall back to HF, don't risk a corrupt
            # cache.
            if expected_size and total_size and total_size != expected_size:
                _safe_unlink(tmp)
                return False, f"size-mismatch:{total_size}!={expected_size}"

            mode = "ab" if resp.status == 206 and existing > 0 else "wb"
            read = 0
            with tmp.open(mode) as fh:
                while True:
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
                    read += len(chunk)

            # Short-read guard — Content-Length lied or the connection
            # dropped silently. Don't rename a truncated file into the
            # snapshot; let HF redownload.
            if length > 0 and read != length:
                _safe_unlink(tmp)
                return False, f"short-read:{read}!={length}"
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ) as e:
        _safe_unlink(tmp)
        return False, type(e).__name__

    # Final size check against HF after the rename — paranoid but cheap.
    final_size = tmp.stat().st_size if tmp.exists() else 0
    if expected_size and final_size != expected_size:
        _safe_unlink(tmp)
        return False, f"final-size-mismatch:{final_size}!={expected_size}"

    try:
        tmp.rename(target)
    except OSError as e:
        _safe_unlink(tmp)
        return False, f"rename:{type(e).__name__}"
    return True, ""


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _hf_fallback_one(
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: Path | None = None,
) -> tuple[bool, str | None]:
    """Download a single file from HuggingFace into the standard cache.

    Returns ``(True, resolved_path)`` on success, ``(False, None)`` on
    failure. The resolved path is whatever ``hf_hub_download`` returns
    (a symlink under ``snapshots/<rev>/`` pointing to a blob). Used for
    the per-file R2 miss path. Codex round-1 NIT #3: capture the path
    rather than re-resolving via ``snap_dir / fname``, so success
    accounting is robust to changes in HF's symlink layout.

    Codex round-2 BLOCKING #4: narrow the exception net. Only expected
    network/cache/HF-API errors are swallowed; programmer errors
    (``TypeError``, ``AttributeError``) and validation errors
    (``HFValidationError``) propagate so a misuse surfaces a real
    stack trace instead of being silently re-routed through the
    ``snapshot_download`` fallback.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return True, path
    except (
        # Expected network / HF API surface — these are legitimate
        # "this file isn't reachable right now" signals.
        EntryNotFoundError,
        RepositoryNotFoundError,
        HfHubHTTPError,
        OSError,
        TimeoutError,
    ):
        return False, None


def _print_dim(msg: str) -> None:
    """Quiet status line. Honors NO_COLOR / non-TTY."""
    print(msg)


def download_with_mirror_fallback(
    repo_id: str,
    cache_dir: Path | None = None,
) -> bool:
    """Download ``repo_id`` to the HF cache via R2-first / HF-fallback.

    Returns True if every file landed in the snapshot dir (mix of R2 +
    HF is fine). Returns False if the caller should fall back to the
    plain ``snapshot_download(repo_id)`` path — typically because we
    couldn't enumerate the repo or because the catalog said this alias
    isn't mirrored AND we want the caller to retain its existing
    fetched-from-HF logging path.

    On False, no partial damage to the cache is left behind — any files
    we did fetch are valid HF-cache entries that ``snapshot_download``
    will skip.
    """
    base = _mirror_base()
    if not base or "/" not in repo_id:
        # Mirror disabled or repo_id isn't a HF-shaped ``owner/name``.
        # Local paths fall here too. Defer to caller's HF path.
        return False

    # HF model_info gives us the canonical revision + per-file sizes.
    # We need both — the revision pins the snapshot dir, and the sizes
    # let us validate R2 responses. Without it we can't pin a revision,
    # which would mean writing files under an unknowable sha — so fall
    # through to HF if this fails.
    try:
        from huggingface_hub import model_info

        info = model_info(repo_id, files_metadata=True)
    except Exception:
        return False

    revision = getattr(info, "sha", None)
    siblings = getattr(info, "siblings", None) or []
    files: list[tuple[str, int | None]] = []
    for s in siblings:
        rname = getattr(s, "rfilename", None)
        if not rname:
            continue
        if not _validate_relative_filename(rname):
            # Path traversal guard — if the HF listing itself is
            # malicious, refuse to act on it. Punt to HF's own loader,
            # which has its own checks.
            return False
        size = getattr(s, "size", None)
        files.append((rname, size if isinstance(size, int) else None))
    if not revision or not files:
        return False

    cache_root = cache_dir if cache_dir else _hf_cache_root()
    owner, _, repo = repo_id.partition("/")
    repo_root = cache_root / f"models--{owner}--{repo}"
    snap_dir = repo_root / "snapshots" / revision
    refs_dir = repo_root / "refs"
    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    # Catalog lookup — only probe R2 if the catalog confirms the alias
    # is mirrored. Catalog fetch failure (network, 5xx) is treated as
    # "no R2 for this run" — we still attempt the HF path per file.
    catalog = fetch_catalog(base)
    catalog_entry = None
    catalog_mirrored = False
    if catalog is not None:
        catalog_entry = find_catalog_entry(catalog, repo_id)
        catalog_mirrored = catalog_entry is not None and _is_mirrored(catalog_entry)

    use_r2 = catalog_mirrored and catalog_entry is not None
    if use_r2:
        dub = str(catalog_entry.get("download_url_base", "")).strip()
        if not dub:
            use_r2 = False

    # Codex round-2 BLOCKING #1: custom (non-default)
    # ``RAPID_MLX_MODEL_MIRROR`` URLs typically point at a simple static
    # HTTP mirror at ``<base>/<owner>/<repo>/<file>`` and do NOT serve
    # ``/api/models``. PR #647's whole-repo prefetch worked against
    # those — silently disabling R2 here would regress that contract.
    # When the catalog is unreachable AND the base is non-default, fall
    # back to the direct URL layout so #647-style mirrors keep working.
    if not use_r2 and catalog is None and base != MIRROR_DEFAULT:
        own, _, rep = repo_id.partition("/")
        if own and rep:
            use_r2 = True
            dub = f"{own}/{rep}"

    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    BOLD = "\x1b[1m" if is_tty else ""
    DIM = "\x1b[2m" if is_tty else ""
    RESET = "\x1b[0m" if is_tty else ""

    if use_r2:
        _print_dim(
            f"  {BOLD}Pulling {repo_id}{RESET} {DIM}(R2 mirror, fallback: HF){RESET}"
        )
    elif catalog is None:
        _print_dim(
            f"  {BOLD}Pulling {repo_id}{RESET} {DIM}(catalog unreachable, "
            f"using HF){RESET}"
        )
    else:
        _print_dim(
            f"  {BOLD}Pulling {repo_id}{RESET} {DIM}(not mirrored, using HF){RESET}"
        )

    # Per-file plan: for each file, attempt R2 first (if eligible),
    # otherwise fall straight to HF. Run a small pool in parallel.
    r2_hits = 0
    hf_hits = 0
    misses: list[str] = []
    total_bytes = 0

    def _do_file(item: tuple[str, int | None]) -> tuple[str, str, int]:
        fname, expected_size = item
        target = snap_dir / fname
        # Belt-and-braces: normalize against snap_dir to refuse symlink
        # or normpath escapes the parts check missed.
        try:
            target_norm = Path(os.path.normpath(str(target)))
            snap_norm = Path(os.path.normpath(str(snap_dir)))
            target_norm.relative_to(snap_norm)
        except ValueError:
            return fname, "skip-traversal", 0

        # Already cached — file present at snapshot path, nothing to do.
        # Codex round-1 BLOCKING #1: a prior interrupted download could
        # leave a non-empty-but-truncated file at the snapshot path. If
        # HF told us the canonical size, the cached file MUST match it
        # before we accept it; otherwise we delete it and re-fetch.
        # When HF didn't expose a size (rare — README-only repos etc.),
        # fall back to the old non-empty heuristic.
        #
        # Codex round-2 BLOCKING #3: if ``target`` is a broken symlink
        # or otherwise unstatable, ``stat()`` raises an OSError that
        # would otherwise collapse this worker into a "miss" and force
        # the whole pull to fall back. Wrap in OSError protection.
        #
        # Codex round-3 NIT #3: if a DIRECTORY occupies the target
        # path, we cannot rename a file over it later — surface that
        # as a definitive "miss" so the outer caller falls back to
        # ``snapshot_download`` (which has its own conflict resolution
        # and a better error path for the user).
        try:
            if target.is_dir() and not target.is_symlink():
                return fname, "miss", 0
            if target.exists():
                cached_size = target.stat().st_size
                if expected_size and cached_size != expected_size:
                    # Stale / truncated cache entry — drop it and fall
                    # through to the R2/HF re-fetch below.
                    _safe_unlink(target)
                elif cached_size > 0:
                    return fname, "cached", cached_size
        except OSError:
            # Target is a broken symlink / permission denied / etc.
            # Try to remove it (best-effort) and fall through. The
            # rename in ``_download_one_from_r2`` will then place a
            # fresh file at the path.
            _safe_unlink(target)

        if use_r2 and catalog_entry is not None:
            url = _build_r2_url(base, dub, fname)
            ok, _reason = _download_one_from_r2(url, target, expected_size)
            if ok:
                try:
                    size = target.stat().st_size if target.exists() else 0
                except OSError:
                    size = 0
                return fname, "r2", size

        # Either R2 not eligible or R2 missed — fall back to HF for
        # this file. Let huggingface_hub handle its own cache layout.
        ok, hf_path = _hf_fallback_one(repo_id, fname, revision, cache_dir=cache_root)
        if ok:
            # ``hf_hub_download`` returns the resolved snapshot path
            # (typically a symlink to a blob). Stat the path it gave us
            # directly — that's the authoritative success signal. Fall
            # back to the predicted snapshot path only if the returned
            # path is missing for some reason (it shouldn't be).
            size = 0
            try:
                if hf_path:
                    size = Path(hf_path).stat().st_size
                else:
                    size = (snap_dir / fname).stat().st_size
            except OSError:
                size = 0
            return fname, "hf", size

        return fname, "miss", 0

    # Concurrency cap — small pool to stay polite. Even when R2 isn't
    # in play, parallel HF downloads (hf_hub_download is thread-safe) is
    # marginally faster than serial.
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_do_file, item): item[0] for item in files}
        for fut in as_completed(futures):
            fname = futures[fut]
            # Codex round-3 NIT #1: narrow the worker exception net.
            # Only convert expected network / filesystem errors into a
            # silent "miss". Programmer errors (TypeError, etc.) and
            # HF validation errors propagate so misuse surfaces a real
            # stack trace instead of disappearing into the fallback.
            try:
                _, kind, size = fut.result()
            except (OSError, urllib.error.URLError, urllib.error.HTTPError):
                kind, size = "miss", 0
            if kind == "r2":
                r2_hits += 1
                total_bytes += size
            elif kind == "hf":
                hf_hits += 1
                total_bytes += size
            elif kind == "cached":
                # Already present — count as r2/hf-neutral but include
                # bytes so the summary reflects the full snapshot size.
                total_bytes += size
            else:
                misses.append(fname)

    if misses:
        # At least one file we couldn't get from either source. Caller
        # should fall back to ``snapshot_download`` — it has more retry
        # logic and will surface a clean error to the user.
        return False

    # Pin the snapshot. ``is_repo_cached`` requires ``refs/main`` to
    # consider the snapshot complete; without this the next run would
    # see a partial-looking cache. ``pull_command`` also reads
    # ``refs/main`` to print "Cached at: …/snapshots/<sha>" — a stale
    # ref would make that line point at the wrong snapshot.
    #
    # We always fetch HEAD of ``main`` (``model_info(repo_id)`` with no
    # revision argument resolves to the default branch's tip), so it's
    # safe — and required — to overwrite ``refs/main`` with our sha.
    # This matches ``snapshot_download``'s own behaviour, which updates
    # ``refs/main`` on every default-revision pull. Codex round-2
    # BLOCKING #1+#2 reverted the round-1 "don't clobber" behaviour: a
    # stale ref left over from a manual ``snapshot_download(revision=
    # "<sha>")`` would otherwise survive our pull, breaking the cache
    # contract for the loader.
    try:
        (refs_dir / "main").write_text(revision)
    except OSError:
        return False

    mb = total_bytes / 1e6
    if r2_hits and hf_hits:
        _print_dim(
            f"  {BOLD}Pulled{RESET} {len(files)} files, {mb:.0f} MB "
            f"{DIM}(R2: {r2_hits}, HF: {hf_hits}){RESET}"
        )
    elif r2_hits:
        _print_dim(
            f"  {BOLD}Pulled{RESET} {len(files)} files, {mb:.0f} MB "
            f"{DIM}(R2: {r2_hits}){RESET}"
        )
    elif hf_hits:
        _print_dim(
            f"  {BOLD}Pulled{RESET} {len(files)} files, {mb:.0f} MB "
            f"{DIM}(HF: {hf_hits}){RESET}"
        )
    else:
        # All files were already cached — quiet success.
        _print_dim(f"  {BOLD}Already cached{RESET} ({len(files)} files, {mb:.0f} MB)")
    return True
