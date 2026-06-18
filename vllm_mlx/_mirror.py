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
import time
import unicodedata
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

    Most callers want to know WHY the catalog isn't available — use
    :func:`fetch_catalog_with_status` to get the HTTP status code
    alongside the body (codex round-6 NIT #3).
    """
    data, _ = fetch_catalog_with_status(base, timeout=timeout)
    return data


def fetch_catalog_with_status(
    base: str, timeout: float = _CATALOG_TIMEOUT
) -> tuple[dict[str, Any] | None, int | None]:
    """Like :func:`fetch_catalog` but also returns the HTTP status code.

    Returned status:
    * 200 on success (with parsed body).
    * The actual status (e.g. 404, 503) on a non-200 response with body
      None — lets the caller distinguish "no catalog endpoint here"
      (404 → safe to try direct-layout) from "transient failure" (5xx
      → don't waste time on direct-layout, just use HF).
    * ``None`` on network errors / malformed JSON — treat as transient.
    """
    url = f"{base.rstrip('/')}/api/models"
    # Codex round-11 NIT #3: ``urllib.request.Request(url)`` raises
    # ``ValueError`` for a malformed URL (e.g. a user typo in
    # ``RAPID_MLX_MODEL_MIRROR``). Construct it inside the guarded block
    # so it routes to "treat as transient, fall through to HF" instead
    # of escaping the whole pull with a raw stack trace.
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if status != 200:
                return None, status
            raw = resp.read()
    except urllib.error.HTTPError as e:
        # 4xx / 5xx — capture the status so the caller can decide.
        return None, e.code
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ):
        return None, None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, status
    if not isinstance(data, dict):
        return None, status
    return data, status


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


def _sidecar_key_for(relpath: str) -> str:
    """Build a filesystem-safe key for a file's sidecar artifacts.

    Codex round-14 BLOCKING #1+#2: the ``.part`` and ``.lock`` sidecars
    used to live next to the target inside ``snapshots/<sha>/``. That
    lets a repository file legitimately named ``.foo.rapid-mlx-mirror
    .part`` (yes, file names can start with dots) collide with the
    temp file for ``foo``, and similarly for ``.lock``. Move sidecars
    into ``repo_root/.rapid-mlx-mirror/`` with a flattened key derived
    from the *relative* path — no chance of collision with an HF
    sibling listing because no HF repo ships files with our key shape.

    Key shape: URL-encoded relpath (path separators replaced with
    ``__``). E.g. ``model/00001-of-00002.safetensors`` →
    ``model__00001-of-00002.safetensors``.
    """
    # ``relpath`` has already been ``_validate_relative_filename``-d so
    # no traversal escapes. The URL-encode handles weird chars; the
    # ``/`` → ``__`` swap flattens directory structure so we can stash
    # everything in a single sidecar dir.
    encoded = urllib.parse.quote(relpath, safe="")
    return encoded.replace("/", "__").replace("%2F", "__").replace("%2f", "__")


def _download_one_from_r2(
    url: str,
    target: Path,
    expected_size: int | None,
    *,
    expected_sha256: str | None = None,
    sidecar_dir: Path,
    sidecar_key: str,
    repo_root: Path | None = None,
) -> tuple[bool, str]:
    """Download a single file from R2 into ``target``.

    Returns ``(True, "")`` on success. On any failure returns
    ``(False, <reason>)`` where reason is a short tag used for the
    summary line. Cleans up partial ``.part`` files on failure.

    Supports resume via ``Range: bytes=<offset>-`` when a non-empty
    ``.part`` already exists from a prior aborted run.

    ``sidecar_dir`` is the per-repo private sidecar directory
    (``repo_root/.rapid-mlx-mirror/``) where the ``.part`` and ``.lock``
    files live — kept OUT of ``snapshots/<sha>/`` so they can't
    collide with legitimate repo assets (codex round-14 BLOCKING #1
    + #2). ``sidecar_key`` is the per-file key from
    :func:`_sidecar_key_for`.

    Codex round-5 BLOCKING #1: when ``expected_sha256`` is provided
    (HF's LFS sha256 metadata, set on weight shards), the downloaded
    bytes are checked against it before the rename. A mirror serving a
    same-size but corrupt object is rejected. For non-LFS files (small
    text assets), there is no LFS sha and the integrity check falls
    back to size-only — the realistic threat surface for tiny config
    files is much smaller.

    Issue #652: when ``expected_sha256`` is provided and ``repo_root``
    is set, the verified bytes land at ``repo_root/blobs/<sha>`` and
    ``target`` becomes a relative symlink to that blob — matching HF's
    own cache layout, so subsequent warm pulls hit the blob-name
    shortcut at the cached-check site (skipping a multi-GB rehash).
    Non-LFS files (no ``expected_sha256``) stay as regular files at
    ``target`` for parity with HF's own layout for tiny configs.
    """
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"mkdir:{type(e).__name__}"

    tmp = sidecar_dir / f"{sidecar_key}.part"
    lock_path = sidecar_dir / f"{sidecar_key}.lock"

    # Codex round-11 BLOCKING #2 / round-12 BLOCKING: ``fcntl.flock``
    # advisory lock on the lock sidecar. The lock file is kept on disk
    # after release so subsequent acquirers see the same inode (which
    # is what flock actually serializes on).
    lock_fh = _acquire_part_lock(lock_path)
    try:
        return _do_r2_download(
            url,
            target,
            tmp,
            expected_size,
            expected_sha256=expected_sha256,
            repo_root=repo_root,
        )
    finally:
        _release_part_lock(lock_fh, lock_path)


def _do_r2_download(
    url: str,
    target: Path,
    tmp: Path,
    expected_size: int | None,
    *,
    expected_sha256: str | None = None,
    repo_root: Path | None = None,
) -> tuple[bool, str]:
    """Inner R2 download body — runs with the per-file lock held.

    Split out from ``_download_one_from_r2`` so the caller can wrap
    every exit path in a single ``finally`` that releases the lock,
    without having to add a release call before each of the many
    ``return`` sites in this function (codex round-11 BLOCKING #2).
    """
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

            # Codex round-4 BLOCKING #2 (refined in round-7 BLOCKING #1):
            # a 206 response with a wrong ``Content-Range`` (proxy bug,
            # mirror serving from the middle, etc.) would let us
            # concatenate corrupt bytes onto an existing .part. The
            # round-4 fix only checked the start byte; codex round-7
            # caught that a partial range like ``bytes 50-99/200`` would
            # pass the startswith check but deliver only 50 of the 150
            # remaining bytes. Validate the FULL start-end/total tuple.
            if resp.status == 206:
                cr = resp.headers.get("Content-Range", "")
                # Expected shape: ``bytes <first>-<last>/<total>``.
                cr_first, cr_last, cr_total = _parse_content_range(cr)
                # Required: first == existing offset (else the server
                # is serving the wrong range). last == existing + length
                # - 1 (else the server is short-changing us). total ==
                # expected_size when HF told us (else the file changed
                # under us — different upload, different bytes).
                fail_reason: str | None = None
                if cr_first != existing:
                    fail_reason = f"wrong-start:{cr_first}!={existing}"
                elif length > 0 and cr_last != existing + length - 1:
                    fail_reason = f"wrong-end:{cr_last}!={existing + length - 1}"
                elif expected_size is not None and cr_total != expected_size:
                    # Codex round-9 NIT #3: when HF gave us a canonical
                    # size, a 206 with a missing / unknown / wrong total
                    # is all suspicious — a well-formed ranged response
                    # for a known-size object must echo that total. Reject
                    # both ``cr_total is None`` (header was ``*`` or
                    # malformed) and ``cr_total != expected_size``.
                    fail_reason = f"wrong-total:{cr_total}!={expected_size}"
                if fail_reason is not None:
                    _safe_unlink(tmp)
                    return False, f"bad-content-range:{fail_reason}"

            # Total final size: resume bytes + body bytes.
            total_size = existing + length if resp.status == 206 else length

            # Integrity precheck — if HF told us the size, R2 must agree.
            # Mismatch means the mirror has a different (possibly stale)
            # build of this file. Fall back to HF, don't risk a corrupt
            # cache. Codex round-5 NIT #3: use ``is not None`` so a
            # legitimate 0-byte file still gets the size check.
            if expected_size is not None and total_size and total_size != expected_size:
                _safe_unlink(tmp)
                return False, f"size-mismatch:{total_size}!={expected_size}"

            # Codex round-9 BLOCKING #1: if we sent ``Range`` but the
            # server returned 200 (range ignored — common with some
            # proxies / R2 misconfig), the response body is the FULL
            # object. Opening in ``"wb"`` correctly discards the stale
            # ``.part`` prefix, but the SHA hasher would otherwise have
            # been pre-fed those discarded bytes — producing a bogus
            # digest that fails the LFS check and forces an unneeded HF
            # fallback. Reset ``existing`` to 0 here so the prefix
            # rehash below is skipped and the mode falls to ``"wb"``.
            if resp.status == 200 and existing > 0:
                existing = 0

            mode = "ab" if resp.status == 206 and existing > 0 else "wb"
            read = 0
            # Codex round-5 BLOCKING #1: stream a SHA-256 of the bytes
            # we write. For resumed downloads (206 + existing > 0) we
            # have to rehash the existing .part prefix first so the
            # running digest covers the whole final file. If hashing the
            # prefix fails (e.g. the .part disappeared between stat and
            # open), wipe and fall back to HF.
            hasher = None
            if expected_sha256 is not None:
                import hashlib

                hasher = hashlib.sha256()
                if existing > 0:
                    try:
                        with tmp.open("rb") as prefix:
                            while True:
                                blk = prefix.read(_CHUNK_BYTES)
                                if not blk:
                                    break
                                hasher.update(blk)
                    except OSError:
                        _safe_unlink(tmp)
                        return False, "prefix-rehash-failed"
            with tmp.open(mode) as fh:
                while True:
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
                    if hasher is not None:
                        hasher.update(chunk)
                    read += len(chunk)

            # Short-read guard — Content-Length lied or the connection
            # dropped silently. Don't rename a truncated file into the
            # snapshot; let HF redownload.
            if length > 0 and read != length:
                _safe_unlink(tmp)
                return False, f"short-read:{read}!={length}"

            # Codex round-5 BLOCKING #1 — SHA-256 check (LFS files
            # only; non-LFS hashers stay ``None``). Same-size-but-
            # corrupt mirror objects fail here.
            if hasher is not None and expected_sha256 is not None:
                got = hasher.hexdigest()
                if got != expected_sha256:
                    _safe_unlink(tmp)
                    return (
                        False,
                        f"sha256-mismatch:{got[:8]}…!={expected_sha256[:8]}…",
                    )
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ) as e:
        _safe_unlink(tmp)
        return False, type(e).__name__

    # Codex round-5 BLOCKING #2: ``tmp.stat()`` was outside the
    # download try block, so an OSError here would escape without
    # cleaning up ``tmp``. Wrap in OSError protection.
    try:
        final_size = tmp.stat().st_size if tmp.exists() else 0
    except OSError as e:
        _safe_unlink(tmp)
        return False, f"final-stat:{type(e).__name__}"
    if expected_size is not None and final_size != expected_size:
        _safe_unlink(tmp)
        return False, f"final-size-mismatch:{final_size}!={expected_size}"

    # Issue #?? — defensive 0-byte rejection when HF didn't tell us a
    # size. Without this, an R2 worker that hits an unexpected error
    # path and returns ``200 OK`` with ``Content-Length: 0`` (instead of
    # the correct 404) is treated as a successful download of a real
    # 0-byte file — the puller writes an empty file at the snapshot
    # path, reports ``kind="r2", size=0`` to the summary logger, and the
    # user sees ``[N/M] file R2 (0 MB)`` for a file that genuinely
    # wasn't on the mirror. Downstream the file looks "cached" (it
    # exists, size matches the empty cached_size on the next pull) so
    # the silent failure can survive multiple warm pulls before the
    # real model code chokes on the empty file.
    #
    # When ``expected_size is not None`` the size-mismatch checks above
    # already catch this — a real 100-byte ``config.json`` returned as
    # 0 bytes fails ``final-size-mismatch:0!=100``. The unprotected case
    # is files where HF's ``siblings`` metadata doesn't expose a size
    # (some non-LFS files in older repos, or repos whose ``model_info``
    # call ran without ``files_metadata=True``). For those the empty
    # response would otherwise look like a legit empty file.
    #
    # A genuine 0-byte file (e.g. an empty ``.gitkeep``) is always
    # listed by HF with ``size == 0`` (not ``None``), so this check
    # doesn't reject the legitimate case — only the silent-failure case
    # where the mirror returns an empty body for a file we have no
    # canonical size for.
    if expected_size is None and final_size == 0:
        _safe_unlink(tmp)
        return False, "empty-response-no-size"

    # Issue #652: for LFS files (``expected_sha256`` known) land the
    # verified bytes at ``repo_root/blobs/<sha>`` and symlink the
    # snapshot path at it — matches HF's own layout exactly so the
    # blob-name shortcut at the warm-cache check fires uniformly for
    # both R2-sourced and HF-sourced cache state. Non-LFS files
    # (config.json, tokenizer.json, etc.) stay as regular files at
    # ``target``; HF does the same for those.
    if expected_sha256 is not None and repo_root is not None:
        ok, reason = _install_lfs_blob_and_symlink(
            tmp, target, repo_root, expected_sha256
        )
        if not ok:
            _safe_unlink(tmp)
            return False, reason
        return True, ""

    try:
        tmp.rename(target)
    except OSError as e:
        _safe_unlink(tmp)
        return False, f"rename:{type(e).__name__}"
    return True, ""


def _install_lfs_blob_and_symlink(
    tmp: Path,
    target: Path,
    repo_root: Path,
    expected_sha256: str,
) -> tuple[bool, str]:
    """Land verified LFS bytes at ``blobs/<sha>`` and symlink ``target``.

    Issue #652. After R2 has produced sha-verified bytes in ``tmp``,
    we want the resulting cache state to match HF's own layout:

    * The bytes live at ``repo_root/blobs/<expected_sha256>``.
    * ``target`` (the snapshot path) is a relative symlink at
      ``../../blobs/<expected_sha256>`` so a warm pull's blob-name
      shortcut (``resolved.name == expected_sha256``) fires and we
      skip rehashing multi-GB shards.

    Atomic + concurrency-safe:

    * Hold an ``fcntl.flock`` on the blob path while we materialize it,
      so two parallel pulls of the same file don't race on the rename.
    * Write to a ``<sha>.tmp`` sibling under ``blobs/`` then ``rename``
      onto the final blob — a kill mid-rename cannot leave a half-
      written blob with the canonical name (rename is atomic on
      POSIX).
    * If another process already materialized ``blobs/<sha>`` before
      we acquired the lock, just symlink to it and drop our ``tmp``
      (size+sha are already validated upstream, so any existing blob
      with that name has the same content).
    """
    blobs_dir = repo_root / "blobs"
    blob_path = blobs_dir / expected_sha256
    try:
        blobs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"blob-mkdir:{type(e).__name__}"

    # Per-blob lock keeps two concurrent pulls of the same LFS file
    # from racing on the blob write. We reuse ``_acquire_part_lock``'s
    # flock-on-lockfile pattern; the lockfile lives next to the blob.
    blob_lock = blobs_dir / f"{expected_sha256}.lock"
    lock_fh = _acquire_part_lock(blob_lock)
    try:
        # Race: another process (or another worker in this process)
        # may have already materialized the blob between our pre-lock
        # check and now. If it's there with content, drop our tmp and
        # just symlink. sha+size were validated by the caller, so any
        # blob already at the canonical name has identical bytes.
        if blob_path.exists() and not blob_path.is_symlink():
            _safe_unlink(tmp)
        else:
            # Atomic install: write to a ``.tmp`` sibling first so a
            # kill mid-rename can't leave a partial file at the
            # canonical blob name. ``Path.rename`` is atomic on POSIX
            # when source and destination share a filesystem (they
            # do — both under ``blobs/``).
            blob_tmp = blobs_dir / f"{expected_sha256}.tmp"
            _safe_unlink(blob_tmp)
            try:
                tmp.rename(blob_tmp)
            except OSError as e:
                return False, f"blob-stage:{type(e).__name__}"
            try:
                blob_tmp.rename(blob_path)
            except OSError as e:
                _safe_unlink(blob_tmp)
                return False, f"blob-rename:{type(e).__name__}"

        # Install the snapshot symlink. HF uses a relative symlink
        # (``../../blobs/<sha>``) so the cache is portable across
        # parent-directory moves; match that exactly. ``target`` may
        # already exist (stale file / broken symlink from a prior
        # interrupted run) — clear it before symlinking.
        _safe_unlink(target)
        # Snapshot files live at ``snapshots/<rev>/<fname>``; blobs at
        # ``blobs/<sha>``. The relative path from the snapshot file's
        # parent (``snapshots/<rev>/``) to the blob is therefore
        # ``../../blobs/<sha>``. Computing it explicitly via
        # ``os.path.relpath`` keeps the symlink correct even if a
        # future change reshapes the layout.
        rel = os.path.relpath(blob_path, start=target.parent)
        try:
            target.symlink_to(rel)
        except OSError as e:
            return False, f"symlink:{type(e).__name__}"
        return True, ""
    finally:
        _release_part_lock(lock_fh, blob_lock)


def _parse_content_range(
    cr: str,
) -> tuple[int | None, int | None, int | None]:
    """Parse ``Content-Range: bytes <first>-<last>/<total>``.

    Returns ``(first, last, total)`` or ``(None, None, None)`` if the
    header is missing, malformed, or uses a non-bytes unit. ``total``
    may be ``None`` if the server sent ``*`` for unknown length but
    the spec format was otherwise valid. Codex round-7 BLOCKING #1.
    """
    if not cr or not cr.startswith("bytes "):
        return None, None, None
    spec = cr[len("bytes ") :]
    range_part, sep, total_part = spec.partition("/")
    if not sep:
        return None, None, None
    first_str, dash, last_str = range_part.partition("-")
    if not dash:
        return None, None, None
    try:
        first = int(first_str)
        last = int(last_str)
    except ValueError:
        return None, None, None
    if first < 0 or last < first:
        return None, None, None
    total: int | None
    if total_part == "*":
        total = None
    else:
        try:
            total = int(total_part)
        except ValueError:
            return None, None, None
        if total < 0:
            return None, None, None
    return first, last, total


def _safe_unlink(path: Path) -> None:
    """Best-effort unlink that also removes broken symlinks.

    Codex round-13 BLOCKING #2: ``Path.exists()`` returns ``False`` for
    a broken symlink (the link target doesn't exist), so the old
    ``if path.exists(): path.unlink()`` guard would silently *skip*
    broken-symlink targets — leaving the dangling link in place. A
    later ``tmp.rename(target)`` would then fail (``ENOTDIR`` /
    ``EEXIST`` depending on platform) and force the whole pull to
    fall back unnecessarily. Use ``unlink(missing_ok=True)``: it
    handles both broken symlinks and truly-missing paths in one call.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _acquire_part_lock(lock_path: Path):
    """Acquire an exclusive advisory lock on ``lock_path``.

    Codex round-11 BLOCKING #2: prevents two concurrent ``rapid-mlx``
    processes from racing on the same ``<file>.rapid-mlx-mirror.part``.
    Uses stdlib ``fcntl.flock`` on posix. ``LOCK_EX`` blocks the
    caller until the holder releases — which is fine since downloads
    are expected to be slow and only one of two competing pulls would
    have made progress anyway.

    Returns the open file handle (caller must release via
    :func:`_release_part_lock`), or ``None`` if locking isn't
    available on this platform (Windows — ``fcntl`` missing).
    Failure modes (lock dir unwritable etc.) also degrade to "no
    lock" rather than aborting the pull.
    """
    try:
        import fcntl  # type: ignore[import-not-found]
    except ImportError:
        # Windows / very old systems — best-effort, no lock. The actual
        # rapid-mlx use case is MLX-only (macOS), so this branch is
        # essentially unreachable in practice.
        return None
    try:
        # Open in append mode so concurrent processes don't truncate.
        fh = open(lock_path, "a+b")
    except OSError:
        return None
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except OSError:
        try:
            fh.close()
        except OSError:
            pass
        return None
    return fh


def _release_part_lock(lock_fh, lock_path: Path) -> None:
    """Release the lock acquired via :func:`_acquire_part_lock`.

    Idempotent. Best-effort — if anything goes wrong (lock file already
    deleted, fh already closed, etc.) we swallow the error rather than
    propagate it past the download's own success/failure signal.

    Codex round-12 BLOCKING: we deliberately do NOT unlink ``lock_path``
    on release. Unlinking would split waiters and new acquirers onto
    different inodes — process A would release+unlink while B's flock
    is still on A's now-deleted inode; C would then open and lock a
    NEW inode at the same path, letting C and B race for the
    ``.part`` file. The lock file is just a sidecar; leaving it on
    disk lets every subsequent acquirer see the same inode, which is
    what ``flock`` actually serializes on. Stale lock files don't
    accumulate in practice — each repo only has a fixed set of
    sibling-paths under ``snapshots/<sha>/``.
    """
    if lock_fh is None:
        return
    try:
        import fcntl  # type: ignore[import-not-found]

        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
    except ImportError:
        pass
    try:
        lock_fh.close()
    except OSError:
        pass
    # NB: ``lock_path`` is intentionally NOT removed (see docstring).
    del lock_path


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


def _safe_display_name(fname: str, max_len: int = 80) -> str:
    """Sanitize a HF-supplied filename for terminal display.

    The catalog and HF model_info pass through filenames from external
    metadata. ``_validate_relative_filename`` already rejects
    path-traversal (``..``, absolute paths) but does NOT strip ANSI /
    control characters — a malicious or accidentally-malformed entry
    like ``evil\x1b[2Jfile.bin`` would clear the terminal when printed
    by the per-file progress line. Filenames used for the actual
    download / on-disk path stay untouched; this transformation only
    applies to display.

    Codex round-2 BLOCKING on PR #657.
    """
    # Strip ASCII controls (0x00-0x1F + DEL) AND Unicode control /
    # format characters — both can drive terminal behavior:
    #   * 0x00-0x1F + 0x7F: standard C0 controls + DEL.
    #   * U+0080-U+009F: C1 controls (```` is CSI — same
    #     terminal-injection vector as ESC[).
    #   * U+200E / U+200F / U+202A-U+202E: bidi overrides — a
    #     ``...‮exe.txt`` filename would render visually as
    #     ``...txt.exe`` and mislead the user.
    # Unicode category ``C*`` covers C0/C1 (Cc), format (Cf — includes
    # bidi marks + zero-width joiners), surrogates (Cs), private-use
    # (Co), unassigned (Cn). Codex round-3 BLOCKING on PR #657.
    cleaned = "".join(c for c in fname if not unicodedata.category(c).startswith("C"))
    if not cleaned:
        cleaned = "<unprintable>"
    if len(cleaned) > max_len:
        # Keep the basename visible — that's the part the user actually
        # recognizes. Truncate the middle. Budget 3 chars for the
        # ellipsis, split the remainder evenly between head and tail.
        budget = max_len - 3
        head_len = budget // 2
        tail_len = budget - head_len
        head = cleaned[:head_len]
        tail = cleaned[-tail_len:] if tail_len else ""
        cleaned = f"{head}...{tail}"
    return cleaned


def download_with_mirror_fallback(
    repo_id: str,
    cache_dir: Path | None = None,
    *,
    revision: str | None = None,
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

    Codex round-9 BLOCKING #2: ``revision`` is reserved for future
    use. Today this function only handles the default branch (HEAD of
    ``main``) — the catalog and the R2 mirror are built from default-
    branch snapshots, and ``refs/main`` is the only ref we write. If a
    caller passes a non-default revision (e.g. ``snapshot_download(...,
    revision="<sha>")``), return False so the caller's
    ``snapshot_download`` runs and pins the right revision instead of us
    silently overwriting ``refs/main`` with HEAD. ``revision=None`` and
    ``revision="main"`` both mean default branch and are accepted.
    """
    base = _mirror_base()
    if not base or "/" not in repo_id:
        # Mirror disabled or repo_id isn't a HF-shaped ``owner/name``.
        # Local paths fall here too. Defer to caller's HF path.
        return False

    # Codex round-9 BLOCKING #2: explicit non-default revision → bail.
    if revision is not None and revision != "main":
        return False

    # HF model_info gives us the canonical revision + per-file sizes.
    # We need both — the revision pins the snapshot dir, and the sizes
    # let us validate R2 responses. Without it we can't pin a revision,
    # which would mean writing files under an unknowable sha — so fall
    # through to HF if this fails.
    #
    # Codex round-6 BLOCKING #1: narrow the exception net. Only swallow
    # expected network / HF API errors; programmer errors (TypeError,
    # AttributeError) and validation errors propagate so misuse
    # surfaces a real stack trace.
    from huggingface_hub import model_info
    from huggingface_hub.errors import (
        EntryNotFoundError,
        HfHubHTTPError,
    )
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = model_info(repo_id, files_metadata=True)
    except (
        EntryNotFoundError,
        RepositoryNotFoundError,
        HfHubHTTPError,
        OSError,
        TimeoutError,
    ):
        return False

    # Reuse the ``revision`` name for the resolved SHA — by now the
    # input parameter has already been validated above (``None`` or
    # ``"main"``); from here on ``revision`` always means the concrete
    # commit hash we'll write under ``snapshots/<sha>/``.
    revision = getattr(info, "sha", None)
    siblings = getattr(info, "siblings", None) or []
    # Each file: (relative_path, expected_size, lfs_sha256).
    # - ``expected_size`` from HF's siblings metadata (None if HF didn't
    #   expose it; use ``size is not None`` to distinguish 0 from
    #   unknown).
    # - ``lfs_sha256`` only present for LFS-tracked files (the big
    #   weight shards). Code paths that need stronger-than-size
    #   integrity check the hash; non-LFS files (small text assets) keep
    #   the size-only check. Codex round-5 BLOCKING #1.
    files: list[tuple[str, int | None, str | None]] = []
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
        size = size if isinstance(size, int) else None
        lfs = getattr(s, "lfs", None)
        sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        sha256 = sha256 if isinstance(sha256, str) and len(sha256) == 64 else None
        files.append((rname, size, sha256))
    if not revision or not files:
        return False

    cache_root = cache_dir if cache_dir else _hf_cache_root()
    owner, _, repo = repo_id.partition("/")
    repo_root = cache_root / f"models--{owner}--{repo}"
    snap_dir = repo_root / "snapshots" / revision
    refs_dir = repo_root / "refs"
    # Codex round-14 BLOCKING #1+#2: keep ``.part`` and ``.lock``
    # sidecars OUT of ``snapshots/<sha>/`` so they can't collide with
    # legitimate repo assets named like our temp files. The leading
    # ``.rapid-mlx-mirror`` directory is namespaced under the repo
    # root, so it shares the lifecycle of the cached model but never
    # mingles with HF's own snapshot files.
    sidecar_dir = repo_root / ".rapid-mlx-mirror"
    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    # Catalog lookup — for the project default mirror, the catalog
    # confirms whether the alias is mirrored and gives us
    # ``download_url_base``. For a custom mirror (user set
    # ``RAPID_MLX_MODEL_MIRROR=<other URL>``), the catalog endpoint may
    # not exist — fall back to the direct-layout convention
    # (``<base>/<owner>/<repo>/<file>``) that PR #647 introduced.
    #
    # Codex round-6 NIT #3: distinguish "catalog endpoint not here"
    # (404 on custom mirror → use direct-layout) from "transient or
    # malformed" (5xx, network error, bad JSON → use HF only). A
    # misconfigured custom mirror returning 500 would otherwise eat
    # ``ceil(files / workers) * 60s`` of per-file timeouts.
    catalog, catalog_status = fetch_catalog_with_status(base)
    catalog_entry: dict[str, Any] | None = None
    catalog_mirrored = False
    if catalog is not None:
        catalog_entry = find_catalog_entry(catalog, repo_id)
        catalog_mirrored = catalog_entry is not None and _is_mirrored(catalog_entry)

    # Decide download_url_base + whether to try R2:
    #
    # * Catalog says mirrored → use the catalog's download_url_base.
    # * Catalog says NOT mirrored → skip R2 entirely (the catalog is
    #   authoritative for the project default mirror).
    # * Catalog absent on the DEFAULT mirror → catalog endpoint is
    #   advertised, so an outage is a real signal; route straight to
    #   HF instead of incurring up to ``ceil(files / workers) * 60s``
    #   of mirror timeouts (codex round-4 BLOCKING #3).
    # * Catalog 4xx on a CUSTOM mirror → "no catalog endpoint here";
    #   fall back to direct layout (PR #647 contract). Codex round-10
    #   BLOCKING: include 400 / 401 / 403 / 404 — many static-bucket
    #   mirrors (S3 with list-bucket denied, vanilla nginx + a 403
    #   error_page, CDN 400 on path-style requests, etc.) don't return
    #   exactly 404 for unknown paths. ANY 4xx on a custom mirror means
    #   "no JSON catalog here, try the legacy layout instead", which is
    #   what PR #647 users have been relying on. Per-file 4xx still
    #   falls back to HF.
    # * Catalog 5xx / network / malformed on a CUSTOM mirror →
    #   transient or misconfigured; skip direct-layout and use HF.
    dub = ""
    use_r2 = False
    is_default_mirror = base.rstrip("/") == MIRROR_DEFAULT.rstrip("/")
    catalog_is_4xx = catalog_status is not None and 400 <= catalog_status < 500
    if catalog_mirrored and catalog_entry is not None:
        dub = str(catalog_entry.get("download_url_base", "")).strip()
        use_r2 = bool(dub)
    elif catalog is None and not is_default_mirror and catalog_is_4xx:
        # Custom mirror without a usable /api/models — try direct layout.
        # The PR #647 contract was ``<base>/<owner>/<repo>/<file>``.
        dub = f"/{owner}/{repo}/"
        use_r2 = True

    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    BOLD = "\x1b[1m" if is_tty else ""
    DIM = "\x1b[2m" if is_tty else ""
    RESET = "\x1b[0m" if is_tty else ""

    if use_r2 and catalog_mirrored:
        _print_dim(
            f"  {BOLD}Pulling {repo_id}{RESET} {DIM}(R2 mirror, fallback: HF){RESET}"
        )
    elif use_r2:
        # Catalog unreachable (custom mirror or transient outage); we
        # still try direct ``<base>/<owner>/<repo>/<file>`` URLs and
        # fall back to HF per file on 404. Preserves PR #647's contract
        # for non-default ``RAPID_MLX_MODEL_MIRROR`` URLs.
        _print_dim(
            f"  {BOLD}Pulling {repo_id}{RESET} {DIM}(mirror direct-layout, "
            f"fallback: HF){RESET}"
        )
    else:
        # Mirror is skipped wholesale (catalog says not mirrored, OR
        # default-mirror catalog 5xx/network/parse failure, OR custom
        # mirror catalog 5xx). Codex round-8 BLOCKING #1+#2: do NOT
        # impersonate ``snapshot_download`` with per-file
        # ``hf_hub_download`` calls — return False so the caller invokes
        # the real ``snapshot_download`` and gets its allow/ignore
        # patterns, retries, and repository-level error reporting.
        return False

    # Issue #651 follow-up: surface the per-file work plan up front so
    # the user sees activity instead of staring at the banner for
    # minutes while multi-GB shards stream. Per-file completion lines
    # are emitted in the ``as_completed`` loop below.
    total_files_planned = len(files)
    total_expected_bytes = sum(s for _, s, _ in files if s is not None)
    if total_expected_bytes > 0:
        gb = total_expected_bytes / 1e9
        _print_dim(
            f"  {DIM}Found {total_files_planned} files (~{gb:.1f} GB total){RESET}"
        )
    else:
        _print_dim(f"  {DIM}Found {total_files_planned} files{RESET}")

    # Per-file plan: for each file, attempt R2 first (if eligible),
    # otherwise fall straight to HF. Run a small pool in parallel.
    r2_hits = 0
    hf_hits = 0
    misses: list[str] = []
    total_bytes = 0

    def _do_file(
        item: tuple[str, int | None, str | None],
    ) -> tuple[str, str, int]:
        fname, expected_size, expected_sha256 = item
        target = snap_dir / fname
        # Belt-and-braces: normalize against snap_dir to refuse symlink
        # or normpath escapes the parts check missed.
        try:
            target_norm = Path(os.path.normpath(str(target)))
            snap_norm = Path(os.path.normpath(str(snap_dir)))
            target_norm.relative_to(snap_norm)
        except ValueError:
            return fname, "skip-traversal", 0

        # Codex round-7 BLOCKING #2: ``relative_to`` only checks the
        # NORMALIZED string path — it doesn't notice that a parent
        # component under ``snap_dir`` may be a SYMLINK pointing
        # outside the snapshot. A malicious or accidental symlink at
        # ``snapshots/<sha>/subdir → /etc`` would let us write to
        # ``/etc/<basename>``. Walk every parent between snap_dir and
        # target and reject if any is a symlink. ``snap_dir`` itself
        # could in principle be a symlink (HF cache layouts do
        # symlink across drives), but everything *inside* it must be a
        # real directory.
        try:
            # Codex round-14 BLOCKING #4: also check ``snap_dir`` itself.
            # A pre-existing ``snapshots/<sha>`` symlink to ``/etc/`` (or
            # any other location) would make every write inside this
            # function escape the HF cache despite the parent walk —
            # the walk skips ``snap_dir`` because it's the loop's
            # terminator. Reject up front if it's a symlink, since
            # legitimate HF caches use real directories at this level.
            if snap_dir.is_symlink():
                return fname, "skip-symlink-snapdir", 0
            parent = target.parent
            # Iterate parents strictly between snap_dir and target.
            while parent != snap_dir and snap_dir in parent.parents:
                if parent.is_symlink():
                    return fname, "skip-symlink-parent", 0
                parent = parent.parent
        except OSError:
            # Permission denied / unusual fs — refuse to write here.
            return fname, "skip-stat", 0

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
        #
        # Codex round-5 NIT #4: ``if expected_size`` treated a
        # legitimate 0-byte file as "size unknown" → use
        # ``is not None`` so empty files still get their size check.
        try:
            if target.is_dir() and not target.is_symlink():
                return fname, "miss", 0
            # Codex round-13 BLOCKING #1: a symlink at the target path
            # pointing OUTSIDE the repo's cache dir (e.g. ``snapshots/
            # <sha>/foo -> /etc/passwd``) would be ``stat()``-ed
            # through, and on the absurd off-chance the destination
            # matches expected_size + sha256 (or HF didn't tell us a
            # sha) we'd "accept" it as cached and pin ``refs/main`` to
            # a malicious-looking snapshot. HF caches legitimately use
            # symlinks (``snapshots/<sha>/foo -> ../../blobs/<hash>``)
            # so we can't blanket-reject them. Instead, resolve the
            # symlink and refuse anything that escapes ``repo_root``.
            if target.is_symlink():
                try:
                    resolved = target.resolve(strict=False)
                    # Codex round-14 BLOCKING #3: tighten the symlink
                    # acceptance window. Round-13 accepted anywhere
                    # under ``repo_root``, which still leaves a tiny
                    # internal-cache attack surface — e.g. a symlink
                    # pointing at ``refs/main`` (40 ASCII bytes) or
                    # another snapshot's file. Real HF cache symlinks
                    # ALWAYS point under ``repo_root/blobs/<hash>``,
                    # so restrict to exactly that subtree.
                    blobs_root_resolved = (repo_root / "blobs").resolve(strict=False)
                except OSError:
                    _safe_unlink(target)
                    raise  # caught by outer OSError handler below
                try:
                    resolved.relative_to(blobs_root_resolved)
                except ValueError:
                    # Symlink escapes the blobs/ store → malicious or
                    # accidentally-misplaced. Drop and refetch.
                    _safe_unlink(target)
                    # Don't try the rest of the cached-checks on a
                    # deleted target — fall through to R2/HF.
                # else: legit HF blob symlink, fall through to the
                # size/sha check below.
            if target.exists():
                cached_size = target.stat().st_size
                if expected_size is not None and cached_size != expected_size:
                    # Stale / truncated cache entry — drop it and fall
                    # through to the R2/HF re-fetch below.
                    _safe_unlink(target)
                elif expected_size is not None and cached_size == expected_size:
                    # Codex round-11 BLOCKING #1: size-only acceptance
                    # of cached LFS files lets a same-size corrupt or
                    # stale weight bypass the SHA-256 integrity check
                    # the rest of the pipeline enforces. When HF told us
                    # an LFS sha256, hash the cached bytes too — if it
                    # mismatches, drop the file and refetch. For non-LFS
                    # files (no sha256), size remains the strongest
                    # check we have, which is fine for tiny configs.
                    #
                    # Codex round-14 NIT #5: re-hashing every cached LFS
                    # shard on a warm pull turns a no-op into a full
                    # disk scan of the model (10s of GB). HuggingFace's
                    # cache layout names blob files by their sha256
                    # (``blobs/<hex>``), so if our target is a symlink
                    # pointing at ``blobs/<expected_sha256>`` we already
                    # know the bytes match — skip the rehash.
                    if expected_sha256 is not None:
                        if target.is_symlink():
                            try:
                                blob_name = target.resolve(strict=False).name
                            except OSError:
                                blob_name = ""
                            if blob_name == expected_sha256:
                                return fname, "cached", cached_size
                            # Symlink name doesn't match HF's blob hash
                            # convention — fall through to a full
                            # rehash to be safe.
                        import hashlib

                        hasher = hashlib.sha256()
                        try:
                            with target.open("rb") as fh:
                                while True:
                                    blk = fh.read(_CHUNK_BYTES)
                                    if not blk:
                                        break
                                    hasher.update(blk)
                        except OSError:
                            _safe_unlink(target)
                        else:
                            if hasher.hexdigest() == expected_sha256:
                                return fname, "cached", cached_size
                            # Stale / corrupted cache — drop + refetch.
                            _safe_unlink(target)
                    else:
                        return fname, "cached", cached_size
                elif expected_size is None and cached_size > 0:
                    # HF didn't expose a size — accept any non-empty
                    # file as cached (matches pre-#650 behavior).
                    return fname, "cached", cached_size
        except OSError:
            # Target is a broken symlink / permission denied / etc.
            # Try to remove it (best-effort) and fall through. The
            # rename in ``_download_one_from_r2`` will then place a
            # fresh file at the path.
            _safe_unlink(target)

        if use_r2:
            # ``dub`` is set above to either the catalog's
            # download_url_base (default mirror) or the synthetic
            # ``/<owner>/<repo>/`` (custom mirror / catalog absent).
            url = _build_r2_url(base, dub, fname)
            ok, _reason = _download_one_from_r2(
                url,
                target,
                expected_size,
                expected_sha256=expected_sha256,
                sidecar_dir=sidecar_dir,
                sidecar_key=_sidecar_key_for(fname),
                repo_root=repo_root,
            )
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
    #
    # ``_hf_fallback_one`` already converts the expected HF surface
    # (``EntryNotFoundError``, ``RepositoryNotFoundError``,
    # ``HfHubHTTPError``, ``OSError``, ``TimeoutError``) to ``(False,
    # None)`` internally — this except clause is a belt-and-braces in
    # case a future refactor leaks one of those out of a worker. Codex
    # round-8 NIT #3: keep this set in sync with ``_hf_fallback_one``.
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
    from huggingface_hub.utils import RepositoryNotFoundError

    # Issue #651 follow-up: track elapsed wall-time for the final
    # summary so users see throughput, not just total bytes.
    pull_started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_do_file, item): item[0] for item in files}
        for fut in as_completed(futures):
            fname = futures[fut]
            # Codex round-3 NIT #1: narrow the worker exception net.
            # Only convert expected network / filesystem / HF errors into
            # a silent "miss". Programmer errors (TypeError, etc.) and
            # HF validation errors propagate so misuse surfaces a real
            # stack trace instead of disappearing into the fallback.
            try:
                _, kind, size = fut.result()
            except (
                OSError,
                TimeoutError,
                urllib.error.URLError,
                urllib.error.HTTPError,
                EntryNotFoundError,
                RepositoryNotFoundError,
                HfHubHTTPError,
            ):
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
            # Issue #651 follow-up: per-file completion line so the
            # user sees forward progress while multi-GB shards stream.
            # We emit one line per file at the point it lands — printing
            # from the ``as_completed`` loop in the main thread avoids
            # needing a stdout lock, even though the underlying workers
            # finish in non-deterministic order. Misses are logged too
            # so the user understands why we'll fall back to
            # ``snapshot_download`` further down.
            completed += 1
            # Only the size-bearing kinds need a MB readout. ``size``
            # is always an int (workers return 0 on miss) but keeping
            # the divide inside the branches that use it makes it
            # obvious that the miss tag never depends on bytes — codex
            # round-1 NIT on PR #657.
            if kind == "r2":
                tag = f"{DIM}R2 ({size / 1e6:.0f} MB){RESET}"
            elif kind == "hf":
                tag = f"{DIM}HF ({size / 1e6:.0f} MB, fallback){RESET}"
            elif kind == "cached":
                tag = f"{DIM}cached ({size / 1e6:.0f} MB){RESET}"
            else:
                # ``miss`` / sanitized failure — surface the reason so
                # users aren't surprised when the outer caller falls
                # back to ``snapshot_download``.
                tag = f"{DIM}miss (will retry via HF snapshot_download){RESET}"
            _print_dim(
                f"  {DIM}[{completed}/{total_files_planned}]{RESET} "
                f"{_safe_display_name(fname)} {tag}"
            )

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
        # Codex round-13 NIT #3: write the ref in deterministic UTF-8
        # rather than the platform default encoding. SHA hashes are
        # ASCII so the bytes are the same in practice, but matching
        # the HF cache writer's encoding keeps cross-platform reads
        # bit-identical.
        (refs_dir / "main").write_text(revision, encoding="utf-8")
    except OSError:
        return False

    mb = total_bytes / 1e6
    # Issue #651 follow-up: surface elapsed time + throughput so users
    # can sanity-check link speed. Skip the elapsed suffix on warm
    # cached pulls where it's just noise (sub-second), and on tiny
    # downloads where the rate is meaningless. We can't perfectly
    # separate cached-bytes from fetched-bytes (workers report kind +
    # size but ``total_bytes`` aggregates both), so the displayed rate
    # is approximate for mixed runs — the user-visible problem in
    # issue #651 was multi-GB cold pulls, where cached_hits is 0 and
    # the rate is exact.
    elapsed = max(0.0, time.monotonic() - pull_started)
    suffix = ""
    if (r2_hits or hf_hits) and elapsed > 0.5:
        rate_mbps = mb / elapsed
        suffix = f" {DIM}in {elapsed:.0f}s ({rate_mbps:.0f} MB/s){RESET}"
    if r2_hits and hf_hits:
        _print_dim(
            f"  {BOLD}Pulled{RESET} {len(files)} files, {mb:.0f} MB "
            f"{DIM}(R2: {r2_hits}, HF: {hf_hits}){RESET}{suffix}"
        )
    elif r2_hits:
        _print_dim(
            f"  {BOLD}Pulled{RESET} {len(files)} files, {mb:.0f} MB "
            f"{DIM}(R2: {r2_hits}){RESET}{suffix}"
        )
    elif hf_hits:
        _print_dim(
            f"  {BOLD}Pulled{RESET} {len(files)} files, {mb:.0f} MB "
            f"{DIM}(HF: {hf_hits}){RESET}{suffix}"
        )
    else:
        # All files were already cached — quiet success.
        _print_dim(f"  {BOLD}Already cached{RESET} ({len(files)} files, {mb:.0f} MB)")
    return True
