"""Mirror a HuggingFace model repo into the rapid-mlx Cloudflare R2 bucket.

This tool is the persisted, in-repo counterpart to the ad-hoc uploads that
seeded ``https://models.rapidmlx.com``. Client-side download uses
``vllm_mlx/_mirror.py``'s ``download_with_mirror_fallback`` which expects
the bucket to hold objects at ``<hf-owner>/<hf-repo>/<filename>`` — the
exact key layout this script writes.

Design (see PR body for full context):

* **Streaming per-file discipline.** For each file in the HF repo we:
  1. HEAD-check R2. If the key already exists with a matching size, SKIP.
  2. Otherwise download the single file to ``<tmp>/<filename>``,
  3. upload it to R2 via boto3 (``s3_client.upload_file`` — which auto-
     splits into multipart uploads at the profile's 64 MB threshold), and
  4. delete the local file BEFORE moving on to the next.

  Peak local disk usage stays under one shard size (~5 GB for typical
  safetensors). Guardrail G11 (100 GB free floor) is respected.

* **boto3 multipart via SDK, not hand-rolled.** ``upload_file`` reads the
  configured ``s3.multipart_threshold`` / ``s3.multipart_chunksize`` from
  the AWS profile (``r2`` profile: 64 MB / 64 MB) and does the multipart
  dance itself. The old ``wrangler r2 put`` path has a 300 MiB cap — this
  script does not use it.

* **Content-Type routing.** ``.json`` → ``application/json``,
  ``.md`` → ``text/markdown``, everything else →
  ``application/octet-stream``. Deterministic + auditable.

* **Resumability via HEAD.** A run that dies mid-repo can be re-run and
  will resume — every file that already exists on R2 with the expected
  size is SKIPped.

* **Fail-fast.** Any upload failure exits nonzero after cleaning the tmp
  dir. We do not silently continue past a broken file.

* **Verification pass.** After uploads, HEAD every expected key on R2 AND
  GET each file via ``https://models.rapidmlx.com/<key>`` to confirm the
  object is publicly readable. Any HTTP-not-200 fails the run.

CLI:

    python scripts/mirror_to_r2.py <hf-repo-id> \\
        [--endpoint-url URL] [--bucket BUCKET] [--profile r2] \\
        [--dry-run] [--verify-only] [--tmp-dir DIR]

Defaults match the production R2 config
(``https://f25478810829faf5ccc86f4ed9a96ef1.r2.cloudflarestorage.com``,
bucket ``rapid-mlx-models``, profile ``r2``).

Install with ``pip install -e .[mirror]`` to pull ``boto3``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Public defaults — persisted so a fresh operator can invoke the tool
# without hunting for the endpoint URL / bucket name.
DEFAULT_ENDPOINT_URL = (
    "https://f25478810829faf5ccc86f4ed9a96ef1.r2.cloudflarestorage.com"
)
DEFAULT_BUCKET = "rapid-mlx-models"
DEFAULT_PROFILE = "r2"
DEFAULT_PUBLIC_BASE = "https://models.rapidmlx.com"

# Cloudflare 403s vanilla ``Python-urllib/*``; use a plausible UA that
# also identifies the tool for R2 log grep.
_USER_AGENT = "Mozilla/5.0 (rapid-mlx mirror uploader)"


def content_type_for(filename: str) -> str:
    """Route filename → Content-Type. Deterministic, three-branch.

    * ``.json`` → ``application/json``
    * ``.md`` → ``text/markdown``
    * everything else → ``application/octet-stream``

    Rationale: existing keys on R2 are ``application/octet-stream``, but
    for future-legibility (curl in a browser, HF-style client sniffing)
    the two textual formats we ship in every repo get their real type.
    Anything else — safetensors, tokenizer.model binaries, images — is
    correctly opaque.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".json":
        return "application/json"
    if ext == ".md":
        return "text/markdown"
    return "application/octet-stream"


@dataclass
class FileMeta:
    """One expected file in the HF repo → R2 mapping.

    ``size`` is ``None`` when HF's siblings metadata didn't expose one
    (some older repos, older HF API versions) — distinct from ``0``,
    which is a legitimate empty file. The distinction matters for the
    skip decision and for the verification path (codex round-2 BLOCKING
    #2 and #3).
    """

    relpath: str  # HF sibling rfilename
    size: int | None  # bytes; None if HF didn't expose it; 0 = real empty
    key: str  # R2 object key = ``<hf-repo-id>/<relpath>``
    lfs_sha256: str | None = None  # HF's LFS sha256 for weight shards; None otherwise


def _hf_files(repo_id: str) -> list[FileMeta]:
    """Enumerate expected files via HF ``model_info(files_metadata=True)``.

    Returns a list of ``FileMeta`` — one per sibling. Fails hard on any
    HF-side error (fail-fast per G8 root-cause discipline).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    files: list[FileMeta] = []
    for s in info.siblings or []:
        rname = getattr(s, "rfilename", None)
        if not rname:
            continue
        # Sanity: reject any path-traversal-shaped filename before we
        # write it to disk. HF's own upload path enforces this too, but
        # a compromised repo could still ship one.
        if rname.startswith("/") or ".." in Path(rname).parts:
            raise ValueError(f"Refusing suspicious HF filename: {rname!r}")
        # Codex round-2 BLOCKING #2: preserve the distinction between
        # "HF didn't expose a size" (None) and "the file is legitimately
        # 0 bytes" (0). Passing `or 0` collapses both to 0 and forces a
        # perpetual re-upload of every empty file.
        raw_size = getattr(s, "size", None)
        size = int(raw_size) if isinstance(raw_size, int) else None
        # LFS-tracked files (weight shards) expose a canonical sha256
        # under ``s.lfs.sha256``. We stash it as R2 object metadata so
        # ``should_skip`` can require both size AND sha match — a stale/
        # corrupt R2 object with the same byte length can't sneak past.
        lfs = getattr(s, "lfs", None)
        lfs_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        if not (isinstance(lfs_sha256, str) and len(lfs_sha256) == 64):
            lfs_sha256 = None
        key = f"{repo_id}/{rname}"
        files.append(FileMeta(relpath=rname, size=size, key=key, lfs_sha256=lfs_sha256))
    return files


# Files that live at the repo root and belong with EVERY quantisation
# subfolder. A subfolder holds weights and tokenizer; provenance and terms
# sit at the top level, and a mirror that drops them hands users a copy of
# the weights with no statement of what they may do with them.
#
# Matched by STEM, case-insensitively, so a repo shipping ``NOTICE.md``,
# ``LICENSE-MODEL`` or ``license.txt`` keeps its terms too. An exact-name
# set silently dropped those while still reporting a successful mirror,
# which is the worst possible way to get redistribution wrong (review
# MINOR).
_ROOT_KEEP_STEMS = ("license", "notice", "readme", "copying")


def _is_root_keep(relpath: str) -> bool:
    """True for a repo-root file that ships with every quantisation."""
    lowered = relpath.lower()
    return any(
        lowered == stem
        or lowered.startswith(stem + ".")
        or lowered.startswith(stem + "-")
        for stem in _ROOT_KEEP_STEMS
    )


# A quantisation subfolder must actually contain a loadable checkpoint.
# Without this, ``--subfolder docs`` uploads a documentation directory,
# verifies the objects it just wrote, and reports success — reproducing
# the "mirror completed but pull still 404s" failure this flag exists to
# prevent (raised in review).
_WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin", ".gguf")


def _select_subfolder(files: list[FileMeta], subfolder: str) -> list[FileMeta]:
    """Narrow ``files`` to one quantisation subfolder plus repo-root metadata.

    Some upstreams now ship every quantisation of a model in ONE repo, as
    sibling directories::

        LiquidAI/LFM2.5-2.6B-MLX/
          LICENSE  README.md  .gitattributes
          4bit/  5bit/  6bit/  8bit/  bf16/  mxfp4/  mxfp8/  nvfp4/

    Mirroring such a repo wholesale copies ~20 GB to serve the 1.6 GB anyone
    actually pulls. Every other repo we mirror is one-quantisation-per-repo,
    where "the repo" and "what we serve" are the same thing — which is why
    this selector did not exist until the first subfolder repo showed up.

    Keys keep their full in-repo path (``…-MLX/4bit/config.json``), so the
    object layout on R2 stays a faithful copy of the source tree and the
    alias resolver can point at the subfolder directly.
    """
    stripped = subfolder.strip("/")
    if not stripped:
        raise ValueError(
            f"--subfolder {subfolder!r} is empty. Pass a real subfolder name, "
            "or omit the flag entirely to mirror the whole repo."
        )

    prefix = stripped + "/"
    selected = [f for f in files if f.relpath.startswith(prefix)]
    available = sorted({f.relpath.split("/")[0] for f in files if "/" in f.relpath})
    if not selected:
        raise ValueError(
            f"--subfolder {subfolder!r} matched no files. "
            f"Available subfolders: {available or '(none — flat repo)'}"
        )

    # A directory that exists is not necessarily a checkpoint. Require the
    # two things a loader cannot start without, as DIRECT children of the
    # subfolder: a config and at least one weight shard.
    direct = {f.relpath[len(prefix) :] for f in selected}
    direct = {name for name in direct if "/" not in name}
    if "config.json" not in direct or not any(
        name.endswith(_WEIGHT_SUFFIXES) for name in direct
    ):
        raise ValueError(
            f"--subfolder {subfolder!r} does not look like a checkpoint: "
            f"expected config.json and a weight file ({', '.join(_WEIGHT_SUFFIXES)}) "
            f"directly inside it, found {sorted(direct)[:8]}. "
            f"Available subfolders: {available or '(none — flat repo)'}"
        )

    roots = [f for f in files if "/" not in f.relpath and _is_root_keep(f.relpath)]
    return roots + selected


def _r2_client(endpoint_url: str, profile: str) -> Any:
    """Build a boto3 S3 client bound to the R2 endpoint / profile.

    ``addressing_style="path"`` mirrors the AWS profile config and is
    what Cloudflare's S3-compat surface expects.
    """
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile)
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        config=Config(s3={"addressing_style": "path"}),
    )


def _r2_head_size(client: Any, bucket: str, key: str) -> int | None:
    """Return the ``Content-Length`` of a key, or ``None`` on 404.

    Any non-404 error raises (fail-fast — we want the operator to see
    permission / endpoint issues immediately rather than skipping past
    them).

    Kept for the unit tests + external callers that only care about
    size. Callers that need the full HEAD response (including x-amz-meta
    fields for sha checking) should use :func:`_r2_head`.
    """
    r = _r2_head(client, bucket, key)
    if r is None:
        return None
    return int(r["ContentLength"])


def _r2_head(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    """Return the raw ``head_object`` response, or ``None`` on 404.

    Any non-404 error raises. The returned dict includes ``ContentLength``
    and ``Metadata`` (a lowercase-keyed dict of x-amz-meta-* headers) —
    enough to run both a size AND checksum check on skip.
    """
    from botocore.exceptions import ClientError

    try:
        return client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        # Boto raises "404" (string) for HEAD-missing objects.
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def should_skip(
    existing_size: int | None,
    expected_size: int | None,
    *,
    existing_sha256: str | None = None,
    expected_sha256: str | None = None,
) -> bool:
    """Skip decision: same size AND (when known) same sha256 = same content.

    Codex round-1 BLOCKING #1: matching byte length alone can be fooled
    by a stale/corrupt R2 object that happens to share the file size
    (e.g. a previous upload from a different repo revision, or a mid-
    stream truncation that boto3's post-flight ``ContentLength`` still
    reports as full). Whenever HF exposes an LFS sha256 (all weight
    shards) we require BOTH size + sha match on the ``x-amz-meta-hf-sha256``
    metadata field we set at upload time.

    Codex round-2 BLOCKING #2: ``expected_size`` is ``None`` when HF
    didn't expose a size, distinct from ``0`` (a legitimate empty
    file). Previously both collapsed to a "must upload" branch; empty
    files were re-uploaded every run despite matching R2 state.

    Behavior matrix:

    * ``existing_size is None`` (R2 HEAD → 404): must upload.
    * ``expected_size is None`` (HF didn't tell us): must upload —
      refuse to trust an R2 short-write when we can't validate length.
    * ``existing_size != expected_size``: must upload.
    * ``expected_sha256`` present but ``existing_sha256`` missing or
      mismatched: must upload — the R2 object is either an older upload
      pre-metadata, or a different bytes-with-same-length case.
    * ``expected_sha256 is None`` (non-LFS: config.json, README.md,
      empty files, etc.): fall back to size-only. Small text files
      don't get LFS hashes; the practical risk of a same-size-but-
      corrupt copy is much lower for a 1 KB config than for a 5 GB
      shard, and forcing users to re-upload every tiny asset on each
      pass defeats the resumability contract.
    """
    if existing_size is None:
        return False
    if expected_size is None:
        return False
    if existing_size != expected_size:
        return False
    # Size matches. When HF told us an LFS sha, demand parity — an R2
    # object without our sha metadata is stale (from a pre-metadata
    # upload) and must be re-uploaded to earn the skip.
    if expected_sha256 is not None:
        return existing_sha256 == expected_sha256
    return True


def _upload_one(
    client: Any,
    local_path: Path,
    bucket: str,
    key: str,
    content_type: str,
    *,
    lfs_sha256: str | None = None,
) -> None:
    """Upload one file to R2 with boto3 multipart auto-split.

    boto3's ``upload_file`` reads ``s3.multipart_threshold`` and
    ``s3.multipart_chunksize`` from the profile config — the ``r2``
    profile sets both to 64 MB. We do NOT reimplement multipart
    ourselves; the SDK handles ``create_multipart_upload`` /
    ``upload_part`` / ``complete_multipart_upload`` including retries.

    When ``lfs_sha256`` is provided (weight shards), it is stored as
    ``x-amz-meta-hf-sha256`` object metadata so a subsequent
    :func:`should_skip` check can require both size AND sha parity
    (codex round-1 BLOCKING #1).
    """
    extra: dict[str, Any] = {"ContentType": content_type}
    if lfs_sha256:
        extra["Metadata"] = {"hf-sha256": lfs_sha256}
    client.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
        ExtraArgs=extra,
    )


def _download_one_hf(repo_id: str, relpath: str, tmp_dir: Path) -> Path:
    """Download a single HF file into ``tmp_dir`` (flat, no snapshot layout).

    Passing ``local_dir=tmp_dir`` puts the file directly under ``tmp_dir``
    as a real file — modern ``huggingface_hub`` (>=0.23) writes directly
    to ``local_dir`` without the legacy symlink-through-cache behavior,
    so we can delete the file after upload without touching the
    operator's shared HF cache — G1 respect. Codex round-3 NIT: earlier
    revisions of this docstring named a ``local_dir_use_symlinks=False``
    kwarg that was removed upstream; the behavior is now the default.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=relpath,
            local_dir=str(tmp_dir),
        )
    )


def _cleanup_local(path: Path) -> None:
    """Best-effort delete of a local file. Never fatal."""
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _public_url(public_base: str, key: str) -> str:
    """Build the public read URL for an R2 key.

    Codex round-1 BLOCKING #2: percent-encode each path segment before
    joining. Filenames legitimately containing spaces, ``#``, ``?``,
    ``%``, or other URL-reserved chars would otherwise turn a valid
    ``mlx-community/repo/model card.md`` key into an unreachable URL
    (space breaks the path; ``#`` starts a fragment). Encoding per
    segment (not whole key) preserves the ``/`` separators. Matches the
    same discipline in ``vllm_mlx/_mirror.py::_build_r2_url``.
    """
    encoded = "/".join(
        urllib.parse.quote(seg, safe="") for seg in key.lstrip("/").split("/") if seg
    )
    return f"{public_base.rstrip('/')}/{encoded}"


def _http_head_status(url: str, timeout: float = 30.0) -> int:
    """HEAD ``url`` and return the HTTP status code.

    Uses stdlib urllib to avoid a hard dep on ``requests``. Cloudflare's
    edge sometimes rejects HEAD; fall through to a byte-range GET on any
    non-2xx status to distinguish "HEAD blocked" from "object missing".
    """
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


def _http_range_get_status(url: str, timeout: float = 30.0) -> int:
    """Range-GET the first byte of ``url`` — proves the object is fetchable.

    Cheaper than a full GET and works when HEAD is rejected by an edge
    rule. Used as the definitive "is this publicly readable?" check.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read(1)
            # 200 (server ignored Range) or 206 (Range honored) both mean
            # the object is publicly reachable.
            status = int(resp.status)
            return 200 if status in (200, 206) else status
    except urllib.error.HTTPError as e:
        return int(e.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


# ------------------- top-level flow -------------------


def mirror_repo(
    repo_id: str,
    *,
    endpoint_url: str = DEFAULT_ENDPOINT_URL,
    bucket: str = DEFAULT_BUCKET,
    profile: str = DEFAULT_PROFILE,
    public_base: str = DEFAULT_PUBLIC_BASE,
    dry_run: bool = False,
    verify_only: bool = False,
    tmp_dir: Path | None = None,
    subfolder: str | None = None,
) -> int:
    """Mirror one HF repo to R2. Return process exit code (0 = ok).

    ``subfolder`` narrows a multi-quantisation repo to one variant; see
    ``_select_subfolder``. Omitted, the whole repo is mirrored, which is
    correct for the one-quantisation-per-repo layout every other upstream
    we mirror uses.
    """
    started = time.monotonic()
    print(f"== mirror {repo_id} → r2://{bucket}/{repo_id}/ ==", flush=True)
    print(f"   endpoint: {endpoint_url}", flush=True)
    print(f"   profile:  {profile}", flush=True)
    if dry_run:
        print("   MODE:     dry-run (no uploads)", flush=True)
    if verify_only:
        print("   MODE:     verify-only (no uploads)", flush=True)

    files = _hf_files(repo_id)
    # ``is not None``, not truthiness: ``--subfolder ""`` (an unset shell
    # variable expanding to nothing) is a caller mistake, and treating it
    # as "no filter" silently mirrors the full 20 GB repo the flag exists
    # to avoid. _select_subfolder rejects it (raised in review).
    if subfolder is not None:
        before = len(files)
        before_bytes = sum((f.size or 0) for f in files)
        files = _select_subfolder(files, subfolder)
        kept_bytes = sum((f.size or 0) for f in files)
        print(
            f"   subfolder: {subfolder!r} — {len(files)}/{before} files, "
            f"{kept_bytes / 1e9:.3f} of {before_bytes / 1e9:.3f} GB",
            flush=True,
        )
    # HF sometimes doesn't expose sizes for a subset of siblings; treat
    # those as 0 for the aggregate banner (the actual bytes-uploaded
    # counter tracks the ground truth below).
    total_bytes = sum((f.size or 0) for f in files)
    print(
        f"   files:    {len(files)} ({total_bytes / 1e9:.3f} GB total)",
        flush=True,
    )

    client = _r2_client(endpoint_url, profile)

    # ---- upload / skip loop
    if not verify_only:
        # Codex round-2 BLOCKING #1: NEVER ``rmtree`` a user-supplied
        # ``--tmp-dir`` — the operator may have pointed us at a shared
        # scratch root (e.g. ``/mnt/scratch``) that holds unrelated data.
        # Always work inside a per-run subdirectory we exclusively
        # created; only that subdirectory gets deleted on exit.
        base_tmp = tmp_dir if tmp_dir is not None else Path("/tmp")
        base_tmp.mkdir(parents=True, exist_ok=True)
        run_tmp = base_tmp / f"mirror-{os.getpid()}"
        run_tmp.mkdir(parents=True, exist_ok=True)
        # Expose the concrete per-run dir under the same variable name
        # the rest of the function (and _download_one_hf) references.
        tmp_dir = run_tmp
        uploaded = 0
        skipped = 0
        bytes_uploaded = 0
        try:
            for idx, f in enumerate(files, 1):
                head = _r2_head(client, bucket, f.key)
                head_size = int(head["ContentLength"]) if head is not None else None
                # Boto3 lowercases user metadata keys on read.
                head_sha = (
                    (head.get("Metadata") or {}).get("hf-sha256")
                    if head is not None
                    else None
                )
                if should_skip(
                    head_size,
                    f.size,
                    existing_sha256=head_sha,
                    expected_sha256=f.lfs_sha256,
                ):
                    skipped += 1
                    tag = "sha+size" if f.lfs_sha256 else "size-only"
                    print(
                        f"[{idx}/{len(files)}] SKIP existing {f.key} "
                        f"({head_size} B, {tag})",
                        flush=True,
                    )
                    continue
                if dry_run:
                    size_label = f.size if f.size is not None else "?"
                    print(
                        f"[{idx}/{len(files)}] DRY-RUN would upload {f.key} "
                        f"({size_label} B, type={content_type_for(f.relpath)})",
                        flush=True,
                    )
                    continue
                # Download → upload → delete, one at a time.
                size_label = f.size if f.size is not None else "?"
                print(
                    f"[{idx}/{len(files)}] UPLOAD started {f.key} ({size_label} B)",
                    flush=True,
                )
                t0 = time.monotonic()
                local = _download_one_hf(repo_id, f.relpath, tmp_dir)
                # Codex round-3 BLOCKING: stat BEFORE cleanup — the
                # ``finally`` below deletes ``local``, so a later
                # ``local.stat()`` in the summary path would race and
                # always find zero. Capture the on-disk size now.
                if f.size is not None:
                    actual_size = f.size
                else:
                    try:
                        actual_size = local.stat().st_size
                    except OSError:
                        actual_size = 0
                try:
                    _upload_one(
                        client,
                        local,
                        bucket,
                        f.key,
                        content_type_for(f.relpath),
                        lfs_sha256=f.lfs_sha256,
                    )
                finally:
                    _cleanup_local(local)
                wall = time.monotonic() - t0
                uploaded += 1
                bytes_uploaded += actual_size
                print(
                    f"[{idx}/{len(files)}] OK {actual_size} B {wall:.1f}s "
                    f"({(actual_size / max(wall, 0.001)) / 1e6:.1f} MB/s) {f.key}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"FAIL {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            # Clean tmp on failure so a re-run isn't confused by stale
            # partial downloads.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return 2
        finally:
            # Empty tmp on success too — the whole point is per-file
            # streaming with no accumulating cache.
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(
            f"   upload summary: {uploaded} uploaded, {skipped} skipped, "
            f"{bytes_uploaded / 1e9:.3f} GB",
            flush=True,
        )

    # ---- verification pass
    print(f"-- verify {repo_id} --", flush=True)
    verify_failed: list[tuple[str, str]] = []
    for f in files:
        head_size = _r2_head_size(client, bucket, f.key)
        if head_size is None:
            verify_failed.append((f.key, "r2-missing"))
            print(f"   FAIL {f.key}: not on R2", flush=True)
            continue
        if f.size is not None and head_size != f.size:
            verify_failed.append((f.key, f"r2-size:{head_size}!={f.size}"))
            print(
                f"   FAIL {f.key}: R2 size {head_size} != HF size {f.size}",
                flush=True,
            )
            continue
        # Public read.
        # Codex round-2 BLOCKING #3: a 0-byte object cannot satisfy
        # ``Range: bytes=0-0`` — the server correctly returns HTTP 416
        # (Range Not Satisfiable), which would falsely fail verify.
        # Route empty files through HEAD (which returns 200 on any
        # publicly-readable object regardless of size). Non-empty files
        # keep the byte-range GET path — it's cheaper than a full body
        # download and works when the edge rejects HEAD.
        url = _public_url(public_base, f.key)
        if head_size == 0:
            status = _http_head_status(url)
        else:
            status = _http_range_get_status(url)
        if status != 200:
            verify_failed.append((f.key, f"public-http:{status}"))
            print(
                f"   FAIL {f.key}: public URL {url} → HTTP {status}",
                flush=True,
            )
            continue
        # Silent on success — printing 1 line per file is enough on the
        # upload pass; the verify pass only reports failures.
    wall = time.monotonic() - started
    if verify_failed:
        print(
            f"== FAILED: {len(verify_failed)} verify errors in {repo_id} ==",
            file=sys.stderr,
            flush=True,
        )
        for k, why in verify_failed:
            print(f"   {k}: {why}", file=sys.stderr, flush=True)
        return 3
    print(
        f"== OK: {repo_id} verified ({len(files)} files, "
        f"{total_bytes / 1e9:.3f} GB, wall {wall:.1f}s) ==",
        flush=True,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mirror_to_r2",
        description=(
            "Mirror a HuggingFace model repo to the rapid-mlx R2 bucket. "
            "Streams file-by-file (peak disk = one shard) and verifies "
            "each object is publicly readable via models.rapidmlx.com."
        ),
    )
    p.add_argument("repo_id", help="HF repo id, e.g. mlx-community/Qwen3-0.6B-4bit")
    p.add_argument(
        "--endpoint-url",
        default=DEFAULT_ENDPOINT_URL,
        help="R2 S3-compat endpoint URL (default: production)",
    )
    p.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="R2 bucket name (default: rapid-mlx-models)",
    )
    p.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="AWS profile name in ~/.aws/credentials (default: r2)",
    )
    p.add_argument(
        "--public-base",
        default=DEFAULT_PUBLIC_BASE,
        help="Public read URL base (default: https://models.rapidmlx.com)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate files, HEAD-check R2, but do not download/upload.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip upload; only run the verification pass.",
    )
    p.add_argument(
        "--subfolder",
        default=None,
        help=(
            "Mirror only this quantisation subfolder (e.g. '4bit') from a "
            "repo that ships several. Repo-root LICENSE/NOTICE/README are "
            "always included. Default: mirror the whole repo."
        ),
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Scratch dir for per-file downloads (default: /tmp/mirror-<pid>)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tmp = Path(args.tmp_dir) if args.tmp_dir else None
    return mirror_repo(
        args.repo_id,
        subfolder=args.subfolder,
        endpoint_url=args.endpoint_url,
        bucket=args.bucket,
        profile=args.profile,
        public_base=args.public_base,
        dry_run=args.dry_run,
        verify_only=args.verify_only,
        tmp_dir=tmp,
    )


if __name__ == "__main__":
    sys.exit(main())
