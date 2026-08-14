# SPDX-License-Identifier: Apache-2.0
"""Auto-pull confirmation gate for large model downloads.

Persona-3 ("Ollama switcher") feedback (2026-05): running

    rapid-mlx chat qwen3-coder-4bit

against an alias that wasn't yet cached silently kicked off a 41.8 GB
download with no ``[Y/n]`` prompt. The download itself ran fine, but
because the spawned ``serve`` subprocess captured stdout to a logfile,
the user saw a blank screen and assumed the CLI was hung.

This module is the user-visible safety net. It is intentionally
self-contained at MODULE level (no rapid-mlx imports) so it stays cheap
to import from ``cli.main()`` on every invocation. The two subfolder
helpers below reach into ``model_aliases`` through function-local
imports, which preserves that property — the registry is only read on
the paths that already touch the filesystem or the Hub.

Public API:

* :func:`estimate_repo_size_bytes` — best-effort HF API size lookup.
* :func:`confirm_or_abort`         — prompt + abort path used by the CLI.
* :func:`is_repo_cached`           — cache-presence probe (so callers can
  short-circuit without re-implementing the path dance).

Design choices:

* The HF call is wrapped in a hard 5-second timeout and a blanket
  ``except Exception`` — a flaky metadata query must never block a
  perfectly-good cached load, and we already gate on cache presence
  upstream of the size estimate.
* The threshold defaults to 10 GiB. Anything under that is too small
  to warrant interrupting the user's flow.
* The env override (``RAPID_MLX_AUTO_PULL=1``) is the documented escape
  hatch for non-interactive CI usage and ``--yes``-style workflows.
* Non-TTY stdin → auto-confirm. Scripts that pipe input into
  ``rapid-mlx`` must not deadlock on a missing terminal.
"""

from __future__ import annotations

import os
import re
import sys
import threading

# File suffixes that contribute to "model weight + tokenizer" footprint.
# Anything outside this set (e.g. ``.gitattributes``, ``README.md``) is a
# rounding error and is excluded so the prompt size matches what the user
# actually waits on.
_WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".safetensors",
    ".bin",
    ".gguf",
    # ``.npz`` is the weight format the mlx audio repos ship (mlx-whisper's
    # ``weights.npz`` is ~2.9 GiB for large-v3). Excluding it made those
    # repos size to ~0 in the download-size estimate — count it so both the
    # ``[Y/n]`` gate and the ``rapid-mlx models`` Size column are accurate.
    ".npz",
    ".json",
    ".txt",
    ".model",
    ".tiktoken",
)

# Suffixes treated as cache-proving by ``is_repo_cached``. mlx-lm's
# high-level loader (the path rapid-mlx serve takes) only globs
# ``model*.safetensors`` — see ``mlx_lm/utils.py:316``.
#
# Codex round-4 BLOCKING #2 trimmed this list to ``.safetensors`` only:
#   * ``.bin``  — PyTorch shards, never loaded by mlx-lm.
#   * ``.gguf`` — mlx-lm has *export* support (convert_to_gguf) but
#                 no load path; ``mx.save_gguf`` is one-way.
#   * ``.npz``  — older mlx-lm convert format; current mlx-lm load
#                 doesn't reach it either.
#
# Keeping these in the cache-proving set lets a non-loadable cache
# (e.g. cached ``weights.npz`` from a 2024-era mlx-community fork) pass
# the gate and route the user back into "silent download in the
# spawned serve subprocess" — which is exactly what B2 exists to fix.
_WEIGHT_ONLY_SUFFIXES: tuple[str, ...] = (".safetensors",)

# 5-second cap on the HF metadata call. Anything slower than this is a
# signal we should fall through silently rather than block startup.
_HF_API_TIMEOUT_SECONDS: float = 5.0

# Cap on a metadata call whose failure ABORTS a download rather than merely
# degrading it (resolving a revision, listing a repo for the mirror). Far more
# generous than the best-effort probe above — a slow-but-working Hub must still
# be allowed to serve a first-run pull — and far below the desktop's 30-minute
# stall window, so a dead network surfaces as an error instead of a hang.
#
# A deadline is the ONLY thing that bounds these calls. huggingface_hub passes
# ``timeout=None`` explicitly into its shared httpx client, and httpx treats an
# explicit ``None`` as "disable the timeout" rather than "use the client
# default" — so configuring the client (``set_client_factory``) does not bound
# them, and measurably does not: a client built with a 2s timeout still hung
# past 12s on a blackholed route when the call passed ``timeout=None``.
_HF_RESOLVE_TIMEOUT_SECONDS: float = 30.0


def _format_size(num_bytes: int) -> str:
    """Render ``num_bytes`` as a human-friendly string (e.g. ``42.3 GiB``).

    Uses base-1024 units (KiB/MiB/GiB) to match the way HF Hub and
    macOS Finder report file sizes. The ``iB`` suffix is the IEC
    standard for base-1024 — clearer than bare ``KB``/``GB`` which is
    ambiguous (powers of 10 in some tools, 2 in others) and matters
    here because the confirmation threshold is denominated in 1024**3.
    """
    if num_bytes < 0:
        num_bytes = 0
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"  # unreachable, keeps mypy happy


def _is_weight_file(name: str) -> bool:
    """True if ``name`` should be counted in the download-size estimate."""
    if not name or name.startswith(".git"):
        return False
    lower = name.lower()
    return any(lower.endswith(s) for s in _WEIGHT_SUFFIXES)


def _sibling_size(sibling) -> int:
    """Extract the on-disk size of an HF ``RepoSibling``.

    Newer huggingface_hub releases expose the LFS pointer's true size
    under ``sibling.lfs.size``; older ones store it directly on
    ``sibling.size``. Try both, prefer the LFS value when both are
    populated (the regular ``size`` may report the pointer-file size,
    not the resolved blob size).
    """
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None:
        lfs_size = getattr(lfs, "size", None)
        if isinstance(lfs_size, int) and lfs_size > 0:
            return lfs_size
    raw = getattr(sibling, "size", None)
    if isinstance(raw, int) and raw > 0:
        return raw
    return 0


# A download attempt that is killed rather than allowed to unwind leaves its
# scratch file behind (see :func:`reap_orphan_incomplete_blobs`). Six hours of
# no writes is the "no live writer" signal: an in-flight transfer touches its
# scratch file continuously, and nothing legitimate goes quiet for this long
# while still intending to finish.
_ORPHAN_INCOMPLETE_MIN_AGE_SECONDS: float = 6 * 60 * 60


def reap_orphan_incomplete_blobs(
    repo_id: str,
    *,
    min_age_seconds: float = _ORPHAN_INCOMPLETE_MIN_AGE_SECONDS,
    now: float | None = None,
) -> tuple[int, int]:
    """Delete abandoned ``blobs/*.incomplete`` scratch files for one repo.

    Returns ``(files_removed, bytes_reclaimed)``; ``(0, 0)`` on any problem.

    huggingface_hub downloads each blob to a per-attempt scratch file named
    ``<etag>.<8 hex>.incomplete`` and unlinks it in a ``finally``. The unique
    name is deliberate upstream (a shared name corrupts the cache on filesystems
    where ``flock`` silently succeeds for every caller), but it means an
    interrupted attempt is never resumed — and a process killed by a signal
    never runs that ``finally``. The desktop cancels a download with SIGTERM and
    then SIGKILL, so every cancel, quit or crash strands one file per blob that
    was in flight, and nothing in the cache ever reclaims them. Measured on a
    developer machine: three files for a single blob, written 49, 75 and 170
    hours apart — one per interrupted attempt, days apart, with no concurrency
    involved.

    Deleting a model already removes its whole ``models--<repo>`` directory, so
    this exists for the repos a user KEEPS, where the files would otherwise
    accumulate for the life of the cache.

    The age gate is the safety property. Concurrent writers to one repo's blobs
    are possible (a background ``pull`` on the mirror path alongside a ``serve``
    falling back to HF), and while unlinking a file another process holds open
    does not corrupt anything on POSIX — the writer keeps its descriptor and
    only fails at the final rename — it would still turn someone else's working
    download into an error. Requiring a long silence removes that entirely.
    """
    import stat as _stat
    import time

    if now is None:
        now = time.time()

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        blobs_dir = os.path.join(
            HF_HUB_CACHE,
            f"models--{repo_id.replace('/', '--')}",
            "blobs",
        )
        names = os.listdir(blobs_dir)
    except OSError:
        return 0, 0

    removed = 0
    reclaimed = 0
    for name in names:
        if not name.endswith(".incomplete"):
            continue
        path = os.path.join(blobs_dir, name)
        try:
            # ``lstat``: never follow a symlink here. Real blobs are regular
            # files and the scratch files are too; anything else in this
            # directory shaped like one is not ours to delete.
            info = os.lstat(path)
            if not _stat.S_ISREG(info.st_mode):
                continue
            if now - info.st_mtime < min_age_seconds:
                continue
            size = info.st_size
            os.remove(path)
        except OSError:
            # Racing writer, permissions, already gone — skip it. Reclaiming
            # scratch space must never be able to fail a pull.
            continue
        removed += 1
        reclaimed += size
    return removed, reclaimed


def call_with_deadline(fn, timeout: float, /, *args, **kwargs):
    """Run ``fn`` in a worker thread and give up after ``timeout`` seconds.

    A deadline is the ONLY thing that bounds a huggingface_hub metadata call.
    The library passes ``timeout=None`` explicitly into its shared httpx client,
    and httpx reads an explicit ``None`` as "disable the timeout" rather than
    "inherit the client's" — so configuring the client does not bound these
    calls, and measurably does not: a client built with a 2s timeout still hung
    past 12s on a blackholed route once the call passed ``timeout=None``.

    Takes the callable rather than importing one itself so callers keep
    resolving their own symbol (``from huggingface_hub import model_info``
    inside the function body), which is what makes these paths patchable in
    tests. Worst case the daemon thread is leaked and reaped at interpreter
    exit — acceptable for a process that is about to fail out anyway.

    Raises ``TimeoutError`` if the deadline lapses; otherwise re-raises whatever
    ``fn`` raised, or returns its value.
    """
    result: dict = {}

    def _call() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            result["error"] = exc

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} exceeded {timeout}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def pin_main_ref(repo_id: str, revision: str) -> None:
    """Atomically record the resolved default-branch revision in HF's cache.

    A download pinned to a commit SHA avoids a second, unbounded Hub metadata
    lookup, but huggingface_hub intentionally does not write ``refs/main`` for
    an explicit SHA. Rapid's warm-cache gate needs that ref, so publish it only
    after the pinned snapshot download succeeds. Failure is best-effort: the
    downloaded snapshot is still valid, and a later run can resolve it online.
    """
    if not revision:
        return
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        refs_dir = os.path.join(
            HF_HUB_CACHE,
            f"models--{repo_id.replace('/', '--')}",
            "refs",
        )
        os.makedirs(refs_dir, exist_ok=True)
        target = os.path.join(refs_dir, "main")
        temporary = f"{target}.{os.getpid()}.tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as fh:
                fh.write(revision)
            os.replace(temporary, target)
        finally:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass
    except OSError:
        pass


def _model_info_with_timeout(repo_id: str, timeout: float):
    """Call ``HfApi().model_info`` with a hard timeout.

    huggingface_hub itself doesn't accept a timeout argument on
    ``model_info`` in every release we support, so we run it in a worker
    thread and join with a deadline. Worst case (network hang) the
    daemon thread is leaked and reaped at interpreter exit — acceptable
    for a CLI that's about to exit anyway one way or the other.
    """
    from huggingface_hub import HfApi

    from ._hf_logging import silence_hf_unauthenticated_warning

    # The Hub currently attaches its "set a HF_TOKEN" advisory to
    # file-download responses (silenced in _mirror), not this metadata
    # probe. But the probe is our first Hub touch on a cold pull, so
    # installing the fail-open filter here too upholds the "filter is up
    # before the first request" invariant whichever response the server
    # decides to tag — it is warn-once per process.
    silence_hf_unauthenticated_warning()

    result: dict = {}

    def _call() -> None:
        try:
            result["info"] = HfApi().model_info(repo_id, files_metadata=True)
        except Exception as exc:  # pragma: no cover - defensive
            result["error"] = exc

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"model_info({repo_id!r}) exceeded {timeout}s")
    if "error" in result:
        raise result["error"]
    return result.get("info")


def _descend_to_checkpoint(snap_dir: str, repo_id: str) -> str:
    """The directory mlx-lm will actually glob inside ``snap_dir``.

    Identity for the ordinary flat repo. For a subfolder-per-quant repo
    it appends the declared folder — but only if that folder is really
    there, so a publisher who reorganises the repo degrades to the
    (honest) "not cached" answer instead of an exception.
    """
    from .model_aliases import checkpoint_prefix

    prefix = checkpoint_prefix(repo_id)
    if not prefix:
        return snap_dir
    candidate = os.path.join(snap_dir, prefix.rstrip("/"))
    return candidate if os.path.isdir(candidate) else snap_dir


def estimate_repo_size_bytes(repo_id: str) -> int | None:
    """Best-effort total size of weight + tokenizer files in ``repo_id``.

    Returns the sum of ``sibling.size`` (preferring LFS-reported size
    when available) across files whose extension marks them as weight
    or tokenizer payload. ``None`` on any failure (network down, gated
    repo, HF outage, timeout) — callers should fall through silently.

    For a repo that ships one folder per quantization, only the folder
    this alias actually downloads is counted. Summing the whole repo
    told a user that ``serve lfm2.5-2.6b-4bit`` was about to pull
    18.7 GiB when the real transfer is 1.6 GiB — a confirm prompt that
    scares people away from a download that would have taken seconds.
    """
    try:
        info = _model_info_with_timeout(repo_id, _HF_API_TIMEOUT_SECONDS)
    except Exception:
        return None

    from .model_aliases import checkpoint_prefix

    prefix = checkpoint_prefix(repo_id)
    siblings = getattr(info, "siblings", None) or []
    total = 0
    for sib in siblings:
        name = getattr(sib, "rfilename", "") or ""
        if prefix and not name.startswith(prefix):
            continue
        if not _is_weight_file(name):
            continue
        total += _sibling_size(sib)
    return total if total > 0 else None


def _is_model_weight_filename(name: str) -> bool:
    """True if ``name`` matches mlx-lm's loader glob ``model*.safetensors``.

    mlx-lm's high-level load path (``mlx_lm/utils.py:316``) is literally
    ``glob.glob(str(model_path / "model*.safetensors"))``. Adapter /
    sidecar files (``adapter.safetensors``, LoRA fine-tunes,
    ``embeddings.safetensors``, etc.) DON'T match this pattern and
    aren't loaded by rapid-mlx's text path — so they must NOT count
    as cache-proof either. Codex round-5 BLOCKING #2.

    Case sensitivity (Codex round-6 BLOCKING #1): the glob is case-
    sensitive on case-sensitive filesystems (Linux, default macOS APFS
    with case-sensitive volumes). A repo whose file is named
    ``Model.safetensors`` would not be picked up by mlx-lm and so must
    not pass the gate either. We mirror Python's ``glob`` rather than
    being lax with a ``.lower()`` comparison.
    """
    if not name.endswith(".safetensors"):
        return False
    return name.startswith("model")


_NUMBERED_MODEL_SHARD_RE = re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$")


def _numbered_shards_are_complete(snap_dir: str) -> bool | None:
    """Validate an index-less ``model-N-of-M`` shard set.

    ``snapshot_download`` does not promise that
    ``model.safetensors.index.json`` lands before the weight files. If a pull
    is interrupted after shard 1 arrives but before the index, the generic
    ``model*.safetensors`` fallback must not mistake that one shard for a
    complete single-file checkpoint.

    Returns ``None`` when the snapshot has no numbered shard names, preserving
    the ordinary ``model.safetensors`` / ``model-q4.safetensors`` path. When
    numbered shards are present, every name must agree on one positive total
    and cover the indices 1...total exactly once. File type and non-zero size
    remain the responsibility of ``_root_model_files_all_non_empty`` below.
    """
    try:
        names = os.listdir(snap_dir)
    except OSError:
        return False

    numbered: list[tuple[int, int]] = []
    for name in names:
        match = _NUMBERED_MODEL_SHARD_RE.fullmatch(name)
        if match is None:
            continue
        numbered.append((int(match.group(1)), int(match.group(2))))
    if not numbered:
        return None

    totals = {total for _, total in numbered}
    if len(totals) != 1:
        return False
    total = totals.pop()
    indices = {index for index, _ in numbered}
    return (
        total > 0
        and len(numbered) == total
        and len(indices) == total
        and min(indices) == 1
        and max(indices) == total
    )


def _snapshot_is_complete(snap_dir: str) -> bool:
    """True if ``snap_dir`` looks like a fully-downloaded model snapshot.

    Originally factored to mirror ``vllm_mlx.doctor.discovery``;
    rounds 4-6 of the codex review tightened this path beyond doctor's.
    The two now have *intentionally* different policies (Codex round-6
    NIT #3 on convergence):
      * Doctor cares "is there a model directory I could try to run?"
        and accepts a single safetensors / npz / gguf as a hint.
      * B2 gate cares "will mlx_lm.load successfully open this without
        downloading more shards?" — answerable only by mirroring the
        actual loader glob (``model*.safetensors``, case-sensitive).
    Unifying would loosen B2 or tighten doctor; both are wrong.

    Strategy:
      1. ``model.safetensors.index.json`` present → parse ``weight_map``
         and require every referenced shard to exist with non-zero
         size. Codex round-4 BLOCKING #1.
      2. Without an index, numbered ``model-N-of-M`` files must form the
         complete 1...M set. This covers an interrupted pull where one shard
         lands before the index file.
      3. Otherwise, a single non-empty ``model*.safetensors`` is sufficient
         (covers single-file non-sharded models).

    Index-but-no-shards (Codex round-5 BLOCKING #1): when an
    ``model.safetensors.index.json`` exists but yields no shard names
    (corrupt schema, alternate-key layout, metadata-only index), DO
    NOT fall back to the single-file probe. The presence of the index
    itself is the loader's signal that this is a sharded model — a
    single stray ``model.safetensors`` next to a non-standard index
    is incomplete by definition.
    """
    index_path = os.path.join(snap_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json

        try:
            with open(index_path) as fh:
                index = json.load(fh)
        except (OSError, json.JSONDecodeError):
            # Truncated index → safer to treat as incomplete and re-prompt.
            return False
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            # Index exists but doesn't yield a usable shard list. We
            # know SOMETHING expects shards here; refuse to fall
            # through to the lax single-file probe.
            return False
        shard_names = set(weight_map.values())
        # Codex round-6 BLOCKING #2: validate the shard filenames
        # themselves match the loader glob, not just that they exist.
        # Codex round-7 BLOCKING #1: AND make sure the shard names
        # don't escape ``snap_dir`` via ``..`` or an absolute path —
        # ``mlx_lm`` only loads ``snap_dir/model*.safetensors``, so a
        # validated basename pointing at ``../somewhere-else`` would
        # otherwise pass while the loader sees nothing.
        for shard in shard_names:
            if not isinstance(shard, str):
                return False
            # No directory traversal, no absolute paths, no nested
            # subdirectories — the loader's glob is non-recursive on
            # the snapshot root.
            if (
                os.path.isabs(shard)
                or os.sep in shard
                or "/" in shard
                or ".." in shard.split("/")
            ):
                return False
            if not _is_model_weight_filename(shard):
                return False
            target = os.path.join(snap_dir, shard)
            try:
                if os.path.getsize(target) <= 0:
                    return False
            except OSError:
                return False
        # Codex round-7 BLOCKING #3: the loader globs every
        # ``model*.safetensors`` at the snapshot root — a stray
        # zero-byte ``model-extra.safetensors`` next to a valid
        # indexed cache would otherwise crash ``mx.load()``. Require
        # every snapshot-root match to be non-zero.
        if not _root_model_files_all_non_empty(snap_dir):
            return False
        return True

    # The index can arrive after the first weight shard. Infer completeness
    # from the standard shard filenames in that window instead of treating
    # shard 1/N as an ordinary single-file checkpoint.
    numbered_complete = _numbered_shards_are_complete(snap_dir)
    if numbered_complete is False:
        return False

    # Single-file (or complete index-less sharded) model. Match mlx-lm's actual loader
    # glob — ``adapter.safetensors`` / ``embeddings.safetensors`` and
    # other sidecars don't count; only ``model*.safetensors`` at the
    # snapshot root (Codex round-7 BLOCKING #2: the glob is NOT
    # recursive — a ``subdir/model.safetensors`` would not be picked
    # up, so we must not credit it as cached).
    if not _root_model_files_all_non_empty(snap_dir):
        return False
    try:
        for name in os.listdir(snap_dir):
            if not _is_model_weight_filename(name):
                continue
            full = os.path.join(snap_dir, name)
            try:
                if os.path.isfile(full) and os.path.getsize(full) > 0:
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def _root_model_files_all_non_empty(snap_dir: str) -> bool:
    """Every ``model*.safetensors`` at the snapshot root must be a
    non-zero regular file (or a symlink to one).

    Codex round-7 BLOCKING #3 caught the zero-byte placeholder case;
    round-8 BLOCKING extended it: ``glob`` returns symlinks to
    directories and dangling symlinks too, and mlx-lm then calls
    ``mx.load`` on them. So any entry matching the loader glob that
    is not a real file must REJECT the snapshot (not be silently
    skipped) — otherwise a ``model-extra.safetensors -> some_dir``
    sibling would crash the loader after the gate said "cached".
    """
    try:
        entries = os.listdir(snap_dir)
    except OSError:
        return False
    for name in entries:
        if not _is_model_weight_filename(name):
            continue
        full = os.path.join(snap_dir, name)
        try:
            # ``isfile`` follows symlinks → a healthy
            # symlink-into-blobs counts; a symlink-to-dir or dangling
            # symlink does NOT, and the loader's glob would still
            # return it. Reject rather than continue.
            if not os.path.isfile(full):
                return False
            if os.path.getsize(full) <= 0:
                return False
        except OSError:
            return False
    return True


def _resolved_snapshot_sha(repo_root: str) -> str | None:
    """Read the sha that ``snapshot_download(repo_id)`` would resolve to.

    ``snapshot_download`` with no explicit revision asks the HF API for
    the default branch's HEAD sha and writes it to
    ``models--<repo>/refs/<default_branch>``. For modern repos that
    file is ``refs/main``.

    Codex round-9 BLOCKING: without any pinning, an old complete
    snapshot could mask a newer-but-incomplete one (interrupted
    update). Codex round-10 BLOCKING: the round-9 fallbacks (single
    non-main ref, "any complete snapshot" when no refs/ exists) could
    mask the same attack a different way — a legacy ``refs/master``
    can shadow what ``main`` resolves to upstream now, and a
    no-refs/ cache could shadow any sha.

    We now ONLY honour ``refs/main``. The cost is a redundant prompt
    for repos whose default branch is renamed (rare; legacy
    ``master`` mostly); the benefit is no silent download in any
    other scenario.
    """
    main_ref = os.path.join(repo_root, "refs", "main")
    try:
        if not os.path.isfile(main_ref):
            return None
        with open(main_ref) as fh:
            sha = fh.read().strip()
        return sha or None
    except OSError:
        return None


def is_repo_cached(repo_id: str) -> bool:
    """True if ``repo_id`` has a usable model snapshot in the HF cache.

    Codex review round 1 caught that an earlier "config.json exists →
    cached" check let a partial cache (config + tokenizer only, weight
    shards missing) bypass the gate. The serve subprocess would then
    silently download the weights inside its log file. Round 4 then
    caught that even "any one safetensors file present" let a partial
    sharded cache (shard 1/2 present, shard 2/2 missing) bypass the
    gate; mlx-lm's loader globs every shard and fails halfway through.

    Revision pinning (Codex round-9 BLOCKING): when ``refs/main``
    (or the single resolved ref) exists, ONLY the snapshot pointed to
    by that ref counts — the loader resolves through the ref, and an
    unrelated old-but-complete snapshot must not mask a current-but-
    incomplete one after an interrupted ``snapshot_download`` update.

    The check delegates to ``_snapshot_is_complete`` so the doctor
    pre-flight and the B2 gate share one source of truth (with the
    intentional policy divergence documented there).

    Returns ``False`` on any internal exception so the caller defaults
    to the safe path (prompting, if the size warrants it).
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_root = os.path.join(
            HF_HUB_CACHE,
            f"models--{repo_id.replace('/', '--')}",
        )
        snap_root = os.path.join(repo_root, "snapshots")
        if not os.path.isdir(snap_root):
            return False

        # Codex round-10 BLOCKING: no "any complete snapshot"
        # fallback. The only safe cache-presence answer is "the
        # snapshot ``snapshot_download(repo_id)`` will actually use".
        # Without ``refs/main``, we don't know which sha will resolve,
        # so we must re-prompt and let the next run populate refs/.
        resolved_sha = _resolved_snapshot_sha(repo_root)
        if resolved_sha is None:
            return False
        snap_dir = os.path.join(snap_root, resolved_sha)
        if not os.path.isdir(snap_dir):
            return False
        # One repo, one folder per quantization (LiquidAI/LFM2.5-2.6B-MLX):
        # the checkpoint mlx-lm will glob is the subfolder, not the
        # snapshot root. Descend before asking "is this complete?" —
        # otherwise a fully cached 4-bit checkpoint reads as uncached and
        # the gate re-prompts on every single serve.
        snap_dir = _descend_to_checkpoint(snap_dir, repo_id)
        return _snapshot_is_complete(snap_dir)
    except Exception:
        pass
    return False


def _snapshot_is_complete_whisper_model(repo_id: str) -> bool:
    """True when a pinned mlx-audio Whisper snapshot can be loaded locally.

    mlx-community Whisper checkpoints intentionally use ``weights.npz`` plus
    ``config.json`` and contain no ``model*.safetensors``. The text-model
    completeness probe therefore rejects a fully downloaded Whisper model.
    Keep this family-specific so an arbitrary NPZ file cannot make a text or
    unknown repository look runnable.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_root = os.path.join(
            HF_HUB_CACHE,
            f"models--{repo_id.replace('/', '--')}",
        )
        resolved_sha = _resolved_snapshot_sha(repo_root)
        if resolved_sha is None:
            return False
        snap_dir = os.path.join(repo_root, "snapshots", resolved_sha)
        repo_root_real = os.path.realpath(repo_root)
        for name in ("config.json", "weights.npz"):
            path = os.path.join(snap_dir, name)
            if not os.path.isfile(path):
                return False
            # HF snapshot files normally point into this repo's own ``blobs``
            # directory. Do not let an unrelated local file satisfy the cache
            # gate through a crafted symlink.
            real = os.path.realpath(path)
            if real != repo_root_real and not real.startswith(repo_root_real + os.sep):
                return False
            if os.path.getsize(path) <= 0:
                return False
        return True
    except Exception:
        return False


def _snapshot_is_complete_split_model(repo_id: str) -> bool:
    """True if the resolved snapshot is a mlx-video component-split model whose
    EVERY declared component weight is cached — the non-text analogue of
    :func:`is_repo_cached` (which only knows mlx-lm's text
    ``model*.safetensors`` layout).

    Video-gen repos don't lay their weights out the way the text loader
    expects: they ship one ``<component>.safetensors`` per component at the
    snapshot root (CogVideoX-Fun → ``transformer`` / ``text_encoder`` / ``vae``;
    LTX-2.3 → ``transformer`` / ``connector`` / ``vae_decoder`` / ``vocoder`` /
    …), never a ``model*.safetensors``. ``is_repo_cached`` therefore reads a
    fully-cached video model as *weightless*, which would make
    :func:`is_weightless_stub` cry wolf ("config cached, weights missing —
    will download ~N GB") on every serve of an already-downloaded video model.

    Completeness, not mere presence (codex BLOCKING): we DON'T infer "non-text,
    weighted" from the presence of any stray ``.safetensors`` — that would
    misread (a) an interrupted multimodal *text* download whose vision tower
    landed before its shards, or (b) an interrupted *video* pull that has only
    one of its components. Instead we anchor on ``split_model.json`` — the
    mlx-video manifest that positively identifies the layout AND enumerates the
    exact component list — and require EVERY ``<component>.safetensors`` to be
    present and non-empty, exactly as :func:`_snapshot_is_complete` walks the
    text ``weight_map``. A text LLM never ships ``split_model.json``, so an
    incomplete text cache always falls through to :func:`is_repo_cached`; a
    partial video cache fails a component check and also falls through.

    ``model_index.json`` (bare diffusers pipeline manifest) is intentionally
    NOT accepted on its own: it names components but not their on-disk weight
    filenames (flat file vs ``component/`` subdir vs sharded — packaging
    dependent), so it can't be completeness-checked reliably here. Both rapid-
    mlx video families (CogVideoX-Fun, LTX-2.3) ship ``split_model.json``; a
    hypothetical manifest-less / index-only repo simply falls back to the
    (cosmetic) false alarm rather than risking a wrong suppression.

    Mirrors :func:`is_repo_cached`'s snapshot resolution (``refs/main`` →
    pinned sha) so both read the same on-disk snapshot. Returns ``False`` on
    any internal error so the caller defaults to the existing text-glob path.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_root = os.path.join(
            HF_HUB_CACHE,
            f"models--{repo_id.replace('/', '--')}",
        )
        resolved_sha = _resolved_snapshot_sha(repo_root)
        if resolved_sha is None:
            return False
        snap_dir = os.path.join(repo_root, "snapshots", resolved_sha)
        if not os.path.isdir(snap_dir):
            return False

        manifest_path = os.path.join(snap_dir, "split_model.json")
        if not os.path.isfile(manifest_path):
            return False
        import json

        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return False
        components = manifest.get("components") if isinstance(manifest, dict) else None
        # A manifest that names no components tells us nothing is complete.
        if not isinstance(components, list) or not components:
            return False

        repo_root_real = os.path.realpath(repo_root)
        for component in components:
            # Component names become ``<component>.safetensors`` at the
            # snapshot root; reject anything that could escape it.
            if not isinstance(component, str) or not component:
                return False
            if (
                os.path.isabs(component)
                or os.sep in component
                or "/" in component
                or ".." in component.split("/")
            ):
                return False
            fpath = os.path.join(snap_dir, f"{component}.safetensors")
            # Must be a real regular file (following the blob symlink), not a
            # directory named ``<component>.safetensors`` nor a dangling /
            # directory symlink — ``os.path.getsize`` alone reports a positive
            # size for a directory, so an empty ``vae.safetensors/`` dir would
            # otherwise pass. ``isfile`` follows symlinks and is False for
            # symlink→dir and dangling symlinks (codex round-5 MAJOR).
            if not os.path.isfile(fpath):
                return False
            # The file (via its blob symlink) must resolve inside this repo's
            # own cache dir — a symlink escaping elsewhere doesn't count.
            real = os.path.realpath(fpath)
            if real != repo_root_real and not real.startswith(repo_root_real + os.sep):
                return False
            try:
                if os.path.getsize(fpath) <= 0:
                    return False
            except OSError:
                return False
        return True
    except Exception:
        return False


def split_model_local_snapshot(repo_id: str) -> str | None:
    """Snapshot directory for a verified-complete cached video repo, else ``None``.

    The component-split analogue of :func:`mflux_local_snapshot`, and it exists
    for the same reason: handing the loader a repo id makes huggingface_hub
    resolve the revision on every start, warm cache included, and that lookup
    has no deadline of its own — see :func:`call_with_deadline`.

    Gated on :func:`_snapshot_is_complete_split_model`, which walks the
    ``split_model.json`` component list, so a half-pulled checkpoint keeps
    downloading instead of being handed over as usable.
    """
    try:
        if not _snapshot_is_complete_split_model(repo_id):
            return None
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_root = os.path.join(
            HF_HUB_CACHE,
            f"models--{repo_id.replace('/', '--')}",
        )
        resolved_sha = _resolved_snapshot_sha(repo_root)
        if resolved_sha is None:
            return None
        snap_dir = os.path.join(repo_root, "snapshots", resolved_sha)
    except Exception:
        return None
    return snap_dir if os.path.isdir(snap_dir) else None


def _mflux_snapshot_dir(repo_id: str) -> tuple[str, str] | None:
    """``(repo_root, snapshot_dir)`` for a registered image-gen repo, or ``None``.

    ``None`` means "no verdict is possible here" — not an image-gen alias, no
    cache under it, or an ambiguous set of unpinned snapshots. Every caller
    treats that as "let the normal online path run".

    Shared by :func:`mflux_missing_weights` (which asks "is what's here
    complete?") and :func:`mflux_local_snapshot` (which asks "where is it?"),
    so the two can never disagree about *which* snapshot they are talking about.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        from vllm_mlx.model_aliases import resolve_profile
    except ImportError:
        return None

    profile = resolve_profile(repo_id)
    if profile is None or profile.modality != "image-gen":
        return None

    repo_root = os.path.join(
        HF_HUB_CACHE,
        f"models--{repo_id.replace('/', '--')}",
    )
    resolved_sha = _resolved_snapshot_sha(repo_root)
    if resolved_sha is None:
        # An interrupted first download can leave component indexes and
        # shards behind before huggingface_hub commits refs/main.  With one
        # mflux-shaped snapshot there is no ambiguity, so inspect it rather
        # than letting the loader consume a partial checkpoint.  Never choose
        # among multiple unpinned snapshots: an older complete snapshot could
        # otherwise mask the interrupted one.
        snapshots_root = os.path.join(repo_root, "snapshots")
        try:
            candidates = [
                name
                for name in os.listdir(snapshots_root)
                if os.path.isdir(os.path.join(snapshots_root, name))
            ]
        except OSError:
            return None
        if len(candidates) != 1:
            return None
        candidate_dir = os.path.join(snapshots_root, candidates[0])
        if not any(
            os.path.lexists(
                os.path.join(candidate_dir, component, "model.safetensors.index.json")
            )
            for component in ("transformer", "text_encoder", "vae")
        ):
            return None
        resolved_sha = candidates[0]
    snap_dir = os.path.join(repo_root, "snapshots", resolved_sha)
    if not os.path.isdir(snap_dir):
        return None
    return repo_root, snap_dir


def mflux_local_snapshot(repo_id: str) -> str | None:
    """Snapshot directory for a verified-complete cached mflux repo, else ``None``.

    Handing mflux this directory instead of the bare repo id keeps a warm start
    entirely off the network. mflux resolves a repo id through
    ``huggingface_hub``, whose revision lookup carries no timeout at all
    (``HfApi.repo_info`` passes ``timeout=None`` into an ``httpx.Client`` built
    with ``timeout=None``), so on a hostile DNS path it does not fail fast — it
    sits in SYN_SENT while the UI shows "Starting".

    Deliberately NOT implemented as ``snapshot_download(local_files_only=True)``.
    That asks whether the *whole repo* is present, and judges it against HF's
    cached tree listing — which names files the mirror never fetches. A complete
    mflux checkpoint pulled through the R2 mirror fails that check on
    ``.DS_Store`` / ``README.md`` / ``.gitattributes`` and would silently keep
    the network round-trip this exists to remove (measured on
    ``filipstrand/Z-Image-Turbo-mflux-4bit``). :func:`mflux_missing_weights` asks
    the question that actually matters — are the weights mflux will load all
    here — so gate on that and resolve the directory from the cache layout.

    ``None`` on every uncertain path, so the caller falls back to today's
    behavior rather than pointing the loader at a checkpoint we can't vouch for.
    That includes the unexpected errors :func:`mflux_missing_weights` propagates
    on purpose: swallowing them costs only this optimization, and the image
    engine calls that function directly in ``_verify_weights_complete`` — ahead
    of the build that reaches this one — so a real bug still surfaces there.
    """
    try:
        if mflux_missing_weights(repo_id) != []:
            return None
        resolved = _mflux_snapshot_dir(repo_id)
    except Exception:
        return None
    if resolved is None:
        return None
    return resolved[1]


def mflux_missing_weights(repo_id: str) -> list[str] | None:
    """Snapshot-relative paths of mflux weight files that are absent or empty.

    ``[]`` means every file the component indexes name is present and
    non-empty. A non-empty list names what is missing, which is exactly what
    an error message needs to be actionable.

    ``None`` is the third state, and the reason this returns a list rather
    than a bool: *no verdict*. It means ``repo_id`` is not a registered
    image-gen alias, or nothing is cached under it yet — in which case mflux
    pulls the whole snapshot itself and there is no partial checkpoint to
    guard against. Callers that gate on this MUST treat ``None`` as "let it
    through": a missing dependency or an unreadable cache is an environment
    problem, and reporting it as a corrupt model would send the user off to
    re-download perfectly good weights.

    Only expected filesystem/JSON errors are absorbed into ``None``; anything
    else propagates so a real bug surfaces as a stack trace rather than a
    phantom "your model is broken".
    """
    resolved = _mflux_snapshot_dir(repo_id)
    if resolved is None:
        return None
    repo_root, snap_dir = resolved

    repo_root_real = os.path.realpath(repo_root)

    def _is_nonempty_repo_file(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        real = os.path.realpath(path)
        if real != repo_root_real and not real.startswith(repo_root_real + os.sep):
            return False
        try:
            return os.path.getsize(path) > 0
        except OSError:
            return False

    import json

    missing: list[str] = []

    # All currently supported mflux families use these three components and a
    # local tokenizer. Requiring the full set prevents an interrupted pull with
    # only one component index from looking runnable.
    if not _is_nonempty_repo_file(
        os.path.join(snap_dir, "tokenizer", "tokenizer.json")
    ):
        missing.append("tokenizer/tokenizer.json")

    for component in ("transformer", "text_encoder", "vae"):
        component_dir = os.path.join(snap_dir, component)
        index_rel = f"{component}/model.safetensors.index.json"
        index_path = os.path.join(component_dir, "model.safetensors.index.json")
        if not _is_nonempty_repo_file(index_path):
            missing.append(index_rel)
            continue
        try:
            with open(index_path) as fh:
                index = json.load(fh)
        except (OSError, json.JSONDecodeError):
            # An index we cannot parse cannot vouch for its shards. Naming it
            # keeps the component fail-closed without pretending to know which
            # weight files it would have listed.
            missing.append(index_rel)
            continue
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            missing.append(index_rel)
            continue
        shard_values = weight_map.values()
        if not all(isinstance(v, str) for v in shard_values):
            # A non-string value (list, number, null) is enough to make
            # ``set()``/``sorted()`` below raise instead of returning a verdict.
            # An index that cannot name its shards as plain strings cannot vouch
            # for them either, so fail the component closed before deduping.
            missing.append(index_rel)
            continue
        for shard in sorted(set(shard_values)):
            if (
                not shard.endswith(".safetensors")
                or os.path.basename(shard) != shard
                or shard in (".", "..")
            ):
                # Path traversal guard: refuse to act on a hostile index at
                # all rather than probing the path it names.
                missing.append(index_rel)
                break
            if not _is_nonempty_repo_file(os.path.join(component_dir, shard)):
                missing.append(f"{component}/{shard}")
    return missing


def _snapshot_is_complete_mflux_model(repo_id: str) -> bool:
    """True when a registered image-gen repo has every mflux weight shard.

    mflux checkpoints keep independent sharded indexes below ``transformer/``,
    ``text_encoder/``, and ``vae/``. They therefore satisfy neither the text
    ``model*.safetensors`` probe nor mlx-video's ``split_model.json`` probe.

    Best-effort boolean for the ``ls``-style callers that only render a
    cached/not-cached column, where "cannot tell" and "incomplete" are the
    same pixel. Anything that gates *behaviour* wants
    :func:`mflux_missing_weights` and its three states instead.
    """
    try:
        return mflux_missing_weights(repo_id) == []
    except Exception:
        return False


def is_weightless_stub(repo_id: str) -> bool:
    """True if ``repo_id``'s config is cached but its weight shards are NOT.

    The "config-only stub" state (0.10.16 dogfood finding ⑥):
    ``config.json`` (and often the tokenizer) sit in the HF cache — from a
    metadata-only ``AutoConfig`` / ``mlx-vlm`` config probe, or an
    interrupted pull that fetched the small files first — while
    ``model*.safetensors`` are absent. To the user the model *looks*
    cached, so ``rapid-mlx serve <alias>`` silently kicks off a multi-GB
    download they didn't expect. (A warm cache commonly holds ~20 Gemma-4
    repos in exactly this state.)

    Distinct from :func:`is_repo_cached`, which is ``False`` for BOTH a
    stub AND a totally-absent repo. This narrows to the stub case so a
    caller can surface "config cached, weights missing — will download"
    instead of a generic notice. Local paths and never-touched repos
    return ``False``.

    Non-text scope (video-gen false-alarm fix): the underlying weight
    probe is mlx-lm's text glob ``model*.safetensors``, which video-gen /
    diffusers repos never satisfy (their weights are component files like
    ``transformer.safetensors`` / ``vae.safetensors``). A fully-cached
    video model would therefore look weightless and mis-fire this notice on
    every serve. :func:`_snapshot_is_complete_split_model` validates the
    mlx-video ``split_model.json`` component manifest so the stub notice
    stays scoped to genuinely-incomplete caches.

    Returns ``False`` on any internal error — a best-effort diagnostic
    must never break an otherwise-fine serve.
    """
    try:
        if os.path.exists(repo_id):
            return False
        from huggingface_hub import try_to_load_from_cache

        # ``try_to_load_from_cache`` returns a str path when the file is
        # in the cache, the ``_CACHED_NO_EXIST`` sentinel when it's known
        # absent, or ``None`` when the repo/file was never fetched. Only
        # a real cached path (str) counts as "config present".
        cached_config = try_to_load_from_cache(repo_id, "config.json")
        if not isinstance(cached_config, str):
            return False
        # A component-split non-text model (video-gen) stores its weights as
        # per-component files the text glob can't see. If its split_model.json
        # manifest lists components and EVERY one is cached, the weights are
        # present — not a stub. Check this BEFORE the text-glob probe so a
        # fully-cached video model doesn't mis-fire the notice.
        if _snapshot_is_complete_split_model(
            repo_id
        ) or _snapshot_is_complete_mflux_model(repo_id):
            return False
        # Config is on disk; the stub is exactly "config present but the
        # loader's weight glob (model*.safetensors) is not satisfied".
        return not is_repo_cached(repo_id)
    except Exception:
        return False


def weightless_stub_notice(repo_id: str) -> str | None:
    """One-line pre-serve notice when ``repo_id`` is a weightless stub.

    Returns a human-readable heads-up string when
    :func:`is_weightless_stub` is True (config cached, weight shards
    missing), else ``None``. Purely informational: the caller prints it
    BEFORE the normal download path runs; it does NOT gate or change
    download behaviour (finding ⑥ asks only to surface the surprise, not
    block it).

    Deliberately size-FREE: computing a byte figure here would fire a
    SECOND synchronous HF metadata request on the startup path, redundant
    with the download path's own size lookup (``_ensure_model_downloaded``)
    and adding latency before boot. The download reports its own progress,
    so the notice just names the surprise without a round-trip.
    """
    if not is_weightless_stub(repo_id):
        return None
    return (
        f"  Note: {repo_id} has its config cached but its model weights are "
        f"missing — serving will start by downloading the missing weights first."
    )


def confirm_or_abort(
    repo_id: str,
    estimated_bytes: int | None,
    *,
    threshold_bytes: int = 10 * 1024**3,  # 10 GiB
    auto_yes_env: str = "RAPID_MLX_AUTO_PULL",
    logfile_hint: str | None = None,
) -> bool:
    """Interactive gate before a large model download begins.

    Returns ``True`` (proceed) without prompting when:

    * the env var ``auto_yes_env`` is set to a truthy value
      (``"1"``/``"true"``/``"yes"``, case-insensitive), OR
    * ``sys.stdin`` is not a TTY (scripts/CI), OR
    * ``estimated_bytes`` is below ``threshold_bytes``.

    When the size estimate is ``None`` (HF lookup failed) we print a
    heads-up but proceed — blocking on a transient API failure would be
    worse than the silent-download problem we're trying to fix.

    Prompt default is Y (``[Y/n]``): the user already typed a subcommand
    naming a specific alias, so pressing Enter is treated as confirmation.
    Only an explicit ``n``/``no`` (case-insensitive) — or Ctrl-C, which is
    mapped to ``n`` internally — triggers the abort hint and
    ``sys.exit(1)``. EOF on stdin is treated as Enter (proceed).
    """
    # Env override always wins.
    env_val = os.environ.get(auto_yes_env, "").strip().lower()
    if env_val in {"1", "true", "yes"}:
        return True

    # Non-interactive: never block; we already burned the user's time
    # if they piped a script that didn't pass --auto-pull.
    if not sys.stdin.isatty():
        return True

    # Unknown size → noisy heads-up but proceed. We get here when the
    # HF metadata API is down or the repo is gated; either way the
    # actual download will surface its own error if there is one.
    if estimated_bytes is None:
        print()
        print(f"  About to download {repo_id}")
        print("    Estimated size: unknown (HF metadata lookup failed)")
        print(
            "    Proceeding without confirmation. Set "
            f"{auto_yes_env}=1 to silence this notice."
        )
        print()
        return True

    # Small downloads don't deserve interruption.
    if estimated_bytes < threshold_bytes:
        return True

    size_str = _format_size(estimated_bytes)
    print()
    print(f"  About to download {repo_id}")
    print(f"    Estimated size: {size_str} (this may take a while on first run)")
    if logfile_hint:
        print(f"    Download progress will appear in {logfile_hint}; tail it to watch.")
    print()
    # Default Y — the user explicitly invoked a subcommand on a specific
    # alias ("rapid-mlx serve qwen…", "rapid-mlx share gemma…"); intent
    # to use that model is clear. Pressing Enter shouldn't punish them
    # with an abort. Ctrl-C still cancels.
    try:
        answer = input("  Continue? [Y/n]: ").strip().lower()
    except EOFError:
        answer = ""  # equivalent to Enter
    except KeyboardInterrupt:
        answer = "n"  # explicit user cancel

    if answer in {"n", "no"}:
        print(
            f"  Aborted. Use 'rapid-mlx pull {repo_id}' to download separately, "
            f"or set {auto_yes_env}=1 to skip this prompt."
        )
        sys.exit(1)
    return True
