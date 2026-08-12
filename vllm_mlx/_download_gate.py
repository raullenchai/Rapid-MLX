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
      2. Otherwise, a single non-empty ``model*.safetensors`` is
         sufficient (covers single-file non-sharded models).

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

    # Single-file (non-sharded) model. Match mlx-lm's actual loader
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
        return None
    snap_dir = os.path.join(repo_root, "snapshots", resolved_sha)
    if not os.path.isdir(snap_dir):
        return None

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
