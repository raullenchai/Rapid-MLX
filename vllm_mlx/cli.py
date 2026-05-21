#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for vllm-mlx.

Commands:
    vllm-mlx serve <model> --port 8000    Start OpenAI-compatible server
    vllm-mlx bench <model>                Run benchmark

Usage:
    vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
    vllm-mlx bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10
"""

import argparse
import os
import sys


def _log_level_choice(value: str) -> str:
    """Argparse ``type`` callable: normalize to upper-case so
    ``--log-level info`` is accepted as ``INFO``. Named (not a lambda)
    so argparse's error messages read sensibly instead of
    ``invalid <lambda> value``.
    """
    return value.upper()


def _print_unknown_model_help(name: str, *, full_path_example: str) -> None:
    """Print fuzzy suggestions + a curated popular-models hint.

    Replaces the older "Did you mean: X?" + "Run `rapid-mlx models`" pattern
    that left users empty-handed when no close fuzzy match existed
    (e.g. ``rapid-mlx chat gemma4-27b`` returned zero suggestions, told the
    user to run another command, and gave no hint of what was actually
    supported). Now: always show *something* — fuzzy matches when we have
    them, curated popular aliases when we don't.
    """
    from vllm_mlx.model_aliases import POPULAR_ALIASES, list_aliases, suggest_similar

    suggestions = suggest_similar(name)
    if suggestions:
        print(f"  Did you mean: {', '.join(suggestions)}?")
    else:
        print(f"  Try one of: {', '.join(POPULAR_ALIASES)}")
    print(f"  Run `rapid-mlx models` to see all {len(list_aliases())} aliases,")
    print(f"  or pass a full path like: {full_path_example}")


def _check_disk_space(model_name: str, force: bool = False) -> None:
    """Verify there's enough disk space to download the model.

    Queries HuggingFace for the repo size and compares with available space
    on the resolved HF cache filesystem (respects ``HF_HOME`` /
    ``HF_HUB_CACHE`` rather than the hard-coded ``~/.cache/huggingface``).

    Behaviour:

    - Model is already a local path → return.
    - ``config.json`` is in the cache → assume already downloaded → return.
    - HF API call fails (offline, gated repo, etc.) → return silently. The
      loader's 404/auth handlers will surface the real error if there is one.
    - Determined size and disk is insufficient → print actionable error
      and ``sys.exit(1)``. ``force=True`` warns instead of aborting.

    The previous behaviour was to print a soft warning then continue. Users
    burned 30+ minutes downloading a 141 GB model on an 8.8 GB disk before
    HF Hub crashed with ``OSError: No space left on device``.
    """
    # Skip if model is a local path that already exists.
    if os.path.exists(model_name):
        return

    # Skip if model is already in the HF cache.
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str) and os.path.exists(cached):
            return
    except Exception:
        pass

    # Query HF for repo size + free space on the actual HF cache filesystem.
    try:
        from huggingface_hub import model_info
        from huggingface_hub.constants import HF_HUB_CACHE

        info = model_info(model_name, files_metadata=True)
        model_size_bytes = sum(
            (s.size or 0)
            for s in (getattr(info, "siblings", None) or [])
            if hasattr(s, "size")
        )
        if model_size_bytes == 0:
            return  # Can't determine size — skip rather than guess.

        # statvfs needs an existing path; HF_HUB_CACHE may not exist yet on
        # a fresh install. Walk up to the first ancestor that does.
        # Resolve to absolute up front so a relative HF_HUB_CACHE doesn't
        # short-circuit to CWD when an ancestor walk hits ".".
        probe = os.path.abspath(HF_HUB_CACHE) if HF_HUB_CACHE else ""
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if not probe or not os.path.exists(probe):
            probe = os.path.expanduser("~")

        stat = os.statvfs(probe)
        available_bytes = stat.f_bavail * stat.f_frsize

        # ~10% headroom for temp files during xet_get / move-into-place.
        required_bytes = int(model_size_bytes * 1.1)
        if available_bytes >= required_bytes:
            return

        model_size_gb = model_size_bytes / (1024**3)
        available_gb = available_bytes / (1024**3)
        need_to_free_gb = (required_bytes - available_bytes) / (1024**3)

        print()
        print("  Error: Insufficient disk space for download.")
        print(f"    Model size:    {model_size_gb:>7.1f} GB")
        print(f"    Free space:    {available_gb:>7.1f} GB  ({probe})")
        print(f"    Need to free:  {need_to_free_gb:>7.1f} GB")
        print()
        print("  Suggestions:")
        print("    - Free disk space, or set HF_HOME to a drive with more room")
        print("    - Pick a smaller variant: rapid-mlx models")
        if not force:
            print(
                "    - Bypass this check (download will likely fail mid-way): "
                "--force-disk-check"
            )
            print()
            sys.exit(1)
        # ``force=True``: warn loudly, let the user proceed at their own risk.
        print("  --force-disk-check set — proceeding anyway.")
        print()
    except SystemExit:
        raise
    except Exception:
        # Network / auth / etc. failures are non-critical — fall through to
        # the loader's own error handling rather than blocking startup on a
        # flaky HF metadata query.
        pass


def _check_memory_capacity(model_name: str) -> None:
    """Pre-flight memory check — warn loudly if loading this model is
    likely to push unified memory past the danger threshold.

    On low-memory Apple Silicon (especially Mac mini M4 24 GB), loading
    a model that forces unified memory past ~85% of total can trip the
    iBoot AMCC async-abort firmware path and **kernel-panic the entire
    machine** rather than raise a userspace OOM. See issue #324.

    This check is best-effort: it warns the user, never aborts. If we
    can't read the model size (offline / gated repo), or psutil isn't
    importable, fall through silently — the existing loader paths still
    surface real failures.

    Working-set estimate is ``model_size * 1.5`` for a typical short
    chat workload — covers KV cache, activations, and OS reserve.
    Long-context (32k+) or high-concurrency serving pushes the
    multiplier higher; the warning under-predicts in those modes
    rather than over-predicts, so a user who configures aggressively
    may still crash. We err on the side of warning earlier than later.

    **Pressure formula uses already-used memory** rather than just
    ``working / total``. The kernel panic fires on absolute unified-
    memory pressure, so a 10 GB model on a 24 GB Mac that already has
    8 GB used by macOS + Chrome lands at projected ``(8 + 15) / 24``
    = 95.8% — kernel-panic territory. The naive formula would have
    reported only 62.5% and stayed silent.
    """
    try:
        import psutil
    except Exception:
        return

    # Resolve model size in bytes — local path, then HF cache, then HF API.
    model_size_bytes = 0
    try:
        if os.path.isdir(model_name):
            for root, _dirs, files in os.walk(model_name):
                for f in files:
                    try:
                        model_size_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        continue
        else:
            from huggingface_hub import model_info, try_to_load_from_cache

            cached = try_to_load_from_cache(model_name, "config.json")
            if isinstance(cached, str) and os.path.exists(cached):
                # Already-downloaded model: walk the snapshot directory.
                snapshot_dir = os.path.dirname(cached)
                for root, _dirs, files in os.walk(snapshot_dir):
                    for f in files:
                        try:
                            model_size_bytes += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            continue
            else:
                info = model_info(model_name, files_metadata=True)
                model_size_bytes = sum(
                    (s.size or 0)
                    for s in (getattr(info, "siblings", None) or [])
                    if hasattr(s, "size")
                )
    except Exception:
        return  # Network / auth failure — fall through.

    if model_size_bytes <= 0:
        return

    try:
        vm = psutil.virtual_memory()
        total_ram_bytes = vm.total
        available_ram_bytes = vm.available
    except Exception:
        return

    if total_ram_bytes <= 0:
        return

    # Projected post-load pressure: already-used + estimated working set.
    # ``available`` is psutil's best estimate of "memory we can grab without
    # swapping," which on macOS includes inactive + cached pages that the
    # kernel will reclaim under pressure. ``total - available`` is therefore
    # a tighter "currently-pinned" floor than ``total - free``.
    estimated_working = int(model_size_bytes * 1.5)
    used_ram_bytes = max(0, total_ram_bytes - available_ram_bytes)
    projected_use = used_ram_bytes + estimated_working
    ratio = projected_use / total_ram_bytes
    if ratio < 0.65:
        return  # Comfortable headroom — no warning.

    model_gb = model_size_bytes / (1024**3)
    working_gb = estimated_working / (1024**3)
    used_gb = used_ram_bytes / (1024**3)
    total_gb = total_ram_bytes / (1024**3)

    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    YELLOW = "\x1b[33m" if is_tty else ""
    RED = "\x1b[31m" if is_tty else ""
    BOLD = "\x1b[1m" if is_tty else ""
    DIM = "\x1b[2m" if is_tty else ""
    RESET = "\x1b[0m" if is_tty else ""

    print()
    if ratio >= 0.85:
        print(
            f"  {RED}{BOLD}!! Memory pressure warning:{RESET} "
            f"this model is likely too large for your hardware."
        )
        print(
            f"  {DIM}Continuing may trigger a macOS kernel panic "
            f"(see issue #324).{RESET}"
        )
    else:
        print(
            f"  {YELLOW}{BOLD}Memory pressure note:{RESET} "
            f"this model uses a large fraction of system RAM."
        )
    print()
    print(f"    Model on disk:           {model_gb:>6.1f} GB")
    print(
        f"    Est. working set:        {working_gb:>6.1f} GB  "
        f"{DIM}(model x 1.5 — short-chat workload; long-context serving will use more){RESET}"
    )
    print(f"    Currently used by OS:    {used_gb:>6.1f} GB")
    print(
        f"    Total system RAM:        {total_gb:>6.1f} GB  "
        f"({ratio * 100:.0f}% projected utilization)"
    )
    print()
    if ratio >= 0.85:
        print("  Apple Silicon firmware can panic the whole system rather than")
        print("  raise an OOM error when unified-memory pressure exceeds the")
        print("  iBoot AMCC threshold. Recommended actions:")
        print()
        print("    - Close other apps to free RAM, or")
        print("    - Pick a smaller model:    rapid-mlx models")
        print(
            "    - Or lower memory headroom: "
            "rapid-mlx serve <model> --gpu-memory-utilization 0.75"
        )
    else:
        print(
            "  If you see crashes or kernel panics, try: --gpu-memory-utilization 0.85"
        )
    print()


def _ensure_model_downloaded(model_name: str) -> None:
    """Pre-fetch a model in the foreground so HF's tqdm progress is visible.

    Used by ``rapid-mlx chat``: the chat REPL spawns ``serve`` as a
    subprocess with stdout/stderr redirected to a log file. If the model
    isn't cached, the user sees a silent multi-minute hang while several
    GB downloads behind the log. Calling ``snapshot_download`` here first
    surfaces the standard HF progress bars on the user's terminal, then
    the spawned server starts as a cache hit.

    No-op when the model is already cached, when ``model_name`` is a local
    path, or when the HF lookup fails (let the loader's own error paths
    handle it).
    """
    if os.path.exists(model_name):
        return
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str) and os.path.exists(cached):
            return
    except Exception:
        return

    # Disk-space gate: a 20 GB partial download that fails on the last
    # shard wastes the user's time. ``_check_disk_space`` queries HF for
    # the repo size and aborts with a clear message + exit(1) if there
    # isn't enough room on the resolved HF cache filesystem.
    _check_disk_space(model_name)

    try:
        from huggingface_hub import model_info, snapshot_download

        size_gb = 0.0
        try:
            info = model_info(model_name, files_metadata=True)
            size_bytes = sum(
                (s.size or 0)
                for s in (getattr(info, "siblings", None) or [])
                if hasattr(s, "size")
            )
            size_gb = size_bytes / (1024**3)
        except Exception:
            pass

        is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        BOLD = "\x1b[1m" if is_tty else ""
        DIM = "\x1b[2m" if is_tty else ""
        RESET = "\x1b[0m" if is_tty else ""
        if size_gb > 0:
            print(
                f"\n  {BOLD}First-time download{RESET} — "
                f"fetching {model_name} {DIM}(~{size_gb:.1f} GB){RESET} "
                "from HuggingFace ..."
            )
        else:
            print(
                f"\n  {BOLD}First-time download{RESET} — "
                f"fetching {model_name} from HuggingFace ..."
            )

        snapshot_download(model_name)
        print()
    except SystemExit:
        # _check_disk_space aborts via sys.exit(1) — let it through.
        raise
    except Exception as e:
        # Definitive 404s are surfaced so callers (e.g. ``/model bogus``)
        # can refuse fast instead of spawning a doomed serve subprocess
        # that fails after ``--ready-timeout``. Other transient errors
        # (network, auth) fall through silently — the spawned server's
        # own loader will retry and surface a real error if needed.
        from huggingface_hub.utils import RepositoryNotFoundError

        if isinstance(e, RepositoryNotFoundError) or "404" in str(e):
            raise RuntimeError(f"Model {model_name!r} not found on HuggingFace") from e
        print(f"\n  Pre-download skipped ({type(e).__name__}); server will retry.")


def serve_command(args):
    """Start the OpenAI-compatible server."""
    import logging
    import os
    import sys

    import uvicorn

    # Interactive auto-upgrade prompt — when serve runs interactively and a
    # newer release is available, ask once before booting the model. Honors
    # RAPID_MLX_DISABLE_VERSION_CHECK, CI=1, and non-TTY stdin. Cached
    # piggy-backs on the existing staleness check's cache (24h TTL).
    from vllm_mlx._version_check import prompt_upgrade_if_available

    if prompt_upgrade_if_available():
        sys.exit(0)

    # Import unified server
    from . import server
    from .middleware.auth import configure_rate_limiter
    from .scheduler import SchedulerConfig
    from .server import app, load_model

    logger = logging.getLogger(__name__)
    uvicorn_log_level = server.configure_logging(args.log_level)

    # Validate tool calling arguments
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        print("Error: --enable-auto-tool-choice requires --tool-call-parser")
        print("Example: --enable-auto-tool-choice --tool-call-parser mistral")
        sys.exit(1)

    # Validate gpu-memory-utilization range
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        print(
            "Error: --gpu-memory-utilization must be between 0.0 (exclusive) and 1.0 (inclusive)"
        )
        sys.exit(1)

    # Auto-detect parser config from model name when not explicitly set.
    # --no-tool-call-parser / --no-reasoning-parser are escape hatches
    # (SOP §10): if the user opts out, do NOT let the AliasProfile auto-
    # populate args.tool_call_parser / args.reasoning_parser. Past
    # incidents: #393-class (auto-detect false positive with no opt-out).
    _opt_out_tool = getattr(args, "no_tool_call_parser", False)
    _opt_out_reasoning = getattr(args, "no_reasoning_parser", False)
    if args.tool_call_parser and _opt_out_tool:
        print(
            "error: --tool-call-parser and --no-tool-call-parser are "
            "mutually exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.reasoning_parser and _opt_out_reasoning:
        print(
            "error: --reasoning-parser and --no-reasoning-parser are "
            "mutually exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.tool_call_parser or not args.reasoning_parser:
        try:
            from .model_auto_config import detect_model_config

            auto_config = detect_model_config(args.model)
            if auto_config:
                if (
                    not args.tool_call_parser
                    and not _opt_out_tool
                    and auto_config.tool_call_parser
                ):
                    args.tool_call_parser = auto_config.tool_call_parser
                    args.enable_auto_tool_choice = True
                    logger.info(
                        f"Auto-configured --tool-call-parser {auto_config.tool_call_parser}"
                    )
                if (
                    not args.reasoning_parser
                    and not _opt_out_reasoning
                    and not args.no_thinking
                    and auto_config.reasoning_parser
                ):
                    args.reasoning_parser = auto_config.reasoning_parser
                    logger.info(
                        f"Auto-configured --reasoning-parser {auto_config.reasoning_parser}"
                    )
        except Exception as e:
            logger.debug(f"Auto-detection failed (non-fatal): {e}")
    if _opt_out_tool:
        logger.info(
            "Tool-call parser auto-detection disabled via --no-tool-call-parser"
        )
    if _opt_out_reasoning:
        logger.info(
            "Reasoning parser auto-detection disabled via --no-reasoning-parser"
        )

    # Pass alias info to server (for /v1/models)
    server._model_alias = getattr(args, "_original_alias", None)

    # Configure server security settings
    server._api_key = args.api_key
    server._default_timeout = args.timeout
    # Configure CORS
    cors_origins = args.cors_origins if args.cors_origins else ["*"]
    server.configure_cors(cors_origins)
    if args.rate_limit > 0:
        server._rate_limiter = configure_rate_limiter(args.rate_limit, enabled=True)

    # Configure GC control
    gc_control = args.gc_control and not args.no_gc_control
    server._gc_control = gc_control

    # Configure --no-thinking: suppress chain-of-thought in chat template
    server._no_thinking = args.no_thinking

    # Configure system prompt pinning
    server._pin_system_prompt = args.pin_system_prompt

    # Configure tool calling
    if args.enable_auto_tool_choice and args.tool_call_parser:
        server._enable_auto_tool_choice = True
        server._tool_call_parser = args.tool_call_parser
        server._enable_tool_logits_bias = getattr(
            args, "enable_tool_logits_bias", False
        )
    else:
        server._enable_auto_tool_choice = False
        server._tool_call_parser = None
        server._enable_tool_logits_bias = False

    # Configure generation defaults
    if args.default_temperature is not None:
        server._default_temperature = args.default_temperature
    if args.default_top_p is not None:
        server._default_top_p = args.default_top_p
    if args.default_top_k is not None:
        server._default_top_k = args.default_top_k
    if args.default_min_p is not None:
        server._default_min_p = args.default_min_p
    if args.default_repetition_penalty is not None:
        server._default_repetition_penalty = args.default_repetition_penalty
    if args.default_presence_penalty is not None:
        server._default_presence_penalty = args.default_presence_penalty
    if args.default_frequency_penalty is not None:
        server._default_frequency_penalty = args.default_frequency_penalty

    # Configure reasoning parser
    if args.reasoning_parser:
        try:
            from .reasoning import get_parser

            parser_cls = get_parser(args.reasoning_parser)
            server._reasoning_parser = parser_cls()
            server._reasoning_parser_name = args.reasoning_parser
            logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ImportError as e:
            print(f"Error: Failed to import reasoning module: {e}")
            sys.exit(1)
        except Exception as e:
            print(
                f"Error: Failed to initialize reasoning parser "
                f"'{args.reasoning_parser}': {e}"
            )
            sys.exit(1)
    else:
        server._reasoning_parser = None

    # DFlash mutual-exclusion gate fires BEFORE the startup banner so
    # the user sees a clean error instead of an optimistic "Features:
    # dflash" line immediately followed by an exit. The deeper SchedulerConfig
    # mutex (suffix vs. mtp) stays below since it doesn't involve DFlash.
    if args.enable_dflash and (args.suffix_decoding or args.enable_mtp):
        print(
            "\n  Error: --enable-dflash cannot combine with --suffix-decoding "
            "or --enable-mtp. DFlash runs a dedicated single-user server "
            "that bypasses BatchedEngine; other spec-decode methods only "
            "apply to the BatchedEngine path.\n"
        )
        sys.exit(1)

    # DFlash eligibility gate fires here, BEFORE the startup banner —
    # so the user sees a clean error rather than an optimistic "DFlash
    # enabled" feature line followed by an exit. Cheap (just reads
    # aliases.json + checks the module spec); no model load yet.
    if args.enable_dflash:
        from .model_aliases import resolve_profile
        from .speculative.dflash import DFlashUnavailable, check
        from .speculative.dflash.eligibility import have_runtime

        _alias_name = getattr(args, "_original_alias", None) or args.model
        _profile = resolve_profile(_alias_name)
        if _profile is None:
            print(
                f"\n  Error: --enable-dflash requires a known alias, got "
                f"{_alias_name!r}. DFlash eligibility is recorded per-alias "
                f"in aliases.json; ad-hoc HuggingFace paths can't be "
                f"validated. Try ``rapid-mlx info qwen3.5-27b-8bit``.\n"
            )
            sys.exit(1)
        try:
            check(_profile, alias=_alias_name)
        except DFlashUnavailable as e:
            print(f"\n  Error: {e}\n")
            sys.exit(1)
        if not have_runtime():
            print(
                "\n  Error: --enable-dflash requires mlx-vlm 0.5.0+ for the "
                "DFlash drafter hooks. Install with: "
                "``pip install 'rapid-mlx[dflash]'``.\n"
            )
            sys.exit(1)

        # Warn about flags that BatchedEngine honours but the DFlash
        # server doesn't — better to surface this once at startup than
        # to let users wonder why their tuning has no effect. Inspected
        # against the actual argparse Namespace so we only mention flags
        # the user explicitly set away from their default.
        _GPU_MEM_DEFAULT = 0.90  # keep in sync with the serve_parser default
        _dflash_ignored: list[str] = []
        if getattr(args, "enable_prefix_cache", False):
            _dflash_ignored.append("--enable-prefix-cache")
        if getattr(args, "kv_cache_quantization", None):
            _dflash_ignored.append("--kv-cache-quantization")
        # gpu-memory-utilization defaults to 0.90 (not None) in the serve
        # parser, so an ``is not None`` check would fire on every invocation.
        # Compare to the real default — only warn when the user explicitly
        # tuned it. Tolerate a tiny float-equality slack for safety.
        _gpu_mem = getattr(args, "gpu_memory_utilization", _GPU_MEM_DEFAULT)
        if _gpu_mem is not None and abs(_gpu_mem - _GPU_MEM_DEFAULT) > 1e-6:
            _dflash_ignored.append("--gpu-memory-utilization")
        if getattr(args, "enable_auto_tool_choice", False):
            _dflash_ignored.append("--enable-auto-tool-choice")
        if getattr(args, "tool_call_parser", None):
            _dflash_ignored.append("--tool-call-parser")
        if getattr(args, "reasoning_parser", None):
            _dflash_ignored.append("--reasoning-parser")
        if getattr(args, "embedding_model", None):
            _dflash_ignored.append("--embedding-model")
        if getattr(args, "mcp_config", None):
            _dflash_ignored.append("--mcp-config")
        if _dflash_ignored:
            print(
                "\n  ⚠ The following flags are ignored under --enable-dflash"
                "\n    (DFlash uses a dedicated single-user server that bypasses"
                "\n    BatchedEngine):"
                f"\n      {', '.join(_dflash_ignored)}"
                "\n    Drop them from your serve command, or run without"
                "\n    --enable-dflash if you need them.\n"
            )

    # Startup summary
    print()
    print("  Rapid-MLX")
    print("  ─────────")
    features = []
    if args.enable_auto_tool_choice:
        bias_info = (
            " + logits bias" if getattr(args, "enable_tool_logits_bias", False) else ""
        )
        features.append(f"tools: {args.tool_call_parser}{bias_info}")
    if args.reasoning_parser:
        features.append(f"reasoning: {args.reasoning_parser}")
    if args.api_key:
        features.append("auth: on")
    if args.rate_limit > 0:
        features.append(f"rate-limit: {args.rate_limit}/min")
    if args.cloud_model:
        features.append(f"cloud: {args.cloud_model}")
    if gc_control:
        features.append("gc-control")
    if args.pin_system_prompt:
        features.append("pin-system-prompt")
    if args.cors_origins:
        features.append(f"cors: {', '.join(args.cors_origins)}")
    if args.enable_dflash:
        features.append("dflash: single-user")
    if features:
        print(f"  Features: {', '.join(features)}")
    print(f"  Model: {args.model}")
    # Store MCP config path for FastAPI startup
    if args.mcp_config:
        print(f"MCP config: {args.mcp_config}")
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Pre-load embedding model if specified
    if args.embedding_model:
        print(f"Pre-loading embedding model: {args.embedding_model}")
        server.load_embedding_model(args.embedding_model, lock=True)
        print(f"Embedding model loaded: {args.embedding_model}")

    # Warn about deprecated flags
    if getattr(args, "simple_engine", False):
        print(
            "\n  ⚠ --simple-engine is deprecated and has no effect."
            "\n    BatchedEngine is now the sole engine — it handles both"
            "\n    single-user and multi-user workloads with equal performance.\n"
        )
    if getattr(args, "kv_bits", None) is not None:
        print(
            "\n  ⚠ --kv-bits is deprecated and has no effect."
            "\n    For prefix cache quantization, use --kv-cache-quantization instead.\n"
        )
    if getattr(args, "draft_model", None):
        print(
            "\n  ⚠ --draft-model is deprecated and has no effect."
            "\n    For DFlash speculative decoding, use --enable-dflash "
            "(requires a DFlash-eligible alias). "
            "For MTP, use --enable-mtp (requires a model with MTP head).\n"
        )
    if getattr(args, "specprefill", False):
        print("\n  ⚠ --specprefill is deprecated and has no effect.\n")

    # Mutual exclusion: turboquant vs standard quantization
    if args.kv_cache_turboquant and args.kv_cache_quantization:
        print(
            "\n  Error: --kv-cache-turboquant and --kv-cache-quantization are "
            "mutually exclusive. Choose one.\n"
        )
        sys.exit(1)

    # Mutual exclusion: only one spec-decode method may wrap _step at a time.
    # (The DFlash-vs-{suffix,mtp} check is upstream, before the banner.)
    if args.suffix_decoding and args.enable_mtp:
        print(
            "\n  Error: --suffix-decoding and --enable-mtp are mutually "
            "exclusive (both monkey-patch the BatchGenerator step). "
            "Pick one.\n"
        )
        sys.exit(1)

    # Build scheduler config
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    scheduler_config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        max_concurrent_requests=args.max_concurrent_requests,
        prefill_batch_size=args.prefill_batch_size,
        completion_batch_size=args.completion_batch_size,
        enable_prefix_cache=enable_prefix_cache,
        prefix_cache_size=args.prefix_cache_size,
        # Memory-aware cache options
        use_memory_aware_cache=not args.no_memory_aware_cache,
        cache_memory_mb=args.cache_memory_mb,
        cache_memory_percent=args.cache_memory_percent,
        # Paged cache options
        use_paged_cache=args.use_paged_cache,
        paged_cache_block_size=args.paged_cache_block_size,
        max_cache_blocks=args.max_cache_blocks,
        # Chunked prefill
        chunked_prefill_tokens=args.chunked_prefill_tokens,
        # Prefill step size (chunk size). Must be plumbed here — BatchedEngine
        # reads it off scheduler_config only; the legacy load_model kwarg was
        # accepted but never used. See #400 and the CLI ↔ Config fidelity
        # audit at scripts/audit_cli_config_fidelity.py.
        prefill_step_size=args.prefill_step_size,
        # MTP
        enable_mtp=args.enable_mtp,
        mtp_num_draft_tokens=args.mtp_num_draft_tokens,
        mtp_optimistic=args.mtp_optimistic,
        # SuffixDecoding
        enable_suffix_decoding=args.suffix_decoding,
        suffix_max_draft=args.suffix_max_draft,
        suffix_max_suffix_len=args.suffix_max_suffix_len,
        suffix_min_confidence=args.suffix_min_confidence,
        suffix_min_draft_len=args.suffix_min_draft_len,
        # KV cache quantization
        kv_cache_quantization=args.kv_cache_quantization,
        kv_cache_quantization_bits=args.kv_cache_quantization_bits,
        kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
        kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        # TurboQuant V-only compression
        kv_cache_turboquant=args.kv_cache_turboquant,
        kv_cache_turboquant_bits=args.kv_cache_turboquant_bits,
        kv_cache_turboquant_group_size=args.kv_cache_turboquant_group_size,
    )

    print("Mode: Continuous batching (for multiple concurrent users)")
    if args.chunked_prefill_tokens > 0:
        print(f"Chunked prefill: {args.chunked_prefill_tokens} tokens per step")
    if args.enable_mtp:
        print(f"MTP: enabled, draft_tokens={args.mtp_num_draft_tokens}")
    if args.suffix_decoding:
        print(
            f"SuffixDecoding: enabled, max_draft={args.suffix_max_draft}, "
            f"max_suffix={args.suffix_max_suffix_len}, "
            f"min_conf={args.suffix_min_confidence}"
        )
    print(f"Stream interval: {args.stream_interval} tokens")
    if args.use_paged_cache:
        print(
            f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
        )
    elif enable_prefix_cache and not args.no_memory_aware_cache:
        cache_info = (
            f"{args.cache_memory_mb}MB"
            if args.cache_memory_mb
            else f"{args.cache_memory_percent * 100:.0f}% of RAM"
        )
        print(f"Memory-aware cache: {cache_info}")
        if args.kv_cache_turboquant:
            bits_str = (
                str(args.kv_cache_turboquant_bits)
                if args.kv_cache_turboquant_bits
                else "auto"
            )
            print(
                f"TurboQuant V-cache: {bits_str}-bit, "
                f"group_size={args.kv_cache_turboquant_group_size} (K stays FP16)"
            )
        elif args.kv_cache_quantization:
            print(
                f"KV cache quantization: {args.kv_cache_quantization_bits}-bit, "
                f"group_size={args.kv_cache_quantization_group_size}"
            )
    elif enable_prefix_cache:
        print(f"Prefix cache: max_entries={args.prefix_cache_size}")

    # Check port availability before loading model (avoid wasting RAM on conflict).
    # Set SO_REUSEADDR to match uvicorn's bind behavior — without it, this
    # preflight fails on a port still in TCP TIME_WAIT (e.g. just after a
    # previous rapid-mlx process exited), even though uvicorn would happily
    # bind it. Caused spurious "port in use" errors for back-to-back server
    # starts in the validation pipeline.
    import socket

    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        _sock.bind((args.host, args.port))
        _sock.close()
    except OSError:
        print(f"\n  Error: Port {args.port} is already in use.")
        print(
            f"  Try a different port: rapid-mlx serve {args.model} --port {args.port + 1}"
        )
        sys.exit(1)

    # Check disk space before downloading model
    _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))

    # Pre-flight memory check — warn (don't abort) if model + working set
    # would push unified memory past the kernel-panic threshold (issue #324).
    _check_memory_capacity(args.model)

    # DFlash fork: when --enable-dflash is set, skip BatchedEngine entirely
    # and run the dedicated DFlash server. The eligibility check above has
    # already validated the alias, so by here we have a known-good profile.
    if args.enable_dflash:
        # DFlash IS a speculative-decode path. The --no-spec-decode escape
        # hatch (SOP §10) must reject it here — otherwise the user thinks
        # they've disabled spec-decode but DFlash silently proceeds via
        # its dedicated server, never touching EngineCore / ModelConfig.
        if getattr(args, "no_spec_decode", False):
            print(
                "error: --enable-dflash and --no-spec-decode are mutually "
                "exclusive — DFlash is a speculative-decode mode.",
                file=sys.stderr,
            )
            sys.exit(2)
        from .model_aliases import resolve_profile
        from .speculative.dflash.server import run_dflash_server

        _alias_name = getattr(args, "_original_alias", None) or args.model
        _profile = resolve_profile(_alias_name)
        # The eligibility check at top of serve_command guarantees this
        # passes — assert to be defensive against future refactors.
        assert _profile is not None and _profile.supports_dflash, (
            f"DFlash profile invariant violated for {_alias_name!r}"
        )
        run_dflash_server(
            main_model_repo=_profile.hf_path,
            drafter_repo=_profile.dflash_draft_model,  # validated non-None by _coerce
            host=args.host,
            port=args.port,
            served_model_name=args.served_model_name or _alias_name,
            default_max_tokens=args.max_tokens,
            cors_origins=cors_origins,
            uvicorn_log_level=uvicorn_log_level,
            no_thinking=args.no_thinking,
        )
        return

    # Load model with unified server
    if args.mllm and args.no_mllm:
        print(
            "error: --mllm and --no-mllm are mutually exclusive — "
            "pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if getattr(args, "force_hybrid", False) and getattr(args, "no_hybrid", False):
        print(
            "error: --force-hybrid and --no-hybrid are mutually exclusive — "
            "pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    if getattr(args, "force_spec_decode", False) and getattr(
        args, "no_spec_decode", False
    ):
        print(
            "error: --force-spec-decode and --no-spec-decode are mutually "
            "exclusive — pick one to override auto-detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        load_model(
            args.model,
            scheduler_config=scheduler_config,
            stream_interval=args.stream_interval,
            max_tokens=args.max_tokens,
            force_mllm=args.mllm,
            force_text=args.no_mllm,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cloud_model=args.cloud_model,
            cloud_threshold=args.cloud_threshold,
            cloud_api_base=args.cloud_api_base,
            cloud_api_key=args.cloud_api_key,
            served_model_name=args.served_model_name,
            mtp=args.enable_mtp,
            force_hybrid=getattr(args, "force_hybrid", False),
            no_hybrid=getattr(args, "no_hybrid", False),
            force_spec_decode=getattr(args, "force_spec_decode", False),
            no_spec_decode=getattr(args, "no_spec_decode", False),
        )
    except Exception as e:
        # Show clean error instead of raw traceback. Catch the typed
        # HF exception class for the 404 case; fall back to substring
        # match for legacy callers (older huggingface_hub) and for
        # non-HF errors that still spell out "not found".
        from huggingface_hub.utils import RepositoryNotFoundError

        is_404 = isinstance(e, RepositoryNotFoundError) or (
            "404" in str(e) or "not found" in str(e).lower()
        )
        if is_404:
            shown = getattr(args, "_original_alias", args.model)
            print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
            _print_unknown_model_help(
                shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
            )
        else:
            print(f"\n  Error loading model: {e}")
        sys.exit(1)

    # Start server
    # Note: Metal shader warmup runs in the FastAPI lifespan hook (server.py).
    # The "Ready:" banner is printed FROM that hook once warmup completes and
    # the port is actually bound — printing it here would lie to users who
    # curl immediately and get connection-refused while shaders compile.
    print()
    host_display = "localhost" if args.host == "0.0.0.0" else args.host
    print(
        f"  Starting server on http://{host_display}:{args.port} (warming up — this can take a few seconds)"
    )
    from vllm_mlx._version_check import print_staleness_warning_if_any

    print_staleness_warning_if_any()
    print()

    # Stash host/port so the lifespan hook can print the real "Ready:" banner
    # after warmup. ServerConfig.bind_host/bind_port → used in server.lifespan().
    from vllm_mlx.config import get_config

    _cfg = get_config()
    _cfg.bind_host = host_display
    _cfg.bind_port = args.port

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
    )


def bench_command(args):
    """Run benchmark."""
    import asyncio
    import time

    from mlx_lm import load

    from .engine_core import AsyncEngineCore, EngineConfig
    from .request import SamplingParams
    from .scheduler import SchedulerConfig

    _check_disk_space(args.model, force=getattr(args, "force_disk_check", False))
    _check_memory_capacity(args.model)

    # Handle prefix cache flags
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    async def run_benchmark():
        print(f"Loading model: {args.model}")
        try:
            model, tokenizer = load(args.model)
        except Exception as e:
            # Mirror serve_command: clean message instead of a 30-line
            # traceback when the user typed a missing repo / bad alias.
            from huggingface_hub.utils import RepositoryNotFoundError

            is_404 = isinstance(e, RepositoryNotFoundError) or (
                "404" in str(e) or "not found" in str(e).lower()
            )
            if is_404:
                shown = getattr(args, "_original_alias", args.model)
                print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
                _print_unknown_model_help(
                    shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
                )
            else:
                print(f"\n  Error loading model: {e}")
            sys.exit(1)

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_concurrent_requests=getattr(args, "max_concurrent_requests", 256),
            prefill_batch_size=args.prefill_batch_size,
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_quantization_bits=args.kv_cache_quantization_bits,
            kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        )
        engine_config = EngineConfig(
            model_name=args.model,
            scheduler_config=scheduler_config,
        )

        if args.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )

        # Generate prompts
        prompts = [
            f"Write a short poem about {topic}."
            for topic in [
                "nature",
                "love",
                "technology",
                "space",
                "music",
                "art",
                "science",
                "history",
                "food",
                "travel",
            ][: args.num_prompts]
        ]

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
        )

        print(
            f"\nRunning benchmark with {len(prompts)} prompts, max_tokens={args.max_tokens}"
        )
        print("-" * 50)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
            await asyncio.sleep(0.1)  # Warm up

            start_time = time.perf_counter()

            # Add all requests
            request_ids = []
            for prompt in prompts:
                rid = await engine.add_request(prompt, params)
                request_ids.append(rid)

            # Collect all outputs
            async def get_output(rid):
                async for out in engine.stream_outputs(rid, timeout=120):
                    if out.finished:
                        return out
                return None

            results = await asyncio.gather(*[get_output(r) for r in request_ids])

            total_time = time.perf_counter() - start_time

        # Calculate stats
        for r in results:
            if r:
                total_prompt_tokens += r.prompt_tokens
                total_completion_tokens += r.completion_tokens

        total_tokens = total_prompt_tokens + total_completion_tokens

        print("\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Prompts/second: {len(prompts) / total_time:.2f}")
        print(f"  Total prompt tokens: {total_prompt_tokens}")
        print(f"  Total completion tokens: {total_completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Tokens/second: {total_completion_tokens / total_time:.2f}")
        print(f"  Throughput: {total_tokens / total_time:.2f} tok/s")

    asyncio.run(run_benchmark())


def models_command(_args):
    """List available model aliases with their per-model profile capabilities.

    Pulls from ``list_profiles()`` so every alias's ``tool_call_parser`` /
    ``reasoning_parser`` / ``is_hybrid`` / ``supports_spec_decode`` /
    ``suffix_decoding_tier`` shows up in the table — letting users pick a
    model on capabilities, not just on name.
    """
    from vllm_mlx._version_check import print_staleness_warning_if_any
    from vllm_mlx.model_aliases import list_profiles

    print_staleness_warning_if_any()

    profiles = list_profiles()
    print()
    print(f"  Available models ({len(profiles)} aliases)")

    # Widths sized to fit the longest values currently in aliases.json:
    # alias 22 (qwen3.5-122b-mxfp4 etc.), tool 16 (qwen3_coder_xml + 1 pad),
    # reasoning 12 (deepseek_r1 + 1 pad), spec 10 ("✗ hybrid"), tier 11,
    # dflash 7 ("✓ ready"/"—").
    cols = (
        ("Alias", 22),
        ("Tools", 16),
        ("Reasoning", 12),
        ("Spec-Decode", 10),
        ("Suffix Tier", 11),
        ("DFlash", 7),
    )
    width = sum(w for _, w in cols) + len(cols) - 1
    sep = "  " + "─" * width
    header = "  " + " ".join(f"{name:<{w}}" for name, w in cols)
    print(sep)
    print(header)
    print(sep)

    for alias in sorted(profiles.keys()):
        p = profiles[alias]
        tools = p.tool_call_parser or "—"
        reasoning = p.reasoning_parser or "—"
        if p.is_hybrid:
            # Hybrid models cannot use spec-decode or suffix-decode regardless
            # of the supports_spec_decode flag (mlx-lm BatchGenerator gate).
            spec = "✗ hybrid"
            tier = "n/a"
        else:
            spec = "✓" if p.supports_spec_decode else "✗"
            tier = p.suffix_decoding_tier
        # DFlash column — eligible aliases show ✓, everything else "—" so
        # the visual scan immediately surfaces what supports it. We don't
        # re-run the eligibility gate here (which would also check that
        # mlx-vlm 0.5.0+ is installed) — that's a runtime concern; the
        # registry column is pure declarative state.
        dflash = "✓" if p.supports_dflash else "—"
        row = (
            f"  {alias:<22} {tools:<16} {reasoning:<12} "
            f"{spec:<10} {tier:<11} {dflash:<7}"
        )
        print(row)

    print(sep)
    print()
    print("  Tip: `rapid-mlx info <alias>` for the full per-model profile")
    print("       `rapid-mlx pull <alias>` to download")
    print("       `rapid-mlx chat <alias>` for an interactive REPL")
    print("       `rapid-mlx serve <alias>` for an OpenAI-compatible server")
    print()


def pull_command(args):
    """Download a model to the HuggingFace cache without serving."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    repo_id = args.model  # already alias-resolved by main()

    print(f"\n  Pulling {repo_id} ...")
    try:
        path = snapshot_download(repo_id)
    except Exception as e:
        is_404 = isinstance(e, RepositoryNotFoundError) or (
            "404" in str(e) or "not found" in str(e).lower()
        )
        if is_404:
            shown = getattr(args, "_original_alias", repo_id)
            print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
            _print_unknown_model_help(
                shown, full_path_example="mlx-community/Qwen3.5-9B-4bit"
            )
            sys.exit(1)
        raise
    print(f"  Cached at: {path}")


def rm_command(args):
    """Remove a model from the HuggingFace cache."""
    from huggingface_hub import scan_cache_dir

    repo_id = args.model
    cache = scan_cache_dir()
    # Filter by repo_type=="model" — same repo_id can refer to a dataset or
    # space, and we don't want ``rapid-mlx rm foo`` deleting a dataset.
    matching = [
        r for r in cache.repos if r.repo_id == repo_id and r.repo_type == "model"
    ]
    if not matching:
        print(f"\n  '{repo_id}' is not in the HuggingFace cache.")
        print("  Nothing to remove.")
        sys.exit(1)

    repo = matching[0]
    revisions = [rev.commit_hash for rev in repo.revisions]
    strategy = cache.delete_revisions(*revisions)
    print(f"\n  Removing {repo_id} ({strategy.expected_freed_size_str}) ...")
    strategy.execute()
    print("  Done.")


def ps_command(_args):
    """List running rapid-mlx servers (process scan)."""
    import time

    import psutil

    rows: list[tuple[int, str, str, str]] = []
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmd = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not any(
            ("rapid-mlx" in c or "vllm_mlx" in c) and "serve" in cmd for c in cmd
        ):
            continue

        # Extract model arg and --port flag. argparse accepts options
        # before positionals, so the model is the first non-flag token
        # after `serve` whose prior token isn't a value-taking flag.
        # The small list of flags here is conservative; unknown flags
        # are assumed to NOT take a value.
        VALUE_FLAGS = {
            "--host",
            "--port",
            "--api-key",
            "--tool-call-parser",
            "--reasoning-parser",
            "--log-level",
            "--mcp-config",
            "--cors-origins",
            "--cloud-model",
            "--cloud-api-base",
            "--cloud-api-key",
            "--served-model-name",
            "--max-tokens",
            "--gpu-memory-utilization",
        }
        model = "(unknown)"
        port = "8000"  # serve's default
        try:
            i = cmd.index("serve") + 1
            # Pre-PR this loop ``break``ed on the first positional, so a
            # ``rapid-mlx serve qwen3.5-4b --port 8005`` ended with
            # port="8000" because the positional model token came before
            # ``--port``. Keep scanning for flags after we've captured the
            # model — argparse accepts them on either side.
            model_seen = False
            while i < len(cmd):
                tok = cmd[i]
                if tok.startswith("--"):
                    if "=" in tok:
                        key, val = tok.split("=", 1)
                        if key == "--port":
                            port = val
                        i += 1
                    elif tok in VALUE_FLAGS:
                        if tok == "--port" and i + 1 < len(cmd):
                            port = cmd[i + 1]
                        i += 2
                    else:
                        i += 1
                else:
                    if not model_seen:
                        model = tok
                        model_seen = True
                    i += 1
        except ValueError:
            pass

        uptime_s = max(0, int(time.time() - proc.info["create_time"]))
        h, m = uptime_s // 3600, (uptime_s % 3600) // 60
        uptime = f"{h}h{m:02d}m" if h else f"{m}m{uptime_s % 60:02d}s"
        rows.append((proc.info["pid"], port, model, uptime))

    if not rows:
        print("\n  No rapid-mlx servers running.")
        return

    print()
    print(f"  {'PID':<8}{'PORT':<8}{'MODEL':<40}{'UPTIME':<10}")
    print(f"  {'-' * 66}")
    # Sort numerically by port — string sort would put "10000" before "8000".
    for pid, port, model, uptime in sorted(rows, key=lambda r: int(r[1])):
        print(f"  {pid:<8}{port:<8}{model:<40}{uptime:<10}")
    print()


def _spawn_chat_server(
    model: str,
    log_path: str,
    served_name: str | None = None,
    *,
    register_in: list | None = None,
) -> tuple[object, str]:
    """Spawn a `serve` subprocess on an ephemeral port for chat REPL use.

    Returns (Popen handle, base_url).

    ``register_in`` is an optional list (typically the chat REPL's
    ``_active_procs``). When provided, the new ``Popen`` is appended to it
    *immediately* after construction — narrowing the SIGTERM-orphan race
    that exists between ``Popen()`` returning and the caller registering
    the handle. Caller-side ``register_in.append(proc)`` would still leave
    one Python statement of unprotected window; doing it inside this
    function closes that window for the caller.

    If ``served_name`` is given, it is passed via ``--served-model-name`` so
    the spawned server exposes the alias as the API model name (e.g. user
    typed ``qwen3.5-4b`` → API requests use ``qwen3.5-4b`` rather than the
    expanded HF path).
    """
    import socket
    import subprocess

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "WARNING",
    ]
    if served_name and served_name != model:
        cmd.extend(["--served-model-name", served_name])
    log = open(log_path, "w")  # noqa: SIM115 — kept open for proc lifetime
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except (OSError, ValueError):
        # Popen raised before constructing the child — the log handle
        # would otherwise leak. Re-raise after closing.
        log.close()
        raise
    # Register first so a SIGTERM landing between here and the caller's
    # next statement still tears the child down.
    if register_in is not None:
        register_in.append(proc)
    # Stash the log handle and path on the proc object so the chat REPL
    # can close+unlink them when the proc is torn down (fixes the file
    # descriptor + tempfile leak across `/model` swaps).
    proc._rapid_mlx_log = log
    proc._rapid_mlx_log_path = log_path
    return proc, base_url


def _wait_for_chat_server(base_url: str, proc, timeout_s: int = 600) -> None:
    """Block until /health/ready returns 200, the proc exits, or timeout.

    On a TTY, draws a spinner + elapsed-seconds counter to stderr so the
    user can see the chat REPL is alive while the spawned server loads
    weights (typically 20-90 s for 4-30 B models on Apple Silicon). The
    line is erased before this function returns so the caller's next
    print lands on a clean line.
    """
    import time

    import requests

    is_tty = sys.stderr.isatty()
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    cyan = "\x1b[36m" if is_tty else ""
    dim = "\x1b[2m" if is_tty else ""
    reset = "\x1b[0m" if is_tty else ""
    start = time.monotonic()
    deadline = start + timeout_s
    tick = 0

    def _draw():
        if not is_tty:
            return
        elapsed = int(time.monotonic() - start)
        ch = spinner[tick % len(spinner)]
        sys.stderr.write(
            f"\r  {cyan}{ch}{reset} loading model ... {dim}{elapsed}s{reset}"
        )
        sys.stderr.flush()

    def _clear():
        if not is_tty:
            return
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()

    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {proc.returncode}); "
                    "see chat-server.log for details"
                )
            # Animate the spinner at 10 fps; only poll /health once a
            # second to keep the spinner smooth and the network polite.
            if tick % 10 == 0:
                try:
                    r = requests.get(f"{base_url}/health/ready", timeout=2)
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
            _draw()
            time.sleep(0.1)
            tick += 1
    finally:
        _clear()
    raise TimeoutError(
        f"server did not become ready within {timeout_s}s "
        "(large models can take longer — pass --ready-timeout)"
    )


def _has_short_pattern_dominating_suffix(
    text: str,
    *,
    window: int = 600,
    max_period: int = 300,
) -> bool:
    """Return True if the trailing ``window`` chars of ``text`` are
    periodic with a cycle length ≤``max_period``.

    Catches the degenerate-model cases the rolling whitespace-token
    counter in ``_stream_chat_response`` misses:

    - ``"BarleyBarleyBarley..."`` (no whitespace separator) — the entire
      suffix collapses to a single ``str.split()`` token whose count
      never increments. Real qwen3.5-4b regression surfaced in the
      0.6.28 onboarding test.
    - Long-cycle phrase loops, e.g. a ~280-char clause that repeats
      verbatim until ``max_tokens``. Surfaced when asked "describe the
      entire history of the Roman Empire in one long unbroken sentence".

    Implementation: compute the KMP failure function over the trailing
    window. The smallest period of the *entire* window is
    ``len(s) - fail[-1]``; a short period (≤``max_period``) means the
    window is dominated by that repetition starting from offset 0.

    Note: KMP itself does NOT detect periods that begin mid-window
    (rotated patterns). Mid-window degeneracy gets caught because this
    helper is invoked after every streaming chunk — once the model has
    been looping long enough to fill the window, the rolling 600-char
    suffix aligns with the pattern and the smallest-period check fires.
    A pure end-of-stream check would miss rotated cases.

    Cost is ``O(window)`` time and memory per call regardless of
    pattern length (the failure-function array is allocated each
    invocation) — much cheaper than the prior
    ``O(window * pattern_max_len)`` anchored scan, and cheap enough
    to run on every streaming chunk.

    The defaults (window=600, max_period=300) leave room for legitimate
    repetitive content like ``[0, 0, 0, ...]`` lists shorter than the
    window. *Long* lists of truly identical values the user explicitly
    asked for will get cut — a user hitting that false positive can
    ``/reset`` and rephrase. The cost of NOT cutting genuine model
    degeneracy (2000+ tokens of garbage) is far higher.
    """
    if len(text) < window:
        return False
    tail = text[-window:]
    n = len(tail)
    # KMP failure function: ``fail[i]`` = longest proper prefix of
    # ``tail[: i + 1]`` that is also a suffix.
    fail = [0] * n
    for i in range(1, n):
        j = fail[i - 1]
        while j > 0 and tail[i] != tail[j]:
            j = fail[j - 1]
        if tail[i] == tail[j]:
            j += 1
        fail[i] = j
    # Smallest period of ``tail``. Always >= 1 (fail[-1] <= n-1, since
    # ``fail`` is the longest *proper* prefix-suffix). ``period == n``
    # means no nontrivial period — the entire window is its own only
    # period and content is aperiodic. Defaults guarantee
    # ``max_period < window`` so this case never trips, but a caller
    # with ``max_period >= window`` would otherwise see aperiodic
    # strings flagged. Explicit ``period < n`` guard locks the contract.
    period = n - fail[-1]
    return period < n and period <= max_period


def _stream_chat_response(
    base_url: str,
    payload: dict,
    timeout_s: int,
    metrics: dict | None = None,
) -> str:
    """POST /v1/chat/completions with stream=True and print tokens as they
    arrive. Returns the full assistant content (concatenated content deltas).

    Reasoning-content deltas (Qwen3, DeepSeek-R1, etc.) are streamed to stdout
    in dim ANSI so the user sees thinking, but excluded from the returned
    string — chat history stores only the final answer, matching the
    OpenAI-compat split between ``content`` and ``reasoning_content``.

    Plain streaming: tokens land directly in the user's terminal as they
    arrive. We deliberately do NOT use ``rich.Live`` + ``Markdown`` here:
    Live re-renders the panel on every refresh and, when the console's
    cursor-overwrite path is unreliable (recordings, some terminal
    multiplexers), each refresh appends rather than overwrites — turning
    a 200-token response into a wall of repeated text. Live markdown
    rendering deserves a separate, more careful effort with explicit
    fallback detection; for now correctness wins over formatting.
    """
    import json

    import requests

    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    RESET = "\x1b[0m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    in_reasoning = False
    full = ""

    # ----- Streaming markdown colorer ------------------------------------
    # Body text streams in the terminal's default color (Claude-Code-style
    # — accents only on chrome). Inline coloring handles the markers users
    # see most often: ``\`code\``` (cyan), ``\`\`\`fence\`\`\``` (dim cyan
    # block), ``**bold**`` (ANSI bold), and ATX headers (``#`` … ``####``)
    # at line start. Lists / italic stay raw so the parser stays small.
    HEADING_STYLE = {
        1: BOLD + CYAN,  # `# h1`     — most prominent
        2: BOLD + MAGENTA,  # `## h2`    — secondary
        3: BOLD,  # `### h3`   — bold only
        4: CYAN,  # `#### h4`  — cyan
        5: MAGENTA,  # `##### h5` — magenta
        6: DIM,  # `###### h6`— dim
    }
    _state = {
        "in_fence": False,  # inside a ``` block
        "in_inline_code": False,  # inside a `code` span
        "in_bold": False,  # inside **bold**
        "in_heading": False,  # inside an ATX heading line
        "at_line_start": True,  # cursor is at start of a logical line
        "pending": "",  # buffered chars awaiting lookahead
    }

    def _emit_with_inline_md(piece: str) -> None:
        if not is_tty:
            sys.stdout.write(piece)
            sys.stdout.flush()
            return
        text = _state["pending"] + piece
        _state["pending"] = ""
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            # Newline closes any line-scoped span (heading) and resets the
            # line-start anchor so the next `#`/`*`/etc. is interpreted in
            # the right context.
            if c == "\n":
                if _state["in_heading"]:
                    out.append(RESET)
                    _state["in_heading"] = False
                out.append("\n")
                _state["at_line_start"] = True
                i += 1
                continue
            # ATX heading: `#`..`######` followed by space at line start.
            # We skip this inside fences (a `#` at line start there is
            # almost always a comment, not a heading).
            if _state["at_line_start"] and c == "#" and not _state["in_fence"]:
                # Count consecutive `#` (1..6).
                j = i
                while j < n and j - i < 6 and text[j] == "#":
                    j += 1
                # Need to see one more char after the hashes to decide
                # heading vs literal "###foo" — buffer if we don't have it.
                if j == n:
                    _state["pending"] = text[i:]
                    break
                hashes = j - i
                if 1 <= hashes <= 6 and text[j] == " ":
                    style = HEADING_STYLE.get(hashes, BOLD)
                    out.append(style)
                    out.append(text[i : j + 1])  # emit "## "
                    _state["in_heading"] = True
                    _state["at_line_start"] = False
                    i = j + 1
                    continue
                # Not a heading — fall through to literal emission below.
            if c == "`":
                # Need 2 chars of lookahead to disambiguate ``` vs `.
                if i + 2 >= n:
                    _state["pending"] = text[i:]
                    break
                if text[i : i + 3] == "```":
                    if _state["in_fence"]:
                        out.append("```" + RESET)
                        _state["in_fence"] = False
                    else:
                        out.append(DIM + CYAN + "```")
                        _state["in_fence"] = True
                    _state["at_line_start"] = False
                    i += 3
                    continue
                # Single backtick.
                if _state["in_fence"]:
                    out.append("`")
                elif _state["in_inline_code"]:
                    out.append("`" + RESET)
                    _state["in_inline_code"] = False
                else:
                    out.append(CYAN + "`")
                    _state["in_inline_code"] = True
                _state["at_line_start"] = False
                i += 1
                continue
            if c == "*" and not _state["in_fence"] and not _state["in_inline_code"]:
                if i + 1 >= n:
                    _state["pending"] = text[i:]
                    break
                if text[i : i + 2] == "**":
                    if _state["in_bold"]:
                        out.append("**" + RESET)
                        _state["in_bold"] = False
                    else:
                        out.append(BOLD + "**")
                        _state["in_bold"] = True
                    _state["at_line_start"] = False
                    i += 2
                    continue
            out.append(c)
            # Whitespace (other than newline, handled above) keeps the
            # line-start anchor true so leading-indent headings still
            # parse — e.g., a list item's child paragraph is rare here.
            if c not in " \t":
                _state["at_line_start"] = False
            i += 1
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _close_open_md_spans() -> None:
        if is_tty and (
            _state["in_fence"]
            or _state["in_inline_code"]
            or _state["in_bold"]
            or _state["in_heading"]
        ):
            sys.stdout.write(RESET)
            sys.stdout.flush()
        if _state["pending"]:
            sys.stdout.write(_state["pending"])
            sys.stdout.flush()
            _state["pending"] = ""

    # ----- Repetition guard ----------------------------------------------
    # Models occasionally degenerate into the same token repeated until
    # max_tokens — filling the screen with "Barley Barley Barley...".
    # Two complementary checks run per delta:
    #
    # 1. Whitespace-token-consecutive: the SAME whitespace-split token
    #    repeats ≥``REPEAT_LIMIT`` times in a row. O(1) rolling counter.
    #    Catches the common form ``"Barley Barley Barley..."``. Earlier
    #    guards used "≤2 unique in last 30" but fired on legit content
    #    like ``[0, 0, 0, ...]`` and markdown table separators, so the
    #    bar is now stricter.
    #
    # 2. Character-level pattern check (``_has_short_pattern_dominating_
    #    suffix``): the trailing window is dominated by a short repeating
    #    pattern. Catches the form ``"BarleyBarleyBarley..."`` (no
    #    whitespace separator), where ``piece.split()`` produces one
    #    giant token whose count never increments — this was a real
    #    qwen3.5-4b regression in 0.6.28 (issue surfaced post-release).
    REPEAT_LIMIT = 25
    repeat_last: str | None = None
    repeat_run = 0
    repetition_aborted = False

    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout_s,
    ) as resp:
        if resp.status_code != 200:
            # With stream=True the body may still be partial / mid-chunk when
            # the server closed the socket; read defensively so we surface a
            # useful HTTP code instead of a ChunkedEncodingError.
            try:
                body = resp.text[:500]
            except Exception:
                body = "(no body)"
            raise RuntimeError(f"HTTP {resp.status_code}: {body}")
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            # When the caller passes ``stream_options.include_usage``,
            # the server emits a final chunk with empty choices and a
            # populated ``usage`` block. Capture it for the speed line.
            usage = chunk.get("usage")
            if usage and metrics is not None:
                metrics["completion_tokens"] = usage.get("completion_tokens")
                metrics["prompt_tokens"] = usage.get("prompt_tokens")
            # The usage-only final chunk has ``choices=[]``; guard
            # against an IndexError there.
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            reasoning = delta.get("reasoning_content")
            piece = delta.get("content")
            if reasoning:
                if not in_reasoning:
                    if is_tty:
                        sys.stdout.write(f"{MAGENTA}[thinking]{RESET} {DIM}")
                    else:
                        sys.stdout.write("[thinking] ")
                    in_reasoning = True
                sys.stdout.write(reasoning)
                sys.stdout.flush()
            if piece:
                if in_reasoning:
                    sys.stdout.write(f"{RESET}\n  " if is_tty else "\n")
                    in_reasoning = False
                # Detect repetition BEFORE emitting. If a single coalesced
                # delta contains the cutoff inside it (server batched many
                # repeated tokens into one chunk), find the position and
                # only emit the prefix up to that token — otherwise the
                # user sees the full degenerate dump before the abort
                # message lands.
                #
                # Rolling counter: each new whitespace-separated token in
                # this delta either extends the current consecutive run
                # or resets it. Aborts only on a single token repeated
                # ``REPEAT_LIMIT`` times in a row, not on diverse-but-
                # repetitive content like ``[0, 0, 0, ...]`` or markdown
                # tables.
                cutoff_idx: int | None = None
                tokens = piece.split()
                for i, tok in enumerate(tokens):
                    if tok == repeat_last:
                        repeat_run += 1
                    else:
                        repeat_last = tok
                        repeat_run = 1
                    if repeat_run >= REPEAT_LIMIT:
                        repetition_aborted = True
                        cutoff_idx = i
                        break
                if cutoff_idx is not None:
                    # Find the byte position in ``piece`` corresponding to
                    # the start of the cutoff token, so we can emit only
                    # the prefix. ``str.split()`` collapses runs of
                    # whitespace, so we walk the original text token-by-
                    # token to recover the offset.
                    pos = 0
                    seen = 0
                    while seen < cutoff_idx and pos < len(piece):
                        # Skip leading whitespace.
                        while pos < len(piece) and piece[pos].isspace():
                            pos += 1
                        # Skip the token itself.
                        while pos < len(piece) and not piece[pos].isspace():
                            pos += 1
                        seen += 1
                    prefix = piece[:pos]
                    if prefix:
                        _emit_with_inline_md(prefix)
                        full += prefix
                else:
                    _emit_with_inline_md(piece)
                    full += piece
                # Char-level guard: catches no-whitespace degenerate
                # output like ``"BarleyBarleyBarley..."`` that the
                # whitespace-token counter misses (the entire chunk
                # collapses to one giant token whose consecutive count
                # never climbs). Cheap enough to run on every chunk.
                #
                # Trade-off: runs *after* the chunk is already emitted,
                # so the user sees one extra chunk of garbage before
                # the abort message lands. We accept this — slicing
                # mid-chunk would require re-running KMP per byte (or
                # binary search) on every delta, and degenerate chunks
                # are typically small (≤64 chars) since servers stream
                # token-by-token.
                if not repetition_aborted and _has_short_pattern_dominating_suffix(
                    full
                ):
                    repetition_aborted = True
                if repetition_aborted:
                    break
    _close_open_md_spans()
    if in_reasoning and is_tty:
        sys.stdout.write(RESET)
        sys.stdout.flush()
    if repetition_aborted:
        msg = (
            f"\n\n  {DIM}(response cut: model began repeating itself — "
            f"try /reset or a larger model){RESET}"
            if is_tty
            else "\n\n(response cut: repetition detected)"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
    return full


def chat_command(args):
    """Interactive REPL chat with a model.

    Spawns a local `serve` on an ephemeral port (or connects to an existing
    server via --base-url / --port), then loops stdin → /v1/chat/completions
    (streaming) → stdout. Maintains multi-turn history; `/reset` clears it.
    Exits cleanly on Ctrl-D, Ctrl-C, or `exit` / `quit`.
    """
    import atexit
    import signal
    import subprocess
    import tempfile

    base_url: str
    proc = None
    log_path: str | None = None
    # Tracks every spawned server (initial + every /model candidate) so
    # the SIGTERM/atexit cleanup tears down in-flight candidates too —
    # not just the bound ``proc``. A SIGTERM landing while a /model
    # swap is mid-spawn would otherwise orphan the candidate server.
    _active_procs: list[subprocess.Popen] = []

    # TTY-gated ANSI palette for the chat UI. NO_COLOR is honoured.
    _is_tty = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    BOLD = "\x1b[1m" if _is_tty else ""
    DIM = "\x1b[2m" if _is_tty else ""
    GREEN = "\x1b[32m" if _is_tty else ""
    CYAN = "\x1b[36m" if _is_tty else ""
    YELLOW = "\x1b[33m" if _is_tty else ""
    RED = "\x1b[31m" if _is_tty else ""
    RESET = "\x1b[0m" if _is_tty else ""

    def _teardown_proc(p) -> None:
        """Terminate a spawned chat server and free its log file.

        Used by `_cleanup` (process exit) and `_switch_model` (mid-
        session swap). Idempotent — safe to call when the proc has
        already exited or never existed. Also reaps the killed child
        with wait(timeout=1) so repeated /model swaps don't leave
        zombies until the parent exits.
        """
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        p.kill()
                        # Reap the SIGKILL'd child — without this,
                        # repeated /model swaps stack zombie entries.
                        try:
                            p.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                    except (ProcessLookupError, OSError):
                        pass
                except (ProcessLookupError, OSError):
                    pass
        finally:
            # Drop from the tracked set so a subsequent _cleanup walk
            # doesn't double-tear it down.
            try:
                _active_procs.remove(p)
            except ValueError:
                pass
            # Close the log handle and unlink the tempfile so /model
            # swaps don't leak FDs and tempfiles. Both attributes set
            # by _spawn_chat_server.
            fh = getattr(p, "_rapid_mlx_log", None)
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
            lp = getattr(p, "_rapid_mlx_log_path", None)
            if lp:
                try:
                    os.unlink(lp)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _cleanup():
        # Walk every tracked proc — covers the active server and any
        # in-flight /model candidate. Iterate over a snapshot since
        # _teardown_proc mutates _active_procs.
        for p in list(_active_procs):
            _teardown_proc(p)

    # Install SIGTERM handler + atexit BEFORE any spawn. Otherwise a
    # SIGTERM landing in the window between `Popen()` and `signal.signal`
    # uses Python's default handler (calls `_exit`, skips atexit) and
    # orphans the spawned server. SIGINT is *deliberately* left on the
    # default handler so Ctrl-C unblocks ``input()`` via the natural
    # KeyboardInterrupt path, the REPL loop's ``except
    # KeyboardInterrupt: break`` fires, and atexit runs ``_cleanup``.
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(143)))
    except (ValueError, OSError):
        pass
    atexit.register(_cleanup)

    if args.base_url:
        base_url = args.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
    elif args.port is not None:
        base_url = f"http://127.0.0.1:{args.port}"
    else:
        # Pre-download in the foreground so the HF tqdm progress bar lands
        # in the user's terminal. Otherwise the serve subprocess swallows
        # the bar into the log file and `rapid-mlx chat` looks frozen for
        # several minutes on first run with a fresh model.
        _ensure_model_downloaded(args.model)

        log_path = tempfile.NamedTemporaryFile(
            prefix="rapid-mlx-chat-", suffix=".log", delete=False
        ).name
        print(f"\n  Starting server {DIM}(log: {log_path}){RESET} ...")
        # If main() resolved an alias, expose the alias as the API model name
        # so the chat request body matches what the user typed.
        original = getattr(args, "_original_alias", None)
        proc, base_url = _spawn_chat_server(
            args.model,
            log_path,
            served_name=original,
            register_in=_active_procs,
        )

        try:
            _wait_for_chat_server(base_url, proc, timeout_s=args.ready_timeout)
        except (RuntimeError, TimeoutError) as e:
            print(f"\n  {RED}Failed to start server:{RESET} {e}")
            sys.exit(1)
        print(f"  {GREEN}✓ Ready.{RESET}\n")

    from vllm_mlx._version_check import print_staleness_warning_if_any

    print_staleness_warning_if_any()

    print(
        f"  {BOLD}Chat{RESET} — "
        f"{DIM}type {RESET}{BOLD}/help{RESET}{DIM} for commands, "
        f"Ctrl-D to exit.{RESET}"
    )
    print(
        f"  {DIM}For a Claude Code-like TUI: `rapid-mlx agents codex --setup`, "
        f"then run `codex` in any project.{RESET}\n"
    )

    served_name = getattr(args, "_original_alias", args.model)
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    # The rapid-mlx server's ChatCompletionRequest exposes a top-level
    # ``enable_thinking`` field — ``chat_template_kwargs`` is not a recognized
    # request field and would be silently dropped.
    #
    # Default thinking OFF in the REPL. Reasoning models (Qwen3.5/3.6, etc.)
    # otherwise emit raw chain-of-thought to stdout AND, on the default
    # qwen3.5-4b model, degenerate into infinite repetition until max-tokens
    # truncates the response — producing zero usable output for a brand-new
    # user. ``--think`` opts back in for users who explicitly want to see
    # reasoning traces; ``--no-think`` is preserved as the legacy form.
    extra: dict = {}
    if not args.think:
        extra["enable_thinking"] = False

    import time

    import requests

    # Importing ``readline`` upgrades the built-in ``input()`` so that
    # the arrow keys recall earlier prompts (and Ctrl-A/E/U/R work).
    # The module is stdlib on macOS/Linux; on Windows it doesn't exist
    # and we fall back to plain input(). When readline IS available we
    # need to wrap the colored prompt's ANSI escapes in \001/\002 so
    # readline's column counter doesn't include the invisible bytes —
    # otherwise long history entries wrap incorrectly and Ctrl-A jumps
    # to the wrong column (especially on libedit-backed Apple system
    # python). The wrappers are no-op on a terminal, so it's safe to
    # always emit them when readline is loaded.
    have_readline = False
    try:
        import readline  # noqa: F401 — side-effect import

        have_readline = True
    except ImportError:
        pass

    def _wrap_invisible(esc: str) -> str:
        if have_readline and esc:
            return "\001" + esc + "\002"
        return esc

    if _is_tty:
        prompt = _wrap_invisible(BOLD + CYAN) + ">" + _wrap_invisible(RESET) + " "
        cont_prompt = _wrap_invisible(DIM) + "…" + _wrap_invisible(RESET) + " "
    else:
        prompt = "> "
        cont_prompt = "… "

    def _print_help():
        print(
            f"\n  {BOLD}Slash commands{RESET}\n"
            f"    {BOLD}/help{RESET}              show this help\n"
            f"    {BOLD}/reset{RESET}, {BOLD}/clear{RESET}     clear conversation history\n"
            f"    {BOLD}/model <alias>{RESET}     switch model "
            f"{DIM}(restarts the server, resets history){RESET}\n"
            f"    {BOLD}/save <path>{RESET}       save conversation to a markdown file\n"
            f"    {BOLD}/exit{RESET}, {BOLD}/quit{RESET}       exit chat\n"
            f"\n  {BOLD}Multi-line input{RESET}\n"
            f'    type {BOLD}"""{RESET} on its own line to start, again to end '
            f"{DIM}(paste code blocks){RESET}\n"
            f"\n  {BOLD}Keys{RESET}\n"
            f"    {BOLD}Ctrl-C{RESET}             cancel the current response, "
            f"or exit at empty prompt\n"
            f"    {BOLD}Ctrl-D{RESET}             exit\n"
        )

    def _save_conversation(path_arg: str):
        # Refuse early on an empty conversation — otherwise we create a
        # near-empty file then lock the user out of the same path on
        # the next try (since exclusive-mode open refuses overwrite).
        non_system = [m for m in messages if m.get("role") != "system"]
        if not non_system:
            print(
                f"  {YELLOW}Nothing to save yet.{RESET} "
                f"{DIM}(send a chat turn first){RESET}\n"
            )
            return
        path = os.path.expanduser(path_arg)
        # Auto-create parent directories; otherwise users see a confusing
        # "No such file or directory" for /save logs/2026-05/convo.md.
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:
                print(f"  {RED}Save failed:{RESET} cannot create {parent}: {exc}\n")
                return
        try:
            # Mode "x" (O_CREAT | O_EXCL) is atomic — refuses if the path
            # already exists, with no TOCTOU window between exists() and
            # open() that an exists()-then-open("w") check has. Also
            # naturally rejects existing symlinks pointing elsewhere.
            with open(path, "x", encoding="utf-8") as f:
                f.write(f"# rapid-mlx chat — {served_name}\n\n")
                for m in messages:
                    if m["role"] == "system":
                        continue
                    f.write(f"## {m['role'].capitalize()}\n\n{m['content']}\n\n")
            print(f"  {GREEN}✓{RESET} Saved {len(messages)} messages to {path}\n")
        except FileExistsError:
            print(
                f"  {YELLOW}{path} already exists.{RESET} "
                f"{DIM}(/save won't overwrite — pick a different path){RESET}\n"
            )
        except IsADirectoryError:
            print(
                f"  {RED}Save failed:{RESET} {path} is a directory — "
                f"{DIM}give a file path, not a directory{RESET}\n"
            )
        except OSError as exc:
            print(f"  {RED}Save failed:{RESET} {exc}\n")

    def _read_multiline() -> str:
        lines: list[str] = []
        while True:
            try:
                more = input(cont_prompt)
            except (EOFError, KeyboardInterrupt):
                # Tell the user how many lines they're losing — silent
                # discard on Ctrl-C/Ctrl-D mid-paste is hostile.
                if lines:
                    print(
                        f"\n  {YELLOW}(multi-line cancelled — "
                        f"{len(lines)} line{'' if len(lines) == 1 else 's'} "
                        f"discarded){RESET}\n"
                    )
                else:
                    print(f"\n  {YELLOW}(multi-line cancelled){RESET}\n")
                return ""
            if more.rstrip() == '"""':
                # Preserve leading/trailing whitespace verbatim — the
                # heredoc is meant for code paste, where stripping
                # indentation actively corrupts the input.
                return "\n".join(lines)
            lines.append(more)

    def _switch_model(new_alias: str) -> None:
        """Hot-swap the spawned chat server to a new model alias.

        Order matters: validate + pre-download the new model BEFORE
        terminating the old one. If anything fails (bogus alias, disk
        gate, network), the old server stays running and the REPL is
        usable. Only when the new model is on-disk and the new server is
        spawn-ready do we tear down the old proc and rebind.
        """
        nonlocal proc, base_url, log_path, served_name, messages
        if proc is None:
            print(
                f"  {YELLOW}/model is only available when chat spawns its "
                f"own server (not with --base-url / --port).{RESET}\n"
            )
            return
        from vllm_mlx.model_aliases import resolve_model

        resolved = resolve_model(new_alias) or new_alias
        print(f"  {DIM}Preparing {new_alias} → {resolved} ...{RESET}")

        # 1. Pre-download the new model (this also runs the disk-space
        #    gate). The current server keeps running while we do this so
        #    a download failure leaves the user where they were.
        try:
            _ensure_model_downloaded(resolved)
        except SystemExit:
            # Disk gate aborted via sys.exit(1); old server is untouched.
            print(
                f"  {RED}Model switch aborted{RESET} "
                f"{DIM}(disk gate); previous server still running.{RESET}\n"
            )
            return
        except RuntimeError as exc:
            # Definitive 404 from HF; old server stays.
            print(
                f"  {RED}Model switch aborted:{RESET} {exc}  "
                f"{DIM}(previous server still running){RESET}\n"
            )
            return

        # 2. Allocate a new log file and spawn the new server. We don't
        #    tear down the old one yet; we want a working candidate
        #    before we commit.
        new_log_path = tempfile.NamedTemporaryFile(
            prefix="rapid-mlx-chat-", suffix=".log", delete=False
        ).name
        print(f"  Starting server {DIM}(log: {new_log_path}){RESET} ...")
        # ``register_in=_active_procs`` makes the candidate visible to
        # ``_cleanup`` *inside* ``_spawn_chat_server`` — before the
        # readiness wait, before any further Python statement runs in
        # this scope. A SIGTERM/Ctrl-C during the (possibly multi-second)
        # load tears the child down via the cleanup walk.
        new_proc, new_base_url = _spawn_chat_server(
            resolved,
            new_log_path,
            served_name=new_alias,
            register_in=_active_procs,
        )
        try:
            _wait_for_chat_server(new_base_url, new_proc, timeout_s=args.ready_timeout)
        except (RuntimeError, TimeoutError) as exc:
            print(
                f"  {RED}Failed to start new server:{RESET} {exc}  "
                f"{DIM}(previous server still running){RESET}\n"
            )
            # Roll back: tear down the half-spawned new proc + free its
            # log file. The old proc/base_url/log_path stay bound.
            _teardown_proc(new_proc)
            return

        # 3. New server is healthy — commit. Rebind ``proc`` BEFORE
        #    tearing down the old one so a SIGTERM during teardown
        #    walks the new (still-running) proc, not just a freshly
        #    killed corpse.
        old_proc = proc
        proc = new_proc
        base_url = new_base_url
        log_path = new_log_path
        served_name = new_alias
        messages = [{"role": "system", "content": args.system}] if args.system else []
        _teardown_proc(old_proc)
        print(
            f"  {GREEN}✓ Switched to {new_alias}.{RESET} "
            f"{DIM}(history cleared){RESET}\n"
        )

    while True:
        try:
            line = input(prompt).rstrip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        # Heredoc-pasted content must NEVER be dispatched as a slash
        # command — a markdown doc whose first line starts with `/path`
        # or whose content includes `/save` would otherwise be silently
        # eaten by the slash dispatcher. Track the source so we know.
        is_heredoc = False
        if line == '"""':
            line = _read_multiline()
            if not line:
                continue
            is_heredoc = True
        if not is_heredoc:
            # Parse the leading word as the command and dispatch on
            # *exact* match. ``startswith("/save")`` would otherwise treat
            # ``/savefoo`` as ``/save`` (with arg ``foo``), silently
            # writing a file from a typo. Same for ``/modelfoo``.
            # ``str.split(maxsplit=1)`` (no separator arg) splits on any
            # whitespace, so ``/save\tpath.md`` works the same as
            # ``/save path.md``.
            parts = line.split(maxsplit=1)
            cmd = parts[0] if parts else ""
            rest = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("exit", "quit", "/exit", "/quit"):
                break
            if cmd in ("/help", "/?"):
                _print_help()
                continue
            if cmd in ("/reset", "/clear"):
                messages = (
                    [{"role": "system", "content": args.system}] if args.system else []
                )
                print(f"  {DIM}(history cleared){RESET}\n")
                continue
            if cmd == "/save":
                if not rest:
                    print(f"  {YELLOW}Usage: /save <path>{RESET}\n")
                else:
                    _save_conversation(rest)
                continue
            if cmd == "/model":
                if not rest:
                    print(
                        f"  {YELLOW}Usage: /model <alias>{RESET}  "
                        f"{DIM}(see `rapid-mlx models`){RESET}\n"
                    )
                else:
                    _switch_model(rest)
                continue
            if cmd.startswith("/"):
                print(
                    f"  {YELLOW}Unknown command: {cmd}{RESET}  "
                    f"{DIM}(type /help){RESET}\n"
                )
                continue

        messages.append({"role": "user", "content": line})
        payload = {
            "model": served_name,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **extra,
        }
        # Claude-Code-style turn marker: a colored bullet introduces the
        # assistant's response so the user can visually scan turn
        # boundaries when scrolling back through long conversations.
        sys.stdout.write(f"\n  {CYAN}●{RESET} ")
        sys.stdout.flush()
        metrics: dict = {}
        start_t = time.monotonic()
        try:
            assistant = _stream_chat_response(
                base_url,
                payload,
                timeout_s=args.response_timeout,
                metrics=metrics,
            )
        except KeyboardInterrupt:
            print(f"\n  {YELLOW}(response interrupted){RESET}\n")
            messages.pop()
            continue
        except RuntimeError as e:
            print(f"\n  {RED}{e}{RESET}\n")
            messages.pop()
            continue
        except requests.RequestException as e:
            # Connection refused, timeout, dropped midstream — keep the REPL
            # alive and roll back the failed user turn so the next request
            # doesn't carry a dangling user role with no assistant reply.
            print(f"\n  {RED}Request failed:{RESET} {e}\n")
            messages.pop()
            continue
        elapsed = time.monotonic() - start_t
        # Speed line: prefer server-reported usage, fall back to a rough
        # 4-chars-per-token estimate when the server doesn't ship usage
        # in the stream.
        tokens = metrics.get("completion_tokens")
        if not tokens:
            tokens = max(1, len(assistant) // 4)
            tokens_label = f"~{tokens}"
        else:
            tokens_label = str(tokens)
        if assistant and elapsed > 0:
            tps = tokens / elapsed
            print(
                f"\n  {DIM}{tokens_label} tok · {elapsed:.1f}s · "
                f"{tps:.0f} tok/s{RESET}\n"
            )
        else:
            print()
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        else:
            messages.pop()


def info_command(args):
    """Print the per-model profile for a model name or alias.

    Stage 1 (regex match) only — does NOT load the model, so this is fast
    and works without weights. Stage 2 (ArraysCache probe) is skipped.
    """
    from vllm_mlx.model_aliases import resolve_model, resolve_profile
    from vllm_mlx.model_auto_config import (
        detect_model_config,
        format_profile_table,
    )

    # ``main()`` (cli.py:~3400) pre-resolves ``args.model`` from alias →
    # HF path before dispatch, stashing the user-typed alias on
    # ``args._original_alias``. Pull from that first so DFlash
    # eligibility (alias-keyed) and the start-command hint render with
    # the alias the user actually typed, not the resolved HF repo.
    original_alias = getattr(args, "_original_alias", None) or args.model
    name = args.model
    resolved = (
        resolve_model(name) if not getattr(args, "_original_alias", None) else None
    )
    if resolved and resolved != name:
        print(f"  alias: {name} → {resolved}")
        name = resolved

    cfg = detect_model_config(name)
    print()
    print(format_profile_table(name, cfg))
    print()

    # DFlash eligibility — render the report so users can see which
    # gates pass/fail without consulting the docs. Skipped for unknown
    # models since AliasProfile is alias-keyed.
    profile = resolve_profile(original_alias)
    if profile is not None:
        _print_dflash_status(original_alias, profile)

    if cfg is None:
        print("  No pattern matched — runtime probe will run when the model loads.")
        print()


def _print_dflash_status(alias: str, profile) -> None:
    """Render a 3-row DFlash status block for ``rapid-mlx info <alias>``.

    Shows each gate (declared support / not MoE / not 4-bit / drafter
    present) so a user who tried ``--enable-dflash`` and got a vague
    error can see exactly which gate they're tripping.
    """
    from vllm_mlx.speculative.dflash.eligibility import (
        _looks_like_4bit,
        have_runtime,
        report,
    )

    r = report(profile, alias=alias)
    inner = 60
    sep = "─" * inner

    def _row(text: str) -> str:
        return f"│ {text:<{inner}} │"

    def _yes(ok: bool, msg_ok: str, msg_no: str) -> str:
        return ("✓ " + msg_ok) if ok else ("✗ " + msg_no)

    rows = [
        (
            "Declared support",
            _yes(profile.supports_dflash, "yes (supports_dflash=true)", "no"),
        ),
        ("Not MoE", _yes(not profile.is_moe, "yes (dense)", "no (MoE)")),
        (
            "Precision ≥8-bit",
            _yes(
                not _looks_like_4bit(profile.hf_path),
                "yes",
                "no (4-bit/mxfp4/nvfp4)",
            ),
        ),
        (
            "Drafter declared",
            _yes(
                bool(profile.dflash_draft_model),
                profile.dflash_draft_model or "yes",
                "no (dflash_draft_model unset)",
            ),
        ),
        (
            "mlx-vlm 0.5.0+",
            _yes(have_runtime(), "installed", "missing (need rapid-mlx[dflash])"),
        ),
    ]

    eligible = not r.reasons and have_runtime()
    summary = "✓ eligible" if eligible else "✗ ineligible"

    top = "┌" + "─" * (inner + 2) + "┐"
    bot = "└" + "─" * (inner + 2) + "┘"

    body = [top, _row(f"DFlash eligibility: {summary}"), _row(sep)]
    for k, v in rows:
        body.append(_row(f"{k:<18}: {v}"))
    body.append(bot)
    print("\n".join(body))
    print()
    if eligible:
        print(f"  Start with: rapid-mlx serve {alias} --enable-dflash")
        print()


def agents_command(args):
    """List, configure, and test agent integrations."""
    from vllm_mlx.agents import get_profile, list_profiles
    from vllm_mlx.agents.adapter import get_setup_instructions, setup_agent_config

    agent_name = args.agent_name
    base_url = args.base_url

    # No agent specified → list all profiles
    if not agent_name:
        profiles = list_profiles()
        print()
        print("  Supported AI Agents")
        print("  " + "─" * 56)
        for p in profiles:
            fc = "FC" if p.needs_function_calling else "  "
            stars = f"{p.stars // 1000}K" if p.stars and p.stars >= 1000 else ""
            if p.recommended_models:
                shown = p.recommended_models[:3]
                models = ", ".join(shown)
                if len(p.recommended_models) > 3:
                    models += f" +{len(p.recommended_models) - 3}"
            else:
                models = ""
            print(f"  {p.name:<15} {p.display_name:<20} {stars:>5}  [{fc}]  {models}")
        print()
        print(f"  {len(profiles)} agents supported")
        print("  Usage: rapid-mlx agents <name>          Show setup guide")
        print("         rapid-mlx agents <name> --setup   Auto-configure")
        print("         rapid-mlx agents <name> --test    Run integration tests")
        print()
        return

    # Get profile
    profile = get_profile(agent_name)
    if not profile:
        print(f"  Unknown agent: {agent_name}")
        print("  Run 'rapid-mlx agents' to see available agents.")
        sys.exit(1)

    # --test: run integration tests
    if args.test:
        from vllm_mlx.agents.testing import AgentTestRunner

        model_id = args.model or None
        runner = AgentTestRunner(
            profile,
            base_url=base_url,
            model_id=model_id,
            agent_version=args.agent_version,
        )
        if not runner._server_available():
            print(f"\n  Server not running at {base_url}")
            print("  Start it first: rapid-mlx serve <model>")
            sys.exit(1)

        report = runner.run()
        success = report.print_summary()
        sys.exit(0 if success else 1)

    # --setup: auto-configure agent
    if args.setup:
        # Detect model from running server
        model_id = args.model or "default"
        if model_id == "default":
            try:
                import httpx

                resp = httpx.get(f"{base_url}/models", timeout=3)
                model_id = resp.json()["data"][0]["id"]
            except Exception:
                pass

        summary = setup_agent_config(
            profile, base_url, model_id, agent_version=args.agent_version
        )
        print(f"\n  {profile.display_name} configured!")
        print(f"  {summary}")
        print()
        return

    # Default: show setup instructions
    # Pass "default" to trigger auto-detection of running model
    model_id = args.model or "default"
    instructions = get_setup_instructions(
        profile, base_url, model_id, agent_version=args.agent_version
    )
    print()
    print(instructions)
    print()


def upgrade_command(args):
    """Detect install method and (optionally) run the right upgrade command."""
    import subprocess

    from vllm_mlx._version_check import (
        _installed_version,
        _parse_version,
        detect_install_method,
        get_latest_version,
    )

    current = _installed_version() or "dev"
    print()
    print(f"  Current:  rapid-mlx {current}")

    latest = get_latest_version(force_refresh=True)
    if latest is None:
        print("  Latest:   (could not reach GitHub — check your network)\n")
        sys.exit(1)
    print(f"  Latest:   rapid-mlx {latest}")

    cur = _parse_version(current)
    lat = _parse_version(latest)
    if cur is not None and lat is not None and cur >= lat:
        print("\n  ✓ Already up to date.\n")
        return

    info = detect_install_method()
    print(f"  Install:  {info.method} ({info.binary_path or 'unknown path'})")
    print(f"  Command:  {info.upgrade_command}")
    print()

    if info.method == "unknown":
        print(
            "  Could not auto-detect install method — run the command above manually.\n"
        )
        return

    if args.yes:
        confirmed = True
    else:
        try:
            answer = input("  Run now? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        confirmed = answer in {"y", "yes"}

    if not confirmed:
        print("  Skipped — run the command above when ready.\n")
        return

    print()
    try:
        # Use argv form (shell=False) so paths with spaces in
        # ``sys.executable`` (or any other argv entry) can't be reinterpreted
        # as shell separators. install.sh's pipe is wrapped as ``bash -c``
        # in upgrade_argv, so we still get the pipe semantics it needs.
        result = subprocess.run(info.upgrade_argv, check=False)
    except KeyboardInterrupt:
        print("\n  Interrupted.\n")
        sys.exit(130)
    print()
    sys.exit(result.returncode)


def telemetry_command(args) -> None:
    """Manage anonymous usage telemetry — see Issue #236.

    Five actions: ``status`` / ``enable`` / ``disable`` / ``preview`` /
    ``reset``. Defaults to ``status`` when no action given so users can
    type ``rapid-mlx telemetry`` and immediately see what's set up.
    """
    # Imports kept inside the function so the telemetry package is only
    # loaded when actually needed — keeps `--help` and unrelated
    # subcommands cheap.
    import json

    from vllm_mlx import __version__ as rapid_mlx_version
    from vllm_mlx.telemetry import (
        consent_source,
        get_consent_state,
        get_or_create_client_id,
        is_enabled,
        record_consent,
        reset_state,
    )
    from vllm_mlx.telemetry.schema import sample_preview_payload
    from vllm_mlx.telemetry.state import client_id_path, consent_path

    action = getattr(args, "telemetry_action", None) or "status"
    cli_no = getattr(args, "no_telemetry", False)

    if action == "status":
        state = get_consent_state()
        print()
        print(
            f"  Telemetry: {'ENABLED' if is_enabled(cli_no_telemetry=cli_no) else 'disabled'}"
        )
        print(f"  Source:    {consent_source(cli_no_telemetry=cli_no)}")
        if state is not None:
            print(
                f"  Consent:   {state.consent} (recorded {state.prompted_at}, "
                f"by rapid-mlx {state.prompted_version})"
            )
        else:
            print("  Consent:   never prompted")
        print(f"  Files:     {consent_path()}")
        print(f"             {client_id_path()}")
        print()
        print("  Subcommands:  enable | disable | preview | reset")
        print()
        return

    if action == "enable":
        record_consent(True, rapid_mlx_version=rapid_mlx_version)
        # Generate the client_id eagerly so `preview` immediately after
        # has a real id to show.
        get_or_create_client_id()
        print()
        print("  Telemetry: ENABLED. Thanks for helping us prioritise.")
        print("  Disable anytime with `rapid-mlx telemetry disable`.")
        print("  Preview what we'd send: `rapid-mlx telemetry preview`.")
        print()
        return

    if action == "disable":
        record_consent(False, rapid_mlx_version=rapid_mlx_version)
        print()
        print("  Telemetry: disabled. No data will be sent.")
        print("  Re-enable anytime with `rapid-mlx telemetry enable`.")
        print()
        return

    if action == "preview":
        cid = get_or_create_client_id()
        payload = sample_preview_payload(
            client_id=cid, rapid_mlx_version=rapid_mlx_version
        )
        print()
        print("  Sample payload (this is exactly the shape we send):")
        print()
        print(json.dumps(payload.to_dict(), indent=2))
        print()
        if not is_enabled(cli_no_telemetry=cli_no):
            print("  Telemetry is currently disabled — nothing is actually sent.")
            print()
        return

    if action == "reset":
        reset_state()
        print()
        print("  Removed consent + client-id files. Next interactive run re-prompts.")
        print()
        return

    # Unknown action — argparse choices=[] would have caught this earlier
    # in normal flow; defensive guard for future maintainers.
    print(f"  Unknown telemetry action: {action!r}")
    sys.exit(1)


def main():
    from importlib.metadata import version as pkg_version

    try:
        _version = pkg_version("rapid-mlx")
    except Exception:
        _version = "dev"

    parser = argparse.ArgumentParser(
        description="Rapid-MLX: AI inference for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rapid-mlx serve qwen3.5-9b --port 8000
  rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --port 8000
  rapid-mlx models
        """,
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"rapid-mlx {_version}"
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable anonymous usage telemetry for this run "
        "(equivalent to RAPID_MLX_TELEMETRY=0).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible server")
    serve_parser.add_argument("model", type=str, help="Model to serve")
    serve_parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not specified, the model argument is used.",
    )
    serve_parser.add_argument(
        "--force-disk-check",
        action="store_true",
        help=(
            "Skip the pre-flight disk-space check that aborts when the model "
            "is larger than free disk. Use only if you know the HF cache lives "
            "on a different filesystem (e.g. external drive via HF_HOME)."
        ),
    )
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind"
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument(
        "--log-level",
        type=_log_level_choice,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for Python logging and uvicorn (case-insensitive)",
    )
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
    )
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=256,
        help=(
            "Admission cap on in-flight requests (queued + running). When "
            "exceeded, new requests return HTTP 503 with Retry-After. "
            "Default 256; operators on memory-constrained devices may want "
            "to set this near ``--max-num-seqs`` to limit queue depth."
        ),
    )
    serve_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    serve_parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Completion batch size"
    )
    serve_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching for repeated prompts (default: enabled)",
    )
    serve_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    serve_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    serve_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    serve_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    serve_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    serve_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    serve_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # TurboQuant KV cache compression (V-only, experimental)
    serve_parser.add_argument(
        "--kv-cache-turboquant",
        action="store_true",
        help="Enable TurboQuant V-cache compression (3-4 bit, ~86%% prefix cache savings "
        "on dense models). K stays FP16. Experimental — mutually exclusive with "
        "--kv-cache-quantization.",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-bits",
        type=int,
        default=None,
        choices=[3, 4],
        help="Bit width for TurboQuant (default: auto-select by head_dim — "
        "3-bit for head_dim>=96, 4-bit for head_dim=64)",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-group-size",
        type=int,
        default=32,
        help="Group size for TurboQuant quantization (default: 32)",
    )
    serve_parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=throughput)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation (default: 32768)",
    )
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        default=True,
        help="Enable continuous batching (default: on).",
    )
    # Deprecated flags — accepted silently to avoid breaking user scripts
    serve_parser.add_argument(
        "--simple-engine",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    # DFlash — block-diffusion drafter speculative decoding (z-lab / mlx-vlm).
    # Currently single-user serial mode; runs a dedicated DFlash server that
    # bypasses BatchedEngine. Eligible aliases declare ``supports_dflash=true``
    # in aliases.json (dense, ≥8-bit, drafter available — qwen3.5-27b-8bit
    # is the only validated one today). PoC: 1.83–2.18× on Qwen3.5-27B-8bit.
    serve_parser.add_argument(
        "--enable-dflash",
        action="store_true",
        default=False,
        help="Enable DFlash speculative decoding (block-diffusion drafter, "
        "single-user serial mode). Requires a DFlash-eligible alias "
        "(see ``rapid-mlx info <alias>``). Loads the drafter from the "
        "alias's ``dflash_draft_model`` field. Install with "
        "``pip install 'rapid-mlx[dflash]'``.",
    )
    serve_parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-threshold",
        type=int,
        default=8192,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-keep-pct",
        type=float,
        default=0.3,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-draft-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of device memory for Metal allocation limit and emergency "
        "cache clear threshold (0.0-1.0, default: 0.90). Increase to 0.95 for "
        "large models (200GB+) that need more memory headroom.",
    )
    # Paged cache options (experimental)
    serve_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    serve_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    serve_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Chunked prefill
    serve_parser.add_argument(
        "--chunked-prefill-tokens",
        type=int,
        default=0,
        help="Max prefill tokens per scheduler step (0=disabled). "
        "Prevents starvation of active requests during long prefills.",
    )
    # MTP (Multi-Token Prediction)
    serve_parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=False,
        help="Enable MTP (Multi-Token Prediction) for models with built-in MTP heads. "
        "Uses cache snapshot/restore for speculative generation.",
    )
    serve_parser.add_argument(
        "--mtp-num-draft-tokens",
        type=int,
        default=1,
        help="Number of draft tokens per MTP step (default: 1)",
    )
    serve_parser.add_argument(
        "--mtp-optimistic",
        action="store_true",
        default=False,
        help="Skip MTP acceptance check for maximum speed. "
        "~5-10%% wrong tokens. Best for chat, not for code.",
    )
    # SuffixDecoding — drafter-free spec-decode using a suffix tree over
    # generated tokens. Big wins on agent/tool/JSON workloads (3-5x);
    # ~zero overhead on free-form chat. Pure-attention only.
    serve_parser.add_argument(
        "--suffix-decoding",
        action="store_true",
        default=False,
        help="Enable SuffixDecoding spec-decode (drafter-free, statistical). "
        "Speedup is workload-dependent: 3-5x on tool-call/JSON/code-edit, "
        "~1x on free-form chat. Auto-disabled on hybrid models "
        "(Qwen3.5/3.6, Granite4, Mamba/Jamba/RWKV).",
    )
    serve_parser.add_argument(
        "--suffix-max-draft",
        type=int,
        default=8,
        help="Max draft tokens per verify step (default: 8). "
        "Verify forward cost grows linearly with this.",
    )
    serve_parser.add_argument(
        "--suffix-max-suffix-len",
        type=int,
        default=4,
        help="Max k-gram length indexed for suffix matching (default: 4).",
    )
    serve_parser.add_argument(
        "--suffix-min-confidence",
        type=float,
        default=0.3,
        help="Vote confidence floor for draft truncation (default: 0.3). "
        "Lower → more optimistic drafts; higher → fewer but more reliable.",
    )
    serve_parser.add_argument(
        "--suffix-min-draft-len",
        type=int,
        default=2,
        help="Skip the verify forward when drafter returns fewer than "
        "this many tokens (default: 2). Protects free-form chat from "
        "verify overhead on weak 1-token drafts. Set to 1 to verify "
        "every draft (more aggressive; can regress chat).",
    )
    # Prefill step size
    serve_parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Chunk size for prompt prefill processing. Larger values use more memory "
        "but can improve prefill throughput. (default: 2048)",
    )
    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )
    # Security options
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    serve_parser.add_argument(
        "--cors-origins",
        type=str,
        nargs="+",
        default=None,
        metavar="ORIGIN",
        help=(
            "Allowed CORS origins (default: * for all origins). "
            "Example: --cors-origins http://localhost:3000 https://myapp.com"
        ),
    )
    serve_parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Default request timeout in seconds (default: 1800 = 30 min)",
    )
    # Tool calling options
    serve_parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Enable auto tool choice for supported models. Use --tool-call-parser to specify which parser to use.",
    )
    serve_parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        choices=[
            "auto",
            "mistral",
            "qwen",
            "qwen3_coder",
            "qwen3_coder_xml",
            "qwen3_xml",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
            "glm47",
            "minimax",
            "harmony",
            "gpt-oss",
            "gemma4",
        ],
        help=(
            "Select the tool call parser for the model. Options: "
            "auto (auto-detect), mistral, qwen/qwen3/qwen3_xml (reasoning models, "
            "<tool_call>JSON</tool_call> format), qwen3_coder/qwen3_coder_xml "
            "(Coder model, <function=NAME> XML format), llama, hermes, "
            "deepseek, kimi, granite, nemotron, xlam, functionary, glm47, minimax, "
            "harmony/gpt-oss, gemma4. "
            "Required for --enable-auto-tool-choice."
        ),
    )
    # Tool logits bias (jump-forward decoding for tool call structural tokens)
    serve_parser.add_argument(
        "--enable-tool-logits-bias",
        action="store_true",
        default=False,
        help="Bias logits toward structural tool call tokens for faster generation. "
        "Only active when --tool-call-parser is also set. Currently supports minimax.",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    serve_parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            "Extracts <think>...</think> tags into reasoning_content field. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    serve_parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help=(
            "Disable reasoning/thinking parser even if auto-detected. "
            "Thinking tokens will appear as regular content. "
            "Useful for faster responses when chain-of-thought is not needed."
        ),
    )
    serve_parser.add_argument(
        "--no-tool-call-parser",
        dest="no_tool_call_parser",
        action="store_true",
        default=False,
        help=(
            "Force-disable tool-call parser auto-detection from the alias "
            "profile. Escape hatch (SOP §10) when AliasProfile's auto-"
            "selected parser misfires for a specific deployment. Mutually "
            "exclusive with --tool-call-parser."
        ),
    )
    serve_parser.add_argument(
        "--no-reasoning-parser",
        dest="no_reasoning_parser",
        action="store_true",
        default=False,
        help=(
            "Force-disable reasoning parser auto-detection from the alias "
            "profile. Distinct from --no-thinking (which also suppresses "
            "the chain-of-thought prompt template) — this flag ONLY skips "
            "the auto-config step. Mutually exclusive with --reasoning-parser."
        ),
    )
    # SOP §10 profile-override escape hatches. Pair every binary
    # auto-routing field with both force-on and force-off CLI flags so
    # users always have an override path when the AliasProfile
    # auto-detection misfires. Registered in
    # tests/test_no_mllm_flag.py::test_auto_routing_flags_have_force_on_and_force_off_pair.
    serve_parser.add_argument(
        "--force-hybrid",
        dest="force_hybrid",
        action="store_true",
        default=False,
        help=(
            "Force-treat the model as a hybrid (linear-attention / Mamba) "
            "architecture even when AliasProfile says otherwise. Disables "
            "spec/suffix decode paths that are unsound on hybrids. "
            "Mutually exclusive with --no-hybrid."
        ),
    )
    serve_parser.add_argument(
        "--no-hybrid",
        dest="no_hybrid",
        action="store_true",
        default=False,
        help=(
            "Force-treat the model as non-hybrid (full attention) even when "
            "AliasProfile says it's hybrid. Use when the profile mis-labels "
            "your model and you want spec/suffix decode enabled. "
            "Mutually exclusive with --force-hybrid."
        ),
    )
    serve_parser.add_argument(
        "--force-spec-decode",
        dest="force_spec_decode",
        action="store_true",
        default=False,
        help=(
            "Force-enable speculative-decode eligibility even when "
            "AliasProfile says the model doesn't support it. Risky on "
            "hybrid models — use only when you've verified the profile "
            "is wrong. Mutually exclusive with --no-spec-decode."
        ),
    )
    serve_parser.add_argument(
        "--no-spec-decode",
        dest="no_spec_decode",
        action="store_true",
        default=False,
        help=(
            "Force-disable speculative-decode eligibility (suffix / MTP / "
            "DFlash) even when AliasProfile says the model supports it. "
            "Mutually exclusive with --force-spec-decode."
        ),
    )
    # GC control (Tier 0 optimization)
    serve_parser.add_argument(
        "--gc-control",
        action="store_true",
        default=True,
        help="Disable Python GC during generation to avoid latency spikes (default: enabled)",
    )
    serve_parser.add_argument(
        "--no-gc-control",
        action="store_true",
        help="Disable GC control (allow normal GC during generation)",
    )
    # Pinned prefix cache (Tier 0 optimization)
    serve_parser.add_argument(
        "--pin-system-prompt",
        action="store_true",
        default=False,
        help="Auto-pin system prompt in prefix cache to prevent eviction under memory pressure",
    )
    # Multimodal option
    serve_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force load model as multimodal (vision) even if name doesn't match auto-detection patterns",
    )
    serve_parser.add_argument(
        "--no-mllm",
        "--text-only",
        dest="no_mllm",
        action="store_true",
        help="Force load model as text-only LLM even when auto-detection would route it to the multimodal/VLM path. Escape hatch for incomplete vision-tower checkpoints (#393) and text-only forks of multimodal architectures whose config.json still declares vision_config.",
    )
    # Generation defaults
    serve_parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Override default temperature for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Override default top_p for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="Override default top_k for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-min-p",
        type=float,
        default=None,
        help="Override default min_p for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help="Override default repetition_penalty for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-presence-penalty",
        type=float,
        default=None,
        help="Override default presence_penalty for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-frequency-penalty",
        type=float,
        default=None,
        help="Override default frequency_penalty for all requests (default: use model default)",
    )
    # Cloud routing options
    serve_parser.add_argument(
        "--cloud-model",
        type=str,
        default=None,
        help="Cloud model string for litellm (e.g. 'anthropic/claude-sonnet-4-5-20250929'). "
        "When set, large-context requests are routed to the cloud provider.",
    )
    serve_parser.add_argument(
        "--cloud-threshold",
        type=int,
        default=20000,
        help="New token threshold to trigger cloud routing (default: 20000). "
        "Only requests with more new (uncached) tokens than this are routed.",
    )
    serve_parser.add_argument(
        "--cloud-api-base",
        type=str,
        default=None,
        help="Custom API base URL for cloud model (for OpenAI-compatible providers like Zhipu).",
    )
    serve_parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="API key for cloud model (overrides environment variable).",
    )
    # Embedding model option
    serve_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup (e.g. mlx-community/embeddinggemma-300m-6bit)",
    )
    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("model", type=str, help="Model to benchmark")
    bench_parser.add_argument(
        "--force-disk-check",
        action="store_true",
        help=(
            "Skip the pre-flight disk-space check that aborts when the model "
            "is larger than free disk. Use only if you know the HF cache lives "
            "on a different filesystem (e.g. external drive via HF_HOME)."
        ),
    )
    bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts"
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens per prompt"
    )
    bench_parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="Max concurrent sequences"
    )
    bench_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    bench_parser.add_argument(
        "--completion-batch-size", type=int, default=16, help="Completion batch size"
    )
    bench_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: enabled)",
    )
    bench_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    bench_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    bench_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    bench_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    bench_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    bench_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    bench_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # Paged cache options (experimental)
    bench_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    bench_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    bench_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )

    # Models command
    subparsers.add_parser("models", help="List available model aliases")

    # Version + help — utility commands that mirror the existing flags but
    # are scriptable as plain subcommands.
    subparsers.add_parser("version", help="Show version number")
    help_parser = subparsers.add_parser("help", help="Show help for a subcommand")
    help_parser.add_argument(
        "subcommand", nargs="?", help="Subcommand to show help for (omit for top-level)"
    )

    # Pull / rm / ps — Ollama-style cache and process management.
    pull_parser = subparsers.add_parser(
        "pull", help="Download a model to the HuggingFace cache (no server)"
    )
    pull_parser.add_argument(
        "model", help="Model alias (e.g. qwen3.5-4b) or HF repo (org/name)"
    )
    rm_parser = subparsers.add_parser(
        "rm", help="Remove a cached model from the HuggingFace cache"
    )
    rm_parser.add_argument(
        "model", help="Model alias (e.g. qwen3.5-4b) or HF repo (org/name)"
    )
    subparsers.add_parser("ps", help="List running rapid-mlx servers")

    # Upgrade — detect install method and run the right upgrade command
    upgrade_parser = subparsers.add_parser(
        "upgrade",
        help="Upgrade rapid-mlx to the latest version (brew / pip / install.sh)",
    )
    upgrade_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and run the upgrade immediately.",
    )

    # Chat — interactive REPL backed by a (spawned or existing) server
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive chat REPL with a model"
    )
    chat_parser.add_argument(
        "model",
        nargs="?",
        default="qwen3.5-4b",
        help="Model alias (e.g. qwen3.5-4b) or HF repo (org/name). "
        "Defaults to qwen3.5-4b when omitted.",
    )
    chat_parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt prepended to the conversation",
    )
    chat_parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable thinking/reasoning mode (default: off in chat REPL — "
        "reasoning models like Qwen3.5 otherwise leak raw chain-of-thought "
        "and can loop until max-tokens). Use --think to surface reasoning, "
        "--no-think is also accepted for back-compat.",
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per assistant response (default: 2048)",
    )
    chat_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    chat_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Connect to existing server on 127.0.0.1:<port> instead of spawning",
    )
    chat_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Connect to existing server URL (e.g. http://host:8000) "
        "instead of spawning. Overrides --port.",
    )
    chat_parser.add_argument(
        "--ready-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the spawned server to become ready (default: 600)",
    )
    chat_parser.add_argument(
        "--response-timeout",
        type=int,
        default=600,
        help="Seconds to wait for a single assistant response (default: 600)",
    )

    # Info command — show the per-model profile (parsers + capability gates)
    info_parser = subparsers.add_parser(
        "info",
        help="Show the per-model profile for a model name or alias",
    )
    info_parser.add_argument(
        "model",
        help="Model alias (e.g. qwen3.5-4b) or HF repo (e.g. mlx-community/SmolLM3-3B-4bit)",
    )

    # Agents command
    agents_parser = subparsers.add_parser(
        "agents", help="List, configure, and test agent integrations"
    )
    agents_parser.add_argument(
        "agent_name",
        nargs="?",
        default=None,
        help="Agent name (e.g. hermes, goose, aider). Omit to list all.",
    )
    agents_parser.add_argument(
        "--setup",
        action="store_true",
        help="Auto-configure the agent to point at this server",
    )
    agents_parser.add_argument(
        "--test",
        action="store_true",
        help="Run integration tests for this agent",
    )
    agents_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: auto-detect from running server)",
    )
    agents_parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Rapid-MLX server URL (default: http://localhost:8000/v1)",
    )
    agents_parser.add_argument(
        "--agent-version",
        type=str,
        default=None,
        help="Agent version for version-specific config (e.g. 0.8.5)",
    )

    # Doctor command — regression harness
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run regression harness (smoke / check / full / benchmark)",
    )
    doctor_parser.add_argument(
        "tier",
        nargs="?",
        default="smoke",
        choices=["smoke", "check", "full", "benchmark"],
        help="Which tier to run (default: smoke)",
    )
    doctor_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model alias for check tier (default: qwen3.5-35b)",
    )
    doctor_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model aliases for full / benchmark tiers "
        "(full default: qwen3.5-35b,qwen3.6-35b; "
        "benchmark default: auto-discovered from local cache)",
    )
    doctor_parser.add_argument(
        "--update-baselines",
        action="store_true",
        help="Record current run as the new baseline (check / full only). "
        "Ignored with a warning for smoke / benchmark tiers.",
    )

    # Telemetry subcommand — opt-in anonymous usage data (Issue #236).
    # See vllm_mlx/telemetry/ for what we collect / don't collect, and
    # the README "Telemetry" section for the user-facing summary.
    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Manage anonymous usage telemetry (opt-in)",
    )
    telemetry_subparsers = telemetry_parser.add_subparsers(
        dest="telemetry_action",
        help="Telemetry actions",
    )
    telemetry_subparsers.add_parser(
        "status", help="Show whether telemetry is enabled and why"
    )
    telemetry_subparsers.add_parser(
        "enable", help="Opt in to anonymous usage telemetry"
    )
    telemetry_subparsers.add_parser(
        "disable", help="Opt out of anonymous usage telemetry"
    )
    telemetry_subparsers.add_parser(
        "preview",
        help="Print a sample payload showing exactly what telemetry would send",
    )
    telemetry_subparsers.add_parser(
        "reset",
        help="Delete the consent + client-id files (next run re-prompts)",
    )

    args = parser.parse_args()

    # First-run consent prompt — fires at most once per machine, only on
    # interactive subcommands when stdin is a tty. Safe no-op otherwise.
    # Must run *before* heavy subcommand work so the user sees the
    # disclosure before any model load logs scroll past.
    if getattr(args, "command", None) is not None:
        from vllm_mlx.telemetry import maybe_prompt_for_consent

        maybe_prompt_for_consent(
            args.command,
            cli_no_telemetry=getattr(args, "no_telemetry", False),
        )

    # Resolve model aliases before dispatch.
    #
    # The doctor subcommand is exempt: it intentionally keeps the alias
    # form so per-model artefacts (baseline filenames, scorecard rows,
    # report check names) stay human-readable and stable across runs.
    # Doctor does its own alias→path resolution inside the server-spawn
    # path via discovery, so resolving here would write the wrong
    # baseline filename and confuse multi-model loops.
    if (
        hasattr(args, "model")
        and args.model
        and getattr(args, "command", None) != "doctor"
    ):
        from vllm_mlx.model_aliases import resolve_model

        resolved = resolve_model(args.model)
        if resolved != args.model:
            print(f"  Alias: {args.model} → {resolved}")
            args._original_alias = args.model
            args.model = resolved
        elif "/" not in args.model and not os.path.exists(args.model):
            # Not an alias, not a HuggingFace org/name path, not a local
            # directory — fail fast with suggestions instead of letting the
            # request hit HuggingFace and 404 with a 30-line stack trace.
            print(
                f"\n  Error: '{args.model}' is not a known alias or HuggingFace path."
            )
            _print_unknown_model_help(
                args.model, full_path_example="mlx-community/Qwen3.5-9B-4bit"
            )
            sys.exit(1)

    if args.command == "serve":
        serve_command(args)
    elif args.command == "bench":
        bench_command(args)
    elif args.command == "models":
        models_command(args)
    elif args.command == "version":
        print(f"rapid-mlx {_version}")
    elif args.command == "help":
        target = getattr(args, "subcommand", None)
        if not target:
            parser.print_help()
        elif target in subparsers.choices:
            subparsers.choices[target].print_help()
        else:
            print(f"Unknown subcommand: {target}")
            print("Run `rapid-mlx help` for the list of subcommands.")
            sys.exit(1)
    elif args.command == "pull":
        pull_command(args)
    elif args.command == "rm":
        rm_command(args)
    elif args.command == "ps":
        ps_command(args)
    elif args.command == "upgrade":
        upgrade_command(args)
    elif args.command == "chat":
        chat_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "agents":
        agents_command(args)
    elif args.command == "doctor":
        from vllm_mlx.doctor.cli import doctor_command

        # Parse --models comma-list now so the doctor module gets a clean list.
        if getattr(args, "models", None):
            args.models = [m.strip() for m in args.models.split(",") if m.strip()]
        doctor_command(args)
    elif args.command == "telemetry":
        telemetry_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
