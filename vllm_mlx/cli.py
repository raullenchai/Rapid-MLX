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

    # Import unified server
    from . import server
    from .scheduler import SchedulerConfig
    from .server import RateLimiter, app, load_model

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

    # Auto-detect parser config from model name when not explicitly set
    if not args.tool_call_parser or not args.reasoning_parser:
        try:
            from .model_auto_config import detect_model_config

            auto_config = detect_model_config(args.model)
            if auto_config:
                if not args.tool_call_parser and auto_config.tool_call_parser:
                    args.tool_call_parser = auto_config.tool_call_parser
                    args.enable_auto_tool_choice = True
                    logger.info(
                        f"Auto-configured --tool-call-parser {auto_config.tool_call_parser}"
                    )
                if (
                    not args.reasoning_parser
                    and not args.no_thinking
                    and auto_config.reasoning_parser
                ):
                    args.reasoning_parser = auto_config.reasoning_parser
                    logger.info(
                        f"Auto-configured --reasoning-parser {auto_config.reasoning_parser}"
                    )
        except Exception as e:
            logger.debug(f"Auto-detection failed (non-fatal): {e}")

    # Pass alias info to server (for /v1/models)
    server._model_alias = getattr(args, "_original_alias", None)

    # Configure server security settings
    server._api_key = args.api_key
    server._default_timeout = args.timeout
    # Configure CORS
    cors_origins = args.cors_origins if args.cors_origins else ["*"]
    server.configure_cors(cors_origins)
    if args.rate_limit > 0:
        server._rate_limiter = RateLimiter(
            requests_per_minute=args.rate_limit, enabled=True
        )

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
            "\n    For speculative decoding, use --enable-mtp (requires model with MTP head).\n"
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

    # Load model with unified server
    try:
        load_model(
            args.model,
            scheduler_config=scheduler_config,
            stream_interval=args.stream_interval,
            max_tokens=args.max_tokens,
            force_mllm=args.mllm,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prefill_step_size=args.prefill_step_size,
            cloud_model=args.cloud_model,
            cloud_threshold=args.cloud_threshold,
            cloud_api_base=args.cloud_api_base,
            cloud_api_key=args.cloud_api_key,
            served_model_name=args.served_model_name,
            mtp=args.enable_mtp,
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
            from vllm_mlx.model_aliases import suggest_similar

            shown = getattr(args, "_original_alias", args.model)
            print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
            suggestions = suggest_similar(shown)
            if suggestions:
                print(f"  Did you mean: {', '.join(suggestions)}?")
            print("  Run `rapid-mlx models` to see available aliases,")
            print(
                "  or use a full HuggingFace path like: mlx-community/Qwen3.5-9B-4bit"
            )
        else:
            print(f"\n  Error loading model: {e}")
        sys.exit(1)

    # Start server
    # Note: Metal shader warmup runs in the FastAPI lifespan hook (server.py)
    # so it works for all engine types.
    print()
    host_display = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"  Ready: http://{host_display}:{args.port}/v1")
    print(f"  Docs:  http://{host_display}:{args.port}/docs")
    print()
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
                from vllm_mlx.model_aliases import suggest_similar

                shown = getattr(args, "_original_alias", args.model)
                print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
                suggestions = suggest_similar(shown)
                if suggestions:
                    print(f"  Did you mean: {', '.join(suggestions)}?")
                print("  Run `rapid-mlx models` to see available aliases,")
                print(
                    "  or use a full HuggingFace path like: mlx-community/Qwen3.5-9B-4bit"
                )
            else:
                print(f"\n  Error loading model: {e}")
            sys.exit(1)

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
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
    # reasoning 12 (deepseek_r1 + 1 pad), spec 10 ("✗ hybrid"), tier 11.
    cols = (
        ("Alias", 22),
        ("Tools", 16),
        ("Reasoning", 12),
        ("Spec-Decode", 10),
        ("Suffix Tier", 11),
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
        row = f"  {alias:<22} {tools:<16} {reasoning:<12} {spec:<10} {tier:<11}"
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
            from vllm_mlx.model_aliases import suggest_similar

            shown = getattr(args, "_original_alias", repo_id)
            print(f"\n  Error: Model '{shown}' not found on HuggingFace.")
            suggestions = suggest_similar(shown)
            if suggestions:
                print(f"  Did you mean: {', '.join(suggestions)}?")
            print("  Run `rapid-mlx models` to see available aliases,")
            print(
                "  or use a full HuggingFace path like: mlx-community/Qwen3.5-9B-4bit"
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
                    model = tok
                    break
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
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
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
    # The earlier guard ("≤2 unique tokens in the last 30") fired on
    # legitimate content like ``[0, 0, 0, 0, 0]``, markdown table
    # separators ``| --- | --- |``, and any list of identical short
    # numbers. The current criterion is stricter: the SAME token must
    # repeat ≥``REPEAT_LIMIT`` times *consecutively*. Tracked in a
    # rolling counter so cost is O(1) per chunk regardless of response
    # length (the previous ``full.split()`` was O(n²)).
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
            cmd, _, rest = line.partition(" ")
            rest = rest.strip()
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
    from vllm_mlx.model_aliases import resolve_model
    from vllm_mlx.model_auto_config import (
        detect_model_config,
        format_profile_table,
    )

    name = args.model
    resolved = resolve_model(name)
    if resolved and resolved != name:
        print(f"  alias: {name} → {resolved}")
        name = resolved

    cfg = detect_model_config(name)
    print()
    print(format_profile_table(name, cfg))
    print()
    if cfg is None:
        print("  No pattern matched — runtime probe will run when the model loads.")
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
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for Python logging and uvicorn",
    )
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
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
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
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

    args = parser.parse_args()

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
        from vllm_mlx.model_aliases import resolve_model, suggest_similar

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
            suggestions = suggest_similar(args.model)
            if suggestions:
                print(f"  Did you mean: {', '.join(suggestions)}?")
            print("  Run `rapid-mlx models` to see all aliases,")
            print("  or pass a full path like: mlx-community/Qwen3.5-9B-4bit")
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
